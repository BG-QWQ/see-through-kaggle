"""Gradio request/queue owner; models live only in one-shot subprocesses."""

import atexit
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

import gradio as gr
from gradio.processing_utils import move_resource_to_block_cache
from PIL import Image

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
from inference_process import InferenceProcess
from inference_jobs import JobStore

DTYPE = os.environ.get("SEETHROUGH_DTYPE", "fp16")
if DTYPE not in {"bf16", "fp16", "fp32"}:
    raise ValueError("SEETHROUGH_DTYPE must be bf16, fp16 or fp32")
DEPTH_RESOLUTION = int(os.environ.get("SEETHROUGH_DEPTH_RESOLUTION", 768))
MAX_RESOLUTION = int(os.environ.get("SEETHROUGH_MAX_RESOLUTION", 1536))
if MAX_RESOLUTION < 512 or MAX_RESOLUTION % 64:
    raise ValueError("SEETHROUGH_MAX_RESOLUTION must be a multiple of 64 and at least 512")
REQUEST_TIMEOUT = float(os.environ.get("SEETHROUGH_REQUEST_TIMEOUT", 5400))
DRAIN_GRACE = float(os.environ.get("SEETHROUGH_DRAIN_GRACE", 60))
if DRAIN_GRACE < 0:
    raise ValueError("SEETHROUGH_DRAIN_GRACE cannot be negative")
_WORKER = InferenceProcess(
    _script_dir / "inference_worker.py", timeout=REQUEST_TIMEOUT,
    stop_timeout=float(os.environ.get("SEETHROUGH_WORKER_STOP_TIMEOUT", 5)),
)
atexit.register(_WORKER.close)
_JOBS = JobStore(os.environ.get("SEETHROUGH_JOB_DIR", str(Path(tempfile.gettempdir()) / "see-through-jobs")))

_PHASE_LABELS = {
    "loading_layerdiff": "Loading LayerDiff model",
    "body_prepare": "Preparing body layers",
    "body": "LayerDiff · Body sampling",
    "body_decode": "Decoding and saving body layers",
    "head_prepare": "Preparing head layers",
    "head": "LayerDiff · Head sampling",
    "head_decode": "Decoding and saving head layers",
    "loading_depth": "Loading Marigold depth model",
    "depth_prepare": "Preparing depth inputs",
    "depth": "Marigold · Depth sampling",
    "depth_decode": "Decoding and saving depth maps",
    "psd": "Assembling PSD",
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cache_file(job_dir, relative_path):
    source = (job_dir / relative_path).resolve()
    if not source.is_relative_to(job_dir) or not source.is_file():
        raise RuntimeError("Result file is missing or does not belong to this request.")
    return move_resource_to_block_cache(str(source), demo)


def create_request():
    return _JOBS.create()


def _saved_outputs(request_id, record):
    folder = _JOBS.job_dir(request_id)
    result = record["result"]
    return (
        _cache_file(folder, result["psd"]),
        [(_cache_file(folder, item["path"]), item["label"]) for item in result["gallery"]],
    )


def recover_request(request_id):
    if not request_id:
        return "Run creates a request ID before inference. Keep it to retrieve results after a disconnection.", None, []
    try:
        record = _JOBS.get(request_id)
    except ValueError as error:
        raise gr.Error(str(error)) from None
    if record is None:
        return "No saved request with this ID in this Kaggle session. Recovery does not start a new inference.", None, []
    state = record["state"]
    if state == "completed":
        output_path, gallery = _saved_outputs(request_id, record)
        return "Completed. Saved results retrieved without running inference again.", output_path, gallery
    if state == "running":
        if record["owner"] != _JOBS.instance_id:
            return "This request belongs to another server instance. Check Kaggle logs; its current state cannot be confirmed here. Do not resubmit automatically.", None, []
        event = record.get("progress", {})
        phase = event.get("phase", "Starting isolated worker")
        description = _PHASE_LABELS.get(phase, phase)
        if event.get("total") is not None:
            description += f" — {event['completed']}/{event['total']} steps"
        return f"Running: {description}. Check again later; do not click Run again.", None, []
    if state == "queued":
        return "Prepared / queued; inference has not started yet. Check Kaggle logs before submitting another request.", None, []
    return f"{state.capitalize()}: {record.get('error', '')}", None, []


def restore_request(request_id):
    return request_id, *recover_request(request_id)


def cancel_queued_request(request_id):
    if request_id:
        _JOBS.cancel_queued(request_id)


async def inference(request_id: str, image: Image.Image, resolution: int = 1024, seed: int = 42,
                    tblr_split: bool = False, steps: int = 30, progress=gr.Progress()):
    started = time.monotonic()
    try:
        _JOBS.start(request_id, {
            "resolution": resolution, "seed": seed, "steps": steps, "tblr_split": bool(tblr_split),
        })
    except (ValueError, KeyError) as error:
        raise gr.Error(str(error)) from None

    def update_progress(event):
        _JOBS.progress(request_id, event)
        description = _PHASE_LABELS.get(event["phase"], event["phase"])
        if event["total"] is None:
            progress(None, desc=description)
        else:
            progress((event["completed"], event["total"]), desc=description)

    try:
        if image is None:
            raise gr.Error("Please upload an image.")
        resolution = round(resolution / 64) * 64
        if not 64 <= resolution <= MAX_RESOLUTION:
            raise gr.Error(f"Resolution must be between 64 and {MAX_RESOLUTION}; use 64-pixel steps.")
        seed, steps = int(seed), int(steps)
        _log(f"Request {request_id}: Resolution: {resolution}, Seed: {seed}, Image: {image.size}")
        with tempfile.TemporaryDirectory(prefix="seethrough_") as temporary:
            job_dir = Path(temporary).resolve()
            image.save(job_dir / "input.png")
            (job_dir / "request.json").write_text(json.dumps({
                "dtype": DTYPE, "resolution": resolution, "seed": seed, "steps": steps,
                "depth_resolution": DEPTH_RESOLUTION, "tblr_split": bool(tblr_split),
            }), encoding="utf-8")
            progress(None, desc="Starting isolated inference worker")
            result = await _WORKER.run(job_dir, on_progress=update_progress)
            record = await _JOBS.persist(request_id, job_dir, result)
        output_path, gallery = await asyncio.to_thread(_saved_outputs, request_id, record)
        progress(1.0, desc="Complete")
        _log(f"Request {request_id}: Total inference time: {time.monotonic() - started:.1f}s")
        return output_path, gallery, "Completed. Results saved; keep the request ID to retrieve them again."
    except asyncio.CancelledError:
        _JOBS.fail(request_id, "cancelled", "Request cancelled; the inference worker has been reaped.")
        _log("Request cancelled; inference worker has been reaped.")
        raise
    except (TimeoutError, RuntimeError) as error:
        _JOBS.fail(request_id, "failed", str(error))
        raise gr.Error(str(error)) from None
    except Exception as error:
        _JOBS.fail(request_id, "failed", str(error))
        raise


with gr.Blocks(title="See-through: Layer Decomposition", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(
        "# See-through: Single-image Layer Decomposition for Anime Characters\n\n"
        '<a href="https://github.com/shitagaki-lab/see-through" target="_blank">GitHub</a> | '
        '<a href="https://arxiv.org/abs/2602.03749" target="_blank">Paper (arXiv:2602.03749)</a>\n\n'
        "Upload an anime character illustration to decompose it into "
        "fully-inpainted semantic layers with depth ordering, "
        "exported as a layered PSD file.\n\n"
        "**Kaggle T4:** fp16 compute with fp32 VAEs; 1024 resolution is a speed/quality compromise. "
        "Each request uses GPU0 only; two T4s do not pool their VRAM. "
        "Every request runs in a fresh worker; failures do not reuse its CUDA state. "
        "A single request can still exhaust GPU or host memory. "
        "Do not run batch workers at the same time. "
        "This is a public service without authentication. "
        "Uploaded images and generated outputs are temporarily stored on Kaggle. "
        "Do not upload sensitive content or content you are not authorized to use.\n\n"
        "**Disclaimer:** This demo uses a newer model checkpoint and "
        "may not fully reproduce identical results reported in the paper."
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload image (non-square images will be padded)")
            resolution = gr.Slider(
                minimum=512, maximum=MAX_RESOLUTION,
                value=int(os.environ.get("SEETHROUGH_RESOLUTION", 1024)), step=64,
                label="Resolution",
                info="64-pixel steps. Limit: SEETHROUGH_MAX_RESOLUTION. Larger images use more memory on one GPU.",
            )
            seed = gr.Slider(minimum=0, maximum=9999,
                             value=int(os.environ.get("SEETHROUGH_SEED", 42)), step=1, label="Seed")
            steps = gr.Slider(10, 50, value=int(os.environ.get("SEETHROUGH_STEPS", 30)),
                              step=1, label="Inference steps")
            tblr_split = gr.Checkbox(
                value=False,
                label="Split arms, eyes and ears into left/right layers",
                info="Gives the arms, eyes and ears separate left and right layers. Legs are not split, and a part stays whole if its left and right sides touch.",
            )
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary")
                cancel_btn = gr.Button("Cancel this page's request")
        with gr.Column(scale=2):
            psd_output = gr.File(label="Download layered PSD")
            gallery_output = gr.Gallery(label="Separated layers", columns=4, height="auto")
    request_id = gr.Textbox(
        label="Request ID / recovery ID",
        info="Kept in this URL's fragment. Copy the ID before switching to a backup URL.",
    )
    recover_btn = gr.Button("Check status / retrieve saved result")
    request_status = gr.Markdown(
        "If the connection is lost, do not click Run again. Reopen this URL or enter the request ID, "
        "then check status. Recovery never restarts inference. Cancel applies only to this page's submitted request."
    )

    run_event = run_btn.click(fn=create_request, outputs=[request_id], queue=False).success(
        fn=inference,
        inputs=[request_id, input_image, resolution, seed, tblr_split, steps],
        outputs=[psd_output, gallery_output, request_status],
        concurrency_limit=1,
    )
    cancel_btn.click(fn=cancel_queued_request, inputs=[request_id], cancels=[run_event], queue=False)
    recover_btn.click(
        fn=recover_request, inputs=[request_id],
        outputs=[request_status, psd_output, gallery_output], queue=False,
    )
    request_id.change(
        fn=None, inputs=[request_id], outputs=[], queue=False,
        js="""(id) => {
            const params = new URLSearchParams(window.location.hash.slice(1));
            if (id) params.set("request", id); else params.delete("request");
            const hash = params.toString();
            history.replaceState(null, "", location.pathname + location.search + (hash ? "#" + hash : ""));
            return [];
        }""",
    )
    demo.load(
        fn=restore_request, inputs=[request_id],
        outputs=[request_id, request_status, psd_output, gallery_output], queue=False,
        js="() => [new URLSearchParams(window.location.hash.slice(1)).get('request') || '']",
    )
    gr.Examples(
        examples=[["common/assets/test_image.png", 1024, 42, False]],
        inputs=[input_image, resolution, seed, tblr_split],
    )

def serve():
    stopped = threading.Event()
    forced = threading.Event()
    draining = False

    def handle_signal(signum, frame):
        nonlocal draining
        if signum == getattr(signal, "SIGUSR1", None):
            if not forced.is_set():
                draining = True
        else:
            draining = False
            if not forced.is_set():
                forced.set()
                # A handler must not re-enter locks held by interrupted cleanup.
                threading.Thread(target=_WORKER.close, name="stop-worker", daemon=True).start()
        stopped.set()

    try:
        demo.queue(max_size=16).launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", 7860)),
            share=os.environ.get("SEETHROUGH_SHARE", "1") == "1",
            prevent_thread_lock=True,
        )
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, handle_signal)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, handle_signal)
        stopped.wait()
        if draining:
            _log("Session draining: rejecting new work and waiting for the active request.")
            handoff_deadline = time.monotonic() + REQUEST_TIMEOUT + _WORKER.stop_timeout
            _WORKER.drain()
            # A reaped worker is not yet a delivered Gradio result: cache handoff
            # and output postprocessing must finish before closing the server.
            while (not forced.is_set() and demo._queue.get_active_worker_count()
                   and time.monotonic() < handoff_deadline):
                forced.wait(0.1)
            if not forced.is_set() and not demo._queue.get_active_worker_count():
                _log(f"Results ready; retaining download endpoints for {DRAIN_GRACE:g}s.")
                forced.wait(DRAIN_GRACE)
    finally:
        _WORKER.close()
        demo.close()


if __name__ == "__main__":
    serve()
