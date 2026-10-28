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


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cache_file(job_dir, relative_path):
    source = (job_dir / relative_path).resolve()
    if not source.is_relative_to(job_dir) or not source.is_file():
        raise RuntimeError("推理结果文件缺失或不属于本次请求。")
    return move_resource_to_block_cache(str(source), demo)


async def inference(image: Image.Image, resolution: int = 1024, seed: int = 42,
                    tblr_split: bool = False, steps: int = 30, progress=gr.Progress()):
    started = time.monotonic()
    if image is None:
        raise gr.Error("Please upload an image.")
    resolution = round(resolution / 64) * 64
    if not 64 <= resolution <= MAX_RESOLUTION:
        raise gr.Error(f"Resolution must be between 64 and {MAX_RESOLUTION}; use 64-pixel steps.")
    seed, steps = int(seed), int(steps)
    _log(f"Resolution: {resolution}, Seed: {seed}, Image: {image.size}")
    stage_progress = {"layerdiff": (0, "LayerDiff"), "depth": (1, "Marigold depth"), "psd": (2, "PSD assembly")}

    def update_stage(stage):
        if stage in stage_progress:
            index, description = stage_progress[stage]
            progress((index, 3), desc=description)

    try:
        # Files outlive the child; only the parent may remove the job directory.
        with tempfile.TemporaryDirectory(prefix="seethrough_") as temporary:
            job_dir = Path(temporary).resolve()
            image.save(job_dir / "input.png")
            (job_dir / "request.json").write_text(json.dumps({
                "dtype": DTYPE, "resolution": resolution, "seed": seed, "steps": steps,
                "depth_resolution": DEPTH_RESOLUTION, "tblr_split": bool(tblr_split),
            }), encoding="utf-8")
            progress((0, 3), desc="Starting isolated inference")
            result = await _WORKER.run(job_dir, on_stage=update_stage)
            output_path = _cache_file(job_dir, result["psd"])
            gallery = [(_cache_file(job_dir, item["path"]), item["label"]) for item in result["gallery"]]
            progress((3, 3), desc="Complete")
            _log(f"Total inference time: {time.monotonic() - started:.1f}s")
            return output_path, gallery
    except asyncio.CancelledError:
        _log("Request cancelled; inference worker has been reaped.")
        raise
    except (TimeoutError, RuntimeError) as error:
        raise gr.Error(str(error)) from None


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
        "Public links are temporary and accessible to anyone — do not upload private images.\n\n"
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
                cancel_btn = gr.Button("Cancel request")
        with gr.Column(scale=2):
            psd_output = gr.File(label="Download layered PSD")
            gallery_output = gr.Gallery(label="Separated layers", columns=4, height="auto")

    run_event = run_btn.click(
        fn=inference,
        inputs=[input_image, resolution, seed, tblr_split, steps],
        outputs=[psd_output, gallery_output],
        concurrency_limit=1,
    )
    cancel_btn.click(fn=None, cancels=[run_event], queue=False)
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
