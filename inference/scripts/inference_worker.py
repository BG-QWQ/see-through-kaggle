"""Run one file-backed inference request in a fresh, parent-owned process."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import traceback
import types


def _terminate_orphan():
    # Never signal the caller's process group when invoked without a new session.
    if os.name == "posix" and os.getpgrp() == os.getpid():
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            pass
    os._exit(1)


def _watch_parent(parent_pid):
    """Install the stdin lifeline before importing torch or any model code."""
    if os.getppid() != parent_pid:
        _terminate_orphan()
    stdin_fd = sys.stdin.fileno()

    def watch_stdin():
        try:
            while os.read(stdin_fd, 1):
                pass
        except OSError:
            pass
        _terminate_orphan()

    threading.Thread(target=watch_stdin, name="parent-lifeline", daemon=True).start()
    if os.getppid() != parent_pid:
        _terminate_orphan()


def _write_result(job_dir, result):
    temporary = job_dir / "result.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, job_dir / "result.json")


def _snapshot_gallery(saved_dir, job_dir):
    # Byte copies detach the gallery from further_extr's in-place PNG edits.
    gallery_dir = job_dir / "gallery"
    gallery_dir.mkdir()
    gallery = []
    for image_path in sorted(saved_dir.iterdir()):
        if image_path.suffix != ".png":
            continue
        tag = image_path.stem
        if tag in {"src_img", "src_head", "reconstruction"} or tag.endswith("_depth"):
            continue
        destination = gallery_dir / image_path.name
        shutil.copyfile(image_path, destination)
        gallery.append({"path": destination.relative_to(job_dir).as_posix(), "label": tag})
    return gallery


def _run_request(job_dir):
    with (job_dir / "request.json").open(encoding="utf-8") as stream:
        request = json.load(stream)
    input_path = job_dir / "input.png"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing request image: {input_path}")

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent.parent
    sys.path.append(str(script_dir.parent))
    sys.path.insert(0, str(root / "common"))
    sys.path.insert(0, str(script_dir))

    import torch
    from inference_psd_quantized import (
        build_layerdiff_pipeline,
        build_marigold_pipeline,
        run_layerdiff,
        run_marigold,
    )
    from utils.inference_utils import further_extr
    from utils.torch_utils import seed_everything

    dtype = request["dtype"]
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    args = types.SimpleNamespace(
        quant_mode="none",
        repo_id_layerdiff="layerdifforg/seethroughv0.0.2_layerdiff3d",
        repo_id_depth="24yearsold/seethroughv0.0.1_marigold",
        cpu_offload=False,
        group_offload=True,
        compute_dtype=dtype_map[dtype],
        vae_dtype=torch.bfloat16 if dtype == "bf16" else torch.float32,
    )
    seed = int(request["seed"])
    steps = int(request["steps"])
    resolution = int(request["resolution"])

    print("SEETHROUGH_STAGE layerdiff", flush=True)
    seed_everything(seed)
    layerdiff_pipe = build_layerdiff_pipeline(args)
    run_layerdiff(layerdiff_pipe, str(input_path), str(job_dir), seed, steps, resolution)
    del layerdiff_pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("SEETHROUGH_STAGE depth", flush=True)
    marigold_pipe = build_marigold_pipeline(args)
    run_marigold(
        marigold_pipe, str(input_path), str(job_dir), seed,
        resolution_depth=request["depth_resolution"],
    )
    del marigold_pipe
    gc.collect()
    torch.cuda.empty_cache()

    saved_dir = job_dir / "input"
    gallery = _snapshot_gallery(saved_dir, job_dir)
    print("SEETHROUGH_STAGE psd", flush=True)
    further_extr(
        str(saved_dir), rotate=False, save_to_psd=True,
        tblr_split=request["tblr_split"],
    )
    psd_path = job_dir / "input.psd"
    if not psd_path.is_file() or psd_path.stat().st_size == 0:
        raise RuntimeError("PSD generation failed — no output file produced.")
    return {"status": "completed", "psd": "input.psd", "gallery": gallery}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--parent-pid", type=int, required=True)
    cli = parser.parse_args()
    _watch_parent(cli.parent_pid)
    job_dir = cli.job_dir.resolve()
    try:
        result = _run_request(job_dir)
        _write_result(job_dir, result)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        torch = sys.modules.get("torch")
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        is_cuda_oom = oom_type is not None and isinstance(error, oom_type)
        status = "cuda_oom" if is_cuda_oom else "error"
        message = str(error) or type(error).__name__
        try:
            _write_result(job_dir, {"status": status, "error": message})
        except Exception:
            traceback.print_exc(file=sys.stderr)
        return 2 if is_cuda_oom else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
