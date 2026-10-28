"""Own one short-lived inference process without importing model libraries."""

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


class InferenceProcess:
    def __init__(self, worker_path, timeout=5400, stop_timeout=5):
        if timeout <= 0 or stop_timeout <= 0:
            raise ValueError("Worker timeouts must be positive")
        self.worker_path = Path(worker_path)
        self.timeout = float(timeout)
        self.stop_timeout = float(stop_timeout)
        self._job_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._process = None
        self._closing = False

    @property
    def pid(self):
        with self._state_lock:
            return self._process.pid if self._process is not None else None

    def _stop_process(self, process):
        with self._termination_lock:
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=self.stop_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait()
            else:
                process.wait()

    def close(self):
        """Stop admission and terminate the active process, if any."""
        with self._state_lock:
            self._closing = True
            process = self._process
        if process is not None:
            self._stop_process(process)

    def drain(self):
        """Stop admission; the active request retains its existing deadline."""
        with self._state_lock:
            self._closing = True
        if not self._idle.wait(self.timeout + self.stop_timeout):
            self.close()

    def _finish(self, process, reader):
        self._stop_process(process)
        reader.join()
        process.stdout.close()
        process.stdin.close()

    async def _finish_uninterrupted(self, process, reader):
        cleanup = asyncio.create_task(asyncio.to_thread(self._finish, process, reader))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError

    async def run(self, job_dir, on_progress=None):
        await self._job_lock.acquire()
        process = reader = None
        try:
            loop = asyncio.get_running_loop()
            lines = asyncio.Queue()
            environment = dict(os.environ)
            environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
            # Popen does not yield between spawning and registering ownership.
            # stdin is the worker's parent-death lifeline, not a data channel.
            with self._state_lock:
                if self._closing:
                    raise RuntimeError("服务正在关闭，不再接受新的推理任务。")
                process = subprocess.Popen(
                    [sys.executable, "-u", str(self.worker_path), str(job_dir),
                     "--parent-pid", str(os.getpid())],
                    cwd=self.worker_path.parent.parent.parent, env=environment,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    bufsize=1, start_new_session=os.name == "posix",
                )
                self._process = process
                self._idle.clear()
            print(f"[worker {process.pid}] Started isolated inference", flush=True)

            def read_output():
                try:
                    for line in process.stdout:
                        loop.call_soon_threadsafe(lines.put_nowait, line)
                finally:
                    loop.call_soon_threadsafe(lines.put_nowait, None)

            reader = threading.Thread(target=read_output, name=f"worker-log-{process.pid}", daemon=True)
            reader.start()
            deadline = time.monotonic() + self.timeout
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    line = await asyncio.wait_for(lines.get(), timeout=remaining)
                    if line is None:
                        break
                    print(f"[worker {process.pid}] {line}", end="", flush=True)
                    if on_progress is not None and line.startswith("SEETHROUGH_PROGRESS "):
                        on_progress(json.loads(line.removeprefix("SEETHROUGH_PROGRESS ")))
                code = await asyncio.to_thread(process.wait, timeout=max(0, deadline - time.monotonic()))
            except (TimeoutError, subprocess.TimeoutExpired):
                raise TimeoutError(f"推理超过 {self.timeout:g} 秒时限；本次 worker 将被退出回收。") from None

            result_path = Path(job_dir) / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
            if code != 0:
                if result.get("status") == "cuda_oom":
                    raise RuntimeError("显存不足；本次推理进程已退出。请降低分辨率后重新提交，下一请求会使用新进程。")
                if code < 0:
                    name = signal.Signals(-code).name
                    raise RuntimeError(f"推理进程被 {name} 终止；若为 SIGKILL，请检查宿主 RAM 与系统日志。")
                detail = result.get("error", "未生成错误详情，请查看 worker 日志")
                raise RuntimeError(f"推理进程失败（退出码 {code}）：{detail}")
            if result.get("status") != "completed":
                raise RuntimeError("推理进程未提交完整结果，请查看 worker 日志。")
            return result
        finally:
            try:
                if process is not None and reader is not None:
                    await self._finish_uninterrupted(process, reader)
                elif process is not None:
                    await asyncio.to_thread(self._stop_process, process)
                    process.stdout.close()
                    process.stdin.close()
            finally:
                with self._state_lock:
                    self._process = None
                    self._idle.set()
                self._job_lock.release()
