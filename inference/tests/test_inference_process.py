"""CPU-only OS-process lifecycle regressions, not model-output substitutes."""

import asyncio
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from inference_process import InferenceProcess


@unittest.skipUnless(os.name == "posix", "The deployed worker uses POSIX process groups")
class InferenceProcessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker = self.root / "inference" / "scripts" / "lifecycle_probe.py"
        self.worker.parent.mkdir(parents=True)
        # A real process with observable PID and signal behavior. It deliberately
        # does not implement an inference success payload or fake image output.
        self.worker.write_text(
            "import argparse, os, signal, sys, time\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "from inference_worker import _watch_parent, _report_progress\n"
            "parser=argparse.ArgumentParser();parser.add_argument('job_dir');parser.add_argument('--parent-pid',type=int);args=parser.parse_args()\n"
            "_watch_parent(args.parent_pid)\n"
            "job=Path(args.job_dir)\n"
            "if (job/'ignore-term').exists(): signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "(job/'started').write_text(str(os.getpid()))\n"
            "if (job/'progress').exists():\n"
            "    sys.stderr.write('\\rnative progress without newline');sys.stderr.flush()\n"
            "    _report_progress('body',1,4)\n"
            "if (job/'crash').exists(): os.kill(os.getpid(),signal.SIGKILL)\n"
            "while not (job/'release').exists(): time.sleep(0.01)\n"
        )
        self.runner = InferenceProcess(self.worker, timeout=10, stop_timeout=0.1)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.runner.close)

    def job(self, name):
        folder = self.root / name
        folder.mkdir()
        return folder

    async def started(self, folder):
        async def wait_for_marker():
            while not (folder / "started").exists():
                await asyncio.sleep(0.01)
            return int((folder / "started").read_text())
        return await asyncio.wait_for(wait_for_marker(), timeout=5)

    def assert_reaped(self, pid):
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertIsNone(self.runner.pid)

    async def run_and_cancel(self, folder):
        task = asyncio.create_task(self.runner.run(folder))
        try:
            pid = await self.started(folder)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assert_reaped(pid)
            return pid
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def test_cancel_reaps_worker_and_allows_new_request(self):
        first = self.job("first")
        (first / "ignore-term").touch()
        first_pid = await self.run_and_cancel(first)
        second_pid = await self.run_and_cancel(self.job("second"))
        self.assertNotEqual(first_pid, second_pid)

    async def test_cancelling_queued_request_leaves_active_worker_untouched(self):
        active_folder = self.job("active")
        queued_folder = self.job("queued")
        active = asyncio.create_task(self.runner.run(active_folder))
        try:
            pid = await self.started(active_folder)
            queued = asyncio.create_task(self.runner.run(queued_folder))
            await asyncio.sleep(0)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            self.assertFalse(active.done())
            self.assertEqual(self.runner.pid, pid)
            os.kill(pid, 0)
            self.assertFalse((queued_folder / "started").exists())
        finally:
            active.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await active
        self.assert_reaped(pid)

    async def test_parent_death_stops_worker_without_cooperative_shutdown(self):
        folder = self.job("parent-death")
        code = (
            "import asyncio, sys\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "from inference_process import InferenceProcess\n"
            f"asyncio.run(InferenceProcess({str(self.worker)!r}).run({str(folder)!r}))\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = None
        try:
            pid = await self.started(folder)
            parent.kill()
            await asyncio.to_thread(parent.wait, timeout=5)

            async def wait_for_exit():
                while True:
                    observed = await asyncio.to_thread(
                        subprocess.run, ["ps", "-p", str(pid), "-o", "stat="],
                        capture_output=True, text=True,
                    )
                    self.assertIn(observed.returncode, (0, 1), observed.stderr)
                    # An orphan zombie has exited and no longer owns CUDA;
                    # its adoption/reaping belongs to the host init process.
                    if observed.returncode == 1 or observed.stdout.strip().startswith("Z"):
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_exit(), timeout=5)
        finally:
            if parent.poll() is None:
                parent.kill()
            await asyncio.to_thread(parent.wait, timeout=5)
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    async def test_sigkill_is_reported_and_does_not_poison_next_request(self):
        failed = self.job("failed")
        (failed / "crash").touch()
        with self.assertRaisesRegex(RuntimeError, "SIGKILL"):
            await self.runner.run(failed)
        failed_pid = int((failed / "started").read_text())
        self.assert_reaped(failed_pid)
        next_pid = await self.run_and_cancel(self.job("after-crash"))
        self.assertNotEqual(failed_pid, next_pid)

    async def test_timeout_kills_uncooperative_worker_before_unlocking(self):
        folder = self.job("timeout")
        (folder / "ignore-term").touch()
        self.runner.timeout = 5
        task = asyncio.create_task(self.runner.run(folder))
        pid = await self.started(folder)
        with self.assertRaises(TimeoutError):
            await task
        self.assert_reaped(pid)
        await self.run_and_cancel(self.job("after-timeout"))

    async def test_drain_finishes_active_request_and_rejects_new_workers(self):
        folder = self.job("drain")
        task = asyncio.create_task(self.runner.run(folder))
        pid = await self.started(folder)
        drain = asyncio.create_task(asyncio.to_thread(self.runner.drain))
        try:
            # Closing admission is checked through the subsequent real call,
            # not by asserting private lock state or incidental defaults.
            await asyncio.sleep(0)
            self.assertFalse(drain.done())
            (folder / "release").touch()
            with self.assertRaisesRegex(RuntimeError, "未提交完整结果"):
                await task
            await drain
            self.assert_reaped(pid)
            rejected = self.job("after-drain")
            with self.assertRaisesRegex(RuntimeError, "关闭"):
                await self.runner.run(rejected)
            self.assertFalse((rejected / "started").exists())
        finally:
            self.runner.close()
            await drain

    async def test_progress_arrives_before_exit_with_partial_native_stderr(self):
        folder = self.job("progress")
        (folder / "progress").touch()
        intermediate = asyncio.Event()

        def observe(event):
            if event["phase"] == "body" and 0 < event["completed"] < event["total"]:
                intermediate.set()

        task = asyncio.create_task(self.runner.run(folder, on_progress=observe))
        pid = None
        try:
            pid = await self.started(folder)
            await asyncio.wait_for(intermediate.wait(), timeout=5)
            self.assertFalse(task.done(), "Progress must arrive while inference is still running")
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assert_reaped(pid)


if __name__ == "__main__":
    unittest.main()
