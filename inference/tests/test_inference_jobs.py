"""CPU-only receipt and artifact persistence using real temporary files."""

import asyncio
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from inference_jobs import JobStore


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = JobStore(self.root / "receipts")

    def running(self):
        token = self.store.create()
        self.store.start(token, {"seed": 123, "steps": 30})
        return token

    def outputs(self, directory):
        directory = Path(directory)
        (directory / "gallery").mkdir(parents=True)
        (directory / "output.psd").write_bytes(b"PSD artifact bytes")
        (directory / "gallery" / "layer.png").write_bytes(b"gallery artifact bytes")
        (directory / "input.png").write_bytes(b"private input bytes")
        (directory / "intermediate.bin").write_bytes(b"temporary intermediate")
        return {
            "status": "completed", "psd": "output.psd",
            "gallery": [{"path": "gallery/layer.png", "label": "Layer"}],
        }

    def test_duplicate_start_cannot_replace_running_request(self):
        token = self.running()
        self.store.progress(token, {"phase": "body", "completed": 2, "total": 30})
        first = self.store.get(token)
        with self.assertRaises(ValueError):
            self.store.start(token, {"seed": 456})
        self.assertEqual(self.store.get(token), first)

    def test_completed_files_survive_source_cleanup_and_new_store(self):
        token = self.running()
        with tempfile.TemporaryDirectory(dir=self.root) as worker:
            result = self.outputs(worker)
            completed = self.store.complete(token, worker, result)
            self.assertEqual(completed["state"], "completed")
            self.assertFalse((Path(worker) / "output.psd").exists())
            self.assertFalse((Path(worker) / "gallery/layer.png").exists())
            self.assertTrue((Path(worker) / "input.png").is_file())
            self.assertTrue((Path(worker) / "intermediate.bin").is_file())
        reopened = JobStore(self.store.root)
        self.assertNotEqual(reopened.instance_id, self.store.instance_id)
        record = reopened.get(token)
        self.assertEqual(record, completed)
        self.assertEqual(record["owner"], self.store.instance_id)
        self.assertEqual(record["result"], {
            "psd": "artifacts/output.psd",
            "gallery": [{"path": "artifacts/gallery/layer.png", "label": "Layer"}],
        })
        directory = reopened.job_dir(token)
        self.assertEqual((directory / record["result"]["psd"]).read_bytes(), b"PSD artifact bytes")
        self.assertEqual((directory / record["result"]["gallery"][0]["path"]).read_bytes(), b"gallery artifact bytes")
        self.assertFalse((directory / "artifacts/input.png").exists())
        self.assertFalse((directory / "artifacts/intermediate.bin").exists())
        with self.assertRaises(ValueError):
            reopened.start(token, {})
        self.assertEqual(reopened.get(token), completed)

    def test_delivery_failure_cannot_demote_completed_request(self):
        token = self.running()
        worker = self.root / "worker"
        result = self.outputs(worker)
        completed = self.store.complete(token, worker, result)
        self.store.fail(token, "failed", "browser delivery disconnected")
        self.store.fail(token, "cancelled", "listener disconnected")
        self.store.progress(token, {"phase": "late event"})
        self.assertEqual(self.store.get(token), completed)
        with self.assertRaises(ValueError):
            self.store.complete(token, worker, result)
        self.assertEqual(self.store.get(token), completed)

    def test_failure_and_cancellation_are_terminal_without_results(self):
        for state in ("failed", "cancelled"):
            with self.subTest(state=state):
                token = self.running()
                self.store.fail(token, state, RuntimeError("original worker error"))
                first = self.store.get(token)
                self.assertEqual(first["state"], state)
                self.assertEqual(first["error"], "original worker error")
                self.assertNotIn("result", first)
                self.assertIn("finished_at", first)
                self.store.fail(token, "failed", "later cleanup error")
                self.store.progress(token, {"phase": "late"})
                with self.assertRaises(ValueError):
                    self.store.start(token, {})
                with self.assertRaises(ValueError):
                    self.store.complete(token, self.root, {})
                self.assertEqual(self.store.get(token), first)

    def test_queued_request_can_be_cancelled_without_starting(self):
        token = self.store.create()
        self.assertTrue(self.store.cancel_queued(token))
        record = self.store.get(token)
        self.assertEqual(record["state"], "cancelled")
        self.assertNotIn("owner", record)
        self.assertNotIn("result", record)
        self.assertFalse(self.store.cancel_queued(token))
        self.assertEqual(self.store.get(token), record)
        with self.assertRaises(ValueError):
            self.store.start(token, {})

    def test_queued_cancellation_leaves_running_and_finished_requests_unchanged(self):
        running = self.running()
        completed = self.running()
        worker = self.root / "worker"
        self.store.complete(completed, worker, self.outputs(worker))
        failed = self.running()
        self.store.fail(failed, "failed", "worker error")
        for token in (running, completed, failed):
            with self.subTest(state=self.store.get(token)["state"]):
                record = self.store.get(token)
                self.assertFalse(self.store.cancel_queued(token))
                self.assertEqual(self.store.get(token), record)

    def test_unknown_and_invalid_request_ids(self):
        self.assertIsNone(self.store.get("0" * 32))
        with self.assertRaises(KeyError):
            self.store.start("0" * 32, {})
        for token in ("../outside", "A" * 32, "0" * 31, "0" * 33,
                      "0" * 32 + "\n", "", None, 123):
            with self.subTest(token=token):
                for operation in (self.store.get, self.store.job_dir):
                    with self.assertRaises(ValueError):
                        operation(token)

    def test_escaping_artifacts_are_rejected_before_any_move(self):
        outside = self.root / "outside.psd"
        outside.write_bytes(b"not a worker output")
        for relative in ("../outside.psd", str(outside), "gallery/../../outside.psd"):
            with self.subTest(relative=relative):
                token = self.running()
                worker = self.root / token
                result = self.outputs(worker)
                result["gallery"][0]["path"] = relative
                with self.assertRaises(ValueError):
                    self.store.complete(token, worker, result)
                self.assertEqual(self.store.get(token)["state"], "running")
                self.assertNotIn("result", self.store.get(token))
                self.assertTrue((worker / "output.psd").is_file())
                self.assertFalse((self.store.job_dir(token) / "artifacts").exists())
        self.assertEqual(outside.read_bytes(), b"not a worker output")

    def test_symlinks_cannot_escape_storage_or_worker_directory(self):
        outside = self.root / "outside"
        outside.mkdir()
        alias_token = "1" * 32
        (self.store.root / alias_token).symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.store.get(alias_token)
        token = self.running()
        worker = self.root / "worker"
        result = self.outputs(worker)
        (outside / "external.png").write_bytes(b"external artifact")
        (worker / "gallery/link.png").symlink_to(outside / "external.png")
        result["gallery"][0]["path"] = "gallery/link.png"
        with self.assertRaises(ValueError):
            self.store.complete(token, worker, result)
        self.assertTrue((worker / "output.psd").is_file())
        self.assertEqual(self.store.get(token)["state"], "running")
        (self.store.job_dir(token) / "artifacts").symlink_to(outside, target_is_directory=True)
        result["gallery"][0]["path"] = "gallery/layer.png"
        with self.assertRaises(ValueError):
            self.store.complete(token, worker, result)
        self.assertEqual((outside / "external.png").read_bytes(), b"external artifact")

    def test_missing_artifact_keeps_existing_outputs_and_receipt_uncompleted(self):
        token = self.running()
        worker = self.root / "worker"
        result = self.outputs(worker)
        result["gallery"][0]["path"] = "gallery/missing.png"
        with self.assertRaises(FileNotFoundError) as raised:
            self.store.complete(token, worker, result)
        self.assertTrue((worker / "output.psd").is_file())
        self.assertEqual(self.store.get(token)["state"], "running")
        self.store.fail(token, "failed", raised.exception)
        record = self.store.get(token)
        self.assertEqual(record["state"], "failed")
        self.assertNotIn("result", record)

    def test_running_receipt_survives_reopen_without_resuming_computation(self):
        token = self.running()
        first = self.store.get(token)
        reopened = JobStore(self.store.root)
        self.assertEqual(reopened.get(token), first)
        self.assertNotEqual(reopened.instance_id, first["owner"])
        with self.assertRaises(ValueError):
            reopened.start(token, {})
        self.assertEqual(reopened.get(token), first)

    def test_invalid_failure_state_does_not_change_receipt(self):
        token = self.running()
        first = self.store.get(token)
        with self.assertRaises(ValueError):
            self.store.fail(token, "completed", "not success")
        self.assertEqual(self.store.get(token), first)


class _PausedJobStore(JobStore):
    def __init__(self, root):
        super().__init__(root)
        self.started = Event()
        self.release = Event()

    def complete(self, token, job_dir, result):
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("Test did not release the disk handoff")
        return super().complete(token, job_dir, result)


class PersistenceCancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = _PausedJobStore(self.root / "receipts")
        self.token = self.store.create()
        self.store.start(self.token, {"seed": 123})
        self.worker_temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.worker_temporary.cleanup)
        self.worker = Path(self.worker_temporary.name)
        (self.worker / "output.psd").write_bytes(b"persisted PSD bytes")
        (self.worker / "layer.png").write_bytes(b"persisted layer bytes")
        (self.worker / "input.png").write_bytes(b"synthetic private input")
        self.result = {
            "psd": "output.psd",
            "gallery": [{"path": "layer.png", "label": "Layer"}],
        }
        self.tasks = []

    async def asyncTearDown(self):
        self.store.release.set()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def begin_handoff(self):
        task = asyncio.create_task(self.store.persist(self.token, self.worker, self.result))
        self.tasks.append(task)
        self.assertTrue(await asyncio.to_thread(self.store.started.wait, 5))
        return task

    def assert_saved_after_worker_cleanup(self):
        self.worker_temporary.cleanup()
        reopened = JobStore(self.store.root)
        record = reopened.get(self.token)
        self.assertEqual(record["state"], "completed")
        directory = reopened.job_dir(self.token)
        self.assertEqual((directory / record["result"]["psd"]).read_bytes(), b"persisted PSD bytes")
        self.assertEqual((directory / record["result"]["gallery"][0]["path"]).read_bytes(), b"persisted layer bytes")
        self.assertFalse((directory / "artifacts/input.png").exists())
        return record

    async def test_uninterrupted_handoff_returns_durable_receipt(self):
        task = await self.begin_handoff()
        self.store.release.set()
        record = await task
        self.assertEqual(record, self.assert_saved_after_worker_cleanup())

    async def test_repeated_cancellation_waits_for_durable_outputs_then_propagates(self):
        task = await self.begin_handoff()
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertTrue((self.worker / "output.psd").is_file())
            self.assertEqual(self.store.get(self.token)["state"], "running")
        self.store.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        completed = self.assert_saved_after_worker_cleanup()
        self.store.fail(self.token, "cancelled", "Late cancellation cleanup")
        self.assertEqual(self.store.get(self.token), completed)

    async def test_disk_failure_after_cancellation_is_not_reported_as_completion(self):
        self.result["gallery"][0]["path"] = "missing.png"
        task = await self.begin_handoff()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.store.release.set()
        with self.assertRaises(FileNotFoundError) as raised:
            await task
        self.assertTrue((self.worker / "output.psd").is_file())
        self.assertEqual(self.store.get(self.token)["state"], "running")
        self.store.fail(self.token, "failed", raised.exception)
        record = self.store.get(self.token)
        self.assertEqual(record["state"], "failed")
        self.assertNotIn("result", record)


if __name__ == "__main__":
    unittest.main()
