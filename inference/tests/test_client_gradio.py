"""Consumer regressions for non-replaying submissions and atomic PSD retrieval."""

import hashlib
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import ssl
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from client_gradio import (ArtifactError, RequestStateError, SeeThroughClient,
                           SeeThroughError, SubmissionUncertain, WaitTimeout, main)


class InterruptedBody(httpx.SyncByteStream):
    def __iter__(self):
        yield b"partial"
        raise httpx.ReadError("peer closed stream")


class ClientGradioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.payload = b"verified PSD content"
        self.request_id = "a" * 32
        self.metadata = {
            "path": "/cache/private output.psd", "size": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.sleep = patch("client_gradio.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def client(self, handler, **kwargs):
        client = SeeThroughClient("https://primary.test", **kwargs)
        client._http.close()
        client._http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        self.addCleanup(client.close)
        return client

    def completed(self, metadata=None):
        return httpx.Response(200, json={"data": [{
            "request_id": self.request_id, "state": "completed", "psd": metadata or self.metadata,
        }]})

    def final(self):
        path = self.root / self.request_id / "input.psd"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"existing valid output")
        return path

    def assert_no_partials(self):
        self.assertEqual(list((self.root / self.request_id).glob("*.part")), [])
        self.assertEqual(list((self.root / self.request_id).glob(".request-*")), [])

    def test_ambiguous_submission_persists_receipt_and_recovery_never_resubmits(self):
        operations = []
        image = self.root / "image.png"
        image.write_bytes(b"input bytes")

        def handler(request):
            operations.append(request.url.path)
            if request.url.path.endswith("/upload"):
                return httpx.Response(200, json=["/uploaded/image.png"])
            if request.url.path.endswith("/create_request"):
                return httpx.Response(200, json={"data": [self.request_id]})
            if request.url.path.endswith("/call/inference"):
                receipt = self.root / self.request_id / "request.json"
                self.assertEqual(json.loads(receipt.read_text())["request_id"], self.request_id)
                self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
                raise httpx.ReadTimeout("submission acknowledgement lost")
            if request.url.path.endswith("/request_snapshot"):
                return self.completed()
            if request.method == "GET":
                return httpx.Response(200, content=self.payload)
            self.fail("Unexpected operation")

        client = self.client(handler)
        with self.assertRaises(SubmissionUncertain) as caught:
            client.start(image, self.root)
        self.assertEqual(caught.exception.request_id, self.request_id)
        self.assertEqual(client.wait(self.request_id)["state"], "completed")
        self.assertEqual(client.download(self.request_id, self.root).read_bytes(), self.payload)
        self.assertEqual(operations.count("/gradio_api/upload"), 1)
        self.assertEqual(operations.count("/gradio_api/run/create_request"), 1)
        self.assertEqual(operations.count("/gradio_api/call/inference"), 1)

    def test_corrupt_or_truncated_download_cannot_replace_existing_file(self):
        for mode in ("wrong-hash", "truncated", "interrupted"):
            with self.subTest(mode=mode):
                target = self.final()
                downloads = []

                def handler(request):
                    if request.method == "POST":
                        return self.completed()
                    downloads.append(request.url.host)
                    if mode == "interrupted":
                        return httpx.Response(200, stream=InterruptedBody())
                    body = b"x" * len(self.payload) if mode == "wrong-hash" else b"short"
                    return httpx.Response(200, content=body)

                client = self.client(handler, attempts=2)
                with self.assertRaises((ArtifactError, httpx.ReadError)):
                    client.download(self.request_id, self.root)
                self.assertEqual(len(downloads), 2)
                self.assertEqual(target.read_bytes(), b"existing valid output")
                self.assert_no_partials()

    def test_transient_download_rotates_to_backup_and_atomically_succeeds(self):
        target = self.final()
        hosts = []

        def handler(request):
            if request.method == "POST":
                return self.completed()
            hosts.append(request.url.host)
            self.assertEqual(target.read_bytes(), b"existing valid output")
            if len(hosts) == 1:
                return httpx.Response(503)
            return httpx.Response(200, content=self.payload)

        client = self.client(handler, backup_url="https://backup.test", attempts=3)
        self.assertEqual(client.download(self.request_id, self.root), target)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(hosts, ["primary.test", "backup.test"])
        self.assert_no_partials()

    def test_transient_download_exhaustion_is_bounded_and_preserves_output(self):
        target = self.final()
        downloads = []

        def handler(request):
            if request.method == "POST":
                return self.completed()
            downloads.append(request.url.host)
            return httpx.Response(502)

        client = self.client(handler, backup_url="https://backup.test", attempts=3)
        with self.assertRaises(httpx.HTTPStatusError):
            client.download(self.request_id, self.root)
        self.assertEqual(downloads, ["primary.test", "backup.test", "primary.test"])
        self.assertEqual(target.read_bytes(), b"existing valid output")
        self.assert_no_partials()

    def test_auth_and_certificate_errors_are_not_retried(self):
        for operation in ("snapshot", "download"):
            for failure in ("auth", "certificate"):
                with self.subTest(operation=operation, failure=failure):
                    failures = []

                    def handler(request):
                        if operation == "download" and request.method == "POST":
                            return self.completed()
                        failures.append(request.url.host)
                        if failure == "auth":
                            return httpx.Response(401)
                        try:
                            raise ssl.SSLCertVerificationError("untrusted certificate")
                        except ssl.SSLCertVerificationError as cause:
                            raise httpx.ConnectError("TLS handshake failed") from cause

                    client = self.client(handler, backup_url="https://backup.test")
                    if operation == "snapshot":
                        with self.assertRaises(httpx.HTTPError):
                            client.snapshot(self.request_id)
                    else:
                        with self.assertRaises(httpx.HTTPError):
                            client.download(self.request_id, self.root)
                    self.assertEqual(failures, ["primary.test"])
                    if operation == "download":
                        self.assert_no_partials()

    def test_missing_and_unconfirmed_recovery_never_starts_inference(self):
        for state in ("missing", "unconfirmed"):
            with self.subTest(state=state):
                operations = []

                def handler(request):
                    operations.append(request.url.path)
                    self.assertTrue(request.url.path.endswith("/request_snapshot"))
                    return httpx.Response(200, json={"data": [{"request_id": self.request_id, "state": state}]})

                client = self.client(handler)
                with self.assertRaises(RequestStateError) as caught:
                    client.wait(self.request_id)
                self.assertEqual(caught.exception.state, state)
                with self.assertRaises(RequestStateError):
                    client.download(self.request_id, self.root)
                self.assertEqual(operations, ["/gradio_api/run/request_snapshot"] * 2)

    def test_upload_retry_reopens_input_and_never_uses_backup(self):
        image = self.root / "source.png"
        image.write_bytes(b"all source bytes")
        uploads = []
        inference_calls = []

        def handler(request):
            if request.url.path.endswith("/upload"):
                uploads.append((request.url.host, request.read()))
                if len(uploads) == 1:
                    raise httpx.ReadError("upload response lost")
                return httpx.Response(200, json=["/cache/source.png"])
            if request.url.path.endswith("/create_request"):
                return httpx.Response(200, json={"data": [self.request_id]})
            inference_calls.append(request.url.path)
            return httpx.Response(200, json={"event_id": "queued-event"})

        client = self.client(handler, backup_url="https://backup.test")
        self.assertEqual(client.start(image, self.root), self.request_id)
        self.assertEqual([host for host, _ in uploads], ["primary.test"] * 2)
        self.assertTrue(all(b"all source bytes" in body for _, body in uploads))
        self.assertEqual(inference_calls, ["/gradio_api/call/inference"])

    def test_backup_snapshot_must_match_id_and_cannot_change_artifact(self):
        lookups = []

        def handler(request):
            lookups.append(request.url.host)
            if len(lookups) == 1:
                raise httpx.RemoteProtocolError("unexpected EOF")
            return httpx.Response(200, json={"data": [{"request_id": "different-request", "state": "missing"}]})

        client = self.client(handler, backup_url="https://backup.test")
        with self.assertRaises(SeeThroughError):
            client.snapshot(self.request_id)
        self.assertEqual(lookups, ["primary.test", "backup.test"])

    def test_recovered_receipt_pins_hash_across_client_instances(self):
        target = self.final()
        receipt = target.parent / "request.json"
        receipt.write_text(json.dumps({"request_id": self.request_id, "psd": self.metadata}))
        operations = []
        changed = dict(self.metadata, sha256="0" * 64)

        def handler(request):
            operations.append(request.method)
            return self.completed(changed)

        client = self.client(handler)
        with self.assertRaises(ArtifactError):
            client.download(self.request_id, self.root)
        self.assertEqual(operations, ["POST"])
        self.assertEqual(target.read_bytes(), b"existing valid output")
        self.assertEqual(json.loads(receipt.read_text())["psd"], self.metadata)

    def test_cache_refresh_can_change_path_but_not_hash_or_size(self):
        for changed_hash in (False, True):
            with self.subTest(changed_hash=changed_hash):
                self.final()
                snapshots = []
                downloads = []
                refreshed = dict(self.metadata, path="/new-cache/output.psd")
                if changed_hash:
                    refreshed["sha256"] = "1" * 64

                def handler(request):
                    if request.method == "POST":
                        snapshots.append(request.url.path)
                        return self.completed(self.metadata if len(snapshots) == 1 else refreshed)
                    downloads.append(str(request.url))
                    if len(downloads) == 1:
                        return httpx.Response(404)
                    return httpx.Response(200, content=self.payload)

                client = self.client(handler)
                if changed_hash:
                    with self.assertRaises(ArtifactError):
                        client.download(self.request_id, self.root)
                    self.assertEqual(len(downloads), 1)
                else:
                    self.assertEqual(client.download(self.request_id, self.root).read_bytes(), self.payload)
                    self.assertIn("new-cache", downloads[-1])
                self.assert_no_partials()
                # Isolate the receipt identity between the two scenarios.
                (self.root / self.request_id / "request.json").unlink()

    def test_download_does_not_follow_redirect_to_another_host(self):
        operations = []

        def handler(request):
            operations.append(request.url.host)
            if request.method == "POST":
                return self.completed()
            return httpx.Response(302, headers={"location": "https://untrusted.test/secret.psd"})

        client = self.client(handler)
        with self.assertRaises(httpx.HTTPStatusError):
            client.download(self.request_id, self.root)
        self.assertEqual(operations, ["primary.test", "primary.test"])
        self.assert_no_partials()

    def test_wait_timeout_leaves_existing_request_untouched(self):
        operations = []

        def handler(request):
            operations.append(request.url.path)
            return httpx.Response(200, json={"data": [{"request_id": self.request_id, "state": "running"}]})

        client = self.client(handler)
        original_handler = handler

        def slow_status(request):
            response = original_handler(request)
            clock.return_value = 2
            return response

        client._http.close()
        client._http = httpx.Client(transport=httpx.MockTransport(slow_status))
        with patch("client_gradio.time.monotonic", return_value=0) as clock:
            with self.assertRaises(WaitTimeout) as caught:
                client.wait(self.request_id, timeout=1)
        self.assertEqual(caught.exception.request_id, self.request_id)
        self.assertEqual(operations, ["/gradio_api/run/request_snapshot"])

    def test_receipt_boundary_rejects_invalid_ids_without_touching_outside_files(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "request.json"
        sentinel.write_bytes(b"outside receipt")
        output = self.root / "outputs"
        client = self.client(lambda request: self.fail("Unexpected transport"))
        for request_id in ("../outside", str(outside), ".", "a" * 31, "A" * 32):
            with self.subTest(request_id=request_id):
                with self.assertRaises(SeeThroughError):
                    client._receipt(request_id, output)
                with self.assertRaises(SeeThroughError):
                    client.download(request_id, output)
        self.assertEqual(sentinel.read_bytes(), b"outside receipt")
        self.assertFalse(output.exists())

    def test_request_directory_symlink_cannot_read_or_write_outside_root(self):
        outside = self.root / "outside"
        outside.mkdir()
        receipt = outside / "request.json"
        receipt.write_bytes(b"outside receipt")
        artifact = outside / "input.psd"
        artifact.write_bytes(b"outside PSD")
        output = self.root / "outputs"
        output.mkdir()
        (output / self.request_id).symlink_to(outside, target_is_directory=True)
        client = self.client(lambda request: self.fail("Unexpected transport"))
        with self.assertRaises(SeeThroughError):
            client.download(self.request_id, output)
        with self.assertRaises(SeeThroughError):
            client._receipt(self.request_id, output, self.metadata)
        self.assertEqual(receipt.read_bytes(), b"outside receipt")
        self.assertEqual(artifact.read_bytes(), b"outside PSD")
        self.assertEqual(sorted(path.name for path in outside.iterdir()), ["input.psd", "request.json"])

    def test_receipt_and_artifact_symlinks_are_rejected_before_transport(self):
        outside = self.root / "sentinel"
        outside.write_bytes(b"do not read or replace")
        directory = self.root / self.request_id
        directory.mkdir()
        client = self.client(lambda request: self.fail("Unexpected transport"))
        for name in ("request.json", "input.psd"):
            for dangling in (False, True):
                with self.subTest(name=name, dangling=dangling):
                    target = self.root / "absent" if dangling else outside
                    link = directory / name
                    link.symlink_to(target)
                    with self.assertRaises(SeeThroughError):
                        client.download(self.request_id, self.root)
                    with self.assertRaises(SeeThroughError):
                        client._receipt(self.request_id, self.root, self.metadata)
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(outside.read_bytes(), b"do not read or replace")
                    self.assertFalse((self.root / "absent").exists())
                    link.unlink()

    def test_artifact_symlink_created_during_download_cannot_replace_outside_file(self):
        sentinel = self.root / "sentinel"
        sentinel.write_bytes(b"outside PSD")
        target = self.root / self.request_id / "input.psd"

        def handler(request):
            if request.method == "POST":
                return self.completed()
            target.symlink_to(sentinel)
            return httpx.Response(200, content=self.payload)

        client = self.client(handler)
        with self.assertRaises(SeeThroughError):
            client.download(self.request_id, self.root)
        self.assertTrue(target.is_symlink())
        self.assertEqual(sentinel.read_bytes(), b"outside PSD")
        self.assert_no_partials()

    def test_cli_rejects_escaping_output_before_constructing_transport(self):
        project = self.root / "project"
        project.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_bytes(b"unchanged")
        link = project / "linked-output"
        link.symlink_to(outside, target_is_directory=True)
        outputs = (outside, project / ".." / "outside", link, "../outside")
        for command in ("run", "recover"):
            for output in outputs:
                with self.subTest(command=command, output=output):
                    argument = "image.png" if command == "run" else self.request_id
                    with patch("client_gradio.os.getcwd", return_value=str(project)):
                        with patch("client_gradio.SeeThroughClient") as transport:
                            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                                result = main([command, "https://primary.test", argument, "--output", str(output)])
                    self.assertEqual(result, 1)
                    transport.assert_not_called()
                    self.assertEqual(sentinel.read_bytes(), b"unchanged")
                    self.assertEqual(list(outside.iterdir()), [sentinel])

    def test_library_keeps_explicit_custom_output_root(self):
        custom = self.root / "custom-output"

        def handler(request):
            if request.method == "POST":
                return self.completed()
            return httpx.Response(200, content=self.payload)

        client = self.client(handler)
        target = client.download(self.request_id, custom)
        self.assertEqual(target, custom / self.request_id / "input.psd")
        self.assertEqual(target.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
