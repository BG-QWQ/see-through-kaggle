"""Small synchronous HTTP client for the public, queued SeeThrough service.

Submission is deliberately not retried. Recover with the saved request ID instead.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import sys
import tempfile
import time
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

_RECEIPT_FILE = "request.json"


class SeeThroughError(RuntimeError):
    """An invalid service response or an unsuccessful request."""


class SubmissionUncertain(SeeThroughError):
    """The submission may have reached the queue; never submit it again."""

    def __init__(self, request_id):
        self.request_id = request_id
        super().__init__(f"Submission outcome uncertain; recover request {request_id}, do not resubmit")


class RequestStateError(SeeThroughError):
    def __init__(self, request_id, state, detail=None):
        self.request_id = request_id
        self.state = state
        super().__init__(f"Request {request_id} is {state}" + (f": {detail}" if detail else ""))


class WaitTimeout(SeeThroughError):
    def __init__(self, request_id):
        self.request_id = request_id
        super().__init__(f"Timed out waiting for {request_id}; request was not cancelled")


class ArtifactError(SeeThroughError):
    """Artifact metadata changed or downloaded bytes failed verification."""


class _CorruptDownload(ArtifactError):
    pass


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _base_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("Service URL must be HTTP(S), without credentials, query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("Invalid service port") from exc
    return urlunsplit(parsed).rstrip("/")


def _request_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise SeeThroughError("Invalid request ID")
    return value


def _cli_output(value):
    # CLI arguments are untrusted; the caller's working directory is the boundary.
    root = Path.cwd().resolve()
    output = Path(value).resolve()
    if not output.is_relative_to(root):
        raise ValueError("CLI output must stay within the current working directory")
    return output


def _local_path(directory, name):
    path = directory / name
    if path.is_symlink() or not path.resolve().is_relative_to(directory):
        raise SeeThroughError("Local output must not escape its directory or use symlinks")
    return path


def _request_directory(output_dir, request_id, *, create=False):
    request_id = _request_id(request_id)
    # Library callers explicitly select their trusted root, including custom roots.
    root = Path(output_dir).resolve()
    directory = _local_path(root, request_id)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    # Validate both children before reading a receipt or updating either output.
    _local_path(directory, _RECEIPT_FILE)
    _local_path(directory, "input.psd")
    return directory


def _certificate_failure(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ssl.SSLCertVerificationError) or "certificate verify failed" in str(exc).lower() or "certificate_verify_failed" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _retryable(exc):
    if _certificate_failure(exc):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 502, 503, 504)
    return isinstance(exc, (httpx.NetworkError, httpx.TimeoutException,
                            httpx.RemoteProtocolError, httpx.ProxyError, _CorruptDownload))


def _psd(snapshot):
    metadata = snapshot.get("psd")
    if not isinstance(metadata, dict):
        raise ArtifactError("Completed request has no PSD metadata")
    path, size, digest = (metadata.get(key) for key in ("path", "size", "sha256"))
    if (not isinstance(path, str) or not path.startswith("/") or path.startswith("//")
            or isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
        raise ArtifactError("Invalid PSD metadata")
    return {"path": path, "size": size, "sha256": digest.lower()}


class SeeThroughClient:
    def __init__(self, base_url, backup_url=None, timeout=120, attempts=3):
        self.base_url = _base_url(base_url)
        self.backup_url = _base_url(backup_url) if backup_url is not None else None
        self.timeout = _positive(timeout, "timeout")
        self.attempts = _integer(attempts, "attempts")
        self._urls = [self.base_url]
        if self.backup_url and self.backup_url != self.base_url:
            self._urls.append(self.backup_url)
        self._metadata = {}
        self._http = httpx.Client(
            verify=True, follow_redirects=False,
            timeout=httpx.Timeout(connect=30, read=timeout, write=timeout, pool=30),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._http.close()

    def _retry(self, operation, *, primary_only=False):
        for attempt in range(self.attempts):
            base = self.base_url if primary_only else self._urls[attempt % len(self._urls)]
            try:
                return operation(base)
            except Exception as exc:
                if attempt + 1 == self.attempts or not _retryable(exc):
                    raise
                time.sleep(min(0.25 * 2 ** min(attempt, 3), 2))

    @staticmethod
    def _data(response):
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("data"), list) or len(body["data"]) != 1:
            raise SeeThroughError("Invalid service response envelope")
        return body["data"][0]

    def _receipt(self, request_id, output_dir, metadata=None, *, initial=False):
        directory = _request_directory(output_dir, request_id, create=True)
        target = _local_path(directory, _RECEIPT_FILE)
        record = {"request_id": request_id, "base_url": self.base_url}
        if self.backup_url:
            record["backup_url"] = self.backup_url
        if metadata is not None:
            record["psd"] = metadata
        if initial:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".request-", dir=directory)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(record, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                _request_directory(output_dir, request_id)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return directory

    def start(self, image_path, output_dir, resolution=1024, seed=42, steps=30, tblr_split=False):
        _integer(resolution, "resolution")
        _integer(seed, "seed", minimum=0)
        _integer(steps, "steps")
        if not isinstance(tblr_split, bool):
            raise ValueError("tblr_split must be boolean")
        image_path = Path(image_path)

        def upload(base):
            # Reopen for every attempt: a failed multipart send may consume the file.
            with image_path.open("rb") as source:
                response = self._http.post(base + "/gradio_api/upload", files={"files": (image_path.name, source)})
            response.raise_for_status()
            paths = response.json()
            if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str) or not paths[0].startswith("/"):
                raise SeeThroughError("Invalid upload response")
            return paths[0]

        uploaded_path = self._retry(upload, primary_only=True)
        # Neither create nor inference POST is automatically retried.
        request_id = _request_id(self._data(self._http.post(
            self.base_url + "/gradio_api/run/create_request", json={"data": []})))
        self._receipt(request_id, output_dir, initial=True)
        try:
            response = self._http.post(self.base_url + "/gradio_api/call/inference", json={"data": [
                request_id, {"path": uploaded_path, "orig_name": image_path.name, "meta": {"_type": "gradio.FileData"}},
                resolution, seed, tblr_split, steps,
            ]})
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("event_id"), str) or not body["event_id"]:
                raise SeeThroughError("Inference response has no event ID")
        except Exception as exc:
            raise SubmissionUncertain(request_id) from exc
        return request_id

    def _pin(self, request_id, metadata):
        previous = self._metadata.get(request_id)
        if previous is not None and any(previous[key] != metadata[key] for key in ("size", "sha256")):
            raise ArtifactError("PSD size/hash changed for the same request")
        self._metadata[request_id] = metadata

    def snapshot(self, request_id):
        request_id = _request_id(request_id)

        def lookup(base):
            return self._data(self._http.post(base + "/gradio_api/run/request_snapshot", json={"data": [request_id]}))

        result = self._retry(lookup)
        if not isinstance(result, dict) or result.get("request_id") != request_id:
            raise SeeThroughError("Snapshot does not match the requested ID")
        if result.get("state") not in ("queued", "running", "completed", "failed", "cancelled", "missing", "unconfirmed"):
            raise SeeThroughError("Invalid request state")
        if result["state"] == "completed":
            self._pin(request_id, _psd(result))
        return result

    @staticmethod
    def _terminal(snapshot):
        state = snapshot["state"]
        if state in ("failed", "cancelled", "missing", "unconfirmed"):
            raise RequestStateError(snapshot["request_id"], state, snapshot.get("error"))

    def wait(self, request_id, timeout=7200, poll_interval=5):
        _request_id(request_id)
        deadline = time.monotonic() + _positive(timeout, "wait timeout")
        _positive(poll_interval, "poll interval")
        while True:
            if time.monotonic() >= deadline:
                raise WaitTimeout(request_id)
            result = self.snapshot(request_id)
            self._terminal(result)
            if result["state"] == "completed":
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WaitTimeout(request_id)
            time.sleep(min(poll_interval, remaining))

    def _load_receipt(self, request_id, directory):
        receipt = _local_path(directory, _RECEIPT_FILE)
        if not receipt.exists():
            return
        descriptor = os.open(receipt, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            saved = json.load(stream)
        if not isinstance(saved, dict) or saved.get("request_id") != request_id:
            raise SeeThroughError("Local receipt does not match the requested ID")
        if "psd" in saved:
            self._pin(request_id, _psd(saved))

    def _completed_metadata(self, request_id):
        result = self.snapshot(request_id)
        self._terminal(result)
        if result["state"] != "completed":
            raise RequestStateError(request_id, result["state"], "PSD is not ready")
        return _psd(result)

    @staticmethod
    def _write_verified(response, stream, metadata):
        response.raise_for_status()
        digest = hashlib.sha256()
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > metadata["size"]:
                raise _CorruptDownload("PSD exceeds the advertised size")
            digest.update(chunk)
            stream.write(chunk)
        if size != metadata["size"] or digest.hexdigest() != metadata["sha256"]:
            raise _CorruptDownload("PSD size or SHA256 verification failed")
        stream.flush()
        os.fsync(stream.fileno())

    def _fetch_artifact(self, base, directory, metadata):
        target = _local_path(directory, "input.psd")
        descriptor, temporary = tempfile.mkstemp(prefix=".input-", suffix=".part", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                # Only a quoted absolute cache path, never a server-provided URL.
                url = base + "/gradio_api/file=" + quote(metadata["path"], safe="")
                with self._http.stream("GET", url) as response:
                    if response.status_code == 404:
                        return None
                    self._write_verified(response, stream, metadata)
            _request_directory(directory.parent, directory.name)
            os.replace(temporary, target)
            return target
        finally:
            Path(temporary).unlink(missing_ok=True)

    def download(self, request_id, output_dir):
        request_id = _request_id(request_id)
        directory = _request_directory(output_dir, request_id)
        self._load_receipt(request_id, directory)
        metadata = self._completed_metadata(request_id)
        directory = self._receipt(request_id, output_dir, metadata)

        def fetch(base):
            nonlocal metadata
            target = self._fetch_artifact(base, directory, metadata)
            if target is None:
                metadata = self._completed_metadata(request_id)
                # Refresh only the cache path; snapshot pins size and hash.
                raise _CorruptDownload("PSD cache path missing; refreshed metadata")
            return target

        return self._retry(fetch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "recover"):
        command = commands.add_parser(name)
        command.add_argument("url")
        command.add_argument("image" if name == "run" else "request_id")
        command.add_argument("--output", default="workspace/client-results", type=Path)
        command.add_argument("--backup")
        command.add_argument("--timeout", type=float, default=120)
        command.add_argument("--attempts", type=int, default=3)
        command.add_argument("--wait-timeout", type=float, default=7200)
        if name == "run":
            command.add_argument("--resolution", type=int, default=1024)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--steps", type=int, default=30)
            command.add_argument("--tblr-split", action="store_true")
    args = parser.parse_args(argv)
    request_id = args.request_id if args.command == "recover" else None
    try:
        _positive(args.wait_timeout, "wait timeout")
        output = _cli_output(args.output)
        with SeeThroughClient(args.url, args.backup, args.timeout, args.attempts) as client:
            if args.command == "run":
                request_id = client.start(args.image, output, args.resolution, args.seed, args.steps, args.tblr_split)
                print(f"Saved request ID: {request_id}", flush=True)
            client.wait(request_id, timeout=args.wait_timeout)
            print(client.download(request_id, output))
        return 0
    except (SeeThroughError, httpx.HTTPError, OSError, ValueError) as exc:
        if isinstance(exc, SubmissionUncertain):
            request_id = exc.request_id
        # Transport exception strings may include private URLs; do not echo them.
        print(f"Request failed ({type(exc).__name__}).", file=sys.stderr)
        if isinstance(exc, RequestStateError):
            print(f"Request state: {exc.state}", file=sys.stderr)
        if request_id is not None:
            print(f"Recover existing request {request_id}: client_gradio.py recover URL {request_id} --output DIR"
                  " (use the same service URL and output directory; optionally --backup URL)",
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
