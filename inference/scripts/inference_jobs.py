"""Disk-bound inference receipts and final artifacts, independent of UI delivery."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from threading import RLock
import uuid


_TOKEN = re.compile(r"[0-9a-f]{32}")


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.instance_id = uuid.uuid4().hex
        self._lock = RLock()

    def job_dir(self, token):
        if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
            raise ValueError("Invalid request ID")
        directory = self.root / token
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.root):
            raise ValueError("Request directory escapes storage")
        return directory

    def _read(self, token):
        receipt = self.job_dir(token) / "receipt.json"
        if receipt.is_symlink():
            raise ValueError("Receipt cannot be a symbolic link")
        try:
            with receipt.open(encoding="utf-8") as stream:
                return json.load(stream)
        except FileNotFoundError:
            return None

    def _require(self, token):
        record = self._read(token)
        if record is None:
            raise KeyError(token)
        return record

    def _write(self, token, record):
        directory = self.job_dir(token)
        temporary = directory / "receipt.json.tmp"
        if temporary.is_symlink():
            raise ValueError("Receipt temporary file cannot be a symbolic link")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, directory / "receipt.json")

    def create(self):
        with self._lock:
            token = uuid.uuid4().hex
            self.job_dir(token).mkdir()
            timestamp = _now()
            self._write(token, {
                "id": token, "state": "queued",
                "created_at": timestamp, "updated_at": timestamp,
            })
            return token

    def get(self, token):
        with self._lock:
            return self._read(token)

    def start(self, token, parameters):
        with self._lock:
            record = self._require(token)
            if record["state"] != "queued":
                raise ValueError("Only queued requests can start")
            timestamp = _now()
            record.update(state="running", owner=self.instance_id,
                          started_at=timestamp, updated_at=timestamp,
                          parameters=parameters)
            self._write(token, record)

    def progress(self, token, event):
        with self._lock:
            record = self._require(token)
            if record["state"] != "running":
                return
            if not isinstance(event, dict):
                raise ValueError("Progress must be a phase/count dictionary")
            record.update(progress=dict(event), updated_at=_now())
            self._write(token, record)

    def complete(self, token, job_dir, result):
        with self._lock:
            record = self._require(token)
            if record["state"] != "running":
                raise ValueError("Only running requests can complete")
            source_root = Path(job_dir).resolve(strict=True)
            destination_root = self.job_dir(token) / "artifacts"
            if destination_root.is_symlink():
                raise ValueError("Artifact directory cannot be a symbolic link")
            moves = {}

            def artifact(relative):
                if not isinstance(relative, str):
                    raise ValueError("Artifact paths must be relative strings")
                path = Path(relative)
                if path.is_absolute() or not path.parts or ".." in path.parts:
                    raise ValueError("Artifact path escapes worker directory")
                source = source_root / path
                resolved = source.resolve(strict=True)
                if not resolved.is_relative_to(source_root) or source != resolved:
                    raise ValueError("Artifact path escapes worker directory or uses a symbolic link")
                if not source.is_file():
                    raise ValueError("Artifact must be a file")
                destination = destination_root / path
                if not destination.resolve().is_relative_to(destination_root):
                    raise ValueError("Artifact destination escapes storage")
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(destination)
                moves[path] = (source, destination)
                return (Path("artifacts") / path).as_posix()

            saved_result = {
                "psd": artifact(result["psd"]),
                "gallery": [
                    {"path": artifact(item["path"]), "label": item["label"]}
                    for item in result["gallery"]
                ],
            }
            # Preflight every path before moving any output. Originals and worker
            # intermediates never enter persistent storage.
            for source, destination in moves.values():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            timestamp = _now()
            record.update(state="completed", result=saved_result,
                          finished_at=timestamp, updated_at=timestamp)
            self._write(token, record)
            return record

    def cancel_queued(self, token):
        with self._lock:
            record = self._require(token)
            if record["state"] != "queued":
                return False
            timestamp = _now()
            record.update(state="cancelled", error="Cancelled before execution",
                          finished_at=timestamp, updated_at=timestamp)
            self._write(token, record)
            return True

    def fail(self, token, state, error):
        if state not in {"failed", "cancelled"}:
            raise ValueError("Failure state must be failed or cancelled")
        with self._lock:
            record = self._require(token)
            # UI delivery can fail after durable completion; it must not erase
            # outputs. Repeated cleanup must also preserve the original error.
            if record["state"] in {"completed", "failed", "cancelled"}:
                return
            timestamp = _now()
            record.update(state=state, error=str(error),
                          finished_at=timestamp, updated_at=timestamp)
            self._write(token, record)
