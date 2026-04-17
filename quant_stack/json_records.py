from __future__ import annotations

import json
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _advisory_file_lock(handle, timeout_seconds: float = 5.0):
    start = time.monotonic()
    if handle.closed:
        raise RuntimeError("Cannot lock closed file handle.")

    if os.name == "nt":
        import msvcrt

        locked = False
        while not locked:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                if (time.monotonic() - start) >= timeout_seconds:
                    raise TimeoutError("Timed out acquiring Windows file lock.")
                time.sleep(0.01)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    locked = False
    while not locked:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError:
            if (time.monotonic() - start) >= timeout_seconds:
                raise TimeoutError("Timed out acquiring POSIX file lock.")
            time.sleep(0.01)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _looks_like_json_array(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            probe = handle.read(256)
    except OSError:
        return False
    stripped = probe.lstrip()
    return bool(stripped) and stripped.startswith("[")


def read_json_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    if _looks_like_json_array(path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, list):
            records = [dict(item) for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            records = [payload]
        else:
            records = []
        if limit is not None:
            return records[-max(int(limit), 0) :]
        return records

    maxlen = None if limit is None else max(int(limit), 0)
    container: deque[dict[str, Any]] | list[dict[str, Any]]
    container = deque(maxlen=maxlen) if maxlen is not None else []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    if isinstance(container, deque):
                        container.append(payload)
                    else:
                        container.append(payload)
    except OSError:
        return []
    return list(container)


def append_json_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        with _advisory_file_lock(handle):
            handle.seek(0)
            probe = handle.read(256)
            stripped = probe.lstrip()
            if stripped.startswith("["):
                handle.seek(0)
                try:
                    existing_payload = json.loads(handle.read())
                except Exception:
                    existing_payload = []
                existing = (
                    [dict(item) for item in existing_payload if isinstance(item, dict)]
                    if isinstance(existing_payload, list)
                    else ([existing_payload] if isinstance(existing_payload, dict) else [])
                )
                handle.seek(0)
                handle.truncate(0)
                for record in existing:
                    handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
                    handle.write("\n")

            handle.seek(0, 2)
            handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
            handle.flush()
