from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from .errors import ValidationError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
                require(
                    isinstance(value, dict),
                    f"{path}:{line_number}: expected a JSON object",
                )
                yield value
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read JSONL {path}: {exc}") from exc


def atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    atomic_write(path, canonical_json(dict(value)) + b"\n", mode=mode)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(dict(value)) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short append")
            view = view[written:]
    finally:
        os.close(fd)


def ensure_empty_directory(path: Path) -> None:
    if path.exists():
        require(path.is_dir(), f"output is not a directory: {path}")
        require(not any(path.iterdir()), f"output directory is not empty: {path}")
    else:
        path.mkdir(parents=True)


def monotonic_ns() -> int:
    return time.monotonic_ns()


def wall_time_ns() -> int:
    return time.time_ns()


def unique_strings(values: Iterable[Any], context: str) -> set[str]:
    result: set[str] = set()
    for value in values:
        require(isinstance(value, str) and bool(value), f"{context}: invalid identifier")
        require(value not in result, f"{context}: duplicate identifier {value!r}")
        result.add(value)
    return result
