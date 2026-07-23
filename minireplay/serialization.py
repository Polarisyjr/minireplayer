from __future__ import annotations

import base64
import dataclasses
import enum
import math
from pathlib import Path
from typing import Any

from .util import sha256_bytes


def jsonable(value: Any) -> Any:
    """Encode native boundary values without reducing them to an opaque repr."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"$float": repr(value)}
        return value
    if isinstance(value, bytes):
        return {
            "$bytes_base64": base64.b64encode(value).decode("ascii"),
            "size": len(value),
            "sha256": sha256_bytes(value),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return jsonable(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return jsonable(model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        encoded = [jsonable(item) for item in value]
        return sorted(encoded, key=repr)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return jsonable(to_dict())
    raise TypeError(f"native boundary value is not serializable: {type(value).__qualname__}")
