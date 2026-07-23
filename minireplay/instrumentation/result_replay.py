from __future__ import annotations

import copy
from typing import Any

from minireplay.serialization import jsonable


def restore_object_field(native: Any, recorded: Any, field: str) -> Any:
    native_encoded = jsonable(native)
    if not isinstance(native_encoded, dict) or not isinstance(recorded, dict):
        raise RuntimeError("tool result field restoration requires object results")
    if field not in native_encoded or field not in recorded:
        raise RuntimeError(f"tool result has no restorable {field!r} field")
    restored = copy.copy(native)
    if isinstance(restored, dict):
        restored[field] = recorded[field]
    elif hasattr(restored, field):
        setattr(restored, field, recorded[field])
    else:
        raise RuntimeError(f"native tool result cannot restore field {field!r}")
    return restored


def encode_framework_output(value: Any) -> dict[str, Any]:
    return {"output": jsonable(value)}


def restore_framework_output(native: Any, recorded: Any) -> Any:
    if not isinstance(recorded, dict) or set(recorded) != {"output"}:
        raise RuntimeError("recorded framework output has an invalid envelope")
    expected = recorded["output"]
    actual = jsonable(native)
    if actual == expected:
        return native
    if isinstance(native, str) and isinstance(expected, str):
        return expected
    if isinstance(native, bool) and isinstance(expected, bool):
        return expected
    if isinstance(native, int) and not isinstance(native, bool) and isinstance(expected, int):
        return expected
    if isinstance(native, float) and isinstance(expected, (int, float)):
        return float(expected)
    if native is None and expected is None:
        return None
    if isinstance(native, list) and isinstance(expected, list):
        return expected
    if isinstance(native, tuple) and isinstance(expected, list):
        return tuple(expected)
    if isinstance(native, dict) and isinstance(expected, dict):
        return expected
    raise RuntimeError("native tool output drifted in a type that cannot be safely restored")
