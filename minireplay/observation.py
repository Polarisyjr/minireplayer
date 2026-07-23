from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .constants import RESULT_CONTRACT_SCHEMA
from .errors import ValidationError
from .util import sha256_json


@dataclass(frozen=True)
class ObservationProjection:
    value: Any
    evidence: list[dict[str, Any]]


def exact_result_contract() -> dict[str, Any]:
    return {
        "schema_version": RESULT_CONTRACT_SCHEMA,
        "kind": "exact",
        "fields": [],
    }


def recorded_output_result_contract(pointer: str = "/output") -> dict[str, Any]:
    return {
        "schema_version": RESULT_CONTRACT_SCHEMA,
        "kind": "recorded-observation",
        "fields": [{"json_pointer": pointer, "optional": True}],
    }


def validate_result_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("tool result contract must be an object")
    if set(value) != {"schema_version", "kind", "fields"}:
        raise ValidationError("tool result contract has an invalid field inventory")
    if value.get("schema_version") != RESULT_CONTRACT_SCHEMA:
        raise ValidationError("tool result contract has an unsupported schema")
    kind = value.get("kind")
    if kind not in {"exact", "recorded-observation"}:
        raise ValidationError(f"tool result contract has an invalid kind: {kind!r}")
    fields = value.get("fields")
    if not isinstance(fields, list):
        raise ValidationError("tool result contract fields must be a list")
    if kind == "exact" and fields:
        raise ValidationError("exact tool result contract cannot declare observation fields")
    if kind == "recorded-observation" and not fields:
        raise ValidationError("recorded observation contract has no fields")
    seen: set[str] = set()
    for index, field in enumerate(fields):
        context = f"tool result contract field {index}"
        if not isinstance(field, dict) or set(field) != {"json_pointer", "optional"}:
            raise ValidationError(f"{context} has an invalid field inventory")
        pointer = field.get("json_pointer")
        if not isinstance(pointer, str) or not pointer.startswith("/") or pointer in seen:
            raise ValidationError(f"{context} has an invalid or duplicate JSON pointer")
        if not isinstance(field.get("optional"), bool):
            raise ValidationError(f"{context}.optional must be boolean")
        seen.add(pointer)
    return value


def _pointer_part(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _pointer_parent(document: Any, pointer: str) -> tuple[Any, str] | None:
    parts = [_pointer_part(part) for part in pointer[1:].split("/")]
    cursor = document
    for part in parts[:-1]:
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
            continue
        if isinstance(cursor, list):
            try:
                cursor = cursor[int(part)]
                continue
            except (ValueError, IndexError):
                pass
        return None
    return cursor, parts[-1]


def _get_pointer(document: Any, pointer: str) -> tuple[bool, Any]:
    parent = _pointer_parent(document, pointer)
    if parent is None:
        return False, None
    cursor, leaf = parent
    if isinstance(cursor, dict) and leaf in cursor:
        return True, cursor[leaf]
    if isinstance(cursor, list):
        try:
            return True, cursor[int(leaf)]
        except (ValueError, IndexError):
            pass
    return False, None


def _set_pointer(document: Any, pointer: str, value: Any) -> None:
    parent = _pointer_parent(document, pointer)
    if parent is None:
        raise ValidationError(f"recorded observation pointer disappeared: {pointer}")
    cursor, leaf = parent
    if isinstance(cursor, dict) and leaf in cursor:
        cursor[leaf] = value
        return
    if isinstance(cursor, list):
        try:
            cursor[int(leaf)] = value
            return
        except (ValueError, IndexError):
            pass
    raise ValidationError(f"recorded observation pointer disappeared: {pointer}")


def project_result(value: Any, contract: dict[str, Any]) -> ObservationProjection:
    validate_result_contract(contract)
    projected = copy.deepcopy(value)
    if contract["kind"] == "exact":
        return ObservationProjection(value=projected, evidence=[])

    evidence: list[dict[str, Any]] = []
    for field in contract["fields"]:
        pointer = field["json_pointer"]
        exists, actual = _get_pointer(projected, pointer)
        if not exists:
            if field["optional"]:
                continue
            raise ValidationError(f"recorded observation field is missing: {pointer}")
        evidence.append(
            {
                "json_pointer": pointer,
                "kind": "recorded-framework-observation",
                "actual_sha256": sha256_json(actual),
            }
        )
        _set_pointer(projected, pointer, "<recorded-framework-observation>")
    return ObservationProjection(value=projected, evidence=evidence)
