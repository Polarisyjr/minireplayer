from __future__ import annotations

from pathlib import Path
from typing import Any

_TYPED_PATH_FIELDS = {
    "mini-swe": frozenset({"cwd"}),
    "trae": frozenset({"path", "file_path", "cwd"}),
    "coral": frozenset(
        {"filePath", "filepath", "outputPath", "file", "path", "dir", "cwd", "physical_path"}
    ),
    "owl": frozenset({"path", "file_path", "cwd", "physical_path"}),
}


def bind_deployment_path(value: str, mapping: dict[str, str]) -> str:
    """Normalize one field explicitly declared to contain a path."""

    for physical, logical in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        physical = physical.rstrip("/")
        logical = logical.rstrip("/")
        if value == physical:
            return logical
        if value.startswith(f"{physical}/"):
            return f"{logical}{value[len(physical) :]}"
    return value


LOGICAL_RUN = "/native-run"
LOGICAL_REPO = "/native-repo"


def run_path_map(run_root: Path, repo: Path) -> dict[str, str]:
    """Physical directories of this run, mapped to run-independent names.

    A recorded tool argument names the directory the recording ran in. Comparing
    that verbatim against a replay in a different directory would reject a correct
    replay, so both sides are reduced to these logical names before comparison and
    expanded back to the live directories before the call is made.

    Longest path first, so a nested directory wins over its parent.
    """

    return {
        str((run_root / "workspace").resolve()): "/native-workspace",
        str(run_root.resolve()): LOGICAL_RUN,
        str(repo.resolve()): LOGICAL_REPO,
    }


def to_physical(
    adapter: str,
    value: Any,
    run_root: Path,
    repo: Path,
    workspace_path: Path | None = None,
) -> Any:
    """Expand logical names back into this run's directories."""

    deployment = {
        "/native-workspace": str(
            (workspace_path if workspace_path is not None else run_root / "workspace").resolve()
        ),
        LOGICAL_RUN: str(run_root.resolve()),
        LOGICAL_REPO: str(repo.resolve()),
    }
    return bind_typed_fields(adapter, value, deployment)


def bind_typed_fields(adapter: str, value: Any, path_map: dict[str, str]) -> Any:
    declared = _TYPED_PATH_FIELDS[adapter]

    def visit(item: Any, field: str | None = None) -> Any:
        if isinstance(item, dict):
            return {key: visit(child, str(key)) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, field) for child in item]
        if isinstance(item, str) and field in declared:
            return bind_deployment_path(item, path_map)
        return item

    return visit(value)


def recorded_path_map(adapter: str, native_value: Any, logical_value: Any) -> dict[str, str]:
    """Recover exact source-to-logical bindings from one recorded operation.

    Forced LLM output can contain the source run's absolute tool arguments. During
    replay, those values arrive before the dispatch hook has a chance to rebind
    them to the live workspace. Pairing the record's native and logical arguments
    lets the entry gate recognize only those exact, typed source paths.
    """

    declared = _TYPED_PATH_FIELDS[adapter]
    mapping: dict[str, str] = {}

    def visit(native: Any, logical: Any, field: str | None = None) -> None:
        if isinstance(native, dict) and isinstance(logical, dict):
            for key in native.keys() & logical.keys():
                visit(native[key], logical[key], str(key))
            return
        if isinstance(native, list) and isinstance(logical, list):
            for native_child, logical_child in zip(native, logical, strict=False):
                visit(native_child, logical_child, field)
            return
        if (
            isinstance(native, str)
            and isinstance(logical, str)
            and field in declared
            and native != logical
        ):
            previous = mapping.setdefault(native, logical)
            if previous != logical:
                raise ValueError(
                    f"recorded path {native!r} maps to both {previous!r} and {logical!r}"
                )

    visit(native_value, logical_value)
    return mapping


def recorded_workspace_path_map(
    adapter: str,
    native_value: Any,
    logical_value: Any,
) -> dict[str, str]:
    """Recover source workspace roots proved by typed path arguments.

    A later shell command can embed the same absolute workspace in free text.
    We do not parse or rewrite arbitrary shell syntax; this derives only the
    physical prefix whose typed counterpart was already normalized to
    ``/native-workspace``.
    """

    mapping: dict[str, str] = {}
    for physical, logical in recorded_path_map(adapter, native_value, logical_value).items():
        for root in ("/native-workspace",):
            if logical == root:
                mapping[physical] = root
                continue
            prefix = f"{root}/"
            if not logical.startswith(prefix):
                continue
            suffix = logical[len(root) :]
            if physical.endswith(suffix):
                mapping[physical[: -len(suffix)]] = root
    return mapping


def rebind_embedded_coral_paths(
    value: Any,
    source_path_map: dict[str, str],
    run_root: Path,
    workspace_path: Path | None,
) -> Any:
    """Rebind proven source paths inside CORAL shell ``command`` fields."""

    deployment = {
        "/native-workspace": str(
            (workspace_path if workspace_path is not None else run_root / "workspace").resolve()
        ),
        LOGICAL_RUN: str(run_root.resolve()),
    }
    replacements = [
        (physical, deployment[logical])
        for physical, logical in source_path_map.items()
        if logical in deployment
    ]
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def visit(item: Any, field: str | None = None) -> Any:
        if isinstance(item, dict):
            return {key: visit(child, str(key)) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, field) for child in item]
        if isinstance(item, str) and field == "command":
            result = item
            for physical, live in replacements:
                result = result.replace(physical, live)
            return result
        return item

    return visit(value)


def restore_embedded_coral_source_paths(
    value: Any,
    source_path_map: dict[str, str],
    run_root: Path,
    workspace_path: Path | None,
) -> Any:
    """Return live CORAL commands to their recorded spelling for claims only."""

    live_workspace = str(
        (workspace_path if workspace_path is not None else run_root / "workspace").resolve()
    )
    source_workspaces = sorted(
        (
            physical
            for physical, logical in source_path_map.items()
            if logical == "/native-workspace"
        ),
        key=len,
        reverse=True,
    )
    if not source_workspaces:
        return value
    source_workspace = source_workspaces[0]

    def visit(item: Any, field: str | None = None) -> Any:
        if isinstance(item, dict):
            return {key: visit(child, str(key)) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, field) for child in item]
        if isinstance(item, str) and field == "command":
            return item.replace(live_workspace, source_workspace)
        return item

    return visit(value)


def rebind_typed_fields(
    adapter: str,
    value: Any,
    source_path_map: dict[str, str],
    run_root: Path,
) -> Any:
    logical = bind_typed_fields(adapter, value, source_path_map)
    deployment = {
        "/native-workspace": str((run_root / "workspace").resolve()),
        "/native-run": str(run_root.resolve()),
    }
    return bind_typed_fields(adapter, logical, deployment)
