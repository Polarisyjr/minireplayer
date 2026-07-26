"""CORAL-only replay control derived from a fixed bundle."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import ValidationError


def fixed_invocation_limits(
    llm: Iterable[dict[str, Any]],
    cutoff_tails: dict[str, Any],
) -> dict[str, int]:
    """Return the fixed OpenCode invocation count for each CORAL agent.

    A replay restart is useful only when the source bundle has LLM work (closed
    or cutoff) in the next invocation. Native restarts that produced no captured
    work before the team cutoff are control-only and need not be recreated.
    """

    limits: dict[str, int] = {}
    requests = [
        *llm,
        *(
            record
            for record in cutoff_tails.get("llm_requests", [])
            if isinstance(record, dict)
        ),
    ]
    for record in requests:
        actor = record.get("actor_id")
        session = record.get("session_id")
        if not isinstance(actor, str) or not actor:
            continue
        if not isinstance(session, str):
            continue
        match = re.match(rf"^{re.escape(actor)}/invocation-(\d+)(?:/|$)", session)
        if match is None:
            continue
        limit = int(match.group(1)) + 1
        limits[actor] = max(limits.get(actor, 0), limit)
    return limits


def recorded_restart_controls(
    *,
    actors: list[dict[str, Any]],
    task_terminals: list[dict[str, Any]],
    graders: list[dict[str, Any]],
    invocation_limits: dict[str, int],
) -> list[dict[str, Any]]:
    """Read the exact native restart prompts that lead to fixed invocations."""

    run_dirs = {
        str(terminal["actor_id"]): Path(result["run_dir"])
        for terminal in task_terminals
        if isinstance(terminal, dict)
        and isinstance(terminal.get("actor_id"), str)
        and isinstance((result := terminal.get("result")), dict)
        and isinstance(result.get("run_dir"), str)
    }
    graders_by_actor: dict[str, list[dict[str, Any]]] = {}
    for grader in graders:
        graders_by_actor.setdefault(str(grader.get("actor_id")), []).append(grader)

    controls: list[dict[str, Any]] = []
    for actor in actors:
        actor_id = str(actor.get("actor_id") or "")
        limit = invocation_limits.get(actor_id, 0)
        lane = actor.get("lane")
        if limit <= 1 or not isinstance(lane, dict) or lane.get("lane_kind") != "agent":
            continue
        parent = lane.get("parent_actor_id")
        agent_id = lane.get("agent_id")
        run_dir = run_dirs.get(str(parent))
        if run_dir is None or not isinstance(agent_id, str):
            raise ValidationError(f"CORAL actor {actor_id!r} has no native restart log directory")

        logs = run_dir / ".coral" / "public" / "logs"
        for invocation_index in range(1, limit):
            path = logs / f"{agent_id}.{invocation_index}.log"
            try:
                first = next(line for line in path.read_text().splitlines() if line.strip())
                prompt_record = json.loads(first)
            except (OSError, StopIteration, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"CORAL actor {actor_id!r} invocation {invocation_index} "
                    "has no readable control prompt"
                ) from exc
            prompt = prompt_record.get("prompt")
            source = prompt_record.get("source")
            if not isinstance(prompt, str) or not isinstance(source, str):
                raise ValidationError(
                    f"CORAL actor {actor_id!r} invocation {invocation_index} "
                    "has an invalid control prompt"
                )

            trigger_attempt_id = None
            commit = re.search(r"(?m)^Commit:\s*([0-9a-f]{7,40})\s*$", prompt)
            if commit is not None:
                matches = [
                    grader
                    for grader in graders_by_actor.get(actor_id, [])
                    if str(grader.get("trigger_id") or "").startswith(commit.group(1))
                ]
                if len(matches) != 1:
                    raise ValidationError(
                        f"CORAL actor {actor_id!r} invocation {invocation_index} "
                        f"control matches {len(matches)} grader attempts"
                    )
                trigger_attempt_id = str(matches[0]["attempt_id"])
            controls.append(
                {
                    "actor_id": actor_id,
                    "agent_id": agent_id,
                    "invocation_index": invocation_index,
                    "source": source,
                    "prompt": prompt,
                    "trigger_grader_attempt_id": trigger_attempt_id,
                }
            )
    return sorted(
        controls,
        key=lambda record: (str(record["actor_id"]), int(record["invocation_index"])),
    )


def grader_recorded(run_root: Path, attempt_id: str | None) -> bool:
    """Whether replay has completed the grader that triggered a restart."""

    if attempt_id is None:
        return True
    path = run_root / "stage" / "graders.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("attempt_id") == attempt_id:
            return True
    return False
