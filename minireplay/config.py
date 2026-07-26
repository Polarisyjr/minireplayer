"""Run configuration.

This replaces the old plan/prepare-launch pair. There is no task-resolution step:
the sweep script owns the task list, and an actor names itself at the gate (see
``instrumentation.gate.derive_actor_id``), so nothing here needs to know which
tasks will run. The config only says which sweep to invoke and where its LLM
traffic goes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import CONFIG_SCHEMA, SUPPORTED_ADAPTERS
from .util import read_json, require

LEGACY_LOAD_MODELS = {"steady": True, "fire-once": False}
DEFAULT_DURATION_S = 180
CORAL_TEAM_SIZE = 4
CORAL_AGENT_TURNS = 100
CORAL_GLOBAL_TURNS = 10
CORAL_RESTART_EXITED = True

# Only these keys may differ between the run that produced a bundle and a run
# that replays it. Everything else is workload identity and must match.
DEPLOYMENT_KEYS = frozenset({"targets", "env", "results_root", "gpu_ids", "cpuset"})


@dataclass(frozen=True)
class RunConfig:
    framework: str
    repo: Path
    concurrency: int
    duration_s: int
    seed: int
    targets: dict[str, str]
    env: dict[str, str] = field(default_factory=dict)
    coral_dataset: str = "frontier_cs_algo"
    # CORAL has two distinct turn budgets. ``agent_turns`` counts OpenCode
    # ``step_finish`` records inside one invocation. ``global_turns`` counts
    # invocations across the four-agent team, including the initial four.
    coral_agent_turns: int = CORAL_AGENT_TURNS
    coral_global_turns: int = CORAL_GLOBAL_TURNS
    coral_restart_exited: bool = CORAL_RESTART_EXITED
    # Whether a completed actor is replaced with the next task in seeded order.
    # This is one workload dimension shared by every framework.
    refill: bool = True
    results_root: str | None = None
    cpuset: str | None = None
    gpu_ids: list[int] = field(default_factory=list)
    # Only needed for `vllm-up` and `--mode full`; tool-only replay ignores it.
    serving: dict[str, Any] = field(default_factory=dict)

    @property
    def adapter(self) -> str:
        return self.framework

    @property
    def coral_team_size(self) -> int:
        return CORAL_TEAM_SIZE

    def workload_identity(self) -> dict[str, Any]:
        """The part of the config a replay must reproduce exactly."""

        identity = {
            "framework": self.framework,
            "concurrency": self.concurrency,
            "duration_s": self.duration_s,
            "seed": self.seed,
            "refill": self.refill,
        }
        if self.framework == "coral":
            identity["concurrency_unit"] = "coral-team"
            identity["coral_team_size"] = self.coral_team_size
            identity["coral_dataset"] = self.coral_dataset
            identity["coral_agent_turns"] = self.coral_agent_turns
            identity["coral_global_turns"] = self.coral_global_turns
            identity["coral_restart_exited"] = self.coral_restart_exited
        return identity

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "framework": self.framework,
            "repo": str(self.repo),
            "concurrency": self.concurrency,
            "duration_s": self.duration_s,
            "seed": self.seed,
            "targets": dict(self.targets),
            "env": dict(self.env),
            "coral_dataset": self.coral_dataset,
            "coral_team_size": self.coral_team_size,
            "coral_agent_turns": self.coral_agent_turns,
            "coral_global_turns": self.coral_global_turns,
            "coral_restart_exited": self.coral_restart_exited,
            "refill": self.refill,
            "results_root": self.results_root,
            "cpuset": self.cpuset,
            "gpu_ids": list(self.gpu_ids),
            "serving": dict(self.serving),
        }

    def serving_spec(self):
        """Resolve the serving block into a ServingSpec, or explain what is missing."""

        from .serving import DEFAULT_IMAGE, ServingSpec

        raw = self.serving
        require(
            bool(raw.get("configs")),
            "config.serving.configs is required to start vLLM "
            '(e.g. ["serving/configs/qwen3-coder-30b-tp8.yaml:1"])',
        )
        root = Path(str(raw.get("state_dir", "~/.minireplay"))).expanduser()
        gpu_mode = str(raw.get("gpu_mode", "nvidia"))
        require(
            gpu_mode in {"nvidia", "cdi"},
            "config.serving.gpu_mode must be 'nvidia' or 'cdi'",
        )
        return ServingSpec(
            repo=self.repo,
            configs=[str(value) for value in raw["configs"]],
            image=str(raw.get("image", DEFAULT_IMAGE)),
            secret_path=Path(str(raw.get("secret_path", root / "force-secret"))).expanduser(),
            audit_dir=Path(str(raw.get("audit_dir", root / "audit"))).expanduser(),
            gpu_mode=gpu_mode,
        )


def _int(value: Any, context: str, *, minimum: int) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"config.{context} must be an integer >= {minimum}",
    )
    return int(value)


def load_config(path: Path) -> RunConfig:
    raw = read_json(path.expanduser().resolve())
    require(raw.get("schema_version") == CONFIG_SCHEMA, "config: unsupported schema_version")

    framework = raw.get("framework")
    require(
        framework in SUPPORTED_ADAPTERS,
        f"config.framework must be one of {sorted(SUPPORTED_ADAPTERS)}",
    )

    repo = Path(str(raw.get("repo", ""))).expanduser()
    require(repo.is_dir(), f"config.repo is not a directory: {repo}")
    sweep = repo / "scripts" / str(framework) / "sweep.sh"
    require(sweep.is_file(), f"config.repo has no sweep script: {sweep}")

    targets = raw.get("targets")
    require(
        isinstance(targets, dict) and bool(targets),
        "config.targets must be a non-empty object",
    )
    for name, url in targets.items():
        require(isinstance(name, str) and bool(name), "config.targets has an invalid target id")
        require(
            isinstance(url, str) and url.startswith(("http://", "https://")),
            f"config.targets[{name}] must be an http(s) URL",
        )

    env = raw.get("env", {})
    require(isinstance(env, dict), "config.env must be an object")
    for name, value in env.items():
        require(
            isinstance(name, str) and isinstance(value, str),
            "config.env must map string names to string values",
        )

    gpu_ids = raw.get("gpu_ids", [])
    require(isinstance(gpu_ids, list), "config.gpu_ids must be a list")
    for gpu in gpu_ids:
        require(
            isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0,
            "config.gpu_ids must contain non-negative integers",
        )

    results_root = raw.get("results_root")
    require(
        results_root is None or isinstance(results_root, str),
        "config.results_root must be a string or null",
    )
    cpuset = raw.get("cpuset")
    require(cpuset is None or isinstance(cpuset, str), "config.cpuset must be a string or null")

    serving = raw.get("serving", {})
    require(isinstance(serving, dict), "config.serving must be an object")

    # ``load_model`` was the original owl-only spelling. Keep it as an input-only
    # compatibility alias so old configs do not silently change workload, but emit
    # only the framework-independent ``refill`` field from now on.
    legacy_load_model = raw.get("load_model")
    require(
        legacy_load_model is None or legacy_load_model in LEGACY_LOAD_MODELS,
        f"config.load_model must be one of {sorted(LEGACY_LOAD_MODELS)}",
    )
    require(
        not ("refill" in raw and legacy_load_model is not None),
        "config.refill and deprecated config.load_model cannot both be set",
    )
    refill = (
        LEGACY_LOAD_MODELS[str(legacy_load_model)]
        if legacy_load_model is not None
        else raw.get("refill", True)
    )
    require(isinstance(refill, bool), "config.refill must be a boolean")
    coral_restart_exited = raw.get("coral_restart_exited", CORAL_RESTART_EXITED)
    require(
        isinstance(coral_restart_exited, bool),
        "config.coral_restart_exited must be a boolean",
    )
    coral_team_size = raw.get("coral_team_size", CORAL_TEAM_SIZE)
    require(
        framework != "coral" or coral_team_size == CORAL_TEAM_SIZE,
        f"config.coral_team_size must be {CORAL_TEAM_SIZE}",
    )
    coral_agent_turns = _int(
        raw.get("coral_agent_turns", CORAL_AGENT_TURNS),
        "coral_agent_turns",
        minimum=1,
    )
    coral_global_turns = _int(
        raw.get("coral_global_turns", CORAL_GLOBAL_TURNS),
        "coral_global_turns",
        minimum=0,
    )
    require(
        framework != "coral" or coral_global_turns == 0 or coral_global_turns >= 4,
        "config.coral_global_turns must be 0 or at least the four initial CORAL agents",
    )

    return RunConfig(
        framework=str(framework),
        repo=repo.resolve(),
        concurrency=_int(raw.get("concurrency"), "concurrency", minimum=1),
        duration_s=_int(raw.get("duration_s", DEFAULT_DURATION_S), "duration_s", minimum=1),
        seed=_int(raw.get("seed", 42), "seed", minimum=0),
        targets={str(k): str(v) for k, v in targets.items()},
        env={str(k): str(v) for k, v in env.items()},
        coral_dataset=str(raw.get("coral_dataset", "frontier_cs_algo")),
        coral_agent_turns=coral_agent_turns,
        coral_global_turns=coral_global_turns,
        coral_restart_exited=coral_restart_exited,
        refill=refill,
        results_root=results_root,
        cpuset=cpuset,
        gpu_ids=[int(gpu) for gpu in gpu_ids],
        serving=serving,
    )
