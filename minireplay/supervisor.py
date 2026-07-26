"""Drive one recording or one replay.

Record and replay run the same loop. They differ only in whether a bundle
constrains the framework, and in what ends the run:

* recording ends when the sweep emits its own ``sample_end``;
* replay ends when every slot in the closed recorded prefix has been consumed;
  truncated tails are evidence only and never enter replay.

Keeping one loop is what makes the two runs comparable: the same gate, the same
counter baselines, the same teardown, so a makespan difference is a property of the
workload and not of the harness.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import (
    Bundle,
    build_bundle,
    bundle_id_for,
    load_bundle,
    materialize_pre_dispatch_tails,
)
from .cgroup import RoleCgroup
from .config import RunConfig
from .constants import LEDGER_FILES, TERMINAL_SCHEMA
from .coral_control import fixed_invocation_limits, recorded_restart_controls
from .cutoff import close_stage_at_cutoff
from .errors import InfrastructureError, MismatchError
from .lane_record import materialize_lane_recording
from .launcher import native_command, tmux_tmpdir
from .llm_store import FORCED_REPLAY_MODES
from .metrics import GPUSampler, RunMetrics, operation_summary, wait_until
from .services import ReplayServices
from .step3 import export_step3
from .timeline import build_timeline
from .util import (
    atomic_write_json,
    ensure_empty_directory,
    iter_jsonl,
    monotonic_ns,
    read_json,
    require,
    wall_time_ns,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# The framework launcher starts uninstrumented so the supervisor can attach the
# cgroup before any framework code runs. Only these names survive into it.
_BOOTSTRAP_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TERM",
    "SHELL",
    "TMPDIR",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
)


@dataclass
class RunResult:
    run_id: str
    root: Path
    metrics: dict[str, Any]
    terminal: dict[str, Any]
    timeline: dict[str, Any]


class Supervisor:
    def __init__(
        self,
        *,
        mode: str,
        config: RunConfig,
        output: Path,
        run_id: str,
        bundle: Bundle | None = None,
        replay_mode: str = "tool-only",
        fast_claim: bool = False,
    ) -> None:
        require(mode in {"record", "replay"}, f"invalid supervisor mode: {mode!r}")
        self.mode = mode
        self.config = config
        self.output = output.expanduser().resolve()
        self.run_id = run_id
        self.bundle = bundle
        self.replay_mode = replay_mode
        if mode == "replay" and replay_mode == "llm-only" and config.adapter != "mini-swe":
            # Simulating a tool means not entering its native implementation, which
            # every adapter has to opt into at its own execution point. Refuse the
            # mode rather than silently running a full replay under its name.
            raise InfrastructureError(
                f"--mode llm-only is implemented for mini-swe only, not {config.adapter!r}"
            )
        self.fast_claim = fast_claim
        self.force_secret, self.audit_path = self._forced_wiring()

        self.stage = self.output / "stage"
        self.logs = self.output / "logs"
        self.ready_dir = self.output / "actor-ready"
        self.lane_binding_dir = self.output / "lane-bindings"
        self.terminal_dir = self.output / "task-terminals"
        self.status_dir = self.output / "instrumentation"
        self.lane_event_dir = self.output / "lane-events"
        self.gate_path = self.output / "start.gate"
        self.phase_events = self.output / "sweep-phase-events.jsonl"
        self.tmux_dir = tmux_tmpdir(self.output, self.run_id)

    def _forced_wiring(self) -> tuple[str | None, Path | None]:
        """Resolve the shared secret and audit file, if serving is configured.

        Recording uses them for capture mode, so the bundle carries the engine step
        window a later forced replay needs. Absent, both modes still work: recording
        simply produces a bundle that `--mode full` will refuse.
        """

        if not self.config.serving:
            if self.mode == "replay" and self.replay_mode in FORCED_REPLAY_MODES:
                raise InfrastructureError(
                    f"--mode {self.replay_mode} needs config.serving; "
                    "start vLLM with `minireplay vllm-up`"
                )
            return None, None
        from .serving import ensure_secret

        spec = self.config.serving_spec()
        if not spec.audit_path_on_host.exists():
            if self.mode == "replay" and self.replay_mode in FORCED_REPLAY_MODES:
                raise InfrastructureError(
                    f"forced-decoding audit file is missing: {spec.audit_path_on_host}. "
                    "Start vLLM with `minireplay vllm-up`."
                )
            return None, None
        return ensure_secret(spec.secret_path), spec.audit_path_on_host

    # ---- layout --------------------------------------------------------------

    def _prepare(self) -> None:
        ensure_empty_directory(self.output)
        for directory in (
            self.stage,
            self.logs,
            self.ready_dir,
            self.lane_binding_dir,
            self.terminal_dir,
            self.status_dir,
            self.lane_event_dir,
            self.output / "runtime-identities",
            # Declared to the sweep via TMUX_TMPDIR and SWEEP_RESULTS_ROOT. tmux
            # will not create its socket directory, so this run owns creating both.
            self.tmux_dir,
            self.output / "sweep-results",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for relative in ("llm.jsonl", "spans.jsonl", *LEDGER_FILES.values()):
            (self.stage / relative).touch()

    def _remove_external_tmux_dir(self) -> None:
        """Drop the socket directory when it had to live outside the run root.

        Teardown may only touch what this run owns, and the fallback directory is
        named after the run, so removing it leaves no residue either way.
        """

        if self.output in self.tmux_dir.parents:
            return
        with suppress(OSError):
            shutil.rmtree(self.tmux_dir)

    # ---- environment ---------------------------------------------------------

    def _bootstrap_environment(self) -> dict[str, str]:
        environment = {key: os.environ[key] for key in _BOOTSTRAP_ENV_KEYS if key in os.environ}
        environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        return environment

    def _runtime_environment(self, services: ReplayServices) -> dict[str, str]:
        assert services.boundary_endpoint is not None
        assert services.llm_endpoint is not None
        native = native_command(
            self.config,
            self.output,
            run_id=self.run_id,
            skip_vllm=self.mode == "replay" and self.replay_mode == "tool-only",
            record=self.mode == "record",
            duration_s=self._native_duration_s(),
        )

        upstream_targets = {
            target: f"{url.rstrip('/')}/v1" for target, url in self.config.targets.items()
        }
        proxy_url = f"{services.llm_endpoint.url}/v1"
        first_target = next(iter(self.config.targets))

        environment = dict(os.environ)
        environment.update(self._bootstrap_environment())
        environment.update(native.environment)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join([str(PACKAGE_ROOT / "bootstrap"), str(PACKAGE_ROOT)]),
                "PYTHONUNBUFFERED": "1",
                "NATIVE_REPLAY_ADAPTER": self.config.adapter,
                "NATIVE_REPLAY_MODE": self.mode,
                "NATIVE_REPLAY_RUN_ID": self.run_id,
                "NATIVE_REPLAY_RUN_ROOT": str(self.output),
                "NATIVE_REPLAY_PACKAGE_ROOT": str(PACKAGE_ROOT),
                # The sweep is launched directly, so this role is inherited by every
                # process under it and must be the instrumented one. The sweep's own
                # Python helpers are excluded by path (scripts/lib/...) and by being
                # inline `python3 -` heredocs; see instrumentation.bootstrap.install.
                "NATIVE_REPLAY_PROCESS_ROLE": "framework",
                "NATIVE_REPLAY_BOUNDARY_URL": services.boundary_endpoint.url,
                "NATIVE_REPLAY_BOUNDARY_TOKEN": self.auth_token,
                "NATIVE_REPLAY_PROXY_URL": proxy_url,
                "NATIVE_REPLAY_LLM_PROXY_ORIGINS": json.dumps([services.llm_endpoint.url]),
                "NATIVE_REPLAY_UPSTREAM_TARGETS": json.dumps(upstream_targets, sort_keys=True),
                "NATIVE_REPLAY_TARGET_ID": first_target,
                "NATIVE_REPLAY_TARGET_MAP": json.dumps(self._target_map(), sort_keys=True),
                "NATIVE_REPLAY_ROLE_TARGETS": json.dumps(self._role_targets(), sort_keys=True),
                "NATIVE_REPLAY_START_GATE": str(self.gate_path),
                "NATIVE_REPLAY_READY_DIR": str(self.ready_dir),
                "NATIVE_REPLAY_LANE_BINDING_DIR": str(self.lane_binding_dir),
                "NATIVE_REPLAY_TERMINAL_DIR": str(self.terminal_dir),
                "NATIVE_REPLAY_INSTRUMENTATION_STATUS_DIR": str(self.status_dir),
                "NATIVE_REPLAY_LANE_EVENT_DIR": str(self.lane_event_dir),
                "NATIVE_REPLAY_RUNTIME_IDENTITY_DIR": str(self.output / "runtime-identities"),
                "NATIVE_REPLAY_OPENCODE_IDENTITY": "opencode-native-replay-plugin",
                "NATIVE_REPLAY_CONCURRENCY": str(self.config.concurrency),
                "NATIVE_REPLAY_SCOPE": f"C{self.config.concurrency}",
                "NATIVE_REPLAY_GATE_TIMEOUT_S": str(max(600, self.config.duration_s * 2)),
                "NATIVE_REPLAY_ACTOR_MAP": json.dumps(self._actor_map(), sort_keys=True),
                "NATIVE_REPLAY_TASK_MAP": json.dumps(self._task_map(), sort_keys=True),
                "NATIVE_REPLAY_ARRIVAL_OFFSETS": json.dumps({}),
                # The framework's own OpenAI client must reach the store, not vLLM.
                "OPENAI_BASE_URL": proxy_url,
                "OPENAI_API_BASE": proxy_url,
                "OPENAI_API_KEY": environment.get("OPENAI_API_KEY", "native-replay"),
            }
        )
        # Recording has no inventory to enforce: an actor names itself. Only set the
        # variable when there is a real inventory, because the gate treats any
        # non-empty value as a list to check membership against.
        inventory = self._actor_inventory()
        if inventory:
            environment["NATIVE_REPLAY_ACTORS"] = json.dumps(inventory, sort_keys=True)
        if self.config.cpuset:
            environment["NATIVE_REPLAY_DOCKER_CPUSET"] = self.config.cpuset
        if self.bundle is not None:
            bindings = self.bundle.manifest.get("identity_bindings")
            if bindings:
                environment["NATIVE_REPLAY_SOURCE_IDENTITY_BINDINGS"] = json.dumps(
                    bindings, sort_keys=True
                )
            if self.config.adapter == "coral":
                limits = fixed_invocation_limits(
                    self.bundle.llm,
                    self.bundle.cutoff_tails,
                )
                controls = list(self.bundle.manifest.get("coral_controls", []))
                expected_controls = {
                    (actor_id, invocation_index)
                    for actor_id, limit in limits.items()
                    for invocation_index in range(1, limit)
                }
                observed_controls = {
                    (str(control.get("actor_id")), int(control.get("invocation_index", -1)))
                    for control in controls
                }
                if expected_controls != observed_controls:
                    raise InfrastructureError(
                        "CORAL replay bundle has incomplete recorded restart controls: "
                        f"expected={sorted(expected_controls)!r} "
                        f"observed={sorted(observed_controls)!r}; re-record the source"
                    )
                environment["NATIVE_REPLAY_CORAL_INVOCATION_LIMITS"] = json.dumps(
                    limits,
                    sort_keys=True,
                )
                environment["NATIVE_REPLAY_CORAL_CONTROLS"] = json.dumps(
                    controls,
                    sort_keys=True,
                )
        # The config may legitimately need to extend PYTHONPATH (a framework
        # installed outside site-packages, say). It must never replace it: the
        # instrumentation entries carry sitecustomize, and losing them would
        # silently produce an uninstrumented run rather than a failed one.
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(PACKAGE_ROOT / "bootstrap"),
                str(PACKAGE_ROOT),
                *(
                    entry
                    for entry in self.config.env.get("PYTHONPATH", "").split(os.pathsep)
                    if entry
                ),
            ]
        )
        return {str(k): str(v) for k, v in environment.items()}

    def _target_map(self) -> dict[str, str]:
        return {}

    def _role_targets(self) -> dict[str, str]:
        first = next(iter(self.config.targets))
        return {
            role: role if role in self.config.targets else first for role in self.config.targets
        }

    def _actor_map(self) -> dict[str, str]:
        """Recording lets actors name themselves; replay pins the recorded names."""

        return self.bundle.actor_map() if self.bundle is not None else {}

    def _actor_inventory(self) -> list[str]:
        return self.bundle.actor_ids() if self.bundle is not None else []

    def _task_map(self) -> dict[str, Any]:
        if self.bundle is None:
            return {}
        return {
            str(actor["actor_id"]): actor.get("task", {"actor": actor["actor_id"]})
            for actor in self.bundle.manifest["actors"]
        }

    # ---- native launch -------------------------------------------------------

    def _boundary_timeout_s(self) -> float:
        return self.config.duration_s * 4 + 900

    def _native_duration_s(self) -> int:
        """Keep replay's native driver alive beyond the harness watchdog.

        Recording is bounded by the requested source window. Replay is bounded by
        ledger completion, so the native sweep's own ``sample_end`` must be kept
        out of the way. The supervisor watchdog remains the infrastructure escape
        hatch if fixed work can never complete.
        """

        if self.mode == "record":
            return self.config.duration_s
        return int(self._boundary_timeout_s() + max(600, self.config.duration_s))

    def _launch(self, environment: dict[str, str], cgroup: RoleCgroup) -> subprocess.Popen:
        native = native_command(
            self.config,
            self.output,
            run_id=self.run_id,
            skip_vllm=self.mode == "replay" and self.replay_mode == "tool-only",
            record=self.mode == "record",
            duration_s=self._native_duration_s(),
        )
        environment_file = self.output / "exec-environment.json"
        atomic_write_json(environment_file, environment)

        log = (self.logs / "native.log").open("wb")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "minireplay.exec_gate",
                "--env-file",
                str(environment_file),
                "--",
                *native.command,
            ],
            cwd=str(native.cwd),
            env=self._bootstrap_environment() | {"PYTHONPATH": str(PACKAGE_ROOT)},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_stopped(process.pid)
        cgroup.add_pid(process.pid)
        os.kill(process.pid, signal.SIGCONT)
        return process

    def _wait_stopped(self, pid: int, *, timeout_s: float = 60.0) -> None:
        """Wait for exec_gate to stop itself, so nothing runs before accounting starts."""

        deadline = time.monotonic() + timeout_s
        state_path = Path(f"/proc/{pid}/stat")
        while time.monotonic() < deadline:
            try:
                fields = state_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            except (OSError, IndexError) as exc:
                raise InfrastructureError(
                    "native launcher exited before it could be attached"
                ) from exc
            if fields and fields[0] == "T":
                return
            time.sleep(0.005)
        raise InfrastructureError("native launcher did not stop for cgroup attachment")

    # ---- gate ----------------------------------------------------------------

    def _wait_actors_ready(
        self,
        process: subprocess.Popen,
        *,
        timeout_s: float,
    ) -> list[dict[str, Any]]:
        """Wait until as many actors have reached the gate as the run declares.

        Waiting on a count rather than on an inventory is what lets recording run
        without resolving the task list first: an actor announces itself, and the
        run starts once the declared concurrency has arrived.
        """

        expected = self.config.concurrency
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready = [read_json(path) for path in sorted(self.ready_dir.glob("*.json"))]
            if len(ready) >= expected:
                return ready
            if process.poll() is not None:
                raise InfrastructureError(
                    f"the native sweep exited with code {process.returncode} before "
                    f"{expected} actors reached the gate "
                    f"(see {self.logs / 'native.log'})"
                )
            time.sleep(0.02)
        found = sorted(path.stem for path in self.ready_dir.glob("*.json"))
        raise InfrastructureError(
            f"only {len(found)} of {expected} actors reached the gate: {found}"
        )

    def _open_gate(self) -> int:
        opened = monotonic_ns()
        atomic_write_json(
            self.gate_path,
            {
                "schema_version": "minireplay.start-gate/v1",
                "run_id": self.run_id,
                "opened_at_ns": opened,
                "opened_at_epoch_ns": wall_time_ns(),
            },
            mode=0o644,
        )
        return opened

    # ---- boundary ------------------------------------------------------------

    def _phase_event_ns(self, event: str) -> int | None:
        """Translate a sweep phase event's wall clock into the monotonic domain."""

        if not self.phase_events.is_file():
            return None
        for record in iter_jsonl(self.phase_events):
            if record.get("event") != event:
                continue
            epoch = record.get("ts_epoch")
            if not isinstance(epoch, (int, float)):
                continue
            now_wall = wall_time_ns()
            now_mono = monotonic_ns()
            return now_mono - (now_wall - int(float(epoch) * 1e9))
        return None

    def _wait_boundary(
        self,
        services: ReplayServices,
        process: subprocess.Popen,
        *,
        timeout_s: float,
    ) -> tuple[int, str]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            services.assert_healthy()
            if self.mode == "replay" and services.expected_complete():
                return monotonic_ns(), "fixed-work-complete"
            sample_end = self._phase_event_ns("sample_end")
            if sample_end is not None:
                if self.mode == "record":
                    return sample_end, "sweep-sample-end"
                raise MismatchError(
                    "the native sweep reached sample_end before the recorded work "
                    f"completed: {json.dumps(services.status(), sort_keys=True)}"
                )
            if process.poll() is not None:
                if self.mode == "replay" and services.expected_complete():
                    return monotonic_ns(), "fixed-work-complete"
                raise InfrastructureError(
                    f"the native sweep exited with code {process.returncode} before its "
                    "sample boundary"
                )
            time.sleep(0.05)
        raise InfrastructureError("timed out waiting for the run boundary")

    # ---- teardown ------------------------------------------------------------

    def _stop_native(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)

    def _collect_terminals(self) -> list[dict[str, Any]]:
        return [read_json(path) for path in sorted(self.terminal_dir.glob("*.json"))]

    def _instrumentation_report(self) -> dict[str, Any]:
        processes = [read_json(path) for path in sorted(self.status_dir.glob("*.json"))]
        failures = [
            {"pid": entry.get("pid"), "failures": entry["failures"]}
            for entry in processes
            if entry.get("failures")
        ]
        return {"processes": len(processes), "failures": failures}

    def _stage_records(self) -> dict[str, list[dict[str, Any]]]:
        records = {
            kind: list(iter_jsonl(self.stage / relative)) for kind, relative in LEDGER_FILES.items()
        }
        records["llm"] = list(iter_jsonl(self.stage / "llm.jsonl"))
        return records

    # ---- main loop -----------------------------------------------------------

    def run(self) -> RunResult:
        self._prepare()
        self.auth_token = secrets.token_urlsafe(32)
        cgroup = RoleCgroup(self.run_id, "framework")
        services = ReplayServices(
            mode=self.mode,
            stage_dir=self.stage,
            auth_token=self.auth_token,
            adapter=self.config.adapter,
            upstreams=self.config.targets,
            run_root=self.output,
            repo=self.config.repo,
            bundle=self.bundle,
            replay_mode=self.replay_mode,
            fast_claim=self.fast_claim,
            force_secret=self.force_secret,
            audit_path=self.audit_path,
            audit_namespace=self.run_id,
        )
        metrics = RunMetrics(gpu=GPUSampler(self.config.gpu_ids), cgroup=cgroup)
        usage = None
        usage_path = self.config.env.get("PIN_CPU_USAGE_PATH")
        if usage_path:
            from pin_cpu.usage import UsageCollector

            usage = UsageCollector(
                run_id=self.run_id,
                output=Path(usage_path),
                include_vllm=not (
                    self.mode == "replay" and self.replay_mode == "tool-only"
                ),
                policy_path=Path(self.config.env["PIN_CPU_POLICY_PATH"]),
                sample_interval_ms=(
                    int(self.config.env["PIN_CPU_SAMPLE_INTERVAL_MS"])
                    if self.config.env.get("PIN_CPU_SAMPLE_INTERVAL_MS")
                    else None
                ),
            )
        process: subprocess.Popen | None = None
        failure: BaseException | None = None
        reason = "unknown"
        cutoff_tails: dict[str, Any] = {"operations": [], "llm_requests": []}

        try:
            cgroup.create()
            services.start()
            environment = self._runtime_environment(services)
            process = self._launch(environment, cgroup)

            self._wait_actors_ready(process, timeout_s=max(600.0, self.config.duration_s))
            services.assert_healthy()

            if usage is not None:
                usage.mark_before_gate()
            gate_at_ns = self._open_gate()
            metrics.mark_gate(gate_at_ns)

            terminal_at_ns, reason = self._wait_boundary(
                services,
                process,
                timeout_s=self._boundary_timeout_s(),
            )

            cutoff_tails = services.freeze_source_cutoff(terminal_at_ns)
            # Handlers that already returned may still be appending to the ledgers.
            # Let them drain before anything reads those files.
            wait_until(lambda: services.boundary.active_writes == 0, timeout_s=5.0)
            metrics.mark_terminal(terminal_at_ns)
            if usage is not None:
                usage.mark_terminal(
                    gate_at_ns=gate_at_ns,
                    terminal_at_ns=terminal_at_ns,
                )
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            failure = exc
            if metrics.terminal_at_ns == 0:
                metrics.mark_terminal(monotonic_ns())
        finally:
            if usage is not None:
                usage.close()
            if process is not None:
                self._stop_native(process)
            cgroup.kill()
            services.stop()
            self._remove_external_tmux_dir()

        terminals = self._collect_terminals()
        observed_records: dict[str, list[dict[str, Any]]] | None = None
        if self.mode == "record" and failure is None:
            try:
                cutoff_tails["operations"] = materialize_lane_recording(
                    event_dir=self.lane_event_dir,
                    stage_dir=self.stage,
                    cutoff_at_ns=metrics.terminal_at_ns,
                    auth_token=self.auth_token,
                    adapter=self.config.adapter,
                    run_root=self.output,
                    repo=self.config.repo,
                )
                cutoff_tails = materialize_pre_dispatch_tails(
                    adapter=self.config.adapter,
                    llm=list(iter_jsonl(self.stage / "llm.jsonl")),
                    dispatches=list(iter_jsonl(self.stage / LEDGER_FILES["dispatch"])),
                    cutoff_tails=cutoff_tails,
                )
                gate = read_json(self.gate_path)
                cutoff_tails = _apply_coral_task_cutoffs(
                    adapter=self.config.adapter,
                    cutoff_tails=cutoff_tails,
                    task_terminals=terminals,
                    actors=_actors_from_stage(
                        self.stage,
                        self.ready_dir,
                        cutoff_tails,
                        self.lane_binding_dir,
                    ),
                    gate_at_ns=metrics.gate_at_ns,
                    gate_at_epoch_ns=int(gate["opened_at_epoch_ns"]),
                )
                # Step3 describes what was observed inside the source window.
                # Preserve that view before causal closure removes completed
                # descendants of an operation that remained open at cutoff.
                observed_records = self._stage_records()
            except BaseException as exc:  # noqa: BLE001 - becomes the run failure
                failure = exc

        atomic_write_json(self.output / "cutoff-tails.json", cutoff_tails)
        atomic_write_json(self.output / "llm-models.json", services.llm.model_catalogue)

        if self.mode == "record" and failure is None:
            close_stage_at_cutoff(self.stage, cutoff_tails=cutoff_tails)

        records = self._stage_records()
        if failure is None:
            failed_tasks = _unexpected_failed_tasks(
                mode=self.mode,
                reason=reason,
                bundle=self.bundle,
                task_terminals=terminals,
            )
            if failed_tasks:
                failure = InfrastructureError(
                    "native task terminal reported failure for " + ", ".join(sorted(failed_tasks))
                )
        terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "failure" if failure is not None else "success",
            "reason": reason,
            "task_terminals": terminals,
            "instrumentation": self._instrumentation_report(),
        }
        atomic_write_json(self.output / "terminal.json", terminal)

        timeline = build_timeline(
            records=records,
            gate_at_ns=metrics.gate_at_ns,
            terminal_at_ns=metrics.terminal_at_ns,
        )
        atomic_write_json(self.output / "timeline.json", timeline)

        if self.mode == "record" and failure is None:
            gate = read_json(self.gate_path)
            step3_actors = _actors_from_stage(
                self.stage,
                self.ready_dir,
                cutoff_tails,
                self.lane_binding_dir,
            )
            export_step3(
                output=self.output,
                run_id=self.run_id,
                framework=self.config.framework,
                records=observed_records or records,
                cutoff_tails=cutoff_tails,
                scope_event_dir=self.lane_event_dir,
                gate_at_ns=metrics.gate_at_ns,
                gate_at_epoch_ns=int(gate["opened_at_epoch_ns"]),
                terminal_at_ns=metrics.terminal_at_ns,
                actor_lanes={
                    str(actor["actor_id"]): actor["lane"]
                    for actor in step3_actors
                    if isinstance(actor.get("lane"), dict)
                },
                task_terminals=terminals,
            )

        summary = metrics.summary(
            run_id=self.run_id,
            bundle_id=(self.bundle.manifest["bundle_id"] if self.bundle else self.run_id),
            mode=self.mode,
            replay_mode=self.replay_mode if self.mode == "replay" else "n/a",
            operations=operation_summary(records),
            busy_span_seconds=timeline["busy_span_s"],
        )
        atomic_write_json(self.output / "metrics.json", summary)

        cgroup.cleanup(best_effort=True)

        if failure is not None:
            raise failure

        try:
            services.assert_consumed() if self.mode == "replay" else None
        except MismatchError as exc:
            atomic_write_json(
                self.output / "verdict.json",
                {"valid": False, "mode": self.mode, "reason": str(exc)},
            )
            raise
        atomic_write_json(
            self.output / "verdict.json",
            {"valid": True, "mode": self.mode, "reason": reason},
        )
        return RunResult(
            run_id=self.run_id,
            root=self.output,
            metrics=summary,
            terminal=terminal,
            timeline=timeline,
        )


def _unexpected_failed_tasks(
    *,
    mode: str,
    reason: str,
    bundle: Bundle | None,
    task_terminals: list[dict[str, Any]],
) -> list[str]:
    """Reject native failures except an expected cutoff-open task timeout."""

    cutoff_sources = (
        bundle.cutoff_source_actor_ids()
        if mode == "replay" and reason == "fixed-work-complete" and bundle is not None
        else set()
    )
    actor_map = bundle.actor_map() if bundle is not None else {}
    failed: list[str] = []
    for terminal in task_terminals:
        if terminal.get("status") != "failure":
            continue
        actor_id = str(terminal.get("actor_id", "unknown"))
        task = terminal.get("task")
        source = task.get("source_actor_id") if isinstance(task, dict) else None
        result = terminal.get("result")
        error_type = result.get("error_type") if isinstance(result, dict) else None
        expected_cutoff = (
            isinstance(source, str)
            and source in cutoff_sources
            and actor_map.get(source) == actor_id
            and error_type == "TimeoutError"
        )
        if not expected_cutoff:
            failed.append(actor_id)
    return failed


def _actors_from_stage(
    stage: Path,
    ready_dir: Path,
    cutoff_tails: dict[str, Any] | None = None,
    lane_binding_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Inventory the actors a recording actually saw."""

    by_actor: dict[str, dict[str, Any]] = {}

    def declare(
        actor: str,
        *,
        source: Any = None,
        process_role: Any = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        entry = by_actor.setdefault(
            actor,
            {
                "actor_id": actor,
                "source_actor_id": None,
                "source_actor_ids": [],
                "process_role": process_role,
                "task": {"dynamic_actor": actor},
            },
        )
        if isinstance(source, str) and source:
            aliases = entry["source_actor_ids"]
            if source not in aliases:
                aliases.append(source)
            if entry.get("source_actor_id") is None:
                entry["source_actor_id"] = source
                entry["task"] = {"source_actor_id": source}
        if entry.get("process_role") is None and isinstance(process_role, str):
            entry["process_role"] = process_role
        if metadata is not None:
            require(isinstance(metadata, dict), f"actor {actor!r} metadata is not an object")
            previous = entry.get("lane")
            require(
                previous is None or previous == metadata,
                f"actor {actor!r} declared conflicting lane metadata",
            )
            entry["lane"] = metadata
        return entry

    for path in sorted(ready_dir.glob("*.json")):
        ready = read_json(path)
        actor = str(ready["actor_id"])
        declare(
            actor,
            source=ready.get("source_actor_id"),
            process_role=ready.get("process_role"),
            metadata=ready.get("actor_metadata"),
        )

    if lane_binding_dir is not None and lane_binding_dir.is_dir():
        for path in sorted(lane_binding_dir.glob("*.json")):
            binding = read_json(path)
            actor = str(binding.get("actor_id", ""))
            require(bool(actor), f"native lane binding has no actor: {path}")
            declare(
                actor,
                source=binding.get("source_actor_id"),
                process_role=binding.get("process_role"),
                metadata=binding.get("actor_metadata"),
            )

    def observe(record: dict[str, Any]) -> None:
        actor = str(record.get("actor_id", ""))
        if actor:
            # Sweeps refill work into actors that never sat at the gate. An actor
            # first seen immediately before cutoff may exist only in a tail.
            declare(actor, process_role=record.get("process_role"))

    for relative in ("llm.jsonl", *LEDGER_FILES.values()):
        path = stage / relative
        if not path.is_file():
            continue
        for record in iter_jsonl(path):
            observe(record)
    for key in ("llm_requests", "operations"):
        for record in (cutoff_tails or {}).get(key, []):
            observe(record)
    return [by_actor[key] for key in sorted(by_actor)]


def _apply_coral_task_cutoffs(
    *,
    adapter: str,
    cutoff_tails: dict[str, Any],
    task_terminals: list[dict[str, Any]],
    actors: list[dict[str, Any]],
    gate_at_ns: int,
    gate_at_epoch_ns: int,
) -> dict[str, Any]:
    """Clamp killed CORAL agent work to the manager's real team cutoff.

    The sweep source window may continue after ``max_total_turns`` kills a team.
    That later sample boundary must not make an interrupted LLM or OpenCode tool
    look active through the idle remainder of the 180-second window.
    """

    rendered = {
        "operations": [dict(record) for record in cutoff_tails.get("operations", [])],
        "llm_requests": [dict(record) for record in cutoff_tails.get("llm_requests", [])],
    }
    if adapter != "coral":
        return rendered

    children: dict[str, set[str]] = {}
    for actor in actors:
        actor_id = actor.get("actor_id")
        lane = actor.get("lane")
        if not isinstance(actor_id, str) or not isinstance(lane, dict):
            continue
        parent = lane.get("parent_actor_id")
        if lane.get("lane_kind") == "agent" and isinstance(parent, str):
            children.setdefault(parent, set()).add(actor_id)

    actor_cutoffs: dict[str, tuple[int, int, str]] = {}
    for terminal in task_terminals:
        if not isinstance(terminal, dict):
            continue
        team_actor = terminal.get("actor_id")
        result = terminal.get("result")
        if not isinstance(team_actor, str) or not isinstance(result, dict):
            continue
        cutoff_epoch = result.get("replay_cutoff_at_epoch_ns")
        reason = result.get("termination_reason")
        if not isinstance(cutoff_epoch, int) or not isinstance(reason, str):
            continue
        cutoff_monotonic = gate_at_ns + (cutoff_epoch - gate_at_epoch_ns)
        for actor_id in children.get(team_actor, set()):
            actor_cutoffs[actor_id] = (cutoff_monotonic, cutoff_epoch, reason)

    def clamp(record: dict[str, Any]) -> None:
        cutoff = actor_cutoffs.get(str(record.get("actor_id")))
        if cutoff is None:
            return
        cutoff_monotonic, cutoff_epoch, reason = cutoff
        started = record.get("started_at_ns", record.get("source_started_at_ns"))
        if not isinstance(started, int):
            return
        current_end = started + int(record.get("elapsed_ns", 0))
        if (
            record.get("replay_entry") == "enter-and-preserve-descendants"
            and record.get("name") == "task"
        ):
            # A synthesized CORAL task boundary is proven active by its child
            # sessions even when OpenCode omitted tool.execute.before. Its real
            # end is the manager's team cutoff, not the zero-duration point at
            # which the parent LLM emitted the task call.
            current_end = cutoff_monotonic
        observed_end = record.get("interrupted_at_ns")
        if isinstance(observed_end, int):
            current_end = min(current_end, observed_end)
        ended = max(started, min(current_end, cutoff_monotonic))
        record["elapsed_ns"] = ended - started
        record["lane_terminated_at_ns"] = cutoff_monotonic
        record["lane_terminated_at_epoch_ns"] = cutoff_epoch
        record["lane_termination_reason"] = reason

    for record in rendered["llm_requests"]:
        clamp(record)
    for record in rendered["operations"]:
        if record.get("process_role") == "coral-opencode":
            clamp(record)
    return rendered


def _identity_bindings(directory: Path) -> dict[str, Any]:
    """Collect what each actor's framework-internal runtime identities were.

    An adapter writes one file per actor while recording. They have to survive into
    the bundle, because replay binds a recorded identity to the live one; without
    them the adapter fails per task, the framework treats that as a finished task
    and refills, and the run drifts for a reason that looks nothing like its cause.
    """

    if not directory.is_dir():
        return {}
    bindings: dict[str, Any] = {}
    for path in sorted(directory.glob("*.json")):
        record = read_json(path)
        actor = record.get("actor_id")
        require(isinstance(actor, str) and bool(actor), f"runtime identity has no actor: {path}")
        bindings[actor] = record
    return bindings


def record_bundle(
    *,
    config: RunConfig,
    output: Path,
    bundle_output: Path,
    run_id: str | None = None,
) -> Bundle:
    identity = config.workload_identity()
    bundle_id = bundle_id_for(identity)
    run = Supervisor(
        mode="record",
        config=config,
        output=output,
        # Bundle identity is intentionally stable; native sweep output ownership
        # is not. A unique suffix prevents a retry of the same workload from
        # reopening the previous framework results directory.
        run_id=run_id or f"record-{bundle_id}-{secrets.token_hex(4)}",
    ).run()

    cutoff_tails = read_json(run.root / "cutoff-tails.json")
    actors = _actors_from_stage(
        run.root / "stage",
        run.root / "actor-ready",
        cutoff_tails,
        run.root / "lane-bindings",
    )
    require(bool(actors), "the recording observed no actors")
    llm = list(iter_jsonl(run.root / "stage" / "llm.jsonl"))
    graders = list(iter_jsonl(run.root / "stage" / "graders.jsonl"))
    invocation_limits = fixed_invocation_limits(llm, cutoff_tails)
    coral_controls = (
        recorded_restart_controls(
            actors=actors,
            task_terminals=list(run.terminal.get("task_terminals", [])),
            graders=graders,
            invocation_limits=invocation_limits,
        )
        if config.adapter == "coral"
        else []
    )
    return build_bundle(
        stage_dir=run.root / "stage",
        output=bundle_output,
        bundle_id=bundle_id,
        adapter=config.adapter,
        workload=identity,
        actors=actors,
        window={
            "gate_at_ns": run.metrics["gate_at_ns"],
            "terminal_at_ns": run.metrics["terminal_at_ns"],
        },
        terminal=run.terminal,
        cutoff_tails=cutoff_tails,
        llm_models=read_json(run.root / "llm-models.json"),
        identity_bindings=_identity_bindings(run.root / "runtime-identities"),
        coral_controls=coral_controls,
    )


def replay_bundle(
    *,
    config: RunConfig,
    bundle_dir: Path,
    output: Path,
    replay_mode: str = "tool-only",
    fast_claim: bool = False,
    run_id: str | None = None,
) -> RunResult:
    bundle = load_bundle(bundle_dir)
    recorded = bundle.manifest["workload"]
    current = config.workload_identity()
    if recorded != current:
        raise MismatchError(
            f"config does not match the bundle's workload: bundle={recorded} config={current}"
        )
    return Supervisor(
        mode="replay",
        config=config,
        output=output,
        run_id=run_id or f"replay-{secrets.token_hex(4)}",
        bundle=bundle,
        replay_mode=replay_mode,
        fast_claim=fast_claim,
    ).run()
