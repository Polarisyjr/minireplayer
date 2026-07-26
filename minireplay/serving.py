"""Bring up a vLLM fleet that forced decoding can actually use.

Serving topology — GPU allocation, CDI-vs-nvidia runtime, cpuset pinning, page-cache
warmup — lives in `multiagent/serving/scripts/start_vllm_multi.sh` and stays there.
This module only adds the three things forced decoding needs on top:

* the patched image, whose sampler commits recorded tokens after sampling;
* a secret shared with the proxy, so the engine can verify what it is asked to force;
* a writable directory for the plugin's audit file, which the proxy reads back to
  confirm the engine really did force what it was told to.

Reimplementing the launcher here would duplicate several hundred lines of knowledge
that is easy to get subtly wrong (this host, for instance, needs the nvidia runtime
because its Docker daemon has no `features.cdi`).
"""

from __future__ import annotations

import os
import secrets as secrets_module
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import InfrastructureError
from .util import require

DEFAULT_IMAGE = "minireplay-vllm:v0.19.0"
AUDIT_MOUNT = "/native-replay-audit"


@dataclass(frozen=True)
class ServingSpec:
    repo: Path
    configs: list[str]
    image: str
    secret_path: Path
    audit_dir: Path
    gpu_mode: str = "nvidia"

    @property
    def audit_path_in_container(self) -> str:
        return f"{AUDIT_MOUNT}/forced-audit.jsonl"

    @property
    def audit_path_on_host(self) -> Path:
        return self.audit_dir / "forced-audit.jsonl"


def ensure_secret(path: Path) -> str:
    """Read the shared forced-decoding secret, creating it on first use."""

    path = path.expanduser()
    if path.is_file():
        secret = path.read_text(encoding="utf-8").strip()
        require(bool(secret), f"forced-decoding secret is empty: {path}")
        return secret
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets_module.token_urlsafe(32)
    path.write_text(secret, encoding="utf-8")
    path.chmod(0o600)
    return secret


def start_command(spec: ServingSpec, secret: str) -> tuple[list[str], dict[str, str]]:
    script = spec.repo / "serving" / "scripts" / "start_vllm_multi.sh"
    require(script.is_file(), f"serving launcher not found: {script}")
    environment = dict(os.environ)
    environment.update(
        {
            "VLLM_IMAGE": spec.image,
            # The plugin is constructed for every request and hard-requires both,
            # so a missing one keeps the engine from starting at all.
            "VLLM_EXTRA_ENV": (
                f"NATIVE_REPLAY_FORCE_SECRET={secret} "
                f"NATIVE_REPLAY_FORCE_AUDIT={spec.audit_path_in_container}"
            ),
            "VLLM_EXTRA_MOUNTS": f"{spec.audit_dir.resolve()}:{AUDIT_MOUNT}",
        }
    )
    environment["VLLM_GPU_MODE"] = spec.gpu_mode
    return ["bash", str(script), *spec.configs], environment


def start(spec: ServingSpec) -> str:
    secret = ensure_secret(spec.secret_path)
    spec.audit_dir.mkdir(parents=True, exist_ok=True)
    # The plugin appends; the file must exist and be writable by the container.
    spec.audit_path_on_host.touch(exist_ok=True)
    spec.audit_path_on_host.chmod(0o666)

    command, environment = start_command(spec, secret)
    result = subprocess.run(
        command,
        cwd=str(spec.repo),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise InfrastructureError(
            f"vLLM fleet did not start ({result.returncode}):\n{result.stdout}{result.stderr}"
        )
    return secret


def stop(repo: Path) -> None:
    script = repo / "serving" / "scripts" / "stop_vllm_multi.sh"
    require(script.is_file(), f"serving teardown not found: {script}")
    subprocess.run(["bash", str(script)], cwd=str(repo), check=False, capture_output=True)


def assert_forced_capable(container_name: str) -> None:
    """Refuse to run forced decoding against an engine that cannot force anything.

    Without this a `--mode full` run would look successful while every token came
    from ordinary sampling, which is exactly the silent failure this tool exists to
    prevent.
    """

    checks = {
        "sampler patch": (
            "grep -c 'native-agent-replay post-sampling commitment' "
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/sampler.py"
        ),
        "model runner patch": (
            "grep -c 'native-agent-replay valid-sample mask' "
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
        ),
        "forced secret": 'test -n "$NATIVE_REPLAY_FORCE_SECRET" && echo 1',
        "audit path variable": 'test -n "$NATIVE_REPLAY_FORCE_AUDIT" && echo 1',
        "audit mount": f"test -w {AUDIT_MOUNT}/forced-audit.jsonl && echo 1",
    }
    failures = []
    for name, script in checks.items():
        result = subprocess.run(
            ["docker", "exec", container_name, "sh", "-lc", script],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() not in {"1"}:
            failures.append(name)
    if failures:
        raise InfrastructureError(
            f"vLLM container {container_name!r} is not forced-decoding capable: "
            f"missing {failures}. Start it with `minireplay vllm-up`."
        )


def running_vllm_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--filter", "label=vllm-serving=1", "--format", "{{.Names}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [name for name in result.stdout.split() if name]


def wait_serving_ready(
    targets: list[str],
    *,
    timeout_s: float = 600.0,
    poll_s: float = 2.0,
) -> None:
    """Wait for every configured model API, not merely its listening socket."""

    pending = {target.rstrip("/") for target in targets}
    deadline = time.monotonic() + timeout_s
    last_errors: dict[str, str] = {}
    while pending:
        for target in tuple(pending):
            try:
                with urllib.request.urlopen(f"{target}/v1/models", timeout=3) as response:
                    if response.status == 200:
                        pending.remove(target)
                        last_errors.pop(target, None)
                    else:
                        last_errors[target] = f"HTTP {response.status}"
            except (OSError, urllib.error.URLError) as exc:
                last_errors[target] = str(exc)
        if not pending:
            return
        if time.monotonic() >= deadline:
            details = ", ".join(
                f"{target}: {last_errors.get(target, 'not ready')}"
                for target in sorted(pending)
            )
            raise InfrastructureError(
                f"vLLM fleet did not become API-ready within {timeout_s:g}s ({details})"
            )
        time.sleep(poll_s)
