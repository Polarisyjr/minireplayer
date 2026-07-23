from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InfrastructureError, ValidationError

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_V2_ROOT = Path("/sys/fs/cgroup/unified")
_V1_ROOTS = {
    "cpuacct": Path("/sys/fs/cgroup/cpu,cpuacct"),
    "blkio": Path("/sys/fs/cgroup/blkio"),
    "memory": Path("/sys/fs/cgroup/memory"),
    "pids": Path("/sys/fs/cgroup/pids"),
}


def _safe_name(value: str, context: str) -> str:
    if _SAFE_NAME.fullmatch(value) is None:
        raise ValidationError(f"{context}: invalid cgroup name {value!r}")
    return value


def _sudo(*command: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["sudo", "-n", *command],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "")
        raise InfrastructureError(f"cgroup command failed: {' '.join(command)}: {stderr}") from exc


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError) as exc:
        raise InfrastructureError(f"cannot read cgroup counter {path}: {exc}") from exc


def _read_cpu_stat(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            name, raw = line.split(None, 1)
            values[name] = int(raw)
    except (OSError, ValueError) as exc:
        raise InfrastructureError(f"cannot read cgroup cpu.stat {path}: {exc}") from exc
    return values


def _read_blkio(path: Path) -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    if not path.exists():
        return read_bytes, write_bytes
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            operation = parts[1].lower()
            if operation == "read":
                read_bytes += int(parts[2])
            elif operation == "write":
                write_bytes += int(parts[2])
    except (OSError, ValueError) as exc:
        raise InfrastructureError(f"cannot read blkio accounting {path}: {exc}") from exc
    return read_bytes, write_bytes


@dataclass(frozen=True)
class CgroupCounters:
    cpu_seconds: float
    user_seconds: float
    system_seconds: float
    read_bytes: int
    write_bytes: int
    memory_peak_bytes: int
    live_pids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_seconds": self.cpu_seconds,
            "user_seconds": self.user_seconds,
            "system_seconds": self.system_seconds,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "memory_peak_bytes": self.memory_peak_bytes,
            "live_pids": list(self.live_pids),
        }

    def delta(self, baseline: CgroupCounters) -> CgroupCounters:
        return CgroupCounters(
            cpu_seconds=max(0.0, self.cpu_seconds - baseline.cpu_seconds),
            user_seconds=max(0.0, self.user_seconds - baseline.user_seconds),
            system_seconds=max(0.0, self.system_seconds - baseline.system_seconds),
            read_bytes=max(0, self.read_bytes - baseline.read_bytes),
            write_bytes=max(0, self.write_bytes - baseline.write_bytes),
            memory_peak_bytes=max(self.memory_peak_bytes, baseline.memory_peak_bytes),
            live_pids=self.live_pids,
        )


def counters_from_paths(
    *,
    v2_path: Path,
    v1_paths: dict[str, Path],
    live_pids: tuple[int, ...] = (),
) -> CgroupCounters:
    v2_cpu = _read_cpu_stat(v2_path / "cpu.stat")
    cpu_seconds = float(v2_cpu.get("usage_usec", 0)) / 1e6
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cpu_stat = _read_cpu_stat(v1_paths["cpuacct"] / "cpuacct.stat")
    user_seconds = float(cpu_stat.get("user", 0)) / ticks
    system_seconds = float(cpu_stat.get("system", 0)) / ticks
    blkio_root = v1_paths["blkio"]
    blkio_file = blkio_root / "blkio.throttle.io_service_bytes"
    if not blkio_file.exists():
        blkio_file = blkio_root / "blkio.io_service_bytes"
    read_bytes, write_bytes = _read_blkio(blkio_file)
    memory_path = v1_paths["memory"] / "memory.max_usage_in_bytes"
    memory_peak = _read_int(memory_path) if memory_path.exists() else 0
    return CgroupCounters(
        cpu_seconds=cpu_seconds,
        user_seconds=user_seconds,
        system_seconds=system_seconds,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        memory_peak_bytes=memory_peak,
        live_pids=live_pids,
    )


class RoleCgroup:
    """Hybrid-cgroup accounting for one native process role.

    The unified hierarchy supplies an exited-descendant-safe CPU total and a
    reliable kill boundary. The v1 controller hierarchies supply user/system,
    block-I/O and memory counters on hosts that still use hybrid cgroups.
    """

    def __init__(self, run_id: str, role: str) -> None:
        self.run_id = _safe_name(run_id, "run_id")
        self.role = _safe_name(role, "role")
        relative = Path("native-replay") / self.run_id / self.role
        self.v2_path = _V2_ROOT / relative
        self.v1_paths = {name: root / relative for name, root in _V1_ROOTS.items()}
        self.created = False

    def create(self) -> None:
        if self.created:
            raise InfrastructureError(f"cgroup already created: {self.role}")
        paths = [self.v2_path, *self.v1_paths.values()]
        uid = str(os.getuid())
        gid = str(os.getgid())
        try:
            for path in paths:
                _sudo("mkdir", "-p", str(path))
                membership = path / ("cgroup.procs" if path == self.v2_path else "tasks")
                _sudo("chown", f"{uid}:{gid}", str(membership))
        except Exception:
            self.cleanup(best_effort=True)
            raise
        self.created = True

    def add_pid(self, pid: int) -> None:
        if not self.created:
            raise InfrastructureError("cannot add a PID before cgroup creation")
        if pid <= 0:
            raise ValidationError(f"invalid PID: {pid}")
        for path in (self.v2_path, *self.v1_paths.values()):
            membership = path / ("cgroup.procs" if path == self.v2_path else "tasks")
            _sudo("sh", "-c", f"printf '%s\\n' {pid} > {membership}")

    def pids(self) -> tuple[int, ...]:
        if not self.created:
            return ()
        try:
            values = [int(value) for value in (self.v2_path / "cgroup.procs").read_text().split()]
        except (OSError, ValueError) as exc:
            raise InfrastructureError(f"cannot read cgroup PIDs for {self.role}: {exc}") from exc
        return tuple(sorted(values))

    def counters(self) -> CgroupCounters:
        if not self.created:
            raise InfrastructureError("cannot read counters before cgroup creation")
        return counters_from_paths(
            v2_path=self.v2_path,
            v1_paths=self.v1_paths,
            live_pids=self.pids(),
        )

    def kill(self, timeout_s: float = 5.0) -> None:
        if not self.created:
            return
        kill_file = self.v2_path / "cgroup.kill"
        if kill_file.exists():
            with suppress(InfrastructureError):
                _sudo("sh", "-c", f"printf 1 > {kill_file}")
        deadline = time.monotonic() + timeout_s
        while self.pids() and time.monotonic() < deadline:
            for pid in self.pids():
                with suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
            time.sleep(0.05)
        if self.pids():
            raise InfrastructureError(f"cgroup still has live PIDs after kill: {self.pids()}")

    def cleanup(self, *, best_effort: bool = False) -> None:
        errors: list[str] = []
        paths = [*self.v1_paths.values(), self.v2_path]
        for path in paths:
            try:
                if path.exists():
                    _sudo("rmdir", str(path))
                parent = path.parent
                if parent.name == self.run_id and parent.exists():
                    with suppress(InfrastructureError):
                        _sudo("rmdir", str(parent))
            except InfrastructureError as exc:
                errors.append(str(exc))
        self.created = False
        if errors and not best_effort:
            raise InfrastructureError("; ".join(errors))

    def __enter__(self) -> RoleCgroup:
        self.create()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        try:
            self.kill()
        finally:
            self.cleanup()
