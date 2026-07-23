"""Run-level measurement.

Scope is what section 7 of the design asks a run to report: makespan, CPU, GPU, I/O,
network and per-operation timing. Nothing here reconstructs the process tree or
audits containers; per-operation CPU and duration already come from the boundary
records, and what remains is the envelope around them.

CPU and I/O come from the run's own cgroup rather than from process sampling,
because a sampler misses processes that live and die between two samples and an
agent workload is full of those.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from .constants import METRICS_SCHEMA
from .util import monotonic_ns


@dataclass
class GPUSample:
    at_ns: int
    per_gpu: dict[int, dict[str, float]]


class GPUSampler:
    """Poll GPU utilisation on a background thread.

    Sampling runs at a low rate on purpose: the value being measured is a workload
    lasting minutes, and a tight poll loop would show up in the CPU number this same
    run is trying to report.
    """

    def __init__(self, gpu_ids: list[int], *, interval_s: float = 1.0) -> None:
        self.gpu_ids = gpu_ids
        self.interval_s = interval_s
        self.samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = bool(gpu_ids) and shutil.which("nvidia-smi") is not None

    def _read(self) -> dict[int, dict[str, float]]:
        query = "index,utilization.gpu,memory.used,power.draw"
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    ",".join(str(gpu) for gpu in self.gpu_ids),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if result.returncode != 0:
            return {}
        values: dict[int, dict[str, float]] = {}
        for line in result.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                values[int(parts[0])] = {
                    "utilization_percent": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                    "power_watts": float(parts[3]),
                }
            except ValueError:
                continue
        return values

    def start(self) -> None:
        if not self.available:
            return

        def run() -> None:
            while not self._stop.wait(self.interval_s):
                per_gpu = self._read()
                if per_gpu:
                    self.samples.append(GPUSample(at_ns=monotonic_ns(), per_gpu=per_gpu))

        self._thread = threading.Thread(target=run, name="minireplay-gpu", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summarize(self, gate_at_ns: int, terminal_at_ns: int) -> dict[str, Any]:
        window = [s for s in self.samples if gate_at_ns <= s.at_ns <= terminal_at_ns]
        if not window:
            return {"available": self.available, "samples": 0, "gpu_active_seconds": 0.0}
        active_seconds = 0.0
        peak_memory = 0.0
        utilisation_sum = 0.0
        for previous, current in zip(window, window[1:], strict=False):
            span_s = (current.at_ns - previous.at_ns) / 1e9
            busiest = max(
                (values["utilization_percent"] for values in previous.per_gpu.values()),
                default=0.0,
            )
            active_seconds += span_s * busiest / 100.0
        for sample in window:
            for values in sample.per_gpu.values():
                peak_memory = max(peak_memory, values["memory_used_mib"])
                utilisation_sum += values["utilization_percent"]
        return {
            "available": True,
            "samples": len(window),
            "gpu_active_seconds": round(active_seconds, 3),
            "peak_memory_mib": round(peak_memory, 1),
            "mean_utilization_percent": round(
                utilisation_sum / max(1, sum(len(s.per_gpu) for s in window)), 2
            ),
        }


@dataclass
class HostCounters:
    net_sent_bytes: int
    net_recv_bytes: int
    disk_read_bytes: int
    disk_write_bytes: int

    @staticmethod
    def sample() -> HostCounters:
        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        return HostCounters(
            net_sent_bytes=int(net.bytes_sent) if net else 0,
            net_recv_bytes=int(net.bytes_recv) if net else 0,
            disk_read_bytes=int(disk.read_bytes) if disk else 0,
            disk_write_bytes=int(disk.write_bytes) if disk else 0,
        )

    def delta(self, baseline: HostCounters) -> dict[str, int]:
        return {
            "net_sent_bytes": max(0, self.net_sent_bytes - baseline.net_sent_bytes),
            "net_recv_bytes": max(0, self.net_recv_bytes - baseline.net_recv_bytes),
            "disk_read_bytes": max(0, self.disk_read_bytes - baseline.disk_read_bytes),
            "disk_write_bytes": max(0, self.disk_write_bytes - baseline.disk_write_bytes),
        }


@dataclass
class RunMetrics:
    gpu: GPUSampler
    cgroup: Any | None = None
    gate_at_ns: int = 0
    terminal_at_ns: int = 0
    _host_baseline: HostCounters | None = field(default=None, repr=False)
    _cgroup_baseline: Any | None = field(default=None, repr=False)
    _host_final: dict[str, int] = field(default_factory=dict, repr=False)
    _cgroup_final: Any | None = field(default=None, repr=False)

    def mark_gate(self, at_ns: int) -> None:
        """Zero every counter at the instant the workload is released.

        Setup work — image pulls, interpreter startup, service binding — happens
        before this point and is not part of what the experiment measures.
        """

        self.gate_at_ns = at_ns
        self._host_baseline = HostCounters.sample()
        if self.cgroup is not None:
            self._cgroup_baseline = self.cgroup.counters()
        self.gpu.start()

    def mark_terminal(self, at_ns: int) -> None:
        """Freeze counters at the sweep boundary, before any evidence processing."""

        self.terminal_at_ns = at_ns
        if self.cgroup is not None and self._cgroup_baseline is not None:
            self._cgroup_final = self.cgroup.counters().delta(self._cgroup_baseline)
        if self._host_baseline is not None:
            self._host_final = HostCounters.sample().delta(self._host_baseline)
        self.gpu.stop()

    def summary(
        self,
        *,
        run_id: str,
        bundle_id: str,
        mode: str,
        replay_mode: str,
        operations: dict[str, Any],
        busy_span_seconds: float | None = None,
    ) -> dict[str, Any]:
        makespan_s = max(0.0, (self.terminal_at_ns - self.gate_at_ns) / 1e9)
        cgroup = self._cgroup_final.as_dict() if self._cgroup_final is not None else {}
        return {
            "schema_version": METRICS_SCHEMA,
            "run_id": run_id,
            "bundle_id": bundle_id,
            "mode": mode,
            "replay_mode": replay_mode,
            # Gate to terminal. For a recording this is the sweep's window, which
            # can outlast the work; compare `busy_span_seconds` across runs instead.
            "makespan_seconds": round(makespan_s, 3),
            "busy_span_seconds": busy_span_seconds,
            "gate_at_ns": self.gate_at_ns,
            "terminal_at_ns": self.terminal_at_ns,
            "framework_cgroup": cgroup,
            "host_deltas": dict(self._host_final),
            "gpu": self.gpu.summarize(self.gate_at_ns, self.terminal_at_ns),
            "operations": operations,
        }


def operation_summary(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate per-operation timing and outcome from the ledger records."""

    summary: dict[str, Any] = {}
    for kind, entries in records.items():
        if not entries:
            summary[kind] = {"count": 0}
            continue
        durations = [
            (int(entry["ended_at_ns"]) - int(entry["started_at_ns"])) / 1e9
            for entry in entries
            if "ended_at_ns" in entry and "started_at_ns" in entry
        ]
        cpu = [float(entry.get("cpu_seconds", 0.0)) for entry in entries]
        statuses: dict[str, int] = {}
        for entry in entries:
            status = str(entry.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
        summary[kind] = {
            "count": len(entries),
            "total_seconds": round(sum(durations), 3),
            "max_seconds": round(max(durations), 3) if durations else 0.0,
            "mean_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "cpu_seconds": round(sum(cpu), 3),
            "status_counts": statuses,
        }
    return summary


def wait_until(predicate, *, timeout_s: float, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()
