"""Inference sessions for the layout models."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import onnxruntime

logger = logging.getLogger(__name__)

LAYOUT_THREADS_ENV = "BABELDOC_LAYOUT_THREADS"


def _cgroup_v2_limit(root: Path, proc_cgroup: str) -> float | None:
    path = next(
        (line[3:].strip() for line in proc_cgroup.splitlines() if line.startswith("0::")),
        None,
    )
    if path is None:
        return None
    parts = [part for part in path.split("/") if part and part != ".."]
    limits = []
    # A parent's quota caps its children, so take the tightest on the way up.
    for depth in range(len(parts), -1, -1):
        try:
            quota, period = (root.joinpath(*parts[:depth]) / "cpu.max").read_text().split()[:2]
            if quota != "max":
                limits.append(int(quota) / int(period))
        except (OSError, ValueError):
            continue
    return min(limits, default=None)


def _cgroup_v1_limit(root: Path) -> float | None:
    for name in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
        try:
            quota = int((root / name / "cpu.cfs_quota_us").read_text())
            period = int((root / name / "cpu.cfs_period_us").read_text())
        except (OSError, ValueError):
            continue
        return quota / period if quota > 0 and period > 0 else None
    return None


def cgroup_cpu_limit(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """CPUs the cgroup quota allows this process, or None when there is none."""
    try:
        proc_cgroup = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None
    if (root / "cgroup.controllers").exists():
        return _cgroup_v2_limit(root, proc_cgroup)
    return _cgroup_v1_limit(root)


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def layout_threads() -> int:
    """Intra-op threads for a layout session, 0 to let the runtime choose.

    onnxruntime starts one thread per physical core of the host and ignores
    the container's CPU quota, so in a pod limited to 6 CPUs on a 16-vCPU node
    its threads get throttled. Use the quota (or CPU affinity) instead;
    BABELDOC_LAYOUT_THREADS overrides both.
    """
    raw = os.environ.get(LAYOUT_THREADS_ENV)
    if raw:
        try:
            threads = int(raw)
        except ValueError:
            threads = 0
        if threads > 0:
            return threads
        logger.warning("Ignoring %s=%r: expected a positive integer", LAYOUT_THREADS_ENV, raw)
    cpus = _usable_cpus()
    limit = cgroup_cpu_limit()
    if limit is not None:
        return max(1, min(cpus, math.floor(limit)))
    if cpus < (os.cpu_count() or cpus):
        return cpus
    return 0


def ort_session_options(threads: int) -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return options
