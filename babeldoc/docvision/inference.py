"""Inference sessions for the layout models."""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path

import onnxruntime

logger = logging.getLogger(__name__)

LAYOUT_THREADS_ENV = "BABELDOC_LAYOUT_THREADS"
LAYOUT_BACKEND_ENV = "BABELDOC_LAYOUT_BACKEND"
LAYOUT_MODEL_ENV = "BABELDOC_LAYOUT_MODEL"
LAYOUT_MODELS = ("doclayout-yolo", "pp-doclayoutv3")


def layout_model_name() -> str:
    """The layout model BABELDOC_LAYOUT_MODEL selects, DocLayout-YOLO by default."""
    name = os.environ.get(LAYOUT_MODEL_ENV, "").strip().lower()
    if not name:
        return LAYOUT_MODELS[0]
    if name not in LAYOUT_MODELS:
        logger.warning(
            "Unknown %s=%r, using %s", LAYOUT_MODEL_ENV, name, LAYOUT_MODELS[0]
        )
        return LAYOUT_MODELS[0]
    return name


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


class OpenVINOSession:
    """Runs an ONNX model on OpenVINO's CPU plugin behind onnxruntime's run()."""

    def __init__(self, model_path: str | Path, threads: int):
        import openvino

        core = openvino.Core()
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            config["INFERENCE_NUM_THREADS"] = threads
        self._compiled = core.compile_model(
            core.read_model(str(model_path)), "CPU", config
        )
        # A compiled model's implicit infer request serves one call at a time.
        self._lock = threading.Lock()

    def run(self, _output_names, feeds):
        with self._lock:
            result = self._compiled(feeds)
        return [result[output] for output in self._compiled.outputs]


def openvino_session(model_path: str | Path, threads: int) -> OpenVINOSession | None:
    """An OpenVINO session if BABELDOC_LAYOUT_BACKEND=openvino, else None.

    OpenVINO computes in bf16 on CPUs that support it (AVX512_BF16 or AMX),
    which roughly halves the layout model's time with the same boxes; in f32
    it is about as fast as onnxruntime. Falls back to onnxruntime (None) when
    openvino is missing or fails to load the model.
    """
    backend = os.environ.get(LAYOUT_BACKEND_ENV, "").strip().lower()
    if backend in ("", "onnxruntime"):
        return None
    if backend != "openvino":
        logger.warning(
            "Unknown %s=%r, using onnxruntime", LAYOUT_BACKEND_ENV, backend
        )
        return None
    try:
        session = OpenVINOSession(model_path, threads)
    except ImportError:
        logger.warning(
            "%s=openvino but openvino is not installed, using onnxruntime",
            LAYOUT_BACKEND_ENV,
        )
        return None
    except Exception:
        logger.warning(
            "OpenVINO failed to load the layout model, using onnxruntime",
            exc_info=True,
        )
        return None
    logger.info("Layout model runs on OpenVINO")
    return session
