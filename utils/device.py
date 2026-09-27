"""Device selection shared by CLI, GUI, and training adapters."""
from __future__ import annotations

import os
from pathlib import Path


def select_device(requested: str, *, cuda_available: bool | None = None) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    if requested == "cpu":
        return "cpu"
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA was selected but is unavailable in this Python environment.")
    return "cuda" if cuda_available else "cpu"


def child_environment(device: str) -> dict[str, str]:
    environment = os.environ.copy()
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def available_memory_gib() -> float | None:
    """Available Linux RAM, capped by the current container memory limit."""
    available: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        pass
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            raw_limit = Path(limit_path).read_text(encoding="utf-8").strip()
            if raw_limit == "max":
                continue
            remaining = max(0, int(raw_limit) - int(Path(used_path).read_text(encoding="utf-8")))
            available = min(available, remaining) if available is not None else remaining
        except (OSError, ValueError):
            pass
    return available / 1024**3 if available is not None else None
