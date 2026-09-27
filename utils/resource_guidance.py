"""Planning guidance for UFT training devices and short-clip smoke runs."""
from __future__ import annotations

from utils.model_registry import get_model_spec


# These are planning budgets for short clips and small batches, not tested
# minimums for every language, checkpoint, clip length, or dataset.
# Hard adapter guards are identified in the note rather than implied here.
_GUIDANCE = {
    "piper": (8, 6, "Piper can train on CPU; a published voice may need extra disk for conversion."),
    "xtts_v1": (32, 12, "XTTS CPU training can be exceptionally slow; use short clips and batch size 1."),
    "xtts_v2": (32, 12, "XTTS CPU training can be exceptionally slow; use short clips and batch size 1."),
    "mms_vits": (24, 12, "MMS needs a language-matched base checkpoint."),
    "styletts2": (24, 12, "Hard CUDA guard: at least 9 GiB free VRAM. Batch size 2. Short-clip CPU training verified; CPU uses the initial acoustic stage."),
    "omnivoice": (24, 16, "Hard guards: CUDA needs at least 16 GiB total and 12 GiB free VRAM; CPU needs Linux and 16 GiB available RAM; either needs 20 GiB free disk. Short-clip CPU LoRA training verified."),
    "f5_tts": (24, 12, "Hard guards: CUDA needs at least 9 GiB free VRAM; either device needs 12 GiB free disk. CPU uses full precision; CPU training is unverified."),
}
_CPU_STEP_VERIFIED = {"align_tts", "piper", "styletts2", "omnivoice"}


def _available_ram_gib() -> float | None:
    from utils.device import available_memory_gib
    return available_memory_gib()


def _gpu_memory() -> tuple[float, float] | None:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return total / 1024**3, free / 1024**3
    except (ImportError, OSError, RuntimeError):
        pass
    return None


def training_resource_guidance(model_key: str, device: str = "auto") -> str:
    spec = get_model_spec(model_key)
    ram_gib, vram_gib, note = _GUIDANCE.get(model_key, (16, 8, "Use batch size 1 and short clips on a small machine."))
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    gpu = _gpu_memory()
    actual = "cuda" if device == "cuda" or (device == "auto" and gpu) else "cpu"
    if model_key == "piper" and device == "auto" and actual == "cpu":
        try:
            import torch
            if torch.backends.mps.is_available():
                actual = "mps"
        except (ImportError, AttributeError):
            pass
    ram = _available_ram_gib()
    current = f"Available RAM: {ram:.1f} GiB" if ram is not None else "Available RAM: unknown"
    current += f"; GPU: {gpu[0]:.1f} GiB total, {gpu[1]:.1f} GiB free" if gpu else "; CUDA GPU: unavailable in this environment"
    budget = f"plan for about {ram_gib} GiB system RAM and {vram_gib} GiB VRAM" if actual == "cuda" else f"plan for about {ram_gib} GiB system RAM"
    shortfalls = []
    if ram is not None and ram < ram_gib:
        shortfalls.append("available RAM is below the planning estimate")
    if actual == "cuda":
        if gpu is None:
            shortfalls.append("CUDA is unavailable here")
        elif gpu[1] < vram_gib:
            shortfalls.append("free VRAM is below the planning estimate")
    comparison = " Current resource check: " + "; ".join(shortfalls) + "." if shortfalls else ""
    verification = " CPU optimizer-step training has not yet been verified for this engine." if actual == "cpu" and model_key not in _CPU_STEP_VERIFIED else ""
    return (
        f"{spec.label} · selected device: {actual}. For short clips and a small batch, {budget}. "
        f"This is a planning estimate, not a guaranteed minimum. {note} {current}. "
        f"Longer clips, larger batches, and other checkpoints may need more.{comparison}{verification}"
    )
