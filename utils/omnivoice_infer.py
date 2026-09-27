"""Isolated OmniVoice LoRA inference using the official OmniVoice API."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_INFER_SCRIPT = r'''
import sys
import torch
import soundfile as sf
from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.lora import load_lora_adapter

base_model, adapter, language, text, output, device = sys.argv[1:]
model = OmniVoice.from_pretrained(base_model, device_map=device, dtype=torch.float32 if device == "cpu" else torch.float16)
model = load_lora_adapter(model, adapter)
audio = model.generate(text=text, language=language)
sf.write(output, audio[0], model.sampling_rate)
'''


def synthesize_omnivoice(artifacts: dict, text: str, language: str, output_path: str | Path,
                         progress=None, device: str = "auto") -> Path:
    """Run the upstream `OmniVoice.from_pretrained` + LoRA + `generate` flow.

    OmniVoice has a separate Transformers dependency stack, so all model imports
    happen in the interpreter used for training, in a child process. Offline
    flags prevent this optional integration from fetching missing assets.
    """
    from utils.model_registry import OMNIVOICE_LANGUAGES

    if not text or not text.strip():
        raise ValueError("Text is required for synthesis.")
    language = language.lower().replace("_", "-")
    if language not in OMNIVOICE_LANGUAGES:
        raise ValueError(f"OmniVoice does not publish support for language {language!r}.")
    checkpoint = Path(artifacts.get("checkpoint", "")).expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"OmniVoice LoRA checkpoint directory not found: {checkpoint}")
    python = artifacts.get("python_executable") or os.environ.get("UFT_OMNIVOICE_PYTHON") or sys.executable
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if progress:
        progress("Loading the cached OmniVoice base model and LoRA adapter in its isolated environment...")
    try:
        completed = subprocess.run(
            [str(python), "-c", _INFER_SCRIPT, artifacts.get("base_model") or artifacts.get("pretrained_model_id", "k2-fsa/OmniVoice"),
             str(checkpoint), language, text, str(destination), device],
            capture_output=True, text=True, env=env, timeout=900,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("OmniVoice inference exceeded the 15 minute timeout.") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"OmniVoice inference failed in {python}:\n{detail[-5000:]}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("OmniVoice inference exited successfully without writing audio.")
    if progress:
        progress("OmniVoice synthesis complete.")
    return destination
