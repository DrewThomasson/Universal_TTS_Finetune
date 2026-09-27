"""Isolated OmniVoice LoRA inference using the official OmniVoice API."""
from __future__ import annotations

import os
import shutil
import subprocess
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
    checkpoint_value = artifacts.get("checkpoint")
    checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else None
    if (checkpoint is None or not checkpoint.is_dir()) and artifacts.get("artifacts_file"):
        checkpoint = Path(artifacts["artifacts_file"]).parent / "model"
    if checkpoint is None or not checkpoint.is_dir():
        raise FileNotFoundError(f"OmniVoice LoRA checkpoint directory not found: {checkpoint}")
    override = os.environ.get("UFT_OMNIVOICE_PYTHON")
    python = override or artifacts.get("python_executable")
    if override and not shutil.which(override):
        raise FileNotFoundError(f"OmniVoice Python override not found: {override}")
    base_model = artifacts.get("base_model") or artifacts.get("pretrained_model_id", "k2-fsa/OmniVoice")
    hf_home = artifacts.get("hf_home")
    from setup_omnivoice import runtime_python, setup, is_ready, _manifest
    managed_python = runtime_python().absolute()
    upgrade_cuda = (device == "cuda" and not override and python
                    and Path(python).expanduser().absolute() == managed_python
                    and not is_ready(cuda=True))
    # A saved interpreter/cache can belong to another host or container.
    # Recreate the managed runtime locally instead of executing stale paths.
    missing_base = Path(base_model).is_absolute() and not Path(base_model).is_dir()
    if not python or not shutil.which(str(python)) or missing_base or upgrade_cuda:
        runtime = _manifest() if is_ready(cuda=device == "cuda") else setup(cpu=device != "cuda", progress=progress)
        python = override or runtime["python_executable"]
        base_model = runtime["base_model"]
        hf_home = runtime["hf_home"]
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if progress:
        progress("Loading the cached OmniVoice base model and LoRA adapter in its isolated environment...")
    try:
        completed = subprocess.run(
            [str(python), "-c", _INFER_SCRIPT, base_model,
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
