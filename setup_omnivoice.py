"""Install the optional OmniVoice runtime and cache its base models."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from utils.model_paths import get_models_dir
from utils.runtime_lock import runtime_lock


SOURCE_REVISION = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
Progress = Callable[[str], None] | None


def runtime_home() -> Path:
    return get_models_dir() / "omnivoice"


def runtime_python() -> Path:
    return runtime_home() / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _notify(progress: Progress, message: str) -> None:
    print(message, flush=True)
    if progress:
        progress(message)


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"OmniVoice setup command failed: {' '.join(command)}\n{(result.stderr or result.stdout)[-3000:]}")


def _manifest() -> dict:
    path = runtime_home() / "runtime.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_ready(*, cuda: bool = False) -> bool:
    manifest = _manifest()
    return (
        runtime_python().is_file()
        and manifest.get("source_revision") == SOURCE_REVISION
        and manifest.get("model_revision") == MODEL_REVISION
        and manifest.get("qwen_revision") == QWEN_REVISION
        and (not cuda or manifest.get("torch_backend") == "cuda")
        and all(manifest.get(key) and Path(manifest[key]).is_dir()
                for key in ("base_model", "audio_tokenizer", "qwen_model"))
        and all(manifest.get(key) and (Path(manifest[key]) / "model.safetensors").is_file()
                for key in ("base_model", "audio_tokenizer", "qwen_model"))
    )


def _setup_unlocked(*, cpu: bool = False, progress: Progress = None) -> dict:
    home = runtime_home()
    home.mkdir(parents=True, exist_ok=True)
    if not shutil.which("uv"):
        raise RuntimeError("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    if not shutil.which("git"):
        raise RuntimeError("OmniVoice setup needs git installed.")
    environment = os.environ.copy()
    environment.setdefault("UV_CACHE_DIR", str(home / "uv_cache"))
    environment.setdefault("UV_PYTHON_INSTALL_DIR", str(home / "uv_python"))
    environment["HF_HOME"] = str(home / "huggingface")
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment.setdefault("HF_HUB_DISABLE_XET", "1")
    if shutil.disk_usage(home).free < 10 * 1024 ** 3:
        raise RuntimeError("OmniVoice first-use setup needs at least 10 GiB free disk space.")
    _notify(progress, "OmniVoice first-use setup downloads several GB. Its weights are noncommercial: https://huggingface.co/k2-fsa/OmniVoice ; tokenizer terms: https://huggingface.co/k2-fsa/OmniVoice/blob/main/audio_tokenizer/LICENSE")

    python = runtime_python()
    current = _manifest()
    backend_file = home / ".torch_backend"
    backend = backend_file.read_text(encoding="utf-8").strip() if backend_file.is_file() else current.get("torch_backend", "")
    if python.is_file() and backend not in {"cpu", "cuda"}:
        probe = subprocess.run(
            [str(python), "-c", "import torch; print('cuda' if torch.version.cuda else 'cpu')"],
            capture_output=True, text=True, check=True,
        )
        backend = probe.stdout.strip().splitlines()[-1]
        backend_file.write_text(backend, encoding="utf-8")
    needs_environment = not python.is_file() or (not cpu and backend != "cuda")
    if needs_environment:
        _notify(progress, "Installing isolated OmniVoice uv environment (several GB)...")
        temporary = home / ".venv.installing"
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            _run(["uv", "venv", "--python", "3.12", str(temporary)], environment)
            temporary_python = temporary / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            torch_command = ["uv", "pip", "install", "--python", str(temporary_python)]
            if cpu:
                torch_command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
            _run(torch_command + ["torch==2.8.0", "torchaudio==2.8.0"], environment)
            _run([
                "uv", "pip", "install", "--python", str(temporary_python),
                f"git+https://github.com/k2-fsa/OmniVoice.git@{SOURCE_REVISION}",
                "peft>=0.20", "torch==2.8.0", "torchaudio==2.8.0",
            ], environment)
            _run([str(temporary_python), "-c", "import omnivoice.utils.lora, peft, torch, torchaudio"], environment)
            if python.parent.parent.exists():
                shutil.rmtree(python.parent.parent)
            temporary.rename(home / ".venv")
            backend_file.write_text("cpu" if cpu else "cuda", encoding="utf-8")
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    _notify(progress, "Downloading OmniVoice and Qwen base models (about 5 GB)...")
    script = (
        "import json,sys; from huggingface_hub import snapshot_download; "
        "base=snapshot_download('k2-fsa/OmniVoice',revision=sys.argv[1]); "
        "qwen=snapshot_download('Qwen/Qwen3-0.6B',revision=sys.argv[2]); "
        "print(json.dumps({'base_model':base,'audio_tokenizer':base+'/audio_tokenizer','qwen_model':qwen}))"
    )
    result = subprocess.run(
        [str(python), "-c", script, MODEL_REVISION, QWEN_REVISION],
        env=environment, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"OmniVoice model download failed: {(result.stderr or result.stdout)[-3000:]}")
    assets = json.loads(result.stdout.strip().splitlines()[-1])
    for key, path in assets.items():
        if not Path(path).is_dir():
            raise RuntimeError(f"Downloaded OmniVoice asset is missing: {key}: {path}")
    if not any(Path(assets["base_model"]).glob("*.safetensors")):
        raise RuntimeError("OmniVoice base checkpoint is missing from its downloaded snapshot.")
    manifest = {
        **assets,
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "qwen_revision": QWEN_REVISION,
        "torch_backend": ("cpu" if cpu else "cuda") if needs_environment else backend,
        "python_executable": str(python),
        "hf_home": environment["HF_HOME"],
    }
    temporary_manifest = home / "runtime.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(home / "runtime.json")
    _notify(progress, "OmniVoice runtime and models are ready.")
    return manifest


def setup(*, cpu: bool = False, progress: Progress = None) -> dict:
    with runtime_lock(runtime_home() / ".setup.lock"):
        return _setup_unlocked(cpu=cpu, progress=progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Install CPU-only PyTorch wheels.")
    arguments = parser.parse_args()
    try:
        setup(cpu=arguments.cpu)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"OmniVoice setup failed: {error}", file=sys.stderr)
        sys.exit(1)
