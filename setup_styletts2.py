"""Install the optional StyleTTS2 runtime without changing UFT's environment."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from utils.model_paths import get_models_dir
from utils.runtime_lock import runtime_lock


SOURCE_REVISION = "5cedc71c333f8d8b8551ca59378bdcc7af4c9529"
MODEL_REVISION = "33161ec703934b0e6e226c631d42c2a3eed3ee6a"
MODEL_SHA256 = "1164ffe19a17449d2c722234cecaf2836b35a698fb8ffd42562d2663657dca0a"
MONOTONIC_REVISION = "c6e5e6cb19882164027eb6e35118e841eed9298e"
WAVLM_REPO_ID = "microsoft/wavlm-base-plus"
WAVLM_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
MODEL_RELATIVE = Path("Models/LibriTTS/epochs_2nd_00020.pth")
AUXILIARY_FILES = {
    "Utils/ASR/epoch_00080.pth": 90_000_000,
    "Utils/JDC/bst.t7": 20_000_000,
    "Utils/PLBERT/step_1000000.t7": 20_000_000,
}
Progress = Callable[[str], None] | None


def runtime_paths() -> tuple[Path, Path, Path]:
    home = get_models_dir() / "styletts2"
    python = home / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return home / "source", home / MODEL_RELATIVE, python


def wavlm_path(home: Path | None = None) -> Path:
    return (home or (get_models_dir() / "styletts2")) / "wavlm-base-plus"


def wavlm_is_ready(home: Path | None = None) -> bool:
    destination = wavlm_path(home)
    weights = destination / "pytorch_model.bin"
    try:
        return (weights.is_file() and weights.stat().st_size >= 300_000_000
                and (destination / "config.json").is_file()
                and (destination / ".uft_revision").read_text(encoding="utf-8").strip() == WAVLM_REVISION)
    except OSError:
        return False


def _ensure_wavlm(python: Path, home: Path, progress: Progress = None) -> Path:
    destination = wavlm_path(home)
    config = destination / "config.json"
    weights = destination / "pytorch_model.bin"
    if wavlm_is_ready(home):
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    _notify(progress, f"Downloading {WAVLM_REPO_ID} ({WAVLM_REVISION[:12]}) for offline StyleTTS2 training and inference...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
        "allow_patterns=['config.json', 'pytorch_model.bin'])"
    )
    environment = os.environ.copy()
    environment["HF_HOME"] = str(home / "huggingface")
    _run([str(python), "-c", script, WAVLM_REPO_ID, WAVLM_REVISION, str(destination)], environment=environment)
    if not config.is_file() or not weights.is_file() or weights.stat().st_size < 300_000_000:
        raise RuntimeError(f"Hugging Face snapshot for {WAVLM_REPO_ID} is incomplete at {destination}")
    (destination / ".uft_revision").write_text(WAVLM_REVISION + "\n", encoding="utf-8")
    return destination


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"StyleTTS2 setup command failed: {' '.join(command)}\n{(result.stderr or result.stdout)[-3000:]}")


def _download(url: str, destination: Path, *, minimum_size: int, sha256: str | None = None,
              progress: Progress = None) -> None:
    if destination.is_file() and destination.stat().st_size >= minimum_size:
        if sha256 is None:
            return
        existing_digest = hashlib.sha256()
        with destination.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                existing_digest.update(chunk)
        if existing_digest.hexdigest() == sha256:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    print(f"Downloading {destination.name}...", flush=True)
    digest = hashlib.sha256()
    downloaded = 0
    next_notice = 128 * 1024 ** 2
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded >= next_notice:
                    _notify(progress, f"{destination.name}: {downloaded // (1024 ** 2)} MiB downloaded")
                    next_notice += 128 * 1024 ** 2
        if temporary.stat().st_size < minimum_size:
            raise RuntimeError(f"Downloaded file is too small: {url}")
        if sha256 and digest.hexdigest() != sha256:
            raise RuntimeError(f"Downloaded file failed checksum verification: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def is_ready(*, cuda: bool = False) -> bool:
    source, checkpoint, python = runtime_paths()
    return (
        (source / "train_finetune_accelerate.py").is_file()
        and checkpoint.is_file() and checkpoint.stat().st_size >= 700_000_000
        and python.is_file()
        and (not cuda or _torch_backend(source.parent) == "cuda")
        and wavlm_is_ready()
        and all((source / relative).is_file() and (source / relative).stat().st_size >= size
                for relative, size in AUXILIARY_FILES.items())
    )


def _torch_backend(home: Path) -> str:
    marker = home / ".torch_backend"
    try:
        return marker.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _notify(progress: Progress, message: str) -> None:
    print(message, flush=True)
    if progress:
        progress(message)


def _setup_unlocked(*, cpu: bool = False, progress: Progress = None) -> tuple[Path, Path, Path]:
    source, checkpoint, python = runtime_paths()
    home = source.parent
    home.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("UV_CACHE_DIR", str(home / "uv_cache"))
    environment.setdefault("UV_PYTHON_INSTALL_DIR", str(home / "uv_python"))
    if not shutil.which("uv"):
        raise RuntimeError("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    if not shutil.which("git"):
        raise RuntimeError("StyleTTS2 setup needs git installed.")
    if not shutil.which("espeak-ng"):
        raise RuntimeError("StyleTTS2 needs espeak-ng installed (the UFT Docker image includes it).")
    if shutil.disk_usage(home).free < 8 * 1024 ** 3:
        raise RuntimeError("StyleTTS2 first-use setup needs at least 8 GiB free disk space.")
    _notify(progress, "StyleTTS2 first-use setup downloads its runtime and about 1 GB of weights. LibriTTS model terms: https://github.com/yl4579/StyleTTS2#pre-trained-models")

    if not (source / "train_finetune_accelerate.py").is_file():
        _notify(progress, "Installing official StyleTTS2 source...")
        temporary = home / "source.installing"
        if temporary.exists():
            shutil.rmtree(temporary)
        if source.exists():
            shutil.rmtree(source)
        try:
            clone_env = {**environment, "GIT_LFS_SKIP_SMUDGE": "1"}
            _run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/yl4579/StyleTTS2.git", str(temporary)], environment=clone_env)
            _run(["git", "-C", str(temporary), "checkout", "--detach", SOURCE_REVISION], environment=clone_env)
            temporary.rename(source)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    for relative, minimum_size in AUXILIARY_FILES.items():
        url = f"https://raw.githubusercontent.com/yl4579/StyleTTS2/{SOURCE_REVISION}/{relative}"
        if not (source / relative).is_file() or (source / relative).stat().st_size < minimum_size:
            _notify(progress, f"Downloading StyleTTS2 {Path(relative).name}...")
        _download(url, source / relative, minimum_size=minimum_size, progress=progress)
    if not checkpoint.is_file() or checkpoint.stat().st_size < 700_000_000:
        _notify(progress, "Downloading the 771 MB StyleTTS2 LibriTTS checkpoint...")
    _download(
        f"https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/{MODEL_REVISION}/{MODEL_RELATIVE.name}",
        checkpoint,
        minimum_size=700_000_000,
        sha256=MODEL_SHA256,
        progress=progress,
    )

    needs_environment = not python.is_file() or (not cpu and _torch_backend(home) != "cuda")
    if needs_environment:
        _notify(progress, "Creating the isolated StyleTTS2 uv environment...")
        temporary_env = home / ".venv.installing"
        if temporary_env.exists():
            shutil.rmtree(temporary_env)
        try:
            _run(["uv", "venv", "--python", "3.12", str(temporary_env)], environment=environment)
            temporary_python = temporary_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            torch_command = ["uv", "pip", "install", "--python", str(temporary_python)]
            if cpu:
                torch_command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
            _run(torch_command + ["torch==2.5.1", "torchaudio==2.5.1"], environment=environment)
            upstream_requirements = (source / "requirements.txt").read_text(encoding="utf-8").splitlines()
            pinned_requirements = home / "requirements-styletts2.txt"
            pinned_requirements.write_text(
                "\n".join(line for line in upstream_requirements if "resemble-ai/monotonic_align" not in line)
                + f"\ngit+https://github.com/resemble-ai/monotonic_align.git@{MONOTONIC_REVISION}\n",
                encoding="utf-8",
            )
            _run([
                "uv", "pip", "install", "--python", str(temporary_python),
                "-r", str(pinned_requirements), "transformers==4.44.2", "phonemizer",
                "tensorboard", "click", "pandas", "torch==2.5.1", "torchaudio==2.5.1",
            ], environment=environment)
            _run([str(temporary_python), "-c", "import torch, torchaudio, phonemizer, monotonic_align, accelerate"], environment=environment)
            if python.parent.parent.exists():
                shutil.rmtree(python.parent.parent)
            temporary_env.rename(home / ".venv")
            (home / ".torch_backend").write_text("cpu" if cpu else "cuda", encoding="utf-8")
        finally:
            if temporary_env.exists():
                shutil.rmtree(temporary_env)
    _ensure_wavlm(python, home, progress)
    _notify(progress, f"StyleTTS2 ready. Source: {source}\nCheckpoint: {checkpoint}\nPython: {python}")
    return source, checkpoint, python


def setup(*, cpu: bool = False, progress: Progress = None) -> tuple[Path, Path, Path]:
    with runtime_lock(get_models_dir() / "styletts2" / ".setup.lock"):
        return _setup_unlocked(cpu=cpu, progress=progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Use CPU-only PyTorch wheels for the StyleTTS2 environment.")
    arguments = parser.parse_args()
    try:
        setup(cpu=arguments.cpu)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"StyleTTS2 setup failed: {error}", file=sys.stderr)
        sys.exit(1)
