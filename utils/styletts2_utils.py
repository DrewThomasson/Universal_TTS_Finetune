"""Standalone adapter for the official StyleTTS2 fine-tuning runner.

This module deliberately does not import StyleTTS2 into Universal TTS's process.
It stages the existing UFT dataset into StyleTTS2's 24 kHz list format, checks
the upstream checkout and local assets, and starts its official script in an
isolated subprocess. It never downloads checkpoints or dependencies.
"""
from __future__ import annotations

import json
import math
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str], None]
OFFICIAL_REPO = "https://github.com/yl4579/StyleTTS2"
STYLE_SAMPLE_RATE = 24000


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"StyleTTS2 {description} not found: {path}")
    return path.resolve()


def _read_uft_rows(dataset_dir: Path, split_name: str) -> list[tuple[str, str]]:
    """Read UFT's pipe-delimited `clip|clean_text|original_text` split."""
    manifest = _required_file(dataset_dir / f"metadata_{split_name}.csv", f"{split_name} manifest")
    rows: list[tuple[str, str]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, raw in enumerate(stream, start=1):
            raw = raw.rstrip("\r\n")
            if not raw.strip():
                continue
            fields = raw.split("|", 2)
            if len(fields) < 2:
                raise ValueError(f"Malformed {manifest.name} line {line_number}: expected clip|text")
            clip, text = fields[0].strip(), fields[1].strip()
            if not clip or not text:
                continue
            if Path(clip).name != clip:
                raise ValueError(f"Unsafe clip path in {manifest.name} line {line_number}: {clip!r}")
            # UFT's prepare-dataset writes LJSpeech clip IDs without the WAV
            # extension; hand-authored manifests may include it already.
            if not Path(clip).suffix:
                clip += ".wav"
            if "|" in text or "\n" in text or "\r" in text:
                raise ValueError(
                    f"Unsupported pipe or newline in transcript at {manifest.name} line {line_number}; "
                    "StyleTTS2's upstream manifest parser uses pipe-delimited single lines."
                )
            rows.append((clip, text))
    if not rows:
        raise ValueError(f"No usable speech/text rows in {manifest}")
    return rows


def _stage_audio(source: Path, destination: Path) -> None:
    """Copy mono WAVs at 24 kHz, using the already-required torchaudio stack."""
    try:
        import torch
        import torchaudio
    except Exception as exc:
        raise RuntimeError(
            "StyleTTS2 dataset staging requires working torch and torchaudio in the UFT environment."
        ) from exc
    try:
        waveform, sample_rate = torchaudio.load(str(source))
        if waveform.ndim != 2 or waveform.shape[1] == 0:
            raise ValueError("empty or malformed audio")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != STYLE_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, STYLE_SAMPLE_RATE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(destination), waveform.to(dtype=torch.float32), STYLE_SAMPLE_RATE)
    except Exception as exc:
        raise RuntimeError(f"Could not stage audio file {source}: {exc}") from exc


def _make_ood_texts(rows: list[tuple[str, str]], requested_min_length: int) -> tuple[list[str], int]:
    """Build OOD entries safe for upstream's rejection-sampling loop.

    meldataset.py repeatedly samples until it gets text at least min_length
    characters long, and its random index call requires two or more entries.
    UFT utterance-level transcripts are often shorter than the default 50.
    """
    words = " ".join(text.strip() for _, text in rows if text.strip()).split()
    full_text = " ".join(words)
    if len(full_text) < 2:
        raise ValueError("StyleTTS2 needs at least two transcript characters to build OOD text entries.")
    min_length = min(max(int(requested_min_length), 1), max(1, len(full_text) // 2))
    target_length = max(min_length, 180)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_length + extra > target_length and current_length >= min_length:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
            extra = len(word)
        current.append(word)
        current_length += extra
    if current:
        tail = " ".join(current)
        if chunks and len(tail) < min_length:
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)
    if len(chunks) < 2:
        midpoint = len(full_text) // 2
        chunks = [full_text[:midpoint].strip(), full_text[midpoint:].strip()]
    effective_min_length = min(min_length, *(len(chunk) for chunk in chunks))
    if effective_min_length < 1 or len(chunks) < 2:
        raise ValueError("Could not build at least two non-empty StyleTTS2 OOD text entries.")
    return chunks, effective_min_length


def _validate_runtime(repo: Path, checkpoint: Path, python_executable: str) -> dict[str, Path]:
    if not repo.is_dir():
        raise FileNotFoundError(
            f"StyleTTS2 checkout is required at {repo}; clone {OFFICIAL_REPO} locally first."
        )
    train_script = _required_file(repo / "train_finetune_accelerate.py", "official single-GPU fine-tuning script")
    base_config = _required_file(repo / "Configs" / "config_ft.yml", "official fine-tuning config")
    checkpoint = _required_file(checkpoint, "LibriTTS base checkpoint")
    # Check the actual interpreter used for training, which may differ from
    # this process (for example, when UFT is launched outside its venv).
    modules = ("torch", "torchaudio", "yaml", "munch", "numpy", "pandas", "librosa", "einops", "transformers", "accelerate", "tensorboard", "monotonic_align", "click")
    probe = "import importlib.util,sys; missing=[m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]; print('\\n'.join(missing)); sys.exit(bool(missing))"
    try:
        checked = subprocess.run(
            [python_executable, "-c", probe, *modules],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Cannot run StyleTTS2 Python interpreter {python_executable!r}: {exc}") from exc
    if checked.returncode:
        missing = checked.stdout.strip().replace("\n", ", ") or "unknown dependency"
        raise RuntimeError(
            f"StyleTTS2 dependencies missing from training interpreter {python_executable!r}: {missing}. "
            "Install the official StyleTTS2 requirements into that interpreter."
        )
    for relative, description in (
        ("Utils/ASR/config.yml", "ASR config"),
        ("Utils/ASR/epoch_00080.pth", "ASR aligner checkpoint"),
        ("Utils/JDC/bst.t7", "JDC pitch checkpoint"),
    ):
        _required_file(repo / relative, description)
    plbert_dir = repo / "Utils" / "PLBERT"
    _required_file(plbert_dir / "config.yml", "PL-BERT config")
    if not list(plbert_dir.glob("step_*.t7")):
        raise FileNotFoundError(
            f"StyleTTS2 PL-BERT checkpoint not found under {plbert_dir}; expected a step_*.t7 file."
        )
    return {"repo": repo.resolve(), "config": base_config, "checkpoint": checkpoint, "train_script": train_script}


def train_styletts2(
    *,
    dataset_dir: str | os.PathLike[str],
    training_root: str | os.PathLike[str],
    styletts2_repo: str | os.PathLike[str],
    pretrained_checkpoint: str | os.PathLike[str],
    language: str = "en",
    epochs: int = 50,
    batch_size: int = 8,
    python_executable: str = sys.executable,
    stream_logs: bool = True,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fine-tune official StyleTTS2; caller supplies local repo and weights.

    Returns run metadata, or on success a `ready/artifacts.json` compatible
    descriptor. This adapter intentionally supports English only: upstream's
    default ASR/PL-BERT/token inventory are English-oriented.
    """
    if language.lower().replace("_", "-") not in {"en", "en-us", "en-gb"}:
        raise ValueError("StyleTTS2 adapter currently supports English only (language='en').")
    if epochs < 1 or batch_size < 2:
        raise ValueError("StyleTTS2 needs at least one epoch and batch size 2 or greater; upstream fails at batch size 1.")
    dataset = Path(dataset_dir).expanduser().resolve()
    root = Path(training_root).expanduser().resolve()
    paths = _validate_runtime(Path(styletts2_repo).expanduser().resolve(), Path(pretrained_checkpoint).expanduser().resolve(), python_executable)
    train_rows = _read_uft_rows(dataset, "train")
    val_rows = _read_uft_rows(dataset, "val")
    if len(train_rows) < batch_size:
        raise ValueError(f"StyleTTS2 needs at least {batch_size} training clips for batch size {batch_size}.")

    gpu_total_gib = None
    if not dry_run:
        cuda_probe = subprocess.run(
            [python_executable, "-c", "import torch,sys; print(torch.cuda.get_device_properties(0).total_memory / 1024**3, torch.cuda.mem_get_info(0)[0] / 1024**3) if torch.cuda.is_available() else sys.exit(1)"],
            check=False,
            capture_output=True,
            text=True,
        )
        if cuda_probe.returncode:
            raise RuntimeError(
                "StyleTTS2 fine-tuning needs CUDA in its training Python environment. "
                "CPU RAM does not replace GPU VRAM; no training process was started."
            )
        gpu_total_gib, gpu_free_gib = map(float, cuda_probe.stdout.strip().splitlines()[-1].split())
        if gpu_free_gib < 9:
            raise RuntimeError(f"StyleTTS2 needs at least 9 GiB free VRAM for this guarded run; {gpu_free_gib:.1f} GiB is free. No training process was started.")

    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / "wavs"
    list_root = root / "data" / "lists"
    log_dir = root / "Models" / "StyleTTS2"
    list_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    def stage(rows: list[tuple[str, str]], split: str) -> Path:
        destination_list = list_root / f"{split}_list.txt"
        output_lines: list[str] = []
        for index, (clip, text) in enumerate(rows, start=1):
            source = dataset / "wavs" / clip
            _required_file(source, f"dataset WAV for {split} row {index}")
            _stage_audio(source, data_root / clip)
            output_lines.append(f"{clip}|{text}|0")
            if index % 100 == 0:
                _notify(progress, f"StyleTTS2 staging {split} data: {index}/{len(rows)}")
        destination_list.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        return destination_list

    _notify(progress, "Preparing StyleTTS2 24 kHz dataset manifests...")
    train_list = stage(train_rows, "train")
    val_list = stage(val_rows, "val")
    # Upstream loops until a random OOD string is long enough and its random
    # index calculation requires two or more entries. Combine short utterances.
    ood_path = list_root / "OOD_texts.txt"

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("StyleTTS2 configuration staging requires PyYAML.") from exc
    with paths["config"].open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid upstream StyleTTS2 config: {paths['config']}")
    config.update({"log_dir": str(log_dir), "epochs": int(epochs), "batch_size": int(batch_size), "device": "cuda"})
    if gpu_total_gib is not None and gpu_total_gib < 16:
        # The official recipe's 400-frame crops and SLM adversarial phase can
        # overrun a 12 GiB card. Keep the initial acoustic fine-tuning stage.
        config["max_len"] = min(int(config.get("max_len", 400)), 96)
        config.setdefault("loss_params", {})["joint_epoch"] = int(epochs) + 1
        _notify(progress, f"StyleTTS2 low-memory profile: {gpu_total_gib:.1f} GiB GPU, max_len={config['max_len']}, SLM adversarial phase disabled.")
    config["save_freq"] = 1
    config["pretrained_model"] = str(paths["checkpoint"])
    config["second_stage_load_pretrained"] = True
    config["load_only_params"] = True
    config["F0_path"] = str(paths["repo"] / "Utils/JDC/bst.t7")
    config["ASR_config"] = str(paths["repo"] / "Utils/ASR/config.yml")
    config["ASR_path"] = str(paths["repo"] / "Utils/ASR/epoch_00080.pth")
    config["PLBERT_dir"] = str(paths["repo"] / "Utils/PLBERT")
    config.setdefault("preprocess_params", {})["sr"] = STYLE_SAMPLE_RATE
    data_params = config.setdefault("data_params", {})
    ood_texts, ood_min_length = _make_ood_texts(train_rows, int(data_params.get("min_length", 50)))
    data_params.update({
        "train_data": str(train_list),
        "val_data": str(val_list),
        "root_path": str(data_root),
        "OOD_data": str(ood_path),
        "min_length": ood_min_length,
    })
    ood_path.write_text("\n".join(f"{text}|0" for text in ood_texts) + "\n", encoding="utf-8")
    staged_config = root / "config_ft.yml"
    staged_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log_path = root / "training.log"
    command = [
        python_executable,
        "-m",
        "accelerate.commands.launch",
        "--mixed_precision=fp16",
        "--num_processes=1",
        str(paths["train_script"]),
        "--config_path",
        str(staged_config),
    ]
    result: dict[str, Any] = {
        "model_key": "styletts2",
        "model_label": "StyleTTS2",
        "family": "styletts2",
        "training_root": str(root),
        "dataset_dir": str(dataset),
        "styletts2_repo": str(paths["repo"]),
        "base_checkpoint": str(paths["checkpoint"]),
        "config": str(staged_config),
        "train_manifest": str(train_list),
        "validation_manifest": str(val_list),
        "log_path": str(log_path),
        "command": command,
        "gpu_total_gib": gpu_total_gib,
    }
    if dry_run:
        result["status"] = "dry-run"
        return result

    _notify(progress, "Starting official StyleTTS2 fine-tuning...")
    environment = os.environ.copy()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    if gpu_total_gib is not None and gpu_total_gib < 16:
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    timeout_seconds = int(environment.get("UFT_STYLETTS2_TIMEOUT_SECONDS", "5400"))
    if timeout_seconds < 60:
        raise ValueError("UFT_STYLETTS2_TIMEOUT_SECONDS must be at least 60.")
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(paths["repo"]),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        timed_out = threading.Event()
        def stop_on_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    return
                def force_stop() -> None:
                    if process.poll() is None:
                        try:
                            if hasattr(os, "killpg"):
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                        except ProcessLookupError:
                            pass
                threading.Timer(10, force_stop).start()
        timer = threading.Timer(timeout_seconds, stop_on_timeout)
        timer.start()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_stream.write(line)
                log_stream.flush()
                if stream_logs:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                _notify(progress, line.rstrip())
            return_code = process.wait()
        finally:
            timer.cancel()
    if timed_out.is_set():
        raise TimeoutError(f"StyleTTS2 fine-tuning exceeded {timeout_seconds} seconds and its subprocess group was stopped. See {log_path}.")
    if return_code:
        raise RuntimeError(
            f"StyleTTS2 fine-tuning exited with status {return_code}. See {log_path}. "
            "The upstream runner requires a CUDA device and compatible local assets."
        )

    checkpoints = sorted(log_dir.glob("epoch_2nd_*.pth"), key=lambda item: item.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"StyleTTS2 finished without an epoch_2nd checkpoint under {log_dir}")
    step_probe = subprocess.run(
        [python_executable, "-c", "import sys,torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('iters', 0)))", str(checkpoints[-1])],
        check=False,
        capture_output=True,
        text=True,
    )
    if step_probe.returncode:
        raise RuntimeError(f"Could not verify StyleTTS2 optimizer steps in {checkpoints[-1]}: {step_probe.stderr.strip()[-1000:]}")
    trained_steps = int(step_probe.stdout.strip().splitlines()[-1])
    if trained_steps < 1:
        raise RuntimeError(f"StyleTTS2 saved {checkpoints[-1]} without a completed training step.")
    ready = root / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    final_checkpoint = ready / "model.pth"
    shutil.copy2(checkpoints[-1], final_checkpoint)
    ready_config = ready / "config_ft.yml"
    trained_config = log_dir / staged_config.name
    config_source = staged_config
    if trained_config.is_file():
        trained_values = yaml.safe_load(trained_config.read_text(encoding="utf-8"))
        sigma_data = trained_values.get("model_params", {}).get("diffusion", {}).get("dist", {}).get("sigma_data")
        if isinstance(sigma_data, (int, float)) and math.isfinite(sigma_data):
            config_source = trained_config
    shutil.copy2(config_source, ready_config)
    artifacts = {
        **result,
        "checkpoint": str(final_checkpoint),
        "config": str(ready_config),
        "upstream_checkpoint": str(checkpoints[-1].resolve()),
        "artifacts_file": str(ready / "artifacts.json"),
        "status": "complete",
        "trained_steps": trained_steps,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inference_note": "UFT inference uses the official LibriTTS model flow; requires local StyleTTS2 source/assets and a speaker reference WAV.",
    }
    artifacts_file = ready / "artifacts.json"
    artifacts_file.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    _notify(progress, f"StyleTTS2 fine-tuning complete; checkpoint: {final_checkpoint}")
    return artifacts
