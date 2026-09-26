"""Standalone, guarded adapter for fine-tuning the upstream OmniVoice model.

The function in this module deliberately does not install packages or fetch
models. Install ``omnivoice`` (including its LoRA extra) in the active Python
environment and cache the base model/tokenizer before starting a run.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


OMNIVOICE_REPOSITORY = "https://github.com/k2-fsa/OmniVoice"
DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_AUDIO_TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"
GIB = 1024 ** 3


def _require_module(module: str, install_hint: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(
            f"OmniVoice dependency '{module}' is missing from {sys.executable}. "
            f"Install it in this environment first: {install_hint}. No install was attempted."
        )


def _local_hf_snapshot(repo_id: str) -> Path:
    """Resolve a cached HF snapshot only; never initiate a network download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "OmniVoice's huggingface_hub dependency is unavailable. Install the official "
            "OmniVoice package in this environment before running fine-tuning."
        ) from exc
    try:
        return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"Required checkpoint '{repo_id}' is not cached locally. Download/cache it "
            "explicitly before starting a run; this adapter never downloads model files."
        ) from exc


def _validate_manifest(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"OmniVoice JSONL manifest not found: {path}")
    count = 0
    raw_bytes = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_no}: {exc}") from exc
            missing = [field for field in ("id", "audio_path", "text") if not row.get(field)]
            if missing:
                raise ValueError(
                    f"Manifest line {line_no} is missing required field(s): {', '.join(missing)}"
                )
            audio = Path(row["audio_path"]).expanduser()
            if not audio.is_absolute():
                audio = (path.parent / audio).resolve()
            if not audio.is_file():
                raise FileNotFoundError(
                    f"Audio file on manifest line {line_no} does not exist: {audio}"
                )
            raw_bytes += audio.stat().st_size
            count += 1
    if count == 0:
        raise ValueError("OmniVoice JSONL manifest has no usable rows.")
    return count, raw_bytes


def _cuda_preflight(min_free_gib: float, min_total_gib: float) -> tuple[int, float, float]:
    _require_module("torch", "the PyTorch build matching your CUDA runtime")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "OmniVoice fine-tuning requires a supported GPU for tokenization/training. "
            "CUDA is unavailable in the active Python environment."
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This safe runner requires exactly one visible CUDA GPU; found "
            f"{torch.cuda.device_count()}. Set CUDA_VISIBLE_DEVICES to one device."
        )
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / GIB
    free, _ = torch.cuda.mem_get_info(0)
    free_gib = free / GIB
    if total < min_total_gib:
        raise RuntimeError(
            f"Visible GPU '{props.name}' has {total:.1f} GiB VRAM; this guarded "
            f"OmniVoice LoRA profile requires at least {min_total_gib:.1f} GiB total."
        )
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"GPU '{props.name}' has only {free_gib:.1f} GiB free of {total:.1f} GiB. "
            f"Free at least {min_free_gib:.1f} GiB before starting OmniVoice."
        )
    # Index 0 is the sole visible device; torch device properties do not expose
    # a stable public ``index`` attribute across torch versions.
    return 0, total, free_gib


def _run(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    progress: Callable[[str], None] | None,
) -> None:
    if progress:
        progress("Running: " + " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if progress:
                progress(line.rstrip())
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"OmniVoice command failed with exit code {return_code}: {' '.join(command)}. "
            f"Full output is saved at {log_path}."
        )


def run_omnivoice_finetune(
    train_jsonl: str | Path,
    output_dir: str | Path,
    *,
    dev_jsonl: str | Path | None = None,
    base_model: str = DEFAULT_MODEL_ID,
    audio_tokenizer: str = DEFAULT_AUDIO_TOKENIZER,
    steps: int = 1000,
    save_steps: int = 500,
    batch_tokens: int = 512,
    min_free_vram_gib: float = 12.0,
    min_total_vram_gib: float = 16.0,
    min_free_disk_gib: float = 20.0,
    max_dataset_audio_gib: float = 50.0,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Prepare data and run a single-GPU OmniVoice SDPA + LoRA fine-tune.

    ``train_jsonl`` and optional ``dev_jsonl`` use upstream JSONL rows with
    ``id``, ``audio_path``, and ``text`` fields (``language_id`` is optional).
    Checkpoints are capped to the last two, and model downloads are disabled.
    Returns the output directory containing the training run. The conservative
    default requires a 16 GiB GPU with at least 12 GiB currently free; it will
    reject a 12 GiB device before tokenization or training starts.
    """
    if steps < 1 or save_steps < 1:
        raise ValueError("steps and save_steps must be positive integers")
    if batch_tokens < 128 or batch_tokens > 512:
        raise ValueError("Safe single-GPU batch_tokens must be between 128 and 512")

    train_path = Path(train_jsonl).expanduser().resolve()
    train_count, train_audio_bytes = _validate_manifest(train_path)
    dev_path = Path(dev_jsonl).expanduser().resolve() if dev_jsonl else None
    dev_count, dev_audio_bytes = _validate_manifest(dev_path) if dev_path else (0, 0)
    dataset_gib = (train_audio_bytes + dev_audio_bytes) / GIB
    if dataset_gib > max_dataset_audio_gib:
        raise RuntimeError(
            f"Dataset audio totals {dataset_gib:.1f} GiB, above this runner's "
            f"{max_dataset_audio_gib:.1f} GiB safety limit."
        )

    _require_module("omnivoice", "pip install 'omnivoice[lora]'")
    _require_module("accelerate", "pip install 'omnivoice[lora]'")
    _require_module("peft", "pip install 'omnivoice[lora]'")
    _require_module("huggingface_hub", "pip install 'omnivoice[lora]'")
    _cuda_preflight(min_free_vram_gib, min_total_vram_gib)

    # Force local-only resolution before any work directory or subprocess is created.
    base_path = Path(base_model).expanduser()
    if not base_path.exists():
        base_path = _local_hf_snapshot(base_model)
    tokenizer_path = Path(audio_tokenizer).expanduser()
    if not tokenizer_path.exists():
        tokenizer_path = _local_hf_snapshot(audio_tokenizer)
    llm_path = _local_hf_snapshot("Qwen/Qwen3-0.6B")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    free_disk_gib = shutil.disk_usage(out).free / GIB
    # Tokenized shards plus two full training-state snapshots and working room.
    estimated_need_gib = max(min_free_disk_gib, 12.0 + dataset_gib * 1.5)
    if free_disk_gib < estimated_need_gib:
        raise RuntimeError(
            f"Output volume has {free_disk_gib:.1f} GiB free; this run requires an "
            f"estimated {estimated_need_gib:.1f} GiB safety reserve. Choose another volume "
            "or free disk space before training."
        )

    _require_module("omnivoice", "pip install 'omnivoice[lora]'")
    work = out / "omnivoice_work"
    token_dir = work / "tokens"
    train_dir = token_dir / "train"
    dev_dir = token_dir / "dev"
    work.mkdir(parents=True, exist_ok=True)
    config_dir = work / "config"
    config_dir.mkdir(exist_ok=True)
    train_token_jsonl = work / "train.jsonl"
    dev_token_jsonl = work / "dev.jsonl" if dev_path else None
    train_manifest = train_dir / "data.lst"
    dev_manifest = dev_dir / "data.lst" if dev_path else None

    # Pin offline mode so transitive HF loaders cannot unexpectedly fetch weights.
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # Store generated training configs under the caller-selected output directory.
    data_cfg = {
        "train": [{"manifest_path": [str(train_manifest)], "repeat": 1}],
        "dev": ([{"manifest_path": [str(dev_manifest)], "repeat": 1}] if dev_manifest else []),
    }
    # Start from the upstream train_config_finetune_lora.json so required model,
    # data, and LoRA fields remain aligned with the official TrainingConfig.
    train_cfg = {
        "llm_name_or_path": str(llm_path),
        "audio_vocab_size": 1025,
        "audio_mask_id": 1024,
        "num_audio_codebook": 8,
        "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
        "drop_cond_ratio": 0.1,
        "prompt_ratio_range": [0.0, 0.3],
        "mask_ratio_range": [0.0, 1.0],
        "language_ratio": 0.8,
        "use_pinyin_ratio": 0.0,
        "instruct_ratio": 0.0,
        "only_instruct_ratio": 0.0,
        "resume_from_checkpoint": None,
        "init_from_checkpoint": str(base_path),
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_modules_to_save": ["audio_embeddings", "audio_heads"],
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "steps": steps,
        "seed": 42,
        "warmup_type": "ratio",
        "warmup_ratio": 0.01,
        "warmup_steps": 0,
        "attn_implementation": "sdpa",
        # The SDPA batching code treats batch_tokens as a padded-token budget;
        # keep a single sample from exceeding that budget as well.
        "max_sample_tokens": min(512, batch_tokens),
        "min_sample_tokens": 50,
        "max_batch_size": 4,
        "batch_tokens": batch_tokens,
        "gradient_accumulation_steps": 1,
        "num_workers": 1,
        "mixed_precision": "bf16",
        "allow_tf32": True,
        "logging_steps": 10,
        "eval_steps": save_steps,
        "save_steps": save_steps,
        "keep_last_n_checkpoints": 2,
    }
    data_config_path = config_dir / "data.json"
    train_config_path = config_dir / "train.json"
    data_config_path.write_text(json.dumps(data_cfg, indent=2), encoding="utf-8")
    train_config_path.write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")

    if progress_callback:
        progress_callback(
            f"Validated {train_count} training and {dev_count} validation clips; "
            f"audio total {dataset_gib:.2f} GiB. Preparing token shards."
        )
    for split, source, shard_dir, token_jsonl in (
        ("train", train_path, train_dir, train_token_jsonl),
        ("dev", dev_path, dev_dir, dev_token_jsonl),
    ):
        if source is None or token_jsonl is None:
            continue
        # Rewrite relative audio paths against the input manifest before staging it.
        with source.open("r", encoding="utf-8") as incoming, token_jsonl.open("w", encoding="utf-8") as staged:
            for line in incoming:
                if not line.strip():
                    continue
                row = json.loads(line)
                audio = Path(row["audio_path"]).expanduser()
                if not audio.is_absolute():
                    audio = source.parent / audio
                row["audio_path"] = str(audio.resolve())
                staged.write(json.dumps(row, ensure_ascii=False) + "\n")
        shard_dir.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
            "--input_jsonl", str(token_jsonl),
            "--tar_output_pattern", str(shard_dir / "audios" / "shard-%06d.tar"),
            "--jsonl_output_pattern", str(shard_dir / "txts" / "shard-%06d.jsonl"),
            "--tokenizer_path", str(tokenizer_path),
            "--nj_per_gpu", "1",
            "--loader_workers", "1",
            "--shuffle", "True" if split == "train" else "False",
        ], out, env, work / "logs" / f"tokenize-{split}.log", progress_callback)

    if not train_manifest.is_file() or (dev_manifest and not dev_manifest.is_file()):
        raise RuntimeError("OmniVoice tokenization finished without producing the expected data.lst manifest.")
    _run([
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "-m", "omnivoice.cli.train",
        "--train_config", str(train_config_path),
        "--data_config", str(data_config_path),
        "--output_dir", str(out / "checkpoints"),
    ], out, env, work / "logs" / "train.log", progress_callback)
    if progress_callback:
        progress_callback(f"OmniVoice fine-tuning complete: {out / 'checkpoints'}")
    return out / "checkpoints"


__all__ = ["OMNIVOICE_REPOSITORY", "run_omnivoice_finetune"]
