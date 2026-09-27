"""F5-TTS orchestration without importing its optional dependencies into UFT."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path


def _run(mode, config, root, progress=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f'f5_{mode}_', suffix='.json', dir=root, delete=False) as temporary:
        request = Path(temporary.name)
    request.write_text(json.dumps(config), encoding='utf-8')
    python = config.get('python_executable') or os.environ.get('UFT_F5TTS_PYTHON')
    if not python:
        raise ValueError('Set UFT_F5TTS_PYTHON to a separate environment with f5-tts and CUDA PyTorch installed.')
    env = os.environ.copy()
    env.setdefault('CACHED_PATH_CACHE_ROOT', str(root / 'download_cache'))
    env.update(WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1', PYTHONUNBUFFERED='1')
    log_path = root / ('f5_train.log' if mode == 'train' else request.stem + '.log')
    command = [str(python), str(Path(__file__).with_name('f5tts_worker.py')), mode, str(request)]
    if config.get("device") == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if progress:
        progress(f'Running official F5-TTS {mode}; log: {log_path}')
    if mode == 'train':
        from utils.pipeline import register_active_process
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=root, start_new_session=True)
        if mode == 'train':
            register_active_process(process)
        try:
            offset = 0
            deadline = time.monotonic() + (5400 if mode == 'train' else 900)
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'F5-TTS {mode} exceeded its time limit; see {log_path}.')
                if progress:
                    with log_path.open(encoding='utf-8', errors='replace') as reader:
                        reader.seek(offset)
                        new_output = reader.read()
                        offset = reader.tell()
                    if new_output.strip():
                        progress(new_output[-1500:].strip())
                time.sleep(1)
            if process.returncode:
                excerpt = log_path.read_text(encoding='utf-8', errors='replace')[-5000:]
                raise RuntimeError(f'F5-TTS {mode} exited {process.returncode}:\n{excerpt}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if mode == 'train':
                register_active_process(None)
            request.unlink(missing_ok=True)


def train_f5tts(*, dataset_dir, training_root, language, epochs, batch_size, grad_accum,
                max_audio_seconds, restore_path=None, python_executable=None, dry_run=False, progress=None, device="auto"):
    if epochs < 1 or batch_size < 1 or grad_accum < 1 or not 0.3 <= max_audio_seconds <= 30:
        raise ValueError('F5-TTS requires positive epochs/batch/accumulation and an audio limit between 0.3 and 30 seconds.')
    config = dict(dataset_dir=str(dataset_dir), training_root=str(training_root), language=language,
                  epochs=epochs, batch_size=batch_size, grad_accum=grad_accum, max_audio_seconds=max_audio_seconds,
                  restore_path=restore_path, python_executable=python_executable or os.environ.get('UFT_F5TTS_PYTHON'), device=device)
    if restore_path:
        raise ValueError('F5-TTS local checkpoint overrides and resume are not supported; use the official base.')
    if dry_run:
        return dict(config, model_key='f5_tts', family='f5_tts', status='dry-run')
    _run('train', config, training_root, progress)
    artifacts_file = Path(training_root) / 'ready' / 'artifacts.json'
    artifacts = json.loads(artifacts_file.read_text(encoding='utf-8'))
    artifacts['artifacts_file'] = str(artifacts_file)
    return artifacts


def synthesize_f5tts(artifacts, text, language, reference_wav, output_file, progress=None, device="auto"):
    from utils.model_registry import pretrained_model_choices
    if not pretrained_model_choices('f5_tts', language):
        raise ValueError(f'F5-TTS v1 does not support language {language}. Select en or zh-cn.')
    if reference_wav and Path(reference_wav).resolve() != Path(artifacts['reference_wav']).resolve():
        raise ValueError('F5-TTS uses the packaged reference audio and exact transcript; a different speaker WAV is unsupported.')
    config = dict(artifacts, text=text, output_file=str(output_file), device=device)
    _run('infer', config, Path(output_file).parent, progress)
    if not Path(output_file).is_file() or Path(output_file).stat().st_size <= 44:
        raise RuntimeError('F5-TTS inference produced no audio.')
    return Path(output_file)
