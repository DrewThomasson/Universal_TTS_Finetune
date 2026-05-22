from __future__ import annotations

import os
import sys
import shutil
import urllib.request
import json
import subprocess
from pathlib import Path
import torch

def ensure_monotonic_align_compiled():
    """Auto-compiles the monotonic_align Cython extension for Piper training if not already compiled."""
    base_dir = Path(__file__).resolve().parent.parent / "piper" / "src" / "python" / "piper_train" / "vits" / "monotonic_align"
    target_dir = base_dir / "monotonic_align"
    target_dir.mkdir(exist_ok=True)
    
    # Check if compiled file exists in target_dir
    existing = list(target_dir.glob("core*.so")) + list(target_dir.glob("core*.pyd"))
    if existing:
        return
        
    print("Compiling monotonic_align Cython extension...")
    try:
        # Run setup.py in-place
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(base_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Move built extension files into the target monotonic_align/ directory
        built_files = list(base_dir.glob("core*.so")) + list(base_dir.glob("core*.pyd")) + list(base_dir.glob("core*.dylib"))
        for f in built_files:
            dest_file = target_dir / f.name
            if dest_file.exists():
                dest_file.unlink()
            shutil.move(str(f), str(target_dir / f.name))
        print("monotonic_align compiled successfully.")
    except Exception as e:
        print(f"Warning: Failed to compile monotonic_align Cython extension: {e}. Training might be slow or fail.")

def resolve_piper_checkpoint(language: str, quality: str = "medium") -> dict[str, str]:
    """Resolves the pre-trained checkpoint for a given language code.
    Tries HF API dynamically, falling back to a static catalog of common models.
    """
    lang = language.split("-")[0].split("_")[0].lower() # Normalize code (e.g. en-US -> en)
    
    # Static catalog mapping standard languages to reliable pre-trained models
    static_fallbacks = {
        "en": {
            "lang": "en",
            "locale": "en_US",
            "voice": "lessac",
            "quality": "medium",
            "ckpt": "epoch=2164-step=1355540.ckpt",
            "config": "config.json"
        },
        "es": {
            "lang": "es",
            "locale": "es_ES",
            "voice": "davefx",
            "quality": "medium",
            "ckpt": "epoch=5629-step=1605020.ckpt",
            "config": "config.json"
        },
        "de": {
            "lang": "de",
            "locale": "de_DE",
            "voice": "thorsten",
            "quality": "medium",
            "ckpt": "epoch=3135-step=2702056.ckpt",
            "config": "config.json"
        },
        "fr": {
            "lang": "fr",
            "locale": "fr_FR",
            "voice": "siwis",
            "quality": "medium",
            "ckpt": "epoch=3304-step=2050940.ckpt",
            "config": "config.json"
        }
    }
    
    try:
        # Tries to query Hugging Face API dynamically
        api_url = f"https://huggingface.co/api/datasets/rhasspy/piper-checkpoints/tree/main/{lang}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            locales = json.loads(response.read().decode())
            if locales and isinstance(locales, list):
                # Select first locale (e.g. en/en_US)
                locale_path = locales[0]["path"]
                
                # Query locale directory for voices
                voice_url = f"https://huggingface.co/api/datasets/rhasspy/piper-checkpoints/tree/main/{locale_path}"
                with urllib.request.urlopen(urllib.request.Request(voice_url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=5) as res:
                    voices = json.loads(res.read().decode())
                    if voices and isinstance(voices, list):
                        voice_path = voices[0]["path"]
                        
                        # Query voice directory for qualities
                        quality_url = f"https://huggingface.co/api/datasets/rhasspy/piper-checkpoints/tree/main/{voice_path}"
                        with urllib.request.urlopen(urllib.request.Request(quality_url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=5) as r:
                            qualities = json.loads(r.read().decode())
                            
                            # Attempt to find the requested quality or take first
                            selected_quality_path = None
                            for q in qualities:
                                if q["path"].endswith(quality):
                                    selected_quality_path = q["path"]
                                    break
                            if not selected_quality_path and qualities:
                                selected_quality_path = qualities[0]["path"]
                                
                            if selected_quality_path:
                                # Query selected quality directory for files
                                files_url = f"https://huggingface.co/api/datasets/rhasspy/piper-checkpoints/tree/main/{selected_quality_path}"
                                with urllib.request.urlopen(urllib.request.Request(files_url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=5) as rf:
                                    files = json.loads(rf.read().decode())
                                    ckpt_file = None
                                    config_file = "config.json"
                                    for f in files:
                                        if f["path"].endswith(".ckpt"):
                                            ckpt_file = Path(f["path"]).name
                                    
                                    if ckpt_file:
                                        parts = selected_quality_path.split("/")
                                        return {
                                            "lang": lang,
                                            "locale": parts[1] if len(parts) > 1 else "",
                                            "voice": parts[2] if len(parts) > 2 else "",
                                            "quality": parts[3] if len(parts) > 3 else quality,
                                            "ckpt": ckpt_file,
                                            "config": config_file
                                        }
    except Exception as e:
        print(f"HF API resolution failed ({e}). Falling back to static mappings.")
        
    # Return matched static config, or fallback to English lessac
    if lang in static_fallbacks:
        return static_fallbacks[lang]
    else:
        print(f"Language '{lang}' has no pre-trained checkpoint. Defaulting to English (en_US/lessac) as base model.")
        return static_fallbacks["en"]

def download_piper_checkpoint(checkpoint_info: dict[str, str], progress_callback=None) -> tuple[Path, Path]:
    """Downloads checkpoint ckpt and config files locally into models/piper/checkpoints/."""
    lang = checkpoint_info["lang"]
    locale = checkpoint_info["locale"]
    voice = checkpoint_info["voice"]
    quality = checkpoint_info["quality"]
    ckpt_filename = checkpoint_info["ckpt"]
    config_filename = checkpoint_info["config"]
    
    base_url = f"https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{lang}/{locale}/{voice}/{quality}"
    
    local_dir = Path(__file__).resolve().parent.parent / "models" / "piper" / "checkpoints" / lang / locale / voice / quality
    local_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = local_dir / ckpt_filename
    config_path = local_dir / config_filename
    
    def _download(url: str, dest: Path, label: str):
        if dest.exists() and dest.stat().st_size > 1000000:
            return
        if progress_callback:
            progress_callback(f"Downloading {label}...")
        print(f"Downloading {url} to {dest}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total_size > 0 and progress_callback:
                        percent = int(downloaded * 100 / total_size)
                        progress_callback(f"Downloading {label}: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                        
    _download(f"{base_url}/{config_filename}", config_path, f"{voice} config")
    _download(f"{base_url}/{ckpt_filename}", ckpt_path, f"{voice} checkpoint")
    
    return ckpt_path, config_path

def normalize_espeak_language(language_code: str) -> str:
    """Verifies and normalizes the language code for piper_phonemize.
    If the language code raises an error when phonemizing, it falls back to the base language code (e.g. en-gb -> en).
    """
    try:
        import piper_phonemize
        # Try phonemizing a dummy word to see if voice sets up successfully
        piper_phonemize.phonemize_espeak("test", language_code)
        return language_code
    except Exception as e:
        print(f"eSpeak voice setup failed for '{language_code}': {e}")
        # Try finding a base language code
        if "-" in language_code:
            base_code = language_code.split("-")[0]
            try:
                import piper_phonemize
                piper_phonemize.phonemize_espeak("test", base_code)
                print(f"Fell back to base voice: '{base_code}'")
                return base_code
            except Exception as e_base:
                print(f"eSpeak voice setup also failed for base voice '{base_code}': {e_base}")
        elif "_" in language_code:
            base_code = language_code.split("_")[0]
            try:
                import piper_phonemize
                piper_phonemize.phonemize_espeak("test", base_code)
                print(f"Fell back to base voice: '{base_code}'")
                return base_code
            except Exception as e_base:
                print(f"eSpeak voice setup also failed for base voice '{base_code}': {e_base}")

        # If base fails, or there is no delimiter, we can fall back to 'en'
        print("Falling back to default 'en' voice.")
        return "en"

def preprocess_piper_dataset(dataset_dir: Path, output_dir: Path, language_code: str, sample_rate: int):
    """Executes piper_train.preprocess as a python subprocess to prepare the LJSpeech dataset."""
    # Normalize language code to prevent eSpeak voice setup failures
    normalized_lang = normalize_espeak_language(language_code)
    
    piper_python_src = Path(__file__).resolve().parent.parent / "piper" / "src" / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(piper_python_src) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    
    cmd = [
        sys.executable,
        "-m", "piper_train.preprocess",
        "--language", normalized_lang,
        "--input-dir", str(dataset_dir),
        "--output-dir", str(output_dir),
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", str(sample_rate)
    ]
    print(f"Running Piper preprocessing: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

def train_piper_model(
    preprocessed_dir: Path,
    base_ckpt_path: Path,
    epochs: int,
    batch_size: int,
    quality: str = "medium",
    stream_logs: bool = True,
    progress_callback=None
) -> str:
    """Invokes pytorch-lightning training via piper_train as a subprocess."""
    piper_python_src = Path(__file__).resolve().parent.parent / "piper" / "src" / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(piper_python_src) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    
    # Configure accelerator based on availability (Mac GPUs use mps, otherwise cpu/gpu)
    devices = 1
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"
        
    cmd = [
        sys.executable,
        "-m", "piper_train",
        "--dataset-dir", str(preprocessed_dir),
        "--accelerator", accelerator,
        "--devices", str(devices),
        "--batch-size", str(batch_size),
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(epochs),
        "--resume_from_checkpoint", str(base_ckpt_path),
        "--checkpoint-epochs", "1",
        "--precision", "32",
        "--quality", quality
    ]
    
    if progress_callback:
        progress_callback("Launching Piper training...")
        
    print(f"Running training command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    log_lines = []
    if process.stdout:
        for line in process.stdout:
            log_lines.append(line)
            if stream_logs:
                sys.stdout.write(line)
                sys.stdout.flush()
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"Piper training subprocess failed with code {process.returncode}")
        
    return "".join(log_lines)

def export_piper_onnx(ckpt_path: Path, onnx_path: Path, config_path: Path):
    """Exports PyTorch Lightning checkpoint to ONNX format and copies its configuration file."""
    piper_python_src = Path(__file__).resolve().parent.parent / "piper" / "src" / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(piper_python_src) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    
    cmd = [
        sys.executable,
        "-m", "piper_train.export_onnx",
        str(ckpt_path),
        str(onnx_path)
    ]
    print(f"Running ONNX export: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    
    # Copy configuration file to match the onnx model path
    dest_config = Path(f"{onnx_path}.json")
    shutil.copy2(config_path, dest_config)
    print(f"ONNX model successfully exported to {onnx_path}")

def synthesize_piper(onnx_path: str, config_path: str, text: str, output_wav_path: str):
    """Generates speech wav file using ONNX Runtime and piper python_run scripts."""
    import wave
    
    piper_run_src = Path(__file__).resolve().parent.parent / "piper" / "src" / "python_run"
    if str(piper_run_src) not in sys.path:
        sys.path.insert(0, str(piper_run_src))
        
    from piper.voice import PiperVoice
    
    voice = PiperVoice.load(model_path=onnx_path, config_path=config_path)
    
    output_path = Path(output_wav_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with wave.open(str(output_path), "wb") as wav_file:
        voice.synthesize(text, wav_file)
        
    print(f"Audio successfully generated and saved to {output_wav_path}")
