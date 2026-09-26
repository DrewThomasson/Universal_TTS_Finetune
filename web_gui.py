from __future__ import annotations

# Patch pkgutil.ImpImporter and importlib.machinery.FileFinder.find_module for Python 3.12 compatibility with older pkg_resources / setuptools
import pkgutil
import importlib.machinery

if not hasattr(pkgutil, "ImpImporter"):
    class DummyImpImporter:
        pass
    pkgutil.ImpImporter = DummyImpImporter

if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def find_module_shim(self, fullname, path=None):
        spec = self.find_spec(fullname, path)
        return spec.loader if spec is not None else None
    importlib.machinery.FileFinder.find_module = find_module_shim

# Patch PyTorch 2.6+ to default to weights_only=False in torch.load for compatibility with older checkpoints
try:
    import torch
    if hasattr(torch, "load"):
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
except ImportError:
    pass

import argparse
import os
import re
import json
from pathlib import Path

import gradio as gr

from utils.pipeline import (
    default_test_output,
    dropdown_choices,
    find_latest_artifacts,
    format_exception,
    load_artifacts,
    prepare_dataset,
    synthesize,
    train_model,
    pause_training,
    resume_training,
)
from utils.model_registry import MMS_LANGUAGES, OMNIVOICE_LANGUAGES, XTTS_LANGUAGES, pretrained_model_choices
from utils.asr import MMS_ASR_LANGUAGES
from utils.e2a_export import export_e2a_zip
from utils.language_support import coqui_phoneme_language

LANGUAGE_CHOICES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "ko",
    "ja",
    "hi",
    "bg", "da", "et", "ga", "uk", "fa", "be", "el", "fi",
    "hr", "lt", "lv", "mt", "ro", "sk", "sl", "sr", "sv", "ca",
    "cy", "is", "ka", "kk", "lb", "ne", "no", "sw", "ur", "vi",
]
# Published Piper training checkpoints include these languages in addition to
# the shared language list. zh-cn in the shared list resolves to Piper's zh family.
PIPER_LANGUAGE_CHOICES = sorted(set(LANGUAGE_CHOICES) | {
    "bn", "he", "id", "ku", "ml", "mr", "te", "th",
})
VITS_LANGUAGE_CHOICES = sorted(set(LANGUAGE_CHOICES) | {"bn"})
XTTS_LANGUAGE_CHOICES = {
    key: [language for language in LANGUAGE_CHOICES if language in supported]
    for key, supported in XTTS_LANGUAGES.items()
}
WHISPER_CHOICES = ["large-v3", "large-v2", "large", "distil-large-v3", "distil-large-v2", "medium", "medium.en", "small", "small.en", "base", "base.en", "tiny", "tiny.en"]
MODEL_CHOICES = [(label, key) for key, label in dropdown_choices()]
INFERENCE_MODEL_CHOICES = list(MODEL_CHOICES)
MMS_LANGUAGE_CHOICES = [(f"{name} ({code})", code) for code, name in MMS_LANGUAGES.items()]
OMNIVOICE_LANGUAGE_CHOICES = [(f"{name} ({code})", code) for code, name in OMNIVOICE_LANGUAGES.items()]
DATASET_LANGUAGE_CHOICES = LANGUAGE_CHOICES + [
    (f"{MMS_LANGUAGES.get(code, code)} ({code})", code)
    for code in sorted(MMS_ASR_LANGUAGES) if code not in LANGUAGE_CHOICES
]


def _phonemizer_language_choices(choices):
    supported = []
    for language in choices:
        try:
            coqui_phoneme_language(language)
            supported.append(language)
        except ValueError:
            pass
    return supported


def update_finetune_language_choices(model_key):
    if model_key == "styletts2":
        return gr.update(choices=["en"], value="en")
    if model_key == "omnivoice":
        return gr.update(choices=OMNIVOICE_LANGUAGE_CHOICES, value="en")
    if model_key in XTTS_LANGUAGE_CHOICES:
        return gr.update(choices=XTTS_LANGUAGE_CHOICES[model_key], value="en")
    if model_key == "mms_vits":
        return gr.update(choices=MMS_LANGUAGE_CHOICES, value="eng")
    if model_key == "piper":
        return gr.update(choices=PIPER_LANGUAGE_CHOICES, value="en")
    if model_key == "vits_tts":
        choices = _phonemizer_language_choices(VITS_LANGUAGE_CHOICES)
    elif model_key == "align_tts":
        choices = ["en"]
    else:
        choices = _phonemizer_language_choices(LANGUAGE_CHOICES)
    return gr.update(choices=choices, value="en" if "en" in choices else (choices[0] if choices else None))


class PreprocessProgressTracker:
    def __init__(self, progress_bar: gr.Progress):
        self.progress_bar = progress_bar
        self.last_fraction = 0.0
        
    def __call__(self, message: str) -> None:
        if "complete" in message.lower() or "finished" in message.lower():
            self.last_fraction = 1.0
            self.progress_bar(1.0, desc=message)
            return
            
        match = re.search(r"Processing\s+(\d+)\s*/\s*(\d+)", message)
        if match:
            curr = int(match.group(1))
            total = int(match.group(2))
            self.last_fraction = 0.8 * (curr / total)
            self.progress_bar(self.last_fraction, desc=message)
            return
            
        match2 = re.search(r"Extracting voice blueprints:\s*(\d+)\s*/\s*(\d+)", message)
        if match2:
            curr = int(match2.group(1))
            total = int(match2.group(2))
            self.last_fraction = 0.8 + 0.15 * (curr / total)
            self.progress_bar(self.last_fraction, desc=message)
            return
            
        self.progress_bar(self.last_fraction, desc=message[:60] + "..." if len(message) > 60 else message)


class TrainingProgressTracker:
    def __init__(self, progress_bar: gr.Progress, total_epochs: int):
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_epoch = None
        
    def __call__(self, log_line: str) -> None:
        if "complete" in log_line.lower() or "finished" in log_line.lower():
            self.progress_bar(1.0, desc=log_line.strip())
            return
            
        epoch_match = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", log_line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            total_num = int(epoch_match.group(2))
            if self.start_epoch is None:
                self.start_epoch = epoch_num
            self.current_epoch = epoch_num
            self.total_epochs = max(total_num - self.start_epoch, 1)
            relative_epoch = self.current_epoch - self.start_epoch
        else:
            epoch_match2 = re.search(r"Epoch\s*:\s*(\d+)", log_line, re.IGNORECASE)
            if not epoch_match2:
                epoch_match2 = re.search(r"epoch\s*=\s*(\d+)", log_line, re.IGNORECASE)
            if epoch_match2:
                epoch_num = int(epoch_match2.group(1))
                if self.start_epoch is None:
                    self.start_epoch = epoch_num
                self.current_epoch = epoch_num
                relative_epoch = self.current_epoch - self.start_epoch
            else:
                relative_epoch = None
                
        if relative_epoch is not None and self.total_epochs > 0:
            fraction = min(max(relative_epoch / self.total_epochs, 0.0), 1.0)
            self.progress_bar(fraction, desc=f"Training: Epoch {self.current_epoch}/{self.total_epochs + (self.start_epoch or 0)}")
        else:
            clean = log_line.strip()
            if clean:
                desc = clean[:60] + "..." if len(clean) > 60 else clean
                if self.total_epochs > 0 and self.current_epoch > 0:
                    start = self.start_epoch or 0
                    rel = self.current_epoch - start
                    fraction = min(max(rel / self.total_epochs, 0.0), 1.0)
                else:
                    fraction = 0.0
                self.progress_bar(fraction, desc=desc)


def list_datasets(output_root: str | None) -> list[str]:
    if not output_root:
        return []
    try:
        base = Path(output_root) / "dataset"
        if not base.exists():
            return []
        paths = []
        for p in base.iterdir():
            if p.is_dir() and ((p / "metadata.csv").exists() or (p / "metadata_train.csv").exists()):
                paths.append(str(p.resolve()))
        return sorted(paths)
    except Exception:
        return []


def list_trained_models(output_root: str | None, model_key: str | None) -> list[tuple[str, str]]:
    if not output_root:
        return []
    try:
        base = Path(output_root) / "training_runs"
        if not base.exists():
            return []
        
        choices = []
        search_dirs = [base / model_key] if model_key else list(base.iterdir())
        
        for model_dir in search_dirs:
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                artifacts_file = run_dir / "ready" / "artifacts.json"
                if artifacts_file.exists():
                    label = f"{model_dir.name} - {run_dir.name}"
                    choices.append((label, str(artifacts_file.resolve())))
        return sorted(choices, key=lambda x: x[0], reverse=True)
    except Exception:
        return []


def get_adaptive_defaults(model_key: str, dataset_dir: gr.Dropdown | str | None) -> tuple[int, int]:
    if model_key == "omnivoice":
        return 10, 1
    if model_key == "styletts2":
        return 10, 2
    epochs = 10
    batch_size = 8
    
    resolved_dir = None
    if dataset_dir:
        val = getattr(dataset_dir, "value", dataset_dir)
        if isinstance(val, str) and val.strip():
            resolved_dir = val.strip()
            
    duration_seconds = 0.0
    if resolved_dir:
        try:
            info_file = Path(resolved_dir) / "dataset_info.json"
            if info_file.exists():
                info = json.loads(info_file.read_text(encoding="utf-8"))
                duration_seconds = float(info.get("total_audio_seconds", 0.0))
        except Exception:
            pass
            
    is_piper = "piper" in model_key.lower() if model_key else False
    
    if is_piper:
        if duration_seconds == 0:
            epochs = 100
            batch_size = 8
        elif duration_seconds < 120:
            epochs = 300
            batch_size = 8
        elif duration_seconds < 600:
            epochs = 150
            batch_size = 8
        else:
            epochs = 80
            batch_size = 16
    else:
        if duration_seconds == 0:
            epochs = 10
            batch_size = 8
        elif duration_seconds < 120:
            epochs = 12
            batch_size = 4
        elif duration_seconds < 600:
            epochs = 8
            batch_size = 4
        else:
            epochs = 5
            batch_size = 8
            
    return epochs, batch_size


def update_dataset_choices(out_root: str | None) -> gr.Dropdown:
    choices = list_datasets(out_root)
    return gr.update(choices=choices)


def update_trained_models(out_root: str | None, model_key: str | None) -> gr.Dropdown:
    choices = list_trained_models(out_root, model_key)
    val = choices[0][1] if choices else None
    return gr.update(choices=choices, value=val)


def update_resume_models(out_root: str | None, model_key: str | None) -> gr.Dropdown:
    if model_key in {"styletts2", "omnivoice"}:
        return gr.update(choices=[("None", "")], value="", interactive=False)
    choices = [("None", "")] + list_trained_models(out_root, model_key)
    return gr.update(choices=choices, value="", interactive=True)


def resolve_resume_checkpoint(artifacts_file_path: str) -> str:
    if not artifacts_file_path:
        return ""
    try:
        with open(artifacts_file_path, "r", encoding="utf-8") as f:
            artifacts = json.load(f)
        family = artifacts.get("family")
        training_root = Path(artifacts.get("training_root"))
        
        if family == "piper":
            # Search for .ckpt files in the training root (or preprocessed/lightning_logs)
            from utils.pipeline import _latest_matching_file
            ckpt = _latest_matching_file(training_root, ["**/*.ckpt", "*.ckpt"])
            if ckpt:
                return str(ckpt.resolve())
        else:
            # Search for best_model.pth or other .pth files in workspace
            from utils.pipeline import _latest_matching_file
            pth = _latest_matching_file(training_root / "workspace", ["**/best_model.pth", "**/*.pth"])
            if pth:
                return str(pth.resolve())
            # Fallback to ready checkpoint if workspace is cleaned up or empty
            ready_pth = Path(artifacts.get("checkpoint"))
            if ready_pth.exists():
                return str(ready_pth.resolve())
    except Exception as e:
        print(f"Error resolving resume checkpoint: {e}")
    return ""


def _path_value(value):
    return getattr(value, "name", value) if value else None


def _gradio_progress(progress: gr.Progress | None):
    if progress is None:
        return None

    def callback(message: str) -> None:
        progress(0, desc=message)

    return callback


def _clean_audio_path(path_val):
    if not path_val:
        return None
    try:
        p = Path(path_val)
        if p.exists() and p.is_file():
            return str(p.resolve())
    except Exception:
        pass
    return None


def preprocess_dataset(
    audio_files, audio_dir, transcript_file, language, whisper_model, asr_backend, out_path, dataset_name, diarize_speakers,
    expected_speakers=0, diarize_threshold=0.3,
    generate_synthetic=False, synthetic_audio_file=None, synthetic_vtt_file=None,
    auto_split_sentences=True,
    progress=gr.Progress()
):
    try:
        tracker = PreprocessProgressTracker(progress)
        
        if generate_synthetic:
            if not synthetic_audio_file:
                raise ValueError("Synthetic data import active, but no synthesized audiobook file was uploaded.")
            if not synthetic_vtt_file:
                raise ValueError("Synthetic data import active, but no matching .vtt file was uploaded.")
            
            resolved_audio = _path_value(synthetic_audio_file)
            resolved_vtt = _path_value(synthetic_vtt_file)
            
            audio_files = [resolved_audio]
            audio_dir = None
            transcript_file = resolved_vtt
            diarize_speakers = False

        result = prepare_dataset(
            output_root=out_path,
            audio_files=audio_files,
            audio_dir=audio_dir or None,
            transcript_file=_path_value(transcript_file),
            language=language,
            whisper_model_name=whisper_model,
            asr_backend=asr_backend,
            dataset_name=dataset_name or "LJSpeech-1.1",
            diarize_speakers=diarize_speakers,
            expected_speakers=int(expected_speakers or 0),
            diarize_threshold=float(diarize_threshold or 0.3),
            auto_split_sentences=auto_split_sentences,
            progress=tracker,
        )
        
        speakers_list = result.get("all_speakers", [])
        speaker_choices = []
        
        if speakers_list:
            for s in speakers_list:
                dir_name = Path(s["dataset_dir"]).name
                label = f"{dir_name} (Duration: {s['total_audio_seconds']}s, Clips: {s['created_sample_count']})"
                speaker_choices.append((label, s["dataset_dir"]))
            
            message = f"Dataset split into {len(speakers_list)} speakers. Select speaker below to preview and activate."
            default_speaker_dir = speakers_list[0]["dataset_dir"]
            default_ref = speakers_list[0]["reference_wav"]
            default_info = f"**Dataset path**: `{default_speaker_dir}`\n**Duration**: {speakers_list[0]['total_audio_seconds']} seconds\n**Total clips**: {speakers_list[0]['created_sample_count']}"
        else:
            message = f"Dataset ready with {result['created_sample_count']} samples at {result['dataset_dir']}"
            default_speaker_dir = result["dataset_dir"]
            default_ref = result["reference_wav"]
            default_info = f"**Dataset path**: `{default_speaker_dir}`\n**Duration**: {result['total_audio_seconds']} seconds\n**Total clips**: {result['created_sample_count']}"
            
        choices = list_datasets(out_path)
        if default_speaker_dir not in choices:
            choices.append(default_speaker_dir)
            choices = sorted(choices)

        show_speakers = gr.update(visible=bool(speakers_list), choices=speaker_choices, value=default_speaker_dir if speakers_list else None)
        show_container = gr.update(visible=bool(speakers_list))
        return (
            message,
            default_speaker_dir,
            result["metadata_train"],
            result["metadata_val"],
            default_ref,
            gr.update(choices=choices, value=default_speaker_dir),
            _clean_audio_path(default_ref),
            show_speakers,
            show_container,
            _clean_audio_path(default_ref),
            default_info,
            speakers_list,
        )
    except Exception as exc:
        return (
            format_exception(exc), "", "", "", "",
            gr.update(choices=list_datasets(out_path), value=None), None,
            gr.update(visible=False, choices=[]), gr.update(visible=False),
            None, "", []
        )


def preprocess_re_diarize(dataset_dir, expected_speakers, diarize_threshold, out_path, progress=gr.Progress()):
    try:
        if not dataset_dir:
            raise ValueError("No dataset directory selected. Please select a valid dataset directory.")
        tracker = PreprocessProgressTracker(progress)
        from utils.pipeline import re_diarize_dataset
        result = re_diarize_dataset(
            dataset_dir=dataset_dir,
            expected_speakers=int(expected_speakers or 0),
            diarize_threshold=float(diarize_threshold or 0.35),
            progress=tracker,
        )
        
        speakers_list = result.get("all_speakers", [])
        speaker_choices = []
        
        if speakers_list:
            for s in speakers_list:
                dir_name = Path(s["dataset_dir"]).name
                label = f"{dir_name} (Duration: {s['total_audio_seconds']}s, Clips: {s['created_sample_count']})"
                speaker_choices.append((label, s["dataset_dir"]))
            
            message = f"Dataset re-diarized into {len(speakers_list)} speakers. Select speaker below to preview and activate."
            default_speaker_dir = speakers_list[0]["dataset_dir"]
            default_ref = speakers_list[0]["reference_wav"]
            default_info = f"**Dataset path**: `{default_speaker_dir}`\n**Duration**: {speakers_list[0]['total_audio_seconds']} seconds\n**Total clips**: {speakers_list[0]['created_sample_count']}"
        else:
            message = f"Dataset ready with {result['created_sample_count']} samples at {result['dataset_dir']}"
            default_speaker_dir = result["dataset_dir"]
            default_ref = result["reference_wav"]
            default_info = f"**Dataset path**: `{default_speaker_dir}`\n**Duration**: {result['total_audio_seconds']} seconds\n**Total clips**: {result['created_sample_count']}"
            
        choices = list_datasets(out_path)
        if default_speaker_dir not in choices:
            choices.append(default_speaker_dir)
            choices = sorted(choices)

        show_speakers = gr.update(visible=bool(speakers_list), choices=speaker_choices, value=default_speaker_dir if speakers_list else None)
        show_container = gr.update(visible=bool(speakers_list))
        return (
            message,
            default_speaker_dir,
            result["metadata_train"],
            result["metadata_val"],
            default_ref,
            gr.update(choices=choices, value=default_speaker_dir),
            _clean_audio_path(default_ref),
            show_speakers,
            show_container,
            _clean_audio_path(default_ref),
            default_info,
            speakers_list,
        )
    except Exception as exc:
        return (
            format_exception(exc), "", "", "", "",
            gr.update(choices=list_datasets(out_path), value=None), None,
            gr.update(visible=False, choices=[]), gr.update(visible=False),
            None, "", []
        )


def run_training(model_key, dataset_dir, language, num_epochs, batch_size, grad_accum, out_path, max_audio_length, restore_path, use_pretrained, pretrained_model_id, extra_overrides_json, sample_epoch_interval=0, sample_text="", progress=gr.Progress()):
    try:
        tracker = TrainingProgressTracker(progress, int(num_epochs))
        result = train_model(
            model_key=model_key,
            output_root=out_path,
            dataset_dir=dataset_dir or None,
            language=language,
            epochs=int(num_epochs),
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            max_audio_seconds=int(max_audio_length),
            restore_path=restore_path or None,
            use_pretrained=use_pretrained,
            pretrained_model_id=pretrained_model_id if use_pretrained and not restore_path else None,
            extra_overrides_json=extra_overrides_json or None,
            progress=tracker,
            sample_epoch_interval=int(sample_epoch_interval),
            sample_text=sample_text,
        )
        message = f"Training finished. Ready artifacts saved in {Path(result['training_root']) / 'ready'}"
        
        updated_models = list_trained_models(out_path, model_key)
        new_val = updated_models[0][1] if updated_models else None
        
        return (
            message,
            result["training_root"],
            result["artifacts_file"],
            result["checkpoint"],
            result["config"],
            result.get("reference_wav") or None,
            result["artifacts_file"],
            _clean_audio_path(result.get("reference_wav")),
            model_key,
            gr.update(choices=updated_models, value=new_val),
            gr.update(choices=[("None", "")] + updated_models, value=""),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", None, "", None, model_key, gr.update(), gr.update()


def locate_artifacts(out_path, model_key):
    try:
        artifacts = find_latest_artifacts(out_path, model_key=model_key or None)
        updated_models = list_trained_models(out_path, model_key)
        new_val = artifacts["artifacts_file"]
        return (
            f"Loaded latest artifacts for {artifacts['model_label']}",
            artifacts["training_root"],
            artifacts["artifacts_file"],
            artifacts["checkpoint"],
            artifacts["config"],
            artifacts.get("reference_wav") or None,
            artifacts["artifacts_file"],
            _clean_audio_path(artifacts.get("reference_wav")),
            artifacts["model_key"],
            gr.update(choices=updated_models, value=new_val),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", "", "", None, model_key, gr.update()


def inspect_artifacts(artifacts_path, model_key):
    try:
        artifacts = load_artifacts(artifacts_path, model_key=model_key or None)
        return (
            f"Artifacts loaded for {artifacts['model_label']}",
            artifacts["training_root"],
            artifacts["artifacts_file"],
            artifacts["checkpoint"],
            artifacts["config"],
            artifacts.get("reference_wav") or None,
            _clean_audio_path(artifacts.get("reference_wav")),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", None, None


def run_inference(artifacts_path, model_key, language, tts_text, speaker_audio_file, out_path, progress=gr.Progress()):
    try:
        result = synthesize(
            artifacts_path_or_dir=artifacts_path,
            model_key=model_key or None,
            text=tts_text,
            language=language,
            speaker_wav=speaker_audio_file or None,
            output_file=default_test_output(out_path),
            progress=_gradio_progress(progress),
        )
        return "Speech generated.", _clean_audio_path(result["output_file"]), _clean_audio_path(result.get("speaker_wav"))
    except ValueError as exc:
        # Display validation/user errors cleanly in the GUI status
        return f"Error: {exc}", None, None
    except Exception as exc:
        return format_exception(exc), None, None


def export_for_e2a(artifacts_path, out_path):
    try:
        artifacts = load_artifacts(artifacts_path)
        destination = Path(out_path).expanduser().resolve() / "exports" / f"{artifacts['model_key']}-{Path(artifacts['training_root']).name}-e2a.zip"
        result = export_e2a_zip(artifacts, str(destination))
        return f"E2A upload ZIP created: {', '.join(result['files'])}", result["zip_file"]
    except Exception as exc:
        return f"E2A export failed: {exc}", None


def on_model_change(selected_model):
    try:
        from utils.model_registry import get_model_spec
        spec = get_model_spec(selected_model)
        req = spec.requires_speaker_wav or selected_model == "styletts2"
    except Exception:
        req = False
    return gr.update(visible=req), gr.update(visible=req)


def on_select_speaker(selected_dir, speakers_state):
    if not selected_dir or not speakers_state:
        return gr.update(), "", None, "", gr.update()
    
    speaker_info = next((s for s in speakers_state if s["dataset_dir"] == selected_dir), None)
    if not speaker_info:
        return gr.update(), "", None, "", gr.update()
        
    info_md = f"**Dataset path**: `{selected_dir}`\n**Duration**: {speaker_info['total_audio_seconds']} seconds\n**Total clips**: {speaker_info['created_sample_count']}"
    ref_wav = speaker_info["reference_wav"]
    
    return selected_dir, ref_wav, _clean_audio_path(ref_wav), info_md, gr.update(value=selected_dir)


def select_trained_model(val):
    return val


def on_training_params_change(model_key, dataset_dir):
    epochs, batch_size = get_adaptive_defaults(model_key, dataset_dir)
    return epochs, gr.update(value=batch_size, interactive=model_key not in {"styletts2", "omnivoice"})


def update_checkpoint_choices(model_key, language):
    if model_key == "piper":
        from utils.piper_utils import list_piper_checkpoint_choices, resolve_piper_checkpoint
        checkpoints = list(list_piper_checkpoint_choices(language))
        if checkpoints:
            preferred = resolve_piper_checkpoint(language)["id"]
            checkpoints.sort(key=lambda item: (item["id"] != preferred, item["id"]))
        choices = [
            (f"{item['locale']} · {item['voice'].replace('_', ' ').title()} · {item['quality']}", item["id"])
            for item in checkpoints
        ]
    else:
        choices = [
            (f"{parts[2].replace('_', ' ').title()} · {parts[3].replace('_', ' ').title()}" if len(parts) >= 4 else model_id, model_id)
            for model_id in pretrained_model_choices(model_key, language)
            for parts in [model_id.split("/")]
        ]
    return gr.update(choices=choices, value=choices[0][1] if choices else None, interactive=bool(choices))


def update_training_options(model_key, language, use_pretrained, pretrained_model_id=None):
    try:
        from utils.model_registry import get_model_spec
        spec = get_model_spec(model_key)
        model_label = spec.label
        choices = pretrained_model_choices(model_key, language)
        official_model_id = pretrained_model_id if pretrained_model_id in choices else (choices[0] if choices else None)
        family = spec.family
    except Exception as exc:
        return f"Error loading model spec: {exc}", gr.update()

    if family == "styletts2":
        if language != "en":
            return "❌ StyleTTS2 currently supports English fine-tuning only.", gr.update(value=True, interactive=False)
        return (
            "🟡 **StyleTTS2** uses the official LibriTTS base checkpoint. Set `UFT_STYLETTS2_REPO` "
            "to an official StyleTTS2 checkout and `UFT_STYLETTS2_CHECKPOINT` to a local base checkpoint "
            "before training. Its optional dependencies must be installed separately. Batch size 2 and gradient "
            "accumulation 1 are required; inference uses the local official LibriTTS runtime, and E2A export is unavailable.",
            gr.update(value=True, interactive=False),
        )

    if family == "omnivoice":
        if not choices:
            return f"❌ OmniVoice does not publish support for `{language}`.", gr.update(value=True, interactive=False)
        return (
            f"🟡 **OmniVoice** uses `{official_model_id}` for the published language ID `{language}`. "
            "Set `UFT_OMNIVOICE_PYTHON` to the optional OmniVoice environment. The base model, audio tokenizer, "
            "and Qwen3-0.6B must already be cached there; UFT will not download them. Batch size and gradient "
            "accumulation are fixed at 1. Inference uses the official LoRA API in that environment. E2A export is unavailable.",
            gr.update(value=True, interactive=False),
        )

    if family == "tts" and model_key != "align_tts":
        try:
            coqui_phoneme_language(language)
        except ValueError as exc:
            return f"❌ **{model_label}** cannot train `{language}`: {exc}", gr.update(value=False, interactive=False)
    if model_key == "align_tts" and language != "en":
        return "❌ Align TTS currently uses an English-only character vocabulary. Choose `en`.", gr.update(value=False, interactive=False)

    # 1. XTTS family
    if family == "xtts":
        if not choices:
            return f"❌ **{model_label}** does not support `{language}`. Choose another language or model.", gr.update(value=False, interactive=False)
        msg = f"🟢 **{model_label}** supports `{language}` using a multilingual checkpoint.\n\n"
        msg += f"Fine-tuning requires the official multilingual checkpoint: `{official_model_id}` (or a local restore checkpoint)."
        return msg, gr.update(value=True, interactive=False)

    # 2. Piper family
    elif family == "piper":
        from utils.piper_utils import normalize_espeak_language, resolve_piper_checkpoint
        try:
            lang = language.split("-")[0].split("_")[0].lower()
            selected_id = pretrained_model_id if pretrained_model_id and pretrained_model_id.startswith(f"piper:{lang}/") else None
            checkpoint_info = resolve_piper_checkpoint(language, checkpoint_id=selected_id)
        except (LookupError, ValueError) as exc:
            checkpoint_info = None
            checkpoint_error = exc
        voice_language = checkpoint_info["locale"].lower().replace("_", "-") if checkpoint_info else language
        try:
            phoneme_language = normalize_espeak_language(voice_language)
        except ValueError as exc:
            return f"❌ **Piper TTS** cannot train `{language}`: {exc}", gr.update(value=False, interactive=False)
        if checkpoint_info is None:
            return (
                f"🟡 **Piper TTS** has no matching training checkpoint for `{language}`: {checkpoint_error}. "
                "Training from scratch remains available, but usually needs hours of audio and much longer training for intelligible speech.",
                gr.update(value=False, interactive=False),
            )
        msg = f"🟢 **Piper TTS** has a `{language}` training checkpoint: `{checkpoint_info['id']}`.\n\n"
        if phoneme_language != voice_language:
            msg += f"Piper will use eSpeak voice `{phoneme_language}` for `{voice_language}`.\n\n"
        if use_pretrained:
            msg += "Fine-tuning will download and load this checkpoint."
        else:
            msg += "Training will start from scratch. A larger dataset (hours of audio) and much longer training are usually needed for intelligible speech."
        return msg, gr.update(interactive=True)

    # 3. MMS checkpoints use a published language-specific generator and vocab.
    elif family == "mms":
        if not official_model_id:
            return f"❌ **{model_label}** has no published checkpoint for `{language}`.", gr.update(value=False, interactive=False)
        return (
            f"🟢 **{model_label}** uses `{official_model_id}`. Meta publishes these checkpoints under CC BY-NC 4.0. "
            "Keep pretrained loading enabled to fine-tune this language's generator and vocabulary.",
            gr.update(value=True, interactive=False),
        )

    # 4. Single-language models
    else:
        if official_model_id:
            msg = f"🟢 **{model_label}** has a pre-trained `{language}` checkpoint mapped: `{official_model_id}`.\n\n"
            if use_pretrained:
                msg += "Fine-tuning will download and use this pre-trained base model."
            else:
                msg += (
                    "**Training from scratch** (random initialization). The model weights start blank, "
                    "so it must learn this language and speech from your dataset. Expect hours of audio "
                    "and substantially longer training for intelligible speech."
                )
            if family == "tts" and model_key != "align_tts":
                msg += "\n\nUFT checks the dataset's phoneme symbols before training. If the mapped checkpoint cannot represent them, select training from scratch."
            return msg, gr.update(interactive=True)
        else:
            msg = f"🟡 **{model_label}** has no pre-trained checkpoint mapped for `{language}`.\n\n"
            msg += "**Training from scratch** (random initialization) is required. The model must learn this language and speech from your dataset; expect hours of audio and substantially longer training for intelligible speech.\n\n"
            msg += f"The backend adapts the recipe and phoneme vocabulary to the `{language}` dataset when available."
            return msg, gr.update(value=False, interactive=False)


def preprocess_and_train(
    audio_files, audio_dir, transcript_file, language, whisper_model, asr_backend, out_path, dataset_name, diarize_speakers,
    expected_speakers, diarize_threshold,
    generate_synthetic, synthetic_audio_file, synthetic_vtt_file,
    model_key, train_language, num_epochs, batch_size, grad_accum, max_audio_length, restore_path, use_pretrained, pretrained_model_id, extra_overrides_json,
    sample_epoch_interval, sample_text,
    tts_text,
    auto_split_sentences=True,
    progress=gr.Progress()
):
    try:
        progress(0, desc="Starting step 1: Preprocessing dataset...")
        preprocess_res = preprocess_dataset(
            audio_files, audio_dir, transcript_file, language, whisper_model, asr_backend, out_path, dataset_name, diarize_speakers,
            expected_speakers, diarize_threshold,
            generate_synthetic, synthetic_audio_file, synthetic_vtt_file,
            auto_split_sentences,
            progress
        )
        status_msg, dataset_dir = preprocess_res[0], preprocess_res[1]
        if not dataset_dir or "failed" in status_msg.lower():
            train_status_msg = f"Training skipped because dataset preparation failed: {status_msg}"
            empty_train = (train_status_msg, "", "", "", "", "", "", None, model_key, gr.update(), gr.update())
            empty_infer = (f"Inference skipped: Preprocessing failed.", None, None)
            return empty_train + preprocess_res + empty_infer
            
        progress(0.4, desc="Preprocessing complete! Starting step 2: Training model...")
        
        train_res = run_training(
            model_key, dataset_dir, train_language, num_epochs, batch_size, grad_accum, out_path, max_audio_length, restore_path, use_pretrained, pretrained_model_id, extra_overrides_json,
            sample_epoch_interval, sample_text,
            progress
        )
        artifacts_file_val = train_res[2]
        speaker_ref_val = train_res[7] or (preprocess_res[4] if model_key == "styletts2" else None)
        
        if not artifacts_file_val or "failed" in train_res[0].lower():
            train_status_msg = f"Inference skipped because training failed: {train_res[0]}"
            empty_infer = (train_status_msg, None, None)
            return train_res + preprocess_res + empty_infer

        progress(0.9, desc="Training complete! Starting step 3: Generating test speech...")
        
        infer_res = run_inference(
            artifacts_file_val, model_key, train_language, tts_text, speaker_ref_val, out_path, progress
        )
        return train_res + preprocess_res + infer_res
    except Exception as exc:
        err = format_exception(exc)
        empty_train = (f"Pipeline error: {err}", "", "", "", "", "", "", None, model_key, gr.update(), gr.update())
        empty_prep = (err, "", "", "", "", gr.update(choices=list_datasets(out_path), value=None), None, gr.update(visible=False, choices=[]), gr.update(visible=False), None, "", [])
        empty_infer = (f"Pipeline error: {err}", None, None)
        return empty_train + empty_prep + empty_infer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Coqui TTS fine-tuning web UI")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind the web UI to")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--out_path", type=str, default=str(Path.cwd() / "finetune_models"))
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_acumm", type=int, default=1)
    parser.add_argument("--max_audio_length", type=int, default=11)
    args = parser.parse_args()

    theme = gr.themes.Origin(
        primary_hue="green",
        secondary_hue="amber",
        neutral_hue="gray",
        radius_size="lg",
        font_mono=["JetBrains Mono", "monospace", "Consolas", "Menlo", "Liberation Mono"],
    )

    css_str = """
    .primary-btn {
        background: linear-gradient(90deg, #22c55e 0%, #eab308 100%) !important;
        color: white !important;
        border: none !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .primary-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4) !important;
    }
    .primary-btn:active {
        transform: translateY(0);
    }
    """

    with gr.Blocks(title='Universal TTS Finetune') as demo:
        gr.Markdown(
            "# Universal TTS Finetune\n"
            "Prepare an LJSpeech-style dataset, fine-tune a supported Coqui recipe, and test the trained model."
        )

        with gr.Tab("1 - Prepare dataset"):
            out_path = gr.Textbox(label="Output root", value=args.out_path)
            dataset_name = gr.Textbox(label="Dataset name", value="dataset_1")
            audio_upload = gr.File(
                file_count="multiple",
                label="Audio files (wav, mp3, flac, m4a, ogg)",
            )
            audio_dir = gr.Textbox(label="Audio folder path (optional)", value="")
            transcript_file = gr.File(label="Optional transcript map or alignment file (.vtt, .txt, .csv, .tsv, .json)")
            auto_split_sentences = gr.Checkbox(label="Auto-split sentences for forced alignment (plain text input)", value=True)
            gr.Markdown(
                "💡 **How to provide text/transcripts:**\n"
                "- **None (auto-transcribe)**: Leave blank to transcribe. If an **Audio folder path** contains matching `.vtt` or `.txt` files (e.g., `chapter1.mp3` and `chapter1.txt`), the app uses those first. For files without text, Auto uses Whisper with short language codes and MMS ASR with three-letter codes.\n"
                "- **WebVTT (.vtt)**: Upload a WebVTT file along with the full audiobook file (e.g. generated by `ebook2audiobook`) to slice it instantly with 0% transcription errors.\n"
                "- **Plain Text (.txt) - Forced Alignment**: Upload a plain text book file (e.g. converted from ePUB using Calibre) along with a single full audiobook file to run **Forced Alignment**. If the text contains multiple sentences on a line or is a single paragraph, leave **Auto-split sentences for forced alignment** checked to automatically chunk it into sentences.\n"
                "- **Transcript Map (.csv, .tsv, .json, or delimited .txt)**: Upload a mapping file of `audio_file|text` matching a folder of pre-split audio files."
            )
            language = gr.Dropdown(label="Dataset language", choices=DATASET_LANGUAGE_CHOICES, value="en", allow_custom_value=True)
            asr_backend = gr.Dropdown(label="Transcription backend (used only when text is missing)", choices=["auto", "whisper", "mms"], value="auto")
            whisper_model = gr.Dropdown(label="Whisper model", choices=WHISPER_CHOICES, value="small", allow_custom_value=True)
            diarize_speakers = gr.Checkbox(label="Diarize speakers (split multi-speaker audio)", value=False)
            with gr.Row(visible=False) as diarize_options:
                expected_speakers = gr.Slider(label="Expected speaker count (0 for auto)", minimum=0, maximum=20, step=1, value=0)
                diarize_threshold = gr.Slider(label="Diarization threshold (distance, only if auto)", minimum=0.05, maximum=1.0, step=0.05, value=0.35)

            # Synthetic Data Generation Option
            generate_synthetic = gr.Checkbox(
                label="Don't have enough training data? Import synthetic data from ebook2audiobook",
                value=False
            )
            with gr.Group(visible=False) as synthetic_options:
                gr.Markdown("### Import Synthetic Data")
                synthetic_audio_file = gr.File(
                    label="Upload synthesized audiobook (mp3, wav, flac, etc.)",
                    file_count="single"
                )
                synthetic_vtt_file = gr.File(
                    label="Upload matching .vtt file",
                    file_count="single"
                )

            def _toggle_synthetic_group(enabled):
                return gr.update(visible=enabled)

            generate_synthetic.change(
                fn=_toggle_synthetic_group,
                inputs=[generate_synthetic],
                outputs=[synthetic_options]
            )
            
            # Speaker preview group (initially hidden)
            speakers_state = gr.State([])
            with gr.Group(visible=False) as speakers_container:
                gr.Markdown("### Detected Speakers Preview")
                speaker_selector = gr.Dropdown(label="Select Speaker", choices=[])
                speaker_preview_audio = gr.Audio(label="Speaker Sample Audio", interactive=False)
                speaker_details = gr.Markdown("")
            
            dataset_status = gr.Textbox(label="Status", interactive=False)
            dataset_dir = gr.Textbox(label="Dataset directory")
            train_csv = gr.Textbox(label="Train metadata")
            val_csv = gr.Textbox(label="Validation metadata")
            dataset_reference = gr.Textbox(label="Reference WAV")
            with gr.Row():
                prepare_btn = gr.Button(value="Step 1 - Create dataset", elem_classes=["primary-btn"])
                prepare_and_train_btn = gr.Button(value="Create dataset & Start training", variant="secondary")
            
            with gr.Accordion("Re-diarize an Existing Dataset", open=False):
                gr.Markdown("Select a previously created dataset (mixed or diarized) and re-run speaker diarization using updated settings without re-transcribing.")
                re_diarize_source = gr.Dropdown(
                    label="Select dataset to re-diarize",
                    choices=list_datasets(args.out_path),
                    value=None,
                    allow_custom_value=True,
                    interactive=True,
                )
                with gr.Row():
                    re_diarize_expected = gr.Slider(label="Expected speaker count (0 for auto)", minimum=0, maximum=20, step=1, value=0)
                    re_diarize_thresh = gr.Slider(label="Diarization threshold (distance, only if auto)", minimum=0.05, maximum=1.0, step=0.05, value=0.35)
                re_diarize_btn = gr.Button(value="Re-diarize Dataset", variant="secondary")

        with gr.Tab("2 - Train model"):
            model_key = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value="xtts_v2")
            model_checkpoint_warning = gr.Markdown(
                value="🟢 **XTTS v2** supports English.\n\nFine-tuning will start from the official multilingual checkpoint: `tts_models/multilingual/multi-dataset/xtts_v2`."
            )
            train_dataset_dir = gr.Dropdown(
                label="Dataset directory",
                choices=list_datasets(args.out_path),
                value=None,
                allow_custom_value=True,
                interactive=True,
            )
            train_language = gr.Dropdown(label="Fine-tuning language", choices=XTTS_LANGUAGE_CHOICES["xtts_v2"], value="en", info="Choose your dataset language; available base checkpoints update automatically. MMS uses its published three-letter language codes.")
            with gr.Row():
                restore_model_dropdown = gr.Dropdown(
                    label="Resume from previous training run",
                    choices=[("None", "")] + list_trained_models(args.out_path, "xtts_v2"),
                    value=None,
                    interactive=True,
                )
                restore_path = gr.Textbox(label="Optional checkpoint to continue from", value="")
            use_pretrained = gr.Checkbox(label="Start from a pretrained checkpoint", value=True)
            pretrained_model_id = gr.Dropdown(
                label="Starting checkpoint",
                choices=list(pretrained_model_choices("xtts_v2", "en")),
                value=pretrained_model_choices("xtts_v2", "en")[0],
                interactive=True,
                info="Choose a published voice or model for this language. Your local restore path takes precedence.",
            )
            num_epochs = gr.Slider(label="Epochs", minimum=1, maximum=1000, step=1, value=args.num_epochs)
            batch_size = gr.Slider(label="Batch size", minimum=1, maximum=128, step=1, value=args.batch_size)
            grad_accum = gr.Slider(label="Grad accumulation", minimum=1, maximum=128, step=1, value=args.grad_acumm)
            max_audio_length = gr.Slider(label="Max audio length (seconds)", minimum=2, maximum=30, step=1, value=args.max_audio_length)
            extra_overrides_json = gr.Code(
                label="Optional config overrides JSON",
                language="json",
                value="{}",
            )
            with gr.Accordion("Periodic Progress Audio Sampling (Piper Only)", open=False):
                sample_epoch_interval = gr.Slider(
                    label="Sample Interval (Epochs)",
                    minimum=0,
                    maximum=500,
                    step=10,
                    value=0,
                    info="Generate and save an audio sample every N epochs. Set to 0 to disable."
                )
                sample_text = gr.Textbox(
                    label="Sample Test Phrase",
                    value="This is a periodic audio sample to test training progress.",
                    info="Text sentence to synthesize at each interval."
                )
            train_status = gr.Textbox(label="Status", interactive=False)
            training_root = gr.Textbox(label="Training root")
            artifacts_file = gr.Textbox(label="Artifacts file")
            checkpoint_path = gr.Textbox(label="Checkpoint path")
            config_path = gr.Textbox(label="Config path")
            trained_reference = gr.Textbox(label="Reference WAV")
            with gr.Row():
                train_btn = gr.Button(value="Step 2 - Train model", elem_classes=["primary-btn"])
                pause_btn = gr.Button(value="Pause Training", variant="secondary")
                resume_btn = gr.Button(value="Resume Training", variant="secondary")
            latest_btn = gr.Button(value="Load latest trained model")

        with gr.Tab("3 - Inference"):
            infer_model_key = gr.Dropdown(label="Model", choices=INFERENCE_MODEL_CHOICES, value="xtts_v2")
            infer_trained_model = gr.Dropdown(
                label="Select previously fine-tuned model",
                choices=list_trained_models(args.out_path, "xtts_v2"),
                value=None,
                interactive=True,
            )
            infer_artifacts = gr.Textbox(label="Artifacts file or ready/training folder", value="")
            speaker_reference_audio = gr.Audio(
                label="Speaker reference audio – required for XTTS and StyleTTS2",
                type="filepath",
                sources=["upload"],
            )
            infer_language = gr.Dropdown(label="Inference language", choices=XTTS_LANGUAGE_CHOICES["xtts_v2"], value="en")
            tts_text = gr.Textbox(label="Input text", value="This fine-tuned model is ready to test.")
            infer_status = gr.Textbox(label="Status", interactive=False)
            generated_audio = gr.Audio(label="Generated audio")
            used_reference_audio = gr.Audio(label="Reference audio used")
            inspect_btn = gr.Button(value="Inspect artifacts")
            tts_btn = gr.Button(value="Step 3 - Generate speech", elem_classes=["primary-btn"])
            gr.Markdown("### E2A custom model upload")
            e2a_export_btn = gr.Button(value="Create E2A upload ZIP")
            e2a_export_status = gr.Textbox(label="Export status", interactive=False)
            e2a_export_file = gr.File(label="Download E2A ZIP", interactive=False)

        prepare_btn.click(
            fn=preprocess_dataset,
            inputs=[
                audio_upload,
                audio_dir,
                transcript_file,
                language,
                whisper_model,
                asr_backend,
                out_path,
                dataset_name,
                diarize_speakers,
                expected_speakers,
                diarize_threshold,
                generate_synthetic,
                synthetic_audio_file,
                synthetic_vtt_file,
                auto_split_sentences,
            ],
            outputs=[
                dataset_status,
                dataset_dir,
                train_csv,
                val_csv,
                dataset_reference,
                train_dataset_dir,
                speaker_reference_audio,
                speaker_selector,
                speakers_container,
                speaker_preview_audio,
                speaker_details,
                speakers_state,
            ],
        )

        re_diarize_btn.click(
            fn=preprocess_re_diarize,
            inputs=[
                re_diarize_source,
                re_diarize_expected,
                re_diarize_thresh,
                out_path,
            ],
            outputs=[
                dataset_status,
                dataset_dir,
                train_csv,
                val_csv,
                dataset_reference,
                train_dataset_dir,
                speaker_reference_audio,
                speaker_selector,
                speakers_container,
                speaker_preview_audio,
                speaker_details,
                speakers_state,
            ],
        )

        re_diarize_source.focus(
            fn=lambda op: gr.update(choices=list_datasets(op)),
            inputs=[out_path],
            outputs=[re_diarize_source],
        )

        prepare_and_train_btn.click(
            fn=preprocess_and_train,
            inputs=[
                # Preprocessing inputs
                audio_upload,
                audio_dir,
                transcript_file,
                language,
                whisper_model,
                asr_backend,
                out_path,
                dataset_name,
                diarize_speakers,
                expected_speakers,
                diarize_threshold,
                generate_synthetic,
                synthetic_audio_file,
                synthetic_vtt_file,
                # Training inputs
                model_key,
                train_language,
                num_epochs,
                batch_size,
                grad_accum,
                max_audio_length,
                restore_path,
                use_pretrained,
                pretrained_model_id,
                extra_overrides_json,
                sample_epoch_interval,
                sample_text,
                # Inference input
                tts_text,
                auto_split_sentences,
            ],
            outputs=[
                # Training outputs (11 items)
                train_status,
                training_root,
                artifacts_file,
                checkpoint_path,
                config_path,
                trained_reference,
                infer_artifacts,
                speaker_reference_audio,
                infer_model_key,
                infer_trained_model,
                restore_model_dropdown,
                # Preprocessing outputs (12 items)
                dataset_status,
                dataset_dir,
                train_csv,
                val_csv,
                dataset_reference,
                train_dataset_dir,
                speaker_reference_audio,
                speaker_selector,
                speakers_container,
                speaker_preview_audio,
                speaker_details,
                speakers_state,
                # Inference outputs (3 items)
                infer_status,
                generated_audio,
                used_reference_audio,
            ],
        )

        speaker_selector.change(
            fn=on_select_speaker,
            inputs=[speaker_selector, speakers_state],
            outputs=[
                dataset_dir,
                dataset_reference,
                speaker_preview_audio,
                speaker_details,
                train_dataset_dir,
            ],
        )

        train_btn.click(
            fn=run_training,
            inputs=[
                model_key,
                train_dataset_dir,
                train_language,
                num_epochs,
                batch_size,
                grad_accum,
                out_path,
                max_audio_length,
                restore_path,
                use_pretrained,
                pretrained_model_id,
                extra_overrides_json,
                sample_epoch_interval,
                sample_text,
            ],
            outputs=[
                train_status,
                training_root,
                artifacts_file,
                checkpoint_path,
                config_path,
                trained_reference,
                infer_artifacts,
                speaker_reference_audio,
                infer_model_key,
                infer_trained_model,
                restore_model_dropdown,
            ],
        )

        latest_btn.click(
            fn=locate_artifacts,
            inputs=[out_path, model_key],
            outputs=[
                train_status,
                training_root,
                artifacts_file,
                checkpoint_path,
                config_path,
                trained_reference,
                infer_artifacts,
                speaker_reference_audio,
                infer_model_key,
                infer_trained_model,
            ],
        )

        inspect_btn.click(
            fn=inspect_artifacts,
            inputs=[infer_artifacts, infer_model_key],
            outputs=[
                infer_status,
                training_root,
                artifacts_file,
                checkpoint_path,
                config_path,
                trained_reference,
                speaker_reference_audio,
            ],
        )

        tts_btn.click(
            fn=run_inference,
            inputs=[
                infer_artifacts,
                infer_model_key,
                infer_language,
                tts_text,
                speaker_reference_audio,
                out_path,
            ],
            outputs=[infer_status, generated_audio, used_reference_audio],
        )

        e2a_export_btn.click(
            fn=export_for_e2a,
            inputs=[infer_artifacts, out_path],
            outputs=[e2a_export_status, e2a_export_file],
        )

        def toggle_diarize_options(visible):
            return gr.update(visible=visible)

        diarize_speakers.change(
            fn=toggle_diarize_options,
            inputs=[diarize_speakers],
            outputs=[diarize_options],
        )

        model_key.change(
            fn=on_model_change,
            inputs=[model_key],
            outputs=[speaker_reference_audio, used_reference_audio],
        )
        model_key.change(
            fn=on_training_params_change,
            inputs=[model_key, train_dataset_dir],
            outputs=[num_epochs, batch_size],
        )
        model_key.change(
            fn=lambda model: gr.update(
                label="Local StyleTTS2 base checkpoint" if model == "styletts2" else "Restore path unavailable for OmniVoice" if model == "omnivoice" else "Optional checkpoint to continue from",
                value="",
                interactive=model != "omnivoice",
            ),
            inputs=[model_key],
            outputs=[restore_path],
        )
        model_key.change(
            fn=lambda model: gr.update(value=1, interactive=False) if model == "omnivoice" else gr.update(value=1, interactive=False) if model == "styletts2" else gr.update(interactive=True),
            inputs=[model_key],
            outputs=[grad_accum],
        )
        model_key.change(
            fn=update_finetune_language_choices,
            inputs=[model_key],
            outputs=[train_language],
        ).then(
            fn=update_checkpoint_choices,
            inputs=[model_key, train_language],
            outputs=[pretrained_model_id],
        ).then(
            fn=update_training_options,
            inputs=[model_key, train_language, use_pretrained, pretrained_model_id],
            outputs=[model_checkpoint_warning, use_pretrained],
        )
        train_language.change(
            fn=update_training_options,
            inputs=[model_key, train_language, use_pretrained, pretrained_model_id],
            outputs=[model_checkpoint_warning, use_pretrained],
        )
        train_language.change(fn=update_checkpoint_choices, inputs=[model_key, train_language], outputs=[pretrained_model_id])
        use_pretrained.change(
            fn=lambda m, l, u, p: update_training_options(m, l, u, p)[0],
            inputs=[model_key, train_language, use_pretrained, pretrained_model_id],
            outputs=[model_checkpoint_warning],
        )
        pretrained_model_id.change(
            fn=lambda m, l, u, p: update_training_options(m, l, u, p)[0],
            inputs=[model_key, train_language, use_pretrained, pretrained_model_id],
            outputs=[model_checkpoint_warning],
        )

        train_dataset_dir.change(
            fn=on_training_params_change,
            inputs=[model_key, train_dataset_dir],
            outputs=[num_epochs, batch_size],
        )

        infer_model_key.change(
            fn=on_model_change,
            inputs=[infer_model_key],
            outputs=[speaker_reference_audio, used_reference_audio],
        )
        infer_model_key.change(
            fn=update_finetune_language_choices,
            inputs=[infer_model_key],
            outputs=[infer_language],
        )
        infer_model_key.change(
            fn=update_trained_models,
            inputs=[out_path, infer_model_key],
            outputs=[infer_trained_model],
        )

        infer_trained_model.change(
            fn=select_trained_model,
            inputs=[infer_trained_model],
            outputs=[infer_artifacts],
        )

        out_path.change(
            fn=update_dataset_choices,
            inputs=[out_path],
            outputs=[train_dataset_dir],
        )
        out_path.change(
            fn=update_trained_models,
            inputs=[out_path, infer_model_key],
            outputs=[infer_trained_model],
        )

        def on_restore_dropdown_change(artifacts_path):
            if not artifacts_path:
                return ""
            return resolve_resume_checkpoint(artifacts_path)

        restore_model_dropdown.change(
            fn=on_restore_dropdown_change,
            inputs=[restore_model_dropdown],
            outputs=[restore_path],
        )

        model_key.change(
            fn=update_resume_models,
            inputs=[out_path, model_key],
            outputs=[restore_model_dropdown],
        )

        out_path.change(
            fn=update_resume_models,
            inputs=[out_path, model_key],
            outputs=[restore_model_dropdown],
        )

        pause_btn.click(
            fn=pause_training,
            inputs=[],
            outputs=[train_status],
        )

        resume_btn.click(
            fn=resume_training,
            inputs=[],
            outputs=[train_status],
        )

        demo.load(
            fn=update_training_options,
            inputs=[model_key, train_language, use_pretrained, pretrained_model_id],
            outputs=[model_checkpoint_warning, use_pretrained],
        )

    allowed = [
        str(Path(args.out_path).resolve()),
        str(Path.home()),
        str(Path.cwd().resolve()),
        str(Path.cwd().parent.parent.resolve())
    ]
    demo.launch(share=args.share, debug=False, server_name=args.host, server_port=args.port, allowed_paths=allowed, theme=theme, css=css_str)
