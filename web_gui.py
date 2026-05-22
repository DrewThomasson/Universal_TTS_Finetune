from __future__ import annotations

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
)

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
    "zh",
    "hu",
    "ko",
    "ja",
]
WHISPER_CHOICES = ["large-v3", "large-v2", "large", "medium", "small", "base"]
MODEL_CHOICES = [(label, key) for key, label in dropdown_choices()]


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
        
    def __call__(self, log_line: str) -> None:
        if "complete" in log_line.lower() or "finished" in log_line.lower():
            self.progress_bar(1.0, desc=log_line.strip())
            return
            
        epoch_match = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", log_line)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            self.total_epochs = int(epoch_match.group(2))
        else:
            epoch_match2 = re.search(r"Epoch\s*:\s*(\d+)", log_line, re.IGNORECASE)
            if not epoch_match2:
                epoch_match2 = re.search(r"epoch\s*=\s*(\d+)", log_line, re.IGNORECASE)
            if epoch_match2:
                self.current_epoch = int(epoch_match2.group(1))
                
        if self.total_epochs > 0 and self.current_epoch > 0:
            fraction = min(self.current_epoch / self.total_epochs, 1.0)
            self.progress_bar(fraction, desc=f"Training: Epoch {self.current_epoch}/{self.total_epochs}")
        else:
            clean = log_line.strip()
            if clean:
                desc = clean[:60] + "..." if len(clean) > 60 else clean
                fraction = min(self.current_epoch / self.total_epochs, 1.0) if self.total_epochs > 0 and self.current_epoch > 0 else 0.0
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
    val = choices[0][1] if choices else ""
    return gr.update(choices=choices, value=val)


def _path_value(value):
    return getattr(value, "name", value) if value else None


def _gradio_progress(progress: gr.Progress | None):
    if progress is None:
        return None

    def callback(message: str) -> None:
        progress(0, desc=message)

    return callback


def preprocess_dataset(audio_files, audio_dir, transcript_file, language, whisper_model, out_path, dataset_name, diarize_speakers, progress=gr.Progress()):
    try:
        tracker = PreprocessProgressTracker(progress)
        result = prepare_dataset(
            output_root=out_path,
            audio_files=audio_files,
            audio_dir=audio_dir or None,
            transcript_file=_path_value(transcript_file),
            language=language,
            whisper_model_name=whisper_model,
            dataset_name=dataset_name or "LJSpeech-1.1",
            diarize_speakers=diarize_speakers,
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
            default_ref,
            show_speakers,
            show_container,
            default_ref,
            default_info,
            speakers_list,
        )
    except Exception as exc:
        return (
            format_exception(exc), "", "", "", "",
            gr.update(choices=list_datasets(out_path), value=""), "",
            gr.update(visible=False, choices=[]), gr.update(visible=False),
            None, "", []
        )


def run_training(model_key, dataset_dir, language, num_epochs, batch_size, grad_accum, out_path, max_audio_length, restore_path, use_pretrained, extra_overrides_json, progress=gr.Progress()):
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
            extra_overrides_json=extra_overrides_json or None,
            progress=tracker,
        )
        message = f"Training finished. Ready artifacts saved in {Path(result['training_root']) / 'ready'}"
        
        updated_models = list_trained_models(out_path, model_key)
        new_val = updated_models[0][1] if updated_models else ""
        
        return (
            message,
            result["training_root"],
            result["artifacts_file"],
            result["checkpoint"],
            result["config"],
            result.get("reference_wav", ""),
            result["artifacts_file"],
            result.get("reference_wav", ""),
            model_key,
            gr.update(choices=updated_models, value=new_val),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", "", "", "", model_key, gr.update()


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
            artifacts.get("reference_wav", ""),
            artifacts["artifacts_file"],
            artifacts.get("reference_wav", ""),
            artifacts["model_key"],
            gr.update(choices=updated_models, value=new_val),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", "", "", "", model_key, gr.update()


def inspect_artifacts(artifacts_path, model_key):
    try:
        artifacts = load_artifacts(artifacts_path, model_key=model_key or None)
        return (
            f"Artifacts loaded for {artifacts['model_label']}",
            artifacts["training_root"],
            artifacts["artifacts_file"],
            artifacts["checkpoint"],
            artifacts["config"],
            artifacts.get("reference_wav", ""),
            artifacts.get("reference_wav", ""),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", "", ""


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
        return "Speech generated.", result["output_file"], result.get("speaker_wav") or None
    except Exception as exc:
        return format_exception(exc), None, None


def on_model_change(selected_model):
    try:
        from utils.model_registry import get_model_spec
        spec = get_model_spec(selected_model)
        req = spec.requires_speaker_wav
    except Exception:
        req = False
    return gr.update(visible=req), gr.update(visible=req)


def on_select_speaker(selected_dir, speakers_state):
    if not selected_dir or not speakers_state:
        return gr.update(), "", "", "", gr.update()
    
    speaker_info = next((s for s in speakers_state if s["dataset_dir"] == selected_dir), None)
    if not speaker_info:
        return gr.update(), "", "", "", gr.update()
        
    info_md = f"**Dataset path**: `{selected_dir}`\n**Duration**: {speaker_info['total_audio_seconds']} seconds\n**Total clips**: {speaker_info['created_sample_count']}"
    ref_wav = speaker_info["reference_wav"]
    
    return selected_dir, ref_wav, ref_wav, info_md, gr.update(value=selected_dir)


def select_trained_model(val):
    return val


def on_training_params_change(model_key, dataset_dir):
    epochs, batch_size = get_adaptive_defaults(model_key, dataset_dir)
    return epochs, batch_size


def preprocess_and_train(
    audio_files, audio_dir, transcript_file, language, whisper_model, out_path, dataset_name, diarize_speakers,
    model_key, train_language, num_epochs, batch_size, grad_accum, max_audio_length, restore_path, use_pretrained, extra_overrides_json,
    progress=gr.Progress()
):
    try:
        progress(0, desc="Starting step 1: Preprocessing dataset...")
        preprocess_res = preprocess_dataset(
            audio_files, audio_dir, transcript_file, language, whisper_model, out_path, dataset_name, diarize_speakers, progress
        )
        status_msg, dataset_dir = preprocess_res[0], preprocess_res[1]
        if not dataset_dir or "failed" in status_msg.lower():
            train_status_msg = f"Training skipped because dataset preparation failed: {status_msg}"
            empty_train = (train_status_msg, "", "", "", "", "", "", "", model_key, gr.update())
            return empty_train + preprocess_res
            
        progress(0.5, desc="Preprocessing complete! Starting step 2: Training model...")
        
        train_res = run_training(
            model_key, dataset_dir, train_language, num_epochs, batch_size, grad_accum, out_path, max_audio_length, restore_path, use_pretrained, extra_overrides_json, progress
        )
        return train_res + preprocess_res
    except Exception as exc:
        err = format_exception(exc)
        empty_train = (f"Pipeline error: {err}", "", "", "", "", "", "", "", model_key, gr.update())
        empty_prep = (err, "", "", "", "", gr.update(choices=list_datasets(out_path), value=""), "", gr.update(visible=False, choices=[]), gr.update(visible=False), None, "", [])
        return empty_train + empty_prep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Coqui TTS fine-tuning web UI")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--out_path", type=str, default=str(Path.cwd() / "finetune_models"))
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_acumm", type=int, default=1)
    parser.add_argument("--max_audio_length", type=int, default=11)
    args = parser.parse_args()

    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
    )

    css_str = """
    .primary-btn {
        background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .primary-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4) !important;
    }
    .primary-btn:active {
        transform: translateY(0);
    }
    """

    with gr.Blocks(title="Universal TTS Finetune", theme=theme, css=css_str) as demo:
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
            transcript_file = gr.File(label="Optional transcript map (csv, tsv, pipe-delimited txt, or json)")
            language = gr.Dropdown(label="Dataset language", choices=LANGUAGE_CHOICES, value="en")
            whisper_model = gr.Dropdown(label="Whisper model", choices=WHISPER_CHOICES, value="small")
            diarize_speakers = gr.Checkbox(label="Diarize speakers (split multi-speaker audio)", value=False)
            
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

        with gr.Tab("2 - Train model"):
            model_key = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value="xtts_v2")
            train_dataset_dir = gr.Dropdown(
                label="Dataset directory",
                choices=list_datasets(args.out_path),
                value="",
                allow_custom_value=True,
                interactive=True,
            )
            train_language = gr.Dropdown(label="Model language (XTTS/Piper support multilingual)", choices=LANGUAGE_CHOICES, value="en")
            restore_path = gr.Textbox(label="Optional checkpoint to continue from", value="")
            use_pretrained = gr.Checkbox(label="Auto-download matching pretrained model when available", value=True)
            num_epochs = gr.Slider(label="Epochs", minimum=1, maximum=1000, step=1, value=args.num_epochs)
            batch_size = gr.Slider(label="Batch size", minimum=1, maximum=128, step=1, value=args.batch_size)
            grad_accum = gr.Slider(label="Grad accumulation", minimum=1, maximum=128, step=1, value=args.grad_acumm)
            max_audio_length = gr.Slider(label="Max audio length (seconds)", minimum=2, maximum=30, step=1, value=args.max_audio_length)
            extra_overrides_json = gr.Code(
                label="Optional config overrides JSON",
                language="json",
                value="{}",
            )
            train_status = gr.Textbox(label="Status", interactive=False)
            training_root = gr.Textbox(label="Training root")
            artifacts_file = gr.Textbox(label="Artifacts file")
            checkpoint_path = gr.Textbox(label="Checkpoint path")
            config_path = gr.Textbox(label="Config path")
            trained_reference = gr.Textbox(label="Reference WAV")
            train_btn = gr.Button(value="Step 2 - Train model", elem_classes=["primary-btn"])
            latest_btn = gr.Button(value="Load latest trained model")

        with gr.Tab("3 - Inference"):
            infer_model_key = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value="xtts_v2")
            infer_trained_model = gr.Dropdown(
                label="Select previously fine-tuned model",
                choices=list_trained_models(args.out_path, "xtts_v2"),
                value="",
                interactive=True,
            )
            infer_artifacts = gr.Textbox(label="Artifacts file or ready/training folder", value="")
            speaker_reference_audio = gr.Textbox(label="Optional speaker reference WAV (XTTS)", value="")
            infer_language = gr.Dropdown(label="Inference language", choices=LANGUAGE_CHOICES, value="en")
            tts_text = gr.Textbox(label="Input text", value="This fine-tuned model is ready to test.")
            infer_status = gr.Textbox(label="Status", interactive=False)
            generated_audio = gr.Audio(label="Generated audio")
            used_reference_audio = gr.Audio(label="Reference audio used")
            inspect_btn = gr.Button(value="Inspect artifacts")
            tts_btn = gr.Button(value="Step 3 - Generate speech", elem_classes=["primary-btn"])

        prepare_btn.click(
            fn=preprocess_dataset,
            inputs=[
                audio_upload,
                audio_dir,
                transcript_file,
                language,
                whisper_model,
                out_path,
                dataset_name,
                diarize_speakers,
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

        prepare_and_train_btn.click(
            fn=preprocess_and_train,
            inputs=[
                # Preprocessing inputs
                audio_upload,
                audio_dir,
                transcript_file,
                language,
                whisper_model,
                out_path,
                dataset_name,
                diarize_speakers,
                # Training inputs
                model_key,
                train_language,
                num_epochs,
                batch_size,
                grad_accum,
                max_audio_length,
                restore_path,
                use_pretrained,
                extra_overrides_json,
            ],
            outputs=[
                # Training outputs (10 items)
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
                extra_overrides_json,
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

    demo.launch(share=args.share, debug=False, server_port=args.port)
