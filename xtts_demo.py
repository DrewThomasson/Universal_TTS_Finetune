from __future__ import annotations

import argparse
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
MODEL_CHOICES = dropdown_choices()


def _path_value(value):
    return getattr(value, "name", value) if value else None


def _gradio_progress(progress: gr.Progress | None):
    if progress is None:
        return None

    def callback(message: str) -> None:
        progress(0, desc=message)

    return callback


def preprocess_dataset(audio_files, audio_folder_path, transcript_file, language, whisper_model, out_path, progress=gr.Progress()):
    try:
        result = prepare_dataset(
            output_root=out_path,
            audio_files=audio_files,
            audio_dir=audio_folder_path or None,
            transcript_file=_path_value(transcript_file),
            language=language,
            whisper_model_name=whisper_model,
            progress=_gradio_progress(progress),
        )
        message = (
            f"Dataset ready with {result['created_sample_count']} samples at {result['dataset_dir']}"
        )
        return message, result["dataset_dir"], result["metadata_train"], result["metadata_val"], result["reference_wav"]
    except Exception as exc:
        return format_exception(exc), "", "", "", ""



def run_training(model_key, dataset_dir, language, num_epochs, batch_size, grad_accum, out_path, max_audio_length, restore_path, use_pretrained, extra_overrides_json, progress=gr.Progress()):
    try:
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
            progress=_gradio_progress(progress),
        )
        message = f"Training finished. Ready artifacts saved in {Path(result['training_root']) / 'ready'}"
        return (
            message,
            result["training_root"],
            result["artifacts_file"],
            result["checkpoint"],
            result["config"],
            result.get("reference_wav", ""),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", ""



def locate_artifacts(out_path, model_key):
    try:
        artifacts = find_latest_artifacts(out_path, model_key=model_key or None)
        return (
            f"Loaded latest artifacts for {artifacts['model_label']}",
            artifacts["training_root"],
            artifacts["artifacts_file"],
            artifacts["checkpoint"],
            artifacts["config"],
            artifacts.get("reference_wav", ""),
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", ""



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
        )
    except Exception as exc:
        return format_exception(exc), "", "", "", "", ""



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
        return "Speech generated.", result["output_file"], result.get("speaker_wav", "")
    except Exception as exc:
        return format_exception(exc), None, None


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

    with gr.Blocks(title="Universal TTS Finetune") as demo:
        gr.Markdown(
            "# Universal TTS Finetune\n"
            "Prepare an LJSpeech-style dataset, fine-tune a supported Coqui recipe, and test the trained model."
        )

        with gr.Tab("1 - Prepare dataset"):
            out_path = gr.Textbox(label="Output root", value=args.out_path)
            audio_upload = gr.File(
                file_count="multiple",
                label="Audio files (wav, mp3, flac, m4a, ogg)",
            )
            audio_folder_path = gr.Textbox(label="Audio folder path (optional)", value="")
            transcript_file = gr.File(label="Optional transcript map (csv, tsv, pipe-delimited txt, or json)")
            language = gr.Dropdown(label="Dataset language", choices=LANGUAGE_CHOICES, value="en")
            whisper_model = gr.Dropdown(label="Whisper model", choices=WHISPER_CHOICES, value="small")
            dataset_status = gr.Textbox(label="Status", interactive=False)
            dataset_dir = gr.Textbox(label="Dataset directory")
            train_csv = gr.Textbox(label="Train metadata")
            val_csv = gr.Textbox(label="Validation metadata")
            dataset_reference = gr.Textbox(label="Reference WAV")
            prepare_btn = gr.Button(value="Step 1 - Create dataset")

        with gr.Tab("2 - Train model"):
            model_key = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value="xtts_v2")
            train_dataset_dir = gr.Textbox(label="Dataset directory", value="")
            train_language = gr.Dropdown(label="Model language (XTTS only for non-English)", choices=LANGUAGE_CHOICES, value="en")
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
            train_btn = gr.Button(value="Step 2 - Train model")
            latest_btn = gr.Button(value="Load latest trained model")

        with gr.Tab("3 - Inference"):
            infer_model_key = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value="xtts_v2")
            infer_artifacts = gr.Textbox(label="Artifacts file or ready/training folder", value="")
            speaker_reference_audio = gr.Textbox(label="Optional speaker reference WAV (XTTS)", value="")
            infer_language = gr.Dropdown(label="Inference language", choices=LANGUAGE_CHOICES, value="en")
            tts_text = gr.Textbox(label="Input text", value="This fine-tuned model is ready to test.")
            infer_status = gr.Textbox(label="Status", interactive=False)
            generated_audio = gr.Audio(label="Generated audio")
            used_reference_audio = gr.Audio(label="Reference audio used")
            inspect_btn = gr.Button(value="Inspect artifacts")
            tts_btn = gr.Button(value="Step 3 - Generate speech")

        prepare_btn.click(
            fn=preprocess_dataset,
            inputs=[audio_upload, audio_folder_path, transcript_file, language, whisper_model, out_path],
            outputs=[dataset_status, dataset_dir, train_csv, val_csv, dataset_reference],
        )
        prepare_btn.click(fn=lambda path: path, inputs=[dataset_dir], outputs=[train_dataset_dir])
        prepare_btn.click(fn=lambda path: path, inputs=[dataset_reference], outputs=[speaker_reference_audio])

        train_btn.click(
            fn=run_training,
            inputs=[model_key, train_dataset_dir, train_language, num_epochs, batch_size, grad_accum, out_path, max_audio_length, restore_path, use_pretrained, extra_overrides_json],
            outputs=[train_status, training_root, artifacts_file, checkpoint_path, config_path, trained_reference],
        )
        train_btn.click(fn=lambda path: path, inputs=[artifacts_file], outputs=[infer_artifacts])
        train_btn.click(fn=lambda path: path, inputs=[trained_reference], outputs=[speaker_reference_audio])
        train_btn.click(fn=lambda key: key, inputs=[model_key], outputs=[infer_model_key])

        latest_btn.click(
            fn=locate_artifacts,
            inputs=[out_path, model_key],
            outputs=[train_status, training_root, artifacts_file, checkpoint_path, config_path, trained_reference],
        )
        latest_btn.click(fn=lambda path: path, inputs=[artifacts_file], outputs=[infer_artifacts])
        latest_btn.click(fn=lambda path: path, inputs=[trained_reference], outputs=[speaker_reference_audio])
        latest_btn.click(fn=lambda key: key, inputs=[model_key], outputs=[infer_model_key])

        inspect_btn.click(
            fn=inspect_artifacts,
            inputs=[infer_artifacts, infer_model_key],
            outputs=[infer_status, training_root, artifacts_file, checkpoint_path, config_path, trained_reference],
        )
        inspect_btn.click(fn=lambda path: path, inputs=[trained_reference], outputs=[speaker_reference_audio])

        tts_btn.click(
            fn=run_inference,
            inputs=[infer_artifacts, infer_model_key, infer_language, tts_text, speaker_reference_audio, out_path],
            outputs=[infer_status, generated_audio, used_reference_audio],
        )

    demo.launch(share=args.share, debug=False, server_port=args.port)
