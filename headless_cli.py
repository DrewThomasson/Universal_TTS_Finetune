from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.pipeline import (
    default_test_output,
    dropdown_choices,
    find_latest_artifacts,
    list_supported_models,
    prepare_dataset,
    synthesize,
    train_model,
)


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless workflow for Universal Coqui TTS fine-tuning.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List supported training recipes.")

    prepare = subparsers.add_parser("prepare-dataset", help="Build an LJSpeech-style dataset from audio files.")
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--audio-dir")
    prepare.add_argument("--audio-file", action="append", default=[])
    prepare.add_argument("--transcript-file")
    prepare.add_argument("--language", default="en")
    prepare.add_argument("--whisper-model", default="small")
    prepare.add_argument("--eval-percentage", type=float, default=0.15)
    prepare.add_argument("--min-segment-seconds", type=float, default=0.5)
    prepare.add_argument("--max-segment-seconds", type=float, default=12.0)
    prepare.add_argument("--segment-buffer-seconds", type=float, default=0.2)

    train = subparsers.add_parser("train", help="Train or fine-tune a selected Coqui recipe.")
    train.add_argument("--model", required=True, choices=[key for key, _ in dropdown_choices()])
    train.add_argument("--output-root", required=True)
    train.add_argument("--dataset-dir")
    train.add_argument("--language", default="en")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--grad-accum", type=int, default=1)
    train.add_argument("--max-audio-seconds", type=int, default=11)
    train.add_argument("--restore-path")
    train.add_argument("--extra-overrides-json")
    train.add_argument("--no-pretrained", action="store_true")
    train.add_argument("--dry-run", action="store_true")

    infer = subparsers.add_parser("synthesize", help="Generate speech from the latest or selected trained model.")
    infer.add_argument("--artifacts", required=True, help="Path to artifacts.json, a ready/ folder, or an output root.")
    infer.add_argument("--model")
    infer.add_argument("--text", required=True)
    infer.add_argument("--language", default="en")
    infer.add_argument("--speaker-wav")
    infer.add_argument("--output-file")

    workflow = subparsers.add_parser("workflow", help="Prepare dataset, train, and optionally synthesize in one command.")
    workflow.add_argument("--model", required=True, choices=[key for key, _ in dropdown_choices()])
    workflow.add_argument("--output-root", required=True)
    workflow.add_argument("--audio-dir")
    workflow.add_argument("--audio-file", action="append", default=[])
    workflow.add_argument("--transcript-file")
    workflow.add_argument("--language", default="en")
    workflow.add_argument("--whisper-model", default="small")
    workflow.add_argument("--epochs", type=int, default=10)
    workflow.add_argument("--batch-size", type=int, default=8)
    workflow.add_argument("--grad-accum", type=int, default=1)
    workflow.add_argument("--max-audio-seconds", type=int, default=11)
    workflow.add_argument("--restore-path")
    workflow.add_argument("--extra-overrides-json")
    workflow.add_argument("--no-pretrained", action="store_true")
    workflow.add_argument("--test-text")
    workflow.add_argument("--speaker-wav")
    workflow.add_argument("--output-file")

    latest = subparsers.add_parser("latest-artifacts", help="Resolve the newest trained model artifacts.")
    latest.add_argument("--output-root", required=True)
    latest.add_argument("--model")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list-models":
        _print_json({"models": list_supported_models()})
        return

    if args.command == "prepare-dataset":
        result = prepare_dataset(
            output_root=args.output_root,
            audio_dir=args.audio_dir,
            audio_files=args.audio_file,
            transcript_file=args.transcript_file,
            language=args.language,
            whisper_model_name=args.whisper_model,
            eval_percentage=args.eval_percentage,
            min_segment_seconds=args.min_segment_seconds,
            max_segment_seconds=args.max_segment_seconds,
            segment_buffer_seconds=args.segment_buffer_seconds,
        )
        _print_json(result)
        return

    if args.command == "train":
        result = train_model(
            model_key=args.model,
            output_root=args.output_root,
            dataset_dir=args.dataset_dir,
            language=args.language,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_audio_seconds=args.max_audio_seconds,
            restore_path=args.restore_path,
            use_pretrained=not args.no_pretrained,
            extra_overrides_json=args.extra_overrides_json,
            dry_run=args.dry_run,
        )
        _print_json(result)
        return

    if args.command == "synthesize":
        output_file = args.output_file or default_test_output(str(Path(args.artifacts).resolve()))
        result = synthesize(
            artifacts_path_or_dir=args.artifacts,
            model_key=args.model,
            text=args.text,
            language=args.language,
            speaker_wav=args.speaker_wav,
            output_file=output_file,
        )
        _print_json(result)
        return

    if args.command == "workflow":
        dataset = prepare_dataset(
            output_root=args.output_root,
            audio_dir=args.audio_dir,
            audio_files=args.audio_file,
            transcript_file=args.transcript_file,
            language=args.language,
            whisper_model_name=args.whisper_model,
        )
        training = train_model(
            model_key=args.model,
            output_root=args.output_root,
            dataset_dir=dataset["dataset_dir"],
            language=args.language,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_audio_seconds=args.max_audio_seconds,
            restore_path=args.restore_path,
            use_pretrained=not args.no_pretrained,
            extra_overrides_json=args.extra_overrides_json,
        )
        payload = {"dataset": dataset, "training": training}
        if args.test_text:
            payload["synthesis"] = synthesize(
                artifacts_path_or_dir=training["training_root"],
                model_key=args.model,
                text=args.test_text,
                language=args.language,
                speaker_wav=args.speaker_wav,
                output_file=args.output_file or default_test_output(args.output_root),
            )
        _print_json(payload)
        return

    if args.command == "latest-artifacts":
        _print_json(find_latest_artifacts(args.output_root, model_key=args.model))
        return


if __name__ == "__main__":
    main()
