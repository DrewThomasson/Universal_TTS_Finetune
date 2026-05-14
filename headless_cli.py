from __future__ import annotations

import argparse
import json

from utils.pipeline import (
    default_test_output,
    dropdown_choices,
    find_latest_artifacts,
    list_supported_models,
    load_artifacts,
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
    train.add_argument("--no-stream-logs", action="store_true", help="Disable streaming of training logs to the console")

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
    workflow.add_argument("--no-stream-logs", action="store_true", help="Disable streaming of training logs to the console")

    batch_test = subparsers.add_parser("batch-test", help="Test all supported models sequentially on the same dataset.")
    batch_test.add_argument("--output-root", required=True)
    batch_test.add_argument("--audio-dir")
    batch_test.add_argument("--audio-file", action="append", default=[])
    batch_test.add_argument("--transcript-file")
    batch_test.add_argument("--language", default="en")
    batch_test.add_argument("--whisper-model", default="small")
    batch_test.add_argument("--epochs", type=int, default=1)
    batch_test.add_argument("--batch-size", type=int, default=8)
    batch_test.add_argument("--grad-accum", type=int, default=1)
    batch_test.add_argument("--max-audio-seconds", type=int, default=11)
    batch_test.add_argument("--test-text", default="This is a quick validation sample from the batch test.")
    batch_test.add_argument("--discard-models", action="store_true", help="Delete model checkpoints after generating sample audio to save space.")
    batch_test.add_argument("--no-stream-logs", action="store_true", help="Disable streaming of training logs to the console")
    batch_test.add_argument("--extra-overrides-json")

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
            stream_logs=not args.no_stream_logs,
        )
        _print_json(result)
        return

    if args.command == "synthesize":
        artifacts = load_artifacts(args.artifacts, model_key=args.model)
        output_file = args.output_file or default_test_output(artifacts["training_root"])
        result = synthesize(
            artifacts_path_or_dir=artifacts["artifacts_file"],
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
            stream_logs=not args.no_stream_logs,
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

    if args.command == "batch-test":
        dataset = prepare_dataset(
            output_root=args.output_root,
            audio_dir=args.audio_dir,
            audio_files=args.audio_file,
            transcript_file=args.transcript_file,
            language=args.language,
            whisper_model_name=args.whisper_model,
        )
        
        import shutil
        from pathlib import Path
        import traceback
        
        batch_results_dir = Path(args.output_root) / "batch_results"
        batch_results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {"dataset": dataset, "models": {}}
        
        for model_key, model_label in dropdown_choices():
            print(f"\n==================================================")
            print(f"Batch testing: {model_label} ({model_key})")
            print(f"==================================================\n")
            
            try:
                training = train_model(
                    model_key=model_key,
                    output_root=args.output_root,
                    dataset_dir=dataset["dataset_dir"],
                    language=args.language,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    grad_accum=args.grad_accum,
                    max_audio_seconds=args.max_audio_seconds,
                    restore_path=None,
                    use_pretrained=True,
                    extra_overrides_json=args.extra_overrides_json,
                    stream_logs=not args.no_stream_logs,
                )
                
                output_wav = batch_results_dir / f"{model_key}.wav"
                
                synthesis = synthesize(
                    artifacts_path_or_dir=training["training_root"],
                    model_key=model_key,
                    text=args.test_text,
                    language=args.language,
                    speaker_wav=None,
                    output_file=str(output_wav),
                )
                
                results["models"][model_key] = {
                    "status": "success",
                    "training": training,
                    "synthesis": synthesis,
                    "sample_audio": str(output_wav)
                }
                
                if args.discard_models:
                    print(f"Discarding model artifacts for {model_key} to save space...")
                    shutil.rmtree(training["training_root"], ignore_errors=True)
                    results["models"][model_key]["discarded"] = True
                    
            except Exception as e:
                print(f"FAILED to test {model_key}: {e}")
                traceback.print_exc()
                results["models"][model_key] = {
                    "status": "error",
                    "error_message": str(e)
                }
                
        _print_json(results)
        return

    if args.command == "latest-artifacts":
        _print_json(find_latest_artifacts(args.output_root, model_key=args.model))
        return


if __name__ == "__main__":
    main()
