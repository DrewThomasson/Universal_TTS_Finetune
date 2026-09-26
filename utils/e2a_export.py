"""Create flat custom-model ZIP archives accepted by Ebook2audiobook."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


# Matches lib/conf_models.py in ebook2audiobook for engines with custom uploads.
E2A_FILES = {
    "xtts_v1": {"config.json": "config", "model.pth": "checkpoint", "vocab.json": "vocab", "ref.wav": "reference_wav"},
    "xtts_v2": {"config.json": "config", "model.pth": "checkpoint", "vocab.json": "vocab", "ref.wav": "reference_wav"},
    "vits_tts": {"config.json": "config", "best_model.pth": "checkpoint", "ref.wav": "reference_wav"},
    "mms_vits": {"config.json": "config", "G_100000.pth": "checkpoint", "vocab.txt": "vocab", "ref.wav": "reference_wav"},
    "piper": {"config.onnx.json": "config", "model.onnx": "checkpoint", "ref.wav": "reference_wav"},
}


def export_e2a_zip(artifacts: dict, output_file: str) -> dict:
    model_key = artifacts.get("model_key")
    required = E2A_FILES.get(model_key)
    if required is None:
        raise ValueError(f"E2A custom model upload does not support UFT engine {model_key!r}.")
    sources = {}
    for name, key in required.items():
        value = artifacts.get(key)
        if not value and key == "reference_wav":
            dataset_dir = artifacts.get("dataset_dir")
            if dataset_dir:
                import json
                info_file = Path(dataset_dir) / "dataset_info.json"
                if info_file.is_file():
                    value = json.loads(info_file.read_text(encoding="utf-8")).get("reference_wav")
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"Cannot export {model_key}: required E2A file {name} ({key}) is missing: {value!r}")
        sources[name] = Path(value)
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name, source in sources.items():
                archive.write(source, name)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"model_key": model_key, "zip_file": str(destination), "files": list(required)}
