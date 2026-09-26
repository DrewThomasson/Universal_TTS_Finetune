import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_models_dir()->Path:
    configured = os.environ.get('UFT_MODELS_DIR')
    models_dir = Path(configured).expanduser().resolve() if configured else _PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
