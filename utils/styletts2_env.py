"""Runtime environment helpers for the optional StyleTTS2 engine."""
from __future__ import annotations

import os
from pathlib import Path


def configure_styletts2_hf_home(environment: dict[str, str], repo: str | os.PathLike[str],
                               managed_repo: str | os.PathLike[str]) -> None:
    """Keep managed StyleTTS2 Hugging Face assets under its relocatable runtime."""
    managed = Path(managed_repo).expanduser().resolve()
    if Path(repo).expanduser().resolve() == managed:
        environment["HF_HOME"] = str(managed.parent / "huggingface")
