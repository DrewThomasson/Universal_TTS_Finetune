"""Resolve languages used by the Coqui recipes' phoneme tokenizers."""

from functools import lru_cache
from pathlib import Path


_PHONEMIZER_ALIASES = {
    "en": "en-us",
    "fr": "fr-fr",
    "ja": "ja-jp",
    "ko": "ko-kr",
    "no": "nb",
}
_PROBE_TEXT = {
    "be": "Прывітанне",
    "bn": "নমস্কার",
    "ja": "こんにちは",
    "ko": "안녕하세요",
    "zh-cn": "你好",
}


@lru_cache(maxsize=128)
def coqui_phoneme_language(language: str) -> str:
    """Return a supported phonemizer code or fail before downloading a model."""
    try:
        from TTS.tts.utils.text import phonemizers
    except (ImportError, OSError) as exc:
        raise ValueError("Coqui phonemizer dependencies are unavailable.") from exc

    code = language.lower().replace("_", "-")
    candidates = (_PHONEMIZER_ALIASES[code], code) if code in _PHONEMIZER_ALIASES else (code,)
    last_error = None
    for candidate in candidates:
        try:
            get_default = getattr(phonemizers, "get_default_phonemizer", None)
            name = get_default(candidate) if get_default else phonemizers.DEF_LANG_TO_PHONEMIZER.get(candidate)
            if not name:
                continue
            phonemizer = phonemizers.get_phonemizer_by_name(name, language=candidate)
            if name != "espeak":
                phonemizer.phonemize(_PROBE_TEXT.get(code, "test"))
            return candidate
        except Exception as exc:
            last_error = exc
    detail = f" ({last_error})" if last_error else ""
    raise ValueError(
        f"No Coqui phonemizer is available for '{language}'{detail}. "
        "Install a phonemizer for this language or select another training language."
    )


def coqui_dataset_extra_symbols(dataset_dir: Path, language: str, cleaner_name: str) -> str:
    """Find phonemes the default Coqui IPA tokenizer would silently discard."""
    from TTS.tts.utils.text.characters import IPAPhonemes
    from TTS.tts.utils.text import cleaners
    from TTS.tts.utils.text import phonemizers

    code = coqui_phoneme_language(language)
    name = phonemizers.DEF_LANG_TO_PHONEMIZER[code]
    phonemizer = phonemizers.get_phonemizer_by_name(name, language=code)
    cleaner = getattr(cleaners, cleaner_name)
    known = set(IPAPhonemes().vocab)
    used = set()
    metadata = dataset_dir / "metadata.csv"
    with metadata.open(encoding="utf-8") as rows:
        for line_number, row in enumerate(rows, 1):
            fields = row.rstrip("\n").split("|", 2)
            if len(fields) != 3:
                raise ValueError(f"Invalid LJSpeech metadata at {metadata}:{line_number}.")
            try:
                cleaned = cleaner(fields[2].strip())
                used.update(phonemizer.phonemize(cleaned, separator=""))
            except Exception as exc:
                raise ValueError(
                    f"Coqui cannot phonemize {metadata}:{line_number} as '{language}': {exc}"
                ) from exc
    return "".join(sorted(used - known))
