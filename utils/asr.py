"""Language selection and word-timestamp transcription for dataset preparation."""

from pathlib import Path
from types import SimpleNamespace


MMS_ASR_MODEL = "facebook/mms-1b-all"
MMS_ASR_LANGUAGES = frozenset(
    line.strip() for line in (Path(__file__).parent.parent / "assets" / "mms_asr_languages.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
)


def mms_asr_language(language: str) -> str:
    """Resolve common two-letter codes to an MMS adapter and reject absent adapters."""
    code = language.lower().strip()
    if code not in MMS_ASR_LANGUAGES:
        from langcodes import Language
        code = Language.get(code.replace("_", "-")).to_alpha3()
    if code not in MMS_ASR_LANGUAGES:
        raise ValueError(f"MMS ASR has no published adapter for {language!r}; supply transcripts or choose another ASR language.")
    return code


def select_asr_backend(backend: str, language: str) -> str:
    backend = backend.lower().strip()
    if backend not in {"auto", "whisper", "mms"}:
        raise ValueError("ASR backend must be auto, whisper, or mms.")
    if backend == "auto":
        code = language.strip().lower()
        backend = "mms" if code in MMS_ASR_LANGUAGES or len(code) == 3 else "whisper"
    if backend == "mms":
        mms_asr_language(language)
    return backend


class MMSTranscriber:
    def __init__(self, language: str, cache_dir: str, device: str):
        from transformers import AutoModelForCTC, AutoProcessor, pipeline

        code = mms_asr_language(language)
        processor = AutoProcessor.from_pretrained(MMS_ASR_MODEL, cache_dir=cache_dir)
        processor.tokenizer.set_target_lang(code)
        model = AutoModelForCTC.from_pretrained(MMS_ASR_MODEL, cache_dir=cache_dir)
        model.load_adapter(code)
        self.pipe = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            device=0 if device == "cuda" else -1,
        )

    def transcribe_words(self, audio_path: str):
        result = self.pipe(audio_path, chunk_length_s=20, stride_length_s=2, return_timestamps="word")
        words = []
        for chunk in result.get("chunks", []):
            start, end = chunk.get("timestamp", (None, None))
            word = chunk.get("text", "").strip()
            if word and start is not None and end is not None and end > start:
                words.append(SimpleNamespace(start=start, end=end, word=" " + word))
        return words
