"""Mocked routing checks for UFT inference; these do not load models or synthesize audio."""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from utils.styletts2_infer import _WORKER


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_import_stubs():
    """Allow importing pipeline.py in a minimal Python environment."""
    class DummyTensor:
        def unsqueeze(self, _dimension):
            return self

    torch = _module(
        "torch",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda *args, **kwargs: None,
        tensor=lambda value: DummyTensor(),
    )
    tts_api = _module("TTS.api", TTS=object)
    xtts_config = _module("TTS.tts.configs.xtts_config", XttsConfig=object)
    xtts_model = _module("TTS.tts.models.xtts", Xtts=object)
    tts_manage = _module("TTS.utils.manage", ModelManager=object)
    modules = {
        "numpy": _module("numpy"),
        "scipy": _module("scipy"),
        "scipy.cluster": _module("scipy.cluster"),
        "scipy.cluster.hierarchy": _module("scipy.cluster.hierarchy", fcluster=object(), linkage=object()),
        "scipy.spatial": _module("scipy.spatial"),
        "scipy.spatial.distance": _module("scipy.spatial.distance", pdist=object()),
        "soundfile": _module("soundfile"),
        "torch": torch,
        "torchaudio": _module("torchaudio"),
        "faster_whisper": _module("faster_whisper", WhisperModel=object),
        "TTS": _module("TTS", __path__=[]),
        "TTS.api": tts_api,
        "TTS.tts": _module("TTS.tts", __path__=[]),
        "TTS.tts.configs": _module("TTS.tts.configs", __path__=[]),
        "TTS.tts.configs.xtts_config": xtts_config,
        "TTS.tts.models": _module("TTS.tts.models", __path__=[]),
        "TTS.tts.models.xtts": xtts_model,
        "TTS.utils": _module("TTS.utils", __path__=[]),
        "TTS.utils.manage": tts_manage,
        "utils.tokenizer": _module("utils.tokenizer", multilingual_cleaners=object()),
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


_install_import_stubs()
pipeline = importlib.import_module("utils.pipeline")


class _FakeProgress:
    def __call__(self, *_args, **_kwargs):
        return None


sys.modules.setdefault(
    "gradio",
    _module("gradio", Progress=_FakeProgress, update=lambda **kwargs: kwargs),
)
web_gui = importlib.import_module("web_gui")


class InferenceDispatchTests(unittest.TestCase):
    """Exercise every built-in engine branch with fake artifacts and runtimes."""

    def test_all_17_builtin_engines_route_to_the_expected_runtime(self):
        coqui_keys = (
            "align_tts", "delightful_tts", "fast_pitch", "fast_speech", "fastspeech2",
            "glow_tts", "neuralhmm_tts", "overflow", "speedy_speech",
            "tacotron2_capacitron", "tacotron2_dca", "tacotron2_ddc", "vits_tts",
        )
        cases = [(key, "tts") for key in coqui_keys]
        cases += [("xtts_v1", "xtts"), ("xtts_v2", "xtts"), ("mms_vits", "mms"), ("piper", "piper")]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for key, family in cases:
                with self.subTest(engine=key):
                    artifacts = {
                        "model_key": key,
                        "model_label": key,
                        "family": family,
                        "training_root": str(root),
                        "artifacts_file": str(root / "artifacts.json"),
                        "checkpoint": str(root / ("model.onnx" if family == "piper" else "model.pth")),
                        "config": str(root / ("model.onnx.json" if family == "piper" else "config.json")),
                        "reference_wav": str(root / "reference.wav") if family == "xtts" else "",
                    }
                    runtime = Mock()
                    fake_xtts = Mock()
                    fake_xtts.config = types.SimpleNamespace(
                        gpt_cond_len=1, max_ref_len=1, sound_norm_refs=False,
                        temperature=0.7, length_penalty=1, repetition_penalty=1,
                        top_k=1, top_p=1,
                    )
                    fake_xtts.get_conditioning_latents.return_value = ("latent", "embedding")
                    fake_xtts.inference.return_value = {"wav": [0.0]}
                    piper_synth = Mock()
                    piper_module = _module("utils.piper_utils", synthesize_piper=piper_synth)
                    with patch.object(pipeline, "load_artifacts", return_value=artifacts), \
                         patch.object(pipeline, "_load_tts_runtime", return_value=runtime) as load_tts, \
                         patch.object(pipeline, "_load_xtts_runtime", return_value=fake_xtts) as load_xtts, \
                         patch.object(pipeline, "_save_waveform") as save_waveform, \
                         patch.dict(sys.modules, {"utils.piper_utils": piper_module}):
                        result = pipeline.synthesize(
                            artifacts_path_or_dir=str(root),
                            text="Mocked dispatch sample.",
                            output_file=str(root / f"{key}.wav"),
                            language="en",
                            device="cpu",
                        )

                    self.assertEqual(result["model_key"], key)
                    if family == "xtts":
                        load_xtts.assert_called_once_with(artifacts, "cpu")
                        fake_xtts.inference.assert_called_once()
                        save_waveform.assert_called_once()
                        load_tts.assert_not_called()
                    elif family == "piper":
                        piper_synth.assert_called_once()
                        self.assertEqual(piper_synth.call_args.kwargs["onnx_path"], artifacts["checkpoint"])
                        load_tts.assert_not_called()
                    else:
                        load_tts.assert_called_once_with(artifacts, None, "cpu")
                        runtime.tts_to_file.assert_called_once()

    def test_styletts2_routes_to_local_optional_adapter_with_reference_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = {"family": "styletts2", "model_key": "styletts2", "reference_wav": ""}
            adapter_call = Mock(return_value={"model_key": "styletts2", "output_file": str(root / "out.wav")})
            adapter = _module("utils.styletts2_infer", synthesize_styletts2=adapter_call)
            with patch.object(pipeline, "load_artifacts", return_value=artifacts), \
                 patch.dict(sys.modules, {"utils.styletts2_infer": adapter}):
                result = pipeline.synthesize(
                    artifacts_path_or_dir=str(root), text="sample", speaker_wav="reference.wav",
                    output_file=str(root / "out.wav"),
                )
            adapter_call.assert_called_once()
            self.assertEqual(adapter_call.call_args.kwargs["reference_wav"], "reference.wav")
            self.assertEqual(result["model_key"], "styletts2")

    def test_omnivoice_routes_to_isolated_optional_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = {"family": "omnivoice", "model_key": "omnivoice", "language": "en", "artifacts_file": str(root / "artifacts.json")}
            route = Mock(return_value=root / "out.wav")
            adapter = _module("utils.omnivoice_infer", synthesize_omnivoice=route)
            with patch.object(pipeline, "load_artifacts", return_value=artifacts), \
                 patch.dict(sys.modules, {"utils.omnivoice_infer": adapter}):
                result = pipeline.synthesize(
                    artifacts_path_or_dir=str(root), text="sample", output_file=str(root / "out.wav"),
                )
            route.assert_called_once()
            self.assertEqual(route.call_args.args[0], artifacts)
            self.assertEqual(route.call_args.args[2], "en")
            self.assertEqual(result["model_key"], "omnivoice")

    def test_omnivoice_published_language_catalog_and_invalid_language_rejection(self):
        from utils.model_registry import OMNIVOICE_LANGUAGES, pretrained_model_choices
        self.assertEqual(len(OMNIVOICE_LANGUAGES), 646)
        self.assertTrue(pretrained_model_choices("omnivoice", "en"))
        self.assertFalse(pretrained_model_choices("omnivoice", "not-a-language"))
        self.assertEqual(len(web_gui.OMNIVOICE_LANGUAGE_CHOICES), 646)

    def test_omnivoice_rejects_reference_wav_instead_of_silently_ignoring_it(self):
        artifacts = {"family": "omnivoice", "language": "es"}
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "load_artifacts", return_value=artifacts):
            with self.assertRaisesRegex(ValueError, "reference voice cloning"):
                pipeline.synthesize(artifacts_path_or_dir=folder, text="Hello", output_file=str(Path(folder) / "out.wav"), speaker_wav="reference.wav")

    def test_omnivoice_invalid_training_language_rejected_before_dataset_access(self):
        with self.assertRaisesRegex(ValueError, "no supported starting checkpoint"):
            pipeline.train_model(
                model_key="omnivoice", output_root="unused", dataset_dir="unused",
                language="not-a-language", batch_size=1, grad_accum=1,
            )

    def test_styletts2_official_worker_source_is_syntactically_valid(self):
        compile(_WORKER, "styletts2_inference_worker.py", "exec")

    def test_gui_inference_choices_include_styletts2(self):
        gui_keys = {key for _label, key in web_gui.INFERENCE_MODEL_CHOICES}
        expected_keys = {key for key, _family in [
            ("align_tts", "tts"), ("delightful_tts", "tts"), ("fast_pitch", "tts"),
            ("fast_speech", "tts"), ("fastspeech2", "tts"), ("glow_tts", "tts"),
            ("neuralhmm_tts", "tts"), ("overflow", "tts"), ("speedy_speech", "tts"),
            ("tacotron2_capacitron", "tts"), ("tacotron2_dca", "tts"),
            ("tacotron2_ddc", "tts"), ("vits_tts", "tts"), ("mms_vits", "mms"),
            ("xtts_v1", "xtts"), ("xtts_v2", "xtts"), ("piper", "piper"),
        ]}
        self.assertEqual(gui_keys, expected_keys | {"styletts2", "omnivoice", "f5_tts"})

    def test_gui_run_inference_delegates_once_to_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "sample.wav"
            output_file.write_bytes(b"mock")
            result = {
                "output_file": str(output_file),
                "speaker_wav": "",
            }
            with patch.object(web_gui, "default_test_output", return_value=str(output_file)), \
                 patch.object(web_gui, "synthesize", return_value=result) as synthesize:
                response = web_gui.run_inference(
                    "artifacts.json", "vits_tts", "en", "hello", None, temp_dir, progress=None
                )

            synthesize.assert_called_once()
            kwargs = synthesize.call_args.kwargs
            self.assertEqual(kwargs["artifacts_path_or_dir"], "artifacts.json")
            self.assertEqual(kwargs["model_key"], "vits_tts")
            self.assertEqual(kwargs["language"], "en")
            self.assertEqual(kwargs["text"], "hello")
            self.assertEqual(response[0], "Speech generated.")
            self.assertEqual(response[1], str(output_file.resolve()))


if __name__ == "__main__":
    unittest.main()
