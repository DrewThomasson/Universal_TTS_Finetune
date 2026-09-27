import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.omnivoice_infer import synthesize_omnivoice


class OmniVoiceRuntimeRelocationTests(unittest.TestCase):
    def test_moved_artifact_uses_local_managed_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            ready = Path(folder)
            (ready / "model").mkdir()
            base = ready / "base"
            base.mkdir()
            output = ready / "audio.wav"
            runtime = {"python_executable": sys.executable, "base_model": str(base), "hf_home": str(ready / "cache")}

            def run(command, **kwargs):
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(command[3], str(base))
                self.assertEqual(command[4], str(ready / "model"))
                self.assertEqual(kwargs["env"]["HF_HOME"], runtime["hf_home"])
                output.write_bytes(b"audio")
                return SimpleNamespace(returncode=0)

            with patch.dict(os.environ, {}, clear=True), \
                 patch("setup_omnivoice.is_ready", return_value=False), \
                 patch("setup_omnivoice.setup", return_value=runtime) as setup, \
                 patch("utils.omnivoice_infer.subprocess.run", side_effect=run):
                synthesize_omnivoice({"checkpoint": "/missing/model", "python_executable": "/missing/python",
                                     "base_model": "/missing/base", "artifacts_file": str(ready / "artifacts.json")},
                                    "Hello.", "en", output, device="cpu")
                setup.assert_called_once_with(cpu=True, progress=None)

    def test_explicit_interpreter_overrides_saved_interpreter(self):
        with tempfile.TemporaryDirectory() as folder:
            ready = Path(folder)
            output = ready / "audio.wav"

            def run(command, **kwargs):
                self.assertEqual(command[0], sys.executable)
                output.write_bytes(b"audio")
                return SimpleNamespace(returncode=0)

            with patch.dict(os.environ, {"UFT_OMNIVOICE_PYTHON": sys.executable}), \
                 patch("setup_omnivoice.setup") as setup, \
                 patch("utils.omnivoice_infer.subprocess.run", side_effect=run):
                synthesize_omnivoice({"checkpoint": folder, "python_executable": "/stale/python",
                                     "base_model": folder}, "Hello.", "en", output, device="cpu")
                setup.assert_not_called()

    def test_invalid_override_fails_before_downloading(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {"UFT_OMNIVOICE_PYTHON": "/missing/override"}), \
             patch("setup_omnivoice.setup") as setup:
            with self.assertRaisesRegex(FileNotFoundError, "override"):
                synthesize_omnivoice({"checkpoint": folder}, "Hello.", "en", Path(folder) / "out.wav")
            setup.assert_not_called()
