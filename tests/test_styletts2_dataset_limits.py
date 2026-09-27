import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from utils.styletts2_utils import _stage_cpu_compatibility, train_styletts2


class StyleTTS2DatasetLimitsTests(unittest.TestCase):
    def test_incomplete_validation_batch_is_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("utils.styletts2_utils._read_uft_rows", side_effect=[[{}, {}], [{}]]), \
             patch("utils.styletts2_utils._validate_runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "at least 2 validation clips"):
                train_styletts2(dataset_dir=folder, training_root=folder,
                               styletts2_repo="missing", pretrained_checkpoint="missing", batch_size=2)
            runtime.assert_not_called()

    def test_cpu_compatibility_stages_modules_and_preserves_upstream_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "upstream"
            modules = repo / "Modules"
            modules.mkdir(parents=True)
            (modules / "__init__.py").write_text("", encoding="utf-8")
            source = {
                "discriminators.py": "WINDOW = 'self.window.to(y.get_device())'\n",
                "hifigan.py": "F0 = \"torch.ones(1, 1, F0_down).to('cuda')\"\nN = \"torch.ones(1, 1, N_down).to('cuda')\"\n",
                "istftnet.py": "F0 = \"torch.ones(1, 1, F0_down).to('cuda')\"\nN = \"torch.ones(1, 1, N_down).to('cuda')\"\n",
            }
            for filename, contents in source.items():
                (modules / filename).write_text(contents, encoding="utf-8")
            official_script = "length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')"
            staged_script, staged_modules = _stage_cpu_compatibility(repo, root / "run", official_script)
            self.assertIn(".to(device)", staged_script.read_text(encoding="utf-8"))
            self.assertIn("self.window.to(y.device)", (staged_modules / "discriminators.py").read_text(encoding="utf-8"))
            self.assertIn(".to(F0_curve.device)", (staged_modules / "hifigan.py").read_text(encoding="utf-8"))
            self.assertIn(".to(N.device)", (staged_modules / "istftnet.py").read_text(encoding="utf-8"))
            for filename, contents in source.items():
                self.assertEqual((modules / filename).read_text(encoding="utf-8"), contents)

            env = {**os.environ, "PYTHONPATH": str(root / "run") + os.pathsep + str(repo)}
            probe = subprocess.run(
                [sys.executable, "-c", "import Modules.discriminators; print(Modules.discriminators.__file__)"],
                cwd=root / "run", env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(Path(probe.stdout.strip()).resolve(), (staged_modules / "discriminators.py").resolve())

    def test_cpu_compatibility_rejects_changed_upstream_literals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "upstream"
            (repo / "Modules").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "compatibility point changed"):
                _stage_cpu_compatibility(repo, root / "run", "unexpected validation code")
