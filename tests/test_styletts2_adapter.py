"""Small regression checks for the optional StyleTTS2 adapter."""
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.styletts2_utils import _make_ood_texts, _read_uft_rows, train_styletts2
from utils.styletts2_infer import _WORKER, _artifact_file, _pick_runtime


class StyleTTS2AdapterTests(unittest.TestCase):
    @staticmethod
    def _repo(path):
        (path / "Demo").mkdir(parents=True)
        (path / "models.py").write_text("", encoding="utf-8")
        (path / "Demo" / "Inference_LibriTTS.ipynb").write_text("{}", encoding="utf-8")
        return path

    def test_saved_runtime_paths_are_retained_when_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved_repo = self._repo(root / "saved-source")
            managed_repo = self._repo(root / "managed-source")
            saved_python = Path(sys.executable)
            with patch("setup_styletts2.runtime_paths", return_value=(managed_repo, root / "model.pth", saved_python.parent / "missing-python")), \
                 patch("setup_styletts2.setup") as setup:
                repo, python = _pick_runtime(
                    artifacts={"styletts2_repo": str(saved_repo), "python_executable": str(saved_python)},
                    explicit_repo=None, explicit_python=None, cpu=True,
                )
            self.assertEqual(repo, saved_repo.resolve())
            self.assertEqual(python, str(saved_python))
            setup.assert_not_called()

    def test_missing_saved_paths_relocate_to_managed_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            managed_repo = self._repo(root / "managed-source")
            managed_python = Path(sys.executable)
            with patch("setup_styletts2.runtime_paths", return_value=(root / "not-installed", root / "model.pth", root / "missing-python")), \
                 patch("setup_styletts2.setup", return_value=(managed_repo, root / "model.pth", managed_python)) as setup:
                repo, python = _pick_runtime(
                    artifacts={"styletts2_repo": "/host/models/styletts2/source", "python_executable": "/host/.venv/bin/python"},
                    explicit_repo=None, explicit_python=None, cpu=True,
                )
            self.assertEqual(repo, managed_repo.resolve())
            self.assertEqual(python, str(managed_python))
            setup.assert_called_once_with(cpu=True, progress=None)

    def test_explicit_runtime_overrides_win(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            override_repo = self._repo(root / "override-source")
            with patch("setup_styletts2.runtime_paths", return_value=(root / "unused", root / "model.pth", root / "missing-python")), \
                 patch("setup_styletts2.setup") as setup:
                repo, python = _pick_runtime(
                    artifacts={"styletts2_repo": "/saved/source", "python_executable": "/saved/python"},
                    explicit_repo=str(override_repo), explicit_python=sys.executable, cpu=False,
                )
            self.assertEqual(repo, override_repo.resolve())
            self.assertEqual(python, sys.executable)
            setup.assert_not_called()

    def test_missing_run_files_resolve_next_to_artifacts_file(self):
        with tempfile.TemporaryDirectory() as folder:
            ready = Path(folder) / "moved" / "ready"
            ready.mkdir(parents=True)
            model = ready / "model.pth"
            config = ready / "config_ft.yml"
            model.write_bytes(b"weights")
            config.write_text("{}", encoding="utf-8")
            artifacts = {"artifacts_file": str(ready / "artifacts.json"),
                         "checkpoint": "/host/path/ready/model.pth", "config": "/host/path/ready/config_ft.yml"}
            self.assertEqual(_artifact_file(artifacts, "checkpoint", "model.pth", "checkpoint"), model)
            self.assertEqual(_artifact_file(artifacts, "config", "config_ft.yml", "config"), config)

    def test_worker_rebases_only_unavailable_auxiliary_assets(self):
        self.assertIn("if not Path(config.get(key, '')).is_file()", _WORKER)
        self.assertIn("if not plbert_dir.is_dir() or not list(plbert_dir.glob('step_*.t7'))", _WORKER)

    def test_prepared_ljspeech_clip_ids_resolve_to_wavs(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "metadata_train.csv"
            manifest.write_text("001|hello.|Hello.\n002.wav|goodbye.|Goodbye.\n", encoding="utf-8")
            self.assertEqual(
                _read_uft_rows(Path(folder), "train"),
                [("001.wav", "hello."), ("002.wav", "goodbye.")],
            )

    def test_batch_size_one_is_rejected_before_runtime_or_filesystem_checks(self):
        # The upstream predictor cannot handle batch size 1. Bad paths prove
        # the adapter rejects it before attempting runtime discovery.
        with self.assertRaisesRegex(ValueError, "batch size 2 or greater"):
            train_styletts2(
                dataset_dir="/does/not/exist",
                training_root="/does/not/exist/output",
                styletts2_repo="/does/not/exist/repo",
                pretrained_checkpoint="/does/not/exist/base.pth",
                epochs=1,
                batch_size=1,
                dry_run=True,
            )

    def test_ood_text_generation_produces_two_entries_for_short_utterances(self):
        texts, min_length = _make_ood_texts(
            [("clip1.wav", "Hi."), ("clip2.wav", "Yes.")],
            requested_min_length=50,
        )
        self.assertEqual(len(texts), 2)
        self.assertTrue(all(text.strip() for text in texts))
        self.assertGreaterEqual(min_length, 1)
        self.assertTrue(all(len(text) >= min_length for text in texts))


if __name__ == "__main__":
    unittest.main()
