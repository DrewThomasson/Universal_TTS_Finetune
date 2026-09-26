import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils import omnivoice_utils as adapter


class OmniVoiceAdapterTests(unittest.TestCase):
    def test_extraction_launcher_uses_one_fork_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            launcher = Path(temp_dir) / "extract.py"
            adapter._write_extraction_launcher(launcher)
            contents = launcher.read_text()
        self.assertIn("mp.set_start_method('fork', force=True)", contents)
        self.assertIn("upstream.mp.set_start_method = lambda", contents)
        self.assertIn("upstream.main()", contents)

    def test_adapter_uses_selected_python_and_emits_artifact_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            audio = tmp_path / "clip.wav"
            audio.write_bytes(b"audio")
            manifest = tmp_path / "train.jsonl"
            manifest.write_text(json.dumps({"id": "clip", "audio_path": str(audio), "text": "hello"}) + "\n")
            base_model = tmp_path / "base"
            base_model.mkdir()
            tokenizer = tmp_path / "tokenizer"
            tokenizer.mkdir()
            output = tmp_path / "out"
            chosen_python = shutil.which(sys.executable)
            commands = []

            def fake_run(command, cwd, env, log_path, progress):
                commands.append(command)
                if "extract_audio_tokens_uft.py" in command[1]:
                    tar_pattern = Path(command[command.index("--tar_output_pattern") + 1])
                    shard_dir = tar_pattern.parent.parent
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    (shard_dir / "data.lst").write_text("dataset\n")

            with patch.object(adapter, "_require_module"), \
                 patch.object(adapter, "_cuda_preflight"), \
                 patch.object(adapter, "_local_hf_snapshot", return_value=tmp_path / "llm"), \
                 patch.object(adapter, "_run", side_effect=fake_run), \
                 patch.object(adapter.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * adapter.GIB)):
                result = adapter.run_omnivoice_finetune(
                    manifest,
                    output,
                    base_model=str(base_model),
                    audio_tokenizer=str(tokenizer),
                    steps=1,
                    save_steps=1,
                    python_executable=chosen_python,
                )

            selected = str(Path(chosen_python).absolute())
            self.assertEqual(result, output / "checkpoints")
            self.assertEqual(commands[0][0], selected)
            self.assertEqual(commands[1][:3], [selected, "-m", "accelerate.commands.launch"])
            train_config = json.loads((output / "omnivoice_work/config/train.json").read_text())
            self.assertEqual(train_config["num_workers"], 1)
            self.assertEqual(train_config["batch_tokens"], 512)
            artifact = json.loads((output / "artifacts.json").read_text())
            self.assertEqual(artifact["python_executable"], selected)

    def test_invalid_steps_fail_before_environment_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "positive integers"):
                adapter.run_omnivoice_finetune("missing.jsonl", temp_dir, steps=0)


if __name__ == "__main__":
    unittest.main()
