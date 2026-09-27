"""Small checks for optional runtime cache discovery and downloads."""
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup_omnivoice
import setup_styletts2


class OptionalRuntimeSetupTests(unittest.TestCase):
    def test_missing_runtime_is_not_ready(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"UFT_MODELS_DIR": folder}):
            self.assertFalse(setup_styletts2.is_ready())
            self.assertFalse(setup_omnivoice.is_ready())

    def test_checksum_download_and_cached_reuse(self):
        payload = b"test model weights"
        checksum = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.pth"
            with patch("setup_styletts2.urllib.request.urlopen", return_value=io.BytesIO(payload)) as download:
                setup_styletts2._download("https://example.invalid/model", destination,
                                          minimum_size=1, sha256=checksum)
                self.assertEqual(destination.read_bytes(), payload)
                setup_styletts2._download("https://example.invalid/model", destination,
                                          minimum_size=1, sha256=checksum)
                download.assert_called_once()

    def test_failed_checksum_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.pth"
            with patch("setup_styletts2.urllib.request.urlopen", return_value=io.BytesIO(b"bad weights")):
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    setup_styletts2._download("https://example.invalid/model", destination,
                                              minimum_size=1, sha256="0" * 64)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("model.pth.download").exists())


if __name__ == "__main__":
    unittest.main()
