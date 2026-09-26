"""Small regression checks for the optional StyleTTS2 adapter."""
import tempfile
import unittest
from pathlib import Path

from utils.styletts2_utils import _make_ood_texts, train_styletts2


class StyleTTS2AdapterTests(unittest.TestCase):
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
