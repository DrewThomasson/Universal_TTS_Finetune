import tempfile
import unittest
from unittest.mock import patch

from utils.styletts2_utils import train_styletts2


class StyleTTS2DatasetLimitsTests(unittest.TestCase):
    def test_incomplete_validation_batch_is_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("utils.styletts2_utils._read_uft_rows", side_effect=[[{}, {}], [{}]]), \
             patch("utils.styletts2_utils._validate_runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "at least 2 validation clips"):
                train_styletts2(dataset_dir=folder, training_root=folder,
                               styletts2_repo="missing", pretrained_checkpoint="missing", batch_size=2)
            runtime.assert_not_called()
