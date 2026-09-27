import unittest
from unittest.mock import patch

from utils.device import child_environment, select_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_stays_cpu_even_when_cuda_is_available(self):
        self.assertEqual(select_device("cpu", cuda_available=True), "cpu")
        self.assertEqual(select_device("auto", cuda_available=True), "cuda")
        self.assertEqual(select_device("auto", cuda_available=False), "cpu")

    def test_unavailable_explicit_cuda_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_device("cuda", cuda_available=False)

    def test_cpu_hides_cuda_from_children(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            self.assertEqual(child_environment("cpu")["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(child_environment("cuda")["CUDA_VISIBLE_DEVICES"], "0")


if __name__ == "__main__":
    unittest.main()
