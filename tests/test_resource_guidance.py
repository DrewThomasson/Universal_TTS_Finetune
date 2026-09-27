import unittest
from unittest.mock import patch

from utils.model_registry import MODEL_SPECS
from utils.resource_guidance import training_resource_guidance


class ResourceGuidanceTests(unittest.TestCase):
    def test_every_engine_has_device_and_memory_guidance(self):
        with patch("utils.resource_guidance._available_ram_gib", return_value=30.0), patch(
            "utils.resource_guidance._gpu_memory", return_value=(12.0, 10.0)
        ):
            for spec in MODEL_SPECS:
                for device in ("cpu", "cuda"):
                    with self.subTest(model=spec.key, device=device):
                        message = training_resource_guidance(spec.key, device)
                        self.assertIn(spec.label, message)
                        self.assertIn(f"selected device: {device}", message)
                        self.assertIn("GiB system RAM", message)
                        if device == "cuda":
                            self.assertIn("VRAM", message)
                        else:
                            self.assertIn("GPU:", message)
                        self.assertIn("planning estimate", message)
                        self.assertIn("Available RAM: 30.0 GiB", message)

    def test_auto_uses_available_gpu(self):
        with patch("utils.resource_guidance._available_ram_gib", return_value=None), patch(
            "utils.resource_guidance._gpu_memory", return_value=None
        ):
            self.assertIn("selected device: cpu", training_resource_guidance("piper"))
        with patch("utils.resource_guidance._gpu_memory", return_value=(12.0, 10.0)):
            self.assertIn("selected device: cuda", training_resource_guidance("piper"))


if __name__ == "__main__":
    unittest.main()
