import unittest

import gradio as gr

from utils.e2a_export import E2A_FILES
from web_gui import update_e2a_export_visibility


class E2AExportVisibilityTests(unittest.TestCase):
    def test_supported_export_engines_are_visible(self):
        self.assertEqual(set(E2A_FILES), {"xtts_v1", "xtts_v2", "vits_tts", "mms_vits", "piper"})
        for model_key in E2A_FILES:
            with self.subTest(model_key=model_key):
                self.assertEqual(update_e2a_export_visibility(model_key), gr.update(visible=True))

    def test_engines_without_export_formats_are_hidden(self):
        for model_key in ("styletts2", "omnivoice", "f5_tts", "align_tts"):
            with self.subTest(model_key=model_key):
                self.assertEqual(update_e2a_export_visibility(model_key), gr.update(visible=False))


if __name__ == "__main__":
    unittest.main()
