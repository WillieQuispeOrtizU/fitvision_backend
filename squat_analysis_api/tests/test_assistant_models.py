from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from squat_analysis_api.assistant_models import default_gemma_model, main, safe_dir_name


class AssistantModelsTest(unittest.TestCase):
    def test_safe_dir_name(self):
        self.assertEqual(safe_dir_name("org/model:name"), "org__model__name")

    @patch("platform.machine", return_value="arm64")
    @patch("platform.system", return_value="Darwin")
    def test_default_gemma_model_uses_mlx_on_apple_silicon(self, *_):
        self.assertEqual(default_gemma_model(), "mlx-community/gemma-4-e4b-it-8bit")

    @patch("platform.machine", return_value="AMD64")
    @patch("platform.system", return_value="Windows")
    def test_default_gemma_model_uses_torch_elsewhere(self, *_):
        self.assertEqual(default_gemma_model(), "google/gemma-4-E4B-it")

    @patch("platform.machine", return_value="arm64")
    @patch("platform.system", return_value="Darwin")
    def test_download_command_prints_env_values(self, *_):
        with patch("huggingface_hub.snapshot_download") as snapshot_download:
            with tempfile.TemporaryDirectory() as tmp:
                exit_code = main(["--model-dir", tmp])

            self.assertEqual(exit_code, 0)
            self.assertEqual(snapshot_download.call_count, 2)
            local_dirs = {Path(call.kwargs["local_dir"]).name for call in snapshot_download.call_args_list}
        self.assertIn("tiiuae__Falcon-Perception-300M", local_dirs)
        self.assertIn("mlx-community__gemma-4-e4b-it-8bit", local_dirs)


if __name__ == "__main__":
    unittest.main()
