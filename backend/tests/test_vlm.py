"""Picture reader (Qwen3.5 through mlx-vlm): pinned model, answer clean-up, and the face model default."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vlm  # noqa: E402


class VlmTests(unittest.TestCase):
    def test_pinned(self):
        self.assertEqual(len(vlm.REVISION), 40)
        self.assertTrue(vlm.SPEC.startswith("mlx-vlm=="))

    def test_clean(self):
        self.assertEqual(vlm.clean('<think>\n\n</think>\n\n"A red fox\nin snow."'), "A red fox in snow.")
        self.assertEqual(vlm.clean("Prompt: A cat on a sofa."), "A cat on a sofa.")

    def test_prompts(self):
        self.assertIn("natural", vlm.STYLES)
        self.assertIn("short", vlm.STYLES)


class FaceDefaultTests(unittest.TestCase):
    def test_codeformer_default(self):
        try:
            from postprocess.face_restore import model_key
        except Exception as e:
            self.skipTest(str(e))
        self.assertEqual(model_key(None), "codeformer")
        self.assertEqual(model_key("restoreformer"), "restoreformer")
        self.assertEqual(model_key("CodeFormer"), "codeformer")


if __name__ == "__main__":
    unittest.main()
