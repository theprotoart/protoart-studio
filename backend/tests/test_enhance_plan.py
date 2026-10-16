"""Enhance & upscale speed planning (no torch needed): work size, real output size, tile planner."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postprocess import finegrain  # noqa: E402


class PlanTests(unittest.TestCase):
    def test_work_size(self):
        self.assertEqual(finegrain.work_side({"turbo": True}, ram_gb=8), 512)
        self.assertEqual(finegrain.work_side({"num_inference_steps": 10}, ram_gb=8), 512)
        self.assertEqual(finegrain.work_side({"num_inference_steps": 18}, ram_gb=8), 768)
        self.assertEqual(finegrain.work_side({"turbo": True}, ram_gb=32), 768)
        self.assertEqual(finegrain.work_side({"turbo": True, "work_size": 768}, ram_gb=8), 768)

    def test_output_size(self):
        self.assertEqual(finegrain.output_size(640, 800, 2), (1536, 1920))
        self.assertEqual(finegrain.output_size(640, 800, 2, 512), (1024, 1280))

    def test_planner(self):
        # a 768 px square fits one tile on an 8 GB Mac
        self.assertEqual(finegrain.choose_tile(96, 96, 9216), (96, 96))
        th, tw = finegrain.choose_tile(160, 128, 9216)
        self.assertLessEqual(th * tw, 9216)
        self.assertTrue(th >= 64 and tw >= 64 and th % 8 == 0 and tw % 8 == 0)
        n = finegrain._count_tiles(160, 128, th, tw)
        self.assertLess(n, finegrain._count_tiles(160, 128, 64, 64))  # fewer than fixed 512 px tiles

    def test_profiles(self):
        self.assertEqual(finegrain.memory_profile(8)["esrgan_tile"], 192)
        self.assertIsNone(finegrain.memory_profile(64)["slice"])


if __name__ == "__main__":
    unittest.main()
