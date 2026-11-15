"""Relight's ⚡ Fast / Standard / ★ Best: Fast skips the slow 1.5× detail pass."""
import unittest

from postprocess import relight as rl


class RelightSpeed(unittest.TestCase):
    def test_plans(self):
        self.assertEqual(rl.speed_plan("fast", True), (12, False))   # never the detail pass
        self.assertEqual(rl.speed_plan("standard", True), (20, True))
        self.assertEqual(rl.speed_plan("standard", False), (20, False))
        self.assertEqual(rl.speed_plan("best", True), (30, True))

    def test_old_settings_and_steps(self):
        self.assertEqual(rl.speed_plan(None, True), (20, True))      # saved before Speed existed
        self.assertEqual(rl.speed_plan("weird", True), (20, True))
        self.assertEqual(rl.speed_plan("fast", True, steps=200), (50, False))
        self.assertEqual(rl.speed_plan("best", True, steps=3), (8, True))


if __name__ == "__main__":
    unittest.main()
