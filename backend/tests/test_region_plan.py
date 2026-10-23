import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import region_plan as R
import regions


class PlanTests(unittest.TestCase):
    def test_model_lines(self):
        raw = ("SCENE: a rocky battlefield, stormy sky\nLEFT: a knight in silver armour\n"
               "RIGHT: a green dragon breathing fire")
        p = R.plan("ignored", raw)
        self.assertEqual((p["layout"], p["ratios"], p["source"]), ("cols", "1,1", "model"))
        self.assertEqual(p["prompts"], ["a knight in silver armour", "a green dragon breathing fire"])
        self.assertEqual(p["scene"], "a rocky battlefield, stormy sky")

    def test_messy_model_lines(self):
        p = R.plan("x", "Sure!\n1. **Right** - a blue truck\n2. **Left**: a red car\nScene: a city street")
        self.assertEqual(p["prompts"], ["a red car", "a blue truck"])  # left to right, whatever order they came in

    def test_rows_and_grid(self):
        p = R.plan("x", "TOP: a full moon\nBOTTOM: a howling wolf")
        self.assertEqual((p["layout"], p["prompts"]), ("rows", ["a full moon", "a howling wolf"]))
        p = R.plan("x", "TOP: lightning\nLEFT: a knight\nRIGHT: a dragon")
        self.assertEqual((p["layout"], p["ratios"]), ("grid", "1;1,1"))
        self.assertEqual(p["prompts"], ["lightning", "a knight", "a dragon"])

    def test_grid_columns_line_up(self):
        p = R.plan("x", "SCENE: a meadow\nTOP LEFT: a tree\nBOTTOM RIGHT: a house\nBOTTOM LEFT: a lake")
        self.assertEqual((p["layout"], p["ratios"]), ("grid", "1,1;1,1"))
        self.assertEqual(p["prompts"], ["a tree", "a meadow", "a lake", "a house"])

    def test_words_fallback(self):
        p = R.plan("a woman in a red coat on the left and a man with an umbrella in the middle and a small dog on "
                   "the right, rainy street", raw="I can't do that.")
        self.assertEqual(p["source"], "words")
        self.assertEqual(p["prompts"], ["a woman in a red coat", "a man with an umbrella", "a small dog"])
        self.assertEqual(p["scene"], "rainy street")

    def test_no_parts(self):
        with self.assertRaises(ValueError):
            R.plan("a cat sleeping", raw="")

    def test_at_most_four_areas(self):
        raw = "\n".join(f"{k}: thing {i}" for i, k in enumerate(["LEFT", "CENTER", "RIGHT", "TOP", "BOTTOM"]))
        self.assertLessEqual(len(R.plan("x", raw)["prompts"]), 4)

    def test_plans_work_with_the_regional_prompter(self):
        for raw in ("LEFT: a cat\nRIGHT: a dog", "TOP: the moon\nBOTTOM: a wolf", "TOP: clouds\nLEFT: a knight\nRIGHT: a dragon",
                    "TOP LEFT: a tree\nBOTTOM RIGHT: a house\nBOTTOM LEFT: a lake"):
            p = R.plan("x", raw)
            boxes = regions.parse_layout(p["layout"], p["ratios"], len(p["prompts"]))
            self.assertEqual(len(boxes), len(p["prompts"]))
            self.assertNotIn((0, 0, 1, 1), boxes)  # every area has its own place


if __name__ == "__main__":
    unittest.main()
