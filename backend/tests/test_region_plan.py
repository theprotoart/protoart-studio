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
                   "the right, rainy street", raw="LEFT: something else\nRIGHT: another thing")
        self.assertEqual(p["source"], "words")
        self.assertEqual(p["prompts"], ["a woman in a red coat", "a man with an umbrella", "a small dog"])
        self.assertEqual(p["scene"], "rainy street")

    def test_your_places_come_first(self):
        p = R.plan("a cat on the left and a dog on the right", raw="LEFT: a horse\nRIGHT: a cow")
        self.assertEqual((p["source"], p["prompts"]), ("words", ["a cat", "a dog"]))
        self.assertFalse(R.needs_model("a cat on the left and a dog on the right"))
        self.assertTrue(R.needs_model("two cats on a windowsill"))

    def test_one_place_with_with(self):
        p = R.plan("a cosy cabin in the snow with the northern lights above")
        self.assertEqual((p["layout"], p["prompts"]), ("rows", ["the northern lights", "a cosy cabin in the snow"]))
        p = R.plan("a lighthouse on a cliff at sunset with a sailing ship in the bottom right")
        self.assertEqual((p["layout"], p["ratios"]), ("grid", "1;1,1"))
        self.assertEqual(p["prompts"][0], "a lighthouse on a cliff at sunset")
        self.assertEqual(p["prompts"][2], "a sailing ship")

    def test_repetitive_model_plan_is_refused(self):
        raw = ("LEFT: black cat with a black collar sitting on the windowsill, looking at a vase\n"
               "RIGHT: black cat with a black collar sitting on the windowsill, looking at a book")
        with self.assertRaises(ValueError):
            R.plan("two cats on a windowsill", raw)

    def test_similar_but_different_subjects_are_kept(self):
        p = R.plan("two cats on a windowsill", "LEFT: a black cat sitting calmly on the left side of the windowsill\n"
                                               "RIGHT: a white cat sitting on the right side of the windowsill")
        self.assertEqual(len(p["prompts"]), 2)

    def test_no_parts(self):
        with self.assertRaises(ValueError):
            R.plan("a cat sleeping", raw="")

    def test_at_most_four_areas(self):
        subjects = ["a red fox", "a grey owl", "a brown bear", "the full moon", "a frozen lake"]
        raw = "\n".join(f"{k}: {t}" for k, t in zip(["LEFT", "CENTER", "RIGHT", "TOP", "BOTTOM"], subjects))
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
