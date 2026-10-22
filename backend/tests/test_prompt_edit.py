import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prompt_edit as pe


class PromptEditTests(unittest.TestCase):
    def test_plain_prompt(self):
        self.assertEqual(pe.schedule("a cat, [soft] light", 20), [(0.0, "a cat, [soft] light")])
        self.assertFalse(pe.has_edits("a (cat:1.2), [soft]"))

    def test_swap_at_fraction_and_step(self):
        self.assertEqual(pe.schedule("a [cat:dog:0.5] on a sofa", 20), [(0.0, "a cat on a sofa"), (0.5, "a dog on a sofa")])
        self.assertEqual(pe.schedule("a [cat:dog:5] sitting", 20), [(0.0, "a cat sitting"), (0.25, "a dog sitting")])

    def test_add_and_remove(self):
        self.assertEqual(pe.schedule("portrait, [hat:0.3], smile", 10), [(0.0, "portrait, smile"), (0.3, "portrait, hat, smile")])
        self.assertEqual(pe.schedule("portrait, [hat::0.3], smile", 10), [(0.0, "portrait, hat, smile"), (0.3, "portrait, smile")])

    def test_several_points(self):
        s = pe.schedule("[red:blue:0.25] car, [rain:0.6]", 20)
        self.assertEqual([f for f, _ in s], [0.0, 0.25, 0.6])
        self.assertEqual(s[-1][1], "blue car, rain")


if __name__ == "__main__":
    unittest.main()
