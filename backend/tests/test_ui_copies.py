"""The 🔤 Replace text controls appear twice in the page (Tools and Image to image): the two copies must stay the
same, so a fix made in one is never missing from the other."""
import re
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "app" / "renderer" / "index.html"


def blocks(html):
    start = "<template v-if=\"txLayer() && txLayer().type === 'shape'\">"
    out = []
    i = html.find(start)
    while i >= 0:
        j = html.find("</details>", i)
        out.append(re.sub(r"\s+", " ", html[i:j]))
        i = html.find(start, j)
    return out


class ReplaceTextCopies(unittest.TestCase):
    def test_tools_and_image_to_image_have_the_same_controls(self):
        found = blocks(PAGE.read_text(encoding="utf-8"))
        self.assertEqual(len(found), 2, "expected the Replace text controls in Tools and in Image to image")
        self.assertEqual(found[0], found[1], "the two copies of the Replace text controls differ: make the same change in both")


if __name__ == "__main__":
    unittest.main()
