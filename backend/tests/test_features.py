"""Tests for prompt weights, wildcards, the comparison grid and the inpaint/outpaint helpers (no torch needed)."""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PROTOART_HOME", tempfile.mkdtemp(prefix="protoart_test_"))

import prompt_weights as pw  # noqa: E402
import wildcards  # noqa: E402
import grid  # noqa: E402


class FakeTokenizer:
    """Word-level stand-in for CLIP's tokenizer: one token per word or comma."""
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 2

    def __init__(self):
        self.vocab = {",</w>": 3}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 10)

    def __call__(self, text, add_special_tokens=False, truncation=False):
        import re
        toks = re.findall(r"[^\s,]+|,", text)
        ids = [self._id(",</w>" if t == "," else t) for t in toks]

        class R:
            input_ids = ids
        return R()

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, 0)


class PromptWeightTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(pw.parse("a cat"), [["a cat", 1.0]])

    def test_weights(self):
        r = dict((t, round(w, 3)) for t, w in pw.parse("a (cat:1.3) on [grass] ((sky))"))
        self.assertEqual(r["cat"], 1.3)
        self.assertEqual(r["grass"], round(1 / 1.1, 3))
        self.assertEqual(r["sky"], round(1.1 * 1.1, 3))

    def test_escapes_and_unclosed(self):
        self.assertEqual(pw.parse(r"a \(literal\) b")[0][0], "a (literal) b")
        r = pw.parse("(unclosed")
        self.assertAlmostEqual(r[0][1], 1.1)

    def test_break(self):
        r = pw.parse("one BREAK two")
        self.assertIn(["BREAK", -1], r)

    def test_needs_custom(self):
        tok = FakeTokenizer()
        self.assertFalse(pw.needs_custom("a cat", "", tok))
        self.assertTrue(pw.needs_custom("a (cat:1.2)", "", tok))
        self.assertTrue(pw.needs_custom(" ".join(f"w{i}" for i in range(80)), "", tok))

    def test_chunks(self):
        tok = FakeTokenizer()
        long = ", ".join(f"tag{i}" for i in range(60))  # 60 words + 59 commas = 119 tokens
        chunks = pw.tokenize_chunks(None, tok, long)
        self.assertEqual(len(chunks), 2)
        for ids, ws in chunks:
            self.assertEqual(len(ids), 77)
            self.assertEqual(len(ws), 77)
            self.assertEqual(ids[0], tok.bos_token_id)
        # nothing lost: every word appears once across the chunks
        words = [i for ids, _ in chunks for i in ids if i >= 10]
        self.assertEqual(len(words), 60)

    def test_chunk_weights(self):
        tok = FakeTokenizer()
        (ids, ws), = pw.tokenize_chunks(None, tok, "a (cat:1.5)")
        self.assertEqual(ws[1], 1.0)
        self.assertEqual(ws[2], 1.5)

    def test_break_makes_new_chunk(self):
        tok = FakeTokenizer()
        self.assertEqual(len(pw.tokenize_chunks(None, tok, "a BREAK b")), 2)


class WildcardTests(unittest.TestCase):
    def test_braces(self):
        rng = random.Random(1)
        out = {wildcards.resolve("{red|blue|green} dress", rng) for _ in range(50)}
        self.assertEqual(out, {"red dress", "blue dress", "green dress"})

    def test_nested_and_multi(self):
        rng = random.Random(2)
        for _ in range(20):
            r = wildcards.resolve("{a {big|small} cat|a dog}", rng)
            self.assertIn(r, {"a big cat", "a small cat", "a dog"})
        r = wildcards.resolve("{2$$x|y|z}", random.Random(3))
        self.assertEqual(len(r.split(", ")), 2)

    def test_same_seed_same_prompt(self):
        self.assertEqual(wildcards.for_image("{a|b|c|d|e}", 42), wildcards.for_image("{a|b|c|d|e}", 42))

    def test_files(self):
        tmp = Path(tempfile.mkdtemp(prefix="wild_"))
        orig = wildcards.folder
        wildcards.folder = lambda: tmp  # never touch the real wildcards folder
        self.addCleanup(lambda: setattr(wildcards, "folder", orig))
        d = wildcards.folder()
        (d / "sub").mkdir(parents=True, exist_ok=True)
        (d / "hair.txt").write_text("# comment\nlong hair\nshort hair\n")
        (d / "sub" / "col.txt").write_text("red\n")
        self.assertIn(wildcards.resolve("__hair__", random.Random(0)), {"long hair", "short hair"})
        self.assertEqual(wildcards.resolve("__sub/col__ hat", random.Random(0)), "red hat")
        self.assertEqual(wildcards.resolve("__missing__", random.Random(0)), "__missing__")
        self.assertEqual(wildcards.resolve("__../../etc/passwd__", random.Random(0)), "__../../etc/passwd__")
        names = {f["name"]: f["count"] for f in wildcards.list_files()}
        self.assertEqual(names["hair"], 2)

    def test_plain_untouched(self):
        self.assertEqual(wildcards.resolve("a (cat:1.2)", random.Random()), "a (cat:1.2)")


class GridTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(grid.parse_values("cfg", "5, 7.5, 9"), [5.0, 7.5, 9.0])
        self.assertEqual(grid.parse_values("steps", "10-20 (+5)"), [10, 15, 20])
        self.assertEqual(grid.parse_values("sampler", "euler, dpmpp_2m"), ["euler", "dpmpp_2m"])

    def test_apply(self):
        base = {"cfg": 7, "loras": [{"name": "A", "weight": 1.0}]}
        self.assertEqual(grid.apply(base, "cfg", 5.0)["cfg"], 5.0)
        self.assertEqual(grid.apply(base, "lora_weight", 0.5, "A")["loras"][0]["weight"], 0.5)
        self.assertEqual(base["loras"][0]["weight"], 1.0)  # base unchanged
        swapped = grid.apply(base, "lora", "B", "A")["loras"]
        self.assertEqual([l["name"] for l in swapped], ["B"])
        self.assertEqual(grid.apply(base, "lora", "none", "A")["loras"], [])

    def test_compose(self):
        from PIL import Image
        ims = {(x, y): Image.new("RGB", (64, 64), (x * 80, y * 80, 0)) for x in range(3) for y in range(2)}
        ims[(1, 1)] = None
        out = grid.compose(ims, [5, 7, 9], ["a", "b"], "cfg", "sampler", title="test")
        self.assertGreater(out.width, 64 * 3)
        self.assertGreater(out.height, 64 * 2)


class MaskTests(unittest.TestCase):
    def test_outpaint_canvas(self):
        from PIL import Image
        import detailer
        img = Image.new("RGB", (100, 80), (200, 10, 10))
        canvas, mask = detailer.outpaint_canvas(img, left=40, bottom=20)
        self.assertEqual(canvas.size, (140, 100))
        self.assertEqual(mask.size, (140, 100))
        self.assertGreater(mask.getpixel((5, 50)), 200)     # new area: fill
        self.assertLess(mask.getpixel((90, 40)), 30)        # original: keep
        self.assertEqual(canvas.getpixel((90, 40)), (200, 10, 10))

    def test_decode_mask(self):
        import base64
        import io
        from PIL import Image
        import detailer
        m = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
        for x in range(10, 20):
            for y in range(10, 20):
                m.putpixel((x, y), (255, 255, 255, 255))
        buf = io.BytesIO()
        m.save(buf, "PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        out = detailer.decode_mask(url, (100, 100))
        self.assertEqual(out.size, (100, 100))
        self.assertEqual(out.getpixel((30, 30)), 255)
        self.assertEqual(out.getpixel((80, 80)), 0)

    def test_face_crop_paste(self):
        from PIL import Image
        import detailer
        img = Image.new("RGB", (300, 200), (50, 50, 50))
        crop, region, mask = detailer.face_crop(img, (120, 60, 170, 120), work=128)
        self.assertEqual(crop.size, (128, 128))
        self.assertEqual(region[2] - region[0], region[3] - region[1])
        out = detailer.paste_back(img, Image.new("RGB", (128, 128), (255, 255, 255)), region, mask)
        self.assertEqual(out.size, img.size)
        cx, cy = 145, 90
        self.assertGreater(out.getpixel((cx, cy))[0], 200)   # face centre replaced
        self.assertEqual(out.getpixel((5, 5)), (50, 50, 50))  # far away untouched


class EditsTests(unittest.TestCase):
    def test_round_trip(self):
        import time as _t
        from PIL import Image
        import edits
        tmp = Path(tempfile.mkdtemp(prefix="edits_"))
        orig = edits.folder
        edits.folder = lambda: tmp
        self.addCleanup(lambda: setattr(edits, "folder", orig))
        src = tmp.parent / f"src_{os.getpid()}.png"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(src)
        work = Path(edits.start(src, 7, {"prompt": "x"}, "PhotoCraft"))
        self.assertTrue(work.exists())
        got = []
        imp = lambda f, info: got.append((Path(f).name, info.get("source_id"))) or {"id": len(got)}
        old_time = _t.time() - 10
        os.utime(work, (old_time, old_time))  # unchanged copy: pretend it's old, not new
        st = edits._load()
        st[work.name]["mtime"] = old_time
        edits._save(st)
        self.assertEqual(edits.check(imp), [])          # not saved yet: nothing imported
        Image.new("RGB", (32, 32), (200, 0, 0)).save(work)  # "save" in the editor
        os.utime(work, (_t.time() - 5, _t.time() - 5))
        self.assertEqual(len(edits.check(imp)), 1)
        self.assertEqual(got[0][1], 7)
        extra = tmp / (work.stem + " copy.png")          # "Save As" next to it
        Image.new("RGB", (32, 32), (0, 200, 0)).save(extra)
        os.utime(extra, (_t.time() - 5, _t.time() - 5))
        self.assertEqual(len(edits.check(imp)), 1)
        self.assertEqual(got[1], (extra.name, 7))
        self.assertEqual(edits.check(imp), [])          # nothing new
        edits.finish()
        self.assertEqual(edits.pending(), [])


if __name__ == "__main__":
    unittest.main()
