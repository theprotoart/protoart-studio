"""📰 MOSS-Image page layouts: the renderer's builder makes layouts the engine's check accepts, with every word kept."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moss_layout_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def build(page, describe, w, h):
    js = (f"const m=require({json.dumps(str(ROOT / 'app/renderer/moss_page.js'))});"
          f"console.log(JSON.stringify(m.buildLayout({json.dumps(page)},{json.dumps(describe)},{w},{h}).layout))")
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def test_pages():
    if not shutil.which("node"):
        return
    panels = [{"scene": "Santa walks into a gym", "bubble": "Ho ho ho! Leg day!"}, {"caption": "Later that night..."},
              {}, {"bubble": "Who needs reindeer?", "bubble_side": "right", "bubble_kind": "thought"},
              {"bubble": "five"}, {"bubble": "six"}]
    for tpl in ("3", "4", "6", "splash", "poster"):
        for w, h in ((768, 1024), (832, 1216)):
            for title in ("", "THE GYM SANTA"):
                lay = build({"template": tpl, "title": title, "subtitle": "Open 7 am", "panels": panels}, "x", w, h)
                assert moss_layout_check.check(lay, w, h) == [], (tpl, w, h, title)
                words = [i.get("text") for i in lay["items"] if i["label"] == "text"]
                if title:
                    assert "THE GYM SANTA" in words
                if tpl == "4":
                    assert "Ho ho ho! Leg day!" in words and "Who needs reindeer?" in words
    assert moss_layout_check.check({"summary": {"caption": "x"}, "items": [{"label": "text", "text": "HI",
                                                                            "bbox": [-600, 0, 0, 10]}]}, 768, 1024)


if __name__ == "__main__":
    test_pages()
    print("ok")
