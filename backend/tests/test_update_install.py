"""↻ Install and restart: the new app is copied in while the old one is still open (so reopening it too soon can't
leave the old version), and the script that runs after quitting only swaps the folders and opens the new app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_copy_before_quit_swap_after():
    main = (ROOT / "app" / "main.js").read_text()
    body = main[main.index('handle("update:install"'):main.index('handle("update:openPage"')]
    prep, swap = body.index("const prep = ["), body.index("const script = [")
    assert prep < swap and "ditto" in body[prep:swap] and "ditto" not in body[swap:]
    assert body.index("await new Promise((done)") < body.index("app.quit()")      # the copy finishes before quitting
    assert "kill -0" in body[swap:] and "mv " in body[swap:] and "open " in body[swap:]
    assert "await shell.openPath(dmg)" in body[prep:swap + 2000]                    # a failed copy opens the installer


def test_installing_shown_and_ram_text():
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    assert "upd.busy === 'install'" in html and "Installing…" in html
    assert "needRamText(m)" in html and "GB+ is faster" in app


def test_sweet_treat_faces_and_font():
    app = (ROOT / "app" / "renderer" / "app.js").read_text()
    html = (ROOT / "app" / "renderer" / "index.html").read_text()
    blk = app[app.index('id: "tf_sweet"'):][:3000]
    assert 'kind: "choice"' in blk and '"cupcake"' in blk and '"own"' in blk and "exact form" in blk
    assert 'x[0] === "own")' in app                                   # typing your own treat picks My own
    assert 'id: "faces", icon: "🙂"' in app and "selFaceAt(p, sub)" in app and 'call("faceswap_faces"' in app
    word = html[html.index("<template v-if=\"shp.how === 'word'\">"):][:2500]
    assert "shape-font-own" in word                                    # own font right under the font buttons
