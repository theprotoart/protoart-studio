"""1.6.0 fixes: prompt conversion, negative embeddings by name, the app's own Union ProMax folder isn't listed as yours."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_convert_styles_pass_through():
    from postprocess.prompt_enhance import style_for, SYSTEM, EXAMPLES
    for s in ("to_tags", "to_natural"):
        assert style_for(s) == s and s in SYSTEM and s in EXAMPLES


def test_to_tags_adds_nothing():
    from postprocess.prompt_enhance import clean
    out = clean("Tags: 1boy, beard, red cape, beard, city. night", "a bearded man", "to_tags")
    assert out == "1boy, beard, red cape, city, night"
    assert "masterpiece" not in out


def test_to_natural_is_plain_text():
    from postprocess.prompt_enhance import clean
    s = "A bearded man in a red cape stands in a city at night."
    assert clean(s, "1boy, beard", "to_natural") == s


def test_negative_embeddings_by_name():
    from embeddings import is_negative_name
    for n in ("EasyNegativeV2", "ng_deepnegative_v1_75t", "badhandv4", "bad-hands-5", "verybadimagenegative_v1.3",
              "unaestheticXL", "negative_hand-neg", "SimpleNegativeV3"):
        assert is_negative_name(n), n
    for n in ("Evang_2027STUDIO-step750", "epiCRealism", "CharTurnerV2", "my_character"):
        assert not is_negative_name(n), n


def test_own_union_folder_is_not_a_user_controlnet():
    import controlnet
    import paths
    d = Path(tempfile.mkdtemp())
    (d / ".union-promax").mkdir()
    (d / ".union-promax" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 16)
    (d / "mine.safetensors").write_bytes(b"x" * 16)
    old = paths.CONTROLNET_DIR
    try:
        paths.CONTROLNET_DIR = d
        assert [f["name"] for f in controlnet.user_files()] == ["mine"]
    finally:
        paths.CONTROLNET_DIR = old


def test_keep_faces_redoes_each_face_close_up(tmp_path=None):
    import engine
    from PIL import Image
    tmp = Path(tempfile.mkdtemp())
    eng = engine.Engine.__new__(engine.Engine)
    eng.emit = lambda m: None
    import threading
    eng.cancel_event = threading.Event()

    class FF:
        def find(self, im, min_score=0.6, min_size=12):
            return [(40, 30, 80, 80)]
    eng._face_finder_ready = lambda: FF()
    calls = []

    class Mlx:
        def request(self, msg, on_event=None):
            calls.append(msg)
            im = Image.new("RGB", (msg["width"], msg["height"]), (200, 30, 30))
            for x in range(0, msg["width"], 8):  # a pattern (the face's detail)
                for y in range(msg["height"]):
                    im.putpixel((x, y), (255, 255, 255))
            im.save(msg["out"])
    eng.mlx = Mlx()
    original = Image.new("RGB", (160, 200), (90, 90, 90))
    result = Image.new("RGB", (320, 400), (20, 120, 20))
    files = []
    out, n = eng._klein_faces(original, result, "colour this photo", 1, 4, tmp, None, files)
    assert n == 1 and len(calls) == 1 and out.size == result.size
    assert len(calls[0]["image_paths"]) == 2 and "same person" in calls[0]["prompt"]
    import numpy as np
    face = np.asarray(out.crop((45, 35, 75, 75)))
    assert (face != np.array([20, 120, 20])).any()      # the face area came from the close-up
    assert out.getpixel((5, 390)) == (20, 120, 20)      # elsewhere: the edit's result
    eng._face_finder_ready = lambda: None                # no face finder yet: nothing changes
    assert eng._klein_faces(original, result, "x", 1, 4, tmp, None, [])[1] == 0


def test_panel_seams_found_and_largest_panel_kept():
    import numpy as np
    from PIL import Image
    from postprocess import compose as c
    rng = np.random.default_rng(0)
    a = (rng.random((400, 600)) * 120 + 60).astype(np.uint8)
    assert c.panel_seams(Image.fromarray(a)) == []
    b = a.copy()
    b[:, 298:303] = 250          # a white gutter down the middle
    b[180:184, 303:] = 250       # and across the right half (a 3-panel page)
    img = Image.fromarray(b)
    assert c.panel_seams(img)
    l, t, r, bt = c.largest_panel(img)
    assert r <= 300 and (r - l) * (bt - t) > 0.4 * 600 * 400  # the left panel, the biggest
    w = np.full((400, 600), 255, np.uint8)
    w[100:300, 200:400] = rng.integers(0, 255, (200, 200))     # a picture on a plain background: no panels
    assert c.panel_seams(Image.fromarray(w)) == []
