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
