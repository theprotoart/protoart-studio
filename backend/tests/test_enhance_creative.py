"""✨ Enhance: answers that only echo the prompt are noticed, repeats are cleaned out, each press gets a direction."""
from postprocess import prompt_enhance as pe


def test_echo_is_repetitive():
    assert pe.repetitive("a fox in snow, a fox in the snow, fox in snow", "a fox in snow")
    assert not pe.repetitive("A red fox with a frost-tipped winter coat pauses on a fallen birch at dawn, breath "
                             "curling in the cold air.", "a fox in snow")


def test_repeats_removed():
    assert pe.dedupe_sentences("A fox runs. A fox runs. The snow is very very white.") == "A fox runs. The snow is very white."
    assert pe.dedupe_tags("red dress, dress red, smile, smile") == "red dress, smile"
    assert pe.clean("1girl, smile, smile, red dress", "a girl", "tags").count("smile") == 1


def test_bigger_model_question():
    q = pe.vlm_question("a cat", "natural", "square", pe.ANGLES[2])
    assert q.rstrip().endswith("Answer:") and "a cat" in q and pe.ANGLES[2] in q
    assert len(pe.ANGLES) >= 8


def test_zengineer_listed_and_not_ready(tmp_path):
    import paths
    import zengineer
    old, paths.HOME = paths.HOME, tmp_path
    try:
        st = {z["id"]: z for z in zengineer.status()}
    finally:
        paths.HOME = old
    assert set(st) == {"zeng_v4", "zeng_v6"} and not st["zeng_v6"]["ready"]
    assert "uncensored" in st["zeng_v4"]["note"]
    paths.HOME = tmp_path
    try:
        zengineer.write("zeng_v6", "a cat", "square")
        raise AssertionError("should need set-up")
    except zengineer.ZengError:
        pass
    finally:
        paths.HOME = old
