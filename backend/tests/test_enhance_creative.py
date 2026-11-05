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
