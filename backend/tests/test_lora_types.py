"""LoRA model types: the app's type buttons match the backend's list, MOSS LoRAs are recognised, and MOSS's LoKr maths."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loras  # noqa: E402
import moss_lora  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app" / "renderer" / "app.js"


def test_app_type_list_matches_backend():
    js = APP.read_text()
    m = re.search(r"const LORA_FAMILIES = \[([^\]]*)\]", js)
    assert m, "LORA_FAMILIES missing from app.js"
    app = tuple(re.findall(r'"([a-z0-9]+)"', m.group(1)))
    assert app == loras.FAMILIES, f"app.js LORA_FAMILIES {app} != loras.FAMILIES {loras.FAMILIES}"
    labels = re.search(r"const FAMILY_LABEL = \{(.*?)\};", js, re.S).group(1)
    for f in loras.FAMILIES:  # every type needs a name on its button
        assert re.search(rf"\b{f}:", labels), f"{f} has no name in FAMILY_LABEL"


def test_moss_detected():
    lokr = {"diffusion_model.layers.0.proj.ffn_down.lokr_w1": {}, "diffusion_model.layers.0.proj.k.alpha": {}}
    assert loras.detect_family(lokr) == "moss"
    kohya = {"diffusion_model.layers.3.proj.q.lora_down.weight": {}, "diffusion_model.layers.3.proj.q.lora_up.weight": {}}
    assert loras.detect_family(kohya) == "moss"
    assert loras.detect_family({"__metadata__": {"ss_base_model_version": "moss_image"}}) == "moss"
    # others unchanged
    assert loras.detect_family({"transformer.layers.0.self_attn.q_proj.lora_A.weight": {}}) == "ernie"
    assert loras.detect_family({"lora_unet_down_blocks_0_attentions_0.lora_down.weight": {}}) == "sd15"


def test_key_grouping_and_kinds():
    g = moss_lora.group_keys(["diffusion_model.layers.0.proj.k.lokr_w1", "diffusion_model.layers.0.proj.k.lokr_w2",
                              "diffusion_model.layers.0.proj.k.alpha", "transformer.layers.1.proj.q.lora_A.weight",
                              "transformer.layers.1.proj.q.lora_B.weight", "layers.2.proj.o.lora_down.weight",
                              "layers.2.proj.o.lora_up.weight", "something_else"])
    assert set(g) == {"layers.0.proj.k", "layers.1.proj.q", "layers.2.proj.o"}
    assert moss_lora.kind_of(g["layers.0.proj.k"]) == "lokr"
    assert moss_lora.kind_of(g["layers.1.proj.q"]) == "peft"
    assert moss_lora.kind_of(g["layers.2.proj.o"]) == "lora"
    assert moss_lora.kind_of({"hada_w1_a": 1}) == "loha"
    assert moss_lora.lokr_scale({}, 0, 0, 4.0) == 1.0  # both factors whole: LyCORIS ignores alpha
    assert moss_lora.lokr_scale({}, 0, 8, 4.0) == 0.5


def test_lokr_without_building_the_big_matrix():
    # (W1 ⊗ W2) x == W1 · X · W2ᵀ with x reshaped to (b1, b2): what LoKrLinear computes
    rng = np.random.default_rng(0)
    w1, w2 = rng.standard_normal((4, 3)), rng.standard_normal((5, 6))
    x = rng.standard_normal((2, 3 * 6))
    want = x @ np.kron(w1, w2).T
    got = (w1 @ (x.reshape(2, 3, 6) @ w2.T)).reshape(2, 4 * 5)
    assert np.allclose(got, want)
