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


def test_fused_qkv_split_into_q_k_v():
    # the MLX port keeps proj = {"q", "k", "v", "o", ...}; an adapter for the original's fused proj.qkv goes on all three
    class Layer:
        def __init__(self):
            self.proj = {"q": "Q", "k": "K", "v": "V", "o": "O"}

    class Model:
        layers = [Layer()]

    def resolve(model, name):
        parts = name.split(".")
        node = model
        for p in parts[:-1]:
            node = node[int(p)] if isinstance(node, list) else node[p] if isinstance(node, dict) else getattr(node, p)
        return node, parts[-1]

    def get(parent, key):
        return parent[int(key)] if isinstance(parent, list) else parent[key] if isinstance(parent, dict) else getattr(parent, key)

    m = Model()
    assert [t[2] for t in moss_lora.split_targets(m, "layers.0.proj.qkv", resolve, get)] == ["Q", "K", "V"]
    assert [t[2] for t in moss_lora.split_targets(m, "layers.0.proj.o", resolve, get)] == ["O"]
    assert moss_lora.split_targets(m, "layers.0.attn.to_out", resolve, get) == []


def test_fused_lokr_slices_add_up():
    # q, k and v each take their rows of the fused (W1 ⊗ W2) x: together they are exactly the fused output
    rng = np.random.default_rng(1)
    w1, w2 = rng.standard_normal((4, 4)), rng.standard_normal((6, 3))  # fused out 24 = 3 × 8, in 12
    x = rng.standard_normal((5, 12))
    full = x @ np.kron(w1, w2).T
    t = (w1 @ (x.reshape(5, 4, 3) @ w2.T)).reshape(5, 24)
    parts = [t[:, s:s + 8] for s in (0, 8, 16)]
    assert np.allclose(np.concatenate(parts, 1), full)


def test_minimax_h3_and_new_model_loras_found():
    h3 = {"__metadata__": {"modelspec.architecture": "minimaxh3image/lora"},
          "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": {}}
    assert loras.detect_family(h3) == "h3"
    keys = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": {}, "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": {},
            "diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_B.weight": {}}
    assert loras.detect_family(keys) == "h3"  # by its own layer names, no metadata
    zimage = {"diffusion_model.layers.0.attention.to_q.lora_A.weight": {}}
    assert loras.detect_family(zimage, "Me-zimage-epoch-10 (1)") == "zimage"  # the file name tells
    assert loras.detect_family(zimage, "oksana-ernie-epoch-10") == "ernie"
    assert loras.detect_family(zimage) == "ernie"


def test_every_catalog_model_type_takes_loras_or_is_known_not_to():
    """A new model added to the catalog gets a LoRA type (FAMILIES, the app's list, Fix a type) at the same time."""
    import re
    from pathlib import Path
    e = (Path(__file__).resolve().parents[1] / "engine.py").read_text()
    no_loras = {"anima"}
    assert set(re.findall(r'"family": "(\w+)"', e)) - set(loras.FAMILIES) <= no_loras
