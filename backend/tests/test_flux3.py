"""FLUX 3 Image Klein: catalog entry, LoRA type, and 🗺 Regions turned into its layout boxes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loras  # noqa: E402
import regions  # noqa: E402


def test_catalog_entry():
    import engine
    m = engine.all_models()["flux3_klein_mlx_q4"]
    assert m["family"] == "flux3" and m["engine"] == "mlx" and m["min_ram_gb"] == 8
    assert m["mlx_base"] == "flux3-image-klein" and len(m["mlx_revision"]) == 40
    assert m["defaults"]["steps"] == 16 and "flux3" in loras.FAMILIES


def test_lora_type():
    assert loras.detect_family({"transformer.mode_blocks.image.0.attn.to_q.lora_A.weight": {}}) == "flux3"
    assert loras.detect_family({"diffusion_model.content_mode_blocks.image.3.qkv.lora_down.weight": {}}) == "flux3"
    assert loras.detect_family({"__metadata__": {"ss_base_model_version": "flux3_image_klein"}}) == "flux3"
    assert loras.detect_family({"transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": {}}) != "flux3"


def test_regions_become_layout_boxes():
    p = regions.flux3_boxes("A sunny park", "cols", "2,1", ["a woman in a red coat", "", "a dog"])
    head, _, tail = p.partition(" [")
    assert head.startswith("A sunny park. a woman in a red coat <region_1>; a dog <region_3>.")
    items = json.loads("[" + tail)
    assert [i["id"] for i in items] == ["region_1", "region_3"]  # the empty area is skipped
    assert items[0]["bbox"] == [0, 0, 1000, 500]  # [top, left, bottom, right] on 0-1000
    assert items[1]["bbox"] == [0, 750, 1000, 1000]
    assert regions.flux3_boxes("Just this", "cols", "", ["", ""]) == "Just this"


def test_recommended_hf_loras_have_a_type():
    import bee_backend
    assert all(f in loras.FAMILIES for f in bee_backend.RECIPE_HF_LORAS.values())
    assert bee_backend.RECIPE_HF_LORAS[("F16/z-image-turbo-flow-dpo", "zit_fdpo_v1.safetensors")] == "zimage"


def test_styles_file():
    js = (Path(__file__).resolve().parents[2] / "app" / "renderer" / "zimage_styles.js").read_text()
    data = json.loads(js[js.index("["):js.rindex("]") + 1])
    assert len(data) >= 100 and {d["cat"] for d in data} == {"Photo", "Illustration", "Wild"}
    assert all("{$" not in d["pre"] + d["post"] for d in data)  # no template commands left
    assert "MIT" in js[:600] and "Martin Rizzo" in js[:600]
