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
    assert head.startswith("A sunny park. a woman in a red coat <obj_1>; a dog <obj_3>.")
    items = json.loads("[" + tail)
    assert [i["id"] for i in items] == ["obj_1", "obj_3"]  # the empty area is skipped
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


def test_every_model_type_has_a_chip_and_shop_entry():
    """Each picture-model type in the catalog has a filter chip on Create and a type in the Models tab (FLUX 3 was missing)."""
    import re
    root = Path(__file__).resolve().parents[2]
    e = (root / "backend" / "engine.py").read_text()
    a = (root / "app" / "renderer" / "app.js").read_text()
    shop = a[a.index("SHOP_FAMILIES"):a.index("UPSCALER_GROUPS")]
    chips = a[a.index("    modelFilters() {"):a.index("    detailList()")] + re.search(r"const NEW_FAST = \[(.*?)\]", a).group(1)
    for f in set(re.findall(r'"family": "(\w+)"', e)):
        assert f'"{f}"' in shop, f
        assert f'"{f}"' in chips, f


def test_ideogram4_entry():
    import engine
    import mlx_worker
    m = engine.MODELS["ideogram4_mlx_q4"]
    assert m["family"] == "ideogram4" and m["gated"] and m["min_ram_gb"] >= 24 and m["mlx_quantize"] == 4
    assert mlx_worker._family(m["mlx_base"]) == "ideogram4"
    assert mlx_worker.IDEOGRAM_PRESETS[m["defaults"]["steps"]] == "V4_DEFAULT_20"


def test_drawn_boxes_for_flux3_ideogram4_and_moss():
    lay = {"layout": "free", "prompts": ["a red kite", "a sign that reads 'SALE'"],
           "boxes": [[0.1, 0.2, 0.5, 0.6], [0.6, 0.7, 0.9, 1.0]]}
    f3 = regions.layout_prompt("flux3", "A beach", lay)
    items = json.loads(f3[f3.index("["):])
    assert items[0] == {"id": "obj_1", "bbox": [200, 100, 600, 500], "desc": "a red kite"}
    ideo = json.loads(regions.layout_prompt("ideogram4", "A beach poster", lay))
    els = ideo["compositional_deconstruction"]["elements"]
    assert ideo["high_level_description"] == "A beach poster"
    assert els[0]["type"] == "obj" and els[1] == {"type": "text", "bbox": [700, 600, 1000, 900], "text": "SALE",
                                                   "desc": "a sign that reads 'SALE'"}
    moss = json.loads(regions.layout_prompt("moss", "A beach", lay, 1024, 768))
    assert moss["bbox_format"] == "yxyx" and moss["items"][0]["bbox"] == [-231, -410, 77, 0]
    assert moss["items"][1]["label"] == "text" and moss["items"][1]["text"] == "SALE"
    assert regions.layout_prompt("moss", "Plain", {"layout": "free", "prompts": [""], "boxes": []}) == "Plain"
