"""Tests for the library, presets, LoRA detection and sampler catalogue.

Run with:  python -m unittest discover -s backend/tests
No GPU, torch or model downloads needed.
"""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library import Library  # noqa: E402
import loras  # noqa: E402
import samplers  # noqa: E402


def write_fake_safetensors(path, keys, metadata=None):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\x00\x00")


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = Library(Path(self.tmp.name) / "lib.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, prompt, loras_=(), model="sdxl", seed=1):
        return self.lib.add_image(f"/x/{prompt}.png", {
            "prompt": prompt, "negative_prompt": "blurry", "model": model, "sampler": "euler",
            "steps": 20, "cfg": 7, "seed": seed, "width": 512, "height": 512,
            "loras": [{"name": n, "weight": w} for n, w in loras_],
        })

    def test_star_and_filter(self):
        a = self.add("a cat")
        self.add("a dog")
        self.assertFalse(a["starred"])
        self.lib.set_starred(a["id"], True)
        starred = self.lib.query(starred_only=True)
        self.assertEqual([r["prompt"] for r in starred], ["a cat"])
        self.lib.set_starred(a["id"], False)
        self.assertEqual(self.lib.query(starred_only=True), [])

    def test_search_by_lora_requires_all(self):
        self.add("one", [("pixel", 0.8)])
        self.add("two", [("pixel", 1.0), ("watercolor", 0.5)])
        self.add("three", [("watercolor", 0.7)])
        self.assertEqual({r["prompt"] for r in self.lib.query(loras=["pixel"])}, {"one", "two"})
        self.assertEqual([r["prompt"] for r in self.lib.query(loras=["pixel", "watercolor"])], ["two"])
        used = {u["name"]: u["count"] for u in self.lib.used_loras()}
        self.assertEqual(used, {"pixel": 2, "watercolor": 2})
        rec = self.lib.query(loras=["watercolor"], text="thr")[0]
        self.assertEqual(rec["loras"], [{"name": "watercolor", "weight": 0.7}])

    def test_search_by_seed(self):
        self.add("a cat", seed=3844)
        self.add("a dog", seed=12)
        self.add("seed 3844 in the prompt", seed=7)
        self.assertEqual([r["prompt"] for r in self.lib.query(seed=3844)], ["a cat"])
        self.assertEqual([r["prompt"] for r in self.lib.query(seed="12")], ["a dog"])
        self.assertEqual(self.lib.query(seed="abc"), [])
        self.assertEqual(len(self.lib.query(seed="")), 3)  # empty = no filter
        # a number typed in the search box finds the seed and prompts containing it
        self.assertEqual({r["prompt"] for r in self.lib.query(text="3844")}, {"a cat", "seed 3844 in the prompt"})

    def test_text_search_escapes_wildcards(self):
        self.add("100% real")
        self.add("1000 cats")
        self.assertEqual([r["prompt"] for r in self.lib.query(text="0%")], ["100% real"])

    def test_postprocessed_images_link_to_parent(self):
        src = self.add("portrait", [("pixel", 0.8)])
        child = self.lib.add_image("/x/p_faces.png", dict(src["settings"], face_restore={"strength": 0.8}),
                                   parent_id=src["id"], kind="faces")
        self.assertEqual(child["parent_id"], src["id"])
        self.assertEqual(child["kind"], "faces")
        self.assertEqual([r["id"] for r in self.lib.query(kind="faces")], [child["id"]])
        self.assertEqual(len(self.lib.query(loras=["pixel"])), 2)  # LoRA search finds both versions

    def test_migrates_old_database(self):
        import sqlite3
        db = Path(self.tmp.name) / "old.sqlite"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, created_at REAL NOT NULL,"
                    " prompt TEXT NOT NULL DEFAULT '', negative_prompt TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',"
                    " sampler TEXT NOT NULL DEFAULT '', steps INTEGER, cfg REAL, seed INTEGER, width INTEGER, height INTEGER,"
                    " settings_json TEXT NOT NULL DEFAULT '{}', starred INTEGER NOT NULL DEFAULT 0)")
        con.execute("INSERT INTO images (path, created_at) VALUES ('/old.png', 1)")
        con.commit(); con.close()
        lib = Library(db)
        self.assertEqual(lib.query()[0]["kind"], "generated")

    def _add_file(self, name, parent_id=None, kind="generated"):
        p = Path(self.tmp.name) / "imgs" / "2026-10-12" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"png")
        return self.lib.add_image(p, {"prompt": name, "loras": []}, parent_id=parent_id, kind=kind), p

    def test_delete_removes_files_permanently(self):
        a, pa = self._add_file("a.png")
        b, pb = self._add_file("b.png")
        dry = self.lib.delete_images([a["id"]], dry_run=True)
        self.assertTrue(pa.exists())
        self.assertEqual(dry["paths"], [str(pa)])
        res = self.lib.delete_images([a["id"]])
        self.assertFalse(pa.exists())
        self.assertTrue(pb.exists())
        self.assertEqual(res["ids"], [a["id"]])
        self.assertIsNone(self.lib.get_image(a["id"]))

    def test_delete_with_and_without_versions(self):
        orig, po = self._add_file("o.png")
        face, pf = self._add_file("o_faces.png", parent_id=orig["id"], kind="faces")
        enh, pe = self._add_file("o_faces_enh.png", parent_id=face["id"], kind="enhanced")
        self.assertEqual(self.lib.delete_images([orig["id"]], dry_run=True)["versions"], 1)
        self.assertEqual(self.lib.delete_images([orig["id"]], include_versions=True, dry_run=True)["versions"], 2)
        self.lib.delete_images([orig["id"]])  # keep versions
        self.assertFalse(po.exists())
        self.assertTrue(pf.exists())
        self.assertIsNone(self.lib.get_image(face["id"])["parent_id"])  # now standalone
        self.lib.delete_images([face["id"]], include_versions=True)
        self.assertFalse(pf.exists())
        self.assertFalse(pe.exists())
        self.assertEqual(self.lib.query(), [])

    def test_delete_missing_file_still_removes_entry(self):
        a, pa = self._add_file("gone.png")
        pa.unlink()
        self.assertEqual(self.lib.delete_images([a["id"]])["errors"], [])
        self.assertIsNone(self.lib.get_image(a["id"]))

    def test_relocate_moves_files_and_updates_paths(self):
        a, pa = self._add_file("a.png")
        new_dir = Path(self.tmp.name) / "elsewhere"
        res = self.lib.relocate(Path(self.tmp.name) / "imgs", new_dir)
        self.assertEqual(res, {"moved": 1, "failed": []})
        moved = Path(self.lib.get_image(a["id"])["path"])
        self.assertEqual(moved, (new_dir / "2026-10-12" / "a.png").resolve())
        self.assertTrue(moved.exists())
        self.assertFalse(pa.exists())

    def test_rewrite_path_prefix_after_folder_rename(self):
        self.lib.add_image("/Users/x/.beestudio/images/2026-10-12/a.png", {"prompt": "a", "loras": []})
        self.lib.add_image("/Volumes/Art/b.png", {"prompt": "b", "loras": []})
        self.lib.set_setting("images_dir", "/Users/x/.beestudio/images")
        n = self.lib.rewrite_path_prefix("/Users/x/.beestudio", "/Users/x/.protoartstudio")
        self.assertEqual(n, 1)
        paths_ = sorted(r["path"] for r in self.lib.query())
        self.assertEqual(paths_, ["/Users/x/.protoartstudio/images/2026-10-12/a.png", "/Volumes/Art/b.png"])
        self.assertEqual(self.lib.get_setting("images_dir"), "/Users/x/.protoartstudio/images")

    def test_settings_roundtrip(self):
        self.assertIsNone(self.lib.get_setting("images_dir"))
        self.lib.set_setting("images_dir", "/Volumes/Art")
        self.assertEqual(self.lib.get_setting("images_dir"), "/Volumes/Art")

    def test_model_filter(self):
        self.add("x", model="sd15")
        self.add("y", model="ernie")
        self.assertEqual([r["prompt"] for r in self.lib.query(model="ernie")], ["y"])

    def test_presets_roundtrip_and_overwrite(self):
        s = {"model": "sdxl", "sampler": "dpmpp_2m_karras", "steps": 30, "cfg": 6,
             "loras": [{"name": "pixel", "weight": 0.8}], "junk": "ignored"}
        presets = self.lib.save_preset("My look", s)
        self.assertEqual(presets[0]["name"], "My look")
        self.assertNotIn("junk", presets[0]["settings"])
        self.assertEqual(presets[0]["settings"]["loras"][0]["weight"], 0.8)
        self.lib.save_preset("My look", dict(s, steps=40))
        presets = self.lib.list_presets()
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["settings"]["steps"], 40)
        self.assertEqual(self.lib.delete_preset("My look"), [])
        with self.assertRaises(ValueError):
            self.lib.save_preset("  ", s)


class ContentFilterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = Library(Path(self.tmp.name) / "lib.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, prompt, negative=""):
        return self.lib.add_image(f"/x/{len(prompt)}.png", {"prompt": prompt, "negative_prompt": negative,
                                                            "model": "sd15", "seed": 1})

    def test_prompt_words(self):
        import nsfw
        for p in ("a nude woman on a beach", "(naked:1.3), forest", "score_9, rating_explicit, 1girl",
                  "NSFW, portrait", "bare_breasts", "my-nsfw-thing"):
            self.assertTrue(nsfw.prompt_flag(p), p)
        for p in ("grilled chicken breast", "a bare tree in winter", "sextant on a desk", "cocktail bar",
                  "Essex countryside", "", None, "a nudibranch underwater"):
            self.assertFalse(nsfw.prompt_flag(p), p)

    def test_model_names_and_lora_tags(self):
        import json as _json
        import nsfw
        self.assertTrue(nsfw.name_flag("realisticNSFW_v2"))
        self.assertTrue(nsfw.name_flag("pony", "hentai_mix.safetensors"))
        self.assertFalse(nsfw.name_flag("Juggernaut XL", "dreamshaper_8.safetensors", "Essex style"))
        for n in ("female ejaculation1", "AnalGapeCreampie-v4", "CheekBulgeFellatioMS", "hugefaketitsv2-000008",
                  "Fucking_POV_by_IsnAI", "bukkake_v0", "LickingOralLoRA"):
            self.assertTrue(nsfw.name_flag(n), n)
        for n in ("Analysis", "analog film", "comic strip", "cumulus clouds", "dickens portrait", "cocktail party",
                  "breastplate armor", "titanic", "Cockpit view", "greg simkins Surrealism v1", "Sneek_Ernie_epoch_7"):
            self.assertFalse(nsfw.name_flag(n), n)
        tags = {"10_x": {"1girl": 40, "nude": 30, "smile": 12}}
        self.assertTrue(nsfw.meta_flag({"ss_tag_frequency": _json.dumps(tags), "ss_num_train_images": "40"}))
        tags = {"10_x": {"1girl": 40, "nude": 1, "smile": 12}}
        self.assertFalse(nsfw.meta_flag({"ss_tag_frequency": _json.dumps(tags), "ss_num_train_images": "40"}))
        self.assertFalse(nsfw.meta_flag({"ss_tag_frequency": "not json"}))
        self.assertTrue(nsfw.meta_flag({"modelspec.nsfw": "True"}))

    def test_negative_prompt_does_not_count(self):
        safe = self.add("a cat", negative="nsfw, nude, lowres")
        self.assertFalse(safe["nsfw"])

    def test_hide_and_marks(self):
        cat = self.add("a cat")
        nude = self.add("nude figure study")
        beach = self.add("a beach")
        self.assertTrue(nude["nsfw"])
        self.assertEqual({r["id"] for r in self.lib.query(hide_nsfw=True)}, {cat["id"], beach["id"]})
        self.lib.set_nsfw_score(beach["id"], 0.93)           # the checker says so
        self.assertEqual([r["id"] for r in self.lib.query(hide_nsfw=True)], [cat["id"]])
        self.lib.set_nsfw_mark([beach["id"], nude["id"]], False)  # your mark wins
        self.assertEqual(len(self.lib.query(hide_nsfw=True)), 3)
        self.lib.set_nsfw_mark([cat["id"]], True)
        self.assertNotIn(cat["id"], {r["id"] for r in self.lib.query(hide_nsfw=True)})
        self.lib.set_nsfw_mark([cat["id"], beach["id"], nude["id"]], None)  # back to automatic
        self.assertEqual([r["id"] for r in self.lib.query(hide_nsfw=True)], [cat["id"]])
        self.assertEqual(len(self.lib.query()), 3)  # off = everything
        self.assertEqual(self.lib.nsfw_counts(), {"total": 3, "nsfw": 2, "unchecked": 2})
        self.assertEqual({i for i, _ in self.lib.unchecked()}, {cat["id"], nude["id"]})

    def test_existing_pictures_get_prompt_flags(self):
        import sqlite3
        db = Path(self.tmp.name) / "old2.sqlite"
        con = sqlite3.connect(db)
        con.executescript("""CREATE TABLE images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL,
            created_at REAL NOT NULL, prompt TEXT NOT NULL DEFAULT '', negative_prompt TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '', sampler TEXT NOT NULL DEFAULT '', steps INTEGER, cfg REAL,
            seed INTEGER, width INTEGER, height INTEGER, settings_json TEXT NOT NULL DEFAULT '{}',
            starred INTEGER NOT NULL DEFAULT 0);
            INSERT INTO images (path, created_at, prompt) VALUES ('/a.png', 1, 'naked man'), ('/b.png', 2, 'a cat');""")
        con.commit()
        con.close()
        lib = Library(db)
        self.assertEqual([r["prompt"] for r in lib.query(hide_nsfw=True)], ["a cat"])

    def test_marks_file(self):
        import nsfw
        import paths
        old = paths.HOME
        paths.HOME = Path(self.tmp.name)
        try:
            nsfw.set_mark("loras", "spicy", True)
            nsfw.set_mark("models", "sd15", False)
            nsfw.set_mark("people_loras", "jane", True)
            self.assertEqual(nsfw.load_marks(), {"models": {"sd15": False}, "loras": {"spicy": True},
                                                 "people_models": {}, "people_loras": {"jane": True}})
            nsfw.set_mark("people_loras", "jane", None)
            nsfw.set_mark("loras", "spicy", None)
            self.assertEqual(nsfw.load_marks()["loras"], {})
            self.assertTrue(nsfw.effective(False, True))
            self.assertFalse(nsfw.effective(True, False))
            self.assertTrue(nsfw.effective(True, None))
        finally:
            paths.HOME = old


class LoraDetectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detects_families(self):
        write_fake_safetensors(self.d / "sd15.safetensors",
                               ["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", "lora_te_text_model.x"])
        write_fake_safetensors(self.d / "xl.safetensors",
                               ["lora_unet_input_blocks_4_1.lora_down.weight", "lora_te2_text_model.x"])
        write_fake_safetensors(self.d / "xl_meta.safetensors", ["lora_unet_x"],
                               {"ss_base_model_version": "sdxl_base_v1-0"})
        write_fake_safetensors(self.d / "ernie.safetensors",
                               ["transformer.layers.0.attn.to_q.lora_A.weight"])
        write_fake_safetensors(self.d / "tagged.safetensors", ["anything"], {"beestudio_family": "ernie"})
        write_fake_safetensors(self.d / "flux_kohya.safetensors",
                               ["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight"])
        write_fake_safetensors(self.d / "flux_diffusers.safetensors",
                               ["transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight"])
        (self.d / "broken.safetensors").write_bytes(b"nope")
        found = {l["name"]: l["family"] for l in loras.scan(self.d)}
        self.assertEqual(found, {"sd15": "sd15", "xl": "sdxl", "xl_meta": "sdxl", "ernie": "ernie",
                                 "tagged": "ernie", "broken": "unknown",
                                 "flux_kohya": "flux", "flux_diffusers": "flux"})


class LoraTypeOverrideTests(unittest.TestCase):
    def test_your_pick_wins_over_folder_and_file(self):
        import tempfile as _tf
        old_f, old_n = dict(loras.FOLDER_FAMILY), dict(loras.NAME_FAMILY)
        with _tf.TemporaryDirectory() as d:
            write_fake_safetensors(Path(d) / "inky.safetensors", ["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", "lora_te_text_model.x"])
            write_fake_safetensors(Path(d) / "other.safetensors", ["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", "lora_te_text_model.x"])
            try:
                loras.FOLDER_FAMILY.clear(); loras.NAME_FAMILY.clear()
                loras.FOLDER_FAMILY[d] = "sdxl"
                loras.NAME_FAMILY["inky"] = "flux"
                rows = {r["name"]: r for r in loras.all_loras(d)}
                self.assertEqual(rows["inky"]["family"], "flux")
                self.assertTrue(rows["inky"]["family_set"])
                self.assertEqual(rows["other"]["family"], "sdxl")      # the folder's type
                self.assertEqual(rows["other"]["detected_family"], "sd15")
            finally:
                loras.FOLDER_FAMILY.clear(); loras.FOLDER_FAMILY.update(old_f)
                loras.NAME_FAMILY.clear(); loras.NAME_FAMILY.update(old_n)


class ModelTypeOverrideTests(unittest.TestCase):
    def test_you_can_retype_a_model_file(self):
        import engine
        import model_hub
        mid = "local:/x/dream.safetensors"
        old_l, old_f = list(model_hub.LOCAL_MODELS), dict(model_hub.MODEL_FAMILY)
        try:
            model_hub.LOCAL_MODELS[:] = [{"id": mid, "path": "/x/dream.safetensors", "kind": "file", "family": "sd15", "label": "dream"}]
            self.assertEqual(engine.all_models()[mid]["family"], "sd15")
            model_hub.MODEL_FAMILY[mid] = "sdxl"
            spec = engine.all_models()[mid]
            self.assertEqual((spec["family"], spec["detected_family"], spec["defaults"]["width"]), ("sdxl", "sd15", 1024))
            self.assertTrue(spec["family_set"])
        finally:
            model_hub.LOCAL_MODELS[:] = old_l
            model_hub.MODEL_FAMILY.clear()
            model_hub.MODEL_FAMILY.update(old_f)


class LoraFolderTypeTests(unittest.TestCase):
    def test_folder_type_wins_and_deepest_folder_counts(self):
        old = dict(loras.FOLDER_FAMILY)
        try:
            loras.FOLDER_FAMILY.clear()
            loras.FOLDER_FAMILY.update({"/x/loras": "sd15", "/x/loras/xl": "sdxl", "/y": "bogus"})
            self.assertEqual(loras.folder_family("/x/loras/a.safetensors"), "sd15")
            self.assertEqual(loras.folder_family("/x/loras/xl/sub/b.tdict"), "sdxl")
            self.assertIsNone(loras.folder_family("/x/lorasX/c.safetensors"))  # not a sub-folder
            self.assertIsNone(loras.folder_family("/y/d.safetensors"))          # unknown type ignored
        finally:
            loras.FOLDER_FAMILY.clear()
            loras.FOLDER_FAMILY.update(old)


class SamplerTests(unittest.TestCase):
    def test_catalogue(self):
        for fam in ("sd15", "sdxl", "ernie"):
            lst = samplers.list_for(fam)
            self.assertEqual(sum(1 for s in lst if s["default"]), 1, fam)
        ernie_ids = {s["id"] for s in samplers.list_for("ernie")}
        self.assertTrue(all(i.startswith("flow_") for i in ernie_ids))
        self.assertIn("dpmpp_2m_karras", {s["id"] for s in samplers.list_for("sdxl")})
        self.assertEqual([x["id"] for x in samplers.list_for("flux")], ["flow_euler"])
        self.assertEqual(samplers.recommended_steps("flux", "flow_euler", 4)[0], 4)

    def test_sdxl_flavors(self):
        import engine
        d = loras.detect_flavor
        self.assertEqual(d("ponyDiffusionV6XL_v6StartWithThisOne"), "pony")
        self.assertEqual(d("autismmixSDXL_autismmixPony"), "pony")
        self.assertEqual(d("waiNSFWIllustrious_v140"), "illustrious")
        self.assertEqual(d("noobaiXLNAIXL_vPred10"), "noobai")
        self.assertEqual(d("NoobAI-XL based on Illustrious"), "noobai")
        self.assertIsNone(d("juggernautXL_v9"))
        self.assertIsNone(d("ponytail_girl"))  # not a whole word
        self.assertEqual(loras.flavor_from_meta({"ss_sd_model_name": "ponyDiffusionV6XL.safetensors"}, "my_lora"), "pony")
        self.assertEqual(engine.model_flavor({"family": "sdxl", "label": "Pony Realism (added)"}), "pony")
        self.assertIsNone(engine.model_flavor({"family": "sd15", "label": "pony"}))
        tags = engine.QUALITY_TAGS["pony"]["prompt"]
        self.assertEqual(engine.add_quality_tags("1girl, smile", tags), tags + ", 1girl, smile")
        self.assertEqual(engine.add_quality_tags(tags + ", 1girl", tags), tags + ", 1girl")  # not twice

    def test_sdxl_training_base(self):
        import engine
        full = engine.MODELS["sdxl_base_fp16"]
        int8 = engine.MODELS["sdxl_int8"]
        self.assertEqual(full["repo"], int8["repo"])  # same download the 8-bit build is made from
        self.assertFalse(full.get("quantize_unet") or full.get("local_name"))

    def test_flux_models_and_preview(self):
        import numpy as np
        import engine
        import preview
        flux = {k: v for k, v in engine.MODELS.items() if v["family"] == "flux"}
        self.assertEqual(len(flux), 3)
        for v in flux.values():
            self.assertEqual(v["engine"], "mlx")
            self.assertEqual(v["mlx_base"], "schnell")
            self.assertEqual(v["defaults"]["steps"], 4)
        # FLUX packs 2x2 patches of 16 channels into 64 numbers per token
        h, w = 64, 96
        lat = np.random.rand(16, h // 8, w // 8).astype(np.float32)
        tokens = lat.reshape(16, h // 16, 2, w // 16, 2).transpose(1, 3, 0, 2, 4).reshape(-1, 64)
        np.testing.assert_allclose(preview.unpack_flux_tokens(tokens, h, w), lat)
        self.assertEqual(preview.latent_to_rgb(lat, "flux").shape, (h // 8, w // 8, 3))


if __name__ == "__main__":
    unittest.main()
