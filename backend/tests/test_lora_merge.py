"""LoRA merging: the exact merge reproduces the weighted sum of the LoRAs' changes; 'smaller' is close."""
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError:  # the app's own Python has them; a bare test machine may not
    torch = None

import lora_merge


def make(path, rank, alpha, seed, layout="kohya", conv=False, extra=None):
    g = torch.Generator().manual_seed(seed)
    d, u = (".lora_down.weight", ".lora_up.weight") if layout == "kohya" else (".lora_A.weight", ".lora_B.weight")
    sd = {}
    for base, (o, i) in {"lora_unet_a": (40, 32), "lora_unet_b": (24, 32)}.items():
        sd[base + d] = torch.randn(rank, i, generator=g)
        sd[base + u] = torch.randn(o, rank, generator=g)
        if layout == "kohya":
            sd[base + ".alpha"] = torch.tensor(float(alpha))
    if conv:
        sd["lora_unet_c" + d] = torch.randn(rank, 8, 3, 3, generator=g)
        sd["lora_unet_c" + u] = torch.randn(16, rank, 1, 1, generator=g)
        if layout == "kohya":
            sd["lora_unet_c.alpha"] = torch.tensor(float(alpha))
    sd.update(extra or {})
    save_file(sd, str(path), metadata={"ss_tag_frequency": '{"x": {"cat": 3}}'})
    return sd


def delta(sd, base, layout="kohya"):
    d, u = (".lora_down.weight", ".lora_up.weight") if layout == "kohya" else (".lora_A.weight", ".lora_B.weight")
    down, up = sd[base + d].float(), sd[base + u].float()
    a = sd.get(base + ".alpha")
    s = (float(a) / down.shape[0]) if a is not None else 1.0
    return s * (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1))


@unittest.skipIf(torch is None, "needs torch + safetensors (run with the app's Python)")
class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_merge_is_the_weighted_sum(self):
        a = make(self.d / "a.safetensors", 4, 2, 1, conv=True)
        b = make(self.d / "b.safetensors", 8, 8, 2, conv=True)
        r = lora_merge.merge([{"path": str(self.d / "a.safetensors"), "weight": 0.7},
                              {"path": str(self.d / "b.safetensors"), "weight": 0.4}], self.d / "m.safetensors")
        self.assertEqual(r["rank"], 12)
        from safetensors.torch import load_file
        m = load_file(str(self.d / "m.safetensors"))
        for base in ("lora_unet_a", "lora_unet_b", "lora_unet_c"):
            want = 0.7 * delta(a, base) + 0.4 * delta(b, base)
            got = delta(m, base)
            self.assertLess(float((got - want).abs().max() / want.abs().max()), 2e-3, base)  # fp16 file

    def test_smaller_merge_is_close_and_small(self):
        make(self.d / "a.safetensors", 8, 8, 3)
        make(self.d / "b.safetensors", 8, 8, 4)
        r = lora_merge.merge([{"path": str(self.d / "a.safetensors"), "weight": 1},
                              {"path": str(self.d / "b.safetensors"), "weight": 1}], self.d / "s.safetensors",
                             mode="smaller", rank=16)
        self.assertEqual(r["rank"], 16)  # 8 + 8 fits exactly in 16
        from safetensors.torch import load_file
        a, b, m = (load_file(str(self.d / f)) for f in ("a.safetensors", "b.safetensors", "s.safetensors"))
        want = delta(a, "lora_unet_a") + delta(b, "lora_unet_a")
        self.assertLess(float((delta(m, "lora_unet_a") - want).abs().max() / want.abs().max()), 5e-3)
        r = lora_merge.merge([{"path": str(self.d / "a.safetensors"), "weight": 1},
                              {"path": str(self.d / "b.safetensors"), "weight": 1}], self.d / "t.safetensors",
                             mode="smaller", rank=4)
        self.assertEqual(r["rank"], 4)

    def test_peft_layout_and_mixed_layouts(self):
        a = make(self.d / "a.safetensors", 4, 4, 5, layout="peft")
        b = make(self.d / "b.safetensors", 4, 4, 6, layout="peft")
        lora_merge.merge([{"path": str(self.d / "a.safetensors"), "weight": 1},
                          {"path": str(self.d / "b.safetensors"), "weight": 0.5}], self.d / "p.safetensors")
        from safetensors.torch import load_file
        m = load_file(str(self.d / "p.safetensors"))
        self.assertFalse(any(k.endswith(".alpha") for k in m))  # PEFT files carry no alpha keys
        want = delta(a, "lora_unet_a", "peft") + 0.5 * delta(b, "lora_unet_a", "peft")
        self.assertLess(float((delta(m, "lora_unet_a", "peft") - want).abs().max() / want.abs().max()), 2e-3)
        make(self.d / "k.safetensors", 4, 4, 7)
        with self.assertRaises(lora_merge.MergeError):
            lora_merge.plan([{"path": str(self.d / "a.safetensors")}, {"path": str(self.d / "k.safetensors")}])

    def test_refuses_lycoris_and_mismatched_models(self):
        make(self.d / "a.safetensors", 4, 4, 8)
        make(self.d / "h.safetensors", 4, 4, 9, extra={"lora_unet_x.hada_w1_a": torch.zeros(2, 2)})
        with self.assertRaises(lora_merge.MergeError):
            lora_merge.merge([{"path": str(self.d / "a.safetensors")}, {"path": str(self.d / "h.safetensors")}],
                             self.d / "x.safetensors")
        g = torch.Generator().manual_seed(1)
        save_file({"lora_unet_a.lora_down.weight": torch.randn(4, 64, generator=g),
                   "lora_unet_a.lora_up.weight": torch.randn(40, 4, generator=g)}, str(self.d / "w.safetensors"))
        with self.assertRaises(lora_merge.MergeError):
            lora_merge.merge([{"path": str(self.d / "a.safetensors")}, {"path": str(self.d / "w.safetensors")}],
                             self.d / "y.safetensors")
        with self.assertRaises(lora_merge.MergeError):
            lora_merge.merge([{"path": str(self.d / "a.safetensors")}], self.d / "z.safetensors")


if __name__ == "__main__":
    unittest.main()
