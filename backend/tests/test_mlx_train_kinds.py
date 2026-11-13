"""Fast-engine LoRA training: ERNIE, FLUX.2 Klein (4B / 9B) and Z-Image Turbo get mflux configs of their own."""
import unittest

import mlx_train


class Kinds(unittest.TestCase):
    def cfg(self, spec, targets="attention"):
        return mlx_train.build_config(model_path="/m", data_dir="/d", out_dir="/o", steps=100, n_samples=10, lr=1e-4,
                                      rank=8, res=512, targets=targets, save_every=0, ram_gb=8, spec=spec)

    def test_each_fast_model_has_its_own_settings(self):
        e = self.cfg({"family": "ernie", "engine": "mlx"})
        self.assertEqual((e["model"], e["steps"]), ("ernie-image-turbo", 8))
        k4 = self.cfg({"family": "flux2", "engine": "mlx", "mlx_base": "flux2-klein-4b"}, "full")
        k9 = self.cfg({"family": "flux2", "engine": "mlx", "mlx_base": "flux2-klein-9b"})
        self.assertEqual(k4["model"], "flux2-klein-4b")
        self.assertEqual(k9["model"], "flux2-klein-9b")
        ends = {t["blocks"]["end"] for t in k9["lora_layers"]["targets"]}
        self.assertEqual(ends, {8, 24})  # 9B: 8 double + 24 single blocks
        self.assertEqual({t["blocks"]["end"] for t in k4["lora_layers"]["targets"]}, {5, 20})
        z = self.cfg({"family": "zimage", "engine": "mlx"})
        self.assertEqual((z["model"], z["guidance"], z["training_loop"]["timestep_low"]), ("z-image-turbo", 0.0, 4))

    def test_only_fast_engine_models_of_those_kinds(self):
        self.assertEqual(mlx_train.kind_of({"family": "flux2", "engine": "mlx"}), "flux2")
        self.assertIsNone(mlx_train.kind_of({"family": "flux", "engine": "mlx"}))
        self.assertIsNone(mlx_train.kind_of({"family": "sdxl"}))


if __name__ == "__main__":
    unittest.main()
