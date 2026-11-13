"""🧊 Make 3D: the preview drawing, where its files live, and that it's a queued job asking for the licence."""
import types
import unittest

import numpy as np


class Mesh3D(unittest.TestCase):
    def test_preview_draws_a_shaded_shape(self):
        import mesh3d_worker
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        img = mesh3d_worker.preview(types.SimpleNamespace(vertices=v, faces=f), size=128)
        a = np.asarray(img)
        self.assertEqual(a.shape, (128, 128, 3))
        self.assertGreater((a.max(axis=2) > 80).sum(), 200)  # something lit was drawn

    def test_image_encoder_names_for_transformers_5(self):
        """Hunyuan3D's DINOv2 checkpoint (transformers 4 names) is renamed to transformers 5's, and the SwiGLU's
        weights_in is split into gate_proj (first half) and up_proj (second half)."""
        import mesh3d_worker as w

        class T(np.ndarray):
            def chunk(self, n, dim=0):
                return np.split(self, n, axis=dim)
        p = "main_image_encoder.model.encoder.layer.3."
        win = np.arange(8.0).reshape(4, 2).view(T)
        state = {p + "attention.attention.query.weight": 1, p + "attention.attention.key.bias": 2,
                 p + "attention.attention.value.weight": 3, p + "attention.output.dense.bias": 4,
                 p + "mlp.weights_in.weight": win, p + "mlp.weights_out.bias": 5, p + "norm1.weight": 6,
                 "model.blocks.0.attn.qkv.weight": 7}
        out = w.dinov2_new_names(state)
        self.assertEqual(out[p + "attention.q_proj.weight"], 1)
        self.assertEqual(out[p + "attention.k_proj.bias"], 2)
        self.assertEqual(out[p + "attention.v_proj.weight"], 3)
        self.assertEqual(out[p + "attention.o_proj.bias"], 4)
        self.assertEqual(out[p + "mlp.down_proj.bias"], 5)
        self.assertEqual(out[p + "norm1.weight"], 6)
        self.assertEqual(out["model.blocks.0.attn.qkv.weight"], 7)
        self.assertEqual(out[p + "mlp.gate_proj.weight"].tolist(), [[0, 1], [2, 3]])
        self.assertEqual(out[p + "mlp.up_proj.weight"].tolist(), [[4, 5], [6, 7]])
        self.assertEqual(len(out), len(state) + 1)

    def test_files_live_in_the_data_folder_and_the_code_is_pinned(self):
        import mesh3d
        import paths
        self.assertTrue(str(mesh3d.site()).startswith(str(paths.HOME)))
        self.assertEqual(len(mesh3d.COMMIT), 40)
        self.assertIn("not affiliated", mesh3d.NOTICE)

    def test_only_wheels_for_this_python_on_apple_silicon_are_picked(self):
        import sys
        import mesh3d
        v = f"cp{sys.version_info[0]}{sys.version_info[1]}"
        self.assertTrue(mesh3d._wheel_ok("einops-0.8.1-py3-none-any.whl"))
        self.assertTrue(mesh3d._wheel_ok(f"scikit_image-0.25.2-{v}-{v}-macosx_12_0_arm64.whl"))
        self.assertFalse(mesh3d._wheel_ok(f"scikit_image-0.25.2-{v}-{v}-manylinux_2_17_x86_64.whl"))
        self.assertFalse(mesh3d._wheel_ok(f"x-1-{v}-{v}-macosx_10_9_x86_64.whl"))
        self.assertFalse(mesh3d._wheel_ok("x-1-cp27-cp27m-macosx_11_0_arm64.whl"))

    def test_make3d_is_a_queued_job(self):
        import bee_backend as b
        self.assertIn("make3d", b.HEAVY)


if __name__ == "__main__":
    unittest.main()
