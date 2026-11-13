"""💡 Relight (IC-Light): the light-direction start pictures, sizes, the grey-backed subject and the crop helper."""
import unittest

import numpy as np
from PIL import Image

from postprocess import relight as rl


class RelightHelpers(unittest.TestCase):
    def test_gradient_is_bright_where_the_light_comes_from(self):
        for d, bright, dark in (("left", (0, 0), (0, -1)), ("right", (0, -1), (0, 0)),
                                ("top", (0, 0), (-1, 0)), ("bottom", (-1, 0), (0, 0))):
            g = rl.gradient(d, 64, 48)
            self.assertEqual(g.shape, (48, 64, 3))
            self.assertEqual(g[bright][0], 255, d)
            self.assertEqual(g[dark][0], 0, d)

    def test_sizes_are_multiples_of_64_with_the_long_side_asked_for(self):
        w, h = rl.sizes(3000, 2000, 640)
        self.assertEqual((w % 64, h % 64), (0, 0))
        self.assertEqual(w, 640)
        self.assertAlmostEqual(w / h, 1.5, delta=0.1)
        self.assertEqual(rl.sizes(500, 4000, 512)[1], 512)

    def test_foreground_puts_the_subject_on_mid_grey(self):
        img = Image.new("RGB", (4, 2), (255, 0, 0))
        a = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
        fg = rl.foreground(img, a)
        self.assertEqual(tuple(fg[0, 0]), (255, 0, 0))
        self.assertEqual(tuple(fg[0, 3]), (127, 127, 127))

    def test_fit_crops_to_the_exact_size(self):
        arr = np.zeros((100, 300, 3), dtype=np.uint8)
        self.assertEqual(rl.fit(arr, 128, 128).shape, (128, 128, 3))

    def test_place_back_keeps_the_background_and_takes_the_subject(self):
        from PIL import Image
        import postprocess.relight as rl
        orig = Image.new("RGB", (200, 100), (0, 0, 255))
        alpha = np.zeros((100, 200), dtype=np.float32)
        alpha[30:70, 80:120] = 1.0
        relit = Image.new("RGB", (128, 64), (255, 0, 0))
        out = np.asarray(rl.place_back(orig, alpha, relit))
        self.assertEqual(out.shape, (64, 128, 3))
        self.assertGreater(out[32, 64, 0], 200)   # the middle: the relit subject
        self.assertGreater(out[2, 2, 2], 200)     # a corner: the original background
        self.assertLess(out[2, 2, 0], 60)

    def test_relit_background_keeps_its_detail_and_takes_the_new_light(self):
        from PIL import Image
        import postprocess.relight as rl
        rng = np.random.default_rng(0)
        detail = rng.integers(60, 140, (64, 128, 3)).astype(np.uint8)  # a busy grey background
        orig = Image.fromarray(detail)
        alpha = np.zeros((64, 128), dtype=np.float32)
        alpha[20:44, 54:74] = 1.0
        relit = Image.new("RGB", (128, 64), (230, 120, 40))  # IC-Light's new background: warm orange light
        out = np.asarray(rl.place_back(orig, alpha, relit, relight_bg=True)).astype(float)
        corner = out[:12, :30]
        self.assertGreater(corner[..., 0].mean(), corner[..., 2].mean() + 40)    # now warm
        self.assertGreater(corner[..., 0].std(), 8)                             # the detail is still there
        kept = np.asarray(rl.place_back(orig, alpha, relit)).astype(float)[:12, :30]
        self.assertLess(abs(kept[..., 0].mean() - kept[..., 2].mean()), 10)     # Keep mine: unchanged grey

    def test_finish_keeps_the_original_size_and_throws_a_shadow_away_from_the_light(self):
        from PIL import Image
        import postprocess.relight as rl
        W, H = 1000, 750
        orig = Image.new("RGB", (W, H), (120, 120, 120))
        alpha = np.zeros((H, W), dtype=np.float32)
        alpha[150:650, 450:550] = 1.0
        w, h = rl.sizes(W, H, 512)
        relit = Image.new("RGB", (w, h), (200, 60, 60))
        out = np.asarray(rl.finish(orig, alpha, relit, mode="keep", direction="left", blend=False)).astype(float)
        self.assertEqual(out.shape, (H, W, 3))
        self.assertGreater(out[400, 500, 0], 180)              # the relit subject, at full size
        right, left = out[540:600, 620:720].mean(), out[540:600, 280:380].mean()
        self.assertLess(right, left - 5)                        # light from the left: the shadow falls to the right
        new = np.asarray(rl.finish(orig, alpha, relit, mode="new")).astype(float)
        self.assertGreater(new[50, 50, 0], 180)                 # New from Relight: its own background
        warm = np.asarray(rl.finish(orig, alpha, relit, mode="keep", shadow=False, blend=False, warmth=1.0)).astype(float)
        self.assertGreater(warm[50, 50, 0], warm[50, 50, 2] + 20)

    def test_depth_shading_brightens_what_faces_the_light(self):
        from PIL import Image
        import postprocess.relight as rl
        d = np.tile(np.linspace(0, 255, 200)[None, :], (100, 1)).astype(np.uint8)  # nearer to the right
        g = rl.depth_shading(Image.fromarray(d), (1.0, 0.0))
        self.assertEqual(g.shape, (100, 200))
        self.assertGreater(g[:, 150:].mean(), g[:, :50].mean())

    def test_relight_is_a_queued_job(self):
        import bee_backend as b
        self.assertIn("relight", b.HEAVY)


if __name__ == "__main__":
    unittest.main()
