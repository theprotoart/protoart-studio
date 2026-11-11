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

    def test_files_live_in_the_data_folder_and_the_code_is_pinned(self):
        import mesh3d
        import paths
        self.assertTrue(str(mesh3d.site()).startswith(str(paths.HOME)))
        self.assertEqual(len(mesh3d.COMMIT), 40)
        self.assertIn("not affiliated", mesh3d.NOTICE)

    def test_make3d_is_a_queued_job(self):
        import bee_backend as b
        self.assertIn("make3d", b.HEAVY)


if __name__ == "__main__":
    unittest.main()
