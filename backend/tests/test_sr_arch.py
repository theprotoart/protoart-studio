"""⏩ Instant's newer upscalers (postprocess/sr_arch.py): the DAT2 and RealPLKSR networks have exactly the tensors (names
and shapes) of the released checkpoints, the settings read back from a checkpoint are the ones it was made with, and
the networks make pictures 4× larger. The fingerprints are sha256 of the sorted "name:shape" lines of
4xNomos2_hq_dat2.pth (DAT2, the same shape as 4x-UltraSharpV2), 4xNomosWebPhoto_RealPLKSR.safetensors and
4xNomos2_realplksr_dysample.pth."""
import hashlib
import unittest

try:
    import torch
except ImportError:  # the app's own Python has torch; a bare test machine may not
    torch = None

DAT2 = dict(embed_dim=180, depth=[6] * 6, num_heads=[6] * 6, split_size=[8, 32], expansion_factor=2.0, upscale=4)


def fingerprint(net):
    lines = sorted(f"{k}:{'x'.join(map(str, v.shape))}" for k, v in net.state_dict().items())
    return len(lines), hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


@unittest.skipIf(torch is None, "needs torch")
class SRArch(unittest.TestCase):
    def test_dat2_matches_the_checkpoint(self):
        from postprocess import sr_arch
        net = sr_arch.DAT(**DAT2)
        self.assertEqual(fingerprint(net), (2116, "944c251045411b9a"))
        state = net.state_dict()
        self.assertTrue(sr_arch.is_dat(state))
        again, scale = sr_arch.dat_from_state(state)
        self.assertEqual(scale, 4)
        self.assertEqual(fingerprint(again), fingerprint(net))
        again.load_state_dict(state, strict=True)

    def test_realplksr_matches_the_checkpoints(self):
        from postprocess import sr_arch
        plain = sr_arch.RealPLKSR()
        self.assertEqual(fingerprint(plain), (340, "922dcc281f6d583e"))
        self.assertEqual(fingerprint(sr_arch.RealPLKSR(dysample=True)), (346, "efffa14f54b692ed"))
        state = plain.state_dict()
        self.assertTrue(sr_arch.is_realplksr(state) and not sr_arch.is_dat(state))
        again, scale = sr_arch.realplksr_from_state(state)
        self.assertEqual(scale, 4)
        again.load_state_dict(state, strict=True)

    def test_four_times_larger(self):
        from postprocess import sr_arch
        x = torch.rand(1, 3, 20, 28)
        with torch.no_grad():
            small_dat = sr_arch.DAT(embed_dim=36, depth=[2], num_heads=[2], split_size=[4, 8], upscale=4).eval()
            self.assertEqual(tuple(small_dat(x).shape), (1, 3, 80, 112))
            for dys in (False, True):
                plk = sr_arch.RealPLKSR(dim=16, n_blocks=2, dysample=dys).eval()
                self.assertEqual(tuple(plk(x).shape), (1, 3, 80, 112))


if __name__ == "__main__":
    unittest.main()
