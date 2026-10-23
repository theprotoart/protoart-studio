import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kv_cache


def tiny_unet(linear=False):
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=24, attention_head_dim=4, norm_num_groups=8, use_linear_projection=linear).eval()


class KVCacheTests(unittest.TestCase):
    def test_same_output_and_reuse(self):
        import torch
        torch.manual_seed(0)
        for linear in (False, True):
            unet = tiny_unet(linear)
            x = torch.randn(2, 4, 8, 8)
            ctx = torch.randn(2, 7, 24)
            with torch.no_grad():
                want = [unet(x, t, encoder_hidden_states=ctx).sample for t in (900, 500, 100)]
                n = kv_cache.install(unet)
                self.assertGreater(n, 0)
                got = [unet(x, t, encoder_hidden_states=ctx).sample for t in (900, 500, 100)]
            for a, b in zip(want, got):
                self.assertLess((a - b).abs().max().item(), 1e-5)
            self.assertEqual(kv_cache.hits(unet), 2 * n)  # first step works them out, the next two reuse them

    def test_new_prompt_is_worked_out_again(self):
        import torch
        torch.manual_seed(1)
        unet = tiny_unet()
        x = torch.randn(1, 4, 8, 8)
        c1, c2 = torch.randn(1, 7, 24), torch.randn(1, 7, 24)
        with torch.no_grad():
            want = unet(x, 500, encoder_hidden_states=c2).sample
            kv_cache.install(unet)
            unet(x, 500, encoder_hidden_states=c1)
            got = unet(x, 500, encoder_hidden_states=c2).sample
        self.assertLess((want - got).abs().max().item(), 1e-5)

    def test_slicing_and_removal(self):
        import torch
        unet = tiny_unet()
        unet.set_attention_slice("auto")  # what attention slicing on 8 GB Macs does
        kv_cache.install(unet)
        names = {type(p).__name__ for k, p in unet.attn_processors.items() if ".attn1." in k}
        self.assertEqual(names, {"SlicedAttnProcessor"})  # self-attention keeps the app's choice
        kv_cache.remove(unet)
        self.assertFalse(any(isinstance(p, kv_cache.CachedCrossAttn) for p in unet.attn_processors.values()))


if __name__ == "__main__":
    unittest.main()
