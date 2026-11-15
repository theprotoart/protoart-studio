"""Relight's ⚡ Fast / Standard / ★ Best: Fast skips the slow 1.5× detail pass."""
import unittest

from postprocess import relight as rl


class RelightSpeed(unittest.TestCase):
    def test_plans(self):
        self.assertEqual(rl.speed_plan("fast", True), (12, False))   # never the detail pass
        self.assertEqual(rl.speed_plan("standard", True), (20, True))
        self.assertEqual(rl.speed_plan("standard", False), (20, False))
        self.assertEqual(rl.speed_plan("best", True), (30, True))

    def test_turbo(self):
        self.assertEqual(rl.speed_plan("turbo", True), (6, True))    # cheap enough to keep the detail pass
        self.assertLessEqual(rl.TURBO_CFG, 2.0)

    def test_lcm_lora_skips_the_widened_first_layer(self):
        state = {"lora_unet_conv_in.lora_down.weight": 1, "unet.conv_in.lora_A.weight": 2,
                 "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": 3}
        self.assertEqual(list(rl.lcm_for_iclight(state)), ["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"])

    def test_old_settings_and_steps(self):
        self.assertEqual(rl.speed_plan(None, True), (20, True))      # saved before Speed existed
        self.assertEqual(rl.speed_plan("weird", True), (20, True))
        self.assertEqual(rl.speed_plan("fast", True, steps=200), (50, False))
        self.assertEqual(rl.speed_plan("best", True, steps=3), (8, True))


if __name__ == "__main__":
    unittest.main()
