import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class AttachWordsTests(unittest.TestCase):
    def test_new_words_train_with_this_transformers(self):
        try:
            import torch
            from transformers import CLIPTextConfig, CLIPTextModel
        except Exception:
            self.skipTest("torch / transformers not installed")
        import ti_trainer
        cfg = CLIPTextConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, max_position_embeddings=16)
        te = CLIPTextModel(cfg).eval()
        te.requires_grad_(False)
        base = te.get_input_embeddings()
        ids = [98, 99]
        words = ti_trainer._NewWords(base, ids, base.weight[ids].detach().float())
        ti_trainer.attach_words(te, words)
        self.assertIs(te.get_input_embeddings(), words)
        out = te(torch.tensor([[1, 98, 5, 99, 2]]))[0]
        out.pow(2).mean().backward()
        self.assertIsNotNone(words.vec.grad)
        self.assertGreater(float(words.vec.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
