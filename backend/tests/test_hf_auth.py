"""Hugging Face sign-in for gated models: the token is only used when you saved one, and errors explain what to do."""
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import net  # noqa: E402


class GatedRepoError(Exception):
    pass


class HfAuthTests(unittest.TestCase):
    def tearDown(self):
        net.apply_hf_token(None)

    def test_token_only_when_saved(self):
        net.apply_hf_token("hf_" + "a" * 30)
        self.assertEqual(os.environ["HF_TOKEN"], "hf_" + "a" * 30)
        self.assertEqual(os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "0")
        net.apply_hf_token(None)
        self.assertNotIn("HF_TOKEN", os.environ)
        self.assertEqual(os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")

    def test_messages(self):
        m = net.hf_auth_problem(GatedRepoError("Access to model a/b is restricted"), "a/b")
        self.assertIn("accept its licence", m)
        self.assertIn("Access Tokens", m)
        net.apply_hf_token("hf_" + "b" * 30)
        self.assertIn("turned the download", net.hf_auth_problem(Exception("401 Client Error: Unauthorized"), "a/b"))
        self.assertIsNone(net.hf_auth_problem(Exception("404 Client Error: not found")))

    def test_token_format(self):
        self.assertTrue(re.fullmatch(net.HF_TOKEN_RE, "hf_" + "Ab3" * 10))
        self.assertFalse(re.fullmatch(net.HF_TOKEN_RE, "not-a-token"))


if __name__ == "__main__":
    unittest.main()
