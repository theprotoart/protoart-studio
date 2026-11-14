"""Fixes from the security audit (SECURITY.md): an id is never a path, the Civitai key only goes to civitai.com, and
the data folder and its database are readable by your own account only."""
import os
import stat
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import civitai
import paths
import zengineer


class ZengineerIdIsNotAPath(unittest.TestCase):
    def test_unknown_ids_are_refused(self):
        for bad in ("../../Documents", "zeng_v6/../../x", "/tmp/x", "", None):
            with self.assertRaises(ValueError):
                zengineer.folder(bad)

    def test_remove_never_deletes_outside(self):
        with mock.patch("shutil.rmtree") as rm:
            with self.assertRaises(ValueError):
                zengineer.remove("../../../Pictures")
            rm.assert_not_called()

    def test_known_ids_still_work(self):
        for mid in zengineer.MODELS:
            self.assertTrue(str(zengineer.folder(mid)).endswith(f"{mid}-mlx-4bit"))


class CivitaiKeyOnlyToCivitai(unittest.TestCase):
    def sent(self, url):
        """-> (the URL asked for, the headers): with the key saved, for a file at url."""
        seen = {}

        def fake(req, timeout):
            seen["url"] = req.full_url
            seen["headers"] = {**req.headers, **req.unredirected_hdrs}
            raise civitai.urllib.error.HTTPError(req.full_url, 500, "stop", {}, None)

        info = {"ok": True, "kind": "lora", "file": {"url": url, "name": "x.safetensors"}}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(civitai, "_urlopen", fake):
            with self.assertRaises(ValueError):
                civitai.download(info, d, token="SECRET-KEY")
        return seen["url"], seen["headers"]

    def test_civitai_gets_the_key(self):
        url, headers = self.sent("https://civitai.com/api/download/models/1")
        self.assertIn("SECRET-KEY", url)
        self.assertIn("SECRET-KEY", " ".join(headers.values()))

    def test_civitai_red_too(self):
        url, _ = self.sent("https://civitai.red/api/download/models/1")
        self.assertIn("SECRET-KEY", url)

    def test_other_hosts_never_do(self):
        for u in ("https://evil.example/file", "https://civitai.com.evil.example/x", "http://civitai.com/api/download/models/1",
                  "https://notcivitai.com/x", "https://civitai.red.evil.example/x"):
            url, headers = self.sent(u)
            self.assertNotIn("SECRET-KEY", url, u)
            self.assertNotIn("SECRET-KEY", " ".join(headers.values()), u)


class PrivateDataFolder(unittest.TestCase):
    def test_folder_and_database_are_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "home"
            home.mkdir(mode=0o755)
            db = home / "library.db"
            db.write_text("")
            os.chmod(db, 0o644)
            paths.private(home)
            paths.private(str(db), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(home).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.stat(db).st_mode), 0o600)

    def test_links_are_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "elsewhere"
            target.mkdir(mode=0o755)
            link = Path(d) / "link"
            link.symlink_to(target)
            paths.private(link)
            self.assertEqual(stat.S_IMODE(os.stat(target).st_mode), 0o755)


if __name__ == "__main__":
    unittest.main()
