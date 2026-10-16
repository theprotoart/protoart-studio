"""Civitai import: link parsing, reading the API answer, and the download (against a local test server)."""
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import civitai  # noqa: E402

LORA_MODEL = {
    "id": 12345, "name": "Pencil Sketch", "type": "LORA", "nsfw": False,
    "modelVersions": [
        {"id": 222, "name": "v2", "baseModel": "Pony", "trainedWords": ["pencil sketch", "graphite"],
         "files": [{"name": "pencil_v2.safetensors", "sizeKB": 147456, "type": "Model", "primary": True,
                    "metadata": {"format": "SafeTensor"}, "downloadUrl": "https://civitai.com/api/download/models/222"}]},
        {"id": 111, "name": "v1", "baseModel": "SD 1.5", "trainedWords": ["pencil"],
         "files": [{"name": "pencil_v1.pt", "sizeKB": 100, "type": "Model", "metadata": {"format": "PickleTensor"}},
                   {"name": "pencil_v1.safetensors", "sizeKB": 1024, "type": "Model", "metadata": {"format": "SafeTensor"},
                    "downloadUrl": "https://civitai.com/api/download/models/111"}]},
    ],
}


class ParseTests(unittest.TestCase):
    def test_links(self):
        self.assertEqual(civitai.parse_ref("https://civitai.com/models/12345/pencil-sketch?modelVersionId=111"), (12345, 111))
        self.assertEqual(civitai.parse_ref("https://civitai.com/models/12345"), (12345, None))
        self.assertEqual(civitai.parse_ref("https://civitai.com/api/download/models/678"), (None, 678))
        self.assertEqual(civitai.parse_ref(" 12345 "), (12345, None))
        with self.assertRaises(ValueError):
            civitai.parse_ref("https://example.com/x")

    def test_families(self):
        self.assertEqual(civitai.family_of("SD 1.5"), ("sd15", None))
        self.assertEqual(civitai.family_of("SDXL 1.0"), ("sdxl", None))
        self.assertEqual(civitai.family_of("Pony"), ("sdxl", "pony"))
        self.assertEqual(civitai.family_of("Illustrious"), ("sdxl", "illustrious"))
        self.assertEqual(civitai.family_of("Flux.1 D"), ("flux", None))
        self.assertEqual(civitai.family_of("Wan Video"), (None, None))

    def test_inspect_picks_version_and_safetensors(self):
        get = lambda url, token=None: LORA_MODEL
        latest = civitai.inspect("https://civitai.com/models/12345", get=get)
        self.assertEqual((latest["version_id"], latest["family"], latest["flavor"], latest["kind"]), (222, "sdxl", "pony", "lora"))
        self.assertEqual(latest["trained_words"], ["pencil sketch", "graphite"])
        self.assertTrue(latest["ok"])
        v1 = civitai.inspect("https://civitai.com/models/12345?modelVersionId=111", get=get)
        self.assertEqual(v1["file"]["name"], "pencil_v1.safetensors")  # never the pickled .pt
        self.assertEqual(v1["family"], "sd15")

    def test_unsupported(self):
        flux_ckpt = {"id": 1, "name": "Flux dev", "type": "Checkpoint",
                     "modelVersions": [{"id": 2, "baseModel": "Flux.1 D", "files": [
                         {"name": "f.safetensors", "type": "Model", "downloadUrl": "u"}]}]}
        info = civitai.inspect("1", get=lambda u, t=None: flux_ckpt)
        self.assertFalse(info["ok"])
        pickled = {"id": 1, "name": "Old", "type": "LORA",
                   "modelVersions": [{"id": 2, "baseModel": "SD 1.5", "files": [{"name": "x.pt", "type": "Model"}]}]}
        self.assertFalse(civitai.inspect("1", get=lambda u, t=None: pickled)["ok"])

    def test_safe_filename(self):
        self.assertEqual(civitai.safe_filename("../../evil/x.safetensors"), "x.safetensors")
        self.assertEqual(civitai.safe_filename('a:b*c?.safetensors'), "a_b_c_.safetensors")
        self.assertEqual(civitai.safe_filename("noext"), "noext.safetensors")


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b"S" * 300_000
    html = False
    need_key = None   # the key this server accepts (None = anyone)
    seen_auth = []    # Authorization headers per path

    def do_GET(self):
        _Handler.seen_auth.append((self.path.split("?")[0], self.headers.get("Authorization")))
        if self.path.startswith("/file") and self.need_key is not None:
            if f"token={self.need_key}" not in self.path:
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(302)  # like Civitai: on to the file server
            self.send_header("Location", "/storage/file")
            self.end_headers()
            return
        if self.html:
            data = b"<html>log in</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            self.send_response(206)
        else:
            self.send_response(200)
        data = self.body[start:]
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/file"
        _Handler.html = False
        _Handler.need_key = None
        _Handler.seen_auth = []

    def tearDown(self):
        self.srv.shutdown()
        self.tmp.cleanup()

    def info(self):
        return {"ok": True, "name": "Pencil", "model_id": 1, "version_id": 2, "trained_words": ["pencil sketch"],
                "file": {"name": "pencil.safetensors", "url": self.url}}

    def test_download_resume_and_sidecar(self):
        part = Path(self.tmp.name, "pencil.safetensors.part")
        part.write_bytes(_Handler.body[:100_000])  # an earlier, interrupted try
        path = civitai.download(self.info(), self.tmp.name)
        self.assertEqual(Path(path).read_bytes(), _Handler.body)
        self.assertFalse(part.exists())
        side = json.loads(Path(self.tmp.name, "pencil.civitai.info").read_text())
        self.assertEqual(side["trainedWords"], ["pencil sketch"])
        with self.assertRaises(ValueError):  # never overwrites
            civitai.download(self.info(), self.tmp.name)

    def test_login_page_is_not_saved(self):
        _Handler.html = True
        with self.assertRaises(ValueError) as cm:
            civitai.download(self.info(), self.tmp.name)
        self.assertIn("API key", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "pencil.safetensors")))

    def test_needs_key_and_key_stays_on_civitai(self):
        _Handler.need_key = "goodkey1234567890"
        with self.assertRaises(ValueError) as cm:
            civitai.download(self.info(), self.tmp.name)
        self.assertIn("signed-in accounts", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            civitai.download(self.info(), self.tmp.name, token="wrongkey1234567890")
        self.assertIn("turned down your API key", str(cm.exception))
        _Handler.seen_auth = []
        path = civitai.download(self.info(), self.tmp.name, token="goodkey1234567890")
        self.assertEqual(Path(path).read_bytes(), _Handler.body)
        auth = dict(_Handler.seen_auth)
        self.assertEqual(auth["/file"], "Bearer goodkey1234567890")
        self.assertIsNone(auth["/storage/file"])  # never passed on to the file server


if __name__ == "__main__":
    unittest.main()


class BlockedConnectionTests(unittest.TestCase):
    """A firewall/VPN blocking Python ('[Errno 9] Bad file descriptor'): retry once through curl, explain if not."""

    def test_red_links(self):
        self.assertEqual(civitai.parse_ref("https://civitai.red/models/123?modelVersionId=45"), (123, 45))
        self.assertEqual(civitai.parse_ref("https://civitai.red/api/download/models/522995?fileId=441187"), (None, 522995))

    def test_blocked_is_recognised(self):
        import errno
        import urllib.error
        import net
        self.assertTrue(net.is_blocked_connection(urllib.error.URLError(OSError(errno.EBADF, "Bad file descriptor"))))
        self.assertTrue(net.is_blocked_connection(OSError(9, "Bad file descriptor")))
        self.assertFalse(net.is_blocked_connection(urllib.error.HTTPError("u", 404, "x", {}, None)))
        self.assertFalse(net.is_blocked_connection(TimeoutError("timed out")))

    def test_retries_through_curl_once(self):
        import urllib.error
        from unittest import mock
        import net
        calls = []

        def fake_open(req, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.URLError(OSError(9, "Bad file descriptor"))
            return "ok"
        with mock.patch("urllib.request.urlopen", fake_open), mock.patch.object(net, "use_curl_for_urllib", return_value=True):
            self.assertEqual(civitai._urlopen(object(), 5), "ok")
        self.assertEqual(len(calls), 2)

    def test_friendly_message_when_still_blocked(self):
        import urllib.error
        from unittest import mock
        import net

        def fake_open(req, timeout=None):
            raise urllib.error.URLError(OSError(9, "Bad file descriptor"))
        with mock.patch("urllib.request.urlopen", fake_open), mock.patch.object(net, "use_curl_for_urllib", return_value=False):
            with self.assertRaises(ValueError) as cm:
                civitai._get_json("https://civitai.com/api/v1/models/1")
        self.assertIn("blocked", str(cm.exception))
        self.assertIn("Retry", str(cm.exception))
