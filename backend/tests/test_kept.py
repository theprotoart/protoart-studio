"""⚡ Models kept on this Mac: copies named like the fast engine's, found by repo, removed only if the app made them."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kept  # noqa: E402


def test_keep_find_and_remove():
    home = Path(tempfile.mkdtemp())
    old = os.environ.get("PROTOART_HOME")
    real_usage, real_snapshot = kept.shutil.disk_usage, kept.snapshot
    os.environ["PROTOART_HOME"] = str(home)
    try:
        snap = home / "hub" / "models--org--pic-model" / "snapshots" / "abc123"
        (snap / "unet").mkdir(parents=True)
        (snap / "unet" / "w.safetensors").write_bytes(b"x" * 1000)
        kept.snapshot = lambda repo, revision=None: snap  # (no Hugging Face cache in the test)
        assert kept.dest_for(snap).name == "models--org--pic-model--abc123"
        assert kept.kept_path("org/pic-model") is None
        kept.shutil.disk_usage = lambda p: type("U", (), {"free": 100 << 30})()
        dest = kept.keep("org/pic-model")
        assert (dest / "unet" / "w.safetensors").read_bytes() == b"x" * 1000
        assert kept.kept_path("org/pic-model") == dest
        (home / "fast-copies" / "models--org--pic-model--mine").mkdir()  # not made by the app: left alone
        kept.remove("org/pic-model")
        assert kept.kept_path("org/pic-model") is None and (home / "fast-copies" / "models--org--pic-model--mine").exists()
        assert (snap / "unet" / "w.safetensors").exists()  # the original is untouched
    finally:
        kept.shutil.disk_usage, kept.snapshot = real_usage, real_snapshot
        if old is None:
            os.environ.pop("PROTOART_HOME", None)
        else:
            os.environ["PROTOART_HOME"] = old
