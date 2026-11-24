"""A download cut short by the network is tried again, after its partial files are cleared (only that model's)."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine  # noqa: E402


def test_retry_after_cut_short():
    t = Path(tempfile.mkdtemp(prefix="protoart_test_"))  # the system temp folder (it used to be left in your home folder)
    blobs = t / "models--a--b" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "x.incomplete").write_text("1")
    (blobs / "done").write_text("1")
    other = t / "models--c--d" / "blobs"
    other.mkdir(parents=True)
    (other / "y.incomplete").write_text("1")
    hub, sleep = engine._hub_dir, engine.time.sleep
    engine._hub_dir, engine.time.sleep = (lambda: t), (lambda s: None)
    try:
        calls = []

        def job():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("Consistency check failed: file should be of size 164654177 but has size 132051692")
            return "ok"

        assert engine._retry_download(job, ["a/b"]) == "ok" and len(calls) == 2
        assert sorted(p.name for p in blobs.iterdir()) == ["done"]
        assert (other / "y.incomplete").exists()  # another model's partial download is left alone

        def bad():
            raise ValueError("401 Unauthorized")
        try:
            engine._retry_download(bad, ["a/b"])
            raise AssertionError("not a network hiccup: should not retry")
        except ValueError:
            pass
    finally:
        engine._hub_dir, engine.time.sleep = hub, sleep
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    test_retry_after_cut_short()
    print("ok")
