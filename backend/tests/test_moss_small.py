"""MOSS-Image's 8 GB version: the recipe the worker uses and the loader rebuilds must agree, and the app must pick it
on small Macs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moss_small  # noqa: E402


def test_recipe():
    spec = moss_small.small_spec(64, ["video_embedder"], 30)
    b = moss_small.bits_for
    assert b(spec, "layers.5.proj.ffn_up_gate") == 4
    assert b(spec, "layers.5.proj.ffn_down") == 4
    assert b(spec, "layers.5.proj.q") == 8 and b(spec, "layers.5.proj.o") == 8
    assert b(spec, "layers.0.proj.ffn_down") == 8 and b(spec, "layers.29.proj.ffn_down") == 8
    assert b(spec, "layers.10.proj.ffn_down") == 4 and b(spec, "layers.2.proj.ffn_down") == 4  # not "layers.0."/"layers.29."
    assert b(spec, "text_embedder.blocks.0.qkv") == 8
    assert spec["skip"] == ["video_embedder"] and spec["mode"] == "affine"


def test_catalog():
    import engine
    m = engine.all_models()
    for k in ("moss_turbo_mlx", "moss_mlx", "zimage_turbo_mlx_q4"):
        assert m[k]["min_ram_gb"] == 8, k
    src = (Path(engine.__file__).parent / "engine.py").read_text()
    assert '"cmd": "shrink"' in src and "student_bf16" in src
    w = (Path(engine.__file__).parent / "moss_worker.py").read_text()
    assert 'cmd == "shrink"' in w and "mx.set_default_device(previous)" in w


if __name__ == "__main__":
    test_recipe()
    test_catalog()
    print("ok")


def test_ready_made_copied_from_external_drive():
    import tempfile
    import moss_engine
    tmp_path = Path(tempfile.mkdtemp()).resolve()  # macOS: /var → /private/var
    ext = tmp_path / "ext" / "snap"
    (ext / "transformer").mkdir(parents=True)
    blob = tmp_path / "ext" / "blob"
    blob.write_bytes(b"x" * 10)
    (ext / "transformer" / "model.safetensors").symlink_to(blob)  # HF snapshots are links to blobs
    old = moss_engine._HOME, moss_engine.SLOW_PREFIX
    moss_engine._HOME, moss_engine.SLOW_PREFIX = tmp_path / "home", str(tmp_path / "ext")
    try:
        out = Path(moss_engine.ready_made_local(ext, free_gb=100))
        f = out / "transformer" / "model.safetensors"
        assert out.parent.name == "fast-copies" and f.read_bytes() == b"x" * 10 and not f.is_symlink()
        assert blob.exists()  # the download itself is left alone
        assert moss_engine.ready_made_local(ext, free_gb=100) == str(out)  # once
        assert moss_engine.ready_made_local(ext, free_gb=0) == str(out)
        moss_engine.SLOW_PREFIX = "/Volumes/"
        assert moss_engine.ready_made_local(ext) == str(ext)  # on this Mac's drive already: used where it is
    finally:
        moss_engine._HOME, moss_engine.SLOW_PREFIX = old


def test_ready_made_pinned():
    import moss_engine
    assert moss_engine.SMALL_REPO == "brushpenbob/moss-image-sft-mlx-8gb"
    assert len(moss_engine.SMALL_REVISION) == 40


def test_enhance_frees_the_fast_engine_on_small_macs():
    """Z-Image's fast-engine worker (~6 GB) must count as a loaded model, or Enhance's 4B writer swaps next to it."""
    from types import SimpleNamespace
    import engine

    class P:
        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    held = engine.Engine._holds_model
    e = SimpleNamespace(pipe=None, mlx=None, moss=None)
    assert not held(e)
    e.mlx = SimpleNamespace(proc=P(None))  # running
    assert held(e)
    e.mlx = SimpleNamespace(proc=P(0))  # exited
    assert not held(e)
    e.moss = SimpleNamespace(proc=P(None))
    assert held(e)
