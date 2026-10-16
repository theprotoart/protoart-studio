"""Enhance & upscale speed test on this Mac: Turbo vs Fast vs Standard, same picture, same seed.
Run with "Enhance Speed Test.command". Results: enhance_bench.log and enhance_bench/*.png (to compare quality)."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "enhance_bench")
os.makedirs(OUT, exist_ok=True)


def out(*a):
    print(*a, flush=True)


import platform  # noqa: E402
import subprocess  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from postprocess import finegrain  # noqa: E402

try:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
except Exception:
    chip = platform.machine()
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"({e})"


def memory():
    swap = sh("sysctl -n vm.swapusage")
    free = sh("memory_pressure -Q 2>/dev/null | tail -1")
    return f"swap {swap} · {free}"


out(f"Mac: {chip} · {finegrain._ram_gb():.0f} GB RAM · torch {torch.__version__}")

# a real picture if there is one, else a generated test pattern
src = None
for cand in (os.path.join(ROOT, "ProtoArtBeautify", "test_input.png"),):
    if os.path.exists(cand):
        src = cand
if src:
    img = Image.open(src).convert("RGB")
else:
    import numpy as np
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:800, 0:640]
    base = np.stack([120 + 80 * np.sin(x / 90), 110 + 60 * np.cos(y / 70), 140 + 50 * np.sin((x + y) / 120)], -1)
    img = Image.fromarray(np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype("uint8"))
out("Memory before:", memory())
out("Biggest apps now (MB):")
for line in sh("ps -Ao rss=,comm= -m | head -8").splitlines():
    kb, _, name = line.strip().partition(" ")
    out(f"   {int(kb) // 1024:6d}  {os.path.basename(name)[:50]}")
out(f"Picture: {img.width}x{img.height}")

t0 = time.time()
enh = finegrain.FinegrainEnhancer("mps" if torch.backends.mps.is_available() else "cpu")
out(f"Load: {time.time() - t0:.0f}s · detail LoRA layers merged into the model: {enh.merged} · "
    f"sliced attention layers: {getattr(enh, 'sliced', '?')} · memory profile: {getattr(enh, 'profile', {}).get('name')}")
out("Memory after load:", memory())

import threading  # noqa: E402
LIMIT = 15 * 60  # give up on a speed after 15 minutes (that alone answers the question)

runs = [
    ("INSTANT", dict(instant=True)),
    ("TURBO", dict(turbo=True, num_inference_steps=6)),
    ("FAST", dict(num_inference_steps=10, solver="DPMSolver")),
    ("STANDARD", dict(num_inference_steps=18, solver="DDIM")),
]
for name, opts in runs:
    t = time.time()
    last = [0.0]
    stop = threading.Event()
    timer = threading.Timer(LIMIT, stop.set)
    timer.daemon = True
    timer.start()

    def prog(done, total):
        now = time.time()
        if now - last[0] > 30:
            last[0] = now
            out(f"   {name}: {done}/{total} tile-steps · {now - t:.0f}s · {memory()}")
    try:
        res, used = enh.enhance(img, seed=42, on_progress=prog, cancel_event=stop, upscale_factor=2, denoise_strength=0.35,
                                controlnet_scale=0.6, **opts)
        secs = time.time() - t
        res.save(os.path.join(OUT, f"{name}.png"))
        out(f"{name:9s} {res.width}x{res.height}: {secs:.0f}s ({used.get('num_inference_steps')} steps, {used.get('solver')}, worked at {used.get('work_size')} px, "
            f"tiles {used.get('tile_width')}x{used.get('tile_height')} latent)")
    except finegrain.EnhanceCancelled:
        out(f"{name:9s} STOPPED after {LIMIT // 60} min (too slow) · {memory()}")
    except Exception as e:
        import traceback
        out(f"{name:9s} FAILED: {type(e).__name__}: {e}")
        out(traceback.format_exc()[-1500:])
    finally:
        timer.cancel()
out("Memory at the end:", memory())
out("ENHANCE BENCH DONE")
