"""Real-hardware self-test for Merge LoRAs and embedding training (run with the app's own Python, app closed).
Writes a report to selftest_train.log next to the project and removes everything it created."""
import json, os, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LOG = HERE.parent / "selftest_train.log"
out = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()

log("== unit tests (merge maths)")
r = subprocess.run([sys.executable, "-m", "unittest", "tests.test_lora_merge", "-v"], cwd=HERE, capture_output=True, text=True)
log(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.stdout[-300:])

p = subprocess.Popen([sys.executable, str(HERE / "bee_backend.py")], cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)
msgs, errs = [], []
threading.Thread(target=lambda: [msgs.append(json.loads(l)) for l in p.stdout if l.startswith("{")], daemon=True).start()
threading.Thread(target=lambda: [errs.append(l) for l in p.stderr], daemon=True).start()
mid = [0]
def call(cmd, timeout=1800, **args):
    mid[0] += 1; i = mid[0]
    p.stdin.write(json.dumps({"id": i, "cmd": cmd, "args": args}) + "\n"); p.stdin.flush()
    end = time.time() + timeout; last = 0
    while time.time() < end:
        for m in list(msgs):
            if m.get("id") == i and m.get("type") in ("result", "error"):
                return m
        prog = [m for m in msgs if m.get("type") == "train_status"]
        if prog and len(prog) != last:
            last = len(prog); log("   ", {k: prog[-1][k] for k in ("step", "steps", "loss", "sec_per_step")})
        time.sleep(0.5)
    return {"type": "error", "msg": "timed out"}

created = []
try:
    time.sleep(3)
    models = call("list_models")["data"]
    sd15 = [m for m in models if m["family"] == "sd15" and (m.get("cached") or m["id"].startswith(("file:", "tdict:", "local:")))]
    log("SD 1.5 models here:", [m["id"] for m in sd15][:5])
    all_l = [l for l in call("list_loras")["data"] if str(l.get("file", "")).endswith(".safetensors")]
    fam = next((f for f in ("sd15", "sdxl", "flux", "ernie") if sum(l.get("family") == f for l in all_l) >= 2), None)
    loras = sorted([l for l in all_l if l.get("family") == fam], key=lambda l: l.get("size_mb", 999))
    log(".safetensors LoRAs by type:", {f: sum(l.get("family") == f for l in all_l) for f in ("sd15", "sdxl", "flux", "ernie")},
        "→ merging", fam)

    log("== merge (exact, then smaller)")
    if len(loras) >= 2:
        two = [{"name": loras[0]["name"], "weight": 0.8}, {"name": loras[1]["name"], "weight": 0.6}]
        for mode in ("exact", "smaller"):
            t = time.time()
            r = call("lora_merge", items=two, name=f"pa_selftest_{mode}", mode=mode, rank=16, timeout=600)
            log(mode, r.get("type"), json.dumps(r.get("data") or r.get("msg"))[:300], f"{time.time() - t:.1f}s")
            if r.get("type") == "result":
                created.append(r["data"]["path"])
        for f in list(created):
            listed = [l for l in call("list_loras")["data"] if l.get("file") == f]
            log("merged file listed as:", [(l["name"], l.get("family")) for l in listed])
        if fam == "sd15" and sd15 and created:
            call("list_loras")
            t = time.time()
            r = call("generate", model=sd15[0]["id"], prompt="a cat, ink drawing", steps=6, width=384, height=384, seed=7,
                     loras=[{"name": "pa_selftest_exact", "weight": 1.0}], num_images=1, timeout=900)
            log("generate with merged LoRA:", r.get("type"), (r.get("msg") or "")[:200], f"{time.time() - t:.0f}s")
            if r.get("type") == "result":
                created += [x.get("path") for x in r["data"] if isinstance(x, dict) and x.get("path")]
                ids = [x.get("id") for x in r["data"] if isinstance(x, dict) and x.get("id") is not None]
                if ids: call("library_delete", ids=ids)
    else:
        log("fewer than 2 .safetensors LoRAs of one type: merge skipped")

    log("== embedding training (50 steps)")
    lib = HERE.parent / ".selftest_pics"
    if sd15:
        imgs_dir = Path.home() / ".protoartstudio" / "images"
        pics = sorted([q for q in imgs_dir.rglob("*") if q.suffix.lower() in (".png", ".jpg", ".jpeg")])[:5]
        tmpd = Path(tempfile.mkdtemp(prefix="pa_ti_"))
        for q in pics: shutil.copy(q, tmpd / q.name)
        log("pictures:", len(pics))
        call("train_scan", folder=str(tmpd))
        t = time.time()
        r = call("train_embedding", model=sd15[0]["id"], folder=str(tmpd), name="pa_selftest_emb", init_word="cat",
                 kind="object", vectors=2, steps=50, timeout=3600)
        log("train_embedding:", r.get("type"), json.dumps(r.get("data") or r.get("msg"))[:300], f"{time.time() - t:.0f}s")
        if r.get("type") == "result":
            created.append(r["data"]["path"])
            embs = call("list_embeddings")["data"]
            log("listed in embeddings:", any(e.get("name") == "pa_selftest_emb" for e in embs),
                [e for e in embs if e.get("name") == "pa_selftest_emb"][:1])
            t = time.time()
            r = call("generate", model=sd15[0]["id"], prompt="a portrait of pa_selftest_emb", steps=6, width=384, height=384,
                     seed=7, num_images=1, timeout=900)
            log("generate with the embedding:", r.get("type"), (r.get("msg") or "")[:200], f"{time.time() - t:.0f}s")
            if r.get("type") == "result":
                ids = [x.get("id") for x in r["data"] if isinstance(x, dict) and x.get("id") is not None]
                if ids: call("library_delete", ids=ids)
        shutil.rmtree(tmpd, ignore_errors=True)
finally:
    for f in created:
        try:
            if f and os.path.exists(f) and "pa_selftest" in f: os.remove(f); log("removed", f)
        except OSError as e: log("couldn't remove", f, e)
    try: p.stdin.close(); p.wait(timeout=20)
    except Exception: p.kill()
    tail = [l for l in errs if "Traceback" in l or "Error" in l][-8:]
    log("== backend errors (last):", "".join(tail) or "none")
    log("== done")
