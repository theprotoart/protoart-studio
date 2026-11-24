"""🧊 Make 3D worker (a short-lived process; see idol3d.py): Idol turns a picture into a 3D Gaussian splat, saved as a
standard .ply, plus a preview picture and the picture Idol saw (background removed, framed).
in: argv[1] = JSON {image, out, weights, dinov3, vae, mode, steps, gaussians, seed}
out (stdout, JSON lines): {"type": "status"|"progress"|"done"|"error", ...}
"""
import json
import os
import sys
import time
import traceback


def say(**m):
    print(json.dumps(m), flush=True)


def main():
    a = json.loads(sys.argv[1])
    # Idol's code and plyfile + the tensordict stand-in, after the app's own libraries (its torch, transformers…)
    sys.path.append(os.environ["PROTOART_IDOL_SITE"])
    sys.path.insert(0, os.environ["PROTOART_IDOL_CODE"])
    import numpy as np
    import torch
    from idol import IdolPipeline
    from PIL import Image

    import idol3d
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.time()
    say(type="status", msg="Make 3D: loading Idol…")
    pipe = IdolPipeline(a["weights"], device=device, optimize=False,
                        image_encoders={"dinov3": a["dinov3"], "flux2_vae": a["vae"]})  # a folder with the VAE in vae/
    steps, cfg = int(a["steps"]), 3.0
    calls = {"n": 0}
    per_step = 2 if cfg > 0 else 1  # the guided and the unguided pass
    forward = pipe.denoiser.forward

    def counted(*args, **kw):
        out = forward(*args, **kw)
        calls["n"] += 1
        if calls["n"] % per_step == 0:
            say(type="progress", step=calls["n"] // per_step, steps=steps, image=1, images=1)
        return out

    pipe.denoiser.forward = counted
    say(type="status", msg="Make 3D: finding the object and building it in 3D…")
    out = pipe(a["image"], mode=a.get("mode") or "synthetic", seed=int(a["seed"]), steps=steps, cfg_strength=cfg,
               num_gaussians=int(a["gaussians"]))
    say(type="status", msg="Make 3D: saving…")
    ply = a["out"] + ".ply"
    out.gaussian.save_ply(ply)
    seen = a["out"] + "_input.png"
    out.preprocessed_image.save(seen)
    g = out.gaussian
    xyz = g.xyz.detach().float().cpu().numpy()
    xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], 1)  # z-up -> y-up (as in the .ply)
    rgb = np.clip(0.5 + 0.28209479177387814 * g.features_dc[:, 0, :].detach().float().cpu().numpy(), 0, 1)
    alpha = g.opacity[:, 0].detach().float().cpu().numpy()
    scale = g.scaling.detach().float().cpu().numpy().max(1)
    prev = a["out"] + "_preview.png"
    idol3d.render_preview(xyz, rgb, alpha, scale).save(prev)
    say(type="done", ply=ply, preview=prev, input=seen, gaussians=int(xyz.shape[0]), camera=out.camera,
        seconds=round(time.time() - started, 1))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        msg = str(e) or e.__class__.__name__
        if "out of memory" in msg.lower() or "MPS backend out of memory" in msg:
            msg = "This Mac ran out of memory. Close other apps, or pick ⚡ Quick, and try again."
        say(type="error", msg=msg)
        sys.exit(1)
