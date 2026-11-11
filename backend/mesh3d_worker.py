"""🧊 Make 3D worker (a short-lived process; see mesh3d.py): Hunyuan3D 2 mini turns a cut-out picture into a mesh,
cleaned up (floating bits and broken faces removed, fewer faces), saved as .glb and .obj, plus a shaded preview picture.
in: argv[1] = JSON {image, out, repo, subfolder, steps, octree, guidance, seed, faces}
out (stdout, JSON lines): {"type": "status"|"progress"|"done"|"error", ...}
"""
import json
import os
import sys
import traceback


def say(**m):
    print(json.dumps(m), flush=True)


def preview(mesh, size=768):
    """A shaded picture of the mesh, turned a little to the side (drawn here: no 3D library needed)."""
    import numpy as np
    from PIL import Image, ImageDraw
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    v = v - (v.max(0) + v.min(0)) / 2
    yaw, pitch = np.radians(-30), np.radians(12)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    v = v @ ry.T @ rx.T
    scale = 0.85 * size / max(1e-6, (v[:, :2].max(0) - v[:, :2].min(0)).max())
    xy = np.stack([size / 2 + v[:, 0] * scale, size / 2 - v[:, 1] * scale], 1)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([-0.4, 0.6, 0.7]); light /= np.linalg.norm(light)
    shade = np.clip(np.abs(n @ light), 0, 1) * 0.75 + 0.2
    order = np.argsort(tri[:, :, 2].mean(1))  # far faces first
    img = Image.new("RGB", (size, size), (46, 48, 52))
    d = ImageDraw.Draw(img)
    for i in order:
        g = int(shade[i] * 230)
        d.polygon([tuple(p) for p in xy[f[i]]], fill=(g, g, int(g * 0.97)))
    return img


def main():
    a = json.loads(sys.argv[1])
    sys.path.insert(0, os.environ["PROTOART_MESH3D_CODE"])
    sys.path.append(os.environ["PROTOART_MESH3D_SITE"])  # after the app's own libraries
    try:
        import torch
        from PIL import Image
        say(type="status", msg="Make 3D: loading Hunyuan3D 2 mini…")
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(a["repo"], subfolder=a["subfolder"], use_safetensors=True,
                                                                device=device, dtype=torch.float32)
        image = Image.open(a["image"]).convert("RGBA")
        steps = int(a.get("steps") or 30)

        def cb(i, t, out):
            say(type="progress", image=1, images=1, step=min(i + 1, steps), steps=steps + 2)

        say(type="status", msg="Make 3D: shaping…")
        with torch.no_grad():
            mesh = pipe(image=image, num_inference_steps=steps, octree_resolution=int(a.get("octree") or 380),
                        guidance_scale=float(a.get("guidance") or 5.5), num_chunks=4000,
                        generator=torch.Generator().manual_seed(int(a.get("seed") or 0)), output_type="trimesh",
                        callback=cb, callback_steps=1, enable_pbar=False)[0]
        del pipe
        say(type="progress", image=1, images=1, step=steps + 1, steps=steps + 2)
        say(type="status", msg="Make 3D: cleaning up the mesh…")
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        faces = int(a.get("faces") or 40000)
        if faces > 0 and len(mesh.faces) > faces:
            mesh = FaceReducer()(mesh, max_facenum=faces)
        out = a["out"]
        mesh.export(out + ".glb")
        mesh.export(out + ".obj")
        preview(mesh).save(out + "_preview.png")
        say(type="progress", image=1, images=1, step=steps + 2, steps=steps + 2)
        say(type="done", glb=out + ".glb", obj=out + ".obj", preview=out + "_preview.png",
            vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)))
    except Exception as e:
        traceback.print_exc()
        say(type="error", msg=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
