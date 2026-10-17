"""Picture reader worker (runs in ~/.protoartstudio/venv-vlm with mlx-vlm): reads one JSON request on stdin
({repo, revision, image, prompt, max_tokens}), writes JSON lines on stdout ({"type": "status"|"done"|"error"}), exits.
"""
import json
import sys


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    req = json.loads(sys.stdin.readline())
    try:
        from huggingface_hub import snapshot_download
        try:
            path = snapshot_download(req["repo"], revision=req.get("revision"), local_files_only=True)
        except Exception:
            send({"type": "status", "msg": "Downloading the picture reader (first time only)…"})
            path = snapshot_download(req["repo"], revision=req.get("revision"))
        send({"type": "status", "msg": "Reading the picture…"})
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor = load(path)
        prompt = apply_chat_template(processor, model.config, req["prompt"], num_images=1)
        out = generate(model, processor, prompt, image=[req["image"]], max_tokens=int(req.get("max_tokens") or 320),
                       temperature=0.0, verbose=False)
        text = getattr(out, "text", out)
        send({"type": "done", "text": str(text)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        send({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
