"""Publish MOSS-Image's 8 GB version (made by the app on this Mac) to Protoart Studio's Hugging Face page, so other
8 GB Macs can download it ready-made. Run by "Upload MOSS 8GB version.command"; you paste a Hugging Face *write*
token when asked (it's used for this upload only and never saved).

Everything uploaded is Apache-2.0 (MOSS-Image by OpenMOSS; the Qwen3.5-9B text encoder; the FLUX.2 small VAE decoder;
the MLX conversion by felix-ab / VisualInference). Its LICENSE and NOTICE go with it, and the model card says what was
changed (the picture model re-stored at 4-bit with 8-bit attention and end blocks), as the licence asks.
"""
import getpass
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

REPO = os.environ.get("MOSS_SMALL_REPO", "brushpenbob/moss-image-sft-mlx-8gb")

CARD = """---
license: apache-2.0
library_name: moss-image-mlx
base_model: OpenMOSS-Team/MOSS-Image-SFT
tags: [mlx, text-to-image, apple-silicon, moss-image, protoart-studio]
---

# MOSS-Image SFT · MLX · 8 GB Macs

A smaller copy of [VisualInference/MOSS-Image-SFT-MLX-compact](https://huggingface.co/VisualInference/MOSS-Image-SFT-MLX-compact)
for Macs with 8 GB of memory, made for and used by Protoart Studio. Runs with
[moss-image-mlx](https://github.com/felix-ab/moss-image-mlx).

**What was changed** (Apache-2.0 §4b): only the picture model (`transformer/`). It was re-stored from 8-bit to
4-bit (group 64) except attention (q, k, v, o), the text embedder and the first and last blocks, which stay 8-bit
(~5.3 GB → {gb} GB). The layers kept in bf16 by the original are unchanged. The text encoder, tokenizer, scheduler and
VAE decoder are the compact tier's files, unchanged. Quantization spec: in `transformer/config.json`.

Expect slightly less fine detail than the 8-bit original; lettering is decided by the unchanged text encoder and the
8-bit attention.

## Credits and licences (all Apache-2.0)
- **MOSS-Image** © OpenMOSS: [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
- **Text encoder:** Qwen3.5-9B © Alibaba Qwen
- **VAE:** FLUX.2 with the FLUX.2 small decoder © Black Forest Labs
- **MLX port and compact tier:** felix-ab / VisualInference ([moss-image-mlx](https://github.com/felix-ab/moss-image-mlx))

See `LICENSE` and `NOTICE`. Not affiliated with OpenMOSS.
"""


def main():
    import moss_engine
    from huggingface_hub import HfApi
    d = moss_engine.small_dir()
    if not moss_engine.small_ready(d):
        sys.exit("MOSS-Image's 8 GB version isn't on this Mac yet: make one picture with MOSS-Image in the app first.")
    info = json.loads((d / ".protoart-done").read_text())
    stage = d.with_name(d.name + ".upload")
    shutil.rmtree(stage, ignore_errors=True)
    # real files (hard links where possible, so nothing is copied): the 8 GB version shares most files with the download
    for root, _dirs, files in os.walk(d, followlinks=True):
        for f in files:
            if f.startswith("."):
                continue
            src = Path(root) / f
            dst = stage / src.relative_to(d)
            dst.parent.mkdir(parents=True, exist_ok=True)
            real = src.resolve()
            try:
                os.link(real, dst)
            except OSError:
                shutil.copy2(real, dst)
    (stage / "README.md").write_text(CARD.replace("{gb}", str(info.get("transformer_gb", "~3.5"))))
    for need in ("LICENSE", "NOTICE"):
        if not (stage / need).exists():
            sys.exit(f"{need} is missing from the download, so it can't be shared: stopping.")
    total = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 2**30
    print(f"Ready to upload {total:.1f} GB to https://huggingface.co/{REPO} (public).")
    token = getpass.getpass("Paste a Hugging Face WRITE token (hidden; not saved), or press Return to stop: ").strip()
    if not token:
        sys.exit("Stopped: nothing uploaded.")
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    print("Uploading (this takes a while; if it stops, run this again: files already sent are skipped)…")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=str(stage))
    else:  # newer huggingface_hub: upload_folder sends in chunks and skips what the Hub already has
        api.upload_folder(repo_id=REPO, repo_type="model", folder_path=str(stage),
                          commit_message="MOSS-Image SFT 8 GB version (Protoart Studio)")
    sha = api.model_info(REPO).sha
    shutil.rmtree(stage, ignore_errors=True)
    print(f"\nDone. Commit: {sha}\nSend this commit to Claude to switch the app to the ready-made copy.")


if __name__ == "__main__":
    main()
