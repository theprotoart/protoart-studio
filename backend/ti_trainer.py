"""Embedding training (textual inversion) for SD 1.5 models (Train tab → Embedding).

Teaches the model one new word from 3-30 pictures: only a few word-vectors are learned (the model itself is not
changed), so the result is a tiny file (a few KB) that works in any prompt, like EasyNegative does.

Designed for 8 GB Macs, like the LoRA trainer:
  * pictures are encoded once by the VAE, which is then freed;
  * the UNet stays frozen in float16 with gradient checkpointing and attention slicing;
  * only the new word's vectors are trained (in float32); the rest of the vocabulary is put back after every step.
Saved as .safetensors ("emb_params", the A1111 layout) in the app's embeddings folder.
"""
import math
import random
import re
import time
from pathlib import Path

import trainer

OBJECT_TEMPLATES = [
    "a photo of a {}", "a rendering of a {}", "a cropped photo of the {}", "the photo of a {}", "a photo of a clean {}",
    "a close-up photo of a {}", "a bright photo of the {}", "a good photo of the {}", "a picture of a {}",
    "an illustration of a {}", "a photo of one {}", "a dark photo of the {}", "a photo of the large {}",
]
STYLE_TEMPLATES = [
    "a painting in the style of {}", "a rendering in the style of {}", "a picture in the style of {}",
    "an illustration in the style of {}", "a drawing in the style of {}", "artwork in the style of {}",
    "a close-up painting in the style of {}", "a cropped picture in the style of {}", "a good painting in the style of {}",
]


class TIError(Exception):
    pass


def _new_words_module():
    import torch

    class NewWords(torch.nn.Module):
        """The text encoder's word table with the new word's vectors swapped in from a small trainable parameter."""

        def __init__(self, base, ids, init):
            super().__init__()
            self.base = base
            self.ids = list(ids)
            self.vec = torch.nn.Parameter(init.clone())

        @property
        def weight(self):
            return self.base.weight

        def forward(self, input_ids):
            e = self.base(input_ids)
            for k, i in enumerate(self.ids):
                e = torch.where((input_ids == i).unsqueeze(-1), self.vec[k].to(e.dtype), e)
            return e
    return NewWords


def _NewWords(base, ids, init):
    return _new_words_module()(base, ids, init)


def safe_name(name):
    n = re.sub(r"[^\w\-]+", "_", (name or "").strip()).strip("_-")
    return n[:48] or "my_embedding"


def save_embedding(path, vectors, meta):
    import torch
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    save_file({"emb_params": vectors.detach().to("cpu", torch.float32).contiguous()}, str(tmp),
              metadata={k: str(v) for k, v in meta.items()})
    tmp.replace(path)
    return str(path)


def train(engine, args, emit, cancel_event, out_dir):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps

    items = trainer.scan_dataset(args.get("folder"))
    if len(items) < 3:
        raise TIError("Use at least 3 pictures (5-20 is a good amount).")
    name = safe_name(args.get("name"))
    init_word = (args.get("init_word") or "").strip().split(" ")[0] or ("art" if args.get("kind") == "style" else "person")
    style = args.get("kind") == "style"
    n_vec = max(1, min(8, int(args.get("vectors") or 2)))
    steps = max(50, min(10000, int(args.get("steps") or 1500)))
    lr = float(args.get("lr") or 5e-3)
    save_every = int(args.get("save_every") or 0)
    res = 512
    flip = bool(args.get("flip", True))
    out_dir = Path(out_dir)
    final_path = out_dir / f"{name}.safetensors"
    if final_path.exists() and not args.get("overwrite"):
        raise TIError(f"An embedding called '{name}' already exists. Pick another name.")

    from engine import all_models
    model_id = args.get("model")
    spec = all_models().get(model_id)
    if not spec or spec["family"] != "sd15" or spec.get("engine"):
        raise TIError("Embedding training works with SD 1.5 models for now (SDXL needs a 16 GB+ Mac and comes later).")
    emit({"type": "status", "msg": f"Loading {spec['label']} for embedding training…"})
    engine.load(model_id, low_memory=False)
    engine.apply_loras([])
    pipe = engine.pipe
    device = engine.device()
    wdtype = torch.float16 if device == "mps" else torch.float32
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    from diffusers import DDPMScheduler
    noise_sched = DDPMScheduler.from_config(engine.base_scheduler_config)
    pred_type = noise_sched.config.get("prediction_type", "epsilon")

    # 1. the new word: n placeholder tokens, started from the starting word's meaning
    tokens = [f"<{name}>"] + [f"<{name}_{i}>" for i in range(1, n_vec)]
    added = tok.add_tokens(tokens)
    if added != n_vec:
        raise TIError(f"'{name}' clashes with a word the model already knows; pick another name.")
    new_ids = tok.convert_tokens_to_ids(tokens)
    init_ids = tok.encode(init_word, add_special_tokens=False) or tok.encode("art", add_special_tokens=False)
    te.resize_token_embeddings(len(tok))
    emb = te.get_input_embeddings()
    with torch.no_grad():
        start = emb.weight[init_ids].mean(0)
        for k, i in enumerate(new_ids):
            emb.weight[i] = start + (0.0 if k == 0 else 0.01) * torch.randn_like(start)
    word = " ".join(tokens)

    try:
        # 2. encode the pictures once (and mirrored copies)
        lat_mean, lat_std = [], []
        vae.to(device, wdtype)
        with torch.no_grad():
            for i, it in enumerate(items):
                if cancel_event.is_set():
                    raise trainer._Cancelled()
                emit({"type": "status", "msg": f"Preparing pictures {i + 1}/{len(items)}…"})
                try:
                    img = trainer._open_rgb(it["path"])
                except Exception:
                    continue
                img = trainer.square(img, res, it.get("fit"))
                for v in ([img, ImageOps.mirror(img)] if flip else [img]):
                    x = torch.from_numpy(np.asarray(v, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
                    dist = vae.encode(x.to(device, wdtype)).latent_dist
                    lat_mean.append(dist.mean.squeeze(0).to("cpu", torch.float16))
                    lat_std.append(dist.std.squeeze(0).to("cpu", torch.float16))
        if not lat_mean:
            raise TIError("None of the pictures could be opened.")
        scaling = vae.config.scaling_factor
        pipe.vae = None
        vae.to("cpu")
        del vae
        engine.release_memory()

        # 3. train only the new word's vectors
        unet.requires_grad_(False)
        unet.to(device, wdtype)
        try:
            unet.enable_gradient_checkpointing()
        except Exception:
            pass
        try:
            unet.set_attention_slice("auto")
        except Exception:
            pass
        unet.train()
        # only the new word's vectors are a trainable parameter (a few KB); the vocabulary stays frozen, so there are
        # no gradients or optimiser state for its 49,000 other words (saves ~600 MB on an 8 GB Mac)
        te.requires_grad_(False)
        base_emb = te.get_input_embeddings()
        words = _NewWords(base_emb, new_ids, base_emb.weight[new_ids].detach().float())
        te.text_model.embeddings.token_embedding = words
        te.to(device, wdtype)
        words.vec.data = words.vec.data.to(device, torch.float32)
        te.train()
        opt = torch.optim.AdamW([words.vec], lr=lr, weight_decay=0.0)
        warmup = max(1, int(steps * 0.03))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))) * 0.9 + 0.1))
        scaler = None
        if wdtype == torch.float16:  # the UNet runs in half precision: scale the loss so small gradients survive
            try:
                scaler = torch.amp.GradScaler(device)
            except Exception:
                scaler = None
        templates = STYLE_TEMPLATES if style else OBJECT_TEMPLATES
        meta = {"name": name, "base_model": spec["label"], "family": "sd15", "vectors": n_vec, "init_word": init_word,
                "kind": "style" if style else "object", "n_images": len(items), "trainer": "Protoart Studio"}

        def vectors():
            return words.vec.detach().clone()

        n = len(lat_mean)
        order, losses, saved = [], [], []
        started = time.time()
        emit({"type": "status", "msg": f"Training the embedding {name}: {steps} steps on {len(items)} pictures…"})
        try:
            for step in range(steps):
                if hasattr(engine, "wait_if_paused"):
                    engine.wait_if_paused()
                if cancel_event.is_set():
                    raise trainer._Cancelled()
                if not order:
                    order = list(range(n))
                    random.shuffle(order)
                idx = order.pop()
                lat = (lat_mean[idx].float() + lat_std[idx].float() * torch.randn_like(lat_mean[idx].float()))
                lat = (lat * scaling).unsqueeze(0).to(device, wdtype)
                noise = torch.randn_like(lat)
                ts = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device).long()
                noisy = noise_sched.add_noise(lat, noise, ts)
                ids = tok(random.choice(templates).format(word), padding="max_length", max_length=tok.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to(device)
                cond = te(ids)[0].to(wdtype)
                target = noise_sched.get_velocity(lat, noise, ts) if pred_type == "v_prediction" else noise
                pred = unet(noisy, ts, cond).sample
                loss = F.mse_loss(pred.float(), target.float())
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                if scaler is not None:
                    try:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    except (RuntimeError, NotImplementedError) as e:
                        print("[ti] grad scaler unavailable, continuing without it:", e, flush=True)
                        scaler = None
                        opt.zero_grad(set_to_none=True)
                        continue
                else:
                    loss.backward()
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                done = step + 1
                avg = sum(losses[-20:]) / len(losses[-20:])
                emit({"type": "progress", "image": 1, "images": 1, "step": done, "steps": steps, "loss": round(avg, 4)})
                if done % 10 == 0 or done == 1:
                    el = time.time() - started
                    print(f"[ti] step {done}/{steps} loss {avg:.4f} {el / done:.2f}s/step", flush=True)
                    emit({"type": "train_status", "step": done, "steps": steps, "loss": round(avg, 4),
                          "sec_per_step": round(el / done, 2)})
                if save_every and done % save_every == 0 and done < steps:
                    p = save_embedding(out_dir / f"{name}-step{done}.safetensors", vectors(), dict(meta, steps=done))
                    saved.append(p)
                    emit({"type": "status", "msg": f"Saved checkpoint {Path(p).name}"})
        except trainer._Cancelled:
            if losses:
                p = save_embedding(out_dir / f"{name}-stopped-step{len(losses)}.safetensors", vectors(),
                                   dict(meta, steps=len(losses)))
                emit({"type": "status", "msg": f"Training stopped; saved progress as {Path(p).name}"})
            raise
        path = save_embedding(final_path, vectors(), dict(meta, steps=steps))
        return {"path": path, "name": name, "steps": steps, "seconds": round(time.time() - started),
                "checkpoints": saved, "vectors": n_vec, "loss": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4),
                "kind": "embedding"}
    finally:
        engine.unload()  # the text encoder now knows extra words: always start fresh afterwards
