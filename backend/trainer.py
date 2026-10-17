"""LoRA training for SD 1.5 and SDXL models (Train tab).

Designed for small Macs (8 GB):
  * images and captions are encoded once up front, then the VAE and text encoder are freed,
    so only the UNet (in float16, frozen) plus the small LoRA (float32) are in memory;
  * gradient checkpointing keeps activation memory low;
  * attention slicing keeps macOS from swapping.

Output formats:
  * "kohya"    - the common .safetensors LoRA layout used by A1111, ComfyUI, Draw Things, DiffusionBee
  * "diffusers" - diffusers/PEFT layout (load_lora_weights)
Both are saved with training metadata (base model, trigger word, tag frequency), which the app
reads back to fill in trigger words automatically.
"""
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import pictures  # noqa: E402  (HEIC, TIFF, AVIF, RAW… are read through macOS, like everywhere else in the app)

IMAGE_EXT = set(pictures.SUPPORTED)


def _open_rgb(path):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(pictures.prepare(path))).convert("RGB")
FRAMING_FILE = ".protoart_framing.json"   # per-picture framing chosen in the Train tab (beside the pictures)
FITS = ("center", "top", "bottom", "left", "right", "pad")
SKIP = "skip"   # left out of training (the file itself stays where it is)
_CENTERING = {"center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0), "left": (0.0, 0.5), "right": (1.0, 0.5)}


def parse_box(fit):
    """A square cropped by hand: "box:x,y,s" with x, y the box's top-left corner as a share of the picture's width and
    height, and s its side as a share of the picture's shorter side (s=1: as big as fits). None if fit isn't one."""
    if not isinstance(fit, str) or not fit.startswith("box:"):
        return None
    try:
        x, y, s = (float(v) for v in fit[4:].split(","))
    except ValueError:
        return None
    if not (0.02 <= s <= 1.0 and 0 <= x <= 1 and 0 <= y <= 1):
        return None
    return x, y, s


def valid_fit(fit):
    return fit in FITS or fit == SKIP or parse_box(fit) is not None


def square(img, res, fit=None):
    """A res x res training picture. fit: which part to keep when cropping (center/top/bottom/left/right), "pad"
    to keep the whole picture and fill the sides with a blurred copy of itself (nothing cut off), or "box:x,y,s"
    for a square picked by hand."""
    from PIL import Image, ImageFilter, ImageOps
    box = parse_box(fit)
    if box:
        w, h = img.size
        side = max(8, round(box[2] * min(w, h)))
        left = min(max(0, round(box[0] * w)), w - side)
        top = min(max(0, round(box[1] * h)), h - side)
        return img.crop((left, top, left + side, top + side)).resize((res, res), Image.LANCZOS)
    fit = fit if fit in FITS else "center"
    if fit == "pad":
        bg = ImageOps.fit(img, (res, res), Image.LANCZOS).filter(ImageFilter.GaussianBlur(max(4, res // 24)))
        fg = ImageOps.contain(img, (res, res), Image.LANCZOS)
        bg.paste(fg, ((res - fg.width) // 2, (res - fg.height) // 2))
        return bg
    return ImageOps.fit(img, (res, res), Image.LANCZOS, centering=_CENTERING[fit])


def load_framing(folder):
    try:
        data = json.loads((Path(folder) / FRAMING_FILE).read_text())
        return {k: v for k, v in data.items() if isinstance(k, str) and valid_fit(v)}
    except (OSError, ValueError, AttributeError):
        return {}


def save_framing(folder, framing):
    """framing: {file name: fit} (a FITS name, "box:x,y,s", or "skip" to leave a picture out). Only names of pictures
    in this folder are kept; "center" (the default) isn't stored."""
    d = Path(folder)
    names = {p.name for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT}
    clean = {k: v for k, v in (framing or {}).items() if k in names and valid_fit(v) and v != "center"}
    f = d / FRAMING_FILE
    if clean:
        f.write_text(json.dumps(clean, indent=1))
    elif f.exists():
        f.unlink()
    return len(clean)


def save_square_copies(folder, res, framing=None, out_name=None):
    """Write square res x res copies of every picture (with its caption) into a new folder beside it, so you can
    see and keep exactly what training will use. The originals aren't touched. Returns the new folder."""
    src = Path(folder)
    framing = dict(load_framing(src), **(framing or {}))
    out = src.parent / (out_name or f"{src.name} · square {res}")
    k = 2
    while out.exists():
        out = src.parent / f"{src.name} · square {res} ({k})"
        k += 1
    out.mkdir(parents=True)
    n = 0
    for it in scan_dataset(src):
        try:
            img = _open_rgb(it["path"])
        except Exception:
            continue
        name = Path(it["name"]).stem
        square(img, res, framing.get(it["name"])).save(out / f"{name}.png")
        if it["caption"]:
            (out / f"{name}.txt").write_text(it["caption"] + "\n")
        n += 1
    return {"folder": str(out), "count": n}


UNET_TARGETS = {
    "attention": ["to_q", "to_k", "to_v", "to_out.0"],
    "full": ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"],
}
TE_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


class TrainError(Exception):
    pass


# ------------------------------------------------------------------ dataset
def scan_dataset(folder, limit=1000, include_skipped=False):
    """The training pictures in a folder, with their captions and framing. Pictures left out in the Train tab
    ("skip") are only listed when include_skipped (the Train tab shows them, dimmed; training never uses them)."""
    d = Path(folder or "")
    if not d.is_dir():
        raise TrainError("Pick a folder of training images.")
    items = []
    framing = load_framing(d)
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("."):
            cap = p.with_suffix(".txt")
            text = ""
            if cap.is_file():
                try:
                    text = cap.read_text(errors="ignore").strip()
                except OSError:
                    pass
            w = h = None
            try:  # reads only the file header, so this stays fast for big folders
                from PIL import Image
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                pass
            fit = framing.get(p.name, "center")
            if fit == SKIP and not include_skipped:
                continue
            items.append({"path": str(p), "name": p.name, "caption": text, "has_caption": bool(text),
                          "width": w, "height": h, "fit": fit})
            if len(items) >= limit:
                break
    return items


def save_captions(items):
    n = 0
    for it in items or []:
        p = Path(it.get("path", ""))
        if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
            continue
        p.with_suffix(".txt").write_text((it.get("caption") or "").strip() + "\n")
        n += 1
    return n


def suggest_steps(n_images):
    """~100 steps per image, kept between 400 and 2000."""
    return int(min(2000, max(400, round(n_images * 100, -1))))


def _caption_for(item, trigger):
    cap = (item.get("caption") or "").strip()
    if trigger:
        t = trigger.strip()
        if not re.search(r"(^|[\s,])" + re.escape(t) + r"($|[\s,])", cap, re.IGNORECASE):
            cap = f"{t}, {cap}" if cap else t
    return cap


# ------------------------------------------------------------------ saving
def to_kohya(unet_sd, te_sd, alpha):
    """PEFT state dicts -> Kohya keys (lora_unet_/lora_te_ + lora_down/lora_up/alpha)."""
    import torch
    out = {}
    for prefix, sd in (("lora_unet_", unet_sd), ("lora_te_", te_sd or {})):
        modules = {}
        for k, v in sd.items():
            m = re.match(r"^(.*)\.lora_([AB])(?:\.default)?\.weight$", k)
            if not m:
                continue
            modules.setdefault(m.group(1), {})[m.group(2)] = v
        for mod, ab in modules.items():
            if "A" not in ab or "B" not in ab:
                continue
            path = mod.replace("base_model.model.", "")
            name = prefix + path.replace(".", "_")
            out[name + ".lora_down.weight"] = ab["A"].detach().to("cpu", torch.float32).contiguous()
            out[name + ".lora_up.weight"] = ab["B"].detach().to("cpu", torch.float32).contiguous()
            out[name + ".alpha"] = torch.tensor(float(alpha))
    return out


def to_diffusers(unet_sd, te_sd, main_prefix="unet."):
    import torch
    out = {}
    for prefix, sd in ((main_prefix, unet_sd), ("text_encoder.", te_sd or {})):
        for k, v in sd.items():
            k = k.replace("base_model.model.", "").replace(".default.weight", ".weight")
            out[prefix + k] = v.detach().to("cpu", torch.float32).contiguous()
    return out


def save_lora(path, unet_sd, te_sd, cfg, fmt, precision, tag_counts, steps_done):
    import torch
    from safetensors.torch import save_file
    if cfg.get("family") == "ernie":
        fmt = "diffusers"  # ERNIE LoRAs use the diffusers layout (transformer.*)
        sd = to_diffusers(unet_sd, None, main_prefix="transformer.")
    else:
        sd = to_kohya(unet_sd, te_sd, cfg["alpha"]) if fmt == "kohya" else to_diffusers(unet_sd, te_sd)
    if precision == "fp16":
        sd = {k: (v.half() if v.dtype == torch.float32 and v.dim() > 0 else v) for k, v in sd.items()}
    meta = {
        "ss_base_model_version": {"sdxl": "sdxl_base_v1-0", "ernie": "ernie_image"}.get(cfg.get("family"), "sd_v1"),
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(cfg["rank"]), "ss_network_alpha": str(cfg["alpha"]),
        "ss_output_name": Path(path).stem, "ss_steps": str(steps_done), "ss_learning_rate": str(cfg["lr"]),
        "ss_resolution": str(cfg["resolution"]), "ss_num_train_images": str(cfg["n_images"]),
        "ss_tag_frequency": json.dumps({"dataset": dict(tag_counts.most_common(200))}),
        "modelspec.architecture": {"sdxl": "stable-diffusion-xl-v1-base/lora", "ernie": "ernie-image/lora"}.get(
            cfg.get("family"), "stable-diffusion-v1/lora"),
        "modelspec.trigger_phrase": cfg.get("trigger") or "",
        "beestudio_family": cfg.get("family", "sd15"), "protoart_base_model": cfg.get("base_label", ""),
        "protoart_format": fmt,
        # the base model's name, so other apps (and Protoart) can tell a Pony / Illustrious LoRA apart
        "ss_sd_model_name": cfg.get("base_label", ""),
    }
    tmp = Path(str(path) + ".tmp")
    save_file(sd, str(tmp), metadata=meta)
    tmp.replace(path)
    return str(path)


# ------------------------------------------------------------------ training
def train(engine, args, emit, cancel_event, lora_dir):
    """Runs on the backend's job thread. engine: the loaded Engine (used to load the base model)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps

    folder = args.get("folder")
    items = scan_dataset(folder)
    if len(items) < 3:
        raise TrainError("Use at least 3 images (10-30 is a good amount).")
    name = re.sub(r"[^\w\- .]+", "_", (args.get("name") or "").strip()).strip() or "my_lora"
    fmt = "diffusers" if args.get("format") == "diffusers" else "kohya"
    precision = "fp32" if args.get("precision") == "fp32" else "fp16"
    trigger = (args.get("trigger") or "").strip()
    res = int(args.get("resolution") or 512)
    res = max(256, min(1024, res - res % 64))
    rank = int(args.get("rank") or 16)
    alpha = float(args.get("alpha") or rank)
    steps = int(args.get("steps") or suggest_steps(len(items)))
    lr = float(args.get("lr") or 1e-4)
    save_every = int(args.get("save_every") or 0)
    flip = bool(args.get("flip", True))
    train_te = bool(args.get("train_text_encoder"))
    targets = UNET_TARGETS["full" if args.get("targets") == "full" else "attention"]
    model_id = args.get("model")
    cfg = {"rank": rank, "alpha": alpha, "lr": lr, "resolution": res, "n_images": len(items), "trigger": trigger}

    out_dir = Path(lora_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{name}.safetensors"
    if final_path.exists() and not args.get("overwrite"):
        raise TrainError(f"A LoRA called '{name}' already exists. Pick another name.")

    # 1. base model (SD 1.5 family only), loaded at full fp16 precision (no 8-bit) and without LoRAs
    from engine import all_models, total_ram_gb
    spec = all_models().get(model_id)
    if spec and spec.get("engine") == "mlx" and spec["family"] == "ernie":
        # ERNIE on the fast engine: QLoRA on the 4-bit model, runs on much smaller Macs than the full model
        import mlx_train
        import paths as _paths
        try:
            return mlx_train.train(spec=spec, items=items, caption_for=_caption_for, args=args, name=name,
                                   final_path=final_path, work_dir=_paths.HOME / "tmp" / "mlx_train",
                                   emit=emit, cancel_event=cancel_event, ram_gb=total_ram_gb(),
                                   pause_event=getattr(engine, "pause_event", None))
        except mlx_train.Cancelled:
            raise _Cancelled()
        except mlx_train.TrainError as e:
            raise TrainError(str(e))
    if not spec or spec["family"] not in ("sd15", "sdxl", "ernie") or spec.get("engine"):
        raise TrainError("LoRA training works with SD 1.5, SDXL and (on 32 GB+ Macs) ERNIE models.")
    family = spec["family"]
    if family == "ernie":
        return train_ernie(engine, spec, args, items, cfg, name, fmt, precision, trigger, res, rank, alpha, steps, lr,
                           save_every, flip, final_path, out_dir, emit, cancel_event)
    if family == "sdxl" and (spec.get("quantize_unet") or spec.get("local_name")):
        # The 8-bit build's weights are compressed and can't learn: train on the original full-precision
        # SDXL instead (already on this Mac if the 8-bit copy was made from it; otherwise a one-time download).
        # LoRAs trained on it work with the 8-bit build too.
        model_id = "sdxl_base_fp16"
        spec = all_models()[model_id]
        emit({"type": "status", "msg": "Training on the full-precision SDXL Base (the 8-bit build can't be trained)…"})
    if family == "sdxl":
        train_te = False  # SDXL: UNet only (two text encoders would not fit on most Macs)
        if 0 < total_ram_gb() < 16:
            emit({"type": "status", "msg": "SDXL training on this Mac will be very slow (it needs 16 GB+ to run well)…"})
    cfg["base_label"] = spec["label"]
    cfg["family"] = family
    emit({"type": "status", "msg": f"Loading {spec['label']} for training…"})
    engine.load(model_id, low_memory=False)
    engine.apply_loras([])
    pipe = engine.pipe
    device = engine.device()
    wdtype = torch.float16 if device == "mps" else torch.float32
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    from diffusers import DDPMScheduler
    noise_sched = DDPMScheduler.from_config(engine.base_scheduler_config)
    pred_type = noise_sched.config.get("prediction_type", "epsilon")

    # 2. encode images (+ flipped copies) and captions once
    tag_counts = Counter()
    latents_mean, latents_std, text_embs, caps = [], [], [], []
    vae_dtype = torch.float32 if family == "sdxl" else wdtype  # SDXL's VAE produces NaNs in float16
    vae.to(device, vae_dtype)
    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items):
            if cancel_event.is_set():
                raise _Cancelled()
            emit({"type": "status", "msg": f"Preparing images {i + 1}/{len(items)}…"})
            try:
                img = _open_rgb(it["path"])
            except Exception:
                continue
            img = square(img, res, it.get("fit"))  # resize + crop (or pad) to a square, as framed in the Train tab
            variants = [img, ImageOps.mirror(img)] if flip else [img]
            for v in variants:
                x = torch.from_numpy(np.asarray(v, dtype=np.float32) / 127.5 - 1.0)
                x = x.permute(2, 0, 1).unsqueeze(0).to(device, vae_dtype)
                dist = vae.encode(x).latent_dist
                latents_mean.append(dist.mean.squeeze(0).to("cpu", torch.float16))
                latents_std.append(dist.std.squeeze(0).to("cpu", torch.float16))
                caps.append(_caption_for(it, trigger))
            for tag in re.split(r",\s*", _caption_for(it, trigger)):
                if tag.strip():
                    tag_counts[tag.strip()] += 1
        if not latents_mean:
            raise TrainError("None of the images could be opened.")
        pooled_embs = []
        if family == "sdxl":
            for m in (pipe.text_encoder, pipe.text_encoder_2):
                m.to(device)
            cache = {}
            for c in sorted(set(caps)):
                pe, _, pooled, _ = pipe.encode_prompt(c, device=device, num_images_per_prompt=1,
                                                      do_classifier_free_guidance=False)
                cache[c] = (pe.squeeze(0).to("cpu", wdtype), pooled.squeeze(0).to("cpu", wdtype))
            text_embs = [cache[c][0] for c in caps]
            pooled_embs = [cache[c][1] for c in caps]
        elif not train_te:
            te.to(device)
            unique = sorted(set(caps))
            cache = {}
            for c in unique:
                ids = tok(c, padding="max_length", max_length=tok.model_max_length, truncation=True,
                          return_tensors="pt").input_ids.to(device)
                cache[c] = te(ids)[0].squeeze(0).to("cpu", wdtype)
            text_embs = [cache[c] for c in caps]
    print(f"[train] prepared {len(latents_mean)} samples in {time.time() - t0:.1f}s", flush=True)
    scaling = vae.config.scaling_factor

    # 3. free what's not needed during training
    pipe.vae = None
    vae.to("cpu")
    del vae
    if family == "sdxl":
        pipe.text_encoder_2.to("cpu")
        pipe.text_encoder.to("cpu")
        time_ids = torch.tensor([[res, res, 0, 0, res, res]], device=device, dtype=wdtype)
    elif not train_te:
        te.to("cpu")
    engine.release_memory()

    # 4. LoRA on the UNet (and optionally the text encoder)
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    unet.requires_grad_(False)
    unet.to(device, wdtype)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=targets))
    params = [p for p in unet.parameters() if p.requires_grad]
    if train_te:
        te.requires_grad_(False)
        te.to(device, wdtype)
        te.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=TE_TARGETS))
        params += [p for p in te.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # train the small LoRA weights in float32 for stability
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass
    try:
        unet.set_attention_slice("auto")
    except Exception:
        pass
    unet.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    warmup = max(1, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))) * 0.9 + 0.1))
    scaler = None
    if wdtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler(device)
        except Exception:
            scaler = None
    alphas = noise_sched.alphas_cumprod.to(device)
    snr_gamma = 5.0

    def snr_weights(ts):
        a = alphas[ts]
        snr = a / (1 - a)
        w = torch.clamp(snr, max=snr_gamma) / snr
        return w + 1 if pred_type == "v_prediction" else w

    def save(path, done):
        te_sd = get_peft_model_state_dict(te) if train_te else None
        return save_lora(path, get_peft_model_state_dict(unet), te_sd, cfg, fmt, precision, tag_counts, done)

    n = len(latents_mean)
    order = []
    started = time.time()
    losses = []
    emit({"type": "status", "msg": f"Training {name}: {steps} steps on {len(items)} images…"})
    saved = []
    try:
        for step in range(steps):
            if hasattr(engine, "wait_if_paused"):
                engine.wait_if_paused()  # Pause: hold between training steps
            if cancel_event.is_set():
                raise _Cancelled()
            if not order:
                order = list(range(n))
                random.shuffle(order)
            idx = order.pop()
            lat = (latents_mean[idx].float() + latents_std[idx].float() * torch.randn_like(latents_mean[idx].float()))
            lat = (lat * scaling).unsqueeze(0).to(device, wdtype)
            noise = torch.randn_like(lat)
            noise = noise + 0.05 * torch.randn(lat.shape[0], lat.shape[1], 1, 1, device=device, dtype=wdtype)
            ts = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=device).long()
            noisy = noise_sched.add_noise(lat, noise, ts)
            if train_te:
                ids = tok(caps[idx], padding="max_length", max_length=tok.model_max_length, truncation=True,
                          return_tensors="pt").input_ids.to(device)
                emb = te(ids)[0]
            else:
                emb = text_embs[idx].unsqueeze(0).to(device, wdtype)
            target = noise_sched.get_velocity(lat, noise, ts) if pred_type == "v_prediction" else noise
            if family == "sdxl":
                cond = {"text_embeds": pooled_embs[idx].unsqueeze(0).to(device, wdtype), "time_ids": time_ids}
                pred = unet(noisy, ts, emb, added_cond_kwargs=cond).sample
            else:
                pred = unet(noisy, ts, emb).sample
            loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
            loss = (loss * snr_weights(ts)).mean()
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue  # skip a bad step instead of ruining the LoRA
            if scaler is not None:
                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                except (RuntimeError, NotImplementedError) as e:  # loss scaling not supported here
                    print("[train] grad scaler unavailable, continuing without it:", e, flush=True)
                    scaler = None
                    opt.zero_grad(set_to_none=True)
                    continue
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(float(loss.detach()))
            done = step + 1
            avg = sum(losses[-20:]) / len(losses[-20:])
            emit({"type": "progress", "image": 1, "images": 1, "step": done, "steps": steps, "loss": round(avg, 4)})
            if done % 10 == 0 or done == 1:
                el = time.time() - started
                print(f"[train] step {done}/{steps} loss {avg:.4f} {el / done:.2f}s/step", flush=True)
                emit({"type": "train_status", "step": done, "steps": steps, "loss": round(avg, 4),
                      "sec_per_step": round(el / done, 2)})
            if save_every and done % save_every == 0 and done < steps:
                p = save(out_dir / f"{name}-step{done}.safetensors", done)
                saved.append(p)
                emit({"type": "status", "msg": f"Saved checkpoint {Path(p).name}"})
    except _Cancelled:
        if losses:
            p = save(out_dir / f"{name}-stopped-step{len(losses)}.safetensors", len(losses))
            emit({"type": "status", "msg": f"Training stopped; saved progress as {Path(p).name}"})
        raise
    finally:
        # leave the base model clean: the training LoRA must not stay attached
        engine.unload()

    path = save(final_path, steps)
    secs = time.time() - started
    return {"path": path, "name": name, "steps": steps, "seconds": round(secs), "checkpoints": saved,
            "loss": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4), "format": fmt, "trigger": trigger}


class _Cancelled(Exception):
    pass


ERNIE_MIN_RAM_GB = 32


def _lora_targets(model, wanted):
    """Module names for LoRA: the usual attention projections, or any attention Linear as a fallback."""
    import torch
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    hits = [n for n in names if any(n == w or n.endswith("." + w) for w in wanted)]
    if hits:
        return wanted
    return [n for n in names if ("attn" in n or "attention" in n) and n.rsplit(".", 1)[-1] in
            ("q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj", "to_qkv", "out_proj")]


def train_ernie(engine, spec, args, items, cfg, name, fmt, precision, trigger, res, rank, alpha, steps, lr,
                save_every, flip, final_path, out_dir, emit, cancel_event):
    """ERNIE-Image LoRA (flow matching). Needs a full-precision ERNIE model and a Mac with 32 GB+."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from engine import total_ram_gb

    ram = total_ram_gb()
    if 0 < ram < ERNIE_MIN_RAM_GB:
        raise TrainError(f"ERNIE LoRA training needs a Mac with {ERNIE_MIN_RAM_GB} GB or more (this one has {ram:.0f} GB).")
    if spec.get("quant"):
        raise TrainError("This ERNIE build is compressed (4/6/8-bit) and can't be trained on. Add the full model "
                         "(e.g. baidu/ERNIE-Image-Turbo or a local copy) in Settings → Models.")
    cfg.update(family="ernie", base_label=spec["label"])
    res = max(512, min(1024, res - res % 64))
    cfg["resolution"] = res
    emit({"type": "status", "msg": f"Loading {spec['label']} for training…"})
    engine.load(args.get("model"))
    engine.apply_loras([])
    pipe = engine.pipe
    device = engine.device()
    dt = torch.bfloat16
    vae, tr = pipe.vae, pipe.transformer
    if pipe.text_encoder is None:
        raise TrainError("The text encoder isn't loaded (low-memory mode); ERNIE training needs it.")
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, torch.float32)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device, torch.float32)

    tag_counts = Counter()
    lat_mean, lat_std, caps = [], [], []
    with torch.no_grad():
        for i, it in enumerate(items):
            if cancel_event.is_set():
                raise _Cancelled()
            emit({"type": "status", "msg": f"Preparing images {i + 1}/{len(items)}…"})
            try:
                img = _open_rgb(it["path"])
            except Exception:
                continue
            img = square(img, res, it.get("fit"))
            for v in ([img, ImageOps.mirror(img)] if flip else [img]):
                x = torch.from_numpy(np.asarray(v, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
                dist = vae.encode(x.to(device, dt)).latent_dist
                lat_mean.append(dist.mean.squeeze(0).to("cpu", torch.float32))
                lat_std.append(dist.std.squeeze(0).to("cpu", torch.float32))
                caps.append(_caption_for(it, trigger))
            for tag in re.split(r",\s*", _caption_for(it, trigger)):
                if tag.strip():
                    tag_counts[tag.strip()] += 1
        if not lat_mean:
            raise TrainError("None of the images could be opened.")
        text_cache = {}
        for c in sorted(set(caps)):
            text_cache[c] = pipe.encode_prompt(c, device, 1)[0].to("cpu", dt)
    text_in_dim = tr.config.text_in_dim
    pipe.vae = None
    pipe.text_encoder = None
    del vae
    engine.release_memory()

    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    tr.requires_grad_(False)
    targets = _lora_targets(tr, ["to_q", "to_k", "to_v", "to_out.0"])
    if not targets:
        raise TrainError("Couldn't find attention layers to train in this ERNIE model.")
    tr.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=targets))
    params = [p for p in tr.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    try:
        tr.enable_gradient_checkpointing()
    except Exception:
        pass
    tr.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    warmup = max(1, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s_: min(1.0, (s_ + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, s_ / steps))) * 0.9 + 0.1))
    shift = float(engine.base_scheduler_config.get("shift", 1.0) or 1.0)
    n_train_t = float(engine.base_scheduler_config.get("num_train_timesteps", 1000) or 1000)

    def save(path, done):
        return save_lora(path, get_peft_model_state_dict(tr), None, cfg, "diffusers", precision, tag_counts, done)

    n = len(lat_mean)
    order, losses, saved = [], [], []
    started = time.time()
    emit({"type": "status", "msg": f"Training {name}: {steps} steps on {len(items)} images…"})
    try:
        for step in range(steps):
            if hasattr(engine, "wait_if_paused"):
                engine.wait_if_paused()  # Pause: hold between training steps
            if cancel_event.is_set():
                raise _Cancelled()
            if not order:
                order = list(range(n))
                random.shuffle(order)
            idx = order.pop()
            z = (lat_mean[idx] + lat_std[idx] * torch.randn_like(lat_mean[idx])).unsqueeze(0).to(device)
            b, c, h, w = z.shape
            z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
            x0 = ((z - bn_mean) / bn_std).to(dt)
            noise = torch.randn_like(x0)
            sigma = torch.sigmoid(torch.randn((1,), device=device))  # logit-normal timestep sampling
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
            xt = (1 - sigma.view(-1, 1, 1, 1)) * x0 + sigma.view(-1, 1, 1, 1) * noise
            text_bth, text_lens = pipe._pad_text(text_hiddens=[text_cache[caps[idx]].to(device)], device=device,
                                                 dtype=dt, text_in_dim=text_in_dim)
            pred = tr(hidden_states=xt, timestep=(sigma * n_train_t).to(dt), text_bth=text_bth, text_lens=text_lens,
                      return_dict=False)[0]
            loss = F.mse_loss(pred.float(), (noise - x0).float())
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(float(loss.detach()))
            done = step + 1
            avg = sum(losses[-20:]) / len(losses[-20:])
            emit({"type": "progress", "image": 1, "images": 1, "step": done, "steps": steps, "loss": round(avg, 4)})
            if done % 10 == 0 or done == 1:
                el = time.time() - started
                print(f"[train] ernie step {done}/{steps} loss {avg:.4f} {el / done:.2f}s/step", flush=True)
                emit({"type": "train_status", "step": done, "steps": steps, "loss": round(avg, 4),
                      "sec_per_step": round(el / done, 2)})
            if save_every and done % save_every == 0 and done < steps:
                saved.append(save(out_dir / f"{name}-step{done}.safetensors", done))
    except _Cancelled:
        if losses:
            p = save(out_dir / f"{name}-stopped-step{len(losses)}.safetensors", len(losses))
            emit({"type": "status", "msg": f"Training stopped; saved progress as {Path(p).name}"})
        raise
    path = save(final_path, steps)
    return {"path": path, "name": name, "steps": steps, "seconds": round(time.time() - started), "checkpoints": saved,
            "loss": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4), "format": "diffusers", "trigger": trigger}
