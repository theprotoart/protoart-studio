"""AnimateDiff: short animated clips from a prompt with an SD 1.5 model (guoyww's motion adapter v1.5-3,
Apache-2.0, ~0.9 GB once). The clip is saved as an animated WebP (shown in the Library) and a GIF beside it.

8 GB Macs: up to 384 px (320 px above 12 frames); each step works on every frame at once, so even an 8-frame clip
takes a few minutes there.
"""
from pathlib import Path

ADAPTER_REPO = "guoyww/animatediff-motion-adapter-v1-5-3"
ADAPTER_GB = 0.9
LICENCE = "AnimateDiff motion adapter: Apache-2.0"


def is_cached():
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(ADAPTER_REPO, local_files_only=True)
        return True
    except Exception:
        return False


def build(sd_pipe, dtype, emit=lambda m: None):
    """An AnimateDiff pipeline sharing the loaded SD 1.5 model's text encoder and VAE."""
    import diffusers
    if not is_cached():
        emit({"type": "status", "msg": f"Downloading the AnimateDiff motion model (first time only, ~{ADAPTER_GB} GB)…"})
    adapter = diffusers.MotionAdapter.from_pretrained(ADAPTER_REPO, torch_dtype=dtype)
    pipe = diffusers.AnimateDiffPipeline.from_pipe(sd_pipe, motion_adapter=adapter, torch_dtype=dtype)
    pipe.scheduler = diffusers.DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear", clip_sample=False, timestep_spacing="linspace", steps_offset=1)
    # The VAE decodes the frames a few at a time, and the picture's attention is worked out in slices: every step
    # works on all the frames at once, and without slicing a 384 px clip needed several GB for attention alone.
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
        # ...but only for the picture's own attention layers. The motion layers attend across the frames for every
        # pixel; sliced, that became tens of thousands of tiny steps per layer (a 384 px clip never finished its
        # first step). They keep the normal attention, which needs little memory there (8–24 frames).
        from diffusers.models.attention_processor import AttnProcessor2_0
        procs = pipe.unet.attn_processors
        pipe.unet.set_attn_processor({k: (AttnProcessor2_0() if "motion_modules" in k else v) for k, v in procs.items()})
    except Exception as e:
        print("[animate] attention set-up:", e, flush=True)
    return pipe


def save(frames, folder, stem, fps=8):
    """frames: list of PIL images -> (webp path, gif path)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    webp = folder / f"{stem}.webp"
    n = 2
    while webp.exists():
        webp = folder / f"{stem}~{n}.webp"
        n += 1
    ms = int(1000 / max(1, fps))
    frames[0].save(webp, save_all=True, append_images=frames[1:], duration=ms, loop=0, quality=90, method=4)
    gif = webp.with_suffix(".gif")
    pal = [f.convert("P", palette=1, colors=256) for f in frames]  # adaptive palette per frame
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=ms, loop=0, disposal=2)
    return webp, gif
