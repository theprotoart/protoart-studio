"""Anima (CircleStone Labs): an anime / illustration model built on NVIDIA Cosmos-Predict2 2B, with a Qwen3 0.6B text
encoder and the Qwen-Image VAE. Reads Danbooru tags, plain sentences or both.

Runs through diffusers' modular Anima pipeline (official diffusers files from circlestone-labs). The Turbo build
swaps in the official distilled transformer (anima-turbo-v1.1, 8–12 steps at guidance 1), which is much quicker
on small Macs.

Licence: CircleStone Labs Non-Commercial License (and the NVIDIA Open Model License): the model may not be used
commercially; pictures made with it may.
"""
DIFFUSERS_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
WEIGHTS_REPO = "circlestone-labs/Anima"
TURBO_FILE = "split_files/diffusion_models/anima-turbo-v1.1.safetensors"
LICENCE = "CircleStone Labs Non-Commercial License (model weights: non-commercial; pictures may be used commercially)"
BASE_COMPONENTS = ["text_encoder", "tokenizer", "t5_tokenizer", "vae", "text_conditioner", "scheduler"]


def load_pipeline(turbo, dtype, device, emit=lambda m: None, local_only=False):
    """-> diffusers AnimaModularPipeline on `device`."""
    import diffusers
    from diffusers import ModularPipeline
    pipe = ModularPipeline.from_pretrained(DIFFUSERS_REPO)
    names = list(BASE_COMPONENTS) + ([] if turbo else ["transformer"])
    try:
        pipe.load_components(names=names, dtype=dtype)
    except TypeError:  # older keyword
        pipe.load_components(names=names, torch_dtype=dtype)
    if turbo:
        from huggingface_hub import hf_hub_download
        emit({"type": "status", "msg": "Loading Anima Turbo…"})
        path = hf_hub_download(WEIGHTS_REPO, TURBO_FILE, local_files_only=local_only)
        tr, adapter = load_turbo(path, dtype)
        pipe.text_conditioner.load_state_dict(adapter, strict=True)  # Turbo has its own text adapter weights too
        pipe.update_components(transformer=tr)
    pipe.to(device)
    return pipe


COMFY_PREFIX = "model.diffusion_model."


def load_turbo(path, dtype):
    """The official Turbo file is in ComfyUI's layout: the transformer under "model.diffusion_model." (Cosmos names)
    plus the text adapter under "…llm_adapter.". -> (CosmosTransformer3DModel, adapter state dict)"""
    import diffusers
    import torch
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
    from safetensors.torch import load_file
    raw = load_file(path)
    ad_prefix = COMFY_PREFIX + "llm_adapter."
    adapter = {k[len(ad_prefix):]: v.to(dtype) for k, v in raw.items() if k.startswith(ad_prefix)}
    net = {"net." + k[len(COMFY_PREFIX):]: v for k, v in raw.items()
           if k.startswith(COMFY_PREFIX) and not k.startswith(ad_prefix)}
    del raw
    sd = convert_cosmos_transformer_checkpoint_to_diffusers(net)
    del net
    cfg = diffusers.CosmosTransformer3DModel.load_config(DIFFUSERS_REPO, subfolder="transformer")
    with init_empty_weights():
        tr = diffusers.CosmosTransformer3DModel.from_config(cfg)
    tr.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
    del sd
    tr.eval()
    return tr.to(dtype), adapter


def lean_memory(pipe, device):
    """8 GB Macs: keep only the part that's working in GPU memory. The text encoder (~1.2 GB) moves aside once the
    prompt is read, and the transformer (~3.9 GB) moves aside while the picture is decoded; each moves back when
    it's needed again (a second or two, as Mac memory is shared). The decoder also works in tiles."""
    import torch
    vae = getattr(pipe, "vae", None)
    for name in ("enable_tiling", "enable_vae_tiling"):
        if vae is not None and hasattr(vae, name):
            try:
                getattr(vae, name)()
                break
            except Exception:
                pass

    def free():
        if device == "mps":
            torch.mps.empty_cache()

    def place(module, where):
        if module is not None and next(module.parameters()).device.type != where:
            module.to(where)
            free()

    def before_text(mod, args, kwargs=None):
        place(pipe.transformer, "cpu")
        place(pipe.text_encoder, device)

    def before_denoise(mod, args, kwargs=None):
        place(pipe.text_encoder, "cpu")
        place(pipe.transformer, device)

    def before_decode(mod, args, kwargs=None):
        place(pipe.transformer, "cpu")

    hooks = [pipe.text_encoder.register_forward_pre_hook(before_text),
             pipe.transformer.register_forward_pre_hook(before_denoise)]
    dec = getattr(vae, "decoder", None)
    if dec is not None:
        hooks.append(dec.register_forward_pre_hook(before_decode))
    return hooks


def set_guidance(pipe, cfg):
    from diffusers.guiders import ClassifierFreeGuidance
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(cfg)))


def is_cached(turbo):
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    try:
        snapshot_download(DIFFUSERS_REPO, local_files_only=True,
                          ignore_patterns=["transformer/*"] if turbo else None)
    except Exception:
        return False
    return not turbo or isinstance(try_to_load_from_cache(WEIGHTS_REPO, TURBO_FILE), str)


class StepWatch:
    """Counts denoising steps from the transformer's calls (two per step with guidance) and lets a callback
    stop the run (raise inside it) or report progress."""

    def __init__(self, transformer, on_step):
        import torch
        self.torch = torch
        self.last = None
        self.step = -1
        self.on_step = on_step
        self.handle = transformer.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        t = kwargs.get("timestep")
        if t is None and args:
            t = args[1] if len(args) > 1 else None
        v = float(t.flatten()[0]) if self.torch.is_tensor(t) else t
        if v != self.last:
            self.last = v
            self.step += 1
            self.on_step(self.step)

    def remove(self):
        self.handle.remove()
