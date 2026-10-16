"""Sampler catalogue.

Each entry maps a friendly sampler id to a diffusers scheduler class plus config overrides.
The scheduler is always built from the loaded model's own scheduler config, so things
like beta schedules and prediction type stay correct for that model.

SD 1.5 / SDXL are noise-prediction models and support the full classic set.
ERNIE-Image is a flow-matching model, so only flow-matching samplers apply.
"""

# id: (label, scheduler class name, overrides, needs)
UNET_SAMPLERS = {
    "euler":              ("Euler", "EulerDiscreteScheduler", {}, None),
    "euler_karras":       ("Euler Karras", "EulerDiscreteScheduler", {"use_karras_sigmas": True}, None),
    "euler_a":            ("Euler Ancestral", "EulerAncestralDiscreteScheduler", {}, None),
    "heun":               ("Heun", "HeunDiscreteScheduler", {}, None),
    "dpm2_karras":        ("DPM2 Karras", "KDPM2DiscreteScheduler", {"use_karras_sigmas": True}, None),
    "dpm2_a_karras":      ("DPM2 Ancestral Karras", "KDPM2AncestralDiscreteScheduler", {"use_karras_sigmas": True}, None),
    "dpmpp_2m":           ("DPM++ 2M", "DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}, None),
    "dpmpp_2m_karras":    ("DPM++ 2M Karras", "DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}, None),
    "dpmpp_2m_beta":      ("DPM++ 2M Beta", "DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_beta_sigmas": True}, "scipy"),
    "dpmpp_2m_sde_karras": ("DPM++ 2M SDE Karras", "DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}, None),
    "dpmpp_3m_sde_karras": ("DPM++ 3M SDE Karras", "DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 3, "use_karras_sigmas": True}, None),
    "dpmpp_sde_karras":   ("DPM++ SDE Karras", "DPMSolverSDEScheduler", {"use_karras_sigmas": True}, "torchsde"),
    "dpmpp_2s_karras":    ("DPM++ 2S Karras", "DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}, None),
    "unipc":              ("UniPC", "UniPCMultistepScheduler", {}, None),
    "unipc_karras":       ("UniPC Karras", "UniPCMultistepScheduler", {"use_karras_sigmas": True}, None),
    "deis":               ("DEIS", "DEISMultistepScheduler", {}, None),
    "lms_karras":         ("LMS Karras", "LMSDiscreteScheduler", {"use_karras_sigmas": True}, "scipy"),
    "ddim":               ("DDIM", "DDIMScheduler", {}, None),
    "pndm":               ("PNDM", "PNDMScheduler", {}, None),
    "lcm":                ("LCM (use with an LCM LoRA)", "LCMScheduler", {}, None),
    "tcd":                ("TCD (use with a TCD LoRA)", "TCDScheduler", {}, None),
}

FLOW_SAMPLERS = {
    "flow_euler":            ("Flow Euler (default)", "FlowMatchEulerDiscreteScheduler", {}, None),
    "flow_euler_stochastic": ("Flow Euler Stochastic", "FlowMatchEulerDiscreteScheduler", {"stochastic_sampling": True}, None),
    "flow_euler_karras":     ("Flow Euler Karras", "FlowMatchEulerDiscreteScheduler", {"use_karras_sigmas": True}, None),
    "flow_euler_beta":       ("Flow Euler Beta", "FlowMatchEulerDiscreteScheduler", {"use_beta_sigmas": True}, "scipy"),
}

DEFAULTS = {"sd15": "dpmpp_2m_karras", "sdxl": "dpmpp_2m_karras", "ernie": "flow_euler", "flux": "flow_euler",
            "other": "default"}

# Recommended step counts (common community guidance for SD 1.5 / SDXL). Each value is
# (recommended, low end of the useful range, high end). Fast converging solvers need fewer steps;
# ancestral/SDE samplers keep changing the image, so they benefit from a few more.
RECOMMENDED_STEPS = {
    "euler": (30, 20, 40), "euler_karras": (30, 20, 40), "euler_a": (30, 20, 40),
    "heun": (20, 15, 30),  # two model evaluations per step: slower, so fewer steps
    "dpm2_karras": (20, 15, 30), "dpm2_a_karras": (25, 20, 35),
    "dpmpp_2m": (25, 20, 30), "dpmpp_2m_karras": (25, 20, 30), "dpmpp_2m_beta": (25, 20, 30),
    "dpmpp_2m_sde_karras": (25, 20, 35), "dpmpp_3m_sde_karras": (30, 25, 40),
    "dpmpp_sde_karras": (18, 12, 25), "dpmpp_2s_karras": (20, 15, 25),
    "unipc": (18, 12, 25), "unipc_karras": (18, 12, 25), "deis": (20, 15, 25),
    "lms_karras": (30, 25, 40), "ddim": (45, 30, 50), "pndm": (50, 40, 60),
    "lcm": (6, 4, 8), "tcd": (6, 4, 8),
}
# Flow samplers (ERNIE): the right count depends on the model (Turbo 8, base 50), so the
# model's own default is used; these are multipliers on it.
FLOW_STEP_FACTOR = {"flow_euler": 1.0, "flow_euler_stochastic": 1.0, "flow_euler_karras": 1.0, "flow_euler_beta": 1.0}


def recommended_steps(family, sampler_id, model_default=None):
    """Return (recommended, low, high) for a sampler; None if unknown."""
    if family in ("ernie", "flux"):
        if not model_default:
            return None
        n = max(1, round(model_default * FLOW_STEP_FACTOR.get(sampler_id, 1.0)))
        return (n, max(1, round(n * 0.75)), round(n * 1.5))
    return RECOMMENDED_STEPS.get(sampler_id)


OTHER_SAMPLERS = {"default": ("Model default", None, {}, None)}  # models added from Hugging Face


def table_for(family):
    if family == "ernie":
        return FLOW_SAMPLERS
    if family == "flux":
        return {"flow_euler": FLOW_SAMPLERS["flow_euler"]}  # the fast engine uses its own flow sampler
    if family in ("sd15", "sdxl"):
        return UNET_SAMPLERS
    return OTHER_SAMPLERS


def list_for(family):
    return [
        {"id": k, "label": v[0], "default": k == DEFAULTS.get(family),
         "steps": list(RECOMMENDED_STEPS[k]) if k in RECOMMENDED_STEPS else None}
        for k, v in table_for(family).items()
    ]


def build_scheduler(family, sampler_id, base_config):
    """Create a diffusers scheduler instance for the given sampler id."""
    import diffusers  # imported lazily so the catalogue can be tested without torch

    table = table_for(family)
    if sampler_id not in table:
        sampler_id = DEFAULTS.get(family, next(iter(table)))
    _, cls_name, overrides, _ = table[sampler_id]
    if cls_name is None:  # keep the model's own scheduler (rebuilt fresh from its config)
        import importlib
        cls = getattr(diffusers, base_config.get("_class_name", ""), None)
        return (cls.from_config(base_config) if cls else None), sampler_id
    cls = getattr(diffusers, cls_name)
    return cls.from_config(base_config, **overrides), sampler_id
