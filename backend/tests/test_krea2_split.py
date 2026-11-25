"""🪶 Krea 2 Turbo on 16 GB Macs: the fast engine builds the picture model without its 8 GB text reader, reads a new
prompt with the reader on its own (the picture model out of memory first, the reader freed after), and the same prompt
again starts at once. mflux and MLX are stood in for (no Apple GPU needed); what's checked is the order of the steps."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
LOG = []
LIVE = set()  # the big parts in memory right now


class Comp:
    def __init__(self, name):
        self.name = name


class Part:
    def __init__(self, name):
        self.name = name
        LIVE.add(name)
        LOG.append(("load", name, frozenset(LIVE)))

    def __del__(self):
        LIVE.discard(self.name)

    def parameters(self):
        return {}


def _stubs():
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m
        return m

    class D:
        @staticmethod
        def get_components():
            return [Comp("vae"), Comp("transformer"), Comp("text_encoder")]

        @staticmethod
        def get_download_patterns(name=None):
            return ["*"]

    class Loaded:
        def __init__(self, names):
            self.names = names

    class WeightLoader:
        @staticmethod
        def load(weight_definition, model_path=None, download_patterns=None):
            names = [c.name for c in weight_definition.get_components()]
            LOG.append(("weights", tuple(names)))
            return Loaded(names)

    class WeightApplier:
        @staticmethod
        def apply_and_quantize(weights, models, quantize_arg, weight_definition):
            assert set(models) == set(weights.names), (models, weights.names)  # every model has its weights
            assert all(v is not None for v in models.values())
            return 4

    class Init:
        @staticmethod
        def _init_config(m, cfg):
            m.prompt_cache = {}
            m.callbacks = types.SimpleNamespace(register=lambda h: None)

        @staticmethod
        def _init_tokenizers(m, path):
            m.tokenizers = {"qwen3vl": "tok"}

        @staticmethod
        def _apply_lora(m, paths, scales, bake, weights):
            pass

    class Encoder:
        @staticmethod
        def encode_prompt_pair(prompt, negative_prompt, guidance, tokenizer, text_encoder, prompt_cache):
            assert text_encoder is not None and tokenizer == "tok"
            LOG.append(("read", prompt, frozenset(LIVE)))
            prompt_cache[(prompt, negative_prompt, guidance)] = ("emb", None)

    class Module:
        def __init__(self):
            pass

    mod("mlx")
    mods["mlx"].nn = mod("mlx.nn", Module=Module)
    mods["mlx"].core = mod("mlx.core", eval=lambda *a: None, clear_cache=lambda: None)
    mod("mflux.models.common.config", ModelConfig=types.SimpleNamespace(
        from_name=lambda b: types.SimpleNamespace(model_name=b, transformer_overrides=None)))
    mod("mflux.models.common.weights.loading.weight_loader", WeightLoader=WeightLoader)
    mod("mflux.models.common.weights.loading.weight_applier", WeightApplier=WeightApplier)
    mod("mflux.models.krea2.weights.krea2_weight_definition", Krea2WeightDefinition=D)
    mod("mflux.models.krea2.krea2_initializer", Krea2Initializer=Init)
    mod("mflux.models.krea2.model.krea2_transformer.transformer", Krea2Transformer=lambda **k: Part("transformer"))
    mod("mflux.models.qwen.model.qwen_vae.qwen_vae", QwenVAE=lambda: Part("vae"))
    mod("mflux.models.krea2.model.krea2_text_encoder.text_encoder", Krea2TextEncoder=lambda: Part("text_encoder"))
    mod("mflux.models.krea2.model.krea2_text_encoder.prompt_encoder", Krea2PromptEncoder=Encoder)
    return mods


class FakeKrea2:
    def generate_image(self, seed, prompt, num_inference_steps, width, height, guidance, **kw):
        key = (prompt, kw.get("negative_prompt"), guidance)
        assert key in self.prompt_cache, "the prompt should have been read already"
        LOG.append(("draw", prompt, frozenset(LIVE)))

        class R:
            image = types.SimpleNamespace(save=lambda out: None)
        return R()


def test_reader_never_shares_memory_with_the_picture_model():
    stubs = _stubs()
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        _run()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run():
    import mlx_worker
    send, mlx_worker.send = mlx_worker.send, (lambda m: None)
    try:
        _flow(mlx_worker)
    finally:
        mlx_worker.send = send


def _flow(mlx_worker):
    w = mlx_worker.Worker()
    w.family, w.low_ram, w.path, w.cls_path = "krea2", True, "/models/krea2", "x:Krea2"
    w.key = ("mflux-community/krea-2-turbo-mflux-q4", "krea-2", (), None, None)

    def load(repo, base, loras, low_ram, force=False, **kw):
        w.model = None
        w.model = w._krea2_without_reader(FakeKrea2, "/models/krea2", base, None, {})
    w.load = load
    load(*w.key[:2], [], True)
    assert w.model.text_encoder is None and "text_encoder" not in LIVE
    assert ("weights", ("vae", "transformer")) in LOG              # the reader's 8 GB file isn't even opened

    args = {"prompt": "a red fox", "seed": 1, "steps": 8, "width": 1024, "height": 1024, "cfg": 1.0, "out": "/tmp/x.png"}
    w.generate(args)
    read = [e for e in LOG if e[0] == "read"][-1]
    assert "transformer" not in read[2] and "vae" not in read[2]   # the picture model was out of memory while reading
    assert ("weights", ("text_encoder",)) in LOG
    loads = [e for e in LOG if e[0] == "load"]
    assert all("text_encoder" not in e[2] for e in loads if e[1] == "transformer")  # reader freed before the reload
    draw = [e for e in LOG if e[0] == "draw"][-1]
    assert "text_encoder" not in draw[2] and "transformer" in draw[2]

    n = len(LOG)
    w.generate({**args, "seed": 2})                                  # the same prompt: no reading, no reloading
    assert [e[0] for e in LOG[n:]] == ["draw"]


def test_wired_up():
    eng = (ROOT / "backend" / "engine.py").read_text()
    block = eng[eng.index('"krea2_turbo_mlx_q4": {'):][:700]
    assert '"min_ram_gb": 32, "low_memory_min_ram_gb": 16' in block and "(16 GB+)" in block
    assert '"low_ram": 0 < ram < 20' in eng  # 16 GB Macs get the split; 24 GB+ keep it all loaded
