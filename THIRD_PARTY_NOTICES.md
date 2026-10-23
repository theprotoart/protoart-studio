# Third-party notices

Protoart Studio is licensed under the Apache License 2.0 (see `LICENSE`).
Copyright 2026 Protoart Studio contributors.

## Code included in this repository

| File(s) | Origin | Licence |
| --- | --- | --- |
| `backend/postprocess/enhancer.py`, parts of `backend/postprocess/finegrain.py` | [Finegrain Image Enhancer](https://github.com/peanutcocktail/Finegrain-Image-Enhancer), © 2024 Lagon Technologies | MIT, see `backend/postprocess/LICENSE-finegrain-enhancer.txt` |
| `backend/postprocess/restoreformer_arch.py` | [GFPGAN](https://github.com/TencentARC/GFPGAN) `gfpgan/archs/restoreformer_arch.py`, © 2021 THL A29 Limited (Tencent); RestoreFormer by Wang et al., CVPR 2022 | Apache-2.0, see `backend/postprocess/LICENSE-gfpgan-restoreformer.txt` |
| `backend/postprocess/codeformer_arch.py` | [CodeFormer](https://github.com/sczhou/CodeFormer) `basicsr/archs/vqgan_arch.py` + `codeformer_arch.py` @ b33cc7d, © 2022 S-Lab; CodeFormer by Zhou et al., NeurIPS 2022 | **S-Lab License 1.0 (non-commercial use only)**, see `backend/postprocess/LICENSE-codeformer.txt` |
| `backend/postprocess/esrgan_model.py` | Written for Protoart Studio. The network layout follows the published ESRGAN generator ([xinntao/ESRGAN](https://github.com/xinntao/ESRGAN), Apache-2.0) so existing checkpoints load. | Apache-2.0 (this project) |
| `backend/tdict_reader.py` | Written for Protoart Studio to read DiffusionBee's `.tdict` file format (interoperability). No DiffusionBee code is included; DiffusionBee itself is AGPL-3.0. | Apache-2.0 (this project) |

## Software the app is built with (bundled in the macOS app)

These are not stored in this repository. The packaged app includes them, and each one ships its own licence file
(inside `Protoart Studio.app/Contents/Resources/python/lib/python3.12/site-packages/*.dist-info/` and
`Contents/Resources/LICENSES.chromium.html`).

| Component | Licence |
| --- | --- |
| Electron (with Chromium and Node.js) | MIT (Chromium: BSD-3-Clause and others, listed in `LICENSES.chromium.html`) |
| Vue | MIT |
| Python (python-build-standalone) | PSF License |
| PyTorch, torchvision | BSD-3-Clause |
| diffusers, transformers, accelerate, peft, huggingface_hub, safetensors | Apache-2.0 |
| refiners | MIT |
| facexlib | MIT |
| onnxruntime | MIT |
| opencv-python | Apache-2.0 (OpenCV), MIT (packaging) |
| NumPy, SciPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| sentencepiece, torchsde | Apache-2.0 |
| protobuf | BSD-3-Clause |
| gguf, sdnq | see each package's licence file |
| mflux (fast engine for ERNIE and FLUX, installed on first use, not bundled) | MIT |
| MLX (Apple, used by mflux) | MIT |
| mlx-vlm (picture reader, installed on first use, not bundled) | MIT |

## Models

The app does not include any model weights. Models are downloaded from Hugging Face (or chosen from your own files)
when you first use them, and each comes with its own licence that you accept by using it. Examples:

| Model | Licence |
| --- | --- |
| Stable Diffusion 1.5 | CreativeML OpenRAIL-M |
| Stable Diffusion XL | CreativeML OpenRAIL++-M |
| ERNIE-Image | Apache-2.0 |
| FLUX.1 [schnell] (Black Forest Labs; MLX conversions by mflux-community) | Apache-2.0 |
| Qwen3.5 4B (Alibaba Qwen; 4-bit MLX conversion [mlx-community/Qwen3.5-4B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit), downloaded on first use, not bundled) | Apache-2.0 |
| sd-vae-ft-mse-original (Stability AI) and sdxl-vae-fp16-fix (madebyollin), recommended VAEs, downloaded on request | MIT |
| FLUX.1 [dev] (Black Forest Labs, official gated files, offered for Macs with 32 GB+) | FLUX.1 [dev] Non-Commercial License |
| FLUX.2 [klein] 4B ([black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B); 4-bit MLX conversion [ar9av/FLUX.2-klein-4B-mflux-4bit](https://huggingface.co/ar9av/FLUX.2-klein-4B-mflux-4bit), downloaded on first use, not bundled) | Apache-2.0 (the original model). The conversion's model card carries a FLUX.1-dev non-commercial tag; check it before commercial use |
| FLUX.2 [klein] 4B outpaint LoRA ([fal/flux-2-klein-4B-outpaint-lora](https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora), pinned revision b11770a, 76 MB, downloaded on first use of ↔ Expand with FLUX.2, not bundled) | Apache-2.0 |
| Juggernaut Reborn (enhancer) | CreativeML OpenRAIL-M |
| 4x-UltraSharp (enhancer upscaler) | CC BY-NC-SA 4.0 (non-commercial) |
| Florence-2 (image to prompt) | MIT |
| WD v3 tagger (Danbooru tags) | Apache-2.0 |
| Depth Anything V2 Small | Apache-2.0 |
| Falconsai nsfw_image_detection (content filter picture checker, optional) | Apache-2.0 |
| Depth Anything V2 Base / Large | CC BY-NC 4.0 (non-commercial) |
| IP-Adapter (h94/IP-Adapter) | Apache-2.0 |
| RestoreFormer weights (via GFPGAN releases) | see the GFPGAN repository |
| CodeFormer weights (CodeFormer v0.1.0 release) | S-Lab License 1.0 (**non-commercial**) |
| LaMa inpainting ([Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX), `lama_fp32.onnx`; original [advimman/lama](https://github.com/advimman/lama), Samsung AI Center), used by 🩹 Remove | Apache-2.0 |
| ControlNet 1.1 (lllyasviel: canny, depth, lineart, scribble, soft edge, openpose, tile) | CreativeML OpenRAIL-M |
| LCM-LoRA SD 1.5 / SDXL (latent-consistency, ⚡ fast mode) | CreativeML OpenRAIL++-M |
| RT-DETR (PekingU/rtdetr_r50vd_coco_o365, person detector for pose) | Apache-2.0 |
| ViTPose (usyd-community/vitpose-base-simple, pose) | Apache-2.0 |
| Enhancer extras (philz1337x/loras, philz1337x/embeddings, Civitai-originated detail LoRAs / negative embeddings) | as published by their authors; see each file's source page |
| ERNIE-Image community builds (OzzyGT SDNQ 4/8-bit, unsloth GGUF, mflux-community MLX) | Apache-2.0 (derived from baidu/ERNIE-Image); see each repository |
| Qwen2.5 0.5B / 1.5B Instruct (prompt enhancer) | Apache-2.0 |
| Anima / Anima Turbo ([circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima), [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers); built on NVIDIA Cosmos-Predict2 2B, Qwen3 0.6B, Qwen-Image VAE; downloaded on first use, not bundled) | CircleStone Labs Non-Commercial License + NVIDIA Open Model License (**model: non-commercial**; generated pictures may be used commercially) |
| Xinsir ControlNet Union ProMax SDXL ([xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0), downloaded on first use) | Apache-2.0 |
| inswapper_128 (fp16) and ArcFace w600k_r50 (InsightFace; files from [facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0), downloaded on first use of 🔁 Face swap) | InsightFace models: **non-commercial research use only** |
| Models and LoRAs you download from Civitai or add yourself | whatever their authors chose: check the model's page |

Always check the model's own page for the current terms before using results commercially.
