# Changelog

## 1.4.2

- Puppet warp: **✂ Separate the subject** (`pwSelectSubject` / `pwUseSubject`): the selection tools pick the subject;
  `heal` with `temp` + `grow` fills what's behind it (LaMa, not saved to the Library); `PuppetEngine.setSubject(matte,
  background)` draws the filled background, then only the subject, bent and cut out by its mask (blended), in the
  preview and the full-size export
- Adjust: **🖌 paint where a step shows** (`AdjustEngine.setPaint`, `pointMask` takes a painted texture as well as the
  points), Paint / Erase / Clear, brush size and strength, [ ] keys
- Enhance: **🎨 Keep the picture's style** (`keep_style`, `style_words`): no SDXLrender LoRA, more_details ≤ 0.3,
  creativity ≤ 0.25, resemblance ≥ 0.8, "photo…" negative, then `color_match.keep_tones` puts the original's broad
  colour and light back
- Edit with words and Inpaint: `color_match.match_to_original` corrects the redraw's broad light/colour to the
  original around the painted area (normalised-convolution blur of the difference) before compositing
- Convert → LoRA: **🔍 Find it** (`lora_base_find` job, `lora_extract.find_base`: relative weight difference over ~80
  sampled weights, .safetensors or .tdict); extraction reports `kept_pct` and `covered_pct`, and the UI warns when the
  base looks wrong
- [ ] / − + change the brush size wherever you paint
- Fixes: Merge LoRAs listed nothing (a second `mergeFamily` computed property, from the model merger, replaced it;
  also `mergeJob`), now `mrgFamily` / `mrgJob`; embedding training with transformers 5 (`attach_words` via
  `set_input_embeddings`); a selection belongs to one tab (`mask.tab`, `maskIsFor`); Vectorize ✕ clear; the Library
  selection bar hides under Compare and full-window boxes sit above it

## 1.4.1

- Convert → LoRA: **What is it?** (Style / Person / Character / Not sure) tags the new LoRA as it's made
  (`content_mark` people, or the 🏷 style / character tag)
- Model → LoRA reads DiffusionBee **.tdict** models directly (`lora_extract.open_weights` / `_TDictWeights`: memory-mapped,
  one tensor at a time, original SD key names), as the model or the base. UI: **→ LoRA** next to a model in the
  DiffusionBee list, .tdict drops in LoRA mode (`cvtAddTdicts`), DiffusionBee models in its lists (`cvtModels`), the
  base guessed from the plain SD 1.5 / SDXL model (`cvtSetFile`)
- Tips on buttons that had none: Convert → LoRA size (rank 16–128), Hi-res ×1.5 / ×2, tiled upscale size, batch
  export size / format / dpi, and Image to image's ⚡ / ★ sampler picks
- Tools → Convert: **drag and drop** (`convertDrop`, paths via `webUtils.getPathForFile`): .tdict files and folders →
  `convert_list` (ticked), .ckpt / .pt / .pth → `ckpt_convert` one after another, .safetensors that are your models →
  the LoRA maker's model, then base
- Models: **💾 Keep a copy on this Mac** (right-click a `local:` .safetensors under /Volumes). `backend/local_copy.py`:
  free-space check (size + 3 GB), chunked read-only copy to `MODELS_DIR/<name>.part` → fsync → size check → rename,
  sidecars (.civitai.info, previews) copied if absent, never overwrites a different file, cancellable. `kept_copies`
  setting (original → copy). `all_models` makes the old `local:` id an alias of the `file:` copy (`kept_as`, not
  listed; old settings and queued jobs still work); the copy has `kept_from`. NSFW / People marks and a fixed type
  move to the copy; the UI moves favourite, default model, VAE choice, model defaults and chain steps
  (`renameModelId`). Trash the copy and the drive's file is listed again
- Image to image → LoRAs: the ★ favourites-only button, the order and the kind / ● People / Not People / 🏷 tag filter,
  shared with Create (one `pickLoras(search)` for both lists); ★ and ● can be set in its list
- LoRA lists: **Newest first / Oldest first**. Each LoRA row has `added` (`loras.added_time`: the later of st_mtime
  and st_ctime, so a copied or downloaded file counts from when it arrived)

## 1.4.0

- Adjust: **Bloom** filter (Procreate-style) and a Bloom look. `bloomPick` keeps the pixels above a threshold set by
  Transition (soft knee, in linear light), two Gaussian blurs (Size and 3.5× Size) give a core and a halo, screened back
  on; Burn adds the light linearly (can blow out). `withDefaults` / `fullOrder` add new filters to saved settings and
  presets (before Film Grain when that's last)
- VAE **Automatic** (Create → VAE, the default; `""` = Automatic, `"__own__"` = the model's own): `vaes.auto_for`
  picks the VAE a Civitai checkpoint comes with (`vae` in its `.civitai.info`), else SDXL fp16-fix for SDXL, else
  vae-ft-mse for an SD 1.5 .safetensors with no `first_stage_model.` weights (`has_own_vae` reads only the header);
  otherwise the model's own. An fp16-fix VAE is built with `force_upcast=False` (decoded in fp16: faster, less
  memory). Civitai downloads fetch a checkpoint's VAE file too (`download_vae`). `vae_auto` command for the UI;
  pictures record the VAE actually used (+ `vae_auto`)
- Regions: **✨ Plan the layout** (`plan_regions` job, `backend/region_plan.py`). Places said in the sentence ("on the
  left", "above", "in the bottom right", one "… with … above" split) are used as they are, with no model. Otherwise
  a language model writes `SCENE:` + `LEFT/RIGHT/TOP/… : subject` lines (few-shot): Qwen3.5 4B through the picture
  reader's worker when it's set up (`vlm.chat`, text only, no thinking; one retry), else the prompt enhancer's
  Qwen2.5. Plans that repeat one subject in several areas are refused (`_distinct`). Lines → cols / rows / grid
  (columns kept aligned, ≤ 4 areas) + the scene for the main prompt. Omost's idea without its 8B LLM. UI: a sentence
  box + button + ↩ undo in Create → Regions. On an 8 GB M1: ~10–20 s a plan with the 4B model
- Speed: cross-attention keys/values worked out once per picture (`backend/kv_cache.py`, attn2 only, self-attention
  keeps its slicing; off with an image prompt or `PROTOART_NO_KV_CACHE`), ~3% per step on SD 1.5, identical output
- Adjust: **◎ control points** per step (`p.points[base|filterId] = [{x, y, r, op}]`, up to 16): a step's result is
  blended back over the picture from before it by a `pointMask` pass (max of soft discs, full strength to 35 % of the
  radius). Add / Remove / move on the preview, Size and Opacity per point; looks and presets keep the points, presets
  and Last used don't store them
- Tools: **🪡 Puppet warp** (`app/renderer/puppet.js`): pins (x, y → tx, ty), moving-least-squares rigid deformation
  (Schaefer et al. 2006) on a 24–80 grid, drawn as a textured WebGL mesh over the original; full-size export to PNG,
  saved through `adjust_save` (`editor: "Puppet warp"`, suffix `_warped`)
- Library: 💾 Save prompt / ⧉ Copy prompt in picture details; saved prompts carry an `nsfw` flag (`prompt_flags`;
  older ones flagged once at start) and are hidden while Hide NSFW is on. Select → ⤓ Prompts .txt / ⤓ Settings .json
  (`settings_list`, read by 📥 Settings as its first item)
- Privacy: **🔒 Privacy mode** (`opts.privacy_mode`) hides the Hide NSFW / Hide People buttons in Create and the
  Library; both switches and the hidden / unchecked counts moved to Settings → Privacy (the Library shows only the
  picture count)
- 8 GB Macs: SDXL always loads with the 8-bit UNet when RAM < the model's `min_ram_gb`; with an IP-Adapter loaded
  only the self-attention layers are sliced (`_slice_self_attention`), instead of no slicing (90+ s/step on SD 1.5)
- Create → reset all: clears LoRAs, unloads the model (`unload_model`), returns to the ☆ default model, folds all.
  Reset Tools settings also clears the patchbay cables (undo restores them). Knob tips go above the knob and stay
  hidden while turning. Image prompt label says "downloaded" (`ip_status`). Convert: Embedding option removed
- Research (not in the app yet): `backend/mlx_sd/unet.py`, an MLX UNet for SD 1.5 / SDXL (diffusers weights,
  NHWC, fused attention in query chunks, optional 8/4-bit linear layers). Matches PyTorch (max diff 0.004 on SD 1.5).
  On an 8 GB M1: SD 1.5 512 px 1.3 s/step (same as PyTorch); SDXL 1024 px 16.5 s/step 4-bit vs 32.7 s/step today, but
  4-bit changes SDXL's output too much (9% mean difference) to ship as is. The real limit on 8 GB is memory: above
  ~3 GB of graphics memory the Mac swaps and steps take 15–25× longer

## 1.3.0

- Downloads: transformers' "Can't load image processor / configuration / tokenizer" now counts as a missing
  first-time download (the OpenPose finder, Depth Anything…), so the job retries online instead of failing with
  "couldn't reach huggingface.co". Download errors get a **⬇ Download it now** button (`allow_download`: the job runs
  with the connection open)
- Tiled upscale: a detail prompt (+ optional words) instead of Create's scene prompt, which made every tile redraw
  the scene (extra faces); gentler defaults (0.25), presets (Gentle / Detail / Creative / Print / Quick),
  CFG ≤ 6, a duplicate-subject negative prompt
- Regional prompts: areas use only their own prompt by default (`add_main`, off); the main prompt reaches them through
  the base blend. FLUX.1 / ERNIE (fast engine): regions by parts (`regions_by_parts`: positional prompt for the
  whole picture, then each area redrawn by image to image from its own prompt and feather-blended back; `refine`).
  FLUX.2 Klein too (each area refined with Klein's edit mode, `_klein_edit`; the slider sets the blend). Anima shows a
  note. ✨ per area: the enhancer's new `region` / `region_tags` styles describe one subject only
- Model chain: one module per step with its own LoRA + weight, extra words (a picked LoRA's trigger words added),
  reorder, and the step's result thumbnail
- Image to image: the prompt box's owner (Create / Image to image) is matched to the opening tab at start-up, and
  💡 suggest from the picture writes into Image to image's prompt even if you switched tabs
- Image to image: 💡 suggest from the picture in every mode (tags for Anima / Pony / Illustrious / NoobAI)
- Right-click menus: models (use, favourite, Reconnect… = `model_relocate` for a moved / missing file, Show in Finder,
  fix type, copy, remove), LoRAs, queue jobs, Settings folders; native Cut / Copy / Paste / spelling menu in text boxes
  (`dialog:locateFile`). `list_models` reports `missing`; loading a missing file says how to reconnect it
- Image to image: Picture / Model chain / ControlNet / Prompt cards fold like Sampler and LoRAs; collapse all · expand
  all; Model chain steps get a LoRA search box; FLUX.2 Klein as a chain step (`edit_words` with the step's words as
  the instruction and its own LoRA; `edit_words` now takes `loras`)
- Tools → Convert → Convert into: Copy / LoRA / Embedding. Model → LoRA (`backend/lora_extract.py`): per-layer
  (fine-tuned − base) of the UNet attention / feed-forward / proj layers and the first text encoder, SVD to rank
  16–128 with an outlier clamp (√S split between up and down), saved as a kohya-format LoRA with diffusers layer names (works on 8 GB: one
  tensor at a time). Embedding: explains it's trained from pictures and opens Train
- Model lists sort A → Z / Z → A / biggest (`opts.model_sort`; favourites first): Create's list, the Image to
  image model menu and the model chain
- Library: sticky selection bar and a 🗑 Delete button beside Compare; Tools → Detail with a LoRA: LoRA search
- Anima (CircleStone Labs, Cosmos-Predict2 2B + Qwen3 0.6B + Qwen-Image VAE) through diffusers' modular
  `AnimaModularPipeline` (`backend/anima.py`): official diffusers files; Anima Turbo = the official
  `anima-turbo-v1.1` ComfyUI file converted on load (`model.diffusion_model.` → Cosmos `net.` names → diffusers,
  plus its own `llm_adapter` → `text_conditioner` weights). 8 GB Macs: text encoder / transformer rotate out of GPU
  memory around their steps, VAE tiling. Text to image and image to image; `anima_turbo`, `anima_base` catalogue entries
  (bf16 only: fp16 overflows to black). The Qwen-Image VAE's `QwenImageCausalConv3d` gives wrong results on MPS
  (every picture flat grey); on MPS each one now runs as a sum of 2D convolutions, one per time tap
  (`anima.fix_vae_for_mps`, identical on CPU to 2e-6). Prompts get Anima's `safe` rating tag (and the other ratings
  in the negative prompt) unless they already name a rating
- Regional prompts (`backend/regions.py`): columns / rows / grid layouts, a cross-attention processor where the
  cond half attends per region (masks at each layer's resolution) blended with the whole prompt (`base`)
- Model chain (Image to image): sequential `generate` jobs across models; fast engine img2img for FLUX.1 / ERNIE
  (mflux `image_path` / `image_strength`)
- Tiled upscale (`backend/tiled.py`, MultiDiffusion): per-step averaged tile noise predictions with Gaussian
  weights, VAE tiling, SDXL per-tile time ids; attention slicing on small Macs (without it each tile step was ~10× slower on 8 GB)
- Face swap (`backend/faceswap.py`): RetinaFace landmarks (facexlib), ArcFace w600k_r50 identity → inswapper_128
  (fp16, ONNX Runtime, CPU), optional 256 px "pixel boost" (2 × 2 tiles), soft inverse-warp paste, CodeFormer tidy;
  consent required, NSFW-checked (explicit results refused), non-commercial labelling. New dependency: `onnx`
- ControlNet Union ProMax (SDXL): `union:<map>` types → `ControlNetUnionModel` + the Union (img2img) pipelines with
  `control_mode`; "Which one should I use?" guide (IP-Adapter Plus / Plus Face / Union)
- AnimateDiff was tried in the 1.3.0 test builds and taken out: far too slow on 8 GB Macs (about 11 minutes a step
  at 384 px). Protoart Studio stays focused on pictures
- Smaller copies: SDNQ int8 / int6 / uint4 UNet copies saved as model folders (`quantize_model`)
- Checkpoint merger (`backend/ckpt_merge.py`): weighted sum / add difference, streamed safetensors writer
- Prompt editing (`backend/prompt_edit.py`): `[a:b:when]`, `[a:when]`, `[a::when]`, switched via
  `callback_on_step_end` (`prompt_embeds`, SDXL `add_text_embeds`)
- Seamless tiling (`_set_seamless`: circular padding in UNet and VAE convolutions)
- Settings import (`backend/settings_import.py`): A1111 / Forge `parameters`, ComfyUI API graphs and workflows,
  InvokeAI, this app's own; PNG text chunks, EXIF user comments, .json; ⤓ Settings .json export
- Fixed: patchbay DETAIL step took the `generate` result list as one picture

## 1.2.2 (test version for friends)

- Speed / ControlNet: diffusers' `from_pipe` casts to float32 unless given a dtype, so every ControlNet, img2img and
  inpaint pipeline doubled its memory (8 GB Macs swapped: 20–75 s per step) and, in float32 on MPS, the ControlNet
  residuals made the latents diverge (std 1 → 9: grey Canny, Scribble noise). `_keep_dtype(pipe)` passes the shared
  model's dtype to all three `from_pipe` calls; Depth / OpenPose detectors are freed after making the map on Macs under 16 GB
- ControlNet: official models load with `use_safetensors=None` (Tile ships only a .bin); `download_problem` tells a
  file missing on Hugging Face (404 / EntryNotFound) from being offline
- Inpainting checkpoints (9-channel UNet): loaded as `StableDiffusion(XL)InpaintPipeline` with a local recipe
  (`local_configs.ensure_inpaint`), reused by `_inpaint_pipe`
- Patchbay: SELECT tab under each module (`patchSelect`, `patchPicked`, `patchPickLabel`) swaps a copy's settings into
  the rack unit (`patchCopyInit/Save`, `patchUnswap`) with a ⚙ badge, replacing the separate panel; ✕ `patchHide` /
  `patchUnhide` (`patch.hidden`, saved, part of flow presets); 🔎 DETAIL module (`post.det`, `detailArgs` → a `generate`
  img2img pass with one LoRA); 🛡 protect mask (`patchProtect`, backend `protect_merge` composites the original back);
  live preview in Tools (Finegrain `on_preview`)
- CodeFormer presets (`FACE_PRESETS`, `facePreset`) in Create and Tools
- 🔄 Contrast prompt: `prompt_enhance` styles `contrast` / `contrast_tags`, `enhance_prompt` arg `contrast`
- Per-model defaults (`modelDefaults`, `applyModelDefaults` after `onModelChange`) and per-LoRA default weight
  (`loraWeights`, `loraDefaultWeight`)
- Library right-click menu (`ctx`, `openCtx`); Tools image → prompt ⧉ Copy / 🖼 Use in Image to image (`promptToI2i`) /
  ✍ Use in Create
- `.ckpt` → `.safetensors` (`backend/ckpt_convert.py`): restricted unpickler (tensors and plain containers only, other
  globals become inert placeholders, every pickle read goes through it), optional fp16, written to
  `~/.protoartstudio/models`; `ckpt_list` / `ckpt_convert` (queued job); "Weights only load failed" now points to it
- Settings → Main folder (`main_folder_get/set`, Models / LoRAs / Embeddings / VAE / ControlNet sub-folders, existing
  A1111-style names reused, only missing ones created); "More model folders"; `paths.folder_access` → ⚠ for folders
  that are missing or not yet allowed by macOS; Refresh reloads models, LoRAs, embeddings and the image folder;
  `NSRemovableVolumesUsageDescription` in the app's Info.plist
- LoRA tags can also be saved as trigger words; LoRA list open state remembered; FLUX.2 [klein] 4B shown as "Klein 4B";
  outpaint LoRA hidden from LoRA lists (`loras.INTERNAL_LORAS`); active LoRAs highlighted
- Variation seed ("Small change"): `engine.variation_noise` (slerp of the seed's and the variation seed's noise,
  made the way diffusers makes it, so strength 0 is the same picture) passed as `latents` for SD 1.5 / SDXL
  text-to-image; `var_seed` / `var_strength` saved with the picture and restored by Use these settings
- 📒 My prompts export / import (.json, or .txt one per line; duplicates skipped, name clashes renamed)
- Save picture as… / Copy picture (`image:saveAs`, `image:copy` IPC) in the details window and right-click menus
- SD 1.5: Med+ (704) size tier, 704 / 768 × 1024 shapes; size warning only when both sides exceed 768 or one exceeds 1024
- Settings → Moved a folder? → Remove (`folder_forget`); Tools → Convert → Pick files or a folder… (`dialog:pickTdict`,
  `convert_list` with `paths`; picked roots allowed for this session)
- Library toolbar: Select / Deselect / Compare laid out in a grid (no absolute positioning), selection bar wraps
- Model libraries: one entry per folder even when found by two spellings (macOS is case-insensitive; dedupe by inode)
- Startup: a status note when reading folders waits on macOS's removable-volume permission
- Splash VU meter redrawn on one circle; prompt list button "Add prompts above to queue them"; detail window buttons
  no longer overlap the text

## 1.2.1 (test version)

- Expand with FLUX.2: reference saved at the exact output size (mflux centre-crops references to /16 at their own
  size); `Engine.align_to_original` (phase correlation, edge-padded shift, 1 − NCC) lines the result up before the
  blend; if it was redrawn (> 0.35) the whole FLUX.2 picture is kept; wider feather; result frame without black bars
- Expand: `initDims` read whenever the picture changes (was stale or 512 × 512, so presets used another picture's
  size); `expandAdvice` warnings; ★ wider / taller recommended; 💡 `suggestExpandPrompt` (Florence-2 description)
- Patchbay: click a module (`patchSelect`, `patch.sel`): rack unit lights up, copies get their own settings
  (`patch.copySettings`, used by `patchRun`); signal flow presets (`patch.presets`, Save flow / load / delete)
- 🏷 LoRA tags (`loraTags`, localStorage `protoart.loraTags`; Models → LoRA tags panel with batch tag / untag /
  rename): `tag:` filters in `loraKinds`, `#tag` search in Create and Image to image
- Expand with FLUX.2: model menu shows FLUX.2 Klein (`i2iUsesKlein`), "outpaint" enhancer style
  (`prompt_enhance.SYSTEM["outpaint"]`), results kept in Image to image (`I2I_KINDS` + inpaint / outpaint)
- Image to image: Picture card first; Sampler and LoRAs hidden when FLUX.2 is used
- Models tab: Embeddings, VAEs & ControlNets foldable (`shopFolds.extras`); Your model files moved to the bottom
- Tooltips: `tips.js` walks up from the pointer, borrows the label / hint next to sliders, menus and boxes, pattern
  tips (`BY_PATTERN`); titles added to ~30 menus and fields
- Create: collapse / expand and Reset all on one row (`.controls > .cfold-all` no wrap)
- ↔ Expand with FLUX.2 Klein + fal's outpaint LoRA (`Engine.outpaint_klein`, `green_canvas`, `OUTPAINT_LORA` pinned
  revision, 76 MB, Apache-2.0): pure-green border canvas, prompt "Fill the green spaces according to the image", LoRA
  scale 1.1, through the Edit with words queue entry (`edit_words` with `outpaint`); original pixels composited back.
  `_mlx_load(extra_loras=…)`. Image to image → Expand: `form.outpaint_engine` ("klein" default / "model")
- Model libraries from other apps (`backend/libraries.py`): finds Stability Matrix (its library.json / settings.json,
  Application Support, portable Data folders), ComfyUI (desktop config + extra_models_config.yaml base_path),
  Automatic1111 / Forge, InvokeAI and Fooocus; read-only (only listing and file sizes). "Existing model library
  detected" window (`libOffer`), Settings → Model libraries from other apps (`libraries_get/refresh/use/add/dismiss`).
  Using one adds its folders to `model_dirs`, `lora_dirs`, `embedding_dirs` and the new `vae_dirs` /
  `controlnet_dirs` (`vaes.EXTRA_DIRS`, `controlnet.EXTRA_DIRS`); Stop using removes only what it added
- Watched model folders: several at once, numbered list, ↻ Refresh (`refreshWatched`)
- Library: drag-to-select / box select in Select mode (`galDown`, `.sel-box`, autoscroll at the edges); picture
  details: ‹ › no longer close the window (the global `button:active` translate moved them off the pointer, so the
  click hit the backdrop); Close / Delete beside the facts (`.detail-facts`, `.detail-side`); Esc closes
- Reset all lever is momentary (`flipResetAll`, `resetFlip`): springs back, no undo; Deselect all centred under the
  Select button (`.sel-stack`); ⇆ Compare moved into the toolbar beside it
- Compare: linked zoom / pan (`cmp`, `cmpWheel`, `cmpZoomTo`, `cmpDown`, Fit, 1:1); Deselect all buttons;
  drag-select reworked: tile positions measured once, one update per frame, selection only touched when it changes,
  pointer capture once dragging, continuous edge scrolling, ends on pointercancel / window blur
- Picture details window: fixed size (`.modal > .detail`), picture fits inside, details scroll; Delete spaced away from
  Close
- Patchbay starts with no cables (`PATCH_DEFAULT = []`, one-time clear via `protoart.patchV2`); buttons ↺ reset /
  clear cables / example
- Train: ✕ Clear the chosen training pictures (`trainClearFolder`; files untouched)
- Patchbay: control bar moved above the panel; modules reorder by dragging the name plate or ‹ › (`patch.order`,
  `protoart.patchOrder`, `patchMoveModule`, `patchHeadDown`); ↺ default resets the order
- LoRA rows: DiffusionBee shown as "BEE"; tag and size in fixed columns (`loraTag`, `.l-meta`), under the name in a
  narrow panel (container query); Image to image rows get the same columns
- Create prompt tools: one toolbar (`ptools`, `ptool`, `togglePtool`) with Enhance / Saved / From image / Random /
  Wildcards / List panels; random subject "animal" (`natural_prompt.js` ANIMAL_KINDS, `prompt_generator.js`
  ANIMAL_TAGS); `rnd.every` new random prompt on each Generate
- Reset all: horizontal blue lever (`reset-toggle`): clears prompts, defaults, folds every section; flip back = undo
- Seed under the negative prompt; negative prompt presets (`kind: "negative"` in 📒 My prompts, `negPresets`)
- `recSettings`: ★ recommended steps / guidance for the sampler (fast: low end, best quality: higher)
- Image to image: sampler menu in Steps card; foldable cards (`i2iFolds`)
- Models tab: Refresh / Add from your Mac swapped; model groups as folding panels (`shopFolds`); Pony / Illustrious /
  NoobAI filters; "Model imported" note (`localOk`); Trigger words panel with batch add / remove / reset
  (`trigBatch`, `batchTriggers`)
- Library: Select in the middle; ⇆ Compare side by side (`compareView`); ‹ › and arrow keys in picture details
  (`detailList`, `detailStep`)
- Patchbay: extra module copies (`patch.extra`, `patchNodes`, `patchAddModule`); width follows the module count
  (`:view-box.camel`, the in-DOM template lowercased `viewBox`)
- How to: search (`helpQ`, `helpHits`)
- ControlNet: `controlnet.clean_from_unet` / `unet_base_state` build the ControlNet from the UNet without LoRA (PEFT)
  layers (".base_layer" / "lora_A"): DiffusionBee ControlNets failed with "Missing key(s) … Unexpected key(s) …
  lora_A" whenever a LoRA was on, then fell back to a download
- `bee_backend.download_problem`: plain message when a one-time Hugging Face download can't happen
- Inpainting checkpoints (`model_hub.is_inpaint_header`: 9-channel conv_in; DiffusionBee by name) flagged `inpaint`,
  labelled, hidden from Create, used for Inpaint / Expand (`inpaintModel`, `opts.use_inpaint_model`); generating a
  whole picture with one gives a clear error
- `dropMismatchedLoras`: LoRAs for another model family are left out of the picture with a note
- LoRA scan includes the watched model folders (`loras._extra_dirs`)
- Random LoRAs: a new mix replaces every LoRA in the list
- Hide People in the Library: `people_prompt` (prompt words, `nsfw.people_flag`) and `faces` (RetinaFace count on new
  pictures; `people_scan` job for older ones), `library.query(hide_people)`
- Image to image: Edit with words / Remove progress there (`i2i` on edit_words / heal jobs), ✕ Clear results, drop
  replaces the picture, enhance now writes into the tab that asked
- Embedding training: unloads everything first, text-encoder gradient checkpointing, 448 px under 16 GB, friendly
  out-of-memory error that saves progress
- LoRA rows: fixed marker and size columns; People / Not People filter; queue buttons no longer cover the text; left
  panels never scroll sideways
- Own application menu (`setAppMenu`): no Reload / Developer Tools in the installed app; `devTools: false` there
- How to: 📜 Model licences is a fold-out row under Troubleshooting (closed by default); the page says version 1.2.1
- Removed: QR code / pattern ControlNet support (pattern picker, `qr.js`, `patterns.js`, Tools → Shape maker, the QR
  and brightness guides, the experimental float32 retries). They gave flat grey pictures on Apple Silicon.
  `controlnet.is_pattern_name`: QR / pattern ControlNet files are listed in Models as not supported and aren't offered
- Image to image → ControlNet: picking one applies its guide's settings (`applyCnGuide(true)` from a watcher, with
  `undoCnGuide`); a "Recommended settings" box under the picker (`cnRecText`) warns about settings that spoil it
  (`cnGuideOff`: Start from this picture on, strength too low, too few steps)
- Fixed: SD 1.5 ControlNets you add yourself failed to load with “doesn't match an
  SD 1.5 ControlNet's layout (52 parts missing)”. `xl_config` no longer gives SD 1.5's last down block attention just
  because the middle block has it
- Models → Move to Trash: the list is refreshed at once (`list_models {fresh}` skips the 10 s folder-scan cache);
  your own model files that no longer exist aren't listed; Trash errors scroll the message into view
- ⏳ Slow-settings warning before Generate on Macs under 24 GB (`slowWarnings`, `slowCheck`): ControlNet size,
  SDXL ControlNet, full-precision SDXL, Hi-res fix with SDXL, big sizes, many SDXL pictures; one-click fixes;
  per-warning "don't warn again" (`protoart.slowWarnOff`), reset in Settings → Speed
- Your own ControlNets (SD 1.5 too) load through the empty-weights path, read tensor by tensor into float16
  (`controlnet._read_half`); memory released after loading
- Queue: ▲ / ▼ reorder waiting jobs (`queue_move`, `JobQueue.move`); Models tab: readable ★ default tag
- Depth map → ControlNet: `depthToControlNet` switches to Change it and an SD 1.5 model; ControlNet guide moved to the
  right-hand column; Library tile actions in one flex row (`.tile-acts`), 🗑 beside ★
- Image to image → ControlNet guides (`CN_GUIDES`, `cnGuide`, `cnModelPicks`, `applyCnGuide`): what it does, the picture
  to give it, best installed models (★ by name), steps, one-click settings; model groups say what they support; "Use ControlNet instead" from Edit with words
- Create and Image to image keep their own prompt / negative prompt (`swapPrompts`, synchronous tab watcher);
  Tools Remove results tagged `from_tools` so they stay out of Image to image's preview
- `recommendedVae` is a computed value (it was a method, so the ⤓ get link never went away)
- Train: leave pictures out (`skip` framing, `scan_dataset(include_skipped)`), crop by hand (`box:x,y,s`,
  `trainer.parse_box`, crop editor), striped tile for pictures the window can't show, caption buttons wrap
- Training checks free memory and swap first on Macs under 16 GB (`Engine._room_to_train`); a missing model file
  gets a plain message instead of diffusers' "Invalid pretrained_model_name_or_path"
- Queue: ⤵ Run the others first (`sendBack` / `resendSentBack`; generate jobs re-queue only their unfinished images)
- Narrow windows: Settings' Look controls drop their words under 1360 px
- Launch screen redesigned as a rack unit powering up (lamps, VU needle, amber display, READY before fading)
- README: screenshots (`docs/screenshots`)
- Fix: the window went blank (and flashed during ✨ enhance now) as soon as a job reported progress: `genLeft` had been
  put among the computed values instead of the methods
- Fast engine LoRAs: not baked on Macs under 24 GB (`mlx_worker.bake_loras`, `bake_lora=False`; override with
  `PROTOART_BAKE_LORA=0/1`). mflux re-quantizes every LoRA-touched q4 layer at q8 when baking, doubling the model:
  FLUX.2 Klein went from 16.7 to 108 s/step and ERNIE to 80–113 s/step on an 8 GB M1. Unbaked, the full LoRA applies
- Tools → 🔌 Patchbay (PATCHBAY toggle, `patch`, `PATCH_NODES`): an SVG back-of-rack panel with IN / OUT jacks for
  Remove, Faces, Enhance, Depth and Vector between PICTURE and RESULT; drag or click-click to patch, drag a plug or
  double-click to unplug; hanging cables (`cablePath`), one cable per jack, no loops, Vector last; ▶ Run patch runs
  the chain with each module's settings (`patchRun`, each step on the previous result's `image_id`), ■ Stop after a
  step; cables saved in `protoart.patch`; a plug click sound (`sounds.js`)
- Tools → Remove: `heal` runs in a side lane (`SIDE_LANE`, `run_side`) while a job is running, with its own progress
  events (`Engine.heal(emit=…)`); LaMa is ONNX on the CPU, so it doesn't compete with the graphics chip
- Object selection: GrabCut at ≤512 px with the mask scaled back up; "Finding the object…" on the picture; no outline
  found → the box itself is selected
- Marching-ants outline around the painted / selected area (`antsMake`, drawn on the selection layer at full opacity)
- Brush-size well follows the slider over its whole range (`brushWellPx`)
- Time left: one figure (`genLeft`) for the progress line, the queue and the TIME LEFT clock; counts down between
  steps; per-model after-last-step time learned (`perf.models[m].tail`) instead of a fixed 2 steps
- SDXL ControlNets: your own files (diffusers or original layout) built at their own size from the SDXL UNet config
  (`controlnet.xl_config`, empty weights + `assign`), `StableDiffusionXLControlNet(Img2Img)Pipeline`; files are
  inspected from their header (`inspect_file`: family, layout, unsupported Control-LoRA / LLLite / Union / T2I-Adapter);
  a name's map word (canny, depth, lineart, openpose…) picks the control map (`guess_map`); Civitai SDXL ControlNets
  allowed


## 1.2.0

- Compare: values are picked with keys (blue, shaded to match the look; lit when picked; typing still works)
- 🛡 SFW switch moved to the top of Settings, beside Tips and Sounds
- Selection tools wherever you paint a mask: Brush, Erase, Lasso, Polygonal lasso, Magic wand (tolerance, connected
  area, pixels read from the file), Object selection (box → OpenCV GrabCut, `select_object.py`, no download); Add /
  Subtract (⌥), keys B E L P W O, Esc / Enter; dashed outlines drawn on their own layer
- VAE: each model remembers its VAE (`protoart.vaeByModel`), a VAE for another model type is never kept, and Create
  offers the recommended one in one click (`vae_get`: vae-ft-mse-840000 for SD 1.5, sdxl-vae-fp16-fix for SDXL, MIT,
  pinned revisions)
- Create: 🎲 Random / { } Wildcards / 📋 List as three bevelled keys in one row
- 🛡 Safe-for-work mode (`settings.sfw_mode`, `bee_backend.sfw_check`): adult prompts, NSFW-marked or -named LoRAs and
  models refused in the engine; NSFW hidden and not openable; every picture checked; SFW lamp
- VAEs: Civitai "VAE" downloads to `~/.protoartstudio/vaes` (`vaes.py`), Create → Extras → VAE swaps the pipeline's VAE
  (`engine._apply_vae`, LDM files converted in place, no download); recorded in the picture's settings
- Your own SD 1.5 ControlNets: Civitai "Controlnet" downloads to `~/.protoartstudio/controlnets`, listed as `user:<name>`
  ControlNet types, converted in place like DiffusionBee's; they follow the picture as it is
- Models tab: Embeddings, VAEs & ControlNets section (filter by type, search, open folder)
- FLUX.1 Dev catalog entry (`flux_dev_mlx_q8`): official gated files (only the diffusers parts), 8-bit while loading
  (`mlx_quantize`, `mlx_patterns`), 32 GB+, FLUX.1 [dev] Non-Commercial License
- Tools: Original / Result / Split (draggable) / Side by side views in the picture window (`tool.view`, `cmpDrag`)
- Brush-size ring following the pointer while painting (`brushAt`), shown on slider moves (`brushPreview`); a recessed
  metal size well beside each brush slider
- LoRA list: sort (A–Z, Z–A, biggest, smallest; favourites first) and filter by kind / flavour (`sortLoras`, `loraKinds`)
- 🔔 Notification sounds (`sounds.js`, Web Audio, no files): finished pictures, training done, failures; Settings switch
- How to: Community & support (Discord, contact, report a problem, updates) and a model licence table
- Negative prompt ✨ suggest / 🎲 random (`negative_prompt.js`): tags for SD 1.5 / SDXL / Pony / Illustrious (flavour
  low-quality tags), plain words for ERNIE / FLUX, the opposite of the prompt's style, subject fixes; written
  automatically after 🎲 Random prompt and ✨ enhance unless you typed your own (`autoNegative`)
- Prompt enhancer: tags for SDXL too; new "edit" style rewrites Edit with words instructions; 📒 My prompts (saved
  prompt + negative pairs, `protoart.prompts`)
- A FLUX.1 LoRA can't be loaded into FLUX.2 (or the other way) even when its type was set by hand (checked against the
  file); the fast engine's "could not be applied" error is explained in plain words (`mlx_engine.friendly`)
- Two of your model files with the same name are told apart in the lists ("· in “SD1”", "· a copy")
- Remove buttons for added models are locked (🔒) until unlocked, and every remove asks first (`confirmAsk`)
- Library cards: 🗑 delete at the top left (asks first), ↺ Prompt at the bottom; Tools: ✕ Clear results
- Tips also for the toggle labels (Start from this picture, Use the picture's size, append, …)
- Image to image: ✨ Enhance prompt switch and "enhance now" in its Prompt card (not for Edit with words or Remove);
  its results (img2img, healed, word_edit) no longer appear in Create's session list (`createSession`, `I2I_KINDS`);
  Create's "clear all" leaves them alone
- 📝 Image → prompt "Describe": Qwen3.5 4B (Apache-2.0; `mlx-community/Qwen3.5-4B-MLX-4bit`, pinned revision 32f3e8e,
  ~3 GB) through mlx-vlm 0.7.4 in its own environment (`~/.protoartstudio/venv-vlm`, `backend/vlm.py`); a short-lived
  worker (`vlm_worker.py`) per picture so its memory is freed at once. Florence-2 is now "Quick"; the WD tagger "Tags".
  The Tools module shows the written prompt
- Header: a chrome bat-handle TIPS toggle (`.bat-toggle`, drawn from its own parts: nut, lever, ball; ON / OFF) beside
  bigger Power / Busy / Fault lamps
- Remove: "✨ Smart fill" (LaMa) / "⚡ Quick fill" with a one-line explanation; Image to image's picture keeps the same
  size in every mode (max 420 px high)
- Tools: Remove objects paints on Tools' own picture (`tool.healing`, `toolPaintRemove`, `toolHeal`; the mask code
  follows `maskTarget()`); bigger picture caption with "✕ Remove picture" and "✏️ Edit in …" buttons; shorter modules
  (icon beside the title; Enhance's Creativity / Resemblance under fine-tune)
- TIPS switch with a blue lamp beside Power / Busy / Fault in the rack header
- 🎲 Random prompt: natural-language scenes for FLUX / FLUX.2 / ERNIE (`natural_prompt.js`, `window.NaturalPrompt`;
  subject, detail and "with a look" options; the negative prompt is left alone); Danbooru tags stay for SD 1.5 / SDXL
- Image to image model list: "FLUX.2 (✨ Edit with words)" group; picking it turns on Edit with words, picking another
  model leaves it (`pickI2iModel`)
- ✨ Edit with words (Image to image → new first mode): FLUX.2 [klein] 4B edit (`Flux2KleinEdit`, same weights) on
  the fast engine; new queued job `edit_words` (`engine.edit_words`): the picture plus up to 3 extra pictures as
  references, ~0.6 MP on 8 GB Macs (1 MP otherwise), result resized back; an optional painted area is composited over
  the untouched original with a soft edge. Saved as kind `word_edit`. The worker loads FLUX.2 as `variant="edit"`
- Tools: every tool is a vertical rack module (`.module-rack`): Run chain (Enhance → Restore faces), Enhance (main
  controls up front, steps / work size / solver / seed / prompt under "fine-tune"), Restore faces, Depth map, Remove
  objects, Vectorize, Image → prompt, Convert: an even 4 × 2 grid (2 columns when narrow, container query); Convert's
  file list opens as a full-width panel under the rack
- Sampler picks: at most two "Fastest" and two "Best quality" chips
- New model: **FLUX.2 [klein] 4B** on the fast engine (`flux2_klein_mlx_q4`, family `flux2`): mflux `Flux2Klein`
  with the 4-bit conversion `ar9av/FLUX.2-klein-4B-mflux-4bit` pinned to revision 86214a1 (~4.6 GB), 4 steps, guidance 1,
  no negative prompt. On small Macs the Qwen3 text encoder is dropped while drawing; encoded prompts are cached in the
  worker (`_cache_flux2_prompts`). Live preview via the FLUX.2 autoencoder factors. Fast-engine downloads and cache
  checks accept a pinned `mlx_revision`
- FLUX.2 LoRAs: detected from metadata, Civitai base model "Flux.2 …", or the block count (5 double + 20 single
  blocks); a FLUX.2 type folder and filters in Models, LoRA folders and Fix a type
- 🩹 Remove (spot healing): Image to image → Remove mode (paint, press Remove) and Tools → Remove objects.
  `postprocess/heal.py`: LaMa (Apache-2.0; Carve/LaMa-ONNX `lama_fp32.onnx`, pinned revision c3c0c9e, ~208 MB,
  ONNX Runtime on the CPU; context box around the mask, 512 px, blended back with a soft edge) or OpenCV Telea "Quick".
  Saved as kind "healed" linked to the original. New queued job `heal`
- ✒️ Vectorize (Tools): `postprocess/vectorize.py`, k-means palette (2-24), speck cleanup, traced outlines with holes,
  simplified and drawn as smooth quadratic curves (sharp at the picture's edges), background colour from the border;
  SVG saved beside the pictures. New queued job `vectorize`
- ★ favourite LoRAs (listed first; ★ filter beside the search), stored in `protoart.favLoras`
- Tips: app-wide quick tooltips (`tips.js`; element titles plus explanations for untitled buttons) with a Settings switch
- 🎲 Random LoRAs laid out as a sentence; trigger-word hints explained
- LoRAs of another model type are put aside per type and restored on switching back (they used to be dropped); the
  default model is applied once per launch, not again when the window reloads (sessionStorage)
- Fix: Image prompt (IP-Adapter) on 8-bit / quantized SDXL UNets ("Casting a quantized model to a new dtype is
  unsupported"): `unet.to` only moves the device while the adapter loads; the adapter's own layers get the working dtype
- Civitai: downloads are queued (box freed at once; Downloads list with Stop / Remove); possible duplicates found by
  .civitai.info version id, same file name or same size (`civitai.find_duplicates`) and confirmed before downloading
- Create → "reset all" (everything but prompt, model and presets, with undo; `defaultForm()`); Tools → "Reset Tools
  settings" (`defaultPost()`, with undo)
- Restore faces: **CodeFormer** added as a second model beside RestoreFormer, in Create, Tools and the picture details,
  with a Fidelity slider (CodeFormer only). CodeFormer code (`backend/postprocess/codeformer_arch.py`) and weights are
  S-Lab License 1.0, non-commercial; CodeFormer is the default (listed first; saved settings switched once)
- Face models are verified against pinned SHA-256 hashes after download (and re-fetched if damaged), loaded with
  `torch.load(weights_only=True)`; the face detector is only prepared when faces are first restored
- Restored pictures record the face model, strength and fidelity in their settings; presets keep the face model
- Help → What's new and About updated

## 1.1.0

From the first testers' notes:

- Train → Square framing: per-picture Crop centre / top / bottom / left / right or Keep whole (blurred fill), one by one
  or for all (or only the non-square ones); tiles preview exactly what training sees. Saved per folder in
  `.protoart_framing.json` and used by LoRA (SD 1.5 / SDXL / ERNIE fast engine) and embedding training
  (`trainer.square`). "Save copies" writes square 512 / 768 / 1024 PNGs with captions to a new folder beside it
- Image to image: its own LoRAs card (add / weight / remove; shares Create's LoRA list, as both tabs use one model)
- Create: ✕ on each picture removes it from the screen (kept in the Library) and "clear all"; the model that made each
  picture is shown along its bottom. Results of Tools jobs (enhance, faces, depth) no longer appear in Create
- Library: Hide People switch (hides pictures made with ● People models or LoRAs) next to Hide NSFW
- Tools: Pause and Stop grouped in one column (Stop was pushed onto a cramped second row)
- Civitai: civitai.red links; a firewall / VPN block ("[Errno 9] Bad file descriptor", "connection abort") is retried
  once through macOS's curl; plain-language message; ⟳ Retry and Clear; Check and key checks run beside the command
  reader, so a slow or blocked site no longer freezes the app (they used to block every other command), and they no
  longer switch the whole engine online; the window gives up after 90 s instead of waiting forever
- "Your Mac is short of memory" note before SD 1.5 / SDXL pictures on Macs under 16 GB when little memory is free and
  several GB are already swapped (an 8 GB Mac with browsers open ran SDXL at 30-100 s a step)
- SDXL 8-bit: after the first-time conversion on Macs under 24 GB, the model is reloaded from the saved compact copy
  (conversion leftovers no longer linger); Image to image / inpainting pipelines no longer print progress bars to the log
- Stop: a job that hasn't stopped 2.5 s after Stop / Stop all is force-stopped automatically (engine restart; waiting
  jobs re-added, or dropped after Stop all). Training keeps the graceful stop (it saves a checkpoint) and the Force stop
  now button
- Version 1.1.0

## 0.3.0

- Train → Merge LoRAs (lora_merge.py): two or more LoRAs of one model type into one, each at its own weight. Exact
  (ranks stacked: identical to using them together) or Smaller (SVD to rank 16-128). kohya / PEFT / old diffusers
  layouts, linear + conv layers; LyCORIS (LoHa / LoKr / DoRA / LoCon-mid) and mixed layouts refused with a message.
  Saved fp16 into Protoart's LoRA folder for that type with merged trigger-word tags; queued job with progress
- Train → Embedding (ti_trainer.py): textual inversion for SD 1.5 (1-8 vectors, subject or style templates, starting
  word); latents cached and VAE freed, UNet frozen in fp16 with gradient checkpointing, only the new word's vectors
  trained (rest of the vocabulary restored every step); Pause / Stop (keeps progress) / checkpoints; saved as
  "emb_params" .safetensors in the embeddings folder
- Colour themes on shared tokens: Amber (classic charcoal and amber), Cream (warm cream, yellow lamps, deep red),
  Slate (blue-grey steel, ivory lamps) and Ember (black and red-orange), each in Light
  and Dark; faceplate metal tinted per theme (rack.js), lit controls, glows, Generate button and the light faceplate's
  painted parts follow the theme. Original designs inspired by studio hardware
- Lit state: radio choices are push buttons that light up fully; every "on" control glows a little more
- ★ Favourite models (kept on this Mac, files untouched): listed first in Create, ★ Favorites filter in Create and the
  Models tab; ★ ● ⚑ markers in a fixed column on every model / LoRA row (names clamp to two lines, full name on hover)
- Models tab: Your model files, Add from Civitai, Add from Hugging Face and Fix a type moved here from Settings
  (folded until opened); Settings → Model folders keeps watched folders and the downloaded-models folder
- Settings sections never spill into the next column (narrow windows)
- Content filter: many more explicit words recognised in LoRA / model names (also with version numbers and in
  run-together names), so Hide NSFW keeps them out of 🎲 Random LoRAs and the lists
- How to → Disclaimer & licences
- Picture details and confirmation pop-ups: no decorative screws (they looked like close buttons)
- make_dmg.sh: cleans up its temporary files even when it stops, removes leftovers of earlier failed runs, builds the
  install window in a sparse image (needs ~3 GB free, not 4+)
- ❚❚ Pause / ▶ Resume (Create, Image to image, the queue, Tools, Train): the running picture waits at its next step
  (diffusers: in the step callback; fast engine and fast-engine training: the worker process is suspended with
  SIGSTOP / SIGCONT), enhancing waits between tiles, training between steps, and no waiting job starts. Stop / Skip
  still work while paused; Stop all ends a pause. Time-left clocks don't count paused time
- ↺ Reuse prompt: on hover in Create / Image to image / Tools results and the Library (↺ Prompt), and in a picture's
  details (↺ Reuse prompt, ↺ Original prompt when Enhance prompt or wildcards changed it); ⚙ All settings on hover
- Sampler list: ⚡ Fastest and ★ Best quality picks for the chosen model's type (SD 1.5, SDXL, Pony / Illustrious,
  FLUX, ERNIE), as option groups and chips outlined in faint blue
- Fix a type: 🔒 lock per row (a type you pick locks itself), and Show all
- Image prompt (IP-Adapter) on a fresh install: “does not appear to have a file named …” is now recognised as “not
  downloaded yet”, so the adapter downloads instead of failing
- Pictures you drop or pick (Tools, Image to image, image prompt, image → prompt) can be HEIC/HEIF, TIFF, AVIF, PSD,
  JPEG 2000, GIF, BMP, TGA, EXR or camera RAW (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2…), besides PNG / JPEG / WebP:
  converted once to PNG by macOS's own reader (sips / ImageIO, no new decoder libraries), cached, capped at 8192 px
  The same goes for 🎚 Adjust, Batch edit & export, Train (folders of HEIC / TIFF / RAW photos) and pictures saved
  back from an outside editor (PSD, TIFF, HEIC)
- Models and Library toolbars wrap onto more lines in a narrow window (the top panel grows) instead of being cut off;
  How to uses the full width of the window
- The name plate at the top (How to, Train…) is centred on the window, like the Look switch
- Library → 🗂 Batch edit & export (PhotoMill-style): resize (long/short side, fit in a box, percent, never enlarge),
  convert (same / JPEG / PNG / WebP / TIFF), sRGB, dpi, apply an Adjust Look or preset, rename from a pattern, keep or
  remove the details inside (removed by default), never overwrites; ready-made and your own export presets; progress,
  Stop and a size report
- 🗜 Compress (ported from ProtoArt Press): the smallest JPEG / PNG that still measures as looking the same (windowed
  SSIM incl. worst 64×64 region and brightness shift; JPEG budget relative to a q96 baseline; PNG lossless with exact
  palette / grey, optional invisible 256-colour version, verified pixel for pixel). From a picture's details or a
  selection
- Hugging Face access token (Settings → Models) for gated models: steps shown in place when a model needs it,
  checked when saved, sent only to huggingface.co (env only, never ~/.cache/huggingface; tokens saved by other apps
  are still never used); friendly messages for gated / 401 / 403 downloads everywhere
- Settings: the Look switch is bigger, its title sits over its description, and it's centred on the window
- 🎚 Adjust, in the Tools tab (also reachable from the Library's picture details and Image to image): Lightroom /
  Camera Raw-style Light, Color and Presence sliders and Color Efex-style filters (Detail Extractor, Dark Contrasts,
  Bleach Bypass, Glamour Glow, Film Grain, in a reorderable stack, switched with amber push buttons), plus one-click
  Looks; the picture on the left is the live preview (GPU, WebGL 2); saves an adjusted copy to the Library, linked to
  the original. Works even before the app is fully restarted after an update (reads/saves through the engine).
  Compare with Edited / Original / Split (drag the line; \\ flips), ✕ Close Adjust, your own named presets (kept in
  the library database), Save & enhance (the AI enhances the adjusted picture), Save & send to Image to image, and
  Several versions at once: one picture per preset, optionally enhanced (enhance once then filters, or filters then
  one enhance per version through the queue)
- Tools: 🖼 Send to Image to image for the original and for every result; How to has a "What's new" page
- Create: every section folds down to one line showing what's set (click its name; collapse all / expand all;
  remembered), like DiffusionBee. Settings starts with only This Mac and Models open (fold all / open all; links to a
  setting open its section)
- Civitai: most downloads now need a signed-in account. The Civitai card asks for the API key up front (with the steps),
  and again in place when a download is refused (Save & download); the key is checked with Civitai when saved, sent as
  a header only to civitai.com (never to the file server it redirects to); clearer messages for a missing vs refused
  key; Troubleshooting lists every Civitai message
- Enhance & upscale speed-ups from ProtoArt Beautify: a tile planner (fewest tiles that fit the Mac's memory; padded
  to 64 px and cropped back), GPU memory handed back to macOS after every pass, sliced attention, ESRGAN in small
  tiles on 8 GB Macs, Turbo now loads (the tile ControlNet is unplugged while it's attached), new ⏩ Instant speed
  (upscaler only) and "Work at" 512 / 768 px (Auto = 512 px for the quick speeds on Macs under 16 GB)
- Embeddings (textual inversions) for SD 1.5 and SDXL (Pony / Illustrious / NoobAI too): Settings → Embeddings adds
  files (copied into Protoart's embeddings folder) or folders (used in place), From Civitai downloads them
  ("Textual Inversion", .safetensors or .pt read weights-only), and embeddings left in LoRA folders are found and no
  longer listed as LoRAs. Create shows only the embeddings for the chosen model's type; click one or type its name
- Enhance & upscale is much faster: new 🚀 Turbo speed (distilled LCM add-on: 4–8 steps, no negative-prompt pass),
  the two detail LoRAs are merged into the model once instead of applied on every step, Fast now works on Apple
  GPUs, and the output size shown is the real one (pictures are worked at 768 px on the short side × the factor).
  "Enhance Speed Test.command" times Turbo / Fast / Standard on your Mac
- Fix a type (Settings → LoRA folders, or "fix a type" under Create's model list) covers your own base model files
  too (SD 1.5 or SDXL); "Show unknown type" lists everything the app couldn't identify
- LoRA lists show only LoRAs made for the chosen model's type (unknown-type LoRAs are no longer shown for every
  model; the list counts them and links to Fix a LoRA's type, which has an "Unknown type only" filter)
- How to and Troubleshooting updated for everything in this release (LoRA types and folders, moved folders,
  Civitai, stalled downloads, Pony 8-bit conversion, image prompt errors, large enhances, deleting models)
- From Civitai (Settings → Models; also in the Models tab and under Create's model list): paste a civitai.com link,
  Check, Download. LoRAs go to the folder set to their type, SD 1.5 / SDXL / Pony checkpoints to your checkpoints
  folder; trigger words saved alongside; only .safetensors; resumes; optional Civitai API key for downloads that
  need an account; Civitai's NSFW flag is applied
- Sort into type folders (Settings → LoRA folders): moves each LoRA of a folder into a sub-folder for its type,
  reusing ones you have (Flux, sdxl, SD1…) and making the rest; shows the plan first; suggested when you add a
  folder of mixed LoRAs. A new install starts with SD 1.5 / SDXL / FLUX / ERNIE folders in Protoart's LoRA folder
- Fix a LoRA's type (Settings → LoRA folders, or "LoRA missing or the wrong type?" in Create's LoRA list): pick
  SD 1.5 / SDXL / FLUX / ERNIE for any LoRA; optionally the file (and its trigger-word / preview files) moves into
  the LoRA folder set to that type. Unclear DiffusionBee .tdict LoRAs count as SD 1.5
- Moved a folder? (Settings): point the app at a folder's new place (e.g. a new drive) and everything that used it
  follows: LoRA folders and their types, watched folders, added models, ⚑/● marks, presets and the Library's model
  records. Missing folders are flagged at startup. Picking the wrong folder is refused
- Long LoRA names wrap instead of being cut off; the ● People marker sits after the name, next to ⚑
- Enhance & upscale: Speed presets (⚡ Fast 10 steps DPM-Solver · Standard · ★ Best 30 steps) next to the Look presets,
  and a warning when the result would be very large
- LoRA folders can be assigned a model type (SD 1.5 / SDXL / FLUX / ERNIE, or Auto); DiffusionBee .tdict LoRAs are
  read from your own folders too
- Settings no longer loses sections in a small window (one column when narrow); tabs form two even rows when narrow
- Create's model and LoRA lists: ⚑ marks NSFW (click to change); ● tags a model or LoRA as made of real people,
  and the new Hide People switch hides those. Settings sections fold away (click a heading); long lists fold too
- DiffusionBee .tdict LoRAs, models and embeddings are now read from your own folders too (LoRA folders and watched
  model folders), not only from ~/.diffusionbee. Adding or watching a models folder that also holds LoRAs makes it a
  LoRA folder as well. (Before, a message wrongly said such files were already listed.)
- Models tab: ☆ Make default (Create starts with it), 🗑 Delete downloaded models (shared files kept), Move your own
  model files to the Trash; ↻ Refresh in Create, the Models tab and Settings
- Fixed: Pony and other SDXL checkpoint files failed in Low-memory mode ("Cannot copy out of meta tensor"). They're
  now converted to 8-bit once and the copy is saved, so later loads are quick
- Fixed: downloads could hang forever on a silent connection (the enhancer's first download, FLUX). Stalled
  connections now give up after 2 minutes and the download resumes next time; downloads use plain resumable HTTP
- Pony / Illustrious / NoobAI quality tags can be put straight into the prompt (⤴ put in prompt)
- On/off switches are now push buttons that light up amber when on, matching the rest of the panel
- Speed: a model can be loaded in the background before you press Generate: when the app opens (the one you used
  last, by default) and as soon as you pick one in Create. Uses idle time only, never downloads; Create shows
  "getting ready…" / "loaded, ready"
- Content filter: a Hide NSFW switch in Create, Library and Settings hides NSFW pictures (left out of the Library,
  covered elsewhere, click to look), models and LoRAs. Pictures are judged by their prompt and, if you turn it on,
  an on-device picture checker (Falconsai NSFW detector, ~340 MB once); models and LoRAs by name and training
  tags. Your own marks (pictures, models, LoRAs) always win
- New look: the whole app is a studio rack unit. Brushed-aluminium faceplate with procedurally generated scratches,
  dings and wear (drawn on this Mac at launch, no image files), rack ears and screws, engraved labels
- Bakelite knobs for Steps, Guidance and Images (drag, scroll or use the arrow keys; type in the amber readout),
  a detented rotary switch for the maximum size, bakelite faders, chrome bat-handle toggles and LED push buttons
- Analog VU meter whose needle shows the running job's progress, amber readouts for time left and the whole queue,
  Power / Busy / Fault LEDs
- Prompts are written in marker on strips of masking tape
- FLUX.1 Schnell on the fast engine (4 steps): 3-bit for 8 GB Macs, 4-bit and 8-bit for bigger ones; FLUX LoRAs and live previews
- Pony, Illustrious and NoobAI SDXL models are recognised: labelled, their LoRAs listed first, and their quality tags added automatically (can be switched off)
- Train SDXL and Pony / Illustrious / NoobAI LoRAs: the built-in SDXL trains on its full-precision original; time estimates per model type
- Train ERNIE LoRAs on the fast engine (on top of the 4-bit model; experimental on 8 GB Macs)
- Settings: choose the downloaded-models folder (e.g. an external drive) and move existing downloads there
- Models tab: download any model ahead of time (with progress), including ones too big for this Mac
- Settings → Look: Dark (black anodised, the default), Light (brushed aluminium with orange rack ears and trim)
  or Match macOS
- How to / About is now a printed operator's manual taped to the rack: dark ink on paper, easy to read in either look
- Knob readouts take exact typed values (e.g. CFG 12.3); the knobs turn in steps
- Library: search by seed (seed box, or type a number in the search box); "find all with this seed" in the picture details

## 0.2.0

- Long prompts and `(word:1.3)` weights, wildcards, prompt lists, random LoRAs
- Hi-res fix, face detailer, image prompts (IP-Adapter), comparison grid
- Inpainting, expanding (outpainting), OpenPose ControlNet, Depth Anything V2 depth maps
- CLIP skip with per-model defaults; edit in PhotoCraft or any editor
- Downloads work behind per-app firewalls (curl fallback)
- Security: Electron 44, scoped IPC handlers, deny-by-default permissions, strict CSP in the packaged app,
  Electron fuses, verified Python download, pinned dependencies, no file paths in picture metadata,
  redacted logs, no personal paths in builds, DMG checksums
- Apache-2.0 licence; third-party notices

## 0.1.0

- First test version: SD 1.5, SDXL and ERNIE-Image, LoRAs, presets, library, DiffusionBee import, training, tools
