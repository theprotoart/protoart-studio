# Changelog

## 1.2.0

- 📝 Image → prompt "Describe": Qwen3.5 4B (Apache-2.0; `mlx-community/Qwen3.5-4B-MLX-4bit`, pinned revision 32f3e8e,
  ~3 GB) through mlx-vlm 0.7.4 in its own environment (`~/.protoartstudio/venv-vlm`, `backend/vlm.py`); a short-lived
  worker (`vlm_worker.py`) per picture so its memory is freed at once. Florence-2 is now "Quick"; the WD tagger "Tags".
  The Tools module shows the written prompt
- Header: a chrome bat-handle TIPS toggle (ON / OFF) beside bigger Power / Busy / Fault lamps
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
