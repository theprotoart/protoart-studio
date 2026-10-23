/* Protoart Studio renderer. Talks to the Python backend through window.bee (see preload.js). */
const { createApp } = Vue;

// ------------------------------------------------------------------ backend client
let nextId = 1;
const pending = new Map();
const listeners = [];
const HEAVY = new Set(["ckpt_merge", "lora_extract", "faceswap", "tiled_upscale", "quantize", "ckpt_convert", "heal", "edit_words", "vae_get", "vectorize", "preload", "nsfw_scan", "downloads_move", "model_download", "generate", "postprocess", "interrogate", "enhance_prompt", "train_lora", "train_embedding", "lora_merge", "train_autocaption",
                        "control_preview", "convert_tdict", "compare"]);
const SOUND_JOBS = new Set(["generate", "edit_words", "heal", "postprocess", "train_lora", "train_embedding", "lora_merge", "vectorize"]);
const jobArgs = new Map(); // queue job id -> {cmd, args}, so jobs can be re-queued after a Force stop

window.bee.onMessage((m) => {
  if ((m.type === "result" || m.type === "error") && pending.has(m.id)) {
    const { resolve, reject } = pending.get(m.id);
    const job = jobArgs.get(m.id);
    if (job && window.ProtoSound) {  // 🔔 a chime when pictures or training are finished, a low tone on failure
      if (m.type === "error" && !m.cancelled && SOUND_JOBS.has(job.cmd)) window.ProtoSound.play("error");
      else if (m.type === "result" && /^train_(lora|embedding)$/.test(job.cmd)) window.ProtoSound.play("trained");
      else if (m.type === "result" && SOUND_JOBS.has(job.cmd)) window.ProtoSound.play("done");
    }
    pending.delete(m.id);
    jobArgs.delete(m.id);
    // a job that stopped because a first-time download didn't happen: the app offers ⬇ Download it now
    if (m.type === "error" && job && /one-time download/i.test(m.msg || "")) listeners.forEach((fn) => fn({ type: "download_needed", job }));
    if (m.type === "result") resolve(m.data);
    else reject(Object.assign(new Error(m.msg), { cancelled: !!m.cancelled }));
    return;
  }
  listeners.forEach((fn) => fn(m));
});

function call(cmd, args = {}) {
  const id = nextId++;
  // Vue's reactive objects can't cross Electron's process boundary, so always send a plain copy.
  const plainArgs = JSON.parse(JSON.stringify(args ?? {}));
  if (HEAVY.has(cmd)) jobArgs.set(id, { cmd, args: plainArgs });
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    window.bee.send({ id, cmd, args: plainArgs }).then(
      (ok) => {
        if (!ok) {
          pending.delete(id);
          reject(new Error("Backend is not running"));
        }
      },
      (err) => {
        pending.delete(id);
        reject(err);
      }
    );
  });
}

// a quick command that waits on a website: give up after ms so a button can never stay stuck on "Checking…"
function callWithin(ms, cmd, args = {}) {
  return Promise.race([call(cmd, args), new Promise((_, rej) => setTimeout(() =>
    rej(new Error("No answer from the website after " + Math.round(ms / 1000) + " s. Check your internet connection, then press Retry.")), ms))]);
}

const STOP_GRACE_MS = 2500; // after Stop, how long a job may take to stop by itself before it's force-stopped

// Create / Image to image settings as they are on a fresh install (also used by "reset all")
function defaultPost() {  // Tools settings on a fresh install
  return {
    depth: { model: "small", style: "gray" },
    face_strength: 0.8,
    face_model: "codeformer",   // or "restoreformer"
    face_fidelity: 0.5,            // CodeFormer only
    do_enhance: true,
    do_faces: false,
    enh: { upscale_factor: 2, num_inference_steps: 18, denoise_strength: 0.35, controlnet_scale: 0.6,
           solver: "DDIM", seed: -1, prompt: "", turbo: false, instant: false, work_size: "auto" },
    vec: { colors: 8, detail: "medium", smooth: true },   // ✒️ Vectorize
    det: { lora: "", weight: 0.8, strength: 0.35, steps: 20 },  // 🔎 Detail with a LoRA (patchbay module)
  };
}

function defaultForm() {
  return { model: "sdxl", prompt: "", negative_prompt: "", width: 1024, height: 1024, steps: 30, cfg: 6,
    sampler: "", seed: -1, num_images: 1, loras: [],
    var_seed: -1, var_strength: 0,  // 🎲 Small change (variation seed): 0 = off
    seamless: false,  // 🧱 Seamless: the picture tiles
    regions: { on: false, layout: "cols", ratios: "1,1", prompts: ["", ""], base: 0.2, add_main: false, refine: 0.5 },  // 🗺 Regional prompts
    swap: { face: "", which: "all", boost: 1, restore: true, consent: false },  // 🔁 Face swap (Image to image)
    vae: "",
    restore_faces: false, face_strength: 0.8, face_model: "codeformer", face_fidelity: 0.5, enhance_prompt: false, clip_skip: 1, quality_tags: true,
    init_image: "", strength: 0.6, use_init_size: false, size_link: false, size_ratio: 1,
    fast: false, fast_prev: null, i2i_use_init: true, control_image: "",
    control: { type: "", scale: 1.0, start: 0.0, end: 1.0, already_map: false },
    hires: { on: false, scale: 1.5, strength: 0.45, steps: 15 },
    face_detailer: { on: false, strength: 0.4, prompt: "" },
    ip_adapter: { image: "", kind: "style", scale: 0.6 },
    heal_method: "lama",
    edit_refs: [],      // ✨ Edit with words: extra pictures to take a style, object or outfit from
    i2i_mode: "change", inpaint: { strength: 0.85, blur: 4, only_masked: false },
    outpaint: { left: 0, right: 0, top: 0, bottom: 0 },
    outpaint_engine: "klein" };  // ↔ Expand fills with FLUX.2 Klein + outpaint LoRA ("klein") or the model at the top ("model")
}

const FLAVOR_LABEL = { pony: "Pony", illustrious: "Illustrious", noobai: "NoobAI" };
const QUALITY_TAGS = {
  pony: "score_9, score_8_up, score_7_up",
  illustrious: "masterpiece, best quality, amazing quality, very aesthetic, absurdres",
  noobai: "masterpiece, best quality, newest, absurdres, highres",
};
const QUALITY_NEG = {
  pony: "score_6, score_5, score_4",
  illustrious: "worst quality, low quality, bad quality, lowres",
  noobai: "worst quality, old, early, low quality, lowres",
};
// Recommended samplers per model type: quickest to a clean picture first (fewest model runs), then best quality.
// (Pony / Illustrious / NoobAI were trained and tuned with Euler a, so it leads their quality list.)
const SAMPLER_PICKS = {
  sd15: { fast: ["unipc", "dpmpp_2m_karras", "deis"], quality: ["dpmpp_2m_karras", "dpmpp_sde_karras", "dpmpp_2m_sde_karras"] },
  sdxl: { fast: ["dpmpp_2m_karras", "unipc", "deis"], quality: ["dpmpp_2m_sde_karras", "dpmpp_2m_karras", "euler_a"] },
  anime_xl: { fast: ["euler_a", "dpmpp_2m_karras", "unipc"], quality: ["euler_a", "dpmpp_2m_sde_karras", "dpmpp_2m_karras"] },
  ernie: { fast: ["flow_euler"], quality: ["flow_euler_beta", "flow_euler_karras"] },
  flux: { fast: ["flow_euler"], quality: ["flow_euler_beta", "flow_euler_karras"] },
  flux2: { fast: ["flow_euler"], quality: [] },
};
const LICENCES = [{"model": "Stable Diffusion 1.5", "licence": "CreativeML OpenRAIL-M", "note": ""}, {"model": "Stable Diffusion XL", "licence": "CreativeML OpenRAIL++-M", "note": ""}, {"model": "ERNIE-Image (and its community builds)", "licence": "Apache-2.0", "note": ""}, {"model": "FLUX.1 [schnell]", "licence": "Apache-2.0", "note": ""}, {"model": "FLUX.2 [klein] 4B", "licence": "Apache-2.0", "note": "the 4-bit build's page also carries a non-commercial tag: check it before selling pictures"}, {"model": "Qwen3.5 4B (picture reader)", "licence": "Apache-2.0", "note": ""}, {"model": "Qwen2.5 (prompt enhancer)", "licence": "Apache-2.0", "note": ""}, {"model": "Florence-2 (Quick describe)", "licence": "MIT", "note": ""}, {"model": "WD v3 tagger (Danbooru tags)", "licence": "Apache-2.0", "note": ""}, {"model": "Depth Anything V2 Small", "licence": "Apache-2.0", "note": ""}, {"model": "Depth Anything V2 Base / Large", "licence": "CC BY-NC 4.0", "note": "non-commercial"}, {"model": "CodeFormer (face restore)", "licence": "S-Lab License 1.0", "note": "non-commercial"}, {"model": "RestoreFormer (face restore)", "licence": "see the GFPGAN repository", "note": ""}, {"model": "LaMa (Smart fill)", "licence": "Apache-2.0", "note": ""}, {"model": "Juggernaut Reborn (Enhance)", "licence": "CreativeML OpenRAIL-M", "note": ""}, {"model": "4x-UltraSharp (Enhance upscaler)", "licence": "CC BY-NC-SA 4.0", "note": "non-commercial"}, {"model": "IP-Adapter", "licence": "Apache-2.0", "note": ""}, {"model": "ControlNet 1.1", "licence": "CreativeML OpenRAIL-M", "note": ""}, {"model": "LCM-LoRA (⚡ fast mode)", "licence": "CreativeML OpenRAIL++-M", "note": ""}, {"model": "Anima / Anima Turbo (CircleStone Labs)", "licence": "CircleStone Labs Non-Commercial License + NVIDIA Open Model License", "note": "non-commercial: the model may not be used commercially; pictures made with it may"}, {"model": "Xinsir ControlNet Union ProMax (SDXL)", "licence": "Apache-2.0", "note": ""}, {"model": "inswapper_128 (Face swap)", "licence": "InsightFace: non-commercial research only", "note": "non-commercial; only faces you have permission to use"}, {"model": "ArcFace w600k_r50 (Face swap)", "licence": "InsightFace: non-commercial research only", "note": "non-commercial"}, {"model": "Models and LoRAs from Civitai or your own files", "licence": "whatever their authors chose", "note": "check each model's page"}];
const LINKS = { discord: "https://discord.gg/zRC6zrufYE", issues: "https://github.com/theprotoart/protoart-studio/issues/new", repo: "https://github.com/theprotoart/protoart-studio", notices: "https://github.com/theprotoart/protoart-studio/blob/main/THIRD_PARTY_NOTICES.md", civitai: "https://civitai.com/user/theprotoartent677" };
const I2I_KINDS = ["img2img", "healed", "word_edit", "inpaint", "outpaint", "controlnet", "faceswap"];  // pictures made in Image to image
// Tools → Tiled upscale presets (size, how much new detail, steps)
const TILED_PRESETS = [
  { id: "gentle", label: "Gentle", tip: "Sharper and bigger, almost no change: photos and faces", set: { scale: 2, strength: 0.2, steps: 24 } },
  { id: "detail", label: "Detail", tip: "Adds texture (skin, fabric, foliage) and keeps the picture", set: { scale: 2, strength: 0.25, steps: 24 } },
  { id: "creative", label: "Creative", tip: "Redraws more: new small details, may change little things", set: { scale: 2, strength: 0.4, steps: 28 } },
  { id: "print", label: "Print", tip: "3× for printing; slow on 8 GB Macs", set: { scale: 3, strength: 0.25, steps: 24 } },
  { id: "quick", label: "Quick", tip: "1.5× with few steps: a fast check", set: { scale: 1.5, strength: 0.25, steps: 12 } },
];
const FAMILY_LABEL = { sd15: "SD 1.5", sdxl: "SDXL", ernie: "ERNIE", flux: "FLUX", flux2: "Klein 4B", anima: "Anima", sd2: "SD 2" };  // FLUX.2 [klein] 4B: named by the model, so its LoRAs and models match
const OPTS_KEY = "beestudio.opts";
const PERF_KEY = "protoart.perf";
// Tools → Patchbay: the back of the rack. Each module has an IN and an OUT jack; a cable from one module's OUT to the
// next one's IN sets the order the picture goes through them (like an analog patchbay, or the back of Reason's rack).
// Image to image → ControlNet: a short guide for each kind (built-in ones by id; your own files by words in the name):
// what it does, which models suit it (matched against your installed models' names), how to use it, and the
// settings it works best with (applied with one click).
const CN_GUIDES = {
  canny: { title: "Canny edges", what: "Keeps the outlines of your picture exactly and lets everything else change: colours, materials, style, lighting.",
    picture: "Any picture with clear shapes (a photo, a product, a building, a drawing). Preview the map: you should see clean white outlines.",
    models: null, modelsText: "Any SD 1.5 model: pick the one with the style you want the result in.",
    steps: ["Put the picture in 1 · Picture. Switch <b>Start from this picture</b> on to keep its colours too, off for a fresh look.",
      "Describe the new version (“the same room as a cosy watercolour”).", "Strength 0.8–1.0, The whole time."],
    set: { scale: 0.9, start: 0, end: 1 } },
  depth: { title: "Depth", what: "Keeps the 3-D layout (what's near and far) and lets shapes, materials and style change more freely than Canny.",
    picture: "Any photo or render with depth. Bigger depth models (Base, Large) read fine detail better but are slower.",
    models: null, modelsText: "Any SD 1.5 model.", steps: ["Put the picture in 1 · Picture.", "Describe the new scene in the same layout.", "Strength 0.8–1.0, Mostly."],
    set: { scale: 0.9, start: 0, end: 0.8 } },
  lineart: { title: "Line art", what: "Colours and renders a drawing while keeping its lines.", picture: "A clean drawing or sketch (dark lines on white).",
    models: /(anime|anything|counterfeit|toon|cartoon|dream ?shaper)/i, modelsText: "Anime or illustration SD 1.5 models for comics and cartoons; realistic ones to turn a drawing into a photo.",
    steps: ["Put the drawing in 1 · Picture, <b>Start from this picture</b> off.", "Describe colours and style (“flat colours, cel shading”).", "Strength 1.0, The whole time."],
    set: { scale: 1, start: 0, end: 1, use_init: false } },
  scribble: { title: "Scribble", what: "Turns a rough doodle into a finished picture.", picture: "A quick sketch: dark strokes on a light background (a few lines is enough).",
    models: null, modelsText: "Any SD 1.5 model.", steps: ["Put the scribble in 1 · Picture, <b>Start from this picture</b> off.", "Describe what the doodle is.", "Strength 0.8–1.0, Layout only for freer results."],
    set: { scale: 0.9, start: 0, end: 0.8, use_init: false } },
  softedge: { title: "Soft edges", what: "Like Canny but looser: keeps the main shapes and lets small details change.", picture: "Any picture.",
    models: null, modelsText: "Any SD 1.5 model.", steps: ["Put the picture in 1 · Picture.", "Describe the new version.", "Strength 0.8, Mostly."],
    set: { scale: 0.8, start: 0, end: 0.8 } },
  openpose: { title: "OpenPose", what: "Copies a person's pose (body, arms, legs) into a new picture; faces, clothes and setting come from your prompt.",
    picture: "A photo with the whole person visible. Preview the map to check the stick figure found everyone.",
    models: null, modelsText: "Any SD 1.5 model with people in it (realistic or anime).", steps: ["Put the photo in 1 · Picture, <b>Start from this picture</b> off.", "Describe the person and setting.", "Strength 1.0, The whole time."],
    set: { scale: 1, start: 0, end: 1, use_init: false } },
  tile: { title: "Tile", what: "Adds detail while keeping the picture: sharpening, upscaling, cleaning up a blurry picture.", picture: "The picture you want more detail in.",
    models: null, modelsText: "The model that made the picture (or one in the same style).", steps: ["Put the picture in 1 · Picture with <b>Start from this picture</b> on.", "Change strength 0.3–0.5 (low keeps it the same).", "ControlNet strength 1.0, The whole time; describe the picture briefly."],
    set: { scale: 1, start: 0, end: 1, use_init: true, strength: 0.4 } },
};
const CN_NAME_GUIDES = [];  // your own ControlNets matched to a guide by name
// CodeFormer starting points (fidelity = how much of the person's own face is kept, strength = how much of the
// repaired face is blended over the original). Upscaling is the ENHANCE module's job, so it isn't part of these.
const FACE_PRESETS = [
  { id: "balanced", label: "Balanced", fidelity: 0.5, strength: 1.0, tip: "Good default for most generated portraits" },
  { id: "identity", label: "Keep likeness", fidelity: 0.8, strength: 1.0, tip: "Keeps the likeness of a real person, or a face from a person LoRA" },
  { id: "heavy", label: "Heavy repair", fidelity: 0.3, strength: 1.0, tip: "Fixes badly distorted or tiny faces (small faces in wide shots)" },
  { id: "subtle", label: "Subtle touch-up", fidelity: 0.7, strength: 0.6, tip: "Light clean-up blended with the original, so skin texture survives" },
  { id: "old", label: "Old photo", fidelity: 0.4, strength: 1.0, tip: "Scanned or worn photographs" },
  { id: "gentle", label: "Gentle", fidelity: 0.85, strength: 1.0, tip: "Faces that are already clear and just need a polish" },
];

const PATCH_NODES = [
  { id: "src", label: "PICTURE", sub: "your picture", out: true },
  { id: "heal", label: "REMOVE", sub: "objects", icon: "🩹", in: true, out: true },
  { id: "faces", label: "FACES", sub: "restore", icon: "🙂", in: true, out: true },
  { id: "enhance", label: "ENHANCE", sub: "& upscale", icon: "✨", in: true, out: true },
  { id: "depth", label: "DEPTH", sub: "map", icon: "🌊", in: true, out: true },
  { id: "vector", label: "VECTOR", sub: "SVG", icon: "✒️", in: true, out: true },
  { id: "detail", label: "DETAIL", sub: "with a LoRA", icon: "🔎", in: true, out: true },
  { id: "result", label: "RESULT", sub: "to Library", in: true },
];
const PATCH_X = (i) => 78 + i * 140;           // jack columns in the panel's own 1000-wide drawing
const PATCH_IN_Y = 118, PATCH_OUT_Y = 206;
const CABLE_COLOURS = ["#c8372d", "#e0882a", "#2f7fc1", "#3d9a4b", "#d9b52c", "#8a4fb5"];
const PATCH_DEFAULT = [];  // the patchbay starts with no cables: you plug in what you want
const PATCH_EXAMPLE = [{ from: "src", to: "enhance", c: 0 }, { from: "enhance", to: "faces", c: 1 }, { from: "faces", to: "result", c: 2 }];
// Starting guesses (seconds per step per megapixel) until this Mac has real measurements.
const DEFAULT_SPS_PER_MP = { sd15: 2.4, sdxl: 3.5, ernie: 9, flux: 12, flux2: 7 };
const DEFAULT_LOAD_S = { sd15: 15, sdxl: 30, ernie: 60, flux: 60, flux2: 40 };
const DEFAULT_JOB_S = { faces: 15, enhance: 180, interrogate: 20, enhance_prompt: 25 };
const FORM_KEY = "beestudio.lastForm";
// Pictures you can drop or pick anywhere: these open directly; HEIC, TIFF, RAW… are converted by macOS first
const DIRECT_EXT = ["png", "jpg", "jpeg", "jpe", "jfif", "webp", "gif", "bmp"];
const PICTURE_EXT = DIRECT_EXT.concat(["heic", "heif", "hif", "avif", "tif", "tiff", "psd", "jp2", "j2k", "jpx", "tga", "ico", "icns",
  "exr", "hdr", "dng", "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "raf", "orf", "rw2", "pef", "srw", "x3f", "3fr",
  "erf", "kdc", "mrw", "raw", "rwl", "iiq"]);
const extOf = (name) => String(name || "").split(".").pop().toLowerCase();
const isPicture = (f) => !!f && ((f.type || "").startsWith("image/") || PICTURE_EXT.includes(extOf(f.name)));
const ADJ_KEY = "protoart.adjustLast";  // the last Adjust settings you saved ("Last used")
let adjEngine = null;                   // the WebGL engine (kept outside Vue: not reactive)
let adjFrame = 0;
// Adjust panel: Camera Raw-style sliders (Light, Color, Presence) and the Color Efex-style filter stack
const ADJ_SECTIONS = [
  { id: "light", label: "Light", tip: "Like Camera Raw's Light panel. Double-click a slider to reset it.", sliders: [
    { k: "exposure", label: "Exposure", min: -2, max: 2, step: 0.05, tip: "Overall brightness, in stops" },
    { k: "contrast", label: "Contrast", tip: "Difference between light and dark" },
    { k: "highlights", label: "Highlights", tip: "Left recovers bright areas; right brightens them" },
    { k: "shadows", label: "Shadows", tip: "Right opens up dark areas; left deepens them" },
    { k: "whites", label: "Whites", tip: "Where pure white starts" },
    { k: "blacks", label: "Blacks", tip: "Where pure black starts" } ] },
  { id: "color", label: "Color", tip: "White balance and colour strength.", sliders: [
    { k: "temperature", label: "Temperature", tip: "Cooler (blue) ↔ warmer (yellow)" },
    { k: "tint", label: "Tint", tip: "Green ↔ magenta" },
    { k: "vibrance", label: "Vibrance", tip: "Boosts muted colours and protects skin tones" },
    { k: "saturation", label: "Saturation", tip: "All colours; far left = black and white" } ] },
  { id: "presence", label: "Presence & effects", tip: "Lightroom's Clarity, Sharpening and Vignette.", sliders: [
    { k: "clarity", label: "Clarity", tip: "Mid-tone contrast: crisper texture (left softens)" },
    { k: "sharpen", label: "Sharpening", min: 0, max: 1, tip: "Fine-edge sharpening" },
    { k: "vignette", label: "Vignette", tip: "Darkens the corners (left lightens them)" } ] },
];
const ADJ_FILTERS = {
  extract: { label: "Detail Extractor", on: "extractOn", tip: "Pulls out texture everywhere, an HDR-like look.", sliders: [
    { k: "deAmount", label: "Detail", min: 0, max: 1 }, { k: "deContrast", label: "Contrast" }, { k: "deSaturation", label: "Saturation" }],
    seg: { k: "deRadius", label: "Size", options: [[0, "Fine"], [1, "Normal"], [2, "Coarse"]] } },
  dark: { label: "Dark Contrasts", on: "darkOn", tip: "Deeper, moodier tones with crisp mid-tone detail.", sliders: [
    { k: "dcStrength", label: "Strength", min: 0, max: 1 }, { k: "dcBrightness", label: "Brightness" }, { k: "dcSaturation", label: "Saturation" }] },
  bleach: { label: "Bleach Bypass", on: "bleachOn", tip: "Silvery, high-contrast, half-desaturated film look.", sliders: [
    { k: "bbBrightness", label: "Brightness" }, { k: "bbSaturation", label: "Saturation" }, { k: "bbContrast", label: "Contrast" },
    { k: "bbLocalContrast", label: "Local contrast", min: 0, max: 1 }] },
  glow: { label: "Glamour Glow", on: "glowOn", tip: "A soft, glowing bloom: flattering for portraits.", sliders: [
    { k: "ggGlow", label: "Glow", min: 0, max: 1 }, { k: "ggSaturation", label: "Saturation" }, { k: "ggWarmth", label: "Warmth" }] },
  grain: { label: "Film Grain", on: "grainOn", tip: "Photographic grain. Usually best last in the stack.", sliders: [
    { k: "grainAmount", label: "Intensity", min: 0, max: 1 }, { k: "grainSize", label: "Grain size", min: 0, max: 1 },
    { k: "grainSoftness", label: "Softness", min: 0, max: 1 }], toggle: { k: "grainColor", off: "Mono", on: "Color" } },
};
const CLIP_KEY = "protoart.clipSkip"; // per-model CLIP skip you picked (overrides the recommendation)

function loadJSON(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: settings just won't persist */
  }
}

// Launch screen: stays at least ~1.5 s, then fades out once the engine has answered.
const SPLASH_STARTED = Date.now();
function hideSplash(msg) {
  const el = document.getElementById("splash");
  if (!el || el.dataset.hiding) return;
  if (msg) {
    const st = document.getElementById("splash-status");
    if (st) st.textContent = msg;
  }
  el.dataset.hiding = "1";
  setTimeout(() => {
    el.classList.add("ready");  // READY lights, the needle settles, then the unit fades into the app
    setTimeout(() => {
      el.classList.add("gone");
      setTimeout(() => el.remove(), 600);
    }, 450);
  }, Math.max(0, 1900 - (Date.now() - SPLASH_STARTED)));
}
setTimeout(() => hideSplash("Taking a while…"), 25000); // never block the app forever

const protoartApp = createApp({
  // Packaged app: the page template is compiled at build time (packaging/precompile_template.js),
  // so Vue's runtime-only build runs under a strict Content Security Policy (no eval).
  ...(window.__PROTOART_RENDER__ ? { render: window.__PROTOART_RENDER__ } : {}),
  data() {
    return {
      facePresets: FACE_PRESETS,
      tab: "create",
      theme: window.RackTheme ? window.RackTheme.get() : "dark", // faceplate: light / dark / system (rack.js)
      skin: window.RackTheme && window.RackTheme.getSkin ? window.RackTheme.getSkin() : "studio", // theme colours (rack.js)
      SKIN_LIST: [
        { id: "studio", label: "Amber", title: "Amber: brushed metal and charcoal panels with amber lamps (the classic look)" },
        { id: "console", label: "Cream", title: "Cream: warm cream and muted metal, yellow lamps and deep red accents" },
        { id: "puig", label: "Slate", title: "Slate: blue-grey steel, cream lettering and soft ivory lamps" },
        { id: "magma", label: "Ember", title: "Ember: black panels with red-orange lamps" },
      ],
      sys: {},
      models: [],
      samplers: [],
      loras: [],
      i2iLora: { q: "", open: false },
      lorasAside: loadJSON("protoart.lorasAside", {}),  // model type -> LoRAs put aside when you switched type
      frame: { size: 512, busy: false, msg: "", copied: null },  // Train tab: square framing of the pictures
      cropEd: null,
      slowWarn: null,      // the "this will be slow on this Mac" box: {title, lines, fixes, id, resolve, mute}
      // Train tab → ✂ Crop by hand: {it, x, y, s, w, h} (x, y, s as in trainer.parse_box)
      forcing: false,
      stopAllAt: 0,
      sentBack: {},          // job id -> {cmd, args}: moved to the back of the queue (re-sent once it has stopped)
      memLow: null,          // other apps have the memory: shown above the pictures until closed
      memLowShown: false,
      presets: [],
      selectedPreset: "",
      newPresetName: "",
      presetIncludesPrompt: false,
      loraToAdd: "",
      loraSearch: "",
      loraOpen: loadJSON("protoart.loraOpen", true),  // Create's LoRA list shows until you close it (remembered)
      logCopied: false,
      resetAsk: "",
      edits: { pending: [], timer: null, notInstalled: false, folder: "" },
      editorChoices: ["PhotoCraft", "Adobe Photoshop 2025", "Pixelmator Pro", "Affinity Photo 2", "GIMP", "Preview"],
      wild: { open: false, files: [], folder: "", preview: [] },
      i2iFolds: loadJSON("protoart.i2iFolds", {}),  // Image to image cards folded down to their title
      ptool: "",  // Create → the prompt tools row: which small panel is open ("enhance", "saved", "image", "random")
      batch: { open: false, text: "", keepPrompt: false },
      loraRand: Object.assign({ min: 1, max: 2, wmin: 0.6, wmax: 1.0, every: false, knownOnly: true, picked: [], words: [] },
                              loadJSON("protoart.loraRand", {})),
      ipDragOver: false,
      ipKinds: [
        { id: "style", label: "Style & look", tip: "Borrows the overall look: colours, style, mood, subject" },
        { id: "plus", label: "Stronger", tip: "Follows the reference picture more closely (IP-Adapter Plus)" },
        { id: "face", label: "Face", tip: "Borrows a person's face from a close-up photo (IP-Adapter Plus Face)" },
      ],
      mask: { erase: false, size: 40, data: null, for: "", painting: false, last: null,
              tool: "brush", subtract: false, tolerance: 32 },
      sel: null,           // a selection being drawn: {tool, pts, start, end, hover, sub}
      SEL_TOOLS: [
        { id: "brush", icon: "🖌", label: "Brush", tip: "Paint the area (B)" },
        { id: "erase", icon: "🧽", label: "Erase", tip: "Rub out paint (E)" },
        { id: "lasso", icon: "➰", label: "Lasso", tip: "Draw around an area freehand; it fills when you let go (L)" },
        { id: "poly", icon: "⬠", label: "Polygon", tip: "Polygonal lasso: click the corners for straight edges; click the first point or double-click to close (P)" },
        { id: "wand", icon: "🪄", label: "Wand", tip: "Magic wand: click a colour to pick the connected area of similar colour (W)" },
        { id: "object", icon: "⬚", label: "Object", tip: "Object selection: drag a box around something and its outline is found (O)" },
      ],
      brush: { show: false, x: 0, y: 0 },
      objFinding: false,   // object selection is working out the outline
      soundsOn: window.ProtoSound ? window.ProtoSound.enabled : true,
      LICENCES, TILED_PRESETS,
      loraSort: loadJSON("protoart.loraSort", "az"),   // LoRA list order
      loraKind: "",                                      // LoRA list filter ('' = all)   // the brush-size ring that follows the pointer while painting
      initDims: null,
      resetFlip: false,      // the Reset all lever, while it's flipped (it springs back by itself)
      resetUndo: null,       // Create settings before "Reset to defaults", for one undo
      quitAsk: false,        // Tools -> Force quit: waiting for confirmation
      engineExits: [],       // when the engine stopped unexpectedly (to avoid restart loops)
      cmp: { open: false, x: "cfg", xv: "4, 6, 8, 10", y: "", yv: "", lora: "" },
      cmpAxes: { cfg: "Guidance (CFG)", steps: "Steps", sampler: "Sampler", seed: "Seed", clip_skip: "CLIP skip",
                 lora_weight: "LoRA weight", lora: "Which LoRA", strength: "Change strength (Image to image)" },
      form: (() => {
        const f = Object.assign(defaultForm(), loadJSON(FORM_KEY, {}));
        f.regions = { ...defaultForm().regions, ...(f.regions || {}) };
        f.swap = { ...defaultForm().swap, ...(f.swap || {}), consent: false };  // permission is asked again each time the app opens
        if (!loadJSON("protoart.cfDefault", false)) {  // 1.2: CodeFormer became the default face model (once)
          f.face_model = "codeformer";
          saveJSON("protoart.cfDefault", true);
        }
        return f;
      })(),
      dims: [512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1536],
      queue: [],          // jobs from the backend: running first, then queued
      progress: null,
      status: "Starting…",
      error: "",
      dlJob: null,
      detLoraQ: "",
      gmenu: null,
      regionBusy: -1, regionUndo: {},  // 🗺 ✨ enhance an area's prompt      // right-click menus for models, LoRAs, the queue and folders (pictures have their own)
      i2iSuggesting: false,  // Image to image → 💡 suggest from the picture     // Tools → Detail with a LoRA: search box      // the job that stopped for a missing first-time download (⬇ Download it now)
      dlBusy: false,
      session: [],
      lib: { text: "", seed: "", model: "", kind: "", starred: false, loras: [], results: [] },
      dragOver: false,
      stopping: {},          // job id -> time Stop was pressed
      strictOffline: false,  // Settings → Privacy → Strict offline (never go online)
      paused: false,         // Pause: the running image waits at its next step; no new job starts
      pausedAt: 0,
      requeue: [],           // jobs to re-send after a Force stop restart
      libHasMore: false,
      startDragOver: false, editRefOver: false,
      lastEnhanced: null,
      undoPrompt: null,
      timing: { job: null, image: 0, lastAt: 0, stepTimes: [] },
      jobStartedAt: 0,
      firstStepAt: 0,
      perf: loadJSON(PERF_KEY, { models: {}, jobs: {} }),
      lastModel: null,
      now: Date.now(),
      i2p: Object.assign({ mode: "describe", append: false }, loadJSON("beestudio.i2p", {})),
      rnd: Object.assign({ subject: "random", quality: "standard", detail: 2, style: true, every: false }, loadJSON("beestudio.rnd", {})),
      post: defaultPost(),
      ask: null,             // the "Are you sure?" box (confirmAsk)
      promptPresets: loadJSON("protoart.prompts", []),   // 📒 My prompts
      promptIoMsg: "",
      qnt: { model: "", bits: 8, msg: "" },  // Tools → Smaller copy
      tup: { scale: 2, strength: 0.25, steps: 24, prompt: "" },  // Tools → Tiled upscale
      mrg: { a: "", b: "", c: "", mode: "sum", m: 0.5, name: "", msg: "" },  // Tools → Model merger
      cvt: { to: "copy", model: "", base: "", rank: 32, te: true, name: "", msg: "" },  // Tools → Convert into (LoRA…)
      chain: { steps: loadJSON("protoart.chain", []).map((x) => ({ lora: "", lora_weight: 1, extra: "", ...x, result: "", q: "" })),
               running: false, stopped: false, at: -1, done: 0 },  // 🔗 Model chain
      swapOver: false,
      promptBox: null,       // the 📒 My prompts window: {name} = saving, {name: null} = managing
      undoNegative: null,
      lastAutoNegative: "",
      unlocked: {},          // model rows unlocked for Remove (locked by default)
      vecResult: null,       // the last ✒️ Vectorize result {path, shapes, colors, bytes}
      toolsUndo: null,
      libLoras: [],
      detail: null,
      selectMode: false,
      selected: [],
      confirmDel: null,   // {ids, count, versions, include_versions}
      imagesDir: { path: "", default: "" },
      moveExisting: true,
      opts: Object.assign({ low_memory: false, force_load: false, trash: false, image_format: "png", jpeg_quality: 92,
                            auto_triggers: true, previews: true, editor: "PhotoCraft", embed_meta: "details",
                            // speed: get a model ready before you press Generate ("last" = the one you last used)
                            preload_start: "last", preload_on_pick: true,
                            // content filter (off until you turn it on)
                            hide_nsfw: false, check_pictures: false, hide_people: false, sfw_mode: false, use_inpaint_model: true,
                            tooltips: true,       // quick tips when pointing at buttons (Settings → Look)
                            default_model: "" },   // the model Create starts with (Models tab → ☆ Default)
        loadJSON(OPTS_KEY, {})),
      dbee: { available: false, enabled: true, checkpoints: 0, loras: 0, embeddings: 0, skipped: 0, skipped_kinds: [] },
      // Enhance & upscale speed/quality presets. Only steps × creativity steps actually run (it refines the
      // picture rather than starting from noise); DPM-Solver gets sharp results in fewer steps than DDIM.
      enhSpeed: [
        { label: "⏩ Instant", steps: 0, solver: "DDIM", instant: true, tip: "The 4× sharpening upscaler only: seconds instead of minutes, but it invents no new detail. Good for previews and big batches." },
        { label: "🚀 Turbo", steps: 6, solver: "DDIM", turbo: true, tip: "A distilled few-step add-on (LCM-LoRA, ~135 MB once): a few steps and no negative-prompt pass, several times faster. Slightly softer detail." },
        { label: "⚡ Fast", steps: 10, solver: "DPMSolver", tip: "Fewer steps that still look clean: about half the time of Standard" },
        { label: "Standard", steps: 18, solver: "DDIM", tip: "The enhancer's own default" },
        { label: "★ Best", steps: 30, solver: "DPMSolver", tip: "Most detail and the cleanest textures; about 1.7× the time of Standard" },
      ],
      enhPresets: [
        { label: "Subtle", c: 0.2, r: 1.0, tip: "Sharper and cleaner, almost no new detail" },
        { label: "Balanced", c: 0.35, r: 0.6, tip: "The default: adds detail, keeps the picture" },
        { label: "Creative", c: 0.55, r: 0.45, tip: "Lots of new texture and detail; may change small things" },
      ],
      livePreview: null,
      // Adjust (sliders + filters after generating): {open, rec, p, before, busy, msg, error, openFilter, size}
      adj: { open: false, rec: null, p: null, before: false, busy: false, msg: "", error: "", openFilter: "", size: null,
             fold: {}, which: "original", view: "after", split: 0.5, presetName: "", savingPreset: false,
             // several versions at once: which settings ("current" and/or preset names), enhance too, and in which order
             batch: ["current"], batchEnhance: false, batchOrder: "enhance_first" },
      adjPresets: [],   // your saved Adjust presets: [{name, set}]
      // Library → Batch edit & export (PhotoMill-style) with Press compression
      bx: { open: false, ids: [], adjust: "", format: "same", quality: 90, compress: "balanced", dpi: 72, keep_meta: false,
            pattern: "{name}", dest: "", resize: { mode: "none", value: 2048, w: 1080, h: 1350, no_upscale: true },
            running: false, done: 0, total: 0, results: [], errors: [], cancel: false, presetName: "", savingPreset: false },
      exportPresets: [],
      bxOne: null,      // the picture "Compress / export" was pressed on (it may not be in the Library list)
      // Create: sections you folded down to one line (at first: the image prompt and extras are folded)
      cfolds: loadJSON("protoart.createFolds", { ipa: true, extras: true }),
      adjSections: ADJ_SECTIONS,
      ProtoLooks: window.ProtoAdjust ? window.ProtoAdjust.LOOKS : [],
      adjFilters: ADJ_FILTERS,
      helpOpen: "",
      helpQ: "",
      compareView: null,
      cmp: { z: 1, x: 0, y: 0, drag: null },  // ⇆ Compare: one zoom and position for every picture  // Library → ⇆ Compare side by side: the selected pictures  // How to → search the manual
      preloading: "",     // model being loaded in the background (kept out of the visible queue)
      readyModel: "",     // model known to be loaded and waiting
      content: { checker_installed: false, checker_mb: 340, total: 0, nsfw: 0, unchecked: 0, faces_unchecked: 0, scanning: false, msg: "" },
      revealDetail: false,
      modelDel: "",
      folders: [],         // Settings → Moved a folder?: your folders and whether each is there
      relinkMsg: "",
      loraFix: { q: "", move: true, msg: "", unknownOnly: false, showAll: false },  // Settings → Fix a LoRA's type
      favModels: loadJSON("protoart.favModels", []),
      favLoras: loadJSON("protoart.favLoras", []),    // ★ favourite LoRAs (by name; files never touched)
      loraFavOnly: false,  // ★ favourite models (kept on this Mac; files are never touched)
      typeLocks: loadJSON("protoart.typeLocks", {}),  // Fix a type: 🔒 rows (a type you picked is locked until you unlock it)
      sortPlan: null,      // Settings → LoRA folders → Sort into type folders (the plan, shown before moving)       // result of the last "Moved it…" 
      depthModels: [
        { id: "small", label: "Small", size: "~100 MB", tip: "Fast and light (fine on 8 GB Macs). Free for any use (Apache 2.0)." },
        { id: "base", label: "Base", size: "~400 MB", tip: "Sharper edges and finer detail; a little slower. Non-commercial use only (CC BY-NC 4.0)." },
        { id: "large", label: "Large", size: "~1.3 GB", tip: "Best detail; slowest, and on 8 GB Macs it frees the image model first. Non-commercial use only (CC BY-NC 4.0)." },
      ],
      helpTabs: [
        { id: "new", go: "tools", icon: "🆕", title: "What's new", summary: "The latest improvements, in one place", goLabel: "Open Tools",
          sections: [
            { h: "1.3.0: new tools", items: [
              "<b>New model: Anima</b> (CircleStone Labs, anime and illustration; Danbooru tags, sentences or both). <b>Anima Turbo</b> makes a picture in 10 steps; <b>Anima</b> uses 30 steps for the most detail. ~5.9 GB once, in the Models tab under Anima. Non-commercial licence for the model (pictures may be used commercially). It can also start from a picture in Image to image. Anima pictures are kept all-ages by default: the app adds Anima's <b>safe</b> rating tag unless your prompt already has a rating tag (safe, sensitive, nsfw or explicit).",
              "<b>🗺 Regions</b> (Create, SD 1.5 and SDXL): a different prompt for each part of the picture: columns, rows or a grid, with a small map of the areas. “Keep one scene” blends them so it stays one picture.",
              "<b>🔗 Model chain</b> (Image to image → 🔄 Change it): run a picture through several models in turn, e.g. SD 1.5 → SDXL → FLUX.1 or ERNIE, each with its own change strength. FLUX.1 Schnell and ERNIE (fast engine) can now start from a picture too.",
              "<b>Right-click menus across the app</b>: models (use, favourite, <b>⟲ Reconnect…</b> a moved or missing file, Show in Finder, fix the type, copy the name or location, remove from the list), LoRAs (add or take out, trigger words, Show in Finder, fix the type), the queue (start next, move up or down, stop), your folders in Settings (reconnect, show, stop using) and text boxes (cut, copy, paste, spelling). Pictures keep their own menu.",
              "<b>Library → Select</b>: the selection bar stays at the top while you scroll, and there's a 🗑 Delete button next to Compare. <b>Detail with a LoRA</b> (Tools) has a search box for the LoRA list.",
              "<b>🧩 Tiled upscale</b> (Tools): enlarges a picture 1.5–3× and lets your Create model add real detail tile by tile (MultiDiffusion), with no seams and little memory.",
              "<b>🔁 Face swap</b> (Image to image): put the face from one photo onto the person in another, tidied with CodeFormer (fidelity 0.7, 60%). Only for faces you have permission to use; non-commercial licence (inswapper_128); explicit results aren't saved.",
              "<b>ControlNet Union ProMax for SDXL</b>: one ~2.5 GB model for Canny, Depth, OpenPose, Line art, Scribble, Soft edges and Tile with any SDXL, Pony, Illustrious or NoobAI model. <b>Which one should I use?</b> (Image prompt and ControlNet) sets up the right tool for “the same person”, “copy a style or outfit” or “follow a pose or drawing”.",
              "<b>🗜 Smaller copy</b> (Tools): make an 8, 6 or 4-bit copy of an SD 1.5 or SDXL model that needs less memory. <b>🧪 Model merger</b> (Tools): blend two models into a new one, or add a difference.",
              "<b>Prompt editing</b> like A1111: <i>[cat:dog:0.5]</i> switches halfway through the steps, <i>[hat:0.3]</i> adds a word later, <i>[hat::0.3]</i> removes it. <b>🧱 Seamless</b> (Create): pictures that tile with no seam.",
              "<b>📥 Settings</b> (Create, next to List): load the settings a picture was made with (A1111, ComfyUI, InvokeAI or this app write them into PNGs) or a .json (a ComfyUI workflow, or ⤓ Settings .json from a picture's details). Dropping a .json on the prompt box works too.",
              "Fixed: the patchbay's DETAIL module lost track of its picture after the redraw." ] },
            { h: "1.2.2 (test version for friends): fixes from your notes", items: [
              "<b>ControlNet works again, and Image to image is much faster</b>. Switching to ControlNet, Image to image or Inpaint quietly turned the model into full precision: twice the memory, so 8 GB Macs crawled (20 minutes and more), and the ControlNet guidance pushed the picture off course, which gave flat grey (Canny), noise (Scribble) or nothing at all. The model now keeps its precision. Depth and OpenPose also free their helper model once the map is made.",
              "<b>Tile, OpenPose and Scribble no longer say “couldn't reach huggingface.co”</b> when you are online: Tile's file is in an older format (now read too), and a file that's genuinely missing on Hugging Face says so instead of blaming your internet.",
              "<b>Inpainting models</b> (files with “inpainting” in the name) load for Inpaint and Expand instead of failing with a conv_in size error.",
              "<b>Patchbay</b>: a <b>SELECT</b> tab at the bottom of each module picks it; the rack unit below then shows <b>that module's settings</b>, with a ⚙ FACES 2 badge so you know which copy you're editing (no second settings panel). <b>✕</b> on a module takes it off the rack (bring it back from Add a module). New <b>🔎 DETAIL</b> module: a light redraw with an “add detail” LoRA. <b>🛡 Protect areas…</b>: paint parts of the picture the rack must not change. The picture forming shows in Tools while it runs.",
              "<b>CodeFormer presets</b> (Restore faces, in Create and Tools): Balanced, Keep likeness, Heavy repair, Subtle touch-up, Old photo, Gentle. Each sets fidelity and strength; you can still change both.",
              "<b>🗺 Regions</b> (SD 1.5 and SDXL): tick <i>Use regional prompts</i>, pick Columns, Rows or Grid and the sizes (1,1 = halves; 1,2 = a third and two thirds; Grid 1,1;1 = two on top, one below), and write a prompt for each numbered area. Each area follows its own prompt; the main prompt above sets the scene around them, and <i>Keep one scene</i> sets how much (0.2 is a good start). Describe each area fully (“a woman in a red coat, full body”) and use a wide size for columns (768 × 512 for three). <i>Add the main prompt to every area</i> blends them more but can drown the subjects. <b>FLUX.1 and ERNIE</b> do regions in two passes: the whole picture with each area's place in words, then each area redrawn from its own prompt and blended in (<i>Redraw each area</i>: 0 = skip, 0.5 = a good start); it takes about as long as one picture per area plus one. <b>FLUX.2 Klein</b> works the same way, but each area is <i>edited with words</i> to show its prompt (the slider sets how strongly the edit shows). Each area's <b>✨</b> button turns a few words (“a vintage 60s car”) into a detailed description of that subject (tags for SD models); ↩ puts yours back. Anima can't use areas: describe positions in words instead (“on the left…, on the right…”).",
              "<b>Prompt editing</b>: <i>[cat:dog:0.5]</i> draws a cat for the first half of the steps and a dog after (a dog in a cat's pose); a number above 1 is a step (<i>[cat:dog:10]</i>). <i>[hat:0.3]</i> adds “hat” after 30 %, <i>[hat::0.3]</i> removes it. SD 1.5 and SDXL. The shape is mostly settled early, so for a bigger change switch sooner (<i>[cat:dog:0.2]</i>).",
              "<b>🧱 Seamless</b>: the picture's edges wrap around, so copies placed side by side join with no seam: textures, patterns, wallpapers.",
              "<b>📥 Settings</b>: load a picture's settings (prompt, negative prompt, steps, guidance, seed, size, sampler, model and LoRAs if you have them) from a PNG made by A1111, Forge, ComfyUI, InvokeAI or this app, or from a .json. Anything you don't have is listed. ↩ undo brings your prompt back.",
              "<b>🎲 Small change</b> (under Seed, SD 1.5 and SDXL): keep a picture you like and change only small details. Press <i>try a small change</i> (it keeps the last picture's seed and picks a variation seed), then Generate; press it again for another. The slider sets how much changes (0 = off, 0.05–0.2 small details, 0.6 a lot). The variation seed and strength are saved with the picture, so Use these settings brings them back.",
              "<b>📒 My prompts</b>: ⤓ Export saves them all to a file (to keep a copy or give to a friend); ⤒ Import adds prompts from such a file, or from a .txt with one prompt per line. Prompts with the same name and text aren't added twice.",
              "<b>🔄 Contrast prompt</b> (Create → ✨ Enhance, and Image to image): suggests the same scene turned into its opposite, like a waterfall frozen into blue-white ice or a summer street after light snowfall. ↩ undo puts yours back.",
              "<b>Your own defaults</b>: 💾 Save as this model's defaults (sampler, steps, guidance) next to Reset in Create and in Image to image's Steps card; they load whenever you pick that model. A LoRA's ☆ sets the weight it starts at.",
              "<b>Library: right-click a picture</b> for Open, Star, Send to Image to image or Tools, Show in Finder and Delete.",
              "<b>Image → prompt</b> (Tools): ⧉ Copy, 🖼 Use in Image to image (picture and prompt together) and ✍ Use in Create.",
              "<b>Old .ckpt models</b>: Tools → Convert → <b>Old .ckpt files…</b> makes a .safetensors copy that loads (read safely: nothing inside the file is run; the .ckpt is left as it is).",
              "<b>Settings → Main folder</b>: pick one folder (an external drive is fine) and its Models, LoRAs, Embeddings, VAE and ControlNet sub-folders are all used. <b>↻ Refresh</b> re-reads everything; a drive macOS hasn't allowed yet, or one that isn't plugged in, now says so.",
              "<b>LoRA tags can also be trigger words</b> (tick “also as trigger words”), the LoRA list remembers it's open, Klein 4B is the name for FLUX.2 [klein] 4B everywhere, the outpaint LoRA no longer shows in Create, and the active LoRAs in Create stand out.",
              "<b>🎲 Small change</b> (Create, under Seed, SD 1.5 and SDXL): a variation seed that keeps the picture and changes small details. Press <i>try a small change</i> after a picture you like; 0.05–0.2 is subtle, higher changes more.",
              "<b>📒 My prompts: ⤓ Export / ⤒ Import</b> to keep a copy of your saved prompts or share them (a .json file, or a .txt with one prompt per line).",
              "<b>Save a picture anywhere</b>: 💾 Save as… in a picture's details, or right-click the picture (Save picture as… / Copy picture), in the details and in the Library.",
              "<b>SD 1.5 sizes</b>: Med+ (704) on the Max size dial, and Square 704, Tall 704 × 1024, Tall 768 × 1024 and Wide 1024 × 704 shapes.",
              "<b>Settings → Moved a folder?</b> now has <b>Remove</b> for a folder you don't use any more (the files stay where they are). <b>Tools → Convert → Pick files or a folder…</b> converts DiffusionBee .tdict files from anywhere.",
              "<b>Library toolbar in a narrow window</b>: ⇆ Compare no longer gets cut off, and the selection bar wraps instead of running off the edge.",
              "Smaller things: the splash meter's red zone sits on the scale, the prompt list says “Add prompts above” instead of “Queue 0 jobs”, and Close / Delete in a picture's details no longer overlap its text." ] },
            { h: "1.2.1 (test version): fixes from your notes", items: [
              "<b>Expand with FLUX.2: fewer seams and doubled objects</b>. Your picture goes to FLUX.2 at exactly the size it draws, its result is lined up with your picture before the two are blended (over a wider band), and if FLUX.2 redrew the picture its whole version is kept instead of a patchwork. The result also no longer sits between black bars.",
              "<b>Expand advice</b>: ★ make it wider / ★ make it taller are the recommended choices, and a note warns when you expand both ways at once or by a lot. <b>💡 suggest from the picture</b> writes a starting prompt from your picture.",
              "<b>Patchbay: click a module</b> to see its settings: the first copy lights up its rack unit below; a copy you added (FACES 2…) gets <b>its own settings</b> right under the panel, so two face restores can use different strengths. <b>💾 Save flow</b> keeps the whole signal flow as a preset to load again.",
              "<b>🏷 Your own LoRA tags</b> (Models → 🏷 LoRA tags): tag LoRAs (e.g. <i>pencil</i>), many at once, then filter Create's and Image to image's LoRA lists by a tag (pick 🏷 pencil in the list's menu, or type <i>#pencil</i> in its search).",
              "<b>Expand with FLUX.2</b> now picks FLUX.2 Klein in the model menu for you, has its own <b>✨ enhance now</b> (turns “more of the street” into a clear description of what goes around the picture), and its result stays in Image to image instead of Create.",
              "<b>Image to image: the picture comes first</b>, at the top next to the preview; Sampler, steps &amp; guidance and LoRAs follow (and hide when FLUX.2 does the work, as they don't apply).",
              "<b>Models tab</b>: Embeddings, VAEs &amp; ControlNets folds up like the other lists, and <b>Your model files</b> moved to the bottom.",
              "<b>Tips everywhere</b>: every button, slider, menu and switch now shows a tip when you point at it (sliders such as Fidelity use the words next to them).",
              "<b>Tidier top of Create</b>: collapse all · expand all and the Reset all lever share one line, with no gap above Model.",
              "<b>↔ Expand with FLUX.2 + outpaint LoRA</b> (Image to image → Expand): fills the new area on any picture, whatever model made it, using FLUX.2 Klein with fal's outpaint LoRA (Apache-2.0, 76 MB, downloaded once). Your picture stays exactly as it was. The SD 1.5 / SDXL way is still there (<i>The model at the top</i>).",
              "<b>📚 Use the model library you already have</b>: Protoart Studio looks for <b>Stability Matrix</b>, <b>ComfyUI</b>, <b>Automatic1111 / Forge</b>, <b>InvokeAI</b> and <b>Fooocus</b> on your Mac and asks “Existing model library detected. Would you like to use it?”. Checkpoints, LoRAs, embeddings, VAEs and ControlNets are read where they are: nothing is copied, moved, renamed or changed. Change it in Settings → Model folders.",
              "<b>Watch as many model folders as you like</b> (Settings → Model folders → ＋ Add watched folders; ⌘-click to pick several), and <b>↻ Refresh</b> reads them again if new files didn't show the first time.",
              "<b>LoRA lists line up</b>: DiffusionBee LoRAs are marked <b>BEE</b>, and the type and size sit under the name so long names have room.",
              "<b>Tidier prompt tools</b> (Create): one row of small keys under the prompt: <b>✨ Enhance · 📒 Saved · 🖼 From image · 🎲 Random · { } Wildcards · 📋 List</b>. Each opens its own small panel underneath, one at a time; a lamp shows when Enhance or New random prompt is on.",
              "<b>🎲 Random prompts with animals</b> (pets, wild animals, birds and sea life, each in a place that suits it), and <b>New random prompt every time I press Generate</b>.",
              "<b>Reset all is a blue lever</b> (top of Create): flip it to clear both prompts, put every setting back to the start and fold every section. It's spring-loaded: the lever goes straight back to rest, and there's no undo, so save a preset first if you might want your settings again.",
              "<b>Seed</b> now sits under the negative prompt. <b>📒 Negative prompt presets</b>: save a negative prompt on its own and use it with any prompt.",
              "<b>★ Recommended steps and guidance for the sampler you picked</b> (next to Reset to defaults, and in Image to image): ⚡ fast samplers get fewer steps, ★ best-quality ones more.",
              "<b>Image to image</b>: the Sampler, steps &amp; guidance card has its own sampler menu, and it and the LoRAs card fold up like Create's sections.",
              "<b>Models tab</b>: ↻ Refresh and + Add from your Mac swapped places; <b>On this Mac</b> and the download lists fold up under Fix a type; Pony, Illustrious and NoobAI filters; a clear “✓ Model imported” note after adding one.",
              "<b>Trigger words panel</b> (Models tab): search your LoRAs, edit their trigger words in place, or tick several and add / remove a word on all of them at once.",
              "<b>Library</b>: ☑ Select sits in the middle of the toolbar; select 2–6 pictures and press <b>⇆ Compare</b> (in the toolbar, next to ✕ Deselect all, so it stays in view while you scroll); in a picture's details, <b>‹ ›</b> (or ← →) step through the pictures.",
              "<b>Select by dragging</b> (Library → ☑ Select): drag across pictures, or draw a box from the space between them, to select many at once; start on a selected picture to unselect. Fixed: the picture details' <b>‹ ›</b> arrows closed the picture instead of going to the next one. <b>✕ Close</b> and <b>🗑 Delete</b> now sit beside the picture's details, well apart so one isn't pressed for the other (Esc closes too). The window keeps the same size while you step through pictures.",
              "<b>Compare side by side zooms</b> (mouse wheel, pinch, － ＋ / Fit / 1:1; all pictures together), <b>✕ Deselect all</b> under Selecting · Done, and smoother drag-to-select.",
              "<b>Train → ✕ Clear</b>: start again with no training pictures chosen (the folder and its files aren't changed).",
              "<b>Patchbay: add more modules</b> (Tools): ＋ Add a module puts in another Faces, Enhance, Remove or Depth, so you can run Restore faces twice (or any module again) in one chain; the panel widens for them.",
              "<b>Patchbay controls on top, and modules you can move</b>: Run patch, ↺ reset, clear cables and ＋ Add a module now sit above the panel, and the patchbay <b>starts with no cables</b>: plug in only what you want (or press <i>example</i> for Picture → Enhance → Faces → Result). Drag a module's name left or right (or point at it and press ‹ ›) to put the modules in the order you like; ↺ reset puts them back and pulls out the cables.",
              "<b>🔎 Search the manual</b>: a search box at the top of How to shows every matching tip from all the sections.",
              "<b>ControlNets work again with a LoRA switched on</b>: DiffusionBee's ControlNets (Depth, Tile, Line art, Scribble, OpenPose) are built partly from your model, and a LoRA in the mix broke that copy, so the app tried to download the official ones instead (which failed without internet). The LoRA parts are now left out of the copy.",
              "<b>Clear messages when a one-time download can't happen</b>: instead of “does not appear to have a file named config.json” or “Can't load image processor”, the app says it couldn't reach Hugging Face and what to check.",
              "<b>Inpainting models go where they belong</b>: an inpainting checkpoint (like dreamshaper_8Inpainting) is recognised from the file, labelled “inpainting”, kept out of Create, and used for Image to image → Inpaint and Expand (switch: 🖌 Paint with an inpainting model).",
              "<b>LoRAs for another model type are left out, not an error</b>: a FLUX.1 LoRA with FLUX.2 (or SD 1.5 with SDXL) is skipped for that picture with a note, and the picture is made.",
              "<b>LoRAs in your watched model folders are found</b> (Models → Watch a folder), like your ERNIE .safetensors LoRA in a sub-folder: press ↻ Refresh after adding one.",
              "<b>🎲 New random mix swaps every LoRA</b>, including ones added by hand or by a preset (before, only the randomizer's own picks changed).",
              "<b>Hide People hides pictures of people</b> in the Library: the prompt describes a person, or the face finder (the one Restore faces uses) sees a face. Older pictures: <b>check now</b> next to the picture count.",
              "<b>Image to image</b>: progress of Edit with words shows in Image to image (not Create); <b>✕ Clear results</b>; drop another picture on 1 · Picture to replace it; <b>enhance now</b> keeps the text in the tab that asked for it.",
              "<b>Embedding training on 8 GB Macs</b>: everything else is unloaded first, memory-saving is on for the text encoder too, pictures are 448 px under 16 GB, and running out of memory saves your progress with a clear message.",
              "<b>Tidier lists</b>: LoRA list markers and sizes line up on every row, the queue's ⤵ no longer covers the model name, ● People / Not People filters in the LoRA list, and the left panel no longer scrolls sideways.",
              "<b>No developer menu in the installed app</b>: View → Reload / Toggle Developer Tools are gone (they're only in the test version you start from the folder).",
              "<b>How to</b>: 📜 Model licences is now a fold-out row under Troubleshooting (click to open), so it no longer takes up half the page.",
              "<b>QR code / pattern ControlNets removed</b>: they gave flat grey pictures on Apple Silicon Macs, so this version doesn't offer them (the pattern picker and the Shape maker are gone too). A QR / pattern ControlNet file you added shows in Models → Embeddings, VAEs &amp; ControlNets as not supported; <b>Open folder</b> there takes you to it if you want to delete it. Every other ControlNet works as before.",
              "<b>Recommended settings for each ControlNet</b>: picking one in Image to image puts in the settings it works best with straight away (strength, when it guides, steps), shown in a box under the picker with <b>undo</b>. If you change something that would spoil the result (for example far too few steps), the box turns orange and says why, with <b>Use recommended settings</b>.",
              "<b>Fixed: some SD 1.5 ControlNets you add yourself failed to load</b> with “doesn't match an SD 1.5 ControlNet's layout (52 parts missing)”.",
              "<b>Move to Trash works for your own model files</b>: the file was moved, but the list kept showing it for a while, so it looked as if nothing happened. Now it disappears straight away (and ↻ Refresh looks in your folders again at once). Model files that were moved or renamed outside the app are no longer listed until they're back.",
              "<b>⏳ “This will be slow on this Mac”</b>: before a picture starts, settings that need more memory than your Mac has get a short warning with a one-click faster choice (for example <i>Use 512 × 512</i> for a ControlNet, the 8-bit SDXL instead of full precision, or Hi-res fix off). 8 GB Macs see it for big ControlNet and SDXL jobs, 16 GB Macs only for the heaviest, and Macs with 24 GB or more never. <i>Don't warn me about this again</i> switches one off; Settings → Speed turns them back on.",
              "<b>Your own ControlNets load faster on 8 GB Macs</b>: ControlNet files (many are 1.4 GB, full precision) are now read straight into half precision and built without spare copies, so loading one no longer pushes the Mac into swap (30–60 s a step). <b>Use these settings</b> picks 512 × 512 on Macs under 16 GB where it matters, since a ControlNet roughly doubles the work of each step.",
              "<b>Reorder the queue</b>: ▲ and ▼ beside each waiting job move it up or down, so you choose which prompt starts next. <b>Models tab</b>: the ★ default tag is now easy to read (it glowed too brightly).",
              "<b>Tools → Depth map → Use as a ControlNet depth map</b> now always lands in Image to image ready to go: it switches to 🔄 Change it and an SD 1.5 model (Edit with words or Inpaint used to hide the ControlNet). The <b>ControlNet guide</b> now sits in the roomy right-hand column above the preview, with its steps open. <b>Library</b>: Img2img, ↺ Prompt and Tools sit in one even row along the bottom of a picture, and 🗑 moved next to ★ so it no longer covers the picture's label.",
              "<b>ControlNet guides</b> (Image to image): pick a ControlNet and a short guide appears: what it does, what picture to give it, the <b>best models</b> you have for it (★ = a good match; click one to switch) and <b>how to use it, step by step</b>. <b>Use these settings</b> sets its strength, timing and picture options in one click. The model list now says what each group can do (image to image, inpaint, ControlNet, image prompt), and while editing with words the ControlNet card suggests which ControlNets would suit and has <b>Use ControlNet instead</b>.",
              "<b>Create and Image to image keep their own prompts</b>: each tab has its own prompt and negative prompt, so a prompt written in one no longer turns up in the other (and an ✨ enhanced prompt stays with the tab it was made in). Pictures made by <b>Tools → 🩹 Remove</b> stay in Tools and the Library instead of showing in Image to image's preview.",
              "<b>Train → leave pictures out and crop by hand</b>: the <b>✕</b> on a picture (or <i>⊘ Leave out</i> in its list) leaves it out of training without touching the file (<b>↺</b> puts it back). <b>✂ Crop by hand…</b> (or click a picture) opens it large with a square you drag to move and resize by its corner: training uses exactly that square. Pictures the window can't show (some HEIC files) get a striped tile instead of a broken one; training still reads them. The caption buttons no longer run into the next column.",
              "<b>⤵ Run the others first</b> (top of the queue, or ⤵ on the running job): moves the job that's running to the back of the queue so the waiting ones run now. Pictures it already finished are kept; the one it was making starts again when its turn comes.",
              "<b>Training won't freeze the Mac any more</b>: on Macs under 16 GB it checks first and, if memory is already full (most of it moved to the disk), says which apps to quit instead of starting. If a model file has been moved or renamed, it now says so instead of “Invalid pretrained_model_name_or_path”.",
              "<b>VAE</b>: once the recommended VAE is downloaded, the <b>⤓ get</b> link goes away (it stayed before, so it looked as if nothing had happened). <b>Narrow windows</b>: the Look buttons at the top of Settings shrink to their icons instead of running off the edge. <b>A new launch screen</b>: the unit powers up, with lamps, a VU needle and an amber display.",
              "<b>DiffusionBee .tdict LoRAs</b> are SD 1.5 or SDXL (DiffusionBee makes no ERNIE LoRAs), so they're listed with SD 1.5 / SDXL models, not ERNIE ones.",
              "<b>🩹 Remove in Tools no longer waits</b> behind a picture being made: it runs alongside it (it uses the processor, not the graphics chip), so it takes seconds even while Create is busy.",
              "<b>⬚ Object selection</b> shows <i>Finding the object…</i> on the picture while it works, is faster (about a second), and if it can't find a clear outline it selects the box itself so you can trim it, instead of doing nothing.",
              "<b>Selections are easy to see</b>: everything painted or selected (brush, lasso, 🪄 wand, object) gets a black-and-white dashed outline (“marching ants”), so it shows on any picture, pink ones included. The <b>brush-size well</b> beside the slider now grows across the slider's whole range (it stopped growing at 44 before); the ring on the picture is the exact brush size.",
              "<b>Time left agrees everywhere</b>: the TIME LEFT clock, the queue and the progress line now show the same figure. It counts down between steps, and learns how long each model takes to finish a picture after its last step (it used to guess 2 extra steps, which was minutes too much on slow models).",
              "<b>LoRAs on the ⚡ fast engine are much faster on 8 GB Macs</b> (ERNIE, FLUX.2 Klein, FLUX): a LoRA used to be folded into the model, which turned the 4-bit model into a twice-as-big 8-bit one, so the Mac ran out of memory and worked off the disk (80–110 s a step instead of ~15). Now, on Macs with less than 24 GB, the LoRA stays a small add-on beside the 4-bit model. It also applies the LoRA's full effect (folding lost part of it).",
              "<b>🔌 Patchbay</b> (Tools, the PATCHBAY toggle): wire Remove objects, Restore faces, Enhance, Depth map and Vectorize together with cables, in any order, like the back of a studio rack, then <b>▶ Run patch</b>. For example: remove an object, then restore the faces, then enhance. See How to → Tools.",
              "<b>SDXL ControlNets</b> (Pony, Illustrious and NoobAI too): download them from Civitai into the ControlNet folder and pick them in Image to image → ControlNet with an SDXL model. Full, mid and small sizes are read from the file. A name with <i>canny, depth, lineart, openpose, softedge, scribble</i> or <i>tile</i> gets that map made from your picture (SD 1.5 ones too). Control-LoRA, LLLite and Union files aren't supported yet: the Models tab says so next to the file." ] },
            { h: "Version 1.2: new tools, and more of your testers' notes", items: [
              "<b>✨ Edit with words</b> (Image to image): describe a change (“make the jacket red”, “add a hat”, “turn it into a watercolour”) and FLUX.2 Klein makes just that change. Paint over a part to keep everything else exactly as it was, or add more pictures to take a style or an outfit from.",
              "<b>🎲 Random prompt speaks the model's language</b>: with FLUX, FLUX.2 or ERNIE picked it writes a scene in sentences (“Medium shot of a fisherman with a white beard, mending a net on a harbour wall at dawn…”); with SD 1.5 and SDXL it still writes tags. In Image to image, <b>FLUX.2 Klein</b> now shows at the top of the model list while Edit with words is on.",
              "<b>Selection tools</b> wherever you paint over a picture (Inpaint, Remove, Edit with words, Tools → Remove objects): <b>🖌 Brush</b>, <b>🧽 Erase</b>, <b>➰ Lasso</b> (draw around an area), <b>⬠ Polygon</b> (click the corners), <b>🪄 Wand</b> (click a colour; set the tolerance) and <b>⬚ Object</b> (drag a box and the object's outline is found). <b>＋ Add / － Subtract</b> (or hold ⌥ Option); keys B E L P W O; Esc cancels. <b>VAE</b>: each model remembers its VAE, only matching ones are listed, and <b>⤓ get</b> downloads the recommended one. <b>Random / Wildcards / List</b> are now three keys.",
              "<b>🛡 Safe-for-work mode</b> (Settings, the 🛡 SFW switch at the top): refuses prompts that ask for adult content, NSFW LoRAs and models, hides NSFW pictures everywhere (they can't be opened) and checks every new picture on this Mac. An SFW lamp shows at the top. <b>VAEs</b>: download them from Civitai (or drop files in the VAE folder) and pick one in Create → Extras → VAE (SD 1.5 and SDXL). <b>Your own ControlNets</b> (SD 1.5, e.g. a canny or depth one from Civitai) appear in Image to image → ControlNet and follow your picture as it is. The <b>Models</b> tab lists <b>Embeddings, VAEs &amp; ControlNets</b> with filters. <b>FLUX.1 Dev</b> (higher quality FLUX, 25 steps) is in the Models tab for Macs with 32 GB or more; it's free for personal use only (non-commercial licence) and needs a free Hugging Face account.",
              "<b>Compare in Tools</b>: after Enhance, Restore faces, Remove or a depth map, the picture window shows <b>Original / Result / Split / Side by side</b> (drag the line in Split). <b>Brush size</b>: a ring the size of the brush follows the pointer while you paint, and moving the size slider shows it on the picture; the metal well next to the slider shows the size too. <b>LoRA list</b>: sort A → Z, Z → A, biggest or smallest, and show only one kind (.safetensors, DiffusionBee, Pony…). <b>🔔 Sounds</b>: a chime when pictures or training are finished (switch in Settings, next to Tips). <b>How to</b> now starts with Community &amp; support (Discord, contact, report a problem) and the model licences.",
              "<b>Negative prompts that fit</b>: next to <i>Negative prompt</i> (Create and Image to image), <b>✨ suggest</b> writes one for your prompt and model type: tags for SD 1.5 / SDXL / Pony / Illustrious (with each one's low-quality tags), plain words for ERNIE and FLUX, plus the opposite of the style you asked for (a cartoon prompt avoids “realistic, photo”). <b>🎲 random</b> adds a few extra fixes each time. 🎲 Random prompt and ✨ enhance write a matching negative too, unless you typed your own. ✨ Enhance now writes tags for SDXL and Pony as well as SD 1.5, and in <b>Edit with words</b> it makes your instruction clearer. <b>📒 My prompts</b> keeps prompts you like (with their negative prompt) to use again.",
              "<b>Image to image</b> now has <b>✨ Enhance prompt</b> in its Prompt card (the same switch as in Create, plus <i>enhance now</i>), and the pictures it makes stay in Image to image and the Library instead of also filling Create's screen.",
              "<b>📝 Image → prompt reads pictures much better</b>: <b>Describe</b> now uses <b>Qwen3.5 4B</b>, a 2026 vision model running on the Mac's GPU, which writes a detailed prompt in sentences (subject, clothes, pose, setting, light, colours, framing, style), what FLUX, FLUX.2 and ERNIE read best. The first time it sets itself up and downloads ~3 GB; then it works offline. <b>Quick</b> is the older Florence-2, and <b>Tags</b> the Danbooru tagger. <b>CodeFormer</b> is now the default face model. The <b>TIPS</b> toggle sits at the top beside the Power / Busy / Fault lamps.",
              "<b>🩹 Remove objects right in Tools</b>: press <b>🖌 Paint</b> in the Remove module, paint on the picture on the left and press <b>🩹 Remove</b> (no trip to Image to image). The <b>TIPS</b> switch beside the Power / Busy / Fault lamps at the top turns the pop-up tips on and off.",
              "<b>Tools redesigned</b>: every tool (Run chain, Enhance, Restore faces, Depth map, Remove objects, Vectorize, Image → prompt) is now a tall rack module side by side; Enhance's less-used settings are under <i>fine-tune</i>, each with its controls and its own button. <b>Sampler picks</b> show just the top two for speed and for quality (the rest are in the list).",
              "<b>New model: FLUX.2 Klein 4B</b> (Black Forest Labs, January 2026) on the ⚡ fast engine, made for 8 GB Macs: a 4.6 GB download, 4 steps a picture. It's newer than FLUX.1 Schnell and better at following long prompts and writing text in pictures. Find it in the <b>Models</b> tab under FLUX.2. It ignores the negative prompt, and FLUX.1 LoRAs don't work with it (FLUX.2 LoRAs do).",
              "<b>🩹 Remove objects</b> (spot healing, like a healing brush): in <b>Image to image</b> choose <b>🩹 Remove</b>, paint over anything you don't want (a stray object, an extra finger, text, a blemish) and press <b>🩹 Remove</b>. It's filled in from what's around it; only the painted part changes. <b>✨ Smart fill</b> rebuilds what was behind it with an AI fill model (LaMa, a ~208 MB download the first time, then offline); <b>⚡ Quick fill</b> is instant for small spots and scratches. Also in <b>Tools → 🩹 Remove objects</b>.",
              "<b>✒️ Vectorize</b> (Tools): turns a picture into an SVG of flat colour shapes, for logos, icons, stickers and cut files. Choose the number of colours, the detail and smooth or sharp outlines; it opens in Illustrator, Affinity, Inkscape, Figma or Cricut.",
              "<b>★ Favourite LoRAs</b>: click the ★ on a LoRA in the list. Favourites come first, and the <b>★</b> button next to the search box shows only them.",
              "<b>💬 Tips</b>: point at almost any button and a short explanation appears. Switch them off (or on) in <b>Settings</b>, next to the colour themes.",
              "<b>🎲 Random LoRAs</b> now reads as a sentence (pick 1 to 2 LoRAs, each at a strength between 0.6 and 1), and the boxes no longer overlap.",
              "<b>LoRAs no longer disappear</b>: when you switch to a model of another type, its LoRAs are put aside and come back when you switch back; and the app no longer jumps back to your default model when its window reloads.",
              "<b>Image prompt with an 8-bit SDXL model</b> works (it stopped with “Casting a quantized model to a new dtype is unsupported”).",
              "<b>Civitai</b>: <b>Download</b> adds to a queue, so you can paste the next link straight away; a <b>Downloads</b> list shows them all. If you may already have the file (same Civitai version, same name or same size), it says so and asks before downloading.",
              "<b>↺ reset all</b> at the top of Create, and <b>↺ Reset Tools settings</b> in Tools: back to the starting settings, with undo. Your presets aren't touched." ] },
            { h: "A second face restorer", items: [
              "<b>Restore faces now offers CodeFormer</b> as well as RestoreFormer, in Create, Tools and a picture's details. <b>CodeFormer</b> is now the default (it's listed first); <b>RestoreFormer</b> stays as the natural, free-for-commercial-use choice. <b>CodeFormer</b> repairs badly blurred or damaged faces more strongly and has a <b>Fidelity</b> slider (low = cleanest face, high = keeps more of the person's own features). CodeFormer's licence allows <b>non-commercial use only</b>.",
              "Both face models download once, the first time you use them, and are now checked against a pinned SHA-256 before loading. A damaged or tampered file is deleted and fetched again.",
              "Each restored picture records which model, strength and fidelity were used in its settings." ] },
            { h: "Version 1.1: from our first testers' notes", items: [
              "<b>Train → Square framing</b>: training uses a square of every picture, and now you choose which part. Each picture gets <b>Crop · centre / top / bottom / left / right</b> or <b>Keep whole</b> (all of it, with the sides filled by a soft blur), one by one or for all at once. What each tile shows is what the training sees. <b>Save copies</b> writes square 512 / 768 / 1024 copies with their captions to a new folder (your originals aren't changed).",
              "<b>Image to image has its own LoRAs card</b>: add, remove and weigh LoRAs right there (the same LoRAs as Create, since both tabs use one model).",
              "<b>Create: ✕ on each picture</b> takes it off the screen (it stays in the Library), and <b>clear all</b> empties the screen. The model that made each picture is written along its bottom.",
              "<b>Tools results stay in Tools</b> (and the Library): enhanced, face-restored and depth pictures no longer appear among Create's pictures.",
              "<b>Library → Hide People</b>, next to Hide NSFW: hides pictures made with models or LoRAs you tagged ● People.",
              "<b>Civitai</b>: <i>civitai.red</i> links and download links work; when a firewall or VPN blocks the app (“Bad file descriptor”), it tries again through macOS's own downloader by itself. <b>⟳ Retry</b> and <b>Clear</b> buttons, and a slow or blocked Civitai no longer makes the rest of the app stop responding.",
              "<b>“Your Mac is short of memory”</b> note: when other apps have already taken the memory, the app says so before a picture starts (that, not the model, is what made SDXL take 30–100 s a step on an 8 GB Mac with browsers open).",
              "<b>■ Stop stops right away</b>: no more pressing <b>Force stop now</b>. If the picture hasn't stopped within a couple of seconds, the app force-stops the engine by itself (waiting jobs carry on). Training still finishes its step first, so it can save what it learned.",
              "Tools: <b>❚❚ Pause</b> and <b>■ Stop</b> sit side by side, centred, while enhancing." ] },
            { h: "🧬 Merge LoRAs and 🔤 train embeddings", items: [
              "<b>Train → Merge LoRAs</b>: combine two or more LoRAs for the same model type into one, each at its own weight. <b>Exact</b> keeps every detail; <b>Smaller</b> makes a much smaller file that's a close match.",
              "<b>Train → Embedding</b>: teach an SD 1.5 model one new word from 5–20 pictures (textual inversion). The result is a tiny file you use by typing its name. Pause / Stop work as usual." ] },
            { h: "🎛 Themes, favourites and a tidier Models tab", items: [
              "<b>Colour themes</b> (Settings, at the top): <b>Amber</b> (charcoal and amber, the classic look), <b>Cream</b> (warm cream, yellow lamps, deep red), <b>Slate</b> (blue-grey steel, ivory lamps) and <b>Ember</b> (black and red-orange). Each works in Light and Dark; everything that's on lights up in the theme's colour.",
              "Buttons that are on now light up fully, like an illuminated studio switch, including the choices in Settings (PNG / JPEG and the like).",
              "<b>★ Favourite models</b>: click the ★ on a model (Create's list or the Models tab). Favourites come first, and <b>★ Favorites</b> shows only them.",
              "Markers ★ ● ⚑ sit in their own column on every row, so long names no longer push them around (a long name shows in full when you point at it).",
              "The <b>Models tab</b> now holds your model files, Add from Civitai, Add from Hugging Face and Fix a type; <b>Settings</b> keeps the folders. Settings no longer overlaps itself in a narrow window.",
              "<b>Hide NSFW</b> recognises many more explicit LoRA and model names, so 🎲 Random LoRAs and the lists leave them out while it's on. You can still mark any item yourself with ⚑.",
              "Picture details: the decorative screws are gone from the pop-up, so only <b>Close</b> closes it.",
              "New How to topic: <b>⚖️ Disclaimer &amp; licences</b>." ] },
            { h: "❚❚ Pause, ↺ Reuse prompt, recommended samplers", items: [
              "<b>❚❚ Pause</b> (next to Stop, and at the top of the queue): the picture being made waits at its next step and nothing new starts; <b>▶ Resume</b> carries on exactly where it was. Works for generating, enhancing in Tools and training. The clocks stop while paused.",
              "<b>↺ Prompt</b> on any picture you made (hover it in Create or the Library), or <b>↺ Reuse prompt</b> in its details: puts its prompt back in the prompt box and keeps your current model and settings. <b>⚙ All settings</b> / <b>Use these settings</b> loads everything (model, LoRAs, size, seed…). If Enhance prompt rewrote it, <b>↺ Original prompt</b> gives you what you typed.",
              "Under the Sampler list: <b>⚡ Fastest</b> and <b>★ Best quality</b> picks for the chosen model's type, outlined in faint blue.",
              "<b>Fix a type</b> (Models tab): <b>🔒</b> locks a row's type so it can't change by accident, and <b>Show all</b> lists every file.",
              "Image prompt (Style / Face…) on a fresh install: the adapter downloads by itself the first time instead of stopping with “does not appear to have a file named…”." ] },
            { h: "🎚 Adjust: edit after generating (Lightroom / Color Efex style)", items: [
              "In <b>Tools</b>: <b>🎚 Adjust the original</b> or <b>Adjust the result</b>; also <b>🎚 Adjust</b> in the Library's picture details and in Image to image.",
              "Light, Color and Presence sliders (Exposure … Vignette), five Color Efex-style filters in a stack you can reorder, six ready Looks, and <b>your own presets</b>.",
              "Live preview on the Mac's graphics chip; compare with <b>Edited / Original / Split</b> (or the \\ key). Saved as a new picture; the original never changes.",
              "<b>🖼 Send to Image to image</b> from Tools: the original, any result, or straight from Adjust (<b>Save &amp; send to Image to image</b>).",
              "<b>Save &amp; enhance</b> runs the AI enhancer on your adjusted picture, and <b>Several versions at once</b> makes one picture per preset in a single run, optionally enhanced too (through the queue). More under Tools → 🎚 Adjust." ] },
            { h: "⚡ Enhance & upscale is far faster", items: [
              "On an 8 GB M1, a 640×800 picture at 2× now takes about <b>75 s with 🚀 Turbo</b>, <b>3 min with ⚡ Fast</b> and <b>8 min with Standard</b> (it used to take over an hour).",
              "Why: the speed-ups from ProtoArt Beautify: the detail add-ons are merged into the model once, the picture is cut into the fewest tiles that fit your Mac's memory, memory goes back to macOS after every pass, and the upscaler works in small pieces on 8 GB Macs.",
              "New <b>⏩ Instant</b> speed (seconds: the sharpening upscaler only, no new detail) and <b>Work at</b> 512 / 768 px: <b>Auto</b> picks 512 px for the quick speeds on Macs under 16 GB (less than half the work; the → size next to Enhance shows what you'll get).",
              "Tip: run only one image app at a time on 8 GB (not ProtoArt Beautify or DiffusionBee alongside); that alone can make it many times slower." ] },
            { h: "🔤 Embeddings", items: [
              "<b>Settings → Embeddings</b>: add embedding files (copied in), add folders (used in place), or download them <b>From Civitai</b> (“Textual Inversion”). SD 1.5 and SDXL (Pony, Illustrious, NoobAI too).",
              "Above the negative prompt, Create shows only the embeddings made for the chosen model's type; click one or type its name. Embeddings left in LoRA folders are found too and no longer listed as LoRAs." ] },
            { h: "🗂 Batch edit & export, and 🗜 Compress", items: [
              "In the <b>Library</b>: select pictures and export them all at once: resize, convert (JPEG / PNG / WebP / TIFF), apply an Adjust look or preset, rename from a pattern, choose dpi, and keep or remove the details inside (removed by default). Ready-made presets and your own.",
              "<b>🗜 Compress</b> (from ProtoArt Press): copies that are usually 40–70% smaller and measured to look the same. One picture: <b>🗜 Compress / export…</b> in its details." ] },
            { h: "Also new in this version", items: [
              "<b>Almost any picture opens</b>: besides PNG, JPEG and WebP you can drop or pick iPhone <b>HEIC</b>, <b>TIFF</b>, <b>AVIF</b>, <b>PSD</b>, GIF, BMP, JPEG 2000 and camera <b>RAW</b> (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2…) in Tools, Image to image, the image prompt and image → prompt. macOS's own reader converts them once (nothing extra is installed).",
              "<b>Gated Hugging Face models</b>: save a Hugging Face access token (Models tab) to download models whose owners ask you to accept a licence; the app shows the steps when one needs it.",
              "<b>Tidier screens</b>: Create's sections fold down to one line (click a section's name; the line still shows what's set), and Settings starts with only This Mac and Models open.",
              "<b>Civitai sign-in</b>: Civitai now needs a free account for most downloads. The Civitai card asks for your API key once, with the steps to get it; see Troubleshooting → Civitai downloads for every message you might see.",
              "<b>From Civitai</b> (Models tab): paste a link to download LoRAs, embeddings and SD 1.5 / SDXL / Pony checkpoints straight into the right folder.",
              "LoRAs and embeddings show only for the matching model type; <b>Fix a type</b> and <b>Show unknown type</b> (Models tab) sort out files the app couldn't identify, and <b>Sort into type folders</b> tidies a folder.",
              "<b>Moved a folder?</b> (Settings): moved your models / LoRAs to another drive? Point the app at the new place and everything keeps working.",
              "<b>Hide NSFW</b> and <b>Hide People</b> (top of the window), a default model, deleting models, and getting a model ready in memory before you press Generate (Settings → Speed)." ] },
          ] },
        { id: "create", go: "create", icon: "✨", title: "Create", summary: "Make images from a text prompt",
          sections: [
            { h: "Prompt tools", items: [
              "Under the prompt, one row of keys: <b>✨ Enhance</b> (rewrite the prompt with more detail now, or every time you Generate), <b>📒 Saved</b> (prompts you kept), <b>🖼 From image</b> (turn a picture into a prompt; or drop a picture on the prompt box), <b>🎲 Random</b> (a random prompt: a woman, a man, two people, a landscape or an animal), <b>{ } Wildcards</b> and <b>📋 List</b>. Each opens its own panel underneath; click it again to close it.",
              "<b>🎲 New random prompt every time I press Generate</b> (in Random's panel): each picture gets a fresh random prompt. <b>New random mix</b> under LoRAs does the same for LoRAs (every LoRA in the list is swapped).",
              "<b>📒 Negative prompts</b>: the small menu above the negative prompt box keeps negative prompts on their own (＋ save), to use with any prompt. The <b>Seed</b> is under the negative prompt.",
              "<b>Reset all</b> (the blue lever at the top): flip it to clear both prompts, put every setting back to the start and fold all the sections. The lever springs straight back to rest; there's no undo, so save a preset first (PRESET → Save current settings as) if you might want them again." ] },
            { h: "Making an image", items: [
              "Pick a <b>model</b> (filter by type or search). Not downloaded yet? It downloads once, then works offline.",
              "Type a <b>prompt</b> describing what you want, and a <b>negative prompt</b> for what to avoid.",
              "Press <b>Generate</b> or ⌘↵. Press it again to queue more jobs; each one shows its time left.",
              "Your new pictures appear on the right with the <b>model that made them</b> along the bottom. <b>✕</b> (point at a picture) takes it off the screen and <b>clear all</b> empties it; they all stay in the Library. Pictures from Tools (enhanced, face-restored) stay in Tools and the Library.",
              "<b>Too crowded?</b> Click a section's name (Model, Prompt, LoRAs, Image prompt, Size, steps &amp; seed, Extras) to fold it down to one line, like DiffusionBee; the folded line still shows what's set (for example <i>512×768 · 25 steps · CFG 7</i>). Click again to open it. <b>collapse all / expand all</b> at the top does every section; the app remembers what you folded." ] },
            { h: "Knobs, switches & meters", items: [
              "<b>Knobs</b> (Steps, Guidance, Images): <b>drag</b> up or right to turn up, down or left to turn down; hold <b>Shift</b> while dragging for fine control. You can also <b>scroll</b> over a knob, or click it and use the <b>arrow keys</b> (Page Up / Page Down jump further, Home / End go to the lowest / highest).",
              "<b>Double-click a knob</b> to put it back to its default (the model's own steps and guidance, 1 image). <b>↺ Reset to defaults</b> under the knobs resets everything in Create to the model's defaults (size, steps, guidance, sampler, 1 image, random seed, recommended CLIP skip, extras off) and keeps your prompt, model and LoRAs; <i>undo</i> brings your settings back.",
              "<b>Type an exact value</b>: click the amber number under a knob, type, and press Return. Esc cancels. Knobs click in steps (whole numbers for Steps and Images, 0.5 for Guidance); typed values are kept exactly, e.g. Guidance 7.3. The knob and its number always show the same value.",
              "<b>Max size</b> switch: click it to step to the next size, drag to turn it, or click a label (Small, Medium, Large). If you type your own width and height it shows <i>custom size</i>.",
              "<b>Toggle switches</b>: lever up = on, down = off. Buttons that are on light up amber.",
              "<b>Top panel</b>: Power, Busy and Fault lights; amber clocks for the time left on the current job and on the whole queue; the <b>VU meter</b> needle shows the same progress as the progress bar, and its lamp warms up while a job runs. The clocks and meter are hidden on Library, Settings and How to." ] },
            { h: "Prompt helpers", items: [
              "<b>Negative prompt ✨ suggest / 🎲 random</b> (next to <i>Negative prompt</i>, in Create and Image to image): writes a negative prompt that fits your prompt and model: tags for SD 1.5 / SDXL / Pony / Illustrious (with each one's own low-quality tags), plain words for ERNIE and FLUX, and the opposite of the style you asked for (a cartoon prompt avoids “realistic, photo, 3d”). 🎲 random adds a few extra fixes each time; ↩ undo puts yours back. Your embeddings stay in.",
              "<b>🎲 Random prompt</b> and <b>✨ enhance</b> also write the matching negative prompt, unless you typed your own. With FLUX, FLUX.2 or ERNIE the random prompt is a scene written in sentences; with SD 1.5 / SDXL it's tags. Enhance writes tags for SDXL and Pony too.",
              "<b>📒 My prompts</b> (under the prompt): <b>＋ save</b> keeps the prompt and negative prompt under a name; pick one from the list to put it back; <b>manage…</b> to use or delete them (asks first). Kept on this Mac.",
              "<b>🎲 Random · { } Wildcards · 📋 List</b> are the three keys under the prompt: a random prompt, wildcards ({red|blue}, __name__) and a list of prompts (one job per line).",
              "<b>✨ Enhance prompt</b> turns a short idea into a full prompt in the model's style: tags for SD 1.5, a detailed description for SDXL and ERNIE (small built-in language model, ~1 GB the first time).",
              "<b>🔄 Contrast prompt</b> (next to ✨ Enhance now, and in Image to image) suggests the same scene turned into its striking opposite: another season, weather, time of day or material (a waterfall → frozen blue-white ice with icicles; a summer street → winter after light snowfall). ↩ undo brings your prompt back.",
              "<b>Your own defaults</b>: set the sampler, steps and guidance you like, then <b>💾 Save as this model's defaults</b> (next to Reset). Picking that model again loads them; ↺ my defaults puts them back after changes, <i>forget</i> clears them. For a LoRA, set its weight and press ☆ under the active LoRAs: it starts at that weight from then on.",
              "<b>Restore faces presets</b> (CodeFormer): Balanced (fidelity 0.5), Keep likeness (0.8, for real people or person LoRAs), Heavy repair (0.3, tiny or badly distorted faces), Subtle touch-up (0.7, blended 60% so skin texture stays), Old photo (0.4) and Gentle (0.85, already clear faces).",
              "<b>enhance now</b> (button) rewrites the prompt once, right away, and puts it in the prompt box so you can read it, edit it or <b>↩ undo</b> before generating. Best for polishing one prompt by hand.",
              "<b>Enhance prompt when generating</b> (checkbox) leaves your prompt box alone and enhances your short idea behind the scenes every time you press Generate, so each image gets a fresh variation. The prompt that was used shows under the box afterwards and is saved with the image. Best for exploring many variations; it adds a few seconds per image.",
              "<b>🎲 Random prompt</b> fits the model: for <b>FLUX, FLUX.2 and ERNIE</b> (which read sentences) it writes a short scene description (who, what they wear and do, where, when, the light and the look); for <b>SD 1.5 and SDXL</b> it writes Danbooru-style tags with a quality preset. Pick the subject and how detailed it should be next to it.",
              "Drop an image on the prompt box to turn it into a prompt: <b>Describe</b> (sentences) or <b>Danbooru tags</b>." ] },
            { h: "Long prompts & word weights (SD 1.5 / SDXL)", items: [
              "Prompts can be as long as you like: past 75 tokens they're read in parts, so nothing is cut off. <b>BREAK</b> starts a new part.",
              "<b>(word:1.3)</b> makes a word stronger, <b>(word:0.7)</b> weaker; <b>(word)</b> = ×1.1, <b>((word))</b> = ×1.21, <b>[word]</b> = ÷1.1. Same as A1111, Draw Things and Civitai prompts, so copied prompts look as intended. Use <b>\\(</b> for a real bracket." ] },
            { h: "Wildcards & prompt lists", items: [
              "<b>{red|blue|green} dress</b> picks one option for each image; <b>{2$$a|b|c}</b> picks two. Options can be nested.",
              "<b>__hair__</b> picks a random line from <i>hair.txt</i> in your wildcards folder (<b>{ } Wildcards</b> shows your files; sub-folders work as <i>__clothes/tops__</i>). The same seed gives the same words again, and the words used are saved with each image.",
              "<b>📋 Prompt list</b> queues one job per line (type them or load a .txt file), each with your current settings." ] },
            { h: "🎲 Random LoRAs", items: [
              "Picks random LoRAs that work with the chosen model: set how many (e.g. 1 to 2) and the weight range.",
              "Their trigger words are added to the prompt (if auto-add is on in Settings) and taken back out on the next pick. LoRAs you added yourself stay.",
              "<b>new pick every Generate</b> re-rolls each time you press Generate. It only ever picks LoRAs made for the chosen model's type, and skips ones hidden by Hide NSFW / Hide People." ] },
            { h: "Image prompt (IP-Adapter)", items: [
              "Drop a reference picture to borrow its look without training a LoRA: <b>Style & look</b> (colours, style, mood), <b>Stronger</b> (closer to the picture) or <b>Face</b> (a person's face from a close-up).",
              "<b>How much to borrow</b>: 0.4–0.6 blends it with your prompt; 0.8+ follows the picture closely. Works in Image to image too.",
              "Downloads ~2.6 GB the first time. On 8 GB Macs it works best with SD 1.5 at 512–768 px." ] },
            { h: "Extras: Hi-res fix & Face detailer", items: [
              "<b>🔍 Hi-res fix</b> makes the picture at the model's natural size, then redraws it ×1.5 or ×2 bigger. You get real detail at large sizes without the doubled heads you get from generating big directly. <b>redraw</b> 0.35–0.5 adds detail and keeps the picture.",
              "<b>🙂 Face detailer</b> (like ADetailer) finds each face, redraws it at full detail with your model and prompt, and blends it back: faces stay in the picture's style. Add a face prompt (e.g. <i>detailed eyes, smile</i>) if you like.",
              "Both add time: Hi-res fix about as much as the picture itself; the face detailer a few seconds per face." ] },
            { h: "🔬 Compare settings", items: [
              "Pick the values with the <b>blue keys</b>: every key you light up adds a column (Across) or a row (Down), and the button counts the pictures (36 at most). Each setting has its own keys (CFG, steps, samplers, seeds, CLIP skip, LoRA weights, which LoRA, change strength); <i>or type your own</i> is underneath. The blue follows your look.",
              "Makes the same picture with different settings, side by side in one labelled grid: CFG, steps, sampler, seed, CLIP skip, a LoRA's weight, which LoRA, or change strength.",
              "Pick <b>Across</b> and (optionally) <b>Down</b>, type values separated by commas, or a range like <i>3-12 (+3)</i>. Everything else (prompt, model, LoRAs, size) comes from the Create tab, with one seed for every picture.",
              "Up to 36 pictures. The grid and each picture are saved to the Library.",
              "The switch next to <b>Compare settings</b> shows whether it's on; click the heading or <b>Turn off compare</b> to close it." ] },
            { h: "LoRAs & embeddings", items: [
              "Add as many LoRAs as you like, each with its own weight (−2 to 2).",
              "<b>🏷 Your own tags</b>: in Models → 🏷 LoRA tags, search your LoRAs, tick some (or <i>select all</i>), type a tag such as <i>pencil</i> and press <b>＋ Tag</b>. Then in Create or Image to image pick <b>🏷 pencil</b> in the LoRA list's menu, or type <b>#pencil</b> in its search box, to see only those. A LoRA can have several tags; <b>✕</b> on a tag takes it off, and <b>Rename</b> renames a tag everywhere (renaming to an existing tag merges them). <i>🏷 Make or manage your own tags…</i> at the bottom of the list's menu opens the panel.",
              "<b>Reading the LoRA list</b>: under each name are its type and size. <b>BEE</b> means the LoRA comes from your DiffusionBee folder (read where it is); ⚠ after the type means it may not match the model you picked. Point at a row to see it all spelled out.",
              "<b>Only LoRAs made for the chosen model's type are listed</b>: an ERNIE model shows only ERNIE LoRAs, SD 1.5 only SD 1.5 ones, and so on (Pony / Illustrious LoRAs count as SDXL). Switching to another type of model removes LoRAs that don't fit.",
              "<b>Fix a type</b> (Models tab): <b>Show all</b> lists every LoRA and base model file; <b>🔒</b> locks a row so a stray click can't change its type (a type you pick locks itself; click the lock to change it).",
              "LoRAs whose type the app couldn't work out aren't shown for any model; the list says how many and <b>Give them a type →</b> takes you to fix them. <b>LoRA missing or the wrong type? Fix it →</b> does the same for one that's filed under the wrong type.",
              "Markers after a name: red <b>⚑</b> = NSFW, amber <b>●</b> = tagged People. Point at a row and click either one to set or clear it.",
              "Trigger words are found automatically and can be added to the prompt with one click; save your own with <i>add / edit</i>.",
              "<b>Embeddings</b> (textual inversions like <i>EasyNegative</i>) work with SD 1.5 and SDXL models (Pony / Illustrious / NoobAI too): click one above the negative prompt, or type its name in either prompt. Only embeddings made for the chosen model's type are shown.",
              "Add embeddings in <b>Settings → Embeddings</b>: <i>Add embedding files…</i> copies .safetensors / .pt files in, <i>Add folders…</i> uses a folder where they are, and <b>From Civitai</b> (Models tab) downloads them (Civitai calls them “Textual Inversion”). Embeddings in your LoRA folders and DiffusionBee's are found too." ] },
            { h: "Size, steps & speed", items: [
              "The <b>Max size</b> switch (Small / Medium / Large) sets the size for the model; presets set the shape; 🔗 links width and height.",
              "Each sampler shows its <b>recommended steps</b>; picking one sets them for you. Under the Sampler list, <b>⚡ Fastest</b> and <b>★ Best quality</b> show the recommended samplers for the chosen model's type (outlined in faint blue; Pony / Illustrious models lead with Euler a). The list itself is grouped the same way.",
              "<b>⚡ Fast mode</b> (SD 1.5 / SDXL) makes images in 6 steps. <b>Presets</b> (at the top of the window in Create) save a whole setup, LoRAs included; switch on <i>prompt</i> to save the prompt too.",
              "<b>❚❚ Pause</b> holds the picture being made at its next step (and the queue); <b>▶ Resume</b> carries on where it was, nothing lost. Memory stays in use while paused, so on an 8 GB Mac Stop frees more if you need the Mac for something else.",
              "<b>⏭ Skip image</b> skips one image of a batch, <b>■ Stop</b> stops the job, <b>Stop all</b> clears the queue (and ends a pause).",
              "<b>⏳ This will be slow on this Mac</b>: on a Mac with less than 24 GB, pressing Generate with settings that are too heavy for its memory (a ControlNet above 512 px on 8 GB, an SDXL ControlNet, full-precision SDXL, SDXL with Hi-res fix, very big sizes) shows what will be slow and why, with buttons for the faster choice. <b>Go ahead anyway</b> starts it as it is. <i>Don't warn me about this again</i> switches that warning off; <b>Settings → Speed</b> brings them back.",
              "<b>▲ / ▼</b> beside a waiting job (point at it) move it up or down the queue: the top one starts next.",
              "<b>⤵ Run the others first</b> (top of the queue, or the ⤵ on the running job) moves the job that's running to the back of the queue, so the jobs waiting behind it run now. The pictures it already finished are kept; the rest are made when its turn comes again (the one it was in the middle of starts over).",
              "Create and Image to image each keep <b>their own prompt and negative prompt</b>: switching tabs shows that tab's own.",
              "Next to the model's name: <b>getting ready…</b> while it loads in the background, <b>● loaded, ready</b> when the next picture can start straight away (Settings → Speed)." ] },
            { h: "Extras", items: [
              "<b>VAE</b> (SD 1.5 and SDXL): the part that turns the model's picture into pixels. Another VAE can fix washed-out colours or blotches. Only VAEs made for the model's type are listed, each model remembers its own, and <b>⤓ get</b> downloads the usual one in one click (vae-ft-mse for SD 1.5, SDXL fp16-fix for SDXL; MIT licence). VAEs from Civitai go to the VAE folder too.",
              "<b>CLIP skip</b> (SD 1.5 / SDXL): reads your prompt from an earlier layer of the text encoder. <b>Off</b> is normal; <b>2</b> is what many anime-style models and LoRAs were made with and often follows their style better. 3–4 is looser.",
              "Each model starts on the CLIP skip it's meant for (marked <b>★</b>): <b>2</b> for anime-style SD 1.5 models (Anything, Counterfeit, MeinaMix…), <b>Off</b> for photo and general models. SDXL always reads prompts this way, so its scale starts at <b>Standard</b>. Change it any time; your choice is remembered for that model, and <i>use recommended</i> puts it back.",
              "If a LoRA's file says which CLIP skip it was trained with, a <i>use</i> link appears under the setting. It's saved with each image and in presets, and appears in the image's details.",
              "<b>Restore faces</b> fixes faces right after each image is made. Choose <b>CodeFormer</b> (the default: stronger repair, with a <b>Fidelity</b> slider; non-commercial licence)." ] },
          ] },
        { id: "models", go: "create", icon: "🧠", title: "Models & their defaults", summary: "Why each model starts with different settings, and CLIP skip",
          sections: [
            { h: "Every model starts ready to use", items: [
              "Picking a model sets its own <b>size</b>, <b>steps</b>, <b>guidance (CFG)</b> and <b>sampler</b>: SD 1.5 at 512 px, SDXL at 1024 px, ERNIE at 768 px on 8 GB Macs (1024 on bigger ones), with the step count each one is made for.",
              "Each sampler shows its <b>recommended steps</b>; <b>⚡ Fast mode</b> switches SD 1.5 / SDXL to 6 steps.",
              "Everything can be changed. To get a model's defaults back, pick another model and then pick it again." ] },
            { h: "CLIP skip: what it is", items: [
              "The text encoder reads your prompt in layers, each one more refined than the last. <b>CLIP skip</b> takes the prompt's meaning from an earlier layer instead of the very last one.",
              "Models are trained with one setting and look best with the same one. Anime-style SD 1.5 models (most descend from the NovelAI model) were made with <b>2</b>; photo and general models with <b>Off</b> (1).",
              "Using the wrong one doesn't break anything, but images can follow the prompt less well or lose the model's intended style. 3–4 is looser and more stylised." ] },
            { h: "CLIP skip: what the app does for you", items: [
              "Each model starts on its recommended setting, marked <b>★</b>: <b>2</b> for anime-style SD 1.5 models (Anything, Counterfeit, MeinaMix, AbyssOrangeMix…), <b>Off</b> for photo and general ones (base SD 1.5, Realistic Vision, DreamShaper…). It's worked out from the model's name, so check it for models with unusual names.",
              "<b>SDXL</b> always reads prompts the way other apps call “CLIP skip 2”, so its scale starts at <b>Standard</b>; 3 and 4 go further. The numbers match other apps, so shared prompts and settings look the same here.",
              "<b>Your choice is remembered per model.</b> Click another value and that model keeps it next time; <i>use recommended</i> puts it back.",
              "If a <b>LoRA</b> you add says in its file that it was trained with a different CLIP skip, a <i>use 2</i> link appears so you can match it.",
              "It's saved with every image (shown in the image's details and in its embedded settings, as “Clip skip: 2”), and in presets. ERNIE and the fast engine don't use it, so it's hidden for them." ] },
          ] },
        { id: "i2i", go: "i2i", icon: "🖼", title: "Image to image", summary: "Change a picture, or follow its structure with ControlNet",
          sections: [
            { h: "🔁 Face swap, 🔗 model chain, Union ProMax", items: [
              "<b>🔁 Face swap</b>: put the picture to change in 1 · Picture, pick 🔁 Face swap, drop a clear front-facing photo of the face to use, choose <i>All faces</i> or <i>The main face</i>, and tick that you have permission. <i>Sharper (256 px)</i> is for big, close-up faces. <i>Tidy the new face</i> runs CodeFormer (fidelity 0.7, blended 60%). ~460 MB the first time. The swap model (inswapper_128) is for non-commercial use only; every result is checked and explicit results aren't saved.",
              "<b>🔗 Model chain</b> (🔄 Change it): ＋ Add a model for each step, set how much each may change (0.3–0.5 keeps the picture), write the prompt, and press ▶ Run chain. Each model redraws the last one's result: e.g. an SD 1.5 picture → SDXL for detail → FLUX.1 Schnell or ERNIE for the finish. Each step has its own <b>LoRA</b> (only ones made for that step's model type are listed) and <b>extra words</b> (picking a LoRA adds its trigger words): e.g. step 1 SD 1.5 with your character LoRA, step 2 FLUX.1 to finish it. ↑ ↓ reorder the steps; each step's result appears beside it (click it to start from that picture). Each step's picture is kept in the Library.",
              "<b>ControlNet Union ProMax</b> (SDXL models): in 2 · ControlNet pick a <i>Union ProMax</i> type (Canny, Depth, OpenPose, Line art, Scribble, Soft edges, Tile). One ~2.5 GB download covers them all. On 8 GB use 768–1024 px.",
              "<b>Which one should I use?</b> The same person in different pictures → Image prompt · <b>Face</b> (a close-up photo). A picture's style, outfit or colours → Image prompt · <b>Stronger</b>. A pose, drawing, silhouette or layout → <b>ControlNet</b> (OpenPose for poses, Line art or Scribble for drawings, Canny or Depth for shapes). They combine: Face + an OpenPose ControlNet keeps the person and copies the pose.",
              "<b>Any model can start from a picture</b> now except FLUX.2 Klein (which uses ✨ Edit with words): SD 1.5, SDXL, Anima, FLUX.1 Schnell and ERNIE (fast engine)." ] },
            { h: "Starting from a picture", items: [
              "<b>✨ Edit with words</b> uses FLUX.2 Klein (it's at the top of the model list while that mode is on). Its <i>✨ enhance now</i> turns a quick note into a clear instruction. The other modes have <b>✨ Enhance prompt</b> and the negative prompt's ✨ suggest / 🎲 random too.",
              "Pictures made in Image to image stay in Image to image and the Library; they don't fill Create's screen. <b>✕ Clear results</b> takes them off the screen. Drop another picture on 1 · Picture to replace it.",
              "<b>Inpainting models</b> (files like <i>dreamshaper_8Inpainting</i>) are made for filling in painted areas: the app recognises them, keeps them out of Create, and Inpaint and Expand use one automatically when your model at the top is the same type (switch: <b>🖌 Paint with an inpainting model</b>).",
              "Drop a picture, or use <b>🖼 Img2img</b> on any Library image.",
              "The <b>model</b> for Image to image is chosen at the top of the window, next to the brand plate.",
              "<b>Change strength</b>: low (0.3) keeps most of the original, high (0.8+) changes a lot.",
              "Pick the model right here; steps and CFG are next to it.",
              "<b>✨ Edit with words</b> (FLUX.2 Klein): describe the change in the prompt — “make the jacket red”, “add sunglasses”, “replace the sky with a sunset”, “turn it into a watercolour” — and press <b>✨ Edit</b>. It keeps the rest of the picture. Paint over a part first and only that part can change (everything else stays exactly as it was): best for precise edits. Add up to 3 <b>more pictures</b> to borrow from (“dress her in the outfit from the second picture”, “paint it in the style of the second picture”). It always uses FLUX.2 Klein, whatever model Create has (4.6 GB the first time), takes about as long as one FLUX.2 picture, and ignores LoRAs and the negative prompt.",
              "<b>🩹 Remove</b>: paint over what you want gone and press 🩹 Remove; it's filled in from its surroundings (no prompt or model needed). <b>✨ Smart fill</b> handles objects, people and patterns (~208 MB download the first time); <b>⚡ Quick fill</b> is instant for dust, spots and thin lines. Cover the thing fully, a little past its edges.",
              "<b>LoRAs</b> card: add a LoRA (search the list), set its weight with the slider, ✕ removes it. They're the same LoRAs as in Create (both tabs share the model); trigger words and 🎲 Random LoRAs are in Create." ] },
            { h: "🖌 Inpaint", items: [
              "<b>Selection tools</b> (everywhere you paint over a picture): <b>🖌 Brush</b>, <b>🧽 Erase</b>, <b>➰ Lasso</b> (draw around an area; it fills when you let go), <b>⬠ Polygon</b> (click the corners; click the first point, double-click or press Return to close), <b>🪄 Wand</b> (click a colour; the tolerance slider sets how similar), <b>⬚ Object</b> (drag a box around something and its outline is found). <b>＋ Add / － Subtract</b>, or hold ⌥ Option to subtract. Keys: B E L P W O; Esc cancels.",
              "<b>Brush size</b>: a ring the size of the brush follows the pointer (dashed when erasing); moving the size slider shows it on the picture, and the metal well next to the slider shows the size.",
              "Choose <b>Inpaint</b>, paint over what you want to change (Paint / Erase, brush size, invert, clear), and describe what should be there.",
              "<b>Change strength</b> 0.7–0.9 replaces what's painted; 1.0 starts from scratch there. <b>Edge softness</b> blends the edges.",
              "<b>Redraw only the painted area at full detail</b> is best for small things like hands, eyes or text: it redraws a box around your strokes at full resolution and pastes it back.",
              "Works with your normal SD 1.5 and SDXL models; everything you didn't paint stays exactly as it was." ] },
            { h: "↔ Expand: getting clean results", items: [
              "<b>One direction at a time</b>: ★ make it wider (or ★ make it taller), press <i>Use the result as the new picture</i>, then expand the other way. Expanding on every side at once is the hardest case and is more likely to give seams, repeated objects or smeared corners; a note under the sizes says so.",
              "<b>A moderate amount</b>: up to about a third of the picture per side. Bigger additions tend to repeat things from the picture or leave blurry areas.",
              "<b>💡 suggest from the picture</b> (under the prompt, with FLUX.2) describes your picture on this Mac and asks for more of the same scene around it; edit it or press ✨ enhance now. A prompt that says what should be there (“more tables and chairs, a window on the left”) helps more than none.",
              "If the result still has a seam, try another seed: each one fills the space differently." ] },
            { h: "The Image to image layout", items: [
              "From the top: <b>1 · Picture</b> (your picture and the mode: Edit with words, Change it, Inpaint, Expand, Remove), then <b>Sampler, steps &amp; guidance</b> and <b>LoRAs</b> (both fold up; hidden when FLUX.2 Klein does the work), <b>2 · ControlNet</b> and <b>3 · Prompt</b>. The preview and results are on the right." ] },
            { h: "↔ Expand (outpaint)", items: [
              "Choose <b>Expand</b> and how many pixels to add on each side (or use <i>make it wider / taller</i>). The new area is filled in to match.",
              "<b>Fill the new area with ✨ FLUX.2 + outpaint LoRA</b> (the default): works on any picture, whatever model made it (SD 1.5, SDXL, FLUX, a photo). It uses FLUX.2 Klein (the Edit with words model) with <b>fal's outpaint LoRA</b> (Apache-2.0, free to use, 76 MB, downloaded once). The picture goes in with a green border where the new area will be, and the model fills it in. The prompt is optional: add what should appear around the picture. Your picture itself is kept exactly as it was.",
              "Choosing ✨ FLUX.2 + outpaint LoRA puts <b>FLUX.2 Klein</b> in the model menu at the top for you (picking another model there switches Expand to that model). Its <b>✨ enhance now</b> link (under the prompt) turns a quick note like “more of the street” into a clear description of what should fill the new space. The result appears in Image to image, next to your picture, and in the Library.",
              "<b>The model at the top</b>: the SD 1.5 / SDXL model you picked (or its inpainting model) fills the new area instead. Describe the whole scene in the prompt, including what should appear around the original.",
              "Expanding a little at a time (up to about half the picture's size per side) gives the best results; expand again for more." ] },
            { h: "ControlNet (SD 1.5, and your own SDXL ones)", items: [
              "<b>Recommended settings</b>: picking a ControlNet puts in the settings it works best with, shown in a box right under the picker (<b>undo</b> puts yours back). If a setting would spoil the result, the box turns orange and says which; <b>Use recommended settings</b> fixes it.",
              "<b>The guide</b>: when you pick a ControlNet, a panel on the right (above the preview) says what it does, what picture to give it, which of your models suit it best (★, click to switch) and how to use it step by step. <b>Use these settings</b> sets the strength, timing and picture options it works best with.",
              "<b>Your own ControlNets</b> (SD 1.5 or SDXL), e.g. an SD 1.5 or SDXL canny / depth / openpose from Civitai: paste the Civitai link in Models → Add from Civitai, then pick it in the ControlNet list. The list shows the ones that fit the model you picked: SD 1.5 models get the built-in ones and your SD 1.5 files, SDXL / Pony / Illustrious / NoobAI models get your SDXL files. If its name says <i>canny, depth, lineart, openpose, softedge, scribble</i> or <i>tile</i>, that map is made from your picture; otherwise it follows your picture as it is. SDXL ControlNets are big, so they're slow on 8 GB Macs. Control-LoRA, LLLite and Union files, and QR code / pattern ControlNets, can't be used in this version (the Models tab says so next to them).",
              "<b>Tile</b> adds detail but keeps the picture: great with low change strength for sharpening.",
              "<b>Canny / Line art / Soft edges</b> keep the outlines; <b>Depth</b> keeps the 3-D layout; <b>Scribble</b> turns a sketch into a picture.",
              "<b>How closely to follow the map</b>: 1.0 is normal. <b>When it guides</b>: <i>Layout only</i> sets composition and leaves details free.",
              "<b>Preview the map</b> shows what the ControlNet sees. Your DiffusionBee ControlNets are used when you have them.",
              "<b>Depth</b> uses <b>Depth Anything V2</b>: pick Small (fast, ~100 MB), Base or Large (more detail; non-commercial licence).",
              "<b>OpenPose</b> copies a person's pose: it finds each person and their joints (~550 MB the first time) and draws the stick figure the ControlNet follows. Preview the map to check it found everyone. Already have a pose image? Tick <i>My picture is already a control map</i>." ] },
            { h: "Live preview", items: [
              "The image forming shows in the tab that started it: Image to image jobs here, Create jobs in Create. If a Create job is running, this tab says it's waiting its turn." ] },
            { h: "Results", items: [
              "The Preview window shows the image as it forms, then the result; tick <b>compare with the original</b> to see both.",
              "<b>Use the result as the new picture</b> lets you refine it again." ] },
          ] },
        { id: "tools", go: "tools", icon: "🧰", title: "Tools", summary: "Enhance, 🎚 adjust, restore faces, 🩹 remove objects, ✒️ vectorize, 🔌 patchbay, image → prompt, convert files",
          sections: [
            { h: "🔌 Patchbay: run the tools in your own order", items: [
              "<b>＋ Add a module</b> (above the panel) puts in another copy of Faces, Enhance, Remove or Depth, like a second rack unit: cable it in to run that tool twice in one chain (for example Restore faces, then Restore faces again). <b>✕ last added</b> takes the newest copy out.",
              "<b>Each module's settings</b>: press <b>SELECT</b> at the bottom of a module. Its rack unit below lights up and shows that module's settings, with a ⚙ badge naming it; a copy you added, such as FACES 2, keeps <b>its own settings</b> in the same rack unit (model, strength, fidelity for faces; upscale, detail, steps for enhance…), so each copy can do something different. Press SELECTED again to finish.",
              "<b>✕</b> (top right of a module) takes it off the rack; <b>Add a module → ↩ put … back</b> returns it. <b>🔎 DETAIL</b> redraws lightly with the Create model and one LoRA (an “add detail” LoRA works well; change strength 0.2–0.4 keeps the picture). It needs an SD 1.5 or SDXL model in Create.",
              "<b>🛡 Protect areas…</b>: paint over parts the rack must not change (a face you like, a logo), press ✓ Done protecting, then ▶ Run patch: those parts are put back from your original at the end. The picture forming shows in the preview while it runs.",
              "<b>Saving a signal flow</b>: type a name next to <b>💾 Save flow</b> and press it: the cables, the copies you added, the module order and the copies' own settings are kept. Load it again from <i>load a saved flow…</i>; 🗑 deletes the saved one (the patchbay stays as it is).",
              "<b>Moving modules</b>: drag a module's name plate left or right, or point at it and press <b>‹</b> / <b>›</b>. PICTURE stays first and RESULT last. Moving a module doesn't change what runs: the <b>cables</b> set the order the picture goes through (shown in the amber readout). <b>↺ reset</b> puts the modules back in their places and pulls out every cable.",
              "Switch on <b>PATCHBAY</b> (the toggle at the top of the right-hand column). The back of the rack appears: every tool has an <b>IN</b> and an <b>OUT</b> jack. <b>PICTURE</b> (your picture) has only an OUT, <b>RESULT</b> (the Library) only an IN.",
              "<b>Plug cables</b>: drag from an OUT jack to an IN jack (or click one, then the other). The picture goes PICTURE → along the cables → RESULT, so the order of the cables is the order the tools run in. The amber readout under the panel spells it out, for example <i>PICTURE ▸ REMOVE ▸ FACES ▸ ENHANCE ▸ RESULT</i>. Green lamps show the tools that are in the chain.",
              "<b>Unplug</b>: drag a plug away from its jack, or double-click a cable. One cable per jack: plugging into a used jack replaces its cable. Loops aren't allowed, and <b>VECTOR</b> has to be last (it makes a drawing, not a picture). The patchbay starts with <b>no cables</b>. <b>example</b> plugs in PICTURE ▸ ENHANCE ▸ FACES ▸ RESULT; <b>clear cables</b> pulls them all out; <b>↺ reset</b> also puts the modules back in their places.",
              "<b>▶ Run patch</b> runs each patched tool in turn, each on the previous one's result, using that tool's own settings in its module below (upscale, face model, depth look, colours…). <b>REMOVE</b> uses what you painted on the picture: if nothing is painted yet, Run patch turns Paint on so you can paint first. Every step is saved in the Library; the last one shows in the picture window to compare. <b>■ Stop</b> stops after the current step.",
              "Your cables are remembered on this Mac, and the patchbay stays hidden until you switch it on." ] },
            { h: "🩹 Remove objects and ✒️ Vectorize", items: [
              "<b>Remove objects in Tools</b>: press <b>🖌 Paint</b> in the module, paint on the picture on the left (all the selection tools work), choose <b>✨ Smart fill</b> (rebuilds what was behind it) or <b>⚡ Quick fill</b> (instant, for small spots), and press <b>🩹 Remove</b>.",
              "<b>🩹 Remove objects</b>: opens the picture in Image to image in Remove mode. Paint over what you want gone and press <b>🩹 Remove</b>; it's filled in from what's around it. The result is saved as a new picture in the Library (“Objects removed”); the original stays.",
              "<b>✒️ Vectorize</b>: makes an SVG of flat colour shapes, saved next to your pictures (Show in Finder opens it). <b>Colours</b>: fewer gives bolder, simpler shapes (logos: 2–6; illustrations: 8–16). <b>Detail</b>: Simple, Balanced or Detailed. <b>Smooth curves</b> off gives straight edges, for pixel art and geometric logos. Photos come out like a screen-printed poster.",
              "<b>↺ Reset Tools settings</b> (top of the right-hand column) puts Enhance, Restore faces, Depth and Vectorize back to their starting settings; <b>undo</b> brings yours back." ] },
            { h: "⏻ Engine: force stop & force quit", items: [
              "Both buttons are at the top of the window when the Tools tab is open.",
              "<b>■ Stop</b> now really stops: if the picture hasn't stopped by itself within a couple of seconds (one step can take a minute on a busy Mac), the engine is force-stopped and restarted for you, and the jobs still waiting carry on. Training is the exception: its Stop finishes the current step so it can save what it learned, and <b>Force stop now</b> appears if you don't want to wait.",
              "<b>⟲ Force stop &amp; restart</b> stops the engine at once (it doesn't wait for the current step), frees all its memory and starts it again. The job that was running is lost; waiting jobs are added back.",
              "<b>⏻ Force quit</b> closes the whole app immediately, even in the middle of a job (it asks once to confirm).",
              "Use them when a picture takes far too long, the Mac feels frozen, or ■ Stop doesn't respond. Pictures, settings and models are never affected." ] },
            { h: "Enhance & upscale", items: [
              "<b>Compare the result</b>: after any tool, the picture window shows <b>Original / Result / Split / Side by side</b> (drag the line in Split). <b>✕ Clear results</b> takes the results off the screen (they stay in the Library). The tools are rack modules: <b>Run chain</b> does Enhance then Restore faces in one go; Enhance's less-used settings are under <i>fine-tune</i>.",
              "Redraws the image bigger with fine detail (Finegrain enhancer, about 3.5 GB the first time).",
              "<b>Creativity</b> = how much new detail it may invent. <b>Resemblance</b> = how closely it sticks to the original.",
              "<b>Look</b>: <b>Subtle / Balanced / Creative</b> set both for you. <b>Speed</b>: <b>⏩ Instant</b> (the 4× sharpening upscaler only: seconds, but no new detail), <b>🚀 Turbo</b> (a distilled few-step add-on, LCM-LoRA, ~135 MB the first time: a handful of steps and no negative-prompt pass, several times faster, slightly softer detail), <b>⚡ Fast</b> (10 steps, DPM-Solver), <b>Standard</b> (18, DDIM) or <b>★ Best</b> (30, DPM-Solver: most detail). <b>1× (refine only)</b> is quickest of all.",
              "Size: the enhancer first resizes every picture to a fixed short side, then upscales by the factor you pick, so a big photo isn't slower than a small one. <b>Work at</b>: <b>768 px</b> (2× gives about 1536 px) or <b>512 px</b> (2× gives about 1024 px, less than half the work). <b>Auto</b> uses 512 px for Instant / Turbo / Fast on Macs with less than 16 GB and 768 px otherwise. The → size next to Enhance shows exactly what you'll get.",
              "Speed-ups from ProtoArt Beautify (automatic): the detail add-ons are merged into the model once; the fewest tiles that fit your Mac's memory (a 768 px picture is one tile instead of many); memory handed back to macOS after every pass; the upscaler in small pieces on 8 GB Macs.",
              "The result appears under the original with the time left; <b>■ Stop</b> works within seconds." ] },
            { h: "Restore faces", items: [
              "Fixes blurry or distorted faces. Pick the model: <b>CodeFormer</b> (the default) repairs even badly blurred or damaged faces, with a Fidelity slider (non-commercial licence); <b>RestoreFormer</b> is natural, faithful and free for commercial use.",
              "<b>Strength</b> blends the result with the original (0.7–0.8 looks most natural). <b>Fidelity</b> (CodeFormer only): low gives the cleanest face, high (0.7+) keeps more of the person's own features.",
              "Each model downloads once, the first time you use it (RestoreFormer ~280 MB, CodeFormer ~360 MB), and is checked against a pinned SHA-256. CodeFormer's licence allows <b>non-commercial use only</b>.",
              "Tick both in <b>Run on this image</b> to enhance first, then restore faces, as one finished picture." ] },
            { h: "🎚 Adjust: light, colour and filters", items: [
              "Edits a picture the way Lightroom / Camera Raw and Nik Color Efex do, live on the Mac's graphics chip. It works in the <b>Tools</b> tab: press <b>🎚 Adjust the original</b> (or <b>Adjust the result</b> after enhancing) under the picture; <b>🎚 Adjust</b> in the Library's picture details and in Image to image brings the picture to Tools for you.",
              "The picture on the left becomes the live preview, and the controls are in the <b>Adjust</b> card on the right.",
              "<b>Compare</b> with the buttons under the picture: <b>Edited</b> (with your changes), <b>Original</b> (without them), or <b>Split</b> (original on the left, edited on the right; drag on the picture to move the line). The <b>\\</b> key flips between Edited and Original.",
              "<b>✕ Close Adjust</b> (under the picture, or at the bottom of the card) leaves without saving; the picture goes back to normal.",
              "<b>Light</b>: Exposure, Contrast, Highlights, Shadows, Whites, Blacks. <b>Color</b>: Temperature, Tint, Vibrance (protects skin tones), Saturation (far left = black and white). <b>Presence &amp; effects</b>: Clarity, Sharpening, Vignette.",
              "<b>Filters</b> (Color Efex-style): Detail Extractor, Dark Contrasts, Bleach Bypass, Glamour Glow and Film Grain, applied top to bottom; reorder them with ▲▼. Click a filter's button to switch it on (amber) and open its settings; click again to switch it off.",
              "<b>Looks</b> (Punchy, Detail pop, Moody, Film, Soft glow, B&amp;W contrast) are starting points: pick one, then fine-tune. <b>Last used</b> brings back the settings you last saved. Double-click a slider to reset it.",
              "<b>My presets</b>: set the sliders and filters the way you like, press <b>＋ Save as preset</b> and give it a name. It's listed next to the Looks for every picture from then on (kept on this Mac, also after updates); click a preset to apply it, <b>✕</b> deletes it. Saving with a name you already used replaces that preset.",
              "<b>Save as new picture</b> adds an adjusted copy to the Library, linked to the original (which never changes); it appears under the original in Tools.",
              "<b>Save &amp; enhance ✨</b> saves the adjusted copy and then runs <b>Enhance &amp; upscale</b> on it in the queue (with the settings in the Enhance card), so the AI works on your adjusted picture. To enhance first and adjust afterwards, enhance as usual and press <b>🎚 Adjust the result</b>.",
              "<b>Save &amp; send to Image to image 🖼</b> saves the adjusted picture and opens it in Image to image as the starting picture. Any result in Tools (enhanced, adjusted, faces restored) also has <b>🖼 Send to Image to image</b> under it, and so does the original under the picture.",
              "<b>Several versions at once</b> (bottom of the Adjust card): pick <b>Current settings</b> and/or any of your presets, and press <b>Make N versions</b>: one picture per choice, all saved to the Library. Switch on <b>✨ Enhance &amp; upscale too</b> to put the AI in the same run: <b>Enhance first, then filters</b> enhances once and puts each version's filters on top (quickest), <b>Filters first, then enhance each</b> queues one enhance per version (each one shows in the queue with its time left)." ] },
            { h: "✏️ Edit in PhotoCraft (or your own editor)", items: [
              "<b>✏️ Edit in PhotoCraft</b> (for layers, masks and retouching; for quick light and colour work use <b>🎚 Adjust</b>) (Library picture details, Tools, Image to image results) opens a copy of the picture in <b>PhotoCraft</b>, a free open-source editor in the style of Photoshop: layers, masks, adjustments, selections, text, PSD files.",
              "Each time you <b>save</b> there (⌘S), the new version is added to your Library as <i>Edited</i>, linked to the original; “Save As”/export into the same folder works too. Your original is never changed. Click <b>Done editing</b> in the bar at the top when you're finished.",
              "PhotoCraft is a separate app (early test version): the first time, <b>Get PhotoCraft</b> opens its download page; drag it to Applications. Prefer Photoshop, Pixelmator Pro, Affinity Photo, GIMP or Krita? Pick it in <b>Settings → Image editor</b>." ] },
            { h: "Depth map (Depth Anything V2)", items: [
              "Works out how far away everything is: near = bright, far = dark. Saved to the Library, linked to the original.",
              "<b>Small</b> (~100 MB) is fast and free for any use; <b>Base</b> (~400 MB) and <b>Large</b> (~1.3 GB) show finer detail but are licensed for non-commercial use only.",
              "<b>Grey</b> is what ControlNet and other apps expect; <b>Colour</b> is easier to read by eye.",
              "<b>Use as a ControlNet depth map</b> sends it to Image to image, ready to make new pictures with the same 3-D layout." ] },
            { h: "Image → prompt & converter", items: [
              "<b>Describe</b> uses <b>Qwen3.5 4B</b>, a 2026 vision model on the Mac's GPU: a detailed prompt in sentences (subject, clothes, pose, setting, light, colours, framing, style). The first time it sets itself up and downloads ~3 GB, then it works offline. <b>Quick</b> is Florence-2 (shorter, lighter); <b>Tags</b> is the Danbooru tagger for anime models. FLUX.2 Klein can't do this: it draws pictures, it doesn't read them.",
              "Turn any image into a prompt (sentences or Danbooru tags) for the Create tab.",
              "<b>🧩 Tiled upscale</b>: pick the picture above, a size (1.5×, 2×, 3×) and how much <i>new detail</i> (0.2–0.3 keeps the picture), or one of the <b>presets</b> (Gentle, Detail, Creative, Print, Quick). It uses the model, sampler and seed from Create (SD 1.5 or SDXL) and a short detail prompt, not Create's prompt (each tile would try to draw that whole scene again); the optional <i>Detail prompt</i> adds texture or style words. The enlarged picture is redrawn in overlapping tiles that are averaged at every step, so there are no seams.",
              "<b>🗜 Smaller copy</b>: pick an SD 1.5 or SDXL model and 8, 6 or 4 bits; a new model “… · 8-bit” appears in the list. 8-bit looks the same and needs about half the memory for the UNet; 4-bit is smallest and a little softer. Not faster per step.",
              "<b>🧪 Model merger</b>: <i>Blend</i> mixes two models (amount = how much of B); <i>Add difference</i> adds what model B learned on top of its base model C to model A (for example to make an inpainting version of your favourite model). The new model is saved in your checkpoints folder; the VAE comes from model A.",
              "When the prompt is ready: <b>⧉ Copy</b>, <b>🖼 Use in Image to image</b> (opens the picture there with the prompt in the box) or <b>✍ Use in Create</b>.",
              "<b>Convert DiffusionBee files</b> makes standard .safetensors copies of your old models, LoRAs and embeddings.",
              "<b>Old .ckpt files…</b> lists the .ckpt checkpoints in your folders (or pick one) and makes a .safetensors copy in your checkpoints folder. It reads only the weights, so nothing hidden in the file can run; <i>half size</i> makes a smaller copy that gives the same pictures. The .ckpt isn't changed." ] },
          ] },
        { id: "train", go: "train", icon: "🎓", title: "Train", summary: "Teach the app your own character, object or style; merge LoRAs",
          sections: [
            { h: "Three things you can make", items: [
              "At the top of the Train tab, <b>What do you want to make?</b>: <b>🎓 LoRA</b> (learns a lot: a face, an outfit, a style; added in Create under LoRAs), <b>🔤 Embedding</b> (learns one new word; a tiny file you use by typing its name) or <b>🧬 Merge LoRAs</b> (several LoRAs become one)." ] },
            { h: "🔤 Embeddings (textual inversion)", items: [
              "Pick an SD 1.5 model, a folder of <b>5–20 pictures</b> that share the thing you're teaching, a <b>name</b> (the word you'll type, e.g. <i>my_cat</i>) and a <b>starting word</b> (what it's close to: cat, woman, painting).",
              "<b>Subject</b> or <b>Style</b> changes the practice sentences it learns from. <b>Vectors</b>: 2 is standard; 4–8 hold more detail (good for styles). <b>Steps</b>: 1500 is standard; about 30–60 minutes on an 8 GB M1.",
              "Only the new word is learned; the model itself isn't changed. The finished embedding appears in the list above the negative prompt; type its name or click it. Pause and Stop work as for LoRAs (Stop keeps what it learned so far).",
              "Embeddings can only steer what the model already knows how to draw. For a new face or a very particular look, a LoRA learns more. SDXL embeddings need a 16 GB+ Mac and aren't available yet." ] },
            { h: "🧬 Merging LoRAs", items: [
              "Pick the model type, add two or more of its LoRAs, set how strongly each counts (1 = full, 0.5 = half, below 0 pushes away), name the result and press <b>Merge</b>. Seconds, on the CPU; the originals are never changed.",
              "<b>Exact</b>: the LoRAs are stacked, so the new LoRA does exactly what using them together does (the file is about their sizes added together). <b>Smaller</b>: a close match squeezed to the rank you pick (16–128), much smaller.",
              "Use the merged LoRA at weight 1 (your weights are inside it). It keeps the trigger words of the LoRAs that went in.",
              "Only LoRAs for the same model type can be merged. DiffusionBee .tdict files need converting first (Tools → Convert DiffusionBee files); LyCORIS types (LoHa, LoKr, DoRA) and LoRAs saved in different layouts can't be merged, and the app says which." ] },
            { h: "Before you start", items: [
              "Collect <b>10–30 images</b> of one subject or style in a folder; varied angles and backgrounds work best.",
              "Pick a <b>trigger word</b> that isn't a real word (e.g. <i>pxlstyle</i>): you'll type it to use the LoRA.",
              "Choose <b>what you're teaching</b> (character / person, object, art style, pose / concept): the app then shows how many pictures to collect and what to vary.",
              "<b>Picture size</b> depends on the base model: <b>512 × 512</b> for SD 1.5, <b>1024 × 1024</b> for SDXL and ERNIE (portrait and landscape shapes work too). Each picture should be at least that size on its short side; the app checks your folder and tells you which ones are too small." ] },
            { h: "Square framing: crop or keep whole", items: [
              "Training works on squares. Pictures that aren't square get an amber note in <b>Square framing</b> (below the training settings), and each tile shows exactly the square the training will see.",
              "Under each picture choose <b>Crop · centre</b> (the default), <b>top</b>, <b>bottom</b>, <b>left</b> or <b>right</b> to pick the part that's kept, or <b>Keep whole</b> to fit the whole picture in (the sides are filled with a soft blur of it). <b>All pictures:</b> sets every picture at once; <b>keep the non-square ones whole</b> changes only those.",
              "<b>✂ Crop by hand…</b> (in a picture's list, or click the picture): the picture opens large with a square on it. Drag the square to move it, drag its amber corner (or use the <b>Size</b> slider) to make it bigger or smaller, then <b>Use this square</b>. The size readout warns when the square is smaller than 512 px (it's scaled up). <b>Centre, biggest</b> starts over.",
              "<b>Leave a picture out</b>: the <b>✕</b> in a picture's corner (or <i>⊘ Leave out</i> in its list) leaves it out of training and of <b>Save copies</b>; it stays in the folder, dimmed and marked LEFT OUT. <b>↺</b> puts it back. The count at the top says how many the training uses.",
              "Your choices are saved with the folder (a small hidden file next to the pictures) and used by LoRA and embedding training. Your picture files aren't changed.",
              "<b>Save copies</b> writes square 512, 768 or 1024 px copies (with their captions) into a new folder next to yours, to keep or use elsewhere; <b>Use that folder</b> trains on them." ] },
            { h: "Captions", items: [
              "Let the app caption the images (tags for anime/illustration, sentences for photos), then edit them.",
              "Describe what is <i>not</i> the thing you're teaching; the trigger word stands for what is." ] },
            { h: "Training", items: [
              "<b>Pick your pictures</b> with <i>Choose a folder of images…</i> (2 · Training images); <b>✕ Clear</b> forgets them so you can start again with another folder. Nothing in the folder is changed or deleted (if you edited captions without saving, it asks first).",
              "<b>Quick / Standard / Thorough</b> set steps, learning rate and rank. Standard is about 100 steps per image.",
              "Watch the progress, time left and the loss chart (it should drift down and level off). <b>Stop</b> keeps the progress so far.",
              "SD 1.5 suits 8 GB Macs. <b>SDXL and Pony / Illustrious / NoobAI</b> can be trained too: pick the built-in SDXL (it trains on the full-precision SDXL Base, a 7 GB download if it isn't on your Mac yet) or any full SDXL checkpoint you've added. Under 16 GB it works but is very slow (often many hours); train at 768 to lighten it. LoRAs trained on a Pony model are labelled Pony automatically. ERNIE can be trained on the <b>⚡ fast-engine ERNIE</b> (it learns a LoRA on top of the 4-bit model, with low-memory settings): it suits 16 GB+ Macs and is experimental on 8 GB, where it may run out of memory or take many hours. The full-precision ERNIE needs 32 GB+.",
              "FLUX.1 LoRAs can't be trained in the app yet: the fast engine's training only supports other models.",
              "Save as <b>Kohya .safetensors</b> (works in most apps) or Diffusers format. The LoRA appears in your list, ready to use." ] },
          ] },
        { id: "library", go: "library", icon: "🗂", title: "Library", summary: "Everything you've made, searchable",
          sections: [
            { h: "Finding images", items: [
              "<b>Right-click a picture</b> for a quick menu: Open details, ★ Star, Send to Image to image, Send to Tools, Show in Finder and Delete (moves it to the Trash after asking; Esc closes the menu).",
              "<b>☑ Select</b> (middle of the toolbar), tick 2–6 pictures and press <b>⇆ Compare</b> in the toolbar (next to ✕ Deselect all; it stays in view while you scroll down). In a picture's details, <b>‹ ›</b> or the ← → keys go to the previous / next picture, and <b>✕ Close</b> / <b>Delete</b> sit beside the details (Esc closes).",
              "<b>Selecting many at once</b>: in ☑ Select, drag across the pictures (or draw a box starting between them); the list scrolls when you rest the pointer at the edge. Start the drag on a picture that's already selected to unselect instead. <b>✕ Deselect all</b> (centred under ☑ Selecting · Done, and in the selection bar) unselects everything.",
              "<b>Zoom in Compare</b>: scroll the mouse wheel or pinch over any picture (or use － ＋, Fit and 1:1); drag to look around, double-click to zoom in or back out. Every picture zooms and moves together, so you can compare the same spot. <i>details</i> under a picture opens it.",
              "Search by prompt, filter by model, version type (originals, enhanced, face-restored…) or ★ starred.",
              "<b>Hide People</b> hides pictures of people: ones whose prompt describes a person, ones where the face finder sees a face, and ones made with models or LoRAs you tagged ● People. Pictures made before this version: <b>check now</b> next to the picture count looks for faces in them once (on this Mac). <b>Hide NSFW</b> hides NSFW pictures.",
              "<b>Search by seed</b>: type a seed in the amber <b>seed</b> box to see only pictures made with it, or type a number in the search box to find that seed and prompts containing it. In a picture's details, <b>find all with this seed</b> does the same.",
              "<b>Filter by LoRA</b> shows images made with particular LoRAs; clicking a LoRA in an image's details does the same.",
              "<b>Hide NSFW</b> covers pictures flagged NSFW; <b>Hide People</b> hides pictures made with a model or LoRA you tagged ● People (the same switch as in Create and Settings)." ] },
            { h: "Working with an image", items: [
              "Click an image to see its prompt and settings; <b>↺ Reuse prompt</b> puts just its prompt in Create (your model and settings stay), <b>Use these settings</b> loads everything. Hovering a picture shows <b>↺ Prompt</b> too.",
              "Restore faces, enhance & upscale, or <b>🖼 Send to Image to image</b> straight from there.",
              "Delete one image or many (<b>Select</b>); files are removed from your Mac or moved to the Trash (Settings).",
              "<b>The details window</b> keeps the same size as you step through pictures with <b>‹ ›</b> or ← →. <b>✕ Close</b> is at the top of the buttons beside the details and <b>🗑 Delete</b> at the bottom, well apart (Delete always asks first). Esc closes." ] },
            { h: "🗂 Batch edit & export (like PhotoMill)", items: [
              "<b>Select</b> pictures (or <b>Select all</b>), then <b>🗂 Batch edit &amp; export…</b>. Copies are saved to a folder you pick (default <i>Pictures → Protoart Exports</i>); the pictures in the Library are never changed, and nothing is ever overwritten (a -2, -3… is added).",
              "<b>Adjust</b>: apply one of the Looks or your own Adjust presets to every picture. <b>Size</b>: keep, long side, short side, fit in a box (for example 1080 × 1350) or percent; <b>Never enlarge</b> keeps small pictures as they are.",
              "<b>Format</b>: same as original, JPEG, PNG, WebP or TIFF; <b>Resolution</b>: 72 / 150 / 300 dpi. Colours are converted to sRGB so they look the same everywhere.",
              "<b>Picture details</b>: <b>Remove (private)</b> (the default) leaves out the prompt, settings and any location; <b>Keep prompt &amp; settings</b> puts them inside (file paths on your Mac are never included).",
              "<b>File names</b>: a pattern with {name} {n} {date} {time} {model} {seed} {width} {height} {preset}; the example under it shows what you'll get.",
              "<b>Presets</b>: 🗜 Compress (Press), Web · 2048 px JPEG, Instagram 1080 and Print · TIFF 300 dpi are ready to use; <b>＋ Save as preset</b> keeps your own combination. The bar shows progress; <b>■ Stop</b> stops after the current picture. At the end it says how much smaller the files got." ] },
            { h: "🗜 Compress (Press)", items: [
              "Makes copies that are much smaller (usually 40–70%) but still look the same, the way <b>ProtoArt Press</b> does: every result is compared with the original (SSIM, including its worst 64×64 area and any banding) before it's used.",
              "<b>JPEG</b>: finds the lowest quality that stays within this picture's own damage budget (Gentle ≈ quality 90, Balanced ≈ 85, Strong ≈ 80 on typical photos). <b>PNG</b>: lossless repacking always; with Balanced or Strong a 256-colour version when it's invisible; what's written is read back and checked pixel for pixel.",
              "One picture: <b>🗜 Compress / export…</b> in its details. Several: <b>Select</b> them and press <b>🗜 Compress</b>. It uses the Compress setting with the same format and size; change anything else in the panel before pressing Export." ] },
            { h: "Hiding NSFW pictures", items: [
              "<b>🛡 Safe-for-work mode</b> (the 🛡 SFW switch at the top of Settings, asks first): prompts that ask for adult content are refused (checked in the engine), NSFW LoRAs, models and embeddings can't be used and disappear from the lists, NSFW pictures are hidden everywhere and can't be opened, and every new picture is checked on this Mac. An SFW lamp shows next to Power.",
              "<b>Hide NSFW</b> (at the top of Create and Library, and in Settings → Content filter) leaves NSFW pictures out of the Library and covers them in Create, Image to image and Tools. Click a covered picture to look at it. NSFW models and LoRAs are hidden from the lists too.",
              "A picture counts as NSFW when its prompt asks for it (the negative prompt doesn't count), or when the on-device <b>picture checker</b> says so (Settings → Content filter; ~340 MB download, once).",
              "Your own mark always wins: in a picture's details use <b>mark NSFW</b> / <b>mark safe</b> / <b>automatic</b>, or <b>Select</b> several and use <b>Mark NSFW</b> / <b>Mark safe</b>.",
              "In an image's details, the <b>Content</b> line says why a picture counts as NSFW.",
              "In Create's model and LoRA lists, a red <b>⚑</b> after a name means NSFW. Point at a name and click its ⚑ to mark it (or unmark it).",
              "<b>● People</b>: an amber dot after a name means you tagged it as made of real people (for example an SD 1.5 model or LoRA trained on photos of someone). Point at a name and click the dot to tag it. <b>Hide People</b> (top of Create) hides everything tagged that way." ] },
          ] },
        { id: "models", go: "models", icon: "📦", title: "Models", summary: "What's on this Mac, and what you can download",
          sections: [
            { h: "Using another app's model library (Stability Matrix, ComfyUI…)", items: [
              "When the app starts it looks in the usual places for <b>Stability Matrix</b> (its settings, ~/Library/Application Support/StabilityMatrix, a portable <i>Data</i> folder next to the app), <b>ComfyUI</b> (including the desktop app's base path), <b>Automatic1111 / Forge</b>, <b>InvokeAI</b> and <b>Fooocus</b>. If it finds one it asks <i>“Existing model library detected. Would you like to use it?”</i>, with how many checkpoints, LoRAs, embeddings, VAEs and ControlNets it holds.",
              "<b>Use it</b> adds its folders to the places this app looks. The files are <b>read in place</b>: nothing is copied, moved, renamed, changed or deleted, and the other app keeps working as before. Each file's type (SD 1.5, SDXL, Pony, Illustrious, FLUX…) is read from the file itself, so only models that work here are offered for the model you pick.",
              "Settings → Model folders → <b>Model libraries from other apps</b>: <b>↻ Look for libraries again</b> (also re-reads every watched folder, for files the app missed the first time), <b>Pick a library folder…</b> for one somewhere unusual (an external drive, or Stability Matrix's <i>Data</i> / <i>Models</i> folder), and <b>Stop using</b> to take one off the lists again.",
              "A plain folder of models (not another app's library) goes in <b>Watched model folders</b> instead: add as many as you like." ] },
            { h: "Organising your models and LoRAs", items: [
              "<b>Embeddings, VAEs &amp; ControlNets</b> folds up too (click its name), and <b>Your model files</b> (base models you added from this Mac) is at the bottom of the tab.",
              "<b>On this Mac</b>, <b>Available to download</b> and the other lists fold up (click their names), under the Your model files / Civitai / Hugging Face / Fix a type panels. Filters at the top include <b>Pony</b>, <b>Illustrious</b> and <b>NoobAI</b>.",
              "<b>+ Add from your Mac</b> says “✓ Model imported” when it's done; inpainting models are recognised and go to Image to image → Inpaint / Expand.",
              "<b>Trigger words</b> panel: search LoRAs, edit each one's words, or tick several and add / remove a word on all of them. Your words are added to the prompt when you add the LoRA. <b>↺ Back to the file's words</b> forgets your edits.",
              "New files not showing? Press <b>↻ Refresh</b> (top of the Models tab, or in Create's model list). LoRAs anywhere in a watched model folder are found too." ] },
            { h: "Finding a model", items: [
              "<b>On this Mac</b>: models that are downloaded, your own files, DiffusionBee models and ones you added. Only these appear in Create's model list, so it stays short.",
              "<b>Available to download</b>: models this Mac can run. <b>⤓ Download</b> fetches it now (progress and time left show on the card and in the queue; <b>Stop</b> stops it and keeps what's already downloaded). <b>Use</b> picks it right away; it then downloads the first time you press Generate.",
              "<b>Needs more memory</b>: models that are too big for this Mac. You can still download them (for example to copy to a bigger Mac, or to try with Settings → Load anyway; it may be very slow).",
              "<b>Pony, Illustrious and NoobAI</b> are SDXL models trained further for anime and illustration. The app recognises them from their name or the file's own details, labels them, and sorts LoRAs made for them first (LoRAs for another kind are marked “may not match”). Create then shows <b>Add … quality tags</b> (on by default), which puts their usual quality tags in front of your prompt, e.g. <i>score_9, score_8_up, score_7_up</i> for Pony. They use CLIP skip 2 like other SDXL models. Click the tags next to the switch (<b>⤴ put in prompt</b>) to write them into your prompt and negative prompt so you can see and edit them. On 8 GB Macs the first load converts the file to 8-bit and saves that copy (a few minutes, once; ~2.6 GB in your data folder).",
              "<b>FLUX.1 Schnell</b> (Black Forest Labs, Apache-2.0 licence, free to use) makes a picture in 4 steps on the fast engine. Pick the <b>3-bit</b> build on 8 GB Macs (7.5 GB download); the 4-bit (9.6 GB) needs 16 GB and the 8-bit (18 GB) 32 GB for the best quality. FLUX prefers sentences, ignores the negative prompt, and takes FLUX LoRAs.",
              "<b>FLUX.2 Klein 4B</b> (Black Forest Labs, January 2026, Apache-2.0; the 4-bit build's page also carries a non-commercial tag, so check it before selling pictures) is the newest FLUX and the best start on an 8 GB Mac: a 4.6 GB download (a 4-bit build for the fast engine), 4 steps a picture, good at long prompts and at writing words in pictures. Write sentences; the negative prompt is ignored and the CFG stays at 1. LoRAs must be made for FLUX.2 Klein 4B (Civitai base model “Flux.2 Klein”); FLUX.1 LoRAs are listed under FLUX and won't load. On 8 GB the app sets the size to Medium (768) and frees the text part of the model while it draws, so changing the prompt reloads that part (a few seconds).",
              "<b>Keeping models on an external drive</b>: base model files and LoRAs you already have can stay anywhere (Settings → <i>Watch a models folder</i> and <i>Add LoRA folders</i>; nothing is copied). Models this app downloads go to the <b>Downloaded models folder</b> in Settings, which can be on an external drive too (it can move the ones you already have).",
              "<b>Make default</b>: Create starts with that model every time the app opens (and gets it ready in the background). <b>🗑 Delete</b> removes a downloaded model's files from this Mac (click twice; files another model still needs are kept, and it downloads again if you use it later). Your own files get <b>Move to Trash</b> instead, so you can put them back.",
              "<b>↻ Refresh</b> (here, under Create's model list, and in Settings) looks again for models and LoRAs: files you've just put in your folders, or a download that finished.",
              "<b>+ Add from Civitai…</b>: paste a civitai.com model link and press <b>Check</b>, then <b>⤓ Download</b>. LoRAs go into the folder set to their type, SD 1.5 / SDXL / Pony checkpoints into your checkpoints folder, with their trigger words. Only .safetensors files are downloaded (and .pt embeddings).",
              "<b>Sign in first (once):</b> Civitai now lets only signed-in accounts download most files. Make a free account on civitai.com, open <b>Account settings → API Keys → Add API key</b>, copy the key and paste it in the box in the Civitai card (<b>Save key only</b>, or <b>Save &amp; download</b>). The box shows until a key is saved; the key stays on this Mac and is only sent to civitai.com. Errors you may see and what they mean are in Troubleshooting → Civitai downloads.",
              "Filter by type (ERNIE, SDXL, SD 1.5) or search. <b>+ Add from your Mac</b> and <b>+ Add from Hugging Face</b> bring in models from elsewhere. In Create, the link under the model list opens this tab." ] },
            { h: "Managing your models", items: [
              "<b>Embeddings, VAEs &amp; ControlNets</b> (bottom of the Models tab): the smaller files that work with a base model, with SD 1.5 / SDXL filters, search and Open folder.",
              "<b>Remove</b> buttons for your own models are locked 🔒: click the lock first, then Remove (it always asks). Two of your files with the same name show the folder each is in, and “a copy” when they're the same size.",
              "<b>LoRA list</b> (Create): sort by name A → Z / Z → A or by size, and show only one kind (.safetensors, DiffusionBee, Pony…). ★ favourites always come first.",
              "<b>Bigger Macs</b>: FLUX.1 Dev (32 GB or more; personal use only, needs a free Hugging Face account that accepted its licence). <b>8 GB Macs</b>: FLUX.2 Klein 4B, FLUX.1 Schnell 3-bit, ERNIE-Image Turbo 4-bit and SD 1.5 work best.",
              "At the top of this tab: <b>Your model files</b> (files you added from this Mac, with Remove), <b>Add from Civitai</b>, <b>Add from Hugging Face</b> (with the models you added) and <b>Fix a type</b>. They stay folded until you click one; the + Add buttons open the right one.",
              "<b>★ Favorites</b>: click the ★ on a model (here, or when you point at it in Create's model list) to make it a favourite. Favourites are listed first in Create, and <b>★ Favorites</b> (in Create and in this tab's toolbar) shows only them. They're remembered on this Mac; the model files are never changed or copied.",
              "Markers, in the same place on every row: <b>★</b> favourite, <b>●</b> People (made of real people), <b>⚑</b> NSFW (red = marked). Lit = set; point at a row to see the unlit ones, and click to switch them. <b>Hide NSFW</b> / <b>Hide People</b> hide marked models and LoRAs everywhere, including 🎲 Random LoRAs.",
              "Folders (watched model folders, the downloaded-models folder, LoRA and embedding folders) are set in <b>Settings</b>." ] },
          ] },
        { id: "disclaimer", go: "settings", icon: "⚖️", title: "Disclaimer & licences", summary: "What to keep in mind about AI pictures and the models you use",
          goLabel: "Open Settings",
          sections: [
            { h: "Pictures made by AI", items: [
              "Pictures are made by AI models on this Mac. They can be inaccurate, distorted or unexpected, and may contain things you didn't ask for, including content that's offensive or not suitable for work. Check a picture before you share or use it.",
              "Text in pictures, hands, faces and small details are often wrong. Don't rely on a picture as a factual record of a real person, place or event.",
              "<b>Hide NSFW</b> and <b>Hide People</b> help keep unwanted content out of view, but they're filters, not guarantees: names, tags and the on-device picture checker can miss things." ] },
            { h: "Models and licences", items: [
              "Each model and LoRA has its own licence, set by whoever made it (for example Apache-2.0 for FLUX.1 Schnell and ERNIE-Image, CreativeML OpenRAIL for Stable Diffusion, non-commercial for some others). Some don't allow commercial use or certain kinds of content. Check the licence on the model's page (Hugging Face or Civitai) before using pictures commercially.",
              "Models and LoRAs you download from Civitai or Hugging Face are made by other people; Protoart Studio doesn't review them.",
              "LoRAs that copy a real person's likeness, or an artist's style, may be restricted by law or by the person's rights where you live. Don't make pictures that harm, mislead or harass real people." ] },
            { h: "Your responsibility", items: [
              "You're responsible for what you make with the app and how you use it. Follow the law where you live and the terms of the models you use.",
              "Protoart Studio is a test version, provided as is, without warranty. Keep copies of pictures that matter to you.",
              "Privacy: everything runs on this Mac. Pictures, prompts and settings aren't sent anywhere; the only network use is downloading models (and the Civitai / Hugging Face sign-in keys you save, which go only to those sites)." ] },
          ] },
        { id: "settings", go: "settings", icon: "⚙️", title: "Settings", summary: "Folders, privacy and options",
          sections: [
            { h: "Look: light, dark and themes", items: [
              "At the top of Settings: <b>Light / Dark / Match macOS</b> sets how bright the faceplate is; the colour buttons next to it set its theme: <b>Amber</b>, <b>Cream</b>, <b>Slate</b> or <b>Ember</b>. Both are remembered.",
              "Whatever is switched on lights up in the theme's lamp colour (amber, incandescent yellow, ivory or ember); what's off stays dark." ] },
            { h: "Finding your way around", items: [
              "<b>TIPS</b> (the lever at the top): flip it up to see what a button does when you point at it, down to hide the tips. <b>🔔 Sounds</b> (Settings): a chime when pictures or training are finished. <b>How to → Community &amp; support</b>: Discord, contact, report a problem, updates, and the model licences.",
              "Settings starts tidy: only <b>This Mac</b> and <b>Model folders</b> are open, every other section is folded down to its heading. Click a heading to open it, click again to fold it; the app remembers. <b>fold all / open all</b> sit at the top.",
              "Links elsewhere in the app that point to a setting (for example <i>Add from Civitai</i> or <i>Fix a type</i>) open the right section for you." ] },
            { h: "Models & LoRAs", items: [
              "<b>From your Mac</b>: add your own checkpoint files or model folders, or watch a folder (offline).",
              "<b>From Hugging Face</b>: paste a model id or link to add new models as they're released.",
              "<b>Gated Hugging Face models</b> (marked “gated”: their owners want you to accept a licence) need a free huggingface.co account: sign in, open the model's page and accept the licence, then make a <b>Read</b> access token (huggingface.co → Settings → Access Tokens) and paste it under <b>Hugging Face access token</b>. A box with these steps opens by itself when a model needs it. The token stays on this Mac, is sent only to huggingface.co, and a token other apps saved is never used.",
              "<b>From Civitai</b>: paste the link of a model page on civitai.com and press <b>Check</b>. LoRAs (SD 1.5, SDXL, Pony, Illustrious, FLUX…) and SD 1.5 / SDXL / Pony checkpoints download straight into the right folder for their type, with their trigger words. Embeddings (“Textual Inversion”) go to Settings → Embeddings.",
              "<b>Sign in first (once):</b> Civitai now lets only signed-in accounts download most files. Make a free account on civitai.com, open <b>Account settings → API Keys → Add API key</b>, copy the key and paste it in the box in the Civitai card (<b>Save key only</b>, or <b>Save &amp; download</b>). The box shows until a key is saved; the key stays on this Mac and is only sent to civitai.com. Errors you may see and what they mean are in Troubleshooting → Civitai downloads.",
              "<b>LoRA folders</b>: use LoRAs where they are, including external drives. <b>DiffusionBee</b>: use your old library in place.",
              "Each LoRA folder can be set to a model type (<b>Auto</b> reads each file). A LoRA listed under the wrong type? <b>Fix a LoRA's type</b> (Settings → LoRA folders, or the link at the bottom of Create's LoRA list): search for it and pick SD 1.5 / SDXL / FLUX / ERNIE. With <b>Move the file into that type's folder</b> on, the file and its trigger-word / preview files move into the LoRA folder you set to that type.",
              "<b>Fix a type</b> also works for your own <b>base model files</b>, but only as <b>SD 1.5</b> or <b>SDXL</b> (Pony, Illustrious and NoobAI are SDXL): a single .safetensors / .ckpt / DiffusionBee file can only be loaded as one of those. <b>FLUX</b> and <b>ERNIE</b> come as whole model folders (from the Models tab, Hugging Face, or a folder you add), and their type is read from the folder, so there's nothing to fix. <b>Show unknown type</b> lists everything the app couldn't identify.",
              "<b>Moved a folder?</b>: after moving your models / LoRAs folder (e.g. to a new drive), click <b>Find it…</b> and pick its new place; everything that used it follows." ] },
            { h: "Speed: a model ready in memory", items: [
              "Loading a model is the slow part of the first picture (about 5–15 s for SD 1.5 on an 8 GB Mac, longer for SDXL, ERNIE and FLUX). After that it stays loaded, so the next pictures start at once.",
              "<b>When the app opens, load…</b> (the model you used last, by default) and <b>Load a model as soon as I pick it</b> do that loading in the background while you write the prompt. Create shows <b>getting ready…</b> and then <b>● loaded, ready</b> next to the model's name.",
              "It only uses idle time (a picture you ask for always goes first) and never downloads anything. It speeds up the first picture, not each step. Only one model fits in memory at a time." ] },
            { h: "Content filter (NSFW)", items: [
              "Turn on <b>Hide NSFW</b> to hide NSFW pictures, models and LoRAs everywhere (the same switch is at the top of Create and Library).",
              "<b>Check each picture with the on-device picture checker</b>: downloads a small checker once (~340 MB, Apache 2.0), then judges every new picture on this Mac in a fraction of a second, and can check your existing Library.",
              "Models and LoRAs are marked automatically from their names (and, for LoRAs, the tags they were trained on). A ⚑ after a name in Create's lists marks NSFW; point at a name and click its ⚑ to change it; your marks are listed in Settings, with <b>Automatic</b> to undo." ] },
            { h: "Images & options", items: [
              "Choose where images are saved, PNG or JPEG, and whether deleting removes files or moves them to the Trash.",
              "Live preview, low-memory mode (SDXL) and the backend log (useful when something goes wrong)." ] },
            { h: "Finding your way around Settings", items: [
              "Click a section's heading to fold it away (▾); the app remembers which ones you folded. Long lists (your marks, People tags) start folded.",
              "In a narrow window the sections stack in one column, so nothing is cut off.",
              "<b>Main folder</b> (top of Model folders): one folder of yours that holds everything. Its <i>Models</i>, <i>LoRAs</i>, <i>Embeddings</i>, <i>VAE</i> and <i>ControlNet</i> sub-folders are used (missing ones are made, empty; existing ones named the usual ways, like <i>Stable-diffusion</i> or <i>Lora</i>, are reused). Nothing is copied or moved. <b>↻ Refresh</b> reads every folder again; Stop using takes them off the lists.",
              "<b>More model folders</b>: <b>＋ Add folders…</b> as many times as you like (⌘-click to pick several at once); each is listed as Folder 1, 2… with Remove. <b>↻ Refresh</b> reads them again. Below, <b>Model libraries from other apps</b> uses a Stability Matrix / ComfyUI / A1111 / InvokeAI / Fooocus library in place." ] },
            { h: "Look", items: [
              "The Look switch is at the top of the window when Settings is open.",
              "<b>Dark</b> (black faceplate, the default), <b>Light</b> (brushed aluminium with orange trim) or <b>Match macOS</b> (follows your Mac's light / dark appearance)." ] },
            { h: "Reset", items: [
              "<b>Reset folder locations</b> or <b>Reset settings</b> puts things back to how they were when the app was new. Pictures, models, LoRAs and presets are never deleted." ] },
          ] },
        { id: "trouble", go: "settings", icon: "🛟", title: "Troubleshooting", summary: "Common problems and how to fix them",
          goLabel: "Open Settings (backend log, restart, reset)",
          sections: [
            { h: "New in 1.3.0: if something doesn't work", items: [
              "<b>Where did 🎞 Animation go?</b> It was in the 1.3.0 test builds and has been taken out: on an 8 GB Mac a short clip took hours. Protoart Studio is about pictures.",
              "<b>A model says “⚠ file not found”</b> (or “isn't where it was” when you press Generate): its file was moved or renamed, or the drive it's on isn't connected. Connect the drive, or right-click the model → <b>⟲ Reconnect…</b> and pick the file where it is now; your favourites and default model follow it. A whole folder that moved: Settings → your folders → right-click → Moved it…",
              "<b>“This needs a one-time download from Hugging Face … couldn't reach huggingface.co” although you're online</b> (OpenPose / Union ProMax OpenPose, Depth and some other ControlNets): fixed. The pose and depth finders' first download wasn't recognised as a download, so it was never tried. The message now also has a <b>⬇ Download it now</b> button that runs the same job again with the connection open.",
              "<b>🧩 Tiled upscale turned the picture into something else</b> (new faces, extra people, another style): fixed. It used Create's prompt, so every tile tried to draw that whole scene. It now uses a short detail prompt (add your own texture or style words in <i>Detail prompt</i>) and gentler defaults; pick a preset.",
              "<b>🗺 Regional prompts barely changed the picture</b>: each area now follows only its own prompt (before, the whole main prompt was added to every area and drowned the subjects). Describe each area fully and keep the main prompt to the setting.",
              "<b>Tools → Convert → Convert into</b>: <b>Copy</b> (the .safetensors copies as before), <b>LoRA</b> (turn a fine-tuned model into a small LoRA: pick the model, the base model it was made from, and a size; rank 32 is a good start, 64–128 keeps more of its look) or <b>Embedding</b> (embeddings are learned from pictures, so it opens Train). The LoRA appears in your LoRA list and works in other apps too.",
              "<b>Sort the models A → Z</b> (or Z → A, or biggest first): the Sort menu beside Create's model search, and beside the Image to image model list. ★ Favourites stay first.",
              "<b>Image to image</b>: every card (Picture, Sampler, LoRAs, Model chain, ControlNet, Prompt) folds down to its title with a click on it, and <i>collapse all · expand all</i> sits at the top. Each <b>Model chain</b> step has a search box for its LoRAs, and <b>FLUX.2 Klein 4B</b> can be a chain step: it edits the picture with the words you write for that step (“make it a watercolour”), with its own LoRA if you like.",
              "<b>🗺 Regions with FLUX.2 Klein 4B</b> too, and a <b>✨</b> button on each area to describe its subject in detail.",
              "<b>🗺 Regions with FLUX.1 and ERNIE</b>: now possible. These models run on the fast engine, which can't split one picture's prompt by area, so the app draws the whole picture with each area's place in words, then redraws each area from its own prompt and blends it back in.",
              "<b>💡 suggest from the picture</b> is now in every Image to image mode (Change it, Inpaint, Expand, Edit with words), not only FLUX.2 Expand: it looks at your picture and writes a prompt describing it (Danbooru tags for Anima, Pony, Illustrious and NoobAI models), ready to edit. ↩ undo puts your prompt back.",
              "<b>💡 suggest from the picture (Image to image) showed its prompt in Create</b>: fixed. Create and Image to image keep separate prompts; after a restart the app could mix up which one was in the box.",
              "<b>Anima is slow</b>: it's a 2-billion-parameter model; on an 8 GB Mac a 10-step <b>Anima Turbo</b> picture takes about 3–4 minutes at 768 × 768 and 6–7 minutes at 832 × 1216. Quit other big apps. The first picture also loads the model (about 20 s).",
              "<b>Anima pictures came out flat grey</b> (1.3.0 test builds): fixed. The Mac graphics chip worked out the Anima picture decoder (VAE) wrongly; the app now runs it a different way that gives the right picture.",
              "<b>Anima added “safe” to my prompt</b>: that's on purpose, so pictures stay all-ages unless you ask otherwise. Put your own rating tag in the prompt (safe, sensitive, nsfw or explicit) and the app leaves it alone.",
              "<b>Face swap says no face was found</b>: use a clear, front-facing, well-lit photo for the face, and a picture where the face is at least ~40 px across. <b>“The result looks explicit”</b>: face swap never saves explicit results.",
              "<b>Union ProMax is greyed out or slow</b>: it's for SDXL models (pick one at the top); on 8 GB keep the size at 768–1024 px.",
              "<b>Regional prompts look like one prompt</b>: lower <i>Keep one scene</i> (0.1) and make each area's prompt distinct; with very different areas, raise the steps a little. They can't be combined with an image prompt yet.",
              "<b>Model chain stopped at FLUX.2 Klein</b>: Klein changes pictures with ✨ Edit with words, so it isn't offered in the chain; FLUX.1 Schnell and ERNIE are.",
              "<b>Tiled upscale is slow</b>: each tile is one small picture's worth of work: 2× of a 768 px picture is about 12 tiles × 7 steps. Use 1.5× or lower <i>New detail</i> for quicker results.",
              "<b>📥 Settings found nothing</b>: the picture was saved without its settings (many websites remove them), or by an app that doesn't write them." ] },
            { h: "Things fixed in 1.2.2 (and what to do if you still see them)", items: [
              "<b>ControlNet gave a flat grey picture, noise or nothing</b>: fixed (the model kept its precision only in Create before). If a map still looks wrong, press <b>Preview the map</b> first: an empty map means the picture had no clear edges or no person for OpenPose.",
              "<b>Image to image, ControlNet or Inpaint took 20+ minutes on an 8 GB Mac</b>: fixed for the same reason. Still slow? Quit big apps (browsers with many tabs, Photoshop, the Claude app) while generating: on 8 GB everything else squeezes the model into the disk. SDXL on 8 GB is slow by nature (1–3 minutes a picture); a model on an external drive loads more slowly the first time.",
              "<b>“Couldn't reach huggingface.co” for Tile / OpenPose / Scribble while online</b>: fixed. If it says a file wasn't found on Hugging Face, the model's page moved: tell us which one.",
              "<b>The app opened with no models at all</b> (new install or new version, models on an external drive): macOS is asking whether Protoart Studio may read the drive, and the question can hide behind the app's window. Find it (it says “would like to access files on a removable volume”) and click <b>Allow</b>; the models appear straight away. The status line says so after a few seconds.",
              "<b>A folder on an external drive showed nothing until a restart</b>: macOS asks once before an app may read a removable drive. Click <b>Allow</b>, then <b>↻ Refresh</b> in Settings → Model folders. Missed it? System Settings → Privacy &amp; Security → Files and Folders → Protoart Studio → Removable Volumes. Settings now shows ⚠ next to a folder it can't read.",
              "<b>The engine stopped with “Failed to allocate private MTLBuffer” at 768 × 768</b>: the same full-precision problem as above (it asked for over 5 GB in one go); fixed. If it happens with an IP-Adapter image prompt on an 8 GB Mac, use 512–640.",
              "<b>An old .ckpt model won't load</b> (“Weights only load failed”): Tools → Convert → <b>Old .ckpt files…</b> → Convert, then pick the .safetensors copy in Create.",
              "<b>conv_in.weight … [320, 9, 3, 3]</b> with an inpainting model: fixed; it's used by Inpaint and Expand (not Create, which needs a normal model)." ] },
            { h: "Things fixed in 1.2.1 (and what to do if you still see them)", items: [
              "<b>Expand with FLUX.2 gave seams, doubled objects or smeared areas</b>: much improved (the result is lined up with your picture before blending, and kept whole if FLUX.2 redrew it). The rest usually comes from asking for a lot at once: expand one direction at a time (★ make it wider, then ★ make it taller) and up to about a third per side, try another seed, and describe what should be around the picture (💡 suggest from the picture).",
              "<b>“FLUX.2 redrew the picture a little, so its whole version was kept”</b>: its copy of your picture differed too much to paste your original back without seams. Try another seed or a smaller expansion to keep your original pixel-for-pixel.",
              "<b>Expand's “New size” showed another picture's size</b>: fixed; it reads the size of the picture you're expanding (presets like make it wider used the old size before, which gave odd results).",
              "<b>Black bars beside the result</b> in Image to image: fixed; the frame fits the picture.",
              "<b>Two copies of a module ran with the same settings</b> on the patchbay: click the copy (e.g. FACES 2) to give it its own settings.",
              "<b>A button or slider showed no tip</b>: fixed; every control now has one (point at it for a moment). Tips can be switched off with <b>Tips</b> at the top of the window.",
              "<b>The expanded picture appeared in Create</b>: fixed; Expand, Inpaint and ControlNet results now stay in Image to image (and the Library).",
              "<b>Expand with FLUX.2 showed Stable Diffusion in the model menu</b>: fixed; it shows FLUX.2 Klein, which is what fills the new area. Picking another model there switches Expand to <i>The model at the top</i>.",
              "<b>Can't find Your model files in the Models tab</b>: it's now the last panel at the bottom of the tab. <b>Embeddings, VAEs &amp; ControlNets</b> can be folded by clicking its name.",
              "<b>A 🏷 tag filter shows nothing</b>: the list only shows LoRAs that fit the model you picked; a tag on SDXL LoRAs shows nothing while an SD 1.5 model is picked. Tags are kept on this Mac (Settings → Reset settings clears them).",
              "<b>A big gap at the top of Create</b> (the Reset all lever on a line of its own): fixed; it shares one line with collapse all · expand all.",
              "<b>Where's Compare side by side?</b> It moved up into the toolbar: <b>⇆ Compare</b>, right of ✕ Deselect all, so it's always in view however far you scroll. It lights up once 2 or more pictures are selected.",
              "<b>Reset all stayed flipped (“Reset · undo”)</b>: fixed; the lever is now spring-loaded and returns to rest right after resetting. There's no undo, so save a preset first if you want your settings back later.",
              "<b>Dragging to select pictures felt sticky</b> (Library → Select): fixed; it now follows the pointer smoothly, keeps scrolling while you rest at the top or bottom edge, and stops as soon as you let go (even outside the window). Start the drag on a picture or in the gap between pictures, not on its ★ or other buttons.",
              "<b>No way to unselect</b>: press <b>✕ Deselect all</b> under ☑ Selecting · Done (or in the selection bar). <b>☑ Selecting · Done</b> leaves Select mode.",
              "<b>Couldn't zoom in Compare side by side</b>: scroll the mouse wheel or pinch over a picture, or use － ＋ / Fit / 1:1 at the top; drag to look around. Every picture follows, so the same spot is compared. Clicking a picture no longer closes Compare: use <i>details</i> under it.",
              "<b>The picture details window changed size between pictures</b>: fixed; it keeps one size and the picture fits inside. <b>Close and Delete were too close</b>: Delete is now at the bottom of that column, well away from Close, and still asks before deleting.",
              "<b>The patchbay's cables are gone</b>: from 1.2.1 it starts empty, so you only plug in what you want. Press <b>example</b> for Picture → Enhance → Faces → Result, or drag OUT to IN. Cables you plug in are remembered. Modules can be moved by their names; <b>↺ reset</b> puts them back.",
              "<b>“BEE” in the LoRA list</b>: that LoRA is from your DiffusionBee folder (used where it is). The type and size now sit under the name, so the rows line up.",
              "<b>Train still shows an old folder of pictures</b>: press <b>✕ Clear</b> next to <i>open</i> (2 · Training images). The folder and its files aren't touched.",
              "<b>Expand with FLUX.2: “needs a one-time download of the outpaint LoRA”</b>: the 76 MB LoRA (and FLUX.2 Klein, if you haven't used Edit with words yet) download the first time. Connect to the internet and turn Strict offline off (Settings → Privacy) once; afterwards it works offline. Or pick <i>The model at the top</i>.",
              "<b>Expand with FLUX.2 left a green edge or a seam</b>: expand less at a time (up to about half the picture per side), try another seed, or add to the prompt what should be around the picture. Very thin borders (a few pixels) don't work well; 64 px or more is best.",
              "<b>The ‹ › arrows in a picture's details closed it</b>: the arrow moved slightly when pressed, so the click landed on the dark background, which closes the window. Fixed: the arrows stay put. The ← → keys also work.",
              "<b>My Stability Matrix / ComfyUI models weren't found</b>: Settings → Model folders → <b>↻ Look for libraries again</b>. If it's on an external drive, connect it first; if it's somewhere unusual, use <b>Pick a library folder…</b> and choose the library's main folder (for Stability Matrix, the <i>Data</i> folder that holds <i>Models</i>). A plain folder of models goes in Watched model folders instead.",
              "<b>Some models from the other app don't show</b>: a few formats other apps use aren't supported here yet (for example GGUF files, and diffusion-model-only files that need a separate text encoder), so they're left out of the lists. The other app can still use them, and nothing in the library is ever changed.",
              "<b>“⚠ not found (is the drive connected?)”</b> next to a library: its folder isn't there right now. Connect the drive and press ↻ Look for libraries again.",
              "<b>“This needs a one-time download from Hugging Face, and your Mac couldn't reach huggingface.co”</b>: a ControlNet, the pose finder or another helper isn't on your Mac yet. Check the internet connection and that Strict offline is off (Settings → Privacy), then try again; afterwards it works offline. DiffusionBee's ControlNets (Depth, Tile, Line art, Scribble, OpenPose) need no download.",
              "<b>A LoRA “was made for FLUX.1, so it can't be used with FLUX.2”</b>: LoRAs for another model type are now left out of the picture with a note. If a LoRA is listed under the wrong type, fix it in Models → Fix a type.",
              "<b>A LoRA or model you just added doesn't show</b>: press ↻ Refresh (Models tab, or Create's model list). LoRAs in watched model folders are found too; a LoRA folder can be added in Settings → LoRA folders.",
              "<b>“Command buffer execution failed: Insufficient Memory” while training</b>: the Mac ran out of graphics memory. Quit other apps and train again; embeddings train at 448 px on Macs under 16 GB, and progress is saved if memory runs out.",
              "<b>An inpainting model makes an error in Create</b>: inpainting models only fill in painted areas. They're used by Image to image → Inpaint and Expand." ] },
            { h: "“The engine stopped”", items: [
              "<b>Why:</b> the engine is the part of the app that makes pictures. If macOS stops it, it's usually because the Mac ran out of memory (big sizes, Hi-res fix and Face detailer together, or other heavy apps open).",
              "<b>What happens:</b> the app restarts the engine by itself and adds waiting jobs back; only the job that was running needs to be made again. If it keeps happening it stops restarting and asks you to press <b>Restart engine</b> (Tools or Settings).",
              "<b>Fix:</b> close other big apps, use a smaller size, or turn off Hi-res fix / Face detailer for that picture. Settings → Backend log shows the details." ] },
            { h: "Downloads fail: “couldn't connect to huggingface.co” or “Bad file descriptor”", items: [
              "<b>Why:</b> the first time you use a model or tool it downloads once. Something blocked the connection: no internet, a VPN, or a <b>firewall app</b> (Little Snitch, LuLu, Radio Silence, antivirus) that doesn't know Protoart Studio yet.",
              "<b>Fix:</b> the app now switches to macOS's own downloader (curl) automatically when its Python is blocked, so just try again. If it still fails, open your firewall app and <b>allow “python3.12” (Protoart Studio)</b>; in Little Snitch, Silent Mode blocks without asking.",
              "Also check <b>Settings → Privacy → Strict offline</b> is off (it blocks every download on purpose), and that the Mac is online.",
              "After the first download everything works offline." ] },
            { h: "macOS won't open the app the first time", items: [
              "<b>“can't be opened because Apple cannot check it”</b>: this test version isn't signed with an Apple developer certificate. On macOS 15 or newer open <b>System Settings → Privacy &amp; Security</b>, scroll down and click <b>Open Anyway</b>. On older macOS, right-click the app → <b>Open</b> → <b>Open</b>. Only needed once.",
              "<b>“is damaged and can't be opened”</b>: the download was changed or broken on the way. Don't force it open: delete it, download it again from the official release page, and compare its checksum (<code>shasum -a 256</code> on the DMG) with the one published there.",
              "Needs a Mac with <b>Apple Silicon (M1 or newer)</b>." ] },
            { h: "“The Python backend stopped” / “Failed to allocate … MTLBuffer” / the Mac freezes", items: [
              "<b>Why:</b> the picture needed more memory than the Mac has (common on 8 GB Macs with big sizes, SDXL, the Enhancer, or several images at once).",
              "<b>Fix:</b> click <b>Restart backend</b>, then use a smaller size (<b>Small</b>/<b>Medium</b>), fewer images per job, and quit other heavy apps. For SDXL on 8 GB, turn on <b>Low-memory mode</b> in Settings. Image to image is capped at a safe size automatically.",
              "Copy the details with <b>Settings → Backend log → Copy all</b> if you want to report it." ] },
            { h: "Nothing is happening / the queue doesn't move", items: [
              "Look for <b>❚❚ paused</b> at the top of the queue: press <b>▶ Resume</b>. While paused, the picture being made waits and no new job starts (Stop all also ends a pause).",
              "Paused pictures keep their memory. On an 8 GB Mac, if other apps feel slow while paused, <b>■ Stop</b> instead frees it." ] },
            { h: "Generating is very slow", items: [
              "The first picture after opening the app is slower (the model loads into memory); the next ones are faster.",
              "<b>“Your Mac is short of memory”</b> above the pictures: other apps (a browser with many tabs, Discord, other image apps) are holding the memory, so the Mac moves the model to the disk and every step waits for it. Quitting them makes SDXL several times faster (on an 8 GB Mac, about 30–100 s a step with browsers open, a handful without). The note's ✕ hides it.",
              "On 8 GB Macs: <b>SD 1.5 at 512 px</b> is quickest, <b>⚡ Fast mode</b> cuts steps to 6, ERNIE runs best on the fast engine at <b>Medium (768)</b>. Quit other apps so macOS doesn't swap.",
              "Hi-res fix, the face detailer and the image prompt each add time; turn them off for quick drafts. Turning off the live preview (Settings) helps a little." ] },
            { h: "🎚 Adjust: “No handler registered” or the preview stays black", items: [
              "<b>“No handler registered for 'image:read'”</b>: the app was updated while it was open. Adjust now works anyway (it reads the picture through the engine), but quit Protoart Studio fully (⌘Q) and reopen it for the quicker way.",
              "<b>Black preview or “WebGL 2” error</b>: Adjust uses the Mac's graphics chip. Close it and press <b>🎚 Adjust</b> again; if it keeps happening, quit and reopen the app.",
              "<b>“bigger than this Mac's graphics can filter”</b>: very large pictures (over 8,000–16,000 px) can't be saved from Adjust; make a smaller copy first." ] },
            { h: "An embedding doesn't show up, or does nothing", items: [
              "<b>Not listed above the negative prompt:</b> embeddings show only with models of their own type. An SDXL embedding (it holds “clip_l” and “clip_g” parts) needs an SDXL / Pony / Illustrious model; an SD 1.5 one needs an SD 1.5 model. Settings → Embeddings lists every one found and its type.",
              "<b>Not in Settings → Embeddings either:</b> put the file in Protoart's embeddings folder (<i>Add embedding files…</i>) or add its folder, then <i>Rescan</i>. .safetensors, .pt and .bin files work; SD 2 embeddings can't be used; FLUX and ERNIE don't use embeddings.",
              "<b>No effect:</b> the name in the prompt must match the file name (it isn't case-sensitive). The status line says “Loading embedding …” the first time it's used with a model." ] },
            { h: "The result is a flat grey image", items: [
              "<b>Why:</b> the maths overflowed, usually with a ControlNet set very strong or a model/LoRA mismatch.",
              "<b>Fix:</b> lower the ControlNet strength (try 0.6–1.0), try another ControlNet type, remove LoRAs one at a time, or use the model's recommended steps and CFG. A broken DiffusionBee ControlNet file is swapped for the official one automatically." ] },
            { h: "LoRA problems", items: [
              "<b>“LoRA … is for SDXL, but the loaded model is SD 1.5”</b> (or the other way round): LoRAs only work with the model type they were made for. Pick a matching model; the LoRA list only shows ones that fit.",
              "<b>A LoRA doesn't show for my model</b>: the list only shows LoRAs of the chosen model's type. Check, in order: (1) its folder is added in <b>Settings → LoRA folders</b> (adding model <i>files</i> doesn't add the folder they're in for LoRAs); (2) its type: <b>Models tab → Fix a type</b> (tick <i>Show unknown type</i> to see the ones the app couldn't identify; base model files can be fixed there too, as SD 1.5 or SDXL); (3) <b>Hide NSFW</b> / <b>Hide People</b> aren't hiding it; (4) press <b>Refresh</b> under LoRAs.",
              "<b>DiffusionBee LoRAs (.tdict)</b> are SD 1.5. They're read from DiffusionBee's own folder and from any LoRA folder you add, so after deleting DiffusionBee's copies, add the folder where you keep yours.",
              "<b>LoRAs of every type mixed in one folder</b>: <b>Settings → LoRA folders → Sort into type folders…</b> moves them into SD 1.5 / SDXL / FLUX / ERNIE sub-folders (it shows the plan first and reuses folders you already have). Or set a whole folder's type with the buttons under it.",
              "<b>“Couldn't load LoRA”</b>: the file may be for another model family or a format the app can't read (e.g. LyCORIS variants). Try it with another model type, or re-download it.",
              "<b>Trigger words not found</b>: add your own with <i>add / edit</i> under the LoRA; they're saved for next time." ] },
            { h: "A folder or drive can't be found", items: [
              "<b>Why:</b> you moved or renamed your models / LoRAs folder, or the drive it's on isn't plugged in. The app says so at startup.",
              "<b>Fix:</b> plug the drive in, or open <b>Settings → Moved a folder?</b>, click <b>Find it…</b> next to the red one and pick the folder in its new place. LoRA folders and their types, watched folders, added models, your ⚑ / ● marks, presets and the Library all follow. Picking a different folder by mistake is refused." ] },
            { h: "A download is stuck, or stops with “stalled”", items: [
              "<b>Why:</b> the connection went quiet (slow Wi-Fi, VPN, a firewall). Downloads give up on a silent connection after 2 minutes, and a model download stops with a message after 5 minutes without progress, instead of hanging for hours.",
              "<b>Fix:</b> check the internet connection and press <b>Download</b> (or Generate / Enhance) again: it carries on from where it stopped. If the queue looks frozen, <b>Tools → ⟲ Force stop &amp; restart</b> first." ] },
            { h: "Hugging Face: “gated”, 401 / 403, or “turned the download down”", items: [
              "<b>“… is gated on Hugging Face”</b>: the model's owners ask you to sign in and accept their licence. On huggingface.co: sign in (free), open the model's page, press the button to agree (some owners review requests; wait for their email), then <b>Settings → Access Tokens → Create new token</b> (type <b>Read</b>) and paste it in Models tab → Add from Hugging Face → Hugging Face access token (or the box that opened). Then press Check / Download again.",
              "<b>“Hugging Face turned the download … down”</b> with a token saved: you haven't accepted that model's licence yet (or it's still waiting for the owner), or the token was deleted or mistyped: make a new one. Fine-grained tokens need “Read access to contents of all public gated repos” switched on; a plain <b>Read</b> token just works.",
              "Most models (SD 1.5, SDXL, the ones in the Models tab) aren't gated and need no account at all." ] },
            { h: "A picture won't open (HEIC, RAW, TIFF…)", items: [
              "PNG, JPEG, WebP, GIF and BMP open directly; HEIC, TIFF, AVIF, PSD, JPEG 2000 and camera RAW files are converted to PNG by macOS's own image reader the first time (kept for next time, up to 8192 px on the long side).",
              "<b>“macOS couldn't read …”</b>: the file is damaged, or a kind macOS doesn't know (a very new camera's RAW may need a macOS update). Export it as JPEG or PNG from Photos or your camera app and use that. Files over 600 MB aren't opened." ] },
            { h: "Batch export / Compress", items: [
              "<b>“Choose the folder to save to”</b>: press <b>Change…</b> and pick a folder (an external drive is fine). <b>“The picture file is missing”</b>: the picture was moved or deleted outside the app; it's skipped and the rest carry on.",
              "<b>Not much smaller</b>: pictures that are already compressed (or tiny) can't shrink much without looking different; Press never trades quality away to make a number look good. Try <b>Strong</b>, a smaller size, or JPEG instead of PNG for photos.",
              "<b>WebP / TIFF</b>: Press works on JPEG and PNG; WebP uses the quality slider and TIFF is lossless." ] },
            { h: "Civitai downloads", items: [
              "<b>Before your first download:</b> sign in. Civitai only lets signed-in accounts download most files, so save your free API key once (the box in the Civitai card, or Models tab → Add from Civitai → Civitai API key). Without it you'll see the first error below.",
              "<b>“Civitai only lets signed-in accounts download this one”</b> (HTTP 401), <b>“wants you to log in”</b> or <b>“sent a page instead of the file”</b>: Civitai now asks for a free account for most downloads. A box opens right in the Civitai card: sign in on civitai.com → <b>Account settings</b> → <b>API Keys</b> → <b>Add API key</b>, copy it, paste it in the box and press <b>Save &amp; download</b>. You only do this once (it's kept on this Mac, sent only to civitai.com; also in Models tab → Add from Civitai → Civitai API key).",
              "<b>“Civitai turned down your API key”</b>: the key is mistyped, deleted or expired; make a new one and paste it. If a new key is refused too, open the model's page on civitai.com while signed in: Early Access models (supporters only for a while) and some with a licence to accept can't be downloaded until then.",
              "<b>“Couldn't reach Civitai: the connection was blocked on this Mac”</b> (the old “[Errno 9] Bad file descriptor”): a firewall app (Little Snitch, LuLu…) or a VPN stopped the app. It already retried through macOS's own downloader; allow Protoart Studio and curl to reach civitai.com in that app, then press <b>⟳ Retry</b>. <b>Clear</b> starts over with an empty link.",
              "<b>“Couldn't reach Civitai”</b>: no internet, a VPN or a firewall app is blocking it, or Strict offline mode is on (Settings → Privacy). <b>“Civitai answered with an error (500/502/503)”</b>: Civitai itself is busy or down; try again in a few minutes.",
              "<b>“The download stopped before the end”</b>: the connection dropped. Press <b>⤓ Download</b> again: it carries on where it stopped. <b>“… is already in …”</b>: you have that file already; it's never overwritten.",
              "<b>“isn't an embedding the app can read”</b>: the file Civitai sent wasn't an SD 1.5 / SDXL embedding, so it wasn't kept.",
              "<b>“Only .safetensors files are downloaded”</b>: older pickle files (.pt / .ckpt) can run hidden code, so they're skipped; pick a version that has a .safetensors file.",
              "<b>FLUX / other checkpoints “can't be loaded from a single file yet”</b>: Civitai checkpoints work for SD 1.5, SDXL, Pony, Illustrious and NoobAI. FLUX LoRAs from Civitai work; for FLUX itself use the Models tab." ] },
            { h: "Pony and other SDXL files: first load takes minutes", items: [
              "On 8 GB Macs, SDXL checkpoint files (Pony, Illustrious, your own) are converted to 8-bit the first time (<i>Converting … to 8-bit (first time only)</i>) and the copy is saved (~2.6 GB in the data folder), so later loads take well under a minute.",
              "<b>“Cannot copy out of meta tensor”</b> came from the old way of loading these files and is fixed; if you still see it, quit and start the app again." ] },
            { h: "Image prompt (IP-Adapter) errors", items: [
              "<b>“h94/IP-Adapter does not appear to have a file named …”</b>: that kind of image prompt (Style, Stronger or Face, SD 1.5 or SDXL) hasn't been downloaded yet. The app now downloads it by itself the first time (about 0.1–1 GB, plus a 2.6 GB image reader shared by all of them). If it still appears, check the internet connection and that <b>Strict offline</b> is off (Settings → Privacy), then generate again.",
              "<b>“SlicedAttnProcessor … slice_size”</b> or <b>“'tuple' object has no attribute 'shape'”</b>: a clash between the image prompt and the memory-saving setting on 8 GB Macs, now fixed. If it appears again, press <b>Tools → ⟲ Force stop &amp; restart</b> and generate again." ] },
            { h: "Enhance & upscale is very slow", items: [
              "The enhancer redraws the picture in tiles, several steps each; on an 8 GB Mac that's the slow part (the size of your original doesn't matter: it's worked at 512 or 768 px on the short side × the upscale).",
              "<b>Fix:</b> choose <b>🚀 Turbo</b> or <b>⚡ Fast</b> with <b>Work at: Auto</b> (512 px on 8 GB Macs), use <b>2×</b> rather than 3–4×, or <b>⏩ Instant</b> when you only need it sharper and bigger. Quit other apps so macOS doesn't swap memory to disk: <b>don't run ProtoArt Beautify, DiffusionBee or another image app at the same time</b>; two image engines on 8 GB is what turns minutes into hours.",
              "If it's still crawling, <b>Tools → ⟲ Force stop &amp; restart</b> and try again with Turbo." ] },
            { h: "Models: “isn't a diffusers model folder”, or a model doesn't show up", items: [
              "Pick model <b>files</b> (.safetensors / .ckpt) or a folder; the app looks inside folders for models. DiffusionBee's own folders don't need adding: those models are listed automatically (marked “DiffusionBee”) while <b>Settings → DiffusionBee</b> is on.",
              "Checkpoints smaller than 1.5 GB are treated as LoRAs/embeddings, not base models. Watched folders are re-checked every few seconds; switch tabs to refresh the list.",
              "ERNIE and other newer models need their whole model folder (the one with <i>model_index.json</i>), or add them from Hugging Face." ] },
            { h: "Image to image, inpainting and ControlNet", items: [
              "ControlNet works with <b>SD 1.5</b> models (built-in and your own) and <b>SDXL</b> models (your own SDXL ControlNets from Civitai); image to image and inpainting work with SD 1.5 and SDXL (not ERNIE).",
              "Inpainting/Expand and ControlNet can't be used together yet: the app tells you to switch one off.",
              "<b>“Paint over the part you want to change first”</b>: paint on the picture in Inpaint mode (pink = will be redrawn). <b>OpenPose “no people found”</b>: use a picture where the whole person is clearly visible." ] },
            { h: "Prompts", items: [
              "Long prompts and <b>(word:1.3)</b> weights work with SD 1.5 and SDXL. ERNIE reads plain descriptions instead (weights are taken literally there).",
              "<b>Enhance prompt</b> downloads a small language model (~1 GB) the first time; after that it takes a few seconds. It always writes in English.",
              "Wildcards <b>{a|b}</b> pick per image; if every picture looks the same, check the seed isn't fixed with only one option in the braces." ] },
            { h: "Editing in PhotoCraft", items: [
              "<b>“PhotoCraft isn't installed”</b>: click <b>Get PhotoCraft</b>, download the macOS .dmg and drag it to Applications, or choose another editor in <b>Settings → Image editor</b>.",
              "<b>The edit didn't come back</b>: save (⌘S) the file the app opened (or save into the same <i>edits</i> folder); it appears in the Library within a few seconds while the “Editing…” bar is showing." ] },
            { h: "Disk space", items: [
              "Models are big (SD 1.5 ~2 GB, SDXL ~5–7 GB, ERNIE ~7 GB, helpers 0.1–3.5 GB each). The app warns before a download if space is short.",
              "Free space by deleting pictures you don't need (Library → select), or downloaded models you don't use: <b>Models tab → 🗑 Delete</b> (files another model still needs are kept).",
              "Downloaded models can live on an external drive: <b>Settings → Downloaded models folder</b>." ] },
            { h: "Still stuck?", items: [
              "<b>Settings → Backend log → Copy all</b> copies the technical details; paste them when asking for help.",
              "<b>Settings → Reset</b> puts settings and folder locations back to the defaults without deleting anything.",
              "<b>Restart backend</b> (Settings) clears most stuck states without closing the app." ] },
          ] },
      ],
      conv: { items: null, search: "", kind: "", selected: [], out_dir: "", free_gb: 0, loras_to_lora_folder: true, result: null, picked: false },
      controlTypes: [],
      controlMap: null,
      cnApplied: null,  // {type, before}: the guide's settings were put in when this ControlNet was picked (undo)
      i2iCompare: false,
      logPath: "",
      controlTimings: [
        { id: "all", label: "The whole time", start: 0, end: 1, tip: "Follows the map from start to finish (normal)" },
        { id: "layout", label: "Layout only", start: 0, end: 0.5, tip: "Sets composition and shapes, then lets the model add its own details" },
        { id: "mostly", label: "Mostly", start: 0, end: 0.8, tip: "Follows the map, but frees the last steps for natural detail" },
        { id: "details", label: "Details only", start: 0.4, end: 1, tip: "Leaves the layout free and uses the map for the finer structure" },
      ],
      libLoraOpen: false,
      modelDirs: { dirs: [], local: [] },
      ck: { items: null, half: true, out_dir: "", msg: "" },  // Tools → Convert → old .ckpt files
      mainFolder: { path: "", access: "", subs: {} },  // Settings → Main folder: one folder that holds everything
      libs: { found: [], used: [], asked: false, offer: [] },
      libOffer: null,
      ctx: null,        // Library right-click menu: {x, y, r, ids}   // the "Existing model library detected" window
      libBusy: false,
      libMsg: "",
      localErr: "",
      localOk: "",  // "Model imported" note after Add from your Mac
      train: Object.assign({ model: "", folder: "", items: [], capMode: "tags", name: "", trigger: "", format: "kohya", kind: "character",
                             precision: "fp16", save_every: 0, preset: "standard", steps: 1000, lr: 0.0001, rank: 16,
                             resolution: 512, targets: "attention", flip: true, train_text_encoder: false,
                             dirty: false, result: null, lossHistory: [], secPerStep: null, spsBy: {},
                             mode: "lora", emb: { name: "", init: "", kind: "object", vectors: 2, steps: 1500, save_every: 0 } },
                           (() => { const t = loadJSON("protoart.train", {}); delete t.items; delete t.result; delete t.lossHistory; delete t.dirty; return t; })()),
      merge: { family: "", items: [], q: "", mode: "exact", rank: 32, name: "", result: null, error: "", busy: false },
      hub: { source: "", busy: false, error: "", found: null, needToken: false, token: "", tokenSet: false, msg: "" },
      civ: { url: "", busy: false, error: "", found: null, dest: "", key: "", keySet: false, msg: "", needKey: "", last: "", dupAsk: false },  // From Civitai     // {job, image, src}
      trigInfo: {},          // lora name -> {found, suggested, saved}
      trigEdit: { name: "", text: "" },
      embeddings: [],
      vaes: [],
      vaeByModel: loadJSON("protoart.vaeByModel", {}),   // model id -> the VAE you picked for it
      vaeFolder: "",
      userControlnets: [],
      controlnetFolder: "",
      extras: { tab: "emb", family: "", q: "" },   // Models tab → Embeddings, VAEs & ControlNets
      loraDirs: { dirs: [], default: "" },
      embDirs: { dirs: [], default: "" },
      embMsg: "",
      modelFilter: "all",
      shop: { family: "all", search: "", fav: false }, // Models tab
      mpanels: { own: false, civitai: false, hub: false, fix: false, trig: false, tags: false },
      loraTags: loadJSON("protoart.loraTags", {}),
      loraWeights: loadJSON("protoart.loraWeights", {}),  // ★ default weight per LoRA: { "name": 0.6 }
      modelDefaults: loadJSON("protoart.modelDefaults", {}),  // your own starting sampler / steps / guidance per model  // your own tags on LoRAs: { "LoRA name": ["pencil", "ink"] }
      tagBatch: { q: "", sel: [], tag: "", asTrig: loadJSON("protoart.tagAsTrig", true), busy: false },  // Models → LoRA tags
      trigBatch: { q: "", sel: [], word: "", busy: false },  // Models → Trigger words // Models tab panels (closed until you open one)
      shopFolds: loadJSON("protoart.shopFolds", { get: true, big: true }),  // Models tab model groups folded (On this Mac open)
      dl: { path: "", default: "", custom: false, missing: null, free_gb: null, sizeGb: null, move: true, busy: false }, // downloaded-models folder
      modelSearch: "",
      tool: { source: null, results: [], dragOver: false, working: null, latest: null, healing: false, protecting: false, promptText: "", promptCopied: false, view: "split", split: 0.5 },
      PATCH_NODES,
      patchInY: PATCH_IN_Y,
      patchOutY: PATCH_OUT_Y,
      patch: { on: loadJSON("protoart.patchOn", false), cables: loadJSON("protoart.patchV2", false) ? loadJSON("protoart.patch", PATCH_DEFAULT) : [],  // 1.2.1: cables start cleared
               extra: loadJSON("protoart.patchExtra", []),  // more copies of modules you added: [{id: "faces#2", type: "faces"}]
               order: loadJSON("protoart.patchOrder", []),  // the modules' left-to-right order, if you moved them
               moving: null,  // a module being dragged sideways by its name plate
               sel: "",       // the module you clicked: its settings are shown (a copy's own, or the rack module below)
               copySettings: loadJSON("protoart.patchCopySettings", {}),  // settings of the copies you added, by id
               presets: loadJSON("protoart.patchPresets", []),  // saved signal flows: cables, copies, order, copy settings
               preset: "", presetName: "",
               hidden: loadJSON("protoart.patchHidden", []),  // modules you took off the rack (＋ Add a module puts them back)
               protect: null,       // 🛡 areas painted to keep as they were: {data, for}
               stash: null,         // the first copy's settings, while a copy's are shown in its rack unit
               drag: null, running: "", step: 0, steps: 0, stopped: false, msg: "" },
      freeConfirm: null,
      logs: "",
      backendDead: false,
      debounceTimer: null,
    };
  },

  computed: {
    detailOriginalPrompt() {
      const d = this.detail;
      const o = d && d.settings && d.settings.original_prompt;
      return o && o !== d.prompt ? o : "";
    },
    busy() {
      return this.queue.length > 0;
    },
    runningJob() {
      return this.queue.find((j) => j.state === "running");
    },
    currentModel() {
      return this.models.find((m) => m.id === this.form.model);
    },
    family() {
      return this.currentModel?.family || "sdxl";
    },
    familyLabel() {
      return FAMILY_LABEL[this.family];
    },
    clipSkipOk() {
      // CLIP skip changes how the prompt is read by SD 1.5 / SDXL text encoders (not ERNIE or other models)
      return ["sd15", "sdxl"].includes(this.family) && this.currentModel?.engine !== "mlx";
    },
    clipSkipRec() {
      return this.currentModel?.clip_skip || null;
    },
    clipSkipOptions() {
      // SDXL always reads the second-to-last layer, which other apps call "CLIP skip 2", so its scale starts at 2
      if (this.family === "sdxl")
        return [{ v: 2, label: "Standard", title: "How SDXL normally reads prompts (called CLIP skip 2 in other apps)" },
                { v: 3, label: "3", title: "Skip one more layer: looser, more stylised" },
                { v: 4, label: "4", title: "Skip two more layers" }];
      return [{ v: 1, label: "Off", title: "Normal" },
              { v: 2, label: "2", title: "Skip the last layer: what anime-style SD 1.5 models were made with" },
              { v: 3, label: "3", title: "Skip 2 layers: looser" },
              { v: 4, label: "4", title: "Skip 3 layers" }];
    },
    clipSkipText() {
      const v = this.form.clip_skip || 1, rec = this.clipSkipRec;
      if (this.family === "sdxl") return v <= 2 ? "normal for SDXL" : "looser, more stylised";
      if (v === rec) return v === 2 ? "recommended: anime-style model" : "recommended for this model";
      if (v === 1) return rec === 2 ? "this model usually wants 2" : "normal";
      return v === 2 ? "common for anime models" : "stronger, looser";
    },
    loraClipSkip() {
      // LoRAs that say (in their file) which CLIP skip they were trained with
      const out = [];
      for (const l of this.form.loras) {
        const info = this.loras.find((x) => x.name === l.name);
        if (info?.clip_skip) out.push({ name: l.name, value: info.clip_skip });
      }
      return out;
    },
    clipSkipHint() {
      const want = this.loraClipSkip.find((x) => x.value !== (this.form.clip_skip || 1));
      return want ? want : null;
    },
    // Create shows only models that are ready on this Mac (downloaded, your own files, DiffusionBee, ones you
    // added), plus the one that's selected. The rest are in the Models tab, ready to download.
    installedModels() {
      return this.models.filter((m) => (this.isInstalled(m) || m.id === this.form.model) && this.safeModel(m));
    },
    hideNsfw() {
      return !!this.opts.hide_nsfw || !!this.opts.sfw_mode;  // safe-for-work mode always hides
    },
    libModels() {
      return this.models.filter((m) => this.safeModel(m));
    },
    // Library pictures, minus the ones made with a model or LoRA you tagged ● People while Hide People is on
    libShown() {
      if (!this.opts.hide_people) return this.lib.results;
      const pm = new Set(this.models.filter((m) => m.people).map((m) => m.id));
      const pl = new Set(this.loras.filter((l) => l.people).map((l) => l.name));
      if (!pm.size && !pl.size) return this.lib.results;
      return this.lib.results.filter((r) => !pm.has(r.model) && !(r.loras || []).some((l) => pl.has(l.name)));
    },
    libPeopleHidden() {
      return this.lib.results.length - this.libShown.length;
    },
    libLoraChips() {
      if (!this.opts.hide_people) return this.libLoras;
      const pl = new Set(this.loras.filter((l) => l.people).map((l) => l.name));
      return this.libLoras.filter((l) => !pl.has(l.name));
    },
    markedItems() {
      // models and LoRAs you marked yourself (Settings → Content filter)
      return [
        ...this.models.filter((m) => m.nsfw_mark != null).map((m) => ({ kind: "model", name: m.id, label: m.label, nsfw: m.nsfw_mark })),
        ...this.loras.filter((l) => l.nsfw_mark != null).map((l) => ({ kind: "lora", name: l.name, label: l.name, nsfw: l.nsfw_mark })),
      ];
    },
    allLoraTags() {  // every tag you've used, with how many LoRAs have it
      const m = new Map();
      for (const [name, tags] of Object.entries(this.loraTags || {})) for (const t of tags || []) m.set(t, (m.get(t) || 0) + 1);
      return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([tag, n]) => ({ tag, n }));
    },
    tagList() {  // Models → LoRA tags: the LoRAs matching the search (name, or #tag), 100 at most
      const q = this.tagBatch.q.trim().toLowerCase();
      const tq = q.startsWith("#") ? q.slice(1) : "";
      return this.loras.filter((l) => !q || (tq ? this.loraTagsOf(l.name).some((t) => t.includes(tq))
        : l.name.toLowerCase().includes(q) || this.loraTagsOf(l.name).includes(q))).slice(0, 100);
    },
    trigList() {  // Models → Trigger words: the LoRAs matching the search (60 at most)
      const q = this.trigBatch.q.trim().toLowerCase();
      return this.loras.filter((l) => !q || l.name.toLowerCase().includes(q)).slice(0, 60);
    },
    loraFixList() {
      // LoRAs and your own base model files, by name (or only the ones of unknown type)
      const q = this.loraFix.q.trim().toLowerCase();
      if (!q && !this.loraFix.unknownOnly && !this.loraFix.showAll) return [];
      const loras = this.loras.map((l) => ({ ...l, kind: "lora", key: "l:" + l.name, label: l.name,
        unknown: !l.family || l.family === "unknown" }));
      const models = this.models.filter((m) => m.single_file).map((m) => ({ ...m, kind: "model", key: "m:" + m.id,
        label: m.label, name: m.id, unknown: !["sd15", "sdxl"].includes(m.family) }));
      return [...models, ...loras].filter((x) => (!q || x.label.toLowerCase().includes(q)) &&
        (!this.loraFix.unknownOnly || x.unknown)).slice(0, this.loraFix.showAll ? 2000 : 60);
    },
    peopleItems() {
      return [
        ...this.models.filter((m) => m.people).map((m) => ({ kind: "model", name: m.id, label: m.label, item: m })),
        ...this.loras.filter((l) => l.people).map((l) => ({ kind: "lora", name: l.name, label: l.name, item: l })),
      ];
    },
    autoFlagged() {
      return {
        models: this.models.filter((m) => m.nsfw_auto && m.nsfw_mark == null).length,
        loras: this.loras.filter((l) => l.nsfw_auto && l.nsfw_mark == null).length,
      };
    },
    previewCovered() {
      // the live preview can't be checked while it's being made: cover it when the prompt asks for NSFW
      if (!this.hideNsfw || !this.livePreview) return false;
      const j = this.queue.find((x) => x.id === this.livePreview.job);
      return !!(j && j.nsfw_prompt);
    },
    downloadableCount() {
      return this.models.filter((m) => !this.isInstalled(m)).length;
    },
    shopModels() {
      const q = this.shop.search.trim().toLowerCase();
      const fam = this.shop.family;
      const list = this.models.filter((m) => this.safeModel(m) &&
        (fam === "all" || (fam.startsWith("fl:") ? m.flavor === fam.slice(3) : m.family === fam)) &&
        (!this.shop.fav || this.isFav(m)) &&
        (!q || `${m.label} ${FAMILY_LABEL[m.family]} ${m.quant || ""}`.toLowerCase().includes(q)));
      return [
        { id: "ready", title: "On this Mac", hint: "ready to use", models: list.filter((m) => this.isInstalled(m)) },
        { id: "get", title: "Available to download", hint: "downloads once the first time you make a picture with it, then works offline",
          models: list.filter((m) => !this.isInstalled(m) && m.enough_ram) },
        { id: "big", title: "Needs more memory than this Mac has", hint: "you can still download them (for another Mac, or to try with Settings → Load anyway)",
          models: list.filter((m) => !this.isInstalled(m) && !m.enough_ram) },
      ].filter((g) => g.models.length);
    },
    modelFilters() {
      const count = (f) => this.installedModels.filter(f).length;
      const list = [
        { id: "all", label: "All", count: this.installedModels.length },
        { id: "fav", label: "★ Favorites", count: count((m) => this.isFav(m)) },
        { id: "flux2", label: "Klein 4B", count: count((m) => m.family === "flux2") },
        { id: "flux", label: "FLUX", count: count((m) => m.family === "flux") },
        { id: "ernie", label: "ERNIE", count: count((m) => m.family === "ernie") },
        { id: "anima", label: "Anima", count: count((m) => m.family === "anima") },
        { id: "sdxl", label: "SDXL", count: count((m) => m.family === "sdxl") },
        { id: "sd15", label: "SD 1.5", count: count((m) => m.family === "sd15") },
        ...Object.entries(FLAVOR_LABEL).map(([k, label]) => ({ id: "fl:" + k, label, count: count((m) => m.flavor === k) })),
        { id: "mine", label: "My checkpoints", count: count((m) => m.id.startsWith("file:")) },
        { id: "dbee", label: "DiffusionBee (BEE)", count: count((m) => m.id.startsWith("tdict:")) },
        { id: "hub", label: "Added", count: count((m) => m.id.startsWith("hf:") || m.id.startsWith("local:")) },
      ];
      return list.filter((f) => f.id === "all" || f.count);
    },
    detailList() {  // the pictures the open one is among (the ‹ › arrows step through them)
      if (!this.detail) return [];
      if (this.tab === "library") return this.libShown;
      if (this.tab === "i2i") return this.i2iResults;
      return this.createSession;
    },
    detailIndex() {
      return this.detailList.findIndex((r) => r.id === this.detail?.id);
    },
    helpHits() {  // How to → search: every tip whose text has all the words typed
      const words = this.helpQ.toLowerCase().split(/\s+/).filter(Boolean);
      if (!words.length) return [];
      const strip = (h) => String(h).replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").toLowerCase();
      const out = [];
      for (const t of this.helpTabs || []) {
        for (const sec of t.sections || []) {
          const head = strip(t.title + " " + sec.h);
          for (const line of sec.items || []) {
            const text = head + " " + strip(line);
            if (words.every((w) => text.includes(w))) out.push({ tabId: t.id, tab: t.title, icon: t.icon, sec: sec.h, html: line });
          }
        }
      }
      return out;
    },
    negPresets() {  // 📒 negative prompts saved on their own
      return this.promptPresets.filter((p) => p.kind === "negative");
    },
    promptOnlyPresets() {
      return this.promptPresets.filter((p) => p.kind !== "negative");
    },
    inpaintModel() {
      // Image to image → Inpaint / Expand with an SD 1.5 (or SDXL) model: an inpainting checkpoint of the same type,
      // when you have one, does the painting (it's made for filling in; Create never uses it). Off: your own model.
      if (this.opts.use_inpaint_model === false) return null;
      return this.models.find((m) => m.inpaint && m.family === this.family && this.isInstalled(m) && this.safeModel(m)) || null;
    },
    inpaintModels() {
      return this.models.filter((m) => m.inpaint && this.isInstalled(m));
    },
    filteredModels() {
      const q = this.modelSearch.trim().toLowerCase();
      // favourites first, then the usual order; inpainting models only fill in painted areas (Image to image)
      const list = this.sortModels([...this.installedModels].filter((m) => !m.inpaint));
      return list.filter((m) => {
        if (this.modelFilter === "fav" ? !this.isFav(m)
            : this.modelFilter === "mine" ? !m.id.startsWith("file:")
            : this.modelFilter === "dbee" ? !m.id.startsWith("tdict:")
            : this.modelFilter === "hub" ? !(m.id.startsWith("hf:") || m.id.startsWith("local:"))
            : this.modelFilter.startsWith("fl:") ? m.flavor !== this.modelFilter.slice(3)
            : this.modelFilter !== "all" && m.family !== this.modelFilter) return false;
        if (!q) return true;
        return `${m.label} ${FAMILY_LABEL[m.family]} ${m.quant || ""}`.toLowerCase().includes(q);
      });
    },
    modelGroups() {
      const groups = [
        { name: "FLUX.2 Klein 4B", models: this.models.filter((m) => m.family === "flux2") },
        { name: "FLUX.1", models: this.models.filter((m) => m.family === "flux") },
        { name: "ERNIE-Image", models: this.models.filter((m) => m.family === "ernie") },
        { name: "Anima (anime)", models: this.models.filter((m) => m.family === "anima") },
        { name: "Stable Diffusion", models: this.models.filter((m) => !["ernie", "flux", "flux2", "anima"].includes(m.family) && !m.id.includes(":")) },
        { name: "Your checkpoints", models: this.models.filter((m) => m.id.startsWith("file:")) },
        { name: "From DiffusionBee", models: this.models.filter((m) => m.id.startsWith("tdict:")) },
        { name: "Your model files", models: this.models.filter((m) => m.id.startsWith("local:")) },
        { name: "Added from Hugging Face", models: this.models.filter((m) => m.id.startsWith("hf:")) },
      ];
      return groups.filter((g) => g.models.length);
    },
    compatibleLoras() {
      // only LoRAs made for this model's type (ones of unknown type are left out: see unknownTypeLoras)
      const list = this.loras.filter((l) => l.family === this.family && !(this.hideNsfw && l.nsfw) &&
        !(this.opts.hide_people && l.people));
      // Pony / Illustrious / NoobAI model: LoRAs made for it first, then the rest
      const fl = this.currentFlavor;
      return fl ? [...list.filter((l) => l.flavor === fl), ...list.filter((l) => l.flavor !== fl)] : list;
    },
    compatibleEmbeddings() {
      // only embeddings made for this model's type (SD 1.5, or SDXL incl. Pony / Illustrious / NoobAI)
      return (this.embeddings || []).filter((e) => e.family === this.family && !(this.hideNsfw && e.nsfw_auto));
    },
    unknownTypeLoras() {
      return this.loras.filter((l) => !l.family || l.family === "unknown").length;
    },
    currentFlavor() {
      return this.currentModel?.flavor || null;
    },
    tagsInPrompt() {
      const t = this.flavorTags;
      return !!t && (this.form.prompt || "").toLowerCase().includes(t.split(",")[0].trim().toLowerCase());
    },
    flavorTags() {
      return this.currentFlavor ? QUALITY_TAGS[this.currentFlavor] : "";
    },
    addableLoras() {
      const used = new Set(this.form.loras.map((l) => l.name));
      return this.compatibleLoras.filter((l) => !used.has(l.name));
    },
    filteredLoras() {
      const q = this.loraSearch.trim().toLowerCase();
      const k = this.loraKind;
      const tagQ = q.startsWith("#") ? q.slice(1) : "";
      return this.sortLoras(this.addableLoras.filter((l) => (!q || (tagQ ? this.loraTagsOf(l.name).some((t) => t.includes(tagQ))
          : l.name.toLowerCase().includes(q) || this.loraTagsOf(l.name).includes(q)))
        && (!this.loraFavOnly || this.isFavLora(l))
        && (!k || (k.startsWith("tag:") ? this.loraTagsOf(l.name).includes(k.slice(4))
          : k === "people" ? !!l.people : k === "not_people" ? !l.people : this.loraKindOf(l) === k || l.flavor === k))));
    },
    loraKinds() {  // what the LoRA list can be narrowed to: file kind, and Pony / Illustrious / NoobAI when present
      const kinds = new Map();
      for (const l of this.addableLoras) {
        const t = this.loraKindOf(l);
        kinds.set(t, (kinds.get(t) || 0) + 1);
        if (l.flavor) kinds.set(l.flavor, (kinds.get(l.flavor) || 0) + 1);
      }
      const label = { safetensors: ".safetensors", diffusionbee: "DiffusionBee", ...Object.fromEntries(Object.entries(FLAVOR_LABEL)) };
      const out = [...kinds.entries()].map(([id, n]) => ({ id, n, label: label[id] || id }));
      const people = this.addableLoras.filter((l) => l.people).length;  // ● People: LoRAs of real people
      out.push({ id: "people", n: people, label: "● People" }, { id: "not_people", n: this.addableLoras.length - people, label: "Not People" });
      const tags = new Map();  // your own 🏷 tags
      for (const l of this.addableLoras) for (const t of this.loraTagsOf(l.name)) tags.set(t, (tags.get(t) || 0) + 1);
      for (const [t, n] of [...tags.entries()].sort((a, b) => a[0].localeCompare(b[0]))) out.push({ id: "tag:" + t, n, label: "🏷 " + t });
      return out;
    },
    shownLoras() {
      return this.filteredLoras.slice(0, 200); // render at most 200 rows at once (keeps older Macs smooth)
    },
    // Image to image's own LoRA picker (the same LoRAs as Create: both tabs use one model and one LoRA list)
    i2iShownLoras() {
      const q = this.i2iLora.q.trim().toLowerCase();
      const tq = q.startsWith("#") ? q.slice(1) : "";  // #pencil: your 🏷 tag
      return this.sortLoras(this.addableLoras.filter((l) => !q || (tq ? this.loraTagsOf(l.name).some((t) => t.includes(tq))
        : l.name.toLowerCase().includes(q) || this.loraTagsOf(l.name).includes(q)))).slice(0, 200);
    },
    moreLoras() {
      return Math.max(0, this.filteredLoras.length - 200);
    },
    sizeKey: {
      get() {
        return `${this.form.width}x${this.form.height}`;
      },
      set(v) {
        const [w, h] = v.split("x").map(Number);
        this.form.width = w;
        this.form.height = h;
      },
    },
    trainModels() {
      return this.models.filter((m) => !this.trainBlock(m));
    },
    trainModelGroups() {
      // the 8-bit SDXL trains on its full-precision original, so list only that one
      const hasFull = this.models.some((m) => m.id === "sdxl_base_fp16");
      const by = (f) => this.models.filter((m) => f(m) && this.safeModel(m) && !(hasFull && m.id === "sdxl_int8"));
      return [
        { name: "SD 1.5", models: by((m) => m.family === "sd15") },
        { name: "SDXL", models: by((m) => m.family === "sdxl") },
        { name: "ERNIE", models: by((m) => m.family === "ernie") },
        { name: "Other", models: by((m) => !["sd15", "sdxl", "ernie"].includes(m.family)) },
      ].filter((g) => g.models.length);
    },
    trainKinds() {
      return [
        { id: "character", label: "Character / person", tip: "A face or character that should look the same in every picture" },
        { id: "object", label: "Object", tip: "A product, prop, vehicle, pet…" },
        { id: "style", label: "Art style", tip: "A look: brushwork, line art, colours, a photographic style" },
        { id: "concept", label: "Pose / concept", tip: "A pose, outfit, composition or effect" },
      ];
    },
    // What size the training pictures should be, for this base model and this kind of LoRA.
    trainSizeAdvice() {
      const fam = this.trainFamily;
      const ram = this.sys.ram_gb || 0;
      const native = { sd15: 512, sdxl: 1024, ernie: 1024 }[fam] || 512;
      const label = { sd15: "SD 1.5", sdxl: "SDXL", ernie: "ERNIE-Image" }[fam] || "this model";
      const train = this.train.resolution || native;
      const kind = this.train.kind || "character";
      const shapes = {
        sd15: "512 × 512 square (or 512 × 768 / 768 × 512 for portrait and landscape)",
        sdxl: "1024 × 1024 square (or 832 × 1216 / 1216 × 832)",
        ernie: "1024 × 1024 square (or 848 × 1264 / 1264 × 848)",
      }[fam] || `${native} × ${native}`;
      const tips = {
        character: ["15–30 pictures of the same person or character, cropped so they fill most of the frame.",
                    "Mix it up: about half close-ups of the face, a third half-body, the rest full-body; different expressions, angles, clothes and backgrounds.",
                    "Avoid other people in the picture, heavy filters, text and watermarks."],
        object: ["10–20 pictures of the object from different sides and distances.",
                 "Use different backgrounds and lighting so the LoRA learns the object, not the room it was in.",
                 "Keep the object sharp and centred; crop away empty space."],
        style: ["20–40 pictures in the style, of many different subjects (people, places, things).",
                "Use whole pictures rather than tight crops, so the composition and colours are learned too.",
                "Keep the style consistent: leave out pictures that don't look like the rest."],
        concept: ["15–30 pictures that all show the pose, outfit or effect clearly.",
                  "Vary everything else (people, colours, backgrounds) so only the concept is learned.",
                  "Describe everything except the concept in the captions."],
      }[kind];
      // pictures smaller than this look soft after training: the model's own size, or 768 for SDXL / ERNIE trained lighter
      const minSide = Math.max(Math.min(native, train), fam === "sd15" ? 512 : 768);
      const sized = this.train.items.filter((it) => it.width && it.height);
      const small = sized.filter((it) => Math.min(it.width, it.height) < minSide).length;
      return { label, native, train, shapes, tips, minSide, small, checked: sized.length,
               lowRam: fam !== "sd15" && ram && ram < 16 };
    },
    trainFastErnie() {
      const m = this.models.find((x) => x.id === this.train.model);
      return !!(m && m.family === "ernie" && m.engine === "mlx");
    },
    trainFamily() {
      return this.models.find((m) => m.id === this.train.model)?.family || "sd15";
    },
    trainModelNote() {
      const ram = this.sys.ram_gb || 0;
      if (this.trainFamily === "sdxl")
        return ram && ram < 16 ? "SDXL training needs 16 GB+ to run well; on this Mac it will work but be very slow (expect many hours). Use 768 to lighten it."
                               : "SDXL: UNet LoRA at 1024 (or 768 to save memory and time).";
      if (this.trainFastErnie)
        return ram && ram < 16
          ? "ERNIE on the fast engine: trains a LoRA on the 4-bit model. Experimental on 8 GB Macs: it may run out of memory or take many hours. Use 512 px, rank 8 and attention layers, and close other apps."
          : "ERNIE on the fast engine: trains a LoRA on the 4-bit model (low-memory settings are used automatically). The LoRA works with the fast-engine ERNIE.";
      if (this.trainFamily === "ernie")
        return "ERNIE: transformer LoRA (flow matching), saved in Diffusers format. Needs the full-precision model and 32 GB+.";
      return ram && ram < 16 ? "SD 1.5: the best fit for this Mac. Your DiffusionBee, downloaded and own SD 1.5 models all work as a base."
                             : "SD 1.5: fastest to train. On this Mac you can also train SDXL" + (ram >= 32 ? " and ERNIE." : ".");
    },
    trainSps() {
      // seconds per training step: measured on this Mac per model type, otherwise a rough guess
      return (fam) => {
        const got = (this.train.spsBy || {})[fam];
        if (got) return got;
        const ram = this.sys.ram_gb || 0;
        if (fam === "sdxl") return ram && ram < 16 ? 25 : 6;
        if (fam === "ernie") return ram && ram < 16 ? 40 : 10;
        return ram && ram < 12 ? 4 : 2;
      };
    },
    queuedCount() {
      return this.queue.filter((x) => x.state === "queued").length;
    },
    trainSkipped() {
      return this.train.items.filter((it) => it.fit === "skip").length;
    },
    trainCaptioned() {
      return this.train.items.filter((i) => (i.caption || "").trim()).length;
    },
    trainPresets() {
      const n = this.train.items.length || 15;
      const base = Math.min(2000, Math.max(400, Math.round(n * 100 / 10) * 10));
      return [
        { id: "quick", label: "Quick", steps: Math.round(base * 0.5 / 50) * 50, lr: 0.0002, rank: 8, tip: "Fast test run" },
        { id: "standard", label: "Standard", steps: base, lr: 0.0001, rank: 16, tip: "Good default (~100 steps per image)" },
        { id: "thorough", label: "Thorough", steps: Math.min(4000, base * 2), lr: 0.0001, rank: 32, tip: "More detail, takes twice as long" },
      ];
    },
    trainBlock() {
      return (m) => {
        if (this.train.mode === "embedding") {
          if (m.family !== "sd15" || m.engine === "mlx") return "embeddings: SD 1.5 only for now";
          return "";
        }
        if (m.family === "ernie" && m.engine === "mlx") return ""; // fast engine: trains on the 4-bit model
        if (m.family === "ernie") {
          if (this.sys.ram_gb && this.sys.ram_gb < 32) return "needs a Mac with 32 GB+ (use the ⚡ fast engine one)";
          if (m.quant) return "compressed build can't be trained; use the ⚡ fast engine one";
        }
        if (!["sd15", "sdxl", "ernie"].includes(m.family)) return "not supported for training";
        if (m.engine === "mlx") return "not supported for training";
        if (m.family === "sdxl" && (m.disk_gb || /8-bit/.test(m.label)) && !this.models.some((x) => x.id === "sdxl_base_fp16"))
          return "8-bit build can't be trained";
        return "";
      };
    },
    trainProblem() {
      if (!this.trainModels.length) return "No SD 1.5 model available: download Stable Diffusion 1.5 in Create first.";
      if (!this.train.folder) return "";
      if (this.train.items.length && this.train.items.length < 3) return "Use at least 3 images.";
      if (this.train.name && this.loras.some((l) => l.name === this.train.name.trim()))
        return "There's already a LoRA with this name; pick another.";
      return "";
    },
    embProblem() {
      const e = this.train.emb, n = (e.name || "").trim();
      if (n && !/^[\w-]+$/.test(n)) return "Use letters, numbers, _ or - only (it's the word you'll type in prompts).";
      if (n && (this.embeddings || []).some((x) => x.name.toLowerCase() === n.toLowerCase())) return "There's already an embedding with this name; pick another.";
      if (this.train.items.length && this.train.items.length < 3) return "Use at least 3 pictures.";
      return "";
    },
    embReady() {
      const m = this.models.find((x) => x.id === this.train.model);
      return m && !this.trainBlock(m) && this.train.items.length >= 3 && (this.train.emb.name || "").trim() && !this.embProblem && !this.trainJob;
    },
    embEstimate() {
      if (!this.train.items.length) return "";
      const measured = (this.train.spsBy || {}).sd15_ti;
      const sps = measured || ((this.sys.ram_gb || 8) < 12 ? 2.5 : 1.2);
      return `≈ ${this.formatDuration(this.train.emb.steps * sps)} on this Mac` + (measured ? "" : " (rough guess until the first run)");
    },
    mergeFamilies() {
      const c = (f) => this.loras.filter((l) => l.family === f && /\.safetensors$/i.test(l.file || "")).length;
      return [["sd15", "SD 1.5"], ["sdxl", "SDXL"], ["flux", "FLUX"], ["flux2", "Klein 4B"], ["ernie", "ERNIE"]].map(([id, label]) => ({ id, label, count: c(id) }))
        .filter((f) => f.count);
    },
    mergeFamily() {
      return this.merge.family || (this.mergeFamilies.find((f) => f.id === this.family) || this.mergeFamilies[0] || {}).id || "";
    },
    mergeChoices() {
      const q = this.merge.q.trim().toLowerCase(), picked = new Set(this.merge.items.map((i) => i.name));
      return this.loras.filter((l) => l.family === this.mergeFamily && /\.safetensors$/i.test(l.file || "") && !picked.has(l.name)
        && !(this.hideNsfw && l.nsfw) && !(this.opts.hide_people && l.people) && (!q || l.name.toLowerCase().includes(q))).slice(0, 60);
    },
    mergeOtherCount() {
      // LoRAs that can't be merged here (DiffusionBee .tdict files, unknown type): shown as a note
      return this.loras.filter((l) => l.family === this.mergeFamily && !/\.safetensors$/i.test(l.file || "")).length;
    },
    tiledJob() {
      return this.queue.find((j) => j.cmd === "tiled_upscale");
    },
    tiledEstimate() {  // the result's size and a rough count of the work
      const src = this.tool.source || {}, sc = this.tup.scale;
      if (!src.width) return "";
      const W = Math.round(src.width * sc / 8) * 8, H = Math.round(src.height * sc / 8) * 8;
      const t = this.family === "sdxl" ? 1024 : 512, st = t * 0.75;
      const n = Math.max(1, Math.ceil((W - t) / st) + 1) * Math.max(1, Math.ceil((H - t) / st) + 1);
      return `${W} × ${H} px · ${n} tile${n === 1 ? "" : "s"} × ${Math.max(1, Math.round(this.tup.steps * this.tup.strength))} steps`;
    },
    regionBoxes() {  // the areas as boxes (x0, y0, x1, y1 in 0..1), the same way the engine splits them
      const r = this.form.regions || {}, n = (r.prompts || []).length || 2;
      const split = (parts) => { const p = parts.map((x) => Math.max(0.01, Number(x) || 1)), t = p.reduce((a, b) => a + b, 0); let acc = 0; return p.map((x) => { const s = [acc / t, (acc + x) / t]; acc += x; return s; }); };
      if (r.layout === "grid" && String(r.ratios).includes(";")) {
        const rows = String(r.ratios).split(";").filter((x) => x.trim()), hs = split(rows.map(() => 1)), out = [];
        rows.forEach((row, i) => split(row.split(",").filter((x) => x.trim())).forEach(([a, b]) => out.push([a, hs[i][0], b, hs[i][1]])));
        return out.slice(0, n);
      }
      let parts = String(r.ratios || "").split(/[;,]/).filter((x) => x.trim());
      parts = [...parts, ...Array(n).fill(1)].slice(0, n);
      return split(parts).map(([a, b]) => (r.layout === "rows" ? [0, a, 1, b] : [a, 0, b, 1]));
    },
    quantJob() {
      return this.queue.find((j) => j.cmd === "quantize");
    },
    quantModels() {
      return this.models.filter((m) => ["sd15", "sdxl"].includes(m.family) && m.engine !== "mlx" && !m.inpaint
        && !/(8|6|4)-bit/.test(m.label) && this.isInstalled(m));
    },
    mergeJob() {
      return this.queue.find((j) => j.cmd === "lora_merge");
    },
    mergeProblem() {
      const n = this.merge.name.trim();
      if (n && this.loras.some((l) => l.name === n)) return "There's already a LoRA with this name; pick another.";
      return "";
    },
    mergeReady() {
      return this.merge.items.length >= 2 && this.merge.name.trim() && !this.mergeProblem && !this.mergeJob;
    },
    trainReady() {
      return this.train.model && this.train.items.length >= 3 && this.train.name.trim() && !this.trainProblem && !this.trainJob;
    },
    trainEstimate() {
      if (!this.train.items.length) return "";
      const fam = this.trainFamily;
      const measured = (this.train.spsBy || {})[fam];
      const sps = this.trainSps(fam);
      const base = fam === "sdxl" ? 1024 : fam === "ernie" ? 768 : 512;
      const mult = measured ? 1 : (this.train.resolution / base) ** 2 * (this.train.train_text_encoder ? 1.3 : 1);
      return `≈ ${this.formatDuration(this.train.steps * sps * mult)} on this Mac` + (measured ? "" : " (rough guess until the first run)");
    },
    trainJob() {
      return this.queue.find((j) => j.cmd === "train_lora" || j.cmd === "train_embedding");
    },
    trainPct() {
      const p = this.progress, j = this.trainJob;
      if (!j || !p || p.job !== j.id || j.state !== "running") return 0;
      return Math.round((p.step / p.steps) * 100);
    },
    trainProgressText() {
      const j = this.trainJob, p = this.progress;
      if (!j) return "";
      if (j.state !== "running") return "Waiting in the queue…";
      if (!p || p.job !== j.id) return this.status;
      const sps = this.train.secPerStep;
      return `Step ${p.step}/${p.steps}` + (p.loss != null ? ` · loss ${p.loss}` : "") +
        (sps ? ` · ${sps}s/step · ${this.formatDuration((p.steps - p.step) * sps)} left` : "");
    },
    lossPoints() {
      const h = this.train.lossHistory;
      if (h.length < 2) return "";
      const max = Math.max(...h), min = Math.min(...h), span = max - min || 1;
      return h.map((v, i) => `${(i / (h.length - 1)) * 300},${70 - ((v - min) / span) * 66 - 2}`).join(" ");
    },
    vecJob() {
      return this.queue.find((j) => j.cmd === "vectorize");
    },
    toolJob() {
      return this.tool.working ? this.queue.find((j) => ["postprocess", "heal", "vectorize"].includes(j.cmd)) : null;
    },
    patchNodes() {  // PICTURE, the modules, the copies you added, then RESULT (always last)
      const base = PATCH_NODES.filter((n) => n.id !== "result");
      const extra = (this.patch.extra || []).map((x) => {
        const t = PATCH_NODES.find((n) => n.id === x.type) || {};
        const k = x.id.split("#")[1] || "2";
        return { ...t, id: x.id, label: `${t.label} ${k}`, sub: (t.sub || "") + " · copy" };
      });
      const hidden = this.patch.hidden || [];
      const mid = [...base.filter((n) => n.id !== "src" && !hidden.includes(n.id)), ...extra];
      const ord = this.patch.order || [];
      const at = (n) => { const k = ord.indexOf(n.id); return k < 0 ? 1000 + mid.indexOf(n) : k; };
      mid.sort((a, b) => at(a) - at(b));
      return [base.find((n) => n.id === "src"), ...mid, PATCH_NODES.find((n) => n.id === "result")];
    },
    patchW() {  // the panel's drawing widens with every module you add
      return Math.max(1000, 156 + (this.patchNodes.length - 1) * 140);
    },
    patchChain() {  // the modules the picture goes through, in order: follow the cables from PICTURE's OUT
      const out = [];
      let at = "src";
      const seen = new Set([at]);
      for (;;) {
        const c = this.patch.cables.find((x) => x.from === at);
        if (!c || c.to === "result" || seen.has(c.to)) return { steps: out, done: !!c && c.to === "result" };
        out.push(c.to);
        seen.add(c.to);
        at = c.to;
      }
    },
    patchLoose() {  // modules with a cable that the picture never reaches (patched, but not from PICTURE)
      const reached = new Set(["src", ...this.patchChain.steps]);
      return this.patch.cables.filter((c) => !reached.has(c.from)).map((c) => c.from);
    },
    toolPct() {
      const j = this.toolJob;
      if (!j || j.state !== "running") return 0;
      const p = this.progress;
      if (p && p.job === j.id && p.steps) return Math.min(99, Math.round((p.step / p.steps) * 100));
      const e = this.jobEtas.find((x) => x.id === j.id);
      if (!e) return 0;
      const done = (this.now - this.tool.working.started) / 1000;
      return Math.min(97, Math.round((done / (done + e.remaining)) * 100));
    },
    i2iModelGroups() {
      const ok = (m) => (m.family === "sd15" || m.family === "sdxl") && m.engine !== "mlx";
      const mark = (list) => list.map((m) => ({ ...m, i2iOk: ok(m) }));
      return [
        { name: "SD 1.5 · image to image, inpaint, ControlNet (built-in + yours), image prompt (IP-Adapter)", models: mark(this.sortModels(this.installedModels.filter((m) => m.family === "sd15" && !m.inpaint))) },
        { name: "SDXL / Pony · image to image, inpaint, image prompt (IP-Adapter), your SDXL ControlNets", models: mark(this.sortModels(this.installedModels.filter((m) => m.family === "sdxl" && !m.inpaint))) },
      ].filter((g) => g.models.length);
    },
    convShown() {
      const q = this.conv.search.trim().toLowerCase();
      return (this.conv.items || []).filter((i) => (!this.conv.kind || i.kind === this.conv.kind) && (!q || i.name.toLowerCase().includes(q)));
    },
    convSelectedSize() {
      const set = new Set(this.conv.selected);
      const mb = (this.conv.items || []).filter((i) => set.has(i.file)).reduce((a, i) => a + (i.size_mb || 0), 0);
      return mb > 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`;
    },
    mergeJob() {
      return this.queue.find((j) => j.cmd === "ckpt_merge");
    },
    mergeModels() {  // your single-file .safetensors SD 1.5 / SDXL models
      return this.models.filter((m) => m.path && /\.safetensors$/i.test(m.path) && ["sd15", "sdxl"].includes(m.family) && !m.inpaint);
    },
    mergeFamily() {
      return (this.mergeModels.find((m) => m.path === this.mrg.a) || {}).family || "";
    },
    cvtJob() {
      return this.queue.find((j) => j.cmd === "lora_extract");
    },
    ckJob() {
      return this.queue.find((j) => j.cmd === "ckpt_convert");
    },
    convJob() {
      return this.queue.find((j) => j.cmd === "convert_tdict");
    },
    controlTimingText() {
      const { start, end } = this.form.control;
      if (start === 0 && end === 1) return "Now: it guides every step, so the result keeps the map's structure closely.";
      if (start === 0) return `Now: it guides the first ${Math.round(end * 100)}% (the layout) and then lets go, so details are freer.`;
      if (end === 1) return `Now: it starts after ${Math.round(start * 100)}%, so the overall layout is free but the finer structure follows the map.`;
      return `Now: it guides only between ${Math.round(start * 100)}% and ${Math.round(end * 100)}% of the steps.`;
    },
    recommendedVae() {
      const r = { sd15: { name: "vae-ft-mse-840000-ema-pruned", label: "vae-ft-mse (SD 1.5)", mb: 335 },
                  sdxl: { name: "sdxl-vae-fp16-fix", label: "SDXL fp16-fix", mb: 335 } }[this.family];
      return r && !this.vaes.some((v) => v.name === r.name) ? r : null;
    },
    controlMapKind() {  // the map the picked ControlNet needs ("union:depth" → "depth")
      const t = this.form.control.type || "";
      return t.startsWith("union:") ? t.slice(6) : t;
    },
    controlTypesShown() {  // the ControlNets for the current model's type (SD 1.5's built-in ones, your own SD 1.5 or SDXL)
      const fam = this.family === "sdxl" ? "sdxl" : "sd15";
      return this.controlTypes.filter((t) => (t.family || "sd15") === fam || t.id === this.form.control.type);
    },
    cnGuide() {  // the guide for the ControlNet picked in Image to image
      const t = this.controlTypeInfo;
      if (!t) return null;
      if (CN_GUIDES[t.id]) return CN_GUIDES[t.id];
      if (t.union && CN_GUIDES[t.makes_map]) {  // Union ProMax (SDXL): the same advice, at SDXL sizes
        const g = CN_GUIDES[t.makes_map];
        return { ...g, title: "Union ProMax · " + g.title, models: null,
                 modelsText: "Any SDXL model (also Pony, Illustrious, NoobAI). One ~2.5 GB download covers every Union map type.",
                 set: g.set ? { ...g.set, size: 1024 } : g.set };
      }
      const name = t.id.replace(/^user:/, "");
      const hit = CN_NAME_GUIDES.find(([re]) => re.test(name));
      if (hit) return CN_GUIDES[hit[1]];
      if (t.makes_map && CN_GUIDES[t.makes_map]) return CN_GUIDES[t.makes_map];
      return { title: "Your ControlNet", what: "A ControlNet you added. It follows your picture as it is (no map is made from it), so give it the kind of picture it was trained on: its Civitai page says which.",
               picture: "The picture it expects (a pattern, a map, a sketch…).", models: null, modelsText: "Models of the same type as the ControlNet.",
               steps: ["Put the picture in 1 · Picture.", "Start with strength 1.0 and Mostly, then adjust."], set: { scale: 1, start: 0, end: 0.8 } };
    },
    cnRecSize() {  // the guide's picture size on this Mac (512 under 16 GB: a ControlNet doubles each step's work)
      const c = this.cnGuide && this.cnGuide.set;
      if (!c || !c.size) return 0;
      const xl = (this.controlTypeInfo || {}).family === "sdxl";
      return (this.sys.ram_gb || 8) < 16 ? Math.min(c.size, xl ? 768 : 512) : c.size;
    },
    cnRecText() {  // the guide's settings in a few words, shown under the ControlNet picker
      const c = this.cnGuide && this.cnGuide.set;
      if (!c) return "";
      const timing = (this.controlTimings.find((p) => p.start === c.start && p.end === c.end) || {}).label
        || `${Math.round(c.start * 100)}–${Math.round(c.end * 100)}%`;
      return [`strength ${c.scale}`, `guides: ${timing}`, c.steps ? `${c.steps} steps` : "", c.cfg ? `guidance ${c.cfg}` : "",
              this.cnRecSize ? `${this.cnRecSize} × ${this.cnRecSize}` : "",
              c.use_init === false ? "Start from this picture: off" : ""].filter(Boolean).join(" · ");
    },
    cnGuideOff() {  // settings that will spoil this ControlNet's result (shown as warnings with a one-click fix)
      const c = this.cnGuide && this.cnGuide.set, f = this.form;
      if (!c || !f.control.type) return [];
      const out = [];
      if (c.use_init === false && f.i2i_use_init)
        out.push("<b>Start from this picture</b> is on: this ControlNet works best without it, so switch it off");
      if (c.scale >= 1.1 && f.control.scale < c.scale - 0.25)
        out.push(`strength ${f.control.scale.toFixed(2)} is too weak for this one (try ${c.scale}): it will barely follow your picture`);
      if (c.steps && f.steps < c.steps - 10) out.push(`${f.steps} steps is too few (try ${c.steps})`);
      return out;
    },
    cnModelPicks() {  // installed models that suit the picked ControlNet (best first), for one-click switching
      const t = this.controlTypeInfo, g = this.cnGuide;
      if (!t || !g) return [];
      const fam = t.family || "sd15";
      const ok = this.installedModels.filter((m) => m.family === fam && m.engine !== "mlx");
      const good = g.models ? ok.filter((m) => g.models.test(m.label || m.id)) : [];
      return (good.length ? good : ok).slice(0, 4).map((m) => ({ ...m, matched: good.includes(m) }));
    },
    controlTypeInfo() {
      return this.controlTypes.find((t) => t.id === this.form.control.type);
    },
    compatibleVaes() {
      return this.vaes.filter((v) => v.family === this.family);
    },
    extrasFolder() {
      return this.extras.tab === "vae" ? this.vaeFolder : this.extras.tab === "cn" ? this.controlnetFolder : (this.embDirs?.default || "");
    },
    extrasShown() {
      const q = this.extras.q.trim().toLowerCase();
      const list = this.extras.tab === "vae" ? this.vaes : this.extras.tab === "cn" ? this.userControlnets : this.embeddings;
      return (list || []).filter((x) => (!q || (x.name || "").toLowerCase().includes(q))
        && (this.extras.tab === "cn" || !this.extras.family || x.family === this.extras.family)
        && !(this.hideNsfw && x.nsfw_auto)).slice(0, 400);
    },
    createSession() {  // Create's pictures: Image to image's results stay in Image to image (and the Library)
      return this.session.filter((r) => !I2I_KINDS.includes(r.kind));
    },
    expandAdvice() {  // ↔ Expand: a warning when the request is the kind that tends to give seams or repeats
      const o = this.form.outpaint, d = this.initDims;
      if (this.form.i2i_mode !== "outpaint" || !d) return null;
      const L = o.left || 0, R = o.right || 0, T = o.top || 0, B = o.bottom || 0;
      if (!(L || R || T || B)) return null;
      const added = (d.w + L + R) * (d.h + T + B) / (d.w * d.h) - 1;
      const sides = [L, R, T, B].filter(Boolean).length;
      const bigSide = Math.max(L / d.w, R / d.w, T / d.h, B / d.h);
      if (sides >= 3 || (L || R) && (T || B)) {
        return { level: "warn", text: "Expanding in both directions at once (wider and taller) is the hardest case: expect seams, repeated objects or smeared patches, especially in the corners. For the cleanest result, make it wider first, press “Use the result as the new picture”, then make it taller." };
      }
      if (bigSide > 0.5 || added > 0.8) {
        return { level: "warn", text: `That adds ${Math.round(added * 100)}% more picture. Big additions often repeat things from the picture or leave blurry areas; add up to about a third of the picture per side, then expand again from the result.` };
      }
      return { level: "ok", text: "One direction at a time, a moderate amount: this gives the cleanest results." };
    },
    i2iUsesKlein() {  // Image to image runs on FLUX.2 Klein: Edit with words, or Expand with the outpaint LoRA
      return this.form.i2i_mode === "edit" || (this.form.i2i_mode === "outpaint" && this.form.outpaint_engine !== "model");
    },
    i2iResults() {
      return this.session.filter((r) => I2I_KINDS.includes(r.kind));
    },
    i2iLatest() {
      return this.i2iResults[0] || null;
    },
    canSkipImage() {
      const j = this.runningJob, p = this.progress;
      if (!j || j.cmd !== "generate" || (j.images || 1) < 2) return false;
      return !p || p.job !== j.id || p.image < p.images; // not on the last image
    },
    localModels() {
      return this.models.filter((m) => m.id.startsWith("local:")).map((m) => ({ ...m, file: m.id.slice(6) }));
    },
    addedModels() {
      return this.models.filter((m) => m.id.startsWith("hf:"));
    },
    sizeTiers() {
      // Small / Medium / Large for each model type (side of an equivalent square).
      const t = { sd15: [512, 640, 768], sdxl: [768, 1024, 1280], anima: [768, 1024, 1280], ernie: [640, 768, 1024], flux: [512, 768, 1024], flux2: [512, 768, 1024], other: [512, 768, 1024] };
      const [s, m, l] = t[this.family] || t.other;
      if (this.family === "sd15")  // SD 1.5 also gets 704: a little more detail than 640, still quick on 8 GB
        return [{ id: "s", label: "Small", side: 512 }, { id: "m", label: "Medium", side: 640 }, { id: "m2", label: "Med+", side: 704 }, { id: "l", label: "Large", side: 768 }];
      return [{ id: "s", label: "Small", side: s }, { id: "m", label: "Medium", side: m }, { id: "l", label: "Large", side: l }];
    },
    currentTier() {
      const side = Math.sqrt(this.form.width * this.form.height);
      const hit = this.sizeTiers.find((t) => Math.abs(side - t.side) / t.side < 0.06);
      return hit ? hit.id : null;
    },
    sizePresets() {
      // Shapes each model family was trained around (all multiples of 8).
      const sets = {
        sd15: [["Square 1:1", 512, 512], ["Portrait 2:3", 512, 768], ["Landscape 3:2", 768, 512],
               ["Portrait 3:4", 576, 768], ["Landscape 4:3", 768, 576], ["Tall 9:16", 432, 768],
               ["Wide 16:9", 768, 432], ["Square 704", 704, 704], ["Large square", 768, 768],
               ["Tall 704 × 1024", 704, 1024], ["Tall 768 × 1024", 768, 1024], ["Wide 1024 × 704", 1024, 704]],
        sdxl: [["Square 1:1", 1024, 1024], ["Portrait 2:3", 832, 1216], ["Landscape 3:2", 1216, 832],
               ["Portrait 3:4", 896, 1152], ["Landscape 4:3", 1152, 896], ["Tall 9:16", 768, 1344],
               ["Wide 16:9", 1344, 768], ["Small square", 768, 768]],
      };
      sets.ernie = [["Square 1:1", 1024, 1024], ["Portrait 2:3", 848, 1264], ["Landscape 3:2", 1264, 848],
                    ["Portrait 3:4", 896, 1200], ["Landscape 4:3", 1200, 896], ["Tall 9:16", 768, 1376],
                    ["Wide 16:9", 1376, 768], ["Medium square", 768, 768]];
      // FLUX: the SDXL shapes (all multiples of 16 too)
      sets.flux = sets.sdxl.map(([l, w, h]) => [l === "Small square" ? "Medium square" : l, w, h]);
      sets.flux2 = sets.flux;
      sets.anima = sets.sdxl;  // Anima: the SDXL shapes (multiples of 16)
      return (sets[this.family] || sets.sdxl).map(([label, w, h]) => ({ label, w, h }));
    },
    livePreviewShown() {
      const lp = this.livePreview;
      // Create tab: only for jobs started there (Image to image jobs preview in their own tab)
      return !!(lp && this.opts.previews && this.runningJob && this.runningJob.id === lp.job && !this.runningJob.i2i);
    },
    hasWildcards() {
      const t = (this.form.prompt || "") + " " + (this.form.negative_prompt || "");
      return /\{[^{}]*\|[^{}]*\}|__[\w\-./ ]+?__/.test(t);
    },
    randomPool() {
      // LoRAs the randomizer may pick: ones that work with this model (and not already added by hand)
      return this.compatibleLoras;
    },
    hiresTarget() {
      const ram = this.sys.ram_gb || 8;
      const cap = this.family === "sd15" ? (ram < 24 ? 1024 : 1536) : (ram < 24 ? 1536 : 2048);
      let w = this.form.width * this.form.hires.scale, h = this.form.height * this.form.hires.scale;
      if (Math.max(w, h) > cap) { const f = cap / Math.max(w, h); w *= f; h *= f; }
      w = Math.floor(w / 8) * 8; h = Math.floor(h / 8) * 8;
      return `${this.form.width}×${this.form.height} → ${w}×${h}`;
    },
    batchLines() {
      return (this.batch.text || "").split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
    },
    cmpCount() {
      const n = (axis, v) => {
        if (!axis) return 1;
        const t = String(v || "").trim();
        const m = t.match(/^([\d.]+)\s*-\s*([\d.]+)\s*\(\s*\+?\s*([\d.]+)\s*\)$/);
        if (m && !["sampler", "lora"].includes(axis)) {
          const a = +m[1], b = +m[2], st = +m[3] || 1;
          return Math.max(0, Math.floor((b - a) / st + 1e-9) + 1);
        }
        return t.split(",").filter((x) => x.trim()).length;
      };
      return n(this.cmp.x, this.cmp.xv) * (this.cmp.y ? n(this.cmp.y, this.cmp.yv) : 1);
    },
    outpaintSize() {
      const o = this.form.outpaint, d = this.initDims;
      if (!d) return "";
      return `${d.w + (o.left || 0) + (o.right || 0)}×${d.h + (o.top || 0) + (o.bottom || 0)} (from ${d.w}×${d.h})`;
    },
    i2iRunning() {
      // the running job is an Image to image one (started from that tab)
      const j = this.runningJob;
      return !!(j && j.i2i && ["generate", "edit_words", "heal"].includes(j.cmd));
    },
    recSettings() {
      // ★ Recommended for the picked sampler: ⚡ fastest ones at the low end of their step range, ★ best-quality ones
      // further up (where they shine), others at their usual count; guidance is the model's own
      const r = this.recSteps, m = this.currentModel;
      if (!r || !m) return null;
      const picks = this.samplerPicks ? this.samplerPicks() : { fast: [], quality: [] };
      const id = this.form.sampler, s = this.samplers.find((x) => x.id === id);
      const kind = picks.fast.some((x) => x.id === id) ? "fast" : (picks.quality || []).some((x) => x.id === id) ? "best" : "";
      const steps = kind === "fast" ? r[1] : kind === "best" ? Math.round((r[0] + r[2]) / 2) : r[0];
      const cfg = m.defaults?.cfg ?? this.form.cfg;
      return { steps, cfg, kind, label: s ? s.label : "this sampler" };
    },
    recSteps() {
      // Recommended step count for the chosen sampler: [recommended, low, high].
      const s = this.samplers.find((x) => x.id === this.form.sampler);
      if (s?.steps) return s.steps;
      const d = this.currentModel?.defaults?.steps; // ERNIE (flow samplers): the model's own default
      return d ? [d, Math.max(1, Math.round(d * 0.75)), Math.round(d * 1.5)] : null;
    },
    aspectText() {
      const w = this.form.width, h = this.form.height;
      if (!w || !h) return "";
      const gcd = (a, b) => (b ? gcd(b, a % b) : a);
      const g = gcd(w, h);
      const [a, b] = [w / g, h / g];
      const named = [[1, 1], [4, 3], [3, 2], [16, 9], [21, 9], [5, 4], [7, 5], [2, 1]];
      const r = w / h;
      for (const [x, y] of named) {
        if (Math.abs(r - x / y) < 0.02) return `${x}:${y}`;
        if (Math.abs(r - y / x) < 0.02) return `${y}:${x}`;
      }
      return a <= 32 && b <= 32 ? `${a}:${b}` : (w >= h ? `${r.toFixed(2)}:1` : `1:${(1 / r).toFixed(2)}`);
    },
    brushWellPx() {  // the size well beside the slider: in step with the slider over its whole range (4–160)
      return Math.round(4 + (Math.max(4, Math.min(160, this.mask.size)) - 4) * 40 / 156);
    },
    progressPct() {
      const p = this.progress;
      if (!p || !this.runningJob || p.job !== this.runningJob.id) return 0;
      return Math.round((((p.image - 1) + p.step / p.steps) / p.images) * 100);
    },
    progressText() {
      const p = this.progress;
      if (!p) return this.status;
      let text = `Image ${p.image}/${p.images} · step ${p.step}/${p.steps}`;
      const left = this.genLeft(p);
      if (left) text += ` · ${left.perStep.toFixed(1)} s/step · ${this.formatDuration(left.seconds)} left`;
      return text;
    },
    jobEtas() {
      // Estimated seconds for each queued job, plus when it should finish (cumulative).
      let clock = 0;
      let loaded = this.lastModel;
      return this.queue.map((j) => {
        let est, measured = true;
        const elapsed = j.state === "running" && this.jobStartedAt ? (this.clockNow() - this.jobStartedAt) / 1000 : 0;
        if (j.cmd === "generate") {
          const fam = this.models.find((m) => m.id === j.model)?.family || "sdxl";
          const mp = ((j.width || 1024) * (j.height || 1024)) / 1e6;
          const p = this.perf.models[j.model] || {};
          if (!p.sps) measured = false;
          const sps = (p.sps || DEFAULT_SPS_PER_MP[fam]) * mp;
          const steps = Math.max(1, Math.ceil((j.steps || 25) * (j.strength || 1)));
          est = sps * (steps + 2) * (j.images || 1); // +2 steps' worth for decoding/saving each image
          if (j.model !== loaded) est += p.load ?? DEFAULT_LOAD_S[fam];
          if (j.enhance) est += this.perf.jobs.enhance_prompt ?? DEFAULT_JOB_S.enhance_prompt;
          loaded = j.model;
          const left = j.state === "running" && this.progress && this.progress.job === j.id && this.genLeft(this.progress);
          if (left) est = elapsed + left.seconds;  // the same figure the progress line shows
        } else if (j.cmd === "train_lora" || j.cmd === "train_embedding") {
          const sps = this.trainSps(this.models.find((x) => x.id === j.model)?.family || "sd15");
          const pr = this.progress;
          est = (j.steps || 1000) * sps + 60;
          if (j.state === "running" && pr && pr.job === j.id) est = elapsed + (pr.steps - pr.step) * sps;
          measured = !!(this.train.spsBy || {})[this.models.find((x) => x.id === j.model)?.family];
        } else if (j.cmd === "model_download") {
          const gb = this.models.find((m) => m.id === j.model)?.download_gb || 5;
          est = gb * 1000 / 25; // ~25 MB/s until it's measured
          const pr = this.progress;
          if (j.state === "running" && pr && pr.job === j.id && pr.step > 20 && elapsed > 5) {
            est = elapsed * pr.steps / pr.step;
          }
          measured = false;
        } else {
          const key = j.cmd === "postprocess" ? j.action : j.cmd;
          const one = (k) => this.perf.jobs[k] ?? DEFAULT_JOB_S[k] ?? 30;
          est = key === "both" ? one("enhance") + one("faces") : one(key);
          if ((key === "enhance" || key === "both") && j.upscale) est *= (j.upscale / 2) ** 2;
          measured = this.perf.jobs[key] != null;
        }
        const remaining = Math.max(1, est - elapsed);
        clock += remaining;
        return { id: j.id, remaining, finishIn: clock, measured };
      });
    },
    elapsedText() {
      if (!this.runningJob || !this.jobStartedAt) return "";
      const s = Math.round((this.clockNow() - this.jobStartedAt) / 1000);
      return s >= 2 ? ` (${this.formatDuration(s)})` : "";
    },
    statusText() {
      return this.backendDead ? "Backend stopped — see Settings" : this.status;
    },
    // Rack skin: the VU needle reads exactly what the progress bar for the running job shows
    // (Tools and Train have their own bars; everything else uses the queue bar). The amber readouts show time left.
    vuPct() {
      const j = this.runningJob;
      if (!j || this.backendDead) return 0;
      let pct;
      if (this.toolJob && this.toolJob.id === j.id) pct = this.toolPct;
      else if (this.trainJob && this.trainJob.id === j.id) pct = this.trainPct;
      else pct = this.jobProgress(j);
      return Number.isFinite(pct) ? Math.max(0, Math.min(100, pct)) : 0;
    },
    rackTimeLeft() {
      const j = this.runningJob;
      const e = j && this.jobEtas.find((x) => x.id === j.id);
      return e ? this.clockText(e.remaining) : "--:--";
    },
    rackQueueLeft() {
      const last = this.jobEtas[this.jobEtas.length - 1];
      return last && this.queue.length > 1 ? this.clockText(last.finishIn) : "--:--";
    },
  },

  watch: {
    loraOpen(v) { saveJSON("protoart.loraOpen", !!v); },
    loraKind(v) {  // "🏷 Manage tags…" in the LoRA list's filter opens the tags panel
      if (v === "__tags") { this.loraKind = ""; this.openModelsPanel("tags"); }
    },
    // picking a ControlNet in Image to image puts in the settings it works best with (strength, timing,
    // steps… which are easy to miss); "undo" puts yours back
    "form.control.type"(v, old) {
      if (!v || v === old || this.tab !== "i2i") return;
      this.$nextTick(() => {
        if (this.form.control.type === v && this.cnGuide && this.cnGuide.set) this.applyCnGuide(true);
      });
    },
    // Create and Image to image each keep their own prompt and negative prompt: the prompt box shows the one for
    // the tab you're in (synchronous, so code that switches tab and then sets a prompt writes into the right one)
    tab: { flush: "sync", handler(v, old) {
      this.swapPrompts(v === "i2i" ? "i2i" : "create");
      if (v === "settings") { this.loadFolders(false); this.$nextTick(() => this.applySettingsFolds()); }
      // leaving Tools drops Adjust's GPU preview; coming back reloads it with the same settings
      if (old === "tools" && this.adj.open && adjEngine) { adjEngine.destroy(); adjEngine = null; }
      if (v === "tools" && this.adj.open && !adjEngine && this.adj.rec && !this.adj.busy)
        this.$nextTick(() => this.openAdjust(this.adj.rec, true));
    } },
    form: { deep: true, handler(v) {
      saveJSON(FORM_KEY, v);
      // ↺ Reset to defaults' undo link lasts until you change something after it
      if (this.resetUndo && Date.now() - (this._resetAt || 0) > 2500) this.resetUndo = null;
    } },
    "adj.p": { deep: true, handler() { this.adjRender(); } },
    "adj.before"() { this.adjRender(); },
    "adj.view"() { this.adjRender(); },
    "adj.split"() { this.adjRender(); },
    i2p: { deep: true, handler(v) { saveJSON("beestudio.i2p", v); } },
    rnd: { deep: true, handler(v) { saveJSON("beestudio.rnd", v); } },
    loraSort(v) { saveJSON("protoart.loraSort", v); },
    loraRand: { deep: true, handler(v) { saveJSON("protoart.loraRand", v); } },
    "form.init_image": {
      immediate: true,
      handler(v) {  // the picture's real size, for Expand's sizes and presets (also when the paint area isn't shown)
        this.initDims = null;
        if (!v) return;
        const im = new Image();
        im.onload = () => { if (this.form.init_image === v && im.naturalWidth) this.initDims = { w: im.naturalWidth, h: im.naturalHeight }; };
        im.src = this.fileUrl(v);
      },
    },
    "tool.source"() { this.tool.healing = false; },  // a new picture in Tools: stop painting on the old one
    train: { deep: true, handler(v) {
      const { items, result, lossHistory, dirty, ...keep } = v;
      saveJSON("protoart.train", keep);
    } },
  },

  async mounted() {
    // the saved form remembers whose prompt is in the box (Create's or Image to image's): match it to the tab the
    // app opens on, or Image to image's prompt would show (and be written) in Create
    this.swapPrompts(this.tab === "i2i" ? "i2i" : "create");
    window.ProtoTips?.setEnabled(this.opts.tooltips !== false);
    window.addEventListener("keydown", (e) => this.adjKey(e));  // Adjust: \ flips Edited / Original
    window.addEventListener("keydown", (e) => this.selKey(e));
    window.addEventListener("keydown", (e) => { if (e.key === "Escape" && this.ctx) this.ctx = null; });  // right-click menu
    window.addEventListener("keydown", (e) => {  // a picture's details: ← → step through the pictures, Esc closes
      if (!this.detail || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName || "")) return;
      if (e.key === "ArrowLeft") { this.detailStep(-1); e.preventDefault(); }
      else if (e.key === "ArrowRight") { this.detailStep(1); e.preventDefault(); }
      else if (e.key === "Escape" && !this.ask && !this.confirmDel) { this.detail = null; e.preventDefault(); }
    });  // painting: tool keys, Esc cancels a selection
    setInterval(() => {
      if (this.queue.length || Object.keys(this.stopping).length) this.now = Date.now();
      this.stopForGood();
    }, 500);
    listeners.push(this.onEvent);
    this.swapPrompts(this.tab === "i2i" ? "i2i" : "create");  // a window reloaded while in Image to image
    window.bee.onLog((s) => {
      // home folders (your username) are shown as ~, like the saved log
      this.logs = (this.logs + String(s).replace(/\/Users\/[^/\s'"]+/g, "~")).slice(-20000);
    });
    window.bee.onExit((info) => this.onEngineExit(info));
    // The backend's first "ready" message can arrive before this page has loaded,
    // so initialise now as well (requests are queued in the backend's stdin).
    this.init();
  },

  methods: {
    swapPrompts(to) {
      const f = this.form;
      const owner = f.prompt_owner === "i2i" ? "i2i" : "create";
      if (to === owner) return;
      if (to === "i2i") {
        f.create_prompt = f.prompt; f.create_negative = f.negative_prompt;
        f.prompt = f.i2i_prompt || ""; f.negative_prompt = f.i2i_negative || "";
      } else {
        f.i2i_prompt = f.prompt; f.i2i_negative = f.negative_prompt;
        f.prompt = f.create_prompt || ""; f.negative_prompt = f.create_negative || "";
      }
      f.prompt_owner = to;
      this.lastEnhanced = null;  // an enhanced prompt belongs to the tab it was made in
    },
    // -------------------------------------------------------------- startup
    onEvent(m) {
      if (m.type === "ready") { this.setPaused(false); return this.init(); }
      if (m.type === "status") this.status = m.msg;
      if (m.type === "progress") this.onProgress(m);
      if (m.type === "enhanced_prompt") this.lastEnhanced = { prompt: m.prompt, original: m.original };
      if (m.type === "model_ready") this.readyModel = m.model;
      if (m.type === "queue") {
        if (typeof m.paused === "boolean") this.setPaused(m.paused);
        const prev = this.runningJob;
        // background preloads aren't shown as jobs (no meter, no clocks): just a "getting ready" note
        this.preloading = (m.jobs.find((x) => x.cmd === "preload") || {}).model || "";
        m = { ...m, jobs: m.jobs.filter((x) => x.cmd !== "preload") };
        this.queue = m.jobs;
        const nowRunning = this.runningJob?.id;
        if (nowRunning !== prev?.id) {
          if (this.runningJob && this.runningJob.model !== this.readyModel) this.readyModel = ""; // another job may unload it
          if (prev) this.recordPerf(prev);
          this.jobStartedAt = nowRunning ? Date.now() : 0;
          this.firstStepAt = 0;
        }
        if (!m.jobs.length) {
          this.progress = null;
          this.livePreview = null;
        }
        const ids = new Set(m.jobs.map((x) => x.id));
        for (const id of Object.keys(this.stopping)) if (!ids.has(Number(id))) delete this.stopping[id];
        if (Object.keys(this.sentBack).length) this.resendSentBack();
      }
      if (m.type === "skipped") this.livePreview = null;
      if (m.type === "memory_low") {
        this.memLow = { free: m.free_gb, swap: m.swap_gb, sdxl: m.family === "sdxl" };
        this.memLowShown = true;
      }
      if (m.type === "train_status") {
        this.train.secPerStep = m.sec_per_step;
        const tj = this.trainJob;
        const fam = (tj && this.models.find((x) => x.id === tj.model)?.family) || this.trainFamily;
        const key = tj && tj.cmd === "train_embedding" ? "sd15_ti" : fam;
        this.train.spsBy = { ...(this.train.spsBy || {}), [key]: m.sec_per_step };
        if (Number.isFinite(m.loss)) this.train.lossHistory.push(m.loss); // the fast engine reports speed only
        if (this.train.lossHistory.length > 300) this.train.lossHistory.splice(0, this.train.lossHistory.length - 300);
      }
      if (m.type === "download_needed") this.dlJob = m.job;
      if (m.type === "preview") this.livePreview = { job: m.job, image: m.image, src: m.src };
      if (m.type === "image") {
        // learn how long a picture takes after its last step (decoding + saving) for this model's time-left figure
        const run = this.runningJob;
        if (run && run.cmd === "generate" && run.id === m.job && this.timing.job === m.job && this.timing.lastAt) {
          const tail = (Date.now() - this.timing.lastAt) / 1000;
          if (tail > 0 && tail < 600) {
            const pm = (this.perf.models[run.model] ||= {});
            pm.tail = pm.tail == null ? tail : pm.tail * 0.6 + tail * 0.4;
            saveJSON(PERF_KEY, this.perf);
          }
        }
        // Tools results (enhance, face restore, depth) stay in Tools and the Library, not in Create's pictures
        // (a Remove from Tools is tagged from_tools: Image to image's 🩹 Remove makes the same kind of picture)
        const fromJob = this.queue.find((j) => j.id === m.job);
        const sent = jobArgs.get(m.job);
        const toolKinds = ["enhanced", "faces", "depth", "edited", "vector"];
        const fromTools = sent ? (sent.cmd === "postprocess" || sent.cmd === "vectorize" || !!sent.args.from_tools)
          : fromJob ? fromJob.cmd === "postprocess" : toolKinds.includes(m.record.kind);
        if (!fromTools) this.session.unshift(m.record);
        if (m.record.model && m.record.kind === "generated") this.readyModel = m.record.model;
        if (m.record.nsfw && this.hideNsfw) this.status = "A new picture is covered by the NSFW filter (click it to look)";
        this.livePreview = null;
      }
    },
    async init() {
      this.backendDead = false;
      try {
        await call("set_settings", this.opts);
        // Reading a folder on an external drive waits until you answer macOS's "removable volume" question, which
        // can hide behind other windows: say so instead of looking empty.
        const slow = setTimeout(() => {
          this.status = "Still reading your folders… If macOS asks to let Protoart Studio access a removable volume (it may be behind this window), click Allow.";
        }, 8000);
        try {
          [this.sys, this.models, this.presets, this.imagesDir, this.queue] = await Promise.all([
            call("system_info"), call("list_models"), call("presets_list"), call("images_dir_get"), call("queue_get"),
          ]);
        } finally {
          clearTimeout(slow);
        }
        this.queue = this.queue.filter((x) => x.cmd !== "preload");
        this.loadContentStatus();
        call("privacy_get").then((r) => { this.strictOffline = !!(r && r.strict); }).catch(() => {});
        await this.refreshLoras();
        call("diffusionbee_info").then((d) => { this.dbee = d; }).catch(() => {});
        call("list_embeddings").then((e) => { this.embeddings = e; }).catch(() => {});
        this.refreshExtras(false);
        call("embedding_dirs_get").then((d) => { this.embDirs = d; }).catch(() => {});
        call("adjust_presets_get").then((d) => { this.adjPresets = Array.isArray(d) ? d : []; }).catch(() => {});
        call("lora_dirs_get").then((d) => { this.loraDirs = d; }).catch(() => {});
        call("model_dirs_get").then((d) => { this.modelDirs = d; }).catch(() => {});
        call("main_folder_get").then((d) => { this.mainFolder = d; }).catch(() => {});
        // Model libraries other apps keep (Stability Matrix, ComfyUI…): offer them once, read in place.
        call("libraries_get").then((r) => { this.libs = r; if (r.offer?.length) this.libOffer = r.offer.map((f) => ({ ...f, pick: true })); }).catch(() => {});
        call("civitai_key_get").then((d) => { this.civ.keySet = !!(d && d.set); }).catch(() => {});
        call("hf_token_get").then((d) => { this.hub.tokenSet = !!(d && d.set); }).catch(() => {});
        call("export_presets_get").then((d) => { this.exportPresets = Array.isArray(d) ? d : []; }).catch(() => {});
        call("default_export_dir").then((d) => { if (d && d.path && !this.bx.dest) this.bx.dest = loadJSON("protoart.exportDir", d.path); }).catch(() => {});
        this.loadDownloadsDir();
        this.refreshEdits();
        for (const l of this.form.loras) this.fetchTriggers(l.name);
        // the default model applies once per app launch, not again when the window reloads or the engine restarts
        // (that switched models behind your back and dropped the LoRAs you'd picked)
        let launched = false;
        try { launched = sessionStorage.getItem("protoart.launched") === "1"; sessionStorage.setItem("protoart.launched", "1"); } catch { /* ignore */ }
        if (launched) this.defaultApplied = true;
        if (!this.defaultApplied && this.opts.default_model && this.opts.default_model !== this.form.model &&
            this.models.some((m) => m.id === this.opts.default_model)) {
          this.form.model = this.opts.default_model; // your default model, every time the app opens
          await this.onModelChange();
        }
        this.defaultApplied = true;
        if (!this.currentModel) {
          // saved model no longer exists: pick one that fits and apply ITS defaults (size, steps, CFG)
          this.form.model = (this.models.find((m) => m.enough_ram) || this.models[0]).id;
          await this.onModelChange();
        } else if (this.currentModel.family === "sd15" && Math.max(this.form.width, this.form.height) > 768) {
          const d = this.currentModel.defaults;
          Object.assign(this.form, { width: d.width, height: d.height });
          this.status = "Reset size to 512×512 (SD 1.5 is made for 512; 1024 is many times slower)";
        }
        if (this.currentModel) this.form.clip_skip = this.clipSkipFor(this.currentModel); // model's default or your choice
        await this.loadSamplers(false);
        this.status = "Ready";
        hideSplash();
        await this.loadFolders(true); // warn if one of your folders has gone (moved, or drive not connected)
        // Speed: load a model now, while you're still writing the prompt
        const start = this.opts.preload_start === "last" ? this.form.model : this.opts.preload_start;
        if (start && !this.requeue.length) setTimeout(() => this.preloadModel(start), 1500);
        if (this.requeue.length) {
          const jobs = this.requeue;
          this.requeue = [];
          for (const jb of jobs) this.runJob(jb.cmd, jb.args); // results arrive as image events
          this.status = `Engine restarted · re-queued ${jobs.length} job${jobs.length === 1 ? "" : "s"}`;
        }
      } catch (e) {
        this.error = e.message;
        hideSplash();
      }
    },
    async refreshLoras() {
      this.loras = await call("list_loras");
    },
    // -------------------------------------------------------------- size / steps
    snap(v) {
      const m = ["ernie", "flux", "flux2"].includes(this.family) ? 16 : 8;
      return Math.min(2048, Math.max(256, Math.round(v / m) * m));
    },
    setSide(side, raw, final) {
      // While typing (final=false) keep the raw number so the field doesn't jump; snap when done.
      const v = Number(raw);
      if (!v || v < 1) return;
      const val = final ? this.snap(v) : v;
      this.form[side] = val;
      if (this.form.size_link && this.form.size_ratio > 0) {
        const other = side === "width" ? "height" : "width";
        const o = side === "width" ? val / this.form.size_ratio : val * this.form.size_ratio;
        if (final || v >= 256) this.form[other] = this.snap(o);
      }
      if (final) this.$forceUpdate(); // show the snapped value in the field
    },
    applyTier(t) {
      // keep the current shape, scale to the tier's pixel count
      const r = this.form.width / this.form.height || 1;
      const area = t.side * t.side;
      this.form.width = this.snap(Math.sqrt(area * r));
      this.form.height = this.snap(Math.sqrt(area / r));
      if (this.form.size_link) this.form.size_ratio = this.form.width / this.form.height;
    },
    isAddedLocal(id) {
      return (this.modelDirs.local || []).some((m) => m.id === id);
    },
    openModelSettings(which) {
      // adding models (Civitai / Hugging Face / your files) lives in the Models tab now
      this.openModelsPanel(which === "hub" ? "hub" : which === "civitai" ? "civitai" : "own");
    },
    openModelsPanel(which) {
      this.mpanels[which] = true;
      this.tab = "models";
      this.$nextTick(() => {
        const el = document.getElementById("mp-" + which);
        el?.scrollIntoView({ behavior: "smooth", block: "start" });
        if (which === "hub") setTimeout(() => this.$refs.hubInput?.focus(), 350);
        if (which === "civitai") setTimeout(() => this.$refs.civInput?.focus(), 350);
      });
    },
    toggleModelsPanel(which) {
      this.mpanels[which] = !this.mpanels[which];
    },
    openModelFolders() {
      this.tab = "settings";
      this.$nextTick(() => {
        this.settingsUnfold(document.getElementById("models-section"));
        document.getElementById("models-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    },
    async addLocalModels() {
      this.localErr = "";
      this.localOk = "";
      if (!window.bee.pickModels) {
        this.localErr = "Quit and reopen Protoart Studio once to enable this button (the app itself was updated).";
        return;
      }
      const paths = await window.bee.pickModels();
      if (!paths || !paths.length) return;
      try {
        const r = await call("model_local_inspect", { paths });
        if (r.found.length) {
          const local = await call("model_local_add", { models: r.found });
          this.modelDirs = { ...this.modelDirs, local };
          this.models = await call("list_models");
          this.status = `Added ${r.found.length} model${r.found.length === 1 ? "" : "s"}: ` + r.found.map((m) => m.label).join(", ");
          this.localOk = `✓ ${r.found.length === 1 ? "Model" : r.found.length + " models"} imported successfully: ` +
            r.found.map((m) => m.label + (m.inpaint ? " (inpainting: used by Image to image → Inpaint / Expand)" : "")).join(", ") +
            (this.tab === "create" ? "" : ". It's in Create's model list now.");
          clearTimeout(this._okT);
          this._okT = setTimeout(() => (this.localOk = ""), 20000);
          if (this.tab === "models") this.mpanels = { ...this.mpanels, own: true };
          if (this.tab === "create") {
            // show it in the list, and pick it straight away when it's just one
            this.modelFilter = "all";
            this.modelSearch = "";
            if (r.found.length === 1 && this.models.some((m) => m.id === r.found[0].id)) this.selectModel(r.found[0].id);
          }
        }
        await this.useLoraFolders(r.lora_folders);
        if (r.errors.length) this.localErr = r.errors.join("\n");
        else if (!r.found.length && !(r.lora_folders || []).length) this.localErr = "No models were found there (a model is a .safetensors / .ckpt file of 1.5 GB or more, or a folder with model_index.json).";
      } catch (e) {
        this.localErr = e.message;
      }
    },
    async useLoraFolders(found) {
      // a models folder that also holds LoRAs (or DiffusionBee .tdict files): use it as a LoRA folder too
      const add = (found || []).map((f) => f.path).filter((p) => !this.loraDirs.dirs.includes(p));
      if (!add.length) return;
      this.loraDirs = await call("lora_dirs_set", { dirs: [...this.loraDirs.dirs, ...add] });
      await this.refreshLoras();
      const n = found.reduce((a, f) => a + f.loras, 0);
      this.status = (this.status ? this.status + " · " : "") +
        `Also using ${add.length === 1 ? "that folder" : add.length + " folders"} for LoRAs (${n} found; listed in Settings → LoRA folders)`;
    },
    confirmAsk(title, text, ok = "Remove") {  // a "Are you sure?" box with Cancel; resolves true / false
      return new Promise((resolve) => { this.ask = { title, text, ok, resolve }; });
    },
    answerAsk(yes) {
      const a = this.ask;
      this.ask = null;
      if (a) a.resolve(!!yes);
    },
    async removeLocalModel(id) {
      const m = this.models.find((x) => x.id === id);
      if (!(await this.confirmAsk(`Remove “${m ? m.label : "this model"}”?`,
        "It's taken off the list. The file itself stays where it is on your Mac; you can add it again any time."))) return;
      this.unlocked[id] = false;
      const local = await call("model_local_remove", { id });
      this.modelDirs = { ...this.modelDirs, local };
      this.models = await call("list_models");
      if (this.form.model === id) { this.form.model = this.models[0].id; await this.onModelChange(); }
    },
    async loadDownloadsDir() {
      try { Object.assign(this.dl, await call("downloads_dir_get")); } catch {}
    },
    async measureDownloads() {
      this.dl.sizeGb = "…";
      try { this.dl.sizeGb = (await call("downloads_dir_size", { path: this.dl.path })).gb; } catch { this.dl.sizeGb = null; }
    },
    async changeDownloadsDir(useDefault) {
      let target = null;
      if (useDefault !== true) {
        target = await window.bee.pickFolder({ title: "Choose where downloaded models are kept", defaultPath: this.dl.path, button: "Use this folder" });
        if (!target) return;
      }
      const from = this.dl.path;
      const to = target || this.dl.default;
      if (to === from) return;
      this.dl.busy = true;
      try {
        await call("downloads_dir_set", { path: target });
        if (this.dl.move) {
          this.status = "Moving downloaded models… (this can take a while for a different drive)";
          const r = await this.runJob("downloads_move", { from, to });
          if (r) this.status = `Moved ${r.moved} model${r.moved === 1 ? "" : "s"} (${r.gb} GB) to ${to}` + (r.skipped?.length ? ` · ${r.skipped.length} were already there` : "");
        }
        this.status = (this.status || "") + " · restarting the engine to use the new folder…";
        await window.bee.restart();
      } finally {
        this.dl.busy = false;
        setTimeout(() => { this.loadDownloadsDir(); call("list_models").then((m) => { this.models = m; }).catch(() => {}); }, 4000);
      }
    },
    async libAnswer(use) {
      const pick = (this.libOffer || []).filter((f) => f.pick);
      this.libOffer = null;
      try {
        if (use && pick.length) {
          for (const f of pick) this.libs = await call("libraries_use", { root: f.root, use: true, info: f });
          await this.afterLibChange(`Using ${pick.map((f) => f.name).join(" and ")} · nothing in ${pick.length === 1 ? "it" : "them"} was moved or changed`);
        } else {
          this.libs = await call("libraries_dismiss");
          this.status = "OK · you can use it later in Settings → Model folders";
        }
      } catch (e) { this.status = "Couldn't use that library: " + e.message; }
    },
    async libToggle(f, use) {
      if (!use && !(await this.confirmAsk("Stop using " + f.name + "?", f.root + "\n\nIts models leave the lists; the files stay where they are (nothing is deleted).", "Stop using"))) return;
      try {
        this.libs = await call("libraries_use", { root: f.root, use, info: use ? f : null });
        await this.afterLibChange(use ? `Using ${f.name} (${f.summary})` : `Stopped using ${f.name} · its files weren't touched`);
      } catch (e) { this.libMsg = e.message; }
    },
    async libRefresh() {
      this.libBusy = true; this.libMsg = "";
      try {
        this.libs = await call("libraries_refresh");
        await this.afterLibChange(`Looked again · ${this.libs.found.length} model ${this.libs.found.length === 1 ? "library" : "libraries"} found, every watched folder re-read`);
      } catch (e) { this.libMsg = e.message; } finally { this.libBusy = false; }
    },
    async libPick() {
      const picked = await window.bee.pickFolder({ title: "Choose a model library folder (e.g. Stability Matrix's Data folder)", button: "Use this library" });
      if (!picked) return;
      try {
        this.libs = await call("libraries_add", { path: picked });
        const u = this.libs.used.find((x) => picked === x.root || picked.startsWith(x.root + "/")) || {};
        await this.afterLibChange(`Using ${u.name || "that library"}${u.summary ? " (" + u.summary + ")" : ""}`);
      } catch (e) { this.libMsg = e.message; }
    },
    async afterLibChange(msg) {
      this.libMsg = "";
      this.modelDirs = await call("model_dirs_get");
      this.mainFolder = await call("main_folder_get").catch(() => this.mainFolder);
      this.imagesDir = await call("images_dir_get").catch(() => this.imagesDir);
      call("embedding_dirs_get").then((d) => { this.embDirs = d; }).catch(() => {});
      this.embeddings = await call("list_embeddings").catch(() => this.embeddings);
      this.loraDirs = await call("lora_dirs_get");
      this.models = await call("list_models");
      await this.refreshLoras();
      this.status = msg;
    },
    libReady(u) {
      // models from this library the app can use now (each file's type is read from its header)
      const dirs = Object.values(u.dirs || {}).flat();
      return (this.localModels || []).filter((m) => dirs.some((d) => (m.file || "").startsWith(d + "/"))).length;
    },
    async pickMainFolder() {
      const picked = await window.bee.pickFolder({ title: "Choose your main folder (models, LoRAs, embeddings…)", defaultPath: this.mainFolder.path || undefined, button: "Use as main folder" });
      if (!picked) return;
      this.libBusy = true;
      try {
        this.mainFolder = await call("main_folder_set", { path: picked });
        await this.afterLibChange(`Main folder: ${picked} · ${this.localModels.length} of your models found`);
      } catch (e) { this.error = e.message; } finally { this.libBusy = false; }
    },
    async stopMainFolder() {
      if (!(await this.confirmAsk("Stop using this main folder?", this.mainFolder.path + "\n\nIts models, LoRAs and embeddings leave the lists; the files stay where they are (nothing is deleted).", "Stop using"))) return;
      try {
        this.mainFolder = await call("main_folder_set", { path: "" });
        await this.afterLibChange("Stopped using the main folder · its files weren't touched");
      } catch (e) { this.error = e.message; }
    },
    mainSubName(kind) {
      const p = (this.mainFolder.subs || {})[kind] || "";
      return p.split("/").pop();
    },
    mainKindLabel(k) {
      return { checkpoints: "Models", loras: "LoRAs", embeddings: "Embeddings", vaes: "VAE", controlnets: "ControlNet" }[k] || k;
    },
    folderProblem(info, d) {
      return this.accessProblem(((info && info.access) || {})[d]);
    },
    accessProblem(a) {  // a folder the app can't read: the drive isn't plugged in, or macOS hasn't allowed it yet
      if (a === "missing") return "⚠ Not found: is the drive connected? Plug it in, then press ↻ Refresh.";
      if (a === "denied") return "⚠ macOS hasn't let Protoart Studio read this drive yet. Click Allow when macOS asks about a removable volume (or System Settings → Privacy & Security → Files and Folders → Protoart Studio → Removable Volumes), then press ↻ Refresh.";
      return "";
    },
    async refreshWatched() {
      this.libBusy = true;
      try {
        this.libs = await call("libraries_refresh");
        await this.afterLibChange(`Read all your folders again · ${this.localModels.length} of your models, ${this.loras.length} LoRAs found`);
      } catch (e) { this.status = e.message; } finally { this.libBusy = false; }
    },
    loraTagsOf(name) {
      return (this.loraTags && this.loraTags[name]) || [];
    },
    cleanTags(text) {  // "Pencil, ink sketch" → ["pencil", "ink sketch"]
      return String(text || "").split(",").map((t) => t.trim().toLowerCase().replace(/^#/, "").replace(/\s+/g, " ").slice(0, 30)).filter(Boolean);
    },
    saveLoraTags() {
      saveJSON("protoart.loraTags", this.loraTags);
    },
    async tagLoras(action) {  // ＋ / − a tag on every selected LoRA (and, if ticked, as its trigger words too)
      const tags = this.cleanTags(this.tagBatch.tag);
      if (!tags.length || !this.tagBatch.sel.length) return;
      saveJSON("protoart.tagAsTrig", !!this.tagBatch.asTrig);
      const next = { ...this.loraTags };
      for (const name of this.tagBatch.sel) {
        const cur = new Set(next[name] || []);
        for (const t of tags) action === "add" ? cur.add(t) : cur.delete(t);
        if (cur.size) next[name] = [...cur].sort(); else delete next[name];
      }
      this.loraTags = next;
      this.saveLoraTags();
      this.status = `${action === "add" ? "Tagged" : "Untagged"} ${this.tagBatch.sel.length} LoRA${this.tagBatch.sel.length === 1 ? "" : "s"}: ${tags.map((t) => "🏷 " + t).join(", ")}`;
      if (this.tagBatch.asTrig) {  // the same words as trigger words: they show as chips under the LoRA in Create
        this.tagBatch.busy = true;
        try {
          for (const name of this.tagBatch.sel) {
            await this.fetchTriggers(name);
            let words = [...this.activeTriggers(name)];
            if (action === "add") { for (const w of tags) if (!words.some((x) => x.toLowerCase() === w)) words.push(w); }
            else words = words.filter((x) => !tags.includes(x.toLowerCase()));
            await this.saveTriggerWords(name, words.join(", "));
          }
          this.status += action === "add" ? " · also added as trigger words" : " · also removed from the trigger words";
        } catch (e) { this.error = e.message; } finally { this.tagBatch.busy = false; }
      }
    },
    removeLoraTag(name, tag) {
      const left = this.loraTagsOf(name).filter((t) => t !== tag);
      const next = { ...this.loraTags };
      if (left.length) next[name] = left; else delete next[name];
      this.loraTags = next;
      this.saveLoraTags();
    },
    renameLoraTag(tag) {  // rename (or merge) a tag everywhere
      const to = this.cleanTags(this.tagBatch.tag)[0];
      if (!to || to === tag) return;
      const next = {};
      for (const [name, tags] of Object.entries(this.loraTags)) next[name] = [...new Set(tags.map((t) => (t === tag ? to : t)))].sort();
      this.loraTags = next;
      this.saveLoraTags();
      this.status = `🏷 ${tag} is now 🏷 ${to}`;
    },
    loraTag(l) {
      // Short so it fits beside the size on every row: BEE = from DiffusionBee, then the LoRA's type.
      const bits = [];
      if (l.source === "diffusionbee") bits.push("BEE");
      if (l.family === "unknown") bits.push("type?");
      if (l.flavor && this.flavorName(l.flavor)) bits.push(this.flavorName(l.flavor) + (this.currentFlavor && l.flavor !== this.currentFlavor ? " ⚠" : ""));
      for (const t of this.loraTagsOf(l.name)) bits.push("🏷 " + t);
      return bits.join(" · ");
    },
    loraMetaTitle(l) {
      const bits = [];
      if (l.source === "diffusionbee") bits.push("BEE: from your DiffusionBee folder (read in place)");
      if (l.family === "unknown") bits.push("type unknown");
      if (l.flavor) bits.push(this.flavorName(l.flavor) + (this.currentFlavor && l.flavor !== this.currentFlavor ? " (may not match this model)" : ""));
      const tags = this.loraTagsOf(l.name);
      if (tags.length) bits.push("your tags: " + tags.join(", "));
      bits.push(l.size_mb + " MB");
      return bits.join(" · ");
    },
    async addModelDir() {
      let picked = await window.bee.pickFolder({ title: "Choose a folder that holds your models", multi: true, button: "Watch" });
      if (typeof picked === "string") picked = [picked];
      if (!picked || !picked.length) return;
      this.modelDirs = await call("model_dirs_set", { dirs: [...this.modelDirs.dirs, ...picked] });
      this.models = await call("list_models");
      this.status = `Watching ${picked.length} folder${picked.length === 1 ? "" : "s"} · ${this.localModels.length} of your models found`;
      try {
        const r = await call("model_local_inspect", { paths: picked }); // only to spot LoRAs in the same folders
        await this.useLoraFolders(r.lora_folders);
      } catch { /* nothing to add */ }
    },
    async removeModelDir(d) {
      if (!(await this.confirmAsk("Stop watching this folder?", d + "\n\nIts models leave the list; the files stay where they are."))) return;
      this.modelDirs = await call("model_dirs_set", { dirs: this.modelDirs.dirs.filter((x) => x !== d) });
      this.models = await call("list_models");
    },
    async hubInspect() {
      Object.assign(this.hub, { busy: true, error: "", found: null, needToken: false, msg: "" });
      try {
        this.hub.found = await call("model_inspect", { source: this.hub.source });
      } catch (e) {
        this.hub.error = e.message;
        // a gated model: ask for the Hugging Face access token right here
        this.hub.needToken = /gated on Hugging Face|turned the download|access token/i.test(e.message || "");
      }
      this.hub.busy = false;
    },
    async hfSaveToken(clear, thenRetry = false) {
      try {
        const r = await call("hf_token_set", { token: clear ? "" : this.hub.token, check: !clear });
        this.hub.tokenSet = r.set;
        this.hub.token = "";
        if (r.ok === false) { this.hub.error = r.msg; this.hub.needToken = true; return; }
        this.hub.msg = r.set ? "Hugging Face token saved on this Mac" + (r.ok ? " · Hugging Face accepted it" : "") + (r.msg ? " · " + r.msg : "")
          : "Hugging Face token removed";
        if (r.set && thenRetry && this.hub.source.trim()) await this.hubInspect();
      } catch (e) {
        this.hub.error = e.message;
      }
    },
    // -------------------------------------------------------------- Civitai
    async civInspect() {
      const url = this.civ.url.trim();
      if (!url) return;
      Object.assign(this.civ, { busy: true, error: "", found: null, msg: "", needKey: "", last: "inspect" });
      try {
        const f = await callWithin(90000, "civitai_inspect", { url });
        this.civ.found = f;
        this.civ.dest = (f.destinations[0] || {}).path || "";
      } catch (e) {
        this.civ.error = e.message;
      }
      this.civ.busy = false;
    },
    civJob() {
      return this.queue.find((j) => j.cmd === "civitai_download");
    },
    // after an error: do the same thing again (Check, or Download if the model was already found)
    civRetry() {
      this.civ.error = "";
      if (this.civ.last === "download" && this.civ.found) return this.civDownload();
      return this.civInspect();
    },
    // start over: empty link, no card, no message
    civClear() {
      Object.assign(this.civ, { url: "", busy: false, error: "", found: null, dest: "", msg: "", needKey: "", last: "" });
      this.$nextTick(() => this.$refs.civInput?.focus());
    },
    civJobs() {  // every Civitai download in the queue, running first
      return this.queue.filter((j) => j.cmd === "civitai_download");
    },
    // Download: goes into the queue (one after another), and the box is free for the next link at once.
    // If you may already have it, ask first (force = "download anyway").
    async civDownload(force = false) {
      const f = this.civ.found;
      if (!f) return;
      if (f.duplicates?.length && !force) { this.civ.dupAsk = true; return; }
      const item = { url: this.civ.url.trim(), dest: this.civ.dest, found: f };
      Object.assign(this.civ, { msg: `${f.name} added to the downloads below`, error: "", needKey: "", last: "download",
                                dupAsk: false, found: null, url: "" });
      this.$nextTick(() => this.$refs.civInput?.focus());
      let r = null;
      try {
        r = await call("civitai_download", { url: item.url, dest: item.dest, name: f.name });
      } catch (e) {
        if (e.cancelled) return;
        const m = String(e.message || e);
        // put it back in the card, so ⟳ Retry (or the API key box) works on it
        Object.assign(this.civ, { found: item.found, url: item.url, dest: item.dest, error: `${f.name}: ${m}`, msg: "" });
        // Civitai wants a signed-in account: ask for the API key right here, in the card
        this.civ.needKey = /turned down your API key/.test(m) ? "refused" : /signed-in accounts|log in/.test(m) ? "missing" : "";
        if (this.civ.needKey) this.$nextTick(() => this.$refs.civKeyInline?.focus());
        return;
      }
      if (!r) return;
      this.civ.msg = `${f.name} is ready · saved in ${r.dest}`;
      this.models = await call("list_models");
      await this.refreshLoras();
      await this.refreshEmbeddings();
      await this.refreshExtras();
      this.status = this.civ.msg + (r.kind === "lora" ? " · it's in the LoRA list"
        : r.kind === "embedding" ? " · click it above the negative prompt (with a model of its type)"
        : r.kind === "vae" ? " · pick it in Create → Extras → VAE"
        : r.kind === "controlnet" ? " · pick it in Image to image → ControlNet" : " · it's in Create's model list");
    },
    async civSaveKey(clear, thenDownload = false) {
      try {
        const r = await callWithin(90000, "civitai_key_set", { key: clear ? "" : this.civ.key, check: !clear });
        this.civ.keySet = r.set;
        this.civ.key = "";
        if (r.ok === false) {
          this.civ.error = r.msg;
          this.civ.needKey = "refused";
          return;
        }
        this.civ.msg = r.set ? "Civitai API key saved on this Mac" + (r.ok ? " · Civitai accepted it" : "") : "Civitai API key removed";
        if (r.set && thenDownload && this.civ.found) {
          this.civ.needKey = "";
          this.civ.error = "";
          await this.civDownload();
        }
      } catch (e) {
        this.civ.error = e.message;
      }
    },
    civSize(mb) {
      return mb > 1000 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
    },
    async hubAdd() {
      try {
        await call("model_add", { model: this.hub.found });
        this.models = await call("list_models");
        this.status = `Added ${this.hub.found.label}. Pick it in the model list (it downloads on first use).`;
        Object.assign(this.hub, { source: "", found: null });
      } catch (e) {
        this.hub.error = e.message;
      }
    },
    async hubRemove(m) {
      if (!(await this.confirmAsk(`Remove “${m.label}”?`, "It's taken off your model list. Downloaded files are kept until you delete them in the Models tab."))) return;
      this.unlocked[m.id] = false;
      await call("model_remove", { id: m.id });
      this.models = await call("list_models");
      if (this.form.model === m.id) {
        this.form.model = this.models[0].id;
        await this.onModelChange();
      }
    },
    applySizePreset(p) {
      this.form.width = p.w;
      this.form.height = p.h;
      if (this.form.size_link) this.form.size_ratio = p.w / p.h;
    },
    toggleSizeLink() {
      this.form.size_link = !this.form.size_link;
      if (this.form.size_link) this.form.size_ratio = this.form.width / this.form.height;
    },
    swapSize() {
      [this.form.width, this.form.height] = [this.form.height, this.form.width];
      if (this.form.size_ratio) this.form.size_ratio = 1 / this.form.size_ratio;
    },
    toggleFast(on) {
      if (on) {
        this.form.fast_prev = { steps: this.form.steps, cfg: this.form.cfg, sampler: this.form.sampler };
        Object.assign(this.form, { fast: true, steps: 6, cfg: 1.5, sampler: "lcm" });
        this.status = "Fast mode: 6 steps, CFG 1.5, LCM (downloads a small LoRA the first time)";
      } else {
        const p = this.form.fast_prev;
        this.form.fast = false;
        if (p) Object.assign(this.form, { steps: p.steps, cfg: p.cfg, sampler: p.sampler });
        else this.onSamplerChange();
        this.form.fast_prev = null;
      }
    },
    samplerPicks() {
      // {fast: [samplers], quality: [samplers], rest: [samplers]} for the chosen model, only ones it offers
      const fam = this.currentFlavor && this.family === "sdxl" ? "anime_xl" : this.family;
      const p = SAMPLER_PICKS[fam] || { fast: [], quality: [] };
      const by = Object.fromEntries((this.samplers || []).map((x) => [x.id, x]));
      // only the top two of each, so the choice stays simple; the others are under "All other samplers"
      const fast = p.fast.map((id) => by[id]).filter(Boolean).slice(0, 2);
      const quality = p.quality.map((id) => by[id]).filter(Boolean).slice(0, 2);
      const shown = new Set([...fast, ...quality].map((x) => x.id));
      return { fast, quality, rest: (this.samplers || []).filter((x) => !shown.has(x.id)),
               qualityOnly: quality.filter((x) => !fast.some((f) => f.id === x.id)) };
    },
    pickFor() {
      // name for the sampler picks: Pony / Illustrious models get their own (anime-style SDXL)
      const f = this.currentFlavor;
      if (f && this.family === "sdxl") return { pony: "Pony", illustrious: "Illustrious", noobai: "NoobAI" }[String(f).toLowerCase()] || "anime-style SDXL";
      return this.familyLabel;
    },
    isFavLora(l) {
      return !!l && this.favLoras.includes(l.name);
    },
    toggleFavLora(l) {
      const i = this.favLoras.indexOf(l.name);
      if (i >= 0) this.favLoras.splice(i, 1);
      else this.favLoras.push(l.name);
      saveJSON("protoart.favLoras", this.favLoras);
    },
    favFirst(list) {  // favourite LoRAs first, otherwise in the same order
      if (!this.favLoras.length) return list;
      return [...list.filter((l) => this.isFavLora(l)), ...list.filter((l) => !this.isFavLora(l))];
    },
    isFav(m) {
      return !!m && this.favModels.includes(m.id);
    },
    toggleFav(m) {
      const i = this.favModels.indexOf(m.id);
      if (i >= 0) this.favModels.splice(i, 1);
      else this.favModels.push(m.id);
      saveJSON("protoart.favModels", this.favModels);
    },
    samplerIsPick(id) {
      const p = this.samplerPicks();
      return p.fast.some((x) => x.id === id) || p.quality.some((x) => x.id === id);
    },
    pickSampler(id) {
      if (this.form.fast && this.currentModel?.fast_mode) return;
      this.form.sampler = id;
      this.onSamplerChange();
    },
    onSamplerChange() {
      const r = this.recSteps;
      if (r) {
        this.form.steps = r[0];
        const label = this.samplers.find((x) => x.id === this.form.sampler)?.label || "this sampler";
        this.status = `Steps set to ${r[0]}, recommended for ${label}`;
      }
      if (["lcm", "tcd"].includes(this.form.sampler) && this.form.cfg > 2) this.form.cfg = 1.5;
    },
    async setDiffusionBee(enabled) {
      try {
        this.dbee = await call("diffusionbee_set", { enabled });
        await this.refreshDiffusionBee();
      } catch (e) {
        this.error = e.message;
      }
    },
    async refreshDiffusionBee() {
      try {
        this.status = "Scanning DiffusionBee folder…";
        [this.dbee, this.models] = await Promise.all([call("diffusionbee_info"), call("list_models")]);
        await this.refreshLoras();
        this.status = "Ready";
      } catch (e) {
        this.error = e.message;
      }
    },
    async loadSamplers(resetToDefault) {
      this.samplers = await call("list_samplers", { family: this.family });
      if (resetToDefault || !this.samplers.some((s) => s.id === this.form.sampler)) {
        this.form.sampler = (this.samplers.find((s) => s.default) || this.samplers[0]).id;
      }
    },

    // -------------------------------------------------------------- model / LoRAs
    // Create -> "Reset to defaults": the model's own size, steps, guidance and sampler; 1 image; random seed;
    // its recommended CLIP skip; extras off. Prompt, negative prompt, model and LoRAs are kept.
    async resetCreate() {
      const m = this.currentModel;
      if (!m) return;
      this.resetUndo = JSON.parse(JSON.stringify(this.form));
      this.form.fast = false;
      this.form.fast_prev = null;
      Object.assign(this.form, { num_images: 1, seed: -1, clip_skip: m.clip_skip || 1, restore_faces: false });
      if (this.form.hires) this.form.hires.on = false;
      if (this.form.face_detailer) this.form.face_detailer.on = false;
      this.cmp.open = false;
      await this.onModelChange(); // size, steps, guidance and the default sampler for this model
      this.status = `Create settings reset to the defaults for ${m.label}`;
    },
    flipResetAll() {  // a spring-loaded lever: flips, resets, and returns to rest a moment later
      if (this.resetFlip) return;
      this.resetFlip = true;
      this.resetCreateAll();
      setTimeout(() => { this.resetFlip = false; }, 550);
    },
    // Create → reset all: every setting back to the start (prompt, model and presets stay)
    async resetCreateAll() {
      const m = this.currentModel;
      if (!m) return;
      this.resetUndo = null;  // no undo: the lever springs back
      // a clean slate: no prompt or negative prompt either (Image to image's own prompt is left alone)
      const keep = { model: this.form.model, prompt: "", negative_prompt: "", prompt_owner: this.form.prompt_owner,
                     i2i_prompt: this.form.i2i_prompt, i2i_negative: this.form.i2i_negative };
      this.form = Object.assign(defaultForm(), keep);
      this.cmp.open = false;
      this.lastEnhanced = null;
      this.lorasAside = {};
      saveJSON("protoart.lorasAside", this.lorasAside);
      await this.onModelChange();
      this.cfAll(true);  // every section folded down to one line
      this.status = "Create is reset: prompts cleared, everything back to the start, sections folded";
    },
    resetTools() {
      this.toolsUndo = JSON.parse(JSON.stringify(this.post));
      this.post = defaultPost();
      this.status = "Tools settings reset · undo puts them back";
    },
    undoResetTools() {
      if (!this.toolsUndo) return;
      this.post = this.toolsUndo;
      this.toolsUndo = null;
      this.status = "Tools settings restored";
    },
    undoResetCreate() {
      if (!this.resetUndo) return;
      this.form = this.resetUndo;
      this.resetUndo = null;
      if (this.resetFolds) { this.cfolds = this.resetFolds; saveJSON("protoart.createFolds", this.cfolds); this.resetFolds = null; }
      this.status = "Settings restored";
    },
    async onModelChange() {
      const m = this.currentModel;
      if (!m) return;
      this.pickVaeFor(m);
      if (this.opts.preload_on_pick) {
        clearTimeout(this.preloadTimer); // wait until you've settled on one
        this.preloadTimer = setTimeout(() => this.preloadModel(this.form.model), 1200);
      }
      this.form.clip_skip = this.clipSkipFor(m);
      Object.assign(this.form, { width: m.defaults.width, height: m.defaults.height, steps: m.defaults.steps, cfg: m.defaults.cfg });
      if (["ernie", "flux", "flux2"].includes(m.family) && this.sys.ram_gb && this.sys.ram_gb < 12) {
        Object.assign(this.form, { width: 768, height: 768 }); // 8 GB Macs: Medium is ~2x faster than 1024
        this.status = `Size set to Medium (768) for ${FAMILY_LABEL[m.family]} on this Mac; Large (1024) takes about twice as long`;
      }
      if (this.form.size_link) this.form.size_ratio = m.defaults.width / m.defaults.height;
      if (this.form.fast) {
        this.form.fast_prev = null;
        if (m.fast_mode) Object.assign(this.form, { steps: 6, cfg: 1.5 });
        else this.form.fast = false;
      }
      // LoRAs that don't fit the new model's type are put aside (per type), and come back when you switch back
      if (this.loras.length) {
        const keep = this.form.loras.filter((l) => this.compatibleLoras.some((c) => c.name === l.name));
        const aside = this.form.loras.filter((l) => !keep.includes(l));
        for (const l of aside) {
          const fam = this.loras.find((x) => x.name === l.name)?.family;
          if (fam && fam !== "unknown") (this.lorasAside[fam] ||= []).some((x) => x.name === l.name) || this.lorasAside[fam].push({ ...l });
        }
        const back = (this.lorasAside[m.family] || []).filter((l) => !keep.some((k) => k.name === l.name)
          && this.compatibleLoras.some((c) => c.name === l.name));
        this.lorasAside[m.family] = [];
        this.form.loras = keep.concat(back);
        if (aside.length || back.length) {
          this.status = (aside.length ? `Put ${aside.length} LoRA${aside.length === 1 ? "" : "s"} for another model type aside` : "")
            + (aside.length && back.length ? "; " : "") + (back.length ? `brought back ${back.map((l) => l.name).join(", ")}` : "")
            + (aside.length ? " (they come back when you pick that type again)" : "");
        }
        saveJSON("protoart.lorasAside", this.lorasAside);
      }
      await this.loadSamplers(true);
      this.applyModelDefaults(m, true);  // your own starting settings for this model, if you saved some
    },
    applyModelDefaults(m = this.currentModel, quiet = false) {
      const d = m && (this.modelDefaults || {})[m.id];
      if (!d) return false;
      if (d.sampler && (this.samplers || []).some((x) => x.id === d.sampler)) this.form.sampler = d.sampler;
      for (const k of ["steps", "cfg", "clip_skip", "width", "height"]) if (typeof d[k] === "number") this.form[k] = d[k];
      if (!quiet) this.status = `Your defaults for ${m.label}: ${this.modelDefaultsText(m)}`;
      return true;
    },
    modelDefaultsText(m = this.currentModel) {
      const d = m && (this.modelDefaults || {})[m.id];
      if (!d) return "";
      const sl = (this.samplers || []).find((x) => x.id === d.sampler);
      return [sl ? sl.label : d.sampler, d.steps + " steps", "guidance " + d.cfg].filter(Boolean).join(" · ");
    },
    saveModelDefaults() {  // 💾 this model starts with the sampler, steps and guidance you have now
      const m = this.currentModel;
      if (!m) return;
      const f = this.form;
      this.modelDefaults = { ...this.modelDefaults, [m.id]: { sampler: f.sampler, steps: f.steps, cfg: f.cfg, clip_skip: f.clip_skip,
                                                              width: f.width, height: f.height } };
      saveJSON("protoart.modelDefaults", this.modelDefaults);
      this.status = `Saved as ${m.label}'s defaults: ${this.modelDefaultsText(m)} (used each time you pick it; you can still change them)`;
    },
    forgetModelDefaults() {
      const m = this.currentModel;
      if (!m) return;
      const next = { ...this.modelDefaults };
      delete next[m.id];
      this.modelDefaults = next;
      saveJSON("protoart.modelDefaults", next);
      this.status = `${m.label} uses the app's starting settings again`;
    },
    addLoraByName(name) {
      if (!this.form.loras.some((l) => l.name === name)) {
        this.form.loras.push({ name, weight: this.loraDefaultWeight(name) ?? 1.0 });  // your saved weight, if any
        this.fetchTriggers(name).then(() => {
          if (!this.opts.auto_triggers) return;
          const words = this.activeTriggers(name).filter((w) => !this.inPrompt(w));
          if (words.length) {
            this.addPromptWords(words);
            this.status = `Added trigger words for ${name}: ${words.join(", ")}`;
          }
        });
      }
      this.loraSearch = "";
    },
    loraDefaultWeight(name) {  // the weight you saved for this LoRA (★), or null
      const w = (this.loraWeights || {})[name];
      return typeof w === "number" ? w : null;
    },
    setLoraDefaultWeight(name, w) {
      this.loraWeights = { ...this.loraWeights, [name]: Number(w) };
      saveJSON("protoart.loraWeights", this.loraWeights);
      this.status = `${name} will start at ${w} each time you add it`;
    },
    // -------------------------------------------------------------- trigger words
    async fetchTriggers(name, force) {
      if (this.trigInfo[name] && !force) return this.trigInfo[name];
      try {
        this.trigInfo[name] = await call("lora_triggers", { name });
      } catch {
        this.trigInfo[name] = { found: [], suggested: [], saved: null };
      }
      return this.trigInfo[name];
    },
    activeTriggers(name) {
      const t = this.trigInfo[name];
      return t ? (t.saved || t.found || []) : [];
    },
    suggestedTriggers(name) {
      const t = this.trigInfo[name];
      if (!t || t.saved) return [];
      // only offer training-tag suggestions when nothing definite was found
      return (t.found || []).length ? [] : (t.suggested || []).slice(0, 5);
    },
    wordRe(w) {
      return new RegExp(`(^|[\\s,(])${w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?=$|[\\s,):])`, "i");
    },
    inPrompt(w) {
      return this.wordRe(w).test(this.form.prompt || "");
    },
    inNegative(w) {
      return this.wordRe(w).test(this.form.negative_prompt || "");
    },
    addPromptWords(words) {
      const p = (this.form.prompt || "").trim().replace(/,\s*$/, "");
      this.form.prompt = (p ? p + ", " : "") + words.join(", ");
    },
    removeWord(text, w) {
      return (text || "").replace(this.wordRe(w), "$1").replace(/\s*,\s*,/g, ",").replace(/^\s*,\s*|\s*,\s*$/g, "").trim();
    },
    togglePromptWord(w) {
      if (this.inPrompt(w)) this.form.prompt = this.removeWord(this.form.prompt, w);
      else this.addPromptWords([w]);
    },
    toggleNegativeWord(w) {
      if (this.inNegative(w)) this.form.negative_prompt = this.removeWord(this.form.negative_prompt, w);
      else {
        const n = (this.form.negative_prompt || "").trim().replace(/,\s*$/, "");
        this.form.negative_prompt = (n ? n + ", " : "") + w;
      }
    },
    loadTrigListSoon() {
      clearTimeout(this._trigT);
      this._trigT = setTimeout(() => this.loadTrigList(), 300);
    },
    async loadTrigList() {  // read the trigger words of the listed LoRAs (a few at a time)
      const todo = this.trigList.filter((l) => !this.trigInfo[l.name]).map((l) => l.name);
      for (let i = 0; i < todo.length; i += 6) await Promise.all(todo.slice(i, i + 6).map((n) => this.fetchTriggers(n)));
      this.trigBatch.sel = this.trigBatch.sel.filter((n) => this.trigList.some((l) => l.name === n));
    },
    async saveTriggerWords(name, text) {
      const words = String(text || "").split(",").map((w) => w.trim()).filter(Boolean);
      const saved = await call("lora_triggers_save", { name, words });
      this.trigInfo[name] = Object.assign({}, this.trigInfo[name] || { found: [], suggested: [] }, { saved: saved || null });
    },
    async batchTriggers(how) {  // add / remove words for every selected LoRA, or put the file's own words back
      const b = this.trigBatch, add = b.word.split(",").map((w) => w.trim()).filter(Boolean);
      b.busy = true;
      try {
        for (const name of b.sel) {
          await this.fetchTriggers(name);
          if (how === "reset") { await this.saveTriggerWords(name, ""); continue; }
          let words = [...this.activeTriggers(name)];
          if (how === "add") for (const w of add) { if (!words.some((x) => x.toLowerCase() === w.toLowerCase())) words.push(w); }
          else words = words.filter((x) => !add.some((w) => w.toLowerCase() === x.toLowerCase()));
          await this.saveTriggerWords(name, words.join(", "));
        }
        this.status = how === "reset" ? `${b.sel.length} LoRAs use the words in their files again`
          : `${how === "add" ? "Added" : "Removed"} “${add.join(", ")}” ${how === "add" ? "to" : "from"} ${b.sel.length} LoRA${b.sel.length === 1 ? "" : "s"}`;
      } catch (e) {
        this.error = e.message;
      } finally {
        b.busy = false;
      }
    },
    startTrigEdit(name) {
      const words = this.activeTriggers(name).length ? this.activeTriggers(name) : this.suggestedTriggers(name);
      this.trigEdit = { name, text: words.join(", ") };
    },
    async saveTrigEdit() {
      const { name, text } = this.trigEdit;
      const words = text.split(",").map((w) => w.trim()).filter(Boolean);
      try {
        const saved = await call("lora_triggers_save", { name, words });
        this.trigInfo[name] = Object.assign({}, this.trigInfo[name] || { found: [], suggested: [] }, { saved: saved || null });
        this.status = words.length ? `Saved trigger words for ${name}` : `Cleared saved trigger words for ${name}`;
      } catch (e) {
        this.error = e.message;
      }
      this.trigEdit = { name: "", text: "" };
    },
    // -------------------------------------------------------------- embeddings
    async refreshEmbeddings() {
      try {
        [this.embeddings, this.embDirs] = await Promise.all([call("list_embeddings"), call("embedding_dirs_get")]);
      } catch {}
    },
    embeddingsIn(d) {
      const pre = d.replace(/\/+$/, "") + "/";
      return (this.embeddings || []).filter((e) => (e.file || "").startsWith(pre)).length;
    },
    embSourceName(e) {
      return { protoart: "Protoart folder", folder: "your folder", lora_folder: "in a LoRA folder", diffusionbee: "DiffusionBee" }[e.source] || "";
    },
    async addEmbeddingFiles() {
      const files = await window.bee.pickEmbeddings?.();
      if (!files || !files.length) return;
      try {
        const r = await call("embedding_add_files", { files });
        await this.refreshEmbeddings();
        this.embMsg = (r.added.length ? `Added ${r.added.join(", ")}` : "Nothing added") +
          (r.skipped.length ? ` · skipped ${r.skipped.map((x) => `${x.file} (${x.why})`).join(", ")}` : "");
      } catch (e) {
        this.embMsg = e.message;
      }
    },
    async addEmbeddingDir() {
      let picked = await window.bee.pickFolder({ title: "Choose embedding folders (hold ⌘ to pick several)",
                                                 defaultPath: this.embDirs.default, multi: true, button: "Add" });
      if (typeof picked === "string") picked = [picked];
      if (!picked || !picked.length) return;
      this.embDirs = await call("embedding_dirs_set", { dirs: [...(this.embDirs.dirs || []), ...picked] });
      await this.refreshEmbeddings();
      this.embMsg = `Added ${picked.length} folder${picked.length === 1 ? "" : "s"} · ${this.embeddings.length} embeddings in total`;
    },
    async removeEmbeddingDir(d) {
      if (!(await this.confirmAsk("Stop using this embeddings folder?", d + "\n\nIts embeddings leave the list; the files stay where they are."))) return;
      this.embDirs = await call("embedding_dirs_set", { dirs: (this.embDirs.dirs || []).filter((x) => x !== d) });
      await this.refreshEmbeddings();
    },
    async addLoraDir() {
      let picked = await window.bee.pickFolder({ title: "Choose LoRA folders (hold ⌘ to pick several)",
                                                 defaultPath: this.loraDirs.default, multi: true, button: "Add" });
      if (typeof picked === "string") picked = [picked]; // app not restarted yet: one folder at a time
      if (!picked || !picked.length) return;
      this.loraDirs = await call("lora_dirs_set", { dirs: [...this.loraDirs.dirs, ...picked] });
      await this.refreshLoras();
      this.planSort(picked[0], true); // a folder of mixed LoRAs: suggest sorting it into type folders
      this.status = `Added ${picked.length} LoRA folder${picked.length === 1 ? "" : "s"} · ${this.loras.length} LoRAs available in total`;
    },
    async loadFolders(warn) {
      try { const f = await call("folders_get"); this.folders = Array.isArray(f) ? f : []; } catch { return; }
      const gone = this.folders.filter((f) => !f.found);
      if (warn && gone.length)
        this.status = `Can't find ${gone.length === 1 ? gone[0].path : gone.length + " of your folders"} · moved it or drive not connected? Settings → Moved a folder?`;
    },
    async forgetFolder(path) {
      if (!(await this.confirmAsk("Stop using this folder?", path + "\n\nIts models, LoRAs and embeddings leave the lists. The files stay where they are: nothing is deleted.", "Remove"))) return;
      try {
        const r = await call("folder_forget", { path });
        this.folders = r.folders;
        await this.afterLibChange("Stopped using " + path.split("/").pop() + " · its files weren't touched");
      } catch (e) { this.error = e.message; }
    },
    async relinkFolder(old) {
      const name = old.split("/").filter(Boolean).pop();
      const picked = await window.bee.pickFolder({ title: `Where is “${name}” now?`, button: "Use this folder" });
      const nw = Array.isArray(picked) ? picked[0] : picked;
      if (!nw) return;
      try {
        const r = await call("folder_relink", { old, new: nw });
        this.folders = r.folders;
        // your own settings that name a model inside the folder
        const move = (id) => {
          if (typeof id !== "string") return id;
          for (const pre of ["local:", ""]) {
            const b = id.slice(pre.length);
            if (id.startsWith(pre) && (b === r.old || b.startsWith(r.old + "/"))) return pre + r.new + b.slice(r.old.length);
          }
          return id;
        };
        this.opts.default_model = move(this.opts.default_model);
        this.opts.preload_start = move(this.opts.preload_start);
        await this.saveOpts();
        this.form.model = move(this.form.model);
        [this.models, this.loraDirs, this.modelDirs] = await Promise.all([call("list_models"), call("lora_dirs_get"), call("model_dirs_get")]);
        await this.refreshLoras();
        this.relinkMsg = `Now using ${r.new} · ${r.changed} setting${r.changed === 1 ? "" : "s"} and ${r.pictures} Library picture${r.pictures === 1 ? "" : "s"} updated`;
        this.status = this.relinkMsg;
      } catch (e) {
        this.relinkMsg = "";
        this.error = e.message;
      }
    },
    foldSettings(ev) {
      // click a section heading in Settings to fold it away (remembered)
      const h = ev.target.closest && ev.target.closest(".set-cols .set-sec > h2");
      if (!h) return;
      const sec = h.parentElement;
      sec.classList.toggle("collapsed");
      const folded = [...document.querySelectorAll(".set-cols .set-sec.collapsed > h2")].map((x) => x.textContent.trim());
      saveJSON("protoart.settingsFolded", folded);
    },
    applySettingsFolds() {
      // First time: only the sections you need to get started are open (This Mac, Models); the rest are folded
      // down to their headings. Click a heading to open it; what you open or fold is remembered.
      const saved = loadJSON("protoart.settingsFolded", null);
      const OPEN_AT_FIRST = ["This Mac", "Model folders"];
      const heads = [...document.querySelectorAll(".set-cols .set-sec > h2")];
      const folded = new Set(Array.isArray(saved) ? saved
        : heads.map((h) => h.textContent.trim()).filter((t) => !OPEN_AT_FIRST.includes(t)));
      for (const h of heads) h.parentElement.classList.toggle("collapsed", folded.has(h.textContent.trim()));
    },
    // open the Settings section that holds `el` (links that jump to a setting), and remember it as open
    settingsUnfold(el) {
      const sec = el && el.closest(".set-sec");
      if (!sec) return;
      const title = sec.querySelector(":scope > h2")?.textContent.trim();
      let saved = loadJSON("protoart.settingsFolded", null);
      if (!Array.isArray(saved))
        saved = [...document.querySelectorAll(".set-cols .set-sec > h2")].map((h) => h.textContent.trim())
          .filter((t) => !["This Mac", "Models"].includes(t));
      saveJSON("protoart.settingsFolded", saved.filter((t) => t !== title));
      sec.classList.remove("collapsed");
    },
    settingsFoldAll(fold) {
      const heads = [...document.querySelectorAll(".set-cols .set-sec > h2")];
      for (const h of heads) h.parentElement.classList.toggle("collapsed", fold);
      saveJSON("protoart.settingsFolded", fold ? heads.map((h) => h.textContent.trim()) : []);
    },
    async planSort(dir, quiet) {
      // look first: which type folders would be used or made, and how many LoRAs would move
      try {
        const p = await call("lora_sort", { dir, dry_run: true });
        if (quiet && !p.moves) return;
        this.sortPlan = p;
      } catch (e) {
        if (!quiet) this.error = e.message;
      }
    },
    sortPlanText(p) {
      const parts = Object.entries(p.by_type || {}).map(([f, n]) => `${n} ${FAMILY_LABEL[f]}`);
      return parts.length ? parts.join(", ") : "nothing";
    },
    async doSort() {
      const p = this.sortPlan;
      if (!p) return;
      try {
        const r = await call("lora_sort", { dir: p.dir });
        this.sortPlan = null;
        this.loraDirs = await call("lora_dirs_get");
        await this.refreshLoras();
        this.status = `Sorted ${r.moved} LoRA${r.moved === 1 ? "" : "s"} into type folders` +
          (r.unknown.length ? ` · ${r.unknown.length} of unknown type left where they were` : "") +
          (r.errors.length ? ` · ${r.errors.length} couldn't be moved: ${r.errors[0]}` : "");
      } catch (e) {
        this.error = e.message;
      }
    },
    subTypes(dir) {
      // sub-folders of this LoRA folder that have a type (shown under the folder)
      const pre = dir.replace(/\/?$/, "/");
      return Object.entries(this.loraDirs.families || {}).filter(([d]) => d.startsWith(pre))
        .map(([d, f]) => ({ path: d, name: d.slice(pre.length), family: f }));
    },
    loraTypeText(l) {
      const t = (l.kind === "model" ? "base model · " : "LoRA · ") + (FAMILY_LABEL[l.family] || "type unknown");
      if (l.family_set) return `${t} · you picked it` + (l.detected_family && l.detected_family !== l.family ? ` (file looked like ${FAMILY_LABEL[l.detected_family] || "?"})` : "");
      if (l.detected_family && l.detected_family !== l.family) return `${t} · from its folder (file looked like ${FAMILY_LABEL[l.detected_family] || "?"})`;
      return `${t} · read from the file`;
    },
    async setModelFamily(m, family) {
      // a base model file you know is SD 1.5 / SDXL (null = automatic)
      try {
        await call("model_family", { id: m.id, family: family || null });
        this.models = await call("list_models");
        this.loraFix.msg = `${m.label} → ${family ? FAMILY_LABEL[family] : "automatic"}` + (m.id === this.form.model ? " · loads as that next time" : "");
        this.status = this.loraFix.msg;
        if (m.id === this.form.model) this.onModelChange();
      } catch (e) {
        this.error = e.message;
      }
    },
    typeLocked(l) {
      // like a locked layer in Photoshop: a type you picked yourself stays locked until you unlock it
      const v = this.typeLocks[l.key];
      return v === undefined ? !!l.family_set : !!v;
    },
    toggleTypeLock(l) {
      this.typeLocks = { ...this.typeLocks, [l.key]: !this.typeLocked(l) };
      saveJSON("protoart.typeLocks", this.typeLocks);
    },
    async setLoraFamily(l, family) {
      if (this.typeLocked(l)) return;  // 🔒: unlock it first
      if (family) {  // picking a type locks it, so a stray click can't change it later
        this.typeLocks = { ...this.typeLocks, [l.key]: true };
        saveJSON("protoart.typeLocks", this.typeLocks);
      }
      if (l.kind === "model") return this.setModelFamily(l, family);
      // you know better: this LoRA is SD 1.5 / SDXL / FLUX / ERNIE (null = automatic again); optionally move the file
      try {
        const r = await call("lora_family", { name: l.name, family: family || null, move: this.loraFix.move });
        await this.refreshLoras();
        const t = family ? FAMILY_LABEL[family] : "automatic";
        const m = r.moved;
        this.loraFix.msg = `${l.name} → ${t}` + (m && m.to ? ` · moved to ${m.to}` : m && m.skipped && family && this.loraFix.move ? ` · not moved: ${m.skipped}` : "");
        this.status = this.loraFix.msg;
      } catch (e) {
        this.error = e.message;
      }
    },
    fixLoraType(name, unknownOnly) {
      // from Create's LoRA list: open Settings at "Fix a LoRA's type" with this name filled in
      this.loraFix.q = name || "";
      this.loraFix.unknownOnly = !!unknownOnly;
      this.loraOpen = false;
      this.openModelsPanel("fix");
    },
    enhWorkSide() {
      // the short side the picture is worked at (same rule as the engine): "auto" = 512 px for Instant / Turbo / Fast
      // on Macs with less than 16 GB (keeps them out of swap: less than half the work), else 768 px
      const e = this.post.enh, ws = Number(e.work_size);
      if (ws === 512 || ws === 768) return ws;
      const ram = this.sys.ram_gb || 0;
      const quick = e.turbo || e.instant || Number(e.num_inference_steps || 18) <= 10;
      return ram > 0 && ram < 16 && quick ? 512 : 768;
    },
    enhOutSize(w, h, f) {
      // what the enhancer really makes: the picture is first resized so its short side is 512 or 768 px, then × factor
      const side = this.enhWorkSide();
      const s = side / (Math.max(1, Math.min(w, h)) * 8);
      return [Math.floor(Math.floor(w * s) * 8 * f), Math.floor(Math.floor(h * s) * 8 * f)];
    },
    pickEnhSpeed(p) {
      Object.assign(this.post.enh, { solver: p.solver, turbo: !!p.turbo, instant: !!p.instant });
      if (p.steps) this.post.enh.num_inference_steps = p.steps;
    },
    enhSpeedOn(p) {
      const e = this.post.enh;
      if (p.instant || e.instant) return !!p.instant === !!e.instant;
      return !!e.turbo === !!p.turbo && (p.turbo || (e.num_inference_steps === p.steps && e.solver === p.solver));
    },
    async setLoraDirFamily(dir, family) {
      this.loraDirs = await call("lora_dir_family", { dir, family: family || null });
      await this.refreshLoras();
      const n = this.lorasIn(dir);
      this.status = family ? `${n} LoRA${n === 1 ? "" : "s"} in that folder now count as ${FAMILY_LABEL[family]}` : "That folder's LoRA types are read from the files again";
    },
    lorasIn(dir) {
      return this.loras.filter((l) => l.file && l.file.startsWith(dir.replace(/\/?$/, "/"))).length;
    },
    async removeLoraDir(d) {
      if (!(await this.confirmAsk("Stop using this LoRA folder?", d + "\n\nIts LoRAs leave the list; the files stay where they are."))) return;
      this.loraDirs = await call("lora_dirs_set", { dirs: this.loraDirs.dirs.filter((x) => x !== d) });
      await this.refreshLoras();
    },
    addLora() {
      if (!this.loraToAdd) return;
      this.form.loras.push({ name: this.loraToAdd, weight: 1.0 });
      this.loraToAdd = "";
    },
    clipSkipFor(m) {
      // your own choice for this model if you've changed it, otherwise what the model is meant for
      const saved = loadJSON(CLIP_KEY, {})[m.id];
      if (saved) return saved;
      return m.clip_skip || 1;
    },
    setClipSkip(v) {
      this.form.clip_skip = v;
      const m = this.currentModel;
      if (!m) return;
      const all = loadJSON(CLIP_KEY, {});
      if (v === (m.clip_skip || 1)) delete all[m.id]; // back to the recommendation: forget the override
      else all[m.id] = v;
      saveJSON(CLIP_KEY, all);
    },
    isInstalled(m) {
      return m.id.includes(":") || !!m.cached; // file:, tdict:, local:, hf: are yours; built-in ones once downloaded
    },
    downloadJob(m) {
      return this.queue.find((j) => j.cmd === "model_download" && j.model === m.id);
    },
    async downloadModel(m) {
      if (this.downloadJob(m)) return;
      this.status = `Downloading ${m.label}…`;
      const res = await this.runJob("model_download", { model: m.id });
      this.models = await call("list_models");
      if (res) this.status = res.already ? `${m.label} is already on this Mac` : `${m.label} downloaded: it's in Create's model list`;
    },
    useModel(m) {
      // Models tab -> pick it and go to Create (a model that isn't downloaded yet downloads on the first Generate)
      this.selectModel(m.id);
      this.modelFilter = "all";
      this.modelSearch = "";
      this.tab = "create";
      if (!this.isInstalled(m)) this.status = `${m.label} selected · it downloads (${Math.round(m.download_gb)} GB) the first time you press Generate`;
    },
    selectModel(id) {
      if (this.form.model === id) return;
      this.form.model = id;
      this.onModelChange();
    },
    sizeWarning() {
      const m = this.currentModel;
      if (!m) return "";
      const big = Math.max(this.form.width, this.form.height);
      if (m.family === "sd15" && big > 1024)
        return "SD 1.5 is made for 512–768 px. This size is several times slower and looks worse.";
      if (m.family === "sd15" && big > 768 && Math.min(this.form.width, this.form.height) > 768)
        return "SD 1.5 is made for 512–768 px: up to 1024 on one side works (tall or wide pictures), but both sides above 768 can repeat things.";
      if (m.family !== "sd15" && big > 1280 && this.sys.ram_gb && this.sys.ram_gb < 16)
        return "Large sizes need a lot of memory; on this Mac 1024 is the sweet spot.";
      return "";
    },
    flavorName(f) {
      return FLAVOR_LABEL[f] || "";
    },
    familyName(f) {
      return FAMILY_LABEL[f] || f;
    },
    modelSizeText(m) {
      if (m.id.startsWith("file:")) return "your file";
      if (m.id.startsWith("tdict:")) return "DiffusionBee file";
      if (m.id.startsWith("local:")) return "your file";
      if (m.cached) return "downloaded";
      return `${m.download_gb < 10 ? Number(m.download_gb).toFixed(1).replace(/\.0$/, "") : Math.round(m.download_gb)} GB download` + (m.disk_gb ? ` (keeps ${m.disk_gb} GB)` : "");
    },
    async freeOriginal(m) {
      if (this.freeConfirm !== m.id) {
        this.freeConfirm = m.id;
        return;
      }
      this.freeConfirm = null;
      try {
        const res = await call("free_original", { model: m.id });
        this.status = `Removed the original download (freed ${res.freed_gb} GB)`;
        this.models = await call("list_models");
        this.sys = await call("system_info");
      } catch (e) {
        this.error = e.message;
      }
    },

    // -------------------------------------------------------------- tools tab
    openInTools(r) {
      this.tool.source = { path: r.path, image_id: r.id, width: r.width, height: r.height,
                           label: `Library image #${r.id}` };
      this.tool.latest = null;
      this.detail = null;
      this.tab = "tools";
      this.status = `Library image #${r.id} is ready in Tools`;
    },
    toolDrop(e) {
      this.tool.dragOver = false;
      const file = [...(e.dataTransfer?.files || [])].find(isPicture);
      if (file) this.setToolFile(window.bee.pathForFile(file), file.name);
      else this.error = "Drop a picture (PNG, JPEG, WebP, HEIC, TIFF, AVIF, PSD, GIF, BMP or camera RAW).";
    },
    toolPicked(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) this.setToolFile(window.bee.pathForFile(file), file.name);
    },
    // HEIC (iPhone), TIFF, camera RAW, PSD… : macOS converts them to a PNG once (cached); PNG / JPEG / WebP open as they are
    async preparePicture(path) {
      if (!path) return path;
      if (DIRECT_EXT.includes(extOf(path))) return path;
      if (!PICTURE_EXT.includes(extOf(path))) {
        this.error = "That isn't a picture this app can open (PNG, JPEG, WebP, HEIC, TIFF, AVIF, PSD, GIF, BMP or camera RAW).";
        return null;
      }
      this.status = `Opening ${path.split("/").pop()}…`;
      try {
        const r = await call("image_prepare", { path });
        this.status = "";
        return r.path;
      } catch (e) {
        this.error = e.message;
        return null;
      }
    },
    async setToolFile(path, name) {
      if (!path) return;
      this.error = "";
      const usable = await this.preparePicture(path);
      if (!usable) return;
      this.closeAdjust();
      this.tool.latest = null;
      this.tool.source = { path: usable, image_id: null, width: 0, height: 0, label: name || path.split("/").pop() };
    },
    onToolImgLoad(e) {
      if (this.tool.source && !this.tool.source.width) {
        this.tool.source.width = e.target.naturalWidth;
        this.tool.source.height = e.target.naturalHeight;
      }
    },
    toolArgs() {
      const s = this.tool.source;
      return s.image_id != null ? { image_id: s.image_id } : { path: s.path };
    },
    async toolRun(label, args) {
      this.tool.working = { label, started: Date.now() };
      this.tool.latest = null;
      try {
        const rec = await this.runJob("postprocess", { ...this.toolArgs(), ...args });
        if (rec) {
          this.tool.latest = rec;
          this.afterPost(rec);
        }
        return rec;
      } finally {
        this.tool.working = null;
      }
    },
    async toolEnhance() {
      const { seed, ...options } = this.post.enh;
      return this.toolRun("Enhancing & upscaling", { action: "enhance", seed, options });
    },
    async toolRunSelected() {
      const { do_enhance, do_faces } = this.post;
      if (do_enhance && !do_faces) return this.toolEnhance();
      if (do_faces && !do_enhance) return this.toolRestore();
      const { seed, ...options } = this.post.enh;
      return this.toolRun("Enhancing, then restoring faces", { action: "both", seed, options, ...this.faceArgs() });
    },
    async toolRestore() {
      return this.toolRun("Restoring faces", { action: "faces", ...this.faceArgs() });
    },
    // -------------------------------------------------------------- Tools → Patchbay
    patchToggle() {
      this.patch.on = !this.patch.on;
      saveJSON("protoart.patchOn", this.patch.on);
    },
    patchSave() {
      saveJSON("protoart.patch", this.patch.cables);
      saveJSON("protoart.patchV2", true);  // cables you plug in from now on are kept
    },
    patchX(i) {
      return PATCH_X(i);
    },
    patchJack(id, kind) {  // a jack's position in the panel's drawing
      const i = this.patchNodes.findIndex((n) => n.id === id);
      return { x: PATCH_X(i), y: kind === "in" ? PATCH_IN_Y : PATCH_OUT_Y };
    },
    cablePath(a, b) {  // a hanging cable: it sags under its own weight, more the longer it is
      const d = Math.hypot(b.x - a.x, b.y - a.y);
      const sag = Math.min(70, 26 + d * 0.16);
      return `M${a.x},${a.y} C${a.x + (b.x - a.x) * 0.15},${Math.max(a.y, b.y) + sag} ${b.x - (b.x - a.x) * 0.15},${Math.max(a.y, b.y) + sag} ${b.x},${b.y}`;
    },
    patchCablePath(c) {
      return this.cablePath(this.patchJack(c.from, "out"), this.patchJack(c.to, "in"));
    },
    patchPoint(e) {  // the pointer in the panel's drawing coordinates
      const svg = this.$refs.patchSvg;
      if (!svg) return { x: 0, y: 0 };
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const m = svg.getScreenCTM();
      return m ? pt.matrixTransform(m.inverse()) : { x: 0, y: 0 };
    },
    patchDown(id, kind, e) {
      if (this.patch.running) return;
      e.preventDefault();
      const p = this.patchPoint(e);
      if (this.patch.drag && this.patch.drag.armed) {  // second click of click-to-connect
        if (kind === "in") return this.patchConnect(this.patch.drag.from, id, this.patch.drag.c);
        this.patch.drag = null;
        if (kind !== "out") return;
      }
      let from = id, c = this.nextCableColour();
      if (kind === "out") {  // pull the cable out of this OUT (if there's one) and carry it
        const i = this.patch.cables.findIndex((x) => x.from === id);
        if (i >= 0) { c = this.patch.cables[i].c; this.patch.cables.splice(i, 1); }
      } else {  // an IN: pull out the cable plugged in here and carry its loose end
        const i = this.patch.cables.findIndex((x) => x.to === id);
        if (i < 0) return;
        from = this.patch.cables[i].from;
        c = this.patch.cables[i].c;
        this.patch.cables.splice(i, 1);
      }
      this.patch.drag = { from, c, x: p.x, y: p.y, sx: e.clientX, sy: e.clientY, armed: false };
      this.patchSave();
      window.ProtoSound?.play("plug");
    },
    patchFields(type) {  // the rack-unit settings a module's copies have of their own
      return { faces: ["face_strength", "face_model", "face_fidelity"], enhance: ["enh"], depth: ["depth"], detail: ["det"] }[type] || [];
    },
    patchCopyInit(id) {  // a copy starts with the settings its module has now; after that they're its own
      const type = String(id).split("#")[0], P = this.post;
      if (this.patch.copySettings[id]) return;
      const fields = this.patchFields(type);
      if (!fields.length) return;
      const start = Object.fromEntries(fields.map((k) => [k, JSON.parse(JSON.stringify(P[k]))]));
      this.patch.copySettings = { ...this.patch.copySettings, [id]: start };
      this.patchCopySave();
    },
    patchCopySave() {
      saveJSON("protoart.patchCopySettings", this.patch.copySettings);
    },
    patchUnswap() {  // a copy's settings shown in the rack unit go back to the copy; the first module's come back
      const sel = this.patch.sel, st = this.patch.stash;
      if (!sel || !String(sel).includes("#") || !st) return;
      const fields = this.patchFields(String(sel).split("#")[0]);
      const mine = Object.fromEntries(fields.map((k) => [k, JSON.parse(JSON.stringify(this.post[k]))]));
      this.patch.copySettings = { ...this.patch.copySettings, [sel]: mine };
      this.patchCopySave();
      for (const k of fields) this.post[k] = st[k];
      this.patch.stash = null;
    },
    patchSelect(id) {  // pick a module: its rack unit below shows (and edits) that module's settings
      if (id === "src" || id === "result") return;
      const next = this.patch.sel === id ? "" : id;
      this.patchUnswap();
      this.patch.sel = next;
      if (!next) return;
      const type = String(next).split("#")[0];
      if (String(next).includes("#")) {  // a copy: its own settings go into the rack unit while it's picked
        this.patchCopyInit(next);
        const fields = this.patchFields(type);
        this.patch.stash = Object.fromEntries(fields.map((k) => [k, JSON.parse(JSON.stringify(this.post[k]))]));
        const mine = this.patch.copySettings[next] || {};
        for (const k of fields) if (mine[k] !== undefined) this.post[k] = JSON.parse(JSON.stringify(mine[k]));
      }
      this.$nextTick(() => {  // its rack unit: scroll to it and light it up
        const el = document.getElementById("rack-" + type);
        if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
      });
    },
    facePreset(target, p) {  // target: form (Create) or post (Tools)
      target.face_model = "codeformer";
      target.face_fidelity = p.fidelity;
      target.face_strength = p.strength;
    },
    facePresetOn(target, p) {
      return target.face_model !== "restoreformer" && Math.abs(target.face_fidelity - p.fidelity) < 0.001 && Math.abs(target.face_strength - p.strength) < 0.001;
    },
    patchPicked(type) {  // this rack unit belongs to the module picked on the patchbay
      return !!(this.patch.on && this.patch.sel && String(this.patch.sel).split("#")[0] === type);
    },
    patchPickLabel() {
      return (this.patchNodes.find((n) => n.id === this.patch.sel) || {}).label || "";
    },
    patchHide(id) {  // ✕ on a module: a copy is removed; a first module is taken off the rack (＋ Add a module brings it back)
      if (this.patch.running) return;
      if (this.patch.sel === id) this.patchSelect(id);
      if (String(id).includes("#")) return this.patchRemoveModule(id);
      this.patch.hidden = [...new Set([...(this.patch.hidden || []), id])];
      this.patch.cables = this.patch.cables.filter((c) => c.from !== id && c.to !== id);
      saveJSON("protoart.patchHidden", this.patch.hidden);
      this.patchSave();
      this.patch.msg = `Took ${(PATCH_NODES.find((n) => n.id === id) || {}).label || id} off the rack · ＋ Add a module puts it back`;
    },
    patchUnhide(id) {
      this.patch.hidden = (this.patch.hidden || []).filter((x) => x !== id);
      saveJSON("protoart.patchHidden", this.patch.hidden);
    },
    patchProtect(on) {  // 🛡 paint areas the chain mustn't change (they're put back from your picture at the end)
      if (!this.tool.source) return;
      if (on) {
        if (this.adj.open) this.closeAdjust();
        this.mask.data = this.patch.protect && this.patch.protect.for === this.tool.source.path ? this.patch.protect.data : null;
        this.mask.for = this.tool.source.path;
        this.tool.protecting = true;
        this.tool.healing = true;  // the paint area on the left
        this.status = "Paint over what the patch mustn't change, then press ✓ Done";
      } else {
        if (this.mask.data) this.patch.protect = { data: this.mask.data, for: this.tool.source.path };
        this.tool.protecting = false;
        this.tool.healing = false;
        this.mask.data = null;
        this.status = this.patch.protect ? "Protected areas set: they stay as they are when the patch runs" : "";
      }
    },
    patchSavePreset() {
      const name = (this.patch.presetName || "").trim().slice(0, 40);
      if (!name) return;
      this.patchUnswap();
      const keepSel = this.patch.sel;
      this.patch.sel = "";
      const p = { name, cables: JSON.parse(JSON.stringify(this.patch.cables)), extra: JSON.parse(JSON.stringify(this.patch.extra)),
                  hidden: [...(this.patch.hidden || [])],
                  order: [...(this.patch.order || [])], copySettings: JSON.parse(JSON.stringify(this.patch.copySettings)) };
      this.patch.presets = [...this.patch.presets.filter((x) => x.name !== name), p].sort((a, b) => a.name.localeCompare(b.name));
      saveJSON("protoart.patchPresets", this.patch.presets);
      this.patch.preset = name;
      this.patch.presetName = "";
      this.patch.msg = `Signal flow saved as “${name}”`;
      if (keepSel) this.patchSelect(keepSel);
    },
    patchLoadPreset(name) {
      const p = this.patch.presets.find((x) => x.name === name);
      if (!p || this.patch.running) return;
      this.patchUnswap();
      this.patch.sel = "";
      this.patch.hidden = [...(p.hidden || [])];
      saveJSON("protoart.patchHidden", this.patch.hidden);
      this.patch.extra = JSON.parse(JSON.stringify(p.extra || []));
      this.patch.order = [...(p.order || [])];
      this.patch.copySettings = { ...this.patch.copySettings, ...JSON.parse(JSON.stringify(p.copySettings || {})) };
      this.patch.cables = JSON.parse(JSON.stringify(p.cables || []));
      saveJSON("protoart.patchExtra", this.patch.extra);
      saveJSON("protoart.patchOrder", this.patch.order);
      this.patchCopySave();
      this.patchSave();
      this.patch.preset = name;
      this.patch.sel = "";
      this.patch.msg = `Loaded “${name}”`;
    },
    async patchDeletePreset() {
      const name = this.patch.preset;
      if (!name || !(await this.confirmAsk(`Delete the signal flow “${name}”?`, "Only the saved preset is removed; the patchbay stays as it is.", "Delete"))) return;
      this.patch.presets = this.patch.presets.filter((x) => x.name !== name);
      saveJSON("protoart.patchPresets", this.patch.presets);
      this.patch.preset = "";
    },
    patchMoveModule(id, to) {  // put a module in another place, left to right (PICTURE and RESULT stay at the ends)
      const mid = this.patchNodes.slice(1, -1).map((n) => n.id);
      const from = mid.indexOf(id);
      if (from < 0) return;
      to = Math.max(0, Math.min(mid.length - 1, to));
      if (to === from) return;
      mid.splice(from, 1);
      mid.splice(to, 0, id);
      this.patch.order = mid;
      saveJSON("protoart.patchOrder", mid);
      this.patch.msg = "Moved. The order the picture goes through is still set by the cables.";
    },
    patchHeadDown(id, e) {  // drag a module's name plate left or right to move it
      if (this.patch.running || this.patch.drag) return;
      e.preventDefault();
      const p = this.patchPoint(e);
      this.patch.moving = { id, x0: p.x, dx: 0 };
    },
    patchMove(e) {
      if (this.patch.moving) { this.patch.moving.dx = this.patchPoint(e).x - this.patch.moving.x0; return; }
      if (!this.patch.drag) return;
      const p = this.patchPoint(e);
      this.patch.drag.x = p.x;
      this.patch.drag.y = p.y;
    },
    patchUp(e) {
      const mv = this.patch.moving;
      if (mv) {
        this.patch.moving = null;
        const i = this.patchNodes.findIndex((n) => n.id === mv.id) - 1;
        if (Math.abs(mv.dx) > 20) this.patchMoveModule(mv.id, i + Math.round(mv.dx / 140));
        else this.patchSelect(mv.id);  // a click, not a drag: show this module's settings
        return;
      }
      const d = this.patch.drag;
      if (!d || d.armed) return;
      const el = document.elementFromPoint(e.clientX, e.clientY);
      const jack = el && el.closest && el.closest("[data-in]");
      if (jack) return this.patchConnect(d.from, jack.getAttribute("data-in"), d.c);
      if (Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < 6) {
        d.armed = true;  // a click, not a drag: the cable follows the pointer until you click an IN
        this.patch.msg = "Now click an IN jack to plug the cable in (click anywhere else to drop it)";
        return;
      }
      this.patch.drag = null;  // let go in mid-air: the cable is put away
    },
    patchCancel() {
      this.patch.moving = null;
      if (this.patch.drag) { this.patch.drag = null; this.patch.msg = ""; }
    },
    patchConnect(from, to, c) {
      this.patch.drag = null;
      this.patch.msg = "";
      if (from === to) return;
      if (from === "src" && to === "result") return (this.patch.msg = "Patch at least one module between PICTURE and RESULT.");
      if (from.split("#")[0] === "vector" && to !== "result") return (this.patch.msg = "VECTOR makes an SVG drawing, so it can only go to RESULT (it has to be last).");
      // no loops: the module being patched into mustn't already lead back here
      let at = to;
      for (let n = 0; n < 10; n++) {
        const nx = this.patch.cables.find((x) => x.from === at);
        if (!nx) break;
        if (nx.to === from) return (this.patch.msg = "That would make a loop: the picture would go round for ever.");
        at = nx.to;
      }
      this.patch.cables = this.patch.cables.filter((x) => x.to !== to && x.from !== from);  // one cable per jack
      this.patch.cables.push({ from, to, c: c ?? this.nextCableColour() });
      this.patchSave();
      window.ProtoSound?.play("plug");
    },
    patchUnplug(i) {
      if (this.patch.running) return;
      this.patch.cables.splice(i, 1);
      this.patchSave();
    },
    nextCableColour() {
      const used = this.patch.cables.map((x) => x.c);
      for (let k = 0; k < CABLE_COLOURS.length; k++) if (!used.includes(k)) return k;
      return this.patch.cables.length % CABLE_COLOURS.length;
    },
    cableColour(c) {
      return CABLE_COLOURS[(c || 0) % CABLE_COLOURS.length];
    },
    patchReset(kind) {
      if (this.patch.running) return;
      this.patch.cables = kind === "example" ? JSON.parse(JSON.stringify(PATCH_EXAMPLE)) : [];
      if (kind === "reset") {
        this.patchUnswap(); this.patch.sel = "";
        this.patch.hidden = []; saveJSON("protoart.patchHidden", []); this.patch.order = []; saveJSON("protoart.patchOrder", []); }  // modules back in their places
      this.patch.msg = "";
      this.patchSave();
    },
    patchAddModule(type) {
      if (!type) return;
      if (type.startsWith("back:")) {  // a first module you took off the rack
        this.patchUnhide(type.slice(5));
        this.patch.msg = `Put ${(PATCH_NODES.find((n) => n.id === type.slice(5)) || {}).label} back on the rack`;
        return;
      }
      if ((this.patch.hidden || []).includes(type)) { this.patchUnhide(type); this.patch.msg = `Put ${(PATCH_NODES.find((n) => n.id === type) || {}).label} back on the rack`; return; }
      const n = 1 + (this.patch.extra || []).filter((x) => x.type === type).length + 1;
      let id = `${type}#${n}`;
      for (let k = n; this.patch.extra.some((x) => x.id === id); k++) id = `${type}#${k + 1}`;
      this.patch.extra = [...this.patch.extra, { id, type }];
      saveJSON("protoart.patchExtra", this.patch.extra);
      this.patchCopyInit(id);
      this.patch.sel = id;  // show the new copy's own settings
      this.patch.msg = `Added ${(PATCH_NODES.find((x) => x.id === type) || {}).label} ${id.split("#")[1]}: cable it in where you want it to run again`;
      window.ProtoSound?.play("plug");
    },
    patchRemoveModule(id) {
      const ex0 = this.patch.extra, target = id || (ex0.length ? ex0[ex0.length - 1].id : "");
      if (this.patch.sel && this.patch.sel === target) this.patchSelect(target);  // its settings back where they belong
      const ex = this.patch.extra;
      if (!ex.length) return;
      id = id || ex[ex.length - 1].id;
      this.patch.extra = ex.filter((x) => x.id !== id);
      this.patch.cables = this.patch.cables.filter((c) => c.from !== id && c.to !== id);
      saveJSON("protoart.patchExtra", this.patch.extra);
      this.patchSave();
    },
    patchStepLabel(id) {
      id = String(id).split("#")[0];
      return { heal: "Removing what you painted", faces: "Restoring faces", enhance: "Enhancing & upscaling",
               depth: "Making a depth map", vector: "Vectorizing", detail: "Adding detail with a LoRA" }[id] || id;
    },
    async patchRun() {
      const steps = this.patchChain.steps;
      if (!this.tool.source) return (this.error = "Drop a picture into Tools first.");
      if (!steps.length) return (this.error = "Patch a cable from PICTURE's OUT into a module's IN first.");
      if (steps.some((x) => String(x).split("#")[0] === "heal") && !(this.mask.data && this.mask.for === this.tool.source.path)) {
        this.toolPaintRemove(true);
        return (this.error = "REMOVE is patched in: paint over what to remove on the picture (Remove objects → 🖌 Paint is now on), then press ▶ Run patch again.");
      }
      if (steps.some((x) => String(x).split("#")[0] === "detail") && !["sd15", "sdxl"].includes(this.family))
        return (this.error = "DETAIL redraws the picture lightly with an SD 1.5 or SDXL model and a LoRA: pick an SD 1.5 or SDXL model in Create first.");
      if (this.tool.protecting) this.patchProtect(false);  // finish painting the protected areas
      this.error = "";
      const mask = this.mask.for === this.tool.source.path ? this.mask.data : null;
      const reselect = this.patch.sel;
      this.patchUnswap();  // copies' settings shown in a rack unit are saved before running
      this.patch.sel = "";
      let cur = this.toolArgs(), curPath = this.tool.source.path;
      const origin = { path: this.tool.source.path, id: this.tool.source.image_id };
      let last = null;
      this.patch.steps = steps.length;
      this.patch.stopped = false;
      this.tool.latest = null;
      try {
        for (let k = 0; k < steps.length; k++) {
          const node = steps[k], id = String(node).split("#")[0];  // a copy you added runs like its module
          this.patch.running = node;
          this.patch.step = k + 1;
          this.tool.working = { label: `Patch ${k + 1}/${steps.length} · ${this.patchStepLabel(id)}`, started: Date.now() };
          let rec = null;
          if (id === "heal") {
            rec = await this.runJob("heal", { ...cur, mask, method: this.form.heal_method || "lama", from_tools: true });
          } else if (id === "faces") {
            const c = this.patch.copySettings[node];  // a copy you added has its own settings
            rec = await this.runJob("postprocess", { ...cur, action: "faces", ...(c ? { strength: c.face_strength, face_model: c.face_model, face_fidelity: c.face_fidelity } : this.faceArgs()) });
          } else if (id === "enhance") {
            const c = this.patch.copySettings[node];
            const { seed, ...options } = (c && c.enh) || this.post.enh;
            rec = await this.runJob("postprocess", { ...cur, action: "enhance", seed, options });
          } else if (id === "depth") {
            const dset = (this.patch.copySettings[node] && this.patch.copySettings[node].depth) || this.post.depth;
            rec = await this.runJob("postprocess", { ...cur, action: "depth", depth_model: dset.model, style: dset.style });
          } else if (id === "detail") {
            const d = (this.patch.copySettings[node] && this.patch.copySettings[node].det) || this.post.det;
            rec = await this.runJob("generate", this.detailArgs(d, curPath));
            if (Array.isArray(rec)) rec = rec[rec.length - 1] || null;  // generate answers with a list of pictures
          } else if (id === "vector") {
            const v = this.post.vec;
            rec = await this.runJob("vectorize", { ...cur, colors: v.colors, detail: v.detail, smooth: v.smooth });
            if (rec) this.vecResult = rec;
          }
          if (!rec || this.patch.stopped) {
            if (!this.error) this.status = `Patch stopped at step ${k + 1} (${this.patchStepLabel(id).toLowerCase()})`;
            break;
          }
          if (id !== "vector") {
            last = rec;
            this.afterPost(rec);
            cur = rec.id != null ? { image_id: rec.id } : { path: rec.path };
            curPath = rec.path || curPath;
          }
          if (k === steps.length - 1) this.status = `Patch done: ${steps.length} step${steps.length === 1 ? "" : "s"}` + (id === "vector" ? " (the SVG is in Vectorize)" : "");
        }
        const pr = this.patch.protect;
        if (last && last.id != null && pr && pr.for === origin.path && !this.patch.stopped) {
          // 🛡 the areas you protected come back from your picture
          const merged = await call("protect_merge", { image_id: last.id, original: origin.path, original_id: origin.id, mask: pr.data });
          if (merged) { last = merged; this.afterPost(merged); this.status += " · protected areas kept"; }
        }
      } catch (e) {
        this.error = e.message;
      } finally {
        if (reselect && !this.patch.sel) this.patchSelect(reselect);
        this.tool.working = null;
        this.patch.running = "";
        if (last) {
          this.tool.latest = last;
          if (this.tool.healing) { this.tool.healing = false; this.mask.data = null; }
        }
      }
    },
    detailArgs(d, path) {  // 🔎 DETAIL: a light image-to-image pass with the Create model and one LoRA
      const f = JSON.parse(JSON.stringify(this.form));
      this.cleanExtras(f);
      delete f.hires; delete f.control; delete f.outpaint; delete f.inpaint; delete f.face_detailer;
      const lora = d.lora && this.compatibleLoras.some((l) => l.name === d.lora) ? [{ name: d.lora, weight: Number(d.weight) || 0.8 }] : [];
      return { ...f, init_image: path, i2i_mode: "change", i2i_use_init: true, use_init_size: true, strength: Number(d.strength) || 0.35,
               steps: Number(d.steps) || 20, num_images: 1, loras: lora, fast: false, enhance_prompt: false, from_tools: true,
               prompt: (this.form.prompt || "").trim() || "highly detailed, sharp focus, intricate details" };
    },
    patchStop() {
      this.patch.stopped = true;
      if (this.toolJob) this.cancelJob(this.toolJob.id);
    },
    faceArgs() {
      // Tools / picture detail: strength + which face model (RestoreFormer or CodeFormer) + CodeFormer fidelity
      return { strength: this.post.face_strength, face_model: this.post.face_model, face_fidelity: this.post.face_fidelity };
    },
    async copyPromptText() {
      try { await navigator.clipboard.writeText(this.tool.promptText || ""); } catch { /* clipboard blocked: nothing to do */ }
      this.tool.promptCopied = true;
      setTimeout(() => (this.tool.promptCopied = false), 1600);
    },
    promptToI2i() {  // Tools → Image → prompt → picture + prompt in Image to image
      const text = this.tool.promptText;
      this.useAsStart(this.tool.source);
      this.form.prompt = text;
      this.status = "Picture and prompt are in Image to image: change the prompt and press Generate";
    },
    async toolToPrompt(mode) {
      this.tool.promptText = "";
      const res = await this.runJob("interrogate", { path: this.tool.source.path, mode, detail: "detailed" });
      if (!res) return;
      this.form.prompt = res.prompt;
      this.tool.promptText = res.prompt;
      this.status = "Prompt ready in the Create tab" + (res.model ? " (" + res.model + ")" : "");
    },

    downloadText(m) {
      return `${m.download_gb < 10 ? Number(m.download_gb).toFixed(1).replace(/\.0$/, "") : Math.round(m.download_gb)} GB download`;
    },
    roughTokens(text) {
      // CLIP uses ~1.3 tokens per English word; good enough for a warning.
      return Math.ceil((text || "").trim().split(/\s+/).filter(Boolean).length * 1.3);
    },
    modelLabel(id) {
      const m = this.models.find((m) => m.id === id);
      if (m) return m.label;
      // a model that's no longer listed (or your own file): show its file name, not the whole path
      const s = String(id || "");
      return s.replace(/^(local|file|hub):/, "").split("/").pop() || s;
    },
    // Create: take a picture off this screen (it stays in the Library)
    hideFromSession(r) {
      this.session = this.session.filter((x) => x.id !== r.id);
    },
    clearSession() {  // Create's "clear all": Image to image's results stay where they are
      this.session = this.session.filter((r) => I2I_KINDS.includes(r.kind));
    },
    clearI2iResults() {  // Image to image's "clear all": takes its results off the screen (they stay in the Library)
      this.session = this.session.filter((r) => !I2I_KINDS.includes(r.kind));
      this.i2iCompare = false;
    },

    // -------------------------------------------------------------- generate
    chainModels() {  // models a chain step can use: anything that can start from a picture (FLUX.2 Klein edits it with words)
      return this.sortModels(this.models.filter((m) => this.isInstalled(m) && !m.inpaint &&
        (["sd15", "sdxl", "anima"].includes(m.family) && m.engine !== "mlx" || (m.engine === "mlx" && ["flux", "ernie", "flux2"].includes(m.family)))));
    },
    chainAdd() {
      if (this.chain.steps.length >= 4) return;
      const used = new Set(this.chain.steps.map((x) => x.model));
      const next = this.chainModels().find((m) => !used.has(m.id)) || this.chainModels()[0];
      this.chain.steps.push({ model: next ? next.id : "", strength: 0.4, lora: "", lora_weight: 1, extra: "", result: "", q: "" });
      this.chainSave();
    },
    chainSave() {  // (the results are this session's only)
      saveJSON("protoart.chain", this.chain.steps.map(({ result, q, ...x }) => x));
    },
    chainModelOf(st) { return this.models.find((m) => m.id === st.model) || null; },
    chainFamilyLabel(st) { const m = this.chainModelOf(st); return m ? (FAMILY_LABEL[m.family] || m.family) : "pick a model"; },
    chainLoras(st, all = false) {  // LoRAs made for this step's model type (narrowed by the step's search box)
      const m = this.chainModelOf(st);
      if (!m) return [];
      const list = this.loras.filter((l) => l.family === m.family && !(this.hideNsfw && l.nsfw) && !(this.opts.hide_people && l.people));
      const q = (st.q || "").trim().toLowerCase();
      if (all || !q) return list;
      return list.filter((l) => [l.name, l.file, ...(this.activeTriggers(l.name) || [])]
        .some((v) => typeof v === "string" && v.toLowerCase().includes(q)));
    },
    chainModelChanged(st) {
      if (st.lora && !this.chainLoras(st).some((l) => l.name === st.lora)) st.lora = "";  // a LoRA for another model type
      this.chainSave();
    },
    async chainLoraPicked(st) {
      if (st.lora) {
        const w = this.loraDefaultWeight(st.lora);
        if ((this.chainModelOf(st) || {}).family === "flux2") { st.lora_weight = w ?? 1; this.chainSave(); return; }  // Klein: extra words are the instruction
        st.lora_weight = w ?? 1;
        try {  // its trigger words go in the step's extra words (when it has any and they aren't there yet)
          await this.fetchTriggers(st.lora);
          const words = this.activeTriggers(st.lora).filter((x) => typeof x === "string");
          const add = words.slice(0, 3).filter((x) => !(st.extra || "").toLowerCase().includes(x.toLowerCase()));
          if (add.length) st.extra = [st.extra, ...add].filter(Boolean).join(", ");
        } catch (e) { /* no trigger words known */ }
      }
      this.chainSave();
    },
    chainMove(k, d) {
      const s = this.chain.steps, j = k + d;
      if (j < 0 || j >= s.length) return;
      [s[k], s[j]] = [s[j], s[k]];
      this.chainSave();
    },
    async runChain() {  // 🔗 the picture goes through each model in turn (e.g. SD 1.5 → SDXL → FLUX)
      const f = this.form;
      if (!f.init_image) return (this.error = "Add a starting picture first (1 · Picture), or send one from Create.");
      const steps = this.chain.steps.filter((x) => x.model);
      if (!steps.length) return (this.error = "Add at least one model to the chain.");
      if (!(f.prompt || "").trim()) return (this.error = "Describe the picture in the prompt: each model in the chain uses it.");
      this.error = "";
      Object.assign(this.chain, { running: true, stopped: false, at: 0, done: 0 });
      this.chain.steps.forEach((x) => { x.result = ""; });
      let path = f.init_image, last = null;
      try {
        for (let k = 0; k < steps.length; k++) {
          if (this.chain.stopped) break;
          this.chain.at = k;
          const st = steps[k], m = this.models.find((x) => x.id === st.model);
          if (!m) throw new Error("A model in the chain isn't installed any more.");
          const d = m.defaults || {};
          const args = JSON.parse(JSON.stringify(f));
          this.cleanExtras(args);
          delete args.hires; delete args.control; delete args.outpaint; delete args.inpaint; delete args.face_detailer; delete args.ip_adapter;
          const extra = (st.extra || "").trim();
          Object.assign(args, { model: m.id, init_image: path, i2i_mode: "change", i2i_use_init: true, use_init_size: true,
                                strength: Number(st.strength) || 0.4, steps: d.steps || args.steps, cfg: d.cfg ?? args.cfg, sampler: "",
                                num_images: 1, fast: false, enhance_prompt: false, var_strength: 0,
                                prompt: extra ? `${args.prompt.trim()}, ${extra}` : args.prompt,
                                // each step uses its own LoRA (if any): a character LoRA in the SD 1.5 step, none in FLUX…
                                loras: st.lora ? [{ name: st.lora, weight: Number(st.lora_weight ?? 1) }] : [],
                                chain_step: k + 1, chain_len: steps.length });
          this.status = `🔗 Chain step ${k + 1} of ${steps.length}: ${m.label}`;
          let rec;
          if (m.family === "flux2") {
            // FLUX.2 Klein redraws the picture by following an instruction (Edit with words): the step's extra words
            // are the change; without any, it redraws the picture in the prompt's description
            const words = extra || `Redraw this picture as: ${f.prompt.trim()}. Keep the composition, pose and layout.`;
            rec = await this.runJob("edit_words", { path, prompt: words, seed: f.seed, loras: args.loras });
          } else {
            rec = await this.runJob("generate", args);
          }
          if (Array.isArray(rec)) rec = rec[rec.length - 1] || null;
          if (!rec) break;
          last = rec;
          path = rec.path;
          st.result = rec.path;
          this.chain.done = k + 1;
        }
        if (last) this.status = this.chain.stopped ? "Chain stopped: the last finished step is in the Library" : `🔗 Chain done: ${steps.length} model${steps.length === 1 ? "" : "s"}`;
      } catch (e) {
        this.error = e.message;
      } finally {
        Object.assign(this.chain, { running: false, at: -1 });
      }
    },
    chainStop() {
      this.chain.stopped = true;
      const j = this.runningJob;
      if (j) this.cancelJob(j.id);
    },
    whichOne(goal, inI2i = false) {  // "Which one should I use?": set up the right tool for the goal
      if (goal === "face" || goal === "plus") {
        this.form.ip_adapter.kind = goal;
        if (!this.form.ip_adapter.scale || this.form.ip_adapter.scale < 0.5) this.form.ip_adapter.scale = goal === "face" ? 0.7 : 0.6;
        if (this.tab !== "create") this.tab = "create";
        this.status = goal === "face" ? "Image prompt → Face: drop a close-up photo of the person, then describe the new picture"
          : "Image prompt → Stronger: drop the picture whose style, outfit or colours you want";
        this.$nextTick(() => document.querySelector(".ipa")?.scrollIntoView({ behavior: "smooth", block: "center" }));
        return;
      }
      if (!inI2i) this.openI2I();
      this.setI2iMode && this.form.i2i_mode !== "change" && this.setI2iMode("change");
      const pose = this.family === "sdxl" ? "union:openpose" : "openpose";
      if (!this.form.control.type) this.form.control.type = this.controlTypes.some((t) => t.id === pose) ? pose : "";
      this.status = "ControlNet picked: OpenPose copies a pose; for a drawing pick Line art or Scribble, for a silhouette or layout Canny or Depth";
    },
    smallChange() {  // 🎲 Small change: the same picture (seed kept) with small details changed
      const last = this.session[0];
      if (this.form.seed === -1 && last && last.seed != null) this.form.seed = last.seed;
      if (this.form.seed === -1) this.form.seed = Math.floor(Math.random() * 2 ** 31);
      this.form.var_seed = Math.floor(Math.random() * 2 ** 31);
      if (!this.form.var_strength) this.form.var_strength = 0.1;
      this.status = `Small change ready: seed ${this.form.seed} kept, variation seed ${this.form.var_seed} at ${this.form.var_strength.toFixed(2)}. Press Generate.`;
    },
    createArgs(overrides) {
      // everything the Create tab sends; pictures, masks and ControlNet live in the Image to image tab
      const args = JSON.parse(JSON.stringify(this.form));
      args.init_image = "";
      delete args.control;
      delete args.control_image;
      delete args.outpaint;
      delete args.inpaint;
      this.cleanExtras(args);
      return Object.assign(args, overrides || {});
    },
    cleanExtras(args) {
      // send only the extras that are switched on (and that this model can use)
      const sd = ["sd15", "sdxl"].includes(this.family) && this.currentModel?.engine !== "mlx";
      if (!sd || !args.hires?.on) delete args.hires;
      if (!sd || !args.face_detailer?.on) delete args.face_detailer;
      if (!sd || !args.ip_adapter?.image) delete args.ip_adapter;
      // regions: SD 1.5 / SDXL inside the model; FLUX.1 and ERNIE (fast engine) by parts (see regions_by_parts)
      const byParts = this.currentModel?.engine === "mlx" && ["flux", "ernie", "flux2"].includes(this.family);
      if (!(sd || byParts) || !args.regions?.on) delete args.regions;
      if (!sd) delete args.seamless;
      return args;
    },
    // -------------------------------------------------------------- "this will be slow on this Mac"
    // Before a picture starts: settings that need more memory than this Mac has get a short warning with a one-click
    // faster choice. 8 GB Macs see it for big ControlNet / SDXL jobs, 16 GB Macs only for the heaviest ones, and
    // Macs with 24 GB or more never. Each warning can be switched off ("Don't warn me about this again").
    slowWarnings(mode) {
      const ram = this.sys.ram_gb || 0;
      if (!ram || ram >= 24) return [];
      const low = ram < 12, f = this.form, m = this.currentModel || {};
      if (m.engine === "mlx" || !["sd15", "sdxl"].includes(m.family)) return [];
      // the size it will really be: Image to image with "Use the picture's size" uses the picture's (capped)
      let W = f.width, H = f.height;
      if (mode === "i2i" && f.use_init_size && this.initDims && this.initDims.w) {
        const cap = m.family === "sdxl" ? 1024 : 768, k = Math.min(1, cap / Math.max(this.initDims.w, this.initDims.h));
        W = Math.round(this.initDims.w * k); H = Math.round(this.initDims.h * k);
      }
      const px = W * H, out = [];
      const cn = mode === "i2i" && f.i2i_mode === "change" && f.control.type;
      const fit = (side) => {  // the same shape, its longer side at `side`
        const k = side / Math.max(W, H);
        return [Math.max(256, Math.round((W * k) / 64) * 64), Math.max(256, Math.round((H * k) / 64) * 64)];
      };
      const sizeFix = (side) => {
        const [w, h] = fit(side);
        return { label: `Use ${w} × ${h} (much faster)`, apply: () => { f.width = w; f.height = h; f.use_init_size = false; } };
      };
      if (cn && m.family === "sdxl")
        out.push({ id: "cn_sdxl", text: `An SDXL ControlNet with an SDXL model needs more memory than ${low ? "an 8 GB" : "a 16 GB"} Mac has free, so each step can take minutes.`,
                   fixes: px > (low ? 600e3 : 1.1e6) ? [sizeFix(low ? 768 : 1024)] : [] });
      else if (cn && px > (low ? 350e3 : 1.0e6))
        out.push({ id: "cn_big", text: `A ControlNet at ${W} × ${H} on ${low ? "an 8 GB" : "a 16 GB"} Mac: the ControlNet roughly doubles each step's work, so this can take a minute or more a step.`,
                   fixes: [sizeFix(low ? 512 : 768)] });
      if (low && m.family === "sdxl" && /fp16|full/i.test(m.id + " " + (m.label || ""))) {
        const q = this.installedModels.find((x) => x.family === "sdxl" && /int8|8-bit/i.test(x.id + " " + x.label));
        out.push({ id: "sdxl_full", text: "The full-precision SDXL model is about 7 GB: on an 8 GB Mac most of it ends up on the disk, which is very slow.",
                   fixes: q ? [{ label: `Use ${q.label} (half the memory)`, apply: () => this.selectModel(q.id) }] : [] });
      }
      if (!cn && px > (m.family === "sdxl" ? (low ? 1.2e6 : 2.2e6) : (low ? 0.8e6 : 1.6e6)))
        out.push({ id: "big_size", text: `${W} × ${H} is bigger than this model was made for and more than ${low ? "an 8 GB" : "a 16 GB"} Mac handles quickly.`,
                   fixes: [sizeFix(m.family === "sdxl" ? 1024 : 768)] });
      if (low && mode !== "i2i" && f.hires && f.hires.on && m.family === "sdxl")
        out.push({ id: "hires_sdxl", text: "Hi-res fix with SDXL on an 8 GB Mac redraws a picture of over 2 million pixels: it can take many minutes.",
                   fixes: [{ label: "Switch Hi-res fix off", apply: () => { f.hires.on = false; } }] });
      if (low && (f.num_images || 1) > 4 && m.family === "sdxl" && mode !== "i2i")
        out.push({ id: "many_sdxl", text: `${f.num_images} SDXL pictures in one go on an 8 GB Mac will keep it busy for a long time (the queue can be paused or stopped).`, fixes: [] });
      const muted = loadJSON("protoart.slowWarnOff", {});
      return out.filter((w) => !muted[w.id]);
    },
    async slowCheck(mode) {  // true = go ahead (nothing to warn about, or "Go ahead anyway" / a fix was picked)
      const list = this.slowWarnings(mode);
      if (!list.length) return true;
      return new Promise((resolve) => {
        this.slowWarn = { list, fixes: list.flatMap((w) => w.fixes), resolve, mute: false, ram: this.sys.ram_gb };
      });
    },
    resetSlowWarnings() {
      saveJSON("protoart.slowWarnOff", {});
      this.status = "Slow-down warnings are on again";
    },
    answerSlow(choice) {  // choice: a fix, "go" or null (cancel)
      const w = this.slowWarn;
      if (!w) return;
      this.slowWarn = null;
      if (w.mute && choice) {
        const muted = loadJSON("protoart.slowWarnOff", {});
        for (const x of w.list) muted[x.id] = true;
        saveJSON("protoart.slowWarnOff", muted);
      }
      if (choice && choice !== "go") choice.apply();
      w.resolve(!!choice);
    },
    dropMismatchedLoras() {
      // a LoRA made for another model type (a FLUX.1 LoRA with FLUX.2, an SD 1.5 one with SDXL…) can't be used: it's
      // left out of this picture with a note, instead of the whole picture failing (presets and older settings
      // could bring one in). LoRAs whose type isn't known are left alone.
      const fam = this.family, names = { flux: "FLUX.1", flux2: "FLUX.2", sd15: "SD 1.5", sdxl: "SDXL", ernie: "ERNIE" };
      const bad = this.form.loras.filter((x) => {
        const l = this.loras.find((y) => y.name === x.name);
        return l && l.family && l.family !== "unknown" && l.family !== fam;
      });
      if (!bad.length) return;
      this.form.loras = this.form.loras.filter((x) => !bad.includes(x));
      const what = bad.map((x) => { const l = this.loras.find((y) => y.name === x.name); return `${x.name} (${names[l.family] || l.family})`; });
      this.status = `Left out ${what.join(", ")}: made for another model type than ${this.familyLabel}`;
    },
    async generate() {
      if (!(await this.slowCheck("create"))) return;
      this.dropMismatchedLoras();
      if (this.rnd.every) this.randomPrompt();  // 🎲 a new random prompt each time
      if (this.loraRand.every && this.randomPool.length) await this.randomLoras(true);
      const res = await this.runJob("generate", this.createArgs());
      if (res) this.models = await call("list_models"); // refresh "downloaded" flags
    },

    // -------------------------------------------------------------- wildcards & prompt lists
    async loadWildcards() {
      try {
        const r = await call("wildcards_list");
        this.wild.files = r.files;
        this.wild.folder = r.folder;
      } catch (e) {
        this.error = e.message;
      }
    },
    shopShut(id) {
      return !!this.shopFolds[id];
    },
    shopFoldToggle(id) {
      this.shopFolds = { ...this.shopFolds, [id]: !this.shopFolds[id] };
      saveJSON("protoart.shopFolds", this.shopFolds);
    },
    detailStep(d) {  // ‹ / › in a picture's details: the previous / next picture
      const list = this.detailList, i = this.detailIndex + d;
      if (i < 0 || i >= list.length) return;
      this.openDetail(list[i]);
    },
    cmpClamp() {  // don't let the picture slide out of its frame
      const f = document.querySelector(".cmp-frame");
      if (!f || this.cmp.z <= 1) { Object.assign(this.cmp, { x: 0, y: 0 }); return; }
      const mx = (f.clientWidth * (this.cmp.z - 1)) / 2, my = (f.clientHeight * (this.cmp.z - 1)) / 2;
      this.cmp.x = Math.max(-mx, Math.min(mx, this.cmp.x));
      this.cmp.y = Math.max(-my, Math.min(my, this.cmp.y));
    },
    cmpZoomTo(z, px = 0, py = 0) {  // px, py: the point to keep still, from the frame's centre
      const c = this.cmp, nz = Math.max(1, Math.min(16, z));
      c.x = px - ((px - c.x) * nz) / c.z;
      c.y = py - ((py - c.y) * nz) / c.z;
      c.z = nz;
      this.cmpClamp();
    },
    cmpZoomBy(f, e) {
      let px = 0, py = 0;
      if (e && e.currentTarget) { const r = e.currentTarget.getBoundingClientRect(); px = e.clientX - r.left - r.width / 2; py = e.clientY - r.top - r.height / 2; }
      this.cmpZoomTo(this.cmp.z * f, px, py);
    },
    cmpWheel(e) {  // mouse wheel / trackpad pinch over any picture zooms all of them around the pointer
      const r = e.currentTarget.getBoundingClientRect();
      const k = e.ctrlKey ? 0.01 : 0.0018;  // a pinch arrives as small ctrl+wheel steps
      this.cmpZoomTo(this.cmp.z * Math.exp(-e.deltaY * k), e.clientX - r.left - r.width / 2, e.clientY - r.top - r.height / 2);
    },
    cmpDown(e) {
      if (this.cmp.z <= 1 || e.button !== 0) return;
      e.preventDefault();
      const c = this.cmp, sx = e.clientX - c.x, sy = e.clientY - c.y;
      c.drag = true;
      const move = (ev) => { c.x = ev.clientX - sx; c.y = ev.clientY - sy; this.cmpClamp(); };
      const up = () => { c.drag = null; removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
      addEventListener("pointermove", move);
      addEventListener("pointerup", up);
    },
    cmpOneToOne() {  // the zoom that shows the first picture's real pixels
      const r = (this.compareView || [])[0], f = document.querySelector(".cmp-frame img");
      if (!r || !f || !f.clientWidth) return 2;
      const shown = Math.min(f.clientWidth / (f.naturalWidth || r.width), f.clientHeight / (f.naturalHeight || r.height));
      return Math.max(1, 1 / (shown || 1));
    },
    cmpReset() {
      Object.assign(this.cmp, { z: 1, x: 0, y: 0 });
    },
    openCompareView() {
      this.cmpReset();
      const ids = this.selected.slice(0, 6);
      const byId = new Map(this.libShown.map((r) => [r.id, r]));
      this.compareView = ids.map((id) => byId.get(id)).filter(Boolean);
      if (this.selected.length > 6) this.status = "Showing the first 6 selected pictures";
    },
    openHelpAt(h) {  // a search result → open its section and scroll to it
      this.helpOpen = h.tabId;
      this.$nextTick(() => {
        const el = [...document.querySelectorAll(".acc-item.open .acc-sec h4")].find((x) => x.textContent.trim() === h.sec);
        (el || document.querySelector(".acc-item.open"))?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    },
    i2iShut(k) {
      return !!this.i2iFolds[k];
    },
    i2iFoldToggle(k, e) {
      if (e && e.target.closest("button, a, input, select")) return;  // "clear all" etc. in the title still work
      this.i2iFolds = { ...this.i2iFolds, [k]: !this.i2iFolds[k] };
      saveJSON("protoart.i2iFolds", this.i2iFolds);
    },
    i2iFoldAll(shut) {  // Image to image → collapse all / expand all
      const all = {};
      for (const k of ["picture", "chain", "cn", "prompt", "steps", "loras"]) all[k] = shut;
      this.i2iFolds = all;
      saveJSON("protoart.i2iFolds", this.i2iFolds);
    },
    togglePtool(k) {  // one prompt-tool panel open at a time (Wildcards keeps its own switch)
      if (k === "wild") {
        this.ptool = "";
        return this.openWildcards();
      }
      this.wild.open = false;
      this.ptool = this.ptool === k ? "" : k;
    },
    async openWildcards() {
      this.wild.open = !this.wild.open;
      if (this.wild.open) await this.loadWildcards();
    },
    async previewWildcards() {
      try {
        this.wild.preview = await call("wildcards_preview", { text: this.form.prompt, count: 4 });
      } catch (e) {
        this.error = e.message;
      }
    },
    insertPromptText(t) {
      const p = (this.form.prompt || "").replace(/\s+$/, "");
      this.form.prompt = p ? (/[,\s]$/.test(p) ? p + " " : p + ", ") + t : t;
    },
    onBatchFile(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file) return;
      const r = new FileReader();
      r.onload = () => { this.batch.text = String(r.result || ""); };
      r.readAsText(file);
    },
    queueBatch() {
      const lines = this.batchLines;
      const base = (this.form.prompt || "").trim();
      for (const line of lines) {
        const prompt = this.batch.keepPrompt && base ? `${line}, ${base}` : line;
        this.runJob("generate", this.createArgs({ prompt })); // queued; each resolves when its job is done
      }
      this.status = `Queued ${lines.length} job${lines.length === 1 ? "" : "s"} from the prompt list`;
      this.batch.open = false;
    },

    // -------------------------------------------------------------- LoRA randomizer
    async randomLoras(quiet) {
      const r = this.loraRand;
      // a new mix replaces every LoRA in the list (the ones added by hand or by a preset too: before, only the
      // randomizer's own picks changed, so the first LoRA never did); the trigger words it added go too
      this.form.loras = [];
      for (const w of r.words) if (this.inPrompt(w)) this.form.prompt = this.removeWord(this.form.prompt, w);
      r.picked = [];
      r.words = [];
      const pool = [...this.randomPool];
      if (!pool.length) {
        if (!quiet) this.status = `No ${this.familyLabel} LoRAs to pick from`;
        return;
      }
      const lo = Math.max(1, Math.min(r.min, r.max)), hi = Math.max(lo, r.max);
      const count = Math.min(pool.length, lo + Math.floor(Math.random() * (hi - lo + 1)));
      const wlo = Math.min(r.wmin, r.wmax), whi = Math.max(r.wmin, r.wmax);
      const added = [];
      for (let k = 0; k < count; k++) {
        const l = pool.splice(Math.floor(Math.random() * pool.length), 1)[0];
        const weight = Math.round((wlo + Math.random() * (whi - wlo)) * 20) / 20;
        this.form.loras.push({ name: l.name, weight });
        r.picked.push(l.name);
        added.push(`${l.name} (${weight})`);
      }
      if (this.opts.auto_triggers) {
        for (const name of r.picked) {
          await this.fetchTriggers(name);
          const words = this.activeTriggers(name).filter((w) => !this.inPrompt(w)).slice(0, 4);
          if (words.length) {
            this.addPromptWords(words);
            r.words.push(...words);
          }
        }
      }
      this.status = "🎲 Random LoRAs: " + added.join(", ");
    },

    // -------------------------------------------------------------- image prompt (IP-Adapter)
    onIpDrop(e) {
      this.ipDragOver = false;
      const file = [...(e.dataTransfer?.files || [])].find(isPicture);
      if (file) this.preparePicture(window.bee.pathForFile(file)).then((p) => { if (p) this.form.ip_adapter.image = p; });
    },
    onIpPicked(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) this.preparePicture(window.bee.pathForFile(file)).then((p) => { if (p) this.form.ip_adapter.image = p; });
    },

    // -------------------------------------------------------------- inpaint mask painting
    setI2iMode(m) {
      this.form.i2i_mode = m;
      if (m !== "change") this.controlMap = null;
    },
    initMaskCanvas() {
      const img = this.$refs.maskImg, cv = this.$refs.maskCanvas;
      if (!img || !cv) return;
      this.initDims = { w: img.naturalWidth, h: img.naturalHeight };
      const scale = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
      cv.width = Math.round(img.naturalWidth * scale);
      cv.height = Math.round(img.naturalHeight * scale);
      const sc = this.$refs.selCanvas;
      if (sc) { sc.width = cv.width; sc.height = cv.height; }
      this.sel = null;
      this._wandPix = null;
      this._ants = null;
      if (this.mask.data && this.mask.for === this.maskTarget()) {
        const prev = new Image();
        prev.onload = () => { cv.getContext("2d").drawImage(prev, 0, 0, cv.width, cv.height); this.maskSave(); };
        prev.src = this.mask.data;
      } else {
        this.mask.data = null;
      }
    },
    maskPoint(e) {
      const cv = this.$refs.maskCanvas, r = cv.getBoundingClientRect();
      return { x: (e.clientX - r.left) * (cv.width / r.width), y: (e.clientY - r.top) * (cv.height / r.height),
               k: cv.width / r.width };
    },
    maskDraw(a, b) {
      const ctx = this.$refs.maskCanvas.getContext("2d");
      ctx.globalCompositeOperation = this.mask.erase ? "destination-out" : "source-over";
      ctx.strokeStyle = "rgba(255, 64, 129, 1)";
      ctx.fillStyle = "rgba(255, 64, 129, 1)";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = this.mask.size * b.k;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(b.x, b.y, (this.mask.size * b.k) / 2, 0, Math.PI * 2);
      ctx.fill();
    },
    maskDown(e) {
      if (!this.$refs.maskCanvas) return;
      const tool = this.mask.tool || "brush";
      const sub = this.mask.subtract || e.altKey;
      const p = this.maskPoint(e);
      if (tool === "brush" || tool === "erase") {
        e.target.setPointerCapture?.(e.pointerId);
        this.mask.painting = true;
        this.mask.erase = tool === "erase" || sub;
        this.mask.last = p;
        this.maskDraw(p, p);
      } else if (tool === "lasso") {
        e.target.setPointerCapture?.(e.pointerId);
        this.sel = { tool, pts: [p], sub };
      } else if (tool === "poly") {
        if (!this.sel || this.sel.tool !== "poly") this.sel = { tool, pts: [p], sub };
        else {
          const f = this.sel.pts[0];
          if (this.sel.pts.length > 2 && Math.hypot(p.x - f.x, p.y - f.y) < 10 * p.k) return this.selFinishPoly();
          this.sel.pts.push(p);
        }
      } else if (tool === "object") {
        e.target.setPointerCapture?.(e.pointerId);
        this.sel = { tool, start: p, end: p, sub };
      } else if (tool === "wand") {
        this.wandAt(p, sub);
      }
      this.selDraw();
    },
    maskMove(e) {
      const tool = this.mask.tool || "brush";
      if (this.mask.painting && (tool === "brush" || tool === "erase")) {
        const p = this.maskPoint(e);
        this.maskDraw(this.mask.last, p);
        this.mask.last = p;
        return;
      }
      if (!this.sel) return;
      const p = this.maskPoint(e);
      if (this.sel.tool === "lasso" && e.buttons) this.sel.pts.push(p);
      else if (this.sel.tool === "poly") this.sel.hover = p;
      else if (this.sel.tool === "object" && e.buttons) this.sel.end = p;
      this.selDraw();
    },
    maskUp() {
      if (this.mask.painting) {
        this.mask.painting = false;
        this.mask.erase = (this.mask.tool || "brush") === "erase";
        this.maskSave();
        return;
      }
      const s = this.sel;
      if (!s) return;
      if (s.tool === "lasso") {
        if (s.pts.length > 2) this.selFill(s.pts, s.sub);
        this.sel = null;
        this.selDraw();
      } else if (s.tool === "object") {
        this.sel = null;
        this.selDraw();
        const x0 = Math.min(s.start.x, s.end.x), y0 = Math.min(s.start.y, s.end.y);
        const w = Math.abs(s.end.x - s.start.x), h = Math.abs(s.end.y - s.start.y);
        if (w > 6 && h > 6) this.objectSelect(x0, y0, w, h, s.sub);
      }
    },
    setMaskTool(id) {
      this.mask.tool = id;
      this.mask.erase = id === "erase";
      this.sel = null;
      this.selDraw();
    },
    selFinishPoly() {
      const s = this.sel;
      if (!s || s.tool !== "poly") return;
      if (s.pts.length > 2) this.selFill(s.pts, s.sub);
      this.sel = null;
      this.selDraw();
    },
    selKey(e) {  // Esc cancels a selection; B E L P W O pick a tool (while a picture can be painted)
      if (!this.$refs.maskCanvas || e.metaKey || e.ctrlKey) return;
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      if (e.key === "Escape" && this.sel) { this.sel = null; this.selDraw(); return; }
      if (e.key === "Enter" && this.sel?.tool === "poly") { e.preventDefault(); return this.selFinishPoly(); }
      const map = { b: "brush", e: "erase", l: "lasso", p: "poly", w: "wand", o: "object" };
      const id = map[(e.key || "").toLowerCase()];
      if (id) this.setMaskTool(id);
    },
    selDraw() {  // the dashed outline of a selection being drawn (on its own see-through layer)
      const c = this.$refs.selCanvas;
      if (!c) return;
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      if (this._ants && this._ants.w === c.width && this._ants.h === c.height) ctx.putImageData(this._ants.img, 0, 0);
      const s = this.sel;
      if (!s) return;
      const k = (s.pts && s.pts[0] && s.pts[0].k) || (s.start && s.start.k) || 1;
      ctx.lineWidth = 1.5 * k;
      const stroke = (path) => {
        ctx.setLineDash([6 * k, 4 * k]);
        ctx.strokeStyle = "#000"; path(); ctx.stroke();
        ctx.lineDashOffset = 6 * k;
        ctx.strokeStyle = "#fff"; path(); ctx.stroke();
        ctx.lineDashOffset = 0;
      };
      if (s.tool === "object") {
        stroke(() => { ctx.beginPath(); ctx.rect(Math.min(s.start.x, s.end.x), Math.min(s.start.y, s.end.y), Math.abs(s.end.x - s.start.x), Math.abs(s.end.y - s.start.y)); });
        return;
      }
      const pts = s.tool === "poly" && s.hover ? [...s.pts, s.hover] : s.pts;
      ctx.fillStyle = s.sub ? "rgba(255, 255, 255, .18)" : "rgba(255, 64, 129, .25)";
      ctx.beginPath();
      pts.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.closePath();
      ctx.fill();
      stroke(() => { ctx.beginPath(); pts.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y))); if (s.tool === "lasso") ctx.closePath(); });
      if (s.tool === "poly") {  // corner points; the first one is where to click to close
        s.pts.forEach((q, i) => { ctx.beginPath(); ctx.arc(q.x, q.y, (i ? 2.5 : 5) * k, 0, Math.PI * 2); ctx.fillStyle = i ? "#fff" : "rgba(255,200,80,.95)"; ctx.fill(); });
      }
    },
    selFill(pts, sub) {  // fill a closed outline into the painted area (or cut it out)
      const ctx = this.$refs.maskCanvas.getContext("2d");
      ctx.globalCompositeOperation = sub ? "destination-out" : "source-over";
      ctx.fillStyle = "rgba(255, 64, 129, 1)";
      ctx.beginPath();
      pts.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.closePath();
      ctx.fill();
      ctx.globalCompositeOperation = "source-over";
      this.maskSave();
    },
    async wandPixels() {  // the picture's own pixels at the painting area's size (read from the file, not the screen)
      const cv = this.$refs.maskCanvas;
      const key = this.maskTarget() + "@" + cv.width + "x" + cv.height;
      if (this._wandPix && this._wandPix.key === key) return this._wandPix;
      const bmp = await createImageBitmap(new Blob([await this.readImageBytes(this.maskTarget())]));
      const off = document.createElement("canvas");
      off.width = cv.width;
      off.height = cv.height;
      const octx = off.getContext("2d");
      octx.drawImage(bmp, 0, 0, cv.width, cv.height);
      this._wandPix = { key, data: octx.getImageData(0, 0, cv.width, cv.height).data };
      return this._wandPix;
    },
    async wandAt(p, sub) {  // magic wand: the connected area of similar colour around the click
      const cv = this.$refs.maskCanvas;
      let px;
      try {
        px = (await this.wandPixels()).data;
      } catch (e) {
        this.error = "The magic wand couldn't read this picture: " + (e.message || e);
        return;
      }
      const W = cv.width, H = cv.height;
      const x0 = Math.max(0, Math.min(W - 1, Math.round(p.x))), y0 = Math.max(0, Math.min(H - 1, Math.round(p.y)));
      const i0 = (y0 * W + x0) * 4;
      const r0 = px[i0], g0 = px[i0 + 1], b0 = px[i0 + 2];
      const tol = this.mask.tolerance || 32;
      const seen = new Uint8Array(W * H);
      const stack = [y0 * W + x0];
      seen[y0 * W + x0] = 1;
      const out = new ImageData(W, H);
      while (stack.length) {
        const n = stack.pop();
        const j = n * 4;
        out.data[j] = 255; out.data[j + 1] = 64; out.data[j + 2] = 129; out.data[j + 3] = 255;
        const x = n % W, y = (n - x) / W;
        for (const m of [x > 0 ? n - 1 : -1, x < W - 1 ? n + 1 : -1, y > 0 ? n - W : -1, y < H - 1 ? n + W : -1]) {
          if (m < 0 || seen[m]) continue;
          const q = m * 4;
          if (Math.max(Math.abs(px[q] - r0), Math.abs(px[q + 1] - g0), Math.abs(px[q + 2] - b0)) <= tol) {
            seen[m] = 1;
            stack.push(m);
          }
        }
      }
      this.selStamp(out, sub);
    },
    selStamp(imageData, sub) {  // add (or cut) a picked area into the painted area
      const cv = this.$refs.maskCanvas;
      const tmp = document.createElement("canvas");
      tmp.width = cv.width;
      tmp.height = cv.height;
      tmp.getContext("2d").putImageData(imageData, 0, 0);
      const ctx = cv.getContext("2d");
      ctx.globalCompositeOperation = sub ? "destination-out" : "source-over";
      ctx.drawImage(tmp, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      this.maskSave();
    },
    async objectSelect(x, y, w, h, sub) {  // object selection: the outline of the thing inside the box (OpenCV GrabCut)
      const cv = this.$refs.maskCanvas;
      this.status = "Finding the object…";
      this.objFinding = true;
      this.error = "";
      try {
        const r = await call("select_object", { path: this.maskTarget(), width: cv.width, height: cv.height,
                                                rect: [x / cv.width, y / cv.height, w / cv.width, h / cv.height] });
        const img = new Image();
        await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; img.src = r.mask; });
        const tmp = document.createElement("canvas");
        tmp.width = cv.width;
        tmp.height = cv.height;
        tmp.getContext("2d").drawImage(img, 0, 0, cv.width, cv.height);
        if (r.found) {
          this.selStamp(tmp.getContext("2d").getImageData(0, 0, cv.width, cv.height), sub);
          this.status = "Object selected (outlined): fix the edges with the brush or eraser if needed";
        } else {  // no clear outline: take the box itself, so something always happens
          this.selFill([{ x, y }, { x: x + w, y }, { x: x + w, y: y + h }, { x, y: y + h }], sub);
          this.status = "Couldn't find a clear outline in that box, so the whole box is selected: trim it with the eraser, or drag a tighter box";
        }
      } catch (e) {
        this.error = "Object selection failed: " + (e.message || e);
      } finally {
        this.objFinding = false;
      }
    },
    maskSave() {
      const cv = this.$refs.maskCanvas;
      if (!cv) return;
      const ctx = cv.getContext("2d");
      const px = ctx.getImageData(0, 0, cv.width, cv.height).data;
      let any = false;
      for (let i = 3; i < px.length; i += 16) if (px[i] > 32) { any = true; break; }
      this.mask.data = any ? cv.toDataURL("image/png") : null;
      this.mask.for = this.maskTarget();
      this.antsMake(px, cv.width, cv.height);
      this.selDraw();
    },
    antsMake(px, W, H) {
      // "marching ants": a black-and-white dashed line around everything that's painted or selected, so the
      // selection shows on any picture (the pink fill alone is hard to see on pink, red or busy pictures)
      const out = new ImageData(W, H);
      const d = out.data;
      const on = (x, y) => x >= 0 && y >= 0 && x < W && y < H && px[(y * W + x) * 4 + 3] > 32;
      const t = Math.max(1, Math.round(W / 700));  // a little thicker on big pictures
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          if (!on(x, y) || (on(x - 1, y) && on(x + 1, y) && on(x, y - 1) && on(x, y + 1))) continue;
          const white = ((x + y) >> 2) & 1;
          for (let dy = 0; dy < t; dy++) for (let dx = 0; dx < t; dx++) {
            const xx = Math.min(W - 1, x + dx), yy = Math.min(H - 1, y + dy);
            const j = (yy * W + xx) * 4;
            const v = white ? 255 : 0;
            d[j] = v; d[j + 1] = v; d[j + 2] = v; d[j + 3] = 255;
          }
        }
      }
      this._ants = { w: W, h: H, img: out };
    },
    cmpDrag(e) {  // Tools → Split view: drag on the picture to move the line
      const box = this.$refs.cmpSplit;
      if (!box) return;
      e.preventDefault();
      const move = (ev) => {
        const r = box.getBoundingClientRect();
        this.tool.split = Math.min(1, Math.max(0, (ev.clientX - r.left) / Math.max(1, r.width)));
      };
      move(e);
      const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    },
    brushAt(e) {  // the ring follows the pointer (in the painting area's own coordinates)
      const cv = this.$refs.maskCanvas;
      if (!cv) return;
      const r = cv.getBoundingClientRect();
      this.brush = { show: true, x: e.clientX - r.left, y: e.clientY - r.top };
    },
    brushPreview() {  // moving the size slider shows the ring in the middle of the picture for a moment
      const cv = this.$refs.maskCanvas;
      if (!cv) return;
      const r = cv.getBoundingClientRect();
      this.brush = { show: true, x: r.width / 2, y: r.height / 2 };
      clearTimeout(this._brushT);
      this._brushT = setTimeout(() => { if (!this.mask.painting) this.brush.show = false; }, 900);
    },
    loraKindOf(l) {
      return /\.tdict$/i.test(l.file || "") || l.source === "diffusionbee" ? "diffusionbee" : "safetensors";
    },
    sortLoras(list) {  // ★ favourites first, then the chosen order (A–Z, Z–A, biggest, smallest, newest list order)
      const by = this.loraSort;
      const name = (l) => (l.name || "").toLowerCase();
      let out = list.slice();
      if (by === "az") out.sort((a, b) => name(a).localeCompare(name(b)));
      else if (by === "za") out.sort((a, b) => name(b).localeCompare(name(a)));
      else if (by === "big") out.sort((a, b) => (b.size_mb || 0) - (a.size_mb || 0));
      else if (by === "small") out.sort((a, b) => (a.size_mb || 0) - (b.size_mb || 0));
      return this.favFirst(out);
    },
    maskTarget() {  // the picture being painted on: Tools' picture, or Image to image's
      return this.tab === "tools" ? (this.tool.source?.path || "") : this.form.init_image;
    },
    // Tools → 🩹 Remove objects, right in Tools: paint on the picture on the left, then Remove
    toolPaintRemove(on) {
      if (!this.tool.source) return;
      if (on && this.adj.open) this.closeAdjust();
      this.tool.healing = !!on;
      if (on) this.status = "Paint over what you want gone on the picture, then press 🩹 Remove";
    },
    async toolHeal() {
      const s = this.tool.source;
      if (!s) return;
      if (!this.mask.data || this.mask.for !== s.path) return (this.error = "Paint over what you want to remove first.");
      this.error = "";
      const mask = this.mask.data;
      this.tool.working = { label: "Removing what you painted", started: Date.now() };
      this.tool.latest = null;
      try {
        const rec = await this.runJob("heal", { ...this.toolArgs(), mask, method: this.form.heal_method || "lama", from_tools: true });
        if (rec) {
          this.tool.latest = rec;
          this.afterPost(rec);
          this.tool.healing = false;
          this.mask.data = null;
        }
      } finally {
        this.tool.working = null;
      }
    },
    maskClear() {
      const cv = this.$refs.maskCanvas;
      if (cv) cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
      this.mask.data = null;
      this._ants = null;
      this.selDraw();
    },
    maskInvert() {
      const cv = this.$refs.maskCanvas;
      if (!cv) return;
      const ctx = cv.getContext("2d");
      const im = ctx.getImageData(0, 0, cv.width, cv.height);
      const d = im.data;
      for (let i = 0; i < d.length; i += 4) {
        const on = d[i + 3] > 32;
        d[i] = 255; d[i + 1] = 64; d[i + 2] = 129; d[i + 3] = on ? 0 : 255;
      }
      ctx.putImageData(im, 0, 0);
      this.maskSave();
    },
    onInitLoad(e) {
      this.initDims = { w: e.target.naturalWidth, h: e.target.naturalHeight };
    },
    outpaintPreset(kind) {
      const d = this.initDims;
      if (!d && kind !== "none") return (this.status = "Still reading the picture's size: try again in a moment");
      const step = (v) => Math.round(v / 32) * 32;
      if (kind === "all") Object.assign(this.form.outpaint, { left: step(d.w / 4), right: step(d.w / 4), top: step(d.h / 4), bottom: step(d.h / 4) });
      else if (kind === "wide") Object.assign(this.form.outpaint, { left: step(d.w / 4), right: step(d.w / 4), top: 0, bottom: 0 });
      else if (kind === "tall") Object.assign(this.form.outpaint, { left: 0, right: 0, top: step(d.h / 4), bottom: step(d.h / 4) });
      else Object.assign(this.form.outpaint, { left: 0, right: 0, top: 0, bottom: 0 });
    },

    // -------------------------------------------------------------- comparison grid
    // Compare: the values as keys to click (each picked key = one column / row of the grid)
    cmpOptions(axis) {
      const nums = (list) => list.map((v) => ({ v: String(v), label: String(v) }));
      switch (axis) {
        case "cfg": return nums([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]);
        case "steps": return nums([4, 8, 10, 15, 20, 25, 30, 40, 50]);
        case "sampler": return (this.samplers || []).slice(0, 16).map((x) => ({ v: x.id, label: x.label, title: x.id }));
        case "seed": return nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        case "clip_skip": return nums([1, 2, 3, 4]);
        case "lora_weight": return nums([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]);
        case "strength": return nums([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        case "lora": return [{ v: "none", label: "none", title: "without the LoRA" },
          ...[...new Set([...this.form.loras.map((l) => l.name), ...this.compatibleLoras.slice(0, 15).map((l) => l.name)])]
            .map((n) => ({ v: n, label: n.length > 18 ? n.slice(0, 17) + "…" : n, title: n }))];
        default: return [];
      }
    },
    cmpList(which) {
      return String(this.cmp[which] || "").split(",").map((x) => x.trim()).filter(Boolean);
    },
    cmpHas(which, v) {
      return this.cmpList(which).some((x) => x === v || (Number(x) === Number(v) && x !== "" && !isNaN(Number(x)) && !isNaN(Number(v))));
    },
    cmpToggle(which, v, axisKey) {
      const axis = this.cmp[axisKey];
      const order = this.cmpOptions(axis).map((o) => o.v);
      let list = this.cmpList(which);
      if (this.cmpHas(which, v)) list = list.filter((x) => !(x === v || (!isNaN(Number(x)) && Number(x) === Number(v))));
      else list.push(v);
      list.sort((a, b) => { const i = order.indexOf(a), j = order.indexOf(b); return (i < 0 ? 999 : i) - (j < 0 ? 999 : j); });
      this.cmp[which] = list.join(", ");
    },
    cmpHelp(axis) {
      return { cfg: "e.g. 4, 6, 8, 10  or  3-12 (+3)", steps: "e.g. 15, 25, 35", sampler: "sampler names, e.g. " + this.samplers.slice(0, 3).map((x) => x.id).join(", "),
               seed: "e.g. 1, 2, 3, 4", clip_skip: "e.g. 1, 2", lora_weight: "e.g. 0.4, 0.7, 1.0", lora: "LoRA names, comma-separated; “none” = without",
               strength: "e.g. 0.3, 0.5, 0.7" }[axis] || "";
    },
    cmpPlaceholder(axis) {
      return { cfg: "4, 6, 8, 10", steps: "15, 25, 35", sampler: this.samplers.slice(0, 4).map((x) => x.id).join(", "), seed: "1, 2, 3, 4",
               clip_skip: "1, 2", lora_weight: "0.4, 0.7, 1.0", lora: "none, " + this.compatibleLoras.slice(0, 2).map((l) => l.name).join(", "),
               strength: "0.3, 0.5, 0.7" }[axis] || "";
    },
    cmpDefaults(which) {
      const axis = which === "x" ? this.cmp.x : this.cmp.y;
      if (which === "y" && this.cmp.y === this.cmp.x) this.cmp.y = "";
      if (axis) this.cmp[which + "v"] = this.cmpPlaceholder(axis);
    },
    async runCompare() {
      const base = this.createArgs({ num_images: 1 });
      if (this.cmp.x === "strength" || this.cmp.y === "strength") {
        // change strength only means something with a starting picture
        if (!this.form.init_image) return (this.error = "Add a picture in Image to image to compare change strength.");
        base.init_image = this.form.init_image;
        base.strength = this.form.strength;
      }
      const res = await this.runJob("compare", {
        base, x_axis: this.cmp.x, x_values: this.cmp.xv, y_axis: this.cmp.y || null, y_values: this.cmp.yv,
        lora: this.cmp.lora || null, model: base.model,
      });
      if (res && res.grid) this.status = `Comparison grid ready (${res.images.length} pictures) · it's in the Library too`;
    },

    // -------------------------------------------------------------- jobs (one heavy job at a time)
    async runJob(cmd, args) {
      // Heavy work goes into the backend's queue; this resolves when the job finishes.
      this.error = "";
      try {
        return await call(cmd, args);
      } catch (e) {
        if (!e.cancelled) this.error = e.message;
        return null;
      }
    },
    cancelJob(id) {
      const job = this.queue.find((j) => j.id === id);
      if (job?.state === "running") {
        this.stopping = { ...this.stopping, [id]: Date.now() };
        this.status = "Stopping…";
      }
      call("cancel_job", { job_id: id });
    },
    // -------------------------------------------------------------- training
    async openTrain() {
      this.tab = "train";
      if (!this.train.model || !this.trainModels.some((m) => m.id === this.train.model))
        this.train.model = (this.trainModels.find((m) => m.cached) || this.trainModels[0] || {}).id || "";
      if (this.train.folder && !this.train.items.length) await this.trainScan();
    },
    async trainPickFolder() {
      const dir = await window.bee.pickFolder({ title: "Choose a folder of training images", defaultPath: this.train.folder || undefined, button: "Use these images" });
      const d = Array.isArray(dir) ? dir[0] : dir;
      if (!d) return;
      this.train.folder = d;
      if (!this.train.name) this.train.name = d.split("/").filter(Boolean).pop().replace(/[^\w\- ]+/g, "_");
      await this.trainScan(true);
    },
    async trainClearFolder() {  // forget the chosen pictures (the folder and its files aren't touched)
      if (this.train.dirty && !(await this.confirmAsk("Clear the training pictures?", "Your caption edits haven't been saved yet and will be lost. The pictures and the folder stay as they are.", "Clear"))) return;
      Object.assign(this.train, { folder: "", items: [], dirty: false, name: "" });
      this.frame.copied = null;
      this.frame.msg = "";
      this.status = "Training pictures cleared · nothing in the folder was changed";
    },
    async trainScan(applyPreset) {
      try {
        const r = await call("train_scan", { folder: this.train.folder });
        this.train.items = r.items;
        this.train.dirty = false;
        if (applyPreset) this.applyTrainPreset(this.trainPresets.find((p) => p.id === this.train.preset) || this.trainPresets[1]);
      } catch (e) {
        this.train.items = [];
        this.error = e.message;
      }
    },
    // ----------------------------------------------- Train: how each picture becomes a square
    trainNotSquare() {
      return this.train.items.filter((it) => it.width && it.height && it.width !== it.height).length;
    },
    frameStyle(it) {
      const pos = { center: "50% 50%", top: "50% 0%", bottom: "50% 100%", left: "0% 50%", right: "100% 50%" }[it.fit || "center"];
      return it.fit === "pad" ? { objectFit: "contain" } : { objectFit: "cover", objectPosition: pos };
    },
    isBoxFit(fit) {
      return typeof fit === "string" && fit.startsWith("box:");
    },
    parseBox(fit) {
      const v = this.isBoxFit(fit) ? fit.slice(4).split(",").map(Number) : null;
      return v && v.length === 3 && v.every(Number.isFinite) ? { x: v[0], y: v[1], s: v[2] } : null;
    },
    cropBgStyle(it) {  // a tile showing just the square picked by hand
      const b = this.parseBox(it.fit), w = it.width || 1, h = it.height || 1;
      if (!b) return {};
      const side = b.s * Math.min(w, h);
      const left = Math.min(Math.max(0, b.x * w), w - side), top = Math.min(Math.max(0, b.y * h), h - side);
      return { backgroundImage: `url("${this.fileUrl(it.path)}")`, backgroundSize: `${(w / side) * 100}% ${(h / side) * 100}%`,
               backgroundPosition: `${w - side > 0 ? (left / (w - side)) * 100 : 0}% ${h - side > 0 ? (top / (h - side)) * 100 : 0}%` };
    },
    pickFrame(it, e) {
      const v = e.target.value;
      if (v === "box") {
        e.target.value = this.isBoxFit(it.fit) ? "box" : (it.fit || "center");
        return this.openCrop(it);
      }
      this.setFrame(it, v);
    },
    openCrop(it) {
      const b = this.parseBox(it.fit);
      this.cropEd = { it, x: b ? b.x : 0, y: b ? b.y : 0, s: b ? b.s : 1, w: 0, h: 0 };
    },
    cropImgLoaded(e) {
      const c = this.cropEd;
      if (!c) return;
      c.w = e.target.naturalWidth;
      c.h = e.target.naturalHeight;
      if (!this.parseBox(c.it.fit)) {  // start from what the tile showed: the biggest square, centred
        c.s = 1;
        c.x = c.w > c.h ? (1 - c.h / c.w) / 2 : 0;
        c.y = c.h > c.w ? (1 - c.w / c.h) / 2 : 0;
      }
      this.cropClamp();
    },
    cropSidePx() {
      const c = this.cropEd;
      return c.s * Math.min(c.w, c.h);
    },
    cropClamp() {
      const c = this.cropEd;
      if (!c || !c.w) return;
      c.s = Math.min(1, Math.max(0.1, c.s));
      const side = this.cropSidePx();
      c.x = Math.min(Math.max(0, c.x), (c.w - side) / c.w);
      c.y = Math.min(Math.max(0, c.y), (c.h - side) / c.h);
    },
    cropBoxStyle() {
      const c = this.cropEd, side = this.cropSidePx();
      return { left: c.x * 100 + "%", top: c.y * 100 + "%", width: (side / c.w) * 100 + "%", height: (side / c.h) * 100 + "%" };
    },
    cropPixels() {
      const side = Math.round(this.cropSidePx());
      return `${side} × ${side} px` + (side < 512 ? " · smaller than 512: it will be scaled up" : "");
    },
    cropDrag(e, mode) {
      const stage = this.$refs.cropStage, c = this.cropEd;
      if (!stage || !c) return;
      e.preventDefault();
      const r = stage.getBoundingClientRect();
      const start = { px: e.clientX, py: e.clientY, x: c.x, y: c.y, s: c.s };
      const move = (ev) => {
        const dx = (ev.clientX - start.px) / r.width, dy = (ev.clientY - start.py) / r.height;
        if (mode === "move") {
          c.x = start.x + dx;
          c.y = start.y + dy;
        } else {  // the corner: grow or shrink the square, keeping its top-left corner where it is
          const short = Math.min(c.w, c.h);
          const grow = Math.max(dx * c.w, dy * c.h);
          c.s = start.s + grow / short;
        }
        this.cropClamp();
      };
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    },
    async cropSave() {
      const c = this.cropEd;
      if (!c) return;
      const r = (v) => Math.round(v * 10000) / 10000;
      const it = c.it;
      this.cropEd = null;
      await this.setFrame(it, `box:${r(c.x)},${r(c.y)},${r(c.s)}`);
    },
    async setFrame(it, fit) {
      it.fit = fit;
      await this.saveFraming();
    },
    async setFrameAll(fit, onlyNonSquare) {
      for (const it of this.train.items) {
        if (it.fit === "skip") continue;  // left-out pictures stay left out
        if (!onlyNonSquare || (it.width !== it.height)) it.fit = fit;
      }
      await this.saveFraming();
    },
    async saveFraming() {
      const framing = Object.fromEntries(this.train.items.map((it) => [it.name, it.fit || "center"]));
      try {
        await call("train_framing_save", { folder: this.train.folder, framing });
        this.frame.msg = "Framing saved · training uses it";
      } catch (e) {
        this.frame.msg = "";
        this.error = e.message;
      }
    },
    async makeSquareCopies() {
      const framing = Object.fromEntries(this.train.items.map((it) => [it.name, it.fit || "center"]));
      Object.assign(this.frame, { busy: true, msg: "", copied: null });
      try {
        const r = await call("train_square_copies", { folder: this.train.folder, framing, size: this.frame.size });
        this.frame.copied = r;
        this.frame.msg = `Saved ${r.count} square copies (${this.frame.size}×${this.frame.size}) with their captions`;
      } catch (e) {
        this.error = e.message;
      }
      this.frame.busy = false;
    },
    async useSquareCopies() {
      if (!this.frame.copied) return;
      this.train.folder = this.frame.copied.folder;
      this.frame.copied = null;
      this.frame.msg = "";
      await this.trainScan(false);
    },
    onTrainModelChange() {
      const ram = this.sys.ram_gb || 0;
      if (this.trainFastErnie) {
        this.train.resolution = ram && ram < 16 ? 512 : 768;
        this.train.train_text_encoder = false;
        if (ram && ram < 16) Object.assign(this.train, { rank: 8, targets: "attention" });
      } else if (this.trainFamily === "ernie") {
        this.train.resolution = 1024;
        this.train.train_text_encoder = false;
        this.train.format = "diffusers";
      } else if (this.trainFamily === "sdxl") {
        this.train.resolution = ram && ram < 16 ? 768 : 1024;
        this.train.train_text_encoder = false;
      } else if (this.train.resolution > 768) this.train.resolution = 512;
    },
    applyTrainPreset(p) {
      Object.assign(this.train, { preset: p.id, steps: p.steps, lr: p.lr, rank: p.rank });
    },
    async trainSaveCaptions() {
      const n = await call("train_captions_save", { items: this.train.items.map(({ path, caption }) => ({ path, caption })) });
      this.train.dirty = false;
      this.status = `Saved ${n} caption files`;
    },
    async trainAutoCaption(overwrite) {
      if (this.train.dirty) await this.trainSaveCaptions();
      const r = await this.runJob("train_autocaption", { folder: this.train.folder, mode: this.train.capMode, overwrite });
      if (r) {
        this.train.items = r.items;
        this.status = `Captioned ${r.captioned} image${r.captioned === 1 ? "" : "s"}`;
      }
    },
    async startTraining() {
      if (this.train.dirty) await this.trainSaveCaptions();
      this.train.result = null;
      this.train.lossHistory = [];
      const t = this.train;
      const args = { model: t.model, folder: t.folder, name: t.name.trim(), trigger: t.trigger.trim(), format: t.format,
                     precision: t.precision, save_every: t.save_every, steps: t.steps, lr: t.lr, rank: t.rank, alpha: t.rank,
                     resolution: t.resolution, targets: t.targets, flip: t.flip, train_text_encoder: t.train_text_encoder };
      const res = await this.runJob("train_lora", args);
      if (res) {
        this.train.result = res;
        await this.refreshLoras();
        if (res.trigger) await call("lora_triggers_save", { name: res.name, words: [res.trigger] });
        this.status = `LoRA ${res.name} saved`;
      }
    },
    setTrainMode(mode) {
      this.train.mode = mode;
      if (mode === "merge") return;
      const cur = this.models.find((x) => x.id === this.train.model);
      if (!cur || this.trainBlock(cur)) {  // keep a model this kind of training can use
        const ok = this.models.filter((m) => !this.trainBlock(m));
        const pick = ok.find((m) => m.id === this.form.model) || ok.find((m) => this.isInstalled(m)) || ok[0];
        if (pick) this.train.model = pick.id;
      }
      this.onTrainModelChange();
    },
    async startEmbeddingTraining() {
      if (this.train.dirty) await this.trainSaveCaptions();
      this.train.result = null;
      this.train.lossHistory = [];
      const t = this.train, e = t.emb;
      const args = { model: t.model, folder: t.folder, name: e.name.trim(), init_word: e.init.trim(), kind: e.kind,
                     vectors: e.vectors, steps: e.steps, save_every: e.save_every, flip: t.flip };
      const res = await this.runJob("train_embedding", args);
      if (res) {
        this.train.result = res;
        await this.refreshEmbeddings();
        this.status = `Embedding ${res.name} saved`;
      }
    },
    useTrainedEmbedding() {
      const r = this.train.result;
      if (!r) return;
      const m = this.models.find((x) => x.id === this.train.model);
      if (m && this.form.model !== m.id) { this.form.model = m.id; this.onModelChange(); }
      this.tab = "create";
      if (!this.inPrompt(r.name)) this.addPromptWords([r.name]);
    },
    mergeAdd(l) {
      if (!this.merge.family) this.merge.family = this.mergeFamily;
      this.merge.items.push({ name: l.name, weight: 1 });
      if (!this.merge.name) this.merge.name = "";
    },
    mergeRemove(i) {
      this.merge.items.splice(i, 1);
    },
    mergeSetFamily(f) {
      if (f !== this.mergeFamily) this.merge.items = [];
      this.merge.family = f;
    },
    async startMerge() {
      this.merge.error = "";
      this.merge.result = null;
      const res = await this.runJob("lora_merge", { items: this.merge.items.map((i) => ({ name: i.name, weight: Number(i.weight) })),
                                                    name: this.merge.name.trim(), mode: this.merge.mode, rank: this.merge.rank });
      if (res) {
        this.merge.result = res;
        await this.refreshLoras();
        this.status = `Merged LoRA ${res.name} saved`;
      } else if (this.error) {
        this.merge.error = this.error;
      }
    },
    useMergedLora() {
      const r = this.merge.result;
      if (!r) return;
      this.tab = "create";
      this.$nextTick(() => this.addLoraByName(r.name));
    },
    useTrainedLora() {
      const r = this.train.result;
      if (!r) return;
      const m = this.models.find((x) => x.id === this.train.model);
      if (m && this.form.model !== m.id) { this.form.model = m.id; this.onModelChange(); }
      this.tab = "create";
      this.$nextTick(() => this.addLoraByName(r.name));
    },
    filterByLora(name) {
      this.detail = null;
      this.lib.loras = [name];
      this.libLoraOpen = true;
      if (this.tab !== "library") this.openLibrary();
      else this.queryLibrary();
    },
    async togglePause() {
      const on = !this.paused;
      this.setPaused(on);
      this.status = on ? "Paused" : "Resuming…";
      try {
        const r = await call("pause", { on });
        this.setPaused(!!(r && r.paused));
      } catch (e) {
        this.setPaused(!on);
        this.error = "Couldn't " + (on ? "pause" : "resume") + ": " + e.message;
      }
    },
    setPaused(on) {
      if (on === this.paused) return;
      if (on) {
        this.pausedAt = Date.now();
      } else if (this.pausedAt) {
        // the clocks don't count paused time
        const gap = Date.now() - this.pausedAt;
        if (this.jobStartedAt) this.jobStartedAt += gap;
        if (this.firstStepAt) this.firstStepAt += gap;
        if (this.timing.lastAt) this.timing.lastAt += gap;
        if (this.tool.working && this.tool.working.started) this.tool.working.started += gap;
        this.pausedAt = 0;
      }
      this.paused = on;
    },
    clockNow() {
      return this.paused && this.pausedAt ? this.pausedAt : this.now;
    },
    reusePrompt(r, original = false) {
      const s = r.settings || {};
      const text = original && s.original_prompt ? s.original_prompt : r.prompt;
      if (!text) return;
      this.form.prompt = text;
      if (r.negative_prompt != null) this.form.negative_prompt = r.negative_prompt;
      // an enhanced prompt is already detailed: don't enhance it again (the original can be)
      if (!original && s.original_prompt && s.original_prompt !== r.prompt) this.form.enhance_prompt = false;
      this.detail = null;
      if (this.tab !== "create" && this.tab !== "i2i") this.tab = "create";
      this.status = "Prompt reused: change anything you like, then Generate";
    },
    skipImage() {
      call("skip_image", {});
      this.status = "Skipping this image…";
    },
    queuedIndex(j) {  // place among the waiting jobs (0 = starts next)
      return this.queue.filter((x) => x.state === "queued").findIndex((x) => x.id === j.id);
    },
    async moveJob(id, delta) {  // ▲ / ▼: change which waiting job starts next
      try {
        await call("queue_move", { job_id: id, delta });
      } catch (e) {
        this.error = e.message;
      }
    },
    canSendBack(j) {  // jobs that can simply start again later (not training, downloads or file moves)
      return !!jobArgs.get(j.id) && !["train_lora", "train_embedding", "model_download", "downloads_move", "civitai_download",
                                      "model_delete", "convert_tdict", "lora_merge"].includes(j.cmd);
    },
    sendBack(id) {
      // ⤵ Run the others first: stop the running job and add it again at the end of the queue. A picture job only
      // re-makes what it hadn't finished: the finished pictures are already saved, and the seed moves on with them.
      const sent = jobArgs.get(id);
      if (!sent) return;
      const args = JSON.parse(JSON.stringify(sent.args));
      const p = this.progress;
      if (sent.cmd === "generate" && p && p.job === id && p.images > 1 && p.image > 1) {
        const done = p.image - 1;
        args.num_images = Math.max(1, (args.num_images || p.images) - done);
        if (Number.isFinite(args.seed) && args.seed >= 0) args.seed += done;
      }
      this.sentBack = { ...this.sentBack, [id]: { cmd: sent.cmd, args } };
      this.cancelJob(id);
      this.status = "Moving it to the back of the queue: the others run first";
    },
    resendSentBack() {  // called when the queue changes: a job that was sent back has stopped, so add it again
      const ids = new Set(this.queue.map((x) => x.id));
      for (const [id, job] of Object.entries(this.sentBack)) {
        if (ids.has(Number(id))) continue;
        const rest = { ...this.sentBack };
        delete rest[id];
        this.sentBack = rest;
        call(job.cmd, job.args).catch(() => {});  // its pictures arrive like any other job's
      }
    },
    stopAll() {
      this.stopAllAt = Date.now();
      this.sentBack = {};  // Stop all means all: nothing comes back
      if (this.runningJob) this.stopping = { ...this.stopping, [this.runningJob.id]: Date.now() };
      call("cancel_all");
      this.status = "Stopping…";
    },
    // Stop means stop: a job that hasn't stopped on its own a moment after Stop (a step can take a minute on
    // a busy Mac) is force-stopped automatically: the engine restarts and the jobs still waiting carry on.
    // Training is left to finish its step, because its Stop saves what it learned so far.
    stopForGood() {
      const job = this.runningJob;
      if (!job || this.forcing) return;
      const t = this.stopping[job.id];
      if (!t || Date.now() - t < STOP_GRACE_MS) return;
      if (["train_lora", "train_embedding"].includes(job.cmd)) return;
      this.forcing = true;
      this.forceStop(!this.stopAllAt || Date.now() - this.stopAllAt > 10000)
        .finally(() => { this.forcing = false; this.stopAllAt = 0; });
      this.status = "Stopped";
    },
    canForceStop(job) {
      // only training still offers the extra button (everything else is force-stopped by itself)
      const t = this.stopping[job.id];
      return job.state === "running" && t && this.now - t > 4000 && ["train_lora", "train_embedding"].includes(job.cmd);
    },
    // The engine (Python backend) stopped without being asked to. Say why in plain words and, unless it
    // keeps happening, start it again by itself and re-add the jobs that were still waiting.
    onEngineExit(info) {
      this.backendDead = true;
      const tail = (info.tail || "").replace(/\/Users\/[^/\s'"]+/g, "~").slice(-600);
      const killed = info.signal === "SIGKILL" || info.signal === "SIGTERM";
      const why = killed
        ? "macOS stopped it, usually because the Mac ran out of memory. Close other big apps, use a smaller size or fewer extras (Hi-res fix and Face detailer use the most)."
        : info.signal ? `It crashed (${info.signal}).` : `It exited (code ${info.code}).`;
      const now = Date.now();
      this.engineExits = this.engineExits.filter((t) => now - t < 120000).concat(now);
      const running = this.queue.find((j) => j.state === "running");
      if (this.engineExits.length <= 2) {
        this.error = `The engine stopped. ${why} It has been restarted` +
          (running ? `; the job that was running (“${running.label || running.cmd}”) didn't finish` : "") +
          `. Details are in Settings → Backend log.\n\n${tail}`;
        this.forceStop(); // restart and re-add waiting jobs
      } else {
        this.error = `The engine stopped again. ${why} It wasn't restarted automatically because this keeps happening: ` +
          `press Restart engine (Tools or Settings) when you're ready.\n\n${tail}`;
      }
    },
    async forceQuit() {
      this.quitAsk = false;
      if (window.bee.forceQuit) await window.bee.forceQuit();
    },
    async forceStop(keepWaiting = true) {
      // Kill the engine process immediately (frees all its memory), restart it, and re-add waiting jobs.
      const waiting = !keepWaiting ? [] : this.queue.filter((j) => j.state === "queued").map((j) => jobArgs.get(j.id)).filter(Boolean);
      for (const [id, p] of pending) {
        if (jobArgs.has(id)) {
          pending.delete(id);
          jobArgs.delete(id);
          p.reject(Object.assign(new Error("Stopped"), { cancelled: true }));
        }
      }
      this.requeue = waiting;
      this.queue = [];
      this.progress = null;
      this.stopping = {};
      this.status = "Force-stopping and restarting the engine…";
      await window.bee.restart();
    },
    jobProgress(job) {
      const p = this.progress;
      if (job.state !== "running" || !p || p.job !== job.id) return null;
      return Math.round((((p.image - 1) + p.step / p.steps) / p.images) * 100);
    },

    // -------------------------------------------------------------- progress / ETA
    recordPerf(job) {
      // Learn this Mac's real speed so later estimates get better.
      const took = (Date.now() - this.jobStartedAt) / 1000;
      if (!this.jobStartedAt || took < 1) return;
      const ema = (old, v) => (old == null ? v : old * 0.6 + v * 0.4);
      if (job.cmd === "generate") {
        const t = this.timing;
        if (t.job === job.id && t.stepTimes.length >= 3) {
          const sorted = [...t.stepTimes].sort((a, b) => a - b);
          const mp = ((job.width || 1024) * (job.height || 1024)) / 1e6;
          const p = (this.perf.models[job.model] ||= {});
          p.sps = ema(p.sps, sorted[Math.floor(sorted.length / 2)] / 1000 / mp);
          if (this.firstStepAt && job.model !== this.lastModel) {
            p.load = ema(p.load, (this.firstStepAt - this.jobStartedAt) / 1000);
          }
        }
        this.lastModel = job.model;
      } else {
        const key = job.cmd === "postprocess" ? job.action : job.cmd;
        let v = took;
        if (key === "enhance" && job.upscale) v /= (job.upscale / 2) ** 2;
        this.perf.jobs[key] = ema(this.perf.jobs[key], v);
      }
      saveJSON(PERF_KEY, this.perf);
    },
    onProgress(m) {
      if (!this.firstStepAt) this.firstStepAt = Date.now();
      const t = this.timing;
      const nowMs = Date.now();
      if (t.job !== m.job) {
        Object.assign(t, { job: m.job, image: m.image, lastAt: nowMs, stepTimes: [] });
      } else if (t.image !== m.image) {
        Object.assign(t, { image: m.image, lastAt: nowMs }); // new image: skip its setup time
      } else {
        t.stepTimes.push(nowMs - t.lastAt);
        if (t.stepTimes.length > 30) t.stepTimes.shift();
        t.lastAt = nowMs;
      }
      this.progress = m;
    },
    genLeft(pr) {
      // Seconds left in the running picture job, measured from its own steps: the steps still to go, plus the
      // time each picture takes after its last step (decoding and saving, learned per model; ~2 steps until then),
      // counting down between steps. The progress line, the queue and the TIME LEFT clock all use this one figure.
      const times = this.timing.stepTimes;
      if (!pr || !times.length || this.timing.job !== pr.job) return null;
      const sorted = [...times].sort((a, b) => a - b);
      const perStep = sorted[Math.floor(sorted.length / 2)] / 1000;
      const job = this.queue.find((x) => x.id === pr.job);
      const tail = (job && this.perf.models[job.model]?.tail) ?? Math.min(2 * perStep, 12);
      const stepsLeft = (pr.steps - pr.step) + (pr.images - pr.image) * pr.steps;
      const since = Math.max(0, (this.clockNow() - this.timing.lastAt) / 1000);
      const stepPart = Math.max(stepsLeft ? 1 : 0, stepsLeft * perStep - since);
      const tailPart = (pr.images - pr.image + 1) * tail - (stepsLeft ? 0 : since);
      return { perStep, seconds: Math.max(1, stepPart + Math.max(0, tailPart)) };
    },
    etaText(job) {
      const e = this.jobEtas.find((x) => x.id === job.id);
      if (!e) return "";
      const guess = e.measured ? "" : " (rough guess until the first run)";
      if (job.state === "running" && this.paused) return `❚❚ Paused · ~${this.formatDuration(e.remaining)} left when resumed`;
      return job.state === "running"
        ? `~${this.formatDuration(e.remaining)} left`
        : `~${this.formatDuration(e.remaining)} · done in ~${this.formatDuration(e.finishIn)}${guess}`;
    },
    clockText(sec) {
      sec = Math.max(0, Math.round(sec));
      const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
      const p2 = (n) => String(n).padStart(2, "0");
      return hh ? `${hh}:${p2(mm)}:${p2(ss)}` : `${p2(mm)}:${p2(ss)}`;
    },
    formatDuration(sec) {
      sec = Math.max(0, Math.round(sec));
      if (sec < 60) return `${sec} s`;
      const m = Math.floor(sec / 60), s = sec % 60;
      return m < 60 ? `${m} min ${String(s).padStart(2, "0")} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
    },

    // -------------------------------------------------------------- enhance prompt / start image
    async enhanceNow(style) {
      const edit = style === "edit" || style === "outpaint";  // instructions for FLUX.2 Klein, not a whole prompt
      const contrast = style === "contrast";  // 🔄 the same scene turned into its striking opposite
      const f = this.form, owner = f.prompt_owner === "i2i" ? "i2i" : "create";  // the tab that asked
      const res = await this.runJob("enhance_prompt", { prompt: f.prompt, width: f.width, height: f.height,
                                                        model: f.model, family: edit ? style : this.family, contrast });
      if (res) {
        const here = (f.prompt_owner === "i2i" ? "i2i" : "create") === owner;
        if (here) {
          this.undoPrompt = res.original;
          f.prompt = res.prompt;
          if (!edit) this.autoNegative();
        } else if (owner === "i2i") {
          f.i2i_prompt = res.prompt;  // you moved to another tab meanwhile: it goes back where it came from
        } else {
          f.create_prompt = res.prompt;
        }
        this.status = (contrast ? "Contrast prompt written (↩ undo puts yours back)" : style === "outpaint" ? "Described what goes around the picture" : edit ? "Instruction made clearer" : "Prompt enhanced (and a matching negative prompt written if yours was empty)")
          + (here ? "" : ` (in ${owner === "i2i" ? "Image to image" : "Create"})`);
      }
    },
    async toolTiled() {
      const f = this.form;
      // Not Create's prompt: every tile would try to draw that whole scene (new faces and people in each tile). A short
      // "detail" prompt (or your own words for the texture/style) keeps the picture and adds detail.
      const r = await this.runJob("tiled_upscale", { ...this.toolArgs(), model: f.model, prompt: (this.tup.prompt || "").trim(),
                                                     scale: this.tup.scale, strength: this.tup.strength, steps: this.tup.steps,
                                                     cfg: Math.min(f.cfg || 6, 6), sampler: f.sampler, seed: f.seed });
      if (r) this.afterPost(r);
    },
    setRegionLayout(l) {
      const r = this.form.regions;
      r.layout = l;
      const n = r.prompts.length;
      if (l === "grid") r.ratios = n === 3 ? "1,1;1" : n === 4 ? "1,1;1,1" : "1;1";
      else if (/;/.test(r.ratios)) r.ratios = Array(n).fill(1).join(",");
    },
    detLoras() {  // the Detail module's LoRA list, narrowed by its search box (name, file or trigger words)
      const q = (this.detLoraQ || "").trim().toLowerCase();
      const all = this.compatibleLoras || [];
      if (!q) return all;
      return all.filter((l) => [l.name, l.file, l.label, ...(l.triggers || l.trigger_words || [])]
        .some((v) => typeof v === "string" && v.toLowerCase().includes(q)));
    },
    // -------------------------------------------------------------- right-click menus
    openMenu(e, title, items) {
      const list = items.filter(Boolean);
      while (list.length && list[list.length - 1] === "-") list.pop();
      const h = 40 + list.reduce((a, it) => a + (it === "-" ? 9 : 32), 0);
      this.ctx = null;
      this.gmenu = { x: Math.max(4, Math.min(e.clientX, innerWidth - 260)), y: Math.max(4, Math.min(e.clientY, innerHeight - h - 8)), title, items: list };
    },
    async runMenuItem(it) {
      this.gmenu = null;
      if (it.disabled || typeof it.run !== "function") return;
      try { await it.run(); } catch (e) { this.error = e.message; }
    },
    async copyText(text, what) {
      try { await navigator.clipboard.writeText(String(text || "")); this.status = `Copied ${what}`; }
      catch { this.error = "Couldn't use the clipboard."; }
    },
    menuModel(e, m) {
      const local = m.id.startsWith("local:") && (this.modelDirs?.local || []).some((x) => x.id === m.id);
      this.openMenu(e, m.label, [
        { label: "✓ Use this model", run: () => this.selectModel(m.id), disabled: m.missing },
        { label: this.isFav(m) ? "★ Remove from favourites" : "☆ Add to favourites", run: () => this.toggleFav(m) },
        "-",
        m.missing ? { label: "⟲ Reconnect… (find the file)", tip: "The file isn't where it was: moved, renamed, or its drive isn't connected. Point the app at it", run: () => this.reconnectModel(m) } : null,
        m.path && !m.missing ? { label: "📂 Show in Finder", run: () => window.bee.showInFolder(m.path) } : null,
        m.path && !m.missing ? { label: "⟲ Use a different file for it…", tip: "Point this model at another copy of the file (e.g. after moving it)", run: () => this.reconnectModel(m) } : null,
        !m.path && !m.cached && !m.missing ? { label: "⬇ Download it", run: () => this.downloadModel(m) } : null,
        m.path ? { label: "🏷 Wrong type? Fix it…", tip: "Listed as SD 1.5 but it's SDXL (or the other way round)?", run: () => this.fixLoraType(m.label.replace(/ \(your( inpainting)? file\)$/, "")) } : null,
        "-",
        { label: m.people ? "Untag People" : "Tag as People (made of real people)", run: () => this.markPeople("model", m) },
        { label: m.nsfw ? "Mark safe" : "Mark NSFW", run: () => this.markItem("model", m) },
        "-",
        { label: "⧉ Copy name", run: () => this.copyText(m.label, "the model's name") },
        m.path ? { label: "⧉ Copy file location", run: () => this.copyText(m.path, "the file's location") } : null,
        local ? { label: "Remove from the list (the file stays)", danger: true, run: () => this.removeLocalModel(m.id) } : null,
      ]);
    },
    async reconnectModel(m) {
      const name = (m.path || m.label).split("/").pop();
      const dir = m.path ? m.path.split("/").slice(0, -1).join("/") : undefined;
      const picked = await window.bee.locateFile({ title: `Where is “${name}” now?`, defaultPath: dir, folders: !/\.[a-z0-9]+$/i.test(name) });
      if (!picked) return;
      const r = await call("model_relocate", { id: m.id, path: picked });
      this.models = await call("list_models");
      const fix = (id) => (id === r.old ? r.id : id);  // your own settings that named the old file
      if (this.form.model === r.old) { this.form.model = r.id; await this.onModelChange(); }
      this.opts.default_model = fix(this.opts.default_model);
      this.opts.preload_start = fix(this.opts.preload_start);
      if (this.favModels.includes(r.old)) { this.favModels = this.favModels.map(fix); saveJSON("protoart.favModels", this.favModels); }
      this.chain.steps.forEach((st) => { st.model = fix(st.model); });
      try { this.modelDirs = await call("model_dirs_get"); } catch { /* the list is enough */ }
      this.status = `Reconnected “${r.label || name}” to ${picked}`;
    },
    menuLora(e, l, activeIndex = -1) {
      const name = l.name;
      this.openMenu(e, name, [
        activeIndex < 0 ? { label: "＋ Add to the prompt", run: () => this.addLoraByName(name) }
                        : { label: "✕ Take it out of the prompt", run: () => this.form.loras.splice(activeIndex, 1) },
        this.isFavLora && l.family ? { label: this.isFavLora(l) ? "★ Remove from favourites" : "☆ Add to favourites", run: () => this.toggleFavLora(l) } : null,
        "-",
        l.file ? { label: "📂 Show in Finder", run: () => window.bee.showInFolder(l.file) } : null,
        { label: "🔤 Trigger words…", tip: "See, copy or set the words that switch this LoRA on", run: async () => {
          await this.fetchTriggers(name);
          const w = this.activeTriggers(name);
          if (w.length) await this.copyText(w.join(", "), `the trigger words (${w.join(", ")})`);
          else this.status = `No trigger words known for ${name}: add them in its row (🔤) after adding it to the prompt`;
        } },
        { label: "🏷 Wrong type? Fix it…", run: () => this.fixLoraType(name) },
        "-",
        l.family ? { label: l.people ? "Untag People" : "Tag as People (made of real people)", run: () => this.markPeople("lora", l) } : null,
        l.family ? { label: l.nsfw ? "Mark safe" : "Mark NSFW", run: () => this.markItem("lora", l) } : null,
        "-",
        { label: "⧉ Copy name", run: () => this.copyText(name, "the LoRA's name") },
        l.file ? { label: "⧉ Copy file location", run: () => this.copyText(l.file, "the file's location") } : null,
      ]);
    },
    menuJob(e, j) {
      const queued = j.state === "queued";
      this.openMenu(e, j.label || j.cmd, [
        queued && this.queuedIndex(j) > 0 ? { label: "⤒ Start it next", run: () => this.moveJob(j.id, -this.queuedIndex(j)) } : null,
        queued ? { label: "▲ Move up", run: () => this.moveJob(j.id, -1) } : null,
        queued ? { label: "▼ Move down", run: () => this.moveJob(j.id, 1) } : null,
        j.state === "running" && this.canSkipImage ? { label: "⏭ Skip this picture", run: () => this.skipImage() } : null,
        "-",
        { label: j.state === "running" ? "■ Stop it" : "✕ Take it out of the queue", danger: true, run: () => this.cancelJob(j.id) },
      ]);
    },
    menuFolder(e, f) {
      this.openMenu(e, f.path, [
        f.found ? { label: "📂 Show in Finder", run: () => window.bee.openFolder(f.path) } : null,
        { label: f.found ? "⟲ Moved it… (point to its new place)" : "⟲ Reconnect… (find it)", run: () => this.relinkFolder(f.path) },
        { label: "⧉ Copy location", run: () => this.copyText(f.path, "the folder's location") },
        "-",
        { label: "Stop using this folder (files stay)", danger: true, run: () => this.forgetFolder(f.path) },
      ]);
    },
    regionsByParts() {  // FLUX.1 / ERNIE on the fast engine: regions are drawn by parts
      return this.currentModel?.engine === "mlx" && ["flux", "ernie", "flux2"].includes(this.family);
    },
    // ✨ on an area: a short prompt becomes a detailed description of that one subject (tags for SD models)
    async enhanceRegion(i) {
      const r = this.form.regions, text = (r.prompts[i] || "").trim();
      if (!text) return (this.error = "Write something in this area first (e.g. “a vintage 60s car”).");
      const tags = ["sd15", "sdxl"].includes(this.family) && this.currentModel?.engine !== "mlx" && !this.naturalPrompts();
      this.regionBusy = i;
      try {
        const res = await this.runJob("enhance_prompt", { prompt: text, width: this.form.width, height: this.form.height,
                                                          model: this.form.model, family: tags ? "region_tags" : "region" });
        if (res && res.prompt) {
          this.regionUndo = { ...this.regionUndo, [i]: text };
          r.prompts[i] = res.prompt;
          this.status = `Area ${i + 1} described in detail (↩ puts yours back)`;
        }
      } finally { this.regionBusy = -1; }
    },
    tupPreset() {
      const t = this.tup;
      const p = TILED_PRESETS.find((x) => x.set.scale === t.scale && Math.abs(x.set.strength - t.strength) < 0.001 && x.set.steps === t.steps);
      return p ? p.id : "";
    },
    // model lists: ★ favourites first, then the order picked in the list's Sort menu (A–Z, Z–A, biggest, the usual)
    sortModels(list) {
      const how = this.opts.model_sort || "";
      const name = (m) => (m.label || "").replace(/^[^a-z0-9]+/i, "").toLowerCase();
      const size = (m) => Number(m.disk_gb || m.download_gb || 0);
      const by = how === "az" ? (a, b) => name(a).localeCompare(name(b), undefined, { numeric: true })
        : how === "za" ? (a, b) => name(b).localeCompare(name(a), undefined, { numeric: true })
        : how === "size" ? (a, b) => size(b) - size(a)
        : () => 0;
      return [...list].sort((a, b) => ((this.isFav(b) ? 1 : 0) - (this.isFav(a) ? 1 : 0)) || by(a, b));
    },
    setModelSort(v) { this.opts.model_sort = v; this.saveOpts(); },
    needsDownload(msg) { return !!(this.dlJob && /one-time download/i.test(msg || "")); },
    // ⬇ Download it now: run the same job again with the connection open for it, so whatever it needs is fetched
    async downloadAndRetry() {
      const job = this.dlJob;
      if (!job || this.dlBusy) return;
      this.dlBusy = true;
      this.error = "";
      this.status = "Downloading what this needs, then running it again…";
      try {
        const r = await this.runJob(job.cmd, { ...job.args, allow_download: true });
        if (r) { this.dlJob = null; this.status = "Downloaded: from now on this works offline."; }
      } finally { this.dlBusy = false; }
    },
    async runQuantize() {
      this.qnt.msg = "";
      const r = await this.runJob("quantize", { model: this.qnt.model, bits: this.qnt.bits });
      if (!r) return;
      this.qnt.msg = `✓ “${r.label}” (${r.gb} GB) is in the model list.`;
      this.models = await call("list_models").catch(() => this.models);
    },
    cvtFamily() {
      return (this.mergeModels.find((m) => m.path === this.cvt.model) || {}).family || "";
    },
    async runLoraExtract() {  // Tools → Convert → LoRA
      const c = this.cvt;
      c.msg = "";
      const r = await this.runJob("lora_extract", { model: c.model, base: c.base, rank: c.rank, text_encoder: c.te, name: c.name.trim() || null });
      if (!r) return;
      c.msg = `✓ “${r.name}” (${r.mb} MB, ${r.layers} layers) is in your LoRA list${r.family ? " for " + this.familyName(r.family) : ""}.`;
      this.loras = await call("list_loras").catch(() => this.loras);
    },
    async runMerge() {
      const g = this.mrg;
      g.msg = "";
      const r = await this.runJob("ckpt_merge", { a: g.a, b: g.b, c: g.mode === "add" ? g.c : null, mode: g.mode, multiplier: g.m, name: g.name.trim() || null });
      if (!r) return;
      g.msg = `✓ Saved “${r.name}” (${r.gb} GB): pick it in Create's model list.`;
      this.models = await call("list_models").catch(() => this.models);
    },
    async loadCkptList() {
      try {
        const r = await call("ckpt_list");
        this.ck.items = r.items;
        this.ck.out_dir = r.out_dir;
      } catch (e) { this.error = e.message; }
    },
    async pickCkpt() {
      const picked = await window.bee.pickModels();
      const f = (Array.isArray(picked) ? picked : [picked]).find((x) => /\.(ckpt|pt|pth)$/i.test(x || ""));
      if (f) this.convertCkpt(f);
      else if (picked && (!Array.isArray(picked) || picked.length)) this.error = "Pick a .ckpt file.";
    },
    async convertCkpt(path) {
      this.ck.msg = "";
      const r = await this.runJob("ckpt_convert", { path, half: this.ck.half });
      if (!r) return;
      this.ck.msg = `✓ Saved ${r.path.split("/").pop()} (${r.gb} GB): pick it in Create's model list.`;
      this.models = await call("list_models").catch(() => this.models);
      this.loadCkptList();
    },
    async pickConvertFiles() {
      const picked = await window.bee.pickTdict();
      if (picked && picked.length) this.loadConvertList(picked);
    },
    async loadConvertList(picked) {
      try {
        const r = await call("convert_list", Array.isArray(picked) ? { paths: picked } : {});
        this.conv.picked = !!r.picked;
        if (r.picked) this.conv.selected = r.items.map((i) => i.file);  // what you picked is ticked
        this.conv.items = r.items;
        this.conv.free_gb = r.free_gb;
        if (!this.conv.out_dir) this.conv.out_dir = r.default_out;
        if (!r.items.length) this.error = r.picked ? "No DiffusionBee files (.tdict) that can be converted were found there."
          : "No DiffusionBee files found (~/.diffusionbee). Use “Pick files or a folder…” for ones somewhere else.";
      } catch (e) {
        this.error = e.message;
      }
    },
    convKindLabel(k) {
      return { lora: "LoRA", checkpoint: "Model", embedding: "Embedding", controlnet: "ControlNet" }[k] || k;
    },
    convSelectShown(on) {
      const shown = new Set(this.convShown.map((i) => i.file));
      this.conv.selected = on ? [...new Set([...this.conv.selected, ...shown])] : this.conv.selected.filter((f) => !shown.has(f));
    },
    async convPickOut() {
      const d = await window.bee.pickFolder({ title: "Where should converted files go?", defaultPath: this.conv.out_dir, button: "Use this folder" });
      const dir = Array.isArray(d) ? d[0] : d;
      if (dir) this.conv.out_dir = dir;
    },
    async runConvert() {
      this.conv.result = null;
      const r = await this.runJob("convert_tdict", { files: [...this.conv.selected], out_dir: this.conv.out_dir,
                                                     loras_to_lora_folder: this.conv.loras_to_lora_folder });
      if (r) {
        this.conv.result = r;
        this.conv.selected = [];
        await this.refreshLoras();
        this.models = await call("list_models");
        this.status = `Converted ${r.converted.length} file${r.converted.length === 1 ? "" : "s"} to .safetensors`;
      }
    },
    // -------------------------------------------------------------- edit in PhotoCraft (or another editor)
    editorName() {
      return (this.opts.editor || "PhotoCraft").replace(/^Adobe /, "").replace(/ \d{4}$/, "");
    },
    async editIn(r) {
      // r: a library record ({id, path}) or any picture ({path})
      this.error = "";
      this.edits.notInstalled = false;
      try {
        const res = await call("edit_start", r.id != null ? { image_id: r.id, editor: this.opts.editor } : { path: r.path, editor: this.opts.editor });
        this.edits.folder = res.folder;
        const opened = await window.bee.openInEditor?.(res.file, this.opts.editor);
        if (!opened) {
          this.error = "Quit and reopen Protoart Studio once to enable editing (the app itself was updated).";
          return;
        }
        if (!opened.ok) {
          if (opened.notInstalled) this.edits.notInstalled = true;
          else this.error = `Couldn't open ${this.editorName()}: ${opened.error || "unknown error"}`;
          await call("edit_finish", { name: res.file.split("/").pop() });
          return;
        }
        this.status = `Opened in ${this.editorName()}: save there (⌘S) and the edited version appears in your Library`;
        await this.refreshEdits();
      } catch (e) {
        this.error = e.message;
      }
    },
    async refreshEdits() {
      try {
        const r = await call("edit_pending");
        this.edits.pending = r.pending;
        this.edits.folder = r.folder;
      } catch {
        return;
      }
      this.watchEdits();
    },
    watchEdits() {
      if (this.edits.pending.length && !this.edits.timer) {
        this.edits.timer = setInterval(() => this.checkEdits(), 2500);
      } else if (!this.edits.pending.length && this.edits.timer) {
        clearInterval(this.edits.timer);
        this.edits.timer = null;
      }
    },
    async checkEdits() {
      try {
        const r = await call("edit_check");
        this.edits.pending = r.pending;
        if (r.imported.length) {
          this.status = `✏️ Saved ${r.imported.length} edited version${r.imported.length === 1 ? "" : "s"} to the Library`;
          if (this.tab === "library") this.queryLibrary();
        }
      } catch {
        /* backend busy or restarting: try again next time */
      }
      this.watchEdits();
    },
    async finishEdits(name) {
      await this.checkEdits(); // pick up a last save first
      try {
        this.edits.pending = (await call("edit_finish", name ? { name } : {})).pending;
      } catch (e) {
        this.error = e.message;
      }
      this.watchEdits();
    },
    getPhotoCraft() {
      window.bee.openExternal?.("https://github.com/storytold/photocraft/releases");
    },
    askReset(kind) {
      this.resetAsk = kind;
    },
    setTheme(v) {
      this.theme = window.RackTheme ? window.RackTheme.set(v) : v;
    },
    async setStrictOffline(on) {
      try {
        const r = await call("privacy_set", { strict: !!on });
        this.strictOffline = !!(r && r.strict);
        this.status = this.strictOffline ? "Strict offline: the app won't go online" : "Downloads allowed again (only what you ask for)";
      } catch (e) {
        this.error = "Couldn't change Strict offline: " + e.message;
      }
    },
    setSkin(v) {
      this.skin = window.RackTheme && window.RackTheme.setSkin ? window.RackTheme.setSkin(v) : v;
    },
    async doReset(kind) {
      this.resetAsk = "";
      try {
        const r = await call("settings_reset", { folders: kind === "folders", settings: kind === "settings" });
        if (kind === "folders") {
          this.imagesDir = await call("images_dir_get");
          this.loraDirs = await call("lora_dirs_get");
          this.modelDirs = await call("model_dirs_get");
          await this.refreshLoras();
          this.models = await call("list_models");
          this.status = "Folder locations reset · images now go to " + r.images_dir;
          return;
        }
        // settings: forget everything this window remembered, then start fresh
        for (const k of [OPTS_KEY, FORM_KEY, CLIP_KEY, "protoart.loraRand", "beestudio.i2p", "beestudio.rnd", "protoart.train", "protoart.theme"]) {
          try { localStorage.removeItem(k); } catch {}
        }
        await window.bee.resetWindow?.();
        location.reload();
      } catch (e) {
        this.error = e.message;
      }
    },
    selectLog() {
      const box = this.$refs.logBox;
      if (!box) return;
      const range = document.createRange();
      range.selectNodeContents(box);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      box.focus();
    },
    async clearLog() {
      try {
        await call("log_clear");
        this.logs = "";
        this.status = "Backend log cleared";
      } catch (e) {
        this.error = e.message;
      }
    },
    async copyLog() {
      const text = this.logs || "";
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        this.selectLog(); // fallback: copy the selected text
        document.execCommand("copy");
      }
      this.logCopied = true;
      setTimeout(() => (this.logCopied = false), 1800);
    },
    async loadLogTail() {
      try {
        const r = await call("log_tail", { bytes: 30000 });
        this.logPath = r.path;
        if (r.text !== undefined) this.logs = r.text;
      } catch {
        /* backend not ready yet */
      }
    },
    async openI2I() {
      this.tab = "i2i";
      if (!["sd15", "sdxl"].includes(this.family)) {
        // ERNIE / other models can't do image to image: switch to an SD model
        const sd = this.models.find((m) => m.family === "sd15" && m.cached) || this.models.find((m) => m.family === "sd15")
          || this.models.find((m) => m.family === "sdxl" && m.engine !== "mlx");
        if (sd) {
          this.selectModel(sd.id);
          this.status = `Switched to ${sd.label} for image to image`;
        }
      }
      if (!this.controlTypes.length) this.controlTypes = await call("control_types").catch(() => []);
    },
    applyCnGuide(auto = false) {  // ControlNet → "Use these settings": the guide's strength, timing and picture settings
      const g = this.cnGuide;
      if (!g || !g.set) return;
      const c = g.set, f = this.form;
      this.cnApplied = { type: f.control.type, auto: auto === true,
                         before: { scale: f.control.scale, start: f.control.start, end: f.control.end, i2i_use_init: f.i2i_use_init,
                                   strength: f.strength, steps: f.steps, cfg: f.cfg, width: f.width, height: f.height,
                                   use_init_size: f.use_init_size } };
      f.control.scale = c.scale ?? f.control.scale;
      f.control.start = c.start ?? f.control.start;
      f.control.end = c.end ?? f.control.end;
      if (c.use_init != null) f.i2i_use_init = c.use_init;
      if (c.strength != null) f.strength = c.strength;
      if (c.steps) f.steps = c.steps;
      if (c.cfg) f.cfg = c.cfg;
      // big sizes cost a lot with a ControlNet on a Mac under 16 GB: 512 there (768 has 2¼ times the work)
      const size = this.cnRecSize;
      if (size) { f.width = size; f.height = size; f.use_init_size = false; }
      this.status = "Recommended settings for " + g.title + " are in";
    },
    undoCnGuide() {  // put back the settings that were there before the guide's
      const a = this.cnApplied, f = this.form;
      if (!a) return;
      const b = a.before;
      Object.assign(f.control, { scale: b.scale, start: b.start, end: b.end });
      Object.assign(f, { i2i_use_init: b.i2i_use_init, strength: b.strength, steps: b.steps, cfg: b.cfg, width: b.width,
                         height: b.height, use_init_size: b.use_init_size });
      this.cnApplied = null;
      this.status = "Your own settings are back";
    },
    switchToControlNet() {  // from Edit with words / Inpaint: go to Change it with an SD 1.5 model for ControlNet
      const sd = this.installedModels.find((m) => m.family === "sd15" && m.engine !== "mlx" && m.id === this.form.model)
        || this.installedModels.find((m) => m.family === "sd15" && m.engine !== "mlx");
      this.setI2iMode("change");
      if (sd && this.form.model !== sd.id) this.selectModel(sd.id);
      this.status = sd ? `ControlNet is on: ${sd.label} (SD 1.5) is picked` : "Add an SD 1.5 model in the Models tab to use ControlNet";
    },
    async previewControlMap() {
      if (!this.form.init_image || !this.form.control.type) return;
      const r = await this.runJob("control_preview", { path: this.form.init_image, type: this.form.control.type,
                                                       already_map: this.form.control.already_map,
                                                       depth_model: this.form.control.depth_model || "small" });
      if (r) this.controlMap = r.src;
    },
    async generateI2I() {
      if (!["edit", "remove"].includes(this.form.i2i_mode)) this.dropMismatchedLoras();
      const f = this.form;
      if (!f.init_image) return (this.error = "Add a picture first.");
      if (f.i2i_mode === "faceswap") {  // 🔁 inswapper_128: no model or prompt involved
        const sw = f.swap || {};
        if (!sw.face) return (this.error = "Add the photo of the face to use.");
        if (!sw.consent) return (this.error = "Tick “I have permission from the person whose face this is” first.");
        this.error = "";
        await this.runJob("faceswap", { init_image: f.init_image, face: sw.face, which: sw.which || "all", boost: sw.boost || 1,
                                        restore: sw.restore !== false, consent: true });
        return;
      }
      if (f.i2i_mode === "remove") {  // 🩹 spot healing: no model or prompt involved
        if (!this.mask.data || this.mask.for !== f.init_image) return (this.error = "Paint over what you want to remove first.");
        this.error = "";
        await this.runJob("heal", { init_image: f.init_image, mask: this.mask.data, method: f.heal_method || "lama" });
        return;
      }
      if (f.i2i_mode === "edit") {  // ✨ FLUX.2 Klein, whatever model Create has
        if (!(f.prompt || "").trim()) return (this.error = "Describe the change in the prompt, e.g. “make the jacket red” or “turn it into a watercolour”.");
        this.error = "";
        const painted = this.mask.data && this.mask.for === f.init_image ? this.mask.data : null;
        await this.runJob("edit_words", { init_image: f.init_image, prompt: f.prompt, mask: painted,
                                          refs: (f.edit_refs || []).slice(0, 3), seed: f.seed });
        return;
      }
      if (f.i2i_mode === "outpaint" && f.outpaint_engine !== "model") {  // ↔ Expand with FLUX.2 Klein + outpaint LoRA
        const o = f.outpaint;
        if (!(o.left || o.right || o.top || o.bottom)) return (this.error = "Choose how far to expand on at least one side.");
        this.error = "";
        await this.runJob("edit_words", { init_image: f.init_image, prompt: f.prompt || "", seed: f.seed,
                                          outpaint: { left: o.left || 0, right: o.right || 0, top: o.top || 0, bottom: o.bottom || 0 } });
        return;
      }
      if (f.i2i_mode === "change" && !f.i2i_use_init && !f.control.type) return (this.error = "Tick “Start from this picture”, pick a ControlNet, or both.");
      if (!["sd15", "sdxl"].includes(this.family)) return (this.error = "Image to image works with SD 1.5 and SDXL models: pick one at the top.");
      const cnFam = f.i2i_mode === "change" && f.control.type ? (this.controlTypeInfo?.family || "sd15") : null;
      if (cnFam && cnFam !== this.family) return (this.error = `This ControlNet is made for ${cnFam === "sdxl" ? "SDXL (Pony, Illustrious, NoobAI)" : "SD 1.5"} models: pick one in the model list at the top, or another ControlNet.`);
      if (!(await this.slowCheck("i2i"))) return;
      this.error = "";
      const args = JSON.parse(JSON.stringify(f));
      this.cleanExtras(args);
      delete args.hires; // Hi-res fix is for making new pictures (Create)
      args.fast = false;
      if (f.i2i_mode === "inpaint") {
        if (!this.mask.data || this.mask.for !== f.init_image) return (this.error = "Paint over the part you want to change first.");
        Object.assign(args, { mask: this.mask.data, strength: f.inpaint.strength, mask_blur: f.inpaint.blur,
                              only_masked: f.inpaint.only_masked });
        delete args.control;
        delete args.outpaint;
      } else if (f.i2i_mode === "outpaint") {
        const o = f.outpaint;
        if (!(o.left || o.right || o.top || o.bottom)) return (this.error = "Choose how far to expand on at least one side.");
        delete args.control;
      } else {
        delete args.outpaint;
        args.control_image = f.init_image;
        if (!f.i2i_use_init) args.init_image = "";
        if (!f.control.type) delete args.control;
      }
      delete args.inpaint;
      const im = this.inpaintModel;
      if (im && ["inpaint", "outpaint"].includes(f.i2i_mode)) args.model = im.id;  // a model made for inpainting
      const res = await this.runJob("generate", args);
      if (res) this.models = await call("list_models");
    },
    // VAE: each model remembers the VAE you used with it; a VAE made for another model type is never kept
    pickVaeFor(m) {
      const saved = (this.vaeByModel || {})[m.id];
      const ok = (name) => this.vaes.some((v) => v.name === name && v.family === m.family);
      this.form.vae = saved && ok(saved) ? saved : "";  // never a VAE made for another model type
    },
    rememberVae() {
      if (!this.form.model) return;
      this.vaeByModel = { ...(this.vaeByModel || {}), [this.form.model]: this.form.vae || "" };
      saveJSON("protoart.vaeByModel", this.vaeByModel);
    },
    async getRecommendedVae() {
      const fam = this.family;
      const row = await this.runJob("vae_get", { family: fam });
      if (!row) return;
      await this.refreshExtras(false);
      if (this.family === fam) {
        this.form.vae = row.name;
        this.rememberVae();
      }
      this.status = `${row.name} is ready and picked for this model`;
    },
    async refreshExtras(all = true) {
      const [v, c] = await Promise.all([call("list_vaes").catch(() => null), call("list_controlnets").catch(() => null)]);
      if (v) { this.vaes = v.vaes; this.vaeFolder = v.folder; }
      if (c) { this.userControlnets = c.controlnets; this.controlnetFolder = c.folder; }
      if (all) {
        this.embeddings = await call("list_embeddings").catch(() => this.embeddings);
        this.controlTypes = await call("control_types").catch(() => this.controlTypes);
      }
    },
    pickI2iModel(id) {  // Image to image's model list: FLUX.2 Klein means ✨ Edit with words (or Expand with its LoRA)
      if (id === "flux2_klein_mlx_q4") {
        if (this.form.i2i_mode === "outpaint") { this.form.outpaint_engine = "klein"; return; }
        return this.setI2iMode("edit");
      }
      if (this.form.i2i_mode === "edit") this.setI2iMode("change");
      if (this.form.i2i_mode === "outpaint") this.form.outpaint_engine = "model";  // expand with the model you picked
      this.selectModel(id);
    },
    editModel() {  // the model Edit with words uses
      return this.models.find((m) => m.id === "flux2_klein_mlx_q4") || null;
    },
    async addEditRefs(files) {
      for (const file of files) {
        if ((this.form.edit_refs || []).length >= 3) break;
        const p = await this.preparePicture(window.bee.pathForFile(file));
        if (p) this.form.edit_refs = [...(this.form.edit_refs || []), p];
      }
    },
    onEditRefDrop(e) {
      this.editRefOver = false;
      this.addEditRefs([...(e.dataTransfer?.files || [])].filter(isPicture));
    },
    onEditRefPicked(e) {
      const files = [...(e.target.files || [])];
      e.target.value = "";
      this.addEditRefs(files);
    },
    onStartDrop(e) {
      this.startDragOver = false;
      const file = [...(e.dataTransfer?.files || [])].find(isPicture);
      if (file) this.preparePicture(window.bee.pathForFile(file)).then((p) => { if (p) { this.form.init_image = p; this.controlMap = null; } });
    },
    async onSwapDrop(e) {
      this.swapOver = false;
      const file = [...(e.dataTransfer?.files || [])].find(isPicture);
      if (file) { const p = await this.preparePicture(window.bee.pathForFile(file)); if (p) this.form.swap.face = p; }
    },
    async onSwapPicked(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) { const p = await this.preparePicture(window.bee.pathForFile(file)); if (p) this.form.swap.face = p; }
    },
    onStartPicked(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) this.preparePicture(window.bee.pathForFile(file)).then((p) => { if (p) this.form.init_image = p; });
    },
    // Tools → 🩹 Remove objects: the picture opens in Image to image, in Remove mode, ready to paint
    openRemove(src) {
      if (!src) return;
      this.form.init_image = src.path;
      this.form.use_init_size = true;
      this.form.i2i_mode = "remove";
      this.controlMap = null;
      this.detail = null;
      this.openI2I();
      this.status = "Paint over what you want to remove, then press 🩹 Remove";
    },
    async toolVectorize() {
      const v = this.post.vec;
      this.vecResult = null;
      const r = await this.runJob("vectorize", { ...this.toolArgs(), colors: v.colors, detail: v.detail, smooth: v.smooth });
      if (r) this.vecResult = r;
    },
    async useAsStart(r) {
      if (!["sd15", "sdxl"].includes(this.family)) {
        const sd = this.models.find((m) => m.family === "sdxl") || this.models.find((m) => m.family === "sd15");
        if (sd) this.selectModel(sd.id);
      }
      this.form.init_image = r.path;
      this.form.use_init_size = true;
      this.form.i2i_use_init = true;
      this.controlMap = null;
      this.detail = null;
      this.openI2I();
    },

    // -------------------------------------------------------------- image -> prompt
    onSettingsPicked(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) this.loadSettingsFile(window.bee.pathForFile(file));
    },
    async loadSettingsFile(path) {  // 📥 a picture's settings (A1111 / ComfyUI / InvokeAI / this app) or a .json
      try {
        const r = await call("read_settings", { path });
        this.applyImported(r);
      } catch (e) { this.error = e.message; }
    },
    applyImported(r) {
      const f = this.form, notes = [];
      this.undoPrompt = f.prompt;
      if (r.prompt) f.prompt = r.prompt;
      if (r.negative_prompt != null) f.negative_prompt = r.negative_prompt;
      for (const k of ["steps", "cfg", "seed", "clip_skip"]) if (r[k] != null) f[k] = r[k];
      if (r.width && r.height) { f.width = Math.round(r.width / 8) * 8; f.height = Math.round(r.height / 8) * 8; }
      if (r.model_name) {  // the model, if one of yours has the same name
        const want = String(r.model_name).toLowerCase().replace(/\.(safetensors|ckpt)$/, "").split(/[\\/]/).pop();
        const m = this.models.find((x) => this.isInstalled(x) && (x.label || "").toLowerCase().includes(want))
          || this.models.find((x) => this.isInstalled(x) && String(x.path || x.id).toLowerCase().includes(want));
        if (m) { if (m.id !== f.model) this.selectModel(m.id); } else notes.push(`model “${r.model_name}” isn't here`);
      }
      if (r.sampler) {  // A1111 / ComfyUI sampler names → this app's
        const norm = (x) => String(x).toLowerCase().replace(/[^a-z0-9+]/g, "");
        const want = norm(r.sampler) + (r.scheduler && /karras/i.test(r.scheduler) ? "karras" : "");
        const s = this.samplers.find((x) => norm(x.label) === want || norm(x.id) === want)
          || this.samplers.find((x) => norm(x.label).startsWith(norm(r.sampler)) || norm(x.id) === norm(r.sampler));
        if (s) f.sampler = s.id; else notes.push(`sampler “${r.sampler}”`);
      }
      if (Array.isArray(r.loras) && r.loras.length) {
        const got = [];
        for (const l of r.loras) {
          const have = (this.loras || []).find((x) => x.name.toLowerCase() === String(l.name).toLowerCase());
          if (have && !f.loras.some((x) => x.name === have.name)) { f.loras.push({ name: have.name, weight: Number(l.weight) || 1 }); got.push(have.name); }
          else if (!have) notes.push(`LoRA “${l.name}”`);
        }
      }
      this.status = `📥 Settings loaded from ${r.source || "the file"}` + (notes.length ? ` · not found here: ${notes.slice(0, 3).join(", ")}${notes.length > 3 ? "…" : ""}` : "") + " · ↩ undo puts your prompt back";
    },
    onDrop(e) {
      this.dragOver = false;
      const json = [...(e.dataTransfer?.files || [])].find((x) => /\.json$/i.test(x.name || ""));
      if (json) return this.loadSettingsFile(window.bee.pathForFile(json));  // a .json of settings / a ComfyUI workflow
      const file = [...(e.dataTransfer?.files || [])].find(isPicture);
      if (!file) {
        this.error = "Drop a picture (PNG, JPEG, WebP, HEIC, TIFF, AVIF, PSD, GIF, BMP or camera RAW).";
        return;
      }
      this.imageToPrompt(window.bee.pathForFile(file));
    },
    onFilePicked(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) this.imageToPrompt(window.bee.pathForFile(file));
    },
    async suggestI2iPrompt() {  // 💡 Image to image: describe the picture as a prompt (tags for tag-reading models)
      const f = this.form;
      if (!f.init_image) return (this.error = "Add a picture first.");
      const tags = this.family === "anima" || ["pony", "illustrious", "noobai"].includes(this.currentFlavor);
      this.i2iSuggesting = true;
      try {
        const res = await this.runJob("interrogate", { path: f.init_image, mode: tags ? "tags" : "quick", detail: "detailed" });
        if (!res || !res.prompt) return;
        const text = String(res.prompt).replace(/\s+/g, " ").trim();
        if (f.prompt_owner !== "i2i") {  // you went to another tab meanwhile: it goes into Image to image's prompt
          f.i2i_prompt = text;
          this.status = "Suggested a prompt from your picture · it's in Image to image's prompt box";
          return;
        }
        this.undoPrompt = f.prompt;
        f.prompt = text;
        this.status = "Suggested a prompt from your picture · change a word or two to change the picture (↩ undo puts yours back)";
      } finally { this.i2iSuggesting = false; }
    },
    async suggestExpandPrompt() {  // 💡 Expand: describe the picture, then ask for more of the same scene around it
      const f = this.form;
      if (!f.init_image) return (this.error = "Add a picture first.");
      const res = await this.runJob("interrogate", { path: f.init_image, mode: "quick", detail: "detailed" });
      if (!res || !res.prompt) return;
      const desc = String(res.prompt).replace(/\s+/g, " ").trim().replace(/[.\s]+$/, "");
      const text = `The same scene continues naturally around the picture, with matching lighting, colours and style: ${desc}.`;
      if (f.prompt_owner !== "i2i") {  // you went to another tab meanwhile: it goes into Image to image's prompt
        f.i2i_prompt = text;
        this.status = "Suggested a prompt from your picture · it's in Image to image's prompt box";
        return;
      }
      this.undoPrompt = f.prompt;
      f.prompt = text;
      this.status = "Suggested a prompt from your picture · edit it, or press ✨ enhance now";
    },
    async imageToPrompt(path) {
      if (!path) {
        this.error = "Couldn't read that file's location. Try saving it to disk first.";
        return;
      }
      const res = await this.runJob("interrogate", { path, mode: this.i2p.mode, detail: "detailed" });
      if (!res) return;
      this.form.prompt = this.i2p.append && this.form.prompt.trim()
        ? `${this.form.prompt.trim()}, ${res.prompt}` : res.prompt;
      this.status = this.i2p.mode === "tags"
        ? `Tagged image (${res.tags.length} tags${res.rating ? ", rating: " + res.rating : ""})`
        : "Described image" + (res.model ? " with " + res.model : "");
    },

    // -------------------------------------------------------------- random prompt
    naturalPrompts() {  // models that read sentences get sentence prompts from 🎲 Random prompt
      return ["ernie", "flux", "flux2"].includes(this.family);
    },
    randomPrompt() {
      if (this.naturalPrompts()) {
        this.form.prompt = window.NaturalPrompt.generate(this.rnd).prompt;
        this.autoNegative();
        this.status = `Random ${FAMILY_LABEL[this.family]} prompt written as sentences`;
        return;
      }
      const r = window.DanbooruPrompt.generate(this.rnd);
      this.form.prompt = r.prompt;
      this.autoNegative();
    },
    // ✨ suggest / 🎲 random negative prompt: fits the prompt and the model type (tags or plain words)
    makeNegative(random) {
      const opts = { family: this.family, flavor: this.currentFlavor };
      return random ? window.NegativePrompt.random(this.form.prompt, opts) : window.NegativePrompt.suggest(this.form.prompt, opts);
    },
    suggestNegative(random) {
      this.undoNegative = this.form.negative_prompt;
      const embeds = this.compatibleEmbeddings.map((e) => e.name).filter((n) => this.inNegative(n));  // keep your embeddings
      this.form.negative_prompt = [...embeds, this.makeNegative(random)].join(", ");
      this.lastAutoNegative = this.form.negative_prompt;
      this.status = "Negative prompt written for " + this.familyLabel + (random ? " (with random extras)" : "") + ": edit it as you like";
    },
    autoNegative() {
      // a new prompt from 🎲 Random prompt or ✨ enhance: write the matching negative too, unless you typed your own
      const cur = (this.form.negative_prompt || "").trim();
      const embeds = this.compatibleEmbeddings.map((e) => e.name);
      const onlyEmbeds = cur.split(",").map((x) => x.trim()).filter(Boolean).every((x) => embeds.includes(x));
      if (cur && cur !== this.lastAutoNegative && !onlyEmbeds) return;
      const keep = embeds.filter((n) => this.inNegative(n));
      this.form.negative_prompt = [...keep, this.makeNegative(false)].join(", ");
      this.lastAutoNegative = this.form.negative_prompt;
    },
    // 📒 My prompts: saved prompt + negative prompt pairs (kept on this Mac)
    savePromptPreset(kind) {  // opens 📒 My prompts with a name ready to edit (kind "negative": the negative prompt only)
      this.promptBox = { name: this.promptPresetName(kind), kind: kind === "negative" ? "negative" : "prompt" };
    },
    exportPrompts() {  // 📒 My prompts → a .json file (the save dialog asks where)
      const data = { app: "Protoart Studio", kind: "prompts", version: 1, prompts: this.promptPresets };
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `Protoart prompts ${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      this.promptIoMsg = `Exported ${this.promptPresets.length} prompt${this.promptPresets.length === 1 ? "" : "s"}.`;
    },
    importPrompts(e) {  // a .json from Export, or a .txt with one prompt per line (# lines skipped)
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file) return;
      if (file.size > 5 * 1024 * 1024) { this.promptIoMsg = "That file is too big for a prompt list."; return; }
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        let incoming = [];
        try {
          const j = JSON.parse(text);
          const arr = Array.isArray(j) ? j : Array.isArray(j?.prompts) ? j.prompts : [];
          incoming = arr.filter((p) => p && typeof p === "object" && typeof p.name === "string")
            .map((p) => p.kind === "negative"
              ? { name: p.name.slice(0, 120), kind: "negative", negative: String(p.negative || "").slice(0, 4000) }
              : { name: p.name.slice(0, 120), prompt: String(p.prompt || "").slice(0, 4000), negative: String(p.negative || "").slice(0, 4000) });
        } catch {
          incoming = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith("#"))
            .map((l) => ({ name: l.replace(/[(),:\[\]]/g, " ").split(/\s+/).filter(Boolean).slice(0, 5).join(" ").slice(0, 60) || "Imported prompt", prompt: l.slice(0, 4000), negative: "" }));
        }
        if (!incoming.length) { this.promptIoMsg = "No prompts found in that file."; return; }
        const list = [...this.promptPresets];
        let added = 0, renamed = 0;
        for (const p of incoming) {
          const same = list.find((x) => x.name === p.name);
          if (same && (same.prompt || "") === (p.prompt || "") && (same.negative || "") === (p.negative || "")) continue;  // already there
          if (same) {  // same name, different text: keep both
            let i = 2, base = p.name;
            while (list.some((x) => x.name === p.name)) p.name = `${base} (${i++})`;
            renamed++;
          }
          list.push(p);
          added++;
        }
        this.promptPresets = list.slice(0, 500);
        saveJSON("protoart.prompts", this.promptPresets);
        this.promptIoMsg = `Imported ${added} prompt${added === 1 ? "" : "s"}` + (renamed ? ` (${renamed} renamed to keep both)` : "") + (added < incoming.length ? ` · ${incoming.length - added} were already here` : "") + ".";
      };
      reader.readAsText(file);
    },
    confirmSavePrompt() {
      const name = (this.promptBox?.name || "").trim();
      if (!name) return;
      const neg = this.promptBox?.kind === "negative";
      const list = this.promptPresets.filter((p) => p.name !== name);
      list.unshift(neg ? { name, kind: "negative", negative: this.form.negative_prompt || "" }
                       : { name, prompt: this.form.prompt, negative: this.form.negative_prompt || "" });
      this.promptPresets = list.slice(0, 500);
      saveJSON("protoart.prompts", this.promptPresets);
      this.promptBox = null;
      this.status = `Saved “${name}” in 📒 My prompts`;
    },
    promptPresetName(kind) {
      const src = kind === "negative" ? (this.form.negative_prompt || "") : this.form.prompt;
      const words = (kind === "negative" ? "Negative: " : "") + src.replace(/[(),:\[\]]/g, " ").split(/\s+/).filter(Boolean).slice(0, 5).join(" ");
      let name = words.trim() || (kind === "negative" ? "My negative prompt" : "My prompt");
      let i = 2;
      while (this.promptPresets.some((p) => p.name === name)) name = `${words} (${i++})`;
      return name;
    },
    applyPromptPreset(name) {
      const p = this.promptPresets.find((x) => x.name === name);
      if (!p) return;
      if (p.kind === "negative") {  // a negative prompt preset: only the negative prompt changes
        this.undoNegative = this.form.negative_prompt;
        this.form.negative_prompt = p.negative || "";
        this.status = `“${name}” is in the negative prompt`;
        return;
      }
      this.undoPrompt = this.form.prompt;
      this.form.prompt = p.prompt;
      if (p.negative != null) this.form.negative_prompt = p.negative;
      this.status = `“${name}” is in the prompt boxes`;
    },
    deletePromptPreset() {  // opens 📒 My prompts to pick which to delete
      this.promptBox = { name: null };
    },
    async removePromptPreset(p) {
      if (!(await this.confirmAsk(`Delete “${p.name}”?`, "It's removed from 📒 My prompts. Pictures made with it aren't affected.", "Delete"))) return;
      this.promptPresets = this.promptPresets.filter((x) => x.name !== p.name);
      saveJSON("protoart.prompts", this.promptPresets);
    },

    // -------------------------------------------------------------- post-processing
    async restoreFaces(r) {
      const rec = await this.runJob("postprocess", { action: "faces", image_id: r.id, ...this.faceArgs() });
      if (rec) this.afterPost(rec);
    },
    async enhance(r) {
      const { seed, ...options } = this.post.enh;
      const rec = await this.runJob("postprocess", { action: "enhance", image_id: r.id, seed, options });
      if (rec) this.afterPost(rec);
    },
    afterPost(rec) {
      if (this.tab === "library") this.queryLibrary();
      this.tool.results.unshift(rec);
      if (this.tab !== "tools") this.detail = rec;
      this.status = `Saved ${this.kindLabel(rec.kind).toLowerCase()} version (${rec.width}×${rec.height})`;
    },
    kindLabel(kind) {
      return { generated: "Original", img2img: "Image-to-image", faces: "Face-restored", enhanced: "Enhanced",
               depth: "Depth map", inpaint: "Inpainted", outpaint: "Expanded", grid: "Comparison grid",
               edited: "Edited", healed: "Objects removed", word_edit: "Edited with words", faceswap: "Face swap",
               chain: "Model chain", tiled: "Tiled upscale", animation: "Animation", layer: "Layer" }[kind] || kind;
    },
    depthModelInfo(id) {
      return this.depthModels.find((d) => d.id === id) || this.depthModels[0];
    },
    async toolDepth() {
      const { model, style } = this.post.depth;
      return this.toolRun("Making a depth map", { action: "depth", depth_model: model, style });
    },
    depthToControlNet(r) {
      // a ready-made depth map: ControlNet follows it as-is; the picture itself isn't the starting image
      // (Image to image is put in 🔄 Change it with an SD 1.5 model, whatever it was doing before: the depth
      // ControlNet needs both, and Edit with words / Inpaint would hide it)
      this.form.init_image = r.path;
      this.form.i2i_use_init = false;
      this.form.use_init_size = true;
      this.mask.data = null;
      this.setI2iMode("change");
      this.form.control = { ...this.form.control, type: "depth", already_map: true };
      this.controlMap = null;
      this.detail = null;
      this.tab = "i2i";
      if (this.family !== "sd15") {
        const sd = this.installedModels.find((m) => m.family === "sd15" && m.engine !== "mlx");
        if (sd) this.selectModel(sd.id);
      }
      this.openI2I();
      this.status = r.settings?.depth?.style === "color"
        ? "Depth map loaded. Tip: ControlNet expects the grey depth map; make one with Look → Grey for best results"
        : "Depth map loaded as the ControlNet map: describe the new picture and press Generate";
      this.$nextTick(() => window.scrollTo({ top: 0, behavior: "smooth" }));
    },
    async openById(id) {
      const found = [...this.session, ...this.lib.results].find((x) => x.id === id);
      if (found) this.detail = found;
      else this.error = `Image #${id} isn't loaded; find it in the Library.`;
    },

    // -------------------------------------------------------------- presets
    async savePreset() {
      const name = this.newPresetName.trim();
      if (!name) return;
      const settings = JSON.parse(JSON.stringify(this.form));
      if (!this.presetIncludesPrompt) delete settings.prompt;
      delete settings.seed; // presets keep the look, not one exact image
      this.presets = await call("presets_save", { name, settings });
      this.selectedPreset = name;
      this.newPresetName = "";
      this.status = `Saved preset "${name}"`;
    },
    async applyPreset() {
      const p = this.presets.find((x) => x.name === this.selectedPreset);
      if (!p) return;
      const s = JSON.parse(JSON.stringify(p.settings));
      this.form.model = s.model || this.form.model;
      await this.loadSamplers(false);
      Object.assign(this.form, s);
      const missing = (s.loras || []).filter((l) => !this.loras.some((x) => x.name === l.name));
      this.status = missing.length
        ? `Preset applied, but these LoRAs are missing: ${missing.map((l) => l.name).join(", ")}`
        : `Applied preset "${p.name}"`;
    },
    async deletePreset() {
      this.presets = await call("presets_delete", { name: this.selectedPreset });
      this.selectedPreset = "";
    },

    // -------------------------------------------------------------- library
    async openLibrary() {
      this.tab = "library";
      this.libLoras = await call("library_loras");
      await this.queryLibrary();
    },
    async queryLibrary(more = false) {
      const PAGE = 120; // small pages keep older Macs smooth with big libraries
      if (more !== true) this.loadContentStatus();
      const rows = await call("library_query", {
        hide_nsfw: this.hideNsfw, hide_people: !!this.opts.hide_people,
        text: this.lib.text || null, seed: String(this.lib.seed ?? "").trim() || null,
        model: this.lib.model || null, kind: this.lib.kind || null,
        starred_only: this.lib.starred, loras: this.lib.loras, limit: PAGE,
        offset: more === true ? this.lib.results.length : 0,
      });
      this.lib.results = more === true ? this.lib.results.concat(rows) : rows;
      this.libHasMore = rows.length === PAGE;
    },
    findSeed(seed) {
      // Library -> every picture made with this seed
      this.detail = null;
      Object.assign(this.lib, { text: "", seed: String(seed), model: "", kind: "", starred: false, loras: [] });
      this.tab = "library";
      this.queryLibrary();
    },
    queryLibraryDebounced() {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => this.queryLibrary(), 250);
    },
    toggleLibLora(name) {
      const i = this.lib.loras.indexOf(name);
      if (i >= 0) this.lib.loras.splice(i, 1);
      else this.lib.loras.push(name);
      this.queryLibrary();
    },
    async toggleStar(r) {
      const updated = await call("library_star", { id: r.id, starred: !r.starred });
      for (const list of [this.session, this.lib.results]) {
        const item = list.find((x) => x.id === r.id);
        if (item) item.starred = updated.starred;
      }
      if (this.detail && this.detail.id === r.id) this.detail.starred = updated.starred;
      if (this.tab === "library" && this.lib.starred && !updated.starred) this.queryLibrary();
    },
    insertQualityTags() {
      // Pony / Illustrious / NoobAI: write their quality tags into the prompts, visible and editable
      // (the automatic adding then sees them there and doesn't add them twice)
      const fl = this.currentFlavor;
      if (!fl) return;
      const put = (text, tags) => {
        const t = (text || "").trim();
        return t.toLowerCase().includes(tags.split(",")[0].trim().toLowerCase()) ? t : (t ? `${tags}, ${t}` : tags);
      };
      this.form.prompt = put(this.form.prompt, QUALITY_TAGS[fl]);
      this.form.negative_prompt = put(this.form.negative_prompt, QUALITY_NEG[fl]);
      this.status = `${FLAVOR_LABEL[fl]} quality tags added to the prompt and negative prompt`;
    },
    // -------------------------------------------------------------- models: refresh, default, delete
    async refreshModels() {
      this.status = "Looking for models and LoRAs…";
      try {
        const [models] = await Promise.all([call("list_models", { fresh: true }), this.refreshLoras()]);
        this.models = models;
        call("diffusionbee_info").then((d) => { this.dbee = d; }).catch(() => {});
        call("libraries_refresh").then((r) => { this.libs = r; }).catch(() => {});  // other apps' libraries too
        this.status = `Found ${this.models.filter((m) => this.isInstalled(m)).length} models on this Mac and ${this.loras.length} LoRAs`;
      } catch (e) {
        this.error = e.message;
      }
    },
    async setDefaultModel(m) {
      this.opts.default_model = this.opts.default_model === m.id ? "" : m.id;
      await this.saveOpts();
      this.status = this.opts.default_model ? `${m.label} is now the default: Create starts with it` : "No default model: Create starts with the one you used last";
    },
    isOwnModel(m) {
      return m.id.includes(":") && !m.id.startsWith("hf:");
    },
    async deleteModel(m) {
      if (this.modelDel !== m.id) {
        this.modelDel = m.id; // first click arms it
        clearTimeout(this.modelDelTimer);
        this.modelDelTimer = setTimeout(() => { this.modelDel = ""; }, 6000);
        return;
      }
      this.modelDel = "";
      if (this.isOwnModel(m)) {
        let err;
        try {
          err = m.path ? await window.bee.trashModel(m.path) : "No file to move";
        } catch (e) {
          err = e.message || String(e);
        }
        if (err && !/isn't there any more/.test(err)) {
          this.error = `Couldn't move ${m.label} to the Trash: ${err}`;
          window.scrollTo({ top: 0, behavior: "smooth" });  // the message is at the top of the page
          return;
        }
        if (m.id.startsWith("local:")) await call("model_local_remove", { id: m.id }).catch(() => {});
        this.status = `${m.label} moved to the Trash (put it back from the Trash if you change your mind)`;
      } else {
        const r = await this.runJob("model_delete", { model: m.id });
        if (!r) return;
        this.status = `Deleted ${m.label} · ${r.freed_gb} GB freed` + (r.kept_for.length ? ` · kept files that ${r.kept_for[0]} also uses` : "") +
          " · it downloads again if you use it";
      }
      if (this.opts.default_model === m.id) { this.opts.default_model = ""; this.saveOpts(); }
      if (this.readyModel === m.id) this.readyModel = "";
      this.models = await call("list_models", { fresh: true });  // look in the folders again, so it's gone from the list now
      if (this.form.model === m.id && !this.isInstalled(this.models.find((x) => x.id === m.id) || {})) {
        const next = this.models.find((x) => this.isInstalled(x) && x.enough_ram);
        if (next) this.selectModel(next.id);
      }
    },
    openDetail(r) {
      if (this.opts.sfw_mode && r && r.nsfw) {
        this.status = "Hidden by safe-for-work mode";
        return;
      }
      this.detail = r;
      this.revealDetail = false;
    },
    // -------------------------------------------------------------- Library: batch edit & export (+ Press)
    bxBuiltins() {
      return [
        { name: "🗜 Compress (Press)", tip: "Same format and size, the smallest file that still looks the same (usually 40–70% smaller); details and location removed",
          set: { adjust: "", format: "same", compress: "balanced", resize: { mode: "none" }, keep_meta: false, dpi: 72 } },
        { name: "Web · 2048 px JPEG", tip: "Long side 2048 px, pressed JPEG: great for websites and sharing",
          set: { adjust: "", format: "jpeg", compress: "balanced", resize: { mode: "long", value: 2048, no_upscale: true }, keep_meta: false, dpi: 72 } },
        { name: "Instagram 1080", tip: "Fits 1080 × 1350 (portrait 4:5), pressed JPEG",
          set: { adjust: "", format: "jpeg", compress: "balanced", resize: { mode: "fit", w: 1080, h: 1350, no_upscale: true }, keep_meta: false, dpi: 72 } },
        { name: "Print · TIFF 300 dpi", tip: "Full size, lossless TIFF marked 300 dpi",
          set: { adjust: "", format: "tiff", compress: "off", resize: { mode: "none" }, keep_meta: false, dpi: 300 } },
      ];
    },
    bxCurrent() {
      const { adjust, format, quality, compress, dpi, keep_meta, pattern, resize } = this.bx;
      return JSON.parse(JSON.stringify({ adjust, format, quality, compress, dpi, keep_meta, pattern, resize }));
    },
    bxApply(set) {
      const r = Object.assign({ mode: "none", value: 2048, w: 1080, h: 1350, no_upscale: true }, set.resize || {});
      Object.assign(this.bx, JSON.parse(JSON.stringify(set)), { resize: r });
    },
    bxPresetOn(set) {
      const cur = this.bxCurrent();
      return ["adjust", "format", "compress", "keep_meta", "dpi"].every((k) => set[k] === undefined || cur[k] === set[k]) &&
        (set.resize?.mode ?? "none") === cur.resize.mode;
    },
    openBatch(ids, preset) {
      this.bx.ids = (ids || this.selected).slice();
      if (!this.bx.ids.length) return;
      this.bx.results = []; this.bx.errors = []; this.bx.done = 0; this.bx.total = 0;
      if (preset) this.bxApply(preset.set);
      this.bx.open = true;
      this.$nextTick(() => document.getElementById("batch-card")?.scrollIntoView({ behavior: "smooth", block: "start" }));
    },
    compressOne(r) {
      // "🗜 Compress / export" on one picture: the Library's batch panel with the Press preset
      this.bxOne = r;
      this.detail = null;
      this.tab = "library";
      this.openBatch([r.id], this.bxBuiltins()[0]);
    },
    async bxPickDest() {
      const p = await window.bee.pickFolder({ title: "Where should the exported pictures go?", defaultPath: this.bx.dest, button: "Use this folder" });
      const d = Array.isArray(p) ? p[0] : p;
      if (d) { this.bx.dest = d; saveJSON("protoart.exportDir", d); }
    },
    bxAdjustSet() {
      const a = this.bx.adjust || "";
      if (a.startsWith("look:")) { const l = (window.ProtoAdjust?.LOOKS || []).find((x) => "look:" + x.id === a); return l ? { name: l.label, set: l.set } : null; }
      if (a.startsWith("preset:")) { const pr = this.adjPresets.find((x) => "preset:" + x.name === a); return pr ? { name: pr.name, set: pr.set } : null; }
      return null;
    },
    bxRecord(id) {
      return this.lib.results.find((r) => r.id === id) || (this.bxOne && this.bxOne.id === id ? this.bxOne : null);
    },
    bxExample() {
      const r = this.bxRecord(this.bx.ids[0]);
      const name = r ? r.path.split("/").pop().replace(/\.[^.]+$/, "") : "picture";
      const ext = { png: "png", jpeg: "jpg", webp: "webp", tiff: "tif" }[this.bx.format] || (r ? r.path.split(".").pop() : "png");
      const adj = this.bxAdjustSet();
      const out = (this.bx.pattern || "{name}").replace(/\{(name|n|date|time|model|seed|width|height|preset)\}/g, (m, k) =>
        ({ name, n: "001", date: new Date().toISOString().slice(0, 10), time: "153000", model: r?.model ? String(r.model).split("/").pop().split(":").pop().replace(/\.[^.]+$/, "") : "",
           seed: r?.seed ?? "", width: r?.width ?? "", height: r?.height ?? "", preset: adj ? adj.name : "" })[k]);
      return out + "." + ext;
    },
    bxSizeText(bytes) {
      return bytes > 1e6 ? (bytes / 1e6).toFixed(1) + " MB" : Math.max(1, Math.round(bytes / 1e3)) + " KB";
    },
    bxSaved() {
      const a = this.bx.results.reduce((n, r) => n + (r.src_bytes || 0), 0), b = this.bx.results.reduce((n, r) => n + (r.bytes || 0), 0);
      return { before: a, after: b, pct: a ? Math.round((1 - b / a) * 100) : 0 };
    },
    async bxRun() {
      if (this.bx.running || !this.bx.ids.length) return;
      if (!this.bx.dest) { await this.bxPickDest(); if (!this.bx.dest) return; }
      Object.assign(this.bx, { running: true, cancel: false, done: 0, total: this.bx.ids.length, results: [], errors: [] });
      const adj = this.bxAdjustSet();
      let eng = null;
      const opts = { format: this.bx.format, quality: this.bx.quality, compress: this.bx.compress, dpi: this.bx.dpi,
                     keep_meta: this.bx.keep_meta, pattern: this.bx.pattern, resize: JSON.parse(JSON.stringify(this.bx.resize)) };
      try {
        if (adj) eng = new window.ProtoAdjust.AdjustEngine(document.createElement("canvas"));
        for (const [i, id] of this.bx.ids.entries()) {
          if (this.bx.cancel) break;
          const r = this.bxRecord(id);
          const args = { image_id: id, dest: this.bx.dest, opts, n: i + 1, preset: adj ? adj.name : "" };
          try {
            if (adj && r) {
              // the Adjust preset / Look runs here on the graphics chip; the engine does the rest
              await eng.setImage(await createImageBitmap(new Blob([await this.readImageBytes(r.path)])), 256);
              const blob = await eng.exportPNG(Object.assign(window.ProtoAdjust.defaults(), JSON.parse(JSON.stringify(adj.set))));
              const bytes = new Uint8Array(await blob.arrayBuffer());
              try {
                if (!window.bee.writeAdjusted) throw new Error("No handler registered");
                args.tmp_file = await window.bee.writeAdjusted(bytes);
              } catch (e) {
                if (!/No handler registered/.test(String(e.message || e))) throw e;
                let bin = "";
                for (let k = 0; k < bytes.length; k += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(k, k + 0x8000));
                args.data_b64 = btoa(bin);
              }
            }
            this.bx.results.push(await call("export_image", args));
          } catch (e) {
            this.bx.errors.push(`${r ? r.path.split("/").pop() : "#" + id}: ${e.message || e}`);
          }
          this.bx.done = i + 1;
        }
      } finally {
        if (eng) eng.destroy();
        this.bx.running = false;
      }
      const s = this.bxSaved();
      this.status = `Exported ${this.bx.results.length} picture${this.bx.results.length === 1 ? "" : "s"} to ${this.bx.dest}` +
        (s.before && s.pct > 0 ? ` · ${this.bxSizeText(s.before)} → ${this.bxSizeText(s.after)} (${s.pct}% smaller)` : "");
    },
    async bxSavePreset() {
      const name = (this.bx.presetName || "").trim();
      if (!name) return;
      const list = this.exportPresets.filter((x) => x.name.toLowerCase() !== name.toLowerCase());
      list.push({ name, set: this.bxCurrent() });
      try {
        this.exportPresets = await call("export_presets_set", { presets: list });
        this.bx.presetName = ""; this.bx.savingPreset = false;
      } catch (e) { this.error = e.message; }
    },
    async bxDeletePreset(pr) {
      try { this.exportPresets = await call("export_presets_set", { presets: this.exportPresets.filter((x) => x !== pr) }); }
      catch (e) { this.error = e.message; }
    },
    // -------------------------------------------------------------- Create: fold sections down (like DiffusionBee)
    cfShut(k) {
      return !!this.cfolds[k];
    },
    cfToggle(k, ev) {
      // clicks on the label's own buttons, links and fields do their own thing
      if (ev && ev.target && ev.target.closest("button, a, input, select, textarea")) return;
      this.cfolds = { ...this.cfolds, [k]: !this.cfolds[k] };
      saveJSON("protoart.createFolds", this.cfolds);
    },
    cfAll(shut) {
      const all = {};
      for (const k of ["model", "prompt", "loras", "ipa", "size", "extras"]) all[k] = shut;
      this.cfolds = all;
      saveJSON("protoart.createFolds", all);
    },
    cfSummary(k) {
      const f = this.form, cut = (t, n) => (t.length > n ? t.slice(0, n - 1) + "…" : t);
      if (k === "model") return this.currentModel ? cut(this.currentModel.label, 46) : "none";
      if (k === "prompt") return f.prompt ? cut(f.prompt.replace(/\s+/g, " "), 52) : "empty";
      if (k === "loras") return f.loras.length ? cut(f.loras.map((l) => l.name).join(", "), 46) : "none";
      if (k === "ipa") return f.ip_adapter && f.ip_adapter.image ? "on" : "none";
      if (k === "regions") return f.regions && f.regions.on ? `${f.regions.prompts.filter((x) => x.trim()).length} areas · ${f.regions.layout}` : "off";
      if (k === "size") return `${f.width}×${f.height} · ${f.fast && this.currentModel?.fast_mode ? "⚡ fast" : f.steps + " steps"} · CFG ${f.cfg} · ${f.num_images} image${f.num_images === 1 ? "" : "s"}`;
      if (k === "extras") {
        const on = [];
        if (this.clipSkipOk && Number(f.clip_skip) > 1) on.push("CLIP skip " + f.clip_skip);
        if (f.hires && f.hires.on) on.push("Hi-res ×" + f.hires.scale);
        if (f.face_detailer && f.face_detailer.on) on.push("face detailer");
        if (f.restore_faces) on.push("restore faces (" + (f.face_model === "codeformer" ? "CodeFormer" : "RestoreFormer") + ")");
        return on.length ? on.join(" · ") : "none";
      }
      return "";
    },
    // -------------------------------------------------------------- Adjust: sliders and filters on the GPU
    // Adjust lives in the Tools tab: the picture on the left becomes the live preview, the controls are a card on the right.
    async openAdjust(r, keepSettings = false) {
      if (!r || !r.path) return;
      if (!window.ProtoAdjust) { this.error = "Adjust didn't load: quit and reopen the app."; return; }
      if (this.tab !== "tools" || !this.tool.source) {
        // from the Library / Image to image: bring the picture into Tools first
        if (r.id != null && (!this.tool.source || this.tool.source.path !== r.path)) this.openInTools(r);
        else if (!this.tool.source) this.tool.source = { path: r.path, image_id: null, width: r.width || 0, height: r.height || 0, label: r.path.split("/").pop() };
        this.tab = "tools";
      }
      const p = keepSettings && this.adj.p ? this.adj.p : window.ProtoAdjust.defaults();
      Object.assign(this.adj, { open: true, rec: r, p, before: false, view: "after", busy: true, msg: "Loading the picture…", error: "",
                                size: null, which: this.tool.latest && this.tool.latest.path === r.path ? "result" : "original" });
      try {
        const bmp = await createImageBitmap(new Blob([await this.readImageBytes(r.path)]));
        await this.$nextTick();
        const canvas = this.$refs.adjCanvas;
        if (!canvas) throw new Error("the preview isn't on screen");
        if (adjEngine) adjEngine.destroy();
        adjEngine = new window.ProtoAdjust.AdjustEngine(canvas);
        await adjEngine.setImage(bmp, 1600);
        this.adj.size = [bmp.width, bmp.height];
        this.adj.msg = "";
        this.adjRender();
        this.$nextTick(() => document.getElementById("adjust-card")?.scrollIntoView({ behavior: "smooth", block: "start" }));
      } catch (e) {
        this.adj.error = "Couldn't open the picture for adjusting: " + (e.message || e);
      }
      this.adj.busy = false;
    },
    async readImageBytes(path) {
      // the window's own file reader (needs the app fully restarted after an update), else via the engine
      const web = /\.(png|jpe?g|webp|gif|bmp)$/i.test(path);  // what the window can decode itself
      try {
        if (web && window.bee.readImage) return await window.bee.readImage(path);
      } catch (e) {
        if (!/No handler registered/.test(String(e.message || e))) throw e;
      }
      const r = await call("image_b64", { path });
      return Uint8Array.from(atob(r.data), (c) => c.charCodeAt(0));
    },
    closeAdjust() {
      if (adjEngine) { adjEngine.destroy(); adjEngine = null; }
      this.adj.open = false;
    },
    adjRender() {
      if (!adjEngine || !this.adj.p) return;
      cancelAnimationFrame(adjFrame);
      adjFrame = requestAnimationFrame(() => {
        const view = this.adj.before ? "before" : this.adj.view;
        try { adjEngine.render(this.adj.p, view, this.adj.split); } catch (e) { this.adj.error = e.message; }
      });
    },
    adjDefault(k) {
      return window.ProtoAdjust.defaults()[k];
    },
    adjSlider(sl) {
      return { min: sl.min ?? -1, max: sl.max ?? 1, step: sl.step ?? 0.01 };
    },
    adjValue(k) {
      const v = this.adj.p[k];
      return k === "exposure" ? (v > 0 ? "+" : "") + v.toFixed(2) : Math.round(v * 100);
    },
    adjSectionOn(sec) {
      return sec.sliders.some((sl) => this.adj.p[sl.k] !== this.adjDefault(sl.k));
    },
    adjResetSection(sec) {
      for (const sl of sec.sliders) this.adj.p[sl.k] = this.adjDefault(sl.k);
    },
    adjResetFilter(id) {
      const f = ADJ_FILTERS[id];
      for (const sl of f.sliders) this.adj.p[sl.k] = this.adjDefault(sl.k);
      if (f.seg) this.adj.p[f.seg.k] = this.adjDefault(f.seg.k);
      if (f.toggle) this.adj.p[f.toggle.k] = this.adjDefault(f.toggle.k);
    },
    adjToggleFilter(id) {
      // the filter's push button: on (and its controls open) / off
      const f = ADJ_FILTERS[id];
      this.adj.p[f.on] = !this.adj.p[f.on];
      this.adj.openFilter = this.adj.p[f.on] ? id : (this.adj.openFilter === id ? "" : this.adj.openFilter);
    },
    adjMove(id, dir) {
      const o = this.adj.p.order.slice(), i = o.indexOf(id), j = i + dir;
      if (j < 0 || j >= o.length) return;
      [o[i], o[j]] = [o[j], o[i]];
      this.adj.p.order = o;
    },
    adjLook(look) {
      this.adj.p = Object.assign(window.ProtoAdjust.defaults(), look.set);
      this.adj.openFilter = "";
    },
    adjLookOn(look) {
      const d = Object.assign(window.ProtoAdjust.defaults(), look.set);
      return Object.keys(d).every((k) => k === "order" || this.adj.p[k] === d[k]);
    },
    adjResetAll() {
      this.adj.p = window.ProtoAdjust.defaults();
      this.adj.openFilter = "";
    },
    // compare: Edited / Original / Split (drag the line); the \\ key flips between Edited and Original
    adjSetView(v) {
      this.adj.view = v;
    },
    adjSplitDrag(e) {
      if (this.adj.view !== "split") return;
      const c = this.$refs.adjCanvas;
      if (!c) return;
      const move = (ev) => {
        const r = c.getBoundingClientRect();
        this.adj.split = Math.min(1, Math.max(0, (ev.clientX - r.left) / Math.max(1, r.width)));
      };
      move(e);
      const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    },
    adjKey(e) {
      if (!this.adj.open || this.tab !== "tools" || e.metaKey || e.ctrlKey) return;
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName || "") && e.target.type !== "range") return;
      if (e.key === "\\") { e.preventDefault(); this.adj.view = this.adj.view === "before" ? "after" : "before"; }
    },
    // -------------------------------------------------------------- Adjust presets (yours)
    adjCurrentSet() {
      const d = window.ProtoAdjust.defaults(), out = {};
      for (const k of Object.keys(d)) {
        if (k === "order") { if (this.adj.p.order.join() !== d.order.join()) out.order = this.adj.p.order.slice(); }
        else if (this.adj.p[k] !== d[k]) out[k] = this.adj.p[k];
      }
      return out;
    },
    async adjSavePreset() {
      const name = (this.adj.presetName || "").trim();
      if (!name || !this.adjChanged()) return;
      const list = this.adjPresets.filter((x) => x.name.toLowerCase() !== name.toLowerCase());
      list.push({ name, set: this.adjCurrentSet() });
      try {
        this.adjPresets = await call("adjust_presets_set", { presets: list });
        this.adj.presetName = "";
        this.adj.savingPreset = false;
        this.status = `Saved the Adjust preset “${name}”`;
      } catch (e) {
        this.adj.error = e.message;
      }
    },
    adjApplyPreset(pr) {
      this.adj.p = Object.assign(window.ProtoAdjust.defaults(), JSON.parse(JSON.stringify(pr.set || {})));
      this.adj.openFilter = "";
    },
    adjPresetOn(pr) {
      const d = Object.assign(window.ProtoAdjust.defaults(), pr.set || {});
      return Object.keys(d).every((k) => (k === "order" ? d.order.join() === this.adj.p.order.join() : this.adj.p[k] === d[k]));
    },
    async adjDeletePreset(pr) {
      try {
        this.adjPresets = await call("adjust_presets_set", { presets: this.adjPresets.filter((x) => x !== pr) });
        this.status = `Deleted the Adjust preset “${pr.name}”`;
      } catch (e) {
        this.adj.error = e.message;
      }
    },
    adjHasLast() {
      return !!loadJSON(ADJ_KEY, null);
    },
    adjUseLast() {
      const last = loadJSON(ADJ_KEY, null);
      if (last) this.adj.p = Object.assign(window.ProtoAdjust.defaults(), last);
    },
    adjChanged() {
      return this.adj.p && window.ProtoAdjust && window.ProtoAdjust.anyOn(this.adj.p);
    },
    // Save one adjusted version: settings `set` applied by `engine` (its full-size picture), linked to `srcRec`.
    async adjSaveVersion(engine, set, srcRec) {
      const blob = await engine.exportPNG(set);
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const args = { image_id: srcRec.id ?? null, path: srcRec.path, adjust: JSON.parse(JSON.stringify(set)) };
      try {
        if (!window.bee.writeAdjusted) throw new Error("No handler registered");
        args.file = await window.bee.writeAdjusted(bytes);
      } catch (e) {
        if (!/No handler registered/.test(String(e.message || e))) throw e;
        let bin = "";  // the app wasn't fully restarted after the update: send the picture through the engine
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        args.data_b64 = btoa(bin);
      }
      const rec = await call("adjust_save", args);
      if (rec) this.tool.results.unshift(rec);
      return rec;
    },
    async adjSave(thenEnhance = false) {
      if (!adjEngine || !this.adjChanged()) return;
      const r = this.adj.rec, p = this.adj.p;
      this.adj.busy = true;
      this.adj.msg = "Applying to the full-size picture…";
      this.adj.error = "";
      try {
        const rec = await this.adjSaveVersion(adjEngine, p, r);
        saveJSON(ADJ_KEY, p);
        this.closeAdjust();
        if (rec && rec.id != null) {
          this.tool.latest = rec;
          this.status = "Adjusted copy saved to the Library (the original is unchanged)";
          if (thenEnhance === "i2i") this.useAsStart(rec);  // Save & send to Image to image
          else if (thenEnhance) this.adjEnhance(rec);
        }
      } catch (e) {
        this.adj.error = "Couldn't save: " + (e.message || e);
      }
      this.adj.busy = false;
      this.adj.msg = "";
    },
    adjEnhanceArgs(rec) {
      const { seed, ...options } = this.post.enh;
      return { ...(rec.id != null ? { image_id: rec.id } : { path: rec.path }), action: "enhance", seed, options };
    },
    // queue Enhance & upscale (the settings in the Enhance card) on one picture; resolves when it's done
    async adjEnhance(rec) {
      return this.toolRun("Enhancing the adjusted picture", this.adjEnhanceArgs(rec));
    },
    // -------------------------------------------------------------- several versions at once (presets in a queue)
    adjBatchItems() {
      const items = [];
      for (const id of this.adj.batch) {
        if (id === "current") { if (this.adjChanged()) items.push({ name: "current settings", set: JSON.parse(JSON.stringify(this.adj.p)) }); }
        else {
          const pr = this.adjPresets.find((x) => x.name === id);
          if (pr) items.push({ name: pr.name, set: Object.assign(window.ProtoAdjust.defaults(), JSON.parse(JSON.stringify(pr.set))) });
        }
      }
      return items;
    },
    adjBatchToggle(id) {
      const i = this.adj.batch.indexOf(id);
      if (i >= 0) this.adj.batch.splice(i, 1);
      else this.adj.batch.push(id);
    },
    async adjBatch() {
      const items = this.adjBatchItems();
      if (!items.length || !adjEngine) return;
      const src = this.adj.rec;
      this.adj.busy = true;
      this.adj.error = "";
      const made = [];
      try {
        if (this.adj.batchEnhance && this.adj.batchOrder === "enhance_first") {
          // AI first (once), then every version's filters on top of the enhanced picture
          this.adj.msg = "Enhancing first (in the queue)…";
          const enhanced = await this.adjEnhance(src);
          if (!enhanced) throw new Error("the enhance step didn't finish");
          const eng = new window.ProtoAdjust.AdjustEngine(document.createElement("canvas"));
          try {
            await eng.setImage(await createImageBitmap(new Blob([await this.readImageBytes(enhanced.path)])), 256);
            for (const [i, it] of items.entries()) {
              this.adj.msg = `Version ${i + 1} of ${items.length} (${it.name})…`;
              made.push(await this.adjSaveVersion(eng, it.set, enhanced));
            }
          } finally {
            eng.destroy();
          }
        } else {
          for (const [i, it] of items.entries()) {
            this.adj.msg = `Version ${i + 1} of ${items.length} (${it.name})…`;
            made.push(await this.adjSaveVersion(adjEngine, it.set, src));
          }
          if (this.adj.batchEnhance) {
            // filters first: each version goes into the queue to be enhanced (the AI enhances the filtered picture)
            this.adj.msg = `${made.length} versions saved · enhancing each one in the queue…`;
            this.status = `Queued Enhance & upscale for ${made.length} adjusted versions (see the queue)`;
            const outs = await Promise.all(made.filter(Boolean).map((rec) => this.runJob("postprocess", this.adjEnhanceArgs(rec))));
            for (const o of outs) if (o) { this.tool.results.unshift(o); this.tool.latest = o; }
          }
        }
        const last = made.filter(Boolean).pop();
        if (last && !(this.adj.batchEnhance && this.adj.batchOrder !== "enhance_first")) this.tool.latest = last;
        this.status = `Made ${made.length} adjusted version${made.length === 1 ? "" : "s"}` +
          (this.adj.batchEnhance ? (this.adj.batchOrder === "enhance_first" ? " of the enhanced picture" : ", each enhanced") : "") +
          " · all in the Library";
      } catch (e) {
        this.adj.error = "Stopped: " + (e.message || e) + (made.length ? ` (${made.length} version${made.length === 1 ? "" : "s"} saved before that)` : "");
      }
      this.adj.busy = false;
      this.adj.msg = "";
    },
    // -------------------------------------------------------------- speed: a model ready in memory
    preloadModel(id) {
      const m = this.models.find((x) => x.id === id);
      // only models already on this Mac, only while nothing else is running, and never twice
      if (!m || !this.isInstalled(m) || this.backendDead || this.queue.length) return;
      if (this.readyModel === id || this.preloading === id || this.preloadAsked === id) return;
      const loras = id === this.form.model ? this.form.loras.map((l) => ({ name: l.name, weight: l.weight })) : [];
      this.preloadAsked = id;
      call("preload", { model: id, loras }).catch(() => {}) // errors show when you press Generate instead
        .finally(() => { if (this.preloadAsked === id) this.preloadAsked = ""; });
    },
    // -------------------------------------------------------------- content filter (NSFW)
    safeModel(m) {
      return !(this.hideNsfw && m.nsfw) && !(this.opts.hide_people && m.people);
    },
    async setHidePeople(on) {
      // hides the models and LoRAs you tagged ● People (made of real people)
      this.opts.hide_people = !!on;
      await this.saveOpts();
      if (this.tab === "library") this.queryLibrary();
      if (!on) { this.status = "People models, LoRAs and pictures shown again"; return; }
      const dropped = this.form.loras.filter((l) => this.loras.find((x) => x.name === l.name)?.people);
      if (dropped.length) this.form.loras = this.form.loras.filter((l) => !dropped.includes(l));
      let msg = dropped.length ? `removed ${dropped.length} People LoRA${dropped.length === 1 ? "" : "s"} from the mix` : "";
      if (this.currentModel?.people) {
        const next = this.models.find((m) => this.isInstalled(m) && m.enough_ram && this.safeModel(m));
        if (next) { this.selectModel(next.id); msg = (msg ? msg + "; " : "") + "switched to " + next.label; }
      }
      const n = this.models.filter((m) => m.people).length + this.loras.filter((l) => l.people).length;
      this.status = (n ? `${n} People model${n === 1 ? "" : "s"}/LoRA${n === 1 ? "" : "s"} hidden` :
        "Nothing is tagged People yet: click the ● after a name in Create's lists to tag one") + (msg ? " · " + msg : "");
    },
    async peopleScan() {  // Library → Hide People: look for faces in the older pictures (once)
      const r = await this.runJob("people_scan", {});
      if (r) {
        this.status = `Checked ${r.checked} picture${r.checked === 1 ? "" : "s"}: ${r.with_people} with people`;
        await this.queryLibrary();
      }
    },
    async markPeople(kind, item) {
      // ● before a model or LoRA name: tag / untag it as made of real people (Hide People)
      const name = kind === "model" ? item.id : item.name;
      await call("content_mark", { kind, name, tag: "people", value: !item.people });
      if (kind === "model") this.models = await call("list_models");
      else await this.refreshLoras();
      this.status = `${kind === "model" ? item.label : item.name}: ${item.people ? "no longer tagged People" : "tagged People"}`;
    },
    covered(r) {
      return this.hideNsfw && !!r && !!r.nsfw;
    },
    async loadContentStatus() {
      try {
        Object.assign(this.content, await call("content_status"));
      } catch { /* older backend */ }
    },
    async setSfwMode(on) {
      const ok = on
        ? await this.confirmAsk("Turn on safe-for-work mode?",
            "Prompts that ask for adult content are refused, NSFW pictures, LoRAs, models and embeddings are hidden everywhere " +
            "(and can't be opened), and every new picture is checked on this Mac. Good for work, school and sharing a screen.", "Turn on")
        : await this.confirmAsk("Turn off safe-for-work mode?",
            "Adult prompts, NSFW LoRAs and models will be allowed again. Hide NSFW stays as you set it.", "Turn off");
      if (!ok) return;
      this.opts.sfw_mode = !!on;
      if (on) await this.setHideNsfw(true);  // drop NSFW LoRAs from the mix, switch away from an NSFW model
      await this.saveOpts();
      if (on && !this.opts.check_pictures) await this.setCheckPictures(true);
      this.status = on ? "Safe-for-work mode is on" : "Safe-for-work mode is off";
    },
    async setHideNsfw(on) {
      if (!on && this.opts.sfw_mode) {
        this.status = "Safe-for-work mode keeps NSFW hidden (switch it off in Settings → Content filter)";
        return;
      }
      this.opts.hide_nsfw = !!on;
      await this.saveOpts();
      if (on) {
        // leave nothing flagged in the current setup
        const dropped = this.form.loras.filter((l) => this.loras.find((x) => x.name === l.name)?.nsfw);
        if (dropped.length) this.form.loras = this.form.loras.filter((l) => !dropped.includes(l));
        let msg = dropped.length ? `Removed ${dropped.length} NSFW LoRA${dropped.length === 1 ? "" : "s"} from the mix` : "";
        if (this.currentModel?.nsfw) {
          const safe = this.models.find((m) => !m.nsfw && this.isInstalled(m) && m.enough_ram);
          if (safe) {
            this.selectModel(safe.id);
            msg = (msg ? msg + "; " : "") + "switched to " + safe.label;
          }
        }
        this.status = "NSFW filter on" + (msg ? " · " + msg : "");
      } else this.status = "NSFW filter off";
      if (this.tab === "library") this.queryLibrary();
    },
    async markItem(kind, item) {
      // ⚑ on a model or LoRA: flip what the filter thinks (your mark always wins)
      const name = kind === "model" ? item.id : item.name;
      const auto = !!item.nsfw_auto;
      const want = !item.nsfw;
      await call("content_mark", { kind, name, value: want === auto ? null : want }); // matching the guess = automatic
      if (kind === "model") this.models = await call("list_models");
      else await this.refreshLoras();
      this.status = `${kind === "model" ? item.label : item.name}: ${want ? "marked NSFW" : "marked safe"}`;
    },
    async unmarkItem(it) {
      await call("content_mark", { kind: it.kind, name: it.name, value: null });
      if (it.kind === "model") this.models = await call("list_models");
      else await this.refreshLoras();
    },
    async markPictures(ids, mark) {
      // mark: true (NSFW), false (fine) or null (automatic)
      const rows = await call("library_nsfw_mark", { ids, mark });
      const byId = new Map(rows.filter(Boolean).map((r) => [r.id, r]));
      for (const list of [this.session, this.lib.results]) {
        for (const item of list) {
          const u = byId.get(item.id);
          if (u) Object.assign(item, { nsfw: u.nsfw, nsfw_mark: u.nsfw_mark, nsfw_score: u.nsfw_score, nsfw_prompt: u.nsfw_prompt });
        }
      }
      if (this.detail && byId.has(this.detail.id)) Object.assign(this.detail, byId.get(this.detail.id));
      this.loadContentStatus();
      if (this.tab === "library" && this.hideNsfw && mark) this.queryLibrary();
    },
    nsfwReason(r) {
      if (r.nsfw_mark === true) return "you marked it NSFW";
      if (r.nsfw_mark === false) return "you marked it safe";
      const why = [];
      if (r.nsfw_prompt) why.push("the prompt asks for it");
      if (r.nsfw_score != null && r.nsfw_score >= 0.5) why.push(`the checker is ${Math.round(r.nsfw_score * 100)}% sure`);
      if (why.length) return "NSFW: " + why.join(" and ");
      return r.nsfw_score != null ? `looks safe (checker ${Math.round(r.nsfw_score * 100)}%)` : "not checked by the picture checker yet";
    },
    async setCheckPictures(on) {
      this.opts.check_pictures = !!on;
      await this.saveOpts();
      if (on && (!this.content.checker_installed || this.content.unchecked)) await this.scanPictures();
    },
    async scanPictures() {
      this.content.scanning = true;
      this.content.msg = this.content.checker_installed ? "Checking pictures…" : `Downloading the checker (~${this.content.checker_mb} MB), then checking…`;
      const r = await this.runJob("nsfw_scan", {});
      this.content.scanning = false;
      if (r) {
        this.content.msg = `Checked ${r.checked} picture${r.checked === 1 ? "" : "s"}; ${r.flagged} look NSFW` +
          (r.missing ? ` · ${r.missing} file${r.missing === 1 ? "" : "s"} missing` : "") + (r.left ? ` · stopped with ${r.left} left` : "");
        if (this.tab === "library") this.queryLibrary();
      } else this.content.msg = "";
      await this.loadContentStatus();
    },
    async reuse(r) {
      const s = r.settings || {};
      this.form.model = s.model || r.model;
      await this.loadSamplers(false);
      Object.assign(this.form, {
        prompt: r.prompt, negative_prompt: r.negative_prompt, width: r.width, height: r.height,
        steps: r.steps, cfg: r.cfg, sampler: r.sampler, seed: r.seed, enhance_prompt: false,
        clip_skip: s.clip_skip || 1,
        var_seed: s.var_strength ? s.var_seed : -1, var_strength: s.var_strength || 0,
        loras: JSON.parse(JSON.stringify(r.loras)), num_images: 1,
        init_image: s.init_image || "", strength: s.strength || 0.6,
      });
      this.detail = null;
      this.tab = "create";
    },

    // -------------------------------------------------------------- deleting
    toggleSelect(r) {
      const i = this.selected.indexOf(r.id);
      if (i >= 0) this.selected.splice(i, 1);
      else this.selected.push(r.id);
    },
    selectAll() {
      this.selected = this.libShown.map((r) => r.id);
    },
    exitSelect() {
      this.selectMode = false;
      this.selected = [];
    },
    openCtx(e, r, inDetail = false) {  // right-click a picture: a small menu (the selected pictures, if it's one of them)
      const ids = !inDetail && this.selectMode && this.selected.includes(r.id) && this.selected.length > 1 ? [...this.selected] : [r.id];
      this.ctx = { x: Math.min(e.clientX, innerWidth - 240), y: Math.min(e.clientY, innerHeight - 340), r, ids, inDetail };
    },
    exportSettings(r) {  // a picture's settings as .json (📥 Settings loads it back)
      const st = { ...(r.settings || {}), prompt: r.prompt, negative_prompt: r.negative_prompt, width: r.width, height: r.height,
                   steps: r.steps, cfg: r.cfg, sampler: r.sampler, seed: r.seed, model: r.model, loras: r.loras || [] };
      delete st.init_image;  // file paths on this Mac stay private
      const url = URL.createObjectURL(new Blob([JSON.stringify({ app: "Protoart Studio", kind: "settings", version: 1, settings: st }, null, 2)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `Protoart settings ${r.seed ?? r.id}.json`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      this.status = "Settings saved as .json: load them again with 📥 Settings in Create";
    },
    async savePictureAs(r) {
      try {
        const res = await window.bee.saveImageAs(r.path);
        if (res && res.ok) this.status = "Saved a copy: " + res.path.replace(/^\/Users\/[^/]+/, "~");
        else if (res && !res.canceled) this.error = res.msg || "Couldn't save the picture.";
      } catch (e) { this.error = e.message; }
    },
    async copyPicture(r) {
      try {
        this.status = (await window.bee.copyImage(r.path)) ? "Picture copied: paste it into another app (⌘V)" : "Couldn't copy that picture.";
      } catch (e) { this.error = e.message; }
    },
    galDown(e) {  // Select mode: drag across pictures (or draw a box) to select several; start on a picked one to unpick
      if (!this.selectMode || e.button !== 0 || e.target.closest("button")) return;
      const gal = e.currentTarget;
      const scroller = (() => { for (let el = gal; el; el = el.parentElement) { const o = getComputedStyle(el).overflowY; if ((o === "auto" || o === "scroll") && el.scrollHeight > el.clientHeight) return el; } return document.scrollingElement; })();
      const fig = e.target.closest("figure[data-id]");
      const startId = fig ? Number(fig.dataset.id) : null;
      const top0 = scroller.scrollTop;
      // where every picture is, measured once (in the list's own coordinates, so scrolling doesn't change them)
      const tiles = [...gal.querySelectorAll("figure[data-id]")].map((f) => {
        const r = f.getBoundingClientRect();
        return { id: Number(f.dataset.id), l: r.left, r: r.right, t: r.top + top0, b: r.bottom + top0 };
      });
      const st = { x: e.clientX, y: e.clientY + top0, cx: e.clientX, cy: e.clientY, moved: false, raf: 0, last: "",
                   remove: startId != null && this.selected.includes(startId), base: new Set(this.selected) };
      const box = document.createElement("div");
      box.className = "sel-box";
      const sr = () => (scroller === document.scrollingElement ? { top: 0, bottom: innerHeight } : scroller.getBoundingClientRect());
      const frame = () => {  // one update per screen refresh, and keep scrolling while the pointer rests at an edge
        st.raf = 0;
        const r = sr(), lo = Math.min(r.bottom, innerHeight), hi = Math.max(r.top, 0);
        let scrolled = false;
        if (st.cy > lo - 36) { const before = scroller.scrollTop; scroller.scrollTop += Math.min(24, 6 + (st.cy - (lo - 36))); scrolled = scroller.scrollTop !== before; }
        else if (st.cy < hi + 36 && st.cy < st.y - scroller.scrollTop) { const before = scroller.scrollTop; scroller.scrollTop -= Math.min(24, 6 + (hi + 36 - st.cy)); scrolled = scroller.scrollTop !== before; }
        const top = scroller.scrollTop;
        const x1 = Math.min(st.x, st.cx), x2 = Math.max(st.x, st.cx);
        const y1 = Math.min(st.y, st.cy + top), y2 = Math.max(st.y, st.cy + top);
        Object.assign(box.style, { left: x1 + "px", top: (y1 - top) + "px", width: (x2 - x1) + "px", height: (y2 - y1) + "px" });
        const hit = tiles.filter((t) => t.r > x1 && t.l < x2 && t.b > y1 && t.t < y2).map((t) => t.id);
        const key = hit.join(",");
        if (key !== st.last) {  // only touch the selection when it actually changes
          st.last = key;
          const set = new Set(st.base);
          for (const id of hit) st.remove ? set.delete(id) : set.add(id);
          this.selected = [...set];
        }
        if (scrolled) st.raf = requestAnimationFrame(frame);
      };
      const move = (ev) => {
        st.cx = ev.clientX; st.cy = ev.clientY;
        if (!st.moved) {
          if (Math.hypot(ev.clientX - st.x, ev.clientY + scroller.scrollTop - st.y) < 4) return;
          st.moved = true;
          document.body.appendChild(box);
          try { gal.setPointerCapture(e.pointerId); } catch { /* not capturable */ }  // keep the drag even off the list
        }
        if (!st.raf) st.raf = requestAnimationFrame(frame);
      };
      const up = () => {
        removeEventListener("pointermove", move);
        removeEventListener("pointerup", up);
        removeEventListener("pointercancel", up);
        removeEventListener("blur", up);
        if (st.raf) cancelAnimationFrame(st.raf);
        try { gal.releasePointerCapture(e.pointerId); } catch { /* already released */ }
        box.remove();
        if (st.moved) this._galDragged = Date.now();  // the click that ends a drag doesn't toggle a picture
      };
      e.preventDefault();  // no text or picture dragging while selecting
      addEventListener("pointermove", move);
      addEventListener("pointerup", up);
      addEventListener("pointercancel", up);
      addEventListener("blur", up);
    },
    onThumbClick(r) {
      if (this.selectMode && Date.now() - (this._galDragged || 0) < 120) return;
      if (this.selectMode) this.toggleSelect(r);
      else this.openDetail(r);
    },
    async askDelete(ids) {
      if (!ids || !ids.length) return;
      try {
        const dry = await call("library_delete", { ids: [...ids], dry_run: true });
        if (!dry.ids.length) {
          this.error = "Those images are no longer in the library (refreshing).";
          return this.queryLibrary();
        }
        this.confirmDel = { ids: [...ids], count: dry.ids.length, versions: dry.versions, include_versions: false };
      } catch (e) {
        this.error = "Couldn't delete: " + e.message;
      }
    },
    async doDelete() {
      const c = this.confirmDel;
      this.confirmDel = null;
      try {
        const res = await call("library_delete", {
          ids: c.ids, include_versions: c.include_versions, delete_files: !this.opts.trash,
        });
        let errors = res.errors || [];
        if (this.opts.trash) errors = errors.concat(await window.bee.trash(res.paths));
        const gone = new Set(res.ids);
        this.session = this.session.filter((r) => !gone.has(r.id));
        this.lib.results = this.lib.results.filter((r) => !gone.has(r.id));
        this.selected = this.selected.filter((id) => !gone.has(id));
        if (this.detail && gone.has(this.detail.id)) this.detail = null;
        if (this.tab === "library") this.libLoras = await call("library_loras");
        const n = res.ids.length;
        this.status = `${this.opts.trash ? "Moved" : "Deleted"} ${n} image${n === 1 ? "" : "s"}${this.opts.trash ? " to the Trash" : ""}`;
        if (errors.length) this.error = "Some files couldn't be removed:\n" + errors.join("\n");
        if (!this.selected.length) this.selectMode = false;
      } catch (e) {
        this.error = e.message;
      }
    },

    // -------------------------------------------------------------- image folder
    async changeImagesDir() {
      const picked = await window.bee.pickFolder(this.imagesDir.path);
      if (!picked || picked === this.imagesDir.path) return;
      try {
        const res = await call("images_dir_set", { path: picked, move_existing: this.moveExisting });
        this.imagesDir = await call("images_dir_get");
        this.status = `Images will be saved to ${res.path}` + (res.moved ? ` (moved ${res.moved} existing)` : "");
        if (res.failed?.length) this.error = "Some images couldn't be moved:\n" + res.failed.join("\n");
        this.session = [];
        if (this.tab === "library") this.queryLibrary();
      } catch (e) {
        this.error = e.message;
      }
    },
    async resetImagesDir() {
      if (this.imagesDir.path === this.imagesDir.default) return;
      try {
        await call("images_dir_set", { path: this.imagesDir.default, move_existing: this.moveExisting });
        this.imagesDir = await call("images_dir_get");
      } catch (e) {
        this.error = e.message;
      }
    },

    // -------------------------------------------------------------- misc
    thumbUrl(r) {
      return this.fileUrl(r.thumb || r.path);
    },
    fileUrl(p) {
      return "file://" + encodeURI(p).replace(/#/g, "%23").replace(/\?/g, "%3F");
    },
    showInFolder(p) {
      window.bee.showInFolder(p);
    },
    openFolder(p) {
      if (p) window.bee.openFolder(p);
    },
    openLink(key) {  // opens in your web browser (needs the internet)
      const url = LINKS[key];
      if (url) window.bee.openExternal?.(url);
      this.status = "Opened in your web browser";
    },
    setSounds(on) {
      window.ProtoSound?.setEnabled(on);
      this.soundsOn = !!on;
      if (on) window.ProtoSound?.play("done");
      this.status = on ? "🔔 A chime plays when pictures or training are finished" : "Sounds switched off";
    },
    setTooltips(on) {
      this.opts.tooltips = !!on;
      window.ProtoTips?.setEnabled(this.opts.tooltips);
      this.saveOpts();
      this.status = on ? "Tips shown when you point at buttons" : "Tips switched off (the TIPS switch at the top turns them back on)";
    },
    async saveOpts() {
      saveJSON(OPTS_KEY, this.opts);
      await call("set_settings", this.opts);
    },
    async restartBackend() {
      this.status = "Restarting backend…";
      await window.bee.restart();
    },
  },
});
if (window.RackParts) protoartApp.use(window.RackParts); // knobs, rotary switch and VU meter (rack.js)
protoartApp.mount("#app");
