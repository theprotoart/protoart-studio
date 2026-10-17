/* Negative prompt helper: writes a negative prompt that fits the prompt and the model type.
 *  * SD 1.5 / SDXL / Pony / Illustrious / NoobAI: tags (each flavour's own low-quality tags first)
 *  * ERNIE / FLUX / FLUX.2: plain words
 * It also adds the opposite of the style the prompt asks for (a cartoon prompt avoids "realistic, photo";
 * a photo prompt avoids "anime, cartoon, illustration"), and things that often go wrong for the subject
 * (faces, hands, landscapes with stray people). "random" adds a few extra common fixes each time.
 */
(function () {
  const BASE = {
    sd15: ["lowres", "bad anatomy", "bad hands", "missing fingers", "extra digit", "fewer digits", "cropped",
      "worst quality", "low quality", "jpeg artifacts", "signature", "watermark", "username", "blurry"],
    sdxl: ["lowres", "blurry", "bad anatomy", "bad hands", "deformed", "extra fingers", "watermark", "text",
      "signature", "jpeg artifacts", "low quality"],
    pony: ["score_4", "score_5", "score_6", "lowres", "bad anatomy", "bad hands", "text", "watermark", "signature"],
    illustrious: ["lowres", "worst quality", "bad quality", "bad anatomy", "bad hands", "jpeg artifacts", "signature",
      "watermark", "text", "old", "early"],
    natural: ["blurry", "low quality", "distorted face", "deformed hands", "extra fingers", "watermark", "text",
      "oversaturated colours"],
  };
  // style the prompt asks for -> what to avoid (tags, plain words)
  const STYLES = [
    { re: /\b(anime|manga|cartoon|illustration|chibi|cel[- ]?shad|comic|toon|disney|pixar style|2d)\b/i,
      tags: ["realistic", "photorealistic", "photo", "3d", "real life"],
      words: ["photorealistic rendering", "real photograph", "3D render"] },
    { re: /\b(photo|photograph|photorealistic|realistic|35mm|dslr|film still|cinematic|raw photo|portrait photo)\b/i,
      tags: ["anime", "cartoon", "illustration", "drawing", "painting", "3d render", "cgi", "doll", "plastic skin"],
      words: ["cartoon", "anime", "illustration", "painting", "3D render", "plastic-looking skin"] },
    { re: /\b(3d render|3d|octane|unreal engine|blender|cgi|claymation|pixar)\b/i,
      tags: ["2d", "flat color", "sketch", "photo"], words: ["flat 2D drawing", "sketch", "photograph"] },
    { re: /\b(watercolou?r|oil painting|painting|gouache|impressionis|acrylic)\b/i,
      tags: ["photo", "3d render", "cgi", "smooth"], words: ["photograph", "3D render", "glossy CGI look"] },
    { re: /\b(pixel art|8-?bit|16-?bit)\b/i,
      tags: ["blurry", "smooth gradient", "photo", "anti-aliasing"], words: ["blur", "smooth gradients", "photograph"] },
    { re: /\b(line ?art|sketch|ink|pencil|monochrome|black and white|greyscale|grayscale)\b/i,
      tags: ["color", "colorful", "photo"], words: ["colour", "photograph"] },
  ];
  // subject -> what often goes wrong
  const SUBJECT = [
    { re: /\b(portrait|face|close-?up|headshot|eyes|smil)/i,
      tags: ["deformed face", "asymmetrical eyes", "cross-eyed", "bad teeth"], words: ["deformed face", "mismatched eyes"] },
    { re: /\b(hand|hands|holding|fingers|gloves|waving)\b/i,
      tags: ["bad hands", "fused fingers", "too many fingers"], words: ["malformed hands", "fused fingers"] },
    { re: /\b(full[- ]body|standing|walking|running|dancing|pose|sitting)\b/i,
      tags: ["extra limbs", "missing limbs", "bad proportions", "cropped feet"], words: ["extra limbs", "odd proportions"] },
    { re: /\b(landscape|scenery|no humans|cityscape|mountain|forest|ocean|architecture|interior|room)\b/i,
      tags: ["people", "person", "text", "frame"], words: ["people", "text", "borders"] },
    { re: /\b(text|sign|logo|lettering|title|poster)\b/i,
      tags: ["misspelled", "garbled text"], words: ["misspelled words", "garbled letters"] },
  ];
  const EXTRA = {
    tags: ["extra limbs", "duplicate", "mutated", "poorly drawn face", "out of frame", "grainy", "overexposed",
      "underexposed", "oversaturated", "noise", "cloned face", "long neck", "disfigured", "gross proportions",
      "bad composition", "error", "ugly", "tiling", "morbid", "blurry background"],
    words: ["grainy noise", "overexposed highlights", "muddy colours", "duplicate figures", "cut-off framing",
      "smeared details", "warped perspective", "plastic skin", "bad composition", "harsh artifacts"],
  };

  const NATURAL = ["ernie", "flux", "flux2"];
  const pickN = (arr, n) => {
    const c = arr.slice(), out = [];
    while (out.length < n && c.length) out.push(c.splice(Math.floor(Math.random() * c.length), 1)[0]);
    return out;
  };

  /** opts: { family: "sd15"|"sdxl"|"ernie"|"flux"|"flux2", flavor: "pony"|"illustrious"|"noobai"|null, random: bool } */
  function suggest(prompt, opts = {}) {
    const natural = NATURAL.includes(opts.family);
    const fl = String(opts.flavor || "").toLowerCase();
    let base;
    if (natural) base = BASE.natural;
    else if (fl === "pony") base = BASE.pony;
    else if (fl === "illustrious" || fl === "noobai") base = BASE.illustrious;
    else if (opts.family === "sdxl") base = BASE.sdxl;
    else base = BASE.sd15;
    const p = String(prompt || "");
    const out = [...base];
    const add = (list) => list.forEach((x) => out.push(x));
    // the opposite of the style asked for (only the first style that matches, so they don't contradict)
    const st = STYLES.find((s) => s.re.test(p));
    if (st) add(natural ? st.words : st.tags);
    for (const s of SUBJECT) if (s.re.test(p)) add(natural ? s.words : s.tags);
    if (opts.random) add(pickN(natural ? EXTRA.words : EXTRA.tags, 3 + Math.floor(Math.random() * 3)));
    // never avoid something the prompt asks for
    const asked = p.toLowerCase();
    const seen = new Set();
    return out.filter((t) => {
      const k = t.toLowerCase();
      if (seen.has(k) || (k.length > 3 && asked.includes(k))) return false;
      seen.add(k);
      return true;
    }).join(", ");
  }

  window.NegativePrompt = { suggest, random: (prompt, opts = {}) => suggest(prompt, { ...opts, random: true }) };
})();
