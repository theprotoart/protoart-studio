// Tooltips for the whole app: a quick, readable tip when you point at a button or control, instead of
// macOS's slow built-in one. Uses each element's title (moved to data-tip so the built-in one doesn't also
// appear), or, for buttons without one, a plain explanation from the table below.
// Settings → Look → "Show tips when pointing at buttons" switches them all off (ProtoTips.setEnabled).
(function () {
  "use strict";
  const DELAY_MS = 350;
  // Buttons that carry no title of their own: what they do, by their (lower-case) label.
  const BY_TEXT = {
    "create": "Make pictures from a text prompt",
    "image to image": "Change a picture you already have, inpaint, expand it, or follow it with ControlNet",
    "tools": "Enhance & upscale, restore faces, Adjust, image → prompt and conversions",
    "train": "Teach the app a character, object or style (LoRA or embedding), or merge LoRAs",
    "library": "Every picture you've made, with its prompt and settings",
    "models": "Download models, add your own files, Civitai and Hugging Face",
    "settings": "Folders, privacy, looks and everything else you can change",
    "how to": "The manual: how everything works, troubleshooting and what's new",
    "save": "Save this",
    "cancel": "Close without changing anything",
    "close": "Close this",
    "close list": "Close the list",
    "refresh": "Look for new or removed files again",
    "clear": "Empty this",
    "clear all": "Remove them all",
    "remove": "Remove this from the list (the file itself isn't deleted)",
    "undo": "Put back what was just changed",
    "↩ undo": "Put back what was just changed",
    "reset": "Back to the starting values",
    "reset all": "Every slider back to its starting value",
    "use recommended": "Use the value recommended for this model",
    "expand all": "Open every section",
    "collapse all": "Fold every section",
    "fold all": "Fold every section",
    "open all": "Open every section",
    "select": "Pick several pictures (to export, edit or delete them together)",
    "select all": "Pick every picture shown",
    "done": "Finish picking",
    "delete": "Delete this picture (asks first)",
    "delete selected": "Delete the picked pictures (asks first)",
    "show in finder": "Show the file in a Finder window",
    "open": "Open the folder in Finder",
    "open the folder": "Open the folder in Finder",
    "open lora folder": "Open the folder Protoart Studio keeps LoRAs in",
    "change…": "Pick another folder",
    "change folder…": "Pick another folder",
    "use the default": "Go back to the standard folder",
    "use default": "Go back to the standard folder",
    "load more": "Show more pictures",
    "rescan": "Look through the folder again",
    "restart engine": "Restart the image engine: frees all its memory (a running job stops)",
    "■ stop": "Stop this job right away",
    "stop all": "Stop the running job and remove everything waiting",
    "force stop now": "Stop the engine at once and restart it",
    "⏭ skip image": "Skip the picture being made now and go on to the next one",
    "enhance": "Make the picture sharper and bigger",
    "restore faces": "Repair blurry or distorted faces",
    "make depth map": "Make a black-and-white depth picture (near = light, far = dark)",
    "🖌 paint": "Paint over the part to change",
    "🧽 erase": "Rub out paint",
    "invert": "Swap painted and unpainted areas",
    "+ all sides": "Make the picture bigger on every side",
    "make it wider": "Add space on the left and right",
    "make it taller": "Add space at the top and bottom",
    "preview the map": "See the ControlNet map the picture will follow",
    "set in create": "This is chosen in the Create tab",
    "use the result as the new picture": "Keep working on the result instead of the original",
    "🎚 adjust": "Light, colour and filter sliders, Lightroom-style",
    "🎚 adjust the result": "Light, colour and filter sliders for the result",
    "✕ close adjust": "Close Adjust (your edits are kept until you save or leave)",
    "start training": "Start training (Pause and Stop keep what's learned so far)",
    "caption the missing ones": "Write captions for pictures that don't have one yet",
    "redo all": "Write new captions for every picture",
    "save captions": "Save the captions as .txt files next to the pictures",
    "use it in create": "Add the result to Create's LoRA list",
    "🧬 merge": "Merge the LoRAs above into one new LoRA",
    "⤓ download": "Download it into the folder shown",
    "check": "Look up this link and show what it is",
    "add model": "Add this model to your list",
    "save key only": "Keep the key on this Mac without downloading",
    "save & download": "Keep the key on this Mac and start the download",
    "save & check again": "Keep the token on this Mac and check again",
    "add model files or folders…": "Use model files you already have (they stay where they are)",
    "↻ refresh model and lora lists": "Look for new or removed model and LoRA files",
    "watch a models folder…": "Every model in this folder appears in the app, including ones added later",
    "add lora folders…": "Use LoRA folders you already have (an external drive is fine)",
    "sort them": "Put the files into folders by model type",
    "not now": "Ask again later",
    "check them now": "Check the Library's pictures with the content filter",
    "reset settings": "Every setting back to the standard (your pictures, presets and models stay)",
    "reset folder locations": "Folders back to the standard places (files aren't moved)",
    "show the log file": "Open the engine's log (useful when something went wrong)",
    "load a .txt file…": "Load one prompt per line from a text file",
    "🖼 send to image to image": "Open this picture in Image to image",
    "🧰 send to tools": "Open this picture in Tools (enhance, faces, Adjust)",
    "✨ enhance & upscale…": "Make this picture sharper and bigger in Tools",
    "★ starred": "Starred · click to unstar",
    "☆ star": "Star this picture (★ Starred only shows them)",
    "mark nsfw": "Mark this picture NSFW (covered while Hide NSFW is on)",
    "mark safe": "Mark this picture safe",
    "automatic": "Let the content filter decide",
    "describe (florence-2)": "Write a sentence describing the picture",
    "danbooru tags": "List anime-style tags for the picture",
    "turn off compare": "Back to making single pictures",
    "untag": "Remove the People tag",
    "start from this picture (image-to-image)": "On: the result starts from your picture (Change strength says how much may change). Off: only ControlNet follows it",
    "use the picture's size (otherwise the size from create)": "On: the result has your picture's size and shape. Off: the size set in Create is used",
    "append": "Add the new prompt after the one already in the box, instead of replacing it",
    "smooth curves": "Curved outlines; off gives straight edges",
    "redraw only the painted area at full detail": "Redraws a box around the paint at full resolution, then pastes it back: best for hands, eyes or a logo",
    "move the file into that type's folder": "When you pick a type, the file moves into the LoRA folder set to that type",
    "show unknown type": "List LoRAs the app couldn't identify",
    "show all": "List every LoRA",
    "new random mix every time i press generate": "Each Generate picks a new set of random LoRAs",
    "✨ enhance prompt when generating": "Before each picture, a small language model turns your prompt into a detailed one",
    "hide nsfw": "Cover pictures, LoRAs and models marked NSFW",
    "hide people": "Hide pictures, LoRAs and models tagged with real people",
    "incl. prompt": "Also save the prompt in the preset",
    "use": "Use this",
    "＋ save": "Save the prompt in the box so you can use it again",
    "choose a folder of images…": "Pick the folder with your training pictures (they aren't moved or changed)",
    "change folder…": "Pick another folder",
    "use in create": "Pick this model in Create",
    "↻ refresh": "Look for new or removed files again",
    "vectorize": "Trace the picture into an SVG drawing of flat colour shapes",
    "show my files": "Open the folder the converted files were saved in",
    "＋ add watched folders…": "Watch more folders of models: every model in them appears in the app (⌘-click to pick several)",
    "↻ look for libraries again": "Look again for Stability Matrix, ComfyUI, A1111 / Forge, InvokeAI and Fooocus libraries, and re-read your watched folders",
    "pick a library folder…": "Use another app's model library that's somewhere unusual (e.g. on an external drive)",
    "how much space do they use?": "Show how much disk space the downloaded models take",
    "fix a type → models tab": "Correct a model or LoRA that's listed under the wrong type",
    "models tab": "Open the Models tab",
    "get photocraft (free, open source)": "Open PhotoCraft's download page",
    "★ starred only": "Show only the pictures you starred",
    "mirror images too (doubles variety; skip for text or logos)": "Also train on left-right flipped copies: more variety, but don't use it for text, logos or one-sided details",
  };
  // Buttons whose label changes (counts, shortcuts): matched by pattern.
  const BY_PATTERN = [
    [/^(\/|~)/, "Open this folder in Finder"],
    [/^generate\b/, "Make the picture(s) with these settings (⌘↵). While something is running it's added to the queue"],
    [/^add to queue$/, "Add this job to the queue: it runs when the one before it finishes"],
    [/^make comparison grid/, "Make one picture for each value and put them side by side in a grid"],
    [/^(all|added|sd 1\.5|sdxl|flux|flux\.2|ernie|pony|illustrious|noobai|anima)( \d+)?$/, "Show only this type"],
    [/^(embeddings|vaes|controlnets) \d+$/, "Show this kind of file"],
    [/^🖼 start from a picture/, "Go to Image to image to start from a picture or use ControlNet"],
    [/\bsd 1\.5 · your file\b|\(your file\)|· your file|checkpoint\)|\(diffusionbee\)/, "Use this model"],
    [/^(png|jpeg) \(/, "The file format new pictures are saved in"],
    [/^embed (the prompt|nothing)/, "What's written inside each saved picture file"],
    [/^(delete permanently|move to the trash)/, "What Delete does with a picture's file"],
    [/^(load a model as soon as|load anyway when|low-memory mode|show a live preview|check each picture|hide people models|hide nsfw pictures|when i add a lora|move my existing images|move the models already|also train the text encoder)/, "Switch this on or off (the note below explains it)"],
  ];

  let enabled = true;
  let tipEl = null;
  let timer = null;
  let current = null;

  function el() {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "proto-tip";
      tipEl.setAttribute("role", "tooltip");
      document.body.appendChild(tipEl);
    }
    return tipEl;
  }

  function hide() {
    clearTimeout(timer);
    current = null;
    if (tipEl) tipEl.classList.remove("show");
  }

  function textFor(node) {
    // the element's own title (Vue may have put a fresh one back since last time) or the saved one
    const t = node.getAttribute("title");
    if (t) {
      node.dataset.tip = t;
      node.removeAttribute("title");  // so macOS doesn't show its own, slower tip as well
    }
    if (node.dataset.tip) return node.dataset.tip;
    if (node.tagName === "BUTTON" || node.tagName === "LABEL" || node.getAttribute("role") === "button") {
      const label = (node.textContent || "").replace(/\s+/g, " ").trim().toLowerCase();
      if (BY_TEXT[label]) return BY_TEXT[label];
      const hit = BY_PATTERN.find(([re]) => re.test(label));
      return hit ? hit[1] : "";
    }
    return "";
  }

  function place(node) {
    const t = el();
    const r = node.getBoundingClientRect();
    t.style.left = "0px";
    t.style.top = "0px";
    t.classList.add("show");
    const w = t.offsetWidth, h = t.offsetHeight, pad = 8;
    let x = r.left + r.width / 2 - w / 2;
    x = Math.max(pad, Math.min(x, window.innerWidth - w - pad));
    // a knob's tip goes above it: below, it would cover the knob's number
    const above = node.closest && node.closest(".rack-knob") && r.top - h - 8 > pad;
    let y = above ? r.top - h - 8 : r.bottom + 8;
    if (!above && y + h > window.innerHeight - pad) y = r.top - h - 8;  // no room below: above it
    t.style.left = `${Math.round(x)}px`;
    t.style.top = `${Math.round(Math.max(pad, y))}px`;
  }

  const CONTROL = "button, label, input, select, textarea, a, [role=button], [role=switch]";

  function ownTip(node) {
    const t = node.getAttribute && node.getAttribute("title");
    if (t) { node.dataset.tip = t; node.removeAttribute("title"); }
    return (node.dataset && node.dataset.tip) || "";
  }

  // A slider, menu or box with no tip of its own borrows the one of the words that name it: its <label for>,
  // the label or hint just before it (e.g. "fidelity" ▸ slider), or the label it sits in.
  function borrowedTip(node) {
    if (!/^(INPUT|SELECT|TEXTAREA)$/.test(node.tagName)) return "";
    if (node.id) {
      const l = document.querySelector(`label[for="${CSS.escape(node.id)}"]`);
      if (l && (ownTip(l) || "")) return l.dataset.tip;
    }
    for (let sib = node.previousElementSibling, n = 0; sib && n < 3; sib = sib.previousElementSibling, n++) {
      const t = ownTip(sib) || (sib.querySelector && [...sib.querySelectorAll("[title], [data-tip]")].map(ownTip).find(Boolean));
      if (t) return t;
      if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(sib.tagName)) break;  // that one belongs to another control
    }
    return "";
  }

  // From what the pointer is on, up to the page: the first thing that has something to say.
  function findTip(target) {
    for (let node = target, depth = 0; node && node !== document.body && depth < 8; node = node.parentElement, depth++) {
      if (node.nodeType !== 1) continue;
      const text = textFor(node) || borrowedTip(node);
      if (text) return { node, text };
      if (node.matches && node.matches(CONTROL) && node.tagName !== "LABEL") {
        // a control with nothing to say: an enclosing label or card title may still explain it
        continue;
      }
    }
    return null;
  }

  // While a knob or slider is being turned (button held), no tip: it would cover the numbers being changed.
  // (Knobs use pointer events, which can swallow the mousedown that normally hides a tip.)
  let pressed = false;
  document.addEventListener("pointerdown", () => { pressed = true; hide(); }, true);
  for (const ev of ["pointerup", "pointercancel"]) document.addEventListener(ev, () => { pressed = false; }, true);
  window.addEventListener("blur", () => { pressed = false; });
  document.addEventListener("mouseover", (e) => {
    if (pressed) return;
    const hit = e.target && e.target.nodeType === 1 ? findTip(e.target) : null;
    const node = hit && hit.node;
    if (!node || node === current) return;
    hide();
    const text = hit.text;
    if (!enabled || !text) return;
    current = node;
    timer = setTimeout(() => {
      if (pressed || current !== node || !document.body.contains(node)) return;
      el().textContent = node.dataset.tip || text;
      place(node);
    }, DELAY_MS);
  }, true);
  document.addEventListener("mouseout", (e) => {
    if (current && (!e.relatedTarget || !current.contains(e.relatedTarget))) hide();
  }, true);
  for (const ev of ["mousedown", "keydown", "scroll", "wheel"]) document.addEventListener(ev, hide, true);
  window.addEventListener("blur", hide);
  // the thing it points at went away (a tab or panel changed): take the tip away too
  setInterval(() => {
    if (current && (!document.body.contains(current) || !current.offsetParent)) hide();
  }, 300);

  window.ProtoTips = {
    setEnabled(on) {
      enabled = !!on;
      if (!enabled) hide();
    },
    get enabled() { return enabled; },
    _byText: BY_TEXT,
    _find: (n) => findTip(n),
  };
})();
