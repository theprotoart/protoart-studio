// UI smoke test: opens the app's page in Chromium with a pretend backend, visits every tab (and Image to image's
// modes), and reports buttons and controls that are cut off or stick out of their panel — the kind of layout problem
// that otherwise only shows up after a build. Screenshots go to packaging/ui_smoke/out/.
//
//   node packaging/ui_smoke/smoke.js [path/to/vue.global.prod.js]
//
// Needs Playwright with Chromium (npm i -D playwright; npx playwright install chromium). Vue is taken from
// app/node_modules (or the path given). Exit code 1 when something is cut off.
const fs = require("fs");
const os = require("os");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(__dirname, "out");
const vueArg = process.argv[2] || path.join(ROOT, "app", "node_modules", "vue", "dist", "vue.global.prod.js");

// the pretend backend: answers every command with something of the right shape, so the page can start
const MOCK = `
(() => {
  const listeners = [];
  const lists = /^(list_|library_list|libraries$|queue_get|poselib_list|wildcards_list|convert_list|library_loras|control_types|.*_dirs_get$)/;
  const answers = {
    system_info: { ram_gb: 8, free_disk_gb: 120, device: "mps", torch: "2.8", diffusers: "0.36", data_dir: "/Users/test/.protoartstudio" },
    list_models: [
      { id: "sd15", label: "Stable Diffusion 1.5", family: "sd15", downloaded: true, installed: true },
      { id: "sdxl", label: "SDXL Base 1.0 · 8-bit", family: "sdxl", downloaded: true, installed: true },
      { id: "klein", label: "FLUX.2 Klein 4B · 4-bit", family: "flux2", engine: "mlx", downloaded: true, installed: true },
    ],
    read_settings: {}, privacy_get: { strict: false }, main_folder_get: { path: "/Users/test/Protoart", subs: [], access: true },
    images_dir_get: { path: "/Users/test/Pictures/Protoart", access: true, default: true }, store_info: { kinds: [] }, log_tail: { text: "" },
    diffusionbee_info: { enabled: false, folder: "", loras: [], checkpoints: [], embeddings: [], skipped: [] },
    library_list: { rows: [], total: 0 }, queue_get: [], edit_pending: { pending: [], folder: "" },
    list_vaes: { vaes: [], folder: "" }, list_controlnets: { controlnets: [], folder: "" }, libraries_get: { offer: [], used: [], found: [] },
    lora_dirs_get: { dirs: [], default: "/Users/test/.protoartstudio/loras", families: {} }, model_dirs_get: { dirs: [] },
    embedding_dirs_get: { dirs: [], default: "/Users/test/.protoartstudio/embeddings" }, list_loras: [], export_presets_get: [], adjust_presets_get: [],
  };
  window.bee = {
    send(msg) {
      setTimeout(() => {
        let data = answers[msg.cmd];
        if (data === undefined) data = lists.test(msg.cmd) ? [] : {};
        listeners.forEach((fn) => fn({ type: "result", id: msg.id, data }));
      }, 5);
      return Promise.resolve(true);
    },
    onMessage(fn) { listeners.push(fn); setTimeout(() => fn({ type: "ready" }), 20); },
    onLog() {}, onExit() {}, restart: async () => true, forceQuit: async () => true,
    showInFolder: async () => {}, openFolder: async () => {}, readImage: async () => new Uint8Array(),
    listFonts: async () => ["Helvetica Neue", "Avenir Next", "Futura"], pathForFile: (f) => "/tmp/" + (f && f.name),
    trash: async () => [], openExternal: async () => {}, writeAdjusted: async () => "/tmp/x.png",
  };
  window.__errors = [];
  window.addEventListener("error", (e) => window.__errors.push(String(e.message)));
})();
`;

// what counts as cut off: a button or control wider than its own box (text clipped), or reaching past the edge of
// the panel / rack module it sits in
const CHECK = `
(() => {
  const bad = [];
  const label = (el) => (el.innerText || el.value || el.getAttribute("title") || el.tagName).trim().replace(/\\s+/g, " ").slice(0, 50);
  const visible = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 2 && r.height > 2 && s.visibility !== "hidden" && s.display !== "none" && el.offsetParent !== null; };
  for (const el of document.querySelectorAll("button, select, .chip, label.check, input[type=range]")) {
    if (!visible(el)) continue;
    if (el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).overflow !== "visible" && el.tagName !== "SELECT")
      bad.push({ kind: "text cut off", what: label(el) });
    const box = el.closest(".rack-module, .card, .mod-card, .i2i-panel");
    if (box) {
      const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
      if (r.right > b.right + 2 || r.left < b.left - 2) bad.push({ kind: "sticks out of its panel", what: label(el), by: Math.round(Math.max(r.right - b.right, b.left - r.left)) + " px" });
    }
  }
  return bad;
})()
`;

(async () => {
  const { chromium } = require("playwright");
  if (!fs.existsSync(vueArg)) { console.error("Vue not found at " + vueArg); process.exit(2); }
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "protoart-ui-"));
  fs.cpSync(path.join(ROOT, "app", "renderer"), path.join(tmp, "renderer"), { recursive: true });
  fs.mkdirSync(path.join(tmp, "node_modules", "vue", "dist"), { recursive: true });
  fs.copyFileSync(vueArg, path.join(tmp, "node_modules", "vue", "dist", "vue.global.prod.js"));
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  let problems = 0;
  const looks = (process.env.UI_LOOKS || "").split(",").filter(Boolean);  // e.g. UI_LOOKS=powder:light,powder:dark
  for (const width of [1440, 1180]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    await page.addInitScript(MOCK);
    // the patchbay switched on, so the Tools screenshot shows it (its rows of modules)
    await page.addInitScript(() => { try { localStorage.setItem("protoart.patchOn", "true"); } catch {} });
    if (looks.length) {  // screenshots in other looks (skin:theme) too
      const [skin, theme] = looks[0].split(":");
      await page.addInitScript(([k, t]) => { localStorage.setItem("protoart.skin", JSON.stringify(k)); if (t) localStorage.setItem("protoart.theme", JSON.stringify(t)); }, [skin, theme]);
    }
    page.on("console", (m) => { if (m.type() === "error") console.log(`  [console] ${m.text().split("\n").slice(0, 4).join(" | ").slice(0, 400)}`); });
    page.on("pageerror", (e) => console.log(`  [page error] ${(e.stack || e.message).split("\n").slice(0, 3).join(" | ")}`));
    await page.goto("file://" + path.join(tmp, "renderer", "index.html"));
    await page.waitForTimeout(1500);
    const tabs = ["create", "i2i", "tools", "train", "library", "models", "settings", "help"];  // the order of the tab buttons
    for (let ti = 0; ti < tabs.length; ti++) {
      const t = tabs[ti];
      const ok = await page.evaluate((i) => { const b = document.querySelectorAll("nav > button")[i]; if (b) b.click(); return !!b; }, ti);
      if (!ok) {  // the previous tab broke the page (with the pretend backend's answers): start it again
        console.log(`  (the page had to be reloaded before ${t})`);
        await page.reload(); await page.waitForTimeout(1500);
        await page.evaluate((i) => document.querySelectorAll("nav > button")[i].click(), ti);
      }
      await page.waitForTimeout(600);
      // Image to image's modes: click each mode button (by its words)
      const views = t === "i2i" ? ["Change it", "Edit with words", "Remove", "Inpaint", "Text & balloons", "Relight", "Camera angle", "Combine"] : [null];
      for (const v of views) {
        if (v) {
          const found = await page.evaluate((v) => {
            const b = [...document.querySelectorAll("button")].find((x) => x.offsetParent && x.textContent.replace(/^\W+/, "").trim().startsWith(v));
            if (b) b.click();
            return !!b;
          }, v);
          if (!found) { console.log(`  (no "${v}" button)`); continue; }
          await page.waitForTimeout(350);
        }
        const name = `${width}-${t}${v ? "-" + v.toLowerCase().replace(/\W+/g, "-") : ""}`;
        await page.screenshot({ path: path.join(OUT, name + ".png"), fullPage: true });
        const bad = await page.evaluate(CHECK);
        if (bad.length) {
          problems += bad.length;
          console.log(`${name}: ${bad.length} problem${bad.length > 1 ? "s" : ""}`);
          for (const b of bad.slice(0, 12)) console.log(`   · ${b.kind}${b.by ? " (" + b.by + ")" : ""}: ${b.what}`);
        } else console.log(`${name}: ok`);
      }
    }
    const errs = await page.evaluate(() => window.__errors);
    if (errs.length) console.log("  page errors:", errs.slice(0, 5));
    await page.close();
  }
  await browser.close();
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(problems ? `\n${problems} layout problem(s); screenshots in ${OUT}` : `\nNo layout problems; screenshots in ${OUT}`);
  process.exit(problems ? 1 : 0);
})();
