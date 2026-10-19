// Protoart Studio main process: owns the window and the Python backend process.
const { app, BrowserWindow, ipcMain, shell, dialog, Menu } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const ROOT = path.join(__dirname, "..");
// Standalone app (Protoart Studio.app): Python and the backend are inside the app's Resources folder.
// Development (start.sh): the backend folder next to app/ with its own .venv.
// (app.isPackaged only checks that the executable isn't named "Electron", which it still is inside
// Protoart Studio.app, so also check whether these files live inside an app bundle's Resources folder.)
const PACKAGED = app.isPackaged || __dirname.startsWith(process.resourcesPath + path.sep);
const BACKEND = PACKAGED
  ? path.join(process.resourcesPath, "backend", "bee_backend.py")
  : path.join(ROOT, "backend", "bee_backend.py");
const VENV_PY = PACKAGED
  ? path.join(process.resourcesPath, "python", "bin", "python3")
  : path.join(ROOT, "backend", ".venv", "bin", "python");

app.setName("Protoart Studio");
// Preferences folder: move the old "BeeStudio" one over once so saved settings survive the rename.
{
  const appData = app.getPath("appData");
  const newDir = path.join(appData, "Protoart Studio");
  const oldDir = path.join(appData, "BeeStudio");
  try {
    if (!fs.existsSync(newDir) && fs.existsSync(oldDir)) fs.renameSync(oldDir, newDir);
  } catch (err) {
    console.error("couldn't move preferences folder", err);
  }
  app.setPath("userData", newDir);
}

let win = null;
let backend = null;
let jobsInQueue = 0; // from the backend's "queue" messages; dev auto-restart waits until this is 0
let stderrTail = "";
let quitting = false;

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

function startBackend() {
  if (PACKAGED && !fs.existsSync(VENV_PY)) {
    // never fall back to some other Python on this Mac: the app's own one is missing, so say so
    send("backend:exit", { code: -1, tail: `The app's built-in Python is missing (${VENV_PY}). Reinstall Protoart Studio.` });
    return;
  }
  const python = fs.existsSync(VENV_PY) ? VENV_PY : "python3";
  const env = { ...process.env, PYTORCH_ENABLE_MPS_FALLBACK: "1", PYTHONUNBUFFERED: "1", PYTHONNOUSERSITE: "1" };
  delete env.PYTHONHOME; // never let another Python install's settings leak in
  delete env.PYTHONPATH;
  if (PACKAGED) {
    env.PYTHONDONTWRITEBYTECODE = "1"; // the app bundle stays untouched (it's read-only and signed)
    // Apps opened from Finder get a minimal PATH; add the usual places for command-line tools
    env.PATH = ["/opt/homebrew/bin", "/usr/local/bin", env.PATH || "/usr/bin:/bin:/usr/sbin:/sbin"].join(":");
  }
  const proc = spawn(python, [BACKEND], { cwd: path.dirname(BACKEND), env });
  proc.expectedExit = false;
  backend = proc;

  let buf = "";
  proc.stdout.on("data", (chunk) => {
    if (proc !== backend) return; // output from an old process that's shutting down
    buf += chunk.toString("utf8");
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        if (msg && msg.type === "queue" && Array.isArray(msg.jobs)) jobsInQueue = msg.jobs.length;
        send("backend:msg", msg);
      } catch {
        send("backend:log", line);
      }
    }
  });
  proc.stderr.on("data", (chunk) => {
    const s = chunk.toString("utf8");
    if (proc === backend) stderrTail = (stderrTail + s).slice(-2000);
    send("backend:log", s);
  });
  proc.on("close", (code, signal) => {
    if (proc === backend) { backend = null; jobsInQueue = 0; } // only clear it if it's still the current process
    if (!quitting && !proc.expectedExit) {
      const why = signal ? `(ended by signal ${signal})` : `(exit code ${code})`;
      console.error("backend stopped unexpectedly", why);
      send("backend:exit", { code, signal: signal || null, tail: stderrTail + "\n" + why });
    }
  });
  proc.on("error", (err) => send("backend:exit", { code: -1, tail: String(err) }));
}

// Stop a backend process right away (used by Force stop / restart). SIGKILL if it doesn't exit quickly.
function stopBackend() {
  const proc = backend;
  if (!proc) return;
  proc.expectedExit = true;
  backend = null;
  try { proc.kill("SIGTERM"); } catch {}
  setTimeout(() => { try { if (proc.exitCode === null) proc.kill("SIGKILL"); } catch {} }, 2000);
}

// Window size/position is remembered between launches.
const BOUNDS_FILE = () => path.join(app.getPath("userData"), "window.json");
function loadBounds() {
  try {
    return JSON.parse(fs.readFileSync(BOUNDS_FILE(), "utf8"));
  } catch {
    return {};
  }
}
function saveBounds() {
  if (!win || win.isDestroyed()) return;
  try {
    fs.writeFileSync(BOUNDS_FILE(), JSON.stringify({ ...win.getNormalBounds(), maximized: win.isMaximized() }));
  } catch {}
}

// Saved window position only if it's still on a connected screen (a display that was unplugged, or a
// position saved mid-crash, would otherwise open the window where it can't be seen).
function visibleBounds(b) {
  const { screen } = require("electron");
  if (!Number.isFinite(b.x) || !Number.isFinite(b.y)) return { width: b.width, height: b.height, maximized: b.maximized };
  const w = Math.max(720, b.width || 1320), h = Math.max(520, b.height || 880);
  const onScreen = screen.getAllDisplays().some(({ workArea: a }) =>
    b.x + 80 > a.x && b.x + 80 < a.x + a.width && b.y + 20 >= a.y && b.y + 20 < a.y + a.height);
  return onScreen ? { ...b, width: w, height: h } : { width: Math.min(w, 1320), height: Math.min(h, 880), maximized: b.maximized };
}

function createWindow() {
  let b = {};
  try { b = visibleBounds(loadBounds()); } catch (err) { console.error("window position", err); }
  win = new BrowserWindow({
    width: b.width || 1320,
    height: b.height || 880,
    x: b.x,
    y: b.y,
    minWidth: 720,
    minHeight: 520,
    resizable: true,
    maximizable: true,
    fullscreenable: true,
    title: "Protoart Studio",
    icon: path.join(__dirname, "build", "icon.png"),
    backgroundColor: "#15161a",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      devTools: !PACKAGED,  // the installed app has no developer tools (no menu item, no shortcut)
    },
  });
  if (b.maximized) win.maximize();
  // always bring the window to the front when the app starts (started from Terminal it may open behind it)
  win.once("ready-to-show", () => { win.show(); win.focus(); if (process.platform === "darwin") app.focus({ steal: true }); });
  let t = null;
  const later = () => { clearTimeout(t); t = setTimeout(saveBounds, 400); };
  win.on("resize", later);
  win.on("move", later);
  win.on("close", saveBounds);
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
  // Open external links in the browser, never inside the app window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e) => e.preventDefault());
}

// ---------------------------------------------------------------------------------------------------
// IPC: the page's only way to reach the Mac. Every handler checks that the call comes from the app's
// own page, and limits what it may touch (security audit M1).
const RENDERER_INDEX = path.join(__dirname, "renderer", "index.html");
const DATA_HOME = process.env.PROTOART_HOME || process.env.BEESTUDIO_HOME ||
  (fs.existsSync(path.join(app.getPath("home"), ".protoartstudio")) || !fs.existsSync(path.join(app.getPath("home"), ".beestudio"))
    ? path.join(app.getPath("home"), ".protoartstudio") : path.join(app.getPath("home"), ".beestudio"));
const EDITS_DIR = path.join(DATA_HOME, "edits");
const IMAGE_EXT = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]);
// macOS treats these folders as apps or packages: "opening" one would run it
const BUNDLE_RE = /\.(app|bundle|framework|plugin|kext|pkg|mpkg|prefpane|appex|xpc|command|tool|workflow)$/i;

function fromOurPage(e) {
  try {
    const u = new URL(e.senderFrame.url);
    return u.protocol === "file:" && path.resolve(decodeURIComponent(u.pathname)) === RENDERER_INDEX;
  } catch {
    return false;
  }
}
function handle(channel, fn) {
  ipcMain.handle(channel, (e, ...args) => {
    if (!fromOurPage(e)) throw new Error(`refused: ${channel} from an unexpected page`);
    return fn(e, ...args);
  });
}
function realInside(p, dir) {
  try {
    const real = fs.realpathSync(p);
    const base = fs.realpathSync(dir);
    return real === base || real.startsWith(base + path.sep);
  } catch {
    return false;
  }
}

handle("backend:send", (_e, msg) => {
  if (!backend || !msg || typeof msg !== "object") return false;
  backend.stdin.write(JSON.stringify(msg) + "\n");
  return true;
});
handle("backend:restart", () => {
  stopBackend();
  stderrTail = "";
  jobsInQueue = 0;
  startBackend();
  return true;
});
// Tools -> Force quit: stop the engine at once (no waiting for the current step) and close the app.
handle("app:forceQuit", () => {
  quitting = true;
  const proc = backend;
  backend = null;
  try { if (proc) proc.kill("SIGKILL"); } catch {}
  saveBounds();
  setTimeout(() => app.exit(0), 50);
  return true;
});
// Open a picture in an outside image editor (PhotoCraft by default): only the working copies the app
// made in ~/.protoartstudio/edits, and only an app given by name. macOS: `open -a <App> <file>`.
handle("editor:open", (_e, file, appName) => new Promise((resolve) => {
  if (typeof file !== "string" || !path.isAbsolute(file) || !fs.existsSync(file)) return resolve({ ok: false, error: "missing file" });
  if (!realInside(file, EDITS_DIR) || !IMAGE_EXT.has(path.extname(file).toLowerCase())) {
    return resolve({ ok: false, error: "only pictures prepared for editing can be opened" });
  }
  const name = typeof appName === "string" && appName.trim() ? appName.trim() : "PhotoCraft";
  if (name !== "(default app)" && !/^[\w .+'()&-]{1,64}$/.test(name)) return resolve({ ok: false, error: "not a valid app name" });
  if (process.platform !== "darwin") {
    shell.openPath(file).then((err) => resolve(err ? { ok: false, error: err } : { ok: true }));
    return;
  }
  const args = name === "(default app)" ? [file] : ["-a", name, file];
  const p = spawn("open", args);
  let err = "";
  p.stderr.on("data", (d) => { err += d.toString(); });
  p.on("close", (code) => resolve(code === 0 ? { ok: true } : { ok: false, notInstalled: /Unable to find application/i.test(err), error: err.trim() }));
  p.on("error", (e) => resolve({ ok: false, error: String(e) }));
}));
handle("shell:openExternal", (_e, url) => {
  if (typeof url === "string" && /^https:\/\//.test(url)) shell.openExternal(url);
});
handle("window:reset", () => {
  try { fs.unlinkSync(BOUNDS_FILE()); } catch {}
  if (win && !win.isDestroyed()) {
    if (win.isMaximized()) win.unmaximize();
    win.setSize(1320, 880);
    win.center();
  }
  return true;
});
// Reveal in Finder only selects the item; it never opens or runs it.
handle("shell:showItem", (_e, p) => {
  if (typeof p === "string" && path.isAbsolute(p)) shell.showItemInFolder(p);
});
// Open a folder in Finder: folders only, never an app or package; new folders only inside the app's data folder.
handle("shell:openFolder", (_e, p) => {
  if (typeof p !== "string" || !path.isAbsolute(p)) return "not a folder";
  const target = path.resolve(p);
  if (BUNDLE_RE.test(target)) return "not a folder";
  if (!fs.existsSync(target)) {
    if (!(target === DATA_HOME || target.startsWith(DATA_HOME + path.sep))) return "folder not found";
    fs.mkdirSync(target, { recursive: true });
  }
  try {
    if (!fs.statSync(target).isDirectory() || BUNDLE_RE.test(fs.realpathSync(target))) return "not a folder";
  } catch {
    return "folder not found";
  }
  return shell.openPath(target);
});

// pickFolder(currentPath) -> one folder (image folder); pickFolder({title, defaultPath, multi}) -> list when multi
handle("dialog:pickFolder", async (_e, opts) => {
  const o = typeof opts === "object" && opts ? opts : { defaultPath: opts };
  const res = await dialog.showOpenDialog(win, {
    title: typeof o.title === "string" ? o.title.slice(0, 120) : "Choose where Protoart Studio saves images",
    defaultPath: typeof o.defaultPath === "string" ? o.defaultPath : undefined,
    properties: ["openDirectory", "createDirectory", ...(o.multi ? ["multiSelections"] : [])],
    buttonLabel: typeof o.button === "string" ? o.button.slice(0, 40) : "Use this folder",
  });
  if (res.canceled) return null;
  return o.multi ? res.filePaths : res.filePaths[0];
});
handle("dialog:pickModels", async () => {
  const res = await dialog.showOpenDialog(win, {
    title: "Add base models from your Mac (.safetensors / .ckpt files, or diffusers model folders)",
    properties: ["openFile", "openDirectory", "multiSelections"],
    filters: [{ name: "Model checkpoints", extensions: ["safetensors", "ckpt"] }],
    buttonLabel: "Add",
  });
  return res.canceled ? [] : res.filePaths;
});
// Adjust (Lightroom-style sliders and filters, done on the GPU in the page): the page needs the picture's
// bytes (a file:// image can't be read back from a WebGL canvas), and hands back the finished PNG.
handle("image:read", async (_e, p) => {
  if (typeof p !== "string" || !path.isAbsolute(p) || !IMAGE_EXT.has(path.extname(p).toLowerCase())) throw new Error("not a picture");
  const st = await fs.promises.stat(p);
  if (!st.isFile() || st.size > 400 * 1024 * 1024) throw new Error("not a picture");
  return fs.promises.readFile(p);
});
handle("adjust:writeTemp", async (_e, bytes) => {
  if (!(bytes instanceof Uint8Array) || bytes.length < 8 || bytes.length > 1024 * 1024 * 1024) throw new Error("not a picture");
  const PNG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (!PNG.every((b, i) => bytes[i] === b)) throw new Error("not a PNG");
  const dir = path.join(DATA_HOME, "adjust");
  await fs.promises.mkdir(dir, { recursive: true });
  const file = path.join(dir, `adjust_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.png`);
  await fs.promises.writeFile(file, bytes);
  return file;
});
handle("dialog:pickEmbeddings", async () => {
  const res = await dialog.showOpenDialog(win, {
    title: "Add embeddings (textual inversions: .safetensors or .pt files)",
    properties: ["openFile", "multiSelections"],
    filters: [{ name: "Embeddings", extensions: ["safetensors", "pt", "bin"] }],
    buttonLabel: "Add",
  });
  return res.canceled ? [] : res.filePaths;
});
// Move pictures to the macOS Trash (Settings -> "Move to Trash"): regular image files only.
handle("shell:trash", async (_e, paths) => {
  const errors = [];
  for (const p of Array.isArray(paths) ? paths.slice(0, 5000) : []) {
    if (typeof p !== "string" || !path.isAbsolute(p)) continue;
    let st;
    try { st = fs.lstatSync(p); } catch { continue; }
    if (!st.isFile() || !IMAGE_EXT.has(path.extname(p).toLowerCase())) {
      errors.push(`${p}: only picture files can be moved to the Trash from here`);
      continue;
    }
    try {
      await shell.trashItem(p);
    } catch (err) {
      errors.push(`${p}: ${err.message}`);
    }
  }
  return errors;
});

// Move one of your own model files (or a diffusers model folder) to the Trash: Models tab → Move to Trash.
const MODEL_EXT = new Set([".safetensors", ".ckpt", ".tdict", ".gguf", ".bin"]);
handle("shell:trashModel", async (_e, p) => {
  if (typeof p !== "string" || !path.isAbsolute(p)) return "Not a file path";
  let st;
  try { st = fs.lstatSync(p); } catch { return "The file isn't there any more"; }
  const ok = st.isFile() ? MODEL_EXT.has(path.extname(p).toLowerCase())
    : st.isDirectory() && fs.existsSync(path.join(p, "model_index.json"));
  if (!ok) return "Only model files (.safetensors, .ckpt…) or model folders can be moved to the Trash from here";
  try {
    await shell.trashItem(p);
    return "";
  } catch (err) {
    return err.message;
  }
});

// Auto-reload while Protoart Studio is being developed: when files in renderer/ change the window
// reloads; when backend .py files change the backend restarts (images in progress finish first
// only if you wait; a restart stops the current job).
function watchForUpdates() {
  let timer = null;
  let backendChanged = false;
  const schedule = (isBackend) => {
    backendChanged = backendChanged || isBackend;
    clearTimeout(timer);
    timer = setTimeout(function fire() {
      if (backendChanged && jobsInQueue > 0) {
        // a job is running: restarting now would stop it, so wait until the queue is empty
        timer = setTimeout(fire, 2000);
        return;
      }
      if (backendChanged) {
        stopBackend();
        stderrTail = "";
        startBackend();
      }
      if (win && !win.isDestroyed()) win.webContents.reloadIgnoringCache();
      backendChanged = false;
    }, 1200);
  };
  const watch = (dir, isBackend, filter) => {
    try {
      fs.watch(dir, { recursive: true }, (_ev, file) => {
        if (file && filter(String(file))) schedule(isBackend);
      });
    } catch (err) {
      console.error("watch failed", dir, err);
    }
  };
  watch(path.join(__dirname, "renderer"), false, (f) => /\.(js|html|css)$/.test(f));
  watch(path.join(ROOT, "backend"), true, (f) => f.endsWith(".py") && !f.includes(".venv") && !f.includes("__pycache__"));
}

// Every web page the app ever creates: no extra windows (links open in the browser, https only),
// no navigation away from the app, no <webview> (security audit L4).
app.on("web-contents-created", (_e, contents) => {
  contents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  contents.on("will-navigate", (e) => e.preventDefault());
  contents.on("will-attach-webview", (e) => e.preventDefault());
});

// The app's own menu bar. Electron's default one has View → Reload / Force Reload / Toggle Developer Tools, which
// people using the app shouldn't see (a reload throws away the screen's state mid-job). The dev version
// (Start Protoart Studio.command) keeps them for testing.
function setAppMenu() {
  const isMac = process.platform === "darwin";
  const view = [
    { role: "resetZoom", label: "Actual Size" }, { role: "zoomIn" }, { role: "zoomOut" },
    { type: "separator" }, { role: "togglefullscreen" },
  ];
  if (!PACKAGED) view.unshift({ role: "reload" }, { role: "forceReload" }, { role: "toggleDevTools" }, { type: "separator" });
  const template = [
    ...(isMac ? [{ label: app.name, submenu: [
      { role: "about" }, { type: "separator" }, { role: "hide" }, { role: "hideOthers" }, { role: "unhide" },
      { type: "separator" }, { role: "quit" }] }] : []),
    { label: "File", submenu: [isMac ? { role: "close" } : { role: "quit" }] },
    { label: "Edit", submenu: [  // keeps ⌘C / ⌘V / ⌘A / ⌘Z working in the prompt boxes
      { role: "undo" }, { role: "redo" }, { type: "separator" }, { role: "cut" }, { role: "copy" }, { role: "paste" },
      { role: "pasteAndMatchStyle" }, { role: "delete" }, { role: "selectAll" }] },
    { label: "View", submenu: view },
    { label: "Window", submenu: [{ role: "minimize" }, { role: "zoom" }, ...(isMac ? [{ type: "separator" }, { role: "front" }] : [{ role: "close" }])] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

app.whenReady().then(() => {
  setAppMenu();
  // No camera, microphone, location, notifications…: the app never needs them (security audit L2).
  // Writing to the clipboard (Backend log -> Copy all) is the one thing the page may do.
  const { session } = require("electron");
  const allowed = new Set(["clipboard-sanitized-write"]);
  session.defaultSession.setPermissionRequestHandler((_wc, permission, cb) => cb(allowed.has(permission)));
  session.defaultSession.setPermissionCheckHandler((_wc, permission) => allowed.has(permission));
  const icon = path.join(__dirname, "build", "icon.png");
  if (process.platform === "darwin" && app.dock && fs.existsSync(icon)) app.dock.setIcon(icon);
  // the window first, so something always appears even if the engine can't start
  try { createWindow(); } catch (err) { console.error("couldn't open the window", err); }
  try { startBackend(); } catch (err) {
    console.error("couldn't start the engine", err);
    send("backend:exit", { code: -1, tail: String(err) });
  }
  if (!PACKAGED) watchForUpdates();
});
app.on("before-quit", () => {
  quitting = true;
  stopBackend();
});
app.on("window-all-closed", () => app.quit());
// Dock icon clicked with no window open (macOS): open one again.
app.on("activate", () => { if (!win || win.isDestroyed()) createWindow(); });
process.on("uncaughtException", (err) => console.error("Protoart Studio main process error:", err));
