// Exposes a small, explicit API to the page. The page gets no Node.js access.
const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("bee", {
  send: (msg) => ipcRenderer.invoke("backend:send", msg),
  restart: () => ipcRenderer.invoke("backend:restart"),
  // Tools -> Force quit: kill the engine immediately and close the app.
  forceQuit: () => ipcRenderer.invoke("app:forceQuit"),
  onMessage: (fn) => ipcRenderer.on("backend:msg", (_e, m) => fn(m)),
  onLog: (fn) => ipcRenderer.on("backend:log", (_e, s) => fn(s)),
  onExit: (fn) => ipcRenderer.on("backend:exit", (_e, info) => fn(info)),
  showInFolder: (p) => ipcRenderer.invoke("shell:showItem", p),
  // Save a copy of a picture where you choose / put it on the clipboard (to paste into another app).
  saveImageAs: (p) => ipcRenderer.invoke("image:saveAs", p),
  copyImage: (p) => ipcRenderer.invoke("image:copy", p),
  openFolder: (p) => ipcRenderer.invoke("shell:openFolder", p),
  pickFolder: (current) => ipcRenderer.invoke("dialog:pickFolder", current),
  // Pick model files (.safetensors/.ckpt) and/or model folders from this Mac.
  pickModels: () => ipcRenderer.invoke("dialog:pickModels"),
  // Tools → Convert: pick .tdict files or folders.
  pickTdict: () => ipcRenderer.invoke("dialog:pickTdict"),
  // Pick embedding files (.safetensors / .pt) to copy into Protoart's embeddings folder.
  pickEmbeddings: () => ipcRenderer.invoke("dialog:pickEmbeddings"),
  // Adjust: read a picture's bytes, and save the adjusted PNG for the backend to add to the Library.
  readImage: (p) => ipcRenderer.invoke("image:read", p),
  writeAdjusted: (bytes) => ipcRenderer.invoke("adjust:writeTemp", bytes),
  trash: (paths) => ipcRenderer.invoke("shell:trash", paths),
  trashModel: (p) => ipcRenderer.invoke("shell:trashModel", p),
  // Open a picture in an outside editor (PhotoCraft, Photoshop…) and open web links in the browser.
  openInEditor: (file, appName) => ipcRenderer.invoke("editor:open", file, appName),
  openExternal: (url) => ipcRenderer.invoke("shell:openExternal", url),
  // Forget the saved window size/position (Settings -> Reset settings).
  resetWindow: () => ipcRenderer.invoke("window:reset"),
  // Real filesystem path of a dragged-in file (File.path was removed in recent Electron).
  pathForFile: (file) => webUtils.getPathForFile(file),
});
