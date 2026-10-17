/* Ready-made patterns for QR / pattern / illusion ControlNets (Image to image): black shapes on white with a margin,
 * drawn on a canvas. ProtoPatterns.LIST = [{id, label, tip}], ProtoPatterns.render(id, {text, invert, size}) -> canvas.
 * "qr" draws a real, scannable QR code of the text (qr.js, error correction H so it survives being painted over);
 * "text" writes the text in big bold letters.
 */
(function () {
  "use strict";
  const LIST = [
    { id: "qr", label: "QR code", tip: "A real QR code of the text or link you type (it still scans in the finished picture)" },
    { id: "text", label: "Word", tip: "Your word in big bold letters, hidden in the scene" },
    { id: "spiral", label: "Spiral", tip: "The classic hypnotic spiral illusion" },
    { id: "rings", label: "Rings", tip: "Circles inside circles" },
    { id: "checker", label: "Checks", tip: "A chessboard: works well with buildings, fields and tiles" },
    { id: "stripes", label: "Stripes", tip: "Diagonal stripes" },
    { id: "waves", label: "Waves", tip: "Flowing wavy bands" },
    { id: "sunburst", label: "Sunburst", tip: "Rays from the centre" },
    { id: "circle", label: "Circle", tip: "One big round shape" },
    { id: "heart", label: "Heart", tip: "A heart" },
    { id: "star", label: "Star", tip: "A five-pointed star" },
    { id: "triangle", label: "Triangle", tip: "A triangle" },
  ];

  function render(id, opts = {}) {
    const S = opts.size || 768;
    const c = document.createElement("canvas");
    c.width = c.height = S;
    const g = c.getContext("2d");
    g.fillStyle = "#fff";
    g.fillRect(0, 0, S, S);
    g.fillStyle = "#000";
    g.strokeStyle = "#000";
    const m = S * 0.1, W = S - 2 * m, cx = S / 2, cy = S / 2;  // a margin all round, like a QR code's quiet zone
    const text = String(opts.text || "").trim();
    if (id === "qr") {
      const q = window.ProtoQR.make(text || "https://github.com/theprotoart/protoart-studio", "H");
      const n = q.size + 8;  // + 4 light modules each side (the quiet zone scanners need)
      const cell = Math.floor(S / n), off = Math.floor((S - cell * q.size) / 2);
      q.modules.forEach((row, y) => row.forEach((dark, x) => { if (dark) g.fillRect(off + x * cell, off + y * cell, cell, cell); }));
    } else if (id === "text") {
      const word = (text || "ART").slice(0, 24);
      g.textAlign = "center";
      g.textBaseline = "middle";
      let fs = S * 0.7;
      g.font = `900 ${fs}px "Helvetica Neue", Arial Black, Arial, sans-serif`;
      while (fs > 20 && g.measureText(word).width > W) { fs *= 0.92; g.font = `900 ${fs}px "Helvetica Neue", Arial Black, Arial, sans-serif`; }
      g.fillText(word, cx, cy + fs * 0.04);
    } else if (id === "spiral") {
      g.beginPath();
      const turns = 5, R = W / 2;
      for (let a = 0; a <= turns * Math.PI * 2; a += 0.02) {
        const r = (a / (turns * Math.PI * 2)) * R;
        g.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
      }
      g.lineWidth = R / turns / 2;
      g.lineCap = "round";
      g.stroke();
    } else if (id === "rings") {
      const n = 6, R = W / 2;
      for (let i = n; i >= 1; i--) {
        g.fillStyle = i % 2 ? "#000" : "#fff";
        g.beginPath(); g.arc(cx, cy, (R * i) / n, 0, Math.PI * 2); g.fill();
      }
    } else if (id === "checker") {
      const n = 8, s = W / n;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if ((x + y) % 2 === 0) g.fillRect(m + x * s, m + y * s, s, s);
    } else if (id === "stripes") {
      g.save(); g.beginPath(); g.rect(m, m, W, W); g.clip();
      g.translate(cx, cy); g.rotate(-Math.PI / 4);
      const w = W / 10;
      for (let x = -S; x < S; x += 2 * w) g.fillRect(x, -S, w, 2 * S);
      g.restore();
    } else if (id === "waves") {
      g.save(); g.beginPath(); g.rect(m, m, W, W); g.clip();
      const bands = 7, h = W / bands;
      for (let b = 0; b < bands; b += 2) {
        g.beginPath();
        for (let x = m; x <= m + W; x += 4) g.lineTo(x, m + b * h + Math.sin((x - m) / W * Math.PI * 4) * h * 0.45);
        for (let x = m + W; x >= m; x -= 4) g.lineTo(x, m + (b + 1) * h + Math.sin((x - m) / W * Math.PI * 4) * h * 0.45);
        g.closePath(); g.fill();
      }
      g.restore();
    } else if (id === "sunburst") {
      const n = 16, R = W / 2;
      for (let i = 0; i < n; i += 2) {
        g.beginPath(); g.moveTo(cx, cy);
        g.arc(cx, cy, R, (i / n) * Math.PI * 2, ((i + 1) / n) * Math.PI * 2);
        g.closePath(); g.fill();
      }
    } else if (id === "circle") {
      g.beginPath(); g.arc(cx, cy, W * 0.4, 0, Math.PI * 2); g.fill();
    } else if (id === "heart") {
      const s = W / 2;
      g.beginPath();
      g.moveTo(cx, cy + s * 0.85);
      g.bezierCurveTo(cx - s * 1.25, cy + s * 0.05, cx - s * 0.75, cy - s * 0.95, cx, cy - s * 0.35);
      g.bezierCurveTo(cx + s * 0.75, cy - s * 0.95, cx + s * 1.25, cy + s * 0.05, cx, cy + s * 0.85);
      g.fill();
    } else if (id === "star") {
      const R = W * 0.48, r = R * 0.42;
      g.beginPath();
      for (let i = 0; i < 10; i++) {
        const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r : R;
        g.lineTo(cx + rr * Math.cos(a), cy + rr * Math.sin(a) + R * 0.08);
      }
      g.closePath(); g.fill();
    } else if (id === "triangle") {
      g.beginPath(); g.moveTo(cx, m + W * 0.08); g.lineTo(m + W * 0.96, m + W * 0.9); g.lineTo(m + W * 0.04, m + W * 0.9); g.closePath(); g.fill();
    }
    if (opts.invert) {  // white shapes on black
      const d = g.getImageData(0, 0, S, S);
      for (let i = 0; i < d.data.length; i += 4) { d.data[i] = 255 - d.data[i]; d.data[i + 1] = 255 - d.data[i + 1]; d.data[i + 2] = 255 - d.data[i + 2]; }
      g.putImageData(d, 0, 0);
    }
    return c;
  }

  window.ProtoPatterns = { LIST, render };
})();
