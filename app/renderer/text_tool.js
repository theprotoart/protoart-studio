// 🔤 Replace text: draw text onto a picture like a paint program's Type tool. The layers are previewed as HTML
// over the picture and drawn onto a canvas at the picture's full size to save, with the same maths for both.
(function () {
  const FONTS = [
    ["Helvetica Neue", "Helvetica Neue, Helvetica, Arial, sans-serif"], ["Arial", "Arial, Helvetica, sans-serif"],
    ["Avenir Next", "Avenir Next, Avenir, sans-serif"], ["Futura", "Futura, Trebuchet MS, sans-serif"],
    ["Gill Sans", "Gill Sans, Gill Sans MT, sans-serif"], ["Impact", "Impact, Haettenschweiler, sans-serif"],
    ["Georgia", "Georgia, serif"], ["Times New Roman", "Times New Roman, Times, serif"], ["Baskerville", "Baskerville, Georgia, serif"],
    ["Didot", "Didot, Bodoni 72, serif"], ["Copperplate", "Copperplate, serif"], ["American Typewriter", "American Typewriter, Courier New, monospace"],
    ["Courier New", "Courier New, Courier, monospace"], ["Menlo", "Menlo, Monaco, monospace"], ["Marker Felt", "Marker Felt, Comic Sans MS, cursive"],
    ["Chalkboard SE", "Chalkboard SE, Comic Sans MS, cursive"], ["Snell Roundhand", "Snell Roundhand, Apple Chancery, cursive"],
    ["Bradley Hand", "Bradley Hand, cursive"], ["Papyrus", "Papyrus, fantasy"],
  ];
  const stack = (name) => (FONTS.find((f) => f[0] === name) || FONTS[0])[1];

  function layer(x, y, size) {
    return { text: "Your text", x, y, size: Math.round(size), font: "Helvetica Neue", colour: "#ffffff", bold: true, italic: false,
             align: "center", stroke: 0, strokeColour: "#000000", shadow: 0, rotate: 0, opacity: 1, spacing: 0, lineHeight: 1.15 };
  }
  function fontCss(L, px) {
    return `${L.italic ? "italic " : ""}${L.bold ? "700 " : "400 "}${px}px ${stack(L.font)}`;
  }
  // the HTML preview of one layer, at `scale` (displayed px per picture px)
  function style(L, scale) {
    const px = L.size * scale;
    const st = {
      left: L.x * 100 + "%", top: L.y * 100 + "%", font: fontCss(L, px), color: L.colour, textAlign: L.align,
      lineHeight: String(L.lineHeight), letterSpacing: (L.spacing || 0) * px / 100 + "px", opacity: String(L.opacity),
      transform: `translate(-50%, -50%) rotate(${L.rotate || 0}deg)`,
    };
    if (L.stroke > 0) { st.webkitTextStroke = `${L.stroke * px / 100 * 2}px ${L.strokeColour}`; st.paintOrder = "stroke fill"; }
    if (L.shadow > 0) st.textShadow = `${L.shadow * px / 100 * 0.6}px ${L.shadow * px / 100 * 0.8}px ${L.shadow * px / 100 * 1.2}px rgba(0,0,0,.75)`;
    return st;
  }
  // draw every layer onto ctx (the picture already on it), w × h = the picture's size
  function draw(ctx, layers, w, h) {
    for (const L of layers) {
      if (!String(L.text || "").trim()) continue;
      const px = L.size, lines = String(L.text).split("\n"), lh = px * L.lineHeight;
      ctx.save();
      ctx.globalAlpha = L.opacity;
      ctx.translate(L.x * w, L.y * h);
      ctx.rotate(((L.rotate || 0) * Math.PI) / 180);
      ctx.font = fontCss(L, px);
      if ("letterSpacing" in ctx) ctx.letterSpacing = (L.spacing || 0) * px / 100 + "px";
      ctx.textBaseline = "middle";
      ctx.textAlign = L.align;
      const bw = Math.max(...lines.map((t) => ctx.measureText(t).width));
      const ax = L.align === "left" ? -bw / 2 : L.align === "right" ? bw / 2 : 0;
      lines.forEach((t, i) => {
        const y = (i - (lines.length - 1) / 2) * lh;
        if (L.shadow > 0) {
          ctx.save();
          const s = L.shadow * px / 100;
          ctx.shadowColor = "rgba(0,0,0,.75)"; ctx.shadowOffsetX = s * 0.6; ctx.shadowOffsetY = s * 0.8; ctx.shadowBlur = s * 1.2;
          ctx.fillStyle = L.colour; ctx.fillText(t, ax, y);
          ctx.restore();
        }
        if (L.stroke > 0) {
          ctx.lineJoin = "round"; ctx.lineWidth = L.stroke * px / 100 * 2; ctx.strokeStyle = L.strokeColour;  // half sits under the fill (as in the preview)
          ctx.strokeText(t, ax, y);
        }
        ctx.fillStyle = L.colour;
        ctx.fillText(t, ax, y);
      });
      ctx.restore();
    }
  }
  const api = { FONTS, layer, style, draw };
  if (typeof window !== "undefined") window.ProtoText = api;
  if (typeof module !== "undefined") module.exports = api;
})();
