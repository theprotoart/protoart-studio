// 🔤 Replace text: draw text onto a picture like a paint program's Type tool. The layers are previewed as HTML
// over the picture and drawn onto a canvas at the picture's full size to save, with the same maths for both.
(function () {
  // the comic lettering pack that comes with the app (assets/fonts, SIL Open Font License / Apache 2.0)
  const PACK = [
    ["Comic Neue", "'Comic Neue', 'Comic Sans MS', cursive", "dialogue: clean, like hand lettering"],
    ["Shantell Sans", "'Shantell Sans', 'Comic Neue', cursive", "dialogue: lively, a little bouncy"],
    ["Patrick Hand", "'Patrick Hand', 'Comic Neue', cursive", "dialogue: hand-lettered"],
    ["Architects Daughter", "'Architects Daughter', 'Comic Neue', cursive", "captions and notes: all caps feel"],
    ["Bangers", "'Bangers', Impact, sans-serif", "sound effects and titles: POW!"],
    ["Luckiest Guy", "'Luckiest Guy', Impact, sans-serif", "shouts and titles: big and round"],
    ["Bowlby One SC", "'Bowlby One SC', Impact, sans-serif", "heavy sound effects"],
    ["Titan One", "'Titan One', Impact, sans-serif", "bold titles"],
    ["Chewy", "'Chewy', 'Comic Neue', cursive", "cartoon and kids' titles"],
    ["Boogaloo", "'Boogaloo', 'Comic Neue', cursive", "retro cartoon"],
    ["Bubblegum Sans", "'Bubblegum Sans', 'Comic Neue', cursive", "soft, friendly"],
    ["Permanent Marker", "'Permanent Marker', 'Marker Felt', cursive", "marker, signs and notes"],
  ];
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
  // a font from the pack, the Mac list, or any other font installed on this Mac (by its family name)
  const stack = (name) => {
    const f = PACK.find((x) => x[0] === name) || FONTS.find((x) => x[0] === name);
    return f ? f[1] : `"${String(name || "Helvetica Neue").replace(/["\\]/g, "")}", Helvetica Neue, sans-serif`;
  };

  function layer(x, y, size) {
    return { text: "Your text", x, y, size: Math.round(size), font: "Helvetica Neue", colour: "#ffffff", bold: true, italic: false,
             align: "center", vertical: false, stroke: 0, strokeColour: "#000000", shadow: 0, rotate: 0, opacity: 1, spacing: 0, lineHeight: 1.15 };
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
    if (L.vertical) {  // letters stacked top to bottom: columns (the first on the right), one letter per row, as draw() does
      Object.assign(st, { display: "flex", flexDirection: "row-reverse", alignItems: { left: "flex-start", right: "flex-end" }[L.align] || "center",
                          letterSpacing: "0px", lineHeight: (1 + (L.spacing || 0) / 100) * px + "px", textAlign: "center" });
    }
    if (L.stroke > 0) { st.webkitTextStroke = `${L.stroke * px / 100 * 2}px ${L.strokeColour}`; st.paintOrder = "stroke fill"; }
    if (L.shadow > 0) st.textShadow = `${L.shadow * px / 100 * 0.6}px ${L.shadow * px / 100 * 0.8}px ${L.shadow * px / 100 * 1.2}px rgba(0,0,0,.75)`;
    return st;
  }
  // draw every layer onto ctx (the picture already on it), w × h = the picture's size
  function draw(ctx, layers, w, h) {
    for (const S of layers) if (S.type === "shape") drawShape(ctx, S, w, h);  // balloons and shapes under the words
    for (const L of layers) {
      if (L.type === "shape" || !String(L.text || "").trim()) continue;
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
      // ↕ vertical: each line is a column (the first on the right), its letters stacked upright, one em apart
      const spots = [];
      if (L.vertical) {
        if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";
        ctx.textAlign = "center";
        const step = px * (1 + (L.spacing || 0) / 100), cols = lines.map((t) => [...t]);
        const H = Math.max(...cols.map((c) => c.length)) * step;
        cols.forEach((chars, i) => {
          const x = ((lines.length - 1) / 2 - i) * lh, n = chars.length;
          chars.forEach((ch, j) => {
            const y = L.align === "left" ? -H / 2 + (j + 0.5) * step : L.align === "right" ? H / 2 - (n - j - 0.5) * step : (j - (n - 1) / 2) * step;
            spots.push([ch, x, y]);
          });
        });
      } else lines.forEach((t, i) => spots.push([t, ax, (i - (lines.length - 1) / 2) * lh]));
      spots.forEach(([t, ax, y]) => {
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
  // ------------------------------------------------------------------ 💬 balloons and shapes
  // A shape: centre (x, y) and size (w, h) as parts of the picture, a tail tip (tx, ty) for balloons, fill and outline.
  const SHAPES = [
    ["speech", "💬 Speech", "A speech balloon with a tail"], ["thought", "💭 Thought", "A cloud with little bubbles towards the speaker"],
    ["shout", "🗯 Shout", "A jagged balloon for shouting"], ["whisper", "🤫 Whisper", "A dashed balloon for whispering"],
    ["box", "▢ Box", "A rounded box balloon with a tail"], ["caption", "▭ Caption", "A narration box (no tail)"],
    ["burst", "💥 Burst", "A spiky explosion for sound effects (no tail)"],
  ];
  const TAIL = { speech: 1, thought: 1, shout: 1, whisper: 1, box: 1 };
  function shape(kind, x, y, nw, nh) {
    const m = Math.min(nw, nh);
    const w = kind === "caption" ? 0.36 : kind === "burst" ? 0.34 : 0.32, h = kind === "caption" ? 0.1 : kind === "burst" ? 0.3 : 0.2;
    return { type: "shape", kind, x, y, w: w * m / nw, h: h * m / nh, tx: Math.max(0.02, x - 0.08), ty: Math.min(0.98, y + 0.2),
             tail: !!TAIL[kind], fill: kind === "caption" ? "#fff6c8" : kind === "burst" ? "#ffd23f" : "#ffffff",
             line: "#111111", lw: Math.max(2, Math.round(m * 0.005)), opacity: 1 };
  }
  const f1 = (v) => Math.round(v * 10) / 10;
  function ellipse(cx, cy, rx, ry) {
    return `M${f1(cx - rx)},${f1(cy)}A${f1(rx)},${f1(ry)} 0 1,0 ${f1(cx + rx)},${f1(cy)}A${f1(rx)},${f1(ry)} 0 1,0 ${f1(cx - rx)},${f1(cy)}Z`;
  }
  function poly(pts) { return "M" + pts.map((p) => f1(p[0]) + "," + f1(p[1])).join("L") + "Z"; }
  function roundRect(x0, y0, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    return `M${f1(x0 + r)},${f1(y0)}H${f1(x0 + w - r)}A${f1(r)},${f1(r)} 0 0,1 ${f1(x0 + w)},${f1(y0 + r)}V${f1(y0 + h - r)}` +
      `A${f1(r)},${f1(r)} 0 0,1 ${f1(x0 + w - r)},${f1(y0 + h)}H${f1(x0 + r)}A${f1(r)},${f1(r)} 0 0,1 ${f1(x0)},${f1(y0 + h - r)}` +
      `V${f1(y0 + r)}A${f1(r)},${f1(r)} 0 0,1 ${f1(x0 + r)},${f1(y0)}Z`;
  }
  // the shape's outline(s) in picture pixels: { parts: [svg path…], dash }
  function shapePaths(S, w, h) {
    const cx = S.x * w, cy = S.y * h, rx = S.w * w / 2, ry = S.h * h / 2, tx = S.tx * w, ty = S.ty * h;
    const parts = [];
    const ang = Math.atan2(ty - cy, tx - cx), dx = Math.cos(ang), dy = Math.sin(ang);
    const tailTri = (width) => {  // from inside the balloon to the tip
      const bw = width * Math.min(rx, ry), px = -dy, py = dx, bx = cx + dx * rx * 0.55, by = cy + dy * ry * 0.55;
      return poly([[bx + px * bw, by + py * bw], [tx, ty], [bx - px * bw, by - py * bw]]);
    };
    if (S.kind === "speech" || S.kind === "whisper") {
      if (S.tail) parts.push(tailTri(0.22));
      parts.push(ellipse(cx, cy, rx, ry));
    } else if (S.kind === "box") {
      if (S.tail) parts.push(tailTri(0.2));
      parts.push(roundRect(cx - rx, cy - ry, rx * 2, ry * 2, Math.min(rx, ry) * 0.35));
    } else if (S.kind === "caption") {
      parts.push(roundRect(cx - rx, cy - ry, rx * 2, ry * 2, Math.min(rx, ry) * 0.06));
    } else if (S.kind === "shout" || S.kind === "burst") {
      const n = S.kind === "burst" ? 18 : 24, inner = S.kind === "burst" ? 0.62 : 0.84, pts = [];
      for (let i = 0; i < n * 2; i++) {
        const a = (i / (n * 2)) * Math.PI * 2, k = i % 2 ? inner : 1, j = S.kind === "burst" ? (i % 4 === 0 ? 1.08 : 1) : 1;
        pts.push([cx + Math.cos(a) * rx * k * j, cy + Math.sin(a) * ry * k * j]);
      }
      if (S.tail && S.kind === "shout") parts.push(tailTri(0.2));
      parts.push(poly(pts));
    } else if (S.kind === "thought") {
      const n = 11, pts = [];
      for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; pts.push([cx + Math.cos(a) * rx * 0.86, cy + Math.sin(a) * ry * 0.86]); }
      let d = `M${f1(pts[0][0])},${f1(pts[0][1])}`;
      for (let i = 0; i < n; i++) {
        const b = pts[(i + 1) % n], r = Math.hypot(b[0] - pts[i][0], b[1] - pts[i][1]) * 0.62;
        d += `A${f1(r)},${f1(r)} 0 0,1 ${f1(b[0])},${f1(b[1])}`;
      }
      parts.push(d + "Z");
      if (S.tail) {  // three bubbles getting smaller towards the speaker
        const ex = cx + dx * rx, ey = cy + dy * ry, m = Math.min(rx, ry);
        [0.2, 0.55, 0.88].forEach((t, i) => {
          const r = m * [0.17, 0.11, 0.07][i];
          parts.push(ellipse(ex + (tx - ex) * t, ey + (ty - ey) * t, r, r));
        });
      }
    }
    return { parts, dash: S.kind === "whisper" };
  }
  function drawShape(ctx, S, w, h) {
    const { parts, dash } = shapePaths(S, w, h);
    ctx.save();
    ctx.globalAlpha = S.opacity ?? 1;
    ctx.lineJoin = "round";
    if (S.lw > 0) {  // outlines first, then the fills over them: the tail joins the balloon without a line across
      ctx.lineWidth = S.lw * 2; ctx.strokeStyle = S.line;
      if (dash) ctx.setLineDash([S.lw * 3, S.lw * 2.2]);
      for (const d of parts) ctx.stroke(new Path2D(d));
      ctx.setLineDash([]);
    }
    ctx.fillStyle = S.fill;
    for (const d of parts) ctx.fill(new Path2D(d));
    ctx.restore();
  }
  // the HTML preview's columns for vertical text: [[letters of line 1], …] and each column's width (px on screen)
  function columns(L, scale) { return { cols: String(L.text || " ").split("\n").map((t) => [...t]), w: L.size * scale * L.lineHeight }; }
  const api = { FONTS, PACK, SHAPES, stack, layer, style, draw, columns, shape, shapePaths, drawShape };
  if (typeof window !== "undefined") window.ProtoText = api;
  if (typeof module !== "undefined") module.exports = api;
})();
