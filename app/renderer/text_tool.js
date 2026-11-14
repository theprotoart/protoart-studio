// 🔤 Replace text: draw text onto a picture like a paint program's Type tool. The layers are previewed as HTML
// over the picture and drawn onto a canvas at the picture's full size to save, with the same maths for both.
(function () {
  // the comic lettering pack that comes with the app (assets/fonts, SIL Open Font License / Apache 2.0)
  const PACK = [
    // Protoart Studio's own lettering (packaging/fonts/build_comic_fonts.py)
    ["Protoart Speech", "'Protoart Speech', 'Comic Neue', cursive", "Protoart original: dialogue lettering, in capitals"],
    ["Protoart Kaboom", "'Protoart Kaboom', Impact, sans-serif", "Protoart original: chunky sound effects, KA-BOOM!"],
    ["Protoart Chatter", "'Protoart Chatter', 'Comic Neue', cursive", "Protoart original: bouncy cartoon lettering"],
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
             align: "center", vertical: false, stroke: 0, strokeColour: "#000000", shadow: 0, rotate: 0, opacity: 1, spacing: 0, lineHeight: 1.15,
             ...EFFECTS };
  }
  // the layer styles (like Photoshop's), all off to start with; old saved texts get them from here too
  const EFFECTS = {
    caps: false,                                         // ALL CAPITALS
    stroke2: 0, stroke2Colour: "#ffffff",                // a second outline, outside the first (comic sound effects)
    gradient: false, colour2: "#ff8a00", gradAngle: 90,  // a two-colour fill (90 = top to bottom)
    bevel: 0, bevelStyle: "raised", bevelAngle: 135,     // bevel & emboss: depth, raised or pressed, where the light is
    bevelSoft: 0, bevelHi: "#ffffff", bevelLo: "#000000",
    glow: 0, glowColour: "#ffe45c",                      // outer glow
    inner: 0, innerColour: "#000000",                    // inner shadow
    shadowColour: "#000000", shadowAngle: 53, shadowSoft: 1.2,  // drop shadow: colour, direction (degrees, 90 = straight down), softness
    curve: 0,                                            // arc: bend the line(s) up (+) or down (−), in degrees of the circle
    skewX: 0, skewY: 0,                                  // slant sideways / up and down, in degrees
    warp: "none", warpAmt: 50,                           // warp the whole text: "bottle" (round a can or bottle), "bulge", "wave", "rise"
    perspH: 0, perspV: 0,                                // perspective: turned away left/right (−/+) and leaning back/forward, −100…100
    quad: null,                                          // free transform: how far each corner (TL, TR, BR, BL) was dragged, in text sizes
  };
  const WARPS = [
    { id: "none", label: "None" },
    { id: "bottle", label: "🍾 Wrap round", tip: "Wrap the words round something round in the picture, like a label on a bottle or can: they narrow towards the sides and the lines curve (only the words bend, not the picture). + seen from above, − from below" },
    { id: "bulge", label: "🎈 Bulge", tip: "Swell the middle (+) or pinch it (−)" },
    { id: "wave", label: "🌊 Wave", tip: "Ripple the text up and down" },
    { id: "rise", label: "📈 Rise", tip: "Lift one end: + rises to the right, − to the left (the letters stay upright)" },
  ];
  function withEffects(L) { for (const k in EFFECTS) if (L[k] === undefined) L[k] = EFFECTS[k]; return L; }
  function layerText(L) { return L.caps ? String(L.text || "").toUpperCase() : String(L.text || ""); }
  function hexA(hex, a) {  // #rrggbb -> rgba()
    const h = String(hex || "#000000").replace("#", "");
    return `rgba(${parseInt(h.slice(0, 2), 16) || 0},${parseInt(h.slice(2, 4), 16) || 0},${parseInt(h.slice(4, 6), 16) || 0},${a})`;
  }
  function fontCss(L, px) {
    return `${L.italic ? "italic " : ""}${L.bold ? "700 " : "400 "}${px}px ${stack(L.font)}`;
  }
  // the HTML box of one layer, at `scale` (displayed px per picture px): the same size and place as the text, used to
  // pick and drag it; what you see is drawn on a canvas over it by draw(), so the preview and the saved picture match
  function style(L, scale) {
    const px = L.size * scale;
    const st = {
      left: L.x * 100 + "%", top: L.y * 100 + "%", font: fontCss(L, px), color: "transparent", textAlign: L.align,
      lineHeight: String(L.lineHeight), letterSpacing: (L.spacing || 0) * px / 100 + "px",
      transform: `translate(-50%, -50%) rotate(${L.rotate || 0}deg) skew(${L.skewX || 0}deg, ${L.skewY || 0}deg)`, textTransform: L.caps ? "uppercase" : "none",
    };
    if (L.vertical) {  // letters stacked top to bottom: columns (the first on the right), one letter per row, as draw() does
      Object.assign(st, { display: "flex", flexDirection: "row-reverse", alignItems: { left: "flex-start", right: "flex-end" }[L.align] || "center",
                          letterSpacing: "0px", lineHeight: (1 + (L.spacing || 0) / 100) * px + "px", textAlign: "center" });
    }
    if (L.curve) st.paddingBottom = Math.min(Math.abs(L.curve), 180) / 180 * px * 1.2 + "px";  // room for the bend
    return st;
  }
  // where each piece of text goes, in the layer's own frame: [text, x, y, turn] (a line, a letter of vertical text, or
  // a letter on an arc)
  function spotsOf(ctx, L, px, lines, lh) {
    const spots = [];
    const bw = Math.max(...lines.map((t) => ctx.measureText(t).width));
    const ax = L.align === "left" ? -bw / 2 : L.align === "right" ? bw / 2 : 0;
    if (L.vertical) {  // each line is a column (the first on the right), its letters stacked upright, one em apart
      if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";
      ctx.textAlign = "center";
      const step = px * (1 + (L.spacing || 0) / 100), cols = lines.map((t) => [...t]);
      const H = Math.max(...cols.map((c) => c.length)) * step;
      cols.forEach((chars, i) => {
        const x = ((lines.length - 1) / 2 - i) * lh, n = chars.length;
        chars.forEach((ch, j) => {
          const y = L.align === "left" ? -H / 2 + (j + 0.5) * step : L.align === "right" ? H / 2 - (n - j - 0.5) * step : (j - (n - 1) / 2) * step;
          spots.push([ch, x, y, 0]);
        });
      });
      return { spots, bw: lines.length * lh, bh: H };
    }
    const curve = Math.max(-300, Math.min(300, +L.curve || 0));
    if (Math.abs(curve) >= 1) {  // ⌒ an arc: the letters follow a circle, each turned to it
      ctx.textAlign = "center";
      const sp = (L.spacing || 0) * px / 100;
      lines.forEach((t, li) => {
        const chars = [...t], widths = chars.map((c) => ctx.measureText(c).width + sp), total = widths.reduce((a, b) => a + b, 0);
        const R = total / (Math.abs(curve) * Math.PI / 180), sign = curve > 0 ? 1 : -1;
        const y0 = (li - (lines.length - 1) / 2) * lh;
        let s0 = -total / 2;
        chars.forEach((c, k) => {
          const a = (s0 + widths[k] / 2) / R;
          spots.push([c, R * Math.sin(a), y0 + sign * R * (1 - Math.cos(a)), sign * a]);
          s0 += widths[k];
        });
      });
      return { spots, bw, bh: lines.length * lh + px };
    }
    lines.forEach((t, i) => spots.push([t, ax, (i - (lines.length - 1) / 2) * lh, 0]));
    return { spots, bw, bh: lines.length * lh };
  }
  function each(ctx, spots, fn) {  // fn(text, x, y) for every spot, turned where needed
    for (const [t, x, y, r] of spots) {
      if (!r) { fn(t, x, y); continue; }
      ctx.save(); ctx.translate(x, y); ctx.rotate(r); fn(t, 0, 0); ctx.restore();
    }
  }
  function fillStyleOf(ctx, L, bw, bh) {
    if (!L.gradient) return L.colour;
    const a = ((+L.gradAngle || 90) * Math.PI) / 180, r = Math.abs(Math.cos(a)) * bw / 2 + Math.abs(Math.sin(a)) * bh / 2;
    const g = ctx.createLinearGradient(-Math.cos(a) * r, -Math.sin(a) * r, Math.cos(a) * r, Math.sin(a) * r);
    g.addColorStop(0, L.colour); g.addColorStop(1, L.colour2 || L.colour);
    return g;
  }
  // draw every layer onto ctx (the picture already on it), w × h = the picture's size
  function draw(ctx, layers, w, h) {
    for (const S of layers) if (S.type === "shape") drawShape(ctx, S, w, h);  // balloons and shapes under the words
    for (const L0 of layers) {
      if (L0.type === "shape" || !String(L0.text || "").trim()) continue;
      const L = withEffects({ ...L0 });
      const px = L.size, lines = layerText(L).split("\n"), lh = px * L.lineHeight;
      const setup = (c) => {
        c.font = fontCss(L, px);
        if ("letterSpacing" in c) c.letterSpacing = (L.spacing || 0) * px / 100 + "px";
        c.textBaseline = "middle";
        c.textAlign = L.align;
      };
      ctx.save();
      ctx.globalAlpha = L.opacity;
      ctx.translate(L.x * w, L.y * h);
      ctx.rotate(((L.rotate || 0) * Math.PI) / 180);
      const sx = Math.tan(((Math.max(-70, Math.min(70, +L.skewX || 0))) * Math.PI) / 180);
      const sy = Math.tan(((Math.max(-70, Math.min(70, +L.skewY || 0))) * Math.PI) / 180);
      if (sx || sy) ctx.transform(1, sy, sx, 1, 0, 0);  // the same as CSS skew(skewX, skewY)
      setup(ctx);
      const { spots, bw, bh } = spotsOf(ctx, L, px, lines, lh);
      const u = px / 100;  // effect sizes are in hundredths of the text size, so they scale with it
      const amt = Math.max(-100, Math.min(100, +L.warpAmt || 0));
      const warp = L.warp && L.warp !== "none" && amt;
      const Q = quadOf(L, bw / 2, bh / 2), distort = !!Q;
      if (warp || distort) {
        // drawn flat on its own canvas first (with every effect), then bent and/or put in perspective onto the picture
        const pad = Math.ceil(((L.stroke || 0) + (L.stroke2 || 0)) * u * 2 + (L.glow || 0) * u * 3 + (L.shadow || 0) * u * 2.5 + px * 0.3 + 6);
        const HW = Math.ceil(bw / 2 + pad), HH = Math.ceil(bh / 2 + pad);
        const canvas = (w2, h2) => { const c = document.createElement("canvas"); c.width = w2 * 2; c.height = h2 * 2; const x = c.getContext("2d"); x.translate(w2, h2); return [c, x]; };
        const [c, cx] = canvas(HW, HH);
        setup(cx);
        layerBody(cx, L, spots, setup, bw, bh, u);
        if (!distort) warpOnto(ctx, c, HW, HH, bw / 2, bh / 2, L.warp, amt / 100);
        else {
          let src = c, sw = HW, shh = HH;
          if (warp) {  // bend first, on a taller canvas (a wave or a bulge reaches above and below), then the perspective
            const [c2, x2] = canvas(HW, HH * 2);
            warpOnto(x2, c, HW, HH, bw / 2, bh / 2, L.warp, amt / 100);
            src = c2; shh = HH * 2;
          }
          perspOnto(ctx, src, sw, shh, Q, bw / 2, bh / 2);
        }
      } else layerBody(ctx, L, spots, setup, bw, bh, u);
      ctx.restore();
    }
  }
  // one text with its effects, drawn round (0, 0) of ctx
  function layerBody(ctx, L, spots, setup, bw, bh, u) {
    const fill = fillStyleOf(ctx, L, bw, bh);
    const outer = (L.stroke > 0 ? L.stroke * u * 2 : 0) + (L.stroke2 > 0 ? L.stroke2 * u * 2 : 0);
    if (L.glow > 0) {  // ✨ outer glow
      ctx.save();
      ctx.shadowColor = L.glowColour; ctx.shadowBlur = L.glow * u * 2;
      ctx.fillStyle = L.glowColour; ctx.strokeStyle = L.glowColour; ctx.lineJoin = "round"; ctx.lineWidth = Math.max(1, outer);
      for (let k = 0; k < 2; k++) each(ctx, spots, (t, x, y) => { if (outer) ctx.strokeText(t, x, y); ctx.fillText(t, x, y); });
      ctx.restore();
    }
    if (L.shadow > 0) {  // drop shadow: its colour, direction and softness
      ctx.save();
      const d = L.shadow * u, a = ((+L.shadowAngle || 53) * Math.PI) / 180;
      ctx.shadowColor = hexA(L.shadowColour, 0.75); ctx.shadowOffsetX = Math.cos(a) * d; ctx.shadowOffsetY = Math.sin(a) * d;
      ctx.shadowBlur = d * (L.shadowSoft ?? 1.2);
      ctx.fillStyle = L.colour; ctx.strokeStyle = L.colour; ctx.lineJoin = "round"; ctx.lineWidth = Math.max(1, outer);
      each(ctx, spots, (t, x, y) => { if (outer) ctx.strokeText(t, x, y); ctx.fillText(t, x, y); });
      ctx.restore();
    }
    ctx.lineJoin = "round";
    if (L.stroke2 > 0) {  // the second (outer) outline
      ctx.lineWidth = outer; ctx.strokeStyle = L.stroke2Colour;
      each(ctx, spots, (t, x, y) => ctx.strokeText(t, x, y));
    }
    if (L.stroke > 0) {
      ctx.lineWidth = L.stroke * u * 2; ctx.strokeStyle = L.strokeColour;  // half sits under the fill
      each(ctx, spots, (t, x, y) => ctx.strokeText(t, x, y));
    }
    ctx.fillStyle = fill;
    each(ctx, spots, (t, x, y) => ctx.fillText(t, x, y));
    if (L.bevel > 0 || L.inner > 0) insideEffects(ctx, L, spots, setup, bw, bh, u);
  }
  // the four corners (TL, TR, BR, BL) of the text's box (half sizes hw, hh) after 📐 perspective and ✥ free transform, in
  // the layer's own frame (before slant and turn); null when the box is a plain rectangle
  function quadOf(L, hw, hh) {
    const ph = Math.max(-100, Math.min(100, +L.perspH || 0)) / 100, pv = Math.max(-100, Math.min(100, +L.perspV || 0)) / 100;
    const q = Array.isArray(L.quad) && L.quad.length === 4 ? L.quad : null;
    const moved = q && q.some((d) => d && (Math.abs(+d[0] || 0) > 1e-4 || Math.abs(+d[1] || 0) > 1e-4));
    if (!ph && !pv && !moved) return null;
    const a = ph * 0.6, b = pv * 0.6, size = +L.size || 40;
    // the perspective of the sliders is set on a frame a little bigger than the words (so it stays gentle), then the box's
    // corners are taken from it; the dragged corners are added on top
    const ew = hw + size * 0.5, eh = hh + size * 0.5;
    const E = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => [sx * ew * (1 + sy * b), sy * eh * (1 + sx * a)]);  // far side smaller
    const H = squareToQuad(E);
    const at = (x, y) => { const u = (x + ew) / (2 * ew), v = (y + eh) / (2 * eh), d = H[6] * u + H[7] * v + 1;
      return [(H[0] * u + H[1] * v + H[2]) / d, (H[3] * u + H[4] * v + H[5]) / d]; };
    return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy], i) => {
      const p = at(sx * hw, sy * hh), d = moved ? q[i] || [0, 0] : [0, 0];
      return [p[0] + (+d[0] || 0) * size, p[1] + (+d[1] || 0) * size];
    });
  }
  // 📐 perspective / ✥ free transform (like Photoshop's Transform → Perspective and Distort): the flat picture src (its
  // middle at HW, HH) drawn so the text's box (half sizes hw, hh) lands on the four corners Q round (0, 0) of ctx, with
  // true foreshortening
  function perspOnto(ctx, src, HW, HH, Q, hw, hh) {
    const H = squareToQuad(Q);
    const map = (x, y) => {  // a point of src (its pixels) -> ctx; the box is the unit square, the margin round it extends it
      const u = (x - HW + hw) / (2 * hw), v = (y - HH + hh) / (2 * hh);
      const d = H[6] * u + H[7] * v + 1;
      return d > 0.05 ? [(H[0] * u + H[1] * v + H[2]) / d, (H[3] * u + H[4] * v + H[5]) / d] : null;  // past the horizon: skipped
    };
    // a grid of small cells, each drawn as a straight (affine) piece and overlapping its neighbours by a pixel: with
    // cells this small the error is invisible, and nothing is clipped, so no hairline seams
    const W = HW * 2, Hh = HH * 2, N = Math.max(8, Math.min(48, Math.ceil(Math.max(W, Hh) / 24)));
    const cw = W / N, chh = Hh / N;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const d00 = map(i * cw, j * chh), d10 = map((i + 1) * cw, j * chh), d01 = map(i * cw, (j + 1) * chh);
      if (!d00 || !d10 || !d01) continue;
      ctx.save();
      ctx.transform((d10[0] - d00[0]) / cw, (d10[1] - d00[1]) / cw, (d01[0] - d00[0]) / chh, (d01[1] - d00[1]) / chh, d00[0], d00[1]);
      const sx = i * cw, sy = j * chh;
      const ew = Math.min(cw + 1.2, W - sx), eh = Math.min(chh + 1.2, Hh - sy);  // a little over into the next cell
      ctx.drawImage(src, sx, sy, ew, eh, 0, 0, ew, eh);
      ctx.restore();
    }
  }
  function squareToQuad(q) {  // the projective map from the unit square to the four corners q (TL, TR, BR, BL), Heckbert's way
    const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
    const sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;
    if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) return [x1 - x0, x3 - x0, x0, y1 - y0, y3 - y0, y0, 0, 0];
    const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2, den = dx1 * dy2 - dx2 * dy1;
    const g = (sx * dy2 - dx2 * sy) / den, h = (dx1 * sy - sx * dy1) / den;
    return [x1 - x0 + g * x1, x3 - x0 + h * x3, x0, y1 - y0 + g * y1, y3 - y0 + h * y3, y0, g, h];
  }
  // bend the flat text (canvas src, its middle at HW, HH; the words reach ±tw, ±th from it) onto ctx round (0, 0),
  // one thin column at a time. k = −1…1
  function warpOnto(ctx, src, HW, HH, tw, th, kind, k) {
    const step = 2, H = HH * 2;
    for (let x0 = 0; x0 < HW * 2; x0 += step) {
      const u = x0 + step / 2 - HW;          // the column's middle, from the text's middle
      const t = Math.max(-1, Math.min(1, u / Math.max(1, tw)));  // −1 … 1 across the words
      let dx = u, dw = step, dy = 0, sh = 1;
      if (kind === "bottle") {  // round a cylinder: the sides turn away (narrower) and the lines curve with the view
        const span = Math.max(0.1, (Math.abs(k) * 160 * Math.PI) / 180);  // how far round the words reach (both sides)
        const R = Math.max(1, tw) / (span / 2);                              // the label's distance along = R × angle
        const ang = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, u / R));
        dx = R * Math.sin(ang); dw = step * Math.cos(ang);
        dy = -Math.sign(k) * R * (1 - Math.cos(ang)) * 0.35;                // seen a little from above (+) or below (−)
      } else if (kind === "bulge") {  // the middle taller (+) or shorter (−)
        sh = 1 + k * 0.85 * Math.cos((Math.min(1, Math.abs(t)) * Math.PI) / 2);
      } else if (kind === "wave") {
        dy = k * th * 0.9 * Math.sin(t * Math.PI);
      } else if (kind === "rise") {
        dy = -k * th * 1.2 * t;
      }
      if (dw <= 0.05) continue;
      ctx.drawImage(src, x0, 0, step, H, dx - dw / 2 - 0.25, dy - HH * sh, dw + 0.5, H * sh);
    }
  }
  // 🔆 bevel & emboss and inner shadow: worked out on a small canvas around the layer from the letters' own shape, then
  // laid over the letters (only inside them)
  function insideEffects(ctx, L, spots, setup, bw, bh, u) {
    const M = Math.ceil(Math.hypot(bw, bh) / 2 + L.size * 0.6 + 8), S = M * 2;
    const make = () => { const c = document.createElement("canvas"); c.width = S; c.height = S; const x = c.getContext("2d"); x.translate(M, M); setup(x); return [c, x]; };
    const [mask, mx] = make();  // the letters, solid
    mx.fillStyle = "#000";
    each(mx, spots, (t, x, y) => mx.fillText(t, x, y));
    const shifted = (dx, dy, blur) => {  // the letters' shape moved by (dx, dy), softened
      const [c, x] = make();
      x.setTransform(1, 0, 0, 1, 0, 0);
      if (blur) x.filter = `blur(${blur}px)`;
      x.drawImage(mask, dx, dy);
      return c;
    };
    const edge = (dx, dy, colour, alpha, soft) => {  // inside the letters, where the moved copy doesn't reach
      const [c, x] = make();
      x.setTransform(1, 0, 0, 1, 0, 0);
      x.fillStyle = colour; x.fillRect(0, 0, S, S);
      x.globalCompositeOperation = "destination-out";
      x.drawImage(shifted(dx, dy, soft), 0, 0);
      x.globalCompositeOperation = "destination-in";
      x.drawImage(mask, 0, 0);
      ctx.save();
      ctx.globalAlpha *= alpha;
      ctx.drawImage(c, -M, -M);
      ctx.restore();
    };
    if (L.inner > 0) {  // inner shadow: from the top-left, like light falling into the letters
      const d = L.inner * u;
      edge(d * 0.7, d * 0.7, L.innerColour, 0.75, d * 0.8);
    }
    if (L.bevel > 0) {
      const d = Math.max(1, L.bevel * u), a = ((+L.bevelAngle || 135) * Math.PI) / 180;
      const lx = -Math.cos(a), ly = -Math.sin(a);  // towards the light (135° = from the top left)
      const soft = (L.bevelSoft || 0) * u;
      const raised = L.bevelStyle !== "pressed";
      // the edges facing the light catch it (raised) or fall into shadow (pressed); the opposite edges the other way
      edge(-lx * d, -ly * d, raised ? L.bevelHi : L.bevelLo, raised ? 0.8 : 0.7, soft);
      edge(lx * d, ly * d, raised ? L.bevelLo : L.bevelHi, raised ? 0.7 : 0.8, soft);
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
             line: "#111111", lw: Math.max(2, Math.round(m * 0.005)), opacity: 1, rot: 0 };
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
    const cx = S.x * w, cy = S.y * h, rx = S.w * w / 2, ry = S.h * h / 2;
    // a turned shape (S.rot degrees) is drawn turned; its tail tip stays where you put it, so it's worked out here in
    // the shape's own, unturned frame
    const rot = ((S.rot || 0) * Math.PI) / 180;
    let tx = S.tx * w, ty = S.ty * h;
    if (rot) {
      const dx0 = tx - cx, dy0 = ty - cy, c = Math.cos(-rot), s0 = Math.sin(-rot);
      tx = cx + dx0 * c - dy0 * s0; ty = cy + dx0 * s0 + dy0 * c;
    }
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
    return { parts, dash: S.kind === "whisper", rot: S.rot || 0, cx, cy };
  }
  function drawShape(ctx, S, w, h) {
    const { parts, dash, rot, cx, cy } = shapePaths(S, w, h);
    ctx.save();
    if (rot) { ctx.translate(cx, cy); ctx.rotate((rot * Math.PI) / 180); ctx.translate(-cx, -cy); }
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
  // where a text's corners and edge middles are on the picture (picture pixels), for the handles you drag: its box after
  // perspective / free transform, slant and turn (warps are left out). Also the box's half sizes and the slant factors.
  let _measure = null;
  function geometry(L0, w, h) {
    const L = withEffects({ ...L0 });
    if (!_measure) _measure = document.createElement("canvas").getContext("2d");
    const px = L.size, lines = layerText(L).split("\n"), lh = px * L.lineHeight, c = _measure;
    c.font = fontCss(L, px);
    if ("letterSpacing" in c) c.letterSpacing = (L.spacing || 0) * px / 100 + "px";
    c.textAlign = L.align; c.textBaseline = "middle";
    const { bw, bh } = spotsOf(c, L, px, lines, lh);
    const hw = Math.max(4, bw / 2), hh = Math.max(4, bh / 2);
    const Q = quadOf(L, hw, hh) || [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
    const sx = Math.tan(((Math.max(-70, Math.min(70, +L.skewX || 0))) * Math.PI) / 180);
    const sy = Math.tan(((Math.max(-70, Math.min(70, +L.skewY || 0))) * Math.PI) / 180);
    const r = ((L.rotate || 0) * Math.PI) / 180, cs = Math.cos(r), sn = Math.sin(r), cx = L.x * w, cy = L.y * h;
    const out = ([x, y]) => { const X = x + sx * y, Y = sy * x + y; return [cx + X * cs - Y * sn, cy + X * sn + Y * cs]; };
    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    return { corners: Q.map(out), top: out(mid(Q[0], Q[1])), right: out(mid(Q[1], Q[2])), bottom: out(mid(Q[2], Q[3])),
             left: out(mid(Q[3], Q[0])), hw, hh, sx, sy, Q, base: quadOf({ ...L, quad: null }, hw, hh) || [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]] };
  }
  // a point on the picture (pixels) in the text's own frame (before slant and turn): for dragging its handles
  function toLocal(L, w, h, px, py, sx, sy) {
    const r = (-(L.rotate || 0) * Math.PI) / 180, dx = px - L.x * w, dy = py - L.y * h;
    const X = dx * Math.cos(r) - dy * Math.sin(r), Y = dx * Math.sin(r) + dy * Math.cos(r);
    const det = 1 - sx * sy || 1;
    return [(X - sx * Y) / det, (Y - sy * X) / det];
  }
  const api = { FONTS, PACK, SHAPES, EFFECTS, WARPS, geometry, toLocal, fontCss, stack, layer, withEffects, style, draw, columns, shape, shapePaths, drawShape };
  if (typeof window !== "undefined") window.ProtoText = api;
  if (typeof module !== "undefined") module.exports = api;
})();
