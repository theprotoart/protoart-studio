// Protoart Studio "studio rack" skin: procedurally generated metal, tape and screw textures,
// plus the hardware controls (bakelite knob, detented rotary switch, VU meter).
// Runs before app.js. Textures are drawn once per launch on a canvas with a fixed seed, so the
// scratches are always in the same places (it's the same faceplate every time you open the app).
(function () {
  "use strict";

  // ---------------------------------------------------------------- seeded random
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function canvas(w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    return c;
  }

  // Draw fn at (x, y) and at the wrapped copies so the tile repeats without seams.
  function wrapped(w, h, x, y, reach, fn) {
    for (const dx of [-w, 0, w]) {
      for (const dy of [-h, 0, h]) {
        const X = x + dx, Y = y + dy;
        if (X + reach < 0 || X - reach > w || Y + reach < 0 || Y - reach > h) continue;
        fn(X, Y);
      }
    }
  }

  // ---------------------------------------------------------------- brushed aluminium
  function brushedAluminium(opts) {
    const { w = 1024, h = 1024, seed = 7, base = [196, 199, 202], brush = 7, grain = 5,
            scratches = 1, dings = 9, wear = 12 } = opts || {};
    const r = mulberry32(seed);
    const c = canvas(w, h);
    const g = c.getContext("2d");
    const img = g.createImageData(w, h);
    const d = img.data;
    const raw = new Float32Array(w);
    const row = new Float32Array(w);
    const R = 48; // brushing streak length (horizontal blur radius)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) raw[x] = r() - 0.5;
      // circular box blur along the row -> long horizontal streaks that tile
      let acc = 0;
      for (let k = -R; k <= R; k++) acc += raw[(k + w) % w];
      for (let x = 0; x < w; x++) {
        row[x] = acc / (2 * R + 1);
        acc += raw[(x + R + 1) % w] - raw[(x - R + w) % w];
      }
      const rowTone = gauss(r) * 2.2 + (r() < 0.025 ? gauss(r) * 7 : 0); // whole-row variation + rare deep strokes
      const o = y * w * 4;
      for (let x = 0; x < w; x++) {
        const v = rowTone + row[x] * brush * 9 + (r() - 0.5) * grain * 2;
        d[o + x * 4] = base[0] + v;
        d[o + x * 4 + 1] = base[1] + v;
        d[o + x * 4 + 2] = base[2] + v * 1.05;
        d[o + x * 4 + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);

    // wear: grime in places, rubbed-bright patches in others
    for (let i = 0; i < wear; i++) {
      const x = r() * w, y = r() * h, rad = 60 + r() * 220;
      const dark = r() < 0.55;
      wrapped(w, h, x, y, rad, (X, Y) => {
        const gr = g.createRadialGradient(X, Y, 0, X, Y, rad);
        gr.addColorStop(0, dark ? "rgba(40,36,30,0.07)" : "rgba(255,255,255,0.08)");
        gr.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = gr;
        g.fillRect(X - rad, Y - rad, rad * 2, rad * 2);
      });
    }
    drawScratches(g, w, h, r, scratches, false);
    drawDings(g, w, h, r, dings);
    return c;
  }

  // Hairline scratches, a few deep gouges and some circular rub marks.
  function drawScratches(g, w, h, r, amount, onDark) {
    const light = onDark ? "rgba(255,255,255," : "rgba(255,255,255,";
    const dark = onDark ? "rgba(0,0,0," : "rgba(30,30,32,";
    g.lineCap = "round";
    const hair = Math.round(85 * amount);
    for (let i = 0; i < hair; i++) {
      const x = r() * w, y = r() * h;
      const len = 4 + Math.pow(r(), 2.6) * 110;
      const ang = r() < 0.35 ? (r() - 0.5) * 0.4 : r() * Math.PI; // many roughly along the grain
      const bend = (r() - 0.5) * len * 0.25;
      const a = (onDark ? 0.08 : 0.12) + r() * (onDark ? 0.16 : 0.26);
      const dx = Math.cos(ang) * len, dy = Math.sin(ang) * len;
      wrapped(w, h, x, y, len, (X, Y) => {
        const mx = X + dx / 2 - Math.sin(ang) * bend, my = Y + dy / 2 + Math.cos(ang) * bend;
        g.lineWidth = 0.6 + r() * 0.5;
        g.strokeStyle = dark + (a * 0.55).toFixed(3) + ")";
        g.beginPath(); g.moveTo(X + 0.6, Y + 0.6); g.quadraticCurveTo(mx + 0.6, my + 0.6, X + dx + 0.6, Y + dy + 0.6); g.stroke();
        g.strokeStyle = light + a.toFixed(3) + ")";
        g.beginPath(); g.moveTo(X, Y); g.quadraticCurveTo(mx, my, X + dx, Y + dy); g.stroke();
      });
    }
    const deep = Math.round(2 * amount);
    for (let i = 0; i < deep; i++) {
      const x = r() * w, y = r() * h;
      const len = 60 + r() * 170, ang = r() * Math.PI, bend = (r() - 0.5) * 60;
      const dx = Math.cos(ang) * len, dy = Math.sin(ang) * len;
      wrapped(w, h, x, y, len, (X, Y) => {
        const mx = X + dx / 2 - Math.sin(ang) * bend, my = Y + dy / 2 + Math.cos(ang) * bend;
        g.lineWidth = 1.2;
        g.strokeStyle = dark + (onDark ? "0.45)" : "0.16)");
        g.beginPath(); g.moveTo(X, Y); g.quadraticCurveTo(mx, my, X + dx, Y + dy); g.stroke();
        g.lineWidth = 0.6;
        g.strokeStyle = light + (onDark ? "0.3)" : "0.5)");
        g.beginPath(); g.moveTo(X - 0.8, Y - 0.8); g.quadraticCurveTo(mx - 0.8, my - 0.8, X + dx - 0.8, Y + dy - 0.8); g.stroke();
      });
    }
    // rub marks: little bundles of concentric arcs, like something was dragged across the plate
    const swirls = Math.round(3 * amount);
    for (let i = 0; i < swirls; i++) {
      const x = r() * w, y = r() * h, rad = 30 + r() * 90, start = r() * Math.PI * 2, span = 0.4 + r() * 1.1;
      const n = 5 + Math.floor(r() * 8);
      wrapped(w, h, x, y, rad + 20, (X, Y) => {
        for (let k = 0; k < n; k++) {
          const rr = rad + k * (1.2 + r() * 2);
          g.lineWidth = 0.5;
          g.strokeStyle = light + (0.08 + r() * 0.14).toFixed(3) + ")";
          g.beginPath(); g.arc(X, Y, rr, start + r() * 0.2, start + span + r() * 0.2); g.stroke();
        }
      });
    }
  }

  function drawDings(g, w, h, r, n) {
    for (let i = 0; i < n; i++) {
      const x = r() * w, y = r() * h, rad = 0.8 + Math.pow(r(), 3) * 2.6;
      wrapped(w, h, x, y, rad * 3, (X, Y) => {
        const gr = g.createRadialGradient(X, Y, 0, X, Y, rad * 2.2);
        gr.addColorStop(0, "rgba(25,25,28,0.35)");
        gr.addColorStop(0.45, "rgba(25,25,28,0.12)");
        gr.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = gr;
        g.beginPath(); g.arc(X, Y, rad * 2.2, 0, Math.PI * 2); g.fill();
        g.strokeStyle = "rgba(255,255,255,0.55)"; // bright lip where the metal was pushed up
        g.lineWidth = 0.8;
        g.beginPath(); g.arc(X, Y, rad * 1.1, Math.PI * 0.95, Math.PI * 1.75); g.stroke();
      });
    }
  }

  // Transparent overlay with only the scratches: laid over bakelite, the brand plate and buttons.
  function scratchOverlay(seed, size) {
    const r = mulberry32(seed);
    const c = canvas(size, size);
    const g = c.getContext("2d");
    drawScratches(g, size, size, r, 0.5, true);
    return c;
  }

  // Masking tape: creamy paper, fibres along the roll, a few darker blotches.
  function maskingTape(seed) {
    const r = mulberry32(seed);
    const w = 512, h = 128;
    const c = canvas(w, h);
    const g = c.getContext("2d");
    const img = g.createImageData(w, h);
    const d = img.data;
    for (let y = 0; y < h; y++) {
      const rowT = gauss(r) * 1.6;
      for (let x = 0; x < w; x++) {
        const n = (r() - 0.5) * 10 + rowT;
        const o = (y * w + x) * 4;
        d[o] = 233 + n; d[o + 1] = 219 + n; d[o + 2] = 178 + n * 0.8; d[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    for (let i = 0; i < 260; i++) { // paper fibres
      const x = r() * w, y = r() * h, len = 4 + r() * 22;
      wrapped(w, h, x, y, len, (X, Y) => {
        g.strokeStyle = r() < 0.5 ? "rgba(255,250,230,0.35)" : "rgba(150,125,70,0.18)";
        g.lineWidth = 0.6;
        g.beginPath(); g.moveTo(X, Y); g.lineTo(X + len, Y + (r() - 0.5) * 3); g.stroke();
      });
    }
    for (let i = 0; i < 9; i++) { // grubby thumb marks
      const x = r() * w, y = r() * h, rad = 10 + r() * 40;
      wrapped(w, h, x, y, rad, (X, Y) => {
        const gr = g.createRadialGradient(X, Y, 0, X, Y, rad);
        gr.addColorStop(0, "rgba(140,110,60,0.10)");
        gr.addColorStop(1, "rgba(140,110,60,0)");
        g.fillStyle = gr;
        g.fillRect(X - rad, Y - rad, rad * 2, rad * 2);
      });
    }
    return c;
  }

  // Cross-head rack screw seen from the front (a few slightly different turns of the screwdriver).
  function screw(angle, seed) {
    const r = mulberry32(seed);
    const s = 64, c = canvas(s, s), g = c.getContext("2d"), m = s / 2;
    // countersink shadow in the plate
    let gr = g.createRadialGradient(m, m + 2, 20, m, m + 2, 31);
    gr.addColorStop(0, "rgba(0,0,0,0.55)"); gr.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = gr; g.beginPath(); g.arc(m, m + 2, 31, 0, Math.PI * 2); g.fill();
    // domed chrome head
    gr = g.createRadialGradient(m - 8, m - 9, 2, m, m, 26);
    gr.addColorStop(0, "#ffffff"); gr.addColorStop(0.35, "#d2d4d8"); gr.addColorStop(0.75, "#7d8086"); gr.addColorStop(1, "#3c3e42");
    g.fillStyle = gr; g.beginPath(); g.arc(m, m, 25, 0, Math.PI * 2); g.fill();
    g.strokeStyle = "rgba(0,0,0,0.6)"; g.lineWidth = 1.5; g.stroke();
    // cross
    g.save(); g.translate(m, m); g.rotate(angle);
    for (const rot of [0, Math.PI / 2]) {
      g.save(); g.rotate(rot);
      g.fillStyle = "#1b1c1f"; g.fillRect(-15, -3.2, 30, 6.4);
      g.fillStyle = "rgba(255,255,255,0.55)"; g.fillRect(-15, 3.2, 30, 1.2);
      g.restore();
    }
    g.restore();
    // tool marks
    g.strokeStyle = "rgba(255,255,255,0.35)"; g.lineWidth = 0.7;
    for (let i = 0; i < 6; i++) {
      const a = r() * Math.PI * 2, rr = 8 + r() * 14;
      g.beginPath(); g.arc(m, m, rr, a, a + 0.3 + r() * 0.6); g.stroke();
    }
    return c;
  }

  // ---------------------------------------------------------------- light / dark faceplate
  // "light" = brushed aluminium, "dark" = black anodised, "system" = follow macOS. Applied before the
  // page paints, so the app never flashes the wrong colour. Saved in this window's storage.
  const root = document.documentElement;
  const THEME_KEY = "protoart.theme";
  const darkQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  let themePref = "dark";
  function applyTheme(pref) {
    themePref = ["light", "dark", "system"].includes(pref) ? pref : "dark"; // dark is the default
    const dark = themePref === "dark" || (themePref === "system" && !!(darkQuery && darkQuery.matches));
    root.dataset.theme = dark ? "dark" : "light";
    return themePref;
  }
  try { applyTheme(JSON.parse(localStorage.getItem(THEME_KEY)) || "dark"); } catch { applyTheme("dark"); }
  if (darkQuery && darkQuery.addEventListener) darkQuery.addEventListener("change", () => { if (themePref === "system") applyTheme("system"); });
  // Themes ("skins"): the colour of the metal and of the lit controls. Light / dark still applies on top.
  //   studio  = Classic Studio (charcoal + amber)       console = Vintage Console (warm cream, red & yellow lamps)
  //   puig    = Vintage Compressor (blue-grey steel)    magma   = Magma (black and ember)
  const SKIN_KEY = "protoart.skin";
  const SKINS = {
    studio: { alu: [204, 202, 198], aluDark: [154, 153, 150], black: [40, 41, 44] },
    console: { alu: [224, 213, 192], aluDark: [178, 166, 146], black: [54, 47, 40] },
    puig: { alu: [180, 190, 199], aluDark: [134, 145, 156], black: [46, 54, 64] },
    magma: { alu: [196, 186, 182], aluDark: [150, 139, 135], black: [26, 21, 21] },
  };
  let skin = "studio";
  try { const v = JSON.parse(localStorage.getItem(SKIN_KEY)); if (SKINS[v]) skin = v; } catch { /* default */ }
  root.dataset.skin = skin;
  window.RackTheme = {
    get: () => themePref,
    set(pref) {
      const v = applyTheme(pref);
      try { localStorage.setItem(THEME_KEY, JSON.stringify(v)); } catch { /* storage unavailable */ }
      return v;
    },
    skins: () => Object.keys(SKINS),
    getSkin: () => skin,
    setSkin(name) {
      if (!SKINS[name] || name === skin) return skin;
      skin = name;
      root.dataset.skin = name;
      try { localStorage.setItem(SKIN_KEY, JSON.stringify(name)); } catch { /* storage unavailable */ }
      try { metalTextures(); } catch (err) { console.warn("[rack] textures", err); }
      return skin;
    },
  };

  // ---------------------------------------------------------------- install textures
  function metalTextures() { // the faceplate metal, in the current theme's tone
    const k = SKINS[skin] || SKINS.studio;
    setTex("--tex-alu", brushedAluminium({ seed: 1979, base: k.alu }), "image/jpeg");
    setTex("--tex-alu-dark", brushedAluminium({ seed: 1176, base: k.aluDark, w: 768, h: 768, scratches: 0.8, dings: 8 }), "image/jpeg");
    setTex("--tex-black", brushedAluminium({ seed: 1176, base: k.black, brush: 4, grain: 3, scratches: 0.55, dings: 6, wear: 9 }), "image/jpeg");
  }
  function setTex(name, c, type) {
    try {
      root.style.setProperty(name, `url("${c.toDataURL(type || "image/png", 0.88)}")`);
    } catch (err) {
      console.warn("rack texture", name, err);
    }
  }
  try {
    const t0 = performance.now();
    metalTextures();
    setTex("--tex-scratch", scratchOverlay(2, 384));
    setTex("--tex-tape", maskingTape(5), "image/jpeg");
    [0.3, 1.1, 0.75, 1.45].forEach((a, i) => setTex(`--screw-${i}`, screw(a, 40 + i)));
    root.classList.add("rack-ready");
    console.log(`[rack] textures ready in ${Math.round(performance.now() - t0)} ms`);
  } catch (err) {
    console.warn("[rack] textures skipped", err); // the skin still works with its flat fallbacks
  }

  // ---------------------------------------------------------------- hardware controls (Vue)
  // Written with render functions so they also work with Vue's runtime-only build in the packaged app.
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const SWEEP = 270; // knob travel in degrees

  // Chunky bakelite knob with a printed scale and an amber readout you can type into.
  //   <rack-knob v-model.number="form.steps" :min="1" :max="150" :step="1" label="Steps"></rack-knob>
  const RackKnob = {
    name: "RackKnob",
    props: {
      modelValue: { type: Number, default: 0 },
      min: { type: Number, default: 0 },
      max: { type: Number, default: 100 },
      step: { type: Number, default: 1 },
      label: { type: String, default: "" },
      decimals: { type: Number, default: -1 },
      size: { type: Number, default: 56 },
      title: { type: String, default: "" },
      defaultValue: { type: Number, default: null }, // double-click the knob to go back to this
    },
    emits: ["update:modelValue", "change"],
    data() { return { drag: null, editText: null }; },
    computed: {
      value() { return Number.isFinite(this.modelValue) ? this.modelValue : this.min; },
      dec() {
        if (this.decimals >= 0) return this.decimals;
        const s = String(this.step);
        return s.includes(".") ? s.split(".")[1].length : 0;
      },
      angle() {
        const f = (clamp(this.value, this.min, this.max) - this.min) / (this.max - this.min || 1);
        return -SWEEP / 2 + f * SWEEP;
      },
    },
    methods: {
      snap(v) {
        const s = this.step || 1;
        const q = Math.round((v - this.min) / s) * s + this.min;
        return Number(clamp(q, this.min, this.max).toFixed(this.dec));
      },
      set(v) {
        const n = this.snap(v);
        if (n !== this.modelValue) {
          this.$emit("update:modelValue", n);
          this.$emit("change", n);
        }
      },
      onDown(e) {
        if (e.button !== 0) return;
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        e.currentTarget.focus();
        this.drag = { y: e.clientY, x: e.clientX, v: this.value };
      },
      onMove(e) {
        if (!this.drag) return;
        const span = this.max - this.min;
        // up / right turns it up; 300 px of travel = full sweep (shift = fine)
        const px = (this.drag.y - e.clientY) + (e.clientX - this.drag.x) * 0.5;
        const fine = e.shiftKey ? 0.2 : 1;
        this.set(this.drag.v + (px / 300) * span * fine);
      },
      onUp() { this.drag = null; },
      onWheel(e) {
        e.preventDefault();
        const dir = e.deltaY < 0 || e.deltaX > 0 ? 1 : -1;
        this.set(this.value + dir * (this.step || 1) * (e.shiftKey ? 1 : this.coarse()));
      },
      coarse() {
        const steps = (this.max - this.min) / (this.step || 1);
        return steps > 120 ? 2 : 1;
      },
      onKey(e) {
        const k = e.key;
        const s = this.step || 1;
        const big = Math.max(s, (this.max - this.min) / 10);
        let v = null;
        if (k === "ArrowUp" || k === "ArrowRight") v = this.value + s;
        else if (k === "ArrowDown" || k === "ArrowLeft") v = this.value - s;
        else if (k === "PageUp") v = this.value + big;
        else if (k === "PageDown") v = this.value - big;
        else if (k === "Home") v = this.min;
        else if (k === "End") v = this.max;
        if (v !== null) { e.preventDefault(); this.set(v); }
      },
      // Typed numbers are taken as they are (only kept inside min-max), not rounded to the knob's steps:
      // the knob is for quick turning, the readout for exact values.
      typed(raw) {
        const v = parseFloat(String(raw).replace(",", "."));
        if (!Number.isFinite(v)) return null;
        const c = clamp(v, this.min, this.max);
        return (this.step || 1) >= 1 && Number.isInteger(this.step || 1) ? Math.round(c) : Number(c.toFixed(2));
      },
      onFocusRead(e) {
        this.editText = this.shown();
        this.$nextTick(() => e.target.select());
      },
      onInputRead(e) {
        this.editText = e.target.value;
        const v = parseFloat(e.target.value.replace(",", "."));
        // update live while typing, but only for complete numbers that are already in range
        if (Number.isFinite(v) && v >= this.min && v <= this.max && !/[.,]$/.test(e.target.value)) {
          const n = this.typed(v);
          if (n !== this.modelValue) { this.$emit("update:modelValue", n); this.$emit("change", n); }
        }
      },
      onBlurRead() {
        const n = this.editText === null ? null : this.typed(this.editText);
        if (n !== null && n !== this.modelValue) { this.$emit("update:modelValue", n); this.$emit("change", n); }
        this.editText = null;
      },
      shown() {
        const v = this.value;
        return Number.isInteger(v) ? v.toFixed(this.dec) : String(Number(v.toFixed(Math.max(this.dec, 2))));
      },
    },
    render() {
      const h = Vue.h;
      const ticks = [];
      for (let i = 0; i <= 10; i++) {
        ticks.push(h("i", { class: ["k-tick", { major: i % 5 === 0 }], style: { transform: `rotate(${-SWEEP / 2 + (i * SWEEP) / 10}deg)` } }));
      }
      return h("div", { class: "rack-knob", style: { "--k": this.size + "px" } }, [
        this.label ? h("div", { class: "k-label" }, this.label) : null,
        h("div", { class: "k-dial" }, [
          h("div", { class: "k-scale" }, ticks),
          h("span", { class: "k-min" }, String(this.min)),
          h("span", { class: "k-max" }, String(this.max)),
          h("div", {
            class: ["k-cap", { turning: !!this.drag }],
            tabindex: 0,
            role: "slider",
            "aria-label": this.label || this.title,
            "aria-valuemin": this.min, "aria-valuemax": this.max, "aria-valuenow": this.value,
            title: (this.title ? this.title + "\n" : "") + "Drag up/down or scroll to turn · Shift = fine · arrow keys work too" +
              (this.defaultValue !== null ? ` · double-click: back to the default (${this.defaultValue})` : ""),
            onPointerdown: this.onDown, onPointermove: this.onMove, onPointerup: this.onUp, onPointercancel: this.onUp,
            onWheel: this.onWheel, onKeydown: this.onKey,
            onDblclick: () => {
              if (this.defaultValue !== null && Number.isFinite(this.defaultValue)) this.set(this.defaultValue);
              else if (this.$refs.read) this.$refs.read.focus();
            },
          }, [
            h("div", { class: "k-skirt" }),
            h("div", { class: "k-top", style: { transform: `rotate(${this.angle}deg)` } }, [h("i", { class: "k-pointer" })]),
          ]),
        ]),
        h("input", {
          ref: "read", class: ["k-read", { editing: this.editText !== null }], type: "text", inputmode: "decimal",
          value: this.editText !== null ? this.editText : this.shown(),
          title: `Click to type an exact value (${this.min}–${this.max})`,
          onFocus: this.onFocusRead, onInput: this.onInputRead, onBlur: this.onBlurRead,
          onKeydown: (e) => {
            if (e.key === "Enter") e.target.blur();
            if (e.key === "Escape") { this.editText = null; e.target.blur(); }
            if (e.key === "ArrowUp" || e.key === "ArrowDown") { // nudge like a number field
              e.preventDefault();
              const n = this.snap(this.value + (e.key === "ArrowUp" ? 1 : -1) * (this.step || 1));
              this.$emit("update:modelValue", n); this.editText = n.toFixed(this.dec);
            }
          },
          "aria-label": (this.label || "value") + " (type a number)",
        }),
      ]);
    },
  };

  // Detented rotary switch: clicks into a fixed set of positions, like a range selector.
  //   <rack-switch :options="[{id, label, sub}]" :model-value="id or null" @update:model-value="..."></rack-switch>
  const RackSwitch = {
    name: "RackSwitch",
    props: {
      options: { type: Array, default: () => [] },
      modelValue: { default: null },
      label: { type: String, default: "" },
      spread: { type: Number, default: 110 }, // degrees between the first and last position
      offLabel: { type: String, default: "" }, // shown when the value isn't one of the positions
    },
    emits: ["update:modelValue"],
    data() { return { drag: null }; },
    computed: {
      index() { return this.options.findIndex((o) => o.id === this.modelValue); },
      positions() {
        const n = this.options.length;
        return this.options.map((_, i) => (n < 2 ? 0 : -this.spread / 2 + (i * this.spread) / (n - 1)));
      },
      angle() {
        if (this.index >= 0) return this.positions[this.index];
        return this.positions.length ? (this.positions[0] + this.positions[this.positions.length - 1]) / 2 + 180 : 0; // parked
      },
    },
    methods: {
      pick(i) {
        const o = this.options[clamp(i, 0, this.options.length - 1)];
        if (o && o.id !== this.modelValue) this.$emit("update:modelValue", o.id);
      },
      step(dir) {
        const i = this.index < 0 ? (dir > 0 ? 0 : this.options.length - 1) : this.index + dir;
        this.pick(i);
      },
      onDown(e) {
        if (e.button !== 0) return;
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        e.currentTarget.focus();
        this.drag = { x: e.clientX, y: e.clientY, i: Math.max(0, this.index), moved: false };
      },
      onMove(e) {
        if (!this.drag) return;
        const px = (e.clientX - this.drag.x) + (this.drag.y - e.clientY);
        if (Math.abs(px) > 4) this.drag.moved = true;
        if (this.drag.moved) this.pick(this.drag.i + Math.round(px / 36));
      },
      onUp() {
        if (this.drag && !this.drag.moved) this.step(this.index >= this.options.length - 1 ? -(this.options.length - 1) : 1);
        this.drag = null;
      },
      onKey(e) {
        if (["ArrowUp", "ArrowRight"].includes(e.key)) { e.preventDefault(); this.step(1); }
        if (["ArrowDown", "ArrowLeft"].includes(e.key)) { e.preventDefault(); this.step(-1); }
      },
    },
    render() {
      const h = Vue.h;
      const R = 62;
      const marks = this.options.map((o, i) => {
        const a = (this.positions[i] - 90) * (Math.PI / 180);
        return h("button", {
          class: ["sw-pos", { on: i === this.index }],
          style: { left: `calc(50% + ${Math.cos(a) * R}px)`, top: `${80 + Math.sin(a) * R}px` },
          title: o.title || "",
          onClick: () => this.pick(i),
        }, [h("span", { class: "sw-led" }), h("b", o.label), o.sub ? h("small", String(o.sub)) : null]);
      });
      const detents = this.positions.map((p) => h("i", { class: "sw-detent", style: { transform: `rotate(${p}deg)` } }));
      return h("div", { class: "rack-switch" }, [
        h("div", { class: "sw-body" }, [
          ...marks,
          h("div", { class: "sw-ring" }, detents),
          h("div", {
            class: ["sw-cap", { parked: this.index < 0 }], tabindex: 0, role: "radiogroup",
            "aria-label": this.label,
            title: "Click to step to the next position, drag to turn, or click a label",
            onPointerdown: this.onDown, onPointermove: this.onMove, onPointerup: this.onUp, onPointercancel: () => { this.drag = null; },
            onKeydown: this.onKey,
          }, [h("div", { class: "sw-bar", style: { transform: `rotate(${this.angle}deg)` } }, [h("i")])]),
        ]),
        this.label ? h("div", { class: "k-label" }, this.label) : null,
        this.index < 0 && this.offLabel ? h("div", { class: "sw-off" }, this.offLabel) : null,
      ]);
    },
  };

  // Analog VU meter. pct 0-100 moves the needle; "live" adds the little flutter a real needle has.
  const RackVu = {
    name: "RackVu",
    props: {
      pct: { type: Number, default: 0 },
      live: { type: Boolean, default: false },
      lit: { type: Boolean, default: true },
      caption: { type: String, default: "PROGRESS" },
    },
    computed: {
      angle() { return -46 + clamp(this.pct || 0, 0, 100) * 0.92; },
    },
    render() {
      const h = Vue.h;
      const cx = 80, cy = 104, R = 74;
      const pt = (deg, rad) => {
        const a = (deg - 90) * (Math.PI / 180);
        return [cx + Math.cos(a) * rad, cy + Math.sin(a) * rad];
      };
      const arc = (d0, d1, rad) => {
        const [x0, y0] = pt(d0, rad), [x1, y1] = pt(d1, rad);
        return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${rad} ${rad} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
      };
      const kids = [];
      kids.push(h("path", { d: arc(-46, 27.6, R), class: "vu-arc" }));
      kids.push(h("path", { d: arc(27.6, 46, R), class: "vu-arc red" }));
      kids.push(h("path", { d: arc(27.6, 46, R + 3), class: "vu-arc red thick" }));
      for (let i = 0; i <= 20; i++) {
        const deg = -46 + i * 4.6;
        const major = i % 5 === 0;
        const [x0, y0] = pt(deg, R), [x1, y1] = pt(deg, R + (major ? 9 : 5));
        kids.push(h("line", { x1: x0, y1: y0, x2: x1, y2: y1, class: ["vu-tick", { major, red: deg > 27 }] }));
        if (major) {
          const [tx, ty] = pt(deg, R + 16);
          kids.push(h("text", { x: tx, y: ty, class: ["vu-num", { red: deg > 27 }] }, String(i * 5)));
        }
      }
      kids.push(h("text", { x: cx, y: 70, class: "vu-big" }, "VU"));
      kids.push(h("text", { x: cx, y: 82, class: "vu-cap" }, this.caption));
      const [nx, ny] = pt(0, R + 6);
      kids.push(h("g", { class: ["vu-needle", { live: this.live }], style: { transform: `rotate(${this.angle}deg)` } }, [
        h("line", { x1: cx, y1: cy, x2: nx, y2: ny }),
      ]));
      return h("div", { class: ["rack-vu", { lit: this.lit, live: this.live }] }, [
        h("div", { class: "vu-lamp warm" }),
        h("div", { class: "vu-lamp hot" }),
        h("svg", { viewBox: "0 0 160 98", "aria-hidden": "true" }, kids),
        h("div", { class: "vu-glass" }),
      ]);
    },
  };

  // A studio fader (like a graphic EQ's): a brushed plate with a printed scale, a slot and a ridged cap with a light
  // line across it. Drag the cap, scroll, use the arrow keys, or click a number on the scale.
  //   <rack-fader v-model.number="form.num_images" :min="1" :max="16" label="Images" :marks="[1, 2, 4, 8, 12, 16]"></rack-fader>
  const RackFader = {
    name: "RackFader",
    props: {
      modelValue: { type: Number, default: 0 },
      min: { type: Number, default: 0 },
      max: { type: Number, default: 10 },
      step: { type: Number, default: 1 },
      label: { type: String, default: "" },
      marks: { type: Array, default: null },     // the numbers printed on the scale
      height: { type: Number, default: 128 },   // travel, in px
      title: { type: String, default: "" },
      defaultValue: { type: Number, default: null },
      typeMax: { type: Number, default: null },  // the number under the fader can be typed up to this (default: max)
    },
    emits: ["update:modelValue", "change"],
    data() { return { drag: null, editText: null }; },
    computed: {
      top() { return Math.max(this.max, this.typeMax || this.max); },
      typedValue() { return Number.isFinite(this.modelValue) ? clamp(this.modelValue, this.min, this.top) : this.min; },
      value() { return clamp(this.typedValue, this.min, this.max); },
      frac() { return (this.value - this.min) / (this.max - this.min || 1); },
      scale() {
        if (this.marks) return this.marks;
        const out = [];
        for (let v = this.min; v <= this.max; v += Math.max(this.step, (this.max - this.min) / 6)) out.push(Math.round(v));
        return out;
      },
    },
    methods: {
      set(v) {
        const s = this.step || 1;
        const n = clamp(Math.round((v - this.min) / s) * s + this.min, this.min, this.max);
        if (n !== this.modelValue) { this.$emit("update:modelValue", n); this.$emit("change", n); }
      },
      at(e) {  // the value under the pointer
        const r = this.$refs.slot.getBoundingClientRect();
        return this.min + (1 - clamp((e.clientY - r.top) / Math.max(1, r.height), 0, 1)) * (this.max - this.min);
      },
      onDown(e) {
        if (e.button !== 0) return;
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        this.$refs.cap.focus();
        this.drag = true;
        this.set(this.at(e));
      },
      onMove(e) { if (this.drag) this.set(this.at(e)); },
      onUp() { this.drag = null; },
      onWheel(e) { e.preventDefault(); this.set(this.value + (e.deltaY < 0 ? 1 : -1) * (this.step || 1)); },
      onKey(e) {
        const k = e.key, s = this.step || 1;
        let v = null;
        if (k === "ArrowUp" || k === "ArrowRight") v = this.value + s;
        else if (k === "ArrowDown" || k === "ArrowLeft") v = this.value - s;
        else if (k === "Home") v = this.min;
        else if (k === "End") v = this.max;
        if (v !== null) { e.preventDefault(); this.set(v); }
      },
      // the number under the fader: click it and type an exact amount (Return to set, Esc to cancel)
      typed(raw) {
        const v = parseFloat(String(raw).replace(",", "."));
        if (!Number.isFinite(v)) return null;
        const s = this.step || 1;
        return clamp(Math.round((v - this.min) / s) * s + this.min, this.min, this.top);
      },
      commitTyped() {
        const n = this.editText === null ? null : this.typed(this.editText);
        if (n !== null && n !== this.modelValue) { this.$emit("update:modelValue", n); this.$emit("change", n); }
        this.editText = null;
      },
    },
    render() {
      const h = Vue.h, H = this.height, span = this.max - this.min || 1;
      const ticks = [];
      for (let v = this.min; v <= this.max; v += this.step || 1) {
        const y = (1 - (v - this.min) / span) * H, major = this.scale.includes(v);
        ticks.push(h("i", { class: ["f-tick", { major }], style: { top: y + "px" } }));
        if (major) ticks.push(h("b", { class: ["f-num", { on: v === this.value }], style: { top: y + "px" },
                                      onClick: () => this.set(v), title: `Set to ${v}` }, String(v)));
      }
      return h("div", { class: "rack-fader", style: { "--h": H + "px" }, title: this.title }, [
        this.label ? h("div", { class: "f-label" }, this.label) : null,
        h("div", { class: "f-plate" }, [
          h("i", { class: "f-screw top" }), h("i", { class: "f-screw bottom" }),
          h("div", { class: "f-scale" }, ticks),
          h("div", { ref: "slot", class: "f-slot", onPointerdown: this.onDown, onPointermove: this.onMove,
                     onPointerup: this.onUp, onPointercancel: this.onUp, onWheel: this.onWheel }, [
            h("div", {
              ref: "cap", class: ["f-cap", { moving: !!this.drag }], tabindex: 0, role: "slider",
              "aria-label": this.label || this.title, "aria-valuemin": this.min, "aria-valuemax": this.max, "aria-valuenow": this.value,
              title: (this.title ? this.title + "\n" : "") + "Slide it up or down, scroll, or click a number · arrow keys work too" +
                (this.defaultValue !== null ? ` · double-click: back to ${this.defaultValue}` : ""),
              style: { top: (1 - this.frac) * H + "px" }, onKeydown: this.onKey,
              onDblclick: () => { if (this.defaultValue !== null) this.set(this.defaultValue); },
            }),
          ]),
        ]),
        h("input", {
          class: ["f-read", { editing: this.editText !== null }], type: "text", inputmode: "numeric",
          value: this.editText !== null ? this.editText : String(this.typedValue),
          title: `Click to type an exact number (${this.min}–${this.top})`,
          "aria-label": (this.label || "value") + " (type a number)",
          onFocus: (e) => { this.editText = String(this.typedValue); this.$nextTick(() => e.target.select()); },
          onInput: (e) => { this.editText = e.target.value; },
          onBlur: this.commitTyped,
          onKeydown: (e) => {
            if (e.key === "Enter") e.target.blur();
            else if (e.key === "Escape") { this.editText = null; e.target.blur(); }
            else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
              e.preventDefault();
              const n = this.typed((this.typed(this.editText) ?? this.typedValue) + (e.key === "ArrowUp" ? 1 : -1) * (this.step || 1));
              this.editText = String(n);
            }
          },
        }),
      ]);
    },
  };

  window.RackParts = {
    install(app) {
      app.component("rack-knob", RackKnob);
      app.component("rack-fader", RackFader);
      app.component("rack-switch", RackSwitch);
      app.component("rack-vu", RackVu);
    },
  };
})();
