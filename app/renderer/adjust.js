// Adjust: Lightroom / Camera Raw-style sliders and Nik Color Efex-style filters, rendered live on the GPU
// (WebGL 2). Same controls and the same order as ProtoArt Beautify's Core Image engine:
//   Light (exposure, contrast, highlights, shadows, whites, blacks) → Color (temperature, tint, vibrance,
//   saturation) → Presence (clarity, sharpening, vignette) → the filter stack, top to bottom
//   (Detail Extractor, Dark Contrasts, Bleach Bypass, Glamour Glow, Bloom, Film Grain).
// Radii are relative to the picture's long side, so the small live preview and the full-size save match.
/* global window */
(function () {
  "use strict";

  const FILTER_ORDER = ["extract", "dark", "bleach", "glow", "bloom", "grain"];

  function defaults() {
    return {
      exposure: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
      temperature: 0, tint: 0, vibrance: 0, saturation: 0,
      clarity: 0, sharpen: 0, vignette: 0,
      extractOn: false, deAmount: 0.5, deContrast: 0, deSaturation: 0, deRadius: 1,
      darkOn: false, dcStrength: 0.5, dcBrightness: 0, dcSaturation: 0,
      bleachOn: false, bbBrightness: 0, bbSaturation: 0, bbContrast: 0, bbLocalContrast: 0,
      glowOn: false, ggGlow: 0.5, ggSaturation: 0, ggWarmth: 0,
      bloomOn: false, blAmount: 0.6, blTransition: 0.4, blSize: 0.4, blBurn: 0,
      grainOn: false, grainAmount: 0.5, grainSize: 0.3, grainSoftness: 0.3, grainColor: false,
      order: FILTER_ORDER.slice(),
    };
  }

  // Ready-made looks (Color Efex-style starting points); each starts from defaults().
  const LOOKS = [
    { id: "punchy", label: "Punchy", tip: "More contrast, clarity and colour", set: { contrast: 0.25, clarity: 0.35, vibrance: 0.35, shadows: 0.15, highlights: -0.2 } },
    { id: "detail", label: "Detail pop", tip: "Detail Extractor + a little sharpening: texture everywhere", set: { extractOn: true, deAmount: 0.45, sharpen: 0.3 } },
    { id: "moody", label: "Moody", tip: "Dark Contrasts, cooler, a soft vignette", set: { darkOn: true, dcStrength: 0.55, temperature: -0.15, vignette: 0.35 } },
    { id: "film", label: "Film", tip: "Bleach Bypass with fine grain", set: { bleachOn: true, bbSaturation: 0.1, grainOn: true, grainAmount: 0.35, grainSize: 0.2 } },
    { id: "glow", label: "Soft glow", tip: "Glamour Glow, warm: flattering skin", set: { glowOn: true, ggGlow: 0.45, ggWarmth: 0.2, contrast: -0.05 } },
    { id: "bloom", label: "Bloom", tip: "Procreate-style Bloom: the highlights glow", set: { bloomOn: true, blAmount: 0.6, blTransition: 0.45, blSize: 0.45 } },
    { id: "bw", label: "B&W contrast", tip: "Black and white with deep contrast", set: { saturation: -1, darkOn: true, dcStrength: 0.5, clarity: 0.2 } },
  ];

  // Settings saved before a filter existed: fill in the new filter's defaults and put it in the stack
  // (at the end, or just before Film Grain when that's last).
  function fullOrder(order) {
    const o = (Array.isArray(order) ? order : []).filter((id) => FILTER_ORDER.includes(id));
    for (const id of FILTER_ORDER) {
      if (o.includes(id)) continue;
      if (id !== "grain" && o[o.length - 1] === "grain") o.splice(o.length - 1, 0, id); else o.push(id);
    }
    return o;
  }
  function withDefaults(set) {
    const p = Object.assign(defaults(), set || {});
    p.order = fullOrder(p.order);
    return p;
  }

  function isDefault(p) {
    const d = defaults();
    return Object.keys(d).every((k) => k === "order" || p[k] === d[k]);
  }
  function anyOn(p) {
    return !isDefault(Object.assign({}, p, { order: FILTER_ORDER }));
  }

  // ------------------------------------------------------------------ shaders
  const VS = `#version 300 es
in vec2 pos; out vec2 uv;
void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }`;
  const HEAD = `#version 300 es
precision highp float;
in vec2 uv; out vec4 o;
uniform sampler2D img;
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 toLin(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGam(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
vec3 colorControls(vec3 c, float contrast, float sat, float bright) {
  c = mix(vec3(luma(c)), c, sat);
  c = (c - 0.5) * contrast + 0.5;
  return c + bright;
}
`;
  const FS = {
    copy: `void main() { o = texture(img, uv); }`,
    // 4-tap box downsample (the first step of a big blur)
    down: `uniform vec2 texel;
void main() {
  vec4 a = texture(img, uv + texel * vec2(-0.5, -0.5)), b = texture(img, uv + texel * vec2(0.5, -0.5));
  vec4 c = texture(img, uv + texel * vec2(-0.5, 0.5)), d = texture(img, uv + texel * vec2(0.5, 0.5));
  o = (a + b + c + d) * 0.25;
}`,
    blur: `uniform vec2 dir; uniform float sigma; uniform int radius;
void main() {
  vec4 acc = texture(img, uv); float wsum = 1.0;
  for (int i = 1; i <= 48; i++) {
    if (i > radius) break;
    float w = exp(-float(i * i) / (2.0 * sigma * sigma));
    acc += (texture(img, uv + dir * float(i)) + texture(img, uv - dir * float(i))) * w;
    wsum += 2.0 * w;
  }
  o = acc / wsum;
}`,
    // Light + Color, in Camera Raw's order
    basic: `uniform sampler2D blurred; uniform sampler2D curve;
uniform float exposure, contrast, saturation, temperature, tint, vibrance, shadowLift, highlightAmt;
uniform bool useCurve, useRecovery;
void main() {
  vec3 c = texture(img, uv).rgb;
  if (exposure != 0.0) c = toGam(toLin(c) * exp2(exposure));
  if (useRecovery) {
    float B = luma(texture(blurred, uv).rgb);
    float ws = 1.0 - smoothstep(0.0, 0.6, B), wh = smoothstep(0.4, 1.0, B);
    float L = luma(c);
    c += shadowLift * ws * 0.6 * (1.0 - L) * (c + 0.08);
    c -= (1.0 - highlightAmt) * wh * 0.55 * c * L;
  }
  if (useCurve) c = vec3(texture(curve, vec2(clamp(c.r, 0.0, 1.0) * 0.996 + 0.002, 0.5)).r,
                         texture(curve, vec2(clamp(c.g, 0.0, 1.0) * 0.996 + 0.002, 0.5)).r,
                         texture(curve, vec2(clamp(c.b, 0.0, 1.0) * 0.996 + 0.002, 0.5)).r);
  if (contrast != 0.0 || saturation != 0.0) c = colorControls(c, 1.0 + 0.5 * contrast, max(0.0, 1.0 + saturation), 0.0);
  if (temperature != 0.0 || tint != 0.0) {
    vec3 l = toLin(c);
    float y0 = luma(l);
    l *= vec3(1.0 + 0.28 * temperature, 1.0 - 0.18 * tint, 1.0 - 0.28 * temperature);
    l *= y0 / max(luma(l), 1e-4);  // keep the brightness
    c = toGam(l);
  }
  if (vibrance != 0.0) {
    float mx = max(c.r, max(c.g, c.b)), mn = min(c.r, min(c.g, c.b));
    float sat = mx - mn;
    // skin tones (red > green > blue) get less: Lightroom's vibrance protects them
    float skin = (c.r > c.g && c.g > c.b) ? 0.5 : 1.0;
    float amt = vibrance * (1.0 - sat) * skin * (vibrance > 0.0 ? 1.2 : 1.0);
    c = mix(vec3(luma(c)), c, max(0.0, 1.0 + amt));
  }
  o = vec4(c, 1.0);
}`,
    // unsharp mask: c + amount * (c - blur); midtone-weighted for Clarity
    local: `uniform sampler2D blurred; uniform float amount; uniform bool midtones;
void main() {
  vec3 c = texture(img, uv).rgb, b = texture(blurred, uv).rgb;
  float w = 1.0;
  if (midtones) { float L = luma(c); w = 1.0 - pow(abs(L - 0.5) * 2.0, 2.0); }
  o = vec4(c + amount * w * (c - b), 1.0);
}`,
    // CIHighlightShadowAdjust-style lift (Detail Extractor, Dark Contrasts)
    recover: `uniform sampler2D blurred; uniform float shadowLift, highlightAmt;
void main() {
  vec3 c = texture(img, uv).rgb;
  float B = luma(texture(blurred, uv).rgb);
  float ws = 1.0 - smoothstep(0.0, 0.6, B), wh = smoothstep(0.4, 1.0, B);
  float L = luma(c);
  c += shadowLift * ws * 0.6 * (1.0 - L) * (c + 0.08);
  c -= (1.0 - highlightAmt) * wh * 0.55 * c * L;
  o = vec4(c, 1.0);
}`,
    controls: `uniform float contrast, sat, bright, gamma;
void main() {
  vec3 c = texture(img, uv).rgb;
  if (gamma != 1.0) c = pow(max(c, 0.0), vec3(gamma));
  o = vec4(colorControls(c, contrast, sat, bright), 1.0);
}`,
    bleach: `void main() {
  vec3 bg = clamp(texture(img, uv).rgb, 0.0, 1.0);
  vec3 fg = vec3(luma(bg));
  vec3 ov = mix(2.0 * bg * fg, 1.0 - 2.0 * (1.0 - bg) * (1.0 - fg), step(0.5, bg));
  o = vec4(mix(bg, ov, 0.65), 1.0);
}`,
    glow: `uniform sampler2D blurred; uniform float amount, warmth;
void main() {
  vec3 c = clamp(texture(img, uv).rgb, 0.0, 1.0), b = clamp(texture(blurred, uv).rgb, 0.0, 1.0);
  vec3 scr = 1.0 - (1.0 - c) * (1.0 - b);           // screen blend: soft light bloom
  c = mix(c, scr, amount * 0.7);
  c = mix(c, c * 0.9 + 0.1 * b, amount * 0.3);        // and a touch of softness
  c *= vec3(1.0 + 0.06 * warmth, 1.0 + 0.01 * warmth, 1.0 - 0.06 * warmth);
  o = vec4(c, 1.0);
}`,
    // Bloom (like Procreate's): the bright parts of the picture spill soft light around themselves.
    // Transition = how much of the picture counts as "bright" (low: only the brightest highlights);
    // the blurred light is then screened back on, and Burn pushes it further, towards blown-out light.
    bloomPick: `uniform float threshold, knee;
void main() {
  vec3 c = clamp(texture(img, uv).rgb, 0.0, 1.0);
  float l = max(max(c.r, c.g), c.b);
  float w = smoothstep(threshold - knee, threshold + knee, l);
  o = vec4(toLin(c) * w, 1.0);
}`,
    bloom: `uniform sampler2D near; uniform sampler2D far; uniform float amount, burn;
void main() {
  vec3 c = clamp(texture(img, uv).rgb, 0.0, 1.0);
  vec3 light = texture(near, uv).rgb * 0.6 + texture(far, uv).rgb * 0.8;   // a tight core and a wide halo
  light *= amount * (1.6 + 4.0 * burn);
  vec3 lin = toLin(c);
  vec3 scr = 1.0 - (1.0 - lin) * (1.0 - clamp(light, 0.0, 1.0));            // screen: never darkens
  lin = mix(scr, lin + light, burn);                                           // burn: adds light, can blow out
  o = vec4(clamp(toGam(lin), 0.0, 1.0), 1.0);
}`,
    // ◎ Control points (like Nik Color Efex): blend a step's result back into the picture only around the points.
    // pts[i] = (x, y, radius, opacity): x, y in 0..1 from the top left, radius in units of the long side.
    pointMask: `uniform sampler2D orig; uniform vec4 pts[16]; uniform int n; uniform vec2 scale;
void main() {
  float m = 0.0;
  for (int i = 0; i < 16; i++) {
    if (i >= n) break;
    vec4 q = pts[i];
    float d = length((uv - q.xy) * scale) / max(q.z, 1e-4);
    m = max(m, q.w * (1.0 - smoothstep(0.35, 1.0, d)));   // full strength in the middle, a soft edge
  }
  o = mix(texture(orig, uv), texture(img, uv), m);
}`,
    vignette: `uniform float amount, aspect;
void main() {
  vec3 c = texture(img, uv).rgb;
  vec2 d = (uv - 0.5) * vec2(aspect, 1.0) / max(aspect, 1.0) * 2.0;
  float r = length(d) / 1.4142;
  float v = smoothstep(0.35, 1.05, r);
  c = amount >= 0.0 ? c * (1.0 - amount * 0.85 * v) : mix(c, vec3(1.0), -amount * 0.7 * v);
  o = vec4(c, 1.0);
}`,
    grain: `uniform sampler2D noise; uniform float amount; uniform bool colorGrain;
void main() {
  vec3 bg = clamp(texture(img, uv).rgb, 0.0, 1.0);
  vec3 n = texture(noise, uv).rgb;
  if (!colorGrain) n = vec3(n.r);
  else n = mix(vec3(luma(n)), n, 0.35);
  n = (n - 0.5) * amount + 0.5;                       // centred on mid-grey
  vec3 ov = mix(2.0 * bg * n, 1.0 - 2.0 * (1.0 - bg) * (1.0 - n), step(0.5, bg));
  o = vec4(ov, 1.0);
}`,
    // textures hold the picture top row first; the screen wants it the other way up
    // split view: the original left of the line, the adjusted picture right of it
    compare: `uniform sampler2D orig; uniform float split, lineW;
void main() {
  vec2 t = vec2(uv.x, 1.0 - uv.y);
  vec3 c = t.x < split ? texture(orig, t).rgb : texture(img, t).rgb;
  if (abs(t.x - split) < lineW) c = vec3(1.0, 0.68, 0.17);
  o = vec4(clamp(c, 0.0, 1.0), 1.0);
}`,
    final: `uniform bool flipY;
void main() { o = vec4(clamp(texture(img, flipY ? vec2(uv.x, 1.0 - uv.y) : uv).rgb, 0.0, 1.0), 1.0); }`,
  };

  // monotone cubic (Fritsch–Carlson) through the tone-curve points -> 256-entry table
  function curveTable(pts) {
    const n = pts.length, xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const d = [], m = new Array(n).fill(0);
    for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / Math.max(1e-6, xs[i + 1] - xs[i]));
    m[0] = d[0]; m[n - 1] = d[n - 2];
    for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
    for (let i = 0; i < n - 1; i++) {
      if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
      const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
      if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
    }
    const out = new Uint8Array(256 * 4);
    for (let k = 0; k < 256; k++) {
      const x = k / 255;
      let y;
      if (x <= xs[0]) y = ys[0];
      else if (x >= xs[n - 1]) y = ys[n - 1];
      else {
        let i = 0;
        while (x > xs[i + 1]) i++;
        const h = xs[i + 1] - xs[i], t = (x - xs[i]) / h, t2 = t * t, t3 = t2 * t;
        y = (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
      }
      const v = Math.round(Math.min(1, Math.max(0, y)) * 255);
      out.set([v, v, v, 255], k * 4);
    }
    return out;
  }

  // small seeded random generator so the grain is the same in the preview and the saved picture
  function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
  }

  class AdjustEngine {
    constructor(canvas) {
      const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, premultipliedAlpha: false, antialias: false });
      if (!gl) throw new Error("This Mac's graphics don't support WebGL 2, which Adjust needs.");
      this.gl = gl;
      this.canvas = canvas;
      this.float = !!gl.getExtension("EXT_color_buffer_float");
      gl.getExtension("OES_texture_float_linear");
      this.maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      const vb = (this.vb = gl.createBuffer());
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      this.progs = {};
      for (const [k, src] of Object.entries(FS)) this.progs[k] = this._program(HEAD + src);
      this.pool = [];
      this.src = null;
      this.curveTex = this._tex(256, 1, false);
    }

    _program(fs) {
      const gl = this.gl;
      const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
      gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
      gl.bindAttribLocation(p, 0, "pos");
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      return { p, u: {} };
    }

    _tex(w, h, float, data) {
      const gl = this.gl;
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (float) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data || null);
      return { t, w, h, float };
    }

    // render targets are recycled between frames
    _target(w, h, float = this.float) {
      const i = this.pool.findIndex((r) => !r.busy && r.w === w && r.h === h && r.float === float);
      if (i >= 0) { this.pool[i].busy = true; return this.pool[i]; }
      const gl = this.gl;
      const tex = this._tex(w, h, float);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex.t, 0);
      const r = Object.assign(tex, { fb, busy: true });
      this.pool.push(r);
      return r;
    }
    _release(...rs) { for (const r of rs) if (r && r.fb) r.busy = false; }
    _freePool() {
      const gl = this.gl;
      for (const r of this.pool) { gl.deleteFramebuffer(r.fb); gl.deleteTexture(r.t); }
      this.pool = [];
    }

    _pass(name, input, target, uniforms = {}, textures = {}) {
      const gl = this.gl, pr = this.progs[name];
      gl.useProgram(pr.p);
      const loc = (k) => (k in pr.u ? pr.u[k] : (pr.u[k] = gl.getUniformLocation(pr.p, k)));
      let unit = 0;
      const bindTex = (k, tex) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, tex.t);
        gl.uniform1i(loc(k), unit++);
      };
      bindTex("img", input);
      for (const [k, tex] of Object.entries(textures)) bindTex(k, tex);
      for (const [k, v] of Object.entries(uniforms)) {
        const l = loc(k);
        if (l === null) continue;
        if (Array.isArray(v)) gl.uniform2f(l, v[0], v[1]);
        else if (typeof v === "boolean") gl.uniform1i(l, v ? 1 : 0);
        else if (v instanceof Float32Array) gl.uniform4fv(l, v);
        else if (Number.isInteger(v) && (k === "radius" || k === "n")) gl.uniform1i(l, v);
        else gl.uniform1f(l, v);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null);
      gl.viewport(0, 0, target ? target.w : this.canvas.width, target ? target.h : this.canvas.height);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vb);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return target;
    }

    // Gaussian blur with standard deviation `sigma` (pixels of `input`): downsampled first when it's big.
    _blur(input, sigma) {
      let cur = input, w = input.w, h = input.h, ds = 1, owned = null;
      while (sigma / ds > 6 && w > 16 && h > 16) {
        const nw = Math.max(1, Math.ceil(w / 2)), nh = Math.max(1, Math.ceil(h / 2));
        const t = this._target(nw, nh);
        this._pass("down", cur, t, { texel: [1 / w, 1 / h] });
        this._release(owned);
        owned = t; cur = t; w = nw; h = nh; ds *= 2;
      }
      const s = Math.max(0.5, sigma / ds), radius = Math.min(48, Math.ceil(s * 3));
      const a = this._target(w, h), b = this._target(w, h);
      this._pass("blur", cur, a, { dir: [1 / w, 0], sigma: s, radius });
      this._pass("blur", a, b, { dir: [0, 1 / h], sigma: s, radius });
      this._release(owned, a);
      return b;
    }

    _local(input, sigma, amount, midtones = false) {
      if (Math.abs(amount) < 0.001) return input;
      const bl = this._blur(input, sigma);
      const out = this._target(input.w, input.h);
      this._pass("local", input, out, { amount, midtones }, { blurred: bl });
      this._release(bl);
      return out;
    }

    _step(input, name, uniforms, textures) {
      const out = this._target(input.w, input.h);
      this._pass(name, input, out, uniforms, textures);
      return out;
    }

    _grainTexture(w, h, p) {
      const unit = Math.max(w, h) / 1000;
      const cell = unit * (0.5 + 2.5 * p.grainSize);
      const nw = Math.max(2, Math.ceil(w / cell)), nh = Math.max(2, Math.ceil(h / cell));
      const r = rng(1234567 + nw * 31 + nh);
      const data = new Uint8Array(nw * nh * 4);
      for (let i = 0; i < nw * nh; i++) {
        data[i * 4] = r() * 255; data[i * 4 + 1] = r() * 255; data[i * 4 + 2] = r() * 255; data[i * 4 + 3] = 255;
      }
      const gl = this.gl;
      const tex = this._tex(nw, nh, false, data);
      if (p.grainSoftness < 0.15) {  // crisp grain: hard cells
        gl.bindTexture(gl.TEXTURE_2D, tex.t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        return { tex, blurred: null };
      }
      const bl = this._blur(tex, 0.6 * p.grainSoftness);
      return { tex, blurred: bl };
    }

    // Run the whole chain on `src` (a texture); returns the result texture (a pooled target).
    _chain(src, p) {
      const unit = Math.max(src.w, src.h) / 1000;
      let cur = this._step(src, "copy", {});
      const swap = (next) => { if (next !== cur) { this._release(cur); cur = next; } };

      // ◎ a step with control points only shows around its points: keep the picture from before it, then blend
      const pointsFor = (id) => ((p.points || {})[id] || []).filter((q) => q && q.op > 0 && q.r > 0).slice(0, 16);
      const begin = (id) => (pointsFor(id).length ? this._step(cur, "copy", {}) : null);
      const end = (id, before) => {
        if (!before) return;
        const pts = pointsFor(id), arr = new Float32Array(64);
        pts.forEach((q, i) => arr.set([q.x, q.y, q.r, q.op], i * 4));
        const long = Math.max(cur.w, cur.h);
        swap(this._step(cur, "pointMask", { pts: arr, n: pts.length, scale: [cur.w / long, cur.h / long] }, { orig: before }));
        this._release(before);
      };
      const baseBefore = begin("base");

      // 1. Light and Color
      const recovery = p.shadows > 0 || p.highlights < 0;
      const curve = p.shadows < 0 || p.highlights > 0 || p.whites !== 0 || p.blacks !== 0;
      if (p.exposure || recovery || curve || p.contrast || p.saturation || p.temperature || p.tint || p.vibrance) {
        let bl = null;
        if (recovery) bl = this._blur(cur, 12 * unit);
        if (curve) {
          const pts = [[0.08 * Math.max(0, -p.blacks), 0.08 * Math.max(0, p.blacks)],
            [0.25, 0.25 + 0.08 * Math.min(0, p.shadows)], [0.5, 0.5],
            [0.75, 0.75 + 0.08 * Math.max(0, p.highlights)],
            [1 - 0.08 * Math.max(0, p.whites), 1 - 0.08 * Math.max(0, -p.whites)]];
          const gl = this.gl;
          gl.bindTexture(gl.TEXTURE_2D, this.curveTex.t);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, curveTable(pts));
        }
        swap(this._step(cur, "basic", {
          exposure: p.exposure, contrast: p.contrast, saturation: p.saturation, temperature: p.temperature,
          tint: p.tint, vibrance: p.vibrance, shadowLift: Math.max(0, p.shadows), highlightAmt: 1 + Math.min(0, p.highlights),
          useCurve: curve, useRecovery: recovery,
        }, { blurred: bl || cur, curve: this.curveTex }));
        this._release(bl);
      }
      // 2. Presence: Clarity (big-radius midtone contrast), Sharpening (fine), Vignette
      if (p.clarity) swap(this._local(cur, 30 * unit, 0.9 * p.clarity, true));
      if (p.sharpen > 0) swap(this._local(cur, 0.9 * unit, 1.6 * p.sharpen));
      if (p.vignette) swap(this._step(cur, "vignette", { amount: p.vignette, aspect: cur.w / cur.h }));

      end("base", baseBefore);

      // 3. Filters, in the stacked order
      for (const id of fullOrder(p.order)) {
        const on = { extract: p.extractOn, dark: p.darkOn, bleach: p.bleachOn, glow: p.glowOn, bloom: p.bloomOn, grain: p.grainOn }[id];
        const before = on ? begin(id) : null;
        if (id === "extract" && p.extractOn) {
          const r = [2, 5, 14][Math.min(2, Math.max(0, p.deRadius | 0))] * unit;
          const bl = this._blur(cur, 12 * unit);
          swap(this._step(cur, "recover", { shadowLift: 0.5 * p.deAmount, highlightAmt: 1 - 0.3 * p.deAmount }, { blurred: bl }));
          this._release(bl);
          swap(this._local(cur, r, 1.6 * p.deAmount));
          swap(this._step(cur, "controls", { contrast: 1 + 0.3 * p.deContrast, sat: Math.max(0, 1 + p.deSaturation), bright: 0, gamma: 1 }));
        } else if (id === "dark" && p.darkOn) {
          swap(this._local(cur, 5 * unit, 1.4 * p.dcStrength));
          swap(this._step(cur, "controls", { contrast: 1, sat: 1, bright: 0, gamma: Math.max(0.4, 1 + 0.35 * p.dcStrength - 0.45 * p.dcBrightness) }));
          const bl = this._blur(cur, 12 * unit);
          swap(this._step(cur, "recover", { shadowLift: 0.4 * p.dcStrength, highlightAmt: 1 }, { blurred: bl }));
          this._release(bl);
          swap(this._step(cur, "controls", { contrast: 1 + 0.12 * p.dcStrength, sat: Math.max(0, 0.8 + 0.8 * p.dcSaturation), bright: 0, gamma: 1 }));
        } else if (id === "bleach" && p.bleachOn) {
          swap(this._step(cur, "bleach", {}));
          swap(this._step(cur, "controls", { sat: Math.max(0, 0.5 + 0.5 * p.bbSaturation), contrast: Math.max(0.25, 1 + 0.45 * p.bbContrast), bright: 0.2 * p.bbBrightness, gamma: 1 }));
          swap(this._local(cur, 22 * unit, 1.4 * p.bbLocalContrast));
        } else if (id === "glow" && p.glowOn) {
          const bl = this._blur(cur, 10 * unit);
          swap(this._step(cur, "glow", { amount: p.ggGlow, warmth: p.ggWarmth }, { blurred: bl }));
          this._release(bl);
          if (p.ggSaturation) swap(this._step(cur, "controls", { contrast: 1, sat: Math.max(0, 1 + p.ggSaturation), bright: 0, gamma: 1 }));
        } else if (id === "bloom" && p.bloomOn && p.blAmount > 0) {
          const t = Math.min(1, Math.max(0, p.blTransition));
          const pick = this._step(cur, "bloomPick", { threshold: 0.92 - 0.7 * t, knee: 0.04 + 0.2 * t });
          const r = (4 + 56 * Math.min(1, Math.max(0, p.blSize))) * unit;
          const near = this._blur(pick, r), far = this._blur(pick, r * 3.5);
          this._release(pick);
          swap(this._step(cur, "bloom", { amount: p.blAmount, burn: Math.min(1, Math.max(0, p.blBurn)) }, { near, far }));
          this._release(near, far);
        } else if (id === "grain" && p.grainOn) {
          const g = this._grainTexture(cur.w, cur.h, p);
          swap(this._step(cur, "grain", { amount: 0.7 * p.grainAmount, colorGrain: !!p.grainColor }, { noise: g.blurred || g.tex }));
          this._release(g.blurred);
          this.gl.deleteTexture(g.tex.t);
        }
        end(id, before);
      }
      return cur;
    }

    // ------------------------------------------------------------ public
    // Load the picture. `bitmap` = ImageBitmap of the full picture; the preview uses a smaller copy.
    async setImage(bitmap, previewMax = 1600) {
      this.full = bitmap;
      const s = Math.min(1, previewMax / Math.max(bitmap.width, bitmap.height));
      const pw = Math.max(1, Math.round(bitmap.width * s)), ph = Math.max(1, Math.round(bitmap.height * s));
      const small = s < 1 ? await window.createImageBitmap(bitmap, { resizeWidth: pw, resizeHeight: ph, resizeQuality: "high" }) : bitmap;
      if (this.src) this.gl.deleteTexture(this.src.t);
      this.src = this._upload(small);
      this._freePool();
      this.canvas.width = pw;
      this.canvas.height = ph;
    }

    _upload(bitmap) {
      const gl = this.gl;
      const t = this._tex(bitmap.width, bitmap.height, false);
      gl.bindTexture(gl.TEXTURE_2D, t.t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);  // top row first
      return t;
    }

    // Draw the preview. view: "after" (adjusted), "before" (the original) or "split" (original left of `split`, 0..1).
    render(p, view = "after", split = 0.5) {
      if (!this.src) return;
      if (view === true) view = "before";
      if (view === "before" || !anyOn(p)) {
        if (view !== "split") { this._pass("final", this.src, null, { flipY: true }); return; }
      }
      const out = anyOn(p) ? this._chain(this.src, p) : this.src;
      if (view === "split") this._pass("compare", out, null, { split, lineW: 1.5 / this.canvas.width }, { orig: this.src });
      else this._pass("final", out, null, { flipY: true });
      this._release(out);
    }

    // Apply to the full-size picture -> PNG Blob.
    async exportPNG(p) {
      const full = this.full;
      if (Math.max(full.width, full.height) > this.maxSize)
        throw new Error(`The picture is bigger than this Mac's graphics can filter (${this.maxSize} px).`);
      const src = this._upload(full);
      const saved = this.pool;
      this.pool = [];
      try {
        const out = this._chain(src, p);
        const fin = this._target(src.w, src.h, false);
        this._pass("final", out, fin, { flipY: false });
        const gl = this.gl;
        const px = new Uint8ClampedArray(src.w * src.h * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fin.fb);
        gl.readPixels(0, 0, src.w, src.h, gl.RGBA, gl.UNSIGNED_BYTE, px);  // top row first, like the texture
        const c = new OffscreenCanvas(src.w, src.h);
        c.getContext("2d").putImageData(new ImageData(px, src.w, src.h), 0, 0);
        return await c.convertToBlob({ type: "image/png" });
      } finally {
        this._freePool();
        this.pool = saved;
        this.gl.deleteTexture(src.t);
      }
    }

    destroy() {
      this._freePool();
      const gl = this.gl;
      if (this.src) gl.deleteTexture(this.src.t);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    }
  }

  window.ProtoAdjust = { AdjustEngine, defaults, withDefaults, fullOrder, LOOKS, FILTER_ORDER, isDefault, anyOn };
})();
