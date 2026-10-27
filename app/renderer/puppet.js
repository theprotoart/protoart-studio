// 🪡 Puppet warp (Tools), like Photoshop's: pins on the picture; drag a pin and the picture bends around it, while
// the other pins hold their places. The bend is a "moving least squares" rigid deformation (Schaefer, McPhail &
// Warren, 2006) worked out on a grid of points and drawn on the GPU as a textured mesh, so it updates live.
// Pins are kept in picture units (0..1 from the top left), so the small preview and the full-size save match.
/* global window */
(function () {
  "use strict";

  // Where a point of the picture moves to, given the pins: rest positions p (where you put them) and targets q
  // (where you dragged them). "Moving least squares" (Schaefer, McPhail & Warren, 2006), three kinds, like
  // Photoshop's modes: "rigid" (parts bend and turn but don't stretch: arms, legs), "normal" (they may also grow or
  // shrink evenly) and "distort" (stretchy: they may stretch and slant too).
  function mls(vx, vy, p, q, aspect, mode) {
    const n = p.length;
    if (!n) return [vx, vy];
    // distances in a square space, so a round area of influence stays round on a wide picture
    let sw = 0, px = 0, py = 0, qx = 0, qy = 0;
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const dx = (p[i][0] - vx) * aspect, dy = p[i][1] - vy;
      const d2 = dx * dx + dy * dy;
      if (d2 < 1e-12) return [q[i][0], q[i][1]];
      w[i] = (p[i][2] || 1) / d2;
      sw += w[i];
      px += w[i] * p[i][0] * aspect; py += w[i] * p[i][1];
      qx += w[i] * q[i][0] * aspect; qy += w[i] * q[i][1];
    }
    px /= sw; py /= sw; qx /= sw; qy /= sw;
    if (n === 1) return [vx + (qx - px) / aspect, vy + (qy - py)];  // one pin: the whole picture moves with it
    const vxa = vx * aspect - px, vya = vy - py;
    if (mode === "distort") {  // affine: v̂ (Σ w p̂ᵀp̂)⁻¹ Σ w p̂ᵀq̂
      let a = 0, b = 0, d = 0, m00 = 0, m01 = 0, m10 = 0, m11 = 0;
      for (let i = 0; i < n; i++) {
        const phx = p[i][0] * aspect - px, phy = p[i][1] - py, qhx = q[i][0] * aspect - qx, qhy = q[i][1] - qy;
        a += w[i] * phx * phx; b += w[i] * phx * phy; d += w[i] * phy * phy;
        m00 += w[i] * phx * qhx; m01 += w[i] * phx * qhy; m10 += w[i] * phy * qhx; m11 += w[i] * phy * qhy;
      }
      const det = a * d - b * b;
      if (Math.abs(det) > 1e-12 * Math.max(1e-12, a * d)) {
        const ia = d / det, ib = -b / det, id = a / det;   // inverse of [[a b][b d]]
        const ux = vxa * ia + vya * ib, uy = vxa * ib + vya * id;
        return [(ux * m00 + uy * m10 + qx) / aspect, ux * m01 + uy * m11 + qy];
      }
      mode = "normal";  // all the pins on one line: no way to know the slant, fall back
    }
    let fx = 0, fy = 0, mu = 0;
    for (let i = 0; i < n; i++) {
      const phx = p[i][0] * aspect - px, phy = p[i][1] - py;
      const qhx = q[i][0] * aspect - qx, qhy = q[i][1] - qy;
      // A_i = w_i [p̂; -p̂⊥] [v-p*; -(v-p*)⊥]ᵀ, applied to q̂_i
      const a = phx * vxa + phy * vya, b = phx * vya - phy * vxa;
      fx += w[i] * (qhx * a - qhy * b);
      fy += w[i] * (qhx * b + qhy * a);
      mu += w[i] * (phx * phx + phy * phy);
    }
    if (mode === "normal") return [(fx / mu + qx) / aspect, fy / mu + qy];
    const len = Math.hypot(fx, fy), dist = Math.hypot(vxa, vya);
    if (len < 1e-12) return [vx + (qx - px) / aspect, vy + (qy - py)];
    return [(dist * fx / len + qx) / aspect, dist * fy / len + qy];
  }
  function mlsRigid(vx, vy, p, q, aspect) { return mls(vx, vy, p, q, aspect, "rigid"); }

  // The pins as control points. A pin you turned (its `rot`, in radians) also gets four close helper points around
  // it that turn with it, so the part around it pivots there like a joint (Photoshop: ⌥-drag near a pin).
  const SAT = 0.035;  // how far the helper points sit from the pin (part of the picture's height)
  function controls(pins, aspect) {
    const p = [], q = [];
    for (const k of pins) {
      p.push([k.x, k.y, 1]); q.push([k.tx, k.ty]);
      if (k.rot == null || !isFinite(k.rot)) continue;
      const c = Math.cos(k.rot), s = Math.sin(k.rot);
      for (const [dx, dy] of [[SAT, 0], [-SAT, 0], [0, SAT], [0, -SAT]]) {
        p.push([k.x + dx / aspect, k.y + dy, 0.6]);
        q.push([k.tx + (dx * c - dy * s) / aspect, k.ty + (dx * s + dy * c)]);
      }
    }
    return [p, q];
  }

  // ✂ The subject's edge: grow or shrink it (Expansion, in pixels) and soften it (Feather, in pixels).
  // `matte`: a bitmap/canvas, white = the subject. Returns a canvas of size w x h.
  function shapeMatte(matte, w, h, expand, feather) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(matte, 0, 0, w, h);
    const e = Math.round(expand || 0);
    if (e) {
      const d = ctx.getImageData(0, 0, w, h), px = d.data, a = new Uint8ClampedArray(w * h);
      for (let i = 0; i < a.length; i++) a[i] = px[i * 4];
      const out = e > 0 ? growMax(a, w, h, e) : growMin(a, w, h, -e);
      for (let i = 0; i < a.length; i++) { px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = out[i]; px[i * 4 + 3] = 255; }
      ctx.putImageData(d, 0, 0);
    }
    const f = Math.max(0, feather || 0);
    if (f > 0.3) {
      const c2 = document.createElement("canvas");
      c2.width = w; c2.height = h;
      const x2 = c2.getContext("2d");
      x2.fillStyle = "#000"; x2.fillRect(0, 0, w, h);
      x2.filter = `blur(${f / 2}px)`;  // a gaussian whose spread is half the feather: the edge fades over ~the feather
      x2.drawImage(c, 0, 0);
      return c2;
    }
    return c;
  }
  // a round brush's worth of growing (max) or shrinking (min): rows then columns, each a sliding window
  function slide(a, w, h, r, pick, horiz) {
    const out = new Uint8ClampedArray(a.length), len = horiz ? w : h, lines = horiz ? h : w;
    const at = horiz ? (l, i) => l * w + i : (l, i) => i * w + l;
    const dq = new Int32Array(len);
    for (let l = 0; l < lines; l++) {
      let head = 0, tail = 0;
      for (let i = 0, j = 0; i < len; i++) {
        for (; j < len && j <= i + r; j++) {
          const v = a[at(l, j)];
          while (tail > head && !pick(a[at(l, dq[tail - 1])], v)) tail--;
          dq[tail++] = j;
        }
        while (dq[head] < i - r) head++;
        out[at(l, i)] = a[at(l, dq[head])];
      }
    }
    return out;
  }
  const growMax = (a, w, h, r) => slide(slide(a, w, h, r, (x, y) => x > y, true), w, h, r, (x, y) => x > y, false);
  const growMin = (a, w, h, r) => slide(slide(a, w, h, r, (x, y) => x < y, true), w, h, r, (x, y) => x < y, false);

  const VS = `#version 300 es
in vec2 pos; in vec2 tc; out vec2 uv; uniform bool flipY;
void main() { uv = tc; vec2 c = pos * 2.0 - 1.0; gl_Position = vec4(c.x, flipY ? -c.y : c.y, 0.0, 1.0); }`;
  // With a subject: only the subject (its mask, carried by the bend) is drawn, over the filled-in background.
  const FS = `#version 300 es
precision highp float; in vec2 uv; out vec4 o; uniform sampler2D img; uniform sampler2D matte; uniform bool useMatte; uniform float alpha;
void main() { o = vec4(texture(img, uv).rgb, useMatte ? texture(matte, uv).r : alpha); }`;

  class PuppetEngine {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext("webgl2", { premultipliedAlpha: false, preserveDrawingBuffer: true });
      if (!gl) throw new Error("This Mac's graphics can't run the warp preview (WebGL 2 is off).");
      this.gl = gl;
      const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      const pr = gl.createProgram();
      gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS));
      gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS));
      gl.bindAttribLocation(pr, 0, "pos");
      gl.bindAttribLocation(pr, 1, "tc");
      gl.linkProgram(pr);
      if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
      this.prog = pr;
      this.posBuf = gl.createBuffer();
      this.tcBuf = gl.createBuffer();
      this.idxBuf = gl.createBuffer();
      this.quadBuf = gl.createBuffer();
      this.maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      this.grid = 48;
      this.mode = "rigid";
      this.edge = { expand: 0, feather: 0 };  // the subject's edge, in pixels of the full-size picture
      this.hideBg = false;
    }

    setMode(m) { this.mode = ["rigid", "normal", "distort"].includes(m) ? m : "rigid"; }

    // the subject's mask at the preview's size, shaped (grown / shrunk, feathered), as a texture
    _shapePreview() {
      const gl = this.gl;
      if (this.matteTex) { gl.deleteTexture(this.matteTex); this.matteTex = null; }
      this.cellIn = null;
      if (!this.matteFull) return;
      const pw = this.canvas.width, ph = this.canvas.height, k = pw / this.full.width;
      const shaped = shapeMatte(this.matteFull, pw, ph, this.edge.expand * k, this.edge.feather * k);
      this.matteTex = this._upload(shaped);
      this._cells(shaped);
    }

    // which squares of the grid the subject covers (the mesh is shown, like Photoshop's, only over the subject)
    _cells(shaped) {
      const n = this.grid, c = document.createElement("canvas");
      c.width = n * 2; c.height = n * 2;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(shaped, 0, 0, n * 2, n * 2);
      const px = ctx.getImageData(0, 0, n * 2, n * 2).data, cin = new Uint8Array(n * n);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        let m = 0;
        for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) m = Math.max(m, px[((j * 2 + y) * n * 2 + i * 2 + x) * 4]);
        cin[j * n + i] = m > 40 ? 1 : 0;
      }
      this.cellIn = cin;
    }

    setEdge(expand, feather) {
      this.edge = { expand: +expand || 0, feather: +feather || 0 };
      this._shapePreview();
    }

    // ✂ Subject: `matte` = a canvas/bitmap (white = the subject), `background` = the picture with the subject filled
    // in. Then only the subject bends; the background stays put and shows where the subject moved away from.
    async setSubject(matte, background) {
      const gl = this.gl;
      for (const k of ["matteTex", "bgTex"]) if (this[k]) { gl.deleteTexture(this[k]); this[k] = null; }
      this.matteFull = matte || null;
      this.bgFull = background || null;
      this.cellIn = null;
      if (!matte || !background) { this.matteFull = this.bgFull = null; return; }
      const pw = this.canvas.width, ph = this.canvas.height;
      const small = async (b) => (b.width === pw && b.height === ph ? b
        : await window.createImageBitmap(b, { resizeWidth: pw, resizeHeight: ph, resizeQuality: "high" }));
      this.bgTex = this._upload(await small(background));
      this._shapePreview();
    }

    async setImage(bitmap, previewMax = 1600) {
      this.full = bitmap;
      const s = Math.min(1, previewMax / Math.max(bitmap.width, bitmap.height));
      const pw = Math.max(1, Math.round(bitmap.width * s)), ph = Math.max(1, Math.round(bitmap.height * s));
      const small = s < 1 ? await window.createImageBitmap(bitmap, { resizeWidth: pw, resizeHeight: ph, resizeQuality: "high" }) : bitmap;
      if (this.tex) this.gl.deleteTexture(this.tex);
      this.tex = this._upload(small);
      this.canvas.width = pw;
      this.canvas.height = ph;
      this._mesh();
    }

    _upload(bitmap) {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);  // top row first: v=0 is the top
      return t;
    }

    // the grid of points (and triangles) the picture is drawn on; more points = smoother bends
    _mesh() {
      const gl = this.gl, n = this.grid, tc = new Float32Array((n + 1) * (n + 1) * 2), idx = [];
      for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) { tc[(j * (n + 1) + i) * 2] = i / n; tc[(j * (n + 1) + i) * 2 + 1] = j / n; }
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
        idx.push(a, b, c, b, d, c);
      }
      this.rest = tc;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tcBuf);
      gl.bufferData(gl.ARRAY_BUFFER, tc, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
      this.count = idx.length;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    }

    setDensity(n) {
      this.grid = Math.max(8, Math.min(96, n | 0));
      if (this.tex) this._mesh();
      if (this.matteFull) this._shapePreview();
    }

    // the bent mesh, to draw over the picture: points (0..1) on the (grid+1)² grid, and which squares to show
    mesh(pins) {
      if (!this.rest) return null;
      return { n: this.grid, pts: this._deformed(pins, this.canvas.width, this.canvas.height), cells: this.cellIn };
    }

    // pins: [{x, y, tx, ty}] in 0..1 (x, y = where you put it; tx, ty = where it is now)
    _deformed(pins, w, h) {
      const aspect = w / h, [p, q] = controls(pins, aspect), mode = this.mode;
      const out = new Float32Array(this.rest.length);
      for (let i = 0; i < this.rest.length; i += 2) {
        const [x, y] = mls(this.rest[i], this.rest[i + 1], p, q, aspect, mode);
        out[i] = x; out[i + 1] = y;
      }
      return out;
    }

    _draw(tex, pins, w, h, fb, flipY, showOriginal, layers) {
      const gl = this.gl;
      const sub = layers && layers.matte && layers.bg ? layers : null;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(this.prog, "img"), 0);
      gl.uniform1i(gl.getUniformLocation(this.prog, "flipY"), flipY ? 1 : 0);
      gl.uniform1i(gl.getUniformLocation(this.prog, "matte"), 1);
      const useMatte = gl.getUniformLocation(this.prog, "useMatte");
      gl.uniform1i(useMatte, 0);
      const alpha = gl.getUniformLocation(this.prog, "alpha");
      // underneath: the picture as it was, so nothing is left empty where the bent part moved away from;
      // with a subject, the background with the subject filled in (and the original only while comparing)
      gl.uniform1f(alpha, 1);
      if (sub && !showOriginal) gl.bindTexture(gl.TEXTURE_2D, sub.bg);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      if (sub && sub.hideBg && !showOriginal) {  // the background switched off: only the subject, on see-through
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
      } else gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      if (showOriginal) return;
      if (sub) {  // the subject on top, cut out by its mask (soft edge) and bent with the pins
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, sub.matte);
        gl.activeTexture(gl.TEXTURE0);
        gl.uniform1i(useMatte, 1);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this._deformed(pins, w, h), gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tcBuf);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
      gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
      if (sub) { gl.disable(gl.BLEND); gl.uniform1i(useMatte, 0); }
    }

    render(pins, showOriginal = false) {
      if (!this.tex) return;
      this._draw(this.tex, pins, this.canvas.width, this.canvas.height, null, true, showOriginal,
                 this.matteTex ? { matte: this.matteTex, bg: this.bgTex, hideBg: this.hideBg } : null);
    }

    // the full-size picture, warped -> PNG Blob
    async exportPNG(pins) {
      const full = this.full, gl = this.gl;
      if (Math.max(full.width, full.height) > this.maxSize)
        throw new Error(`The picture is bigger than this Mac's graphics can warp (${this.maxSize} px).`);
      const tex = this._upload(full);
      let layers = null;
      if (this.matteFull && this.bgFull) {  // the subject's mask and the filled background, at full size
        const fit = async (b) => (b.width === full.width && b.height === full.height ? b
          : await window.createImageBitmap(b, { resizeWidth: full.width, resizeHeight: full.height, resizeQuality: "high" }));
        const shaped = shapeMatte(this.matteFull, full.width, full.height, this.edge.expand, this.edge.feather);
        layers = { matte: this._upload(shaped), bg: this._upload(await fit(this.bgFull)) };
      }
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, full.width, full.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      try {
        this._draw(tex, pins, full.width, full.height, fb, false, false, layers);
        const px = new Uint8ClampedArray(full.width * full.height * 4);
        gl.readPixels(0, 0, full.width, full.height, gl.RGBA, gl.UNSIGNED_BYTE, px);  // row 0 = the top (drawn unflipped)
        const c = new OffscreenCanvas(full.width, full.height);
        c.getContext("2d").putImageData(new ImageData(px, full.width, full.height), 0, 0);
        return await c.convertToBlob({ type: "image/png" });
      } finally {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fb);
        gl.deleteTexture(t);
        gl.deleteTexture(tex);
        if (layers) { gl.deleteTexture(layers.matte); gl.deleteTexture(layers.bg); }
      }
    }

    destroy() {
      const gl = this.gl;
      if (this.tex) gl.deleteTexture(this.tex);
      for (const k of ["matteTex", "bgTex"]) if (this[k]) gl.deleteTexture(this[k]);
      for (const b of [this.posBuf, this.tcBuf, this.idxBuf, this.quadBuf]) gl.deleteBuffer(b);
      gl.deleteProgram(this.prog);
    }
  }

  window.ProtoPuppet = { PuppetEngine, mlsRigid, mls, controls, shapeMatte,
    grow: (a, w, h, r) => (r > 0 ? growMax(a, w, h, r) : r < 0 ? growMin(a, w, h, -r) : a) };
})();
