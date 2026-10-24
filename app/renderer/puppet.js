// 🪡 Puppet warp (Tools), like Photoshop's: pins on the picture; drag a pin and the picture bends around it, while
// the other pins hold their places. The bend is a "moving least squares" rigid deformation (Schaefer, McPhail &
// Warren, 2006) worked out on a grid of points and drawn on the GPU as a textured mesh, so it updates live.
// Pins are kept in picture units (0..1 from the top left), so the small preview and the full-size save match.
/* global window */
(function () {
  "use strict";

  // Where a point of the picture moves to, given the pins: rest positions p (where you put them) and targets q
  // (where you dragged them). Rigid: parts bend and turn but don't stretch or squash.
  function mlsRigid(vx, vy, p, q, aspect) {
    const n = p.length;
    if (!n) return [vx, vy];
    // distances in a square space, so a round area of influence stays round on a wide picture
    let sw = 0, px = 0, py = 0, qx = 0, qy = 0;
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const dx = (p[i][0] - vx) * aspect, dy = p[i][1] - vy;
      const d2 = dx * dx + dy * dy;
      if (d2 < 1e-12) return [q[i][0], q[i][1]];
      w[i] = 1 / d2;
      sw += w[i];
      px += w[i] * p[i][0] * aspect; py += w[i] * p[i][1];
      qx += w[i] * q[i][0] * aspect; qy += w[i] * q[i][1];
    }
    px /= sw; py /= sw; qx /= sw; qy /= sw;
    if (n === 1) return [vx + (qx - px) / aspect, vy + (qy - py)];  // one pin: the whole picture moves with it
    const vxa = vx * aspect - px, vya = vy - py;
    let fx = 0, fy = 0;
    for (let i = 0; i < n; i++) {
      const phx = p[i][0] * aspect - px, phy = p[i][1] - py;
      const qhx = q[i][0] * aspect - qx, qhy = q[i][1] - qy;
      // A_i = w_i [p̂; -p̂⊥] [v-p*; -(v-p*)⊥]ᵀ, applied to q̂_i
      const a = phx * vxa + phy * vya, b = phx * vya - phy * vxa;
      fx += w[i] * (qhx * a - qhy * b);
      fy += w[i] * (qhx * b + qhy * a);
    }
    const len = Math.hypot(fx, fy), dist = Math.hypot(vxa, vya);
    if (len < 1e-12) return [vx + (qx - px) / aspect, vy + (qy - py)];
    return [(dist * fx / len + qx) / aspect, dist * fy / len + qy];
  }

  const VS = `#version 300 es
in vec2 pos; in vec2 tc; out vec2 uv; uniform bool flipY;
void main() { uv = tc; vec2 c = pos * 2.0 - 1.0; gl_Position = vec4(c.x, flipY ? -c.y : c.y, 0.0, 1.0); }`;
  const FS = `#version 300 es
precision highp float; in vec2 uv; out vec4 o; uniform sampler2D img; uniform float alpha;
void main() { o = vec4(texture(img, uv).rgb, alpha); }`;

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
    }

    // pins: [{x, y, tx, ty}] in 0..1 (x, y = where you put it; tx, ty = where it is now)
    _deformed(pins, w, h) {
      const p = pins.map((k) => [k.x, k.y]), q = pins.map((k) => [k.tx, k.ty]);
      const out = new Float32Array(this.rest.length), aspect = w / h;
      for (let i = 0; i < this.rest.length; i += 2) {
        const [x, y] = mlsRigid(this.rest[i], this.rest[i + 1], p, q, aspect);
        out[i] = x; out[i + 1] = y;
      }
      return out;
    }

    _draw(tex, pins, w, h, fb, flipY, showOriginal) {
      const gl = this.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(this.prog, "img"), 0);
      gl.uniform1i(gl.getUniformLocation(this.prog, "flipY"), flipY ? 1 : 0);
      const alpha = gl.getUniformLocation(this.prog, "alpha");
      // underneath: the picture as it was, so nothing is left empty where the bent part moved away from
      gl.uniform1f(alpha, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      if (showOriginal) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this._deformed(pins, w, h), gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tcBuf);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
      gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
    }

    render(pins, showOriginal = false) {
      if (!this.tex) return;
      this._draw(this.tex, pins, this.canvas.width, this.canvas.height, null, true, showOriginal);
    }

    // the full-size picture, warped -> PNG Blob
    async exportPNG(pins) {
      const full = this.full, gl = this.gl;
      if (Math.max(full.width, full.height) > this.maxSize)
        throw new Error(`The picture is bigger than this Mac's graphics can warp (${this.maxSize} px).`);
      const tex = this._upload(full);
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, full.width, full.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      try {
        this._draw(tex, pins, full.width, full.height, fb, false, false);
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
      }
    }

    destroy() {
      const gl = this.gl;
      if (this.tex) gl.deleteTexture(this.tex);
      for (const b of [this.posBuf, this.tcBuf, this.idxBuf, this.quadBuf]) gl.deleteBuffer(b);
      gl.deleteProgram(this.prog);
    }
  }

  window.ProtoPuppet = { PuppetEngine, mlsRigid };
})();
