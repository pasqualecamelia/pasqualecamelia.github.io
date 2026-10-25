// ── QGT Hydrogen 3D — LIVE split-operator engine (64³, plain JS) ──────────
// iħ∂Ψ/∂t = [-∇²/2 - 1/√(r²+ε²) + α·M_eff·sech²(r/ℓ_pen)
//            + f_t·cos(2Γτ t)·e^(−r²/2)] Ψ      (a.u., ħ=m=1)
// Parameters printed in the section header; all constants structural:
//   Γτ = φ/√5, a0_QGT = Γτ², f_t = φ²/10, ℓ_pen = 1/(π·α_phys), M_eff = 1.
// Softening ε = dx/2 (grid regularisation of the Coulomb singularity; stated).
// Initial state: (|1s⟩ + |2p_z⟩ + |3d_z²⟩)/√3 with hydrogenic a0 = a0_QGT.
// Method: Strang splitting  e^{-iV dt/2} · F⁻¹ e^{-ik²dt/2} F · e^{-iV dt/2}
// — unitary up to FFT roundoff: norm conserved to ~1e-13/step.

(function (global) {
  'use strict';

  var PHI = (1 + Math.sqrt(5)) / 2;
  var GAMMA_TAU = PHI / Math.sqrt(5);          // 0.7236067977...
  var ALPHA_INV_PHYS = 137.035999170;
  var ALPHA = 1 / ALPHA_INV_PHYS;

  var CFG = {
    N: 64,
    dx: 0.375,                                  // grid step (a.u.) → box 24 a.u.
    dt: 0.24,                                   // a.u. per step (as published)
    a0: GAMMA_TAU * GAMMA_TAU,                  // 0.5236068 — a0_QGT
    f_t: PHI * PHI / 10,                        // 0.2618034 — breathing amplitude
    M_eff: 1.0,
    ell_pen: ALPHA_INV_PHYS / Math.PI,          // 1/(π α) ≈ 43.62 (a.u.)
    eps: 0.375 / 2                              // Coulomb softening = dx/2
  };

  // ---- tiny iterative radix-2 complex FFT, length N (power of 2) ----------
  function makeFFT(N) {
    var levels = Math.log2(N) | 0;
    var rev = new Uint32Array(N);
    for (var i = 0; i < N; i++) {
      var x = i, r = 0;
      for (var b = 0; b < levels; b++) { r = (r << 1) | (x & 1); x >>= 1; }
      rev[i] = r;
    }
    var cosT = new Float64Array(N / 2), sinT = new Float64Array(N / 2);
    for (i = 0; i < N / 2; i++) {
      cosT[i] = Math.cos(2 * Math.PI * i / N);
      sinT[i] = Math.sin(2 * Math.PI * i / N);
    }
    // in-place FFT on (re,im) with stride access handled by caller (contiguous line)
    return function fft(re, im, inverse) {
      var n = N, j, k, len, half, tablestep;
      for (i = 0; i < n; i++) {
        j = rev[i];
        if (j > i) {
          var tr = re[i]; re[i] = re[j]; re[j] = tr;
          var ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
      }
      for (len = 2; len <= n; len <<= 1) {
        half = len >> 1; tablestep = n / len;
        for (i = 0; i < n; i += len) {
          for (j = i, k = 0; j < i + half; j++, k += tablestep) {
            var c = cosT[k], s = inverse ? sinT[k] : -sinT[k];
            var l = j + half;
            var xr = re[l] * c - im[l] * s;
            var xi = re[l] * s + im[l] * c;
            re[l] = re[j] - xr; im[l] = im[j] - xi;
            re[j] += xr;        im[j] += xi;
          }
        }
      }
      if (inverse) for (i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    };
  }

  function Engine() {
    var N = CFG.N, N2 = N * N, N3 = N * N * N, dx = CFG.dx;
    this.N = N; this.cfg = CFG;
    this.re = new Float64Array(N3);
    this.im = new Float64Array(N3);
    this.t = 0; this.steps = 0;

    // coordinates: x_i = (i - N/2 + 0.5)·dx  → no grid point at r = 0
    var xs = new Float64Array(N);
    for (var i = 0; i < N; i++) xs[i] = (i - N / 2 + 0.5) * dx;
    this.xs = xs;

    // static potential and gaussian (breathing envelope), precomputed
    this.Vstat = new Float64Array(N3);
    this.gauss = new Float64Array(N3);
    var eps2 = CFG.eps * CFG.eps;
    var idx = 0;
    for (var a = 0; a < N; a++) for (var b = 0; b < N; b++) for (var c = 0; c < N; c++, idx++) {
      var r2 = xs[a] * xs[a] + xs[b] * xs[b] + xs[c] * xs[c];
      var r = Math.sqrt(r2);
      this.Vstat[idx] = -1 / Math.sqrt(r2 + eps2)
        + ALPHA * CFG.M_eff / Math.pow(Math.cosh(r / CFG.ell_pen), 2);
      this.gauss[idx] = Math.exp(-r2 / 2);
    }

    // kinetic phase per step: exp(-i k²/2 · dt)
    var ks = new Float64Array(N);
    for (i = 0; i < N; i++) {
      var f = (i <= N / 2 ? i : i - N) * (2 * Math.PI / (N * dx));
      ks[i] = f;
    }
    this.kcos = new Float64Array(N3); this.ksin = new Float64Array(N3);
    idx = 0;
    for (a = 0; a < N; a++) for (b = 0; b < N; b++) for (c = 0; c < N; c++, idx++) {
      var k2 = ks[a] * ks[a] + ks[b] * ks[b] + ks[c] * ks[c];
      var ph = -0.5 * k2 * CFG.dt;
      this.kcos[idx] = Math.cos(ph); this.ksin[idx] = Math.sin(ph);
    }

    this.fft = makeFFT(N);
    this.lre = new Float64Array(N); this.lim = new Float64Array(N);
    this.initState();
  }

  Engine.prototype.initState = function () {
    var N = this.N, xs = this.xs, a0 = CFG.a0;
    var idx = 0;
    for (var a = 0; a < N; a++) for (var b = 0; b < N; b++) for (var c = 0; c < N; c++, idx++) {
      var x = xs[a], y = xs[b], z = xs[c];
      var r = Math.sqrt(x * x + y * y + z * z);
      var psi1s  = Math.exp(-r / a0);                          // ∝ e^{-r/a0}
      var psi2pz = z * Math.exp(-r / (2 * a0));                // ∝ z e^{-r/2a0}
      var psi3dz = (3*z*z - r*r) * Math.exp(-r / (3 * a0));    // ∝ (3z²−r²) e^{-r/3a0}
      this.re[idx] = psi1s; this.im[idx] = 0;
      this._tmp2p = this._tmp2p || new Float64Array(N * N * N);
      this._tmp3d = this._tmp3d || new Float64Array(N * N * N);
      this._tmp2p[idx] = psi2pz;
      this._tmp3d[idx] = psi3dz;
    }
    // normalise each component on the grid, then superpose 1/√2 each
    var n1 = 0, n2 = 0, n3 = 0, N3 = N * N * N;
    for (idx = 0; idx < N3; idx++) {
      n1 += this.re[idx] * this.re[idx];
      n2 += this._tmp2p[idx] * this._tmp2p[idx];
      n3 += this._tmp3d[idx] * this._tmp3d[idx];
    }
    n1 = Math.sqrt(n1); n2 = Math.sqrt(n2); n3 = Math.sqrt(n3);
    var s = 1 / Math.sqrt(3);
    for (idx = 0; idx < N3; idx++) {
      this.re[idx] = s * (this.re[idx] / n1 + this._tmp2p[idx] / n2 + this._tmp3d[idx] / n3);
      this.im[idx] = 0;
    }
    // grid-normalise the superposition (1s/2pz not exactly orthogonal on a finite grid)
    var nn = 0;
    for (idx = 0; idx < N3; idx++) nn += this.re[idx] * this.re[idx];
    nn = Math.sqrt(nn);
    for (idx = 0; idx < N3; idx++) this.re[idx] /= nn;
    this.t = 0; this.steps = 0;
  };

  Engine.prototype._applyV = function (halfdt) {
    // V(t) = Vstat + f_t·cos(2Γτ t)·gauss   (time read at current t)
    var mod = CFG.f_t * Math.cos(2 * GAMMA_TAU * this.t);
    var N3 = this.N * this.N * this.N;
    for (var idx = 0; idx < N3; idx++) {
      var ph = -(this.Vstat[idx] + mod * this.gauss[idx]) * halfdt;
      var c = Math.cos(ph), s = Math.sin(ph);
      var r = this.re[idx], i = this.im[idx];
      this.re[idx] = r * c - i * s;
      this.im[idx] = r * s + i * c;
    }
  };

  Engine.prototype._fftAxis = function (axis, inverse) {
    var N = this.N, N2 = N * N, re = this.re, im = this.im;
    var lre = this.lre, lim = this.lim, fft = this.fft;
    var a, b, i, base, stride;
    for (a = 0; a < N; a++) for (b = 0; b < N; b++) {
      if (axis === 0) { base = a * N + b; stride = N2; }          // vary first index
      else if (axis === 1) { base = a * N2 + b; stride = N; }     // vary second
      else { base = a * N2 + b * N; stride = 1; }                  // vary third
      for (i = 0; i < N; i++) { lre[i] = re[base + i * stride]; lim[i] = im[base + i * stride]; }
      fft(lre, lim, inverse);
      for (i = 0; i < N; i++) { re[base + i * stride] = lre[i]; im[base + i * stride] = lim[i]; }
    }
  };

  Engine.prototype.step = function () {
    var half = CFG.dt / 2;
    this._applyV(half);
    this._fftAxis(0, false); this._fftAxis(1, false); this._fftAxis(2, false);
    var N3 = this.N * this.N * this.N;
    for (var idx = 0; idx < N3; idx++) {
      var c = this.kcos[idx], s = this.ksin[idx];
      var r = this.re[idx], i = this.im[idx];
      this.re[idx] = r * c - i * s;
      this.im[idx] = r * s + i * c;
    }
    this._fftAxis(0, true); this._fftAxis(1, true); this._fftAxis(2, true);
    this.t += CFG.dt;          // mid-point convention: second half-kick at t+dt
    this._applyV(half);
    this.steps++;
  };

  Engine.prototype.norm = function () {
    var n = 0, N3 = this.N * this.N * this.N;
    for (var idx = 0; idx < N3; idx++) n += this.re[idx] * this.re[idx] + this.im[idx] * this.im[idx];
    return n;
  };

  Engine.prototype.meanR = function () {
    var num = 0, den = 0, N = this.N, xs = this.xs, idx = 0;
    for (var a = 0; a < N; a++) for (var b = 0; b < N; b++) for (var c = 0; c < N; c++, idx++) {
      var p = this.re[idx] * this.re[idx] + this.im[idx] * this.im[idx];
      num += p * Math.sqrt(xs[a] * xs[a] + xs[b] * xs[b] + xs[c] * xs[c]);
      den += p;
    }
    return num / den;
  };

  // voxel cloud for volumetric rendering: [x,y,z,p/pmax,phase] above adaptive threshold
  Engine.prototype.voxels = function (maxCount) {
    var N = this.N, N2 = N*N, N3 = N*N*N, re = this.re, im = this.im, xs = this.xs, dx = this.cfg.dx;
    if (!this._pArr) this._pArr = new Float32Array(N3);
    var pA = this._pArr, pmax = 0, i;
    for (i = 0; i < N3; i++) { var q = re[i]*re[i] + im[i]*im[i]; pA[i] = q; if (q > pmax) pmax = q; }
    var thr = pmax * 0.0012, count, pass;
    for (pass = 0; pass < 6; pass++) {
      count = 0;
      for (i = 0; i < N3; i++) if (pA[i] > thr) count++;
      if (count <= maxCount) break;
      thr *= 1.8;
    }
    var X=new Float32Array(count), Y=new Float32Array(count), Z=new Float32Array(count);
    var P=new Float32Array(count), PH=new Float32Array(count);
    var JX=new Float32Array(count), JY=new Float32Array(count), JZ=new Float32Array(count);
    var n=0, jmax=0, JIT=0.45*dx;
    var RCLIP = 10.5;   // display clip (a.u.): exclude the rim where periodic images land
    for (var a=1; a<N-1; a++) for (var b=1; b<N-1; b++) for (var c=1; c<N-1; c++) {
      if (Math.abs(xs[a]) > RCLIP || Math.abs(xs[b]) > RCLIP || Math.abs(xs[c]) > RCLIP) continue;
      var idx = a*N2 + b*N + c;
      if (pA[idx] > thr && n < count) {
        X[n]=xs[a]+(Math.random()-0.5)*2*JIT;
        Y[n]=xs[b]+(Math.random()-0.5)*2*JIT;
        Z[n]=xs[c]+(Math.random()-0.5)*2*JIT;
        P[n]=pA[idx]/pmax; PH[n]=Math.atan2(im[idx],re[idx]);
        // j = Im(psi* grad psi), differenze centrali
        var jx = re[idx]*(im[idx+N2]-im[idx-N2]) - im[idx]*(re[idx+N2]-re[idx-N2]);
        var jy = re[idx]*(im[idx+N ]-im[idx-N ]) - im[idx]*(re[idx+N ]-re[idx-N ]);
        var jz = re[idx]*(im[idx+1 ]-im[idx-1 ]) - im[idx]*(re[idx+1 ]-re[idx-1 ]);
        JX[n]=jx; JY[n]=jy; JZ[n]=jz;
        var jm = Math.sqrt(jx*jx+jy*jy+jz*jz); if (jm>jmax) jmax=jm;
        n++;
      }
    }
    return { n:n, x:X, y:Y, z:Z, p:P, ph:PH, jx:JX, jy:JY, jz:JZ, jmax:jmax };
  };

  // central slices |Ψ|², scaled 0..255 (same convention as the old player)
  Engine.prototype.slices = function () {
    var N = this.N, N2 = N * N, h = N >> 1;
    function blank() { var m = []; for (var i = 0; i < N; i++) m.push(new Float64Array(N)); return m; }
    var xy = blank(), xz = blank(), yz = blank();
    var re = this.re, im = this.im;
    var max = 0, a, b, v;
    for (a = 0; a < N; a++) for (b = 0; b < N; b++) {
      v = sq(a * N2 + b * N + h); xy[a][b] = v; if (v > max) max = v;   // z = 0 plane
      v = sq(a * N2 + h * N + b); xz[a][b] = v; if (v > max) max = v;   // y = 0
      v = sq(h * N2 + a * N + b); yz[a][b] = v; if (v > max) max = v;   // x = 0
    }
    function sq(i) { return re[i] * re[i] + im[i] * im[i]; }
    // rescan because sq was hoisted after use in some engines — recompute safely
    max = 0;
    for (a = 0; a < N; a++) for (b = 0; b < N; b++) {
      xy[a][b] = re[(a * N2 + b * N + h)] ** 2 + im[(a * N2 + b * N + h)] ** 2;
      xz[a][b] = re[(a * N2 + h * N + b)] ** 2 + im[(a * N2 + h * N + b)] ** 2;
      yz[a][b] = re[(h * N2 + a * N + b)] ** 2 + im[(h * N2 + a * N + b)] ** 2;
      max = Math.max(max, xy[a][b], xz[a][b], yz[a][b]);
    }
    var s = max > 0 ? 255 / max : 0;
    for (a = 0; a < N; a++) for (b = 0; b < N; b++) {
      xy[a][b] = Math.round(xy[a][b] * s);
      xz[a][b] = Math.round(xz[a][b] * s);
      yz[a][b] = Math.round(yz[a][b] * s);
    }
    return { xy: xy, xz: xz, yz: yz };
  };

  global.QGTH3DEngine = Engine;
  if (typeof module !== 'undefined' && module.exports) module.exports = Engine;
})(typeof window !== 'undefined' ? window : globalThis);


// ── QGT Hydrogen 3D — LIVE controller (replaces the pre-computed player) ──
var H3D_TOT = 51;          // frames per loop: t = 0 .. 12.0 a.u. ≈ 290 as (dt = 0.24)
var h3d_eng = null, h3d_frames = [], h3d_fi = 0;
var h3d_playing = false, h3d_timer = null;

function h3dSnapshot() {
  var s = h3d_eng.slices();
  return { t: h3d_eng.t, norm: h3d_eng.norm(), xy: s.xy, xz: s.xz, yz: s.yz,
           vox: h3d_eng.voxels(20000) };
}

// ── volumetric renderer ──────────────────────────────────────
var h3d_rotY = 0.25, h3d_tilt = 1.15, h3d_sprites = null;
function h3dSprites() {
  if (h3d_sprites) return h3d_sprites;
  var mk = function (hue, white) {
    var c = document.createElement('canvas'); c.width = c.height = 32;
    var g = c.getContext('2d');
    var col = white ? 'rgba(255,255,255,' : 'hsla(' + hue + ',95%,62%,';
    var gr = g.createRadialGradient(16, 16, 0, 16, 16, 16);
    gr.addColorStop(0, col + '1)'); gr.addColorStop(0.35, col + '0.55)');
    gr.addColorStop(1, col + '0)');
    g.fillStyle = gr; g.fillRect(0, 0, 32, 32);
    return c;
  };
  var arr = [];
  for (var i = 0; i < 12; i++) arr.push(mk(200 + i * 30, false));
  var lines = [];
  for (var j = 0; j < 12; j++) lines.push('hsl(' + (200 + j * 30) + ',95%,68%)');
  h3d_sprites = { hue: arr, white: mk(0, true), line: lines };
  return h3d_sprites;
}
function h3dBand(p) {
  var w = 0, B = [[0.45,0.07,0.7],[0.13,0.03,1.0],[0.035,0.010,1.4],[0.008,0.0028,1.1]];
  for (var i = 0; i < 4; i++) { var d = (p - B[i][0]) / B[i][1]; w += B[i][2] * Math.exp(-d * d); }
  return w;
}
function h3dRender3D(fr) {
  var c = document.getElementById('h3d-3d');
  if (!c || !fr.vox) return;
  var ctx = c.getContext('2d');
  var W = c.width, H = c.height, cx = W / 2, cy = H / 2, scale = 21;
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = 'lighter';
  var sp = h3dSprites(), v = fr.vox;
  var cyr = Math.cos(h3d_rotY), syr = Math.sin(h3d_rotY);
  var ct = Math.cos(h3d_tilt), st = Math.sin(h3d_tilt);
  var jmax = v.jmax || 1e-30;
  for (var i = 0; i < v.n; i++) {
    var p = v.p[i];
    var X = v.x[i] * cyr + v.z[i] * syr;
    var Z = -v.x[i] * syr + v.z[i] * cyr;
    var Y = v.y[i] * ct - Z * st;
    var u = cx + scale * X, vv = cy - scale * Y;
    var a = Math.pow(Math.min(p, 0.30), 0.5);
    var bd = h3dBand(p);
    var phgate = 0.5 + 0.5 * Math.cos(8 * v.ph[i]);          // iso-fase: maglia
    var w = 0.06 * Math.pow(p, 0.30) + 0.40 * bd * (0.35 + 0.65 * phgate);
    var size = (0.5 + 1.0 * a + 0.8 * bd) * 16;
    if (u + size < 0 || vv + size < 0 || u - size > W || vv - size > H) continue;
    var b = ((v.ph[i] + Math.PI) / (2 * Math.PI) * 12) | 0; if (b > 11) b = 11;
    ctx.globalAlpha = Math.min(0.85, w * 0.45);
    ctx.drawImage(sp.hue[b], u - size / 2, vv - size / 2, size, size);
    if (p > 0.5) {
      ctx.globalAlpha = Math.min(0.7, (p - 0.5) * 1.1);
      ctx.drawImage(sp.white, u - size * 0.35, vv - size * 0.35, size * 0.7, size * 0.7);
    }
    // filamento lungo la corrente di probabilità (trama)
    var jx = v.jx[i], jy = v.jy[i], jz = v.jz[i];
    var jm = Math.sqrt(jx*jx + jy*jy + jz*jz);
    if (jm > 0.05 * jmax && p > 0.003 && p < 0.7) {
      var jXp = jx * cyr + jz * syr;
      var jZp = -jx * syr + jz * cyr;
      var jYp = jy * ct - jZp * st;
      var nrm = Math.sqrt(jXp*jXp + jYp*jYp) + 1e-12;
      var L = 4 + 13 * (jm / jmax);
      var du = jXp / nrm * L, dv = -jYp / nrm * L;
      ctx.globalAlpha = Math.min(0.55, 0.12 + 0.7 * (jm / jmax));
      ctx.strokeStyle = sp.line[b];
      ctx.beginPath();
      ctx.moveTo(u - du, vv - dv);
      ctx.lineTo(u + du, vv + dv);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
  // proton: fixed at the origin as the centre of the Coulomb potential (infinite-mass approx.)
  ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 2*Math.PI);
  ctx.fillStyle = '#fff'; ctx.fill();
  ctx.beginPath(); ctx.arc(cx, cy, 5.5, 0, 2*Math.PI);
  ctx.strokeStyle = 'rgba(255,80,80,0.9)'; ctx.lineWidth = 1.4; ctx.stroke();
  ctx.font = '11px monospace'; ctx.fillStyle = 'rgba(255,120,120,0.95)';
  ctx.fillText('p\u207A', cx + 9, cy - 7);
}
(function h3dDragInit() {
  document.addEventListener('DOMContentLoaded', function () {
    var c = document.getElementById('h3d-3d');
    if (!c) return;
    var drag = false, lx = 0, ly = 0;
    c.addEventListener('pointerdown', function (e) { drag = true; lx = e.clientX; ly = e.clientY; c.setPointerCapture(e.pointerId); c.style.cursor='grabbing'; });
    c.addEventListener('pointermove', function (e) {
      if (!drag) return;
      h3d_rotY += (e.clientX - lx) * 0.012;
      h3d_tilt = Math.max(0.1, Math.min(2.6, h3d_tilt + (e.clientY - ly) * 0.008));
      lx = e.clientX; ly = e.clientY;
      if (h3d_frames[h3d_fi]) h3dRender3D(h3d_frames[h3d_fi]);
    });
    var up = function () { drag = false; c.style.cursor='grab'; };
    c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
  });
})();
function h3dInit() {
  h3d_eng = new QGTH3DEngine();
  h3d_frames = [h3dSnapshot()];
  h3d_fi = 0;
}
function h3dDraw(fi) {
  if (!h3d_eng) h3dInit();
  fi = Math.max(0, Math.min(fi, H3D_TOT - 1));
  while (h3d_frames.length <= fi) { h3d_eng.step(); h3d_frames.push(h3dSnapshot()); }
  h3d_fi = fi;
  var fr = h3d_frames[fi], N = h3d_eng.N;

  var sl = document.getElementById('h3d-slider');
  sl.max = H3D_TOT - 1; sl.value = fi;
  document.getElementById('h3d-tval').textContent = 't = ' + fr.t.toFixed(2) + ' a.u. = ' + (fr.t*24.19).toFixed(0) + ' as';

  h3d_rotY += 0.045;            // rotazione lenta legata all'avanzamento
  h3dRender3D(fr);
  h3dPaint('h3d-xy', fr.xy, N);
  h3dPaint('h3d-xz', fr.xz, N);
  h3dPaint('h3d-yz', fr.yz, N);

  var norm0 = h3d_frames[0].norm;
  var drift = Math.abs(fr.norm - norm0) / norm0;
  document.getElementById('h3d-info').innerHTML =
    '<b>LIVE</b> split-operator &nbsp;|&nbsp; ' +
    '<b>t</b> = ' + fr.t.toFixed(2) + ' a.u. (' + (fr.t*24.19).toFixed(0) + ' as) &nbsp;|&nbsp; ' +
    '<b>norm</b> = ' + fr.norm.toFixed(10) + ' &nbsp;|&nbsp; ' +
    '<b>&Delta;norm</b> = ' + drift.toExponential(1) + ' &nbsp;|&nbsp; ' +
    '<b>a&#8320;_QGT</b> = ' + h3d_eng.cfg.a0.toFixed(6) + ' &nbsp;|&nbsp; ' +
    '<b>f_t</b> = ' + h3d_eng.cfg.f_t.toFixed(6) + ' [P] &nbsp;|&nbsp; ' +
    'frame ' + fi + '/' + (H3D_TOT - 1) + ' (loop) &nbsp;|&nbsp; ' + (fr.vox?fr.vox.n:0) + ' splats';
}
var _h3dOff = null;
function h3dPaint(id, mat, N) {
  var c = document.getElementById(id);
  if (!c) return;
  if (!_h3dOff) { _h3dOff = document.createElement('canvas'); }
  _h3dOff.width = N; _h3dOff.height = N;
  var octx = _h3dOff.getContext('2d');
  var img = octx.createImageData(N, N);
  for (var i = 0; i < N; i++) for (var j = 0; j < N; j++) {
    var v = mat[i][N - 1 - j];
    var r = Math.min(255, v * 3 | 0);
    var g = Math.min(255, v * 1.5 | 0);
    var b = Math.min(255, v * 0.5 | 0);
    var p = (j * N + i) * 4;
    img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  var R = 256;                       // display resolution, bilinear upscale
  c.width = R; c.height = R;
  var ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(_h3dOff, 0, 0, R, R);
}
function h3dGoto(fi) { h3dDraw(+fi); }
function h3dTogglePlay() {
  if (h3d_playing) {
    clearInterval(h3d_timer);
    h3d_playing = false;
    document.getElementById('h3d-playbtn').textContent = '\u25B6 Play';
  } else {
    h3d_playing = true;
    document.getElementById('h3d-playbtn').textContent = '\u23F8 Pause';
    h3d_timer = setInterval(function () {
      var next = h3d_fi + 1;
      if (next >= H3D_TOT) { h3dInit(); h3dDraw(0); }   // seamless loop, recomputed live
      else h3dDraw(next);
    }, 160);
  }
}
document.addEventListener('DOMContentLoaded', function () {
  var sec = document.getElementById('hydrogen3d');
  if (!sec) return;
  var obs = new IntersectionObserver(function (entries) {
    if (entries[0].isIntersecting) { h3dDraw(0); obs.disconnect(); }
  }, { threshold: 0.1 });
  obs.observe(sec);
});