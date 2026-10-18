/* OilStudio live preview (WebGL 2) ---------------------------------------
   Makes every slider update the After image while it moves.

   The After image is drawn here, on the GPU, in three possible modes:

     A  the painting from the last Run (its bare canvas and paint height,
        sent by the server).  The settings applied on top of it - Relief,
        Lightening, Blend with Original, Light, Canvas, Frame, Text - are
        GPU passes, so they follow the slider exactly.
     B  a stroke settings slider moved after the Run.  The strokes come from
        the planner (oilpaint/planner.py) and are painted by the GPU: the
        strokes of the Run itself when only their look changed (Intensity,
        Hair Thickness, Brush Density, Abstract colours...), or a new plan
        for the new settings - a small, quick one while the slider is being
        dragged, the full one when it is let go.
     C  Canvas Oil drafts: that style has too many CPU stages for the GPU,
        so the server sends quick drafts and the GPU finishes them.

   B and C are previews: Run paints the final picture.  The maths of every
   pass follows oilpaint/engine.py, planner.py and decoration.py - keep them
   in step.                                                               */
'use strict';

(function () {
  const FINISH_KEYS = ['relief', 'lightening', 'blend_original', 'blend_mode'];
  const DECO_PREFIX = ['light_', 'canvas_', 'frame_', 'text_'];
  /* stroke settings that only change how the planned strokes look */
  const LOOK_KEYS = ['intensity', 'hair_thickness', 'brush_density',
    'abstract_enabled', 'ab_color', 'ab_saturation', 'ab_contrast',
    'cor_hue', 'cor_saturation', 'cor_brightness',
    'edge_smoothness', 'edge_hair_thickness', 'edge_relief'];
  const BLEND_MODES = ['Normal', 'Multiply', 'Screen', 'Overlay', 'Soft Light',
    'Hard Light', 'Darken', 'Lighten', 'Linear Burn', 'Linear Dodge', 'Color Burn',
    'Color Dodge', 'Difference', 'Hue', 'Color', 'Luminosity'];
  const NCOL = 20;
  const C = { B: 0, JB: 3, T: 6, TX: 7, OP: 8, HA: 9, KIND: 10, FLAGS: 11, R0: 12, SEED: 18 };
  const KIND_LAST = 1, KIND_FILL = 2, KIND_EDGE = 3;
  const DATA_W = 2048;               /* stroke data texture width (4 texels a stroke) */

  const isPost = k => FINISH_KEYS.includes(k) || DECO_PREFIX.some(p => k.startsWith(p));
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  /* ======================================================== settings keys */
  function strokeKeys(p) {
    return Object.keys(p).filter(k => !isPost(k) && k !== 'guides').sort();
  }
  /* the guiding lines as params.clean() keeps them */
  function guidesKey(g) {
    const lines = [];
    for (const line of (g || []).slice(0, 400)) {
      const pts = (line || []).slice(0, 600).filter(pt => pt && pt[0] >= 0 && pt[0] <= 1 && pt[1] >= 0 && pt[1] <= 1)
        .map(([x, y]) => [Math.round(x * 1e4) / 1e4, Math.round(y * 1e4) / 1e4]);
      if (pts.length >= 2) lines.push(pts);
    }
    return JSON.stringify(lines);
  }
  /* everything that changes where the strokes go */
  function layoutKey(p) {
    const keys = strokeKeys(p).filter(k => !LOOK_KEYS.includes(k) && !k.startsWith('dist_'));
    const dist = p.abstract_enabled && Number(p.dist_strength) > 0
      ? [p.dist_strength, p.dist_curvature, p.dist_seed] : null;
    return JSON.stringify(keys.map(k => [k, p[k]])) + '|' + JSON.stringify(dist) + '|' + guidesKey(p.guides);
  }
  /* everything that changes the painting itself (not what goes on top) */
  function strokeKey(p) {
    return JSON.stringify(strokeKeys(p).map(k => [k, p[k]])) + '|' + guidesKey(p.guides);
  }

  /* ======================================================== colour maths */
  /* OpenCV 8-bit BGR <-> HSV (H 0..180), as used by abstract.py */
  function bgr2hsv8(b, g, r) {
    b = Math.floor(clamp(b, 0, 255)); g = Math.floor(clamp(g, 0, 255)); r = Math.floor(clamp(r, 0, 255));
    const v = Math.max(r, g, b), mn = Math.min(r, g, b), d = v - mn;
    const s = v ? Math.round(255 * d / v) : 0;
    let h = 0;
    if (d) {
      if (v === r) h = 60 * (g - b) / d;
      else if (v === g) h = 120 + 60 * (b - r) / d;
      else h = 240 + 60 * (r - g) / d;
      if (h < 0) h += 360;
    }
    return [Math.round(h / 2) % 180, s, v];
  }
  function hsv2bgr8(h, s, v) {
    h = Math.floor(clamp(h, 0, 255)); s = Math.floor(clamp(s, 0, 255)) / 255; v = Math.floor(clamp(v, 0, 255));
    const hh = (h * 2) % 360 / 60, i = Math.floor(hh), f = hh - i;
    const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    let r, g, b;
    switch (i) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      default: r = v; g = p; b = q;
    }
    return [Math.round(b), Math.round(g), Math.round(r)];
  }
  function correctionOn(p) {
    return !!Number(p.abstract_enabled) &&
      (Number(p.cor_hue) || Number(p.cor_saturation) || Number(p.cor_brightness));
  }
  /* abstract.correct() for one colour */
  function correct(b, g, r, p) {
    const hue = Number(p.cor_hue), sat = Number(p.cor_saturation), bri = Number(p.cor_brightness);
    let [h, s, v] = bgr2hsv8(b, g, r);
    if (hue) h = ((h + hue / 2) % 180 + 180) % 180;
    if (sat) s = clamp(s * (1 + sat / 100), 0, 255);
    let [ob, og, or] = hsv2bgr8(h, s, v);
    if (bri) {
      const f = bri / 100 * 0.85;
      if (bri > 0) { ob += (255 - ob) * f; og += (255 - og) * f; or += (255 - or) * f; }
      else { ob *= 1 + f; og *= 1 + f; or *= 1 + f; }
    }
    return [clamp(ob, 0, 255), clamp(og, 0, 255), clamp(or, 0, 255)];
  }

  /* planner.stroke_style() + stroke_colours(), packed for the stroke shader.
     4 RGBA32F texels a stroke:
       0  r g b (0..255)   opacity
       1  half-width       hair      bristles   softness
       2  paint height     shade var flat       seed
       3  points           damp      -          -                        */
  function styleStrokes(plan, p) {
    const n = plan.n, st = plan.strokes, offs = plan.offs;
    const rows = Math.max(1, Math.ceil(n * 4 / DATA_W));
    const out = new Float32Array(rows * DATA_W * 4);
    const inten = Number(p.intensity) / 100;
    const softK = lerp(0.85, 0, inten), opac = lerp(0.62, 1, inten), shadeV = lerp(0.04, 0.20, inten);
    const hairFrac = Number(p.hair_thickness) / 100, density = Number(p.brush_density) / 100;
    const flatAll = hairFrac > 0.72;
    const eb = plan.edgeBase;
    const ehair = clamp(eb * Number(p.edge_hair_thickness) / 100, 1, Math.max(1, eb));
    const enb = Math.floor(clamp(eb / Math.max(ehair, 0.9) * 1.6, 1, 14));
    const esoft = lerp(0.2, 4.5, Number(p.edge_smoothness) / 100);
    const eheight = 0.6 + 2.2 * Number(p.edge_relief) / 100;
    const corr = correctionOn(p);
    const abOn = !!Number(p.abstract_enabled);
    const cval = Number(p.ab_color), sval = Number(p.ab_saturation), kval = Number(p.ab_contrast);

    for (let i = 0; i < n; i++) {
      const o = i * NCOL, kind = st[o + C.KIND], t = st[o + C.T], tx = st[o + C.TX];
      const flags = st[o + C.FLAGS] | 0;
      let hair = clamp(t * hairFrac * 0.9, 1.1, Math.max(1.1, t));
      let nbf = clamp(1 + density * (t / hair) * 1.6, 1, 14);
      let soft = softK * Math.max(0.25, hair * 0.30);
      let op = Math.min(1, opac * (kind === KIND_LAST ? 1 : 0.94) * (0.74 + 0.26 * tx) * st[o + C.OP]);
      let sv = shadeV * (0.35 + 0.65 * tx), height = st[o + C.HA], flat = flatAll;
      if (kind === KIND_LAST) soft *= 0.55;
      let nb = Math.floor(nbf);
      if (kind === KIND_FILL) { nb = Math.floor(clamp(nbf, 1, 10)); op = opac; sv = shadeV; flat = false; }
      else if (kind === KIND_EDGE) {
        hair = ehair; nb = enb; soft = esoft; op = st[o + C.OP]; height = eheight; sv = 0.18; flat = false;
      }

      /* colour: correction, palette jitter, Abstract recolouring */
      let b = st[o], g = st[o + 1], r = st[o + 2];
      if ((flags & 1) && corr) [b, g, r] = correct(b, g, r, p);
      b = clamp(b + st[o + C.JB], 0, 255); g = clamp(g + st[o + C.JB + 1], 0, 255); r = clamp(r + st[o + C.JB + 2], 0, 255);
      if (abOn && (flags & 2)) {
        const R = o + C.R0;
        if (cval > 0 && st[R] < cval / 90 * 0.85) {
          let [h, s, v] = bgr2hsv8(b, g, r);
          h = (((h + (st[R + 1] * 2 - 1) * (cval / 90) * 90) % 180) + 180) % 180;
          [b, g, r] = hsv2bgr8(h, s, v);
        }
        if (sval > 0 && st[R + 2] < 0.6) {
          let [h, s, v] = bgr2hsv8(b, g, r);
          s = clamp(s * (1 + st[R + 3] * sval / 40 * 1.6), 0, 255);
          [b, g, r] = hsv2bgr8(h, s, v);
        }
        if (kval > 0 && st[R + 4] < 0.5) {
          const f = 1 + (st[R + 5] * 2 - 1) * (kval / 100) * 0.75;
          b = 128 + (b - 128) * f; g = 128 + (g - 128) * f; r = 128 + (r - 128) * f;
        }
        b = clamp(b, 0, 255); g = clamp(g, 0, 255); r = clamp(r, 0, 255);
      }
      const lum = (b + g + r) / 3 / 255;
      const damp = clamp(1.15 - 0.70 * Math.pow(lum, 1.6), 0.35, 1.15);

      const q = i * 16;
      out[q] = r; out[q + 1] = g; out[q + 2] = b; out[q + 3] = op;
      out[q + 4] = Math.max(0.5, t * 0.5); out[q + 5] = hair; out[q + 6] = nb; out[q + 7] = soft;
      out[q + 8] = height; out[q + 9] = sv; out[q + 10] = flat ? 1 : 0; out[q + 11] = st[o + C.SEED];
      out[q + 12] = offs[i + 1] - offs[i]; out[q + 13] = damp;
    }
    return { data: out, rows };
  }

  /* decode a PNG for a texture without premultiplying its alpha; older
     WebKit does not take createImageBitmap options, so fall back to <img> */
  async function decodeImage(blob) {
    try {
      return await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    } catch (err) {
      const url = URL.createObjectURL(blob);
      try {
        const img = new Image();
        img.src = url;
        await img.decode();
        return img;
      } finally { URL.revokeObjectURL(url); }
    }
  }

  const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
  function toHalf(v) {
    _f32[0] = v; const x = _u32[0];
    const sign = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
    if (e <= 0) return sign;
    if (e >= 31) return sign | 0x7bff;
    return sign | (e << 10) | ((x & 0x7fffff) >>> 13);
  }
  const nextFrame = () => new Promise(r => requestAnimationFrame(() => r()));

  /* ======================================================== plan parsing */
  function parsePlan(buf) {
    const dv = new DataView(buf);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== 'OSP1') throw new Error('Bad plan data');
    const hl = dv.getUint32(4, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
    let at = 8 + hl;
    const strokes = new Float32Array(buf, at, head.n * head.ncol); at += head.n * head.ncol * 4;
    const pts = new Float32Array(buf, at, head.m * 2); at += head.m * 8;
    const offs = new Uint32Array(buf, at, head.n + 1); at += (head.n + 1) * 4;
    const ground = new Uint8Array(buf, at, head.gw * head.gh * 3);
    return { w: head.w, h: head.h, n: head.n, m: head.m, strokes, pts, offs, ground,
      gw: head.gw, gh: head.gh, edgeBase: head.edge_base };
  }

  /* triangle ribbons along every stroke, with round end caps */
  function buildGeometry(plan) {
    const { n, pts, offs } = plan;
    let rows = 0, segs = 0;
    for (let i = 0; i < n; i++) { const np = offs[i + 1] - offs[i]; rows += np + 2; segs += np + 1; }
    const V = new Float32Array(rows * 2 * 8);
    const I = new Uint32Array(segs * 6);
    const ends = new Uint32Array(n);
    let v = 0, ix = 0, row = 0;
    const put = (x, y, nx, ny, along, cap, id) => {
      for (let s = -1; s <= 1; s += 2) {
        V[v++] = x; V[v++] = y; V[v++] = nx; V[v++] = ny;
        V[v++] = along; V[v++] = s; V[v++] = cap; V[v++] = id;
      }
    };
    for (let i = 0; i < n; i++) {
      const o = offs[i], np = offs[i + 1] - o;
      const first = row;
      for (let j = -1; j <= np; j++) {
        const k = clamp(j, 0, np - 1);
        let dx, dy;
        if (np < 3) { dx = pts[(o + np - 1) * 2] - pts[o * 2]; dy = pts[(o + np - 1) * 2 + 1] - pts[o * 2 + 1]; }
        else if (k === 0) { dx = pts[(o + 1) * 2] - pts[o * 2]; dy = pts[(o + 1) * 2 + 1] - pts[o * 2 + 1]; }
        else if (k === np - 1) { dx = pts[(o + k) * 2] - pts[(o + k - 1) * 2]; dy = pts[(o + k) * 2 + 1] - pts[(o + k - 1) * 2 + 1]; }
        else { dx = (pts[(o + k + 1) * 2] - pts[(o + k - 1) * 2]) * 0.5; dy = (pts[(o + k + 1) * 2 + 1] - pts[(o + k - 1) * 2 + 1]) * 0.5; }
        let len = Math.hypot(dx, dy);
        if (len < 1e-6) { dx = 1; dy = 0; len = 1; }
        dx /= len; dy /= len;
        const along = np > 1 ? k / (np - 1) : 0;
        put(pts[(o + k) * 2], pts[(o + k) * 2 + 1], -dy, dx, along, j < 0 ? -1 : (j >= np ? 1 : 0), i);
        row++;
      }
      for (let r = first; r < row - 1; r++) {
        const a = r * 2;
        I[ix++] = a; I[ix++] = a + 1; I[ix++] = a + 2;
        I[ix++] = a + 1; I[ix++] = a + 3; I[ix++] = a + 2;
      }
      ends[i] = ix;
    }
    return { V, I, count: ix, ends };
  }

  /* ============================================================ shaders */
  const VS_FULL = `#version 300 es
  void main(){ vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

  const GLSL_COMMON = `
  precision highp float; precision highp int; precision highp sampler2D;
  vec3 rgb2hsv(vec3 c){ float v=max(c.r,max(c.g,c.b)), mn=min(c.r,min(c.g,c.b)), d=v-mn; float h=0.;
    if(d>1e-6){ if(v==c.r) h=(c.g-c.b)/d; else if(v==c.g) h=2.+(c.b-c.r)/d; else h=4.+(c.r-c.g)/d; h=fract(h/6.+1.); }
    return vec3(h, v>0.?d/v:0., v); }
  vec3 hsv2rgb(vec3 c){ vec3 p=abs(fract(c.xxx+vec3(1.,2./3.,1./3.))*6.-3.);
    return c.z*mix(vec3(1.),clamp(p-1.,0.,1.),c.y); }
  vec3 rgb2hls(vec3 c){ float mx=max(c.r,max(c.g,c.b)), mn=min(c.r,min(c.g,c.b)), l=(mx+mn)*.5, d=mx-mn, h=0., s=0.;
    if(d>1e-6){ s = l<.5 ? d/(mx+mn) : d/(2.-mx-mn);
      if(mx==c.r) h=(c.g-c.b)/d+(c.g<c.b?6.:0.); else if(mx==c.g) h=(c.b-c.r)/d+2.; else h=(c.r-c.g)/d+4.; h/=6.; }
    return vec3(h,l,s); }
  float h2r(float p,float q,float t){ t=fract(t); if(t<1./6.) return p+(q-p)*6.*t; if(t<.5) return q;
    if(t<2./3.) return p+(q-p)*(2./3.-t)*6.; return p; }
  vec3 hls2rgb(vec3 x){ if(x.z<=0.) return vec3(x.y); float q = x.y<.5 ? x.y*(1.+x.z) : x.y+x.z-x.y*x.z, p=2.*x.y-q;
    return vec3(h2r(p,q,x.x+1./3.), h2r(p,q,x.x), h2r(p,q,x.x-1./3.)); }
  /* abstract.correct(): x hue (deg), y saturation, z brightness */
  vec3 correctC(vec3 c255, vec3 k){
    vec3 h = rgb2hsv(floor(clamp(c255,0.,255.))/255.);
    if(k.x!=0.) h.x = fract(h.x + k.x/360. + 1.);
    if(k.y!=0.) h.y = clamp(h.y*(1.+k.y/100.),0.,1.);
    vec3 o = floor(hsv2rgb(h)*255.+.5);
    if(k.z>0.) o += (255.-o)*(k.z/100.)*.85; else if(k.z<0.) o *= 1.+k.z/100.*.85;
    return clamp(o,0.,255.); }
  `;

  /* -- the stroke painter: a ribbon per stroke, bristles in the shader -- */
  const VS_STROKE = `#version 300 es
  ${GLSL_COMMON}
  layout(location=0) in vec2 aPos; layout(location=1) in vec2 aNrm;
  layout(location=2) in float aAlong; layout(location=3) in float aSide;
  layout(location=4) in float aCap; layout(location=5) in float aId;
  uniform sampler2D uData; uniform vec2 uSize;
  out vec2 vLocal; out float vAlong;
  flat out vec4 vA; flat out vec4 vB; flat out vec4 vC; flat out vec4 vD;
  vec4 fetchD(int id,int k){ int i=id*4+k; return texelFetch(uData, ivec2(i & ${DATA_W - 1}, i >> ${Math.log2(DATA_W)}), 0); }
  void main(){
    int id = int(aId + .5);
    vA=fetchD(id,0); vB=fetchD(id,1); vC=fetchD(id,2); vD=fetchD(id,3);
    float ext = vB.x + vB.y*.5 + 2.5*vB.w + 1.5;
    vec2 tng = vec2(aNrm.y, -aNrm.x);
    vec2 p = aPos + aNrm*aSide*ext + tng*aCap*ext;
    vLocal = vec2(aSide*ext, aCap*ext); vAlong = aAlong;
    gl_Position = vec4(p/uSize*2.-1., 0., 1.);
  }`;
  const FS_STROKE = `#version 300 es
  ${GLSL_COMMON}
  in vec2 vLocal; in float vAlong;
  flat in vec4 vA; flat in vec4 vB; flat in vec4 vC; flat in vec4 vD;
  uniform int uPass; out vec4 o;
  float hash(float a, float b){ return fract(sin(a*91.3458 + b*47.9898)*43758.5453); }
  void main(){
    float hlf=vB.x, hair=vB.y, nb=vB.z, soft=vB.w, flat_=vC.z, seed=vC.w, npts=vD.x;
    float t = clamp(vAlong, 0., 1.);
    bool taper = flat_ < .5 && npts >= 6.;
    float prof = 1.;
    if(taper){ float sg = min(floor(t*7.), 6.); prof = .42 + .58*pow(sin(3.14159265*(sg+.5)/7.), .45); }
    float hw = max(.5, floor(2.*hlf*prof + .5)*.5);
    float x = hw - length(vLocal);
    /* OpenCV's anti-aliased edge (about a 1 px ramp) blurred by the
       stroke's softness, as a smooth step of the same width */
    float s = sqrt(soft*soft + .09);
    float a = 1./(1.+exp(-1.7*x/s));
    /* bristles: streaks of lighter and darker paint along the stroke */
    float sh = 1.;
    float hp = max(1., floor(hair + .5));
    float stepk = nb > 1.5 ? 2.*hlf/(nb-1.) : 0.;
    float kc = nb > 1.5 ? (vLocal.x + hlf)/stepk : 0.;
    int k0 = int(floor(kc)) - 1;
    float tr = min(.28, max(1., hlf)/max(npts-1., 1.));
    for(int j=0;j<4;j++){
      int k = k0 + j;
      if(k < 0 || float(k) > nb - 1.) continue;
      float off = (nb > 1.5 ? -hlf + float(k)*stepk : 0.) + (hash(seed, float(k))*2.-1.)*.3*hair;
      float trim = flat_ > .5 ? 0. : pow(abs(off)/max(hlf,1e-6), 2.)*tr;
      if(t < trim || t > 1.-trim) continue;
      if(abs(vLocal.x - off) <= hp*.5)
        sh = clamp(1. + (hash(seed + .37, float(k))*2.-1.)*vC.y*vD.y, 10./128., 255./128.);
    }
    a *= vA.a;
    if(uPass == 0) o = vec4(vA.rgb/255.*sh, a);
    else o = vec4(a*vC.x*(.55 + .75*sh), 0., 0., 1.);
  }`;

  const FS_GROUND = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uTex; uniform vec2 uSize; uniform vec3 uCorr; uniform int uCorrOn; out vec4 o;
  void main(){ vec3 c = texture(uTex, gl_FragCoord.xy/uSize).rgb*255.;
    if(uCorrOn==1) c = correctC(c, uCorr);
    o = vec4(c/255., 1.); }`;

  /* GaussianBlur(height, sigma 0.8) - OpenCV's 7-tap kernel */
  const FS_HBLUR = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uH; uniform ivec2 uDim; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); float w[4] = float[4](1., .45783336, .043936934, .00088383);
    float acc=0., ws=0.;
    for(int y=-3;y<=3;y++) for(int x=-3;x<=3;x++){ float k=w[abs(x)]*w[abs(y)];
      acc += k*texelFetch(uH, clamp(p+ivec2(x,y), ivec2(0), uDim-1), 0).r; ws += k; }
    o = vec4(acc/ws, 0., 0., 1.); }`;

  /* |relief| samples, read back to find the percentile apply_relief uses */
  const FS_EMB = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uH; uniform ivec2 uDim; uniform vec2 uStride; out vec4 o;
  float h(ivec2 p){ return texelFetch(uH, clamp(p, ivec2(0), uDim-1), 0).r; }
  void main(){ ivec2 p = ivec2((floor(gl_FragCoord.xy)+.5)*uStride);
    float gx = h(p+ivec2(1,-1))+2.*h(p+ivec2(1,0))+h(p+ivec2(1,1)) - h(p+ivec2(-1,-1))-2.*h(p+ivec2(-1,0))-h(p+ivec2(-1,1));
    float gy = h(p+ivec2(-1,1))+2.*h(p+ivec2(0,1))+h(p+ivec2(1,1)) - h(p+ivec2(-1,-1))-2.*h(p+ivec2(0,-1))-h(p+ivec2(1,-1));
    o = vec4(abs((gx+gy)*-.70710678), 0., 0., 1.); }`;

  /* engine.finish(): relief, lightening, blend with the original, brush edits */
  const FS_FINISH = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uCol; uniform sampler2D uHB; uniform sampler2D uOrig; uniform sampler2D uEdits;
  uniform ivec2 uDim; uniform float uRelief; uniform float uScale; uniform float uLight;
  uniform float uBlend; uniform int uMode; uniform int uSteps; uniform int uEditsOn;
  out vec4 o;
  float h(ivec2 p){ return texelFetch(uHB, clamp(p, ivec2(0), uDim-1), 0).r; }
  vec3 blendMode(vec3 a, vec3 b, int m){
    if(m==0) return b; if(m==1) return a*b; if(m==2) return 1.-(1.-a)*(1.-b);
    if(m==3) return mix(2.*a*b, 1.-2.*(1.-a)*(1.-b), vec3(greaterThan(a, vec3(.5))));
    if(m==4) return mix(a-(1.-2.*b)*a*(1.-a), a+(2.*b-1.)*(sqrt(clamp(a,0.,1.))-a), vec3(greaterThan(b, vec3(.5))));
    if(m==5) return mix(2.*a*b, 1.-2.*(1.-a)*(1.-b), vec3(greaterThan(b, vec3(.5))));
    if(m==6) return min(a,b); if(m==7) return max(a,b);
    if(m==8) return clamp(a+b-1.,0.,1.); if(m==9) return clamp(a+b,0.,1.);
    if(m==10) return 1.-clamp((1.-a)/max(b,vec3(1e-4)),0.,1.);
    if(m==11) return clamp(a/max(1.-b,vec3(1e-4)),0.,1.);
    if(m==12) return abs(a-b);
    vec3 ah = rgb2hls(a), bh = rgb2hls(b);
    if(m==13) ah.x = bh.x; else if(m==14){ ah.x = bh.x; ah.z = bh.z; } else ah.y = bh.y;
    return hls2rgb(ah); }
  void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 c = texelFetch(uCol, p, 0).rgb*255.;
    if((uSteps & 1) != 0 && uRelief > .005){
      float gx = h(p+ivec2(1,-1))+2.*h(p+ivec2(1,0))+h(p+ivec2(1,1)) - h(p+ivec2(-1,-1))-2.*h(p+ivec2(-1,0))-h(p+ivec2(-1,1));
      float gy = h(p+ivec2(-1,1))+2.*h(p+ivec2(0,1))+h(p+ivec2(1,1)) - h(p+ivec2(-1,-1))-2.*h(p+ivec2(0,-1))-h(p+ivec2(1,-1));
      float emb = clamp((gx+gy)*-.70710678/uScale, -1.6, 1.6);
      c *= 1. + emb*(.42*uRelief);
      c += pow(max(emb,0.),3.)*(.03*uRelief)*255.;
    }
    if((uSteps & 2) != 0 && uLight > .005){
      float lum = dot(floor(clamp(c,0.,255.)), vec3(.299,.587,.114))/255.;
      float m = pow(clamp((lum-.42)/.58, 0., 1.), 1.4);
      c += (255.-c)*(m*uLight*.85);
    }
    if((uSteps & 4) != 0 && uBlend > .002){
      vec3 a = clamp(c,0.,255.)/255., b = texelFetch(uOrig, p, 0).rgb;
      c = clamp(a + (blendMode(a,b,uMode)-a)*uBlend, 0., 1.)*255.;
    }
    if(uEditsOn == 1){ vec4 e = texelFetch(uEdits, p, 0); c = mix(c, e.rgb*255., e.a); }
    o = vec4(clamp(c,0.,255.)/255., 1.);
  }`;

  /* Light: vignette from the key light */
  const FS_LIGHT1 = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform vec2 uDim; uniform vec2 uLightPos; uniform float uSpread; uniform float uVig; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 c = texelFetch(uIn, p, 0).rgb*255.;
    float d = length(vec2(p) - uLightPos)/(length(uDim)*uSpread); float key = exp(-d*d*.5);
    if(uVig > .001) c *= 1. - uVig*.92*(1.-key);
    o = vec4(clamp(c,0.,255.)/255., 1.); }`;
  /* Light: the hot paint that blooms, at half size */
  const FS_HOT = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform ivec2 uDim; out vec4 o;
  void main(){ ivec2 q = ivec2(gl_FragCoord.xy)*2; vec3 acc = vec3(0.);
    for(int y=0;y<2;y++) for(int x=0;x<2;x++){
      vec3 c = texelFetch(uIn, min(q+ivec2(x,y), uDim-1), 0).rgb*255.;
      float lum = dot(floor(c), vec3(.299,.587,.114))/255.;
      acc += c*pow(clamp((lum-.55)/.45,0.,1.),1.5); }
    o = vec4(acc*.25/255., 1.); }`;
  const FS_GAUSS = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform ivec2 uDim; uniform ivec2 uDir; uniform float uSigma; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); int r = int(ceil(uSigma*3.)); vec3 acc=vec3(0.); float ws=0.;
    for(int i=-48;i<=48;i++){ if(i<-r||i>r) continue; float k=exp(-float(i*i)/(2.*uSigma*uSigma));
      acc += k*texelFetch(uIn, clamp(p+uDir*i, ivec2(0), uDim-1), 0).rgb; ws += k; }
    o = vec4(acc/ws, 1.); }`;
  /* Light: bloom, then warm light / cool shadow */
  const FS_LIGHT2 = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform sampler2D uBloom; uniform vec2 uDim; uniform vec2 uLightPos; uniform float uSpread;
  uniform float uGlow; uniform float uWarm; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 c = texelFetch(uIn, p, 0).rgb*255.;
    if(uGlow > .001){ vec3 b = texture(uBloom, (vec2(p)+.5)/uDim).rgb*255.; c = 255. - (255.-c)*(255.-b*uGlow*.85)/255.; }
    if(abs(uWarm) > .001){
      float d = length(vec2(p) - uLightPos)/(length(uDim)*uSpread); float key = exp(-d*d*.5);
      vec3 tint = vec3(.22, .05, -.16)*uWarm;
      c *= 1. + tint*key; c *= 1. - tint*.45*(1.-key); }
    o = vec4(clamp(c,0.,255.)/255., 1.); }`;
  /* Canvas: the surface's relief and hollows */
  const FS_CANVAS = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform sampler2D uSurf; uniform ivec2 uDim; uniform ivec2 uSurfDim;
  uniform float uAngle; uniform float uSc; uniform float uRelief; uniform float uStrength; out vec4 o;
  float h(ivec2 p){ return texelFetch(uSurf, clamp(p, ivec2(0), uSurfDim-1), 0).r; }
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 c = texelFetch(uIn, p, 0).rgb*255.;
    ivec2 s = ivec2(vec2(p)*vec2(uSurfDim)/vec2(uDim));
    float gx = h(s+ivec2(1,-1))+2.*h(s+ivec2(1,0))+h(s+ivec2(1,1)) - h(s+ivec2(-1,-1))-2.*h(s+ivec2(-1,0))-h(s+ivec2(-1,1));
    float gy = h(s+ivec2(-1,1))+2.*h(s+ivec2(0,1))+h(s+ivec2(1,1)) - h(s+ivec2(-1,-1))-2.*h(s+ivec2(0,-1))-h(s+ivec2(1,-1));
    float emb = clamp((gx*cos(uAngle)+gy*sin(uAngle))/uSc, -1.5, 1.5);
    float shade = 1. + emb*(.55*uRelief*uStrength);
    float tone = 1. - (1.-h(s))*(.22*uStrength);
    o = vec4(clamp(c*shade*tone,0.,255.)/255., 1.); }`;
  /* Frame: the picture inside the frame overlay */
  const FS_FRAME = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform sampler2D uFrame; uniform ivec2 uInDim; uniform vec2 uOutDim;
  uniform int uBw; uniform int uRaw; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec4 f = texture(uFrame, (vec2(p)+.5)/uOutDim);
    if(uRaw == 1){ vec3 c = texelFetch(uIn, p, 0).rgb; o = vec4(mix(c, f.rgb, f.a), 1.); return; }
    ivec2 q = p - ivec2(uBw);
    if(q.x >= 0 && q.y >= 0 && q.x < uInDim.x && q.y < uInDim.y) o = vec4(texelFetch(uIn, q, 0).rgb, 1.);
    else o = vec4(f.rgb, 1.); }`;
  /* Text: inscription and logo, at the Opacity slider */
  const FS_TEXT = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform sampler2D uText; uniform sampler2D uLogo; uniform vec2 uDim;
  uniform float uOp; uniform int uLogoOn; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 uv = (vec2(p)+.5)/uDim;
    vec3 c = texelFetch(uIn, p, 0).rgb; vec4 t = texture(uText, uv);
    c = mix(c, t.rgb, t.a*uOp);
    if(uLogoOn == 1){ vec4 l = texture(uLogo, uv); c = mix(c, l.rgb, l.a*uOp); }
    o = vec4(c, 1.); }`;
  /* to the screen (images are stored top row first) */
  const FS_BLIT = `#version 300 es
  ${GLSL_COMMON}
  uniform sampler2D uIn; uniform ivec2 uDim; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); o = vec4(texelFetch(uIn, ivec2(p.x, uDim.y-1-p.y), 0).rgb, 1.); }`;

  /* ============================================================ renderer */
  class Live {
    constructor() {
      this.supported = false;
      this.active = false;            /* the After view shows this canvas */
      this.mode = 'A';
      this.params = null;
      this.order = ['Light', 'Canvas', 'Frame', 'Text'];
      this.overlays = {};
      this.listeners = [];
      this.planBusy = false; this.wantPlan = null;
      this.draftBusy = false; this.wantDraft = null;
      this.raf = 0;
      this.dirty = { strokes: true, height: true };
      try { this._init(); } catch (err) { console.warn('Live preview unavailable:', err); }
    }

    _init() {
      const cv = document.createElement('canvas');
      cv.id = 'live-canvas';
      cv.hidden = true;
      const gl = cv.getContext('webgl2', { antialias: false, premultipliedAlpha: false,
        preserveDrawingBuffer: true, alpha: false });
      if (!gl) return;
      if (!gl.getExtension('EXT_color_buffer_float')) return;
      this.canvas = cv; this.gl = gl;
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      this.prog = {
        stroke: this._program(VS_STROKE, FS_STROKE),
        ground: this._program(VS_FULL, FS_GROUND), hblur: this._program(VS_FULL, FS_HBLUR),
        emb: this._program(VS_FULL, FS_EMB), finish: this._program(VS_FULL, FS_FINISH),
        light1: this._program(VS_FULL, FS_LIGHT1), hot: this._program(VS_FULL, FS_HOT),
        gauss: this._program(VS_FULL, FS_GAUSS), light2: this._program(VS_FULL, FS_LIGHT2),
        canvas: this._program(VS_FULL, FS_CANVAS), frame: this._program(VS_FULL, FS_FRAME),
        text: this._program(VS_FULL, FS_TEXT), blit: this._program(VS_FULL, FS_BLIT)
      };
      this.emptyVao = gl.createVertexArray();
      this.dummy = this._tex(1, 1, 'rgba8', new Uint8Array([0, 0, 0, 0]));
      this.pool = {};
      this.supported = true;
    }

    /* -------------------------------------------------- GL plumbing */
    _shader(type, src) {
      const gl = this.gl, s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    _program(vs, fs) {
      const gl = this.gl, p = gl.createProgram();
      gl.attachShader(p, this._shader(gl.VERTEX_SHADER, vs));
      gl.attachShader(p, this._shader(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) {
        const info = gl.getActiveUniform(p, i);
        u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name);
      }
      return { p, u };
    }
    _fmt(kind) {
      const gl = this.gl;
      return {
        rgba8: [gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR],
        r16f: [gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST],
        r32f: [gl.R32F, gl.RED, gl.FLOAT, gl.NEAREST],
        rgba16f: [gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR],
        rgba32f: [gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST]
      }[kind];
    }
    _tex(w, h, kind, data = null, tex = null) {
      const gl = this.gl, [ifmt, fmt, type, filter] = this._fmt(kind);
      tex = tex || gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      tex.w = w; tex.h = h;
      return tex;
    }
    _texFromImage(bitmap, tex = null) {
      const gl = this.gl;
      tex = tex || gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      tex.w = bitmap.width; tex.h = bitmap.height;
      return tex;
    }
    /* a render target from the pool, by name and size */
    _target(name, w, h, kind = 'rgba8') {
      const gl = this.gl;
      if (name.startsWith('stage')) name += '_' + w + 'x' + h;
      let t = this.pool[name];
      if (!t || t.w !== w || t.h !== h || t.kind !== kind) {
        if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
        const tex = this._tex(w, h, kind);
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        t = this.pool[name] = { tex, fbo, w, h, kind };
      }
      t.used = this.frameNo;
      return t;
    }
    _evict() {
      for (const [k, t] of Object.entries(this.pool)) {
        if (k.startsWith('stage') && t.used !== this.frameNo) {
          this.gl.deleteTexture(t.tex); this.gl.deleteFramebuffer(t.fbo); delete this.pool[k];
        }
      }
    }
    _bindTarget(t) {
      const gl = this.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, t ? t.fbo : null);
      gl.viewport(0, 0, t ? t.w : this.canvas.width, t ? t.h : this.canvas.height);
    }
    _use(prog, uniforms) {
      const gl = this.gl;
      gl.useProgram(prog.p);
      let unit = 0;
      for (const [k, v] of Object.entries(uniforms)) {
        const loc = prog.u[k];
        if (loc === undefined || loc === null) continue;
        if (v && v.tex !== undefined) {           /* render target */
          gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, v.tex); gl.uniform1i(loc, unit++);
        } else if (v instanceof WebGLTexture) {
          gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, v); gl.uniform1i(loc, unit++);
        } else if (Array.isArray(v)) {
          const f = v.isInt ? ['uniform1iv', 'uniform2iv', 'uniform3iv'] : ['uniform1fv', 'uniform2fv', 'uniform3fv'];
          gl[f[v.length - 1]](loc, v);
        } else if (typeof v === 'object' && v.i !== undefined) gl.uniform1i(loc, v.i);
        else gl.uniform1f(loc, v);
      }
    }
    _full() { const gl = this.gl; gl.bindVertexArray(this.emptyVao); gl.drawArrays(gl.TRIANGLES, 0, 3); }

    /* -------------------------------------------------- public API */
    onChange(fn) { this.listeners.push(fn); }
    _emit(what) { this.listeners.forEach(fn => fn(what)); }
    get size() { return this.outSize || [0, 0]; }
    get previewing() { return this.mode !== 'A'; }

    /* the painting changed on the server (Run, Timeline, Photoshop, open) */
    async load() {
      if (!this.supported) return false;
      const st = await (await fetch('/api/live/state', { cache: 'no-store' })).json();
      if (!st.painted) { this.active = false; this.state = null; return false; }
      const get = name => fetch('/api/live/layer/' + name + '?v=' + st.version + '_' + Date.now(), { cache: 'no-store' });
      const [rc, ro, rh, re] = await Promise.all(['canvas', 'original', 'height', 'edits'].map(get));
      const [bc, bo] = await Promise.all([rc.blob(), ro.blob()].map(async b => decodeImage(await b)));
      this.state = st;
      this.P = st.size;
      this.texCanvas = this._texFromImage(bc, this.texCanvas);
      this.texOrig = this._texFromImage(bo, this.texOrig);
      if (rh.status === 200) {
        const f = new Float32Array(await rh.arrayBuffer());
        this.texHeightA = this._tex(this.P[0], this.P[1], 'r32f', f, this.texHeightA);
        this.reliefScaleA = parseFloat(rh.headers.get('X-Relief-Scale')) || 1;
      } else this.texHeightA = null;
      await this._loadEdits(re);
      this.plan = null; this.lookKey = null; this.draftKey = null;
      this.dirty.strokes = true;
      this.ranStrokeKey = st.ran ? strokeKey(st.ran) : null;
      this.ranLayoutKey = st.ran ? layoutKey(st.ran) : null;
      this.dirty.heightA = true;
      this.active = true;
      if (this.params) this.setParams(this.params, { dragging: false, order: this.order });
      else { this.mode = 'A'; this.invalidate(); }
      return true;
    }
    async _loadEdits(resp) {
      resp = resp || await fetch('/api/live/layer/edits?t=' + Date.now(), { cache: 'no-store' });
      if (resp.status === 200) {
        const bm = await decodeImage(await resp.blob());
        this.texEdits = this._texFromImage(bm, this.texEdits);
        this.editsOn = true;
      } else this.editsOn = false;
    }
    async reloadEdits() {
      if (!this.active) return;
      await this._loadEdits();
      this.invalidate();
    }

    /* every settings change goes through here */
    setParams(p, { dragging = false, order } = {}) {
      this.params = p;
      if (order) this.order = order;
      if (!this.active || !this.state) return;
      const sk = strokeKey(p);
      let mode = 'A';
      if (this.ranStrokeKey !== null && sk !== this.ranStrokeKey)
        mode = ['Canvas Oil', 'Halftone', 'Painted Photo', 'Impression'].includes(p.paint_style) ? 'C' : 'B';
      /* a Timeline shot or a Photoshop result has no strokes to restyle */
      if (!this.state.ran) mode = 'A';
      const modeChanged = mode !== this.mode;
      this.mode = mode;

      if (mode === 'B') {
        const lk = layoutKey(p);
        if (this.plan && this.plan.layoutKey === lk && !(this.plan.proxy && !dragging)) {
          /* same strokes: only their look changed */
          if (this.lookKey !== this._lookKey(p)) this._restyle();
        } else if (lk === this.ranLayoutKey && this.state.has_plan) {
          this._requestPlan({ source: 'run', key: lk });
        } else {
          this._requestPlan({ source: 'preview', params: p, proxy: dragging, key: lk });
        }
        if (this.plan && this.lookKey !== this._lookKey(p)) this._restyle();
      } else if (mode === 'C') {
        this._requestDraft(p, dragging);
      }
      if (modeChanged) this._emit('mode');
      this._overlaysFor();
      this.invalidate();
    }

    _lookKey(p) { return JSON.stringify(LOOK_KEYS.map(k => p[k])); }
    _restyle() {
      if (!this.plan) return;
      const { data, rows } = styleStrokes(this.plan, this.params);
      this.texData = this._tex(DATA_W, rows, 'rgba32f', data, this.texData);
      this.lookKey = this._lookKey(this.params);
      this.dirty.strokes = true;
    }

    /* -------------------------------------------------- stroke plans */
    _requestPlan(want) {
      want.id = want.key + (want.source === 'run' ? '|run' : (want.proxy ? '|proxy' : '|full'));
      if (this.plan && this.plan.id === want.id) return;
      if (this.inflightPlan === want.id) return;
      this.wantPlan = want;
      this._pumpPlan();
    }
    async _pumpPlan() {
      if (this.planBusy || !this.wantPlan) return;
      const want = this.wantPlan; this.wantPlan = null;
      this.planBusy = true; this.inflightPlan = want.id;
      this._emit('busy');
      try {
        const res = await fetch('/api/live/plan', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: want.source, params: want.params, proxy: !!want.proxy })
        });
        if (res.status === 200) {
          const plan = parsePlan(await res.arrayBuffer());
          plan.layoutKey = want.key; plan.id = want.id; plan.proxy = !!want.proxy;
          this._setPlan(plan);
        }
      } catch (err) { console.warn(err); }
      this.planBusy = false; this.inflightPlan = null;
      this._emit('busy');
      if (this.wantPlan) this._pumpPlan();
    }
    /* ribbons of a plan into vertex/index buffers (a vertex array) */
    _uploadGeometry(geo, mesh = {}) {
      const gl = this.gl;
      if (!mesh.vao) {
        mesh.vao = gl.createVertexArray();
        mesh.vbo = gl.createBuffer(); mesh.ibo = gl.createBuffer();
        gl.bindVertexArray(mesh.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, mesh.vbo);
        const stride = 32;
        [[0, 2, 0], [1, 2, 8], [2, 1, 16], [3, 1, 20], [4, 1, 24], [5, 1, 28]].forEach(([loc, n, off]) => {
          gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, n, gl.FLOAT, false, stride, off);
        });
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.ibo);
      }
      gl.bindVertexArray(mesh.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.vbo);
      gl.bufferData(gl.ARRAY_BUFFER, geo.V, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.I, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      mesh.count = geo.count; mesh.ends = geo.ends;
      return mesh;
    }
    _groundTex(plan, tex) {
      const g = new Uint8Array(plan.gw * plan.gh * 4);
      for (let i = 0, j = 0; i < plan.gw * plan.gh; i++) {
        g[j++] = plan.ground[i * 3]; g[j++] = plan.ground[i * 3 + 1]; g[j++] = plan.ground[i * 3 + 2]; g[j++] = 255;
      }
      return this._tex(plan.gw, plan.gh, 'rgba8', g, tex);
    }
    _setPlan(plan) {
      this.plan = plan;
      this.mesh = this._uploadGeometry(buildGeometry(plan), this.mesh);
      this.vao = this.mesh.vao;
      this.indexCount = this.mesh.count;
      this.texGround = this._groundTex(plan, this.texGround);
      this._restyle();
      this.invalidate();
    }

    /* ---------------------------------------------- Run on the GPU */
    /* Paint a whole plan at full size on the graphics card - the fast Run.
       Returns the bare painting (PNG), the paint height (float16) and the
       Timeline shots (JPEG), or null when stopped.                        */
    async paintFull(plan, p, { snapshots = 8, onProgress, shouldStop } = {}) {
      const gl = this.gl, w = plan.w, h = plan.h;
      const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      if (w > max || h > max) throw new Error('The picture is too large for this graphics card.');
      const t0 = performance.now();
      const mesh = this._uploadGeometry(buildGeometry(plan));
      const { data, rows } = styleStrokes(plan, p);
      const texData = this._tex(DATA_W, rows, 'rgba32f', data);
      const texGround = this._groundTex(plan);
      const col = this._target('runPaint', w, h, 'rgba8');
      const hgt = this._target('runHeight', w, h, 'r16f');
      const cleanup = () => {
        gl.deleteVertexArray(mesh.vao); gl.deleteBuffer(mesh.vbo); gl.deleteBuffer(mesh.ibo);
        gl.deleteTexture(texData); gl.deleteTexture(texGround);
        for (const k of ['runPaint', 'runHeight']) {
          const t = this.pool[k]; if (!t) continue;
          gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); delete this.pool[k];
        }
      };
      try {
        /* the canvas before painting */
        gl.disable(gl.BLEND);
        this._bindTarget(col);
        this._use(this.prog.ground, { uTex: texGround, uSize: [w, h],
          uCorr: [Number(p.cor_hue), Number(p.cor_saturation), Number(p.cor_brightness)],
          uCorrOn: { i: correctionOn(p) ? 1 : 0 } });
        this._full();
        this._bindTarget(hgt);
        gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);

        /* the strokes, in order, a chunk at a time: a Timeline shot after each */
        const chunks = Math.max(1, snapshots + 1), shots = [];
        let doneStrokes = 0;
        for (let c = 1; c <= chunks; c++) {
          if (shouldStop && shouldStop()) return null;
          const upto = Math.round(plan.n * c / chunks);
          const i0 = doneStrokes ? mesh.ends[doneStrokes - 1] : 0;
          const i1 = upto ? mesh.ends[upto - 1] : 0;
          if (i1 > i0) {
            gl.enable(gl.BLEND);
            gl.bindVertexArray(mesh.vao);
            this._bindTarget(col);
            gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE);
            this._use(this.prog.stroke, { uData: texData, uSize: [w, h], uPass: { i: 0 } });
            gl.drawElements(gl.TRIANGLES, i1 - i0, gl.UNSIGNED_INT, i0 * 4);
            this._bindTarget(hgt);
            gl.blendFunc(gl.ONE, gl.ONE);
            this._use(this.prog.stroke, { uData: texData, uSize: [w, h], uPass: { i: 1 } });
            gl.drawElements(gl.TRIANGLES, i1 - i0, gl.UNSIGNED_INT, i0 * 4);
            gl.bindVertexArray(null);
            gl.disable(gl.BLEND);
          }
          doneStrokes = upto;
          if (c < chunks) shots.push(await this._readColour(col, 'image/jpeg', 0.92));
          if (onProgress) onProgress(c / chunks);
          await nextFrame();
        }
        const canvas = await this._readColour(col, 'image/png');
        const height = this._readHeight(hgt);
        return { canvas, height, shots, w, h, seconds: (performance.now() - t0) / 1000 };
      } finally {
        cleanup();
        this.invalidate();
      }
    }
    /* an RGBA8 target as an image file (rows are already top first) */
    _readColour(t, type, quality) {
      const gl = this.gl;
      this._bindTarget(t);
      const px = new Uint8ClampedArray(t.w * t.h * 4);
      gl.readPixels(0, 0, t.w, t.h, gl.RGBA, gl.UNSIGNED_BYTE, px);
      const c2 = document.createElement('canvas');
      c2.width = t.w; c2.height = t.h;
      c2.getContext('2d').putImageData(new ImageData(px, t.w, t.h), 0, 0);
      return new Promise(res => c2.toBlob(res, type, quality));
    }
    /* the paint height as float16, read in bands to keep memory low */
    _readHeight(t) {
      const gl = this.gl, out = new Uint16Array(t.w * t.h), band = 128;
      this._bindTarget(t);
      const buf = new Float32Array(t.w * band * 4);
      for (let y = 0; y < t.h; y += band) {
        const n = Math.min(band, t.h - y);
        gl.readPixels(0, y, t.w, n, gl.RGBA, gl.FLOAT, buf);
        for (let i = 0, o = y * t.w; i < t.w * n; i++) out[o + i] = toHalf(buf[i * 4]);
      }
      return out;
    }

    /* -------------------------------------------------- Canvas Oil drafts */
    _requestDraft(p, dragging) {
      const key = strokeKey(p) + (dragging ? '|proxy' : '|full');
      if (this.draftKey === key || this.inflightDraft === key) return;
      this.wantDraft = { p, dragging, key };
      this._pumpDraft();
    }
    async _pumpDraft() {
      if (this.draftBusy || !this.wantDraft) return;
      const want = this.wantDraft; this.wantDraft = null;
      this.draftBusy = true; this.inflightDraft = want.key;
      this._emit('busy');
      try {
        const res = await fetch('/api/live/draft', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ params: want.p, proxy: want.dragging })
        });
        if (res.status === 200) {
          const bm = await decodeImage(await res.blob());
          this.texDraft = this._texFromImage(bm, this.texDraft);
          this.draftKey = want.key;
          this.invalidate();
        }
      } catch (err) { console.warn(err); }
      this.draftBusy = false; this.inflightDraft = null;
      this._emit('busy');
      if (this.wantDraft) this._pumpDraft();
    }
    get busy() { return this.planBusy || this.draftBusy; }

    /* -------------------------------------------------- decoration layers */
    /* the size of the picture as it enters each decoration */
    _stages() {
      const p = this.params || {};
      let [w, h] = this.P;
      const out = [];
      for (const name of this.order) {
        const on = !!Number(p[name.toLowerCase() + '_enabled']);
        const st = { name, on, w, h };
        if (name === 'Frame' && on) {
          const bw = Math.max(4, Math.floor(Math.min(h, w) * Number(p.frame_width) / 100));
          st.bw = bw; st.raw = p.frame_style === 'Raw Canvas Edge';
          if (!st.raw) { w += 2 * bw; h += 2 * bw; }
          st.ow = w; st.oh = h;
        }
        if (name === 'Text' && on && Number(p.text_opacity) <= 0) st.on = false;
        out.push(st);
      }
      this.outSize = [w, h];
      return out;
    }
    _overlaysFor() {
      if (!this.params || !this.P) return;
      const p = this.params;
      const pick = pre => Object.keys(p).filter(k => k.startsWith(pre) && k !== pre + 'opacity' &&
        !['canvas_relief', 'canvas_light', 'canvas_strength'].includes(k)).sort().map(k => [k, p[k]]);
      for (const st of this._stages()) {
        if (!st.on) continue;
        if (st.name === 'Frame') this._overlay('frame', st.w, st.h, pick('frame_'));
        if (st.name === 'Text') {
          this._overlay('text', st.w, st.h, pick('text_'));
          this._overlay('logo', st.w, st.h, [['text_size', p.text_size], ['logo', this.logoVersion || 0]]);
        }
        if (st.name === 'Canvas') this._overlay('surface', st.w, st.h,
          [['canvas_texture', p.canvas_texture], ['canvas_scale', p.canvas_scale], ['custom', this.textureVersion || 0]]);
      }
    }
    _overlay(kind, w, h, deps) {
      const key = JSON.stringify([w, h, deps]);
      const ov = this.overlays[kind] || (this.overlays[kind] = {});
      if (ov.key === key || ov.inflight === key) return;
      ov.want = { key, w, h };
      if (!ov.inflight) this._pumpOverlay(kind);
    }
    async _pumpOverlay(kind) {
      const ov = this.overlays[kind];
      if (!ov.want) return;
      const want = ov.want; ov.want = null; ov.inflight = want.key;
      try {
        const res = await fetch('/api/live/overlay', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind, params: this.params, size: [want.w, want.h] })
        });
        if (res.status === 200 && kind === 'surface') {
          ov.scales = (res.headers.get('X-Scales') || '1').split(',').map(Number);
          ov.tex = this._tex(want.w, want.h, 'r32f', new Float32Array(await res.arrayBuffer()), ov.tex);
          ov.ready = true;
        } else if (res.status === 200) {
          const bm = await decodeImage(await res.blob());
          ov.tex = this._texFromImage(bm, ov.tex);
          ov.ready = true;
        } else if (res.status === 204) ov.ready = false;
        ov.key = want.key;
      } catch (err) { console.warn(err); }
      ov.inflight = null;
      this.invalidate();
      if (ov.want) this._pumpOverlay(kind);
    }
    customChanged(kind) {
      if (kind === 'logo') this.logoVersion = (this.logoVersion || 0) + 1;
      else this.textureVersion = (this.textureVersion || 0) + 1;
      this._overlaysFor();
    }

    /* -------------------------------------------------- rendering */
    invalidate() {
      if (this.raf || !this.active) return;
      this.raf = requestAnimationFrame(() => { this.raf = 0; this.render(); });
    }

    _paintStrokes() {
      const gl = this.gl, [w, h] = this.P, p = this.params;
      const col = this._target('paint', w, h, 'rgba8');
      const hgt = this._target('height', w, h, 'r16f');
      /* the canvas before painting */
      this._bindTarget(col);
      gl.disable(gl.BLEND);
      this._use(this.prog.ground, { uTex: this.texGround, uSize: [w, h],
        uCorr: [Number(p.cor_hue), Number(p.cor_saturation), Number(p.cor_brightness)],
        uCorrOn: { i: correctionOn(p) ? 1 : 0 } });
      this._full();
      this._bindTarget(hgt);
      gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      if (!this.indexCount) return;
      /* strokes, in order: wet paint over wet paint, paint height adding up */
      gl.enable(gl.BLEND);
      gl.bindVertexArray(this.vao);
      this._bindTarget(col);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE);
      this._use(this.prog.stroke, { uData: this.texData, uSize: [w, h], uPass: { i: 0 } });
      gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);
      this._bindTarget(hgt);
      gl.blendFunc(gl.ONE, gl.ONE);
      this._use(this.prog.stroke, { uData: this.texData, uSize: [w, h], uPass: { i: 1 } });
      gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }

    /* blurred paint height, and the scale apply_relief divides by */
    _prepareHeight(src, readScale, name) {
      const gl = this.gl, [w, h] = this.P;
      const hb = this._target(name, w, h, 'r16f');
      this._bindTarget(hb);
      this._use(this.prog.hblur, { uH: src, uDim: Object.assign([w, h], { isInt: true }) });
      this._full();
      if (!readScale) return;
      const n = 128, t = this._target('emb', n, n, 'rgba32f');
      this._bindTarget(t);
      this._use(this.prog.emb, { uH: hb, uDim: Object.assign([w, h], { isInt: true }), uStride: [w / n, h / n] });
      this._full();
      const buf = new Float32Array(n * n * 4);
      gl.readPixels(0, 0, n, n, gl.RGBA, gl.FLOAT, buf);
      const v = new Float32Array(n * n);
      for (let i = 0; i < n * n; i++) v[i] = buf[i * 4];
      v.sort();
      this.reliefScaleB = v[Math.floor(0.97 * (v.length - 1))] + 1e-6;
    }

    render() {
      if (!this.active || !this.P || !this.params) return;
      const gl = this.gl, p = this.params, [w, h] = this.P;
      const ivec = a => Object.assign(a, { isInt: true });
      let colour, steps, scale = 1, hblur = null;
      this.frameNo = (this.frameNo || 0) + 1;

      if (this.mode === 'B' && this.plan) {
        if (this.dirty.strokes) {
          this._paintStrokes();
          this._prepareHeight(this.pool.height, true, 'hblurB');
          this.dirty.strokes = false;
        }
        colour = this.pool.paint; steps = 7; scale = this.reliefScaleB; hblur = this.pool.hblurB;
      } else if (this.mode === 'C' && this.texDraft) {
        colour = this.texDraft; steps = 4;
      } else {
        colour = this.texCanvas;
        steps = (this.state.finish.includes('relief') ? 1 : 0) |
          (this.state.finish.includes('lightening') ? 2 : 0) | (this.state.finish.includes('blend') ? 4 : 0);
        if (this.texHeightA) {
          if (this.dirty.heightA) { this._prepareHeight(this.texHeightA, false, 'hblurA'); this.dirty.heightA = false; }
          hblur = this.pool.hblurA; scale = this.reliefScaleA;
        }
      }
      if (!hblur) steps &= ~1;

      /* finishing */
      const fin = this._target('stageA', w, h);
      this._bindTarget(fin);
      this._use(this.prog.finish, {
        uCol: colour, uHB: hblur || this.dummy, uOrig: this.texOrig,
        uEdits: this.texEdits || this.dummy, uDim: ivec([w, h]),
        uRelief: Number(p.relief) / 100, uScale: scale, uLight: Number(p.lightening) / 100,
        uBlend: Number(p.blend_original) / 100, uMode: { i: Math.max(0, BLEND_MODES.indexOf(p.blend_mode)) },
        uSteps: { i: steps }, uEditsOn: { i: this.mode === 'A' && this.editsOn ? 1 : 0 }
      });
      this._full();

      /* decoration, in the order of the tabs */
      let cur = fin, flip = 'stageB';
      const next = (ow, oh) => { const t = this._target(flip, ow, oh); flip = flip === 'stageA' ? 'stageB' : 'stageA'; return t; };
      for (const st of this._stages()) {
        if (!st.on) continue;
        const iw = cur.w, ih = cur.h;
        if (st.name === 'Light') {
          const lp = [Number(p.light_x) / 100 * (iw - 1), Number(p.light_y) / 100 * (ih - 1)];
          const spread = Math.max(0.12, Number(p.light_spread) / 100);
          const a = next(iw, ih); this._bindTarget(a);
          this._use(this.prog.light1, { uIn: cur, uDim: [iw, ih], uLightPos: lp, uSpread: spread, uVig: Number(p.light_vignette) / 100 });
          this._full();
          const glow = Number(p.light_glow) / 100;
          let bloom = this.dummy;
          if (glow > 0.001) {
            const hw = Math.max(1, Math.ceil(iw / 2)), hh = Math.max(1, Math.ceil(ih / 2));
            const g1 = this._target('glow1', hw, hh, 'rgba16f'), g2 = this._target('glow2', hw, hh, 'rgba16f');
            this._bindTarget(g1); this._use(this.prog.hot, { uIn: a, uDim: ivec([iw, ih]) }); this._full();
            const sigma = Math.max(3, Math.min(iw, ih) * 0.022) / 2;
            this._bindTarget(g2); this._use(this.prog.gauss, { uIn: g1, uDim: ivec([hw, hh]), uDir: ivec([1, 0]), uSigma: Math.min(sigma, 16) }); this._full();
            this._bindTarget(g1); this._use(this.prog.gauss, { uIn: g2, uDim: ivec([hw, hh]), uDir: ivec([0, 1]), uSigma: Math.min(sigma, 16) }); this._full();
            bloom = g1;
          }
          const b = next(iw, ih); this._bindTarget(b);
          this._use(this.prog.light2, { uIn: a, uBloom: bloom, uDim: [iw, ih], uLightPos: lp, uSpread: spread,
            uGlow: glow, uWarm: Number(p.light_warmth) / 100 });
          this._full();
          cur = b;
        } else if (st.name === 'Canvas') {
          const ov = this.overlays.surface;
          if (!ov || !ov.ready) continue;
          const lightT = Number(p.canvas_light) / 100;
          const ang = (45 + 270 * lightT) * Math.PI / 180;
          const fi = lightT * 72, i0 = Math.floor(fi), i1 = Math.min(72, i0 + 1);
          const sc = ov.scales.length > 1 ? lerp(ov.scales[i0], ov.scales[i1], fi - i0) : ov.scales[0];
          const t = next(iw, ih); this._bindTarget(t);
          this._use(this.prog.canvas, { uIn: cur, uSurf: ov.tex, uDim: ivec([iw, ih]), uSurfDim: ivec([ov.tex.w, ov.tex.h]),
            uAngle: ang, uSc: sc, uRelief: Number(p.canvas_relief) / 100, uStrength: Number(p.canvas_strength) / 100 });
          this._full();
          cur = t;
        } else if (st.name === 'Frame') {
          const ov = this.overlays.frame;
          if (!ov || !ov.ready) continue;
          const t = next(st.ow, st.oh); this._bindTarget(t);
          this._use(this.prog.frame, { uIn: cur, uFrame: ov.tex, uInDim: ivec([iw, ih]), uOutDim: [st.ow, st.oh],
            uBw: { i: st.bw }, uRaw: { i: st.raw ? 1 : 0 } });
          this._full();
          cur = t;
        } else if (st.name === 'Text') {
          const ov = this.overlays.text, lg = this.overlays.logo;
          if (!ov || !ov.ready) continue;
          const t = next(iw, ih); this._bindTarget(t);
          this._use(this.prog.text, { uIn: cur, uText: ov.tex, uLogo: (lg && lg.ready) ? lg.tex : this.dummy,
            uDim: [iw, ih], uOp: Number(p.text_opacity) / 100, uLogoOn: { i: lg && lg.ready ? 1 : 0 } });
          this._full();
          cur = t;
        }
      }

      /* to the screen */
      if (this.canvas.width !== cur.w || this.canvas.height !== cur.h) {
        this.canvas.width = cur.w; this.canvas.height = cur.h;
      }
      this.outSize = [cur.w, cur.h];
      this._bindTarget(null);
      this._use(this.prog.blit, { uIn: cur, uDim: ivec([cur.w, cur.h]) });
      this._full();
      this._evict();
      this._emit('render');
    }

    /* the rendered After image as PNG (for tests and comparisons) */
    snapshot() { return this.canvas.toDataURL('image/png'); }
  }

  window.OilLive = { Live, layoutKey, strokeKey, isPost, LOOK_KEYS, styleStrokes, parsePlan };
})();
