/* OilStudio - "Oil Paint FX" style on the graphics card (WebGL 2) -------
   The same steps as oilpaint/oilfx.py, pass by pass, so a GPU Run and a
   CPU Run give the same picture.  Keep both files in step.

     surface blur (bilateral) -> oil paint (smoothing along the image's own
     structure) -> high pass x2 (overlay) -> local contrast + 2 sharpens ->
     vibrance + fitted tone curve -> colour tone, crack texture, grunge
     mask over the yellow ground -> vibrance, curves, colour balance

   Heavy passes are drawn in bands with a pause between them, so a big
   picture never runs one long draw (Windows resets a GPU that is busy for
   ~2 s, which used to drop the WebGL context).                           */
'use strict';

(() => {
  const REF_SHORT = 2000;
  const C = {               /* the template's settings - see oilfx.py */
    SB_SIGMA_COLOR: 0.08, SB_RADIUS: 12, FLOW_SIGMA: 4, LIC_LEN: 6, LIC_STEPS: 4,
    LOCAL_AMT: 0.4, LOCAL_SIGMA: 12, SHARP1_AMT: 1.93, SHARP1_SIGMA: 1.5,
    SHARP2_AMT: 2.0, SHARP2_SIGMA: 0.6, VIB: 0.5, CB_AMOUNT: 0.35,
    BG: [227 / 255, 216 / 255, 128 / 255]
  };
  const BAND = 192;         /* rows per draw in the heavy passes */
  /* memory kept on the card between previews: the reused render targets,
     and the kept stages (only for preview-sized pictures). Integrated
     graphics share the computer's memory - too much and Windows resets
     the card ("the graphics card disconnected") */
  const POOL_BYTES = 160 * 1024 * 1024;
  const MEMO_MAX_PX = 2.7e6;

  const VS = `#version 300 es
  void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2. - 1., 0., 1.); }`;
  const HEAD = `#version 300 es
  precision highp float; precision highp int;
  uniform ivec2 uDim; out vec4 o;
  /* OpenCV BORDER_REFLECT: abc|cba */
  ivec2 refl(ivec2 p){ p = ivec2(p.x < 0 ? -p.x - 1 : p.x, p.y < 0 ? -p.y - 1 : p.y);
    return ivec2(p.x >= uDim.x ? 2 * uDim.x - p.x - 1 : p.x, p.y >= uDim.y ? 2 * uDim.y - p.y - 1 : p.y); }
  vec4 at(sampler2D t, ivec2 p){ return texelFetch(t, refl(p), 0); }
  float lum(vec3 c){ return dot(c, vec3(.299, .587, .114)); }
  vec3 ovl(vec3 b, vec3 s){ return mix(1. - 2. * (1. - b) * (1. - s), 2. * b * s, step(b, vec3(.4999999))); }
  `;
  /* Canvas Oil: hash noise - the same integer maths as canvas_oil.py (rnd, vn) */
  const CO = HEAD + `
  uint hh(uint x){ x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
  float rnd(float cx, float cy, float k){
    uint v = hh((uint(int(cx)) * 0x8da6b343u) ^ hh((uint(int(cy)) * 0xd8163841u) ^ hh(uint(int(k)))));
    return float(v >> 8u) * (1. / 16777216.); }
  float vn(float x, float y, float k){ float fx = floor(x), fy = floor(y), tx = x - fx, ty = y - fy;
    float ux = tx * tx * (3. - 2. * tx), uy = ty * ty * (3. - 2. * ty);
    float a = rnd(fx, fy, k), b = rnd(fx + 1., fy, k), c = rnd(fx, fy + 1., k), d = rnd(fx + 1., fy + 1., k);
    float top = a + (b - a) * ux, bot = c + (d - c) * ux; return top + (bot - top) * uy; }
  float sst(float e0, float e1, float x){ float t = clamp((x - e0) / (e1 - e0), 0., 1.); return t * t * (3. - 2. * t); }
  const vec3 CW = vec3(244., 242., 238.) / 255., PALE = vec3(220., 218., 215.) / 255., INK = vec3(42., 28., 20.) / 255.;
  `;
  const FS = {
    /* bilateral, like cv2.bilateralFilter (L1 colour distance, round window) */
    bilateral: HEAD + `uniform sampler2D uIn; uniform int uR; uniform int uStep; uniform float uSs; uniform float uSc;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 c0 = at(uIn, p).rgb;
      vec3 acc = vec3(0.); float ws = 0.;
      for (int y = -uR; y <= uR; y += uStep) for (int x = -uR; x <= uR; x += uStep) {
        float r2 = float(x * x + y * y); if (r2 > float(uR * uR)) continue;
        vec3 c = at(uIn, p + ivec2(x, y)).rgb; vec3 d = abs(c - c0); float dc = d.r + d.g + d.b;
        float w = exp(-r2 / (2. * uSs * uSs) - dc * dc / (2. * uSc * uSc));
        acc += c * w; ws += w; }
      o = vec4(acc / ws, 1.); }`,
    /* one direction of a Gaussian blur (radius 4 sigma, like OpenCV) */
    gauss: HEAD + `uniform sampler2D uIn; uniform ivec2 uDir; uniform float uS;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); int r = int(ceil(uS * 4.));
      vec4 acc = vec4(0.); float ws = 0.;
      for (int i = -r; i <= r; i++) { float w = exp(-float(i * i) / (2. * uS * uS)); acc += at(uIn, p + uDir * i) * w; ws += w; }
      o = acc / ws; }`,
    luma: HEAD + `uniform sampler2D uIn; void main(){ o = vec4(lum(at(uIn, ivec2(gl_FragCoord.xy)).rgb)); }`,
    /* Sobel 3x3 -> structure tensor terms (gx^2, gy^2, gx gy) */
    tensor: HEAD + `uniform sampler2D uIn;
    float L(ivec2 p){ return at(uIn, p).r; }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      float gx = (L(p+ivec2(1,-1)) + 2.*L(p+ivec2(1,0)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(-1,0)) + L(p+ivec2(-1,1)));
      float gy = (L(p+ivec2(-1,1)) + 2.*L(p+ivec2(0,1)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(0,-1)) + L(p+ivec2(1,-1)));
      o = vec4(gx * gx, gy * gy, gx * gy, 1.); }`,
    /* smoothing along a short straight line in the stroke direction */
    lic: HEAD + `uniform sampler2D uIn; uniform sampler2D uT; uniform float uLen; uniform int uSteps;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 t = texelFetch(uT, p, 0).rgb;
      float a = .5 * atan(2. * t.b, t.r - t.g) + 1.5707963;
      vec2 d = vec2(cos(a), sin(a)); vec2 sz = vec2(uDim);
      vec3 acc = texelFetch(uIn, p, 0).rgb; float st = uLen / float(uSteps);
      for (int i = 1; i <= uSteps; i++) for (int s = -1; s <= 1; s += 2) {
        vec2 q = vec2(p) + .5 + d * (st * float(i) * float(s));
        acc += texture(uIn, q / sz).rgb; }
      o = vec4(acc / float(1 + 2 * uSteps), 1.); }`,
    /* high pass in overlay: x over clip(x - blur + .5) */
    highpass: HEAD + `uniform sampler2D uIn; uniform sampler2D uB;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb, b = texelFetch(uB, p, 0).rgb;
      o = vec4(ovl(x, clamp(x - b + .5, 0., 1.)), 1.); }`,
    sharpen: HEAD + `uniform sampler2D uIn; uniform sampler2D uB; uniform float uAmt;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb, b = texelFetch(uB, p, 0).rgb;
      o = vec4(clamp(x + uAmt * (x - b), 0., 1.), 1.); }`,
    /* vibrance + the fitted tone curve (per channel, 256 entries) */
    colour: HEAD + `uniform sampler2D uIn; uniform sampler2D uLut; uniform float uVib; uniform float uAmt;
    float lut(float v, int c){ float t = clamp(v, 0., 1.) * 255.; int i0 = int(floor(t)); int i1 = min(i0 + 1, 255); float f = t - float(i0);
      return mix(texelFetch(uLut, ivec2(i0, 0), 0)[c], texelFetch(uLut, ivec2(i1, 0), 0)[c], f); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float l = lum(x); float s = max(x.r, max(x.g, x.b)) - min(x.r, min(x.g, x.b));
      x = clamp(l + (x - l) * (1. + uVib * (1. - s) * (1. - s)), 0., 1.);
      vec3 y = vec3(lut(x.r, 0), lut(x.g, 1), lut(x.b, 2));
      o = vec4(mix(x, y, uAmt), 1.); }`,
    /* tone, texture, mask over the ground, then the adjustment layers */
    composite: HEAD + `uniform sampler2D uIn; uniform sampler2D uTone; uniform sampler2D uTex; uniform sampler2D uMask;
    uniform sampler2D uCurve; uniform float uToneAmt, uTexAmt, uEdges, uAdjust, uCb; uniform vec3 uBg;
    float curve(float v){ float t = clamp(v, 0., 1.) * 255.; int i0 = int(floor(t)); int i1 = min(i0 + 1, 255);
      return mix(texelFetch(uCurve, ivec2(i0, 0), 0).r, texelFetch(uCurve, ivec2(i1, 0), 0).r, t - float(i0)); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      vec4 tn = texelFetch(uTone, p, 0); float ta = tn.a * uToneAmt; x = x * (1. - ta) + tn.rgb * ta;
      float tx = texelFetch(uTex, p, 0).r; x = x + (ovl(x, vec3(tx)) - x) * uTexAmt;
      float m = 1. - uEdges * (1. - texelFetch(uMask, p, 0).r); x = x * m + uBg * (1. - m);
      /* Vibrance 20 / Saturation 15 */
      float mx = max(x.r, max(x.g, x.b)), mn = min(x.r, min(x.g, x.b)); float l = lum(x);
      vec3 y = l + (x - l) * (1. + .2 * (1. - (mx - mn)) * (1. - (mx - mn)));
      l = lum(y); y = clamp(l + (y - l) * (1. + .15 * .6), 0., 1.);
      y = vec3(curve(y.r), curve(y.g), curve(y.b));
      /* Color Balance midtones +10/0/-12, preserve luminosity */
      l = lum(y); float w = clamp(1. - abs(2. * l - 1.), 0., 1.);
      y = clamp(y + vec3(.10, 0., -.12) * uCb * w, 0., 1.);
      y = clamp(y + (l - lum(y)), 0., 1.);
      x = x + (y - x) * uAdjust;
      o = vec4(clamp(x, 0., 1.), 1.); }`,
    /* --- Sketch Paint (7TH template) --- */
    /* the fitted tone curve alone */
    lut: HEAD + `uniform sampler2D uIn; uniform sampler2D uLut;
    float lut(float v, int c){ float t = clamp(v, 0., 1.) * 255.; int i0 = int(floor(t)); int i1 = min(i0 + 1, 255); float f = t - float(i0);
      return mix(texelFetch(uLut, ivec2(i0, 0), 0)[c], texelFetch(uLut, ivec2(i1, 0), 0)[c], f); }
    void main(){ vec3 x = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb; o = vec4(lut(x.r, 0), lut(x.g, 1), lut(x.b, 2), 1.); }`,
    /* 4-quadrant Kuwahara ((r+1)^2 boxes, pixel in a corner) - Dry Brush */
    kuwahara: HEAD + `uniform sampler2D uIn; uniform int uR;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 best = vec3(0.); float bv = 1e9;
      for (int q = 0; q < 4; q++) {
        ivec2 o0 = ivec2((q == 1 || q == 3) ? -uR : 0, (q >= 2) ? -uR : 0);
        vec3 m = vec3(0.), m2 = vec3(0.);
        for (int y = 0; y <= uR; y++) for (int x = 0; x <= uR; x++) { vec3 c = at(uIn, p + o0 + ivec2(x, y)).rgb; m += c; m2 += c * c; }
        float n = float((uR + 1) * (uR + 1)); m /= n; m2 /= n;
        vec3 v3 = m2 - m * m; float v = v3.r + v3.g + v3.b;
        if (v < bv) { bv = v; best = m; } }
      o = vec4(best, 1.); }`,
    /* posterize the brightness only (soft steps), colours kept */
    posterL: HEAD + `uniform sampler2D uIn; uniform float uN, uSoft, uMix;
    void main(){ vec3 x = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb; float L = lum(x);
      float t = L * uN; float f = t - floor(t); float q = (floor(t) + clamp((f - .5) * uSoft + .5, 0., 1.)) / uN;
      float r = (q + .02) / (L + .02); o = vec4(clamp(x + (x * r - x) * uMix, 0., 1.), 1.); }`,
    /* dark edges: darker than around (L1 = lum of a light blur, L2 = its blur) */
    edgeDark: HEAD + `uniform sampler2D uIn; uniform sampler2D uL1; uniform sampler2D uL2; uniform float uAmt;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float e = clamp(texelFetch(uL2, p, 0).r - texelFetch(uL1, p, 0).r, 0., 1.);
      o = vec4(clamp(x * (1. - uAmt * e), 0., 1.), 1.); }`,
    /* Photocopy line from the sharpened brightness and its blur */
    photocopy: HEAD + `uniform sampler2D uL; uniform sampler2D uB; uniform float uGain, uOff;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); float d = texelFetch(uB, p, 0).r - texelFetch(uL, p, 0).r;
      o = vec4(vec3(clamp(1. - d * uGain + uOff, 0., 1.)), 1.); }`,
    /* pencil sketch (overlay), Levels, ink outline (multiply) */
    sketchComposite: HEAD + `uniform sampler2D uIn; uniform sampler2D uSketch; uniform sampler2D uLine;
    uniform float uSk, uLv, uOut; uniform vec3 uLevels;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float sk = texelFetch(uSketch, p, 0).r; x = x + (ovl(x, vec3(sk)) - x) * uSk;
      vec3 v = clamp((x - uLevels.x / 255.) / ((uLevels.y - uLevels.x) / 255.), 0., 1.);
      v = pow(v, vec3(1. / uLevels.z)); x = x + (v - x) * uLv;
      float l = texelFetch(uLine, p, 0).r; x = x + (x * l - x) * uOut;
      o = vec4(clamp(x, 0., 1.), 1.); }`,
    /* --- Rough Brush (8TH template) --- */
    /* one direction of a box mean (like cv2.blur, reflected border) */
    box: HEAD + `uniform sampler2D uIn; uniform ivec2 uDir; uniform int uR;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec4 acc = vec4(0.);
      for (int i = -uR; i <= uR; i++) acc += at(uIn, p + uDir * i);
      o = acc / float(2 * uR + 1); }`,
    square: HEAD + `uniform sampler2D uIn; void main(){ vec4 x = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0); o = vec4(x.rgb * x.rgb, 1.); }`,
    guidedA: HEAD + `uniform sampler2D uM; uniform sampler2D uMM; uniform float uEps;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 m = texelFetch(uM, p, 0).rgb; vec3 v = texelFetch(uMM, p, 0).rgb - m * m;
      o = vec4(v / (v + uEps), 1.); }`,
    guidedB: HEAD + `uniform sampler2D uM; uniform sampler2D uA;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 m = texelFetch(uM, p, 0).rgb, a = texelFetch(uA, p, 0).rgb;
      o = vec4(m - a * m, 1.); }`,
    guidedOut: HEAD + `uniform sampler2D uIn; uniform sampler2D uA; uniform sampler2D uB;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      o = vec4(texelFetch(uA, p, 0).rgb * texelFetch(uIn, p, 0).rgb + texelFetch(uB, p, 0).rgb, 1.); }`,
    /* the stroke direction (cos, sin) from the structure tensor */
    dir: HEAD + `uniform sampler2D uT;
    void main(){ vec3 t = texelFetch(uT, ivec2(gl_FragCoord.xy), 0).rgb; float a = .5 * atan(2. * t.b, t.r - t.g) + 1.5707963;
      o = vec4(cos(a), sin(a), 0., 1.); }`,
    /* smoothing along the flow, following it as it bends */
    licTrace: HEAD + `uniform sampler2D uIn; uniform sampler2D uD; uniform float uLen; uniform int uSteps;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim); float st = uLen / float(uSteps);
      vec3 acc = texelFetch(uIn, p, 0).rgb;
      for (int s = 0; s < 2; s++) { float sg = s == 0 ? 1. : -1.;
        vec2 pos = vec2(p); vec2 pd = texelFetch(uD, p, 0).xy * sg;
        for (int i = 0; i < uSteps; i++) {
          vec2 v = texture(uD, (pos + .5) / sz).xy; if (dot(v, pd) < 0.) v = -v;
          pos += v * st; pd = v; acc += texture(uIn, (pos + .5) / sz).rgb; } }
      o = vec4(clamp(acc / float(1 + 2 * uSteps), 0., 1.), 1.); }`,
    posterize: HEAD + `uniform sampler2D uIn; uniform float uN;
    void main(){ vec3 x = clamp(texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb, 0., 1.);
      vec3 q = floor(x * 255. * uN / 256. + 1e-4); o = vec4(floor(q * 255. / (uN - 1.) + .5) / 255., 1.); }`,
    /* Cutout: colour x tone mask (alpha = mask), blurred, then gathered */
    levelMask: HEAD + `uniform sampler2D uIn; uniform float uV, uN;
    void main(){ vec3 x = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb; float t = clamp(floor(lum(x) * uN), 0., uN - 1.);
      float m = t == uV ? 1. : 0.; o = vec4(x * m, m); }`,
    levelAcc: HEAD + `uniform sampler2D uIn; uniform sampler2D uAcc; uniform sampler2D uB; uniform float uV, uN;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb; float t = clamp(floor(lum(x) * uN), 0., uN - 1.);
      vec4 b = texelFetch(uB, p, 0); vec3 acc = texelFetch(uAcc, p, 0).rgb;
      o = vec4(t == uV ? b.rgb / max(b.a, 1e-6) : acc, 1.); }`,
    mix: HEAD + `uniform sampler2D uA; uniform sampler2D uB; uniform float uT;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); o = vec4(mix(texelFetch(uA, p, 0).rgb, texelFetch(uB, p, 0).rgb, uT), 1.); }`,
    /* Displace through the brush map (stretched to fit, edges repeated) */
    displace: HEAD + `uniform sampler2D uIn; uniform sampler2D uMap; uniform vec2 uAmp;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim);
      float m = texelFetch(uMap, p, 0).r * 255.; vec2 off = vec2((m - 128.) / 128.);
      vec2 q = clamp(vec2(p) + off * uAmp, vec2(0.), sz - 1.);
      o = vec4(texture(uIn, (q + .5) / sz).rgb, 1.); }`,
    brushComposite: HEAD + `uniform sampler2D uIn; uniform sampler2D uTex; uniform sampler2D uCurve; uniform float uT, uAdjust, uVib, uSat;
    float curve(float v){ float t = clamp(v, 0., 1.) * 255.; int i0 = int(floor(t)); int i1 = min(i0 + 1, 255);
      return mix(texelFetch(uCurve, ivec2(i0, 0), 0).r, texelFetch(uCurve, ivec2(i1, 0), 0).r, t - float(i0)); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float tx = texelFetch(uTex, p, 0).r; x = clamp(x + (ovl(x, vec3(tx)) - x) * uT, 0., 1.);
      float mx = max(x.r, max(x.g, x.b)), mn = min(x.r, min(x.g, x.b)); float l = lum(x);
      vec3 y = l + (x - l) * (1. + uVib * (1. - (mx - mn)) * (1. - (mx - mn)));
      l = lum(y); y = clamp(l + (y - l) * (1. + uSat * .6), 0., 1.);
      y = vec3(curve(y.r), curve(y.g), curve(y.b));
      o = vec4(clamp(x + (y - x) * uAdjust, 0., 1.), 1.); }`,
    /* --- Watercolor (5TH template) --- */
    /* shrink by averaging (like cv2 INTER_AREA): uF source px per output px */
    down: HEAD + `uniform sampler2D uIn; uniform vec2 uF; uniform ivec2 uSrc;
    void main(){ vec2 p = floor(gl_FragCoord.xy); vec3 acc = vec3(0.); float n = 0.;
      int fx = int(ceil(uF.x)), fy = int(ceil(uF.y));
      for (int y = 0; y < fy; y++) for (int x = 0; x < fx; x++) {
        vec2 q = p * uF + (vec2(x, y) + .5) * uF / vec2(float(fx), float(fy));
        acc += texture(uIn, q / vec2(uSrc)).rgb; n += 1.; }
      o = vec4(acc / n, 1.); }`,
    /* back to full size, bilinear (like cv2 INTER_LINEAR) */
    up: HEAD + `uniform sampler2D uIn; void main(){ o = vec4(texture(uIn, gl_FragCoord.xy / vec2(uDim)).rgb, 1.); }`,
    satAdj: HEAD + `uniform sampler2D uIn; uniform float uS;
    void main(){ vec3 x = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb; float l = lum(x); o = vec4(clamp(l + (x - l) * uS, 0., 1.), 1.); }`,
    /* Crosshatch: mean along short +45 / -45 degree lines, mixed in */
    hatch: HEAD + `uniform sampler2D uIn; uniform float uLen, uMix;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim); vec3 x = texelFetch(uIn, p, 0).rgb; vec3 out_ = vec3(0.);
      for (int d = 0; d < 2; d++) { vec2 dir = d == 0 ? vec2(.70710678, .70710678) : vec2(.70710678, -.70710678);
        vec3 acc = x; for (int i = 1; i <= 4; i++) for (int s = -1; s <= 1; s += 2) {
          vec2 q = vec2(p) + .5 + dir * (uLen * .5 * float(i) / 4. * float(s)); acc += texture(uIn, q / sz).rgb; }
        out_ += acc / 9.; }
      o = vec4(x + (out_ * .5 - x) * uMix, 1.); }`,
    /* one blend-mode layer: 0 normal 1 screen 2 soft light 3 darken 4 linear burn 5 overlay */
    blendTex: HEAD + `uniform sampler2D uBase; uniform sampler2D uTop; uniform sampler2D uMask; uniform int uMode, uMaskOn, uGray;
    uniform float uOp, uFrame;
    vec3 soft(vec3 b, vec3 s){ vec3 d = mix(sqrt(clamp(b, 0., 1.)), ((16. * b - 12.) * b + 4.) * b, step(b, vec3(.25)));
      return mix(b + (2. * s - 1.) * (d - b), b - (1. - 2. * s) * b * (1. - b), step(s, vec3(.5))); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 b = texelFetch(uBase, p, 0).rgb;
      vec3 s = texelFetch(uTop, p, 0).rgb; if (uGray == 1) s = vec3(s.r);
      vec3 r = s;
      if (uMode == 1) r = 1. - (1. - b) * (1. - s);
      else if (uMode == 2) r = soft(b, s);
      else if (uMode == 3) r = min(b, s);
      else if (uMode == 4) r = clamp(b + s - 1., 0., 1.);
      else if (uMode == 5) r = mix(1. - 2. * (1. - b) * (1. - s), 2. * b * s, step(b, vec3(.5)));
      float m = 1.;
      if (uMaskOn == 1) m = 1. - uFrame * (1. - texture(uMask, (vec2(p) + .5) / vec2(uDim)).r);
      o = vec4(b + (r - b) * uOp * m, 1.); }`,
    /* Levels, Curves, Hue/Saturation, Vibrance, the fitted Color Lookup */
    wcAdjust: HEAD + `uniform sampler2D uIn; uniform sampler2D uCurve; uniform sampler2D uLook; uniform float uAmt;
    uniform vec4 uLv; uniform float uLvGamma, uHS, uVib, uVSat;
    float tab(sampler2D t, float v, int c){ float x = clamp(v, 0., 1.) * 255.; int i0 = int(floor(x)); int i1 = min(i0 + 1, 255);
      return mix(texelFetch(t, ivec2(i0, 0), 0)[c], texelFetch(t, ivec2(i1, 0), 0)[c], x - float(i0)); }
    void main(){ vec3 x0 = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb;
      vec3 v = pow(clamp((x0 - uLv.x / 255.) / ((uLv.y - uLv.x) / 255.), 0., 1.), vec3(1. / uLvGamma));
      vec3 x = clamp(uLv.z / 255. + v * (uLv.w - uLv.z) / 255., 0., 1.);
      x = vec3(tab(uCurve, x.r, 0), tab(uCurve, x.g, 0), tab(uCurve, x.b, 0));
      float l = lum(x); x = clamp(l + (x - l) * (1. + uHS), 0., 1.);
      float mx = max(x.r, max(x.g, x.b)), mn = min(x.r, min(x.g, x.b)); l = lum(x);
      x = l + (x - l) * (1. + uVib * (1. - (mx - mn)) * (1. - (mx - mn)));
      l = lum(x); x = clamp(l + (x - l) * (1. + uVSat * .6), 0., 1.);
      x = vec3(tab(uLook, x.r, 0), tab(uLook, x.g, 1), tab(uLook, x.b, 2));
      o = vec4(x0 + (x - x0) * uAmt, 1.); }`,
    /* --- Halftone --- */
    /* Black & White (Photoshop default mix) + the Levels stretch -> grey in rgb */
    htBW: HEAD + `uniform sampler2D uIn; uniform float uLo, uHi;
    void main(){ vec3 c = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb; float r = c.r, g = c.g, b = c.b;
      float mx = max(max(r, g), b), mn = min(min(r, g), b), md = r + g + b - mx - mn;
      float prim = r >= max(g, b) ? .40 : (g >= b ? .40 : .20);
      float sec = b <= min(r, g) ? .60 : (r <= g ? .60 : .80);
      float v = mn + (md - mn) * sec + (mx - md) * prim;
      v = clamp((v - uLo / 255.) / max((uHi - uLo) / 255., .05), 0., 1.);
      o = vec4(vec3(v), 1.); }`,
    /* posterize 9 + grain specks + black edge specks */
    htStipple: HEAD + `uniform sampler2D uG, uB25, uB07, uB20, uB06, uN; uniform float uGrain, uEdges, uThr;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); float g = texelFetch(uG, p, 0).r, n = texelFetch(uN, p, 0).r;
      float step_ = 1. / 8.; float q = min(floor(g / step_ + .5) * step_, 1.);
      float det = (texelFetch(uB25, p, 0).r - texelFetch(uB07, p, 0).r) / .03;
      float amp = (1.35 - 2. * max(g - .5, 0.)) * (uGrain / 50.);
      float o_ = (n * amp + det) > 1. ? max(q - 2. * step_, 0.) : q;
      if (uEdges > 0. && (texelFetch(uB20, p, 0).r - texelFetch(uB06, p, 0).r + n * .012) > uThr) o_ = 0.;
      o = vec4(vec3(o_), 1.); }`,
    /* cloth (Linear Burn), veins (Color Dodge), the film ramp */
    htFinish: HEAD + `uniform sampler2D uX, uCloth, uVeins, uRamp; uniform float uGrid; uniform int uClothOn;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uX, p, 0).rgb;
      if (uClothOn == 1) { float t = 1. - (1. - texelFetch(uCloth, p, 0).r) * uGrid; x = clamp(x + t - 1., 0., 1.); }
      vec3 v = texelFetch(uVeins, p, 0).rgb;
      x = mix(min(vec3(1.), x / max(1. - v, vec3(1e-4))), vec3(1.), step(vec3(1.), v));
      float l = x.r * .3 + x.g * .59 + x.b * .11;
      int i = clamp(int(l * 255. + .5), 0, 255);
      o = vec4(texelFetch(uRamp, ivec2(i, 0), 0).rgb, 1.); }`,
    /* --- Impression --- */
    sq2: HEAD + `uniform sampler2D uIn; void main(){ float x = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).r; o = vec4(x, x * x, 0., 1.); }`,
    trace: HEAD + `uniform sampler2D uT; void main(){ vec3 t = texelFetch(uT, ivec2(gl_FragCoord.xy), 0).rgb; o = vec4(vec3(t.r + t.g), 1.); }`,
    /* stroke direction (doubled angle) mixed with the swirl where the photo is flat */
    imFlowCS: HEAD + `uniform sampler2D uT; uniform sampler2D uSw; uniform float uP90, uSwirl;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 t = texelFetch(uT, p, 0).rgb;
      float jxx = t.r, jyy = t.g, jxy = t.b; float ang2 = atan(2. * jxy, jxx - jyy) + 3.14159265;
      float tr = jxx + jyy; float coh = sqrt((jxx - jyy) * (jxx - jyy) + 4. * jxy * jxy) / (tr + 1e-6);
      float st = clamp(coh * clamp(tr / (uP90 + 1e-6), 0., 1.) * 1.6, 0., 1.);
      float sw = texelFetch(uSw, p, 0).r * 3.14159265 * (.6 + .8 * uSwirl) + .6;
      o = vec4(st * cos(ang2) + (1. - st) * cos(2. * sw), st * sin(ang2) + (1. - st) * sin(2. * sw), 0., 1.); }`,
    imDir: HEAD + `uniform sampler2D uCS; void main(){ vec2 c = texelFetch(uCS, ivec2(gl_FragCoord.xy), 0).rg; float a = .5 * atan(c.g, c.r);
      o = vec4(cos(a), sin(a), 0., 1.); }`,
    /* LIC along the flow, weights fading with distance (impression.lic) */
    licW: HEAD + `uniform sampler2D uIn; uniform sampler2D uD; uniform float uLen; uniform int uSteps;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim); float st = uLen / float(uSteps);
      vec3 acc = texelFetch(uIn, p, 0).rgb; float ws = 1.;
      for (int s = 0; s < 2; s++) { float sg = s == 0 ? 1. : -1.;
        vec2 pos = vec2(p); vec2 pd = texelFetch(uD, p, 0).xy * sg;
        for (int i = 0; i < uSteps; i++) {
          vec2 v = texture(uD, (pos + .5) / sz).xy; if (dot(v, pd) < 0.) v = -v;
          pos += v * st; pd = v; float wt = 1. - float(i + 1) / float(uSteps + 1) * .5;
          acc += texture(uIn, (pos + .5) / sz).rgb * wt; ws += wt; } }
      o = vec4(acc / ws, 1.); }`,
    pack2: HEAD + `uniform sampler2D uA; uniform sampler2D uB; uniform vec2 uInv;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); o = vec4(texelFetch(uA, p, 0).r * uInv.x, texelFetch(uB, p, 0).r * uInv.y, 0., 1.); }`,
    /* the 7 dab-shuffled copies and their median */
    median7: HEAD + `uniform sampler2D uPre, uN0, uN1, uN2, uN3, uN4, uN5, uN6; uniform float uAmp;
    vec3 tap(sampler2D n, ivec2 p, vec2 sz){ vec2 d = texelFetch(n, p, 0).rg * uAmp; return texture(uPre, (vec2(p) + d + .5) / sz).rgb; }
    void sw_(inout float a, inout float b){ float t = min(a, b); b = max(a, b); a = t; }
    float med(float a, float b, float c, float d, float e, float f, float g){
      sw_(a,b); sw_(c,d); sw_(e,f); sw_(a,c); sw_(b,d); sw_(e,g); sw_(a,e); sw_(b,f); sw_(c,g);
      sw_(b,c); sw_(d,e); sw_(f,g); sw_(b,e); sw_(d,g); sw_(c,e); sw_(d,f); sw_(c,d); sw_(e,f); sw_(d,e);
      return d; }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim);
      vec3 a = tap(uN0, p, sz), b = tap(uN1, p, sz), c = tap(uN2, p, sz), d = tap(uN3, p, sz),
           e = tap(uN4, p, sz), f = tap(uN5, p, sz), g = tap(uN6, p, sz);
      o = vec4(med(a.r,b.r,c.r,d.r,e.r,f.r,g.r), med(a.g,b.g,c.g,d.g,e.g,f.g,g.g), med(a.b,b.b,c.b,d.b,e.b,f.b,g.b), 1.); }`,
    /* bicubic enlarge like cv2.INTER_CUBIC (a = -0.75, edges repeated) */
    upCubic: HEAD + `uniform sampler2D uIn; uniform ivec2 uSrc;
    float cw(float x){ x = abs(x); float a = -.75; return x <= 1. ? ((a + 2.) * x - (a + 3.)) * x * x + 1. : (x < 2. ? ((a * x - 5. * a) * x + 8. * a) * x - 4. * a : 0.); }
    void main(){ vec2 q = (gl_FragCoord.xy) * vec2(uSrc) / vec2(uDim) - .5; vec2 b = floor(q); vec2 f = q - b;
      vec4 acc = vec4(0.);
      for (int j = -1; j <= 2; j++) { float wy = cw(f.y - float(j));
        for (int i = -1; i <= 2; i++) { float wx = cw(f.x - float(i));
          ivec2 t = clamp(ivec2(b) + ivec2(i, j), ivec2(0), uSrc - 1); acc += texelFetch(uIn, t, 0) * wx * wy; } }
      o = vec4(acc.rgb, 1.); }`,
    lumClip: HEAD + `uniform sampler2D uIn; void main(){ o = vec4(vec3(lum(clamp(texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb, 0., 1.))), 1.); }`,
    imTone: HEAD + `uniform sampler2D uIn; uniform float uTX[10], uTY[10];
    void main(){ vec3 x = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb; float L = lum(clamp(x, 0., 1.));
      float Lt = uTY[9]; for (int i = 0; i < 9; i++) if (L <= uTX[i + 1]) { Lt = mix(uTY[i], uTY[i + 1], (L - uTX[i]) / (uTX[i + 1] - uTX[i])); break; }
      if (L <= uTX[0]) Lt = uTY[0];
      o = vec4(clamp(x * ((Lt + .02) / (L + .02)), 0., 1.), 1.); }`,
    imStripe: HEAD + `uniform sampler2D uIn; uniform sampler2D uMod; uniform float uP, uAmt;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float st = sin(6.2831853 * float(p.x) / uP) * texelFetch(uMod, p, 0).r; o = vec4(x * (1. + st * uAmt), 1.); }`,
    sobelMag: HEAD + `uniform sampler2D uIn; float L(ivec2 p){ return at(uIn, p).r; }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      float gx = (L(p+ivec2(1,-1)) + 2.*L(p+ivec2(1,0)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(-1,0)) + L(p+ivec2(-1,1)));
      float gy = (L(p+ivec2(-1,1)) + 2.*L(p+ivec2(0,1)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(0,-1)) + L(p+ivec2(1,-1)));
      o = vec4(vec3(sqrt(gx * gx + gy * gy)), 1.); }`,
    /* bristle ripples: emboss(rid) * detail * (0.4 + 0.8 L) */
    imRid: HEAD + `uniform sampler2D uIn, uRid, uDet, uL; uniform float uInvStd, uP80, uAmt; uniform vec2 uOff;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim); vec3 x = texelFetch(uIn, p, 0).rgb;
      float r = texelFetch(uRid, p, 0).r * uInvStd, rs = texture(uRid, (vec2(p) - uOff + .5) / sz).r * uInvStd;
      float det = clamp(texelFetch(uDet, p, 0).r / (uP80 + 1e-6), 0., 1.);
      float L = texelFetch(uL, p, 0).r;
      o = vec4(x * (1. + (r - rs) * det * (.4 + .8 * L) * uAmt), 1.); }`,
    usNC: HEAD + `uniform sampler2D uIn; uniform sampler2D uB; uniform float uAmt; uniform int uClamp;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb, b = texelFetch(uB, p, 0).rgb;
      vec3 y = x + (x - b) * uAmt; if (uClamp == 1) y = clamp(y, 0., 1.); o = vec4(y, 1.); }`,
    imLight: HEAD + `uniform sampler2D uIn, uLong, uM; uniform float uInvStd, uAmt;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 e = texelFetch(uIn, p, 0).rgb;
      vec3 lg = clamp(texelFetch(uLong, p, 0).rgb * .93, 0., 1.);
      float cover = clamp((texelFetch(uM, p, 0).r * uInvStd + .85) * 2.2, 0., 1.) * uAmt;
      o = vec4(e + (max(e, lg) - e) * cover, 1.); }`,
    imSurface: HEAD + `uniform sampler2D uIn, uSrc, uBump, uCanvas; uniform float uRelief, uCanvasAmt; uniform int uBumpOn, uCanvasOn;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      if (uBumpOn == 1) { float b = .5 + (texelFetch(uBump, p, 0).r - .5) * uRelief;
        b = .5 + (b - .5) * (.4 + .6 * lum(texelFetch(uSrc, p, 0).rgb)); x = ovl(x, vec3(b)); }
      if (uCanvasOn == 1) { float c = .5 + (texelFetch(uCanvas, p, 0).r - .5) * uCanvasAmt; x = ovl(x, vec3(c)); }
      o = vec4(clamp(x, 0., 1.), 1.); }`,
    /* white border: the picture shrunk into the frame */
    border: HEAD + `uniform sampler2D uIn; uniform float uB;
    void main(){ vec2 sz = vec2(uDim); vec2 q = gl_FragCoord.xy * (sz + 2. * uB) / sz - uB;
      if (q.x < 0. || q.y < 0. || q.x > sz.x || q.y > sz.y) { o = vec4(1.); return; }
      o = vec4(texture(uIn, q / sz).rgb, 1.); }`,
    /* --- Painted Photo --- */
    /* 4-channel shrink / enlarge (alpha kept) for the wide masked blurs */
    down4: HEAD + `uniform sampler2D uIn; uniform vec2 uF; uniform ivec2 uSrc;
    void main(){ vec2 p = floor(gl_FragCoord.xy); vec4 acc = vec4(0.); float n = 0.;
      int fx = int(ceil(uF.x)), fy = int(ceil(uF.y));
      for (int y = 0; y < fy; y++) for (int x = 0; x < fx; x++) {
        vec2 q = p * uF + (vec2(x, y) + .5) * uF / vec2(float(fx), float(fy)); acc += texture(uIn, q / vec2(uSrc)); n += 1.; }
      o = acc / n; }`,
    up4: HEAD + `uniform sampler2D uIn; void main(){ o = texture(uIn, gl_FragCoord.xy / vec2(uDim)); }`,
    ppLevel: HEAD + `uniform sampler2D uIn; uniform float uLo, uHi, uN;
    void main(){ float L = lum(texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb);
      o = vec4(vec3(clamp(floor((L - uLo) / (uHi - uLo + 1e-6) * uN), 0., uN - 1.)), 1.); }`,
    /* median of the levels in a ks x ks window (cv2.medianBlur, edges repeated) */
    medianLv: HEAD + `uniform sampler2D uIn; uniform int uK, uN;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); int r = uK / 2, half_ = (uK * uK + 1) / 2;
      float m = float(uN - 1);
      for (int k = 0; k < 16; k++) {                 /* the first level with half the window at or below it */
        if (k >= uN - 1) break;
        int c = 0;
        for (int y = -r; y <= r; y++) for (int x = -r; x <= r; x++)
          c += texelFetch(uIn, clamp(p + ivec2(x, y), ivec2(0), uDim - 1), 0).r < float(k) + .5 ? 1 : 0;
        if (c >= half_) { m = float(k); break; }
      }
      o = vec4(vec3(m), 1.); }`,
    /* the colour family (nearest k-means centre in Lab a*b*), one-hot */
    labFam: HEAD + `uniform sampler2D uIn; uniform float uCen[12]; uniform int uPart;
    float lin(float c){ return c > .04045 ? pow((c + .055) / 1.055, 2.4) : c / 12.92; }
    float f3(float t){ return t > .008856 ? pow(t, 1. / 3.) : 7.787 * t + 16. / 116.; }
    void main(){ vec3 c = clamp(texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb, 0., 1.);
      vec3 l = vec3(lin(c.r), lin(c.g), lin(c.b));
      float X = dot(l, vec3(.412453, .357580, .180423)) / .950456, Y = dot(l, vec3(.212671, .715160, .072169)),
            Z = dot(l, vec3(.019334, .119193, .950227)) / 1.088754;
      vec2 ab = vec2(500. * (f3(X) - f3(Y)), 200. * (f3(Y) - f3(Z)));
      int best = 0; float bd = 1e20;
      for (int i = 0; i < 6; i++) { vec2 d = ab - vec2(uCen[i * 2], uCen[i * 2 + 1]); float dd = dot(d, d); if (dd < bd) { bd = dd; best = i; } }
      int j = best - uPart * 4; o = vec4(equal(ivec4(j), ivec4(0, 1, 2, 3))); }`,
    ppKey: HEAD + `uniform sampler2D uLv, uA, uB;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec4 a = texelFetch(uA, p, 0), b = texelFetch(uB, p, 0);
      int f = 0; float m = a.r;
      if (a.g > m) { m = a.g; f = 1; } if (a.b > m) { m = a.b; f = 2; } if (a.a > m) { m = a.a; f = 3; }
      if (b.r > m) { m = b.r; f = 4; } if (b.g > m) { m = b.g; f = 5; }
      o = vec4(vec3(texelFetch(uLv, p, 0).r * 6. + float(f)), 1.); }`,
    keyMask: HEAD + `uniform sampler2D uIn, uKey; uniform float uV;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); float m = abs(texelFetch(uKey, p, 0).r - uV) < .5 ? 1. : 0.;
      o = vec4(texelFetch(uIn, p, 0).rgb * m, m); }`,
    keyAcc: HEAD + `uniform sampler2D uAcc, uB, uKey; uniform float uV;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 a = texelFetch(uAcc, p, 0).rgb;
      if (abs(texelFetch(uKey, p, 0).r - uV) < .5) { vec4 b = texelFetch(uB, p, 0); a = b.rgb / max(b.a, 1e-6); }
      o = vec4(a, 1.); }`,
    upNearest: HEAD + `uniform sampler2D uIn; uniform ivec2 uSrc;
    void main(){ ivec2 q = min(ivec2(floor(floor(gl_FragCoord.xy) * vec2(uSrc) / vec2(uDim))), uSrc - 1); o = vec4(texelFetch(uIn, q, 0).rgb, 1.); }`,
    /* ragged patch edges (remap through noise), the paint filters' darkening */
    ppRemap: HEAD + `uniform sampler2D uIn, uDx, uDy; uniform float uAmt;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim);
      vec2 q = vec2(p) + vec2(texelFetch(uDx, p, 0).r, texelFetch(uDy, p, 0).r) * uAmt;
      o = vec4(pow(clamp(texture(uIn, (q + .5) / sz).rgb, 0., 1.), vec3(1.1)), 1.); }`,
    ppEdge: HEAD + `uniform sampler2D uMag, uL, uGL; uniform float uThr, uDen, uDark;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      float e = clamp((texelFetch(uMag, p, 0).r - uThr) / uDen, 0., 1.) * ((texelFetch(uGL, p, 0).r - texelFetch(uL, p, 0).r) > uDark ? 1. : 0.);
      o = vec4(vec3(e), 1.); }`,
    /* cv2.dilate with a w x w square (anchor in the middle) */
    maxFilt: HEAD + `uniform sampler2D uIn; uniform int uW;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); float m = 0.; int a = uW / 2;
      for (int y = 0; y < uW; y++) for (int x = 0; x < uW; x++) m = max(m, texelFetch(uIn, clamp(p + ivec2(x - a, y - a), ivec2(0), uDim - 1), 0).r);
      o = vec4(vec3(m), 1.); }`,
    ppOutline: HEAD + `uniform sampler2D uIn, uE, uF;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      o = vec4(texelFetch(uIn, p, 0).rgb * (1. - texelFetch(uE, p, 0).r * .85) * (1. - texelFetch(uF, p, 0).r * .7), 1.); }`,
    ppTexture: HEAD + `uniform sampler2D uIn, uCells, uShade; uniform float uT;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float n = texelFetch(uCells, p, 0).r * 2.5; float t = (exp(n) - exp(-n)) / (exp(n) + exp(-n));
      x = clamp(x * (1. + t * .02 * uT), 0., 1.);
      o = vec4(clamp(x * (1. + texelFetch(uShade, p, 0).r * (2. * uT) / 50. * 2.), 0., 1.), 1.); }`,
    ppPaper: HEAD + `uniform sampler2D uIn, uKraft, uOld; uniform float uAmt;
    vec3 soft(vec3 b, vec3 s){ vec3 d = mix(sqrt(clamp(b, 0., 1.)), ((16. * b - 12.) * b + 4.) * b, step(b, vec3(.25)));
      return mix(b + (2. * s - 1.) * (d - b), b - (1. - 2. * s) * b * (1. - b), step(s, vec3(.5))); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      x = x + (soft(x, texelFetch(uKraft, p, 0).rgb) - x) * uAmt;
      x = x + (x * texelFetch(uOld, p, 0).rgb - x) * .5 * uAmt;
      o = vec4(clamp(x, 0., 1.), 1.); }`,
    /* the warm film grade on the smooth picture, the grain put back as light/dark */
    ppGrade: HEAD + `uniform sampler2D uIn, uSm; uniform float uG[39]; uniform float uAmt;
    vec3 gr(vec3 c){ float r = c.r, g = c.g, b = c.b;
      vec3 y = vec3(0.);
      #define T(i, f) y += (f) * vec3(uG[i * 3], uG[i * 3 + 1], uG[i * 3 + 2]);
      T(0, r) T(1, g) T(2, b) T(3, r*r) T(4, g*g) T(5, b*b) T(6, r*g) T(7, r*b) T(8, g*b) T(9, r*r*r) T(10, g*g*g) T(11, b*b*b) T(12, 1.)
      #undef T
      return mix(c, clamp(y, 0., 1.), uAmt); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb, sm = texelFetch(uSm, p, 0).rgb;
      vec3 det = vec3((x.r - sm.r + x.g - sm.g + x.b - sm.b) / 3.);
      vec3 g0 = gr(sm); float e = .03;
      vec3 sl = clamp((gr(clamp(sm + e, 0., 1.)) - gr(clamp(sm - e, 0., 1.))) / (2. * e), .15, 3.);
      float ls = (sl.r + sl.g + sl.b) / 3.; sl = ls + .3 * (sl - ls);
      o = vec4(clamp(g0 + det * sl, 0., 1.), 1.); }`,
    /* --- Oil Smear --- */
    /* the Oil Paint brush relief, then Posterize 10 on the brightness */
    smPoster: HEAD + `uniform sampler2D uIn, uRel; uniform float uAmt;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      vec3 x = clamp(texelFetch(uIn, p, 0).rgb * (1. + texelFetch(uRel, p, 0).r * .06 * uAmt), 0., 1.);
      float L = lum(x);
      o = vec4(clamp(x * (floor(L * 9. + .5) / 9. + .02) / (L + .02), 0., 1.), 1.); }`,
    /* tone, the texture laid on (Overlay + Multiply), vibrance, S curve, levels */
    smFinish: HEAD + `uniform sampler2D uIn, uTex, uTone, uAux; uniform float uOv, uMul, uTone_;
    float L1(sampler2D t, float v, int c){ float k = clamp(v, 0., 1.) * 255.; int i0 = int(floor(k)); int i1 = min(i0 + 1, 255);
      return mix(texelFetch(t, ivec2(i0, 0), 0)[c], texelFetch(t, ivec2(i1, 0), 0)[c], k - float(i0)); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 c = texelFetch(uIn, p, 0).rgb;
      vec3 x = vec3(L1(uTone, c.r, 0), L1(uTone, c.g, 1), L1(uTone, c.b, 2));
      float t = texelFetch(uTex, p, 0).r;
      if (uOv > 0.) x = x + (ovl(x, vec3(t)) - x) * uOv;
      if (uMul > 0.) x = x * (1. - uMul * (1. - L1(uAux, t, 0)));
      if (uTone_ > 0.) {
        x = clamp(x, 0., 1.);
        float l = lum(x), sat = max(x.r, max(x.g, x.b)) - min(x.r, min(x.g, x.b));
        vec3 v = l + (x - l) * (1. - .12 * (1. - sat) * (1. - sat));
        l = lum(v); v = clamp(l + (v - l) * (1. + .06 * .6), 0., 1.);
        x = x + (v - x) * min(1., uTone_);
        vec3 xc = clamp(x, 0., 1.);
        vec3 cu = vec3(L1(uAux, xc.r, 1), L1(uAux, xc.g, 1), L1(uAux, xc.b, 1));
        x = x + (cu - x) * min(1., 217. / 255. * uTone_);
        vec3 lv = pow(clamp((clamp(x, 0., 1.) - 12. / 255.) / (243. / 255.), 0., 1.), vec3(1. / 1.51));
        x = x + (lv - x) * min(1., 107. / 255. * uTone_);
      }
      o = vec4(clamp(x, 0., 1.), 1.); }`,
    /* --- Canvas Oil (see canvas_oil.py) --- */
    coTone: CO + `uniform sampler2D uIn; uniform float uLo, uHi, uC, uV;
    void main(){ vec3 f = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb;
      if (uC > 0.) { f = clamp((f - uLo) / max(.2, uHi - uLo), 0., 1.); f = f + (f * f * (3. - 2. * f) - f) * (.55 * uC); }
      if (abs(uV) > 1e-3) {
        float mx = max(f.r, max(f.g, f.b)), mn = min(f.r, min(f.g, f.b)), sat = (mx - mn) / max(mx, 1e-6);
        float ns = uV > 0. ? sat + (1. - sat) * sat * uV * 1.6 + sat * uV * .25 : sat * (1. + uV);
        ns = clamp(ns, 0., 1.);
        if (sat > 1e-6) f = mx - (mx - f) * (ns / max(sat, 1e-6)); }
      o = vec4(clamp(f, 0., 1.), 1.); }`,
    coScaleR: CO + `uniform sampler2D uIn; uniform float uP;
    void main(){ o = vec4(vec3(clamp(texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).r / (uP + 1e-6), 0., 1.)), 1.); }`,
    /* knife-cut patches: the best of the 3 x 3 nearest cells of a jittered grid */
    coCells: CO + `uniform sampler2D uSrc; uniform float uSize, uSq, uK, uLam; uniform int uWarp;
    void main(){ vec2 p = gl_FragCoord.xy, sz = vec2(uDim); float cw = uSize, ch = uSize * uSq;
      vec2 q = p;
      if (uWarp == 1) { float amp = max(1., uSize * .18); vec2 n = p / (1.2 * uSize);
        q = p + vec2(vn(n.x, n.y, uK) - .5, vn(n.x, n.y, uK + 1.) - .5) * amp; }
      vec3 pix = texelFetch(uSrc, ivec2(p), 0).rgb;
      float gx = floor(q.x / cw), gy = floor(q.y / ch), best = 1e30; vec3 oc = vec3(0.);
      for (int oy = -1; oy <= 1; oy++) for (int ox = -1; ox <= 1; ox++) {
        float cx = gx + float(ox), cy = gy + float(oy);
        vec2 sp = vec2((cx + .05 + .9 * rnd(cx, cy, uK + 2.)) * cw, (cy + .05 + .9 * rnd(cx, cy, uK + 3.)) * ch);
        vec3 col = texture(uSrc, clamp(sp, vec2(.5), sz - .5) / sz).rgb;
        vec2 d = (q - sp) / vec2(cw, ch); vec3 dc = col - pix;
        float cost = dot(d, d) + uLam * dot(dc, dc);
        if (cost < best) { best = cost; oc = col; } }
      o = vec4(oc, 1.); }`,
    coFine: CO + `uniform sampler2D uA, uB, uDm; uniform float uBig, uD, uK;
    void main(){ vec2 p = gl_FragCoord.xy; ivec2 ip = ivec2(p);
      float jit = (vn(p.x / (1.5 * uBig), p.y / (1.5 * uBig), uK) - .5) * .5;
      float m = sst(.62 - .5 * uD, .78 - .5 * uD, texelFetch(uDm, ip, 0).r + jit);
      vec3 a = texelFetch(uA, ip, 0).rgb; o = vec4(a + (texelFetch(uB, ip, 0).rgb - a) * m, 1.); }`,
    /* knife dabs (cells without colour pull) and the wet-paint runs */
    coDabs: CO + `uniform sampler2D uIn, uDm; uniform float uSize, uSq, uK, uAmt, uDrag, uS, uKs;
    void main(){ vec2 p = gl_FragCoord.xy, sz = vec2(uDim); ivec2 ip = ivec2(p); vec3 x = texelFetch(uIn, ip, 0).rgb;
      if (uAmt > .01) {
        float cw = uSize, ch = uSize * uSq, gx = floor(p.x / cw), gy = floor(p.y / ch), best = 1e30;
        vec3 col = vec3(0.); float bx = 0., by = 0.;
        for (int oy = -1; oy <= 1; oy++) for (int ox = -1; ox <= 1; ox++) {
          float cx = gx + float(ox), cy = gy + float(oy);
          vec2 sp = vec2((cx + .05 + .9 * rnd(cx, cy, uK + 2.)) * cw, (cy + .05 + .9 * rnd(cx, cy, uK + 3.)) * ch);
          vec2 d = (p - sp) / vec2(cw, ch); float cost = dot(d, d);
          if (cost < best) { best = cost; col = texture(uIn, clamp(sp, vec2(.5), sz - .5) / sz).rgb; bx = cx; by = cy; } }
        float shade = (rnd(bx, by, uK + 1.) + rnd(bx, by, uK + 2.) + rnd(bx, by, uK + 3.) - 1.5) * .11;
        vec3 tint = (vec3(rnd(bx, by, uK + 4.), rnd(bx, by, uK + 5.), rnd(bx, by, uK + 6.)) - .5) * .0624;
        vec3 cell = col * (1. + shade + tint), shaded = x * (1. + shade * .8);
        float drift = sst(.25, .75, (vn(p.x / (uSize * 14.), p.y / (uSize * 14.), uK + 10.) * .6 + vn(p.x / (uSize * 5.), p.y / (uSize * 5.), uK + 11.) * .4) / 1.);
        float f = (1. - texelFetch(uDm, ip, 0).r * .85) * uAmt * (.35 + .65 * drift);
        shaded = x + (shaded - x) * f;
        x = shaded + (cell - shaded) * clamp(f * .9, 0., .6); }
      if (uDrag > .01) {
        float st = clamp((vn(p.x / (3. * uS), p.y / (40. * uS), uKs) - .5) * 3., -1., 1.);
        x = x * (1. + st * .1 * uDrag * (1. - lum(x))); }
      o = vec4(x, 1.); }`,
    coDir: CO + `uniform sampler2D uT; uniform float uK, uS, uLoose;
    void main(){ vec2 p = gl_FragCoord.xy; vec3 t = texelFetch(uT, ivec2(p), 0).rgb;
      float a = .5 * atan(2. * t.b, t.r - t.g) + 1.5707963 + (vn(p.x / (30. * uS), p.y / (30. * uS), uK) - .5) * uLoose;
      o = vec4(cos(a), sin(a), 0., 1.); }`,
    /* like licTrace, without clamping (paint can be a little over 1) */
    coLic: HEAD + `uniform sampler2D uIn; uniform sampler2D uD; uniform float uLen; uniform int uSteps;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim); float st = uLen / float(uSteps);
      vec3 acc = texelFetch(uIn, p, 0).rgb;
      for (int s = 0; s < 2; s++) { float sg = s == 0 ? 1. : -1.;
        vec2 pos = vec2(p); vec2 pd = texelFetch(uD, p, 0).xy * sg;
        for (int i = 0; i < uSteps; i++) {
          vec2 v = texture(uD, (pos + .5) / sz).xy; if (dot(v, pd) < 0.) v = -v;
          pos += v * st; pd = v; acc += texture(uIn, (pos + .5) / sz).rgb; } }
      o = vec4(acc / float(1 + 2 * uSteps), 1.); }`,
    coNoise: CO + `uniform float uSw, uK;
    void main(){ vec2 p = gl_FragCoord.xy;
      o = vec4(vn(p.x / uSw, p.y / uSw, uK + 1.), rnd(floor(p.x), floor(p.y), uK), 0., 1.); }`,
    coHgt: CO + `uniform sampler2D uA, uB;
    void main(){ ivec2 ip = ivec2(gl_FragCoord.xy);
      float sf = clamp((texelFetch(uA, ip, 0).r - .5) * 6., -1., 1.), br = clamp((texelFetch(uB, ip, 0).g - .5) * 7., -1.5, 1.5);
      o = vec4(sf * .6 + br * .4, sf, br, 1.); }`,
    coSweep: CO + `uniform sampler2D uA, uB, uDm;
    void main(){ ivec2 ip = ivec2(gl_FragCoord.xy); vec3 a = texelFetch(uA, ip, 0).rgb;
      o = vec4(a + (texelFetch(uB, ip, 0).rgb - a) * (1. - texelFetch(uDm, ip, 0).r), 1.); }`,
    coBrush: CO + `uniform sampler2D uIn, uQ, uQb, uH; uniform float uLoose, uAmt;
    void main(){ ivec2 ip = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, ip, 0).rgb, q = texelFetch(uQ, ip, 0).rgb;
      vec3 h = texelFetch(uH, ip, 0).rgb;
      float emb = at(uH, ip + ivec2(-1, -1)).r - at(uH, ip + ivec2(1, 1)).r;
      float lit = h.g * .10 * (.5 + uLoose) + h.b * .05 + emb * .35 * (.48 + .2 * uLoose);
      vec3 st = q + (q - texelFetch(uQb, ip, 0).rgb) * .5;
      st = st * (1. + lit);
      o = vec4(x + (st - x) * min(1., uAmt * 1.1), 1.); }`,
    coRimA: CO + `uniform sampler2D uIn; float L(ivec2 p){ return at(uIn, p).r; }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      float gx = (L(p+ivec2(1,-1)) + 2.*L(p+ivec2(1,0)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(-1,0)) + L(p+ivec2(-1,1)));
      float gy = (L(p+ivec2(-1,1)) + 2.*L(p+ivec2(0,1)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(0,-1)) + L(p+ivec2(1,-1)));
      float e = -(gx + gy) * .7071; o = vec4(e, abs(e), sqrt(gx * gx + gy * gy), 1.); }`,
    coMagN: CO + `uniform sampler2D uIn; uniform float uP;
    void main(){ o = vec4(vec3(clamp(texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).b / (uP + 1e-6), 0., 1.)), 1.); }`,
    coRims: CO + `uniform sampler2D uIn, uE, uSeam, uG, uGb; uniform float uE98, uM99, uRel;
    void main(){ ivec2 ip = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, ip, 0).rgb; vec3 e = texelFetch(uE, ip, 0).rgb;
      float emb = clamp(e.r / (uE98 + 1e-6), -1.6, 1.6);
      x = x * (1. + emb * .34 * uRel);
      float mag = e.b / (uM99 + 1e-6);
      float darker = texelFetch(uGb, ip, 0).r > texelFetch(uG, ip, 0).r ? 1. : 0.;
      x = x * (1. - texelFetch(uSeam, ip, 0).r * darker * .35 * uRel);
      float line = sst(.6, 1., mag) * (1. - darker) * .35 * uRel;
      o = vec4(x + (1. - x) * line, 1.); }`,
    coSurface: CO + `uniform sampler2D uIn; uniform float uK, uS, uDrag, uCanvas;
    void main(){ vec2 p = gl_FragCoord.xy; vec3 x = texelFetch(uIn, ivec2(p), 0).rgb; float s = uS;
      float mott = ((vn(p.x / (40. * s), p.y / (40. * s), uK) * .45 + vn(p.x / (12. * s), p.y / (12. * s), uK + 1.) * .35
                   + vn(p.x / (4. * s), p.y / (4. * s), uK + 2.) * .2) / 1. - .5) * 1.6;
      float grain = (vn(p.x / (.9 * s + .5), p.y / ((2. + 5. * uDrag) * 1.5 * s + .5), uK + 4.) - .5) * 4.;
      x = x * (1. + mott * .14 + grain * .035);
      if (uCanvas > .01) {
        float Lp = lum(x);
        float blot = ((vn(p.x / (90. * s), p.y / (90. * s), uK + 10.) * .5 + vn(p.x / (30. * s), p.y / (30. * s), uK + 11.) * .3
                     + vn(p.x / (10. * s), p.y / (10. * s), uK + 12.) * .2) / 1. - .5) * 1.6 + .5;
        float th = .88 - .2 * uCanvas + (blot - .5) * .28;
        float m = sst(th, th + .1, Lp);
        m = clamp(m * (.75 + .5 * vn(p.x / (4. * s), p.y / (4. * s), uK + 14.)), 0., 1.);
        m = m > .55 ? min(1., m * 1.15) : m * .6;
        x = x + (CW - x) * (m * min(.92, .3 + .8 * uCanvas)); }
      o = vec4(x, 1.); }`,
    coDiff: CO + `uniform sampler2D uA, uB;
    void main(){ ivec2 ip = ivec2(gl_FragCoord.xy); o = vec4(vec3(texelFetch(uA, ip, 0).r - texelFetch(uB, ip, 0).r), 1.); }`,
    coEMG: CO + `uniform sampler2D uIn, uMag; float L(ivec2 p){ return at(uIn, p).r; }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy);
      float gx = (L(p+ivec2(1,-1)) + 2.*L(p+ivec2(1,0)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(-1,0)) + L(p+ivec2(-1,1)));
      float gy = (L(p+ivec2(-1,1)) + 2.*L(p+ivec2(0,1)) + L(p+ivec2(1,1))) - (L(p+ivec2(-1,-1)) + 2.*L(p+ivec2(0,-1)) + L(p+ivec2(1,-1)));
      o = vec4(L(p), sqrt(gx * gx + gy * gy) / 8., texelFetch(uMag, p, 0).r, 1.); }`,
    coSketch: CO + `uniform sampler2D uIn, uEMG; uniform float uThr, uA, uS, uK;
    void main(){ vec2 p = gl_FragCoord.xy, sz = vec2(uDim); vec3 x = texelFetch(uIn, ivec2(p), 0).rgb; float s = uS;
      float lw = .9 * max(1., .8 * s), ink = 0., cell = 25. * s;
      for (int i = 0; i < 2; i++) {
        float kk = uK + 100. + 10. * float(i); vec2 n = p / (18. * s);
        vec2 w = p + vec2(vn(n.x, n.y, kk) - .5, vn(n.x, n.y, kk + 1.) - .5) * 3. * s + vec2((float(i) - .5) * 1.2 * s, 0.);
        vec3 q = texture(uEMG, clamp(w, vec2(.5), sz - .5) / sz).rgb;
        float dist = abs(q.r) / (q.g + 1e-5);
        float li = (1. - sst(.35 * lw, lw + .35, dist)) * sst(.6 * uThr, uThr, q.b);
        li *= vn(p.x / cell, p.y / cell, kk + 2.) > .32 ? 1. : 0.;
        ink = max(ink, li * (.55 + .45 * rnd(floor(p.x / cell), floor(p.y / cell), kk + 3.))); }
      float hc = 40. * s, hx = floor(p.x / hc), hy = floor(p.y / hc);
      float ha = (30. + 30. * rnd(hx, hy, uK + 120.)) * .017453292519943295, gap = (3. + 2.5 * rnd(hx, hy, uK + 121.)) * s;
      float tt = (p.x * cos(ha) + p.y * sin(ha)) / gap, lwh = .65 * max(1., s);
      float hl = 1. - sst(.3 * lwh, lwh, abs(tt - floor(tt) - .5) * gap);
      float dk = 1. - lum(x);
      float hm = sst(.55, .8, vn(p.x / (22. * s), p.y / (22. * s), uK + 122.)) * sst(.35, .75, dk * dk * 1.6);
      ink = max(ink, hl * hm * (.3 + .4 * rnd(hx, hy, uK + 123.)) * min(1., uA * 1.6));
      float dry = .45 + .55 * sst(.15, .5, vn(p.x / (2. * s), p.y / (2. * s), uK + 124.));
      ink = clamp(ink, 0., 1.) * dry * (.55 + .4 * uA);
      o = vec4(x + (x * .22 + INK * .78 - x) * ink, 1.); }`,
    coScratch: CO + `uniform sampler2D uIn, uDm; uniform float uA, uS, uK;
    void main(){ vec2 p = gl_FragCoord.xy, sz = vec2(uDim); vec3 x = texelFetch(uIn, ivec2(p), 0).rgb; float s = uS;
      float light = 0., dark = 0., lw = max(1., s / 1.4), R = 110. * s, g0x = floor(p.x / R), g0y = floor(p.y / R);
      for (int oy = -1; oy <= 1; oy++) for (int ox = -1; ox <= 1; ox++) {
        float cx = g0x + float(ox), cy = g0y + float(oy);
        vec2 cc = (vec2(cx, cy) + vec2(rnd(cx, cy, uK + 130.), rnd(cx, cy, uK + 131.))) * R;
        vec2 cs = clamp(cc, vec2(.5), sz - .5) / sz;
        float pref = (.25 + .75 * (1. - lum(texture(uIn, cs).rgb))) * (.35 + .65 * texture(uDm, cs).r);
        if (!(rnd(cx, cy, uK + 132.) < min(1., uA * 2.2) * pref)) continue;
        float rad = (18. + 37. * rnd(cx, cy, uK + 133.)) * s; vec2 r = p - cc;
        if (!(dot(r, r) < rad * rad)) continue;
        float cid = cx * 7. + cy;
        for (int j = 0; j < 2; j++) {
          float fj = float(j), a0 = j == 0 ? 30. : 120.;
          float ang = (a0 + 30. * rnd(cx, cy, uK + 134. + fj)) * .017453292519943295, gp = (6. + 7. * rnd(cx, cy, uK + 136. + fj)) * s;
          float ca = cos(ang), sa = sin(ang), off = (-r.x * sa + r.y * ca) / gp, li = floor(off);
          float dl = abs(off - li - .5) * gp, along = r.x * ca + r.y * sa;
          if (!(rnd(li, cid, uK + 138. + fj) > .35)) continue;
          float oc = (li + .5) * gp, half_ = sqrt(max(0., rad * rad - oc * oc)) * (.4 + .6 * rnd(li, cid, uK + 140. + fj));
          if (!(abs(along) < half_ && half_ >= 4. * s)) continue;
          float cov = clamp(lw - dl + .5, 0., 1.) * (.35 + .65 * rnd(li, cid, uK + 142. + fj));
          if (rnd(li, cid, uK + 144. + fj) < .8) light = max(light, cov); else dark = max(dark, cov); } }
      float Rs = 40. * s, h0x = floor(p.x / Rs), h0y = floor(p.y / Rs);
      for (int oy = -1; oy <= 1; oy++) for (int ox = -1; ox <= 1; ox++) {
        float cx = h0x + float(ox), cy = h0y + float(oy);
        if (!(rnd(cx, cy, uK + 150.) < min(1., uA * .5))) continue;
        vec2 sp = (vec2(cx, cy) + vec2(rnd(cx, cy, uK + 151.), rnd(cx, cy, uK + 152.))) * Rs;
        float deg = (rnd(cx, cy, uK + 153.) < .5 ? 45. : 135.) + (rnd(cx, cy, uK + 154.) - .5) * 16.;
        float ang = deg * .017453292519943295, ln = (8. + 18. * rnd(cx, cy, uK + 155.)) * s;
        vec2 u = vec2(cos(ang), sin(ang)), r = p - sp; float tq = clamp(dot(r, u), 0., ln);
        float cov = clamp(lw - length(r - u * tq) + .5, 0., 1.);
        light = max(light, cov * (.3 + .5 * rnd(cx, cy, uK + 156.))); }
      light = clamp(light, 0., 1.) * (.25 + .4 * uA); dark = clamp(dark, 0., 1.) * .35;
      x = x + (x * .4 + PALE * .6 - x) * light;
      o = vec4(x * (1. - dark * .5), 1.); }`,
    coDots: CO + `uniform sampler2D uIn, uT; uniform float uA, uS, uK;
    void main(){ vec2 p = gl_FragCoord.xy, sz = vec2(uDim); float s = uS; vec3 dots = vec3(0.); float alpha = 0.;
      for (int f = 0; f < 2; f++) {
        float k0 = uK + (f == 0 ? 170. : 160.), C = sqrt(1e6 / ((f == 0 ? 90. : 700.) * uA)) * s;
        float g0x = floor(p.x / C), g0y = floor(p.y / C);
        for (int oy = -1; oy <= 1; oy++) for (int ox = -1; ox <= 1; ox++) {
          float cx = g0x + float(ox), cy = g0y + float(oy);
          vec2 sp = (vec2(cx, cy) + vec2(rnd(cx, cy, k0), rnd(cx, cy, k0 + 1.))) * C;
          vec2 ss = clamp(sp, vec2(.5), sz - .5) / sz;
          float r, av; vec3 col;
          if (f == 1) {
            r = max(1., floor((1. + 1.6 * rnd(cx, cy, k0 + 2.)) * s + .5));
            vec3 base = texture(uIn, ss).rgb; col = clamp(base * 1.6 + 60. / 255., 0., 1.);
            av = (.15 + .3 * rnd(cx, cy, k0 + 3.)) * (.4 + .6 * (1. - lum(base)));
          } else {
            r = max(1., floor((1.2 + 2. * rnd(cx, cy, k0 + 2.)) * s + .5));
            vec3 base = texture(uT, ss).rgb; float mx = max(base.r, max(base.g, base.b));
            vec3 viv = clamp(mx - (mx - base) * 1.6, 0., 1.);
            if (!(rnd(cx, cy, k0 + 4.) < .7)) viv = CW;
            col = clamp(viv * 1.1 + 25. / 255., 0., 1.); av = .6 + .35 * rnd(cx, cy, k0 + 3.);
          }
          float cov = clamp(r + .5 - length(p - sp), 0., 1.);
          if (cov > 0.) { dots = col; alpha = max(alpha, cov * av); } } }
      o = vec4(dots * alpha, alpha); }`,
    coDotMix: CO + `uniform sampler2D uIn, uD1, uD2;
    void main(){ ivec2 ip = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, ip, 0).rgb; vec4 d = texelFetch(uD1, ip, 0);
      vec3 c = d.rgb / max(d.a, 1e-4); x = x * (1. - d.a) + c * d.a;
      o = vec4(x + texelFetch(uD2, ip, 0).rgb * .45, 1.); }`,
    coWeaveTex: CO + `uniform float uK, uS;
    void main(){ vec2 p = gl_FragCoord.xy; float s = uS, per = max(2.5, 4.2 * s);
      float jx = (vn(p.x / (60. * s), p.y / (60. * s), uK) - .5) * per * 1.2, jy = (vn(p.x / (60. * s), p.y / (60. * s), uK + 1.) - .5) * per * 1.2;
      float wx = sin((p.x + jx) * (6.283185307179586 / per)), wy = sin((p.y + jy) * (6.283185307179586 / (per * 1.12)));
      float weave = (wx * wy > 0. ? wx : wy) * .6, gs = .9 * max(1., s);
      float grain = (vn(p.x / gs, p.y / gs, uK + 2.) - .5) * 4.;
      float fibre = ((vn(p.x / (25. * s), p.y / (25. * s), uK + 3.) * .4 + vn(p.x / (8. * s), p.y / (8. * s), uK + 4.) * .35
                    + vn(p.x / (3. * s), p.y / (3. * s), uK + 5.) * .25) / 1. - .5) * 3.2;
      o = vec4(vec3(weave * .55 + grain * .25 + fibre * .35), 1.); }`,
    coWeave: CO + `uniform sampler2D uIn, uW; uniform float uCanvas;
    float W(ivec2 p){ return at(uW, p).r; }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float e = -W(p + ivec2(-1, -1)) - W(p + ivec2(0, -1)) - W(p + ivec2(-1, 0)) + W(p + ivec2(1, 0)) + W(p + ivec2(0, 1)) + W(p + ivec2(1, 1));
      float t = clamp((W(p) + e * .18) * 1.333, -1., 1.);
      o = vec4(x * (1. + t * (.04 + .12 * lum(x)) * (.4 + .9 * uCanvas)), 1.); }`,
    /* --- Silk Oil (see silkoil.py) --- */
    slMix: HEAD + `uniform sampler2D uA, uB; uniform float uAmt;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 a = texelFetch(uA, p, 0).rgb;
      o = vec4(a + (texelFetch(uB, p, 0).rgb - a) * uAmt, 1.); }`,
    /* Glass (displacement by the texture's slope) and Texturizer relief */
    slGlass: HEAD + `uniform sampler2D uIn, uH; uniform float uAmp, uTz, uLx, uLy;
    float H(ivec2 p){ return at(uH, p).r; }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 sz = vec2(uDim);
      float gx = (H(p+ivec2(1,-1)) + 2.*H(p+ivec2(1,0)) + H(p+ivec2(1,1))) - (H(p+ivec2(-1,-1)) + 2.*H(p+ivec2(-1,0)) + H(p+ivec2(-1,1)));
      float gy = (H(p+ivec2(-1,1)) + 2.*H(p+ivec2(0,1)) + H(p+ivec2(1,1))) - (H(p+ivec2(-1,-1)) + 2.*H(p+ivec2(0,-1)) + H(p+ivec2(1,-1)));
      vec3 x = uAmp != 0. ? texture(uIn, (gl_FragCoord.xy + vec2(gx, gy) * uAmp) / sz).rgb : texelFetch(uIn, p, 0).rgb;
      float emb = -(gx * uLx + gy * uLy) / 8.;
      o = vec4(x * (1. + emb * uTz), 1.); }`,
    slPenIn: CO + `uniform sampler2D uIn; uniform float uK;
    void main(){ vec2 p = gl_FragCoord.xy;
      o = vec4(vec3(lum(texelFetch(uIn, ivec2(p), 0).rgb) + (rnd(floor(p.x), floor(p.y), uK) - .5) * .5), 1.); }`,
    slConstDir: HEAD + `void main(){ o = vec4(.70710678, .70710678, 0., 1.); }`,
    slPen: HEAD + `uniform sampler2D uIn, uP; uniform float uT, uMix;
    vec3 soft(vec3 b, vec3 s){ vec3 d = mix(sqrt(clamp(b, 0., 1.)), ((16. * b - 12.) * b + 4.) * b, step(b, vec3(.25)));
      return mix(b + (2. * s - 1.) * (d - b), b - (1. - 2. * s) * b * (1. - b), step(s, vec3(.5))); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float pen = clamp((texelFetch(uP, p, 0).r - uT) / .08 + .5, 0., 1.);
      o = vec4(x + (soft(x, vec3(pen)) - x) * uMix, 1.); }`,
    slNoise: CO + `uniform float uK;
    void main(){ vec2 p = gl_FragCoord.xy; o = vec4(vec3(rnd(floor(p.x), floor(p.y), uK)), 1.); }`,
    /* Oil Paint, Darken: the smear with its bristle streaks, the darker one wins */
    slOil: HEAD + `uniform sampler2D uIn, uO, uG; uniform float uGroove;
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      float z = clamp((texelFetch(uG, p, 0).r - .5) * 14., -2.5, 2.5);
      o = vec4(min(x, texelFetch(uO, p, 0).rgb * (1. + z * uGroove)), 1.); }`,
    slColor: HEAD + `uniform sampler2D uIn, uLut; uniform float uSat, uAmt;
    float L1(float v, int c){ float k = clamp(v, 0., 1.) * 255.; int i0 = int(floor(k)); int i1 = min(i0 + 1, 255);
      return mix(texelFetch(uLut, ivec2(i0, 0), 0)[c], texelFetch(uLut, ivec2(i1, 0), 0)[c], k - float(i0)); }
    void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 x = texelFetch(uIn, p, 0).rgb;
      if (uAmt > .001) { float l = lum(x); vec3 y = clamp(l + (x - l) * uSat, 0., 1.);
        y = vec3(L1(y.r, 0), L1(y.g, 1), L1(y.b, 2)); x = x + (y - x) * uAmt; }
      o = vec4(clamp(x, 0., 1.), 1.); }`,
    /* to RGBA8, top row first (images arrive top row first) */
    out: HEAD + `uniform sampler2D uIn; void main(){ o = vec4(texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).rgb, 1.); }`
  };

  const pause = () => new Promise(r => {
    let done = false; const go = () => { if (!done) { done = true; r(); } };
    requestAnimationFrame(go); setTimeout(go, 30);
  });

  class OilFX {
    constructor() {
      this.ok = false;
      try {
        const cv = document.createElement('canvas'); cv.width = cv.height = 1;
        const gl = cv.getContext('webgl2', { antialias: false, premultipliedAlpha: false,
          alpha: false, powerPreference: 'high-performance' });
        if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;
        this.gl = gl; this.canvas = cv;
        cv.addEventListener('webglcontextlost', e => {
          e.preventDefault(); this.ok = false; this.lost = true;
          if (window.logGpu) window.logGpu('context lost (styles)', 'the graphics card dropped the filter context',
            JSON.stringify({ last: window.OilFX && window.OilFX.lastTimes, memo: Object.keys(this.memo || {}) }));
        });
        cv.addEventListener('webglcontextrestored', () => { try { this._setup(); this.lost = false; } catch (err) { console.warn(err); } });
        this._setup();
      } catch (err) { console.warn('Oil Paint FX on the GPU unavailable:', err); }
    }
    _setup() {
      const gl = this.gl;
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      this.pool = new Map(); this.memo = {}; this.memoStyle = null;
      /* shaders are compiled when first used: one a graphics card cannot
         compile only stops the style that needs it */
      this.prog = {}; this.bad = {};
      this._get('out');
      this.vao = gl.createVertexArray();
      this.ok = true;
    }
    _program(vs, fs) {
      const gl = this.gl, p = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s);
      }
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, a.name); }
      return { p, u };
    }
    _tex(w, h, ifmt, fmt, type, data, filter) {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter || gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter || gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
      return t;
    }
    _img(bitmap, linear) {
      const gl = this.gl, t = gl.createTexture(), f = linear ? gl.LINEAR : gl.NEAREST;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, f], [gl.TEXTURE_MAG_FILTER, f],
        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      return t;
    }
    _target(w, h) {
      const gl = this.gl;
      const tex = this._tex(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, null, gl.LINEAR);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fbo };
    }
    /* one full-screen pass into target t; heavy passes go in bands */
    _get(name) {
      let pr = this.prog[name];
      if (pr) return pr;
      if (this.bad[name]) throw new Error(this.bad[name]);
      this.compiles = (this.compiles || 0) + 1;          /* (a render that built a step is not timed) */
      try { pr = this.prog[name] = this._program(VS, FS[name]); }
      catch (err) {
        this.bad[name] = `The graphics card could not build the "${name}" step: ${String(err.message || err).slice(0, 400)}`;
        if (window.logGpu) window.logGpu('shader', this.bad[name]);
        throw new Error(this.bad[name]);
      }
      return pr;
    }
    async _pass(name, t, uni, heavy) {
      if (this.stopFn && this.stopFn()) throw new Error('Stopped');
      const gl = this.gl, pr = this._get(name);
      gl.useProgram(pr.p);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.viewport(0, 0, this.w, this.h);
      let unit = 0;
      const all = Object.assign({ uDim: [this.w, this.h, 'i'] }, uni);
      for (const [k, v] of Object.entries(all)) {
        const loc = pr.u[k]; if (loc == null) continue;
        if (v instanceof WebGLTexture) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, v); gl.uniform1i(loc, unit++); }
        else if (Array.isArray(v) && v[v.length - 1] === 'i') gl['uniform' + (v.length - 1) + 'i'](loc, ...v.slice(0, -1));
        else if (Array.isArray(v) && v.length > 4) gl.uniform1fv(loc, new Float32Array(v));
        else if (Array.isArray(v)) gl['uniform' + v.length + 'f'](loc, ...v);
        else if (typeof v === 'object' && 'i' in v) gl.uniform1i(loc, v.i);
        else gl.uniform1f(loc, v);
      }
      gl.bindVertexArray(this.vao);
      if (!heavy) { gl.drawArrays(gl.TRIANGLES, 0, 3); return; }
      gl.enable(gl.SCISSOR_TEST);
      for (let y = 0; y < this.h; y += BAND) {
        gl.scissor(0, y, this.w, Math.min(BAND, this.h - y));
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.flush();
        await pause();
        if (gl.isContextLost()) throw new Error('The graphics card stopped responding');
      }
      gl.disable(gl.SCISSOR_TEST);
    }
    /* run fn with every pass at sw x sh (targets made for that size) */
    async _at(sw, sh, fn) {
      const W = this.w, H = this.h; this.w = sw; this.h = sh;
      try { return await fn(); } finally { this.w = W; this.h = H; }
    }
    /* a small copy (<= 256 px long, averaged like cv2 INTER_AREA) read back:
       [Float32Array rgba, w, h] */
    async _readSmall(tex, w, h, maxSide = 256) {
      const gl = this.gl, k = Math.min(1, maxSide / Math.max(w, h));
      const sw = Math.max(1, Math.round(w * k)), sh = Math.max(1, Math.round(h * k));
      const t = this._target(sw, sh);
      try {
        await this._at(sw, sh, () => this._pass('down', t, { uIn: tex, uF: [w / sw, h / sh], uSrc: [w, h, 'i'] }));
        const buf = new Float32Array(sw * sh * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
        gl.readPixels(0, 0, sw, sh, gl.RGBA, gl.FLOAT, buf);
        return [buf, sw, sh];
      } finally { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
    }
    /* percentile q of the red channel of a small copy (numpy's method) */
    async _pct(tex, w, h, q) {
      const [buf, sw, sh] = await this._readSmall(tex, w, h);
      const v = new Float32Array(sw * sh); for (let i = 0; i < v.length; i++) v[i] = buf[i * 4];
      v.sort(); const pos = (v.length - 1) * q / 100, i0 = Math.floor(pos), i1 = Math.min(v.length - 1, i0 + 1);
      return v[i0] + (v[i1] - v[i0]) * (pos - i0);
    }
    /* standard deviation of the red channel (mean of x and x*x, averaged down) */
    async _std(tex, w, h, tmp) {
      await this._at(w, h, () => this._pass('sq2', tmp, { uIn: tex }));
      const [buf, sw, sh] = await this._readSmall(tmp.tex, w, h, 64);
      let m = 0, m2 = 0; const n = sw * sh;
      for (let i = 0; i < n; i++) { m += buf[i * 4]; m2 += buf[i * 4 + 1]; }
      m /= n; m2 /= n; return Math.sqrt(Math.max(m2 - m * m, 0));
    }
    /* self-guided filter (brushfx / filters.guided), twice: src -> dst.
       t: 6 targets at the current size */
    async _guided2(src, dst, r, eps, t) {
      const [a, b, c, d, e, tmp] = t;
      const box = async (srcT, out) => {
        await this._pass('box', tmp, { uIn: srcT, uDir: [1, 0, 'i'], uR: { i: r } });
        await this._pass('box', out, { uIn: tmp.tex, uDir: [0, 1, 'i'], uR: { i: r } });
      };
      let cur = src;
      for (let it = 0; it < 2; it++) {
        await box(cur, b);
        await this._pass('square', c, { uIn: cur });
        await box(c.tex, d);
        await this._pass('guidedA', c, { uM: b.tex, uMM: d.tex, uEps: eps });
        await this._pass('guidedB', e, { uM: b.tex, uA: c.tex });
        await box(c.tex, d);
        await box(e.tex, b);
        await this._pass('guidedOut', it === 0 ? a : dst, { uIn: cur, uA: d.tex, uB: b.tex });
        cur = a.tex;
      }
    }
    /* painted.cutout: flat patches - brightness levels split by colour
       family, each pixel the mean colour of the nearby pixels of its own
       level and family. ana: {lo, hi, cen} from the server. -> dst (W x H) */
    async _cutout(ctx, srcTex, dst, s, levels, detail, ana, cutSigma, wideStep) {
      const W = this.w, H = this.h;
      let cw = W, ch = H, cs = s; while (cs > 0.6) { cw = Math.floor(cw / 2); ch = Math.floor(ch / 2); cs /= 2; }
      const Tc = () => ctx.sized(cw, ch);
      const cBase = Tc(), cA = Tc(), cB = Tc(), cC = Tc(), cD = Tc(), cE = Tc(), cTmp = Tc(), cAcc1 = Tc(), cAcc2 = Tc(), cKey = Tc(), cLv = Tc();
      let acc;
      await this._at(cw, ch, async () => {
        let src = srcTex;
        if (cw !== W || ch !== H) { await this._pass('down', cE, { uIn: srcTex, uF: [W / cw, H / ch], uSrc: [W, H, 'i'] }); src = cE.tex; }
        const r = Math.max(1, Math.round(Math.max(2, (10 - 6 * detail / 100) * cs)));
        await this._guided2(src, cBase, r, 0.003, [cA, cB, cC, cD, cAcc1, cTmp]);
        await this._pass('ppLevel', cB, { uIn: cBase.tex, uLo: ana.lo, uHi: ana.hi, uN: levels });
        const ks = Math.max(3, Math.min(((Math.floor((9 - 4 * detail / 100) * cs)) | 1), 255));
        await this._pass('medianLv', cLv, { uIn: cB.tex, uK: { i: ks }, uN: { i: levels } }, true);
        const cen = [].concat(...ana.cen.map(c => [c[0], c[1]]));
        await this._pass('labFam', cC, { uIn: cBase.tex, uCen: cen, uPart: { i: 0 } });
        await this._pass('labFam', cD, { uIn: cBase.tex, uCen: cen, uPart: { i: 1 } });
        await this._blur(cC.tex, cC, cTmp, 2.5 * cs);
        await this._blur(cD.tex, cD, cTmp, 2.5 * cs);
        await this._pass('ppKey', cKey, { uLv: cLv.tex, uA: cC.tex, uB: cD.tex });
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, cAcc1.fbo); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
        let nxt = cAcc2; acc = cAcc1;
        /* (small quick previews blur on a shrunk copy too - a Run is always
           big enough for this to be the same as the CPU's wide_blur) */
        const sig = cutSigma * cs, f = sig >= 6 ? Math.max(2, Math.floor(sig / wideStep)) : Math.floor(sig / wideStep);
        const sw = Math.max(1, Math.floor(cw / Math.max(f, 1))), sh = Math.max(1, Math.floor(ch / Math.max(f, 1)));
        const S1 = f >= 2 ? ctx.sized(sw, sh) : null, S2 = f >= 2 ? ctx.sized(sw, sh) : null, S3 = f >= 2 ? ctx.sized(sw, sh) : null;
        for (let v = 0; v < levels * 6; v++) {
          await this._pass('keyMask', cB, { uIn: cBase.tex, uKey: cKey.tex, uV: v });
          if (f >= 2) {
            await this._at(sw, sh, async () => {
              await this._pass('down4', S1, { uIn: cB.tex, uF: [cw / sw, ch / sh], uSrc: [cw, ch, 'i'] });
              await this._blur(S1.tex, S2, S3, sig / f);
            });
            await this._pass('up4', cE, { uIn: S2.tex });
          } else await this._blur(cB.tex, cE, cTmp, sig);
          await this._pass('keyAcc', nxt, { uAcc: acc.tex, uB: cE.tex, uKey: cKey.tex, uV: v });
          [acc, nxt] = [nxt, acc];
        }
      });
      if (cw !== W || ch !== H) await this._pass('upNearest', dst, { uIn: acc.tex, uSrc: [cw, ch, 'i'] });
      else await this._pass('out', dst, { uIn: acc.tex });
    }
    /* percentile q of channel ch of a small copy (numpy's method) */
    async _pctCh(tex, w, h, q, ch) {
      const [buf, sw, sh] = await this._readSmall(tex, w, h);
      const v = new Float32Array(sw * sh); for (let i = 0; i < v.length; i++) v[i] = buf[i * 4 + ch];
      v.sort(); const pos = (v.length - 1) * q / 100, i0 = Math.floor(pos), i1 = Math.min(v.length - 1, i0 + 1);
      return v[i0] + (v[i1] - v[i0]) * (pos - i0);
    }
    /* percentiles over the r, g and b values together */
    async _pctRGB(tex, w, h, qs) {
      const [buf, sw, sh] = await this._readSmall(tex, w, h);
      const n = sw * sh, v = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) { v[i * 3] = buf[i * 4]; v[i * 3 + 1] = buf[i * 4 + 1]; v[i * 3 + 2] = buf[i * 4 + 2]; }
      v.sort();
      return qs.map(q => { const pos = (v.length - 1) * q / 100, i0 = Math.floor(pos), i1 = Math.min(v.length - 1, i0 + 1);
        return v[i0] + (v[i1] - v[i0]) * (pos - i0); });
    }
    /* a wide Gaussian on a copy shrunk by a whole factor (canvas_oil.wide) */
    async _wide(ctx, srcTex, dst, sigma) {
      const W = this.w, H = this.h, f = Math.floor(sigma / 4);
      if (f < 2) { await this._blur(srcTex, dst, ctx.T(), sigma); return; }
      const sw = Math.max(1, Math.floor(W / f)), sh = Math.max(1, Math.floor(H / f));
      const S1 = ctx.sized(sw, sh), S2 = ctx.sized(sw, sh), S3 = ctx.sized(sw, sh);
      await this._at(sw, sh, async () => {
        await this._pass('down', S1, { uIn: srcTex, uF: [W / sw, H / sh], uSrc: [W, H, 'i'] });
        await this._blur(S1.tex, S2, S3, sigma / f);
      });
      await this._pass('up', dst, { uIn: S2.tex });
    }
    async _blur(src, dst, tmp, sigma) {
      const heavy = sigma * this.w > 4000;
      await this._pass('gauss', tmp, { uIn: src, uDir: [1, 0, 'i'], uS: sigma }, heavy);
      await this._pass('gauss', dst, { uIn: tmp.tex, uDir: [0, 1, 'i'], uS: sigma }, heavy);
    }

    /* style: 'Oil Paint FX' | 'Sketch Paint'.  src: ImageBitmap (the
       photo, RGB).  assets: that style's template pictures (ImageBitmaps at
       the same size).  look: its numbers (tone curve...).  p: the settings.
       Returns Uint8ClampedArray RGBA (top row first). */
    /* one picture at a time on the card: a new render waits for the last
       (a stopped one ends at its next step) */
    async render(...args) {
      while (this.busy) { try { await this.busy; } catch (e) { /* the last one failed: fine */ } }
      const job = this._render(...args);
      this.busy = job;
      try { return await job; } finally { if (this.busy === job) this.busy = null; }
    }
    async _render(style, src, assets, look, p, onProgress, textures, analysis, shouldStop, memoOn) {
      if (!this.ok) throw new Error('WebGL 2 is not available');
      this.stopFn = shouldStop || null;
      this.memoHits = 0;
      const gl = this.gl, w = src.width, h = src.height;
      const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      if (w > max || h > max) throw new Error('The picture is too large for this graphics card.');
      this.w = w; this.h = h;
      const made = [], imgs = [];
      /* render targets are kept between renders and handed out again (a
         slider move does not allocate a hundred megabytes on the card) */
      this.pool = this.pool || new Map();
      const take = (tw, th) => {
        const list = this.pool.get(tw + 'x' + th);
        const t = (list && list.length) ? list.pop() : this._target(tw, th);
        t.w = tw; t.h = th; made.push(t); return t;
      };
      /* stages kept from the last render of this style and picture: when
         only a later setting moved, the earlier stages are not made again
         (live previews only - a Run computes everything once) */
      if (memoOn && w * h > MEMO_MAX_PX) memoOn = false;       /* big pictures: the card's memory is kept free */
      if (this.memoStyle !== style || !memoOn) this._memoFree();
      this.memoStyle = style;
      const srcId = srcIds.get(src) || (srcIds.set(src, ++srcSeq), srcSeq);
      const ctx = {
        k: Math.min(w, h) / REF_SHORT,
        T: () => take(w, h),
        img: (b, linear) => { const t = this._img(b, linear); imgs.push(t); return t; },
        sized: (sw, sh) => take(sw, sh),
        memo: async (name, deps, fn) => {
          const key = JSON.stringify([srcId, w, h, deps]);
          const slot = this.memo && this.memo[name];
          if (memoOn && slot && slot.key === key) { this.memoHits++; return slot.val; }
          if (slot) { this._freeTargets(slot.targets); delete this.memo[name]; }
          const targets = [];
          const keep = memoOn ? () => { const t = this._target(w, h); targets.push(t); return t; } : () => take(w, h);
          const val = await fn(keep);
          if (memoOn) { this.memo = this.memo || {}; this.memo[name] = { key, val, targets }; }
          return val;
        },
        lutTex: (rows) => {                       /* 256 x [r,g,b] -> float texture */
          const d = new Float32Array(256 * 4);
          for (let i = 0; i < 256; i++) d.set([rows[i][0], rows[i][1], rows[i][2], 1], i * 4);
          const t = this._tex(256, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, d); imgs.push(t); return t;
        },
        /* a float16 texture made on the server (random textures): ch = 1 or 3 */
        f16: (tt, linear) => {
          const fmt = tt.ch === 3 ? [gl.RGB16F, gl.RGB] : [gl.R16F, gl.RED];
          const t = this._tex(tt.w, tt.h, fmt[0], fmt[1], gl.HALF_FLOAT, tt.data, linear ? gl.LINEAR : gl.NEAREST);
          imgs.push(t); return t;
        },
        step: (fr) => onProgress && onProgress(fr)
      };
      try {
        const pipe = PIPELINES[style];
        if (!pipe) throw new Error(style + ' has no GPU version');
        const final = await pipe.call(this, ctx, src, assets, look, p, textures || {}, analysis || {});
        ctx.step(0.92);
        const out = this._target8(w, h);
        await this._pass('out', out, { uIn: final.tex });
        const px = new Uint8ClampedArray(w * h * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, out.fbo);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
        gl.deleteTexture(out.tex); gl.deleteFramebuffer(out.fbo);
        if (gl.isContextLost()) throw new Error('The graphics card stopped responding');
        ctx.step(1);
        return px;
      } finally {
        /* back to the pool; sizes not used by this render are let go */
        const used = new Set(made.map(t => t.w + 'x' + t.h));
        for (const [key, list] of this.pool) if (!used.has(key)) { this._freeTargets(list); this.pool.delete(key); }
        let budget = POOL_BYTES;
        for (const t of made) {
          const key = t.w + 'x' + t.h, bytes = t.w * t.h * 8;
          if (budget < bytes) { this._freeTargets([t]); continue; }   /* over the budget: let it go */
          budget -= bytes;
          if (!this.pool.has(key)) this.pool.set(key, []);
          this.pool.get(key).push(t);
        }
        for (const t of imgs) gl.deleteTexture(t);
      }
    }
    _freeTargets(list) {
      const gl = this.gl;
      for (const t of list) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
    }
    _memoFree() {
      if (this.memo) for (const k of Object.keys(this.memo)) this._freeTargets(this.memo[k].targets);
      this.memo = {};
    }
    /* let go of everything kept on the card (a new photo, a lost context) */
    release() {
      this._memoFree();
      if (this.pool) for (const list of this.pool.values()) this._freeTargets(list);
      this.pool = new Map();
    }
    /* shared start of both styles: Surface Blur, then Oil Paint smoothing */
    async _surfaceOil(ctx, srcTex, radius, sigmaColor, lenPx, A, B, Bl, Tmp, Tn) {
      const k = ctx.k;
      let cur = { tex: srcTex };
      if (radius > 0) {
        const R = Math.max(1, Math.round(radius));
        await this._pass('bilateral', A, { uIn: srcTex, uR: { i: R }, uStep: { i: Math.max(1, Math.floor(R / 6)) },
          uSs: R / 2, uSc: sigmaColor }, true);
      } else await this._pass('out', A, { uIn: srcTex });
      cur = A;
      ctx.step(0.25);
      await this._blur(cur.tex, Bl, Tmp, 1.5 * k);
      await this._pass('luma', Tmp, { uIn: Bl.tex });
      await this._pass('tensor', Tn, { uIn: Tmp.tex });
      await this._blur(Tn.tex, Tn, Tmp, C.FLOW_SIGMA * k);
      if (lenPx >= 0.3) {
        await this._pass('lic', B, { uIn: cur.tex, uT: Tn.tex, uLen: lenPx, uSteps: { i: C.LIC_STEPS } }, true);
        cur = B;
      }
      ctx.step(0.45);
      return cur;
    }
    _target8(w, h) {
      const gl = this.gl, tex = this._tex(w, h, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fbo };
    }
  }

  /* --- variations: a seed nudges the main settings (oilpaint/variation.py) */
  function jitter(seed, i) {
    if (!seed) return 0;
    const v = Math.sin(seed * 12.9898 + i * 78.233) * 43758.5453;
    return (v - Math.floor(v)) * 2 - 1;
  }
  function vary(st, table, seed) {
    seed = Number(seed) || 0;
    if (!seed) return st;
    const out = Object.assign({}, st);
    for (const [name, slot, amt] of table) out[name] = Math.max(0, out[name] * (1 + amt * jitter(seed, slot)));
    return out;
  }
  /* the same tables as VARY in oilfx.py, sketchfx.py, brushfx.py, watercolor.py */
  const VARY = {
    'Oil Paint FX': [['smooth', 0, 0.35], ['brush', 1, 0.45], ['detail', 2, 0.30], ['color', 3, 0.20], ['texture', 4, 0.35], ['tone', 5, 0.50]],
    'Sketch Paint': [['smooth', 0, 0.35], ['brush', 1, 0.45], ['blotch', 2, 0.40], ['poster', 3, 0.40], ['edges', 4, 0.35], ['outline', 5, 0.30]],
    'Rough Brush': [['paint', 0, 0.40], ['smooth', 1, 0.40], ['poster', 2, 0.40], ['warp', 3, 0.40], ['blur', 4, 0.40], ['texture', 5, 0.30]],
    'Silk Oil': [['smooth', 0, 0.35], ['smear', 1, 0.40], ['streaks', 2, 0.45], ['texture', 3, 0.40], ['light', 4, 0.40]],
    'Watercolor': [['wash', 0, 0.40], ['blotch', 1, 0.45], ['detail', 2, 0.40], ['color', 3, 0.25], ['paper', 4, 0.35]]
  };
  const SEED_KEY = { 'Silk Oil': 'sl_seed', 'Canvas Oil': 'co_seed', 'Halftone': 'ht_seed', 'Impression': 'im_seed', 'Painted Photo': 'pp_seed', 'Oil Smear': 'sm_seed', 'Oil Paint FX': 'fx_seed', 'Sketch Paint': 'sk_seed', 'Rough Brush': 'rb_seed', 'Watercolor': 'wc_seed' };

  /* --- the styles, step by step (see oilfx.py / sketchfx.py ...) ------- */
  const PIPELINES = {
    'Oil Paint FX': async function (ctx, src, assets, look, p) {
      const k = ctx.k, f = (key) => Number(p[key] ?? 100) / 100;
      const st = vary({ smooth: f('fx_smooth'), brush: f('fx_brush'), detail: f('fx_detail'), color: f('fx_color'),
        texture: f('fx_texture'), tone: f('fx_tone'), edges: f('fx_edges'), adjust: f('fx_adjust') }, VARY['Oil Paint FX'], p.fx_seed);
      const tSrc = ctx.img(src), tTone = ctx.img(assets.tone), tTexA = ctx.img(assets.texture), tMask = ctx.img(assets.mask);
      const tLut = ctx.lutTex(look.lut), tCurve = ctx.lutTex(look.curve.map(v => [v, v, v]));
      const A = ctx.T(), B = ctx.T(), Bl = ctx.T(), Tmp = ctx.T(), Tn = ctx.T();
      const radius = st.smooth > 0.01 ? C.SB_RADIUS * k * st.smooth : 0;
      const cur = await this._surfaceOil(ctx, tSrc, radius, C.SB_SIGMA_COLOR, C.LIC_LEN * k * st.brush, A, B, Bl, Tmp, Tn);
      let a = cur, b = cur === A ? B : A;
      for (const r of [2, 3]) {
        await this._blur(a.tex, Bl, Tmp, r * k);
        await this._pass('highpass', b, { uIn: a.tex, uB: Bl.tex });
        [a, b] = [b, a];
      }
      for (const [amt, sig] of [[C.LOCAL_AMT, C.LOCAL_SIGMA], [C.SHARP1_AMT, C.SHARP1_SIGMA], [C.SHARP2_AMT, C.SHARP2_SIGMA]]) {
        await this._blur(a.tex, Bl, Tmp, sig * k);
        await this._pass('sharpen', b, { uIn: a.tex, uB: Bl.tex, uAmt: amt * st.detail });
        [a, b] = [b, a];
      }
      ctx.step(0.75);
      await this._pass('colour', b, { uIn: a.tex, uLut: tLut, uVib: C.VIB * st.color, uAmt: Math.min(1, st.color) });
      [a, b] = [b, a];
      await this._pass('composite', b, { uIn: a.tex, uTone: tTone, uTex: tTexA, uMask: tMask, uCurve: tCurve,
        uToneAmt: st.tone, uTexAmt: st.texture, uEdges: st.edges, uAdjust: st.adjust, uCb: C.CB_AMOUNT, uBg: C.BG });
      return b;
    },

    'Sketch Paint': async function (ctx, src, assets, look, p) {
      const k = ctx.k, f = (key) => Number(p[key] ?? 100) / 100, S = SK;
      const st = vary({ smooth: f('sk_smooth'), brush: f('sk_brush'), detail: f('sk_detail'), blotch: f('sk_blotch'),
        poster: f('sk_poster'), edges: f('sk_edges'), outline: f('sk_outline'), sketch: f('sk_sketch'), contrast: f('sk_contrast') },
        VARY['Sketch Paint'], p.sk_seed);
      const tSrc = ctx.img(src), tSketch = ctx.img(assets.sketch), tLut = ctx.lutTex(look.lut);
      const A = ctx.T(), B = ctx.T(), Bl = ctx.T(), Tmp = ctx.T(), Tn = ctx.T(), E1 = ctx.T(), E2 = ctx.T(), Ln = ctx.T();
      const radius = st.smooth > 0.01 ? S.SB_RADIUS * k * st.smooth : 0;
      const pre = await this._surfaceOil(ctx, tSrc, radius, S.SB_SIGMA_COLOR, S.LIC_LEN * k * st.brush, A, B, Bl, Tmp, Tn);
      /* the Photo Outline layer first (needs the shared start only) */
      let line = null;
      if (st.outline > 0.001) {
        await this._pass('luma', E1, { uIn: pre.tex });
        await this._blur(E1.tex, Bl, Tmp, S.OUT_SIGMA * k);
        await this._pass('sharpen', E2, { uIn: E1.tex, uB: Bl.tex, uAmt: S.OUT_SHARP });
        await this._blur(E2.tex, Bl, Tmp, S.PC_SIGMA * k);
        await this._pass('photocopy', Ln, { uL: E2.tex, uB: Bl.tex, uGain: S.PC_GAIN, uOff: S.PC_OFFSET });
        line = Ln;
      }
      ctx.step(0.55);
      /* the Photo Effect layer */
      await this._blur(pre.tex, Bl, Tmp, S.SHARP_SIGMA * k);
      await this._pass('sharpen', E1, { uIn: pre.tex, uB: Bl.tex, uAmt: S.SHARP_AMT * st.detail });
      await this._pass('lut', E2, { uIn: E1.tex, uLut: tLut });
      let a = E2, b = E1;
      if (st.blotch > 0.01) {
        const r = Math.max(1, Math.round(S.KW_RADIUS * k * st.blotch));
        for (let i = 0; i < S.KW_PASSES; i++) {
          await this._pass('kuwahara', b, { uIn: a.tex, uR: { i: r } }, true);
          [a, b] = [b, a];
        }
      }
      const mix = S.POSTER_MIX * st.poster;
      if (mix > 0.001) {
        await this._pass('posterL', b, { uIn: a.tex, uN: S.POSTER_LEVELS, uSoft: S.POSTER_SOFT, uMix: mix });
        [a, b] = [b, a];
      }
      if (st.edges > 0.001) {
        await this._blur(a.tex, Bl, Tmp, 0.7 * k);
        await this._pass('luma', A, { uIn: Bl.tex });            /* A, B are free now */
        await this._blur(A.tex, B, Tmp, S.EDGE_SIGMA * k);
        await this._pass('edgeDark', b, { uIn: a.tex, uL1: A.tex, uL2: B.tex, uAmt: S.EDGE_AMT * st.edges });
        [a, b] = [b, a];
      }
      ctx.step(0.8);
      if (!line) { await this._pass('out', Ln, { uIn: tSrc }); }   /* unused when outline is 0 */
      await this._pass('sketchComposite', b, { uIn: a.tex, uSketch: tSketch, uLine: line ? line.tex : Ln.tex,
        uSk: S.SKETCH_OPACITY * st.sketch, uLv: st.contrast, uOut: line ? S.OUTLINE_OPACITY * st.outline : 0,
        uLevels: S.LEVELS });
      return b;
    }
  };
  /* Rough Brush settings - see brushfx.py */
  const RB = {
    GUIDED_R: 3.75, GUIDED_EPS: 0.0012, FLOW_BLUR: 2, FLOW_SIGMA: 4, LIC_LEN: 20, LIC_STEPS: 6,
    POSTER_LEVELS: 10, CUT_LEVELS: 8, CUT_SIGMA: 25, DISP_H: 40, DISP_V: 10, BLUR: 2,
    TEXTURE_OPACITY: 0.6118, VIB: 8, SAT: 2
  };
  PIPELINES['Rough Brush'] = async function (ctx, src, assets, look, p) {
    const f = (key) => Number(p[key] ?? 100) / 100;
    const st = vary({ paint: f('rb_paint'), smooth: f('rb_smooth'), poster: f('rb_poster'), cutout: f('rb_cutout'),
      warp: f('rb_warp'), blur: f('rb_blur'), texture: f('rb_texture'), adjust: f('rb_adjust') }, VARY['Rough Brush'], p.rb_seed);
    const k = Math.max(this.h / look.doc[1], this.w / look.doc[0]);      /* template px -> photo px */
    const tSrc = ctx.img(src), tMap = ctx.img(assets.displace), tTex = ctx.img(assets.texture);
    const tLut = ctx.lutTex(look.lut), tCurve = ctx.lutTex(look.curve.map(v => [v, v, v]));
    const A = ctx.T(), B = ctx.T(), C1 = ctx.T(), C2 = ctx.T(), Tmp = ctx.T(), Tn = ctx.T();
    const boxBlur = async (src_, dst, r) => {
      await this._pass('box', Tmp, { uIn: src_, uDir: [1, 0, 'i'], uR: { i: r } });
      await this._pass('box', dst, { uIn: Tmp.tex, uDir: [0, 1, 'i'], uR: { i: r } });
    };
    /* Oil Paint: guided smoothing */
    let x = { tex: tSrc };
    await this._pass('out', A, { uIn: tSrc }); x = A;
    if (st.smooth > 0.01) {
      const r = Math.max(1, Math.round(RB.GUIDED_R * k));
      await boxBlur(x.tex, C1, r);                                   /* mean I */
      await this._pass('square', B, { uIn: x.tex });
      await boxBlur(B.tex, C2, r);                                   /* mean I*I */
      await this._pass('guidedA', B, { uM: C1.tex, uMM: C2.tex, uEps: RB.GUIDED_EPS * st.smooth });
      await this._pass('guidedB', Tn, { uM: C1.tex, uA: B.tex });
      await boxBlur(B.tex, C2, r);                                   /* mean a */
      await boxBlur(Tn.tex, C1, r);                                  /* mean b */
      await this._pass('guidedOut', B, { uIn: x.tex, uA: C2.tex, uB: C1.tex });
      x = B;
    }
    ctx.step(0.2);
    /* Oil Paint: smoothing along the flow */
    const lenPx = RB.LIC_LEN * k * st.paint;
    if (lenPx >= 0.3) {
      await this._blur(x.tex, C1, Tmp, RB.FLOW_BLUR * k);
      await this._pass('luma', C2, { uIn: C1.tex });
      await this._pass('tensor', Tn, { uIn: C2.tex });
      await this._blur(Tn.tex, Tn, Tmp, RB.FLOW_SIGMA * k + 1);
      await this._pass('dir', C2, { uT: Tn.tex });
      const y = x === A ? B : A;
      await this._pass('licTrace', y, { uIn: x.tex, uD: C2.tex, uLen: lenPx, uSteps: { i: RB.LIC_STEPS } }, true);
      x = y;
    }
    ctx.step(0.4);
    let other = x === A ? B : A;
    /* Posterize */
    if (st.poster > 0.01) {
      const lv = Math.max(2, Math.round(RB.POSTER_LEVELS / Math.max(st.poster, 0.1)));
      await this._pass('posterize', other, { uIn: x.tex, uN: lv });
      [x, other] = [other, x];
    }
    /* Cutout: tone-limited mean colour */
    if (st.cutout > 0.01) {
      const n = RB.CUT_LEVELS;
      let acc = C1, nxt = C2;
      gl_clear(this.gl, acc);
      for (let v = 0; v < n; v++) {
        await this._pass('levelMask', Tn, { uIn: x.tex, uV: v, uN: n });
        await this._blur(Tn.tex, Tn, Tmp, RB.CUT_SIGMA * k);
        await this._pass('levelAcc', nxt, { uIn: x.tex, uAcc: acc.tex, uB: Tn.tex, uV: v, uN: n });
        [acc, nxt] = [nxt, acc];
      }
      await this._pass('mix', other, { uA: x.tex, uB: acc.tex, uT: Math.min(1, st.cutout) });
      [x, other] = [other, x];
    }
    ctx.step(0.65);
    /* Displace + blur */
    if (st.warp > 0.01) {
      await this._pass('displace', other, { uIn: x.tex, uMap: tMap,
        uAmp: [RB.DISP_H / 100 * 128 * k * st.warp, RB.DISP_V / 100 * 128 * k * st.warp] });
      [x, other] = [other, x];
    }
    if (RB.BLUR * k * st.blur > 0.05) {
      await this._blur(x.tex, other, Tmp, RB.BLUR * k * st.blur);
      [x, other] = [other, x];
    }
    await this._pass('lut', other, { uIn: x.tex, uLut: tLut });
    [x, other] = [other, x];
    ctx.step(0.85);
    await this._pass('brushComposite', other, { uIn: x.tex, uTex: tTex, uCurve: tCurve,
      uT: RB.TEXTURE_OPACITY * st.texture, uAdjust: st.adjust, uVib: RB.VIB / 100, uSat: RB.SAT / 100 });
    return other;
  };
  function gl_clear(gl, t) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  }

  /* Watercolor settings - see watercolor.py */
  const WC = {
    DOC: 3000, BG: [240 / 255, 234.26 / 255, 226.82 / 255], BIG_BLUR_STEP: 8,
    LAYERS: [
      ['L2', { blur: 132, sat: 2.4 }, 'm_l2', 0, 1.0], ['L2c', { blur: 92, sat: 2.4 }, 'm_l2c', 0, 0.82],
      ['L1', { kw: 8, us: 0.92, us_r: 4.2 }, 'm_l1', 0, 1.0], ['L3', { kw: 7 }, 'm_l3', 1, 0.722],
      ['L4', { kw: 6 }, 'm_l4', 2, 0.62], ['L5', { poster: 8 }, 'm_l4', 0, 1.0],
      ['L6', { hatch: 10, hatch_mix: 0.5 }, 'm_l4', 0, 1.0], ['L7', { kw: 4, kwn: 2 }, 'm_l3', 3, 0.922]
    ],
    PAPERS: [['paper1', 4, 0.522, 0], ['paper2', 5, 0.22, 1], ['paper3', 5, 0.506, 1], ['paper4', 2, 0.22, 1]],
    LEVELS: [12, 235, 22, 255], GAMMA: 1.22, HUESAT: 18, VIB: 42, VSAT: -4
  };
  PIPELINES['Watercolor'] = async function (ctx, src, assets, look, p) {
    const f = (key) => Number(p[key] ?? 100) / 100;
    const st = vary({ frame: f('wc_frame'), wash: f('wc_wash'), blotch: f('wc_blotch'), detail: f('wc_detail'),
      color: f('wc_color'), paper: f('wc_paper'), adjust: f('wc_adjust') }, VARY['Watercolor'], p.wc_seed);
    const W = this.w, H = this.h, k = Math.max(W, H) / look.doc;
    const tSrc = ctx.img(src);
    const masks = {}; for (const n of ['m_l2', 'm_l2c', 'm_l1', 'm_l3', 'm_l4']) masks[n] = ctx.img(assets[n], true);
    const luts = {}; for (const [n] of WC.LAYERS) luts[n] = ctx.lutTex(look.layers[n]);
    const tLook = ctx.lutTex(look.lookup), tCurve = ctx.lutTex(look.curve.map(v => [v, v, v]));
    const A = ctx.T(), B = ctx.T(), X = ctx.T(), Y = ctx.T(), Tmp = ctx.T();
    const flip = (a, b) => [b, a];
    /* paper colour */
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, A.fbo);
    this.gl.clearColor(WC.BG[0], WC.BG[1], WC.BG[2], 1); this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    let acc = A, accN = B;
    const bigBlur = async (inTex, dst, sigma) => {
      if (sigma <= WC.BIG_BLUR_STEP * 2) { await this._blur(inTex, dst, Tmp, sigma); return; }
      const fct = Math.floor(sigma / WC.BIG_BLUR_STEP);
      const sw = Math.max(1, Math.floor(W / fct)), sh = Math.max(1, Math.floor(H / fct));
      const S1 = ctx.sized(sw, sh), S2 = ctx.sized(sw, sh), S3 = ctx.sized(sw, sh);
      this.w = sw; this.h = sh;
      try {
        await this._pass('down', S1, { uIn: inTex, uF: [W / sw, H / sh], uSrc: [W, H, 'i'] });
        await this._blur(S1.tex, S2, S3, sigma / fct);
      } finally { this.w = W; this.h = H; }
      await this._pass('up', dst, { uIn: S2.tex });
    };
    let i = 0;
    for (const [name, P, mask, mode, op] of WC.LAYERS) {
      let a = { tex: tSrc }, b = X, c = Y;
      if (P.blur) { await bigBlur(a.tex, b, P.blur * k * st.wash); a = b; [b, c] = [c, b]; }
      if (P.kw && st.blotch > 0.01) {
        const r = Math.max(1, Math.round(P.kw * k * st.blotch));
        for (let n = 0; n < (P.kwn || 1); n++) {
          await this._pass('kuwahara', b, { uIn: a.tex, uR: { i: r } }, true); a = b; b = (a === X) ? Y : X;
        }
      }
      if (P.hatch) { await this._pass('hatch', b, { uIn: a.tex, uLen: P.hatch * k, uMix: P.hatch_mix }); a = b; b = (a === X) ? Y : X; }
      if (P.poster) { await this._pass('posterL', b, { uIn: a.tex, uN: P.poster, uSoft: 6, uMix: 1 }); a = b; b = (a === X) ? Y : X; }
      if (P.us) {
        await this._blur(a.tex, Tmp, b, P.us_r * k);
        await this._pass('sharpen', b, { uIn: a.tex, uB: Tmp.tex, uAmt: P.us * st.detail }); a = b; b = (a === X) ? Y : X;
      }
      if (P.sat) { await this._pass('satAdj', b, { uIn: a.tex, uS: 1 + (P.sat - 1) * st.color }); a = b; b = (a === X) ? Y : X; }
      await this._pass('lut', b, { uIn: a.tex, uLut: luts[name] }); a = b;
      await this._pass('blendTex', accN, { uBase: acc.tex, uTop: a.tex, uMask: masks[mask], uMode: { i: mode },
        uMaskOn: { i: 1 }, uGray: { i: 0 }, uOp: op, uFrame: st.frame });
      [acc, accN] = flip(acc, accN);
      ctx.step(0.05 + 0.75 * (++i) / WC.LAYERS.length);
    }
    if (st.adjust > 0.001) {
      await this._pass('wcAdjust', accN, { uIn: acc.tex, uCurve: tCurve, uLook: tLook, uAmt: st.adjust,
        uLv: WC.LEVELS, uLvGamma: WC.GAMMA, uHS: WC.HUESAT / 100, uVib: WC.VIB / 100, uVSat: WC.VSAT / 100 });
      [acc, accN] = flip(acc, accN);
    }
    for (const [n, mode, op, gray] of WC.PAPERS) {
      const t = ctx.img(assets[n]);
      await this._pass('blendTex', accN, { uBase: acc.tex, uTop: t, uMask: t, uMode: { i: mode }, uMaskOn: { i: 0 },
        uGray: { i: gray }, uOp: op * st.paper, uFrame: 0 });
      [acc, accN] = flip(acc, accN);
    }
    return acc;
  };

  /* Halftone - see halftone.py (apply) */
  function hexRGB(c) { c = String(c || '#000000').replace('#', ''); return [0, 2, 4].map(i => parseInt(c.substr(i, 2), 16) / 255); }
  function htRamp(film, dark, light, filmAmt) {
    const d = hexRGB(dark), l = hexRGB(light), f = filmAmt / 100, n = film.length, rows = [];
    for (let i = 0; i < 256; i++) {
      const x = i / 255, row = [];
      for (let c = 0; c < 3; c++) {
        /* np.interp over the film curve at t = 0, 1/8 ... 1 */
        const pos = x * (n - 1), j = Math.min(n - 2, Math.floor(pos)), fr = pos - j;
        const cv = (k) => film[k][c] * f + (k / (n - 1)) * (1 - f);
        row.push(d[c] + (l[c] - d[c]) * (cv(j) + (cv(j + 1) - cv(j)) * fr));
      }
      rows.push(row);
    }
    return rows;
  }
  PIPELINES['Halftone'] = async function (ctx, src, assets, look, p, tex) {
    const W = this.w, H = this.h, s = Math.max(W, H) / look.ref_side;
    const num = (k, d) => Number(p[k] ?? d);
    const c = (num('ht_contrast', 50) - 50) / 50, lo = 15 + 20 * c, hi = 224 - 20 * c;
    const kc = num('ht_clarity', 50) / 50;
    const tSrc = ctx.img(src), tN = ctx.f16(tex.grain), tV = ctx.f16(tex.veins);
    const tC = tex.cloth ? ctx.f16(tex.cloth) : tN;
    const tRamp = ctx.lutTex(htRamp(look.film, p.ht_dark || '#21130d', p.ht_light || '#f9f4c4', num('ht_film', 100)));
    const G = ctx.T(), G2 = ctx.T(), Bl = ctx.T(), Tmp = ctx.T(), B1 = ctx.T(), B2 = ctx.T(), B3 = ctx.T(), B4 = ctx.T();
    await this._pass('htBW', G, { uIn: tSrc, uLo: lo, uHi: hi });
    await this._blur(G.tex, Bl, Tmp, 30 * s);                                   /* clarity */
    await this._pass('sharpen', G2, { uIn: G.tex, uB: Bl.tex, uAmt: 0.45 * kc });
    await this._blur(G2.tex, Bl, Tmp, 5 * s);                                   /* texture */
    await this._pass('sharpen', G, { uIn: G2.tex, uB: Bl.tex, uAmt: 0.50 * kc });
    ctx.step(0.35);
    await this._blur(G.tex, B1, Tmp, 2.5 * s);
    await this._blur(G.tex, B2, Tmp, 0.7 * s);
    await this._blur(G.tex, B3, Tmp, 2.0 * s);
    await this._blur(G.tex, B4, Tmp, 0.6 * s);
    const edges = num('ht_edges', 50);
    await this._pass('htStipple', G2, { uG: G.tex, uB25: B1.tex, uB07: B2.tex, uB20: B3.tex, uB06: B4.tex, uN: tN,
      uGrain: num('ht_grain', 50), uEdges: edges, uThr: 0.065 * Math.pow(50 / Math.max(edges, 1), 0.6) });
    ctx.step(0.7);
    const grid = num('ht_grid', 100) / 100;
    await this._pass('htFinish', G, { uX: G2.tex, uCloth: tC, uVeins: tV, uRamp: tRamp, uGrid: grid,
      uClothOn: { i: grid > 0 && tex.cloth ? 1 : 0 } });
    return G;
  };

  /* Impression - see impression.py (apply) */
  PIPELINES['Impression'] = async function (ctx, src, assets, look, p, tex) {
    const W = this.w, H = this.h, s = Math.max(W, H) / look.ref_side;
    const hs = Math.max(8, Math.floor(H / 2)), ws = Math.max(8, Math.floor(W / 2));
    const num = (k, d) => Number(p[k] ?? d);
    const dab = num('im_dab', 50) / 50, light = num('im_light', 50) / 50, relief = num('im_relief', 50) / 50,
      canvasAmt = num('im_canvas', 50) / 50, swirl = num('im_swirl', 50) / 100, border = num('im_border', 50) / 50;
    const tSrc = ctx.img(src);
    const T = () => ctx.T(), Th = () => ctx.sized(ws, hs);
    const A = T(), B = T(), C = T(), D = T(), Tmp = T(), Lt = T();
    const hA = Th(), hB = Th(), hC = Th(), hDir = Th(), hTmp = Th(), small = Th(), pre = Th();
    /* 1. flow at half size */
    await this._at(ws, hs, () => this._pass('down', small, { uIn: tSrc, uF: [W / ws, H / hs], uSrc: [W, H, 'i'] }));
    let p90 = 0;
    await this._at(ws, hs, async () => {
      await this._pass('luma', hA, { uIn: small.tex });
      await this._pass('tensor', hB, { uIn: hA.tex });
      await this._blur(hB.tex, hB, hTmp, 10 * (s / 2) + 1);
      await this._pass('trace', hC, { uT: hB.tex });
    });
    p90 = await this._pct(hC.tex, ws, hs, 90);
    const tSw = ctx.f16(tex.swirl);
    await this._at(ws, hs, async () => {
      await this._pass('imFlowCS', hA, { uT: hB.tex, uSw: tSw, uP90: p90, uSwirl: swirl });
      await this._blur(hA.tex, hA, hTmp, 3 * (s / 2) + 0.5);
      await this._pass('imDir', hDir, { uCS: hA.tex });
    });
    ctx.step(0.15);
    /* 2. dabs: 7 shuffled copies, their median */
    await this._blur(tSrc, A, Tmp, 3 * s);
    await this._at(ws, hs, () => this._pass('down', pre, { uIn: A.tex, uF: [W / ws, H / hs], uSrc: [W, H, 'i'] }));
    const packs = [];
    for (let i = 0; i < 7; i++) {
      const nxT = ctx.f16(tex['dx' + i]), nyT = ctx.f16(tex['dy' + i]);
      await this._at(ws, hs, async () => {
        await this._pass('licW', hA, { uIn: nxT, uD: hDir.tex, uLen: 5, uSteps: { i: 4 } });
        await this._pass('licW', hB, { uIn: nyT, uD: hDir.tex, uLen: 5, uSteps: { i: 4 } });
      });
      const sx = await this._std(hA.tex, ws, hs, hTmp), sy = await this._std(hB.tex, ws, hs, hTmp);
      const pk = Th();
      await this._at(ws, hs, () => this._pass('pack2', pk, { uA: hA.tex, uB: hB.tex, uInv: [1 / (sx + 1e-6), 1 / (sy + 1e-6)] }));
      packs.push(pk);
    }
    await this._at(ws, hs, () => this._pass('median7', hC, { uPre: pre.tex, uN0: packs[0].tex, uN1: packs[1].tex,
      uN2: packs[2].tex, uN3: packs[3].tex, uN4: packs[4].tex, uN5: packs[5].tex, uN6: packs[6].tex, uAmp: 7 * s * dab }));
    ctx.step(0.4);
    await this._pass('upCubic', A, { uIn: hC.tex, uSrc: [ws, hs, 'i'] });
    await this._pass('lumClip', Lt, { uIn: A.tex });                 /* L before the tone curve */
    await this._pass('imTone', B, { uIn: A.tex, uTX: look.tone_x, uTY: look.tone_y });
    await this._pass('imStripe', A, { uIn: B.tex, uMod: ctx.f16(tex.mod), uP: 5.75 * s + 0.4, uAmt: 0.012 * relief });
    /* bristle ripples */
    await this._at(ws, hs, () => this._pass('licW', hA, { uIn: ctx.f16(tex.rid), uD: hDir.tex, uLen: 6 * s + 1, uSteps: { i: 4 } }));
    await this._pass('upCubic', C, { uIn: hA.tex, uSrc: [ws, hs, 'i'] });
    const ridStd = await this._std(C.tex, W, H, Tmp);
    await this._blur(tSrc, B, Tmp, 1.5 * s);
    await this._pass('luma', D, { uIn: B.tex });
    await this._pass('sobelMag', B, { uIn: D.tex });
    const p80 = await this._pct(B.tex, W, H, 80);
    await this._blur(B.tex, D, Tmp, 12 * s + 1);
    const ang = 135 * Math.PI / 180;
    await this._pass('imRid', B, { uIn: A.tex, uRid: C.tex, uDet: D.tex, uL: Lt.tex, uInvStd: 1 / (ridStd + 1e-6),
      uP80: p80, uAmt: 0.035 * relief, uOff: [Math.cos(ang) * 1.0, -Math.sin(ang) * 1.0] });
    /* Unsharp Mask x2 */
    await this._blur(B.tex, C, Tmp, 2.8 * s);
    await this._pass('usNC', A, { uIn: B.tex, uB: C.tex, uAmt: 1.81, uClamp: { i: 0 } });
    await this._blur(A.tex, C, Tmp, 1.6 * s);
    await this._pass('usNC', B, { uIn: A.tex, uB: C.tex, uAmt: 1.01, uClamp: { i: 1 } });
    let cur = B;
    ctx.step(0.65);
    /* 3. light strokes */
    if (light > 0 && tex.mask) {
      await this._at(ws, hs, async () => {
        await this._pass('licW', hA, { uIn: small.tex, uD: hDir.tex, uLen: 28 * s + 2, uSteps: { i: 10 } }, true);
        await this._pass('licW', hB, { uIn: ctx.f16(tex.mask), uD: hDir.tex, uLen: 14 * s + 1, uSteps: { i: 6 } });
      });
      await this._pass('upCubic', C, { uIn: hA.tex, uSrc: [ws, hs, 'i'] });
      await this._pass('upCubic', D, { uIn: hB.tex, uSrc: [ws, hs, 'i'] });
      const mStd = await this._std(D.tex, W, H, Tmp);
      await this._pass('imLight', A, { uIn: cur.tex, uLong: C.tex, uM: D.tex, uInvStd: 1 / (mStd + 1e-6), uAmt: Math.min(1, light) });
      cur = A;
    }
    ctx.step(0.8);
    /* 4. bristle grooves + canvas (Overlay) */
    const other = cur === A ? B : A;
    await this._pass('imSurface', other, { uIn: cur.tex, uSrc: tSrc, uBump: tex.bump ? ctx.f16(tex.bump) : tSrc,
      uCanvas: tex.canvas ? ctx.f16(tex.canvas) : tSrc, uRelief: relief, uCanvasAmt: canvasAmt,
      uBumpOn: { i: relief > 0 && tex.bump ? 1 : 0 }, uCanvasOn: { i: canvasAmt > 0 && tex.canvas ? 1 : 0 } });
    cur = other;
    /* 5. border */
    if (border > 0) {
      const b = Math.round(125 * s * border);
      const o2 = cur === A ? B : A;
      await this._pass('border', o2, { uIn: cur.tex, uB: b });
      cur = o2;
    }
    return cur;
  };

  /* Painted Photo - see painted.py (apply) */
  PIPELINES['Painted Photo'] = async function (ctx, src, assets, look, p, tex, ana) {
    const W = this.w, H = this.h, s = Math.max(W, H) / look.ref_side;
    const num = (k, d) => Number(p[k] ?? d);
    const detail = num('pp_detail', 50), levels = Math.round(num('pp_levels', 8));
    const tSrc = ctx.img(src);
    const A = ctx.T(), B = ctx.T(), C = ctx.T(), D = ctx.T(), E = ctx.T(), Tmp = ctx.T();
    /* 1. flat patches */
    /* (kept for the next live preview: the later settings do not remake it) */
    const cut = await ctx.memo('pp1', [detail, levels, ana], async keep => {
      const o = keep();
      await this._cutout(ctx, tSrc, o, s, levels, detail, ana, look.cut_sigma, look.wide_step);
      return o;
    });
    ctx.step(0.5);
    /* 2. brushwork */
    await this._pass('ppRemap', B, { uIn: cut.tex, uDx: ctx.f16(tex.dx), uDy: ctx.f16(tex.dy), uAmt: 1.6 * s });
    await this._blur(B.tex, C, Tmp, 12 * s);
    await this._pass('usNC', A, { uIn: B.tex, uB: C.tex, uAmt: 0.3, uClamp: { i: 1 } });
    let cur = A;
    const outline = num('pp_outline', 50);
    if (outline > 0) {
      const thr = 0.2 * Math.pow(50 / Math.max(outline, 1), 0.5);
      await this._blur(tSrc, B, Tmp, 3 * s); await this._pass('luma', C, { uIn: B.tex });        /* Lb */
      await this._pass('sobelMag', B, { uIn: C.tex });
      await this._blur(C.tex, D, Tmp, 2.5 * s);
      await this._pass('ppEdge', E, { uMag: B.tex, uL: C.tex, uGL: D.tex, uThr: thr, uDen: 0.18, uDark: 0 });
      await this._pass('maxFilt', B, { uIn: E.tex, uW: { i: Math.max(1, Math.round(3 * s)) } });
      await this._blur(B.tex, E, Tmp, 0.6 * s);                                                   /* edge */
      await this._blur(tSrc, B, Tmp, 1.0 * s); await this._pass('luma', C, { uIn: B.tex });     /* Lf */
      await this._pass('sobelMag', B, { uIn: C.tex });
      await this._blur(C.tex, D, Tmp, 2.0 * s);
      const fineT = ctx.T();
      await this._pass('ppEdge', fineT, { uMag: B.tex, uL: C.tex, uGL: D.tex, uThr: 2.2 * thr, uDen: 0.2, uDark: 0.04 });
      await this._blur(fineT.tex, D, Tmp, 0.5 * s);
      await this._pass('ppOutline', B, { uIn: cur.tex, uE: E.tex, uF: D.tex });
      cur = B;
    }
    const t = num('pp_texture', 50) / 50;
    if (t > 0 && tex.cells) {
      const o2 = cur === A ? B : A;
      await this._pass('ppTexture', o2, { uIn: cur.tex, uCells: ctx.f16(tex.cells), uShade: ctx.f16(tex.shade), uT: t });
      cur = o2;
    }
    ctx.step(0.75);
    const paper = num('pp_paper', 100) / 100;
    if (paper > 0 && tex.kraft) {
      const o2 = cur === A ? B : A;
      await this._pass('ppPaper', o2, { uIn: cur.tex, uKraft: ctx.f16(tex.kraft), uOld: ctx.f16(tex.oldpaper), uAmt: paper });
      cur = o2;
    }
    /* 3. film grade */
    await this._blur(cur.tex, C, Tmp, 1.6 * s);
    const o3 = cur === A ? B : A;
    await this._pass('ppGrade', o3, { uIn: cur.tex, uSm: C.tex, uG: [].concat(...look.grade), uAmt: num('pp_grade', 100) / 100 });
    return o3;
  };

  /* Oil Smear - see smear.py (apply) */
  PIPELINES['Oil Smear'] = async function (ctx, src, assets, look, p, tex, ana) {
    const W = this.w, H = this.h, s = Math.min(W, H) / look.ref_side;
    const num = (k, d) => Number(p[k] ?? d);
    const detail = num('sm_detail', 50), levels = Math.round(num('sm_levels', 8));
    const direction = num('sm_direction', 0) / 100, smear = num('sm_smear', 50) / 50;
    const texAmt = num('sm_texture', 50) / 50, toneAmt = num('sm_tone', 50) / 50, relief = num('sm_relief', 50) / 50;
    const tSrc = ctx.img(src), tTex = ctx.f16(tex.tex);
    const A = ctx.T(), B = ctx.T(), C = ctx.T(), D = ctx.T(), E = ctx.T(), F = ctx.T(), Tmp = ctx.T();
    /* 1-2. the flat paint - kept for the next live preview (Smear, Texture,
       Contrast and Direction of the smear do not remake it) */
    const paint = await ctx.memo('sm1', [detail, levels, relief, p.sm_seed, p.sm_direction, p.sm_curve, ana], async keep => {
      /* Oil Paint clean-up, its brush relief, Posterize 10 */
      const r = Math.max(1, Math.round(Math.max(1, 6 * s * (1.5 - detail / 100))));
      await this._guided2(tSrc, A, r, 0.003, [B, C, D, E, F, Tmp]);
      await this._pass('smPoster', B, { uIn: A.tex, uRel: ctx.f16(tex.relief), uAmt: relief });
      ctx.step(0.2);
      /* cut-out, blur 2 */
      await this._cutout(ctx, B.tex, A, s, levels, detail, ana, look.cut_sigma, look.wide_step);
      const o = keep();
      await this._blur(A.tex, o, Tmp, 2 * s);
      return o;
    });
    ctx.step(0.7);
    /* 3. smear through the stroke texture */
    let cur = paint;
    if (smear > 0) {
      const k = 128 * Math.min(W, H) / look.ref_side;
      await this._pass('displace', A, { uIn: paint.tex, uMap: tTex,
        uAmp: [(30 + 30 * direction) * smear / 100 * k, (60 - 30 * direction) * smear / 100 * k] });
      cur = A;
    }
    /* 4-6. tone, surface, finish */
    const out = cur === A ? B : A;
    await this._pass('smFinish', out, { uIn: cur.tex, uTex: tTex, uTone: ctx.lutTex(look.tone), uAux: ctx.lutTex(look.aux),
      uOv: Math.min(1, 145 / 255 * texAmt), uMul: texAmt > 0 ? Math.min(1, 69 / 255 * texAmt) : 0, uTone_: toneAmt });
    return out;
  };

  /* Canvas Oil - see canvas_oil.py (apply) */
  PIPELINES['Canvas Oil'] = async function (ctx, src, assets, look, p) {
    const W = this.w, H = this.h, s = Math.max(W, H) / look.ref_side;
    const f = (k, d) => Number(p[k] ?? d) / 100;
    const st = { patch: f('co_patch', 45), detail: f('co_detail', 55), drag: f('co_drag', 40), dabs: f('co_dabs', 55),
      relief: f('co_relief', 45), canvas: f('co_canvas', 45), scratch: f('co_scratches', 20), speck: f('co_speckles', 40),
      brush: f('co_brush', 70), size: f('co_stroke_size', 45), loose: f('co_loose', 55), sketch: f('co_sketch', 45),
      contrast: f('co_contrast', 35), vib: f('co_vibrance', 40) };
    const K = (Math.trunc(Number(p.co_seed ?? 7)) % 100000) * 101;
    const tSrc = ctx.img(src);
    const X = ctx.T(), Y = ctx.T(), Z = ctx.T(), Tmp = ctx.T();
    const pool = [ctx.T(), ctx.T(), ctx.T(), ctx.T()];
    const other = (...used) => pool.find(t => !used.includes(t));
    /* stages 1-5 are kept for the next live preview: a later setting
       (rims, canvas, sketch, scratches, speckles) does not remake them */
    const d1 = [st.contrast, st.vib];
    /* 1. tone, the colour the patches take, where the detail is */
    const { Tn, CS, DM } = await ctx.memo('co1', d1, async keep => {
      const Tn = keep(), CS = keep(), DM = keep();
      let lo = 0, hi = 1;
      if (st.contrast > 0) {
        const [q0, q1] = await this._pctRGB(tSrc, W, H, [0.6, 99.4]);
        lo = q0 * st.contrast; hi = 1 - (1 - q1) * st.contrast;
      }
      await this._pass('coTone', Tn, { uIn: tSrc, uLo: lo, uHi: hi, uC: st.contrast, uV: st.vib });
      await this._blur(Tn.tex, CS, Tmp, 1.0 * s);
      await this._pass('luma', X, { uIn: Tn.tex });
      await this._pass('sobelMag', Y, { uIn: X.tex });
      await this._wide(ctx, Y.tex, Z, 10 * s);
      const p90 = await this._pct(Z.tex, W, H, 90);
      await this._pass('coScaleR', DM, { uIn: Z.tex, uP: p90 });
      return { Tn, CS, DM };
    });
    /* 2-4. patches, knife dabs, wet-paint runs */
    const d2 = d1.concat([st.patch, st.detail, st.drag, st.dabs, K]);
    let cur = await ctx.memo('co2', d2, async keep => {
      const big = (10 + 50 * st.patch) * s;
      let c = pool[0];
      await this._pass('coCells', c, { uSrc: CS.tex, uSize: big, uSq: 1 + 0.9 * st.drag, uK: K + 10, uLam: 8, uWarp: { i: 1 } }, true);
      if (st.detail > 0.01) {
        const fb = other(c), mx = other(c, fb);
        await this._pass('coCells', fb, { uSrc: CS.tex, uSize: Math.max(4 * s, big * 0.32), uSq: 1 + 0.54 * st.drag, uK: K + 20, uLam: 8, uWarp: { i: 1 } }, true);
        await this._pass('coFine', mx, { uA: c.tex, uB: fb.tex, uDm: DM.tex, uBig: big, uD: st.detail, uK: K + 30 });
        c = mx;
      }
      const out = keep();
      if (st.dabs > 0.01 || st.drag > 0.01) {
        await this._pass('coDabs', out, { uIn: c.tex, uDm: DM.tex, uSize: (6 + 10 * st.patch) * s, uSq: 1 + 0.8 * st.drag,
          uK: K + 40, uAmt: st.dabs, uDrag: st.drag, uS: s, uKs: K + 60 }, true);
      } else await this._pass('out', out, { uIn: c.tex });
      return out;
    });
    ctx.step(0.25);
    /* 5. brushwork */
    let relief = st.relief;
    if (st.brush > 0.01) {
      const d3 = d2.concat([st.brush, st.size, st.loose]);
      const base = cur;
      cur = await ctx.memo('co3', d3, async keep => {
        await this._blur(base.tex, X, Tmp, 1.5 * s);
        await this._pass('luma', Y, { uIn: X.tex });
        await this._pass('tensor', Z, { uIn: Y.tex });
        await this._blur(Z.tex, Z, Tmp, 4 * s + 1);
        await this._pass('coDir', Y, { uT: Z.tex, uK: K + 70, uS: s, uLoose: st.loose });
        const ln = (8 + 22 * st.size + 10 * st.loose) * s, sw = (3 + 8 * st.size) * s;
        const S1 = pool[0], S2 = pool[1], Qs = pool[2];
        await this._pass('coLic', S1, { uIn: base.tex, uD: Y.tex, uLen: ln * 0.6, uSteps: { i: 6 } }, true);
        await this._pass('coLic', S2, { uIn: base.tex, uD: Y.tex, uLen: ln * 1.8, uSteps: { i: 8 } }, true);
        await this._pass('coSweep', Qs, { uA: S1.tex, uB: S2.tex, uDm: DM.tex });
        await this._pass('coNoise', X, { uSw: sw, uK: K + 71 });
        await this._pass('coLic', S1, { uIn: X.tex, uD: Y.tex, uLen: ln, uSteps: { i: 6 } }, true);
        await this._pass('coLic', S2, { uIn: X.tex, uD: Y.tex, uLen: ln * 1.5, uSteps: { i: 6 } }, true);
        await this._pass('coHgt', Z, { uA: S1.tex, uB: S2.tex });
        await this._blur(Qs.tex, X, Tmp, 6 * s);
        const out = keep();
        await this._pass('coBrush', out, { uIn: base.tex, uQ: Qs.tex, uQb: X.tex, uH: Z.tex, uLoose: st.loose, uAmt: st.brush });
        return out;
      });
      relief *= 1 - 0.55 * st.brush;
    }
    ctx.step(0.5);
    /* 6. rims */
    await this._pass('luma', X, { uIn: cur.tex });
    await this._blur(X.tex, Y, Tmp, 0.9 * s);
    await this._pass('coRimA', Z, { uIn: Y.tex });
    const e98 = await this._pctCh(Z.tex, W, H, 98.5, 1), m99 = await this._pctCh(Z.tex, W, H, 99, 2);
    await this._pass('coMagN', Y, { uIn: Z.tex, uP: m99 });
    await this._blur(Y.tex, Y, Tmp, 0.6 * s);
    const Gb = other(cur), o3 = other(cur, Gb);
    await this._blur(X.tex, Gb, Tmp, 2 * s);
    await this._pass('coRims', o3, { uIn: cur.tex, uE: Z.tex, uSeam: Y.tex, uG: X.tex, uGb: Gb.tex, uE98: e98, uM99: m99, uRel: relief });
    cur = o3;
    /* 7. mottled pigment, grain, the canvas opening up */
    const o4 = other(cur);
    await this._pass('coSurface', o4, { uIn: cur.tex, uK: K + 80, uS: s, uDrag: st.drag, uCanvas: st.canvas });
    cur = o4;
    ctx.step(0.65);
    /* 8. sketch lines */
    if (st.sketch > 0.01) {
      await this._pass('luma', X, { uIn: CS.tex });
      await this._blur(X.tex, Y, Tmp, 1.6 * s);
      await this._pass('sobelMag', Z, { uIn: Y.tex });
      const thr = await this._pct(Z.tex, W, H, 100 - (4 + 10 * st.sketch));
      const B1 = other(cur), B2 = other(cur, B1);
      await this._blur(X.tex, B1, Tmp, 1.0 * s);
      await this._blur(X.tex, B2, Tmp, 2.2 * s);
      await this._pass('coDiff', Y, { uA: B1.tex, uB: B2.tex });
      await this._pass('coEMG', B1, { uIn: Y.tex, uMag: Z.tex });
      await this._pass('coSketch', B2, { uIn: cur.tex, uEMG: B1.tex, uThr: thr, uA: st.sketch, uS: s, uK: K });
      cur = B2;
    }
    /* 9. knife scratches */
    if (st.scratch > 0.01) {
      const o5 = other(cur);
      await this._pass('coScratch', o5, { uIn: cur.tex, uDm: DM.tex, uA: st.scratch, uS: s, uK: K }, true);
      cur = o5;
    }
    /* 10. speckles */
    if (st.speck > 0.01) {
      await this._pass('coDots', X, { uIn: cur.tex, uT: Tn.tex, uA: st.speck, uS: s, uK: K }, true);
      await this._blur(X.tex, Y, Tmp, 0.5 * s);
      await this._blur(X.tex, Z, Tmp, 2.5 * s);
      const o6 = other(cur);
      await this._pass('coDotMix', o6, { uIn: cur.tex, uD1: Y.tex, uD2: Z.tex });
      cur = o6;
    }
    ctx.step(0.85);
    /* 11. canvas weave, a light unsharp mask */
    if (st.canvas > 0.01) {
      await this._pass('coWeaveTex', X, { uK: K + 180, uS: s });
      const o7 = other(cur);
      await this._pass('coWeave', o7, { uIn: cur.tex, uW: X.tex, uCanvas: st.canvas });
      cur = o7;
    }
    await this._blur(cur.tex, X, Tmp, 1.2 * s);
    const o8 = other(cur);
    await this._pass('usNC', o8, { uIn: cur.tex, uB: X.tex, uAmt: 0.35, uClamp: { i: 1 } });
    return o8;
  };

  /* Silk Oil - see silkoil.py (apply) */
  PIPELINES['Silk Oil'] = async function (ctx, src, assets, look, p) {
    const W = this.w, H = this.h, k = Math.min(W, H) / look.ref_short;
    const f = key => Number(p[key] ?? 100) / 100;
    const st = vary({ smooth: f('sl_smooth'), smear: f('sl_smear'), streaks: f('sl_streaks'), texture: f('sl_texture'),
      light: f('sl_light'), color: f('sl_color') }, VARY['Silk Oil'], p.sl_seed);
    const K = (Math.trunc(Number(p.sl_seed ?? 0)) % 100000) * 101;
    const tSrc = ctx.img(src), tTex = ctx.img(assets.texture);
    const A = ctx.T(), B = ctx.T(), C = ctx.T(), D = ctx.T(), Tmp = ctx.T();
    const t = st.texture;
    /* the stages are kept for the next live preview: moving Colour only
       remakes the colour, moving Streaks only the oil paint and colour */
    const d1 = [st.smooth, t];
    const { Ht, base } = await ctx.memo('sl1', d1, async keep => {
      /* 1. underpainting (Kuwahara, 30 %) */
      let cur = { tex: tSrc };
      if (st.smooth > 0.01) {
        const r = Math.max(1, Math.round(3 * k));
        await this._pass('kuwahara', A, { uIn: tSrc, uR: { i: r } }, true);
        await this._pass('slMix', B, { uA: tSrc, uB: A.tex, uAmt: 0.30 * st.smooth });
        cur = B;
      }
      /* 2. glass + texturizer */
      const Ht = keep(), base = keep();
      await this._blur(tTex, Ht, Tmp, 1.0 * k);
      if (t > 0.01) {
        const a = Math.PI * 225 / 180;
        await this._pass('slGlass', base, { uIn: cur.tex, uH: Ht.tex, uAmp: 3.0 * k * t, uTz: 0.15 * t, uLx: Math.cos(a), uLy: -Math.sin(a) });
      } else await this._pass('out', base, { uIn: cur.tex });
      return { Ht, base };
    });
    ctx.step(0.25);
    /* 3. Graphic Pen, Soft Light */
    const d2 = d1.concat([st.light, K]);
    const pen = await ctx.memo('sl2', d2, async keep => {
      const o = keep();
      if (st.light > 0.01) {
        await this._pass('slPenIn', D, { uIn: base.tex, uK: K + 7 });
        await this._pass('slConstDir', Tmp, {});
        await this._pass('coLic', A, { uIn: D.tex, uD: Tmp.tex, uLen: 15 * k, uSteps: { i: 6 } }, true);
        await this._pass('slPen', o, { uIn: base.tex, uP: A.tex, uT: 0.15, uMix: 0.30 * st.light });
      } else await this._pass('out', o, { uIn: base.tex });
      return o;
    });
    ctx.step(0.4);
    /* 4. Oil Paint (Darken): smear along the flow with bristle streaks */
    const d3 = d2.concat([st.smear, st.streaks]);
    let cur = await ctx.memo('sl3', d3, async keep => {
      if (!(st.smear > 0.01)) return pen;
      await this._blur(pen.tex, A, Tmp, 2.0 * k);
      await this._pass('luma', B, { uIn: A.tex });
      await this._pass('tensor', A, { uIn: B.tex });
      await this._blur(A.tex, A, Tmp, 40 * k + 1);
      await this._pass('dir', D, { uT: A.tex });
      await this._pass('coLic', A, { uIn: pen.tex, uD: D.tex, uLen: 45 * k * st.smear, uSteps: { i: 8 } }, true);
      await this._pass('slNoise', B, { uK: K + 11 });
      await this._pass('coLic', Tmp, { uIn: B.tex, uD: D.tex, uLen: 24 * k, uSteps: { i: 10 } }, true);
      const o = keep();
      await this._pass('slOil', o, { uIn: pen.tex, uO: A.tex, uG: Tmp.tex, uGroove: 0.07 * st.streaks });
      return o;
    });
    ctx.step(0.75);
    /* 5. texturizer again, lit from the top */
    if (t > 0.01) {
      await this._pass('slGlass', C, { uIn: cur.tex, uH: Ht.tex, uAmp: 0, uTz: 0.15 * t, uLx: Math.cos(Math.PI / 2), uLy: -Math.sin(Math.PI / 2) });
      cur = C;
    }
    /* 6. colour */
    const o6 = cur === A ? B : A;
    await this._pass('slColor', o6, { uIn: cur.tex, uLut: ctx.lutTex(look.lut), uSat: look.sat, uAmt: st.color });
    return o6;
  };

  /* Sketch Paint settings - see sketchfx.py */
  const SK = {
    SB_RADIUS: 6, SB_SIGMA_COLOR: 0.09, LIC_LEN: 8, SHARP_AMT: 3.0, SHARP_SIGMA: 4.5,
    KW_RADIUS: 3, KW_PASSES: 3, POSTER_LEVELS: 6, POSTER_SOFT: 6, POSTER_MIX: 0.5,
    EDGE_AMT: 10, EDGE_SIGMA: 1.5, OUT_SHARP: 1.21, OUT_SIGMA: 4.5,
    PC_SIGMA: 8, PC_GAIN: 12, PC_OFFSET: 0.1, SKETCH_OPACITY: 0.702, OUTLINE_OPACITY: 0.71,
    LEVELS: [28, 211, 0.94]
  };
  const ASSETS = { 'Oil Paint FX': ['mask', 'tone', 'texture'], 'Sketch Paint': ['sketch'],
    'Rough Brush': ['displace', 'texture'], 'Silk Oil': ['texture'],
    'Watercolor': ['m_l2', 'm_l2c', 'm_l1', 'm_l3', 'm_l4', 'paper1', 'paper2', 'paper3', 'paper4'] };
  /* pictures that may come at half size (soft masks), sampled smoothly */
  const HALF = new Set(['m_l2', 'm_l2c', 'm_l1', 'm_l3', 'm_l4']);

  /* every source picture gets a number (the stage memo knows its input) */
  const srcIds = new WeakMap(); let srcSeq = 0;

  /* --- glue: fetch the template's pictures for a size and render ------- */
  const cache = { look: {}, assets: {}, key: {} };
  async function bitmap(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error('Could not load ' + url);
    return createImageBitmap(await r.blob(), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  }
  const texCache = new Map();
  async function textures(style, w, h, p) {
    const r = await fetch('/api/fx/textures', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ style, w, h, params: p }) });
    if (!r.ok) throw new Error('Could not make the textures');
    const meta = await r.json();
    if (!meta.token) return {};
    if (texCache.has(meta.token)) return texCache.get(meta.token);
    const out = {};
    await Promise.all(meta.items.map(async it => {
      const b = await fetch(`/api/fx/texture/${meta.token}/${it.name}`);
      if (!b.ok) throw new Error('Could not load the texture ' + it.name);
      out[it.name] = { data: new Uint16Array(await b.arrayBuffer()), w: it.w, h: it.h, ch: it.ch };
    }));
    texCache.set(meta.token, out);
    while (texCache.size > 4) texCache.delete(texCache.keys().next().value);
    return out;
  }
  async function prepared(style, w, h) {
    const q = 'style=' + encodeURIComponent(style);
    if (!cache.look[style]) cache.look[style] = await (await fetch('/api/fx/look?' + q)).json();
    const key = w + 'x' + h;
    if (cache.key[style] !== key) {
      const names = ASSETS[style] || [];
      const sz = n => HALF.has(n) ? `w=${Math.max(8, Math.round(w / 2))}&h=${Math.max(8, Math.round(h / 2))}` : `w=${w}&h=${h}`;
      const pics = await Promise.all(names.map(n => bitmap(`/api/fx/asset/${n}?${q}&${sz(n)}`)));
      cache.assets[style] = Object.fromEntries(names.map((n, i) => [n, pics[i]]));
      cache.key[style] = key;
    }
    return { look: cache.look[style], assets: cache.assets[style] };
  }
  let fx = null;
  const get = () => (fx = fx || new OilFX());
  /* a source picture at another size, made once and kept (the stage memo
     then sees the same picture again) */
  const sizedCache = new WeakMap();
  async function resized(src, w, h) {
    if (w === src.width && h === src.height) return src;
    let m = sizedCache.get(src);
    if (!m) { m = {}; sizedCache.set(src, m); }
    const key = w + 'x' + h;
    if (!m[key]) {
      const c = document.createElement('canvas'); c.width = w; c.height = h;
      const x = c.getContext('2d'); x.imageSmoothingQuality = 'high'; x.drawImage(src, 0, 0, w, h);
      m[key] = await createImageBitmap(c);
    }
    return m[key];
  }
  /* numbers the server reads from the photo (Painted Photo, Oil Smear),
     kept for the settings they depend on */
  const anaCache = new Map();
  async function analysed(style, p, source, w, h) {
    const pre = { 'Painted Photo': 'pp_', 'Oil Smear': 'sm_' }[style] || '';
    const deps = Object.keys(p).filter(k => pre && k.startsWith(pre)).sort().map(k => [k, p[k]]);
    const key = JSON.stringify([style, source, w, h, deps]);
    if (source !== 'run' && anaCache.has(key)) return anaCache.get(key);
    const r = await fetch('/api/fx/analyse', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ style, params: p, source, w, h }) });
    if (!r.ok) throw new Error('Could not read the photo');
    const v = await r.json();
    if (source !== 'run') { anaCache.set(key, v); while (anaCache.size > 12) anaCache.delete(anaCache.keys().next().value); }
    return v;
  }

  window.OilFX = {
    STYLES: Object.keys(PIPELINES),          /* styles the GPU can make */
    SEED_KEY,
    has(style) { return style in PIPELINES; },
    get ok() { return get().ok; },
    /* src: ImageBitmap of the photo; p.paint_style picks the style;
       returns RGBA pixels */
    /* src: ImageBitmap of the photo; p.paint_style picks the style.
       opts.proxy: a long side to make a quick preview at (while a slider
       is dragged - like Photoshop's proxy); opts.memo: keep the stages for
       the next preview; opts.source: 'run' | 'live'. Returns RGBA pixels at
       src's size, or {px, w, h} at the size it was made when opts.raw. */
    async run(src, p, onProgress, opts) {
      opts = opts || {};
      const style = p.paint_style;
      if (!cache.look[style]) await prepared(style, 8, 8);
      const W0 = src.width, H0 = src.height;
      const ms = cache.look[style].min_side || 0;
      const side = cache.look[style].min_by === 'short' ? Math.min : Math.max;
      let tw = W0, th = H0;
      if (opts.proxy && Math.max(W0, H0) > opts.proxy) {
        const k = opts.proxy / Math.max(W0, H0); tw = Math.round(W0 * k); th = Math.round(H0 * k);
      } else if (!opts.proxy && ms && side(W0, H0) < ms) {
        /* styles whose grain lives in pixels are made at least min_side big
           (the server already did that for a Run; a live preview is smaller) */
        const k = ms / side(W0, H0); tw = Math.round(W0 * k); th = Math.round(H0 * k);
      }
      const big = await resized(src, tw, th);
      const T0 = performance.now();
      const { look, assets } = await prepared(style, tw, th);
      /* the server's part (random textures, photo numbers) at the same time */
      const [tex, ana] = await Promise.all([
        cache.look[style].textures ? textures(style, tw, th, p) : {},
        cache.look[style].analyse ? analysed(style, p, opts.source || 'live', tw, th) : {}]);
      if (opts.shouldStop && opts.shouldStop()) throw new Error('Stopped');
      const T1 = performance.now();
      const c0 = get().compiles || 0;
      const px = await get().render(style, big, assets, look, p, onProgress, tex, ana, opts.shouldStop, !!opts.memo);
      /* where the time goes (written to the log for live previews) */
      this.lastTimes = { style, w: tw, h: th, server_ms: Math.round(T1 - T0), gpu_ms: Math.round(performance.now() - T1),
        /* fresh: every step made, none built - the true cost of this style at this size */
        fresh: (get().compiles || 0) === c0 && !get().memoHits };
      if (opts.raw) return { px, w: tw, h: th };
      if (tw === W0 && th === H0) return px;
      const c1 = document.createElement('canvas'); c1.width = tw; c1.height = th;
      c1.getContext('2d').putImageData(new ImageData(px, tw, th), 0, 0);
      const c2 = document.createElement('canvas'); c2.width = W0; c2.height = H0;
      const x2 = c2.getContext('2d'); x2.imageSmoothingQuality = 'high'; x2.drawImage(c1, 0, 0, W0, H0);
      return x2.getImageData(0, 0, W0, H0).data;
    },
    /* the same, straight to an ImageBitmap at src's size (no PNG on the way:
       the live preview puts it on screen at once); opts.keepSize: at the
       size it was made (a proxy stays small) */
    async runBitmap(src, p, opts) {
      const r = await this.run(src, p, null, Object.assign({}, opts, { raw: true }));
      const T2 = performance.now();
      let bm = await createImageBitmap(new ImageData(r.px, r.w, r.h));
      if (!(opts && opts.keepSize) && (r.w !== src.width || r.h !== src.height))
        bm = await createImageBitmap(bm, { resizeWidth: src.width, resizeHeight: src.height, resizeQuality: 'high' });
      if (this.lastTimes) this.lastTimes.copy_ms = Math.round(performance.now() - T2);
      return bm;
    },
    /* a new photo: let go of what is kept on the card */
    release() { if (fx) fx.release(); anaCache.clear(); },
    bitmap,
    /* RGBA pixels -> PNG blob / ImageBitmap */
    toBlob(px, w, h, type = 'image/png', q) {
      const c = document.createElement('canvas'); c.width = w; c.height = h;
      c.getContext('2d').putImageData(new ImageData(px, w, h), 0, 0);
      return new Promise(r => c.toBlob(r, type, q));
    },
    toBitmap(px, w, h) { return createImageBitmap(new ImageData(px, w, h)); }
  };
})();
