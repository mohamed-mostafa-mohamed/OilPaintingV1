/* OilStudio workspace ----------------------------------------------------
   Handles the settings panel, the Before/After image window with its
   drawing overlay (guiding lines and post-processing brushes), the
   Timeline, presets, export and batch processing.                        */
'use strict';

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

const state = {
  open: false, painted: false, running: false,
  tab: 'before', tool: 'pan', zoom: 0, fit: true,
  guides: [], drawing: null, version: 0,
  natural: { before: [0, 0], after: [0, 0] },
  toolOpts: {}, poll: null, ranKey: null,
  pan: [0, 0], peek: false, space: false
};

/* The live preview: the After image is drawn on the GPU (static/js/live.js)
   so every slider updates it while it moves.  Without WebGL 2 the After
   image falls back to server-rendered previews of the finishing and
   decoration settings.                                                   */
const live = new window.OilLive.Live();
const liveOn = () => live.supported && live.active;

/* ---------------------------------------------------------------- toast */
let toastTimer;
function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast ' + kind;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === 'bad' ? 5200 : 2600);
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  const type = res.headers.get('content-type') || '';
  if (!type.includes('application/json')) {
    if (!res.ok) throw new Error('Request failed (' + res.status + ')');
    return res;
  }
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

/* ----------------------------------------------------------- parameters */
function collect() {
  /* start from the defaults so settings without a control are included */
  const p = Object.assign({}, window.DEFAULTS);
  $$('.par, .par-num').forEach(el => { p[el.dataset.key] = Number(el.value); });
  $$('.par-check').forEach(el => { p[el.dataset.key] = el.checked ? 1 : 0; });
  $$('.par-choice').forEach(el => { p[el.dataset.key] = el.value; });
  $$('.par-colour').forEach(el => { p[el.dataset.key] = el.value; });
  $$('.par-text').forEach(el => { p[el.dataset.key] = el.value; });
  p.guides = state.guides;
  return p;
}

function applyParams(p) {
  Object.entries(p || {}).forEach(([k, v]) => {
    $$(`[data-key="${CSS.escape(k)}"]`).forEach(el => {
      if (el.type === 'checkbox') el.checked = !!Number(v);
      else el.value = v;
    });
  });
  syncDeps();
}

function syncDeps() {
  $$('fieldset.dep').forEach(fs => {
    const src = $(`.par-check[data-key="${CSS.escape(fs.dataset.dep)}"]`);
    fs.disabled = !(src && src.checked);
  });
  syncStyle();
}

/* Brush Strokes uses the Painting panel; Canvas Oil and Halftone have their own */
const CPU_STYLES = ['Canvas Oil', 'Halftone', 'Painted Photo', 'Impression'];
function syncStyle() {
  const sel = $('#f-paint_style');
  if (!sel) return;
  const canvasOil = sel.value === 'Canvas Oil';
  const halftone = sel.value === 'Halftone';
  const painted = sel.value === 'Painted Photo';
  const impression = sel.value === 'Impression';
  $('#g-impression').hidden = !impression;
  $('#g-canvas-oil').hidden = !canvasOil;
  $('#g-halftone').hidden = !halftone;
  $('#g-painted').hidden = !painted;
  $('#g-painting').hidden = canvasOil || halftone || painted || impression;
  $('#style-note').textContent = canvasOil
    ? 'Brush strokes over knife-cut patches, with canvas, scratches and speckles, like the Envato “Canvas Oil Painting” action. Photoshop is not needed. Set Brush Strokes to 0 for the fast, flat version.'
    : halftone
      ? 'A vintage halftone print: sepia ink, stipple grain, a woven dot grid and light scratches. Pick your own ink and paper colours below.'
      : painted
        ? 'Flat, cut-out paint patches on kraft paper with a warm film look — like a hand-painted poster.'
        : impression
          ? 'Soft impressionist paint dabs with light strokes, bristle grooves and a woven canvas, inside a white border.'
          : 'Real brush strokes, painted one at a time.';
}
document.addEventListener('change', e => {
  if (e.target && e.target.id === 'f-paint_style') syncStyle();
});

/* keep the slider and its number box in step, and update the After image
   while the slider moves ('input') and once more when it is let go
   ('change') - a dragged stroke slider previews a quick, smaller plan of
   the strokes, the full one follows on release */
function onSetting(e, moving) {
  const key = e.target.dataset && e.target.dataset.key;
  if (!key) return;
  if (moving && (e.target.classList.contains('par') || e.target.classList.contains('par-num'))) {
    $$(`[data-key="${CSS.escape(key)}"]`).forEach(el => {
      if (el !== e.target && (el.type === 'range' || el.type === 'number')) el.value = e.target.value;
    });
  }
  if (e.target.classList.contains('par-check')) syncDeps();
  /* checkboxes and lists fire both events: act on 'change' only */
  if (moving && (e.target.type === 'checkbox' || e.target.tagName === 'SELECT')) return;
  settingsChanged(key, moving && e.target.type === 'range');
}
document.addEventListener('input', e => onSetting(e, true));
document.addEventListener('change', e => onSetting(e, false));

/* settings applied on top of the painting: no repainting needed */
const isPostKey = k => window.OilLive.isPost(k);

function settingsChanged(key, dragging = false) {
  if (!state.painted) return;
  const p = collect();
  if (liveOn()) {
    live.setParams(p, { dragging, order: decoOrder() });
    if (!key || isPostKey(key) || key === 'deco_order') scheduleSync();
  } else if (!key || isPostKey(key) || key === 'deco_order') {
    scheduleSync(); fallbackPreview();
  }
  updatePreviewBadge();
}

/* the server keeps its own copy of the finishing and decoration settings,
   which Save uses: send it the latest values shortly after they change */
const postParams = () => {
  const p = {};
  Object.entries(collect()).forEach(([k, v]) => { if (isPostKey(k)) p[k] = v; });
  return p;
};
let syncTimer, syncing = null;
function scheduleSync() {
  clearTimeout(syncTimer);
  syncTimer = setTimeout(flushSync, 220);
}
async function flushSync() {
  clearTimeout(syncTimer);
  if (!state.painted) return;
  if (syncing) await syncing;
  syncing = api('/api/decoration', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ params: postParams(), deco_order: decoOrder() })
  }).then(r => { state.version = r.version; })
    .catch(err => toast(err.message, 'bad'))
    .finally(() => { syncing = null; });
  return syncing;
}

/* without WebGL 2: the server renders the After image, one at a time */
let fbBusy = false, fbAgain = false;
async function fallbackPreview() {
  if (fbBusy) { fbAgain = true; return; }
  fbBusy = true;
  try {
    const res = await fetch('/api/decoration', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: postParams(), deco_order: decoOrder(), render: true })
    });
    if (res.ok) {
      const url = URL.createObjectURL(await res.blob());
      const img = $('#img-after');
      img.onload = () => {
        URL.revokeObjectURL(url);
        state.natural.after = [img.naturalWidth, img.naturalHeight];
        if (state.tab === 'after') { if (state.fit) fitZoom(); else applyZoom(); }
      };
      img.src = url;
    }
  } catch (err) { toast(err.message, 'bad'); }
  fbBusy = false;
  if (fbAgain) { fbAgain = false; fallbackPreview(); }
}

/* "this is a preview" badge and a nudge on the Run button */
function updatePreviewBadge() {
  const badge = $('#live-badge');
  let previewing = false, text = '';
  if (state.painted && liveOn()) {
    previewing = live.previewing;
    text = live.mode === 'C'
      ? 'Draft preview — press Run to paint it in full quality'
      : 'Live preview — press Run to paint it in full quality';
  } else if (state.painted && state.ranKey) {
    previewing = window.OilLive.strokeKey(collect()) !== state.ranKey;
    text = 'Settings changed — press Run to paint them';
  }
  const lm = $('#live-mode');
  if (lm) lm.textContent = !state.painted ? '' :
    (liveOn() ? '⚡ Live sliders: GPU' : (live.supported ? '' : 'Live sliders: server (no WebGL 2)'));
  badge.hidden = !previewing;
  badge.textContent = text;
  badge.classList.toggle('busy', liveOn() && live.busy);
  $('#btn-run').classList.toggle('pulse', previewing && !state.running);
}
live.onChange(what => {
  if (what === 'render') {
    const [w, h] = live.size;
    const [ow, oh] = state.natural.after;
    state.natural.after = [w, h];
    if (state.tab === 'after' && (w !== ow || h !== oh)) { if (state.fit) fitZoom(); else applyZoom(); }
  }
  updatePreviewBadge();
});

const decoOrder = () =>
  $$('#deco-tabs .subtab').map(b => b.dataset.sub[0].toUpperCase() + b.dataset.sub.slice(1));

/* --------------------------------------------------------------- panels */
$$('.subtabs').forEach(bar => {
  bar.addEventListener('click', e => {
    const btn = e.target.closest('.subtab');
    if (!btn) return;
    const group = bar.parentElement;
    $$('.subtab', bar).forEach(b => b.classList.toggle('active', b === btn));
    $$(':scope > .sub', group).forEach(s =>
      s.classList.toggle('active', s.dataset.sub === btn.dataset.sub));
  });
});

$$('.ghead.collapsible').forEach(h => {
  h.addEventListener('click', () => {
    h.classList.toggle('collapsed');
    $('#' + h.dataset.target).hidden = h.classList.contains('collapsed');
  });
});

/* drag the Canvas / Frame / Text tabs to reorder the effects */
(() => {
  const bar = $('#deco-tabs');
  let dragged = null;
  bar.addEventListener('dragstart', e => {
    dragged = e.target.closest('.subtab');
    if (dragged) dragged.classList.add('drag');
  });
  bar.addEventListener('dragend', () => {
    if (dragged) dragged.classList.remove('drag');
    dragged = null;
    settingsChanged('deco_order');
  });
  bar.addEventListener('dragover', e => {
    e.preventDefault();
    const over = e.target.closest('.subtab');
    if (!over || !dragged || over === dragged) return;
    const rect = over.getBoundingClientRect();
    bar.insertBefore(dragged, (e.clientX - rect.left) / rect.width > 0.5
      ? over.nextSibling : over);
  });
})();

/* ------------------------------------------------------------ open file */
const IMAGE_EXT = /\.(jpe?g|png|bmp|webp|tiff?)$/i;

async function openImage(file) {
  if (!file) return;
  if (!IMAGE_EXT.test(file.name)) { toast('That is not an image OilStudio can open (JPG, PNG, BMP, WEBP, TIFF).', 'bad'); return; }
  if (state.running) { toast('Wait for the painting to finish, or press Stop.', 'bad'); return; }
  const fd = new FormData();
  fd.append('image', file);
  try {
    const r = await api('/api/open', { method: 'POST', body: fd });
    state.open = true; state.painted = false; state.guides = []; state.ranKey = null;
    live.active = false; live.canvas && (live.canvas.hidden = true);
    updatePreviewBadge();
    $('#empty-state').hidden = true;
    $('#btn-run').disabled = false;
    $('#btn-export').disabled = true;
    $('#btn-snapshot').disabled = true;
    $('#img-after').hidden = true;
    $('#stats').textContent = `${r.width} × ${r.height}`;
    setTab('before');
    refreshBefore();
    renderShots([], -1);
    toast(`Opened ${r.name}`, 'good');
  } catch (err) { toast(err.message, 'bad'); }
}

$('#file-open').addEventListener('change', async e => {
  await openImage(e.target.files[0]);
  e.target.value = '';
});

/* drag an image file from the desktop or Explorer onto the window */
(() => {
  const wrap = $('#canvas-wrap');
  let depth = 0;
  const hasFiles = e => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files');
  window.addEventListener('dragenter', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    if (++depth === 1) wrap.classList.add('drop-target');
  });
  window.addEventListener('dragover', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  window.addEventListener('dragleave', e => {
    if (!hasFiles(e)) return;
    if (--depth <= 0) { depth = 0; wrap.classList.remove('drop-target'); }
  });
  window.addEventListener('drop', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();                    /* never let the window navigate to the file */
    depth = 0; wrap.classList.remove('drop-target');
    const files = Array.from(e.dataTransfer.files || []);
    const img = files.find(f => IMAGE_EXT.test(f.name)) || files[0];
    if (files.length > 1) toast('Opening the first image — use Batch for several.', '');
    openImage(img);
  });
})();

function bust(url) { return url + '?v=' + Date.now(); }

function refreshBefore() {
  const img = $('#img-before');
  img.onload = () => {
    state.natural.before = [img.naturalWidth, img.naturalHeight];
    $('#img-before-b').src = img.src;
    if (state.fit) fitZoom(); else applyZoom();
  };
  img.src = bust('/img/before');
}
/* the painting changed on the server: load it into the After view */
async function refreshAfter() {
  try {
    const st = await api('/api/live/state');
    state.ranKey = st.ran ? window.OilLive.strokeKey(st.ran) : null;
  } catch { state.ranKey = null; }
  if (live.supported) {
    try {
      live.params = collect(); live.order = decoOrder();
      if (await live.load()) { showAfterView(); updatePreviewBadge(); return; }
    } catch (err) { console.warn('Live preview failed, using images:', err); live.active = false; }
  }
  const img = $('#img-after');
  img.onload = () => {
    /* a frame changes the proportions, so the After image carries its own */
    state.natural.after = [img.naturalWidth, img.naturalHeight];
    if (state.fit) fitZoom(); else applyZoom();
  };
  img.src = bust('/img/after');
  showAfterView();
  updatePreviewBadge();
}
/* the brushes, Undo and Redo only change the brushed pixels */
function refreshEdits() {
  if (liveOn()) live.reloadEdits(); else refreshAfter();
}
function showAfterView() { updateView(); }
function updateView() {
  const after = mainShowsAfter() && !state.peek;
  $('#img-before').hidden = after;
  $('#img-after').hidden = !after || liveOn();
  if (live.canvas) live.canvas.hidden = !after || !liveOn();
  const split = splitOn();
  $('#viewport-b').hidden = !split;
  $('#label-after').hidden = !split;
  $('#label-after').textContent = state.peek ? 'Before' : 'After';
  $('#btn-peek').hidden = !mainShowsAfter();
  updatePanCursor();
}

/* what the main pane shows: the After view, or the photograph (Before
   tab, nothing painted yet, or the Before button held down) */
const mainShowsAfter = () => state.painted && state.tab !== 'before';
function naturalSize() {
  const n = state.natural[mainShowsAfter() ? 'after' : 'before'];
  return (n && n[0]) ? n : state.natural.before;
}

/* ------------------------------------------------------------ zoom/pan */
const GAP = 18;                                   /* between the Split panes */
const splitOn = () => state.tab === 'split' && state.painted;
function fitZoom() {
  const wrap = $('#canvas-wrap');
  const [w, h] = naturalSize();
  if (!w) return;
  let z;
  if (splitOn()) {
    /* two panes of the same size side by side */
    z = Math.min((wrap.clientWidth - 60 - GAP) / (2 * w), (wrap.clientHeight - 60) / h, 3);
  } else {
    /* small paintings are allowed to fill the stage, up to 3x */
    z = Math.min((wrap.clientWidth - 60) / w, (wrap.clientHeight - 60) / h, 3);
  }
  state.zoom = z; state.fit = true; state.pan = [0, 0]; applyZoom(true);
}
function placePanes() {
  const [x, y] = state.pan;
  $('#panes').style.transform = `translate(calc(-50% + ${Math.round(x)}px), calc(-50% + ${Math.round(y)}px))`;
}
function applyZoom(keepFit) {
  const [w, h] = naturalSize();
  if (!w) return;
  const z = state.zoom || 1;
  const vw = Math.max(1, Math.round(w * z)), vh = Math.max(1, Math.round(h * z));
  const vp = $('#viewport');
  vp.style.width = vw + 'px'; vp.style.height = vh + 'px';
  $$('#viewport img, #live-canvas').forEach(i => { i.style.width = vw + 'px'; i.style.height = vh + 'px'; });
  const c = $('#overlay');
  c.width = vw; c.height = vh; c.style.width = vw + 'px'; c.style.height = vh + 'px';
  /* the photograph pane of the Split view: exactly the size of the After
     pane, whatever resolution the photo and the painting are kept at */
  const vb = $('#viewport-b'), ib = $('#img-before-b');
  vb.style.width = ib.style.width = vw + 'px';
  vb.style.height = ib.style.height = vh + 'px';
  $('#zoom-level').textContent = keepFit && state.fit ? 'Fit' : Math.round(z * 100) + '%';
  placePanes();
  drawOverlay();
}
function setZoom(z) {
  const nz = Math.max(0.08, Math.min(8, z)), k = nz / (state.zoom || 1);
  state.pan = [state.pan[0] * k, state.pan[1] * k];
  state.fit = false; state.zoom = nz; applyZoom();
}
/* zoom keeping the point under the cursor still */
function zoomAt(z, clientX, clientY) {
  const r = $('#canvas-wrap').getBoundingClientRect();
  const nz = Math.max(0.08, Math.min(8, z)), k = nz / (state.zoom || 1);
  const cx = clientX - (r.left + r.width / 2), cy = clientY - (r.top + r.height / 2);
  state.pan = [cx - (cx - state.pan[0]) * k, cy - (cy - state.pan[1]) * k];
  state.fit = false; state.zoom = nz; applyZoom();
}
$('#zoom-in').onclick = () => setZoom((state.zoom || 1) * 1.25);
$('#zoom-out').onclick = () => setZoom((state.zoom || 1) / 1.25);
$('#zoom-fit').onclick = fitZoom;
window.addEventListener('resize', () => { if (state.fit) fitZoom(); });

/* the mouse wheel zooms, Ctrl or not */
$('#canvas-wrap').addEventListener('wheel', e => {
  if (!state.open) return;
  e.preventDefault();
  zoomAt((state.zoom || 1) * Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY);
}, { passive: false });

/* drag the picture: with the Hand tool, with the middle mouse button, or
   with the Space bar held down - whatever tool is selected */
const panReady = () => state.tool === 'pan' || state.space || splitOn();
function startPan(e) {
  const wrap = $('#canvas-wrap');
  const sx = e.clientX, sy = e.clientY, [px, py] = state.pan;
  wrap.classList.add('panning');
  const move = ev => { state.pan = [px + ev.clientX - sx, py + ev.clientY - sy]; state.fit = false; placePanes(); };
  const up = () => {
    wrap.classList.remove('panning');
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    window.removeEventListener('pointercancel', up);
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
  window.addEventListener('pointercancel', up);
}
$('#canvas-wrap').addEventListener('pointerdown', e => {
  if (!state.open || e.target.closest('.peek')) return;
  if (e.button === 1 || (e.button === 0 && panReady())) {
    e.preventDefault(); e.stopPropagation();
    startPan(e);
  }
}, true);
function updatePanCursor() {
  $('#canvas-wrap').classList.toggle('can-pan', state.open && panReady());
}

/* hold to see the photograph: the button at the bottom left, or the B key */
function setPeek(on) {
  if (state.peek === on) return;
  state.peek = on;
  $('#btn-peek').classList.toggle('on', on);
  updateView();
}
(() => {
  const b = $('#btn-peek');
  b.addEventListener('pointerdown', e => { e.preventDefault(); b.setPointerCapture(e.pointerId); setPeek(true); });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(ev => b.addEventListener(ev, () => setPeek(false)));
})();

/* ---------------------------------------------------------------- tabs */
function setTab(tab) {
  const wasSplit = state.tab === 'split';
  state.tab = tab;
  $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  updateView();
  $$('.tool.before-only').forEach(b => b.style.display = tab === 'before' ? '' : 'none');
  $$('.tool.after-only').forEach(b => b.style.display = tab === 'after' ? '' : 'none');
  const t = state.tool;
  /* Split is for looking: the Hand and Zoom tools only */
  if ((tab === 'before' && window.TOOLS[t]) ||
      (tab === 'after' && (t === 'direction' || t === 'eraser')) ||
      (tab === 'split' && t !== 'pan' && t !== 'zoom')) setTool('pan');
  if (state.fit || wasSplit !== (tab === 'split')) fitZoom(); else applyZoom();
  drawOverlay();
  updateHint();
}
$$('.tab').forEach(b => b.onclick = () => {
  if (b.dataset.tab !== 'before' && !state.painted) { toast('Run the effect first.'); return; }
  setTab(b.dataset.tab);
});

/* --------------------------------------------------------------- tools */
function setTool(tool) {
  if (splitOn() && tool !== 'pan' && tool !== 'zoom') {
    toast('Split view is for comparing — switch to Before or After to use that tool.');
    tool = 'pan';
  }
  state.tool = tool;
  $$('.tool').forEach(b => b.classList.toggle('active', b.dataset.tool === tool));
  renderToolOptions();
  const c = $('#overlay');
  c.style.cursor = tool === 'pan' ? 'grab' : (tool === 'zoom' ? 'zoom-in' : 'crosshair');
  updatePanCursor();
  updateHint();
}
$('#toolbar').addEventListener('click', e => {
  const b = e.target.closest('.tool');
  if (!b || b.id === 'btn-clear-guides') return;
  setTool(b.dataset.tool);
});
$('#btn-clear-guides').onclick = () => {
  state.guides = []; drawOverlay(); toast('Guiding lines cleared'); settingsChanged('guides');
};

function updateHint() {
  const map = {
    pan: 'Hand — drag to move the picture, wheel to zoom',
    zoom: 'Zoom — click to zoom in, Alt+click to zoom out',
    direction: 'Stroke Direction — draw the lines the brush should follow, then Run',
    eraser: 'Eraser — click a guiding line to remove it',
    smudge: 'Smudge — drag to mix the paint',
    history: 'History Brush — drag to fade the effect back to the photograph',
    oil: 'Oil Brush — drag to add a stroke of paint',
    brush3d: '3D Brush — drag to add relief without changing the colours'
  };
  $('#tool-hint').textContent = !state.open ? 'Open an image to begin'
    : splitOn() ? 'Split — drag to move both pictures, wheel to zoom'
    : (map[state.tool] || '') + (state.tool !== 'pan' ? ' · Space+drag moves the picture' : '');
}

function renderToolOptions() {
  const box = $('#tool-options');
  const spec = window.TOOLS[state.tool];
  if (!spec || state.tab !== 'after') { box.hidden = true; box.innerHTML = ''; return; }
  const o = state.toolOpts[state.tool] || (state.toolOpts[state.tool] = {});
  let html = `<h4>${spec.label}</h4>`;
  (spec.params || []).forEach(([key, lo, hi, dflt]) => {
    if (o[key] === undefined) o[key] = dflt;
    html += `<div class="row"><label>${key[0].toUpperCase() + key.slice(1)}</label>
      <input type="range" class="topt" data-k="${key}" min="${lo}" max="${hi}" value="${o[key]}">
      <input type="number" class="num topt" data-k="${key}" min="${lo}" max="${hi}" value="${o[key]}"></div>`;
  });
  (spec.flags || []).forEach(([key, dflt]) => {
    if (o[key] === undefined) o[key] = dflt;
    html += `<div class="row check"><label><input type="checkbox" class="topt-check" data-k="${key}"
      ${o[key] ? 'checked' : ''}> ${key.replace(/_/g, ' ')}</label></div>`;
  });
  if (spec.colour) {
    if (o.colour === undefined) o.colour = '#c8912f';
    html += `<div class="row"><label>Colour</label>
      <input type="color" class="topt-colour" value="${o.colour}"></div>`;
  }
  box.innerHTML = html; box.hidden = false;

  $$('.topt', box).forEach(el => el.oninput = () => {
    o[el.dataset.k] = Number(el.value);
    $$(`.topt[data-k="${el.dataset.k}"]`, box).forEach(x => { x.value = el.value; });
  });
  $$('.topt-check', box).forEach(el => el.onchange = () => { o[el.dataset.k] = el.checked; });
  const col = $('.topt-colour', box);
  if (col) col.oninput = () => { o.colour = col.value; };
}

/* ------------------------------------------------------------- overlay */
const overlay = $('#overlay');

function toNorm(evt) {
  const r = overlay.getBoundingClientRect();
  return [(evt.clientX - r.left) / r.width, (evt.clientY - r.top) / r.height];
}

overlay.addEventListener('pointerdown', e => {
  if (!state.open) return;
  if (state.tool === 'pan') { startPan(e); return; }
  if (state.tool === 'zoom') { zoomAt((state.zoom || 1) * (e.altKey ? 1 / 1.3 : 1.3), e.clientX, e.clientY); return; }
  if (state.tool === 'eraser') { eraseGuide(toNorm(e)); return; }
  if (state.tab === 'before' && state.tool !== 'direction') return;
  if (state.tab === 'after' && !window.TOOLS[state.tool]) return;
  if (state.tab === 'after' && !state.painted) { toast('Run the effect first.'); return; }
  if (state.tab === 'after' && liveOn() && live.previewing) {
    toast('This is a preview of new settings — press Run, then use the brushes on the painting.');
    return;
  }

  overlay.setPointerCapture(e.pointerId);
  state.drawing = [toNorm(e)];
  drawOverlay();
});

overlay.addEventListener('pointermove', e => {
  if (!state.drawing) return;
  const p = toNorm(e);
  const last = state.drawing[state.drawing.length - 1];
  if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 0.002) return;
  state.drawing.push(p);
  drawOverlay();
});

function endStroke() {
  const path = state.drawing;
  state.drawing = null;
  if (!path || path.length < 2) { drawOverlay(); return; }
  if (state.tab === 'before') { state.guides.push(path); drawOverlay(); settingsChanged('guides'); }
  else sendBrush(path);
}
overlay.addEventListener('pointerup', endStroke);
overlay.addEventListener('pointercancel', endStroke);

function eraseGuide([x, y]) {
  const tol = 0.02;
  for (let i = state.guides.length - 1; i >= 0; i--) {
    if (state.guides[i].some(p => Math.hypot(p[0] - x, p[1] - y) < tol)) {
      state.guides.splice(i, 1); drawOverlay(); settingsChanged('guides'); return;
    }
  }
}

function drawOverlay() {
  const c = overlay, ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const paths = state.tab === 'before'
    ? state.guides.concat(state.drawing ? [state.drawing] : [])
    : (state.drawing ? [state.drawing] : []);
  if (!paths.length) return;

  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  paths.forEach(path => {
    ctx.beginPath();
    path.forEach(([x, y], i) => {
      const px = x * c.width, py = y * c.height;
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    if (state.tab === 'before') {
      ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.lineWidth = 5; ctx.stroke();
      ctx.strokeStyle = '#f0a85c'; ctx.lineWidth = 2.2; ctx.stroke();
    } else {
      const size = (state.toolOpts[state.tool] || {}).size || 40;
      ctx.strokeStyle = 'rgba(240,168,92,.45)';
      ctx.lineWidth = Math.max(2, size * (c.width / 1000));
      ctx.stroke();
    }
  });
}

async function sendBrush(path) {
  drawOverlay();
  try {
    const r = await api('/api/brush', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: state.tool, path, options: state.toolOpts[state.tool] || {} })
    });
    state.version = r.version;
    $('#btn-undo').disabled = false;
    refreshEdits();
  } catch (err) { toast(err.message, 'bad'); }
  state.drawing = null;
  drawOverlay();
}

/* ----------------------------------------------------------------- run */
/* Paint on: GPU (fast) or CPU (finest bristle detail) - remembered */
const paintOnGpu = () => live.supported && $('#paint-on').value === 'gpu' &&
  !CPU_STYLES.includes(collect().paint_style);
(() => {
  const sel = $('#paint-on');
  if (!live.supported) { $('#paint-on-wrap').hidden = true; sel.value = 'cpu'; return; }
  try { const v = localStorage.getItem('oilstudio.paintOn'); if (v) sel.value = v; } catch { /* private mode */ }
  sel.addEventListener('change', () => { try { localStorage.setItem('oilstudio.paintOn', sel.value); } catch { /* ignore */ } });
})();

/* Run on the graphics card: the server plans the strokes, the GPU paints
   them at full size, the server finishes the picture */
async function gpuRun() {
  const p = collect(), quality = $('#quality').value;
  state.running = true; state.gpuStop = false;
  $('#btn-run').disabled = true; $('#btn-stop').disabled = false;
  $('#btn-run').classList.remove('pulse');
  $('#progress').hidden = false;
  const bar = f => { $('#bar-fill').style.width = Math.round(f * 100) + '%'; };
  const label = t => { $('#bar-label').textContent = t; };
  const done = () => {
    state.running = false;
    $('#btn-run').disabled = false; $('#btn-stop').disabled = true;
    $('#progress').hidden = true;
  };
  try {
    bar(0.04); label('Planning the strokes');
    const res = await fetch('/api/gpu/plan', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: p, quality, deco_order: decoOrder() })
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || 'Planning failed');
    const plan = window.OilLive.parsePlan(await res.arrayBuffer());
    label(`Painting ${plan.n.toLocaleString()} strokes on the graphics card`);
    const out = await live.paintFull(plan, p, {
      snapshots: 8, shouldStop: () => state.gpuStop,
      onProgress: f => bar(0.15 + 0.7 * f)
    });
    if (!out) { done(); toast('Stopped — the previous painting is kept'); return; }
    bar(0.9); label('Finishing');
    const fd = new FormData();
    fd.append('canvas', out.canvas, 'canvas.png');
    fd.append('height', new Blob([out.height.buffer]), 'height.f16');
    out.shots.forEach((b, i) => fd.append('shots', b, `shot${i}.jpg`));
    fd.append('seconds', String(out.seconds));
    const r = await api('/api/gpu/result', { method: 'POST', body: fd });
    done();
    state.painted = true;
    state.version = r.version;
    $('#btn-export').disabled = false;
    $('#btn-snapshot').disabled = false;
    shotIds = '';
    const s = await api('/api/progress');
    renderShots(s.shots, s.selected);
    setTab('after');
    refreshAfter();
    $('#stats').textContent = `${r.strokes.toLocaleString()} strokes · ${r.seconds}s · GPU`;
    toast('Painting finished', 'good');
  } catch (err) {
    done();
    toast(err.message + ' — painting on the CPU instead.', 'bad');
    cpuRun();
  }
}

$('#btn-run').onclick = async () => {
  if (!state.open || state.running) return;
  if (state.painted) await flushSync();
  if (paintOnGpu()) gpuRun(); else cpuRun();
};

async function cpuRun() {
  try {
    await api('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        params: collect(), quality: $('#quality').value,
        deco_order: decoOrder(), snapshots: 8
      })
    });
    state.running = true;
    $('#btn-run').disabled = true; $('#btn-stop').disabled = false;
    $('#btn-run').classList.remove('pulse');
    $('#progress').hidden = false;
    startPolling();
  } catch (err) { toast(err.message, 'bad'); }
};

$('#btn-stop').onclick = () => {
  state.gpuStop = true;
  api('/api/stop', { method: 'POST' }).catch(() => {});
};

function startPolling() {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let s;
    try { s = await api('/api/progress'); } catch { return; }
    $('#bar-fill').style.width = Math.round(s.progress * 100) + '%';
    $('#bar-label').textContent = s.label || 'Painting…';
    renderShots(s.shots, s.selected);
    if (s.running) return;

    clearInterval(state.poll);
    state.running = false;
    $('#btn-run').disabled = false;
    $('#btn-stop').disabled = true;
    $('#progress').hidden = true;
    if (s.error) { toast(s.error, 'bad'); if (window.actionJobDone) window.actionJobDone(); return; }
    state.painted = true;
    state.version = s.version;
    $('#btn-export').disabled = false;
    $('#btn-snapshot').disabled = false;
    setTab('after');
    refreshAfter();
    if (s.engine === 'action') {
      $('#stats').textContent = `${s.action || 'Action'} · ${s.seconds}s`;
      const sk = s.skipped || [];
      if (sk.length) {
        /* steps OilStudio cannot do yet were skipped, like pressing Continue in Photoshop */
        const list = sk.slice(0, 3).map(x => `${x.step} “${x.name}”`).join(', ') + (sk.length > 3 ? '…' : '');
        toast(`Action finished — skipped ${sk.length} step${sk.length > 1 ? 's' : ''}: ${list}`, 'bad');
        console.info('Skipped action steps:', sk);
      } else toast('Action finished', 'good');
      if (window.actionJobDone) window.actionJobDone();
      return;
    }
    $('#stats').textContent = ['canvas_oil', 'halftone', 'painted', 'impression'].includes(s.engine)
      ? `${{ halftone: 'Halftone', painted: 'Painted Photo', impression: 'Impression' }[s.engine] || 'Canvas Oil'} · ${s.seconds}s`
      : `${s.strokes.toLocaleString()} strokes · ${s.seconds}s` + (s.stopped ? ' · stopped' : '');
    toast(s.stopped ? 'Stopped — the part that was painted is kept' : 'Painting finished', 'good');
  }, 420);
}

/* ------------------------------------------------------------ timeline */
let shotIds = '';
function renderShots(shots, selected) {
  const key = JSON.stringify(shots) + '|' + selected;
  if (key === shotIds) return;
  shotIds = key;
  const strip = $('#tl-strip');
  if (!shots || !shots.length) {
    strip.innerHTML = '<div class="tl-empty">Shots taken while the picture is painted appear here.</div>';
    return;
  }
  strip.innerHTML = shots.map((s, i) => `
    <div class="shot ${i === selected ? 'sel' : ''} ${s.manual ? 'manual' : ''}" data-id="${s.id}">
      <img src="/img/shot/${s.id}" alt="Stage ${i + 1}">
      ${s.pinned ? '<span class="pin">📌</span>' : ''}
    </div>`).join('');
}

$('#tl-strip').addEventListener('click', async e => {
  const el = e.target.closest('.shot');
  if (!el) return;
  try {
    const r = await api('/api/shot/select', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: el.dataset.id })
    });
    state.painted = true; state.version = r.version;
    setTab('after'); refreshAfter(); shotIds = '';
    const s = await api('/api/progress'); renderShots(s.shots, s.selected);
  } catch (err) { toast(err.message, 'bad'); }
});

$('#tl-strip').addEventListener('dblclick', async e => {
  const el = e.target.closest('.shot');
  if (!el) return;
  try {
    const r = await api('/api/shot/pin', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: el.dataset.id })
    });
    toast(r.pinned ? 'Shot pinned — it survives the next run' : 'Shot unpinned');
    shotIds = ''; const s = await api('/api/progress'); renderShots(s.shots, s.selected);
  } catch (err) { toast(err.message, 'bad'); }
});

$('#tl-delete').onclick = async () => {
  const el = $('#tl-strip .shot.sel');
  if (!el) { toast('Select a shot first.'); return; }
  try {
    await api('/api/shot/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: el.dataset.id })
    });
    shotIds = ''; const s = await api('/api/progress'); renderShots(s.shots, s.selected);
  } catch (err) { toast(err.message, 'bad'); }
};

$('#btn-snapshot').onclick = async () => {
  try {
    await api('/api/shot/snapshot', { method: 'POST' });
    shotIds = ''; const s = await api('/api/progress'); renderShots(s.shots, s.selected);
    toast('Snapshot added', 'good');
  } catch (err) { toast(err.message, 'bad'); }
};

/* --------------------------------------------------------- undo / redo */
async function doUndo(kind) {
  try {
    const r = await api('/api/' + kind, { method: 'POST' });
    state.version = r.version;
    $('#btn-redo').disabled = kind === 'redo' && false;
    refreshEdits();
  } catch (err) { toast(err.message, 'bad'); }
}
$('#btn-undo').onclick = () => { doUndo('undo'); $('#btn-redo').disabled = false; };
$('#btn-redo').onclick = () => doUndo('redo');
window.addEventListener('keydown', e => {
  if (e.target.matches('input,select,textarea')) return;
  if (e.code === 'Space') { e.preventDefault(); if (!state.space) { state.space = true; updatePanCursor(); } return; }
  if (e.key.toLowerCase() === 'b' && !e.ctrlKey && !e.metaKey && mainShowsAfter()) { setPeek(true); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); $('#btn-undo').click(); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); $('#btn-redo').click(); }
  if (e.key === 'h') setTool('pan');
  if (e.key === 'z') setTool('zoom');
});
window.addEventListener('keyup', e => {
  if (e.code === 'Space') { state.space = false; updatePanCursor(); }
  if (e.key.toLowerCase() === 'b') setPeek(false);
});
window.addEventListener('blur', () => { state.space = false; setPeek(false); updatePanCursor(); });

/* -------------------------------------------------------------- export */
$('#btn-export').onclick = async () => {
  await flushSync();
  if (liveOn() && live.previewing)
    toast('Saved the last Run. The new stroke settings are only a preview — press Run to paint them.');
  window.location = '/api/export?format=' + $('#export-format').value;
};

$('#file-batch').addEventListener('change', async e => {
  const files = Array.from(e.target.files || []);
  e.target.value = '';
  if (!files.length) return;
  const fd = new FormData();
  files.slice(0, 24).forEach(f => fd.append('images', f));
  fd.append('params', JSON.stringify(collect()));
  fd.append('deco_order', JSON.stringify(decoOrder()));
  fd.append('quality', $('#quality').value);
  fd.append('format', $('#export-format').value);
  toast(`Converting ${files.length} image(s)…`);
  try {
    const res = await fetch('/api/batch', { method: 'POST', body: fd });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.error || 'Batch processing failed');
    }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'oil_paintings.zip';
    a.click();
    URL.revokeObjectURL(a.href);
    toast('Batch finished', 'good');
  } catch (err) { toast(err.message, 'bad'); }
});

/* ------------------------------------------------------------- presets */
async function loadPresets(select) {
  const d = await api('/api/presets');
  const sel = $('#preset-list');
  sel.innerHTML =
    '<optgroup label="Built-in">' +
    d.builtin.map(n => `<option>${n}</option>`).join('') + '</optgroup>' +
    (d.user.length ? '<optgroup label="My presets">' +
      d.user.map(n => `<option>${n}</option>`).join('') + '</optgroup>' : '');
  if (select) sel.value = select;
}

$('#preset-list').onchange = async e => {
  try {
    const d = await api('/api/preset/' + encodeURIComponent(e.target.value));
    applyParams(d.params);
    settingsChanged(null);
    toast('Preset "' + e.target.value + '" loaded');
  } catch (err) { toast(err.message, 'bad'); }
};

$('#preset-save').onclick = async () => {
  const name = prompt('Name for this preset:');
  if (!name) return;
  try {
    const r = await api('/api/preset', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, params: collect() })
    });
    await loadPresets(r.name);
    toast('Preset saved', 'good');
  } catch (err) { toast(err.message, 'bad'); }
};

$('#preset-del').onclick = async () => {
  const name = $('#preset-list').value;
  if (!name || !confirm('Delete the preset "' + name + '"?')) return;
  try {
    await api('/api/preset/' + encodeURIComponent(name), { method: 'DELETE' });
    await loadPresets();
    toast('Preset deleted');
  } catch (err) { toast(err.message, 'bad'); }
};

$('#reset-all').onclick = () => {
  applyParams(window.DEFAULTS);
  state.guides = []; drawOverlay(); settingsChanged(null);
  toast('Settings reset');
};

$('#seed-random').onclick = () => {
  const el = $('#f-random_seed');
  el.value = Math.floor(Math.random() * 10000);
  settingsChanged('random_seed');
};

$('#co-seed-random').onclick = () => {
  $('#f-co_seed').value = Math.floor(Math.random() * 10000);
  settingsChanged('co_seed');
};

$('#ht-seed-random').onclick = () => {
  $('#f-ht_seed').value = Math.floor(Math.random() * 10000);
  settingsChanged('ht_seed');
};

$('#im-seed-random').onclick = () => {
  $('#f-im_seed').value = Math.floor(Math.random() * 10000);
  settingsChanged('im_seed');
};

$('#pp-seed-random').onclick = () => {
  $('#f-pp_seed').value = Math.floor(Math.random() * 10000);
  settingsChanged('pp_seed');
};

$('#edge-random').onclick = () => {
  const r = (a, b) => Math.floor(a + Math.random() * (b - a));
  applyParams({
    edge_number: r(10, 80), edge_thickness: r(8, 70), edge_max_length: r(15, 70),
    edge_smoothness: r(10, 90), edge_hair_thickness: r(15, 50),
    edge_relief: r(10, 90), edge_seed: r(0, 9999)
  });
  settingsChanged('edge_seed');
};

/* ----------------------------------------------- custom canvas and logo */
async function uploadExtra(input, kind) {
  const f = input.files[0];
  input.value = '';
  if (!f) return;
  const fd = new FormData();
  fd.append('image', f); fd.append('kind', kind);
  try {
    await api('/api/texture', { method: 'POST', body: fd });
    if (kind === 'canvas') {
      const sel = $('#f-canvas_texture');
      if (sel && [...sel.options].some(o => o.value === 'Custom')) sel.value = 'Custom';
    }
    live.customChanged && live.supported && live.customChanged(kind);
    settingsChanged(kind === 'logo' ? 'text_logo' : 'canvas_texture');
    toast(kind === 'logo' ? 'Logo loaded' : 'Canvas sample loaded', 'good');
  } catch (err) { toast(err.message, 'bad'); }
}
$('#file-texture').addEventListener('change', e => uploadExtra(e.target, 'canvas'));
$('#file-logo').addEventListener('change', e => uploadExtra(e.target, 'logo'));

/* ---------------------------------------------------- settings panel */
/* hide / show the settings panel - the picture gets the whole width */
function setPanel(hidden, remember = true) {
  $('.workspace').classList.toggle('panel-hidden', hidden);
  const b = $('#panel-toggle');
  b.textContent = hidden ? '⇤' : '⇥';
  b.title = hidden ? 'Show the settings panel (Tab)' : 'Hide the settings panel (Tab)';
  if (remember) { try { localStorage.setItem('oilstudio.panelHidden', hidden ? '1' : '0'); } catch { /* ignore */ } }
  /* refit once the panel has finished sliding */
  setTimeout(() => { if (state.fit) fitZoom(); else applyZoom(); }, 220);
}
$('#panel-toggle').onclick = e => { setPanel(!$('.workspace').classList.contains('panel-hidden')); e.currentTarget.blur(); };
window.addEventListener('keydown', e => {
  if (e.key !== 'Tab' || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.target.matches('input,select,textarea,button')) return;
  e.preventDefault();
  $('#panel-toggle').click();
});
try { if (localStorage.getItem('oilstudio.panelHidden') === '1') setPanel(true, false); } catch { /* ignore */ }

/* resize the panel by dragging its left edge; double-click resets it */
(() => {
  const MIN = 240, MAX = 680, root = document.documentElement;
  const handle = $('#panel-resizer'), panel = $('#panel');
  const maxNow = () => Math.min(MAX, Math.max(MIN, window.innerWidth - 420));
  const setW = (w, remember) => {
    w = Math.round(Math.max(MIN, Math.min(maxNow(), w)));
    root.style.setProperty('--panel-w', w + 'px');
    if (remember) { try { localStorage.setItem('oilstudio.panelWidth', String(w)); } catch { /* ignore */ } }
  };
  try { const w = +localStorage.getItem('oilstudio.panelWidth'); if (w) setW(w, false); } catch { /* ignore */ }
  handle.addEventListener('pointerdown', e => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const x0 = e.clientX, w0 = panel.getBoundingClientRect().width;
    handle.classList.add('dragging'); document.body.classList.add('resizing-panel');
    const move = ev => { setW(w0 + (x0 - ev.clientX), false); if (state.fit) fitZoom(); };
    const up = () => {
      handle.classList.remove('dragging'); document.body.classList.remove('resizing-panel');
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.removeEventListener('pointercancel', up);
      setW(panel.getBoundingClientRect().width, true);
      if (state.fit) fitZoom(); else applyZoom();
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
    handle.addEventListener('pointercancel', up);
  });
  handle.addEventListener('dblclick', () => {
    root.style.removeProperty('--panel-w');
    try { localStorage.removeItem('oilstudio.panelWidth'); } catch { /* ignore */ }
    setTimeout(() => { if (state.fit) fitZoom(); else applyZoom(); }, 30);
  });
  window.addEventListener('resize', () => {
    const w = panel.getBoundingClientRect().width;
    if (w > maxNow()) setW(w, false);
  });
})();

/* ---------------------------------------------------------------- init */
console.info('OilStudio live preview:', live.supported ? 'WebGL 2' : 'not available (server previews)');
if (live.supported) $('#viewport').insertBefore(live.canvas, $('#overlay'));
syncDeps();
setTool('pan');
setTab('before');
loadPresets().catch(() => {});
renderShots([], -1);
