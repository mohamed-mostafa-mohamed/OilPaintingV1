/* OilStudio - Actions panel ---------------------------------------------
   Plays Envato / any Photoshop actions (.atn) inside OilStudio: every step
   is read from the .atn file and done here, with the pack's brushes (.abr)
   and patterns (.pat). Photoshop is not needed. The result comes back into
   the After tab, so Save, Snapshot, the brushes and Undo all work on it.  */
'use strict';

(() => {
  const act = { packs: [], busy: false, scanSeq: 0 };

  const pack = () => act.packs.find(p => p.id === $('#ps-pack').value);
  const post = (url, data) => api(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data)
  });

  /* ------------------------------------------------------------- packs */
  async function loadPacks() {
    const sel = $('#ps-pack'), keep = sel.value;
    try {
      const d = await api('/api/actions/packs');
      act.packs = d.packs;
      $('#ps-folder').textContent = 'Library: ' + d.folder;
    } catch (err) { toast(err.message, 'bad'); act.packs = []; }
    sel.innerHTML = '';
    if (!act.packs.length) sel.innerHTML = '<option value="">— no packs yet: press ＋ Add action —</option>';
    const cats = {};
    act.packs.forEach(p => (cats[p.category || 'Effects'] = cats[p.category || 'Effects'] || []).push(p));
    Object.keys(cats).sort().forEach(cat => {
      const og = document.createElement('optgroup'); og.label = cat;
      cats[cat].forEach(p => {
        const o = document.createElement('option');
        o.value = p.id; o.textContent = p.name + (p.error ? '  ⚠' : '');
        og.appendChild(o);
      });
      sel.appendChild(og);
    });
    if (keep && act.packs.some(p => p.id === keep)) sel.value = keep;
    showPack();
  }

  function fillActions(names, chosen) {
    const sel = $('#ps-action');
    sel.innerHTML = '';
    (names.length ? names : ['']).forEach(n => {
      const o = document.createElement('option'); o.value = n; o.textContent = n || '—'; sel.appendChild(o);
    });
    if (chosen && names.includes(chosen)) sel.value = chosen;
  }

  function showPack() {
    const p = pack();
    const img = $('#ps-preview');
    $('#ps-scan').textContent = '';
    if (!p) { fillActions([]); img.hidden = true; $('#ps-notes').textContent = ''; updateButtons(); return; }
    img.hidden = !p.has_preview;
    if (p.has_preview) img.src = '/actions/preview/' + encodeURIComponent(p.id);
    fillActions(p.actions || [], p.action);
    $('#ps-brush').value = ['subject', 'full', 'none'].includes(p.brush) ? p.brush : 'subject';
    $('#ps-size').value = p.long_side ?? 3500;
    const lines = [];
    if (p.error) lines.push('⚠ ' + p.error);
    else {
      lines.push(`Set: ${p.set}`);
      lines.push('Files: ' + p.files.join(', '));
      if (p.notes) lines.push(p.notes);
    }
    $('#ps-notes').textContent = lines.join('\n');
    updateButtons();
    if (!p.error) scan();
  }

  /* how much of the action OilStudio plays: exact / close / not yet */
  async function scan() {
    const p = pack(); if (!p) return;
    const seq = ++act.scanSeq;
    const box = $('#ps-scan');
    box.textContent = 'Reading the action…';
    try {
      const d = await post('/api/actions/scan', { pack: p.id, action: $('#ps-action').value });
      if (seq !== act.scanSeq) return;
      let t = `${d.steps} steps · ${d.percent}% run in OilStudio (${d.exact} exact, ${d.close} close to Photoshop)`;
      if (d.missing.length) {
        t += `\nNot supported yet (skipped): ` +
          d.missing.slice(0, 6).map(m => `${m.step} ${m.what}`).join(', ') + (d.missing.length > 6 ? '…' : '');
      }
      box.textContent = t;
    } catch (err) { if (seq === act.scanSeq) box.textContent = '⚠ ' + err.message; }
  }

  function body() {
    return {
      pack: $('#ps-pack').value, action: $('#ps-action').value,
      brush: $('#ps-brush').value, long_side: Number($('#ps-size').value) || 0
    };
  }

  function updateButtons() {
    const p = pack();
    const usable = !!p && !p.error;
    $('#ps-run').disabled = !usable || act.busy || state.running;
    $('#ps-remove').disabled = !p || act.busy;
  }

  /* ---------------------------------------------------------- running */
  $('#ps-reload').onclick = () => loadPacks().then(() => toast('Library reloaded', 'good'));
  $('#ps-open-folder').onclick = () =>
    api('/api/actions/open-folder', { method: 'POST' })
      .then(() => toast('Add a pack folder there, then press ↻'))
      .catch(err => toast(err.message, 'bad'));
  $('#ps-pack').onchange = showPack;
  $('#ps-action').onchange = scan;
  $('#ps-brush').onchange = updateButtons;

  function startJob(label) {
    act.busy = true; state.running = true;
    $('#btn-run').disabled = true;
    $('#btn-stop').disabled = false;
    $('#progress').hidden = false;
    $('#bar-fill').style.width = '3%';
    $('#bar-label').textContent = label;
    updateButtons();
    startPolling();                     /* the normal progress polling of app.js */
  }
  window.actionJobDone = () => { act.busy = false; updateButtons(); };

  $('#ps-run').onclick = async () => {
    if (!state.open) { toast('Open an image first.', 'bad'); return; }
    const b = body();
    try {
      await post('/api/actions/run', b);
      startJob(`Starting ${b.action || 'the action'}…`);
    } catch (err) { toast(err.message, 'bad'); }
  };

  $('#ps-batch-files').addEventListener('change', async e => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    if (!files.length) return;
    const p = pack(); if (!p) { toast('Choose an effect pack first.', 'bad'); return; }
    const b = body();
    const fd = new FormData();
    files.slice(0, 24).forEach(f => fd.append('images', f));
    fd.append('pack', b.pack); fd.append('action', b.action);
    fd.append('brush', b.brush);
    fd.append('long_side', b.long_side);
    fd.append('format', $('#export-format').value === 'png' ? 'png' : 'jpg');
    act.busy = true; updateButtons();
    $('#progress').hidden = false; $('#bar-fill').style.width = '30%';
    $('#bar-label').textContent = `Running the action on ${Math.min(files.length, 24)} photo(s)… this can take a while`;
    try {
      const res = await fetch('/api/actions/batch', { method: 'POST', body: fd });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.error || 'Batch failed');
      }
      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${p.id}_actions.zip`;
      a.click();
      URL.revokeObjectURL(a.href);
      toast('Batch finished', 'good');
    } catch (err) { toast(err.message, 'bad'); }
    $('#progress').hidden = true;
    act.busy = false; updateButtons();
  });

  /* ---------------------------------------------------- add / remove pack */
  const addForm = $('#ps-add'), addFiles = $('#ps-add-files');
  /* WebKit on macOS greys out unknown types like .atn, so filter only on Windows */
  if (!/Mac/i.test(navigator.platform || navigator.userAgent))
    addFiles.accept = '.atn,.abr,.pat,.asl,.grd,.shc,.csh,.aco,.ase,.tpl,.cube,.3dl,.look,.csp,.jsx,.js,.jsxbin,.psd,.psb,.zxp,.ccx,.zip,.jpg,.jpeg,.png,.txt,.pdf,.rtf';

  function resetAdd() {
    addForm.reset();
    $('#ps-add-list').textContent = '';
    $('#ps-add-pick').textContent = 'Choose files…';
    $('#ps-add-save').disabled = true;
  }
  $('#ps-add-toggle').onclick = () => {
    addForm.hidden = !addForm.hidden;
    if (!addForm.hidden) {
      resetAdd();
      const cats = [...new Set(act.packs.map(p => p.category || 'Effects'))].sort();
      $('#ps-cat-list').innerHTML = cats.map(c => `<option value="${c.replace(/"/g, '&quot;')}">`).join('');
    }
  };
  $('#ps-add-cancel').onclick = () => { addForm.hidden = true; resetAdd(); };

  addFiles.addEventListener('change', () => {
    const files = Array.from(addFiles.files || []);
    const hasAction = files.some(f => /\.(atn|zip)$/i.test(f.name));
    $('#ps-add-list').textContent = files.map(f => `${f.name}  (${(f.size / 1048576).toFixed(1)} MB)`).join('\n');
    $('#ps-add-pick').textContent = files.length ? `${files.length} file(s) chosen — change…` : 'Choose files…';
    $('#ps-add-save').disabled = !hasAction;
    if (files.length && !hasAction) toast('Include the .atn action file (or the Envato .zip).', 'bad');
    if (!$('#ps-add-name').value) {
      const atn = files.find(f => /\.atn$/i.test(f.name)) || files.find(f => /\.zip$/i.test(f.name));
      if (atn) $('#ps-add-name').placeholder = atn.name.replace(/\.(atn|zip)$/i, '');
    }
  });

  addForm.addEventListener('submit', async e => {
    e.preventDefault();
    const files = Array.from(addFiles.files || []);
    if (!files.length) return;
    const fd = new FormData();
    files.forEach(f => fd.append('files', f, f.name));
    fd.append('name', $('#ps-add-name').value.trim());
    fd.append('category', $('#ps-add-cat').value.trim());
    fd.append('brush', $('#ps-add-brush').value);
    const btn = $('#ps-add-save');
    btn.disabled = true; btn.textContent = 'Adding…';
    try {
      const res = await fetch('/api/actions/add-pack', { method: 'POST', body: fd });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(d.error || 'Could not add the pack.');
      addForm.hidden = true; resetAdd();
      await loadPacks();
      $('#ps-pack').value = d.pack.id; showPack();
      const sc = d.pack.scan;
      toast(`Added “${d.pack.name}”` + (sc ? ` — ${sc.percent}% of its ${sc.steps} steps run in OilStudio` : ''), 'good');
    } catch (err) { toast(err.message, 'bad'); btn.disabled = false; }
    btn.textContent = 'Add to library';
  });

  $('#ps-remove').onclick = async () => {
    const p = pack(); if (!p) return;
    if (!confirm(`Remove “${p.name}” from the OilStudio library?\nYour original Envato files are not deleted.`)) return;
    try {
      await post('/api/actions/remove-pack', { pack: p.id });
      await loadPacks();
      toast(`Removed “${p.name}”`, 'good');
    } catch (err) { toast(err.message, 'bad'); }
  };

  /* the Run button in app.js enables itself when an image opens - follow it */
  new MutationObserver(updateButtons).observe($('#btn-run'), { attributes: true, attributeFilter: ['disabled'] });

  loadPacks();
})();
