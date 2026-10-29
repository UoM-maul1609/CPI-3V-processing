// Admin → Machine learning: train habit classifiers on the GPU from the site's
// classifications, keep or delete each version, run a version over projects,
// flights or chosen particles, and keep or delete each set of results.
//
// The training and classifying run on the server (ml/habit_ml.py, Keras) as
// background jobs; this page starts them and shows their progress and reports.

let C = null;          // context from admin.js / app.js
let box = null;
let poll = null;
const st = { models: [], sets: [], jobs: [], summary: null, check: null,
  train: { flights: new Set(), minPeople: 1, minAgree: 66, minExamples: 20, imgSize: 160, epochs: 20, base: 'EfficientNetV2B0', classes: null, name: '', scratch: false, orient: false, scale: 'fit' },
  run: { model: '', flights: new Set(), name: '' } };

const pct = (x) => (x == null || Number.isNaN(x) ? '–' : `${(100 * x).toFixed(1)}%`);
const when = (s) => (s ? new Date(s.replace(' ', 'T') + (/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? '' : 'Z')).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');

export function leaveMl() { clearTimeout(poll); poll = null; }

export async function mlView(inner, ctx) {
  C = ctx;
  const { h } = C;
  box = h('div', { class: 'mlpage' });
  inner.append(
    h('p', { class: 'page-lede' }, 'Train a classifier on the particles people have classified, check how well it does on flights it hasn’t seen, keep the versions worth keeping, and use them to classify whole projects, flights or chosen particles. Training and classifying run on the server’s GPU, one job at a time.'),
    box);
  await refresh(true);
}

async function refresh(all = false) {
  const { api } = C;
  try {
    const calls = [api.adminGet('ml_jobs'), api.adminGet('ml_models'), api.adminGet('ml_sets')];
    if (all || !st.summary) calls.push(api.adminGet('ml_summary', { min_people: String(st.train.minPeople), min_agree: String(st.train.minAgree / 100) }));
    const [j, m, s, sum] = await Promise.all(calls);
    st.jobs = j.jobs || []; st.models = m.models || []; st.sets = s.sets || [];
    if (sum) st.summary = sum.flights || {};
  } catch (e) { C.toast(e.message, 5000); }
  render();
  clearTimeout(poll);
  if (box?.isConnected) poll = setTimeout(() => refresh(false), st.jobs.some((x) => x.status === 'running' || x.status === 'queued') ? 3000 : 15000);
}

function render() {
  const { h } = C;
  const keepOpen = [...box.querySelectorAll('details[open]')].map((d) => d.dataset.k);
  const scroll = window.scrollY;
  box.innerHTML = '';
  box.append(...[setupCard(), jobsCard(), trainCard(), modelsCard(), runCard(), setsCard()].filter(Boolean));
  box.querySelectorAll('details[data-k]').forEach((d) => { if (keepOpen.includes(d.dataset.k)) d.open = true; });
  window.scrollTo(0, scroll);
}

// ---------------------------------------------------------------- setup check
function setupCard() {
  const { h } = C;
  const c = st.check;
  const r = c?.result;
  const body = !c ? h('p', { class: 'muted small' }, 'Check that the server can run TensorFlow and sees the GPU (takes a few seconds).')
    : r && !r.error ? h('p', {}, h('span', { class: r.gpus?.length ? 'okdot' : 'warndot' }), ` Python ${r.python}, TensorFlow ${r.tensorflow}, Keras ${r.keras}. `,
      r.gpus?.length ? `GPU: ${r.gpus.join(', ')}` : h('b', {}, 'No GPU found: training would run on the CPU (very slow).'))
    : h('div', {}, h('p', {}, h('span', { class: 'warndot' }), ` TensorFlow didn’t start with ${c.python}.`),
      h('pre', { class: 'mllog' }, (r?.error || c.output || '').slice(-1500)),
      h('p', { class: 'muted small' }, 'Install it in its own virtual environment and point the site at it with ', h('code', {}, "'ml_python'"), ' in api/config.php (see deploy/DEPLOY.md, “Machine learning”).'));
  return h('section', { class: 'card mlcard' },
    h('div', { class: 'sec-h' }, h('h2', {}, 'Server'), h('span', { class: 'grow' }),
      h('button', { class: 'btn small', onclick: async (e) => {
        e.currentTarget.disabled = true; e.currentTarget.textContent = 'Checking…';
        try { st.check = await C.api.adminGet('ml_check'); } catch (x) { st.check = { output: x.message, python: '?' }; }
        render();
      } }, c ? 'Check again' : 'Check the GPU')),
    body);
}

// ---------------------------------------------------------------- jobs
function jobsCard() {
  const { h } = C;
  const jobs = st.jobs.slice(0, 6);
  if (!jobs.length) return null;
  return h('section', { class: 'card mlcard' }, h('h2', {}, 'Jobs'),
    jobs.map((j) => {
      const p = j.progress || {};
      let frac = null, what = '';
      if (j.kind === 'train') {
        if (p.epochs) { frac = ((p.epoch || 0) - (p.batch ? 1 : 0) + (p.batch ? 0.5 : 0)) / p.epochs; what = `Epoch ${p.epoch} of ${p.epochs}` + (p.val_accuracy != null ? ` · test accuracy ${pct(p.val_accuracy)}` : ''); }
        if (p.stage === 'reading') what = 'Reading the classified particles…';
        if (p.stage === 'testing') what = 'Testing on the held-back flights…';
      } else if (p.total) { frac = (p.done || 0) / p.total; what = `${C.fmtInt(p.done || 0)} of ${C.fmtInt(p.total)} particles` + (p.flight ? ` · ${p.flight}` : ''); }
      if (j.status === 'done') { frac = 1; what = j.kind === 'train' ? 'Finished: see Versions below' : 'Finished: see Classification sets below'; }
      if (j.status === 'queued') what = 'Waiting for the GPU (another job is running)';
      const title = j.kind === 'train' ? `Training (${C.fmtInt(j.n_labels || 0)} particles, ${j.flights} flight${j.flights === 1 ? '' : 's'})` : `Classifying: ${j.name || ''}`;
      return h('div', { class: 'mljob ' + j.status },
        h('div', { class: 'mljob-head' }, h('b', {}, title), h('span', { class: 'pill tiny ' + j.status }, j.status), h('span', { class: 'grow' }),
          h('span', { class: 'muted small' }, when(j.created)),
          (j.status === 'running' || j.status === 'queued') ? h('button', { class: 'btn ghost small danger', onclick: async () => {
            if (!confirm('Stop this job?')) return;
            try { await C.api.adminPost('ml_cancel', { id: j.id }); } catch (e) { C.toast(e.message, 4000); }
            refresh();
          } }, 'Stop') : null),
        frac != null ? h('div', { class: 'mlbar' }, h('i', { style: { width: Math.round(100 * Math.min(1, Math.max(0, frac))) + '%' } })) : null,
        h('div', { class: 'muted small' }, what || j.error || ''),
        h('details', { 'data-k': 'log' + j.id }, h('summary', { class: 'muted small' }, 'Log'), h('pre', { class: 'mllog' }, j.log_tail || '')));
    }));
}

// ---------------------------------------------------------------- flight picker
/** Projects with their flights, each ticked in `sel`; counts: flight id -> number to show. */
function flightPicker(sel, counts, onChange, countLabel) {
  const { h } = C;
  const flights = C.flights();
  const byProj = new Map();
  for (const f of [...flights].sort((a, b) => (a.campaign || '~').localeCompare(b.campaign || '~') || (a.date || '').localeCompare(b.date || '') || a.id.localeCompare(b.id))) {
    const k = f.campaign || 'No project';
    if (!byProj.has(k)) byProj.set(k, []);
    byProj.get(k).push(f);
  }
  const wrap = h('div', { class: 'mlpicker' });
  for (const [proj, list] of byProj) {
    const n = list.filter((f) => sel.has(f.id)).length;
    const tot = list.reduce((s, f) => s + (counts ? counts(f) : f.n || 0), 0);
    const cb = h('input', { type: 'checkbox', checked: n === list.length, onchange: (e) => { for (const f of list) { if (e.target.checked) sel.add(f.id); else sel.delete(f.id); } onChange(); } });
    cb.indeterminate = n > 0 && n < list.length;
    wrap.append(h('details', { class: 'mlproj', 'data-k': 'p:' + countLabel + proj, open: n > 0 && n < list.length },
      h('summary', {}, h('label', { class: 'check', onclick: (e) => e.stopPropagation() }, cb, h('b', {}, proj)),
        h('span', { class: 'muted small' }, ` ${n ? `${n}/${list.length} flights · ` : `${list.length} flight${list.length === 1 ? '' : 's'} · `}${C.fmtInt(tot)} ${countLabel}`)),
      h('div', { class: 'mlflights' }, list.map((f) => h('label', { class: 'check small' },
        h('input', { type: 'checkbox', checked: sel.has(f.id), onchange: (e) => { if (e.target.checked) sel.add(f.id); else sel.delete(f.id); onChange(); } }),
        h('span', {}, f.label || f.id), h('span', { class: 'muted' }, ` ${C.fmtInt(counts ? counts(f) : f.n || 0)}`))))));
  }
  return wrap;
}

// ---------------------------------------------------------------- train
function trainCard() {
  const { h } = C;
  const T = st.train;
  const scheme = C.getScheme();
  const sum = st.summary || {};
  const perFlight = (f) => Object.values(sum[f.id] || {}).reduce((a, b) => a + b, 0);
  const byClass = {};
  for (const id of T.flights) for (const [c, k] of Object.entries(sum[id] || {})) byClass[c] = (byClass[c] || 0) + k;
  const chosen = T.classes || new Set(Object.keys(byClass));
  const usable = Object.entries(byClass).filter(([c, k]) => chosen.has(c) && k >= T.minExamples);
  const nUse = usable.reduce((s, [, k]) => s + k, 0);
  const num = (key, min, max, step, width) => h('input', { type: 'number', min, max, step, value: T[key], style: { width: width || '64px' },
    onchange: (e) => { T[key] = Math.max(min, Math.min(max, +e.target.value || min)); if (key === 'minPeople' || key === 'minAgree') refresh(true); else render(); } });
  const sel = (key, opts) => h('select', { onchange: (e) => { T[key] = isNaN(+e.target.value) ? e.target.value : +e.target.value; render(); } },
    opts.map(([v, l]) => h('option', { value: v, selected: String(T[key]) === String(v) }, l)));
  const classChips = scheme.classes.filter((c) => byClass[c.id]).map((c) => {
    const k = byClass[c.id];
    const on = chosen.has(c.id);
    return h('label', { class: 'mlclass' + (k < T.minExamples ? ' few' : ''), style: { '--c': c.color }, title: k < T.minExamples ? `Fewer than ${T.minExamples}: left out` : '' },
      h('input', { type: 'checkbox', checked: on, onchange: (e) => { const s = new Set(chosen); if (e.target.checked) s.add(c.id); else s.delete(c.id); T.classes = s; render(); } }),
      c.name, h('small', {}, ` ${C.fmtInt(k)}`));
  });
  return h('section', { class: 'card mlcard' },
    h('h2', {}, 'Train a classifier'),
    h('p', { class: 'muted small' }, 'Tick the projects or flights whose classified particles to learn from. Each particle’s class is the one most people gave it. A few whole flights are held back to test the result. Along with the class it learns the riming degree and the flags (hollow, chain, hit by drops, breaking up, cut off) once people have given each at least 10 times.'),
    flightPicker(T.flights, perFlight, () => render(), 'classified'),
    h('div', { class: 'mlopts' },
      h('label', {}, 'Use particles classified by at least ', num('minPeople', 1, 10, 1, '52px'), ' people, with ', num('minAgree', 0, 100, 5, '60px'), '% agreeing'),
      h('label', {}, 'Classes need at least ', num('minExamples', 2, 5000, 5), ' examples'),
      h('label', {}, 'Picture size ', sel('imgSize', [[128, '128 px (fast)'], [160, '160 px'], [192, '192 px'], [224, '224 px (detailed, slower)']])),
      h('label', {}, 'Network ', sel('base', [['EfficientNetV2B0', 'EfficientNetV2-B0 (quick)'], ['EfficientNetV2B1', 'EfficientNetV2-B1'], ['EfficientNetV2S', 'EfficientNetV2-S (larger)']])),
      h('label', { title: 'Turn each particle so its longest direction lies left-right (near-round ones are left as they are). The model then needn’t learn every angle, which helps most when there are few classified particles.' },
        h('input', { type: 'checkbox', checked: T.orient, onchange: (e) => { T.orient = e.target.checked; } }), ' Turn particles to their long axis'),
      h('label', { title: 'Scale to fill: every particle is enlarged or shrunk so its longer side fills the picture (small ones get blurry). Instrument pixels: particles keep their pixel size (the same µm per pixel), so relative size shows in the picture; only those bigger than the picture are shrunk.' },
        'Size in the picture ', sel('scale', [['fit', 'scale each to fill'], ['native', 'keep instrument pixels (shrink big ones only)']])),
      h('label', {}, 'Up to ', num('epochs', 1, 200, 1, '60px'), ' passes over the data (stops early when it stops improving)'),
      h('label', {}, 'Name ', h('input', { value: T.name, placeholder: 'e.g. habits v1', onchange: (e) => { T.name = e.target.value; } }))),
    T.flights.size ? h('div', { class: 'mlclasses' }, h('span', { class: 'muted small' }, 'Classes:'), classChips) : null,
    h('div', { class: 'mlgo' },
      h('span', {}, T.flights.size ? `${C.fmtInt(nUse)} particles in ${usable.length} classes from ${T.flights.size} flight${T.flights.size === 1 ? '' : 's'}` : 'Tick some flights first'),
      h('span', { class: 'grow' }),
      h('button', { class: 'btn primary', disabled: nUse < 20 || usable.length < 2, onclick: async (e) => {
        e.currentTarget.disabled = true;
        try {
          await C.api.adminPost('ml_train', { flights: [...T.flights], name: T.name, options: {
            min_people: T.minPeople, min_agree: T.minAgree / 100, min_examples: T.minExamples, img_size: T.imgSize, epochs: T.epochs, base: T.base,
            classes: T.classes ? [...T.classes] : [], from_scratch: T.scratch, orient: T.orient, scale: T.scale } });
          C.toast('Training started. It runs on the server; you can leave this page.', 4000);
        } catch (x) { C.toast(x.message, 5000); }
        refresh();
      } }, 'Start training')));
}

// ---------------------------------------------------------------- versions
function modelsCard() {
  const { h } = C;
  const ms = st.models.filter((m) => m.status !== 'training');
  return h('section', { class: 'card mlcard' },
    h('h2', {}, 'Versions'),
    !ms.length ? h('p', { class: 'muted' }, 'No trained classifiers yet.') :
      h('div', { class: 'tablewrap' }, h('table', { class: 'data mltable' },
        h('thead', {}, h('tr', {}, ['Version', 'Trained', 'Classes', 'Particles (train / test)', 'Test accuracy', 'Top 2', 'Mean F1', ''].map((x) => h('th', {}, x)))),
        h('tbody', {}, ms.map((m) => h('tr', { class: m.kept ? '' : 'unsaved' },
          h('td', {}, h('b', {}, m.name || m.id), m.kept ? null : h('span', { class: 'pill tiny warn' }, 'not kept yet'), h('div', { class: 'muted small mono' }, m.id), h('div', { class: 'muted small' }, prepText(m.prep, m.img_size))),
          h('td', { class: 'small' }, when(m.created), m.by ? h('div', { class: 'muted' }, m.by) : null),
          h('td', {}, String(m.classes.length)),
          h('td', { class: 'n' }, `${C.fmtInt(m.n_train || 0)} / ${C.fmtInt(m.n_test || 0)}`),
          h('td', { class: 'n' }, h('b', {}, pct(m.accuracy))),
          h('td', { class: 'n' }, pct(m.top2)),
          h('td', { class: 'n' }, m.macro_f1 == null ? '–' : m.macro_f1.toFixed(2)),
          h('td', { class: 'mlacts' },
            h('button', { class: 'btn small', onclick: () => openReport(m.id) }, 'Report'),
            m.kept ? h('button', { class: 'btn small', onclick: () => { st.run.model = m.id; render(); box.querySelector('#mlrun')?.scrollIntoView({ behavior: 'smooth' }); } }, 'Use')
              : h('button', { class: 'btn small primary', onclick: () => keep(m) }, 'Keep'),
            h('button', { class: 'btn ghost small danger', onclick: async () => {
              if (!confirm(`Delete ${m.name || m.id}? Classification sets made with it are kept.`)) return;
              try { st.models = (await C.api.adminPost('ml_model_delete', { id: m.id })).models; } catch (e) { C.toast(e.message, 4000); }
              render();
            } }, 'Delete'))))))));
}
async function keep(m) {
  const name = prompt('Name this version (e.g. "habits v1, PICASSO + ACTIVE")', m.name || '');
  if (name === null) return;
  try { st.models = (await C.api.adminPost('ml_model_keep', { id: m.id, name })).models; C.toast('Kept'); } catch (e) { C.toast(e.message, 4000); }
  render();
}

// how a version's pictures were prepared, in words
function prepText(prep, size) {
  if (!prep) return `${size ? size + ' px, ' : ''}scaled to fill`;
  return [size ? `${size} px` : '', prep.orient ? 'turned to long axis' : '', prep.scale === 'native' ? 'instrument pixels' : 'scaled to fill'].filter(Boolean).join(', ');
}

async function openReport(id) {
  const { h } = C;
  let info;
  try { info = (await C.api.adminGet('ml_model', { id })).model; } catch (e) { C.toast(e.message, 4000); return; }
  const scheme = C.getScheme();
  const cls = new Map(scheme.classes.map((c) => [c.id, c]));
  const name = (c) => cls.get(c)?.name || c;
  const flagName = (id) => (id === 'none' ? 'Not rimed' : scheme.flags.find((f) => f.id === id)?.name || id);
  const M = info.metrics || {};
  const cm = M.confusion || [];
  const max = Math.max(1, ...cm.map((r) => Math.max(...r)));
  const exUrl = (pid) => { const [fid, stem, idx] = pid.split(':'); return `data/flights/${encodeURIComponent(fid)}/img/${encodeURIComponent(stem)}/${idx}.png`; };
  const hist = info.history || {};
  const curve = (key) => {
    const v = hist[key] || []; if (v.length < 2) return null;
    const W = 260, H = 60, lo = Math.min(...v), hi = Math.max(...v);
    const pts = v.map((y, i) => `${(i / (v.length - 1)) * W},${H - ((y - lo) / ((hi - lo) || 1)) * (H - 6) - 3}`).join(' ');
    return h('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'mlcurve', html: `<polyline fill="none" stroke="currentColor" stroke-width="2" points="${pts}"/>` });
  };
  const d = h('dialog', { class: 'dlg mlreport' });
  d.append(h('div', { class: 'dlg-inner' },
    h('div', { class: 'sec-h' }, h('h3', {}, info.state?.name || id), h('span', { class: 'grow' }), h('button', { class: 'btn ghost', onclick: () => d.close() }, 'Close')),
    h('p', { class: 'muted small' }, `${info.base} (${info.pretrained === 'imagenet' ? 'pretrained' : 'trained from scratch'}), ${info.img_size} px pictures; `
      + `${C.fmtInt(info.n_train)} particles for training, ${C.fmtInt(info.n_test)} for testing: ${info.split}. ${info.epochs_run} passes, ${Math.round((info.train_seconds || 0) / 60)} min. `
      + `Pictures: ${prepText(info.prep, info.img_size)}.`),
    h('div', { class: 'mlstats' },
      h('div', {}, h('b', {}, pct(M.accuracy)), h('span', {}, 'test accuracy')),
      h('div', {}, h('b', {}, pct(M.top2)), h('span', {}, 'right class in its top 2')),
      h('div', {}, h('b', {}, M.macro_f1 == null ? '–' : M.macro_f1.toFixed(2)), h('span', {}, 'mean F1 over classes')),
      h('div', {}, curve('val_accuracy'), h('span', {}, 'test accuracy by pass'))),
    h('details', { class: 'mlseen' }, h('summary', {}, 'What the model sees'),
      h('p', { class: 'muted small' }, 'Some training particles after preparation (contrast stretched' + (info.prep?.orient ? ', turned to their long axis' : '') + '), before the random turns, flips and contrast changes added in training.'),
      h('img', { loading: 'lazy', alt: 'Sample of prepared particles', src: C.api.adminUrl('ml_seen', { id }), onerror: (e) => { e.target.replaceWith(h('p', { class: 'muted small' }, 'No sample sheet for this version (made before it was added).')); } })),
    h('h4', {}, 'Per class'),
    h('table', { class: 'data small' }, h('thead', {}, h('tr', {}, ['Class', 'Train', 'Test', 'Precision', 'Recall', 'F1'].map((x) => h('th', {}, x)))),
      h('tbody', {}, (M.per_class || []).map((r) => h('tr', {}, h('td', {}, h('span', { class: 'dot', style: { background: cls.get(r.class)?.color || '#888' } }), ' ', name(r.class)),
        h('td', { class: 'n' }, C.fmtInt(r.train)), h('td', { class: 'n' }, C.fmtInt(r.test)),
        h('td', { class: 'n' }, pct(r.precision)), h('td', { class: 'n' }, pct(r.recall)), h('td', { class: 'n' }, r.f1 == null ? '–' : r.f1.toFixed(2)))))),
    Object.keys(info.dropped_classes || {}).length ? h('p', { class: 'muted small' }, 'Left out (too few examples): ' + Object.entries(info.dropped_classes).map(([c, k]) => `${name(c)} ${k}`).join(', ')) : null,
    h('h4', {}, 'Confusion matrix ', h('small', { class: 'muted' }, 'rows: what people said; columns: what the model said')),
    h('div', { class: 'tablewrap' }, h('table', { class: 'mlcm' },
      h('thead', {}, h('tr', {}, h('th', {}), info.classes.map((c) => h('th', { title: name(c) }, h('span', {}, name(c)))))),
      h('tbody', {}, cm.map((row, i) => h('tr', {}, h('th', {}, name(info.classes[i])),
        row.map((v, j) => h('td', { class: i === j ? 'diag' : '', style: { '--a': (v / max).toFixed(3) }, title: `${name(info.classes[i])} → ${name(info.classes[j])}: ${v}` }, v || ''))))))),
    (M.by_contrast || []).length ? [h('h4', {}, 'By image contrast ', h('small', { class: 'muted' }, 'thirds of the test particles; faint images (e.g. stray light) are in the first')),
      h('table', { class: 'data small' }, h('tbody', {}, M.by_contrast.map((b) => h('tr', {}, h('td', {}, b.band),
        h('td', { class: 'muted small' }, `${b.from == null ? '' : b.from.toFixed(2)} – ${b.to == null ? '' : b.to.toFixed(2)}`),
        h('td', { class: 'n' }, pct(b.accuracy)), h('td', { class: 'n muted' }, `${C.fmtInt(b.n)} particles`)))))] : null,
    M.riming ? [h('h4', {}, 'Riming degree ', h('small', { class: 'muted' }, `${pct(M.riming.accuracy)} right; rows: people, columns: model`)),
      h('table', { class: 'mlcm' }, h('thead', {}, h('tr', {}, h('th', {}), M.riming.levels.map((l) => h('th', {}, h('span', {}, flagName(l)))))),
        h('tbody', {}, M.riming.confusion.map((row, i) => { const mx = Math.max(1, ...M.riming.confusion.flat()); return h('tr', {}, h('th', {}, flagName(M.riming.levels[i])),
          row.map((v, j) => h('td', { class: i === j ? 'diag' : '', style: { '--a': (v / mx).toFixed(3) } }, v || ''))); })))] : null,
    (M.flags || []).length ? [h('h4', {}, 'Flags'),
      h('table', { class: 'data small' }, h('thead', {}, h('tr', {}, ['Flag', 'Test', 'Precision', 'Recall'].map((x) => h('th', {}, x)))),
        h('tbody', {}, M.flags.map((f) => h('tr', {}, h('td', {}, flagName(f.flag)), h('td', { class: 'n' }, C.fmtInt(f.test)), h('td', { class: 'n' }, pct(f.precision)), h('td', { class: 'n' }, pct(f.recall))))))] : null,
    (info.flags_not_learnt || []).length ? h('p', { class: 'muted small' }, 'Not learnt yet (fewer than 10 examples): ' + info.flags_not_learnt.map(flagName).join(', ')) : null,
    Object.keys(M.by_instrument || {}).length ? [h('h4', {}, 'By instrument'),
      h('table', { class: 'data small' }, h('tbody', {}, Object.entries(M.by_instrument).map(([k, v]) => h('tr', {}, h('td', {}, k), h('td', { class: 'n' }, pct(v.accuracy)), h('td', { class: 'n muted' }, `${C.fmtInt(v.n)} particles`)))))] : null,
    (info.mistakes || []).length ? [h('h4', {}, 'Most confident mistakes ', h('small', { class: 'muted' }, 'often worth a second look: the people may be wrong')),
      h('div', { class: 'mlmistakes' }, info.mistakes.slice(0, 40).map((x) => h('figure', {},
        h('img', { src: exUrl(x.pid), alt: '', loading: 'lazy' }),
        h('figcaption', {}, h('span', {}, name(x.true)), ' → ', h('b', {}, name(x.pred)), ` ${Math.round(100 * x.prob)}%`))))] : null,
    h('p', { class: 'muted small' }, 'Trained on: ' + (info.train_flights || []).join(', ') + (info.test_flights?.length ? '. Tested on: ' + info.test_flights.join(', ') : ''))));
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
}

// ---------------------------------------------------------------- run a model
function runCard() {
  const { h } = C;
  const R = st.run;
  const ready = st.models.filter((m) => m.status === 'trained');
  if (!ready.length) return null;
  if (!ready.some((m) => m.id === R.model)) R.model = (ready.find((m) => m.kept) || ready[0]).id;
  const n = C.flights().filter((f) => R.flights.has(f.id)).reduce((s, f) => s + (f.n || 0), 0);
  return h('section', { class: 'card mlcard', id: 'mlrun' },
    h('h2', {}, 'Classify with a model'),
    h('p', { class: 'muted small' }, 'Tick projects or flights for the model to classify. The results are kept as a classification set, which you can look through on the Classify page and download. To classify chosen particles, select them on the Classify page and use “Classify with a model”.'),
    h('div', { class: 'mlopts' },
      h('label', {}, 'Model ', h('select', { onchange: (e) => { R.model = e.target.value; } },
        ready.map((m) => h('option', { value: m.id, selected: m.id === R.model }, `${m.name || m.id}${m.kept ? '' : ' (not kept yet)'} · ${pct(m.accuracy)}`)))),
      h('label', {}, 'Name ', h('input', { value: R.name, placeholder: 'optional', onchange: (e) => { R.name = e.target.value; } }))),
    flightPicker(R.flights, null, () => render(), 'particles'),
    h('div', { class: 'mlgo' }, h('span', {}, R.flights.size ? `${C.fmtInt(n)} particles in ${R.flights.size} flight${R.flights.size === 1 ? '' : 's'}` : 'Tick some flights'),
      h('span', { class: 'grow' }),
      h('button', { class: 'btn primary', disabled: !R.flights.size, onclick: async (e) => {
        e.currentTarget.disabled = true;
        try { await C.api.adminPost('ml_predict', { model: R.model, flights: [...R.flights], name: R.name }); C.toast('Classifying started'); R.name = ''; } catch (x) { C.toast(x.message, 5000); }
        refresh();
      } }, 'Classify')));
}

// ---------------------------------------------------------------- classification sets
function setsCard() {
  const { h } = C;
  return h('section', { class: 'card mlcard' },
    h('h2', {}, 'Classification sets'),
    !st.sets.length ? h('p', { class: 'muted' }, 'None yet.') :
      h('div', { class: 'tablewrap' }, h('table', { class: 'data mltable' },
        h('thead', {}, h('tr', {}, ['Set', 'Model', 'Particles', 'Made', ''].map((x) => h('th', {}, x)))),
        h('tbody', {}, st.sets.map((s) => h('tr', {},
          h('td', {}, h('b', {}, s.name || `Set ${s.id}`), s.status !== 'ready' ? h('span', { class: 'pill tiny ' + (s.status === 'failed' ? 'err' : '') }, s.status) : null,
            h('div', { class: 'muted small' }, (s.scope?.flights || []).length ? `${s.scope.flights.length} flight${s.scope.flights.length === 1 ? '' : 's'}` : `${C.fmtInt(s.scope?.n_pids || 0)} chosen particles`)),
          h('td', { class: 'small' }, s.model_name || s.model_id),
          h('td', { class: 'n' }, C.fmtInt(s.n)),
          h('td', { class: 'small' }, when(s.created_at), h('div', { class: 'muted' }, s.created_by || '')),
          h('td', { class: 'mlacts' },
            s.status === 'ready' ? h('button', { class: 'btn small primary', onclick: () => C.showSet(s) }, 'Look through') : null,
            s.status === 'ready' ? h('a', { class: 'btn small', href: C.api.adminUrl('ml_set_csv', { id: String(s.id) }) }, 'CSV') : null,
            h('button', { class: 'btn ghost small', onclick: async () => {
              const name = prompt('Name', s.name || ''); if (name === null) return;
              try { await C.api.adminPost('ml_set_rename', { id: s.id, name }); } catch (e) { C.toast(e.message, 4000); }
              refresh();
            } }, 'Rename'),
            h('button', { class: 'btn ghost small danger', onclick: async () => {
              if (!confirm(`Delete “${s.name || 'set ' + s.id}” (${C.fmtInt(s.n)} results)? People’s classifications are not touched.`)) return;
              try { await C.api.adminPost('ml_set_delete', { id: s.id }); C.toast('Deleted'); } catch (e) { C.toast(e.message, 4000); }
              refresh();
            } }, 'Delete'))))))));
}
