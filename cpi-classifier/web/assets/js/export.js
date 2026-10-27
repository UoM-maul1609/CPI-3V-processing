// Download particle images: drawn in the browser (no server work), optionally
// to a common scale, with outline, scale bar and an information strip, and
// packed into a ZIP together with a CSV of the particles' properties.

import { loadBoundaries, fmtTime } from './data.js';
import { OVERLAYS, infoLines, niceBar } from './annot.js';

// ------------------------------------------------------------------ options
export const EXPORT_DEFAULTS = {
  scope: 'shown',            // selected | shown | mine | everyone
  folders: 'class',          // class | flat
  scale: 'fixed',            // native | fixed
  magnify: 2,                // native: screen pixels per image pixel
  frameUm: 400,              // fixed: frame width = height in µm
  umPerPx: 1.25,             // fixed: µm per output pixel
  colours: 'display',        // display (current colour map, brightness, contrast) | grey
  outline: false,
  scalebar: true,
  info: ['size', 'aspect', 'temp', 'flight'],
  classLabel: true,
  limit: 2000,
};

// ------------------------------------------------------------------ drawing
const imgCache = new Map();
function loadImage(src) {
  if (!imgCache.has(src)) {
    imgCache.set(src, new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error('Image missing: ' + src)); im.src = src; }));
    if (imgCache.size > 400) imgCache.delete(imgCache.keys().next().value);
  }
  return imgCache.get(src);
}

/**
 * Draw particle g. ctx = {D, display, S, label: {name, color}|null}. Returns {canvas, clipped}.
 */
export async function renderParticle(g, o, ctx) {
  const { D, display, S } = ctx;
  const im = await loadImage(D.img(g));
  const w = im.naturalWidth, hh = im.naturalHeight;
  const F = D.flights[D.fl[g]];
  const pix = D.pixel(g);
  const baked = F.meta.colormap && F.meta.colormap !== 'gray';

  // 1. colours, at the image's own resolution
  const src = document.createElement('canvas');
  src.width = w; src.height = hh;
  const sc = src.getContext('2d', { willReadFrequently: true });
  sc.drawImage(im, 0, 0);
  const id = sc.getImageData(0, 0, w, hh);
  const px = id.data;
  const useDisplay = o.colours === 'display';
  const b = useDisplay ? +S.brightness || 1 : 1, c = useDisplay ? +S.contrast || 1 : 1;
  const lut = useDisplay && !baked && S.cmap !== 'gray' ? display.colormap(S.cmap).rgb : null;
  const adj = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) adj[v] = Math.round(((v / 255) * b - 0.5) * c * 255 + 127.5);
  const border = [];
  for (let i = 0; i < px.length; i += 4) {
    let r = adj[px[i]], gg = adj[px[i + 1]], bb = adj[px[i + 2]];
    if (lut) { const t = lut[r]; r = t[0]; gg = t[1]; bb = t[2]; }
    px[i] = r; px[i + 1] = gg; px[i + 2] = bb; px[i + 3] = 255;
    const p = i / 4, x = p % w, y = (p / w) | 0;
    if (x === 0 || y === 0 || x === w - 1 || y === hh - 1) border.push((r << 16) | (gg << 8) | bb);
  }
  sc.putImageData(id, 0, 0);
  border.sort((a1, a2) => (a1 >> 16) + ((a1 >> 8) & 255) + (a1 & 255) - ((a2 >> 16) + ((a2 >> 8) & 255) + (a2 & 255)));
  const bg = border[border.length >> 1] ?? 0x888888;
  const bgCss = `rgb(${bg >> 16},${(bg >> 8) & 255},${bg & 255})`;

  // 2. scale and frame
  let k, W, H, ox, oy, clipped = false;
  if (o.scale === 'fixed') {
    k = pix / o.umPerPx;
    W = H = Math.max(16, Math.round(o.frameUm / o.umPerPx));
    ox = (W - w * k) / 2; oy = (H - hh * k) / 2;
    clipped = w * k > W + 0.5 || hh * k > H + 0.5;
  } else {
    k = o.magnify;
    W = Math.round(w * k); H = Math.round(hh * k);
    ox = 0; oy = 0;
  }
  const pxPerUm = k / pix;

  // 3. information strip
  const lines = infoLines(D, g, o.info, S.sizeField);
  const showLabel = o.classLabel && ctx.label;
  const nLines = lines.length + (showLabel ? 1 : 0);
  const fontPx = 12, lineH = 15;
  const stripH = nLines ? nLines * lineH + 10 : 0;
  const meas = document.createElement('canvas').getContext('2d');
  meas.font = `${fontPx}px system-ui, sans-serif`;
  const textW = Math.max(0, ...lines.map((l) => meas.measureText(l).width), showLabel ? meas.measureText(ctx.label.name).width + 18 : 0);
  const outW = Math.max(W, nLines ? Math.ceil(textW + 16) : 0);
  const imgX = Math.round((outW - W) / 2);

  const out = document.createElement('canvas');
  out.width = outW; out.height = H + stripH;
  const x = out.getContext('2d');
  x.fillStyle = bgCss; x.fillRect(0, 0, outW, H);
  x.save();
  x.beginPath(); x.rect(imgX, 0, W, H); x.clip();
  x.imageSmoothingEnabled = k < 1;
  x.drawImage(src, imgX + ox, oy, w * k, hh * k);

  // 4. outline
  if (o.outline) {
    const b2 = await loadBoundaries(F.id, D.stem(g));
    const pts = b2[String(D.roi[g])];
    if (pts && pts.length >= 6) {
      x.beginPath();
      for (let i = 0; i < pts.length; i += 2) x[i ? 'lineTo' : 'moveTo'](imgX + ox + pts[i] * k, oy + pts[i + 1] * k);
      x.closePath();
      x.lineWidth = Math.max(1, Math.min(2, k * 0.6));
      x.strokeStyle = S.outlineColor || '#ff3b6b';
      x.stroke();
    }
  }
  x.restore();

  // 5. scale bar (inside the image, bottom left)
  if (o.scalebar) {
    const L = niceBar(pxPerUm, Math.max(20, W * 0.15), Math.max(28, W * 0.4));
    const len = L * pxPerUm, bx = imgX + 6, by = H - 8;
    x.font = `bold ${Math.max(9, Math.min(12, W / 9))}px system-ui, sans-serif`;
    x.lineWidth = 3; x.strokeStyle = 'rgba(0,0,0,.75)'; x.fillStyle = '#fff';
    x.strokeRect(bx, by - 3, len, 3); x.fillRect(bx, by - 3, len, 3);
    const lab = `${L} µm`;
    x.strokeText(lab, bx, by - 6); x.fillText(lab, bx, by - 6);
  }

  // 6. strip
  if (nLines) {
    x.fillStyle = '#111820'; x.fillRect(0, H, outW, stripH);
    x.font = `${fontPx}px system-ui, sans-serif`; x.textBaseline = 'top';
    let y = H + 5;
    if (showLabel) {
      x.fillStyle = ctx.label.color || '#888'; x.fillRect(8, y + 2, 10, 10);
      x.fillStyle = '#fff'; x.font = `bold ${fontPx}px system-ui, sans-serif`;
      x.fillText(ctx.label.name, 24, y); y += lineH;
      x.font = `${fontPx}px system-ui, sans-serif`;
    }
    x.fillStyle = '#e6ecf2';
    for (const l of lines) { x.fillText(l, 8, y); y += lineH; }
  }
  return { canvas: out, clipped };
}
/** The CPI pipeline's own id for a particle (its PNG name), when the flight came from the pipeline export. */
export function sourceId(D, g) {
  const src = D.flights[D.fl[g]].meta.source;
  if (!src || !src.project) return '';
  return `${src.project}_${D.stem(g)}_roi${String(D.roi[g] + 1).padStart(7, '0')}_p01`;
}
const toBlob = (canvas) => new Promise((res) => canvas.toBlob(res, 'image/png'));

// ------------------------------------------------------------------ zip (stored, no compression: PNGs are already compressed)
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(u8) { let c = 0xffffffff; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
export class Zip {
  constructor() { this.parts = []; this.central = []; this.offset = 0; this.count = 0; }
  add(name, u8) {
    const nameB = new TextEncoder().encode(name);
    const crc = crc32(u8);
    const d = new Date();
    const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
    const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true);
    lh.setUint16(10, time, true); lh.setUint16(12, date, true); lh.setUint32(14, crc, true);
    lh.setUint32(18, u8.length, true); lh.setUint32(22, u8.length, true); lh.setUint16(26, nameB.length, true); lh.setUint16(28, 0, true);
    this.parts.push(lh.buffer, nameB, u8);
    const ch = new DataView(new ArrayBuffer(46));
    ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true);
    ch.setUint16(10, 0, true); ch.setUint16(12, time, true); ch.setUint16(14, date, true); ch.setUint32(16, crc, true);
    ch.setUint32(20, u8.length, true); ch.setUint32(24, u8.length, true); ch.setUint16(28, nameB.length, true);
    ch.setUint32(42, this.offset, true);
    this.central.push(ch.buffer, nameB);
    this.offset += 30 + nameB.length + u8.length;
    this.count++;
  }
  blob() {
    const cdSize = this.central.reduce((s, p) => s + (p.byteLength ?? p.length), 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, this.count, true); end.setUint16(10, this.count, true);
    end.setUint32(12, cdSize, true); end.setUint32(16, this.offset, true);
    return new Blob([...this.parts, ...this.central, end.buffer], { type: 'application/zip' });
  }
}
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}
const safe = (s) => String(s).replace(/[^A-Za-z0-9_.-]+/g, '_');
const csvCell = (v) => { if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) return ''; const s = String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };

// ------------------------------------------------------------------ one particle
export async function downloadOne(g, C) {
  const o = { ...EXPORT_DEFAULTS, ...(C.S.exportOpts || {}) };
  const D = C.getD();
  const m = C.labels.mine.get(D.pid(g));
  const cls = m && C.classById().get(m.label);
  const { canvas } = await renderParticle(g, o, { D, display: C.display, S: C.S, label: cls ? { name: cls.name, color: cls.color } : null });
  download(await toBlob(canvas), `${safe(D.pid(g).replace(/:/g, '_'))}.png`);
}

// ------------------------------------------------------------------ dialog
export function openExport(C) {
  const { h, S } = C;
  const D = C.getD();
  if (!D) { C.toast('Choose some flights first'); return; }
  const o = { ...EXPORT_DEFAULTS, ...(S.exportOpts || {}) };
  if (!D.flights.some((F) => F.meta.pixel_um) || !o.umPerPx) o.umPerPx = 1.25;
  const save = () => { S.exportOpts = { ...o }; C.savePrefs(); };
  const sel = C.getSelection();
  const order = C.getOrder();
  const mine = [...C.labels.mine.keys()].filter((pid) => D.index.has(pid)).map((pid) => D.index.get(pid));
  const majority = new Map();
  for (const [pid, votes] of C.labels.counts) {
    if (!D.index.has(pid)) continue;
    const top = Object.entries(votes).sort((a, b) => b[1] - a[1]);
    if (top.length && (top.length === 1 || top[0][1] > top[1][1])) majority.set(D.index.get(pid), { label: top[0][0], n: top[0][1], total: Object.values(votes).reduce((a, x) => a + x, 0) });
  }
  const scopes = [
    ['selected', `Selected particles (${sel.length})`, sel.map((p) => order[p]), !sel.length],
    ['shown', `Shown with the current filters (${order.length.toLocaleString()})`, order, false],
    ['mine', `Everything I’ve classified (${mine.length.toLocaleString()})`, mine, !mine.length],
    ['everyone', `Classified by the group, majority label (${majority.size.toLocaleString()})`, [...majority.keys()], !majority.size],
  ];
  if (scopes.find((x) => x[0] === o.scope)?.[3]) o.scope = sel.length ? 'selected' : 'shown';

  const d = h('dialog', { class: 'dlg exportdlg', 'aria-label': 'Download images' });
  const preview = h('div', { class: 'exp-preview' });
  const note = h('p', { class: 'muted small' });
  const progress = h('div', { class: 'exp-progress', hidden: true }, h('div', { class: 'progress-track' }, h('i', {})), h('span', { class: 'muted small' }));
  const goBtn = h('button', { class: 'btn primary' }, 'Download ZIP');
  let cancel = false;
  let pv = 0;

  const radio = (name, value, label, disabled) => h('label', { class: 'check' + (disabled ? ' off' : '') },
    h('input', { type: 'radio', name, value, checked: o[name] === value, disabled, onchange: () => { o[name] = value; refresh(); } }), label);
  const num = (key, attrs) => h('input', { type: 'number', value: o[key], ...attrs, oninput: (e) => { const v = parseFloat(e.target.value); if (v > 0) { o[key] = v; refresh(); } } });
  const chk = (key, label) => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: !!o[key], onchange: (e) => { o[key] = e.target.checked; refresh(); } }), label);
  const pixHint = [...new Set(D.flights.map((F) => F.pixel_um))].join(', ');

  d.append(h('div', { class: 'dlg-inner' },
    h('div', { class: 'imp-head' }, h('h2', {}, 'Download images')),
    h('div', { class: 'exp-grid' },
      h('div', { class: 'exp-opts' },
        h('fieldset', {}, h('legend', {}, 'Which particles'), scopes.map(([v, l, , dis]) => radio('scope', v, l, dis)),
          h('div', { class: 'row' }, h('span', { class: 'small' }, 'At most'), num('limit', { min: 1, step: 1, style: { width: '90px' } }), h('span', { class: 'muted small' }, 'images'))),
        h('fieldset', {}, h('legend', {}, 'Scale'),
          radio('scale', 'fixed', 'Same scale for every particle'),
          h('div', { class: 'row indent' }, h('span', { class: 'small' }, 'Frame'), num('frameUm', { min: 10, step: 10, style: { width: '80px' } }), h('span', { class: 'small' }, 'µm square at'),
            num('umPerPx', { min: 0.1, step: 0.05, style: { width: '72px' } }), h('span', { class: 'small' }, 'µm per pixel')),
          h('p', { class: 'muted small indent' }, `Every image covers the same area, so sizes compare and cropped ROIs don’t look bigger than they are. The camera’s pixel is ${pixHint} µm. Particles bigger than the frame are cut and marked in the CSV.`),
          radio('scale', 'native', 'As recorded (each ROI as cropped)'),
          h('div', { class: 'row indent' }, h('span', { class: 'small' }, 'Magnify ×'), num('magnify', { min: 1, max: 8, step: 1, style: { width: '64px' } }))),
        h('fieldset', {}, h('legend', {}, 'Colours'),
          radio('colours', 'display', `As on screen (${C.display.colormap(S.cmap).name}, brightness and contrast)`),
          radio('colours', 'grey', 'Original grey levels')),
        h('fieldset', {}, h('legend', {}, 'Draw on the image'),
          chk('outline', 'Particle outline'), chk('scalebar', 'Scale bar'), chk('classLabel', 'Class label'),
          h('div', { class: 'ovchips', style: { marginTop: '6px' } }, OVERLAYS.filter(([k]) => k !== 'scalebar').map(([k, label]) => h('button', {
            class: 'ovchip', 'aria-pressed': String(o.info.includes(k)),
            onclick: (e) => { o.info = o.info.includes(k) ? o.info.filter((x) => x !== k) : [...o.info, k]; e.currentTarget.setAttribute('aria-pressed', String(o.info.includes(k))); refresh(); } }, label))),
          h('p', { class: 'muted small' }, 'Turn everything off for clean images, e.g. for training a classifier. All the properties are in the CSV either way.')),
        h('fieldset', {}, h('legend', {}, 'In the ZIP'),
          radio('folders', 'class', 'A folder for each class'), radio('folders', 'flat', 'All in one folder'),
          h('p', { class: 'muted small' }, 'Plus particles.csv: id, flight, project, MAT file and index, class, sizes, shape, focus, time, position and temperature for each image.'))),
      h('div', { class: 'exp-side' }, h('div', { class: 'muted small' }, 'Preview'), preview, note)),
    progress,
    h('div', { class: 'dlg-actions sticky' },
      h('button', { class: 'btn ghost', onclick: () => { cancel = true; d.close(); } }, 'Cancel'),
      h('span', { class: 'grow' }),
      h('button', { class: 'btn', onclick: async () => { const gs = list(); if (gs.length) await downloadOne(gs[0], C); } }, 'Download the preview'),
      goBtn)));
  goBtn.onclick = run;
  d.addEventListener('close', () => { cancel = true; d.remove(); });
  d.addEventListener('click', (e) => { if (e.target === d) d.close(); });
  document.body.append(d);
  d.showModal();
  refresh();

  function list() { return scopes.find((x) => x[0] === o.scope)[2]; }
  function labelFor(g) {
    const pid = D.pid(g);
    const lab = o.scope === 'everyone' ? majority.get(g)?.label : C.labels.mine.get(pid)?.label;
    const c = lab && C.classById().get(lab);
    return c ? { id: c.id, name: c.name, color: c.color } : null;
  }
  async function refresh() {
    save();
    const gs = list();
    const n = Math.min(gs.length, o.limit);
    goBtn.textContent = `Download ${n.toLocaleString()} image${n === 1 ? '' : 's'} as ZIP`;
    goBtn.disabled = !n;
    const my = ++pv;
    if (!gs.length) { preview.innerHTML = ''; note.textContent = 'No particles in this choice.'; return; }
    const g = gs[0];
    try {
      const { canvas, clipped } = await renderParticle(g, o, { D, display: C.display, S, label: labelFor(g) });
      if (my !== pv) return;
      preview.innerHTML = '';
      canvas.className = 'exp-canvas';
      preview.append(canvas);
      note.textContent = `${canvas.width} × ${canvas.height} pixels` + (clipped ? ' – this particle is bigger than the frame' : '') + (gs.length > o.limit ? `. Only the first ${o.limit.toLocaleString()} of ${gs.length.toLocaleString()} will be included.` : '');
    } catch (e) { note.textContent = e.message; }
  }

  async function run() {
    const gs = list().slice(0, o.limit);
    if (!gs.length) return;
    cancel = false;
    goBtn.disabled = true;
    progress.hidden = false;
    const bar = progress.querySelector('i'), txt = progress.querySelector('span');
    const zip = new Zip();
    const fields = ['len', 'wid', 'deq', 'len_feret', 'shadow_deq', 'area', 'perimeter', 'aspect', 'area_ratio', 'round', 'orientation', 'focus', 'T_min', 'edge_touch', 'cutoff_pct', 'lat', 'lon', 'alt', 'temp', 'x0', 'y0', 'w', 'h']
      .filter((f) => D.col[f]);
    const rows = [['file', 'pid', 'source_id', 'flight', 'project', 'mat_file', 'index', 'class', 'flags', 'votes', 'time_utc', ...fields.map((f) => (D.fields[f]?.unit ? `${f}_${D.fields[f].unit.replace('µ', 'u').replace('²', '2').replace('°', 'deg').replace('%', 'pct')}` : f)), 'pixel_um', 'frame_clipped']];
    const used = new Set();
    let failed = 0;
    for (let i = 0; i < gs.length; i++) {
      if (cancel) { progress.hidden = true; return; }
      const g = gs[i];
      const pid = D.pid(g);
      const F = D.flights[D.fl[g]];
      const lab = labelFor(g);
      const folder = o.folders === 'class' ? (lab ? safe(lab.name) : 'unclassified') + '/' : '';
      let name = `${folder}${safe(F.id)}_${safe(D.stem(g))}_${D.roi[g]}.png`;
      while (used.has(name)) name = name.replace(/\.png$/, '_b.png');
      used.add(name);
      try {
        const { canvas, clipped } = await renderParticle(g, o, { D, display: C.display, S, label: lab });
        zip.add(name, new Uint8Array(await (await toBlob(canvas)).arrayBuffer()));
        const mine = C.labels.mine.get(pid);
        const votes = C.labels.counts.get(pid);
        rows.push([name, pid, sourceId(D, g), F.id, F.meta.campaign || '', D.stem(g), D.roi[g], lab ? lab.id : '',
          o.scope === 'everyone' ? '' : (mine?.flags || []).join(';'),
          votes ? Object.entries(votes).map(([k, v]) => `${k}:${v}`).join(';') : '',
          fmtTime(D.col.t[g], true), ...fields.map((f) => D.col[f][g]), D.pixel(g), o.scale === 'fixed' ? (clipped ? 1 : 0) : '']);
      } catch { failed++; }
      if (i % 10 === 0 || i === gs.length - 1) {
        bar.style.width = ((i + 1) / gs.length * 100).toFixed(1) + '%';
        txt.textContent = `Drawing ${(i + 1).toLocaleString()} of ${gs.length.toLocaleString()}`;
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    zip.add('particles.csv', new TextEncoder().encode(rows.map((r) => r.map(csvCell).join(',')).join('\n') + '\n'));
    zip.add('README.txt', new TextEncoder().encode(
      `Particle images from the Ice Crystal Classifier, ${new Date().toISOString().slice(0, 16).replace('T', ' ')} UTC\n` +
      `Particles: ${scopes.find((x) => x[0] === o.scope)[1]}\n` +
      (o.scale === 'fixed' ? `Scale: every image is a ${o.frameUm} µm square frame at ${o.umPerPx} µm per pixel.\n` : `Scale: each ROI as recorded, magnified ${o.magnify}x (see pixel_um in particles.csv).\n`) +
      `Colours: ${o.colours === 'grey' ? 'original grey levels' : `${C.display.colormap(S.cmap).name}, brightness ${S.brightness}, contrast ${S.contrast}`}\n` +
      `pid = flight:MAT file:index (0-based index into the MAT file's arrays).\n`));
    const stamp = new Date().toISOString().slice(0, 10);
    download(zip.blob(), `cpi-particles-${stamp}.zip`);
    txt.textContent = `Done: ${gs.length - failed} images` + (failed ? `, ${failed} could not be drawn` : '');
    bar.style.width = '100%';
    goBtn.disabled = false;
    C.toast(`Downloaded ${gs.length - failed} images`);
  }
}
