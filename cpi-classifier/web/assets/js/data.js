// Loading the exported particle data and combining several flights into one
// set of column arrays the page can filter and sort quickly.

export const DATA_ROOT = 'data/';

let manifest = null;
const catalogs = new Map();   // flight id -> catalog JSON (promise)
const boundaries = new Map(); // "flight/stem" -> promise of {idx: [x,y,...]}

/** Forget everything loaded, e.g. after a flight has been imported or removed. */
export function resetData() {
  manifest = null;
  catalogs.clear();
  boundaries.clear();
  metaCache = new Map();
}

export async function loadManifest() {
  if (manifest) return manifest;
  const res = await fetch(DATA_ROOT + 'manifest.json', { cache: 'no-cache' });
  if (!res.ok) throw new Error('No particle data found yet (data/manifest.json is missing). Run the exporter first.');
  manifest = await res.json();
  manifest.byId = new Map(manifest.flights.map((f) => [f.id, f]));
  return manifest;
}

export function loadCatalog(id) {
  if (!catalogs.has(id)) {
    const p = fetch(`${DATA_ROOT}flights/${encodeURIComponent(id)}/catalog.json`)
      .then((r) => { if (!r.ok) throw new Error(`Could not load particles for ${id}`); return r.json(); });
    catalogs.set(id, p);
    p.catch(() => catalogs.delete(id));
  }
  return catalogs.get(id);
}

let metaCache = new Map();
export function loadMeta(id) {
  if (!metaCache.has(id)) {
    metaCache.set(id, fetch(`${DATA_ROOT}flights/${encodeURIComponent(id)}/meta.json`).then((r) => r.json()));
  }
  return metaCache.get(id);
}

/** Outline points for every particle of one MAT file (loaded when first needed). */
export function loadBoundaries(flightId, stem) {
  const key = flightId + '/' + stem;
  if (!boundaries.has(key)) {
    const p = fetch(`${DATA_ROOT}flights/${encodeURIComponent(flightId)}/bnd/${encodeURIComponent(stem)}.json`)
      .then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
    boundaries.set(key, p);
  }
  return boundaries.get(key);
}

/**
 * Combine catalogues into one dataset D:
 *   D.n                  number of particles
 *   D.col[name]          Float64Array (NaN = missing)
 *   D.fl[g], D.file[g], D.roi[g]
 *   D.flights[k]         {id, meta, files, pixel_um}
 *   D.fields[name]       {label, unit, group}
 *   D.pid(g), D.img(g)   helpers;  D.index  Map pid -> g
 */
export async function combine(ids) {
  const cats = await Promise.all(ids.map(loadCatalog));
  const n = cats.reduce((s, c) => s + c.n, 0);
  const fields = {};
  for (const c of cats) Object.assign(fields, c.fields);
  fields.t = { label: 'Time', unit: 'UTC', group: 'time' };
  const names = Object.keys(fields);
  const col = {};
  for (const k of names) col[k] = new Float64Array(n).fill(NaN);
  const fl = new Uint16Array(n), file = new Uint16Array(n), roi = new Uint32Array(n);
  const flights = [];
  let off = 0;
  cats.forEach((c, k) => {
    const meta = manifest.byId.get(c.flight) || { id: c.flight };
    flights.push({ id: c.flight, meta, files: c.files, pixel_um: meta.pixel_um || 2.3, offset: off, n: c.n });
    for (const name of names) {
      const src = c.cols[name];
      if (!src) continue;
      const dst = col[name];
      for (let i = 0; i < c.n; i++) { const v = src[i]; if (v !== null) dst[off + i] = v; }
    }
    fl.fill(k, off, off + c.n);
    for (let i = 0; i < c.n; i++) { file[off + i] = c.cols.f[i]; roi[off + i] = c.cols.i[i]; }
    off += c.n;
  });
  const D = { n, col, fields, fl, file, roi, flights };
  D.pid = (g) => { const F = flights[fl[g]]; return `${F.id}:${F.files[file[g]]}:${roi[g]}`; };
  D.stem = (g) => flights[fl[g]].files[file[g]];
  D.img = (g) => { const F = flights[fl[g]]; return `${DATA_ROOT}flights/${encodeURIComponent(F.id)}/img/${encodeURIComponent(F.files[file[g]])}/${roi[g]}.png`; };
  D.pixel = (g) => flights[fl[g]].pixel_um;
  D.index = new Map();
  for (let g = 0; g < n; g++) D.index.set(D.pid(g), g);
  return D;
}

// ---------------------------------------------------------------- formatting
export function fmtTime(t, withDate = false) {
  if (!Number.isFinite(t)) return '–';
  const d = new Date(t * 1000);
  const iso = d.toISOString();
  return withDate ? iso.slice(0, 10) + ' ' + iso.slice(11, 19) : iso.slice(11, 19);
}
export function fmtNum(v, unit = '') {
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  const s = a >= 1000 ? Math.round(v).toLocaleString() : a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
  return unit ? `${s} ${unit}` : s;
}
export const fmtInt = (n) => Number(n || 0).toLocaleString();
