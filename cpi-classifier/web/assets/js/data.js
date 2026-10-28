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

/** A flight's particle table: catalog.json, plus catalog.bin (typed columns) when the
 *  exporter wrote one. Older exports keep their columns in the JSON itself. */
export function loadCatalog(id) {
  if (!catalogs.has(id)) {
    const base = `${DATA_ROOT}flights/${encodeURIComponent(id)}/`;
    const p = fetch(base + 'catalog.json')
      .then((r) => { if (!r.ok) throw new Error(`Could not load particles for ${id}`); return r.json(); })
      .then(async (c) => {
        if (!c.bin) return c;
        const r = await fetch(base + c.bin.file + '?v=' + encodeURIComponent(c.bin.bytes));
        if (!r.ok) throw new Error(`Could not load particles for ${id}`);
        const buf = await r.arrayBuffer();
        const T = { u32: Uint32Array, f64: Float64Array, f32: Float32Array };
        c.cols = {};
        for (const [name, [type, off]] of Object.entries(c.bin.cols)) c.cols[name] = new T[type](buf, off, c.n);
        c.typed = true;
        return c;
      });
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
export async function combine(ids, onProgress = null) {
  let done = 0;
  const cats = await Promise.all(ids.map((id) => loadCatalog(id).then((c) => { done++; onProgress?.(done, ids.length); return c; })));
  const n = cats.reduce((s, c) => s + c.n, 0);
  const fields = {};
  for (const c of cats) Object.assign(fields, c.fields);
  fields.t = { label: 'Time', unit: 'UTC', group: 'time' };
  const names = Object.keys(fields);
  const col = {};
  // times need float64; everything else is fine (and half the memory) as float32
  for (const k of names) col[k] = (k === 't' ? new Float64Array(n) : new Float32Array(n)).fill(NaN);
  const fl = new Uint16Array(n), file = new Uint32Array(n), roi = new Uint32Array(n);
  const flights = [];
  let off = 0;
  cats.forEach((c, k) => {
    const meta = manifest.byId.get(c.flight) || { id: c.flight };
    flights.push({ id: c.flight, meta, files: c.files, pixel_um: meta.pixel_um || 2.3, offset: off, n: c.n });
    for (const name of names) {
      const src = c.cols[name];
      if (!src) continue;
      const dst = col[name];
      if (c.typed) dst.set(src, off);
      else for (let i = 0; i < c.n; i++) { const v = src[i]; if (v !== null) dst[off + i] = v; }
    }
    fl.fill(k, off, off + c.n);
    if (c.typed) { file.set(c.cols.f, off); roi.set(c.cols.i, off); }
    else for (let i = 0; i < c.n; i++) { file[off + i] = c.cols.f[i]; roi[off + i] = c.cols.i[i]; }
    off += c.n;
  });
  const D = { n, col, fields, fl, file, roi, flights };
  D.pid = (g) => { const F = flights[fl[g]]; return `${F.id}:${F.files[file[g]]}:${roi[g]}`; };
  D.stem = (g) => flights[fl[g]].files[file[g]];
  D.img = (g) => { const F = flights[fl[g]]; return `${DATA_ROOT}flights/${encodeURIComponent(F.id)}/img/${encodeURIComponent(F.files[file[g]])}/${roi[g]}.png`; };
  D.pixel = (g) => flights[fl[g]].pixel_um;
  D.index = pidIndex(D);
  return D;
}

/** pid -> particle number, without making a string for every particle up front (millions of
 *  them would take seconds and hundreds of MB). Each flight's lookup is built the first
 *  time one of its particles is asked for, keyed by numbers. Behaves like a read-only Map. */
function pidIndex(D) {
  const byId = new Map(D.flights.map((F, k) => [F.id, k]));
  const perFlight = new Map();     // k -> {stems: Map stem->file index, at: Map key->g}
  const lookup = (k) => {
    let L = perFlight.get(k);
    if (!L) {
      const F = D.flights[k];
      L = { stems: new Map(F.files.map((s, i) => [s, i])), at: new Map() };
      for (let g = F.offset, end = F.offset + F.n; g < end; g++) L.at.set(D.file[g] * 4294967296 + D.roi[g], g);
      perFlight.set(k, L);
    }
    return L;
  };
  const find = (pid) => {
    if (typeof pid !== 'string') return undefined;
    const a = pid.indexOf(':'), b = pid.lastIndexOf(':');
    if (a < 0 || b <= a) return undefined;
    const k = byId.get(pid.slice(0, a));
    if (k === undefined) return undefined;
    const L = lookup(k);
    const fi = L.stems.get(pid.slice(a + 1, b));
    if (fi === undefined) return undefined;
    return L.at.get(fi * 4294967296 + Number(pid.slice(b + 1)));
  };
  return { get: find, has: (pid) => find(pid) !== undefined, get size() { return D.n; } };
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
