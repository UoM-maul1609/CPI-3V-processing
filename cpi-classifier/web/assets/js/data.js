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

const TYPES = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, f64: Float64Array, f32: Float32Array, q16: Uint16Array };
const INT_RANK = { u8: 1, u16: 2, u32: 3 };

/** 16-bit measurement codes (the exporter's quantise) back to numbers; 65535 = missing. */
function dequantise(q, p) {
  const out = new Float32Array(q.length);
  const span = (p.hi - p.lo) || 1;
  if (p.m === 'log') {
    const k = span / 65533;
    for (let i = 0; i < q.length; i++) { const c = q[i]; out[i] = c === 65535 ? NaN : c === 0 ? 0 : Math.exp(p.lo + (c - 1) * k); }
  } else {
    const k = span / 65534;
    for (let i = 0; i < q.length; i++) { const c = q[i]; out[i] = c === 65535 ? NaN : p.lo + c * k; }
  }
  return out;
}

/** A flight's particle table header (catalog.json). Older exports also carry all their
 *  columns: in the JSON itself, or in one catalog.bin, loaded here as c.cols. Format 3
 *  (c.colfiles) keeps one file per column, fetched only when needed (loadColumn). */
export function loadCatalog(id) {
  if (!catalogs.has(id)) {
    const base = `${DATA_ROOT}flights/${encodeURIComponent(id)}/`;
    const p = fetch(base + 'catalog.json', { cache: 'no-cache' })
      .then((r) => { if (!r.ok) throw new Error(`Could not load particles for ${id}`); return r.json(); })
      .then(async (c) => {
        c.base = base;
        if (c.colfiles) { c.cols = {}; c.typed = true; return c; }
        if (!c.bin) return c;
        const r = await fetch(base + c.bin.file + '?v=' + encodeURIComponent(c.bin.bytes));
        if (!r.ok) throw new Error(`Could not load particles for ${id}`);
        const buf = await r.arrayBuffer();
        c.cols = {};
        for (const [name, [type, off]] of Object.entries(c.bin.cols)) c.cols[name] = new TYPES[type](buf, off, c.n);
        c.typed = true;
        return c;
      });
    catalogs.set(id, p);
    p.catch(() => catalogs.delete(id));
  }
  return catalogs.get(id);
}

/** Names of the columns a catalogue has. */
const colNames = (c) => Object.keys(c.colfiles ? c.colfiles.cols : c.cols || {});

/** One column of one flight, as a typed array (null if the flight hasn't got it). */
async function loadColumn(c, name) {
  if (!c.colfiles) return c.cols?.[name] || null;
  const spec = c.colfiles.cols[name];
  if (!spec) return null;
  const r = await fetch(`${c.base}${c.colfiles.dir}/${encodeURIComponent(name)}.bin?v=${spec[1]}`);
  if (!r.ok) throw new Error(`Could not load ${name} for ${c.flight}`);
  const a = new TYPES[spec[0]](await r.arrayBuffer());
  return spec[0] === 'q16' ? dequantise(a, spec[2]) : a;
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
/** Columns every view needs, whatever the filters: which particle it is, its time, its
 *  picture size and whether it touches the edge. */
export const CORE_COLUMNS = ['f', 'i', 't', 'w', 'h', 'edge_touch'];

export async function combine(ids, onProgress = null, need = null) {
  let done = 0;
  const cats = await Promise.all(ids.map((id) => loadCatalog(id)));
  const n = cats.reduce((s, c) => s + c.n, 0);
  const fields = {};
  for (const c of cats) Object.assign(fields, c.fields);
  fields.t = { label: 'Time', unit: 'UTC', group: 'time' };
  const fl = new Uint16Array(n), file = new Uint32Array(n), roi = new Uint32Array(n);
  const flights = [];
  let off = 0;
  cats.forEach((c, k) => {
    const meta = manifest.byId.get(c.flight) || { id: c.flight };
    flights.push({ id: c.flight, meta, files: c.files, pixel_um: meta.pixel_um || 2.3, offset: off, n: c.n, cat: c, have: new Set(), loading: new Map() });
    fl.fill(k, off, off + c.n);
    off += c.n;
  });
  const col = {};
  const D = { n, col, fields, fl, file, roi, flights };
  // whole-number columns every flight stores as integers stay integers (half or a quarter
  // of the memory); the rest are float32 (float64 for times), NaN where a flight lacks them
  const newColumn = (name) => {
    let rank = 0;
    for (const F of flights) {
      const t = F.cat.colfiles?.cols?.[name]?.[0];
      if (!INT_RANK[t]) { rank = 0; break; }
      rank = Math.max(rank, INT_RANK[t]);
    }
    if (rank) return new [null, Uint8Array, Uint16Array, Uint32Array][rank](n);
    return (name === 't' ? new Float64Array(n) : new Float32Array(n)).fill(NaN);
  };

  /** Make sure columns are loaded (for every flight, or only flights ks). Older exports
   *  arrive with all their columns at once; format 3 fetches each column as it's needed. */
  D.ensure = async (names, ks = null, progress = null) => {
    const want = [...new Set(names)].filter((x) => x !== 'f' && x !== 'i');
    const todo = (ks || flights.map((_, k) => k)).filter((k) => want.some((x) => !flights[k].have.has(x)));
    let m = 0;
    let added = false;
    const one = async (k) => {
      const F = flights[k], c = F.cat;
      const list = c.colfiles ? want.filter((x) => !F.have.has(x)) : colNames(c).filter((x) => x !== 'f' && x !== 'i' && !F.have.has(x));
      await Promise.all(list.map((name) => {
        // one fetch per column and flight, even if two parts of the page ask at once
        if (!F.loading.has(name)) {
          F.loading.set(name, loadColumn(c, name).then((src) => {
            if (src) {
              if (!col[name]) col[name] = newColumn(name);
              const dst = col[name];
              if (c.typed) dst.set(src, F.offset);
              else for (let i = 0; i < c.n; i++) { const v = src[i]; if (v !== null) dst[F.offset + i] = v; }
              added = true;
            }
            F.have.add(name);
          }).finally(() => F.loading.delete(name)));
        }
        return F.loading.get(name);
      }));
      if (!c.colfiles) for (const x of want) F.have.add(x);   // everything it has came at once
      m++; progress?.(m, todo.length);
    };
    const queue = todo.slice();
    await Promise.all(Array.from({ length: Math.min(4, queue.length) }, async () => { while (queue.length) await one(queue.shift()); }));
    return added;
  };
  D.allColumns = () => Object.keys(fields);

  // which particle each row is: the file and image index columns, for every flight
  const q = flights.map((_, k) => k);
  await Promise.all(Array.from({ length: Math.min(4, q.length) }, async () => {
    while (q.length) {
      const k = q.shift(), F = flights[k], c = F.cat;
      const [f, i] = await Promise.all([loadColumn(c, 'f'), loadColumn(c, 'i')]);
      if (c.typed) { file.set(f, F.offset); roi.set(i, F.offset); }
      else for (let j = 0; j < c.n; j++) { file[F.offset + j] = f[j]; roi[F.offset + j] = i[j]; }
      F.have.add('f'); F.have.add('i');
    }
  }));
  await D.ensure([...CORE_COLUMNS, ...(need || Object.keys(fields))], null, (d, t) => { done = d; onProgress?.(done, t); });
  // older exports come with every column at once; don't keep a second copy of them cached
  for (const F of flights) if (!F.cat.colfiles) { catalogs.delete(F.id); F.cat = { ...F.cat, cols: {} }; }
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
  const perFlight = new Map();     // k -> {stems: Map stem->file index, keys, at}
  // Per flight: the particles' (file, ROI) keys sorted, with each one's row, found by binary
  // search. 12 bytes a particle, where a Map would take several times that.
  const lookup = (k) => {
    let L = perFlight.get(k);
    if (!L) {
      const F = D.flights[k];
      const m = F.n, o = F.offset;
      const keys = new Float64Array(m);
      let sorted = true;
      for (let j = 0; j < m; j++) { keys[j] = D.file[o + j] * 4294967296 + D.roi[o + j]; if (j && keys[j] < keys[j - 1]) sorted = false; }
      let at = null;                       // row offsets in key order (null: already in order)
      if (!sorted) {
        at = new Uint32Array(m);
        for (let j = 0; j < m; j++) at[j] = j;
        at.sort((x, y) => keys[x] - keys[y]);
        const ks = new Float64Array(m);
        for (let j = 0; j < m; j++) ks[j] = keys[at[j]];
        keys.set(ks);
      }
      L = { stems: new Map(F.files.map((s, i) => [s, i])), keys, at, o };
      perFlight.set(k, L);
    }
    return L;
  };
  const search = (L, key) => {
    let lo = 0, hi = L.keys.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1, v = L.keys[mid];
      if (v === key) return L.o + (L.at ? L.at[mid] : mid);
      if (v < key) lo = mid + 1; else hi = mid - 1;
    }
    return undefined;
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
    return search(L, fi * 4294967296 + Number(pid.slice(b + 1)));
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
