// How particle images look: live colour map, brightness and contrast, and the
// outline colour. Everything is done with one SVG filter and a CSS rule, so
// no pixels are touched in JavaScript and scrolling stays smooth.
//
// The colour map only applies to images exported in grey (the default). If a
// flight was exported with a colour map baked in, its images keep that map and
// only brightness/contrast apply.

let maps = [];
const byId = new Map();
const SVGNS = 'http://www.w3.org/2000/svg';

export async function loadColormaps() {
  try {
    const r = await fetch('config/colormaps.json');
    const j = await r.json();
    maps = j.maps.map((m) => {
      const rgb = [];
      for (let i = 0; i < m.hex.length; i += 6) rgb.push([0, 2, 4].map((o) => parseInt(m.hex.substr(i + o, 2), 16)));
      return { ...m, rgb };
    });
  } catch {
    maps = [{ id: 'gray', name: 'Grey', desc: '', rgb: Array.from({ length: 256 }, (_, i) => [i, i, i]) }];
  }
  byId.clear();
  for (const m of maps) byId.set(m.id, m);
  return maps;
}
export const colormaps = () => maps;
export const colormap = (id) => byId.get(id) || byId.get('gray');

/** CSS linear-gradient for a map (for swatches). */
export function gradient(id, dir = '90deg') {
  const m = colormap(id);
  if (!m) return '#888';
  const stops = [];
  for (let i = 0; i <= 16; i++) { const c = m.rgb[Math.min(255, i * 16)]; stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${(i / 16 * 100).toFixed(1)}%`); }
  return `linear-gradient(${dir}, ${stops.join(',')})`;
}

// -------------------------------------------------------------- SVG filter
let filterSvg = null;
function ensureFilter() {
  if (filterSvg) return;
  filterSvg = document.createElementNS(SVGNS, 'svg');
  filterSvg.setAttribute('width', '0'); filterSvg.setAttribute('height', '0');
  filterSvg.setAttribute('aria-hidden', 'true');
  filterSvg.style.position = 'absolute';
  filterSvg.innerHTML = `<filter id="cmf" color-interpolation-filters="sRGB" x="0" y="0" width="1" height="1">
    <feComponentTransfer><feFuncR type="discrete"/><feFuncG type="discrete"/><feFuncB type="discrete"/></feComponentTransfer></filter>`;
  document.body.append(filterSvg);
}
function setFilterTable(id) {
  ensureFilter();
  const m = colormap(id);
  const ch = [0, 1, 2].map((c) => m.rgb.map((v) => (v[c] / 255).toFixed(4)).join(' '));
  const f = filterSvg.querySelectorAll('feFuncR, feFuncG, feFuncB');
  f.forEach((el, c) => el.setAttribute('tableValues', ch[c]));
}

// -------------------------------------------------------------- view style
const styleEl = document.createElement('style');
styleEl.id = 'viewstyle';
document.head.append(styleEl);

/**
 * opts: {cmap, brightness, contrast, outline, bakedFlights: [ids]}
 * Particle images are recognised by their src (data/flights/...).
 */
const view = { full: '', bc: '', baked: new Set(), auto: false };
export function applyView(opts) {
  const b = +opts.brightness || 1, c = +opts.contrast || 1;
  const bc = (b !== 1 ? `brightness(${b}) ` : '') + (c !== 1 ? `contrast(${c}) ` : '');
  const live = opts.cmap && opts.cmap !== 'gray';
  if (live) setFilterTable(opts.cmap);
  let css = `:root { --outline: ${opts.outline || '#ff3b6b'}; }\n`;
  const full = (bc + (live ? 'url(#cmf)' : '')).trim();
  const changed = view.full !== full || view.bc !== bc || view.auto !== !!opts.autoContrast;
  Object.assign(view, { full, bc: bc.trim(), baked: new Set(opts.bakedFlights || []), auto: !!opts.autoContrast });
  if (changed) document.querySelectorAll('img[src^="data/flights/"]').forEach(stretch);
  if (full) css += `img[src^="data/flights/"] { filter: ${full}; }\n`;
  for (const id of opts.bakedFlights || []) {
    css += `img[src^="data/flights/${CSS.escape(encodeURIComponent(id))}/"] { filter: ${bc.trim() || 'none'}; }\n`;
  }
  styleEl.textContent = css;
}

// -------------------------------------------------------------- per-particle contrast
// "Stretch each particle": faint, washed-out images (stray light in the camera) are
// stretched so their own background is white and their darkest part black. Measured
// once per image on a small copy; applied as a CSS filter before the colour map.
const stats = new Map();     // src -> [b, c] or null
function measure(img) {
  const k = img.getAttribute('src');
  if (stats.has(k)) return stats.get(k);
  let r = null;
  try {
    const w = Math.max(1, Math.min(64, img.naturalWidth)), hh = Math.max(1, Math.min(64, img.naturalHeight));
    const cv = document.createElement('canvas'); cv.width = w; cv.height = hh;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.drawImage(img, 0, 0, w, hh);
    const px = cx.getImageData(0, 0, w, hh).data;
    const v = new Float32Array(w * hh);
    for (let i = 0; i < v.length; i++) v[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
    v.sort();
    const B = v[Math.floor(0.95 * (v.length - 1))], Dk = v[Math.floor(0.005 * (v.length - 1))];
    if (B - Dk > 0.04 && B + Dk > 0) r = [1 / (B + Dk), (B + Dk) / (B - Dk)];
  } catch { r = null; }
  stats.set(k, r);
  return r;
}
function bakedImg(img) {
  const m = /^data\/flights\/([^/]+)\//.exec(img.getAttribute('src') || '');
  return m && view.baked.has(decodeURIComponent(m[1]));
}
function stretch(img) {
  if (!view.auto) { if (img.dataset.ac) { img.style.filter = ''; delete img.dataset.ac; } return; }
  if (!img.complete || !img.naturalWidth) return;
  const r = measure(img);
  if (!r) { if (img.dataset.ac) { img.style.filter = ''; delete img.dataset.ac; } return; }
  const rest = bakedImg(img) ? view.bc : view.full;
  img.style.filter = `brightness(${r[0].toFixed(3)}) contrast(${r[1].toFixed(3)}) ${rest}`.trim();
  img.dataset.ac = '1';
}
document.addEventListener('load', (e) => {
  const t = e.target;
  if (view.auto && t && t.tagName === 'IMG' && (t.getAttribute('src') || '').startsWith('data/flights/')) stretch(t);
}, true);
