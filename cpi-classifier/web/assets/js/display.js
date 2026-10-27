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
export function applyView(opts) {
  const b = +opts.brightness || 1, c = +opts.contrast || 1;
  const bc = (b !== 1 ? `brightness(${b}) ` : '') + (c !== 1 ? `contrast(${c}) ` : '');
  const live = opts.cmap && opts.cmap !== 'gray';
  if (live) setFilterTable(opts.cmap);
  let css = `:root { --outline: ${opts.outline || '#ff3b6b'}; }\n`;
  const full = (bc + (live ? 'url(#cmf)' : '')).trim();
  if (full) css += `img[src^="data/flights/"] { filter: ${full}; }\n`;
  for (const id of opts.bakedFlights || []) {
    css += `img[src^="data/flights/${CSS.escape(encodeURIComponent(id))}/"] { filter: ${bc.trim() || 'none'}; }\n`;
  }
  styleEl.textContent = css;
}
