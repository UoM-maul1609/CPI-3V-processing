// Text and scale bars drawn with particles: shared by the on-screen captions
// and the image export, so both say the same thing.

import { fmtNum, fmtTime } from './data.js';

export const OVERLAYS = [
  ['size', 'Size'],
  ['aspect', 'Aspect ratio'],
  ['area', 'Area ratio'],
  ['temp', 'Temperature'],
  ['alt', 'Altitude'],
  ['time', 'Time'],
  ['flight', 'Flight and project'],
  ['scalebar', 'Scale bar'],
];

const SIZE_SHORT = { len: 'L', wid: 'W', deq: 'Deq', len_feret: 'Feret', wid_feret: 'Feret W', len_ellipse: 'Ellipse', wid_ellipse: 'Ellipse W', shadow_deq: 'Dshadow' };
const minus = (s) => s.replace(/^-/, '−');

/** Lines of text for particle g, each a single string. */
export function infoLines(D, g, keys, sizeField = 'len') {
  return infoItems(D, g, keys, sizeField).map((l) => l.join('   '));
}

/** The same as lists of items: [[...shape], [...conditions], [flight]], empty lines dropped. */
export function infoItems(D, g, keys, sizeField = 'len') {
  const on = new Set(keys);
  const c = D.col;
  const fin = (k) => c[k] && Number.isFinite(c[k][g]);
  const shape = [], env = [], where = [];
  if (on.has('size')) {
    const f = fin(sizeField) ? sizeField : 'len';
    if (f === 'len' && fin('wid')) shape.push(`L ${fmtNum(c.len[g])} × W ${fmtNum(c.wid[g])} µm`);
    else if (fin(f)) shape.push(`${SIZE_SHORT[f] || f} ${fmtNum(c[f][g])} µm`);
  }
  if (on.has('aspect') && fin('aspect')) shape.push(`L/W ${fmtNum(c.aspect[g])}`);
  if (on.has('area') && fin('area_ratio')) shape.push(`Ar ${fmtNum(c.area_ratio[g])}`);
  if (on.has('temp') && fin('temp')) env.push(minus(`${c.temp[g].toFixed(1)} °C`));
  if (on.has('alt') && fin('alt')) env.push(`${Math.round(c.alt[g]).toLocaleString()} m`);
  if (on.has('time') && fin('t')) env.push(fmtTime(c.t[g]) + ' UTC');
  if (on.has('flight')) {
    const F = D.flights[D.fl[g]];
    where.push([F.meta.campaign, F.meta.label || F.id].filter(Boolean).join(' / '));
  }
  return [shape, env, where].filter((l) => l.length);
}

/** A round scale-bar length (µm) that is between minPx and maxPx long at pxPerUm. */
export function niceBar(pxPerUm, minPx = 30, maxPx = 140) {
  const options = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
  let best = options.find((l) => l * pxPerUm >= minPx) || options[options.length - 1];
  if (best * pxPerUm > maxPx) best = [...options].reverse().find((l) => l * pxPerUm <= maxPx) || best;
  return best;
}
