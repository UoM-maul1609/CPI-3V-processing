// Ice Crystal Classifier — page logic.
// All filtering, sorting and sampling runs here in the browser; the server
// only stores accounts and classifications (see api.js).

import * as api from './api.js';
import { server } from './api.js';
import { loadManifest, loadMeta, loadBoundaries, combine, fmtTime, fmtNum, fmtInt, resetData } from './data.js';
import { viewAdmin, leaveAdmin } from './admin.js';
import { RangeHist } from './rangehist.js';
import * as display from './display.js';
import { OVERLAYS, infoLines, infoItems, niceBar } from './annot.js';
import { openExport, downloadOne, sourceId } from './export.js';
import { initProjects, viewProject, projectCards, projectHref, projectIndex, flightNotes, noteHtml } from './projects.js';
let fNotes = new Map();   // flight id -> note (from the project pages)
import { initSessions, openDownload, unsavedCount, practiceMeta, uploadCard, sessionsCard, openProfile } from './session.js';

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
function h(tag, attrs = {}, ...kids) {
  const el = tag === 'svg' ? document.createElementNS('http://www.w3.org/2000/svg', 'svg') : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.setAttribute('class', v);
    else if (k === 'style' && typeof v === 'object') { for (const [sk, sv] of Object.entries(v)) { if (sk.startsWith('--')) el.style.setProperty(sk, sv); else el.style[sk] = sv; } }
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) el.append(c.nodeType ? c : String(c));
  return el;
}
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function toast(msg, ms = 2200, action = null) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.toggle('withaction', !!action);
  if (action) t.append(h('button', { class: 'toastbtn', onclick: () => { t.classList.remove('show'); action.run(); } }, action.label));
  t.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove('show'), ms);
}
function lsGet(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } }
function lsSet(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } }
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ settings
const DEFAULTS = {
  flights: null,            // selected flight ids (null = not chosen yet)
  view: 'grid',             // grid | single
  scale: 'true',            // true | fit
  zoom: 1.2,                // screen px per µm ×2.5 (true scale) — see pxPerUm()
  tile: 130,                // tile size in fit mode
  outline: false,
  cmap: 'gray',             // live colour map (grey exports only)
  brightness: 1,
  contrast: 1,
  outlineColor: '#ff3b6b',
  overlays: [],             // shown with each particle: size, aspect, area, temp, alt, time, flight, scalebar
  inspector: true,
  show: 'todo',             // which particles: todo | all | mine | none | disagree | cls:<id>
  hideEdge: true,
  order: 'random',
  seed: 1,
  sizeField: 'len',
  ranges: {},               // dim key -> [min|null, max|null]
  extraDims: [],            // extra filter fields the person added
  page: 120,
  theme: null,
};
const S = Object.assign({}, DEFAULTS, lsGet('cpi.prefs', {}));
const savePrefs = () => lsSet('cpi.prefs', S);
if (S.boost) { S.contrast = 1.7; delete S.boost; }
if (typeof S.caption === 'string') { S.overlays = { size: ['size'], shape: ['aspect', 'area'], time: ['time'] }[S.caption] || []; delete S.caption; }
const session = { n: 0, start: 0, last: 0, active: 0 };   // particles classified this visit

// ------------------------------------------------------------------ globals
let scheme = { version: '', classes: [], flags: [] };
let manifest = null;
let D = null;                 // combined dataset for the selected flights
let Dkey = '';
const labels = { mine: new Map(), counts: new Map() };   // pid -> {label, flags[]} / {label: n}
let order = [];               // filtered + ordered particle indices
let shown = 0;                // how many of `order` are on screen (grid)
let focusPos = 0;             // position in `order`
let anchorPos = -1;
const selection = new Set();  // positions in `order`
const armed = new Set();      // flags waiting for the next class key
const undoStack = [];         // [{changes:[{pid, before, after}], desc, t}]
const redoStack = [];
let hoverG = -1;
let classById = new Map(), classByKey = new Map(), flagByKey = new Map();

// ------------------------------------------------------------------ theme
function applyTheme() {
  if (S.theme) document.documentElement.dataset.theme = S.theme;
  else delete document.documentElement.dataset.theme;
}
$('#themeBtn').addEventListener('click', () => {
  const dark = S.theme ? S.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  S.theme = dark ? 'light' : 'dark';
  savePrefs(); applyTheme();
});
applyTheme();

// ------------------------------------------------------------------ account
function renderUserBox() {
  $('#adminNav').hidden = !server.user?.is_admin;
  const box = $('#userBox');
  box.innerHTML = '';
  if (server.demo) {
    box.append(h('span', { class: 'pill demo', title: 'A demonstration site: classifications stay in your browser and are not saved.' }, 'Demonstration'));
    return;
  }
  if (server.reachable === false) {
    box.append(h('span', { class: 'pill', title: 'The classification server is not reachable, so labels cannot be saved. Practice mode still works.' }, 'Practice only'));
    return;
  }
  if (!server.user) {
    box.append(server.setup
      ? h('button', { class: 'btn primary small', onclick: () => openAuth({ setup: true }) }, 'Set up the administrator account')
      : h('button', { class: 'btn primary small', onclick: () => openAuth() }, 'Sign in'));
    return;
  }
  const u = server.user;
  const initials = u.display_name.split(/\s+/).map((w) => w[0]).slice(0, 2).join('').toUpperCase();
  const btn = h('button', { class: 'userchip', 'aria-haspopup': 'menu', 'aria-expanded': 'false', title: 'Your account',
    onclick: (e) => { e.stopPropagation(); toggleUserMenu(btn); } },
    h('span', { class: 'avatar' }, initials), h('span', { class: 'uname' }, u.display_name), h('span', { class: 'caret', 'aria-hidden': 'true' }, '▾'));
  box.append(btn);
}
let userMenu = null;
function toggleUserMenu(anchor) {
  if (userMenu?.isConnected) { userMenu.remove(); anchor.setAttribute('aria-expanded', 'false'); return; }
  const u = server.user;
  userMenu = h('div', { class: 'popover usermenu', role: 'menu', onclick: (e) => e.stopPropagation() },
    h('div', { class: 'um-head' }, h('b', {}, u.display_name), h('span', { class: 'muted small' }, u.username + (u.is_admin ? ', administrator' : '')),
      u.expertise ? h('span', { class: 'muted small' }, u.expertise) : null),
    h('button', { class: 'um-item', role: 'menuitem', onclick: () => { userMenu.remove(); openProfile(); } }, 'Your details'),
    h('button', { class: 'um-item', role: 'menuitem', onclick: () => { userMenu.remove(); openChangePassword(false); } }, 'Change password'),
    u.is_admin ? h('a', { class: 'um-item', role: 'menuitem', href: '#/admin', onclick: () => userMenu.remove() }, 'Admin: people, classes and flights') : null,
    h('button', { class: 'um-item', role: 'menuitem', onclick: signOut }, 'Sign out'));
  document.body.append(userMenu);
  const r = anchor.getBoundingClientRect();
  userMenu.style.top = (r.bottom + 6) + 'px';
  userMenu.style.left = Math.max(12, r.right - userMenu.offsetWidth) + 'px';
  anchor.setAttribute('aria-expanded', 'true');
}
document.addEventListener('click', () => { if (userMenu?.isConnected) userMenu.remove(); });
async function signOut() {
  userMenu?.remove();
  await api.flush();
  await api.logout().catch(() => {});
  server.user = null;
  undoStack.length = 0; redoStack.length = 0;
  await refreshLabels();
  renderUserBox();
  if (location.hash.startsWith('#/admin')) location.hash = '#/classify'; else route();
  toast('Signed out. You’re in practice mode: nothing you classify is saved.', 3500);
}

// ------------------------------------------------------------------ practice mode
// Without an account (or on a demonstration site) people can still classify:
// the labels stay in this browser only and never reach the database.
const PKEY = 'cpi.practice';
const practice = {
  get on() { return !server.user; },
  load() { return lsGet(PKEY, {}); },
  save(items) {
    const st = this.load();
    const now = Math.round(Date.now() / 1000);
    for (const it of items) { if (it.label) st[it.pid] = [it.label, (it.flags || []).join(','), now]; else delete st[it.pid]; }
    lsSet(PKEY, st);
    const m = practiceMeta.load();
    if (!m.started) { m.started = now; practiceMeta.save(m); }
  },
  clear() { lsSet(PKEY, {}); const m = practiceMeta.load(); m.started = 0; m.saved = 0; m.savedCount = 0; practiceMeta.save(m); },
  count() { return Object.keys(this.load()).length; },
};
function saveItems(items) {
  if (practice.on) { practice.save(items); remindToSave(); }
  else api.queueSave(items, scheme.version);
}
/** Every so often in practice mode, suggest saving the work to a file. */
let lastNudge = 0;
function remindToSave() {
  const n = unsavedCount();
  const step = Math.floor(n / 25);
  if (n < 25 || step <= lastNudge) { if (step < lastNudge) lastNudge = step; return; }
  lastNudge = step;
  // after the "N particles → class" message has had its moment
  setTimeout(() => toast(`${fmtInt(unsavedCount())} practice classifications not saved anywhere yet.`, 8000, { label: 'Save to a file', run: openDownload }), 1600);
}
initProjects({
  h, $, toast, fmtInt, server, api, sparkline: (tl) => sparkline(tl),
  visibleFlights: () => visibleFlights(),
  classifyFlights: (ids) => { S.flights = ids; savePrefs(); location.hash = '#/classify'; },
  fetchStats: () => (server.reachable && !server.demo ? api.flush().then(() => api.fetchStats()) : Promise.resolve(null)),
  projectsChanged: async () => { fNotes = await flightNotes(true); },
});
initSessions({
  h, $, toast, server, api, fmtInt, practice,
  getScheme: () => scheme, getData: () => D,
  onSaved: () => { lastNudge = 0; updateSaveState(); if (location.hash.startsWith('#/progress')) route(); },
  onProfile: () => renderUserBox(),
});
/** After signing in: offer to keep what was done in practice mode. */
function offerPracticeKeep() {
  const n = practice.count();
  if (!n || !server.user || server.user.must_change_pw) return;
  const d = h('dialog', { class: 'dlg' });
  d.append(h('div', { class: 'dlg-inner' },
    h('h3', {}, `Keep your ${n} practice classification${n === 1 ? '' : 's'}?`),
    h('p', { class: 'dlg-why' }, 'You classified these before signing in. Save them under your name, or throw them away and start fresh.'),
    h('div', { class: 'dlg-actions' },
      h('button', { class: 'btn ghost', onclick: () => { practice.clear(); d.close(); toast('Practice classifications thrown away'); } }, 'Throw them away'),
      h('button', { class: 'btn primary', autofocus: true, onclick: async () => {
        const st = practice.load();
        const items = Object.entries(st).map(([pid, [label, flags]]) => ({ pid, label, flags: flags ? flags.split(',') : [] }));
        api.queueSave(items, scheme.version);
        await api.flush();
        practice.clear();
        d.close();
        await refreshLabels(); route();
        toast(`Saved ${items.length} classification${items.length === 1 ? '' : 's'} under your name`);
      } }, 'Save them'))));
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
}

// ------------------------------------------------------------------ change password
function openChangePassword(forced) {
  const d = h('dialog', { class: 'dlg pwdlg' });
  const err = h('p', { class: 'formerr', role: 'alert' });
  const mk = (name, label, ac) => {
    const i = h('input', { name, type: 'password', autocomplete: ac, required: true });
    return [h('label', { class: 'fld' }, h('span', {}, label), h('span', { class: 'pwrow' }, i,
      h('button', { type: 'button', class: 'btn ghost small', 'aria-label': 'Show password', onclick: (e) => { i.type = i.type === 'password' ? 'text' : 'password'; e.currentTarget.textContent = i.type === 'password' ? 'Show' : 'Hide'; } }, 'Show'))), i];
  };
  const [curL, cur] = mk('current', forced ? 'Temporary password' : 'Current password', 'current-password');
  const [newL, nw] = mk('new', 'New password', 'new-password');
  const [repL, rep2] = mk('repeat', 'New password again', 'new-password');
  const meter = h('div', { class: 'pwmeter' }, h('i', {}), h('span', { class: 'muted small' }, 'At least 10 characters. A few unrelated words together works well.'));
  nw.addEventListener('input', () => {
    const v = nw.value; let sc = Math.min(4, Math.floor(v.length / 4)) + (/[^a-z]/i.test(v) ? 1 : 0);
    if (v.length < 10) sc = Math.min(sc, 1);
    meter.querySelector('i').style.width = (sc / 5 * 100) + '%';
    meter.querySelector('i').dataset.level = sc;
  });
  const form = h('form', { class: 'dlg-inner', onsubmit: async (e) => {
    e.preventDefault(); err.textContent = '';
    if (nw.value !== rep2.value) { err.textContent = 'The two new passwords don’t match'; return; }
    try {
      await api.changePassword(cur.value, nw.value);
      d.close();
      renderUserBox();
      toast('Password changed');
      if (forced) { await refreshLabels(); offerPracticeKeep(); route(); }
    } catch (x) { err.textContent = x.message; }
  } },
  h('h3', {}, forced ? `Welcome, ${server.user.display_name.split(' ')[0]}. Choose your own password` : 'Change your password'),
  h('p', { class: 'dlg-why' }, forced ? 'You signed in with a temporary password. Choose one that only you know. You’ll use it from now on.' : 'Your classifications are kept; only the password changes.'),
  curL, newL, meter, repL, err,
  h('div', { class: 'dlg-actions' },
    forced ? h('button', { type: 'button', class: 'btn ghost', onclick: () => { d.close(); signOut(); } }, 'Sign out')
      : h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'),
    h('button', { class: 'btn primary' }, 'Save password')));
  d.append(form);
  d.addEventListener('cancel', (e) => { if (forced) e.preventDefault(); });
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
  (forced ? cur : cur).focus();
}

const dlg = $('#authDlg');
const form = $('#authForm');
function setAuthTab(tab) {
  form.classList.toggle('register', tab === 'register');
  $('#authDlg .tabs').hidden = !!form.dataset.setup || !server.allowRegistration;
  form.classList.toggle('needs-code', server.needsCode);
  $$('.tabs button', form).forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  $('#authSubmit').textContent = tab === 'register' ? 'Create account' : 'Sign in';
  form.password.autocomplete = tab === 'register' ? 'new-password' : 'current-password';
  $('#authErr').textContent = '';
}
function openAuth(opts = {}) {
  if (typeof opts === 'string') opts = { why: opts };
  if (server.demo) { toast('This is a demonstration site: classifications stay in your browser.'); return; }
  if (server.reachable === false) { toast('The server is not reachable, so classifications cannot be saved right now.'); return; }
  form.dataset.setup = opts.setup ? '1' : '';
  $('#authWhy').textContent = opts.setup
    ? 'This is the first account on the site, so it becomes the administrator. Use one of the admin usernames from api/config.php. You can then add other people from the Admin page.'
    : opts.why || (server.allowRegistration ? 'Sign in so your classifications are saved under your name.' : 'Sign in so your classifications are saved under your name. No account yet? Ask the administrator: they’ll give you a username and a temporary password.');
  $('#regTab').hidden = !server.allowRegistration;
  setAuthTab(opts.setup ? 'register' : 'login');
  if (opts.setup) $('#authSubmit').textContent = 'Create the administrator account';
  dlg.showModal();
  form.username.focus();
}
$$('.tabs button', form).forEach((b) => b.addEventListener('click', () => setAuthTab(b.dataset.tab)));
$('#authCancel').addEventListener('click', () => dlg.close());
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const reg = form.classList.contains('register');
  const f = Object.fromEntries(new FormData(form));
  $('#authErr').textContent = '';
  $('#authSubmit').disabled = true;
  try {
    if (reg) await api.register(f); else await api.login(f.username, f.password);
    dlg.close();
    form.reset();
    server.setup = false;
    renderUserBox();
    if (server.user.must_change_pw) {
      if (location.hash.startsWith('#/admin') && !server.user.is_admin) location.hash = '#/classify';
      openChangePassword(true); return;
    }
    api.restoreQueue();
    await refreshLabels();
    toast(`Signed in as ${server.user.display_name}`);
    if (location.hash.startsWith('#/admin') && !server.user.is_admin) location.hash = '#/classify'; else route();
    offerPracticeKeep();
  } catch (err) {
    $('#authErr').textContent = err.message;
  } finally {
    $('#authSubmit').disabled = false;
  }
});

// ------------------------------------------------------------------ labels
async function refreshLabels() {
  if (!D) return;
  let d = { mine: {}, counts: {} };
  if (server.reachable && !server.demo) {
    try { d = await api.fetchLabels(D.flights.map((f) => f.id)); } catch (e) { console.warn(e); }
  }
  labels.counts = new Map(Object.entries(d.counts || {}));
  if (practice.on) {
    labels.mine = new Map();
    for (const [pid, [label, flags]] of Object.entries(practice.load())) {
      if (D.index.has(pid)) setMine(pid, { label, flags: flags ? flags.split(',') : [] });
    }
  } else {
    labels.mine = new Map(Object.entries(d.mine || {}).map(([pid, [label, flags]]) => [pid, { label, flags: flags ? flags.split(',') : [] }]));
  }
}
function setMine(pid, val) {
  const prev = labels.mine.get(pid);
  const c = Object.assign({}, labels.counts.get(pid) || {});
  if (prev) { c[prev.label] = (c[prev.label] || 1) - 1; if (c[prev.label] <= 0) delete c[prev.label]; }
  if (val) { labels.mine.set(pid, val); c[val.label] = (c[val.label] || 0) + 1; } else labels.mine.delete(pid);
  if (Object.keys(c).length) labels.counts.set(pid, c); else labels.counts.delete(pid);
}
const totalVotes = (pid) => { const c = labels.counts.get(pid); let n = 0; if (c) for (const k in c) n += c[k]; return n; };

// ------------------------------------------------------------------ router
const views = { classify: viewClassify, flights: viewFlights, progress: viewProgress, guide: viewGuide,
  project: (root) => viewProject(root, decodeURIComponent(location.hash.replace(/^#\/?project\/?/, '').split('?')[0])),
  admin: (root) => viewAdmin(root, adminCtx) };
const adminCtx = {
  h, $, $$, toast, fmtInt, server, api, display,
  selectOnly: (id) => { S.flights = [id]; savePrefs(); },
  getScheme: () => scheme,
  setScheme: (sch) => setScheme(sch),
  onDataChanged: async () => {
    resetData();
    try { manifest = await loadManifest(); } catch { manifest = null; }
    Dkey = ''; D = null;
    applyDisplay();
  },
};
const exportCtx = {
  h, toast, labels, S, display, savePrefs,
  getD: () => D, getOrder: () => order, getSelection: () => [...selection].sort((a, b) => a - b), classById: () => classById,
};
const DL_ICON = '<svg viewBox="0 0 24 24"><path d="M11 3h2v9.6l3.3-3.3 1.4 1.4L12 16.4 6.3 10.7l1.4-1.4L11 12.6V3zM4 18h16v2H4z"/></svg>';
/** Map project name -> flights (sorted by project, then date). Flights with no project come last. */
function groupByProject(flights) {
  const sorted = [...flights].sort((a, b) => (a.campaign ? 0 : 1) - (b.campaign ? 0 : 1) || (a.campaign || '').localeCompare(b.campaign || '') || (a.date || '').localeCompare(b.date || '') || a.id.localeCompare(b.id));
  const m = new Map();
  for (const f of sorted) { const k = f.campaign || ''; if (!m.has(k)) m.set(k, []); m.get(k).push(f); }
  return m;
}
/** Flights offered to people: hidden ones only to administrators. */
function visibleFlights() {
  if (!manifest) return [];
  return manifest.flights.filter((f) => !f.hidden || server.user?.is_admin);
}
async function route() {
  const name = (location.hash.replace(/^#\/?/, '').split(/[?/]/)[0]) || 'classify';
  const fn = views[name] || viewClassify;
  const navName = name === 'project' ? 'flights' : name;
  $$('.mainnav a').forEach((a) => { if (a.dataset.route === navName) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  document.removeEventListener('keydown', onKey);
  if (name !== 'admin') leaveAdmin();
  await fn($('#view'));
}
window.addEventListener('hashchange', route);

// ============================================================== CLASSIFY VIEW
let dims = [];            // [{key, field, rh}]
let classifyEl = null;

function selectedFlights() {
  const ids = visibleFlights().map((f) => f.id);
  if (S.flights === null) S.flights = ids.length <= 3 ? ids : ids.slice(0, 1);
  S.flights = S.flights.filter((id) => ids.includes(id));
  return S.flights;
}

async function ensureData() {
  const ids = selectedFlights();
  const key = ids.join('|');
  if (key === Dkey && D) return;
  Dkey = key;
  D = ids.length ? await combine(ids) : null;
  if (D) await refreshLabels();
}

async function viewClassify(root) {
  root.className = '';
  root.innerHTML = '';
  if (!manifest || !visibleFlights().length) return emptyNoData(root);
  root.append(h('div', { class: 'empty' }, h('p', {}, 'Loading particles…')));
  try { await ensureData(); } catch (e) { root.innerHTML = ''; root.append(h('div', { class: 'empty' }, h('h2', {}, 'Could not load the particles'), h('p', {}, e.message))); return; }
  root.innerHTML = '';

  classifyEl = h('div', { class: 'classify' + (S.inspector ? '' : ' no-insp') });
  classifyEl.append(buildSidebar(), buildToolbar(),
    h('div', { class: 'stage', id: 'stage', tabindex: '-1' }),
    h('aside', { class: 'inspector', id: 'inspector', 'aria-label': 'Particle details' }),
    buildPalette());
  root.append(classifyEl);
  root.style.display = 'contents';
  document.addEventListener('keydown', onKey);
  recompute();
}

function emptyNoData(root) {
  root.append(h('div', { class: 'page' }, h('div', { class: 'empty' },
    h('svg', { class: 'flake', viewBox: '0 0 32 32', html: '<use href="#flake"/>' }),
    h('h2', {}, 'No particle data yet'),
    server.user?.is_admin
      ? [h('p', {}, 'Import a folder of MAT files from the Admin page to make its particles available here.'), h('p', {}, h('a', { class: 'btn primary', href: '#/admin' }, 'Go to Admin'))]
      : h('p', {}, 'No flights have been imported yet. An administrator can import them from the Admin page.'))));
}

// ---------------------------------------------------------------- sidebar
function buildSidebar() {
  const side = h('div', { class: 'sidebar', id: 'sidebar' });

  // match count
  side.append(h('section', {},
    h('button', { class: 'btn primary small mobile-only', style: { float: 'right' }, onclick: () => classifyEl.classList.remove('show-side') }, 'Show particles'),
    h('div', { class: 'matchcount', id: 'matchcount' }, '–'),
    h('div', { class: 'muted', id: 'matchsub', style: { fontSize: '.85rem', marginTop: '4px' } }, '')));

  // flights
  const fl = h('div', { class: 'flightlist' });
  const byProject = groupByProject(visibleFlights());
  for (const [project, list] of byProject) {
    if (byProject.size > 1 || project) {
      const n = list.filter((f) => S.flights.includes(f.id)).length;
      const cb = h('input', { type: 'checkbox', checked: n === list.length, onchange: (e) => {
        const ids = list.map((f) => f.id);
        S.flights = e.target.checked ? [...new Set([...S.flights, ...ids])] : S.flights.filter((x) => !ids.includes(x));
        savePrefs(); route();
      } });
      cb.indeterminate = n > 0 && n < list.length;
      fl.append(h('label', { class: 'flightrow projrow', title: `All flights in ${project || 'no project'}` }, cb,
        h('span', { class: 'fl-name' }, project || 'No project',
        project ? h('a', { class: 'projinfo', href: projectHref(project), title: `About ${project}: description, pictures and links`, 'aria-label': `About ${project}`, onclick: (e) => e.stopPropagation(),
          html: '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 11v6M12 7.5v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>' }) : null),
        h('span', { class: 'fl-n' }, fmtInt(list.reduce((a, f) => a + f.n, 0)))));
    }
  for (const f of list) {
    const on = S.flights.includes(f.id);
    fl.append(h('label', { class: 'flightrow' + (project ? ' inproj' : ''), title: [f.campaign, f.platform, f.date].filter(Boolean).join(', ') },
      h('input', { type: 'checkbox', checked: on, onchange: async (e) => {
        S.flights = e.target.checked ? [...S.flights, f.id] : S.flights.filter((x) => x !== f.id);
        savePrefs(); route();
      } }),
      h('svg', { viewBox: '0 0 24 24', html: `<use href="#${f.kind === 'aircraft' ? 'i-plane' : 'i-ground'}"/>` }),
      h('span', { class: 'fl-name' }, h('span', { class: 'fl-t' }, f.label || f.id), f.hidden ? h('span', { class: 'pill tiny' }, 'hidden') : null,
        fNotes.has(f.id) ? h('span', { class: 'notemark', title: fNotes.get(f.id), 'aria-label': 'Note: ' + fNotes.get(f.id),
          html: '<svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true"><path d="M5 4h14v11l-5 5H5z M14 20v-5h5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>' }) : null),
      h('span', { class: 'fl-n' }, fmtInt(f.n))));
  }
  }
  side.append(h('section', {},
    h('div', { class: 'side-h' }, h('h3', {}, 'Flights'),
      h('span', {},
        h('button', { class: 'btn ghost', onclick: () => { S.flights = visibleFlights().map((f) => f.id); savePrefs(); route(); } }, 'All'),
        h('button', { class: 'btn ghost', onclick: () => { S.flights = []; savePrefs(); route(); } }, 'None'))),
    fl));

  if (!D) return side;

  // which + order
  const showSel = h('select', { id: 'showSel', 'aria-label': 'Which particles', onchange: (e) => { S.show = e.target.value; savePrefs(); recompute(); } },
    h('option', { value: 'todo' }, 'Not yet classified by me'),
    h('option', { value: 'all' }, 'All particles'),
    h('option', { value: 'none' }, 'Not classified by anyone'),
    h('option', { value: 'disagree' }, 'People disagree'),
    h('option', { value: 'mine' }, 'Classified by me (review)'),
    h('optgroup', { label: 'I classified as…' }, scheme.classes.map((c) => h('option', { value: 'cls:' + c.id }, c.name))));
  showSel.value = S.show;
  const orderSel = h('select', { 'aria-label': 'Order', onchange: (e) => { S.order = e.target.value; savePrefs(); recompute(); } },
    h('option', { value: 'random' }, 'Random'),
    h('option', { value: 'stratified' }, 'Random, mixed sizes'),
    h('option', { value: 'time' }, 'Chronological'),
    h('option', { value: 'flight' }, 'By flight, then time'),
    h('option', { value: 'size-desc' }, 'Largest first'),
    h('option', { value: 'size-asc' }, 'Smallest first'),
    h('option', { value: 'least' }, 'Fewest classifications first'));
  orderSel.value = S.order;
  side.append(h('section', {},
    h('div', { class: 'side-h' }, h('h3', {}, 'Show')),
    h('div', { style: { display: 'grid', gap: '8px' } },
      showSel,
      h('div', { style: { display: 'flex', gap: '6px' } }, orderSel,
        h('button', { class: 'btn small', title: 'New random sample', onclick: () => { S.seed = (Math.random() * 1e9) | 0; savePrefs(); if (!S.order.startsWith('random') && S.order !== 'stratified') { S.order = 'random'; orderSel.value = 'random'; } recompute(); } }, 'Shuffle')),
      h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: S.hideEdge, onchange: (e) => { S.hideEdge = e.target.checked; savePrefs(); recompute(); } }), 'Hide particles cut off at the edge'))));

  // filters
  const fsec = h('section', {}, h('div', { class: 'side-h' }, h('h3', {}, 'Filters'),
    h('button', { class: 'btn ghost', onclick: () => { S.ranges = {}; dims.forEach((d) => { d.rh.range = [null, null]; }); savePrefs(); recompute(); } }, 'Clear')));
  dims = [];
  const sizeFields = Object.entries(D.fields).filter(([, m]) => m.group === 'size' && m.unit === 'µm');
  if (!D.fields[S.sizeField]) S.sizeField = sizeFields[0]?.[0] || 'len';
  const addDim = (key, field, opts = {}) => {
    const m = D.fields[field];
    if (!m) return;
    const isTime = field === 't';
    const rh = new RangeHist({
      key, label: opts.label || m.label, unit: isTime ? 'UTC' : m.unit,
      log: opts.log ?? false,
      format: isTime ? (v) => fmtTime(v) : (v) => String(+(+v).toPrecision(3)),
      parse: isTime ? (s, r) => parseClock(s, r.lo) : undefined,
      removable: opts.removable, onRemove: opts.onRemove, headExtra: opts.headExtra,
      onChange: (range) => { S.ranges[key] = range.slice(); savePrefs(); recompute(); },
    });
    rh.setDomain(D.col[field]);
    rh.range = (S.ranges[key] || [null, null]).slice();
    dims.push({ key, field, rh });
    fsec.append(rh.el);
  };
  addDim('size', S.sizeField, {
    log: true,
    headExtra: (lab) => {
      const sel = h('select', { 'aria-label': 'Size measure', onchange: (e) => { S.sizeField = e.target.value; delete S.ranges.size; savePrefs(); route(); } },
        sizeFields.map(([k, m]) => h('option', { value: k }, m.label)));
      sel.value = S.sizeField;
      lab.append(sel, h('span', { class: 'muted', style: { fontWeight: 400, marginLeft: '4px' } }, 'µm'));
    },
  });
  addDim('width', 'wid', { log: true, label: 'Width' });
  addDim('aspect', 'aspect', { log: true });
  addDim('area_ratio', 'area_ratio');
  addDim('focus', 'focus', { log: true, label: 'Focus' });
  if (D.fields.temp) addDim('temp', 'temp');
  if (D.fields.alt) addDim('alt', 'alt');
  addDim('t', 't', { label: 'Time of day' });
  for (const f of S.extraDims) {
    if (!D.fields[f] || dims.some((d) => d.field === f)) continue;
    addDim('x:' + f, f, { removable: true, onRemove: () => { S.extraDims = S.extraDims.filter((x) => x !== f); delete S.ranges['x:' + f]; savePrefs(); route(); } });
  }
  const used = new Set(dims.map((d) => d.field));
  const addSel = h('select', { 'aria-label': 'Add a filter', onchange: (e) => { if (e.target.value) { S.extraDims.push(e.target.value); savePrefs(); route(); } } },
    h('option', { value: '' }, 'Add a filter on…'),
    Object.entries(D.fields).filter(([k]) => !used.has(k) && !['imageType'].includes(k))
      .map(([k, m]) => h('option', { value: k }, m.label + (m.unit ? ` (${m.unit})` : ''))));
  fsec.append(h('div', { style: { marginTop: '14px' } }, addSel));
  side.append(fsec);
  side.append(h('section', {}, h('p', { class: 'muted', style: { fontSize: '.84rem', margin: 0 } },
    'Drag across a histogram to choose a range. Double-click it to clear. Bars show particles that pass the other filters.')));
  return side;
}

function parseClock(s, refUnix) {
  const m = s.match(/^(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?$/);
  if (!m) return NaN;
  const day = Math.floor(refUnix / 86400) * 86400;
  let t = day + (+m[1]) * 3600 + (+m[2]) * 60 + (+(m[3] || 0));
  if (t < refUnix - 3600) t += 86400;
  return t;
}

// ---------------------------------------------------------------- toolbar
function pxPerUm() { return S.zoom / 2.5; }

function buildToolbar() {
  const seg = (items, cur, onPick) => {
    const el = h('div', { class: 'seg', role: 'group' });
    for (const [v, label, title] of items) {
      el.append(h('button', { 'aria-pressed': String(v === cur), title, onclick: (e) => { const me = e.currentTarget; $$('button', el).forEach((b) => b.setAttribute('aria-pressed', String(b === me))); onPick(v); } }, label));
    }
    return el;
  };
  const tb = h('div', { class: 'toolbar' });
  tb.append(
    h('button', { class: 'btn small mobile-only', onclick: () => classifyEl.classList.toggle('show-side') }, 'Filters'),
    seg([['grid', 'Grid', 'Many particles at once (V)'], ['single', 'One at a time', 'One large particle at a time (V)']], S.view, (v) => { S.view = v; savePrefs(); renderStage(); }),
    seg([['true', 'True scale', 'Every particle at the same µm per screen pixel, so sizes compare'], ['fit', 'Fit', 'Each particle scaled to fill its tile']], S.scale, (v) => { S.scale = v; savePrefs(); updateZoomUi(); renderStage(); }),
    h('label', { class: 'inline', title: 'Zoom' }, h('svg', { class: 'zoomicon', viewBox: '0 0 24 24', 'aria-hidden': 'true', html: '<path d="M10 3a7 7 0 0 1 5.6 11.2l5.1 5.1-1.4 1.4-5.1-5.1A7 7 0 1 1 10 3zm0 2a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm-1 2h2v2h2v2h-2v2H9v-2H7V9h2V7z"/>' }),
      h('input', { type: 'range', id: 'zoom', min: '0.4', max: '5', step: '0.1', 'aria-label': 'Zoom', oninput: (e) => {
        if (S.scale === 'true') S.zoom = +e.target.value; else S.tile = +e.target.value;
        savePrefs(); updateZoomUi(); renderStage(true);
      } })),
    h('div', { class: 'scalebar', id: 'scalebar' }),
    h('span', { class: 'grow' }),
    buildDisplayButton(),
    h('button', { class: 'iconbtn', title: 'Download particle images (to scale, with outline and information) as a ZIP', 'aria-label': 'Download images', onclick: () => openExport(exportCtx), html: DL_ICON }),
    h('button', { class: 'iconbtn', title: 'Keyboard shortcuts (?)', 'aria-label': 'Keyboard shortcuts', onclick: openKeys,
      html: '<svg viewBox="0 0 24 24"><path d="M3 6h18a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1zm1 2v8h16V8H4zm2 1h2v2H6V9zm3 0h2v2H9V9zm3 0h2v2h-2V9zm3 0h3v2h-3V9zM6 12h2v2H6v-2zm3 0h6v2H9v-2zm7 0h2v2h-2v-2z"/></svg>' }),
    h('button', { class: 'iconbtn insp-toggle', title: 'Show or hide the details panel', 'aria-pressed': String(S.inspector), 'aria-label': 'Details panel', onclick: (e) => { S.inspector = !S.inspector; savePrefs(); e.currentTarget.setAttribute('aria-pressed', String(S.inspector)); classifyEl.classList.toggle('no-insp', !S.inspector); },
      html: '<svg viewBox="0 0 24 24"><path d="M4 4h16a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zm10 2v12h5V6h-5z"/></svg>' }),
  );
  queueMicrotask(updateZoomUi);
  return tb;
}
function updateZoomUi() {
  const z = $('#zoom'); if (!z) return;
  if (S.scale === 'true') { z.min = '0.4'; z.max = '5'; z.step = '0.1'; z.value = S.zoom; }
  else { z.min = '70'; z.max = '260'; z.step = '5'; z.value = S.tile; }
  const sb = $('#scalebar');
  if (S.scale !== 'true' || S.view !== 'grid') { sb.innerHTML = ''; return; }
  const k = pxPerUm();
  const L = [10, 20, 50, 100, 200, 500, 1000, 2000].find((l) => l * k >= 45) || 2000;
  sb.innerHTML = `<i style="width:${(L * k).toFixed(1)}px"></i><span>${L} µm</span>`;
}
function applyDisplay() {
  const baked = manifest ? manifest.flights.filter((f) => f.colormap && f.colormap !== 'gray').map((f) => f.id) : [];
  display.applyView({ cmap: S.cmap, brightness: S.brightness, contrast: S.contrast, outline: S.outlineColor, bakedFlights: baked });
  const chip = $('#dispChip');
  if (chip) chip.style.background = display.gradient(S.cmap);
}

// ---------------------------------------------------------------- display panel
// One popover for how images look: colour map, brightness/contrast, outline, captions.
const OUTLINE_COLOURS = [['#ff3b6b', 'Magenta'], ['#ffd23f', 'Yellow'], ['#3fe0ff', 'Cyan'], ['#ffffff', 'White'], ['#111111', 'Black']];
let dispPanel = null;
function buildDisplayButton() {
  const btn = h('button', { class: 'btn small dispbtn', id: 'dispBtn', 'aria-haspopup': 'dialog', 'aria-expanded': 'false', title: 'Colour map, brightness, contrast, outline and captions (M cycles colour maps)',
    onclick: (e) => { e.stopPropagation(); toggleDisplayPanel(); } },
  h('i', { class: 'dispchip', id: 'dispChip', style: { background: display.gradient(S.cmap) } }), 'Display');
  return btn;
}
function toggleDisplayPanel(force) {
  const open = force ?? !(dispPanel && dispPanel.isConnected);
  if (!open) { dispPanel?.remove(); $('#dispBtn')?.setAttribute('aria-expanded', 'false'); return; }
  dispPanel?.remove();
  dispPanel = h('div', { class: 'popover disppanel', role: 'dialog', 'aria-label': 'Display settings', onclick: (e) => e.stopPropagation() });
  document.body.append(dispPanel);
  renderDisplayPanel();
  const r = $('#dispBtn').getBoundingClientRect();
  dispPanel.style.top = (r.bottom + 8) + 'px';
  dispPanel.style.left = Math.max(12, Math.min(window.innerWidth - dispPanel.offsetWidth - 12, r.right - dispPanel.offsetWidth)) + 'px';
  $('#dispBtn').setAttribute('aria-expanded', 'true');
}
document.addEventListener('click', () => { if (dispPanel?.isConnected) toggleDisplayPanel(false); });
function refreshDisplayPanel() { if (dispPanel?.isConnected) renderDisplayPanel(); applyDisplay(); }
function setDisplay(patch, rerender = false) {
  Object.assign(S, patch); savePrefs(); applyDisplay();
  if (rerender) renderStage(true);
  if (dispPanel?.isConnected) renderDisplayPanel();
}
function renderDisplayPanel() {
  const p = dispPanel;
  const baked = D ? D.flights.filter((F) => F.meta.colormap && F.meta.colormap !== 'gray') : [];
  const allBaked = D && baked.length === D.flights.length;
  const slider = (label, key, min, max, step) => h('label', { class: 'dsl' },
    h('span', {}, label), h('input', { type: 'range', min, max, step, value: S[key], 'aria-label': label,
      oninput: (e) => { S[key] = +e.target.value; savePrefs(); applyDisplay(); e.target.nextSibling.textContent = (+e.target.value).toFixed(2); } }),
    h('output', { class: 'num' }, (+S[key]).toFixed(2)));
  const seg = (items, cur, onPick) => h('div', { class: 'seg' }, items.map(([v, l]) => h('button', { 'aria-pressed': String(v === cur), onclick: () => onPick(v) }, l)));
  p.innerHTML = '';
  p.append(...[
    h('div', { class: 'pop-h' }, h('h3', {}, 'Colour map'), h('span', { class: 'muted' }, display.colormap(S.cmap).name)),
    allBaked ? h('p', { class: 'pop-note' }, `These images were exported with the ${baked[0].meta.colormap} colour map, so they keep it. Export in grey to choose here.`) : null,
    h('div', { class: 'cmgrid' + (allBaked ? ' off' : '') }, display.colormaps().map((m) => h('button', {
      class: 'cm' + (m.id === S.cmap ? ' on' : ''), title: m.desc, 'aria-pressed': String(m.id === S.cmap),
      onclick: () => setDisplay({ cmap: m.id }) },
    h('i', { style: { background: display.gradient(m.id) } }), h('span', {}, m.name)))),
    baked.length && !allBaked ? h('p', { class: 'pop-note' }, `${baked.map((F) => F.id).join(', ')} ${baked.length === 1 ? 'was' : 'were'} exported with a colour map and keep${baked.length === 1 ? 's' : ''} it.`) : null,
    h('div', { class: 'pop-h' }, h('h3', {}, 'Brightness and contrast'),
      h('button', { class: 'btn ghost small', onclick: () => setDisplay({ brightness: 1, contrast: 1 }) }, 'Reset')),
    slider('Brightness', 'brightness', 0.4, 2, 0.02),
    slider('Contrast', 'contrast', 0.4, 3.5, 0.05),
    h('div', { class: 'pop-h' }, h('h3', {}, 'Outline'),
      h('label', { class: 'switch' }, h('input', { type: 'checkbox', checked: S.outline, onchange: (e) => setDisplay({ outline: e.target.checked }, true) }), h('span', {}, S.outline ? 'On' : 'Off'))),
    h('div', { class: 'swatches' }, OUTLINE_COLOURS.map(([c, n]) => h('button', {
      class: 'sw' + (S.outlineColor === c ? ' on' : ''), title: n, 'aria-label': `${n} outline`, style: { background: c },
      onclick: () => setDisplay({ outlineColor: c, outline: true }, !S.outline) }))),
    h('div', { class: 'pop-h' }, h('h3', {}, 'Show with each particle'),
      S.overlays.length ? h('button', { class: 'btn ghost small', onclick: () => setDisplay({ overlays: [] }, true) }, 'None') : null),
    h('div', { class: 'ovchips' }, OVERLAYS.map(([k, label]) => h('button', {
      class: 'ovchip', 'aria-pressed': String(S.overlays.includes(k)),
      onclick: () => setDisplay({ overlays: S.overlays.includes(k) ? S.overlays.filter((x) => x !== k) : [...S.overlays, k] }, true) }, label))),
    h('p', { class: 'pop-note keys-note' }, h('kbd', { class: 'k' }, 'M'), ' next colour map  ', h('kbd', { class: 'k' }, 'B'), ' contrast  ', h('kbd', { class: 'k' }, 'O'), ' outline')].filter(Boolean));
}
function cycleColormap(dir) {
  const ids = display.colormaps().map((m) => m.id);
  const i = ids.indexOf(S.cmap);
  const next = ids[(i + dir + ids.length) % ids.length];
  setDisplay({ cmap: next });
  toast(`Colour map: ${display.colormap(next).name}`, 1200);
}

// ---------------------------------------------------------------- keyboard overlay
function openKeys() {
  const dlgK = h('dialog', { class: 'dlg keysdlg', 'aria-label': 'Keyboard shortcuts' });
  const row = (keys, what) => [h('span', { class: 'kk' }, keys.map((k) => h('kbd', { class: 'k' }, k))), h('span', {}, what)];
  dlgK.append(h('div', { class: 'dlg-inner' },
    h('h3', {}, 'Keyboard shortcuts'),
    h('div', { class: 'keys-cols' },
      h('div', {}, h('h4', {}, 'Classify'),
        h('div', { class: 'keys' },
          scheme.classes.map((c) => row([c.key.toUpperCase()], c.name)).flat(),
          row(['Del'], 'Remove your label'))),
      h('div', {}, h('h4', {}, 'Flags'),
        h('div', { class: 'keys' }, scheme.flags.map((f) => row([f.key.toUpperCase()], f.name)).flat()),
        h('h4', {}, 'Move and select'),
        h('div', { class: 'keys' },
          row(['←', '→'], 'Previous / next'), row(['↑', '↓'], 'Row above / below'), row(['Space'], 'Select (grid)'),
          row(['Shift', 'click'], 'Select a run'), row(['Ctrl', 'A'], 'Select all shown'), row(['Esc'], 'Clear selection'))),
      h('div', {}, h('h4', {}, 'History'),
        h('div', { class: 'keys' }, row(['Z'], 'Undo'), row(['Y'], 'Redo')),
        h('h4', {}, 'View'),
        h('div', { class: 'keys' }, row(['V'], 'Grid / one at a time'), row(['Enter'], 'Open one at a time'), row(['M'], 'Next colour map'),
          row(['B'], 'Contrast'), row(['O'], 'Outline'), row(['?'], 'This list')))),
    h('div', { class: 'dlg-actions' }, h('a', { class: 'btn ghost', href: '#/guide', onclick: () => dlgK.close() }, 'Open the full guide'),
      h('button', { class: 'btn primary', autofocus: true, onclick: () => dlgK.close() }, 'Close'))));
  dlgK.addEventListener('close', () => dlgK.remove());
  dlgK.addEventListener('click', (e) => { if (e.target === dlgK) dlgK.close(); });
  document.body.append(dlgK);
  dlgK.showModal();
}

// ---------------------------------------------------------------- palette
function buildPalette() {
  const pal = h('div', { class: 'palette', id: 'palette' });
  const cls = h('div', { class: 'palette-classes', role: 'group', 'aria-label': 'Classes. Click, press the key, or drag particles here.' });
  for (const c of scheme.classes) {
    const btn = h('button', { class: 'cls', 'data-cls': c.id, style: { '--c': c.color }, title: `${c.desc}\nClick, press ${c.key.toUpperCase()}, or drag particles here.`, onclick: () => classify(c.id) },
      h('kbd', {}, c.key.toUpperCase()), h('span', { class: 'nm' }, c.name),
      h('span', { class: 'cnt', role: 'link', onclick: (e) => { e.stopPropagation(); if (e.currentTarget.textContent) reviewBin(c.id); } }));
    makeDropTarget(btn, (p) => classify(c.id, p));
    cls.append(btn);
  }
  const fl = h('div', { class: 'palette-flags', role: 'group', 'aria-label': 'Flags' });
  for (const f of scheme.flags) {
    fl.append(h('button', { class: 'flag', 'data-flag': f.id, 'aria-pressed': 'false', title: f.desc, onclick: () => toggleFlag(f.id) },
      h('kbd', {}, f.key.toUpperCase()), f.name));
  }
  const rm = h('button', { class: 'btn ghost small binremove', title: 'Remove your label from the selected particles (Delete). You can also drag particles here.', onclick: () => removeLabel() }, 'Remove label');
  makeDropTarget(rm, (p) => removeLabel(p));
  const st = h('div', { class: 'palette-status' },
    h('span', { class: 'sessionstat num', id: 'sessionstat' }, session.n ? `${fmtInt(session.n)} this session` : ''),
    h('span', { class: 'savestate', id: 'savestate' }, ''),
    rm,
    h('button', { class: 'btn ghost small', id: 'historyBtn', title: 'See and change what you’ve classified this session', onclick: openHistory }, 'History'),
    h('span', { class: 'undoredo' },
      h('button', { class: 'btn ghost small', id: 'undoBtn', title: 'Undo (Z or Ctrl/⌘ Z)', onclick: undo, disabled: true }, '↶ Undo'),
      h('button', { class: 'btn ghost small', id: 'redoBtn', title: 'Redo (Y, Shift Z or Ctrl/⌘ Shift Z)', onclick: redo, disabled: true }, 'Redo ↷')));
  pal.append(cls, fl, st);
  queueMicrotask(() => { updateSaveState(); updateBinCounts(); });
  return pal;
}
function updateSaveState() {
  const el = $('#savestate'); if (!el) return;
  el.className = 'savestate';
  if (practice.on) {
    el.classList.add('practice');
    el.innerHTML = '';
    const un = unsavedCount();
    el.append(h('span', { title: 'Classifications made now stay in this browser and are not saved to the database.' }, server.demo ? 'Demonstration: not saved' : 'Practice mode'));
    if (practice.count()) el.append( h('button', { class: 'linkbtn', title: 'Save your practice classifications to a file you keep', onclick: openDownload },
        un ? `Save ${fmtInt(un)} to a file` : 'Saved to a file ✓'));
    if (!server.demo && server.reachable !== false) el.append(h('button', { class: 'linkbtn', onclick: () => openAuth() }, 'Sign in'));
    $('#undoBtn') && ($('#undoBtn').disabled = !undoStack.length);
    $('#redoBtn') && ($('#redoBtn').disabled = !redoStack.length);
    return;
  }
  const s = api.saveState;
  if (s.error) { el.classList.add('error'); el.textContent = s.error; }
  else if (s.pending) { el.classList.add('pending'); el.textContent = `Saving ${s.pending}…`; }
  else el.textContent = 'All saved';
  $('#undoBtn').disabled = !undoStack.length;
  $('#redoBtn').disabled = !redoStack.length;
}
api.onSaveState(updateSaveState);

function updateSelChip() {
  const st = $('#stage'); if (!st) return;
  let chip = $('#selchip');
  if (S.view !== 'grid' || selection.size < 2) { chip?.remove(); return; }
  if (!chip) { chip = h('div', { class: 'selchip', id: 'selchip', role: 'status' }); st.append(chip); }
  chip.innerHTML = '';
  chip.append(h('b', {}, `${fmtInt(selection.size)} selected`), h('span', {}, 'Press a class key, click a class, or drag them onto one'),
    h('button', { class: 'btn ghost small', onclick: () => openExport(exportCtx) }, 'Download'),
    h('button', { class: 'btn ghost small', onclick: () => { const s = [...selection]; selection.clear(); redecorate(s); refreshPaletteState(); } }, 'Clear'));
}
function refreshPaletteState() {
  updateSelChip();
  const g = currentG();
  const mine = g >= 0 ? labels.mine.get(D.pid(g)) : null;
  $$('#palette .cls').forEach((b) => b.classList.toggle('current', !!mine && mine.label === b.dataset.cls && (S.view === 'single' || selection.size <= 1)));
  $$('#palette .flag').forEach((b) => {
    const f = b.dataset.flag;
    const on = armed.has(f) || (S.view === 'single' && mine && mine.flags.includes(f));
    b.setAttribute('aria-pressed', String(!!on));
  });
}

// ---------------------------------------------------------------- filtering
function recompute() {
  if (!D) { order = []; renderStage(); updateCount(0); return; }
  const n = D.n;
  const col = D.col;
  const edge = col.edge_touch;
  const nd = dims.length;
  for (const d of dims) d.rh.all.fill(0);
  const vals = dims.map((d) => col[d.field]);
  const bins = dims.map((d) => d.rh);
  const show = S.show;
  const showCls = show.startsWith('cls:') ? show.slice(4) : null;
  const out = [];
  for (let g = 0; g < n; g++) {
    if (S.hideEdge && edge && edge[g] > 0) continue;
    if (show !== 'all') {
      const pid = D.pid(g);
      const m = labels.mine.get(pid);
      if (show === 'todo' && m) continue;
      if (show === 'mine' && !m) continue;
      if (showCls && (!m || m.label !== showCls)) continue;
      if (show === 'none' && labels.counts.has(pid)) continue;
      if (show === 'disagree') { const c = labels.counts.get(pid); if (!c || Object.keys(c).length < 2) continue; }
    }
    let fails = 0, last = -1;
    for (let k = 0; k < nd; k++) {
      if (!bins[k].pass(vals[k][g])) { fails++; last = k; if (fails > 1) break; }
    }
    if (fails === 0) {
      for (let k = 0; k < nd; k++) { const b = bins[k].bin(vals[k][g]); if (b >= 0) bins[k].all[b]++; }
      out.push(g);
    } else if (fails === 1) {
      const b = bins[last].bin(vals[last][g]); if (b >= 0) bins[last].all[b]++;
    }
  }
  for (const d of dims) d.rh.draw();
  order = sortOrder(out);
  selection.clear(); anchorPos = -1; focusPos = 0;
  updateCount(out.length);
  renderStage();
}

function sortOrder(arr) {
  const t = D.col.t, size = D.col[S.sizeField] || D.col.len;
  const rnd = mulberry32(S.seed || 1);
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  switch (S.order) {
    case 'time': return arr.sort((a, b) => t[a] - t[b]);
    case 'flight': return arr; // already flight by flight, chronological within each
    case 'size-desc': return arr.sort((a, b) => (size[b] || 0) - (size[a] || 0));
    case 'size-asc': return arr.sort((a, b) => (size[a] || 0) - (size[b] || 0));
    case 'least': {
      shuffle(arr);
      const v = new Map(arr.map((g) => [g, totalVotes(D.pid(g))]));
      return arr.sort((a, b) => v.get(a) - v.get(b));
    }
    case 'stratified': {
      // round-robin across logarithmic size bins, so small and large particles both turn up
      shuffle(arr);
      const groups = new Map();
      for (const g of arr) { const s = size[g]; const k = Number.isFinite(s) && s > 0 ? Math.floor(Math.log(s) / Math.log(1.5)) : -999; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(g); }
      const lists = [...groups.values()];
      const res = [];
      for (let i = 0; res.length < arr.length; i++) for (const l of lists) if (i < l.length) res.push(l[i]);
      return res;
    }
    default: return shuffle(arr);
  }
}

function updateCount(n) {
  const el = $('#matchcount'); if (!el) return;
  el.innerHTML = `${fmtInt(n)} <small>particles</small>`;
  const sub = $('#matchsub');
  let mineN = 0;
  if (D) for (const pid of labels.mine.keys()) if (D.index.has(pid)) mineN++;
  sub.textContent = D ? `of ${fmtInt(D.n)} in ${D.flights.length} flight${D.flights.length === 1 ? '' : 's'}` + ` · you've classified ${fmtInt(mineN)}${practice.on ? ' (practice)' : ''}` : '';
}

// ---------------------------------------------------------------- stage
function currentG() {
  if (!order.length) return -1;
  return order[Math.max(0, Math.min(order.length - 1, focusPos))];
}

function renderStage(keepScroll = false) {
  const st = $('#stage'); if (!st) return;
  updateZoomUi();
  applyDisplay();
  const top = st.scrollTop;
  st.innerHTML = '';
  if (!D) {
    st.append(h('div', { class: 'empty' }, h('svg', { class: 'flake', viewBox: '0 0 32 32', html: '<use href="#flake"/>' }),
      h('h2', {}, 'Choose some flights'), h('p', {}, 'Tick one or more flights on the left to start looking at particles.')));
    renderInspector(-1);
    return;
  }
  if (!order.length) {
    const done = S.show === 'todo';
    const go = (patch) => () => { Object.assign(S, patch); savePrefs(); route(); };
    st.append(h('div', { class: 'empty' + (done ? ' done' : '') },
      h('svg', { class: 'flake', viewBox: '0 0 32 32', html: '<use href="#flake"/>' }),
      h('h2', {}, done ? 'All done here' : 'No particles match'),
      h('p', {}, done ? 'You have classified every particle that matches these filters.' : 'Try widening a range on the left, or showing a different set of particles.'),
      h('div', { class: 'empty-actions' },
        done ? h('button', { class: 'btn', onclick: go({ show: 'mine' }) }, 'Review what I classified') : null,
        done ? h('button', { class: 'btn', onclick: go({ show: 'all', order: 'least' }) }, 'Help with particles others have done') : null,
        h('button', { class: 'btn primary', onclick: go({ ranges: {} }) }, 'Clear all filters'))));
    renderInspector(-1);
    return;
  }
  if (S.view === 'single') renderSingle(st);
  else renderGrid(st, keepScroll ? Math.max(shown, S.page) : S.page);
  if (keepScroll) st.scrollTop = top;
  renderInspector(currentG());
  refreshPaletteState();
}

// ---- grid
let io = null;
function renderGrid(st, count) {
  if (!lsGet('cpi.welcomed', false)) {
    const card = h('div', { class: 'welcome' },
      h('div', {},
        h('h2', {}, 'Three ways to classify'),
        h('ol', {},
          h('li', {}, h('b', {}, 'Press a key. '), 'Point at a particle and press its class key, shown on each class button below.'),
          h('li', {}, h('b', {}, 'Click. '), 'Select particles, then click a class button.'),
          h('li', {}, h('b', {}, 'Drag. '), 'Drag particles onto a class button. Select several first to move them together.')),
        h('p', { class: 'muted' }, 'Changed your mind? Press another class, or Z to undo. Press ? at any time for all the shortcuts.'),
        practice.on ? h('p', { class: 'muted', style: { marginTop: '4px' } }, server.demo
          ? 'This is a demonstration: try it freely. Nothing is saved to the database.'
          : 'You’re in practice mode until you sign in: try it freely, nothing is saved to the database.') : null),
      h('button', { class: 'btn primary', onclick: () => { lsSet('cpi.welcomed', true); card.remove(); } }, 'Got it'));
    st.append(card);
  }
  const grid = h('div', { class: 'grid' + (S.scale === 'fit' ? ' fit' : ''), id: 'grid', role: 'listbox', 'aria-multiselectable': 'true', 'aria-label': 'Particles' });
  grid.style.setProperty('--tile-size', S.tile + 'px');
  st.append(grid);
  const more = h('div', { class: 'more', id: 'more' });
  st.append(more);
  shown = 0;
  appendTiles(Math.min(count, order.length));
  io?.disconnect();
  io = new IntersectionObserver((ents) => { if (ents.some((e) => e.isIntersecting)) appendTiles(S.page); }, { root: st, rootMargin: '600px' });
  io.observe(more);
}
function appendTiles(k) {
  const grid = $('#grid'); if (!grid) return;
  const end = Math.min(order.length, shown + k);
  const frag = document.createDocumentFragment();
  const maxW = Math.max(200, ($('#stage').clientWidth || 800) - 60);
  for (let pos = shown; pos < end; pos++) frag.append(tileEl(pos, maxW));
  grid.append(frag);
  shown = end;
  const more = $('#more');
  more.innerHTML = '';
  if (shown < order.length) more.textContent = `Showing ${fmtInt(shown)} of ${fmtInt(order.length)} – scroll for more`;
  else more.append(h('span', {}, `All ${fmtInt(order.length)} shown. `),
    S.show === 'todo' ? h('button', { class: 'btn small', onclick: recompute, style: { marginLeft: '8px' } }, 'Load the next unclassified set') : '');
  if (S.outline) drawOutlines();
}
function tileEl(pos, maxW) {
  const g = order[pos];
  const w = D.col.w[g], hh = D.col.h[g];
  const img = h('img', { src: D.img(g), alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' });
  if (img.complete && img.naturalWidth) img.classList.add('ok');
  else { img.addEventListener('load', () => img.classList.add('ok'), { once: true }); img.addEventListener('error', () => img.classList.add('ok', 'err'), { once: true }); }
  const pic = h('div', { class: 'pic' }, img);
  const t = h('div', { class: 'tile', role: 'option', 'data-pos': pos, 'data-g': g });
  if (S.scale === 'true') {
    let s = D.pixel(g) * pxPerUm();
    let dw = w * s, dh = hh * s;
    const lim = Math.min(maxW / dw, 520 / dh, 1);
    if (lim < 1) { dw *= lim; dh *= lim; s *= lim; t.append(h('span', { class: 'shrunk', title: 'Too big to show at true scale; shrunk to fit' }, `×${lim.toFixed(2)}`)); }
    img.width = Math.max(1, Math.round(dw)); img.height = Math.max(1, Math.round(dh));
    if (s >= 1.5) img.classList.add('px');
  } else {
    img.classList.add('px');
  }
  t.prepend(pic);
  const lines = infoItems(D, g, S.overlays, S.sizeField);
  if (lines.length) t.append(h('span', { class: 'cap num' }, lines.map((l) => h('span', {}, l.map((it) => h('b', {}, it))))));
  if (S.overlays.includes('scalebar')) {
    const place = () => tileScaleBar(pic, img, g);
    if (S.scale === 'true') place(); else img.addEventListener('load', place, { once: true });
  }
  decorateTile(t);
  t.draggable = true;
  t.addEventListener('dragstart', (e) => {
    if (!selection.has(pos)) {
      const old = [...selection, focusPos];
      selection.clear(); selection.add(pos); focusPos = pos; anchorPos = pos;
      redecorate([...old, pos]);
    }
    startDrag(e, [...selection].sort((a, b) => a - b));
  });
  t.addEventListener('dragend', endDrag);
  t.addEventListener('click', onTileClick);
  t.addEventListener('dblclick', () => { focusPos = pos; S.view = 'single'; savePrefs(); route(); });
  t.addEventListener('mouseenter', () => { hoverG = g; renderInspector(g); });
  t.addEventListener('mouseleave', () => { hoverG = -1; renderInspector(currentG()); });
  return t;
}
/** A small scale bar in the corner of a tile, sized for how big the image is drawn. */
function tileScaleBar(pic, img, g) {
  const shownW = img.width && S.scale === 'true' ? img.width : img.clientWidth;
  if (!shownW) return;
  const pxPerUm = shownW / D.col.w[g] / D.pixel(g);
  const L = niceBar(pxPerUm, 18, Math.max(24, shownW * 0.45));
  pic.querySelector('.tsb')?.remove();
  pic.append(h('span', { class: 'tsb' }, h('i', { style: { width: (L * pxPerUm).toFixed(1) + 'px' } }), `${L} µm`));
}
function decorateTile(t) {
  const pos = +t.dataset.pos, g = +t.dataset.g;
  const pid = D.pid(g);
  t.querySelectorAll('.tag, .tagname, .votes').forEach((e) => e.remove());
  const m = labels.mine.get(pid);
  t.classList.toggle('done', !!m && S.show === 'todo');
  t.classList.toggle('sel', selection.has(pos));
  t.classList.toggle('focus', pos === focusPos);
  t.setAttribute('aria-selected', String(selection.has(pos)));
  if (m) {
    const c = classById.get(m.label);
    t.append(h('span', { class: 'tag', style: { background: c?.color || '#888' } }),
      h('span', { class: 'tagname' }, (c?.name || m.label) + (m.flags.length ? ' +' + m.flags.length : '')));
  }
  const n = totalVotes(pid) - (m ? 1 : 0);
  if (n > 0) t.append(h('span', { class: 'votes', title: `${n} classification${n === 1 ? '' : 's'} by other people` }, `${n} other${n === 1 ? '' : 's'}`));
  t.title = m ? `You: ${classById.get(m.label)?.name || m.label}` : '';
}
function tileAt(pos) { return $(`.tile[data-pos="${pos}"]`, $('#grid') || document); }
function redecorate(positions) { for (const p of positions) { const t = tileAt(p); if (t) decorateTile(t); } }

function onTileClick(e) {
  const pos = +e.currentTarget.dataset.pos;
  const prevFocus = focusPos;
  if (e.shiftKey && anchorPos >= 0) {
    const [a, b] = anchorPos < pos ? [anchorPos, pos] : [pos, anchorPos];
    if (!(e.metaKey || e.ctrlKey)) selection.clear();
    for (let p = a; p <= b; p++) selection.add(p);
  } else if (e.metaKey || e.ctrlKey) {
    if (selection.has(pos)) selection.delete(pos); else selection.add(pos);
    anchorPos = pos;
  } else {
    const only = selection.size === 1 && selection.has(pos);
    selection.clear();
    if (!only) selection.add(pos);
    anchorPos = pos;
  }
  focusPos = pos;
  $$('#grid .tile.sel, #grid .tile.focus').forEach((t) => decorateTile(t));
  redecorate([pos, prevFocus]);
  for (const p of selection) { const t = tileAt(p); if (t) decorateTile(t); }
  renderInspector(order[pos]);
  refreshPaletteState();
}

function moveFocus(newPos, extend = false) {
  newPos = Math.max(0, Math.min(order.length - 1, newPos));
  const old = focusPos;
  focusPos = newPos;
  if (extend) { selection.add(old); selection.add(newPos); }
  if (S.view === 'single') { renderStage(); return; }
  if (newPos >= shown) appendTiles(newPos - shown + S.page);
  redecorate([old, newPos]);
  const t = tileAt(newPos);
  t?.scrollIntoView({ block: 'nearest' });
  renderInspector(order[newPos]);
  refreshPaletteState();
}
function verticalNeighbour(dir) {
  const cur = tileAt(focusPos); if (!cur) return focusPos + dir;
  const r = cur.getBoundingClientRect();
  const cx = r.left + r.width / 2;
  let best = -1, bestD = Infinity, rowEdge = null;
  for (let k = 1; k <= 120; k++) {
    const p = focusPos + dir * k;
    const t = tileAt(p); if (!t) break;
    const q = t.getBoundingClientRect();
    const inNextRow = dir > 0 ? q.top >= r.bottom - 2 : q.bottom <= r.top + 2;
    if (!inNextRow) continue;
    if (rowEdge === null) rowEdge = dir > 0 ? q.bottom : q.top;
    else if (dir > 0 ? q.top >= rowEdge - 2 : q.bottom <= rowEdge + 2) break;   // reached the row after
    const d = Math.abs(q.left + q.width / 2 - cx);
    if (d < bestD) { bestD = d; best = p; }
  }
  return best >= 0 ? best : focusPos;
}

// ---- outlines
function outlineSvg(g, pts) {
  if (!pts || pts.length < 6) return null;
  const w = D.col.w[g], hh = D.col.h[g];
  let s = '';
  for (let i = 0; i < pts.length; i += 2) s += pts[i] + ',' + pts[i + 1] + ' ';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'outline');
  svg.setAttribute('viewBox', `0 0 ${w} ${hh}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.innerHTML = `<polygon points="${s.trim()}"/>`;
  return svg;
}
async function outlineFor(g) {
  const F = D.flights[D.fl[g]];
  const b = await loadBoundaries(F.id, D.stem(g));
  return b[String(D.roi[g])];
}
async function drawOutlines() {
  const tiles = $$('#grid .tile').filter((t) => !t.querySelector('svg.outline'));
  const byStem = new Map();
  for (const t of tiles) {
    const g = +t.dataset.g;
    const k = D.flights[D.fl[g]].id + '/' + D.stem(g);
    if (!byStem.has(k)) byStem.set(k, []);
    byStem.get(k).push(t);
  }
  for (const list of byStem.values()) {
    const g0 = +list[0].dataset.g;
    await loadBoundaries(D.flights[D.fl[g0]].id, D.stem(g0));
    for (const t of list) {
      const g = +t.dataset.g;
      const svg = outlineSvg(g, await outlineFor(g));
      const pic = t.querySelector('.pic');
      if (svg && pic && !pic.querySelector('svg')) { if (S.scale === 'fit') fitOverlay(pic, svg); else pic.append(svg); }
    }
  }
}
function fitOverlay(pic, svg) {
  // in fit mode the img is letterboxed: wrap it so the overlay matches the image box
  const img = pic.querySelector('img');
  const wrap = h('span', { style: { position: 'relative', display: 'inline-block', lineHeight: 0, maxWidth: '100%', maxHeight: '100%' } });
  img.replaceWith(wrap);
  wrap.append(img, svg);
}

// ---- single view
function renderSingle(st) {
  focusPos = Math.max(0, Math.min(order.length - 1, focusPos));
  const g = order[focusPos];
  const strip = h('div', { class: 'filmstrip', 'aria-label': 'Nearby particles' });
  const a = Math.max(0, focusPos - 10), b = Math.min(order.length, focusPos + 14);
  for (let p = a; p < b; p++) {
    const gg = order[p];
    const m = labels.mine.get(D.pid(gg));
    const c = m && classById.get(m.label);
    const t = h('div', { class: 'tile' + (p === focusPos ? ' cur' : ''), title: c ? c.name : 'Not classified', onclick: () => moveFocus(p) },
      h('img', { src: D.img(gg), alt: '', loading: 'lazy' }),
      c ? h('span', { class: 'tag', style: { background: c.color } }) : null);
    strip.append(t);
  }
  // biggest integer-ish zoom that fits the space
  const W = (st.clientWidth || 800) - 150, Hh = (st.clientHeight || 600) - 215;
  const w = D.col.w[g], hh = D.col.h[g];
  let s = Math.min(W / w, Hh / hh, 8);
  if (s > 1) s = Math.max(1, Math.floor(s * 2) / 2);
  const img = h('img', { src: D.img(g), alt: `Particle ${D.pid(g)}`, width: Math.round(w * s), height: Math.round(hh * s) });
  img.draggable = true;
  img.addEventListener('dragstart', (e) => startDrag(e, [focusPos]));
  img.addEventListener('dragend', endDrag);
  const pic = h('div', { class: 'single-pic' }, img);
  const k = s / D.pixel(g);
  const L = [10, 20, 50, 100, 200, 500, 1000].find((l) => l * k >= 50) || 1000;
  const m = labels.mine.get(D.pid(g));
  const c = m && classById.get(m.label);
  const main = h('div', { class: 'single-main' },
    h('div', { class: 'single-pos num' }, `${fmtInt(focusPos + 1)} of ${fmtInt(order.length)}`),
    h('button', { class: 'single-nav prev', 'aria-label': 'Previous particle', onclick: () => moveFocus(focusPos - 1), disabled: focusPos === 0 }, '‹'),
    h('div', { class: 'single-plate' }, pic,
      (() => { const ls = infoLines(D, g, S.overlays, S.sizeField); return ls.length ? h('div', { class: 'infobox num' }, ls.map((l) => h('div', {}, l))) : null; })(),
      h('div', { class: 'row between', style: { marginTop: '10px' } },
        h('div', { class: 'scalebar', style: { alignItems: 'flex-start' } }, h('i', { style: { width: (L * k).toFixed(1) + 'px' } }), h('span', {}, `${L} µm`)),
        h('button', { class: 'btn ghost small', onclick: () => downloadOne(g, exportCtx), title: 'Download this image' }, 'Download'))),
    h('button', { class: 'single-nav next', 'aria-label': 'Next particle', onclick: () => moveFocus(focusPos + 1), disabled: focusPos >= order.length - 1 }, '›'),
    h('div', { class: 'single-mylabel' },
      c ? [h('span', { class: 'dot', style: { background: c.color } }), `You said ${c.name}` + (m.flags.length ? ` (${m.flags.map((f) => scheme.flags.find((x) => x.id === f)?.name || f).join(', ')})` : '')]
        : h('span', { class: 'muted' }, 'Press a class key, or drag the image to a class'),
      c ? h('button', { class: 'btn ghost small', onclick: () => removeLabel([focusPos]) }, 'Remove') : null));
  st.append(h('div', { class: 'single' }, strip, main));
  if (S.outline) outlineFor(g).then((pts) => { const svg = outlineSvg(g, pts); if (svg) pic.append(svg); });
  // preload the next few
  for (let p = focusPos + 1; p < Math.min(order.length, focusPos + 6); p++) { const im = new Image(); im.src = D.img(order[p]); }
}

// ---------------------------------------------------------------- inspector
const GROUPS = [['size', 'Size'], ['shape', 'Shape'], ['quality', 'Image quality'], ['position', 'Where and when'], ['instrument', 'Instrument']];
function renderInspector(g) {
  const el = $('#inspector'); if (!el) return;
  if (g < 0 || !D) { el.innerHTML = '<p class="muted">Point at a particle to see its details.</p>'; return; }
  const pid = D.pid(g);
  const F = D.flights[D.fl[g]];
  const w = D.col.w[g], hh = D.col.h[g];
  const sc = Math.min(236 / w, 236 / hh, 4);
  const pic = h('div', { class: 'single-pic' }, h('img', { src: D.img(g), alt: '', width: Math.round(w * sc), height: Math.round(hh * sc) }));
  if (S.outline) outlineFor(g).then((pts) => { const svg = outlineSvg(g, pts); if (svg) pic.append(svg); });
  const dl = h('dl', { class: 'kv' });
  for (const [grp, title] of GROUPS) {
    const rows = [];
    if (grp === 'position') rows.push(['Date', fmtTime(D.col.t[g], true).slice(0, 10)], ['Time (UTC)', fmtTime(D.col.t[g])]);
    for (const [k, m] of Object.entries(D.fields)) {
      if (m.group !== grp || k === 't') continue;
      const v = D.col[k][g];
      if (!Number.isFinite(v)) continue;
      let txt = k === 'edge_touch' ? (v ? 'yes' : 'no') : fmtNum(v, m.unit);
      if (k === 'lat' || k === 'lon') txt = v.toFixed(4) + '°';
      rows.push([m.label, txt]);
    }
    if (grp === 'instrument') {
      rows.unshift(['Flight', F.meta.label || F.id], ['MAT file', D.stem(g)], ['Index', String(D.roi[g])]);
      if (F.meta.source) rows.push(['Pipeline id', sourceId(D, g)]);
      if (fNotes.has(F.id)) rows.splice(1, 0, ['Flight note', h('span', { class: 'flightnote insp', html: noteHtml(fNotes.get(F.id)) })]);
    }
    if (!rows.length) continue;
    dl.append(h('div', { class: 'grp' }, title));
    for (const [k, v] of rows) dl.append(h('dt', {}, k), h('dd', {}, v));
  }
  const votes = labels.counts.get(pid);
  let vb = null;
  if (votes) {
    const tot = Object.values(votes).reduce((a, b) => a + b, 0);
    vb = h('div', {}, h('div', { class: 'grp', style: { fontWeight: 700, marginTop: '12px' } }, `Classifications (${tot})`),
      h('div', { class: 'votebars' }, Object.entries(votes).sort((a, b) => b[1] - a[1]).map(([lab, n]) => {
        const c = classById.get(lab);
        return h('div', { class: 'votebar' }, h('div', {}, c?.name || lab, h('b', { style: { width: (100 * n / tot) + '%', background: c?.color || '#888' } })), h('span', { class: 'num' }, n));
      })));
  }
  let yours = null;
  {
    const m = labels.mine.get(pid);
    const c = m && classById.get(m.label);
    yours = h('div', { class: 'yours' + (c ? '' : ' none') },
      c ? [h('span', { class: 'dot', style: { background: c.color } }), h('span', { class: 'yours-txt' }, 'You: ', h('b', {}, c.name),
            m.flags.length ? ' + ' + m.flags.map((f) => scheme.flags.find((x) => x.id === f)?.name || f).join(', ') : '')]
        : h('span', { class: 'yours-txt muted' }, 'Not classified by you'),
      h('button', { class: 'btn small', onclick: () => openFixer(pid) }, c ? 'Change' : 'Classify'));
  }
  el.innerHTML = '';
  el.append(h('div', { class: 'insp-pic' }, pic), h('div', { class: 'insp-id row between' }, h('span', {}, pid),
    h('button', { class: 'btn ghost small', title: 'Download this image with the current download settings', onclick: () => downloadOne(g, exportCtx) }, 'Download')));
  if (yours) el.append(yours);
  if (vb) el.append(vb);
  el.append(dl);
}

// ---------------------------------------------------------------- classify
// Every change to your labels goes through commit(): it records what each
// particle was before and after, so any change can be undone and redone.
function targets() {
  if (S.view === 'grid' && selection.size) return [...selection].sort((a, b) => a - b);
  return order.length ? [focusPos] : [];
}
const snap = (v) => (v ? { label: v.label, flags: v.flags.slice() } : null);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const itemFor = (pid, v) => (v ? { pid, label: v.label, flags: v.flags } : { pid, label: null });
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

function needUser() {
  if (server.user?.must_change_pw) { openChangePassword(true); return false; }
  return true;     // signed in, or practice mode (saved in this browser only)
}
function commit(changes, desc) {
  changes = changes.filter((c) => !same(c.before, c.after));
  if (!changes.length) return false;
  for (const c of changes) setMine(c.pid, c.after);
  saveItems(changes.map((c) => itemFor(c.pid, c.after)));
  undoStack.push({ changes, desc, t: Date.now() });
  if (undoStack.length > 500) undoStack.shift();
  redoStack.length = 0;
  return true;
}
function replay(entry, dir) {
  const key = dir === 'undo' ? 'before' : 'after';
  for (const c of entry.changes) setMine(c.pid, c[key]);
  saveItems(entry.changes.map((c) => itemFor(c.pid, c[key])));
}
function undo() {
  const e = undoStack.pop();
  if (!e) { toast('Nothing to undo'); return; }
  replay(e, 'undo'); redoStack.push(e);
  toast(`Undone: ${e.desc}`);
  afterChange(e.changes.map((c) => c.pid), true);
}
function redo() {
  const e = redoStack.pop();
  if (!e) { toast('Nothing to redo'); return; }
  replay(e, 'redo'); undoStack.push(e);
  toast(`Redone: ${e.desc}`);
  afterChange(e.changes.map((c) => c.pid), true);
}
/** Refresh whatever shows these particles. jump = move the focus to the first one. */
function afterChange(pids, jump = false) {
  if (D && $('#stage')) {
    const gs = new Set(pids.map((p) => D.index.get(p)));
    if (jump) { const p0 = order.indexOf(D.index.get(pids[0])); if (p0 >= 0) focusPos = p0; }
    if (S.view === 'single') renderStage();
    else {
      $$('#grid .tile').forEach((t) => { if (gs.has(+t.dataset.g) || t.classList.contains('focus') || +t.dataset.pos === focusPos) decorateTile(t); });
      if (jump) tileAt(focusPos)?.scrollIntoView({ block: 'nearest' });
      renderInspector(hoverG >= 0 ? hoverG : currentG());
    }
    updateCount(order.length);
    refreshPaletteState();
    updateBinCounts();
  }
  updateSaveState();
  refreshFixer();
  refreshHistory();
}

function hitBin(clsId) {
  const btn = $(`#palette .cls[data-cls="${clsId}"]`);
  if (btn) { btn.classList.remove('hit'); void btn.offsetWidth; btn.classList.add('hit'); }
}

/** Put the particles at these positions into a class (key press, click, or drop). */
function classify(clsId, pos = targets()) {
  if (!needUser() || !pos.length) return;
  const c = classById.get(clsId);
  const changes = pos.map((p) => {
    const pid = D.pid(order[p]);
    const prev = labels.mine.get(pid);
    return { pid, before: snap(prev), after: { label: clsId, flags: armed.size ? [...armed] : (prev ? prev.flags.slice() : []) } };
  });
  const changed = changes.filter((x) => x.before && x.before.label !== clsId).length;
  bumpSession(changes.filter((x) => !x.before).length);
  commit(changes, `${plural(pos.length, 'particle')} → ${c.name}`);
  armed.clear();
  hitBin(clsId);
  if (S.view === 'single') {
    if (focusPos < order.length - 1) focusPos++;
    renderStage();
  } else {
    for (const p of pos) { const t = tileAt(p); if (t) { t.style.setProperty('--flash', c.color); t.classList.remove('flash'); void t.offsetWidth; t.classList.add('flash'); } }
    selection.clear();
    let next = pos[pos.length - 1] + 1;
    while (next < order.length && labels.mine.has(D.pid(order[next]))) next++;
    const old = focusPos;
    focusPos = Math.min(next, order.length - 1);
    redecorate([...pos, old, focusPos]);
    if (focusPos >= shown) appendTiles(S.page);
    tileAt(focusPos)?.scrollIntoView({ block: 'nearest' });
    renderInspector(hoverG >= 0 ? hoverG : currentG());
  }
  if (changed) toast(`Changed ${plural(changed, 'particle')} to ${c.name}`);
  else if (pos.length > 1) toast(`${pos.length} particles → ${c.name}`);
  updateCount(order.length);
  refreshPaletteState();
  updateSaveState();
  updateBinCounts();
  refreshHistory();
}

/** Remove your label from particles (Delete key, the Remove bin, or the fix dialog). */
function removeLabel(pos = targets()) {
  if (!needUser()) return;
  const changes = pos.map((p) => D.pid(order[p])).filter((pid) => labels.mine.has(pid))
    .map((pid) => ({ pid, before: snap(labels.mine.get(pid)), after: null }));
  if (!changes.length) { toast('Nothing to remove: you haven’t classified that yet'); return; }
  commit(changes, `Removed ${plural(changes.length, 'label')}`);
  toast(`Removed ${plural(changes.length, 'label')} (Z to undo)`);
  afterChange(changes.map((c) => c.pid));
}

function toggleFlag(flagId) {
  if (!needUser()) return;
  const pos = targets();
  const labelled = pos.filter((p) => labels.mine.has(D.pid(order[p])));
  if (!labelled.length) {
    // nothing classified yet: arm the flag for the next class key
    if (armed.has(flagId)) armed.delete(flagId); else armed.add(flagId);
    refreshPaletteState();
    return;
  }
  const turnOn = !labelled.every((p) => labels.mine.get(D.pid(order[p])).flags.includes(flagId));
  const fname = scheme.flags.find((f) => f.id === flagId)?.name;
  const changes = labelled.map((p) => {
    const pid = D.pid(order[p]);
    const m = labels.mine.get(pid);
    const flags = turnOn ? [...new Set([...m.flags, flagId])] : m.flags.filter((f) => f !== flagId);
    return { pid, before: snap(m), after: { label: m.label, flags } };
  });
  commit(changes, `${fname} ${turnOn ? 'added to' : 'removed from'} ${plural(changes.length, 'particle')}`);
  toast(`${fname} ${turnOn ? 'added' : 'removed'}`);
  afterChange(changes.map((c) => c.pid));
}

/** Change one particle by id, wherever it is (fix dialog, history). */
function setLabelForPid(pid, after, desc) {
  if (!needUser()) return;
  if (commit([{ pid, before: snap(labels.mine.get(pid)), after }], desc)) afterChange([pid]);
}

function bumpSession(n) {
  if (!n) return;
  const now = Date.now();
  if (!session.start) session.start = now;
  if (session.last) session.active += Math.min(now - session.last, 90000);   // ignore long breaks
  session.last = now;
  session.n += n;
  const el = $('#sessionstat');
  if (el) {
    const mins = session.active / 60000;
    el.textContent = `${fmtInt(session.n)} this session` + (mins > 1 ? `, ${(session.n / mins).toFixed(1)}/min` : '');
    el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump');
  }
}

// ---------------------------------------------------------------- drag and drop
let dragPos = null;
function startDrag(e, positions) {
  if (!needUser()) { e.preventDefault(); return; }
  dragPos = positions;
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', positions.map((p) => D.pid(order[p])).join('\n'));
  const ghost = h('div', { class: 'dragghost' },
    positions.slice(0, 3).map((p, i) => h('img', { src: D.img(order[p]), alt: '', style: { transform: `translate(${i * 6}px, ${i * 6}px) rotate(${(i - 1) * 4}deg)` } })),
    h('span', { class: 'dragcount' }, positions.length > 1 ? `${positions.length} particles` : '1 particle'));
  document.body.append(ghost);
  e.dataTransfer.setDragImage(ghost, 30, 30);
  setTimeout(() => ghost.remove(), 0);
  document.body.classList.add('dragging');
  for (const p of positions) tileAt(p)?.classList.add('dragsrc');
}
function endDrag() {
  dragPos = null;
  document.body.classList.remove('dragging');
  $$('.dragsrc').forEach((t) => t.classList.remove('dragsrc'));
  $$('.over').forEach((t) => t.classList.remove('over'));
}
function makeDropTarget(el, onDrop) {
  el.addEventListener('dragover', (e) => { if (!dragPos) return; e.preventDefault(); e.dataTransfer.dropEffect = 'move'; el.classList.add('over'); });
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', (e) => { if (!dragPos) return; e.preventDefault(); const p = dragPos; endDrag(); onDrop(p); });
}

// ---------------------------------------------------------------- bins: counts
function updateBinCounts() {
  if (!D) return;
  const n = new Map();
  for (const [pid, m] of labels.mine) if (D.index.has(pid)) n.set(m.label, (n.get(m.label) || 0) + 1);
  $$('#palette .cls').forEach((b) => {
    const k = n.get(b.dataset.cls) || 0;
    const cnt = b.querySelector('.cnt');
    const prev = cnt.textContent;
    cnt.textContent = k ? fmtInt(k) : '';
    if (prev && cnt.textContent !== prev) { cnt.classList.remove('bump'); void cnt.offsetWidth; cnt.classList.add('bump'); }
    cnt.title = k ? `Show the ${plural(k, 'particle')} you put in ${classById.get(b.dataset.cls).name}` : '';
  });
  const hb = $('#historyBtn');
  if (hb) hb.textContent = undoStack.length ? `History (${undoStack.length})` : 'History';
}
function reviewBin(clsId) {
  S.show = 'cls:' + clsId; savePrefs();
  const sel = $('#showSel'); if (sel) sel.value = S.show;
  recompute();
  toast(`Showing your ${classById.get(clsId).name} particles. Press another class key or drag to move any that are wrong.`, 4000);
}

// ---------------------------------------------------------------- fix dialog
// A small window for changing one particle's label from anywhere (details
// panel, history list) without losing your place.
let fixPid = null;
const fixDlg = h('dialog', { class: 'dlg fixdlg', 'aria-label': 'Change classification' });
document.body.append(fixDlg);
fixDlg.addEventListener('close', () => { fixPid = null; });
fixDlg.addEventListener('click', (e) => { if (e.target === fixDlg) fixDlg.close(); });
function openFixer(pid) {
  if (!needUser()) return;
  fixPid = pid;
  refreshFixer();
  if (!fixDlg.open) fixDlg.showModal();
}
function refreshFixer() {
  if (!fixPid || !D) return;
  const g = D.index.get(fixPid);
  const m = labels.mine.get(fixPid);
  const c = m && classById.get(m.label);
  const w = D.col.w[g], hh = D.col.h[g];
  const sc = Math.min(300 / w, 220 / hh, 4);
  fixDlg.innerHTML = '';
  fixDlg.append(h('div', { class: 'dlg-inner' },
    h('div', { class: 'fix-top' },
      h('div', { class: 'insp-pic' }, h('div', { class: 'single-pic' }, h('img', { src: D.img(g), alt: '', width: Math.round(w * sc), height: Math.round(hh * sc) }))),
      h('div', {},
        h('h3', {}, 'Change classification'),
        h('div', { class: 'insp-id' }, fixPid),
        h('p', { class: 'fix-now' }, c ? [h('span', { class: 'dot', style: { background: c.color } }), ' You said ', h('b', {}, c.name)] : h('span', { class: 'muted' }, 'You haven’t classified this one.')))),
    h('div', { class: 'fix-classes' }, scheme.classes.map((k) => h('button', {
      class: 'cls' + (m && m.label === k.id ? ' current' : ''), style: { '--c': k.color },
      onclick: () => { setLabelForPid(fixPid, { label: k.id, flags: m ? m.flags.slice() : [] }, `1 particle → ${k.name}`); hitBin(k.id); },
    }, h('kbd', {}, k.key.toUpperCase()), h('span', { class: 'nm' }, k.name)))),
    h('div', { class: 'palette-flags fix-flags' }, scheme.flags.map((f) => h('button', {
      class: 'flag', 'aria-pressed': String(!!m && m.flags.includes(f.id)), disabled: !m,
      onclick: () => {
        const on = !m.flags.includes(f.id);
        setLabelForPid(fixPid, { label: m.label, flags: on ? [...m.flags, f.id] : m.flags.filter((x) => x !== f.id) }, `${f.name} ${on ? 'added' : 'removed'}`);
      },
    }, h('kbd', {}, f.key.toUpperCase()), f.name))),
    h('div', { class: 'dlg-actions' },
      h('button', { class: 'btn ghost danger', disabled: !m, onclick: () => setLabelForPid(fixPid, null, 'Removed 1 label') }, 'Remove my label'),
      h('span', { style: { flex: 1 } }),
      h('button', { class: 'btn ghost', onclick: undo, disabled: !undoStack.length }, 'Undo'),
      h('button', { class: 'btn primary', onclick: () => fixDlg.close() }, 'Done'))));
}

// ---------------------------------------------------------------- history
const histDlg = h('dialog', { class: 'dlg histdlg', 'aria-label': 'History' });
document.body.append(histDlg);
histDlg.addEventListener('click', (e) => { if (e.target === histDlg) histDlg.close(); });
function openHistory() { refreshHistory(true); histDlg.showModal(); }
function refreshHistory(force = false) {
  updateBinCounts();
  if (!force && !histDlg.open) return;
  const list = h('ol', { class: 'histlist', reversed: true });
  const entries = undoStack.slice(-60).reverse();
  for (const e of entries) {
    list.append(h('li', {},
      h('div', { class: 'hist-head' }, h('span', { class: 'num muted' }, new Date(e.t).toTimeString().slice(0, 5)), h('span', {}, e.desc)),
      h('div', { class: 'hist-thumbs' }, e.changes.slice(0, 12).map((ch) => {
        const g = D?.index.get(ch.pid);
        if (g === undefined) return null;
        const now = labels.mine.get(ch.pid);
        const c = now && classById.get(now.label);
        return h('button', { class: 'hist-thumb', title: (c ? c.name : 'No label') + ' – click to change', onclick: () => { histDlg.close(); openFixer(ch.pid); } },
          h('img', { src: D.img(g), alt: '' }), c ? h('span', { class: 'tag', style: { background: c.color } }) : null);
      }), e.changes.length > 12 ? h('span', { class: 'muted' }, `+${e.changes.length - 12}`) : null)));
  }
  histDlg.innerHTML = '';
  histDlg.append(h('div', { class: 'dlg-inner' },
    h('h3', {}, 'What you’ve done this session'),
    h('p', { class: 'dlg-why' }, 'Newest first. Click a particle to change or remove its label. Undo steps back one change at a time; Redo puts it back.'),
    entries.length ? list : h('p', { class: 'muted' }, 'Nothing yet. Classify a particle and it will appear here.'),
    h('div', { class: 'dlg-actions' },
      h('button', { class: 'btn', onclick: undo, disabled: !undoStack.length }, 'Undo'),
      h('button', { class: 'btn', onclick: redo, disabled: !redoStack.length }, 'Redo'),
      h('span', { style: { flex: 1 } }),
      h('button', { class: 'btn primary', onclick: () => histDlg.close() }, 'Close'))));
}

// ---------------------------------------------------------------- keyboard
function onKey(e) {
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'select' || tag === 'textarea' || dlg.open) return;
  if (histDlg.open) return;
  if (fixDlg.open) {
    const c = classByKey.get(e.key.toLowerCase());
    if (c && fixPid && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      const m = labels.mine.get(fixPid);
      setLabelForPid(fixPid, { label: c.id, flags: m ? m.flags.slice() : [] }, `1 particle → ${c.name}`);
    }
    return;
  }
  if (!D || !$('#stage')) return;
  const k = e.key;
  const mod = e.metaKey || e.ctrlKey;
  if (mod && k.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
  if (mod && k.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
  if (mod && k.toLowerCase() === 'a' && S.view === 'grid') { e.preventDefault(); for (let p = 0; p < shown; p++) selection.add(p); $$('#grid .tile').forEach(decorateTile); return; }
  if (mod || e.altKey) return;
  const lk = k.toLowerCase();
  if (classByKey.has(lk)) { e.preventDefault(); classify(classByKey.get(lk).id); return; }
  if (flagByKey.has(lk)) { e.preventDefault(); toggleFlag(flagByKey.get(lk).id); return; }
  const grid = S.view === 'grid';
  switch (k) {
    case 'ArrowRight': e.preventDefault(); moveFocus(focusPos + 1, e.shiftKey && grid); break;
    case 'ArrowLeft': e.preventDefault(); moveFocus(focusPos - 1, e.shiftKey && grid); break;
    case 'ArrowDown': if (grid) { e.preventDefault(); moveFocus(verticalNeighbour(1), e.shiftKey); } break;
    case 'ArrowUp': if (grid) { e.preventDefault(); moveFocus(verticalNeighbour(-1), e.shiftKey); } break;
    case ' ':
      e.preventDefault();
      if (grid) { if (selection.has(focusPos)) selection.delete(focusPos); else selection.add(focusPos); anchorPos = focusPos; redecorate([focusPos]); refreshPaletteState(); }
      else moveFocus(focusPos + 1);
      break;
    case 'Enter': if (grid) { S.view = 'single'; savePrefs(); route(); } break;
    case 'Escape':
      if (selection.size) { const s = [...selection]; selection.clear(); redecorate(s); refreshPaletteState(); }
      else if (!grid) { S.view = 'grid'; savePrefs(); route(); }
      armed.clear(); refreshPaletteState();
      break;
    case 'z': undo(); break;
    case 'Z': redo(); break;      // Shift Z
    case 'y': case 'Y': redo(); break;
    case 'Delete': case 'Backspace': e.preventDefault(); removeLabel(); break;
    case 'v': case 'V': S.view = grid ? 'single' : 'grid'; savePrefs(); route(); break;
    case 'o': case 'O': setDisplay({ outline: !S.outline }, true); toast(S.outline ? 'Outline on' : 'Outline off', 1000); break;
    case 'b': case 'B': setDisplay({ contrast: S.contrast > 1.2 ? 1 : 1.7 }); toast(S.contrast > 1 ? 'Contrast boosted' : 'Contrast normal', 1000); break;
    case 'm': cycleColormap(1); break;
    case 'M': cycleColormap(-1); break;
    case '?': openKeys(); break;
  }
}

// ============================================================== FLIGHTS VIEW
async function viewFlights(root) {
  root.className = 'page'; root.style.display = '';
  root.innerHTML = '';
  const inner = h('div', { class: 'page-inner' });
  root.append(inner);
  if (!manifest || !visibleFlights().length) return emptyNoData(inner);
  const tot = visibleFlights().reduce((s, f) => s + f.n, 0);
  inner.append(h('h1', {}, 'Flights and sessions'),
    h('p', { class: 'page-lede' }, `${fmtInt(visibleFlights().length)} dataset${visibleFlights().length === 1 ? '' : 's'} with ${fmtInt(tot)} particle images. Open a project for its description, pictures and links. Tick the flights you want to classify, or open a row for its instrument and processing details.`));
  const pc = await projectCards(groupByProject(visibleFlights()));
  if (pc) inner.append(pc);
  const tbody = h('tbody');
  const table = h('table', { class: 'data' }, h('thead', {}, h('tr', {},
    h('th', {}, ''), h('th', {}, 'Flight'), h('th', {}, 'Date'), h('th', {}, 'Platform'),
    h('th', {}, 'Instrument'), h('th', { class: 'n' }, 'Particles'), h('th', {}, 'Through the flight'), h('th', {}, 'Time (UTC)'), h('th', {}, ''))), tbody);
  const sel = new Set(selectedFlights());
  for (const [project, list] of groupByProject(visibleFlights())) {
    tbody.append(h('tr', { class: 'grouprow' }, h('td', { colspan: '9' }, h('b', {}, project || 'No project'),
      h('span', { class: 'muted small' }, `  ${list.length} flight${list.length === 1 ? '' : 's'}, ${fmtInt(list.reduce((a, f) => a + f.n, 0))} particles  `),
      project ? h('a', { class: 'btn ghost small', href: projectHref(project) }, 'About this project') : null,
      h('button', { class: 'btn ghost small', onclick: () => { S.flights = list.map((f) => f.id); savePrefs(); location.hash = '#/classify'; } }, 'Classify this project'))));
  for (const f of list) {
    const det = h('tr', { class: 'detail', hidden: true }, h('td', { colspan: '9' }, h('span', { class: 'muted' }, 'Loading…')));
    const tr = h('tr', {},
      h('td', {}, h('input', { type: 'checkbox', checked: sel.has(f.id), 'aria-label': 'Include ' + f.id, onchange: (e) => {
        if (e.target.checked) sel.add(f.id); else sel.delete(f.id);
        S.flights = visibleFlights().map((x) => x.id).filter((id) => sel.has(id)); savePrefs();
      } })),
      h('td', { class: 'flightname' }, h('svg', { class: 'kindicon', viewBox: '0 0 24 24', html: `<use href="#${f.kind === 'aircraft' ? 'i-plane' : 'i-ground'}"/>` }), h('b', {}, f.label || f.id), f.hidden ? h('span', { class: 'pill tiny' }, 'hidden') : null, h('div', { class: 'muted', style: { fontSize: '.82rem' } }, f.id),
        fNotes.has(f.id) ? h('div', { class: 'flightnote', html: noteHtml(fNotes.get(f.id)) }) : null),
      h('td', { class: 'num nowrap' }, f.date || '–'),
      h('td', {}, f.platform || '–'),
      h('td', {}, [f.instrument?.name, f.instrument?.version && 'v' + f.instrument.version].filter(Boolean).join(' ') || '–', f.pixel_um ? h('div', { class: 'muted', style: { fontSize: '.82rem' } }, `${f.pixel_um} µm/px`) : null),
      h('td', { class: 'n' }, fmtInt(f.n)),
      h('td', {}, sparkline(f.timeline)),
      h('td', { class: 'num nowrap' }, f.t_start ? `${f.t_start.slice(11, 16)}–${f.t_end.slice(11, 16)}` : '–'),
      h('td', {}, h('span', { style: { display: 'flex', gap: '6px' } },
        h('button', { class: 'btn small', onclick: async (e) => {
          det.hidden = !det.hidden;
          e.currentTarget.textContent = det.hidden ? 'Details' : 'Hide';
          if (!det.hidden && !det.dataset.loaded) {
            det.dataset.loaded = 1;
            const m = await loadMeta(f.id);
            det.firstChild.innerHTML = '';
            det.firstChild.append(metaDetail(m));
          }
        } }, 'Details'),
        h('button', { class: 'btn small primary', onclick: () => { S.flights = [f.id]; savePrefs(); location.hash = '#/classify'; } }, 'Classify'))));
    tbody.append(tr, det);
  }
  }
  inner.append(h('div', { class: 'tablewrap' }, table),
    h('p', { style: { marginTop: '16px' } }, h('a', { class: 'btn primary', href: '#/classify' }, 'Classify the ticked flights')));
}
/** Particles per time bin (bars), with altitude as a line when the flight has it. */
function sparkline(tl) {
  if (!tl || !tl.n) return h('span', { class: 'muted' }, '–');
  const W = 150, H = 34, nb = tl.n.length, bw = W / nb;
  const max = Math.max(1, ...tl.n);
  let svg = '';
  tl.n.forEach((v, i) => { if (v) { const hh = Math.max(1, (v / max) * (H - 4)); svg += `<rect x="${(i * bw).toFixed(1)}" y="${(H - hh).toFixed(1)}" width="${Math.max(1, bw - 0.6).toFixed(1)}" height="${hh.toFixed(1)}"/>`; } });
  let title = `Particles through the flight, peak ${fmtInt(max)} per bin`;
  if (tl.alt && tl.alt.some((v) => v !== null)) {
    const vals = tl.alt.filter((v) => v !== null);
    const lo = Math.min(...vals), hi = Math.max(...vals), span = Math.max(1, hi - lo);
    const pts = tl.alt.map((v, i) => (v === null ? null : `${((i + 0.5) * bw).toFixed(1)},${(H - 3 - ((v - lo) / span) * (H - 8)).toFixed(1)}`)).filter(Boolean).join(' ');
    svg += `<polyline points="${pts}"/>`;
    title += `; line: altitude ${fmtInt(lo)}–${fmtInt(hi)} m`;
  }
  return h('svg', { class: 'spark', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': title, html: `<title>${esc(title)}</title>` + svg });
}
function metaDetail(m) {
  const dl = h('dl', { class: 'kv', style: { maxWidth: '560px' } });
  const row = (k, v) => { if (v !== null && v !== undefined && v !== '') dl.append(h('dt', {}, k), h('dd', { style: { textAlign: 'left' } }, String(v))); };
  row('Kind', m.kind); row('Instrument', [m.instrument?.name, m.instrument?.version, m.instrument?.serial].filter(Boolean).join(' / '));
  row('Pixel size', m.pixel_um && m.pixel_um + ' µm'); row('MAT files', (m.files || []).join(', '));
  row('First / last particle', m.t_start && `${m.t_start} – ${m.t_end}`);
  row('Navigation file', m.nav?.file); row('Site', m.site && JSON.stringify(m.site));
  row('Processing code', m.processing?.code_version); row('Processed', m.processing?.processed_utc); row('Method', m.processing?.method);
  row('Exported', m.exported_utc); row('Notes', m.notes);
  const wrap = h('div', {}, dl);
  if (m.processing?.options && Object.keys(m.processing.options).length) {
    wrap.append(h('details', { style: { marginTop: '8px' } }, h('summary', {}, 'All processing options'), h('pre', { class: 'json' }, JSON.stringify(m.processing.options, null, 1))));
  }
  return wrap;
}

// ============================================================== PROGRESS VIEW
/** What has been classified in practice mode, in this browser. */
function practiceCard() {
  const st = practice.load();
  const n = Object.keys(st).length;
  const by = new Map();
  for (const [, [lab]] of Object.entries(st)) by.set(lab, (by.get(lab) || 0) + 1);
  const max = Math.max(1, ...by.values());
  const card = h('div', { class: 'card practicecard' },
    h('h2', {}, server.demo ? 'Your classifications in this demonstration' : 'Practice classifications in this browser'),
    h('p', { class: 'muted' }, n
      ? `${fmtInt(n)} particle${n === 1 ? '' : 's'} classified. These stay in this browser and are not in the database. Save them to a file so they can’t be lost` + (server.demo ? '; the file can be added to the database later.' : '; or sign in and keep them under your name.')
      : 'Nothing yet. Classify some particles on the Classify page; they’ll be counted here.'),
    n ? h('div', { class: 'hbars' }, scheme.classes.filter((c) => by.has(c.id)).map((c) => h('div', { class: 'hbar' },
      h('span', {}, c.name), h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: (100 * by.get(c.id) / max) + '%', background: c.color } })),
      h('span', { class: 'v' }, fmtInt(by.get(c.id)))))) : null,
    n ? h('div', { class: 'row', style: { marginTop: '14px', flexWrap: 'wrap' } },
      h('button', { class: 'btn' + (unsavedCount() ? ' primary' : ''), onclick: openDownload }, 'Save to a file'),
      h('button', { class: 'btn ghost danger', onclick: () => { if (confirm(`Clear all ${n} practice classifications from this browser?`)) { practice.clear(); refreshLabels().then(() => route()); } } }, 'Clear them'),
      !server.demo && server.reachable !== false ? h('button', { class: 'btn', onclick: () => openAuth() }, 'Sign in to save them') : null) : null,
    n ? savedLine() : null);
  return card;
}
function savedLine() {
  const m = practiceMeta.load();
  const un = unsavedCount();
  if (!m.saved) return h('p', { class: 'muted small', style: { margin: '10px 0 0' } }, 'Not saved to a file yet.');
  const when = new Date(m.saved).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
  return h('p', { class: 'muted small', style: { margin: '10px 0 0' } },
    `Last saved to a file ${when}` + (m.name ? ` as ${m.name}` : '') + (un ? `; ${fmtInt(un)} classified since then.` : '; nothing new since.'));
}

async function viewProgress(root) {
  root.className = 'page'; root.style.display = '';
  root.innerHTML = '';
  const inner = h('div', { class: 'page-inner' });
  root.append(inner);
  inner.append(h('h1', {}, 'Progress'));
  if (practice.on) inner.append(practiceCard());
  if (server.reachable === false || server.demo) return;
  let st;
  try { await api.flush(); st = await api.fetchStats(); } catch (e) { inner.append(h('p', {}, e.message)); return; }
  const reload = async () => { await refreshLabels(); route(); };
  const sessBits = server.user && !server.user.must_change_pw
    ? [uploadCard(reload), await sessionsCard(reload)] : [];
  const total = st.by_label.reduce((s, r) => s + +r.n, 0);
  const mineTot = st.mine.reduce((s, r) => s + +r.n, 0);
  inner.append(h('p', { class: 'page-lede' }, server.user
    ? `You have classified ${fmtInt(mineTot)} particle${mineTot === 1 ? '' : 's'}. Everyone together has made ${fmtInt(total)} classifications.`
    : `${fmtInt(total)} classifications so far. Sign in to see your own.`));

  const bars = (rows, title) => {
    const tot = rows.reduce((s, r) => s + +r.n, 0) || 1;
    const max = Math.max(1, ...rows.map((r) => +r.n));
    const byId = new Map(rows.map((r) => [r.label, +r.n]));
    return h('div', { class: 'card' }, h('h2', {}, title),
      rows.length ? h('div', { class: 'hbars' }, scheme.classes.filter((c) => byId.has(c.id)).map((c) => h('div', { class: 'hbar' },
        h('span', {}, c.name), h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: (100 * byId.get(c.id) / max) + '%', background: c.color } })),
        h('span', { class: 'v' }, `${fmtInt(byId.get(c.id))}`)))) : h('p', { class: 'muted' }, 'None yet.'),
      rows.length ? h('p', { class: 'muted', style: { fontSize: '.85rem', marginBottom: 0 } }, `${fmtInt(tot)} in total`) : null);
  };
  const cards = h('div', { class: 'cards' });
  if (server.user) cards.append(bars(st.mine, 'Your classifications'));
  cards.append(bars(st.by_label, 'Everyone'));
  inner.append(cards);

  // flights
  const fbody = h('tbody');
  const byF = new Map(st.by_flight.map((r) => [r.flight_id, r]));
  for (const f of manifest?.flights || []) {
    const r = byF.get(f.id);
    const done = r ? +r.particles : 0;
    fbody.append(h('tr', {}, h('td', {}, f.label || f.id), h('td', { class: 'n' }, fmtInt(done)), h('td', { class: 'n' }, fmtInt(f.n)),
      h('td', {}, h('div', { class: 'progress-track' }, h('i', { style: { width: Math.min(100, 100 * done / Math.max(1, f.n)) + '%' } }))),
      h('td', { class: 'n' }, r ? fmtInt(r.n) : '0')));
  }
  inner.append(h('h2', { style: { marginTop: '32px' } }, 'By flight'),
    h('div', { class: 'tablewrap' }, h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'Flight'), h('th', { class: 'n' }, 'Particles classified'), h('th', { class: 'n' }, 'Particles'), h('th', {}, 'Coverage'), h('th', { class: 'n' }, 'Classifications'))), fbody)));

  // people + agreement
  const ag = st.agreement || {};
  const pbody = h('tbody', {}, st.people.map((p) => h('tr', {}, h('td', {}, p.name, +p.is_external ? h('span', { class: 'pill tiny', title: 'Added from a session file; no login' }, 'from file') : null,
    p.expertise ? h('div', { class: 'muted small' }, p.expertise) : null), h('td', { class: 'n' }, fmtInt(p.n)), h('td', { class: 'num' }, (p.last || '').slice(0, 10)))));
  inner.append(h('div', { class: 'cards', style: { marginTop: '32px' } },
    h('div', { class: 'card' }, h('h2', {}, 'Contributors'), !server.user ? h('p', { class: 'muted' }, 'Sign in to see who has been classifying.') : st.people.length ? h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'Name'), h('th', { class: 'n' }, 'Classified'), h('th', {}, 'Last active'))), pbody) : h('p', { class: 'muted' }, 'Nobody yet – be the first.')),
    h('div', { class: 'card' }, h('h2', {}, 'Agreement'),
      +ag.multi ? h('p', {}, `${fmtInt(ag.multi)} particles have been classified by more than one person. All agree on ${fmtInt(ag.unanimous)} of them (${Math.round(100 * ag.unanimous / ag.multi)}%).`)
        : h('p', { class: 'muted' }, 'No particle has been classified by two people yet. Choose "Fewest classifications first" or "Not classified by anyone" to spread the work, or "People disagree" to review conflicts.'),
      h('a', { class: 'btn small', href: '#/classify', onclick: () => { S.show = 'disagree'; savePrefs(); } }, 'Review disagreements'))));

  if (server.user) {
    inner.append(h('div', { class: 'card', style: { marginTop: '32px' } }, h('h2', {}, 'Download'),
      h('p', { class: 'muted' }, server.user.is_admin ? 'As an administrator you can download everyone’s classifications. Each row is one person’s label for one particle; the pid gives the flight, MAT file and index.' : 'Download your own classifications. Each row gives the flight, MAT file and index of the particle.'),
      h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } },
        h('a', { class: 'btn', href: api.exportUrl('csv', true) }, 'My classifications (CSV)'),
        server.user.is_admin ? h('a', { class: 'btn primary', href: api.exportUrl('csv') }, 'Everyone (CSV)') : null,
        server.user.is_admin ? h('a', { class: 'btn', href: api.exportUrl('json') }, 'Everyone (JSON)') : null)));
  }
  const sb = sessBits.filter(Boolean);
  if (sb.length) inner.append(h('div', { class: 'cards sesscards', style: { marginTop: '32px' } }, ...sb));
}

// ============================================================== GUIDE VIEW
async function viewGuide(root) {
  root.className = 'page'; root.style.display = '';
  root.innerHTML = '';
  const inner = h('div', { class: 'page-inner' });
  root.append(inner);
  inner.append(h('h1', {}, 'How to classify'), h('p', { class: 'page-lede' }, scheme.help || ''));
  // examples: particles with the most agreement for each class, from the flights you have loaded
  const examples = new Map();
  if (D) {
    const best = new Map();
    for (const [pid, c] of labels.counts) {
      if (!D.index.has(pid)) continue;
      const tot = Object.values(c).reduce((a, b) => a + b, 0);
      for (const [lab, n] of Object.entries(c)) {
        const score = n * 10 - (tot - n) * 15;
        if (n / tot < 0.66) continue;
        if (!best.has(lab)) best.set(lab, []);
        best.get(lab).push([score, D.index.get(pid)]);
      }
    }
    for (const [lab, arr] of best) examples.set(lab, arr.sort((a, b) => b[0] - a[0]).slice(0, 6).map((x) => x[1]));
  }
  inner.append(h('h2', {}, 'Classes'), h('div', { class: 'guide-classes' }, scheme.classes.map((c) => h('div', { class: 'guide-cls', style: { '--c': c.color } },
    h('kbd', { class: '', style: { background: c.color } }, c.key.toUpperCase()), h('b', {}, c.name), h('p', {}, c.desc),
    examples.get(c.id) ? h('div', { class: 'guide-ex' }, examples.get(c.id).map((g) => h('img', { src: D.img(g), alt: c.name + ' example', loading: 'lazy' }))) : null))));
  inner.append(h('h2', { style: { marginTop: '28px' } }, 'Flags'),
    h('p', { class: 'muted' }, 'Flags add detail to a class. Press a flag key before the class key to include it, or press it after classifying to add or remove it.'),
    h('div', { class: 'guide-classes' }, scheme.flags.map((f) => h('div', { class: 'guide-cls', style: { '--c': 'var(--accent)' } },
      h('kbd', { class: 'k' }, f.key.toUpperCase()), h('b', {}, f.name), h('p', {}, f.desc)))));
  const keys = [
    ['← →', 'Previous / next particle'], ['↑ ↓', 'Row above / below (grid)'], ['Space', 'Select or deselect (grid); next particle (one at a time)'],
    ['Click, Shift-click, ⌘/Ctrl-click', 'Select one, a run, or add to the selection'], ['⌘/Ctrl A', 'Select everything shown'],
    ['Enter / double-click', 'Open one at a time'], ['V', 'Switch grid / one at a time'], ['Esc', 'Clear selection, or back to grid'],
    ['O', 'Show / hide the particle outline'], ['B', 'Boost contrast'], ['M / Shift M', 'Next / previous colour map'],
    ['Z / ⌘/Ctrl Z', 'Undo'], ['Y / Shift Z / ⌘/Ctrl Shift Z', 'Redo'], ['Delete', 'Remove your label from the selected particles'],
    ['Drag', 'Drag particles (or a whole selection) onto a class to classify them, or onto Remove label'], ['?', 'This page'],
  ];
  inner.append(h('h2', { style: { marginTop: '28px' } }, 'Keyboard'),
    h('div', { class: 'card' }, h('div', { class: 'keys' }, keys.map(([k, d]) => [h('span', {}, k.split(/( \/ |, )/).map((p) => /^( \/ |, )$/.test(p) ? p : h('kbd', { class: 'k' }, p))), h('span', {}, d)]))));
  inner.append(h('h2', { style: { marginTop: '28px' } }, 'Tips'),
    h('ul', { class: 'page-lede' },
      h('li', {}, 'Use True scale when comparing sizes: every particle is drawn at the same µm per screen pixel, with a scale bar in the toolbar.'),
      h('li', {}, 'In the grid, select several particles that look alike and press one key, or drag them onto a class, to classify them together.'),
      h('li', {}, 'Made a mistake? Press another class key on the particle to change it, Delete to remove it, or Z to undo. The number on each class button shows how many you’ve put there; click it to review them.'),
      h('li', {}, 'Use Random, mixed sizes to see a sample that covers small and large particles evenly.'),
      h('li', {}, 'If you’re not sure, classify your best guess and add the Unsure flag, or use Can’t tell. Both are useful.')));
}

// ============================================================== start
async function start() {
  await api.whoAmI();
  const sch = (server.reachable ? await api.fetchScheme() : null) || await fetch('config/scheme.json').then((r) => r.json()).catch(() => null);
  if (sch) setScheme(sch);
  await display.loadColormaps();
  try { manifest = await loadManifest(); } catch { manifest = null; }
  fNotes = await flightNotes();
  applyDisplay();
  renderUserBox();
  if (server.user?.must_change_pw) openChangePassword(true);
  else api.restoreQueue();
  route();
}
/** Install a classification scheme (at start, or after an admin edits the classes). */
function setScheme(sch) {
  scheme = sch;
  scheme.flags = scheme.flags || [];
  classById = new Map(scheme.classes.map((c) => [c.id, c]));
  classByKey = new Map(scheme.classes.filter((c) => c.key).map((c) => [c.key.toLowerCase(), c]));
  flagByKey = new Map(scheme.flags.filter((f) => f.key).map((f) => [f.key.toLowerCase(), f]));
}
start();
