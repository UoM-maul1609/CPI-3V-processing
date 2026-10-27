// Talking to the small PHP API. Everything else happens in the browser.

const BASE = 'api/';
const HDR = { 'X-Requested-With': 'cpi-classifier' };

export const server = { reachable: null, user: null, allowRegistration: false, needsCode: false, demo: false, setup: false };

async function call(action, { body, params } = {}) {
  const q = new URLSearchParams({ a: action, ...(params || {}) });
  const opts = { credentials: 'same-origin', headers: { ...HDR } };
  if (body !== undefined) {
    opts.method = 'POST';
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(BASE + '?' + q, opts);
  } catch (e) {
    throw Object.assign(new Error('The server could not be reached. Check your connection.'), { offline: true });
  }
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok || !data || data.ok === false) {
    const err = new Error((data && data.error) || `Server error (${res.status})`);
    err.status = res.status;
    if (!data) err.offline = true;
    throw err;
  }
  return data;
}

export async function whoAmI() {
  try {
    const d = await call('me');
    server.reachable = true;
    server.user = d.user;
    server.allowRegistration = d.allow_registration;
    server.needsCode = d.needs_code;
    server.demo = !!d.demo;
    server.setup = !!d.setup;
    server.expertiseLevels = d.expertise_levels || [];
  } catch (e) {
    server.reachable = false;
    server.user = null;
  }
  return server.user;
}

export async function login(username, password) {
  const d = await call('login', { body: { username, password } });
  server.user = d.user;
  return d.user;
}
export async function register(fields) {
  const d = await call('register', { body: fields });
  server.user = d.user;
  return d.user;
}
export async function logout() {
  await call('logout', { body: {} });
  server.user = null;
}
export async function fetchLabels(flightIds) {
  if (!server.reachable || !flightIds.length) return { mine: {}, counts: {} };
  return call('labels', { params: { flights: flightIds.join(',') } });
}
export async function fetchStats() { return call('stats'); }
export async function changePassword(current, next) {
  const d = await call('change_password', { body: { current, new: next } });
  server.user = d.user;
  return d.user;
}
export async function fetchScheme() {
  try { return (await call('scheme')).scheme; } catch { return null; }
}
// sessions added from files, and your own details
export const uploadSession = (body) => call('session_upload', { body });
export const fetchSessions = () => call('sessions');
export const savePrefs = (prefs) => call('prefs_save', { body: { prefs } });
export const fetchExamples = (flights, per = 8, seed = '') => call('examples', { params: { flights: flights.join(','), per: String(per), seed } });
export const deleteSession = (id) => call('session_delete', { body: { id } });
export const updateProfile = (body) => call('profile_update', { body });
// admin actions (admin page only)
export const adminGet = (action, params) => call('admin_' + action, { params });
export const adminPost = (action, body) => call('admin_' + action, { body });
export function exportUrl(format = 'csv', mine = false) {
  return BASE + '?' + new URLSearchParams({ a: 'export', format, ...(mine ? { mine: '1' } : {}) });
}

// ---------------------------------------------------------------- save queue
// Labels are saved in small batches shortly after you make them. Unsent ones
// are kept in this browser so a dropped connection doesn't lose work.
const QKEY = 'cpi.pending';
let queue = new Map();     // pid -> item
let timer = null;
let inflight = false;
const listeners = new Set();
export const saveState = { pending: 0, error: null, savedTotal: 0 };

function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

function persist() {
  if (!server.user) return;
  lsSet(QKEY + '.' + server.user.id, JSON.stringify([...queue.values()]));
}
export function restoreQueue() {
  if (!server.user) return;
  try {
    const arr = JSON.parse(lsGet(QKEY + '.' + server.user.id) || '[]');
    for (const it of arr) queue.set(it.pid, it);
  } catch { /* ignore */ }
  notify();
  if (queue.size) schedule(200);
}
export function onSaveState(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function notify() {
  saveState.pending = queue.size;
  for (const fn of listeners) fn(saveState);
}
function schedule(ms = 700) {
  clearTimeout(timer);
  timer = setTimeout(flush, ms);
}
export function queueSave(items, schemeVersion) {
  for (const it of items) queue.set(it.pid, { ...it, v: schemeVersion });
  persist();
  notify();
  schedule();
}
export async function flush() {
  if (inflight || !queue.size || !server.user) return;
  inflight = true;
  const batch = [...queue.values()].slice(0, 500);
  try {
    await call('save', { body: { items: batch, scheme_version: batch[0].v } });
    for (const it of batch) if (queue.get(it.pid) === it) queue.delete(it.pid);
    saveState.error = null;
    saveState.savedTotal += batch.length;
  } catch (e) {
    saveState.error = e.status === 401 ? 'Signed out - sign in again to save' : e.message;
    inflight = false;
    persist(); notify();
    if (e.status !== 401) schedule(8000);
    return;
  }
  inflight = false;
  persist(); notify();
  if (queue.size) schedule(100);
}
window.addEventListener('beforeunload', (e) => {
  if (queue.size) { flush(); e.preventDefault(); e.returnValue = ''; }
});
window.addEventListener('online', () => schedule(100));
