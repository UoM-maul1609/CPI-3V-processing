// Session files: classifications made in practice mode, saved to a CSV the
// person keeps, and later added to the database (as themselves, or by an
// administrator on their behalf) with details such as level of expertise.
//
// File format (cpi-session-v1): '#' lines with the details, then a table.
//   # cpi-session-v1
//   # name: Sam Student
//   # expertise: Some experience
//   ...
//   pid,flight,mat_file,index,pipeline_id,class,class_name,flags,classified_utc
// The reader also accepts the older practice download and the site's own
// export (columns pid, label, flags).

import { sourceId } from './export.js';

let C;                 // context from app.js: h, $, toast, server, api, fmtInt, practice, getScheme, getData, refresh
export function initSessions(ctx) { C = ctx; }

const META_KEY = 'cpi.practiceMeta';
const FIELDS = ['name', 'expertise', 'affiliation', 'notes'];
const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } };
const plural = (n, w) => `${C.fmtInt(n)} ${w}${n === 1 ? '' : 's'}`;
const iso = (t) => (t ? new Date(t * 1000).toISOString().replace(/\.\d+Z$/, 'Z') : '');
const levels = () => C.server.expertiseLevels?.length ? C.server.expertiseLevels : ['New to ice crystal images', 'Some experience', 'Experienced', 'Expert'];

export const practiceMeta = {
  load() { return Object.assign({ name: '', expertise: '', affiliation: '', notes: '', started: 0, saved: 0, savedCount: 0 }, lsGet(META_KEY, {})); },
  save(m) { lsSet(META_KEY, m); },
};

// ------------------------------------------------------------------ CSV
const cell = (v) => { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
const oneLine = (s) => String(s ?? '').replace(/\s*[\r\n]+\s*/g, ' / ').trim();

/** The practice classifications in this browser, as a session file. */
export function buildSessionCsv() {
  const st = C.practice.load();
  const m = practiceMeta.load();
  const scheme = C.getScheme();
  const D = C.getData();
  const name = new Map(scheme.classes.map((c) => [c.id, c.name]));
  const entries = Object.entries(st).sort((a, b) => (a[1][2] || 0) - (b[1][2] || 0));
  const times = entries.map(([, v]) => v[2]).filter(Boolean);
  const head = [
    'cpi-session-v1',
    `name: ${oneLine(m.name)}`,
    `expertise: ${oneLine(m.expertise)}`,
    `affiliation: ${oneLine(m.affiliation)}`,
    `notes: ${oneLine(m.notes)}`,
    `started_utc: ${iso(times.length ? Math.min(...times) : m.started)}`,
    `saved_utc: ${iso(Date.now() / 1000)}`,
    `site: ${location.origin}${location.pathname}`,
    `scheme_version: ${scheme.version || ''}`,
    `classes: ${scheme.classes.map((c) => `${c.id}=${c.name}`).join('; ')}`,
    `count: ${entries.length}`,
  ].map((l) => `# ${l}`);
  const rows = [['pid', 'flight', 'mat_file', 'index', 'pipeline_id', 'class', 'class_name', 'flags', 'classified_utc']];
  for (const [pid, [lab, flags, t]] of entries) {
    const [f, mat, i] = pid.split(':');
    const g = D?.index.get(pid);
    rows.push([pid, f, mat, i, g === undefined ? '' : sourceId(D, g), lab, name.get(lab) || '', flags || '', iso(t)]);
  }
  return head.join('\n') + '\n' + rows.map((r) => r.map(cell).join(',')).join('\n') + '\n';
}

function fileName(m) {
  const who = (m.name || 'practice').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'practice';
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `cpi-session-${who}-${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}.csv`;
}

function saveFile() {
  const m = practiceMeta.load();
  const blob = new Blob([buildSessionCsv()], { type: 'text/csv' });
  const a = C.h('a', { href: URL.createObjectURL(blob), download: fileName(m) });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  m.saved = Date.now(); m.savedCount = C.practice.count();
  practiceMeta.save(m);
  C.onSaved?.();
}

/** Parse a session file (or an older practice / export CSV). */
export function parseSessionCsv(text) {
  const meta = {};
  const lines = text.replace(/^﻿/, '').split(/\r?\n/);
  let k = 0;
  for (; k < lines.length; k++) {
    const l = lines[k];
    if (!l.startsWith('#')) break;
    const m = /^#\s*([a-z_]+)\s*:\s?(.*)$/i.exec(l);
    if (m) meta[m[1].toLowerCase()] = m[2].trim();
    else if (/cpi-session-v\d/.test(l)) meta.format = l.replace(/^#\s*/, '').trim();
  }
  const rows = csvRows(lines.slice(k).join('\n'));
  if (!rows.length) throw new Error('The file is empty.');
  const hdr = rows[0].map((s) => s.trim().toLowerCase());
  const col = (...names) => names.map((n) => hdr.indexOf(n)).find((i) => i >= 0) ?? -1;
  const iPid = col('pid'), iLab = col('class', 'label'), iFl = col('flags'), iT = col('classified_utc', 'updated_utc', 'created_utc');
  if (iPid < 0 || iLab < 0) throw new Error('This doesn’t look like a classification file: it needs “pid” and “class” columns.');
  const items = [];
  const seen = new Map();
  let bad = 0;
  for (const r of rows.slice(1)) {
    if (r.length === 1 && r[0] === '') continue;
    const pid = (r[iPid] || '').trim(), label = (r[iLab] || '').trim();
    if (!/^[A-Za-z0-9._-]+:[A-Za-z0-9._-]+:\d+$/.test(pid) || !label) { bad++; continue; }
    const t = iT >= 0 && r[iT] ? Date.parse(r[iT].replace(' ', 'T') + (/[zZ]|[+-]\d\d:?\d\d$/.test(r[iT]) ? '' : 'Z')) / 1000 : null;
    const it = { pid, label, flags: iFl >= 0 && r[iFl] ? r[iFl].split(/[;,]/).map((s) => s.trim()).filter(Boolean) : [], t: Number.isFinite(t) ? Math.round(t) : null };
    if (seen.has(pid)) items[seen.get(pid)] = it; else { seen.set(pid, items.length); items.push(it); }
  }
  // the site's own export has a person per row: use the first name if it is the only one
  const iName = col('name');
  if (!meta.name && iName >= 0) {
    const names = new Set(rows.slice(1).map((r) => r[iName]).filter(Boolean));
    if (names.size === 1) meta.name = [...names][0];
    else if (names.size > 1) meta.several_people = [...names].slice(0, 5).join(', ') + (names.size > 5 ? ', …' : '');
  }
  return { meta, items, bad };
}
function csvRows(s) {
  const out = []; let row = [], f = '', q = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (q) {
      if (ch === '"') { if (s[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += ch;
    } else if (ch === '"') q = true;
    else if (ch === ',') { row.push(f); f = ''; }
    else if (ch === '\n' || ch === '\r') { if (ch === '\r' && s[i + 1] === '\n') i++; row.push(f); out.push(row); row = []; f = ''; }
    else f += ch;
  }
  if (f !== '' || row.length) { row.push(f); out.push(row); }
  return out;
}

// ------------------------------------------------------------------ shared form bits
function detailsFields(m, opts = {}) {
  const { h } = C;
  const name = h('input', { type: 'text', maxlength: 80, value: m.name || '', placeholder: opts.namePh || 'e.g. Sam Student', autocomplete: 'name' });
  const exp = h('select', {}, h('option', { value: '' }, '(not given)'), levels().map((l) => h('option', { value: l, selected: l === m.expertise }, l)));
  if (m.expertise && !levels().includes(m.expertise)) exp.append(h('option', { value: m.expertise, selected: true }, m.expertise));
  const aff = h('input', { type: 'text', maxlength: 120, value: m.affiliation || '', placeholder: 'e.g. University of Manchester' });
  const notes = h('textarea', { rows: 2, maxlength: 1000, placeholder: opts.notesPh || 'Anything that helps later, e.g. “first go”, “used the guide”' }, m.notes || '');
  const els = { name, expertise: exp, affiliation: aff, notes };
  const wrap = h('div', { class: 'detailgrid' },
    h('label', { class: 'fld' }, h('span', {}, 'Name'), name),
    h('label', { class: 'fld' }, h('span', {}, 'Level of expertise'), exp),
    h('label', { class: 'fld' }, h('span', {}, 'Affiliation ', h('small', {}, '(optional)')), aff),
    h('label', { class: 'fld wide' }, h('span', {}, 'Notes ', h('small', {}, '(optional)')), notes));
  return { wrap, els, values: () => Object.fromEntries(FIELDS.map((k) => [k, els[k].value.trim()])) };
}
function dialog(cls, ...kids) {
  const d = C.h('dialog', { class: 'dlg ' + cls });
  d.append(C.h('div', { class: 'dlg-inner' }, ...kids));
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
  return d;
}

// ------------------------------------------------------------------ download (practice)
/** Ask for the person's details (once), then save the file. */
export function openDownload() {
  const { h } = C;
  const n = C.practice.count();
  if (!n) { C.toast('Nothing to save yet: classify some particles first.'); return; }
  const m = practiceMeta.load();
  const f = detailsFields(m);
  const err = h('p', { class: 'formerr', role: 'alert' });
  const d = dialog('sessdlg',
    h('h3', {}, 'Save your work to a file'),
    h('p', { class: 'dlg-why' }, `Your ${plural(n, 'classification')} are only kept in this browser. Save them to a file so they can’t be lost; the file can be added to the database later, by you once you have an account, or by the administrator.`),
    h('form', { onsubmit: (e) => {
      e.preventDefault();
      const v = f.values();
      if (!v.name) { err.textContent = 'Please give your name, so your work is credited to you.'; f.els.name.focus(); return; }
      Object.assign(m, v); practiceMeta.save(m);
      saveFile();
      d.close();
      C.toast(`Saved ${plural(n, 'classification')} to a file. Keep it somewhere safe.`, 3500);
    } },
    f.wrap, err,
    h('div', { class: 'dlg-actions' },
      h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'),
      h('button', { type: 'submit', class: 'btn primary' }, 'Save file'))));
  (m.name ? d.querySelector('button[type=submit]') : f.els.name).focus();
}

/** Unsaved practice work since the last file was saved. */
export function unsavedCount() {
  const m = practiceMeta.load();
  const st = C.practice.load();
  let n = 0;
  for (const v of Object.values(st)) if (!m.saved || !v[2] || v[2] * 1000 > m.saved) n++;
  return n;
}

// ------------------------------------------------------------------ upload
/** Progress page card: add classifications from a file (signed in only). */
export function uploadCard(onDone) {
  const { h } = C;
  const input = h('input', { type: 'file', accept: '.csv,text/csv', hidden: true, onchange: () => input.files[0] && read(input.files[0]) });
  const zone = h('div', { class: 'dropzone', tabindex: 0, role: 'button',
    onclick: () => input.click(), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } },
    ondragover: (e) => { e.preventDefault(); zone.classList.add('over'); },
    ondragleave: () => zone.classList.remove('over'),
    ondrop: (e) => { e.preventDefault(); zone.classList.remove('over'); const fl = e.dataTransfer.files[0]; if (fl) read(fl); } },
  h('svg', { class: 'ico', viewBox: '0 0 24 24', width: 28, height: 28, html: '<path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>' }),
  h('b', {}, 'Drop a session file here, or click to choose one'),
  h('span', { class: 'muted small' }, C.server.user?.is_admin
    ? 'Practice work someone saved to a file (cpi-session-….csv). You can add it as your own, or on their behalf.'
    : 'Practice work you saved to a file before you had an account (cpi-session-….csv).'));
  async function read(file) {
    input.value = '';
    let parsed;
    try {
      if (file.size > 20e6) throw new Error('That file is too big for a session file.');
      parsed = parseSessionCsv(await file.text());
      if (!parsed.items.length) throw new Error('No classifications found in that file.');
    } catch (e) { C.toast(e.message, 4000); return; }
    openUploadPreview(file.name, parsed, onDone);
  }
  return h('div', { class: 'card uploadcard' },
    h('h2', {}, 'Add classifications from a file'),
    zone, input);
}

async function openUploadPreview(fileName, parsed, onDone) {
  const { h, server, api } = C;
  const scheme = C.getScheme();
  const D = C.getData();
  const me = server.user;
  const known = new Map(scheme.classes.map((c) => [c.id, c]));
  const items = parsed.items.filter((it) => known.has(it.label));
  const unknown = new Map();
  for (const it of parsed.items) if (!known.has(it.label)) unknown.set(it.label, (unknown.get(it.label) || 0) + 1);
  const notHere = D ? items.filter((it) => !D.index.has(it.pid)).length : 0;
  const by = new Map();
  for (const it of items) by.set(it.label, (by.get(it.label) || 0) + 1);
  const max = Math.max(1, ...by.values());

  // people (admins can add work on someone else's behalf)
  let people = [];
  if (me.is_admin) { try { people = (await api.adminGet('users')).users || []; } catch { /* only "mine" then */ } }
  const others = people.filter((p) => p.id !== me.id);
  const match = others.find((p) => parsed.meta.name && p.display_name.toLowerCase() === parsed.meta.name.toLowerCase());
  const looksLikeMe = !parsed.meta.name || parsed.meta.name.toLowerCase() === me.display_name.toLowerCase();

  const f = detailsFields({ name: parsed.meta.name || '', expertise: parsed.meta.expertise || '', affiliation: parsed.meta.affiliation || '', notes: parsed.meta.notes || '' },
    { namePh: 'Who made these classifications' });
  let target = me.is_admin ? (looksLikeMe ? 'me' : match ? 'existing' : 'new') : 'me';
  const pick = h('select', { 'aria-label': 'Person' }, others.map((p) => h('option', { value: p.id, selected: p === match },
    `${p.display_name} (${p.is_external ? 'no login' : p.username})`)));
  const seg = h('div', { class: 'whoopts', role: 'radiogroup' });
  const opt = (val, title, sub, extra) => h('label', { class: 'whoopt' },
    h('input', { type: 'radio', name: 'who', value: val, checked: target === val, onchange: () => { target = val; sync(); } }),
    h('span', {}, h('b', {}, title), h('small', {}, sub)), extra || null);
  seg.append(opt('me', `Mine (${me.display_name})`, 'They’re added to your own classifications.'));
  if (me.is_admin) {
    seg.append(opt('new', 'Someone without an account', 'Kept under their name, with no login. You can give them a password later on the Admin page.'));
    if (others.length) seg.append(opt('existing', 'Someone already listed', 'Added to that person’s classifications.', pick));
  }
  const replace = h('input', { type: 'checkbox' });
  const replaceLbl = h('span');
  const err = h('p', { class: 'formerr', role: 'alert' });
  const go = h('button', { type: 'submit', class: 'btn primary' });
  function sync() {
    pick.disabled = target !== 'existing';
    f.els.name.closest('.fld').classList.toggle('dim', target === 'me');
    const who = target === 'me' ? 'you' : 'they';
    replaceLbl.textContent = `Where ${who} already classified a particle, use the class in this file (otherwise ${target === 'me' ? 'your' : 'their'} existing one is kept)`;
    go.textContent = `Add ${plural(items.length, 'classification')}`;
  }
  sync();

  const metaNote = [parsed.meta.saved_utc && `saved ${parsed.meta.saved_utc.replace('T', ' ').replace('Z', ' UTC')}`,
    parsed.meta.site && `from ${parsed.meta.site}`, parsed.meta.scheme_version && `classes “${parsed.meta.scheme_version}”`].filter(Boolean).join(' · ');
  const warn = [];
  if (unknown.size) warn.push(`${plural([...unknown.values()].reduce((a, b) => a + b, 0), 'classification')} use classes this site doesn’t have (${[...unknown.keys()].slice(0, 6).join(', ')}) and will be skipped.`);
  if (notHere) warn.push(`${plural(notHere, 'particle')} ${notHere === 1 ? 'is' : 'are'} from flights not imported here; ${notHere === 1 ? 'it' : 'they'}’ll be stored and show up once the flight is imported.`);
  if (parsed.bad) warn.push(`${plural(parsed.bad, 'row')} couldn’t be read and will be skipped.`);
  if (parsed.meta.several_people) warn.push(`This file has classifications from several people (${parsed.meta.several_people}); they will all be added under one name.`);

  const d = dialog('sessdlg uploaddlg',
    h('h3', {}, 'Add classifications from a file'),
    h('p', { class: 'dlg-why' }, h('b', {}, fileName), metaNote ? ` · ${metaNote}` : ''),
    h('form', { onsubmit: async (e) => {
      e.preventDefault();
      err.textContent = '';
      const meta = Object.assign(f.values(), { file_name: fileName, scheme_version: parsed.meta.scheme_version || '', started_utc: parsed.meta.started_utc || '' });
      if (target !== 'me' && !meta.name) { err.textContent = 'Give the name of the person who made these classifications.'; f.els.name.focus(); return; }
      go.disabled = true; go.textContent = 'Adding…';
      try {
        const r = await api.uploadSession({ meta, target: target === 'existing' ? Number(pick.value) : target, replace: replace.checked, items });
        d.close();
        const kept = r.kept_existing ? `; ${plural(r.kept_existing, 'particle')} already classified, kept as ${target === 'me' ? 'you' : 'they'} had ${r.kept_existing === 1 ? 'it' : 'them'}` : '';
        C.toast(`Added ${plural(r.saved, 'classification')} for ${r.contributor}${kept}`, 5000);
        await onDone?.();
      } catch (ex) { err.textContent = ex.message; go.disabled = false; sync(); }
    } },
    h('div', { class: 'upsum' },
      h('div', { class: 'upnum' }, h('b', {}, C.fmtInt(items.length)), h('span', {}, items.length === 1 ? 'classification' : 'classifications')),
      h('div', { class: 'hbars' }, scheme.classes.filter((c) => by.has(c.id)).map((c) => h('div', { class: 'hbar' },
        h('span', {}, c.name), h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: (100 * by.get(c.id) / max) + '%', background: c.color } })),
        h('span', { class: 'v' }, C.fmtInt(by.get(c.id))))))),
    warn.length ? h('ul', { class: 'upwarn' }, warn.map((w) => h('li', {}, w))) : null,
    h('h4', { class: 'subh' }, 'Whose work is this?'), seg,
    h('h4', { class: 'subh' }, 'About the person ', h('small', { class: 'muted' }, 'from the file; change anything that’s wrong')),
    f.wrap,
    h('label', { class: 'check' }, replace, replaceLbl),
    err,
    h('div', { class: 'dlg-actions sticky' },
      h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'), go)));
  if (!items.length) { go.disabled = true; err.textContent = 'None of the classes in this file exist on this site.'; }
}

// ------------------------------------------------------------------ list of uploaded sessions
export async function sessionsCard(onChange) {
  const { h, api, server } = C;
  let list = [];
  try { list = (await api.fetchSessions()).sessions || []; } catch { return null; }
  if (!list.length) return null;
  const tb = h('tbody');
  for (const s of list) {
    const del = h('button', { class: 'btn small ghost danger', onclick: async () => {
      if (!confirm(`Remove the ${plural(+s.n_current, 'classification')} added from “${s.file_name || 'this file'}” for ${s.contributor}?\n\nClassifications changed by hand since then are kept.`)) return;
      del.disabled = true;
      try { const r = await api.deleteSession(s.id); C.toast(`Removed ${plural(r.removed, 'classification')}`); await onChange?.(); }
      catch (e) { C.toast(e.message, 4000); del.disabled = false; }
    } }, 'Remove');
    tb.append(h('tr', {},
      h('td', {}, h('b', {}, s.contributor), +s.is_external ? h('span', { class: 'pill tiny', title: 'Added from a file; this person has no login' }, 'no login') : null,
        h('div', { class: 'muted small' }, [s.expertise, s.affiliation].filter(Boolean).join(' · '))),
      h('td', { class: 'n' }, C.fmtInt(+s.n_current), +s.n_current !== +s.n_labels ? h('div', { class: 'muted small' }, `of ${C.fmtInt(+s.n_labels)} added`) : null),
      h('td', {}, h('div', { class: 'ellip', title: s.file_name || '' }, s.file_name || '—'), s.notes ? h('div', { class: 'muted small ellip', title: s.notes }, s.notes) : null),
      h('td', { class: 'muted small' }, (s.created_at || '').slice(0, 16).replace(' ', ' at '), server.user?.is_admin && s.uploaded_by !== s.contributor_id ? h('div', {}, `by ${s.uploader}`) : null),
      h('td', { class: 'n' }, del)));
  }
  return h('div', { class: 'card' },
    h('h2', {}, server.user?.is_admin ? 'Sessions added from files' : 'Your sessions added from files'),
    h('div', { class: 'tablewrap' }, h('table', { class: 'data sesstbl' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Person'), h('th', { class: 'n' }, 'Classifications'), h('th', {}, 'File'), h('th', {}, 'Added'), h('th', {}))),
      tb)));
}

// ------------------------------------------------------------------ your details (signed in)
export function openProfile() {
  const { h, server, api } = C;
  const u = server.user;
  const f = detailsFields({ name: u.display_name, expertise: u.expertise || '', affiliation: u.affiliation || '' });
  f.els.notes.closest('.fld').remove();
  const err = h('p', { class: 'formerr', role: 'alert' });
  const d = dialog('sessdlg',
    h('h3', {}, 'Your details'),
    h('p', { class: 'dlg-why' }, 'Your level of expertise is stored with your classifications, so they can be compared later.'),
    h('form', { onsubmit: async (e) => {
      e.preventDefault();
      const v = f.values();
      if (!v.name) { err.textContent = 'Give your name.'; return; }
      try {
        const r = await api.updateProfile({ display_name: v.name, expertise: v.expertise, affiliation: v.affiliation });
        server.user = r.user; C.onProfile?.();
        d.close(); C.toast('Details saved');
      } catch (ex) { err.textContent = ex.message; }
    } }, f.wrap, err,
    h('div', { class: 'dlg-actions' },
      h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'),
      h('button', { type: 'submit', class: 'btn primary' }, 'Save'))));
  f.els.expertise.focus();
}
