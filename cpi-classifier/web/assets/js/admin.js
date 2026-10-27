// Admin page: import flights from MAT files that live on the server, watch
// imports run, and hide or remove flights. Only shown to administrators.

let C = null;          // context from app.js: {h, $, $$, toast, fmtInt, server, api, display, onDataChanged}
let pollTimer = null;
let lastJobs = new Map();
let state = { folders: [], root: '', flights: [], jobs: [], filter: '' };

export async function viewAdmin(root, ctx) {
  C = ctx;
  const { h, server } = C;
  root.className = 'page'; root.style.display = '';
  root.innerHTML = '';
  const inner = h('div', { class: 'page-inner admin' });
  root.append(inner);
  if (!server.user || !server.user.is_admin) {
    inner.append(h('h1', {}, 'Admin'), h('p', { class: 'page-lede' }, 'Only administrators can manage people, classes and flights. Sign in with an administrator account.'));
    return;
  }
  const tab = location.hash.split('/')[2] || 'flights';
  const tabs = [['flights', 'Flights and imports'], ['people', 'People'], ['classes', 'Classes']];
  inner.append(h('h1', {}, 'Admin'),
    h('nav', { class: 'admintabs', 'aria-label': 'Admin sections' }, tabs.map(([k, l]) =>
      h('a', { href: `#/admin/${k}`, 'aria-current': k === tab ? 'page' : null }, l))));
  if (tab === 'people') return peopleView(inner);
  if (tab === 'classes') return classesView(inner);
  inner.append(
    h('p', { class: 'page-lede', id: 'adminLede' }, 'Import a folder of MAT files once to make its particles available for classifying. Importing reads the MAT files and saves small images and a table of particle properties for the page; the MAT files themselves are not changed.'),
    h('section', { id: 'checkBox' }),
    h('section', { id: 'jobsBox' }),
    h('section', { id: 'foldersBox' }),
    h('section', { id: 'flightsBox' }));
  await Promise.all([loadChecks(), loadFolders(), loadFlights(), loadJobs()]);
}

// ================================================================== people
async function peopleView(inner) {
  const { h } = C;
  const box = h('section', {});
  inner.append(
    h('div', { class: 'sec-h' },
      h('p', { class: 'page-lede', style: { margin: 0 } }, 'Add the people who will classify. Each gets a temporary password and chooses their own the first time they sign in. Their classifications are saved under their name.'),
      h('span', { class: 'grow' }),
      h('button', { class: 'btn primary', onclick: () => addPersonDialog(() => loadPeople(box)) }, 'Add a person')),
    box);
  await loadPeople(box);
}
async function loadPeople(box) {
  const { h, fmtInt } = C;
  let d;
  try { d = await C.api.adminGet('users'); } catch (e) { box.innerHTML = ''; box.append(h('p', { class: 'formerr' }, e.message)); return; }
  const tbody = h('tbody');
  for (const u of d.users) {
    const me = u.id === d.me;
    const admin = u.is_admin || u.admin_in_config;
    const ext = !!u.is_external;
    const levels = d.expertise_levels || [];
    const exSel = h('select', { class: 'exsel', 'aria-label': `Level of expertise for ${u.display_name}`, onchange: async () => {
      try { await C.api.adminPost('user_update', { id: u.id, expertise: exSel.value }); C.toast(`Saved: ${u.display_name}, ${exSel.value || 'expertise not given'}`); }
      catch (e) { C.toast(e.message, 4000); }
    } }, h('option', { value: '' }, '—'), levels.map((l) => h('option', { value: l, selected: l === u.expertise }, l)));
    if (u.expertise && !levels.includes(u.expertise)) exSel.append(h('option', { value: u.expertise, selected: true }, u.expertise));
    const status = ext ? h('span', { class: 'jstat', title: 'Their classifications were added from a session file. They can’t sign in until you give them a password.' }, 'No login')
      : u.disabled ? h('span', { class: 'jstat failed' }, 'Switched off')
      : u.must_change_pw ? h('span', { class: 'jstat queued', title: 'They haven’t signed in and chosen their own password yet' }, 'Waiting for first sign-in')
      : h('span', { class: 'jstat done' }, 'Active');
    tbody.append(h('tr', { class: u.disabled ? 'is-hidden' : '' },
      h('td', {}, h('b', {}, u.display_name), me ? h('span', { class: 'pill tiny' }, 'you') : null, ext ? h('span', { class: 'pill tiny' }, 'from file') : null,
        u.email || u.affiliation ? h('div', { class: 'muted small' }, [u.affiliation, u.email].filter(Boolean).join(' · ')) : null),
      h('td', { class: 'mono' }, ext ? h('span', { class: 'muted' }, '—') : u.username),
      h('td', {}, exSel),
      h('td', {}, h('div', {}, ext ? 'Contributor' : admin ? 'Administrator' : 'Classifier'), h('div', { class: 'rolestat' }, status),
        u.admin_in_config ? h('div', { class: 'muted small' }, 'admin set in config.php') : null),
      h('td', { class: 'n' }, fmtInt(u.n_labels)),
      h('td', { class: 'num nowrap' }, u.last_login ? u.last_login.slice(0, 10) : h('span', { class: 'muted' }, 'never')),
      h('td', { class: 'actions' },
        h('button', { class: 'btn small', title: ext ? 'Give them a username and temporary password so they can sign in and carry on' : 'Give them a new temporary password (e.g. if they’ve forgotten theirs)', onclick: async () => {
          if (ext) { giveLoginDialog(u, () => loadPeople(box)); return; }
          if (!confirm(`Give ${u.display_name} a new temporary password? Their current password stops working.`)) return;
          try { const r = await C.api.adminPost('user_reset', { id: u.id }); credentialsDialog(r, false); loadPeople(box); } catch (e) { C.toast(e.message, 4000); }
        } }, ext ? 'Give a login' : 'New password'),
        !me && !ext && !u.admin_in_config ? h('button', { class: 'btn small ghost', onclick: async () => {
          try { await C.api.adminPost('user_update', { id: u.id, is_admin: !u.is_admin }); loadPeople(box); } catch (e) { C.toast(e.message, 4000); }
        } }, u.is_admin ? 'Remove admin' : 'Make admin') : null,
        !me && !ext ? h('button', { class: 'btn small ghost' + (u.disabled ? '' : ' danger'), title: u.disabled ? 'Let them sign in again' : 'Stop them signing in. Their classifications are kept.', onclick: async () => {
          try { await C.api.adminPost('user_update', { id: u.id, disabled: !u.disabled }); loadPeople(box); C.toast(u.disabled ? `${u.display_name} can sign in again` : `${u.display_name} is switched off`); } catch (e) { C.toast(e.message, 4000); }
        } }, u.disabled ? 'Switch on' : 'Switch off') : null)));
  }
  box.innerHTML = '';
  box.append(h('div', { class: 'tablewrap' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'Name'), h('th', {}, 'Username'), h('th', {}, 'Expertise'), h('th', {}, 'Role'), h('th', { class: 'n' }, 'Classified'), h('th', {}, 'Last sign-in'), h('th', {}, ''))), tbody)));
}
/** Someone known only from an uploaded session file: give them a username and password. */
function giveLoginDialog(u, done) {
  const { h } = C;
  const d = h('dialog', { class: 'dlg' });
  const err = h('p', { class: 'formerr', role: 'alert' });
  const user = h('input', { required: true, autocomplete: 'off', value: suggestUsername(u.display_name) });
  d.append(h('form', { class: 'dlg-inner', onsubmit: async (e) => {
    e.preventDefault(); err.textContent = '';
    try {
      const r = await C.api.adminPost('user_update', { id: u.id, username: user.value.trim().toLowerCase() });
      const p = await C.api.adminPost('user_reset', { id: u.id });
      d.close(); credentialsDialog(Object.assign({ username: r.username }, p), true); done();
    } catch (ex) { err.textContent = ex.message; }
  } },
  h('h3', {}, `Give ${u.display_name} a login`),
  h('p', { class: 'dlg-why' }, 'They’ll get a temporary password to change the first time they sign in. Everything already added from their session files stays theirs.'),
  h('label', { class: 'fld' }, h('span', {}, 'Username ', h('small', {}, 'what they type to sign in')), user),
  err,
  h('div', { class: 'dlg-actions' },
    h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'),
    h('button', { type: 'submit', class: 'btn primary' }, 'Create login'))));
  d.addEventListener('close', () => d.remove());
  document.body.append(d); d.showModal(); user.select();
}
function suggestUsername(name) {
  const parts = name.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').split(/[^a-z0-9]+/).filter(Boolean);
  if (!parts.length) return '';
  return (parts.length > 1 ? parts[0][0] + parts[parts.length - 1] : parts[0]).slice(0, 32);
}
function addPersonDialog(done) {
  const { h } = C;
  const d = h('dialog', { class: 'dlg' });
  const err = h('p', { class: 'formerr', role: 'alert' });
  const name = h('input', { name: 'display_name', required: true, autocomplete: 'off', placeholder: 'e.g. Jo Smith' });
  const user = h('input', { name: 'username', required: true, autocomplete: 'off', placeholder: 'e.g. jsmith' });
  const email = h('input', { name: 'email', type: 'email', autocomplete: 'off' });
  const admin = h('input', { type: 'checkbox' });
  let touched = false;
  user.addEventListener('input', () => { touched = true; });
  name.addEventListener('input', () => { if (!touched) user.value = suggestUsername(name.value); });
  d.append(h('form', { class: 'dlg-inner', onsubmit: async (e) => {
    e.preventDefault(); err.textContent = '';
    try {
      const r = await C.api.adminPost('user_add', { display_name: name.value.trim(), username: user.value.trim(), email: email.value.trim(), is_admin: admin.checked });
      d.close(); credentialsDialog(r, true); done && done();
    } catch (x) { err.textContent = x.message; }
  } },
  h('h3', {}, 'Add a person'),
  h('p', { class: 'dlg-why' }, 'They’ll get a temporary password to give them. They choose their own the first time they sign in.'),
  h('label', { class: 'fld' }, h('span', {}, 'Name'), name),
  h('label', { class: 'fld' }, h('span', {}, 'Username ', h('small', {}, 'what they type to sign in')), user),
  h('label', { class: 'fld' }, h('span', {}, 'Email ', h('small', {}, '(optional, for your records)')), email),
  h('label', { class: 'check', style: { margin: '4px 0 12px' } }, admin, 'Administrator (can add people, edit classes and import flights)'),
  err,
  h('div', { class: 'dlg-actions' }, h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'), h('button', { class: 'btn primary' }, 'Add'))));
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
  name.focus();
}
/** Shows a username and temporary password once, with copy buttons. */
function credentialsDialog(r, isNew) {
  const { h } = C;
  const site = location.origin + location.pathname;
  const message = `Hello ${r.display_name.split(' ')[0]},\n\nYou can now classify CPI ice crystal images at:\n${site}\n\nUsername: ${r.username}\nTemporary password: ${r.temp_password}\n\nYou'll be asked to choose your own password when you first sign in.`;
  const copy = async (text, btn) => {
    try { await navigator.clipboard.writeText(text); btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = btn.dataset.l; }, 1500); }
    catch { C.toast('Select the text and copy it'); }
  };
  const b1 = h('button', { type: 'button', class: 'btn small', 'data-l': 'Copy password', onclick: (e) => copy(r.temp_password, e.currentTarget) }, 'Copy password');
  const b2 = h('button', { type: 'button', class: 'btn primary', 'data-l': 'Copy message', onclick: (e) => copy(message, e.currentTarget) }, 'Copy message');
  const d = h('dialog', { class: 'dlg creddlg' });
  d.append(h('div', { class: 'dlg-inner' },
    h('h3', {}, isNew ? `${r.display_name} can now sign in` : `New temporary password for ${r.display_name}`),
    h('p', { class: 'dlg-why' }, 'This password is shown only now. Give it to them in person or in a message; they’ll choose their own when they sign in.'),
    h('dl', { class: 'cred' }, h('dt', {}, 'Username'), h('dd', { class: 'mono' }, r.username), h('dt', {}, 'Temporary password'), h('dd', { class: 'mono big' }, r.temp_password, ' ', b1)),
    h('details', {}, h('summary', {}, 'A message you can send them'), h('pre', { class: 'msg' }, message)),
    h('div', { class: 'dlg-actions' }, b2, h('button', { type: 'button', class: 'btn', onclick: () => d.close() }, 'Done'))));
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
}

// ================================================================== classes
const NEW_COLOURS = ['#0f9b9b', '#b5651d', '#6b8e23', '#d6457a', '#3d5a98', '#c9a227', '#7b4fa0', '#2a9d5c', '#e07b39', '#5f7a8a', '#a33c3c', '#4fb3d9'];
const RESERVED = new Set(['z', 'y', 'v', 'o', 'b', 'm', '?', ' ']);
async function classesView(inner) {
  const { h } = C;
  const orig = JSON.parse(JSON.stringify(C.getScheme()));
  let sch = JSON.parse(JSON.stringify(orig));
  let usage = new Map();
  try { const st = await C.api.fetchStats(); usage = new Map(st.by_label.map((r) => [r.label, +r.n])); } catch { /* no stats */ }
  const box = h('div', { class: 'classeditor' });
  inner.append(h('p', { class: 'page-lede' }, 'The classes people sort particles into, and the flags they can add. Change names, colours, keyboard keys and order here. A class’s id never changes once people have used it, so renaming is safe.'), box);
  render();

  function render() {
    box.innerHTML = '';
    const problems = check();
    const preview = h('div', { class: 'palette-classes preview' }, sch.classes.map((c) => h('span', { class: 'cls', style: { '--c': c.color } }, h('kbd', {}, (c.key || '').toUpperCase()), h('span', { class: 'nm' }, c.name || '…'))));
    const row = (kind, c, i) => {
      const list = sch[kind];
      const used = usage.get(c.id) || 0;
      const isNew = !orig[kind].some((o) => o.id === c.id);
      return h('div', { class: 'cerow' + (kind === 'flags' ? ' flag' : '') },
        h('span', { class: 'ce-move' },
          h('button', { class: 'iconbtn sm', 'aria-label': 'Move up', disabled: i === 0, onclick: () => { [list[i - 1], list[i]] = [list[i], list[i - 1]]; render(); } }, '↑'),
          h('button', { class: 'iconbtn sm', 'aria-label': 'Move down', disabled: i === list.length - 1, onclick: () => { [list[i + 1], list[i]] = [list[i], list[i + 1]]; render(); } }, '↓')),
        kind === 'classes' ? h('input', { type: 'color', value: c.color, 'aria-label': 'Colour', onchange: (e) => { c.color = e.target.value; render(); } }) : null,
        h('input', { class: 'ce-key', value: (c.key || '').toUpperCase(), maxlength: '1', 'aria-label': 'Key', title: 'Keyboard key', onchange: (e) => { c.key = e.target.value.toLowerCase(); render(); } }),
        h('span', { class: 'ce-text' },
          h('input', { class: 'ce-name', value: c.name, placeholder: 'Name', 'aria-label': 'Name', onchange: (e) => { c.name = e.target.value; if (isNew) c.id = ''; render(); } }),
          h('input', { class: 'ce-desc', value: c.desc || '', placeholder: 'What it looks like (shown as a tip and in the guide)', 'aria-label': 'Description', onchange: (e) => { c.desc = e.target.value; } })),
        h('span', { class: 'ce-used muted small' }, used ? `${used.toLocaleString()} used` : isNew ? 'new' : 'unused'),
        h('button', { class: 'iconbtn sm danger', 'aria-label': `Remove ${c.name}`, title: 'Remove', onclick: () => {
          if (used && !confirm(`${used} classifications use “${c.name}”. They stay in the database (and in downloads) under the id “${c.id}”, but people can no longer choose it. Remove it?`)) return;
          list.splice(i, 1); render();
        } }, '×'));
    };
    box.append(
      h('div', { class: 'card' }, h('div', { class: 'muted small' }, 'How the classes look on the classify page'), preview),
      h('h2', {}, 'Classes'),
      h('div', { class: 'celist' }, sch.classes.map((c, i) => row('classes', c, i))),
      h('button', { class: 'btn small', onclick: () => {
        const used = new Set([...sch.classes, ...sch.flags].map((x) => x.key));
        const key = '1234567890qwertasdfgjklcxn'.split('').find((k) => !used.has(k)) || '';
        const taken = new Set(sch.classes.map((c) => c.color.toLowerCase()));
        const color = NEW_COLOURS.find((c) => !taken.has(c)) || `#${Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0')}`;
        sch.classes.push({ id: '', name: '', key, color, desc: '' });
        render();
        box.querySelector('.celist .cerow:last-child .ce-name')?.focus();
      } }, '+ Add a class'),
      h('h2', { style: { marginTop: '28px' } }, 'Flags'),
      h('p', { class: 'muted small' }, 'Extra details people can add to any class, e.g. rimed or hollow.'),
      h('div', { class: 'celist' }, sch.flags.map((c, i) => row('flags', c, i))),
      h('button', { class: 'btn small', onclick: () => { sch.flags.push({ id: '', name: '', key: '', desc: '' }); render(); box.querySelector('.celist:last-of-type .cerow:last-child .ce-name')?.focus(); } }, '+ Add a flag'),
      h('h2', { style: { marginTop: '28px' } }, 'Guidance for people classifying'),
      h('textarea', { class: 'ce-help', rows: '3', onchange: (e) => { sch.help = e.target.value; } }, sch.help || ''),
      problems.length ? h('ul', { class: 'formerr' }, problems.map((p) => h('li', {}, p))) : null,
      h('div', { class: 'dlg-actions sticky ce-actions' },
        h('button', { class: 'btn ghost', onclick: async () => {
          if (!confirm('Go back to the classes that came with the classifier? Your edited list is removed.')) return;
          try { await C.api.adminPost('scheme_reset', {}); const s2 = await C.api.fetchScheme(); if (s2) { C.setScheme(s2); sch = JSON.parse(JSON.stringify(s2)); } C.toast('Default classes restored'); render(); } catch (e) { C.toast(e.message, 4000); }
        } }, 'Restore the defaults'),
        h('span', { class: 'grow' }),
        h('button', { class: 'btn', onclick: () => { sch = JSON.parse(JSON.stringify(orig)); render(); } }, 'Undo changes'),
        h('button', { class: 'btn primary', disabled: problems.length > 0, onclick: async () => {
          try {
            const r = await C.api.adminPost('scheme_save', { scheme: sch });
            C.setScheme(r.scheme);
            Object.assign(orig, JSON.parse(JSON.stringify(r.scheme)));
            sch = JSON.parse(JSON.stringify(r.scheme));
            C.toast('Classes saved. Everyone sees them the next time they open the page.', 3500);
            render();
          } catch (e) { C.toast(e.message, 5000); }
        } }, 'Save classes')));
  }
  function check() {
    const p = [];
    const keys = new Map();
    for (const kind of ['classes', 'flags']) for (const c of sch[kind]) {
      if (!c.name.trim()) p.push(`A ${kind === 'classes' ? 'class' : 'flag'} has no name`);
      const k = (c.key || '').toLowerCase();
      if (k && RESERVED.has(k)) p.push(`${c.name || 'A class'}: the key ${k.toUpperCase()} is used by the page (undo, view, outline…)`);
      if (k && keys.has(k)) p.push(`The key ${k.toUpperCase()} is used by both ${keys.get(k)} and ${c.name}`);
      if (k) keys.set(k, c.name);
    }
    if (sch.classes.length < 2) p.push('Keep at least two classes');
    return p;
  }
}

export function leaveAdmin() { clearTimeout(pollTimer); pollTimer = null; }

// ------------------------------------------------------------------ server check
async function loadChecks() {
  const { h, $ } = C;
  const box = $('#checkBox'); if (!box) return;
  let checks;
  try { checks = (await C.api.adminGet('check')).checks; } catch (e) { box.innerHTML = ''; box.append(h('div', { class: 'card warn' }, e.message)); return; }
  const bad = checks.filter((c) => !c.ok && !c.optional);
  const items = h('ul', { class: 'checks' }, checks.map((c) => h('li', { class: c.ok ? 'ok' : (c.optional ? 'opt' : 'bad') },
    h('span', { class: 'ck', 'aria-hidden': 'true' }, c.ok ? '✓' : c.optional ? '–' : '!'),
    h('b', {}, c.name), h('span', { class: 'muted' }, c.detail))));
  box.innerHTML = '';
  if (bad.length) {
    box.append(h('div', { class: 'card warn' }, h('h2', {}, 'The server needs setting up'),
      h('p', {}, 'Fix the items marked ! (usually in api/config.php or folder permissions), then reload this page.'), items));
  } else {
    const py = checks.find((c) => c.name === 'Python');
    box.append(h('details', { class: 'checkline' },
      h('summary', {}, h('span', { class: 'ck ok' }, '✓'), ' Server ready to import', h('span', { class: 'muted' }, ` (${py ? py.detail : ''})`)), items));
  }
}

// ------------------------------------------------------------------ jobs
const STAGE = { loading: 'Reading', images: 'Making images', navigation: 'Adding position and temperature' };
async function loadJobs() {
  let jobs = [];
  try { jobs = (await C.api.adminGet('jobs')).jobs; } catch (e) { /* shown elsewhere */ }
  // a job that has just finished: refresh everything
  let finished = false;
  for (const j of jobs) {
    const prev = lastJobs.get(j.id);
    if (prev && prev !== j.status && (j.status === 'done' || j.status === 'failed')) {
      finished = true;
      if (j.status === 'done') C.toast(`Imported ${j.flight}. Its particles are ready to classify.`, 4000);
      else C.toast(`Import of ${j.flight} failed - see the log`, 5000);
    }
  }
  lastJobs = new Map(jobs.map((j) => [j.id, j.status]));
  state.jobs = jobs;
  renderJobs();
  if (finished) { await C.onDataChanged(); await Promise.all([loadFolders(), loadFlights()]); }
  clearTimeout(pollTimer);
  if (jobs.some((j) => j.status === 'running' || j.status === 'queued')) pollTimer = setTimeout(() => { if (document.getElementById('jobsBox')) loadJobs(); }, 1500);
}
function renderJobs() {
  const { h, $, fmtInt } = C;
  const box = $('#jobsBox'); if (!box) return;
  box.innerHTML = '';
  const recent = state.jobs.slice(0, 8);
  if (!recent.length) return;
  box.append(h('h2', {}, 'Imports'));
  const list = h('div', { class: 'jobs' });
  for (const j of recent) {
    const p = j.progress || {};
    let pct = 0, what = '';
    if (j.status === 'queued') what = 'Waiting for the current import to finish';
    else if (j.status === 'running') {
      const files = p.files || 1, f = (p.file || 1) - 1;
      const within = p.stage === 'images' && p.total ? p.done / p.total : p.stage === 'navigation' ? 1 : 0;
      pct = Math.min(99, ((f + within) / files) * 100);
      what = p.stage ? `${STAGE[p.stage] || p.stage}${p.stem ? ` ${p.stem}.mat` : ''}${p.stage === 'images' && p.total ? `: ${fmtInt(p.done)} of ${fmtInt(p.total)}` : ''}${files > 1 ? ` (file ${p.file} of ${files})` : ''}` : 'Starting';
    } else if (j.status === 'done') { pct = 100; what = 'Finished'; }
    else if (j.status === 'failed') what = j.error || 'Failed: see the log below';
    else if (j.status === 'cancelled') what = 'Cancelled';
    const when = new Date(j.started || j.created);
    const row = h('div', { class: `job ${j.status}` },
      h('div', { class: 'job-head' },
        h('span', { class: `jstat ${j.status}` }, { queued: 'Waiting', running: 'Importing', done: 'Imported', failed: 'Failed', cancelled: 'Cancelled' }[j.status] || j.status),
        h('b', {}, j.flight),
        h('span', { class: 'muted' }, what),
        h('span', { class: 'grow' }),
        h('span', { class: 'muted num' }, when.toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })),
        (j.status === 'running' || j.status === 'queued') ? h('button', { class: 'btn ghost small', onclick: async () => { await C.api.adminPost('job_cancel', { id: j.id }); loadJobs(); } }, 'Cancel') : null,
        j.status === 'done' ? h('a', { class: 'btn small', href: '#/classify', onclick: () => C.selectOnly(j.flight) }, 'Classify') : null),
      (j.status === 'running' || j.status === 'done') ? h('div', { class: 'progress-track' }, h('i', { style: { width: pct + '%' } })) : null,
      h('details', { open: j.status === 'failed' }, h('summary', {}, 'Log'), h('pre', { class: 'log' }, j.log_tail || '(empty)')));
    list.append(row);
  }
  box.append(list);
}

// ------------------------------------------------------------------ folders
async function loadFolders() {
  const { h, $ } = C;
  const box = $('#foldersBox'); if (!box) return;
  try {
    const d = await C.api.adminGet('scan');
    state.folders = d.folders; state.root = d.root;
  } catch (e) {
    box.innerHTML = ''; box.append(h('h2', {}, 'MAT folders on the server'), h('p', { class: 'muted' }, e.message)); return;
  }
  renderFolders();
}
function renderFolders() {
  const { h, $, fmtInt } = C;
  const box = $('#foldersBox'); if (!box) return;
  const q = state.filter.toLowerCase();
  const rows = state.folders.filter((f) => !q || f.dir.toLowerCase().includes(q) || f.flights.some((x) => x.id.toLowerCase().includes(q)));
  const todo = state.folders.filter((f) => !f.flights.some((x) => x.exported)).length;
  box.innerHTML = '';
  box.append(
    h('div', { class: 'sec-h' },
      h('h2', {}, 'MAT folders on the server'),
      h('span', { class: 'muted' }, state.folders.length ? `${fmtInt(state.folders.length)} folders, ${fmtInt(todo)} not imported yet` : ''),
      h('span', { class: 'grow' }),
      h('input', { type: 'search', placeholder: 'Find a folder or flight', value: state.filter, 'aria-label': 'Find a folder',
        oninput: (e) => { state.filter = e.target.value; renderFolders(); const i = $('#foldersBox input[type=search]'); i.focus(); i.setSelectionRange(i.value.length, i.value.length); } }),
      h('button', { class: 'btn small', onclick: loadFolders }, 'Look again')),
    h('p', { class: 'muted small' }, 'Looking in ', h('code', {}, state.root), '. Put each flight’s MAT files (and any navigation file) in a folder there: ', h('code', {}, '<project>/<flight>/'), '. Then import it.'));
  if (!state.folders.length) {
    box.append(h('div', { class: 'card empty-card' }, h('p', {}, 'No MAT files found yet. Copy a flight’s MAT files into a folder under ', h('code', {}, state.root), ', then press Look again.')));
    return;
  }
  const tbody = h('tbody');
  const busy = new Set(state.jobs.filter((j) => j.status === 'running' || j.status === 'queued').map((j) => j.flight));
  const groups = new Map();
  const groupOf = (f) => (f.catalogue ? 'Exports from your CPI pipeline' : splitDir(f.dir).project);
  for (const f of rows) { const pr = groupOf(f); if (!groups.has(pr)) groups.set(pr, []); groups.get(pr).push(f); }
  for (const [project, list] of groups) {
  const notDone = list.filter((f) => !f.flights.some((x) => x.exported)).length;
  tbody.append(h('tr', { class: 'grouprow' }, h('td', { colspan: '5' },
    h('b', {}, project || 'Not in a project folder'),
    h('span', { class: 'muted small' }, `  ${fmtInt(list.length)} folder${list.length === 1 ? '' : 's'}` + (notDone ? `, ${fmtInt(notDone)} not imported` : ', all imported')))));
  for (const f of list) {
    if (f.catalogue) { tbody.append(projectRow(f, busy)); continue; }
    const done = f.flights.filter((x) => x.exported);
    const importing = f.flights.find((x) => busy.has(x.id));
    tbody.append(h('tr', {},
      h('td', { class: 'indent' }, h('b', { class: 'folder' }, splitDir(f.dir).flight || f.dir || '(top level)'), h('div', { class: 'muted small mono' }, f.dir),
        f.nav.length ? h('div', { class: 'muted small' }, 'Other files: ', f.nav.map((n) => n.name).join(', ')) : null),
      h('td', { class: 'n' }, fmtInt(f.mats.length), h('div', { class: 'muted small' }, fmtBytes(f.bytes))),
      h('td', { class: 'num nowrap' }, new Date(f.mtime * 1000).toISOString().slice(0, 10)),
      h('td', {}, importing ? h('span', { class: 'jstat running' }, `Importing ${importing.id}…`) : done.length ? done.map((x) => h('div', { class: 'fchip' }, h('span', { class: 'jstat done' }, 'Imported'), h('b', {}, x.id),
        h('span', { class: 'muted' }, `${fmtInt(x.n)} particles`))) : f.flights.length ? h('span', { class: 'jstat queued' }, 'Settings saved, not imported') : h('span', { class: 'muted' }, 'Not imported')),
      h('td', { class: 'actions' },
        done.length || f.flights.length
          ? [h('button', { class: 'btn small', onclick: () => openForm(f, f.flights[0].id) }, 'Change settings'),
             h('button', { class: 'btn small', onclick: () => reimport(f.flights[0].id, false) }, 'Import again')]
          : h('button', { class: 'btn small primary', onclick: () => openForm(f, null) }, 'Import…'))));
  }
  }
  box.append(h('div', { class: 'tablewrap' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'Folder'), h('th', { class: 'n' }, 'MAT files'), h('th', {}, 'Changed'), h('th', {}, 'Status'), h('th', {}, ''))), tbody)));
}
/** A folder written by the CPI pipeline's PNG export: particle_catalogue.csv + PNG folders. */
function projectRow(f, busy) {
  const { h, fmtInt } = C;
  const cfg = f.flights.find((x) => x.project);
  const importing = cfg && busy.has(cfg.id);
  return h('tr', {},
    h('td', { class: 'indent' }, h('b', { class: 'folder' }, f.dir || '(top level)'),
      h('div', { class: 'muted small' }, 'particle_catalogue.csv' + (f.catalogue.flights_csv ? ' and flights.csv' : '') + ', PNGs in date/flight folders')),
    h('td', { class: 'n' }, fmtInt(f.mats.length), h('div', { class: 'muted small' }, f.mats.length ? fmtBytes(f.bytes) : 'MATs elsewhere')),
    h('td', { class: 'num nowrap' }, new Date(f.mtime * 1000).toISOString().slice(0, 10)),
    h('td', {}, importing ? h('span', { class: 'jstat running' }, `Importing ${cfg.id}…`)
      : cfg && cfg.exported ? h('div', { class: 'fchip' }, h('span', { class: 'jstat done' }, 'Imported'), h('b', {}, cfg.label),
          h('span', { class: 'muted' }, `${fmtInt(cfg.flights)} flights, ${fmtInt(cfg.n)} particles`))
      : cfg ? h('span', { class: 'jstat queued' }, 'Settings saved, not imported') : h('span', { class: 'muted' }, 'Not imported')),
    h('td', { class: 'actions' }, cfg
      ? [h('button', { class: 'btn small', onclick: () => openProjectForm(f, cfg.id) }, 'Change settings'),
         h('button', { class: 'btn small', onclick: () => reimport(cfg.id, false) }, 'Import again')]
      : h('button', { class: 'btn small primary', onclick: () => openProjectForm(f, null) }, 'Import project…')));
}

async function openProjectForm(folder, cfgId) {
  const { h } = C;
  let summary, cfg = {};
  try { summary = (await C.api.adminGet('catsummary', { path: folder.dir })).summary; }
  catch (e) { C.toast(e.message, 5000); return; }
  if (cfgId) { try { cfg = (await C.api.adminGet('flight_get', { id: cfgId })).config; } catch { cfg = {}; } }
  const inst = cfg.instrument || {};
  const chosen = new Set(cfg.groups || summary.groups.map((g) => g.key));
  const F = {};
  const inp = (name, attrs = {}) => (F[name] = h('input', { name, ...attrs }));
  const field = (label, control, hint) => h('label', { class: 'fld' }, h('span', {}, label, hint ? h('small', {}, ` ${hint}`) : null), control);
  let kind = cfg.kind || 'aircraft';
  const kindSeg = h('div', { class: 'seg' }, [['aircraft', 'Aircraft'], ['ground', 'Ground site'], ['lab', 'Lab / calibration']].map(([v, l]) =>
    h('button', { type: 'button', 'aria-pressed': String(v === kind), onclick: (e) => { kind = v; [...kindSeg.children].forEach((b) => b.setAttribute('aria-pressed', String(b === e.currentTarget))); } }, l)));
  const total = () => summary.groups.filter((g) => chosen.has(g.key)).reduce((a, g) => a + g.n, 0);
  const sumEl = h('span', { class: 'muted small' });
  const upd = () => { sumEl.textContent = `${chosen.size} of ${summary.groups.length} chosen, ${total().toLocaleString()} particles`; };
  const list = h('div', { class: 'grouplist' }, summary.groups.map((g) => h('label', { class: 'check' },
    h('input', { type: 'checkbox', checked: chosen.has(g.key), onchange: (e) => { if (e.target.checked) chosen.add(g.key); else chosen.delete(g.key); upd(); } }),
    h('b', {}, g.flight || g.date || g.key), g.flight ? h('span', { class: 'muted small' }, g.date) : h('span', { class: 'muted small' }, 'no flight'),
    h('span', { class: 'muted small num' }, `${g.n.toLocaleString()} particles`))));
  upd();
  const undated = h('select', {}, h('option', { value: '' }, 'the same kind as above'), h('option', { value: 'ground' }, 'ground or test days'), h('option', { value: 'lab' }, 'lab or calibration'));
  undated.value = cfg.undated_kind || '';
  const err = h('p', { class: 'formerr', role: 'alert' });
  const d = h('dialog', { class: 'dlg importdlg', 'aria-label': 'Import a pipeline project' });
  const project = cfg.campaign || summary.project;
  d.append(h('form', { class: 'dlg-inner', novalidate: true, onsubmit: (e) => e.preventDefault() },
    h('div', { class: 'imp-head' }, h('h2', {}, cfgId ? `Settings for ${cfgId}` : 'Import a project'), h('code', {}, folder.dir)),
    h('p', { class: 'muted small' }, `Made by your CPI pipeline’s PNG export: ${summary.n.toLocaleString()} particle images listed in particle_catalogue.csv. The images are used as they are. Sizes, focus and outlines come from the MAT files. Each flight becomes its own entry on the classify page, grouped under the project.`),
    h('fieldset', {}, h('legend', {}, 'Project'),
      h('div', { class: 'grid2' },
        field('Project name', inp('campaign', { value: project })),
        field('Project id', inp('id', { value: cfg.id || cleanId(project), readonly: !!cfgId }), cfgId ? '(fixed)' : 'flight ids become <id>-<flight>')),
      h('div', { class: 'fld' }, h('span', {}, 'Kind'), kindSeg),
      h('div', { class: 'grid2' },
        field('Platform', inp('platform', { value: cfg.platform || '', placeholder: 'e.g. Grob Egrett' })),
        field('Particles with no flight are from', undated)),
      field('Notes for people classifying', inp('notes', { value: cfg.notes || '' }))),
    h('fieldset', {}, h('legend', {}, 'Instrument'),
      h('div', { class: 'grid3' },
        field('Name', inp('inst_name', { value: inst.name || 'CPI' })),
        field('Version', inp('inst_version', { value: inst.version || '', list: 'cpiVersions2', placeholder: 'e.g. 1.0 or 3V-CPI' })),
        field('Serial', inp('inst_serial', { value: inst.serial || '' }))),
      h('datalist', { id: 'cpiVersions2' }, ['1.0', '2.0', '2.5', '3V-CPI'].map((v) => h('option', { value: v }))),
      field('Pixel size (µm)', inp('pixel_um', { type: 'number', step: 'any', min: '0', value: cfg.pixel_um ?? '', placeholder: 'from the processing' }), '(blank: the value the processing recorded)')),
    h('fieldset', {}, h('legend', {}, 'MAT files'),
      field('Folder with the MAT files', inp('mat_dir', { value: cfg.mat_dir && cfg.catalogue_dir && cfg.mat_dir !== cfg.catalogue_dir ? cfg.mat_dir.replace(state.root + '/', '') : '', placeholder: folder.mats.length ? `this folder (${folder.mats.length} MAT files)` : 'e.g. ACTIVE/mat' }),
        folder.mats.length ? '(blank: this folder)' : '(relative to the MAT folder root)')),
    h('fieldset', {}, h('legend', {}, 'Flights to import'),
      h('div', { class: 'row between' }, sumEl, h('span', {},
        h('button', { type: 'button', class: 'btn ghost small', onclick: () => { summary.groups.forEach((g) => chosen.add(g.key)); list.querySelectorAll('input').forEach((i) => { i.checked = true; }); upd(); } }, 'All'),
        h('button', { type: 'button', class: 'btn ghost small', onclick: () => { summary.groups.filter((g) => !g.flight).forEach((g) => chosen.delete(g.key)); list.querySelectorAll('input').forEach((i, k) => { if (!summary.groups[k].flight) i.checked = false; }); upd(); } }, 'Only flights'),
        h('button', { type: 'button', class: 'btn ghost small', onclick: () => { chosen.clear(); list.querySelectorAll('input').forEach((i) => { i.checked = false; }); upd(); } }, 'None'))),
      list),
    err,
    h('div', { class: 'dlg-actions sticky' },
      h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'),
      h('span', { class: 'grow' }),
      h('button', { type: 'button', class: 'btn', onclick: () => submit(false) }, 'Save settings only'),
      h('button', { type: 'button', class: 'btn primary', onclick: () => submit(true) }, cfgId ? 'Save and import again' : 'Import'))));
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();

  async function submit(doImport) {
    err.textContent = '';
    const id = F.id.value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(id)) { err.textContent = 'The project id can only use letters, digits, . _ - (up to 64).'; return; }
    if (!chosen.size) { err.textContent = 'Choose at least one flight.'; return; }
    const num = (v) => (v === '' ? null : +v);
    const config = { ...cfg, id, campaign: F.campaign.value.trim() || id, kind, platform: F.platform.value.trim() || null, notes: F.notes.value.trim() || null,
      undated_kind: undated.value || null,
      instrument: { name: F.inst_name.value.trim() || 'CPI', version: F.inst_version.value.trim(), serial: F.inst_serial.value.trim() },
      pixel_um: num(F.pixel_um.value) };
    for (const k of ['catalogue_dir', 'mat_dir', 'groups', 'source_dir', 'saved_by', 'saved_utc']) delete config[k];
    try {
      const r = await C.api.adminPost('save_project', { config, source_dir: folder.dir, mat_dir: F.mat_dir.value.trim(), groups: [...chosen], import: doImport });
      d.close();
      C.toast(doImport ? `Importing ${r.saved}…` : `Saved settings for ${r.saved}`);
      await Promise.all([loadJobs(), loadFolders()]);
      document.getElementById('jobsBox')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) { err.textContent = e.message; }
  }
}

async function reimport(id, force) {
  try { await C.api.adminPost('import', { id, force }); C.toast(`Importing ${id} again`); loadJobs(); } catch (e) { C.toast(e.message, 4000); }
}
const fmtBytes = (b) => (b > 1e9 ? (b / 1e9).toFixed(1) + ' GB' : (b / 1e6).toFixed(0) + ' MB');

// ------------------------------------------------------------------ flights
async function loadFlights() {
  try { state.flights = (await C.api.adminGet('flights')).flights; } catch { state.flights = []; }
  renderFlights();
}
function renderFlights() {
  const { h, $, fmtInt, display } = C;
  const box = $('#flightsBox'); if (!box) return;
  box.innerHTML = '';
  box.append(h('div', { class: 'sec-h' }, h('h2', {}, 'Imported flights'),
    h('span', { class: 'muted' }, state.flights.length ? `${fmtInt(state.flights.length)} flight${state.flights.length === 1 ? '' : 's'}, ${fmtInt(state.flights.reduce((s, f) => s + f.n, 0))} particles` : '')));
  if (!state.flights.length) { box.append(h('p', { class: 'muted' }, 'None yet. Import a folder above.')); return; }
  const tbody = h('tbody');
  for (const f of state.flights) {
    tbody.append(h('tr', { class: f.hidden ? 'is-hidden' : '' },
      h('td', {}, h('b', {}, f.label), h('div', { class: 'muted small' }, f.id + (f.source_dir ? ` from ${f.source_dir}` : ''))),
      h('td', {}, f.campaign || h('span', { class: 'muted' }, '–')),
      h('td', { class: 'n' }, fmtInt(f.n)),
      h('td', {}, h('span', { class: 'cmchip' }, h('i', { style: { background: display.gradient(f.colormap) } }), display.colormap(f.colormap)?.name || f.colormap)),
      h('td', { class: 'num nowrap' }, (f.exported_utc || '').slice(0, 10)),
      h('td', {}, h('label', { class: 'switch', title: 'Hidden flights are not offered to people classifying. Their labels are kept.' },
        h('input', { type: 'checkbox', checked: !f.hidden, onchange: async (e) => {
          try { await C.api.adminPost('flight_hidden', { id: f.id, hidden: !e.target.checked }); C.toast(e.target.checked ? `${f.id} is visible` : `${f.id} is hidden`); await C.onDataChanged(); loadFlights(); }
          catch (err) { C.toast(err.message, 4000); e.target.checked = !e.target.checked; }
        } }), h('span', {}, f.hidden ? 'Hidden' : 'Shown'))),
      h('td', { class: 'actions' },
        f.has_config ? h('button', { class: 'btn small', onclick: () => {
          const folder = state.folders.find((x) => x.dir === f.source_dir);
          if (!folder) { C.toast('Its folder is no longer on the server', 4000); return; }
          if (folder.catalogue) openProjectForm(folder, f.config_id); else openForm(folder, f.id);
        } }, 'Change settings') : null,
        h('button', { class: 'btn small ghost danger', onclick: () => confirmRemove(f) }, 'Remove'))));
  }
  box.append(h('div', { class: 'tablewrap' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'Flight'), h('th', {}, 'Project'), h('th', { class: 'n' }, 'Particles'), h('th', {}, 'Colour map'), h('th', {}, 'Imported'), h('th', {}, 'On the classify page'), h('th', {}, ''))), tbody)));
}
function confirmRemove(f) {
  const { h } = C;
  const d = h('dialog', { class: 'dlg' });
  const input = h('input', { 'aria-label': 'Flight id', autocomplete: 'off' });
  const err = h('p', { class: 'formerr' });
  d.append(h('form', { class: 'dlg-inner', method: 'dialog', onsubmit: async (e) => {
    e.preventDefault();
    try {
      await C.api.adminPost('flight_delete', { id: f.id, confirm: input.value.trim() });
      d.close(); C.toast(`Removed ${f.id}`); await C.onDataChanged(); loadFlights(); loadFolders();
    } catch (x) { err.textContent = x.message; }
  } },
  h('h3', {}, `Remove ${f.label}?`),
  h('p', { class: 'dlg-why' }, 'This deletes the imported images and particle table, so the flight disappears from the classify page. Everyone’s classifications stay in the database, the MAT files are not touched, and you can import the folder again at any time.'),
  h('label', { class: 'fld' }, h('span', {}, `Type ${f.id} to confirm`), input), err,
  h('div', { class: 'dlg-actions' }, h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'), h('button', { class: 'btn danger-solid' }, 'Remove'))));
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
}

// ------------------------------------------------------------------ import form
const guessers = {
  time: [/^time$/i, /^(utc|datetime|timestamp|time_?utc|date_?time)/i, /time/i],
  lat: [/^lat(itude)?$/i, /^lat_gin$/i, /lat/i],
  lon: [/^lon(gitude)?$/i, /^lon_gin$/i, /lon/i],
  alt: [/^alt(itude)?$/i, /^alt_gin$/i, /gps.?alt|height|alt/i],
  temp: [/^tat_di_r$/i, /^(tat|temp|temperature|t)$/i, /tat|temp|air_?t/i],
};
function guess(cols, key) {
  for (const re of guessers[key]) { const c = cols.find((x) => re.test(x.name)); if (c) return c.name; }
  return '';
}
const cleanId = (x) => x.replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^[^A-Za-z0-9]+/, '').replace(/-+$/, '').slice(0, 64);
/** Folders are expected as <project>/<flight>; the first part is the project. */
function splitDir(dir) {
  const parts = (dir || '').split('/').filter(Boolean);
  if (parts.length < 2) return { project: '', flight: parts[0] || '' };
  return { project: parts[0], flight: parts.slice(1).join('/') };
}
function suggestId(dir) {
  const { project, flight } = splitDir(dir);
  const taken = new Set([...state.flights.map((f) => f.id), ...state.folders.flatMap((f) => f.flights.map((x) => x.id))]);
  const short = cleanId(flight.split('/').pop() || 'flight') || 'flight';
  if (!taken.has(short)) return short;
  const long = cleanId(`${project}-${flight}`) || short;
  if (!taken.has(long)) return long;
  for (let i = 2; ; i++) if (!taken.has(`${long}-${i}`)) return `${long}-${i}`;
}

async function openForm(folder, flightId) {
  const { h, display } = C;
  let cfg = {};
  if (flightId) {
    try { cfg = (await C.api.adminGet('flight_get', { id: flightId })).config; } catch { cfg = {}; }
  }
  const saved = JSON.parse(JSON.stringify(cfg));
  const base = (p) => (p || '').split('/').pop();
  const chosenMats = new Set(cfg.mat_files ? cfg.mat_files.map(base) : folder.mats.map((m) => m.name));
  const nav = cfg.nav || {};
  const sel = cfg.select || {};
  const inst = cfg.instrument || {};
  const site = cfg.site || {};

  const d = h('dialog', { class: 'dlg importdlg', 'aria-label': 'Import a folder' });
  const err = h('p', { class: 'formerr', role: 'alert' });
  const F = {};   // named inputs
  const inp = (name, attrs = {}) => (F[name] = h('input', { name, ...attrs }));
  const field = (label, control, hint) => h('label', { class: 'fld' }, h('span', {}, label, hint ? h('small', {}, ` ${hint}`) : null), control);

  // kind
  let kind = cfg.kind || 'aircraft';
  const kindSeg = h('div', { class: 'seg' }, [['aircraft', 'Aircraft'], ['ground', 'Ground site'], ['lab', 'Lab / calibration']].map(([v, l]) =>
    h('button', { type: 'button', 'aria-pressed': String(v === kind), onclick: (e) => { kind = v; [...kindSeg.children].forEach((b) => b.setAttribute('aria-pressed', String(b === e.currentTarget))); siteBox.hidden = kind === 'aircraft'; } }, l)));

  // MAT files
  const matList = h('div', { class: 'matlist' }, folder.mats.map((m) => h('label', { class: 'check' },
    h('input', { type: 'checkbox', checked: chosenMats.has(m.name), onchange: (e) => { if (e.target.checked) chosenMats.add(m.name); else chosenMats.delete(m.name); matCount(); } }),
    h('span', { class: 'mono' }, m.name), h('span', { class: 'muted small' }, fmtBytes(m.size)))));
  const matSum = h('span', { class: 'muted small' });
  const matCount = () => { matSum.textContent = `${chosenMats.size} of ${folder.mats.length} chosen`; };
  matCount();

  // navigation
  const navFile = h('select', { 'aria-label': 'Navigation file' }, h('option', { value: '' }, 'None'), folder.nav.map((n) => h('option', { value: n.name }, n.name)));
  navFile.value = nav.file ? base(nav.file) : '';
  const navBox = h('div', { class: 'navmap' });
  const navCols = {};
  async function loadNav() {
    navBox.innerHTML = '';
    if (!navFile.value) { navBox.append(h('p', { class: 'muted small' }, 'Without a navigation file, particles have no position, altitude or temperature (ground sites can give a fixed position below).')); return; }
    navBox.append(h('p', { class: 'muted small' }, 'Reading the file…'));
    let info;
    try { info = (await C.api.adminGet('navcols', { path: (folder.dir ? folder.dir + '/' : '') + navFile.value })).nav; }
    catch (e) { navBox.innerHTML = ''; navBox.append(h('p', { class: 'formerr' }, e.message)); return; }
    navBox.innerHTML = '';
    if (info.error) { navBox.append(h('p', { class: 'formerr' }, info.error)); return; }
    const cols = info.columns || [];
    const mk = (key, label) => {
      const s = h('select', { 'aria-label': label }, h('option', { value: '' }, '(none)'), cols.map((c) => h('option', { value: c.name },
        c.name + (c.units ? ` [${c.units}]` : c.example ? ` e.g. ${c.example}` : ''))));
      s.value = key === 'time' ? (nav.time_column || guess(cols, 'time')) : ((nav.fields || {})[key] ?? guess(cols, key));
      navCols[key] = s;
      return field(label, s);
    };
    navCols.format = info.format;
    const tunits = h('select', { 'aria-label': 'Time format' },
      info.format === 'netcdf' ? h('option', { value: '' }, 'From the file (units attribute)') : null,
      h('option', { value: 'iso' }, 'Date and time text, e.g. 2026-02-11T10:15:00Z'),
      h('option', { value: 'unix' }, 'Unix seconds'),
      h('option', { value: 'datenum' }, 'MATLAB datenum'),
      h('option', { value: 'seconds since' }, 'Seconds since a date…'));
    const since = h('input', { type: 'date', 'aria-label': 'Seconds since date', hidden: true });
    const tu = nav.time_units || (info.format === 'netcdf' ? '' : 'iso');
    if (tu.startsWith('seconds since')) { tunits.value = 'seconds since'; since.hidden = false; since.value = tu.replace('seconds since', '').trim().slice(0, 10); } else tunits.value = tu;
    tunits.onchange = () => { since.hidden = tunits.value !== 'seconds since'; };
    navCols.tunits = tunits; navCols.since = since;
    const tempU = h('select', {}, h('option', { value: 'K' }, 'kelvin'), h('option', { value: 'C' }, '°C'));
    const tempName = (nav.fields || {}).temp ?? guess(cols, 'temp');
    const tempGuessK = (() => { const c = cols.find((x) => x.name === tempName); return !!c && (/^k/i.test(c.units || '') || +c.example > 150); })();
    tempU.value = nav.temp_units || (tempGuessK ? 'K' : 'C');
    const altU = h('select', {}, h('option', { value: 'm' }, 'metres'), h('option', { value: 'ft' }, 'feet'));
    altU.value = nav.alt_units || 'm';
    const gap = h('input', { type: 'number', min: '0', step: '1', value: nav.max_gap_s ?? 10 });
    navCols.tempU = tempU; navCols.altU = altU; navCols.gap = gap;
    navBox.append(h('div', { class: 'grid2' },
      mk('time', 'Time column'), field('Time format', h('div', { class: 'row' }, tunits, since)),
      mk('lat', 'Latitude'), mk('lon', 'Longitude'),
      mk('alt', 'Altitude'), field('Altitude units', altU),
      mk('temp', 'Temperature'), field('Temperature units', tempU),
      field('Leave blank if no data within', h('div', { class: 'row' }, gap, h('span', { class: 'muted small' }, 'seconds')))));
  }
  navFile.onchange = loadNav;

  const siteBox = h('div', { class: 'grid3', hidden: kind === 'aircraft' },
    field('Site latitude', inp('site_lat', { type: 'number', step: 'any', value: site.lat ?? '' })),
    field('Site longitude', inp('site_lon', { type: 'number', step: 'any', value: site.lon ?? '' })),
    field('Site altitude (m)', inp('site_alt', { type: 'number', step: 'any', value: site.alt ?? '' })));

  // colour map
  let cmap = cfg.colormap || 'gray';
  const cmPick = h('div', { class: 'cmgrid small' }, display.colormaps().map((m) => h('button', {
    type: 'button', class: 'cm' + (m.id === cmap ? ' on' : ''), title: m.desc,
    onclick: (e) => { cmap = m.id; [...cmPick.children].forEach((b) => b.classList.toggle('on', b === e.currentTarget)); updateForce(); } },
  h('i', { style: { background: display.gradient(m.id) } }), h('span', {}, m.name + (m.id === 'gray' ? ' (recommended)' : '')))));
  const forceChk = h('input', { type: 'checkbox' });
  const forceNote = h('span', { class: 'muted small' });
  function updateForce() {
    const changed = flightId && ((saved.colormap || 'gray') !== cmap || !!saved.stretch !== F.stretch.checked || (saved.transpose ?? true) !== F.transpose.checked);
    if (changed) { forceChk.checked = true; forceNote.textContent = 'Needed because the colour map, stretch or orientation changed.'; }
    else forceNote.textContent = flightId ? 'Only needed if the images should change.' : '';
  }

  d.append(h('form', { class: 'dlg-inner', novalidate: true, onsubmit: (e) => e.preventDefault() },
    h('div', { class: 'imp-head' }, h('h2', {}, flightId ? `Settings for ${flightId}` : 'Import a folder'), h('code', {}, folder.dir || '(top level)')),
    h('fieldset', {}, h('legend', {}, 'What it is'),
      h('div', { class: 'grid2' },
        field('Flight id', inp('id', { value: cfg.id || suggestId(folder.dir), required: true, pattern: '[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', readonly: !!flightId }),
          flightId ? '(fixed: classifications refer to it)' : 'letters, digits, . _ -'),
        field('Name shown to people', inp('label', { value: cfg.label || '', placeholder: splitDir(folder.dir).flight || 'e.g. C123 frontal cirrus' }))),
      h('div', { class: 'fld' }, h('span', {}, 'Kind'), kindSeg),
      h('div', { class: 'grid3' },
        field('Project', inp('campaign', { value: cfg.campaign ?? splitDir(folder.dir).project, placeholder: 'e.g. PICASSO' })),
        field('Platform', inp('platform', { value: cfg.platform || '', placeholder: 'e.g. FAAM BAe-146' })),
        field('Date', inp('date', { type: 'date', value: cfg.date || '' }), '(blank: from the data)')),
      field('Notes for people classifying', inp('notes', { value: cfg.notes || '' }))),
    h('fieldset', {}, h('legend', {}, 'Instrument'),
      h('div', { class: 'grid3' },
        field('Name', inp('inst_name', { value: inst.name || 'CPI' })),
        field('Version', inp('inst_version', { value: inst.version || '', list: 'cpiVersions', placeholder: 'e.g. 3V-CPI' })),
        field('Serial', inp('inst_serial', { value: inst.serial || '' }))),
      h('datalist', { id: 'cpiVersions' }, ['1.0', '2.0', '2.5', '3V-CPI', 'CPI-v1'].map((v) => h('option', { value: v }))),
      field('Pixel size (µm)', inp('pixel_um', { type: 'number', step: 'any', min: '0', value: cfg.pixel_um ?? '', placeholder: 'from the processing' }), '(blank: use the value the processing recorded)')),
    h('fieldset', {}, h('legend', {}, 'MAT files'), h('div', { class: 'row between' }, matSum,
      h('span', {}, h('button', { type: 'button', class: 'btn ghost small', onclick: () => { folder.mats.forEach((m) => chosenMats.add(m.name)); matList.querySelectorAll('input').forEach((i) => { i.checked = true; }); matCount(); } }, 'All'),
        h('button', { type: 'button', class: 'btn ghost small', onclick: () => { chosenMats.clear(); matList.querySelectorAll('input').forEach((i) => { i.checked = false; }); matCount(); } }, 'None'))), matList),
    h('fieldset', {}, h('legend', {}, 'Position and temperature'),
      field('Navigation or met file in this folder', navFile, '(CSV or netCDF)'), navBox, siteBox),
    h('fieldset', {}, h('legend', {}, 'Which particles'),
      h('label', { class: 'check' }, inp('valid_only', { type: 'checkbox', checked: sel.valid_only ?? true }), 'Only particles the processing measured (skips empty and background ROIs)'),
      h('label', { class: 'check' }, inp('exclude_edge', { type: 'checkbox', checked: !!sel.exclude_edge_touch }), 'Leave out particles cut off at the image edge'),
      h('div', { class: 'grid3' },
        field('Smallest size (µm)', inp('min_size', { type: 'number', min: '0', step: 'any', value: sel.min_size_um || '' })),
        field('Largest size (µm)', inp('max_size', { type: 'number', min: '0', step: 'any', value: sel.max_size_um || '' })),
        field('Measured by', (() => { const s = h('select', {}, [['len', 'Maximum length'], ['deq', 'Equivalent diameter'], ['len_feret', 'Feret length'], ['shadow_deq', 'Shadow-circle diameter']].map(([v, l]) => h('option', { value: v }, l))); s.value = sel.size_field || 'len'; F.size_field = s; return s; })()))),
    h('fieldset', {}, h('legend', {}, 'Images'),
      h('p', { class: 'muted small' }, 'Grey is recommended: people can then pick any colour map on the classify page. Choosing a map here fixes it into the images for everyone.'),
      cmPick,
      h('label', { class: 'check' }, inp('stretch', { type: 'checkbox', checked: saved.stretch === 'per-image', onchange: updateForce }), 'Stretch each image’s contrast (makes grey levels incomparable between particles)'),
      h('label', { class: 'check' }, inp('transpose', { type: 'checkbox', checked: saved.transpose ?? true, onchange: updateForce }), 'Show the CPI x direction across the screen'),
      flightId ? h('label', { class: 'check' }, forceChk, 'Rewrite the existing images', ' ', forceNote) : null),
    err,
    h('div', { class: 'dlg-actions sticky' },
      h('button', { type: 'button', class: 'btn ghost', onclick: () => d.close() }, 'Cancel'),
      h('span', { class: 'grow' }),
      h('button', { type: 'button', class: 'btn', onclick: () => submit(false) }, 'Save settings only'),
      h('button', { type: 'button', class: 'btn primary', onclick: () => submit(true) }, flightId ? 'Save and import again' : 'Import'))));
  updateForce();
  loadNav();
  d.addEventListener('close', () => d.remove());
  d.addEventListener('cancel', (e) => { if (!confirm('Close without saving?')) e.preventDefault(); });
  document.body.append(d);
  d.showModal();

  async function submit(doImport) {
    err.textContent = '';
    const id = F.id.value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(id)) { err.textContent = 'The flight id can only use letters, digits, . _ - (up to 64), starting with a letter or digit.'; F.id.focus(); return; }
    if (!chosenMats.size) { err.textContent = 'Choose at least one MAT file.'; return; }
    const num = (v) => (v === '' || v === null || v === undefined ? null : +v);
    const config = {
      ...saved,
      id, label: F.label.value.trim() || splitDir(folder.dir).flight.split('/').pop() || id, kind,
      campaign: F.campaign.value.trim() || null, platform: F.platform.value.trim() || null,
      date: F.date.value || null, notes: F.notes.value.trim() || null,
      instrument: { name: F.inst_name.value.trim() || 'CPI', version: F.inst_version.value.trim(), serial: F.inst_serial.value.trim() },
      pixel_um: num(F.pixel_um.value),
      transpose: F.transpose.checked,
      colormap: cmap,
      stretch: F.stretch.checked ? 'per-image' : null,
      select: { valid_only: F.valid_only.checked, exclude_edge_touch: F.exclude_edge.checked, size_field: F.size_field.value,
        min_size_um: num(F.min_size.value), max_size_um: num(F.max_size.value) },
    };
    delete config.mat_files; delete config.source_dir; delete config.saved_by; delete config.saved_utc;
    if (kind !== 'aircraft' && (F.site_lat.value || F.site_lon.value || F.site_alt.value)) config.site = { lat: num(F.site_lat.value), lon: num(F.site_lon.value), alt: num(F.site_alt.value) };
    else delete config.site;
    if (navFile.value && navCols.time) {
      if (!navCols.time.value) { err.textContent = 'Choose the time column of the navigation file.'; return; }
      const fields = {};
      for (const k of ['lat', 'lon', 'alt', 'temp']) if (navCols[k]?.value) fields[k] = navCols[k].value;
      let tu = navCols.tunits.value;
      if (tu === 'seconds since') { if (!navCols.since.value) { err.textContent = 'Give the date the seconds count from.'; return; } tu = `seconds since ${navCols.since.value}T00:00:00Z`; }
      config.nav = { format: navCols.format, time_column: navCols.time.value, ...(tu ? { time_units: tu } : {}), fields,
        temp_units: navCols.tempU.value, alt_units: navCols.altU.value, max_gap_s: num(navCols.gap.value) ?? 10 };
    } else delete config.nav;
    try {
      const r = await C.api.adminPost('save_flight', {
        config, source_dir: folder.dir, mat_files: [...chosenMats], nav_file: navFile.value || null,
        import: doImport, force: !!(flightId && forceChk.checked) });
      d.close();
      C.toast(doImport ? `Importing ${r.saved}…` : `Saved settings for ${r.saved}`);
      await Promise.all([loadJobs(), loadFolders()]);
      document.getElementById('jobsBox')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) { err.textContent = e.message; }
  }
}
