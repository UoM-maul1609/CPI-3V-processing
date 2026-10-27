// Project pages: what the project was, links and pictures, and its flights.
// The text, links and pictures live in data/projects/<slug>/info.json (written
// by the server when an administrator edits them); everything else comes from
// the flight list that the page already has.

let C;   // context from app.js: h, $, toast, fmtInt, server, api, getManifest, visibleFlights, sparkline, classifyFlights, fetchStats
export function initProjects(ctx) { C = ctx; }

export const projectSlug = (name) => (name.replace(/[^A-Za-z0-9]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase() || 'no-project').slice(0, 60);
const base = (name) => `data/projects/${projectSlug(name)}/`;
export const projectHref = (name) => '#/project/' + encodeURIComponent(name);

let indexCache = null;
/** Summary of every project page (title, summary, cover picture), or {} if none yet. */
export async function projectIndex(force = false) {
  if (indexCache && !force) return indexCache;
  try {
    const r = await fetch('data/projects/index.json', { cache: 'no-cache' });
    indexCache = r.ok ? (await r.json()).projects || {} : {};
  } catch { indexCache = {}; }
  return indexCache;
}
/** Notes on individual flights, from every project page: Map flight id -> text. */
export async function flightNotes(force = false) {
  const idx = await projectIndex(force);
  const m = new Map();
  for (const p of Object.values(idx)) for (const [id, t] of Object.entries(p.flight_notes || {})) m.set(id, t);
  return m;
}
async function loadInfo(name) {
  try {
    const r = await fetch(base(name) + 'info.json', { cache: 'no-cache' });
    if (r.ok) return await r.json();
  } catch { /* none yet */ }
  return { project: name, title: '', summary: '', body: '', links: [], images: [] };
}

// ------------------------------------------------------------------ text
// A little formatting, safe by construction (everything is escaped first):
// blank line = new paragraph, "- " = list, "## " = heading, **bold**, *italic*,
// [text](https://…) and bare https:// links.
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const okUrl = (u) => /^(https?:\/\/|mailto:)/i.test(u);
function inline(s) {
  let t = esc(s);
  t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, txt, url) => okUrl(url.replace(/&amp;/g, '&')) ? `<a href="${url}" target="_blank" rel="noopener">${txt}</a>` : m);
  t = t.replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?])/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  t = t.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<i>$2</i>');
  return t;
}
export function richText(src) {
  const out = [];
  for (const block of String(src || '').split(/\n\s*\n/)) {
    const lines = block.split('\n').filter((l) => l.trim());
    if (!lines.length) continue;
    if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) out.push('<ul>' + lines.map((l) => `<li>${inline(l.replace(/^\s*[-*•]\s+/, ''))}</li>`).join('') + '</ul>');
    else if (/^#{1,3}\s/.test(lines[0])) {
      out.push(`<h3>${inline(lines[0].replace(/^#+\s*/, ''))}</h3>`);
      if (lines.length > 1) out.push(`<p>${lines.slice(1).map(inline).join('<br>')}</p>`);
    } else out.push(`<p>${lines.map(inline).join('<br>')}</p>`);
  }
  return out.join('');
}
/** A flight's note: one short block, links and bold/italic allowed. */
export const noteHtml = (t) => String(t || '').split(/\n+/).map(inline).join('<br>');
const host = (u) => { try { return u.startsWith('mailto:') ? u.slice(7) : new URL(u).host.replace(/^www\./, ''); } catch { return u; } };

// ------------------------------------------------------------------ the page
export async function viewProject(root, name) {
  const { h, fmtInt, server } = C;
  root.className = 'page'; root.style.display = '';
  root.innerHTML = '';
  const inner = h('div', { class: 'page-inner projpage' });
  root.append(inner);
  const flights = C.visibleFlights().filter((f) => (f.campaign || '') === name);
  const info = await loadInfo(name);
  const isAdmin = !!server.user?.is_admin;
  if (!flights.length && !info.updated_utc) {
    inner.append(h('p', { class: 'crumbs' }, h('a', { href: '#/flights' }, 'Flights')), h('h1', {}, name), h('p', { class: 'muted' }, 'There are no flights in this project yet.'));
    return;
  }

  // ---- header
  const dates = flights.map((f) => f.date).filter(Boolean).sort();
  const total = flights.reduce((s, f) => s + f.n, 0);
  const uniq = (xs) => [...new Set(xs.filter(Boolean))];
  const platforms = uniq(flights.map((f) => f.platform));
  const instruments = uniq(flights.map((f) => [f.instrument?.name, f.instrument?.version && 'v' + f.instrument.version].filter(Boolean).join(' ')));
  const fmtDate = (d) => new Date(d + 'T00:00:00Z').toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
  const facts = h('dl', { class: 'projfacts' });
  const fact = (k, v) => { if (v) facts.append(h('div', {}, h('dt', {}, k), h('dd', {}, v))); };
  fact('Flights', fmtInt(flights.length));
  fact('Particles', fmtInt(total));
  fact(dates.length > 1 ? 'Dates' : 'Date', dates.length ? (dates.length > 1 && dates[0] !== dates[dates.length - 1] ? `${fmtDate(dates[0])} – ${fmtDate(dates[dates.length - 1])}` : fmtDate(dates[0])) : '');
  fact(platforms.length > 1 ? 'Platforms' : 'Platform', platforms.join(', '));
  fact('Instrument', instruments.join(', '));
  const classifiedFact = h('div', { hidden: true }, h('dt', {}, 'Classified'), h('dd', {}, '…'));
  facts.append(classifiedFact);

  const title = info.title || name;
  inner.append(
    h('p', { class: 'crumbs' }, h('a', { href: '#/flights' }, 'Flights'), h('span', { 'aria-hidden': 'true' }, ' › '), name),
    h('div', { class: 'projhead' },
      h('div', { class: 'grow' },
        h('h1', {}, title, info.title && info.title !== name ? h('span', { class: 'pill projcode' }, name) : null),
        info.summary ? h('p', { class: 'page-lede' }, info.summary) : null),
      h('div', { class: 'projactions' },
        isAdmin ? h('button', { class: 'btn', onclick: () => editProject(root, name, info) }, svgIcon('edit'), 'Edit page') : null,
        flights.length ? h('button', { class: 'btn primary', onclick: () => C.classifyFlights(flights.map((f) => f.id)) }, 'Classify this project') : null)),
    facts);

  // ---- pictures
  const imgs = info.images || [];
  if (imgs.length) inner.append(gallery(name, imgs));

  // ---- text and links
  const hasBody = !!(info.body || '').trim();
  const links = info.links || [];
  if (hasBody || links.length) {
    inner.append(h('div', { class: 'projcols' + (hasBody && links.length ? '' : ' one') },
      hasBody ? h('div', { class: 'prose', html: richText(info.body) }) : null,
      links.length ? h('aside', { class: 'projlinks' }, h('h2', {}, 'Links'),
        h('ul', {}, links.map((l) => h('li', {},
          h('a', { href: l.url, target: '_blank', rel: 'noopener' }, h('b', {}, l.title), h('span', { class: 'host' }, host(l.url), ' ', svgIcon('ext'))),
          l.note ? h('p', {}, l.note) : null)))) : null));
  } else if (!imgs.length) {
    inner.append(h('div', { class: 'card projempty' },
      isAdmin ? [h('p', {}, 'This project has no description yet. Add what it was, where and when, who took part, links to the campaign website, papers and data, and some pictures.'),
        h('button', { class: 'btn primary', onclick: () => editProject(root, name, info) }, 'Add a description and pictures')]
        : h('p', { class: 'muted' }, 'No description has been added for this project yet.')));
  }

  // ---- flights
  if (flights.length) {
    const tb = h('tbody');
    const doneCell = new Map();
    const fnotes = info.flight_notes || {};
    for (const f of flights) {
      const done = h('td', { class: 'n' }, '');
      doneCell.set(f.id, done);
      tb.append(h('tr', {},
        h('td', { class: 'flightname' }, h('svg', { class: 'kindicon', viewBox: '0 0 24 24', html: `<use href="#${f.kind === 'aircraft' ? 'i-plane' : 'i-ground'}"/>` }), h('b', {}, f.label || f.id),
          fnotes[f.id] ? h('div', { class: 'flightnote', html: noteHtml(fnotes[f.id]) }) : null),
        h('td', { class: 'num nowrap' }, f.date || '–'),
        h('td', {}, f.platform || '–'),
        h('td', { class: 'n' }, fmtInt(f.n)), done,
        h('td', {}, C.sparkline(f.timeline)),
        h('td', {}, h('button', { class: 'btn small primary', onclick: () => C.classifyFlights([f.id]) }, 'Classify'))));
    }
    inner.append(h('h2', { class: 'projsec' }, flights.length === 1 ? 'Flight' : 'Flights'),
      h('div', { class: 'tablewrap' }, h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Flight'), h('th', {}, 'Date'), h('th', {}, 'Platform'), h('th', { class: 'n' }, 'Particles'), h('th', { class: 'n' }, 'Classified'), h('th', {}, 'Through the flight'), h('th', {}))), tb)));
    // classified so far (needs the server)
    C.fetchStats?.().then((st) => {
      if (!st) return;
      const by = new Map((st.by_flight || []).map((r) => [r.flight_id, +r.particles]));
      let n = 0;
      for (const f of flights) { const v = by.get(f.id) || 0; n += v; doneCell.get(f.id).textContent = fmtInt(v); }
      classifiedFact.hidden = false;
      classifiedFact.lastChild.textContent = `${fmtInt(n)} (${total ? (100 * n / total).toFixed(n && 100 * n / total < 1 ? 1 : 0) : 0}%)`;
    }).catch(() => {});
  }
  if (info.updated_utc) inner.append(h('p', { class: 'muted small projupdated' }, `Page last edited ${info.updated_utc.slice(0, 10)}${info.updated_by ? ' by ' + info.updated_by : ''}`));
}

function svgIcon(kind) {
  const p = { edit: '<path d="M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>',
    ext: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
    up: '<path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>' }[kind];
  return C.h('svg', { class: 'ico ico-' + kind, viewBox: '0 0 24 24', width: 16, height: 16, 'aria-hidden': 'true', html: p });
}

// ------------------------------------------------------------------ gallery + lightbox
function gallery(name, imgs) {
  const { h } = C;
  const url = (f) => base(name) + f;
  const g = h('div', { class: 'projgallery n' + Math.min(imgs.length, 5) });
  imgs.slice(0, 5).forEach((im, i) => {
    const more = i === 4 && imgs.length > 5 ? imgs.length - 5 : 0;
    g.append(h('button', { class: 'gimg' + (i === 0 ? ' lead' : ''), 'aria-label': `Open picture ${i + 1} of ${imgs.length}${im.caption ? ': ' + im.caption : ''}`, onclick: () => lightbox(name, imgs, i) },
      h('img', { src: url(i === 0 && imgs.length > 1 || imgs.length === 1 ? im.file : im.thumb || im.file), alt: im.caption || '', loading: 'lazy' }),
      im.caption && i === 0 ? h('span', { class: 'gcap' }, im.caption) : null,
      more ? h('span', { class: 'gmore' }, `+${more} more`) : null));
  });
  return g;
}
function lightbox(name, imgs, start) {
  const { h } = C;
  let i = start;
  const img = h('img', { alt: '' });
  const cap = h('p', { class: 'lbcap' });
  const count = h('span', { class: 'lbcount' });
  const show = () => {
    const im = imgs[i];
    img.src = base(name) + im.file; img.alt = im.caption || '';
    cap.innerHTML = '';
    if (im.caption) cap.append(im.caption);
    if (im.credit) cap.append(h('span', { class: 'lbcredit' }, (im.caption ? ' · ' : '') + im.credit));
    count.textContent = imgs.length > 1 ? `${i + 1} / ${imgs.length}` : '';
  };
  const step = (k) => { i = (i + k + imgs.length) % imgs.length; show(); };
  const d = h('dialog', { class: 'lightbox', onclick: (e) => { if (e.target === d || e.target.classList.contains('lbstage')) d.close(); } },
    h('div', { class: 'lbstage' }, img),
    h('div', { class: 'lbbar' }, count, cap, h('span', { class: 'grow' }),
      h('a', { class: 'btn small ghost', href: '#', onclick: (e) => { e.preventDefault(); window.open(base(name) + imgs[i].file, '_blank', 'noopener'); } }, 'Full size'),
      h('button', { class: 'btn small ghost', onclick: () => d.close(), 'aria-label': 'Close' }, '✕')),
    imgs.length > 1 ? h('button', { class: 'lbnav prev', 'aria-label': 'Previous picture', onclick: () => step(-1) }, '‹') : null,
    imgs.length > 1 ? h('button', { class: 'lbnav next', 'aria-label': 'Next picture', onclick: () => step(1) }, '›') : null);
  d.addEventListener('keydown', (e) => { if (e.key === 'ArrowLeft') step(-1); else if (e.key === 'ArrowRight') step(1); });
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  show();
  d.showModal();
}

// ------------------------------------------------------------------ editor (administrators)
function editProject(root, name, info0) {
  const { h, toast } = C;
  const info = JSON.parse(JSON.stringify(info0));
  info.links = info.links || []; info.images = info.images || [];
  root.innerHTML = '';
  const inner = h('div', { class: 'page-inner projpage projedit' });
  root.append(inner);
  let dirty = false;
  const touch = () => { dirty = true; };
  const titleIn = h('input', { type: 'text', maxlength: 160, value: info.title || '', placeholder: name, oninput: touch });
  const summaryIn = h('textarea', { rows: 2, maxlength: 400, placeholder: 'One or two sentences: what, where and when. Shown under the title and on the Flights page.', oninput: touch }, info.summary || '');
  const bodyIn = h('textarea', { rows: 12, maxlength: 20000, class: 'bodyin', oninput: () => { touch(); preview.innerHTML = richText(bodyIn.value) || '<p class="muted">The preview appears here.</p>'; } }, info.body || '');
  const preview = h('div', { class: 'prose preview', html: richText(info.body) || '<p class="muted">The preview appears here.</p>' });

  // links
  const linkBox = h('div', { class: 'linkrows' });
  const drawLinks = () => {
    linkBox.innerHTML = '';
    info.links.forEach((l, i) => {
      const row = h('div', { class: 'linkrow' },
        h('input', { type: 'text', placeholder: 'Title, e.g. Campaign website', value: l.title || '', 'aria-label': 'Link title', oninput: (e) => { l.title = e.target.value; touch(); } }),
        h('input', { type: 'url', placeholder: 'https://…', value: l.url || '', 'aria-label': 'Address', oninput: (e) => { l.url = e.target.value; touch(); } }),
        h('input', { type: 'text', placeholder: 'Note (optional), e.g. Overview paper, QJRMS 2009', value: l.note || '', 'aria-label': 'Note', class: 'wide', oninput: (e) => { l.note = e.target.value; touch(); } }),
        h('div', { class: 'rowbtns' },
          h('button', { type: 'button', class: 'iconbtn', title: 'Move up', disabled: i === 0, onclick: () => { [info.links[i - 1], info.links[i]] = [info.links[i], info.links[i - 1]]; touch(); drawLinks(); } }, '↑'),
          h('button', { type: 'button', class: 'iconbtn', title: 'Move down', disabled: i === info.links.length - 1, onclick: () => { [info.links[i + 1], info.links[i]] = [info.links[i], info.links[i + 1]]; touch(); drawLinks(); } }, '↓'),
          h('button', { type: 'button', class: 'iconbtn danger', title: 'Remove this link', onclick: () => { info.links.splice(i, 1); touch(); drawLinks(); } }, '✕')));
      linkBox.append(row);
    });
    linkBox.append(h('button', { type: 'button', class: 'btn small', onclick: () => { info.links.push({ title: '', url: '', note: '' }); drawLinks(); linkBox.querySelector('.linkrow:last-of-type input')?.focus(); } }, '+ Add a link'));
  };
  drawLinks();

  // pictures
  const picBox = h('div', { class: 'picgrid' });
  const fileIn = h('input', { type: 'file', accept: 'image/jpeg,image/png,image/gif,image/webp', multiple: true, hidden: true, onchange: () => { upload([...fileIn.files]); fileIn.value = ''; } });
  const drop = h('div', { class: 'dropzone picdrop', tabindex: 0, role: 'button',
    onclick: () => fileIn.click(), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileIn.click(); } },
    ondragover: (e) => { e.preventDefault(); drop.classList.add('over'); }, ondragleave: () => drop.classList.remove('over'),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove('over'); upload([...e.dataTransfer.files].filter((f) => f.type.startsWith('image/'))); } },
  svgIcon('up'), h('b', {}, 'Drop pictures here, or click to choose'), h('span', { class: 'muted small' }, 'JPEG, PNG, GIF or WebP, up to 25 MB each. Large photos are shrunk to 2000 pixels. The first picture is the cover.'));
  let dragFrom = null;
  const drawPics = () => {
    picBox.innerHTML = '';
    info.images.forEach((im, i) => {
      const card = h('div', { class: 'piccard' + (i === 0 ? ' cover' : ''), draggable: 'true',
        ondragstart: (e) => { dragFrom = i; e.dataTransfer.effectAllowed = 'move'; card.classList.add('dragging'); },
        ondragend: () => card.classList.remove('dragging'),
        ondragover: (e) => { if (dragFrom !== null) { e.preventDefault(); card.classList.add('dropon'); } },
        ondragleave: () => card.classList.remove('dropon'),
        ondrop: (e) => { e.preventDefault(); card.classList.remove('dropon'); if (dragFrom === null || dragFrom === i) return; const [m] = info.images.splice(dragFrom, 1); info.images.splice(i, 0, m); dragFrom = null; touch(); drawPics(); } },
      h('div', { class: 'picthumb' }, h('img', { src: base(name) + (im.thumb || im.file), alt: '' }), i === 0 ? h('span', { class: 'pill tiny coverpill' }, 'cover') : null),
      h('input', { type: 'text', placeholder: 'Caption', value: im.caption || '', 'aria-label': 'Caption', oninput: (e) => { im.caption = e.target.value; touch(); } }),
      h('input', { type: 'text', placeholder: 'Credit (optional)', value: im.credit || '', 'aria-label': 'Credit', class: 'credit', oninput: (e) => { im.credit = e.target.value; touch(); } }),
      h('div', { class: 'rowbtns' },
        h('button', { type: 'button', class: 'iconbtn', title: 'Move earlier', disabled: i === 0, onclick: () => { [info.images[i - 1], info.images[i]] = [info.images[i], info.images[i - 1]]; touch(); drawPics(); } }, '←'),
        h('button', { type: 'button', class: 'iconbtn', title: 'Move later', disabled: i === info.images.length - 1, onclick: () => { [info.images[i + 1], info.images[i]] = [info.images[i], info.images[i + 1]]; touch(); drawPics(); } }, '→'),
        h('span', { class: 'grow' }),
        h('button', { type: 'button', class: 'iconbtn danger', title: 'Delete this picture', onclick: async () => {
          if (!confirm('Delete this picture? It is removed from the server straight away.')) return;
          try {
            const r = await C.api.adminPost('project_image_delete', { project: name, file: im.file });
            info.images = info.images.filter((x) => x.file !== im.file);
            Object.assign(info0, { images: r.info.images });
            drawPics(); C.projectsChanged?.();
          } catch (e) { toast(e.message, 4000); }
        } }, '✕')));
      picBox.append(card);
    });
  };
  drawPics();
  async function upload(files) {
    for (const f of files) {
      const prog = h('div', { class: 'piccard uploading' }, h('div', { class: 'picthumb' }, h('span', { class: 'muted small' }, f.name)), h('div', { class: 'progress-track' }, h('i', { style: { width: '0%' } })));
      picBox.append(prog);
      try {
        const r = await postFile(name, f, (p) => { prog.querySelector('i').style.width = Math.round(p * 100) + '%'; });
        info.images.push(r.image);
        info0.images = r.info.images;
        C.projectsChanged?.();
      } catch (e) { toast(`${f.name}: ${e.message}`, 5000); }
      prog.remove();
      drawPics();
    }
  }

  // notes for each flight in the project
  const flights = C.visibleFlights().filter((f) => (f.campaign || '') === name);
  const notes = Object.assign({}, info.flight_notes || {});
  const notesSec = flights.length ? h('section', { class: 'card editsec' }, h('h2', {}, 'Flight notes'),
    h('p', { class: 'muted small' }, 'Anything people should know about a flight, shown with it on this page, the Flights page and when classifying. Leave empty if there’s nothing to say.'),
    h('div', { class: 'notegrid' }, flights.map((f) => h('label', { class: 'noterow' },
      h('span', {}, h('b', {}, f.label || f.id), h('small', { class: 'muted' }, [f.date, `${C.fmtInt(f.n)} particles`].filter(Boolean).join(' · '))),
      h('textarea', { rows: 2, maxlength: 4000, placeholder: 'e.g. CPI window iced over after 14:20 · lots of shattering on the first profile · ground test only',
        oninput: (e) => { notes[f.id] = e.target.value; touch(); } }, notes[f.id] || ''))))) : null;

  const err = h('p', { class: 'formerr', role: 'alert' });
  const save = async () => {
    err.textContent = '';
    try {
      const r = await C.api.adminPost('project_save', {
        project: name, title: titleIn.value, summary: summaryIn.value, body: bodyIn.value,
        links: info.links.filter((l) => (l.url || '').trim()),
        flight_notes: notes,
        images: info.images.map((im) => ({ file: im.file, caption: im.caption || '', credit: im.credit || '' })) });
      dirty = false;
      C.projectsChanged?.();
      toast('Project page saved');
      viewProject(root, name);
      return r;
    } catch (e) { err.textContent = e.message; window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }); }
  };
  const cancel = () => { if (dirty && !confirm('Leave without saving your changes?')) return; viewProject(root, name); };
  const fld = (label, el, hint) => h('label', { class: 'fld' }, h('span', {}, label, hint ? h('small', {}, ' ' + hint) : null), el);

  inner.append(
    h('p', { class: 'crumbs' }, h('a', { href: '#/flights' }, 'Flights'), ' › ', h('a', { href: projectHref(name), onclick: (e) => { e.preventDefault(); cancel(); } }, name), ' › Edit'),
    h('div', { class: 'projhead' }, h('h1', { class: 'grow' }, 'Edit project page'),
      h('div', { class: 'projactions' }, h('button', { class: 'btn ghost', onclick: cancel }, 'Cancel'), h('button', { class: 'btn primary', onclick: save }, 'Save'))),
    h('section', { class: 'card editsec' },
      fld('Title', titleIn, `(leave empty to use “${name}”)`),
      fld('Summary', summaryIn)),
    h('section', { class: 'card editsec' }, h('h2', {}, 'Pictures'),
      h('p', { class: 'muted small' }, 'Pictures are added and deleted straight away. Drag them, or use the arrows, to change the order. Captions and order are kept when you save.'),
      picBox, drop, fileIn),
    h('section', { class: 'card editsec' }, h('h2', {}, 'Description'),
      h('div', { class: 'bodyedit' },
        h('div', {}, bodyIn, h('p', { class: 'muted small fmt' }, 'Blank line = new paragraph · ', h('code', {}, '- '), ' list · ', h('code', {}, '## '), ' heading · ', h('code', {}, '**bold**'), ' · ', h('code', {}, '*italic*'), ' · ', h('code', {}, '[text](https://…)'))),
        h('div', {}, h('div', { class: 'previewlbl muted small' }, 'Preview'), preview))),
    h('section', { class: 'card editsec' }, h('h2', {}, 'Links'),
      h('p', { class: 'muted small' }, 'Campaign website, papers, data archive, flight reports, contacts (mailto:)…'),
      linkBox),
    notesSec,
    err,
    h('div', { class: 'editbar' }, h('button', { class: 'btn ghost', onclick: cancel }, 'Cancel'), h('button', { class: 'btn primary', onclick: save }, 'Save')));
  titleIn.focus();
}

function postFile(project, file, onProgress) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append('project', project); fd.append('file', file);
    const x = new XMLHttpRequest();
    x.open('POST', 'api/?a=admin_project_image');
    x.setRequestHeader('X-Requested-With', 'cpi-classifier');
    x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    x.onload = () => {
      let d = null; try { d = JSON.parse(x.responseText); } catch { /* not JSON */ }
      if (x.status === 413) reject(new Error('Too big for the server (raise post_max_size / upload_max_filesize, or LimitRequestBody)'));
      else if (!d || d.ok === false || x.status >= 400) reject(new Error(d?.error || `Server error (${x.status})`));
      else resolve(d);
    };
    x.onerror = () => reject(new Error('The server could not be reached'));
    x.send(fd);
  });
}

// ------------------------------------------------------------------ Flights page: project cards
export async function projectCards(groups) {
  const { h, fmtInt, server } = C;
  const idx = await projectIndex();
  const named = [...groups].filter(([p]) => p);
  if (!named.length) return null;
  const wrap = h('div', { class: 'projcards' });
  for (const [p, list] of named) {
    const meta = idx[p] || {};
    const dates = list.map((f) => f.date).filter(Boolean).sort();
    const yr = dates.length ? (dates[0].slice(0, 4) === dates[dates.length - 1].slice(0, 4) ? dates[0].slice(0, 4) : `${dates[0].slice(0, 4)}–${dates[dates.length - 1].slice(0, 4)}`) : '';
    wrap.append(h('a', { class: 'projcard' + (meta.cover ? '' : ' nocover'), href: projectHref(p) },
      h('div', { class: 'pcover' }, meta.cover ? h('img', { src: `data/projects/${meta.slug}/${meta.cover}`, alt: '', loading: 'lazy' }) : h('svg', { viewBox: '0 0 32 32', class: 'pflake', html: '<use href="#flake"/>' })),
      h('div', { class: 'pbody' },
        h('b', {}, meta.title || p), meta.title && meta.title !== p ? h('span', { class: 'muted small' }, ' ' + p) : null,
        meta.summary ? h('p', {}, meta.summary) : (server.user?.is_admin ? h('p', { class: 'muted' }, 'Add a description, links and pictures →') : null),
        h('span', { class: 'muted small' }, [yr, `${list.length} flight${list.length === 1 ? '' : 's'}`, `${fmtInt(list.reduce((a, f) => a + f.n, 0))} particles`].filter(Boolean).join(' · ')))));
  }
  return wrap;
}
