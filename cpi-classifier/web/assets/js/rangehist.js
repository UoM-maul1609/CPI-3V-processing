// A histogram with two draggable handles: shows the distribution of one
// property and lets you choose a range of it.

const NB = 44;
const DOMAINS = new WeakMap();
const BINS = new WeakMap();      // column array -> {key, b: Uint8Array of bin numbers}   // column array -> {lin|log: {lo, hi, empty}}
const H = 52;

export class RangeHist {
  /**
   * opts: {key, label, unit, log, format(v), parse(str), removable, onChange(range), onRemove(), headExtra(el)}
   */
  constructor(opts) {
    this.o = opts;
    this.lo = 0; this.hi = 1;          // domain
    this.range = [null, null];         // null = open end
    this.all = new Float64Array(NB);   // counts passing the other filters
    this.el = document.createElement('div');
    this.el.className = 'rh';
    this.el.innerHTML = `
      <div class="rh-head"><span class="rh-label"></span><span class="rh-tip" aria-live="polite"></span><span class="rh-tools"></span></div>
      <svg class="rh-svg" role="img"></svg>
      <div class="rh-inputs"><input aria-label="Minimum" placeholder="min"><span>to</span><input aria-label="Maximum" placeholder="max"></div>`;
    this.svg = this.el.querySelector('svg');
    this.inputs = this.el.querySelectorAll('input');
    const lab = this.el.querySelector('.rh-label');
    if (opts.headExtra) opts.headExtra(lab); else lab.textContent = opts.label + (opts.unit ? ` (${opts.unit})` : '');
    if (opts.removable) {
      const x = document.createElement('button');
      x.className = 'rh-x'; x.title = 'Remove this filter'; x.setAttribute('aria-label', 'Remove filter ' + opts.label);
      x.textContent = '×';
      x.onclick = () => opts.onRemove && opts.onRemove();
      this.el.querySelector('.rh-tools').append(x);
    }
    this.inputs.forEach((inp, k) => inp.addEventListener('change', () => {
      const s = inp.value.trim();
      let v = s === '' ? null : (opts.parse ? opts.parse(s, this) : parseFloat(s));
      if (v !== null && !Number.isFinite(v)) v = null;
      this.range[k] = v;
      this._emit(); this.draw();
    }));
    this._bindDrag();
    new ResizeObserver(() => this.draw()).observe(this.el);
  }

  get active() { return this.range[0] !== null || this.range[1] !== null; }

  setDomain(values) {
    // robust domain: ignore the extreme 0.2% so one odd particle doesn't squash the plot.
    // A sample of up to 200,000 is plenty for that, and the answer is remembered per column,
    // so millions of particles don't mean seconds of sorting each time the page is drawn.
    const key = this.o.log ? 'log' : 'lin';
    let memo = DOMAINS.get(values);
    if (memo && memo[key]) { Object.assign(this, memo[key]); return; }
    const n = values.length;
    const step = Math.max(1, Math.floor(n / 200000));
    const v = new Float64Array(Math.ceil(n / step));
    let m = 0;
    for (let i = 0; i < n; i += step) { const x = values[i]; if (Number.isFinite(x) && (!this.o.log || x > 0)) v[m++] = x; }
    let out;
    if (!m) out = { lo: 0, hi: 1, empty: true };
    else {
      const w = v.subarray(0, m).sort();
      const q = (p) => w[Math.min(m - 1, Math.max(0, Math.round(p * (m - 1))))];
      let lo = q(0.002), hi = q(0.998);
      if (hi <= lo) hi = lo + (this.o.log ? lo : 1);
      out = { lo, hi, empty: false };
    }
    if (!memo) { memo = {}; DOMAINS.set(values, memo); }
    memo[key] = out;
    Object.assign(this, out);
  }

  tx(v) { // value -> 0..1
    if (this.o.log) return (Math.log(v) - Math.log(this.lo)) / (Math.log(this.hi) - Math.log(this.lo));
    return (v - this.lo) / (this.hi - this.lo);
  }
  inv(u) {
    if (this.o.log) return Math.exp(Math.log(this.lo) + u * (Math.log(this.hi) - Math.log(this.lo)));
    return this.lo + u * (this.hi - this.lo);
  }
  bin(v) {
    if (!Number.isFinite(v) || (this.o.log && v <= 0)) return -1;
    const b = Math.floor(this.tx(v) * NB);
    return b < 0 ? 0 : b >= NB ? NB - 1 : b;
  }
  /** Bin number of every value in a column (255 = no bin), worked out once and remembered. */
  binsOf(values) {
    const key = `${this.o.log ? 'log' : 'lin'}:${this.lo}:${this.hi}`;
    let memo = BINS.get(values);
    if (memo && memo.key === key) return memo.b;
    const b = new Uint8Array(values.length);
    for (let i = 0; i < values.length; i++) { const k = this.bin(values[i]); b[i] = k < 0 ? 255 : k; }
    BINS.set(values, { key, b });
    return b;
  }
  pass(v) {
    const [a, b] = this.range;
    if (a === null && b === null) return true;
    if (!Number.isFinite(v)) return false;
    return (a === null || v >= a) && (b === null || v <= b);
  }

  draw() {
    const W = this.svg.clientWidth || 260;
    this.W = W;
    const max = Math.max(1, ...this.all);
    const bw = W / NB;
    const [a, b] = this.range;
    const ua = a === null ? 0 : Math.max(0, Math.min(1, this.tx(a)));
    const ub = b === null ? 1 : Math.max(0, Math.min(1, this.tx(b)));
    let s = '';
    for (let i = 0; i < NB; i++) {
      const c = this.all[i];
      if (!c) continue;
      const h = Math.max(1.5, (Math.sqrt(c) / Math.sqrt(max)) * (H - 10));
      const mid = (i + 0.5) / NB;
      const inr = mid >= ua && mid <= ub;
      s += `<rect class="bar${inr ? ' in' : ''}" x="${(i * bw + 0.5).toFixed(1)}" y="${(H - 4 - h).toFixed(1)}" width="${Math.max(1, bw - 1).toFixed(1)}" height="${h.toFixed(1)}"/>`;
    }
    s += `<line class="axis" x1="0" x2="${W}" y1="${H - 3.5}" y2="${H - 3.5}"/>`;
    const xa = ua * W, xb = ub * W;
    s += `<rect class="handle" data-h="0" x="${Math.max(0, xa - 5)}" y="4" width="10" height="${H - 8}" rx="3"/>`;
    s += `<rect class="handle" data-h="1" x="${Math.min(W - 10, xb - 5)}" y="4" width="10" height="${H - 8}" rx="3"/>`;
    this.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    this.svg.innerHTML = s;
    this.svg.setAttribute('aria-label', `${this.o.label} distribution`);
    const f = this.o.format || ((v) => String(+v.toPrecision(3)));
    this.inputs[0].value = a === null ? '' : f(a);
    this.inputs[1].value = b === null ? '' : f(b);
    this.inputs[0].placeholder = this.empty ? 'no data' : f(this.lo);
    this.inputs[1].placeholder = this.empty ? '' : f(this.hi);
    this.el.classList.toggle('active', this.active);
  }

  _emit() {
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => this.o.onChange && this.o.onChange(this.range));
  }

  _bindDrag() {
    let dragging = -1;
    const uAt = (e) => {
      const r = this.svg.getBoundingClientRect();
      return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    };
    const setFromU = (k, u) => {
      let v = this.inv(u);
      if (k === 0 && u <= 0.003) v = null;
      if (k === 1 && u >= 0.997) v = null;
      const other = this.range[1 - k];
      if (v !== null && other !== null && (k === 0 ? v > other : v < other)) v = other;
      this.range[k] = v === null ? null : +v.toPrecision(4);
      this.draw();
      this._emit();
    };
    this.svg.addEventListener('pointerdown', (e) => {
      if (this.empty) return;
      const u = uAt(e);
      const [a, b] = this.range;
      const ua = a === null ? 0 : this.tx(a), ub = b === null ? 1 : this.tx(b);
      dragging = Math.abs(u - ua) <= Math.abs(u - ub) ? 0 : 1;
      if (Math.abs(ua - ub) < 0.01) dragging = u < ua ? 0 : 1;
      this.svg.setPointerCapture(e.pointerId);
      setFromU(dragging, u);
    });
    const tip = this.el.querySelector('.rh-tip');
    const f = (v) => (this.o.format || ((x) => String(+x.toPrecision(3))))(v);
    this.svg.addEventListener('pointermove', (e) => {
      if (dragging >= 0) { setFromU(dragging, uAt(e)); tip.textContent = ''; return; }
      if (this.empty) return;
      const i = Math.min(NB - 1, Math.floor(uAt(e) * NB));
      tip.textContent = `${f(this.inv(i / NB))}–${f(this.inv((i + 1) / NB))}: ${Math.round(this.all[i]).toLocaleString()}`;
    });
    this.svg.addEventListener('pointerleave', () => { tip.textContent = ''; });
    const end = () => { dragging = -1; };
    this.svg.addEventListener('pointerup', end);
    this.svg.addEventListener('pointercancel', end);
    this.svg.addEventListener('dblclick', () => { this.range = [null, null]; this.draw(); this._emit(); });
  }
}
