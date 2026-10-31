#!/usr/bin/env python3
"""
cpi_export.py - turn processed CPI MAT files into data for the web classifier.

One "flight" (or ground/lab session) is described by a small JSON file
(see flight.example.json). It lists one or more MAT files produced by the
python/proc pipeline, plus metadata and, optionally, a navigation / met file
whose time series are interpolated onto every particle.

    python cpi_export.py path/to/flight.json --out ../web/data
    python cpi_export.py path/to/flight.json --out ../web/data --force
    python cpi_export.py --manifest-only --out ../web/data
    python cpi_export.py path/to/flight.json --colormap map2 --force --out ../web/data
    python cpi_export.py path/to/flight.json --colormap-preview preview.html
    python cpi_export.py --list-colormaps

Colour maps
    Images are written in grey by default; the web page can then recolour
    them live with any map. Choose a map at export time ("colormap" in
    flight.json, or --colormap) to bake it into the PNGs instead. Names:
    the maps in colormaps.json (gray, negative, bone, ice, map2, map, blues,
    reds, viridis, cividis, magma, inferno, plasma, turbo, cubehelix), any
    matplotlib map if matplotlib is installed, "file.mat:MAP2" for a 256x3
    table in a MAT file, or a .json/.csv file of 256 RGB rows. Add "_r" to
    reverse a map. The PNGs stay the same size: a colour map is stored as
    a 256-entry PNG palette.

Output (relative to --out):

    manifest.json                          list of all exported flights
    flights/<id>/meta.json                 metadata for one flight
    flights/<id>/catalog.json              particle properties (column arrays)
    flights/<id>/img/<stem>/<idx>.png      one greyscale PNG per particle
    flights/<id>/bnd/<stem>.json           particle outlines, loaded on demand

<idx> is the 0-based index into the arrays of the MAT file (dat.len[idx] in
Python, dat.len(idx+1) in MATLAB). A particle's id everywhere (web page,
database, CSV export) is "<flight id>:<stem>:<idx>".

Needs numpy and scipy only. netCDF navigation files also need netCDF4.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import struct
import sys
import zlib

import numpy as np
import scipy.io as sio

FORMAT_VERSION = 1
MATLAB_UNIX_EPOCH = 719529.0  # datenum of 1970-01-01

# ---------------------------------------------------------------------------
# Particle properties copied from `dat` into the catalogue.
# (key in catalogue, source field in dat, label, unit, decimals, group)
# ---------------------------------------------------------------------------
FIELDS = [
    ("len", "len", "Maximum length", "µm", 1, "size"),
    ("wid", "wid", "Width", "µm", 1, "size"),
    ("deq", "deq", "Equivalent-area diameter", "µm", 1, "size"),
    ("len_feret", "len_feret", "Feret length", "µm", 1, "size"),
    ("wid_feret", "wid_feret", "Feret width", "µm", 1, "size"),
    ("len_ellipse", "len_ellipse", "Ellipse major axis", "µm", 1, "size"),
    ("wid_ellipse", "wid_ellipse", "Ellipse minor axis", "µm", 1, "size"),
    ("shadow_deq", "shadow_deq", "Shadow-circle diameter", "µm", 1, "size"),
    ("area", "area", "Projected area", "µm²", 0, "size"),
    ("perimeter", "perimeter", "Perimeter", "µm", 1, "size"),
    ("round", "round", "Roundness", "", 3, "shape"),
    ("area_ratio", "area_ratio_circle", "Area ratio (circle)", "", 3, "shape"),
    ("orientation", "orientation", "Orientation", "°", 1, "shape"),
    ("interior_bright", "interior_bright_fraction", "Interior bright fraction", "", 3, "shape"),
    ("T_min", "T_min", "Minimum transmittance", "", 3, "quality"),
    ("contrast", "contrast_robust_T", "Robust contrast (T)", "", 3, "quality"),
    ("edge_touch", "edge_touch", "Touches ROI edge", "", 0, "quality"),
    ("cutoff_pct", "cutoff_pct", "Boundary on ROI edge", "%", 1, "quality"),
    ("imageType", "imageType", "Image type", "", 0, "instrument"),
]
# Derived / non-dat columns (filled in code)
EXTRA_FIELDS = {
    "aspect": ("Aspect ratio (L/W)", "", 3, "shape"),
    "focus": ("Focus (edge gradient)", "", 3, "quality"),
    "focus_T": ("Focus (on T)", "", 4, "quality"),
    "x0": ("ROI x in frame", "px", 0, "instrument"),
    "y0": ("ROI y in frame", "px", 0, "instrument"),
    "w": ("Image width", "px", 0, "instrument"),
    "h": ("Image height", "px", 0, "instrument"),
    "lat": ("Latitude", "°N", 5, "position"),
    "lon": ("Longitude", "°E", 5, "position"),
    "alt": ("Altitude", "m", 0, "position"),
    "temp": ("Temperature", "°C", 2, "position"),
}
NAV_KEYS = ("lat", "lon", "alt", "temp")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(*a):
    print(*a, file=sys.stderr, flush=True)


# Machine-readable progress for the web admin page (--progress):
#   @progress {"stage": "images", "file": 1, "files": 3, "stem": "...", "done": 250, "total": 6011}
PROGRESS = {"on": False, "file": 0, "files": 0}


def progress(**kw):
    if PROGRESS["on"]:
        kw.setdefault("file", PROGRESS["file"])
        kw.setdefault("files", PROGRESS["files"])
        print("@progress " + json.dumps(kw), file=sys.stderr, flush=True)


def write_png_gray(path: str, img: np.ndarray) -> None:
    """Minimal 8-bit greyscale PNG writer (no Pillow needed)."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(png)
    os.replace(tmp, path)


def write_png_indexed(path: str, img: np.ndarray, palette: np.ndarray) -> None:
    """8-bit palette PNG: same size as grey, but shown through a colour table."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
           + chunk(b"PLTE", np.ascontiguousarray(palette, dtype=np.uint8).tobytes())
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(png)
    os.replace(tmp, path)


def png_rgb_bytes(img: np.ndarray) -> bytes:
    """RGB PNG in memory (for the preview page)."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))


def builtin_colormaps() -> dict:
    p = os.path.join(HERE, "colormaps.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        maps = json.load(fh)["maps"]
    out = {}
    for m in maps:
        hx = m["hex"]
        out[m["id"]] = {"name": m["name"], "desc": m.get("desc", ""),
                        "rgb": np.array([[int(hx[i:i + 2], 16) for i in (k, k + 2, k + 4)]
                                         for k in range(0, len(hx), 6)], dtype=np.uint8)}
    return out


def _to256(rgb) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] < 3:
        raise SystemExit("a colour table must have rows of R, G, B")
    rgb = rgb[:, :3]
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    if rgb.shape[0] != 256:   # resample any length to 256
        x = np.linspace(0, 1, rgb.shape[0])
        xi = np.linspace(0, 1, 256)
        rgb = np.column_stack([np.interp(xi, x, rgb[:, c]) for c in range(3)])
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def resolve_colormap(spec, base="."):  # -> (name, lut or None)
    """Return (id, 256x3 uint8 table) for a colour map spec; table None means plain grey."""
    if not spec or spec in ("gray", "grey"):
        return "gray", None
    spec = str(spec)
    rev = spec.endswith("_r") and not spec.endswith(".mat_r")
    name = spec[:-2] if rev else spec
    table = None
    built = builtin_colormaps()
    if name in built:
        table = built[name]["rgb"]
    elif ":" in name and name.split(":")[0].endswith(".mat"):
        f, var = name.rsplit(":", 1)
        f = f if os.path.isabs(f) else os.path.join(base, f)
        table = _to256(sio.loadmat(f, variable_names=[var])[var])
    elif name.endswith(".json"):
        f = name if os.path.isabs(name) else os.path.join(base, name)
        with open(f) as fh:
            table = _to256(json.load(fh))
    elif name.endswith(".csv"):
        f = name if os.path.isabs(name) else os.path.join(base, name)
        table = _to256(np.loadtxt(f, delimiter=",", ndmin=2))
    else:
        try:
            from matplotlib import colormaps as mpl_maps
            table = _to256(mpl_maps[name](np.linspace(0, 1, 256)))
        except Exception:
            raise SystemExit(f"Unknown colour map '{spec}'. Run --list-colormaps to see the choices.")
    if rev:
        table = table[::-1].copy()
    return spec, table


def apply_stretch(im: np.ndarray, mode) -> np.ndarray:
    """Optional contrast stretch: 'per-image' (1-99th percentile of each ROI) or None."""
    if mode in (None, False, "none", "absolute"):
        return im
    lo, hi = np.percentile(im, [1, 99])
    if hi <= lo:
        return im
    return np.clip((im.astype(float) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def write_json(path: str, obj, compact=True) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        if compact:
            json.dump(obj, fh, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        else:
            json.dump(obj, fh, indent=1, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def col_to_list(a: np.ndarray, decimals: int):
    """Round a float column and turn NaN/inf into JSON null."""
    a = np.asarray(a, dtype=float)
    out = []
    if decimals == 0:
        for v in a:
            out.append(int(round(v)) if math.isfinite(v) else None)
    else:
        r = np.round(a, decimals)
        for v in r:
            out.append(float(v) if math.isfinite(v) else None)
    return out


def write_catalog(flight_dir, fid, stems, n, fields_meta, cols):
    """Write the particle table as catalog.json (a small header) + catalog.bin (the columns).

    cols: name -> numpy array. f and i (file and image index) are stored as uint32,
    t (seconds since 1970) as float64, everything else as float32 with NaN for
    missing. The browser reads catalog.bin straight into typed arrays: far quicker
    to load, and much smaller, than a column of JSON numbers for millions of particles."""
    layout, parts, off = {}, [], 0
    for name, a in cols.items():
        dtype = "<u4" if name in ("f", "i") else "<f8" if name == "t" else "<f4"
        arr = np.ascontiguousarray(np.asarray(a, dtype=float if dtype != "<u4" else np.int64).astype(dtype))
        pad = (-off) % 8
        if pad:
            parts.append(b"\0" * pad); off += pad
        layout[name] = [{"<u4": "u32", "<f8": "f64", "<f4": "f32"}[dtype], off]
        b = arr.tobytes()
        parts.append(b); off += len(b)
    tmp = os.path.join(flight_dir, "catalog.bin.tmp")
    with open(tmp, "wb") as fh:
        for b in parts:
            fh.write(b)
    os.replace(tmp, os.path.join(flight_dir, "catalog.bin"))
    write_json(os.path.join(flight_dir, "catalog.json"),
               {"version": 2, "flight": fid, "files": stems, "n": int(n), "fields": fields_meta,
                "bin": {"file": "catalog.bin", "bytes": off, "cols": layout}})


def convert_catalogs(data_dir):
    """Rewrite older catalog.json files (JSON columns) in the binary form, in place."""
    done = 0
    for cj in sorted(glob.glob(os.path.join(data_dir, "flights", "*", "catalog.json"))):
        with open(cj, encoding="utf-8") as fh:
            c = json.load(fh)
        if "cols" not in c:
            continue
        cols = {k: np.array([np.nan if v is None else v for v in vals], dtype=float) for k, vals in c["cols"].items()}
        write_catalog(os.path.dirname(cj), c["flight"], c["files"], c["n"], c["fields"], cols)
        done += 1
        log(f"  {c['flight']}: {c['n']} particles")
    log(f"Converted {done} catalogue(s)")


def datenum_to_unix(dn):
    return (np.asarray(dn, dtype=float) - MATLAB_UNIX_EPOCH) * 86400.0


def iso(ts: float | None) -> str | None:
    if ts is None or not math.isfinite(ts):
        return None
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


# ---------------------------------------------------------------------------
# Navigation / met data
# ---------------------------------------------------------------------------
def parse_time_values(values, units: str) -> np.ndarray:
    """Convert a list/array of times to unix seconds.

    units: "iso" | "unix" | "datenum" | "seconds since <iso date>"
           | "hours since <iso>" | "days since <iso>"
    """
    u = (units or "iso").strip()
    if u == "iso":
        out = []
        for v in values:
            s = str(v).strip().replace("Z", "+00:00")
            t = dt.datetime.fromisoformat(s)
            if t.tzinfo is None:
                t = t.replace(tzinfo=dt.timezone.utc)
            out.append(t.timestamp())
        return np.array(out, dtype=float)
    vals = np.asarray(values, dtype=float)
    if u == "unix":
        return vals
    if u == "datenum":
        return datenum_to_unix(vals)
    m = re.match(r"^(seconds|minutes|hours|days)\s+since\s+(.+)$", u)
    if m:
        scale = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[m.group(1)]
        ref = m.group(2).strip().replace("Z", "+00:00").replace(" +0000", "+00:00")
        ref = ref.replace(" ", "T", 1) if "T" not in ref and len(ref) > 10 else ref
        t0 = dt.datetime.fromisoformat(ref)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=dt.timezone.utc)
        return t0.timestamp() + vals * scale
    raise ValueError(f"Unknown time units '{units}'")


def read_nav_csv(path: str, cfg: dict) -> dict:
    tcol = cfg.get("time_column", "time")
    fields = cfg.get("fields", {})
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"{path}: no rows")
    t = parse_time_values([r[tcol] for r in rows], cfg.get("time_units", "iso"))
    out = {"t": t}
    for key, col in fields.items():
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, TypeError, KeyError):
                vals.append(np.nan)
        out[key] = np.array(vals)
    return out


def read_nav_netcdf(path: str, cfg: dict) -> dict:
    try:
        import netCDF4  # noqa
    except ImportError:
        raise SystemExit("Reading netCDF navigation needs the netCDF4 package (pip install netCDF4)")
    ds = netCDF4.Dataset(path)
    tvar = ds.variables[cfg.get("time_column", "Time")]
    units = cfg.get("time_units") or getattr(tvar, "units", "seconds since 1970-01-01")
    t = parse_time_values(np.asarray(tvar[:], dtype=float), units)
    out = {"t": None}
    series = {}
    for key, name in cfg.get("fields", {}).items():
        v = np.ma.filled(np.asarray(ds.variables[name][:], dtype=float), np.nan)
        series[key] = v
    ds.close()
    # FAAM-style (time, sps) arrays: expand time to the sample rate
    t_full = t
    for key, v in series.items():
        if v.ndim == 2:
            sps = v.shape[1]
            tt = (t[:, None] + np.arange(sps)[None, :] / sps).ravel()
            series[key] = (tt, v.ravel())
        else:
            series[key] = (t_full, v)
    out["series"] = series
    return out


def interp_nav(nav: dict, t_particles: np.ndarray, max_gap: float) -> dict:
    """Interpolate navigation series onto particle times; NaN where no data near."""
    res = {}
    if "series" in nav:
        series = nav["series"]
    else:
        series = {k: (nav["t"], v) for k, v in nav.items() if k != "t"}
    for key, (t, v) in series.items():
        good = np.isfinite(t) & np.isfinite(v)
        t, v = t[good], v[good]
        if t.size < 2:
            res[key] = np.full(t_particles.shape, np.nan)
            continue
        order = np.argsort(t)
        t, v = t[order], v[order]
        if key == "lon":
            v = np.rad2deg(np.unwrap(np.deg2rad(v)))
        y = np.interp(t_particles, t, v, left=np.nan, right=np.nan)
        if key == "lon":
            y = (y + 180.0) % 360.0 - 180.0
        # distance to nearest sample: blank out interpolation across gaps
        pos = np.clip(np.searchsorted(t, t_particles), 1, t.size - 1)
        gap = np.minimum(np.abs(t_particles - t[pos - 1]), np.abs(t[pos] - t_particles))
        y[gap > max_gap] = np.nan
        res[key] = y
    return res


# ---------------------------------------------------------------------------
# MAT reading
# ---------------------------------------------------------------------------
def load_mat(path: str):
    try:
        m = sio.loadmat(path, variable_names=["ROI_N", "dat"], squeeze_me=True,
                        struct_as_record=False)
    except NotImplementedError:
        raise SystemExit(f"{path} is a MATLAB v7.3 (HDF5) file. Save it with "
                         "save(..., '-v7') or export it from the Python pipeline.")
    if "dat" not in m or "ROI_N" not in m:
        raise SystemExit(f"{path}: needs both ROI_N and dat (run the stats step first)")
    return m["ROI_N"], m["dat"]


def field(d, name, n):
    if name in d._fieldnames:
        v = np.atleast_1d(np.asarray(getattr(d, name), dtype=float))
        if v.shape[0] == n:
            return v
    return np.full(n, np.nan)


def export_mat(path, stem, flight_dir, cfg, force, limit, transpose, palette=None, stretch=None):
    """Export one MAT file. Returns (columns dict, n_exported, proc_options)."""
    progress(stage="loading", stem=stem, done=0, total=0)
    R, d = load_mat(path)
    n = np.atleast_1d(R.Time).shape[0]
    sel_cfg = cfg.get("select", {})

    L = field(d, "len", n)
    valid = np.isfinite(L) & (field(d, "n_boundary", n) > 0)
    if not sel_cfg.get("valid_only", True):
        valid = np.ones(n, bool)
    size_field = sel_cfg.get("size_field", "len")
    size = field(d, size_field, n)
    if sel_cfg.get("min_size_um"):
        valid &= size >= float(sel_cfg["min_size_um"])
    if sel_cfg.get("max_size_um"):
        valid &= size <= float(sel_cfg["max_size_um"])
    if sel_cfg.get("exclude_edge_touch"):
        valid &= field(d, "edge_touch", n) == 0
    if sel_cfg.get("image_types"):
        valid &= np.isin(field(d, "imageType", n), sel_cfg["image_types"])
    idx = np.where(valid)[0]
    if limit:
        idx = idx[:limit]

    proc = {}
    if "proc_options" in d._fieldnames:
        try:
            proc = json.loads(str(d.proc_options))
        except Exception:
            proc = {"raw": str(d.proc_options)}

    cols = {}
    t = datenum_to_unix(field(d, "Time", n) if "Time" in d._fieldnames else np.atleast_1d(R.Time))
    cols["t"] = t[idx]
    for key, src, *_ in FIELDS:
        cols[key] = field(d, src, n)[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["aspect"] = cols["len"] / cols["wid"]
    cols["x0"] = np.atleast_1d(np.asarray(R.StartX, float))[idx]
    cols["y0"] = np.atleast_1d(np.asarray(R.StartY, float))[idx]

    foc = np.atleast_1d(d.foc) if "foc" in d._fieldnames else None
    focus = np.full(idx.size, np.nan)
    focus_T = np.full(idx.size, np.nan)
    w = np.zeros(idx.size)
    h = np.zeros(idx.size)
    boundaries = {}

    img_dir = os.path.join(flight_dir, "img", stem)
    os.makedirs(img_dir, exist_ok=True)
    images = np.atleast_1d(R.IMAGE)
    written = 0
    for j, k in enumerate(idx):
        im = np.asarray(images[k].IM)
        if im.ndim != 2:
            continue
        if transpose:
            im = im.T
        h[j], w[j] = im.shape
        png = os.path.join(img_dir, f"{k}.png")
        if force or not os.path.exists(png):
            im8 = apply_stretch(np.asarray(im, dtype=np.uint8), stretch)
            if palette is None:
                write_png_gray(png, im8)
            else:
                write_png_indexed(png, im8, palette)
            written += 1
        if foc is not None:
            f = foc[k]
            if hasattr(f, "_fieldnames"):
                if "focus" in f._fieldnames:
                    focus[j] = float(np.asarray(f.focus, float))
                if "focus_T" in f._fieldnames:
                    focus_T[j] = float(np.asarray(f.focus_T, float))
                if "boundaries" in f._fieldnames:
                    b = np.asarray(f.boundaries, float)
                    if b.ndim == 2 and b.shape[0] > 2:
                        # stored as (row, col) pixel centres; SVG wants x, y with
                        # pixel centres at +0.5
                        rows, cols_ = b[:, 0] + 0.5, b[:, 1] + 0.5
                        x, y = (rows, cols_) if transpose else (cols_, rows)
                        xy = np.round(np.column_stack([x, y]).ravel(), 1)
                        boundaries[str(int(k))] = [float(v) for v in xy]
        if (j + 1) % 2000 == 0:
            log(f"    {j + 1}/{idx.size} particles")
        if (j + 1) % 200 == 0 or j + 1 == idx.size:
            progress(stage="images", stem=stem, done=j + 1, total=int(idx.size))
    cols["focus"], cols["focus_T"], cols["w"], cols["h"] = focus, focus_T, w, h

    os.makedirs(os.path.join(flight_dir, "bnd"), exist_ok=True)
    write_json(os.path.join(flight_dir, "bnd", f"{stem}.json"), boundaries)
    cols["_idx"] = idx
    log(f"  {stem}: {idx.size} of {n} ROIs imported ({written} new images)")
    return cols, idx.size, proc


# ---------------------------------------------------------------------------
# Flight export
# ---------------------------------------------------------------------------
def resolve_mat_files(cfg, base):
    spec = cfg.get("mat_files") or []
    if isinstance(spec, str):
        spec = [spec]
    files = []
    for s in spec:
        p = s if os.path.isabs(s) else os.path.join(base, s)
        hits = sorted(glob.glob(p))
        if not hits:
            log(f"  warning: nothing matches {s}")
        files.extend(hits)
    return files


def export_flight(cfg_path, out, force=False, limit=None, colormap=None):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(cfg_path))
    cmap_id, palette = resolve_colormap(colormap or cfg.get("colormap"), base)
    stretch = cfg.get("stretch")
    fid = str(cfg.get("id", "")).strip()
    if not SAFE_ID.match(fid):
        raise SystemExit("flight 'id' must be letters/digits/._- (max 64 chars), e.g. 'B123' or 'MAN-2024-02-11'")
    transpose = bool(cfg.get("transpose", True))
    flight_dir = os.path.join(out, "flights", fid)
    os.makedirs(flight_dir, exist_ok=True)

    mats = resolve_mat_files(cfg, base)
    if not mats:
        raise SystemExit("no MAT files found - check 'mat_files' in " + cfg_path)
    log(f"Flight {fid}: {len(mats)} MAT file(s), colour map: {cmap_id}" + (f", stretch: {stretch}" if stretch else ""))
    old_meta = os.path.join(flight_dir, "meta.json")
    hidden = False
    if os.path.exists(old_meta):
        with open(old_meta, encoding="utf-8") as fh:
            hidden = bool(json.load(fh).get("hidden"))
    PROGRESS["files"] = len(mats)
    if os.path.exists(old_meta) and not force:
        with open(old_meta, encoding="utf-8") as fh:
            prev = json.load(fh)
        if (prev.get("colormap") or "gray") != cmap_id or prev.get("stretch") != stretch:
            log(f"  note: existing images use colour map '{prev.get('colormap') or 'gray'}'. "
                "Add --force to rewrite them with the new one.")
            cmap_id, stretch = prev.get("colormap") or "gray", prev.get("stretch")
            palette = resolve_colormap(cmap_id, base)[1]

    stems, all_cols, file_idx, proc_seen = [], [], [], []
    for i, p in enumerate(mats):
        PROGRESS["file"] = i + 1
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in stems:
            raise SystemExit(f"two MAT files share the name {stem}")
        cols, nexp, proc = export_mat(p, stem, flight_dir, cfg, force, limit, transpose, palette, stretch)
        stems.append(stem)
        all_cols.append(cols)
        file_idx.append(np.full(nexp, i))
        proc_seen.append(proc)

    keys = [k for k in all_cols[0].keys()]
    C = {k: np.concatenate([c[k] for c in all_cols]) for k in keys}
    C["_file"] = np.concatenate(file_idx)

    # chronological order inside the flight
    order = np.argsort(C["t"], kind="stable")
    C = {k: v[order] for k, v in C.items()}

    return finish_flight(C, stems, proc_seen, cfg, base, fid, flight_dir,
                         {"transpose": transpose, "colormap": cmap_id, "stretch": stretch, "hidden": hidden})


def finish_flight(C, stems, proc_seen, cfg, base, fid, flight_dir, meta_extra):
    """Add navigation, then write catalog.json and meta.json for one flight."""
    # navigation / met / fixed site
    progress(stage="navigation", done=0, total=0)
    nav_info = None
    nav_cfg = cfg.get("nav")
    site = cfg.get("site") or {}
    if nav_cfg and nav_cfg.get("file"):
        p = nav_cfg["file"]
        p = p if os.path.isabs(p) else os.path.join(base, p)
        fmt = nav_cfg.get("format") or ("netcdf" if p.endswith((".nc", ".nc4")) else "csv")
        nav = read_nav_netcdf(p, nav_cfg) if fmt == "netcdf" else read_nav_csv(p, nav_cfg)
        vals = interp_nav(nav, C["t"], float(nav_cfg.get("max_gap_s", 10)))
        if nav_cfg.get("temp_units", "C").upper().startswith("K") and "temp" in vals:
            vals["temp"] = vals["temp"] - 273.15
        if nav_cfg.get("alt_units", "m") == "ft" and "alt" in vals:
            vals["alt"] = vals["alt"] * 0.3048
        for k, v in vals.items():
            C[k] = v
        nav_info = {"file": os.path.basename(p), "fields": sorted(vals.keys()),
                    "matched": {k: int(np.isfinite(v).sum()) for k, v in vals.items()}}
        log(f"  navigation: " + ", ".join(f"{k} {nav_info['matched'][k]}/{C['t'].size}" for k in vals))
    for k in NAV_KEYS:
        if k in site and site[k] is not None and k not in C:
            C[k] = np.full(C["t"].size, float(site[k]))

    proc0 = next((p for p in proc_seen if p), {})
    pixel_um = cfg.get("pixel_um") or proc0.get("pixel_um")

    # catalogue -------------------------------------------------------------
    fields_meta = {}
    out_cols = {"f": C["_file"], "i": C["_idx"], "t": np.round(np.asarray(C["t"], dtype=float), 3)}
    for key, src, label, unit, dec, group in FIELDS:
        if np.isfinite(C[key]).any():
            out_cols[key] = np.round(np.asarray(C[key], dtype=float), dec)
            fields_meta[key] = {"label": label, "unit": unit, "group": group, "src": src}
    for key, (label, unit, dec, group) in EXTRA_FIELDS.items():
        if key in C and np.isfinite(C[key]).any():
            out_cols[key] = np.round(np.asarray(C[key], dtype=float), dec)
            fields_meta[key] = {"label": label, "unit": unit, "group": group}

    write_catalog(flight_dir, fid, stems, C["t"].size, fields_meta, out_cols)

    finite_t = C["t"][np.isfinite(C["t"])]
    timeline = None
    if finite_t.size:
        nb = 48
        t0, t1 = float(finite_t.min()), float(finite_t.max())
        if t1 <= t0:
            t1 = t0 + 1
        edges = np.linspace(t0, t1, nb + 1)
        bi = np.clip(np.searchsorted(edges, C["t"], side="right") - 1, 0, nb - 1)
        ok = np.isfinite(C["t"])
        timeline = {"t0": round(t0, 1), "t1": round(t1, 1),
                    "n": np.bincount(bi[ok], minlength=nb).astype(int).tolist()}
        for key in ("alt", "temp"):
            if key in C and np.isfinite(C[key]).any():
                vals = []
                for b in range(nb):
                    v = C[key][(bi == b) & np.isfinite(C[key])]
                    vals.append(round(float(np.median(v)), 1) if v.size else None)
                timeline[key] = vals
    meta = {
        "version": FORMAT_VERSION,
        "id": fid,
        "label": cfg.get("label") or fid,
        "kind": cfg.get("kind", "aircraft"),
        "campaign": cfg.get("campaign"),
        "platform": cfg.get("platform"),
        "date": cfg.get("date") or (iso(float(finite_t.min()))[:10] if finite_t.size else None),
        "instrument": cfg.get("instrument") or {"name": "CPI"},
        "pixel_um": pixel_um,
        "site": site or None,
        "notes": cfg.get("notes"),
        "files": stems,
        "n": int(C["t"].size),
        "t_start": iso(float(finite_t.min())) if finite_t.size else None,
        "t_end": iso(float(finite_t.max())) if finite_t.size else None,
        "nav": nav_info,
        "has_position": any(k in out_cols for k in NAV_KEYS),
        "processing": {"code_version": proc0.get("code_version"),
                       "processed_utc": proc0.get("processed_utc"),
                       "method": proc0.get("method"), "options": proc0},
        "timeline": timeline,
        "exported_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extra": cfg.get("extra"),
    }
    meta.update(meta_extra)
    write_json(os.path.join(flight_dir, "meta.json"), meta, compact=False)
    log(f"  wrote {meta['n']} particles to {flight_dir}")
    return meta


# ---------------------------------------------------------------------------
# Import from the CPI pipeline's own PNG export (exportParticles.py):
#   <png root>/<project>/particle_catalogue.csv
#   <png root>/<project>/<date>[_<flight>]/<project>_<mat>_roi<NNNNNNN>_p01.png
# The PNGs are used as they are (already background-corrected); statistics and
# outlines come from each MAT file's `dat`. One classifier flight is made per
# flight in the catalogue, or per date for particles with no flight.
# ---------------------------------------------------------------------------
CATALOGUE_NAME = "particle_catalogue.csv"


def png_info(path):
    """(width, height, colour type) from a PNG header, without decoding it."""
    with open(path, "rb") as fh:
        b = fh.read(26)
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    w, h = struct.unpack(">II", b[16:24])
    return w, h, b[25]


def place_file(src, dst, mode="link"):
    """Put a copy of src at dst: hard link when possible (no extra space), else copy."""
    if os.path.exists(dst):
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        os.remove(dst)
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    import shutil
    shutil.copyfile(src, dst)


def safe_key(key):
    """A flight/date key made safe for ids and URLs, e.g. 'C077+C078' -> 'C077-C078'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(key)).strip("-") or "undated"


def group_key(flight, date):
    return (flight or "").strip() or (date or "").strip() or "undated"


def read_catalogue(project_dir, name=CATALOGUE_NAME, keep_rows=True):
    """Rows of the pipeline catalogue that have a PNG, and a summary per group.
    keep_rows=False: just the summary (much less memory for a big catalogue)."""
    path = os.path.join(project_dir, name)
    if not os.path.exists(path):
        raise SystemExit(f"No {name} in {project_dir}")
    rows, groups = [], {}
    csv.field_size_limit(10 ** 8)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.reader(fh)
        head = next(rd, [])
        col = {h.strip(): i for i, h in enumerate(head)}
        need = ["png_relative_path", "flight", "date", "source_mat", "roi_index", "source_id", "project"]
        ix = {k: col.get(k) for k in need}
        if ix["png_relative_path"] is None:
            raise SystemExit(f"{name} has no png_relative_path column")
        ip, iflt, idate, imat, iroi, isid, iproj = (ix[k] for k in need)
        get = lambda r, i: r[i] if i is not None and i < len(r) else ""

        def lines():
            # plain lines split on commas (fast); only lines with quotes go through the csv parser
            for line in fh:
                if '"' in line:
                    while line.count('"') % 2:                # a quoted field running onto the next line
                        more = fh.readline()
                        if not more:
                            break
                        line += more
                    yield next(csv.reader([line]))
                else:
                    yield line.rstrip("\r\n").split(",")
        for r in lines():
            # most rows are particles that weren't exported (no PNG): skip them as cheaply as possible
            rel = r[ip].strip() if ip < len(r) else ""
            if not rel:
                continue
            flight, date = get(r, iflt), get(r, idate)
            key = group_key(flight, date)
            g = groups.get(key)
            if g is None:
                g = groups[key] = {"key": key, "flight": flight.strip(), "date": date.strip(), "n": 0, "mats": set(), "project": get(r, iproj)}
            g["n"] += 1
            mat = get(r, imat)
            g["mats"].add(mat)
            if keep_rows:
                rows.append((key, mat, int(get(r, iroi) or 0), rel, get(r, isid)))
        if not keep_rows:
            rows = range(sum(g["n"] for g in groups.values()))      # only its length is used
    return rows, groups


def catalogue_summary(project_dir):
    rows, groups = read_catalogue(project_dir, keep_rows=False)
    out = []
    for g in sorted(groups.values(), key=lambda g: (g["date"], g["key"])):
        out.append({"key": g["key"], "flight": g["flight"], "date": g["date"], "n": g["n"], "mat_files": len(g["mats"])})
    project = next((g["project"] for g in groups.values() if g.get("project")), os.path.basename(project_dir))
    return {"project": project, "groups": out, "n": len(rows)}


def import_catalogue(cfg_path, out, force=False, limit=None):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(cfg_path))
    absp = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    pdir = absp(cfg["catalogue_dir"])
    mat_dir = absp(cfg.get("mat_dir") or pdir)
    png_root = absp(cfg.get("png_root") or os.path.dirname(pdir.rstrip("/")))
    link_mode = cfg.get("png_link", "link")
    rows, groups = read_catalogue(pdir, cfg.get("catalogue_file", CATALOGUE_NAME))
    project = cfg.get("campaign") or next((g["project"] for g in groups.values() if g.get("project")), os.path.basename(pdir))
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", cfg.get("id_prefix") or project).strip("-") or "project"
    include = set(cfg.get("groups") or groups.keys())
    rows = [r for r in rows if r[0] in include]
    if not rows:
        raise SystemExit("No particles with PNGs in the chosen flights")
    log(f"Project {project}: {len(rows)} particles in {len(include & set(groups))} flight(s)/date(s) from {CATALOGUE_NAME}")

    by_mat = {}
    for key, mat, roi, rel, sid in rows:
        by_mat.setdefault(mat, []).append((key, roi - 1, rel, sid))
    acc = {}       # key -> dict of lists
    bnds = {}      # (key, stem) -> {idx: [...]}
    ctypes = {}
    proc_by_key = {}
    stems_by_key = {}
    mats = sorted(by_mat)
    PROGRESS["files"] = len(mats)
    missing_png = 0
    for mi, mat in enumerate(mats):
        PROGRESS["file"] = mi + 1
        stem = os.path.splitext(mat)[0]
        progress(stage="loading", stem=stem, done=0, total=0)
        path = os.path.join(mat_dir, mat)
        try:
            d = sio.loadmat(path, variable_names=["dat"], squeeze_me=True, struct_as_record=False)["dat"]
        except Exception as e:
            log(f"  warning: can't read {mat}: {e}")
            continue
        n = np.atleast_1d(d.len).shape[0]
        proc = {}
        if "proc_options" in d._fieldnames:
            try:
                proc = json.loads(str(d.proc_options))
            except Exception:
                proc = {}
        t = datenum_to_unix(field(d, "Time", n))
        src_cols = {key: field(d, srcf, n) for key, srcf, *_ in FIELDS}
        foc = np.atleast_1d(d.foc) if "foc" in d._fieldnames else None
        items = by_mat[mat][:limit] if limit else by_mat[mat]
        for j, (key, k, rel, sid) in enumerate(items):
            if not (0 <= k < n):
                continue
            src = os.path.join(png_root, rel)
            if not os.path.exists(src):
                alt = os.path.join(pdir, rel.split("/", 1)[-1])
                if os.path.exists(alt):
                    src = alt
                else:
                    missing_png += 1
                    continue
            fid = f"{prefix}-{safe_key(key)}"
            fdir = os.path.join(out, "flights", fid)
            img_dir = os.path.join(fdir, "img", stem)
            os.makedirs(img_dir, exist_ok=True)
            w, h, ctype = png_info(src)
            ctypes.setdefault(key, set()).add(ctype)
            place_file(src, os.path.join(img_dir, f"{k}.png"), link_mode)
            a = acc.setdefault(key, {c: [] for c in ["t", "_idx", "_stem", "w", "h", "focus", "focus_T"] + [f[0] for f in FIELDS]})
            a["t"].append(t[k]); a["_idx"].append(k); a["_stem"].append(stem); a["w"].append(w); a["h"].append(h)
            for c, v in src_cols.items():
                a[c].append(v[k])
            fv = fT = np.nan
            if foc is not None and hasattr(foc[k], "_fieldnames"):
                f = foc[k]
                if "focus" in f._fieldnames:
                    fv = float(np.asarray(f.focus, float))
                if "focus_T" in f._fieldnames:
                    fT = float(np.asarray(f.focus_T, float))
                if "boundaries" in f._fieldnames:
                    b = np.asarray(f.boundaries, float)
                    if b.ndim == 2 and b.shape[0] > 2:
                        # (row, col) pixel centres; the pipeline PNGs are not transposed
                        xy = np.round(np.column_stack([b[:, 1] + 0.5, b[:, 0] + 0.5]).ravel(), 1)
                        bnds.setdefault((key, stem), {})[str(int(k))] = [float(v) for v in xy]
            a["focus"].append(fv); a["focus_T"].append(fT)
            proc_by_key.setdefault(key, proc)
            stems_by_key.setdefault(key, [])
            if stem not in stems_by_key[key]:
                stems_by_key[key].append(stem)
            if (j + 1) % 200 == 0 or j + 1 == len(items):
                progress(stage="images", stem=stem, done=j + 1, total=len(items))
        log(f"  {mat}: {len(items)} particles")
    if missing_png:
        log(f"  warning: {missing_png} PNG files listed in the catalogue were not found")

    progress(stage="navigation", done=0, total=0)
    metas = []
    nav_by_group = cfg.get("nav_by_group") or {}
    for key, a in acc.items():
        fid = f"{prefix}-{safe_key(key)}"
        fdir = os.path.join(out, "flights", fid)
        stems = stems_by_key[key]
        C = {c: np.asarray(v, dtype=float) for c, v in a.items() if c != "_stem"}
        C["_idx"] = np.asarray(a["_idx"], dtype=int)
        C["_file"] = np.asarray([stems.index(sname) for sname in a["_stem"]], dtype=int)
        with np.errstate(divide="ignore", invalid="ignore"):
            C["aspect"] = C["len"] / C["wid"]
        order = np.argsort(C["t"], kind="stable")
        C = {c: v[order] for c, v in C.items()}
        os.makedirs(os.path.join(fdir, "bnd"), exist_ok=True)
        for sname in stems:
            write_json(os.path.join(fdir, "bnd", f"{sname}.json"), bnds.get((key, sname), {}))
        hidden = False
        mp = os.path.join(fdir, "meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as fh:
                hidden = bool(json.load(fh).get("hidden"))
        g = groups[key]
        gcfg = dict(cfg)
        gcfg["label"] = g["flight"] or (f"{g['date']} (no flight)" if g["date"] else key)
        gcfg["campaign"] = project
        gcfg["date"] = g["date"] or None
        gcfg["nav"] = nav_by_group.get(key) or None
        if not g["flight"] and cfg.get("undated_kind"):
            gcfg["kind"] = cfg["undated_kind"]
        ct = ctypes.get(key, {0})
        cmap_id = "gray" if ct <= {0, 4} else (cfg.get("png_colormap") or "blues")
        metas.append(finish_flight(C, stems, [proc_by_key.get(key, {})], gcfg, base, fid, fdir, {
            "transpose": False, "colormap": cmap_id, "stretch": None, "hidden": hidden,
            "config_id": cfg.get("id"), "flight_label": g["flight"] or None,
            "source": {"type": "cpi-pipeline-png", "project": project, "catalogue": os.path.join(pdir, CATALOGUE_NAME),
                       "source_id_format": f"{project}_<mat>_roi<index+1:07d>_p01"}}))
    return metas


def build_manifest(out):
    flights = []
    for mp in sorted(glob.glob(os.path.join(out, "flights", "*", "meta.json"))):
        with open(mp, encoding="utf-8") as fh:
            m = json.load(fh)
        flights.append({k: m.get(k) for k in (
            "id", "label", "kind", "campaign", "platform", "date", "instrument",
            "pixel_um", "n", "t_start", "t_end", "has_position", "files", "colormap", "timeline", "hidden", "source", "config_id")})
        flights[-1]["code_version"] = (m.get("processing") or {}).get("code_version")
    flights.sort(key=lambda f: (f.get("date") or "", f["id"]))
    manifest = {"version": FORMAT_VERSION,
                "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "flights": flights}
    write_json(os.path.join(out, "manifest.json"), manifest, compact=False)
    log(f"manifest: {len(flights)} flight(s), {sum(f['n'] for f in flights)} particles")


def colormap_preview(cfg_path, html_path, maps=None, n=10):
    """Write one HTML page showing a few particles from the flight in every colour map."""
    import base64
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(cfg_path))
    mats = resolve_mat_files(cfg, base)
    if not mats:
        raise SystemExit("no MAT files found")
    R, d = load_mat(mats[0])
    nroi = np.atleast_1d(R.Time).shape[0]
    L = field(d, "len", nroi)
    imgs = np.atleast_1d(R.IMAGE)
    small = np.array([max(np.shape(imgs[k].IM)) <= 180 if np.ndim(imgs[k].IM) == 2 else False for k in range(nroi)])
    ok = np.where(np.isfinite(L) & small)[0]
    if not ok.size:
        ok = np.where(np.isfinite(L))[0]
    if not ok.size:
        raise SystemExit("no measured particles in " + mats[0])
    # a spread of sizes: small to large
    pick = ok[np.argsort(L[ok])][np.linspace(0, ok.size - 1, n).astype(int)]
    transpose = bool(cfg.get("transpose", True))
    built = builtin_colormaps()
    ids = maps or list(built.keys())
    rows = []
    for cid in ids:
        _, table = resolve_colormap(cid, base)
        cells = []
        for k in pick:
            im = np.asarray(np.atleast_1d(R.IMAGE)[k].IM, dtype=np.uint8)
            im = im.T if transpose else im
            im = apply_stretch(im, cfg.get("stretch"))
            rgb = np.repeat(im[:, :, None], 3, 2) if table is None else table[im]
            b64 = base64.b64encode(png_rgb_bytes(rgb)).decode()
            cells.append(f'<img src="data:image/png;base64,{b64}" width="{im.shape[1] * 1.5:.0f}" height="{im.shape[0] * 1.5:.0f}">')
        info = built.get(cid, {"name": cid, "desc": ""})
        bar_tab = table if table is not None else np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1)
        stops = ",".join(f"rgb({r},{g},{b})" for r, g, b in bar_tab[::16].tolist() + [bar_tab[-1].tolist()])
        rows.append(f'<tr><th><b>{info["name"]}</b><code>{cid}</code><i style="background:linear-gradient(90deg,{stops})"></i>'
                    f'<small>{info["desc"]}</small></th><td>{"".join(cells)}</td></tr>')
    html = f"""<!doctype html><meta charset="utf-8"><title>Colour maps - {cfg.get('id')}</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;margin:24px;background:#e2e7ec;color:#16212c}}
table{{border-collapse:collapse}}th{{text-align:left;vertical-align:top;width:230px;padding:10px 14px 10px 0;font-weight:400}}
th b,th code,th small{{display:block}}code{{color:#4a5866;font-size:.85em}}small{{color:#4a5866}}
th i{{display:block;height:10px;border-radius:3px;margin:5px 0}}td{{padding:10px 0;border-top:1px solid #c9d2db;white-space:nowrap}}
td img{{margin:0 6px 0 0;vertical-align:middle;image-rendering:pixelated;border-radius:3px}}</style>
<h1>Colour maps for {cfg.get('label') or cfg.get('id')}</h1>
<p>{n} particles from {os.path.basename(mats[0])}, smallest to largest, all at the same scale (1.5 screen pixels per image pixel). Use the id with <code>--colormap</code>
or <code>"colormap"</code> in flight.json - or keep grey and choose a map live on the web page.</p>
<table>{''.join(rows)}</table>"""
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"wrote {html_path} ({len(ids)} colour maps)")


def nav_columns(path):
    """Describe the columns / variables of a navigation file (used by the admin page)."""
    out = {"file": os.path.basename(path)}
    if path.endswith((".nc", ".nc4", ".cdf")):
        try:
            import netCDF4
        except ImportError:
            return {**out, "format": "netcdf", "error": "netCDF4 is not installed on the server (pip install netCDF4)"}
        ds = netCDF4.Dataset(path)
        out["format"] = "netcdf"
        out["columns"] = [{"name": k, "units": getattr(v, "units", ""), "long_name": getattr(v, "long_name", ""),
                           "dims": list(v.dimensions)} for k, v in ds.variables.items()]
        ds.close()
        return out
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.reader(fh)
        head = next(rd, [])
        rows = [r for _, r in zip(range(3), rd)]
    out["format"] = "csv"
    out["columns"] = [{"name": c, "example": (rows[0][i] if rows and i < len(rows[0]) else "")} for i, c in enumerate(head)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flights", nargs="*", help="flight JSON file(s)")
    ap.add_argument("--out", help="web data directory, e.g. ../web/data")
    ap.add_argument("--force", action="store_true", help="rewrite images that already exist")
    ap.add_argument("--limit", type=int, default=None, help="export at most N particles per MAT file (testing)")
    ap.add_argument("--manifest-only", action="store_true", help="just rebuild manifest.json")
    ap.add_argument("--colormap", help="bake this colour map into the PNGs (overrides flight.json; needs --force to redo existing images)")
    ap.add_argument("--colormap-preview", metavar="HTML", help="write a page comparing colour maps on this flight's particles, then stop")
    ap.add_argument("--list-colormaps", action="store_true", help="list the built-in colour maps")
    ap.add_argument("--progress", action="store_true", help="print machine-readable @progress lines (for the admin page)")
    ap.add_argument("--nav-columns", metavar="FILE", help="print the columns of a navigation file as JSON, then stop")
    ap.add_argument("--convert-catalogs", metavar="DATA_DIR", help="rewrite already-exported catalogues in the fast binary form, then stop")
    ap.add_argument("--catalogue-summary", metavar="DIR", help="print the flights in a pipeline particle_catalogue.csv as JSON, then stop")
    a = ap.parse_args(argv)
    PROGRESS["on"] = a.progress
    if a.nav_columns:
        print(json.dumps(nav_columns(a.nav_columns)))
        return
    if a.catalogue_summary:
        print(json.dumps(catalogue_summary(a.catalogue_summary)))
        return
    if a.convert_catalogs:
        convert_catalogs(a.convert_catalogs)
        return
    if a.list_colormaps:
        for k, v in builtin_colormaps().items():
            print(f"{k:11s} {v['name']:11s} {v['desc']}")
        print("Also: any matplotlib colour map name, 'file.mat:VAR', a .json/.csv table; add _r to reverse.")
        return
    if a.colormap_preview:
        if not a.flights:
            ap.error("give a flight JSON to preview")
        colormap_preview(a.flights[0], a.colormap_preview)
        return
    if not a.out:
        ap.error("--out is required")
    os.makedirs(a.out, exist_ok=True)
    if not a.manifest_only:
        if not a.flights:
            ap.error("give at least one flight JSON (or --manifest-only)")
        for f in a.flights:
            with open(f, encoding="utf-8") as fh:
                is_cat = "catalogue_dir" in json.load(fh)
            if is_cat:
                import_catalogue(f, a.out, a.force, a.limit)
            else:
                export_flight(f, a.out, a.force, a.limit, a.colormap)
    build_manifest(a.out)


if __name__ == "__main__":
    main()
