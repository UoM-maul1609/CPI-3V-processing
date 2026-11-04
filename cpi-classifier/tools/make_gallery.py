#!/usr/bin/env python3
"""
make_gallery.py - a publication-quality gallery of classified CPI particles, one row per
class, for proposals, papers and talks.

It reads the classifications exported from the web site (Progress page -> "Download CSV";
administrators get everyone's), picks good examples of each class (the ones most people
agree on, in focus, not cut off at the edge), and lays them out with each class's name,
how many particles were classified as it, and a scale bar.

    python3 tools/make_gallery.py cpi-labels-2026-11-02.csv --data /var/www/cpi/data
    python3 tools/make_gallery.py labels.csv --data /var/www/cpi/data --per-class 10 \\
        --classes column,plate,dendrite,rosette,aggregate,graupel --out habits.pdf

Useful options (all have sensible defaults):
  --per-class N        particles per class (default 8)
  --classes a,b,c      which classes, in this order (default: every class with examples,
                       in the site's order, leaving out small / unusable / artefact)
  --scale row|same|fit row: each row at one scale (largest in the row fills its tile), with
                       a scale bar per row (default); same: one scale for the whole figure,
                       so sizes compare across rows; fit: every particle fills its tile
  --min-people N --min-agree F   only particles at least N people classified, with at
                       least a fraction F agreeing (default 1 and 0.6)
  --project P / --flights a,b    only particles from these flights or project(s)
  --colours original|grey|<matplotlib colormap, e.g. Blues_r, viridis>
  --width-mm W         figure width (default 180 mm: an A4 page's text width)
  --seed N             a different random choice of examples
  --made-of            separate rows for e.g. "Aggregate of plates", "Aggregate of columns"
  --dark               light particles on a dark background (for slides)
Writes PDF (vector text, embeds the images at full resolution) or PNG (--dpi, default 300),
by the extension of --out; default gallery.pdf and gallery.png.

Needs numpy, pillow and matplotlib.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

LEAVE_OUT = {"small", "unusable", "artefact"}


# ---------------------------------------------------------------- the site's data
def load_scheme(data_dir, path=None):
    """Class names, colours and order: the site's own (Admin -> Classes) if saved, else the default."""
    here = os.path.dirname(os.path.abspath(__file__))
    tries = [path] if path else []
    tries += [os.path.join(os.path.dirname(os.path.abspath(data_dir)), "work", "scheme.json"),
              "/var/www/cpi/work/scheme.json",
              os.path.join(here, "..", "web", "config", "scheme.json")]
    for p in tries:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return {"classes": []}


Q = {"u8": "<u1", "u16": "<u2", "u32": "<u4", "f32": "<f4", "f64": "<f8", "q16": "<u2"}


def _dequantise(q, p):
    out = np.full(q.shape, np.nan, dtype="f4")
    ok = q != 65535
    c = q[ok].astype(float)
    span = (p["hi"] - p["lo"]) or 1.0
    out[ok] = (np.where(c > 0, np.exp(p["lo"] + (c - 1) / 65533 * span), 0.0) if p["m"] == "log"
               else p["lo"] + c / 65534 * span)
    return out


def load_flight(data_dir, fid, want=("f", "i", "focus", "edge_touch", "len")):
    """A flight's pixel size and the few catalogue columns used to pick good examples."""
    fdir = os.path.join(data_dir, "flights", fid)
    with open(os.path.join(fdir, "catalog.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    meta = {}
    try:
        with open(os.path.join(fdir, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except OSError:
        pass
    n = int(c["n"])
    cols = {}
    if "colfiles" in c:
        for k in want:
            spec = c["colfiles"]["cols"].get(k)
            if spec:
                raw = np.fromfile(os.path.join(fdir, c["colfiles"]["dir"], k + ".bin"), dtype=Q[spec[0]], count=n)
                cols[k] = _dequantise(raw, spec[2]) if spec[0] == "q16" else raw
    elif "bin" in c:
        buf = open(os.path.join(fdir, c["bin"]["file"]), "rb").read()
        for k in want:
            if k in c["bin"]["cols"]:
                t, off = c["bin"]["cols"][k]
                cols[k] = np.frombuffer(buf, dtype=Q[t], count=n, offset=off)
    else:
        for k in want:
            if k in c["cols"]:
                cols[k] = np.array([np.nan if v is None else v for v in c["cols"][k]], dtype=float)
    stems = c["files"]
    row = {(stems[int(f)], int(i)): r for r, (f, i) in enumerate(zip(cols["f"], cols["i"]))}
    return {"dir": fdir, "pixel_um": float(meta.get("pixel_um") or 2.3), "cols": cols, "row": row,
            "campaign": meta.get("campaign") or "", "label": meta.get("label") or fid}


def read_labels(path):
    """pid -> list of (label, made_of, flags) per person."""
    out = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pid = (r.get("pid") or "").strip()
            lab = (r.get("label") or "").strip()
            if pid and lab and lab != "(removed)":
                out[pid].append((lab, (r.get("made_of") or "").strip(), (r.get("flags") or "").strip()))
    return out


def consensus(votes, min_people, min_agree, by_made_of):
    """(class key, agreement) if enough people agree, else None."""
    if len(votes) < min_people:
        return None
    keys = [(lab, mo if by_made_of else "") for lab, mo, _ in votes]
    (key, n), = Counter(keys).most_common(1)
    agree = n / len(keys)
    return (key, agree) if agree >= min_agree else None


# ---------------------------------------------------------------- the pictures
def load_particle(path, colours):
    """The particle cropped to itself (with a small margin), as an RGB float array 0-1, plus
    the background colour. Works for grey and colour-mapped PNGs alike."""
    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bg = float(np.percentile(lum, 95))
    dark = lum < bg - 0.12 * max(bg, 0.2)
    ys, xs = np.nonzero(dark)
    if len(ys):
        m = 3
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, lum.shape[0])
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, lum.shape[1])
        rgb, lum = rgb[y0:y1, x0:x1], lum[y0:y1, x0:x1]
    # how much of each pixel is particle (0 = background), with a soft edge, so the particle can
    # be laid on a clean background without the rectangle of its original picture showing
    alpha = np.clip((bg - lum) / max(0.10 * bg, 0.02), 0, 1).astype(np.float32)
    if colours == "original":
        bgc = np.median(rgb.reshape(-1, 3)[lum.reshape(-1) >= np.percentile(lum, 80)], axis=0)
        return np.dstack([rgb, alpha]), bgc
    # grey or a colour map: stretch so the background is white and the darkest part black
    lo = float(np.percentile(lum, 0.5))
    g = np.clip((lum - lo) / max(bg - lo, 1e-3), 0, 1)
    if colours == "grey":
        out, bgc = np.repeat(g[..., None], 3, axis=2), np.full(3, 0.965, dtype=np.float32)   # a faint panel
    else:
        import matplotlib
        cm = matplotlib.colormaps[colours]
        out, bgc = cm(g)[..., :3].astype(np.float32), np.asarray(cm(1.0)[:3], dtype=np.float32)
    return np.dstack([out, alpha]), bgc


def place(rgb, bgc, tile, scale, room=1.0):
    """The particle resized by scale (tile pixels per image pixel) and centred on a square tile,
    in its top `room` fraction (the rest is kept clear for a scale bar)."""
    h, w = rgb.shape[:2]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    lim = tile * room
    if nw > lim or nh > lim:                                    # never bigger than its space
        k = lim / max(nw, nh)
        nw, nh = max(1, round(nw * k)), max(1, round(nh * k))
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGBA").resize(
        (nw, nh), Image.LANCZOS if scale < 1 else Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    out = np.ones((tile, tile, 3), dtype=np.float32) * bgc.reshape(1, 1, 3)
    y, x = int((tile * room - nh) // 2), (tile - nw) // 2
    al = a[..., 3:4]
    out[y:y + nh, x:x + nw] = a[..., :3] * al + out[y:y + nh, x:x + nw] * (1 - al)
    return out, nw / w if w else 1.0


def wrap_name(name, width=16):
    """Class names on up to three lines, broken at spaces (after a "/" where there is one)."""
    words, lines, cur = name.replace(" / ", " /\u00a0").split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines[:3]).replace("\u00a0", " ")


def nice_bar(um_per_px, tile, frac=0.4):
    """A round scale-bar length (um) no longer than frac of the tile."""
    target = tile * frac * um_per_px
    best = 1
    for L in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if L <= target:
            best = L
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="classifications CSV downloaded from the site (Progress -> Download CSV)")
    ap.add_argument("--data", default="/var/www/cpi/data", help="the site's data folder (default /var/www/cpi/data)")
    ap.add_argument("--scheme", help="scheme.json for class names and colours (default: the site's)")
    ap.add_argument("--out", default="gallery.pdf", help="output file: .pdf, .png, .svg or .jpg (default gallery.pdf, plus a .png)")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--classes", help="comma-separated class ids, in the order to show them")
    ap.add_argument("--made-of", action="store_true", help="separate rows for what aggregates etc. are made of")
    ap.add_argument("--scale", choices=["row", "same", "fit"], default="row")
    ap.add_argument("--min-people", type=int, default=1)
    ap.add_argument("--min-agree", type=float, default=0.6)
    ap.add_argument("--project", help="only these projects (comma-separated, as shown on the site)")
    ap.add_argument("--flights", help="only these flight ids (comma-separated)")
    ap.add_argument("--colours", "--colors", default="original", help="original, grey, or a matplotlib colormap name")
    ap.add_argument("--width-mm", type=float, default=180)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--title", default="")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--no-counts", action="store_true", help="leave out the number classified beside each class name")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scheme = load_scheme(a.data, a.scheme)
    cls = {c["id"]: c for c in scheme.get("classes", [])}
    order = [c["id"] for c in scheme.get("classes", [])]

    votes = read_labels(a.labels)
    if not votes:
        sys.exit(f"No classifications in {a.labels}")
    want_flights = set(a.flights.split(",")) if a.flights else None
    want_projects = {p.strip().lower() for p in a.project.split(",")} if a.project else None

    # every agreed particle, by class
    flights, by_class, counts = {}, defaultdict(list), Counter()
    for pid, v in votes.items():
        try:
            fid, stem, idx = pid.rsplit(":", 2)
        except ValueError:
            continue
        if want_flights and fid not in want_flights:
            continue
        r = consensus(v, a.min_people, a.min_agree, a.made_of)
        if not r:
            continue
        (lab, mo), agree = r
        if fid not in flights:
            try:
                flights[fid] = load_flight(a.data, fid)
            except (OSError, KeyError, ValueError) as e:
                print(f"  {fid}: can't read it ({e}); skipped", file=sys.stderr)
                flights[fid] = None
        F = flights[fid]
        if not F:
            continue
        if want_projects and F["campaign"].lower() not in want_projects:
            continue
        path = os.path.join(F["dir"], "img", stem, f"{idx}.png")
        if not os.path.exists(path):
            continue
        row = F["row"].get((stem, int(idx)))
        c = F["cols"]
        focus = float(c["focus"][row]) if row is not None and "focus" in c else float("nan")
        edge = float(c["edge_touch"][row]) if row is not None and "edge_touch" in c else 0.0
        counts[(lab, mo)] += 1
        by_class[(lab, mo)].append({"pid": pid, "path": path, "agree": agree, "people": len(v),
                                    "focus": focus, "edge": edge, "px": F["pixel_um"]})

    # which rows, in what order
    if a.classes:
        ids = [x.strip() for x in a.classes.split(",") if x.strip()]
        keys = [k for i in ids for k in sorted(by_class, key=lambda k: (k[1] != "", k[1])) if k[0] == i]
    else:
        rank = {c: i for i, c in enumerate(order)}
        keys = sorted((k for k in by_class if k[0] not in LEAVE_OUT),
                      key=lambda k: (rank.get(k[0], 999), k[1] != "", k[1]))
    keys = [k for k in keys if by_class[k]]
    if not keys:
        sys.exit("No classified particles with images match those settings")

    # pick the examples: not cut off, most agreed, then the sharpest, then at random
    rnd = random.Random(a.seed)
    picks = {}
    for k in keys:
        cands = by_class[k][:]
        rnd.shuffle(cands)
        whole = [p for p in cands if not p["edge"]] or cands
        fin = [p["focus"] for p in whole if math.isfinite(p["focus"])]
        sharp_cut = float(np.percentile(fin, 40)) if len(fin) >= 3 * a.per_class else -math.inf
        whole.sort(key=lambda p: (-(p["agree"] >= 0.99 and p["people"] > 1),
                                  -(p["focus"] >= sharp_cut if math.isfinite(p["focus"]) else 0)))
        picks[k] = whole[:a.per_class]

    # pictures, and the scale for each row
    imgs = {k: [(p, *load_particle(p["path"], a.colours)) for p in picks[k]] for k in keys}
    tile_px = 220                                                    # resolution of each tile
    room = 0.84                                                      # top part for the particle; below: the scale bar
    extent = lambda p, rgb: max(rgb.shape[:2]) * p["px"]             # um across, after cropping
    if a.scale == "same":
        um_tile = max(extent(p, rgb) for k in keys for p, rgb, _ in imgs[k]) * 1.05
    ncol = a.per_class
    nrow = len(keys)

    # figure geometry (mm): a label column on the left, then the tiles
    W = a.width_mm
    label_w = 30.0
    gap = 1.2
    tile_mm = (W - label_w - gap * (ncol - 1)) / ncol
    head = 8.0 if a.title else 2.0
    H = head + nrow * tile_mm + (nrow - 1) * gap + 2.0
    fg, bgfig = ("#e8e8e8", "#111418") if a.dark else ("#1a1a1a", "white")
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor=bgfig)
    mm = lambda x, y, w, h: [x / W, 1 - (y + h) / H, w / W, h / H]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    if a.title:
        fig.text(0.5, 1 - 4.5 / H, a.title, ha="center", va="center", fontsize=10, fontweight="bold", color=fg)

    for r, k in enumerate(keys):
        lab, mo = k
        y = head + r * (tile_mm + gap)
        c = cls.get(lab, {})
        name = c.get("name", lab)
        if mo:
            of = cls.get(mo, {}).get("plural") or cls.get(mo, {}).get("name") or mo
            name = f"{name} of {of[0].lower() + of[1:] if of else of}"
        colour = c.get("color", "#888888")
        # class name, a colour mark, and how many were classified as it
        lax = fig.add_axes(mm(0, y, label_w - 2, tile_mm))
        lax.axis("off")
        lax.add_patch(Rectangle((0, 0.12), 0.035, 0.76, color=colour, transform=lax.transAxes, clip_on=False))
        lax.text(0.09, 0.47 if not a.no_counts else 0.5, wrap_name(name), transform=lax.transAxes,
                 va="bottom" if not a.no_counts else "center", ha="left", fontsize=7.5, fontweight="bold",
                 color=fg, linespacing=1.1)
        if not a.no_counts:
            lax.text(0.09, 0.40, f"n = {counts[k]:,}", transform=lax.transAxes, va="top", ha="left",
                     fontsize=6, color="#8a8a8a" if not a.dark else "#9aa0a6")
        row = imgs[k]
        if a.scale == "row":
            um_tile = max(extent(p, rgb) for p, rgb, _ in row) * 1.05
        for j in range(ncol):
            ax = fig.add_axes(mm(label_w + j * (tile_mm + gap), y, tile_mm, tile_mm))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j >= len(row):
                ax.axis("off")
                continue
            p, rgb, bgc = row[j]
            if a.dark and a.colours in ("grey", "original"):
                rgb = np.dstack([1 - rgb[..., :3], rgb[..., 3]])
                bgc = 1 - bgc
            if a.scale == "fit":
                scale = tile_px * room * 0.95 / max(rgb.shape[:2])
            else:
                scale = tile_px * room / (um_tile / p["px"])         # tile px per image px
            tile, s_eff = place(rgb, bgc, tile_px, scale, room)
            ax.imshow(tile, interpolation="lanczos")
            ax.set_facecolor(bgc)
            um_per_tile_px = p["px"] / s_eff
            # scale bar: on every tile for "fit", on the last tile of each row otherwise
            if a.scale == "fit" or j == len(row) - 1:
                L = nice_bar(um_per_tile_px, tile_px)
                Lpx = L / um_per_tile_px
                x1 = tile_px * 0.95
                yb = tile_px * 0.95
                bar = "#f0f0f0" if a.dark else "#111111"
                ax.plot([x1 - Lpx, x1], [yb, yb], color=bar, lw=1.4, solid_capstyle="butt")
                ax.text(x1 - Lpx - tile_px * 0.03, yb, f"{L} µm", ha="right", va="center",
                        fontsize=5.5, color=bar)
            ax.set_xlim(0, tile_px); ax.set_ylim(tile_px, 0)

    outs = [a.out]
    if a.out == "gallery.pdf":
        outs.append("gallery.png")
    for o in outs:
        fig.savefig(o, dpi=a.dpi, facecolor=bgfig)
        print(f"Wrote {o}")
    print(f"{nrow} classes, up to {ncol} particles each, from {len(votes):,} classified particles")
    for k in keys:
        print(f"  {k[0]}{' of ' + k[1] if k[1] else ''}: {len(picks[k])} shown of {counts[k]:,} agreed")


if __name__ == "__main__":
    main()
