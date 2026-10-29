#!/usr/bin/env python3
"""
check_images.py - find out why particles or images seem to be missing on the site.

    python3 check_images.py /var/www/cpi/data                 every flight
    python3 check_images.py /var/www/cpi/data PICASSO-C079    flights whose id contains this

For each imported flight it compares three numbers:
  catalogue   particles the pipeline exported a PNG for (particle_catalogue.csv)
  imported    particles the site has for the flight (its catalogue)
  images      of those, how many image files exist and the web server can read

and lists what's behind any gap: PNGs the importer couldn't find, image files that
are missing, or files only their owner can read (the web server then gets
"403 Forbidden" and the tile stays blank). Nothing is changed.

Particles the pipeline left out on purpose (smaller than --min-size-um, less in
focus than --min-focus) are not in the catalogue at all: the first line of the
report shows how many rows the catalogue has without a PNG.
"""
import collections
import csv
import glob
import json
import os
import stat
import sys


def readable_by_others(path):
    try:
        return bool(os.stat(path).st_mode & stat.S_IROTH)
    except OSError:
        return False


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    metas = sorted(glob.glob(os.path.join(data, "flights", "*", "meta.json")))
    if not metas:
        sys.exit(f"No flights in {data}/flights")

    cat_counts = {}           # catalogue path -> (Counter flight-key -> rows with png, rows without png)
    def catalogue(path):
        if path not in cat_counts:
            with_png, without = collections.Counter(), 0
            try:
                with open(path, newline="", encoding="utf-8-sig") as fh:
                    for r in csv.DictReader(fh):
                        if (r.get("png_relative_path") or "").strip():
                            key = r.get("flight") or r.get("date") or ""
                            with_png[key] += 1
                        else:
                            without += 1
            except OSError as e:
                print(f"  can't read {path}: {e}")
            cat_counts[path] = (with_png, without)
        return cat_counts[path]

    print(f"{'Flight':34} {'catalogue':>10} {'imported':>9} {'images':>8} {'no file':>8} {'private':>8}")
    problems = []
    for mf in metas:
        fdir = os.path.dirname(mf)
        fid = os.path.basename(fdir)
        if want and want not in fid:
            continue
        m = json.load(open(mf, encoding="utf-8"))
        src = m.get("source") or {}
        n_cat = "-"
        if src.get("catalogue") and os.path.exists(src["catalogue"]):
            with_png, without = catalogue(src["catalogue"])
            # the flight's key in the catalogue: its flight label, else its date
            n_cat = with_png.get(m.get("flight_label") or "", 0) or with_png.get(m.get("date") or "", 0) or "?"
        c = json.load(open(os.path.join(fdir, "catalog.json"), encoding="utf-8"))
        n = int(c["n"])
        files = c["files"]
        if "bin" in c:
            import numpy as np
            buf = open(os.path.join(fdir, c["bin"]["file"]), "rb").read()
            col = lambda k: np.frombuffer(buf, dtype="<u4", count=n, offset=c["bin"]["cols"][k][1])
            fcol, icol = col("f"), col("i")
        else:
            fcol, icol = c["cols"]["f"], c["cols"]["i"]
        have = nofile = private = 0
        examples = []
        for f, i in zip(fcol, icol):
            p = os.path.join(fdir, "img", files[int(f)], f"{int(i)}.png")
            if not os.path.exists(p):
                nofile += 1
                if len(examples) < 2:
                    examples.append(f"missing: {p}")
            elif not readable_by_others(p):
                private += 1
                if len(examples) < 2:
                    examples.append(f"only its owner can read it: {p}")
            else:
                have += 1
        # folders the web server has to pass through
        for d in [fdir, os.path.join(fdir, "img")] + [os.path.join(fdir, "img", s) for s in files[:3]]:
            if os.path.isdir(d) and not (os.stat(d).st_mode & stat.S_IXOTH):
                examples.append(f"the web server can't enter {d}")
        flag = "" if nofile == 0 and private == 0 and (n_cat in ("-", "?") or n_cat == n) else "   <--"
        print(f"{fid[:34]:34} {str(n_cat):>10} {n:9,} {have:8,} {nofile:8,} {private:8,}{flag}")
        if flag:
            problems.append((fid, n_cat, n, examples))

    for path, (with_png, without) in cat_counts.items():
        print(f"\n{path}: {sum(with_png.values()):,} particles with a PNG; {without:,} rows without one "
              f"(left out by the size/focus limits, or not exported)")
    if not problems:
        print("\nEvery imported particle has a readable image.")
        return
    print()
    for fid, n_cat, n, ex in problems:
        print(f"{fid}:")
        if isinstance(n_cat, int) and n_cat != n:
            print(f"  the catalogue lists {n_cat:,} PNGs but {n:,} were imported: look for "
                  f"'PNG files listed in the catalogue were not found' in the import log, or re-import the flight")
        for e in ex:
            print("  " + e)
    print("\nPrivate files: make them readable with  importable.sh --all <the export folder>  (the images in the site"
          " are hard links to the exported PNGs, so they share their permissions), then reload the page.")


if __name__ == "__main__":
    main()
