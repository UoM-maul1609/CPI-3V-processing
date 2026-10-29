#!/usr/bin/env python3
"""
check_reimport.py - make sure reprocessing and re-importing flights leaves every
classification on the particle it was made for.

Classifications are stored against a particle id, "<flight>:<MAT file>:<ROI number>".
They stay right as long as each MAT file keeps the same ROIs in the same order
(reprocessing the same .roi files does), and the flight keeps the same id. This
script checks both, using each particle's time and size as its fingerprint.

1. Before re-importing, take a snapshot of what the site has now:
       python3 check_reimport.py snapshot /var/www/cpi/data  ~/cpi-before.npz

2. Reprocess, re-export and re-import as usual.

3. Compare:
       python3 check_reimport.py compare  /var/www/cpi/data  ~/cpi-before.npz  [--labels labels.csv]

   --labels: the "Everyone (CSV)" download from the Progress page, to report on
   the classified particles in particular.

It reports, per flight, how many particles still match (same MAT file and ROI
number, same time to 0.01 s, size within 15%), any that moved to a differently
named flight, and any that changed. If particles moved flight (e.g. a campaign
first imported by date and now by flight number), it writes move-labels.sql,
which moves their classifications to the new ids. Check it, then run it with
    mysql -u cpi_web -p cpi < move-labels.sql
Nothing in the database is changed by this script itself.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def load_catalog(flight_dir):
    with open(os.path.join(flight_dir, "catalog.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    n = int(c["n"])
    if "bin" in c:
        buf = open(os.path.join(flight_dir, c["bin"]["file"]), "rb").read()
        T = {"u32": "<u4", "f64": "<f8", "f32": "<f4"}
        col = lambda k: np.frombuffer(buf, dtype=T[c["bin"]["cols"][k][0]], count=n, offset=c["bin"]["cols"][k][1]) if k in c["bin"]["cols"] else None
    else:
        col = lambda k: np.array([np.nan if v is None else v for v in c["cols"][k]], dtype=float) if k in c["cols"] else None
    f, i, t, ln = col("f"), col("i"), col("t"), col("len")
    if ln is None:
        ln = np.full(n, np.nan)
    stems = np.array(c["files"], dtype=object)
    return c["flight"], stems[np.asarray(f, dtype=int)], np.asarray(i, dtype=int), np.asarray(t, dtype=float), np.asarray(ln, dtype=float)


def read_all(data_dir):
    F, S, I, T, L = [], [], [], [], []
    for cj in sorted(glob.glob(os.path.join(data_dir, "flights", "*", "catalog.json"))):
        try:
            fid, s, i, t, ln = load_catalog(os.path.dirname(cj))
        except Exception as e:                      # noqa: BLE001
            print(f"  couldn't read {cj}: {e}")
            continue
        F.append(np.full(len(i), fid, dtype=object)); S.append(s); I.append(i); T.append(t); L.append(ln)
    if not F:
        sys.exit(f"No catalogues found in {data_dir}/flights")
    return np.concatenate(F), np.concatenate(S), np.concatenate(I), np.concatenate(T), np.concatenate(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["snapshot", "compare"])
    ap.add_argument("data_dir")
    ap.add_argument("snapshot_file")
    ap.add_argument("--labels", help="the Everyone (CSV) download, to report on classified particles")
    ap.add_argument("--sql", default="move-labels.sql")
    a = ap.parse_args()

    if a.mode == "snapshot":
        F, S, I, T, L = read_all(a.data_dir)
        np.savez_compressed(a.snapshot_file, flight=F.astype(str), stem=S.astype(str), idx=I, t=T, len=L)
        print(f"Saved {len(I):,} particles from {len(set(F))} flights to {a.snapshot_file}")
        return

    old = np.load(a.snapshot_file, allow_pickle=False)
    oF, oS, oI, oT, oL = old["flight"], old["stem"], old["idx"], old["t"], old["len"]
    nF, nS, nI, nT, nL = read_all(a.data_dir)
    new = {(s, int(i)): (f, t, ln) for f, s, i, t, ln in zip(nF, nS, nI, nT, nL)}

    labelled = None
    if a.labels:
        with open(a.labels, newline="", encoding="utf-8-sig") as fh:
            labelled = {r["pid"] for r in csv.DictReader(fh) if r.get("pid")}
        print(f"{len(labelled):,} classified particles in {a.labels}")

    per = {}          # old flight -> counters
    moves = {}        # (old flight, new flight) -> count
    bad_examples = []
    for f, s, i, t, ln in zip(oF, oS, oI, oT, oL):
        pid = f"{f}:{s}:{int(i)}"
        if labelled is not None and pid not in labelled:
            continue
        c = per.setdefault(f, {"n": 0, "same": 0, "moved": 0, "changed": 0, "gone": 0})
        c["n"] += 1
        hit = new.get((s, int(i)))
        if hit is None:
            c["gone"] += 1
            if len(bad_examples) < 10:
                bad_examples.append(f"{pid}: no longer exported (filtered out, or the MAT file has fewer ROIs)")
            continue
        nf, nt, nln = hit
        same_t = (np.isnan(t) and np.isnan(nt)) or abs(t - nt) < 0.01
        same_l = np.isnan(ln) or np.isnan(nln) or abs(nln - ln) <= 0.15 * max(ln, 1)
        if not (same_t and same_l):
            c["changed"] += 1
            if len(bad_examples) < 10:
                bad_examples.append(f"{pid}: time {t:.2f} -> {nt:.2f}, size {ln:.1f} -> {nln:.1f} µm (a different particle?)")
        elif nf != f:
            c["moved"] += 1
            moves[(f, nf)] = moves.get((f, nf), 0) + 1
        else:
            c["same"] += 1

    what = "classified particles" if labelled is not None else "particles"
    print(f"\n{'Flight (before)':34} {what:>12} {'same':>8} {'moved':>7} {'changed':>8} {'gone':>6}")
    tot = {"n": 0, "same": 0, "moved": 0, "changed": 0, "gone": 0}
    for f in sorted(per):
        c = per[f]
        for k in tot:
            tot[k] += c[k]
        flag = "" if c["changed"] == 0 else "   <-- check"
        print(f"{f[:34]:34} {c['n']:12,} {c['same']:8,} {c['moved']:7,} {c['changed']:8,} {c['gone']:6,}{flag}")
    print(f"{'TOTAL':34} {tot['n']:12,} {tot['same']:8,} {tot['moved']:7,} {tot['changed']:8,} {tot['gone']:6,}")
    if bad_examples:
        print("\nExamples:")
        for b in bad_examples:
            print("  " + b)
    if moves:
        print("\nParticles now in a differently named flight:")
        for (f, nf), k in sorted(moves.items()):
            print(f"  {f} -> {nf}: {k:,}")
        with open(a.sql, "w") as fh:
            fh.write("-- Move classifications to the particles' new flight ids (written by check_reimport.py).\n")
            fh.write("-- Each line moves one particle's labels; label_log keeps the old ids as history.\n")
            for f, s, i in zip(oF, oS, oI):
                hit = new.get((s, int(i)))
                if hit is None or hit[0] == f:
                    continue
                pid = f"{f}:{s}:{int(i)}"
                if labelled is not None and pid not in labelled:
                    continue
                npid = f"{hit[0]}:{s}:{int(i)}"
                fh.write(f"UPDATE labels SET pid='{npid}', flight_id='{hit[0]}' WHERE pid='{pid}';\n")
        print(f"Wrote {a.sql}: check it, then run it with mysql (see the top of this script).")
    if tot["changed"] == 0 and tot["gone"] == 0 and not moves:
        print(f"\nAll {tot['n']:,} {what} are the same particles as before: the classifications are safe.")
    elif tot["changed"]:
        print("\nSome particle ids now point at different particles. Nothing in the database has changed, but don't"
              " classify those flights until it's sorted out: the labels can be matched up again by time and size.")


if __name__ == "__main__":
    main()
