#!/usr/bin/env python3
"""
relabel_catalogue.py - correct the flight name and/or date of some particles in a pipeline
particle_catalogue.csv, without re-exporting the PNGs.

The import window's flights/days come from the catalogue's "flight" and "date" columns
(written by the PNG export from flights.csv). If one flight or day got the wrong label, fix
flights.csv for next time, then fix the catalogue with this, then press "Import project..."
again on the Admin page.

    # a day labelled 2017-02-03 that should be 2017-02-04
    python3 relabel_catalogue.py /mnt/CPI-raw/CLACE2017/particle_catalogue.csv \\
        --date 2017-02-03 --to-date 2017-02-04
    # a flight with the wrong date (and/or name)
    python3 relabel_catalogue.py /mnt/CPI-raw/DCMEX/particle_catalogue.csv \\
        --flight C301 --to-date 2022-07-29 [--to-flight C302]

Rows are chosen by --flight and/or --date (both must match if both are given). Only those
rows change; everything else is copied byte for byte. The PNGs stay where they are (their
folder keeps the old name, which doesn't matter: the catalogue says where each one is).
The original is kept as particle_catalogue.csv.bak. Run without --go to see how many rows
would change.
"""
import argparse
import csv
import io
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogue")
    ap.add_argument("--flight", help="rows whose flight is this")
    ap.add_argument("--date", help="rows whose date is this (as written in the catalogue, e.g. 2017-02-03)")
    ap.add_argument("--to-flight", help="new flight name")
    ap.add_argument("--to-date", help="new date")
    ap.add_argument("--go", action="store_true", help="write the corrected catalogue (otherwise just count)")
    a = ap.parse_args()
    if not (a.flight or a.date):
        sys.exit("Say which rows: --flight and/or --date")
    if not (a.to_flight is not None or a.to_date):
        sys.exit("Say what to change: --to-flight and/or --to-date")
    csv.field_size_limit(10 ** 8)
    src = a.catalogue
    out = src + ".new"
    with open(src, newline="", encoding="utf-8-sig") as fh:
        first = fh.readline()
        head = next(csv.reader([first]))
        col = {h.strip(): i for i, h in enumerate(head)}
        for c in ("flight", "date"):
            if c not in col:
                sys.exit(f"{src} has no '{c}' column")
        iflt, idate = col["flight"], col["date"]
        nl = "\r\n" if first.endswith("\r\n") else "\n"
        token = a.date or a.flight                       # cheap test before parsing a row
        changed = rows = 0
        before = {}
        dst = open(out, "w", newline="", encoding="utf-8") if a.go else None
        if dst:
            dst.write(first)
        for line in fh:
            while line.count('"') % 2:                   # a quoted field running onto the next line
                more = fh.readline()
                if not more:
                    break
                line += more
            rows += 1
            if token not in line:
                if dst:
                    dst.write(line)
                continue
            r = next(csv.reader([line]))
            f = r[iflt].strip() if iflt < len(r) else ""
            d = r[idate].strip() if idate < len(r) else ""
            if (a.flight and f != a.flight) or (a.date and d != a.date):
                if dst:
                    dst.write(line)
                continue
            before[(f, d)] = before.get((f, d), 0) + 1
            if a.to_flight is not None:
                r[iflt] = a.to_flight
            if a.to_date:
                r[idate] = a.to_date
            changed += 1
            if dst:
                buf = io.StringIO()
                csv.writer(buf, lineterminator=nl).writerow(r)
                dst.write(buf.getvalue())
            if rows % 1000000 == 0:
                print(f"\r{rows:,} rows read, {changed:,} to change", end="", flush=True)
    print(f"\r{rows:,} rows read; {changed:,} match", " " * 20)
    for (f, d), n in sorted(before.items()):
        print(f"  flight {f or '(none)'}, date {d or '(none)'}: {n:,} rows -> "
              f"flight {a.to_flight if a.to_flight is not None else (f or '(none)')}, date {a.to_date or d}")
    if not a.go:
        print("Nothing written. Run again with --go to change the catalogue.")
        return
    dst.close()
    if not changed:
        os.remove(out)
        print("No rows matched: the catalogue is unchanged.")
        return
    st = os.stat(src)
    os.replace(src, src + ".bak")
    os.replace(out, src)
    try:
        os.chmod(src, st.st_mode & 0o777)
    except OSError:
        pass
    print(f"Written. The original is kept as {src}.bak (delete it once the import looks right).")
    print("Now press 'Import project...' on the Admin page: it rereads the changed catalogue.")


if __name__ == "__main__":
    main()
