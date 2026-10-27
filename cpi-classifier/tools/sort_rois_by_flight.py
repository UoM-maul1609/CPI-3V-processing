#!/usr/bin/env python3
"""
sort_rois_by_flight.py - group CPI .roi files, or the .mat files made from them,
into one folder per flight.

The flights and their dates come from file names like
    cpi_images_AE04_051116.pdf      (flight AE04 on 2005-11-16)
or from a small text file you write (--flights), one flight per line:
    AE04 2005-11-16

Each file's start time comes from its name, MMDDhhmm.roi or MMDDhhmm.mat
(e.g. 11161023.roi = 16 Nov, 10:23). The year is taken from the flight dates.
Other files (full_backgrounds.mat, timeseries.mat, ...) are left alone.

Files are matched to the flight on the same date. A file up to --slack hours
either side of a flight date also matches, for flights that cross midnight
(UTC). Several separate blocks of files on one date are reported, so you can
check them.

Nothing is changed unless you add --apply. Then each flight gets a folder in
--dest, and the ROI files are hard-linked into it: no extra disk space, and
the originals stay where they are. Use --move to move them instead.

    python3 sort_rois_by_flight.py /mnt/CPI-raw/ACTIVE --dest /mnt/CPI-raw/ACTIVE/flights
    python3 sort_rois_by_flight.py /mnt/CPI-raw/ACTIVE --dest /mnt/CPI-raw/ACTIVE/flights --apply
"""
import argparse
import datetime as dt
import os
import re
import shutil
import sys

PDF_RE = re.compile(r"cpi_images_([A-Za-z]+\d+[A-Za-z]?)_(\d{6})\.pdf$", re.I)
ROI_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})(\d{2})\.(roi|mat)$", re.I)


def flights_from_pdfs(folder):
    out = {}
    for name in os.listdir(folder):
        m = PDF_RE.search(name)
        if m:
            yy, mm, dd = int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:])
            out[m.group(1).upper()] = dt.date(2000 + yy if yy < 70 else 1900 + yy, mm, dd)
    return out


def flights_from_file(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2 and not line.lstrip().startswith("#"):
                out[parts[0]] = dt.date.fromisoformat(parts[1])
    return out


def roi_time(name, flight_dates):
    m = ROI_RE.match(name)
    if not m:
        return None
    mo, d, hh, mi = map(int, m.groups()[:4])
    # choose the year that puts this date nearest to one of the flights
    best = None
    for y in sorted({f.year for f in flight_dates} | {f.year - 1 for f in flight_dates} | {f.year + 1 for f in flight_dates}):
        try:
            t = dt.datetime(y, mo, d, hh, mi)
        except ValueError:
            continue
        gap = min(abs((t.date() - f).days) for f in flight_dates)
        if best is None or gap < best[0]:
            best = (gap, t)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="folder with the .roi files (and the cpi_images_*.pdf files)")
    ap.add_argument("--dest", help="where to make the flight folders (default: <src>/flights)")
    ap.add_argument("--flights", help="text file of 'FLIGHT YYYY-MM-DD' lines, instead of reading the PDF names")
    ap.add_argument("--slack", type=float, default=6, help="hours either side of a flight date that still count (default 6)")
    ap.add_argument("--gap", type=float, default=3, help="a gap longer than this many hours splits a day into separate blocks (default 3)")
    ap.add_argument("--ext", default=".roi", choices=[".roi", ".mat"], help="which files to sort (default .roi)")
    ap.add_argument("--apply", action="store_true", help="actually make the folders (default: just show the plan)")
    ap.add_argument("--move", action="store_true", help="move files instead of hard-linking them")
    a = ap.parse_args()

    flights = flights_from_file(a.flights) if a.flights else flights_from_pdfs(a.src)
    if not flights:
        sys.exit("No flights found: no cpi_images_<FLIGHT>_<YYMMDD>.pdf files in the folder. Use --flights.")
    dest = a.dest or os.path.join(a.src, "flights")

    by_date = {}
    for f, d in flights.items():
        by_date.setdefault(d, []).append(f)
    clash = {d: fs for d, fs in by_date.items() if len(fs) > 1}

    rois, odd = [], []
    for name in sorted(os.listdir(a.src)):
        if not name.lower().endswith(a.ext):
            continue
        t = roi_time(name, list(flights.values()))
        (rois if t else odd).append((name, t))

    plan = {f: [] for f in flights}
    unassigned = []
    for name, t in rois:
        same = by_date.get(t.date())
        if same and len(same) == 1:
            plan[same[0]].append((name, t))
            continue
        # nearest flight date within the slack (for flights crossing midnight)
        best = None
        for f, d in flights.items():
            start, end = dt.datetime.combine(d, dt.time(0)), dt.datetime.combine(d, dt.time(23, 59))
            off = 0 if start <= t <= end else min(abs((t - start).total_seconds()), abs((t - end).total_seconds())) / 3600
            if off <= a.slack and (best is None or off < best[0]):
                best = (off, f)
        if best and not (same and len(same) > 1):
            plan[best[1]].append((name, t))
        else:
            unassigned.append((name, t))

    size = lambda n: os.path.getsize(os.path.join(a.src, n))
    print(f"{len(rois)} {a.ext} files, {len(flights)} flights\n")
    print(f"{'flight':8} {'date':10}  {'files':>5}  {'first':>16}  {'last':>16}  {'GB':>6}  blocks")
    for f in sorted(flights, key=lambda x: (flights[x], x)):
        items = sorted(plan[f], key=lambda x: x[1])
        if not items:
            print(f"{f:8} {flights[f]}      0   (no ROI files)")
            continue
        blocks = 1 + sum((items[i][1] - items[i - 1][1]).total_seconds() > a.gap * 3600 for i in range(1, len(items)))
        gb = sum(size(n) for n, _ in items) / 1e9
        print(f"{f:8} {flights[f]}  {len(items):5}  {items[0][1]:%Y-%m-%d %H:%M}  {items[-1][1]:%Y-%m-%d %H:%M}  {gb:6.2f}  {blocks}{'  <- check: several blocks' if blocks > 1 else ''}")
    if clash:
        print("\nMore than one flight on the same date (files on these dates are left unassigned; use --flights with times or sort them by hand):")
        for d, fs in clash.items():
            print(f"  {d}: {', '.join(fs)}")
    if unassigned:
        print(f"\n{len(unassigned)} ROI files match no flight:")
        for n, t in unassigned[:40]:
            print(f"  {n}  {t:%Y-%m-%d %H:%M}")
        if len(unassigned) > 40:
            print(f"  ... and {len(unassigned) - 40} more")
    if odd:
        print(f"\n{len(odd)} {a.ext} files whose names aren't MMDDhhmm{a.ext} (skipped): " + ", ".join(n for n, _ in odd[:10]) + (" ..." if len(odd) > 10 else ""))

    if not a.apply:
        print(f"\nNothing changed. Add --apply to make the folders in {dest}" + (" (moving files)" if a.move else " (hard links: no extra space)") + ".")
        return
    for f, items in plan.items():
        if not items:
            continue
        fdir = os.path.join(dest, f)
        os.makedirs(fdir, exist_ok=True)
        for n, _ in items:
            src, dst = os.path.join(a.src, n), os.path.join(fdir, n)
            if os.path.exists(dst):
                continue
            if a.move:
                shutil.move(src, dst)
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)   # different disk: copy instead
    print(f"\nDone: flight folders in {dest}")


if __name__ == "__main__":
    main()
