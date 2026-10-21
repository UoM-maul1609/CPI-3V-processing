"""Carry saved flags onto reprocessed MAT files by matching the ROI images themselves.

Flags are normally re-found by ROI time + camera position (``particle_key``).
That breaks when reprocessing changes ROI times -- e.g. after the fix to the
ROI-to-frame timing in ``postProcess.py`` -- so for a copy onto reprocessed
data each old ROI is found in the new file by its corners and its raw pixels,
which no processing step changes.  The flags rows are then rewritten with the
new ROI index, time and ID, and the normal matching attaches them uniquely.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import ParticleCatalog, matlab_datenum_to_datetime


def _unwrap(value):
    v = np.asarray(value)
    while v.dtype == object and v.size == 1:
        v = np.asarray(v.ravel()[0])
    return v


def roi_signatures(mat_file):
    """[(signature, time_matlab, (sx, ex, sy, ey))] for every ROI of a processed MAT file, in ROI order."""
    import scipy.io as sio
    roi = sio.loadmat(str(mat_file), variable_names=["ROI_N"])["ROI_N"]
    images = roi["IMAGE"][0, 0][0, :]
    get = lambda f: np.ravel(roi[f][0, 0]).astype(float)
    times, sx, ex, sy, ey = get("Time"), get("StartX"), get("EndX"), get("StartY"), get("EndY")
    out = []
    for i in range(len(images)):
        im = _unwrap(images[i]["IM"])
        corners = (int(sx[i]), int(ex[i]), int(sy[i]), int(ey[i]))
        digest = hashlib.blake2b(np.ascontiguousarray(im).tobytes(), digest_size=12)
        digest.update(repr((im.shape, corners)).encode())
        out.append((digest.hexdigest(), float(times[i]), corners))
    return out


def roi_index_map(old_mat, new_mat):
    """old ROI index -> (new ROI index, new time); ROIs absent from the new file are left out."""
    old, new = roi_signatures(old_mat), roi_signatures(new_mat)
    pool = defaultdict(list)
    for j, (sig, _, _) in enumerate(new):
        pool[sig].append(j)
    mapping = {}
    for i, (sig, _, _) in enumerate(old):
        if pool.get(sig):
            j = pool[sig].pop(0)          # identical images: keep their order
            mapping[i] = (j, new[j][1])
    return mapping


def _iso(t):
    try:
        return matlab_datenum_to_datetime(t).isoformat(sep=" ", timespec="milliseconds")
    except (ValueError, OverflowError):
        return ""


def remap_flags_file(flags_in, flags_out, file_pairs, progress=None):
    """Rewrite a flags CSV for reprocessed MAT files.

    ``file_pairs``: {basename: (old_mat_path, new_mat_path)}.  Returns a report
    dict: rows, remapped, unchanged (file not available to compare), missing.
    """
    with open(flags_in, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ParticleCatalog.FLAG_FIELDS)
        rows = list(reader)
    by_file = defaultdict(list)
    for row in rows:
        by_file[Path(str(row.get("source_file", ""))).name].append(row)
    report = {"rows": len(rows), "remapped": 0, "unchanged": 0, "missing": 0, "files": 0, "maps": {}}
    for n, (name, file_rows) in enumerate(sorted(by_file.items()), 1):
        pair = file_pairs.get(name)
        if not pair or not Path(pair[0]).exists() or not Path(pair[1]).exists():
            report["unchanged"] += len(file_rows)
            continue
        if progress:
            progress(n, len(by_file), name)
        mapping = roi_index_map(*pair)
        report["maps"][name] = {old: new for old, (new, _) in mapping.items()}
        report["files"] += 1
        new_path = str(Path(pair[1]).resolve())
        for row in file_rows:
            try:
                old_index = int(float(row.get("roi_index", "")))
            except ValueError:
                report["missing"] += 1
                continue
            hit = mapping.get(old_index)
            if hit is None:
                report["missing"] += 1
                continue
            new_index, new_time = hit
            row["roi_index"] = str(new_index)
            row["time_matlab"] = repr(new_time)
            row["time_iso"] = _iso(new_time)
            row["particle_id"] = f"{new_path}#{new_index}"
            report["remapped"] += 1
    for name, pair in file_pairs.items():           # files with batches but no flag rows
        if name not in report["maps"] and Path(pair[0]).exists() and Path(pair[1]).exists():
            report["maps"][name] = {old: new for old, (new, _) in roi_index_map(*pair).items()}
    with open(flags_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report


def split_id(pid):
    """(file basename, ROI index) of a particle ID '<path>#<index>', or (None, None)."""
    src, sep, roi = str(pid).rpartition("#")
    if not sep or not roi.isdigit():
        return None, None
    return Path(src).name, int(roi)
