"""Named time windows within a dataset (e.g. the 15, 40 and 100 µm bead runs of one session).

A window is ``{"name", "start", "end", "label"}`` with start/end as
"YYYY-MM-DD HH:MM:SS" (the ROI clock, as shown in the workbench) and an
optional label (the bead/analogue type accepted in that window).  Windows are
stored in the project file.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np

try:
    from .data import datetime_to_matlab_datenum, matlab_datenum_to_datetime
except ImportError:
    from data import datetime_to_matlab_datenum, matlab_datenum_to_datetime

FORMAT = "%Y-%m-%d %H:%M:%S"


def window_bounds(window) -> tuple:
    """(start, end) as MATLAB datenums."""
    return (datetime_to_matlab_datenum(datetime.strptime(window["start"], FORMAT)),
            datetime_to_matlab_datenum(datetime.strptime(window["end"], FORMAT)))


def make_window(name, start_matlab, end_matlab, label="", pad_seconds=0.0) -> dict:
    pad = timedelta(seconds=pad_seconds)
    start = matlab_datenum_to_datetime(float(start_matlab)) - pad
    end = matlab_datenum_to_datetime(float(end_matlab)) + pad
    # whole seconds, rounded outwards so the edge particles stay inside
    start = start.replace(microsecond=0)
    if end.microsecond:
        end = end.replace(microsecond=0) + timedelta(seconds=1)
    return {"name": str(name), "start": start.strftime(FORMAT), "end": end.strftime(FORMAT), "label": label or ""}


def suggest_windows(times: Iterable[float], gap_minutes: float = 5.0, min_count: int = 10) -> List[dict]:
    """Split acquisition times at gaps longer than ``gap_minutes``; one window per run."""
    t = np.sort(np.asarray([x for x in times if np.isfinite(x)], float))
    if not len(t):
        return []
    cuts = np.flatnonzero(np.diff(t) > gap_minutes / 1440.0) + 1
    out = []
    for k, part in enumerate(np.split(t, cuts)):
        if len(part) >= min_count:
            out.append(make_window(f"Window {len(out) + 1}", part[0], part[-1]))
    return out


def in_window(times, window) -> np.ndarray:
    lo, hi = window_bounds(window)
    t = np.asarray(times, float)
    return (t >= lo) & (t <= hi)


def overlapping(windows) -> List[tuple]:
    """Pairs of window names whose time ranges overlap."""
    bounds = [(w["name"],) + window_bounds(w) for w in windows]
    pairs = []
    for i, (a, a0, a1) in enumerate(bounds):
        for b, b0, b1 in bounds[i + 1:]:
            if a0 <= b1 and b0 <= a1:
                pairs.append((a, b))
    return pairs
