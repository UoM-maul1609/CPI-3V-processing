"""Small, Qt-independent helpers for readable calibration plots."""
from __future__ import annotations

import numpy as np


def data_limits(x, y, margin=0.05):
    """Data-only axis limits, excluding draggable guide lines and annotations.

    Ignore records with missing X or Y. Do not truncate outliers: the initial
    view must actually include every finite plotted particle.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("X and Y data must have the same shape")
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return None
    x, y = x[valid], y[valid]

    def extent(values):
        a, b = float(np.min(values)), float(np.max(values))
        delta = b - a
        if delta == 0:
            delta = max(abs(a) * 0.04, 0.01)
        return a - margin * delta, b + margin * delta

    return extent(x), extent(y)


def median_scan_curve(x, focus, decimals=7):
    """Median measured focus at each discrete stage setting.

    Return sorted X and corresponding medians.  Group near-identical floating
    point stage coordinates, ignore non-finite points, and never infer a fitted
    optical response from the medians.  This is a descriptive scan trace.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(focus, dtype=float)
    if xx.shape != yy.shape:
        raise ValueError("X and focus must have the same shape")
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[valid], yy[valid]
    if not len(xx):
        return np.array([], dtype=float), np.array([], dtype=float)
    rounded = np.round(xx, decimals=decimals)
    unique, inv = np.unique(rounded, return_inverse=True)
    medians = np.array([np.median(yy[inv == k]) for k in range(len(unique))])
    return unique, medians


def median_trace(x, y, max_groups=40, min_count=3):
    """Median and inter-quartile range of ``y`` along ``x``, for any metric.

    Discrete stage scans (few distinct X values) are grouped per X setting,
    exactly as :func:`median_scan_curve`.  Continuous X (e.g. acquisition time)
    is split into equal-count bins so the trace stays readable.  Groups with
    fewer than ``min_count`` points are dropped.  Purely descriptive: no model
    is fitted and the measured values are not altered.

    Returns a dict of arrays: x, median, q25, q75, n (all sorted by x).
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.shape != yy.shape:
        raise ValueError("X and Y must have the same shape")
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[valid], yy[valid]
    empty = {k: np.array([], dtype=float) for k in ("x", "median", "q25", "q75", "n")}
    if len(xx) < min_count:
        return empty
    rounded = np.round(xx, decimals=7)
    unique, inverse = np.unique(rounded, return_inverse=True)
    if len(unique) <= max_groups:
        groups = [np.flatnonzero(inverse == k) for k in range(len(unique))]
        centres = list(unique)
    else:
        order = np.argsort(xx, kind="stable")
        nbins = max(2, min(max_groups, len(xx) // max(min_count, 1)))
        groups = [g for g in np.array_split(order, nbins) if len(g)]
        centres = [float(np.median(xx[g])) for g in groups]
    rows = []
    for centre, g in zip(centres, groups):
        if len(g) < min_count:
            continue
        values = yy[g]
        q25, med, q75 = np.percentile(values, [25, 50, 75])
        rows.append((float(centre), float(med), float(q25), float(q75), float(len(g))))
    if not rows:
        return empty
    arr = np.asarray(rows)
    return {"x": arr[:, 0], "median": arr[:, 1], "q25": arr[:, 2], "q75": arr[:, 3], "n": arr[:, 4]}


def trace_extremum(trace, peak="maximum"):
    """X and value of the maximum (or minimum) of a median trace, or None."""
    if trace is None or len(trace.get("x", ())) < 3:
        return None
    k = int(np.argmax(trace["median"]) if peak == "maximum" else np.argmin(trace["median"]))
    return float(trace["x"][k]), float(trace["median"][k])
