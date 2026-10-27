"""Data-limited, processing-aware optical threshold DOF constraints.

Never changes particle flags or focus parameters. We only estimate a physical
contrast cutoff if the measurements *bracket* it on the appropriate side.
"""
from __future__ import annotations

import numpy as np


def _num(row, key):
    try:
        return float(row.get(key, float('nan')))
    except (TypeError, ValueError):
        return float('nan')


def _side_crossing(d, signal, threshold, sign, min_bin=4, bins=16):
    """First observed outward crossing of the robust/absolute minimum."""
    u = sign * np.asarray(d, dtype=float)
    v = np.asarray(signal, dtype=float)
    valid = np.isfinite(u) & np.isfinite(v) & (u >= 0)
    u, v = u[valid], v[valid]
    if len(u) < min_bin * 3:
        return None
    # Quantile spacing adapts to irregular stage scans; ensure bins contain data.
    edges = np.unique(np.quantile(u, np.linspace(0, 1, bins + 1)))
    if len(edges) < 4:
        return None
    positions, contrasts = [], []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (u >= a) & ((u <= b) if k == len(edges)-2 else (u < b))
        if mask.sum() >= min_bin:
            positions.append(float(np.median(u[mask])))
            contrasts.append(float(np.median(v[mask])))
    if len(positions) < 3 or contrasts[0] >= threshold:
        return None
    for i in range(1, len(positions)):
        if contrasts[i-1] < threshold <= contrasts[i]:
            a, b = contrasts[i-1], contrasts[i]
            fraction = (threshold - a) / (b - a) if b > a else 0.0
            return float(sign * (positions[i-1] + fraction * (positions[i] - positions[i-1])))
    return None


def contrast_dof_from_rows(rows, focus_model, foc_crit=None, robust=False):
    """Data-supported acquisition-threshold DOF, not a particle rejection rule.

    The recorded CPI IThreshold is the threshold on (background - raw image),
    *not* the relative Python boundary depth of 0.55. For each population and
    acquisition setting, find observed outward crossing of the median raw
    shadow deficit below IThreshold. No extrapolation beyond the sampled ROIs.
    """
    try:
        from . import sizing
    except ImportError:
        import sizing
    groups = {}
    for row in rows:
        L, d = _num(row, 'actual_length_um'), _num(row, 'd_signed_plane')
        level = _num(row, 'acq_IThreshold')
        peak = _num(row, 'acq_peak_difference')
        if not all(np.isfinite(v) for v in (L, d, level, peak)) or L <= 0 or level <= 0:
            continue
        key = (str(row.get('label', '') or f'{L:g} µm'), round(L, 3), int(level))
        groups.setdefault(key, []).append((d, peak))
    output = []
    for (label, L, threshold), pairs in sorted(groups.items()):
        baseline = sizing.depth_of_field(focus_model, irel=1.0, foc_crit=foc_crit, L=L)
        d, peak = np.asarray(pairs, float).T
        # Crossing of increasing negative deficit (threshold - peak) on each side.
        neg = _side_crossing(d, threshold - peak, 0.0, -1)
        pos = _side_crossing(d, threshold - peak, 0.0, +1)
        lower, upper = baseline['d_minus'], baseline['d_plus']
        if neg is not None:
            lower = max(lower, neg)
        if pos is not None:
            upper = min(upper, pos)
        total = max(0., upper - lower) if np.isfinite(lower) and np.isfinite(upper) else float('nan')
        output.append(dict(label=label, L_um=L, n=len(pairs), acquisition_threshold_counts=threshold,
                           focus_d_minus=baseline['d_minus'], focus_d_plus=baseline['d_plus'],
                           contrast_d_minus=neg, contrast_d_plus=pos,
                           combined_d_minus=lower, combined_d_plus=upper,
                           combined_total_mm=total,
                           status='observed acquisition deficit crossings only; conditional on saved ROIs'))
    return output
