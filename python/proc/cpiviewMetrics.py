"""CPIview focus and edge-quality measures, alongside this processing (``--cpiview``).

A port of the particle-finding and focus code of SPEC's CPIview 2 (IDL,
Libraries/processParticles.pro), applied to each ROI alongside the normal
processing:

* difference image = ROI - background (grey levels);
* 12 passes of a 3x3 median filter; gradient = |Sobel x| + |Sobel y| (IDL SOBEL),
  zeroed in a 2-pixel border;
* objects = filtered difference < -25 grey levels (-15 if fewer than 25x25 pixels
  pass), holes not touching the ROI edge filled, 8-connected;
* "distinct" = fraction of the object's boundary pixels with gradient > 40
  (CPIview keeps an object only if this exceeds 0.4);
* cut-off = % of boundary pixels within 1 pixel of the ROI edge (CPIview's
  spectra use particles under 6 %);
* focus: at 16 points along the ordered boundary, the gradient profile on a
  +-15 pixel line perpendicular to the boundary; the full width at half maximum
  of the gradient peak nearest the boundary (baseline = mean of the lowest third;
  FWHM < 3 -> 30); sample focus = 100 * peak / FWHM / 110.  CPIview's focus
  value is the mean of the samples; the spread (SD), min, max and median are
  kept too, with the median FWHM (edge width, pixels) and median peak gradient.

The object measured is the one overlapping this processing's particle most
(else the largest).  Values are NaN where CPIview would find nothing.  Only
CPIview's focus and edge measures are kept; sizes come from this processing.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy import ndimage

FIELDS = ("cpv_focus", "cpv_focus_sd", "cpv_focus_min", "cpv_focus_max", "cpv_focus_median", "cpv_samples",
          "cpv_fwhm_px", "cpv_peak", "cpv_distinct", "cpv_cutoff_pct")


def _empty() -> Dict[str, float]:
    return {k: float("nan") for k in FIELDS}


def filter_image(diff: np.ndarray, passes: int = 12):
    """(filtered image, gradient) as CPIview's FilterImg."""
    img = np.asarray(diff, float)
    for _ in range(passes):
        img = ndimage.median_filter(img, size=3, mode="nearest")
    # IDL SOBEL: |Gx| + |Gy| with the 3x3 Sobel kernels
    grad = np.abs(ndimage.sobel(img, axis=0, mode="nearest")) + np.abs(ndimage.sobel(img, axis=1, mode="nearest"))
    grad[:2, :] = 0
    grad[-2:, :] = 0
    grad[:, :2] = 0
    grad[:, -2:] = 0
    return img, grad


def find_objects(filtered: np.ndarray, dark: float = -25.0):
    """Labelled objects (8-connected, interior holes filled) as CPIview's FindObjects."""
    dark = min(dark, -15.0)
    mask = filtered < dark
    if mask.sum() < 2:
        return np.zeros(filtered.shape, int), 0
    if mask.sum() < 25 * 25:
        mask = filtered < dark + 10.0
        if mask.sum() < 2:
            return np.zeros(filtered.shape, int), 0
    # fill background regions that do not touch the ROI edge (2-pixel margin, as CPIview)
    background, nb = ndimage.label(~mask)
    if nb:
        h, w = mask.shape
        for k in range(1, nb + 1):
            rr, cc = np.nonzero(background == k)
            if rr.min() > 1 and rr.max() < h - 2 and cc.min() > 1 and cc.max() < w - 2:
                mask[rr, cc] = True
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    return labels, n


def _ordered_boundary(obj: np.ndarray) -> np.ndarray:
    """Ordered boundary pixel coordinates (row, col) of a single object."""
    from skimage import measure
    padded = np.pad(obj.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return np.zeros((0, 2))
    c = max(contours, key=len) - 1.0
    pts = np.round(c).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, obj.shape[0] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, obj.shape[1] - 1)
    keep = np.ones(len(pts), bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts = pts[keep]
    if len(pts) > 1 and np.all(pts[0] == pts[-1]):
        pts = pts[:-1]
    return pts


def compute_fwhm(y: np.ndarray, start: int):
    """(FWHM in samples, peak value) of the gradient peak nearest ``start`` (CPIview ComputeFWHM)."""
    y = np.asarray(y, float)
    n = len(y)
    ys = np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, "valid")
    ys = np.convolve(np.pad(ys, 1, mode="edge"), np.ones(3) / 3, "valid")
    offsets = np.arange(-3, 4)

    def is_peak(p):
        nb = np.clip(offsets + p, 0, n - 1)
        return int(np.argmax(ys[nb])) == 3

    left = 0
    p = start
    while p >= 3:
        if is_peak(p):
            left = p
            break
        p -= 1
    right = n - 1
    p = start + 1
    while p < n - 3:
        if is_peak(p):
            right = p
            break
        p += 1
    peak_pos = left if (start - left) <= (right - start) else right
    nb = np.clip(offsets + peak_pos, 0, n - 1)
    peak_pos = int(nb[int(np.argmax(y[nb]))])
    peak = float(y[peak_pos])
    base_n = max(1, n // 3)
    baseline = float(np.mean(np.sort(y)[:base_n]))
    half = (peak + baseline) / 2.0
    lpos = 0
    for q in range(peak_pos - 1, -1, -1):
        if y[q] <= half:
            lpos = q
            break
    rpos = n - 1
    for q in range(peak_pos + 1, n):
        if y[q] <= half:
            rpos = q
            break
    return rpos - lpos, peak


def compute_focus(boundary: np.ndarray, grad: np.ndarray, samples: int = 16, half_len: int = 15):
    """CPIview ComputeFocus: per-sample focus values, FWHMs and peaks."""
    nb = len(boundary)
    if nb < 7:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    samples = min(samples, nb)
    step = nb / float(samples)
    rows, cols = grad.shape
    focus, widths, peaks = [], [], []
    for i in range(samples):
        k = min(int(round(i * step)), nb - 1)
        idx = [(k + d) % nb for d in (-2, -1, 0, 1, 2)]
        pts = boundary[idx].astype(float)
        x0, y0 = pts[2]
        d = pts - pts.mean(axis=0)
        # boundary direction (principal axis of the 5 points); the cross-section is perpendicular to it
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        tangent = vt[0]
        normal = np.array([-tangent[1], tangent[0]])
        s = np.array([x0, y0]) - half_len * normal
        e = np.array([x0, y0]) + half_len * normal
        # clip the line to the image, as CPIview does
        lo, hi = 0.0, 1.0
        for dim, size in ((0, rows), (1, cols)):
            a, b = s[dim], e[dim]
            if b != a:
                t0, t1 = (0 - a) / (b - a), (size - 1 - a) / (b - a)
                lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
            elif not 0 <= a <= size - 1:
                lo, hi = 1.0, 0.0
        if hi <= lo:
            continue
        s, e = s + lo * (e - s), s + hi * (e - s)
        n_pts = int(math.sqrt(float(np.sum((e - s) ** 2))))
        if n_pts <= 15:
            continue
        t = np.arange(n_pts) / float(n_pts)
        px = np.round(s[0] + t * (e[0] - s[0])).astype(int)
        py = np.round(s[1] + t * (e[1] - s[1])).astype(int)
        px[-1], py[-1] = int(round(e[0])), int(round(e[1]))
        px = np.clip(px, 0, rows - 1)
        py = np.clip(py, 0, cols - 1)
        if n_pts > 26:
            bound = n_pts // 2
        else:
            bound = int(np.argmin((px - x0) ** 2 + (py - y0) ** 2))
        fwhm, peak = compute_fwhm(grad[px, py], bound)
        if fwhm < 3:
            fwhm = 30
        focus.append(100.0 * peak / float(fwhm) / 110.0)
        widths.append(float(fwhm))
        peaks.append(peak)
    return np.array(focus), np.array(widths), np.array(peaks)


def cpiview_metrics(image: np.ndarray, background: np.ndarray, particle: Optional[np.ndarray] = None,
                    gradient_threshold: float = 40.0, dark: float = -25.0, median_passes: int = 12) -> Dict[str, float]:
    """CPIview measures of the particle in one ROI (see the module docstring)."""
    out = _empty()
    diff = np.asarray(image, float) - np.asarray(background, float)
    if diff.ndim != 2 or min(diff.shape) < 5:
        return out
    filtered, grad = filter_image(diff, median_passes)
    labels, n = find_objects(filtered, dark)
    if n == 0:
        return out
    if particle is not None and particle.shape == labels.shape and np.any(labels[particle] > 0):
        counts = np.bincount(labels[particle].ravel(), minlength=n + 1)[1:]
        pick = int(np.argmax(counts)) + 1
    else:
        pick = int(np.argmax(np.bincount(labels.ravel(), minlength=n + 1)[1:])) + 1
    obj = labels == pick
    edge = obj & ~ndimage.binary_erosion(obj, structure=np.ones((3, 3), bool))
    bpix = np.argwhere(edge)
    if len(bpix) < 6:
        return out
    h, w = obj.shape
    out["cpv_distinct"] = float(np.mean(grad[edge] > gradient_threshold))
    at_edge = (bpix[:, 0] <= 1) | (bpix[:, 0] >= h - 2) | (bpix[:, 1] <= 1) | (bpix[:, 1] >= w - 2)
    out["cpv_cutoff_pct"] = 100.0 * float(np.mean(at_edge))
    boundary = _ordered_boundary(obj)
    focus, widths, peaks = compute_focus(boundary, grad)
    if len(focus):
        out["cpv_focus"] = float(focus.mean())
        out["cpv_focus_sd"] = float(focus.std(ddof=1)) if len(focus) > 1 else 0.0
        out["cpv_focus_min"], out["cpv_focus_max"] = float(focus.min()), float(focus.max())
        out["cpv_focus_median"] = float(np.median(focus))
        out["cpv_fwhm_px"] = float(np.median(widths))
        out["cpv_peak"] = float(np.median(peaks))
    out["cpv_samples"] = float(len(focus))
    return out
