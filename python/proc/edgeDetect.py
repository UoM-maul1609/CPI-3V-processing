"""Particle edge detection and geometry from CPI ROIs, with every option recorded.

Two methods, selected by ``EdgeOptions.method``:

``legacy``
    The original processing (proc/imageStats.py): background difference,
    100 median passes, Otsu threshold, 11 more median passes on the mask,
    contour, mask grown by ``contains_points(radius=2)``, length = ellipse
    major axis, 30 boundary points.  Kept unchanged so old and new results can
    be compared on the same particles.

``transmittance``
    1. Transmittance ``T = image / background``: the fraction of the local
       illumination that reaches the camera.  This removes the dependence on
       laser power and on bright/dark parts of the background, which a
       threshold in grey levels (or on the background difference) has.
       ``T`` is renormalised by the median of the ROI border, so a background
       taken at a slightly different laser power does not matter.
    2. Light denoising only (``denoise`` passes of a 3x3 median; default 1),
       so corners and thin arms survive.
    3. The particle is the connected region with ``T < edge_level`` that
       contains the darkest pixels; holes (the bright centre of transparent
       drops/beads, the Poisson spot) are filled.
    4. Its edge is traced **sub-pixel** at ``T = edge_level`` (marching
       squares on the smoothed transmittance).  For coherent illumination
       the geometric edge of an opaque object in focus lies near
       ``T = 0.25`` (half the amplitude); a mid-way threshold (~0.5) puts it
       outside the object and oversizes by a roughly constant amount.  The
       level is an option so it can be calibrated on beads (choose the level
       at which beads of different sizes come out at their true size in
       focus).
    5. Geometry from the sub-pixel polygon: maximum Feret length (the
       "maximum dimension" of the calibration papers), minimum Feret width,
       area (shoelace), perimeter, equivalent-area diameter, roundness
       ``Ar = area / (pi/4 L^2)``; the ellipse major axis is kept as
       ``len_ellipse`` for comparison with legacy.
    6. Focus: the legacy edge-gradient metric (for continuity) **and**
       ``focus_T``, the same gradient measured on the transmittance, which
       does not scale with local illumination.
    7. The boundary is stored with as many points as the shape needs: the
       sub-pixel contour simplified with a tolerance of ``boundary_tol_px``
       (Douglas–Peucker), so corners are kept and circles stay light.

Everything that defines a run (method, levels, passes, pixel size, version)
is returned by :func:`options_record` and written into every processed file
as ``proc_options``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure

EDGE_CODE_VERSION = "2026.10.20"


@dataclass
class EdgeOptions:
    method: str = "legacy"             # "legacy" or "transmittance"
    pixel_um: float = 2.3
    # transmittance method
    edge_mode: str = "fixed"           # "fixed": edge at T = edge_level;
                                       # "relative": edge at T = 1 - edge_fraction*(1 - T_min), i.e. a fixed
                                       # fraction of the particle's own shadow depth (keeps faint, small,
                                       # defocused particles detectable and measured consistently)
    edge_level: float = 0.5            # T at the edge, fixed mode (0.25 = coherent in-focus edge; calibrate)
    edge_fraction: float = 0.65        # fraction of the shadow depth at the edge, relative mode (best on 210302 beads)
    core_level: float = 0.45           # fixed mode: a particle must have pixels darker than this (T)
    min_contrast: float = 0.15         # relative mode: ... at least this much darker than the background
    min_dark_pixels: int = 10          # ... at least this many pixels
    denoise: int = 1                   # 3x3 median passes on T (legacy uses 100)
    smooth_sigma: float = 0.0          # optional Gaussian (px) before tracing the edge
    border_normalise: bool = True      # divide T by the median transmittance of the ROI border
    fill_holes: bool = True            # bright centres of transparent particles are part of the particle
    boundary_tol_px: float = 0.25      # boundary simplification tolerance (px); 0 = keep every point
    max_boundary_points: int = 2000
    focus_points: int = 60             # points along the edge used for the focus metrics
    notes: str = ""

    def as_record(self) -> Dict[str, object]:
        out = asdict(self)
        out["code_version"] = EDGE_CODE_VERSION
        return out


def options_record(options: EdgeOptions, extra: Optional[dict] = None) -> Dict[str, object]:
    """Everything needed to reproduce (and label) a processing run."""
    record = options.as_record()
    record["processed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if extra:
        record.update(extra)
    return record


def options_json(options: EdgeOptions, extra: Optional[dict] = None) -> str:
    return json.dumps(options_record(options, extra), sort_keys=True)


# ----------------------------------------------------------------------------- geometry helpers
def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(poly: np.ndarray) -> float:
    closed = np.vstack([poly, poly[:1]])
    return float(np.sum(np.hypot(*np.diff(closed, axis=0).T)))


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain) of 2-D points, counter-clockwise."""
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def feret(points: np.ndarray) -> Tuple[float, float, float]:
    """(max Feret length, min Feret width, orientation of the max in degrees) of a point set."""
    pts = np.asarray(points, dtype=float)
    try:
        import cv2
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(float)
    except Exception:                      # OpenCV missing: pure-Python hull
        hull = _hull(pts)
    if len(hull) < 2:
        return 0.0, 0.0, 0.0
    d = hull[:, None, :] - hull[None, :, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    length = float(dist[i, j])
    angle = math.degrees(math.atan2(hull[j, 0] - hull[i, 0], hull[j, 1] - hull[i, 1]))
    if len(hull) < 3:
        return length, 0.0, angle
    edges = np.roll(hull, -1, axis=0) - hull
    norms = np.hypot(edges[:, 0], edges[:, 1])
    ok = norms > 0
    normals = np.column_stack([-edges[ok, 1], edges[ok, 0]]) / norms[ok, None]
    proj = hull @ normals.T                        # (n_points, n_edges): rotating calipers, vectorised
    width = float(np.min(proj.max(axis=0) - proj.min(axis=0)))
    return length, width, angle


def simplify(poly: np.ndarray, tol: float, max_points: int) -> np.ndarray:
    if tol > 0 and len(poly) > 8:
        closed = np.vstack([poly, poly[:1]])
        poly = measure.approximate_polygon(closed, tolerance=tol)[:-1]
    if len(poly) > max_points:
        poly = poly[np.linspace(0, len(poly) - 1, max_points).round().astype(int)]
    return poly


def _resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    closed = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def edge_gradient_focus(image: np.ndarray, poly: np.ndarray, n_points: int, half_width: float = 2.0) -> float:
    """Mean absolute intensity gradient across the edge (per pixel), sampled normal to the contour.

    Same idea as the legacy metric (gradient over +-2 px normal to the
    boundary), on whatever image is passed: the background difference for the
    legacy value, the transmittance for ``focus_T``.
    """
    if len(poly) < 4:
        return float("nan")
    pts = _resample_closed(poly, n_points)
    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    norm[norm == 0] = 1.0
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / norm[:, None]
    a = pts + half_width * normal
    b = pts - half_width * normal
    va = ndimage.map_coordinates(image.astype(float), [a[:, 0], a[:, 1]], order=1, mode="nearest")
    vb = ndimage.map_coordinates(image.astype(float), [b[:, 0], b[:, 1]], order=1, mode="nearest")
    return float(np.mean(np.abs(va - vb)) / (2 * half_width))


# ----------------------------------------------------------------------------- transmittance method
def transmittance(image: np.ndarray, background: np.ndarray, options: EdgeOptions) -> np.ndarray:
    im = image.astype(float)
    bg = np.maximum(background.astype(float), 1.0)
    T = im / bg
    if options.border_normalise and min(T.shape) > 4:
        border = np.concatenate([T[0], T[-1], T[1:-1, 0], T[1:-1, -1]])
        ref = float(np.median(border))
        if ref > 0.2:
            T = T / ref
    for _ in range(max(0, int(options.denoise))):
        T = ndimage.median_filter(T, size=3, mode="nearest")
    if options.smooth_sigma > 0:
        T = ndimage.gaussian_filter(T, options.smooth_sigma)
    return T


def measure_transmittance(image: np.ndarray, background: np.ndarray, options: EdgeOptions) -> Optional[dict]:
    """Geometry, focus and boundary of the main particle in one ROI, or None if there is none."""
    T = transmittance(image, background, options)
    if options.edge_mode == "relative":
        candidate = T < 1.0 - options.min_contrast
        if np.count_nonzero(candidate) < options.min_dark_pixels:
            return None
        labels, n = ndimage.label(candidate)
        depth = ndimage.sum(1.0 - T, labels, index=np.arange(1, n + 1))
        main = labels == (int(np.argmax(depth)) + 1)
        t_min = float(np.sort(T[main])[:3].mean())           # robust shadow depth of this particle
        edge_level = 1.0 - options.edge_fraction * (1.0 - t_min)
    else:
        if np.count_nonzero(T < options.core_level) < options.min_dark_pixels:
            return None
        edge_level = options.edge_level
    dark = T < edge_level
    labels, n = ndimage.label(dark)
    if n == 0:
        return None
    # the region holding the darkest pixels, weighted by size (avoids a single dark speck)
    sizes = ndimage.sum(np.ones_like(T), labels, index=np.arange(1, n + 1))
    depth = ndimage.sum(np.clip(edge_level - T, 0, None), labels, index=np.arange(1, n + 1))
    best = int(np.argmax(depth)) + 1
    region = labels == best
    if options.fill_holes:
        region = ndimage.binary_fill_holes(region)
    if region.sum() < 3:
        return None
    # Trace the T = edge_level contour of this region only: outside it is set bright,
    # filled holes dark, so the iso-line follows the real outer edge sub-pixel.
    grown = ndimage.binary_dilation(region, iterations=2)
    bright = max(1.0, edge_level + 0.5)
    field_ = np.where(grown, T, bright)
    field_ = np.where(region & ~(T < edge_level), np.minimum(field_, edge_level - 0.05), field_)
    padded = np.pad(field_, 1, mode="constant", constant_values=bright)
    contours = measure.find_contours(padded, edge_level)
    if not contours:
        return None
    poly = max(contours, key=lambda c: polygon_area(c)) - 1.0      # undo padding; (row, col)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) < 4:
        return None
    px = options.pixel_um
    length_px, width_px, angle = feret(poly)
    area_px = polygon_area(poly)
    perim_px = polygon_perimeter(poly)
    rows, cols = np.nonzero(region)
    # ellipse major axis of the same region (for comparison with legacy)
    props = measure.regionprops(region.astype(np.uint8))[0]
    major = getattr(props, "axis_major_length", None) or props.major_axis_length
    centroid = poly.mean(axis=0)
    centre_T = float(ndimage.map_coordinates(T, [[centroid[0]], [centroid[1]]], order=1, mode="nearest")[0])
    diff = image.astype(np.int16) - background.astype(np.int16)
    boundary = simplify(poly, options.boundary_tol_px, options.max_boundary_points)
    boundary = np.vstack([boundary, boundary[:1]])      # closed, like the legacy 30-point boundary
    return {
        "len": length_px * px,
        "wid": width_px * px,
        "area": area_px * px ** 2,
        "perimeter": perim_px * px,
        "deq": 2.0 * math.sqrt(area_px / math.pi) * px,
        "round": area_px / (math.pi / 4.0 * length_px ** 2) if length_px > 0 else float("nan"),
        "len_ellipse": float(major) * px,
        "orientation": angle,
        "centroid": centroid,
        "focus": edge_gradient_focus(diff, poly, options.focus_points),
        "focus_T": edge_gradient_focus(T, poly, options.focus_points),
        "T_min": float(T[region].min()),
        "edge_level_used": float(edge_level),
        "T_centre": centre_T,
        "T_border": float(np.median(np.concatenate([T[0], T[-1], T[:, 0], T[:, -1]]))),
        # 1 if the particle reaches the ROI border: it may be cut off, so its size is a lower limit
        "edge_touch": int(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()),
        "boundary": boundary,
        "n_boundary": int(len(boundary) - 1),
    }
