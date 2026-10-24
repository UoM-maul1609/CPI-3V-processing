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
from skimage import draw, measure

EDGE_CODE_VERSION = "2026.10.21.fixed-contrast"


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
    edge_fraction: float = 0.55        # fraction of the shadow depth at the edge, relative mode (210302 beads: sizes
                                       # grow away from focus on both sides; as consistent as 0.65)
    core_level: float = 0.45           # legacy compatibility; fixed mode now uses min_contrast instead
    min_contrast: float = 0.15         # relative mode: ... at least this much darker than the background
    min_dark_pixels: int = 10          # ... at least this many pixels
    denoise: int = 1                   # 3x3 median passes on T (legacy uses 100)
    smooth_sigma: float = 0.0          # optional Gaussian (px) before tracing the edge
    border_normalise: bool = True      # divide T by the median transmittance of the ROI border
    fill_holes: bool = True            # bright centres of transparent particles are part of the particle
    close_px: int = 4                  # bridge notches up to this radius (bright centre spot of beads), shadow pixels only
    merge_parts: bool = True           # join edge-level parts of one shadow split by a bright axial streak (columns)
    merge_min_px: int = 10             # ... parts smaller than this are ignored
    merge_min_fraction: float = 0.05   # ... or smaller than this fraction of the largest part
    merge_max_gap: float = 0.3         # ... and gaps wider than this fraction of the particle's extent are not bridged
    join_faint: float = 0.5            # column walls: connected at T < 1 - join_faint * min_contrast ...
    join_min_aspect: float = 3.0       # ... each at least this elongated
    join_max_angle: float = 10.0       # ... parallel within this many degrees
    join_min_size: float = 0.3         # ... and the other wall at least this fraction of the main one's area
    fill_bays: float = 0.65            # split-and-merged particles: fill concavities at least this enclosed (0: off)
    fill_bays_max_area: float = 0.08   # ... and no larger than this fraction of the particle
    fill_bays_shadow_cover: float = 0.7   # ... or with at least this fraction inside the particle's weak shadow
    fill_bays_min_aspect: float = 2.0  # bay filling also for particles at least this elongated (columns)
    outline: str = "radial"            # "radial": outline found along rays from the centre and smoothed around it
                                       # (compact particles; falls back to "contour" for shapes it cannot follow);
                                       # "edge": round particles outlined at the steepest rise of T (where a person
                                       # draws the edge), kept continuous; others as "contour";
                                       # "contour": the raw iso-transmittance contour
    outline_smooth_px: float = 5.0     # radial outline: running median over about this much arc
    radial_min_roundness: float = 0.6  # radial outline only if area / (pi/4 Lmax^2) of the raw contour is at least this
                                       # (beads ~0.9, hexagonal plates ~0.85; columns, needles, rosettes keep the contour)
    bg_check: bool = True              # if the particle is in its own background, use a clean one (backgroundCheck.py)
    shadow_circle_padding: Optional[float] = None  # optional signed-shadow size, px beyond fitted contour
    shadow_circle_auto: bool = False       # opt-in experimental plateau radius; fixed +8 remains default
    shadow_circle_max_padding: float = 30.  # maximum exploration beyond contour (px)
    cpiview: bool = False              # also store CPIview's own measures and focus (cpiviewMetrics.py, dat.cpv_*)
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


def _pick_component(labels: np.ndarray, n: int, weight: np.ndarray) -> int:
    """Label of the particle the ROI was cut around.

    The CPI crops each ROI around the particle that triggered it, so other
    shadows are neighbours, usually only partly inside the ROI.  Each dark
    component scores its integrated shadow (``weight`` summed over it), down-
    weighted with distance of its centroid from the ROI centre and heavily if
    it touches the ROI border (it is cut off).  A large, dark neighbour in a
    corner therefore no longer beats a smaller, fainter particle in the middle.
    """
    index = np.arange(1, n + 1)
    depth = np.asarray(ndimage.sum(weight, labels, index=index), float)
    if n == 1:
        return 1
    h, w = labels.shape
    centres = np.asarray(ndimage.center_of_mass(np.ones_like(weight), labels, index), float).reshape(n, 2)
    r = np.hypot((centres[:, 0] - (h - 1) / 2) / (h / 2), (centres[:, 1] - (w - 1) / 2) / (w / 2))
    border = np.zeros(n + 1, bool)
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        border[np.unique(edge)] = True
    touches = border[1:]
    score = depth * np.exp(-(r / 0.6) ** 2) * np.where(touches, 0.1, 1.0)
    return int(np.argmax(score)) + 1


def _circular_median(x: np.ndarray, size: int) -> np.ndarray:
    if size < 3:
        return x
    pad = size // 2
    xp = np.concatenate([x[-pad:], x, x[:pad]])
    return ndimage.median_filter(xp, size=size, mode="nearest")[pad:-pad]


def _roundness(poly: np.ndarray) -> float:
    length, _, _ = feret(poly)
    return polygon_area(poly) / (math.pi / 4.0 * length ** 2) if length > 0 else 0.0


def _mask_roundness(mask: np.ndarray) -> float:
    rows, cols = np.nonzero(mask)
    if rows.size < 3:
        return 0.0
    length, _, _ = feret(np.column_stack([rows, cols]).astype(float))
    length += 1.0                                    # pixel centres -> pixel extent
    return float(rows.size) / (math.pi / 4.0 * length ** 2)


def radial_outline(T: np.ndarray, region: np.ndarray, edge_level: float, smooth_px: float = 5.0,
                   step: float = 0.25, edge_fraction: Optional[float] = None, max_level: float = 0.925,
                   check: Optional[np.ndarray] = None, outermost: bool = True) -> Optional[np.ndarray]:
    """Outline of a compact particle found along rays from its centre, smoothed around it.

    On each ray the edge is where T, walking out from the darkest point of the
    particle on that ray, first rises above ``edge_level`` (sub-pixel).  Dark
    rings and a bright spot inside the particle therefore no longer make the
    outline ragged, and a running median over ``smooth_px`` of arc removes
    single-ray spikes.  Returns None when the particle is not star-shaped from
    its centre (aggregates, strongly concave crystals): the caller keeps the raw
    contour then.  Rays leaving the ROI stop at its border.

    With ``edge_fraction`` the edge level is set ray by ray, as that fraction of
    the shadow depth seen along the ray (smoothed around the particle).  A bead
    that is darker on one side than the other -- uneven illumination, or the
    crescent of a defocused bead -- then gets its whole outline, not only the
    dark side.  ``check``: mask the outline must agree with (default ``region``).
    """
    rows, cols = np.nonzero(region)
    if rows.size < 3:
        return None
    c = np.array([rows.mean(), cols.mean()])
    rmax = float(np.max(np.hypot(rows - c[0], cols - c[1]))) + 1.0
    n = int(np.clip(2 * math.pi * rmax, 32, 360))
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rs = np.arange(0, 1.6 * rmax + 3.0, step)
    rr = c[0] + rs[None, :] * np.sin(th)[:, None]
    cc = c[1] + rs[None, :] * np.cos(th)[:, None]
    h, w = T.shape
    inside = (rr >= 0) & (rr <= h - 1) & (cc >= 0) & (cc <= w - 1)
    P = ndimage.map_coordinates(T, [rr.ravel(), cc.ravel()], order=1, mode="nearest").reshape(rr.shape)
    P[~inside] = np.inf
    Ps = np.where((rs <= 1.2 * rmax)[None, :], P, np.inf)
    jmin = np.argmin(Ps, axis=1)
    if edge_fraction is not None:
        tmin = _circular_median(Ps[np.arange(n), jmin], max(3, (n // 8) | 1))
        levels = np.minimum(1.0 - edge_fraction * (1.0 - tmin), max_level)
    else:
        levels = np.full(n, edge_level)
    if edge_fraction is not None and outermost:
        # the particle's own shadow along each ray (grown by 1 px): a lighter patch inside the particle
        # is crossed, the edge is the LAST point darker than the level before leaving the shadow
        grown = ndimage.binary_dilation(region, iterations=1).astype(float)
        F = ndimage.map_coordinates(grown, [rr.ravel(), cc.ravel()], order=0, mode="constant").reshape(rr.shape) > 0.5
    else:
        F = None
    radius = np.empty(n)
    for k in range(n):
        p = P[k]
        j = int(jmin[k])
        level = levels[k]
        if F is not None:
            dark = np.flatnonzero((p[j:] <= level) & F[k, j:])
            if dark.size:
                j = j + int(dark[-1])
        out = np.flatnonzero(p[j:] > level)
        if out.size == 0:
            radius[k] = rs[-1]
            continue
        j1 = j + int(out[0])
        if j1 == 0:
            radius[k] = 0.0
        elif not np.isfinite(p[j1]):
            radius[k] = rs[j1 - 1]                   # reached the ROI border
        else:
            a, b = p[j1 - 1], p[j1]
            t = (level - a) / (b - a) if b != a else 0.5
            radius[k] = rs[j1 - 1] + min(max(t, 0.0), 1.0) * step
    arc = 2 * math.pi * rmax / n
    size = int(round(smooth_px / max(arc, 1e-3)))
    size = max(3, min(size, max(3, n // 8)))
    radius = _circular_median(radius, size | 1)
    poly = np.column_stack([c[0] + radius * np.sin(th), c[1] + radius * np.cos(th)])
    # star-shaped check: the outline must agree with the thresholded region
    filled = np.zeros_like(region, dtype=bool)
    pr, pc = draw.polygon(np.clip(poly[:, 0], 0, h - 1), np.clip(poly[:, 1], 0, w - 1), shape=region.shape)
    filled[pr, pc] = True
    ref = region if check is None else check
    if edge_fraction is None:
        union = np.count_nonzero(filled | ref)
        if union == 0 or np.count_nonzero(filled & ref) / union < 0.8:
            return None
    else:
        # per-ray levels: the outline must cover the dark core and stay inside the (slightly grown) shadow
        grown = ndimage.binary_dilation(region, iterations=1)
        if (np.count_nonzero(filled & ref) < 0.9 * np.count_nonzero(ref)
                or np.count_nonzero(filled & ~grown) > 0.05 * max(np.count_nonzero(filled), 1)):
            return None
    return poly


def steepest_rise_outline(T: np.ndarray, footprint: np.ndarray, step: float = 0.25, sigma: float = 1.0,
                          max_jump_px: float = 0.5, inner: float = 0.3, smooth: int = 7) -> Optional[np.ndarray]:
    """Closed outline along the steepest outward rise of T, kept continuous (polar dynamic programming).

    Rays from the centre of the particle's shadow; the cost of a radius on a ray
    is minus the outward gradient dT/dr there, and neighbouring rays may change
    radius by at most ``max_jump_px`` per pixel of arc, so a single fringe cannot
    pull out a spike.  This is the edge between the grey disc and the bright
    surround that a person would draw, whatever the shading inside.
    """
    rows, cols = np.nonzero(footprint)
    if rows.size < 3:
        return None
    c = np.array([rows.mean(), cols.mean()])
    rmax = float(np.max(np.hypot(rows - c[0], cols - c[1]))) + 1.0
    n = int(np.clip(2 * math.pi * rmax, 32, 360))
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rs = np.arange(0, 1.5 * rmax + 3.0, step)
    rr = c[0] + rs[None, :] * np.sin(th)[:, None]
    cc = c[1] + rs[None, :] * np.cos(th)[:, None]
    h, w = T.shape
    inside = (rr >= 0) & (rr <= h - 1) & (cc >= 0) & (cc <= w - 1)
    Ts = ndimage.gaussian_filter(T, sigma) if sigma > 0 else T
    P = ndimage.map_coordinates(Ts, [rr.ravel(), cc.ravel()], order=1, mode="nearest").reshape(rr.shape)
    G = np.clip(np.gradient(P, step, axis=1), 0, None)
    G[~inside] = 0.0
    # Search only around the outer edge of the shadow on each ray, so a dark core (whose own edge can be
    # steeper) is never taken for the particle's edge.
    F = ndimage.map_coordinates(footprint.astype(float), [rr.ravel(), cc.ravel()], order=0,
                                mode="constant").reshape(rr.shape) > 0.5
    last = np.where(F.any(axis=1), F.shape[1] - 1 - np.argmax(F[:, ::-1], axis=1), 0)
    r_edge = rs[last]
    half = np.maximum(3.0, inner * r_edge)
    G[np.abs(rs[None, :] - r_edge[:, None]) > half[:, None]] = 0.0
    gmax = G.max()
    if gmax <= 0:
        return None
    cost = -G / gmax
    m = len(rs)
    J = max(1, int(round(max_jump_px * (2 * math.pi * rmax / n) / step)))
    reps = np.concatenate([cost, cost, cost], axis=0)      # circular: solve on three turns, keep the middle
    N = reps.shape[0]
    acc = reps[0].copy()
    back = np.zeros((N, m), np.int32)
    idx = np.arange(m)
    for k in range(1, N):
        best = np.full(m, np.inf)
        arg = np.zeros(m, np.int32)
        for dj in range(-J, J + 1):
            src = np.clip(idx + dj, 0, m - 1)
            cand = acc[src]
            better = cand < best
            best[better] = cand[better]
            arg[better] = src[better]
        acc = best + reps[k]
        back[k] = arg
    j = int(np.argmin(acc))
    path = np.empty(N, np.int32)
    for k in range(N - 1, -1, -1):
        path[k] = j
        j = back[k, j]
    jj = path[n:2 * n]
    radius = rs[jj].astype(float)
    for k in range(n):                                      # sub-pixel: parabola through the gradient peak
        j = jj[k]
        if 0 < j < m - 1:
            a, b, d = G[k, j - 1], G[k, j], G[k, j + 1]
            den = a - 2 * b + d
            if den < 0:
                radius[k] += min(max(0.5 * (a - d) / den, -1.0), 1.0) * step
    radius = _circular_median(radius, smooth)
    poly = np.column_stack([c[0] + radius * np.sin(th), c[1] + radius * np.cos(th)])
    area = polygon_area(poly)
    if not (0.6 * footprint.sum() <= area <= 3.0 * footprint.sum() + 30):
        return None
    return poly


def _elongation(mask: np.ndarray) -> float:
    from skimage.measure import regionprops
    p = regionprops(mask.astype(np.uint8))[0]
    major = getattr(p, "axis_major_length", None) or p.major_axis_length
    #minor = getattr(p, "axis_minor_length", None) or p.minor_axis_length
    minor = p.minor_axis_length
    return float(major / max(minor, 1e-6))


def cutoff_percent(region: np.ndarray) -> float:
    """% of the particle's boundary pixels within 1 pixel of the ROI edge (as CPIview's cut-off)."""
    edge = region & ~ndimage.binary_erosion(region, structure=np.ones((3, 3), bool), border_value=0)
    rr, cc = np.nonzero(edge)
    if not len(rr):
        return float("nan")
    h, w = region.shape
    on_edge = (rr <= 1) | (rr >= h - 2) | (cc <= 1) | (cc >= w - 2)
    return 100.0 * float(np.mean(on_edge))


def _fill_bays(region: np.ndarray, enclosed: float, shadow: Optional[np.ndarray] = None,
               max_fraction: float = 0.08, shadow_cover: float = 0.0) -> np.ndarray:
    """Fill concavities ("bays") of a non-round particle that its outline nearly encloses.

    A bright spot focused by the particle (e.g. at the end of a transparent column)
    can open a notch through a thin gap in the shadow.  Each pixel group between the
    region and its convex hull is filled if at least ``enclosed`` of the pixels
    bordering it belong to the region, i.e. the opening to the outside is narrow.
    Wide concavities -- between the arms of a rosette, say -- face the outside
    along much of their rim and are kept, as is any bay larger than ``max_fraction``
    of the region.  ``shadow`` (the particle's whole weak shadow) also counts as rim.
    """
    from skimage.morphology import convex_hull_image
    hull = convex_hull_image(region)
    bays, n = ndimage.label(hull & ~region)
    if n == 0:
        return region
    out = region.copy()
    outside = ~hull
    limit = max_fraction * float(region.sum())
    for k in range(1, n + 1):
        bay = bays == k
        rim = ndimage.binary_dilation(bay) & ~bay
        # the open side of a bay borders the hull edge, i.e. pixels just outside the hull
        open_side = ndimage.binary_dilation(bay) & outside
        wall = region if shadow is None else (region | shadow)
        touching = np.count_nonzero(rim & wall)
        total = touching + np.count_nonzero(open_side)
        if total and touching / total >= enclosed and bay.sum() <= limit:
            out |= bay
        elif shadow is not None and shadow_cover > 0 and np.mean(shadow[bay]) >= shadow_cover:
            out |= bay          # lighter than the edge level, but still this particle's shadow
    return out


def _join_column_walls(T, labels, main, options):
    """Join the two walls of a transparent column when its end caps are too faint to connect them.

    Far from focus a column can image as two parallel dark bars with a bright gap and only
    faint grey end caps, so its shadow is two components.  Another shadow component is joined
    to ``main`` if (1) the two connect at a fainter level (T < 1 - join_faint * min_contrast)
    and (2) both are elongated, parallel, of similar length and side by side.  Two separate
    needles have background between them and do not connect at the fainter level.
    """
    from skimage.measure import regionprops
    faint, _ = ndimage.label(T < 1.0 - options.join_faint * options.min_contrast)
    ids = faint[main]
    if not ids.any():
        return main
    fid = int(np.bincount(ids[ids > 0]).argmax())
    others = [k for k in np.unique(labels[(faint == fid) & ~main]) if k]
    if not others:
        return main

    def shape(mask):
        p = regionprops(mask.astype(np.uint8))[0]
        major = getattr(p, "axis_major_length", None) or p.major_axis_length
        minor = getattr(p, "axis_minor_length", None) or p.minor_axis_length
        return p, major, max(minor, 1e-6)

    pm, Lm, Wm = shape(main)
    if Lm / Wm < options.join_min_aspect:
        return main
    # skimage: orientation = angle between the row axis and the major axis -> (row, col) direction
    axis = np.array([math.cos(pm.orientation), math.sin(pm.orientation)])
    rm, cm = np.nonzero(main)
    proj_m = rm * axis[0] + cm * axis[1]
    out = main.copy()
    for k in others:
        part = labels == k
        if part.sum() < options.join_min_size * main.sum():
            continue
        pk, Lk, Wk = shape(part)
        dtheta = abs(math.degrees(pm.orientation - pk.orientation)) % 180.0
        dtheta = min(dtheta, 180.0 - dtheta)
        if Lk / Wk < options.join_min_aspect or dtheta > options.join_max_angle or min(Lk, Lm) / max(Lk, Lm) < 0.6:
            continue
        rk, ck = np.nonzero(part)
        proj_k = rk * axis[0] + ck * axis[1]
        overlap = min(proj_m.max(), proj_k.max()) - max(proj_m.min(), proj_k.min())
        if overlap >= 0.6 * min(np.ptp(proj_m), np.ptp(proj_k)):
            out |= part
    return out


def _merge_shadow_parts(labels, overlap, best, main, options):
    """All edge-level parts inside one particle's shadow, and the closing radius that joins them.

    ``overlap[k-1]``: pixels of edge-level component k inside the particle's shadow ``main``.
    Parts must lie mostly inside that shadow and not be specks; the gaps between them must be
    small compared with the particle (at most ``merge_max_gap`` of its length).  Returns
    (region, radius); radius 0 when there is nothing to join.
    """
    sizes = np.bincount(labels.ravel(), minlength=len(overlap) + 1)[1:]
    biggest = overlap.max()
    keep = [k + 1 for k in range(len(overlap))
            if overlap[k] >= max(options.merge_min_px, options.merge_min_fraction * biggest)
            and overlap[k] >= 0.8 * sizes[k]]
    region = labels == best
    if len(keep) < 2:
        return region, 0
    footprint = ndimage.binary_fill_holes(main)
    rows, cols = np.nonzero(footprint)
    extent = float(np.hypot(np.ptp(rows), np.ptp(cols))) + 1.0
    max_gap = options.merge_max_gap * extent
    needed = 0.0
    parts = sorted((k for k in keep if k != best), key=lambda k: -overlap[k - 1])
    for k in parts:
        gap = float(ndimage.distance_transform_edt(~region)[labels == k].min())
        if gap > max_gap:
            continue
        region = region | (labels == k)
        needed = max(needed, gap)
    return region, (int(math.ceil(needed / 2.0)) + 1 if needed > 0 else 0)


def measure_transmittance(image: np.ndarray, background: np.ndarray, options: EdgeOptions) -> Optional[dict]:
    """Geometry, focus and boundary of the main particle in one ROI, or None if there is none."""
    T = transmittance(image, background, options)
    # Detect the particle independently of the chosen sizing isophote. In
    # particular, a faint bead (T_min = 0.65) must not be rejected because
    # the old fixed-mode core test required ten pixels below T = 0.45.
    candidate = T < 1.0 - options.min_contrast
    labels, n = ndimage.label(candidate)
    if n == 0:
        return None
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    eligible = np.flatnonzero(counts[1:] >= options.min_dark_pixels) + 1
    if not len(eligible):
        return None
    valid = np.isin(labels, eligible)
    component_labels, nvalid = ndimage.label(valid)  # preserve disconnected candidates
    chosen = _pick_component(component_labels, nvalid, np.clip(1.0 - T, 0.0, None))
    main = component_labels == chosen
    if options.merge_parts and nvalid > 1:
        main = _join_column_walls(T, component_labels, main, options)
    if options.edge_mode == "relative":
        t_min = float(np.sort(T[main])[:3].mean())
        edge_level = 1.0 - options.edge_fraction * (1.0 - t_min)
    elif options.edge_mode == "fixed":
        edge_level = float(options.edge_level)
    else:
        raise ValueError(f"Unknown transmittance edge mode: {options.edge_mode}")
    # A threshold above the normalised ambient baseline is not a valid shadow
    # isophote. Do not silently fall back to another threshold.
    if not 0.0 < edge_level < 1.0:
        return None
    # A particle detected at T < 1-min_contrast may have no contour at the
    # requested level (e.g. an extremely faint particle at T=0.7). The
    # subsequent overlap test deliberately refuses unrelated dark neighbours.
    dark = T < edge_level
    labels, n = ndimage.label(dark)
    if n == 0:
        return None
    bridge_px = 0
    if main is not None:
        # the part(s) of the chosen particle's shadow that are darker than its edge level
        overlap = np.bincount(labels[main].ravel(), minlength=n + 1)[1:]
        best = int(np.argmax(overlap)) + 1 if overlap.any() else _pick_component(labels, n, np.clip(edge_level - T, 0, None))
        if not overlap.any():
            return None
        region = labels == best
        if options.merge_parts and overlap.any():
            region, bridge_px = _merge_shadow_parts(labels, overlap, best, main, options)
    else:
        best = _pick_component(labels, n, np.clip(edge_level - T, 0, None))
        region = labels == best
    if bridge_px > 0:
        # Transparent columns focus light into a bright streak along their axis, splitting the
        # shadow at the edge level into parallel bars.  Join them -- only over pixels of this
        # particle's own shadow, so separate particles (background in between) are never joined.
        # The space between the parts belongs to the particle when the shadow (one connected
        # component) nearly encloses it, so the footprint itself is closed by the same radius first.
        r = int(bridge_px)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r * r
        crop = slice(r + 1, -(r + 1))
        footprint = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(main, r + 1), structure=disk)[crop, crop])
        closed = ndimage.binary_closing(np.pad(region, r + 1), structure=disk)[crop, crop]
        region = region | (closed & footprint)
        main = main | footprint
    if options.close_px > 0 and main is not None:
        # A bright diffraction spot in the middle of a bead can open a notch to the outside,
        # which fill_holes cannot close.  Bridge it with a small closing, but only over pixels
        # that are still part of this particle's shadow (T < 1 - min_contrast, holes filled):
        # real concavities of a crystal show background there and are left alone.
        r = int(options.close_px)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        padded = np.pad(region, r + 1)
        padded = ndimage.binary_closing(padded, structure=(xx ** 2 + yy ** 2) <= r * r)
        closed = padded[r + 1:-(r + 1), r + 1:-(r + 1)]
        region = region | (closed & ndimage.binary_fill_holes(main))
    if options.fill_holes:
        region = ndimage.binary_fill_holes(region)
    if options.fill_bays > 0 and main is not None and region.sum() >= 20:
        # Transparent particles (split-and-merged shadows, or elongated columns): fill concavities
        # that are nearly enclosed, or that lie inside the particle's own (weak) shadow.
        elongated = _elongation(region) >= options.fill_bays_min_aspect
        if bridge_px > 0 or elongated:
            region = _fill_bays(region, options.fill_bays, ndimage.binary_fill_holes(main),
                                max_fraction=options.fill_bays_max_area,
                                shadow_cover=options.fill_bays_shadow_cover)
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
    radial = None
    if options.outline == "edge" and main is not None:
        footprint = ndimage.binary_fill_holes(main)
        if _mask_roundness(footprint) >= options.radial_min_roundness:
            radial = steepest_rise_outline(T, footprint)
            if radial is not None:
                poly = radial
    if options.outline == "radial":
        # only for compact particles: on thin arms or needle tips the smoothing would trim a pixel or two
        footprint = ndimage.binary_fill_holes(main) if main is not None else None
        if footprint is not None and _mask_roundness(footprint) >= options.radial_min_roundness:
            radial = radial_outline(T, footprint, edge_level, options.outline_smooth_px,
                                    edge_fraction=(options.edge_fraction if options.edge_mode == "relative" else None),
                                    max_level=1.0 - options.min_contrast / 2, check=region)
        if radial is None and _roundness(poly) >= options.radial_min_roundness:
            radial = radial_outline(T, region, edge_level, options.outline_smooth_px)
        if radial is not None:
            poly = radial
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
    minor = getattr(props, "axis_minor_length", None) or props.minor_axis_length
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
        "wid_ellipse": float(minor) * px,
        "len_feret": length_px * px,
        "wid_feret": width_px * px,
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
        "cutoff_pct": cutoff_percent(region),
        "boundary": boundary,
        "n_boundary": int(len(boundary) - 1),
        "outline_radial": int(radial is not None),
    }
