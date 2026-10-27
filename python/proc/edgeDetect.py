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

EDGE_CODE_VERSION = "2026.10.26.union-subpixel-guard"


@dataclass
class EdgeOptions:
    method: str = "legacy"             # "legacy" or "transmittance"
    pixel_um: float = 2.3
    # transmittance method
    edge_mode: str = "fixed"           # "fixed": edge at T = edge_level;
                                       # "relative": edge at T = 1 - edge_fraction*(1 - T_min), i.e. a fixed
                                       # fraction of the particle's own shadow depth (keeps faint, small,
                                       # defocused particles detectable and measured consistently)
    count_threshold: float = 41.0    # scaled-background minus ROI, raw 8-bit intensity counts
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
    union_min_component_area_px: int = 10  # absolute minimum area for an additional union fragment
    union_require_weak_shadow: bool = True  # require a connecting path through a common faint shadow
    union_max_separation_px: float = 6.0  # max edge-to-edge separation of threshold components in raw-contour union
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
                                       # "contour": the raw iso-transmittance contour;
                                       # "union": join threshold-dark components belonging to the same
                                       # weak-shadow connected region; preserves genuine concavities
    inset_preserve_necks: bool = False  # union: protect narrow connectors during inset (experimental)
    local_close_radius_px: float = 0.0  # optional post-inset local binary closing radius; 0 disables
    local_close_max_added_fraction: float = 0.03  # maximum total added area relative to post-inset mask
    outline_inset_px: float = 0.0      # optional inward offset applied AFTER the outline is found; useful when
                                       # a higher transmittance threshold follows the correct shape but sits
                                       # systematically too far outside the geometric edge.
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


def transmission_integrals(image: np.ndarray, background: np.ndarray,
                           *, border_normalise: bool = True) -> dict:
    """Full-ROI *pixel sums* of raw background-corrected transmission.

    T = I/B, optionally scaled to make median ROI-border T=1, WITHOUT
    smoothing, clipping or segmentation. All four values use the same ROI.
    They are not contour areas or physical shadow areas.
    """
    opts = EdgeOptions(denoise=0, smooth_sigma=0.0,
                       border_normalise=border_normalise)
    T = transmittance(np.asarray(image), np.asarray(background), opts)
    finite = T[np.isfinite(T)]
    if finite.size == 0:
        return dict(T_peak=float('nan'), T_min_raw=float('nan'),
                    T_excess_sum=float('nan'), T_deficit_sum=float('nan'),
                    T_signed_sum=float('nan'))
    above = float(np.maximum(finite - 1.0, 0.0).sum())
    below = float(np.maximum(1.0 - finite, 0.0).sum())
    return dict(T_peak=float(finite.max()), T_min_raw=float(finite.min()),
                T_excess_sum=above, T_deficit_sum=below,
                T_signed_sum=below - above)


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
    major = p.axis_major_length
    minor = p.axis_minor_length
    return float(major / max(minor, 1e-6))


def _inset_polygon(poly: np.ndarray, region: np.ndarray, inset_px: float):
    """Inset the FINAL measured contour (not the preliminary threshold mask).

    Rasterise the selected outline, take its inside Euclidean distance field and
    extract a subpixel iso-distance contour. This handles concavities and differs
    from merely reducing the reported length by 2*inset_px. An excessively large
    inset yields an explicit failed measurement rather than an invented contour.
    """
    inset_px = float(inset_px)
    if not np.isfinite(inset_px) or inset_px < 0:
        raise ValueError("outline_inset_px must be finite and non-negative")
    if inset_px == 0:
        return poly, region
    mask = np.zeros(region.shape, dtype=bool)
    rr, cc = draw.polygon(poly[:, 0], poly[:, 1], shape=mask.shape)
    mask[rr, cc] = True
    if not mask.any():
        return None, None
    distance = ndimage.distance_transform_edt(mask)
    if distance.max() <= inset_px:
        return None, None
    # Pixel-centre signed distance places the 0-level near the original
    # boundary; half-pixel compensation limits systematic rasterising bias.
    outside = ndimage.distance_transform_edt(~mask)
    signed = np.where(mask, distance - 0.5, 0.5 - outside)
    padded = np.pad(signed, 1, constant_values=-10000.)
    contours = measure.find_contours(padded, inset_px)
    if not contours:
        return None, None
    candidates = [c - 1.0 for c in contours if len(c) >= 4]
    if not candidates:
        return None, None
    newpoly = max(candidates, key=polygon_area)
    if len(newpoly) > 1 and np.allclose(newpoly[0], newpoly[-1]):
        newpoly = newpoly[:-1]
    newregion = np.zeros(mask.shape, dtype=bool)
    rr, cc = draw.polygon(newpoly[:, 0], newpoly[:, 1], shape=mask.shape)
    newregion[rr, cc] = True
    if newregion.sum() < 3:
        return None, None
    return newpoly, newregion


def _inset_preserving_necks(poly: np.ndarray, region: np.ndarray, inset_px: float):
    """Inset while retaining the original connected shape's thin branches.

    Thin one-pixel connections are protected by the (8-connected) medial
    skeleton plus an ~one-pixel radius 4-connected corridor. This is deliberately
    a *local exception* to the requested inset, not a uniform offset: a narrow
    neck may be inset less than requested. Original pixels are never added from
    outside the pre-inset polygon. Return a single contour or fail explicitly.
    """
    from skimage.morphology import skeletonize
    mask = np.zeros(region.shape, dtype=bool)
    rr, cc = draw.polygon(poly[:, 0], poly[:, 1], shape=mask.shape)
    mask[rr, cc] = True
    if not mask.any():
        return None, None, 0
    # Preserve connectivity of the pre-inset silhouette, but avoid altering
    # regions where the ordinary inset already retains the entire particle.
    normal_poly, normal_mask = _inset_polygon(poly, region, inset_px)
    if normal_mask is None:
        normal_mask = np.zeros_like(mask)
    # Eight-connectivity is used consistently for topology. Also safeguard
    # appendages whose tips would otherwise vanish despite one connected island.
    skeleton = skeletonize(mask)
    # Every protected pixel lies within the original silhouette. The cross
    # dilation supports diagonal links without an artificial line outside it.
    protected = ndimage.binary_dilation(skeleton, structure=ndimage.generate_binary_structure(2, 1)) & mask
    combined = normal_mask | protected
    # The protected skeleton should connect the original mask. If a mask has
    # disconnected islands already, do not invent bridges in this step.
    n_initial = ndimage.label(mask, structure=np.ones((3, 3), bool))[1]
    n_final = ndimage.label(combined, structure=np.ones((3, 3), bool))[1]
    if n_initial != 1 or n_final != 1:
        return None, None, 0
    added = int(np.count_nonzero(protected & ~normal_mask))
    # Trace the exterior of the pixel mask; unlike the ordinary offset, this
    # must encompass the protected narrow neck and therefore is not a strict
    # constant-distance isocontour there.
    lines = measure.find_contours(np.pad(combined.astype(float), 1), .5)
    if not lines:
        return None, None, added
    newpoly = max(lines, key=polygon_area) - 1.0
    if len(newpoly) > 1 and np.allclose(newpoly[0], newpoly[-1]):
        newpoly = newpoly[:-1]
    return newpoly, combined, added


def _local_post_close(poly: np.ndarray, shape, radius: float, max_added_fraction: float):
    """Conservative local closing AFTER inset, never a global hull.

    A disk closing proposes narrow bay/slot repairs; each added connected
    patch must fit the remaining total area budget. Preserve the original
    mask everywhere, and do not expand beyond the original mask's convex hull.
    Returns (contour, final_mask, added_pixels, proposals_rejected).
    """
    original = np.zeros(shape, bool)
    rr, cc = draw.polygon(poly[:, 0], poly[:, 1], shape=shape)
    original[rr, cc] = True
    if radius <= 0 or not original.any():
        return poly, original, 0, 0
    from skimage.morphology import convex_hull_image
    r = max(1, int(math.ceil(radius)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = xx * xx + yy * yy <= radius * radius + 1.e-6
    pad = r + 2
    expanded = ndimage.binary_closing(np.pad(original, pad), structure=disk)[pad:-pad, pad:-pad]
    candidate = expanded & convex_hull_image(original) & ~original
    parts, n = ndimage.label(candidate, structure=np.ones((3, 3), bool))
    budget = int(math.floor(original.sum() * max_added_fraction))
    used, rejected = 0, 0
    out = original.copy()
    for k in sorted(range(1, n + 1), key=lambda k: np.count_nonzero(parts == k)):
        part = parts == k
        count = int(part.sum())
        if count <= budget - used:
            out |= part
            used += count
        else:
            rejected += 1
    if not used:
        return poly, original, 0, rejected
    contours = measure.find_contours(np.pad(out.astype(float), 1), .5)
    if not contours:
        return poly, original, 0, rejected
    contour = max(contours, key=polygon_area) - 1
    if len(contour) > 1 and np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    return contour, out, used, rejected


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
        major = p.axis_major_length
        minor = p.axis_minor_length
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


def _union_shadow_components(labels, main, best, level_field, T, options):
    """Join selected threshold fragments using explicit absolute area and gap limits.

    Gap is Euclidean distance between pixel centres, recomputed after each join.
    A join is either an 8-connected weak-shadow route or a direct digital line.
    Neither option uses a convex hull, closing, or filling concavities.
    """
    from scipy.spatial import cKDTree
    from skimage.graph import route_through_array
    from skimage.draw import line

    region = labels == best
    connection_mask = np.zeros_like(region)
    sizes = np.bincount(labels.ravel())
    gap_limit = float(options.union_max_separation_px)
    if not np.isfinite(gap_limit) or not 0 <= gap_limit <= 50:
        raise ValueError("union_max_separation_px must be between 0 and 50")
    min_area = int(options.union_min_component_area_px)
    if not 1 <= min_area <= 10000:
        raise ValueError("union_min_component_area_px must be between 1 and 10000")
    require_shadow = bool(options.union_require_weak_shadow)
    support_level = 1.0 - max(0.02, 0.25 * options.min_contrast)
    support = T < support_level
    foot_labels, _ = ndimage.label(support, structure=np.ones((3, 3), bool))
    ids = foot_labels[main]
    ids = ids[ids > 0]
    parent = foot_labels == int(np.bincount(ids).argmax()) if ids.size else np.zeros_like(region)

    detail = {}
    for k in range(1, len(sizes)):
        if k == best or sizes[k] == 0:
            continue
        part = labels == k
        fraction = float(np.count_nonzero(part & parent)) / float(sizes[k])
        detail[k] = {"component": int(k), "area_px": int(sizes[k]),
                     "weak_shadow_overlap": round(fraction, 3), "gap_px": None,
                     "status": "pending"}
    diagnostics = dict(union_components_found=int(np.count_nonzero(sizes[1:])),
                       union_components_joined=0, union_components_considered=0,
                       union_max_gap_used_px=0.0, union_require_weak_shadow=require_shadow)
    pending = set(detail)
    while pending:
        points_a = np.column_stack(np.nonzero(region))
        tree = cKDTree(points_a)
        neighbours = []
        for k in pending:
            points_b = np.column_stack(np.nonzero(labels == k))
            dists, nearest = tree.query(points_b, k=1)
            j = int(np.argmin(dists))
            gap = float(dists[j])
            detail[k]["gap_px"] = round(gap, 2)
            neighbours.append((gap, k, tuple(points_a[nearest[j]]), tuple(points_b[j])))
        changed = False
        for gap, k, a, b in sorted(neighbours):
            entry = detail[k]
            if entry["area_px"] < options.union_min_component_area_px:
                entry["status"] = "too small"
                continue
            if gap > gap_limit:
                entry["status"] = "gap too large"
                continue
            if require_shadow and entry["weak_shadow_overlap"] < .8:
                entry["status"] = "outside main weak shadow"
                continue
            rr, cc = line(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            if require_shadow:
                pad = max(2, int(np.ceil(gap_limit)))
                r0 = max(0, min(a[0], b[0]) - pad)
                c0 = max(0, min(a[1], b[1]) - pad)
                r1 = min(parent.shape[0], max(a[0], b[0]) + pad + 1)
                c1 = min(parent.shape[1], max(a[1], b[1]) + pad + 1)
                costs = np.where(parent[r0:r1, c0:c1], 1.0, np.inf)
                try:
                    path, _ = route_through_array(costs, (a[0]-r0, a[1]-c0),
                                                  (b[0]-r0, b[1]-c0),
                                                  fully_connected=True, geometric=True)
                    if len(path)-1 > max(1., 2.*gap_limit) or not np.isfinite(costs[tuple(np.array(path).T)]).all():
                        entry["status"] = "no short weak-shadow route"
                        continue
                    rr = np.array([r0+point[0] for point in path], dtype=int)
                    cc = np.array([c0+point[1] for point in path], dtype=int)
                except (ValueError, IndexError):
                    entry["status"] = "no weak-shadow route"
                    continue
            connection_mask[rr, cc] = True
            region[rr, cc] = True
            region |= labels == k
            entry["status"] = "joined"
            pending.remove(k)
            diagnostics["union_components_joined"] += 1
            diagnostics["union_max_gap_used_px"] = max(diagnostics["union_max_gap_used_px"], gap)
            changed = True
            break
        if not changed:
            break
    # Re-evaluate each rejection against the final union; all candidates receive a reason.
    diagnostics["union_components_considered"] = sum(x["area_px"] >= options.union_min_component_area_px
                                                  and (not require_shadow or x["weak_shadow_overlap"] >= .8)
                                                  for x in detail.values())
    diagnostics["union_component_details"] = sorted(detail.values(), key=lambda x: x["component"])
    diagnostics["_union_connections"] = connection_mask
    diagnostics["_union_selected_labels"] = tuple([int(best)] + [int(k) for k, v in detail.items() if v["status"] == "joined"])
    return region, diagnostics


def measure_transmittance(image: np.ndarray, background: np.ndarray, options: EdgeOptions) -> Optional[dict]:
    """Geometry, focus and boundary of the main particle in one ROI, or None if there is none."""
    T = transmittance(image, background, options)
    # Count-threshold mode: scale the background using exactly the same border
    # clear-field estimate as transmittance(), then compare on the raw 8-bit
    # intensity scale. Never cast signed differences to uint8.
    count_mode = options.edge_mode == "scaled_counts"
    if count_mode:
        bg_float = np.maximum(np.asarray(background, dtype=float), 1.0)
        ratio = np.asarray(image, dtype=float) / bg_float
        if options.border_normalise and min(ratio.shape) > 4:
            border = np.concatenate([ratio[0], ratio[-1], ratio[1:-1, 0], ratio[1:-1, -1]])
            scale = float(np.median(border))
            if not np.isfinite(scale) or scale <= 0.2:
                scale = 1.0
        else:
            scale = 1.0
        # Pseudo-transmission has its isophote at 1 - threshold/255.
        # This preserves the existing contour and radial outline machinery.
        level_field = 1.0 - (scale * bg_float - np.asarray(image, dtype=float)) / 255.0
        for _ in range(max(0, int(options.denoise))):
            level_field = ndimage.median_filter(level_field, size=3, mode="nearest")
        if options.smooth_sigma > 0:
            level_field = ndimage.gaussian_filter(level_field, options.smooth_sigma)
    else:
        level_field = T
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
    if options.merge_parts and options.outline != "union" and nvalid > 1:
        main = _join_column_walls(T, component_labels, main, options)
    if options.edge_mode == "relative":
        t_min = float(np.sort(T[main])[:3].mean())
        edge_level = 1.0 - options.edge_fraction * (1.0 - t_min)
    elif options.edge_mode == "fixed":
        edge_level = float(options.edge_level)
    elif count_mode:
        edge_level = 1.0 - float(options.count_threshold) / 255.0
    else:
        raise ValueError(f"Unknown transmittance edge mode: {options.edge_mode}")
    # A threshold above the normalised ambient baseline is not a valid shadow
    # isophote. Do not silently fall back to another threshold.
    if not 0.0 < edge_level < 1.0:
        return None
    # A particle detected at T < 1-min_contrast may have no contour at the
    # requested level (e.g. an extremely faint particle at T=0.7). The
    # subsequent overlap test deliberately refuses unrelated dark neighbours.
    dark = level_field < edge_level
    labels, n = ndimage.label(dark)
    if n == 0:
        return None
    bridge_px = 0
    if main is not None:
        # the part(s) of the chosen particle's shadow that are darker than its edge level
        overlap = np.bincount(labels[main].ravel(), minlength=n + 1)[1:]
        best = int(np.argmax(overlap)) + 1 if overlap.any() else _pick_component(labels, n, np.clip(edge_level - level_field, 0, None))
        if not overlap.any():
            return None
        region = labels == best
        if options.merge_parts and options.outline != "union" and overlap.any():
            region, bridge_px = _merge_shadow_parts(labels, overlap, best, main, options)
    else:
        best = _pick_component(labels, n, np.clip(edge_level - level_field, 0, None))
        region = labels == best
    union_diagnostics = {}
    if options.outline == "union":
        region, union_diagnostics = _union_shadow_components(labels, main, best, level_field, T, options)
        bridge_px = 0
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
    if options.close_px > 0 and main is not None and options.outline != "union":
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
    if options.fill_holes and options.outline != "union":
        region = ndimage.binary_fill_holes(region)
    if options.fill_bays > 0 and main is not None and region.sum() >= 20 and options.outline != "union":
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
    field_ = np.where(grown, level_field, bright)
    field_ = np.where(region & ~(level_field < edge_level), np.minimum(field_, edge_level - 0.05), field_)
    padded = np.pad(field_, 1, mode="constant", constant_values=bright)
    if options.outline == "union":
        # Preserve the ORIGINAL subpixel threshold isophote on each fragment.
        # Only the new, shortest-path connecting corridor is synthetic. Give it
        # sufficient width to survive an optional later inset; otherwise a
        # one-pixel bridge disappears and the measured polygon drops fragments.
        corridor = union_diagnostics.pop("_union_connections")
        selected = union_diagnostics.pop("_union_selected_labels")
        radius = max(1, int(math.ceil(options.outline_inset_px + 1.5)))
        if corridor.any():
            y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
            corridor = ndimage.binary_dilation(corridor, structure=(x*x+y*y <= radius*radius))
        # The original threshold field remains unmodified everywhere except
        # the artificial connector, whose interior is assigned a dark value.
        field_union = padded.copy()
        core = region & ~corridor
        field_union[1:-1, 1:-1][corridor] = np.minimum(
            field_union[1:-1, 1:-1][corridor], edge_level - 0.1)
        contours = measure.find_contours(field_union, edge_level)
        union_diagnostics["union_connector_pixels"] = int(corridor.sum())
        union_diagnostics["union_selected_components"] = len(selected)
        union_diagnostics["union_connection_xy"] = np.column_stack(np.nonzero(corridor)).astype(int).tolist()
        union_selected_masks = [labels == k for k in selected]
        union_diagnostics["union_original_contours"] = [
            (curve - 1.0).tolist()
            for k in selected
            for curve in measure.find_contours(
                np.pad((labels == k).astype(float), 1), 0.5)
            if len(curve) >= 4
        ]
    else:
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
            radial = steepest_rise_outline(level_field, footprint)
            if radial is not None:
                poly = radial
    if options.outline == "radial":
        # only for compact particles: on thin arms or needle tips the smoothing would trim a pixel or two
        footprint = ndimage.binary_fill_holes(main) if main is not None else None
        if footprint is not None and _mask_roundness(footprint) >= options.radial_min_roundness:
            radial = radial_outline(level_field, footprint, edge_level, options.outline_smooth_px,
                                    edge_fraction=(options.edge_fraction if options.edge_mode == "relative" else None),
                                    max_level=1.0 - options.min_contrast / 2, check=region)
        if radial is None and _roundness(poly) >= options.radial_min_roundness:
            radial = radial_outline(level_field, region, edge_level, options.outline_smooth_px)
        if radial is not None:
            poly = radial
    if options.outline == "union":
        # An inset may shrink fine fragments to nothing. Explicitly report
        # final retention instead of silently claiming the union succeeded.
        union_diagnostics["union_components_in_final_outline"] = 0
    if options.outline_inset_px > 0:
        if options.outline == "union" and options.inset_preserve_necks:
            poly, inset_mask, protected_px = _inset_preserving_necks(poly, region, options.outline_inset_px)
            union_diagnostics["union_inset_protected_pixels"] = protected_px
        else:
            poly, inset_mask = _inset_polygon(poly, region, options.outline_inset_px)
        if poly is None or inset_mask is None:
            return None
        region = inset_mask
    if len(poly) < 4:
        return None
    if options.outline == "union" and options.local_close_radius_px > 0:
        if not 0 <= options.local_close_radius_px <= 10:
            raise ValueError("local_close_radius_px must be between 0 and 10")
        if not 0 <= options.local_close_max_added_fraction <= .25:
            raise ValueError("local_close_max_added_fraction must be between 0 and 0.25")
        poly, region, added, rejected = _local_post_close(
            poly, region.shape, options.local_close_radius_px,
            options.local_close_max_added_fraction)
        union_diagnostics["local_close_added_px"] = int(added)
        union_diagnostics["local_close_rejected_patches"] = int(rejected)
        union_diagnostics["local_close_added_pct"] = 100. * added / max(1., region.sum() - added)
    if options.outline == "union":
        # Verify representation in the actual final measured polygon, not just
        # in the intermediary joined segmentation. Mark failures unambiguously.
        final_mask = np.zeros(region.shape, dtype=bool)
        rr, cc = draw.polygon(poly[:, 0], poly[:, 1], shape=final_mask.shape)
        final_mask[rr, cc] = True
        # A fragment is retained if its eroded interior contributes pixels;
        # single-pixel tips may disappear legitimately under an inset.
        overlap = [int(np.count_nonzero(final_mask & m)) for m in union_selected_masks]
        retained = sum(a > 0 for a in overlap)
        union_diagnostics["union_components_in_final_outline"] = int(retained)
        union_diagnostics["union_final_fragment_pixels"] = overlap
        union_diagnostics["union_final_retention_ok"] = retained == len(union_selected_masks)
    px = options.pixel_um
    length_px, width_px, angle = feret(poly)
    area_px = polygon_area(poly)
    perim_px = polygon_perimeter(poly)
    rows, cols = np.nonzero(region)
    # ellipse major axis of the same region (for comparison with legacy)
    props = measure.regionprops(region.astype(np.uint8))[0]
    major = props.axis_major_length
    minor = props.axis_minor_length
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
        **transmission_integrals(image, background, border_normalise=options.border_normalise),
        "T_min": float(T[region].min()),
        "edge_level_used": float(options.count_threshold if count_mode else edge_level),
        "T_centre": centre_T,
        "T_border": float(np.median(np.concatenate([T[0], T[-1], T[:, 0], T[:, -1]]))),
        # 1 if the particle reaches the ROI border: it may be cut off, so its size is a lower limit
        "edge_touch": int(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()),
        "cutoff_pct": cutoff_percent(region),
        "boundary": boundary,
        "n_boundary": int(len(boundary) - 1),
        **union_diagnostics,
        "outline_radial": int(radial is not None),
    }
