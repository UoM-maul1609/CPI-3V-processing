"""Signed-shadow integrated diameter within a fitted aperture.

Both modes treat unobserved regions outside the ROI as unobstructed light
(normalised transmittance T=1, contributing no signed shadow deficit).
Adaptive mode still requires sufficient observed annular coverage to avoid a
false plateau produced by the ROI boundary. The method does NOT produce a
geometric particle boundary.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

try:
    from .edgeDetect import EdgeOptions, transmittance
except ImportError:
    from edgeDetect import EdgeOptions, transmittance


def fit_circle(boundary):
    """Centre the shadow aperture on the filled contour's area centroid.

    The radius encloses every contour vertex, which is preferable to a freely
    optimised least-squares circle for elongated or incomplete ice analogues.
    Coordinates are (row, column). Return (centre, radius, radial RMS).
    """
    b = np.asarray(boundary, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("Missing valid stored boundary to centre the shadow aperture")
    b = b[np.isfinite(b).all(axis=1)]
    if len(b) > 3 and np.allclose(b[0], b[-1]):
        b = b[:-1]
    if len(b) < 5:
        raise ValueError("Contour has fewer than five usable points")
    # Shoelace polygon centroid. Works for either clockwise or anticlockwise
    # outlines, and does not bias the centre toward oversampled edge sections.
    x, y = b[:, 1], b[:, 0]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    twice_area = float(np.sum(cross))
    if abs(twice_area) > 1e-8:
        cx = float(np.sum((x + np.roll(x, -1)) * cross) / (3 * twice_area))
        cy = float(np.sum((y + np.roll(y, -1)) * cross) / (3 * twice_area))
    else:
        cy, cx = map(float, np.mean(b, axis=0))
    centre = (cy, cx)
    distances = np.linalg.norm(b - np.array(centre), axis=1)
    radius = float(np.max(distances))
    if not np.isfinite(radius) or radius < 1:
        raise ValueError("Contour too small for an enclosing circle")
    rms = float(np.sqrt(np.mean((distances - radius) ** 2)))
    return centre, radius, rms


def _fractional_disk(shape, centre, radius, supersample=4):
    h, w = shape
    cy, cx = centre
    y0 = max(0, int(np.floor(cy-radius-1)))
    y1 = min(h, int(np.ceil(cy+radius+2)))
    x0 = max(0, int(np.floor(cx-radius-1)))
    x1 = min(w, int(np.ceil(cx+radius+2)))
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Circle outside ROI")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = np.zeros_like(yy, float)
    for iy in range(supersample):
        for ix in range(supersample):
            dy, dx = (iy+.5)/supersample-.5, (ix+.5)/supersample-.5
            mask += ((yy+dy-cy)**2 + (xx+dx-cx)**2 <= radius**2)
    return (slice(y0, y1), slice(x0, x1)), mask/(supersample**2)


def cumulative_shadow(T, centre, max_radius, *, subpixel=4):
    """Per-pixel-radius cumulative signed deficit, with annulus ROI coverage.

    Outside-ROI pixels are defined as T=1: missing parts of an annulus
    contribute exactly zero deficit (no symmetry extrapolation). Coverage is
    returned separately to assess reliability of apparent plateaus.
    """
    h, w = T.shape
    cy, cx = centre
    n = int(np.ceil(max_radius)) + 1
    y0 = max(0, int(np.floor(cy-max_radius-1)))
    y1 = min(h, int(np.ceil(cy+max_radius+2)))
    x0 = max(0, int(np.floor(cx-max_radius-1)))
    x1 = min(w, int(np.ceil(cx+max_radius+2)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    deficit = (1. - T[y0:y1, x0:x1]).ravel()
    total = np.zeros(n, float)
    seen = np.zeros(n, float)
    for iy in range(subpixel):
        for ix in range(subpixel):
            dy, dx = (iy+.5)/subpixel-.5, (ix+.5)/subpixel-.5
            bins = np.floor(np.hypot(yy+dy-cy, xx+dx-cx)).astype(int).ravel()
            total += np.bincount(bins, weights=deficit, minlength=n)[:n] / subpixel**2
            seen += np.bincount(bins, minlength=n)[:n] / subpixel**2
    radii = np.arange(n, dtype=float)
    full = np.pi * (2*radii + 1)
    coverage = np.minimum(1., seen / full)
    # Missing pixels are equivalent to T=1, hence (1-T)=0.  Do not scale
    # the observed deficit up by the fraction of the ring that was seen.
    return radii + 1, np.cumsum(total), coverage, total


def _adaptive_radius(T, centre, seed_radius, max_padding, *, plateau_fraction=.015,
                     min_coverage=.85, window=4):
    if max_padding < 6 or max_padding > 60:
        raise ValueError("Adaptive maximum padding must be between 6 and 60 pixels")
    radii, areas, coverage, _ = cumulative_shadow(T, centre, seed_radius + max_padding)
    valid_limit = min(len(radii), int(np.floor(seed_radius + max_padding)))
    start = max(int(np.ceil(seed_radius + 3)), 2*window+1)
    for i in range(start, valid_limit):
        # Missing areas contribute zero, which can create an artificial plateau
        # at the ROI edge.  Refuse to infer *convergence* from poorly sampled
        # annuli even though fixed-radius integration can still return a value.
        if np.min(coverage[:i+1]) < min_coverage:
            raise ValueError("Aperture cropped: insufficient annulus coverage to establish plateau")
        # Both the last 4-pixel interval AND the preceding interval must be stable.
        past = areas[i-2*window:i+1]
        ref = float(np.median(past))
        if ref <= 0 or not np.isfinite(ref):
            continue
        tolerance = max(plateau_fraction * abs(ref), .05)
        if np.ptp(past) <= tolerance:
            # Exact disk integration at the selected radius for parity with fixed mode.
            return float(radii[i]), {"plateau_fraction_observed": float(np.ptp(past)/abs(ref)),
                                      "coverage_min": float(np.min(coverage[:i+1])),
                                      "radius_series_px": radii[:i+1],
                                      "area_series_px": areas[:i+1]}
    raise ValueError("No sustained shadow-integral plateau before ROI edge or maximum radius")


def measure(image, background, boundary, *, padding_px=8., pixel_um=2.3,
            denoise=1, smooth_sigma=0., radius_mode="fixed", max_padding_px=30.,
            plateau_fraction=.015):
    """Signed-shadow size and drawn aperture; no calibration is applied.

    Missing parts of an integration circle have T=1 and contribute no deficit.
    The returned coverage quantifies how much of the circle was actually seen.
    Adaptive selection still requires sufficient coverage for a trustworthy
    plateau; it does not extrapolate missing annuli.
    """
    im, bg = np.asarray(image), np.asarray(background)
    if im.ndim != 2 or im.shape != bg.shape or min(im.shape) < 4:
        raise ValueError("Missing ROI or matching background")
    if not (np.isfinite(im).all() and np.isfinite(bg).all()):
        raise ValueError("Nonfinite image/background")
    if not 0 <= float(padding_px) <= 30:
        raise ValueError("Aperture padding must be in [0, 30] px")
    centre, seed_r, rms = fit_circle(boundary)
    opts = EdgeOptions(method="transmittance", pixel_um=float(pixel_um),
                       denoise=int(denoise), smooth_sigma=float(smooth_sigma),
                       border_normalise=True, bg_check=False)
    T = transmittance(im, bg, opts)
    if radius_mode == "fixed":
        radius = seed_r + float(padding_px)
        extras = {"radius_mode": "fixed", "plateau_fraction_observed": float("nan")}
    elif radius_mode == "adaptive":
        radius, extras = _adaptive_radius(T, centre, seed_r, float(max_padding_px),
                                         plateau_fraction=float(plateau_fraction))
        extras["radius_mode"] = "adaptive"
    else:
        raise ValueError("Unknown circle-radius mode")
    crop, mask = _fractional_disk(im.shape, centre, radius)
    # By definition an unobserved pixel outside the ROI has T=1: its signed
    # deficit (1-T) is zero. Summing only observed disk pixels is *identical*
    # to inserting T=1 outside the ROI, and avoids allocating padded images.
    area_px = float(np.sum((1.-T[crop])*mask))
    observed_circle_px = float(np.sum(mask))
    theoretical_circle_px = float(np.pi * radius**2)
    observed_fraction = float(np.clip(observed_circle_px/theoretical_circle_px, 0., 1.))
    missing_fraction = 1.-observed_fraction
    # The 4x subpixel area approximation can differ slightly from pi*r^2;
    # use a geometric border check to avoid marking interior circles as cropped.
    cy, cx = centre
    h, w = im.shape
    clipped = bool(cy-radius < -.5 or cy+radius > h-.5 or
                   cx-radius < -.5 or cx+radius > w-.5)
    extras.update({
        "aperture_coverage": observed_fraction,
        "aperture_missing_fraction": missing_fraction,
        "roi_truncated": clipped,
        "outside_transmittance": 1.,
        "roi_extrapolated": False,
        "coverage_min": min(extras.get("coverage_min", 1.), observed_fraction),
    })
    if not np.isfinite(area_px) or area_px <= 0:
        raise ValueError("Nonpositive signed shadow area")
    phi = np.linspace(0, 2*np.pi, 129)
    overlay = np.column_stack([cy+radius*np.sin(phi), cx+radius*np.cos(phi)])
    return {"shadow_diameter": float(2*np.sqrt(area_px/np.pi)*float(pixel_um)),
            "shadow_area_px": area_px, "aperture_radius_px": float(radius),
            "seed_radius_px": float(seed_r), "circle_rms_px": rms,
            "aperture_centre": centre, "boundary": overlay,
            "overlay_kind": "integration circle", "image_shape": im.shape,
            "T_border": float(np.median(np.r_[T[0], T[-1], T[:, 0], T[:, -1]])),
            **extras}


def local_illumination(image, background, boundary):
    """Contour-local brightness from stored background scaled by image/BG ratio."""
    from scipy.ndimage import map_coordinates
    im, bg = np.asarray(image, float), np.asarray(background, float)
    b = np.asarray(boundary, float)
    if im.ndim != 2 or im.shape != bg.shape or b.ndim != 2 or b.shape[1] != 2:
        return float("nan")
    good = np.isfinite(bg) & (bg > 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(good, im/bg, np.nan)
    border = np.r_[q[0], q[-1], q[:, 0], q[:, -1]]
    border = border[np.isfinite(border) & (border > 0)]
    if len(border) < 8:
        return float("nan")
    scale = float(np.median(border))
    b = b[np.isfinite(b).all(axis=1)]
    if len(b) < 3:
        return float("nan")
    positions = b.T
    sampled_bg = map_coordinates(bg, positions, order=1, mode="nearest")
    sampled_bg = sampled_bg[np.isfinite(sampled_bg) & (sampled_bg > 1)]
    return scale*float(np.median(sampled_bg)) if len(sampled_bg) else float("nan")
