"""Exploratory optical-interior diagnostics for CPI particles.

These are measured from normalised transmittance *inside the saved outer
particle outline*. Bright optical regions are not necessarily geometrical
holes. An empty/invalid measurement returns {} (stored as NaN), not zeros.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.draw import polygon as raster_polygon

INTERIOR_FIELD_NAMES = (
    "interior_bright_fraction", "interior_bright_threshold",
    "interior_near_background_fraction", "interior_above_one_fraction",
    "interior_enclosed_bright_fraction", "interior_mean_T",
    "interior_effective_deficit", "interior_pixel_count",
)
# MATLAB v5 struct field names are limited to 31 chars by default in SciPy.
# Map the two longer diagnostic names to compact MAT names, retaining full
# user-facing and NetCDF names everywhere else.
MAT_FIELD_NAMES = {
    "interior_near_background_fraction": "interior_near_bg_fraction",
    "interior_enclosed_bright_fraction": "interior_enclosed_fraction",
}


def interior_light_metrics(transmittance, boundary, *, bright_level=None):
    """Experimental bright-inside-contour indicators (NOT geometrical void area).

    Fraction of the filled contour above the particle-relative edge isophote
    (default 55% of the local contrast), above 0.90 and above 1.0, plus
    mean signed transmittance.  The two-pixel
    eroded interior avoids subpixel contour-edge and interpolation artefacts.
    """
    T = np.asarray(transmittance, dtype=float)
    b = np.asarray(boundary if boundary is not None else [], dtype=float)
    if T.ndim != 2 or b.ndim != 2 or b.shape[1] != 2:
        return {}
    b = b[np.isfinite(b).all(axis=1)]
    if len(b) < 3:
        return {}
    rr, cc = raster_polygon(b[:, 0], b[:, 1], shape=T.shape)
    filled = np.zeros(T.shape, dtype=bool)
    filled[rr, cc] = True
    # Optical bright pixels on outer edge are not 'interior holes'.
    inside = ndimage.binary_erosion(filled, iterations=2)
    usable = inside & np.isfinite(T)
    count = int(np.count_nonzero(usable))
    if count < 4:
        return {}
    vals = T[usable]
    # Default threshold uses the *particle's own* shadow depth, as in the
    # relative-edge processor; T=0.90 alone would miss partially transparent
    # crystals with genuinely brighter internal regions.
    level = (float(bright_level) if bright_level is not None else
             1.0 - 0.55*(1.0 - float(np.percentile(vals, 2))))
    bright = usable & (T >= level)
    # Closed islands of bright material are distinguished from paths touching
    # the mask rim: those paths need not be a true interior hole.
    components, n = ndimage.label(bright)
    rim = filled & ~inside
    adjacent_rim = ndimage.binary_dilation(rim, iterations=2)
    enclosed_pixels = 0
    for label_id in range(1, n + 1):
        component = components == label_id
        if not np.any(component & adjacent_rim):
            enclosed_pixels += int(np.count_nonzero(component))
    return dict(interior_bright_fraction=float(np.mean(vals >= level)),
                interior_bright_threshold=float(level),
                interior_near_background_fraction=float(np.mean(vals >= 0.90)),
                interior_above_one_fraction=float(np.mean(vals > 1.0)),
                interior_enclosed_bright_fraction=float(enclosed_pixels/count),
                interior_mean_T=float(np.mean(vals)),
                interior_effective_deficit=float(np.mean(1.0-vals)),
                interior_pixel_count=count)
