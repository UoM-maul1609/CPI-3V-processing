"""Read-only transmission profiles for calibration particle ROIs.

ROI axes and stored boundaries use (camera x, camera y) coordinates.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def normalised_transmission(particle, background, edge=2):
    """Spatial background division followed by clear-border laser-power scaling.

    Return (transmission, laser_scale). Invalid/saturated background is NaN.
    Nothing is clipped to [0, 1], retaining bright diffraction fringes.
    """
    p = np.asarray(particle, dtype=float)
    b = np.asarray(background, dtype=float)
    if p.shape != b.shape or p.ndim != 2:
        raise ValueError('Particle and background must be matching 2-D arrays')
    valid = np.isfinite(p) & np.isfinite(b) & (b > 2) & (b < 255) & (p < 255)
    ratio = np.full(p.shape, np.nan, dtype=float)
    np.divide(p, b, out=ratio, where=valid)
    if min(p.shape) < 2 * edge + 3:
        raise ValueError('ROI too small to estimate clear border')
    border = np.zeros(p.shape, dtype=bool)
    border[:edge, :] = border[-edge:, :] = True
    border[:, :edge] = border[:, -edge:] = True
    reference = ratio[border & valid]
    if reference.size < 8:
        raise ValueError('Insufficient valid clear border pixels')
    scale = float(np.median(reference))
    if not np.isfinite(scale) or scale <= 0.2:
        raise ValueError('Unreliable border laser-power scaling')
    return ratio / scale, scale


def feret_profile(transmission, boundary, pixel_um, true_length_um, normalise_x=True,
                  points=301, span=1.6):
    """Profile along maximum-Feret axis, centred on the midpoint of its endpoints.

    x is signed distance/L_true if normalise_x, otherwise signed distance in µm.
    NaN marks locations outside the recorded ROI. No extrapolation.
    """
    b = np.asarray(boundary, dtype=float)
    b = b[np.isfinite(b).all(axis=1)]
    if len(b) < 2 or not np.isfinite(pixel_um) or pixel_um <= 0:
        raise ValueError('Missing boundary or pixel scale')
    if not np.isfinite(true_length_um) or true_length_um <= 0:
        raise ValueError('Invalid labelled true length')
    # Brute force is acceptable for small outlines; convex-hull vertices
    # substantially reduce memory for large outlines.
    from scipy.spatial import ConvexHull
    if len(b) > 2:
        try:
            b = b[ConvexHull(b).vertices]
        except Exception:
            pass
    delta = b[:, None, :] - b[None, :, :]
    i, j = np.unravel_index(np.argmax(np.sum(delta * delta, axis=2)), (len(b), len(b)))
    end1, end2 = b[i], b[j]
    centre = 0.5 * (end1 + end2)
    vec = end2 - end1
    vec /= np.linalg.norm(vec)
    xnorm = np.linspace(-span, span, int(points))
    physical_um = xnorm * true_length_um
    coords = centre[:, None] + vec[:, None] * (physical_um[None, :] / pixel_um)
    T = np.asarray(transmission, dtype=float)
    values = map_coordinates(T, coords, order=1, mode='constant', cval=np.nan)
    return (xnorm if normalise_x else physical_um), values


def aggregate_profiles(profiles, statistic='mean'):
    """NaN-aware mean/median with sample counts on a common coordinate grid."""
    stack = np.asarray(profiles, dtype=float)
    count = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        if statistic == 'median':
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = np.nanmedian(stack, axis=0)
        else:
            result = np.nansum(stack, axis=0) / np.maximum(count, 1)
    result[count == 0] = np.nan
    return result, count
