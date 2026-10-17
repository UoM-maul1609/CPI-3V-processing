"""Exploratory spatial best-focus fitting; original CPI focus values are unchanged.

A quadratic peak in the recorded focus statistic is fitted jointly with a
linear spatial surface r_best(u,v).  This is a working hypothesis, not an
optical calibration or a per-particle focus measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass
class SurfaceResult:
    coefficients: list[float]  # radial best position at centre, slope in u, slope in v
    centre_px: list[float]
    scale_px: list[float]
    baseline: float
    curvature: float
    n_particles: int
    radial_range: list[float]
    rms_residual: float
    peak: str

    def as_dict(self):
        return vars(self).copy()


def predict_best_radius(surface: dict, u, v):
    a, b, c = surface["coefficients"]
    uc, vc = surface["centre_px"]
    us, vs = surface["scale_px"]
    return a + b * ((np.asarray(u)-uc)/us) + c * ((np.asarray(v)-vc)/vs)


def fit_surface(records, peak="maximum", min_count=30):
    """Fit focus ~ baseline + curvature*(radius-r_best(u,v))**2.

    Warn/refuse when no stage sweep or when particle positions cannot constrain
    both spatial slopes.  Fit uses bounded robust least squares, not a raw
    quadratic vertex extrapolated beyond the observed stage range.
    """
    values = np.array([[r.stage_x, r.stage_y, r.image_x_px, r.image_y_px, r.focus]
                       for r in records], dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < min_count:
        raise ValueError(f"Need at least {min_count} particles with finite focus, stage and image coordinates; found {len(values)}")
    radius = np.hypot(values[:, 0], values[:, 1])
    u, v, focus = values[:, 2], values[:, 3], values[:, 4]
    extent = np.ptp(radius)
    if extent < 1e-5 or len(np.unique(np.round(radius, 5))) < 5:
        raise ValueError("At least five distinct stage radii spanning a meaningful interval are required")
    centre = [float(np.median(u)), float(np.median(v))]
    scales = [float(np.ptp(u)), float(np.ptp(v))]
    if min(scales) < 1:
        raise ValueError("Image positions do not span both camera axes; cannot fit two tilt slopes")
    un = (u-centre[0])/scales[0]
    vn = (v-centre[1])/scales[1]
    design = np.column_stack((np.ones(len(u)), un, vn))
    if np.linalg.matrix_rank(design) < 3 or np.linalg.cond(design) > 1e4:
        raise ValueError("Image positions do not constrain a two-dimensional tilted surface")
    sign = -1 if peak == "maximum" else 1
    fscale = float(np.percentile(focus, 95) - np.percentile(focus, 5))
    if fscale <= 1e-12:
        raise ValueError("Focus metric has no usable variation")
    # r_best constrained to observed radial range at the sampled image positions.
    r0 = float(np.median(radius))
    fr0 = float(np.percentile(focus, 90 if sign < 0 else 10))
    curvature0 = sign * fscale / max(extent**2, 1e-12)
    def residual(p):
        rb = p[0] + p[1]*un + p[2]*vn
        predicted = p[3] + sign*np.exp(p[4])*(radius-rb)**2
        # Penalise extrapolation of best-focus outside the sampled radial range.
        penalty = np.maximum(radius.min()-rb, 0) + np.maximum(rb-radius.max(), 0)
        return np.r_[ (predicted-focus)/fscale, penalty/extent*3 ]
    start = [r0, 0, 0, fr0, np.log(max(abs(curvature0), 1e-10))]
    fit = least_squares(residual, start, loss="soft_l1", max_nfev=1800,
                        bounds=([radius.min()-extent, -2*extent, -2*extent, -np.inf, -40],
                                [radius.max()+extent, 2*extent, 2*extent, np.inf, 40]))
    p = fit.x
    rb = p[0] + p[1]*un + p[2]*vn
    if not fit.success or np.any(rb < radius.min()-0.01*extent) or np.any(rb > radius.max()+0.01*extent):
        raise ValueError("Fit did not find a well-supported peak inside the measured stage range")
    predicted = p[3]+sign*np.exp(p[4])*(radius-rb)**2
    return SurfaceResult([float(x) for x in p[:3]], centre, scales,
                         float(p[3]), float(sign*np.exp(p[4])), len(values),
                         [float(radius.min()), float(radius.max())],
                         float(np.sqrt(np.mean((predicted-focus)**2))), peak)
