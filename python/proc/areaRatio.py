"""Projected contour area divided by its minimum-enclosing-circle area.

This is CPI morphology, not an independent microscope reference area ratio.
"""
import numpy as np
import cv2


def contour_area_ratio(boundary, area_um2, pixel_um):
    pts = np.asarray(boundary, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        return float('nan')
    if not np.isfinite(area_um2) or area_um2 <= 0 or pixel_um <= 0:
        return float('nan')
    (cx, cy), radius = cv2.minEnclosingCircle(pts[:, ::-1].astype(np.float32))
    if radius <= 0:
        return float('nan')
    ratio = float(area_um2) / (np.pi * (float(radius) * pixel_um) ** 2)
    return ratio if np.isfinite(ratio) and 0 < ratio <= 1.05 else float('nan')
