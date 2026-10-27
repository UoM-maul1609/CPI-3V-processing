"""Read-only CPI contour geometry and interior optical diagnostics.

Saved and tested boundaries are in CPI (row, column) pixel coordinates.
A contour is treated as the *outer* filled silhouette; optical bright
regions within it are measured separately and never removed from its area.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage
from skimage.draw import polygon as raster_polygon
from skimage.measure import regionprops


def contour_metrics(boundary, pixel_um=2.3):
    """Feret / ellipse / enclosing-circle metrics from one real contour.

    Only measurements with adequate contour geometry are returned.  No change
    is made to saved CPI measurements or image segmentation.
    """
    b = np.asarray(boundary if boundary is not None else [], dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        return {}
    b = b[np.isfinite(b).all(axis=1)]
    if len(b) > 1 and np.allclose(b[0], b[-1]):
        b = b[:-1]
    if len(b) < 3:
        return {}
    try:
        from proc.edgeDetect import feret, polygon_area
    except ImportError:
        from ..proc.edgeDetect import feret, polygon_area
    try:
        import cv2
        (cx, cy), r = cv2.minEnclosingCircle(b[:, ::-1].astype(np.float32))
        circle_radius = max(float(r), float(np.max(np.linalg.norm(b - [cy, cx], axis=1))))
    except ImportError:
        circle_radius = float('nan')
    # Contour area (shoelace) does not rely on pixel rasterisation resolution.
    area = float(polygon_area(b))
    length, width, _ = feret(b)
    y0, x0 = np.floor(b.min(axis=0)).astype(int) - 2
    high = np.ceil(b.max(axis=0)).astype(int) + 3
    height, width_shape = np.maximum(high - [y0, x0], 3)
    rr, cc = raster_polygon(b[:, 0]-y0, b[:, 1]-x0,
                            shape=(int(height), int(width_shape)))
    mask = np.zeros((int(height), int(width_shape)), dtype=np.uint8)
    mask[rr, cc] = 1
    ell_major = ell_minor = float('nan')
    if np.count_nonzero(mask) > 3:
        props = regionprops(mask)[0]
        # Use current scikit-image names, avoiding deprecation warnings.
        ell_major = float(props.axis_major_length)
        ell_minor = float(props.axis_minor_length)
    px = float(pixel_um)
    ratio = area / (math.pi*circle_radius**2) if circle_radius > 0 else float('nan')
    return dict(len_feret_um=float(length)*px, wid_feret_um=float(width)*px,
                len_ellipse_um=ell_major*px, wid_ellipse_um=ell_minor*px,
                contour_area_um2=area*px**2,
                deq_um=2*math.sqrt(max(0.,area)/math.pi)*px,
                area_ratio_circle=ratio,
                enclosing_circle_radius_px=circle_radius)


# The GUI and production processing use exactly the same optical diagnostic.
try:
    from proc.interiorLight import interior_light_metrics, INTERIOR_FIELD_NAMES
except ImportError:  # `python gui.py` run directly from the calibration directory
    import sys
    from pathlib import Path
    _python_root = str(Path(__file__).resolve().parents[1])
    if _python_root not in sys.path:
        sys.path.insert(0, _python_root)
    from proc.interiorLight import interior_light_metrics, INTERIOR_FIELD_NAMES
