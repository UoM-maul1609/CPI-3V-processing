#!/usr/bin/env python3
"""Particle geometry and focus statistics for CPI ROI images."""

from __future__ import annotations

import numpy as np
from cv2 import THRESH_BINARY, THRESH_OTSU, medianBlur, setNumThreads, threshold
from matplotlib import path as mpl_path
from scipy import signal
from scipy.interpolate import RectBivariateSpline
from skimage import measure
from tqdm import tqdm

try:
    from edgeDetect import EdgeOptions, measure_transmittance, options_json
except ImportError:  # imported as proc.imageStats
    from .edgeDetect import EdgeOptions, measure_transmittance, options_json

NB = 30
LD = 3
PIXEL_SIZE_UM = 2.3


def _median_filter_interior(image, passes):
    """Repeat the legacy 3x3 median filter without repeated dtype allocations.

    The old implementation called scipy.signal.medfilt2d(image.astype('B')) on
    every pass and copied only [1:-1, 1:-1] back.  OpenCV gives the same 3x3
    median values for those interior pixels, so this preserves the algorithm
    while being substantially faster.
    """
    work = np.asarray(image, dtype=np.uint8).copy()
    for _ in range(passes):
        filtered = medianBlur(work, 3)
        work[1:-1, 1:-1] = filtered[1:-1, 1:-1]
    return work


def _points_for_shape(r, c, cache):
    """Return the legacy row/column point ordering used by Path.contains_points."""
    key = (r, c)
    points = cache.get(key)
    if points is None:
        x, y = np.meshgrid(np.arange(r), np.arange(c))
        points = np.column_stack((x.ravel(), y.ravel()))
        cache[key] = points
    return points


def _remove_centre_strip(arr, start_x):
    """CPI 1.0 centre-strip artefact (rows 511/512 of the full frame), as in the legacy code."""
    h, _ = arr.shape
    end1 = start_x + h
    if end1 >= 512 and start_x < 511:
        idx = 512 - start_x
        if 2 <= idx <= h:
            arr[idx - 1: idx, :] = arr[idx - 2: idx - 1, :]
    if end1 >= 514 and start_x <= 512:
        idx = 512 - start_x
        if 0 <= idx < h - 1:
            arr[idx: idx + 1, :] = arr[idx + 1: idx + 2, :]
    return arr


def imageStats(ROI_N, BG, cpiv1, b_flag, position, desc, globalLock, options=None):
    """Particle statistics for every particle ROI.

    ``options`` (an EdgeOptions, a dict or a JSON string) selects the edge
    method; the default is the legacy algorithm, unchanged.  The options used
    are stored in ``dat["proc_options"]`` (JSON) so every processed file says
    how it was made.
    """
    # Avoid nested OpenCV threading when several file subprocesses run at once.
    setNumThreads(1)
    if options is None:
        options = EdgeOptions(pixel_um=PIXEL_SIZE_UM)
    elif isinstance(options, str):
        import json
        options = EdgeOptions(**json.loads(options))
    elif isinstance(options, dict):
        options = EdgeOptions(**options)
    if options.method == "transmittance":
        return _image_stats_transmittance(ROI_N, BG, cpiv1, b_flag, position, desc, options)

    image_type = 33857 if cpiv1 else 89
    ind = np.flatnonzero(ROI_N["imageType"][0, 0][:, 0] == image_type)
    nroi = len(ROI_N["IMAGE"][0, 0][0, :])

    dat = {
        "len": np.zeros((nroi, 1), dtype=float),
        "wid": np.zeros((nroi, 1), dtype=float),
        "area": np.zeros((nroi, 1), dtype=float),
        "round": np.zeros((nroi, 1), dtype=float),
        "orientation": np.zeros((nroi, 1), dtype=float),
        "centroid": np.zeros((nroi, 2), dtype=float),
    }

    if b_flag:
        foc = np.zeros(
            nroi, dtype=[("focus", "float"), ("boundaries", "(30,2)float")]
        )
    else:
        foc = np.zeros(nroi, dtype=[("focus", "float")])
    foc.fill(np.nan)
    dat["foc"] = foc
    dat["Time"] = ROI_N["Time"][0, 0][:, 0]
    dat["proc_options"] = options_json(options, {"note": "legacy algorithm (Otsu, 100 median passes, "
                                                          "radius=2 mask, ellipse major axis, 30 boundary points)"})
    # Store the small classification field here so later stages do not need to
    # reload ROI_N, whose nested IMAGE field can dominate memory.
    dat["imageType"] = ROI_N["imageType"][0, 0][:, 0]

    if ind.size == 0:
        return dat

    points_cache = {}
    with tqdm(
        total=len(ind),
        position=position,
        leave=True,
        desc=f"{desc} {position}",
        mininterval=0.5,
    ) as pbar:
        for roi_index in ind:
            image = ROI_N["IMAGE"][0, 0][0, roi_index]["IM"][0, 0]
            background = BG[0, roi_index]["BG"][0, 0]

            # int16 is sufficient for uint8 - uint8 and uses a quarter of the
            # memory of the platform-default int64 used previously.
            arr = image.astype(np.int16) - background.astype(np.int16)

            if np.count_nonzero(arr <= -15) <= 20:
                pbar.update(1)
                continue

            arr -= arr.min()

            if cpiv1:
                # Remove the CPI centre-strip artefact exactly as in the legacy code.
                h, _ = arr.shape
                start1 = int(ROI_N["StartX"][0, 0][0, roi_index])
                end1 = start1 + h
                if end1 >= 512 and start1 < 511:
                    idx = 512 - start1
                    if 2 <= idx <= h:
                        arr[idx - 1 : idx, :] = arr[idx - 2 : idx - 1, :]
                if end1 >= 514 and start1 <= 512:
                    idx = 512 - start1
                    if 0 <= idx < h - 1:
                        arr[idx : idx + 1, :] = arr[idx + 1 : idx + 2, :]

            # Preserve the original 100 iterations but avoid 100 SciPy calls
            # and 100 full arr.astype('B') allocations.
            filtered = _median_filter_interior(arr, 100)
            arr[1:-1, 1:-1] = filtered[1:-1, 1:-1]

            r, c = arr.shape
            if r <= 15 and c <= 15:
                pbar.update(1)
                continue

            p90 = np.percentile(arr, 90)
            if np.count_nonzero(arr.ravel() - p90 < -30) <= 10:
                pbar.update(1)
                continue

            _, level = threshold(
                arr.astype(np.uint8), 0, 255, THRESH_BINARY + THRESH_OTSU
            )

            # The old code performed one median pass and then ten more.
            BW2 = _median_filter_interior(level, 11)
            contours = measure.find_contours(
                (BW2 - np.min(BW2)) / 255.0, 0.4
            )
            if not contours:
                pbar.update(1)
                continue

            contour = max(contours, key=len)
            if len(contour) <= 5 or len(contour) > 10000:
                pbar.update(1)
                continue

            points = _points_for_shape(r, c, points_cache)
            mask = mpl_path.Path(contour).contains_points(points, radius=2)
            inside = mask.reshape((c, r)).T

            stats = measure.regionprops(inside.astype(np.uint8))
            if not stats:
                pbar.update(1)
                continue

            region = stats[0]
            # scikit-image renamed these properties in 0.26; support both APIs.
            major_axis = (
                region.axis_major_length
                if hasattr(region, "axis_major_length")
                else region.major_axis_length
            )
            minor_axis = (
                region.axis_minor_length
                if hasattr(region, "axis_minor_length")
                else region.minor_axis_length
            )
            filled_area = (
                region.area_filled
                if hasattr(region, "area_filled")
                else region.filled_area
            )

            if major_axis == 0 or region.eccentricity > 0.9999:
                pbar.update(1)
                continue

            length = major_axis * PIXEL_SIZE_UM
            area = filled_area * PIXEL_SIZE_UM**2
            if area <= PIXEL_SIZE_UM**2:
                pbar.update(1)
                continue

            dat["len"][roi_index] = length
            dat["wid"][roi_index] = minor_axis * PIXEL_SIZE_UM
            dat["area"][roi_index] = area
            dat["round"][roi_index] = filled_area / (
                np.pi / 4.0 * major_axis**2
            )
            dat["centroid"][roi_index, :] = region.centroid
            dat["orientation"][roi_index] = region.orientation
            dat["foc"][roi_index] = calculate_focus(contour, arr, b_flag)

            pbar.update(1)

    return dat


def calculate_focus(boundaries, IM, b_flag):
    """Calculate the legacy edge-gradient focus metric."""
    if b_flag:
        foc = np.zeros(
            1, dtype=[("focus", "float"), ("boundaries", "(30,2)float")]
        )
    else:
        foc = np.zeros(1, dtype=[("focus", "float")])
    foc["focus"][0] = np.nan

    boundaries = np.asarray(boundaries)
    if boundaries.ndim != 2 or boundaries.shape[0] == 0:
        return foc

    boundaries2 = np.zeros((NB, 2), dtype=float)
    xs = np.zeros((LD, NB - 1), dtype=float)
    ys = np.zeros((LD, NB - 1), dtype=float)
    focus = np.full(NB - 1, np.nan, dtype=float)
    intensity = np.zeros(LD, dtype=float)

    ind1 = np.linspace(1, NB, boundaries.shape[0])
    ind2 = np.linspace(1, ind1[-1], NB)

    if boundaries.shape[0] > 9:
        boundaries2[:, 0] = np.interp(
            ind2, ind1, signal.filtfilt([1, 1, 1], 3, boundaries[:, 0])
        )
        boundaries2[:, 1] = np.interp(
            ind2, ind1, signal.filtfilt([1, 1, 1], 3, boundaries[:, 1])
        )
    else:
        boundaries2[:, 0] = np.interp(ind2, ind1, boundaries[:, 0])
        boundaries2[:, 1] = np.interp(ind2, ind1, boundaries[:, 1])

    midpointx = (boundaries2[1:, 0] + boundaries2[:-1, 0]) / 2.0
    midpointy = (boundaries2[1:, 1] + boundaries2[:-1, 1]) / 2.0

    deltax = boundaries2[1:, 0] - boundaries2[:-1, 0]
    deltay = boundaries2[1:, 1] - boundaries2[:-1, 1]
    gradient1 = deltay / (1.0e-10 + deltax)
    gradient2 = -1.0 / (gradient1 + 1.0e-10)

    dx1 = -2.0 / np.sqrt(1.0 + gradient2**2)
    dy1 = -2.0 / np.sqrt(1.0 + 1.0 / gradient2**2)
    dx2 = 2.0 / np.sqrt(1.0 + gradient2**2)
    dy2 = 2.0 / np.sqrt(1.0 + 1.0 / gradient2**2)

    for i in range(NB - 1):
        xs[:, i] = np.linspace(midpointx[i] + dx1[i], midpointx[i] + dx2[i], LD)
        ys[:, i] = np.linspace(midpointy[i] - dy1[i], midpointy[i] - dy2[i], LD)

    reverse_y = ((deltay < 0) & (deltax > 0)) | ((deltay > 0) & (deltax < 0))
    for i in np.flatnonzero(reverse_y):
        ys[:, i] = np.linspace(midpointy[i] + dy1[i], midpointy[i] + dy2[i], LD)

    try:
        nr, nc = IM.shape
        # Cubic splines need at least four samples in each dimension.
        if nr < 4 or nc < 4:
            raise ValueError("image too small for cubic focus spline")
        spline = RectBivariateSpline(np.arange(nr), np.arange(nc), IM)
        for i in range(NB - 1):
            for j in range(LD):
                intensity[j] = spline.ev(xs[j, i], ys[j, i])
            focus[i] = abs(np.gradient(intensity, edge_order=2)[1])
        foc["focus"][0] = np.nanmean(focus) / 2.0
    except (ValueError, TypeError, FloatingPointError):
        foc["focus"][0] = np.nan

    if b_flag:
        foc["boundaries"][0] = boundaries2
    return foc


def _unwrap(value):
    """The 2-D image inside however many 1x1 MATLAB cell/struct wrappers loadmat produced."""
    v = np.asarray(value)
    while v.dtype == object and v.size == 1:
        v = np.asarray(v.ravel()[0])
    return v


def _image_stats_transmittance(ROI_N, BG, cpiv1, b_flag, position, desc, options):
    """Transmittance edge method (see edgeDetect.py); variable-length boundaries."""
    image_type = 33857 if cpiv1 else 89
    ind = np.flatnonzero(ROI_N["imageType"][0, 0][:, 0] == image_type)
    nroi = len(ROI_N["IMAGE"][0, 0][0, :])
    nan = lambda *shape: np.full(shape, np.nan)
    dat = {
        "len": nan(nroi, 1), "wid": nan(nroi, 1), "area": nan(nroi, 1), "round": nan(nroi, 1),
        "orientation": nan(nroi, 1), "centroid": nan(nroi, 2),
        "len_ellipse": nan(nroi, 1), "perimeter": nan(nroi, 1), "deq": nan(nroi, 1),
        "T_min": nan(nroi, 1), "T_centre": nan(nroi, 1), "T_border": nan(nroi, 1), "edge_level_used": nan(nroi, 1),
        "n_boundary": np.zeros((nroi, 1)), "edge_touch": np.zeros((nroi, 1)),
    }
    if b_flag:
        foc = np.zeros(nroi, dtype=[("focus", "float"), ("focus_T", "float"), ("boundaries", "O")])
        for i in range(nroi):
            foc[i]["boundaries"] = np.zeros((0, 2))
    else:
        foc = np.zeros(nroi, dtype=[("focus", "float"), ("focus_T", "float")])
    foc["focus"] = np.nan
    foc["focus_T"] = np.nan
    dat["foc"] = foc
    dat["Time"] = ROI_N["Time"][0, 0][:, 0]
    dat["imageType"] = ROI_N["imageType"][0, 0][:, 0]
    dat["proc_options"] = options_json(options)
    if ind.size == 0:
        return dat
    with tqdm(total=len(ind), position=position, leave=True, desc=f"{desc} {position}", mininterval=0.5) as pbar:
        for roi_index in ind:
            pbar.update(1)
            image = _unwrap(ROI_N["IMAGE"][0, 0][0, roi_index]["IM"])
            background = _unwrap(BG[0, roi_index]["BG"])
            if image.shape != background.shape or min(image.shape) < 3:
                continue
            image = image.astype(np.float64)
            background = background.astype(np.float64)
            if cpiv1:
                start = int(ROI_N["StartX"][0, 0][0, roi_index])
                image = _remove_centre_strip(image, start)
                background = _remove_centre_strip(background, start)
            try:
                res = measure_transmittance(image, background, options)
            except (ValueError, IndexError, FloatingPointError):
                res = None
            if res is None:
                continue
            for key in ("len", "wid", "area", "round", "orientation", "len_ellipse", "perimeter", "deq",
                        "T_min", "T_centre", "T_border", "n_boundary", "edge_touch", "edge_level_used"):
                dat[key][roi_index] = res[key]
            dat["centroid"][roi_index, :] = res["centroid"]
            foc[roi_index]["focus"] = res["focus"]
            foc[roi_index]["focus_T"] = res["focus_T"]
            if b_flag:
                foc[roi_index]["boundaries"] = res["boundary"]
    return dat
