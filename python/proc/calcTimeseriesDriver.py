#!/usr/bin/env python3
"""Calculate CPI concentration time series from per-file particle statistics."""

from __future__ import annotations

import os

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

from io_utils import atomic_savemat, normalise_directory


def _field_names(mat_struct):
    names = getattr(mat_struct.dtype, "names", None)
    return set(names or ())


DEFAULT_PIXEL_UM = 2.3


def _frame_size_px(path1, cpiv1):
    """(width, height) of a full camera frame in pixels, from the folder's backgrounds if possible."""
    try:
        full = sio.loadmat(os.path.join(path1, "full_backgrounds.mat"), variable_names=["FULL_BG"])["FULL_BG"]
        frame = np.asarray(full[0, 0]["IMAGE"][0, 0][0, 0]["BG"])
        if frame.ndim == 2 and min(frame.shape) >= 512:
            return int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        pass
    raise ValueError("No full-frame background to establish sensor dimensions; concentration unsafe. Supply a verified geometry before computing timeseries.")


def _processing_pixel_um(dat):
    """Pixel size the file was processed with (dat.proc_options), else the nominal 2.3 µm."""
    import json
    try:
        raw = dat["proc_options"][0, 0]
        text = str(np.asarray(raw).ravel()[0])
        value = float(json.loads(text).get("pixel_um", DEFAULT_PIXEL_UM))
        return value if np.isfinite(value) and value > 0 else DEFAULT_PIXEL_UM
    except Exception:
        return DEFAULT_PIXEL_UM


def calcTimeseriesDriver(path1, filename1, foc_crit, dt, ds, vel, outputfile, cpiv1, max_cutoff=None):
    """``max_cutoff``: leave out particles with more than this % of their outline on the ROI edge
    (dat.cutoff_pct; files without it are not filtered)."""
    path1 = normalise_directory(path1)
    frame_w, frame_h = _frame_size_px(path1, cpiv1)
    pixel_sizes = []
    cut_removed = 0

    background_data = sio.loadmat(
        os.path.join(path1, "full_backgrounds.mat"), variable_names=["t_range"]
    )
    t_range = background_data["t_range"]

    print("====================calculating timeseries=========================")
    print("Set-up arrays")

    dt_days = dt / 86400.0
    Time = np.arange(t_range[0, 0], t_range[0, 1] + 0.5 * dt_days, dt_days)
    size1 = np.arange(0, 2300 + ds, ds)
    size_edges = np.arange(0, 2300 + 2 * ds, ds)
    ar1 = np.arange(0.0, 1.2, 0.2)
    ar_edges = np.append(ar1, ar1[-1] + 0.2)

    nt = len(Time)
    nl = len(size1)
    na = len(ar1)
    time_edges = np.concatenate((Time - dt_days / 2.0, [Time[-1] + dt_days / 2.0]))

    timeser = {
        "Time": Time,
        "size1": size1,
        "size2": size1 + ds,
        "midsize": size1 + ds / 2.0,
        "ar1": ar1,
        "ar2": ar1 + 0.2,
        "conc2": np.zeros((nt, nl), dtype=float),
        "conc": np.zeros((nt, 1), dtype=float),
        "deadtimes": np.zeros((nt, 1), dtype=float),
        "nimages": np.zeros((nt, 1), dtype=float),
        "conc2ar": np.zeros((nt, nl, na), dtype=float),
    }


    processed_files = 0
    missing_mat_files = []

    for filename in filename1:
        mat_file = os.path.join(path1, os.path.splitext(filename)[0] + ".mat")
        if not os.path.isfile(mat_file):
            print(f"WARNING: Skipping {filename}: corresponding MAT file is missing", flush=True)
            missing_mat_files.append(filename)
            continue
        print(f"Loading from file... {filename}")
        data = sio.loadmat(mat_file, variable_names=["HOUSE", "IMAGE1", "dat"])
        processed_files += 1
        HOUSE = data["HOUSE"]
        IMAGE1 = data["IMAGE1"]
        dat = data["dat"]
        from instrumentProfile import from_mat_data
        cpiv1 = from_mat_data(data)
        image_type_value = 33857 if cpiv1 else 89

        # New files carry imageType in dat, avoiding the very large ROI_N image
        # structure.  Fall back for already-processed legacy MAT files.
        if "imageType" in _field_names(dat):
            image_types = dat["imageType"][0, 0][0, :]
        else:
            legacy = sio.loadmat(mat_file, variable_names=["ROI_N"])
            image_types = legacy["ROI_N"]["imageType"][0, 0][:, 0]

        pixel_sizes.append(_processing_pixel_um(dat))
        focus = dat["foc"][0, 0]["focus"][0, :]
        selected = (image_types == image_type_value) & (focus > foc_crit)
        if max_cutoff is not None and "cutoff_pct" in _field_names(dat):
            cutoff = np.ravel(np.asarray(dat["cutoff_pct"][0, 0], float))
            if cutoff.size == selected.size:
                selected &= ~(cutoff > max_cutoff)          # NaN (unknown) is kept
                cut_removed += int(np.count_nonzero((image_types == image_type_value) & (focus > foc_crit)
                                                    & (cutoff > max_cutoff)))
        ind = np.flatnonzero(selected)

        # Number of image frames in each time window.
        image_times = np.asarray(IMAGE1["Time1"][0, 0][:, 0], dtype=float)
        if image_times.size:
            counts, _ = np.histogram(image_times, bins=time_edges)
            timeser["nimages"][:, 0] += counts

        # Dead time.  The CPIV1 clock can be put directly on the MATLAB-day
        # time base, so this path is vectorized.  Retain the legacy interpolation
        # for the older clock representation.
        hk_format = str(np.asarray(HOUSE['format'][0, 0]).ravel()[0]).strip() if 'format' in _field_names(HOUSE) else ('legacy' if cpiv1 else '3v83')
        if hk_format == 'legacy':
            if image_times.size:
                house_raw = np.asarray(HOUSE["Time"][0, 0][0, :], dtype=float)
                house_time = (
                    house_raw / 86400.0
                    - np.floor(house_raw / 86400.0)
                    + np.floor(image_times[0])
                )
                deadtime = np.asarray(HOUSE["deadtime"][0, 0][:, 0], dtype=float)
                dead_hist, _ = np.histogram(
                    house_time, bins=time_edges, weights=deadtime
                )
                timeser["deadtimes"][:, 0] += dead_hist
        else:
            image_clock = np.asarray(IMAGE1["Time"][0, 0][0, :], dtype=float)
            if image_times.size >= 2 and image_clock.size >= 2:
                clock_map = interp1d(
                    image_times,
                    image_clock.T,
                    kind="linear",
                    fill_value="extrapolate",
                )
                house_time = np.asarray(HOUSE["Time"][0, 0][0, :], dtype=float)
                house_dead = np.asarray(HOUSE["deadtime"][0, 0][:, 0], dtype=float)
                for j in range(nt):
                    twin = clock_map(time_edges[j : j + 2])
                    in_window = (house_time >= twin[0]) & (house_time < twin[1])
                    timeser["deadtimes"][j, 0] += np.nansum(house_dead[in_window])

        if ind.size:
            particle_times = np.asarray(dat["Time"][0, 0][0, ind], dtype=float)
            lengths = np.asarray(dat["len"][0, 0][ind, 0], dtype=float)
            roundness = np.asarray(dat["round"][0, 0][ind, 0], dtype=float)
            histogram, _ = np.histogramdd(
                (particle_times, lengths, roundness),
                bins=(time_edges, size_edges, ar_edges),
            )
            timeser["conc2ar"] += histogram

    print(f"Time-series input: {processed_files} MAT file(s) used; "
          f"{len(missing_mat_files)} missing MAT file(s) skipped", flush=True)
    if processed_files == 0:
        raise RuntimeError("No converted MAT files available for time-series calculation")

    # Sample area: the image projected onto the plane across the flow (object plane at 45°), using the
    # actual frame size and the pixel size the files were processed with (dat.proc_options).
    pixel_um = float(np.median(pixel_sizes)) if pixel_sizes else DEFAULT_PIXEL_UM
    if pixel_sizes and np.ptp(pixel_sizes) > 1e-9:
        print(f"WARNING: files were processed with different pixel sizes {sorted(set(pixel_sizes))}; using {pixel_um}")
    sa = frame_w * frame_h / np.sqrt(2.0) * (pixel_um * 1e-6) ** 2
    sv = sa * np.sqrt(2.0) * 3e-3
    print(f"Sample area: {frame_w} x {frame_h} px at {pixel_um:g} µm/px -> {sa * 1e6:.4f} mm^2")
    if max_cutoff is not None:
        print(f"Left out {cut_removed} particles cut off by the ROI edge by more than {max_cutoff:g}%")
        timeser["max_cutoff_pct"] = float(max_cutoff)
    timeser["pixel_um"] = pixel_um
    timeser["frame_px"] = np.array([frame_w, frame_h])

    # Scale counts by the effective sampled volume.  Broadcasting avoids the
    # two large tiled arrays created by the old implementation.
    effective_volume = (
        (dt - timeser["deadtimes"][:, 0]) * vel * sa
        + timeser["nimages"][:, 0] * sv
    )
    valid = effective_volume > 0
    scaled = np.full_like(timeser["conc2ar"], np.nan, dtype=float)
    scaled[valid, :, :] = (
        timeser["conc2ar"][valid, :, :] / effective_volume[valid, None, None]
    )
    timeser["conc2ar"] = scaled
    timeser["conc2"] = np.nansum(timeser["conc2ar"], axis=2)
    timeser["conc"] = np.nansum(timeser["conc2"], axis=1)

    print("done")
    print("Saving to file...")
    atomic_savemat(os.path.join(path1, outputfile), {"timeser": timeser})
    print("done")


if __name__ == "__main__":
    import sys
    from io_utils import parse_bool

    if len(sys.argv) not in (9, 10) or sys.argv[1] != "worker":
        raise SystemExit(
            "usage: calcTimeseriesDriver.py worker PATH FOC DT DS VEL OUTPUT CPIV1 [MAX_CUTOFF_PCT]"
        )
    worker_path = normalise_directory(sys.argv[2])
    worker_files = sorted(
        f for f in os.listdir(worker_path) if f.endswith(".roi")
    )
    calcTimeseriesDriver(
        worker_path,
        worker_files,
        float(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
        sys.argv[7],
        parse_bool(sys.argv[8]),
        float(sys.argv[9]) if len(sys.argv) == 10 else None,
    )
