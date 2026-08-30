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


def calcTimeseriesDriver(path1, filename1, foc_crit, dt, ds, vel, outputfile, cpiv1):
    path1 = normalise_directory(path1)
    sa = 1280.0 * 1024.0 / np.sqrt(2.0) * 2.3e-6**2
    sv = sa * np.sqrt(2.0) * 3e-3

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

    image_type_value = 33857 if cpiv1 else 89

    for filename in filename1:
        print(f"Loading from file... {filename}")
        mat_file = os.path.join(path1, filename.replace(".roi", ".mat"))
        data = sio.loadmat(mat_file, variable_names=["HOUSE", "IMAGE1", "dat"])
        HOUSE = data["HOUSE"]
        IMAGE1 = data["IMAGE1"]
        dat = data["dat"]

        # New files carry imageType in dat, avoiding the very large ROI_N image
        # structure.  Fall back for already-processed legacy MAT files.
        if "imageType" in _field_names(dat):
            image_types = dat["imageType"][0, 0][0, :]
        else:
            legacy = sio.loadmat(mat_file, variable_names=["ROI_N"])
            image_types = legacy["ROI_N"]["imageType"][0, 0][:, 0]

        focus = dat["foc"][0, 0]["focus"][0, :]
        selected = (image_types == image_type_value) & (focus > foc_crit)
        ind = np.flatnonzero(selected)

        # Number of image frames in each time window.
        image_times = np.asarray(IMAGE1["Time1"][0, 0][:, 0], dtype=float)
        if image_times.size:
            counts, _ = np.histogram(image_times, bins=time_edges)
            timeser["nimages"][:, 0] += counts

        # Dead time.  The CPIV1 clock can be put directly on the MATLAB-day
        # time base, so this path is vectorized.  Retain the legacy interpolation
        # for the older clock representation.
        if cpiv1:
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

    if len(sys.argv) != 9 or sys.argv[1] != "worker":
        raise SystemExit(
            "usage: calcTimeseriesDriver.py worker PATH FOC DT DS VEL OUTPUT CPIV1"
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
    )
