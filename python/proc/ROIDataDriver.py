from __future__ import annotations

import io
import os
import subprocess
import sys

import numpy as np
import scipy.io as sio

from associateBackgrounds import associateBackgrounds
from convertDataToHeaderSA import convertDataToHeaderSA
from convertDataToHouseSA import convertDataToHouseSA
from convertDataToImageSA import convertDataToImageSA
from convertDataToROISA import convertDataToROISA
from fullBackgrounds import fullBackgrounds
from io_utils import atomic_savemat, normalise_directory, parse_bool
from postProcess import postProcess


def _as_matlab_struct(value):
    """Round-trip a small Python dict through MAT format to get scipy's struct form."""
    buffer = io.BytesIO()
    sio.savemat(buffer, {"value": value})
    buffer.seek(0)
    return sio.loadmat(buffer, variable_names=["value"])["value"]


def _empty_full_backgrounds():
    return _as_matlab_struct({"Time": np.array([]), "IMAGE": np.array([])})


def _load_background_state(path1):
    state_file = os.path.join(path1, "full_backgrounds.mat")
    if not os.path.exists(state_file):
        return _empty_full_backgrounds(), 1.0e9, 0.0

    data = sio.loadmat(state_file, variable_names=["FULL_BG", "t_range"])
    full_bg = data.get("FULL_BG", _empty_full_backgrounds())
    t_range = data.get("t_range")
    if t_range is None or t_range.size < 2:
        return full_bg, 1.0e9, 0.0
    return full_bg, float(t_range[0, 0]), float(t_range[0, 1])


def _run_worker(*args):
    command = [sys.executable, os.path.abspath(__file__), *map(str, args)]
    subprocess.run(command, check=True)


def ROIDataDriver(path1, filename, dt, process_sweep1_if_exist, cpiv1):
    """Extract ROI data using one fresh process per file.

    The first sweep is sequential because each file contributes to the shared
    background set.  Importantly, that large background structure stays on disk
    between workers instead of being pickled through multiprocessing pipes.
    """
    path1 = normalise_directory(path1)
    state_file = os.path.join(path1, "full_backgrounds.mat")

    if process_sweep1_if_exist:
        atomic_savemat(state_file, {"FULL_BG": _empty_full_backgrounds()})
    elif not os.path.exists(state_file):
        raise FileNotFoundError(
            "full_backgrounds.mat is required when process_sweep1_if_exist=False; "
            "re-run the first sweep to rebuild it."
        )

    print("=========================1st sweep================================")
    for name in filename:
        _run_worker("sweep1", path1, name, dt, process_sweep1_if_exist, cpiv1)

    data = sio.loadmat(state_file, variable_names=["t_range"])
    if "t_range" not in data:
        raise RuntimeError("First sweep did not produce t_range in full_backgrounds.mat")
    t_range = data["t_range"]

    print("=========================2nd sweep================================")
    for name in filename:
        _run_worker("sweep2", path1, name, cpiv1)

    return t_range


def mult_job(path1, filename1, dt, process_sweep1_if_exist, cpiv1):
    """First-sweep worker for exactly one ROI file."""
    path1 = normalise_directory(path1)
    mat_file = os.path.join(path1, filename1.replace(".roi", ".mat"))
    state_file = os.path.join(path1, "full_backgrounds.mat")

    if os.path.isfile(mat_file) and not process_sweep1_if_exist:
        print(f"Skipping file...{filename1}")
        return

    FULL_BG, t_min, t_max = _load_background_state(path1)

    roi_file = os.path.join(path1, filename1)
    print(f"Reading file...{filename1}")
    with open(roi_file, "rb") as fid:
        raw_bytes = fid.read()
    print("done")

    # The legacy reader searches both 16-bit alignments.  NumPy reproduces the
    # same representation without the very large Python tuples created by
    # struct.unpack('H' * N, ...).
    if len(raw_bytes) % 2:
        raw_bytes = raw_bytes[:-1]
    nwords = len(raw_bytes) // 2

    aligned = np.frombuffer(raw_bytes, dtype=np.dtype("=u2"))
    shifted = np.frombuffer(raw_bytes[1:] + b"1", dtype=np.dtype("=u2"))
    ushort = np.concatenate((aligned, shifted))

    # Map the two alignments back to their approximate original word order.
    # The final shifted word contains the artificial pad byte and should never
    # be a valid block marker, but giving it an order value keeps lengths safe.
    order = np.concatenate(
        (np.arange(nwords, dtype=np.uint32), np.arange(1, nwords + 1, dtype=np.uint32))
    )
    bytes1 = ushort.tobytes()

    print("Finding house keeping...")
    if cpiv1:
        house = np.flatnonzero(ushort == int("0xa1d7", 0))
    else:
        house = np.flatnonzero(ushort == int("0x484B", 0))
    print("done")

    print("Finding image data...")
    images = np.flatnonzero(ushort == int("0xa3d5", 0))
    print("done")

    print("Finding roi data...")
    rois = np.flatnonzero(ushort == int("0xb2e6", 0))
    print("done")

    print("Post-processing data, stage 1...")
    Header = convertDataToHeaderSA(ushort)
    I, images = convertDataToImageSA(bytes1, ushort, order, images)
    R, rois = convertDataToROISA(bytes1, ushort, order, rois)
    H = convertDataToHouseSA(bytes1, ushort, order, house, cpiv1)
    print("done")

    print("Post-processing data, stage 2...")
    ROI_N, HOUSE, IMAGE1 = postProcess(bytes1, rois, R, H, I, Header, cpiv1)
    print("done")

    print("Getting backgrounds...")
    FULL_BG1 = fullBackgrounds(ROI_N, cpiv1)
    if len(FULL_BG1["Time"]):
        FULL_BG1 = _as_matlab_struct(FULL_BG1)
        existing = len(FULL_BG["IMAGE"][0, 0])
        if existing > 0:
            FULL_BG["IMAGE"][0, 0] = np.append(
                FULL_BG["IMAGE"][0, 0], FULL_BG1["IMAGE"][0, 0], axis=1
            )
            FULL_BG["Time"][0, 0] = np.append(
                FULL_BG["Time"][0, 0], FULL_BG1["Time"][0, 0], axis=0
            )
        else:
            FULL_BG = FULL_BG1
    print("done")

    if ROI_N["Time"].size == 0:
        raise RuntimeError(f"No ROI times found in {filename1}")
    t_min = min(float(np.min(ROI_N["Time"])), t_min)
    t_max = max(float(np.max(ROI_N["Time"])), t_max)
    t_range = np.array(
        [np.floor(t_min * 86400 / dt) * dt / 86400, np.ceil(t_max * 86400 / dt) * dt / 86400]
    )

    print("Saving to file...")
    atomic_savemat(mat_file, {"ROI_N": ROI_N, "HOUSE": HOUSE, "IMAGE1": IMAGE1})
    atomic_savemat(state_file, {"FULL_BG": FULL_BG, "t_range": t_range})
    print("done")


def mult_job2(path1, filename1, cpiv1):
    """Second-sweep worker: associate the complete background set with one file."""
    path1 = normalise_directory(path1)
    mat_file = os.path.join(path1, filename1.replace(".roi", ".mat"))
    state_file = os.path.join(path1, "full_backgrounds.mat")

    print("Loading from file...")
    data = sio.loadmat(mat_file, variable_names=["ROI_N", "HOUSE", "IMAGE1"])
    ROI_N = data["ROI_N"]
    HOUSE = data["HOUSE"]
    IMAGE1 = data["IMAGE1"]
    full_data = sio.loadmat(state_file, variable_names=["FULL_BG"])
    FULL_BG = full_data["FULL_BG"]
    print("done")

    print(f"Associate backgrounds...{filename1}")
    BG = associateBackgrounds(ROI_N, FULL_BG)
    print("done")

    print("Saving to file...")
    atomic_savemat(
        mat_file, {"ROI_N": ROI_N, "HOUSE": HOUSE, "IMAGE1": IMAGE1, "BG": BG}
    )
    print("done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("ROIDataDriver worker mode required")

    mode = sys.argv[1]
    if mode == "driver":
        if len(sys.argv) != 6:
            raise SystemExit(
                "usage: ROIDataDriver.py driver PATH DT PROCESS_EXISTING CPIV1"
            )
        driver_path = normalise_directory(sys.argv[2])
        driver_files = sorted(
            f for f in os.listdir(driver_path) if f.endswith(".roi")
        )
        ROIDataDriver(
            driver_path,
            driver_files,
            float(sys.argv[3]),
            parse_bool(sys.argv[4]),
            parse_bool(sys.argv[5]),
        )
    elif mode == "sweep1":
        if len(sys.argv) != 7:
            raise SystemExit(
                "usage: ROIDataDriver.py sweep1 PATH FILE DT PROCESS_EXISTING CPIV1"
            )
        mult_job(
            sys.argv[2],
            sys.argv[3],
            float(sys.argv[4]),
            parse_bool(sys.argv[5]),
            parse_bool(sys.argv[6]),
        )
    elif mode == "sweep2":
        if len(sys.argv) != 5:
            raise SystemExit("usage: ROIDataDriver.py sweep2 PATH FILE CPIV1")
        mult_job2(sys.argv[2], sys.argv[3], parse_bool(sys.argv[4]))
    else:
        raise SystemExit(f"Unknown worker mode: {mode}")
