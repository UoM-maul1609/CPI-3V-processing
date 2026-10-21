"""Which full-frame backgrounds to use: the saved selection and how it is applied.

The CPI's full-frame backgrounds are collected into ``full_backgrounds.mat`` by
the first sweep.  By default every one is used (each ROI gets the nearest in
time).  ``batchProcess.py --backgrounds choose`` opens a window
(``backgroundSelector.py``) to tick the ones to use -- e.g. to leave out frames
that contain particles on a calibration slide -- and saves the choice in the data
folder as ``background_selection.json``; ``--backgrounds saved`` reuses it
without the window.

The selection is keyed by background *time*, not by index, so it still applies
after the folder is reprocessed from the .roi files.  Frames not in the saved
file (new ones) are used.  ``full_backgrounds.mat`` itself is never changed.

The selection in force is passed to the worker processes in the environment
variable ``CPI_BG_SELECTION`` (path of the JSON file); unset means use all.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import List, Optional

import numpy as np

SELECTION_FILE = "background_selection.json"
ENV_VAR = "CPI_BG_SELECTION"
MATCH_TOL_DAYS = 0.5 / 86400.0          # backgrounds matched by time to within 0.5 s


def selection_path(folder: str) -> str:
    return os.path.join(folder, SELECTION_FILE)


def datenum_to_str(t: float) -> str:
    try:
        d = _dt.datetime.fromordinal(int(t)) + _dt.timedelta(days=float(t) % 1) - _dt.timedelta(days=366)
        return d.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OverflowError):
        return ""


# --------------------------------------------------------------------------- frames
def load_frames(folder: str):
    """(times [MATLAB datenum], [frame arrays]) from the folder's full_backgrounds.mat."""
    import scipy.io as sio
    path = os.path.join(folder, "full_backgrounds.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the first processing sweep first")
    full = sio.loadmat(path, variable_names=["FULL_BG"]).get("FULL_BG")
    if full is None or full.size == 0:
        return np.zeros(0), []
    times = np.asarray(full["Time"][0, 0], float).ravel()
    if times.size == 0:
        return times, []
    images = full[0, 0]["IMAGE"][0, :]
    return times, [np.asarray(images[j][0, 0]["BG"]) for j in range(len(times))]


# --------------------------------------------------------------------------- selection file
def save_selection(folder: str, times, use, note: str = "") -> str:
    times = np.asarray(times, float).ravel()
    use = np.asarray(use, bool).ravel()
    record = {
        "version": 1,
        "saved": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_frames": int(times.size),
        "n_used": int(use.sum()),
        "note": note,
        "frames": [{"index": int(j), "time_matlab": float(t), "time": datenum_to_str(t), "use": bool(u)}
                   for j, (t, u) in enumerate(zip(times, use))],
    }
    path = selection_path(folder)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    os.replace(tmp, path)
    return path


def read_selection(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def mask_from_selection(record: dict, times) -> np.ndarray:
    """Use-flag for each background time; frames absent from the record are used."""
    times = np.asarray(times, float).ravel()
    saved = record.get("frames", [])
    if not saved:
        return np.ones(times.size, bool)
    st = np.array([f["time_matlab"] for f in saved], float)
    su = np.array([bool(f["use"]) for f in saved])
    use = np.ones(times.size, bool)
    for j, t in enumerate(times):
        if j < st.size and abs(st[j] - t) <= MATCH_TOL_DAYS:     # same position, same time (keeps duplicates apart)
            use[j] = su[j]
            continue
        k = int(np.argmin(np.abs(st - t)))
        if abs(st[k] - t) <= MATCH_TOL_DAYS:
            use[j] = su[k]
    return use


def active_mask(times) -> Optional[np.ndarray]:
    """Use-flags of the selection in force for these background times, or None (use all)."""
    path = os.environ.get(ENV_VAR, "")
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{ENV_VAR} points to {path}, which does not exist")
    use = mask_from_selection(read_selection(path), times)
    if not use.any():
        raise ValueError(f"The background selection {path} leaves no backgrounds to use")
    return use


# --------------------------------------------------------------------------- spots
def dark_spots(frames: List[np.ndarray], factor: int = 4, level: float = 0.85, min_px: int = 2):
    """For each frame: (number of dark spots, ratio image) relative to the per-pixel median of all frames.

    Particles that sit in a background show up as dark spots against the median
    (which they are absent from, as long as they move between frames).  The
    ratio images are at 1/``factor`` resolution, for display.
    """
    from scipy import ndimage
    if not frames:
        return [], []
    shape = frames[0].shape
    h, w = (shape[0] // factor) * factor, (shape[1] // factor) * factor
    small = []
    for f in frames:
        if f.shape != shape:
            small.append(None)
            continue
        a = np.asarray(f[:h, :w], float).reshape(h // factor, factor, w // factor, factor).mean((1, 3))
        small.append(a)
    stack = np.array([s for s in small if s is not None])
    median = np.maximum(np.median(stack, axis=0), 1.0)
    counts, ratios = [], []
    for s in small:
        if s is None:
            counts.append(-1)
            ratios.append(None)
            continue
        r = s / median
        r = r / max(float(np.median(r)), 1e-6)          # overall brightness drift
        lab, n = ndimage.label(r < level)
        if n:
            sizes = ndimage.sum(np.ones_like(r), lab, index=np.arange(1, n + 1))
            n = int(np.sum(sizes >= min_px))
        counts.append(int(n))
        ratios.append(r)
    return counts, ratios
