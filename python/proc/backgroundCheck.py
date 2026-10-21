"""Make sure a particle is not in its own background.

The CPI takes full-frame backgrounds now and then, and each ROI is divided by
the crop of the background nearest in time.  With particles fixed on a slide
(calibration beads, ice analogues) a background taken during a run contains
the particles, so the particle cancels out of image / background and only a
sliver where it moved is left.  This module checks the assigned crop and,
when the particle is in it, finds a clean one:

1. another full-frame background from ``full_backgrounds.mat`` (nearest in
   time first) in which this part of the frame is clear;
2. failing that, a smooth background fitted to the ROI's own clear pixels
   (a low-order surface through the border and other pixels away from the
   particle) -- loses the fixed pattern of the optics but never contains the
   particle.

``source`` codes stored per particle: 0 assigned background used, 1 another
full-frame background, 2 smooth background fitted to the ROI.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

SOURCE_ASSIGNED, SOURCE_OTHER_FRAME, SOURCE_FITTED = 0, 1, 2


def _relative(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    border = np.concatenate([a[0], a[-1], a[1:-1, 0], a[1:-1, -1]])
    return a / max(float(np.median(border)), 1.0)


def particle_mask(image: np.ndarray, core: float = 0.6, grow: int = 2) -> np.ndarray:
    """Dark core of the particle in the raw image (relative to the ROI border), slightly grown."""
    mask = _relative(image) < core
    if grow and mask.any():
        mask = ndimage.binary_dilation(mask, iterations=grow)
    return mask


def contains_particle(background: np.ndarray, mask: np.ndarray, level: float = 0.85,
                      fraction: float = 0.1) -> bool:
    """True if more than ``fraction`` of the particle's area is also dark in the background."""
    if background.shape != mask.shape or not mask.any():
        return False
    return float(np.mean(_relative(background)[mask] < level)) > fraction


def fitted_background(image: np.ndarray, exclude: np.ndarray, order: int = 2) -> np.ndarray:
    """Smooth background through the ROI's clear pixels (2-D polynomial of the given order)."""
    img = np.asarray(image, float)
    rel = _relative(img)
    clear = ~ndimage.binary_dilation(exclude | (rel < 0.85), iterations=3)
    if clear.sum() < 20:                        # almost nothing clear: use the border only
        clear = np.zeros_like(exclude)
        clear[0] = clear[-1] = True
        clear[:, 0] = clear[:, -1] = True
    rows, cols = np.nonzero(clear)
    h, w = img.shape
    y, x = rows / max(h - 1, 1) - 0.5, cols / max(w - 1, 1) - 0.5
    terms = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if order >= 2 else [])
    A = np.column_stack(terms)
    coef, *_ = np.linalg.lstsq(A, img[rows, cols], rcond=None)
    Y, X = np.mgrid[0:h, 0:w]
    Y, X = Y / max(h - 1, 1) - 0.5, X / max(w - 1, 1) - 0.5
    full = [np.ones_like(X), X, Y] + ([X * X, X * Y, Y * Y] if order >= 2 else [])
    return np.maximum(sum(c * t for c, t in zip(coef, full)), 1.0)


class BackgroundBank:
    """The folder's full-frame backgrounds, for finding a clean crop."""

    def __init__(self, folder: str):
        self.times = np.zeros(0)
        self.frames = []
        path = os.path.join(folder, "full_backgrounds.mat")
        if not os.path.exists(path):
            return
        import scipy.io as sio
        full = sio.loadmat(path, variable_names=["FULL_BG"]).get("FULL_BG")
        if full is None or full.size == 0:
            return
        try:
            self.times = np.asarray(full["Time"][0, 0][:, 0], float)
            images = full[0, 0]["IMAGE"][0, :]
            self.frames = [np.asarray(images[j][0, 0]["BG"]) for j in range(len(self.times))]
        except (IndexError, KeyError, ValueError, TypeError):
            self.times, self.frames = np.zeros(0), []
        if len(self.frames):                    # only the backgrounds selected for use (if a selection is in force)
            try:
                from backgroundSelection import active_mask
            except ImportError:  # pragma: no cover
                from .backgroundSelection import active_mask
            use = active_mask(self.times)
            if use is not None:
                self.times = self.times[use]
                self.frames = [f for f, u in zip(self.frames, use) if u]

    def __len__(self):
        return len(self.frames)

    def clean_crop(self, time: float, xl: int, xu: int, yl: int, yu: int, mask: np.ndarray,
                   prepare=None, max_tries: int = 60) -> Tuple[Optional[np.ndarray], float]:
        """Nearest-in-time background whose crop does not contain the particle: (crop, |dt| in s)."""
        if not self.frames:
            return None, float("nan")
        order = np.argsort(np.abs(self.times - time))[:max_tries]
        for j in order:
            frame = self.frames[j]
            if xu > frame.shape[0] or yu > frame.shape[1]:
                continue
            crop = frame[xl:xu, yl:yu].astype(float)
            if prepare is not None:
                crop = prepare(crop)
            if crop.shape == mask.shape and not contains_particle(crop, mask):
                return crop, abs(float(self.times[j] - time)) * 86400.0
        return None, float("nan")


def choose_background(image: np.ndarray, background: np.ndarray, bank: Optional[BackgroundBank],
                      time: float, corners: Tuple[int, int, int, int], prepare=None):
    """(background to use, source code, |dt| s) for one ROI; image/background already prepared."""
    mask = particle_mask(image)
    if not mask.any() or not contains_particle(background, mask):
        return background, SOURCE_ASSIGNED, 0.0
    if bank is not None and len(bank):
        xl, xu, yl, yu = corners
        crop, dt_s = bank.clean_crop(time, xl, xu, yl, yu, mask, prepare=prepare)
        if crop is not None:
            return crop, SOURCE_OTHER_FRAME, dt_s
    return fitted_background(image, mask), SOURCE_FITTED, float("nan")
