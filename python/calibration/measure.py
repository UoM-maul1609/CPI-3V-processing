"""Images to measure on (full frames, ROIs, image files) and the measurements themselves.

Used by the measuring window (measure_window.py).  CPI images are shown the way
the camera sees them -- camera X across, Y down, the same way round as the
workbench's Camera X-Y plot -- so the arrays here are the transpose of the
stored ROI/background arrays (stored [x, y]).  Coordinates are pixel
coordinates of the image as displayed: x = column = camera x, y = row = camera y
(relative to the ROI's corner for an ROI; absolute for full and rebuilt frames),
with pixel (row r, column c) covering x in [c, c+1), y in [r, r+1), so a pixel
centre is at (c + 0.5, r + 0.5).  Lengths are in pixels at any angle (pixels assumed square);
times a pixel size they are in µm.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from .data import matlab_datenum_to_datetime
except ImportError:  # pragma: no cover
    from data import matlab_datenum_to_datetime

FULL_FRAME_TYPES = (19, 33795)          # CPI 2 / CPI 1 full-frame (background) image types
FULL_FRAME_MIN_PX = 512                  # or anything at least this big in both directions


def _time_text(t) -> str:
    try:
        t = float(t)
        if not math.isfinite(t) or t <= 0:
            return ""
        return matlab_datenum_to_datetime(t).isoformat(sep=" ", timespec="milliseconds")
    except (TypeError, ValueError, OverflowError):
        return ""


@dataclass
class ImageEntry:
    source: str                # file path
    kind: str                  # "full frame", "ROI", "background", "image file"
    index: int                 # ROI index / background index / 0
    shape: tuple
    time: str = ""
    origin: Optional[tuple] = None   # camera (x, y) of the image's top-left pixel corner; None for image files

    @property
    def key(self):
        return (self.source, self.kind if self.kind != "full frame" else "ROI", self.index)

    def title(self) -> str:
        name = Path(self.source).name
        if self.kind == "image file":
            return name
        what = {"background": "background", "full frame": "ROI (full frame)", "ROI": "ROI",
                "frame": "camera frame"}[self.kind]
        return f"{name} · {what} {self.index}" + (f" · {self.time}" if self.time else "")


def _as_list(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


class MatImages:
    """The ROIs (including full frames) of one processed MAT file, and their stored boundaries."""

    def __init__(self, path):
        import scipy.io as sio
        self.path = str(path)
        data = sio.loadmat(self.path, variable_names=["ROI_N", "dat"], simplify_cells=True)
        roi = data.get("ROI_N") or {}
        self.images = [np.asarray(item.get("IM")) if isinstance(item, dict) else np.zeros((0, 0))
                       for item in _as_list(roi.get("IMAGE"))]
        n = len(self.images)
        self.times = np.ravel(np.asarray(roi.get("Time", np.full(n, np.nan)), float))
        types = roi.get("imageType")
        self.types = np.ravel(np.asarray(types, float)) if types is not None else np.full(n, np.nan)
        self._dat = data.get("dat") if isinstance(data.get("dat"), dict) else None
        corner = lambda f: np.ravel(np.asarray(roi.get(f, np.full(n, -1)), float)).astype(int)
        self.start_x, self.end_x = corner("StartX"), corner("EndX")
        self.start_y, self.end_y = corner("StartY"), corner("EndY")
        self._frames = None

    # ---- whole camera frames rebuilt from their ROIs
    def _is_full(self, i) -> bool:
        im = self.images[i]
        return bool((i < len(self.types) and self.types[i] in FULL_FRAME_TYPES) or
                    (im.ndim == 2 and min(im.shape) >= FULL_FRAME_MIN_PX))

    def frames(self):
        """[(time, [ROI indices])] of the camera frames that have particle ROIs, in time order."""
        if self._frames is None:
            groups = {}
            for i in range(len(self.images)):
                if self._is_full(i) or i >= len(self.times) or not np.isfinite(self.times[i]) or self.start_x[i] < 0:
                    continue
                groups.setdefault(round(float(self.times[i]) * 86400.0, 3), []).append(i)
            self._frames = [(k / 86400.0, v) for k, v in sorted(groups.items())]
        return self._frames

    def frame_entries(self) -> List[ImageEntry]:
        return [ImageEntry(self.path, "frame", k, (0, 0), _time_text(t), (0, 0)) for k, (t, _) in enumerate(self.frames())]

    def frame_image(self, k: int, backgrounds: "Optional[BackgroundImages]" = None):
        """(image, background index or None): camera frame k with its ROIs pasted on the nearest background.

        Built in the stored [x, y] layout, returned in camera orientation (transposed)."""
        t, rois = self.frames()[k]
        bg_index = None
        if backgrounds is not None and backgrounds.images:
            bt = np.asarray(backgrounds.times, float)
            bg_index = int(np.nanargmin(np.abs(bt - t))) if np.isfinite(bt).any() else 0
            frame = np.asarray(backgrounds.images[bg_index], float).copy()     # stored [x, y]
        else:
            shape = (max(int(self.end_x[i]) for i in rois) + 1, max(int(self.end_y[i]) for i in rois) + 1)
            fulls = [im for j, im in enumerate(self.images) if self._is_full(j)]
            shape = max([shape] + [im.shape for im in fulls], key=lambda s: s[0] * s[1])
            frame = np.full(shape, 128.0)
        for i in rois:
            xl, xu, yl, yu = self.start_x[i], self.end_x[i] + 1, self.start_y[i], self.end_y[i] + 1
            im = self.images[i]
            if xu <= frame.shape[0] and yu <= frame.shape[1] and im.shape == (xu - xl, yu - yl):
                frame[xl:xu, yl:yu] = im
        return frame.T, bg_index

    def frame_boundary(self, k: int):
        """ROI boxes and stored outlines of camera frame k, in display (row, col) = (camera y, camera x), NaN-separated."""
        parts = []
        for i in self.frames()[k][1]:
            xl, xu, yl, yu = self.start_x[i], self.end_x[i] + 1, self.start_y[i], self.end_y[i] + 1
            box = np.array([[xl, yl], [xl, yu], [xu, yu], [xu, yl], [xl, yl]], float) - 0.5   # pixel edges
            parts += [box, [[np.nan, np.nan]]]
            b = self._stored_boundary(i)
            if b is not None:
                parts += [b + [xl, yl], [[np.nan, np.nan]]]
        return np.vstack(parts)[:, ::-1] if parts else None

    def entries(self) -> List[ImageEntry]:
        out = []
        for i, im in enumerate(self.images):
            full = (i < len(self.types) and self.types[i] in FULL_FRAME_TYPES) or \
                (im.ndim == 2 and min(im.shape) >= FULL_FRAME_MIN_PX)
            origin = (int(self.start_x[i]), int(self.start_y[i])) if i < len(self.start_x) and self.start_x[i] >= 0 \
                else None
            out.append(ImageEntry(self.path, "full frame" if full else "ROI", i, tuple(im.shape[::-1]),
                                  _time_text(self.times[i]) if i < len(self.times) else "", origin))
        return out

    def image(self, index: int) -> np.ndarray:
        """ROI ``index`` in camera orientation (rows = camera y, columns = camera x)."""
        return self.images[index].T

    def boundary(self, index: int) -> Optional[np.ndarray]:
        """Stored outline in display (row, col) = (camera y, camera x), if there is one."""
        b = self._stored_boundary(index)
        return None if b is None else b[:, ::-1]

    def _stored_boundary(self, index: int) -> Optional[np.ndarray]:
        """Stored outline as saved by the processing: (array row, array col) = (camera x, camera y)."""
        if self._dat is None:
            return None
        foc = _as_list(self._dat.get("foc"))
        if index >= len(foc) or not isinstance(foc[index], dict) or "boundaries" not in foc[index]:
            return None
        b = np.asarray(foc[index]["boundaries"], float)
        if b.ndim == 2 and b.shape[1] == 2 and np.isfinite(b).any():
            return b
        return None


class BackgroundImages:
    """The full-frame backgrounds in a folder's full_backgrounds.mat."""

    def __init__(self, path):
        import scipy.io as sio
        self.path = str(path)
        full = sio.loadmat(self.path, variable_names=["FULL_BG"], simplify_cells=True).get("FULL_BG") or {}
        self.images = [np.asarray(item.get("BG")) for item in _as_list(full.get("IMAGE")) if isinstance(item, dict)]
        self.times = np.ravel(np.asarray(full.get("Time", np.full(len(self.images), np.nan)), float))

    def entries(self) -> List[ImageEntry]:
        return [ImageEntry(self.path, "background", i, tuple(im.shape[::-1]),
                           _time_text(self.times[i]) if i < len(self.times) else "", (0, 0))
                for i, im in enumerate(self.images)]

    def image(self, index: int) -> np.ndarray:
        """Background ``index`` in camera orientation (rows = camera y, columns = camera x)."""
        return self.images[index].T

    def boundary(self, index: int):
        return None


def read_image_file(path) -> np.ndarray:
    """Grey-level array of an ordinary image file (PNG, TIFF, BMP, JPEG...)."""
    from PySide6.QtGui import QImage
    img = QImage(str(path))
    if img.isNull():
        raise ValueError(f"Cannot read image {path}")
    img = img.convertToFormat(QImage.Format_Grayscale8)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=h * bpl).reshape(h, bpl)
    return buf[:, :w].copy()


# --------------------------------------------------------------------------- measurements
@dataclass
class Measurement:
    x1: float
    y1: float
    x2: float
    y2: float
    note: str = ""

    @property
    def length_px(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self) -> float:
        """Direction from the first to the second point, anticlockwise from +x as seen on screen (y down)."""
        return math.degrees(math.atan2(-(self.y2 - self.y1), self.x2 - self.x1))

    def length_um(self, pixel_um: Optional[float]) -> float:
        return self.length_px * pixel_um if pixel_um else float("nan")


def constrain(x1, y1, x2, y2, step_deg=45.0):
    """Second point moved onto the nearest multiple of ``step_deg`` from the first (same length)."""
    dx, dy = x2 - x1, y2 - y1
    r = math.hypot(dx, dy)
    if r == 0:
        return x2, y2
    a = round(math.atan2(dy, dx) / math.radians(step_deg)) * math.radians(step_deg)
    return x1 + r * math.cos(a), y1 + r * math.sin(a)


def profile(image: np.ndarray, m: Measurement, step: float = 0.25, extend: float = 0.0):
    """Brightness along a measurement line: (distance from its first point [px], value).

    ``extend`` px beyond each end are included (negative / > length distances).
    Bilinear interpolation between pixel centres.
    """
    from scipy import ndimage
    L = m.length_px
    if L <= 0:
        return np.zeros(0), np.zeros(0)
    s = np.arange(-extend, L + extend + 1e-9, step)
    ux, uy = (m.x2 - m.x1) / L, (m.y2 - m.y1) / L
    x = m.x1 + s * ux - 0.5            # to array index coordinates (pixel centres at integers)
    y = m.y1 + s * uy - 0.5
    v = ndimage.map_coordinates(np.asarray(image, float), [y, x], order=1, mode="nearest")
    return s, v


def nice_bar_length(max_um: float) -> float:
    """1, 2 or 5 × 10^n not exceeding ``max_um``."""
    if max_um <= 0:
        return 0.0
    p = 10 ** math.floor(math.log10(max_um))
    for f in (5, 2, 1):
        if f * p <= max_um:
            return f * p
    return p


CSV_FIELDS = ["source_file", "image", "index", "time", "image_width_px", "image_height_px", "number",
              "x1_px", "y1_px", "x2_px", "y2_px", "x1_camera_px", "y1_camera_px", "x2_camera_px", "y2_camera_px",
              "length_px", "angle_deg", "pixel_um", "length_um", "note"]


def measurement_rows(entry: ImageEntry, measurements, pixel_um: Optional[float]):
    h, w = (entry.shape + (0, 0))[:2]
    for n, m in enumerate(measurements, 1):
        yield {"source_file": entry.source, "image": entry.kind, "index": entry.index, "time": entry.time,
               "image_width_px": w, "image_height_px": h, "number": n,
               "x1_px": round(m.x1, 3), "y1_px": round(m.y1, 3), "x2_px": round(m.x2, 3), "y2_px": round(m.y2, 3),
               **({"x1_camera_px": round(entry.origin[0] + m.x1, 3), "y1_camera_px": round(entry.origin[1] + m.y1, 3),
                   "x2_camera_px": round(entry.origin[0] + m.x2, 3), "y2_camera_px": round(entry.origin[1] + m.y2, 3)}
                  if entry.origin is not None else {}),
               "length_px": round(m.length_px, 3), "angle_deg": round(m.angle_deg, 2),
               "pixel_um": pixel_um or "", "length_um": round(m.length_um(pixel_um), 3) if pixel_um else "",
               "note": m.note}


def write_csv(path, rows) -> int:
    rows = list(rows)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
