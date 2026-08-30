"""Data access and persistent selection state for CPI calibration.

The GUI deliberately keeps only one processed MAT file in memory at a time.
Catalog construction can be run in disposable subprocesses, so scanning many
files does not cause the GUI process RSS to grow with SciPy/NumPy allocations.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio


EXCLUDED_MAT_FILES = {"timeseries.mat", "full_backgrounds.mat"}


def matlab_datenum_to_datetime(value: float) -> datetime:
    day = datetime.fromordinal(int(value))
    return day + timedelta(days=float(value) % 1.0) - timedelta(days=366)


def datetime_to_matlab_datenum(value: datetime) -> float:
    shifted = value + timedelta(days=366)
    midnight = datetime(value.year, value.month, value.day)
    frac = (value - midnight).total_seconds() / 86400.0
    return float(shifted.toordinal()) + frac


def _safe_float(value, default=float("nan")) -> float:
    try:
        result = float(np.asarray(value).squeeze())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _vector(value, length: int, default=float("nan")) -> np.ndarray:
    if value is None:
        return np.full(length, default, dtype=float)
    arr = np.asarray(value).reshape(-1)
    out = np.full(length, default, dtype=float)
    ncopy = min(length, arr.size)
    if ncopy:
        with np.errstate(invalid="ignore"):
            out[:ncopy] = arr[:ncopy].astype(float, copy=False)
    return out


def _struct_list(value, length: int) -> List[dict]:
    if isinstance(value, dict):
        values = [value]
    elif isinstance(value, np.ndarray):
        values = list(value.reshape(-1))
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    result: List[dict] = []
    for item in values[:length]:
        result.append(item if isinstance(item, dict) else {})
    result.extend({} for _ in range(length - len(result)))
    return result


@dataclass
class ParticleRecord:
    source_file: str
    roi_index: int
    time_matlab: float
    length_um: float
    width_um: float
    focus: float
    roundness: float
    image_x_px: float
    image_y_px: float
    centroid_row_px: float
    centroid_col_px: float
    stage_x: float = float("nan")
    stage_y: float = float("nan")
    status: str = "undecided"
    object_plane_um: float = 0.0

    @property
    def particle_id(self) -> str:
        return "{}#{}".format(self.source_file, self.roi_index)

    @property
    def time_iso(self) -> str:
        if not math.isfinite(self.time_matlab):
            return ""
        return matlab_datenum_to_datetime(self.time_matlab).isoformat(sep=" ", timespec="milliseconds")


@dataclass
class FilterSpec:
    length_min: Optional[float] = None
    length_max: Optional[float] = None
    width_min: Optional[float] = None
    width_max: Optional[float] = None
    focus_min: Optional[float] = None
    focus_max: Optional[float] = None
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    time_min: Optional[float] = None
    time_max: Optional[float] = None
    status: Optional[str] = None

    def matches(self, record: ParticleRecord) -> bool:
        checks = (
            (record.length_um, self.length_min, self.length_max),
            (record.width_um, self.width_min, self.width_max),
            (record.focus, self.focus_min, self.focus_max),
            (record.image_x_px, self.x_min, self.x_max),
            (record.image_y_px, self.y_min, self.y_max),
            (record.time_matlab, self.time_min, self.time_max),
        )
        for value, lower, upper in checks:
            if lower is not None and (not math.isfinite(value) or value < lower):
                return False
            if upper is not None and (not math.isfinite(value) or value > upper):
                return False
        if self.status and self.status != "all" and record.status != self.status:
            return False
        return True


class PositionLog:
    """Calibration stage positions sampled in time."""

    def __init__(self, times: np.ndarray, x: np.ndarray, y: np.ndarray):
        order = np.argsort(times)
        self.times = np.asarray(times, dtype=float)[order]
        self.x = np.asarray(x, dtype=float)[order]
        self.y = np.asarray(y, dtype=float)[order]

    @classmethod
    def from_text(cls, filename: os.PathLike) -> "PositionLog":
        times: List[float] = []
        xs: List[float] = []
        ys: List[float] = []
        with open(filename, "r", encoding="utf-8") as handle:
            rows = handle.readlines()
        for row in rows[1:]:
            parts = row.split()
            if len(parts) < 4:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
                timestamp = datetime.strptime(parts[2] + " " + parts[3], "%d/%m/%y %H:%M:%S")
            except ValueError:
                continue
            times.append(datetime_to_matlab_datenum(timestamp))
        if not times:
            raise ValueError("No valid position records found in {}".format(filename))
        return cls(np.asarray(times), np.asarray(xs), np.asarray(ys))

    def previous(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        query = np.asarray(times, dtype=float)
        index = np.searchsorted(self.times, query, side="right") - 1
        index = np.clip(index, 0, len(self.times) - 1)
        return self.x[index], self.y[index]


def list_processed_files(directory: os.PathLike) -> List[Path]:
    root = Path(directory)
    return sorted(
        p for p in root.glob("*.mat")
        if p.name not in EXCLUDED_MAT_FILES and not p.name.startswith(".")
    )


def scan_processed_file(filename: os.PathLike, root: Optional[os.PathLike] = None) -> List[ParticleRecord]:
    """Extract lightweight particle metadata from one processed MAT file."""
    path = Path(filename)
    loaded = sio.loadmat(str(path), variable_names=["ROI_N", "dat"], simplify_cells=True)
    if "ROI_N" not in loaded or "dat" not in loaded:
        return []
    roi = loaded["ROI_N"]
    dat = loaded["dat"]
    if not isinstance(roi, dict) or not isinstance(dat, dict):
        return []

    lengths_raw = np.asarray(dat.get("len", []))
    n = int(lengths_raw.size)
    if n == 0:
        return []

    lengths = _vector(dat.get("len"), n)
    widths = _vector(dat.get("wid"), n)
    rounds = _vector(dat.get("round"), n)
    times = _vector(dat.get("Time", roi.get("Time")), n)
    start_x = _vector(roi.get("StartX"), n)
    end_x = _vector(roi.get("EndX"), n)
    start_y = _vector(roi.get("StartY"), n)
    end_y = _vector(roi.get("EndY"), n)

    centroids = np.asarray(dat.get("centroid", np.full((n, 2), np.nan)), dtype=float)
    if centroids.ndim == 1 and centroids.size == 2 and n == 1:
        centroids = centroids.reshape(1, 2)
    if centroids.shape != (n, 2):
        temp = np.full((n, 2), np.nan)
        flat = centroids.reshape(-1, 2) if centroids.size >= 2 else np.empty((0, 2))
        temp[: min(n, len(flat))] = flat[:n]
        centroids = temp

    foc_structs = _struct_list(dat.get("foc"), n)
    focuses = np.asarray([_safe_float(item.get("focus")) for item in foc_structs])

    source = str(path.relative_to(root)) if root is not None else path.name
    records = []
    for i in range(n):
        records.append(
            ParticleRecord(
                source_file=source,
                roi_index=i,
                time_matlab=float(times[i]),
                length_um=float(lengths[i]),
                width_um=float(widths[i]),
                focus=float(focuses[i]),
                roundness=float(rounds[i]),
                image_x_px=float((start_x[i] + end_x[i]) * 0.5),
                image_y_px=float((start_y[i] + end_y[i]) * 0.5),
                centroid_row_px=float(centroids[i, 0]),
                centroid_col_px=float(centroids[i, 1]),
            )
        )
    return records


def _scan_via_subprocess(path: Path, root: Path) -> List[ParticleRecord]:
    fd, temporary = tempfile.mkstemp(prefix="cpi_catalog_", suffix=".json")
    os.close(fd)
    try:
        # Prefer the installed package entry point.  Fall back to the file path
        # when the module is being run directly from a source checkout.
        if __package__:
            command = [sys.executable, "-m", "calibration.data", "scan-one", str(path), str(root), temporary]
        else:
            command = [sys.executable, os.path.abspath(__file__), "scan-one", str(path), str(root), temporary]
        subprocess.run(command, check=True)
        with open(temporary, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        return [ParticleRecord(**row) for row in rows]
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class ParticleCatalog:
    def __init__(self, root: os.PathLike, records: Sequence[ParticleRecord]):
        self.root = Path(root).resolve()
        self.records = list(records)
        self._by_id = {record.particle_id: record for record in self.records}

    @classmethod
    def from_directory(
        cls,
        directory: os.PathLike,
        position_file: Optional[os.PathLike] = None,
        disposable_scanners: bool = True,
        progress: Optional[Callable[[int, int, Path], None]] = None,
    ) -> "ParticleCatalog":
        root = Path(directory).resolve()
        files = list_processed_files(root)
        records: List[ParticleRecord] = []
        for number, filename in enumerate(files, start=1):
            if progress:
                progress(number, len(files), filename)
            if disposable_scanners:
                records.extend(_scan_via_subprocess(filename, root))
            else:
                records.extend(scan_processed_file(filename, root=root))

        records.sort(key=lambda r: (r.time_matlab, r.source_file, r.roi_index))
        catalog = cls(root, records)
        if position_file:
            catalog.attach_position_log(PositionLog.from_text(position_file))
        return catalog

    def attach_position_log(self, position_log: PositionLog) -> None:
        if not self.records:
            return
        times = np.asarray([r.time_matlab for r in self.records], dtype=float)
        x, y = position_log.previous(times)
        for record, stage_x, stage_y in zip(self.records, x, y):
            record.stage_x = float(stage_x)
            record.stage_y = float(stage_y)

    def filtered_indices(self, spec: FilterSpec) -> List[int]:
        return [i for i, record in enumerate(self.records) if spec.matches(record)]

    def apply_selection_file(self, filename: os.PathLike) -> None:
        path = Path(filename)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record = self._by_id.get(row.get("particle_id", ""))
                if record is None:
                    continue
                status = row.get("status", "undecided")
                if status in {"undecided", "keep", "reject"}:
                    record.status = status
                try:
                    record.object_plane_um = float(row.get("object_plane_um", 0.0))
                except ValueError:
                    record.object_plane_um = 0.0

    def save_selection_file(self, filename: os.PathLike) -> None:
        fieldnames = [
            "particle_id", "source_file", "roi_index", "time_matlab", "time_iso",
            "length_um", "width_um", "focus", "roundness", "image_x_px", "image_y_px",
            "centroid_row_px", "centroid_col_px", "stage_x", "stage_y", "status",
            "object_plane_um",
        ]
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent))
        os.close(fd)
        try:
            with open(temporary, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.records:
                    row = asdict(record)
                    row["particle_id"] = record.particle_id
                    row["time_iso"] = record.time_iso
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class ParticleStore:
    """Lazy access to images/boundaries with a one-file cache."""

    def __init__(self, root: os.PathLike):
        self.root = Path(root).resolve()
        self._source: Optional[str] = None
        self._data: Optional[dict] = None

    def _load_source(self, source_file: str) -> dict:
        if self._source != source_file or self._data is None:
            self._data = sio.loadmat(
                str(self.root / source_file),
                variable_names=["ROI_N", "BG", "dat"],
                simplify_cells=True,
            )
            self._source = source_file
        return self._data

    def particle(self, record: ParticleRecord, subtract_background: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        data = self._load_source(record.source_file)
        roi = data["ROI_N"]
        images = _struct_list(roi.get("IMAGE"), record.roi_index + 1)
        if record.roi_index >= len(images) or "IM" not in images[record.roi_index]:
            raise IndexError("Particle image {} not present in {}".format(record.roi_index, record.source_file))
        image = np.asarray(images[record.roi_index]["IM"])

        if subtract_background and "BG" in data and isinstance(data["BG"], (list, tuple, np.ndarray, dict)):
            backgrounds = _struct_list(data["BG"], record.roi_index + 1)
            if record.roi_index < len(backgrounds) and "BG" in backgrounds[record.roi_index]:
                bg = np.asarray(backgrounds[record.roi_index]["BG"])
                if bg.shape == image.shape:
                    image = image.astype(np.int16) - bg.astype(np.int16)

        boundary = None
        dat = data.get("dat")
        if isinstance(dat, dict):
            foc = _struct_list(dat.get("foc"), record.roi_index + 1)
            if record.roi_index < len(foc) and "boundaries" in foc[record.roi_index]:
                candidate = np.asarray(foc[record.roi_index]["boundaries"], dtype=float)
                if candidate.ndim == 2 and candidate.shape[1] == 2 and np.isfinite(candidate).any():
                    boundary = candidate
        return image, boundary

    def clear(self) -> None:
        self._source = None
        self._data = None


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "scan-one":
        input_file = Path(sys.argv[2])
        root = Path(sys.argv[3])
        output_file = Path(sys.argv[4])
        rows = [asdict(record) for record in scan_processed_file(input_file, root=root)]
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, allow_nan=True)
    else:
        raise SystemExit("internal usage: data.py scan-one INPUT ROOT OUTPUT_JSON")
