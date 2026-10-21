"""Data access and persistent selection state for CPI calibration.

The GUI deliberately keeps only one processed MAT file in memory at a time.
Catalog construction can be run in disposable subprocesses, so scanning many
files does not cause the GUI process RSS to grow with SciPy/NumPy allocations.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio


EXCLUDED_MAT_FILES = {"timeseries.mat", "full_backgrounds.mat"}


def particle_key(time_matlab, image_x_px, image_y_px):
    """Processing-independent identity of a ROI: (time in ms, StartX+EndX, StartY+EndY).

    ``roi_index`` can change when files are reprocessed with different
    criteria, and length/focus change with the algorithm, but the ROI's
    acquisition time and its position on the camera come straight from the
    raw data.  ``image_x_px`` is the ROI centre (StartX+EndX)/2, so twice it is
    an exact integer.  Returns None if any value is missing.
    """
    try:
        t, x, y = float(time_matlab), float(image_x_px), float(image_y_px)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
        return None
    return (int(round(t * 86400000.0)), int(round(2 * x)), int(round(2 * y)))


@dataclass
class MatchReport:
    """Outcome of re-attaching saved flags to the loaded particles."""
    rows: int = 0
    same_id: int = 0          # matched, ROI index unchanged
    remapped: int = 0         # matched by time + position, ROI index changed
    by_id_only: int = 0       # legacy rows without time/position
    unmatched: int = 0        # particle not in the loaded files
    ambiguous: int = 0        # several loaded ROIs share time + position
    id_map: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"{self.same_id + self.remapped + self.by_id_only:,} of {self.rows:,} saved particles matched"]
        if self.remapped:
            parts.append(f"{self.remapped:,} re-identified by time + camera position (ROI numbering changed)")
        if self.unmatched:
            parts.append(f"{self.unmatched:,} not in the loaded files")
        if self.ambiguous:
            parts.append(f"{self.ambiguous:,} ambiguous (skipped)")
        return "; ".join(parts)


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
    reject_reason: str = ""
    object_plane_um: float = 0.0
    actual_length_um: float = float("nan")
    actual_width_um: float = float("nan")
    instrument: str = "CPI3v"
    is_full_frame: bool = False
    label: str = ""  # name in the shared label library (bead type, ice analogue …)
    local_illum: float = float("nan")
    detected_cpi: str = ""  # "CPI1.0" / "CPI3v" from the file's image-type codes (never from user edits)  # median brightness of the ROI's 2-px border (local illumination)
    start_x: float = float("nan")
    end_x: float = float("nan")
    start_y: float = float("nan")
    end_y: float = float("nan")
    focus_T: float = float("nan")      # focus on transmittance (new edge method)
    deq_um: float = float("nan")       # equivalent-area diameter (new edge method)
    edge_touch: int = -1               # 1 = particle reaches the ROI border (size is a lower limit); -1 unknown
    processing: str = ""               # short label of the edge method that produced len/focus
    pixel_um: float = 2.3              # pixel size the file was processed with (dat.proc_options; 2.3 if unrecorded)
    # CPIview's own measures (batchProcess --cpiview, proc/cpiviewMetrics.py); NaN if not computed
    cpv_focus: float = float("nan")         # CPIview focus (mean of 16 boundary samples)
    cpv_focus_sd: float = float("nan")      # spread of those samples
    cpv_fwhm_px: float = float("nan")       # median edge width (FWHM of the gradient across the edge)
    cpv_distinct: float = float("nan")      # fraction of boundary pixels with a strong gradient
    cpv_cutoff_pct: float = float("nan")    # % of boundary pixels on the ROI edge
    cutoff_pct: float = float("nan")        # % of this processing's particle outline on the ROI edge (NaN: unknown)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in ("source_file", "roi_index"):
            self.__dict__.pop("particle_id", None)   # the cached ID follows a renamed source
        elif name in ("time_matlab", "image_x_px", "image_y_px"):
            self.__dict__.pop("_key", None)

    @functools.cached_property
    def particle_id(self) -> str:
        # Built once and then read as a plain attribute: it is asked for millions of times.
        return "{}#{}".format(self.source_file, self.roi_index)

    @property
    def match_key(self):
        """Stable identity that survives reprocessing (see :func:`particle_key`)."""
        if "_key" not in self.__dict__:
            self.__dict__["_key"] = particle_key(self.time_matlab, self.image_x_px, self.image_y_px)
        return self.__dict__["_key"]

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
    ratio_min: Optional[float] = None
    ratio_max: Optional[float] = None
    illum_min: Optional[float] = None
    illum_max: Optional[float] = None
    cutoff_min: Optional[float] = None      # % of the outline on the ROI edge; unknown (NaN) always passes
    cutoff_max: Optional[float] = None

    def matches(self, record: ParticleRecord) -> bool:
        checks = (
            (record.length_um, self.length_min, self.length_max),
            (record.width_um, self.width_min, self.width_max),
            (record.focus, self.focus_min, self.focus_max),
            (record.image_x_px, self.x_min, self.x_max),
            (record.image_y_px, self.y_min, self.y_max),
            (record.time_matlab, self.time_min, self.time_max),
            (record.local_illum, self.illum_min, self.illum_max),
        )
        for value, lower, upper in checks:
            if lower is not None and (not math.isfinite(value) or value < lower):
                return False
            if upper is not None and (not math.isfinite(value) or value > upper):
                return False
        cut = getattr(record, "cutoff_pct", float("nan"))
        if math.isfinite(cut):
            if (self.cutoff_min is not None and cut < self.cutoff_min) or \
                    (self.cutoff_max is not None and cut > self.cutoff_max):
                return False
        if self.ratio_min is not None or self.ratio_max is not None:
            ratio = record.length_um / record.actual_length_um if record.actual_length_um > 0 else float("nan")
            if not math.isfinite(ratio) or (self.ratio_min is not None and ratio < self.ratio_min) or (self.ratio_max is not None and ratio > self.ratio_max):
                return False
        if self.status and self.status != "all" and record.status != self.status:
            return False
        return True


class PositionLog:
    """Calibration stage positions sampled in time."""

    def __init__(self, times: np.ndarray, x: np.ndarray, y: np.ndarray, ends: Optional[np.ndarray] = None):
        """``ends``: exclusive end time of each position (NaN = until the next one starts)."""
        order = np.argsort(times, kind="stable")
        self.times = np.asarray(times, dtype=float)[order]
        self.x = np.asarray(x, dtype=float)[order]
        self.y = np.asarray(y, dtype=float)[order]
        self.ends = (np.full(len(self.times), np.nan) if ends is None
                     else np.asarray(ends, dtype=float)[order])

    @classmethod
    def from_text(cls, filename: os.PathLike, series: Optional[str] = None,
                  time_basis: Optional[str] = "roi") -> "PositionLog":
        """Read historical stage logs or consolidated interval CSV (zero-order hold).

        For consolidated CSV, limit to one series and time basis to avoid
        overlapping full-frame/ROI intervals masquerading as stage changes.
        The original instrument clock timezone is not inferred.
        """
        path = Path(filename)
        times: List[float] = []
        xs: List[float] = []
        ys: List[float] = []
        ends: List[float] = []
        if path.suffix.lower() in (".csv", ".tsv"):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                sample = handle.read(2048); handle.seek(0)
                dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
                reader = csv.DictReader(handle, dialect=dialect)
                fields = set(reader.fieldnames or [])
                if not {"stage_x_original", "stage_z_original", "start_time"}.issubset(fields):
                    raise ValueError("CSV needs stage_x_original, stage_z_original and start_time columns")
                for row in reader:
                    # series / time_basis = None means "use every row".  The particle
                    # type is decided later, by flagging and labelling particles.
                    if series is not None and row.get("series", series) != series:
                        continue
                    if time_basis is not None and row.get("time_basis", time_basis) != time_basis:
                        continue
                    try:
                        stamp = datetime.fromisoformat(row["start_time"].strip())
                        x_value, z_value = float(row["stage_x_original"]), float(row["stage_z_original"])
                        end_text = (row.get("end_time_exclusive") or "").strip()
                        end = datetime_to_matlab_datenum(datetime.fromisoformat(end_text)) if end_text else float("nan")
                        xs.append(x_value); ys.append(z_value)
                        times.append(datetime_to_matlab_datenum(stamp))
                        ends.append(end)
                    except (ValueError, TypeError):
                        continue
        else:
            with path.open("r", encoding="utf-8-sig") as handle:
                for row in list(handle)[1:]:
                    parts = row.split()
                    if len(parts) < 4:
                        continue
                    try:
                        xx, zz = float(parts[0]), float(parts[1])
                        stamp = " ".join(parts[2:4])
                        parsed = None
                        for fmt in ("%d/%m/%y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                            try:
                                parsed = datetime.strptime(stamp, fmt)
                                break
                            except ValueError:
                                pass
                        if parsed is not None:
                            xs.append(xx); ys.append(zz)
                            times.append(datetime_to_matlab_datenum(parsed))
                            ends.append(float("nan"))
                    except (ValueError, IndexError):
                        continue
        if not times:
            raise ValueError(f"No valid position records in {filename} for "
                             f"{series or 'all series'}/{time_basis or 'all time bases'}")
        return cls(np.asarray(times), np.asarray(xs), np.asarray(ys), np.asarray(ends))

    @staticmethod
    def csv_choices(filename: os.PathLike) -> Dict[str, List[str]]:
        """Distinct ``series`` and ``time_basis`` values in a consolidated stage CSV."""
        path = Path(filename)
        found: Dict[str, set] = {"series": set(), "time_basis": set()}
        if path.suffix.lower() not in (".csv", ".tsv"):
            return {k: [] for k in found}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
            for row in csv.DictReader(handle, dialect=dialect):
                for key in found:
                    value = (row.get(key) or "").strip()
                    if value:
                        found[key].add(value)
        return {k: sorted(v) for k, v in found.items()}

    def conflicting_times(self) -> int:
        """Number of identical timestamps that carry different stage positions."""
        if len(self.times) < 2:
            return 0
        same = np.diff(self.times) == 0
        moved = (np.diff(self.x) != 0) | (np.diff(self.y) != 0)
        return int(np.count_nonzero(same & moved))

    def previous(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stage position at each time: the latest position started at or before it.

        If that position has an end time (CSV ``end_time_exclusive``) and the time
        is at or after it, the stage was between positions: NaN.
        """
        query = np.asarray(times, dtype=float)
        index = np.searchsorted(self.times, query, side="right") - 1
        valid = index >= 0
        index = np.clip(index, 0, len(self.times) - 1)
        end = self.ends[index]
        valid &= ~(np.isfinite(end) & (query >= end))
        return np.where(valid, self.x[index], np.nan), np.where(valid, self.y[index], np.nan)


def stage_burst_check(times, position_log: "PositionLog", gap_s: float = 2.0) -> Dict[str, int]:
    """How well a stage log fits the acquisition: particles come in bursts, one per stage position.

    Bursts are runs of particle times without a gap longer than ``gap_s``.  A burst
    that gets two different stage positions usually means the log's times are
    off (wrong time basis, or a time rounded into the burst).
    """
    t = np.sort(np.asarray([v for v in times if math.isfinite(v)], float))
    out = {"bursts": 0, "split": 0, "particles_in_split": 0, "no_position": 0}
    if not len(t):
        return out
    x, y = position_log.previous(t)
    out["no_position"] = int(np.count_nonzero(~(np.isfinite(x) & np.isfinite(y))))
    breaks = np.flatnonzero(np.diff(t) * 86400.0 > gap_s)
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(t) - 1]
    out["bursts"] = int(len(starts))
    for a, b in zip(starts, ends):
        pairs = {(p, q) for p, q in zip(x[a:b + 1], y[a:b + 1]) if math.isfinite(p) and math.isfinite(q)}
        if len(pairs) > 1:
            out["split"] += 1
            out["particles_in_split"] += int(b - a + 1)
    return out


def list_processed_files(directory: os.PathLike) -> List[Path]:
    root = Path(directory)
    return sorted(
        p for p in root.glob("*.mat")
        if p.name not in EXCLUDED_MAT_FILES and not p.name.startswith(".")
    )


# Particle ROI image-type codes written by the two instruments (see proc/imageStats.py).
CPI_PARTICLE_TYPES = {89: "CPI3v", 33857: "CPI1.0"}
CPI_DISPLAY = {"CPI3v": "CPI-3V", "CPI1.0": "CPI 1.0"}


def detect_cpi_type(image_types) -> str:
    """'CPI3v' or 'CPI1.0' from a file's ROI image-type codes ('' if neither is present)."""
    values = np.asarray(image_types).ravel() if image_types is not None else np.array([])
    counts = {name: int(np.count_nonzero(values == code)) for code, name in CPI_PARTICLE_TYPES.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else ""


def roi_edge_brightness(image, width: int = 2) -> float:
    """Local illumination of a ROI: median of its outer ``width``-pixel frame.

    CPI ROIs are padded, so the frame is mostly background around the
    particle, imaged at the same moment and place.  It therefore follows both
    the beam profile across the camera and laser-power drift in time, which
    scale the edge-gradient focus metric.
    """
    im = np.asarray(image)
    if im.ndim != 2 or min(im.shape) < 2 * width + 1:
        return float("nan")
    w = width
    frame = np.concatenate((im[:w].ravel(), im[-w:].ravel(), im[w:-w, :w].ravel(), im[w:-w, -w:].ravel()))
    return float(np.median(frame.astype(float)))


def processing_label(proc_options) -> str:
    """Short name of the edge method recorded in dat.proc_options ("legacy" for files without one)."""
    if proc_options is None or (hasattr(proc_options, "size") and np.size(proc_options) == 0):
        return "legacy (unrecorded)"
    try:
        opts = json.loads(str(proc_options if isinstance(proc_options, str) else np.asarray(proc_options).ravel()[0]))
    except (ValueError, TypeError, IndexError):
        return "unknown"
    if opts.get("method") == "transmittance":
        if opts.get("edge_mode") == "relative":
            return f"transmittance relative f={opts.get('edge_fraction')} denoise={opts.get('denoise')}"
        return f"transmittance T={opts.get('edge_level')} denoise={opts.get('denoise')}"
    return str(opts.get("method", "unknown"))


def processing_pixel_um(proc_options, default: float = 2.3) -> float:
    """Pixel size (µm) recorded in dat.proc_options; ``default`` (the nominal 2.3) for older files."""
    if proc_options is None or (hasattr(proc_options, "size") and np.size(proc_options) == 0):
        return default
    try:
        opts = json.loads(str(proc_options if isinstance(proc_options, str) else np.asarray(proc_options).ravel()[0]))
        value = float(opts.get("pixel_um", default))
    except (ValueError, TypeError, IndexError, AttributeError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def scan_processed_file(filename: os.PathLike, root: Optional[os.PathLike] = None) -> List[ParticleRecord]:
    """Extract lightweight particle metadata from one processed MAT file."""
    path = Path(filename)
    loaded = sio.loadmat(str(path), variable_names=["ROI_N", "dat"], simplify_cells=True)
    if "ROI_N" in loaded and "dat" not in loaded:
        raise ValueError(
            "no particle statistics (dat) in this file: its processing did not finish (backgrounds not assigned / "
            "statistics not computed). Finish it with: python batchProcess.py <folder> --only-stats "
            "--backgrounds all [--cpiv1] [edge options]")
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
    # Always use ROI_N.Time: dat.Time may represent a different acquisition clock.
    times = _vector(roi.get("Time"), n)
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

    detected = detect_cpi_type(roi.get("imageType"))
    image_structs = _struct_list(roi.get("IMAGE"), n)
    illum = [roi_edge_brightness(item.get("IM")) if "IM" in item else float("nan") for item in image_structs]

    foc_structs = _struct_list(dat.get("foc"), n)
    focuses = np.asarray([_safe_float(item.get("focus")) for item in foc_structs])
    focus_T = np.asarray([_safe_float(item.get("focus_T")) for item in foc_structs])
    edge_touch = _vector(dat.get("edge_touch"), n, default=-1)
    deqs = _vector(dat.get("deq"), n)
    cutoffs = _vector(dat.get("cutoff_pct"), n)
    cpv = {k: _vector(dat.get(k), n) for k in ("cpv_focus", "cpv_focus_sd", "cpv_fwhm_px", "cpv_distinct",
                                               "cpv_cutoff_pct")}
    processing = processing_label(dat.get("proc_options"))
    pixel_um = processing_pixel_um(dat.get("proc_options"))

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
                is_full_frame=bool((end_x[i] - start_x[i] >= 900) and (end_y[i] - start_y[i] >= 900)),
                local_illum=float(illum[i]),
                detected_cpi=detected,
                start_x=float(start_x[i]), end_x=float(end_x[i]),
                start_y=float(start_y[i]), end_y=float(end_y[i]),
                focus_T=float(focus_T[i]),
                deq_um=float(deqs[i]),
                cpv_focus=float(cpv["cpv_focus"][i]), cpv_focus_sd=float(cpv["cpv_focus_sd"][i]),
                cpv_fwhm_px=float(cpv["cpv_fwhm_px"][i]), cpv_distinct=float(cpv["cpv_distinct"][i]),
                cpv_cutoff_pct=float(cpv["cpv_cutoff_pct"][i]),
                cutoff_pct=float(cutoffs[i]) if math.isfinite(cutoffs[i]) else float(cpv["cpv_cutoff_pct"][i]),
                edge_touch=int(edge_touch[i]) if np.isfinite(edge_touch[i]) else -1,
                processing=processing,
                pixel_um=pixel_um,
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
        result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            lines = [line for line in (result.stderr or "").strip().splitlines() if line.strip()]
            message = lines[-1] if lines else f"scanner exited with status {result.returncode}"
            raise ValueError(message.split(": ", 1)[1] if message.startswith(("ValueError", "OSError")) else message)
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

    def key_index(self) -> Dict[tuple, List[ParticleRecord]]:
        index: Dict[tuple, List[ParticleRecord]] = {}
        for record in self.records:
            key = record.match_key
            if key is not None:
                index.setdefault(key, []).append(record)
        return index

    @staticmethod
    def resolve_key(candidates, row=None):
        """Pick one record among ROIs sharing a key, or None if still ambiguous."""
        if len(candidates) <= 1 or row is None:
            return candidates[0] if len(candidates) == 1 else None
        narrowed = list(candidates)
        for fields in (("start_x", "end_x", "start_y", "end_y"),):
            try:
                wanted = tuple(float(row[f]) for f in fields)
            except (KeyError, TypeError, ValueError):
                continue
            hit = [r for r in narrowed if tuple(getattr(r, f) for f in fields) == wanted]
            if hit:
                narrowed = hit
        source = Path(str(row.get("source_file", ""))).name
        if len(narrowed) > 1 and source:
            hit = [r for r in narrowed if Path(r.source_file).name == source]
            narrowed = hit or narrowed
        if len(narrowed) > 1 and str(row.get("roi_index", "")).strip():
            hit = [r for r in narrowed if str(r.roi_index) == str(row["roi_index"]).strip()]
            narrowed = hit or narrowed
        return narrowed[0] if len(narrowed) == 1 else None

    def apply_selection_file(self, filename: os.PathLike) -> MatchReport:
        """Re-attach saved flags/labels to the loaded particles.

        Rows are matched by ROI time + camera position (robust to
        reprocessing, which may renumber ROIs).  Only rows that carry no
        time/position (very old files) fall back to the particle ID.
        """
        report = MatchReport()
        path = Path(filename)
        if not path.exists():
            return report
        index = self.key_index()
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                report.rows += 1
                old_id = row.get("particle_id", "")
                key = particle_key(row.get("time_matlab"), row.get("image_x_px"), row.get("image_y_px"))
                if key is not None:
                    candidates = index.get(key, [])
                    if not candidates:
                        report.unmatched += 1
                        continue
                    record = self.resolve_key(candidates, row)
                    if record is None:
                        report.ambiguous += 1
                        continue
                    if record.particle_id == old_id:
                        report.same_id += 1
                    else:
                        report.remapped += 1
                else:
                    record = self._by_id.get(old_id)
                    if record is None:
                        report.unmatched += 1
                        continue
                    report.by_id_only += 1
                report.id_map[old_id] = record.particle_id
                status = row.get("status", "undecided")
                if status in {"undecided", "keep", "reject"}:
                    record.status = status
                record.reject_reason = row.get("reject_reason", "")
                record.label = (row.get("label") or "").strip()
                for field in ("actual_length_um", "actual_width_um", "instrument"):
                    if row.get(field) not in (None, ""):
                        setattr(record, field, row[field] if field == "instrument" else float(row[field]))
                try:
                    record.object_plane_um = float(row.get("object_plane_um", 0.0))
                except ValueError:
                    record.object_plane_um = 0.0
        return report

    # Decisions plus what is needed to re-find each particle (time + ROI corners, robust to
    # reprocessing).  Measurements (length, focus, ...) are not repeated here: they are in the
    # MAT files and in <project>_particle_positions.csv, and never change during a session.
    FLAG_FIELDS = (
        "particle_id", "source_file", "roi_index", "time_matlab", "time_iso",
        "image_x_px", "image_y_px", "status", "reject_reason", "label",
        "actual_length_um", "actual_width_um", "start_x", "end_x", "start_y", "end_y",
    )

    def selection_rows(self, decided_only: bool = False):
        """Snapshot of the flags table (plain tuples), cheap enough to take on the UI thread.

        The parts of a row that never change in a session (ID, time, ROI corners)
        are built once per particle and cached; only the decision columns are
        read each time.
        """
        records = [r for r in self.records if not decided_only or r.status != "undecided" or r.label
                   or r.reject_reason]
        cache = self.__dict__.setdefault("_row_cache", {})
        missing = [r for r in records if id(r) not in cache]
        if missing:
            times = np.array([r.time_matlab for r in missing], float)
            iso = [""] * len(missing)
            finite = np.isfinite(times)
            if finite.any():
                # MATLAB datenum -> ms since 1970, formatted in one call
                ms = np.round((times[finite] - 719529.0) * 86400000.0).astype("int64")
                text = np.datetime_as_string(ms.astype("datetime64[ms]"), unit="ms")
                for k, value in zip(np.flatnonzero(finite), text):
                    iso[k] = value.replace("T", " ")
            for r, stamp in zip(missing, iso):
                cache[id(r)] = (r, (r.particle_id, os.path.basename(r.source_file), r.roi_index, r.time_matlab, stamp,
                                    r.image_x_px, r.image_y_px),
                                (r.start_x, r.end_x, r.start_y, r.end_y))
        rows = []
        for r in records:
            entry = cache[id(r)]
            if entry[0] is not r:            # id() reused by a new object: rebuild this one
                del cache[id(r)]
                return self.selection_rows(decided_only)
            rows.append(entry[1] + (r.status, r.reject_reason, r.label, r.actual_length_um, r.actual_width_um)
                        + entry[2])
        return rows

    @classmethod
    def write_selection_rows(cls, filename: os.PathLike, rows) -> None:
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent))
        os.close(fd)
        try:
            with open(temporary, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(cls.FLAG_FIELDS)
                writer.writerows(rows)
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def save_selection_file(self, filename: os.PathLike, decided_only: bool = False) -> None:
        """Write flags and labels for every particle (``decided_only`` writes only decided ones).

        All rows are written by default, so applying the file to a session that
        already has flags also resets particles that are now unreviewed.  Rows
        carry ROI time and corners, so they are re-found after reprocessing.
        """
        self.write_selection_rows(filename, self.selection_rows(decided_only))


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
