"""Size and focus calibration from labelled particles, and probabilistic size retrieval.

This is the step after labelling in the workbench.  It takes the table of
accepted, labelled particles (``<project>_particle_positions.csv``: true size,
measured length, focus, local illumination and signed object-plane distance d)
and

1. **fits** two forward models, separately on each side of the object plane

   * measured length::

         L_app = s * L * (1 + A± * Zd**p± * (I/I_ref)**kappa) + b
         Zd    = lambda * |d - d0| / (L/2)**2           (Fresnel number, lambda = 0.85 um)

     ``b`` is an additive edge offset in micrometres (the processing mask is
     grown by ``contains_points(radius=2)``, about +2 px on the diameter) and
     ``s`` is a scale (an error in the 2.3 um pixel size shows up here).  With a
     single bead size ``s`` and ``b`` cannot be separated, so ``b`` is held at
     the value the processing code implies unless two or more true sizes are
     present.

   * focus::

         foc = F0 * (I/I_ref)**gamma * exp(-(|d - d0f| / w±)**q) + floor

   together with the scatter of each about the model (log-normal, widening away
   from focus), because the scatter is what makes a retrieval honest;

2. **retrieves** true size L and distance d for any particle from (L_app, foc, I)
   by evaluating the posterior on a grid of (L, d).  The side of the object
   plane can be fixed (calibration test, where it is known) or, for cloud data,
   **marginalised**: both sides are kept with equal prior weight and the
   reported size is the posterior median with a credible interval, together
   with P(d > 0);

3. computes the **depth of field** at the processing focus threshold
   (``foc_crit`` in proc/batchProcess.py), both as the deterministic crossing of
   the mean focus curve and as the acceptance-weighted width
   ``integral P(foc > foc_crit | d) dd``;

4. measures the **pixel size geometrically** where the scan moved the slide in
   the image plane: beads stuck to the slide shift by a known in-plane distance
   between stage steps, so their image shift in pixels gives micrometres per
   pixel independently of diffraction and edge detection.

Everything the retrieval needs is written to one JSON file.

Command line::

    cpi-fit     project_particle_positions.csv -o sizing_model.json --figure fit.png
    cpi-retrieve sizing_model.json particles.csv|processed_dir -o retrieved.csv [--side both|+|-]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

LAMBDA_UM = 0.85            # CPI imaging laser wavelength
NOMINAL_PIXEL_UM = 2.3      # proc/imageStats.py PIXEL_SIZE_UM
PIPELINE_PAD_PX = 2.0       # proc/imageStats.py contains_points(radius=2) grows the mask ~1 px each side
DEFAULT_FOC_CRIT = 5.0      # proc/batchProcess.py foc_crit
MODEL_VERSION = 1


# --------------------------------------------------------------------------- input
def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


# Which measured size a calibration is built on.  The model remembers it, and
# retrieval reads the same column, so a model is never applied to the wrong size.
SIZE_MEASURES = {
    "length": {"columns": ("length_um", "len"), "label": "Length (maximum Feret)"},
    "deq": {"columns": ("deq_um", "deq"), "label": "Equivalent-area diameter (contour)"},
    "area_corrected_shadow": {"columns": ("shape_corrected_shadow_um",),
        "label": "Shape-corrected shadow diameter (shadow / sqrt(CPI area ratio))"},
    "shadow_deq": {"columns": ("shadow_deq_um", "shadow_deq"),
                   "label": "Signed-shadow circle equivalent diameter (extra processing measure)"},
}


# Which focus value a calibration is built on (same rule: the model remembers it, retrieval reads it).
FOCUS_MEASURES = {
    "focus": {"columns": ("focus",), "label": "Focus (as processed)"},
    "focus_per_illum": {"columns": ("focus",), "label": "Focus ÷ local illumination (raw gradient)"},
    "cpv_focus": {"columns": ("cpv_focus",), "label": "CPIview focus (processed with --cpiview)"},
}


@dataclass
class CalibrationTable:
    """Columns needed for fitting, as float arrays (NaN where missing)."""

    true_length: np.ndarray
    length: np.ndarray
    focus: np.ndarray
    illum: np.ndarray
    d: np.ndarray
    d_in_plane: np.ndarray
    x_stage: np.ndarray
    z_stage: np.ndarray
    image_x: np.ndarray
    image_y: np.ndarray
    time: np.ndarray
    group: np.ndarray            # batch / reference name, for cross-validation
    label: np.ndarray
    edge_touch: np.ndarray = None    # 1 = reaches the ROI border (size is a lower limit)
    processing: np.ndarray = None
    instrument: Dict[str, str] = field(default_factory=dict)
    pixel_um: np.ndarray = None      # pixel size each particle was processed with (NaN/None: nominal)
    focus_measure: str = "focus"     # which focus value is in ``focus`` (see FOCUS_MEASURES)
    roundness: np.ndarray = None     # measured projected-area roundness Ar = A / (pi/4 L_app^2), per particle
    size_measure: str = "length"     # which measured size is in ``length`` (see SIZE_MEASURES)
    edge_deq: np.ndarray = None      # independent saved contour equivalent-area diameters for like-for-like comparisons
    edge_length: np.ndarray = None   # independent saved maximum Feret lengths
    shadow_ratio: np.ndarray = None # signed shadow / CPI area-equivalent diameter

    def __len__(self):
        return len(self.length)

    def subset(self, mask) -> "CalibrationTable":
        kw = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
              for k, v in self.__dict__.items()}
        return CalibrationTable(**kw)

    @classmethod
    def from_rows(cls, rows: Sequence[dict], size_measure: str = "length",
                  focus_measure: str = "focus") -> "CalibrationTable":
        def col(*names):
            return np.array([next((_num(r.get(n)) for n in names if r.get(n) not in (None, "")), float("nan"))
                             for r in rows], float)

        def text(*names):
            return np.array([next((str(r.get(n)) for n in names if r.get(n) not in (None, "")), "")
                             for r in rows], object)

        instrument = {}
        for key in ("cpi_type", "cpi_id"):
            values = sorted({str(r.get(key)) for r in rows if r.get(key)})
            if values:
                instrument[key] = ", ".join(values)
        foc = col(*FOCUS_MEASURES[focus_measure]["columns"])
        illumination = col("local_illum")
        if focus_measure == "focus_per_illum":
            with np.errstate(divide="ignore", invalid="ignore"):
                foc = np.where(illumination > 0, foc / illumination, np.nan)
        if size_measure == "area_corrected_shadow":
            with np.errstate(invalid="ignore", divide="ignore"):
                area_ratio = col("area_ratio_circle")
                sized = np.where((area_ratio > 0) & (area_ratio <= 1),
                                 col("shadow_deq_um", "shadow_deq") / np.sqrt(area_ratio), np.nan)
        else:
            sized = col(*SIZE_MEASURES[size_measure]["columns"])
        return cls(true_length=col("actual_length_um", "true_length_um"),
                   length=sized,
                   focus=foc, focus_measure=focus_measure,
                   illum=illumination, size_measure=size_measure,
                   d=col("d_signed_plane", "d_mm"), d_in_plane=col("d_in_plane"),
                   x_stage=col("x_stage", "stage_x"), z_stage=col("z_stage", "stage_z"),
                   image_x=col("image_x_px"), image_y=col("image_y_px"), time=col("time_matlab"),
                   group=cls._groups(rows), label=text("label"), instrument=instrument,
                   edge_touch=col("edge_touch"), processing=text("processing"), pixel_um=col("pixel_um"),
                   roundness=col("roundness"),
                   edge_deq=col("deq_um", "deq"), edge_length=col("length_um", "len"),
                   shadow_ratio=np.divide(col("shadow_deq_um", "shadow_deq"), col("deq_um", "deq"),
                       out=np.full(len(rows), np.nan), where=col("deq_um", "deq") > 0))

    @staticmethod
    def _groups(rows):
        """Batch for cross-validation, qualified by project so names from different projects never merge."""
        out = []
        for r in rows:
            batch = r.get("batch") or r.get("reference_from") or ""
            project = r.get("project") or ""
            out.append(f"{project} · {batch}" if (project and batch) else batch)
        return np.array(out, object)

    @classmethod
    def from_csv(cls, path: os.PathLike, size_measure: str = "length",
                 focus_measure: str = "focus") -> "CalibrationTable":
        with open(path, newline="", encoding="utf-8") as handle:
            return cls.from_rows(list(csv.DictReader(handle)), size_measure, focus_measure)


# --------------------------------------------------------------------------- model
@dataclass
class SizingModel:
    """Everything the retrieval needs.  Lengths in um, d in mm, illumination relative to illum_ref."""

    scale: float = 1.0
    offset_um: float = PIPELINE_PAD_PX * NOMINAL_PIXEL_UM
    offset_fitted: bool = False
    size_plus: Dict[str, float] = field(default_factory=lambda: {"A": 1.0, "p": 1.0})
    size_minus: Dict[str, float] = field(default_factory=lambda: {"A": 1.0, "p": 1.0})
    size_kappa: float = 0.0
    size_d0_mm: float = 0.0
    # Size scaling of the defocus: Zd = lambda|d - d0| / (L/2)^2 * (L/L_ref)^(2 - m).  m = 2 is the
    # Fresnel number (Z/L^2); with two or more true sizes m is fitted, which tests that scaling.
    size_L_power: float = 2.0
    size_L_power_fitted: bool = False
    L_ref_um: float = 100.0
    size_sigma: List[float] = field(default_factory=lambda: [0.03, 0.05])     # log sd = s0 + s1*Zd (+ side)
    size_sigma_minus: Optional[List[float]] = None     # − side, when fitted separately (None = same as + side)
    size_model: str = "defocus"       # defocus = existing model; shadow_linear = near-invariant shadow diameter
    focus_model: str = "legacy"       # legacy or normalised_generalised (focus / local illumination)
    focus_F0: float = 50.0
    focus_gamma: float = 1.0
    focus_w_plus_mm: float = 0.5
    focus_w_minus_mm: float = 0.5
    focus_q: float = 1.5
    focus_d0_mm: float = 0.0
    focus_floor: float = 1.0
    # Size dependence of focus (fitted only with two or more true sizes): peak x (L/L_ref)^c,
    # half-widths x (L/L_ref)^k.  0 = focus depends on Z alone (Connolly et al. 2007).
    focus_L_amp: float = 0.0
    focus_L_width: float = 0.0
    focus_two_width: bool = False
    focus_narrow_fraction: float = 0.5
    focus_narrow_scale: float = 0.35
    focus_shadow_gamma: float = 0.0
    focus_shadow_ref: float = 1.0
    # Particle shape (Park et al. 2026): defocus variable Zd / Ar^alpha (alpha = 1 gives their
    # Z/(L^2 Ar)), focus peak x Ar^c.  Ar = measured projected-area roundness.  0 = off.
    size_Ar_power: float = 0.0
    focus_Ar_power: float = 0.0
    focus_Ar_width_power: float = 0.0
    shape_terms: bool = False
    focus_sigma: List[float] = field(default_factory=lambda: [0.2, 0.1])      # log sd = s0 + s1*|d|
    illum_ref: float = 1.0
    d_range_mm: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    size_range_um: List[float] = field(default_factory=lambda: [100.0, 100.0])
    pixel_um: float = NOMINAL_PIXEL_UM
    foc_crit: float = DEFAULT_FOC_CRIT
    wavelength_um: float = LAMBDA_UM
    sign_convention: str = "d = (X-Xbest)cos(theta) + (Z-Zbest)sin(theta) as in the workbench"
    size_measure: str = "length"       # key of SIZE_MEASURES: the measured size this model corrects
    focus_measure: str = "focus"       # key of FOCUS_MEASURES: the focus value the model was fitted on
    length_factor: float = 1.0         # measured sizes are multiplied by this first (pixel-size correction)
    pixel_mode: str = "processed"      # "processed" (as in the MAT files), "set" (given) or "fitted"
    instrument: Dict[str, str] = field(default_factory=dict)
    inputs: List[Dict[str, object]] = field(default_factory=list)   # projects the fit was made from
    fit_stats: Dict[str, object] = field(default_factory=dict)
    version: int = MODEL_VERSION
    created_utc: str = ""

    # ---- forward model
    @staticmethod
    def ar_clip(ar):
        ar = np.asarray(ar, float)
        return np.clip(np.where(np.isfinite(ar), ar, 1.0), 0.05, 1.5)

    def zd(self, L, d, ar=None):
        L = np.maximum(np.asarray(L, float), 1e-3)
        fresnel = self.wavelength_um * np.abs(np.asarray(d, float) - self.size_d0_mm) * 1000.0 / (L / 2.0) ** 2
        if self.size_L_power != 2.0:
            fresnel = fresnel * (L / self.L_ref_um) ** (2.0 - self.size_L_power)
        if ar is not None and self.size_Ar_power:
            fresnel = fresnel / self.ar_clip(ar) ** self.size_Ar_power
        return fresnel

    def mean_length(self, L, d, irel=1.0, ar=None):
        if self.size_model == "shadow_linear":
            return self.scale * np.asarray(L, float) + self.offset_um
        L = np.asarray(L, float)
        d = np.asarray(d, float)
        plus = (d - self.size_d0_mm) >= 0
        A = np.where(plus, self.size_plus["A"], self.size_minus["A"])
        p = np.where(plus, self.size_plus["p"], self.size_minus["p"])
        irel = np.clip(np.asarray(irel, float), 0.05, 20)
        return self.scale * L * (1.0 + A * self.zd(L, d, ar) ** p * irel ** self.size_kappa) + self.offset_um

    def sigma_length(self, L, d, ar=None):
        if self.size_model == "shadow_linear":
            return np.maximum(self.size_sigma[0], 0.01) + np.zeros_like(np.asarray(d, float))
        zd = self.zd(L, d, ar)
        plus = self.size_sigma[0] + self.size_sigma[1] * zd
        if not self.size_sigma_minus:
            return plus
        minus = self.size_sigma_minus[0] + self.size_sigma_minus[1] * zd
        return np.where((np.asarray(d, float) - self.size_d0_mm) >= 0, plus, minus)

    def mean_focus(self, d, irel=1.0, L=None, ar=None, shadow_ratio=None):
        d = np.asarray(d, float)
        u = d - self.focus_d0_mm
        w = np.where(u >= 0, self.focus_w_plus_mm, self.focus_w_minus_mm)
        amp = self.focus_F0
        if L is not None and (self.focus_L_amp or self.focus_L_width):
            ratio = np.maximum(np.asarray(L, float), 1e-3) / self.L_ref_um
            amp = amp * ratio ** self.focus_L_amp
            w = w * ratio ** self.focus_L_width
        if shadow_ratio is not None and self.focus_shadow_gamma:
            sr = np.asarray(shadow_ratio, float)
            w = w * (np.clip(sr, 0.05, 5.0) / self.focus_shadow_ref) ** self.focus_shadow_gamma
        if ar is not None:
            if self.focus_Ar_power:
                amp = amp * self.ar_clip(ar) ** self.focus_Ar_power
            if self.focus_Ar_width_power:
                w = w * self.ar_clip(ar) ** self.focus_Ar_width_power
        irel = np.clip(np.asarray(irel, float), 0.05, 20)
        broad = np.exp(-(np.abs(u) / np.maximum(w, 1e-8)) ** self.focus_q)
        if self.focus_two_width:
            narrow_w = np.maximum(w * self.focus_narrow_scale, 1e-8)
            narrow = np.exp(-(np.abs(u) / narrow_w) ** self.focus_q)
            fraction = np.clip(self.focus_narrow_fraction, 0.0, 1.0)
            curve = amp * (fraction * narrow + (1.0 - fraction) * broad)
        else:
            curve = amp * broad
        if self.focus_model == "normalised_generalised":
            # The observed variable already is F_raw/I_local; no second illumination division.
            return curve + self.focus_floor
        return curve * irel ** self.focus_gamma + self.focus_floor

    def sigma_focus(self, d):
        return self.focus_sigma[0] + self.focus_sigma[1] * np.abs(np.asarray(d, float) - self.focus_d0_mm)

    def irel(self, illum):
        illum = np.asarray(illum, float)
        return np.where(np.isfinite(illum) & (illum > 0), illum / self.illum_ref, 1.0)

    # ---- persistence
    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "SizingModel":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: os.PathLike) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        Path(path).write_text(json.dumps(self.to_json(), indent=2, default=_json_default), encoding="utf-8")

    @classmethod
    def load(cls, path: os.PathLike) -> "SizingModel":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# --------------------------------------------------------------------------- fitting
def _usable(table: CalibrationTable, illum_min: Optional[float]) -> np.ndarray:
    ok = (np.isfinite(table.true_length) & (table.true_length > 0) & np.isfinite(table.length)
          & (table.length > 0) & np.isfinite(table.focus) & (table.focus > 0) & np.isfinite(table.d))
    if illum_min is not None:
        ok &= np.isfinite(table.illum) & (table.illum >= illum_min)
    if table.edge_touch is not None and len(table.edge_touch) == len(ok):
        ok &= ~(table.edge_touch == 1)          # cut off by the ROI border: size is only a lower limit
    return ok


def _scatter_fit(residual, x):
    """Robust linear fit of sigma(x) = s0 + s1*x from residuals (E|r| = sigma*sqrt(2/pi))."""
    y = np.abs(residual) * math.sqrt(math.pi / 2.0)
    good = np.isfinite(y) & np.isfinite(x)
    if good.sum() < 5:
        return [float(max(np.nanmedian(y), 0.01)), 0.0]
    res = least_squares(lambda c: c[0] + c[1] * x[good] - y[good], [np.median(y[good]), 0.0],
                        bounds=([1e-3, 0.0], [5.0, 50.0]), loss="soft_l1", f_scale=float(np.median(y[good]) + 1e-3))
    return [float(res.x[0]), float(res.x[1])]


def processed_pixel_um(table: CalibrationTable) -> float:
    """Pixel size the table's sizes were processed with (raises if particles disagree)."""
    px = getattr(table, "pixel_um", None)
    if px is None:
        return NOMINAL_PIXEL_UM
    px = np.asarray(px, float)
    px = px[np.isfinite(px) & (px > 0)]
    if not px.size:
        return NOMINAL_PIXEL_UM
    values = np.unique(np.round(px, 6))
    if len(values) > 1:
        raise ValueError(f"particles were processed with different pixel sizes {list(values)} µm; "
                         "reprocess them with the same --pixel-um before fitting together")
    return float(values[0])


def _empirical_focus_widths(L, d, foc):
    """Estimate each size group's FWHM from stage-bin median normalised focus.

    Only use profiles with both half-height crossings observed; no extrapolation.
    Returns (true_size_um, fwhm_mm, effective_group_count).
    """
    out = []
    for size in np.unique(np.round(L, 1)):
        mask = (np.round(L, 1) == size) & np.isfinite(d) & np.isfinite(foc)
        if mask.sum() < 100:
            continue
        xx, yy = d[mask], foc[mask]
        edges = np.arange(xx.min() - 0.013, xx.max() + 0.038, 0.025)
        if len(edges) < 12:
            continue
        cc = (edges[:-1] + edges[1:]) / 2
        mm = np.array([np.median(yy[(xx >= a) & (xx < b)]) if np.count_nonzero((xx >= a) & (xx < b)) >= 5 else np.nan
                       for a, b in zip(edges[:-1], edges[1:])])
        ok = np.isfinite(mm)
        if ok.sum() < 12:
            continue
        floor = float(np.percentile(mm[ok], 10))
        k = int(np.nanargmax(mm))
        peak = float(mm[k])
        if peak <= floor * 1.1:
            continue
        half = floor + .5 * (peak - floor)
        # Use only populated distance bins; real calibration scans have discrete
        # stage steps and empty bins must not stop the half-height search.
        xc, yc = cc[ok], mm[ok]
        pk = int(np.argmax(yc))
        crossings = []
        for direction in (-1, 1):
            j = pk
            while 0 <= j + direction < len(yc) and yc[j + direction] >= half:
                j += direction
            nxt = j + direction
            if not 0 <= nxt < len(yc):
                break
            v1, v2 = yc[j], yc[nxt]
            if v1 == v2:
                break
            crossings.append(float(xc[j] + (half - v1) * (xc[nxt] - xc[j]) / (v2 - v1)))
        if len(crossings) == 2 and 0.03 < crossings[1] - crossings[0] < 10:
            out.append((float(size), crossings[1] - crossings[0], int(mask.sum())))
    return out



def _focus_width_comparison(L, d, foc, model, shadow_ratio=None, observed_baseline=None, roundness=None):
    """Observed versus fitted half-peak widths per true-size population.

    Observed crossings use stage-position medians, interpolating over unoccupied
    stage bins. Unresolved crossings are reported as missing, not extrapolated.
    The theoretical widths are at half the amplitude above the fitted baseline.
    """
    result = []
    for size in np.unique(np.round(L, 1)):
        mask = (np.round(L, 1) == size) & np.isfinite(d) & np.isfinite(foc)
        n = int(mask.sum())
        if n < 100:
            continue
        xx, yy = np.asarray(d[mask]), np.asarray(foc[mask])
        # Quantise stages at 0.025 mm; empty positions do not imply zero focus.
        pos = np.round(xx / .025).astype(int)
        stage, inv, counts = np.unique(pos, return_inverse=True, return_counts=True)
        valid_stages = counts >= 3
        xs = stage[valid_stages] * .025
        ys = np.array([np.median(yy[inv == j]) for j in np.flatnonzero(valid_stages)])
        if len(xs) < 8:
            continue
        floor = float(observed_baseline if observed_baseline is not None else np.percentile(ys, 10))
        peak_index = int(np.argmax(ys))
        peak = float(ys[peak_index])
        half = floor + 0.5 * (peak - floor)
        crossings = [None, None]
        if peak > floor * 1.1:
            for idx, direction in enumerate((-1, 1)):
                j = peak_index
                while 0 <= j + direction < len(ys) and ys[j + direction] >= half:
                    j += direction
                k = j + direction
                if 0 <= k < len(ys) and ys[j] != ys[k]:
                    crossings[idx] = float(xs[j] + (half - ys[j]) * (xs[k] - xs[j]) / (ys[k] - ys[j]))
        # Use the observed peak location to report observable half-widths.
        obs_minus = float(xs[peak_index] - crossings[0]) if crossings[0] is not None else None
        obs_plus = float(crossings[1] - xs[peak_index]) if crossings[1] is not None else None
        if obs_minus is not None and obs_minus <= 0: obs_minus = None
        if obs_plus is not None and obs_plus <= 0: obs_plus = None
        scale = (float(size) / model.L_ref_um) ** model.focus_L_width
        if model.focus_shadow_gamma and shadow_ratio is not None:
            good_ratio = np.asarray(shadow_ratio)[mask]
            good_ratio = good_ratio[np.isfinite(good_ratio) & (good_ratio > 0)]
            if len(good_ratio):
                scale *= (float(np.median(good_ratio)) / model.focus_shadow_ref) ** model.focus_shadow_gamma
        values_ar = np.asarray(roundness)[mask] if roundness is not None else np.ones(n)
        values_ar = values_ar[np.isfinite(values_ar) & (values_ar > 0)]
        median_ar = float(np.median(values_ar)) if values_ar.size else 1.0
        scale *= median_ar ** model.focus_Ar_width_power
        shape_factor = math.log(2) ** (1.0 / model.focus_q)
        if model.focus_two_width:
            from scipy.optimize import brentq
            pred_minus = pred_plus = None
            for side in (-1, 1):
                def half_res(delta):
                    v = float(model.mean_focus(model.focus_d0_mm + side*delta, L=size, ar=median_ar))
                    return v - (model.focus_floor + .5*(float(model.mean_focus(model.focus_d0_mm, L=size, ar=median_ar))-model.focus_floor))
                try:
                    crossing = float(brentq(half_res, 1e-8, 20.))
                except ValueError:
                    crossing = float('nan')
                if side == -1: pred_minus = crossing
                else: pred_plus = crossing
        else:
            pred_minus = float(model.focus_w_minus_mm * scale * shape_factor)
            pred_plus = float(model.focus_w_plus_mm * scale * shape_factor)
        observed = (obs_minus + obs_plus) if obs_minus is not None and obs_plus is not None else None
        predicted = pred_minus + pred_plus
        result.append({"size_um": float(size), "count": n, "median_ar": median_ar,
                       "observed_minus_mm": obs_minus, "observed_plus_mm": obs_plus,
                       "observed_fwhm_mm": observed,
                       "model_minus_mm": pred_minus, "model_plus_mm": pred_plus,
                       "model_fwhm_mm": predicted,
                       "difference_pct": 100 * (predicted / observed - 1) if observed is not None else None})
    return result


def fit_model(table: CalibrationTable, illum_min: Optional[float] = None, fit_offset: Optional[bool] = None,
              offset_um: Optional[float] = None, foc_crit: float = DEFAULT_FOC_CRIT,
              pixel_um: Optional[float] = None, pixel_mode: str = "processed",
              pixel_set_um: Optional[float] = None, shape: bool = False,
              size_model: str = "defocus", focus_model: str = "legacy",
              fit_focus_p: bool = True, asymmetric_focus: bool = True,
              fit_peak_size: bool = True, fit_width_size: bool = True, empirical_width_constraint: bool = False,
              shadow_width_predictor: bool = False, population_balanced_focus: bool = False,
              focus_baseline_mode: str = "free", two_width_focus: bool = False,
              focus_ar_amp: bool = False, focus_ar_width: bool = False) -> SizingModel:
    """Fit the length and focus models to labelled particles with known signed d.

    ``pixel_mode``: "processed" uses sizes as processed (pixel size ``pixel_um``;
    the overall scale s absorbs any error).  "set" rescales every measured size
    by ``pixel_set_um / pixel_um`` first; "fitted" finds that factor from the
    beads (needs two or more true sizes, so that scale and edge offset can be
    told apart) and rescales by it.  In both, s is then fixed at 1, so the
    pixel size alone carries the scale, and retrieval applies the same factor.

    ``pixel_um`` (the pixel size the sizes were processed with) defaults to the
    one recorded per particle (``table.pixel_um``, from dat.proc_options), else
    the nominal 2.3 µm, so a set pixel size is never applied on top of one
    already used in processing.

    ``shape``: also fit the particle-shape terms (Park et al. 2026): Zd / Ar^alpha in the size model
    and Ar^c on the focus peak, Ar = measured roundness.  Needs particles of different shapes
    (e.g. beads and ice analogues); skipped if Ar hardly varies.
    """
    if focus_baseline_mode not in ("free", "zero", "far_focus"):
        raise ValueError("Focus baseline must be Free, Zero or Fixed from far-focus data")
    if size_model not in ("defocus", "shadow_linear"):
        raise ValueError(f"Unknown size model {size_model}")
    if focus_model not in ("legacy", "normalised_generalised"):
        raise ValueError(f"Unknown focus model {focus_model}")
    if focus_model == "normalised_generalised" and table.focus_measure != "focus_per_illum":
        raise ValueError("Normalised focus model requires the 'focus / local illumination' measurement")
    if size_model == "shadow_linear" and table.size_measure not in ("shadow_deq", "area_corrected_shadow"):
        raise ValueError("Shadow-size model requires the signed-shadow processing diameter")
    if size_model == "shadow_linear" and pixel_mode == "fitted":
        raise ValueError("Use as-processed pixel size for the shadow method; its scale is fitted directly")
    if pixel_um is None:
        pixel_um = processed_pixel_um(table)
    if pixel_mode == "fitted":
        first = fit_model(table, illum_min, fit_offset=True if fit_offset is None else fit_offset,
                          offset_um=offset_um, foc_crit=foc_crit, pixel_um=pixel_um, shape=shape,
                          size_model=size_model, focus_model=focus_model, fit_focus_p=fit_focus_p,
                          asymmetric_focus=asymmetric_focus, fit_peak_size=fit_peak_size,
                          fit_width_size=fit_width_size, empirical_width_constraint=empirical_width_constraint,
                          shadow_width_predictor=shadow_width_predictor, population_balanced_focus=population_balanced_focus,
                          focus_baseline_mode=focus_baseline_mode, two_width_focus=two_width_focus,
                          focus_ar_amp=focus_ar_amp, focus_ar_width=focus_ar_width)
        if not first.size_L_power_fitted:
            raise ValueError("fitting the pixel size needs two or more true sizes (scale and edge offset are "
                             "otherwise indistinguishable)")
        factor = 1.0 / first.scale
        model = fit_model(table, illum_min, fit_offset=fit_offset, offset_um=offset_um, foc_crit=foc_crit,
                          pixel_um=pixel_um, pixel_mode="set", pixel_set_um=pixel_um * factor, shape=shape,
                          size_model=size_model, focus_model=focus_model, fit_focus_p=fit_focus_p,
                          asymmetric_focus=asymmetric_focus, fit_peak_size=fit_peak_size,
                          fit_width_size=fit_width_size, empirical_width_constraint=empirical_width_constraint,
                          shadow_width_predictor=shadow_width_predictor, population_balanced_focus=population_balanced_focus,
                          focus_baseline_mode=focus_baseline_mode, two_width_focus=two_width_focus,
                          focus_ar_amp=focus_ar_amp, focus_ar_width=focus_ar_width)
        model.pixel_mode = "fitted"
        return model
    factor = 1.0
    if pixel_mode == "set":
        if not pixel_set_um or pixel_set_um <= 0:
            raise ValueError("give the pixel size to use")
        factor = float(pixel_set_um) / float(pixel_um)
    use = _usable(table, illum_min)
    if shadow_width_predictor:
        if table.shadow_ratio is None:
            raise ValueError("Shadow predictor needs saved shadow and CPI area diameters")
        use &= np.isfinite(table.shadow_ratio) & (table.shadow_ratio > 0)
    if use.sum() < 20:
        raise ValueError(f"only {int(use.sum())} usable particles (need labelled true size, length, focus and d)")
    t = table.subset(use)
    L, Lapp, foc, d = t.true_length, t.length * factor, t.focus, t.d
    illum = np.where(np.isfinite(t.illum) & (t.illum > 0), t.illum, np.nan)
    illum_ref = float(np.nanmedian(illum)) if np.isfinite(illum).any() else 1.0
    irel = np.where(np.isfinite(illum), illum / illum_ref, 1.0)
    ar = SizingModel.ar_clip(t.roundness if t.roundness is not None else np.full(len(L), np.nan))
    shape_note = None
    if size_model == "shadow_linear":
        shape = False  # shape remains represented by conventional stored contour, not the circle
    if shape:
        spread = float(np.nanstd(np.log(ar)))
        if spread < 0.05:
            shape, shape_note = False, f"shape terms skipped: roundness hardly varies (sd of log Ar {spread:.3f})"
    # AR terms require actual spread in saved projected-area roundness.
    ar_spread = float(np.std(np.log(ar)))
    if (focus_ar_amp or focus_ar_width) and ar_spread < .05:
        raise ValueError("Focus aspect-ratio fit requires varying saved roundness (std(log Ar) >= 0.05)")
    focus_ar_amp = bool(focus_ar_amp or shape)  # retain legacy shape-checkbox behaviour
    sizes = np.unique(np.round(L, 1))
    multi = len(sizes) >= 2
    if fit_offset is None:
        fit_offset = multi
    model = SizingModel(illum_ref=illum_ref, foc_crit=foc_crit, pixel_um=pixel_um, instrument=dict(t.instrument),
                        offset_um=(PIPELINE_PAD_PX * pixel_um if offset_um is None else float(offset_um)),
                        offset_fitted=bool(fit_offset),
                        d_range_mm=[float(np.min(d)), float(np.max(d))],
                        size_range_um=[float(np.min(L)), float(np.max(L))],
                        L_ref_um=float(np.exp(np.mean(np.log(sizes)))), size_L_power_fitted=multi,
                        length_factor=factor, pixel_mode=pixel_mode, shape_terms=bool(shape),
                        size_model=size_model, focus_model=focus_model,
                        focus_shadow_ref=(float(np.median(t.shadow_ratio)) if shadow_width_predictor else 1.0))

    # ---- focus first: it pins the object plane and gives the scale of d
    near = np.abs(d) <= np.quantile(np.abs(d), 0.15)
    F0 = float(np.quantile(foc[near], 0.6)) if near.any() else float(np.quantile(foc, 0.9))
    floor0 = float(max(np.quantile(foc, 0.03) * 0.5, 0.1))
    w0 = float(max(np.std(d), 0.05))
    # Population-balanced, stage-median far-focus background estimate.
    # Estimate from the outer 20% of absolute stage positions within each population;
    # avoid giving the largest bead population most of the weight.
    far_floor = None
    if focus_baseline_mode == "far_focus":
        per_population = []
        for sz in sizes:
            sel = (np.round(L, 1) == sz) & np.isfinite(d) & np.isfinite(foc)
            if np.sum(sel) < 30:
                continue
            steps = np.round(d[sel] / 0.025).astype(int)
            levels, inverse, counts = np.unique(steps, return_inverse=True, return_counts=True)
            good = np.flatnonzero(counts >= 3)
            if len(good) < 5:
                continue
            stage_medians = np.array([np.median(foc[sel][inverse == j]) for j in good])
            stage_abs = np.abs(levels[good].astype(float) * .025)
            outer = stage_abs >= np.quantile(stage_abs, .8)
            if np.any(outer):
                per_population.append(float(np.median(stage_medians[outer])))
        if not per_population:
            raise ValueError("Cannot estimate far-focus baseline: insufficient populated stage positions")
        far_floor = float(max(np.median(per_population), 0.0))
        if not np.isfinite(far_floor):
            raise ValueError("Nonfinite far-focus baseline")
    baseline_value = (0.0 if focus_baseline_mode == "zero" else
                      far_floor if focus_baseline_mode == "far_focus" else None)


    def unpack_focus(c, m):
        m.focus_F0, m.focus_gamma, m.focus_w_plus_mm, m.focus_w_minus_mm, m.focus_q, m.focus_d0_mm, m.focus_floor = (
            math.exp(c[0]), float(c[1]), math.exp(c[2]), math.exp(c[3]), float(c[4]), float(c[5]), (float(baseline_value) if baseline_value is not None else math.exp(c[6])))
        if focus_model == "normalised_generalised":
            m.focus_gamma = 0.0  # data already F_raw / I_local; reference illumination irrelevant
        if not asymmetric_focus:
            m.focus_w_minus_mm = m.focus_w_plus_mm
        k = 7
        if multi:
            m.focus_L_amp, m.focus_L_width = (float(c[7]) if fit_peak_size else 0.0,
                                               float(c[8]) if fit_width_size else 0.0)
            k = 9
        if focus_ar_amp:
            m.focus_Ar_power = float(c[k]); k += 1
        if focus_ar_width:
            m.focus_Ar_width_power = float(c[k]); k += 1
        if shadow_width_predictor:
            m.focus_shadow_gamma = float(c[k])

    if focus_model == "normalised_generalised":
        # Equal total influence per size and stage-position group, not per camera frame.
        groups, counts = np.unique(np.column_stack((np.round(L, 1), np.round(d / .02))),
                                   axis=0, return_counts=True)
        lookup = {tuple(map(float, group)): n for group, n in zip(groups, counts)}
        fw = np.array([1.0 / np.sqrt(lookup[(float(round(li, 1)), float(round(di / .02)))])
                       for li, di in zip(L, d)])
        fw /= np.sqrt(np.mean(fw**2))
    else:
        fw = 1.0

    # Optional constraint: match empirically measured focus FWHM scaling between
    # calibration sizes. It is deliberately off by default and never invents
    # widths for groups without both observed half-height crossings.
    empirical_widths = (_empirical_focus_widths(L, d, foc)
                        if empirical_width_constraint and multi and fit_width_size
                        and focus_model == "normalised_generalised" else [])
    if len(empirical_widths) >= 2:
        es = np.array([v[0] for v in empirical_widths])
        ew = np.array([v[1] for v in empirical_widths])
        empirical_beta = float(np.polyfit(np.log(es / model.L_ref_um), np.log(ew), 1)[0])
        empirical_beta = float(np.clip(empirical_beta, 0., 3.))
    else:
        empirical_beta = None

    # Population-balanced experimental objective: each population contributes equally,
    # regardless of its frame count; individual stage medians reduce oversampling bias.
    # The half-height threshold is floor + (peak - floor)/2, NEVER peak/2.
    population_profiles = []
    if population_balanced_focus and focus_model == "normalised_generalised":
        for sz in sizes:
            use_sz = (np.round(L, 1) == sz) & np.isfinite(d) & np.isfinite(foc)
            positions = np.round(d[use_sz] / .025).astype(int)
            stage, inverse, counts = np.unique(positions, return_inverse=True, return_counts=True)
            chosen = np.flatnonzero(counts >= 3)
            if len(chosen) < 8:
                continue
            xx = stage[chosen].astype(float) * .025
            yy = np.asarray([np.median(foc[use_sz][inverse == j]) for j in chosen])
            # Use typical illumination and shape for model profile evaluation.
            ss = np.flatnonzero(use_sz)
            population_profiles.append((float(sz), xx, yy,
                 float(np.median(irel[ss])), float(np.median(ar[ss]))))
        if len(population_profiles) < 2:
            raise ValueError("Population-balanced fit needs at least two sizes with 8 populated stage positions each")

    def _profile_widths(xx, yy):
        """Return (peak-above-background, left half-width, right half-width)."""
        base = float(baseline_value if baseline_value is not None else np.percentile(yy, 10))
        peak_idx = int(np.argmax(yy)); peak = float(yy[peak_idx]); half = base + .5*(peak-base)
        widths = []
        for side in (-1, 1):
            j = peak_idx
            while 0 <= j + side < len(yy) and yy[j+side] >= half:
                j += side
            nxt = j + side
            if not 0 <= nxt < len(yy) or yy[j] == yy[nxt]:
                widths.append(None)
                continue
            crossing = xx[j] + (half-yy[j])*(xx[nxt]-xx[j])/(yy[nxt]-yy[j])
            val = (xx[peak_idx]-crossing) if side == -1 else (crossing-xx[peak_idx])
            widths.append(float(val) if val > 0 else None)
        return peak-base, widths[0], widths[1]

    def balanced_focus_res(c):
        m = SizingModel(**{**model.__dict__})
        unpack_focus(c, m)
        components = []
        n_pop = len(population_profiles)
        for sz, xx, yy, local_illum, roundness in population_profiles:
            predicted = m.mean_focus(xx, np.full_like(xx, local_illum),
                                     np.full_like(xx, sz), np.full_like(xx, roundness))
            # Equal aggregate weight for each population and its stage profile.
            obs_log = np.log(np.maximum(yy, 1e-6))
            pred_log = np.log(np.maximum(predicted, 1e-6))
            residual = (obs_log-pred_log)/np.sqrt(len(xx)*n_pop)
            f = .3
            transformed = np.sign(residual)*f*np.sqrt(2*np.expm1(.5*np.log1p((residual/f)**2)))
            components.extend(transformed)
            obs_height, obs_left, obs_right = _profile_widths(xx, yy)
            # Analytic half-amplitude model crossings avoid variable-length residual
            # vectors when an optimiser trial places crossings outside sampled stages.
            pred_height = float(m.focus_F0 * (sz/m.L_ref_um)**m.focus_L_amp)
            half_factor = math.log(2) ** (1.0/m.focus_q)
            pred_left = float(m.focus_w_minus_mm * (sz/m.L_ref_um)**m.focus_L_width * half_factor)
            pred_right = float(m.focus_w_plus_mm * (sz/m.L_ref_um)**m.focus_L_width * half_factor)
            # Height affects the half-height threshold in both profiles. Penalise
            # amplitude separately, preventing a wrong height from masquerading as a width fit.
            if obs_height > 0 and pred_height > 0:
                components.append(0.35*np.log(pred_height/obs_height)/np.sqrt(n_pop))
            for ow, pw in ((obs_left,pred_left),(obs_right,pred_right)):
                if ow is not None and pw is not None and ow > 0 and pw > 0:
                    components.append(0.7*np.log(pw/ow)/np.sqrt(n_pop))
        return np.asarray(components)

    def focus_res(c, weighted=True):
        m = SizingModel(**{**model.__dict__})
        unpack_focus(c, m)
        residual = np.log(foc) - np.log(m.mean_focus(d, irel, L, ar, t.shadow_ratio if shadow_width_predictor else None))
        return residual * fw if weighted else residual

    # The normal least-squares fit uses soft_l1 on every observation. Applying
    # soft_l1 to a width-prior residual also makes the prior nearly linear at
    # large deviations; for ~70k frames this effectively switches it off.
    # Transform the image residuals to preserve their exact soft_l1 cost, but
    # keep the width prior quadratic so it can exert an actual constraint.
    def constrained_focus_res(c):
        residual = focus_res(c)
        f = 0.3
        robust_residual = np.sign(residual) * f * np.sqrt(
            2.0 * np.expm1(0.5 * np.log1p((residual / f) ** 2)))
        # A group-level quadratic constraint, scaled to remain influential
        # when a calibration contains tens of thousands of frames. This
        # calibration choice is opt-in; the baseline objective is unchanged.
        # Weight scales with the square root of image count, so that adding
        # frames does not drown out the group-level width information.
        constraint_strength = 0.5
        beta_residual = (c[8] - empirical_beta) * constraint_strength * np.sqrt(len(L))
        return np.r_[robust_residual, beta_residual]

    c0 = [math.log(max(F0 - floor0, 1e-3)), 0.0 if focus_model == "normalised_generalised" else 1.0,
          math.log(w0), math.log(w0), 2.0 if focus_model == "normalised_generalised" else 1.5,
          0.0, math.log(floor0)]
    lo = [-5, -3, math.log(1e-3), math.log(1e-3), 0.5, -0.5, math.log(1e-3)]
    hi = [10, 4, math.log(20), math.log(20), 4.0, 0.5, math.log(max(F0, 1.0))]
    if baseline_value is not None:
        # Keep the solver's parameter vector unchanged while fixing the physical floor.
        # exp(c[6]) cannot equal zero, so unpack_focus uses baseline_value instead.
        c0[6] = math.log(max(baseline_value, 1e-9))
        lo[6], hi[6] = c0[6] - 1e-9, c0[6] + 1e-9
    if focus_model == "normalised_generalised":
        lo[1], hi[1] = -1e-10, 1e-10
        if not fit_focus_p:
            lo[4], hi[4] = 2.0 - 1e-8, 2.0 + 1e-8
        if not asymmetric_focus:
            lo[3], hi[3] = c0[3]-1e-8, c0[3]+1e-8
    if multi:
        c0 += [0.35 if fit_peak_size and focus_model == "normalised_generalised" else 0.0,
               0.5 if fit_width_size and focus_model == "normalised_generalised" else 0.0]
        lo += [0.0 if focus_model == "normalised_generalised" and fit_peak_size else (-3.0 if fit_peak_size else -1e-10),
               0.0 if focus_model == "normalised_generalised" and fit_width_size else (-3.0 if fit_width_size else -1e-10)]
        hi += [3.0 if fit_peak_size else 1e-10, 3.0 if fit_width_size else 1e-10]
    if focus_ar_amp:
        c0.append(0.0); lo.append(-3.0); hi.append(3.0)
    if focus_ar_width:
        c0.append(0.0); lo.append(-3.0); hi.append(3.0)
    if shadow_width_predictor:
        c0.append(0.0)
        lo.append(-2.0)
        hi.append(2.0)
    if population_balanced_focus and focus_model == "normalised_generalised":
        fres = least_squares(balanced_focus_res, c0, bounds=(lo, hi), loss="linear")
    elif empirical_beta is not None:
        fres = least_squares(constrained_focus_res, c0, bounds=(lo, hi), loss="linear")
    else:
        fres = least_squares(focus_res, c0, bounds=(lo, hi), loss="soft_l1", f_scale=0.3)
    c = fres.x
    unpack_focus(c, model)
    if two_width_focus and focus_model == "normalised_generalised":
        # Two components share the same centre, shape exponent and size scaling.
        # Fit to stage medians with equal weight per population; do not apply
        # single-component analytic FWHM priors to this alternative shape.
        if not population_profiles:
            for sz in sizes:
                use_sz = (np.round(L, 1) == sz) & np.isfinite(d) & np.isfinite(foc)
                stage_ids = np.round(d[use_sz] / .025).astype(int)
                stage, inv, counts = np.unique(stage_ids, return_inverse=True, return_counts=True)
                valid_stage = np.flatnonzero(counts >= 3)
                if len(valid_stage) < 8: continue
                xx = stage[valid_stage].astype(float) * .025
                yy = np.asarray([np.median(foc[use_sz][inv == j]) for j in valid_stage])
                ss = np.flatnonzero(use_sz)
                population_profiles.append((float(sz), xx, yy, float(np.median(irel[ss])), float(np.median(ar[ss]))))
        if not population_profiles:
            raise ValueError("Two-width focus requires populated stage profiles")
        base = [math.log(model.focus_F0), math.log(model.focus_w_plus_mm),
                math.log(model.focus_w_minus_mm), model.focus_q, model.focus_d0_mm,
                model.focus_L_amp, model.focus_L_width, 0.4, 0.38]
        lower = [-8, math.log(.01), math.log(.01), .55, -.25,
                 0. if fit_peak_size else -1e-10, 0. if fit_width_size else -1e-10, .02, .08]
        upper = [5, math.log(10), math.log(10), 4., .25,
                 3. if fit_peak_size else 1e-10, 3. if fit_width_size else 1e-10, .98, .85]
        if focus_ar_amp:
            base.append(model.focus_Ar_power); lower.append(-3.); upper.append(3.)
        if focus_ar_width:
            base.append(model.focus_Ar_width_power); lower.append(-3.); upper.append(3.)
        if not fit_focus_p:
            lower[3], upper[3] = 2.-1e-8, 2.+1e-8
            base[3] = 2.
        if not asymmetric_focus:
            lower[2], upper[2] = base[2]-1e-9, base[2]+1e-9
        def mixture_res(v):
            m = SizingModel(**{**model.__dict__})
            m.focus_F0 = math.exp(v[0]); m.focus_w_plus_mm = math.exp(v[1]); m.focus_w_minus_mm = math.exp(v[2])
            m.focus_q = v[3]; m.focus_d0_mm = v[4]
            m.focus_L_amp = v[5]; m.focus_L_width = v[6]
            m.focus_two_width = True; m.focus_narrow_fraction = v[7]; m.focus_narrow_scale = v[8]
            j = 9
            if focus_ar_amp:
                m.focus_Ar_power = float(v[j]); j += 1
            if focus_ar_width:
                m.focus_Ar_width_power = float(v[j]); j += 1
            chunks = []
            for sz, xx, yy, illum, ar0 in population_profiles:
                pred = m.mean_focus(xx, np.full_like(xx, illum), np.full_like(xx, sz), np.full_like(xx, ar0))
                chunks.append((np.log(np.maximum(pred, 1e-8))-np.log(np.maximum(yy, 1e-8))) / math.sqrt(len(xx)*len(population_profiles)))
            return np.concatenate(chunks)
        alt = least_squares(mixture_res, np.clip(base, np.asarray(lower)+1e-8, np.asarray(upper)-1e-8),
                            bounds=(lower, upper), loss="soft_l1", f_scale=.2, max_nfev=250)
        vv = alt.x
        model.focus_F0 = math.exp(vv[0]); model.focus_w_plus_mm = math.exp(vv[1]); model.focus_w_minus_mm = math.exp(vv[2])
        model.focus_q = vv[3]; model.focus_d0_mm = vv[4]; model.focus_L_amp = vv[5]; model.focus_L_width = vv[6]
        model.focus_two_width = True; model.focus_narrow_fraction = float(vv[7]); model.focus_narrow_scale = float(vv[8])
        j = 9
        if focus_ar_amp:
            model.focus_Ar_power = float(vv[j]); j += 1
        if focus_ar_width:
            model.focus_Ar_width_power = float(vv[j]); j += 1
        two_width_diagnostics = {"cost": float(alt.cost), "success": bool(alt.success),
                                              "narrow_fraction": float(vv[7]), "narrow_scale": float(vv[8]),
                                              "note": "Stage-median two-component fit; no single-width FWHM constraint"}
    model.fit_stats["focus_baseline_mode"] = focus_baseline_mode
    model.fit_stats["focus_fixed_baseline"] = baseline_value
    model.fit_stats["population_balanced_focus"] = bool(population_balanced_focus)
    model.fit_stats["focus_shadow_predictor"] = ("shadow/CPI area diameter ratio" if shadow_width_predictor else "off")
    # Evaluate the scatter against the FINAL fitted focus model.  In the
    # two-width branch, `c` still represents the preliminary single-width
    # fit; calling focus_res(c) reconstructs that old curve and produces
    # incorrect scatter and acceptance-weighted DOF estimates.
    final_focus_pred = model.mean_focus(
        d, irel, L, ar, t.shadow_ratio if shadow_width_predictor else None)
    final_focus_residual = (np.log(np.maximum(foc, 1e-12))
                            - np.log(np.maximum(final_focus_pred, 1e-12)))
    model.focus_sigma = _scatter_fit(final_focus_residual,
                                     np.abs(d - model.focus_d0_mm))
    width_diagnostics = None
    if empirical_width_constraint:
        width_diagnostics = {
            "enabled": True, "used": empirical_beta is not None,
            "target_exponent": empirical_beta,
            "profiles": [{"size_um": a, "fwhm_mm": b, "count": n} for a, b, n in empirical_widths],
            "fitted_exponent": float(model.focus_L_width),
            "exponent_difference": (float(model.focus_L_width - empirical_beta)
                                    if empirical_beta is not None else None),
            "constraint_strength": 0.5,
        }

    # ---- signed-shadow diameter: fit measured diameter = scale * true diameter + offset.
    # No arbitrary Zd-dependent correction, so size remains a separate observation from focus.
    if size_model == "shadow_linear":
        if len(sizes) < 2:
            # With only one bead size, scale and offset cannot both be identified.
            fit_offset = False
        if fit_offset:
            xy = np.column_stack([L, np.ones_like(L)])
            coeff = np.linalg.lstsq(xy, Lapp, rcond=None)[0]
            initial = [np.clip(coeff[0], .2, 4), np.clip(coeff[1], -20, 40)]
            opt = least_squares(lambda c: np.log(Lapp)-np.log(np.maximum(c[0]*L+c[1], 1e-3)),
                                initial, bounds=([.1, -30.], [5., 50.]), loss="soft_l1", f_scale=.05)
            model.scale, model.offset_um = map(float, opt.x)
        else:
            model.scale = float(np.exp(np.median(np.log(Lapp / np.maximum(L + model.offset_um, 1e-6)))))
            opt = None
        model.offset_fitted = bool(fit_offset)
        resid = np.log(Lapp) - np.log(np.maximum(model.mean_length(L, d), 1e-3))
        model.size_sigma = [float(max(np.median(np.abs(resid)) * 1.4826, .015)), 0.]
        model.size_sigma_minus = None
        model.size_L_power_fitted = False
        model.fit_stats = {
            "n_particles": int(len(L)), "true_sizes_um": [float(v) for v in sizes],
            "processing": sorted({str(v) for v in t.processing if v}) if t.processing is not None else [],
            "excluded_edge_touch": int(np.sum(table.edge_touch == 1)) if table.edge_touch is not None else 0,
            "illum_min": illum_min, "focus_cost": float(fres.cost),
            "size_cost": float(opt.cost if opt is not None else np.sum(resid**2)/2),
            "ratio_at_focus_measured": float(np.median(Lapp[near] / L[near])) if near.any() else None,
            "length_at_focus_um": {f"{v:g}": float(model.mean_length(v, 0)) for v in sizes},
            "effective_pixel_um_if_offset_correct": float(pixel_um * model.scale),
            "shape_note": "circle shadow integral is separate from stored morphology",
            "roundness_range": [float(np.nanpercentile(ar, 5)), float(np.nanpercentile(ar, 95))],
        }
        if model.focus_two_width:
            model.fit_stats["two_width_focus"] = two_width_diagnostics
        model.fit_stats["focus_baseline_mode"] = focus_baseline_mode
        model.fit_stats["focus_fixed_baseline"] = baseline_value
        model.fit_stats["population_balanced_focus"] = bool(population_balanced_focus)
        if width_diagnostics is not None:
            model.fit_stats["empirical_width_constraint"] = width_diagnostics
        if focus_model == "normalised_generalised":
            model.fit_stats["focus_width_comparison"] = _focus_width_comparison(L, d, foc, model, t.shadow_ratio, baseline_value, t.roundness)
        return model

    # ---- length
    r0 = np.median((Lapp[near] - model.offset_um) / L[near]) if near.any() else 1.0

    def unpack_size(c, m):
        m.scale, m.size_plus, m.size_minus = math.exp(c[0]), {"A": math.exp(c[1]), "p": float(c[2])}, \
            {"A": math.exp(c[3]), "p": float(c[4])}
        m.size_kappa, m.size_d0_mm = float(c[5]), float(c[6])
        k = 7
        if fit_offset:
            m.offset_um = float(c[k])
            k += 1
        if multi:
            m.size_L_power = float(c[k])
            k += 1
        if shape:
            m.size_Ar_power = float(c[k])

    def size_res(c):
        m = SizingModel(**{**model.__dict__})
        unpack_size(c, m)
        return np.log(Lapp) - np.log(np.maximum(m.mean_length(L, d, irel, ar), 1e-3))

    s0 = [math.log(max(r0, 0.3)), 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    lo = [math.log(0.3), -12, 0.2, -12, 0.2, -3, -0.5]
    hi = [math.log(3.0), 8, 4.0, 8, 4.0, 3, 0.5]
    if pixel_mode == "set":            # the pixel size carries the scale: s = 1
        s0[0], lo[0], hi[0] = 0.0, -1e-9, 1e-9
    if fit_offset:
        s0.append(model.offset_um)
        lo.append(-20.0)
        hi.append(40.0)
    if multi:
        s0.append(2.0)
        lo.append(0.0)
        hi.append(4.0)
    if shape:
        s0.append(1.0)
        lo.append(0.0)
        hi.append(4.0)
    sres = least_squares(size_res, s0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05)
    unpack_size(sres.x, model)
    resid, zd_all = size_res(sres.x), model.zd(L, d, ar)
    side_plus = (d - model.size_d0_mm) >= 0
    if side_plus.sum() >= 50 and (~side_plus).sum() >= 50:
        # each side its own scatter: defocused images differ on the two sides of the object plane
        model.size_sigma = _scatter_fit(resid[side_plus], zd_all[side_plus])
        model.size_sigma_minus = _scatter_fit(resid[~side_plus], zd_all[~side_plus])
    else:
        model.size_sigma = _scatter_fit(resid, zd_all)
        model.size_sigma_minus = None

    processing = sorted({str(x) for x in (t.processing if t.processing is not None else []) if x})
    model.fit_stats = {
        "n_particles": int(len(L)), "true_sizes_um": [float(s) for s in sizes],
        "processing": processing or ["legacy (unrecorded)"],
        "excluded_edge_touch": int(np.sum(table.edge_touch == 1)) if table.edge_touch is not None else 0,
        "illum_min": illum_min, "focus_cost": float(fres.cost), "size_cost": float(sres.cost),
        "ratio_at_focus_measured": float(np.median(Lapp[near] / L[near])) if near.any() else None,
        "length_at_focus_um": {f"{s:g}": float(model.mean_length(s, model.size_d0_mm)) for s in sizes},
        "effective_pixel_um_if_offset_correct": float(pixel_um * model.scale),
        "shape_note": shape_note,
        "roundness_range": [float(np.nanpercentile(ar, 5)), float(np.nanpercentile(ar, 95))],
    }
    if model.focus_two_width:
        model.fit_stats["two_width_focus"] = two_width_diagnostics
    model.fit_stats["focus_baseline_mode"] = focus_baseline_mode
    model.fit_stats["focus_fixed_baseline"] = baseline_value
    model.fit_stats["population_balanced_focus"] = bool(population_balanced_focus)
    if width_diagnostics is not None:
        model.fit_stats["empirical_width_constraint"] = width_diagnostics
    if focus_model == "normalised_generalised":
        model.fit_stats["focus_width_comparison"] = _focus_width_comparison(L, d, foc, model, t.shadow_ratio, baseline_value, t.roundness)
    return model


# --------------------------------------------------------------------------- retrieval
def retrieve(model: SizingModel, length, focus, illum=None, side: str = "both", n_L: int = 141, n_d: int = 241,
             ar=None, shadow_ratio=None,
             d_max: Optional[float] = None, chunk: int = 256) -> Dict[str, np.ndarray]:
    """Posterior size and distance for each particle.

    side: "both" (cloud data: marginalise over the side, equal prior weight),
          "+" or "-" (calibration test: side known).
    Priors: uniform in d over [-d_max, d_max] (particles fill the sample volume
    uniformly), uniform in log L.
    """
    length = np.atleast_1d(np.asarray(length, float)) * float(getattr(model, "length_factor", 1.0) or 1.0)
    focus = np.atleast_1d(np.asarray(focus, float))
    n = len(length)
    irel = model.irel(np.full(n, np.nan) if illum is None else np.atleast_1d(np.asarray(illum, float)))
    srv = np.full(n, np.nan) if shadow_ratio is None else np.atleast_1d(np.asarray(shadow_ratio, float))
    if model.focus_shadow_gamma and (shadow_ratio is None or len(srv) != n):
        raise ValueError("Shadow-width calibration requires per-particle shadow/CPI diameter ratio")
    arv = SizingModel.ar_clip(np.full(n, np.nan) if ar is None else np.atleast_1d(np.asarray(ar, float)))
    if d_max is None:
        # The model is only known inside the scanned range; beyond it the focus
        # curve is near its floor, so a wider prior just trades size for distance.
        d_max = max(abs(model.d_range_mm[0]), abs(model.d_range_mm[1]), 0.1)
    if side == "+":
        dg = np.linspace(0.0, d_max, n_d)
    elif side == "-":
        dg = np.linspace(-d_max, 0.0, n_d)
    else:
        dg = np.linspace(-d_max, d_max, n_d)
    unit = np.linspace(0.0, 1.0, n_L)
    out = {k: np.full(n, np.nan) for k in ("L_median", "L_p16", "L_p84", "L_mean", "d_median", "abs_d_median",
                                          "p_plus", "L_plus", "L_minus")}
    out["extrapolated"] = np.zeros(n, bool)
    ok = np.isfinite(length) & (length > model.offset_um + 1.0) & np.isfinite(focus) & (focus > 0)
    if model.focus_shadow_gamma:
        ok &= np.isfinite(srv) & (srv > 0)
    idx = np.flatnonzero(ok)
    sf = model.sigma_focus(dg)[None, None, :]
    for start in range(0, len(idx), chunk):
        sl = slice(start, start + chunk)
        ii = idx[sl]
        top = (length[ii] - model.offset_um) / model.scale * 1.1
        Lg = np.exp(np.log(0.2 * top)[:, None] + unit[None, :] * np.log(5.0))              # (m, nL), log-uniform
        a3 = arv[ii][:, None, None]
        mu = model.mean_length(Lg[:, :, None], dg[None, None, :], irel[ii][:, None, None], a3)  # (m, nL, nd)
        sl_ = model.sigma_length(Lg[:, :, None], dg[None, None, :], a3)
        mu_f = model.mean_focus(dg[None, None, :], irel[ii][:, None, None], Lg[:, :, None], a3, srv[ii][:, None, None] if model.focus_shadow_gamma else None)   # (m, nL, nd)
        lp_f = -0.5 * ((np.log(focus[ii])[:, None, None] - np.log(mu_f)) / sf) ** 2 - np.log(sf)
        lp = (-0.5 * ((np.log(length[ii])[:, None, None] - np.log(np.maximum(mu, 1e-3))) / sl_) ** 2 - np.log(sl_)
              + lp_f)
        lp -= lp.max(axis=(1, 2), keepdims=True)
        post = np.exp(lp)
        post /= post.sum(axis=(1, 2), keepdims=True)
        pL = post.sum(axis=2)
        pd = post.sum(axis=1)
        cL = np.cumsum(pL, axis=1)
        for name, q in (("L_p16", 0.16), ("L_median", 0.5), ("L_p84", 0.84)):
            k = np.argmax(cL >= q, axis=1)
            out[name][ii] = Lg[np.arange(len(ii)), k]
        out["L_mean"][ii] = (pL * Lg).sum(axis=1)
        cd = np.cumsum(pd, axis=1)
        out["d_median"][ii] = dg[np.argmax(cd >= 0.5, axis=1)]
        order = np.argsort(np.abs(dg))
        cad = np.cumsum(pd[:, order], axis=1)
        out["abs_d_median"][ii] = np.abs(dg[order])[np.argmax(cad >= 0.5, axis=1)]
        plus = dg > 0
        out["p_plus"][ii] = pd[:, plus].sum(axis=1) + 0.5 * pd[:, dg == 0].sum(axis=1)
        for name, mask in (("L_plus", dg >= 0), ("L_minus", dg <= 0)):
            if mask.any():
                sub = post[:, :, mask].sum(axis=2)
                tot = sub.sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[name][ii] = np.where(tot > 1e-12, (sub * Lg).sum(axis=1) / tot, np.nan)
    lo, hi = model.size_range_um
    out["extrapolated"] = ok & ((out["L_median"] < 0.7 * lo) | (out["L_median"] > 1.4 * hi)
                                | (out["abs_d_median"] > 0.9 * d_max))
    return out


# --------------------------------------------------------------------------- depth of field
def depth_of_field(model: SizingModel, irel: float = 1.0, foc_crit: Optional[float] = None,
                   d_max: float = 5.0, n: int = 4001, L: Optional[float] = None,
                   ar: Optional[float] = None) -> Dict[str, float]:
    """Depth of field at a focus threshold, both sides.

    ``d_minus``/``d_plus``: where the mean focus curve crosses foc_crit;
    ``total_mm``: their separation; ``acceptance_mm``: integral of
    P(foc > foc_crit | d) over d (what the sample volume really needs, since
    focus scatter lets particles in from beyond the mean crossing and loses
    some inside it).  ``capped`` is True when the curve never drops below the
    threshold within +-d_max.
    """
    crit = model.foc_crit if foc_crit is None else foc_crit
    dg = np.linspace(-d_max, d_max, n) + model.focus_d0_mm
    mu = model.mean_focus(dg, irel, L, ar=ar)
    above = mu > crit
    out = {"foc_crit": float(crit), "irel": float(irel), "capped": bool(above[0] or above[-1]),
           "L_um": float(L) if L is not None else None, "Ar": ar}
    if not above.any():
        out.update(d_minus=float("nan"), d_plus=float("nan"), total_mm=0.0, acceptance_mm=0.0)
        return out
    first, last = np.flatnonzero(above)[[0, -1]]
    out["d_minus"], out["d_plus"] = float(dg[first]), float(dg[last])
    out["total_mm"] = out["d_plus"] - out["d_minus"]
    z = (np.log(mu) - math.log(max(crit, 1e-9))) / model.sigma_focus(dg)
    out["acceptance_mm"] = float(np.trapezoid(ndtr(z), dg) if hasattr(np, "trapezoid") else np.trapz(ndtr(z), dg))
    return out


def dof_table(model: SizingModel, irels: Iterable[float] = (0.6, 0.8, 1.0, 1.2, 1.4),
              foc_crit: Optional[float] = None, L: Optional[float] = None) -> List[Dict[str, float]]:
    return [depth_of_field(model, r, foc_crit, L=L) for r in irels]


# --------------------------------------------------------------------------- pixel size
def pixel_scale_from_tracking(table: CalibrationTable, min_shift_mm: float = 0.005,
                              gap_days: float = 30.0 / 1440.0) -> Dict[str, object]:
    """Micrometres per pixel from the image shift of slide-mounted particles between stage steps.

    Particles stuck to the slide move rigidly; between two stage positions the
    in-plane component of the move (d_in_plane) shifts every image by the same
    vector.  The common shift is the mode of all pairwise centre differences.
    Steps whose in-plane offset does not change carry no information and are skipped.
    """
    ok = np.isfinite(table.d_in_plane) & np.isfinite(table.image_x) & np.isfinite(table.image_y) \
        & np.isfinite(table.x_stage) & np.isfinite(table.z_stage) & np.isfinite(table.time)
    if ok.sum() < 10:
        return {"ok": False, "reason": "no stage positions / in-plane offsets in the table"}
    t = table.subset(ok)
    key = np.round(t.x_stage, 4) * 1e6 + np.round(t.z_stage, 4)
    steps = []
    for k in np.unique(key):
        m = key == k
        # a stage position may be revisited; split visits by time gaps
        tt = t.time[m]
        order = np.argsort(tt)
        cuts = np.flatnonzero(np.diff(tt[order]) > gap_days) + 1
        for part in np.split(np.flatnonzero(m)[order], cuts):
            steps.append(part)
    steps.sort(key=lambda p: float(np.median(t.time[p])))
    shifts = []
    for a, b in zip(steps[:-1], steps[1:]):
        if np.median(t.time[b]) - np.median(t.time[a]) > gap_days:
            continue
        delta_um = (np.median(t.d_in_plane[b]) - np.median(t.d_in_plane[a])) * 1000.0
        if abs(delta_um) < min_shift_mm * 1000.0 or len(a) < 2 or len(b) < 2:
            continue
        P = np.column_stack((t.image_x[a], t.image_y[a]))
        Q = np.column_stack((t.image_x[b], t.image_y[b]))
        D = (Q[None, :, :] - P[:, None, :]).reshape(-1, 2)
        reach = abs(delta_um) / NOMINAL_PIXEL_UM * 2.0 + 30.0
        D = D[np.hypot(D[:, 0], D[:, 1]) < reach]
        if len(D) < 3:
            continue
        bins = np.arange(-reach, reach + 2, 2.0)
        H, xe, ye = np.histogram2d(D[:, 0], D[:, 1], bins=(bins, bins))
        i, j = np.unravel_index(np.argmax(H), H.shape)
        centre = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
        close = D[np.hypot(*(D - centre).T) < 3.0]
        if len(close) < 3:
            continue
        shift = np.median(close, axis=0)
        shifts.append((delta_um, shift[0], shift[1], len(close)))
    if len(shifts) < 3:
        return {"ok": False, "reason": "fewer than 3 stage steps that move the slide in the image plane",
                "n_steps": len(shifts)}
    S = np.array(shifts)
    keep = np.ones(len(S), bool)
    for _ in range(3):
        w = S[keep, 3]
        a = np.sum(w * S[keep, 0] * S[keep, 1]) / np.sum(w * S[keep, 0] ** 2)
        c = np.sum(w * S[keep, 0] * S[keep, 2]) / np.sum(w * S[keep, 0] ** 2)
        resid = np.hypot(S[:, 1] - a * S[:, 0], S[:, 2] - c * S[:, 0])
        mad = np.median(resid[keep]) + 0.5
        keep = resid < 4 * mad
    px_per_um = math.hypot(a, c)
    return {"ok": True, "pixel_um": 1.0 / px_per_um, "direction_deg": math.degrees(math.atan2(c, a)),
            "n_steps": int(keep.sum()), "rms_residual_px": float(np.sqrt(np.mean(resid[keep] ** 2))),
            "shifts": S.tolist(), "used": keep.tolist()}


# --------------------------------------------------------------------------- validation
def _folds(table: CalibrationTable, k_default: int = 5):
    groups = np.array([g if g else "" for g in table.group], object)
    names = [g for g in np.unique(groups) if g]
    if len(names) >= 3 and all(groups != ""):
        return [(g, groups == g) for g in names]
    order = np.argsort(np.where(np.isfinite(table.time), table.time, np.arange(len(table))))
    blocks = np.array_split(order, k_default)
    out = []
    for i, b in enumerate(blocks):
        m = np.zeros(len(table), bool)
        m[b] = True
        out.append((f"block {i + 1}", m))
    return out


def validate(table: CalibrationTable, illum_min: Optional[float] = None, **fit_kw) -> Dict[str, object]:
    """Leave-one-batch-out: fit without a batch, retrieve its particles both ways."""
    use = _usable(table, illum_min)
    t = table.subset(use)
    n = len(t)
    res = {k: np.full(n, np.nan) for k in ("L_known", "L_both", "d_known", "absd_both", "p_plus",
                                           "L_known_p16", "L_known_p84", "L_both_p16", "L_both_p84")}
    folds = _folds(t)
    for name, held in folds:
        train = t.subset(~held)
        if len(train) < 20 or held.sum() == 0:
            continue
        m = fit_model(train, illum_min=None, **fit_kw)
        test = t.subset(held)
        idx = np.flatnonzero(held)
        for side_name, sign in (("+", 1), ("-", -1)):
            sm = (np.sign(test.d) >= 0) if sign > 0 else (test.d < 0)
            if sm.any():
                r = retrieve(m, test.length[sm], test.focus[sm], test.illum[sm], side=side_name, shadow_ratio=test.shadow_ratio[sm] if test.shadow_ratio is not None else None,
                             ar=None if test.roundness is None else test.roundness[sm])
                res["L_known"][idx[sm]] = r["L_median"]
                res["L_known_p16"][idx[sm]] = r["L_p16"]
                res["L_known_p84"][idx[sm]] = r["L_p84"]
                res["d_known"][idx[sm]] = r["d_median"]
        r = retrieve(m, test.length, test.focus, test.illum, side="both", ar=test.roundness, shadow_ratio=test.shadow_ratio)
        res["L_both"][idx] = r["L_median"]
        res["L_both_p16"][idx] = r["L_p16"]
        res["L_both_p84"][idx] = r["L_p84"]
        res["absd_both"][idx] = r["abs_d_median"]
        res["p_plus"][idx] = r["p_plus"]

    L = t.true_length

    def err(est):
        e = (est - L) / L
        g = np.isfinite(e)
        if not g.any():
            return {}
        return {"mean_abs": float(np.mean(np.abs(e[g]))), "bias": float(np.median(e[g])),
                "sd": float(np.std(e[g])), "n": int(g.sum())}

    def covered(lo, hi):
        g = np.isfinite(lo) & np.isfinite(hi)
        return float(np.mean((L[g] >= lo[g]) & (L[g] <= hi[g]))) if g.any() else float("nan")

    def corr(a, b):
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() < 3:
            return {}
        r = float(np.corrcoef(a[g], b[g])[0, 1])
        slope = float(np.polyfit(a[g], b[g], 1)[0])
        return {"r": r, "slope": slope}

    # Sizing error vs processing focus threshold: what foc_crit buys what accuracy.
    e_both = np.abs(res["L_both"] - L) / L
    thresholds = []
    good = np.isfinite(e_both)
    for thr in np.unique(np.round(np.quantile(t.focus[good], np.linspace(0, 0.8, 9)), 1)) if good.any() else []:
        sel = good & (t.focus >= thr)
        if sel.sum() >= 10:
            thresholds.append({"foc_crit": float(thr), "kept_fraction": float(sel.sum() / good.sum()),
                               "mean_abs": float(np.mean(e_both[sel])), "p90_abs": float(np.quantile(e_both[sel], 0.9))})
    target = 0.05
    suggested = next((row["foc_crit"] for row in thresholds if row["mean_abs"] <= target), None)
    per_size = []
    for size in np.unique(np.round(L, 1)):
        k = np.round(L, 1) == size
        per_size.append({"size_um": float(size), "n": int(k.sum()),
                         "uncorrected": float(np.nanmean(np.abs(t.length[k] - L[k]) / L[k])),
                         "known_side": float(np.nanmean(np.abs(res["L_known"][k] - L[k]) / L[k])),
                         "marginal_side": float(np.nanmean(np.abs(res["L_both"][k] - L[k]) / L[k])),
                         "interval_width": float(np.nanmedian((res["L_both_p84"][k] - res["L_both_p16"][k]) / L[k]))})
    per_label = []
    labels = t.label if t.label is not None else np.array([""] * n, object)
    for name in sorted({str(x) for x in labels}):
        k = np.array([str(x) == name for x in labels])
        if k.sum() < 5:
            continue
        rel = lambda est: (est[k] - L[k]) / L[k]
        per_label.append({"label": name or "(no label)", "n": int(k.sum()), "true_um": float(np.median(L[k])),
                          "roundness": (float(np.nanmedian(t.roundness[k]))
                                        if t.roundness is not None and np.isfinite(t.roundness[k]).any() else float("nan")),
                          "uncorrected": float(np.nanmean(np.abs(rel(t.length)))),
                          "known_side": float(np.nanmean(np.abs(rel(res["L_known"])))),
                          "marginal_side": float(np.nanmean(np.abs(rel(res["L_both"])))),
                          "bias_marginal": float(np.nanmedian(rel(res["L_both"])))})
    side_ok = np.isfinite(res["p_plus"]) & (np.abs(t.d) > 1e-9)
    summary = {
        "folds": [f for f, _ in folds], "n": n,
        "uncorrected": err(t.length),
        "known_side": err(res["L_known"]), "marginal_side": err(res["L_both"]),
        "coverage_68_known": covered(res["L_known_p16"], res["L_known_p84"]),
        "coverage_68_marginal": covered(res["L_both_p16"], res["L_both_p84"]),
        "d_known": corr(t.d, res["d_known"]), "abs_d_marginal": corr(np.abs(t.d), res["absd_both"]),
        "side_correct_fraction": float(np.mean((res["p_plus"][side_ok] > 0.5) == (t.d[side_ok] > 0)))
        if side_ok.any() else float("nan"),
        "error_vs_foc_crit": thresholds, "per_size": per_size, "per_label": per_label,
        "suggested_foc_crit_5pct": suggested,
        "side_confident_fraction": float(np.mean(np.abs(res["p_plus"][side_ok] - 0.5) > 0.4))
        if side_ok.any() else float("nan"),
    }
    return {"summary": summary, "table": t, "per_particle": res}


# --------------------------------------------------------------------------- report
def format_report(model: SizingModel, validation: Optional[dict] = None, tracking: Optional[dict] = None) -> str:
    fs = model.fit_stats
    lines = [f"Particles fitted: {fs.get('n_particles')}   true sizes: {fs.get('true_sizes_um')} µm   "
             f"size measure: {SIZE_MEASURES.get(model.size_measure, {}).get('label', model.size_measure)}; "
             f"focus: {FOCUS_MEASURES.get(getattr(model, 'focus_measure', 'focus'), {}).get('label', 'focus')}",
             f"Processing: {', '.join(fs.get('processing', ['?']))}"
             + (f"   ({fs['excluded_edge_touch']} particles cut off by the ROI border left out)"
                if fs.get("excluded_edge_touch") else "")]
    if model.inputs:
        lines.append(f"From {len(model.inputs)} project(s):")
        for item in model.inputs:
            lines.append(f"  {item['project']}: {item.get('particles', '?')} particles, saved "
                         f"{item.get('project_saved_utc', '?')}")
            for w in item.get("warnings", ()):
                lines.append(f"    ⚠ {w}")
    if model.focus_shadow_gamma:
        lines.append(f"Experimental focus-width shadow/CPI ratio: exponent {model.focus_shadow_gamma:+.3f}, reference {model.focus_shadow_ref:.3f}; measured shadow and CPI diameters required at retrieval")
    if model.instrument:
        lines.append("Instrument: " + ", ".join(f"{k}={v}" for k, v in model.instrument.items()))
    if getattr(model, "pixel_mode", "processed") != "processed":
        lines.append(f"Pixel size: processed at {model.pixel_um:.3f} µm, {model.pixel_mode} "
                     f"{model.pixel_um * model.length_factor:.4f} µm (measured sizes × {model.length_factor:.4f}; "
                     "scale s fixed at 1)")
    if fs.get("focus_baseline_mode"):
        lines.append("Focus baseline mode: " + str(fs["focus_baseline_mode"]) +
                     (f"; fixed B={fs['focus_fixed_baseline']:.4f}" if fs.get("focus_fixed_baseline") is not None else "; fitted freely"))
    if fs.get("population_balanced_focus"):
        lines.append("Population-balanced focus fit: stage-median profiles equally weighted by population; peak-height-above-baseline and separate negative/positive half-height widths constrained.")
    wd = fs.get("empirical_width_constraint")
    if wd:
        if wd.get("used"):
            lines.append("Empirical focus FWHM constraint: target exponent "
                         f"{wd['target_exponent']:.3f}, fitted {wd['fitted_exponent']:.3f}, "
                         f"difference {wd['exponent_difference']:+.3f}; "
                         f"{len(wd['profiles'])} size populations")
            lines.extend(f"  {r['size_um']:.1f} µm: FWHM {r['fwhm_mm']:.3f} mm "
                         f"(n={r['count']})" for r in wd['profiles'])
        else:
            lines.append("Empirical focus FWHM constraint requested but not applied: "
                         "fewer than two suitable populations with both half-height crossings.")
    comparisons = fs.get("focus_width_comparison", [])
    if comparisons:
        lines.extend(["", "Focus half-height widths: observed versus model (mm)",
                      "Observed crossings use stage medians; missing values are not extrapolated.",
                      "Size (µm) | n | Median Ar | Observed − / + | Model − / + | FWHM observed / model | Model difference"])
        def _fmt(v):
            return f"{v:.3f}" if v is not None and np.isfinite(v) else "—"
        for r in comparisons:
            diff = f"{r['difference_pct']:+.1f}%" if r['difference_pct'] is not None else "—"
            lines.append(
                f"{r['size_um']:8.1f} | {r['count']:5d} | {r['median_ar']:.3f} | "
                f"{_fmt(r['observed_minus_mm'])} / {_fmt(r['observed_plus_mm'])} | "
                f"{_fmt(r['model_minus_mm'])} / {_fmt(r['model_plus_mm'])} | "
                f"{_fmt(r['observed_fwhm_mm'])} / {_fmt(r['model_fwhm_mm'])} | {diff}")
        lines.append("Difference = 100 × (model FWHM / observed FWHM − 1).")
    lines += [
        f"Fit options: size={model.size_model}; focus={model.focus_model}; p={model.focus_q:.3f}; "
        f"amplitude size exponent={model.focus_L_amp:.3f}; width size exponent={model.focus_L_width:.3f}; shadow-ratio exponent={model.focus_shadow_gamma:.3f}",
        "",
        "Shadow-circle size calibration" if model.size_model == "shadow_linear" else "Length at best focus",
        f"  measured median L_app/L near focus: {fs.get('ratio_at_focus_measured', float('nan')):.3f}",
        (f"  model: D_shadow = {model.scale:.4f}·L + {model.offset_um:.2f} µm (no Z term)"
         if model.size_model == "shadow_linear" else
         f"  model: L_app = {model.scale:.4f}·L·(…) + {model.offset_um:.2f} µm")
        + ("  (offset fitted)" if model.offset_fitted else
           "  (offset fixed)" if model.size_model == "shadow_linear" else
           "  (offset fixed = processing mask pad)"),
        f"  length at focus: " + ", ".join(f"{k} µm → {v:.1f} µm" for k, v in fs.get("length_at_focus_um", {}).items()),
        (f"  size assumed independent of Z (focus peak at {model.focus_d0_mm * 1000:+.0f} µm)"
         if model.size_model == "shadow_linear" else
         f"  size minimum at d0 = {model.size_d0_mm * 1000:+.0f} µm (focus peak at {model.focus_d0_mm * 1000:+.0f} µm)"),
        "",
        "Focus",
        (f"  F_raw/I_local = {model.focus_floor:.4f} + {model.focus_F0:.4f} × " +
         (f"[{model.focus_narrow_fraction:.3f} exp(-(|d-d0|/({model.focus_narrow_scale:.3f} w±))^{model.focus_q:.2f}) + "
          f"{1-model.focus_narrow_fraction:.3f} exp(-(|d-d0|/w±)^{model.focus_q:.2f})]"
          if model.focus_two_width else f"exp[-(|d-d0|/w±(L))^{model.focus_q:.2f}]")
         if model.focus_model == "normalised_generalised" else
         f"  F0 = {model.focus_F0:.1f}·(I/I_ref)^{model.focus_gamma:.2f}, I_ref = {model.illum_ref:.1f}"),
        f"  half-widths: + {model.focus_w_plus_mm:.3f} mm, − {model.focus_w_minus_mm:.3f} mm, shape q = "
        f"{model.focus_q:.2f}, floor {model.focus_floor:.2f}",
        f"  scatter σ(log foc) = {model.focus_sigma[0]:.3f} + {model.focus_sigma[1]:.3f}·|d|",
        (f"  size dependence: peak × (L/{model.L_ref_um:.0f} µm)^{model.focus_L_amp:+.2f}, "
         f"half-widths × (L/{model.L_ref_um:.0f} µm)^{model.focus_L_width:+.2f}"
         if len(fs.get("true_sizes_um", [])) > 1 else "  size dependence: no slope identifiable (single true size)"),
        (f"  focus AR dependence: peak × Ar^{model.focus_Ar_power:+.2f}; "
         f"both widths × Ar^{model.focus_Ar_width_power:+.2f}"
         if (abs(model.focus_Ar_power) > 1e-9 or abs(model.focus_Ar_width_power) > 1e-9) else
         "  focus AR dependence: none"),
        "",
        f"Depth of field at {'F_raw/I_local' if model.focus_model == 'normalised_generalised' else 'raw focus'} "
        f"threshold = {model.foc_crit:g}",
    ]
    if model.size_model == "shadow_linear":
        lines[lines.index("Focus"):lines.index("Focus")] = [
            "No contour-based defocus correction is applied to the circle-shadow size proxy.", ""]
    else:
        lines[lines.index("Focus"):lines.index("Focus")] = [
        "Oversizing away from focus (Zd = λ|d|/(L/2)²" + (
            f"·(L/{model.L_ref_um:.0f} µm)^{2 - model.size_L_power:+.2f})" if model.size_L_power != 2.0 else ")"),
        (f"  size scaling: Zd ∝ |d|/L^{model.size_L_power:.2f} (fitted; 2 = Fresnel, Z/L²)"
         if model.size_L_power_fitted else "  size scaling: Z/L² assumed (fit two or more true sizes to test it)"),
        (f"  particle shape: Zd / Ar^{model.size_Ar_power:.2f} (fitted; 1 = Park et al. 2026, Z/(L² Ar)), "
         f"Ar {fs.get('roundness_range', [0, 0])[0]:.2f}–{fs.get('roundness_range', [0, 0])[1]:.2f} in the data"
         if getattr(model, "shape_terms", False) else
         "  particle shape: not used" + (f" ({fs['shape_note']})" if fs.get("shape_note") else "")),
        f"  + side: A = {model.size_plus['A']:.3g}, p = {model.size_plus['p']:.2f}",
        f"  − side: A = {model.size_minus['A']:.3g}, p = {model.size_minus['p']:.2f}",
        f"  illumination exponent κ = {model.size_kappa:.2f};  scatter σ(log L_app) = "
        + (f"+ side {model.size_sigma[0]:.3f} + {model.size_sigma[1]:.3f}·Zd, − side "
           f"{model.size_sigma_minus[0]:.3f} + {model.size_sigma_minus[1]:.3f}·Zd" if model.size_sigma_minus else
           f"{model.size_sigma[0]:.3f} + {model.size_sigma[1]:.3f}·Zd"),
        "",
        ]
    sizes = fs.get("true_sizes_um") or [None]
    for size in sizes:
        if len(sizes) > 1:
            lines.append(f"  {size:g} µm particles:")
        for row in (dof_table(model, irels=(1.0,), L=size)
                    if model.focus_model == "normalised_generalised" else dof_table(model, L=size)):
            cap = "  (≥, curve never reaches threshold)" if row["capped"] else ""
            lines.append(f"  I/I_ref {row['irel']:.1f}: {row['d_minus']:+.2f} … {row['d_plus']:+.2f} mm, "
                         f"total {row['total_mm']:.2f} mm, acceptance-weighted {row['acceptance_mm']:.2f} mm{cap}")
    if tracking is not None:
        lines += ["", "Pixel size from bead tracking"]
        if tracking.get("ok"):
            lines.append(f"  {tracking['pixel_um']:.3f} µm/px from {tracking['n_steps']} stage steps "
                         f"(rms {tracking['rms_residual_px']:.2f} px, shift direction {tracking['direction_deg']:.0f}°)")
        else:
            lines.append(f"  not available: {tracking.get('reason')}")
    if validation is not None:
        s = validation["summary"]

        def e(name):
            v = s.get(name) or {}
            return (f"mean |ΔL|/L {v.get('mean_abs', float('nan')) * 100:.1f}%, bias {v.get('bias', float('nan')) * 100:+.1f}%, "
                    f"sd {v.get('sd', float('nan')) * 100:.1f}%") if v else "n/a"

        lines += ["", f"Leave-one-batch-out validation ({len(s['folds'])} folds, {s['n']} particles)",
                  f"  uncorrected:            {e('uncorrected')}",
                  f"  retrieved, side known:  {e('known_side')}   68% interval covers truth "
                  f"{s['coverage_68_known'] * 100:.0f}%",
                  f"  retrieved, side unknown:{e('marginal_side')}   68% interval covers truth "
                  f"{s['coverage_68_marginal'] * 100:.0f}%"]
        if len(s.get("per_size", [])) > 1:
            lines.append("  by true size (mean |ΔL|/L):            uncorrected  side known  side unknown  68% range")
            for row in s["per_size"]:
                worse = "   ← not helping at this size" if row["marginal_side"] > row["uncorrected"] else ""
                lines.append(f"    {row['size_um']:6g} µm (n={row['n']:5d}):            {row['uncorrected'] * 100:6.1f}%"
                             f"      {row['known_side'] * 100:6.1f}%       {row['marginal_side'] * 100:6.1f}%"
                             f"     ±{row['interval_width'] * 50:4.0f}%{worse}")
        if len(s.get("per_label", [])) > 1:
            lines.append("  by particle type (mean |ΔL|/L):      Ar   uncorrected  side known  side unknown  bias")
            for row in s["per_label"]:
                lines.append(f"    {row['label'][:16]:16s} {row['true_um']:5.0f} µm (n={row['n']:5d})  {row['roundness']:4.2f}"
                             f"     {row['uncorrected'] * 100:6.1f}%     {row['known_side'] * 100:6.1f}%"
                             f"       {row['marginal_side'] * 100:6.1f}%  {row['bias_marginal'] * 100:+5.1f}%")
        if s.get("d_known"):
            lines.append(f"  d (side known): r = {s['d_known']['r']:.3f}, slope {s['d_known']['slope']:.2f}")
        if s.get("abs_d_marginal"):
            lines.append(f"  |d| (side unknown): r = {s['abs_d_marginal']['r']:.3f}, slope "
                         f"{s['abs_d_marginal']['slope']:.2f}")
        if s.get("error_vs_foc_crit"):
            lines.append("  side-unknown error vs processing focus threshold:")
            for row in s["error_vs_foc_crit"]:
                lines.append(f"    foc ≥ {row['foc_crit']:5.1f}: keeps {row['kept_fraction'] * 100:3.0f}%, "
                             f"mean |ΔL|/L {row['mean_abs'] * 100:4.1f}%, 90th pct {row['p90_abs'] * 100:4.1f}%")
            sug = s.get("suggested_foc_crit_5pct")
            if sug is not None:
                d = depth_of_field(model, 1.0, sug)
                lines.append(f"  → foc_crit {sug:g} gives ≤5% mean error; DOF there {d['total_mm']:.2f} mm "
                             f"(acceptance-weighted {d['acceptance_mm']:.2f} mm)")
        lines.append(f"  side identified from size+focus: {s['side_correct_fraction'] * 100:.0f}% correct, "
                     f"{s['side_confident_fraction'] * 100:.0f}% with P > 0.9")
    return "\n".join(lines)


def paired_contour_shadow_errors(table: CalibrationTable):
    """Uncorrected diameter/label errors for *the same* ROIs.

    Shadow is the exact stored/temporarily computed signed integral used by
    Edge Lab; it is not the model-corrected estimate and no true-size label
    enters its calculation. Use contour equivalent-area diameter if present,
    otherwise fall back to maximum Feret length. Never mix rows with missing
    paired measurements or compare disjoint populations.
    """
    if table.size_measure not in ("shadow_deq", "area_corrected_shadow"):
        return None
    true = np.asarray(table.true_length, float)
    shadow = np.asarray(table.length, float)
    edge_deq = (np.full_like(true, np.nan) if table.edge_deq is None
                else np.asarray(table.edge_deq, float))
    edge_length = (np.full_like(true, np.nan) if table.edge_length is None
                   else np.asarray(table.edge_length, float))
    # Prefer the contour equivalent-area diameter, suitable for bead trials.
    contour = edge_deq
    contour_label = "Contour equivalent-area diameter"
    if not np.any(np.isfinite(edge_deq) & (edge_deq > 0)):
        contour = edge_length
        contour_label = "Contour maximum Feret length"
    both = (np.isfinite(true) & (true > 0) & np.isfinite(shadow) & (shadow > 0) &
            np.isfinite(contour) & (contour > 0))
    return (contour[both] / true[both] - 1.,
            shadow[both] / true[both] - 1., contour_label, int(both.sum()))


def make_figure(model: SizingModel, table: CalibrationTable, validation: Optional[dict] = None,
                tracking: Optional[dict] = None, figure=None,
                normalise_focus_peak: bool = False, focus_plot_reference: str = "peak"):
    """Six-panel diagnostic figure.  Pass an existing matplotlib Figure to draw into it."""
    if figure is None:
        from matplotlib.figure import Figure
        figure = Figure(figsize=(13, 8), layout="constrained")
    figure.clear()
    # The embedded Qt canvas is frequently resized; constrained_layout can
    # collapse the axes when long bin labels/equations contribute to margins.
    # Explicit fractional positions keep the six panels aligned and legible.
    figure.set_layout_engine(None)
    eq_ax = figure.add_axes([0.07, 0.84, 0.90, 0.15])
    columns = ((0.13, 0.225), (0.435, 0.225), (0.74, 0.225))
    axes = np.array([[figure.add_axes([x, y, w, h]) for x, w in columns]
                     for y, h in ((0.525, 0.255), (0.135, 0.29))])
    draw_equations(eq_ax, model)
    use = _usable(table, None)
    t = table.subset(use)
    t.length = t.length * float(getattr(model, "length_factor", 1.0) or 1.0)    # on the model's pixel size
    irel = model.irel(t.illum)
    dg = np.linspace(min(t.d.min(), -0.05), max(t.d.max(), 0.05), 300)
    Lref = float(np.median(t.true_length))

    sizes = np.unique(np.round(t.true_length, 1))
    multi = len(sizes) > 1
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]

    ax = axes[0, 0]
    if model.size_model == "shadow_linear":
        for k, size in enumerate(sizes):
            mask = np.round(t.true_length, 1) == size
            ax.scatter(t.d[mask], t.length[mask] / t.true_length[mask], s=4,
                       color=colours[k % 6], alpha=.3, rasterized=True, label=f"{size:g} µm")
            ax.plot(dg, model.mean_length(size, dg) / size * np.ones_like(dg),
                    color=_darker(colours[k % 6]), lw=2)
        ax.axhline(1, color="k", lw=.6)
        ax.set_xlabel("signed distance d [mm]")
        ax.set_title("Circle signed-shadow diameter / true size")
        ax.legend(fontsize=7)
    else:
        if multi:
            # All sizes on one axis: signed Zd (∝ Z/L^m).  If the scaling is right, they collapse.
            sign = np.sign(t.d - model.size_d0_mm)
            for k, size in enumerate(sizes):
                m = np.round(t.true_length, 1) == size
                ax.scatter(sign[m] * model.zd(t.true_length[m], t.d[m]), t.length[m] / t.true_length[m], s=4,
                           color=colours[k % 6], alpha=0.5, rasterized=True, label=f"{size:g} µm")
                ds = np.linspace(min(t.d[m].min(), -0.05), max(t.d[m].max(), 0.05), 200)
                ax.plot(np.sign(ds - model.size_d0_mm) * model.zd(size, ds), model.mean_length(size, ds) / size,
                        color=_darker(colours[k % 6]), lw=2.0, zorder=5, path_effects=_HALO)
            ax.set_xlabel(f"signed Zd = ±λ|d|/(L/2)²" + (f"·(L/L_ref)^{2 - model.size_L_power:+.2f}"
                                                        if model.size_L_power != 2.0 else ""))
            ax.set_title(f"L_app/L, all sizes (m = {model.size_L_power:.2f})", fontsize=10)
            ax.legend(fontsize=7, markerscale=3)
        else:
            sc = ax.scatter(t.d, t.length / t.true_length, s=4, c=irel, cmap="viridis", vmin=0.6, vmax=1.4, alpha=0.6,
                            rasterized=True)
            for r, ls in ((0.8, ":"), (1.0, "-"), (1.2, "--")):
                ax.plot(dg, model.mean_length(Lref, dg, r) / Lref, "r", ls=ls, lw=1.4, label=f"model I/I_ref={r}")
            ax.set_xlabel("signed distance d [mm]")
            ax.set_title(f"L_app / L ({Lref:g} µm)", fontsize=10)
            ax.legend(fontsize=7)
            figure.colorbar(sc, ax=ax, label="I/I_ref")
        ax.axhline(1.0, color="k", lw=0.6)
    ax.set_ylabel("L_app / L")
    if getattr(model, "length_factor", 1.0) != 1.0:
        ax.text(0.02, 0.02, f"pixel {model.pixel_um * model.length_factor:.3f} µm ({model.pixel_mode})",
                transform=ax.transAxes, fontsize=7, color="#555")

    ax = axes[0, 1]
    # Focus divided by the fitted illumination dependence (I/I_ref)^γ: every particle shown as if at I_ref,
    # so the scatter left is what the illumination does not explain.  Curves are the model at I_ref.
    focus_n = (t.focus if model.focus_model == "normalised_generalised"
               else t.focus / np.power(np.clip(irel, 0.05, 20), model.focus_gamma))
    # Data peak is determined from stage-position medians, not from a single noisy ROI.
    # The model curve is divided by its own peak so each series reaches unity.
    def _data_peak(mask):
        x = np.asarray(t.d[mask], dtype=float)
        y = np.asarray(focus_n[mask], dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(good) < 5:
            return np.nan
        x, y = x[good], y[good]
        stages = np.round(x, 4)
        med = [np.median(y[stages == z]) for z in np.unique(stages)
               if np.count_nonzero(stages == z) >= 5]
        if not med:
            return np.nan
        peak = float(np.max(med))
        return peak if np.isfinite(peak) and peak > 0 else np.nan

    # Plot-only normalisation: use the median of stage medians closest to the
    # physical object plane, rather than the empirical maximum stage median.
    # None of these reference values enters the focus fit or DOF calculation.
    def _zero_reference(mask):
        x = np.asarray(t.d[mask], dtype=float)
        y = np.asarray(focus_n[mask], dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(good) < 5:
            return _data_peak(mask)
        x, y = x[good], y[good]
        centres = np.round(x, 4)
        stage_data = [(float(z), float(np.median(y[centres == z])))
                      for z in np.unique(centres) if np.count_nonzero(centres == z) >= 5]
        if not stage_data:
            return _data_peak(mask)
        # Prefer stage positions within ±0.075 mm of d=0. If the scan did
        # not sample this interval, do not silently normalise at a remote stage.
        close = [v for z, v in stage_data if abs(z) <= .075]
        if not close:
            return _data_peak(mask)
        result = float(np.median(close))
        return result if np.isfinite(result) and result > 0 else _data_peak(mask)

    def _plot_ref(mask):
        if not normalise_focus_peak:
            return 1.0
        return (_zero_reference(mask) if focus_plot_reference == 'zero_median'
                else _data_peak(mask))

    def _model_ref(size, shadow_ratio=None, ar=None):
        if not normalise_focus_peak:
            return 1.0
        if focus_plot_reference == 'zero_median':
            value = float(np.asarray(model.mean_focus(np.asarray([0.0]), 1.0, size,
                                                     ar=ar, shadow_ratio=shadow_ratio)).ravel()[0])
            return value if np.isfinite(value) and value > 0 else 1.0
        return None  # peak handled from evaluated curve below

    if multi:
        for k, size in enumerate(sizes):                 # all points first, then the model lines on top
            m = np.round(t.true_length, 1) == size
            peak = _plot_ref(m)
            ax.scatter(t.d[m], focus_n[m] / peak, s=4, color=colours[k % 6], alpha=0.35, rasterized=True, zorder=1)
        for k, size in enumerate(sizes):
            ratio = (float(np.nanmedian(t.shadow_ratio[np.round(t.true_length, 1) == size]))
                     if model.focus_shadow_gamma else None)
            pop_ar = float(np.nanmedian(t.roundness[np.round(t.true_length, 1) == size]))
            curve = np.asarray(model.mean_focus(dg, 1.0, size, ar=pop_ar, shadow_ratio=ratio), dtype=float)
            if normalise_focus_peak:
                ref_fit = (_model_ref(size, ratio, pop_ar) if focus_plot_reference == 'zero_median'
                           else float(np.nanmax(curve)))
                if np.isfinite(ref_fit) and ref_fit > 0:
                    curve = curve / ref_fit
            ax.plot(dg, curve, color=_darker(colours[k % 6]), lw=2.0, zorder=5,
                    path_effects=_HALO)
        # Show empirical stage-position medians: individual-frame density can
        # obscure a narrower small-particle response in the scatter cloud.
        for k, size in enumerate(sizes):
            mask = np.round(t.true_length, 1) == size
            pos = np.unique(np.round(t.d[mask], 4))
            if len(pos) > 80:
                pos = np.unique(np.round(t.d[mask] / .025) * .025)
                centres = np.round(t.d[mask] / .025) * .025
            else:
                centres = np.round(t.d[mask], 4)
            xp, yp = [], []
            values = focus_n[mask] / _plot_ref(mask)
            for x in pos:
                group = values[centres == x]
                if len(group) >= 5:
                    xp.append(x)
                    yp.append(np.median(group))
            if xp:
                ax.plot(xp, yp, linestyle="none", marker="o", markersize=3.5,
                        markerfacecolor="none", markeredgewidth=1.0,
                        color=_darker(colours[k % 6]), zorder=6)
        # No focus legend: the many populations previously obscured the peak.
        width_rows = []
        for k, size in enumerate(sizes):
            mask = np.round(t.true_length, 1) == size
            x = np.asarray(t.d[mask], float)
            y = np.asarray(focus_n[mask] / _plot_ref(mask), float)
            if mask.sum() < 50:
                continue
            # Medians in fixed distance bins suppress unequal stage population.
            edges = np.arange(np.nanmin(x)-0.013, np.nanmax(x)+0.038, .025)
            centres = (edges[:-1] + edges[1:]) / 2
            medians = np.array([np.nanmedian(y[(x >= a) & (x < b)]) if ((x >= a) & (x < b)).sum() >= 5
                                else np.nan for a,b in zip(edges[:-1],edges[1:])])
            good = np.isfinite(medians)
            if good.sum() < 8:
                continue
            floor = float(np.nanpercentile(medians[good], 10))
            peak = float(np.nanmax(medians[good]))
            if peak <= floor:
                continue
            thresh = floor + .5 * (peak-floor)
            nearest = np.nanargmax(medians)
            crossings = []
            for direction in (-1, 1):
                j = nearest
                while 0 <= j+direction < len(medians) and np.isfinite(medians[j+direction]) and medians[j+direction] >= thresh:
                    j += direction
                nxt = j+direction
                if not (0 <= nxt < len(medians)) or not np.isfinite(medians[nxt]):
                    crossings.append(np.nan)
                else:
                    crossings.append(float(np.interp(thresh,
                        sorted([medians[j],medians[nxt]]),
                        [centres[j],centres[nxt]] if medians[j] < medians[nxt] else [centres[nxt],centres[j]])))
            if np.all(np.isfinite(crossings)):
                fwhm = abs(crossings[1]-crossings[0])
                width_rows.append((float(size),fwhm))
                for c in crossings:
                    ax.plot(c,thresh, marker="|", ms=9, mew=1.4, color=_darker(colours[k % 6]), zorder=8)
    else:
        single_peak = _plot_ref(np.ones(len(t.d), dtype=bool))
        ax.scatter(t.d, focus_n / single_peak, s=4, c=irel, cmap="viridis", vmin=0.6, vmax=1.4, alpha=0.6, rasterized=True)
        curve = np.asarray(model.mean_focus(dg, 1.0), float)
        if normalise_focus_peak:
            ref_fit = (_model_ref(float(Lref)) if focus_plot_reference == 'zero_median'
                       else float(np.nanmax(curve)))
            if np.isfinite(ref_fit) and ref_fit > 0:
                curve /= ref_fit
        ax.plot(dg, curve, "r", lw=1.4)
    if not normalise_focus_peak:
        ax.axhline(model.foc_crit, color="grey", lw=0.8, ls="--")
    dof = depth_of_field(model, 1.0, L=None if not multi else float(sizes[-1]))
    for v in (dof["d_minus"], dof["d_plus"]):
        if np.isfinite(v):
            ax.axvline(v, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("signed distance d [mm]")
    ax.set_ylabel(("Focus / median near d=0" if focus_plot_reference == "zero_median" else "Focus / population peak") if normalise_focus_peak else
                  "F_raw / local illumination" if model.focus_model == "normalised_generalised"
                  else f"focus ÷ (I/I_ref)^{model.focus_gamma:.2f}")
    ax.set_title((("Focus profiles · median near d=0" if focus_plot_reference == "zero_median" else "Focus profiles · peak normalised") if normalise_focus_peak else
                  f"Focus · DOF {dof['total_mm']:.2f} mm @ {model.foc_crit:g}")
                 + (f" ({sizes[-1]:g} µm)" if multi else ""), fontsize=10)

    ax = axes[0, 2]
    if model.focus_model == "normalised_generalised":
        xs = np.linspace(max(1., min(sizes)), max(sizes), 65) if multi else sizes
        if abs(model.focus_Ar_width_power) > 1e-8 or abs(model.focus_Ar_power) > 1e-8:
            for ar_value, style in ((1.0, "-"), (0.7, "--"), (0.4, ":")):
                values = [depth_of_field(model, 1.0, L=float(size), ar=ar_value, n=1001) for size in xs]
                ax.plot(xs, [v["total_mm"] for v in values], ls=style, lw=1.7,
                        label=f"Ar={ar_value:g} mean")
        else:
            values = [depth_of_field(model, 1.0, L=float(size)) for size in xs]
            ax.plot(xs, [v["total_mm"] for v in values], label="mean-curve DOF")
            ax.plot(xs, [v["acceptance_mm"] for v in values], "--", label="acceptance-weighted")
        ax.set_xlabel("True diameter [µm]")
        ax.set_ylabel("Depth of field [mm]")
        ax.set_title(f"DOF vs size · F/I ≥ {model.foc_crit:g}", fontsize=10)
        ax.legend(fontsize=7)
    else:
        pass
    if model.focus_model != "normalised_generalised":
        for k, size in enumerate(sizes if multi else [None]):
            rows = dof_table(model, np.linspace(0.5, 1.5, 11), L=size)
            colour = colours[k % 6]
            tag = f"{size:g} µm " if size is not None else ""
            ax.plot([r["irel"] for r in rows], [r["total_mm"] for r in rows], "o-", color=colour, ms=3,
                    label=f"{tag}mean-curve crossing")
            ax.plot([r["irel"] for r in rows], [r["acceptance_mm"] for r in rows], "s--", color=colour, ms=3,
                    label=f"{tag}acceptance-weighted")
        ax.set_xlabel("local illumination I/I_ref")
        ax.set_ylabel("depth of field [mm]")
        ax.set_title("DOF vs illumination", fontsize=10)
        ax.legend(fontsize=6)

    def overall_histogram(ax):
        bins = np.linspace(-0.3, 0.8, 56)
        raw = t.length / float(getattr(model, "length_factor", 1.0) or 1.0)      # as processed
        ax.hist((raw - t.true_length) / t.true_length, bins, histtype="step", color="k", label="uncorrected")
        if validation is not None:
            pp = validation["per_particle"]
            vt = validation["table"]
            ax.hist((pp["L_known"] - vt.true_length) / vt.true_length, bins, histtype="step", color="tab:blue",
                    label="retrieved, side known")
            ax.hist((pp["L_both"] - vt.true_length) / vt.true_length, bins, histtype="step", color="tab:orange",
                    label="retrieved, side unknown")
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel("Relative size error")
        ax.set_title("Sizing error (held-out batches)" if validation is not None
                     else "Raw diameter error (fit sample; no hold-out)", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

    # Display diagnostics on the fitted observations when cross-validation is off.
    # A deterministic, stratified subset keeps the inverse posterior responsive
    # even for workbench projects with tens of thousands of frames.
    if validation is None:
        nmax = 2400
        if len(t) > nmax:
            # Include every size, both sides and the full range of positions.
            strata = np.round(t.true_length, 2).astype(str)
            strata = np.char.add(strata, np.where(t.d >= 0, "+", "-"))
            ids = []
            for value in np.unique(strata):
                ix = np.flatnonzero(strata == value)
                n_take = max(1, round(nmax * len(ix) / len(t)))
                ids.extend(ix[np.linspace(0, len(ix)-1, min(n_take, len(ix)), dtype=int)])
            ix = np.unique(np.asarray(ids, int))
            if len(ix) > nmax:
                ix = ix[np.linspace(0, len(ix)-1, nmax, dtype=int)]
            dt = t.subset(ix)
        else:
            dt = t
        # retrieve() accepts measured lengths in the original processing units.
        ar = dt.roundness if dt.roundness is not None else None
        both = retrieve(model, dt.length / float(getattr(model, "length_factor", 1.) or 1.),
                        dt.focus, dt.illum, side="both", ar=ar, shadow_ratio=dt.shadow_ratio, n_L=41, n_d=81)
        plus = retrieve(model, dt.length / float(getattr(model, "length_factor", 1.) or 1.),
                        dt.focus, dt.illum, side="+", ar=ar, shadow_ratio=dt.shadow_ratio, n_L=41, n_d=81)
        minus = retrieve(model, dt.length / float(getattr(model, "length_factor", 1.) or 1.),
                         dt.focus, dt.illum, side="-", ar=ar, shadow_ratio=dt.shadow_ratio, n_L=41, n_d=81)
        known_d = np.where(dt.d >= 0, plus["d_median"], minus["d_median"])
        diagnostic = {"table": dt, "per_particle": {
            "L_both": both["L_median"], "d_known": known_d,
            "absd_both": both["abs_d_median"]}}
        diag_note = f"In-sample diagnostic (n={len(dt):,} of {len(t):,}; not validation)"
    else:
        diagnostic = validation
        diag_note = "Held-out validation"

    ax = axes[1, 0]
    error_by_zd(ax, model, diagnostic)
    ax.set_title("Sizing error by position", fontsize=10, pad=10)

    ax = axes[1, 1]
    pp = diagnostic["per_particle"]
    vt = diagnostic["table"]
    good = np.isfinite(vt.d) & np.isfinite(pp["d_known"])
    ax.scatter(vt.d[good], pp["d_known"][good], s=4, alpha=.45,
               label="signed position, side known", rasterized=True)
    good_abs = np.isfinite(vt.d) & np.isfinite(pp["absd_both"])
    ax.scatter(np.abs(vt.d[good_abs]), pp["absd_both"][good_abs], s=4, alpha=.45,
               label="absolute position, side unknown", rasterized=True)
    max_abs = max(np.nanmax(np.abs(vt.d)), 0.1)
    ax.plot([-max_abs, max_abs], [-max_abs, max_abs], "k--", lw=.8, label="1:1")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlabel("Actual distance [mm]")
    ax.set_ylabel("Retrieved distance [mm]")
    ax.set_title("Position retrieval", fontsize=10, pad=10)

    ax = axes[1, 2]
    # Paired distributions: both histograms always describe exactly the same observations.
    vt = diagnostic["table"]
    pp = diagnostic["per_particle"]
    raw_error = (vt.length / float(getattr(model, "length_factor", 1.) or 1.)
                 - vt.true_length) / vt.true_length
    corrected_error = (pp["L_both"] - vt.true_length) / vt.true_length
    valid = np.isfinite(raw_error) & np.isfinite(corrected_error)
    if valid.any():
        rr, cc = raw_error[valid], corrected_error[valid]
        lo, hi = np.percentile(np.r_[rr, cc], [0.5, 99.5])
        lo, hi = min(float(lo), -0.1), max(float(hi), 0.1)
        bins = np.linspace(lo, hi, 55)
        ax.hist(rr, bins, histtype="step", lw=1.6, label=f"Uncorrected (n={len(rr):,})")
        ax.hist(cc, bins, histtype="step", lw=1.6, label=f"Corrected (n={len(cc):,})")
    else:
        ax.text(.5, .5, "No paired finite raw/corrected measurements",
                ha="center", transform=ax.transAxes)
    ax.set_title("Size error: before / after", fontsize=10, pad=10)
    ax.axvline(0, color="grey", lw=.8, ls="--")
    ax.set_xlabel("Relative diameter error")
    ax.set_ylabel("Particle count")
    ax.legend(fontsize=7, loc="upper right")
    # In-sample curves are diagnostics, not out-of-sample validation.
    eq_ax.text(.50, .02, diag_note, transform=eq_ax.transAxes, fontsize=8,
               ha="center", va="bottom", color="#555")
    return figure



def shadow_shape_diagnostics(table: CalibrationTable):
    """Observed CPI shape/shadow quantities; no microscopy or fitted correction.

    The geometrical reference uses CPI contour roundness, not an independently
    measured projected area. It is diagnostic only and must not be presented as
    extinction efficiency or geometric ground truth.
    """
    true = np.asarray(table.true_length, float)
    shadow = (np.asarray(table.length, float) if table.size_measure == "shadow_deq"
              else np.full_like(true, np.nan))
    if table.roundness is None:
        roundness = np.full_like(true, np.nan)
    else:
        roundness = np.asarray(table.roundness, float)
    labels = np.asarray(table.label, object)
    valid = (np.isfinite(true) & (true > 0) & np.isfinite(shadow) & (shadow > 0) &
             np.isfinite(roundness) & (roundness > 0) & (roundness <= 1.01))
    return {'roundness': roundness[valid], 'shadow_ratio': shadow[valid] / true[valid],
            'labels': labels[valid], 'n': int(valid.sum())}


def make_shadow_shape_figure(table: CalibrationTable, figure=None):
    """Validation diagnostic, not an image-by-image physics correction.

    Compares CPI-derived shape against squared signed-shadow diameter ratio.
    Microscopy is deliberately absent: a calibration needing reference images
    cannot be applied generally to atmospheric particles.
    """
    if figure is None:
        from matplotlib.figure import Figure
        figure = Figure(figsize=(11, 7), layout='constrained')
    figure.clear()
    ax, hist = figure.subplots(1, 2)
    info = shadow_shape_diagnostics(table)
    if info['n'] == 0:
        for a in (ax, hist):
            a.text(.5, .5, 'Select circle shadow sizing and use data with CPI roundness',
                   ha='center', va='center', transform=a.transAxes, wrap=True)
            a.set_axis_off()
        return figure
    x, y = info['roundness'], info['shadow_ratio'] ** 2
    # CPI roundness is an area / Feret-circle ratio, not necessarily the
    # minimum-enclosing-circle area ratio used in microscopy. Explicit label.
    groups = np.unique(info['labels'])
    for i, label in enumerate(groups[:16]):
        m = info['labels'] == label
        ax.scatter(x[m], y[m], s=4, alpha=.25, rasterized=True,
                   label=str(label)[:20])
    ax.plot([0,1], [0,1], 'k--', lw=1, label='equal area proxies')
    ax.set(xlim=(0, 1.05), xlabel='CPI contour area / Feret circle area',
           ylabel='(D shadow / labelled L)²',
           title='Optical shadow vs CPI morphology (not microscopy)')
    if len(groups) <= 12:
        ax.legend(fontsize=7, markerscale=3)
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = y / x
    efficiency = efficiency[np.isfinite(efficiency) & (efficiency > 0)]
    hist.hist(efficiency, bins=50, histtype='step', color='tab:blue')
    hist.axvline(1, ls='--', color='black', lw=1)
    hist.set(xlabel='(D shadow / labelled L)² / CPI contour area ratio',
             ylabel='Particle count', title='Apparent shadow/contour area ratio')
    figure.suptitle(f'{info["n"]:,} matched CPI images · area correction equals contour equivalent-area size', fontsize=12)
    return figure


def _darker(colour, f=0.6):
    from matplotlib.colors import to_rgb
    return tuple(f * c for c in to_rgb(colour))


def _halo():
    from matplotlib import patheffects
    return [patheffects.Stroke(linewidth=4.0, foreground="white"), patheffects.Normal()]


class _LazyHalo(list):
    def __iter__(self):
        return iter(_halo())


_HALO = _LazyHalo()


def model_equations(model: SizingModel):
    """Mathtext lines describing the fitted model, with its numbers: (size lines, focus lines)."""
    m = model.size_L_power
    Lr = f"{model.L_ref_um:.0f}"
    zd = r"Z_d=\lambda\,|d-d_0|\,/\,(L/2)^2" + (rf"\;\cdot\,(L/{Lr}\,\mu m)^{{{2 - m:.2f}}}"
                                                if abs(2.0 - m) >= 0.005 else "")
    if getattr(model, "shape_terms", False):
        zd += rf"\;/\;A_r^{{{model.size_Ar_power:.2f}}}"
    size = [
        r"$L_{app}=s\,L\,\left[1+A_\pm\,Z_d^{\;p_\pm}\,(I/I_{ref})^{\kappa}\right]+b$",
        "$" + zd + "$",
        rf"$s={model.scale:.3f},\ b={model.offset_um:.2f}\,\mu m,\ \kappa={model.size_kappa:.2f},\ "
        rf"d_0={model.size_d0_mm:+.3f}\,mm,\ \lambda={model.wavelength_um:g}\,\mu m$",
        rf"$A_+={model.size_plus['A']:.3f},\ p_+={model.size_plus['p']:.2f},\ "
        rf"A_-={model.size_minus['A']:.3f},\ p_-={model.size_minus['p']:.2f}$",
    ]
    if model.size_model == "shadow_linear":
        size = [r"$D_{shadow}=s\,L_{true}+b$",
                r"$A_{shadow}=\sum_{circle}(1-T_{ij})\,\Delta x^2$",
                rf"$s={model.scale:.3f},\ b={model.offset_um:.2f}\,\mu m$",
                r"$\mathrm{Stored\ Feret\ length\ and\ morphology\ unchanged}$"]
    Lterm = rf"\,(L/{Lr}\,\mu m)^{{{model.focus_L_amp:.2f}}}" if model.focus_L_amp else ""
    if abs(model.focus_Ar_power) > 1e-9:
        Lterm += rf"\,A_r^{{{model.focus_Ar_power:.2f}}}"
    wterm = rf"w_\pm\,(L/{Lr}\,\mu m)^{{{model.focus_L_width:.2f}}}" if model.focus_L_width else r"w_\pm"
    if abs(model.focus_Ar_width_power) > 1e-9:
        wterm += rf"\,A_r^{{{model.focus_Ar_width_power:.2f}}}"
    focus = [
        rf"$F=F_0\,(I/I_{{ref}})^{{\gamma}}{Lterm}\;\exp\left[-\left(|d-d_{{0f}}|\,/\,{wterm}\right)^{{q}}\right]+F_{{floor}}$",
        rf"$F_0={model.focus_F0:.1f},\ \gamma={model.focus_gamma:.2f},\ q={model.focus_q:.2f},\ "
        rf"F_{{floor}}={model.focus_floor:.2f},\ I_{{ref}}={model.illum_ref:.0f}$",
        rf"$w_+={model.focus_w_plus_mm:.3f}\,mm,\ w_-={model.focus_w_minus_mm:.3f}\,mm,\ "
        rf"d_{{0f}}={model.focus_d0_mm:+.3f}\,mm$",
    ]
    if model.focus_model == "normalised_generalised":
        focus = [
            r"$F_{raw}/I_{local}=B+A(L)\,\exp[-(|d-d_0|/w_\pm(L))^p]$",
            rf"$A(L,A_r)={model.focus_F0:.3f}(L/{Lr})^{{{model.focus_L_amp:.2f}}}A_r^{{{model.focus_Ar_power:.2f}}},\ "
            rf"w_\pm(L,A_r)=w_\pm(L/{Lr})^{{{model.focus_L_width:.2f}}}A_r^{{{model.focus_Ar_width_power:.2f}}}$",
            rf"$w_+={model.focus_w_plus_mm:.3f},\ w_-={model.focus_w_minus_mm:.3f}\,mm,\ "
            rf"p={model.focus_q:.2f},\ B={model.focus_floor:.3f}$",
        ]
    return size, focus


def draw_equations(ax, model: SizingModel):
    """Fitted equations in the strip above the panels.

    The text is kept out of the figure's layout (it must never squeeze the
    panels) and its size follows the figure width; size and focus are stacked
    when the figure is narrow, side by side when it is wide.
    """
    ax.axis("off")
    try:
        size, focus = model_equations(model)
    except Exception:            # never let a formatting problem stop the figure
        return
    width_in = ax.figure.get_size_inches()[0]
    wide = width_in >= 9.5
    big = max(7.5, min(10.5, 0.95 * width_in)) if not wide else max(9.0, min(12.0, 0.78 * width_in))
    small = big * 0.82
    blocks = (((0.0, "Size", size), (0.52, "Focus", focus)) if wide else
              ((0.0, "Size", size + [""] + ["@Focus"] + focus),))
    for x, title, lines in blocks:
        t = ax.text(x, 1.0, title, transform=ax.transAxes, fontsize=big * 0.92, fontweight="bold", va="top")
        t.set_in_layout(False)
        n_main = 2 if title == "Size" else 1
        y = 0.86 if wide else 0.90
        step_main, step_small = (0.22, 0.17) if wide else (0.125, 0.10)
        k_main = 0
        for line in lines:
            if line == "":
                y -= 0.03
                continue
            if line == "@Focus":
                t = ax.text(x, y, "Focus", transform=ax.transAxes, fontsize=big * 0.92, fontweight="bold", va="top")
                t.set_in_layout(False)
                y -= step_main
                n_main, k_main = 1, 0
                continue
            main = k_main < n_main
            k_main += 1
            t = ax.text(x + 0.06 if not wide else x + 0.01, y, line, transform=ax.transAxes,
                        fontsize=big if main else small, va="top", color="#111" if main else "#444")
            t.set_in_layout(False)
            y -= step_main if main else step_small


def error_by_zd(ax, model: SizingModel, validation: dict, nbins: int = 8):
    """Sizing error in bins of signed normalised distance Zd: uncorrected vs retrieved (side unknown).

    One row per Zd bin (equal numbers of particles), each with the two error
    histograms normalised to the same height, and lines through the bin medians,
    so a residual trend with defocus shows up at a glance.
    """
    vt, pp = validation["table"], validation["per_particle"]
    sign = np.sign(vt.d - model.size_d0_mm)
    zd = sign * model.zd(vt.true_length, vt.d)
    raw = (vt.length - vt.true_length) / vt.true_length
    cor = (pp["L_both"] - vt.true_length) / vt.true_length
    ok = np.isfinite(zd) & np.isfinite(raw) & np.isfinite(cor)
    zd, raw, cor = zd[ok], raw[ok], cor[ok]
    if zd.size < 20:
        ax.text(0.5, 0.5, "too few particles", ha="center", transform=ax.transAxes)
        return
    edges = np.unique(np.quantile(zd, np.linspace(0, 1, nbins + 1)))
    lo, hi = np.nanpercentile(np.concatenate([raw, cor]), [1, 97])
    bins = np.linspace(max(min(lo, -0.1), -0.5), min(max(hi, 0.1), 1.0), 71)
    centres = 0.5 * (bins[1:] + bins[:-1])
    med = {"raw": [], "cor": []}
    labels = []
    for i in range(len(edges) - 1):
        m = (zd >= edges[i]) & ((zd < edges[i + 1]) if i < len(edges) - 2 else (zd <= edges[i + 1]))
        for key, values, colour in (("raw", raw[m], "k"), ("cor", cor[m], "tab:orange")):
            h, _ = np.histogram(values, bins)
            if h.max() > 0:
                y = 0.9 * h / h.max()
                ax.fill_between(centres, i, i + y, step="mid", color=colour, alpha=0.18, lw=0)
                ax.step(centres, np.where(h > 0, i + y, np.nan), where="mid", color=colour, lw=0.9)
            md = float(np.median(values)) if values.size else float("nan")
            med[key].append(md)
            ax.plot([md, md], [i, i + 0.9], color=colour, lw=1.8)
        labels.append(f"{edges[i]:+.2f} to {edges[i + 1]:+.2f}")
    rows = np.arange(len(labels)) + 0.45
    ax.plot(med["raw"], rows, "k--", lw=1.2, marker="o", ms=3, label="uncorrected (median)")
    ax.plot(med["cor"], rows, color="tab:orange", lw=1.6, marker="o", ms=3, label="retrieved, side unknown (median)")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.1, len(labels))
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("Relative size error")
    ax.set_ylabel("Normalised position bin", fontsize=8)
    for i in range(1, len(labels)):
        ax.axhline(i, color="#ddd", lw=0.5, zorder=0)
    # Legend removed: it obscured the right-hand sizing-error distributions.


# --------------------------------------------------------------------------- end-to-end
def run_calibration(table: CalibrationTable, illum_min: Optional[float] = None, foc_crit: float = DEFAULT_FOC_CRIT,
                    do_validation: bool = True, **fit_kw):
    model = fit_model(table, illum_min=illum_min, foc_crit=foc_crit, **fit_kw)
    model.size_measure = table.size_measure
    model.focus_measure = table.focus_measure
    tracking = pixel_scale_from_tracking(table)
    validation = validate(table, illum_min=illum_min, foc_crit=foc_crit, **fit_kw) if do_validation else None
    model.fit_stats["pixel_tracking"] = {k: v for k, v in tracking.items() if k not in ("shifts", "used")}
    if validation is not None:
        model.fit_stats["validation"] = validation["summary"]
    model.fit_stats["dof"] = dof_table(model)
    return model, validation, tracking


def fit_main(argv=None):
    ap = argparse.ArgumentParser(description="Fit CPI size/focus calibration from a workbench particle table.")
    ap.add_argument("inputs", nargs="+", help="project .json files (recommended) or one particle-table CSV")
    ap.add_argument("-o", "--output", default="sizing_model.json")
    ap.add_argument("--figure", help="save the diagnostic figure (png/pdf)")
    ap.add_argument("--label", action="append", help="only use particles with this label (repeatable)")
    ap.add_argument("--illum-min", type=float, default=None, help="ignore particles with local illumination below this")
    ap.add_argument("--foc-crit", type=float, default=None,
                    help="focus threshold for DOF: default 5 for original, 0.12 for F/I_local")
    ap.add_argument("--offset-um", type=float, default=None, help="fix the additive edge offset (default 2 px)")
    ap.add_argument("--fit-offset", action="store_true", help="fit the offset (needs two or more true sizes)")
    ap.add_argument("--flip-side", action="store_true", help="multiply d by -1 before fitting")
    ap.add_argument("--no-validation", action="store_true")
    ap.add_argument("--size", choices=tuple(SIZE_MEASURES), default="length",
                    help="measured size to calibrate: length (max Feret) or deq (equivalent-area diameter)")
    ap.add_argument("--shape", action="store_true",
                    help="fit particle-shape terms (Park et al. 2026): Zd / Ar^alpha in the size model and Ar^c on "
                         "the focus peak, Ar = measured roundness (needs beads and non-spherical analogues)")
    ap.add_argument("--max-cutoff", type=float,
                    help="leave out particles with more than this %% of their outline on the ROI edge (cutoff_pct)")
    ap.add_argument("--fit-recipe", choices=("original", "circle", "contour"), default=None,
                    help="quick preset: original CPI, signed-shadow circle + normalised focus, or "
                         "contour equivalent diameter + normalised focus")
    ap.add_argument("--fixed-gaussian", action="store_true", help="fix focus p=2 instead of fitting p")
    ap.add_argument("--symmetric-focus", action="store_true", help="use the same width for +Z and -Z")
    ap.add_argument("--no-peak-size", action="store_true", help="fix focus peak independent of true diameter")
    ap.add_argument("--no-width-size", action="store_true", help="fix focus width independent of true diameter")
    ap.add_argument("--focus", choices=tuple(FOCUS_MEASURES), default="focus",
                    help="focus value to calibrate with: focus (as processed) or cpv_focus (CPIview's, --cpiview)")
    px = ap.add_mutually_exclusive_group()
    px.add_argument("--pixel-um", type=float, help="rescale measured sizes to this pixel size before fitting")
    px.add_argument("--fit-pixel", action="store_true",
                    help="fit the pixel size from the bead sizes (needs two or more), then rescale")
    args = ap.parse_args(argv)
    if args.fit_recipe == "circle":
        args.size, args.focus = "shadow_deq", "focus_per_illum"
    elif args.fit_recipe == "contour":
        args.size, args.focus = "deq", "focus_per_illum"
    elif args.fit_recipe == "original":
        args.size, args.focus = "length", "focus"
    if args.foc_crit is None:
        args.foc_crit = .12 if args.focus == "focus_per_illum" else DEFAULT_FOC_CRIT
    inputs = []
    if all(str(p).lower().endswith(".json") for p in args.inputs):
        from .combine import load_projects_for_fit
        rows, inputs, problems = load_projects_for_fit(args.inputs)
        for problem in problems:
            print("ERROR:", problem)
        if problems:
            raise SystemExit(2)
        for item in inputs:
            for w in item["warnings"]:
                print(f"WARNING: {item['project']} {w}")
        if args.max_cutoff is not None:
            rows = [r for r in rows if not _num(r.get("cutoff_pct")) > args.max_cutoff]
        table = CalibrationTable.from_rows(rows, args.size, args.focus)
    elif len(args.inputs) == 1:
        table = CalibrationTable.from_csv(args.inputs[0], args.size, args.focus)
        for key, value in table.instrument.items():
            if "," in value:
                print(f"WARNING: the table mixes instruments ({key} = {value}); fit one CPI at a time.")
    else:
        raise SystemExit("give project .json files, or a single particle-table CSV")
    if args.label:
        table = table.subset(np.isin(table.label, args.label))
    if args.flip_side:
        table.d = -table.d
    model, validation, tracking = run_calibration(table, illum_min=args.illum_min, foc_crit=args.foc_crit,
                                                  do_validation=not args.no_validation, offset_um=args.offset_um,
                                                  fit_offset=True if args.fit_offset else None, shape=args.shape,
                                                  pixel_mode=("fitted" if args.fit_pixel else "set" if args.pixel_um
                                                              else "processed"),
                                                  pixel_set_um=args.pixel_um,
                                                  size_model=("shadow_linear" if args.size == "shadow_deq" else "defocus"),
                                                  focus_model=("normalised_generalised" if args.focus == "focus_per_illum"
                                                               else "legacy"),
                                                  fit_focus_p=not args.fixed_gaussian,
                                                  asymmetric_focus=not args.symmetric_focus,
                                                  fit_peak_size=not args.no_peak_size,
                                                  fit_width_size=not args.no_width_size)
    model.inputs = inputs
    model.save(args.output)
    print(format_report(model, validation, tracking))
    print(f"\nwrote {args.output}")
    if args.figure:
        make_figure(model, table, validation, tracking).savefig(args.figure, dpi=130)
        print(f"wrote {args.figure}")


def _rows_from_processed(path: Path) -> List[dict]:
    from .data import ParticleCatalog, list_processed_files, scan_processed_file
    files = [path] if path.is_file() else list_processed_files(path)
    rows = []
    for f in files:
        for r in scan_processed_file(f, f.parent):
            rows.append({"particle_id": r.particle_id, "time_iso": r.time_iso, "time_matlab": r.time_matlab,
                         "length_um": r.length_um, "deq_um": r.deq_um, "shadow_deq_um": r.shadow_deq_um,
                         "focus": r.focus, "cpv_focus": r.cpv_focus,
                         "local_illum": r.local_illum,
                         "roundness": r.roundness, "image_x_px": r.image_x_px, "image_y_px": r.image_y_px})
    return rows


def retrieve_main(argv=None):
    ap = argparse.ArgumentParser(description="Retrieve true size and object-plane distance with a fitted model.")
    ap.add_argument("model", help="sizing_model.json from cpi-fit")
    ap.add_argument("input", help="CSV with length_um, focus, local_illum — or a processed .mat file / directory")
    ap.add_argument("-o", "--output", default="retrieved.csv")
    ap.add_argument("--side", choices=("both", "+", "-"), default="both",
                    help="both = side unknown (cloud data); +/- = side known (calibration test)")
    ap.add_argument("--foc-crit", type=float, default=None, help="only rows with focus above this (default: model's)")
    args = ap.parse_args(argv)
    model = SizingModel.load(args.model)
    path = Path(args.input)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = _rows_from_processed(path)
    crit = model.foc_crit if args.foc_crit is None else args.foc_crit
    columns = SIZE_MEASURES.get(getattr(model, "size_measure", "length"), SIZE_MEASURES["length"])["columns"]
    length = np.array([next((_num(r.get(c)) for c in columns if r.get(c) not in (None, "")), float("nan"))
                       for r in rows])
    if not np.isfinite(length).any():
        raise SystemExit(f"no {columns[0]} values in the input: this model was fitted on "
                         f"{SIZE_MEASURES[model.size_measure]['label'].lower()}")
    focus_col = FOCUS_MEASURES.get(getattr(model, "focus_measure", "focus"), FOCUS_MEASURES["focus"])["columns"][0]
    focus = np.array([_num(r.get(focus_col)) for r in rows])
    if model.focus_measure == "focus_per_illum":
        illum_for_focus = np.array([_num(r.get("local_illum")) for r in rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            focus = np.where(illum_for_focus > 0, focus / illum_for_focus, np.nan)
    if not np.isfinite(focus).any():
        raise SystemExit(f"no {focus_col} values in the input: this model was fitted on "
                         f"{FOCUS_MEASURES[model.focus_measure]['label'].lower()}")
    illum = np.array([_num(r.get("local_illum")) for r in rows])
    ar = np.array([_num(r.get("roundness")) for r in rows])
    shadow = np.array([_num(r.get("shadow_deq_um", r.get("shadow_deq"))) for r in rows])
    cpi_deq = np.array([_num(r.get("deq_um", r.get("deq"))) for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        shadow_ratio = np.where(cpi_deq > 0, shadow / cpi_deq, np.nan)
    res = retrieve(model, length, focus, illum, side=args.side, ar=ar, shadow_ratio=shadow_ratio)
    extra = ("L_median", "L_p16", "L_p84", "abs_d_median", "d_median", "p_plus", "extrapolated", "in_dof")
    in_dof = np.isfinite(focus) & (focus > crit)
    fields = list(rows[0].keys()) if rows else []
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields + [f"retr_{k}" for k in extra])
        w.writeheader()
        for i, r in enumerate(rows):
            out = dict(r)
            for k in extra:
                v = in_dof[i] if k == "in_dof" else res[k][i]
                out[f"retr_{k}"] = ("" if isinstance(v, float) and not math.isfinite(v) else
                                    (int(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 4)))
            w.writerow(out)
    print(f"retrieved {int(np.isfinite(res['L_median']).sum())} of {len(rows)} particles → {args.output}")


if __name__ == "__main__":
    fit_main()


def compare_processing(project_files, validate_runs: bool = True, size_measure: str = "length") -> str:
    """Fit each project separately and tabulate the results side by side.

    Meant for the same particles processed different ways (e.g. a project and
    its copy on reprocessed data): what changes is the edge method, so the
    differences in the table are due to the processing.
    """
    from .combine import load_projects_for_fit
    rows_out = []
    for project in project_files:
        rows, inputs, problems = load_projects_for_fit([project])
        if problems:
            raise ValueError("; ".join(problems))
        table = CalibrationTable.from_rows(rows, size_measure)
        model, validation, tracking = run_calibration(table, do_validation=validate_runs)
        use = _usable(table, None)
        t = table.subset(use)
        near = np.abs(t.d - model.size_d0_mm) <= max(0.03, 0.1 * np.ptp(t.d))
        per_size = {}
        for size in np.unique(np.round(t.true_length, 1)):
            k = near & (np.round(t.true_length, 1) == size)
            if k.any():
                per_size[float(size)] = (float(np.median(t.length[k])), float(np.std(t.length[k] / t.true_length[k])),
                                         int(k.sum()))
        rows_out.append({"name": Path(project).stem, "processing": ", ".join(model.fit_stats.get("processing", [])),
                         "model": model, "validation": validation, "per_size": per_size, "n": len(t)})
    sizes = sorted({s for r in rows_out for s in r["per_size"]})
    name_w = max(12, max(len(r["name"]) for r in rows_out))
    lines = ["Same particles, different processing   (size measure: "
             + SIZE_MEASURES[size_measure]["label"].lower() + ")", ""]
    for r in rows_out:
        lines.append(f"{r['name']:<{name_w}}  {r['processing']}   ({r['n']} particles)")
    lines += ["", "Length near best focus (median, and scatter of L_app/L):"]
    header = f"{'':<{name_w}}" + "".join(f"  {s:>8g} µm true      " for s in sizes)
    lines.append(header)
    for r in rows_out:
        cells = []
        for s in sizes:
            if s in r["per_size"]:
                med, sd, n = r["per_size"][s]
                cells.append(f"  {med:7.1f} µm ±{sd * 100:4.1f}% n={n:<4d}")
            else:
                cells.append(f"  {'–':>22}")
        lines.append(f"{r['name']:<{name_w}}" + "".join(cells))
    lines += ["", "Fitted calibration:"]
    lines.append(f"{'':<{name_w}}  edge offset b   scale s   L power m   oversize A+ / A-   focus F0   DOF@1 [mm]")
    for r in rows_out:
        m = r["model"]
        dof = depth_of_field(m, 1.0, L=max(r["per_size"]) if r["per_size"] else None)
        lines.append(f"{r['name']:<{name_w}}  {m.offset_um:9.2f} µm   {m.scale:7.4f}   "
                     f"{m.size_L_power:7.2f}{'*' if not m.size_L_power_fitted else ' '}   "
                     f"{m.size_plus['A']:6.3g} / {m.size_minus['A']:<6.3g}   {m.focus_F0:7.1f}   {dof['total_mm']:8.2f}")
    if any(not r["model"].size_L_power_fitted for r in rows_out):
        lines.append("  * assumed (one true size): label two or more sizes to fit it")
    if validate_runs:
        lines += ["", "Leave-one-batch-out sizing error (mean |ΔL|/L):"]
        lines.append(f"{'':<{name_w}}  uncorrected   retrieved, side known   retrieved, side unknown")
        for r in rows_out:
            v = (r["validation"] or {}).get("summary", {})
            get = lambda k: (v.get(k) or {}).get("mean_abs", float("nan")) * 100
            lines.append(f"{r['name']:<{name_w}}  {get('uncorrected'):9.1f}%   {get('known_side'):17.1f}%   "
                         f"{get('marginal_side'):19.1f}%")
    return "\n".join(lines)


def compare_main(argv=None):
    ap = argparse.ArgumentParser(description="Compare calibrations of the same particles processed different ways.")
    ap.add_argument("projects", nargs="+", help="project .json files (e.g. legacy and reprocessed copies)")
    ap.add_argument("--no-validation", action="store_true")
    ap.add_argument("-o", "--output", help="also write the comparison to this text file")
    ap.add_argument("--size", choices=tuple(SIZE_MEASURES), default="length",
                    help="measured size to compare: length (max Feret) or deq (equivalent-area diameter)")
    args = ap.parse_args(argv)
    text = compare_processing(args.projects, validate_runs=not args.no_validation, size_measure=args.size)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


def make_focus_by_population_figure(model: SizingModel, table: CalibrationTable, figure=None):
    """Separate stage-median measured/model focus comparisons using population median Ar."""
    from matplotlib.figure import Figure
    if figure is None:
        figure = Figure(figsize=(11, 8))
    figure.clear()
    figure.set_layout_engine(None)
    t = table.subset(_usable(table, None))
    sizes = np.unique(np.round(t.true_length, 1))
    ncols = min(3, max(1, len(sizes)))
    nrows = int(np.ceil(len(sizes) / ncols))
    axes = figure.subplots(nrows, ncols, squeeze=False)
    for j, size in enumerate(sizes):
        ax = axes.flat[j]
        mask = np.round(t.true_length, 1) == size
        x = np.asarray(t.d[mask], float)
        y = np.asarray(t.focus[mask], float)
        ar = np.asarray(t.roundness[mask], float)
        good_ar = ar[np.isfinite(ar) & (ar > 0)]
        ref_ar = float(np.median(good_ar)) if good_ar.size else 1.0
        stages = np.round(x / .025).astype(int)
        xx, yy = [], []
        for stage in np.unique(stages):
            vals = y[(stages == stage) & np.isfinite(y)]
            if vals.size >= 3:
                xx.append(stage * .025)
                yy.append(float(np.median(vals)))
        if len(xx):
            xx, yy = np.asarray(xx), np.asarray(yy)
            normalisation = np.median(yy[np.abs(xx) < .075]) if np.any(np.abs(xx) < .075) else np.max(yy)
            normalisation = max(float(normalisation), 1e-8)
            ax.plot(xx, yy / normalisation, 'o', ms=3, alpha=.7, label='Stage medians')
            dg = np.linspace(min(xx.min(), -0.05), max(xx.max(), 0.05), 350)
            pred = np.asarray(model.mean_focus(dg, 1.0, size, ar=ref_ar), float)
            pred_ref = float(model.mean_focus(0.0, 1.0, size, ar=ref_ar))
            ax.plot(dg, pred / max(pred_ref, 1e-8), '-', lw=1.6, label='Fitted focus')
        ax.set_title(f'{size:g} µm · n={mask.sum():,} · median Ar={ref_ar:.3f}', fontsize=10)
        ax.set_xlabel('Signed distance d [mm]')
        ax.set_ylabel('Focus / near-zero reference')
        ax.axvline(0, color='0.65', lw=.7, ls=':')
        ax.grid(alpha=.15)
        if j == 0:
            ax.legend(fontsize=8)
    for ax in axes.flat[len(sizes):]:
        ax.axis('off')
    figure.subplots_adjust(left=.075, right=.98, top=.93, bottom=.08, wspace=.34, hspace=.48)
    return figure
