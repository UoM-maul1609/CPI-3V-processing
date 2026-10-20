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
    instrument: Dict[str, str] = field(default_factory=dict)

    def __len__(self):
        return len(self.length)

    def subset(self, mask) -> "CalibrationTable":
        kw = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return CalibrationTable(**kw)

    @classmethod
    def from_rows(cls, rows: Sequence[dict]) -> "CalibrationTable":
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
        return cls(true_length=col("actual_length_um", "true_length_um"),
                   length=col("length_um", "len"), focus=col("focus"), illum=col("local_illum"),
                   d=col("d_signed_plane", "d_mm"), d_in_plane=col("d_in_plane"),
                   x_stage=col("x_stage", "stage_x"), z_stage=col("z_stage", "stage_z"),
                   image_x=col("image_x_px"), image_y=col("image_y_px"), time=col("time_matlab"),
                   group=cls._groups(rows), label=text("label"), instrument=instrument)

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
    def from_csv(cls, path: os.PathLike) -> "CalibrationTable":
        with open(path, newline="", encoding="utf-8") as handle:
            return cls.from_rows(list(csv.DictReader(handle)))


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
    size_sigma: List[float] = field(default_factory=lambda: [0.03, 0.05])     # log sd = s0 + s1*Zd
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
    focus_sigma: List[float] = field(default_factory=lambda: [0.2, 0.1])      # log sd = s0 + s1*|d|
    illum_ref: float = 1.0
    d_range_mm: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    size_range_um: List[float] = field(default_factory=lambda: [100.0, 100.0])
    pixel_um: float = NOMINAL_PIXEL_UM
    foc_crit: float = DEFAULT_FOC_CRIT
    wavelength_um: float = LAMBDA_UM
    sign_convention: str = "d = (X-Xbest)cos(theta) + (Z-Zbest)sin(theta) as in the workbench"
    instrument: Dict[str, str] = field(default_factory=dict)
    inputs: List[Dict[str, object]] = field(default_factory=list)   # projects the fit was made from
    fit_stats: Dict[str, object] = field(default_factory=dict)
    version: int = MODEL_VERSION
    created_utc: str = ""

    # ---- forward model
    def zd(self, L, d):
        L = np.maximum(np.asarray(L, float), 1e-3)
        fresnel = self.wavelength_um * np.abs(np.asarray(d, float) - self.size_d0_mm) * 1000.0 / (L / 2.0) ** 2
        if self.size_L_power == 2.0:
            return fresnel
        return fresnel * (L / self.L_ref_um) ** (2.0 - self.size_L_power)

    def mean_length(self, L, d, irel=1.0):
        L = np.asarray(L, float)
        d = np.asarray(d, float)
        plus = (d - self.size_d0_mm) >= 0
        A = np.where(plus, self.size_plus["A"], self.size_minus["A"])
        p = np.where(plus, self.size_plus["p"], self.size_minus["p"])
        irel = np.clip(np.asarray(irel, float), 0.05, 20)
        return self.scale * L * (1.0 + A * self.zd(L, d) ** p * irel ** self.size_kappa) + self.offset_um

    def sigma_length(self, L, d):
        return self.size_sigma[0] + self.size_sigma[1] * self.zd(L, d)

    def mean_focus(self, d, irel=1.0, L=None):
        d = np.asarray(d, float)
        u = d - self.focus_d0_mm
        w = np.where(u >= 0, self.focus_w_plus_mm, self.focus_w_minus_mm)
        amp = self.focus_F0
        if L is not None and (self.focus_L_amp or self.focus_L_width):
            ratio = np.maximum(np.asarray(L, float), 1e-3) / self.L_ref_um
            amp = amp * ratio ** self.focus_L_amp
            w = w * ratio ** self.focus_L_width
        irel = np.clip(np.asarray(irel, float), 0.05, 20)
        return amp * irel ** self.focus_gamma * np.exp(-(np.abs(u) / w) ** self.focus_q) + self.focus_floor

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


def fit_model(table: CalibrationTable, illum_min: Optional[float] = None, fit_offset: Optional[bool] = None,
              offset_um: Optional[float] = None, foc_crit: float = DEFAULT_FOC_CRIT,
              pixel_um: float = NOMINAL_PIXEL_UM) -> SizingModel:
    """Fit the length and focus models to labelled particles with known signed d."""
    use = _usable(table, illum_min)
    if use.sum() < 20:
        raise ValueError(f"only {int(use.sum())} usable particles (need labelled true size, length, focus and d)")
    t = table.subset(use)
    L, Lapp, foc, d = t.true_length, t.length, t.focus, t.d
    illum = np.where(np.isfinite(t.illum) & (t.illum > 0), t.illum, np.nan)
    illum_ref = float(np.nanmedian(illum)) if np.isfinite(illum).any() else 1.0
    irel = np.where(np.isfinite(illum), illum / illum_ref, 1.0)
    sizes = np.unique(np.round(L, 1))
    multi = len(sizes) >= 2
    if fit_offset is None:
        fit_offset = multi
    model = SizingModel(illum_ref=illum_ref, foc_crit=foc_crit, pixel_um=pixel_um, instrument=dict(t.instrument),
                        offset_um=(PIPELINE_PAD_PX * pixel_um if offset_um is None else float(offset_um)),
                        offset_fitted=bool(fit_offset),
                        d_range_mm=[float(np.min(d)), float(np.max(d))],
                        size_range_um=[float(np.min(L)), float(np.max(L))],
                        L_ref_um=float(np.exp(np.mean(np.log(sizes)))), size_L_power_fitted=multi)

    # ---- focus first: it pins the object plane and gives the scale of d
    near = np.abs(d) <= np.quantile(np.abs(d), 0.15)
    F0 = float(np.quantile(foc[near], 0.6)) if near.any() else float(np.quantile(foc, 0.9))
    floor0 = float(max(np.quantile(foc, 0.03) * 0.5, 0.1))
    w0 = float(max(np.std(d), 0.05))

    def unpack_focus(c, m):
        m.focus_F0, m.focus_gamma, m.focus_w_plus_mm, m.focus_w_minus_mm, m.focus_q, m.focus_d0_mm, m.focus_floor = (
            math.exp(c[0]), float(c[1]), math.exp(c[2]), math.exp(c[3]), float(c[4]), float(c[5]), math.exp(c[6]))
        if multi:
            m.focus_L_amp, m.focus_L_width = float(c[7]), float(c[8])

    def focus_res(c):
        m = SizingModel(**{**model.__dict__})
        unpack_focus(c, m)
        return np.log(foc) - np.log(m.mean_focus(d, irel, L))

    c0 = [math.log(max(F0 - floor0, 1e-2)), 1.0, math.log(w0), math.log(w0), 1.5, 0.0, math.log(floor0)]
    lo = [-5, -3, math.log(1e-3), math.log(1e-3), 0.5, -0.5, math.log(1e-3)]
    hi = [10, 4, math.log(20), math.log(20), 4.0, 0.5, math.log(max(F0, 1.0))]
    if multi:
        c0 += [0.0, 0.0]
        lo += [-3.0, -3.0]
        hi += [3.0, 3.0]
    fres = least_squares(focus_res, c0, bounds=(lo, hi), loss="soft_l1", f_scale=0.3)
    c = fres.x
    unpack_focus(c, model)
    model.focus_sigma = _scatter_fit(focus_res(c), np.abs(d - model.focus_d0_mm))

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

    def size_res(c):
        m = SizingModel(**{**model.__dict__})
        unpack_size(c, m)
        return np.log(Lapp) - np.log(np.maximum(m.mean_length(L, d, irel), 1e-3))

    s0 = [math.log(max(r0, 0.3)), 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    lo = [math.log(0.3), -12, 0.2, -12, 0.2, -3, -0.5]
    hi = [math.log(3.0), 8, 4.0, 8, 4.0, 3, 0.5]
    if fit_offset:
        s0.append(model.offset_um)
        lo.append(-20.0)
        hi.append(40.0)
    if multi:
        s0.append(2.0)
        lo.append(0.0)
        hi.append(4.0)
    sres = least_squares(size_res, s0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05)
    unpack_size(sres.x, model)
    model.size_sigma = _scatter_fit(size_res(sres.x), model.zd(L, d))

    model.fit_stats = {
        "n_particles": int(len(L)), "true_sizes_um": [float(s) for s in sizes],
        "illum_min": illum_min, "focus_cost": float(fres.cost), "size_cost": float(sres.cost),
        "ratio_at_focus_measured": float(np.median(Lapp[near] / L[near])) if near.any() else None,
        "length_at_focus_um": {f"{s:g}": float(model.mean_length(s, model.size_d0_mm)) for s in sizes},
        "effective_pixel_um_if_offset_correct": float(pixel_um * model.scale),
    }
    return model


# --------------------------------------------------------------------------- retrieval
def retrieve(model: SizingModel, length, focus, illum=None, side: str = "both", n_L: int = 141, n_d: int = 241,
             d_max: Optional[float] = None, chunk: int = 256) -> Dict[str, np.ndarray]:
    """Posterior size and distance for each particle.

    side: "both" (cloud data: marginalise over the side, equal prior weight),
          "+" or "-" (calibration test: side known).
    Priors: uniform in d over [-d_max, d_max] (particles fill the sample volume
    uniformly), uniform in log L.
    """
    length = np.atleast_1d(np.asarray(length, float))
    focus = np.atleast_1d(np.asarray(focus, float))
    n = len(length)
    irel = model.irel(np.full(n, np.nan) if illum is None else np.atleast_1d(np.asarray(illum, float)))
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
    idx = np.flatnonzero(ok)
    sf = model.sigma_focus(dg)[None, None, :]
    for start in range(0, len(idx), chunk):
        sl = slice(start, start + chunk)
        ii = idx[sl]
        top = (length[ii] - model.offset_um) / model.scale * 1.1
        Lg = np.exp(np.log(0.2 * top)[:, None] + unit[None, :] * np.log(5.0))              # (m, nL), log-uniform
        mu = model.mean_length(Lg[:, :, None], dg[None, None, :], irel[ii][:, None, None])  # (m, nL, nd)
        sl_ = model.sigma_length(Lg[:, :, None], dg[None, None, :])
        mu_f = model.mean_focus(dg[None, None, :], irel[ii][:, None, None], Lg[:, :, None])   # (m, nL, nd)
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
                   d_max: float = 5.0, n: int = 4001, L: Optional[float] = None) -> Dict[str, float]:
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
    mu = model.mean_focus(dg, irel, L)
    above = mu > crit
    out = {"foc_crit": float(crit), "irel": float(irel), "capped": bool(above[0] or above[-1]),
           "L_um": float(L) if L is not None else None}
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
                r = retrieve(m, test.length[sm], test.focus[sm], test.illum[sm], side=side_name)
                res["L_known"][idx[sm]] = r["L_median"]
                res["L_known_p16"][idx[sm]] = r["L_p16"]
                res["L_known_p84"][idx[sm]] = r["L_p84"]
                res["d_known"][idx[sm]] = r["d_median"]
        r = retrieve(m, test.length, test.focus, test.illum, side="both")
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
        "error_vs_foc_crit": thresholds, "per_size": per_size, "suggested_foc_crit_5pct": suggested,
        "side_confident_fraction": float(np.mean(np.abs(res["p_plus"][side_ok] - 0.5) > 0.4))
        if side_ok.any() else float("nan"),
    }
    return {"summary": summary, "table": t, "per_particle": res}


# --------------------------------------------------------------------------- report
def format_report(model: SizingModel, validation: Optional[dict] = None, tracking: Optional[dict] = None) -> str:
    fs = model.fit_stats
    lines = [f"Particles fitted: {fs.get('n_particles')}   true sizes: {fs.get('true_sizes_um')} µm"]
    if model.inputs:
        lines.append(f"From {len(model.inputs)} project(s):")
        for item in model.inputs:
            lines.append(f"  {item['project']}: {item.get('particles', '?')} particles, saved "
                         f"{item.get('project_saved_utc', '?')}")
            for w in item.get("warnings", ()):
                lines.append(f"    ⚠ {w}")
    if model.instrument:
        lines.append("Instrument: " + ", ".join(f"{k}={v}" for k, v in model.instrument.items()))
    lines += [
        "",
        "Length at best focus",
        f"  measured median L_app/L near focus: {fs.get('ratio_at_focus_measured', float('nan')):.3f}",
        f"  model: L_app = {model.scale:.4f}·L·(…) + {model.offset_um:.2f} µm"
        + ("  (offset fitted)" if model.offset_fitted else "  (offset fixed = processing mask pad)"),
        f"  length at focus: " + ", ".join(f"{k} µm → {v:.1f} µm" for k, v in fs.get("length_at_focus_um", {}).items()),
        f"  size minimum at d0 = {model.size_d0_mm * 1000:+.0f} µm (focus peak at {model.focus_d0_mm * 1000:+.0f} µm)",
        "",
        "Oversizing away from focus (Zd = λ|d|/(L/2)²" + (
            f"·(L/{model.L_ref_um:.0f} µm)^{2 - model.size_L_power:+.2f})" if model.size_L_power != 2.0 else ")"),
        (f"  size scaling: Zd ∝ |d|/L^{model.size_L_power:.2f} (fitted; 2 = Fresnel, Z/L²)"
         if model.size_L_power_fitted else "  size scaling: Z/L² assumed (fit two or more true sizes to test it)"),
        f"  + side: A = {model.size_plus['A']:.3g}, p = {model.size_plus['p']:.2f}",
        f"  − side: A = {model.size_minus['A']:.3g}, p = {model.size_minus['p']:.2f}",
        f"  illumination exponent κ = {model.size_kappa:.2f};  scatter σ(log L_app) = "
        f"{model.size_sigma[0]:.3f} + {model.size_sigma[1]:.3f}·Zd",
        "",
        "Focus",
        f"  F0 = {model.focus_F0:.1f}·(I/I_ref)^{model.focus_gamma:.2f}, I_ref = {model.illum_ref:.1f}",
        f"  half-widths: + {model.focus_w_plus_mm:.3f} mm, − {model.focus_w_minus_mm:.3f} mm, shape q = "
        f"{model.focus_q:.2f}, floor {model.focus_floor:.2f}",
        f"  scatter σ(log foc) = {model.focus_sigma[0]:.3f} + {model.focus_sigma[1]:.3f}·|d|",
        (f"  size dependence: peak × (L/{model.L_ref_um:.0f} µm)^{model.focus_L_amp:+.2f}, "
         f"half-widths × (L/{model.L_ref_um:.0f} µm)^{model.focus_L_width:+.2f}"
         if model.size_L_power_fitted else "  size dependence: none assumed (single true size)"),
        "",
        f"Depth of field at foc_crit = {model.foc_crit:g}",
    ]
    sizes = fs.get("true_sizes_um") or [None]
    for size in sizes:
        if len(sizes) > 1:
            lines.append(f"  {size:g} µm particles:")
        for row in dof_table(model, L=size):
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


def make_figure(model: SizingModel, table: CalibrationTable, validation: Optional[dict] = None,
                tracking: Optional[dict] = None, figure=None):
    """Six-panel diagnostic figure.  Pass an existing matplotlib Figure to draw into it."""
    if figure is None:
        from matplotlib.figure import Figure
        figure = Figure(figsize=(13, 8), layout="constrained")
    figure.clear()
    axes = figure.subplots(2, 3)
    use = _usable(table, None)
    t = table.subset(use)
    irel = model.irel(t.illum)
    dg = np.linspace(min(t.d.min(), -0.05), max(t.d.max(), 0.05), 300)
    Lref = float(np.median(t.true_length))

    sizes = np.unique(np.round(t.true_length, 1))
    multi = len(sizes) > 1
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]

    ax = axes[0, 0]
    if multi:
        # All sizes on one axis: signed Zd (∝ Z/L^m).  If the scaling is right, they collapse.
        sign = np.sign(t.d - model.size_d0_mm)
        for k, size in enumerate(sizes):
            m = np.round(t.true_length, 1) == size
            ax.scatter(sign[m] * model.zd(t.true_length[m], t.d[m]), t.length[m] / t.true_length[m], s=4,
                       color=colours[k % 6], alpha=0.5, rasterized=True, label=f"{size:g} µm")
            ds = np.linspace(min(t.d[m].min(), -0.05), max(t.d[m].max(), 0.05), 200)
            ax.plot(np.sign(ds - model.size_d0_mm) * model.zd(size, ds), model.mean_length(size, ds) / size,
                    color=colours[k % 6], lw=1.6)
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

    ax = axes[0, 1]
    if multi:
        for k, size in enumerate(sizes):
            m = np.round(t.true_length, 1) == size
            ax.scatter(t.d[m], t.focus[m], s=4, color=colours[k % 6], alpha=0.5, rasterized=True)
            ax.plot(dg, model.mean_focus(dg, 1.0, size), color=colours[k % 6], lw=1.6, label=f"{size:g} µm")
        ax.legend(fontsize=7)
    else:
        ax.scatter(t.d, t.focus, s=4, c=irel, cmap="viridis", vmin=0.6, vmax=1.4, alpha=0.6, rasterized=True)
        for r, ls in ((0.8, ":"), (1.0, "-"), (1.2, "--")):
            ax.plot(dg, model.mean_focus(dg, r), "r", ls=ls, lw=1.4)
    ax.axhline(model.foc_crit, color="grey", lw=0.8, ls="--")
    dof = depth_of_field(model, 1.0, L=None if not multi else float(sizes[-1]))
    for v in (dof["d_minus"], dof["d_plus"]):
        if np.isfinite(v):
            ax.axvline(v, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("signed distance d [mm]")
    ax.set_ylabel("focus")
    ax.set_title(f"Focus · DOF {dof['total_mm']:.2f} mm @ {model.foc_crit:g}"
                 + (f" ({sizes[-1]:g} µm)" if multi else ""), fontsize=10)

    ax = axes[0, 2]
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

    ax = axes[1, 0]
    bins = np.linspace(-0.3, 0.8, 56)
    ax.hist((t.length - t.true_length) / t.true_length, bins, histtype="step", color="k", label="uncorrected")
    if validation is not None:
        pp = validation["per_particle"]
        vt = validation["table"]
        ax.hist((pp["L_known"] - vt.true_length) / vt.true_length, bins, histtype="step", color="tab:blue",
                label="retrieved, side known")
        ax.hist((pp["L_both"] - vt.true_length) / vt.true_length, bins, histtype="step", color="tab:orange",
                label="retrieved, side unknown")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("(L_est − L) / L")
    ax.set_title("Sizing error (held-out batches)", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    if validation is not None:
        pp = validation["per_particle"]
        vt = validation["table"]
        ax.scatter(vt.d, pp["d_known"], s=4, alpha=0.5, label="d, side known", rasterized=True)
        ax.scatter(np.abs(vt.d), pp["absd_both"], s=4, alpha=0.5, label="|d|, side unknown", rasterized=True)
        lim = [min(vt.d.min(), 0), max(np.abs(vt.d).max(), 0.1)]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.legend(fontsize=7)
    ax.set_xlabel("measured d [mm]")
    ax.set_ylabel("retrieved [mm]")
    ax.set_title("Distance retrieval", fontsize=10)

    ax = axes[1, 2]
    if tracking and tracking.get("ok"):
        S = np.array(tracking["shifts"])
        used = np.array(tracking["used"], bool)
        u = math.radians(tracking["direction_deg"])
        mag = S[:, 1] * math.cos(u) + S[:, 2] * math.sin(u)      # shift along the fitted direction
        ax.scatter(S[used, 0], mag[used], s=14, label="stage steps")
        ax.scatter(S[~used, 0], mag[~used], s=14, marker="x", c="grey", label="rejected")
        xx = np.linspace(min(S[:, 0].min(), 0.0), max(S[:, 0].max(), 0.0), 10)
        ax.plot(xx, xx / tracking["pixel_um"], "r", label=f"{tracking['pixel_um']:.3f} µm/px")
        ax.plot(xx, xx / NOMINAL_PIXEL_UM, "k:", label=f"nominal {NOMINAL_PIXEL_UM} µm/px")
        ax.set_xlabel("in-plane stage move [µm]")
        ax.set_ylabel("image shift [px]")
        ax.legend(fontsize=7)
        ax.set_title("Pixel size (bead tracking)", fontsize=10)
    else:
        ax.axis("off")
        reason = (tracking or {}).get("reason", "not computed")
        ax.text(0.02, 0.9, "Pixel size from bead tracking:\n" + reason, transform=ax.transAxes, va="top", fontsize=9,
                wrap=True)
    return figure


# --------------------------------------------------------------------------- end-to-end
def run_calibration(table: CalibrationTable, illum_min: Optional[float] = None, foc_crit: float = DEFAULT_FOC_CRIT,
                    do_validation: bool = True, **fit_kw):
    model = fit_model(table, illum_min=illum_min, foc_crit=foc_crit, **fit_kw)
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
    ap.add_argument("--foc-crit", type=float, default=DEFAULT_FOC_CRIT)
    ap.add_argument("--offset-um", type=float, default=None, help="fix the additive edge offset (default 2 px)")
    ap.add_argument("--fit-offset", action="store_true", help="fit the offset (needs two or more true sizes)")
    ap.add_argument("--flip-side", action="store_true", help="multiply d by -1 before fitting")
    ap.add_argument("--no-validation", action="store_true")
    args = ap.parse_args(argv)
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
        table = CalibrationTable.from_rows(rows)
    elif len(args.inputs) == 1:
        table = CalibrationTable.from_csv(args.inputs[0])
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
                                                  fit_offset=True if args.fit_offset else None)
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
                         "length_um": r.length_um, "focus": r.focus, "local_illum": r.local_illum,
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
    length = np.array([_num(r.get("length_um")) for r in rows])
    focus = np.array([_num(r.get("focus")) for r in rows])
    illum = np.array([_num(r.get("local_illum")) for r in rows])
    res = retrieve(model, length, focus, illum, side=args.side)
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
