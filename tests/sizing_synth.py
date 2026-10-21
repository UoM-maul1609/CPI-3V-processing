"""Synthetic slide-scan of 100 µm beads drawn from a known SizingModel (shared by tests)."""
import math
import numpy as np
from calibration.sizing import SizingModel, CalibrationTable

TRUE_PIXEL_UM = 2.25


def truth_model():
    return SizingModel(scale=1.0, offset_um=4.6, size_plus={"A": 0.9, "p": 1.2}, size_minus={"A": 2.0, "p": 1.1},
                       size_kappa=0.5, size_d0_mm=0.0, size_sigma=[0.015, 0.06], focus_F0=60.0, focus_gamma=1.0,
                       focus_w_plus_mm=0.55, focus_w_minus_mm=0.35, focus_q=1.6, focus_d0_mm=0.0,
                       focus_floor=3.0, focus_sigma=[0.12, 0.10], illum_ref=120.0)


def synth_rows(seed=1, n_batches=4, beads=30, steps=np.linspace(-1.0, 1.0, 21), size=100.0, flip=False):
    rng = np.random.default_rng(seed)
    m = truth_model()
    rows = []
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    for b in range(n_batches):
        x0 = rng.uniform(100, 600, beads)
        y0 = rng.uniform(100, 900, beads)
        illum = np.clip(rng.normal(120, 20, beads), 70, 180)
        t_start = 738000.0 + b * 0.05
        for k, dx in enumerate(steps):      # move X only: d and in-plane both change
            d = dx * c
            inplane = -dx * s
            for i in range(beads):
                if rng.random() < 0.15:      # beads missed in some frames
                    continue
                if m.zd(size, d) > 0.6:      # too defocused to be detected (small beads leave the DOF sooner)
                    continue
                irel = illum[i] / m.illum_ref
                L_app = m.mean_length(size, d, irel) * math.exp(rng.normal(0, m.sigma_length(size, d)))
                foc = m.mean_focus(d, irel) * math.exp(rng.normal(0, m.sigma_focus(d)))
                rows.append(dict(actual_length_um=size, length_um=L_app, focus=foc, local_illum=illum[i],
                                 d_signed_plane=(-d if flip else d), d_in_plane=inplane, x_stage=10 + dx, z_stage=5.0,
                                 image_x_px=x0[i] + inplane * 1000 / TRUE_PIXEL_UM + rng.normal(0, 0.5),
                                 image_y_px=y0[i] + rng.normal(0, 0.5),
                                 time_matlab=t_start + k * 30 / 86400 + i * 1e-6, batch=f"batch {b + 1}",
                                 label="100 micron glass", cpi_type="CPI 1.0", cpi_id="test"))
    return rows


def synth_table(**kw):
    return CalibrationTable.from_rows(synth_rows(**kw))
