"""Processing pixel size: recorded in the files, never applied twice downstream."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

PROC = Path(__file__).resolve().parents[1] / "python" / "proc"
sys.path.insert(0, str(PROC))


def disc_image(d=40, size=96):
    yy, xx = np.mgrid[:size, :size]
    inside = np.hypot(yy - size / 2, xx - size / 2) < d / 2
    bg = np.full((size, size), 180.0)
    return np.where(inside, 20.0, bg), bg


def test_lengths_scale_exactly_with_pixel_size():
    from edgeDetect import EdgeOptions, measure_transmittance
    im, bg = disc_image()
    a = measure_transmittance(im, bg, EdgeOptions(method="transmittance", pixel_um=2.3))
    b = measure_transmittance(im, bg, EdgeOptions(method="transmittance", pixel_um=2.46))
    assert b["len"] / a["len"] == pytest.approx(2.46 / 2.3, rel=1e-9)
    assert b["deq"] / a["deq"] == pytest.approx(2.46 / 2.3, rel=1e-9)


def test_scan_reads_pixel_size_from_proc_options():
    from calibration.data import processing_pixel_um
    assert processing_pixel_um(json.dumps({"method": "legacy", "pixel_um": 2.46})) == 2.46
    assert processing_pixel_um(None) == 2.3 and processing_pixel_um("not json") == 2.3


def _table(px):
    from calibration.sizing import CalibrationTable
    n = len(px)
    f = lambda v: np.full(n, v, float)
    return CalibrationTable(true_length=f(100), length=f(105), focus=f(20), illum=f(100), d=np.linspace(-1, 1, n),
                            d_in_plane=f(0), x_stage=f(0), z_stage=f(0), image_x=f(0), image_y=f(0), time=f(0),
                            group=np.array(["a"] * n, object), label=np.array(["b"] * n, object),
                            pixel_um=np.asarray(px, float))


def test_fit_uses_the_processed_pixel_size():
    from calibration.sizing import processed_pixel_um
    assert processed_pixel_um(_table([np.nan] * 5)) == 2.3              # older files: nominal
    assert processed_pixel_um(_table([2.46] * 5)) == 2.46
    with pytest.raises(ValueError):
        processed_pixel_um(_table([2.3, 2.46]))


def test_set_pixel_size_is_not_applied_twice():
    import sizing_synth as ss
    from calibration.sizing import CalibrationTable, fit_model
    rows = ss.synth_rows(n_batches=3, beads=20, seed=1)
    for r in rows:                         # as if processed at 2.46 µm (lengths already in those units)
        r["pixel_um"] = 2.46
    t = CalibrationTable.from_rows(rows)
    m = fit_model(t, pixel_mode="set", pixel_set_um=2.46)
    assert m.pixel_um == 2.46 and m.length_factor == pytest.approx(1.0)
    m2 = fit_model(t, pixel_mode="set", pixel_set_um=2.5)
    assert m2.length_factor == pytest.approx(2.5 / 2.46)


def test_time_series_sample_area_inputs(tmp_path):
    from calcTimeseriesDriver import _frame_size_px, _processing_pixel_um
    full = {"Time": np.array([[731296.5]]), "IMAGE": np.array([{"BG": np.zeros((1024, 1022), np.uint8)}], dtype=object)}
    sio.savemat(tmp_path / "full_backgrounds.mat", {"FULL_BG": full})
    assert _frame_size_px(str(tmp_path), True) == (1024, 1022)
    assert _frame_size_px(str(tmp_path / "nowhere"), False) == (1280, 1024)
    sio.savemat(tmp_path / "f.mat", {"dat": {"proc_options": json.dumps({"pixel_um": 2.46})}})
    dat = sio.loadmat(tmp_path / "f.mat", variable_names=["dat"])["dat"]
    assert _processing_pixel_um(dat) == 2.46


def test_shape_terms_recover_park_scaling():
    """Synthetic particles whose oversizing follows Zd / Ar (Park et al. 2026): the fit finds alpha ~ 1."""
    from calibration.sizing import CalibrationTable, SizingModel, fit_model
    rng = np.random.default_rng(4)
    truth = SizingModel(size_Ar_power=1.0, focus_Ar_power=-0.3, size_plus={"A": 1.5, "p": 1.0},
                        size_minus={"A": 2.5, "p": 1.0}, L_ref_um=60.0, offset_um=0.0)
    rows = []
    for b in range(12):
        L = float(rng.choice([20.0, 50.0, 100.0]))
        ar = float(rng.uniform(0.3, 1.0))
        for d in rng.uniform(-0.8, 0.8, 40):
            lapp = truth.mean_length(L, d, 1.0, ar) * math.exp(rng.normal(0, 0.02))
            foc = truth.mean_focus(d, 1.0, None, ar) * math.exp(rng.normal(0, 0.1))
            rows.append({"actual_length_um": L, "length_um": lapp, "focus": foc, "d_signed_plane": d,
                         "local_illum": 100.0, "roundness": ar, "batch": f"b{b}", "label": f"L{L:g}"})
    t = CalibrationTable.from_rows(rows)
    off = fit_model(t)
    on = fit_model(t, shape=True)
    assert on.shape_terms and on.size_Ar_power == pytest.approx(1.0, abs=0.25)
    assert on.focus_Ar_power == pytest.approx(-0.3, abs=0.2)
    assert on.fit_stats["size_cost"] < 0.7 * off.fit_stats["size_cost"]
    beads = CalibrationTable.from_rows([dict(r, roundness=1.0) for r in rows])
    assert not fit_model(beads, shape=True).shape_terms          # no shape variation: skipped
