"""CPIview focus and edge-quality measures (proc/cpiviewMetrics.py, batchProcess --cpiview)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "proc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpiviewMetrics import compute_fwhm, cpiview_metrics  # noqa: E402
from test_edge_detect import disc, shape_image  # noqa: E402


def test_fwhm_of_a_gaussian_peak():
    x = np.arange(61, dtype=float)
    sigma = 3.0
    y = 100 * np.exp(-0.5 * ((x - 30) / sigma) ** 2) + 5
    fwhm, peak = compute_fwhm(y, 28)
    assert peak == pytest.approx(105, abs=0.1)
    assert abs(fwhm - 2.355 * sigma) <= 1.5          # integer-sample crossings, as CPIview


def test_sharp_edges_have_higher_focus_and_narrower_edges():
    sharp_im, sharp_bg = shape_image(disc(40), blur=0.7, seed=1)
    soft_im, soft_bg = shape_image(disc(40), blur=3.0, seed=1)
    sharp = cpiview_metrics(sharp_im, sharp_bg)
    soft = cpiview_metrics(soft_im, soft_bg)
    assert sharp["cpv_focus"] > 1.5 * soft["cpv_focus"]
    assert sharp["cpv_fwhm_px"] < soft["cpv_fwhm_px"]
    assert sharp["cpv_samples"] == 16 and sharp["cpv_focus_sd"] >= 0
    assert sharp["cpv_distinct"] > 0.9 and sharp["cpv_cutoff_pct"] == 0


def test_cut_off_by_the_roi_edge_is_reported():
    im, bg = shape_image(disc(40, c=12))             # disc runs off the ROI
    m = cpiview_metrics(im, bg)
    assert m["cpv_cutoff_pct"] > 5


def test_nothing_found_gives_nan():
    bg = np.full((40, 40), 150.0)
    m = cpiview_metrics(bg + np.random.default_rng(0).normal(0, 1, bg.shape), bg)
    assert all(np.isnan(v) for v in m.values())


def test_processing_option_stores_cpiview_fields(tmp_path):
    from imageStats import imageStats
    n = 2
    imgs = np.empty((1, n), dtype=[("IM", object)])
    bgs = np.empty((1, n), dtype=[("BG", object)])
    for i, dd in enumerate((25.0, 40.0)):
        im, bg = shape_image(disc(dd), seed=i)
        imgs[0, i]["IM"] = im.astype(np.uint8)
        bgs[0, i]["BG"] = bg.astype(np.uint8)
    roi = {"IMAGE": imgs, "imageType": np.full((n, 1), 89), "Time": np.arange(n, dtype=float)[:, None],
           "StartX": np.full((1, n), 100.0), "StartY": np.full((1, n), 100.0)}
    sio.savemat(tmp_path / "x.mat", {"ROI_N": roi, "BG": bgs})
    m = sio.loadmat(tmp_path / "x.mat")
    base = {"method": "transmittance", "edge_mode": "relative", "pixel_um": 2.3}
    off = imageStats(m["ROI_N"], m["BG"], False, True, 0, "test", None, options=json.dumps(base))
    assert "cpv_focus" not in off                                   # default: unchanged
    on = imageStats(m["ROI_N"], m["BG"], False, True, 0, "test", None, options=json.dumps(dict(base, cpiview=True)))
    assert np.isfinite(np.ravel(on["cpv_focus"])).all() and np.isfinite(np.ravel(on["cpv_fwhm_px"])).all()
    assert np.ravel(on["len"]) == pytest.approx(np.ravel(off["len"]))   # sizes are this processing's, untouched


def test_fit_can_use_cpiview_focus():
    import sizing_synth as ss
    from calibration.sizing import CalibrationTable, fit_model
    rows = ss.synth_rows(n_batches=3, beads=20, seed=2)
    for r in rows:
        r["cpv_focus"] = float(r["focus"]) * 1.7          # a different focus scale
    t = CalibrationTable.from_rows(rows, focus_measure="cpv_focus")
    assert t.focus_measure == "cpv_focus" and np.allclose(t.focus, [float(r["focus"]) * 1.7 for r in rows])
    m = fit_model(t)
    assert m.focus_d0_mm == pytest.approx(fit_model(CalibrationTable.from_rows(rows)).focus_d0_mm, abs=0.05)
