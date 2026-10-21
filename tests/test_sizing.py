import csv
import numpy as np
import pytest

from calibration.sizing import (SizingModel, CalibrationTable, fit_model, retrieve, depth_of_field,
                                pixel_scale_from_tracking, run_calibration, fit_main, retrieve_main)
from sizing_synth import synth_rows, synth_table, truth_model, TRUE_PIXEL_UM


@pytest.fixture(scope="module")
def table():
    return synth_table(n_batches=3, beads=25)


def test_fit_recovers_truth(table):
    m = fit_model(table)
    t = truth_model()
    assert m.scale == pytest.approx(1.0, abs=0.02)
    assert m.size_minus["A"] > m.size_plus["A"]              # asymmetry found
    assert m.size_plus["A"] == pytest.approx(t.size_plus["A"], rel=0.25)
    assert m.size_minus["A"] == pytest.approx(t.size_minus["A"], rel=0.25)
    assert m.focus_w_plus_mm == pytest.approx(0.55, rel=0.15)
    assert m.focus_w_minus_mm == pytest.approx(0.35, rel=0.15)
    assert m.focus_gamma == pytest.approx(1.0, abs=0.2)
    assert abs(m.size_d0_mm) < 0.05 and abs(m.focus_d0_mm) < 0.05


def test_retrieval_inverts_forward_model():
    m = truth_model()
    for d in (-0.3, 0.0, 0.4):
        Lapp, foc = m.mean_length(100.0, d), m.mean_focus(d)
        known = retrieve(m, [Lapp], [foc], [m.illum_ref], side="+" if d >= 0 else "-")
        assert known["L_median"][0] == pytest.approx(100, rel=0.04)
        assert known["d_median"][0] == pytest.approx(d, abs=0.08 if d else 0.15)   # truncated at 0
        both = retrieve(m, [Lapp], [foc], [m.illum_ref], side="both")
        assert both["L_p16"][0] <= 100 * 1.04 and both["L_p84"][0] >= 100 * 0.96


def test_model_json_round_trip(tmp_path):
    m = truth_model()
    m.save(tmp_path / "m.json")
    m2 = SizingModel.load(tmp_path / "m.json")
    assert m2.mean_length(80, -0.2, 1.1) == pytest.approx(m.mean_length(80, -0.2, 1.1))
    assert m2.mean_focus(0.3, 0.9) == pytest.approx(m.mean_focus(0.3, 0.9))


def test_depth_of_field_matches_threshold_crossing():
    m = truth_model()
    d = depth_of_field(m, 1.0, 20.0)
    assert m.mean_focus(d["d_plus"]) == pytest.approx(20.0, rel=0.02)
    assert m.mean_focus(d["d_minus"]) == pytest.approx(20.0, rel=0.02)
    assert d["d_plus"] > -d["d_minus"]                        # wider on the + side in the truth
    assert depth_of_field(m, 0.6, 20.0)["total_mm"] < d["total_mm"]   # darker → narrower


def test_pixel_size_from_tracking(table):
    tr = pixel_scale_from_tracking(table)
    assert tr["ok"]
    assert tr["pixel_um"] == pytest.approx(TRUE_PIXEL_UM, rel=0.01)


def test_tracking_needs_in_plane_motion():
    rows = synth_rows(n_batches=2, beads=10)
    for r in rows:
        r["d_in_plane"] = 0.0
    assert not pixel_scale_from_tracking(CalibrationTable.from_rows(rows))["ok"]


def test_validation_improves_sizing(table):
    m, v, tr = run_calibration(table)
    s = v["summary"]
    assert len(s["folds"]) == 3                                # leave-one-batch-out
    assert s["known_side"]["mean_abs"] < 0.5 * s["uncorrected"]["mean_abs"]
    assert s["marginal_side"]["mean_abs"] < 0.6 * s["uncorrected"]["mean_abs"]
    assert s["d_known"]["r"] > 0.95
    assert 0.5 < s["coverage_68_marginal"] < 0.85
    assert s["suggested_foc_crit_5pct"] is not None


def test_cli_fit_and_retrieve(tmp_path):
    rows = synth_rows(n_batches=3, beads=15)
    src = tmp_path / "p_particle_positions.csv"
    with src.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    fit_main([str(src), "-o", str(tmp_path / "m.json"), "--figure", str(tmp_path / "f.png"), "--no-validation"])
    assert (tmp_path / "m.json").exists() and (tmp_path / "f.png").stat().st_size > 10000
    retrieve_main([str(tmp_path / "m.json"), str(src), "-o", str(tmp_path / "r.csv"), "--side", "both"])
    with (tmp_path / "r.csv").open() as h:
        out = list(csv.DictReader(h))
    L = np.array([float(r["retr_L_median"]) for r in out if r["retr_L_median"]])
    assert len(L) > 0.9 * len(rows)
    assert abs(np.median(L) - 100) < 5


def test_sizing_dialog_runs(qtbot_free_app=None):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from calibration.gui import SizingDialog
    dlg = SizingDialog(synth_rows(n_batches=3, beads=15))
    assert dlg.label_box.count() == 1
    dlg.run()
    assert dlg.model is not None and dlg.save_button.isEnabled()
    assert "Leave-one-batch-out" in dlg.report.toPlainText()
    dlg.close()


def test_combined_projects_keep_batches_apart(tmp_path):
    a = synth_rows(n_batches=2, beads=10, seed=3)
    b = synth_rows(n_batches=2, beads=10, seed=4)
    for r in a:
        r["project"] = "beads_100"
    for r in b:
        r["project"] = "beads_40"
    t = CalibrationTable.from_rows(a + b)
    assert len(set(t.group)) == 4            # "Accepted batch 1" in each project stays separate
    assert "beads_100 · batch 1" in set(t.group)


def test_multi_size_fit_tests_the_L_scaling():
    import sizing_synth as ss
    from calibration.sizing import fit_model, format_report, run_calibration
    rows = []
    for size, seed in ((40.0, 2), (100.0, 3)):
        rows += [dict(r, batch=f"{size:g} {r['batch']}") for r in ss.synth_rows(n_batches=2, beads=15, seed=seed, size=size)]
    t = CalibrationTable.from_rows(rows)
    m = fit_model(t)
    assert m.size_L_power_fitted and m.size_L_power == pytest.approx(2.0, abs=0.15)
    assert m.offset_fitted and m.offset_um == pytest.approx(4.6, abs=1.0)
    assert abs(m.focus_L_amp) < 0.2 and abs(m.focus_L_width) < 0.2
    # a truth with a different power is recovered (the scaling is tested, not assumed)
    truth = ss.truth_model(); truth.size_L_power = 1.5; truth.L_ref_um = 60.0
    orig = ss.truth_model
    ss.truth_model = lambda: truth
    try:
        rows2 = []
        for size, seed in ((40.0, 2), (100.0, 3)):
            rows2 += ss.synth_rows(n_batches=2, beads=15, seed=seed, size=size)
    finally:
        ss.truth_model = orig
    assert fit_model(CalibrationTable.from_rows(rows2)).size_L_power == pytest.approx(1.5, abs=0.15)
    m, v, tr = run_calibration(t)
    report = format_report(m, v, tr)
    assert "fitted; 2 = Fresnel" in report and "by true size" in report and "40 µm particles" in report
    assert v["summary"]["marginal_side"]["mean_abs"] < v["summary"]["uncorrected"]["mean_abs"]


def test_single_size_keeps_fresnel_scaling(table):
    m = fit_model(table)
    assert not m.size_L_power_fitted and m.size_L_power == 2.0 and m.focus_L_amp == 0.0


def test_size_measure_choice_is_used_and_remembered(tmp_path):
    from calibration.sizing import CalibrationTable, SizingModel
    rows = [{"actual_length_um": "40", "length_um": "45", "deq_um": "41", "focus": "20", "d_signed_plane": "0.1"}]
    assert CalibrationTable.from_rows(rows).length[0] == 45
    t = CalibrationTable.from_rows(rows, "deq")
    assert t.length[0] == 41 and t.size_measure == "deq"
    m = SizingModel(size_measure="deq")
    m.save(tmp_path / "m.json")
    assert SizingModel.load(tmp_path / "m.json").size_measure == "deq"


def test_scatter_fitted_per_side():
    import numpy as np
    from sizing_synth import synth_table
    from calibration.sizing import fit_model, SizingModel
    t = synth_table()
    rng = np.random.default_rng(3)
    neg = t.d < 0
    t.length = t.length * np.where(neg, np.exp(rng.normal(0, 0.06, len(t.length))), 1.0)   # extra scatter, − side
    m = fit_model(t)
    assert m.size_sigma_minus is not None
    assert m.sigma_length(100.0, -0.3) > 1.5 * m.sigma_length(100.0, 0.3)
    old = SizingModel()                                    # models saved before v14.7: one scatter for both sides
    assert old.sigma_length(100.0, -0.3) == old.sigma_length(100.0, 0.3)


def test_pixel_size_set_and_fitted():
    """Data processed with a wrong pixel size: setting the right one, or fitting it, recovers the sizes."""
    import numpy as np
    from sizing_synth import synth_rows
    from calibration.sizing import CalibrationTable, fit_model, retrieve
    rows = []
    for k, s in enumerate((40.0, 100.0)):
        for r in synth_rows(seed=k + 7, size=s):
            r["batch"] = f"{s}-{r.get('batch', '')}"
            r["length_um"] = str(float(r["length_um"]) * 1.06)          # pixel size 6% too large in processing
            rows.append(r)
    t = CalibrationTable.from_rows(rows)
    fitted = fit_model(t, pixel_mode="fitted")
    assert fitted.pixel_mode == "fitted" and fitted.scale == pytest.approx(1.0, abs=1e-6)
    assert fitted.length_factor == pytest.approx(1 / 1.06, rel=0.03)
    set_ = fit_model(t, pixel_mode="set", pixel_set_um=2.3 / 1.06)
    assert set_.length_factor == pytest.approx(1 / 1.06) and set_.scale == pytest.approx(1.0, abs=1e-6)
    near = np.abs(t.d) < 0.1
    r = retrieve(fitted, t.length[near], t.focus[near], t.illum[near], side="both")
    err = np.nanmedian(np.abs(r["L_median"] - t.true_length[near]) / t.true_length[near])
    assert err < 0.05
    with pytest.raises(ValueError):
        fit_model(CalibrationTable.from_rows([r for r in rows if r["batch"].startswith("40")]), pixel_mode="fitted")
