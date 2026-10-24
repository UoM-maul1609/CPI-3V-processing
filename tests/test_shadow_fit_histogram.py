"""Signed-shadow workbench diagnostic uses the identical measured ratio as Edge Lab.

Both observables must come from the same ROIs, and labelled size is only used
in the diagnostic ratio, not in the optical integral.
"""
import numpy as np
import pytest
from calibration import sizing


def _rows():
    return [
        dict(actual_length_um=15., shadow_deq_um=16., deq_um=17., length_um=19.,
             focus=24., local_illum=120., d_signed_plane=-.1, label="15"),
        dict(actual_length_um=40., shadow_deq_um=41., deq_um=44., length_um=45.,
             focus=30., local_illum=120., d_signed_plane=0., label="40"),
        dict(actual_length_um=99.4, shadow_deq_um=99.4, deq_um=102., length_um=105.,
             focus=24., local_illum=110., d_signed_plane=.1, label="100"),
        dict(actual_length_um=15., shadow_deq_um=np.nan, deq_um=16., length_um=18.,
             focus=20., local_illum=110., d_signed_plane=.2, label="15"),
    ]


def test_pairing_uses_shadow_to_label_ratio_from_same_roi():
    table = sizing.CalibrationTable.from_rows(_rows(), "shadow_deq", "focus_per_illum")
    edge, shadow, title, n = sizing.paired_contour_shadow_errors(table)
    assert n == 3  # exclude the ROI with a missing shadow measurement from BOTH curves
    assert title == "Contour equivalent-area diameter"
    np.testing.assert_allclose(shadow, [16 / 15 - 1, 41 / 40 - 1, 0])
    np.testing.assert_allclose(edge, [17 / 15 - 1, 44 / 40 - 1, 102 / 99.4 - 1])


def test_fallback_to_feret_if_no_contour_area():
    rows = _rows()
    for r in rows:
        r.pop("deq_um")
    table = sizing.CalibrationTable.from_rows(rows, "shadow_deq", "focus_per_illum")
    edge, shadow, title, n = sizing.paired_contour_shadow_errors(table)
    assert n == 3
    assert "Feret" in title
    assert edge[0] == pytest.approx(19/15 - 1)


def test_non_shadow_measure_does_not_request_comparison():
    table = sizing.CalibrationTable.from_rows(_rows(), "deq", "focus")
    assert sizing.paired_contour_shadow_errors(table) is None


def test_circle_fit_figure_has_paired_comparison_not_duplicate_histogram():
    rows = _rows() * 15
    table = sizing.CalibrationTable.from_rows(rows, "shadow_deq", "focus_per_illum")
    model = sizing.SizingModel(size_model="shadow_linear", focus_model="normalised_generalised",
                               offset_um=0., scale=1., focus_F0=.2, focus_floor=.01,
                               focus_w_minus_mm=.3, focus_w_plus_mm=.4, foc_crit=.1)
    fig = sizing.make_figure(model, table)
    assert "Contour vs shadow" in fig.axes[6].get_title()
    assert "same 45 ROIs" in fig.axes[6].get_title()
    assert len(fig.axes[6].patches) == 2  # one outline for each paired metric
    assert len(fig.axes[6].get_legend().texts) == 2
    assert "no hold-out" in fig.axes[4].get_title()
