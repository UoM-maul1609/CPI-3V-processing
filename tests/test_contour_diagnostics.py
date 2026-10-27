"""Read-only geometry and optical-inside-polygon diagnostics."""
import numpy as np

from calibration.contour_diagnostics import contour_metrics, interior_light_metrics


def test_rectangular_feret_width_and_area_ratio():
    outline = np.array([[5., 7.], [5., 47.], [15., 47.], [15., 7.], [5., 7.]])
    values = contour_metrics(outline, pixel_um=1.)
    assert np.isclose(values['len_feret_um'], np.hypot(10, 40))
    assert np.isclose(values['wid_feret_um'], 10)
    assert np.isclose(values['contour_area_um2'], 400)
    assert 0 < values['area_ratio_circle'] < 1
    assert values['len_ellipse_um'] > values['wid_ellipse_um']


def test_optical_bright_hole_detected_without_changing_contour_area():
    boundary = np.array([[5., 5.], [5., 54.], [54., 54.], [54., 5.]])
    T = np.full((60,60), 1., dtype=float)
    T[5:55, 5:55] = .3
    T[23:37, 23:37] = 1.15
    props = interior_light_metrics(T, boundary)
    assert props['interior_bright_fraction'] > .06
    assert props['interior_above_one_fraction'] > .06
    assert props['interior_enclosed_bright_fraction'] > .06
    assert props['interior_mean_T'] < 1.0
    assert contour_metrics(boundary,1.)['contour_area_um2'] > 2000


def test_uniform_solid_shadow_no_bright_region():
    T = np.ones((60,60),float)
    T[7:53,7:53] = 0.1
    boundary = [[7,7],[7,52],[52,52],[52,7]]
    out = interior_light_metrics(T,boundary)
    assert out['interior_bright_fraction'] == 0
    assert out['interior_above_one_fraction'] == 0
    assert out['interior_enclosed_bright_fraction'] == 0


def test_missing_or_invalid_boundary_gives_no_metrics():
    assert not contour_metrics(None)
    assert not interior_light_metrics(np.ones((20,20)), None)
