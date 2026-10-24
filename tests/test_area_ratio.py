import numpy as np
from python.proc.areaRatio import contour_area_ratio


def test_circle_has_unit_area_ratio():
    t = np.linspace(0, 2*np.pi, 300, endpoint=False)
    boundary = np.column_stack((20 + 9*np.sin(t), 30 + 9*np.cos(t)))
    assert abs(contour_area_ratio(boundary, np.pi*9**2, 1) - 1) < .002


def test_column_ratio():
    boundary = np.array([[0, 0], [0, 10], [3, 10], [3, 0]])
    ratio = contour_area_ratio(boundary, 30, 1)
    assert 0 < ratio < 1


def test_no_valid_boundary():
    assert np.isnan(contour_area_ratio([], 10, 1))
