import numpy as np
from calibration.sizing import CalibrationTable
from calibration.microscopy import detect_relative_shadow

def test_corrected_shadow_keeps_independent_integral():
    rows = [dict(actual_length_um=100, shadow_deq_um=40, area_ratio_circle=.30,
                 focus=20, local_illum=100, d_signed_plane=0)]
    raw = CalibrationTable.from_rows(rows, 'shadow_deq', 'focus_per_illum')
    corrected = CalibrationTable.from_rows(rows, 'area_corrected_shadow', 'focus_per_illum')
    assert abs(raw.length[0] - 40) < 1.e-8
    assert abs(corrected.length[0] - 40 / np.sqrt(.30)) < 1.e-8
    assert abs(corrected.focus[0] - .2) < 1.e-8

def test_relative_microscope_darkness_level():
    img = np.ones((100,100), np.uint8)*200
    img[30:70,30:70] = 130
    assert detect_relative_shadow(img, .25)
    assert not detect_relative_shadow(img, .5)
