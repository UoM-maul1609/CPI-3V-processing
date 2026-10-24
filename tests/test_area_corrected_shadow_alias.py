"""CPI area-corrected shadow size is algebraically contour equivalent-area size."""
import numpy as np
from calibration import sizing

def test_corrected_shadow_is_contour_area():
    rows = [dict(label='column', actual_length_um=100., deq_um=54.77,
                 shadow_deq_um=40., d_signed_plane=0., focus=5., local_illum_bg=100.)]
    # These size-measure choices must use identical input diameter (and hence
    # no microscopy information is needed at retrieval time).
    a = sizing.CalibrationTable.from_rows(rows, 'deq')
    b = sizing.CalibrationTable.from_rows(rows, 'area_corrected_shadow')
    np.testing.assert_array_equal(a.length, b.length)
    assert b.length[0] == 54.77
