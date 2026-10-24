"""Shape/shadow validation must not use microscopy labels as retrieval inputs."""
import numpy as np
from calibration import sizing


def test_shadow_shape_matched_and_independent():
    rows = [
        dict(actual_length_um=100, shadow_deq_um=40, roundness=.3,
             label='column', d_signed_plane=.01, focus=5, local_illum=100),
        dict(actual_length_um=100, shadow_deq_um=np.nan, roundness=.3,
             label='column', d_signed_plane=.02, focus=5, local_illum=100),
        dict(actual_length_um=40, shadow_deq_um=30, roundness=np.nan,
             label='unknown', d_signed_plane=.03, focus=5, local_illum=100),
    ]
    table = sizing.CalibrationTable.from_rows(rows, 'shadow_deq')
    result = sizing.shadow_shape_diagnostics(table)
    assert result['n'] == 1
    assert np.isclose(result['shadow_ratio'][0] ** 2 / result['roundness'][0], .16/.3)
    assert len(sizing.make_shadow_shape_figure(table).axes) == 2


def test_other_sizing_method_requires_shadow_measurement():
    table = sizing.CalibrationTable.from_rows([
        dict(actual_length_um=40, length_um=40, roundness=.5,
             d_signed_plane=0, focus=5, local_illum=100)
    ])
    assert sizing.shadow_shape_diagnostics(table)['n'] == 0
