import math
from types import SimpleNamespace
import numpy as np
from calibration.reference_sets import make_reference, relative_stage_position, materialized_reference_rows
from calibration.plot_utils import data_limits


def test_signed_offsets_and_radial_are_distinct():
    reference = make_reference(3, 4)
    assert relative_stage_position(4, 6, reference, 'x') == 1
    assert relative_stage_position(4, 6, reference, 'z') == 2
    # Radial difference uses stage radius, NOT stage z alone.
    r = SimpleNamespace(particle_id='p', stage_x=4., stage_y=6.)
    row = materialized_reference_rows({'beads': {'particle_ids': ['p'], 'best_focus': reference}}, [r])[0]
    assert row['x_minus_xbest'] == 1
    assert row['z_minus_zbest'] == 2
    assert math.isclose(row['z_p_radial'], 5 - math.sqrt(52))


def test_overlapping_sets_keep_distinct_committed_coordinates():
    records = [SimpleNamespace(particle_id='p', stage_x=4., stage_y=6.),
               SimpleNamespace(particle_id='missing', stage_x=np.nan, stage_y=np.nan)]
    sets = {'A': {'particle_ids': ['p', 'missing'], 'best_focus': make_reference(3, 4)},
            'B': {'particle_ids': ['p'], 'best_focus': make_reference(2, 5)}}
    rows = materialized_reference_rows(sets, records)
    assert len(rows) == 3
    assert [r['x_minus_xbest'] for r in rows if r['particle_id'] == 'p'] == [1., 2.]
    assert next(r for r in rows if r['particle_id'] == 'missing')['z_minus_zbest'] == ''


def test_first_plot_limits_show_all_points_and_ignore_absent_values():
    result = data_limits([3.1, 4.5, float('nan')], [1, 100, 999])
    assert result[0][0] < 3.1 < 4.5 < result[0][1]
    assert result[1][0] < 1 < 100 < result[1][1]
    assert data_limits([float('nan')], [2]) is None
    assert data_limits([3], [2])[0][0] < 3 < data_limits([3], [2])[0][1]
