"""v10: stage-coordinate corrections and named calibration sets are non-destructive."""
import csv
import io
import json
import math
from types import SimpleNamespace

import pytest
from calibration.reference_sets import (
    corrected_stage_radius, calibrated_radial_position, make_reference,
    materialized_reference_rows, committed_particle_ids,
)
from calibration.selection_state import (
    add_ids_to_saved_set, remove_ids_from_saved_set, normalise_sets,
)


def test_corrected_stage_radius_differs_from_difference_in_absolute_radii():
    ref = make_reference(3., 4.)
    assert corrected_stage_radius(3., 4., ref) == pytest.approx(0)
    assert corrected_stage_radius(6., 8., ref) == pytest.approx(5)
    assert calibrated_radial_position(6., 8., ref) == pytest.approx(-5)
    assert corrected_stage_radius(4., 3., ref) == pytest.approx(math.sqrt(2))
    assert calibrated_radial_position(4., 3., ref) == pytest.approx(0)


def test_commit_many_and_single_then_remove_a_single_member():
    saved = {}
    ref = make_reference(3.5, 4.5)
    add_ids_to_saved_set(saved, 'Run 1', ['P1', 'P2'], ref)
    add_ids_to_saved_set(saved, 'Run 1', ['P3'], make_reference(8, 8))
    # Adding a particle must not silently replace the X/Z of the existing set.
    assert saved['Run 1']['best_focus'] == ref
    assert saved['Run 1']['particle_ids'] == ['P1', 'P2', 'P3']
    remove_ids_from_saved_set(saved, 'Run 1', ['P2'])
    assert saved['Run 1']['particle_ids'] == ['P1', 'P3']
    assert saved['Run 1']['best_focus'] == ref
    assert normalise_sets(json.loads(json.dumps(saved)))['Run 1'] == saved['Run 1']


def test_review_batches_do_not_look_like_processed_calibration_particles():
    sets = {}
    add_ids_to_saved_set(sets, 'Run 1', ['a', 'b'], make_reference(2, 3))
    sets['Keep batch'] = {'description': 'Review batch: keep', 'particle_ids': ['c']}
    assert committed_particle_ids(sets) == {'a', 'b'}


def test_export_has_both_radial_conventions_and_overlapping_set_rows():
    sets = {}
    add_ids_to_saved_set(sets, 'A', ['p'], make_reference(3, 4))
    add_ids_to_saved_set(sets, 'B', ['p'], make_reference(6, 8))
    row = SimpleNamespace(particle_id='p', stage_x=3., stage_y=4.)
    rows = materialized_reference_rows(sets, [row])
    assert len(rows) == 2
    a, b = rows
    assert a['set_name'] == 'A'
    assert a['z_p_from_offsets'] == pytest.approx(0)
    assert b['z_p_from_offsets'] == pytest.approx(5)
    assert b['z_p_radial'] == pytest.approx(5)
    assert b['x_minus_xbest'] == pytest.approx(-3)
    assert b['z_minus_zbest'] == pytest.approx(-4)


def test_missing_stage_coordinates_export_as_blanks():
    sets = {}
    add_ids_to_saved_set(sets, 'A', ['p'], make_reference(3, 4))
    missing = SimpleNamespace(particle_id='p', stage_x=float('nan'), stage_y=4.)
    [row] = materialized_reference_rows(sets, [missing])
    assert row['z_p_from_offsets'] == ''
    assert row['z_p_radial'] == ''


def test_reject_invalid_commit():
    saved = {}
    with pytest.raises(ValueError):
        add_ids_to_saved_set(saved, 'A', [], make_reference(1, 2))
    with pytest.raises(ValueError):
        add_ids_to_saved_set(saved, 'A', ['p'])
    assert saved == {}
