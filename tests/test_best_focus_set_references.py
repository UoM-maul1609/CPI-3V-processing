"""Selection-committed focus references must be editable and reproducible."""
import math
import json
from calibration.reference_sets import (
    make_reference, validate_reference, reference_for_particle,
    calibrated_radial_position, build_reference_index,
)
from calibration.selection_state import normalise_sets, make_saved_set


def test_commit_and_json_round_trip():
    selection = make_saved_set(["roi:1", "roi:2"])
    selection["best_focus"] = make_reference(3.4, 4.5)
    saved = normalise_sets(json.loads(json.dumps({"Run A": selection})))
    assert saved["Run A"]["particle_ids"] == ["roi:1", "roi:2"]
    assert saved["Run A"]["best_focus"]["best_radius"] == math.hypot(3.4, 4.5)


def test_change_x_z_without_changing_particle_membership():
    selection = make_saved_set(["1", "2", "3"])
    selection["best_focus"] = make_reference(2, 3)
    selection["best_focus"] = make_reference(4, 5)
    assert selection["particle_ids"] == ["1", "2", "3"]
    assert calibrated_radial_position(3, 4, selection["best_focus"]) == math.hypot(4, 5) - 5


def test_conflicting_overlap_requires_explicit_choice():
    sets = {
        "Placement 1": {"particle_ids": ["p1", "p2"], "best_focus": make_reference(3, 4)},
        "Placement 2": {"particle_ids": ["p2", "p3"], "best_focus": make_reference(6, 8)},
    }
    assert reference_for_particle("p2", sets)[1]
    assert build_reference_index(sets)["p2"] == (None, True)
    reference, ambiguous = reference_for_particle("p2", sets, "Placement 2")
    assert not ambiguous and reference["x_best"] == 6
    assert build_reference_index(sets, "Placement 2")["p2"][0]["best_radius"] == 10
    assert build_reference_index(sets)["p1"][0]["best_radius"] == 5
    assert "p9" not in build_reference_index(sets)


def test_legacy_set_preserved_without_fabricated_reference():
    sets = normalise_sets({"Older set": {"particle_ids": ["abc"], "description": ""}})
    assert sets["Older set"]["particle_ids"] == ["abc"]
    assert "best_focus" not in sets["Older set"]
    assert validate_reference({"x_best": "nan", "z_best": 1}) is None


def test_reference_removed_when_set_deleted():
    sets = {"old": {"particle_ids": ["a"], "best_focus": make_reference(1, 2)}}
    del sets["old"]
    assert build_reference_index(sets) == {}
