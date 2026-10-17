"""Working-set edits are reversible and must not touch review flags."""
from calibration.selection_state import combine_selection, normalise_sets, make_saved_set


def test_start_camera_then_refine_on_another_plot():
    camera = {"p1", "p2", "p3", "p4"}
    focus = {"p2", "p3", "p5"}
    length = {"p3", "p4"}
    working = combine_selection(set(), camera, "refine")
    assert working == camera
    working = combine_selection(working, focus, "refine")
    assert working == {"p2", "p3"}
    working = combine_selection(working, length, "refine")
    assert working == {"p3"}


def test_add_subtract_replace_and_empty_start():
    assert combine_selection(set(), {"a"}, "refine") == {"a"}
    assert combine_selection({"a"}, {"b"}, "add") == {"a", "b"}
    assert combine_selection({"a", "b"}, {"a"}, "subtract") == {"b"}
    assert combine_selection({"a", "b"}, {"c"}, "replace") == {"c"}
    assert combine_selection({"a"}, set(), "refine") == set()


def test_saved_sets_are_independent_snapshots():
    original = {"a", "b"}
    sets = {"Set 1": make_saved_set(original, "Calibration")}
    original.clear()
    assert sets["Set 1"]["particle_ids"] == ["a", "b"]
    sets = normalise_sets(sets)
    assert sets["Set 1"]["particle_ids"] == ["a", "b"]
    del sets["Set 1"]
    assert not sets


def test_invalid_saved_sets_are_ignored():
    assert normalise_sets({"bad": {"particle_ids": "not-a-list"}}) == {}
    assert normalise_sets({"valid": {"particle_ids": ["a", "a", "b", 10]}})["valid"]["particle_ids"] == ["a", "b"]


def test_remove_review_batch_does_not_undo_another_same_status_batch():
    from calibration.selection_state import exclusive_batch_ids
    batches = {
        "Accepted batch 1": {"particle_ids": ["a", "b"], "description": "Review batch: keep"},
        "Accepted batch 2": {"particle_ids": ["b", "c"], "description": "Review batch: keep"},
        "Rejected batch 1": {"particle_ids": ["a"], "description": "Review batch: reject"},
    }
    assert exclusive_batch_ids(batches, "Accepted batch 1") == {"a"}
    assert exclusive_batch_ids(batches, "Accepted batch 2") == {"c"}
