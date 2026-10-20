"""Box gestures, pick scopes and median traces used by the redesigned workbench."""
import numpy as np
import pytest

from calibration.plot_utils import median_trace, median_scan_curve, trace_extremum
from calibration.selection_state import apply_box


STATUS = {"u1": "undecided", "u2": "undecided", "k1": "keep", "r1": "reject"}


def test_new_box_skips_accepted_particles_by_default():
    sel, rej = apply_box(set(), {"u1", "k1", "r1"}, "new", STATUS)
    assert sel == {"u1"} and rej == set()
    sel, _ = apply_box(set(), {"u1", "k1"}, "new", STATUS, pick_scope="keep")
    assert sel == {"k1"}


def test_add_remove_keep_only():
    sel, _ = apply_box({"u1"}, {"u2", "k1"}, "add", STATUS)
    assert sel == {"u1", "u2"}
    sel, _ = apply_box({"u1", "u2", "k1"}, {"u2", "k1"}, "remove", STATUS)
    assert sel == {"u1"}
    sel, _ = apply_box({"u1", "u2", "k1"}, {"u2", "k1"}, "keep", STATUS)
    assert sel == {"u2", "k1"}


def test_reject_box_limits_itself_to_the_selection():
    sel, rej = apply_box({"u1", "u2"}, {"u2", "k1"}, "reject", STATUS)
    assert rej == {"u2"} and sel == {"u1"}
    # No selection: reject pickable particles inside, never re-reject.
    sel, rej = apply_box(set(), {"u1", "k1", "r1"}, "reject", STATUS)
    assert rej == {"u1"} and sel == set()


def test_median_trace_matches_scan_curve_for_discrete_stage_positions():
    x = np.repeat([1.0, 2.0, 3.0], 5)
    y = np.array([1, 2, 3, 4, 100, 5, 6, 7, 8, 9, 2, 2, 2, 2, 2], dtype=float)
    trace = median_trace(x, y)
    cx, cy = median_scan_curve(x, y)
    assert np.allclose(trace["x"], cx) and np.allclose(trace["median"], cy)
    assert trace["q25"][0] == pytest.approx(2) and trace["q75"][0] == pytest.approx(4)
    assert trace_extremum(trace) == (2.0, 7.0)
    assert trace_extremum(trace, "minimum") == (3.0, 2.0)


def test_median_trace_bins_continuous_x_and_ignores_nan():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y = 3 * x
    y[::50] = np.nan
    trace = median_trace(x, y, max_groups=20)
    assert 2 <= len(trace["x"]) <= 20
    assert np.all(np.diff(trace["x"]) > 0)
    assert np.allclose(trace["median"], 3 * trace["x"], atol=0.15)
    assert len(median_trace([1.0], [2.0])["x"]) == 0


def test_label_library_shared_sizes_renames_and_persistence(tmp_path):
    from types import SimpleNamespace
    import math
    from calibration.labels import LabelLibrary, ParticleLabel, apply_label_dimensions
    path = tmp_path / "labels.json"
    lib = LabelLibrary(path)
    lib.upsert(ParticleLabel("PSL 20", length_um=20))
    lib.upsert(ParticleLabel("Column A", kind="ice analogue", length_um=150, width_um=40))
    lib.save()
    lib = LabelLibrary.load(path)
    assert lib.get("Column A").size_text() == "150 × 40 µm"
    beads = [SimpleNamespace(label="PSL 20", actual_length_um=math.nan, actual_width_um=math.nan),
             SimpleNamespace(label="", actual_length_um=7.0, actual_width_um=7.0)]
    assert apply_label_dimensions(beads, lib) == 1
    assert (beads[0].actual_length_um, beads[0].actual_width_um) == (20, 20)
    assert beads[1].actual_length_um == 7.0  # unlabelled: keeps spreadsheet value
    # Edit size globally, and rename: old datasets still resolve via alias.
    lib.upsert(ParticleLabel("PSL 20.5", length_um=20.5), old_name="PSL 20")
    lib.save()
    lib = LabelLibrary.load(path)
    old = SimpleNamespace(label="PSL 20", actual_length_um=math.nan, actual_width_um=math.nan)
    apply_label_dimensions([old], lib)
    assert old.label == "PSL 20.5" and old.actual_length_um == 20.5
    assert lib.names() == ["PSL 20.5", "Column A"]
    assert lib.merge([{"name": "Column A", "length_um": 1}, {"name": "New", "length_um": 3}]) == ["New"]
    assert lib.get("Column A").length_um == 150


def test_flags_csv_round_trips_label(tmp_path):
    from datetime import datetime
    from calibration.data import ParticleCatalog, ParticleRecord, datetime_to_matlab_datenum
    rec = ParticleRecord("s.mat", 0, datetime_to_matlab_datenum(datetime(2002, 3, 21)), 10, 10, 5, 1, 1, 1, 1, 1)
    rec.label, rec.status = "PSL 20", "keep"
    cat = ParticleCatalog(tmp_path, [rec])
    cat.save_selection_file(tmp_path / "f.csv")
    rec.label = ""
    cat.apply_selection_file(tmp_path / "f.csv")
    assert rec.label == "PSL 20"


def test_signed_plane_distance_keeps_the_side_of_the_plane():
    import math
    from types import SimpleNamespace
    from calibration.reference_sets import (make_reference, signed_plane_distance, in_plane_offset,
                                            corrected_stage_radius, scan_direction_deg,
                                            materialized_reference_rows)
    from calibration.selection_state import add_ids_to_saved_set
    ref = make_reference(10, 5)
    # Scan along 45°: points either side of best focus get opposite signs, same magnitude.
    a = signed_plane_distance(11, 6, ref, 45)
    b = signed_plane_distance(9, 4, ref, 45)
    assert a == pytest.approx(math.sqrt(2)) and b == pytest.approx(-math.sqrt(2))
    assert corrected_stage_radius(11, 6, ref) == pytest.approx(corrected_stage_radius(9, 4, ref))
    assert in_plane_offset(11, 6, ref, 45) == pytest.approx(0)
    # A move within the plane is distance 0 from it.
    assert signed_plane_distance(9, 6, ref, 45) == pytest.approx(0)
    assert scan_direction_deg([9, 10, 11, 12], [4, 5, 6, 7]) == pytest.approx(45)
    assert scan_direction_deg([12, 11, 10], [5, 5, 5]) == pytest.approx(0)
    assert scan_direction_deg([1], [1]) is None
    sets = {}
    add_ids_to_saved_set(sets, "A", ["p"], ref)
    [row] = materialized_reference_rows(sets, [SimpleNamespace(particle_id="p", stage_x=9., stage_y=4.)], 45)
    assert row["d_signed_plane"] == pytest.approx(-math.sqrt(2)) and row["plane_normal_deg"] == 45


def test_time_stepped_default_references():
    from calibration.reference_sets import make_reference, reference_steps, reference_at_time, \
        effective_reference, build_reference_index
    from calibration.selection_state import normalise_sets
    a, b = make_reference(10, 5), make_reference(11, 5)
    sets = {
        "Run 1": {"particle_ids": ["p1", "p2"], "best_focus": a},
        "Accepted batch 2": {"particle_ids": ["p5"], "description": "Review batch: keep"},   # no X/Z
        "Run 2": {"particle_ids": ["p8"], "best_focus": b},
    }
    times = {"p1": 1.0, "p2": 1.5, "p5": 2.0, "p8": 3.0, "p0": 0.5, "p9": 9.0}
    steps = reference_steps(sets, times)
    assert [(t, n) for t, _, n in steps] == [(1.0, "Run 1"), (3.0, "Run 2")]
    assert reference_at_time(steps, 2.0)[1] == "Run 1"     # batch between: last defined before it
    assert reference_at_time(steps, 3.0)[1] == "Run 2"     # step takes effect at its start
    assert reference_at_time(steps, 9.0)[1] == "Run 2"
    assert reference_at_time(steps, 0.5)[1] == "Run 1"     # before all: earliest
    index = build_reference_index(sets)
    assert effective_reference("p5", 2.0, index, steps)[2] == "step"
    assert effective_reference("p8", 3.0, index, steps)[0] == b
    # A stored start time survives a project round trip and wins over member times.
    sets["Run 2"]["ref_time_matlab"] = 2.5
    again = normalise_sets(sets)
    assert again["Run 2"]["ref_time_matlab"] == 2.5
    assert reference_at_time(reference_steps(again, times), 2.6)[1] == "Run 2"


def test_cluster_focus_peak_sessions_and_nearest_group():
    import numpy as np
    from calibration.reference_sets import (focus_peak_reference, session_labels, nearest_group_reference,
                                            make_reference)
    x = np.repeat(np.linspace(10, 12, 9), 5)
    z = np.full_like(x, 5.0)                                   # only X scanned
    f = 30 * np.exp(-((x - 10.75) / 0.4) ** 2) + 5
    ref, why = focus_peak_reference(x, z, f)
    assert ref["x_best"] == pytest.approx(10.75) and ref["z_best"] == 5.0 and why == ""
    ref, why = focus_peak_reference(np.full(10, 10.0), np.full(10, 5.0), np.ones(10))
    assert ref is None and "not scanned" in why
    t = np.array([0.0, 0.001, 0.002, 0.5, 0.501, np.nan])     # 0.5 day gap → new session
    assert list(session_labels(t)) == [0, 0, 0, 1, 1, -1]
    groups = [dict(name="A", reference=make_reference(10, 5), session=0, cx=100, cy=100),
              dict(name="B", reference=make_reference(11, 5), session=0, cx=800, cy=800),
              dict(name="C", reference=make_reference(12, 5), session=1, cx=110, cy=110)]
    assert nearest_group_reference(700, 650, 0, groups)[1] == "B"
    assert nearest_group_reference(120, 120, 0, groups)[1] == "A"   # C is closer but another session
    assert nearest_group_reference(120, 120, 2, groups) == (None, None)


def test_roi_edge_brightness_measures_background_around_particle():
    import numpy as np
    from calibration.data import roi_edge_brightness
    im = np.full((20, 30), 180, dtype=np.uint8)
    im[6:14, 8:22] = 40                            # dark particle in the middle
    assert roi_edge_brightness(im) == 180
    im[0, :] = 0                                   # a few odd border pixels don't move the median
    assert roi_edge_brightness(im) == 180
    assert np.isnan(roi_edge_brightness(np.zeros((3, 3))))
    assert np.isnan(roi_edge_brightness(None))
