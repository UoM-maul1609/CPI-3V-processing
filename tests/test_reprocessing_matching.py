"""Flags, labels and sets must survive reprocessing that renumbers ROIs."""
import csv
import os
from datetime import datetime

import pytest

from calibration.data import ParticleCatalog, ParticleRecord, PositionLog, datetime_to_matlab_datenum, particle_key

T0 = datetime_to_matlab_datenum(datetime(2002, 3, 21, 13, 15, 31))


def rec(source, roi, ms, sx, sy, length=40.0, focus=20.0, w=40):
    r = ParticleRecord(source, roi, T0 + ms / 86400000.0, length, length, focus, 1.0,
                       (sx + sx + w) / 2, (sy + sy + w) / 2, 1, 1)
    r.start_x, r.end_x, r.start_y, r.end_y = sx, sx + w, sy, sy + w
    return r


def original():
    return [rec("a.mat", 0, 0, 100, 100), rec("a.mat", 1, 0, 500, 100),   # same frame, two ROIs
            rec("a.mat", 2, 40, 300, 300), rec("a.mat", 3, 80, 700, 700)]


def reprocessed():
    # New criteria found an extra ROI at the start → every index shifts by one,
    # and the new algorithm reports different lengths / focus.
    return [rec("a.mat", 0, 0, 900, 900, 12, 3),
            rec("a.mat", 1, 0, 100, 100, 44, 25), rec("a.mat", 2, 0, 500, 100, 47, 22),
            rec("a.mat", 3, 40, 300, 300, 41, 18), rec("a.mat", 4, 80, 700, 700, 39, 30)]


def test_flags_follow_particles_after_reprocessing(tmp_path):
    old = ParticleCatalog(tmp_path, original())
    old.records[1].status, old.records[1].label = "keep", "PSL 40"
    old.records[3].status, old.records[3].reject_reason = "reject", "debris"
    old.save_selection_file(tmp_path / "flags.csv")
    new = ParticleCatalog(tmp_path, reprocessed())
    report = new.apply_selection_file(tmp_path / "flags.csv")
    by_pos = {(r.start_x, r.start_y): r for r in new.records}
    assert by_pos[(500, 100)].status == "keep" and by_pos[(500, 100)].label == "PSL 40"
    assert by_pos[(700, 700)].status == "reject" and by_pos[(700, 700)].reject_reason == "debris"
    assert by_pos[(900, 900)].status == "undecided"          # new ROI untouched
    assert by_pos[(100, 100)].status == "undecided"          # old index 1 now points elsewhere — not confused
    assert by_pos[(500, 100)].length_um == 47                 # new measurements kept
    assert report.remapped == 4 and report.unmatched == 0
    assert report.id_map["a.mat#1"] == "a.mat#2"


def test_legacy_flags_without_roi_corners_still_match(tmp_path):
    old = ParticleCatalog(tmp_path, original())
    old.records[2].status = "keep"
    old.save_selection_file(tmp_path / "flags.csv")
    with open(tmp_path / "flags.csv") as h:
        rows = list(csv.DictReader(h))
    keep = [c for c in rows[0] if c not in ("start_x", "end_x", "start_y", "end_y", "label")]
    with open(tmp_path / "legacy.csv", "w", newline="") as h:   # v10-style file
        w = csv.DictWriter(h, fieldnames=keep)
        w.writeheader()
        w.writerows({k: r[k] for k in keep} for r in rows)
    new = ParticleCatalog(tmp_path, reprocessed())
    new.apply_selection_file(tmp_path / "legacy.csv")
    assert [r.status for r in new.records if r.start_x == 300] == ["keep"]


def test_particle_key_is_stable_and_handles_missing():
    r = rec("a.mat", 0, 40, 300, 300)
    assert r.match_key == particle_key(r.time_matlab, r.image_x_px, r.image_y_px)
    assert r.match_key[1:] == (640, 640)
    assert particle_key(float("nan"), 1, 1) is None and particle_key("", 1, 1) is None


def test_stage_csv_all_series_by_default(tmp_path):
    path = tmp_path / "stage.csv"
    path.write_text("series,time_basis,start_time,stage_x_original,stage_z_original\n"
                    "beads,roi,2002-03-21 13:00:00,1,2\n"
                    "analogue,roi,2002-03-21 14:00:00,3,4\n"
                    "analogue,stage,2002-03-21 14:00:00,9,9\n")
    assert PositionLog.csv_choices(path) == {"series": ["analogue", "beads"], "time_basis": ["roi", "stage"]}
    assert len(PositionLog.from_text(path).times) == 2           # all series, roi basis
    assert len(PositionLog.from_text(path, "beads").times) == 1
    assert PositionLog.from_text(path, None, None).conflicting_times() == 1


def test_flags_remapped_by_roi_image_when_times_change(tmp_path):
    """Reprocessing that renumbers ROIs and changes their times still carries every flag across."""
    import csv
    import numpy as np
    import scipy.io as sio
    from calibration.remap import remap_flags_file, roi_signatures
    rng = np.random.default_rng(5)
    n = 30
    corners = rng.integers(0, 900, size=(n, 2))
    corners[5] = corners[6]                     # two ROIs at the same place: told apart by their pixels
    images = [rng.integers(0, 255, size=(12, 10), dtype=np.uint8) for _ in range(n)]
    times = 731296.5 + np.arange(n) * 1e-6

    def write(path, order, dt):
        imgs = np.empty((n,), dtype=[("IM", object)])
        for k, i in enumerate(order):
            imgs[k]["IM"] = images[i]
        roi = {"Time": times[order] + dt, "StartX": corners[order, 0], "EndX": corners[order, 0] + 11,
               "StartY": corners[order, 1], "EndY": corners[order, 1] + 9, "IMAGE": imgs}
        sio.savemat(path, {"ROI_N": roi})

    old, new = tmp_path / "old" / "a.mat", tmp_path / "new" / "a.mat"
    old.parent.mkdir(); new.parent.mkdir()
    write(old, np.arange(n), 0.0)
    perm = rng.permutation(n)
    write(new, perm, 0.15 / 86400)              # ROIs renumbered and 0.15 s later
    flags = tmp_path / "flags.csv"
    with open(flags, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["particle_id", "source_file", "roi_index", "time_matlab", "status", "start_x", "start_y"])
        for i in range(n):
            w.writerow([f"a.mat#{i}", "a.mat", i, times[i], "keep" if i % 3 else "reject", corners[i, 0], corners[i, 1]])
    out = tmp_path / "flags_new.csv"
    report = remap_flags_file(flags, out, {"a.mat": (old, new)})
    assert report["remapped"] == n and report["missing"] == 0
    new_sigs = roi_signatures(new)
    old_sigs = roi_signatures(old)
    rows = list(csv.DictReader(open(out)))
    for i, row in enumerate(rows):
        j = int(row["roi_index"])
        assert new_sigs[j][0] == old_sigs[i][0]                   # same ROI image
        assert abs(float(row["time_matlab"]) - (times[i] + 0.15 / 86400)) < 1e-9


def test_roi_parent_frame_with_two_byte_alignments():
    """IMAGE blocks found at even then odd byte offsets: each ROI gets the frame just before it in the file."""
    import sys, numpy as np
    sys.path.insert(0, "python/proc")
    from postProcess import parent_image_index
    image_order = np.array([10, 50, 90, 30, 70], np.uint32)      # two sorted runs (aligned, shifted)
    roi_order = np.array([12, 35, 55, 75, 95, 5], np.uint32)
    assert parent_image_index(image_order, roi_order).tolist() == [0, 3, 1, 4, 2, -1]
