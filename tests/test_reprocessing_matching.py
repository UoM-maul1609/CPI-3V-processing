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
