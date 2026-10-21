"""Stage positions from interval CSVs: end times, and the burst check."""
import csv

import numpy as np

from calibration.data import PositionLog, datetime_to_matlab_datenum as dn, stage_burst_check
from datetime import datetime, timedelta

T0 = datetime(2002, 3, 21, 13, 0, 0)


def write_csv(path, rows):
    fields = ["series", "stage_x_original", "stage_z_original", "start_time", "end_time_exclusive", "time_basis"]
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dict(zip(fields, r)))


def t(seconds):
    return dn(T0 + timedelta(seconds=seconds))


def test_end_times_leave_gaps_without_position(tmp_path):
    p = tmp_path / "s.csv"
    write_csv(p, [("b", 3.80, 4.60, "2002-03-21 13:00:00", "2002-03-21 13:00:30", "full_frame"),
                  ("b", 3.85, 4.65, "2002-03-21 13:01:00", "", "full_frame")])          # blank end: until next
    log = PositionLog.from_text(p, None, None)
    x, z = log.previous(np.array([t(-1), t(10), t(45), t(70), t(5000)]))
    assert np.isnan(x[0]) and x[1] == 3.80 and np.isnan(x[2])     # 13:00:45: stage between positions
    assert x[3] == 3.85 and x[4] == 3.85 and z[3] == 4.65


def test_legacy_constructor_holds_until_next():
    log = PositionLog(np.array([t(0), t(60)]), np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    x, _ = log.previous(np.array([t(30), t(90)]))
    assert list(x) == [1.0, 2.0]


def test_burst_check_finds_a_boundary_inside_a_burst():
    # bursts at 10-12 s and 40-42 s; positions change at 0 and 30 s (good) or 0 and 41 s (cuts burst 2)
    times = [t(s) for s in np.r_[np.linspace(10, 12, 20), np.linspace(40, 42, 20)]]
    good = PositionLog(np.array([t(0), t(30)]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    bad = PositionLog(np.array([t(0), t(41)]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert stage_burst_check(times, good) == {"bursts": 2, "split": 0, "particles_in_split": 0, "no_position": 0}
    r = stage_burst_check(times, bad)
    assert r["bursts"] == 2 and r["split"] == 1 and r["particles_in_split"] == 20
