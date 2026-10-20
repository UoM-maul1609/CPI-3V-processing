"""Combining projects' particle tables, and detecting which CPI the data came from."""
import csv
import json

import numpy as np

from calibration.combine import combine_projects
from calibration.data import detect_cpi_type


def test_detect_cpi_type_from_image_type_codes():
    assert detect_cpi_type(np.array([33857] * 10 + [33797] * 2)) == "CPI1.0"
    assert detect_cpi_type([89, 89, 19]) == "CPI3v"
    assert detect_cpi_type([19, 19]) == "" and detect_cpi_type(None) == ""


def _project(tmp_path, name, instrument, rows, columns):
    (tmp_path / f"{name}.json").write_text(json.dumps({"instrument": instrument}))
    with open(tmp_path / f"{name}_particle_positions.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return tmp_path / f"{name}.json"


def test_combine_stacks_tables_with_project_and_instrument(tmp_path):
    a = _project(tmp_path, "beads_2021", {"type": "CPI3v", "id": "s/n 12"},
                 [{"cpi_type": "CPI-3V", "cpi_id": "s/n 12", "particle_id": "a#1", "label": "PSL 20", "d_signed_plane": 0.1}],
                 ["cpi_type", "cpi_id", "particle_id", "label", "d_signed_plane"])
    # An older export without instrument columns and with a column the other lacks.
    b = _project(tmp_path, "ice_2002", {"type": "", "detected": "CPI1.0", "id": ""},
                 [{"particle_id": "b#4", "label": "Column A", "local_illum": 150}],
                 ["particle_id", "label", "local_illum"])
    c = tmp_path / "never_saved.json"
    c.write_text("{}")
    summary = combine_projects([a, b, c], tmp_path / "out" / "all.csv")
    assert summary["rows"] == 2 and summary["missing"] == [str(c)]
    with open(tmp_path / "out" / "all.csv") as h:
        rows = list(csv.DictReader(h))
    assert list(rows[0])[:3] == ["project", "cpi_type", "cpi_id"]
    assert (rows[0]["project"], rows[0]["cpi_type"], rows[0]["cpi_id"]) == ("beads_2021", "CPI-3V", "s/n 12")
    assert (rows[1]["project"], rows[1]["cpi_type"], rows[1]["local_illum"]) == ("ice_2002", "CPI 1.0", "150")
    assert rows[0]["local_illum"] == "" and rows[1]["d_signed_plane"] == ""


def test_load_projects_for_fit_records_inputs_and_flags_problems(tmp_path):
    import csv, json, os, time
    from calibration.combine import load_projects_for_fit

    def project(name, cpi="CPI1.0", merged=None, batch=True, stale=False):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"instrument": {"type": cpi, "id": "s/n 1"}, **({"merged_from": merged} if merged else {})}))
        t = tmp_path / f"{name}_particle_positions.csv"
        cols = ["particle_id", "actual_length_um", "length_um", "d_signed_plane"] + (["batch"] if batch else [])
        with t.open("w", newline="") as h:
            w = csv.writer(h); w.writerow(cols); w.writerow(["a#1", 100, 104, 0.1] + (["Accepted batch 1"] if batch else []))
        if stale:
            old = time.time() - 100
            os.utime(t, (old, old))
        return str(p)

    a, b = project("run1"), project("run2", batch=False, stale=True)
    rows, inputs, problems = load_projects_for_fit([a, b])
    assert not problems and len(rows) == 2 and {r["project"] for r in rows} == {"run1", "run2"}
    assert [i["project"] for i in inputs] == ["run1", "run2"] and inputs[0]["particles"] == 1
    assert not inputs[0]["warnings"]
    assert any("older version" in w for w in inputs[1]["warnings"])
    assert any("older than the project" in w for w in inputs[1]["warnings"])
    m = project("merged", merged=["run1", "run2"])
    assert any("merged copy" in w for w in load_projects_for_fit([m])[1][0]["warnings"])
    c = project("other", cpi="CPI3v")
    assert any("different CPIs" in p for p in load_projects_for_fit([a, c])[2])
    (tmp_path / "run1_particle_positions.csv").unlink()
    assert any("no particle table" in p for p in load_projects_for_fit([a])[2])
