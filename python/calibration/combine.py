"""Stack the per-particle tables of several calibration projects into one CSV.

Each saved project writes ``<project>_particle_positions.csv`` (one row per
accepted particle: label, true size, measured size, focus, illumination,
stage X/Z, the batch's Xbest/Zbest and the signed distance d).  This module
concatenates those tables and adds which project and which CPI each row came
from, for a calibration fit across datasets.  Projects are not modified.

Command line::

    cpi-combine combined.csv projectA.json projectB.json ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

LEAD_COLUMNS = ("project", "cpi_type", "cpi_id")
DISPLAY = {"CPI3v": "CPI-3V", "CPI1.0": "CPI 1.0"}


def export_path(project_file) -> Path:
    path = Path(project_file)
    return path.with_name(path.stem + "_particle_positions.csv")


def project_instrument(project_file):
    """(cpi_type, cpi_id) recorded in a project ('' when not set)."""
    try:
        config = json.loads(Path(project_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    inst = config.get("instrument") or {}
    cpi = str(inst.get("type") or inst.get("detected") or "")
    return DISPLAY.get(cpi, cpi), str(inst.get("id", ""))


def _utc(timestamp):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def load_projects_for_fit(project_files):
    """Rows for a calibration fit, read live from each project's saved particle table.

    Returns (rows, inputs, problems).  ``inputs`` records which projects were
    used and when each and its table were last saved (provenance for the
    model).  ``problems`` lists blocking issues (missing tables, different
    CPIs) and ``warnings`` inside inputs flag stale or old-format tables.
    """
    rows, inputs, problems = [], [], []
    instruments = {}
    for project_file in project_files:
        project = Path(project_file).resolve()
        table = export_path(project)
        try:
            config = json.loads(project.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{project.name}: cannot read project ({exc})")
            continue
        if not table.exists():
            problems.append(f"{project.name}: no particle table yet — open the project and Save project")
            continue
        cpi_type, cpi_id = project_instrument(project)
        instruments[project.stem] = (cpi_type, cpi_id)
        warnings = []
        if config.get("merged_from"):
            warnings.append("is a merged copy of " + ", ".join(config["merged_from"]) +
                            " — fitting it together with those projects would count particles twice")
        if table.stat().st_mtime + 2 < project.stat().st_mtime:
            warnings.append("particle table is older than the project — open and Save project to refresh it")
        count = 0
        with table.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "batch" not in (reader.fieldnames or ()):
                warnings.append("table from an older version (no batch column) — open and Save project")
            for row in reader:
                row["cpi_type"] = row.get("cpi_type") or cpi_type
                row["cpi_id"] = row.get("cpi_id") or cpi_id
                row["project"] = project.stem
                rows.append(row)
                count += 1
        inputs.append({"project": project.stem, "path": str(project), "particles": count,
                       "project_saved_utc": _utc(project.stat().st_mtime),
                       "table_saved_utc": _utc(table.stat().st_mtime),
                       "cpi_type": cpi_type, "cpi_id": cpi_id, "warnings": warnings})
    types = {t for t, _ in instruments.values() if t}
    ids = {i for _, i in instruments.values() if i}
    if len(types) > 1 or len(ids) > 1:
        problems.append("projects are from different CPIs: " + "; ".join(
            f"{n} = {' '.join(v for v in pair if v) or 'not set'}" for n, pair in instruments.items()))
    return rows, inputs, problems


def combine_projects(project_files, output):
    """Write the combined table; return a summary dict.

    Columns are the union of all tables (older exports may lack newer columns,
    which are left blank), led by project / cpi_type / cpi_id.  Projects whose
    table is missing are reported, not fatal.
    """
    rows, columns, used, missing = [], [], [], []
    for project_file in project_files:
        table = export_path(project_file)
        if not table.exists():
            missing.append(str(project_file))
            continue
        cpi_type, cpi_id = project_instrument(project_file)
        with table.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or ():
                if name not in columns and name not in LEAD_COLUMNS:
                    columns.append(name)
            count = 0
            for row in reader:
                row.setdefault("cpi_type", "")
                if not row.get("cpi_type"):
                    row["cpi_type"] = cpi_type
                if not row.get("cpi_id"):
                    row["cpi_id"] = cpi_id
                row["project"] = Path(project_file).stem
                rows.append(row)
                count += 1
        used.append((str(project_file), count))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(LEAD_COLUMNS) + columns
    fd, tmp = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".tmp", dir=str(output.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, output)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"rows": len(rows), "projects": used, "missing": missing, "output": str(output)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Combine calibration projects' per-particle tables")
    parser.add_argument("output", help="combined CSV to write")
    parser.add_argument("projects", nargs="+", help="calibration project .json files")
    args = parser.parse_args(argv)
    summary = combine_projects(args.projects, args.output)
    for name, count in summary["projects"]:
        print(f"{count:8d}  {name}")
    for name in summary["missing"]:
        print(f"  missing table (open and save the project first): {name}")
    print(f"{summary['rows']} rows → {summary['output']}")
    return 0 if summary["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
