"""Pure selection operations and named-set persistence for calibration projects.

Particle IDs are stable source-file/ROI identifiers.  Working selections and named
sets never modify image data or review flags; explicit review commands do that.
"""
from __future__ import annotations

from datetime import datetime, timezone

try:
    from .reference_sets import validate_reference
except ImportError:
    from reference_sets import validate_reference


def combine_selection(current, matches, mode):
    """Return an updated working set; 'refine' seeds an empty selection.

    The first rectangle in refine mode starts a working set. Subsequent
    rectangles (including on other plots) intersect with the working set.
    """
    current, matches = set(current), set(matches)
    if mode == "refine":
        return current & matches if current else matches
    if mode == "add":
        return current | matches
    if mode == "subtract":
        return current - matches
    if mode == "replace":
        return matches
    raise ValueError(f"Unknown working-set operation: {mode}")


def normalise_sets(raw):
    """Validate saved sets loaded from project JSON; keep original particle IDs."""
    result = {}
    if not isinstance(raw, dict):
        return result
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, dict):
            continue
        ids = value.get("particle_ids", [])
        if not isinstance(ids, list):
            continue
        entry = {
            "particle_ids": sorted(set(pid for pid in ids if isinstance(pid, str))),
            "created_utc": str(value.get("created_utc", "")),
            "description": str(value.get("description", "")),
        }
        reference = validate_reference(value.get("best_focus"))
        if reference is not None:
            entry["best_focus"] = reference
            try:
                t = float(value.get("ref_time_matlab"))
                if t == t:  # not NaN
                    entry["ref_time_matlab"] = t
            except (TypeError, ValueError):
                pass
        result[name] = entry
    return result


def make_saved_set(ids, description=""):
    return {
        "particle_ids": sorted(set(ids)),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": description,
    }


def exclusive_batch_ids(saved_sets, name):
    """IDs from a review batch not covered by another same-decision batch.

    Used when removing a classified batch so repeated acceptance/rejection of a
    particle isn't accidentally undone by deleting one of several overlapping
    batches. Individual manual flags have no provenance, so the GUI still asks
    for confirmation before clearing any current statuses.
    """
    batch = saved_sets[name]
    description = batch.get("description", "")
    other_ids = set()
    for other_name, other in saved_sets.items():
        if other_name != name and other.get("description") == description:
            other_ids.update(other["particle_ids"])
    return set(batch["particle_ids"]) - other_ids


def add_ids_to_saved_set(sets, name, particle_ids, reference=None):
    """Update a committed set without silently changing an existing X/Z reference."""
    if not name or not name.strip():
        raise ValueError("Choose a set name")
    ids = {str(pid) for pid in particle_ids}
    if not ids:
        raise ValueError("No particles to commit")
    current = sets.get(name)
    if current is None:
        if reference is None:
            raise ValueError("A new set needs a best-focus reference")
        current = make_saved_set(ids, description="Calibration working set")
        current["best_focus"] = validate_reference(reference)
        if current["best_focus"] is None:
            raise ValueError("Invalid best-focus X/Z reference")
        sets[name] = current
    else:
        if current.get("description", "").startswith("Review batch: "):
            raise ValueError("Cannot add calibration particles to a review batch")
        current["particle_ids"] = sorted(set(current["particle_ids"]) | ids)
    return current


def remove_ids_from_saved_set(sets, name, particle_ids):
    """Remove only set membership, keeping X/Z and source images untouched."""
    if name not in sets:
        raise KeyError(name)
    current = sets[name]
    current["particle_ids"] = sorted(set(current["particle_ids"]) - set(particle_ids))
    return current


# Box-selection modes shown in the GUI, mapped onto set operations.
BOX_MODE_OPERATIONS = {"new": "replace", "add": "add", "remove": "subtract", "keep": "refine"}

# Which review states a "New"/"Add" box may pick up.  Defaulting to
# unreviewed means accepted (green) clusters are never picked up twice.
PICK_SCOPES = {
    "undecided": {"undecided"},
    "keep": {"keep"},
    "reject": {"reject"},
    "unrejected": {"undecided", "keep"},
    "any": {"undecided", "keep", "reject"},
}


def apply_box(current, inside, mode, statuses, pick_scope="undecided"):
    """Apply one box gesture to the working selection.

    ``inside`` are particle IDs inside the box (already limited to visible
    points); ``statuses`` maps ID -> review status.  Returns
    ``(new_selection, to_reject)``.  Only "new" and "add" honour the pick scope:
    "remove" and "keep" act on whatever is inside the box.  For "reject", the
    box rejects selected particles inside it when a selection exists,
    otherwise every pickable, not-yet-rejected particle inside it; rejected
    particles leave the selection.
    """
    current, inside = set(current), set(inside)
    allowed = PICK_SCOPES.get(pick_scope, PICK_SCOPES["undecided"])
    if mode == "reject":
        if current:
            targets = inside & current
        else:
            targets = {pid for pid in inside
                       if statuses.get(pid) in allowed and statuses.get(pid) != "reject"}
        return current - targets, targets
    if mode in ("new", "add"):
        inside = {pid for pid in inside if statuses.get(pid) in allowed}
    if mode not in BOX_MODE_OPERATIONS:
        raise ValueError(f"Unknown box mode: {mode}")
    return combine_selection(current, inside, BOX_MODE_OPERATIONS[mode]), set()


# ----------------------------------------------------------------------------- compact project storage
# Particle IDs are "<absolute MAT path>#<roi>".  Writing every ID in full made
# project files tens of MB (the path repeated for every particle of every
# batch).  Projects (format 4) store them grouped by file instead; older plain
# lists still load.

def compact_ids(ids):
    groups, other = {}, []
    for pid in ids:
        src, sep, roi = str(pid).rpartition("#")
        if sep and roi.isdigit():
            groups.setdefault(src, []).append(int(roi))
        else:
            other.append(pid)
    out = {"by_source": [[src, sorted(set(rois))] for src, rois in sorted(groups.items())]}
    if other:
        out["other"] = other
    return out


def expand_ids(value):
    if isinstance(value, dict) and "by_source" in value:
        ids = [f"{src}#{roi}" for src, rois in value["by_source"] for roi in rois]
        return ids + list(value.get("other", []))
    return list(value or [])


def compact_keys(keys):
    """{pid: [ms, 2cx, 2cy]} grouped by file."""
    groups = {}
    for pid, key in keys.items():
        src, sep, roi = str(pid).rpartition("#")
        if sep and roi.isdigit():
            groups.setdefault(src, []).append([int(roi)] + list(key))
    return {"by_source": [[src, sorted(rows)] for src, rows in sorted(groups.items())]}


def expand_keys(value):
    if isinstance(value, dict) and "by_source" in value:
        return {f"{src}#{row[0]}": list(row[1:]) for src, rows in value["by_source"] for row in rows}
    return dict(value or {})


def expand_project_config(config):
    """Turn a format-4 (compact) project into the plain in-memory form, in place."""
    for entry in (config.get("named_sets") or {}).values():
        if isinstance(entry, dict):
            entry["particle_ids"] = expand_ids(entry.get("particle_ids"))
    config["particle_keys"] = expand_keys(config.get("particle_keys"))
    config["selected_ids"] = expand_ids(config.get("selected_ids"))
    return config


def compact_project_sets(named_sets):
    return {name: dict(entry, particle_ids=compact_ids(entry.get("particle_ids", ())))
            for name, entry in named_sets.items()}
