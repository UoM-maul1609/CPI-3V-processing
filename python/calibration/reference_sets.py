"""Best-focus references attached to named, reproducible particle selections.

A named set is a snapshot of stable particle IDs.  Its best-focus coordinates
are values from the original micrometer axes, not translated camera positions.
The derived difference is radial, not a perpendicular optical-plane distance.
"""
from __future__ import annotations

import math


def make_reference(x_best, z_best):
    """Store original (X,Z), alongside derived scalar reference radius."""
    x, z = float(x_best), float(z_best)
    if not (math.isfinite(x) and math.isfinite(z)):
        raise ValueError("X best and Z best must be finite numbers")
    return {"x_best": x, "z_best": z, "best_radius": math.hypot(x, z)}


def validate_reference(raw):
    """Return a validated reference, or None for historical unreferenced sets."""
    if not isinstance(raw, dict):
        return None
    try:
        return make_reference(raw["x_best"], raw["z_best"])
    except (ValueError, TypeError, KeyError):
        return None


def reference_for_particle(particle_id, named_sets, active_set=None):
    """Resolve X/Z for a particle without arbitrary precedence among overlaps.

    If an active set is chosen, its reference applies only to its own members.
    Otherwise, use a reference if all overlapping referenced sets agree.  A
    genuine conflict is explicitly reported, not silently resolved using the
    insertion order of a dict.  Sets without a reference have no effect.

    Returns (reference | None, ambiguous: bool).
    """
    if active_set in named_sets:
        active = named_sets[active_set]
        if particle_id in active.get("particle_ids", ()):
            ref = validate_reference(active.get("best_focus"))
            if ref is not None:
                return ref, False
    matches = []
    for item in named_sets.values():
        if particle_id in item.get("particle_ids", ()):
            ref = validate_reference(item.get("best_focus"))
            if ref is not None:
                matches.append(ref)
    if not matches:
        return None, False
    unique = {(r["x_best"], r["z_best"]) for r in matches}
    if len(unique) > 1:
        return None, True
    return matches[0], False


def calibrated_radial_position(x, z, reference):
    """Signed radial difference in the original X/Z stage units."""
    return reference["best_radius"] - math.hypot(float(x), float(z))


def build_reference_index(named_sets, active_set=None):
    """Build an efficient particle-ID -> (reference, conflict) index.

    Used for repeatedly plotting thousands of particles.  Explicitly activated
    sets take precedence for their members; conflicting inactive references
    produce (None, True) rather than a guessed best-focus offset.
    """
    index = {}
    for name, entry in named_sets.items():
        reference = validate_reference(entry.get("best_focus"))
        if reference is None:
            continue
        for pid in entry.get("particle_ids", ()):
            current = index.get(pid)
            if current is None:
                index[pid] = (reference, False)
            elif current[0] is not None and (
                current[0]["x_best"], current[0]["z_best"]
            ) != (reference["x_best"], reference["z_best"]):
                index[pid] = (None, True)
    if active_set in named_sets:
        active = named_sets[active_set]
        ref = validate_reference(active.get("best_focus"))
        if ref is not None:
            for pid in active.get("particle_ids", ()):
                index[pid] = (ref, False)
    return index


def relative_stage_position(x, z, reference, axis):
    """Signed original stage coordinate minus the committed best-focus reading."""
    ref = validate_reference(reference)
    if ref is None or axis not in ("x", "z"):
        raise ValueError("Expected X/Z best-focus reference and axis 'x' or 'z'")
    raw = float(x if axis == "x" else z)
    return raw - ref["x_best" if axis == "x" else "z_best"]


def corrected_stage_radius(x, z, reference):
    """Unsigned distance from the committed best-focus X/Z stage *point*.

    This is distinct from the historical difference of two absolute radii,
    and is non-negative. Both quantities are retained for scientific review.
    """
    ref = validate_reference(reference)
    if ref is None:
        raise ValueError("No valid committed best-focus X/Z reference")
    return math.hypot(float(x) - ref["x_best"], float(z) - ref["z_best"])


def committed_particle_ids(named_sets):
    """Return particles assigned to named *calibration* sets, not review batches."""
    return {pid for entry in named_sets.values()
            if entry.get("description") != "Review batch: keep"
            and entry.get("description") != "Review batch: reject"
            for pid in entry.get("particle_ids", ())}


def materialized_reference_rows(named_sets, particle_records, normal_deg=None):
    """CSV-ready derived X/Z/radial coordinates for all saved, referenced sets.

    Overlapping named sets get distinct rows. Missing stage measurements are
    blank, and every saved particle ID is represented even if its MAT source
    is presently unavailable. Output is deterministically ordered.
    """
    records_by_id = {r.particle_id: r for r in particle_records}
    result = []
    for name, entry in sorted(named_sets.items()):
        ref = validate_reference(entry.get("best_focus"))
        if ref is None:
            continue
        for particle_id in sorted(set(entry.get("particle_ids", ()))):
            row = dict(set_name=name, particle_id=particle_id,
                       x_stage="", z_stage="", x_best=ref["x_best"],
                       z_best=ref["z_best"], x_minus_xbest="",
                       z_minus_zbest="", z_p_radial="", z_p_from_offsets="")
            if normal_deg is not None:
                row.update(plane_normal_deg=float(normal_deg), d_signed_plane="", d_in_plane="")
            rec = records_by_id.get(particle_id)
            if rec is not None:
                x, z = float(rec.stage_x), float(rec.stage_y)
                if math.isfinite(x) and math.isfinite(z):
                    row.update(x_stage=x, z_stage=z,
                               x_minus_xbest=relative_stage_position(x, z, ref, "x"),
                               z_minus_zbest=relative_stage_position(x, z, ref, "z"),
                               z_p_radial=calibrated_radial_position(x, z, ref),
                               z_p_from_offsets=corrected_stage_radius(x, z, ref))
                    if normal_deg is not None:
                        row.update(d_signed_plane=signed_plane_distance(x, z, ref, normal_deg),
                                   d_in_plane=in_plane_offset(x, z, ref, normal_deg))
            result.append(row)
    return result


# ---------------------------------------------------------------------------
# Signed distance from the object plane
# ---------------------------------------------------------------------------
# sqrt(dX² + dZ²) loses the side of the plane, and the historical radial
# difference hypot(Xbest, Zbest) − hypot(X, Z) measures a change in distance
# from the stage *origin*, not from the plane.  The geometrically correct,
# signed quantity is the projection of the stage offset from best focus onto
# the unit normal of the object plane, n = (cos θ, sin θ) in the stage X–Z frame:
#     d = (X − Xbest)·cos θ + (Z − Zbest)·sin θ
# d > 0 and d < 0 are opposite sides of the plane; d = 0 on it.  The offset
# *within* the plane, (−sin θ, cos θ)·Δ, should stay ~0 for a scan that moves
# straight through the plane, so it is a useful check of θ.


def signed_plane_distance(x, z, reference, normal_deg):
    """Signed distance (stage units) from the object plane through (Xbest, Zbest)."""
    ref = validate_reference(reference)
    if ref is None:
        raise ValueError("No valid committed best-focus X/Z reference")
    theta = math.radians(float(normal_deg))
    return (float(x) - ref["x_best"]) * math.cos(theta) + (float(z) - ref["z_best"]) * math.sin(theta)


def in_plane_offset(x, z, reference, normal_deg):
    """Component of the offset lying *in* the object plane (perpendicular to the normal)."""
    ref = validate_reference(reference)
    if ref is None:
        raise ValueError("No valid committed best-focus X/Z reference")
    theta = math.radians(float(normal_deg))
    return -(float(x) - ref["x_best"]) * math.sin(theta) + (float(z) - ref["z_best"]) * math.cos(theta)


def scan_direction_deg(xs, zs):
    """Direction of travel of a stage scan in the X–Z frame, in degrees.

    Principal axis of the distinct stage positions, oriented so that +X (or
    +Z for a purely vertical scan) is positive.  If the scan moves the
    particles straight through the object plane, this is the plane normal.
    Returns None when fewer than two distinct positions exist.
    """
    pts = {(round(float(x), 9), round(float(z), 9)) for x, z in zip(xs, zs)
           if math.isfinite(float(x)) and math.isfinite(float(z))}
    if len(pts) < 2:
        return None
    import numpy as np
    arr = np.array(sorted(pts))
    arr = arr - arr.mean(axis=0)
    _, _, vt = np.linalg.svd(arr, full_matrices=False)
    vx, vz = vt[0]
    if vx < -1e-12 or (abs(vx) <= 1e-12 and vz < 0):
        vx, vz = -vx, -vz
    return math.degrees(math.atan2(vz, vx))
