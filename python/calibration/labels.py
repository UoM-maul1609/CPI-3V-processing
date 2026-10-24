"""Shared particle labels (bead types, ice-crystal analogues …) with true sizes.

One library is shared by calibration beads and ice analogues: when a project
lives below ``CPI_cals``, the GUI uses ``CPI_cals/ReferenceLibrary/labels.json``
and stores microscopy images and annotations in the adjacent ``microscopy/``
folder. Explicitly selected libraries persist through GUI settings. A legacy
home-directory library is read for migration only; it is never silently moved.

Particles store only the label *name* (in the flags CSV).  Their true
dimensions (``actual_length_um`` / ``actual_width_um``) are taken from the
library whenever labels are applied, so editing a label's size once updates
every particle in every dataset that carries the label.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

LABEL_KINDS = ("bead", "ice crystal", "ice analogue", "droplet", "other")
PALETTE = ("#7048e8", "#e67700", "#0c8599", "#c2255c", "#5c940d", "#1971c2", "#9c36b5", "#d9480f",
           "#087f5b", "#495057")


def default_library_path() -> Path:
    override = os.environ.get("CPI_CALIBRATION_LABELS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cpi_calibration" / "labels.json"


def _positive_or_nan(value) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return float("nan")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Sizes must be positive numbers of micrometres")
    return number


@dataclass
class ParticleLabel:
    name: str
    kind: str = "bead"
    length_um: float = float("nan")   # true maximum dimension
    width_um: float = float("nan")    # unset unless explicitly supplied
    notes: str = ""
    colour: str = ""
    aliases: list = field(default_factory=list)  # former names, so renamed labels still resolve
    modified_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    area_um2: float = float("nan")  # independent SEM / microscope projected area

    def size_text(self) -> str:
        if not math.isfinite(self.length_um):
            return "size not set"
        if not math.isfinite(self.width_um):
            return f"{self.length_um:g} µm (width unset)"
        if abs(self.width_um - self.length_um) < 1e-9:
            return f"{self.length_um:g} µm"
        return f"{self.length_um:g} × {self.width_um:g} µm"

    def as_json(self) -> dict:
        data = asdict(self)
        for key in ("length_um", "width_um", "area_um2"):
            if not math.isfinite(data[key]):
                data[key] = None
        return data

    @classmethod
    def from_json(cls, data: dict) -> "ParticleLabel":
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("A label needs a name")
        length = _positive_or_nan(data.get("length_um"))
        width = _positive_or_nan(data.get("width_um"))
        area = _positive_or_nan(data.get("area_um2"))
        kind = str(data.get("kind") or "other")
        return cls(name=name, kind=kind if kind in LABEL_KINDS else "other", length_um=length,
                   width_um=width, area_um2=area, notes=str(data.get("notes", "")), colour=str(data.get("colour", "")),
                   aliases=[str(a) for a in data.get("aliases", []) if str(a).strip()],
                   modified_utc=str(data.get("modified_utc", "")))


class LabelLibrary:
    """Ordered collection of labels persisted as JSON."""

    def __init__(self, path: Optional[os.PathLike] = None):
        self.path = Path(path) if path is not None else default_library_path()
        self.labels: Dict[str, ParticleLabel] = {}

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[os.PathLike] = None) -> "LabelLibrary":
        library = cls(path)
        if library.path.exists():
            data = json.loads(library.path.read_text(encoding="utf-8"))
            for entry in data.get("labels", []):
                try:
                    label = ParticleLabel.from_json(entry)
                except (ValueError, TypeError):
                    continue
                library.labels[label.name] = label
        return library

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "labels": [label.as_json() for label in self.labels.values()]}
        fd, temporary = tempfile.mkstemp(prefix=".labels.", suffix=".tmp", dir=str(self.path.parent))
        os.close(fd)
        try:
            Path(temporary).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    # -- editing -------------------------------------------------------------
    def names(self):
        return list(self.labels)

    def get(self, name: str) -> Optional[ParticleLabel]:
        return self.labels.get(self.resolve(name))

    def resolve(self, name: str) -> str:
        """Current name for ``name``, following renames; unknown names are returned unchanged."""
        if name in self.labels:
            return name
        for label in self.labels.values():
            if name in label.aliases:
                return label.name
        return name

    def next_colour(self) -> str:
        used = {label.colour for label in self.labels.values()}
        return next((c for c in PALETTE if c not in used), PALETTE[len(self.labels) % len(PALETTE)])

    def upsert(self, label: ParticleLabel, old_name: Optional[str] = None) -> ParticleLabel:
        """Add or edit a label; ``old_name`` renames an existing label."""
        name = label.name.strip()
        if not name:
            raise ValueError("A label needs a name")
        if old_name and old_name != name and name in self.labels:
            raise ValueError(f"A label called '{name}' already exists")
        if not label.colour:
            previous = self.labels.get(old_name or name)
            label.colour = previous.colour if previous and previous.colour else self.next_colour()
        label.name = name
        if old_name and old_name != name:
            previous = self.labels.get(old_name)
            label.aliases = sorted(set((previous.aliases if previous else []) + label.aliases + [old_name]) - {name})
        label.modified_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if old_name and old_name != name:
            # Preserve order when renaming.
            self.labels = {(name if k == old_name else k): (label if k == old_name else v)
                           for k, v in self.labels.items()}
        else:
            self.labels[name] = label
        return label

    def delete(self, name: str) -> None:
        self.labels.pop(name, None)

    def merge(self, entries: Iterable[dict]) -> list[str]:
        """Add labels from a project snapshot that this library lacks.

        Existing library entries win: the library is the single source of truth
        for dimensions.  Returns the names that were added.
        """
        added = []
        for entry in entries or ():
            try:
                label = ParticleLabel.from_json(entry)
            except (ValueError, TypeError):
                continue
            if label.name not in self.labels:
                self.labels[label.name] = label
                added.append(label.name)
        return added


def apply_label_dimensions(records, library: LabelLibrary, names=None) -> int:
    """Copy true sizes from the library onto labelled particle records.

    Only records whose label is in ``names`` (default: all labels) are changed;
    unlabelled records keep any size imported from a truth spreadsheet.
    Returns the number of records updated.
    """
    wanted = set(names) if names is not None else None
    changed = 0
    for record in records:
        name = getattr(record, "label", "")
        if not name or (wanted is not None and name not in wanted):
            continue
        label = library.get(name)
        if label is None:
            continue
        record.label = label.name  # follow renames
        record.actual_length_um = label.length_um
        record.actual_width_um = label.width_um
        changed += 1
    return changed
