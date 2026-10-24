"""Portable, shared CPI reference-library location and safe migration.

The reference folder is a sibling of CPICalibration and ICEAnalogues. A
settings file may remember the choice, but the *labels and microscopy images*
remain beside the projects, not in the user's home settings directory.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def cpi_library_path(source) -> Path | None:
    """Return CPI_cals/ReferenceLibrary/labels.json for a nested CPI source.

    Works for a project JSON, MAT file, CPI subdirectory, or CPI_cals itself;
    case-insensitive on the directory name but preserves its actual spelling.
    """
    p = Path(source).expanduser().resolve()
    for parent in (p, *p.parents):
        if parent.name.casefold() == 'cpi_cals' and parent.is_dir():
            return parent / 'ReferenceLibrary' / 'labels.json'
    return None


def copy_reference_library(source, destination):
    """Copy labels and microscopy as one unit, never overwrite any target.

    The caller must obtain consent if the source holds user data. This does
    *not* remove the source: a successful migration is reversible.
    """
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if source.resolve() == destination.resolve():
        return
    if destination.exists() or (destination.parent / 'microscopy').exists():
        raise FileExistsError('Reference library destination already contains data')
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied = []
    try:
        if source.is_file():
            shutil.copy2(source, destination)
            copied.append(destination)
        if (source.parent / 'microscopy').exists():
            # Register the destination before copying so interrupted copying is
            # cleaned up as well. The target was checked absent above.
            copied.append(destination.parent / 'microscopy')
            shutil.copytree(source.parent / 'microscopy', destination.parent / 'microscopy')
    except Exception:
        # Do not leave a half-copied library that might later be accepted.
        for p in reversed(copied):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
        raise
