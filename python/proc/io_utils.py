"""Small I/O helpers shared by CPI processing drivers."""

from __future__ import annotations

import os
import tempfile

import scipy.io as sio


def normalise_directory(path):
    """Return an absolute, expanded directory path without relying on cwd."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def parse_bool(value):
    """Parse common command-line boolean spellings without using eval()."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError("Expected a boolean value, got {!r}".format(value))


def atomic_savemat(filename, mapping, **kwargs):
    """Write a MAT file atomically in the destination directory.

    scipy writes to a temporary file first; os.replace only exposes the new
    file after the write completed successfully.  This protects an existing
    result if a disposable worker is killed mid-write.
    """
    filename = os.path.abspath(os.fspath(filename))
    directory = os.path.dirname(filename) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(filename) + ".", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        sio.savemat(tmp, mapping, appendmat=False, **kwargs)
        os.replace(tmp, filename)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
