"""Discover historical and 3V-CPI housekeeping independently of image mode.

Validate size/version and word alignment; do not infer packet type from --cpiv1.
Returns (candidate positions, is_legacy). A single packet family is selected
per ROI file to preserve the existing homogeneous H dtype.
"""
import numpy as np

_LEGACY_MARKERS = (0xA1D6, 0xA2D6, 0xA1D7, 0xA1D8)
_LEGACY_LAYOUT = {20: (98,), 25: (106,), 30: (106, 118)}


def detect_housekeeping(words, order):
    half = len(words) // 2
    modern = []
    legacy = []
    for pos in np.flatnonzero(words == 0x484B):
        p = int(pos)
        end = half if p < half else len(words)
        if p + 83 <= end and int(words[p+1]) == 83:
            modern.append(p)
    for pos in np.flatnonzero(np.isin(words, _LEGACY_MARKERS)):
        p = int(pos)
        end = half if p < half else len(words)
        if p + 4 > end:
            continue
        version = int(words[p+3])
        size = int(words[p+1]) | (int(words[p+2]) << 16)
        if size in _LEGACY_LAYOUT.get(version, ()) and p + size//2 <= end:
            legacy.append(p)
    # Keep a single, highest-confidence family. No valid HK: empty modern
    # so the postprocessor safely produces empty arrays in either mode.
    if modern and legacy:
        # A multi-packet run is stronger evidence than isolated marker collisions.
        if len(legacy) > len(modern):
            return np.array(legacy, dtype=np.int64), True
    if modern:
        return np.array(modern, dtype=np.int64), False
    if legacy:
        return np.array(legacy, dtype=np.int64), True
    return np.array([], dtype=np.int64), False
