"""Extract real CPI full-frame backgrounds independent of the particle ROI table.

CPIview scan_image_block.pro: ImageType bit 0 = image present,
bit 1 = background, bit 2 = full frame.  read_backgnd.pro reads a
frame header followed by an image/ROI header and its pixel payload.
Only well-formed, bounded, full-sensor records are accepted here.
"""
from __future__ import annotations

import struct
import numpy as np

_FRAME = b'\xd5\xa3'
_ROI = b'\xe6\xb2'
_FRAME_HEADER_SIZE = 104  # frameheader25 (also used for version 40)
_ROI_HEADER_SIZE = 54      # imageheader25


def scan_recorded_backgrounds(raw_bytes, frame_times=None, *, max_pixels=20_000_000):
    """Return (background list, inventory) from validated A3D5 frame blocks.

    Each returned item is (frame marker byte offset, image ndarray, time).
    Coordinates are in camera X,Y order, matching existing ROI_N['IMAGE']['IM'].
    frame_times, when provided, maps original frame byte offset to MATLAB datenums.
    No synthetic backgrounds are produced.
    """
    result = []
    stats = {'frames': 0, 'background_flags': 0, 'decoded': 0,
             'invalid_backgrounds': 0, 'types': {}, 'versions': {}, 'bg_rates': {}}
    pos = 0
    total = len(raw_bytes)
    while True:
        at = raw_bytes.find(_FRAME, pos)
        if at < 0:
            break
        pos = at + 2
        if at + _FRAME_HEADER_SIZE > total:
            continue
        frame_size, ver, nroi, roi_total = struct.unpack_from('<IHHI', raw_bytes, at+2)
        day, hour, minute, second = struct.unpack_from('<BBBB', raw_bytes, at+14)
        img_type = struct.unpack_from('<H', raw_bytes, at+20)[0]
        bg_rate = struct.unpack_from('<H', raw_bytes, at+30)[0]
        if not (ver in (25, 40) and frame_size == _FRAME_HEADER_SIZE
                and 1 <= day <= 31 and hour < 24 and minute < 60 and second < 60
                and at + frame_size + roi_total <= total and nroi <= 65535):
            continue
        # Enforce record-chain boundary for frames carrying ROI data.  This
        # excludes chance A3D5 sequences within compressed/pixel payloads.
        first_roi = at + frame_size
        if nroi and (roi_total < _ROI_HEADER_SIZE or
                     raw_bytes[first_roi:first_roi+2] != _ROI):
            continue
        stats['frames'] += 1
        for field, val in [('types', img_type), ('versions', ver), ('bg_rates', bg_rate)]:
            d = stats[field]
            d[val] = d.get(val, 0) + 1
        if img_type & 3 != 3:
            continue
        stats['background_flags'] += 1
        if not (img_type & 4 and nroi == 1 and
                raw_bytes[first_roi:first_roi+2] == _ROI):
            stats['invalid_backgrounds'] += 1
            continue
        roi_size, roi_ver = struct.unpack_from('<IH', raw_bytes, first_roi+2)
        sx, sy, ex, ey = struct.unpack_from('<HHHH', raw_bytes, first_roi+8)
        pixel_bytes = struct.unpack_from('<h', raw_bytes, first_roi+16)[0]
        w, h = ex-sx+1, ey-sy+1
        payload_length = w*h*abs(pixel_bytes)
        if not (roi_ver == 25 and 0 <= sx <= ex < 10000 and
                0 <= sy <= ey < 10000 and abs(pixel_bytes) in (1, 2, 4) and
                0 < w*h <= max_pixels and
                roi_size >= _ROI_HEADER_SIZE+payload_length and
                roi_size <= roi_total and
                first_roi+_ROI_HEADER_SIZE+payload_length <= total):
            stats['invalid_backgrounds'] += 1
            continue
        dtype = {1:'u1', 2:'<i2', 4:'<i4'}[abs(pixel_bytes)]
        image = np.frombuffer(raw_bytes, dtype=dtype, count=w*h,
                              offset=first_roi+_ROI_HEADER_SIZE).reshape(h,w).T.copy()
        stamp = None if frame_times is None else frame_times.get(at)
        result.append((at, image, stamp))
        stats['decoded'] += 1
    return result, stats


def image_times_by_offset(raw_bytes, images, times):
    """Resolve the reader's two-alignment order mapping to physical byte offsets."""
    mapping = {}
    for order, when in zip(images['order'], np.asarray(times).ravel()):
        n = int(order)
        for at in (2*n, 2*n-1):
            if 0 <= at <= len(raw_bytes)-2 and raw_bytes[at:at+2] == _FRAME:
                mapping[at] = float(when)
                break
    return mapping
