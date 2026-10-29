"""Decode CPI housekeeping blocks with bounds-checked, version-specific layouts.

The version 20/25/30 layouts are documented by CPIView's HKverXX structs.
Scanner candidates can include markers within image pixels, so the declared
record size, version, and alignment must be checked before decoding.
"""
import warnings
import numpy as np


def _u32_from_words(high, low):
    return (int(high) << 16) | int(low)


def convertDataToHouseSA(bytes1, ushort, order, house, cpiv1):
    """Decode selected packet family; caller detects the family independent of CPIv1."""
    if cpiv1:
        # CPIView HKver20 has an 11-word header and 38 readings (98 bytes).
        # HKver25 has 13 header words and 40 readings (106 bytes).
        # CPIv1 A1D7 version 30 has 13 header words and 40 readings (106 bytes).
        layouts = {20: (98, 11, 38), 25: (106, 13, 40), 30: (118, 13, 46)}
        dtype = [
            ('BlockNum', 'uint16'), ('ulItemSize', 'uint32'),
            ('usVersion', 'uint16'), ('ulXtraData', 'uint32'),
            ('bNewData', 'uint16'), ('usChkSumFlag', 'uint16'),
            ('TopReadings', '(3,1)uint16'), ('Time', 'uint32'),
            ('Readings', '(46,1)uint16'), ('nReadings', 'uint16'),
            ('order', 'uint32'),
        ]
        records = []
        rejected = 0
        # ROIDataDriver concatenates normal and byte-shifted word arrays.
        # Do not let a purported housekeeping record cross their boundary.
        half = len(ushort) // 2
        for candidate in house:
            pos = int(candidate)
            boundary = half if pos < half else len(ushort)
            if pos + 4 > boundary:
                rejected += 1
                continue
            marker = int(ushort[pos])
            if marker not in (0xA1D6, 0xA2D6, 0xA1D7, 0xA1D8):
                rejected += 1
                continue
            ver = int(ushort[pos + 3])
            layout = layouts.get(ver)
            if layout is None:
                rejected += 1
                continue
            item_size, header_words, nreadings = layout
            if pos + header_words + nreadings > boundary:
                rejected += 1
                continue
            # CPIView ItemSize is a 32-bit byte length for versions 20, 25, 30.
            declared_size = _u32_from_words(ushort[pos + 2], ushort[pos + 1])
            if declared_size != item_size:
                rejected += 1
                continue
            rec = np.zeros((), dtype=dtype)
            rec['BlockNum'] = marker
            rec['ulItemSize'] = declared_size
            rec['usVersion'] = ver
            if ver == 20:
                rec['bNewData'] = ushort[pos + 4]
                rec['usChkSumFlag'] = ushort[pos + 5]
                rec['TopReadings'] = ushort[pos + 6:pos + 9].reshape(3, 1)
                rec['Time'] = _u32_from_words(ushort[pos + 10], ushort[pos + 9])
            else:
                rec['ulXtraData'] = _u32_from_words(ushort[pos + 5], ushort[pos + 4])
                rec['bNewData'] = ushort[pos + 6]
                rec['usChkSumFlag'] = ushort[pos + 7]
                rec['TopReadings'] = ushort[pos + 8:pos + 11].reshape(3, 1)
                rec['Time'] = _u32_from_words(ushort[pos + 12], ushort[pos + 11])
            rec['Readings'][:nreadings] = ushort[pos + header_words:pos + header_words + nreadings].reshape(nreadings, 1)
            rec['nReadings'] = nreadings
            rec['order'] = order[pos]
            records.append(rec)
        if rejected:
            warnings.warn(
                f"CPIv1 housekeeping: skipped {rejected} invalid/incomplete marker "
                f"candidate(s); retained {len(records)} valid record(s)",
                RuntimeWarning, stacklevel=2,
            )
        return np.array(records, dtype=dtype)

    dtype = [
        ('BlockNum', 'uint16'), ('ulItemSize', 'uint16'),
        ('Readings1', '(70,1)uint16'), ('TimeMSW', 'uint16'),
        ('TimeISW', 'uint16'), ('TimeLSW', 'uint16'),
        ('Readings', '(8,1)int16'), ('order', 'uint32'),
    ]
    recs = []
    half = len(ushort) // 2
    for candidate in house:
        pos = int(candidate)
        boundary = half if pos < half else len(ushort)
        if pos + 83 > boundary or int(ushort[pos + 1]) != 83:
            continue
        row = np.zeros((), dtype=dtype)
        row['BlockNum'] = ushort[pos]
        row['ulItemSize'] = ushort[pos + 1]
        row['Readings1'] = ushort[pos + 2:pos + 72].reshape(70, 1)
        row['TimeMSW'] = ushort[pos + 72]
        row['TimeISW'] = ushort[pos + 73]
        row['TimeLSW'] = ushort[pos + 74]
        row['Readings'] = ushort[pos + 75:pos + 83].reshape(8, 1)
        row['order'] = order[pos]
        recs.append(row)
    return np.array(recs, dtype=dtype)
