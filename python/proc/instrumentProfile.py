"""Conservative per-file instrument mode from parsed image-type records.

The instrument's optional centre-strip correction is NOT inferable from the
bit mask alone; legacy encoding selects the established legacy correction,
with an explicit warning. Review optical calibration before science use.
"""
import numpy as np

LEGACY_PARTICLE = 33857
MODERN_PARTICLE = 89

def from_image_types(types):
    a = np.asarray(types, dtype=np.int64).ravel()
    old = int(np.count_nonzero(a == LEGACY_PARTICLE))
    new = int(np.count_nonzero(a == MODERN_PARTICLE))
    if old and new:
        raise ValueError('Mixed legacy and modern particle type records: inspect file manually')
    if not (old or new):
        # A background-only file or empty image list cannot establish particle mode.
        raise ValueError('No recognised particle type (33857 or 89): cannot infer instrument profile')
    return bool(old)

def from_mat_roi(roi):
    return from_image_types(roi['imageType'][0, 0])

def from_mat_data(data):
    if 'dat' in data and 'imageType' in (data['dat'].dtype.names or ()):
        return from_image_types(data['dat']['imageType'][0,0])
    return from_mat_roi(data['ROI_N'])
