"""Export-time intensity normalisation using CPI processing's own transmittance."""
import json
import numpy as np
try:
    from edgeDetect import EdgeOptions, transmittance
except ImportError:
    from .edgeDetect import EdgeOptions, transmittance


def unwrap(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.flat[0]
    return np.asarray(value)


def options_from_dat(dat):
    opts = EdgeOptions(method='transmittance')
    try:
        raw = unwrap(dat['proc_options'][0, 0]).item()
        settings = json.loads(str(raw))
        for key in ('border_normalise', 'denoise', 'smooth_sigma'):
            if key in settings:
                setattr(opts, key, settings[key])
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        pass
    return opts


def corrected_roi(roi, bg, bg_used, index, options, *, start_x=None, cpiv1=False):
    """Return background-corrected 8-bit ROI; fail rather than silently export raw."""
    image = unwrap(roi['IM']).astype(np.float64)
    background = unwrap(bg[0, index]['BG']).astype(np.float64)
    if bg_used is not None:
        replacement = unwrap(bg_used[0, index])
        if replacement.size:
            background = replacement.astype(np.float64)
    if cpiv1:
        if start_x is None:
            raise ValueError('CPI v1 requires StartX for centre-strip correction')
        try:
            from imageStats import _remove_centre_strip
        except ImportError:
            from .imageStats import _remove_centre_strip
        image = _remove_centre_strip(image, int(start_x))
        if background.shape != image.shape:
            background = _remove_centre_strip(background, int(start_x))
    if image.ndim != 2 or image.shape != background.shape:
        raise ValueError(f'ROI/background shape mismatch: {image.shape} vs {background.shape}')
    T = transmittance(image, background, options)
    if not np.isfinite(T).all():
        raise ValueError('Nonfinite background-normalised transmittance')
    return np.clip(np.rint(255.0 * T), 0, 255).astype(np.uint8)
