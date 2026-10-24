"""Non-persistent CPI edge-method experiments on a saved batch.

Reads existing MAT ROIs and their associated BG arrays and calls the *same*
transmittance detector used by batch processing.  This module never writes files,
changes ParticleRecord objects, or edits acceptance/batch state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace, dataclass
from pathlib import Path
import shlex
from typing import Callable, Iterable

import numpy as np
import scipy.io as sio

try:
    from ..proc.edgeDetect import EdgeOptions, measure_transmittance, transmittance
except ImportError:
    try:
        from proc.edgeDetect import EdgeOptions, measure_transmittance, transmittance
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from proc.edgeDetect import EdgeOptions, measure_transmittance, transmittance

try:
    from .data import _struct_list
except ImportError:
    from data import _struct_list


METHODS = (
    ("relative/radial", "Relative shadow · radial outline"),
    ("relative/edge", "Relative shadow · steepest gradient"),
    ("relative/contour", "Relative shadow · raw contour"),
    ("fixed/radial", "Absolute transmittance · radial outline"),
    ("fixed/edge", "Absolute transmittance · steepest gradient"),
    ("fixed/contour", "Absolute transmittance · raw contour"),
)


SHADOW_METHOD = ("shadow/circle", "Signed shadow integral · fitted circle (experimental)")

def preview_options(method: str, level: float = .55, denoise: int = 1,
                    smooth: float = 0., *, pixel_um: float = 2.3, cpiview: bool = False) -> EdgeOptions:
    """Return real production processing options with the requested edge definition."""
    if method not in dict(METHODS):
        raise ValueError(f"Unknown edge preview algorithm: {method}")
    if method == "shadow/circle":
        raise ValueError("Use shadow_options() for the experimental integral")
    mode, outline = method.split("/")
    options = EdgeOptions(method="transmittance", pixel_um=float(pixel_um),
                          edge_mode=mode, outline=outline,
                          edge_fraction=float(level), edge_level=float(level),
                          denoise=int(denoise), smooth_sigma=float(smooth),
                          bg_check=False, boundary_tol_px=.35, max_boundary_points=500, cpiview=bool(cpiview))
    return options


def command_flags(method: str, level: float = .55, denoise: int = 1,
                  smooth: float = 0., cpiview: bool = False) -> str:
    """Actual batchProcess.py CLI flags corresponding to the visible trial controls.

    The preview uses stored backgrounds and disables background replacement;
    full processing can differ when its background-check stage replaces a frame.
    """
    if method == "shadow/circle":
        raise ValueError("Signed-shadow integration is experimental; no production CLI flags exist")
    if method not in dict(METHODS):
        raise ValueError(f"Unknown edge preview algorithm: {method}")
    mode, outline = method.split("/")
    args = ["--edge-method", "transmittance", "--edge-mode", mode,
            "--outline", outline,
            "--edge-fraction" if mode == "relative" else "--edge-level", f"{level:g}",
            "--denoise", str(int(denoise)), "--smooth-sigma", f"{smooth:g}"]
    if cpiview:
        args.append("--cpiview")
    return " ".join(shlex.quote(arg) for arg in args)



@dataclass(frozen=True)
class ShadowOptions:
    """Experimental read-only signed-shadow integration around a stored-outline circle.

    The outline is used ONLY to position/size the aperture; the returned size
    comes entirely from the signed transmittance integral, not the outline area.
    """
    padding_px: float = 8.0
    denoise: int = 1
    smooth_sigma: float = 0.0
    pixel_um: float = 2.3
    radius_mode: str = "fixed"
    max_padding_px: float = 30.


def shadow_options(padding_px=8.0, denoise=1, smooth=0., *, pixel_um=2.3,
                   radius_mode="fixed", max_padding_px=30.):
    if not 0 <= float(padding_px) <= 30:
        raise ValueError("Aperture padding must be within 0–30 pixels")
    if radius_mode not in ("fixed", "adaptive"):
        raise ValueError("Unknown circle mode")
    return ShadowOptions(float(padding_px), int(denoise), float(smooth), float(pixel_um),
                         radius_mode, float(max_padding_px))


def trial_focus_value(record, result, metric):
    """Edge Lab must not refit/recompute *production* focus when testing size.

    Focus and focus_T are always the saved CPI measurements, independent of
    aperture, edge threshold, and whether a shadow measurement succeeded.
    The optional CPIview focus uses the trial's new value only when explicitly
    computed (otherwise use the stored CPIview value).
    """
    if metric in ("focus", "focus_T"):
        return float(getattr(record, metric, float("nan")))
    if result is not None and metric in ("cpv_focus", "cpv_fwhm_px"):
        val = result.get(metric)
        if val is not None:
            return float(val)
    return float(getattr(record, metric, float("nan")))


def fit_stored_circle(boundary):
    """Use the same centred enclosing aperture as production processing."""
    try:
        from proc.shadowCircle import fit_circle
    except ImportError:
        from ..proc.shadowCircle import fit_circle
    return fit_circle(boundary)


def _circle_mask(shape, centre, radius, supersample=4):
    """Fractional pixel coverage of the circle, avoiding 1px-size jumps."""
    h, w = shape
    cy, cx = centre
    y0 = max(0, int(np.floor(cy - radius - 1)))
    y1 = min(h, int(np.ceil(cy + radius + 2)))
    x0 = max(0, int(np.floor(cx - radius - 1)))
    x1 = min(w, int(np.ceil(cx + radius + 2)))
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Circle falls outside the ROI")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cov = np.zeros_like(yy, dtype=float)
    for j in range(supersample):
        for i in range(supersample):
            dy = (j + .5) / supersample - .5
            dx = (i + .5) / supersample - .5
            cov += ((yy + dy - cy) ** 2 + (xx + dx - cx) ** 2 <= radius ** 2)
    return (slice(y0, y1), slice(x0, x1)), cov / (supersample ** 2)


def shadow_integral_one(image, background, boundary, options: ShadowOptions):
    """Shadow-equivalent diameter from integral of (1-T) inside fitted circle.

    Dark and bright regions retain their positive and negative contributions.
    No empirical size calibration is implied. Invalid/truncated apertures
    return an explicit error instead of a silently-biased size.
    """
    try:
        from ..proc.shadowCircle import measure
    except ImportError:
        from proc.shadowCircle import measure
    out = measure(image, background, boundary, padding_px=options.padding_px,
                  pixel_um=options.pixel_um, denoise=options.denoise,
                  smooth_sigma=options.smooth_sigma, radius_mode=options.radius_mode,
                  max_padding_px=options.max_padding_px)
    out.update(deq=out["shadow_diameter"], len=float("nan"), wid=float("nan"),
               focus=float("nan"), focus_T=float("nan"), cutoff_pct=0.,
               edge_level_used=float("nan"))
    return out


def preview_one(image, background, options: EdgeOptions):
    """Run an ROI without modifying inputs. Failure to detect is represented explicitly."""
    im, bg = np.asarray(image), np.asarray(background)
    if im.ndim != 2 or im.shape != bg.shape or min(im.shape) < 3:
        return None
    result = measure_transmittance(im, bg, options)
    if result is None:
        return None
    out = {key: result[key] for key in (
        "len", "wid", "deq", "round", "focus", "focus_T", "T_min",
        "edge_level_used", "cutoff_pct", "edge_touch", "boundary", "T_border", "T_centre")}
    if options.cpiview:
        # Match imageStats.py: use the measured ROI polygon to choose the CPIview
        # component, then run the *separate*, native CPIview focus implementation.
        from skimage.draw import polygon
        try:
            from ..proc.cpiviewMetrics import cpiview_metrics
        except ImportError:
            from proc.cpiviewMetrics import cpiview_metrics
        b = np.asarray(result["boundary"], float)
        rr, cc = polygon(b[:, 0], b[:, 1], im.shape)
        mask = np.zeros(im.shape, dtype=bool)
        mask[rr, cc] = True
        out.update(cpiview_metrics(im, bg, mask))
    return out


def _effective_background_entries(raw, n):
    """BG_USED stores only replacements; scipy simplify_cells squeezes n=1."""
    value = raw.get("BG_USED")
    if value is None:
        return [None] * n
    if n == 1:
        return [value]
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 1 and arr.size == n:
        return list(arr)
    return [None] * n


def load_roi_for_preview(root, record):
    """Load the same ROI/effective BG as processing, including CPI 1.0 strip fix."""
    raw = sio.loadmat(str(Path(root) / record.source_file),
                      variable_names=["ROI_N", "BG", "BG_USED"], simplify_cells=True)
    roi = raw["ROI_N"]
    n = max(record.roi_index + 1, len(np.atleast_1d(roi.get("imageType", []))))
    images = _struct_list(roi.get("IMAGE"), n)
    bgs = _struct_list(raw.get("BG"), n)
    i = record.roi_index
    image = np.asarray(images[i]["IM"])
    background = np.asarray(bgs[i]["BG"])
    replacement = _effective_background_entries(raw, n)[i]
    if replacement is not None:
        candidate = np.asarray(replacement)
        if candidate.ndim == 2 and candidate.size and candidate.shape == image.shape:
            background = candidate
    try:
        image_type = int(np.asarray(roi.get("imageType")).ravel()[i])
        start_x = int(np.asarray(roi.get("StartX")).ravel()[i])
    except (ValueError, IndexError, TypeError, AttributeError):
        image_type, start_x = 0, 0
    if image_type == 33857:
        try:
            from ..proc.imageStats import _remove_centre_strip
        except ImportError:
            from proc.imageStats import _remove_centre_strip
        image = _remove_centre_strip(image.copy(), start_x)
        background = _remove_centre_strip(background.copy(), start_x)
    return image, background


def run_preview(root: Path, records: Iterable, options: EdgeOptions,
                progress: Callable[[int, int], None] | None = None,
                cancelled: Callable[[], bool] | None = None):
    """Evaluate selected records, reading each MAT file once, reporting per-ROI errors.

    Results are keyed by stable ParticleRecord.particle_id; unsupported/missing
    backgrounds are *not* replaced with synthetic ones.  The input records are
    read-only.  No output on disk and no calls to any project saving machinery.
    """
    grouped = defaultdict(list)
    for record in records:
        grouped[record.source_file].append(record)
    total = sum(len(group) for group in grouped.values())
    completed = 0
    results, failures = {}, {}
    for source, group in grouped.items():
        if cancelled and cancelled():
            break
        try:
            raw = sio.loadmat(str(Path(root) / source), variable_names=["ROI_N", "BG", "BG_USED", "dat"]
                              if isinstance(options, ShadowOptions) else ["ROI_N", "BG", "BG_USED"],
                              simplify_cells=True)
            roi = raw["ROI_N"]
            images = _struct_list(roi.get("IMAGE"), max(r.roi_index for r in group) + 1)
            bgs = _struct_list(raw.get("BG"), len(images))
            bg_used = raw.get("BG_USED")
            bg_used = _effective_background_entries(raw, len(images))
            stored = (_struct_list(raw.get("dat", {}).get("foc") if isinstance(raw.get("dat"), dict)
                                   else None, len(images)) if isinstance(options, ShadowOptions) else None)
        except Exception as exc:
            for record in group:
                failures[record.particle_id] = f"MAT load failed: {exc}"
            completed += len(group)
            if progress:
                progress(completed, total)
            continue
        for record in group:
            if cancelled and cancelled():
                break
            pid, i = record.particle_id, record.roi_index
            try:
                image = np.asarray(images[i]["IM"])
                background = np.asarray(bgs[i]["BG"])
                if bg_used is not None and i < len(bg_used):
                    candidate = np.asarray(bg_used[i])
                    if candidate.ndim == 2 and candidate.size and candidate.shape == image.shape:
                        background = candidate
                # Match the CPI 1.0 centre-strip preparation of imageStats.
                try:
                    image_type = int(np.asarray(roi.get("imageType")).ravel()[i])
                    start_x = int(np.asarray(roi.get("StartX")).ravel()[i])
                except (ValueError, IndexError, TypeError, AttributeError):
                    image_type, start_x = 0, 0
                if image_type == 33857:
                    try:
                        from ..proc.imageStats import _remove_centre_strip
                    except ImportError:
                        from proc.imageStats import _remove_centre_strip
                    image = _remove_centre_strip(image.copy(), start_x)
                    background = _remove_centre_strip(background.copy(), start_x)
                local_options = replace(options, pixel_um=float(record.pixel_um))
                if isinstance(options, ShadowOptions):
                    result = shadow_integral_one(image, background,
                                                  stored[i].get("boundaries") if stored else None,
                                                  local_options)
                else:
                    result = preview_one(image, background, local_options)
                if result is None:
                    failures[pid] = "No detectable edge / invalid ROI or background"
                else:
                    result["image_shape"] = tuple(image.shape)
                    results[pid] = result
            except (KeyError, IndexError, ValueError, TypeError, FloatingPointError) as exc:
                failures[pid] = f"Could not process ROI: {exc}"
            except Exception as exc:  # other individual ROIs must still be analysed
                failures[pid] = f"ROI processing failed: {type(exc).__name__}: {exc}"
            completed += 1
            if progress and (completed % 10 == 0 or completed == total):
                progress(completed, total)
        del raw, images, bgs
    return results, failures


def transmittance_profile(image, background, options: EdgeOptions, boundary=None):
    """Diagnostic central horizontal/vertical profiles, never persisted."""
    T = transmittance(np.asarray(image), np.asarray(background), options)
    if boundary is not None and np.asarray(boundary).size:
        centre = np.nanmean(np.asarray(boundary, float), axis=0)
    else:
        centre = (np.asarray(T.shape) - 1) / 2
    cy, cx = np.clip(np.rint(centre).astype(int), [0, 0], np.asarray(T.shape) - 1)
    return {"horizontal": T[cy, :].copy(), "vertical": T[:, cx].copy(),
            "centre": (int(cy), int(cx)), "border_median": float(np.median(np.r_[T[0], T[-1], T[:, 0], T[:, -1]]))}
