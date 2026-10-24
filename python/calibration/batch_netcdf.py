"""Lossless, read-only export of reviewed CPI batches to a ragged NetCDF file.

Images are NOT resampled or processed: `roi_pixels` and `background_pixels` are
flattened C-order arrays, with one start/count/shape per particle. The original
ROI indexing is [camera X, camera Y]. Positions are the stage values in mm;
true length comes from the current label library, not image segmentation.

When netCDF4 is installed, output is compressed NETCDF4. Without it, scipy's
NetCDF3 64-bit-offset writer is used for modest exports (same schema).
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat, netcdf_file



class ExportCancelled(Exception):
    """Requested by a GUI user; never leaves a partial destination file."""


def _image(data, *, description):
    a = np.asarray(data)
    if a.ndim != 2 or not a.size:
        raise ValueError(f"{description} must be a non-empty 2-D image")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{description} should contain raw integer camera intensities (found {a.dtype})")
    if a.min() < 0 or a.max() > 32767:
        raise ValueError(f"{description} has values outside the lossless signed-16-bit range")
    return a.astype('<i2', copy=False)


def _mat_struct_field(value, name):
    """Read a field from an unsqueezed scipy MAT struct or scalar struct.

    CPI files have two layouts: older/tests store arrays of MATLAB structs,
    while proc/postProcess.py writes lists of dicts, which savemat serialises
    as *cell arrays of 1x1 structs*.  Keep squeeze_me=False to retain images
    with a single row or column and unwrap only the struct/cell wrappers.
    """
    while isinstance(value, np.ndarray) and value.dtype.kind == 'O' and value.size == 1:
        value = value.flat[0]
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    if (isinstance(value, np.ndarray) and value.dtype.names and
            name in value.dtype.names and value.size == 1):
        return value[name].flat[0]
    if isinstance(value, dict):
        return value.get(name)
    return None


def _mat_struct_entries(value):
    """Return sequence of MATLAB structs, including cell-wrapped structs."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value.flat)
    return [value]


def _mat_roi_image(entry, field, description):
    value = _mat_struct_field(entry, field)
    if value is None:
        raise ValueError(f'Missing {description}: expected struct field {field}')
    return _image(value, description=description)


def _finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _label_size(record, labels):
    if record.label and labels is not None:
        entry = labels.get(record.label)
        if entry is not None:
            return _finite(entry.length_um), _finite(entry.width_um)
    return _finite(record.actual_length_um), _finite(record.actual_width_um)


def _label_area(record, labels):
    """Independent projected area; never infer it from CPI ROI segmentation."""
    if record.label and labels is not None:
        entry = labels.get(record.label)
        if entry is not None:
            return _finite(getattr(entry, 'area_um2', np.nan))
    return np.nan


def _shape_shadow(record):
    """Per-particle shape-corrected signed-shadow diameter (not a SEM size)."""
    shadow = _finite(getattr(record, 'shadow_deq_um', np.nan))
    ratio = _finite(getattr(record, 'area_ratio_circle', np.nan))
    return shadow / math.sqrt(ratio) if shadow > 0 and 0 < ratio <= 1 else np.nan


def _put_strings(nc, key, values):
    """Portable UTF-8 char array (works in both NetCDF3 and NetCDF4)."""
    encoded = [str(value or '').encode('utf-8') for value in values]
    length = max(1, *(len(b) for b in encoded))
    nc.createDimension(f'{key}_bytes', length)
    v = nc.createVariable(key, 'S1', ('particle', f'{key}_bytes'))
    v.encoding = 'UTF-8; zero-padded; decode until first NUL byte'
    out = np.zeros((len(values), length), dtype='S1')
    for i, blob in enumerate(encoded):
        if blob:
            out[i, :len(blob)] = np.frombuffer(blob, dtype='S1')
    v[:] = out


def _create_numeric(nc, name, dims, data, *, units=None, description=None):
    arr = np.asarray(data)
    dtype = 'i4' if arr.dtype.kind in 'iu' else 'f8'
    v = nc.createVariable(name, dtype, dims, zlib=True, complevel=3) if hasattr(nc, 'data_model') else \
        nc.createVariable(name, dtype, dims)
    v[:] = arr.astype(np.int32 if dtype == 'i4' else np.float64)
    if units:
        v.units = units
    if description:
        v.long_name = description
    return v




def batch_position_metadata(records, memberships, saved_sets, plane_normal_deg, fallback_reference):
    """Resolve signed stage/object-plane positions without using an editor draft.

    An overlapping particle assigned different best-focus points in selected
    batches has no unambiguous d, so is exported with NaN (not a chosen side).
    `fallback_reference(record)` returns (reference, source, kind) using saved
    project references; it is called only where the selected batch lacks one.
    """
    try:
        from .reference_sets import validate_reference, signed_plane_distance, in_plane_offset
    except ImportError:  # script-mode workbench
        from reference_sets import validate_reference, signed_plane_distance, in_plane_offset
    result = {}
    for r in records:
        chosen = []
        for name in memberships.get(r.particle_id, ()):
            ref = validate_reference(saved_sets[name].get('best_focus'))
            if ref is not None:
                chosen.append((name, ref))
        unique = {(ref['x_best'], ref['z_best']) for _, ref in chosen}
        if len(unique) > 1:
            result[r.particle_id] = {'reference_source': 'conflicting selected batch references'}
            continue
        if chosen:
            ref, source = chosen[0][1], 'saved batch: ' + ', '.join(n for n, _ in chosen)
        else:
            ref, fallback, kind = fallback_reference(r)
            source = f'{kind}: {fallback}' if ref is not None else 'no saved reference'
        pos = {'reference_source': source}
        if ref is not None:
            pos['x_best'], pos['z_best'] = ref['x_best'], ref['z_best']
            if math.isfinite(r.stage_x) and math.isfinite(r.stage_y):
                pos['d_signed_plane'] = signed_plane_distance(
                    r.stage_x, r.stage_y, ref, plane_normal_deg)
                pos['d_in_plane'] = in_plane_offset(
                    r.stage_x, r.stage_y, ref, plane_normal_deg)
        result[r.particle_id] = pos
    return result

def export_batch_netcdf(output_path, root, records, batch_names, memberships, *,
                        labels=None, positions=None, plane_normal_deg=None,
                        progress=None, cancelled=None):
    """Export exactly the supplied particle IDs; return (count, missing_backgrounds).

    `records` are ParticleRecord snapshots or stable references from the project.
    `memberships`: {particle_id: [names...]}, `positions`: {particle_id: dict
    with x_best,z_best,d_signed_plane,d_in_plane,reference_source}. Defaults
    for missing positions are NaN. Calling this function never modifies MATs,
    labels, flags, calibration projects or other files.
    """
    records = sorted(records, key=lambda r: (r.source_file, r.roi_index))
    if not records:
        raise ValueError('The selected batch has no loaded particles to export')
    if len({r.particle_id for r in records}) != len(records):
        raise ValueError('Duplicate particle IDs in export selection')
    root = Path(root)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    positions = positions or {}
    memberships = memberships or {}

    # Spool exact pixel arrays to disk while reading one source MAT at a time.
    # We only need O(one MAT file) image memory, not O(whole batch) memory.
    with tempfile.TemporaryDirectory(prefix='cpi_nc_') as scratch:
        raw_path = Path(scratch) / 'roi.bin'
        bg_path = Path(scratch) / 'background.bin'
        pixel_starts, pixel_counts, nx, ny, has_bg = [], [], [], [], []
        sources = defaultdict(list)
        for row, r in enumerate(records):
            sources[r.source_file].append((row, r))
        total = len(records)
        done = 0
        with raw_path.open('wb') as raw_out, bg_path.open('wb') as bg_out:
            for source, group in sources.items():
                if cancelled and cancelled():
                    raise ExportCancelled()
                # Do NOT simplify_cells/squeeze_me: scipy otherwise collapses 1xN
                # and Nx1 camera images into vectors and destroys their shape.
                loaded = loadmat(str(root / source), variable_names=['ROI_N', 'BG'],
                                 struct_as_record=True, squeeze_me=False)
                roi = loaded.get('ROI_N')
                images = _mat_struct_entries(_mat_struct_field(roi, 'IMAGE'))
                if not images:
                    raise ValueError(f'Missing ROI_N.IMAGE in {source}')
                bgs = _mat_struct_entries(loaded.get('BG'))
                for row, record in group:
                    if cancelled and cancelled():
                        raise ExportCancelled()
                    i = record.roi_index
                    if i < 0 or i >= len(images):
                        raise ValueError(f'Missing ROI {i} in {source}')
                    img = _mat_roi_image(images[i], 'IM', f'{source} ROI {i}')
                    bg = None
                    if i < len(bgs):
                        try:
                            candidate = _mat_roi_image(bgs[i], 'BG', f'{source} background {i}')
                            if candidate.shape == img.shape:
                                bg = candidate
                        except ValueError:
                            pass  # Missing/empty/incompatible background is recorded, not fabricated.
                    pixel_starts.append(raw_out.tell() // 2)
                    pixel_counts.append(img.size)
                    nx.append(img.shape[0]); ny.append(img.shape[1]); has_bg.append(int(bg is not None))
                    raw_out.write(np.ascontiguousarray(img).tobytes())
                    bg_out.write(np.ascontiguousarray(bg if bg is not None else
                                                    np.full(img.shape, -1, dtype=np.int16)).tobytes())
                    done += 1
                    if progress and (done % 20 == 0 or done == total):
                        progress(done, total)
                del loaded, images, bgs

        n_pixels = sum(pixel_counts)
        if n_pixels >= 2**31:
            raise ValueError('This batch exceeds the 32-bit sample-offset limit; split the selection')
        if n_pixels <= 0:
            raise ValueError('No ROI pixels were exported')
        try:
            from netCDF4 import Dataset
            netcdf4 = True
        except ImportError:
            netcdf4 = False
            if n_pixels * 4 > 256 * 1024 * 1024:
                raise RuntimeError('Large batch: install netCDF4 (pip install netCDF4) for compressed, '
                                   'memory-efficient export')

        tmpname = None
        try:
            fd, tmpname = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent)
            os.close(fd)
            if netcdf4:
                nc = Dataset(tmpname, 'w', format='NETCDF4')
            else:
                nc = netcdf_file(tmpname, mode='w', version=2, mmap=False)
            with nc:
                nc.title = 'CPI calibration batch: raw particle ROIs, backgrounds, positions and labels'
                nc.history = 'Exported ' + datetime.now(timezone.utc).isoformat(timespec='seconds')
                nc.batch_names_json = json.dumps(list(batch_names), ensure_ascii=True)
                nc.image_axis_order = '[camera X, camera Y], C order; zero-based offsets into roi_pixels'
                nc.background_note = 'Unmodified saved ROI background (no division, scaling or edge detection)'
                nc.missing_background_value = -1
                nc.reference_note = 'NaN when no unambiguous saved object-plane reference exists'
                nc.label_area_note = 'Independent SEM/optical reference projected area from label library; missing as NaN'
                nc.shadow_note = ('Shadow diameters and CPI contour area ratios are SAVED processing '
                                  'measurements; NaN if absent or invalid. No reprocessing on export.')
                nc.shape_corrected_shadow_formula = 'shadow_diameter_um / sqrt(cpi_contour_area_ratio)'
                nc.createDimension('particle', total)
                nc.createDimension('sample', n_pixels)
                # These offsets and two dimensions reconstruct each original 2D image.
                for key, values in (
                    ('pixel_start', pixel_starts), ('pixel_count', pixel_counts),
                    ('roi_nx', nx), ('roi_ny', ny), ('background_present', has_bg),
                    ('roi_index', [r.roi_index for r in records])):
                    _create_numeric(nc, key, ('particle',), values)
                _create_numeric(nc, 'time_matlab', ('particle',), [r.time_matlab for r in records],
                                description='Acquisition time as MATLAB datenum')
                number_fields = {
                    'stage_x_mm': ([r.stage_x for r in records], 'mm'),
                    'stage_z_mm': ([r.stage_y for r in records], 'mm'),
                    'camera_x_px': ([r.image_x_px for r in records], 'pixel'),
                    'camera_y_px': ([r.image_y_px for r in records], 'pixel'),
                    'measured_length_um': ([r.length_um for r in records], 'micrometre'),
                    'measured_width_um': ([r.width_um for r in records], 'micrometre'),
                    'cpi_max_feret_length_um': ([getattr(r,'len_feret_um',np.nan) for r in records], 'micrometre'),
                    'cpi_min_feret_width_um': ([getattr(r,'wid_feret_um',np.nan) for r in records], 'micrometre'),
                    'cpi_ellipse_major_axis_um': ([getattr(r,'len_ellipse_um',np.nan) for r in records], 'micrometre'),
                    'cpi_ellipse_minor_axis_um': ([getattr(r,'wid_ellipse_um',np.nan) for r in records], 'micrometre'),
                    'focus': ([r.focus for r in records], None),
                    'pixel_size_um': ([r.pixel_um for r in records], 'micrometre'),
                    'label_length_um': ([_label_size(r, labels)[0] for r in records], 'micrometre'),
                    'label_width_um': ([_label_size(r, labels)[1] for r in records], 'micrometre'),
                    'label_area_um2': ([_label_area(r, labels) for r in records], 'micrometre2'),
                    'label_area_equivalent_diameter_um': (
                        [math.sqrt(4*a/math.pi) if np.isfinite(a) and a > 0 else np.nan
                         for a in (_label_area(r, labels) for r in records)], 'micrometre'),
                    'cpi_contour_equivalent_diameter_um': (
                        [getattr(r, 'deq_um', np.nan) for r in records], 'micrometre'),
                    'cpi_contour_area_ratio': (
                        [getattr(r, 'area_ratio_circle', np.nan) for r in records], None),
                    'shadow_diameter_um': (
                        [getattr(r, 'shadow_deq_um', np.nan) for r in records], 'micrometre'),
                    'shadow_integration_radius_px': (
                        [getattr(r, 'shadow_radius_px', np.nan) for r in records], 'pixel'),
                    'shape_corrected_shadow_diameter_um': (
                        [_shape_shadow(r) for r in records], 'micrometre'),
                    'x_best_mm': ([positions.get(r.particle_id, {}).get('x_best', np.nan) for r in records], 'mm'),
                    'z_best_mm': ([positions.get(r.particle_id, {}).get('z_best', np.nan) for r in records], 'mm'),
                    'distance_from_object_plane_mm': (
                        [positions.get(r.particle_id, {}).get('d_signed_plane', np.nan) for r in records], 'mm'),
                    'in_plane_offset_mm': (
                        [positions.get(r.particle_id, {}).get('d_in_plane', np.nan) for r in records], 'mm'),
                }
                for key, (values, units) in number_fields.items():
                    _create_numeric(nc, key, ('particle',), [_finite(v) for v in values], units=units)
                distances = number_fields['distance_from_object_plane_mm'][0]
                lengths = number_fields['label_length_um'][0]
                zd = [0.85 * float(d) * 1000 / (float(length) / 2)**2
                      if np.isfinite(d) and np.isfinite(length) and length > 0 else np.nan
                      for d, length in zip(distances, lengths)]
                _create_numeric(nc, 'zd', ('particle',), zd,
                                description='Signed normalised defocus; 0.85 um laser assumed, label radius')
                if plane_normal_deg is not None:
                    nc.plane_normal_deg = float(plane_normal_deg)
                strings = {
                    'particle_id': [r.particle_id for r in records],
                    'source_file': [r.source_file for r in records],
                    'label': [r.label for r in records],
                    'batch_memberships_json': [json.dumps(list(memberships.get(r.particle_id, [])), ensure_ascii=False)
                                               for r in records],
                    'reference_source': [positions.get(r.particle_id, {}).get('reference_source', '') for r in records],
                }
                for key, values in strings.items():
                    _put_strings(nc, key, values)
                # Chunked writes: avoid allocating two large in-memory images.
                for name, source in (('roi_pixels', raw_path), ('background_pixels', bg_path)):
                    if netcdf4:
                        v = nc.createVariable(name, 'i2', ('sample',), zlib=True, complevel=4,
                                              chunksizes=(min(n_pixels, 262144),))
                    else:
                        v = nc.createVariable(name, 'i2', ('sample',))
                    v.long_name = 'Flattened original ROI' if name == 'roi_pixels' else 'Flattened saved background'
                    v.units = 'raw camera digital counts'
                    with source.open('rb') as stream:
                        start = 0
                        while start < n_pixels:
                            if cancelled and cancelled():
                                raise ExportCancelled()
                            length = min(262144, n_pixels - start)
                            values = np.fromfile(stream, dtype='<i2', count=length)
                            if len(values) != length:
                                raise IOError('Premature end of temporary pixel storage')
                            v[start:start + length] = values
                            start += length
            if cancelled and cancelled():
                raise ExportCancelled()
            os.replace(tmpname, target)
            tmpname = None
            return total, total - sum(has_bg)
        finally:
            if tmpname is not None:
                try:
                    os.unlink(tmpname)
                except FileNotFoundError:
                    pass
