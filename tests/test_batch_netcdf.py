"""Round-trip raw ROI and background export; no Qt required."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file, savemat
from calibration.data import ParticleRecord
from calibration.batch_netcdf import export_batch_netcdf, ExportCancelled, batch_position_metadata
from calibration.labels import LabelLibrary, ParticleLabel


def _fixture(tmp_path):
    images = [np.array([[0, 255, 11], [20, 99, 253]], dtype=np.uint8),
              np.array([[3, 200], [7, 98], [130, 0]], dtype=np.uint8),
              np.array([[20, 210, 56, 2]], dtype=np.uint8)]
    bg = [np.full(images[0].shape, 250, dtype=np.uint8),
          np.full(images[1].shape, 120, dtype=np.uint8)]
    ims = np.empty((3,), dtype=[('IM', object)])
    bgs = np.empty((3,), dtype=[('BG', object)])
    for i, im in enumerate(images):
        ims[i]['IM'] = im
        bgs[i]['BG'] = bg[i] if i < len(bg) else np.empty((0, 0), dtype=np.uint8)
    savemat(tmp_path / 'sample.mat', {'ROI_N': {'IMAGE': ims}, 'BG': bgs})
    records = []
    for i in range(3):
        r = ParticleRecord('sample.mat', i, 738000. + i / 1000, 21 + i, 20, 10, .9,
                           15+i, 30, 0, 0, stage_x=7. + i, stage_y=5.)
        r.label = 'Bead-15 µm' if i != 1 else ''
        r.actual_length_um = 0. if i != 1 else np.nan  # prove current library is authoritative
        records.append(r)
    labels = LabelLibrary(tmp_path / 'labels.json')
    labels.upsert(ParticleLabel(name='Bead-15 µm', length_um=15., width_um=15.))
    return images, bg, records, labels


def _strings(nc, field):
    a = nc.variables[field][:]
    return [bytes(row).split(b'\0', 1)[0].decode('utf-8') for row in a]


def test_batch_export_preserves_images_background_positions_labels(tmp_path):
    images, bg, records, labels = _fixture(tmp_path)
    groups = {r.particle_id: ['A'] for r in records}
    groups[records[0].particle_id].append('B')
    pos = {r.particle_id: dict(x_best=7, z_best=5, d_signed_plane=i,
                               d_in_plane=0., reference_source='saved batch: A')
           for i, r in enumerate(records)}
    filename = tmp_path / 'batch.nc'
    n, missing = export_batch_netcdf(filename, tmp_path, records, ['A', 'B'], groups,
                                    labels=labels, positions=pos, plane_normal_deg=0)
    assert (n, missing) == (3, 1)
    with netcdf_file(filename, 'r', mmap=False) as nc:
        assert len(nc.dimensions) > 3
        assert json.loads(nc.batch_names_json.decode()) == ['A', 'B']
        assert nc.variables['label_length_um'][:].tolist()[0] == 15.
        assert np.isnan(nc.variables['label_length_um'][:][1])
        assert nc.variables['stage_x_mm'][:].tolist() == [7.,8.,9.]
        assert nc.variables['distance_from_object_plane_mm'][:].tolist() == [0.,1.,2.]
        assert _strings(nc, 'label') == ['Bead-15 µm', '', 'Bead-15 µm']
        assert json.loads(_strings(nc, 'batch_memberships_json')[0]) == ['A','B']
        assert list(nc.variables['background_present'][:]) == [1,1,0]
        starts = nc.variables['pixel_start'][:]
        counts = nc.variables['pixel_count'][:]
        for i, orig in enumerate(images):
            s, c = int(starts[i]), int(counts[i])
            x, y = int(nc.variables['roi_nx'][i]), int(nc.variables['roi_ny'][i])
            assert np.array_equal(nc.variables['roi_pixels'][s:s+c].reshape(x, y), orig)
            data = nc.variables['background_pixels'][s:s+c].reshape(x, y)
            if i < 2:
                assert np.array_equal(data, bg[i])
            else:
                assert np.all(data == -1)


def test_invalid_roi_or_cancel_does_not_replace_destination(tmp_path):
    images, bg, records, labels = _fixture(tmp_path)
    path = tmp_path / 'keep.nc'
    path.write_bytes(b'keep this file')
    with pytest.raises(ExportCancelled):
        export_batch_netcdf(path, tmp_path, records, ['B'], {}, cancelled=lambda: True)
    assert path.read_bytes() == b'keep this file'
    records[0].roi_index = 20
    with pytest.raises(ValueError, match='Missing ROI'):
        export_batch_netcdf(path, tmp_path, records, ['B'], {})
    assert path.read_bytes() == b'keep this file'


def test_empty_rejected(tmp_path):
    with pytest.raises(ValueError, match='no loaded particles'):
        export_batch_netcdf(tmp_path / 'x.nc', tmp_path, [], ['A'], {})


def test_selected_batch_own_reference_and_conflict(tmp_path):
    _, _, records, _ = _fixture(tmp_path)
    sets = {'one': {'best_focus': {'x_best': 7, 'z_best': 5}},
            'two': {'best_focus': {'x_best': 9, 'z_best': 5}},
            'unplaced': {}}
    members = {records[0].particle_id: ['one'],
               records[1].particle_id: ['one', 'two'],
               records[2].particle_id: ['unplaced']}
    calls = []
    def fallback(record):
        calls.append(record.particle_id)
        return None, '', ''
    result = batch_position_metadata(records, members, sets, 0., fallback)
    assert result[records[0].particle_id]['d_signed_plane'] == 0
    assert 'conflicting' in result[records[1].particle_id]['reference_source']
    assert 'd_signed_plane' not in result[records[1].particle_id]
    assert calls == [records[2].particle_id]

@pytest.mark.parametrize('n', [1, 3])
def test_batch_export_handles_real_cpi_matlab_cell_struct_layout(tmp_path, n):
    """Regression: proc/postProcess.py serialises list-of-dict ROIs as cells.

    Unlike our initial synthetic fixture, scipy.savemat writes ROI_N.IMAGE as
    an object cell array containing 1x1 structs when passed a list of dicts.
    The old exporter falsely rejected this layout as 'Invalid ROI_N.IMAGE'.
    """
    raw = [np.arange(12, dtype=np.uint8).reshape(3, 4),
           np.array([[9, 25, 87, 160]], dtype=np.uint8),
           np.array([[42], [1], [59]], dtype=np.uint8)][:n]
    bgs = [np.full(x.shape, 100 + i, dtype=np.uint8) for i, x in enumerate(raw)]
    # Use exactly the list-of-dicts layout produced by postProcess / associateBackgrounds.
    savemat(tmp_path / 'cpi.mat', {'ROI_N': {'IMAGE': [{'IM': im} for im in raw]},
                                    'BG': [{'BG': bg} for bg in bgs]})
    from scipy.io import loadmat
    non_squeezed = loadmat(tmp_path / 'cpi.mat', squeeze_me=False)
    assert non_squeezed['ROI_N']['IMAGE'].flat[0].dtype.names is None
    records = [ParticleRecord('cpi.mat', i, 738000. + i, 14., 14., 1., 1., 8., 8., 0., 0.)
               for i in range(n)]
    destination = tmp_path / 'out.nc'
    snapshot = (tmp_path / 'cpi.mat').read_bytes()
    number, missing = export_batch_netcdf(destination, tmp_path, records, ['cal'], {})
    assert number == n and missing == 0
    assert (tmp_path / 'cpi.mat').read_bytes() == snapshot
    with netcdf_file(destination, 'r', mmap=False) as nc:
        for i in range(n):
            start = int(nc.variables['pixel_start'][i]); count = int(nc.variables['pixel_count'][i])
            shape = (int(nc.variables['roi_nx'][i]), int(nc.variables['roi_ny'][i]))
            assert shape == raw[i].shape
            assert np.array_equal(nc.variables['roi_pixels'][start:start+count].reshape(shape), raw[i])
            assert np.array_equal(nc.variables['background_pixels'][start:start+count].reshape(shape), bgs[i])


def test_cell_struct_layout_missing_background_has_explicit_flag(tmp_path):
    """MAT cell structs may have an empty/invalid background for one ROI."""
    first = np.array([[1, 4], [7, 9]], dtype=np.uint8)
    second = np.array([[3, 1, 4]], dtype=np.uint8)
    savemat(tmp_path / 'cells.mat', {
        'ROI_N': {'IMAGE': [{'IM': first}, {'IM': second}]},
        'BG': [{'BG': np.full(first.shape, 200, dtype=np.uint8)},
               {'BG': np.empty((0, 0), dtype=np.uint8)}]
    })
    records = [ParticleRecord('cells.mat', i, 738000., 4., 4., 0., 0., 0., 0., 0., 0.) for i in range(2)]
    _, missing = export_batch_netcdf(tmp_path / 'cells.nc', tmp_path, records, ['cal'], {})
    assert missing == 1
    with netcdf_file(tmp_path / 'cells.nc', 'r', mmap=False) as nc:
        assert nc.variables['background_present'][:].tolist() == [1, 0]
        start = int(nc.variables['pixel_start'][1]); count = int(nc.variables['pixel_count'][1])
        assert np.all(nc.variables['background_pixels'][start:start+count] == -1)


def test_new_area_shadow_fields_and_unset_width(tmp_path):
    _, _, records, labels = _fixture(tmp_path)
    labels.upsert(ParticleLabel(name='Bead-15 µm', length_um=15., width_um=np.nan, area_um2=125.))
    records[0].deq_um = 11.
    records[0].shadow_deq_um = 10.
    records[0].area_ratio_circle = .25
    records[0].shadow_radius_px = 13.
    out = tmp_path / 'new_fields.nc'
    export_batch_netcdf(out, tmp_path, records, ['A'], {}, labels=labels)
    with netcdf_file(out, 'r', mmap=False) as nc:
        def v(k): return np.array(nc.variables[k][:])
        assert np.isnan(v('label_width_um')[0])
        assert v('label_area_um2')[0] == 125.
        assert np.isclose(v('label_area_equivalent_diameter_um')[0], (500/np.pi)**.5)
        assert v('cpi_contour_equivalent_diameter_um')[0] == 11.
        assert v('cpi_contour_area_ratio')[0] == .25
        assert v('shadow_diameter_um')[0] == 10.
        assert v('shadow_integration_radius_px')[0] == 13.
        assert v('shape_corrected_shadow_diameter_um')[0] == 20.
        assert np.isnan(v('shape_corrected_shadow_diameter_um')[1])
        assert np.isnan(v('label_area_um2')[1])
