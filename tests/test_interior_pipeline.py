"""Production MAT -> catalogue -> NetCDF round-trip for optical interior metrics."""
import numpy as np
import pytest
from scipy.io import savemat, loadmat, netcdf_file

from calibration.contour_diagnostics import interior_light_metrics as gui_metrics
from proc.interiorLight import interior_light_metrics as production_metrics, INTERIOR_FIELD_NAMES, MAT_FIELD_NAMES
from proc.edgeDetect import EdgeOptions, transmittance
from proc.imageStats import imageStats
from calibration.data import scan_processed_file
from calibration.batch_netcdf import export_batch_netcdf


def test_gui_and_processing_share_one_implementation():
    assert gui_metrics is production_metrics
    T = np.ones((80,80))
    T[15:65, 15:65] = .2
    T[35:45,35:45] = 1.1
    boundary = np.array([[15,15],[15,64],[64,64],[64,15]])
    result = production_metrics(T, boundary)
    assert set(result) == set(INTERIOR_FIELD_NAMES)
    assert 0 < result['interior_enclosed_bright_fraction'] < 1
    assert result['interior_pixel_count'] > 0


def test_processed_mat_catalog_netcdf_interior_roundtrip(tmp_path):
    yy, xx = np.indices((90,90))
    bg = np.full((90,90), 200, np.uint8)
    img = np.where((yy-45)**2 + (xx-45)**2 < 23**2, 40, 200).astype(np.uint8)
    # A genuine bright optical island inside the object's dark outer contour.
    img[40:50,40:50] = 218
    ims = np.empty((1,1), dtype=[('IM',object)])
    bgs = np.empty((1,1), dtype=[('BG',object)])
    ims[0,0]['IM'] = img
    bgs[0,0]['BG'] = bg
    roi = dict(IMAGE=ims, imageType=np.array([[89]]), Time=np.array([[738000.]]),
               StartX=np.array([[0.]]), StartY=np.array([[0.]]),
               EndX=np.array([[89.]]), EndY=np.array([[89.]]))
    mat_path = tmp_path / 'cpi.mat'
    savemat(mat_path, {'ROI_N':roi, 'BG':bgs})
    loaded = loadmat(mat_path)
    opt = EdgeOptions(method='transmittance', edge_mode='relative', edge_fraction=.55,
                      bg_check=False, pixel_um=2.3, shadow_circle_padding=8.)
    dat = imageStats(loaded['ROI_N'], loaded['BG'], False, True, 0, 'test', None, options=opt)
    for key in INTERIOR_FIELD_NAMES:
        assert np.isfinite(dat[MAT_FIELD_NAMES.get(key,key)][0,0]), key
    assert dat['interior_above_one_fraction'][0,0] > 0
    assert dat['interior_enclosed_fraction'][0,0] > 0
    assert dat['interior_pixel_count'][0,0] > 0
    assert np.isfinite(dat['len'][0,0]) and np.isfinite(dat['shadow_deq'][0,0])
    saved_boundary = dat['foc'][0]['boundaries']
    expected = production_metrics(transmittance(img,bg,opt), saved_boundary)
    assert dat['interior_bright_fraction'][0,0] == pytest.approx(expected['interior_bright_fraction'])
    savemat(mat_path, {'ROI_N':loaded['ROI_N'], 'BG':loaded['BG'], 'dat':dat})
    records = scan_processed_file(mat_path, tmp_path)
    assert len(records) == 1
    assert records[0].interior_mean_T == pytest.approx(dat['interior_mean_T'][0,0])
    out = tmp_path / 'cpi.nc'
    export_batch_netcdf(out,tmp_path,records,['A'],{})
    netcdf_names = {
        'interior_bright_fraction':'interior_bright_fraction',
        'interior_bright_threshold':'interior_bright_threshold',
        'interior_enclosed_bright_fraction':'interior_enclosed_fraction',
        'interior_near_background_fraction':'interior_near_bg_fraction',
        'interior_above_one_fraction':'interior_above_one_fraction',
        'interior_mean_transmittance':'interior_mean_T',
        'interior_effective_deficit':'interior_effective_deficit',
        'interior_pixel_count':'interior_pixel_count',
    }
    with netcdf_file(out,'r',mmap=False) as nc:
        assert 'optical' in nc.interior_diagnostics_note.decode().lower()
        for out_name, mat_name in netcdf_names.items():
            value = float(nc.variables[out_name][0])
            assert value == pytest.approx(dat[mat_name][0,0]), (out_name,mat_name)


def test_netcdf_old_records_without_interior_serialize_missing(tmp_path):
    from calibration.data import ParticleRecord
    images = np.empty((1,), dtype=[('IM', object)])
    images[0]['IM'] = np.full((6,6), 200, np.uint8)
    bg = np.empty((1,), dtype=[('BG', object)])
    bg[0]['BG'] = np.full((6,6), 220, np.uint8)
    savemat(tmp_path/'old.mat', {'ROI_N':{'IMAGE':images},'BG':bg})
    rec = ParticleRecord('old.mat',0,738000.,10.,7.,1.,.9,5.,5.,0.,0.)
    output = tmp_path/'old.nc'
    export_batch_netcdf(output,tmp_path,[rec],['A'],{})
    with netcdf_file(output,'r',mmap=False) as nc:
        for key in ('interior_bright_fraction','interior_bright_threshold',
                    'interior_enclosed_bright_fraction','interior_near_background_fraction',
                    'interior_above_one_fraction','interior_mean_transmittance',
                    'interior_effective_deficit','interior_pixel_count'):
            assert np.isnan(float(nc.variables[key][0])), key
