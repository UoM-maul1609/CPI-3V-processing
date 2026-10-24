"""Adaptive radius, ROI censoring, and matching backgrounds in processing and Edge Lab."""
import sys
from pathlib import Path
import numpy as np
from scipy.io import savemat, loadmat
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from proc.shadowCircle import measure
from calibration.edge_preview import load_roi_for_preview, run_preview, shadow_options
from calibration.data import ParticleRecord


def disk(n=90, cy=45., cx=45., r=8., multiplier=1.):
    y, x = np.mgrid[:n, :n]
    im = np.where((y-cy)**2+(x-cx)**2 <= r*r, 0., 100.)*multiplier
    bg = np.full((n, n), 100.)
    phi = np.linspace(0, 2*np.pi, 130)
    boundary = np.c_[cy+r*np.sin(phi), cx+r*np.cos(phi)]
    return im, bg, boundary


def test_auto_plateau_has_consistent_scale_and_illumination_correction():
    im, bg, b = disk()
    a = measure(im, bg, b, radius_mode='adaptive', denoise=0, pixel_um=1)
    c = measure(im*2., bg, b, radius_mode='adaptive', denoise=0, pixel_um=1)
    assert a['shadow_diameter'] == pytest.approx(c['shadow_diameter'])
    assert a['shadow_diameter'] == pytest.approx(16, abs=.4)
    assert a['radius_mode'] == 'adaptive'
    assert 0 <= a['plateau_fraction_observed'] <= .015
    assert not a['roi_extrapolated']


def test_auto_small_roi_uses_observed_pixels_only_and_reports_coverage():
    im, bg, b = disk(cy=15., cx=45.)
    a = measure(im, bg, b, radius_mode='adaptive', denoise=0, pixel_um=1)
    assert not a['roi_extrapolated']
    assert a['roi_truncated']
    assert 0 < a['aperture_missing_fraction'] < .15
    assert a['coverage_min'] >= .85
    assert a['shadow_diameter'] == pytest.approx(16, abs=.8)
    im, bg, b = disk(cy=11., cx=45.)
    with pytest.raises(ValueError, match='cropped|coverage'):
        measure(im, bg, b, radius_mode='adaptive', denoise=0, pixel_um=1)


def test_auto_does_not_invent_a_plateau_when_area_is_still_growing():
    im, bg, b = disk()
    yy, xx = np.mgrid[:90, :90]
    # A broader weak shadow keeps contributing to the integral.
    im = np.minimum(im, np.where(np.hypot(yy-45, xx-45)<38, 65., 100.))
    with pytest.raises(ValueError, match='plateau|cropped'):
        measure(im, bg, b, radius_mode='adaptive', denoise=0, pixel_um=1, max_padding_px=18)


@pytest.mark.parametrize('n', [1, 2])
def test_edge_preview_prefers_saved_background_used_in_processing(tmp_path, n):
    im, _, b = disk(n=80, cy=40, cx=40)
    # The original assigned BG is contaminated/incorrect; processing used a clean frame.
    wrong_bg = np.full((80,80), 70.)
    right_bg = np.full((80,80), 100.)
    image_struct = np.empty((1,n), dtype=[('IM', object)])
    bg_struct = np.empty((1,n), dtype=[('BG', object)])
    focus = np.empty((1,n), dtype=[('boundaries', object)])
    bg_used = np.empty((1,n), dtype=object)
    for i in range(n):
        image_struct[0,i]['IM'] = im
        bg_struct[0,i]['BG'] = wrong_bg
        bg_used[0,i] = right_bg
        focus[0,i]['boundaries'] = b
    matfile = tmp_path/'test.mat'
    savemat(matfile, {'ROI_N': {'IMAGE': image_struct, 'imageType': np.ones(n), 'StartX': np.zeros(n)},
                      'BG': bg_struct, 'BG_USED': bg_used, 'dat': {'foc': focus}})
    records=[ParticleRecord('test.mat', i, 0., 16., 16., 3., 1., 0., 0., 0., 0., pixel_um=1)
             for i in range(n)]
    before=matfile.read_bytes()
    for rec in records:
        loaded_im, loaded_bg = load_roi_for_preview(tmp_path, rec)
        assert np.array_equal(loaded_im, im)
        assert np.array_equal(loaded_bg, right_bg)
    outputs, failures = run_preview(tmp_path, records, shadow_options(8, denoise=0, pixel_um=1))
    assert not failures
    assert len(outputs)==n
    assert matfile.read_bytes()==before
