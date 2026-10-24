"""Experimental signed-shadow circle: no Qt and no mutations."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.io import savemat
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from calibration.edge_preview import (
    fit_stored_circle, shadow_options, shadow_integral_one, run_preview, command_flags)
from calibration.data import ParticleRecord


def synthetic(n=80, diameter=20, level=0., brightness=1.):
    y, x = np.mgrid[:n, :n]
    radius = diameter / 2
    background = np.full((n, n), 100., float)
    image = np.where((y - 40.) ** 2 + (x - 40.) ** 2 <= radius * radius,
                     level * 100., 100.) * brightness
    theta = np.linspace(0, 2*np.pi, 100)
    boundary = np.c_[40 + radius * np.sin(theta), 40 + radius * np.cos(theta)]
    return image, background, boundary


def test_shadow_integral_returns_area_equivalent_for_ideal_opaque_disk():
    image, bg, boundary = synthetic(diameter=20)
    original = image.copy()
    out = shadow_integral_one(image, bg, boundary, shadow_options(8, denoise=0, pixel_um=1))
    assert out['deq'] == pytest.approx(20., abs=.25)
    assert out['overlay_kind'] == 'integration circle'
    assert out['aperture_radius_px'] == pytest.approx(18, abs=.1)
    assert out['circle_rms_px'] < .01
    assert out['boundary'].shape == (129, 2)
    assert np.array_equal(image, original)


def test_illumination_scale_invariance_and_bright_fringe_negative_sign():
    im, bg, b = synthetic(diameter=16)
    opt = shadow_options(8, denoise=0, pixel_um=1)
    value = shadow_integral_one(im, bg, b, opt)['shadow_area_px']
    more_illumination = shadow_integral_one(im * 1.8, bg, b, opt)['shadow_area_px']
    assert more_illumination == pytest.approx(value, rel=1e-8)
    y, x = np.mgrid[:80, :80]
    dist = np.hypot(y - 40., x - 40.)
    im[(dist > 9) & (dist < 13)] = 120
    less = shadow_integral_one(im, bg, b, opt)['shadow_area_px']
    assert less < value


def test_truncated_circle_treats_outside_roi_as_clear_light():
    im, bg, b = synthetic(diameter=20)
    # A 40-pixel integration circle just clips the border of this 80px ROI.
    full = shadow_integral_one(im, bg, b, shadow_options(8, denoise=0, pixel_um=1))
    clipped = shadow_integral_one(im, bg, b, shadow_options(30, denoise=0, pixel_um=1))
    assert clipped['roi_truncated']
    assert 0 < clipped['aperture_missing_fraction'] < 0.03
    assert clipped['roi_extrapolated'] is False
    assert clipped['outside_transmittance'] == 1.
    assert clipped['shadow_area_px'] == pytest.approx(full['shadow_area_px'], rel=1e-10)
    # A substantially cropped aperture is still calculated, but flagged.
    # Here its dark disk remains entirely inside the ROI.
    shifted = np.roll(im, -23, axis=1)
    shifted_b = b.copy()
    shifted_b[:, 1] -= 23
    clipped_more = shadow_integral_one(shifted, bg, shifted_b, shadow_options(30, denoise=0, pixel_um=1))
    assert clipped_more['aperture_missing_fraction'] > .15
    assert clipped_more['shadow_area_px'] == pytest.approx(full['shadow_area_px'], rel=.01)
    assert clipped_more['aperture_coverage'] == pytest.approx(1.-clipped_more['aperture_missing_fraction'])
    assert not clipped_more['roi_extrapolated']
    with pytest.raises(ValueError, match='stored boundary'):
        shadow_integral_one(im, bg, None, shadow_options(8, pixel_um=1))
    with pytest.raises(ValueError, match='experimental'):
        command_flags('shadow/circle')


def test_missing_parts_of_circle_match_explicit_t_equals_one_padding():
    # Test the actual signed integral at the transmittance-map level, avoiding
    # differences in background normalisation caused by a different ROI border.
    from proc.shadowCircle import _fractional_disk, cumulative_shadow
    T = np.ones((45, 45), float)
    yy, xx = np.mgrid[:45, :45]
    T[(yy-13)**2 + (xx-9)**2 <= 5**2] = .2
    centre, radius = (13., 9.), 18.
    crop, mask = _fractional_disk(T.shape, centre, radius)
    partial = float(np.sum((1.-T[crop])*mask))
    pad = 35
    expanded = np.ones((45+2*pad, 45+2*pad), float)
    expanded[pad:pad+45, pad:pad+45] = T
    crop2, mask2 = _fractional_disk(expanded.shape, (13.+pad, 9.+pad), radius)
    explicit = float(np.sum((1.-expanded[crop2])*mask2))
    assert partial == pytest.approx(explicit, abs=1e-12)
    rr, integral, coverage, annular = cumulative_shadow(T, centre, radius+1)
    assert coverage[-1] < 1.
    assert integral[int(radius)-1] == pytest.approx(partial, abs=1.)


def test_unseen_dark_shadow_is_not_extrapolated_from_visible_fraction():
    from proc.shadowCircle import _fractional_disk
    yy, xx = np.mgrid[:65, :65]
    full_T = np.ones((65, 65), dtype=float)
    full_T[(yy - 30.)**2 + (xx - 25.)**2 <= 12.**2] = .2
    # Mimic an image crop that cuts through the dark object itself.
    cropped_T = full_T[:, 25:]
    radius = 20.
    full_slice, full_mask = _fractional_disk(full_T.shape, (30., 25.), radius)
    crop_slice, crop_mask = _fractional_disk(cropped_T.shape, (30., 0.), radius)
    full_area = np.sum((1. - full_T[full_slice])*full_mask)
    observed_area = np.sum((1. - cropped_T[crop_slice])*crop_mask)
    assert observed_area < full_area  # Missing deficit is zero, never extrapolated.
    observed_fraction = crop_mask.sum()/(np.pi*radius**2)
    assert observed_area/observed_fraction > observed_area
    # Explicitly representing missing pixels by T=1 produces identical area.
    explicit_T = np.ones_like(full_T)
    explicit_T[:, 25:] = cropped_T
    explicit_area = np.sum((1. - explicit_T[full_slice])*full_mask)
    assert observed_area == pytest.approx(explicit_area, abs=1e-12)


def test_run_preview_uses_mat_stored_outline_without_changing_file(tmp_path):
    im, bg, boundary = synthetic()
    roi = np.empty((1, 2), dtype=[('IM', object)])
    bgs = np.empty((1, 2), dtype=[('BG', object)])
    foc = np.empty((1, 2), dtype=[('boundaries', object)])
    for i in range(2):
        roi[0, i]['IM'], bgs[0, i]['BG'] = im, bg
        foc[0, i]['boundaries'] = boundary if i == 0 else np.empty((0,2))
    path = tmp_path/'synthetic.mat'
    savemat(path, {'ROI_N': {'IMAGE': roi}, 'BG': bgs, 'dat': {'foc': foc}})
    before = path.read_bytes()
    records = [ParticleRecord('synthetic.mat', i, 0., 20., 20., 3., 1., 0., 0., 0., 0., pixel_um=1)
               for i in range(2)]
    results, failures = run_preview(tmp_path, records, shadow_options(8, denoise=0, pixel_um=1))
    assert len(results) == 1
    assert results[records[0].particle_id]['deq'] == pytest.approx(20, abs=.25)
    assert 'boundary' in failures[records[1].particle_id]
    assert path.read_bytes() == before
    assert records[0].length_um == 20
