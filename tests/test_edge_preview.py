"""Non-destructive edge preview tests, independent of Qt."""
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.io as sio
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from calibration.edge_preview import METHODS, preview_options, preview_one, run_preview
from calibration.data import ParticleRecord


def bead(d=24, n=64):
    yy, xx = np.mgrid[:n, :n]
    bg = np.full((n, n), 180, dtype=np.uint8)
    im = np.where(np.hypot(yy-n/2, xx-n/2) < d/2, 55, 180).astype(np.uint8)
    return im, bg


def test_algorithms_are_real_and_images_unchanged():
    image, bg = bead()
    previous = image.copy()
    for method, _ in METHODS:
        opts = preview_options(method, .55, 1, 0., pixel_um=1)
        assert opts.method == 'transmittance' and opts.bg_check is False
        assert (opts.edge_mode, opts.outline) == tuple(method.split('/'))
        result = preview_one(image, bg, opts)
        assert result is not None and result['boundary'].shape[1] == 2
    assert np.array_equal(image, previous)
    assert preview_one(image, bg[:20], opts) is None
    with pytest.raises(ValueError):
        preview_options('unknown')


def test_batch_uses_record_pixel_scale_and_no_writes(tmp_path):
    images = np.empty((1, 3), dtype=[('IM', object)])
    bgs = np.empty((1, 3), dtype=[('BG', object)])
    for i in range(3):
        images[0, i]['IM'], bgs[0, i]['BG'] = bead(20 + 5*i)
    path = tmp_path / 'test.mat'
    sio.savemat(path, {'ROI_N': {'IMAGE': images}, 'BG': bgs})
    raw = path.read_bytes()
    records = [ParticleRecord('test.mat', i, float(i+1), 12., 11., 7., .99, 50., 51., 10., 10.,
                              pixel_um=1.+i) for i in range(3)]
    before = [(r.length_um, r.focus, r.status) for r in records]
    progress = []
    results, failures = run_preview(tmp_path, records, preview_options('relative/contour'),
                                    progress=lambda a,b: progress.append((a,b)))
    assert failures == {} and len(results) == 3
    assert progress[-1] == (3, 3)
    assert results[records[0].particle_id]['len'] == pytest.approx(20., abs=2)
    assert results[records[2].particle_id]['len'] == pytest.approx(90., abs=4)
    assert all(v['image_shape'] == (64, 64) for v in results.values())
    assert [(r.length_um, r.focus, r.status) for r in records] == before
    assert path.read_bytes() == raw and list(tmp_path.iterdir()) == [path]


def test_missing_background_and_cancellation_are_explicit(tmp_path):
    r = ParticleRecord('absent.mat', 0, 1., 10., 10., 1., 1., 1., 1., 1., 1.)
    result, errors = run_preview(tmp_path, [r], preview_options('fixed/contour'))
    assert not result and r.particle_id in errors
    result, errors = run_preview(tmp_path, [r], preview_options('fixed/contour'), cancelled=lambda: True)
    assert not result and not errors


def test_command_flags_represent_every_trial_control():
    import shlex
    from calibration.edge_preview import command_flags
    flags = shlex.split(command_flags('relative/radial', .25, 2, .4, True))
    assert flags == [
        '--edge-method', 'transmittance', '--edge-mode', 'relative',
        '--outline', 'radial', '--edge-fraction', '0.25', '--denoise', '2',
        '--smooth-sigma', '0.4', '--cpiview']
    other = shlex.split(command_flags('fixed/edge', .65, 0, 0., False))
    assert '--edge-level' in other and '--edge-fraction' not in other
    assert '--cpiview' not in other
    with pytest.raises(ValueError):
        command_flags('invalid/radial')


def test_preview_runs_the_independent_cpiview_focus_calculation():
    from calibration.edge_preview import preview_options
    from proc.cpiviewMetrics import cpiview_metrics
    image, bg = bead(32, 96)
    base = preview_one(image, bg, preview_options('relative/radial', .25, cpiview=False))
    extra = preview_one(image, bg, preview_options('relative/radial', .25, cpiview=True))
    assert base is not None and extra is not None
    assert 'cpv_focus' not in base
    assert 'cpv_focus' in extra and 'cpv_fwhm_px' in extra
    assert base['deq'] == pytest.approx(extra['deq'], abs=1e-9)
    assert extra['cpv_focus'] == pytest.approx(cpiview_metrics(image, bg)['cpv_focus'], nan_ok=True)


def test_fixed_threshold_measures_faint_small_bead_without_dark_core():
    from calibration.edge_preview import transmittance_profile
    bg = np.full((64, 64), 180, dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]
    image = np.where(np.hypot(yy - 32, xx - 32) < 9, 115, 180).astype(np.uint8)
    opts = preview_options('fixed/radial', .8, denoise=0, pixel_um=1)
    assert np.min(image / bg) > .45
    r = preview_one(image, bg, opts)
    assert r is not None and r['deq'] > 10
    assert r['edge_level_used'] == pytest.approx(.8)
    diag = transmittance_profile(image, bg, opts, r['boundary'])
    assert diag['border_median'] == pytest.approx(1.)
    assert min(diag['horizontal']) > .45


def test_fixed_threshold_is_invariant_to_multiplicative_laser_changes():
    bg = np.full((70, 70), 90, dtype=np.uint8)
    yy, xx = np.mgrid[:70, :70]
    shadow = np.hypot(yy - 35, xx - 35) < 11
    first = np.where(shadow, 54, 90).astype(np.uint8)
    second = (first * 2).astype(np.uint8)
    opts = preview_options('fixed/contour', .8, denoise=0, pixel_um=1)
    a = preview_one(first, bg, opts)
    b = preview_one(second, bg, opts)
    assert a is not None and b is not None
    assert a['deq'] == pytest.approx(b['deq'], abs=.01)
