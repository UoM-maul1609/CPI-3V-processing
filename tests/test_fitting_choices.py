"""Regression for the new non-destructive shadow+normalised-focus calibration."""
import numpy as np
import pytest
from calibration import sizing
from proc.shadowCircle import measure, local_illumination


def make_rows():
    rng = np.random.default_rng(72)
    rows = []
    for diameter in (15., 40., 99.4):
        for j, z in enumerate(np.linspace(-.45, .45, 13)):
            for i in range(5):
                illum = 80. + 65. * rng.random()
                w = (.22 if z < 0 else .36) * (diameter/40.)**.38
                amp = .20*(diameter/40.)**.31
                norm_focus = .014 + amp*np.exp(-abs(z/.07 if not w else z/w)**2.3)
                rows.append({"actual_length_um": diameter, "shadow_deq_um": 1.025*diameter + 1.1 + rng.normal(0,.24),
                             "length_um": diameter*1.1, "focus": norm_focus*illum*np.exp(rng.normal(0, .03)),
                             "local_illum": illum, "d_signed_plane": z, "batch": f"{diameter:g} stage{j:02}",
                             "pixel_um":2.46, "label": f"{diameter:g}um"})
    return rows


def test_shadow_and_generalized_focus_parameters():
    rows = make_rows()
    table = sizing.CalibrationTable.from_rows(rows, 'shadow_deq', 'focus_per_illum')
    assert np.isclose(table.focus[0], rows[0]['focus']/rows[0]['local_illum'])
    fit = sizing.fit_model(table, size_model="shadow_linear", focus_model="normalised_generalised", offset_um=0)
    assert fit.scale == pytest.approx(1.025, rel=.035)
    assert fit.offset_um == pytest.approx(1.1, abs=1.7)
    assert fit.focus_L_amp > 0 and fit.focus_L_width > 0
    assert fit.focus_q == pytest.approx(2.3, abs=.4)
    assert fit.focus_w_plus_mm > fit.focus_w_minus_mm
    assert fit.focus_gamma == 0
    assert float(fit.mean_length(40., -.3)) == pytest.approx(float(fit.mean_length(40., .3)))
    assert fit.mean_focus(0, 0.5, 40) == pytest.approx(fit.mean_focus(0, 1.4, 40))
    dof15=sizing.depth_of_field(fit, L=15., foc_crit=.09)
    dof100=sizing.depth_of_field(fit, L=99.4, foc_crit=.09)
    assert dof100['total_mm'] > dof15['total_mm']
    restored = sizing.SizingModel.from_json(fit.to_json())
    assert restored.mean_focus(.2, 1, 40) == pytest.approx(fit.mean_focus(.2, 1, 40))
    fig=sizing.make_figure(fit, table)
    assert len(fig.axes) == 7
    assert 'signed distance' in fig.axes[1].get_xlabel()
    report=sizing.format_report(fit)
    assert 'shadow_linear' in report and 'Depth of field' in report


def test_cannot_mix_incompatible_measurements():
    table=sizing.CalibrationTable.from_rows(make_rows(), 'shadow_deq', 'focus_per_illum')
    with pytest.raises(ValueError, match="focus / local illumination"):
        sizing.fit_model(sizing.CalibrationTable.from_rows(make_rows(), 'shadow_deq', 'focus'),
                         size_model='shadow_linear', focus_model='normalised_generalised')
    with pytest.raises(ValueError, match="signed-shadow"):
        sizing.fit_model(sizing.CalibrationTable.from_rows(make_rows(), 'length', 'focus_per_illum'),
                         size_model='shadow_linear')


def test_shadow_circle_and_local_illumination_are_independent():
    shape=(90, 90)
    yy, xx=np.indices(shape)
    bg=np.full(shape, 110.)
    im=2*bg.copy()
    im[(xx-45)**2+(yy-45)**2 < 12**2] *= .3
    angles=np.linspace(0, 2*np.pi, 70)
    bound=np.column_stack((45+12*np.sin(angles),45+12*np.cos(angles)))
    a=measure(im,bg,bound,padding_px=8.,pixel_um=1,denoise=0)
    b=measure(im*.5,bg,bound,padding_px=8.,pixel_um=1,denoise=0)
    assert a['shadow_diameter']==pytest.approx(b['shadow_diameter'],rel=.005)
    assert local_illumination(im,bg,bound) == pytest.approx(220, rel=.01)


def test_processing_stores_shadow_and_illum_without_overwriting_shape(tmp_path):
    from scipy.io import savemat, loadmat
    from proc.imageStats import imageStats
    from proc.edgeDetect import EdgeOptions
    from calibration.data import scan_processed_file
    yy, xx = np.indices((90,90))
    bg = np.full((90,90), 140., float)
    img = np.where((yy-45)**2+(xx-45)**2 <= 18**2, 25, 140).astype(np.uint8)
    imgs = np.empty((1,1),dtype=[('IM',object)])
    bgs = np.empty((1,1),dtype=[('BG',object)])
    imgs[0,0]['IM'] = img
    bgs[0,0]['BG'] = bg.astype(np.uint8)
    roi={'IMAGE':imgs,'imageType':np.array([[89]]),'Time':np.array([[1.]]),
         'StartX':np.array([[0.]]),'StartY':np.array([[0.]]),
         'EndX':np.array([[90.]]),'EndY':np.array([[90.]])}
    source=tmp_path/'proc.mat'
    savemat(source,{'ROI_N':roi,'BG':bgs})
    raw=loadmat(source)
    opt=EdgeOptions(method='transmittance',edge_mode='relative',edge_fraction=.55,
                    bg_check=False,shadow_circle_padding=8.,pixel_um=2.46)
    dat=imageStats(raw['ROI_N'],raw['BG'],False,True,0,'synthetic',None,options=opt)
    assert float(dat['shadow_deq'][0,0]) == pytest.approx(36*2.46*np.sqrt(1-25/140),abs=4)
    assert float(dat['len'][0,0]) > 0
    assert float(dat['local_illum_bg'][0,0]) == pytest.approx(140,rel=.04)
    assert float(dat['shadow_radius_px'][0,0]) > 18
    # The parser reads the additional numeric field; no fallback to the legacy diameter.
    savemat(source,{'ROI_N':raw['ROI_N'],'BG':raw['BG'],'dat':dat})
    records=scan_processed_file(source,tmp_path)
    assert len(records)==1 and records[0].shadow_deq_um > 0
    assert records[0].local_illum == pytest.approx(140,rel=.04)
