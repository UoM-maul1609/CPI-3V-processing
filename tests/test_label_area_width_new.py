import math
from calibration.labels import ParticleLabel, LabelLibrary
from calibration.microscopy import measure


def test_label_unset_width_and_reference_area_round_trip(tmp_path):
    lib = LabelLibrary(tmp_path / 'labels.json')
    lib.upsert(ParticleLabel('analogue',kind='ice analogue',length_um=200,area_um2=3456.789))
    lib.save()
    label = LabelLibrary.load(lib.path).get('analogue')
    assert math.isnan(label.width_um)
    assert label.area_um2 == 3456.789
    assert 'unset' in label.size_text()


def test_microscope_minor_axis_and_exact_scale():
    ann = {'mode':'polygon', 'boundary':[(0,0),(100,0),(100,30),(0,30)],
           'scale_line':[(5,5),(105,5)],'scale_um':2000.0}
    m=measure(ann)
    assert m['um_per_px'] * 100 == 2000.0
    assert 30 <= m['cpi_minor_axis_width_px'] < 40
    assert m['cpi_minor_axis_width_um'] == m['cpi_minor_axis_width_px']*20
    assert m['area_um2'] > 0


def test_microscopy_feret_width_matches_rectangle_and_keeps_ellipse_width():
    m = measure({'mode': 'polygon', 'boundary': [(0, 0), (100, 0), (100, 30), (0, 30)],
                 'scale_line': [(0, 0), (100, 0)], 'scale_um': 2000.0})
    assert math.isclose(m['feret_min_width_px'], 30, abs_tol=1e-5)
    assert math.isclose(m['feret_min_width_um'], 600, abs_tol=1e-4)
    assert m['cpi_minor_axis_width_px'] > m['feret_min_width_px']


def test_microscopy_circle_feret_width_is_diameter():
    m = measure({'mode':'circle', 'boundary_circle':[20, 20, 8]})
    assert math.isclose(m['feret_min_width_px'], 16)
