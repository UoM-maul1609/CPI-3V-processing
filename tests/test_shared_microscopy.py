"""Shared graticule photo: independent outlines and two-axis calibration."""
from pathlib import Path
import math

import pytest

from calibration.labels import LabelLibrary
from calibration.microscopy import (
    SharedMicroscopyStore, measure_shared_object, physical_scales, enclosing_circle,
)


def test_anisotropic_graticule_and_physical_contour(tmp_path):
    image = {
        'scale_line': [[0, 0], [100, 0]], 'scale_um': 200.,
        'scale_line_y': [[0, 0], [0, 100]], 'scale_um_y': 100.,
    }
    assert physical_scales(image) == pytest.approx((2., 1.))
    obj = {'mode': 'polygon', 'boundary': [[0,0], [40,0], [40,20], [0,20]]}
    obj['enclosing_circle'] = enclosing_circle(obj)
    m = measure_shared_object(image, obj)
    assert m['max_dimension_um'] == pytest.approx(math.hypot(80.,20.), rel=1e-5)
    assert m['feret_min_width_um'] == pytest.approx(20., rel=1e-5)
    assert m['area_um2'] == pytest.approx(1600., rel=1e-5)
    assert m['cpi_minor_axis_width_um'] == pytest.approx(4*20/math.sqrt(12), rel=.02)
    assert 0 < m['area_over_circle'] < 1


def test_single_graticule_calibrates_both_axes():
    image = {'scale_line': [[12,9],[12,109]], 'scale_um': 200.,
             'scale_line_y': None, 'scale_um_y': None}
    assert physical_scales(image) == pytest.approx((2, 2))
    assert physical_scales({'scale_line':None, 'scale_um':None}) is None


def test_shared_image_objects_roundtrip_and_no_mutation_of_reference(tmp_path):
    labels = LabelLibrary(tmp_path/'labels.json')
    src = tmp_path/'graticule.png'
    src.write_bytes(b'original image bytes copied verbatim')
    store=SharedMicroscopyStore(labels.path)
    photo=store.import_image(src)
    assert store.image_path(photo).read_bytes() == src.read_bytes()
    photo['scale_line']=[[0,0],[100,0]]
    photo['scale_um']=250.0
    photo['objects']=[
        {'object_id':'one','name':'upper plate','label':'plag102','mode':'polygon',
         'boundary':[[0,0],[5,0],[5,5],[0,5]],'enclosing_circle':None,
         'scale_line':[[0,0],[1,1]],'scale_um':42.0},
        {'object_id':'two','name':'thin column','label':'col50','mode':'polygon',
         'boundary':[[0,0],[10,0],[10,1],[0,1]],'enclosing_circle':None},
    ]
    store.update(photo)
    again=SharedMicroscopyStore(labels.path)
    assert len(again.entries) == 1
    assert [o['label'] for o in again.entries[0]['objects']] == ['plag102','col50']
    assert 'scale_line' not in again.entries[0]['objects'][0]
    assert again.entries[0]['scale_um'] == 250.0
    assert again.image_path(again.entries[0]).read_bytes() == src.read_bytes()
    assert not labels.path.exists()


def test_multiple_objects_are_measured_independently():
    image={'scale_line':None,'scale_um':None}
    a={'mode':'polygon','boundary':[[0,0],[20,0],[20,10],[0,10]]}
    b={'mode':'polygon','boundary':[[0,0],[12,0],[12,12],[0,12]]}
    ma=measure_shared_object(image,a)
    mb=measure_shared_object(image,b)
    assert ma['area_px2'] == pytest.approx(200.)
    assert mb['area_px2'] == pytest.approx(144.)
    assert ma['max_dimension_um'] is None
    assert mb['max_dimension_um'] is None


def test_rotated_two_way_graticule_is_accurate():
    # Grid is tilted 45 degrees, with 100 pixels per 200 µm in both axes.
    u = 100 / math.sqrt(2)
    photo = {'scale_line':[[0,0],[u,u]], 'scale_um':200,
             'scale_line_y':[[0,0],[-u,u]], 'scale_um_y':200}
    from calibration.microscopy import graticule_transform
    transform = graticule_transform(photo)
    assert transform @ [u,u] == pytest.approx([200,0])
    assert transform @ [-u,u] == pytest.approx([0,200])
    assert abs(float(__import__('numpy').linalg.det(transform))) == pytest.approx(4)


def test_parallel_graticule_lines_rejected():
    from calibration.microscopy import graticule_transform
    photo = {'scale_line':[[0,0],[100,0]], 'scale_um':200,
             'scale_line_y':[[0,2],[100,2]], 'scale_um_y':200}
    with pytest.raises(ValueError,match='parallel'):
        graticule_transform(photo)
