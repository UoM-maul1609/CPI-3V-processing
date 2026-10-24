import math
import numpy as np
import cv2
from calibration.microscopy import MicroscopyStore, enclosing_circle, measure, detect_candidates


def test_polygon_circle_and_physical_scale():
    ann = {'mode': 'polygon', 'boundary': [[10,10],[30,10],[30,20],[10,20]],
           'scale_line': [[0,0],[20,0]], 'scale_um': 10}
    ann['enclosing_circle'] = enclosing_circle(ann)
    m = measure(ann)
    assert math.isclose(m['max_dimension_px'], math.hypot(20,10), rel_tol=1e-5)
    assert math.isclose(m['area_um2'], 50)
    assert all(math.dist(p, ann['enclosing_circle'][:2]) <= ann['enclosing_circle'][2] for p in ann['boundary'])
    assert math.isclose(m['area_over_circle'], 200/m['enclosing_circle_area_px2'])


def test_manual_circle_and_uncalibrated():
    ann = {'mode':'circle', 'boundary_circle':[45,55,20]}
    ann['enclosing_circle'] = enclosing_circle(ann)
    m = measure(ann)
    assert math.isclose(m['max_dimension_px'], 40)
    assert math.isclose(m['area_over_circle'], 1)
    assert m['max_dimension_um'] is None


def test_detect_object_without_accepting():
    im = np.full((300,400), 240, np.uint8)
    cv2.circle(im,(175,155),54,20,-1)
    contours = detect_candidates(im)
    assert contours
    assert any(cv2.pointPolygonTest(np.asarray(c,np.float32), (175.,155.), False) >= 0
               for c in contours)


def test_library_microscopy_persists_without_editing_original(tmp_path):
    image = tmp_path/'sem.png'
    source = np.full((20,30), 120, np.uint8)
    cv2.imwrite(str(image),source)
    library = tmp_path/'labels.json'
    store = MicroscopyStore(library)
    entry = store.import_image(image, 'Ice analogue A')
    entry['mode'] = 'circle'
    entry['boundary_circle'] = [10, 10, 5]
    entry['enclosing_circle'] = enclosing_circle(entry)
    entry['scale_line'] = [[0,0],[10,0]]
    entry['scale_um'] = 20
    store.update(entry)
    reread = MicroscopyStore(library).list_for('Ice analogue A')[0]
    assert math.isclose(reread['measurements']['max_dimension_um'],20)
    assert reread['original_filename'] == 'sem.png'
    assert cv2.imread(str(image)).shape[:2] == source.shape
    store.delete(entry['id'])
    assert not store.list_for('Ice analogue A')
    assert store.image_path(entry).exists()


def test_multiple_images_and_label_aliases(tmp_path):
    source = tmp_path / 'a.tif'
    cv2.imwrite(str(source), np.ones((60,90), dtype=np.uint16)*1000)
    store = MicroscopyStore(tmp_path/'labels.json')
    first = store.import_image(source, 'Old label')
    second = store.import_image(source, 'New label')
    assert first['file'] != second['file']
    assert len(MicroscopyStore(tmp_path/'labels.json').list_for('New label', ['Old label'])) == 2


def test_circle_is_non_destructive_to_polygon_measurement():
    data = {'mode':'polygon', 'boundary':[[4,4],[100,4],[110,17],[90,31],[6,31]]}
    before = measure(data)
    data['enclosing_circle'] = enclosing_circle(data)
    after = measure(data)
    assert before['area_px2'] == after['area_px2']
    assert 0 < after['area_over_circle'] <= 1.00001


def test_detection_large_image_native_pixel_coordinates():
    image = np.full((2500, 3100), 240, dtype=np.uint8)
    cv2.circle(image, (1800, 1500), 240, 25, -1)
    contours = detect_candidates(image)
    assert contours
    assert any(cv2.pointPolygonTest(np.asarray(c,np.float32), (1800.,1500.), False) >= 0
               for c in contours)
    assert any(max(p[0] for p in c) > 2000 for c in contours)


def test_physical_scale_changes_area_quadratically():
    ann = {'mode':'polygon', 'boundary':[[0,0],[40,0],[40,30],[0,30]],
           'scale_line':[[0,0],[10,0]], 'scale_um':10}
    m1 = measure(ann)
    ann['scale_um'] = 20
    m2 = measure(ann)
    assert m2['max_dimension_um'] == 2*m1['max_dimension_um']
    assert m2['area_um2'] == 4*m1['area_um2']
