"""Project-local reference-library discovery and non-destructive migration."""
from pathlib import Path
import json
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from calibration.reference_library import cpi_library_path, copy_reference_library
from calibration.labels import LabelLibrary, ParticleLabel


def test_both_calibration_and_ice_analogues_share_root(tmp_path):
    root = tmp_path / 'CPI_cals'
    a = root / 'CPICalibration' / 'Cal210302_test' / '03211101.mat'
    b = root / 'ICEAnalogues' / 'CPI' / 'ice.mat'
    for p in (a, b):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    expected = root / 'ReferenceLibrary' / 'labels.json'
    assert cpi_library_path(a) == cpi_library_path(b) == expected
    assert cpi_library_path(root) == expected
    assert cpi_library_path(tmp_path / 'unrelated' / 'file.mat') is None


def test_migrate_labels_microscopy_annotations_and_images_without_deleting(tmp_path):
    old = tmp_path / 'old' / 'labels.json'
    library = LabelLibrary(old)
    library.upsert(ParticleLabel('Glass bead 40', length_um=40))
    library.save()
    microscope = old.parent / 'microscopy' / 'images'
    microscope.mkdir(parents=True)
    (microscope / 'a.tiff').write_bytes(b'not a real photo; migration byte check')
    (old.parent / 'microscopy' / 'index.json').write_text('{"annotations": [123]}')
    dest = tmp_path / 'CPI_cals' / 'ReferenceLibrary' / 'labels.json'
    copy_reference_library(old, dest)
    assert 'Glass bead 40' in LabelLibrary.load(dest).labels
    assert (dest.parent / 'microscopy' / 'images' / 'a.tiff').read_bytes() == (microscope / 'a.tiff').read_bytes()
    assert (dest.parent / 'microscopy' / 'index.json').read_text() == '{"annotations": [123]}'
    assert old.exists() and (microscope / 'a.tiff').exists()
    with pytest.raises(FileExistsError):
        copy_reference_library(old, dest)
    assert 'Glass bead 40' in LabelLibrary.load(dest).labels


def test_empty_source_can_be_imported_without_making_new_files(tmp_path):
    src = tmp_path / 'missing' / 'labels.json'
    dest = tmp_path / 'CPI_cals' / 'ReferenceLibrary' / 'labels.json'
    copy_reference_library(src, dest)
    assert not dest.exists()
