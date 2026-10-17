import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from calibration.metadata import read_truth_table, attach_truth
from calibration.data import ParticleRecord


def test_truth_table(tmp_path):
    file = tmp_path / 'truth.csv'
    with file.open('w', newline='') as stream:
        w = csv.DictWriter(stream, ['source_file', 'actual_length_um', 'object_plane_um', 'instrument'])
        w.writeheader()
        w.writerow(dict(source_file='sample.mat', actual_length_um=100, object_plane_um=-200, instrument='CPI1.0'))
    records = [ParticleRecord('sample.mat', 0, 0, 40, 30, 0.3, 0.5, 1, 2, 3, 4)]
    count, unmatched = attach_truth(records, read_truth_table(file))
    assert count == 1 and not unmatched
    assert records[0].length_um == 40  # measured value unchanged
    assert records[0].focus == 0.3
    assert records[0].actual_length_um == 100
    assert records[0].object_plane_um == -200
    assert records[0].instrument == 'CPI1.0'
