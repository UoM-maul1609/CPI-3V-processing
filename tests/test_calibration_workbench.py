"""Core invariants for file-based calibration selection."""
from datetime import datetime
import numpy as np
from calibration.data import ParticleCatalog, ParticleRecord, PositionLog, FilterSpec, datetime_to_matlab_datenum

def particle(index, time, focus=10, size=100):
    return ParticleRecord('source.mat', index, datetime_to_matlab_datenum(time),
                          size, size, focus, 1., 20., 40., 10., 10.)

def test_step_hold_no_lookahead():
    t0=datetime_to_matlab_datenum(datetime(2002,3,21,15,20))
    t1=datetime_to_matlab_datenum(datetime(2002,3,21,15,21))
    log=PositionLog(np.array([t0,t1]),np.array([3.,4.]),np.array([5.,6.]))
    x,z=log.previous(np.array([t0-1/86400,t0,t0+30/86400,t1]))
    assert np.isnan(x[0])
    assert np.allclose(x[1:],[3,3,4])
    assert np.allclose(z[1:],[5,5,6])

def test_filter_reversible_and_flag_persistence(tmp_path):
    p=particle(0,datetime(2002,3,21,15,20),focus=5)
    p.actual_length_um=50
    cat=ParticleCatalog(tmp_path,[p])
    assert cat.filtered_indices(FilterSpec(focus_min=10))==[]
    assert p.status=='undecided'
    assert cat.filtered_indices(FilterSpec(ratio_min=1.9,ratio_max=2.1))==[0]
    p.status='reject'
    p.reject_reason='segmentation'
    cat.save_selection_file(tmp_path/'flags.csv')
    p.status='undecided'
    cat.apply_selection_file(tmp_path/'flags.csv')
    assert (p.status,p.reject_reason)==('reject','segmentation')
    p.status='undecided'
    p.reject_reason=''
    cat.save_selection_file(tmp_path/'flags.csv')
    p.status='reject'
    cat.apply_selection_file(tmp_path/'flags.csv')
    assert p.status=='undecided'


def test_filter_by_local_illumination(tmp_path):
    p = particle(0, datetime(2002, 3, 21, 15, 20)); p.local_illum = 60.0
    q = particle(1, datetime(2002, 3, 21, 15, 21)); q.local_illum = 180.0
    cat = ParticleCatalog(tmp_path, [p, q])
    assert cat.filtered_indices(FilterSpec(illum_min=90)) == [1]
