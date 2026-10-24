import numpy as np
from proc.imageStats import _feret_dimensions
from calibration.data import ParticleRecord

def test_feret_rectangle_and_explicit_record_fields():
    boundary=np.array([[0,0],[0,40],[10,40],[10,0]])
    maximum, minimum=_feret_dimensions(boundary,2.3)
    assert np.isclose(maximum,np.hypot(10,40)*2.3)
    assert np.isclose(minimum,10*2.3)
    record=ParticleRecord('test.mat',0,0,50,20,0,1,0,0,0,0)
    assert np.isnan(record.len_feret_um)
    assert np.isnan(record.wid_ellipse_um)
