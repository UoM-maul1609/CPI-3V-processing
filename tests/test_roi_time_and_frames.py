"""MAT-file timestamp and frame classification regression tests."""
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.io import savemat
from calibration.data import scan_processed_file, datetime_to_matlab_datenum

def test_roi_time_preferred_over_dat_time_and_frames(tmp_path):
    t = datetime_to_matlab_datenum(datetime(2002,3,21,15,24))
    roi = {'Time': np.array([t,t+0.0001]), 'StartX':np.array([0,30]),
           'EndX':np.array([1023,100]),'StartY':np.array([0,50]),'EndY':np.array([1021,110])}
    dat = {'len':np.array([100.,120.]), 'wid':np.array([80.,70.]), 'Time':np.array([t+1,t+1]),
           'foc': np.array([{'focus':4.}, {'focus':5.}], dtype=object)}
    p=tmp_path/'synthetic.mat'
    savemat(p,{'ROI_N':roi,'dat':dat})
    rr=scan_processed_file(p)
    assert len(rr)==2
    assert abs(rr[0].time_matlab-t)<1e-8
    assert rr[0].is_full_frame and not rr[1].is_full_frame
