from types import SimpleNamespace
import numpy as np
from calibration.focus_surface import fit_surface, predict_best_radius


def test_known_tilted_surface():
    rng=np.random.default_rng(8)
    records=[]
    for _ in range(600):
        u=rng.uniform(0,1024);v=rng.uniform(0,768);r=rng.uniform(3,6)
        rbest=4.5+0.45*(u-512)/1024-0.35*(v-384)/768
        focus=100-30*(r-rbest)**2+rng.normal(0,0.3)
        records.append(SimpleNamespace(stage_x=r,stage_y=0,image_x_px=u,image_y_px=v,focus=focus))
    fit=fit_surface(records).as_dict()
    assert abs(predict_best_radius(fit,512,384)-4.5)<0.08
    assert abs((predict_best_radius(fit,1024,384)-predict_best_radius(fit,0,384))-0.45)<0.08
    assert abs((predict_best_radius(fit,512,768)-predict_best_radius(fit,512,0))+0.35)<0.08
