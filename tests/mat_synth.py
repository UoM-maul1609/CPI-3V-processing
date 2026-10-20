"""Small processed-MAT calibration scans (stage X sweep) for end-to-end tests."""
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import scipy.io as sio
from calibration.data import datetime_to_matlab_datenum


def make_scan(out, t0=datetime(2026, 3, 2, 10, 0, 0), seed=3, per_step=6, series="beads"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    steps = np.linspace(10.0, 12.0, 11)
    rows = []
    for k, sx in enumerate(steps):
        for j in range(per_step):
            t = t0 + timedelta(seconds=k * 30 + rng.uniform(1, 29))
            cx, cy = rng.uniform(100, 900), rng.uniform(100, 900)
            dz = sx - 11.0
            rows.append((t, cx, cy, 30 * np.exp(-(dz / 0.35) ** 2) + 5, 40 * (1 + 0.6 * dz ** 2)))
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    imgs = np.empty((n,), dtype=[("IM", object)])
    focs = np.empty((n,), dtype=[("focus", object)])
    for i, r in enumerate(rows):
        imgs[i]["IM"] = np.full((16, 16), 200, np.uint8)
        focs[i]["focus"] = float(r[3])
    times = np.array([datetime_to_matlab_datenum(r[0]) for r in rows])
    cx = np.array([r[1] for r in rows]); cy = np.array([r[2] for r in rows])
    roi = {"Time": times, "StartX": cx - 8, "EndX": cx + 8, "StartY": cy - 8, "EndY": cy + 8, "IMAGE": imgs}
    dat = {"len": np.array([r[4] for r in rows]), "wid": np.array([r[4] for r in rows]),
           "round": np.full(n, 0.9), "centroid": np.full((n, 2), 8.0), "foc": focs}
    sio.savemat(out / "scan.mat", {"ROI_N": roi, "dat": dat})
    with open(out / "stage.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["series", "time_basis", "start_time", "stage_x_original", "stage_z_original"])
        for k, sx in enumerate(steps):
            w.writerow([series, "roi", (t0 + timedelta(seconds=k * 30)).isoformat(sep=" "), sx, 5.0])
    return out
