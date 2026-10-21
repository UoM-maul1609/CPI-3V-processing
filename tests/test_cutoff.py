"""Cut-off by the ROI edge: computed per particle, filterable in processing and the workbench."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "proc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edgeDetect import EdgeOptions, cutoff_percent, measure_transmittance  # noqa: E402
from test_edge_detect import disc, shape_image  # noqa: E402

OPT = EdgeOptions(method="transmittance", edge_mode="relative", pixel_um=1.0)


def test_cutoff_zero_inside_and_grows_with_overlap():
    whole = measure_transmittance(*shape_image(disc(40)), OPT)
    assert whole["cutoff_pct"] == 0.0
    a = measure_transmittance(*shape_image(disc(40, c=18)), OPT)["cutoff_pct"]     # just over the edge
    b = measure_transmittance(*shape_image(disc(40, c=8)), OPT)["cutoff_pct"]      # well over it
    assert 0 < a < b


def test_cutoff_of_a_half_disc():
    yy, xx = np.mgrid[:60, :60]
    region = np.hypot(yy - 0, xx - 30) < 20          # centred on the top edge: half is cut off
    pct = cutoff_percent(region)
    assert 15 < pct < 45                             # the straight cut side is ~40/(40+63) of the outline


def test_filter_spec_cutoff_keeps_unknown():
    from calibration.data import FilterSpec, ParticleRecord
    r = ParticleRecord("a.mat", 0, 731000.0, 100, 100, 20, 1, 10, 10, 1, 1)
    spec = FilterSpec(cutoff_max=6.0)
    assert spec.matches(r)                           # NaN (processed before cut-off was recorded) passes
    r.cutoff_pct = 3.0
    assert spec.matches(r)
    r.cutoff_pct = 12.0
    assert not spec.matches(r)
    assert FilterSpec(cutoff_min=10).matches(r) and not FilterSpec(cutoff_min=20).matches(r)
