import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "proc"))
from edgeDetect import EdgeOptions, measure_transmittance, feret, polygon_area  # noqa: E402


def shape_image(mask_fn, size=96, blur=1.0, level=150.0, gradient=0.0, core=0.1, seed=0):
    """Image and background of an opaque object (transmittance `core` inside) with blur and noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size * 4, :size * 4] / 4.0
    inside = mask_fn(yy, xx).reshape(size, 4, size, 4).mean((1, 3))
    T = 1.0 - (1.0 - core) * inside
    T = ndimage.gaussian_filter(T, blur)
    bg = level * (1.0 + gradient * (np.arange(size)[None, :] / size - 0.5)) * np.ones((size, size))
    image = np.clip(bg * T + rng.normal(0, 1.5, bg.shape), 0, 255)
    return image, np.clip(bg + rng.normal(0, 1.0, bg.shape), 1, 255)


def disc(d, c=48.0):
    return lambda y, x: (np.hypot(x - c, y - c) < d / 2).astype(float)


OPT = EdgeOptions(method="transmittance", edge_level=0.5, pixel_um=1.0)


@pytest.mark.parametrize("d", [8.0, 17.4, 43.5])
def test_size_is_right_and_independent_of_illumination(d):
    lengths = []
    for level, gradient in ((150, 0.0), (90, 0.0), (200, 0.3)):
        im, bg = shape_image(disc(d), level=level, gradient=gradient)
        r = measure_transmittance(im, bg, OPT)
        lengths.append(r["len"])
    assert lengths[0] == pytest.approx(d, abs=0.8)                  # no fixed +2 px pad
    assert max(lengths) - min(lengths) < 0.4                        # laser power / gradient don't matter


def test_complex_shape_keeps_its_corners():
    # three overlapping discs (like the stuck-together beads in a real ROI)
    three = lambda y, x: ((np.hypot(x - 30, y - 30) < 14) | (np.hypot(x - 52, y - 40) < 14)
                          | (np.hypot(x - 36, y - 62) < 14)).astype(float)
    im, bg = shape_image(three, blur=0.6)
    r = measure_transmittance(im, bg, OPT)
    true_area = three(*np.mgrid[:384, :384] / 4.0).sum() / 16.0
    assert r["n_boundary"] > 30
    assert r["area"] == pytest.approx(true_area, rel=0.05)
    b = r["boundary"]
    assert np.allclose(b[0], b[-1])                                 # stored closed


def test_bright_centre_is_filled_and_edge_touch_flagged():
    ring = lambda y, x: ((np.hypot(x - 48, y - 48) < 20) & ~(np.hypot(x - 48, y - 48) < 4)).astype(float)
    im, bg = shape_image(ring)
    r = measure_transmittance(im, bg, OPT)
    assert r["len"] == pytest.approx(40, abs=1.0) and r["T_centre"] > 0.5 and r["edge_touch"] == 0
    im, bg = shape_image(disc(40, c=10))
    assert measure_transmittance(im, bg, OPT)["edge_touch"] == 1


def test_feret_of_rectangle():
    pts = np.array([[0, 0], [0, 30], [10, 30], [10, 0]], float)
    length, width, _ = feret(pts)
    assert length == pytest.approx(np.hypot(10, 30)) and width == pytest.approx(10)
    assert polygon_area(pts) == pytest.approx(300)


def test_image_stats_records_options_and_variable_boundaries(tmp_path):
    from imageStats import imageStats
    n = 3
    imgs = np.empty((1, n), dtype=[("IM", object)])
    bgs = np.empty((1, n), dtype=[("BG", object)])
    for i, d in enumerate((20.0, 30.0, 40.0)):
        im, bg = shape_image(disc(d), seed=i)
        imgs[0, i]["IM"] = im.astype(np.uint8)
        bgs[0, i]["BG"] = bg.astype(np.uint8)
    roi = {"IMAGE": imgs, "imageType": np.full((n, 1), 89), "Time": np.arange(n, dtype=float)[:, None],
           "StartX": np.full((1, n), 100.0), "StartY": np.full((1, n), 100.0)}
    sio.savemat(tmp_path / "x.mat", {"ROI_N": roi, "BG": bgs})
    m = sio.loadmat(tmp_path / "x.mat")
    opts = {"method": "transmittance", "edge_level": 0.5, "pixel_um": 2.3}
    dat = imageStats(m["ROI_N"], m["BG"], False, True, 0, "test", None, options=json.dumps(opts))
    assert np.ravel(dat["len"]) == pytest.approx([20 * 2.3, 30 * 2.3, 40 * 2.3], abs=2.3)
    rec = json.loads(dat["proc_options"])
    assert rec["method"] == "transmittance" and rec["edge_level"] == 0.5 and "code_version" in rec
    sizes = [len(dat["foc"][i]["boundaries"]) for i in range(n)]
    assert len(set(sizes)) > 1                                      # variable length
    legacy = imageStats(m["ROI_N"], m["BG"], False, True, 0, "test", None)
    assert json.loads(legacy["proc_options"])["method"] == "legacy"


def test_relative_mode_keeps_faint_small_particles():
    fixed = EdgeOptions(method="transmittance", edge_level=0.5, pixel_um=1.0)
    rel = EdgeOptions(method="transmittance", edge_mode="relative", edge_fraction=0.5, pixel_um=1.0)
    im, bg = shape_image(disc(10.0), blur=3.5)          # shallow shadow: darkest T ~ 0.4
    assert measure_transmittance(im, bg, fixed) is not None  # fixed mode no longer imposes T < 0.45 core
    r = measure_transmittance(im, bg, rel)
    assert r is not None and r["len"] == pytest.approx(10.0, abs=1.5) and r["edge_level_used"] > 0.6
    sizes = [measure_transmittance(*shape_image(disc(17.4), blur=b), rel)["len"] for b in (0.7, 1.5, 2.5, 3.5)]
    assert max(sizes) - min(sizes) < 1.2                  # consistent through defocus


def test_radial_outline_for_round_particles_and_fallback_for_other_shapes():
    import numpy as np
    from edgeDetect import EdgeOptions, measure_transmittance
    yy, xx = np.mgrid[:80, :80]
    bg = np.full((80, 80), 200.0)
    # a disc with a bright spot inside: radial outline, measured close to its true diameter
    disc = np.where(np.hypot(yy - 40, xx - 40) < 15, 60.0, 200.0)
    disc[np.hypot(yy - 40, xx - 43) < 3] = 170.0
    opt = EdgeOptions(method="transmittance", edge_mode="relative", edge_fraction=0.65, pixel_um=1.0)
    r = measure_transmittance(disc, bg, opt)
    assert r["outline_radial"] == 1
    assert abs(r["deq"] - 30) < 1.5 and abs(r["len"] - 30) < 2.0
    # a thin ring arc (not star-shaped from its centroid): falls back to the raw contour
    rr = np.hypot(yy - 40, xx - 40)
    arc = np.where((rr > 22) & (rr < 27) & (xx < 52), 60.0, 200.0)
    r2 = measure_transmittance(arc, bg, opt)
    assert r2 is not None and r2["outline_radial"] == 0
    r3 = measure_transmittance(arc, bg, EdgeOptions(method="transmittance", edge_mode="relative", outline="contour"))
    assert r3["outline_radial"] == 0


def test_lopsided_bead_gets_its_whole_outline():
    """A bead darker on one side (and with a bright centre) is outlined all round, not as a horseshoe."""
    import numpy as np
    from edgeDetect import EdgeOptions, measure_transmittance
    yy, xx = np.mgrid[:80, :80]
    r = np.hypot(yy - 40, xx - 40)
    bg = np.full((80, 80), 200.0)
    transmittance = 0.1 + 0.55 * (55 - yy) / 30                  # T 0.1 at the bottom, 0.65 at the top
    im = np.where(r < 15, 200.0 * transmittance, 200.0)
    im[r < 4] = 190.0                                              # bright centre spot
    opt = EdgeOptions(method="transmittance", edge_mode="relative", edge_fraction=0.65, pixel_um=1.0)
    res = measure_transmittance(im, bg, opt)
    assert res["outline_radial"] == 1
    assert abs(res["deq"] - 30) < 2.5 and abs(res["len"] - 30) < 2.5
    raw = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative",
                                                    edge_fraction=0.65, pixel_um=1.0, outline="contour"))
    assert raw["deq"] < res["deq"] - 3                              # the raw contour misses the grey side


def test_steepest_rise_outline_option():
    import numpy as np
    from scipy import ndimage
    from edgeDetect import EdgeOptions, measure_transmittance
    yy, xx = np.mgrid[:80, :80]
    r = np.hypot(yy - 40, xx - 40)
    bg = np.full((80, 80), 200.0)
    # grey disc of radius 15 with a much darker core: the threshold outline sits inside, the edge option on the rim
    T = np.where(r < 15, 0.6, 1.0)
    T[r < 8] = 0.1
    im = 200.0 * ndimage.gaussian_filter(T, 1.0)
    edge = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative", pixel_um=1.0,
                                                     outline="edge"))
    thr = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative", pixel_um=1.0))
    assert edge["outline_radial"] == 1
    assert abs(edge["deq"] - 30) < 2.0
    assert thr["deq"] < edge["deq"] - 5


def test_particle_in_its_own_background_is_replaced():
    import numpy as np
    from backgroundCheck import choose_background, SOURCE_ASSIGNED, SOURCE_FITTED, SOURCE_OTHER_FRAME, BackgroundBank
    from edgeDetect import EdgeOptions, measure_transmittance
    yy, xx = np.mgrid[:80, :80]
    illum = 200.0 * (1 + 0.1 * (xx - 40) / 40)                    # smooth laser profile
    disc = np.hypot(yy - 40, xx - 40) < 15
    image = np.where(disc, 0.15 * illum, illum)
    contaminated = np.where(np.hypot(yy - 41, xx - 39) < 15, 0.15 * illum, illum)   # same particle, moved 1 px
    opt = EdgeOptions(method="transmittance", edge_mode="relative", pixel_um=1.0)
    bad = measure_transmittance(image, contaminated, opt)
    assert bad is None or bad["deq"] < 20                          # what happened before: a sliver
    bg, source, _ = choose_background(image, contaminated, None, 0.0, (0, 80, 0, 80))
    assert source == SOURCE_FITTED
    good = measure_transmittance(image, bg, opt)
    assert abs(good["deq"] - 30) < 2.5
    # a clean background is left alone
    bg2, source2, _ = choose_background(image, illum, None, 0.0, (0, 80, 0, 80))
    assert source2 == SOURCE_ASSIGNED and bg2 is illum
    # with a bank of full frames, the nearest clean frame is used
    bank = BackgroundBank.__new__(BackgroundBank)
    bank.times = np.array([0.0, 0.001])
    full_clean = np.full((200, 200), 200.0); full_clean[50:130, 60:140] = illum
    full_dirty = full_clean.copy(); full_dirty[50:130, 60:140] = contaminated
    bank.frames = [full_dirty, full_clean]
    bg3, source3, dt = choose_background(image, contaminated, bank, 0.0, (50, 130, 60, 140))
    assert source3 == SOURCE_OTHER_FRAME and np.allclose(bg3, illum) and dt > 0


def _rotated_bar_mask(cx, cy, length, width, angle_deg, size):
    yy, xx = np.mgrid[:size, :size] + 0.5
    a = np.radians(angle_deg)
    u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    return (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2), u, v


def _scene(T, level=180.0, seed=3):
    rng = np.random.default_rng(seed)
    T = ndimage.gaussian_filter(T, 0.8)
    bg = np.full(T.shape, level)
    return np.clip(bg * T + rng.normal(0, 1.5, T.shape), 0, 255), np.clip(bg + rng.normal(0, 1.0, T.shape), 1, 255)


def test_transparent_column_with_bright_axial_streak_is_measured_whole():
    """A column focuses light into a bright streak along its axis; the outline must cover both halves."""
    size = 120
    inside, u, v = _rotated_bar_mask(60, 60, 80, 24, 40, size)
    T = np.where(inside, 0.25, 1.0)
    # bright focused streak (~11 px wide) between two dark bars; at the ends it is only grey, so the
    # shadow is one piece at the weak level but two bars at the edge level -- as in the CPI images
    streak = inside & (np.abs(v) < 5.5)
    T = np.where(streak, np.where(np.abs(u) < 34, 1.25, 0.75), T)
    im, bg = _scene(T)
    opt = EdgeOptions(method="transmittance", edge_mode="relative", edge_fraction=0.55, pixel_um=1.0)
    old = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative",
                                                    edge_fraction=0.55, pixel_um=1.0, merge_parts=False, fill_bays=0))
    new = measure_transmittance(im, bg, opt)
    assert old["wid"] < 16                       # before: one half only
    assert abs(new["wid"] - 24) < 3 and abs(new["len"] - 80) < 4


def test_separate_parallel_needles_are_not_joined():
    size = 120
    a, _, _ = _rotated_bar_mask(60, 48, 80, 8, 0, size)
    b, _, _ = _rotated_bar_mask(60, 72, 80, 8, 0, size)      # 16 px of background between them
    im, bg = _scene(np.where(a, 0.2, np.where(b, 0.3, 1.0)))
    res = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative",
                                                    edge_fraction=0.55, pixel_um=1.0))
    assert res["wid"] < 12                       # one needle, not both


def test_rosette_arms_are_not_filled_in():
    size = 140
    arms = np.zeros((size, size), bool)
    for ang in (0, 60, 120):
        m, _, _ = _rotated_bar_mask(70, 70, 100, 12, ang, size)
        arms |= m
    im, bg = _scene(np.where(arms, 0.2, 1.0))
    res = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative",
                                                    edge_fraction=0.55, pixel_um=1.0))
    area_true = arms.sum()
    assert res["area"] < 1.25 * area_true          # the wedges between arms stay out


def test_column_far_from_focus_two_walls_faint_caps_are_joined():
    """Far from focus a column is two parallel dark walls with only faint grey end caps."""
    size = 130
    inside, u, v = _rotated_bar_mask(65, 65, 90, 26, 40, size)
    walls = inside & (np.abs(v) > 7)                       # 6 px walls, 14 px bright gap
    caps = inside & ~walls & (np.abs(u) > 41)              # faint grey caps joining them at the ends
    T = np.where(walls, 0.15, np.where(caps, 0.9, np.where(inside, 1.05, 1.0)))
    im, bg = _scene(T)
    opt = EdgeOptions(method="transmittance", edge_mode="relative", edge_fraction=0.55, pixel_um=1.0)
    old = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative",
                                                    edge_fraction=0.55, pixel_um=1.0, join_max_angle=-1))
    new = measure_transmittance(im, bg, opt)
    assert old["wid"] < 10                                  # one wall only
    assert abs(new["wid"] - 26) < 4 and abs(new["len"] - 90) < 5


def test_grey_channel_inside_a_column_is_filled():
    """Out of focus, part of a column's shadow is lighter than the edge level: a bay, not background."""
    size = 130
    inside, u, v = _rotated_bar_mask(65, 65, 90, 26, 40, size)
    # dark body, with a long grey channel (still shadow, lighter than the edge level) open at one end
    channel = inside & (v > -2) & (v < 6) & (u > -20)
    T = np.where(inside, np.where(channel, 0.8, 0.2), 1.0)
    im, bg = _scene(T)
    old = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative", edge_fraction=0.55,
                                                    pixel_um=1.0, fill_bays_shadow_cover=0, fill_bays_min_aspect=99))
    new = measure_transmittance(im, bg, EdgeOptions(method="transmittance", edge_mode="relative", edge_fraction=0.55,
                                                    pixel_um=1.0))
    true_area = inside.sum()
    assert old["area"] < 0.9 * true_area                       # the channel was cut out
    assert abs(new["area"] - true_area) < 0.08 * true_area
