"""Choosing which full-frame backgrounds the processing uses (backgroundSelection / backgroundSelector)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROC = Path(__file__).resolve().parents[1] / "python" / "proc"
sys.path.insert(0, str(PROC))

import backgroundSelection as bs  # noqa: E402
from associateBackgrounds import associateBackgrounds  # noqa: E402
from ROIDataDriver import _as_matlab_struct  # noqa: E402

T0 = 731900.5                   # a MATLAB datenum
SEC = 1 / 86400.0


def make_folder(tmp_path, n=4, spot_in=(1,)):
    """full_backgrounds.mat with n 60x80 frames, 60 s apart; a dark particle in the frames listed in spot_in."""
    rng = np.random.default_rng(1)
    frames = []
    for j in range(n):
        f = np.clip(150 + rng.normal(0, 2, (60, 80)) + 10 * j, 0, 255)
        if j in spot_in:
            f[20:30, 30:40] = 30
        frames.append(f.astype(np.uint8))
    times = T0 + 60 * SEC * np.arange(n)
    full = _as_matlab_struct({"Time": times[:, None], "IMAGE": [{"BG": f} for f in frames]})
    sio.savemat(tmp_path / "full_backgrounds.mat", {"FULL_BG": full, "t_range": np.array([[times[0], times[-1]]])})
    return times, frames, full


def test_load_frames_and_spots(tmp_path):
    times, frames, _ = make_folder(tmp_path)
    t, f = bs.load_frames(str(tmp_path))
    assert np.allclose(t, times) and len(f) == 4 and np.array_equal(f[2], frames[2])
    counts, ratios = bs.dark_spots(f)
    assert counts[1] >= 1 and counts[0] == counts[2] == counts[3] == 0
    assert ratios[1].shape == (15, 20)


def test_selection_round_trip_by_time(tmp_path):
    times, _, _ = make_folder(tmp_path)
    bs.save_selection(str(tmp_path), times, [True, False, True, True])
    rec = bs.read_selection(bs.selection_path(str(tmp_path)))
    assert rec["n_used"] == 3 and rec["frames"][1]["use"] is False
    # after reprocessing: same backgrounds (times within 0.1 s), one new frame, order changed
    new_times = np.array([times[3], times[1] + 0.1 * SEC, times[0], T0 + 999 * SEC])
    assert list(bs.mask_from_selection(rec, new_times)) == [True, False, True, True]


def test_active_mask_follows_environment(tmp_path, monkeypatch):
    times, _, _ = make_folder(tmp_path)
    monkeypatch.delenv(bs.ENV_VAR, raising=False)
    assert bs.active_mask(times) is None
    path = bs.save_selection(str(tmp_path), times, [True, False, True, False])
    monkeypatch.setenv(bs.ENV_VAR, path)
    assert list(bs.active_mask(times)) == [True, False, True, False]
    bs.save_selection(str(tmp_path), times, [False] * 4)
    with pytest.raises(ValueError):
        bs.active_mask(times)


def _wrap(a):
    o = np.empty((1, 1), object)
    o[0, 0] = a
    return o


def roi_n(times, corners):
    """The parts of a loaded ROI_N struct that associateBackgrounds reads."""
    n = len(times)
    sx, ex, sy, ey = (np.array([[c[k] for c in corners]], float) for k in range(4))
    return {"Time": _wrap(np.asarray(times, float)[:, None]), "IMAGE": _wrap(np.zeros((1, n), object)),
            "StartX": _wrap(sx), "EndX": _wrap(ex), "StartY": _wrap(sy), "EndY": _wrap(ey)}


def test_associate_uses_only_selected(tmp_path, monkeypatch):
    times, frames, full = make_folder(tmp_path)
    monkeypatch.delenv(bs.ENV_VAR, raising=False)
    rois = roi_n([times[1], times[1] + 25 * SEC], [(20, 29, 30, 39), (0, 9, 0, 9)])
    bg_all = associateBackgrounds(rois, full)
    assert np.array_equal(bg_all[0]["BG"], frames[1][20:30, 30:40])          # nearest: the frame with the particle
    path = bs.save_selection(str(tmp_path), times, [True, False, True, True])
    monkeypatch.setenv(bs.ENV_VAR, path)
    bg_sel = associateBackgrounds(rois, full)
    assert np.array_equal(bg_sel[0]["BG"], frames[0][20:30, 30:40]) or \
        np.array_equal(bg_sel[0]["BG"], frames[2][20:30, 30:40])               # nearest ticked one instead
    assert np.array_equal(bg_sel[1]["BG"], frames[2][0:10, 0:10])               # 85 s: frame 2 (120 s) is nearest ticked


def test_bank_uses_only_selected(tmp_path, monkeypatch):
    from backgroundCheck import BackgroundBank
    times, _, _ = make_folder(tmp_path)
    monkeypatch.delenv(bs.ENV_VAR, raising=False)
    assert len(BackgroundBank(str(tmp_path))) == 4
    monkeypatch.setenv(bs.ENV_VAR, bs.save_selection(str(tmp_path), times, [True, False, False, True]))
    bank = BackgroundBank(str(tmp_path))
    assert len(bank) == 2 and np.allclose(bank.times, times[[0, 3]])


def test_window_saves_and_offers_previous(tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    import backgroundSelector
    times, _, _ = make_folder(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def accept_with_spotty_unticked(dialog):
        # press "Untick those with more spots than" (0) then accept
        for b in dialog.findChildren(QtWidgets.QPushButton):
            if b.text().startswith("Untick those"):
                b.click()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", accept_with_spotty_unticked)
    assert backgroundSelector.run(str(tmp_path)) == 0
    rec = bs.read_selection(bs.selection_path(str(tmp_path)))
    assert [f["use"] for f in rec["frames"]] == [True, False, True, True]

    # second time: asked first; "Use previous selection" keeps it without the window
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda box: 0)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton",
                        lambda box: next(b for b in box.buttons() if b.text().startswith("Use previous")))
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda d: pytest.fail("window should not open"))
    assert backgroundSelector.run(str(tmp_path)) == 0
    assert [f["use"] for f in bs.read_selection(bs.selection_path(str(tmp_path)))["frames"]] == [True, False, True, True]
    # cancel -> status 1
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", lambda box: None)
    assert backgroundSelector.run(str(tmp_path)) == 1


def test_batch_option_saved_requires_a_selection(tmp_path):
    make_folder(tmp_path)
    (tmp_path / "a.mat").write_bytes(b"")
    r = subprocess.run([sys.executable, str(PROC / "batchProcess.py"), str(tmp_path), "--only-stats",
                        "--backgrounds", "saved"], capture_output=True, text=True)
    assert r.returncode != 0 and "No saved background selection" in r.stderr
