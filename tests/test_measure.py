"""Measuring distances on full frames / ROIs / image files (calibration.measure, measure_window)."""
import csv
import math
import os

import numpy as np
import pytest
import scipy.io as sio

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from calibration.measure import (Measurement, MatImages, constrain, nice_bar_length, profile)  # noqa: E402


def test_length_angle_and_um():
    m = Measurement(1.0, 2.0, 4.0, 6.0)
    assert m.length_px == pytest.approx(5.0)
    assert m.length_um(2.3) == pytest.approx(11.5)
    assert math.isnan(m.length_um(None))
    assert Measurement(0, 0, 10, 0).angle_deg == pytest.approx(0)
    assert Measurement(0, 10, 0, 0).angle_deg == pytest.approx(90)          # upwards on screen
    x, y = constrain(0, 0, 10, 1)
    assert (x, y) == pytest.approx((math.hypot(10, 1), 0))
    x, y = constrain(0, 0, 10, 9)
    assert x == pytest.approx(y)


def test_profile_finds_an_edge_between_pixels():
    img = np.zeros((20, 40))
    img[:, 20:] = 100.0                                    # edge at x = 20 (pixel boundary)
    s, v = profile(img, Measurement(10.0, 10.5, 30.0, 10.5), step=0.05)
    x_half = 10.0 + s[np.argmin(np.abs(v - 50.0))]
    assert x_half == pytest.approx(20.0, abs=0.06)


def test_nice_bar():
    assert nice_bar_length(37) == 20 and nice_bar_length(120) == 100 and nice_bar_length(6) == 5


def _mat(tmp_path):
    full = (np.arange(600 * 700).reshape(600, 700) % 251).astype(np.uint8)
    roi = np.full((30, 40), 200, np.uint8)
    roi[10:20, 12:28] = 20
    path = tmp_path / "f.mat"
    images = np.empty((1, 2), object)
    images[0, 0] = {"IM": roi}
    images[0, 1] = {"IM": full}
    sio.savemat(path, {"ROI_N": {"IMAGE": images, "Time": np.array([[731296.5], [731296.6]]),
                                 "imageType": np.array([[1], [19]])},
                       "dat": {"foc": np.array([{"boundaries": np.array([[10.0, 12.0], [10.0, 27.0],
                                                                         [19.0, 27.0], [19.0, 12.0]])},
                                                {"boundaries": np.zeros((0, 2))}], dtype=object)}})
    return path


def test_mat_images_lists_full_frames_and_boundaries(tmp_path):
    src = MatImages(_mat(tmp_path))
    kinds = [(e.index, e.kind, e.shape) for e in src.entries()]
    # shown in camera orientation: stored [x, y] arrays are transposed (rows = camera y)
    assert kinds == [(0, "ROI", (40, 30)), (1, "full frame", (700, 600))]
    assert src.image(0).shape == (40, 30)
    assert src.boundary(0).shape == (4, 2) and src.boundary(1) is None
    assert src.entries()[0].time.startswith("2002-03-21 12:00")


def test_window_measure_export(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from calibration.measure_window import MeasureWindow
    app = QApplication.instance() or QApplication([])
    path = _mat(tmp_path)
    w = MeasureWindow([str(path)])
    w.show()
    app.processEvents()
    assert len(w._entries) == 1                        # "Full frames in this file"
    w.image_list.setCurrentRow(0)
    assert w.entry.kind == "full frame" and w.array.shape == (700, 600)
    # draw a line with the mouse
    vp = w.view.viewport()
    a, b = w.view.mapFromScene(100, 100), w.view.mapFromScene(400, 500)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, a)
    QTest.mouseMove(vp, b)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, b)
    ms = w.current_measurements()
    assert len(ms) == 1 and ms[0].length_px == pytest.approx(500, rel=0.02)
    # drag its second end
    c = w.view.mapFromScene(400, 100)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, w.view.mapFromScene(ms[0].x2, ms[0].y2))
    QTest.mouseMove(vp, c)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, c)
    assert len(ms) == 1 and abs(ms[0].y2 - 100) < 2 and ms[0].length_px == pytest.approx(300, rel=0.02)
    # pixel size from that line (300 px == 690 µm)
    from PySide6.QtWidgets import QInputDialog
    QInputDialog.getDouble = staticmethod(lambda *a, **k: (ms[0].length_px * 2.3, True))
    w.pixel_size_from_line()
    assert w.pixel_um() == pytest.approx(2.3, abs=1e-3)
    # ROI with its stored outline; measurements kept per image
    w.show_roi(str(path), 0)
    assert w.entry.kind == "ROI" and w._boundary_item is not None and not w.current_measurements()
    w.add(Measurement(12, 15, 28, 15))
    png = tmp_path / "roi.png"
    w.export_png(png, scale=4)
    img = QImage(str(png))
    assert img.width() == 120 and img.height() > 160                 # 30 (camera x) by 40 (camera y) px, ×4
    rows = list(csv.DictReader(open(w.export_csv(tmp_path / "m.csv") and tmp_path / "m.csv")))
    assert len(rows) == 2 and {r["image"] for r in rows} == {"full frame", "ROI"}
    roi_row = next(r for r in rows if r["image"] == "ROI")
    assert float(roi_row["length_px"]) == 16 and float(roi_row["length_um"]) == pytest.approx(16 * 2.3, abs=0.01)
    assert roi_row["x1_camera_px"] == ""                            # this test MAT file has no ROI corners
    # an ordinary image file
    QImage(str(png)).save(str(tmp_path / "g.png"))
    w.open_image_file(str(tmp_path / "g.png"))
    assert w.entry.kind == "image file" and w.array.shape[1] == 120
    w.close()


def test_workbench_opens_measure_window_on_current_roi(tmp_path, monkeypatch):
    from datetime import datetime
    from PySide6.QtWidgets import QApplication
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    app = QApplication.instance() or QApplication([])
    from calibration.gui import CalibrationWindow
    from calibration.data import ParticleCatalog, ParticleRecord, ParticleStore, datetime_to_matlab_datenum
    _mat(tmp_path)
    t = datetime_to_matlab_datenum(datetime(2002, 3, 21, 13, 15))
    w = CalibrationWindow()
    w.catalog = ParticleCatalog(tmp_path, [ParticleRecord("f.mat", 0, t, 40, 40, 20, 1, 100, 100, 1, 1)])
    w.store = ParticleStore(tmp_path)
    mw = w.open_measure_window(w.catalog.records[0])
    assert mw.entry is not None and mw.entry.index == 0 and mw.entry.kind == "ROI"
    assert mw.file_combo.count() == 1
    assert w.open_measure_window(None) is mw            # one window, reused
    mw.close()
    w.close()
    app.processEvents()


def test_camera_frames_rebuilt_from_rois(tmp_path):
    from calibration.measure import BackgroundImages
    path = tmp_path / "g.mat"
    bgimg = np.full((100, 120), 150, np.uint8)
    a = np.full((10, 8), 20, np.uint8)
    b = np.full((6, 5), 40, np.uint8)
    images = np.empty((1, 3), object)
    for k, im in enumerate((a, b, np.full((4, 4), 9, np.uint8))):
        images[0, k] = {"IM": im}
    t0 = 731296.5
    sio.savemat(path, {"ROI_N": {"IMAGE": images, "Time": np.array([[t0], [t0], [t0 + 1 / 86400]]),
                                 "imageType": np.array([[1], [1], [1]]),
                                 "StartX": np.array([[10, 50, 0]]), "EndX": np.array([[19, 55, 3]]),
                                 "StartY": np.array([[20, 60, 0]]), "EndY": np.array([[27, 64, 3]])}})
    full = {"Time": np.array([[t0 + 5 / 86400], [t0 + 1]]),
            "IMAGE": np.array([{"BG": bgimg}, {"BG": bgimg * 0}], dtype=object)}
    sio.savemat(tmp_path / "full_backgrounds.mat", {"FULL_BG": full})
    src = MatImages(path)
    assert [r for _, r in src.frames()] == [[0, 1], [2]]
    frame, k = src.frame_image(0, BackgroundImages(tmp_path / "full_backgrounds.mat"))
    # camera orientation: rows = camera y, columns = camera x
    assert k == 0 and frame.shape == (120, 100)
    assert (frame[20:28, 10:20] == 20).all() and (frame[60:65, 50:56] == 40).all() and frame[0, 0] == 150
    box = src.frame_boundary(0)                                      # display (row, col) = (y, x)
    assert box[:5, 1].min() == 9.5 and box[:5, 1].max() == 19.5      # pixel edges of ROI 0: camera x 10..19
    assert box[:5, 0].min() == 19.5 and box[:5, 0].max() == 27.5     # camera y 20..27
    grey, k2 = src.frame_image(1, None)
    assert k2 is None and grey[0, 0] == 9


def test_orientation_matches_camera_xy_plot(tmp_path):
    """Something long in camera X must appear wide (across) everywhere, as on the Camera X-Y plot."""
    from PySide6.QtWidgets import QApplication
    from calibration.gui import ParticleCanvas
    from calibration.measure import measurement_rows
    app = QApplication.instance() or QApplication([])  # noqa: F841
    path = tmp_path / "o.mat"
    im = np.full((40, 12), 200, np.uint8)          # stored [x, y]: 40 px along camera X, 12 along Y
    im[5:35, 4:8] = 10                             # a bar 30 px long in X, 4 px in Y
    images = np.empty((1, 1), object)
    images[0, 0] = {"IM": im}
    sio.savemat(path, {"ROI_N": {"IMAGE": images, "Time": np.array([[731296.5]]), "imageType": np.array([[1]]),
                                 "StartX": np.array([[300]]), "EndX": np.array([[339]]),
                                 "StartY": np.array([[700]]), "EndY": np.array([[711]])},
                       "dat": {"foc": np.array([{"boundaries": np.array([[5.0, 4.0], [34.0, 4.0], [34.0, 7.0],
                                                                         [5.0, 7.0]])}], dtype=object)}})
    src = MatImages(path)
    shown = src.image(0)
    dark_rows, dark_cols = np.nonzero(shown < 100)
    assert np.ptp(dark_cols) == 29 and np.ptp(dark_rows) == 3      # wide across, thin down
    b = src.boundary(0)                                            # display (row, col)
    assert np.ptp(b[:, 1]) == 29 and np.ptp(b[:, 0]) == 3
    # rebuilt frame: the bar sits at camera x 305..334 (columns), y 704..707 (rows)
    frame, _ = src.frame_image(0, None)
    r, c = np.nonzero(frame < 100)
    assert (c.min(), c.max(), r.min(), r.max()) == (305, 334, 704, 707)
    # CSV camera coordinates = ROI corner + position in the ROI
    entry = src.entries()[0]
    row = next(measurement_rows(entry, [Measurement(5.5, 6.0, 34.5, 6.0)], None))
    assert (row["x1_camera_px"], row["y1_camera_px"], row["x2_camera_px"]) == (305.5, 706.0, 334.5)
    # the workbench Particle tab draws it the same way round
    canvas = ParticleCanvas()
    canvas.show_particle(im, np.array([[5.0, 4.0], [34.0, 4.0], [34.0, 7.0], [5.0, 7.0]]), "t")
    shown_img = canvas.axes.images[0].get_array()
    assert shown_img.shape == (12, 40)                              # rows = y, columns = x
    line = canvas.axes.lines[0]
    assert np.ptp(line.get_xdata()) == 29 and np.ptp(line.get_ydata()) == 3
