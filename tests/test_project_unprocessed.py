"""Opening a project whose MAT files are missing or unprocessed must not lose its flags or sources."""
import json
import os

import numpy as np
import pytest
import scipy.io as sio

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_unprocessed_and_missing_files_keep_flags_and_sources(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a[2]) or QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    (tmp_path / "T").mkdir()
    images = np.empty((1, 1), object)
    images[0, 0] = {"IM": np.zeros((5, 5), np.uint8)}
    sio.savemat(tmp_path / "T" / "a.mat", {"ROI_N": {"IMAGE": images, "Time": np.array([[731296.5]])}})  # no dat
    flags = tmp_path / "p_flags.csv"
    flags.write_text("particle_id,source_file,roi_index,status\nx#1,x,1,keep\n")
    project = tmp_path / "p.json"
    project.write_text(json.dumps({"format_version": 4, "sources": ["T/a.mat", "T/gone.mat"],
                                   "flags_file": "p_flags.csv"}))
    from calibration.gui import CalibrationWindow
    w = CalibrationWindow()
    w.open_project(str(project))
    assert not w.catalog.records
    assert warnings and "not found" in warnings[0] and "no particle statistics" in warnings[0]
    w.save_project()
    w.flush_saves()
    assert flags.read_text().count("\n") == 2                       # saved flags untouched
    assert json.loads(project.read_text())["sources"] == ["T/a.mat", "T/gone.mat"]
    w.close()
    w.deleteLater()
    app.processEvents()
    import gc
    gc.collect()
