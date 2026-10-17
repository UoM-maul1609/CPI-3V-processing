"""Headless check of the select → refine → accept → unflag → undo workflow."""
import os
from datetime import datetime

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    from calibration.gui import CalibrationWindow
    from calibration.data import ParticleCatalog, ParticleRecord, ParticleStore, datetime_to_matlab_datenum
    t = datetime_to_matlab_datenum(datetime(2002, 3, 21, 13, 15))
    records = []
    for i in range(40):
        # two clusters on the camera; particles 0-19 left (x=100..), 20-39 right (x=800..)
        x = 100 + i if i < 20 else 800 + i
        length = 300.0 if i in (3, 4) else 40.0  # two size outliers in the left cluster
        r = ParticleRecord("a.mat", i, t + i / 86400, length, length, 20.0, 1, x, 500, 1, 1)
        r.stage_x, r.stage_y = 10 + (i % 5) * 0.1, 5.0
        records.append(r)
    w = CalibrationWindow()
    w.catalog = ParticleCatalog(tmp_path, records)
    w.store = ParticleStore(tmp_path)
    w.selection_file = tmp_path / "flags.csv"
    w.position_metric.setCurrentIndex(w.position_metric.findData("stage_x"))
    w.apply_filters()
    yield w
    w.close()
    app.processEvents()


def test_select_refine_accept_skip_unflag_undo(window):
    from PySide6.QtCore import Qt
    from calibration.labels import ParticleLabel
    w = window
    w._on_box(w.spatial_canvas, 0, (0, 0), (400, 1000), Qt.NoModifier)
    assert len(w.selected_ids) == 20
    # Ctrl-drag on the length plot (bottom axes) drops the outliers.
    length_axes = w._metric_rows.index("length")
    w._on_box(w.plot_canvas, length_axes, (0, 200), (100, 400), Qt.ControlModifier)
    assert len(w.selected_ids) == 18
    w.labels.upsert(ParticleLabel("PSL 40", length_um=40))
    w._refresh_label_widgets()
    w.accept_label.setCurrentIndex(w.accept_label.findData("PSL 40"))
    w.accept_selection()
    accepted = [r for r in w.catalog.records if r.status == "keep"]
    assert len(accepted) == 18 and not w.selected_ids
    assert {r.label for r in accepted} == {"PSL 40"} and accepted[0].actual_length_um == 40
    assert w.selection_file.exists()
    # Re-boxing the same cluster skips accepted particles (only the 2 outliers remain pickable).
    w._on_box(w.spatial_canvas, 0, (0, 0), (400, 1000), Qt.NoModifier)
    assert len(w.selected_ids) == 2
    w.reject_selection()
    assert sum(r.status == "reject" for r in w.catalog.records) == 2
    # Grab accepted ones and unflag them back to the default.
    w.pick_scope.setCurrentIndex(w.pick_scope.findData("keep"))
    w._on_box(w.spatial_canvas, 0, (0, 0), (110, 1000), Qt.NoModifier)
    n = len(w.selected_ids)
    assert n > 0
    w.unflag_selection()
    assert sum(r.status == "keep" for r in w.catalog.records) == 18 - n
    assert all(r.label == "" for r in w.catalog.records if r.status == "undecided")
    w.undo()
    assert sum(r.status == "keep" for r in w.catalog.records) == 18
    assert {r.label for r in w.catalog.records if r.status == "keep"} == {"PSL 40"}


def test_reject_box_and_gallery_removal(window):
    from PySide6.QtCore import Qt
    w = window
    w._on_box(w.spatial_canvas, 0, (0, 0), (400, 1000), Qt.NoModifier)
    w.set_box_mode("reject")
    length_axes = w._metric_rows.index("length")
    w._on_box(w.plot_canvas, length_axes, (0, 200), (100, 400), Qt.NoModifier)
    assert sum(r.status == "reject" for r in w.catalog.records) == 2
    assert len(w.selected_ids) == 18  # rejected ones left the selection
    w.viewer_tabs.setCurrentWidget(w.gallery_panel)
    w.update_gallery()
    assert w.gallery.count() == 18
    w.gallery.item(0).setSelected(True)
    w.gallery_selection("remove")
    assert len(w.selected_ids) == 17
