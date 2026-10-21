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
    w.flush_saves()                        # writes are batched ~1.5 s after the last change
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


def test_box_of_only_accepted_particles_selects_them(window):
    from PySide6.QtCore import Qt
    w = window
    w._on_box(w.spatial_canvas, 0, (0, 0), (400, 1000), Qt.NoModifier)
    w.accept_selection()
    assert not w.selected_ids
    # Default pick scope is still "Unreviewed only", but the box holds only accepted ones.
    w._on_box(w.spatial_canvas, 0, (0, 0), (110, 1000), Qt.NoModifier)
    assert w.selected_ids and all(w.catalog._by_id[p].status == "keep" for p in w.selected_ids)
    assert "selected anyway" in w.statusBar().currentMessage()


def test_escape_cancels_a_box_being_drawn(window):
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest
    w = window
    w.show()
    c, ax = w.spatial_canvas, w.spatial_axes
    def q(x, y):
        px, py = ax.transData.transform((x, y))
        s = c.device_pixel_ratio
        return QPoint(int(px / s), int((c.figure.bbox.height - py) / s))
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, q(90, 400))
    QTest.mouseMove(c, q(400, 600))
    w._key_escape()
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, q(400, 600))
    assert not w.selected_ids


def test_project_sets_follow_reprocessed_particles(window):
    from calibration.reference_sets import make_reference
    from calibration.selection_state import add_ids_to_saved_set
    w = window
    ids = [r.particle_id for r in w.catalog.records[:3]]
    keys = {pid: list(w.catalog._by_id[pid].match_key) for pid in ids}
    # Simulate reprocessing: shift every ROI index by one.
    for r in w.catalog.records:
        r.roi_index += 1
    w.catalog._by_id = {r.particle_id: r for r in w.catalog.records}
    w.saved_sets = {}
    add_ids_to_saved_set(w.saved_sets, "Run 1", ids, make_reference(10, 5))
    w._remap_saved_ids({}, keys)
    expected = sorted(f"a.mat#{i + 1}" for i in range(3))
    assert w.saved_sets["Run 1"]["particle_ids"] == expected


def test_project_restores_best_focus_reference(window, tmp_path):
    import json
    from calibration.reference_sets import make_reference
    w = window
    sets = {"Run 1": {"particle_ids": ["a.mat#0"], "description": "Calibration working set",
                      "created_utc": "", "best_focus": make_reference(10.9, 5.25)}}
    # A project saved by an older version: no draft_reference stored.
    (tmp_path / "old.json").write_text(json.dumps({"sources": [], "named_sets": sets,
                                                   "active_reference_set": "Run 1"}))
    w.open_project(str(tmp_path / "old.json"))
    assert (w.best_x_set.value(), w.best_z_set.value()) == (10.9, 5.25)
    assert "Run 1" in w.active_set_label.text()             # old set shows as the group being edited
    assert any(w.batch_list.item(i).data(0x0100) == "Run 1" for i in range(w.batch_list.count()))
    # Unsaved edits in the editor survive save → reopen.
    w.best_x_set.setValue(11.05)
    w.save_project()
    w.best_x_set.setValue(0)
    w.open_project(str(tmp_path / "old.json"))
    assert w.best_x_set.value() == 11.05 and w.best_z_set.value() == 5.25
    assert w.saved_sets["Run 1"]["best_focus"]["x_best"] == 10.9   # saved set untouched


def test_batch_references_and_time_steps(window, tmp_path):
    import csv
    from pathlib import Path
    from PySide6.QtCore import Qt
    w = window
    w.project_file = Path(tmp_path / "proj.json")
    ids = [r.particle_id for r in w.catalog.records]           # records are 1 s apart, in time order
    def accept(lo, hi):
        w._change_selection(set(ids[lo:hi]), push=False)
        w.accept_selection()
    accept(0, 5); accept(10, 15); accept(20, 25)                 # batches 1, 2, 3 in time order
    def pick_batch(name):
        item = next(w.batch_list.item(i) for i in range(w.batch_list.count())
                    if w.batch_list.item(i).data(Qt.UserRole) == name)
        w.batch_list.clearSelection()           # a plain click
        w.batch_list.setCurrentItem(item)
        w.batch_clicked()
    pick_batch("Accepted batch 1"); w.best_x_set.setValue(10.1); w.best_z_set.setValue(5.0); w.save_batch_reference()
    pick_batch("Accepted batch 3"); w.best_x_set.setValue(10.3); w.save_batch_reference()
    w.active_reference_set = None; w._refresh_saved_sets()
    by_id = w.catalog._by_id
    # Batch 2 (no X/Z of its own) borrows from the nearest calibrated batch on the camera
    # in the same session: batch 1 (x≈102) rather than batch 3 (x≈822).
    ref, src, kind = w._reference_for(by_id[ids[12]], use_editor=False)
    assert (ref["x_best"], src, kind) == (10.1, "Accepted batch 1", "nearest")
    assert "best_focus" not in w.saved_sets["Accepted batch 2"]   # no peak + untouched line → nothing saved
    assert w._reference_for(by_id[ids[22]], use_editor=False)[1:] == ("Accepted batch 3", "own")
    assert w._reference_for(by_id[ids[30]], use_editor=False)[0]["x_best"] == 10.3   # later, unflagged
    # Clicking batch 2 (too few particles for a focus peak) shows what it borrows.
    pick_batch("Accepted batch 2")
    assert w.best_x_set.value() == 10.1 and "no focus peak" in w.batch_ref_label.text()
    w.save_project()
    with open(tmp_path / "proj_particle_positions.csv") as h:
        rows = {r["particle_id"]: r for r in csv.DictReader(h)}
    assert len(rows) == 15
    assert rows[ids[12]]["reference_kind"] == "nearest" and float(rows[ids[12]]["x_best"]) == 10.1
    assert rows[ids[22]]["reference_from"] == "Accepted batch 3"
    assert "Accepted batch 1" in w.timeline_label.text() and "X=10.3" in w.timeline_label.text()


def test_line_follows_cluster_peak_and_accept_saves_it(tmp_path, monkeypatch):
    """Scan with a tilted slide: each cluster gets its own peak, saved on Accept."""
    import numpy as np
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    from PySide6.QtCore import Qt
    from calibration.gui import CalibrationWindow
    from calibration.data import ParticleCatalog, ParticleRecord, ParticleStore, datetime_to_matlab_datenum
    from datetime import datetime
    t0 = datetime_to_matlab_datenum(datetime(2002, 3, 21, 13, 0))
    recs, i = [], 0
    for cx, best in ((150, 10.5), (850, 11.0)):          # left patch focuses at 10.5, right at 11.0
        for sx in np.linspace(10, 11.5, 7):
            for k in range(4):
                r = ParticleRecord("a.mat", i, t0 + i / 86400, 40, 40, 30 * np.exp(-((sx - best) / .3) ** 2) + 5,
                                   1, cx + k, 500 + k, 1, 1)
                r.stage_x, r.stage_y = float(sx), 5.0
                recs.append(r); i += 1
    w = CalibrationWindow()
    w.catalog, w.store = ParticleCatalog(tmp_path, recs), ParticleStore(tmp_path)
    w.selection_file = tmp_path / "flags.csv"
    from pathlib import Path
    w.project_file = Path(tmp_path / "proj.json")
    w.position_metric.setCurrentIndex(w.position_metric.findData("stage_x"))
    w.apply_filters()
    w._on_box(w.spatial_canvas, 0, (0, 0), (400, 1000), Qt.NoModifier)
    assert w.best_x_set.value() == pytest.approx(10.5)        # line jumped to the left cluster's peak
    w.accept_selection()
    w._on_box(w.spatial_canvas, 0, (600, 0), (1000, 1000), Qt.NoModifier)
    assert w.best_x_set.value() == pytest.approx(11.0)        # new cluster → its own peak
    w.best_x_set.setValue(10.95)                               # user override sticks for this cluster
    w._on_box(w.plot_canvas, 0, (11.4, -1), (11.6, 100), Qt.ControlModifier)   # refine: still overridden
    assert w.best_x_set.value() == pytest.approx(10.95)
    w.accept_selection()
    refs = {n: e["best_focus"]["x_best"] for n, e in w.saved_sets.items() if e.get("best_focus")}
    assert refs == {"Accepted batch 1": pytest.approx(10.5), "Accepted batch 2": pytest.approx(10.95)}
    # An old project's batch without X/Z: one click fills it from its own peak.
    w.saved_sets["Accepted batch 2"].pop("best_focus")
    w._refresh_saved_sets()
    w.fill_batch_references(scope="missing")
    assert w.saved_sets["Accepted batch 2"]["best_focus"]["x_best"] == pytest.approx(11.0)
    # "All batches" re-centres batches that already have X/Z too (d = 0 at their focus max).
    w.saved_sets["Accepted batch 1"]["best_focus"]["x_best"] = 10.2
    w._refresh_saved_sets()
    w.fill_batch_references(scope="all")
    assert w.saved_sets["Accepted batch 1"]["best_focus"]["x_best"] == pytest.approx(10.5)
    w.undo()
    assert w.saved_sets["Accepted batch 1"]["best_focus"]["x_best"] == pytest.approx(10.2)
    w.fill_batch_references(scope="all")
    # Selecting all accepted particles shows each batch's own saved X/Z, not the editor's,
    # and does not move the editor line.
    w.best_x_set.setValue(12.0)
    w.select_group("keep")
    assert w.best_x_set.value() == pytest.approx(12.0)
    by_id = w.catalog._by_id
    for name, expect in (("Accepted batch 1", 10.5), ("Accepted batch 2", 11.0)):
        pid = w.saved_sets[name]["particle_ids"][0]
        ref, src, kind = w._reference_for(by_id[pid])
        assert (kind, src, ref["x_best"]) == ("own", name, pytest.approx(expect))
    assert "kept from their batches" in w.save_ref_cb.text()
    # Re-accepting them does not overwrite their X/Z.
    w.accept_selection()
    refs = {n: e["best_focus"]["x_best"] for n, e in w.saved_sets.items() if e.get("best_focus")}
    assert 12.0 not in refs.values()
    w.close()


def test_peak_on_plane_puts_d_zero_at_focus_max():
    import math
    import numpy as np
    from calibration.reference_sets import focus_peak_on_plane, signed_plane_distance
    # Diagonal scan (X and Z both move), peak at step 6 of 11.
    steps = np.arange(11)
    x = np.repeat(10 + 0.05 * steps, 5)
    z = np.repeat(5 + 0.03 * steps, 5)
    proj = x * math.cos(math.radians(45)) + z * math.sin(math.radians(45))
    peak_proj = proj[6 * 5]
    focus = 30 * np.exp(-((proj - peak_proj) / 0.1) ** 2) + 5
    ref, why = focus_peak_on_plane(x, z, focus, 45.0)
    assert ref is not None, why
    assert signed_plane_distance(x[30], z[30], ref, 45.0) == pytest.approx(0.0, abs=1e-9)
    # X-only scan: the reference keeps the scan's Z.
    ref, _ = focus_peak_on_plane(x, np.full_like(x, 5.0), focus, 45.0)
    assert ref["z_best"] == pytest.approx(5.0) and ref["x_best"] == pytest.approx(x[30])


def test_rejecting_or_unflagging_removes_particles_from_their_batch(window):
    from PySide6.QtCore import Qt
    w = window
    ids = [r.particle_id for r in w.catalog.records]
    w._change_selection(set(ids[:6]), push=False)
    w.accept_selection()
    assert len(w.saved_sets["Accepted batch 1"]["particle_ids"]) == 6
    w._change_selection({ids[0]}, push=False)
    w.reject_selection()                                    # debris → rejected, leaves the accepted batch
    w._change_selection({ids[1]}, push=False)
    w.unflag_selection()                                    # back to unreviewed, leaves the batch too
    assert sorted(w.saved_sets["Accepted batch 1"]["particle_ids"]) == sorted(ids[2:6])
    w.undo()                                                # undo restores membership
    assert ids[1] in w.saved_sets["Accepted batch 1"]["particle_ids"]
    w._change_selection(set(ids[1:6]), push=False)
    w.unflag_selection()                                    # last members gone → empty batch disappears
    assert "Accepted batch 1" not in w.saved_sets


def test_undo_batch_still_works_with_pruning(window):
    w = window
    ids = [r.particle_id for r in w.catalog.records]
    w._change_selection(set(ids[:4]), push=False)
    w.accept_selection()
    w.batch_list.setCurrentRow(0)
    w.remove_classified_batch()                 # the fixture answers "No" to questions → nothing happens
    assert "Accepted batch 1" in w.saved_sets
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        w.remove_classified_batch()
    finally:
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
    assert "Accepted batch 1" not in w.saved_sets
    assert all(w.catalog._by_id[p].status == "undecided" for p in ids[:4])


def test_redo_reapplies_what_was_undone(window):
    w = window
    ids = [r.particle_id for r in w.catalog.records]
    w._change_selection(set(ids[:5]), push=False)
    w.accept_selection()
    w._change_selection({ids[0]}, push=False)
    w.reject_selection()
    w.undo()                                   # un-reject
    assert w.catalog._by_id[ids[0]].status == "keep"
    w.undo()                                   # un-accept
    assert all(w.catalog._by_id[p].status == "undecided" for p in ids[:5])
    assert "Accepted batch 1" not in w.saved_sets
    w.redo()                                   # re-accept, batch back
    assert all(w.catalog._by_id[p].status == "keep" for p in ids[:5])
    assert "Accepted batch 1" in w.saved_sets
    w.redo()                                   # re-reject
    assert w.catalog._by_id[ids[0]].status == "reject"
    assert ids[0] not in w.saved_sets["Accepted batch 1"]["particle_ids"]
    w.undo()
    w._change_selection({ids[7]}, push=True)   # a new action clears the redo chain
    assert not w.redo_stack and not w.redo_btn.isEnabled()


def test_flag_history_shows_batch_labels(window):
    from calibration.labels import ParticleLabel
    w = window
    ids = [r.particle_id for r in w.catalog.records]
    w.labels.upsert(ParticleLabel("PSL 40", length_um=40))
    w._refresh_label_widgets()
    w.accept_label.setCurrentIndex(w.accept_label.findData("PSL 40"))
    w._change_selection(set(ids[:4]), push=False)
    w.accept_selection()
    w.accept_label.setCurrentIndex(0)                      # (no label)
    w._change_selection(set(ids[10:13]), push=False)
    w.accept_selection()
    texts = {w.batch_list.item(i).text().split(" · ")[0][2:]: w.batch_list.item(i).text()
             for i in range(w.batch_list.count())}
    assert "🏷 PSL 40" in texts["Accepted batch 1"]
    assert "🏷 no label" in texts["Accepted batch 2"]


def test_accepted_batch_stays_on_screen_and_multi_box_batches(window):
    from PySide6.QtCore import Qt
    w = window
    w.selection_only_cb.setChecked(True)
    # Two boxes (second with Shift) make one batch from two areas.
    w._on_box(w.spatial_canvas, 0, (0, 0), (105, 1000), Qt.NoModifier)       # x 100-105
    first = set(w.selected_ids)
    assert "Shift-drag to add another area" in w.statusBar().currentMessage()
    w._on_box(w.spatial_canvas, 0, (815, 0), (830, 1000), Qt.ShiftModifier)  # x 815-830
    assert first < w.selected_ids and len(w.selected_ids) == len(first) + 11   # x 820-830
    batch_ids = set(w.selected_ids)
    w.accept_selection()
    assert not w.selected_ids
    # The focus/size plots keep showing the batch just accepted, not everything.
    records = w._records()
    shown = {records[i].particle_id for i in w._visible_indices("metric")}
    assert shown == batch_ids
    assert "Accepted batch 1" in w.scope_label.text()
    assert w._effective_browse_scope() == "batch"
    # Esc with nothing selected goes back to showing everything.
    w._key_escape()
    assert len(w._visible_indices("metric")) == len(w._base_indices)
    # Clicking the batch shows it again; starting a new cluster releases it.
    w.batch_list.setCurrentRow(0); w.batch_clicked()
    assert {records[i].particle_id for i in w._visible_indices("metric")} == batch_ids
    w._on_box(w.spatial_canvas, 0, (106, 0), (115, 1000), Qt.NoModifier)
    assert w.active_reference_set is None


def test_hover_and_readouts_name_the_batch(window):
    w = window
    ids = [r.particle_id for r in w.catalog.records]
    w._change_selection(set(ids[:3]), push=False)
    w.accept_selection()
    rec = w.catalog._by_id[ids[1]]
    assert w._batch_text(rec, html=False).startswith("Accepted batch 1")
    assert w._batch_text(w.catalog._by_id[ids[30]], html=False).startswith("no batch")


def test_close_and_switch_projects_save_first(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
    app = QApplication.instance() or QApplication([])
    from calibration.gui import CalibrationWindow
    from calibration.data import ParticleCatalog, ParticleRecord, ParticleStore
    from pathlib import Path
    w = CalibrationWindow()
    recs = [ParticleRecord("a.mat", i, 738000 + i / 86400, 40, 40, 10, 1, 100 + i, 100, 1, 1) for i in range(5)]
    w.catalog, w.store = ParticleCatalog(tmp_path, recs), ParticleStore(tmp_path)
    w.project_file = Path(tmp_path / "first.json")
    w.selection_file = tmp_path / "first_flags.csv"
    saved = []
    monkeypatch.setattr(w, "save_project", lambda: saved.append(w.project_file))
    assert w.close_project()
    assert saved == [tmp_path / "first.json"]          # saved before closing
    assert w.project_file is None and not w.catalog.records
    assert "no project" in w.windowTitle()
    # New project: cancelling the file dialog leaves things as they were.
    w.project_file = Path(tmp_path / "second.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    w.create_project()
    assert w.project_file == tmp_path / "second.json"
    w.project_file = None
    w.close()


def test_new_window_shares_labels_and_blocks_same_project(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
    app = QApplication.instance() or QApplication([])
    from calibration.gui import CalibrationWindow
    from pathlib import Path
    shown = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2])))
    w1 = CalibrationWindow()
    w2 = w1.new_window()
    assert w2 is not w1 and w2.labels is w1.labels
    w1.project_file = Path(tmp_path / "beads.json")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(tmp_path / "beads.json"), "")))
    opened = []
    monkeypatch.setattr(w2, "open_project", lambda f: opened.append(f))
    w2.open_project_dialog()
    assert not opened and "already open" in shown[-1]
    for w in (w1, w2):
        w.project_file = None
        w.close()
    assert w1 not in CalibrationWindow._windows and w2 not in CalibrationWindow._windows


def test_merge_projects_into_new_project(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from pathlib import Path
    import math
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    from calibration.gui import CalibrationWindow
    from calibration.labels import ParticleLabel
    from mat_synth import make_scan
    projects = []
    for name, day, size in (("beads100", 2, 100.0), ("beads40", 9, 40.0)):
        data = make_scan(tmp_path / name, t0=datetime(2026, 3, day, 10), seed=day)
        w = CalibrationWindow()
        w.load_directory(str(data))
        w.load_position_file(str(data / "stage.csv"))
        w.project_file = (data / f"{name}.json").resolve()
        w.selection_file = data / f"{name}_flags.csv"
        w.labels.upsert(ParticleLabel(name, kind="bead", length_um=size)); w.labels.save()
        w._refresh_label_widgets()
        w.accept_label.setCurrentIndex(w.accept_label.findData(name))
        w._on_box(w.spatial_canvas, 0, (0, 0), (1000, 1000), Qt.NoModifier)
        w.accept_selection()
        w.save_project()
        projects.append((w.project_file, len(w.catalog.records)))
        w.project_file = None
        w.close()
    m = CalibrationWindow()
    out = tmp_path / "merged" / "merged.json"
    out.parent.mkdir()
    assert m.merge_projects([str(p) for p, _ in projects], out, ask=False)
    total = sum(n for _, n in projects)
    assert len(m.catalog.records) == total
    assert sum(r.status == "keep" for r in m.catalog.records) == total
    assert {r.label for r in m.catalog.records} == {"beads100", "beads40"}
    assert all(math.isfinite(r.stage_x) for r in m.catalog.records)        # both stage CSVs used
    names = set(m.saved_sets)
    assert {"beads100 · Accepted batch 1", "beads40 · Accepted batch 1"} <= names
    assert all(m.saved_sets[n].get("best_focus") for n in names)
    assert m.merged_from == ["beads100", "beads40"] and "viewing copy" in m.windowTitle()
    m.project_file = None
    m.close()
    # The recommended route: fit the original runs directly; the model records them.
    f = CalibrationWindow()
    dialog = f.fit_from_projects([str(p) for p, _ in projects], confirm=False)
    assert dialog is not None and [i["project"] for i in dialog.inputs] == ["beads100", "beads40"]
    assert len(dialog.rows) == total
    f.close()
    # Reopen the merged project from disk.
    r = CalibrationWindow()
    r.open_project(str(out))
    assert sum(x.status == "keep" for x in r.catalog.records) == total
    assert all(math.isfinite(x.stage_x) for x in r.catalog.records)
    assert len(r.extra_position_files) == 1 and r.position_file
    r.project_file = None
    r.close()


def test_remove_mat_file_and_add_it_back(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from pathlib import Path
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    from calibration.gui import CalibrationWindow
    from mat_synth import make_scan
    a = make_scan(tmp_path / "a", t0=datetime(2026, 3, 2, 10), seed=1)
    b = make_scan(tmp_path / "b", t0=datetime(2026, 3, 2, 14), seed=2)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "proj.json").resolve()
    w.selection_file = tmp_path / "proj_flags.csv"
    w.import_file(a / "scan.mat"); w.import_file(b / "scan.mat")
    w.apply_filters()
    n_a = sum(r.source_file == str((a / "scan.mat").resolve()) for r in w.catalog.records)
    total = len(w.catalog.records)
    w._change_selection({r.particle_id for r in w.catalog.records})
    w.accept_selection()
    assert sum(r.status == "keep" for r in w.catalog.records) == total
    assert w.remove_sources([str((a / "scan.mat").resolve())])
    assert len(w.catalog.records) == total - n_a and len(w.sources) == 1
    assert all(len(e["particle_ids"]) == total - n_a for e in w.saved_sets.values())
    assert (tmp_path / "proj_flags_removed.csv").exists()
    # Adding the file back restores its accepted flags.
    w.import_file(a / "scan.mat")
    assert sum(r.status == "keep" for r in w.catalog.records) == total
    w.close()          # saves into tmp_path


def test_saved_time_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from PySide6.QtWidgets import QApplication, QMessageBox, QDialog
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow, TimeWindowDialog
    from calibration.labels import ParticleLabel
    from mat_synth import make_scan
    # One dataset, three bead runs separated by pauses (15, 40, 100 µm).
    for k, hour in enumerate((10, 11, 13)):
        make_scan(tmp_path / "data" / f"run{k}", t0=datetime(2026, 3, 2, hour), seed=k)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "proj.json").resolve()
    w.selection_file = tmp_path / "proj_flags.csv"
    for k in range(3):
        w.import_file(tmp_path / "data" / f"run{k}" / "scan.mat")
    w.apply_filters()
    total = len(w.catalog.records)
    added = w.suggest_time_windows(gap_minutes=5)
    assert len(added) == 3 and w.window_list.count() == 3 and w.window_combo.count() == 4
    w.labels.upsert(ParticleLabel("15 micron glass", kind="bead", length_um=15.0)); w._refresh_label_widgets()
    # Edit the first window: name + label (dialog auto-accepted with new values).
    def fake_exec(self):
        self.name.setText("15 µm beads")
        self.label.setCurrentIndex(self.label.findData("15 micron glass"))
        return QDialog.Accepted
    monkeypatch.setattr(TimeWindowDialog, "exec", fake_exec)
    w.window_list.setCurrentRow(0)
    w.edit_time_window()
    assert w.time_windows[0]["name"] == "15 µm beads" and w.time_windows[0]["label"] == "15 micron glass"
    w.show_time_window("15 µm beads")
    assert 0 < len(w._base_indices) < total
    assert w._accept_label_name() == "15 micron glass"
    w.save_project()
    # Reopen: windows and the active one come back.
    r = CalibrationWindow()
    r.open_project(str(w.project_file))
    assert [x["name"] for x in r.time_windows][0] == "15 µm beads" and r.active_time_window == "15 µm beads"
    assert len(r._base_indices) == len(w._base_indices)
    r.show_time_window(None)
    assert len(r._base_indices) == total
    w.close(); r.close()


def test_time_window_helpers():
    from calibration.time_windows import suggest_windows, make_window, in_window, overlapping
    t0 = 738000.0
    times = [t0 + i / 86400 for i in range(30)] + [t0 + 1 / 24 + i / 86400 for i in range(30)]
    ws = suggest_windows(times, gap_minutes=5)
    assert len(ws) == 2 and in_window(times, ws[0]).sum() == 30
    a = make_window("a", t0, t0 + 0.01); b = make_window("b", t0 + 0.005, t0 + 0.02)
    assert overlapping([a, b]) == [("a", "b")]


def test_only_selection_is_default_and_s_toggles(monkeypatch, tmp_path):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    from calibration.gui import CalibrationWindow
    w = CalibrationWindow(); w.show()
    assert w.selection_only_cb.isChecked()
    from PySide6.QtGui import QShortcut, QKeySequence
    s_key = [sc for sc in w.findChildren(QShortcut) if sc.key() == QKeySequence("S")]
    assert len(s_key) == 1
    s_key[0].activated.emit()
    assert not w.selection_only_cb.isChecked()
    s_key[0].activated.emit()
    assert w.selection_only_cb.isChecked()
    w.close()


def test_trim_accepted_batch_on_plots(tmp_path, monkeypatch):
    """After Accept, boxes on the focus/length plots act on the batch on show."""
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1, per_step=10)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.load_position_file(str(d / "stage.csv"))
    w.position_metric.setCurrentIndex(w.position_metric.findData("stage_x")); w.apply_filters()
    w._on_box(w.spatial_canvas, 0, (0, 0), (500, 1000), Qt.NoModifier); w.accept_selection()
    w._on_box(w.spatial_canvas, 0, (500, 0), (1000, 1000), Qt.NoModifier); w.accept_selection()
    one, two = (set(w.saved_sets[n]["particle_ids"]) for n in ("Accepted batch 1", "Accepted batch 2"))
    hi = w.metric_axes[1].get_ylim()[1]
    # Reject box with batch 2 on show: rejects only batch-2 particles inside the box.
    w.set_box_mode("reject")
    w._on_box(w.plot_canvas, 1, (9.9, 60), (12.1, hi), Qt.NoModifier)
    w.set_box_mode("new")
    rejected = {r.particle_id for r in w.catalog.records if r.status == "reject"}
    assert rejected and rejected <= two
    assert set(w.saved_sets["Accepted batch 2"]["particle_ids"]) == two - rejected
    # New box on the plot selects accepted batch members; U unflags them out of the batch.
    w._on_box(w.plot_canvas, 1, (9.9, 50), (12.1, hi), Qt.NoModifier)
    picked = set(w.selected_ids)
    assert picked and picked <= two and not (picked & one)
    w.unflag_selection()
    assert not (set(w.saved_sets["Accepted batch 2"]["particle_ids"]) & picked)
    # Clicking a batch in Flag history shows it, and the 'Showing' line says so.
    w._key_escape()
    row = [i for i in range(w.batch_list.count()) if w.batch_list.item(i).data(Qt.UserRole) == "Accepted batch 1"][0]
    w.batch_list.setCurrentRow(row); w.batch_clicked()
    assert "Accepted batch 1" in w.scope_label.text()
    w.close()


def test_change_label_of_batch_and_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow
    from calibration.labels import ParticleLabel
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.apply_filters()
    for name, size in (("40 micron glass", 40.0), ("100 micron glass", 100.0)):
        w.labels.upsert(ParticleLabel(name, kind="bead", length_um=size))
    w._refresh_label_widgets()
    w.accept_label.setCurrentIndex(w.accept_label.findData("100 micron glass"))
    w._on_box(w.spatial_canvas, 0, (0, 0), (1000, 1000), Qt.NoModifier); w.accept_selection()
    members = w.saved_sets["Accepted batch 1"]["particle_ids"]
    by_id = w.catalog._by_id
    assert {by_id[p].label for p in members} == {"100 micron glass"}
    # Wrong label: change the whole batch to 40 µm.
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("40 micron glass", True)))
    n = w.relabel_batch_dialog(name="Accepted batch 1")
    assert n == len(members)
    assert {by_id[p].label for p in members} == {"40 micron glass"}
    assert {by_id[p].actual_length_um for p in members} == {40.0}
    assert all(by_id[p].status == "keep" for p in members)          # flags unchanged
    w.undo()
    assert {by_id[p].label for p in members} == {"100 micron glass"}
    # Relabel a selection (only some particles), flags still unchanged.
    some = set(members[:5])
    w._change_selection(some)
    w.relabel_selection_dialog()
    assert {by_id[p].label for p in some} == {"40 micron glass"}
    assert sum(by_id[p].label == "100 micron glass" for p in members) == len(members) - 5
    w.close()


def test_flag_history_multi_select(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow
    from calibration.labels import ParticleLabel
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1, per_step=9)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.apply_filters()
    w.labels.upsert(ParticleLabel("40 micron glass", kind="bead", length_um=40.0)); w._refresh_label_widgets()
    for x0, x1 in ((0, 330), (330, 660), (660, 1000)):
        w._on_box(w.spatial_canvas, 0, (x0, 0), (x1, 1000), Qt.NoModifier); w.accept_selection()
    w._key_escape()
    names = ["Accepted batch 1", "Accepted batch 3"]
    rows = {w.batch_list.item(i).data(Qt.UserRole): i for i in range(w.batch_list.count())}
    w.batch_list.clearSelection()
    for n in names:
        w.batch_list.item(rows[n]).setSelected(True)
    w.batch_clicked()                       # what a Ctrl-click does
    assert sorted(w._selected_batch_names()) == names
    union = set(w.saved_sets[names[0]]["particle_ids"]) | set(w.saved_sets[names[1]]["particle_ids"])
    ids, text = w._focus_group()
    assert ids == union and "2 batches" in text and "2 batches" in w.scope_label.text()
    assert sorted(w._selected_batch_names()) == names          # survives the list rebuild
    # Change label for both at once.
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("40 micron glass", True)))
    assert w.relabel_batch_dialog() == len(union)
    by_id = w.catalog._by_id
    assert {by_id[p].label for p in union} == {"40 micron glass"}
    middle = set(w.saved_sets["Accepted batch 2"]["particle_ids"])
    assert not any(by_id[p].label for p in middle)
    # Select batch takes the union.
    w.select_batch()
    assert w.selected_ids == union
    w._key_escape(); w._key_escape()
    assert w._focus_group()[0] is None
    w.close()



def test_saves_are_batched_and_happen_on_their_own(tmp_path, monkeypatch):
    import time
    from datetime import datetime
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.apply_filters()
    w._save_timer.setInterval(100)
    calls = []
    original = w.save_project
    monkeypatch.setattr(w, "save_project", lambda exports=True: (calls.append(exports), original(exports=exports)))
    for x0 in (0, 300, 600):                      # three quick accepts → one write
        w._on_box(w.spatial_canvas, 0, (x0, 0), (x0 + 300, 1000), Qt.NoModifier)
        w.accept_selection()
    assert calls == [] and not w.selection_file.exists()
    t0 = time.time()                              # the timer fires and a background thread writes
    while not (w.selection_file.exists() and w.project_file.exists()) and time.time() - t0 < 5:
        app.processEvents(); time.sleep(0.02)
    w._wait_for_writer(); app.processEvents()
    assert w.selection_file.exists() and w.project_file.exists()
    assert calls == []                            # no foreground save, no big exports
    assert "Saved" in w.save_indicator.text()
    import json
    cfg = json.loads(w.project_file.read_text())
    assert cfg["format_version"] == 4 and "by_source" in next(iter(cfg["named_sets"].values()))["particle_ids"]
    w.close()                                     # close writes everything, exports included
    assert calls[-1] is True
    r = CalibrationWindow(); r.open_project(str(w.project_file))
    assert sum(x.status == "keep" for x in r.catalog.records) == sum(x.status == "keep" for x in w.catalog.records)
    assert set(r.saved_sets) == set(w.saved_sets)
    r.project_file = None; r.close()



def test_rejects_do_not_make_batches(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow
    from calibration.selection_state import make_saved_set
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.apply_filters()
    w._on_box(w.spatial_canvas, 0, (0, 0), (300, 1000), Qt.NoModifier); w.reject_selection()
    w.set_box_mode("reject"); w._on_box(w.spatial_canvas, 0, (300, 0), (500, 1000), Qt.NoModifier); w.set_box_mode("new")
    assert sum(r.status == "reject" for r in w.catalog.records) > 0
    assert not any(e.get("description") == "Review batch: reject" for e in w.saved_sets.values())
    w.undo()                                    # Ctrl+Z still undoes a reject
    # reject batches from older projects are kept but hidden unless asked for
    some = [r.particle_id for r in w.catalog.records[:3]]
    w.saved_sets["Rejected batch 1"] = make_saved_set(some, description="Review batch: reject")
    w._refresh_saved_sets()
    names = lambda: {w.batch_list.item(i).data(Qt.UserRole) for i in range(w.batch_list.count())}
    assert "Rejected batch 1" not in names()
    w.show_reject_batches_action.setChecked(True)
    assert "Rejected batch 1" in names()
    w.close()


def test_select_batch_keeps_saved_reference(tmp_path, monkeypatch):
    """Click a batch, then Select batch: the editor must keep the batch's saved X/Z
    (not jump to the selection's focus peak), and an unsaved edit must survive too."""
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    from calibration.gui import CalibrationWindow
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1, per_step=9)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.load_position_file(str(d / "stage.csv")); w.apply_filters()
    w._on_box(w.spatial_canvas, 0, (0, 0), (500, 1000), Qt.NoModifier); w.accept_selection()
    name = "Accepted batch 1"
    row = {w.batch_list.item(i).data(Qt.UserRole): i for i in range(w.batch_list.count())}[name]

    def click():
        w._key_escape()
        w.batch_list.clearSelection(); w.batch_list.setCurrentRow(row); w.batch_clicked()

    click()
    w.best_x_set.setValue(w.best_x_set.value() + 0.7); w.best_z_set.setValue(w.best_z_set.value() - 0.3)
    w.save_batch_reference()
    saved = dict(w.saved_sets[name]["best_focus"])
    click()
    assert (w.best_x_set.value(), w.best_z_set.value()) == pytest.approx((saved["x_best"], saved["z_best"]))
    w.select_batch()
    assert w.active_reference_set == name
    assert (w.best_x_set.value(), w.best_z_set.value()) == pytest.approx((saved["x_best"], saved["z_best"]))
    # An unsaved edit made after clicking survives Select batch too.
    click()
    w.best_x_set.setValue(saved["x_best"] + 0.2)
    w.select_batch()
    assert w.best_x_set.value() == pytest.approx(saved["x_best"] + 0.2)
    assert w.saved_sets[name]["best_focus"]["x_best"] == pytest.approx(saved["x_best"])
    w.close()


def test_reload_stage_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    from datetime import datetime
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1, per_step=9)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.load_position_file(str(d / "stage.csv")); w.apply_filters()
    zs = {r.particle_id: r.stage_y for r in w.catalog.records}
    text = (d / "stage.csv").read_text().splitlines()
    head, rows = text[0], text[1:]
    cols = head.split(",")
    k = cols.index("stage_z_original")
    edited = [",".join(c if i != k else str(float(c) + 1) for i, c in enumerate(r.split(","))) for r in rows]
    (d / "stage.csv").write_text("\n".join([head] + edited) + "\n")
    assert w.reload_stage_positions() > 0
    assert all(abs(r.stage_y - zs[r.particle_id] - 1) < 1e-9 for r in w.catalog.records if r.stage_y == r.stage_y)
    w.close()


def test_copy_project_onto_retimed_reprocessed_data(tmp_path, monkeypatch):
    """Flags survive reprocessing that renumbers ROIs and changes their times (matched by ROI image)."""
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    import shutil
    from datetime import datetime
    import numpy as np
    import scipy.io as sio
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    shown = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2])))
    from calibration.gui import CalibrationWindow
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1, per_step=9)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.apply_filters()
    w._on_box(w.spatial_canvas, 0, (0, 0), (500, 1000), Qt.NoModifier); w.accept_selection()
    kept = {(r.start_x, r.start_y) for r in w.catalog.records if r.status == "keep"}
    assert kept
    batch = "Accepted batch 1"
    by_id = w.catalog._by_id
    assert {(by_id[p].start_x, by_id[p].start_y) for p in w.saved_sets[batch]["particle_ids"]} == kept
    w.save_project(); w.flush_saves()
    # "Reprocessed": same ROIs, reversed order, every time 0.15 s later.
    new_dir = tmp_path / "re"; new_dir.mkdir()
    m = sio.loadmat(d / "scan.mat", squeeze_me=False)
    roi, dat = m["ROI_N"][0, 0], m["dat"][0, 0]
    n = roi["IMAGE"].shape[1]
    rev = np.arange(n)[::-1]
    imgs = np.empty((n,), dtype=[("IM", object)])
    focs = np.empty((n,), dtype=[("focus", object)])
    for k, i in enumerate(rev):
        imgs[k]["IM"] = roi["IMAGE"][0, i]["IM"]
        focs[k]["focus"] = dat["foc"][0, i]["focus"]
    f = lambda a: np.ravel(a)[rev]
    sio.savemat(new_dir / "scan.mat", {
        "ROI_N": {"Time": f(roi["Time"]) + 0.15 / 86400, "StartX": f(roi["StartX"]), "EndX": f(roi["EndX"]),
                  "StartY": f(roi["StartY"]), "EndY": f(roi["EndY"]), "IMAGE": imgs},
        "dat": {"len": f(dat["len"]), "wid": f(dat["wid"]), "round": f(dat["round"]),
                "centroid": np.ravel(dat["centroid"]).reshape(-1, 2)[rev] if dat["centroid"].size == 2 * n
                else dat["centroid"][rev], "foc": focs}})
    out = w.copy_project_to_reprocessed(new_dir, tmp_path / "p_re.json")
    assert out is not None
    now = {(r.start_x, r.start_y) for r in w.catalog.records if r.status == "keep"}
    assert now == kept
    by_id = w.catalog._by_id
    members = w.saved_sets[batch]["particle_ids"]
    assert all(p in by_id for p in members)                   # every batch member found in the new files
    assert {(by_id[p].start_x, by_id[p].start_y) for p in members} == kept
    assert all(by_id[p].status == "keep" for p in members)
    assert any("Matched" in s for s in shown)
    w.close()


def test_normalised_distance_axis(tmp_path, monkeypatch):
    monkeypatch.setenv("CPI_CALIBRATION_LABELS", str(tmp_path / "labels.json"))
    import math
    from datetime import datetime
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    from calibration.gui import CalibrationWindow
    from mat_synth import make_scan
    d = make_scan(tmp_path / "run", t0=datetime(2026, 3, 2, 10), seed=1, per_step=9)
    w = CalibrationWindow()
    w.project_file = (tmp_path / "p.json").resolve(); w.selection_file = tmp_path / "p_flags.csv"
    w.import_file(d / "scan.mat"); w.load_position_file(str(d / "stage.csv")); w.apply_filters()
    w._on_box(w.spatial_canvas, 0, (0, 0), (1000, 1000), Qt.NoModifier); w.accept_selection()
    rec = next(r for r in w.catalog.records if r.status == "keep")
    w.position_metric.setCurrentIndex(w.position_metric.findData("plane_signed"))
    dist = w._plot_x(rec)
    w.position_metric.setCurrentIndex(w.position_metric.findData("plane_zd"))
    assert math.isnan(w._plot_x(rec))                     # no actual size yet
    rec.actual_length_um = 40.0
    assert w._plot_x(rec) == pytest.approx(0.85 * dist * 1000 / 20.0 ** 2)
    w.close()
