#!/usr/bin/env python3
"""Window to choose which full-frame backgrounds the processing uses.

    python backgroundSelector.py DATA_FOLDER

Shows every background in the folder's full_backgrounds.mat as a thumbnail with
a tick box; click one to see it large.  "Spots" counts dark patches against the
median of all the backgrounds -- usually particles sitting in that background
(a slide that did not move, contamination); the large view can show that
difference directly.  "Use ticked" saves background_selection.json in the
folder and exits with status 0; Cancel exits with status 1.  If a selection was
saved before, you are first asked whether to use it as it is.

Normally run for you by ``batchProcess.py --backgrounds choose``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backgroundSelection import (dark_spots, datenum_to_str, load_frames, mask_from_selection,  # noqa: E402
                                 read_selection, save_selection, selection_path)

THUMB_W = 200


def _to_qimage(a: np.ndarray, QtGui, ratio: bool = False):
    """uint8 grey QImage of a frame (stored [x, y], shown with y down the screen)."""
    a = np.asarray(a, float).T
    if ratio:
        a = np.clip((a - 0.6) / (1.15 - 0.6), 0, 1) * 255.0
    a = np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8))
    h, w = a.shape
    img = QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return img.copy()


def run(folder: str) -> int:
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    times, frames = load_frames(folder)
    if not frames:
        QtWidgets.QMessageBox.warning(None, "Backgrounds", f"No full-frame backgrounds in\n{folder}")
        return 1

    previous = None
    if os.path.exists(selection_path(folder)):
        try:
            previous = read_selection(selection_path(folder))
        except (OSError, ValueError):
            previous = None
    prev_use = mask_from_selection(previous, times) if previous else None

    if prev_use is not None:
        box = QtWidgets.QMessageBox()
        box.setWindowTitle("Backgrounds")
        box.setText(f"A background selection was saved for this folder on {previous.get('saved', '?')}:\n"
                    f"{int(prev_use.sum())} of {len(times)} backgrounds used.")
        use_btn = box.addButton("Use previous selection", QtWidgets.QMessageBox.AcceptRole)
        review_btn = box.addButton("Review / change", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(use_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is use_btn:
            save_selection(folder, times, prev_use, note=previous.get("note", ""))
            return 0
        if clicked is not review_btn:
            return 1

    spots, ratios = dark_spots(frames)

    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle(f"Choose backgrounds — {folder}")
    dlg.resize(1400, 850)
    layout = QtWidgets.QVBoxLayout(dlg)
    info = QtWidgets.QLabel(
        f"{len(frames)} full-frame backgrounds. Tick the ones to use: each particle is divided by the nearest "
        "ticked background in time. 'spots' = dark patches compared with the median of all backgrounds "
        "(usually particles sitting in that background).")
    info.setWordWrap(True)
    layout.addWidget(info)

    split = QtWidgets.QSplitter()
    layout.addWidget(split, 1)
    grid = QtWidgets.QListWidget()
    grid.setViewMode(QtWidgets.QListView.IconMode)
    grid.setResizeMode(QtWidgets.QListView.Adjust)
    grid.setMovement(QtWidgets.QListView.Static)
    grid.setSpacing(6)
    split.addWidget(grid)

    right = QtWidgets.QWidget()
    rlay = QtWidgets.QVBoxLayout(right)
    preview = QtWidgets.QLabel()
    preview.setAlignment(QtCore.Qt.AlignCenter)
    preview.setMinimumSize(500, 400)
    preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
    caption = QtWidgets.QLabel()
    show_diff = QtWidgets.QCheckBox("Show difference from the median background (particles in it show dark)")
    rlay.addWidget(preview, 1)
    rlay.addWidget(caption)
    rlay.addWidget(show_diff)
    split.addWidget(right)
    split.setSizes([800, 600])

    initial = prev_use if prev_use is not None else np.ones(len(frames), bool)
    thumbs = []
    order = np.argsort(times, kind="stable")          # shown in time order; item data = frame index
    for j in order:
        j = int(j)
        t, f = times[j], frames[j]
        img = _to_qimage(f, QtGui)
        pix = QtGui.QPixmap.fromImage(img).scaledToWidth(THUMB_W, QtCore.Qt.SmoothTransformation)
        thumbs.append(pix)
        s = spots[j]
        label = f"#{j}  {datenum_to_str(t)[11:]}\n" + ("size differs" if s < 0 else f"{s} spot{'s' if s != 1 else ''}")
        item = QtWidgets.QListWidgetItem(QtGui.QIcon(pix), label)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked if initial[j] else QtCore.Qt.Unchecked)
        item.setToolTip(f"{datenum_to_str(t)}  —  frame {j}")
        item.setData(QtCore.Qt.UserRole, j)
        grid.addItem(item)
    grid.setIconSize(QtCore.QSize(THUMB_W, int(THUMB_W * thumbs[0].height() / max(thumbs[0].width(), 1))))

    state = {"row": 0}

    def frame_of(row):
        return int(grid.item(row).data(QtCore.Qt.UserRole))

    def ticked():
        use = np.zeros(len(frames), bool)
        for r in range(grid.count()):
            use[frame_of(r)] = grid.item(r).checkState() == QtCore.Qt.Checked
        return use

    def show(row=None):
        if row is None:
            row = state["row"]
        if row < 0 or row >= len(frames):
            return
        state["row"] = row
        row = frame_of(row)
        diff = show_diff.isChecked() and ratios[row] is not None
        img = _to_qimage(np.repeat(np.repeat(ratios[row], 4, 0), 4, 1), QtGui, ratio=True) if diff \
            else _to_qimage(frames[row], QtGui)
        pix = QtGui.QPixmap.fromImage(img)
        preview.setPixmap(pix.scaled(preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        caption.setText(f"Background #{row}   {datenum_to_str(times[row])}   {spots[row]} spots   "
                        f"{'used' if ticked()[row] else 'NOT used'}")

    def update_count(*_):
        n = int(ticked().sum())
        ok.setText(f"Use ticked ({n} of {len(frames)})")
        ok.setEnabled(n > 0)
        show()

    grid.currentRowChanged.connect(show)
    grid.itemChanged.connect(update_count)
    show_diff.toggled.connect(lambda *_: show())

    buttons = QtWidgets.QHBoxLayout()
    layout.addLayout(buttons)

    def set_all(flag):
        grid.blockSignals(True)
        for j in range(grid.count()):
            grid.item(j).setCheckState(QtCore.Qt.Checked if flag else QtCore.Qt.Unchecked)
        grid.blockSignals(False)
        update_count()

    def untick_spotty():
        grid.blockSignals(True)
        for r in range(grid.count()):
            if spots[frame_of(r)] > spin.value():
                grid.item(r).setCheckState(QtCore.Qt.Unchecked)
        grid.blockSignals(False)
        update_count()

    def load_previous():
        if prev_use is None:
            return
        grid.blockSignals(True)
        for r in range(grid.count()):
            grid.item(r).setCheckState(QtCore.Qt.Checked if prev_use[frame_of(r)] else QtCore.Qt.Unchecked)
        grid.blockSignals(False)
        update_count()

    b_all = QtWidgets.QPushButton("Tick all")
    b_none = QtWidgets.QPushButton("Untick all")
    b_all.clicked.connect(lambda: set_all(True))
    b_none.clicked.connect(lambda: set_all(False))
    spin = QtWidgets.QSpinBox()
    spin.setRange(0, 10000)
    spin.setValue(0)
    b_spot = QtWidgets.QPushButton("Untick those with more spots than")
    b_spot.clicked.connect(untick_spotty)
    b_prev = QtWidgets.QPushButton("Previous selection")
    b_prev.setEnabled(prev_use is not None)
    b_prev.clicked.connect(load_previous)
    for w in (b_all, b_none, b_spot, spin, b_prev):
        buttons.addWidget(w)
    buttons.addStretch(1)
    cancel = QtWidgets.QPushButton("Cancel")
    ok = QtWidgets.QPushButton()
    ok.setDefault(True)
    cancel.clicked.connect(dlg.reject)
    ok.clicked.connect(dlg.accept)
    buttons.addWidget(cancel)
    buttons.addWidget(ok)

    grid.setCurrentRow(0)
    update_count()
    QtCore.QTimer.singleShot(0, show)
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return 1
    use = ticked()
    path = save_selection(folder, times, use)
    print(f"Background selection saved: {int(use.sum())} of {len(use)} used -> {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: backgroundSelector.py DATA_FOLDER")
    sys.exit(run(sys.argv[1]))
