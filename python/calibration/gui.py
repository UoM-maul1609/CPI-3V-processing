"""PySide6 GUI for interactive CPI calibration-particle selection.

Typical use::

    cpi-calibrate /path/to/processed/calibration

or::

    python -m calibration /path/to/processed/calibration

Selection state is stored in a CSV sidecar. The processed MAT files are never
modified by the GUI.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from .data import FilterSpec, ParticleCatalog, ParticleRecord, ParticleStore, datetime_to_matlab_datenum
except ImportError:  # Support ``python gui.py`` from this directory.
    from data import FilterSpec, ParticleCatalog, ParticleRecord, ParticleStore, datetime_to_matlab_datenum


class ParticleCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(7, 7), constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def show_particle(self, image: np.ndarray, boundary: Optional[np.ndarray], title: str) -> None:
        self.axes.clear()
        self.axes.imshow(image, cmap="gray", origin="upper")
        if boundary is not None:
            finite = np.isfinite(boundary).all(axis=1)
            b = boundary[finite]
            if len(b):
                self.axes.plot(b[:, 1], b[:, 0], linewidth=1.2)
        self.axes.set_title(title)
        self.axes.set_xlabel("ROI x [px]")
        self.axes.set_ylabel("ROI y [px]")
        self.axes.set_aspect("equal")
        self.draw_idle()


class CalibrationWindow(QMainWindow):
    def __init__(self, initial_directory: Optional[str] = None, position_file: Optional[str] = None,
                 z_min_um: float = -10000.0, z_max_um: float = 10000.0, z_step_um: float = 10.0):
        super().__init__()
        self.setWindowTitle("CPI calibration particle editor")
        self.resize(1250, 850)

        self.catalog: Optional[ParticleCatalog] = None
        self.store: Optional[ParticleStore] = None
        self.filtered = []
        self.cursor = 0
        self.selection_file: Optional[Path] = None
        self.position_file = position_file
        self.z_min_um = z_min_um
        self.z_max_um = z_max_um
        self.z_step_um = z_step_um
        self._updating_z = False

        self.canvas = ParticleCanvas()
        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self.setStatusBar(QStatusBar())

        if initial_directory:
            self.load_directory(initial_directory, position_file=position_file)

    def _build_ui(self) -> None:
        central = QWidget()
        main = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._build_control_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        nav = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(lambda: self.navigate(-1))
        self.next_button.clicked.connect(lambda: self.navigate(1))
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        layout.addLayout(nav)

        self.progress_label = QLabel("No calibration directory loaded")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        metrics_box = QGroupBox("Particle")
        metrics = QFormLayout(metrics_box)
        self.metric_labels: Dict[str, QLabel] = {}
        for key, label in (
            ("time", "Time"), ("length", "Length [µm]"), ("width", "Width [µm]"),
            ("focus", "Focus"), ("roundness", "Roundness"), ("xy", "Image x, y [px]"),
            ("stage", "Stage x, y"), ("source", "Source"),
        ):
            widget = QLabel("—")
            widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metric_labels[key] = widget
            metrics.addRow(label, widget)
        layout.addWidget(metrics_box)

        z_box = QGroupBox("Object-plane position")
        z_layout = QVBoxLayout(z_box)
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(max(1, int(round((self.z_max_um - self.z_min_um) / self.z_step_um))))
        self.z_slider.valueChanged.connect(self._slider_changed)
        self.z_slider.sliderReleased.connect(self.autosave)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(self.z_min_um, self.z_max_um)
        self.z_spin.setSingleStep(self.z_step_um)
        self.z_spin.setDecimals(1)
        self.z_spin.setSuffix(" µm")
        self.z_spin.valueChanged.connect(self._spin_changed)
        self.z_spin.editingFinished.connect(self.autosave)
        z_layout.addWidget(self.z_slider)
        z_layout.addWidget(self.z_spin)
        layout.addWidget(z_box)

        decision_box = QGroupBox("Decision")
        decision = QHBoxLayout(decision_box)
        self.keep_button = QPushButton("Keep")
        self.reject_button = QPushButton("Reject")
        self.undecided_button = QPushButton("Undecided")
        self.keep_button.clicked.connect(lambda: self.set_status("keep"))
        self.reject_button.clicked.connect(lambda: self.set_status("reject"))
        self.undecided_button.clicked.connect(lambda: self.set_status("undecided"))
        decision.addWidget(self.keep_button)
        decision.addWidget(self.reject_button)
        decision.addWidget(self.undecided_button)
        layout.addWidget(decision_box)

        self.background_checkbox = QCheckBox("Show background-subtracted ROI")
        self.background_checkbox.stateChanged.connect(lambda _: self.refresh_particle())
        layout.addWidget(self.background_checkbox)

        filter_box = QGroupBox("Review filters (non-destructive)")
        filter_layout = QGridLayout(filter_box)
        filter_layout.addWidget(QLabel("Metric"), 0, 0)
        filter_layout.addWidget(QLabel("Minimum"), 0, 1)
        filter_layout.addWidget(QLabel("Maximum"), 0, 2)
        self.filter_fields: Dict[str, tuple[QLineEdit, QLineEdit]] = {}
        row = 1
        for key, label in (("length", "Length [µm]"), ("width", "Width [µm]"), ("focus", "Focus"),
                           ("x", "Image x [px]"), ("y", "Image y [px]")):
            low, high = QLineEdit(), QLineEdit()
            low.setPlaceholderText("any")
            high.setPlaceholderText("any")
            self.filter_fields[key] = (low, high)
            filter_layout.addWidget(QLabel(label), row, 0)
            filter_layout.addWidget(low, row, 1)
            filter_layout.addWidget(high, row, 2)
            row += 1

        self.time_low = QLineEdit()
        self.time_high = QLineEdit()
        self.time_low.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.time_high.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        filter_layout.addWidget(QLabel("Time"), row, 0)
        filter_layout.addWidget(self.time_low, row, 1)
        filter_layout.addWidget(self.time_high, row, 2)
        row += 1

        self.status_filter = QComboBox()
        self.status_filter.addItems(["all", "undecided", "keep", "reject"])
        filter_layout.addWidget(QLabel("Decision"), row, 0)
        filter_layout.addWidget(self.status_filter, row, 1, 1, 2)
        row += 1

        apply_button = QPushButton("Apply filters")
        clear_button = QPushButton("Clear filters")
        apply_button.clicked.connect(self.apply_filters)
        clear_button.clicked.connect(self.clear_filters)
        filter_layout.addWidget(apply_button, row, 1)
        filter_layout.addWidget(clear_button, row, 2)
        layout.addWidget(filter_box)

        layout.addStretch(1)
        return panel

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open processed directory…", self)
        open_action.triggered.connect(self.choose_directory)
        file_menu.addAction(open_action)
        position_action = QAction("Attach position log…", self)
        position_action.triggered.connect(self.choose_position_file)
        file_menu.addAction(position_action)
        save_as = QAction("Save selection as…", self)
        save_as.triggered.connect(self.save_as)
        file_menu.addAction(save_as)

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.navigate(1))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.navigate(-1))
        QShortcut(QKeySequence("K"), self, activated=lambda: self.set_status("keep"))
        QShortcut(QKeySequence("R"), self, activated=lambda: self.set_status("reject"))
        QShortcut(QKeySequence("U"), self, activated=lambda: self.set_status("undecided"))
        QShortcut(QKeySequence("+"), self, activated=lambda: self.z_spin.setValue(self.z_spin.value() + self.z_step_um))
        QShortcut(QKeySequence("-"), self, activated=lambda: self.z_spin.setValue(self.z_spin.value() - self.z_step_um))

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Processed CPI calibration directory")
        if directory:
            self.load_directory(directory, position_file=self.position_file)

    def choose_position_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Calibration position log", "", "Text files (*.txt);;All files (*)")
        if filename:
            self.position_file = filename
            if self.catalog:
                from .data import PositionLog
                try:
                    self.catalog.attach_position_log(PositionLog.from_text(filename))
                    self.refresh_particle()
                except Exception as exc:
                    QMessageBox.critical(self, "Position log", str(exc))

    def load_directory(self, directory: str, position_file: Optional[str] = None) -> None:
        root = Path(directory).resolve()
        self.statusBar().showMessage("Building particle catalog…")
        QApplication.processEvents()

        def progress(number, total, filename):
            self.statusBar().showMessage("Scanning {} / {}: {}".format(number, total, filename.name))
            QApplication.processEvents()

        try:
            catalog = ParticleCatalog.from_directory(root, position_file=position_file, disposable_scanners=True, progress=progress)
        except Exception as exc:
            QMessageBox.critical(self, "Load calibration", str(exc))
            return

        self.catalog = catalog
        self.store = ParticleStore(root)
        self.selection_file = root / "calibration_selection.csv"
        self.catalog.apply_selection_file(self.selection_file)
        self.filtered = list(range(len(catalog.records)))
        self.cursor = 0
        self.statusBar().showMessage("Loaded {} particles".format(len(catalog.records)), 5000)
        self.refresh_particle()

    def current_record(self) -> Optional[ParticleRecord]:
        if not self.catalog or not self.filtered:
            return None
        self.cursor = max(0, min(self.cursor, len(self.filtered) - 1))
        return self.catalog.records[self.filtered[self.cursor]]

    def navigate(self, delta: int) -> None:
        if not self.filtered:
            return
        self.autosave()
        self.cursor = max(0, min(len(self.filtered) - 1, self.cursor + delta))
        self.refresh_particle()

    def refresh_particle(self) -> None:
        record = self.current_record()
        if record is None or self.store is None:
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, "No particles match the current filters", ha="center", va="center")
            self.canvas.draw_idle()
            self.progress_label.setText("0 particles")
            return
        try:
            image, boundary = self.store.particle(record, subtract_background=self.background_checkbox.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "Particle image", str(exc))
            return

        self.canvas.show_particle(image, boundary, "{} — {}".format(record.status.upper(), record.particle_id))
        self.metric_labels["time"].setText(record.time_iso or "—")
        self.metric_labels["length"].setText(self._format(record.length_um, 2))
        self.metric_labels["width"].setText(self._format(record.width_um, 2))
        self.metric_labels["focus"].setText(self._format(record.focus, 3))
        self.metric_labels["roundness"].setText(self._format(record.roundness, 3))
        self.metric_labels["xy"].setText("{}, {}".format(self._format(record.image_x_px, 1), self._format(record.image_y_px, 1)))
        self.metric_labels["stage"].setText("{}, {}".format(self._format(record.stage_x, 3), self._format(record.stage_y, 3)))
        self.metric_labels["source"].setText("{} [{}]".format(record.source_file, record.roi_index))

        self._updating_z = True
        z = min(self.z_max_um, max(self.z_min_um, record.object_plane_um))
        self.z_spin.setValue(z)
        self.z_slider.setValue(int(round((z - self.z_min_um) / self.z_step_um)))
        self._updating_z = False

        counts = {"keep": 0, "reject": 0, "undecided": 0}
        if self.catalog:
            for item in self.catalog.records:
                counts[item.status] = counts.get(item.status, 0) + 1
        self.progress_label.setText(
            "{} / {} visible ({} total) — keep {}, reject {}, undecided {}".format(
                self.cursor + 1, len(self.filtered), len(self.catalog.records),
                counts.get("keep", 0), counts.get("reject", 0), counts.get("undecided", 0)
            )
        )

    @staticmethod
    def _format(value: float, digits: int) -> str:
        return ("{:.%df}" % digits).format(value) if math.isfinite(value) else "—"

    def _slider_changed(self, index: int) -> None:
        if self._updating_z:
            return
        self._updating_z = True
        value = self.z_min_um + index * self.z_step_um
        self.z_spin.setValue(value)
        self._updating_z = False
        self._set_current_z(value)

    def _spin_changed(self, value: float) -> None:
        if self._updating_z:
            return
        self._updating_z = True
        index = int(round((value - self.z_min_um) / self.z_step_um))
        self.z_slider.setValue(index)
        self._updating_z = False
        self._set_current_z(value)

    def _set_current_z(self, value: float) -> None:
        record = self.current_record()
        if record is not None:
            # Keep drag interaction responsive.  Persistence happens when the
            # slider is released, the spin edit finishes, a decision is made,
            # or the window closes.
            record.object_plane_um = float(value)

    def set_status(self, status: str) -> None:
        record = self.current_record()
        if record is None:
            return
        record.status = status
        self.autosave()
        # Review is fastest if a decision advances to the next candidate.
        if self.cursor < len(self.filtered) - 1:
            self.cursor += 1
        self.refresh_particle()

    def autosave(self) -> None:
        if self.catalog and self.selection_file:
            self.catalog.save_selection_file(self.selection_file)

    def save_as(self) -> None:
        if not self.catalog:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save calibration selection", str(self.selection_file or "calibration_selection.csv"), "CSV files (*.csv)")
        if filename:
            self.selection_file = Path(filename)
            self.autosave()

    def _read_float(self, widget: QLineEdit) -> Optional[float]:
        text = widget.text().strip()
        if not text:
            return None
        return float(text)

    def _read_time(self, widget: QLineEdit) -> Optional[float]:
        text = widget.text().strip()
        if not text:
            return None
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime_to_matlab_datenum(datetime.strptime(text, pattern))
            except ValueError:
                pass
        raise ValueError("Time must look like YYYY-MM-DD HH:MM:SS")

    def apply_filters(self) -> None:
        if not self.catalog:
            return
        try:
            values = {}
            for key, (low, high) in self.filter_fields.items():
                values[key + "_min"] = self._read_float(low)
                values[key + "_max"] = self._read_float(high)
            spec = FilterSpec(
                length_min=values["length_min"], length_max=values["length_max"],
                width_min=values["width_min"], width_max=values["width_max"],
                focus_min=values["focus_min"], focus_max=values["focus_max"],
                x_min=values["x_min"], x_max=values["x_max"],
                y_min=values["y_min"], y_max=values["y_max"],
                time_min=self._read_time(self.time_low), time_max=self._read_time(self.time_high),
                status=self.status_filter.currentText(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Filters", str(exc))
            return
        self.filtered = self.catalog.filtered_indices(spec)
        self.cursor = 0
        self.refresh_particle()

    def clear_filters(self) -> None:
        for low, high in self.filter_fields.values():
            low.clear()
            high.clear()
        self.time_low.clear()
        self.time_high.clear()
        self.status_filter.setCurrentText("all")
        if self.catalog:
            self.filtered = list(range(len(self.catalog.records)))
        self.cursor = 0
        self.refresh_particle()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        self.autosave()
        event.accept()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Review CPI calibration particles one at a time")
    parser.add_argument("directory", nargs="?", help="directory containing processed MAT files")
    parser.add_argument("--position-file", help="optional x/y calibration position text file")
    parser.add_argument("--z-min", type=float, default=-10000.0, help="object-plane slider minimum [µm]")
    parser.add_argument("--z-max", type=float, default=10000.0, help="object-plane slider maximum [µm]")
    parser.add_argument("--z-step", type=float, default=10.0, help="object-plane slider step [µm]")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    window = CalibrationWindow(args.directory, args.position_file, args.z_min, args.z_max, args.z_step)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
