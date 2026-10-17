"""PySide6 workbench for interactive CPI calibration-particle selection.

Typical use::

    cpi-calibrate /path/to/processed/calibration

or::

    python -m calibration /path/to/processed/calibration

Workflow (v11)
--------------
1. Drag a box on the *Camera X-Y* plot: unreviewed particles turn BLUE
   (selected).  Accepted (green) particles are skipped, so a cluster is never
   picked twice.
2. On the *Focus* / *Length* plots beside it, Ctrl/Cmd-drag around outliers to
   drop them from the selection, or use the *Reject* box to flag them red.
3. Press **Accept selection** (K): the selection turns GREEN and is saved.
   **Unflag** (U) returns particles to unreviewed (grey).  Ctrl+Z undoes.
4. Check images in the *Gallery* or *Particle* tabs; remove or reject single
   particles there.

Review flags are stored in a CSV sidecar. The processed MAT files are never
modified by the GUI.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PySide6.QtCore import Qt, QDateTime, QRect, QSize, QTimer, QPoint
from PySide6.QtGui import (QAction, QColor, QCursor, QFont, QIcon, QImage, QKeySequence,
                           QPainter, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDateTimeEdit,
    QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QStatusBar, QTabWidget,
    QToolButton, QVBoxLayout, QWidget,
)

try:
    from .plot_utils import data_limits, median_trace, trace_extremum
    from .selection_state import (apply_box, normalise_sets, make_saved_set, exclusive_batch_ids,
                                  add_ids_to_saved_set, remove_ids_from_saved_set)
    from .reference_sets import (make_reference, signed_plane_distance, in_plane_offset, scan_direction_deg, build_reference_index, calibrated_radial_position,
                                 corrected_stage_radius, relative_stage_position,
                                 materialized_reference_rows, committed_particle_ids)
    from .focus_surface import fit_surface, predict_best_radius
    from .metadata import read_truth_table, attach_truth
    from .labels import LabelLibrary, ParticleLabel, apply_label_dimensions, LABEL_KINDS
    from .data import (FilterSpec, ParticleCatalog, ParticleRecord, ParticleStore, PositionLog,
                       datetime_to_matlab_datenum, matlab_datenum_to_datetime, _scan_via_subprocess)
except ImportError:  # Support ``python gui.py`` from this directory.
    from plot_utils import data_limits, median_trace, trace_extremum
    from selection_state import (apply_box, normalise_sets, make_saved_set, exclusive_batch_ids,
                                 add_ids_to_saved_set, remove_ids_from_saved_set)
    from reference_sets import (make_reference, signed_plane_distance, in_plane_offset, scan_direction_deg, build_reference_index, calibrated_radial_position,
                                corrected_stage_radius, relative_stage_position,
                                materialized_reference_rows, committed_particle_ids)
    from focus_surface import fit_surface, predict_best_radius
    from metadata import read_truth_table, attach_truth
    from labels import LabelLibrary, ParticleLabel, apply_label_dimensions, LABEL_KINDS
    from data import (FilterSpec, ParticleCatalog, ParticleRecord, ParticleStore, PositionLog,
                      datetime_to_matlab_datenum, matlab_datenum_to_datetime, _scan_via_subprocess)


# ----------------------------------------------------------------------------
# One colour language, used on every plot, thumbnail, button and badge.
# ----------------------------------------------------------------------------
STATUS_COLOUR = {"undecided": "#98a3ae", "keep": "#2b9a48", "reject": "#e03b3b"}
STATUS_NAME = {"undecided": "Unreviewed", "keep": "Accepted", "reject": "Rejected"}
SELECTED_COLOUR = "#1c64d6"
COMMITTED_COLOUR = "#0b8f87"
CURRENT_COLOUR = "#f08c00"
MEDIAN_COLOUR = "#13294b"

METRICS = (("focus", "Focus"), ("length", "Length [µm]"), ("width", "Width [µm]"),
           ("ratio", "Measured / actual length"), ("roundness", "Roundness"))
POSITIONS = (("time", "Acquisition time"), ("stage_x", "Stage X"), ("stage_z", "Stage Z"),
             ("plane_signed", "Signed distance to object plane  d = n·(ΔX, ΔZ)"),
             ("stage_x_relative", "X − Xbest"), ("stage_z_relative", "Z − Zbest"),
             ("plane_offset", "Offset within object plane (θ check)"),
             ("stage_radius", "Stage radius √(X² + Z²)"),
             ("relative", "Zp radial difference (best − measured radius) [legacy]"),
             ("stage_offset_radius", "Zp = √(ΔX² + ΔZ²) [unsigned]"))
BOX_MODES = (
    ("new", "▭ New", "Plain drag: replace the selection with the particles inside the box"),
    ("add", "＋ Add", "Add the particles inside the box (or hold Shift while dragging)"),
    ("remove", "− Remove", "Drop the particles inside the box from the selection, without flagging "
                           "them (or hold Ctrl/⌘ while dragging)"),
    ("keep", "∩ Keep only", "Keep only selected particles that are inside the box (or hold Alt/⌥)"),
    ("reject", "✕ Reject", "Flag particles inside the box as REJECTED (red). With a selection, only "
                           "selected particles inside the box are rejected"),
)
PICK_SCOPES = (("undecided", "Unreviewed only"), ("unrejected", "Unreviewed + accepted"),
               ("keep", "Accepted only"), ("reject", "Rejected only"), ("any", "Everything"))
BROWSE_SCOPES = (("auto", "Selection (all if none)"), ("all", "All particles"),
                 ("selection", "Selection only"), ("undecided", "Unreviewed"),
                 ("keep", "Accepted"), ("reject", "Rejected"))
RELATIVE_AXES = ("stage_x_relative", "stage_z_relative", "stage_offset_radius", "plane_signed", "plane_offset")
STAGE_AXES = ("stage_x", "stage_z", "stage_radius", "relative") + RELATIVE_AXES
THUMB = 112
GALLERY_PAGE = 48


def _to_gray8(arr) -> Optional[QImage]:
    arr = np.asarray(arr)
    if arr.ndim != 2 or not arr.size:
        return None
    img = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    return QImage(img.data, img.shape[1], img.shape[0], img.strides[0],
                  QImage.Format_Grayscale8).copy()


class ParticleCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(6, 6), constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def show_message(self, text):
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.text(.5, .5, text, ha="center", va="center", color="#516579", wrap=True)
        self.draw_idle()

    def show_particle(self, image, boundary, title, show_boundary=True, frame_colour=None):
        self.axes.clear()
        self.axes.set_axis_on()
        self.axes.imshow(image, cmap="gray", origin="upper", interpolation="nearest", aspect="equal")
        if show_boundary and boundary is not None:
            finite = np.isfinite(boundary).all(axis=1)
            b = boundary[finite]
            if len(b):
                # MATLAB boundary convention is row, column; retain exact stored vertices.
                self.axes.plot(b[:, 1], b[:, 0], linewidth=1.15, color="#ff654d")
        self.axes.set_title(title, color=frame_colour or "#192b40", fontweight="bold")
        for spine in self.axes.spines.values():
            spine.set_edgecolor(frame_colour or "#8a99a8")
            spine.set_linewidth(4 if frame_colour else 1)
        self.axes.set_xlabel("ROI x [px]")
        self.axes.set_ylabel("ROI y [px]")
        self.draw_idle()


class LiveSelectionCanvas(FigureCanvasQTAgg):
    """Figure canvas with Qt-painted drag boxes over one or more axes.

    The box is painted by Qt during the drag (instant feedback, independent of
    Matplotlib redraws) and coloured by the effective box mode, so the user can
    see whether a drag will select, add, remove or reject.  Zoom and Pan are
    left to Matplotlib's (hidden) navigation toolbar.  The mouse wheel zooms
    about the cursor in every mode.
    """
    MODE_STYLE = {"new": ("#1c64d6", Qt.SolidLine), "add": ("#1c64d6", Qt.DashLine),
                  "keep": ("#6741d9", Qt.DashDotLine), "remove": ("#e8590c", Qt.DashLine),
                  "reject": ("#e03b3b", Qt.SolidLine)}

    def __init__(self, figure, axes_list, kind):
        super().__init__(figure)
        self.kind = kind
        self.axes_list = list(axes_list)
        self.selection_enabled = True
        self.reference_enabled = False
        self.mode_for = lambda modifiers: "new"
        self.on_box = self.on_click = self.on_double_click = None
        self.on_hover = self.on_leave = self.on_wheel = None
        self.on_reference_changed = self.on_navigation_changed = None
        self._drag = None        # [axes index, start QPoint, end QPoint, modifiers]
        self._reference = None   # [axes index, QPoint]
        self.setMouseTracking(True)

    def set_axes(self, axes_list):
        self.axes_list = list(axes_list)

    def set_select_mode(self, enabled):
        self.selection_enabled = bool(enabled)
        self._drag = None
        self.update()

    def set_reference_mode(self, enabled):
        self.reference_enabled = bool(enabled)
        self._reference = None
        self.update()

    def axes_index_at(self, point):
        x, y = self.mouseEventCoords(point)
        for index, ax in enumerate(self.axes_list):
            if ax.contains_point((x, y)):
                return index
        return None

    def data_at(self, index, point):
        x, y = self.mouseEventCoords(point)
        return self.axes_list[index].transData.inverted().transform((x, y))

    def mousePressEvent(self, event):
        point = event.position().toPoint()
        index = self.axes_index_at(point)
        if event.button() == Qt.LeftButton and index is not None:
            if self.reference_enabled:
                self._reference = [index, point]
                self.update(); event.accept(); return
            if self.selection_enabled:
                self._drag = [index, point, point, event.modifiers()]
                self.update(); event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = event.position().toPoint()
        if self._reference is not None:
            self._reference[1] = point
            self.update(); event.accept(); return
        if self._drag is not None:
            self._drag[2] = point
            self._drag[3] = event.modifiers()
            self.update(); event.accept(); return
        super().mouseMoveEvent(event)
        if self.selection_enabled and self.on_hover is not None and not event.buttons():
            self.on_hover(self, self.axes_index_at(point), point)

    def leaveEvent(self, event):
        if self.on_leave is not None:
            self.on_leave()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        point = event.position().toPoint()
        if self._reference is not None and event.button() == Qt.LeftButton:
            index = self._reference[0]
            self._reference = None
            self.update()
            value = float(self.data_at(index, point)[0])
            if math.isfinite(value) and self.on_reference_changed is not None:
                self.on_reference_changed(value)
            event.accept(); return
        if self._drag is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            if self.on_navigation_changed is not None:
                self.on_navigation_changed(self.kind)
            return
        index, start, _end, _mods = self._drag
        modifiers = event.modifiers()
        self._drag = None
        self.update()
        if abs(point.x() - start.x()) < 4 and abs(point.y() - start.y()) < 4:
            if self.on_click is not None:
                self.on_click(self, index, point, modifiers)
            event.accept(); return
        a, b = self.data_at(index, start), self.data_at(index, point)
        if np.isfinite([a[0], a[1], b[0], b[1]]).all() and self.on_box is not None:
            self.on_box(self, index, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), modifiers)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        point = event.position().toPoint()
        index = self.axes_index_at(point)
        if (event.button() == Qt.LeftButton and index is not None and self.selection_enabled
                and self.on_double_click is not None):
            self.on_double_click(self, index, point)
            event.accept(); return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        point = event.position().toPoint()
        index = self.axes_index_at(point)
        if index is not None and self.on_wheel is not None:
            self.on_wheel(self, index, point, event.angleDelta().y())
            event.accept(); return
        super().wheelEvent(event)

    def _axes_rect(self, index):
        bb = self.axes_list[index].bbox
        scale = self.device_pixel_ratio
        return QRect(int(bb.x0 / scale), int((self.figure.bbox.height - bb.y1) / scale),
                     int(bb.width / scale), int(bb.height / scale))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drag is None and self._reference is None:
            return
        painter = QPainter(self)
        if self._reference is not None:
            clip = self._axes_rect(self._reference[0])
            painter.setClipRect(clip)
            pen = QPen(QColor("#d85519"), 3, Qt.DashLine)
            painter.setPen(pen)
            x = self._reference[1].x()
            painter.drawLine(x, clip.top(), x, clip.bottom())
        else:
            index, start, end, modifiers = self._drag
            painter.setClipRect(self._axes_rect(index))
            colour, style = self.MODE_STYLE.get(self.mode_for(modifiers), self.MODE_STYLE["new"])
            fill = QColor(colour)
            fill.setAlpha(40)
            painter.setBrush(fill)
            pen = QPen(QColor(colour), 2.5, style)
            pen.setCosmetic(True)
            painter.setPen(pen)
            rect = QRect(start, end).normalized()
            painter.drawRect(rect)
            label = dict((k, t) for k, t, _ in BOX_MODES).get(self.mode_for(modifiers), "")
            painter.setPen(QColor(colour))
            font = painter.font(); font.setBold(True); painter.setFont(font)
            painter.drawText(rect.topLeft() + QPoint(4, -5), label)
        painter.end()


class HoverPreview(QFrame):
    """Small floating card describing the particle under the cursor."""

    def __init__(self, parent):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #7d91a6; border-radius: 6px; }"
                           "QLabel { border: none; color: #13283e; }")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 8, 6)
        self.image = QLabel()
        self.image.setFixedSize(96, 96)
        self.image.setAlignment(Qt.AlignCenter)
        self.text = QLabel()
        self.text.setTextFormat(Qt.RichText)
        layout.addWidget(self.image)
        layout.addWidget(self.text)
        self.particle_id = None

    def show_at(self, global_pos, html, pixmap=None):
        self.text.setText(html)
        if pixmap is not None:
            self.image.setPixmap(pixmap)
            self.image.setText("")
        else:
            self.image.setPixmap(QPixmap())
            self.image.setText("<span style='color:#8292a3;font-size:10px'>click to<br>load image</span>")
        self.adjustSize()
        self.move(global_pos + QPoint(18, 14))
        self.show()
        self.raise_()


class LabelDialog(QDialog):
    """Create or edit a shared label: name, type, true size (µm), notes, colour."""

    def __init__(self, parent, label: Optional[ParticleLabel] = None, used_in_dataset=0):
        super().__init__(parent)
        self.setWindowTitle("Edit label" if label else "New label")
        self.colour = label.colour if label else ""
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(label.name if label else "")
        self.name.setPlaceholderText("e.g. PSL 20 µm beads, Ice analogue column A")
        self.kind = QComboBox()
        self.kind.addItems(LABEL_KINDS)
        if label:
            self.kind.setCurrentText(label.kind)
        self.length = QDoubleSpinBox(); self.width = QDoubleSpinBox()
        for spin in (self.length, self.width):
            spin.setRange(0, 100000); spin.setDecimals(3); spin.setSuffix(" µm"); spin.setSpecialValueText("not set")
        self.length.setValue(label.length_um if label and math.isfinite(label.length_um) else 0)
        self.same = QCheckBox("Width = length (sphere / bead)")
        same = (label is None or not math.isfinite(label.width_um)
                or (math.isfinite(label.length_um) and abs(label.width_um - label.length_um) < 1e-9))
        self.same.setChecked(same)
        self.width.setValue(label.width_um if label and math.isfinite(label.width_um) else 0)
        self.width.setEnabled(not same)
        self.same.toggled.connect(lambda on: self.width.setEnabled(not on))
        self.notes = QLineEdit(label.notes if label else "")
        self.notes.setPlaceholderText("Supplier, lot, certificate, ...")
        self.colour_button = QPushButton("Choose colour…")
        self.colour_button.clicked.connect(self._choose_colour)
        self._paint_colour()
        form.addRow("Name", self.name)
        form.addRow("Type", self.kind)
        form.addRow("True length / diameter", self.length)
        form.addRow("", self.same)
        form.addRow("True width", self.width)
        form.addRow("Notes", self.notes)
        form.addRow("Colour on plots", self.colour_button)
        layout.addLayout(form)
        note = QLabel("Labels are shared by all calibration datasets on this computer. Changing a size here "
                      "updates every particle carrying the label"
                      + (f" ({used_in_dataset:,} in the loaded data)." if used_in_dataset else "."))
        note.setWordWrap(True)
        note.setObjectName("hint")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _paint_colour(self):
        if self.colour:
            self.colour_button.setStyleSheet(f"background:{self.colour}; color:white;")

    def _choose_colour(self):
        chosen = QColorDialog.getColor(QColor(self.colour or "#7048e8"), self, "Label colour")
        if chosen.isValid():
            self.colour = chosen.name()
            self._paint_colour()

    def _accept(self):
        if not self.name.text().strip():
            QMessageBox.warning(self, "Label", "Give the label a name.")
            return
        self.accept()

    def result_label(self) -> ParticleLabel:
        length = self.length.value() or float("nan")
        width = length if self.same.isChecked() else (self.width.value() or float("nan"))
        return ParticleLabel(name=self.name.text().strip(), kind=self.kind.currentText(), length_um=length,
                             width_um=width, notes=self.notes.text().strip(), colour=self.colour)


class CalibrationWindow(QMainWindow):
    def __init__(self, initial_directory: Optional[str] = None, position_file: Optional[str] = None,
                 z_min_um: float = -10000.0, z_max_um: float = 10000.0, z_step_um: float = 10.0):
        super().__init__()
        self.setWindowTitle("CPI · Calibration Workbench v11")
        self.setAcceptDrops(True)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        self.resize(min(1500, available.width() - 35) if available else 1440,
                    min(900, available.height() - 65) if available else 860)
        self.setMinimumSize(900, 560)
        self.setStyleSheet(STYLE_SHEET)

        # Data / project state
        self.catalog: Optional[ParticleCatalog] = None
        self.store: Optional[ParticleStore] = None
        self.sources: list[str] = []
        self.additional_roots: set[Path] = set()
        self.project_file: Optional[Path] = None
        self.selection_file: Optional[Path] = None
        self.position_file = position_file
        self.position_series = "beads"
        self.position_time_basis = "roi"
        self.z_min_um, self.z_max_um, self.z_step_um = z_min_um, z_max_um, z_step_um
        self._updating_z = False
        self.stage_segments = []
        self.saved_sets = {}
        self.active_reference_set = None
        self._reference_index = {}
        self._active_members = set()
        self._draft_reference_cache = None
        self._last_filter_spec = None
        try:
            self.labels = LabelLibrary.load()
            self._label_error = None
        except Exception as exc:  # unreadable library: keep working, warn once
            self.labels = LabelLibrary()
            self._label_error = str(exc)

        # Interaction state
        self.selected_ids: set[str] = set()
        self.regions = []          # history of box gestures (kept for project files)
        self.undo_stack = []
        self._base_indices: list[int] = []
        self.filtered: list[int] = []  # the browse list (gallery + particle viewer)
        self._filtered_pos: dict[str, int] = {}
        self.cursor = 0
        self.current_id: Optional[str] = None
        self.plot_tool = "select"
        self.box_mode = "new"
        self._view_locked = {"camera": False, "metric": False}
        self._last_metric_key = None
        self._metric_rows = []
        self._drawn = {}
        self._current_artists = []
        self._colorbars = {"camera": None, "metric": None}
        self._median_peak = {}
        self._thumbs: "OrderedDict[str, QImage]" = OrderedDict()
        self._gallery_page = 0
        self._gallery_dirty = True
        self._restoring = False
        self._import_errors = None

        self.hover = HoverPreview(self)
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._hover_load_image)
        self._hover_pending = None

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self.setStatusBar(QStatusBar())
        self.save_indicator = QLabel("No data loaded")
        self.save_indicator.setStyleSheet("color:#516579; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self.save_indicator)
        self.statusBar().showMessage("Drop MAT files or folders onto the window, or use + Add MAT files")
        self.set_plot_tool("select")
        self._refresh_label_widgets()
        self._plane_changed()
        self._refresh_views()
        if self._label_error:
            QMessageBox.warning(self, "Label library", f"Could not read {self.labels.path}:\n{self._label_error}")
        if initial_directory:
            self.load_directory(initial_directory, position_file=position_file)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 4, 6, 2)
        outer.setSpacing(4)
        outer.addWidget(self._build_selection_bar())
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)
        self.viewer_tabs = QTabWidget()
        self.select_tab = self._build_select_tab()
        self.gallery_panel = self._build_gallery_tab()
        self.particle_tab = self._build_particle_tab()
        self.surface_tab = self._build_surface_tab()
        self.viewer_tabs.addTab(self.select_tab, "①  Select on plots")
        self.viewer_tabs.addTab(self.gallery_panel, "②  Gallery")
        self.viewer_tabs.addTab(self.particle_tab, "③  Particle")
        self.viewer_tabs.addTab(self.surface_tab, "Best-focus map")
        self.viewer_tabs.setTabVisible(3, False)
        self.viewer_tabs.currentChanged.connect(self._tab_changed)
        left_layout.addWidget(self.viewer_tabs, 1)
        left_layout.addWidget(self._build_browse_bar())
        splitter.addWidget(left)
        sidebar = self._build_sidebar()
        sidebar.setMinimumWidth(375)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1110, 390])
        outer.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _button(self, text, slot, role=None, tip=None):
        button = QPushButton(text)
        button.clicked.connect(slot)
        if role:
            button.setProperty("role", role)
        if tip:
            button.setToolTip(tip)
        return button

    def _build_selection_bar(self):
        bar = QFrame()
        bar.setObjectName("selectionBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(6)
        self.selection_summary = QLabel()
        self.selection_summary.setTextFormat(Qt.RichText)
        self.selection_summary.setMinimumWidth(250)
        layout.addWidget(self.selection_summary, 1)
        self.accept_btn = self._button("✓  Accept selection", self.accept_selection, "accept",
                                       "Flag every selected particle ACCEPTED (green) and save. "
                                       "Accepted particles are skipped by new boxes. Key K on the plots tab.")
        self.reject_btn = self._button("✕  Reject selection", self.reject_selection, "reject",
                                       "Flag every selected particle REJECTED (red) and save. Key R on the plots tab.")
        self.unflag_btn = self._button("↺  Unflag selection", self.unflag_selection, "neutral",
                                       "Return selected particles to UNREVIEWED (grey), the default. "
                                       "Key U on the plots tab.")
        self.clear_sel_btn = self._button("Clear selection", self.clear_selection, None,
                                          "Deselect everything (Esc). Flags are not changed.")
        self.undo_btn = self._button("↶  Undo", self.undo, None, "Undo the last selection or flag change (Ctrl+Z)")
        accept_as = QLabel("Accept as")
        accept_as.setToolTip("Label given to particles when you accept them (selection, gallery or viewer)")
        layout.addWidget(accept_as)
        self.accept_label = QComboBox()
        self.accept_label.setMinimumWidth(170)
        self.accept_label.setToolTip("Accepting also tags particles with this label and its true size. "
                                     "Manage labels in the Labels tab.")
        self.accept_label.activated.connect(self._accept_label_chosen)
        layout.addWidget(self.accept_label)
        for button in (self.accept_btn, self.reject_btn, self.unflag_btn, self.clear_sel_btn, self.undo_btn):
            layout.addWidget(button)
        return bar

    def _tool_button(self, text, tip, group, value, attr_list):
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setCheckable(True)
        button.setProperty("value", value)
        group.addButton(button)
        attr_list.append(button)
        return button

    def _build_select_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 2)
        layout.setSpacing(3)

        tools = QHBoxLayout()
        tools.setSpacing(3)
        tools.addWidget(QLabel("Mouse"))
        self.mouse_group = QButtonGroup(self)
        self.mouse_buttons = []
        for value, text, tip in (("select", "▭ Select", "Drag boxes / click dots"),
                                 ("zoom", "🔍 Zoom", "Drag a rectangle to zoom (the wheel also zooms)"),
                                 ("pan", "✋ Pan", "Drag to move the view"),
                                 ("reference", "↔ Xbest line",
                                  "With X = Stage X or Stage Z, drag the orange best-focus line")):
            tools.addWidget(self._tool_button(text, tip, self.mouse_group, value, self.mouse_buttons))
        self.mouse_group.buttonClicked.connect(lambda b: self.set_plot_tool(b.property("value")))
        tools.addSpacing(10)
        tools.addWidget(QLabel("Box"))
        self.box_group = QButtonGroup(self)
        self.box_buttons = []
        for value, text, tip in BOX_MODES:
            button = self._tool_button(text, tip, self.box_group, value, self.box_buttons)
            button.setProperty("boxmode", value)
            tools.addWidget(button)
        self.box_group.buttonClicked.connect(lambda b: self.set_box_mode(b.property("value")))
        tools.addSpacing(8)
        tools.addWidget(QLabel("New/Add boxes pick"))
        self.pick_scope = QComboBox()
        for value, label in PICK_SCOPES:
            self.pick_scope.addItem(label, value)
        self.pick_scope.setToolTip("Which review states a New or Add box can pick up. 'Unreviewed only' "
                                   "stops you re-selecting accepted (green) or rejected (red) particles. "
                                   "Choose 'Accepted only' to grab green particles and unflag them.")
        self.pick_scope.currentIndexChanged.connect(lambda _: self._update_hint())
        tools.addWidget(self.pick_scope)
        tools.addStretch(1)
        tools.addWidget(self._button("⤢ Fit view", self.reset_plot_zoom, None, "Reset zoom on all plots"))
        layout.addLayout(tools)

        self.selection_preview = QLabel()
        self.selection_preview.setWordWrap(True)
        self.selection_preview.setObjectName("hint")
        layout.addWidget(self.selection_preview)

        plots = QSplitter(Qt.Horizontal)
        # Camera X–Y
        camera = QWidget()
        camera_layout = QVBoxLayout(camera)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(2)
        head = QHBoxLayout()
        title = QLabel("Camera X–Y")
        title.setObjectName("panelTitle")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(QLabel("Colour"))
        self.colour_by = QComboBox()
        self.colour_by.addItem("Review status", "status")
        self.colour_by.addItem("Label", "label")
        self.colour_by.addItem("Camera X position", "image_x")
        self.colour_by.addItem("Camera Y position", "image_y")
        self.colour_by.currentIndexChanged.connect(lambda _: self._redraw_plots())
        head.addWidget(self.colour_by)
        camera_layout.addLayout(head)
        self.spatial_figure = Figure(figsize=(5, 5), constrained_layout=True)
        self.spatial_axes = self.spatial_figure.add_subplot(111)
        self.spatial_canvas = LiveSelectionCanvas(self.spatial_figure, [self.spatial_axes], "camera")
        self._wire_canvas(self.spatial_canvas)
        self.spatial_toolbar = NavigationToolbar2QT(self.spatial_canvas, self)
        self.spatial_toolbar.hide()
        camera_layout.addWidget(self.spatial_canvas, 1)
        plots.addWidget(camera)

        # Focus / length vs position (stacked, shared X)
        metric = QWidget()
        metric_layout = QVBoxLayout(metric)
        metric_layout.setContentsMargins(0, 0, 0, 0)
        metric_layout.setSpacing(2)
        head = QHBoxLayout()
        title = QLabel("Focus / size")
        title.setObjectName("panelTitle")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(QLabel("X"))
        self.position_metric = QComboBox()
        for key, label in POSITIONS:
            self.position_metric.addItem(label, key)
        self.position_metric.setMaximumWidth(210)
        head.addWidget(self.position_metric)
        head.addWidget(QLabel("Top"))
        self.plot_metric = QComboBox()
        head.addWidget(self.plot_metric)
        head.addWidget(QLabel("Bottom"))
        self.plot_metric2 = QComboBox()
        self.plot_metric2.addItem("(none)", "none")
        head.addWidget(self.plot_metric2)
        for key, label in METRICS:
            self.plot_metric.addItem(label, key)
            self.plot_metric2.addItem(label, key)
        self.plot_metric2.setCurrentIndex(self.plot_metric2.findData("length"))
        for combo in (self.position_metric, self.plot_metric, self.plot_metric2):
            combo.currentIndexChanged.connect(lambda _: self._change_plot_axes())
        metric_layout.addLayout(head)
        options = QHBoxLayout()
        self.selection_only_cb = QCheckBox("Show only the selection here")
        self.selection_only_cb.setToolTip("Hide unselected particles on the focus/size plots so outliers "
                                          "in the current cluster are easy to spot")
        self.selection_only_cb.toggled.connect(lambda _: self._redraw_plots())
        self.median_cb = QCheckBox("Median ± IQR trace")
        self.median_cb.setChecked(True)
        self.median_cb.setToolTip("Median and inter-quartile band of every plotted metric along X "
                                  "(rejected particles excluded). Dashed blue: median of the selection.")
        self.median_cb.toggled.connect(lambda _: self._redraw_plots())
        options.addWidget(self.selection_only_cb)
        options.addWidget(self.median_cb)
        options.addStretch(1)
        metric_layout.addLayout(options)
        self.plot_figure = Figure(figsize=(6, 6), constrained_layout=True)
        self.plot_axes = self.plot_figure.add_subplot(111)
        self.metric_axes = [self.plot_axes]
        self.plot_canvas = LiveSelectionCanvas(self.plot_figure, self.metric_axes, "metric")
        self._wire_canvas(self.plot_canvas)
        self.plot_canvas.on_reference_changed = self.move_reference_line
        self.plot_toolbar = NavigationToolbar2QT(self.plot_canvas, self)
        self.plot_toolbar.hide()
        metric_layout.addWidget(self.plot_canvas, 1)
        plots.addWidget(metric)
        plots.setSizes([520, 640])
        layout.addWidget(plots, 1)

        legend = QHBoxLayout()
        legend.setSpacing(12)
        legend.addWidget(QLabel("Show:"))
        self.show_status = {}
        for status, symbol in (("undecided", "●"), ("keep", "●"), ("reject", "✕")):
            box = QCheckBox()
            box.setChecked(True)
            box.setProperty("status", status)
            box.setStyleSheet(f"QCheckBox {{ color: {STATUS_COLOUR[status]}; font-weight: 700; }}")
            box.toggled.connect(lambda _: self._redraw_plots())
            box.setToolTip(f"Show or hide {STATUS_NAME[status].lower()} particles on the plots "
                           "(selected particles are always shown)")
            self.show_status[status] = box
            legend.addWidget(box)
        legend.addSpacing(8)
        for text, colour in (("◉ Selected", SELECTED_COLOUR), ("◎ In calibration set", COMMITTED_COLOUR),
                             ("☆ Current particle", CURRENT_COLOUR)):
            label = QLabel(text)
            label.setStyleSheet(f"color: {colour}; font-weight: 700;")
            legend.addWidget(label)
        legend.addStretch(1)
        layout.addLayout(legend)
        return page

    def _wire_canvas(self, canvas):
        canvas.mode_for = self._effective_box_mode
        canvas.on_box = self._on_box
        canvas.on_click = self._on_click
        canvas.on_double_click = self._on_double_click
        canvas.on_hover = self._on_hover
        canvas.on_leave = self._hide_hover
        canvas.on_wheel = self._on_wheel
        canvas.on_navigation_changed = self._lock_plot_navigation

    def _build_gallery_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 2)
        hint = QLabel("Click thumbnails to <b>pick</b> them (amber) — Ctrl/⌘ or Shift-click or drag to pick "
                      "several — then remove or flag them below or with Del / K / R / U. "
                      "Double-click opens a particle. Frames: <b style='color:#1c64d6'>blue</b> selected · "
                      "<b style='color:#2b9a48'>green</b> accepted · <b style='color:#e03b3b'>red</b> rejected · "
                      "grey unreviewed.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        layout.addWidget(hint)
        self.gallery = QListWidget()
        self.gallery.setObjectName("gallery")
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setWordWrap(True)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setIconSize(QSize(THUMB + 12, THUMB + 12))
        self.gallery.setGridSize(QSize(THUMB + 34, THUMB + 54))
        self.gallery.setSpacing(4)
        self.gallery.setUniformItemSizes(True)
        self.gallery.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.gallery.itemClicked.connect(self.gallery_choose)
        self.gallery.itemDoubleClicked.connect(self.gallery_open)
        self.gallery.itemSelectionChanged.connect(self._gallery_picks_changed)
        layout.addWidget(self.gallery, 1)
        actions = QHBoxLayout()
        self.gallery_pick_label = QLabel("Nothing picked")
        self.gallery_pick_label.setMinimumWidth(120)
        actions.addWidget(self.gallery_pick_label)
        self.gallery_actions = [
            self._button("− Remove from selection  (Del)", lambda: self.gallery_selection("remove"), "neutral",
                         "Drop picked particles from the selection without flagging them"),
            self._button("+ Add to selection  (A)", lambda: self.gallery_selection("add")),
            self._button("✓ Accept  (K)", lambda: self.flag_gallery("keep"), "accept"),
            self._button("✕ Reject  (R)", lambda: self.flag_gallery("reject"), "reject"),
            self._button("↺ Unflag  (U)", lambda: self.flag_gallery("undecided"), "neutral"),
        ]
        for button in self.gallery_actions:
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self._button("Pick all on page", self.gallery.selectAll))
        layout.addLayout(actions)
        pager = QHBoxLayout()
        pager.addWidget(self._button(f"← Previous page", lambda: self.gallery_page(-1)))
        self.gallery_info = QLabel("No images")
        self.gallery_info.setAlignment(Qt.AlignCenter)
        pager.addWidget(self.gallery_info, 1)
        pager.addWidget(self._button(f"Next page →", lambda: self.gallery_page(1)))
        layout.addLayout(pager)
        return panel

    def _build_particle_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 2)
        self.canvas = ParticleCanvas()
        layout.addWidget(self.canvas, 3)
        side = QWidget()
        side.setMaximumWidth(330)
        col = QVBoxLayout(side)
        col.setSpacing(5)
        self.particle_badge = QLabel("—")
        self.particle_badge.setAlignment(Qt.AlignCenter)
        self.particle_badge.setObjectName("badge")
        col.addWidget(self.particle_badge)
        self.particle_membership = QLabel("")
        self.particle_membership.setWordWrap(True)
        self.particle_membership.setAlignment(Qt.AlignCenter)
        col.addWidget(self.particle_membership)
        self.keep_button = self._button("✓  Accept  (K)", lambda: self.set_status("keep"), "accept")
        self.reject_button = self._button("✕  Reject  (R)", lambda: self.set_status("reject"), "reject")
        self.undecided_button = self._button("↺  Unflag → unreviewed  (U)", lambda: self.set_status("undecided"),
                                             "neutral")
        self.toggle_selection_button = self._button("+ Add to selection  (A)", self.toggle_current_in_selection)
        for button in (self.keep_button, self.reject_button):
            button.setMinimumHeight(32)
            col.addWidget(button)
        reason_row = QHBoxLayout()
        reason_row.addWidget(QLabel("Reject reason"))
        self.reject_reason = QComboBox()
        self.reject_reason.addItems(["", "overlap", "clipped", "segmentation", "artefact", "wrong type", "other"])
        reason_row.addWidget(self.reject_reason, 1)
        col.addLayout(reason_row)
        col.addWidget(self.undecided_button)
        col.addWidget(self.toggle_selection_button)
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label"))
        self.particle_label_combo = QComboBox()
        self.particle_label_combo.setToolTip("Tag this particle with a label (true size comes from the label)")
        self.particle_label_combo.activated.connect(self._particle_label_chosen)
        label_row.addWidget(self.particle_label_combo, 1)
        col.addLayout(label_row)
        self.auto_advance = QCheckBox("Go to next particle after flagging")
        self.auto_advance.setChecked(True)
        col.addWidget(self.auto_advance)
        self.boundary_checkbox = QCheckBox("Overlay stored boundary  (B)")
        self.boundary_checkbox.setChecked(True)
        self.boundary_checkbox.toggled.connect(lambda _: self.refresh_particle())
        col.addWidget(self.boundary_checkbox)
        self.background_checkbox = QCheckBox("Background-subtracted ROI")
        self.background_checkbox.stateChanged.connect(lambda _: self.refresh_particle())
        col.addWidget(self.background_checkbox)
        metrics_box = QGroupBox("Particle")
        metrics = QFormLayout(metrics_box)
        self.metric_labels: Dict[str, QLabel] = {}
        for key, label in (("time", "Time"), ("length", "Length [µm]"), ("width", "Width [µm]"),
                           ("focus", "Focus"), ("roundness", "Roundness"), ("xy", "Image x, y [px]"),
                           ("stage", "Stage X, Z"), ("actual", "Actual L × W [µm]"),
                           ("instrument", "Instrument"), ("source", "Source")):
            widget = QLabel("—")
            widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
            widget.setWordWrap(True)
            self.metric_labels[key] = widget
            metrics.addRow(label, widget)
        col.addWidget(metrics_box)
        col.addStretch(1)
        layout.addWidget(side, 1)
        return page

    def _build_surface_tab(self):
        self.surface_figure = Figure(figsize=(7, 6), constrained_layout=True)
        self.surface_axes = self.surface_figure.add_subplot(111)
        self.surface_canvas = FigureCanvasQTAgg(self.surface_figure)
        return self.surface_canvas

    def _build_browse_bar(self):
        bar = QFrame()
        bar.setObjectName("browseBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Browse"))
        self.view_selector = QComboBox()
        for value, label in BROWSE_SCOPES:
            self.view_selector.addItem(label, value)
        self.view_selector.setToolTip("Which particles the Gallery, Particle viewer and ◀ ▶ step through")
        self.view_selector.currentIndexChanged.connect(self.change_view_scope)
        layout.addWidget(self.view_selector)
        self.prev_button = self._button("◀", lambda: self.navigate(-1), None, "Previous particle (←)")
        self.next_button = self._button("▶", lambda: self.navigate(1), None, "Next particle (→)")
        self.progress_label = QLabel("0 / 0")
        self.progress_label.setMinimumWidth(80)
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.next_button)
        line = QFrame(); line.setFrameShape(QFrame.VLine); line.setStyleSheet("color:#c3d0dc")
        layout.addWidget(line)
        self.current_thumb = QLabel()
        self.current_thumb.setFixedSize(40, 40)
        self.current_thumb.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.current_thumb)
        self.current_label = QLabel("No particle")
        self.current_label.setTextFormat(Qt.RichText)
        layout.addWidget(self.current_label, 1)
        layout.addWidget(self._button("Open in viewer ⤢", self.open_current, None, "Show this particle large (Enter)"))
        return bar

    def _build_sidebar(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 0, 2, 2)
        layout.setSpacing(4)
        quick = QGridLayout()
        quick.setSpacing(3)
        for caption, action, row, col in (("+ Add MAT files", self.choose_files, 0, 0),
                                          ("+ Add folder", self.choose_add_directory, 0, 1),
                                          ("Stage CSV…", self.choose_position_file, 1, 0),
                                          ("Save project", self.save_project, 1, 1)):
            quick.addWidget(self._button(caption, action), row, col)
        layout.addLayout(quick)
        self.control_tabs = QTabWidget()

        def make_tab(title, scrolling=True):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            page_layout.setSpacing(5)
            if scrolling:
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(page)
                scroll.setFrameShape(QFrame.NoFrame)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.control_tabs.addTab(scroll, title)
            else:
                self.control_tabs.addTab(page, title)
            return page_layout

        review_page = make_tab("Review")
        labels_page = make_tab("Labels")
        sets_page = make_tab("Sets / X–Z")
        filter_page = make_tab("Filters")
        advanced_page = make_tab("Advanced")
        self.control_tabs.setTabVisible(4, False)
        layout.addWidget(self.control_tabs, 1)
        self._build_review_page(review_page)
        self._build_labels_page(labels_page)
        self._build_sets_page(sets_page)
        self._build_filter_page(filter_page)
        self._build_advanced_page(advanced_page)
        return panel

    def _build_review_page(self, page):
        progress = QGroupBox("Progress (all loaded particles)")
        grid = QGridLayout(progress)
        grid.setHorizontalSpacing(4)
        self.count_labels = {}
        rows = (("undecided", "● Unreviewed", STATUS_COLOUR["undecided"]),
                ("keep", "● Accepted", STATUS_COLOUR["keep"]),
                ("reject", "✕ Rejected", STATUS_COLOUR["reject"]),
                ("selected", "◉ Selected", SELECTED_COLOUR),
                ("committed", "◎ In calibration sets", COMMITTED_COLOUR))
        for row, (key, text, colour) in enumerate(rows):
            name = QLabel(text)
            name.setStyleSheet(f"color:{colour}; font-weight:700;")
            count = QLabel("0")
            count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            count.setStyleSheet("font-weight:700;")
            self.count_labels[key] = count
            grid.addWidget(name, row, 0)
            grid.addWidget(count, row, 1)
            if key in ("undecided", "keep", "reject", "committed"):
                small = self._button("Select", lambda _=False, k=key: self.select_group(k), None,
                                     "Make these particles the selection")
                small.setMaximumWidth(64)
                grid.addWidget(small, row, 2)
            browse_key = {"selected": "selection"}.get(key, key)
            if browse_key != "committed":
                small = self._button("Browse", lambda _=False, k=browse_key: self.browse_group(k), None,
                                     "Show these particles in the gallery")
                small.setMaximumWidth(68)
                grid.addWidget(small, row, 3)
        self.progress_bar = QLabel()
        self.progress_bar.setTextFormat(Qt.RichText)
        grid.addWidget(self.progress_bar, len(rows), 0, 1, 4)
        page.addWidget(progress)

        howto = QGroupBox("How it works")
        how = QVBoxLayout(howto)
        text = QLabel(
            "<ol style='margin-left:-18px'>"
            "<li><b>Drag a box</b> on Camera X–Y → particles turn <b style='color:#1c64d6'>blue</b>.</li>"
            "<li>On Focus / Length, <b>Ctrl/⌘-drag</b> round outliers to drop them, or use the "
            "<b style='color:#e03b3b'>✕ Reject</b> box.</li>"
            "<li><b>Accept selection</b> (K) → <b style='color:#2b9a48'>green</b>, saved, and skipped by "
            "new boxes.</li>"
            "<li>Check images in the Gallery; Del removes, R rejects.</li>"
            "<li>To undo a flag: <b>Unflag</b> (U), or Ctrl+Z.</li></ol>"
            "<span style='color:#516579'>Click a dot to inspect it · Shift-click adds · Ctrl/⌘-click toggles · "
            "double-click opens · wheel zooms.</span>")
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        how.addWidget(text)
        page.addWidget(howto)

        history = QGroupBox("Flag history")
        hist = QVBoxLayout(history)
        self.batch_list = QListWidget()
        self.batch_list.setMinimumHeight(90)
        self.batch_list.itemDoubleClicked.connect(lambda _: self.select_batch())
        hist.addWidget(self.batch_list)
        row = QHBoxLayout()
        row.addWidget(self._button("Select batch", self.select_batch, None,
                                   "Make the particles of this batch the selection"))
        row.addWidget(self._button("Undo batch…", self.remove_classified_batch, "neutral",
                                   "Return this batch's particles to unreviewed (unless changed since)"))
        hist.addLayout(row)
        page.addWidget(history)
        page.addStretch(1)

    def _build_labels_page(self, page):
        box = QGroupBox("Particle labels (shared by all datasets)")
        col = QVBoxLayout(box)
        intro = QLabel("Label what you accept — bead sizes, ice-crystal analogues — with their true size. "
                       "Pick the label in <b>Accept as</b> (top bar) before accepting, or assign it here.")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        intro.setObjectName("hint")
        col.addWidget(intro)
        self.label_list = QListWidget()
        self.label_list.setMinimumHeight(140)
        self.label_list.itemDoubleClicked.connect(lambda _: self.edit_label())
        self.label_list.currentItemChanged.connect(lambda *_: self._label_selected())
        col.addWidget(self.label_list)
        row = QHBoxLayout()
        row.addWidget(self._button("＋ New…", self.new_label, "primary"))
        row.addWidget(self._button("Edit / resize…", self.edit_label, None,
                                   "Change name, type or true size; applies to every dataset"))
        row.addWidget(self._button("Delete…", self.delete_label))
        col.addLayout(row)
        grid = QGridLayout()
        grid.addWidget(self._button("Select particles with label", lambda: self.select_label("replace"), None,
                                    "Make every loaded particle carrying this label the selection"), 0, 0, 1, 2)
        grid.addWidget(self._button("Add them to selection", lambda: self.select_label("add")), 1, 0)
        grid.addWidget(self._button("Browse them", self.browse_label), 1, 1)
        grid.addWidget(self._button("Tag selection with label", self.assign_label_to_selection, None,
                                    "Give the label (and its size) to the selection without changing "
                                    "accepted/rejected flags"), 2, 0, 1, 2)
        grid.addWidget(self._button("Remove labels from selection", self.clear_label_from_selection), 3, 0, 1, 2)
        col.addLayout(grid)
        self.label_detail = QLabel("")
        self.label_detail.setWordWrap(True)
        self.label_detail.setTextFormat(Qt.RichText)
        col.addWidget(self.label_detail)
        page.addWidget(box)
        lib_box = QGroupBox("Label library file")
        lib = QVBoxLayout(lib_box)
        self.label_path = QLabel()
        self.label_path.setWordWrap(True)
        self.label_path.setObjectName("hint")
        self.label_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lib.addWidget(self.label_path)
        lib.addWidget(self._button("Use another library file…", self.choose_label_library, None,
                                   "E.g. a labels.json on a shared drive used by the whole group"))
        page.addWidget(lib_box)
        page.addStretch(1)

    def _build_sets_page(self, page):
        box = QGroupBox("Calibration sets and best-focus X/Z")
        sets_layout = QVBoxLayout(box)
        sets_layout.setSpacing(4)
        note = QLabel("A calibration set is a named snapshot of particles plus its best-focus Xbest/Zbest. "
                      "Tip: Review → Accepted → Select, then save it here.")
        note.setWordWrap(True)
        note.setObjectName("hint")
        sets_layout.addWidget(note)
        self.set_name_field = QLineEdit()
        self.set_name_field.setPlaceholderText("Set name, e.g. beads · jig placement 1")
        sets_layout.addWidget(self.set_name_field)
        reference_row = QHBoxLayout()
        reference_row.addWidget(QLabel("Xbest"))
        self.best_x_set = QDoubleSpinBox()
        reference_row.addWidget(self.best_x_set, 1)
        reference_row.addWidget(QLabel("Zbest"))
        self.best_z_set = QDoubleSpinBox()
        reference_row.addWidget(self.best_z_set, 1)
        for spin in (self.best_x_set, self.best_z_set):
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(5)
            spin.setSingleStep(.05)
            spin.valueChanged.connect(lambda _value: self._set_reference_changed())
            spin.setToolTip("Original micrometer stage position at maximum focus")
        sets_layout.addLayout(reference_row)
        sets_layout.addWidget(self._button("Snap to median focus peak", self.snap_reference_to_median_peak, None,
                                           "Set Xbest (or Zbest) to the peak of the median focus trace "
                                           "currently plotted against Stage X (or Z)"))
        plane = QGroupBox("Object plane (for signed distance d)")
        plane_layout = QGridLayout(plane)
        plane_layout.addWidget(QLabel("Normal θ"), 0, 0)
        self.plane_normal = QDoubleSpinBox()
        self.plane_normal.setRange(-180, 180)
        self.plane_normal.setDecimals(2)
        self.plane_normal.setSingleStep(1)
        self.plane_normal.setSuffix(" °")
        self.plane_normal.setValue(45.0)
        self.plane_normal.setWrapping(True)
        self.plane_normal.setMaximumWidth(110)
        self.plane_normal.setToolTip("Direction of the object-plane normal in the stage X–Z frame, measured from +X "
                                     "towards +Z. d = (X−Xbest)·cos θ + (Z−Zbest)·sin θ.")
        self.plane_normal.valueChanged.connect(lambda _v: self._plane_changed())
        plane_layout.addWidget(self.plane_normal, 0, 1)
        plane_layout.addWidget(self._button("Flip side", lambda: self.plane_normal.setValue(
            ((self.plane_normal.value() + 360) % 360) - 180), None, "Swap which side counts as positive (θ + 180°)"),
            0, 2)
        plane_layout.addWidget(self._button("Estimate θ from scan direction", self.estimate_plane_normal, None,
                                            "Principal direction of the stage positions (of the selection, if any). "
                                            "Valid if the jig moves particles straight through the object plane."),
                               1, 0, 1, 3)
        self.plane_hint = QLabel()
        self.plane_hint.setWordWrap(True)
        self.plane_hint.setObjectName("hint")
        plane_layout.addWidget(self.plane_hint, 2, 0, 1, 3)
        sets_layout.addWidget(plane)
        self.set_radius_label = QLabel("Best-focus radius = 0 (stage units)")
        self.set_radius_label.setWordWrap(True)
        sets_layout.addWidget(self.set_radius_label)
        commit = self._button("Save selection as set + X/Z", self.commit_working_set, "primary",
                              "Save the current particle-ID snapshot AND these two best-focus values together")
        sets_layout.addWidget(commit)
        individual = QHBoxLayout()
        individual.addWidget(self._button("Add current", self.commit_current_particle, None,
                                          "Add the current particle (☆) to the set named above"))
        individual.addWidget(self._button("Remove current", self.remove_current_from_committed_set, None,
                                          "Remove the current particle (☆) from the chosen set"))
        sets_layout.addLayout(individual)
        self.saved_sets_list = QListWidget()
        self.saved_sets_list.setMinimumHeight(70)
        self.saved_sets_list.setMaximumHeight(140)
        self.saved_sets_list.itemClicked.connect(lambda _: self.saved_set_clicked())
        self.saved_sets_list.itemDoubleClicked.connect(lambda _: self.load_named_set())
        sets_layout.addWidget(self.saved_sets_list)
        self.active_set_label = QLabel("No active calibrated set")
        self.active_set_label.setWordWrap(True)
        self.active_set_label.setObjectName("hint")
        sets_layout.addWidget(self.active_set_label)
        ref_actions = QHBoxLayout()
        ref_actions.addWidget(self._button("Save X/Z changes", self.update_saved_set_reference, None,
                                           "Correct the chosen set's X/Z without changing its particles"))
        ref_actions.addWidget(self._button("Clear X/Z", self.clear_saved_set_reference))
        sets_layout.addLayout(ref_actions)
        set_actions = QGridLayout()
        for label, action, row, col in (("Select set", self.load_named_set, 0, 0),
                                        ("Add to selection", self.add_named_set, 0, 1),
                                        ("Delete set…", self.delete_named_set, 1, 0),
                                        ("Unflag members…", self.clear_named_set_flags, 1, 1)):
            set_actions.addWidget(self._button(label, action), row, col)
        sets_layout.addLayout(set_actions)
        page.addWidget(box)
        page.addStretch(1)

    def _build_filter_page(self, page):
        filter_box = QGroupBox("Filters (non-destructive)")
        filter_layout = QGridLayout(filter_box)
        filter_layout.addWidget(QLabel("Metric"), 0, 0)
        filter_layout.addWidget(QLabel("Min"), 0, 1)
        filter_layout.addWidget(QLabel("Max"), 0, 2)
        self.filter_fields: Dict[str, tuple] = {}
        row = 1
        for key, label in (("length", "Length [µm]"), ("width", "Width [µm]"), ("focus", "Focus"),
                           ("x", "Image x [px]"), ("y", "Image y [px]"), ("ratio", "Meas./true length")):
            low, high = QLineEdit(), QLineEdit()
            low.setPlaceholderText("any")
            high.setPlaceholderText("any")
            low.returnPressed.connect(self.apply_filters)
            high.returnPressed.connect(self.apply_filters)
            self.filter_fields[key] = (low, high)
            filter_layout.addWidget(QLabel(label), row, 0)
            filter_layout.addWidget(low, row, 1)
            filter_layout.addWidget(high, row, 2)
            row += 1
        self.time_low = QDateTimeEdit(); self.time_high = QDateTimeEdit()
        for widget in (self.time_low, self.time_high):
            widget.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            widget.setCalendarPopup(True)
            widget.setDateTime(QDateTime.currentDateTime())
        self.time_filter_checkbox = QCheckBox("Time window")
        filter_layout.addWidget(self.time_filter_checkbox, row, 0, 1, 3)
        row += 1
        filter_layout.addWidget(self.time_low, row, 0, 1, 3)
        row += 1
        filter_layout.addWidget(self.time_high, row, 0, 1, 3)
        row += 1
        self.image_mode = QComboBox()
        self.image_mode.addItem("Cropped ROIs only", "crop")
        self.image_mode.addItem("All images incl. full frames", "all")
        self.image_mode.addItem("Full frames only", "full")
        self.image_mode.currentIndexChanged.connect(self.apply_filters)
        filter_layout.addWidget(QLabel("Images"), row, 0)
        filter_layout.addWidget(self.image_mode, row, 1, 1, 2)
        row += 1
        filter_layout.addWidget(self._button("Show data ranges", self.fill_filter_ranges, None,
                                             "Show min/max of the loaded data as hints"), row, 0)
        filter_layout.addWidget(self._button("Apply", self.apply_filters, "primary"), row, 1)
        filter_layout.addWidget(self._button("Clear", self.clear_filters), row, 2)
        page.addWidget(filter_box)
        page.addStretch(1)

    def _build_advanced_page(self, page):
        z_box = QGroupBox("Object-plane position (current particle)")
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
        page.addWidget(z_box)
        jig_box = QGroupBox("Jig placement / best-focus reference")
        jig = QFormLayout(jig_box)
        self.segment_start = QDateTimeEdit(); self.segment_end = QDateTimeEdit()
        for widget in (self.segment_start, self.segment_end):
            widget.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            widget.setCalendarPopup(True)
            widget.setDateTime(QDateTime.currentDateTime())
        self.best_x = QDoubleSpinBox(); self.best_z = QDoubleSpinBox()
        for widget in (self.best_x, self.best_z):
            widget.setRange(-1e6, 1e6); widget.setDecimals(5)
        jig.addRow("Begins", self.segment_start)
        jig.addRow("Ends", self.segment_end)
        jig.addRow("X at max focus", self.best_x)
        jig.addRow("Z at max focus", self.best_z)
        jig.addRow(self._button("Add reference for this interval", self.set_best_focus_segment))
        self.peak_combo = QComboBox()
        self.peak_combo.addItem("Maximum focus = sharpest", "maximum")
        self.peak_combo.addItem("Minimum focus = sharpest", "minimum")
        self.peak_combo.currentIndexChanged.connect(lambda _: self._redraw_plots())
        jig.addRow("Focus optimum", self.peak_combo)
        jig.addRow(self._button("Fit tilted best-focus surface", self.fit_best_focus_surface, None,
                                "Uses non-rejected particles inside the interval (only the selection, if any)"))
        note = QLabel("Exploratory: focus = baseline + curvature × (radius − best(u,v))²")
        note.setWordWrap(True); note.setObjectName("hint")
        jig.addRow(note)
        self.reference_list = QListWidget()
        self.reference_list.setMaximumHeight(120)
        self.reference_list.currentRowChanged.connect(self.select_reference)
        jig.addRow(self.reference_list)
        refs = QHBoxLayout()
        refs.addWidget(self._button("Update", self.update_reference))
        refs.addWidget(self._button("Delete", self.delete_reference))
        jig.addRow(refs)
        page.addWidget(jig_box)
        instrument_row = QHBoxLayout()
        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems(["CPI1.0", "CPI3v"])
        instrument_row.addWidget(self.instrument_combo)
        instrument_row.addWidget(self._button("Set instrument for all", self.set_instrument))
        page.addLayout(instrument_row)
        page.addStretch(1)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        for text, slot in (("New project…", self.create_project), ("Open project…", self.open_project_dialog),
                           ("Save project", self.save_project), (None, None),
                           ("Add MAT files…", self.choose_files), ("Open processed directory…", self.choose_directory),
                           ("Add another dataset…", self.choose_add_directory),
                           ("Import calibration spreadsheet…", self.choose_truth_table),
                           ("Import stage positions CSV…", self.choose_position_file),
                           ("Save flags as…", self.save_as)):
            if text is None:
                file_menu.addSeparator()
                continue
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        edit_menu = self.menuBar().addMenu("Edit")
        for text, slot, keys in (("Undo", self.undo, "Ctrl+Z"),
                                 ("Select all visible (pickable)", self.select_all_visible, "Ctrl+A"),
                                 ("Clear selection", self.clear_selection, None),
                                 ("Accept selection", self.accept_selection, None),
                                 ("Reject selection", self.reject_selection, None),
                                 ("Unflag selection", self.unflag_selection, None)):
            action = QAction(text, self)
            action.triggered.connect(slot)
            if keys:
                action.setShortcut(QKeySequence(keys))
            edit_menu.addAction(action)
        view_menu = self.menuBar().addMenu("View")
        self.hover_action = QAction("Hover preview on plots", self)
        self.hover_action.setCheckable(True)
        self.hover_action.setChecked(True)
        view_menu.addAction(self.hover_action)
        show_advanced = QAction("Show experimental calibration tools", self)
        show_advanced.setCheckable(True)
        show_advanced.toggled.connect(self._show_advanced)
        view_menu.addAction(show_advanced)

    def _show_advanced(self, visible):
        self.control_tabs.setTabVisible(4, visible)
        self.viewer_tabs.setTabVisible(3, visible)

    def _build_shortcuts(self) -> None:
        def key(sequence, slot):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(slot)
            return shortcut
        self._arrow_shortcuts = [key(Qt.Key_Right, lambda: self.navigate(1)),
                                 key(Qt.Key_Left, lambda: self.navigate(-1))]
        key("K", lambda: self._key_flag("keep"))
        key("R", lambda: self._key_flag("reject"))
        key("U", lambda: self._key_flag("undecided"))
        key("A", lambda: self._key_selection("add"))
        key(Qt.Key_Delete, lambda: self._key_selection("remove"))
        key(Qt.Key_Backspace, lambda: self._key_selection("remove"))
        key(Qt.Key_Escape, self._key_escape)
        key(Qt.Key_Return, self._key_open)
        key(Qt.Key_Enter, self._key_open)
        key("B", lambda: self.boundary_checkbox.setChecked(not self.boundary_checkbox.isChecked()))
        key("+", lambda: self.z_spin.setValue(self.z_spin.value() + self.z_step_um))
        key("-", lambda: self.z_spin.setValue(self.z_spin.value() - self.z_step_um))

    # --------------------------------------------------- tabs & key dispatch --
    def _current_tab(self):
        widget = self.viewer_tabs.currentWidget()
        return {id(self.select_tab): "select", id(self.gallery_panel): "gallery",
                id(self.particle_tab): "particle"}.get(id(widget), "other")

    def _tab_changed(self, _index):
        tab = self._current_tab()
        for shortcut in self._arrow_shortcuts:
            shortcut.setEnabled(tab != "gallery")  # arrows move between thumbnails there
        self._hide_hover()
        if tab == "gallery" and self._gallery_dirty:
            self.update_gallery()
        if tab == "particle":
            self.refresh_particle()
        if tab == "select":
            self._redraw_plots()
        if tab == "other":
            self.update_surface_map()

    def _key_flag(self, status):
        tab = self._current_tab()
        if tab == "particle":
            self.set_status(status)
        elif tab == "gallery":
            self.flag_gallery(status)
        else:
            {"keep": self.accept_selection, "reject": self.reject_selection,
             "undecided": self.unflag_selection}[status]()

    def _key_selection(self, action):
        if self._current_tab() == "gallery":
            self.gallery_selection(action)
            return
        record = self.current_record()
        if record is None:
            return
        if action == "add":
            self.current_to_working()
        else:
            self.remove_current_from_working()

    def _key_escape(self):
        self._hide_hover()
        if self._current_tab() == "gallery" and self.gallery.selectedItems():
            self.gallery.clearSelection()
            return
        self.clear_selection()

    def _key_open(self):
        if self._current_tab() == "gallery":
            items = self.gallery.selectedItems()
            if items:
                self.gallery_open(items[0])
                return
        self.open_current()

    # --------------------------------------------------------- populations --
    def _records(self):
        return self.catalog.records if self.catalog else []

    def _eligible_base_indices(self):
        return list(self._base_indices)

    def _shown_statuses(self):
        return {s for s, box in self.show_status.items() if box.isChecked()}

    def _visible_indices(self, kind):
        """Indices drawn (and box-selectable) on the camera or metric plots."""
        records = self._records()
        shown = self._shown_statuses()
        selected = self.selected_ids
        only_selected = kind == "metric" and self.selection_only_cb.isChecked() and bool(selected)
        out = []
        for i in self._base_indices:
            r = records[i]
            if r.particle_id in selected:
                out.append(i)
            elif not only_selected and r.status in shown:
                out.append(i)
        return out

    def _plot_position(self, record):
        if not (math.isfinite(record.stage_x) and math.isfinite(record.stage_y)):
            return float("nan")
        reference, conflict = self._reference_index.get(record.particle_id, (None, False))
        draft = self._draft_reference_for(record)
        if conflict and draft is None:
            return float("nan")
        if reference is not None or draft is not None:
            return calibrated_radial_position(record.stage_x, record.stage_y,
                                              draft if draft is not None else reference)
        radius = math.hypot(record.stage_x, record.stage_y)
        segment = next((seg for seg in self.stage_segments
                        if seg["start"] <= record.time_matlab < seg["end"]), None)
        if segment is None:
            return float("nan")  # No best-focus reference: Zp is undefined.
        if "surface" in segment:
            best = predict_best_radius(segment["surface"], record.image_x_px, record.image_y_px)
            return float(best - radius)
        return segment["best_radius"] - radius

    @staticmethod
    def _metric_value(record, metric):
        if metric == "focus":
            return record.focus
        if metric == "length":
            return record.length_um
        if metric == "width":
            return record.width_um
        if metric == "roundness":
            return record.roundness
        if metric == "ratio":
            return (record.length_um / record.actual_length_um
                    if math.isfinite(record.actual_length_um) and record.actual_length_um > 0
                    else float("nan"))
        return float("nan")

    def _plot_y(self, record, metric=None):
        return self._metric_value(record, metric or self.plot_metric.currentData())

    def _plot_x(self, record):
        mode = self.position_metric.currentData()
        if mode == "stage_x":
            return record.stage_x
        if mode == "stage_z":
            return record.stage_y
        if mode in RELATIVE_AXES:
            reference = self._draft_reference_for(record)
            if reference is None or not (math.isfinite(record.stage_x) and math.isfinite(record.stage_y)):
                return float("nan")
            if mode == "plane_signed":
                return signed_plane_distance(record.stage_x, record.stage_y, reference, self.plane_normal.value())
            if mode == "plane_offset":
                return in_plane_offset(record.stage_x, record.stage_y, reference, self.plane_normal.value())
            if mode == "stage_offset_radius":
                return corrected_stage_radius(record.stage_x, record.stage_y, reference)
            return relative_stage_position(record.stage_x, record.stage_y, reference,
                                           "x" if mode == "stage_x_relative" else "z")
        if mode == "time":
            return record.time_matlab
        if mode == "stage_radius":
            return (math.hypot(record.stage_x, record.stage_y)
                    if math.isfinite(record.stage_x) and math.isfinite(record.stage_y) else float("nan"))
        return self._plot_position(record)

    def _draft_reference_for(self, record):
        """Use the editor as a preview for the selection / active set only."""
        if record.particle_id in self._active_members or record.particle_id in self.selected_ids:
            if self._draft_reference_cache is None:
                self._draft_reference_cache = make_reference(self.best_x_set.value(), self.best_z_set.value())
            return self._draft_reference_cache
        reference, conflict = self._reference_index.get(record.particle_id, (None, False))
        return None if conflict else reference

    def _values(self, kind, records, metric=None):
        if kind == "camera":
            xs = np.fromiter((r.image_x_px for r in records), float, len(records))
            ys = np.fromiter((r.image_y_px for r in records), float, len(records))
        else:
            xs = np.fromiter((self._plot_x(r) for r in records), float, len(records))
            ys = np.fromiter((self._metric_value(r, metric) for r in records), float, len(records))
        return xs, ys

    # -------------------------------------------------------------- drawing --
    def _redraw_plots(self):
        self.update_spatial()
        self.update_plots()

    def _draw_population(self, ax, records, xs, ys, kind):
        """Layered scatter: unreviewed < rejected < accepted < selected < current."""
        if not records:
            return None
        status = np.array([r.status for r in records])
        selected = np.fromiter((r.particle_id in self.selected_ids for r in records), bool, len(records))
        committed_ids = self._committed_cache
        committed = np.fromiter((r.particle_id in committed_ids for r in records), bool, len(records))
        good = np.isfinite(xs) & np.isfinite(ys)
        colour_by = self.colour_by.currentData()
        mappable = None
        und = good & (status == "undecided") & ~selected
        acc = good & (status == "keep") & ~selected
        rej = good & (status == "reject") & ~selected
        if colour_by in ("image_x", "image_y"):
            values = np.fromiter(((r.image_x_px if colour_by == "image_x" else r.image_y_px) for r in records),
                                 float, len(records))
            base = (und | acc) & np.isfinite(values)
            if base.any():
                mappable = ax.scatter(xs[base], ys[base], c=values[base], cmap="viridis", s=11, alpha=.8,
                                      linewidths=0, rasterized=True, zorder=2)
            if acc.any():
                ax.scatter(xs[acc], ys[acc], facecolors="none", edgecolors=STATUS_COLOUR["keep"], s=30,
                           linewidths=1.0, rasterized=True, zorder=3)
        elif colour_by == "label":
            if und.any():
                ax.scatter(xs[und], ys[und], c=STATUS_COLOUR["undecided"], s=8, alpha=.4, linewidths=0,
                           rasterized=True, zorder=2)
            names = np.array([r.label for r in records])
            for name in sorted(set(names[acc])):
                mask = acc & (names == name)
                label = self.labels.get(name) if name else None
                colour = label.colour if label and label.colour else ("#2b9a48" if not name else "#495057")
                ax.scatter(xs[mask], ys[mask], c=colour, s=18, alpha=.9, edgecolors="#222", linewidths=.3,
                           rasterized=True, zorder=3, label=name or "accepted, no label")
            if acc.any():
                ax.legend(loc="lower right", fontsize=7, framealpha=.85, title="Accepted labels", title_fontsize=7)
        else:
            if und.any():
                ax.scatter(xs[und], ys[und], c=STATUS_COLOUR["undecided"], s=9, alpha=.55, linewidths=0,
                           rasterized=True, zorder=2)
            if acc.any():
                ax.scatter(xs[acc], ys[acc], c=STATUS_COLOUR["keep"], s=16, alpha=.85,
                           edgecolors="#1b6b31", linewidths=.4, rasterized=True, zorder=3)
        if rej.any():
            ax.scatter(xs[rej], ys[rej], c=STATUS_COLOUR["reject"], marker="x", s=14, alpha=.6,
                       linewidths=1.0, rasterized=True, zorder=2.5)
        com = good & committed
        if com.any():
            ax.scatter(xs[com], ys[com], facecolors="none", edgecolors=COMMITTED_COLOUR, s=58,
                       linewidths=1.1, alpha=.9, rasterized=True, zorder=4)
        sel = good & selected
        if sel.any():
            edge = np.where(status[sel] == "keep", STATUS_COLOUR["keep"],
                            np.where(status[sel] == "reject", STATUS_COLOUR["reject"], "#0b2f6b"))
            ax.scatter(xs[sel], ys[sel], c=SELECTED_COLOUR, s=24, edgecolors=edge, linewidths=1.0,
                       alpha=.95, rasterized=True, zorder=6)
        return mappable

    def _subsample(self, indices, cap):
        if len(indices) <= cap:
            return indices
        step = math.ceil(len(indices) / cap)
        records = self._records()
        keep = set(indices[::step])
        keep.update(i for i in indices if records[i].particle_id in self.selected_ids
                    or records[i].particle_id == self.current_id)
        return [i for i in indices if i in keep]

    def _placeholder(self, ax, message):
        ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#7b2a33",
                fontsize=10, wrap=True, bbox=dict(facecolor="#fff4f2", edgecolor="#e5bfc1", pad=10))

    def update_spatial(self) -> None:
        ax = self.spatial_axes
        locked = self._view_locked["camera"] and ax.has_data()
        limits = (ax.get_xlim(), ax.get_ylim()) if locked else None
        if self._colorbars["camera"] is not None:
            try:
                self._colorbars["camera"].remove()
            except Exception:
                pass
            self._colorbars["camera"] = None
        ax.clear()
        self._committed_cache = committed_particle_ids(self.saved_sets)
        records = self._records()
        indices = self._subsample(self._visible_indices("camera"), 30000)
        drawn = [records[i] for i in indices]
        xs, ys = self._values("camera", drawn)
        self._drawn[("camera", 0)] = (drawn, xs, ys)
        if drawn:
            mappable = self._draw_population(ax, drawn, xs, ys, "camera")
            if mappable is not None:
                self._colorbars["camera"] = self.spatial_figure.colorbar(
                    mappable, ax=ax, fraction=.045, pad=.02,
                    label="Camera X [px]" if self.colour_by.currentData() == "image_x" else "Camera Y [px]")
        else:
            self._placeholder(ax, "Drop MAT files or a folder on the window,\nor use + Add MAT files"
                              if not records else "No particles to show — check Filters / Show")
        ax.set_xlabel("Camera X [px]")
        ax.set_ylabel("Camera Y [px]")
        ax.set_aspect("equal", adjustable="box")
        if limits is not None:
            ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        else:
            lim = data_limits(xs, ys) if len(drawn) else None
            if lim is not None:
                ax.set_xlim(lim[0]); ax.set_ylim(lim[1][1], lim[1][0])
            else:
                ax.invert_yaxis()
        self._draw_current_marker(redraw=False)
        self.spatial_canvas.draw_idle()

    def _change_plot_axes(self):
        self._view_locked["metric"] = False
        if hasattr(self, "mouse_group"):
            self.set_plot_tool(self.plot_tool, quiet=True)
        self.update_plots()

    def _rebuild_metric_axes(self, rows):
        if self._colorbars["metric"] is not None:
            try:
                self._colorbars["metric"].remove()
            except Exception:
                pass
            self._colorbars["metric"] = None
        self.plot_figure.clear()
        axes = []
        for k in range(len(rows)):
            axes.append(self.plot_figure.add_subplot(len(rows), 1, k + 1, sharex=axes[0] if axes else None))
        self.metric_axes = axes
        self.plot_axes = axes[0]
        self.plot_canvas.set_axes(axes)
        self._metric_rows = list(rows)

    def update_plots(self):
        if not hasattr(self, "plot_canvas"):
            return
        xmode = self.position_metric.currentData()
        rows = [self.plot_metric.currentData()]
        if self.plot_metric2.currentData() not in ("none", rows[0]):
            rows.append(self.plot_metric2.currentData())
        key = (xmode, tuple(rows))
        locked = self._view_locked["metric"] and self._last_metric_key == key and self.plot_axes.has_data()
        saved = ([ax.get_xlim() for ax in self.metric_axes], [ax.get_ylim() for ax in self.metric_axes]) \
            if locked else None
        if rows != self._metric_rows:
            self._rebuild_metric_axes(rows)
        else:
            if self._colorbars["metric"] is not None:
                try:
                    self._colorbars["metric"].remove()
                except Exception:
                    pass
                self._colorbars["metric"] = None
            for ax in self.metric_axes:
                ax.clear()
        self._last_metric_key = key
        self._committed_cache = committed_particle_ids(self.saved_sets)
        records = self._records()
        indices = self._subsample(self._visible_indices("metric"), 30000)
        drawn = [records[i] for i in indices]
        xs = np.fromiter((self._plot_x(r) for r in drawn), float, len(drawn))
        status = np.array([r.status for r in drawn]) if drawn else np.array([], dtype=str)
        selected = np.fromiter((r.particle_id in self.selected_ids for r in drawn), bool, len(drawn))
        self._median_peak.pop(xmode, None)
        mappable = None
        for index, (ax, metric) in enumerate(zip(self.metric_axes, rows)):
            ys = np.fromiter((self._metric_value(r, metric) for r in drawn), float, len(drawn))
            self._drawn[("metric", index)] = (drawn, xs, ys)
            finite = np.isfinite(xs) & np.isfinite(ys)
            if finite.any():
                m = self._draw_population(ax, drawn, xs, ys, "metric")
                mappable = mappable or m
                if self.median_cb.isChecked():
                    self._draw_median(ax, xs, ys, status, selected, metric, xmode)
                if not saved:
                    lim = data_limits(xs, ys)
                    if lim is not None:
                        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
            elif index == 0:
                self._placeholder(ax, self._empty_plot_message(drawn, xmode, metric))
            ax.set_ylabel(dict(METRICS)[metric])
            ax.grid(True, color="#e3e9ef", lw=.6, zorder=0)
            if index < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            self._draw_reference_guides(ax, xmode, label=index == 0)
        bottom = self.metric_axes[-1]
        bottom.set_xlabel(dict(POSITIONS)[xmode])
        if xmode == "time":
            def format_time_tick(value, _position):
                try:
                    return matlab_datenum_to_datetime(value).strftime("%d/%m %H:%M")
                except (ValueError, OverflowError, OSError):
                    return ""
            for ax in self.metric_axes:
                ax.xaxis.set_major_formatter(FuncFormatter(format_time_tick))
            bottom.tick_params(axis="x", labelrotation=18)
        if mappable is not None:
            self._colorbars["metric"] = self.plot_figure.colorbar(
                mappable, ax=self.metric_axes, fraction=.04, pad=.02,
                label="Camera X [px]" if self.colour_by.currentData() == "image_x" else "Camera Y [px]")
        if saved:
            for ax, xl, yl in zip(self.metric_axes, saved[0], saved[1]):
                ax.set_xlim(xl); ax.set_ylim(yl)
        self._draw_current_marker(redraw=False)
        self.plot_canvas.draw_idle()
        if self.viewer_tabs.currentWidget() is self.surface_tab:
            self.update_surface_map()

    def _draw_median(self, ax, xs, ys, status, selected, metric, xmode):
        base = np.isfinite(xs) & np.isfinite(ys) & (status != "reject")
        trace = median_trace(xs[base], ys[base])
        handles = []
        if len(trace["x"]) > 1:
            ax.fill_between(trace["x"], trace["q25"], trace["q75"], color=MEDIAN_COLOUR, alpha=.11,
                            linewidth=0, zorder=4.5)
            line, = ax.plot(trace["x"], trace["median"], color=MEDIAN_COLOUR, lw=2.2, marker="o", ms=3,
                            zorder=7, label="Median ± IQR (non-rejected)")
            handles.append(line)
            if metric == "focus" and xmode != "time":
                peak = trace_extremum(trace, self.peak_combo.currentData())
                if peak is not None:
                    self._median_peak[xmode] = peak[0]
                    mark, = ax.plot([peak[0]], [peak[1]], marker="^", ms=10, color=CURRENT_COLOUR,
                                    markeredgecolor="#7a4300", ls="none", zorder=9,
                                    label=f"Median peak at {peak[0]:.5g}")
                    handles.append(mark)
        sel = np.isfinite(xs) & np.isfinite(ys) & selected
        if sel.sum() >= 3:
            strace = median_trace(xs[sel], ys[sel])
            if len(strace["x"]) > 1:
                line, = ax.plot(strace["x"], strace["median"], color=SELECTED_COLOUR, lw=1.8, ls="--",
                                zorder=8, label="Median of selection")
                handles.append(line)
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8, framealpha=.85)

    def _draw_reference_guides(self, ax, xmode, label):
        if xmode in ("stage_x", "stage_z"):
            value = self.best_x_set.value() if xmode == "stage_x" else self.best_z_set.value()
            ax.axvline(value, color="#d85519", lw=1.6, linestyle="--", zorder=8)
            if label:
                ax.text(.01, .98, f"Draft {'X' if xmode == 'stage_x' else 'Z'}best = {value:.5g}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#a13d0f")
        elif xmode in RELATIVE_AXES:
            ax.axvline(0, color="#d85519", lw=1.4, linestyle="--", zorder=8)
            if xmode == "plane_signed" and label:
                ax.text(.01, .98, f"◀ − side   object plane (θ = {self.plane_normal.value():.3g}°)   + side ▶",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#a13d0f")

    def _empty_plot_message(self, drawn, xmode, metric):
        if not self._records():
            return "Add MAT files to plot focus and size."
        if not drawn:
            return "No particles to show — check Filters, the Show toggles,\nor 'Show only the selection'."
        has_stage = sum(math.isfinite(r.stage_x) and math.isfinite(r.stage_y) for r in drawn)
        if xmode in STAGE_AXES and has_stage == 0:
            return "No stage X/Z positions yet.\nImport them with  Stage CSV…  (sidebar)."
        if xmode in ("relative",) + RELATIVE_AXES:
            return "No best-focus X/Z reference for these particles.\nSave a set with Xbest/Zbest first."
        return f"No finite {dict(METRICS)[metric].lower()} values against this X axis."

    def _draw_current_marker(self, redraw=True):
        for artist in self._current_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._current_artists = []
        record = self.current_record()
        if record is None:
            return
        targets = [(self.spatial_axes, record.image_x_px, record.image_y_px)]
        x = self._plot_x(record)
        for ax, metric in zip(self.metric_axes, self._metric_rows):
            targets.append((ax, x, self._metric_value(record, metric)))
        for ax, xv, yv in targets:
            if math.isfinite(xv) and math.isfinite(yv):
                artist, = ax.plot([xv], [yv], marker="*", ms=17, markerfacecolor="none",
                                  markeredgecolor=CURRENT_COLOUR, markeredgewidth=2, ls="none", zorder=10)
                self._current_artists.append(artist)
        if redraw:
            self.spatial_canvas.draw_idle()
            self.plot_canvas.draw_idle()

    def update_surface_map(self):
        self.surface_figure.clear()
        ax = self.surface_figure.add_subplot(111)
        self.surface_axes = ax
        fits = [seg for seg in self.stage_segments if "surface" in seg]
        if fits and self.catalog and self.catalog.records:
            surface = fits[-1]["surface"]
            us = np.array([r.image_x_px for r in self.catalog.records if math.isfinite(r.image_x_px)])
            vs = np.array([r.image_y_px for r in self.catalog.records if math.isfinite(r.image_y_px)])
            if len(us) and len(vs):
                u0, u1 = np.percentile(us, [1, 99]); v0, v1 = np.percentile(vs, [1, 99])
                uu, vv = np.meshgrid(np.linspace(u0, u1, 70), np.linspace(v0, v1, 70))
                mesh = ax.pcolormesh(uu, vv, predict_best_radius(surface, uu, vv), shading="auto", cmap="viridis")
                self.surface_figure.colorbar(mesh, ax=ax, label="Best-focus radius (original stage units)")
                ax.set_xlabel("Camera X [px]"); ax.set_ylabel("Camera Y [px]")
                ax.invert_yaxis()
                ax.set_title(f"Most recent fitted surface · {surface['n_particles']} particles")
        else:
            ax.text(.5, .5, "Fit a tilted best-focus surface (Advanced tab) to see the map",
                    ha="center", va="center", transform=ax.transAxes)
        self.surface_canvas.draw_idle()

    # ------------------------------------------------------ mouse handling --
    def set_plot_tool(self, mode, quiet=False):
        """Keep box selection, navigation and X/Z dragging mutually exclusive."""
        if mode not in ("select", "zoom", "pan", "reference"):
            return
        for toolbar in (self.spatial_toolbar, self.plot_toolbar):
            current = getattr(toolbar.mode, "name", str(toolbar.mode)).lower()
            if "zoom" in current:
                toolbar.zoom()
            elif "pan" in current:
                toolbar.pan()
            if mode == "zoom":
                toolbar.zoom()
            elif mode == "pan":
                toolbar.pan()
        self.plot_tool = mode
        for button in self.mouse_buttons:
            button.setChecked(button.property("value") == mode)
        self.spatial_canvas.set_select_mode(mode == "select")
        self.plot_canvas.set_select_mode(mode == "select")
        stage_axis = self.position_metric.currentData() in ("stage_x", "stage_z")
        self.plot_canvas.set_reference_mode(mode == "reference" and stage_axis)
        if mode == "reference" and not stage_axis and not quiet:
            self.statusBar().showMessage("Choose X = Stage X or Stage Z to drag the best-focus line", 6000)
        self._update_hint()

    def set_box_mode(self, mode):
        self.box_mode = mode
        for button in self.box_buttons:
            button.setChecked(button.property("value") == mode)
        if self.plot_tool != "select":
            self.set_plot_tool("select")
        self._update_hint()

    def _effective_box_mode(self, modifiers):
        if modifiers & Qt.ShiftModifier:
            return "add"
        if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
            return "remove"
        if modifiers & Qt.AltModifier:
            return "keep"
        return self.box_mode

    def _update_hint(self):
        if not hasattr(self, "selection_preview"):
            return
        pick = self.pick_scope.currentText().lower()
        if self.plot_tool == "select":
            text = {"new": f"<b>Drag</b> a box to select ({pick}).",
                    "add": f"<b>Drag</b> to add particles ({pick}).",
                    "remove": "<b>Drag</b> round particles to drop them from the selection.",
                    "keep": "<b>Drag</b> a box: only selected particles inside it stay selected.",
                    "reject": "<b>Drag</b> round outliers to flag them <b style='color:#e03b3b'>rejected</b> "
                              "(only selected ones, if there is a selection)."}[self.box_mode]
            text += ("  &nbsp;<span style='color:#516579'>Shift = add · Ctrl/⌘ = remove · Alt/⌥ = keep only · "
                     "click a dot = inspect · double-click = open · wheel = zoom · "
                     "K/R/U = accept/reject/unflag selection · Esc = clear</span>")
        else:
            text = {"zoom": "<b>Zoom</b>: drag a rectangle to enlarge (selection unchanged). ⤢ Fit view resets.",
                    "pan": "<b>Pan</b>: drag to move the view.",
                    "reference": "<b>Move Xbest/Zbest</b>: drag on the focus plot (X = Stage X or Z); "
                                 "then Sets / X–Z → Save."}[self.plot_tool]
        self.selection_preview.setText(text)

    def _lock_plot_navigation(self, kind):
        if self.plot_tool in ("zoom", "pan"):
            self._view_locked[kind] = True

    def reset_plot_zoom(self):
        self._view_locked = {"camera": False, "metric": False}
        self._redraw_plots()

    def _on_wheel(self, canvas, index, point, delta):
        if delta == 0:
            return
        ax = canvas.axes_list[index]
        x, y = canvas.data_at(index, point)
        factor = 0.8 if delta > 0 else 1.25
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(x - (x - x0) * factor, x + (x1 - x) * factor)
        ax.set_ylim(y - (y - y0) * factor, y + (y1 - y) * factor)
        self._view_locked[canvas.kind] = True
        canvas.draw_idle()

    def _nearest_record(self, canvas, index, pixel, radius=12):
        if index is None:
            return None
        data = self._drawn.get((canvas.kind, index))
        if not data or not data[0]:
            return None
        records, xs, ys = data
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not finite.any():
            return None
        ax = canvas.axes_list[index]
        screen = ax.transData.transform(np.column_stack((xs[finite], ys[finite])))
        mx, my = canvas.mouseEventCoords(pixel)
        d2 = (screen[:, 0] - mx) ** 2 + (screen[:, 1] - my) ** 2
        k = int(np.argmin(d2))
        if d2[k] > (radius * canvas.device_pixel_ratio) ** 2:
            return None
        return records[int(np.flatnonzero(finite)[k])]

    def _on_click(self, canvas, index, pixel, modifiers):
        """Click inspects a dot; Shift adds it, Ctrl/⌘ toggles it."""
        if self.catalog is None:
            return
        record = self._nearest_record(canvas, index, pixel)
        if record is None:
            return
        pid = record.particle_id
        if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
            new = set(self.selected_ids) ^ {pid}
            self._change_selection(new, f"{'Removed' if pid in self.selected_ids else 'Added'} ROI "
                                        f"{record.roi_index} {'from' if pid in self.selected_ids else 'to'} "
                                        "the selection", current=pid)
        elif modifiers & Qt.ShiftModifier:
            self._change_selection(set(self.selected_ids) | {pid}, f"Added ROI {record.roi_index}", current=pid)
        else:
            self.set_current(pid)
            self.statusBar().showMessage(
                f"Inspecting ROI {record.roi_index} ({STATUS_NAME[record.status]}). "
                "Shift-click adds it, Ctrl/⌘-click toggles, double-click opens it.", 7000)

    def _on_double_click(self, canvas, index, pixel):
        record = self._nearest_record(canvas, index, pixel)
        if record is not None:
            self.set_current(record.particle_id)
            self.open_current()

    def _on_box(self, canvas, index, first, last, modifiers):
        if self.catalog is None or self.plot_tool != "select":
            return
        mode = self._effective_box_mode(modifiers)
        x0, x1 = sorted((first[0], last[0]))
        y0, y1 = sorted((first[1], last[1]))
        records = self._records()
        visible = [records[i] for i in self._visible_indices(canvas.kind)]
        if canvas.kind == "camera":
            xs, ys = self._values("camera", visible)
            where = "camera X–Y"
            axis_info = ("camera",)
        else:
            metric = self._metric_rows[index]
            xs, ys = self._values("metric", visible, metric)
            where = f"{dict(METRICS)[metric].split(' [')[0].lower()} plot"
            axis_info = ("plot", self.position_metric.currentData(), metric)
        inside_mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        inside = {visible[i].particle_id for i in np.flatnonzero(inside_mask)}
        statuses = {r.particle_id: r.status for r in visible}
        new, to_reject = apply_box(self.selected_ids, inside, mode, statuses, self.pick_scope.currentData())
        region = (axis_info[0], x0, x1, y0, y1, *axis_info[1:], mode)
        if mode == "reject":
            if not to_reject:
                self.statusBar().showMessage("Reject box: nothing to reject inside the box", 6000)
                return
            self._push_undo(f"Reject {len(to_reject)} in box", status_ids=to_reject, sets=True)
            self._set_flags(to_reject, "reject", reason=f"box on {where}")
            self._record_classified_batch("reject", to_reject)
            self.selected_ids = new
            self.regions.append(region)
            self.autosave()
            self.save_project_if_open()
            self._after_change()
            self._toast(f"✕ Rejected {len(to_reject):,} particles inside the box on the {where}. "
                        "They are red now. Ctrl+Z undoes.")
            return
        if new == self.selected_ids:
            skipped = len(inside) - len(inside & new) if mode in ("new", "add") else 0
            msg = "Box changed nothing"
            if skipped:
                msg += (f" — {skipped:,} particles inside are not '{self.pick_scope.currentText()}'. "
                        "Change 'New/Add boxes pick' to include them.")
            self.statusBar().showMessage(msg, 8000)
            return
        before = len(self.selected_ids)
        self.regions.append(region)
        verb = {"new": "Selected", "add": "Added", "remove": "Removed", "keep": "Kept"}[mode]
        count = {"new": len(new), "add": len(new) - before, "remove": before - len(new),
                 "keep": len(new)}[mode]
        skipped = len(inside) - len(inside & new) if mode in ("new", "add") else 0
        skipped_by = {}
        for pid in inside - new if mode in ("new", "add") else ():
            skipped_by[statuses[pid]] = skipped_by.get(statuses[pid], 0) + 1
        note = (" (skipped " + ", ".join(f"{n:,} {STATUS_NAME[k].lower()}" for k, n in sorted(skipped_by.items()))
                + ")") if skipped_by else ""
        self._change_selection(new, f"{verb} {count:,} particles on the {where}{note} · "
                                    f"{len(new):,} selected", push=True)

    def _on_hover(self, canvas, index, pixel):
        if not self.hover_action.isChecked() or self.catalog is None:
            return
        record = self._nearest_record(canvas, index, pixel, radius=9)
        if record is None:
            self._hide_hover()
            return
        if self.hover.isVisible() and self.hover.particle_id == record.particle_id:
            self.hover.move(QCursor.pos() + QPoint(18, 14))
            return
        self.hover.particle_id = record.particle_id
        tags = [f"<b style='color:{STATUS_COLOUR[record.status]}'>{STATUS_NAME[record.status]}</b>"]
        if record.particle_id in self.selected_ids:
            tags.append(f"<b style='color:{SELECTED_COLOUR}'>selected</b>")
        if record.label:
            label = self.labels.get(record.label)
            tags.append(f"<b style='color:{label.colour if label else '#495057'}'>{record.label}</b>"
                        + (f" ({label.size_text()})" if label else ""))
        html = (f"<span style='font-size:16px; font-weight:800'>{self._format(record.length_um, 0)} × "
                f"{self._format(record.width_um, 0)} µm</span><br>"
                f"{' · '.join(tags)}<br>Focus {self._format(record.focus, 1)} · ROI {record.roi_index} · "
                f"<span style='color:#516579'>{Path(record.source_file).name} {record.time_iso[11:19]}</span>")
        image = self._thumbs.get(record.particle_id)
        self.hover.show_at(QCursor.pos(), html, self._plain_pixmap(image, 96) if image is not None else None)
        if image is None and self.store is not None and self.store._source == record.source_file:
            self._hover_pending = record
            self._hover_timer.start(120)  # same MAT file already in memory: cheap

    def _hover_load_image(self):
        record = self._hover_pending
        if record is None or not self.hover.isVisible() or self.hover.particle_id != record.particle_id:
            return
        image = self._thumb_image(record, allow_load=False)
        if image is not None:
            self.hover.image.setPixmap(self._plain_pixmap(image, 96))

    def _hide_hover(self):
        self._hover_timer.stop()
        if hasattr(self, "hover"):
            self.hover.hide()
            self.hover.particle_id = None

    # ------------------------------------------------------- selection core --
    def _push_undo(self, label, status_ids=(), sets=False):
        records = self.catalog._by_id if self.catalog else {}
        entry = {"label": label, "selected": set(self.selected_ids), "regions": list(self.regions),
                 "current": self.current_id,
                 "statuses": {pid: (records[pid].status, records[pid].reject_reason, records[pid].label,
                                    records[pid].actual_length_um, records[pid].actual_width_um)
                              for pid in status_ids if pid in records},
                 "sets": copy.deepcopy(self.saved_sets) if sets else None,
                 "active_set": self.active_reference_set}
        self.undo_stack.append(entry)
        del self.undo_stack[:-60]

    def undo(self):
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo", 4000)
            return
        entry = self.undo_stack.pop()
        self.selected_ids = set(entry["selected"])
        self.regions = list(entry["regions"])
        records = self.catalog._by_id if self.catalog else {}
        for pid, (status, reason, label, length, width) in entry["statuses"].items():
            if pid in records:
                r = records[pid]
                r.status, r.reject_reason, r.label = status, reason, label
                r.actual_length_um, r.actual_width_um = length, width
        if entry["sets"] is not None:
            self.saved_sets = entry["sets"]
            self.active_reference_set = entry["active_set"]
            self._refresh_saved_sets()
            self.save_project_if_open()
        if entry["statuses"]:
            self.autosave()
        self._after_change()
        self._toast(f"↶ Undid: {entry['label']}")

    def _change_selection(self, new_ids, message=None, push=True, current=None):
        if push:
            self._push_undo(message or "selection change")
        self.selected_ids = set(new_ids)
        if current is not None:
            self.current_id = current
        self._after_change()
        if message:
            self._toast(message)

    def _after_change(self):
        """Selection, flags or sets changed: rebuild the browse list and redraw."""
        self._rebuild_browse()
        self._refresh_views()

    def _refresh_views(self):
        self._update_counts()
        if hasattr(self, "label_list"):
            self._refresh_label_widgets()
        self._update_selection_bar()
        self._redraw_plots()
        self._gallery_dirty = True
        if self._current_tab() == "gallery":
            self.update_gallery()
        self.refresh_particle()

    def _toast(self, message, ms=9000):
        self.statusBar().showMessage(message, ms)

    def _update_selection_bar(self):
        n = len(self.selected_ids)
        if not self.catalog:
            self.selection_summary.setText("<span style='color:#516579'>Load data to start.</span>")
        elif not n:
            self.selection_summary.setText(
                "<span style='color:#516579'>Nothing selected — drag a box on the Camera X–Y plot.</span>")
        else:
            by_id = self.catalog._by_id
            counts = {"undecided": 0, "keep": 0, "reject": 0}
            for pid in self.selected_ids:
                record = by_id.get(pid)
                if record is not None:
                    counts[record.status] += 1
            parts = [f"<span style='color:{STATUS_COLOUR[s]}'>{counts[s]:,} {STATUS_NAME[s].lower()}</span>"
                     for s in ("undecided", "keep", "reject") if counts[s]]
            self.selection_summary.setText(
                f"<b style='color:{SELECTED_COLOUR}; font-size:14px'>◉ {n:,} selected</b> &nbsp; "
                + " · ".join(parts))
        for button in (self.accept_btn, self.reject_btn, self.unflag_btn, self.clear_sel_btn):
            button.setEnabled(bool(n))
        self.undo_btn.setEnabled(bool(self.undo_stack))
        if self.undo_stack:
            self.undo_btn.setToolTip(f"Undo: {self.undo_stack[-1]['label']} (Ctrl+Z)")

    def _update_counts(self):
        records = self._records()
        counts = {"undecided": 0, "keep": 0, "reject": 0}
        for r in records:
            counts[r.status] = counts.get(r.status, 0) + 1
        committed = committed_particle_ids(self.saved_sets)
        if hasattr(self, "count_labels"):
            for key in counts:
                self.count_labels[key].setText(f"{counts[key]:,}")
            self.count_labels["selected"].setText(f"{len(self.selected_ids):,}")
            self.count_labels["committed"].setText(f"{len(committed):,}")
            total = max(1, len(records))
            done = counts["keep"] + counts["reject"]
            self.progress_bar.setText(
                f"<span style='color:#516579'>Reviewed {done:,} of {len(records):,} "
                f"({100 * done / total:.0f}%)</span>")
        shown = {s: 0 for s in counts}
        for i in self._base_indices:
            shown[records[i].status] += 1
        for status, box in self.show_status.items():
            box.setText(f"{'✕' if status == 'reject' else '●'} {STATUS_NAME[status]} ({shown[status]:,})")

    def clear_selection(self):
        if not self.selected_ids:
            return
        self.regions.clear()
        self._change_selection(set(), "Selection cleared (flags unchanged). Ctrl+Z restores it.")

    def select_all_visible(self):
        if not self.catalog:
            return
        allowed = {"undecided": {"undecided"}, "unrejected": {"undecided", "keep"}, "keep": {"keep"},
                   "reject": {"reject"}, "any": {"undecided", "keep", "reject"}}[self.pick_scope.currentData()]
        records = self._records()
        ids = {records[i].particle_id for i in self._visible_indices("camera") if records[i].status in allowed}
        self._change_selection(ids, f"Selected all {len(ids):,} visible {self.pick_scope.currentText().lower()} "
                                    "particles")

    def select_group(self, key):
        if not self.catalog:
            return
        if key == "committed":
            ids = committed_particle_ids(self.saved_sets) & set(self.catalog._by_id)
            label = "particles in calibration sets"
        else:
            ids = {r.particle_id for r in self.catalog.records if r.status == key}
            label = f"{STATUS_NAME[key].lower()} particles"
        if key in ("keep", "reject"):
            self.show_status[key].setChecked(True)
        self._change_selection(ids, f"Selected all {len(ids):,} {label}")

    def browse_group(self, key):
        index = self.view_selector.findData(key)
        if index >= 0:
            self.view_selector.setCurrentIndex(index)
        self.viewer_tabs.setCurrentWidget(self.gallery_panel)

    # ------------------------------------------------------------ flagging --
    def _set_flags(self, ids, status, reason=""):
        by_id = self.catalog._by_id
        for pid in ids:
            record = by_id.get(pid)
            if record is not None:
                record.status = status
                record.reject_reason = reason if status == "reject" else ""
                if status == "keep" and self._accept_label_name():
                    self._set_label(record, self._accept_label_name())
                elif status == "undecided":
                    self._set_label(record, "")  # back to the default: unreviewed, unlabelled

    def _set_label(self, record, name):
        """Tag a record with a library label (and its true size), or clear it."""
        if not name:
            if record.label:
                record.actual_length_um = record.actual_width_um = float("nan")
            record.label = ""
            return
        label = self.labels.get(name)
        record.label = label.name if label else name
        if label is not None:
            record.actual_length_um = label.length_um
            record.actual_width_um = label.width_um if math.isfinite(label.width_um) else label.length_um

    def _accept_label_name(self):
        data = self.accept_label.currentData() if hasattr(self, "accept_label") else None
        return data if data and data != "__new__" else ""

    def _flag_ids(self, ids, status, label, clear_selection=False, batch=True, reason=None):
        if not self.catalog:
            return 0
        ids = [pid for pid in dict.fromkeys(ids) if pid in self.catalog._by_id]
        if not ids:
            return 0
        batch = batch and status in ("keep", "reject")
        self._push_undo(label, status_ids=ids, sets=batch)
        self._set_flags(ids, status, reason if reason is not None else
                        (self.reject_reason.currentText() or "manual"))
        if batch:
            self._record_classified_batch(status, ids)
        if clear_selection:
            self.selected_ids.clear()
            self.regions.clear()
        elif status == "reject":
            # A rejected particle must never ride along in a later "Accept selection".
            self.selected_ids.difference_update(ids)
        self.autosave()
        return len(ids)

    def accept_selection(self):
        n = self._flag_ids(self.selected_ids, "keep", f"Accept {len(self.selected_ids):,}", clear_selection=True)
        if n:
            self._after_change()
            label = self._accept_label_name()
            self._toast(f"✓ Accepted {n:,} particles" + (f" as '{label}'" if label else "") +
                        " — now green, saved, and skipped by new boxes. Ctrl+Z undoes; Unflag resets them.")

    def reject_selection(self):
        n = self._flag_ids(self.selected_ids, "reject", f"Reject {len(self.selected_ids):,}",
                           clear_selection=True, reason="graphical selection")
        if n:
            self._after_change()
            self._toast(f"✕ Rejected {n:,} particles — now red and saved. Ctrl+Z undoes.")

    def unflag_selection(self):
        n = self._flag_ids(self.selected_ids, "undecided", f"Unflag {len(self.selected_ids):,}",
                           clear_selection=True, batch=False)
        if n:
            self._after_change()
            self._toast(f"↺ {n:,} particles returned to unreviewed (grey, no label) and saved.")

    def _record_classified_batch(self, decision, ids):
        """Keep/reject actions create a removable, named batch (Flag history)."""
        prefix = "Accepted" if decision == "keep" else "Rejected"
        idx = 1
        while f"{prefix} batch {idx}" in self.saved_sets:
            idx += 1
        self.saved_sets[f"{prefix} batch {idx}"] = make_saved_set(ids, description=f"Review batch: {decision}")
        self._refresh_saved_sets()
        self.save_project_if_open()

    def _clear_selection_flags(self):
        self.unflag_selection()

    def set_status(self, status: str) -> None:
        """Flag the particle in the viewer, then (optionally) step to the next."""
        record = self.current_record()
        if record is None:
            return
        pid = record.particle_id
        old_pos = self._filtered_pos.get(pid, self.cursor)
        reason = (self.reject_reason.currentText() or "manual") if status == "reject" else ""
        self._flag_ids([pid], status, f"{STATUS_NAME[status]} ROI {record.roi_index}", batch=False, reason=reason)
        self._rebuild_browse()
        if self.auto_advance.isChecked() and self.filtered:
            if pid in self._filtered_pos:
                pos = min(len(self.filtered) - 1, self._filtered_pos[pid] + 1)
            else:  # it left the browse list (e.g. browsing unreviewed): next one slid into place
                pos = min(len(self.filtered) - 1, old_pos)
            self.cursor = pos
            self.current_id = self._records()[self.filtered[pos]].particle_id
        self._refresh_views()
        self._toast(f"ROI {record.roi_index} → {STATUS_NAME[status].lower()} (saved)", 5000)

    def _particle_label_chosen(self, _index):
        record = self.current_record()
        if record is None:
            return
        name = self.particle_label_combo.currentData() or ""
        self._push_undo(f"Label ROI {record.roi_index}", status_ids=[record.particle_id])
        self._set_label(record, name)
        self.autosave()
        self._refresh_views()
        self._toast(f"ROI {record.roi_index} labelled '{name}'" if name else f"ROI {record.roi_index}: label removed",
                    5000)

    def toggle_current_in_selection(self):
        record = self.current_record()
        if record is None:
            return
        if record.particle_id in self.selected_ids:
            self.remove_current_from_working()
        else:
            self.current_to_working()

    def current_to_working(self, replace=False):
        record = self.current_record()
        if record is None:
            return
        new = {record.particle_id} if replace else set(self.selected_ids) | {record.particle_id}
        self._change_selection(new, f"Added ROI {record.roi_index} to the selection")

    def remove_current_from_working(self):
        record = self.current_record()
        if record is None or record.particle_id not in self.selected_ids:
            return
        self._change_selection(set(self.selected_ids) - {record.particle_id},
                               f"Removed ROI {record.roi_index} from the selection")

    # ------------------------------------------------------------- browsing --
    def change_view_scope(self):
        self._gallery_page = 0
        self._rebuild_browse()
        self._gallery_dirty = True
        if self._current_tab() == "gallery":
            self.update_gallery()
        self.refresh_particle()

    @property
    def view_scope(self):
        return self.view_selector.currentData()

    def _effective_browse_scope(self):
        scope = self.view_selector.currentData()
        if scope == "auto":
            return "selection" if self.selected_ids else "all"
        return scope

    def _rebuild_browse(self):
        records = self._records()
        scope = self._effective_browse_scope()
        if scope == "selection":
            filtered = [i for i in self._base_indices if records[i].particle_id in self.selected_ids]
        elif scope in ("undecided", "keep", "reject"):
            filtered = [i for i in self._base_indices if records[i].status == scope]
        else:
            filtered = list(self._base_indices)
        self.filtered = filtered
        self._filtered_pos = {records[i].particle_id: pos for pos, i in enumerate(filtered)}
        if self.current_id in self._filtered_pos:
            self.cursor = self._filtered_pos[self.current_id]
        else:
            self.cursor = max(0, min(self.cursor, len(filtered) - 1))
            if self.current_id is None or (self.catalog and self.current_id not in self.catalog._by_id):
                self.current_id = records[filtered[self.cursor]].particle_id if filtered else None
        pages = max(1, (len(filtered) + GALLERY_PAGE - 1) // GALLERY_PAGE)
        self._gallery_page = min(self._gallery_page, pages - 1)

    def current_record(self) -> Optional[ParticleRecord]:
        if not self.catalog:
            return None
        if self.current_id is not None:
            record = self.catalog._by_id.get(self.current_id)
            if record is not None:
                return record
        if self.filtered:
            self.cursor = max(0, min(self.cursor, len(self.filtered) - 1))
            return self.catalog.records[self.filtered[self.cursor]]
        return None

    def set_current(self, pid, open_viewer=False):
        self.current_id = pid
        if pid in self._filtered_pos:
            self.cursor = self._filtered_pos[pid]
        self._draw_current_marker()
        self.refresh_particle()
        if self._current_tab() == "gallery":
            self._redecorate_gallery()
        if open_viewer:
            self.viewer_tabs.setCurrentWidget(self.particle_tab)

    def open_current(self):
        if self.current_record() is not None:
            self.viewer_tabs.setCurrentWidget(self.particle_tab)

    def navigate(self, delta: int) -> None:
        if not self.filtered:
            return
        if self.current_id in self._filtered_pos:
            pos = self._filtered_pos[self.current_id] + delta
        else:
            pos = self.cursor if delta > 0 else self.cursor - 1
        pos = max(0, min(len(self.filtered) - 1, pos))
        self.cursor = pos
        self.set_current(self._records()[self.filtered[pos]].particle_id)
        if self._current_tab() == "gallery":
            page = pos // GALLERY_PAGE
            if page != self._gallery_page:
                self._gallery_page = page
                self.update_gallery()

    def refresh_particle(self) -> None:
        record = self.current_record()
        total = len(self.filtered)
        scope = dict(BROWSE_SCOPES).get(self.view_selector.currentData(), "")
        if record is None:
            self.progress_label.setText(f"0 / {total:,}")
            self.current_label.setText("<span style='color:#516579'>No particle</span>")
            self.current_thumb.clear()
            if self._current_tab() == "particle":
                self.canvas.show_message("No particle to show.\nLoad data, or change Browse / Filters.")
                self.particle_badge.setText("—")
            return
        pid = record.particle_id
        position = self._filtered_pos.get(pid)
        self.progress_label.setText(f"{position + 1:,} / {total:,}" if position is not None else f"– / {total:,}")
        self.progress_label.setToolTip(f"Browsing: {scope}" + ("" if position is not None else
                                       " (current particle is not in this list)"))
        in_sel = pid in self.selected_ids
        self.current_label.setText(
            f"<b>ROI {record.roi_index}</b> · {Path(record.source_file).name} · "
            f"<b style='color:{STATUS_COLOUR[record.status]}'>{STATUS_NAME[record.status]}</b>"
            + (f" · <b style='color:{SELECTED_COLOUR}'>selected</b>" if in_sel else "")
            + (f" · 🏷 {record.label}" if record.label else "")
            + f" · F {self._format(record.focus, 1)} · L {self._format(record.length_um, 1)} µm")
        image = self._thumb_image(record, allow_load=self._current_tab() != "select" or
                                  (self.store is not None and self.store._source == record.source_file))
        if image is not None:
            self.current_thumb.setPixmap(self._plain_pixmap(image, 40))
        else:
            self.current_thumb.setText("·")
        if hasattr(self, "set_radius_label"):
            self._set_reference_changed(redraw=False)
        if self._current_tab() != "particle":
            return
        try:
            image, boundary = self.store.particle(record, subtract_background=self.background_checkbox.isChecked())
        except Exception as exc:
            self.canvas.show_message(f"Cannot read image:\n{exc}")
            return
        colour = SELECTED_COLOUR if in_sel else (STATUS_COLOUR[record.status] if record.status != "undecided"
                                                 else None)
        self.canvas.show_particle(image, boundary, f"ROI {record.roi_index} · {STATUS_NAME[record.status]}"
                                  + (" · selected" if in_sel else ""),
                                  self.boundary_checkbox.isChecked(), colour)
        self.particle_badge.setText(STATUS_NAME[record.status].upper())
        self.particle_badge.setStyleSheet(
            f"background:{STATUS_COLOUR[record.status]}; color:white; font-size:17px; font-weight:800;"
            "border-radius:6px; padding:8px;")
        committed = [name for name, s in self.saved_sets.items()
                     if not s.get("description", "").startswith("Review batch") and pid in s["particle_ids"]]
        self.particle_membership.setText(
            (f"<b style='color:{SELECTED_COLOUR}'>In the selection</b>" if in_sel else "Not selected")
            + (f"<br><span style='color:{COMMITTED_COLOUR}'>In set: {', '.join(committed)}</span>"
               if committed else ""))
        self.particle_membership.setTextFormat(Qt.RichText)
        self.toggle_selection_button.setText("− Remove from selection  (Del)" if in_sel
                                             else "+ Add to selection  (A)")
        self.metric_labels["time"].setText(record.time_iso or "—")
        self.metric_labels["length"].setText(self._format(record.length_um, 2))
        self.metric_labels["width"].setText(self._format(record.width_um, 2))
        self.metric_labels["focus"].setText(self._format(record.focus, 3))
        self.metric_labels["roundness"].setText(self._format(record.roundness, 3))
        self.metric_labels["xy"].setText(f"{self._format(record.image_x_px, 1)}, {self._format(record.image_y_px, 1)}")
        reference = self._draft_reference_for(record)
        side = ""
        if reference is not None and math.isfinite(record.stage_x) and math.isfinite(record.stage_y):
            d = signed_plane_distance(record.stage_x, record.stage_y, reference, self.plane_normal.value())
            side = f"\nd = {d:+.4g} ({'+' if d > 0 else '−' if d < 0 else 'on the'} side{'' if d else ' plane'})"
        self.metric_labels["stage"].setText(f"{self._format(record.stage_x, 3)}, {self._format(record.stage_y, 3)}"
                                            + side)
        self.metric_labels["actual"].setText(
            f"{self._format(record.actual_length_um, 2)} × {self._format(record.actual_width_um, 2)}")
        self.metric_labels["instrument"].setText(record.instrument + (f" · label: {record.label}" if record.label else ""))
        self.metric_labels["source"].setText(f"{Path(record.source_file).name} [{record.roi_index}]")
        self.metric_labels["source"].setToolTip(record.source_file)
        self.reject_reason.setCurrentText(record.reject_reason if record.status == "reject" else "")
        self.particle_label_combo.blockSignals(True)
        self.particle_label_combo.clear()
        self.particle_label_combo.addItem("(no label)", "")
        for name, label in self.labels.labels.items():
            self.particle_label_combo.addItem(self._swatch(label.colour), f"{name} · {label.size_text()}", name)
        if record.label and self.particle_label_combo.findData(record.label) < 0:
            self.particle_label_combo.addItem(f"{record.label} (not in library)", record.label)
        self.particle_label_combo.setCurrentIndex(max(0, self.particle_label_combo.findData(record.label)))
        self.particle_label_combo.blockSignals(False)
        self._updating_z = True
        z = min(self.z_max_um, max(self.z_min_um, record.object_plane_um))
        self.z_spin.setValue(z)
        self.z_slider.setValue(int(round((z - self.z_min_um) / self.z_step_um)))
        self._updating_z = False

    @staticmethod
    def _format(value: float, digits: int) -> str:
        return ("{:.%df}" % digits).format(value) if math.isfinite(value) else "—"

    # --------------------------------------------------------------- gallery --
    def _thumb_image(self, record, allow_load=True) -> Optional[QImage]:
        pid = record.particle_id
        image = self._thumbs.get(pid)
        if image is not None:
            self._thumbs.move_to_end(pid)
            return image
        if self.store is None or (not allow_load and self.store._source != record.source_file):
            return None
        try:
            arr, _ = self.store.particle(record)
        except Exception:
            return None
        image = _to_gray8(arr)
        if image is None:
            return None
        if max(image.width(), image.height()) > 2 * THUMB:
            image = image.scaled(2 * THUMB, 2 * THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._thumbs[pid] = image
        while len(self._thumbs) > 6000:
            self._thumbs.popitem(last=False)
        return image

    @staticmethod
    def _plain_pixmap(image, size):
        return QPixmap.fromImage(image).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _decorated_icon(self, image, record):
        pid = record.particle_id
        selected = pid in self.selected_ids
        size = THUMB + 12
        pm = QPixmap(size, size)
        pm.fill(QColor("#ffffff"))
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        if image is not None:
            scaled = QPixmap.fromImage(image).scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
        else:
            painter.setPen(QColor("#8a99a8"))
            painter.drawText(pm.rect(), Qt.AlignCenter, "no image")
        if selected:
            colour, width = QColor(SELECTED_COLOUR), 6
        elif record.status != "undecided":
            colour, width = QColor(STATUS_COLOUR[record.status]), 5
        else:
            colour, width = QColor("#c9d2db"), 2
        painter.setPen(QPen(colour, width))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(width // 2, width // 2, size - width, size - width)
        badges = []
        if record.status == "keep":
            badges.append(("✓", STATUS_COLOUR["keep"]))
        elif record.status == "reject":
            badges.append(("✕", STATUS_COLOUR["reject"]))
        if selected:
            badges.append(("◉", SELECTED_COLOUR))
        font = QFont(painter.font())
        font.setBold(True)
        font.setPixelSize(13)
        painter.setFont(font)
        for k, (text, col) in enumerate(badges):
            rect = QRect(6 + k * 22, 6, 20, 20)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(col))
            painter.drawEllipse(rect)
            painter.setPen(QColor("white"))
            painter.drawText(rect, Qt.AlignCenter, text)
        if pid == self.current_id:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(CURRENT_COLOUR))
            painter.drawPolygon([QPoint(size - 22, 2), QPoint(size - 2, 2), QPoint(size - 2, 22)])
        painter.end()
        return QIcon(pm)

    def _gallery_text(self, record):
        state = STATUS_NAME[record.status]
        first = f"ROI {record.roi_index} · {record.label or state}"
        if len(first) > 22:
            first = first[:21] + "…"
        return f"{first}\nF {self._format(record.focus, 1)} · {self._format(record.length_um, 0)} µm"

    def gallery_page(self, delta: int) -> None:
        pages = max(1, (len(self.filtered) + GALLERY_PAGE - 1) // GALLERY_PAGE)
        self._gallery_page = max(0, min(pages - 1, self._gallery_page + delta))
        self.update_gallery()

    def update_gallery(self) -> None:
        if self._current_tab() != "gallery":
            self._gallery_dirty = True
            return
        self._gallery_dirty = False
        picked = set(self._gallery_picked_ids())
        self.gallery.blockSignals(True)
        self.gallery.clear()
        if not self.catalog or not self.store:
            self.gallery_info.setText("No images")
            self.gallery.blockSignals(False)
            self._gallery_picks_changed()
            return
        records = self._records()
        page_indices = self.filtered[self._gallery_page * GALLERY_PAGE:(self._gallery_page + 1) * GALLERY_PAGE]
        page_records = [records[i] for i in page_indices]
        # Load images grouped by MAT file, so each (large) file is read once per page.
        missing = [r for r in page_records if r.particle_id not in self._thumbs]
        if missing:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                current_source = self.store._source
                for record in sorted(missing, key=lambda r: (r.source_file != current_source, r.source_file)):
                    self._thumb_image(record)
            finally:
                QApplication.restoreOverrideCursor()
        for record in page_records:
            item = QListWidgetItem(self._decorated_icon(self._thumbs.get(record.particle_id), record),
                                   self._gallery_text(record))
            item.setData(Qt.UserRole, record.particle_id)
            item.setForeground(QColor(STATUS_COLOUR[record.status] if record.status != "undecided" else "#33475b"))
            item.setToolTip(f"{Path(record.source_file).name} · ROI {record.roi_index}\n"
                            f"Focus {self._format(record.focus, 2)} · Length {self._format(record.length_um, 1)} µm"
                            f"\nCamera ({self._format(record.image_x_px, 0)}, {self._format(record.image_y_px, 0)})")
            self.gallery.addItem(item)
            if record.particle_id in picked:
                item.setSelected(True)
        self.gallery.blockSignals(False)
        pages = max(1, (len(self.filtered) + GALLERY_PAGE - 1) // GALLERY_PAGE)
        scope = self._effective_browse_scope()
        label = {"selection": "selected", "all": "", "undecided": "unreviewed", "keep": "accepted",
                 "reject": "rejected"}[scope]
        self.gallery_info.setText(f"Page {self._gallery_page + 1} / {pages} · {len(self.filtered):,} "
                                  f"{label + ' ' if label else ''}particles")
        self._gallery_picks_changed()

    def _redecorate_gallery(self):
        if not self.catalog:
            return
        by_id = self.catalog._by_id
        for row in range(self.gallery.count()):
            item = self.gallery.item(row)
            record = by_id.get(item.data(Qt.UserRole))
            if record is not None:
                item.setIcon(self._decorated_icon(self._thumbs.get(record.particle_id), record))

    def _gallery_picked_ids(self):
        return [item.data(Qt.UserRole) for item in self.gallery.selectedItems()]

    def _gallery_picks_changed(self):
        n = len(self.gallery.selectedItems())
        self.gallery_pick_label.setText(f"<b>{n} picked</b>" if n else "Nothing picked")
        self.gallery_pick_label.setTextFormat(Qt.RichText)
        for button in self.gallery_actions:
            button.setEnabled(bool(n))

    def gallery_choose(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.UserRole)
        if pid:
            self.current_id = pid
            self.cursor = self._filtered_pos.get(pid, self.cursor)
            self.refresh_particle()
            self._redecorate_gallery()

    def gallery_open(self, item):
        self.gallery_choose(item)
        self.open_current()

    def gallery_selection(self, action):
        ids = self._gallery_picked_ids()
        if not ids:
            self.statusBar().showMessage("Pick thumbnails first (click, Ctrl/⌘-click or drag)", 5000)
            return
        if action == "remove":
            new = set(self.selected_ids) - set(ids)
            removed = len(self.selected_ids) - len(new)
            self.gallery.clearSelection()
            self._change_selection(new, f"Removed {removed} picked particles from the selection")
        else:
            new = set(self.selected_ids) | set(ids)
            self._change_selection(new, f"Added {len(new) - len(self.selected_ids)} picked particles "
                                        "to the selection")

    def flag_gallery(self, status: str) -> None:
        ids = self._gallery_picked_ids()
        if not ids:
            self.statusBar().showMessage("Pick thumbnails first (click, Ctrl/⌘-click or drag)", 5000)
            return
        n = self._flag_ids(ids, status, f"{STATUS_NAME[status]} {len(ids)} in gallery",
                           batch=status != "undecided",
                           reason=(self.reject_reason.currentText() or "gallery") if status == "reject" else "")
        self.gallery.clearSelection()
        self._after_change()
        extra = " and removed from the selection" if status == "reject" else ""
        self._toast(f"{n} picked particles → {STATUS_NAME[status].lower()}{extra} (saved)")

    # ---------------------------------------------------------------- labels --
    def _label_counts(self):
        counts = {}
        for record in self._records():
            if record.label:
                counts[record.label] = counts.get(record.label, 0) + 1
        return counts

    def _refresh_label_widgets(self):
        if not hasattr(self, "label_list"):
            return
        counts = self._label_counts()
        chosen = self._accept_label_name()
        self.accept_label.blockSignals(True)
        self.accept_label.clear()
        self.accept_label.addItem("(no label)", "")
        for name, label in self.labels.labels.items():
            self.accept_label.addItem(self._swatch(label.colour), f"{name} · {label.size_text()}", name)
        self.accept_label.addItem("＋ New label…", "__new__")
        index = self.accept_label.findData(self.labels.resolve(chosen)) if chosen else 0
        self.accept_label.setCurrentIndex(max(0, index))
        self.accept_label.blockSignals(False)
        current = self._selected_label_name()
        self.label_list.blockSignals(True)
        self.label_list.clear()
        for name, label in self.labels.labels.items():
            item = QListWidgetItem(self._swatch(label.colour),
                                   f"{name}\n   {label.kind} · {label.size_text()} · {counts.get(name, 0):,} loaded")
            item.setData(Qt.UserRole, name)
            item.setToolTip(label.notes or name)
            self.label_list.addItem(item)
            if name == current:
                self.label_list.setCurrentItem(item)
        unknown = sorted(n for n in counts if self.labels.get(n) is None)
        for name in unknown:
            item = QListWidgetItem(f"{name}  (not in library)\n   {counts[name]:,} loaded — Edit to add a size")
            item.setData(Qt.UserRole, name)
            item.setForeground(QColor("#a61e4d"))
            self.label_list.addItem(item)
        self.label_list.blockSignals(False)
        self.label_path.setText(f"{self.labels.path}")
        if self.label_list.count() == 0:
            self.label_detail.setText("<span style='color:#516579'>No labels yet — click ＋ New…</span>")
        else:
            self._label_selected()

    @staticmethod
    def _swatch(colour):
        pm = QPixmap(14, 14)
        pm.fill(QColor(colour or "#adb5bd"))
        return QIcon(pm)

    def _selected_label_name(self):
        item = self.label_list.currentItem() if hasattr(self, "label_list") else None
        return item.data(Qt.UserRole) if item is not None else None

    def _label_selected(self):
        name = self._selected_label_name()
        if not name:
            self.label_detail.setText("")
            return
        label = self.labels.get(name)
        n = self._label_counts().get(name, 0)
        if label is None:
            self.label_detail.setText(f"<b>{name}</b> is used by {n:,} loaded particles but is not in the "
                                      "library. Use Edit to add it with its size.")
            return
        self.label_detail.setText(
            f"<b style='color:{label.colour}'>{label.name}</b> — {label.kind}, {label.size_text()}<br>"
            f"{n:,} loaded particles" + (f"<br><span style='color:#516579'>{label.notes}</span>" if label.notes else "")
            + (f"<br><span style='color:#516579'>Formerly: {', '.join(label.aliases)}</span>" if label.aliases else ""))

    def _accept_label_chosen(self, _index):
        if self.accept_label.currentData() == "__new__":
            name = self.new_label()
            index = self.accept_label.findData(name) if name else 0
            self.accept_label.setCurrentIndex(max(0, index))

    def _save_label_library(self):
        try:
            self.labels.save()
        except OSError as exc:
            QMessageBox.warning(self, "Label library", f"Could not save {self.labels.path}:\n{exc}")

    def new_label(self):
        dialog = LabelDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return None
        label = dialog.result_label()
        if self.labels.get(label.name) is not None and label.name in self.labels.labels:
            QMessageBox.warning(self, "Label", f"'{label.name}' already exists — use Edit to change it.")
            return None
        self.labels.upsert(label)
        self._save_label_library()
        self._refresh_label_widgets()
        self._toast(f"Created label '{label.name}' ({label.size_text()}). Choose it in 'Accept as' to use it.")
        return label.name

    def edit_label(self):
        name = self._selected_label_name()
        if not name:
            self.statusBar().showMessage("Choose a label in the list first", 4000)
            return
        existing = self.labels.get(name)
        used = self._label_counts().get(name, 0)
        dialog = LabelDialog(self, existing or ParticleLabel(name), used)
        if dialog.exec() != QDialog.Accepted:
            return
        label = dialog.result_label()
        old_name = existing.name if existing else None
        if existing is None and label.name in self.labels.labels:
            QMessageBox.warning(self, "Label", f"'{label.name}' already exists.")
            return
        try:
            self.labels.upsert(label, old_name=old_name)
        except ValueError as exc:
            QMessageBox.warning(self, "Label", str(exc))
            return
        if existing is None and label.name != name:
            label.aliases = sorted(set(label.aliases) | {name})
        self._save_label_library()
        if self.catalog:
            affected = [r.particle_id for r in self.catalog.records if r.label == name]
            if affected:
                self._push_undo(f"Edit label '{name}'", status_ids=affected)
                for pid in affected:
                    self._set_label(self.catalog._by_id[pid], label.name)
                self.autosave()
        self._refresh_label_widgets()
        self._after_change()
        self._toast(f"Label '{label.name}' saved for all datasets: {label.size_text()}"
                    + (f" · {used:,} loaded particles updated" if used else ""))

    def delete_label(self):
        name = self._selected_label_name()
        if not name:
            return
        used = self._label_counts().get(name, 0)
        if QMessageBox.question(self, "Delete label?",
                                f"Delete '{name}' from the shared library?\n\n{used:,} loaded particles carry it; "
                                "they keep their accepted/rejected flags but lose the label and its size. "
                                "Other datasets keep the name but no size until it is re-created."
                                ) != QMessageBox.Yes:
            return
        if self.catalog and used:
            affected = [r.particle_id for r in self.catalog.records if r.label == name]
            self._push_undo(f"Delete label '{name}'", status_ids=affected)
            for pid in affected:
                self._set_label(self.catalog._by_id[pid], "")
            self.autosave()
        self.labels.delete(name)
        self._save_label_library()
        self._refresh_label_widgets()
        self._after_change()

    def select_label(self, how="replace"):
        name = self._selected_label_name()
        if not name or not self.catalog:
            self.statusBar().showMessage("Choose a label in the list first", 4000)
            return
        ids = {r.particle_id for r in self.catalog.records if r.label == name}
        for status in ("keep", "reject"):
            self.show_status[status].setChecked(True)
        new = ids if how == "replace" else set(self.selected_ids) | ids
        self._change_selection(new, f"Selected {len(ids):,} particles labelled '{name}'")

    def browse_label(self):
        self.select_label("replace")
        self.browse_group("selection")

    def assign_label_to_selection(self):
        name = self._selected_label_name()
        if not name or not self.selected_ids:
            self.statusBar().showMessage("Choose a label and select particles first", 5000)
            return
        ids = sorted(self.selected_ids)
        self._push_undo(f"Label {len(ids):,} as '{name}'", status_ids=ids)
        for pid in ids:
            self._set_label(self.catalog._by_id[pid], name)
        self.autosave()
        self._refresh_label_widgets()
        self._after_change()
        self._toast(f"Tagged {len(ids):,} selected particles as '{name}' (flags unchanged).")

    def clear_label_from_selection(self):
        ids = sorted(pid for pid in self.selected_ids if self.catalog and self.catalog._by_id[pid].label)
        if not ids:
            self.statusBar().showMessage("No labelled particles in the selection", 4000)
            return
        self._push_undo(f"Remove labels from {len(ids):,}", status_ids=ids)
        for pid in ids:
            self._set_label(self.catalog._by_id[pid], "")
        self.autosave()
        self._refresh_label_widgets()
        self._after_change()
        self._toast(f"Removed labels from {len(ids):,} particles.")

    def choose_label_library(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Label library (existing or new)", str(self.labels.path),
                                                  "Label library (*.json)",
                                                  options=QFileDialog.DontConfirmOverwrite)
        if not filename:
            return
        try:
            library = LabelLibrary.load(filename)
        except Exception as exc:
            QMessageBox.warning(self, "Label library", f"Could not read {filename}:\n{exc}")
            return
        if not Path(filename).exists():
            library.merge([l.as_json() for l in self.labels.labels.values()])
        self.labels = library
        self._save_label_library()
        os.environ["CPI_CALIBRATION_LABELS"] = str(filename)
        if self.catalog:
            apply_label_dimensions(self.catalog.records, self.labels)
        self._refresh_label_widgets()
        self._after_change()
        self._toast(f"Using label library {filename}. Set CPI_CALIBRATION_LABELS to make this permanent.")

    # ------------------------------------------------------- calibration sets --
    def _calibration_set_names(self):
        return sorted(name for name, s in self.saved_sets.items()
                      if not s.get("description", "").startswith("Review batch: "))

    def _refresh_saved_sets(self):
        self._reference_index = build_reference_index(self.saved_sets, self.active_reference_set)
        self._active_members = set(self.saved_sets.get(self.active_reference_set, {}).get("particle_ids", ()))
        self._draft_reference_cache = None
        if not hasattr(self, "saved_sets_list"):
            return
        current = self._selected_saved_set_name()
        self.saved_sets_list.blockSignals(True)
        self.saved_sets_list.clear()
        active_row = -1
        for name in self._calibration_set_names():
            spec = self.saved_sets[name]
            ref = spec.get("best_focus")
            desc = f"{name}   ({len(spec['particle_ids']):,})"
            if ref is not None:
                desc += f"   X={ref['x_best']:g}  Z={ref['z_best']:g}"
            if name == self.active_reference_set:
                desc = "● " + desc
            item = QListWidgetItem(desc)
            item.setData(Qt.UserRole, name)
            self.saved_sets_list.addItem(item)
            if name == current:
                active_row = self.saved_sets_list.count() - 1
        self.saved_sets_list.blockSignals(False)
        if active_row >= 0:
            self.saved_sets_list.setCurrentRow(active_row)
        self._update_reference_hint()
        # Flag history (review batches), newest first.
        if hasattr(self, "batch_list"):
            selected_batch = self._selected_batch_name()
            self.batch_list.clear()
            batches = [(n, s) for n, s in self.saved_sets.items()
                       if s.get("description", "").startswith("Review batch: ")]
            batches.sort(key=lambda item: item[1].get("created_utc", ""), reverse=True)
            for name, spec in batches:
                keep = spec["description"].endswith("keep")
                try:
                    stamp = datetime.fromisoformat(spec.get("created_utc", "")).astimezone().strftime("%d %b %H:%M")
                except ValueError:
                    stamp = ""
                item = QListWidgetItem(f"{'✓' if keep else '✕'} {name} · {len(spec['particle_ids']):,}"
                                       + (f" · {stamp}" if stamp else ""))
                item.setForeground(QColor(STATUS_COLOUR["keep" if keep else "reject"]))
                item.setData(Qt.UserRole, name)
                self.batch_list.addItem(item)
                if name == selected_batch:
                    item.setSelected(True)
                    self.batch_list.setCurrentItem(item)

    def _selected_saved_set_name(self):
        item = self.saved_sets_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def _selected_batch_name(self):
        item = self.batch_list.currentItem() if hasattr(self, "batch_list") else None
        return item.data(Qt.UserRole) if item is not None else None

    def select_batch(self):
        name = self._selected_batch_name()
        if not name or not self.catalog:
            self.statusBar().showMessage("Choose a batch in Flag history first", 4000)
            return
        ids = set(self.saved_sets[name]["particle_ids"]) & set(self.catalog._by_id)
        status = self.saved_sets[name]["description"].partition(": ")[2]
        if status in self.show_status:
            self.show_status[status].setChecked(True)
        self._change_selection(ids, f"Selected the {len(ids):,} particles of '{name}'")

    def _set_reference_changed(self, redraw=True):
        if hasattr(self, "best_x_set") and hasattr(self, "best_z_set"):
            radius = math.hypot(self.best_x_set.value(), self.best_z_set.value())
            self._draft_reference_cache = None
            rec = self.current_record()
            offset = (f" · current ROI: ΔX={rec.stage_x - self.best_x_set.value():.4g}, "
                      f"ΔZ={rec.stage_y - self.best_z_set.value():.4g}"
                      if rec is not None and math.isfinite(rec.stage_x) and math.isfinite(rec.stage_y) else "")
            self.set_radius_label.setText(f"Draft best radius = {radius:.5g}" + offset)
            if redraw and hasattr(self, "plot_canvas"):
                self.update_plots()

    def _update_reference_hint(self):
        if not hasattr(self, "active_set_label"):
            return
        name = self.active_reference_set
        if name and name in self.saved_sets and self.saved_sets[name].get("best_focus"):
            self.active_set_label.setText(f"Active reference: {name}. It applies to its members; overlapping "
                                          "sets with conflicting references need an active set.")
        else:
            self.active_set_label.setText("No active set. Unique saved X/Z references apply automatically.")

    def move_reference_line(self, value):
        axis = self.position_metric.currentData()
        if axis not in ("stage_x", "stage_z"):
            return
        (self.best_x_set if axis == "stage_x" else self.best_z_set).setValue(value)
        self._toast(f"Draft {axis[-1].upper()}best = {value:.5g}; Sets / X–Z → Save to keep it", 7000)

    def _plane_changed(self):
        theta = self.plane_normal.value()
        c, s_ = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        self.plane_hint.setText(f"d = {c:+.3f}·(X−Xbest) {s_:+.3f}·(Z−Zbest).  d > 0 and d < 0 are opposite sides "
                                "of the plane. Plot X = 'Offset within object plane': it should stay near 0 if θ is "
                                "right for a straight scan.")
        self._draft_reference_cache = None
        if self.position_metric.currentData() in ("plane_signed", "plane_offset"):
            self.update_plots()
        self.refresh_particle()

    def estimate_plane_normal(self):
        records = ([self.catalog._by_id[p] for p in self.selected_ids] if self.selected_ids and self.catalog
                   else self._records())
        angle = scan_direction_deg([r.stage_x for r in records], [r.stage_y for r in records])
        if angle is None:
            QMessageBox.information(self, "Scan direction", "Need particles at two or more distinct stage positions "
                                    "(import the Stage CSV first).")
            return
        old = self.plane_normal.value()
        # Keep the user's sign convention: choose the direction closest to the current θ.
        if abs(((angle - old + 180) % 360) - 180) > 90:
            angle = ((angle + 360) % 360) - 180
        self.plane_normal.setValue(angle)
        self._toast(f"θ = {angle:.2f}° from the scan direction of "
                    f"{'the selection' if self.selected_ids else 'all particles'}. Use Flip side to swap signs.")

    def snap_reference_to_median_peak(self):
        axis = self.position_metric.currentData()
        if axis not in ("stage_x", "stage_z") or axis not in self._median_peak:
            QMessageBox.information(self, "Median focus peak",
                                    "Plot Focus against Stage X (or Stage Z) with the median trace switched on, "
                                    "then try again.")
            return
        (self.best_x_set if axis == "stage_x" else self.best_z_set).setValue(self._median_peak[axis])
        self._toast(f"Draft {axis[-1].upper()}best set to the median focus peak "
                    f"{self._median_peak[axis]:.5g}. Save it with the set.")

    def saved_set_clicked(self):
        name = self._selected_saved_set_name()
        if name is None:
            return
        self.active_reference_set = name
        spec = self.saved_sets[name]
        self.set_name_field.setText(name)
        ref = spec.get("best_focus")
        if ref is not None:
            self.best_x_set.blockSignals(True); self.best_z_set.blockSignals(True)
            self.best_x_set.setValue(ref["x_best"]); self.best_z_set.setValue(ref["z_best"])
            self.best_x_set.blockSignals(False); self.best_z_set.blockSignals(False)
        self._refresh_saved_sets()
        self.update_plots()
        self.save_project_if_open()

    def commit_working_set(self):
        """Atomically save particle IDs and typed X/Z as one named snapshot."""
        if not self.selected_ids:
            QMessageBox.information(self, "Nothing selected", "Select particles first (e.g. Review → Accepted → "
                                                              "Select).")
            return
        name = self.set_name_field.text().strip()
        if not name:
            name, ok = QInputDialog.getText(self, "Name calibration set", "Set name:")
            if not ok or not name.strip():
                return
            name = name.strip()
            self.set_name_field.setText(name)
        if name in self.saved_sets and self.saved_sets[name].get("description", "").startswith("Review batch"):
            QMessageBox.warning(self, "Name in use", "That name belongs to a flag-history batch. Choose another.")
            return
        if name in self.saved_sets and QMessageBox.question(
                self, "Replace calibration set?",
                f"Replace '{name}' with the current {len(self.selected_ids):,} particles and X/Z values?"
        ) != QMessageBox.Yes:
            return
        old = copy.deepcopy(self.saved_sets.get(name))
        self._push_undo(f"Save set '{name}'", sets=True)
        entry = make_saved_set(self.selected_ids, description="Calibration working set")
        entry["best_focus"] = make_reference(self.best_x_set.value(), self.best_z_set.value())
        self.saved_sets[name] = entry
        self.active_reference_set = name
        self._refresh_saved_sets()
        if self.project_file is None:
            self.save_project()
            if self.project_file is None:
                if old is None:
                    del self.saved_sets[name]
                else:
                    self.saved_sets[name] = old
                self.active_reference_set = None
                self.undo_stack.pop()
                self._refresh_saved_sets()
                return
        else:
            self.save_project()
        self._after_change()
        self._toast(f"Saved set '{name}' with {len(self.selected_ids):,} particles and best X/Z.")

    def save_named_set(self):
        self.commit_working_set()

    def commit_current_particle(self):
        record = self.current_record()
        if record is None:
            return
        name = self.set_name_field.text().strip() or self._selected_saved_set_name()
        if not name:
            name, ok = QInputDialog.getText(self, "Add particle to set", "Calibration set name:")
            if not ok or not name.strip():
                return
            name = name.strip()
        old_entry = copy.deepcopy(self.saved_sets.get(name))
        try:
            add_ids_to_saved_set(self.saved_sets, name, [record.particle_id],
                                 make_reference(self.best_x_set.value(), self.best_z_set.value()))
        except ValueError as exc:
            QMessageBox.warning(self, "Add particle", str(exc))
            return
        self.active_reference_set = name
        self.set_name_field.setText(name)
        self._refresh_saved_sets()
        self.save_project()
        if self.project_file is None:
            if old_entry is None:
                self.saved_sets.pop(name, None)
            else:
                self.saved_sets[name] = old_entry
            self.active_reference_set = None
            self._refresh_saved_sets()
            return
        self._after_change()
        self._toast(f"Added ROI {record.roi_index} to '{name}'")

    def remove_current_from_committed_set(self):
        record = self.current_record()
        name = self._selected_saved_set_name() or self.active_reference_set
        if record is None or name not in self.saved_sets:
            QMessageBox.information(self, "Remove particle", "Choose a calibration set in the list first.")
            return
        if record.particle_id not in self.saved_sets[name]["particle_ids"]:
            self.statusBar().showMessage("This particle is not in the chosen set", 5000)
            return
        self._push_undo(f"Remove ROI from '{name}'", sets=True)
        remove_ids_from_saved_set(self.saved_sets, name, [record.particle_id])
        self._refresh_saved_sets()
        self.save_project_if_open()
        self._after_change()
        self._toast(f"Removed ROI {record.roi_index} from '{name}'. X/Z unchanged.")

    def update_saved_set_reference(self):
        name = self._selected_saved_set_name()
        if not name:
            QMessageBox.information(self, "Choose a set", "Select a calibration set in the list first.")
            return
        self._push_undo(f"Update X/Z of '{name}'", sets=True)
        self.saved_sets[name]["best_focus"] = make_reference(self.best_x_set.value(), self.best_z_set.value())
        self.active_reference_set = name
        self._refresh_saved_sets()
        self.update_plots()
        self.save_project_if_open()
        self._toast(f"Updated best X/Z for '{name}' (its particles are unchanged).")

    def clear_saved_set_reference(self):
        name = self._selected_saved_set_name()
        if not name or "best_focus" not in self.saved_sets[name]:
            return
        if QMessageBox.question(self, "Clear best-focus reference?",
                                f"Remove saved X/Z for '{name}'? Its particles are kept.") != QMessageBox.Yes:
            return
        self._push_undo(f"Clear X/Z of '{name}'", sets=True)
        self.saved_sets[name].pop("best_focus", None)
        self._refresh_saved_sets()
        self.update_plots()
        self.save_project_if_open()

    def load_named_set(self):
        name = self._selected_saved_set_name()
        if name is None or not self.catalog:
            return
        valid = set(self.catalog._by_id)
        self.active_reference_set = name
        self._refresh_saved_sets()
        self.regions.clear()
        self._change_selection(set(self.saved_sets[name]["particle_ids"]) & valid,
                               f"Selected the particles of set '{name}'")

    def add_named_set(self):
        name = self._selected_saved_set_name()
        if name is None or not self.catalog:
            return
        valid = set(self.catalog._by_id)
        self._change_selection(set(self.selected_ids) | (set(self.saved_sets[name]["particle_ids"]) & valid),
                               f"Added set '{name}' to the selection")

    def delete_named_set(self):
        name = self._selected_saved_set_name()
        if name is None:
            return
        if QMessageBox.question(self, "Delete calibration set?",
                                f"Delete '{name}'? Review flags and MAT files are not changed.") != QMessageBox.Yes:
            return
        self._push_undo(f"Delete set '{name}'", sets=True)
        del self.saved_sets[name]
        if name == self.active_reference_set:
            self.active_reference_set = None
        self._refresh_saved_sets()
        self.save_project_if_open()
        self._after_change()

    def clear_named_set_flags(self):
        name = self._selected_saved_set_name()
        if name is None or not self.catalog:
            return
        ids = set(self.saved_sets[name]["particle_ids"])
        if QMessageBox.question(self, "Unflag set members?",
                                f"Return the {len(ids):,} members of '{name}' to unreviewed? "
                                "The set itself is kept.") != QMessageBox.Yes:
            return
        self._flag_ids(ids, "undecided", f"Unflag set '{name}'", batch=False)
        self._after_change()

    def remove_classified_batch(self):
        """Undo one accepted/rejected batch without changing unrelated statuses."""
        name = self._selected_batch_name()
        if name is None or not self.catalog:
            self.statusBar().showMessage("Choose a batch in Flag history first", 4000)
            return
        status = self.saved_sets[name].get("description", "").partition(": ")[2]
        ids = exclusive_batch_ids(self.saved_sets, name)
        affected = [r.particle_id for r in self.catalog.records if r.particle_id in ids and r.status == status]
        if QMessageBox.question(self, "Undo batch?",
                                f"Remove '{name}' and return {len(affected):,} particles that are still "
                                f"{STATUS_NAME.get(status, status).lower()} to unreviewed?\n\nParticles flagged "
                                "differently since, or also in another batch with the same decision, "
                                "are left alone.") != QMessageBox.Yes:
            return
        self._push_undo(f"Undo batch '{name}'", status_ids=affected, sets=True)
        self._set_flags(affected, "undecided")
        del self.saved_sets[name]
        if name == self.active_reference_set:
            self.active_reference_set = None
        self._refresh_saved_sets()
        self.autosave()
        self.save_project_if_open()
        self._after_change()
        self._toast(f"Undid '{name}': {len(affected):,} particles back to unreviewed.")

    # ------------------------------------------------- jig / surface (adv.) --
    def fit_best_focus_surface(self):
        if not self.catalog:
            return
        start = self._read_time(self.segment_start)
        end = self._read_time(self.segment_end)
        if start is None or end is None or start >= end:
            QMessageBox.warning(self, "Fit surface", "Set valid start and end times for this jig placement first.")
            return
        eligible = [r for r in self.catalog.records if start <= r.time_matlab < end
                    and r.status != "reject" and (not self.selected_ids or r.particle_id in self.selected_ids)]
        try:
            result = fit_surface(eligible, peak=self.peak_combo.currentData())
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot fit best-focus surface", str(exc))
            return
        previous = [s for s in self.stage_segments if not (s["start"] < end and start < s["end"])]
        self.stage_segments = sorted(previous + [{"start": start, "end": end, "surface": result.as_dict()}],
                                     key=lambda s: s["start"])
        self._refresh_reference_list()
        self.update_plots()
        self.save_project_if_open()
        self.viewer_tabs.setTabVisible(3, True)
        self.viewer_tabs.setCurrentWidget(self.surface_tab)
        QMessageBox.information(self, "Surface fitted",
                                f"Fitted {result.n_particles} particles.\nRMS focus residual: "
                                f"{result.rms_residual:.4g}\nThe quadratic peak assumption is exploratory.")

    def set_best_focus_segment(self):
        start = self._read_time(self.segment_start)
        end = self._read_time(self.segment_end)
        if start is None or end is None or not start < end:
            QMessageBox.warning(self, "Jig interval", "Enter valid start and end times for this jig placement.")
            return
        xbest, zbest = self.best_x.value(), self.best_z.value()
        if any(seg["start"] < end and start < seg["end"] for seg in self.stage_segments):
            QMessageBox.warning(self, "Overlapping jig placements", "Intervals must not overlap.")
            return
        self.stage_segments.append({"start": start, "end": end, "x_best": xbest, "z_best": zbest,
                                    "best_radius": math.hypot(xbest, zbest)})
        self.stage_segments.sort(key=lambda seg: seg["start"])
        self._refresh_reference_list()
        self.update_plots()
        self.save_project_if_open()

    def _refresh_reference_list(self):
        if not hasattr(self, "reference_list"):
            return
        current = self.reference_list.currentRow()
        self.reference_list.blockSignals(True)
        self.reference_list.clear()
        for seg in self.stage_segments:
            try:
                t1, t2 = self._matlab_time_label(seg["start"]), self._matlab_time_label(seg["end"])
            except Exception:
                t1, t2 = str(seg["start"]), str(seg["end"])
            desc = "spatial surface" if "surface" in seg else \
                f"Xbest={seg.get('x_best', float('nan')):.3f}, Zbest={seg.get('z_best', float('nan')):.3f}"
            self.reference_list.addItem(f"{t1} → {t2} · {desc}")
        self.reference_list.blockSignals(False)
        if 0 <= current < self.reference_list.count():
            self.reference_list.setCurrentRow(current)

    @staticmethod
    def _matlab_time_label(value):
        return matlab_datenum_to_datetime(value).strftime("%Y-%m-%d %H:%M:%S")

    def select_reference(self, row):
        if not 0 <= row < len(self.stage_segments):
            return
        seg = self.stage_segments[row]
        self._restore_date(self.segment_start, self._matlab_time_label(seg["start"]))
        self._restore_date(self.segment_end, self._matlab_time_label(seg["end"]))
        self.best_x.setValue(seg.get("x_best", 0.0))
        self.best_z.setValue(seg.get("z_best", 0.0))

    def update_reference(self):
        row = self.reference_list.currentRow()
        if not 0 <= row < len(self.stage_segments):
            QMessageBox.information(self, "Jig reference", "Choose a saved jig placement first.")
            return
        a, b = self._read_time(self.segment_start), self._read_time(self.segment_end)
        if a is None or b is None or a >= b:
            QMessageBox.warning(self, "Jig reference", "Specify a valid start and end date/time.")
            return
        if any(i != row and seg["start"] < b and a < seg["end"] for i, seg in enumerate(self.stage_segments)):
            QMessageBox.warning(self, "Jig reference", "The edited interval overlaps another jig placement.")
            return
        x, z = self.best_x.value(), self.best_z.value()
        self.stage_segments[row] = {"start": a, "end": b, "x_best": x, "z_best": z, "best_radius": math.hypot(x, z)}
        self.stage_segments.sort(key=lambda seg: seg["start"])
        self._refresh_reference_list()
        self.update_plots()
        self.save_project_if_open()

    def delete_reference(self):
        row = self.reference_list.currentRow()
        if 0 <= row < len(self.stage_segments):
            self.stage_segments.pop(row)
            self._refresh_reference_list()
            self.update_plots()
            self.save_project_if_open()

    # ------------------------------------------------------- files / project --
    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if any(url.isLocalFile() and (Path(url.toLocalFile()).is_dir() or
                                      Path(url.toLocalFile()).suffix.lower() == ".mat") for url in urls):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                self.add_directory(str(path), refresh=False)
            elif path.suffix.lower() == ".mat":
                self.import_file(path, refresh=False)
        self._view_locked = {"camera": False, "metric": False}
        self.apply_filters()
        event.acceptProposedAction()

    def save_project_if_open(self):
        if self.project_file:
            self.save_project()

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Processed CPI calibration directory")
        if directory:
            self.load_directory(directory, position_file=self.position_file)

    def choose_add_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Add processed CPI dataset")
        if directory:
            self.add_directory(directory)

    def _reset_data(self, root):
        self.catalog = ParticleCatalog(root, [])
        self.store = ParticleStore(root)
        self.sources = []
        self.saved_sets = {}
        self.active_reference_set = None
        self.selected_ids.clear()
        self.regions.clear()
        self.undo_stack.clear()
        self.current_id = None
        self._thumbs.clear()
        self._refresh_saved_sets()

    def create_project(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "New calibration project", "calibration_project.json",
                                                  "Project JSON (*.json)")
        if filename:
            self._reset_data(Path(filename).parent)
            self.project_file = Path(filename).resolve()
            self.selection_file = self.project_file.with_name(self.project_file.stem + "_flags.csv")
            self.save_project()
            self.apply_filters()

    def open_project_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open calibration project", "", "Project JSON (*.json)")
        if filename:
            self.open_project(filename)

    def open_project(self, filename: str) -> None:
        config = json.loads(Path(filename).read_text(encoding="utf-8"))
        self._restoring = True
        self.project_file = Path(filename).resolve()
        self._reset_data(self.project_file.parent)
        self.saved_sets = normalise_sets(config.get("named_sets", {}))
        requested = config.get("active_reference_set")
        self.active_reference_set = requested if requested in self.saved_sets else None
        self._refresh_saved_sets()
        self.selection_file = self.project_file.with_name(config.get("flags_file", self.project_file.stem + "_flags.csv"))
        for location in config.get("sources", []):
            source = Path(location)
            if not source.is_absolute():
                source = (self.project_file.parent / source).resolve()
            if source.is_file():
                self.import_file(source, refresh=False)
        self.catalog.apply_selection_file(self.selection_file)
        added = self.labels.merge(config.get("labels", []))
        if added:
            self._save_label_library()
            self._toast(f"Added labels from the project to your library: {', '.join(added)}")
        apply_label_dimensions(self.catalog.records, self.labels)
        for key, pair in config.get("filters", {}).items():
            if key in self.filter_fields and len(pair) == 2:
                self.filter_fields[key][0].setText(pair[0])
                self.filter_fields[key][1].setText(pair[1])
        self._restore_date(self.time_low, config.get("time_low", ""))
        self._restore_date(self.time_high, config.get("time_high", ""))
        self.time_filter_checkbox.setChecked(config.get("use_time_filter", False))
        self.image_mode.setCurrentIndex(max(0, self.image_mode.findData(config.get("image_mode", "crop"))))
        scope = {"filtered": "auto", "working": "selection"}.get(config.get("view_scope", "auto"),
                                                                  config.get("view_scope", "auto"))
        self.view_selector.blockSignals(True)
        self.view_selector.setCurrentIndex(max(0, self.view_selector.findData(scope)))
        self.view_selector.blockSignals(False)
        for combo, key, default in ((self.plot_metric, "plot_y", "focus"), (self.plot_metric2, "plot_y2", "length"),
                                    (self.position_metric, "plot_x", "time"),
                                    (self.pick_scope, "pick_scope", "undecided")):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(config.get(key, default))))
            combo.blockSignals(False)
        colour = config.get("colour_by", "status")
        self.colour_by.setCurrentIndex(max(0, self.colour_by.findData("status" if colour == "workflow" else colour)))
        shown = config.get("show_status")
        if isinstance(shown, dict):
            for status, box in self.show_status.items():
                box.blockSignals(True); box.setChecked(bool(shown.get(status, True))); box.blockSignals(False)
        self.selection_only_cb.setChecked(bool(config.get("selection_only", False)))
        self.median_cb.setChecked(bool(config.get("show_median", True)))
        self.set_box_mode(config.get("box_mode", "new") if config.get("box_mode") in dict(
            (k, 1) for k, _, _ in BOX_MODES) else "new")
        self.plane_normal.setValue(float(config.get("plane_normal_deg", 45.0)))
        self.stage_segments = config.get("stage_segments", [])
        self._refresh_reference_list()
        self.regions = [tuple(r) for r in config.get("regions", [])]
        self.selected_ids = set(config.get("selected_ids", [])) & set(self.catalog._by_id)
        self.position_series = config.get("position_series", "beads")
        self.position_time_basis = config.get("position_time_basis", "roi")
        self.position_file = config.get("position_file")
        if self.position_file:
            pos = Path(self.position_file)
            if not pos.is_absolute():
                pos = (self.project_file.parent / pos).resolve()
            self.position_file = str(pos)
            if pos.exists():
                self._apply_stage_positions()
        self._restoring = False
        self._view_locked = {"camera": False, "metric": False}
        self.apply_filters()
        self._toast(f"Opened project {self.project_file.name}")

    def save_project(self) -> None:
        if self.project_file is None:
            filename, _ = QFileDialog.getSaveFileName(self, "Save calibration project", "calibration_project.json",
                                                      "Project JSON (*.json)")
            if not filename:
                return
            self.project_file = Path(filename).resolve()
            self.selection_file = self.project_file.with_name(self.project_file.stem + "_flags.csv")
        self.autosave()

        def relative(value):
            if not value:
                return None
            try:
                return os.path.relpath(str(value), str(self.project_file.parent))
            except ValueError:
                return str(value)
        config = {"format_version": 3, "sources": [relative(s) for s in self.sources],
                  "flags_file": self.selection_file.name, "position_file": relative(self.position_file),
                  "filters": {key: [pair[0].text(), pair[1].text()] for key, pair in self.filter_fields.items()},
                  "time_low": self._date_text(self.time_low), "time_high": self._date_text(self.time_high),
                  "use_time_filter": self.time_filter_checkbox.isChecked(), "image_mode": self.image_mode.currentData(),
                  "selected_ids": sorted(self.selected_ids),
                  "named_sets": self.saved_sets, "active_reference_set": self.active_reference_set,
                  "view_scope": self.view_selector.currentData(),
                  "plot_x": self.position_metric.currentData(), "plot_y": self.plot_metric.currentData(),
                  "plot_y2": self.plot_metric2.currentData(), "colour_by": self.colour_by.currentData(),
                  "pick_scope": self.pick_scope.currentData(), "box_mode": self.box_mode,
                  "show_status": {s: b.isChecked() for s, b in self.show_status.items()},
                  "selection_only": self.selection_only_cb.isChecked(), "show_median": self.median_cb.isChecked(),
                  "plane_normal_deg": self.plane_normal.value(),
                  "regions": self.regions, "stage_segments": self.stage_segments,
                  # Snapshot of the labels used, so the project is self-describing on another computer.
                  "labels": [self.labels.get(n).as_json() for n in sorted(self._label_counts())
                             if self.labels.get(n) is not None],
                  "position_series": self.position_series, "position_time_basis": self.position_time_basis}
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        self._export_committed_positions()
        temp = self.project_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temp, self.project_file)
        self.statusBar().showMessage("Project saved", 3000)

    def _export_committed_positions(self):
        """Atomically export signed X/Z offsets for every committed set member."""
        if self.project_file is None or self.catalog is None:
            return
        path = self.project_file.with_name(self.project_file.stem + "_relative_positions.csv")
        rows = materialized_reference_rows(self.saved_sets, self.catalog.records, self.plane_normal.value())
        if not rows:
            if path.exists():
                path.unlink()
            return
        columns = ("set_name", "particle_id", "x_stage", "z_stage", "x_best", "z_best", "x_minus_xbest",
                   "z_minus_zbest", "z_p_radial", "z_p_from_offsets", "plane_normal_deg", "d_signed_plane",
                   "d_in_plane")
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)

    def choose_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "Add processed MAT files", "", "MAT files (*.mat)")
        self._import_errors = []
        for filename in names:
            self.import_file(Path(filename), refresh=False)
        self._report_import_errors()
        if names:
            self._view_locked = {"camera": False, "metric": False}
            self.apply_filters()

    def import_file(self, file: Path, refresh: bool = True) -> None:
        path = Path(file).resolve()
        if str(path) in self.sources or path.name in ("timeseries.mat", "full_backgrounds.mat"):
            return
        if self.catalog is None:
            self.catalog = ParticleCatalog(path.parent, [])
            self.store = ParticleStore(path.parent)
        self.statusBar().showMessage(f"Indexing {path.name}…")
        QApplication.processEvents()
        try:
            records = _scan_via_subprocess(path, path.parent)
        except Exception as exc:
            if self._import_errors is not None:
                self._import_errors.append(f"{path.name}: {exc}")
            else:
                QMessageBox.warning(self, "MAT file", f"Unable to scan {path}: {exc}")
            return
        for rec in records:
            rec.source_file = str(path)  # absolute file identity across folders
        self.catalog.records.extend(records)
        self.catalog.records.sort(key=lambda r: (r.time_matlab, r.source_file, r.roi_index))
        self.catalog._by_id = {r.particle_id: r for r in self.catalog.records}
        self.sources.append(str(path))
        if self.selection_file is None:
            # Without a project, flags are saved next to the first data file and
            # restored automatically the next time the same files are loaded.
            self.selection_file = path.parent / "calibration_selection.csv"
        if not self._restoring and self.selection_file.exists():
            self.catalog.apply_selection_file(self.selection_file)
            apply_label_dimensions(self.catalog.records, self.labels)
        if self.position_file:
            try:
                self._apply_stage_positions()
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(f"Stage-position import problem: {exc}", 10000)
        if refresh:
            self.apply_filters()

    def _report_import_errors(self):
        errors, self._import_errors = self._import_errors or [], None
        if errors:
            QMessageBox.warning(self, "Some MAT files were skipped",
                                f"{len(errors)} file(s) could not be read:\n\n" + "\n".join(errors[:12])
                                + ("\n…" if len(errors) > 12 else ""))

    def add_directory(self, directory: str, refresh: bool = True) -> None:
        files = sorted(Path(directory).rglob("*.mat"))
        own = self._import_errors is None
        if own:
            self._import_errors = []
        for number, file in enumerate(files, 1):
            self.statusBar().showMessage(f"Indexing {number}/{len(files)}: {file.name}")
            self.import_file(file, refresh=False)
        if own:
            self._report_import_errors()
        if refresh:
            self._view_locked = {"camera": False, "metric": False}
            self.apply_filters()

    def choose_truth_table(self) -> None:
        if not self.catalog:
            QMessageBox.information(self, "Spreadsheet", "Load a processed dataset first.")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Calibration truth spreadsheet", "",
                                                  "Spreadsheets (*.csv *.xlsx)")
        if not filename:
            return
        try:
            matched, unmatched = attach_truth(self.catalog.records, read_truth_table(filename))
            self.autosave()
            self.refresh_particle()
            summary = f"Applied spreadsheet information to {matched} particle(s)."
            if unmatched:
                summary += f"\n{len(unmatched)} unmatched row(s), first: {unmatched[:5]}"
            QMessageBox.information(self, "Spreadsheet imported", summary)
        except Exception as exc:
            QMessageBox.critical(self, "Spreadsheet import", str(exc))

    def _suggest_initial_best_reference(self):
        if self._calibration_set_names() or not self.catalog or not self.catalog.records:
            return
        if self.best_x_set.value() != 0 or self.best_z_set.value() != 0:
            return
        xs = [r.stage_x for r in self.catalog.records if math.isfinite(r.stage_x)]
        zs = [r.stage_y for r in self.catalog.records if math.isfinite(r.stage_y)]
        if xs:
            self.best_x_set.setValue(float(np.median(xs)))
        if zs:
            self.best_z_set.setValue(float(np.median(zs)))

    def _apply_stage_positions(self) -> int:
        if not self.position_file or self.catalog is None or not self.catalog.records:
            return 0
        log = PositionLog.from_text(self.position_file, self.position_series, self.position_time_basis)
        self.catalog.attach_position_log(log)
        return sum(math.isfinite(r.stage_x) and math.isfinite(r.stage_y) for r in self.catalog.records)

    def _use_stage_axis_after_import(self, valid_count: int) -> None:
        if valid_count and self.position_metric.currentData() == "time":
            self.position_metric.setCurrentIndex(self.position_metric.findData("stage_x"))

    def choose_position_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import stage positions (CSV)", "",
            "Stage position CSV (*.csv);;Legacy position logs (*.txt *.tsv);;All files (*)")
        if not filename:
            return
        if Path(filename).suffix.lower() in (".csv", ".tsv"):
            series, ok = QInputDialog.getItem(self, "Position series", "Calibration type", ["beads", "analogue"], 0, False)
            if not ok:
                return
            basis, ok = QInputDialog.getItem(self, "Time basis", "Timestamp column", ["roi", "full_frame", "stage"], 0, False)
            if not ok:
                return
            self.position_series, self.position_time_basis = series, basis
        self.load_position_file(filename)

    def load_position_file(self, filename):
        self.position_file = filename
        if not self.catalog:
            return
        try:
            matched = self._apply_stage_positions()
        except Exception as exc:
            QMessageBox.critical(self, "Position log", str(exc))
            return
        self._suggest_initial_best_reference()
        self._view_locked["metric"] = False
        self._use_stage_axis_after_import(matched)
        self._refresh_views()
        self._toast(f"Stage CSV loaded: {matched:,} / {len(self.catalog.records):,} particles have X/Z positions.",
                    12000)
        if matched == 0:
            QMessageBox.warning(self, "No stage matches",
                                "The CSV was read, but no particles matched its times. Check the calibration "
                                "series (beads/analogue), time basis and clock conventions.")

    def load_directory(self, directory: str, position_file: Optional[str] = None) -> None:
        self._reset_data(directory)
        self.additional_roots.clear()
        self.project_file = None
        self.selection_file = Path(directory).resolve() / "calibration_selection.csv"
        self.add_directory(directory, refresh=False)
        if position_file:
            self.position_file = position_file
            self._use_stage_axis_after_import(self._apply_stage_positions())
            self._suggest_initial_best_reference()
        self._view_locked = {"camera": False, "metric": False}
        self.apply_filters()

    # ----------------------------------------------------------- particle Z --
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
        self.z_slider.setValue(int(round((value - self.z_min_um) / self.z_step_um)))
        self._updating_z = False
        self._set_current_z(value)

    def _set_current_z(self, value: float) -> None:
        record = self.current_record()
        if record is not None:
            record.object_plane_um = float(value)

    def set_instrument(self) -> None:
        if not self.catalog:
            return
        for record in self.catalog.records:
            record.instrument = self.instrument_combo.currentText()
        self.autosave()
        self.refresh_particle()

    def autosave(self) -> None:
        if self.catalog and self.selection_file:
            try:
                self.catalog.save_selection_file(self.selection_file)
            except OSError as exc:
                self.save_indicator.setText(f"⚠ Flags NOT saved: {exc}")
                self.save_indicator.setStyleSheet("color:#c92a2a; font-weight:700; padding: 0 8px;")
                return
            self.save_indicator.setText(f"✓ Flags saved {datetime.now():%H:%M:%S} · {self.selection_file.name}")
            self.save_indicator.setToolTip(str(self.selection_file))
            self.save_indicator.setStyleSheet("color:#2b7a3d; padding: 0 8px;")

    def save_as(self) -> None:
        if not self.catalog:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save review flags", str(self.selection_file or
                                                  "calibration_selection.csv"), "CSV files (*.csv)")
        if filename:
            self.selection_file = Path(filename)
            self.autosave()

    # ---------------------------------------------------------------- filters --
    def _read_float(self, widget: QLineEdit) -> Optional[float]:
        text = widget.text().strip()
        return float(text) if text else None

    @staticmethod
    def _date_text(widget):
        return widget.dateTime().toString("yyyy-MM-dd HH:mm:ss")

    @staticmethod
    def _restore_date(widget, text):
        if text:
            parsed = QDateTime.fromString(text, "yyyy-MM-dd HH:mm:ss")
            if parsed.isValid():
                widget.setDateTime(parsed)

    def _read_time(self, widget) -> Optional[float]:
        text = self._date_text(widget) if isinstance(widget, QDateTimeEdit) else widget.text().strip()
        if not text:
            return None
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime_to_matlab_datenum(datetime.strptime(text, pattern))
            except ValueError:
                pass
        raise ValueError("Time must look like YYYY-MM-DD HH:MM:SS")

    def fill_filter_ranges(self):
        if not self.catalog or not self.catalog.records:
            return
        for key, pair in self.filter_fields.items():
            values = []
            for r in self.catalog.records:
                val = {"length": r.length_um, "width": r.width_um, "focus": r.focus,
                       "x": r.image_x_px, "y": r.image_y_px}.get(key)
                if key == "ratio":
                    val = self._metric_value(r, "ratio")
                if val is not None and math.isfinite(val):
                    values.append(val)
            if values:
                pair[0].setPlaceholderText(f"min {min(values):.4g}")
                pair[1].setPlaceholderText(f"max {max(values):.4g}")
        self.statusBar().showMessage("Data ranges shown as hints; filters unchanged", 6000)

    def apply_filters(self) -> None:
        if not self.catalog:
            self._refresh_views()
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
                x_min=values["x_min"], x_max=values["x_max"], y_min=values["y_min"], y_max=values["y_max"],
                time_min=self._read_time(self.time_low) if self.time_filter_checkbox.isChecked() else None,
                time_max=self._read_time(self.time_high) if self.time_filter_checkbox.isChecked() else None,
                ratio_min=values["ratio_min"], ratio_max=values["ratio_max"])
        except ValueError as exc:
            QMessageBox.warning(self, "Filters", str(exc))
            return
        self._last_filter_spec = spec
        mode = self.image_mode.currentData()
        records = self.catalog.records
        self._base_indices = [i for i in self.catalog.filtered_indices(spec)
                              if mode == "all" or records[i].is_full_frame == (mode == "full")]
        self._rebuild_browse()
        self._refresh_views()
        if self.selection_file is not None and self.save_indicator.text() == "No data loaded":
            self.save_indicator.setText(f"Flags file: {self.selection_file.name}")
            self.save_indicator.setToolTip(str(self.selection_file))

    def clear_filters(self) -> None:
        for low, high in self.filter_fields.values():
            low.clear()
            high.clear()
        self.time_filter_checkbox.setChecked(False)
        self.apply_filters()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        self._hide_hover()
        if self.project_file is None and self.saved_sets:
            choice = QMessageBox.question(
                self, "Save project?",
                "Review flags and labels are already saved in the flags CSV. Save a project too, so your "
                "calibration sets, X/Z references and flag history are kept?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if choice == QMessageBox.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.Yes:
                self.save_project()
                if self.project_file is None:
                    event.ignore()
                    return
        if self.project_file is not None:
            self.save_project()
        else:
            self.autosave()
        event.accept()


STYLE_SHEET = """
QMainWindow, QWidget, QScrollArea, QSplitter, QTabWidget, QTabWidget::pane { background: #f4f7fb; color: #192b40; }
QGroupBox { background: #ffffff; color: #152b43; border: 1px solid #cbd7e5; border-radius: 7px;
    margin-top: 12px; padding: 8px 5px 5px; font-weight: 650; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; background: #ffffff; color: #152b43; }
QLabel, QCheckBox, QRadioButton { color: #192b40; background: transparent; }
QLabel#hint { color: #3d566e; font-size: 11px; }
QLabel#panelTitle { font-weight: 800; color: #12345a; font-size: 12px; }
QPushButton { color: #10283f; background: #e6eff9; padding: 4px 9px; border: 1px solid #a9bfd7;
    border-radius: 5px; font-weight: 600; min-height: 20px; }
QPushButton:hover { background: #d1e4f7; border-color: #5887b2; }
QPushButton:pressed { background: #b5d0eb; }
QPushButton:disabled { color: #9aa9b8; background: #eef2f6; border-color: #d3dce6; }
QPushButton[role="primary"] { color: #ffffff; background: #195a91; border: 1px solid #124878; }
QPushButton[role="primary"]:hover { background: #0f477c; }
QPushButton[role="accept"] { color: #ffffff; background: #2b9a48; border: 1px solid #1e7a37; }
QPushButton[role="accept"]:hover { background: #23853d; }
QPushButton[role="reject"] { color: #ffffff; background: #d63a3a; border: 1px solid #b02a2a; }
QPushButton[role="reject"]:hover { background: #bd3030; }
QPushButton[role="neutral"] { color: #2f3e4d; background: #eceff3; border: 1px solid #a8b4c0; }
QPushButton[role="accept"]:disabled, QPushButton[role="reject"]:disabled, QPushButton[role="primary"]:disabled
    { color: #eef2f6; background: #c3ccd5; border-color: #c3ccd5; }
QToolButton { color: #10283f; background: #ffffff; border: 1px solid #b4c4d6; border-radius: 5px;
    padding: 3px 8px; font-weight: 600; }
QToolButton:hover { background: #e3eef9; }
QToolButton:checked { color: #ffffff; background: #1c64d6; border-color: #1450ad; }
QToolButton[boxmode="remove"]:checked { background: #e8590c; border-color: #b84607; }
QToolButton[boxmode="reject"]:checked { background: #e03b3b; border-color: #b02a2a; }
QToolButton[boxmode="keep"]:checked { background: #6741d9; border-color: #5130b0; }
QFrame#selectionBar { background: #e7f0fd; border: 1px solid #9fbfeb; border-radius: 7px; }
QFrame#browseBar { background: #ffffff; border: 1px solid #cbd7e5; border-radius: 6px; }
QLineEdit, QComboBox, QDoubleSpinBox, QDateTimeEdit, QListWidget { color: #13283e; background: #ffffff;
    border: 1px solid #b6c8db; border-radius: 4px; padding: 2px 3px; selection-background-color: #1c68a1;
    selection-color: white; }
QComboBox QAbstractItemView { color: #13283e; background: #ffffff; selection-background-color: #d2e5f8;
    selection-color: #13283e; }
QListWidget#gallery { background: #f9fbfd; }
QListWidget#gallery::item { color: #33475b; border-radius: 6px; padding: 2px; }
QListWidget#gallery::item:selected { background: #ffd66b; color: #3d2a00; border: 2px solid #f08c00; }
QListWidget#gallery::item:hover { background: #eef4fb; }
QTabBar::tab { color: #163047; background: #e1eaf4; padding: 5px 11px; border: 1px solid #b8c8da;
    border-bottom: none; font-weight: 600; }
QTabBar::tab:selected { color: #ffffff; background: #245d8c; }
QMenuBar, QMenu, QStatusBar { color: #172b40; background: #f4f7fb; }
QMenu::item:selected { background: #dcebf8; color: #122a43; }
QScrollArea { border: 0; }
QToolTip { color: #ffffff; background: #263c55; border: 1px solid #6d8aa6; }
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Select and review CPI calibration particles")
    parser.add_argument("directory", nargs="?", help="directory containing processed MAT files")
    parser.add_argument("--position-file", help="optional stage-position CSV or legacy log")
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
