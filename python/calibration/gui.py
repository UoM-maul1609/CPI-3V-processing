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
import contextlib
import copy
import csv
import json
import math
import os
import shutil
import sys
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as mpl_path_effects
from PySide6.QtCore import Qt, QItemSelectionModel, QObject, QDateTime, QRect, QSize, QTimer, QPoint, QPointF, QThread, Signal, QSettings
from PySide6.QtGui import (QAction, QColor, QCursor, QFont, QIcon, QImage, QKeySequence,
                           QPainter, QPen, QPixmap, QShortcut, QPolygonF)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDateTimeEdit,
    QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QPlainTextEdit, QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QStatusBar, QTabWidget,
    QToolButton, QVBoxLayout, QWidget, QWidgetAction,
)

try:
    from .plot_utils import data_limits, median_trace, trace_extremum
    from .selection_state import (apply_box, normalise_sets, make_saved_set, exclusive_batch_ids,
                                  expand_project_config, compact_project_sets, compact_ids, compact_keys,
                                  add_ids_to_saved_set, remove_ids_from_saved_set)
    from .reference_sets import (make_reference, focus_peak_reference, focus_peak_on_plane, session_labels, nearest_group_reference, reference_steps, effective_reference, validate_reference, signed_plane_distance, in_plane_offset, scan_direction_deg, build_reference_index, calibrated_radial_position,
                                 corrected_stage_radius, relative_stage_position,
                                 materialized_reference_rows, committed_particle_ids)
    from .focus_surface import fit_surface, predict_best_radius
    from .metadata import read_truth_table, attach_truth
    from .labels import LabelLibrary, ParticleLabel, apply_label_dimensions, LABEL_KINDS
    from .reference_library import cpi_library_path, copy_reference_library
    from .combine import combine_projects, project_instrument, load_projects_for_fit
    from . import sizing, edge_preview, batch_netcdf
    from .time_windows import window_bounds, make_window, suggest_windows, in_window, overlapping
    from .data import (FilterSpec, CPI_DISPLAY, particle_key, ParticleCatalog, ParticleRecord, ParticleStore, PositionLog, stage_burst_check,
                       datetime_to_matlab_datenum, matlab_datenum_to_datetime, _scan_via_subprocess)
except ImportError:  # Support ``python gui.py`` from this directory.
    from plot_utils import data_limits, median_trace, trace_extremum
    from selection_state import (apply_box, normalise_sets, make_saved_set, exclusive_batch_ids,
                                 expand_project_config, compact_project_sets, compact_ids, compact_keys,
                                 add_ids_to_saved_set, remove_ids_from_saved_set)
    from reference_sets import (make_reference, focus_peak_reference, focus_peak_on_plane, session_labels, nearest_group_reference, reference_steps, effective_reference, validate_reference, signed_plane_distance, in_plane_offset, scan_direction_deg, build_reference_index, calibrated_radial_position,
                                corrected_stage_radius, relative_stage_position,
                                materialized_reference_rows, committed_particle_ids)
    from focus_surface import fit_surface, predict_best_radius
    from metadata import read_truth_table, attach_truth
    from labels import LabelLibrary, ParticleLabel, apply_label_dimensions, LABEL_KINDS
    from reference_library import cpi_library_path, copy_reference_library
    from combine import combine_projects, project_instrument, load_projects_for_fit
    import sizing
    import edge_preview
    import batch_netcdf
    from time_windows import window_bounds, make_window, suggest_windows, in_window, overlapping
    from data import (FilterSpec, CPI_DISPLAY, particle_key, ParticleCatalog, ParticleRecord, ParticleStore, PositionLog, stage_burst_check,
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

METRICS = (("focus", "Focus"), ("focus_norm", "Focus ÷ local illumination"), ("length", "Length [µm]"),
           ("width", "Width [µm]"), ("ratio", "Measured / actual length"), ("roundness", "Roundness"),
           ("illum", "Local illumination (ROI edge)"), ("focus_T", "Focus on transmittance (new edge method)"),
           ("deq", "Equivalent-area diameter [µm] (new edge method)"),
           ("shadow_deq", "Shadow-circle diameter [µm] (additional processing measure)"),
           ("ratio_shadow", "Shadow-circle diameter / true diameter"),
           ("ratio_deq", "Equivalent-area diameter / actual length"),
           ("cutoff", "Cut off by the ROI edge [% of outline]"),
           ("cpv_focus", "CPIview focus (processed with --cpiview)"),
           ("cpv_focus_sd", "CPIview focus spread round the boundary (SD)"),
           ("cpv_fwhm", "Edge width: FWHM of the gradient [px] (CPIview)"),
           ("cpv_distinct", "Distinct edge: fraction of boundary with a strong gradient (CPIview)"),
           ("cpv_cutoff", "Cut off by the ROI edge [% of boundary] (CPIview)"))
POSITIONS = (("time", "Acquisition time"), ("stage_x", "Stage X"), ("stage_z", "Stage Z"),
             ("plane_signed", "Signed distance to object plane  d = n·(ΔX, ΔZ)"),
             ("plane_zd", "Normalised distance  Zd = λd / (L/2)²  (labelled particles; L = actual size)"),
             ("stage_x_relative", "X − Xbest"), ("stage_z_relative", "Z − Zbest"),
             ("plane_offset", "Offset within object plane (θ check)"),
             ("stage_radius", "Stage radius √(X² + Z²)"),
             ("relative", "Zp radial difference (best − measured radius) [legacy]"),
             ("stage_offset_radius", "Zp = √(ΔX² + ΔZ²) [unsigned]"),
             ("illum", "Local illumination (ROI edge brightness)"))
BOX_MODES = (
    ("new", "▭ New", "Plain drag: replace the selection with the particles inside the box"),
    ("add", "＋ Add", "Add the particles inside the box (or hold Shift while dragging)"),
    ("remove", "− Remove", "Leave particles out of THIS batch: drop them from the selection without flagging "
                           "them — they stay unreviewed for later (or hold Ctrl/⌘ while dragging)"),
    ("keep", "∩ Keep only", "Keep only selected particles that are inside the box (or hold Alt/⌥)"),
    ("reject", "✕ Reject", "Not usable — debris, overlaps, artefacts: flag particles inside the box REJECTED "
                           "(red). With a selection, only selected particles inside the box are rejected"),
)
PICK_SCOPES = (("undecided", "Unreviewed only"), ("unrejected", "Unreviewed + accepted"),
               ("keep", "Accepted only"), ("reject", "Rejected only"), ("any", "Everything"))
BROWSE_SCOPES = (("auto", "Selection, else current batch"), ("all", "All particles"),
                 ("selection", "Selection only"), ("undecided", "Unreviewed"),
                 ("keep", "Accepted"), ("reject", "Rejected"))
RELATIVE_AXES = ("stage_x_relative", "stage_z_relative", "stage_offset_radius", "plane_signed", "plane_offset",
                 "plane_zd")
WAVELENGTH_UM = 0.85   # CPI laser, for the normalised (Fresnel) distance
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

    def show_particle(self, image, boundary, title, show_boundary=True, frame_colour=None,
                      comparison_boundary=None, stored_colour="#ffd000", trial_colour="#00d4ed",
                      comparison_label="Trial overlay (temporary)"):
        self.axes.clear()
        self.axes.set_axis_on()
        # ROI arrays are stored [x, y] (camera X down the rows).  Shown as the camera sees them --
        # X across, Y down -- the same way round as the Camera X-Y plot.
        self.axes.imshow(np.asarray(image).T, cmap="gray", origin="upper", interpolation="nearest", aspect="equal")
        if show_boundary and boundary is not None:
            finite = np.isfinite(boundary).all(axis=1)
            b = boundary[finite]
            if len(b):
                # Stored boundary vertices are (array row, array column) = (x, y).
                self.axes.plot(b[:, 0], b[:, 1], linewidth=2, color=stored_colour,
                               path_effects=[mpl_path_effects.Stroke(linewidth=4, foreground="#000000"),
                                             mpl_path_effects.Normal()], label="Saved boundary")
        if comparison_boundary is not None:
            b = np.asarray(comparison_boundary, dtype=float)
            if b.ndim == 2 and b.shape[1] == 2:
                b = b[np.isfinite(b).all(axis=1)]
                if len(b):
                    self.axes.plot(b[:, 0], b[:, 1], linewidth=2, color=trial_colour,
                                   path_effects=[mpl_path_effects.Stroke(linewidth=4, foreground="#000000"),
                                                 mpl_path_effects.Normal()], label=comparison_label)
                    self.axes.legend(loc="upper right", fontsize=8)
        self.axes.set_title(title, color=frame_colour or "#192b40", fontweight="bold")
        for spine in self.axes.spines.values():
            spine.set_edgecolor(frame_colour or "#8a99a8")
            spine.set_linewidth(4 if frame_colour else 1)
        self.axes.set_xlabel("ROI camera x [px]")
        self.axes.set_ylabel("ROI camera y [px]")
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

    def cancel_drag(self):
        """Abandon a box (or best-focus line) being drawn. Returns True if one was."""
        if self._drag is None and self._reference is None:
            return False
        self._drag = None
        self._reference = None
        self.update()
        return True

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
        if event.button() == Qt.RightButton and self.cancel_drag():
            event.accept(); return  # right-click while dragging = cancel
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
        if self._drag is None and event.button() == Qt.LeftButton and self.selection_enabled:
            event.accept(); return  # the box was cancelled with Esc / right-click
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


class MatLoader(QThread):
    """Load one processed MAT file's images off the GUI thread (for hover previews)."""
    loaded = Signal(str, object)

    def __init__(self, store_root, source_file, parent=None):
        super().__init__(parent)
        self.store_root, self.source_file = Path(store_root), source_file

    def run(self):
        import scipy.io as sio
        try:
            data = sio.loadmat(str(self.store_root / self.source_file), variable_names=["ROI_N", "BG", "dat"],
                               simplify_cells=True)
        except Exception:
            data = None
        self.loaded.emit(self.source_file, data)


class EdgePreviewWorker(QThread):
    """Run one batch's real edge algorithm away from the GUI thread."""
    progress_changed = Signal(int, int)
    completed = Signal(object, object, object)

    def __init__(self, root, records, options, identity, parent=None):
        super().__init__(parent)
        self.root, self.records = Path(root), tuple(records)
        self.options, self.identity = options, identity

    def run(self):
        try:
            results, failures = edge_preview.run_preview(
                self.root, self.records, self.options,
                progress=lambda done, total: self.progress_changed.emit(done, total),
                cancelled=self.isInterruptionRequested)
            self.completed.emit(self.identity, results, failures)
        except Exception as exc:
            self.completed.emit(self.identity, {}, {"__batch__": f"{type(exc).__name__}: {exc}"})


class BatchNetCDFWorker(QThread):
    """Write a read-only NetCDF snapshot; supports progress and cancellation."""
    progress_changed = Signal(int, int)
    completed = Signal(str, int, int)
    failed = Signal(str)

    def __init__(self, output, root, records, names, memberships, labels, positions, theta, parent=None):
        super().__init__(parent)
        self.output, self.root = output, root
        self.records, self.names, self.memberships = records, names, memberships
        self.labels, self.positions, self.theta = labels, positions, theta

    def run(self):
        try:
            n, missing = batch_netcdf.export_batch_netcdf(
                self.output, self.root, self.records, self.names, self.memberships,
                labels=self.labels, positions=self.positions, plane_normal_deg=self.theta,
                progress=lambda done, total: self.progress_changed.emit(done, total),
                cancelled=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.completed.emit(self.output, n, missing)
        except batch_netcdf.ExportCancelled:
            pass
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


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
        same = (label is None and False) or (label is not None and math.isfinite(label.width_um)
                and math.isfinite(label.length_um) and abs(label.width_um - label.length_um) < 1e-9)
        self.same.setChecked(same)
        self.width.setValue(label.width_um if label and math.isfinite(label.width_um) else 0)
        self.width.setEnabled(not same)
        self.clear_width = QPushButton("Reset width to unset")
        self.clear_width.clicked.connect(self._unset_width)
        self.area = QDoubleSpinBox(); self.area.setRange(0, 1e12); self.area.setDecimals(4)
        self.area.setSuffix(" µm²"); self.area.setSpecialValueText("not set")
        self.area.setValue(label.area_um2 if label and math.isfinite(label.area_um2) else 0)
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
        form.addRow("", self.clear_width)
        form.addRow("SEM / microscope projected area", self.area)
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

    def _unset_width(self):
        self.same.setChecked(False)
        self.width.setValue(0)

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
                             width_um=width, area_um2=self.area.value() or float("nan"),
                             notes=self.notes.text().strip(), colour=self.colour)


class _SaveNotifier(QObject):
    """Carries 'save finished' from the writer thread back to the UI thread."""
    done = Signal(str, str)        # (status "ok"/"error", message)


class TimeWindowDialog(QDialog):
    """Name, start/end and optional label of a saved time window."""

    def __init__(self, parent, window, labels):
        super().__init__(parent)
        self.setWindowTitle("Time window")
        form = QFormLayout(self)
        self.name = QLineEdit(window.get("name", ""))
        self.start = QDateTimeEdit(); self.end = QDateTimeEdit()
        for widget, key in ((self.start, "start"), (self.end, "end")):
            widget.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            widget.setCalendarPopup(True)
            widget.setDateTime(QDateTime.fromString(window.get(key, ""), "yyyy-MM-dd HH:mm:ss"))
        self.label = QComboBox()
        self.label.addItem("(no label)", "")
        for name in labels:
            self.label.addItem(name, name)
        self.label.setCurrentIndex(max(0, self.label.findData(window.get("label", ""))))
        self.label.setToolTip("Showing this window also picks this label in 'Accept as'.")
        form.addRow("Name", self.name)
        form.addRow("Start", self.start)
        form.addRow("End", self.end)
        form.addRow("Label", self.label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def value(self):
        return {"name": self.name.text().strip() or "Window",
                "start": self.start.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
                "end": self.end.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
                "label": self.label.currentData() or ""}


class SizingDialog(QDialog):
    """Fit the size/focus model to accepted labelled particles and save it for retrieval."""

    def __init__(self, rows, parent=None, inputs=None):
        super().__init__(parent)
        self.setWindowTitle("Fit size & focus model")
        self.inputs = list(inputs or [])
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        if avail is not None:
            self.resize(int(avail.width() * 0.92), int(avail.height() * 0.9))
        else:
            self.resize(1500, 950)
        self.rows = rows
        self.model = self.validation = self.tracking = None
        self._drawn = None
        layout = QHBoxLayout(self)
        controls = QVBoxLayout()
        labels = sorted({r.get("label") or "" for r in rows} - {""})
        self.label_box = QListWidget()
        self.label_box.setMaximumHeight(150)
        for name in labels:
            n = sum(1 for r in rows if r.get("label") == name)
            item = QListWidgetItem(f"{name}  ({n:,})")
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.label_box.addItem(item)
        form = QFormLayout()
        self.preset = QComboBox()
        # Keep established fitting/retrieval behaviour as the default.
        # Shadow integration is a new, optional calibration experiment.
        self.preset.addItem("Original size and focus fit (default)", "original")
        self.preset.addItem("Contour size + normalised focus", "contour")
        self.preset.addItem("Circle shadow size + normalised focus (experimental)", "circle")
        self.preset.addItem("Shape-corrected shadow (Dshadow / sqrt(area ratio)) + normalised focus", "area_corrected")
        self.preset.setToolTip("Original size and focus is the established default. "
                               "The circle shadow method is experimental and must be selected explicitly; "
                               "individual measurements and fit terms remain editable below.")
        self.method_help = QLabel("")
        self.method_help.setWordWrap(True)
        self.method_help.setObjectName("hint")
        form.addRow("Fitting recipe", self.preset)
        form.addRow(self.method_help)
        self.fit_p = QCheckBox("Fit exponent p (otherwise Gaussian p = 2)")
        self.fit_p.setChecked(True)
        self.focus_asymmetric = QCheckBox("Different widths for −Z and +Z")
        self.focus_asymmetric.setChecked(True)
        self.peak_by_size = QCheckBox("Peak focus increases with true diameter")
        self.peak_by_size.setChecked(True)
        self.width_by_size = QCheckBox("Focus width increases with true diameter")
        self.width_by_size.setChecked(True)
        self.validate_folds = QCheckBox("Evaluate on held-out batches (slower, recommended)")
        self.validate_folds.setChecked(True)
        self.flip = QCheckBox("Flip side (d → −d)")
        self.flip.setToolTip("Use if the + side in the workbench is the side nearer the laser in your convention.")
        self.illum_min = QDoubleSpinBox(); self.illum_min.setRange(0, 255); self.illum_min.setValue(0)
        self.illum_min.setToolTip("Ignore particles whose local illumination (ROI border) is below this; 0 = use all")
        self.max_cutoff = QDoubleSpinBox(); self.max_cutoff.setRange(0, 100); self.max_cutoff.setDecimals(1)
        self.max_cutoff.setValue(100); self.max_cutoff.setSuffix(" %")
        self.max_cutoff.setToolTip("Leave out particles with more than this % of their outline on the ROI edge "
                                   "(their size is cut short); 100 = use all. Particles without a cut-off value "
                                   "(processed before v14.23) are kept.")
        self.foc_crit = QDoubleSpinBox(); self.foc_crit.setRange(0, 500)
        self.foc_crit.setDecimals(4); self.foc_crit.setSingleStep(.01)
        self.foc_crit.setValue(sizing.DEFAULT_FOC_CRIT)
        self.foc_crit.setToolTip("Focus threshold used by proc/batchProcess.py (foc_crit); the depth of field is "
                                 "evaluated here")
        self.offset = QDoubleSpinBox(); self.offset.setRange(-20, 40); self.offset.setDecimals(2)
        self.offset.setValue(sizing.PIPELINE_PAD_PX * sizing.NOMINAL_PIXEL_UM)
        self.offset.setSuffix(" µm")
        self.offset.setToolTip("Additive edge offset. proc/imageStats.py grows the particle mask by "
                               "contains_points(radius=2) ≈ +2 px on the diameter.")
        self.fit_offset = QCheckBox("fit it (needs ≥2 true sizes)")
        off_row = QHBoxLayout(); off_row.addWidget(self.offset); off_row.addWidget(self.fit_offset)
        self.size_measure = QComboBox()
        for key, info in sizing.SIZE_MEASURES.items():
            n = sum(1 for r in rows if r.get(info["columns"][0]) not in (None, "", "nan")
                    and math.isfinite(float(r.get(info["columns"][0]))))
            self.size_measure.addItem(f"{info['label']}  ({n:,} particles)", key)
        self.size_measure.setToolTip(
            "Which measured size to calibrate. Equivalent-area diameter averages out a ragged outline and was "
            "the most consistent across 15/40/100 µm beads; it needs the transmittance edge method. Retrieval "
            "with the saved model uses the same measure.")
        self.focus_measure = QComboBox()
        for key, info in sizing.FOCUS_MEASURES.items():
            col = info["columns"][0]
            n = 0
            for r in rows:
                try:
                    n += math.isfinite(float(r.get(col)))
                except (TypeError, ValueError):
                    pass
            self.focus_measure.addItem(f"{info['label']}  ({n:,} particles)", key)
            if n == 0 and key != "focus":
                self.focus_measure.model().item(self.focus_measure.count() - 1).setEnabled(False)
        self.focus_measure.setToolTip(
            "Which focus value to calibrate with. Focus / local illumination is derived from each raw "
            "CPI gradient focus and local brightness, not focus_T. CPIview focus: peak / width of the gradient across the edge at "
            "16 points round the boundary (as SPEC's CPIview), needs processing with --cpiview. Retrieval with "
            "the saved model uses the same focus.")
        self.pixel_mode = QComboBox()
        try:
            processed_px = sizing.processed_pixel_um(sizing.CalibrationTable.from_rows(self.rows))
            processed_text = f"{processed_px:g} µm"
        except ValueError:               # particles processed with different pixel sizes
            processed_px, processed_text = sizing.NOMINAL_PIXEL_UM, "mixed — reprocess with one --pixel-um"
        self.pixel_mode.addItem(f"As processed ({processed_text})", "processed")
        self.pixel_mode.addItem("Fit from the bead sizes (≥2 sizes)", "fitted")
        self.pixel_mode.addItem("Set to…", "set")
        self.pixel_mode.setToolTip(
            "As processed: sizes are used as processed; the fitted scale s absorbs any pixel-size error.\n"
            "Fit: find the factor that makes the beads' sizes right (needs two or more true sizes, so the scale "
            "and the edge offset can be told apart), rescale every measured size by it, then fit with s = 1.\n"
            "Set: rescale to a pixel size you know (e.g. from a graticule), then fit with s = 1. The factor is\n"
            "set ÷ the pixel size the files were processed with (read from the files), so it is never applied twice.\n"
            "Retrieval with the saved model applies the same factor.")
        self.pixel_um = QDoubleSpinBox(); self.pixel_um.setRange(0.5, 10.0); self.pixel_um.setDecimals(4)
        self.pixel_um.setSingleStep(0.01); self.pixel_um.setValue(processed_px); self.pixel_um.setSuffix(" µm")
        self.pixel_um.setEnabled(False)
        self.pixel_mode.currentIndexChanged.connect(
            lambda *_: self.pixel_um.setEnabled(self.pixel_mode.currentData() == "set"))
        px_row = QHBoxLayout(); px_row.addWidget(self.pixel_mode); px_row.addWidget(self.pixel_um)
        form.addRow(self.flip)
        form.addRow("Size measure", self.size_measure)
        form.addRow("Focus", self.focus_measure)
        form.addRow("Pixel size", px_row)
        form.addRow("Min local illumination", self.illum_min)
        form.addRow("Max cut-off by ROI edge", self.max_cutoff)
        form.addRow("DOF focus threshold", self.foc_crit)
        form.addRow(self.fit_p)
        form.addRow(self.focus_asymmetric)
        form.addRow(self.peak_by_size)
        form.addRow(self.width_by_size)
        form.addRow(self.validate_folds)
        form.addRow("Edge offset b", off_row)
        self.shape_terms = QCheckBox("Particle shape terms (Park et al. 2026)")
        self.shape_terms.setToolTip(
            "Fit the effect of particle shape: defocus variable Zd / Ar^α in the size model (α = 1 is Park et al.'s "
            "Z/(L² Ar)) and Ar^c on the focus peak, with Ar = measured roundness (projected area ÷ circle of the "
            "same length). Needs particles of different shapes, e.g. beads and ice analogues. Retrieval with the "
            "saved model uses each particle's roundness.")
        form.addRow(self.shape_terms)
        self.circle_padding = QDoubleSpinBox(); self.circle_padding.setRange(0, 30)
        self.circle_padding.setDecimals(1); self.circle_padding.setValue(8.)
        self.circle_padding.setSuffix(" px")
        self.circle_compute = QPushButton("Recompute circle sizes as in Edge Lab (temporary)")
        self.circle_compute.setToolTip(
            "Uses each original ROI/background and its SAVED outline, just like Edge Lab. "
            "The fitted-circle aperture may differ from processing if its edge settings changed. "
            "This does not write to any MAT file or change morphology.")
        self.circle_compute.setToolTip("Calculate missing shadow diameters from the saved MAT images/backgrounds, "
                                       "without writing MAT files. For multi-project fits, process each source "
                                       "with --shadow-circle-padding before saving its project.")
        self.circle_compute.clicked.connect(self._compute_circle_sizes)
        self.circle_status = QLabel("")
        self.circle_status.setObjectName("hint")
        self.circle_status.setWordWrap(True)
        form.addRow("Circle expansion", self.circle_padding)
        form.addRow(self.circle_compute)
        form.addRow(self.circle_status)
        self._circle_worker = None
        if (parent is None or getattr(parent, "store", None) is None or
                (self.inputs and len(self.inputs) != 1)):
            self.circle_compute.setEnabled(False)
            self.circle_status.setText("For combined projects, process each run with "
                                       "--shadow-circle-padding 8, then Save project and reopen this fit.")
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.size_measure.currentIndexChanged.connect(self._fit_measure_changed)
        self.focus_measure.currentIndexChanged.connect(self._fit_measure_changed)
        # Start with the established fit, regardless of whether circle data exist.
        # Circle sizing can be selected explicitly and optionally computed in memory.
        self._preset_changed()
        self.run_button = QPushButton("Fit && validate")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run)
        self.save_button = QPushButton("Save model (JSON)…")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        self.figure_button = QPushButton("Save figure…")
        self.figure_button.clicked.connect(self.save_figure)
        self.figure_button.setEnabled(False)
        controls.addWidget(QLabel("Labels to fit"))
        controls.addWidget(self.label_box)
        controls.addLayout(form)
        for b in (self.run_button, self.save_button, self.figure_button):
            controls.addWidget(b)
        controls.addStretch(1)
        panel = QWidget()
        panel.setLayout(controls)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(450)
        layout.addWidget(scroll)
        self.figure = Figure(figsize=(14, 9), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._redraw)
        self.canvas.installEventFilter(self)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True)
        self.report.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.report.setFont(QFont("Menlo, Consolas, DejaVu Sans Mono", 9))
        source_text = "".join(f"  {i['project']}: {i['particles']:,} particles\n" for i in self.inputs)
        self.report.setPlainText(
            f"{len(rows):,} accepted particles with a true size and a best-focus reference"
            + (f", from {len(self.inputs)} project(s):\n{source_text}" if self.inputs else ".\n") + "\n"
            "Fit && validate fits the length and focus models on each side of the object plane, then refits "
            "leaving out one batch at a time and retrieves that batch's sizes (side known, and side unknown "
            "as for cloud data) to show how well the retrieval really works.")
        self.tabs = QTabWidget()
        fig_tab = QWidget()
        fig_layout = QVBoxLayout(fig_tab)
        fig_layout.setContentsMargins(0, 0, 0, 0)
        fig_layout.addWidget(NavigationToolbar2QT(self.canvas, fig_tab))
        fig_layout.addWidget(self.canvas, 1)
        self.tabs.addTab(fig_tab, "Figure")
        # A separate, independent shape/shadow diagnostic: no microscope values
        # are required by either fitting or retrieval.
        self.shape_figure = Figure(figsize=(11, 7), layout="constrained")
        self.shape_canvas = FigureCanvasQTAgg(self.shape_figure)
        shape_tab = QWidget()
        shape_layout = QVBoxLayout(shape_tab)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.addWidget(NavigationToolbar2QT(self.shape_canvas, shape_tab))
        shape_layout.addWidget(self.shape_canvas, 1)
        self.tabs.addTab(shape_tab, "Shadow / shape check")
        self.tabs.addTab(self.report, "Report")
        self.tabs.setCurrentIndex(1)
        layout.addWidget(self.tabs, 1)

    def _preset_changed(self, *_):
        recipe = self.preset.currentData()
        size = {"circle": "shadow_deq", "contour": "deq", "area_corrected": "area_corrected_shadow", "original": "length"}[recipe]
        foc = "focus" if recipe == "original" else "focus_per_illum"
        for widget, key in ((self.size_measure, size), (self.focus_measure, foc)):
            pos = widget.findData(key)
            if pos >= 0:
                widget.setCurrentIndex(pos)
        self.offset.setValue(0. if recipe == "circle" else sizing.PIPELINE_PAD_PX*sizing.NOMINAL_PIXEL_UM)
        self.fit_offset.setChecked(recipe == "circle")
        self.foc_crit.setValue(5. if recipe == "original" else .12)
        self._fit_measure_changed()

    def _fit_measure_changed(self, *_):
        size, foc = self.size_measure.currentData(), self.focus_measure.currentData()
        shadow = size in ("shadow_deq", "area_corrected_shadow")
        norm = foc == "focus_per_illum"
        self.shape_terms.setEnabled(not shadow)
        self.pixel_mode.setEnabled(not shadow)
        if shadow:
            self.pixel_mode.setCurrentIndex(self.pixel_mode.findData("processed"))
        self.fit_p.setEnabled(norm)
        self.focus_asymmetric.setEnabled(norm)
        self.peak_by_size.setEnabled(norm)
        self.width_by_size.setEnabled(norm)
        n = sum(math.isfinite(sizing._num(r.get("shadow_deq_um"))) and
                sizing._num(r.get("shadow_deq_um")) > 0 for r in self.rows)
        self.method_help.setText(
            (f"Circle diameter: {n:,} saved/preview values of {len(self.rows):,}. "
             "Raw plot = signed shadow integral diameter / labelled true diameter, "
             "exactly as in Edge Lab (when outlines and radius match). "
             "Signed deficit → size via a fitted scale and offset, independent of Z. "
             "Conventional contour measurements remain available for morphology. " if shadow else
             ("Shape-corrected diameter = signed-shadow diameter / sqrt(CPI contour area / enclosing-circle area). Raw shadow remains available. " if size == "area_corrected_shadow" else
             "Fits the existing size correction from the selected contour measurement. ")) +
            ("Focus F_raw/I_local: asymmetric exp[-(|Z-Z0|/w±(L))^p], "
             "with an L-dependent peak and width. Threshold is in focus/illumination units."
             if norm else "Original CPI focus model with fitted illumination exponent. "
             "Threshold is in raw focus units."))
        self.circle_padding.setEnabled(shadow)
        self.circle_compute.setVisible(shadow)
        self.circle_status.setVisible(shadow)

    def _compute_circle_sizes(self):
        """Read-only fitting convenience; writes to this dialog's rows, never the project."""
        parent = self.parent()
        if self._circle_worker is not None and self._circle_worker.isRunning():
            return
        if parent is None or getattr(parent, "store", None) is None or getattr(parent, "catalog", None) is None:
            QMessageBox.information(self, "Circle sizes", "Load a calibration project first.")
            return
        if len(self.inputs) > 1:
            QMessageBox.information(self, "Circle sizes", "For multiple projects, calculate the circle size "
                                    "in each source using --shadow-circle-padding and save their project tables.")
            return
        ids = {str(r.get("particle_id")) for r in self.rows}
        records = [r for r in parent.catalog.records if r.particle_id in ids]
        if not records:
            QMessageBox.warning(self, "Circle sizes", "No matching particle IDs in the open project.")
            return
        self.circle_compute.setEnabled(False)
        self.circle_status.setText(f"Calculating {len(records):,} ROI shadows, no files written…")
        worker = EdgePreviewWorker(parent.store.root, records,
                                   edge_preview.shadow_options(self.circle_padding.value()),
                                   "sizing-circle", parent)
        self._circle_worker = worker
        worker.progress_changed.connect(
            lambda done, total: self.circle_status.setText(f"Circle sizes: {done:,}/{total:,} ROIs…"))
        worker.completed.connect(self._circle_finished)
        worker.finished.connect(lambda: self.circle_compute.setEnabled(True))
        worker.start()

    def _circle_finished(self, _identity, results, failures):
        for row in self.rows:
            data = results.get(str(row.get("particle_id")))
            if data is not None:
                row["shadow_deq_um"] = data["shadow_diameter"]
        count = len(results)
        self.circle_status.setText(f"Computed {count:,} circle diameters; {len(failures):,} unavailable. "
                                   "Temporary until source data are reprocessed.")
        item = self.size_measure.model().item(self.size_measure.findData("shadow_deq"))
        if item is not None:
            item.setText(f"Circle signed-shadow diameter  ({count:,} newly calculated)")
        self._fit_measure_changed()

    def closeEvent(self, event):
        if self._circle_worker is not None and self._circle_worker.isRunning():
            self._circle_worker.requestInterruption()
            if not self._circle_worker.wait(1500):
                # Worker owned by the main window to outlive the fitting dialog.
                with contextlib.suppress(RuntimeError, TypeError):
                    self._circle_worker.completed.disconnect(self._circle_finished)
                    self._circle_worker.progress_changed.disconnect()
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.canvas and event.type() == QEvent.Resize and self._drawn is not None:
            self._redraw_timer.start(250)             # redraw to fit once resizing pauses
        return super().eventFilter(obj, event)

    def _redraw(self):
        if self._drawn is None:
            return
        model, table, validation, tracking = self._drawn
        sizing.make_figure(model, table, validation, tracking, figure=self.figure)
        self.canvas.draw_idle()
        sizing.make_shadow_shape_figure(table, figure=self.shape_figure)
        self.shape_canvas.draw_idle()

    def _table(self):
        chosen = {self.label_box.item(i).data(Qt.UserRole) for i in range(self.label_box.count())
                  if self.label_box.item(i).checkState() == Qt.Checked}
        rows = [r for r in self.rows if (r.get("label") or "") in chosen]
        limit = self.max_cutoff.value()
        if limit < 100:
            def cut(r):
                try:
                    return float(r.get("cutoff_pct"))
                except (TypeError, ValueError):
                    return float("nan")
            rows = [r for r in rows if not cut(r) > limit]            # unknown (NaN) is kept
        table = sizing.CalibrationTable.from_rows(rows, self.size_measure.currentData() or "length",
                                                  self.focus_measure.currentData() or "focus")
        if self.flip.isChecked():
            table.d = -table.d
        return table

    def run(self):
        table = self._table()
        shadow = table.size_measure in ("shadow_deq", "area_corrected_shadow")
        normalised = table.focus_measure == "focus_per_illum"
        if shadow and np.count_nonzero(np.isfinite(table.length) & (table.length > 0)) < 20:
            QMessageBox.information(self, "Circle sizing unavailable",
                "Fewer than 20 usable circle diameters are available. Click ‘Compute circle sizes’ "
                "for a temporary fit, or run batchProcess.py --shadow-circle-padding 8 "
                "on a COPY of the source MAT directory, then reopen the project.")
            return
        if normalised and np.count_nonzero(np.isfinite(table.focus) & (table.focus > 0)) < 20:
            QMessageBox.information(self, "Normalised focus unavailable",
                "Focus/local illumination requires raw focus and a positive local illumination. "
                "Check the particle images/background and processing outputs.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mode = self.pixel_mode.currentData() or "processed"
            self.model, self.validation, self.tracking = sizing.run_calibration(
                table, illum_min=(self.illum_min.value() or None), foc_crit=self.foc_crit.value(),
                offset_um=self.offset.value(),
                fit_offset=(self.fit_offset.isChecked() if shadow
                            else True if self.fit_offset.isChecked() else None),
                pixel_mode=mode, pixel_set_um=self.pixel_um.value() if mode == "set" else None,
                shape=self.shape_terms.isChecked() and not shadow,
                size_model="shadow_linear" if shadow else "defocus",
                focus_model="normalised_generalised" if normalised else "legacy",
                fit_focus_p=self.fit_p.isChecked(), asymmetric_focus=self.focus_asymmetric.isChecked(),
                fit_peak_size=self.peak_by_size.isChecked(), fit_width_size=self.width_by_size.isChecked(),
                do_validation=self.validate_folds.isChecked())
        except Exception as exc:   # report fitting problems instead of crashing the workbench
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Fit size & focus model", str(exc))
            return
        QApplication.restoreOverrideCursor()
        if self.flip.isChecked():
            self.model.sign_convention += " (flipped: d → −d)"
        self.model.inputs = self.inputs
        self.table = table
        self._drawn = (self.model, table, self.validation, self.tracking)
        self.tabs.setCurrentIndex(0)
        self._redraw()
        self.report.setPlainText(sizing.format_report(self.model, self.validation, self.tracking))
        self.save_button.setEnabled(True)
        self.figure_button.setEnabled(True)

    def _default_path(self, suffix):
        parent = self.parent()
        project = getattr(parent, "project_file", None)
        return str(project.with_name(project.stem + suffix)) if project else "sizing_model" + suffix[-5:]

    def save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save sizing model", self._default_path("_sizing_model.json"),
                                              "JSON (*.json)")
        if path:
            self.model.save(path)
            self.report.appendPlainText(f"\nSaved model → {path}")

    def save_figure(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", self._default_path("_sizing_fit.png"),
                                              "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=130)


class CalibrationWindow(QMainWindow):
    _windows: list = []          # every open workbench window (File ▸ New window)

    def __init__(self, initial_directory: Optional[str] = None, position_file: Optional[str] = None,
                 z_min_um: float = -10000.0, z_max_um: float = 10000.0, z_step_um: float = 10.0):
        super().__init__()
        CalibrationWindow._windows.append(self)
        self.setWindowTitle("CPI · Calibration Workbench v12")
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
        self.position_series = None   # None = every series in a consolidated stage CSV
        self.position_time_basis = None
        # More stage CSVs (merged projects): [{"path", "series", "time_basis", "from"}]
        self.extra_position_files: list[dict] = []
        self.time_windows: list[dict] = []     # named time windows (e.g. one per bead size)
        self._shown_batches: list[str] = []    # several Flag-history batches shown together
        self.active_time_window = None
        self.z_min_um, self.z_max_um, self.z_step_um = z_min_um, z_max_um, z_step_um
        self._updating_z = False
        self.stage_segments = []
        self.saved_sets = {}
        self.active_reference_set = None
        self._reference_index = {}
        self._active_members = set()
        self._draft_reference_cache = None
        self._steps_cache = None
        self._groups_cache = None
        self._session_of = {}
        self._own_ref_names = {}
        self._groups_of = {}
        self._ref_manual = False      # True once the user moved Xbest/Zbest by hand
        self._ref_set_for_cluster = False  # the line was placed (peak or by hand) for the current cluster
        self._auto_placing = False
        self._last_filter_spec = None
        self._library_settings = QSettings("CPI", "CalibrationWorkbench")
        self._library_mode = ("manual" if os.environ.get("CPI_CALIBRATION_LABELS") else
                              str(self._library_settings.value("reference_library_mode", "auto")))
        saved_path = (os.environ.get("CPI_CALIBRATION_LABELS") or
                      self._library_settings.value("reference_library_path", ""))
        others = [w for w in CalibrationWindow._windows if w is not self and hasattr(w, "labels")]
        try:
            # Windows share one label library, so an edit in one is never overwritten by another.
            self.labels = (others[0].labels if others else
                           LabelLibrary.load(saved_path if saved_path else None))
            self._label_error = None
        except Exception as exc:  # unreadable library: keep working, warn once
            self.labels = LabelLibrary()
            self._label_error = str(exc)

        # Interaction state
        self.selected_ids: set[str] = set()
        self.regions = []          # history of box gestures (kept for project files)
        self.undo_stack = []
        self.redo_stack = []
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
        self._mat_loader = None
        # The edge laboratory is strictly transient: never passed to autosave,
        # project JSON, flags or processing-run metadata.
        self._edge_results = {}  # view of active trial (never persisted)
        self._edge_failures = {}
        self._edge_trials = OrderedDict()
        self._edge_saved_outline_cache = OrderedDict()  # limited, on-demand; no new edge detection
        self._edge_next_id = 1
        self._edge_pending = []
        self._edge_active_id = None
        self._edge_hover_trial_id = None
        self._edge_worker = None
        self._nc_export_worker = None
        self._nc_export_dialog = None
        self._edge_hover_points = []
        # Writes are batched: an action marks what changed and the files are written
        # once, shortly after the last change (and always on close / project switch).
        self._pending_flags = False
        self._pending_project = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1500)
        self._save_timer.timeout.connect(lambda: self.flush_saves(background=True))
        self._writer_thread = None
        self._save_notifier = _SaveNotifier()
        self._save_notifier.done.connect(self._save_finished)

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
        self.edge_tab = self._build_edge_tab()
        self.viewer_tabs.addTab(self.select_tab, "①  Select on plots")
        self.viewer_tabs.addTab(self.gallery_panel, "②  Gallery")
        self.viewer_tabs.addTab(self.particle_tab, "③  Particle")
        self.viewer_tabs.addTab(self.edge_tab, "④  Edge lab (temporary)")
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
                                       "Not a usable calibration particle (debris, overlapping, clipped, out of focus …): "
                                       "flag REJECTED (red), saved, excluded everywhere. Key R on the plots tab.\n"
                                       "To merely leave a good particle out of THIS batch, use Remove (Ctrl/⌘-drag, "
                                       "Del) instead — it stays unreviewed for a later batch.")
        self.unflag_btn = self._button("↺  Unflag selection", self.unflag_selection, "neutral",
                                       "Return selected particles to UNREVIEWED (grey), the default. "
                                       "Key U on the plots tab.")
        self.clear_sel_btn = self._button("Clear selection", self.clear_selection, None,
                                          "Deselect everything (Esc). Flags are not changed.")
        self.undo_btn = self._button("↶  Undo", self.undo, None, "Undo the last selection or flag change (Ctrl+Z)")
        self.redo_btn = self._button("↷  Redo", self.redo, None, "Redo what was undone (Ctrl+Shift+Z or Ctrl+Y)")
        accept_as = QLabel("Accept as")
        accept_as.setToolTip("Label given to particles when you accept them (selection, gallery or viewer)")
        layout.addWidget(accept_as)
        self.accept_label = QComboBox()
        self.accept_label.setMinimumWidth(170)
        self.accept_label.setToolTip("Accepting also tags particles with this label and its true size. "
                                     "Manage labels in the Labels tab.")
        self.accept_label.activated.connect(self._accept_label_chosen)
        layout.addWidget(self.accept_label)
        self.save_ref_cb = QCheckBox("+ Xbest/Zbest")
        self.save_ref_cb.setChecked(True)
        self.save_ref_cb.setToolTip("Accepting also saves the current Xbest/Zbest (the orange line) as this batch's "
                                    "own best focus. The line follows the selection's own focus peak unless you "
                                    "move it.")
        layout.addWidget(self.save_ref_cb)
        for button in (self.accept_btn, self.reject_btn, self.unflag_btn, self.clear_sel_btn, self.undo_btn,
                       self.redo_btn):
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
        tools.addWidget(QLabel("Boxes pick"))
        self.pick_scope = QComboBox()
        for value, label in PICK_SCOPES:
            self.pick_scope.addItem(label, value)
        self.pick_scope.setToolTip("Which particles a New or Add box picks up when the box contains a mix. "
                                   "'Unreviewed only' (default) skips accepted (green) and rejected (red) ones, "
                                   "so a cluster isn't picked twice. If a box contains ONLY flagged particles "
                                   "they are selected anyway, so you never need 'Everything' just to grab an "
                                   "accepted cluster again.")
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
        self.colour_by.addItem("Local illumination", "illum")
        self.colour_by.addItem("Batch Xbest (slide tilt)", "ref_x")
        self.colour_by.addItem("Batch Zbest (slide tilt)", "ref_z")
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
        self.selection_only_cb = QCheckBox("Only selection / current batch (S)")
        self.selection_only_cb.setChecked(True)
        self.selection_only_cb.setToolTip("On: while something is selected (or a batch is shown) the focus/size "
                                          "plots show only that, so outliers are easy to spot; with nothing "
                                          "selected they show everything. Off: always show all particles. "
                                          "Press S to toggle.")
        self.selection_only_cb.toggled.connect(lambda _: self._refresh_views())
        self.median_cb = QCheckBox("Median ± IQR")
        self.median_cb.setChecked(True)
        self.median_cb.setToolTip("Median and inter-quartile band of every plotted metric along X "
                                  "(rejected particles excluded). Dashed blue: median of the selection.")
        self.median_cb.toggled.connect(lambda _: self._redraw_plots())
        options.addWidget(self.selection_only_cb)
        options.addWidget(self.median_cb)
        self.window_combo = QComboBox()
        self.window_combo.setToolTip("Show one saved time window (e.g. one bead size). Manage windows in the "
                                     "Filters tab.")
        self.window_combo.addItem("All times", None)
        self.window_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.window_combo.setMinimumWidth(150)
        self.window_combo.activated.connect(
            lambda _i: self.show_time_window(self.window_combo.currentData()))
        options.addWidget(QLabel("Window"))
        options.addWidget(self.window_combo)
        self.follow_peak_cb = QCheckBox("Line follows focus peak")
        self.follow_peak_cb.setChecked(True)
        self.follow_peak_cb.setToolTip("When you select a cluster, Xbest/Zbest jump to the peak of the cluster's own "
                                       "median focus (blue ▲). Drag the line to override; it then stays put until "
                                       "you select a different cluster.")
        self.follow_peak_cb.toggled.connect(lambda on: (self._auto_place_reference(force=True), self.update_plots())
                                            if on else None)
        options.addWidget(self.follow_peak_cb)
        options.addStretch(1)
        self.scope_label = QLabel("")
        self.scope_label.setObjectName("hint")
        options.addWidget(self.scope_label)
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
        for text, colour in (("◉ Selected", SELECTED_COLOUR),
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
        gallery_display = QHBoxLayout()
        self.gallery_saved_edge = QCheckBox("Stored boundary · yellow")
        self.gallery_test_edge = QCheckBox("Active trial boundary · cyan")
        self.gallery_stat = QComboBox()
        for key, title in (("focus_length", "Focus + length"), ("length", "Length"),
                           ("distance", "Signed distance from object plane"),
                           ("position", "Stage X / Z"), ("sizes", "Stored + tested diameters"),
                           ("shadow", "Shadow-circle diameter + Feret length"),
                           ("all", "Size + distance + position"), ("none", "No measurements")):
            self.gallery_stat.addItem(title, key)
        gallery_display.addWidget(QLabel("Thumbnail overlays"))
        gallery_display.addWidget(self.gallery_saved_edge)
        gallery_display.addWidget(self.gallery_test_edge)
        gallery_display.addWidget(QLabel("Caption"))
        gallery_display.addWidget(self.gallery_stat, 1)
        layout.addLayout(gallery_display)
        for ctrl in (self.gallery_saved_edge, self.gallery_test_edge):
            ctrl.toggled.connect(self._gallery_overlay_changed)
        self.gallery_stat.currentIndexChanged.connect(self._gallery_overlay_changed)
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
                         "Leave picked particles out of this batch; they stay unreviewed for later"),
            self._button("+ Add to selection  (A)", lambda: self.gallery_selection("add")),
            self._button("✓ Accept  (K)", lambda: self.flag_gallery("keep"), "accept"),
            self._button("✕ Reject  (R)", lambda: self.flag_gallery("reject"), "reject",
                         "Debris / unusable: flag picked particles rejected"),
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
        self.reject_reason.addItems(["", "debris", "overlap", "clipped", "out of focus", "segmentation", "artefact",
                                     "wrong type", "other"])
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
        self.test_boundary_checkbox = QCheckBox("Overlay tested boundary · cyan (temporary)")
        self.test_boundary_checkbox.setChecked(True)
        self.test_boundary_checkbox.toggled.connect(lambda _: self.refresh_particle())
        col.addWidget(self.test_boundary_checkbox)
        self.background_checkbox = QCheckBox("Background-subtracted ROI")
        self.background_checkbox.stateChanged.connect(lambda _: self.refresh_particle())
        col.addWidget(self.background_checkbox)
        measure_button = QPushButton("📏 Measure on this ROI / full frames…")
        measure_button.setToolTip("Open the measuring window on this particle's ROI: measure distances at any angle "
                                  "in px (and µm), view full frames, export PNG/CSV")
        measure_button.clicked.connect(lambda: self.open_measure_window(self.current_record()))
        col.addWidget(measure_button)
        metrics_box = QGroupBox("Particle")
        metrics = QFormLayout(metrics_box)
        self.metric_labels: Dict[str, QLabel] = {}
        for key, label in (("time", "Time"), ("length", "Length [µm]"), ("width", "Width [µm]"),
                           ("focus", "Focus"), ("illum", "Local illumination"), ("roundness", "Roundness"),
                           ("xy", "Image x, y [px]"),
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

    def _build_edge_tab(self):
        """Transient, multi-trial comparison of real edge detection algorithms."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(7, 5, 7, 4)
        layout.setSpacing(5)

        header = QHBoxLayout()
        self.edge_batch_name = QLabel("Select one or more batches in Review → Flag history")
        self.edge_batch_name.setStyleSheet("font-weight:700; color:#193c5e;")
        header.addWidget(self.edge_batch_name, 1)
        header.addWidget(self._button("Clear all trials", self._clear_edge_preview, None,
                                      "Discard all Edge Lab trials; never writes MAT, flags, or project data"))
        layout.addLayout(header)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(QLabel("Method"))
        self.edge_algorithm = QComboBox()
        for key, label in (*edge_preview.METHODS, edge_preview.SHADOW_METHOD):
            self.edge_algorithm.addItem(label, key)
        controls.addWidget(self.edge_algorithm, 3)
        self.edge_level_label = QLabel("Shadow fraction")
        controls.addWidget(self.edge_level_label)
        self.edge_level = QDoubleSpinBox()
        self.edge_level.setRange(.05, .95)
        self.edge_level.setDecimals(2)
        self.edge_level.setSingleStep(.01)
        self.edge_level.setValue(.55)
        controls.addWidget(self.edge_level)
        self.edge_radius_mode_label = QLabel("Circle radius")
        self.edge_radius_mode = QComboBox()
        self.edge_radius_mode.addItem("Fixed +px (established)", "fixed")
        self.edge_radius_mode.addItem("Automatic plateau (experimental)", "adaptive")
        self.edge_radius_mode.setToolTip("Fixed circles may extend outside the ROI: missing pixels count as "
                                        "clear transmittance T=1 (zero signed deficit). Adaptive grows "
                                        "the circle until a sustained plateau, requiring enough observed "
                                        "annulus coverage to avoid artificial plateaus at the ROI edge.")
        controls.addWidget(self.edge_radius_mode_label)
        controls.addWidget(self.edge_radius_mode)
        self.edge_padding_label = QLabel("Extra radius [px]")
        self.edge_padding = QDoubleSpinBox()
        self.edge_padding.setRange(0, 30)
        self.edge_padding.setDecimals(1)
        self.edge_padding.setSingleStep(1)
        self.edge_padding.setValue(8)
        self.edge_padding.setToolTip("Add to the radius of the circle fitted to the EXISTING saved boundary. "
                                      "The measured size comes from the signed light deficit, not this boundary.")
        controls.addWidget(self.edge_padding_label)
        controls.addWidget(self.edge_padding)
        controls.addWidget(QLabel("Median passes"))
        self.edge_denoise = QComboBox()
        for i in range(4):
            self.edge_denoise.addItem(str(i), i)
        self.edge_denoise.setCurrentIndex(1)
        controls.addWidget(self.edge_denoise)
        controls.addWidget(QLabel("Gaussian σ"))
        self.edge_smooth = QDoubleSpinBox()
        self.edge_smooth.setRange(0, 3)
        self.edge_smooth.setDecimals(1)
        self.edge_smooth.setSingleStep(.2)
        controls.addWidget(self.edge_smooth)
        self.edge_cpiview = QCheckBox("CPIview focus")
        self.edge_cpiview.setToolTip("Run the separate CPIview 2 algorithm (slower): 12 median passes, "
                                      "16 edge-gradient profiles and peak/FWHM focus")
        controls.addWidget(self.edge_cpiview)
        info = self._button("ⓘ", self._edge_info, None, "Processing and CPIview parameter definitions")
        controls.addWidget(info)
        layout.addLayout(controls)

        flag_row = QHBoxLayout()
        flag_row.addWidget(QLabel("Production CLI flags"))
        self.edge_cli = QLineEdit()
        self.edge_cli.setReadOnly(True)
        self.edge_cli.setToolTip("Copy/pasteable batchProcess.py flags. For shadow integration this "
                                 "adds an independent measurement while keeping the stored edge/focus.")
        flag_row.addWidget(self.edge_cli, 1)
        flag_row.addWidget(self._button("Copy flags", self._edge_copy_flags))
        self.edge_run = self._button("＋ Add trial to plot", self._start_edge_preview, "primary",
                                     "Snapshot these settings and currently selected batches as a new temporary curve")
        flag_row.addWidget(self.edge_run)
        layout.addLayout(flag_row)

        trials_row = QHBoxLayout()
        trials_row.addWidget(QLabel("Trials  (tick to show; select to inspect)"))
        self.edge_trials_list = QListWidget()
        self.edge_trials_list.setMinimumHeight(72)
        self.edge_trials_list.setMaximumHeight(118)
        self.edge_trials_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.edge_trials_list.currentItemChanged.connect(self._edge_select_trial)
        self.edge_trials_list.itemChanged.connect(self._edge_toggle_trial)
        trials_row.addWidget(self.edge_trials_list, 1)
        trials_actions = QVBoxLayout()
        trials_actions.addWidget(self._button("＋ Stored (instant)", self._edge_add_stored,
                                                None, "Add saved measurements as a selectable reference: no recalculation"))
        trials_actions.addWidget(self._button("Remove selected", self._edge_remove_trial))
        trials_actions.addWidget(self._button("Copy selected flags", self._edge_copy_selected_flags))
        trials_actions.addWidget(self._button("Show all", lambda: self._edge_show_all(True)))
        trials_actions.addWidget(self._button("Hide all", lambda: self._edge_show_all(False)))
        trials_row.addLayout(trials_actions)
        layout.addLayout(trials_row)

        axes_row = QHBoxLayout()
        axes_row.addWidget(QLabel("X axis"))
        self.edge_xaxis = QComboBox()
        for key, title in (("plane_signed", "Signed distance d"), ("plane_zd", "Normalised distance Zd"),
                           ("stage_x", "Stage X"), ("stage_z", "Stage Z")):
            self.edge_xaxis.addItem(title, key)
        axes_row.addWidget(self.edge_xaxis)
        axes_row.addWidget(QLabel("Size"))
        self.edge_metric = QComboBox()
        for key, title in (("deq", "Equivalent-area diameter"), ("len", "Maximum Feret length"),
                           ("wid", "Minimum Feret width"),
                           ("ratio_deq", "Contour equivalent diameter / labelled true size"),
                           ("ratio_len", "Feret length / true size"),
                           ("shadow_deq", "Stored shadow-circle diameter (processing)"),
                           ("ratio_shadow", "Stored shadow-circle diameter / true size"),
                           ("ratio_shadow_corrected", "Shadow diameter / true size · reference corrected"),
                           ("ratio_shadow_cpi_area", "Shape-corrected shadow / true size (CPI area ratio)")):
            self.edge_metric.addItem(title, key)
        self.edge_metric.setCurrentIndex(self.edge_metric.findData("ratio_deq"))
        axes_row.addWidget(self.edge_metric)
        axes_row.addWidget(QLabel("Microscopy area ratio"))
        self.edge_microscopy_area_ratio = QDoubleSpinBox()
        self.edge_microscopy_area_ratio.setRange(0.001, 1.0)
        self.edge_microscopy_area_ratio.setDecimals(3)
        self.edge_microscopy_area_ratio.setSingleStep(0.025)
        self.edge_microscopy_area_ratio.setValue(0.30)
        self.edge_microscopy_area_ratio.setToolTip(
            "Independent microscope object area / enclosing-circle area. "
            "Applies sqrt(area ratio / (Dshadow/Llabel)^2) to the optical area; "
            "equivalently corrected ratio = sqrt(area ratio). "
            "This is a reference correction, NOT a measured optical-size retrieval.")
        axes_row.addWidget(self.edge_microscopy_area_ratio)
        axes_row.addWidget(QLabel("Reference Dshadow/Ltrue"))
        self.edge_reference_shadow_ratio = QDoubleSpinBox()
        self.edge_reference_shadow_ratio.setRange(0.001, 3.)
        self.edge_reference_shadow_ratio.setDecimals(3)
        self.edge_reference_shadow_ratio.setSingleStep(0.025)
        self.edge_reference_shadow_ratio.setValue(0.40)
        self.edge_reference_shadow_ratio.setToolTip(
            "Independent/reference measured Dshadow / true maximum length for the same particle type. "
            "Correction multiplier = sqrt(microscopy area ratio) / reference shadow ratio. "
            "This is an empirical shape/optical correction, not CPI extinction efficiency.")
        axes_row.addWidget(self.edge_reference_shadow_ratio)


        axes_row.addWidget(QLabel("Focus"))
        self.edge_focus_metric = QComboBox()
        self.edge_focus_metric.addItem("Edge gradient", "focus")
        self.edge_focus_metric.addItem("Transmittance gradient", "focus_T")
        self.edge_focus_metric.addItem("CPIview 2 focus (16 profiles)", "cpv_focus")
        self.edge_focus_metric.addItem("CPIview edge FWHM [px]", "cpv_fwhm_px")
        axes_row.addWidget(self.edge_focus_metric)
        layout.addLayout(axes_row)

        self.edge_status = QLabel("Select one or more saved batches and press Add trial. Nothing is committed.")
        self.edge_status.setObjectName("hint")
        self.edge_status.setWordWrap(True)
        layout.addWidget(self.edge_status)
        self.edge_diagnostics = QLabel("")
        self.edge_diagnostics.setObjectName("hint")
        self.edge_diagnostics.setWordWrap(True)
        layout.addWidget(self.edge_diagnostics)
        self.edge_figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.edge_size_ax, self.edge_focus_ax = self.edge_figure.subplots(2, 1, sharex=True)
        self.edge_canvas = FigureCanvasQTAgg(self.edge_figure)
        self.edge_canvas.setMouseTracking(True)
        self.edge_canvas.mpl_connect("motion_notify_event", self._edge_mouse_move)
        self.edge_canvas.mpl_connect("figure_leave_event", lambda _: self._hide_hover())
        self.edge_canvas.mpl_connect("button_press_event", self._edge_mouse_click)
        layout.addWidget(self.edge_canvas, 1)
        hint = QLabel("Every curve is selectable, including stored data (loaded without recalculation). "
                      "Hover/right-click a curve to see its same-colour saved or tested outline. "
                      "For Shadow circle the same-colour overlay is the INTEGRATION APERTURE, not a measured edge. "
                      "Shadow Deq is an uncalibrated light-deficit proxy: ratio = Deq / labelled true diameter. "
                      "Focus and transmittance-gradient focus ALWAYS use the original stored value for "
                      "every trial (including particles with failed circles). "
                      "Changing controls affects the NEXT trial only; no changes are committed.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        for ctrl in (self.edge_algorithm, self.edge_level, self.edge_denoise,
                     self.edge_smooth, self.edge_cpiview, self.edge_padding, self.edge_radius_mode):
            signal = (ctrl.valueChanged if isinstance(ctrl, QDoubleSpinBox) else
                      ctrl.toggled if isinstance(ctrl, QCheckBox) else ctrl.currentIndexChanged)
            signal.connect(self._edge_settings_changed)
        for ctrl in (self.edge_xaxis, self.edge_metric):
            ctrl.currentIndexChanged.connect(self._plot_edge_preview)
        self.edge_microscopy_area_ratio.valueChanged.connect(self._plot_edge_preview)
        self.edge_reference_shadow_ratio.valueChanged.connect(self._plot_edge_preview)
        self.edge_focus_metric.currentIndexChanged.connect(self._edge_focus_selected)
        self._edge_settings_changed()
        self._plot_edge_preview()
        return page

    def _edge_info(self):
        QMessageBox.information(self, "Edge Lab: methods and CPIview", "Edge Lab is read-only and runs on "
            "the ROIs and backgrounds already stored in the MAT files. Add Stored instantly to plot "
            "saved measurements with their existing dat.foc boundary (NO edge calculation); saved "
            "equivalent diameter is used when available, otherwise saved Feret length. "
            "Each Add trial snapshots the selected "
            "batches and algorithm parameters; switching controls never changes earlier curves.\n\n"
            "Fixed transmittance: background-scaled image/BG has ambient T ≈ 1; choose any "
            "absolute edge level (e.g. T=0.80 means 20% attenuation). Faint-particle detection "
            "uses a separate minimum-contrast and connected-area test, never a T<0.45 core requirement. "
            "Right-click a trial particle to plot measured centre transmittance profiles.\n\n"
            "Relative shadow: T(edge) = 1 − fraction × (1 − T(min)). A smaller fraction traces "
            "a brighter, generally larger boundary. Radial uses locally adapted thresholds and "
            "smooths around roughly round particles. Steepest gradient seeks the strongest rise in T.\n\n"
            "Edge gradient: raw image-minus-background derivative measured along the detected edge "
            "(60 samples). Transmittance gradient: same profile on normalised image/background.\n\n"
            "CPIview 2 focus is a SEPARATE implementation (not equivalent to those two gradients): "
            "12 median-filter passes on the difference image, its own dark-object segmentation, "
            "16 normal boundary profiles and focus = mean(100 × peak / FWHM / 110), "
            "with FWHM below 3 px set to 30. It is slower and is only calculated when "
            "the CPIview checkbox is enabled. The checkbox adds --cpiview to the CLI flags.\n\n"
            "Signed-shadow circle: fit a circle to the previously STORED boundary, enlarge its radius "
            "by the selected number of pixels and integrate the signed light deficit (1 - T) inside. "
            "Bright diffraction fringes contribute negatively. The resulting diameter is an "
            "UNCALIBRATED shadow-equivalent diameter, not a validated true size, and the drawn circle "
            "is an aperture, not a measured boundary. Circle truncation is reported as a failure. "
            "Batch processing can additionally save this measurement via --shadow-circle-padding 8; "
            "that does not replace existing length, shape or focus.\n\n"
            "The copied flags are for the production batchProcess.py interface; full processing may "
            "select different backgrounds. Running --only-stats on an existing directory WRITES new "
            "statistics: work on a COPY. The laboratory itself never writes them.")

    def _edge_copy_flags(self):
        if self.edge_algorithm.currentData() == "shadow/circle":
            QApplication.clipboard().setText(self.edge_cli.text())
            self._toast("Shadow-circle processing flag copied (no MAT data modified)", 4000)
            return
        QApplication.clipboard().setText(self.edge_cli.text())
        self._toast("Processing flags copied; the Edge Lab has not modified any data.", 4000)

    def _edge_copy_selected_flags(self):
        trial = self._edge_trials.get(self._edge_active_id)
        if not trial:
            self._toast("Choose a trial in the list first", 3500)
            return
        if not trial["flags"]:
            self._toast("Stored reference uses the existing processing settings", 4500)
            return
        QApplication.clipboard().setText(trial["flags"])
        self._toast(f"Copied the exact flags used by trial #{trial['id']}", 3500)

    def _edge_focus_selected(self, *_):
        if self.edge_focus_metric.currentData() in ("cpv_focus", "cpv_fwhm_px"):
            if not self.edge_cpiview.isChecked():
                self.edge_cpiview.setChecked(True)  # CPIview applies to the next trial
                self.edge_status.setText("CPIview requested: add a new trial to calculate it. "
                                         "Older trials without CPIview are unchanged.")
        self._plot_edge_preview()

    def _edge_settings_changed(self, *_):
        if not hasattr(self, "edge_cli"):
            return
        method = self.edge_algorithm.currentData()
        shadow = method == "shadow/circle"
        self.edge_level_label.setVisible(not shadow)
        self.edge_level.setVisible(not shadow)
        self.edge_radius_mode_label.setVisible(shadow)
        self.edge_radius_mode.setVisible(shadow)
        self.edge_padding_label.setVisible(shadow)
        self.edge_padding.setVisible(shadow)
        adaptive = shadow and self.edge_radius_mode.currentData() == "adaptive"
        self.edge_padding_label.setText("Max extra radius [px]" if adaptive else "Extra radius [px]")
        self.edge_padding.setToolTip("Maximum radius to search beyond the saved contour" if adaptive else
                                     "Additional fixed radius beyond the saved contour")
        self.edge_cpiview.setEnabled(not shadow)
        self.edge_level_label.setText("Shadow fraction" if method.startswith("relative") else "Absolute T")
        self.edge_level.setToolTip("Fixed: T=0.8 means 20% attenuation. Relative: fraction of shadow depth.")
        self.edge_cli.setText((f"--shadow-circle-auto --shadow-circle-max-padding {max(6,self.edge_padding.value()):g}"
                               if adaptive else f"--shadow-circle-padding {self.edge_padding.value():g}") if shadow else
            edge_preview.command_flags(method, self.edge_level.value(), self.edge_denoise.currentData(),
                                       self.edge_smooth.value(), self.edge_cpiview.isChecked()))

    def _edge_batch_records(self):
        """Union of ALL selected saved batches, de-duplicated by stable particle ID."""
        names = tuple(n for n in self._selected_batch_names() if n in self.saved_sets)
        if not names or not self.catalog:
            return (), []
        by_id = self.catalog._by_id
        ids = dict.fromkeys(pid for name in names for pid in self.saved_sets[name].get("particle_ids", ())
                            if pid in by_id)
        return names, [by_id[pid] for pid in ids]

    def _open_edge_batch(self):
        names, _ = self._edge_batch_records()
        if not names:
            self._toast("Select one or more saved batches in Review → Flag history first", 6000)
            return
        self.viewer_tabs.setCurrentWidget(self.edge_tab)
        if not self._edge_trials:
            self._edge_add_stored()

    def _edge_batch_changed(self):
        """Selection is only the source for NEW trials; existing trials remain untouched."""
        if not hasattr(self, "edge_batch_name"):
            return
        names, records = self._edge_batch_records()
        self.edge_batch_name.setText((f"Selected for NEXT trial: {len(names)} batch(es), "
                                      f"{len(records):,} distinct particles") if names else
                                     "Select one or more batches in Review → Flag history")
        if names:
            self.edge_status.setText("Click Add trial to snapshot these batches and the displayed settings. "
                                     "Previous curves are retained.")

    def _edge_add_stored(self):
        """Make stored data a selectable, removable curve, without touching MAT pixels."""
        names, records = self._edge_batch_records()
        if not names or not records:
            self.edge_status.setText("Select one or more saved batches in Review first.")
            return
        number = self._edge_next_id
        self._edge_next_id += 1
        label = f"#{number} Stored · existing processing · {len(names)} batch(es)"
        trial = {"id": number, "kind": "stored", "label": label, "names": names,
                 "records": tuple(records), "root": str(self.store.root) if self.store else "",
                 "options": None, "flags": "", "colour": "#ffd000", "shown": True,
                 "results": {}, "failures": {}, "state": "done"}
        self._edge_trials[number] = trial
        item = QListWidgetItem(label + " · saved")
        item.setData(Qt.UserRole, number)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setForeground(QColor(trial["colour"]))
        item.setToolTip(f"{', '.join(names)}\nStored boundary and measurements; no recalculation")
        self.edge_trials_list.addItem(item)
        self.edge_trials_list.setCurrentItem(item)
        self.edge_status.setText(f"Stored reference #{number}: {len(records):,} particles, no processing performed.")
        self._plot_edge_preview()

    def _start_edge_preview(self):
        """Enqueue a NEW trial; never replace curves already on the graph."""
        names, records = self._edge_batch_records()
        if not names or not records or self.store is None:
            self.edge_status.setText("Select saved batches containing loaded ROIs first.")
            return
        method = self.edge_algorithm.currentData()
        level, denoise, smooth = self.edge_level.value(), self.edge_denoise.currentData(), self.edge_smooth.value()
        cpiview = self.edge_cpiview.isChecked()
        shadow = method == "shadow/circle"
        auto = shadow and self.edge_radius_mode.currentData() == "adaptive"
        options = (edge_preview.shadow_options(self.edge_padding.value() if not auto else 8., denoise, smooth,
                                               radius_mode="adaptive" if auto else "fixed",
                                               max_padding_px=max(6., self.edge_padding.value()))
                   if shadow else edge_preview.preview_options(method, level, denoise, smooth, cpiview=cpiview))
        number = self._edge_next_id
        self._edge_next_id += 1
        label = (f"#{number} {dict((*edge_preview.METHODS, edge_preview.SHADOW_METHOD))[method]} · "
                 + ((f"adaptive ≤+{options.max_padding_px:g} px" if auto else f"+{options.padding_px:g} px")
                    if shadow else f"{level:g}")
                 + f" · {len(names)} batch(es)" + (" · CPIview" if cpiview and not shadow else ""))
        palette = ("#009ebc", "#bd4f84", "#e38a13", "#5d58af", "#168b63", "#9e4938", "#557999")
        trial = {"id": number, "kind": "shadow" if shadow else "edge", "label": label,
                 "names": names, "records": tuple(records),
                 "root": str(self.store.root), "options": options,
                 "flags": self.edge_cli.text(),
                 "colour": palette[(number - 1) % len(palette)], "shown": True,
                 "results": {}, "failures": {}, "state": "queued"}
        self._edge_trials[number] = trial
        item = QListWidgetItem(label + " (queued)")
        item.setData(Qt.UserRole, number)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setForeground(QColor(trial["colour"]))
        item.setToolTip(f"{', '.join(names)}\n" + (trial["flags"] if trial["flags"] else
                        "Stored curve: no additional processing flags") +
                        "\nPreview only – no files written")
        self.edge_trials_list.addItem(item)
        self.edge_trials_list.setCurrentItem(item)
        self._edge_pending.append(number)
        self._edge_start_next()
        self._plot_edge_preview()

    def _edge_start_next(self):
        if self._edge_worker is not None:
            return
        while self._edge_pending:
            number = self._edge_pending.pop(0)
            trial = self._edge_trials.get(number)
            if trial is None:
                continue
            trial["state"] = "running"
            worker = EdgePreviewWorker(trial["root"], trial["records"], trial["options"], number, self)
            worker.progress_changed.connect(self._edge_progress)
            worker.completed.connect(self._edge_completed)
            worker.finished.connect(self._edge_finished)
            self._edge_worker = worker
            self.edge_status.setText(f"Computing trial #{number} · {len(trial['records']):,} ROIs (no file writes)…")
            self._edge_update_item(number)
            worker.start()
            return

    def _edge_progress(self, done, total):
        worker = self.sender()
        if worker is self._edge_worker and worker.identity in self._edge_trials:
            self.edge_status.setText(f"Trial #{worker.identity}: {done:,} / {total:,} ROIs · temporary only")

    def _edge_completed(self, number, results, failures):
        trial = self._edge_trials.get(number)
        worker = self.sender()
        if (trial is None or worker is not self._edge_worker or
                worker.isInterruptionRequested() or self.store is None or
                str(self.store.root) != trial["root"]):
            return
        trial["results"], trial["failures"], trial["state"] = results, failures, "done"
        self._edge_update_item(number)
        n = len(results)
        self.edge_status.setText(f"#{number} · {n:,}/{len(trial['records']):,} valid measurements. "
                                 "Saved MAT, flags, references and project unchanged." +
                                 (f" First error: {next(iter(failures.values()))}" if not n and failures else ""))
        if self._edge_active_id == number:
            self._edge_update_inspection()
        self._gallery_dirty = True
        if self._current_tab() == "gallery":
            self.update_gallery()
        if self._current_tab() == "particle":
            self.refresh_particle()
        self._plot_edge_preview()

    def _edge_finished(self):
        worker = self.sender()
        if worker is not self._edge_worker:
            return
        self._edge_worker = None
        worker.deleteLater()
        self._edge_start_next()

    def _edge_update_item(self, number):
        trial = self._edge_trials.get(number)
        if trial is None:
            return
        for i in range(self.edge_trials_list.count()):
            item = self.edge_trials_list.item(i)
            if item.data(Qt.UserRole) == number:
                state = trial["state"]
                suffix = (" · saved, no calculation" if trial.get("kind") == "stored" else
                          f" · {len(trial['results'])}/{len(trial['records'])} valid" if state == "done"
                          else f" ({state})")
                item.setText(trial["label"] + suffix)
                return

    def _edge_select_trial(self, current=None, previous=None):
        number = current.data(Qt.UserRole) if current is not None else None
        self._edge_active_id = number if number in self._edge_trials else None
        self._edge_hover_trial_id = self._edge_active_id
        trial = self._edge_trials.get(self._edge_active_id)
        if trial is not None and trial["options"] is not None:
            opts = trial["options"]
            shadow = trial.get("kind") == "shadow"
            # Restore the selected trial's inputs without recomputing or altering
            # any of the other independent trials on the graph.
            values = ((self.edge_algorithm, "shadow/circle" if shadow else f"{opts.edge_mode}/{opts.outline}"),
                      (self.edge_denoise, opts.denoise), (self.edge_smooth, opts.smooth_sigma),
                      (self.edge_radius_mode, opts.radius_mode if shadow else "fixed"),
                      (self.edge_padding, (opts.max_padding_px if opts.radius_mode == "adaptive" else opts.padding_px)
                       if shadow else self.edge_padding.value()))
            if not shadow:
                values += ((self.edge_level, opts.edge_fraction if opts.edge_mode == "relative" else opts.edge_level),
                           (self.edge_cpiview, opts.cpiview))
            for widget, value in values:
                widget.blockSignals(True)
                if isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                else:
                    widget.setValue(float(value))
                widget.blockSignals(False)
            self._edge_settings_changed()
        self._hide_hover()
        self._edge_update_inspection()
        self._plot_edge_preview()
        self._gallery_dirty = True
        if self._current_tab() == "gallery":
            self.update_gallery()
        if self._current_tab() == "particle":
            self.refresh_particle()

    def _edge_update_inspection(self):
        trial = self._edge_trials.get(self._edge_active_id)
        self._edge_results = trial["results"] if trial else {}
        self._edge_failures = trial["failures"] if trial else {}
        if not trial or trial["state"] != "done":
            self.edge_diagnostics.setText("")
            return
        if trial.get("kind") == "stored":
            self.edge_diagnostics.setText(f"Stored measurements · {len(trial['records']):,} ROIs · "
                                          "saved outlines, no reprocessing")
            return
        differences = [v["deq"] - self.catalog._by_id[pid].deq_um
                       for pid, v in trial["results"].items() if pid in self.catalog._by_id
                       and math.isfinite(self.catalog._by_id[pid].deq_um)]
        ratios = [v["deq"] / self.catalog._by_id[pid].actual_length_um
                  for pid, v in trial["results"].items() if pid in self.catalog._by_id
                  and math.isfinite(self.catalog._by_id[pid].actual_length_um)
                  and self.catalog._by_id[pid].actual_length_um > 0]
        notes = [f"#{trial['id']} · {len(trial['results']):,}/{len(trial['records']):,} valid results",
                 f"ROI cut-off: {sum(v['cutoff_pct'] > 0 for v in trial['results'].values()):,}"]
        if trial.get("kind") == "shadow":
            notes.append("Shadow diameter is NOT calibrated to true size")
        if differences:
            notes.append(f"Median ΔDeq {np.median(differences):+.2f} µm")
        if ratios:
            notes.append(f"Median Deq / known size {np.median(ratios):.3f}")
        self.edge_diagnostics.setText(" · ".join(notes))

    def _edge_toggle_trial(self, item):
        trial = self._edge_trials.get(item.data(Qt.UserRole))
        if trial:
            trial["shown"] = item.checkState() == Qt.Checked
            self._plot_edge_preview()

    def _edge_remove_trial(self, number=None):
        if isinstance(number, bool) or number is None:
            number = self._edge_active_id
        trial = self._edge_trials.pop(number, None)
        if trial is None:
            return
        self._edge_pending = [x for x in self._edge_pending if x != number]
        if self._edge_worker is not None and self._edge_worker.identity == number:
            self._edge_worker.requestInterruption()
        for index in range(self.edge_trials_list.count()):
            if self.edge_trials_list.item(index).data(Qt.UserRole) == number:
                self.edge_trials_list.takeItem(index)
                break
        self._edge_select_trial(self.edge_trials_list.currentItem())
        self._plot_edge_preview()

    def _edge_show_all(self, visible):
        for trial in self._edge_trials.values():
            trial["shown"] = visible
        self.edge_trials_list.blockSignals(True)
        for i in range(self.edge_trials_list.count()):
            self.edge_trials_list.item(i).setCheckState(Qt.Checked if visible else Qt.Unchecked)
        self.edge_trials_list.blockSignals(False)
        self._plot_edge_preview()

    def _clear_edge_preview(self):
        self._edge_pending.clear()
        self._edge_trials.clear()
        self._edge_saved_outline_cache.clear()
        if self._edge_worker is not None:
            self._edge_worker.requestInterruption()
        if hasattr(self, "edge_trials_list"):
            self.edge_trials_list.clear()
        self._edge_active_id = None
        self._edge_hover_trial_id = None
        self._edge_results, self._edge_failures = {}, {}
        self._hide_hover()
        if hasattr(self, "edge_status"):
            self.edge_status.setText("All temporary trials discarded. Saved measurements unchanged.")
            self.edge_diagnostics.setText("")
            self._plot_edge_preview()
        if self._current_tab() == "particle":
            self.refresh_particle()
        self._gallery_dirty = True

    def _edge_trial_result(self, particle_id, number=None):
        if number is None:
            number = self._edge_active_id
        trial = self._edge_trials.get(number)
        return trial["results"].get(particle_id) if trial and trial["state"] == "done" else None

    def _edge_x_value(self, record):
        mode = self.edge_xaxis.currentData()
        if mode == "stage_x":
            return record.stage_x
        if mode == "stage_z":
            return record.stage_y
        if not (math.isfinite(record.stage_x) and math.isfinite(record.stage_y)):
            return float("nan")
        reference = self._draft_reference_for(record)
        if reference is None:
            return float("nan")
        d = signed_plane_distance(record.stage_x, record.stage_y, reference, self.plane_normal.value())
        if mode == "plane_zd":
            L = record.actual_length_um
            return WAVELENGTH_UM * d * 1000.0 / (L / 2.0) ** 2 if math.isfinite(L) and L > 0 else float("nan")
        return d

    @staticmethod
    def _edge_y_value(record, result, metric):
        if result is None:
            if metric == "shadow_deq":
                return record.shadow_deq_um
            if metric == "ratio_shadow":
                return (record.shadow_deq_um / record.actual_length_um
                        if math.isfinite(record.actual_length_um) and record.actual_length_um > 0 else float("nan"))
            if metric == "len":
                return record.length_um
            if metric == "wid":
                return record.width_um
            if metric == "deq":
                return record.deq_um if math.isfinite(record.deq_um) else record.length_um
            if metric in ("ratio_deq", "ratio_len"):
                L = (record.deq_um if math.isfinite(record.deq_um) else record.length_um) \
                    if metric == "ratio_deq" else record.length_um
                return (L / record.actual_length_um if math.isfinite(record.actual_length_um)
                        and record.actual_length_um > 0 else float("nan"))
            if metric == "focus":
                return record.focus
            if metric == "focus_T":
                return record.focus_T
            if metric == "cpv_focus":
                return record.cpv_focus
            if metric == "cpv_fwhm_px":
                return record.cpv_fwhm_px
            return float("nan")
        if metric in ("shadow_deq", "ratio_shadow"):
            raw = result.get("shadow_diameter", record.shadow_deq_um)
            if metric == "ratio_shadow":
                return (raw / record.actual_length_um if math.isfinite(record.actual_length_um)
                        and record.actual_length_um > 0 else float("nan"))
            return raw
        if metric in ("focus", "focus_T", "cpv_focus", "cpv_fwhm_px"):
            return edge_preview.trial_focus_value(record, result, metric)
        if metric in ("ratio_deq", "ratio_len"):
            L = result.get("deq") if metric == "ratio_deq" else result.get("len")
            return (L / record.actual_length_um if L is not None and math.isfinite(record.actual_length_um)
                    and record.actual_length_um > 0 else float("nan"))
        return float(result.get(metric, float("nan")))

    def _plot_edge_preview(self, *_):
        if not hasattr(self, "edge_figure"):
            return
        axes = (self.edge_size_ax, self.edge_focus_ax)
        for ax in axes:
            ax.clear()
            ax.grid(color="#e1e8ef", linewidth=.7)
        self._edge_hover_points = []
        active = [t for t in self._edge_trials.values() if t["shown"] and t["state"] == "done"]
        if not self._edge_trials:
            axes[0].text(.5, .5, "Select batch(es) in Review → Flag history, then Add stored or a trial.",
                         transform=axes[0].transAxes, ha="center", va="center")
        size_metric, focus_metric = self.edge_metric.currentData(), self.edge_focus_metric.currentData()
        # Correct using an INDEPENDENT reference optical ratio for this particle
        # type. Never correct a measurement using its own Dshadow/Ltrue: doing so
        # would just plot sqrt(area ratio), a constant, and hide all scatter.
        correction = (np.sqrt(self.edge_microscopy_area_ratio.value()) /
                      self.edge_reference_shadow_ratio.value())
        for trial in active:
            records = trial["records"]
            xs = np.array([self._edge_x_value(r) for r in records], dtype=float)
            for ax, metric in ((axes[0], size_metric), (axes[1], focus_metric)):
                # Focus is an INDEPENDENT stored measurement: even a shadow
                # aperture that fails must not remove a valid focus point.
                ys = np.array([self._edge_y_value(r, None if trial.get("kind") == "stored" else
                                          trial["results"].get(r.particle_id), metric)
                               if (trial.get("kind") == "stored" or r.particle_id in trial["results"]
                                   or metric in ("focus", "focus_T", "cpv_focus", "cpv_fwhm_px"))
                               else float("nan") for r in records], dtype=float)
                if metric == "ratio_shadow_cpi_area":
                    ys = np.array([self._edge_y_value(r, None if trial.get("kind") == "stored" else
                                    trial["results"].get(r.particle_id), "ratio_shadow") /
                                    np.sqrt(r.area_ratio_circle) if np.isfinite(r.area_ratio_circle)
                                    and r.area_ratio_circle > 0 else np.nan for r in records], dtype=float)
                if metric == "ratio_shadow_corrected":
                    ys = np.array([self._edge_y_value(r, None if trial.get("kind") == "stored"
                                          else trial["results"].get(r.particle_id), "ratio_shadow")
                                   if (trial.get("kind") == "stored" or r.particle_id in trial["results"])
                                   else float("nan") for r in records], dtype=float) * correction
                valid = np.isfinite(xs) & np.isfinite(ys)
                if not valid.any():
                    continue
                colour = trial["colour"]
                active_trial = trial["id"] == self._edge_active_id
                ax.scatter(xs[valid], ys[valid], s=18 if active_trial else 12,
                           color=colour, alpha=.73 if active_trial else .38,
                           edgecolors="none", label=trial["label"] if ax is axes[0] else None,
                           zorder=4 if active_trial else 2)
                self._edge_median(ax, xs[valid], ys[valid], colour,
                                  "Median " + str(trial["id"]) if ax is axes[0] else "")
                self._edge_hover_points.append((ax, [r for r, ok in zip(records, valid) if ok],
                                                xs[valid], ys[valid], trial["id"]))
        if self.edge_xaxis.currentData() in ("plane_signed", "plane_zd"):
            for ax in axes:
                ax.axvline(0, color="#d85519", linewidth=1, linestyle="--")
        if active:
            axes[0].legend(loc="best", fontsize=8, ncol=min(3, max(1, len(active))))
        axes[0].set_ylabel(self.edge_metric.currentText() +
                           ("" if "ratio" in size_metric else " [µm]"))
        if any(t.get("kind") == "shadow" for t in active):
            axes[0].set_title("Shadow: signed deficit; geometric correction is reference-based (not extinction)", fontsize=10)
        axes[1].set_ylabel(self.edge_focus_metric.currentText() +
                           (" (stored; not recalculated)" if focus_metric in ("focus", "focus_T") else ""))
        axes[1].set_xlabel(self.edge_xaxis.currentText())
        self.edge_canvas.draw_idle()

    @staticmethod
    def _edge_median(ax, xs, ys, colour, name):
        if len(xs) < 3:
            return
        trace = median_trace(xs, ys)
        if len(trace["x"]) >= 2:
            ax.plot(trace["x"], trace["median"], color=colour, linewidth=2,
                    label=name or None, zorder=5)

    def _edge_nearest_point(self, event):
        if event.inaxes is None or event.x is None or event.y is None:
            return None
        best, best_d2 = None, 13.0 ** 2
        for ax, records, xs, ys, number in self._edge_hover_points:
            if event.inaxes is not ax or not len(xs):
                continue
            points = ax.transData.transform(np.column_stack((xs, ys)))
            distance = (points[:, 0] - event.x) ** 2 + (points[:, 1] - event.y) ** 2
            i = int(np.argmin(distance))
            if distance[i] < best_d2 or (distance[i] == best_d2 and number == self._edge_active_id):
                best, best_d2 = (records[i], number), float(distance[i])
        return best

    def _edge_mouse_move(self, event):
        if not self.hover_action.isChecked():
            return
        hit = self._edge_nearest_point(event)
        if hit is None:
            self._hide_hover()
        else:
            record, number = hit
            self._show_hover_record(record, trial_id=number)

    def _edge_mouse_click(self, event):
        hit = self._edge_nearest_point(event)
        if hit is None:
            return
        record, number = hit
        self._edge_hover_trial_id = number
        for i in range(self.edge_trials_list.count()):
            item = self.edge_trials_list.item(i)
            if item.data(Qt.UserRole) == number:
                self.edge_trials_list.setCurrentItem(item)
                break
        if event.button == 3:
            self._show_hover_record(record, trial_id=number)
            menu = QMenu(self)
            # Embed the actual ROI and this specific trial's contour in the
            # context menu. Explicit right-click may load one uncached MAT image;
            # ordinary hover stays asynchronous.
            image = self._thumb_image(record, allow_load=True)
            if image is not None:
                preview = QWidget(menu)
                preview_layout = QVBoxLayout(preview)
                preview_layout.setContentsMargins(7, 7, 7, 5)
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setPixmap(self._tested_hover_pixmap(image, record, size=190))
                preview_layout.addWidget(image_label)
                result = self._edge_trial_result(record.particle_id, number)
                trial = self._edge_trials.get(number)
                if trial is not None:
                    if trial.get("kind") == "stored":
                        text = f"Saved boundary · Deq {record.deq_um:.2f} µm"
                    elif trial.get("kind") == "shadow" and result is not None:
                        ratio = (float(result["shadow_diameter"]) / record.actual_length_um
                                 if np.isfinite(record.actual_length_um) and record.actual_length_um > 0
                                 else float("nan"))
                        text = (f"Shadow Deq {result['deq']:.2f} µm · "
                                f"Area ratio {self._format(record.area_ratio_circle, 3)} · "
                                f"Dshadow/Ltrue {self._format(ratio, 3)} · "
                                f"(Dshadow/Ltrue)² {self._format(ratio ** 2, 3)} · "
                                f"ROI outside {100*result.get('aperture_missing_fraction', 0):.1f}% "
                                "(T=1; not an edge)")
                    elif result is not None:
                        text = f"Trial #{number}: Deq {result['deq']:.2f} µm · F(T) {result['focus_T']:.3g}"
                    else:
                        text = "No valid trial measurement"
                    detail = QLabel(text)
                    detail.setAlignment(Qt.AlignCenter)
                    preview_layout.addWidget(detail)
                preview_action = QWidgetAction(menu)
                preview_action.setDefaultWidget(preview)
                menu.addAction(preview_action)
                menu.addSeparator()
            trial = self._edge_trials.get(number)
            if trial is not None and trial.get("kind") == "edge":
                profile_action = menu.addAction("Transmittance profiles / threshold…")
                profile_action.triggered.connect(lambda: self._edge_show_profile(record, number))
            if trial is not None and trial.get("kind") == "shadow":
                integral_action = menu.addAction("Cumulative shadow integral vs circle radius…")
                integral_action.triggered.connect(lambda: self._edge_show_integral_profile(record, number))
            action = menu.addAction(f"Inspect ROI with #{number} outline / circle…")
            action.triggered.connect(lambda: self.set_current(record.particle_id, open_viewer=True))
            menu.addSeparator()
            trial = self._edge_trials.get(number)
            if trial:
                hide = menu.addAction(f"Hide trial #{number} from plot")
                hide.triggered.connect(lambda: self._edge_set_visibility(number, False))
                remove = menu.addAction(f"Remove trial #{number}")
                remove.triggered.connect(lambda: self._edge_remove_trial(number))
            self._hide_hover()
            menu.exec(QCursor.pos())
        elif event.button == 1:
            self.set_current(record.particle_id)

    def _edge_show_integral_profile(self, record, number):
        """Show integral convergence without modifying trial results or the MAT file."""
        trial = self._edge_trials.get(number)
        result = self._edge_trial_result(record.particle_id, number)
        if not trial or trial.get("kind") != "shadow" or not result or self.store is None:
            return
        try:
            from .data import _struct_list
            from proc.shadowCircle import cumulative_shadow
            from proc.edgeDetect import transmittance, EdgeOptions
            image, background = edge_preview.load_roi_for_preview(self.store.root, record)
            opt = trial["options"]
            T = transmittance(image, background, EdgeOptions(
                method="transmittance", pixel_um=record.pixel_um,
                denoise=opt.denoise, smooth_sigma=opt.smooth_sigma,
                border_normalise=True, bg_check=False))
            radii, integral, coverage, _ = cumulative_shadow(
                T, result["aperture_centre"], result["aperture_radius_px"] + 8)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot show shadow convergence", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"ROI {record.roi_index} · signed shadow integral")
        dialog.resize(740, 510)
        layout = QVBoxLayout(dialog)
        fig = Figure(figsize=(7, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(radii, integral, color=trial["colour"], label="Cumulative signed deficit")
        ax.axvline(result["aperture_radius_px"], color=trial["colour"], ls="--",
                   label="Selected aperture")
        ax.set(xlabel="Radius of integration circle [pixels]", ylabel="Signed shadow area [pixels²]")
        ax2 = ax.twinx()
        ax2.plot(radii, coverage, color="gray", ls=":", alpha=.6, label="ROI coverage")
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Observed fraction of each annulus", color="gray")
        ax.legend(loc="best")
        layout.addWidget(FigureCanvasQTAgg(fig), 1)
        layout.addWidget(QLabel(
            f"Dshadow = {result['shadow_diameter']:.2f} µm; aperture = "
            f"{result['aperture_radius_px']:.1f} px; mode = {result.get('radius_mode','fixed')}; "
            f"observed circle = {result.get('aperture_coverage', 1.):.1%}; "
            f"outside ROI = {result.get('aperture_missing_fraction', 0):.1%}. "
            "Unobserved pixels are treated as T=1 (zero deficit). "
            "Grey annular coverage shows where cropping could hide real shadow or "
            "produce a false plateau; adaptive selection requires adequate coverage."))
        dialog.exec()

    def _edge_show_profile(self, record, number):
        """Read exactly one original ROI on demand; diagnostic only, no writes."""
        trial = self._edge_trials.get(number)
        result = self._edge_trial_result(record.particle_id, number)
        if trial is None or trial.get("kind") != "edge" or result is None or self.store is None:
            return
        try:
            import scipy.io as sio
            from .data import _struct_list
            image, background = edge_preview.load_roi_for_preview(self.store.root, record)
            from dataclasses import replace
            opts = replace(trial["options"], pixel_um=float(record.pixel_um))
            profile = edge_preview.transmittance_profile(image, background, opts, result["boundary"])
        except Exception as exc:
            QMessageBox.warning(self, "Cannot display transmittance profile", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"ROI {record.roi_index} · trial #{number} transmittance")
        dialog.resize(720, 520)
        layout = QVBoxLayout(dialog)
        figure = Figure(figsize=(7, 4), constrained_layout=True)
        ax = figure.add_subplot(111)
        for name, values in (("Horizontal", profile["horizontal"]), ("Vertical", profile["vertical"])):
            xs = np.arange(len(values)) - profile["centre"][1 if name == "Horizontal" else 0]
            ax.plot(xs, values, label=name, alpha=.8)
        threshold = result["edge_level_used"]
        ax.axhline(1., color="gray", linestyle=":", label="Unobstructed T = 1")
        ax.axhline(threshold, color="red", linestyle="--", label=f"Trial threshold T = {threshold:.3f}")
        ax.set(xlabel="Position relative to particle centre (pixels)", ylabel="Normalised transmittance")
        ax.legend(loc="best")
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas, 1)
        layout.addWidget(QLabel(f"Boundary: {trial['label']} · minimum T = {result['T_min']:.3f} · "
                                f"ROI border median T = {profile['border_median']:.3f} · "
                                f"measured Deq = {result['deq']:.2f} µm"))
        layout.addWidget(QLabel("Two orthogonal centre profiles are diagnostic slices; the radial detector "
                                "samples many rays. Stored files and particle measurements are unchanged."))
        dialog.exec()

    def _edge_set_visibility(self, number, shown):
        trial = self._edge_trials.get(number)
        if trial is None:
            return
        trial["shown"] = shown
        for i in range(self.edge_trials_list.count()):
            item = self.edge_trials_list.item(i)
            if item.data(Qt.UserRole) == number:
                item.setCheckState(Qt.Checked if shown else Qt.Unchecked)
                break
        self._plot_edge_preview()

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
        sets_page = make_tab("Best focus")
        filter_page = make_tab("Filters")
        advanced_page = make_tab("Advanced")
        self.control_tabs.setTabVisible(4, False)
        layout.addWidget(self.control_tabs, 1)
        self._build_review_page(review_page)
        self._build_labels_page(labels_page)
        self._build_sets_page(sets_page)
        for control in (self.best_x_set, self.best_z_set, self.plane_normal):
            control.valueChanged.connect(lambda _: self._plot_edge_preview())
        self._build_filter_page(filter_page)
        self._build_advanced_page(advanced_page)
        return panel

    def _build_review_page(self, page):
        instrument = QGroupBox("Calibrated instrument")
        grid = QGridLayout(instrument)
        grid.addWidget(QLabel("CPI"), 0, 0)
        self.cpi_type = QComboBox()
        self.cpi_type.addItem("(detect from data)", "")
        for key, text in CPI_DISPLAY.items():
            self.cpi_type.addItem(text, key)
        self.cpi_type.addItem("Other", "other")
        grid.addWidget(self.cpi_type, 0, 1)
        grid.addWidget(QLabel("Name / serial"), 1, 0)
        self.cpi_id = QLineEdit()
        self.cpi_id.setPlaceholderText("e.g. FAAM CPI-3V s/n 12")
        grid.addWidget(self.cpi_id, 1, 1)
        self.cpi_detected = QLabel("")
        self.cpi_detected.setWordWrap(True)
        self.cpi_detected.setTextFormat(Qt.RichText)
        self.cpi_detected.setObjectName("hint")
        grid.addWidget(self.cpi_detected, 2, 0, 1, 2)
        self.cpi_type.currentIndexChanged.connect(lambda _: self._instrument_changed())
        self.cpi_id.editingFinished.connect(self._instrument_changed)
        page.addWidget(instrument)
        progress = QGroupBox("Progress (all loaded particles)")
        grid = QGridLayout(progress)
        grid.setHorizontalSpacing(4)
        self.count_labels = {}
        rows = (("undecided", "● Unreviewed", STATUS_COLOUR["undecided"]),
                ("keep", "● Accepted", STATUS_COLOUR["keep"]),
                ("reject", "✕ Rejected", STATUS_COLOUR["reject"]),
                ("selected", "◉ Selected", SELECTED_COLOUR))
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
            "<li>On Focus / Length: <b>Ctrl/⌘-drag</b> round particles that are fine but don't belong "
            "in <i>this</i> batch (they stay unreviewed); use the <b style='color:#e03b3b'>✕ Reject</b> box "
            "for debris / unusable ones.</li>"
            "<li><b>Accept selection</b> (K) → <b style='color:#2b9a48'>green</b>, saved, and skipped by "
            "new boxes.</li>"
            "<li>Check images in the Gallery: <b>Del</b> leaves a particle out of this batch, <b>R</b> rejects "
            "debris.</li>"
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
        self.batch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.batch_list.setToolTip("Click a batch to show it; Ctrl/⌘- or Shift-click to show several together. "
                                   "Select batch and Change label act on all chosen batches.")
        self.batch_list.itemDoubleClicked.connect(lambda _: self.select_batch())
        self.batch_list.itemClicked.connect(lambda _: self.batch_clicked())   # incl. Ctrl/Shift-click
        hist.addWidget(self.batch_list)
        row = QHBoxLayout()
        row.addWidget(self._button("Select batch", self.select_batch, None,
                                   "Make the particles of this batch the selection"))
        row.addWidget(self._button("Change label…", self.relabel_batch_dialog, None,
                                   "Give every particle in this batch a different label (and its true size)"))
        row.addWidget(self._button("Undo batch…", self.remove_classified_batch, "neutral",
                                   "Return this batch's particles to unreviewed (unless changed since)"))
        hist.addLayout(row)
        hist.addWidget(self._button("Export selected batch(es) to NetCDF…", self.export_batches_netcdf,
                                    None, "Export original ROIs and matching saved backgrounds, label true lengths, "
                                    "stage X/Z and signed object-plane distance. Read-only; Ctrl/Shift for several batches."))
        row = QHBoxLayout()
        row.addWidget(self._button("Save Xbest/Zbest to batch", self.save_batch_reference, "primary",
                                   "Give this batch its own best focus: the Xbest/Zbest currently in the "
                                   "Best focus tab. Later batches without their own X/Z inherit it."))
        row.addWidget(self._button("Clear", self.clear_batch_reference, None,
                                   "Remove the batch's own X/Z; it then borrows from the nearest calibrated batch"))
        row.addWidget(self._button("Test edges →", self._open_edge_batch, None,
                                   "Compare edge algorithms on all selected batches (Ctrl/Shift-click); "
                                   "no measurements are saved"))
        hist.addLayout(row)
        hist.addWidget(self._button("Set batch X/Z from their focus peaks (d = 0 at max focus)…", self.fill_batch_references,
                                    None, "For accepted batches with no Xbest/Zbest yet (e.g. from an older "
                                    "project): use each batch's own median focus peak"))
        self.batch_ref_label = QLabel("")
        self.batch_ref_label.setWordWrap(True)
        self.batch_ref_label.setTextFormat(Qt.RichText)
        self.batch_ref_label.setObjectName("hint")
        hist.addWidget(self.batch_ref_label)
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
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(self._label_context_menu)
        self.label_list.currentItemChanged.connect(lambda *_: self._label_selected())
        col.addWidget(self.label_list)
        row = QHBoxLayout()
        row.addWidget(self._button("＋ New…", self.new_label, "primary"))
        row.addWidget(self._button("Edit / resize…", self.edit_label, None,
                                   "Change name, type or true size; applies to every dataset"))
        row.addWidget(self._button("Delete…", self.delete_label))
        col.addLayout(row)
        col.addWidget(self._button("🔬 Microscope / SEM reference images…", self.open_label_microscopy, "primary",
                                    "Open multiple images for the chosen label; mark the object, scale and dimensions"))
        col.addWidget(self._button("Export SEM / optical figure panel…", self.open_microscopy_panel, None,
                                    "Choose labels and a SEM / optical image for each, then export a consistent PNG figure"))
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
        lib_box = QGroupBox("Shared reference library (labels + microscope / SEM images)")
        lib = QVBoxLayout(lib_box)
        self.label_path = QLabel()
        self.label_path.setWordWrap(True)
        self.label_path.setObjectName("hint")
        self.label_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lib.addWidget(self.label_path)
        lib.addWidget(self._button("Choose reference library folder…", self.choose_label_library, None,
                                   "Select a folder containing labels.json and microscopy/; the choice is remembered"))
        lib.addWidget(self._button("Use CPI_cals/ReferenceLibrary automatically", self.use_automatic_label_library,
                                   None, "Locate one common library beside CPICalibration and ICEAnalogues"))
        page.addWidget(lib_box)
        page.addStretch(1)

    def _build_sets_page(self, page):
        box = QGroupBox("Best focus (Xbest / Zbest)")
        col = QVBoxLayout(box)
        col.setSpacing(4)
        note = QLabel("This is the orange line on Focus vs Stage X / Z. It follows the selected cluster's own "
                      "focus peak; drag it or type here to override. <b>Accept (K)</b> saves it with the new "
                      "batch. To change an existing batch, click it in <b>Review → Flag history</b>, adjust, "
                      "and save.")
        note.setWordWrap(True)
        note.setTextFormat(Qt.RichText)
        note.setObjectName("hint")
        col.addWidget(note)
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
            spin.valueChanged.connect(lambda _value: self._reference_edited())
            spin.setToolTip("Original micrometer stage position at maximum focus")
        col.addLayout(reference_row)
        self.norm_peak_cb = QCheckBox("Find peaks with focus ÷ local illumination")
        self.norm_peak_cb.setToolTip("Laser power and the beam profile scale the focus metric. Tick to find each "
                                     "cluster's best focus from illumination-normalised focus (recommended if "
                                     "illumination varied during the scan).")
        self.norm_peak_cb.toggled.connect(lambda _on: (self._auto_place_reference(force=True), self.update_plots()))
        col.addWidget(self.norm_peak_cb)
        col.addWidget(self._button("Snap to the selection's focus peak", self.snap_reference_to_median_peak, None,
                                   "Put Xbest/Zbest at the peak of the selected cluster's median focus "
                                   "(or of all particles, if nothing is selected)"))
        self.active_set_label = QLabel("")
        self.active_set_label.setWordWrap(True)
        self.active_set_label.setTextFormat(Qt.RichText)
        self.active_set_label.setObjectName("hint")
        col.addWidget(self.active_set_label)
        self.save_to_batch_button = self._button("Save Xbest/Zbest to batch", self.save_batch_reference, "primary",
                                                 "Save these values as the clicked batch's own best focus")
        col.addWidget(self.save_to_batch_button)
        self.set_radius_label = QLabel("")
        self.set_radius_label.setWordWrap(True)
        self.set_radius_label.setObjectName("hint")
        col.addWidget(self.set_radius_label)
        page.addWidget(box)

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
        page.addWidget(plane)

        timeline_box = QGroupBox("Batch best-focus values")
        timeline_layout = QVBoxLayout(timeline_box)
        self.timeline_label = QLabel("")
        self.timeline_label.setWordWrap(True)
        self.timeline_label.setTextFormat(Qt.RichText)
        self.timeline_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        timeline_layout.addWidget(self.timeline_label)
        page.addWidget(timeline_box)
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
                           ("x", "Image x [px]"), ("y", "Image y [px]"), ("ratio", "Meas./true length"),
                           ("illum", "Local illumination"), ("cutoff", "Cut off by ROI edge [%]")):
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
        windows_box = QGroupBox("Saved time windows")
        wl = QVBoxLayout(windows_box)
        hint = QLabel("Name parts of a dataset by time — e.g. the 15, 40 and 100 µm bead runs — and switch "
                      "between them here or with <b>Window</b> above the plots. Saved in the project.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#516579;")
        wl.addWidget(hint)
        self.window_list = QListWidget()
        self.window_list.setMinimumHeight(110)
        self.window_list.itemDoubleClicked.connect(
            lambda item: self.show_time_window(item.data(Qt.UserRole)))
        wl.addWidget(self.window_list)
        grid = QGridLayout()
        for i, (text, slot, tip) in enumerate((
                ("Show", lambda: self.show_time_window(self._chosen_window()), "Filter to the chosen window"),
                ("All times", lambda: self.show_time_window(None), "Remove the time filter"),
                ("Suggest from gaps…", self.suggest_time_windows,
                 "One window per run, split where acquisition paused"),
                ("From selection…", self.time_window_from_selection,
                 "A window spanning the selected particles' times"),
                ("From time filter…", self.time_window_from_filter, "A window from the time filter above"),
                ("Edit…", self.edit_time_window, "Rename, change times or label"),
                ("Delete", self.delete_time_window, "Delete the chosen window (particles are not affected)"))):
            grid.addWidget(self._button(text, slot, "primary" if text == "Show" else None, tip), i // 2, i % 2)
        wl.addLayout(grid)
        page.addWidget(windows_box)
        page.addStretch(1)

    # ------------------------------------------------------- time windows --
    def _chosen_window(self):
        item = self.window_list.currentItem() if hasattr(self, "window_list") else None
        return item.data(Qt.UserRole) if item is not None else None

    def _window(self, name):
        return next((w for w in self.time_windows if w["name"] == name), None)

    def _unique_window_name(self, name, ignore=None):
        base, k, out = name, 2, name
        while any(w["name"] == out and w is not ignore for w in self.time_windows):
            out, k = f"{base} ({k})", k + 1
        return out

    def _refresh_time_windows(self):
        times = np.array([r.time_matlab for r in self.catalog.records], float) if self.catalog else np.array([])
        self.window_list.clear()
        for w in self.time_windows:
            try:
                n = int(in_window(times, w).sum())
            except ValueError:
                n = 0
            text = f"{w['name']}   {w['start'][11:]} – {w['end'][11:]}   ({n:,} particles)"
            if w["start"][:10] != w["end"][:10] or not self.catalog:
                text = f"{w['name']}   {w['start']} – {w['end']}   ({n:,} particles)"
            if w.get("label"):
                text += f"   🏷 {w['label']}"
            item = QListWidgetItem(("▶ " if w["name"] == self.active_time_window else "") + text)
            item.setData(Qt.UserRole, w["name"])
            item.setToolTip(f"{w['start']} → {w['end']}")
            self.window_list.addItem(item)
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        self.window_combo.addItem("All times", None)
        for w in self.time_windows:
            self.window_combo.addItem(w["name"], w["name"])
        self.window_combo.setCurrentIndex(max(0, self.window_combo.findData(self.active_time_window)))
        self.window_combo.blockSignals(False)

    def show_time_window(self, name):
        window = self._window(name) if name else None
        if window is None:
            self.active_time_window = None
            self.time_filter_checkbox.setChecked(False)
            self.apply_filters()
            self._refresh_time_windows()
            self._toast("Showing all times.")
            return
        self.active_time_window = window["name"]
        self._restore_date(self.time_low, window["start"])
        self._restore_date(self.time_high, window["end"])
        self.time_filter_checkbox.setChecked(True)
        self.apply_filters()
        note = ""
        if window.get("label") and self.accept_label.findData(window["label"]) >= 0:
            self.accept_label.setCurrentIndex(self.accept_label.findData(window["label"]))
            note = f" · Accept as: {window['label']}"
        self._refresh_time_windows()
        self._toast(f"Window '{window['name']}': {len(self._base_indices):,} particles shown{note}.")

    def _add_window(self, window, edit=True):
        if edit:
            dialog = TimeWindowDialog(self, window, sorted(self._label_names()))
            if dialog.exec() != QDialog.Accepted:
                return None
            window = dialog.value()
        if window["end"] < window["start"]:
            QMessageBox.warning(self, "Time window", "The end is before the start.")
            return None
        window["name"] = self._unique_window_name(window["name"])
        self.time_windows.append(window)
        self.time_windows.sort(key=lambda w: w["start"])
        clash = [pair for pair in overlapping(self.time_windows) if window["name"] in pair]
        self._refresh_time_windows()
        self.save_project_if_open()
        if clash:
            self._toast(f"Note: '{window['name']}' overlaps " + ", ".join(b if a == window['name'] else a
                                                                         for a, b in clash) + ".")
        return window

    def _label_names(self):
        names = getattr(self.labels, "names", None)
        if callable(names):
            return list(names())
        return [self.accept_label.itemData(i) for i in range(self.accept_label.count())
                if self.accept_label.itemData(i) not in (None, "", "__new__")]

    def time_window_from_selection(self):
        recs = [self.catalog._by_id[p] for p in self.selected_ids if self.catalog and p in self.catalog._by_id]
        times = [r.time_matlab for r in recs if math.isfinite(r.time_matlab)]
        if not times:
            QMessageBox.information(self, "Time window", "Select some particles first (e.g. a box on a "
                                    "time plot, or a batch).")
            return None
        labels = {r.label for r in recs if r.label}
        label = labels.pop() if len(labels) == 1 else self._accept_label_name()
        return self._add_window(make_window(label or f"Window {len(self.time_windows) + 1}", min(times), max(times),
                                            label=label))

    def time_window_from_filter(self):
        return self._add_window({"name": f"Window {len(self.time_windows) + 1}", "label": "",
                                 "start": self._date_text(self.time_low), "end": self._date_text(self.time_high)})

    def suggest_time_windows(self, *_, gap_minutes=None):
        if not self.catalog or not self.catalog.records:
            return []
        if gap_minutes is None:
            gap_minutes, ok = QInputDialog.getDouble(
                self, "Suggest time windows", "Start a new window after a pause in acquisition longer than "
                "(minutes):", 5.0, 0.1, 1440.0, 1)
            if not ok:
                return []
        found = suggest_windows([r.time_matlab for r in self.catalog.records], gap_minutes)
        existing = {(w["start"], w["end"]) for w in self.time_windows}
        added = []
        for w in found:
            if (w["start"], w["end"]) in existing:
                continue
            w["name"] = self._unique_window_name(w["name"])
            self.time_windows.append(w)
            added.append(w)
        self.time_windows.sort(key=lambda w: w["start"])
        self._refresh_time_windows()
        self.save_project_if_open()
        self._toast(f"Added {len(added)} window(s) split at pauses over {gap_minutes:g} min. Show each one, then "
                    "Edit… to name it (e.g. '15 µm beads') and give it a label.", 15000)
        return added

    def edit_time_window(self):
        window = self._window(self._chosen_window())
        if window is None:
            return
        dialog = TimeWindowDialog(self, window, sorted(self._label_names()))
        if dialog.exec() != QDialog.Accepted:
            return
        new = dialog.value()
        new["name"] = self._unique_window_name(new["name"], ignore=window)
        if self.active_time_window == window["name"]:
            self.active_time_window = new["name"]
        window.update(new)
        self.time_windows.sort(key=lambda w: w["start"])
        self._refresh_time_windows()
        self.save_project_if_open()
        if self.active_time_window == window["name"]:
            self.show_time_window(window["name"])

    def delete_time_window(self):
        name = self._chosen_window()
        if not name:
            return
        self.time_windows = [w for w in self.time_windows if w["name"] != name]
        if self.active_time_window == name:
            self.show_time_window(None)
        self._refresh_time_windows()
        self.save_project_if_open()

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
        for text, slot in (("New window", self.new_window),
                           ("New project…", self.create_project), ("Open project…", self.open_project_dialog),
                           ("Save project", self.save_project), ("Close project", self.close_project),
                           (None, None),
                           ("Add MAT files…", self.choose_files), ("Open processed directory…", self.choose_directory),
                           ("Add another dataset…", self.choose_add_directory),
                           ("Remove MAT files from project…", self.remove_sources_dialog),
                           ("Import calibration spreadsheet…", self.choose_truth_table),
                           ("Import stage positions CSV…", self.choose_position_file),
                           ("Reload stage positions CSV (after editing it)", self.reload_stage_positions),
                           ("Save flags as…", self.save_as), (None, None),
                           ("Merge projects into a viewing copy…", self.merge_projects_dialog),
                           ("Copy project onto reprocessed data…", self.copy_project_to_reprocessed_dialog),
                           ("Export several projects' particles to one CSV…", self.combine_projects_dialog)):
            if text is None:
                file_menu.addSeparator()
                continue
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        edit_menu = self.menuBar().addMenu("Edit")
        for text, slot, keys in (("Undo", self.undo, "Ctrl+Z"),
                                 ("Redo", self.redo, "Ctrl+Shift+Z"),
                                 ("Select all visible (pickable)", self.select_all_visible, "Ctrl+A"),
                                 ("Clear selection", self.clear_selection, None),
                                 ("Accept selection", self.accept_selection, None),
                                 ("Reject selection", self.reject_selection, None),
                                 ("Unflag selection", self.unflag_selection, None),
                                 ("Change label of selection…", self.relabel_selection_dialog, "L")):
            action = QAction(text, self)
            action.triggered.connect(slot)
            if keys:
                action.setShortcut(QKeySequence(keys))
            edit_menu.addAction(action)
        calib_menu = self.menuBar().addMenu("Calibration")
        fit_action = QAction("Fit size && focus model — this project…", self)
        fit_action.triggered.connect(self.open_sizing_dialog)
        calib_menu.addAction(fit_action)
        multi_action = QAction("Fit size && focus model — from several projects (runs)…", self)
        multi_action.triggered.connect(self.fit_from_projects_dialog)
        calib_menu.addAction(multi_action)
        compare_action = QAction("Compare processing runs (same particles, different edge methods)…", self)
        compare_action.triggered.connect(self.compare_processing_dialog)
        calib_menu.addAction(compare_action)
        tools_menu = self.menuBar().addMenu("Tools")
        measure_action = QAction("Measure images (full frames, ROIs, image files)…", self)
        measure_action.setShortcut(QKeySequence("Ctrl+M"))
        measure_action.triggered.connect(lambda: self.open_measure_window(None))
        tools_menu.addAction(measure_action)
        measure_roi_action = QAction("Measure on the current particle's ROI…", self)
        measure_roi_action.triggered.connect(lambda: self.open_measure_window(self.current_record()))
        tools_menu.addAction(measure_roi_action)
        view_menu = self.menuBar().addMenu("View")
        self.hover_action = QAction("Hover preview on plots", self)
        self.hover_action.setCheckable(True)
        self.hover_action.setChecked(True)
        view_menu.addAction(self.hover_action)
        self.show_reject_batches_action = QAction("Show reject batches from older projects in Flag history", self)
        self.show_reject_batches_action.setCheckable(True)
        self.show_reject_batches_action.toggled.connect(lambda _: self._refresh_saved_sets())
        view_menu.addAction(self.show_reject_batches_action)
        show_advanced = QAction("Show experimental calibration tools", self)
        show_advanced.setCheckable(True)
        show_advanced.toggled.connect(self._show_advanced)
        view_menu.addAction(show_advanced)

    def _project_mat_files(self):
        if self.catalog is None or self.store is None:
            return []
        files = {}
        for record in self.catalog.records:
            if record.source_file not in files:
                files[record.source_file] = str((self.store.root / record.source_file).resolve())
        return sorted(files.values())

    def open_measure_window(self, record=None):
        """Measuring window (measure_window.py): distances on full frames, ROIs and image files."""
        try:
            from .measure_window import MeasureWindow
        except ImportError:  # pragma: no cover
            from measure_window import MeasureWindow
        files = self._project_mat_files()
        window = getattr(self, "_measure_window", None)
        if window is None or not window.isVisible():
            window = MeasureWindow(files)
            self._measure_window = window
        else:
            for f in files:
                if window.file_combo.findData(f) < 0:
                    window.file_combo.addItem(Path(f).name, f)
        window.show()
        window.raise_()
        window.activateWindow()
        if record is not None and self.store is not None:
            window.show_roi(str((self.store.root / record.source_file).resolve()), record.roi_index)
        return window

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
        key("Ctrl+Y", self.redo)
        key(Qt.Key_Return, self._key_open)
        key(Qt.Key_Enter, self._key_open)
        key("B", lambda: self.boundary_checkbox.setChecked(not self.boundary_checkbox.isChecked()))
        key("S", lambda: self.selection_only_cb.setChecked(not self.selection_only_cb.isChecked()))
        key("+", lambda: self.z_spin.setValue(self.z_spin.value() + self.z_step_um))
        key("-", lambda: self.z_spin.setValue(self.z_spin.value() - self.z_step_um))

    # --------------------------------------------------- tabs & key dispatch --
    def _current_tab(self):
        widget = self.viewer_tabs.currentWidget()
        return {id(self.select_tab): "select", id(self.gallery_panel): "gallery",
                id(self.particle_tab): "particle", id(self.edge_tab): "edge"}.get(id(widget), "other")

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
        if tab == "edge":
            self._plot_edge_preview()
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
        if any(canvas.cancel_drag() for canvas in (self.spatial_canvas, self.plot_canvas)):
            self.statusBar().showMessage("Box cancelled", 3000)
            return
        if self._current_tab() == "gallery" and self.gallery.selectedItems():
            self.gallery.clearSelection()
            return
        if not self.selected_ids and (self.active_reference_set or self._shown_batches):
            self.active_reference_set = None
            self._shown_batches = []
            self._refresh_saved_sets()
            self.batch_list.clearSelection()
            self._after_change()
            self._toast("Showing all particles", 4000)
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

    def _focus_group(self):
        """(ids, description) the focus/size plots concentrate on.

        The selection while there is one; otherwise the batch just accepted or
        clicked in Flag history, so it stays on screen after Accept; else None.
        """
        if self.selected_ids:
            return self.selected_ids, f"the selection ({len(self.selected_ids):,})"
        shown = [n for n in self._shown_batches if n in self.saved_sets]
        if len(shown) > 1:
            ids = set().union(*(self.saved_sets[n].get("particle_ids", ()) for n in shown))
            return ids, f"{len(shown)} batches ({len(ids):,}) — Esc shows everything"
        name = self.active_reference_set
        if name and name in self.saved_sets:
            ids = set(self.saved_sets[name].get("particle_ids", ()))
            if ids:
                return ids, f"{name} ({len(ids):,}) — Esc shows everything"
        return None, "all particles"

    def _visible_indices(self, kind):
        """Indices drawn (and box-selectable) on the camera or metric plots."""
        records = self._records()
        shown = self._shown_statuses()
        selected = self.selected_ids
        group = None
        if kind == "metric" and self.selection_only_cb.isChecked():
            group = self._focus_group()[0]
        out = []
        for i in self._base_indices:
            r = records[i]
            if r.particle_id in selected:
                out.append(i)
            elif group is not None:
                if r.particle_id in group and r.status in shown:
                    out.append(i)
            elif r.status in shown:
                out.append(i)
        return out

    def _plot_position(self, record):
        if not (math.isfinite(record.stage_x) and math.isfinite(record.stage_y)):
            return float("nan")
        reference = self._draft_reference_for(record)
        if reference is not None:
            return calibrated_radial_position(record.stage_x, record.stage_y, reference)
        radius = math.hypot(record.stage_x, record.stage_y)
        segment = next((seg for seg in self.stage_segments
                        if seg["start"] <= record.time_matlab < seg["end"]), None)
        if segment is None:
            return float("nan")  # No best-focus reference: Zp is undefined.
        if "surface" in segment:
            best = predict_best_radius(segment["surface"], record.image_x_px, record.image_y_px)
            return float(best - radius)
        return segment["best_radius"] - radius

    def _metric_value(self, record, metric):
        if metric == "focus":
            return record.focus
        if metric == "focus_norm":
            return self._normalised_focus(record)
        if metric == "illum":
            return record.local_illum
        if metric == "focus_T":
            return record.focus_T
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
        if metric == "deq":
            return record.deq_um
        if metric == "shadow_deq":
            return record.shadow_deq_um
        if metric == "ratio_shadow":
            return (record.shadow_deq_um / record.actual_length_um
                    if math.isfinite(record.actual_length_um) and record.actual_length_um > 0
                    else float("nan"))
        if metric == "cutoff":
            return record.cutoff_pct
        if metric.startswith("cpv_"):
            if metric == "cpv_focus":
                return record.cpv_focus
            if metric == "cpv_focus_sd":
                return record.cpv_focus_sd
            if metric == "cpv_fwhm":
                return record.cpv_fwhm_px
            if metric == "cpv_distinct":
                return record.cpv_distinct
            if metric == "cpv_cutoff":
                return record.cpv_cutoff_pct
        if metric == "ratio_deq":
            return (record.deq_um / record.actual_length_um
                    if math.isfinite(record.actual_length_um) and record.actual_length_um > 0
                    else float("nan"))
        return float("nan")

    def _plot_y(self, record, metric=None):
        return self._metric_value(record, metric or self.plot_metric.currentData())

    def _normalised_focus(self, record):
        """Focus scaled to the typical illumination: focus × median(illum) / illum.

        The focus metric is an edge-gradient sharpness, so to first order it is
        proportional to the illumination contrast; this removes that factor.
        """
        illum = record.local_illum
        ref = self._illum_reference()
        if not (math.isfinite(illum) and illum > 1 and math.isfinite(ref)):
            return float("nan")
        return record.focus * ref / illum

    def _illum_reference(self):
        if getattr(self, "_illum_ref_cache", None) is None:
            values = [r.local_illum for r in self._records() if math.isfinite(r.local_illum) and r.local_illum > 1]
            self._illum_ref_cache = float(np.median(values)) if values else float("nan")
        return self._illum_ref_cache

    def _plot_x(self, record):
        mode = self.position_metric.currentData()
        if mode == "illum":
            return record.local_illum
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
            if mode == "plane_zd":
                L = record.actual_length_um
                if not (math.isfinite(L) and L > 0):
                    return float("nan")
                d = signed_plane_distance(record.stage_x, record.stage_y, reference, self.plane_normal.value())
                return WAVELENGTH_UM * d * 1000.0 / (L / 2.0) ** 2     # d in mm, L in µm; keeps the sign of d
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
        return self._reference_for(record)[0]

    def _reference_for(self, record, use_editor=True):
        """(reference, source, kind) for a particle.

        kind "editor": the Xbest/Zbest editor, previewed on the selection and the
        active set/batch; "own": its set's/batch's saved X/Z; "step": the most
        recently defined X/Z at or before its time; None: no reference at all.
        """
        pid = record.particle_id
        if use_editor and pid in self._active_members:          # the batch being edited
            return self._editor_reference(), "Xbest/Zbest editor", "editor"
        ref, conflict = self._reference_index.get(pid, (None, False))
        if ref is not None and not conflict:                     # saved with its batch: always used as saved
            return ref, self._own_ref_names.get(pid), "own"
        if use_editor and pid in self.selected_ids:              # a new cluster being chosen
            return self._editor_reference(), "Xbest/Zbest editor", "editor"
        self._reference_steps()  # builds groups + sessions too
        ref, name = nearest_group_reference(record.image_x_px, record.image_y_px,
                                            self._session_of.get(pid, -2), self._groups_cache or [])
        if ref is not None:
            return ref, name, "nearest"
        return effective_reference(pid, record.time_matlab, {}, self._reference_steps())

    def _editor_reference(self):
        if self._draft_reference_cache is None:
            self._draft_reference_cache = make_reference(self.best_x_set.value(), self.best_z_set.value())
        return self._draft_reference_cache

    def _has_own_reference(self, pid):
        ref, conflict = self._reference_index.get(pid, (None, False))
        return ref is not None and not conflict

    def _selection_without_own_reference(self):
        """Selected (non-rejected) particles that the editor X/Z would apply to."""
        if not self.catalog:
            return []
        by_id = self.catalog._by_id
        return [p for p in self.selected_ids
                if p in by_id and by_id[p].status != "reject"
                and (p in self._active_members or not self._has_own_reference(p))]

    def _reference_steps(self):
        if self._steps_cache is None:
            records = self._records()
            times = {r.particle_id: r.time_matlab for r in records}
            self._steps_cache = reference_steps(self.saved_sets, times)
            if self._session_of is None or len(self._session_of) != len(records):
                labels = session_labels([r.time_matlab for r in records])
                self._session_of = {r.particle_id: int(s) for r, s in zip(records, labels)}
            groups = []
            by_id = self.catalog._by_id if self.catalog else {}
            for name, entry in self.saved_sets.items():
                ref = validate_reference(entry.get("best_focus"))
                members = [by_id[p] for p in entry.get("particle_ids", ()) if p in by_id]
                if ref is None or not members:
                    continue
                sessions = [self._session_of.get(m.particle_id, -1) for m in members]
                groups.append(dict(name=name, reference=ref, session=max(set(sessions), key=sessions.count),
                                   cx=float(np.nanmedian([m.image_x_px for m in members])),
                                   cy=float(np.nanmedian([m.image_y_px for m in members]))))
            self._groups_cache = groups
        return self._steps_cache

    def _invalidate_reference_steps(self):
        self._steps_cache = None
        self._groups_cache = None
        if hasattr(self, "timeline_label"):
            self._refresh_timeline()

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
        if colour_by in ("ref_x", "ref_z"):
            key = "x_best" if colour_by == "ref_x" else "z_best"
            def own_value(r):
                ref, _, kind = self._reference_for(r, use_editor=False)
                return ref[key] if kind == "own" else float("nan")
            values = np.fromiter((own_value(r) for r in records), float, len(records))
            has = (und | acc) & np.isfinite(values)
            rest = (und | acc) & ~np.isfinite(values)
            if rest.any():
                ax.scatter(xs[rest], ys[rest], c=STATUS_COLOUR["undecided"], s=8, alpha=.35, linewidths=0,
                           rasterized=True, zorder=2)
            if has.any():
                mappable = ax.scatter(xs[has], ys[has], c=values[has], cmap="viridis", s=18, alpha=.9,
                                      edgecolors="#222", linewidths=.25, rasterized=True, zorder=3)
        elif colour_by in ("image_x", "image_y", "illum"):
            values = np.fromiter(((r.image_x_px if colour_by == "image_x" else
                                   r.image_y_px if colour_by == "image_y" else r.local_illum) for r in records),
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

    def _colourbar_label(self):
        return {"image_x": "Camera X [px]", "image_y": "Camera Y [px]", "illum": "Local illumination (ROI edge)",
                "ref_x": "Batch's own Xbest",
                "ref_z": "Batch's own Zbest"}.get(self.colour_by.currentData(), "")

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
                    label=self._colourbar_label())
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
                label=self._colourbar_label())
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
            if metric in ("focus", "focus_norm") and xmode not in ("time", "illum"):
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
                if metric in ("focus", "focus_norm") and xmode in ("stage_x", "stage_z"):
                    speak = trace_extremum(strace, self.peak_combo.currentData())
                    if speak is not None:
                        mark, = ax.plot([speak[0]], [speak[1]], marker="^", ms=11, color=SELECTED_COLOUR,
                                        markeredgecolor="white", ls="none", zorder=9.5,
                                        label=f"Selection's peak at {speak[0]:.5g}")
                        handles.append(mark)
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
            if xmode in ("plane_signed", "plane_zd") and label:
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
            text = {"new": f"<b>Drag</b> a box to select ({pick}; a box of only flagged particles takes them).",
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
                                 "then Best focus tab, or just Accept."}[self.plot_tool]
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
            self._load_image_now(record)
            self.set_current(pid)
            self.statusBar().showMessage(
                f"Inspecting ROI {record.roi_index} ({STATUS_NAME[record.status]}). "
                "Shift-click adds it, Ctrl/⌘-click toggles, double-click opens it.", 7000)

    def _load_image_now(self, record):
        """Explicit request (click): read the image even if its MAT file must be loaded."""
        if record.particle_id in self._thumbs or self.store is None:
            return
        if self.store._source != record.source_file:
            self.statusBar().showMessage(f"Loading images from {Path(record.source_file).name}…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
        try:
            image = self._thumb_image(record, allow_load=True)
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        if image is not None and self.hover.isVisible() and self.hover.particle_id == record.particle_id:
            self.hover.image.setPixmap(self._tested_hover_pixmap(image, record))

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
        fallback_note = ""
        if mode in ("new", "add") and inside and new == (set(self.selected_ids) if mode == "add" else set()):
            # Everything in the box is already flagged: the user clearly means
            # these particles, so take them rather than doing nothing.
            new, _ = apply_box(self.selected_ids, inside, mode, statuses, "any")
            flagged = {}
            for pid in inside:
                flagged[statuses[pid]] = flagged.get(statuses[pid], 0) + 1
            fallback_note = (" — all were already " + " / ".join(STATUS_NAME[k].lower() for k in sorted(flagged))
                             + ", so they were selected anyway (U unflags them)")
        region = (axis_info[0], x0, x1, y0, y1, *axis_info[1:], mode)
        if mode == "reject" and not to_reject and not self.selected_ids:
            # Nothing selected but a batch is on show (just accepted, or clicked in Flag
            # history): the box means "reject these members of that batch".
            name = self.active_reference_set
            if name and name in self.saved_sets:
                members = set(self.saved_sets[name].get("particle_ids", ()))
                to_reject = {pid for pid in inside & members if statuses.get(pid) != "reject"}
        if mode == "reject":
            if not to_reject:
                self.statusBar().showMessage("Reject box: nothing to reject inside the box", 6000)
                return
            self._push_undo(f"Reject {len(to_reject)} in box", status_ids=to_reject, sets=True)
            self._set_flags(to_reject, "reject", reason=f"box on {where}")
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
                + " — box only those to select them)") if skipped_by else fallback_note
        tip = ("  —  Shift-drag to add another area to this batch" if mode == "new" and canvas.kind == "camera"
               else "")
        self._change_selection(new, f"{verb} {count:,} particles on the {where}{note} · "
                                    f"{len(new):,} selected{tip}", push=True)

    def _on_hover(self, canvas, index, pixel):
        if not self.hover_action.isChecked() or self.catalog is None:
            return
        record = self._nearest_record(canvas, index, pixel, radius=9)
        if record is None:
            self._hide_hover()
        else:
            self._show_hover_record(record)

    def _edge_stored_outline(self, record, allow_load=False):
        """Read SAVED dat.foc boundaries, never run an edge detector on stored data."""
        pid = record.particle_id
        cache = self._edge_saved_outline_cache
        if pid in cache:
            cache.move_to_end(pid)
            return cache[pid]
        if self.store is None or (not allow_load and self.store._source != record.source_file):
            return None
        try:
            _, boundary = self.store.particle(record)
        except Exception:
            boundary = None
        cache[pid] = boundary
        if len(cache) > 400:
            cache.popitem(last=False)
        return boundary

    def _tested_hover_pixmap(self, image, record, size=150):
        """Draw the contour OR integration aperture in this dataset's curve colour."""
        trial = self._edge_trials.get(self._edge_hover_trial_id)
        if trial is None:
            return self._plain_pixmap(image, size)
        result = self._edge_trial_result(record.particle_id, self._edge_hover_trial_id)
        if trial.get("kind") == "stored":
            b = self._edge_stored_outline(record)
        else:
            b = result.get("boundary") if result is not None else None
        b = np.asarray(b if b is not None else [], dtype=float)
        shape = result.get("image_shape") if result is not None else None
        if shape is None and self.store is not None and self.store._source == record.source_file:
            try:
                orig, _ = self.store.particle(record)
                shape = orig.shape
            except Exception:
                pass
        if b.ndim != 2 or b.shape[1] != 2 or not shape or min(shape) <= 0:
            return self._plain_pixmap(image, size)
        base = QPixmap.fromImage(image)
        pm = base.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(trial["colour"]), 2.25)
        pen.setCosmetic(True)
        painter.setPen(pen)
        points = [QPointF(float(x) * pm.width() / shape[0], float(y) * pm.height() / shape[1])
                  for x, y in b if math.isfinite(x) and math.isfinite(y)]
        if len(points) >= 2:
            painter.setPen(QPen(QColor("#000000"), 4.5))
            painter.drawPolyline(QPolygonF(points))
            painter.setPen(pen)
            painter.drawPolyline(QPolygonF(points))
        painter.end()
        return pm

    def _show_hover_record(self, record, trial_id=None):
        if trial_id is None:
            trial_id = self._edge_active_id
        if (self.hover.isVisible() and self.hover.particle_id == record.particle_id
                and getattr(self.hover, "trial_id", None) == trial_id):
            self.hover.move(QCursor.pos() + QPoint(18, 14))
            return
        self.hover.particle_id = record.particle_id
        self.hover.trial_id = trial_id
        self._edge_hover_trial_id = trial_id
        tags = [f"<b style='color:{STATUS_COLOUR[record.status]}'>{STATUS_NAME[record.status]}</b>"]
        if record.particle_id in self.selected_ids:
            tags.append(f"<b style='color:{SELECTED_COLOUR}'>selected</b>")
        if record.label:
            label = self.labels.get(record.label)
            tags.append(f"<b style='color:{label.colour if label else '#495057'}'>{record.label}</b>"
                        + (f" ({label.size_text()})" if label else ""))
        html = (f"<span style='font-size:16px; font-weight:800'>{self._format(record.length_um, 0)} × "
                f"{self._format(record.width_um, 0)} µm <span style='font-size:11px'>(stored)</span></span><br>"
                f"{' · '.join(tags)}<br>Focus {self._format(record.focus, 1)} · illum. "
                f"{self._format(record.local_illum, 0)} · ROI {record.roi_index} · "
                f"<span style='color:#516579'>{Path(record.source_file).name} {record.time_iso[11:19]}</span>"
                + (f"<br>🗂 {self._batch_text(record)}" if self._groups_of.get(record.particle_id)
                   or record.status == "keep" else ""))
        trial = self._edge_trials.get(trial_id)
        result = self._edge_trial_result(record.particle_id, trial_id)
        if trial is not None and trial.get("kind") == "stored":
            html += (f"<br><b style='color:{trial['colour']}'>STORED #{trial_id} · saved boundary</b> "
                     f"(no reprocessing): Deq {self._format(record.deq_um, 2)} µm")
        elif result is not None and trial is not None:
            colour = trial["colour"]
            if trial.get("kind") == "shadow":
                ratio = (result['shadow_diameter'] / record.actual_length_um
                         if math.isfinite(record.actual_length_um) and record.actual_length_um > 0 else float('nan'))
                html += (f"<br><b style='color:{colour}'>TRIAL #{trial_id} · integration circle</b>: "
                         f"Dshadow {result['shadow_diameter']:.2f} µm · "
                         f"Dshadow/Ltrue {self._format(ratio, 3)}"
                         f" · (Dshadow/Ltrue)² {self._format(ratio ** 2, 3)}"
                         f"<br>CPI contour area ratio {self._format(record.area_ratio_circle, 3)}"
                         f"<br>Circle radius {result['aperture_radius_px']:.1f} px · "
                         f"fit RMS {result['circle_rms_px']:.2f} px (uncalibrated)")
            else:
                html += (f"<br><b style='color:{colour}'>TRIAL #{trial_id} · tested boundary</b>: "
                         f"Deq {result['deq']:.2f} · Feret {result['len']:.2f} µm"
                         f"<br>Focus {result['focus']:.3g} · Focus(T) {result['focus_T']:.3g}"
                         f"<br>Edge T={result['edge_level_used']:.3f} · "
                         f"cutoff={result['cutoff_pct']:.1f}%")
                if "cpv_focus" in result:
                    html += (f"<br>CPIview focus {result['cpv_focus']:.3g} · "
                             f"edge FWHM {result['cpv_fwhm_px']:.2f} px")
        image = self._thumbs.get(record.particle_id)
        self.hover.image.setFixedSize(150, 150)
        self.hover.show_at(QCursor.pos(), html,
                           self._tested_hover_pixmap(image, record) if image is not None else None)
        needs_saved_outline = (trial is not None and trial.get("kind") == "stored"
                               and record.particle_id not in self._edge_saved_outline_cache)
        if (image is None or needs_saved_outline) and self.store is not None:
            self._hover_pending = record
            if self.store._source == record.source_file:
                self._hover_timer.start(60)
            else:
                if image is None:
                    self.hover.image.setText("<span style='color:#8292a3;font-size:10px'>loading<br>image…</span>")
                self._hover_timer.start(250)

    def _hover_load_image(self):
        record = self._hover_pending
        if record is None or not self.hover.isVisible() or self.hover.particle_id != record.particle_id:
            return
        if self.store is not None and self.store._source != record.source_file:
            if self._mat_loader is None:  # one background load at a time
                self._mat_loader = MatLoader(self.store.root, record.source_file, self)
                self._mat_loader.loaded.connect(self._mat_loaded)
                self._mat_loader.start()
            return
        image = self._thumb_image(record, allow_load=False)
        if image is not None:
            self.hover.image.setPixmap(self._tested_hover_pixmap(image, record))

    def _mat_loaded(self, source_file, data):
        loader, self._mat_loader = self._mat_loader, None
        if loader is not None:
            loader.wait()
            loader.deleteLater()
        if data is None or self.store is None:
            if self.hover.isVisible():
                self.hover.image.setText("<span style='color:#c92a2a;font-size:10px'>image<br>unavailable</span>")
            return
        self.store._data, self.store._source = data, source_file
        record = self._hover_pending
        if record is not None and self.hover.isVisible() and self.hover.particle_id == record.particle_id:
            if record.source_file == source_file:
                self._hover_load_image()
            else:
                self._hover_timer.start(60)  # cursor moved on to another file meanwhile

    def _hide_hover(self):
        self._hover_timer.stop()
        if hasattr(self, "hover"):
            self.hover.hide()
            self.hover.particle_id = None
            self.hover.trial_id = None

    # ------------------------------------------------------- selection core --
    def _snapshot(self, label, status_ids=(), sets=False):
        """Everything one action may change: selection, the given particles' flags, groups."""
        records = self.catalog._by_id if self.catalog else {}
        return {"label": label, "selected": set(self.selected_ids), "regions": list(self.regions),
                "current": self.current_id,
                "statuses": {pid: (records[pid].status, records[pid].reject_reason, records[pid].label,
                                   records[pid].actual_length_um, records[pid].actual_width_um)
                             for pid in status_ids if pid in records},
                "sets": copy.deepcopy(self.saved_sets) if sets else None,
                "active_set": self.active_reference_set}

    def _push_undo(self, label, status_ids=(), sets=False):
        self.undo_stack.append(self._snapshot(label, status_ids, sets))
        del self.undo_stack[:-60]
        self.redo_stack.clear()   # a new action ends the redo chain

    def _restore(self, entry):
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

    def _step_history(self, source, target, verb):
        if not source:
            self.statusBar().showMessage(f"Nothing to {verb.lower()}", 4000)
            return
        entry = source.pop()
        # Record the present state of exactly what this entry touches, so it can be stepped back.
        target.append(self._snapshot(entry["label"], entry["statuses"].keys(), entry["sets"] is not None))
        del target[:-60]
        self._restore(entry)
        self._toast(f"{'↶' if verb == 'Undid' else '↷'} {verb}: {entry['label']}")

    def undo(self):
        self._step_history(self.undo_stack, self.redo_stack, "Undid")

    def redo(self):
        self._step_history(self.redo_stack, self.undo_stack, "Redid")

    def _change_selection(self, new_ids, message=None, push=True, current=None):
        if push:
            self._push_undo(message or "selection change")
        new_ids = set(new_ids)
        active = self.saved_sets.get(self.active_reference_set, {}) if self.active_reference_set else {}
        # Selecting (part of) the batch being edited is not a new cluster: the editor keeps the
        # batch's saved X/Z, or the user's unsaved edit, instead of jumping to the focus peak.
        within_active = bool(new_ids) and new_ids <= set(active.get("particle_ids", ()))
        if not (new_ids & self.selected_ids) and not within_active:
            self._ref_manual = False  # a different cluster: let the line follow its peak again
            self._ref_set_for_cluster = False
            if new_ids and self._shown_batches:
                self._shown_batches = []
            if new_ids and self.active_reference_set and not (new_ids <= set(
                    self.saved_sets.get(self.active_reference_set, {}).get("particle_ids", ()))):
                self.active_reference_set = None   # working on something new: release the shown batch
                self._refresh_saved_sets()
                self.batch_list.clearSelection()
        self.selected_ids = new_ids
        if not (within_active and (validate_reference(active.get("best_focus")) is not None or self._ref_manual)):
            self._auto_place_reference()
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
        if hasattr(self, "scope_label"):
            showing = self._focus_group()[1] if self.selection_only_cb.isChecked() else "all particles"
            self.scope_label.setText(f"Showing: {showing}")
        if hasattr(self, "label_list"):
            self._refresh_label_widgets(update_batches=False)   # batches refresh themselves when they change
        self._update_selection_bar()
        self._redraw_plots()
        self._gallery_dirty = True
        if self._current_tab() == "gallery":
            self.update_gallery()
        self.refresh_particle()

    def _toast(self, message, ms=9000):
        self.statusBar().showMessage(message, ms)

    def _update_save_ref_text(self):
        """Say exactly which Xbest/Zbest an Accept will save with the new batch."""
        if not hasattr(self, "save_ref_cb"):
            return
        cb = self.save_ref_cb
        x, z = self.best_x_set.value(), self.best_z_set.value()
        if not self.selected_ids:
            cb.setText("+ save Xbest/Zbest")
            tip = "Accepting saves the Xbest/Zbest shown on the line as the new batch's own best focus."
            colour = ""
        elif not self._selection_without_own_reference():
            cb.setText("Xbest/Zbest: kept from their batches")
            tip = ("Every selected particle already belongs to a batch with its own saved X/Z; accepting again "
                   "does not change it. Click a batch in Flag history to edit its X/Z.")
            colour = "color:#516579;"
        elif self._ref_set_for_cluster:
            cb.setText(f"+ save Xbest {x:.5g} · Zbest {z:.5g}")
            tip = (f"Accepting saves Xbest = {x:.6g}, Zbest = {z:.6g} (the orange line) as this batch's own best "
                   "focus. The line follows the selection's focus peak unless you move it.")
            colour = "color:#a34a00; font-weight:600;"
        else:
            cb.setText("+ Xbest/Zbest: no peak found — not saved")
            tip = ("No focus peak was found for this selection and the line hasn't been moved, so nothing will be "
                   "saved; the batch borrows from the nearest calibrated batch. Move the line to set it.")
            colour = "color:#a61e4d;"
        cb.setToolTip(tip)
        cb.setStyleSheet(colour)

    def _update_selection_bar(self):
        self._update_save_ref_text()
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
        self.redo_btn.setEnabled(bool(self.redo_stack))
        if self.redo_stack:
            self.redo_btn.setToolTip(f"Redo: {self.redo_stack[-1]['label']} (Ctrl+Shift+Z or Ctrl+Y)")

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
            if "committed" in self.count_labels:
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
        self._prune_batches(ids, status)

    def _prune_batches(self, ids, status):
        """A particle leaves review batches whose decision it no longer has.

        Rejecting or unflagging an accepted particle removes it from its
        accepted batch (and vice versa), so batch counts, "Select batch" and
        the batch's own X/Z only reflect what is still accepted.  Empty
        batches are removed.  Groups from older projects (sets) are kept.
        """
        ids = set(ids)
        changed = False
        for name in list(self.saved_sets):
            entry = self.saved_sets[name]
            description = entry.get("description", "")
            if not description.startswith("Review batch: ") or description.endswith(": " + status):
                continue
            members = entry.get("particle_ids", [])
            if ids.isdisjoint(members):
                continue
            entry["particle_ids"] = [p for p in members if p not in ids]
            changed = True
            if not entry["particle_ids"]:
                del self.saved_sets[name]
                if self.active_reference_set == name:
                    self.active_reference_set = None
        if changed:
            self._refresh_saved_sets()
            self.save_project_if_open()

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
            record.actual_width_um = label.width_um

    def _accept_label_name(self):
        data = self.accept_label.currentData() if hasattr(self, "accept_label") else None
        return data if data and data != "__new__" else ""

    def _flag_ids(self, ids, status, label, clear_selection=False, batch=True, reason=None):
        if not self.catalog:
            return 0
        ids = [pid for pid in dict.fromkeys(ids) if pid in self.catalog._by_id]
        if not ids:
            return 0
        batch = batch and status == "keep"     # rejects are plain flags (Ctrl+Z / Select rejected → Unflag undo them)
        self._push_undo(label, status_ids=ids, sets=True)  # batches may lose members
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
        with self._bulk():
            return self._accept_selection_impl()

    def _accept_selection_impl(self):
        ids = set(self.selected_ids)
        by_id = self.catalog._by_id if self.catalog else {}
        has_stage = any(p in by_id and math.isfinite(by_id[p].stage_x) and math.isfinite(by_id[p].stage_y)
                        for p in ids)
        self._last_batch_name = None
        needs_ref = bool(self._selection_without_own_reference())
        has_stage = has_stage and needs_ref     # particles already in a calibrated batch keep its X/Z
        n = self._flag_ids(ids, "keep", f"Accept {len(ids):,}", clear_selection=True)
        if n:
            ref_note = "" if needs_ref else " (they keep their batches' own Xbest/Zbest)"
            if self.save_ref_cb.isChecked() and has_stage and self._last_batch_name and not self._ref_set_for_cluster:
                ref_note = (" (no Xbest/Zbest saved: no focus peak found for this cluster and the line wasn't "
                            "moved — it borrows from the nearest calibrated batch)")
            if self.save_ref_cb.isChecked() and has_stage and self._last_batch_name and self._ref_set_for_cluster:
                self._store_editor_reference(self._last_batch_name)
                self._refresh_saved_sets()
                self.save_project_if_open()
                ref = self.saved_sets[self._last_batch_name]["best_focus"]
                ref_note = f" with Xbest={ref['x_best']:.5g}, Zbest={ref['z_best']:.5g}"
            self._ref_manual = False
            self._ref_set_for_cluster = False
            if self._last_batch_name in self.saved_sets:
                self.active_reference_set = self._last_batch_name   # keep showing what was just accepted
                self._shown_batches = []
                self._refresh_saved_sets()
                if getattr(self, "_bulk_depth", 0):
                    self._select_batch_after = self._last_batch_name
                else:
                    self._select_batch_row(self._last_batch_name)
            self._after_change()
            label = self._accept_label_name()
            self._toast(f"✓ Accepted {n:,} particles" + (f" as '{label}'" if label else "") + ref_note +
                        " — now green, saved, and skipped by new boxes. Ctrl+Z undoes; Unflag resets them.")

    def reject_selection(self):
        with self._bulk():
            return self._reject_selection_impl()

    def _reject_selection_impl(self):
        n = self._flag_ids(self.selected_ids, "reject", f"Reject {len(self.selected_ids):,}",
                           clear_selection=True, reason="graphical selection")
        if n:
            self._after_change()
            self._toast(f"✕ Rejected {n:,} particles — now red and saved. Ctrl+Z undoes.")

    def unflag_selection(self):
        with self._bulk():
            return self._unflag_selection_impl()

    def _unflag_selection_impl(self):
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
        self._last_batch_name = f"{prefix} batch {idx}"
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
            if self.selected_ids:
                return "selection"
            return "batch" if self._focus_group()[0] else "all"
        return scope

    def _rebuild_browse(self):
        records = self._records()
        scope = self._effective_browse_scope()
        if scope == "selection":
            filtered = [i for i in self._base_indices if records[i].particle_id in self.selected_ids]
        elif scope == "batch":
            group = self._focus_group()[0] or set()
            filtered = [i for i in self._base_indices if records[i].particle_id in group]
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
            + (f" · 🗂 {', '.join(self._groups_of.get(pid, ())[-1:])}" if self._groups_of.get(pid) else "")
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
        candidate = self._edge_trial_result(pid) if self.test_boundary_checkbox.isChecked() else None
        active_trial = self._edge_trials.get(self._edge_active_id, {})
        self.canvas.show_particle(image, boundary, f"ROI {record.roi_index} · {STATUS_NAME[record.status]}"
                                  + (" · selected" if in_sel else ""),
                                  self.boundary_checkbox.isChecked(), colour,
                                  comparison_boundary=(candidate.get("boundary") if candidate and
                                                       active_trial.get("kind") != "stored" else None),
                                  stored_colour=(active_trial.get("colour", "#ffd000") if
                                                 active_trial.get("kind") == "stored" else "#ffd000"),
                                  trial_colour=active_trial.get("colour", "#00d4ed"),
                                  comparison_label=("Integration aperture (not edge)" if
                                                    active_trial.get("kind") == "shadow" else "Tested edge"))
        self.particle_badge.setText(STATUS_NAME[record.status].upper())
        self.particle_badge.setStyleSheet(
            f"background:{STATUS_COLOUR[record.status]}; color:white; font-size:17px; font-weight:800;"
            "border-radius:6px; padding:8px;")
        self.particle_membership.setText(
            (f"<b style='color:{SELECTED_COLOUR}'>In the selection</b>" if in_sel else "Not selected")
            + f"<br>🗂 {self._batch_text(record)}")
        self.particle_membership.setTextFormat(Qt.RichText)
        self.toggle_selection_button.setText("− Remove from selection  (Del)" if in_sel
                                             else "+ Add to selection  (A)")
        self.metric_labels["time"].setText(record.time_iso or "—")
        self.metric_labels["length"].setText(self._format(record.length_um, 2))
        self.metric_labels["width"].setText(self._format(record.width_um, 2))
        self.metric_labels["focus"].setText(f"{self._format(record.focus, 3)}  (÷ illum.: "
                                            f"{self._format(self._normalised_focus(record), 3)})")
        self.metric_labels["illum"].setText(self._format(record.local_illum, 1))
        self.metric_labels["roundness"].setText(self._format(record.roundness, 3))
        self.metric_labels["xy"].setText(f"{self._format(record.image_x_px, 1)}, {self._format(record.image_y_px, 1)}")
        reference, source, kind = self._reference_for(record)
        side = ""
        if reference is not None and math.isfinite(record.stage_x) and math.isfinite(record.stage_y):
            d = signed_plane_distance(record.stage_x, record.stage_y, reference, self.plane_normal.value())
            how = {"editor": "editor (unsaved preview)", "own": f"own — {source}",
                   "nearest": f"nearest calibrated batch on camera — {source}",
                   "step": f"earlier in time — {source}"}[kind]
            side = (f"\nd = {d:+.4g} ({'+' if d > 0 else '−' if d < 0 else 'on the'} side{'' if d else ' plane'})"
                    f"\nXbest/Zbest {reference['x_best']:g}/{reference['z_best']:g}: {how}")
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
        image = _to_gray8(np.asarray(arr).T)        # camera orientation (X across), as the Camera X-Y plot
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

    def _decorated_icon(self, image, record, stored_boundary=None, tested_boundary=None,
                        original_shape=None):
        pid = record.particle_id
        selected = pid in self.selected_ids
        size = THUMB + 12
        pm = QPixmap(size, size)
        pm.fill(QColor("#ffffff"))
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        if image is not None:
            scaled = QPixmap.fromImage(image).scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            left, top = (size - scaled.width()) // 2, (size - scaled.height()) // 2
            painter.drawPixmap(left, top, scaled)
            if original_shape is not None and min(original_shape) > 0:
                for boundary, colour in ((stored_boundary, "#ffd000"),
                                         (tested_boundary, self._edge_trials.get(self._edge_active_id, {}).get("colour", "#00d8ee"))):
                    if boundary is None:
                        continue
                    b = np.asarray(boundary, dtype=float)
                    if b.ndim != 2 or b.shape[1] != 2 or len(b) < 2:
                        continue
                    # MAT outline points are (row, col); thumbnail was transposed.
                    points = [QPointF(left + float(x) / original_shape[0] * scaled.width(),
                                      top + float(y) / original_shape[1] * scaled.height())
                              for x, y in b if math.isfinite(x) and math.isfinite(y)]
                    if len(points) >= 2:
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor("#000000"), 3.5))
                        painter.drawPolyline(QPolygonF(points))
                        painter.setPen(QPen(QColor(colour), 2))
                        painter.drawPolyline(QPolygonF(points))
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
        mode = self.gallery_stat.currentData()
        result = self._edge_trial_result(record.particle_id)
        if mode == "none":
            second = ""
        elif mode == "length":
            second = f"L {self._format(record.length_um, 1)} µm"
        elif mode == "distance":
            second = f"d {self._format(self._gallery_distance(record), 3)} mm"
        elif mode == "position":
            second = f"X {self._format(record.stage_x, 2)} / Z {self._format(record.stage_y, 2)}"
        elif mode == "shadow":
            second = (f"Dshadow {self._format(record.shadow_deq_um, 1)} / "
                      f"Feret {self._format(record.length_um, 1)} µm")
        elif mode == "sizes":
            second = (f"Deq {self._format(record.deq_um, 1)}→"
                      f"{self._format(result['deq'], 1) if result else '—'} µm")
        elif mode == "all":
            second = (f"L {self._format(record.length_um, 0)} · "
                      f"d {self._format(self._gallery_distance(record), 2)} mm")
        else:
            second = f"F {self._format(record.focus, 1)} · {self._format(record.length_um, 0)} µm"
        return f"{first}\n{second}" if second else first

    def _gallery_distance(self, record):
        if not (math.isfinite(record.stage_x) and math.isfinite(record.stage_y)):
            return float("nan")
        reference = self._draft_reference_for(record)
        if reference is None:
            return float("nan")
        return signed_plane_distance(record.stage_x, record.stage_y, reference, self.plane_normal.value())

    def _gallery_overlay_changed(self, *_):
        self._gallery_dirty = True
        if self._current_tab() == "gallery":
            self.update_gallery()

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
        show_stored = self.gallery_saved_edge.isChecked()
        show_test = self.gallery_test_edge.isChecked()
        # Read MAT files in source order to avoid reloading the same multi-MB
        # file dozens of times on a mixed-batch gallery page.
        overlay_cache = {}
        if show_stored:
            for record in sorted(page_records, key=lambda r: r.source_file):
                try:
                    orig, outline = self.store.particle(record)
                    overlay_cache[record.particle_id] = (outline, orig.shape)
                except Exception:
                    pass
        for record in page_records:
            pid = record.particle_id
            stored_outline, original_shape = overlay_cache.get(pid, (None, None))
            test_outline = None
            if show_test:
                result = self._edge_trial_result(pid)
                if result:
                    test_outline = result.get("boundary")
                    if original_shape is None:
                        original_shape = result.get("image_shape")
            overlay_cache[pid] = (stored_outline, test_outline, original_shape)
        self._gallery_overlay_cache = overlay_cache
        for record in page_records:
            stored_outline, test_outline, original_shape = overlay_cache.get(record.particle_id,
                                                                            (None, None, None))
            item = QListWidgetItem(self._decorated_icon(self._thumbs.get(record.particle_id), record,
                                                        stored_outline, test_outline, original_shape),
                                   self._gallery_text(record))
            item.setData(Qt.UserRole, record.particle_id)
            item.setForeground(QColor(STATUS_COLOUR[record.status] if record.status != "undecided" else "#33475b"))
            tested = self._edge_trial_result(record.particle_id)
            item.setToolTip(f"{Path(record.source_file).name} · ROI {record.roi_index}\n"
                            f"Focus {self._format(record.focus, 2)} · Length {self._format(record.length_um, 1)} µm"
                            f"\nSigned object-plane distance: {self._format(self._gallery_distance(record), 4)} mm"
                            f"\nStage X {self._format(record.stage_x, 3)} / Z {self._format(record.stage_y, 3)}"
                            + (f"\nTrial #{self._edge_active_id}: Deq {tested['deq']:.2f} µm "
                               + ("(signed shadow integral)" if tested.get("overlay_kind") == "integration circle"
                                  else f"F {tested['focus']:.2f}") if tested else "")
                            + f"\nCamera ({self._format(record.image_x_px, 0)}, "
                              f"{self._format(record.image_y_px, 0)})"
                            + f"\n{self._batch_text(record, html=False)}")
            self.gallery.addItem(item)
            if record.particle_id in picked:
                item.setSelected(True)
        self.gallery.blockSignals(False)
        pages = max(1, (len(self.filtered) + GALLERY_PAGE - 1) // GALLERY_PAGE)
        scope = self._effective_browse_scope()
        label = {"selection": "selected", "all": "", "undecided": "unreviewed", "keep": "accepted",
                 "reject": "rejected", "batch": f"'{self.active_reference_set}'"}[scope]
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
                boundaries = getattr(self, "_gallery_overlay_cache", {}).get(record.particle_id,
                                                                               (None, None, None))
                item.setIcon(self._decorated_icon(self._thumbs.get(record.particle_id), record,
                                                   *boundaries))

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

    def _refresh_label_widgets(self, update_batches: bool = True):
        if not hasattr(self, "label_list"):
            return
        if update_batches and hasattr(self, "batch_list") and self.catalog:
            self._refresh_saved_sets()        # batch lines show their labels
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
        mode = "automatic" if self._library_mode == "auto" else "selected"
        self.label_path.setText(f"{self.labels.path}  ({mode}; microscopy stored beside labels.json)")
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

    def _label_context_menu(self, position):
        item = self.label_list.itemAt(position)
        if item is None:
            return
        self.label_list.setCurrentItem(item)
        menu = QMenu(self)
        microscope = menu.addAction("🔬 Optical / SEM reference images…")
        editing = menu.addAction("Edit label / dimensions…")
        chosen = menu.exec(self.label_list.mapToGlobal(position))
        if chosen == microscope:
            self.open_label_microscopy()
        elif chosen == editing:
            self.edit_label()

    def open_microscopy_panel(self):
        try:
            from .microscopy_gui import MicroscopyPanelDialog
        except ImportError:
            from microscopy_gui import MicroscopyPanelDialog
        dialog = MicroscopyPanelDialog(self.labels, self)
        dialog.exec()

    def open_label_microscopy(self):
        """Open the selected label's independent optical / SEM image annotations."""
        name = self._selected_label_name()
        if not name:
            self.statusBar().showMessage("Choose a label in Labels first", 4000)
            return
        label = self.labels.get(name)
        if label is None:
            QMessageBox.information(self, "Microscopy reference images",
                                    "Save this label in the shared label library first.")
            return
        try:
            from .microscopy_gui import MicroscopyDialog
        except ImportError:
            from microscopy_gui import MicroscopyDialog
        try:
            dialog = MicroscopyDialog(self.labels, label, self)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Microscopy images", str(exc))
            return
        dialog.exec()

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
        for window in CalibrationWindow._windows:
            if window is not self and window.labels is self.labels and window.catalog is not None:
                apply_label_dimensions(window.catalog.records, window.labels)
                window._refresh_label_widgets()

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

    def _switch_reference_library(self, destination, *, prompt_migration=True):
        """Use a shared library, copying legacy labels/images only with consent."""
        target = Path(destination).expanduser().resolve()
        if target == self.labels.path.expanduser().resolve():
            return True
        if not target.exists() and prompt_migration and self.labels.labels:
            question = QMessageBox(self)
            question.setWindowTitle("Shared CPI reference library")
            question.setText(f"Reference library not found at:\n{target}\n\n"
                             "Copy your existing labels and microscope/SEM images here?")
            question.setInformativeText("Copy leaves the old files untouched. "
                                        "Choose No to start an empty shared library.")
            question.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            question.setDefaultButton(QMessageBox.Yes)
            choice = question.exec()
            if choice == QMessageBox.Cancel:
                return False
            if choice == QMessageBox.Yes:
                try:
                    copy_reference_library(self.labels.path, target)
                    # In-memory labels might not have been written yet.
                    if not target.exists():
                        from copy import deepcopy
                        library = LabelLibrary(target)
                        library.labels = deepcopy(self.labels.labels)
                        library.save()
                except Exception as exc:
                    QMessageBox.warning(self, "Reference library", f"Could not copy library:\n{exc}")
                    return False
        try:
            library = LabelLibrary.load(target)
        except Exception as exc:
            QMessageBox.warning(self, "Reference library", f"Could not open {target}:\n{exc}")
            return False
        self.labels = library
        self._library_settings.setValue("reference_library_path", str(target))
        # An environment override must remain authoritative for this launch.
        if not os.environ.get("CPI_CALIBRATION_LABELS"):
            os.environ.pop("CPI_CALIBRATION_LABELS", None)
        if self.catalog:
            apply_label_dimensions(self.catalog.records, self.labels)
        for window in CalibrationWindow._windows:
            if window is not self and window.labels.path.expanduser().resolve() == target:
                window.labels = self.labels
        if not target.exists():
            try:
                self.labels.save()  # Explicitly established empty shared library.
            except OSError as exc:
                QMessageBox.warning(self, "Reference library", f"Could not create {target}:\n{exc}")
        self._refresh_label_widgets()
        if not self._restoring:
            self._after_change()
        self._toast(f"Using shared reference library: {target.parent}")
        return True

    def _maybe_use_shared_reference_library(self, source):
        if self._library_mode != "auto" or os.environ.get("CPI_CALIBRATION_LABELS"):
            return
        target = cpi_library_path(source)
        if target is not None:
            self._switch_reference_library(target)

    def use_automatic_label_library(self):
        self._library_mode = "auto"
        self._library_settings.setValue("reference_library_mode", "auto")
        hint = (self.project_file or (self.store.root if self.store else None) or
                (self.sources[0] if self.sources else None))
        if hint and cpi_library_path(hint):
            self._maybe_use_shared_reference_library(hint)
        else:
            QMessageBox.information(self, "Shared reference library",
                "Open a project or MAT directory anywhere under CPI_cals first. "
                "The workbench will then use CPI_cals/ReferenceLibrary automatically.")
        self._refresh_label_widgets()

    def choose_label_library(self):
        folder = QFileDialog.getExistingDirectory(self, "Select reference library folder",
                                                  str(self.labels.path.parent))
        if not folder:
            return
        if self._switch_reference_library(Path(folder) / "labels.json"):
            self._library_mode = "manual"
            self._library_settings.setValue("reference_library_mode", "manual")
            self._refresh_label_widgets()

    # ------------------------------------------------------- calibration sets --

    @contextlib.contextmanager
    def _bulk(self):
        """Inside one user action, rebuild the batch list / indexes once at the end, not per step."""
        self._bulk_depth = getattr(self, "_bulk_depth", 0) + 1
        try:
            yield
        finally:
            self._bulk_depth -= 1
            if self._bulk_depth == 0 and getattr(self, "_sets_dirty", False):
                self._sets_dirty = False
                self._refresh_sets_display()
                name = getattr(self, "_select_batch_after", None)
                self._select_batch_after = None
                if name:
                    self._select_batch_row(name)

    def _refresh_saved_sets(self):
        self._reference_index = build_reference_index(self.saved_sets, self.active_reference_set)
        self._active_members = set(self.saved_sets.get(self.active_reference_set, {}).get("particle_ids", ()))
        self._draft_reference_cache = None
        own = {}
        for name, entry in sorted(self.saved_sets.items()):
            if entry.get("best_focus"):
                for pid in entry.get("particle_ids", ()):
                    own.setdefault(pid, name)
        active = self.saved_sets.get(self.active_reference_set, {})
        if active.get("best_focus"):
            for pid in active.get("particle_ids", ()):
                own[pid] = self.active_reference_set
        self._own_ref_names = own
        groups_of = {}
        for name, entry in self.saved_sets.items():
            for pid in entry.get("particle_ids", ()):
                groups_of.setdefault(pid, []).append(name)
        self._groups_of = groups_of
        self._steps_cache = None
        self._groups_cache = None
        if getattr(self, "_bulk_depth", 0):
            self._sets_dirty = True          # display part once, at the end of the action
            return
        self._refresh_sets_display()

    def _refresh_sets_display(self):
        """Timeline, reference hint and the Flag history list (the slow, visible part)."""
        self._invalidate_reference_steps()
        self._update_reference_hint()
        # Flag history (review batches), newest first.
        if hasattr(self, "batch_list"):
            selected_batch = self._selected_batch_name()
            # Keep a multi-selection only while several batches are deliberately shown.
            really = set(self._selected_batch_names())
            chosen = set(self._shown_batches) if len(self._shown_batches) > 1 else (
                {selected_batch} & really)
            self.batch_list.blockSignals(True)
            self.batch_list.clear()
            show_rejects = getattr(self, "show_reject_batches_action", None) is not None and \
                self.show_reject_batches_action.isChecked()
            batches = [(n, e) for n, e in self.saved_sets.items()   # review batches + sets from older projects
                       if show_rejects or e.get("description") != "Review batch: reject"]
            batches.sort(key=lambda item: item[1].get("created_utc", ""), reverse=True)
            for name, spec in batches:
                description = spec.get("description", "")
                is_batch = description.startswith("Review batch: ")
                keep = description.endswith("keep") or not is_batch
                try:
                    stamp = datetime.fromisoformat(spec.get("created_utc", "")).astimezone().strftime("%d %b %H:%M")
                except ValueError:
                    stamp = ""
                ref = spec.get("best_focus")
                symbol = ("✓" if keep else "✕") if is_batch else "◆"
                label_text, label_tip = self._group_label_summary(spec)
                item = QListWidgetItem(f"{symbol} {name} · {len(spec['particle_ids']):,}"
                                       + (f" · {stamp}" if stamp else "")
                                       + (f"  🏷 {label_text}" if keep else "")
                                       + (f"  ⌖ X={ref['x_best']:g} Z={ref['z_best']:g}" if ref else ""))
                item.setToolTip(label_tip)
                item.setForeground(QColor(STATUS_COLOUR["keep" if keep else "reject"] if is_batch else COMMITTED_COLOUR))
                if not is_batch:
                    item.setToolTip("Group from an older project (calibration set). It works like a batch.")
                item.setData(Qt.UserRole, name)
                self.batch_list.addItem(item)
                if name == selected_batch:
                    self.batch_list.setCurrentItem(item, QItemSelectionModel.NoUpdate)
                if name in chosen:
                    item.setSelected(True)
            self.batch_list.blockSignals(False)

    def _batch_text(self, record, html=True):
        """Which batch(es) a particle is in, and the X/Z it uses (for hover / readouts)."""
        names = sorted(self._groups_of.get(record.particle_id, ()),
                       key=lambda n: self.saved_sets.get(n, {}).get("created_utc", ""))
        ref, source, kind = self._reference_for(record, use_editor=False)
        if ref is not None:
            how = {"own": "own", "nearest": f"from nearest: {source}", "step": f"from earlier: {source}"}.get(kind, "")
            xz = f"X/Z {ref['x_best']:.4g}/{ref['z_best']:.4g} ({how})"
        else:
            xz = "no X/Z yet"
        if not names:
            batch = "no batch"
        elif len(names) == 1:
            batch = names[0]
        else:
            batch = f"{names[-1]} (+{len(names) - 1} earlier: {', '.join(names[:-1])})"
        if html:
            return f"<b>{batch}</b> · {xz}"
        return f"{batch} · {xz}"

    def _detected_cpi(self):
        counts = {}
        for r in self._records():
            if r.detected_cpi:
                counts[r.detected_cpi] = counts.get(r.detected_cpi, 0) + 1
        return counts

    def instrument_info(self):
        """(type, id) of the calibrated CPI: the chosen type, else the one detected in the data."""
        chosen = self.cpi_type.currentData()
        if not chosen:
            counts = self._detected_cpi()
            chosen = max(counts, key=counts.get) if counts else ""
        return chosen, self.cpi_id.text().strip()

    def _instrument_changed(self):
        counts = self._detected_cpi()
        if counts:
            detected = ", ".join(f"{CPI_DISPLAY.get(k, k)} ({n:,} ROIs)" for k, n in sorted(counts.items()))
            text = f"Detected from the image-type codes: <b>{detected}</b>."
            chosen = self.cpi_type.currentData()
            if len(counts) > 1:
                text += " <span style='color:#c92a2a'>The loaded files come from different instruments.</span>"
            elif chosen in CPI_DISPLAY and chosen not in counts:
                text += (f" <span style='color:#c92a2a'>This differs from the chosen type "
                         f"({CPI_DISPLAY[chosen]}) — check.</span>")
        else:
            text = "Load data to detect the CPI type."
        self.cpi_detected.setText(text)
        self._update_window_title()

    def _update_window_title(self):
        cpi, ident = self.instrument_info()
        parts = [self.project_file.stem if self.project_file else "no project"]
        if getattr(self, "merged_from", None):
            parts[0] += " (merged viewing copy — label in the original projects)"
        parts += [v for v in (CPI_DISPLAY.get(cpi, cpi), ident) if v]
        self.setWindowTitle("CPI · Calibration Workbench v12 — " + " · ".join(parts))

    def open_sizing_dialog(self):
        if self.project_file is not None:
            self.save_project()                       # fit exactly what is saved, and record it
            return self.fit_from_projects([str(self.project_file)])
        rows = [r for r in self._particle_position_rows()
                if r.get("actual_length_um") not in ("", None) and r.get("d_signed_plane") not in ("", None)]
        if len(rows) < 20:
            QMessageBox.information(
                self, "Fit size & focus model",
                "The fit needs accepted particles that have a label with a true size and a best-focus X/Z "
                f"(so d is known). Found {len(rows)}.")
            return None
        dialog = SizingDialog(rows, self)
        dialog.show()
        self._sizing_dialog = dialog
        return dialog

    def fit_from_projects_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Calibration runs to fit together (one project per run)",
            str(self.project_file.parent) if self.project_file else "", "Project JSON (*.json)")
        if files:
            return self.fit_from_projects(files)
        return None

    def fit_from_projects(self, files, confirm=True):
        """Fit from the projects' saved particle tables, read live (no copies), recording which were used."""
        if self.project_file is not None and str(self.project_file.resolve()) in {str(Path(f).resolve()) for f in files}:
            self.save_project()
        rows, inputs, problems = load_projects_for_fit(files)
        if problems:
            QMessageBox.warning(self, "Fit from projects", "\n".join(problems))
            return None
        warnings = [f"{i['project']}: {w}" for i in inputs for w in i["warnings"]]
        if warnings and confirm and QMessageBox.question(
                self, "Fit from projects", "\n".join(warnings) + "\n\nFit anyway?") != QMessageBox.Yes:
            return None
        usable = [r for r in rows
                  if r.get("actual_length_um") not in ("", None) and r.get("d_signed_plane") not in ("", None)]
        if len(usable) < 20:
            QMessageBox.information(
                self, "Fit size & focus model",
                "The fit needs accepted particles that have a label with a true size and a best-focus X/Z "
                f"(so d is known). Found {len(usable)}.")
            return None
        for item in inputs:
            item["usable"] = sum(1 for r in usable if r["project"] == item["project"])
        dialog = SizingDialog(usable, self, inputs=inputs)
        names = ", ".join(i["project"] for i in inputs)
        dialog.setWindowTitle(f"Fit size & focus model — {names}")
        dialog.show()
        self._sizing_dialog = dialog
        return dialog

    def open_sizing_table_dialog(self, *_, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Particle table to fit",
                                                  str(self.project_file.parent) if self.project_file else "",
                                                  "CSV files (*.csv)")
        if not path:
            return None
        with open(path, newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle)
                    if r.get("actual_length_um") not in ("", None) and r.get("d_signed_plane") not in ("", None)]
        if len(rows) < 20:
            QMessageBox.information(self, "Fit size & focus model",
                                    f"{Path(path).name} has only {len(rows)} particles with a true size and d.")
            return None
        instruments = {(r.get("cpi_type") or "", r.get("cpi_id") or "") for r in rows}
        if len(instruments) > 1:
            names = ", ".join(" ".join(x for x in i if x) or "unnamed" for i in sorted(instruments))
            QMessageBox.warning(self, "Fit size & focus model",
                                f"This table mixes instruments ({names}). A calibration belongs to one CPI — "
                                "combine only that CPI's projects, or untick labels accordingly.")
        dialog = SizingDialog(rows, self)
        dialog.setWindowTitle(f"Fit size & focus model — {Path(path).name}")
        dialog.show()
        self._sizing_dialog = dialog
        return dialog

    def copy_project_to_reprocessed_dialog(self):
        if self.project_file is None:
            QMessageBox.information(self, "Copy project", "Open (and save) the project to copy first.")
            return None
        folder = QFileDialog.getExistingDirectory(
            self, "Folder with the reprocessed MAT files (same file names)", str(self.project_file.parent))
        if not folder:
            return None
        default = str(Path(folder) / (self.project_file.stem + "_" + Path(folder).name + ".json"))
        output, _ = QFileDialog.getSaveFileName(self, "New project for the reprocessed data", default,
                                                "Project JSON (*.json)")
        if not output:
            return None
        return self.copy_project_to_reprocessed(folder, output)

    def copy_project_to_reprocessed(self, folder, output, open_it=True):
        """Same flags, labels, batches and X/Z, pointing at reprocessed MAT files.

        Particles are re-found by ROI time + camera position, so a different
        edge method (or ROI numbering) does not lose any decision.
        """
        self.save_project()
        source = self.project_file
        config = json.loads(source.read_text(encoding="utf-8"))
        folder, output = Path(folder).resolve(), Path(output).resolve()
        missing, sources, pairs = [], [], {}
        for location in config.get("sources", []):
            old = Path(location)
            if not old.is_absolute():
                old = (source.parent / old).resolve()
            new = folder / old.name
            if not new.exists():
                missing.append(old.name)
                continue
            sources.append(os.path.relpath(str(new), str(output.parent)))
            pairs[old.name] = (old, new)
        if missing:
            QMessageBox.warning(self, "Copy project", f"{len(missing)} file(s) not found in {folder}:\n"
                                + "\n".join(missing[:10]))
            if not sources:
                return None
        config["sources"] = sources
        flags = source.with_name(config.get("flags_file", source.stem + "_flags.csv"))
        new_flags = output.with_name(output.stem + "_flags.csv")
        remap_report = None
        if flags.exists():
            # Re-find every flagged ROI in the new files by its pixels and corners, so the flags
            # survive changes to ROI times or numbering made by the reprocessing.
            from .remap import remap_flags_file
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                remap_report = remap_flags_file(
                    flags, new_flags, pairs,
                    progress=lambda n, total, name: (self.statusBar().showMessage(
                        f"Matching flagged ROIs by image: {name} ({n}/{total})"), QApplication.processEvents()))
            except Exception as exc:          # fall back to time + position matching
                shutil.copyfile(flags, new_flags)
                remap_report = {"error": str(exc)}
            finally:
                QApplication.restoreOverrideCursor()
        config["flags_file"] = new_flags.name
        maps = (remap_report or {}).get("maps") or {}
        if maps:
            # Batches and the selection refer to particles by ID and by (time, position) key; both are
            # stale after reprocessing.  Drop the stored keys of remapped files (they could point at a
            # different ROI now) and translate the IDs after opening, with the same image-based map.
            from .remap import split_id
            from .selection_state import expand_project_config
            expand_project_config(config)
            config["particle_keys"] = {pid: key for pid, key in config["particle_keys"].items()
                                       if split_id(pid)[0] not in maps}
        for key in ("position_file",):
            if config.get(key):
                pos = Path(config[key])
                pos = pos if pos.is_absolute() else (source.parent / pos).resolve()
                config[key] = os.path.relpath(str(pos), str(output.parent))
        extras = []
        for entry in config.get("extra_position_files", []) or []:
            pos = Path(entry["path"])
            pos = pos if pos.is_absolute() else (source.parent / pos).resolve()
            extras.append(dict(entry, path=os.path.relpath(str(pos), str(output.parent))))
        config["extra_position_files"] = extras
        config["copied_from"] = {"project": str(source), "data_folder": str(folder),
                                 "copied_utc": datetime.now().astimezone().isoformat(timespec="seconds")}
        output.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._last_remap_report = remap_report
        if open_it:
            self.open_project(str(output))
            if maps:
                self._apply_roi_maps(maps)
            self.save_project()      # writes the particle table with the reprocessed measurements
            self._update_window_title()
            if remap_report and "error" not in remap_report:
                kept = sum(1 for r in self.catalog.records if r.status == "keep") if self.catalog else 0
                QMessageBox.information(
                    self, "Copy project",
                    f"Matched {remap_report['remapped']:,} of {remap_report['rows']:,} flag rows to the reprocessed "
                    f"ROIs by their images ({remap_report['missing']:,} not found"
                    + (f", {remap_report['unchanged']:,} in files that could not be compared" if remap_report['unchanged'] else "")
                    + f").\n\nThe new project has {kept:,} accepted particles.")
            elif remap_report and "error" in remap_report:
                QMessageBox.warning(self, "Copy project", "Could not compare ROI images (" + remap_report["error"]
                                    + "); flags were matched by time and position only.")
        return output

    def _apply_roi_maps(self, maps):
        """Translate batch/selection IDs of the source project with per-file {old ROI index: new}."""
        from .remap import split_id
        if not self.catalog:
            return
        lookup = {(Path(r.source_file).name, r.roi_index): r.particle_id for r in self.catalog.records}

        def translate(ids):
            out = []
            for pid in ids:
                if pid in self.catalog._by_id:
                    out.append(pid)
                    continue
                name, index = split_id(pid)
                if name in maps:
                    new = maps[name].get(index)
                    if new is not None and (name, new) in lookup:
                        out.append(lookup[(name, new)])
                    continue                      # ROI not found in the new file: drop it
                out.append(pid)
            return out

        for entry in self.saved_sets.values():
            entry["particle_ids"] = sorted(set(translate(entry.get("particle_ids", ()))))
        self.selected_ids = set(translate(self.selected_ids))
        self._steps_cache = None
        self._draft_reference_cache = None
        self._refresh_saved_sets()
        self._after_change()

    def compare_processing_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Projects to compare (the same particles processed different ways)",
            str(self.project_file.parent) if self.project_file else "", "Project JSON (*.json)")
        if len(files) < 2:
            if files:
                QMessageBox.information(self, "Compare processing runs", "Pick two or more projects.")
            return None
        if self.project_file is not None and str(self.project_file.resolve()) in {str(Path(f).resolve()) for f in files}:
            self.save_project()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            text = sizing.compare_processing(files)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Compare processing runs", str(exc))
            return None
        QApplication.restoreOverrideCursor()
        dialog = QDialog(self)
        dialog.setWindowTitle("Compare processing runs")
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.NoWrap)
        view.setFont(QFont("Menlo, Consolas, DejaVu Sans Mono", 9))
        layout.addWidget(view)
        dialog.resize(1000, 560)
        dialog.show()
        self._compare_dialog = dialog
        return text

    def merge_projects_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Projects to merge (pick two or more)", str(self.project_file.parent) if self.project_file else "",
            "Project JSON (*.json)")
        if not files:
            return
        if len(files) < 2:
            QMessageBox.information(self, "Merge projects", "Pick at least two projects (Ctrl/⌘-click).")
            return
        start = str(Path(files[0]).with_name("merged_calibration_project.json"))
        output, _ = QFileDialog.getSaveFileName(self, "Merged project (a new file; the originals are not changed)",
                                                start, "Project JSON (*.json)")
        if output:
            self.merge_projects(files, output)

    def merge_projects(self, project_files, output, ask=True):
        """New project containing every listed project's data, flags, labels, batches and stage CSVs."""
        output = Path(output).resolve()
        if output in {Path(f).resolve() for f in project_files}:
            QMessageBox.warning(self, "Merge projects", "Save the merged project under a new name — the originals "
                                "are left unchanged.")
            return False
        instruments = {}
        for f in project_files:
            cpi_type, cpi_id = project_instrument(f)
            instruments[Path(f).stem] = (cpi_type, cpi_id)
        types = {t for t, _ in instruments.values() if t}
        ids = {i for _, i in instruments.values() if i}
        if len(types) > 1 or len(ids) > 1:
            listing = "\n".join(f"  {n}: {' '.join(v for v in pair if v) or 'not set'}"
                                 for n, pair in instruments.items())
            QMessageBox.warning(self, "Merge projects", "These projects are from different CPIs, and a calibration "
                                "belongs to one instrument:\n" + listing)
            return False
        if self.project_file is not None and self.project_file.resolve() in {Path(f).resolve() for f in project_files}:
            self.save_project()       # merge what is on screen, not an older save
        if not self._finish_current_project():
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._restoring = True
            self._reset_data(output.parent)
            self.project_file = output
            self.selection_file = output.with_name(output.stem + "_flags.csv")
            self.position_file, self.position_series, self.position_time_basis = None, None, None
            self.additional_roots = set()
            notes = []
            seen_sources = {}
            for f in project_files:
                notes += self._merge_one_project(Path(f).resolve(), seen_sources)
            self.merged_from = [Path(f).stem for f in project_files]
            if self.extra_position_files:
                first = self.extra_position_files.pop(0)
                self.position_file, self.position_series, self.position_time_basis = (
                    first["path"], first.get("series"), first.get("time_basis"))
            matched = self._apply_stage_positions()
            apply_label_dimensions(self.catalog.records, self.labels)
            self._save_label_library()
            self._restoring = False
            self._invalidate_reference_steps()
            self._refresh_saved_sets()
            self._view_locked = {"camera": False, "metric": False}
            self._use_stage_axis_after_import(matched)
            self.save_project()
            self.apply_filters()
            self._update_window_title()
        finally:
            self._restoring = False
            QApplication.restoreOverrideCursor()
        log = self._position_log()
        if log is not None and log.conflicting_times():
            notes.append(f"{log.conflicting_times()} stage-log timestamps clash between the projects' stage CSVs "
                         "(their times overlap) — check stage positions.")
        kept = sum(r.status == "keep" for r in self.catalog.records)
        summary = (f"Merged {len(project_files)} projects into {output.name}: {len(self.catalog.records):,} "
                   f"particles, {kept:,} accepted, {len(self.saved_sets)} batches/groups.")
        self._toast(summary, 15000)
        if ask and notes:
            QMessageBox.information(self, "Merge projects", summary + "\n\n" + "\n".join(notes))
        return True

    def _merge_one_project(self, path, seen_sources):
        """Add one project's data and decisions to the current (merged) project."""
        config = expand_project_config(json.loads(path.read_text(encoding="utf-8")))
        base, stem, notes = path.parent, path.stem, []
        for location in config.get("sources", []):
            source = Path(location)
            if not source.is_absolute():
                source = (base / source).resolve()
            if str(source) in seen_sources:
                notes.append(f"{source.name} is in both {seen_sources[str(source)]} and {stem}; where both flagged "
                             f"the same particle, {stem}'s decision is kept.")
                continue
            seen_sources[str(source)] = stem
            if source.is_file():
                self.import_file(source, refresh=False)
            else:
                notes.append(f"{stem}: data file not found: {source}")
        flags = base / config.get("flags_file", stem + "_flags.csv")
        report = self.catalog.apply_selection_file(flags)
        keys = config.get("particle_keys", {})
        for name, entry in normalise_sets(config.get("named_sets", {})).items():
            entry["particle_ids"] = sorted(set(self._remap_ids(entry["particle_ids"], report.id_map, keys)))
            new_name, k = f"{stem} · {name}", 2
            while new_name in self.saved_sets:
                new_name, k = f"{stem} · {name} ({k})", k + 1
            entry.setdefault("merged_from", stem)
            self.saved_sets[new_name] = entry
        self.labels.merge(config.get("labels", []))
        for w in config.get("time_windows", []) or []:
            if w.get("start") and w.get("end"):
                self.time_windows.append(dict(w, name=self._unique_window_name(f"{stem} · {w.get('name', 'Window')}")))
        specs = []
        if config.get("position_file"):
            specs.append({"path": config["position_file"],
                          "series": config.get("position_series") if "position_series" in config else None,
                          "time_basis": config.get("position_time_basis") if "position_time_basis" in config else None})
        specs += list(config.get("extra_position_files", []) or [])
        for spec in specs:
            pos = Path(spec["path"])
            if not pos.is_absolute():
                pos = (base / pos).resolve()
            if not pos.exists():
                notes.append(f"{stem}: stage CSV not found: {pos}")
                continue
            entry = {"path": str(pos), "series": spec.get("series"), "time_basis": spec.get("time_basis"),
                     "from": stem}
            if not any(e["path"] == entry["path"] and e.get("series") == entry["series"]
                       and e.get("time_basis") == entry["time_basis"] for e in self.extra_position_files):
                self.extra_position_files.append(entry)
        if not self.cpi_type.currentData() and (config.get("instrument") or {}).get("type"):
            inst = config["instrument"]
            self.cpi_type.setCurrentIndex(max(0, self.cpi_type.findData(inst.get("type", ""))))
            self.cpi_id.setText(str(inst.get("id", "")))
        if report.rows and report.unmatched:
            notes.append(f"{stem}: {report.summary()}")
        return notes

    def combine_projects_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Projects to combine", str(self.project_file.parent)
                                                if self.project_file else "", "Project JSON (*.json)")
        if not files:
            return
        output, _ = QFileDialog.getSaveFileName(self, "Combined particle table", "combined_particle_positions.csv",
                                                "CSV files (*.csv)")
        if not output:
            return
        if self.project_file is not None and str(self.project_file) in {str(Path(f).resolve()) for f in files}:
            self.save_project()   # make sure the open project's table is current
        try:
            summary = combine_projects(files, output)
        except OSError as exc:
            QMessageBox.critical(self, "Combine projects", str(exc))
            return
        lines = [f"{n:,} particles · {Path(f).name}" for f, n in summary["projects"]]
        if summary["missing"]:
            lines.append("\nNo particle table yet (open each and Save project first):")
            lines += [f"  {Path(f).name}" for f in summary["missing"]]
        QMessageBox.information(self, "Combine projects",
                                f"Wrote {summary['rows']:,} accepted particles to\n{output}\n\n" + "\n".join(lines))

    def _group_label_summary(self, spec):
        """Short text + tooltip describing the labels carried by a group's particles."""
        by_id = self.catalog._by_id if self.catalog else {}
        counts, total = {}, 0
        for pid in spec.get("particle_ids", ()):
            record = by_id.get(pid)
            if record is None:
                continue
            total += 1
            counts[record.label or ""] = counts.get(record.label or "", 0) + 1
        if not total:
            return "?", "Particles not loaded"
        named = {k: v for k, v in counts.items() if k}
        unlabelled = counts.get("", 0)
        if not named:
            text = "no label"
        elif len(named) == 1 and not unlabelled:
            text = next(iter(named))
        elif len(named) == 1:
            text = f"{next(iter(named))} ({named[next(iter(named))]:,} of {total:,})"
        else:
            text = f"mixed: {len(named)} labels"
        tip = "\n".join(f"{name or '(no label)'}: {n:,}" for name, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        return text, "Labels of this batch's particles:\n" + tip

    def _selected_batch_name(self):
        item = self.batch_list.currentItem() if hasattr(self, "batch_list") else None
        return item.data(Qt.UserRole) if item is not None else None

    def _select_batch_row(self, name):
        for row in range(self.batch_list.count()):
            if self.batch_list.item(row).data(Qt.UserRole) == name:
                self.batch_list.clearSelection()
                self.batch_list.setCurrentRow(row)
                return

    def _selected_batch_names(self):
        """All batches chosen in Flag history (Ctrl/Shift-click), in list order."""
        if not hasattr(self, "batch_list"):
            return []
        return [self.batch_list.item(i).data(Qt.UserRole) for i in range(self.batch_list.count())
                if self.batch_list.item(i).isSelected()]

    def _batch_export_positions(self, records, names, memberships):
        return batch_netcdf.batch_position_metadata(
            records, memberships, self.saved_sets, self.plane_normal.value(),
            lambda record: self._reference_for(record, use_editor=False))

    def export_batches_netcdf(self):
        """Export the Flag-history selection without changing project or MAT data."""
        if self._nc_export_worker is not None:
            QMessageBox.information(self, 'NetCDF export', 'Another batch export is already running.')
            return
        names, records = self._edge_batch_records()  # same multi-batch snapshot as Edge Lab
        if not names or not records or self.store is None:
            QMessageBox.information(self, 'NetCDF export',
                                    'Select one or more non-empty batches in Review → Flag history first.')
            return
        from dataclasses import replace
        import re
        snapshot = tuple(replace(r) for r in records)
        members_by_name = {n: set(self.saved_sets[n].get('particle_ids', ())) for n in names}
        memberships = {r.particle_id: [n for n in names if r.particle_id in members_by_name[n]]
                       for r in snapshot}
        positions = self._batch_export_positions(snapshot, names, memberships)
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', names[0]).strip('._')[:70] or 'batch'
        if len(names) > 1:
            stem += f'_and_{len(names) - 1}_more'
        default = (self.project_file.parent if self.project_file else self.store.root) / f'{stem}_rois.nc'
        filename, _ = QFileDialog.getSaveFileName(self, 'Export selected batch(es) to NetCDF',
                                                  str(default), 'NetCDF (*.nc)')
        if not filename:
            return
        # Freeze label dimensions so editing the library cannot change an active export.
        labels_snapshot = copy.deepcopy(self.labels)
        worker = BatchNetCDFWorker(filename, str(self.store.root), snapshot, tuple(names),
                                   memberships, labels_snapshot, positions, self.plane_normal.value(), self)
        dialog = QProgressDialog('Exporting original ROIs and their backgrounds…',
                                 'Cancel export', 0, len(snapshot), self)
        dialog.setWindowTitle('Batch NetCDF export')
        dialog.setWindowModality(Qt.NonModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        dialog.canceled.connect(worker.requestInterruption)
        worker.progress_changed.connect(dialog.setValue)
        worker.completed.connect(self._batch_netcdf_completed)
        worker.failed.connect(self._batch_netcdf_failed)
        worker.finished.connect(self._batch_netcdf_finished)
        self._nc_export_worker, self._nc_export_dialog = worker, dialog
        dialog.show()
        worker.start()

    def _batch_netcdf_completed(self, path, n, missing):
        if self._nc_export_dialog is not None:
            self._nc_export_dialog.close()
        message = f'Exported {n:,} distinct particles to:\n{path}'
        if missing:
            message += (f'\n\nWARNING: {missing:,} ROI(s) had no matching saved background. '
                        'The background_present flag is 0 and their background_pixels are -1.')
        QMessageBox.information(self, 'Batch NetCDF exported', message)
        self.statusBar().showMessage(f'Exported {n:,} particles to {Path(path).name}', 8000)

    def _batch_netcdf_failed(self, message):
        if self._nc_export_dialog is not None:
            self._nc_export_dialog.close()
        QMessageBox.warning(self, 'NetCDF export failed', message + '\n\nOriginal files were not changed.')

    def _batch_netcdf_finished(self):
        worker = self.sender()
        if worker is self._nc_export_worker:
            if worker.isInterruptionRequested():
                self.statusBar().showMessage('Batch NetCDF export cancelled; no output replaced.', 7000)
            self._nc_export_worker = None
            self._nc_export_dialog = None
        worker.deleteLater()

    def _choose_label(self, title, text, current=""):
        """Pick a library label (or none, or a new one). Returns name, "" for no label, or None if cancelled."""
        names = sorted(self.labels.names()) if hasattr(self.labels, "names") else []
        options = ["(no label)"] + names + ["New label…"]
        start = options.index(current) if current in options else 0
        choice, ok = QInputDialog.getItem(self, title, text, options, start, False)
        if not ok:
            return None
        if choice == "New label…":
            return self.new_label() or None
        return "" if choice == "(no label)" else choice

    def relabel_ids(self, ids, name, what="particles"):
        """Change the label (and true size) of the given particles; flags unchanged; undoable."""
        records = self.catalog._by_id if self.catalog else {}
        ids = [p for p in ids if p in records and records[p].status != "reject"]
        if not ids:
            return 0
        self._push_undo(f"Label {len(ids):,} {what} '{name or 'no label'}'", status_ids=ids)
        for pid in ids:
            self._set_label(records[pid], name)
        self.autosave()
        self.save_project_if_open()
        self._refresh_saved_sets()
        self._after_change()
        self._toast(f"🏷 {len(ids):,} {what} now labelled '{name or 'no label'}'"
                    + (f" ({self.labels.get(name).size_text()})" if name and self.labels.get(name) is not None
                       and hasattr(self.labels.get(name), "size_text") else "") + ". Ctrl+Z undoes.")
        return len(ids)

    def relabel_batch_dialog(self, *_, name=None, names=None):
        names = names or ([name] if name else None) or self._selected_batch_names() or \
            ([self._selected_batch_name()] if self._selected_batch_name() else [])
        names = [n for n in names if n in self.saved_sets]
        if not names:
            self.statusBar().showMessage("Choose a batch in Flag history first", 4000)
            return None
        members = list(dict.fromkeys(p for n in names for p in self.saved_sets[n].get("particle_ids", ())))
        current = {self.catalog._by_id[p].label for p in members if p in self.catalog._by_id} - {""}
        what = f"'{names[0]}'" if len(names) == 1 else f"the {len(names)} chosen batches"
        label = self._choose_label("Change label", f"New label for every particle in {what} "
                                   f"(now: {', '.join(sorted(current)) or 'no label'}):",
                                   next(iter(current)) if len(current) == 1 else "")
        if label is None:
            return None
        return self.relabel_ids(members, label, f"particles of {what}")

    def relabel_selection_dialog(self):
        if not self.selected_ids:
            self.statusBar().showMessage("Select particles first", 4000)
            return None
        label = self._choose_label("Label selection", f"Label for the {len(self.selected_ids):,} selected "
                                   "particles (their accept/reject flag is not changed):")
        if label is None:
            return None
        return self.relabel_ids(self.selected_ids, label, "selected particles")

    def select_batch(self):
        names = self._selected_batch_names() or ([self._selected_batch_name()] if self._selected_batch_name() else [])
        if not names or not self.catalog:
            self.statusBar().showMessage("Choose a batch in Flag history first", 4000)
            return
        ids = set()
        for name in names:
            ids |= set(self.saved_sets[name]["particle_ids"]) & set(self.catalog._by_id)
            status = self.saved_sets[name]["description"].partition(": ")[2]
            if status in self.show_status:
                self.show_status[status].setChecked(True)
        what = f"'{names[0]}'" if len(names) == 1 else f"{len(names)} batches"
        self._change_selection(ids, f"Selected the {len(ids):,} particles of {what}")

    def _members_start_time(self, name):
        by_id = self.catalog._by_id if self.catalog else {}
        times = [by_id[p].time_matlab for p in self.saved_sets[name].get("particle_ids", ())
                 if p in by_id and math.isfinite(by_id[p].time_matlab)]
        return min(times) if times else None

    def _store_editor_reference(self, name):
        """Save the editor's Xbest/Zbest as set/batch ``name``'s own reference."""
        entry = self.saved_sets[name]
        entry["best_focus"] = make_reference(self.best_x_set.value(), self.best_z_set.value())
        start = self._members_start_time(name)
        if start is not None:
            entry["ref_time_matlab"] = start  # where it starts applying on the timeline

    def batch_clicked(self):
        """Show a batch's X/Z — its own, or the one it inherits — in the editor.

        With several batches chosen, the plots show them together, each at its
        own saved X/Z (the editor is not previewed on any of them).
        """
        names = self._selected_batch_names()
        self._edge_batch_changed()
        if len(names) > 1:
            self._shown_batches = names
            self.active_reference_set = None
            self._refresh_saved_sets()
            self._refresh_views()
            self._plot_edge_preview()
            n = len(set().union(*(self.saved_sets[x].get("particle_ids", ()) for x in names)))
            self.batch_ref_label.setText(f"<b>{len(names)} batches</b> shown together ({n:,} particles), each at "
                                         "its own X/Z. <b>Select batch</b> and <b>Change label…</b> act on all "
                                         "of them; X/Z buttons act on one batch at a time.")
            return
        self._shown_batches = names
        name = names[0] if names else None
        if not name:
            return
        entry = self.saved_sets[name]
        ref = validate_reference(entry.get("best_focus"))
        if ref is not None:
            source = f"its own X/Z"
        else:
            ref, why = self._group_peak(name)
            if ref is not None:
                source = "no X/Z of its own yet — <b>proposed from its own focus peak</b>"
            else:
                start = self._members_start_time(name)
                inherited = None
                if start is not None:
                    from_steps = [s for s in self._reference_steps() if s[2] != name]
                    if from_steps:
                        k = max([i for i, s in enumerate(from_steps) if s[0] <= start] or [0])
                        _, ref, inherited = from_steps[k]
                source = (f"no focus peak ({why}); showing <b>{inherited}</b>'s X/Z" if ref is not None
                          else f"no X/Z yet and no focus peak ({why})")
        if ref is not None:
            for spin, value in ((self.best_x_set, ref["x_best"]), (self.best_z_set, ref["z_best"])):
                spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
        self.active_reference_set = name
        self._refresh_saved_sets()
        self._refresh_views()           # plots and the "Showing: …" line follow the clicked batch
        self._plot_edge_preview()
        self.batch_ref_label.setText(f"<b>{name}</b>: {source}. The editor now previews on this batch — adjust "
                                     "Xbest/Zbest (Best focus tab or drag the line), then <b>Save Xbest/Zbest to "
                                     "batch</b>.")

    def _group_peak(self, name):
        by_id = self.catalog._by_id if self.catalog else {}
        members = [by_id[p] for p in self.saved_sets[name].get("particle_ids", ())
                   if p in by_id and by_id[p].status != "reject"]
        return self._peak_reference(members)

    def fill_batch_references(self, *_, scope=None):
        """Give every accepted batch without X/Z its own, from its median focus peak."""
        accepted = [n for n, e in self.saved_sets.items() if e.get("description") == "Review batch: keep"]
        missing = [n for n in accepted if not self.saved_sets[n].get("best_focus")]
        if not accepted:
            QMessageBox.information(self, "Batch X/Z", "There are no accepted batches yet.")
            return
        if scope in ("all", "missing"):          # scripted use: no dialog
            return self._apply_peak_references(accepted if scope == "all" else missing)
        box = QMessageBox(self)
        box.setWindowTitle("Set X/Z from focus peaks")
        box.setIcon(QMessageBox.Question)
        box.setText("Put d = 0 at the peak of each batch's own median focus trace?")
        box.setInformativeText(
            f"{len(accepted)} accepted batches: {len(accepted) - len(missing)} already have X/Z, "
            f"{len(missing)} have none.\n\nThe peak is found along the object-plane normal "
            f"(θ = {self.plane_normal.value():g}°), so d = 0 lands on the focus maximum. "
            "Ctrl+Z undoes this.")
        all_button = box.addButton(f"All {len(accepted)} batches (replace existing)", QMessageBox.AcceptRole)
        missing_button = box.addButton(f"Only the {len(missing)} without X/Z", QMessageBox.AcceptRole)
        missing_button.setEnabled(bool(missing))
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(missing_button if missing else all_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is all_button:
            todo = accepted
        elif clicked is missing_button:
            todo = missing
        else:
            return
        self._apply_peak_references(todo)

    def _apply_peak_references(self, todo, report=True):
        """Set each named batch's X/Z from its own focus peak (undoable)."""
        self._push_undo("Set batch X/Z from focus peaks", sets=True)
        done, skipped = [], []
        for name in sorted(todo, key=lambda n: self._members_start_time(n) or 0):
            ref, why = self._group_peak(name)
            if ref is None:
                skipped.append(f"{name}: {why}")
                continue
            entry = self.saved_sets[name]
            entry["best_focus"] = ref
            start = self._members_start_time(name)
            if start is not None:
                entry["ref_time_matlab"] = start
            done.append(name)
        self._refresh_saved_sets()
        self.save_project_if_open()
        self._after_change()
        msg = f"Set Xbest/Zbest for {len(done)} batch(es) from their focus peaks."
        if skipped:
            msg += (f"\n\n{len(skipped)} skipped (unchanged — a batch with its own X/Z keeps it, one without "
                    "keeps borrowing from nearby batches):\n" + "\n".join(skipped[:10]))
        if report:
            QMessageBox.information(self, "Batch X/Z", msg)
        return done, skipped

    def save_batch_reference(self):
        name = self._selected_batch_name() or (self.active_reference_set
                                               if self.active_reference_set in self.saved_sets else None)
        if not name:
            self.statusBar().showMessage("Choose a batch in Flag history first", 4000)
            return
        self._push_undo(f"X/Z for '{name}'", sets=True)
        self._store_editor_reference(name)
        self.active_reference_set = name
        self._refresh_saved_sets()
        self.save_project_if_open()
        self._after_change()
        ref = self.saved_sets[name]["best_focus"]
        self.batch_ref_label.setText(f"<b>{name}</b> now has its own X/Z: Xbest={ref['x_best']:g}, "
                                     f"Zbest={ref['z_best']:g}. Later batches without their own inherit it.")
        self._toast(f"Saved Xbest={ref['x_best']:g}, Zbest={ref['z_best']:g} to '{name}'.")

    def clear_batch_reference(self):
        name = self._selected_batch_name()
        if not name or "best_focus" not in self.saved_sets[name]:
            return
        self._push_undo(f"Clear X/Z of '{name}'", sets=True)
        self.saved_sets[name].pop("best_focus", None)
        self.saved_sets[name].pop("ref_time_matlab", None)
        self._refresh_saved_sets()
        self.save_project_if_open()
        self._after_change()
        self.batch_ref_label.setText(f"<b>{name}</b> now inherits X/Z from the timeline.")

    def _refresh_timeline(self):
        steps = self._reference_steps()
        if not steps:
            self.timeline_label.setText(
                "<span style='color:#516579'>No batch has best focus yet. Accept a cluster (K), or use Review → "
                "Flag history → <i>Set X/Z of all batches from their focus peaks</i>.</span>")
            return
        lines = []
        for start, ref, name in steps:
            try:
                when = matlab_datenum_to_datetime(start).strftime("%d %b %H:%M:%S")
            except (ValueError, OverflowError):
                when = f"{start:.6f}"
            lines.append(f"<b>{name}</b> · {when} · X={ref['x_best']:g}, Z={ref['z_best']:g}")
        missing = sum(1 for e in self.saved_sets.values()
                      if e.get("description") == "Review batch: keep" and not e.get("best_focus"))
        tail = (f"<br><span style='color:#a61e4d'>{missing} accepted batch(es) have no X/Z of their own.</span>"
                if missing else "")
        self.timeline_label.setText("<br>".join(lines) + tail + "<br><span style='color:#516579'>Particles without "
                                    "their own X/Z use the nearest of these on the camera (same session), else the "
                                    "latest earlier one.</span>")

    def _reference_edited(self):
        if not self._auto_placing:
            self._ref_manual = True   # the user moved the line / typed: stop following the peak
            self._ref_set_for_cluster = True
        self._set_reference_changed()

    def _selection_peak(self):
        """(reference, reason) from the selection's own median focus traces."""
        if not self.catalog or not self.selected_ids:
            return None, "nothing selected"
        pids = self._selection_without_own_reference()
        if not pids:
            return None, "every selected particle already uses its batch's saved X/Z"
        return self._peak_reference([self.catalog._by_id[p] for p in pids])

    def _peak_reference(self, recs):
        """Xbest/Zbest that puts d = 0 at the median focus peak (along the plane normal)."""
        xs, zs = [r.stage_x for r in recs], [r.stage_y for r in recs]
        fs = [self._peak_focus(r) for r in recs]
        ref, why = focus_peak_on_plane(xs, zs, fs, self.plane_normal.value(), self.peak_combo.currentData())
        if ref is None:
            ref2, why2 = focus_peak_reference(xs, zs, fs, self.peak_combo.currentData())
            if ref2 is not None:
                return ref2, why2
        return ref, why

    def _peak_focus(self, record):
        """Focus value used to find best-focus peaks (raw, or illumination-normalised)."""
        if getattr(self, "norm_peak_cb", None) is not None and self.norm_peak_cb.isChecked():
            return self._normalised_focus(record)
        return record.focus

    def _auto_place_reference(self, force=False):
        """Move Xbest/Zbest to the selection's focus peak unless the user overrode it."""
        if not self.follow_peak_cb.isChecked() or (self._ref_manual and not force) or not self.selected_ids:
            return False
        ref, _ = self._selection_peak()
        if ref is None:
            return False
        self._auto_placing = True
        try:
            for spin, value in ((self.best_x_set, ref["x_best"]), (self.best_z_set, ref["z_best"])):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            self._ref_manual = False
            self._ref_set_for_cluster = True
            self._set_reference_changed(redraw=False)
        finally:
            self._auto_placing = False
        return True

    def _set_reference_changed(self, redraw=True):
        if hasattr(self, "best_x_set") and hasattr(self, "best_z_set"):
            radius = math.hypot(self.best_x_set.value(), self.best_z_set.value())
            self._draft_reference_cache = None
            rec = self.current_record()
            offset = (f" · current ROI: ΔX={rec.stage_x - self.best_x_set.value():.4g}, "
                      f"ΔZ={rec.stage_y - self.best_z_set.value():.4g}"
                      if rec is not None and math.isfinite(rec.stage_x) and math.isfinite(rec.stage_y) else "")
            self.set_radius_label.setText(("Current particle (☆)" + offset.replace(" · current ROI", "")) if offset else "")
            self._update_save_ref_text()
            if redraw and hasattr(self, "plot_canvas"):
                self.update_plots()

    def _update_reference_hint(self):
        if not hasattr(self, "active_set_label"):
            return
        name = self.active_reference_set
        if name and name in self.saved_sets:
            own = self.saved_sets[name].get("best_focus")
            self.active_set_label.setText(
                f"Editing batch <b>{name}</b> ({'has its own X/Z' if own else 'no X/Z of its own yet'}). "
                "Its particles preview these values.")
            self.save_to_batch_button.setEnabled(True)
            self.save_to_batch_button.setText(f"Save Xbest/Zbest to '{name}'")
        else:
            self.active_set_label.setText("No batch chosen: values preview on the selection and are saved with "
                                          "the next Accept.")
            self.save_to_batch_button.setEnabled(False)
            self.save_to_batch_button.setText("Save Xbest/Zbest to batch")

    def move_reference_line(self, value):
        axis = self.position_metric.currentData()
        if axis not in ("stage_x", "stage_z"):
            return
        (self.best_x_set if axis == "stage_x" else self.best_z_set).setValue(value)
        self._toast(f"Draft {axis[-1].upper()}best = {value:.5g}; Accept saves it with the batch", 7000)

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
        if self.selected_ids:
            ref, why = self._selection_peak()
            if ref is None:
                QMessageBox.information(self, "Selection's focus peak", f"No focus peak: {why}.")
                return
            self._auto_place_reference(force=True)
            self.update_plots()
            self._toast(f"Xbest/Zbest set to the selection's focus peak: {ref['x_best']:.5g} / {ref['z_best']:.5g}")
            return
        axis = self.position_metric.currentData()
        if axis not in ("stage_x", "stage_z") or axis not in self._median_peak:
            QMessageBox.information(self, "Median focus peak",
                                    "Plot Focus against Stage X (or Stage Z) with the median trace switched on, "
                                    "then try again.")
            return
        (self.best_x_set if axis == "stage_x" else self.best_z_set).setValue(self._median_peak[axis])
        self._toast(f"Draft {axis[-1].upper()}best set to the median focus peak "
                    f"{self._median_peak[axis]:.5g}. Save it with the set.")

    def _restore_reference_editor(self, draft=None):
        """Put Xbest/Zbest back in the editor when a project opens.

        Use the values that were in the editor when the project was saved;
        projects from before that was stored fall back to the active set's
        saved reference, then to any set with a reference.
        """
        ref = None
        if isinstance(draft, dict):
            try:
                ref = make_reference(draft["x_best"], draft["z_best"])
            except (KeyError, TypeError, ValueError):
                ref = None
        if ref is None and self.active_reference_set in self.saved_sets:
            ref = self.saved_sets[self.active_reference_set].get("best_focus")
        if ref is None:
            ref = next((s.get("best_focus") for n, s in sorted(self.saved_sets.items()) if s.get("best_focus")), None)
        if ref is None:
            return
        for spin, value in ((self.best_x_set, ref["x_best"]), (self.best_z_set, ref["z_best"])):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._draft_reference_cache = None

    def remove_classified_batch(self):
        """Undo one accepted/rejected batch without changing unrelated statuses."""
        name = self._selected_batch_name()
        if name is None or not self.catalog:
            self.statusBar().showMessage("Choose a batch in Flag history first", 4000)
            return
        if not self.saved_sets[name].get("description", "").startswith("Review batch: "):
            if QMessageBox.question(self, "Remove group?", f"Remove the group '{name}' (from an older project)? "
                                    "Flags and labels are not changed.") != QMessageBox.Yes:
                return
            self._push_undo(f"Remove group '{name}'", sets=True)
            del self.saved_sets[name]
            if name == self.active_reference_set:
                self.active_reference_set = None
            self._refresh_saved_sets()
            self.save_project_if_open()
            self._after_change()
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
        self.saved_sets.pop(name, None)   # may already be gone if it emptied
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

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Processed CPI calibration directory")
        if directory:
            self.load_directory(directory, position_file=self.position_file)

    def choose_add_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Add processed CPI dataset")
        if directory:
            self.add_directory(directory)

    def _reset_data(self, root):
        if hasattr(self, "edge_figure"):
            self._clear_edge_preview()
        if hasattr(self, "cpi_type"):
            self.cpi_type.setCurrentIndex(0)
            self.cpi_id.clear()
        self.catalog = ParticleCatalog(root, [])
        self.store = ParticleStore(root)
        self.sources = []
        self.unloaded_sources = []      # project files that were missing or unreadable: kept in the project
        self.saved_sets = {}
        self._shown_batches = []
        self.extra_position_files = []
        self.merged_from = []
        self.time_windows = []
        self.active_time_window = None
        if hasattr(self, "window_list"):
            self._refresh_time_windows()
        self.active_reference_set = None
        self.selected_ids.clear()
        self.regions.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current_id = None
        self._thumbs.clear()
        self._refresh_saved_sets()

    def new_window(self):
        """Another workbench window, for working on a second project side by side."""
        window = CalibrationWindow()
        window.resize(self.size())
        window.move(self.pos() + QPoint(40, 40))
        window.show()
        return window

    def _window_with_project(self, filename):
        target = Path(filename).resolve()
        for window in CalibrationWindow._windows:
            if window is not self and window.project_file is not None and window.project_file.resolve() == target:
                return window
        return None

    def _finish_current_project(self) -> bool:
        """Save the open project (or offer to) before it is closed or replaced. False = cancelled."""
        if self.project_file is None and self.saved_sets:
            choice = QMessageBox.question(
                self, "Save project?",
                "Review flags and labels are already saved in the flags CSV. Save a project too, so your "
                "batches, X/Z references and flag history are kept?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if choice == QMessageBox.Cancel:
                return False
            if choice == QMessageBox.Yes:
                self.save_project()
                if self.project_file is None:
                    return False
        if self.project_file is not None:
            self.save_project()
        elif self.catalog is not None and self.catalog.records:
            self._write_flags()
        return True

    def close_project(self) -> bool:
        """Save and close the current project, leaving an empty workbench."""
        if not self._finish_current_project():
            return False
        closed = self.project_file.stem if self.project_file else None
        self._reset_data(Path.cwd())
        self.project_file = None
        self.selection_file = None
        self.position_file = None
        self.position_series = self.position_time_basis = None
        self.additional_roots = set()
        self.apply_filters()
        self._update_window_title()
        self._toast(f"Saved and closed '{closed}'." if closed else "Closed. Start with File ▸ New project… "
                    "or Open project….")
        return True

    def create_project(self) -> None:
        start = str(self.project_file.parent / "calibration_project.json") if self.project_file else \
            "calibration_project.json"
        filename, _ = QFileDialog.getSaveFileName(self, "New calibration project", start, "Project JSON (*.json)")
        if filename and self._finish_current_project():
            self.position_file = None
            self.position_series = self.position_time_basis = None
            self.additional_roots = set()
            self._reset_data(Path(filename).parent)
            self.project_file = Path(filename).resolve()
            self.selection_file = self.project_file.with_name(self.project_file.stem + "_flags.csv")
            self.save_project()
            self.apply_filters()
            self._update_window_title()
            self._toast(f"New project '{self.project_file.stem}'. Add data with + Add folder / + Add MAT files, "
                        "then Stage CSV….")

    def open_project_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open calibration project", str(self.project_file.parent) if self.project_file else "",
            "Project JSON (*.json)")
        if not filename:
            return
        other = self._window_with_project(filename)
        if other is not None:
            QMessageBox.information(self, "Open project", f"'{Path(filename).stem}' is already open in another "
                                    "window — switching to it. (Two windows on one project would overwrite each "
                                    "other's saves.)")
            other.raise_()
            other.activateWindow()
            return
        if self._finish_current_project():
            self.open_project(filename)
            self._update_window_title()

    def open_project(self, filename: str) -> None:
        config = expand_project_config(json.loads(Path(filename).read_text(encoding="utf-8")))
        self._restoring = True
        self.project_file = Path(filename).resolve()
        library_hint = self.project_file
        if cpi_library_path(library_hint) is None:
            # Projects may live outside CPI_cals but reference MAT data inside it.
            for source_name in config.get("sources", []):
                source = Path(source_name)
                source = source if source.is_absolute() else self.project_file.parent / source
                if cpi_library_path(source) is not None:
                    library_hint = source
                    break
        self._maybe_use_shared_reference_library(library_hint)
        self._reset_data(self.project_file.parent)
        self.saved_sets = normalise_sets(config.get("named_sets", {}))
        requested = config.get("active_reference_set")
        self.active_reference_set = requested if requested in self.saved_sets else None
        self._refresh_saved_sets()
        self.selection_file = self.project_file.with_name(config.get("flags_file", self.project_file.stem + "_flags.csv"))
        missing, previous_errors = [], self._import_errors
        self._import_errors = []
        for location in config.get("sources", []):
            source = Path(location)
            if not source.is_absolute():
                source = (self.project_file.parent / source).resolve()
            if source.is_file():
                self.import_file(source, refresh=False)
            else:
                missing.append(str(source))
        problems, self._import_errors = self._import_errors, previous_errors
        loaded_now = set(self.sources)
        for location in config.get("sources", []):
            source = Path(location)
            if not source.is_absolute():
                source = (self.project_file.parent / source).resolve()
            if str(source.resolve() if source.exists() else source) not in loaded_now and \
                    source.name not in ("timeseries.mat", "full_backgrounds.mat"):
                self.unloaded_sources.append(str(source))
        if missing or problems:
            lines = []
            if missing:
                lines.append(f"{len(missing)} of the project's MAT files were not found, e.g.\n  " +
                             "\n  ".join(missing[:3]))
            if problems:
                lines.append(f"{len(problems)} MAT files could not be read, e.g.\n  " + "\n  ".join(problems[:3]))
            loaded = len(self.catalog.records) if self.catalog else 0
            lines.append(f"{loaded:,} particles loaded." + (" Your saved flags file is left untouched until "
                                                             "particles load." if not loaded else ""))
            QMessageBox.warning(self, "Project files", "\n\n".join(lines))
        report = self.catalog.apply_selection_file(self.selection_file)
        self._remap_saved_ids(report.id_map, config.get("particle_keys", {}))
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
        self.selection_only_cb.setChecked(bool(config.get("selection_only", True)))
        self.median_cb.setChecked(bool(config.get("show_median", True)))
        self.set_box_mode(config.get("box_mode", "new") if config.get("box_mode") in dict(
            (k, 1) for k, _, _ in BOX_MODES) else "new")
        self.plane_normal.setValue(float(config.get("plane_normal_deg", 45.0)))
        inst = config.get("instrument") or {}
        self.cpi_type.blockSignals(True)
        self.cpi_type.setCurrentIndex(max(0, self.cpi_type.findData(inst.get("type", ""))))
        self.cpi_type.blockSignals(False)
        self.cpi_id.setText(str(inst.get("id", "")))
        self.norm_peak_cb.blockSignals(True)
        self.norm_peak_cb.setChecked(bool(config.get("normalised_focus_peaks", False)))
        self.norm_peak_cb.blockSignals(False)
        self._restore_reference_editor(config.get("draft_reference"))
        self.stage_segments = config.get("stage_segments", [])
        self._refresh_reference_list()
        self.regions = [tuple(r) for r in config.get("regions", [])]
        self.selected_ids = set(self._remap_ids(config.get("selected_ids", []), report.id_map,
                                                config.get("particle_keys", {}))) & set(self.catalog._by_id)
        self.position_series = config.get("position_series", "beads") if "position_series" in config else None
        self.position_time_basis = config.get("position_time_basis", "roi") if "position_time_basis" in config \
            else None
        self.position_file = config.get("position_file")
        if self.position_file:
            pos = Path(self.position_file)
            if not pos.is_absolute():
                pos = (self.project_file.parent / pos).resolve()
            self.position_file = str(pos)
        self.merged_from = list(config.get("merged_from", []) or [])
        self.time_windows = [dict(w) for w in config.get("time_windows", []) or [] if w.get("start") and w.get("end")]
        self.active_time_window = config.get("active_time_window") if any(
            w["name"] == config.get("active_time_window") for w in self.time_windows) else None
        self.extra_position_files = []
        for entry in config.get("extra_position_files", []) or []:
            pos = Path(entry.get("path") or "")
            if not pos.is_absolute():
                pos = (self.project_file.parent / pos).resolve()
            self.extra_position_files.append(dict(entry, path=str(pos)))
        if any(Path(p).exists() for p, _, _ in self._position_specs()):
            self._apply_stage_positions()
        self._restoring = False
        self._view_locked = {"camera": False, "metric": False}
        self.apply_filters()
        self._refresh_time_windows()
        self._report_matching(report, f"Opened project {self.project_file.name}")
        if self.merged_from:
            self._toast(f"This is a merged viewing copy of {', '.join(self.merged_from)}. Changes here don't reach "
                        "those projects — label there, and fit with Calibration ▸ from several projects.", 15000)

    def save_project(self, exports: bool = True) -> None:
        """Write the project and flags now.  ``exports`` also writes the particle/relative-position
        tables (needed by fits and Combine), which is skipped for the batched background saves."""
        if hasattr(self, "_save_timer"):
            self._save_timer.stop()
        self._pending_flags = self._pending_project = False
        if self.project_file is None:
            filename, _ = QFileDialog.getSaveFileName(self, "Save calibration project", "calibration_project.json",
                                                      "Project JSON (*.json)")
            if not filename:
                return
            self.project_file = Path(filename).resolve()
            self.selection_file = self.project_file.with_name(self.project_file.stem + "_flags.csv")
        self._write_flags()
        config = self._project_config()
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        if exports and self.catalog is not None and not self.catalog.records:
            exports = False                      # keep the previous exports rather than writing empty tables
        if exports:
            self._export_committed_positions()
            self._export_particle_positions()
            self._exports_stale = False
        else:
            self._exports_stale = True
        temp = self.project_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, self.project_file)
        self.statusBar().showMessage("Project saved", 3000)
        if hasattr(self, "save_indicator"):
            self.save_indicator.setText(f"✓ Saved {datetime.now():%H:%M:%S} · {self.project_file.name}")
            self.save_indicator.setStyleSheet("color:#2b7a3d; padding: 0 8px;")

    def _project_config(self, raw: bool = False):
        """Everything the project file holds (compact form).

        ``raw``: leave the particle lists uncompacted (as copies) so the
        compaction can be done on the writer thread; see :func:`_compact_config`.
        """
        def relative(value):
            if not value:
                return None
            try:
                return os.path.relpath(str(value), str(self.project_file.parent))
            except ValueError:
                return str(value)
        unloaded = [u for u in getattr(self, "unloaded_sources", []) if u not in self.sources]
        config = {"format_version": 4, "sources": [relative(s) for s in self.sources + unloaded],
                  "flags_file": self.selection_file.name, "position_file": relative(self.position_file),
                  "filters": {key: [pair[0].text(), pair[1].text()] for key, pair in self.filter_fields.items()},
                  "time_low": self._date_text(self.time_low), "time_high": self._date_text(self.time_high),
                  "use_time_filter": self.time_filter_checkbox.isChecked(), "image_mode": self.image_mode.currentData(),
                  "selected_ids": sorted(self.selected_ids) if raw else compact_ids(self.selected_ids),
                  "named_sets": ({n: dict(e, particle_ids=list(e.get("particle_ids", ())))
                                  for n, e in self.saved_sets.items()} if raw
                                 else compact_project_sets(self.saved_sets)),
                  "active_reference_set": self.active_reference_set,
                  # What is in the Xbest/Zbest editor, saved or not, so reopening shows it again.
                  "draft_reference": {"x_best": self.best_x_set.value(), "z_best": self.best_z_set.value()},
                  # time-ms, StartX+EndX, StartY+EndY for every ID referenced by sets or the
                  # selection, so they can be re-found after reprocessing renumbers ROIs.
                  "particle_keys": (self._particle_keys_for_project() if raw
                                    else compact_keys(self._particle_keys_for_project())),
                  "view_scope": self.view_selector.currentData(),
                  "plot_x": self.position_metric.currentData(), "plot_y": self.plot_metric.currentData(),
                  "plot_y2": self.plot_metric2.currentData(), "colour_by": self.colour_by.currentData(),
                  "pick_scope": self.pick_scope.currentData(), "box_mode": self.box_mode,
                  "show_status": {s: b.isChecked() for s, b in self.show_status.items()},
                  "selection_only": self.selection_only_cb.isChecked(), "show_median": self.median_cb.isChecked(),
                  "plane_normal_deg": self.plane_normal.value(),
                  "normalised_focus_peaks": self.norm_peak_cb.isChecked(),
                  "instrument": {"type": self.cpi_type.currentData() or "", "id": self.cpi_id.text().strip(),
                                 "detected": self.instrument_info()[0]},
                  "regions": self.regions, "stage_segments": self.stage_segments,
                  # Snapshot of the labels used, so the project is self-describing on another computer.
                  "labels": [self.labels.get(n).as_json() for n in sorted(self._label_counts())
                             if self.labels.get(n) is not None],
                  "position_series": self.position_series, "position_time_basis": self.position_time_basis,
                  "extra_position_files": [dict(e, path=relative(e["path"])) for e in self.extra_position_files],
                  "merged_from": list(getattr(self, "merged_from", []) or []),
                  "time_windows": self.time_windows, "active_time_window": self.active_time_window}
        return config

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

    PARTICLE_POSITION_COLUMNS = (
        "cpi_type", "cpi_id", "particle_id", "source_file", "roi_index", "time_iso", "time_matlab", "batch", "label",
        "actual_length_um", "actual_width_um", "length_um", "width_um", "focus", "local_illum",
        "focus_normalised", "roundness",
        "image_x_px", "image_y_px", "x_stage", "z_stage", "reference_from", "reference_kind",
        "x_best", "z_best", "x_minus_xbest", "z_minus_zbest", "plane_normal_deg",
        "d_signed_plane", "d_in_plane", "processing", "focus_T", "edge_touch", "deq_um", "shadow_deq_um", "area_ratio_circle", "shadow_radius_px", "pixel_um",
        "cpv_focus", "cpv_focus_sd", "cpv_fwhm_px", "cpv_distinct", "cpv_cutoff_pct", "cutoff_pct")

    def _export_particle_positions(self):
        """One row per ACCEPTED particle with the saved reference it actually uses.

        Uses saved X/Z only (own set/batch, else the time step), never the
        unsaved editor preview.  This is the table to fit calibrations from.
        """
        if self.project_file is None or self.catalog is None:
            return
        path = self.project_file.with_name(self.project_file.stem + "_particle_positions.csv")
        rows = self._particle_position_rows()
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.PARTICLE_POSITION_COLUMNS, restval="")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)

    def _particle_position_rows(self):
        if self.catalog is None:
            return []
        theta = self.plane_normal.value()
        cpi_type, cpi_id = self.instrument_info()
        rows = []
        for r in self.catalog.records:
            if r.status != "keep":
                continue
            ref, source, kind = self._reference_for(r, use_editor=False)
            row = dict(cpi_type=CPI_DISPLAY.get(cpi_type, cpi_type), cpi_id=cpi_id,
                       particle_id=r.particle_id, source_file=r.source_file, roi_index=r.roi_index,
                       time_iso=r.time_iso, time_matlab=r.time_matlab, label=r.label,
                       batch=(self._groups_of.get(r.particle_id) or [""])[-1],
                       actual_length_um=r.actual_length_um, actual_width_um=r.actual_width_um,
                       length_um=r.length_um, width_um=r.width_um, focus=r.focus, local_illum=r.local_illum,
                       focus_normalised=self._normalised_focus(r), roundness=r.roundness,
                       image_x_px=r.image_x_px, image_y_px=r.image_y_px, x_stage=r.stage_x, z_stage=r.stage_y,
                       reference_from=source or "", reference_kind=kind or "", plane_normal_deg=theta,
                       processing=r.processing, focus_T=r.focus_T, edge_touch=r.edge_touch, deq_um=r.deq_um,
                       shadow_deq_um=r.shadow_deq_um, area_ratio_circle=r.area_ratio_circle, shadow_radius_px=r.shadow_radius_px,
                       pixel_um=r.pixel_um, cpv_focus=r.cpv_focus, cpv_focus_sd=r.cpv_focus_sd,
                       cpv_fwhm_px=r.cpv_fwhm_px, cpv_distinct=r.cpv_distinct, cpv_cutoff_pct=r.cpv_cutoff_pct,
                       cutoff_pct=r.cutoff_pct)
            if ref is not None and math.isfinite(r.stage_x) and math.isfinite(r.stage_y):
                row.update(x_best=ref["x_best"], z_best=ref["z_best"],
                           x_minus_xbest=r.stage_x - ref["x_best"], z_minus_zbest=r.stage_y - ref["z_best"],
                           d_signed_plane=signed_plane_distance(r.stage_x, r.stage_y, ref, theta),
                           d_in_plane=in_plane_offset(r.stage_x, r.stage_y, ref, theta))
            rows.append({k: ("" if isinstance(v, float) and not math.isfinite(v) else v)
                         for k, v in row.items()})
        return rows

    # ---------------------------------------------------- stable identities --
    def _remap_ids(self, ids, id_map, stored_keys):
        """Translate saved particle IDs to the IDs of the currently loaded particles.

        Preference: (1) the flags-file match (time + position), (2) the key
        stored in the project, (3) the same ID when nothing else is known.
        IDs whose particle is not loaded are kept unchanged.
        """
        if not self.catalog:
            return list(ids)
        if not hasattr(self, "_key_index_cache") or self._key_index_cache[0] is not self.catalog.records:
            self._key_index_cache = (self.catalog.records, self.catalog.key_index())
        index = self._key_index_cache[1]
        out = []
        for pid in ids:
            if pid in id_map:
                out.append(id_map[pid])
                continue
            key = stored_keys.get(pid)
            if key is not None:
                record = ParticleCatalog.resolve_key(index.get(tuple(key), []))
                if record is not None:
                    out.append(record.particle_id)
                    continue
                if pid in self.catalog._by_id:
                    continue  # that ID now belongs to a different ROI: drop it, never guess
            out.append(pid)
        return out

    def _remap_saved_ids(self, id_map, stored_keys):
        self._stored_keys = dict(stored_keys or {})
        for entry in self.saved_sets.values():
            entry["particle_ids"] = sorted(set(self._remap_ids(entry["particle_ids"], id_map, self._stored_keys)))
        self._refresh_saved_sets()

    def _particle_keys_for_project(self):
        wanted = set(self.selected_ids)
        for entry in self.saved_sets.values():
            wanted.update(entry.get("particle_ids", ()))
        keys = {}
        by_id = self.catalog._by_id if self.catalog else {}
        previous = getattr(self, "_stored_keys", {})
        for pid in wanted:                       # compact_keys sorts; keys are cached per particle
            record = by_id.get(pid)
            key = record.match_key if record is not None else previous.get(pid)
            if key is not None:
                keys[pid] = key
        return keys

    def _report_matching(self, report, prefix=""):
        if report is None or not report.rows:
            if prefix:
                self._toast(prefix)
            return
        text = (prefix + ". " if prefix else "") + "Saved flags: " + report.summary() + "."
        self._toast(text, 15000)
        if report.remapped or report.ambiguous:
            QMessageBox.information(
                self, "Flags matched after reprocessing",
                text + "\n\nParticles are matched by ROI time and camera position (StartX+EndX, StartY+EndY), "
                "so flags, labels and sets follow them even when reprocessing changes ROI numbers, lengths or "
                "focus values.")

    def choose_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "Add processed MAT files", "", "MAT files (*.mat)")
        self._import_errors = []
        for filename in names:
            self.import_file(Path(filename), refresh=False)
        self._report_import_errors()
        if names:
            self._view_locked = {"camera": False, "metric": False}
            self.apply_filters()

    def _removed_flags_file(self):
        if self.selection_file is None:
            return None
        return self.selection_file.with_name(self.selection_file.stem + "_removed.csv")

    def remove_sources_dialog(self):
        if not self.sources:
            QMessageBox.information(self, "Remove MAT files", "No MAT files are loaded.")
            return
        counts = {}
        for r in self.catalog.records:
            c = counts.setdefault(r.source_file, [0, 0, 0])
            c[0] += 1
            c[1] += r.status == "keep"
            c[2] += r.status == "reject"
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove MAT files from this project")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Tick the files to remove. The MAT files themselves are not deleted."))
        box = QListWidget()
        for src in self.sources:
            n, k, rj = counts.get(src, [0, 0, 0])
            item = QListWidgetItem(f"{Path(src).name}   —   {n:,} particles ({k:,} accepted, {rj:,} rejected)")
            item.setToolTip(src)
            item.setData(Qt.UserRole, src)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            box.addItem(item)
        layout.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Remove ticked files")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(640, 380)
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = [box.item(i).data(Qt.UserRole) for i in range(box.count())
                  if box.item(i).checkState() == Qt.Checked]
        if chosen:
            self.remove_sources(chosen)

    def remove_sources(self, paths, confirm=True):
        """Take MAT files out of the project, setting their particles' decisions aside (restored if re-added)."""
        paths = {str(p) for p in paths}
        gone = [r for r in self.catalog.records if r.source_file in paths]
        ids = {r.particle_id for r in gone}
        flagged = [r for r in gone if r.status != "undecided" or r.label]
        touched = [n for n, e in self.saved_sets.items() if ids & set(e.get("particle_ids", ()))]
        emptied = [n for n in touched if not (set(self.saved_sets[n].get("particle_ids", ())) - ids)]
        if confirm:
            text = (f"Remove {len(paths)} file(s) — {len(gone):,} particles — from this project?\n\n"
                    f"{len(flagged):,} of them are flagged or labelled. Their decisions are set aside and come back "
                    "automatically if you add the file again.")
            if touched:
                text += (f"\n\n{len(touched)} batch(es) lose these particles"
                         + (f"; {len(emptied)} become empty and are removed" if emptied else "") +
                         ". Batch membership is not restored if the file is added back.")
            text += "\n\nThis can't be undone with Ctrl+Z."
            if QMessageBox.question(self, "Remove MAT files", text) != QMessageBox.Yes:
                return False
        removed_file = self._removed_flags_file()
        if flagged and removed_file is not None:
            kept_rows = []
            if removed_file.exists():
                with removed_file.open(encoding="utf-8", newline="") as handle:
                    kept_rows = [row for row in csv.DictReader(handle) if row.get("particle_id") not in ids]
            ParticleCatalog(self.catalog.root, flagged).save_selection_file(removed_file)
            if kept_rows:
                with removed_file.open("a", encoding="utf-8", newline="") as handle:
                    fields = list(kept_rows[0].keys())
                    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                    writer.writerows(kept_rows)
        self.catalog.records = [r for r in self.catalog.records if r.source_file not in paths]
        self.catalog._by_id = {r.particle_id: r for r in self.catalog.records}
        self.sources = [src for src in self.sources if src not in paths]
        self.selected_ids -= ids
        if self.current_id in ids:
            self.current_id = None
        for name in touched:
            entry = self.saved_sets[name]
            entry["particle_ids"] = sorted(set(entry.get("particle_ids", ())) - ids)
            if not entry["particle_ids"]:
                del self.saved_sets[name]
                if self.active_reference_set == name:
                    self.active_reference_set = None
        self.undo_stack.clear()     # snapshots refer to particles that are no longer loaded
        self.redo_stack.clear()
        self._thumbs.clear()
        if self.store is not None:
            self.store.clear()
        self._session_of = None
        self._invalidate_reference_steps()
        self._refresh_saved_sets()
        self.autosave()
        self.save_project_if_open()
        self.apply_filters()
        self._update_window_title()
        self._toast(f"Removed {len(paths)} file(s) ({len(gone):,} particles) from the project. "
                    "Add a file back to restore its decisions.")
        return True

    def import_file(self, file: Path, refresh: bool = True) -> None:
        path = Path(file).resolve()
        if self.catalog is None:
            self._maybe_use_shared_reference_library(path)
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
        if not self._restoring:
            removed = self._removed_flags_file()
            if removed is not None and removed.exists():
                self.catalog.apply_selection_file(removed)      # decisions set aside when the file was removed
            if self.selection_file.exists():
                self._last_match_report = self.catalog.apply_selection_file(self.selection_file)
            apply_label_dimensions(self.catalog.records, self.labels)
        if self.position_file or self.extra_position_files:
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
        self._maybe_use_shared_reference_library(directory)
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
        if any(e.get("best_focus") for e in self.saved_sets.values()) or not self.catalog or not self.catalog.records:
            return
        if self.best_x_set.value() != 0 or self.best_z_set.value() != 0:
            return
        xs = [r.stage_x for r in self.catalog.records if math.isfinite(r.stage_x)]
        zs = [r.stage_y for r in self.catalog.records if math.isfinite(r.stage_y)]
        if xs:
            self.best_x_set.setValue(float(np.median(xs)))
        if zs:
            self.best_z_set.setValue(float(np.median(zs)))

    def _position_specs(self):
        specs = []
        if self.position_file:
            specs.append((self.position_file, self.position_series, self.position_time_basis))
        specs += [(e["path"], e.get("series"), e.get("time_basis")) for e in self.extra_position_files]
        return specs

    def _position_log(self):
        """One stage log from the project's stage CSV(s) — several after merging projects."""
        logs = [PositionLog.from_text(path, series, basis) for path, series, basis in self._position_specs()
                if path and Path(path).exists()]
        if not logs:
            return None
        if len(logs) == 1:
            return logs[0]
        return PositionLog(np.concatenate([l.times for l in logs]), np.concatenate([l.x for l in logs]),
                           np.concatenate([l.y for l in logs]), np.concatenate([l.ends for l in logs]))

    def _apply_stage_positions(self) -> int:
        if not (self.position_file or self.extra_position_files) or self.catalog is None or not self.catalog.records:
            return 0
        log = self._position_log()
        if log is None:
            return 0
        self.catalog.attach_position_log(log)
        return sum(math.isfinite(r.stage_x) and math.isfinite(r.stage_y) for r in self.catalog.records)

    def stage_check(self):
        """Burst check of the stage positions (data.stage_burst_check) for the loaded particles."""
        log = self._position_log() if self.catalog else None
        if log is None:
            return None
        return stage_burst_check([r.time_matlab for r in self.catalog.records], log)

    def _warn_split_bursts(self):
        try:
            report = self.stage_check()
        except Exception:
            return None
        if report and report["split"]:
            QMessageBox.warning(
                self, "Stage CSV check",
                f"{report['split']} of {report['bursts']} bursts of particles ({report['particles_in_split']:,} "
                "particles) fall across two stage positions.\n\nParticles are recorded in bursts after each stage "
                "move, so each burst should have one position. Usually the CSV's times are on a different basis "
                "from the particle times (e.g. 'roi' instead of 'full_frame'), or a time is rounded into a burst. "
                "Re-import and choose another time basis, or check the CSV around those times.")
        return report

    def reload_stage_positions(self):
        """Re-read the project's stage CSV(s) after editing them; batches keep their saved X/Z."""
        if not self.catalog or not self._position_specs():
            self.statusBar().showMessage("No stage CSV in this project — use Import stage positions CSV…", 6000)
            return 0
        before = {r.particle_id: (r.stage_x, r.stage_y) for r in self.catalog.records}
        try:
            matched = self._apply_stage_positions()
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Stage CSV", str(exc))
            return 0
        changed = sum(1 for r in self.catalog.records
                      if not np.allclose(before.get(r.particle_id, (np.nan, np.nan)), (r.stage_x, r.stage_y),
                                         equal_nan=True))
        self._steps_cache = None
        self._draft_reference_cache = None
        self._refresh_saved_sets()
        self._refresh_views()
        self.save_project_if_open()
        self._toast(f"Stage CSV reloaded: {matched:,} particles have X/Z; {changed:,} changed position.", 12000)
        self._warn_split_bursts()
        return changed

    def _use_stage_axis_after_import(self, valid_count: int) -> None:
        if valid_count and self.position_metric.currentData() == "time":
            self.position_metric.setCurrentIndex(self.position_metric.findData("stage_x"))

    def choose_position_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import stage positions (CSV)", "",
            "Stage position CSV (*.csv);;Legacy position logs (*.txt *.tsv);;All files (*)")
        if not filename:
            return
        self.position_series, self.position_time_basis = None, None
        if Path(filename).suffix.lower() in (".csv", ".tsv"):
            try:
                choices = PositionLog.csv_choices(filename)
            except Exception as exc:
                QMessageBox.critical(self, "Stage CSV", str(exc))
                return
            # Particle type (beads / analogues) is decided by flagging + labels, so
            # use every series unless the user wants to restrict it.
            if len(choices["series"]) > 1:
                options = ["All series (recommended)"] + choices["series"]
                series, ok = QInputDialog.getItem(
                    self, "Stage CSV series", "This CSV has several series. Use stage positions from:",
                    options, 0, False)
                if not ok:
                    return
                self.position_series = None if series == options[0] else series
            if len(choices["time_basis"]) > 1:
                basis_options = choices["time_basis"]
                default = (basis_options.index("full_frame") if "full_frame" in basis_options else
                           basis_options.index("roi") if "roi" in basis_options else 0)
                basis, ok = QInputDialog.getItem(
                    self, "Stage CSV time basis",
                    "Rows exist for several timestamp sources. Which times match the particles?\n"
                    "Since v14 each ROI has the time of its camera frame, so frame times (full_frame) match.\n"
                    "After import, each burst of particles should get a single stage position: you are warned\n"
                    "if bursts are split between positions (a sign of the wrong time basis).",
                    basis_options, default, False)
                if not ok:
                    return
                self.position_time_basis = basis
        self.load_position_file(filename)

    def load_position_file(self, filename):
        self.position_file = filename
        if not self.catalog:
            return
        try:
            matched = self._apply_stage_positions()
            conflicts = PositionLog.from_text(self.position_file, self.position_series,
                                              self.position_time_basis).conflicting_times()
        except Exception as exc:
            QMessageBox.critical(self, "Position log", str(exc))
            return
        if conflicts:
            QMessageBox.warning(self, "Stage CSV",
                                f"{conflicts} timestamps appear more than once with different stage positions "
                                "(e.g. two series or time bases overlapping). Re-import and choose one series / "
                                "time basis if the positions look wrong.")
        self._suggest_initial_best_reference()
        self._view_locked["metric"] = False
        self._use_stage_axis_after_import(matched)
        self._refresh_views()
        self._toast(f"Stage CSV loaded: {matched:,} / {len(self.catalog.records):,} particles have X/Z positions.",
                    12000)
        self._warn_split_bursts()
        if matched == 0:
            QMessageBox.warning(self, "No stage matches",
                                "The CSV was read, but no particles matched its times. Check the calibration "
                                "series (beads/analogue), time basis and clock conventions.")

    def load_directory(self, directory: str, position_file: Optional[str] = None) -> None:
        self._last_match_report = None
        self._maybe_use_shared_reference_library(directory)
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
        self._report_matching(getattr(self, "_last_match_report", None), f"Loaded {Path(directory).name}")

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
        """Flags changed: write them shortly (batched), not in the middle of the action."""
        if self.catalog and self.selection_file:
            self._pending_flags = True
            self._mark_unsaved()

    def save_project_if_open(self):
        if self.project_file:
            self._pending_project = True
            self._mark_unsaved()

    def _mark_unsaved(self):
        if hasattr(self, "save_indicator"):
            self.save_indicator.setText("● saving…")
            self.save_indicator.setStyleSheet("color:#a34a00; padding: 0 8px;")
        if hasattr(self, "_save_timer"):
            self._save_timer.start()

    def flush_saves(self, background: bool = False) -> None:
        """Write whatever is pending (project without the big CSV exports, or just flags).

        ``background``: take a snapshot here and write the files on a worker
        thread, so the window stays responsive while Dropbox-sized files are
        written.  Explicit saves, closing and project switches write in the
        foreground.
        """
        if hasattr(self, "_save_timer"):
            self._save_timer.stop()
        project = self._pending_project and self.project_file is not None
        flags = self._pending_flags or project
        self._pending_flags = self._pending_project = False
        if not flags or not self.catalog or not self.selection_file:
            return
        if not background:
            if project:
                self.save_project(exports=False)
            else:
                self._write_flags()
            return
        self._wait_for_writer()
        rows = self.catalog.selection_rows()
        flags_file = self.selection_file
        project_file = None
        config = None
        if project:
            config = self._project_config(raw=True)
            project_file = self.project_file
            self._exports_stale = True

        def work():
            try:
                ParticleCatalog.write_selection_rows(flags_file, rows)
                if config is not None:
                    config["selected_ids"] = compact_ids(config["selected_ids"])
                    config["named_sets"] = compact_project_sets(config["named_sets"])
                    config["particle_keys"] = compact_keys(config["particle_keys"])
                    config_text = json.dumps(config, separators=(",", ":"))
                    temp = project_file.with_suffix(".json.tmp")
                    temp.write_text(config_text, encoding="utf-8")
                    os.replace(temp, project_file)
                self._save_notifier.done.emit("ok", flags_file.name)
            except OSError as exc:
                self._save_notifier.done.emit("error", str(exc))

        self._writer_thread = threading.Thread(target=work, name="cpi-save", daemon=True)
        self._writer_thread.start()

    def _wait_for_writer(self):
        thread = getattr(self, "_writer_thread", None)
        if thread is not None and thread.is_alive():
            thread.join()

    def _save_finished(self, status, message):
        if status == "ok":
            self.save_indicator.setText(f"✓ Saved {datetime.now():%H:%M:%S} · {message}")
            self.save_indicator.setStyleSheet("color:#2b7a3d; padding: 0 8px;")
        else:
            self.save_indicator.setText(f"⚠ NOT saved: {message}")
            self.save_indicator.setStyleSheet("color:#c92a2a; font-weight:700; padding: 0 8px;")
            self._pending_flags = self._pending_project = True

    def _flags_file_has_rows(self) -> bool:
        try:
            with open(self.selection_file, encoding="utf-8") as handle:
                return sum(1 for _ in zip(range(2), handle)) > 1
        except OSError:
            return False

    def _write_flags(self) -> None:
        self._wait_for_writer()
        if self.catalog is not None and not self.catalog.records and self.selection_file \
                and self._flags_file_has_rows():
            # Nothing loaded (files missing or not processed): never replace saved flags with an empty file.
            self.save_indicator.setText("⚠ Flags not saved: no particles loaded — the saved flags file was kept")
            self.save_indicator.setStyleSheet("color:#c92a2a; font-weight:700; padding: 0 8px;")
            return
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
                       "x": r.image_x_px, "y": r.image_y_px, "illum": r.local_illum,
                       "cutoff": r.cutoff_pct}.get(key)
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
                ratio_min=values["ratio_min"], ratio_max=values["ratio_max"],
                illum_min=values["illum_min"], illum_max=values["illum_max"],
                cutoff_min=values["cutoff_min"], cutoff_max=values["cutoff_max"])
        except ValueError as exc:
            QMessageBox.warning(self, "Filters", str(exc))
            return
        self._last_filter_spec = spec
        mode = self.image_mode.currentData()
        records = self.catalog.records
        self._base_indices = [i for i in self.catalog.filtered_indices(spec)
                              if mode == "all" or records[i].is_full_frame == (mode == "full")]
        self._session_of = None
        self._illum_ref_cache = None
        if hasattr(self, "cpi_detected"):
            self._instrument_changed()
        self._invalidate_reference_steps()
        self._rebuild_browse()
        self._refresh_views()
        if hasattr(self, "window_list"):
            self._refresh_time_windows()
        if self.selection_file is not None and self.save_indicator.text() == "No data loaded":
            self.save_indicator.setText(f"Flags file: {self.selection_file.name}")
            self.save_indicator.setToolTip(str(self.selection_file))

    def clear_filters(self) -> None:
        for low, high in self.filter_fields.values():
            low.clear()
            high.clear()
        self.time_filter_checkbox.setChecked(False)
        self.active_time_window = None
        self.apply_filters()
        self._refresh_time_windows()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if self._nc_export_worker is not None and self._nc_export_worker.isRunning():
            self._nc_export_worker.requestInterruption()
            if not self._nc_export_worker.wait(1000):
                self._nc_export_worker.finished.connect(self.close)
                event.ignore()
                return
        self._edge_pending.clear()
        if self._edge_worker is not None and self._edge_worker.isRunning():
            self._edge_worker.requestInterruption()
            # Never destroy a running QThread or write a preview to disk.
            if not self._edge_worker.wait(750):
                self._edge_worker.finished.connect(self.close)
                event.ignore()
                return
        self._hide_hover()
        if self._mat_loader is not None:
            self._mat_loader.wait(5000)
        if not self._finish_current_project():
            event.ignore()
            return
        if self in CalibrationWindow._windows:
            CalibrationWindow._windows.remove(self)
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
QPushButton[role="neutral"]:disabled { color: #9aa9b8; background: #eef2f6; border-color: #d3dce6; }
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
