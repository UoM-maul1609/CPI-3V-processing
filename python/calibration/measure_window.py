"""Measuring window: view full frames, ROIs or image files and measure distances on them.

* Left-drag draws a measurement line at any angle (hold Shift to keep it at a
  multiple of 45°).  Drag an end of a line to move it.  Click a line to select
  it; Delete removes it.
* Mouse wheel zooms (about the pointer), right- or middle-drag pans.  Pixels
  are drawn as sharp squares, so line ends can be placed to a fraction of a
  pixel when zoomed in.
* Lengths are in pixels; with a pixel size set, also in µm.  "Pixel size from
  selected line" sets the pixel size from a line of known length (e.g. across
  graticule divisions).
* The profile below the table is the brightness along the selected line (with
  a little either side), to place ends on edges consistently.
* Export: the image with its measurements (and a scale bar if a pixel size is
  set) as PNG; all measurements of the session as CSV.

Measurements are kept per image for the session, so you can move between
images and come back.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPolygonF, QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsTextItem, QGraphicsView, QGroupBox, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QApplication)

try:
    from .measure import (BackgroundImages, ImageEntry, MatImages, Measurement, constrain, measurement_rows,
                          nice_bar_length, profile, read_image_file, write_csv)
except ImportError:  # pragma: no cover
    from measure import (BackgroundImages, ImageEntry, MatImages, Measurement, constrain, measurement_rows,
                         nice_bar_length, profile, read_image_file, write_csv)

LINE = QColor("#ffb000")
LINE_SELECTED = QColor("#00c2ff")
HANDLE_PX = 6          # screen pixels: grab distance for line ends
PICK_PX = 5            # screen pixels: grab distance for a line


def gray_qimage(arr: np.ndarray, stretch: bool = False) -> QImage:
    a = np.asarray(arr, float)
    if stretch and a.size:
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = (a - lo) / max(hi - lo, 1e-9) * 255.0
    a = np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8))
    h, w = a.shape
    return QImage(a.data, w, h, a.strides[0], QImage.Format_Grayscale8).copy()


def _fmt(m: Measurement, pixel_um: Optional[float]) -> str:
    text = f"{m.length_px:.2f} px"
    if pixel_um:
        text += f"  {m.length_um(pixel_um):.2f} µm"
    return text


class _LineGraphic:
    """Line, two end handles and a label for one measurement."""

    def __init__(self, scene: QGraphicsScene, number: int):
        self.line = QGraphicsLineItem()
        self.handles = [QGraphicsEllipseItem(-3.5, -3.5, 7, 7) for _ in range(2)]
        self.label = QGraphicsTextItem()
        for h in self.handles:
            h.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations)
            scene.addItem(h)
        self.label.setFlag(QGraphicsTextItem.ItemIgnoresTransformations)
        scene.addItem(self.line)
        scene.addItem(self.label)
        self.number = number

    def update(self, m: Measurement, pixel_um, selected: bool):
        colour = LINE_SELECTED if selected else LINE
        pen = QPen(colour, 2 if selected else 1.5)
        pen.setCosmetic(True)
        self.line.setPen(pen)
        self.line.setLine(m.x1, m.y1, m.x2, m.y2)
        for h, (x, y) in zip(self.handles, ((m.x1, m.y1), (m.x2, m.y2))):
            h.setPen(QPen(colour, 1.5))
            h.setBrush(QBrush(Qt.NoBrush))
            h.setPos(x, y)
        self.label.setHtml(f"<span style='background-color:rgba(0,0,0,150); color:{colour.name()};'>"
                           f"&nbsp;{self.number}: {_fmt(m, pixel_um)}&nbsp;</span>")
        self.label.setPos(max(m.x1, m.x2) + 0.5, (m.y1 + m.y2) / 2)

    def remove(self, scene):
        for item in [self.line, self.label] + self.handles:
            scene.removeItem(item)


class MeasureView(QGraphicsView):
    def __init__(self, window: "MeasureWindow"):
        super().__init__()
        self.window = window
        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setBackgroundBrush(QColor("#30363d"))
        self.setMouseTracking(True)
        self._pan = None
        self._drag = None           # ("new", m) or ("end", index, which)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.crop_mode = False
        self.crop_anchor = None
        self.crop_item = None

    # ---------------------------------------------------------------- display
    def set_image(self, qimage: QImage, keep_view=False):
        scene = self.scene()
        if self.pixmap_item is None:
            self.pixmap_item = QGraphicsPixmapItem()
            self.pixmap_item.setTransformationMode(Qt.FastTransformation)
            self.pixmap_item.setZValue(-10)
            scene.addItem(self.pixmap_item)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        scene.setSceneRect(QRectF(-20, -20, qimage.width() + 40, qimage.height() + 40))
        if not keep_view:
            self.fit()

    def fit(self):
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def zoom(self, factor):
        self.scale(factor, factor)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            self.window.export_rotation.setValue(self.window.export_rotation.value()+event.angleDelta().y()/120)
            event.accept(); return
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.zoom(1.25 ** steps)

    # ---------------------------------------------------------------- mouse
    def _scene(self, event) -> QPointF:
        p = self.mapToScene(event.position().toPoint())
        w, h = self.window.image_size()
        return QPointF(min(max(p.x(), 0.0), float(w)), min(max(p.y(), 0.0), float(h)))

    def _screen_dist(self, x, y, event) -> float:
        q = self.mapFromScene(QPointF(x, y))
        e = event.position()
        return math.hypot(q.x() - e.x(), q.y() - e.y())

    def mousePressEvent(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton or self.pixmap_item is None:
            return super().mousePressEvent(event)
        if self.crop_mode:
            self.crop_anchor = self._scene(event)
            if self.crop_item is not None:
                self.scene().removeItem(self.crop_item)
            self.crop_item = self.scene().addPolygon(QPolygonF(),
                                                  QPen(QColor('#ffe83b'),2,Qt.DashLine))
            return
        ms = self.window.current_measurements()
        # an end of a line?
        best = None
        for i, m in enumerate(ms):
            for which, (x, y) in enumerate(((m.x1, m.y1), (m.x2, m.y2))):
                d = self._screen_dist(x, y, event)
                if d <= HANDLE_PX and (best is None or d < best[0]):
                    best = (d, i, which)
        if best is not None:
            self._drag = ("end", best[1], best[2])
            self.window.select(best[1])
            return
        # on a line?
        for i, m in enumerate(ms):
            a, b = self.mapFromScene(QPointF(m.x1, m.y1)), self.mapFromScene(QPointF(m.x2, m.y2))
            e = event.position()
            vx, vy, wx, wy = b.x() - a.x(), b.y() - a.y(), e.x() - a.x(), e.y() - a.y()
            L2 = vx * vx + vy * vy
            t = 0 if L2 == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / L2))
            if math.hypot(wx - t * vx, wy - t * vy) <= PICK_PX:
                self.window.select(i)
                return
        p = self._scene(event)
        self._drag = ("new", Measurement(p.x(), p.y(), p.x(), p.y()))
        self.window.preview(self._drag[1])

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            d = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        p = self._scene(event)
        if self.crop_anchor is not None:
            from .export_transform import rotated_crop_rect
            w,h=self.window.image_size()
            polygon, fractions, _ = rotated_crop_rect(
                (self.crop_anchor.x(),self.crop_anchor.y()), (p.x(),p.y()),
                w,h,self.window.export_rotation.value(),self.window.crop_aspect_ratio())
            self.crop_item.setPolygon(polygon)
            self._crop_fraction = fractions
            return
        self.window.show_cursor(p.x(), p.y())
        if self._drag is None:
            return
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if self._drag[0] == "new":
            m = self._drag[1]
            m.x2, m.y2 = constrain(m.x1, m.y1, p.x(), p.y()) if shift else (p.x(), p.y())
            self.window.preview(m)
        else:
            _, i, which = self._drag
            m = self.window.current_measurements()[i]
            if which == 0:
                m.x1, m.y1 = constrain(m.x2, m.y2, p.x(), p.y()) if shift else (p.x(), p.y())
            else:
                m.x2, m.y2 = constrain(m.x1, m.y1, p.x(), p.y()) if shift else (p.x(), p.y())
            self.window.changed(i)

    def mouseReleaseEvent(self, event):
        if self.crop_anchor is not None and event.button()==Qt.LeftButton:
            self.crop_anchor=None
            if self.crop_item is not None:
                self.window.set_crop_rect(self._crop_fraction)
            return
        if self._pan is not None and event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan = None
            self.unsetCursor()
            return
        if self._drag is not None and event.button() == Qt.LeftButton:
            drag, self._drag = self._drag, None
            if drag[0] == "new":
                self.window.preview(None)
                if drag[1].length_px >= 0.5:
                    self.window.add(drag[1])
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.window.delete_selected()
            return
        if event.key() == Qt.Key_Escape and self._drag is not None:
            self._drag = None
            self.window.preview(None)
            return
        super().keyPressEvent(event)


class MeasureWindow(QMainWindow):
    """Measure distances on full frames, ROIs and image files."""

    def __init__(self, mat_files=(), parent=None, pixel_um: Optional[float] = None):
        super().__init__(parent)
        self.setWindowTitle("Measure images")
        self.resize(1500, 900)
        self._sources: Dict[str, object] = {}         # path -> MatImages / BackgroundImages / ndarray
        self._entries: List[ImageEntry] = []
        self.entry: Optional[ImageEntry] = None
        self.array: Optional[np.ndarray] = None
        self.measurements: Dict[tuple, List[Measurement]] = {}
        self.entries_seen: Dict[tuple, ImageEntry] = {}
        self.graphics: List[_LineGraphic] = []
        self.selected: Optional[int] = None
        self._preview_graphic: Optional[_LineGraphic] = None
        self._boundary_item = None

        split = QSplitter()
        self.setCentralWidget(split)

        # ---- left: what to look at
        left = QWidget()
        ll = QVBoxLayout(left)
        self.file_combo = QComboBox()
        self.file_combo.setToolTip("Processed MAT files of the project (and any you opened)")
        for f in mat_files:
            self.file_combo.addItem(Path(f).name, str(f))
        open_row = QHBoxLayout()
        b_mat = QPushButton("Open MAT file…")
        b_img = QPushButton("Open image file…")
        b_mat.clicked.connect(self.open_mat_dialog)
        b_img.clicked.connect(self.open_image_dialog)
        open_row.addWidget(b_mat)
        open_row.addWidget(b_img)
        self.show_combo = QComboBox()
        self.show_combo.addItem("Full frames in this file", "full")
        self.show_combo.addItem("All ROIs in this file", "all")
        self.show_combo.addItem("Backgrounds (full_backgrounds.mat in the same folder)", "bg")
        self.show_combo.addItem("Camera frames rebuilt: ROIs pasted on the nearest background", "frames")
        self.show_combo.setToolTip("The CPI stores only the ROIs of each camera frame (plus occasional full "
                                   "backgrounds). 'Camera frames rebuilt' puts each frame's ROIs back at their "
                                   "positions on the background nearest in time, with the ROI boxes outlined.")
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._list_changed)
        goto = QHBoxLayout()
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(0, 10_000_000)
        b_goto = QPushButton("Go to ROI")
        b_goto.clicked.connect(lambda: self.show_roi(self.file_combo.currentData(), self.goto_spin.value()))
        goto.addWidget(b_goto)
        goto.addWidget(self.goto_spin)
        b_frame = QPushButton("Whole camera frame of this ROI")
        b_frame.setToolTip("Show the camera frame the current ROI came from, rebuilt with the other ROIs of that frame")
        b_frame.clicked.connect(self.show_frame_of_current_roi)
        ll.addWidget(QLabel("File"))
        ll.addWidget(self.file_combo)
        ll.addLayout(open_row)
        ll.addWidget(self.show_combo)
        ll.addWidget(self.image_list, 1)
        ll.addLayout(goto)
        ll.addWidget(b_frame)
        split.addWidget(left)

        # ---- centre: the image
        centre = QWidget()
        cl = QVBoxLayout(centre)
        self.title = QLabel("Open a file, or choose an image on the left")
        self.title.setWordWrap(True)
        self.view = MeasureView(self)
        bar = QHBoxLayout()
        for text, slot in (("Fit", lambda: self.view.fit()), ("1:1", self._one_to_one),
                           ("Zoom +", lambda: self.view.zoom(1.5)), ("Zoom −", lambda: self.view.zoom(1 / 1.5))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        self.stretch_box = QCheckBox("Stretch contrast")
        self.stretch_box.toggled.connect(lambda *_: self._redraw_image(keep_view=True))
        self.boundary_box = QCheckBox("Show stored outline")
        self.boundary_box.setChecked(True)
        self.boundary_box.toggled.connect(lambda *_: self._draw_boundary())
        bar.addWidget(self.stretch_box)
        bar.addWidget(self.boundary_box)
        bar.addStretch(1)
        self.cursor_label = QLabel("")
        bar.addWidget(self.cursor_label)
        cl.addWidget(self.title)
        cl.addWidget(self.view, 1)
        cl.addLayout(bar)
        hint = QLabel("Left-drag: measure (Shift: 0/45/90°) · drag a line end to adjust · click a line to select, "
                      "Delete to remove · wheel: zoom · right/middle-drag: pan")
        hint.setStyleSheet("color:#667788")
        cl.addWidget(hint)
        split.addWidget(centre)

        # ---- right: measurements
        right = QWidget()
        rl = QVBoxLayout(right)
        px_box = QGroupBox("Pixel size")
        pf = QFormLayout(px_box)
        self.px_check = QCheckBox("Also show µm")
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.01, 1000.0)
        self.px_spin.setDecimals(4)
        self.px_spin.setSingleStep(0.01)
        self.px_spin.setSuffix(" µm / px")
        self.px_spin.setValue(pixel_um or 2.3)
        self.px_check.setChecked(pixel_um is not None)
        self.px_check.toggled.connect(lambda *_: self._refresh_measurements())
        self.px_spin.valueChanged.connect(lambda *_: self._refresh_measurements())
        b_cal = QPushButton("Pixel size from selected line…")
        b_cal.setToolTip("Enter the true length of the selected line (e.g. N graticule divisions) to set the pixel size")
        b_cal.clicked.connect(self.pixel_size_from_line)
        pf.addRow(self.px_check, self.px_spin)
        pf.addRow(b_cal)
        rl.addWidget(px_box)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "length px", "length µm", "angle °", "from x, y", "to x, y"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._table_selection)
        rl.addWidget(self.table, 2)
        self.summary = QLabel("")
        rl.addWidget(self.summary)
        row = QHBoxLayout()
        b_del = QPushButton("Delete selected")
        b_clear = QPushButton("Clear this image")
        b_del.clicked.connect(self.delete_selected)
        b_clear.clicked.connect(self.clear_image)
        row.addWidget(b_del)
        row.addWidget(b_clear)
        rl.addLayout(row)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.profile_fig = Figure(figsize=(4, 2.4), constrained_layout=True)
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvasQTAgg(self.profile_fig)
        rl.addWidget(QLabel("Brightness along the selected line"))
        rl.addWidget(self.profile_canvas, 1)

        ex = QGroupBox("Export")
        exl = QFormLayout(ex)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(0, 16)
        self.scale_spin.setSpecialValueText("auto")
        self.scale_spin.setSuffix(" ×")
        self.scale_spin.setToolTip("Image pixels are enlarged by this factor (sharp squares) in the PNG; "
                                   "auto: about 1000 px across")
        self.bar_check = QCheckBox("Scale bar (needs pixel size)")
        self.bar_check.setChecked(True)
        self.caption_check = QCheckBox("Caption (file, image, pixel size)")
        self.caption_check.setChecked(True)
        self.export_rotation = QDoubleSpinBox()
        self.export_rotation.setRange(-360,360); self.export_rotation.setDecimals(1)
        self.export_rotation.setSingleStep(1); self.export_rotation.setSuffix('° clockwise')
        self.export_rotation.valueChanged.connect(self.preview_rotation)
        self.export_crop = []
        for name, val in (("Crop left (%)", 0), ("Crop top (%)", 0),
                          ("Crop right (%)", 100), ("Crop bottom (%)", 100)):
            spin = QDoubleSpinBox(); spin.setRange(0, 100); spin.setValue(val)
            spin.setSingleStep(5); spin.setDecimals(3)
            self.export_crop.append(spin)
            exl.addRow(name, spin)
        self.crop_ratio = QComboBox()
        for label,ratio in [('Free',0),('1:1',1),('4:3',4/3),('3:2',1.5),('16:9',16/9),('9:16',9/16),('Custom',-1)]:
            self.crop_ratio.addItem(label,ratio)
        self.crop_ratio_width = QDoubleSpinBox(); self.crop_ratio_width.setRange(.01,10000); self.crop_ratio_width.setValue(4)
        self.crop_ratio_height = QDoubleSpinBox(); self.crop_ratio_height.setRange(.01,10000); self.crop_ratio_height.setValue(3)
        exl.addRow('Crop aspect ratio',self.crop_ratio)
        exl.addRow('Custom aspect width',self.crop_ratio_width)
        exl.addRow('Custom aspect height',self.crop_ratio_height)
        self.crop_drag_button = QPushButton('Drag export crop on image')
        self.crop_drag_button.setCheckable(True)
        self.crop_drag_button.toggled.connect(lambda on: setattr(self.view,'crop_mode',on))
        exl.addRow(self.crop_drag_button)
        exl.addRow("Rotate image / PNG", self.export_rotation)
        self.export_pixel_info = QCheckBox("Include original pixel dimensions and x/y axis arrows")
        self.export_pixel_info.setChecked(True)
        exl.addRow(self.export_pixel_info)
        b_png = QPushButton("Export image as PNG…")
        b_csv = QPushButton("Export all measurements as CSV…")
        b_png.clicked.connect(self.export_png_dialog)
        b_csv.clicked.connect(self.export_csv_dialog)
        exl.addRow("Enlarge", self.scale_spin)
        exl.addRow(self.bar_check)
        exl.addRow(self.caption_check)
        exl.addRow(b_png)
        exl.addRow(b_csv)
        rl.addWidget(ex)
        split.addWidget(right)
        split.setSizes([280, 850, 370])

        self.file_combo.currentIndexChanged.connect(lambda *_: self._fill_list())
        self.show_combo.currentIndexChanged.connect(lambda *_: self._fill_list())
        if mat_files:
            self._fill_list()

    # ================================================================ sources
    def _source(self, path):
        src = self._sources.get(path)
        if src is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                if Path(path).name == "full_backgrounds.mat":
                    src = BackgroundImages(path)
                elif Path(path).suffix.lower() == ".mat":
                    src = MatImages(path)
                else:
                    src = read_image_file(path)
            finally:
                QApplication.restoreOverrideCursor()
            self._sources[path] = src
        return src

    def _add_file(self, path):
        path = str(path)
        i = self.file_combo.findData(path)
        if i < 0:
            self.file_combo.addItem(Path(path).name, path)
            i = self.file_combo.count() - 1
        self.file_combo.setCurrentIndex(i)

    def open_mat_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open processed MAT file", "", "MAT files (*.mat)")
        if path:
            self._add_file(path)

    def open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "",
                                              "Images (*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.gif);;All files (*)")
        if path:
            self.open_image_file(path)

    def open_image_file(self, path):
        try:
            arr = self._source(str(path))
        except Exception as exc:
            QMessageBox.warning(self, "Open image", str(exc))
            return
        entry = ImageEntry(str(path), "image file", 0, tuple(arr.shape))
        self._show(entry, arr, None)

    def _fill_list(self):
        self.image_list.blockSignals(True)
        self.image_list.clear()
        self._entries = []
        path = self.file_combo.currentData()
        if not path:
            self.image_list.blockSignals(False)
            return
        mode = self.show_combo.currentData()
        try:
            if path.lower().endswith(".mat") and Path(path).name != "full_backgrounds.mat":
                if mode == "frames":
                    entries = self._source(path).frame_entries()
                elif mode == "bg":
                    bg = Path(path).parent / "full_backgrounds.mat"
                    entries = self._source(str(bg)).entries() if bg.exists() else []
                else:
                    entries = self._source(path).entries()
                    if mode == "full":
                        entries = [e for e in entries if e.kind == "full frame"]
            elif Path(path).name == "full_backgrounds.mat":
                entries = self._source(path).entries()
            else:
                entries = []
        except Exception as exc:
            QMessageBox.warning(self, "Measure images", f"Cannot read {path}:\n{exc}")
            entries = []
        self._entries = entries
        for e in entries:
            if e.kind == "frame":
                n_roi = len(self._source(e.source).frames()[e.index][1])
                label = f"frame {e.index}   {e.time[11:23] if e.time else ''}   {n_roi} ROI{'s' if n_roi != 1 else ''}"
            else:
                label = f"{'BG' if e.kind == 'background' else 'ROI'} {e.index}   " \
                        f"{e.shape[1] if len(e.shape) > 1 else '?'}×{e.shape[0] if e.shape else '?'}   " \
                        f"{e.time[11:19] if e.time else ''}"
            if self.measurements.get(e.key):
                label += f"   📏{len(self.measurements[e.key])}"
            self.image_list.addItem(QListWidgetItem(label))
        if not entries:
            self.image_list.addItem(QListWidgetItem({"full": "(no full frames in this file — try 'All ROIs' or "
                                                             "'Backgrounds')",
                                                     "all": "(no ROIs)",
                                                     "bg": "(no full_backgrounds.mat in this folder)",
                                                     "frames": "(no particle ROIs)"}.get(mode, "")))
        self.image_list.blockSignals(False)

    def _list_changed(self, row):
        if 0 <= row < len(self._entries):
            e = self._entries[row]
            src = self._source(e.source)
            if e.kind == "frame":
                self._show_frame(e)
            else:
                self._show(e, src.image(e.index), src.boundary(e.index))

    def _backgrounds_for(self, mat_path):
        bg = Path(mat_path).parent / "full_backgrounds.mat"
        if not bg.exists():
            return None
        try:
            return self._source(str(bg))
        except Exception:
            return None

    def _show_frame(self, e: ImageEntry):
        src = self._source(e.source)
        bank = self._backgrounds_for(e.source)
        frame, bg_index = src.frame_image(e.index, bank)
        e.shape = tuple(frame.shape)
        rois = src.frames()[e.index][1]
        self._show(e, frame, src.frame_boundary(e.index))
        note = (f"background {bg_index} ({bank.entries()[bg_index].time[11:19]})" if bg_index is not None
                else "no full_backgrounds.mat: grey behind the ROIs")
        self.title.setText(self.title.text() + f"<br>ROIs {', '.join(map(str, rois))} on {note}")

    def show_frame_of_current_roi(self):
        if self.entry is None or self.entry.kind not in ("ROI", "full frame"):
            QMessageBox.information(self, "Camera frame", "Show an ROI first.")
            return
        src = self._source(self.entry.source)
        roi = self.entry.index
        for k, (_, rois) in enumerate(src.frames()):
            if roi in rois:
                self.show_combo.setCurrentIndex(self.show_combo.findData("frames"))
                for row, e in enumerate(self._entries):
                    if e.kind == "frame" and e.index == k:
                        self.image_list.setCurrentRow(row)
                        return
        QMessageBox.information(self, "Camera frame", f"ROI {roi} is not part of a rebuilt camera frame.")

    def show_roi(self, mat_path, index: int):
        """Show ROI ``index`` of a MAT file (adds the file if needed)."""
        if not mat_path:
            return
        self._add_file(mat_path)
        src = self._source(str(mat_path))
        if not isinstance(src, MatImages) or not 0 <= index < len(src.images):
            QMessageBox.warning(self, "Measure images", f"No ROI {index} in {Path(mat_path).name}")
            return
        entry = src.entries()[index]
        if self.show_combo.currentData() != "all" and entry.kind != "full frame":
            self.show_combo.setCurrentIndex(self.show_combo.findData("all"))
        for row, e in enumerate(self._entries):
            if e.key == entry.key:
                self.image_list.setCurrentRow(row)
                return
        self._show(entry, src.image(index), src.boundary(index))

    # ================================================================ display
    def image_size(self):
        if self.array is None:
            return 0, 0
        return self.array.shape[1], self.array.shape[0]

    def _show(self, entry: ImageEntry, arr, boundary):
        self.entry = entry
        self.array = np.asarray(arr)
        self._boundary = boundary
        self.entries_seen[entry.key] = entry
        self.measurements.setdefault(entry.key, [])
        self.selected = None
        h, w = self.array.shape[:2]
        self.title.setText(f"<b>{entry.title()}</b> · {w} × {h} px")
        self._redraw_image(keep_view=False)
        self._rebuild_graphics()
        self._draw_boundary()

    def _redraw_image(self, keep_view=True):
        if self.array is None:
            return
        self.view.set_image(gray_qimage(self.array, self.stretch_box.isChecked()), keep_view=keep_view)

    def _draw_boundary(self):
        scene = self.view.scene()
        if self._boundary_item is not None:
            scene.removeItem(self._boundary_item)
            self._boundary_item = None
        b = getattr(self, "_boundary", None)
        if b is None or not self.boundary_box.isChecked():
            return
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        first = True
        for r, c in b:
            if not (np.isfinite(r) and np.isfinite(c)):
                first = True
                continue
            pt = QPointF(c + 0.5, r + 0.5)          # stored (row, col) of pixel centres
            if first:
                path.moveTo(pt)
                first = False
            else:
                path.lineTo(pt)
        pen = QPen(QColor("#ff654d"), 1)
        pen.setCosmetic(True)
        self._boundary_item = scene.addPath(path, pen)

    def _one_to_one(self):
        self.view.resetTransform()

    def show_cursor(self, x, y):
        if self.array is None:
            return
        c, r = int(x), int(y)
        h, w = self.array.shape[:2]
        v = f"  value {self.array[r, c]:.0f}" if 0 <= r < h and 0 <= c < w else ""
        o = self.entry.origin if self.entry is not None else None
        cam = f"  (camera {o[0] + x:.1f}, {o[1] + y:.1f})" if o is not None and tuple(o) != (0, 0) else ""
        self.cursor_label.setText(f"x {x:.2f}, y {y:.2f} px{cam}{v}")

    # ================================================================ measurements
    def pixel_um(self) -> Optional[float]:
        return self.px_spin.value() if self.px_check.isChecked() else None

    def current_measurements(self) -> List[Measurement]:
        if self.entry is None:
            return []
        return self.measurements.setdefault(self.entry.key, [])

    def _rebuild_graphics(self):
        scene = self.view.scene()
        for g in self.graphics:
            g.remove(scene)
        self.graphics = []
        for n, _ in enumerate(self.current_measurements(), 1):
            self.graphics.append(_LineGraphic(scene, n))
        self._refresh_measurements()

    def _refresh_measurements(self):
        ms = self.current_measurements()
        px = self.pixel_um()
        for i, (g, m) in enumerate(zip(self.graphics, ms)):
            g.number = i + 1
            g.update(m, px, i == self.selected)
        self.table.blockSignals(True)
        self.table.setRowCount(len(ms))
        for i, m in enumerate(ms):
            values = [str(i + 1), f"{m.length_px:.3f}", f"{m.length_um(px):.3f}" if px else "—",
                      f"{m.angle_deg:.1f}", f"{m.x1:.2f}, {m.y1:.2f}", f"{m.x2:.2f}, {m.y2:.2f}"]
            for j, v in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(v))
        if self.selected is not None and self.selected < len(ms):
            self.table.selectRow(self.selected)
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)
        if ms:
            L = np.array([m.length_px for m in ms])
            text = f"{len(ms)} line{'s' if len(ms) != 1 else ''}: mean {L.mean():.3f} px"
            if len(ms) > 1:
                text += f" ± {L.std(ddof=1):.3f} (sd)"
            if px:
                text += f"  =  {L.mean() * px:.3f} µm"
            self.summary.setText(text)
        else:
            self.summary.setText("No measurements on this image")
        self._draw_profile()
        self._update_list_marker()

    def _update_list_marker(self):
        if self.entry is None:
            return
        for row, e in enumerate(self._entries):
            if e.key == self.entry.key:
                item = self.image_list.item(row)
                text = item.text().split("   📏")[0]
                n = len(self.current_measurements())
                item.setText(text + (f"   📏{n}" if n else ""))

    def _draw_profile(self):
        ax = self.profile_ax
        ax.clear()
        ms = self.current_measurements()
        if self.selected is None or self.selected >= len(ms) or self.array is None:
            ax.text(0.5, 0.5, "select a line", ha="center", va="center", transform=ax.transAxes, color="#889")
            ax.set_axis_off()
            self.profile_canvas.draw_idle()
            return
        m = ms[self.selected]
        ext = max(2.0, 0.15 * m.length_px)
        s, v = profile(self.array, m, extend=ext)
        px = self.pixel_um()
        k = px if px else 1.0
        ax.set_axis_on()
        ax.plot(s * k, v, color="#1f6fb2", lw=1.2)
        for x in (0.0, m.length_px * k):
            ax.axvline(x, color="#ffb000", lw=1.2)
        ax.set_xlabel("distance along line [µm]" if px else "distance along line [px]")
        ax.set_ylabel("brightness")
        ax.grid(alpha=0.3)
        self.profile_canvas.draw_idle()

    def preview(self, m: Optional[Measurement]):
        scene = self.view.scene()
        if m is None:
            if self._preview_graphic is not None:
                self._preview_graphic.remove(scene)
                self._preview_graphic = None
            return
        if self._preview_graphic is None:
            self._preview_graphic = _LineGraphic(scene, len(self.current_measurements()) + 1)
        self._preview_graphic.update(m, self.pixel_um(), True)

    def add(self, m: Measurement):
        self.current_measurements().append(m)
        self.graphics.append(_LineGraphic(self.view.scene(), len(self.graphics) + 1))
        self.selected = len(self.graphics) - 1
        self._refresh_measurements()

    def changed(self, i):
        self.selected = i
        self._refresh_measurements()

    def select(self, i: Optional[int]):
        self.selected = i
        self._refresh_measurements()

    def _table_selection(self):
        rows = sorted({ix.row() for ix in self.table.selectedIndexes()})
        self.selected = rows[0] if rows else None
        ms = self.current_measurements()
        px = self.pixel_um()
        for i, (g, m) in enumerate(zip(self.graphics, ms)):
            g.update(m, px, i == self.selected)
        self._draw_profile()

    def delete_selected(self):
        ms = self.current_measurements()
        rows = sorted({ix.row() for ix in self.table.selectedIndexes()}, reverse=True) or \
            ([self.selected] if self.selected is not None else [])
        for r in rows:
            if 0 <= r < len(ms):
                del ms[r]
        self.selected = None
        self._rebuild_graphics()

    def clear_image(self):
        self.current_measurements().clear()
        self.selected = None
        self._rebuild_graphics()

    def pixel_size_from_line(self):
        ms = self.current_measurements()
        if self.selected is None or self.selected >= len(ms):
            QMessageBox.information(self, "Pixel size", "Draw a line across a known distance and select it first.")
            return
        m = ms[self.selected]
        value, ok = QInputDialog.getDouble(self, "Pixel size from a line",
                                           f"True length of line {self.selected + 1} ({m.length_px:.3f} px) in µm:",
                                           100.0, 0.001, 1e6, 3)
        if ok and value > 0:
            self.px_spin.setValue(value / m.length_px)
            self.px_check.setChecked(True)
            self._refresh_measurements()

    # ================================================================ export
    def render_png(self, scale: int = 0, upright_labels: bool = False) -> QImage:
        """The current image enlarged ``scale`` times (sharp pixels) with its measurements drawn on."""
        if self.array is None:
            raise ValueError("No image")
        h, w = self.array.shape[:2]
        if scale <= 0:
            scale = int(min(16, max(1, round(1000 / max(w, h)))))
        px = self.pixel_um()
        caption_h = 0
        font = QFont()
        font.setPixelSize(max(11, int(min(w, h) * scale / 45)))
        if self.caption_check.isChecked() and not upright_labels:
            caption_h = int(font.pixelSize() * 2.0)
        out = QImage(w * scale, h * scale + caption_h, QImage.Format_RGB32)
        out.fill(QColor("white"))
        p = QPainter(out)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(QRectF(0, 0, w * scale, h * scale), gray_qimage(self.array, self.stretch_box.isChecked()))
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(font)
        b = getattr(self, "_boundary", None)
        if b is not None and self.boundary_box.isChecked():
            pen = QPen(QColor("#ff654d"), max(1.0, scale / 3))
            p.setPen(pen)
            for (r0, c0), (r1, c1) in zip(b[:-1], b[1:]):
                if np.isfinite([r0, c0, r1, c1]).all():
                    p.drawLine(QPointF((c0 + 0.5) * scale, (r0 + 0.5) * scale),
                               QPointF((c1 + 0.5) * scale, (r1 + 0.5) * scale))
        lw = max(1.5, scale / 2.5)
        for n, m in enumerate(self.current_measurements(), 1):
            p.setPen(QPen(LINE, lw))
            a, c = QPointF(m.x1 * scale, m.y1 * scale), QPointF(m.x2 * scale, m.y2 * scale)
            p.drawLine(a, c)
            tick = 3 * lw
            ux, uy = (m.x2 - m.x1), (m.y2 - m.y1)
            L = math.hypot(ux, uy) or 1.0
            nx, ny = -uy / L * tick, ux / L * tick            # end ticks across the line
            for q in (a, c):
                p.drawLine(QPointF(q.x() - nx, q.y() - ny), QPointF(q.x() + nx, q.y() + ny))
            text = f"{n}: {_fmt(m, px)}"
            fm = p.fontMetrics()
            tx = max(m.x1, m.x2) * scale + 2 * tick
            ty = (m.y1 + m.y2) / 2 * scale
            tw = fm.horizontalAdvance(text)
            tx = min(tx, w * scale - tw - 4)
            r = QRectF(tx - 3, ty - fm.ascent() - 2, tw + 6, fm.height() + 4)
            if not upright_labels:
                p.fillRect(r, QColor(0, 0, 0, 150))
                p.setPen(LINE)
                p.drawText(QPointF(tx, ty), text)
        if px and self.bar_check.isChecked():
            bar_um = nice_bar_length(0.25 * w * px)
            bar_px = bar_um / px * scale
            fm = p.fontMetrics()
            x0, y0 = 0.04 * w * scale, h * scale - 0.04 * h * scale
            label = f"{bar_um:g} µm"
            p.fillRect(QRectF(x0 - 6, y0 - fm.height() - 12, max(bar_px, fm.horizontalAdvance(label)) + 12,
                              fm.height() + 22), QColor(0, 0, 0, 140))
            p.fillRect(QRectF(x0, y0 - 4, bar_px, max(3.0, scale / 1.5)), QColor("white"))
            p.setPen(QColor("white"))
            if not upright_labels:
                p.drawText(QPointF(x0, y0 - 8), label)
        if caption_h and not upright_labels:
            p.setPen(QColor("#222222"))
            text = self.entry.title() + f"  ·  {w}×{h} px" + (f"  ·  pixel {px:.4g} µm" if px else "")
            p.drawText(QRectF(6, h * scale, w * scale - 12, caption_h), Qt.AlignVCenter | Qt.AlignLeft, text)
        p.end()
        return out

    def crop_aspect_ratio(self):
        ratio=self.crop_ratio.currentData()
        if ratio==-1:
            return self.crop_ratio_width.value()/self.crop_ratio_height.value()
        return ratio or 0

    def set_crop_rect(self, fractions):
        if fractions is not None:
            values=fractions
            for spin,value in zip(self.export_crop,values):
                spin.setValue(round(max(0,min(100,value*100)),3))

    def preview_rotation(self, angle):
        if self.view.pixmap_item is not None:
            self.view.resetTransform()
            self.view.fit()
            self.view.rotate(float(angle))
            self.view.centerOn(self.view.pixmap_item)

    def export_png(self, path, scale: int = 0) -> str:
        img = self.render_png(scale, upright_labels=True)
        try:
            from .export_transform import crop_rotate_image
        except ImportError:
            from export_transform import crop_rotate_image
        crop = [spin.value() / 100 for spin in self.export_crop]
        angle = self.export_rotation.value()
        img = crop_rotate_image(img, crop, angle)
        # Render all editorial labels after rotating/cropping, in paper orientation.
        from PySide6.QtGui import QFont
        p = QPainter(img)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            font=QFont(); font.setBold(True); font.setPixelSize(max(12,min(30,img.width()//52)))
            p.setFont(font)
            fm=p.fontMetrics()
            y=6
            px=self.pixel_um()
            def line(text):
                nonlocal y
                while fm.horizontalAdvance(text)>img.width()-18 and font.pixelSize()>10:
                    font.setPixelSize(font.pixelSize()-1); p.setFont(font)
                metrics=p.fontMetrics()
                p.fillRect(QRectF(3,y,max(10,min(img.width()-3,metrics.horizontalAdvance(text)+13)),metrics.height()+5), QColor(0,0,0,170))
                p.setPen(QColor('white'))
                p.drawText(9,int(y+metrics.ascent()+2),text)
                y += metrics.height()+8
            if self.caption_check.isChecked():
                line(self.entry.title() + f' · {self.array.shape[1]}×{self.array.shape[0]} source px')
            for n,m in enumerate(self.current_measurements(),1):
                line(f'{n}: {_fmt(m,px)}')
            if px and self.bar_check.isChecked():
                line(f'Scale bar: {nice_bar_length(.25*self.array.shape[1]*px):g} µm')
        finally:
            p.end()
        if self.export_pixel_info.isChecked():
            # The arrows explicitly describe ORIGINAL CPI image coordinates.
            # Rotating the export rotates the arrow vectors consistently.
            from PySide6.QtCore import QPointF
            import math
            s = max(25, min(img.width(), img.height()) // 7)
            if img.width() > 3*s and img.height() > 3*s:
                painter = QPainter(img)
                try:
                    painter.setRenderHint(QPainter.Antialiasing, True)
                    theta = math.radians(angle)
                    ex = (math.cos(theta), math.sin(theta))
                    ey = (-math.sin(theta), math.cos(theta))
                    origin = QPointF(s + 12, s + 12)
                    painter.setPen(QPen(QColor('#000000'), 6))
                    for dx,dy in (ex,ey):
                        painter.drawLine(origin, QPointF(origin.x()+dx*s*.75, origin.y()+dy*s*.75))
                    painter.setPen(QPen(QColor('#ffff38'), 3))
                    for name,(dx,dy) in (('x',ex), ('y',ey)):
                        end = QPointF(origin.x()+dx*s*.75, origin.y()+dy*s*.75)
                        painter.drawLine(origin, end)
                        painter.drawText(QPointF(end.x()+6, end.y()-6), name)
                    painter.setPen(QColor('#ffff38'))
                    painter.drawText(10, img.height()-12,
                        f'CPI source axes · export {img.width()} × {img.height()} pixels')
                finally:
                    painter.end()
        if not img.save(str(path), "PNG"):
            raise OSError(f"Could not write {path}")
        return str(path)

    def export_png_dialog(self):
        if self.array is None:
            return
        stem = Path(self.entry.source).stem + (f"_{self.entry.kind.replace(' ', '')}{self.entry.index}"
                                               if self.entry.kind != "image file" else "_measured")
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", stem + ".png", "PNG (*.png)")
        if path:
            try:
                self.export_png(path, self.scale_spin.value())
            except Exception as exc:
                QMessageBox.warning(self, "Export PNG", str(exc))

    def all_rows(self):
        px = self.pixel_um()
        for key, ms in self.measurements.items():
            if ms and key in self.entries_seen:
                yield from measurement_rows(self.entries_seen[key], ms, px)

    def export_csv(self, path) -> int:
        return write_csv(path, self.all_rows())

    def export_csv_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements", "measurements.csv", "CSV (*.csv)")
        if path:
            n = self.export_csv(path)
            self.statusBar().showMessage(f"{n} measurements written to {path}", 6000)


def main(argv=None):
    import sys
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    files = [a for a in (sys.argv[1:] if argv is None else argv[1:])]
    w = MeasureWindow([f for f in files if f.lower().endswith(".mat")])
    for f in files:
        if not f.lower().endswith(".mat"):
            w.open_image_file(f)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
