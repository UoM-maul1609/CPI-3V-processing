"""Optical-microscope / SEM reference-image annotation window for CPI labels.

Coordinates are retained at the original image resolution.  Annotated PNGs are
exports; the source images and annotations are stored separately.
"""
from __future__ import annotations

import math
import uuid
import csv
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QBrush, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform, QFont
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QGraphicsView, QGraphicsScene, QHBoxLayout, QVBoxLayout,
    QGridLayout, QGroupBox, QGraphicsTextItem, QGraphicsRectItem, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QMessageBox, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSplitter, QWidget, QScrollArea, QInputDialog,
)

try:
    from .microscopy import MicroscopyStore, detect_candidates, detect_relative_shadow, enclosing_circle, measure, SharedMicroscopyStore, measure_shared_object, physical_scales, graticule_transform
except ImportError:  # direct ``python python/calibration/gui.py`` launch
    from microscopy import MicroscopyStore, detect_candidates, detect_relative_shadow, enclosing_circle, measure, SharedMicroscopyStore, measure_shared_object, physical_scales, graticule_transform


LINE_BOUNDARY = QColor('#1fe3b5')
LINE_CIRCLE = QColor('#ff9e3e')
LINE_SCALE = QColor('#ffe866')
LINE_CANDIDATE = QColor('#59b7ff')

def QPainterPath_from_rect(rect):
    path=QPainterPath();path.addRect(rect);return path


def read_display_image(path):
    """Load TIFFs (including 16-bit SEM images), grayscale and RGB images."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None or arr.size == 0:
        raise ValueError(f'Cannot read microscope image: {path}')
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, [0.5, 99.5])
        arr = np.uint8(np.clip((arr.astype(np.float32) - lo) * 255.0 / max(float(hi-lo), 1), 0, 255))
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_Grayscale8).copy()
    else:
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()
    return arr, qimg


class MicroscopeCanvas(QGraphicsView):
    edited = Signal()
    scale_drawn = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage = QGraphicsScene(self)
        self.setScene(self.stage)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setBackgroundBrush(QColor('#17202b'))
        self.setMouseTracking(True)
        self.setToolTip('Mouse wheel: zoom · Draw circle: drag · Polygon: click vertices, double-click to finish · '
                        'Edit mode: drag boundary points or circles')
        self.image = None
        self.source_pixmap = None
        self.pixels = None
        self.entry = None
        self.other_annotations = []
        self.label_name = ''
        self.mode = 'edit'
        self.candidates = []
        self.pending = []
        self.drag = None
        self._fit_pending = False
        self.crop_callback = None
        self.crop_aspect = 0.0
        self.crop_rect = None
        self.crop_fraction = None

    def set_image(self, pixels, image, entry, label):
        self.pixels = pixels
        self.image = image
        self.source_pixmap = QPixmap.fromImage(image)
        self.entry = entry
        self.label_name = label
        self.pending = []
        self.drag = None
        self.candidates = []
        self.crop_rect = None
        self.paint_scene()
        self.fit_image()

    def fit_image(self):
        if self.image is not None:
            self.fitInView(self.stage.sceneRect(), Qt.KeepAspectRatio)

    def set_tool(self, mode):
        self.mode = mode
        self.drag = None
        if mode != 'polygon':
            self.pending = []
        self.setCursor(Qt.CrossCursor if mode in ('circle', 'polygon', 'scale', 'scale_y') else Qt.ArrowCursor)
        self.paint_scene()

    def _path(self, points, close=True):
        path = QPainterPath()
        if not points:
            return path
        path.moveTo(float(points[0][0]), float(points[0][1]))
        for x, y in points[1:]:
            path.lineTo(float(x), float(y))
        if close:
            path.closeSubpath()
        return path

    @staticmethod
    def _pen(colour, width=2, style=Qt.SolidLine):
        pen = QPen(colour, width)
        pen.setCosmetic(True)
        pen.setStyle(style)
        return pen

    def _visible_path(self, points, colour, width=2.5, style=Qt.SolidLine, close=True):
        """Bright annotation on a contrasting black halo (also in PNG export)."""
        path = self._path(points, close=close)
        self.stage.addPath(path, self._pen(QColor('#000000'), width+3, style))
        return self.stage.addPath(path, self._pen(colour, width, style))

    def _visible_circle(self, x, y, r, colour, width=2.5, style=Qt.SolidLine):
        self.stage.addEllipse(x-r, y-r, 2*r, 2*r, self._pen(QColor('#000000'), width+3, style))
        self.stage.addEllipse(x-r, y-r, 2*r, 2*r, self._pen(colour, width, style))

    def paint_scene(self, annotations=True):
        self.stage.clear()
        if self.image is None:
            self.stage.addText('Import an optical microscope or SEM image to begin.')
            return
        w, h = self.image.width(), self.image.height()
        self.stage.setSceneRect(QRectF(0, 0, w, h))
        self.stage.addPixmap(self.source_pixmap)
        if self.crop_rect is not None:
            self.stage.addPolygon(self.crop_rect, self._pen(QColor('#ffdb36'), 3, Qt.DashLine))
        if not annotations or self.entry is None:
            return
        ann = self.entry
        for points in self.candidates:
            self._visible_path(points, LINE_CANDIDATE, 1.5, Qt.DashLine)
        for other in self.other_annotations:
            if other is ann:
                continue
            if other.get('mode') == 'polygon' and other.get('boundary'):
                self._visible_path(other['boundary'], QColor('#61adf7'), 2)
            elif other.get('mode') == 'circle' and other.get('boundary_circle'):
                ox, oy, orad = other['boundary_circle']
                self._visible_circle(ox, oy, orad, QColor('#61adf7'), 2)
            # All objects can be seen simultaneously. Active object is green.
            pts = other.get('boundary') or []
            if pts:
                xx, yy = np.mean(np.asarray(pts), axis=0)
                self._caption(other.get('name') or other.get('label') or 'object', xx, yy, QColor('#c5efff'), 16)
        if ann.get('mode') == 'polygon' and ann.get('boundary'):
            self._visible_path(ann['boundary'], LINE_BOUNDARY)
            if self.mode == 'edit':
                for x, y in ann['boundary']:
                    self.stage.addEllipse(x-5, y-5, 10, 10, self._pen(LINE_BOUNDARY, 1), QBrush(LINE_BOUNDARY))
        elif ann.get('mode') == 'circle' and ann.get('boundary_circle'):
            x,y,r = ann['boundary_circle']
            self._visible_circle(x, y, r, LINE_BOUNDARY)
            self.stage.addEllipse(x+r-4, y-4, 8, 8, self._pen(LINE_BOUNDARY, 1), QBrush(LINE_BOUNDARY))
        if self.pending:
            self._visible_path(self.pending, LINE_BOUNDARY, 2, Qt.DashLine, close=False)
            for x, y in self.pending:
                self.stage.addEllipse(x-2, y-2, 4, 4, self._pen(LINE_BOUNDARY, 1), QBrush(LINE_BOUNDARY))
        circle = ann.get('enclosing_circle')
        if circle:
            x,y,r = circle
            self._visible_circle(x, y, r, LINE_CIRCLE, 2.5, Qt.DashLine)
        scale = ann.get('scale_line')
        if scale:
            a,b = scale
            self.stage.addLine(a[0],a[1],b[0],b[1],self._pen(QColor('#000000'),6))
            self.stage.addLine(a[0],a[1],b[0],b[1],self._pen(LINE_SCALE,3))
            for x,y in (a,b):
                self.stage.addEllipse(x-3,y-3,6,6,self._pen(LINE_SCALE,1),QBrush(LINE_SCALE))
            if ann.get('scale_um'):
                self._caption(f"{ann['scale_um']:g} µm", b[0]+5, b[1]+4, LINE_SCALE, size=max(10,min(25,w//75)))
        scale_y = ann.get('scale_line_y')
        if scale_y:
            a,b = scale_y
            self.stage.addLine(a[0],a[1],b[0],b[1],self._pen(QColor('#000000'),6))
            self.stage.addLine(a[0],a[1],b[0],b[1],self._pen(QColor('#ff87f8'),3))
            for x,y in (a,b):
                self.stage.addEllipse(x-4,y-4,8,8,self._pen(QColor('#ff87f8'),1),QBrush(QColor('#ff87f8')))
            if ann.get('scale_um_y'):
                self._caption(f"Y: {ann['scale_um_y']:g} µm",b[0]+5,b[1]+4,QColor('#ff87f8'),size=max(10,min(25,w//75)))
        base_font = max(20, min(62, w//36))
        # Text is part of both the workspace and exported full-resolution image.
        self._banner(self.label_name, y=4, font_size=base_font, width=w)
        try:
            m = (measure_shared_object(self.shared_image, ann)
                 if getattr(self, 'shared_mode', False) and getattr(self, 'shared_image', None) is not None
                 else measure(ann))
        except ValueError:
            # A partly drawn or nearly parallel two-axis graticule should not
            # prevent interactive editing; display pixel measures meanwhile.
            m = measure({k: v for k, v in ann.items()
                         if k not in ('scale_line', 'scale_um')})
        if m['max_dimension_px']:
            calibrated = (m.get('scale_x_um_per_px') is not None if getattr(self,'shared_mode',False)
                          else m.get('um_per_px') is not None)
            if calibrated:
                info = (f"Max {m['max_dimension_um']:.2f} µm     Area {m['area_um2']:.2f} µm²"
                        + (f"     Enclosing circle {m['enclosing_circle_area_um2']:.2f} µm²"
                           if m['enclosing_circle_area_um2'] else '') )
            else:
                info = (f"Max {m['max_dimension_px']:.1f} px     Area {m['area_px2']:.1f} px²"
                        + ('    Scale not set' ))
            if m.get('equivalent_area_diameter_px') is not None:
                deq = m.get('equivalent_area_diameter_um') if calibrated else m['equivalent_area_diameter_px']
                if deq is not None:
                    info += f"     Deq {deq:.2f} {'µm' if calibrated else 'px'}"
            if m.get('cpi_minor_axis_width_px') is not None:
                value = m.get('cpi_minor_axis_width_um') if calibrated else m['cpi_minor_axis_width_px']
                info += f"     Ellipse W {value:.2f} {'µm' if calibrated else 'px'}"
            if m.get('feret_min_width_px') is not None:
                value = m.get('feret_min_width_um') if calibrated else m['feret_min_width_px']
                info += f"     Feret W {value:.2f} {'µm' if calibrated else 'px'}"
            if m['area_over_circle'] is not None:
                info += f"     Area ratio {m['area_over_circle']:.3f}"
            self._banner(info, y=8+base_font*1.65, font_size=max(16,int(base_font*.82)), width=w)

    def _banner(self, text, y, font_size, width):
        if not text:
            return
        item = self.stage.addText(str(text))
        font = item.font()
        font.setPixelSize(font_size)
        font.setBold(True)
        item.setFont(font)
        item.setDefaultTextColor(Qt.white)
        # Constrain long labels/measurements to available image width.
        if item.boundingRect().width() > width - 12:
            item.setScale(max(.2,(width-12)/max(item.boundingRect().width(),1)))
        item.setPos(9, y)
        rect = item.mapRectToScene(item.boundingRect())
        background = self.stage.addRect(QRectF(4,y, min(width-4,rect.width()+12),rect.height()+2),
                                        QPen(Qt.NoPen), QBrush(QColor(0,0,0,165)))
        background.setZValue(9)
        item.setZValue(10)

    def _caption(self, text,x,y,color,size=14):
        label = self.stage.addText(text)
        font = label.font(); font.setPixelSize(size); font.setBold(True); label.setFont(font)
        label.setDefaultTextColor(color)
        label.setPos(x,y)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier and self.parent() is not None:
            dialog = self.window()
            if hasattr(dialog,"export_rotate"):
                dialog.export_rotate.setValue(dialog.export_rotate.value() + event.angleDelta().y()/120)
                event.accept(); return
        super().wheelEvent(event)

    def _point(self, event):
        p = self.mapToScene(event.position().toPoint())
        if self.image is None or not (0 <= p.x() < self.image.width() and 0 <= p.y() < self.image.height()):
            return None
        return [float(p.x()), float(p.y())]

    def _choose_candidate(self, pt):
        inside = []
        near = []
        for pts in self.candidates:
            contour = np.asarray(pts, dtype=np.float32)
            dist = cv2.pointPolygonTest(contour, tuple(pt), True)
            if dist >= 0:
                inside.append((abs(cv2.contourArea(contour)), pts))
            elif dist >= -10:
                near.append((-dist, pts))
        if inside:
            return min(inside, key=lambda item:item[0])[1]
        if near:
            return min(near, key=lambda item:item[0])[1]
        return None

    def _hit(self, pt):
        ann = self.entry
        # Pixel-space hit tolerance increases when the image is zoomed out.
        tolerance = max(7., 13/max(abs(self.transform().m11()),.01))
        if ann.get('mode') == 'polygon':
            for i, p in enumerate(ann.get('boundary') or []):
                if math.dist(pt,p) <= tolerance:
                    return ('vertex', i)
        if ann.get('mode') == 'circle' and ann.get('boundary_circle'):
            x,y,r = ann['boundary_circle']; d = math.dist(pt,[x,y])
            if abs(d-r) <= tolerance:
                return ('resize', None)
            if d < r:
                return ('move', None)
        for key,hit_name in (('scale_line','scale-point'), ('scale_line_y','scale-y-point')):
            if ann.get(key):
                for i,p in enumerate(ann[key]):
                    if math.dist(pt,p) <= tolerance:
                        return (hit_name,i)
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.entry is None:
            return super().mousePressEvent(event)
        pt = self._point(event)
        if pt is None:
            return
        ann = self.entry
        if self.mode == 'crop':
            self.drag = ('crop', pt, None)
            self.crop_rect = None
            self.crop_fraction = None
            self.paint_scene()
        elif self.mode == 'select':
            chosen = self._choose_candidate(pt)
            if chosen:
                ann.update(mode='polygon', boundary=chosen, boundary_circle=None, enclosing_circle=None)
                self.candidates = []
                self.edited.emit()
                self.paint_scene()
        elif self.mode == 'polygon':
            if not self.pending or math.dist(pt,self.pending[-1]) > 2:
                self.pending.append(pt)
            self.paint_scene()
        elif self.mode == 'circle':
            # A new circle is drawn centre-first; adjust it in Edit mode.
            self.drag = ('new-circle', pt, None)
            ann.update(mode='circle', boundary_circle=[*pt, 0.1], boundary=[], enclosing_circle=None)
            self.paint_scene()
        elif self.mode in ('scale', 'scale_y'):
            self.drag = (self.mode, pt, None)
            ann['scale_line_y' if self.mode == 'scale_y' else 'scale_line'] = [pt,pt]
            self.paint_scene()
        elif self.mode == 'edit':
            hit = self._hit(pt)
            if hit:
                self.drag = (hit[0], pt, hit[1],
                             list(ann.get('boundary_circle') or []))
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.drag or self.entry is None:
            return super().mouseMoveEvent(event)
        pt = self._point(event)
        if pt is None:
            return
        kind, start, index, *old = self.drag
        ann = self.entry
        if kind == 'crop':
            from .export_transform import rotated_crop_rect
            angle = self.window().export_rotate.value()
            self.crop_rect, self.crop_fraction, _ = rotated_crop_rect(
                start, pt, self.image.width(), self.image.height(), angle, self.crop_aspect)
        elif kind == 'new-circle':
            ann['boundary_circle'] = [*start, max(.1,math.dist(start,pt))]
        elif kind in ('scale','scale_y'):
            ann['scale_line_y' if kind == 'scale_y' else 'scale_line'][1] = pt
        elif kind == 'vertex':
            ann['boundary'][index] = pt
            ann['enclosing_circle'] = None
        elif kind in ('scale-point','scale-y-point'):
            ann['scale_line_y' if kind == 'scale-y-point' else 'scale_line'][index] = pt
        elif kind == 'resize':
            x,y,r = old[0]
            ann['boundary_circle'] = [x,y,max(.1,math.dist(pt,[x,y]))]
            ann['enclosing_circle'] = None
        elif kind == 'move':
            x,y,r = old[0]
            ann['boundary_circle'] = [x+pt[0]-start[0],y+pt[1]-start[1],r]
            ann['enclosing_circle'] = None
        self.paint_scene()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.drag and self.entry is not None:
            kind = self.drag[0]
            self.drag = None
            if kind == 'crop':
                if self.crop_callback and self.crop_rect is not None and self.crop_fraction is not None:
                    self.crop_callback(self.crop_fraction)
                self.paint_scene()
                event.accept()
                return
            if kind in ('new-circle','resize','move') and self.entry.get('boundary_circle'):
                x,y,r = self.entry['boundary_circle']
                if r >= 1:
                    self.entry['enclosing_circle'] = [x,y,r]
            self.edited.emit()
            if kind in ('scale','scale_y'):
                self.scale_drawn.emit()
            self.paint_scene()
            event.accept()
            return
        return super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode == 'polygon' and event.button() == Qt.LeftButton:
            self.finish_polygon()
            event.accept()
            return
        return super().mouseDoubleClickEvent(event)

    def finish_polygon(self):
        if self.entry is None or len(self.pending) < 3:
            return False
        self.entry.update(mode='polygon', boundary=list(self.pending), boundary_circle=None, enclosing_circle=None)
        self.pending = []
        self.candidates = []
        self.edited.emit()
        self.paint_scene()
        return True

    def undo_point(self):
        if self.pending:
            self.pending.pop()
            self.paint_scene()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            dialog = self.window()
            if hasattr(dialog, 'export_rotate'):
                dialog.export_rotate.setValue(dialog.export_rotate.value() + event.angleDelta().y() / 120)
                event.accept()
                return
        if self.image is not None:
            factor = 1.18 if event.angleDelta().y() > 0 else 1/1.18
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def export_png(self, path, transform=None):
        if self.image is None:
            raise ValueError('No image loaded')
        # Suggestions and unfinished vertices are editing aids, not annotations.
        shown_candidates, shown_pending = self.candidates, self.pending
        shown_crop = self.crop_rect
        self.crop_rect = None
        self.candidates, self.pending = [], []
        self.paint_scene()
        try:
            image = QImage(self.image.width(), self.image.height(), QImage.Format_ARGB32)
            if image.isNull():
                raise MemoryError('Not enough memory to export at the original resolution')
            image.fill(Qt.white)
            # Keep geometric annotations in source coordinates, but redraw text
            # after the presentation transform so it remains upright.
            text_items = [item for item in self.stage.items()
                          if isinstance(item, QGraphicsTextItem) or
                          (isinstance(item, QGraphicsRectItem) and item.brush().color().alpha() == 165)]
            for item in text_items:
                item.setVisible(False)
            painter = QPainter(image)
            try:
                self.stage.render(painter, QRectF(0,0,image.width(),image.height()),
                                  QRectF(0,0,image.width(),image.height()))
            finally:
                painter.end()
                for item in text_items:
                    item.setVisible(True)
            if transform is not None:
                image = transform(image)
            # Annotation text is placed in final paper-reading orientation.
            p = QPainter(image)
            try:
                p.setRenderHint(QPainter.Antialiasing, True)
                width, height = image.width(), image.height()
                size = max(13, min(45, width // 36))
                font = QFont(); font.setPixelSize(size); font.setBold(True)
                p.setFont(font)
                def banner(text, y, colour=QColor('white')):
                    if not text: return
                    metrics = p.fontMetrics()
                    # Fit long measurements in the export rather than clipping.
                    local = QFont(font)
                    while metrics.horizontalAdvance(text) > width-18 and local.pixelSize()>10:
                        local.setPixelSize(local.pixelSize()-1)
                        p.setFont(local); metrics=p.fontMetrics()
                    p.fillRect(QRectF(3,y, min(width-3,metrics.horizontalAdvance(text)+15),metrics.height()+5), QColor(0,0,0,175))
                    p.setPen(colour); p.drawText(9, int(y+metrics.ascent()+2), text)
                    p.setFont(font)
                    return metrics.height()+8
                y=4
                y += banner(self.label_name, y) or 0
                m=measure(self.entry) if self.entry else None
                if m and m.get('max_dimension_px'):
                    if m.get('um_per_px'):
                        info=f"Max {m['max_dimension_um']:.2f} µm  Area {m['area_um2']:.2f} µm²"
                    else:
                        info=f"Max {m['max_dimension_px']:.1f} px  Area {m['area_px2']:.1f} px²"
                    if m.get('equivalent_area_diameter_px') is not None:
                        deq = m.get('equivalent_area_diameter_um') if m.get('um_per_px') else m['equivalent_area_diameter_px']
                        if deq is not None:
                            info += f"  Deq {deq:.2f} {'µm' if m.get('um_per_px') else 'px'}"
                    if m.get('cpi_minor_axis_width_px') is not None:
                        w = m['cpi_minor_axis_width_um'] if m.get('um_per_px') else m['cpi_minor_axis_width_px']
                        info += f"  Ellipse W {w:.2f} {'µm' if m.get('um_per_px') else 'px'}"
                    if m.get('feret_min_width_px') is not None:
                        w = m['feret_min_width_um'] if m.get('um_per_px') else m['feret_min_width_px']
                        info += f"  Feret W {w:.2f} {'µm' if m.get('um_per_px') else 'px'}"
                    if m.get('area_over_circle') is not None:
                        info+=f"  Area ratio {m['area_over_circle']:.3f}"
                    y += banner(info,y) or 0
                if self.entry and self.entry.get('scale_um'):
                    label=f"Scale reference {self.entry['scale_um']:g} µm"
                    banner(label, max(y, height-size-12), LINE_SCALE)
            finally:
                p.end()
            if not image.save(str(path), 'PNG'):
                raise OSError(f'Could not save PNG: {path}')
        finally:
            self.candidates, self.pending = shown_candidates, shown_pending
            self.crop_rect = shown_crop
            self.paint_scene()


class MicroscopyDialog(QDialog):
    """One reusable microscopy gallery/editor per shared particle label."""
    def __init__(self, library, label, parent=None):
        super().__init__(parent)
        self.library = library
        self.label = label
        self.store = MicroscopyStore(library.path)
        self.active = None
        self._loading = False
        self.setWindowTitle(f'Microscopy reference images — {label.name}')
        self.resize(1380,850)
        self.setMinimumSize(960,630)
        root = QVBoxLayout(self)
        heading = QLabel(f'{label.name}  ·  {label.kind}  ·  {label.size_text()}')
        font = heading.font(); font.setPointSize(14); font.setBold(True); heading.setFont(font)
        root.addWidget(heading)
        help = QLabel('Add multiple microscope/SEM images. Select a detected object or draw its outline; '
                      'draw a scale line and enter its real length. Images and annotations are saved separately '
                      'from MAT files. Scroll to zoom.')
        help.setWordWrap(True)
        root.addWidget(help)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter,1)

        side = QWidget(); left = QVBoxLayout(side)
        left.addWidget(QLabel('Images for this label'))
        self.images = QListWidget(); self.images.setMinimumWidth(185)
        self.images.currentItemChanged.connect(self.load_selected)
        left.addWidget(self.images,1)
        import_button = QPushButton('＋ Import images…'); import_button.clicked.connect(self.import_images)
        left.addWidget(import_button)
        delete_button = QPushButton('Remove from label…'); delete_button.clicked.connect(self.remove_image)
        left.addWidget(delete_button)
        purge_button = QPushButton('Delete image and stored copy…')
        purge_button.clicked.connect(self.delete_image_permanently)
        left.addWidget(purge_button)
        left.addWidget(QLabel('Original images are copied unchanged and kept beside the shared label library.'))
        splitter.addWidget(side)

        self.canvas = MicroscopeCanvas()
        self.canvas.edited.connect(self.persist)
        self.canvas.scale_drawn.connect(self.set_scale_after_draw)
        self.canvas.crop_callback = self._crop_from_rectangle
        splitter.addWidget(self.canvas)

        controls_scroll = QScrollArea(); controls_scroll.setWidgetResizable(True)
        controls = QWidget(); col = QVBoxLayout(controls)
        col.addWidget(QLabel('Annotation tools'))
        self.tools = QComboBox()
        for key, text in [('edit','Edit / move or resize'),('select','Choose detected object (click)'),
                          ('circle','Draw circular boundary (drag)'),('polygon','Draw straight-line polygon (click)'),
                          ('scale','Draw scale reference (drag)'),('crop','Drag crop rectangle (export only)')]:
            self.tools.addItem(text, key)
        self.tools.currentIndexChanged.connect(lambda _:self.canvas.set_tool(self.tools.currentData()))
        col.addWidget(self.tools)
        self.detection_mode = QComboBox()
        self.detection_mode.addItem('Relative darkness below outer background', 'relative')
        self.detection_mode.addItem('Original Canny / Otsu', 'original')
        col.addWidget(self.detection_mode)
        self.detection_fraction = QDoubleSpinBox()
        self.detection_fraction.setRange(1, 95); self.detection_fraction.setValue(25)
        self.detection_fraction.setSuffix('% darker'); self.detection_fraction.setSingleStep(5)
        self.detection_fraction.setToolTip('25% means intensities below 0.75 times the estimated clear-background intensity.')
        col.addWidget(self.detection_fraction)
        self.find_button = QPushButton('Find object boundaries')
        self.find_button.clicked.connect(self.find_objects)
        col.addWidget(self.find_button)
        col.addWidget(QLabel('Click the desired blue outline to select it. Detection is only a suggestion.'))
        finish = QPushButton('Finish / join polygon'); finish.clicked.connect(self.finish_polygon)
        col.addWidget(finish)
        undo = QPushButton('Undo last polygon point'); undo.clicked.connect(self.canvas.undo_point)
        col.addWidget(undo)
        fit = QPushButton('Fit enclosing circle around boundary'); fit.clicked.connect(self.fit_circle)
        col.addWidget(fit)
        reset = QPushButton('Clear object boundary'); reset.clicked.connect(self.clear_boundary)
        col.addWidget(reset)
        zoom = QPushButton('Fit image to window'); zoom.clicked.connect(self.canvas.fit_image)
        col.addWidget(zoom)
        scale_group = QGroupBox('Image scale')
        scale_layout = QVBoxLayout(scale_group)
        scale_layout.addWidget(QLabel('Draw a line across a known scale bar, then enter its length.'))
        row = QHBoxLayout()
        self.scale_value = QDoubleSpinBox()
        self.scale_value.setRange(0.000001,1e7); self.scale_value.setDecimals(6)
        self.scale_value.setValue(10.0)
        self.scale_value.setSuffix(' µm')
        self.scale_value.valueChanged.connect(self.scale_changed)
        row.addWidget(QLabel('Line represents'))
        row.addWidget(self.scale_value)
        scale_layout.addLayout(row)
        self.scale_status = QLabel('No scale reference yet')
        self.scale_status.setWordWrap(True)
        scale_layout.addWidget(self.scale_status)
        col.addWidget(scale_group)

        metric_group = QGroupBox('Measurements')
        metrics = QVBoxLayout(metric_group)
        self.metric_max = QLabel('Maximum dimension: —')
        self.metric_width = QLabel('CPI stored wid (ellipse minor axis): —')
        self.metric_feret_width = QLabel('Minimum Feret width: —')
        self.metric_area = QLabel('Object area: —')
        self.metric_deq = QLabel('Equivalent-area diameter: —')
        self.metric_circle = QLabel('Enclosing circle area: —')
        self.metric_ratio = QLabel('Area / circle area: —')
        for label in (self.metric_max,self.metric_width,self.metric_feret_width,self.metric_area,self.metric_deq,self.metric_circle,self.metric_ratio):
            label.setWordWrap(True); label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            metrics.addWidget(label)
        col.addWidget(metric_group)
        crop_group = QGroupBox('PNG view (original image unchanged)')
        crop_form = QGridLayout(crop_group)
        self.export_rotate = QDoubleSpinBox()
        self.export_rotate.setRange(-360, 360); self.export_rotate.setDecimals(1)
        self.export_rotate.setSingleStep(1); self.export_rotate.setSuffix('° clockwise')
        self.export_rotate.valueChanged.connect(self._preview_rotation)
        self.canvas.setToolTip(self.canvas.toolTip() + ' · Shift + mouse wheel: rotate image')
        crop_form.addWidget(QLabel('Rotate export'), 0, 0)
        crop_form.addWidget(self.export_rotate, 0, 1)
        self.crop_spins = []
        for i, (name, value) in enumerate((('Left %', 0), ('Top %', 0), ('Right %', 100), ('Bottom %', 100))):
            spin = QDoubleSpinBox(); spin.setRange(0, 100); spin.setDecimals(3)
            spin.setValue(value); spin.setSuffix('%'); self.crop_spins.append(spin)
            crop_form.addWidget(QLabel(name), i // 2 + 1, (i % 2) * 2)
            crop_form.addWidget(spin, i // 2 + 1, (i % 2) * 2 + 1)
        self.crop_aspect = QComboBox()
        for name,val in [('Free aspect',0),('1:1',1),('4:3',4/3),('3:2',1.5),('16:9',16/9),('9:16',9/16),('Custom',-1)]:
            self.crop_aspect.addItem(name,val)
        self.crop_custom_width = QDoubleSpinBox(); self.crop_custom_width.setRange(.01,10000); self.crop_custom_width.setValue(4)
        self.crop_custom_height = QDoubleSpinBox(); self.crop_custom_height.setRange(.01,10000); self.crop_custom_height.setValue(3)
        def update_aspect(*_):
            ratio=self.crop_aspect.currentData()
            self.canvas.crop_aspect = (self.crop_custom_width.value()/self.crop_custom_height.value()) if ratio==-1 else ratio
        self.crop_aspect.currentIndexChanged.connect(update_aspect)
        self.crop_custom_width.valueChanged.connect(update_aspect)
        self.crop_custom_height.valueChanged.connect(update_aspect)
        crop_form.addWidget(QLabel('Custom W : H'),4,0)
        custom_ratio=QHBoxLayout(); custom_ratio.addWidget(self.crop_custom_width); custom_ratio.addWidget(self.crop_custom_height)
        crop_form.addLayout(custom_ratio,4,1)
        crop_form.addWidget(QLabel('Final crop W:H'), 3, 0)
        crop_form.addWidget(self.crop_aspect,3,1)
        col.addWidget(QLabel('Select “Drag crop rectangle” above and drag on the image. Shift+wheel rotates the preview.'))
        col.addWidget(crop_group)
        export = QPushButton('Save annotated image as PNG…')
        export.clicked.connect(self.save_png)
        col.addWidget(export)
        col.addWidget(QLabel('Boundary = green · enclosing circle = orange dashed · scale = yellow'))
        col.addStretch(1)
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumWidth(295)
        splitter.addWidget(controls_scroll)
        splitter.setSizes([215,840,325])
        self.status = QLabel('Choose or import an image. Annotations save after each completed edit.')
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.refresh_list()

    def refresh_list(self, chosen=None):
        previous = chosen or (self.active['id'] if self.active else None)
        self._loading = True
        self.images.clear()
        for entry in self.store.list_for(self.label.name, self.label.aliases):
            annotation = '✓ outlined' if entry.get('mode') else 'not outlined'
            scale = ' · scaled' if entry.get('scale_line') and entry.get('scale_um') else ''
            item = QListWidgetItem(f"{entry['original_filename']}\n   {annotation}{scale}")
            item.setData(Qt.UserRole, entry['id'])
            self.images.addItem(item)
            if entry['id'] == previous:
                self.images.setCurrentItem(item)
        self._loading = False
        if self.images.currentItem() is None and self.images.count():
            self.images.setCurrentRow(0)
        elif self.images.currentItem() is not None:
            # Selection signals were deliberately blocked during the rebuild.
            self.load_selected(self.images.currentItem())
        if not self.images.count():
            self.active = None
            self.canvas.entry = None
            self.canvas.image = None
            self.canvas.source_pixmap = None
            self.canvas.paint_scene()
            self.show_metrics()

    def load_selected(self, item=None, _previous=None):
        if self._loading or item is None:
            return
        entry = next((e for e in self.store.entries if e['id'] == item.data(Qt.UserRole)),None)
        if entry is None:
            return
        try:
            arr, img = read_display_image(self.store.image_path(entry))
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Image unavailable', str(exc))
            return
        self.active = entry
        self.canvas.set_image(arr,img,entry,self.label.name)
        self._preview_rotation(self.export_rotate.value())
        self.scale_value.blockSignals(True)
        self.scale_value.setValue(float(entry.get('scale_um') or 10))
        self.scale_value.blockSignals(False)
        self.tools.setCurrentIndex(0)
        self.show_metrics()
        self.status.setText(f"{entry['original_filename']} · {img.width()} × {img.height()} px · "
                            'All annotations stored separately from the original image.')

    def import_images(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Choose microscope or SEM images', '',
            'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)')
        if not filenames:
            return
        added = []
        failed = []
        for filename in filenames:
            try:
                read_display_image(filename)
                entry = self.store.import_image(filename, self.label.name)
                added.append(entry['id'])
            except (OSError, ValueError) as exc:
                failed.append(f'{Path(filename).name}: {exc}')
        if added:
            self.refresh_list(chosen=added[-1])
        if failed:
            QMessageBox.warning(self, 'Some images could not be imported', '\n'.join(failed[:8]))

    def remove_image(self):
        if not self.active:
            return
        if QMessageBox.question(self, 'Remove image from this label?',
            'Remove this reference image and its annotations from the label?\n\n'
            'The copied image file is retained for recovery.') != QMessageBox.Yes:
            return
        self.store.delete(self.active['id'])
        self.active = None
        self.refresh_list()

    def delete_image_permanently(self):
        if not self.active:
            return
        filename = self.active.get('original_filename', self.active.get('file', 'image'))
        if QMessageBox.warning(
            self, 'Permanently delete imported image?',
            f'Delete {filename} and all its saved annotations from this label?\n\n'
            'The imported copy in the microscopy library will be deleted. '
            'The source file you originally imported will NOT be deleted.\n\n'
            'This cannot be undone.',
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        try:
            self.store.delete(self.active['id'], remove_file=True)
        except Exception as exc:
            QMessageBox.warning(self, 'Deletion failed', str(exc))
        self.active = None
        self.refresh_list()

    def persist(self):
        if not self.active:
            return
        try:
            self.store.update(self.active)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Cannot save annotation', str(exc))
        self.show_metrics()

    def show_metrics(self):
        a = measure(self.active) if self.active else {}
        if not a.get('max_dimension_px'):
            for widget in (self.metric_max,self.metric_width,self.metric_feret_width,self.metric_area,self.metric_deq,self.metric_circle,self.metric_ratio):
                widget.setText('— Draw or select an object boundary —')
            self.scale_status.setText('No scale reference yet')
            return
        if a.get('um_per_px'):
            self.metric_max.setText(f"Maximum dimension: {a['max_dimension_um']:.3f} µm")
            self.metric_area.setText(f"Object area: {a['area_um2']:.3f} µm²")
            self.metric_deq.setText(f"Equivalent-area diameter: {a['equivalent_area_diameter_um']:.3f} µm")
            self.metric_circle.setText(f"Enclosing circle area: {a['enclosing_circle_area_um2']:.3f} µm²"
                if a.get('enclosing_circle_area_um2') else 'Enclosing circle area: click Fit circle')
            self.scale_status.setText(f"Scale line: {float(self.active.get('scale_um')):g} µm exactly · {a['um_per_px']:.8g} µm/pixel")
        else:
            self.metric_max.setText(f"Maximum dimension: {a['max_dimension_px']:.2f} px (uncalibrated)")
            self.metric_area.setText(f"Object area: {a['area_px2']:.2f} px²")
            self.metric_deq.setText(f"Equivalent-area diameter: {a['equivalent_area_diameter_px']:.2f} px (uncalibrated)")
            self.metric_circle.setText(f"Enclosing circle area: {a['enclosing_circle_area_px2']:.2f} px²"
                if a.get('enclosing_circle_area_px2') else 'Enclosing circle area: click Fit circle')
            self.scale_status.setText('Draw scale reference to obtain micrometres')
        self.metric_width.setText(('CPI stored wid (ellipse minor axis): ' + (f"{a['cpi_minor_axis_width_um']:.3f} µm" if a.get('cpi_minor_axis_width_um') else f"{a['cpi_minor_axis_width_px']:.2f} px")) if a.get('cpi_minor_axis_width_px') else 'CPI stored wid: —')
        self.metric_feret_width.setText(('Minimum Feret width: ' + (f"{a['feret_min_width_um']:.3f} µm" if a.get('feret_min_width_um') else f"{a['feret_min_width_px']:.2f} px")) if a.get('feret_min_width_px') is not None else 'Minimum Feret width: —')
        self.metric_ratio.setText(f"Area / circle area: {a['area_over_circle']:.4f}"
                                  if a.get('area_over_circle') is not None else 'Area / circle area: —')

    def find_objects(self):
        if not self.active or self.canvas.pixels is None:
            return
        self.canvas.candidates = (detect_relative_shadow(self.canvas.pixels, self.detection_fraction.value() / 100)
                                  if self.detection_mode.currentData() == 'relative'
                                  else detect_candidates(self.canvas.pixels))
        self.tools.setCurrentIndex(self.tools.findData('select'))
        self.canvas.paint_scene()
        self.status.setText(f'{len(self.canvas.candidates)} candidate outlines found. '
                            'Click the outline enclosing your object; draw manually if none is suitable.')

    def finish_polygon(self):
        if not self.canvas.finish_polygon():
            self.status.setText('Polygon needs at least three vertices. Click around the object, then Finish.')
        else:
            self.tools.setCurrentIndex(0)

    def fit_circle(self):
        if not self.active:
            return
        try:
            self.active['enclosing_circle'] = enclosing_circle(self.active)
        except ValueError as exc:
            QMessageBox.information(self,'Enclosing circle',str(exc))
            return
        self.persist()
        self.canvas.paint_scene()

    def clear_boundary(self):
        if not self.active:
            return
        self.active.update(mode=None, boundary=[], boundary_circle=None, enclosing_circle=None)
        self.canvas.pending=[];self.canvas.candidates=[]
        self.persist(); self.canvas.paint_scene()

    def set_scale_after_draw(self):
        if not self.active or not self.active.get('scale_line'):
            return
        line=self.active['scale_line']
        if math.dist(*line) < 3:
            self.active['scale_line']=None
            self.persist(); self.canvas.paint_scene()
            return
        amount, confirmed = QInputDialog.getDouble(
            self, 'Scale-bar calibration', 'Real length of the drawn line (µm):',
            float(self.scale_value.value()), 0.000001, 1e7, 6)
        if confirmed:
            self.active['scale_um'] = amount
            self.scale_value.blockSignals(True)
            self.scale_value.setValue(amount)
            self.scale_value.blockSignals(False)
            self.status.setText(f'Scale saved: {amount:g} µm along the reference line.')
        else:
            # Never silently treat the pre-filled spin-box value as a known scale.
            self.active['scale_um'] = None
            self.status.setText('Reference drawn, but no physical scale set; edit the µm value to calibrate.')
        self.persist(); self.canvas.paint_scene()
        self.tools.setCurrentIndex(0)

    def scale_changed(self, value):
        if self.active and self.active.get('scale_line'):
            self.active['scale_um']=value
            self.persist(); self.canvas.paint_scene()

    def _crop_from_rectangle(self, fractions):
        if self.canvas.image is None: return
        values=fractions
        for spin,val in zip(self.crop_spins,values): spin.setValue(round(max(0,min(100,100*val)),3))
        self.status.setText("Crop rectangle selected; only the exported PNG is cropped.")

    def _preview_rotation(self, angle):
        self.canvas.resetTransform()
        self.canvas.fit_image()
        self.canvas.rotate(float(angle))
        self.canvas.centerOn(self.canvas.stage.sceneRect().center())

    def save_png(self):
        if not self.active:
            return
        stem = Path(self.active['original_filename']).stem
        path, _ = QFileDialog.getSaveFileName(self,'Save annotated microscopy PNG',
            f'{stem}_annotated.png', 'PNG image (*.png)')
        if not path:
            return
        try:
            # Rotate the full annotated image first, then crop in paper orientation.
            try:
                from .export_transform import crop_rotate_image
            except ImportError:
                from export_transform import crop_rotate_image
            self.canvas.export_png(path, transform=lambda image: crop_rotate_image(
                image, [s.value() / 100 for s in self.crop_spins], self.export_rotate.value()))
        except (ValueError,OSError,MemoryError) as exc:
            QMessageBox.warning(self,'Export failed',str(exc))
        else:
            self.status.setText(f'Annotated PNG saved: {path}')


class SharedMicroscopyDialog(QDialog):
    """Graticule images with any number of outlined, independently named particles.

    The photo and calibration are stored once; an object's optional library label
    is a link, not a request to overwrite reference dimensions in labels.json.
    """
    def __init__(self, library, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QLineEdit
        self.library = library
        self.store = SharedMicroscopyStore(library.path)
        self.active_image = None
        self.active_object = None
        self._loading = False
        self.setWindowTitle('Shared microscopy images — multiple ice analogues / graticule')
        self.resize(1540, 900)
        root = QVBoxLayout(self)
        note = QLabel('One image and one graticule calibration can contain many outlined particles. '
                      'Assign an existing label or describe the object locally. The measurements are saved '
                      'with this image; reference-label sizes are never replaced automatically.')
        note.setWordWrap(True)
        root.addWidget(note)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        left_widget = QWidget(); left = QVBoxLayout(left_widget)
        left.addWidget(QLabel('Shared images'))
        self.images = QListWidget()
        self.images.currentItemChanged.connect(self.load_image)
        left.addWidget(self.images, 1)
        import_btn=QPushButton('＋ Import shared image…'); import_btn.clicked.connect(self.import_images)
        left.addWidget(import_btn)
        delete_btn=QPushButton('Remove selected image…'); delete_btn.clicked.connect(self.remove_image)
        left.addWidget(delete_btn)
        purge_btn = QPushButton('Delete image and stored copy…')
        purge_btn.clicked.connect(self.delete_image_permanently)
        left.addWidget(purge_btn)
        left.addWidget(QLabel('Objects on this image'))
        self.objects = QListWidget()
        self.objects.currentItemChanged.connect(self.load_object)
        left.addWidget(self.objects, 1)
        line=QHBoxLayout()
        add_btn=QPushButton('＋ Object');add_btn.clicked.connect(self.add_object);line.addWidget(add_btn)
        remove_btn=QPushButton('− Object');remove_btn.clicked.connect(self.remove_object);line.addWidget(remove_btn)
        left.addLayout(line)
        left.addWidget(QLabel('Object descriptor'))
        self.name_edit = QLineEdit(); self.name_edit.setPlaceholderText('e.g. col50, upper plate, object 2')
        self.name_edit.editingFinished.connect(self.update_object_label)
        left.addWidget(self.name_edit)
        left.addWidget(QLabel('Existing particle label (optional)'))
        self.label_combo=QComboBox()
        self.label_combo.addItem('— Unassigned —', '')
        for label in library.labels.values():
            self.label_combo.addItem(label.name, label.name)
        self.label_combo.currentIndexChanged.connect(self.update_object_label)
        left.addWidget(self.label_combo)
        copy_btn=QPushButton('Copy selected measurements to reference label…')
        copy_btn.setToolTip('Only with confirmation: update the linked label’s SEM length, width and projected area for batch export.')
        copy_btn.clicked.connect(self.copy_to_reference_label)
        left.addWidget(copy_btn)
        splitter.addWidget(left_widget)

        self.canvas = MicroscopeCanvas()
        self.canvas.edited.connect(self.persist)
        self.canvas.scale_drawn.connect(self.set_scale_after_draw)
        self.canvas.crop_callback = self._crop_from_rectangle
        self.canvas.shared_mode = True
        splitter.addWidget(self.canvas)

        controls_scroll=QScrollArea(); controls_scroll.setWidgetResizable(True)
        controls=QWidget(); col=QVBoxLayout(controls)
        col.addWidget(QLabel('Boundary / graticule tools'))
        self.tools=QComboBox()
        for key,name in [('edit','Edit vertices / move circles'),('select','Select detected outline'),
                         ('polygon','Draw polygon (double-click to finish)'),('circle','Draw circular boundary'),
                         ('scale','Draw X-axis graticule reference'),('scale_y','Draw Y-axis graticule reference'),
                         ('crop','Drag crop rectangle (export only)')]:
            self.tools.addItem(name,key)
        self.tools.currentIndexChanged.connect(lambda _:self.canvas.set_tool(self.tools.currentData()))
        col.addWidget(self.tools)
        self.detection_mode=QComboBox();self.detection_mode.addItem('Relative darkness','relative');self.detection_mode.addItem('Canny / Otsu','original')
        col.addWidget(self.detection_mode)
        self.detection_fraction=QDoubleSpinBox();self.detection_fraction.setRange(1,95);self.detection_fraction.setValue(25)
        self.detection_fraction.setSuffix('% darker');col.addWidget(self.detection_fraction)
        detect_btn=QPushButton('Find candidate boundaries');detect_btn.clicked.connect(self.find_objects);col.addWidget(detect_btn)
        finish_btn=QPushButton('Finish polygon');finish_btn.clicked.connect(self.canvas.finish_polygon);col.addWidget(finish_btn)
        undo_btn=QPushButton('Undo polygon vertex');undo_btn.clicked.connect(self.canvas.undo_point);col.addWidget(undo_btn)
        fit_btn=QPushButton('Fit minimum enclosing circle');fit_btn.clicked.connect(self.fit_circle);col.addWidget(fit_btn)
        clear_btn=QPushButton('Clear selected outline');clear_btn.clicked.connect(self.clear_boundary);col.addWidget(clear_btn)
        fit_view=QPushButton('Fit image to view');fit_view.clicked.connect(self.canvas.fit_image);col.addWidget(fit_view)

        scale_box=QGroupBox('Two-way graticule calibration')
        scale_form=QGridLayout(scale_box)
        scale_form.addWidget(QLabel('Draw each reference along the corresponding graticule axis. '
                                    'One line can calibrate both if pixels are square.'),0,0,1,2)
        scale_form.addWidget(QLabel('X reference length'),1,0)
        self.scale_x=QDoubleSpinBox(); self.scale_x.setRange(0.000001,1e7);self.scale_x.setDecimals(6);self.scale_x.setSuffix(' µm')
        self.scale_x.valueChanged.connect(lambda v:self.scale_changed('x',v))
        scale_form.addWidget(self.scale_x,1,1)
        scale_form.addWidget(QLabel('Y reference length'),2,0)
        self.scale_y=QDoubleSpinBox();self.scale_y.setRange(0.000001,1e7);self.scale_y.setDecimals(6);self.scale_y.setSuffix(' µm')
        self.scale_y.valueChanged.connect(lambda v:self.scale_changed('y',v))
        scale_form.addWidget(self.scale_y,2,1)
        self.scale_status=QLabel('Not calibrated'); self.scale_status.setWordWrap(True)
        scale_form.addWidget(self.scale_status,3,0,1,2)
        col.addWidget(scale_box)

        group=QGroupBox('Selected particle measurements'); ml=QVBoxLayout(group)
        self.metrics=QLabel('Select or add an object.');self.metrics.setWordWrap(True)
        self.metrics.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ml.addWidget(self.metrics);col.addWidget(group)

        export_group=QGroupBox('Annotated export (original is unchanged)')
        eg=QGridLayout(export_group)
        eg.addWidget(QLabel('Rotate clockwise'),0,0)
        self.export_rotate=QDoubleSpinBox();self.export_rotate.setRange(-360,360);self.export_rotate.setDecimals(1)
        self.export_rotate.setSuffix('°'); self.export_rotate.valueChanged.connect(self.preview_rotation)
        eg.addWidget(self.export_rotate,0,1)
        self.crop_spins=[]
        for idx,(name,val) in enumerate((('Left %',0),('Top %',0),('Right %',100),('Bottom %',100))):
            w=QDoubleSpinBox();w.setRange(0,100);w.setDecimals(3);w.setValue(val)
            eg.addWidget(QLabel(name),idx//2+1,2*(idx%2));eg.addWidget(w,idx//2+1,2*(idx%2)+1)
            self.crop_spins.append(w)
        self.crop_aspect=QComboBox()
        for name,ratio in [('Free crop',0),('Square',1),('4:3',4/3),('3:2',1.5),('16:9',16/9),('Custom',-1)]:
            self.crop_aspect.addItem(name,ratio)
        self.crop_aspect.currentIndexChanged.connect(self.update_crop_aspect)
        eg.addWidget(self.crop_aspect,4,0)
        self.custom_aspect=QDoubleSpinBox();self.custom_aspect.setRange(.1,20);self.custom_aspect.setValue(1.5)
        self.custom_aspect.valueChanged.connect(self.update_crop_aspect)
        eg.addWidget(self.custom_aspect,4,1)
        self.export_circle = QCheckBox('Include all enclosing circles')
        self.export_circle.setChecked(True)
        eg.addWidget(self.export_circle,5,0,1,2)
        self.export_boxes = QCheckBox('Measurements in translucent boxes beside each particle')
        self.export_boxes.setChecked(False)
        eg.addWidget(self.export_boxes,6,0,1,4)
        col.addWidget(export_group)
        export_btn=QPushButton('Export annotated shared image PNG…');export_btn.clicked.connect(self.export_image)
        col.addWidget(export_btn)
        csv_btn=QPushButton('Export all object measurements CSV…');csv_btn.clicked.connect(self.export_measurements)
        col.addWidget(csv_btn)
        col.addStretch(1)
        controls_scroll.setWidget(controls);controls_scroll.setMinimumWidth(300)
        splitter.addWidget(controls_scroll)
        splitter.setSizes([245,940,330])
        self.status=QLabel('Import a graticule image, draw the X/Y scale lines, then add and trace its particles.')
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.refresh_images()

    def refresh_images(self, chosen=None):
        previous = chosen or (self.active_image.get('id') if self.active_image else None)
        self._loading=True;self.images.clear()
        for entry in self.store.entries:
            item=QListWidgetItem(f"{entry.get('original_filename','image')}\n   {len(entry.get('objects', []))} particle(s)")
            item.setData(Qt.UserRole,entry['id']);self.images.addItem(item)
            if entry['id']==previous:self.images.setCurrentItem(item)
        self._loading=False
        if self.images.currentItem() is None and self.images.count():self.images.setCurrentRow(0)
        elif self.images.currentItem():self.load_image(self.images.currentItem())
        if not self.images.count():
            self.active_image=None;self.active_object=None
            self.canvas.image=None;self.canvas.entry=None;self.canvas.paint_scene()
            self.refresh_objects()

    def import_images(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Import shared microscope / graticule images','',
                                           'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)')
        last=None
        for path in paths:
            try:
                read_display_image(path)
                last=self.store.import_image(path)['id']
            except (ValueError,OSError) as exc:
                QMessageBox.warning(self,'Import error',f'{Path(path).name}: {exc}')
        if last:self.refresh_images(last)

    def remove_image(self):
        if not self.active_image:return
        if QMessageBox.question(self,'Remove shared image?',
                'Remove the image and its object annotations from the shared index? '
                'The original image copy is retained for recovery.') != QMessageBox.Yes:
            return
        self.store.delete(self.active_image['id']);self.active_image=None;self.active_object=None
        self.refresh_images()

    def delete_image_permanently(self):
        if not self.active_image:
            return
        filename = self.active_image.get('original_filename', self.active_image.get('file', 'image'))
        if QMessageBox.warning(
            self, 'Permanently delete shared image?',
            f'Delete {filename}, all its traced objects, and the imported image copy?\n\n'
            'The original source file will NOT be deleted.\n\nThis cannot be undone.',
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        try:
            self.store.delete(self.active_image['id'], remove_file=True)
        except Exception as exc:
            QMessageBox.warning(self, 'Deletion failed', str(exc))
        self.active_image = None
        self.active_object = None
        self.refresh_images()

    def load_image(self, item, *_):
        if self._loading or item is None:return
        entry=next((e for e in self.store.entries if e['id']==item.data(Qt.UserRole)),None)
        if entry is None:return
        try:
            arr,img=read_display_image(self.store.image_path(entry))
        except (OSError,ValueError) as exc:
            QMessageBox.warning(self,'Read error',str(exc));return
        self.active_image=entry;self.active_object=None
        self._pixel_array=arr;self._qimage=img
        for spin,key in ((self.scale_x,'scale_um'),(self.scale_y,'scale_um_y')):
            spin.blockSignals(True);spin.setValue(float(entry.get(key) or 10));spin.blockSignals(False)
        self.refresh_objects()
        self.preview_rotation(self.export_rotate.value())

    def refresh_objects(self, selected=None):
        previous=selected or (self.active_object.get('object_id') if self.active_object else None)
        self._loading=True;self.objects.clear()
        if self.active_image:
            for obj in self.active_image.get('objects',[]):
                label = obj.get('label') or 'unassigned'
                name = obj.get('name') or label
                item=QListWidgetItem(f'{name}   [{label}]')
                item.setData(Qt.UserRole,obj['object_id'])
                self.objects.addItem(item)
                if obj['object_id']==previous:self.objects.setCurrentItem(item)
        self._loading=False
        if self.objects.currentItem() is None and self.objects.count():self.objects.setCurrentRow(0)
        elif self.objects.currentItem():self.load_object(self.objects.currentItem())
        if not self.objects.count():
            self.active_object=None
            if self.active_image:
                self.canvas.shared_image = self.active_image
                self.canvas.set_image(self._pixel_array,self._qimage, None,'Shared image')
            self.canvas.other_annotations=[]
            self.metrics.setText('Add an object, then draw its boundary.')

    def add_object(self):
        if not self.active_image:
            QMessageBox.information(self,'Shared image','Import or select an image first.');return
        objects=self.active_image.setdefault('objects',[])
        n=len(objects)+1
        obj={'object_id':uuid.uuid4().hex,'name':f'Object {n}','label':'',
             'mode':None,'boundary':[],'boundary_circle':None,'enclosing_circle':None}
        objects.append(obj);self.store.update(self.active_image)
        self.refresh_objects(obj['object_id'])

    def remove_object(self):
        if not self.active_image or not self.active_object:return
        if QMessageBox.question(self,'Remove outline?','Remove this object boundary and its saved measurements?') != QMessageBox.Yes:return
        oid=self.active_object['object_id']
        self.active_image['objects']=[o for o in self.active_image['objects'] if o['object_id']!=oid]
        self.active_object=None;self.store.update(self.active_image);self.refresh_objects()

    def load_object(self, item, *_):
        if self._loading or not self.active_image or item is None:return
        obj=next((o for o in self.active_image.get('objects',[]) if o['object_id']==item.data(Qt.UserRole)),None)
        if not obj:return
        self.active_object=obj
        self.name_edit.blockSignals(True);self.name_edit.setText(obj.get('name') or '');self.name_edit.blockSignals(False)
        self.label_combo.blockSignals(True)
        idx=self.label_combo.findData(obj.get('label') or '')
        self.label_combo.setCurrentIndex(max(0,idx))
        self.label_combo.blockSignals(False)
        self.rebind_canvas()
        self.show_metrics()

    def rebind_canvas(self):
        if not self.active_image:return
        if self.active_object is None:
            self.canvas.shared_image = self.active_image
            self.canvas.set_image(self._pixel_array,self._qimage,None,'Shared image')
            self.canvas.other_annotations=list(self.active_image.get('objects',[]))
            self.canvas.paint_scene();return
        obj=self.active_object
        # Interactive canvas expects the current photo's calibration on entry.
        for k in ('scale_line','scale_um','scale_line_y','scale_um_y'):
            obj[k]=self.active_image.get(k)
        self.canvas.shared_image = self.active_image
        self.canvas.other_annotations=[o for o in self.active_image.get('objects',[]) if o is not obj]
        self.canvas.set_image(self._pixel_array,self._qimage,obj,obj.get('name') or obj.get('label') or 'Object')
        self.preview_rotation(self.export_rotate.value())

    def update_object_label(self, *_):
        if self._loading or not self.active_object:return
        self.active_object['name']=self.name_edit.text().strip() or 'Unlabelled particle'
        self.active_object['label']=self.label_combo.currentData() or ''
        self.persist(); self.refresh_objects(self.active_object['object_id'])

    def persist(self):
        if not self.active_image:return
        if self.active_object:
            for k in ('scale_line','scale_um','scale_line_y','scale_um_y'):
                self.active_image[k]=self.active_object.get(k)
        try:
            self.store.update(self.active_image)
        except (ValueError, OSError) as exc:
            self.status.setText(f'Calibration or save error: {exc}')
        self.show_metrics()

    def show_metrics(self):
        if not self.active_image:return
        scales=physical_scales(self.active_image)
        try:
            graticule_transform(self.active_image)
            invalid_graticule = None
        except ValueError as exc:
            invalid_graticule = str(exc)
        if invalid_graticule:
            self.scale_status.setText('Invalid graticule: ' + invalid_graticule)
            self.metrics.setText('Please redraw the graticule reference lines.')
            return
        if scales:
            self.scale_status.setText(f'X: {scales[0]:.8g} µm/px; Y: {scales[1]:.8g} µm/px'
                + (' — check differing X/Y calibration' if abs(scales[0]-scales[1])/max(scales)>0.05 else ''))
        else:
            self.scale_status.setText('No physical calibration (pixel measures only)')
        if not self.active_object:
            self.metrics.setText('Select or add an object.');return
        m=measure_shared_object(self.active_image,self.active_object)
        if not m.get('max_dimension_px'):
            self.metrics.setText('Draw a boundary around this particle.');return
        physical = bool(scales)
        unit = 'µm' if physical else 'px'
        areaunit='µm²' if physical else 'px²'
        def number(metric):
            val=m.get(metric)
            return f'{val:.3f}' if val is not None and math.isfinite(float(val)) else '—'
        suffix='um' if physical else 'px'
        self.metrics.setText(
            f"Maximum Feret length: {number('max_dimension_'+suffix)} {unit}\n"
            f"Minimum Feret width: {number('feret_min_width_'+suffix)} {unit}\n"
            f"Ellipse minor-axis width: {number('cpi_minor_axis_width_'+suffix)} {unit}\n"
            f"Projected area: {number('area_um2' if physical else 'area_px2')} {areaunit}\n"
            f"Equivalent-area diameter: {number('equivalent_area_diameter_'+suffix)} {unit}\n"
            f"Enclosing-circle area: {number('enclosing_circle_area_um2' if physical else 'enclosing_circle_area_px2')} {areaunit}\n"
            f"Area / enclosing-circle area: {number('area_over_circle')}")

    def find_objects(self):
        if not self.active_object:return
        self.canvas.candidates=(detect_relative_shadow(self.canvas.pixels,self.detection_fraction.value()/100.)
                                if self.detection_mode.currentData()=='relative'
                                else detect_candidates(self.canvas.pixels))
        self.tools.setCurrentIndex(self.tools.findData('select'));self.canvas.paint_scene()
        self.status.setText(f'{len(self.canvas.candidates)} candidate outlines. Click the one belonging to the selected object.')

    def fit_circle(self):
        if not self.active_object:return
        try:
            self.active_object['enclosing_circle']=enclosing_circle(self.active_object)
        except ValueError as exc:
            QMessageBox.information(self,'Enclosing circle',str(exc));return
        self.persist();self.canvas.paint_scene()

    def clear_boundary(self):
        if not self.active_object:return
        self.active_object.update(mode=None,boundary=[],boundary_circle=None,enclosing_circle=None)
        self.canvas.pending=[];self.canvas.candidates=[]
        self.persist();self.canvas.paint_scene()

    def set_scale_after_draw(self):
        if not self.active_image or self.active_object is None:return
        y_axis = self.canvas.mode=='scale_y'
        line_key='scale_line_y' if y_axis else 'scale_line'
        len_key='scale_um_y' if y_axis else 'scale_um'
        line=self.active_object.get(line_key)
        if not line or math.dist(*line)<3:
            self.active_object[line_key]=None;self.active_object[len_key]=None
            self.persist();return
        length,ok=QInputDialog.getDouble(self,'Graticule calibration',
                 ('Y' if y_axis else 'X')+' line: actual graticule distance (µm)',
                 float(self.active_image.get(len_key) or 10), 1e-6, 1e7, 6)
        if ok:
            self.active_object[len_key]=float(length)
            spin=self.scale_y if y_axis else self.scale_x
            spin.blockSignals(True);spin.setValue(float(length));spin.blockSignals(False)
        else:
            self.active_object[len_key]=None
        self.persist();self.canvas.paint_scene()
        self.tools.setCurrentIndex(0)

    def scale_changed(self, axis, length):
        if not self.active_image:return
        key='scale_line_y' if axis=='y' else 'scale_line'
        if not self.active_image.get(key):return
        field='scale_um_y' if axis=='y' else 'scale_um'
        self.active_image[field]=length
        if self.active_object:self.active_object[field]=length
        self.persist();self.canvas.paint_scene()

    def update_crop_aspect(self,*_):
        value=self.crop_aspect.currentData()
        self.canvas.crop_aspect=self.custom_aspect.value() if value==-1 else float(value or 0)

    def _crop_from_rectangle(self, fractions):
        for spin,val in zip(self.crop_spins,fractions):
            spin.setValue(round(100*min(1,max(0,val)),3))

    def preview_rotation(self, angle):
        if self.canvas.image is None:return
        self.canvas.resetTransform();self.canvas.fit_image()
        self.canvas.rotate(float(angle))
        self.canvas.centerOn(self.canvas.stage.sceneRect().center())

    def copy_to_reference_label(self):
        if not self.active_image or not self.active_object:
            return
        name = self.active_object.get('label')
        label = self.library.get(name) if name else None
        if label is None:
            QMessageBox.information(self,'Reference label','Choose an existing label for this object first.')
            return
        metrics = measure_shared_object(self.active_image, self.active_object)
        if not metrics.get('max_dimension_um') or not metrics.get('area_um2'):
            QMessageBox.information(self,'Reference label','A closed boundary and a valid graticule scale are required.')
            return
        definitions=['Minimum Feret width', 'Inertia-ellipse minor-axis width']
        selected, ok = QInputDialog.getItem(self, 'Reference width definition',
            'Which width should be saved to the reference label?', definitions, 0, False)
        if not ok:return
        width=(metrics['feret_min_width_um'] if selected == definitions[0]
               else metrics['cpi_minor_axis_width_um'])
        length=metrics['max_dimension_um']; area=metrics['area_um2']
        prompt=(f"Update existing label '{label.name}' from the selected microscopy object?\n\n"
                f"Maximum Feret length: {length:.3f} µm\n"
                f"{selected}: {width:.3f} µm\n"
                f"Projected area: {area:.3f} µm²\n\n"
                'This overwrites those reference values in labels.json for every batch '
                'using this label. The shared image and other objects are unchanged.')
        if QMessageBox.question(self,'Confirm reference measurement update',prompt) != QMessageBox.Yes:
            return
        label.length_um=float(length)
        label.width_um=float(width)
        label.area_um2=float(area)
        self.library.upsert(label)
        try:
            self.library.save()
        except OSError as exc:
            QMessageBox.warning(self,'Cannot update label',str(exc))
            return
        self.status.setText(f"Reference label '{label.name}' updated from the selected object.")

    def export_measurements(self):
        if not self.active_image:return
        path,_=QFileDialog.getSaveFileName(self,'Save all measured particles',
                            Path(self.active_image['original_filename']).stem+'_particles.csv','CSV file (*.csv)')
        if not path:return
        rows=[]
        for obj in self.active_image.get('objects',[]):
            row={'image_id':self.active_image['id'], 'image':self.active_image['original_filename'],
                 'object_id':obj['object_id'],'name':obj.get('name'), 'label':obj.get('label')}
            row.update(measure_shared_object(self.active_image,obj))
            rows.append(row)
        fields=list(dict.fromkeys(k for row in rows for k in row))
        try:
            with open(path,'w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields)
                writer.writeheader();writer.writerows(rows)
            self.status.setText(f'Saved {len(rows)} measurements: {path}')
        except OSError as exc:
            QMessageBox.warning(self,'CSV export error',str(exc))

    def export_image(self):
        if not self.active_image:return
        path,_=QFileDialog.getSaveFileName(self,'Save annotated shared microscope image',
                         Path(self.active_image['original_filename']).stem+'_objects.png','PNG image (*.png)')
        if not path:return
        try:
            from .export_transform import crop_rotate_image
        except ImportError:
            from export_transform import crop_rotate_image
        # Render image + geometric overlays in original coordinates. Rotate and
        # crop them together, then draw a clean upright measurement legend.
        image=QImage(self._qimage.width(),self._qimage.height(),QImage.Format_ARGB32)
        image.fill(Qt.white)
        p=QPainter(image)
        try:
            p.drawImage(0,0,self._qimage)
            p.setRenderHint(QPainter.Antialiasing,True)
            for i,obj in enumerate(self.active_image.get('objects',[]),1):
                pen=QPen(QColor('#00eca9'));pen.setWidth(max(2,int(image.width()*.002)));p.setPen(pen)
                if obj.get('mode')=='polygon' and obj.get('boundary'):
                    points=obj['boundary'];outline=QPainterPath();outline.moveTo(*map(float,points[0]))
                    for xy in points[1:]:outline.lineTo(*map(float,xy))
                    outline.closeSubpath();p.drawPath(outline)
                elif obj.get('mode')=='circle' and obj.get('boundary_circle'):
                    x,y,r=obj['boundary_circle'];p.drawEllipse(QRectF(x-r,y-r,2*r,2*r))
                circle=obj.get('enclosing_circle') if self.export_circle.isChecked() else None
                if circle and len(circle)==3:
                    x,y,r=circle
                    pen=QPen(QColor('#ffa23f'));pen.setWidth(max(2,int(image.width()*.0015)));p.setPen(pen)
                    p.drawEllipse(QRectF(x-r,y-r,2*r,2*r))
            if self.export_boxes.isChecked():
                font_box=QFont('Arial');font_box.setPixelSize(max(12,min(21,image.width()//65)))
                from PySide6.QtGui import QFontMetrics
                metrics=QFontMetrics(font_box)
                # Candidate blocks are evaluated against every particle polygon, not its circle.
                occupancy=[]
                for obj in self.active_image.get('objects',[]):
                    pts=obj.get('boundary') or []
                    if obj.get('mode')=='circle' and obj.get('boundary_circle'):
                        cx,cy,rad=obj['boundary_circle']
                        polygon=QPainterPath();polygon.addEllipse(QRectF(cx-rad,cy-rad,2*rad,2*rad))
                    elif len(pts)>=3:
                        polygon=QPainterPath();polygon.moveTo(*map(float,pts[0]))
                        for xy in pts[1:]:polygon.lineTo(*map(float,xy))
                        polygon.closeSubpath()
                    else:
                        continue
                    occupancy.append(polygon)
                occupied_boxes=[]
                p.setFont(font_box)
                for i,obj in enumerate(self.active_image.get('objects',[]),1):
                    m=measure_shared_object(self.active_image,obj)
                    calibrated=m.get('scale_x_um_per_px') is not None
                    suffix='um' if calibrated else 'px'; unit='µm' if calibrated else 'px'
                    def fmt(key):
                        value=m.get(key)
                        return f'{value:.2f}' if value is not None else '—'
                    lines=[f'{i}. {obj.get("name") or obj.get("label") or "Object"}',
                           f'L {fmt("max_dimension_"+suffix)} {unit}',
                           f'W(Feret) {fmt("feret_min_width_"+suffix)} {unit}',
                           f'W(ellipse) {fmt("cpi_minor_axis_width_"+suffix)} {unit}',
                           f'Area {fmt("area_um2" if calibrated else "area_px2")} {unit}²',
                           f'Area ratio {fmt("area_over_circle")}']
                    box_w=min(image.width()-16,max(metrics.horizontalAdvance(line) for line in lines)+16)
                    box_h=metrics.height()*len(lines)+12
                    if box_w<=0 or box_h<=0 or box_h>image.height()-16:continue
                    best=None
                    for ry in np.linspace(0.05,0.83,14):
                        for rx in np.linspace(0.02,0.96,18):
                            left=min(image.width()-box_w-5,max(5,rx*(image.width()-box_w)))
                            top=min(image.height()-box_h-5,max(5,ry*(image.height()-box_h)))
                            rect=QRectF(left,top,box_w,box_h)
                            score=sum(1000 if poly.intersects(QPainterPath_from_rect(rect)) else 0 for poly in occupancy)
                            score+=sum(1000 if rect.intersects(other) else 0 for other in occupied_boxes)
                            # Guard shared graticule near horizontal/vertical central axes.
                            for axis in ('','_y'):
                                line=self.active_image.get('scale_line'+axis)
                                if line:
                                    from PySide6.QtCore import QLineF
                                    path=QPainterPath();path.moveTo(*line[0]);path.lineTo(*line[1])
                                    if path.intersects(QPainterPath_from_rect(rect)):score+=500
                            score+=abs(rect.center().x()-image.width()*.5)*.005
                            if best is None or score<best[0]:best=(score,rect)
                    if best:
                        rect=best[1];occupied_boxes.append(rect)
                        p.fillRect(rect,QColor(255,255,255,190));p.setPen(QColor('#202020'))
                        for j,line in enumerate(lines):
                            p.drawText(QRectF(rect.x()+8,rect.y()+6+j*metrics.height(),box_w-12,metrics.height()),
                                       Qt.AlignLeft|Qt.AlignVCenter,line)
            for axis, colour in (('',QColor('#ffe866')),('_y',QColor('#ff87f8'))):
                line=self.active_image.get('scale_line'+axis)
                if line:
                    pen=QPen(colour);pen.setWidth(max(2,int(image.width()*.002)));p.setPen(pen)
                    p.drawLine(QPointF(*line[0]),QPointF(*line[1]))
        finally:p.end()
        crop=[s.value()/100 for s in self.crop_spins]
        image=crop_rotate_image(image,crop,self.export_rotate.value())
        # Place compact, upright numbered markers adjacent to each outline in
        # final export coordinates so a shared photo's many objects are
        # unambiguously matched to the measurement legend.
        try:
            from .export_transform import rotated_geometry
        except ImportError:
            from export_transform import rotated_geometry
        forward, _, rw, rh = rotated_geometry(self._qimage.width(), self._qimage.height(),
                                              self.export_rotate.value())
        numfont=QFont('Arial');numfont.setPixelSize(max(13,min(32,image.width()//48)))
        marker=QPainter(image)
        try:
            marker.setFont(numfont)
            for i,obj in enumerate(self.active_image.get('objects',[]),1):
                if obj.get('mode') == 'circle' and obj.get('boundary_circle'):
                    cx,cy,r=obj['boundary_circle']
                    anchor=QPointF(cx-r,cy-r)
                elif obj.get('boundary'):
                    pts=np.asarray(obj['boundary'],dtype=float)
                    anchor=QPointF(float(pts[:,0].min()),float(pts[:,1].min()))
                else:
                    continue
                rp=forward.map(anchor)
                left=rp.x()-crop[0]*rw
                top=rp.y()-crop[1]*rh
                if not (-25<left<image.width()+25 and -25<top<image.height()+25):
                    continue
                left=max(3,min(image.width()-45,left+5))
                top=max(3,min(image.height()-38,top-35))
                rect=QRectF(left,top,44,34)
                marker.fillRect(rect,QColor(255,255,255,190))
                marker.setPen(QColor('#202020'))
                marker.drawText(rect,Qt.AlignCenter,str(i))
        finally:marker.end()
        # A separate white legend strip avoids obscuring either particle outlines
        # or two-way graticule bars, and remains upright after any rotation.
        objs=self.active_image.get('objects',[])
        font=QFont('Arial');font.setPixelSize(max(16,min(32,image.width()//40)))
        from PySide6.QtGui import QFontMetrics
        met=QFontMetrics(font)
        pad=12;line_h=met.height()+5
        if objs and not self.export_boxes.isChecked():
            legend_h=len(objs)*line_h+2*pad
            combined=QImage(image.width(),image.height()+legend_h,QImage.Format_ARGB32)
            combined.fill(Qt.white)
            p=QPainter(combined)
            try:
                p.drawImage(0,0,image);p.setFont(font);p.setPen(QColor('#202020'))
                for i,obj in enumerate(objs,1):
                    m=measure_shared_object(self.active_image,obj)
                    cal=m.get('scale_x_um_per_px') is not None
                    unit='µm' if cal else 'px'
                    name=obj.get('name') or obj.get('label') or f'Object {i}'
                    def fmt(field):
                        v=m.get(field)
                        return f'{v:.2f}' if v is not None else '—'
                    suffix='um' if cal else 'px'
                    text=(f'{i}. {name} [{obj.get("label") or "unassigned"}]  '
                          f'L {fmt("max_dimension_"+suffix)} {unit}; '
                          f'W(Feret) {fmt("feret_min_width_"+suffix)} {unit}; '
                          f'W(ellipse) {fmt("cpi_minor_axis_width_"+suffix)} {unit}; '
                          f'Area {fmt("area_um2" if cal else "area_px2")} {unit}²; '
                          f'Area ratio {fmt("area_over_circle")}')
                    p.drawText(QRectF(pad,image.height()+pad+(i-1)*line_h,
                                     combined.width()-2*pad,line_h),Qt.AlignLeft|Qt.AlignVCenter,text)
            finally:p.end()
            image=combined
        if not image.save(path,'PNG'):
            QMessageBox.warning(self,'PNG export error','Could not save the image.')
        else:
            self.status.setText(f'Saved shared image: {path}')

class MicroscopyPanelDialog(QDialog):
    """Publication montage: selected labels, explicit SEM/optical source per label.

    Blank choices are supported for either column. All panel text is drawn in
    final output coordinates at a shared font size, independently of source DPI.
    """
    def __init__(self, library, parent=None):
        super().__init__(parent)
        self.library = library
        self.store = MicroscopyStore(library.path)
        self.setWindowTitle('Export SEM + optical microscopy figure panel')
        self.resize(920, 650)
        root = QVBoxLayout(self)
        root.addWidget(QLabel('Select particle labels and one image for each SEM / optical panel. '
                              'Either image may be left blank. Source boundaries and measurements remain unchanged.'))
        scroller = QScrollArea(); scroller.setWidgetResizable(True)
        holder = QWidget(); form = QGridLayout(holder)
        for j, name in enumerate(('Include', 'Particle label', 'SEM image', 'Optical image')):
            form.addWidget(QLabel(f'<b>{name}</b>'), 0, j)
        self.rows = []
        for i, label in enumerate(library.labels.values(), 1):
            entries = self.store.list_for(label.name, label.aliases)
            include = QCheckBox(); include.setChecked(bool(entries))
            form.addWidget(include, i, 0)
            form.addWidget(QLabel(label.name), i, 1)
            combos=[]
            for kind in ('sem', 'optical'):
                combo=QComboBox(); combo.addItem('— leave blank —', None)
                default=0
                for j, entry in enumerate(entries, 1):
                    filename=entry.get('original_filename', entry['file'])
                    combo.addItem(filename, entry['id'])
                    low=filename.lower()
                    guessed_sem=('sem' in low or 'electron' in low)
                    if (kind=='sem' and guessed_sem) or (kind=='optical' and not guessed_sem):
                        if default==0: default=j
                combo.setCurrentIndex(default)
                form.addWidget(combo,i,2 if kind=='sem' else 3)
                combos.append(combo)
            self.rows.append((include,label,entries,*combos))
        scroller.setWidget(holder); root.addWidget(scroller,1)
        opts=QGridLayout()
        opts.addWidget(QLabel('Shared text size (px)'),0,0)
        self.font_px=QSpinBox(); self.font_px.setRange(14,90); self.font_px.setValue(30)
        opts.addWidget(self.font_px,0,1)
        opts.addWidget(QLabel('Layout'),0,2)
        self.layout_combo=QComboBox(); self.layout_combo.addItem('Two-column (left / right)','columns'); self.layout_combo.addItem('SEM top row / optical bottom row','stacked')
        opts.addWidget(self.layout_combo,0,3)
        self.show_circle=QCheckBox('Show fitted circle'); self.show_circle.setChecked(True)
        opts.addWidget(self.show_circle,1,0,1,2)
        self.show_letters=QCheckBox('Show panel letters'); self.show_letters.setChecked(True)
        opts.addWidget(self.show_letters,1,2,1,2)
        opts.addWidget(QLabel('Text placement'),2,0)
        self.text_place=QComboBox(); self.text_place.addItem('Automatic (avoid particle)','auto'); self.text_place.addItem('Top on image','top'); self.text_place.addItem('Bottom on image','bottom')
        opts.addWidget(self.text_place,2,1,1,3)
        opts.setColumnStretch(4,1)
        root.addLayout(opts)
        root.addWidget(QLabel('Left/right: one particle per row. Top/bottom: ALL SEM images form the first row and ALL optical images the second row. '
                              'Boundary outlines are carried across where available, and text is drawn upright over the final cropped image.'))
        bar=QHBoxLayout(); export=QPushButton('Export publication panel PNG…'); export.clicked.connect(self.export_panel)
        bar.addStretch();bar.addWidget(export)
        close=QPushButton('Close'); close.clicked.connect(self.accept);bar.addWidget(close)
        root.addLayout(bar)

    @staticmethod
    def _entry_boundary_points(entry):
        if not entry:
            return []
        if entry.get('mode') == 'polygon' and entry.get('boundary'):
            return [(float(x), float(y)) for x, y in entry['boundary']]
        circle = entry.get('boundary_circle')
        if entry.get('mode') == 'circle' and circle and len(circle) == 3:
            x,y,r = map(float, circle)
            import math
            return [(x + r*math.cos(t), y + r*math.sin(t)) for t in np.linspace(0, 2*np.pi, 96, endpoint=False)]
        return []

    @classmethod
    def _annotation_position(cls, entry, text_place, dst_rect, lines_px=110,
                             letter_visible=True, sem=False, text_width=None,
                             source_size=None):
        """Choose an annotation rectangle avoiding the filled particle silhouette.

        Use the *actual* source image size supplied by the renderer, not
        optional metadata dimensions. The enclosing circle is intentionally
        ignored. SEM top/bottom instrument strips and panel letters are kept
        clear when a legal placement exists.
        """
        import cv2
        import numpy as np
        margin = 9.0
        width = min(max(1., float(text_width or 160)), max(1., dst_rect.width()-2*margin))
        top = dst_rect.y()+dst_rect.height()*(.13 if sem else .025)
        bottom = dst_rect.bottom()-dst_rect.height()*(.19 if sem else .025)
        height = min(float(lines_px), max(1., bottom-top-2*margin))
        xmin, xmax = dst_rect.x()+margin, dst_rect.right()-margin-width
        ymin, ymax = top+margin, bottom-margin-height
        xmax=max(xmin,xmax); ymax=max(ymin,ymax)

        # Low-resolution occupancy image in final rendered coordinates. Extend
        # the particle mask slightly to protect its outline from the text.
        mask_w=max(1, int(np.ceil(dst_rect.width())))
        mask_h=max(1, int(np.ceil(dst_rect.height())))
        occupancy=np.zeros((mask_h,mask_w),dtype=np.uint8)
        points=cls._entry_boundary_points(entry)
        if len(points)>=3:
            sw,sh=source_size or (entry.get('width') or 0, entry.get('height') or 0)
            sw=float(sw or 0); sh=float(sh or 0)
            if sw>0 and sh>0:
                pts=np.asarray(points,dtype=float)
                pts[:,0]*=dst_rect.width()/sw
                pts[:,1]*=dst_rect.height()/sh
                polygon=np.rint(pts).astype(np.int32)
                cv2.fillPoly(occupancy,[polygon],255)
                buffer_px=max(4,int(round(dst_rect.width()*.012)))
                k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*buffer_px+1,2*buffer_px+1))
                occupancy=cv2.dilate(occupancy,k)
        integral=cv2.integral(occupancy.astype(np.float32)/255.)
        def collision(x,y):
            x0=int(max(0,min(mask_w,np.floor(x-dst_rect.x()))))
            y0=int(max(0,min(mask_h,np.floor(y-dst_rect.y()))))
            x1=int(max(x0,min(mask_w,np.ceil(x+width-dst_rect.x()))))
            y1=int(max(y0,min(mask_h,np.ceil(y+height-dst_rect.y()))))
            area=max(1,(x1-x0)*(y1-y0))
            covered=(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])
            return float(covered/area)
        x_candidates=np.linspace(xmin,xmax,17) if xmax>xmin else [xmin]
        y_candidates=np.linspace(ymin,ymax,21) if ymax>ymin else [ymin]
        if text_place=='top':
            y_candidates=y_candidates[:max(1,len(y_candidates)//4)]
        elif text_place=='bottom':
            y_candidates=y_candidates[-max(1,len(y_candidates)//4):]
        def penalty(x,y):
            overlap=collision(x,y)
            letter_overlap=0.
            if letter_visible:
                lx,ly=dst_rect.x()+12,dst_rect.y()+12
                letter_overlap=max(0,min(x+width,lx+76)-max(x,lx))*max(0,min(y+height,ly+48)-max(y,ly))
            # Prioritise no overlap with particle or letter; only then favour
            # a neat right-hand / lower position.
            return (1000.*overlap+1000.*letter_overlap/max(width*height,1.)
                    +.005*((xmax-x)/max(1,xmax-xmin)+(ymax-y)/max(1,ymax-ymin)))
        x,y=min(((float(x),float(y)) for y in y_candidates for x in x_candidates),
                key=lambda xy: penalty(*xy))
        return QRectF(x,y,width,height)

    @staticmethod
    def _scaled_image(painter, entry, store, rect, show_circle=False):
        _,source=read_display_image(store.image_path(entry))
        iw,ih=source.width(),source.height()
        if iw<1 or ih<1:
            return rect
        scale=min(rect.width()/iw,rect.height()/ih)
        dw,dh=iw*scale,ih*scale
        dst=QRectF(rect.x()+(rect.width()-dw)/2, rect.y()+(rect.height()-dh)/2,dw,dh)
        painter.drawImage(dst,source)
        painter.save(); painter.translate(dst.x(),dst.y());painter.scale(scale,scale)
        painter.setRenderHint(QPainter.Antialiasing)
        pen=QPen(QColor('#00eeae'));pen.setWidthF(max(1.5,2.5/scale));painter.setPen(pen)
        if entry.get('mode')=='polygon' and entry.get('boundary'):
            poly=entry['boundary'];path=QPainterPath();path.moveTo(*map(float,poly[0]))
            for pt in poly[1:]:path.lineTo(*map(float,pt))
            path.closeSubpath();painter.drawPath(path)
        elif entry.get('mode')=='circle' and entry.get('boundary_circle'):
            x,y,r=entry['boundary_circle'];painter.drawEllipse(QRectF(x-r,y-r,2*r,2*r))
        if show_circle:
            circle = entry.get('enclosing_circle')
            if circle and len(circle) == 3 and float(circle[2]) > 0:
                pen2=QPen(QColor('#ffb347')); pen2.setWidthF(max(1.25,2.2/scale)); painter.setPen(pen2)
                x,y,r = map(float, circle)
                painter.drawEllipse(QRectF(x-r,y-r,2*r,2*r))
        painter.restore()
        return dst, (iw, ih)

    def export_panel(self):
        selected = []
        for checked, label, entries, sem, opt in self.rows:
            if checked.isChecked():
                lookup = {e['id']: e for e in entries}
                selected.append((label, lookup.get(sem.currentData()), lookup.get(opt.currentData())))
        if not selected:
            QMessageBox.information(self, 'Panel export', 'Select at least one label.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save publication panel',
                                              'SEM_optical_panel.png', 'PNG image (*.png)')
        if not path:
            return
        try:
            stacked = self.layout_combo.currentData() == 'stacked'
            placement = self.text_place.currentData() or 'auto'
            px = self.font_px.value()
            margin, gutter = 40, 30
            # No external headings/captions: information goes on the image itself.
            ncols = len(selected) if stacked else 2
            nrows = 2 if stacked else len(selected)
            cell_width = 740 if stacked else 1140
            cell_height = 670
            width = 2 * margin + ncols * cell_width + (ncols - 1) * gutter
            height = 2 * margin + nrows * cell_height + (nrows - 1) * gutter
            canvas = QImage(width, height, QImage.Format_ARGB32)
            canvas.fill(QColor('white'))
            painter = QPainter(canvas)
            try:
                painter.setRenderHint(QPainter.Antialiasing)
                font = QFont('Arial'); font.setPixelSize(px)
                small = QFont('Arial'); small.setPixelSize(max(16, int(px * .9)))
                panels = []
                if stacked:
                    for row in range(2):
                        for col, (label, sem, optical) in enumerate(selected):
                            panels.append((row, col, label, sem if row == 0 else optical,
                                           'SEM' if row == 0 else 'Optical microscope'))
                else:
                    for row, (label, sem, optical) in enumerate(selected):
                        panels.extend(((row, 0, label, sem, 'SEM'),
                                       (row, 1, label, optical, 'Optical microscope')))
                for letter_index, (row, col, label, entry, kind) in enumerate(panels):
                    x = margin + col * (cell_width + gutter)
                    y = margin + row * (cell_height + gutter)
                    frame = QRectF(x, y, cell_width, cell_height)
                    painter.setPen(QColor('#b8b8b8'))
                    painter.drawRect(frame)
                    image_rect = QRectF(x + 10, y + 10, cell_width - 20, cell_height - 20)
                    dst = image_rect
                    source_size = None
                    if entry is not None:
                        dst, source_size = self._scaled_image(painter, entry, self.store, image_rect,
                                                 show_circle=self.show_circle.isChecked())
                    if self.show_letters.isChecked():
                        number = letter_index
                        letters = ''
                        while True:
                            letters = chr(97 + number % 26) + letters
                            number = number // 26 - 1
                            if number < 0:
                                break
                        badge = QRectF(dst.x() + 12, dst.y() + 12, 76, 48)
                        painter.fillRect(badge, QColor(255, 255, 255, 185))
                        painter.setPen(QColor('#202020'))
                        painter.setFont(font)
                        painter.drawText(badge, Qt.AlignCenter, f'({letters})')
                    if entry is None:
                        continue
                    m = measure(entry)
                    unit = 'µm' if m.get('um_per_px') else 'px'
                    width_value = (m.get('feret_min_width_um') if m.get('um_per_px')
                                   else m.get('feret_min_width_px'))
                    lines = [label.name]
                    length_value = (m.get('max_dimension_um') if m.get('um_per_px')
                                    else m.get('max_dimension_px'))
                    area_value = (m.get('area_um2') if m.get('um_per_px')
                                  else m.get('area_px2'))
                    if length_value is not None:
                        lines.append(f'Length {length_value:.2f} {unit}')
                    if width_value is not None:
                        lines.append(f'Width {width_value:.2f} {unit}')
                    if area_value is not None:
                        lines.append(f'Area {area_value:.1f} {unit}²')
                    if m.get('area_over_circle') is not None:
                        lines.append(f'Area ratio {m["area_over_circle"]:.3f}')
                    # Text-sized box: do not reserve an arbitrary portion of image.
                    # Render each line without wrapping so the background is tight.
                    from PySide6.QtGui import QFontMetricsF
                    painter.setFont(small)
                    metrics=QFontMetricsF(small)
                    line_height=metrics.lineSpacing()
                    block_width=max(metrics.horizontalAdvance(t) for t in lines)+16
                    block_height=line_height*len(lines)+12
                    text_rect=self._annotation_position(
                        entry, placement, dst, lines_px=block_height,
                        text_width=block_width,
                        letter_visible=self.show_letters.isChecked(), sem=(kind == 'SEM'),
                        source_size=source_size)
                    painter.fillRect(text_rect, QColor(255,255,255,175))
                    painter.setPen(QColor('#202020'))
                    baseline=text_rect.y()+6+metrics.ascent()
                    for line in lines:
                        painter.drawText(QPointF(text_rect.x()+8,baseline),line)
                        baseline+=line_height
            finally:
                painter.end()
            if not canvas.save(path, 'PNG'):
                raise OSError('Image could not be saved')
            QMessageBox.information(self, 'Panel exported',
                                    f'Saved {len(selected)} particle labels to:\n{path}')
        except Exception as exc:
            QMessageBox.warning(self, 'Export failed', str(exc))
