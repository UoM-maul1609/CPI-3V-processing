"""Microscopy reference-image annotations linked to the shared CPI label library.

The microscope image is copied, unchanged, beside the label library.  All
annotations and measurements are stored in a separate JSON index.  No MAT
files or original microscope images are modified.

The coordinate convention is pixel centres, x to the right and y downwards.
The physical calibration is micrometres per pixel from a drawn reference line.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import pdist


def polygon_area(points) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    return abs(float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) -
                     np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))) / 2.0


def _boundary_points(annotation):
    mode = annotation.get('mode', '')
    if mode == 'circle':
        circle = annotation.get('boundary_circle')
        if not circle or len(circle) != 3 or circle[2] <= 0:
            return np.empty((0, 2), dtype=np.float64)
        t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        return np.column_stack((circle[0] + circle[2] * np.cos(t),
                                circle[1] + circle[2] * np.sin(t)))
    if mode == 'polygon':
        pts = np.asarray(annotation.get('boundary') or [], dtype=np.float64)
        return pts.reshape(-1, 2) if pts.size else np.empty((0, 2), dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def enclosing_circle(annotation):
    """Return a circle containing the *entire* chosen boundary."""
    if annotation.get('mode') == 'circle':
        source = annotation.get('boundary_circle')
        if source and len(source) == 3 and source[2] > 0:
            return [float(v) for v in source]
        raise ValueError('Draw a boundary first.')
    pts = _boundary_points(annotation)
    if len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError('Select or draw a closed object boundary first.')
    (x, y), r = cv2.minEnclosingCircle(pts.astype(np.float32))
    # Roundoff protection also ensures the returned circle contains the whole
    # polygon even for coordinates represented beyond float32 precision.
    r = max(float(r), float(np.linalg.norm(pts - [x, y], axis=1).max())) + 1e-5
    return [float(x), float(y), r]


def measure(annotation: dict) -> dict:
    """Return pixel and calibrated measurements; never invent a physical scale."""
    mode = annotation.get('mode', '')
    pts = _boundary_points(annotation)
    if mode == 'circle' and len(pts):
        r = float(annotation['boundary_circle'][2])
        area = math.pi * r * r
        diameter = 2 * r
    elif mode == 'polygon' and len(pts) >= 3:
        area = polygon_area(pts)
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        diameter = float(pdist(hull).max()) if len(hull) >= 2 else 0.0
    else:
        area, diameter = 0.0, 0.0
    # CPI-compatible width: minor axis of inertia-equivalent ellipse fitted
    # to the filled contour (same regionprops definition as edgeDetect.py).
    width_px = None
    if area > 0:
        if mode == "circle":
            width_px = diameter
        elif mode == "polygon":
            polygon = np.asarray(pts, dtype=np.float32)
            x0,y0 = np.floor(polygon.min(axis=0)).astype(int) - 2
            xy = np.rint(polygon - [x0,y0]).astype(np.int32)
            ww,hh = np.ceil(polygon.max(axis=0) - [x0,y0]).astype(int) + 3
            mask = np.zeros((max(2,hh),max(2,ww)),dtype=np.uint8)
            cv2.fillPoly(mask,[xy],1)
            if mask.sum() >= 3:
                try:
                    from skimage.measure import regionprops
                    prop = regionprops(mask)[0]
                    width_px = float(prop.axis_minor_length)
                except ImportError:
                    yy,xx = np.nonzero(mask)
                    cov = np.cov(np.stack([xx,yy]),bias=True)
                    width_px = 4*math.sqrt(max(0,float(np.linalg.eigvalsh(cov)[0])))
    # Feret width is the minimum projection span over convex-hull edge normals.
    # CPI production's `wid` currently uses the regionprops minor ellipse axis;
    # retain it separately rather than mixing the two definitions.
    feret_width_px = None
    if area > 0:
        if mode == 'circle':
            feret_width_px = diameter
        elif mode == 'polygon':
            hull = cv2.convexHull(np.asarray(pts, dtype=np.float32)).reshape(-1, 2).astype(float)
            if len(hull) >= 3:
                edges = np.roll(hull, -1, axis=0) - hull
                norms = np.linalg.norm(edges, axis=1)
                valid = norms > 1e-12
                normals = np.column_stack((-edges[valid, 1], edges[valid, 0])) / norms[valid, None]
                if len(normals):
                    projections = hull @ normals.T
                    feret_width_px = float(np.min(np.ptp(projections, axis=0)))
    circle = annotation.get('enclosing_circle')
    circle_area = math.pi * float(circle[2]) ** 2 if circle and len(circle) == 3 else 0.0
    scale = annotation.get('scale_line')
    size_um = annotation.get('scale_um')
    um_per_px = None
    if scale is not None and size_um is not None:
        try:
            px = float(np.linalg.norm(np.asarray(scale[1], float) - np.asarray(scale[0], float)))
            length = float(size_um)
            if px > 0 and math.isfinite(px) and length > 0 and math.isfinite(length):
                um_per_px = length / px
        except (ValueError, TypeError, IndexError):
            pass
    ratio = (area / circle_area) if area > 0 and circle_area > 0 else None
    return {
        'max_dimension_px': diameter if diameter > 0 else None,
        'feret_min_width_px': feret_width_px,
        'feret_min_width_um': feret_width_px * um_per_px if feret_width_px is not None and um_per_px else None,
        'cpi_minor_axis_width_px': width_px,
        'cpi_minor_axis_width_um': width_px * um_per_px if width_px and um_per_px else None,
        'area_px2': area if area > 0 else None,
        'enclosing_circle_area_px2': circle_area if circle_area > 0 else None,
        'area_over_circle': ratio,
        'um_per_px': um_per_px,
        'max_dimension_um': diameter * um_per_px if diameter > 0 and um_per_px else None,
        'area_um2': area * um_per_px**2 if area > 0 and um_per_px else None,
        'enclosing_circle_area_um2': circle_area * um_per_px**2 if circle_area > 0 and um_per_px else None,
    }


def detect_candidates(image, max_candidates=80):
    """Return contour candidates for hand-selection; nothing is auto-accepted.

    Includes bright and dark objects. Edges are Canny-based, with an Otsu
    candidate set as fallback. Candidates are ordered largest first, while the
    GUI picks the smallest *enclosing* contour at the clicked position.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        lo, hi = np.percentile(gray, [1, 99])
        gray = np.uint8(np.clip((gray.astype(float) - lo) * 255 / max(hi-lo, 1), 0, 255))
    height, width = gray.shape[:2]
    # Large SEM images can have tens of millions of pixels. Detect on a
    # limited-size preview but always return vertices in *native* pixels so
    # measurements and scale lines remain exact at the original resolution.
    reduction = max(1., width / 2200., height / 2200.)
    if reduction > 1:
        gray = cv2.resize(gray, (int(round(width/reduction)), int(round(height/reduction))),
                          interpolation=cv2.INTER_AREA)
    h_work, w_work = gray.shape
    # Avoid applying a blur kernel that is huge for tiny cropped pictures.
    smooth = cv2.GaussianBlur(gray, (5, 5), 0.9)
    edges = cv2.Canny(smooth, 35, 115)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _th, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (otsu, cv2.bitwise_not(otsu)):
        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours) + list(found)
    min_area = max(12., 0.000015 * h_work * w_work)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > 0.92 * w_work * h_work or len(cnt) < 3:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue
        perimeter = cv2.arcLength(cnt, True)
        simplified = cv2.approxPolyDP(cnt, max(0.6, perimeter * 0.0015), True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        candidates.append((area * reduction**2,
                           [[float(x*reduction), float(y*reduction)] for x,y in simplified]))
    candidates.sort(key=lambda c: c[0], reverse=True)
    # IoU near-duplicates aren't useful in the selection view.
    unique = []
    for area, pts in candidates:
        if any(abs(area - old_area) / max(area, old_area) < .04 and
               np.linalg.norm(np.mean(pts, axis=0) - np.mean(old_pts, axis=0)) < 3
               for old_area, old_pts in unique):
            continue
        unique.append((area, pts))
        if len(unique) >= max_candidates:
            break
    return [pts for _, pts in unique]



def detect_relative_shadow(image, reduction_fraction=0.25, max_candidates=80):
    """Find dark objects relative to robust clear background; returns native-pixel outlines."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    gray = gray.astype(np.float32)
    h, w = gray.shape
    reduction = max(1.0, w / 2200., h / 2200.)
    if reduction > 1:
        gray = cv2.resize(gray, (round(w / reduction), round(h / reduction)), interpolation=cv2.INTER_AREA)
    # Outer border may include a scale bar, so use a high robust quantile.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    bg = float(np.percentile(border, 65))
    if not np.isfinite(bg) or bg <= 0:
        return []
    level = bg * (1.0 - float(reduction_fraction))
    mask = np.uint8(cv2.GaussianBlur(gray, (5, 5), 0.8) < level) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(9, mask.size * .00001) or area > mask.size * .90:
            continue
        perimeter = cv2.arcLength(contour, True)
        pts = cv2.approxPolyDP(contour, max(0.5, perimeter * .0015), True).reshape(-1, 2)
        if len(pts) >= 3:
            out.append((area, [[float(x * reduction), float(y * reduction)] for x, y in pts]))
    out.sort(key=lambda item: item[0], reverse=True)
    return [pts for _, pts in out[:max_candidates]]

class MicroscopyStore:
    """Shared attachment library, located next to the labels.json library."""
    def __init__(self, label_library_path):
        self.root = Path(label_library_path).expanduser().resolve().parent / 'microscopy'
        self.index_path = self.root / 'index.json'
        self.entries = []
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding='utf-8'))
            self.entries = list(data.get('images', []))

    def list_for(self, label, aliases=()):
        names = {label, *aliases}
        return [entry for entry in self.entries if entry.get('label') in names]

    def image_path(self, entry):
        path = (self.root / entry['file']).resolve()
        # The index is user-editable JSON but should never read outside root.
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError('Invalid microscopy image path')
        return path

    def import_image(self, path, label):
        original = Path(path).expanduser().resolve()
        if not original.is_file():
            raise FileNotFoundError(str(original))
        ext = original.suffix.lower()
        if ext not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}:
            raise ValueError('Unsupported image type')
        self.root.mkdir(parents=True, exist_ok=True)
        rel = f'images/{uuid.uuid4().hex}{ext}'
        (self.root / 'images').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, self.root / rel)
        entry = {
            'id': uuid.uuid4().hex, 'label': label, 'file': rel,
            'original_filename': original.name, 'imported_utc': datetime.now(timezone.utc).isoformat(),
            'mode': None, 'boundary': [], 'boundary_circle': None, 'enclosing_circle': None,
            'scale_line': None, 'scale_um': None, 'measurements': {},
        }
        self.entries.append(entry)
        self.save()
        return entry

    def update(self, entry):
        entry['measurements'] = measure(entry)
        for i, old in enumerate(self.entries):
            if old['id'] == entry['id']:
                self.entries[i] = dict(entry)
                self.save()
                return
        raise KeyError('Unknown microscopy image')

    def delete(self, image_id):
        # Only metadata is removed; originals copied into the attachment folder
        # are intentionally retained for recovery.
        self.entries = [e for e in self.entries if e['id'] != image_id]
        self.save()

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.microscopy.', suffix='.tmp', dir=self.root)
        os.close(fd)
        try:
            Path(tmp).write_text(json.dumps({'format_version': 1, 'images': self.entries}, indent=2),
                                 encoding='utf-8')
            os.replace(tmp, self.index_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
