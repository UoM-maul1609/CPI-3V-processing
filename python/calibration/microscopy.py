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
        'equivalent_area_diameter_px': 2 * math.sqrt(area / math.pi) if area > 0 else None,
        'enclosing_circle_area_px2': circle_area if circle_area > 0 else None,
        'area_over_circle': ratio,
        'um_per_px': um_per_px,
        'max_dimension_um': diameter * um_per_px if diameter > 0 and um_per_px else None,
        'area_um2': area * um_per_px**2 if area > 0 and um_per_px else None,
        'equivalent_area_diameter_um': 2 * math.sqrt(area / math.pi) * um_per_px if area > 0 and um_per_px else None,
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

    def delete(self, image_id, *, remove_file=False):
        """Remove an indexed image; optionally delete its local imported copy.

        Never delete the user's source image, only the copy placed under
        the microscopy library. Index changes are saved before unlinking.
        """
        matches = [e for e in self.entries if e['id'] == image_id]
        if not matches:
            raise KeyError('Unknown microscopy image')
        copied_path = self.image_path(matches[0]) if remove_file else None
        self.entries = [e for e in self.entries if e['id'] != image_id]
        try:
            self.save()
        except Exception:
            self.entries.append(matches[0])
            raise
        if copied_path is not None:
            # The copied file is unique per import. Do not follow links or
            # remove other resources even if an index was edited manually.
            if copied_path.exists():
                if copied_path.is_symlink() or not copied_path.is_file():
                    raise ValueError('Refusing to delete a non-regular microscopy file')
                copied_path.unlink()

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


# Shared graticule images contain several independent particles. The calibration
# belongs to the photograph, never to any one particle or reference label.
def physical_scales(image):
    """Return (µm/column pixel, µm/row pixel), or None when uncalibrated.

    One scale line can calibrate both directions; the optional second graticule
    direction provides an independent scale for non-square pixels.
    """
    values = []
    for suffix in ('', '_y'):
        line = image.get('scale_line' + suffix)
        length = image.get('scale_um' + suffix)
        try:
            distance = float(np.linalg.norm(np.asarray(line[1], float) - np.asarray(line[0], float)))
            length = float(length)
            values.append(length / distance if distance > 0 and length > 0 and math.isfinite(length) else None)
        except (TypeError, IndexError, ValueError):
            values.append(None)
    sx, sy = values
    if sx is None and sy is None:
        return None
    return (sx if sx is not None else sy, sy if sy is not None else sx)


def graticule_transform(image):
    """2x2 pixel-to-micrometre transform, including an oblique 2-way graticule.

    Two known vectors (pixel displacement between grid marks) identify the
    physical axes; one known line assumes isotropic pixels. A nearly parallel
    pair cannot define a 2D transformation and is deliberately rejected.
    """
    scales = physical_scales(image)
    if scales is None:
        return None
    line1, line2 = image.get('scale_line'), image.get('scale_line_y')
    length1, length2 = image.get('scale_um'), image.get('scale_um_y')
    if line1 and line2 and length1 and length2:
        v1 = np.diff(np.asarray(line1, dtype=float), axis=0).reshape(2)
        v2 = np.diff(np.asarray(line2, dtype=float), axis=0).reshape(2)
        basis = np.column_stack([v1, v2])
        if abs(np.linalg.det(basis)) < 0.01 * np.linalg.norm(v1) * np.linalg.norm(v2):
            raise ValueError('The X/Y graticule reference lines are nearly parallel.')
        return np.diag([float(length1), float(length2)]) @ np.linalg.inv(basis)
    s = scales[0] if line1 else scales[1]
    return np.eye(2) * s


def measure_shared_object(image, obj):
    """Use the same drawn boundary for all physical measures; never change labels.

    Polygon-area and Feret calculations are performed in calibrated coordinates
    so separate horizontal/vertical graticule scales are handled correctly.
    The ellipse width uses continuous filled-polygon second moments.
    """
    result = measure({k: v for k,v in obj.items() if k not in ('scale_line', 'scale_um')})
    scales = physical_scales(image)
    result['scale_x_um_per_px'] = scales[0] if scales else None
    result['scale_y_um_per_px'] = scales[1] if scales else None
    transform = graticule_transform(image)
    if transform is None or not result.get('max_dimension_px'):
        return result
    pts = _boundary_points(obj) @ transform.T
    if len(pts) < 3:
        return result
    from scipy.spatial.distance import pdist
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    result['max_dimension_um'] = float(pdist(hull).max()) if len(hull) >= 2 else None
    result['area_um2'] = polygon_area(pts)
    result['equivalent_area_diameter_um'] = (2 * math.sqrt(result['area_um2'] / math.pi)
                                             if result['area_um2'] > 0 else None)
    circle = obj.get('enclosing_circle')
    if circle and len(circle) == 3 and circle[2] > 0:
        result['enclosing_circle_area_um2'] = math.pi * float(circle[2])**2 * abs(float(np.linalg.det(transform)))
    edges = np.roll(hull, -1, axis=0) - hull
    norms = np.linalg.norm(edges, axis=1)
    good = norms > 1e-8
    if good.any():
        normals = np.column_stack((-edges[good,1],edges[good,0])) / norms[good,None]
        result['feret_min_width_um'] = float(np.min(np.ptp(hull @ normals.T, axis=0)))
    if obj.get('mode') == 'circle':
        # Under unequal x/y scales a source-pixel circle becomes an ellipse.
        r = float(obj['boundary_circle'][2])
        result['cpi_minor_axis_width_um'] = 2*r*float(np.linalg.svd(transform, compute_uv=False).min())
    else:
        shifted = (pts - pts.mean(axis=0)).astype(np.float32)
        moments = cv2.moments(shifted)
        m00 = float(moments['m00'])
        if m00 > 1e-12:
            cov = np.array([[moments['mu20'],moments['mu11']],
                            [moments['mu11'],moments['mu02']]],float) / m00
            result['cpi_minor_axis_width_um'] = 4*math.sqrt(max(0.0, float(np.linalg.eigvalsh(cov)[0])))
    return result


class SharedMicroscopyStore(MicroscopyStore):
    """One source image, one/two graticule scales, any number of named outlines.

    Uses a distinct index so existing single-label microscopy annotations and
    publication-panel image selectors remain backward-compatible.
    """
    def __init__(self, label_library_path):
        super().__init__(label_library_path)
        self.index_path = self.root / 'shared_images.json'
        if self.index_path.exists():
            self.entries = list(json.loads(self.index_path.read_text(encoding='utf-8')).get('shared_images', []))
        else:
            self.entries = []

    def import_image(self, path):
        entry = super().import_image(path, '')
        entry.update(objects=[], scale_line_y=None, scale_um_y=None)
        self.save()
        return entry

    @staticmethod
    def _persistent(entry):
        import copy
        stored = copy.deepcopy(entry)
        for obj in stored.get('objects', []):
            for key in ('scale_line', 'scale_um', 'scale_line_y', 'scale_um_y'):
                obj.pop(key, None)
        return stored

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.shared_microscopy.', suffix='.tmp', dir=self.root)
        os.close(fd)
        try:
            data = {'format_version': 1,
                    'shared_images': [self._persistent(e) for e in self.entries]}
            Path(tmp).write_text(json.dumps(data, indent=2), encoding='utf-8')
            os.replace(tmp, self.index_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def update(self, entry):
        for obj in entry.get('objects', []):
            obj['measurements'] = measure_shared_object(entry, obj)
        for i, old in enumerate(self.entries):
            if old['id'] == entry['id']:
                self.entries[i] = entry
                self.save()
                return
        raise KeyError('Unknown shared microscopy image')
