"""Presentation transforms: crop coordinates refer to the ROTATED image."""
from PySide6.QtCore import QRect, QRectF, QPointF, Qt
from PySide6.QtGui import QTransform, QPolygonF


def rotated_geometry(width, height, clockwise):
    """Forward/inverse mapping between source and expanded rotated canvas."""
    transform = QTransform().rotate(float(clockwise))
    bounds = transform.mapRect(QRectF(0, 0, width, height))
    forward = QTransform().rotate(float(clockwise)) * QTransform().translate(-bounds.left(), -bounds.top())
    inverse, ok = forward.inverted()
    if not ok:
        raise ValueError('Noninvertible rotation')
    return forward, inverse, bounds.width(), bounds.height()


def rotated_crop_rect(start, end, width, height, clockwise, aspect=0):
    forward, inverse, rw, rh = rotated_geometry(width, height, clockwise)
    a, b = forward.map(QPointF(*start)), forward.map(QPointF(*end))
    x0, y0 = max(0., min(rw, a.x())), max(0., min(rh, a.y()))
    x1, y1 = max(0., min(rw, b.x())), max(0., min(rh, b.y()))
    if aspect and aspect > 0:
        dx,dy=abs(x1-x0),abs(y1-y0)
        if dx/max(dy,1e-9)>aspect: dx=dy*aspect
        else: dy=dx/aspect
        from math import copysign
        x1=x0+copysign(min(dx, rw-x0 if x1>=x0 else x0),x1-x0 or 1)
        y1=y0+copysign(min(dy, rh-y0 if y1>=y0 else y0),y1-y0 or 1)
    rect=QRectF(QPointF(x0,y0),QPointF(x1,y1)).normalized()
    corners=[rect.topLeft(),rect.topRight(),rect.bottomRight(),rect.bottomLeft()]
    polygon=QPolygonF([inverse.map(pt) for pt in corners])
    fractions=(rect.left()/rw,rect.top()/rh,rect.right()/rw,rect.bottom()/rh)
    return polygon, fractions, rect


def crop_rotate_image(image, crop=(0,0,1,1), clockwise=0):
    """Rotate the whole source, then crop axis-aligned in final orientation."""
    rotated = image.transformed(QTransform().rotate(float(clockwise)), mode=Qt.FastTransformation) if clockwise % 360 else image
    w,h=rotated.width(),rotated.height()
    left,top,right,bottom=crop
    x0,y0=round(left*w),round(top*h)
    x1,y1=round(right*w),round(bottom*h)
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):
        raise ValueError('Crop must contain a nonempty region in rotated-image coordinates.')
    return rotated.copy(QRect(x0,y0,x1-x0,y1-y0))
