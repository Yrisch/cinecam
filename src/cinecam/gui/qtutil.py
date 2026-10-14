"""Small Qt helpers shared by the widgets."""

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QPen, QPolygonF


def np_to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def fit_rect(w, h, aspect):
    """Largest rect of the given aspect centred in (w, h)."""
    if w / max(h, 1) > aspect:
        rw, rh = h * aspect, h
    else:
        rw, rh = w, w / aspect
    return QRectF((w - rw) / 2, (h - rh) / 2, rw, rh)


def draw_dots(p, xs, ys, radius, color):
    """Round dots of one size and colour in a single drawPoints call (much faster than ellipses)."""
    if len(xs) == 0:
        return
    pen = QPen(color, 2 * radius)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawPoints(QPolygonF([QPointF(x, y) for x, y in zip(xs.tolist(), ys.tolist())]))

