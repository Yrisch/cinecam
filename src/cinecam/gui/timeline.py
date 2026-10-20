# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Timeline: keyframes, simulation dump track, proxy cache strip, playhead."""

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF


class Timeline(QtWidgets.QWidget):
    M = 14

    def __init__(self, ed):
        super().__init__()
        self.ed = ed
        self.setMinimumHeight(70)
        self.setMaximumHeight(80)
        self._drag = None
        self._span = None  # time scale frozen while a ripple drag changes the duration

    def span(self):
        return self._span or self.ed.path.duration

    def tx(self, t):
        return self.M + t / max(self.span(), 1e-9) * (self.width() - 2 * self.M)

    def xt(self, x):
        return (x - self.M) / max(self.width() - 2 * self.M, 1) * self.span()

    def paintEvent(self, ev):
        ed = self.ed
        path = ed.path
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        H = self.height()
        p.fillRect(self.rect(), QColor(24, 25, 30))
        x0, x1 = self.tx(0), self.tx(path.duration)

        # dump track
        ks = path.keyframes
        if ks and path.dumps:
            ts = np.array([k.t for k in ks])
            D = np.array([k.state.dump for k in ks])
            xs = np.arange(int(x0), int(x1) + 1)
            di = np.clip(np.round(np.interp(self.xt(xs), ts, D)), 0, len(path.dumps) - 1).astype(int)
            start = 0
            for i in range(1, len(xs) + 1):
                if i == len(xs) or di[i] != di[start]:
                    c = QColor(60, 70, 95) if di[start] % 2 == 0 else QColor(48, 56, 78)
                    p.fillRect(QRectF(xs[start], H - 18, xs[i - 1] - xs[start] + 1, 14), c)
                    if xs[i - 1] - xs[start] > 30:
                        p.setPen(QColor(170, 180, 200))
                        p.drawText(QRectF(xs[start] + 3, H - 18, xs[i - 1] - xs[start], 14), Qt.AlignmentFlag.AlignVCenter, f"dump {di[start]}")
                    start = i
            # proxy cache: green where the gas of the frame's dump is already in memory
            if ed.proxy is not None:
                cached = {i for i in set(di.tolist()) if ed.proxy.has(path.dumps[i])}
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(90, 200, 120))
                start = None
                for i in range(len(xs) + 1):
                    on = i < len(xs) and di[i] in cached
                    if on and start is None:
                        start = i
                    elif not on and start is not None:
                        p.drawRect(QRectF(xs[start], H - 4, xs[i - 1] - xs[start] + 1, 2))
                        start = None
                p.setBrush(Qt.BrushStyle.NoBrush)

        # ruler
        p.setPen(QColor(90, 90, 100))
        p.drawLine(QPointF(x0, 26), QPointF(x1, 26))
        step = 1.0
        while (self.tx(step) - x0) < 40:
            step *= 2
        t = 0.0
        while t <= path.duration + 1e-9:
            x = self.tx(t)
            p.setPen(QColor(110, 110, 120))
            p.drawLine(QPointF(x, 22), QPointF(x, 30))
            p.setPen(QColor(150, 150, 160))
            p.drawText(QPointF(x + 2, 15), f"{t:g}s")
            t += step

        # stretched segments: the time between the keys was changed with their ends' speed kept
        for a, b in zip(ks, ks[1:]):
            if abs(a.stretch - 1) > 1e-6:
                xa, xb = self.tx(a.t), self.tx(b.t)
                p.setPen(QPen(QColor(230, 160, 80), 2))
                p.drawLine(QPointF(xa + 7, 40), QPointF(xb - 7, 40))
                if xb - xa > 40:
                    p.drawText(QRectF(xa, 41, xb - xa, 12), Qt.AlignmentFlag.AlignHCenter, f"×{a.stretch:.2g}")

        # keyframes
        on = ed.key_index()
        for i, k in enumerate(ks):
            x = self.tx(k.t)
            col = QColor(255, 210, 60) if i == on else QColor(120, 200, 255)
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.setBrush(col)
            p.drawPolygon(QPolygonF([QPointF(x, 32), QPointF(x + 6, 40), QPointF(x, 48), QPointF(x - 6, 40)]))
            if k.state.follow is not None:  # following a sink: ring around the diamond
                p.setPen(QPen(QColor(80, 230, 230), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(x, 40), 9, 9)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # playhead
        x = self.tx(ed.t)
        p.setPen(QPen(QColor(255, 70, 70), 2))
        p.drawLine(QPointF(x, 18), QPointF(x, H - 2))

    def _key_near(self, x, y):
        if not 28 <= y <= 52:
            return None
        for i, k in enumerate(self.ed.path.keyframes):
            if abs(self.tx(k.t) - x) < 7:
                return i
        return None

    def mousePressEvent(self, ev):
        x, y = ev.position().x(), ev.position().y()
        i = self._key_near(x, y)
        if ev.button() == Qt.MouseButton.RightButton:
            menu = QtWidgets.QMenu(self)
            acts = {}
            if i is not None:
                acts[menu.addAction(f"Delete keyframe {i + 1}")] = lambda: self.ed.delete_key(i)
            s = self.ed.path.segment_at(self.xt(x)) if i is None else None
            if s is not None:
                acts[menu.addAction(f"Duration from key {s + 1} to key {s + 2}…")] = lambda: self.ed.ask_segment_duration(s)
            if acts:
                fn = acts.get(menu.exec(ev.globalPosition().toPoint()))
                if fn is not None:
                    fn()
            return
        if i is not None and i > 0 and ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._drag = ("ripple", i)  # stretch the segment before key i, the keys after it follow
            self._span = self.ed.path.duration
            self.ed.set_time(self.ed.path.keyframes[i].t)
        elif i is not None:
            self._drag = ("key", self.ed.path.keyframes[i])
            self.ed.set_time(self.ed.path.keyframes[i].t)
        else:
            self._drag = ("scrub", None)
            self.ed.set_time(self.xt(x))

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        x = ev.position().x()
        ed = self.ed
        if self._drag[0] == "ripple":
            i = self._drag[1]
            ks = ed.path.keyframes
            t = ed.path.snap(max(self.xt(x), ks[i - 1].t))
            if abs(t - ks[i].t) > 0.5 / ed.path.fps:
                ed.path.retime_segment(i - 1, t - ks[i - 1].t)
                ed.set_time(ks[i].t)
            return
        t = float(np.clip(ed.path.snap(self.xt(x)), 0, ed.path.duration))
        if self._drag[0] == "key":
            k = self._drag[1]
            if all(abs(o.t - t) > 0.5 / ed.path.fps for o in ed.path.keyframes if o is not k):
                k.t = t
                ed.path.sort()
                ed.set_time(t)
        else:
            ed.set_time(t)

    def mouseReleaseEvent(self, ev):
        if self._drag and self._drag[0] in ("key", "ripple"):
            self.ed.commit()
        self._drag = None
        self._span = None
        self.update()
