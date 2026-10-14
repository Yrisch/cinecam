# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""World view: the whole simulation, the camera path, keyframe cameras with drag handles."""

import math
import time

import numpy as np
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF

from ..path.camera import ZNEAR, CamState, frustum_corners, normalize, project, rotate, unproject_delta
from ..render.compose import gas_rgb, splat_column
from .keymaps import KEYMAPS
from .qtutil import draw_dots, np_to_qimage


def slerp(a, b, s, fallback_axis):
    """Rotate unit vector a toward unit vector b by the fraction s of the angle between them."""
    c = float(np.clip(a @ b, -1.0, 1.0))
    if c > 1 - 1e-9:
        return b.copy()
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-6:  # opposite vectors: turn around any perpendicular axis
        axis = fallback_axis - (fallback_axis @ a) * a
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    return normalize(rotate(a, axis, s * math.acos(c)))


class WorldView(QtWidgets.QWidget):
    def __init__(self, ed):
        super().__init__()
        self.ed = ed
        self.view = CamState(pos=[0, -90, 45], target=[0, 0, 0], up=[0, 0, 1], fov=40)
        self.setMinimumSize(300, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._drag = None
        self._hover = None
        self._bg = (None, None)
        self._gizmo_hover = None
        self._press = None
        self._anim = None
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(15)
        self._anim_timer.timeout.connect(self._anim_step)

    ANIM_TIME = 0.3
    GIZMO_R = 30  # axis length in pixels
    GIZMO_AXES = ((0, "X", QColor(240, 80, 80)), (1, "Y", QColor(110, 220, 110)), (2, "Z", QColor(90, 150, 255)))

    # -- helpers

    def screen(self, pts):
        return project(self.view, np.atleast_2d(pts), self.width(), self.height())

    def frame_all(self):
        s = self.ed.sample
        if s is None:
            return
        c = np.median(s, axis=0)
        R = np.percentile(np.linalg.norm(s - c, axis=1), 98)
        self.view.target = c.astype(float)
        self.view.pos = c + normalize(self.view.pos - self.view.target) * R / math.tan(math.radians(self.view.fov) / 2) * 1.1
        self.update()

    def axis_view(self, key):
        self.look_from({"1": (1, -1), "3": (0, 1), "7": (2, 1)}[key])

    def look_from(self, axis):
        """Animate the view to look at the target from the side of the axis (index, sign)."""
        i, sign = axis
        side = np.zeros(3)
        side[i] = sign
        up = np.array([0.0, 1.0, 0.0]) if i == 2 else np.array([0.0, 0.0, 1.0])
        d0, _, u0 = self.view.basis()
        self._anim = (time.perf_counter(), -d0, u0, side, up)
        self._anim_timer.start()

    def _anim_step(self):
        t0, a0, u0, a1, u1 = self._anim
        s = min((time.perf_counter() - t0) / self.ANIM_TIME, 1.0)
        s = s * s * (3 - 2 * s)
        side = slerp(a0, a1, s, u0)
        up = slerp(u0, u1, s, -a0)
        dist = self.view.distance
        self.view.pos = self.view.target + side * dist
        self.view.up = up
        if s >= 1.0:
            self._anim_timer.stop()
        self.update()

    def _background(self):
        ed = self.ed
        W, H = max(self.width() // 2, 1), max(self.height() // 2, 1)
        key = (tuple(self.view.pos), tuple(self.view.target), tuple(self.view.up), W, H, ed.sample_id)
        if self._bg[0] == key:
            return self._bg[1]
        img = None
        if ed.sample is not None:
            # overview at half resolution: half of the sample is plenty (twice the weight)
            sig = splat_column(ed.sample[::2], ed.weight * 2, self.view, W, H, smooth=0.8)
            nz = sig[sig > 0]
            if nz.size:
                lo, hi = np.log10(np.percentile(nz, [30, 99.8]))
                rgb = gas_rgb(sig, (lo, hi + 0.3), "bone") * 0.85
                img = np_to_qimage((rgb * 255).astype(np.uint8))
        self._bg = (key, img)
        return img

    def _handles(self):
        """Screen positions of draggable things: (kind, index, x, y, z)."""
        ed = self.ed
        out = []
        on = ed.key_index()
        for i, k in enumerate(ed.path.keyframes):
            for kind, p in (("kpos", k.state.pos), ("ktgt", k.state.target)):
                sx, sy, z = self.screen(p)
                if z[0] > ZNEAR:
                    out.append((kind, i, sx[0], sy[0], z[0]))
        if on is None:
            for kind, p in (("lpos", ed.live.pos), ("ltgt", ed.live.target)):
                sx, sy, z = self.screen(p)
                if z[0] > ZNEAR:
                    out.append((kind, -1, sx[0], sy[0], z[0]))
        return out

    def _pick(self, x, y):
        best, bd = None, 9.0
        for h in self._handles():
            d = math.hypot(h[2] - x, h[3] - y)
            if d < bd:
                best, bd = h, d
        return best

    # -- painting

    def _line3(self, p, a, b):
        sx, sy, z = self.screen(np.array([a, b]))
        if (z > ZNEAR).all():
            p.drawLine(QPointF(sx[0], sy[0]), QPointF(sx[1], sy[1]))

    def _frustum(self, p, st, length, color, width=1.0):
        aspect = self.ed.aspect()
        cs = frustum_corners(st, aspect, length)
        pen = QPen(color, width)
        p.setPen(pen)
        for c in cs:
            self._line3(p, st.pos, c)
        for i in range(4):
            self._line3(p, cs[i], cs[(i + 1) % 4])
        # little triangle on the top edge shows "up" (roll)
        d, r, u = st.basis()
        top = (cs[0] + cs[1]) / 2
        h = np.linalg.norm(cs[0] - cs[1]) * 0.15
        tri = [cs[0] * 0.7 + cs[1] * 0.3, top + u * h, cs[0] * 0.3 + cs[1] * 0.7]
        sx, sy, z = self.screen(np.array(tri))
        if (z > ZNEAR).all():
            p.setBrush(color)
            p.drawPolygon(QPolygonF([QPointF(a, b) for a, b in zip(sx, sy)]))
            p.setBrush(Qt.BrushStyle.NoBrush)

    def paintEvent(self, ev):
        ed = self.ed
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(14, 15, 20))
        bg = self._background()
        if bg is not None:
            p.drawImage(QRectF(0, 0, self.width(), self.height()), bg)

        # sinks
        sinks = ed.display_sinks()
        if sinks is not None:
            for key, col, rad in (("star_pos", QColor(200, 220, 255), 1.6), ("core_pos", QColor(255, 110, 30), 2.2)):
                pts = sinks[key]
                if len(pts):
                    sx, sy, z = self.screen(pts)
                    ok = z > ZNEAR
                    draw_dots(p, sx[ok], sy[ok], rad, col)

        self._sink_rings(p)

        # path
        smp = ed.path.sample_positions(400)
        if smp is not None:
            tt, P, T = smp
            for arr, alpha, w in ((T, 70, 1.0), (P, 255, 2.0)):
                sx, sy, z = self.screen(arr)
                for i in range(len(tt) - 1):
                    if z[i] > ZNEAR and z[i + 1] > ZNEAR:
                        f = tt[i] / max(ed.path.duration, 1e-9)
                        c = QColor.fromHsvF(0.5 - 0.35 * f, 0.75, 1.0)
                        c.setAlpha(alpha)
                        p.setPen(QPen(c, w))
                        p.drawLine(QPointF(sx[i], sy[i]), QPointF(sx[i + 1], sy[i + 1]))

        # keyframes
        on = ed.key_index()
        L = 0.06 * self.view.distance
        for i, k in enumerate(ed.path.keyframes):
            sel = i == on
            col = QColor(255, 210, 60) if sel else QColor(120, 200, 255)
            self._frustum(p, k.state, L, col, 1.5 if sel else 1.0)
            pen = QPen(col, 1, Qt.PenStyle.DotLine)
            p.setPen(pen)
            self._line3(p, k.state.pos, k.state.target)
            sx, sy, z = self.screen(np.array([k.state.pos, k.state.target]))
            if z[0] > ZNEAR:
                p.setPen(QPen(QColor(0, 0, 0), 1))
                p.setBrush(col)
                p.drawEllipse(QPointF(sx[0], sy[0]), 5, 5)
                p.setPen(col)
                p.drawText(QPointF(sx[0] + 7, sy[0] - 7), str(i + 1))
            if z[1] > ZNEAR:
                p.setPen(QPen(col, 1.5))
                p.drawLine(QPointF(sx[1] - 4, sy[1]), QPointF(sx[1] + 4, sy[1]))
                p.drawLine(QPointF(sx[1], sy[1] - 4), QPointF(sx[1], sy[1] + 4))
            p.setBrush(Qt.BrushStyle.NoBrush)

        # live camera
        live = ed.live
        col = QColor(255, 255, 255) if on is None else QColor(255, 210, 60)
        self._frustum(p, live, L * 1.6, col, 2.0)
        if on is None:
            p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
            self._line3(p, live.pos, live.target)
            sx, sy, z = self.screen(np.array([live.pos, live.target]))
            p.setBrush(QColor(255, 80, 80) if ed.live_dirty else col)
            if z[0] > ZNEAR:
                p.drawEllipse(QPointF(sx[0], sy[0]), 5, 5)
            if z[1] > ZNEAR:
                p.drawEllipse(QPointF(sx[1], sy[1]), 3, 3)
            p.setBrush(Qt.BrushStyle.NoBrush)

        if self._hover is not None:
            h = self._hover
            p.setPen(QPen(QColor(255, 255, 255), 1.5))
            p.drawEllipse(QPointF(h[2], h[3]), 9, 9)

        self._gizmo(p)
        p.setPen(QColor(150, 150, 160))
        p.drawText(8, 16, "WORLD  ·  drag: orbit  shift: pan  wheel: zoom  F: frame  1/3/7 or click an axis: views")

    def _sink_rings(self, p):
        ed = self.ed
        for sink, col, rad in ((ed.selected, QColor(255, 255, 255), 7.0), (ed.live.follow, QColor(80, 230, 230), 10.0)):
            pt = ed.sink_position(sink)
            if pt is None:
                continue
            sx, sy, z = self.screen(pt)
            if z[0] > ZNEAR:
                p.setPen(QPen(col, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(sx[0], sy[0]), rad, rad)

    def _gizmo_center(self):
        return QPointF(self.GIZMO_R + 16, self.height() - self.GIZMO_R - 16)

    def _gizmo_ends(self):
        """[(screen point, depth, (axis, sign), name, colour)], back to front."""
        d, r, u = self.view.basis()
        o = self._gizmo_center()
        ends = []
        for i, name, col in self.GIZMO_AXES:
            for sign in (1, -1):
                v = np.zeros(3)
                v[i] = sign
                e = QPointF(o.x() + self.GIZMO_R * (v @ r), o.y() - self.GIZMO_R * (v @ u))
                ends.append((e, float(v @ d), (i, sign), name if sign > 0 else "-" + name, col))
        ends.sort(key=lambda e: -e[1])
        return ends

    def _gizmo_pick(self, x, y):
        """Axis end under the mouse (front most), None outside of the gizmo."""
        for e, _, axis, _, _ in reversed(self._gizmo_ends()):
            if math.hypot(e.x() - x, e.y() - y) <= 9:
                return axis
        return None

    def _gizmo(self, p):
        o = self._gizmo_center()
        if self._gizmo_hover is not None or (self.underMouse() and self._over_gizmo()):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 25))
            p.drawEllipse(o, self.GIZMO_R + 12, self.GIZMO_R + 12)
        f = p.font()
        f.setPointSizeF(7.5)
        f.setBold(True)
        p.setFont(f)
        for e, depth, axis, name, col in self._gizmo_ends():
            fade = 1.0 if depth < 0.3 else 0.55  # axes pointing away from the viewer are dimmer
            c = QColor(col)
            c.setAlphaF(fade)
            if axis[1] > 0:
                p.setPen(QPen(c, 2))
                p.drawLine(o, e)
            hot = axis == self._gizmo_hover
            p.setPen(QPen(QColor(255, 255, 255) if hot else c, 1.5))
            p.setBrush(c if axis[1] > 0 else QColor(30, 31, 36, int(220 * fade)))
            rad = 8 if hot else 7
            p.drawEllipse(e, rad, rad)
            if axis[1] > 0:
                p.setPen(QColor(20, 20, 20))
                p.drawText(QRectF(e.x() - rad, e.y() - rad, 2 * rad, 2 * rad), Qt.AlignmentFlag.AlignCenter, name)
        f.setBold(False)
        p.setFont(f)

    def _over_gizmo(self):
        pt = self.mapFromGlobal(QtGui.QCursor.pos())
        o = self._gizmo_center()
        return math.hypot(pt.x() - o.x(), pt.y() - o.y()) <= self.GIZMO_R + 12

    # -- events

    def mousePressEvent(self, ev):
        self.setFocus()
        x, y = ev.position().x(), ev.position().y()
        mods = ev.modifiers()
        axis = self._gizmo_pick(x, y) if ev.button() == Qt.MouseButton.LeftButton else None
        if axis is not None:
            # click an axis: look from that side; click the axis already facing you: the other side
            if np.allclose(normalize(self.view.pos - self.view.target)[axis[0]] * axis[1], 1.0, atol=1e-3):
                axis = (axis[0], -axis[1])
            self.look_from(axis)
            return
        if ev.button() == Qt.MouseButton.LeftButton and not (mods & Qt.KeyboardModifier.ShiftModifier):
            h = self._pick(x, y)
            if h is not None:
                self._drag = ("handle", h, x, y, bool(mods & Qt.KeyboardModifier.ControlModifier))
                if h[0] in ("kpos", "ktgt") and h[1] != self.ed.key_index():
                    self.ed.set_time(self.ed.path.keyframes[h[1]].t)
                return
            self._drag = ("orbit", None, x, y, self.view.basis()[2])
            self._press = (x, y, False)
        elif ev.button() == Qt.MouseButton.MiddleButton or (
            ev.button() == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ShiftModifier
        ):
            self._drag = ("pan", None, x, y, None)
        elif ev.button() == Qt.MouseButton.RightButton:
            self._drag = ("orbit", None, x, y, self.view.basis()[2])

    def mouseMoveEvent(self, ev):
        x, y = ev.position().x(), ev.position().y()
        if self._drag is None:
            g = self._gizmo_pick(x, y)
            if g != self._gizmo_hover:
                self._gizmo_hover = g
                self.setCursor(Qt.CursorShape.PointingHandCursor if g else Qt.CursorShape.ArrowCursor)
                self.update()
            h = self._pick(x, y) if g is None else None
            if (h is None) != (self._hover is None) or (h and self._hover and h[:2] != self._hover[:2]):
                self._hover = h
                self.update()
            return
        kind, h, x0, y0, extra = self._drag
        dx, dy = x - x0, y - y0
        self._drag = (kind, h, x, y, extra)
        if self._press and math.hypot(x - self._press[0], y - self._press[1]) > 3:
            self._press = (self._press[0], self._press[1], True)  # a drag, not a click
        if kind == "orbit":
            self._anim_timer.stop()
            self.view.turntable(dx * 0.006, dy * 0.006, world_up=extra)
        elif kind == "pan":
            k = 2 * math.tan(math.radians(self.view.fov) / 2) / self.height()
            self.view.pan(-dx * k, dy * k)
        elif kind == "handle":
            hk, idx, _, _, z = h
            delta = unproject_delta(self.view, z, dx, dy, self.height())
            ed = self.ed
            st = ed.live if hk[0] == "l" or idx == ed.key_index() else ed.path.keyframes[idx].state
            if extra:  # ctrl: move both
                st.pos = st.pos + delta
                st.target = st.target + delta
            elif hk.endswith("pos"):
                st.pos = st.pos + delta
            else:
                st.target = st.target + delta
            if st is ed.live:
                ed.live_changed()
            else:
                ed.path.invalidate()
                ed.path_edited()
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag and self._drag[0] == "handle":
            self.ed.commit()
        if self._press and not self._press[2]:  # click without drag: select a star / core
            self.ed.select_sink(self.ed.pick_sink(self.view, self.width(), self.height(), *self._press[:2]))
        self._press = None
        self._drag = None

    def leaveEvent(self, ev):
        self._gizmo_hover = None
        self.update()
        super().leaveEvent(ev)

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120
        self.view.dolly(0.88**steps)
        self.update()

    def keyPressEvent(self, ev):
        t = ev.text().lower()
        t = KEYMAPS[self.ed.keyboard]["views"].get(t, t)
        if t == "f":
            self.frame_all()
        elif t in ("1", "3", "7"):
            self.axis_view(t)
        else:
            super().keyPressEvent(ev)
