# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Viewfinder: what the camera sees (instant particle proxy, then the shamrock render)."""

import json
import math
import os

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen

from ..path.camera import ZNEAR, normalize, project
from ..render.compose import compose, splat_column
from .keymaps import KEYMAPS, key_char, keymap_hint
from .qtutil import draw_dots, fit_rect, np_to_qimage


class Viewfinder(QtWidgets.QWidget):

    def __init__(self, ed):
        super().__init__()
        self.ed = ed
        self.setMinimumSize(320, 180)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._drag = None
        self._press = None
        self._proxy = (None, None)
        self.show_grid = False

    def proxy_image(self):
        ed = self.ed
        if ed.sample is None:
            return None
        nx, ny = ed.preview_size(cap=480)
        st = ed.live
        key = (json.dumps(st.to_dict()), nx, ny, ed.sample_id, json.dumps(ed.path.look, sort_keys=True), ed.playing)
        if self._proxy[0] != key:
            k = 2 if ed.playing else 1  # half of the sample while playing, to keep up with the frame rate
            sig = splat_column(ed.sample[::k], ed.weight * k, st, nx, ny)
            rgb = compose(sig, st, ed.path.look, ed.display_sinks(), None, None)
            self._proxy = (key, np_to_qimage(rgb))
        return self._proxy[1]

    def paintEvent(self, ev):
        ed = self.ed
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        rect = fit_rect(self.width(), self.height(), ed.aspect())
        p.fillRect(rect, QColor(0, 0, 0))

        refined = ed.preview_valid()
        if refined:
            p.drawImage(rect, ed.preview_img)
            mode = f"SHAMROCK {ed.preview_img.width()}×{ed.preview_img.height()} · {ed.preview_info.get('total', 0):.2f} s"
        else:
            img = self.proxy_image()
            if img is not None:
                p.drawImage(rect, img)
            sinks = ed.display_sinks()
            if ed.path.look.get("show_stars", True) and sinks is not None and len(sinks["star_pos"]):
                self._proxy_stars(p, rect)
            mode = "PROXY" + ("  ·  rendering…" if ed.backend and ed.backend.busy else "")
            if ed.proxy_status():
                mode += "  ·  " + ed.proxy_status()

        # selected / followed sink (editor only, never in the renders)
        for sink, col, rad in ((ed.selected, QColor(255, 255, 255, 200), 8.0), (ed.live.follow, QColor(80, 230, 230), 11.0)):
            pt = ed.sink_position(sink)
            if pt is None:
                continue
            sx, sy, z = project(ed.live, pt[None], rect.width(), rect.height())
            if z[0] > ZNEAR:
                p.setPen(QPen(col, 1.5))
                p.drawEllipse(QPointF(rect.left() + sx[0], rect.top() + sy[0]), rad, rad)

        if self.show_grid:
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for f in (1 / 3, 2 / 3):
                x = rect.left() + rect.width() * f
                y = rect.top() + rect.height() * f
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.PenStyle.DashLine))
            p.drawRect(rect.adjusted(rect.width() * 0.05, rect.height() * 0.05, -rect.width() * 0.05, -rect.height() * 0.05))

        # HUD
        p.setPen(QColor(220, 220, 220))
        f = p.font()
        f.setPointSizeF(9)
        p.setFont(f)
        p.drawText(QPointF(rect.left() + 8, rect.top() + 16), mode)
        if self.hasFocus():
            p.setPen(QColor(160, 160, 170))
            p.drawText(QPointF(rect.left() + 8, rect.top() + 32), keymap_hint(ed.keyboard))
            p.setPen(QColor(220, 220, 220))
        meta = ed.meta
        frame = int(round(ed.t * ed.path.fps))
        info = f"t = {ed.t:6.2f} s  ·  frame {frame}"
        if ed.path.dumps:
            di = ed.path.dump_index(ed.live)
            info += f"  ·  dump {di} {os.path.basename(ed.path.dumps[di])}"
        if meta:
            info += f"  ·  {meta['time_myr']:.2f} Myr"
        p.drawText(QPointF(rect.left() + 8, rect.bottom() - 8), info)
        on = ed.key_index()
        if on is not None:
            p.setPen(QColor(255, 210, 60))
            txt = f"◆ editing keyframe {on + 1}"
        elif ed.live_dirty:
            p.setPen(QColor(255, 110, 90))
            txt = "● camera not keyed — press K"
        else:
            p.setPen(QColor(160, 160, 170))
            txt = "interpolated"
        p.drawText(QPointF(rect.right() - p.fontMetrics().horizontalAdvance(txt) - 8, rect.top() + 16), txt)
        if ed.live.follow is not None:
            p.setPen(QColor(80, 230, 230))
            txt = f"◎ {ed.sink_label(ed.live.follow)} · {'ride along' if ed.live.ride else 'look at'}"
            p.drawText(QPointF(rect.right() - p.fontMetrics().horizontalAdvance(txt) - 8, rect.top() + 32), txt)

    def _proxy_stars(self, p, rect):
        ed = self.ed
        sinks = ed.display_sinks()
        pos = sinks["star_pos"]
        sx, sy, z = project(ed.live, pos, rect.width(), rect.height())
        ok = (z > ZNEAR) & (sx >= 0) & (sx < rect.width()) & (sy >= 0) & (sy < rect.height())
        if not ok.any():
            return
        # same photometry and stretch as the renders, drawn as dots: a few brightness levels and
        # temperature bins, each in one call
        sl = ed.starlight
        lum, T = sl.luminosities(sinks["star_mass"][ok])
        F = sl.fluxes(ed.live, pos[ok], lum, None, ed.pc_per_unit())
        nx = rect.width()
        I = F.mean(axis=1) * sl._moffat(0.0, sl.CORE_FWHM, sl.CORE_BETA) * 10.0 ** ed.live_exposure(nx, rect.height())
        b = np.clip(np.arcsinh(I / sl.I_SOFT) / np.arcsinh(sl.I_WHITE / sl.I_SOFT), 0.0, 1.0)
        level = np.minimum((b * 6).astype(int), 5)
        tbin = np.clip(np.digitize(np.log10(T), np.linspace(3.4, 4.5, 7)), 0, 7)
        cols = sl.colours(10 ** np.linspace(3.3, 4.6, 8))
        X, Y = rect.left() + sx[ok], rect.top() + sy[ok]
        for lv in range(6):
            for tb in range(8):
                m = (level == lv) & (tbin == tb) & (b > 0.5 / 255)
                if m.any():
                    v = (lv + 0.5) / 6
                    c = cols[tb]
                    draw_dots(p, X[m], Y[m], 0.7 + 2.5 * v, QColor.fromRgbF(*np.clip(c ** (1 / 2.2), 0, 1), 0.35 + 0.65 * v))

    # -- events

    def mousePressEvent(self, ev):
        self.setFocus()
        mods = ev.modifiers()
        b = ev.button()
        x, y = ev.position().x(), ev.position().y()
        if b == Qt.MouseButton.MiddleButton or (b == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ShiftModifier):
            self._drag = ("pan", x, y, None)
        elif b == Qt.MouseButton.RightButton or (b == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ControlModifier):
            self._drag = ("look", x, y, None)
        elif b == Qt.MouseButton.LeftButton:
            # orbit around the camera up as set by the user (+Z: classic level turntable)
            self._drag = ("orbit", x, y, normalize(self.ed.live.up))
            self._press = (x, y, False)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        kind, x0, y0, extra = self._drag
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - x0, y - y0
        if self._press and not self._press[2]:
            if math.hypot(x - self._press[0], y - self._press[1]) <= 3:
                return  # maybe a click: do not move the camera yet
            dx, dy = x - self._press[0], y - self._press[1]
            self._press = (self._press[0], self._press[1], True)
        self._drag = (kind, x, y, extra)
        live = self.ed.live
        H = fit_rect(self.width(), self.height(), self.ed.aspect()).height()
        if kind == "orbit":
            live.turntable(dx * 0.005, dy * 0.005, world_up=extra)
        elif kind == "pan":
            k = 2 * math.tan(math.radians(live.fov) / 2) / H
            live.pan(-dx * k, dy * k)
        elif kind == "look":
            k = math.radians(live.fov) / H
            live.look(dx * k, -dy * k)
        self.ed.live_changed()

    def mouseReleaseEvent(self, ev):
        if self._press and not self._press[2]:  # click without drag: select a star / core
            rect = fit_rect(self.width(), self.height(), self.ed.aspect())
            x, y = self._press[0] - rect.left(), self._press[1] - rect.top()
            self.ed.select_sink(self.ed.pick_sink(self.ed.live, rect.width(), rect.height(), x, y))
        elif self._drag is not None:
            self.ed.commit()
        self._press = None
        self._drag = None

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120
        mods = ev.modifiers()
        live = self.ed.live
        if mods & Qt.KeyboardModifier.ControlModifier:
            live.fov = float(np.clip(live.fov * 0.93**steps, 1.0, 150.0))
        elif mods & Qt.KeyboardModifier.AltModifier:
            # dolly zoom: keep the target plane the same size on screen
            steps = (ev.angleDelta().x() or ev.angleDelta().y()) / 120
            new_fov = float(np.clip(live.fov * 0.93**steps, 2.0, 140.0))
            w = live.distance * math.tan(math.radians(live.fov) / 2)
            live.fov = new_fov
            live.dolly(w / math.tan(math.radians(new_fov) / 2) / live.distance)
        else:
            live.dolly(0.9**steps)
        self.ed.live_changed()
        self.ed.commit_soon()

    def focusInEvent(self, ev):
        self.update()
        super().focusInEvent(ev)

    def focusOutEvent(self, ev):
        self.update()
        super().focusOutEvent(ev)

    def keyPressEvent(self, ev):
        c = key_char(ev)
        km = KEYMAPS[self.ed.keyboard]
        fast = 5.0 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        live = self.ed.live
        if c in km["fly"]:
            f, r, u = km["fly"][c]
            s = 0.02 * fast
            live.fly(f * s, r * s, u * s)
        elif c in km["roll"]:
            live.roll(km["roll"][c] * math.radians(2 * fast))
        else:
            return super().keyPressEvent(ev)
        self.ed.live_changed()

    def keyReleaseEvent(self, ev):
        c = key_char(ev)
        km = KEYMAPS[self.ed.keyboard]
        if not ev.isAutoRepeat() and (c in km["fly"] or c in km["roll"]):
            self.ed.commit()
        super().keyReleaseEvent(ev)
