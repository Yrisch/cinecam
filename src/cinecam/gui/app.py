# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
cinecam editor (main window): build a keyframed 3D camera path through a Phantom
simulation and preview it with shamrock column density renders.

    ./cinecam [dumps ...] [--path path.json] [--sycl 0:0]

Layout
    left    world view: whole simulation, camera path, keyframe cameras (drag handles)
    right   viewfinder: what the camera sees (instant particle proxy, then shamrock render)
    bottom  timeline (keyframes, simulation dump track, playhead) + inspectors

Viewfinder controls (click it first to give it focus)
    LMB drag         orbit around the target          wheel        dolly toward target
    Shift+LMB / MMB  pan                              Ctrl+wheel   zoom (fov)
    RMB / Ctrl+LMB   look around (turn the head)      Alt+wheel    dolly zoom (vertigo)
    W/S A/D R/F      fly forward/back, left/right, up/down     Q/E roll    (Shift = x5)
                     (AZERTY layout, View menu or --keyboard azerty: Z/S Q/D R/F fly, A/E roll)

World view controls
    LMB drag         orbit the view / drag a handle (camera position or target)
    Ctrl+drag handle move the camera and its target together
    Shift+LMB / MMB  pan the view      wheel  zoom      F frame all      1/3/7 front/side/top
    axis gizmo       click an axis end (bottom left) to look from that side, again for the opposite side

Global
    K  key the current camera at the playhead      Del  delete the key at the playhead
    Space play/pause   Left/Right step a frame     PgUp/PgDown previous/next key
    Ctrl+Z / Ctrl+Shift+Z undo/redo   Ctrl+S save   Ctrl+O open

When the playhead sits on a keyframe, every camera edit modifies that keyframe.
Elsewhere, edits move a free "live" camera; press K to turn it into a keyframe.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

from ..constants import AU_PER_PC
from ..data.proxycache import ProxyCache
from ..data.readers import READERS, get_reader
from ..data.sinktracks import cache_path
from ..path.camera import ZNEAR, CamState, project
from ..path.campath import CameraPath, make_orbit
from ..render.compose import save_png, splat_column
from ..render.stars import StarLight, star_exposure
from .inspectors import CameraInspector, MovieInspector, OrbitDialog, RenderDialog
from .keymaps import KEYMAPS, keymap_hint
from .qtutil import np_to_qimage
from .timeline import Timeline
from .viewfinder import Viewfinder
from .workers import Backend, ProxyBridge, TracksBuilder
from .worldview import WorldView

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds the cinecam package


class Editor(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.settings = QtCore.QSettings("cinecam", "cinecam")
        self.keyboard = args.keyboard or self.settings.value("keyboard", "qwerty")
        if self.keyboard not in KEYMAPS:
            self.keyboard = "qwerty"
        self.settings.setValue("keyboard", self.keyboard)
        self.path_file = None
        if args.path and os.path.exists(args.path):
            self.path = CameraPath.load(args.path)
            self.path_file = args.path
        else:
            self.path = CameraPath()
            self.path_file = args.path
        dumps = expand_dumps(args.dumps)
        if dumps:
            self.path.dumps = dumps
        if getattr(args, "format", None):
            self.path.format = args.format

        self.t = 0.0
        self.live = CamState(pos=[0, -40, 15], target=[0, 0, 0], up=[0, 0, 1])
        self.live_dirty = False
        self.sample = None
        self.weight = None
        self.sinks = None
        self.meta = None
        self.sample_id = 0
        self.proxy_path = None  # dump of the sample shown in the instant views
        self.proxy = None
        self._proxy_bridge = ProxyBridge()
        self._proxy_bridge.ready.connect(self.on_proxy_ready)
        self._prefetch_key = None
        self._sink_cache = (None, None)
        self.preview_width = 320
        self.preview_img = None
        self.preview_info = {}
        self.preview_key = None
        self.pending = {}
        self.exports = {}
        self.playing = False
        self._play_stop = None  # time where play_to_next_key stops
        self.movie_proc = None
        self.selected = None  # sink index picked in the views
        self.tracks = None  # SinkTracks of self.tracks_dumps
        self.tracks_dumps = None
        self.tracks_status = "no simulation dumps"
        self._tracks_job = None

        self.starlight = StarLight()  # star colours / auto exposure in the editor
        self.backend = Backend(args.sycl, args.sample)
        self.backend.loaded.connect(self.on_loaded)
        self.backend.frame.connect(self.on_frame)
        self.backend.error.connect(self.on_error)
        self.backend.status.connect(lambda m: self.statusBar().showMessage(m, 5000))

        self._build_ui()
        self.ensure_tracks()
        self.ensure_proxy()
        self.history = [self.path.to_dict()]
        self.hpos = 0
        self.saved = self.history[0]

        self.refine_timer = QTimer(self, singleShot=True, interval=200)
        self.refine_timer.timeout.connect(self.request_refine)
        self.commit_timer = QTimer(self, singleShot=True, interval=600)
        self.commit_timer.timeout.connect(self.commit)
        self.play_timer = QTimer(self, interval=33)
        self.play_timer.timeout.connect(self._play_tick)

        if self.path.keyframes:
            self.set_time(0.0)
        self.refresh_all()
        if self.path.dumps:
            self.request_refine()
        else:
            QTimer.singleShot(200, self.add_dumps)

    # ------------------------------------------------------------------ ui

    def _build_ui(self):
        self.resize(1600, 980)
        self.world = WorldView(self)
        self.viewfinder = Viewfinder(self)
        self.timeline = Timeline(self)
        self.cam_insp = CameraInspector(self)
        self.movie_insp = MovieInspector(self)

        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.world)
        split.addWidget(self.viewfinder)
        split.setSizes([700, 900])

        play_row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.setToolTip("play / pause (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        play_row.addWidget(self.play_btn)
        for text, fn, tip in (
            ("▶┆◆", self.play_to_next_key, "play until the next keyframe (Shift+Space)"),
            ("⏮", lambda: self.set_time(0), "go to the start (Home)"),
            ("◆◀", self.prev_key, "jump to the previous keyframe (PgUp)"),
            ("▶◆", self.next_key, "jump to the next keyframe (PgDown)"),
            ("⏭", lambda: self.set_time(self.path.duration), "go to the end (End)"),
        ):
            b = QtWidgets.QPushButton(text)
            b.setFixedWidth(48 if len(text) > 2 else 40)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            play_row.addWidget(b)
        play_row.addWidget(self.timeline, 1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.cam_insp, 3)
        bottom.addWidget(self.movie_insp, 2)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(split, 1)
        v.addLayout(play_row)
        v.addLayout(bottom)
        self.setCentralWidget(central)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        tb = self.addToolBar("main")
        tb.setMovable(False)
        menu = self.menuBar()
        fm = menu.addMenu("&File")
        em = menu.addMenu("&Edit")
        rm = menu.addMenu("&Render")
        vm = menu.addMenu("&View")
        kb = vm.addMenu("Keyboard layout")
        group = QtGui.QActionGroup(self)
        for name in KEYMAPS:
            a = QtGui.QAction(name.upper(), self, checkable=True)
            a.setChecked(name == self.keyboard)
            a.triggered.connect(lambda _=False, n=name: self.set_keyboard(n))
            group.addAction(a)
            kb.addAction(a)

        def act(menu_, text, fn, shortcut=None, toolbar=False):
            a = QtGui.QAction(text, self)
            a.triggered.connect(fn)
            if shortcut:
                a.setShortcut(QtGui.QKeySequence(shortcut))
            menu_.addAction(a)
            if toolbar:
                tb.addAction(a)
            return a

        act(fm, "New path", self.new_path)
        act(fm, "Open path…", self.open_path, "Ctrl+O", True)
        act(fm, "Save path", self.save_path, "Ctrl+S", True)
        act(fm, "Save path as…", self.save_path_as, "Ctrl+Shift+S")
        fm.addSeparator()
        act(fm, "Set simulation dumps…", self.add_dumps, None, True)
        fm.addSeparator()
        act(fm, "Quit", self.close, "Ctrl+Q")
        tb.addSeparator()
        act(em, "Undo", self.undo, "Ctrl+Z")
        act(em, "Redo", self.redo, "Ctrl+Shift+Z")
        em.addSeparator()
        act(em, "Key camera at playhead", self.key_current, "K", True)
        act(em, "Delete key at playhead", lambda: self.delete_key(self.key_index()), "Del")
        act(em, "Insert orbit…", self.insert_orbit, None, True)
        act(em, "Reset live camera to path", self.reset_live, "Esc")
        tb.addSeparator()
        act(rm, "Render preview now", self.request_refine, "Ctrl+R")
        act(rm, "Export current frame (full res)…", self.export_frame, "Ctrl+E", True)
        act(rm, "Render movie…", self.render_movie, None, True)
        act(rm, "Stop movie render", self.stop_movie)

        for key, fn in (
            ("Space", self.toggle_play),
            ("Shift+Space", self.play_to_next_key),
            ("Left", lambda: self.step(-1)),
            ("Right", lambda: self.step(1)),
            ("PgUp", self.prev_key),
            ("PgDown", self.next_key),
            ("Home", lambda: self.set_time(0)),
            ("End", lambda: self.set_time(self.path.duration)),
        ):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            s.activated.connect(fn)

        pal = QtGui.QPalette()
        for role, col in (
            (QtGui.QPalette.ColorRole.Window, QColor(36, 37, 43)),
            (QtGui.QPalette.ColorRole.WindowText, QColor(220, 220, 225)),
            (QtGui.QPalette.ColorRole.Base, QColor(26, 27, 32)),
            (QtGui.QPalette.ColorRole.AlternateBase, QColor(40, 41, 48)),
            (QtGui.QPalette.ColorRole.Text, QColor(220, 220, 225)),
            (QtGui.QPalette.ColorRole.Button, QColor(48, 49, 56)),
            (QtGui.QPalette.ColorRole.ButtonText, QColor(220, 220, 225)),
            (QtGui.QPalette.ColorRole.Highlight, QColor(70, 120, 200)),
            (QtGui.QPalette.ColorRole.HighlightedText, QColor(255, 255, 255)),
            (QtGui.QPalette.ColorRole.ToolTipBase, QColor(48, 49, 56)),
            (QtGui.QPalette.ColorRole.ToolTipText, QColor(220, 220, 225)),
        ):
            pal.setColor(role, col)
        QtWidgets.QApplication.instance().setPalette(pal)

    def refresh_all(self):
        self.update_proxy()
        self.cam_insp.refresh()
        self.movie_insp.refresh()
        self.world.update()
        self.viewfinder.update()
        self.timeline.update()
        name = os.path.basename(self.path_file) if self.path_file else "untitled"
        mod = "*" if self.history[self.hpos] != self.saved else ""
        self.setWindowTitle(f"cinecam — {name}{mod}")

    # ------------------------------------------------------------------ state

    def aspect(self):
        return self.path.resolution[0] / self.path.resolution[1]

    def preview_size(self, cap=None):
        w = self.preview_width if cap is None else min(self.preview_width, cap)
        return w, max(1, int(round(w / self.aspect())))

    def key_index(self):
        return self.path.key_at(self.t)

    def set_time(self, t):
        self.t = float(np.clip(t, 0, self.path.duration))
        st = self.path.evaluate(self.t)
        if st is not None:
            self.live = st
        self.live_dirty = False
        self.refine_timer.start()
        self.refresh_all()

    def step(self, n):
        self.set_time(self.path.snap(self.t) + n / self.path.fps)

    def prev_key(self):
        ts = [k.t for k in self.path.keyframes if k.t < self.t - 1e-6]
        if ts:
            self.set_time(ts[-1])

    def next_key(self):
        ts = [k.t for k in self.path.keyframes if k.t > self.t + 1e-6]
        if ts:
            self.set_time(ts[0])

    def live_changed(self, from_inspector=False):
        """The live camera was edited (viewfinder, world view handle, inspector)."""
        i = self.key_index()
        if i is not None:
            self.path.keyframes[i].state = self.live.copy()
            self.path.invalidate()
            self.live_dirty = False
        else:
            self.live_dirty = True
        self.refine_timer.start()
        if from_inspector:
            self.world.update()
            self.viewfinder.update()
            self.timeline.update()
        else:
            self.refresh_all()

    def path_edited(self):
        """Keyframes or path settings changed elsewhere than on the live camera."""
        if not self.live_dirty:
            st = self.path.evaluate(self.t)
            if st is not None:
                self.live = st
        self.t = min(self.t, self.path.duration)
        self.refine_timer.start()
        self.refresh_all()

    def set_keyboard(self, name):
        self.keyboard = name
        self.settings.setValue("keyboard", name)
        self.statusBar().showMessage(f"keyboard layout: {name.upper()} ({keymap_hint(name)})", 4000)
        self.viewfinder.update()

    # ------------------------------------------------------------------ sinks: tracks, selection, follow

    def ensure_tracks(self):
        """Give the path the sink trajectories of its dumps (read in the background once)."""
        dumps = tuple(self.path.dumps)
        if not dumps:
            self.path.set_tracks(None)
            self.tracks_status = "no simulation dumps"
            return
        if dumps == self.tracks_dumps:
            self.path.set_tracks(self.tracks)
            return
        if self._tracks_job is not None and self._tracks_job.dumps == dumps:
            return
        self.tracks = None
        self.tracks_dumps = None
        self.path.set_tracks(None)
        self.tracks_status = "reading sink trajectories…"
        job = TracksBuilder(dumps, self.path.format)
        job.progress.connect(self._tracks_progress)
        job.done.connect(self._tracks_done)
        self._tracks_job = job
        job.start()

    def _tracks_progress(self, i, n):
        self.tracks_status = f"reading sink trajectories {i}/{n}…"
        self.statusBar().showMessage(self.tracks_status, 2000)
        self.cam_insp.refresh()

    def _tracks_done(self, dumps, tr):
        job, self._tracks_job = self._tracks_job, None
        if job is not None:
            job.wait()
        if isinstance(tr, Exception):
            self.tracks_status = f"cannot read sinks: {tr}"
            self.statusBar().showMessage(self.tracks_status, 10000)
            self.cam_insp.refresh()
            return
        self.tracks, self.tracks_dumps = tr, dumps
        if tuple(self.path.dumps) == dumps:
            self.path.set_tracks(tr)
            self.backend.send_tracks(cache_path(list(dumps), self.path.format))
            self.statusBar().showMessage(f"sink trajectories: {tr.nsink} sinks in {tr.ndump} dumps", 5000)
            self.path_edited()
        else:
            self.ensure_tracks()

    # ------------------------------------------------------------------ proxy samples (instant views)

    def ensure_proxy(self):
        """Proxy cache with the fast reader of the path's dump format."""
        fmt = self.path.format
        if self.proxy is not None and self.proxy.reader.name == fmt:
            return
        if self.proxy is not None:
            self.proxy.close()
        try:
            reader = get_reader(fmt)
        except ValueError as e:
            self.proxy = None
            self.statusBar().showMessage(f"{e}: proxy views from shamrock only", 10000)
            return
        budget = getattr(self.args, "proxy_cache", 1024)
        self.proxy = ProxyCache(reader, self.args.sample, budget, on_ready=self._proxy_bridge.ready.emit)
        self.proxy_path = None
        self._prefetch_key = None

    def update_proxy(self):
        """Show the cached sample of the current dump, and plan the prefetching."""
        if self.proxy is None or not self.path.dumps:
            return
        cur = self.path.dump_path(self.live)
        if cur != self.proxy_path:
            e = self.proxy.get(cur)
            if e is not None:
                self._use_sample(e["pos"], e["weight"], cur)
        self._prefetch()

    def _prefetch(self):
        """Dumps to read: the current one, then the ones the movie visits nearest in time (ahead first)."""
        di = self.path.dump_index(self.live)
        key = (di, int(round(self.t * self.path.fps)), self.playing, tuple((k.t, k.state.dump) for k in self.path.keyframes), len(self.path.dumps))
        if key == self._prefetch_key:
            return
        self._prefetch_key = key
        n = len(self.path.dumps)
        order = [di] + [i for d in range(1, 4) for i in (di + d, di - d) if 0 <= i < n]  # scrubbing the dump
        fd = self.path.frame_dumps()
        if len(fd):
            dt = np.arange(len(fd)) / self.path.fps - self.t
            urgency = np.where(dt >= 0, dt, -3 * dt)  # what comes next first
            order += fd[np.argsort(urgency, kind="stable")].tolist()
        self.proxy.want([self.path.dumps[i] for i in order])

    def _use_sample(self, pos, weight, path):
        first = self.sample is None
        self.sample, self.weight, self.proxy_path = pos, weight, path
        self.sample_id += 1
        if first:
            self.world.frame_all()
            if not self.path.keyframes:
                c = np.median(self.sample, axis=0)
                R = np.percentile(np.linalg.norm(self.sample - c, axis=1), 90)
                self.live = CamState(pos=c + np.array([0, -2.6, 0.9]) * R, target=c, up=[0, 0, 1])
                self.live_dirty = True
                self.refine_timer.start()
                self.refresh_all()

    def on_proxy_ready(self, path, err):
        if err:
            self.statusBar().showMessage(f"proxy: cannot read {os.path.basename(path)}: {err}", 10000)
            return
        if path == self.path.dump_path(self.live) and path != self.proxy_path:
            e = self.proxy.get(path)
            if e is not None:
                self._use_sample(e["pos"], e["weight"], path)
                self.world.update()
                self.viewfinder.update()
        self.timeline.update()

    def proxy_status(self):
        """HUD text: is the shown gas the current dump's?"""
        if self.proxy is None or not self.path.dumps:
            return ""
        cur = self.path.dump_path(self.live)
        if cur == self.proxy_path:
            return ""
        return f"gas of {os.path.basename(self.proxy_path) if self.proxy_path else '—'}, reading {os.path.basename(cur)}…"

    def display_sinks(self):
        """Stars and cores at the current (fractional) dump for the instant views."""
        tr = self.path.tracks
        if tr is None:
            return self.sinks
        d = round(float(self.live.dump), 6)
        if self._sink_cache[0] == (id(tr), d):
            return self._sink_cache[1]
        sk = tr.sinks_at(d)
        self._sink_cache = ((id(tr), d), sk)
        return sk

    def pc_per_unit(self):
        tr = self.path.tracks
        if tr is not None:
            return tr.pc_per_unit
        return self.meta["pc_per_unit"] if self.meta else 1.0

    def sink_points(self, dump):
        """(ids, positions) of the sinks alive at this (fractional) dump."""
        tr = self.path.tracks
        if tr is not None:
            ids = np.flatnonzero(tr.alive[int(np.clip(round(dump), 0, tr.ndump - 1))])
            return ids, tr.positions_at(ids, dump)
        if self.sinks is not None and "star_id" in self.sinks:
            return (
                np.concatenate([self.sinks["star_id"], self.sinks["core_id"]]),
                np.concatenate([self.sinks["star_pos"], self.sinks["core_pos"]]),
            )
        return np.zeros(0, int), np.zeros((0, 3))

    def pick_sink(self, state, w, h, x, y, radius=8.0):
        """Sink drawn under the screen point (x, y) of a view of the camera state, or None."""
        ids, pts = self.sink_points(self.live.dump)
        if not len(ids):
            return None
        sx, sy, z = project(state, pts, w, h)
        d = np.hypot(sx - x, sy - y)
        d[z <= ZNEAR] = np.inf
        i = int(np.argmin(d))
        return int(ids[i]) if d[i] <= radius else None

    def sink_position(self, sink):
        if sink is None:
            return None
        tr = self.path.tracks
        if tr is not None:
            return tr.position(sink, self.live.dump)
        if self.sinks is not None and "star_id" in self.sinks:
            for kind in ("star", "core"):
                hit = np.flatnonzero(self.sinks[kind + "_id"] == sink)
                if len(hit):
                    return self.sinks[kind + "_pos"][hit[0]]
        return None

    def sink_label(self, sink):
        tr = self.path.tracks
        if tr is not None and tr.has(sink):
            return tr.describe(sink, self.live.dump)
        return f"sink #{sink}"

    def select_sink(self, sink):
        self.selected = sink
        if sink is not None:
            self.statusBar().showMessage(f"selected {self.sink_label(sink)}: Look at / Ride along in the camera panel", 5000)
        self.refresh_all()

    def follow(self, mode):
        """Make the live camera look at / ride along the selected sink, or stop following."""
        live = self.live
        if mode == "stop":
            live.follow, live.ride = None, False
        else:
            a = self.path.anchor(self.selected, live.dump)
            if a is None:
                return
            offset = live.pos - live.target
            live.follow, live.ride = int(self.selected), mode == "ride"
            live.target = a.copy()
            if live.ride:
                live.pos = a + offset  # same view offset, now relative to the sink
        self.live_changed()
        self.commit()

    def reset_live(self):
        self.set_time(self.t)

    def key_current(self):
        self.path.set_key(self.t, self.live)
        self.t = self.path.snap(self.t)
        self.live_dirty = False
        self.commit()
        self.refresh_all()
        self.statusBar().showMessage(f"keyframe set at t = {self.t:.3f} s", 3000)

    def delete_key(self, i):
        if i is None:
            return
        self.path.remove_key(i)
        self.commit()
        self.path_edited()

    def auto_clim(self):
        if self.sample is None:
            return
        nx, ny = self.preview_size(cap=320)
        sig = splat_column(self.sample, self.weight, self.live, nx, ny)
        nz = sig[sig > 0]
        if nz.size:
            lo, hi = np.log10(np.percentile(nz, [20, 99.7]))
            self.live.clim = (round(float(lo), 2), round(float(hi), 2))
            self.live_changed()
            self.commit()

    # ------------------------------------------------------------------ undo

    def commit(self):
        self.commit_timer.stop()
        snap = self.path.to_dict()
        if snap != self.history[self.hpos]:
            del self.history[self.hpos + 1 :]
            self.history.append(snap)
            self.hpos += 1
            self.refresh_all()

    def commit_soon(self):
        self.commit_timer.start()

    def _restore(self):
        self.path = CameraPath.from_dict(self.history[self.hpos])
        self.ensure_tracks()
        self.ensure_proxy()
        self.live_dirty = False
        self.set_time(self.t)

    def undo(self):
        self.commit()
        if self.hpos > 0:
            self.hpos -= 1
            self._restore()

    def redo(self):
        if self.hpos < len(self.history) - 1:
            self.hpos += 1
            self._restore()

    # ------------------------------------------------------------------ playback

    def play_to_next_key(self):
        """Play from the playhead and stop on the next keyframe."""
        ts = [k.t for k in self.path.keyframes if k.t > self.t + 0.5 / self.path.fps]
        if not ts:
            self.statusBar().showMessage("no keyframe after the playhead", 3000)
            return
        self._play_stop = ts[0]
        if not self.playing:
            self.toggle_play()

    def toggle_play(self):
        if self.playing:
            self._play_stop = None
        self.playing = not self.playing
        self.play_btn.setText("⏸" if self.playing else "▶")
        if self.playing:
            if self.t >= self.path.duration - 1e-6:
                self.set_time(0)
            self._play_t0 = time.perf_counter() - self.t
            self.play_timer.start()
        else:
            self.play_timer.stop()
            self.set_time(self.path.snap(self.t))

    def _play_tick(self):
        t = time.perf_counter() - self._play_t0
        stop = getattr(self, "_play_stop", None)
        if stop is not None and t >= stop:  # reached the keyframe: land exactly on it
            self._play_stop = None
            self.playing = False
            self.play_btn.setText("▶")
            self.play_timer.stop()
            self.set_time(stop)
            return
        if t > self.path.duration:
            t = 0.0
            self._play_t0 = time.perf_counter()
        self.set_time(t)

    # ------------------------------------------------------------------ rendering

    def look_key(self):
        st = self.live
        return (json.dumps(st.to_dict()), json.dumps(self.path.look, sort_keys=True), self.preview_size(), self.path.dump_path(st))

    def live_exposure(self, nx, ny):
        """Star exposure of the live camera: its own, plus the automatic part with auto exposure."""
        if not self.path.look.get("auto_exposure", False) or self.path.tracks is None:
            return self.live.exposure
        if not self.live_dirty and self.path.keyframes:  # on the path: smoothed over time, as in the movie
            st = self.path.evaluate(self.t)
            return star_exposure(self.path, self.t, self.starlight, nx, ny) - st.exposure + self.live.exposure
        sk = self.display_sinks()
        lum, _ = self.starlight.luminosities(sk["star_mass"])
        e = self.starlight.exposure_for(self.live, sk["star_pos"], lum, nx, ny, self.path.tracks.pc_per_unit)
        return self.live.exposure + (e or 0.0)

    def preview_valid(self):
        return self.preview_img is not None and self.preview_key == self.look_key()

    def request_refine(self):
        if self.playing or not self.path.dumps:
            return
        key = self.look_key()
        if key == self.preview_key or key in self.pending.values():
            return
        nx, ny = self.preview_size()
        rid = self.backend.request(
            self.live, self.path.dump_path(self.live), nx, ny, self.path.look, exposure=self.live_exposure(nx, ny)
        )
        self.pending = {rid: key}
        self.viewfinder.update()

    def on_loaded(self, msg):
        self.sinks = msg["sinks"]
        self.meta = msg["meta"]
        cur = self.path.dump_path(self.live)
        if self.proxy is None or (self.proxy_path != cur and cur in self.proxy.errors):
            self._use_sample(msg["sample"], msg["weight"], msg["meta"]["path"])  # no fast reader for it
        racc = self.sinks.get("core_racc", [])
        if len(racc):
            au = np.median(racc) * self.meta["pc_per_unit"] * AU_PER_PC
            self.movie_insp.core_r.setSpecialValueText(f"physical ({au:.0f} au)")
        m = self.meta
        self.statusBar().showMessage(
            f"loaded {os.path.basename(m['path'])}: {m['npart']} particles, {len(self.sinks['star_pos'])} stars, "
            f"{len(self.sinks['core_pos'])} cores, t = {m['time_myr']:.2f} Myr ({m['load_time']:.1f} s)",
            8000,
        )
        self.refresh_all()

    def on_frame(self, msg):
        if msg.get("keep"):
            fname = self.exports.pop(msg["id"], None)
            if fname:
                save_png(msg["rgb"], fname)
                self.statusBar().showMessage(f"saved {fname} ({msg['info'].get('total', 0):.1f} s)", 8000)
            return
        key = self.pending.pop(msg["id"], None)
        if key is None:
            return
        self.preview_img = np_to_qimage(msg["rgb"])
        self.preview_info = msg["info"]
        self.preview_key = key
        self.viewfinder.update()
        if key != self.look_key():
            self.refine_timer.start()

    def on_error(self, msg):
        self.pending.pop(msg.get("id"), None)
        print(msg["msg"], file=sys.stderr)
        self.statusBar().showMessage("render error: " + msg["msg"].strip().splitlines()[-1], 15000)

    def export_frame(self):
        if not self.path.dumps:
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export frame", f"frame_{int(round(self.t * self.path.fps)):05d}.png", "PNG (*.png)")
        if not fname:
            return
        nx, ny = self.path.resolution
        rid = self.backend.request(
            self.live, self.path.dump_path(self.live), nx, ny, self.path.look, keep=True, exposure=self.live_exposure(nx, ny)
        )
        self.exports[rid] = fname
        self.statusBar().showMessage(f"rendering {nx}×{ny} frame…")

    # ------------------------------------------------------------------ files

    def new_path(self):
        dumps = self.path.dumps
        self.path = CameraPath(dumps=dumps)
        self.ensure_tracks()
        self.path_file = None
        self.live_dirty = True
        self.commit()
        self.set_time(0)

    def open_path(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open camera path", "", "Camera path (*.json)")
        if not fname:
            return
        self.path = CameraPath.load(fname)
        self.ensure_tracks()
        self.ensure_proxy()
        self.path_file = fname
        self.history = [self.path.to_dict()]
        self.hpos = 0
        self.saved = self.history[0]
        self.set_time(0)

    def save_path(self):
        if not self.path_file:
            return self.save_path_as()
        self.commit()
        self.path.save(self.path_file)
        self.saved = self.history[self.hpos]
        self.refresh_all()
        self.statusBar().showMessage(f"saved {self.path_file}", 4000)
        return True

    def save_path_as(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save camera path", self.path_file or "campath.json", "Camera path (*.json)")
        if not fname:
            return False
        self.path_file = fname
        return self.save_path()

    def add_dumps(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Phantom dumps (in time order)", os.getcwd())
        if not files:
            return
        self.path.dumps = sorted(os.path.abspath(f) for f in files)
        self.ensure_tracks()
        self.ensure_proxy()
        self.commit()
        self.path_edited()
        self.request_refine()

    # ------------------------------------------------------------------ tools

    def insert_orbit(self):
        cur = self.live.dump
        dlg = OrbitDialog(self, len(self.path.dumps), cur)
        if not dlg.exec():
            return
        axis = {"camera up": None, "world z": [0, 0, 1], "world y": [0, 1, 0], "world x": [1, 0, 0]}[dlg.axis.currentText()]
        t0 = self.path.snap(self.t)
        dur = dlg.duration.value()
        keys = make_orbit(
            self.live,
            t0,
            dur,
            dlg.angle.value(),
            dlg.keys.value() or None,
            axis,
            dlg.radius.value(),
            dlg.dump_end.value() if abs(dlg.dump_end.value() - cur) > 1e-9 else None,
            tracks=self.path.tracks,
        )
        self.path.keyframes = [k for k in self.path.keyframes if not (t0 - 1e-9 <= k.t <= t0 + dur + 1e-9)]
        for k in keys:
            k.t = self.path.snap(k.t)
            self.path.keyframes.append(k)
        self.path.duration = max(self.path.duration, t0 + dur)
        self.path.sort()
        self.live_dirty = False
        self.commit()
        self.set_time(t0)

    def render_movie(self):
        if self.movie_proc is not None:
            QtWidgets.QMessageBox.information(self, "cinecam", "A movie render is already running.")
            return
        if not self.path.keyframes:
            QtWidgets.QMessageBox.warning(self, "cinecam", "Add at least one keyframe first.")
            return
        if not self.save_path():
            return
        base = os.path.splitext(self.path_file)[0]
        dlg = RenderDialog(self, base + "_frames", self.path.n_frames)
        if not dlg.exec():
            return
        cmd = [
            "-m",
            "cinecam.render_movie",
            self.path_file,
            "--out",
            dlg.out.text(),
            "--frames",
            f"{dlg.first.value()}:{dlg.last.value() + 1}",
            "--scale",
            str(dlg.scale.value()),
            "--sycl",
            self.args.sycl,
        ]
        if dlg.mp4.isChecked():
            cmd += ["--mp4", os.path.join(dlg.out.text(), "movie.mp4")]
        if dlg.skip.isChecked():
            cmd += ["--skip-existing"]
        self.movie_proc = QtCore.QProcess(self)
        self.movie_proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.movie_proc.readyReadStandardOutput.connect(self._movie_output)
        self.movie_proc.finished.connect(self._movie_done)
        env = QtCore.QProcessEnvironment.systemEnvironment()  # the batch renderer imports the cinecam package
        env.insert("PYTHONPATH", os.pathsep.join(p for p in (SRC, env.value("PYTHONPATH")) if p))
        self.movie_proc.setProcessEnvironment(env)
        self.progress.setRange(0, dlg.last.value() + 1 - dlg.first.value())
        self.progress.setValue(0)
        self.progress.show()
        self.movie_proc.start(sys.executable, cmd)
        self.statusBar().showMessage("movie render started: " + " ".join(cmd), 8000)

    def _movie_output(self):
        data = bytes(self.movie_proc.readAllStandardOutput()).decode(errors="replace")
        for line in data.splitlines():
            if line.startswith("PROGRESS"):
                done, tot = line.split()[1].split("/")
                self.progress.setValue(int(done))
                self.statusBar().showMessage(line[9:], 0)
            elif line.startswith(("ERROR", "Traceback", "DONE", "ffmpeg")):
                print(line)
                self.statusBar().showMessage(line, 15000)

    def _movie_done(self, code, status):
        self.progress.hide()
        self.statusBar().showMessage(f"movie render finished (exit code {code})", 0)
        self.movie_proc = None

    def stop_movie(self):
        if self.movie_proc is not None:
            self.movie_proc.kill()

    def closeEvent(self, ev):
        self.commit()
        if self.proxy is not None:
            self.proxy.close()
        if self.history[self.hpos] != self.saved:
            r = QtWidgets.QMessageBox.question(
                self,
                "cinecam",
                "Save the camera path before quitting?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if r == QtWidgets.QMessageBox.StandardButton.Cancel:
                ev.ignore()
                return
            if r == QtWidgets.QMessageBox.StandardButton.Save and not self.save_path():
                ev.ignore()
                return
        self.backend.close()
        if self.movie_proc is not None:
            self.statusBar().showMessage("movie render keeps running in the background")
            self.movie_proc.finished.disconnect()
            self.movie_proc.setParent(None)
        ev.accept()


def expand_dumps(items):
    out = []
    for it in items or []:
        g = sorted(glob.glob(it))
        out += g if g else [it]
    return [os.path.abspath(f) for f in out]


def main():
    ap = argparse.ArgumentParser(prog="cinecam", description="cinecam: camera path editor for Phantom dumps rendered with shamrock")
    ap.add_argument("dumps", nargs="*", help="Phantom dumps in time order (globs accepted, e.g. 'esc_00*')")
    ap.add_argument("--path", help="camera path JSON to open (created on save if missing)")
    ap.add_argument("--sycl", default="0:0", help="shamrock sycl config (default 0:0)")
    ap.add_argument("--sample", type=int, default=400_000, help="particles kept for the instant proxy views")
    ap.add_argument(
        "--format", choices=list(READERS), default=None, help="dump format for the fast readers (default: the path's, phantom)"
    )
    ap.add_argument(
        "--proxy-cache", type=int, default=1024, help="memory for the proxy samples of the dumps, MB (default 1024)"
    )
    ap.add_argument(
        "--keyboard", choices=list(KEYMAPS), default=None, help="fly keys layout (default: last used, else qwerty)"
    )
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    ed = Editor(args)
    ed.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
