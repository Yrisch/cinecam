# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Camera and movie & look panels, orbit and render dialogs."""

import os

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

RESOLUTIONS = {
    "1280x720 (720p)": (1280, 720),
    "1920x1080 (1080p)": (1920, 1080),
    "2560x1440 (1440p)": (2560, 1440),
    "3840x2160 (4K)": (3840, 2160),
    "1080x1920 (vertical)": (1080, 1920),
    "1080x1080 (square)": (1080, 1080),
    "1600x900 (holostar)" : (1600, 900),
}
PREVIEW_WIDTHS = [160, 240, 320, 480, 640, 960,1080]
CMAPS = ["inferno", "magma", "viridis", "plasma", "gist_heat", "afmhot", "cividis", "bone", "gray", "cubehelix", "twilight_shifted"]


def dspin(lo, hi, dec=3, step=0.1, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setKeyboardTracking(False)
    s.setSuffix(suffix)
    s.setMinimumWidth(70)
    return s


class CameraInspector(QtWidgets.QGroupBox):
    def __init__(self, ed):
        super().__init__("Camera")
        self.ed = ed
        lay = QtWidgets.QFormLayout(self)
        lay.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.header = QtWidgets.QLabel()
        lay.addRow(self.header)
        self.vec = {}
        for name in ("pos", "target", "up"):
            row = QtWidgets.QHBoxLayout()
            boxes = []
            for _ in range(3):
                b = dspin(-1e5, 1e5, 3, 0.5 if name != "up" else 0.05)
                b.valueChanged.connect(self._edited)
                row.addWidget(b)
                boxes.append(b)
            self.vec[name] = boxes
            if name == "up":
                for axis in "XYZ":
                    b = QtWidgets.QToolButton()
                    b.setText("+" + axis)
                    b.setToolTip(f"set the up direction exactly to +{axis} (the camera keeps its view direction)")
                    b.clicked.connect(lambda _=False, a=axis: self._set_up("XYZ".index(a)))
                    row.addWidget(b)
            lay.addRow(name, row)
        self.fov = dspin(1, 150, 2, 1, "°")
        self.dump = dspin(0, 1e5, 2, 1)
        self.dump_label = QtWidgets.QLabel()
        self.clo = dspin(-10, 10, 2, 0.1)
        self.chi = dspin(-10, 10, 2, 0.1)
        self.expo = dspin(-10, 10, 2, 0.25, " dex")
        self.ext = dspin(0, 100, 2, 0.25)
        for w in (self.fov, self.dump, self.clo, self.chi, self.expo, self.ext):
            w.valueChanged.connect(self._edited)
        lay.addRow("fov (vertical)", self.fov)
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(self.dump)
        drow.addWidget(self.dump_label, 1)
        lay.addRow("dump index", drow)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(self.clo)
        crow.addWidget(self.chi)
        lay.addRow("log Σ range [M☉/pc²]", crow)
        lay.addRow("star exposure", self.expo)
        lay.addRow("star extinction ×", self.ext)
        self.ease = QtWidgets.QCheckBox("ease: the camera comes to rest at this key")
        self.ease.setToolTip(
            "zero velocity at this keyframe: the camera slows down into it and starts again from rest\n"
            "(typically the first and last keys of a shot)"
        )
        self.ease.toggled.connect(self._edited)
        lay.addRow("motion", self.ease)

        frow = QtWidgets.QHBoxLayout()
        self.follow_label = QtWidgets.QLabel()
        self.follow_label.setMinimumWidth(150)
        self.follow_label.setTextFormat(Qt.TextFormat.RichText)
        frow.addWidget(self.follow_label, 1)
        self.follow_btns = {}
        for mode, text, tip in (
            ("look", "Look at", "keep the selected star / core at the centre, the camera stays in place"),
            ("ride", "Ride along", "move with the selected star / core, keeping the current camera offset"),
            ("stop", "Stop", "stop following (the camera stays where it is)"),
        ):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, m=mode: ed.follow(m))
            frow.addWidget(b)
            self.follow_btns[mode] = b
        lay.addRow("follow", frow)

        btns = QtWidgets.QHBoxLayout()
        for text, fn in (
            ("◆ Key (K)", ed.key_current),
            ("Delete key", lambda: ed.delete_key(ed.key_index())),
            ("Level horizon", self._level),
            ("Auto Σ range", ed.auto_clim),
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addRow(btns)
        self._updating = False

    def _level(self):
        self.ed.live.level()
        self.ed.live_changed()
        self.ed.commit()

    def _set_up(self, axis):
        st = self.ed.live
        up = np.zeros(3)
        up[axis] = 1.0
        if abs(st.basis()[0] @ up) > 0.999:
            self.ed.statusBar().showMessage("the camera looks along this axis: pick another up direction", 4000)
            return
        st.up = up
        self.ed.live_changed()
        self.ed.commit()

    def refresh(self):
        ed = self.ed
        on = ed.key_index()
        if on is not None:
            self.header.setText(f"<b style='color:#ffd23c'>◆ keyframe {on + 1}</b> at t = {ed.path.keyframes[on].t:.3f} s — edits change the key")
        elif ed.live_dirty:
            self.header.setText("<b style='color:#ff6e5a'>● live camera, not keyed</b> — K to insert a keyframe")
        else:
            self.header.setText("interpolated camera — edit it then press K to key it")
        st = ed.live
        self._updating = True
        for name in ("pos", "target", "up"):
            v = getattr(st, name)  # up as typed: Camera3d orthogonalises it to the view direction
            for b, x in zip(self.vec[name], v):
                b.setValue(float(x))
        self.fov.setValue(st.fov)
        self.dump.setMaximum(max(len(ed.path.dumps) - 1, 0))
        self.dump.setValue(st.dump)
        if ed.path.dumps:
            self.dump_label.setText(os.path.basename(ed.path.dumps[ed.path.dump_index(st)]))
        self.clo.setValue(st.clim[0])
        self.chi.setValue(st.clim[1])
        self.expo.setValue(st.exposure)
        self.ext.setValue(st.extinction)
        self.ease.setChecked(st.ease)
        self._updating = False
        # follow row
        lines = []
        if st.follow is not None:
            mode = "ride along" if st.ride else "look at"
            lines.append(f"<span style='color:#50e6e6'>◎ {ed.sink_label(st.follow)} · {mode}</span>")
        if ed.selected is not None and ed.selected != st.follow:
            lines.append(f"selected: {ed.sink_label(ed.selected)}")
        if not lines:
            lines.append(
                "<span style='color:#888'>click a star / core to select it</span>"
                if ed.tracks is not None
                else f"<span style='color:#888'>{ed.tracks_status}</span>"
            )
        self.follow_label.setText("<br>".join(lines))
        can = ed.tracks is not None and ed.selected is not None
        self.follow_btns["look"].setEnabled(can)
        self.follow_btns["ride"].setEnabled(can)
        self.follow_btns["stop"].setEnabled(st.follow is not None)

    def _edited(self):
        if self._updating:
            return
        # only the edited field: the others would be rounded to the displayed decimals
        st = self.ed.live
        w = self.sender()
        for name, boxes in self.vec.items():
            if w in boxes:
                v = getattr(st, name).copy()
                v[boxes.index(w)] = w.value()
                if name == "up" and np.linalg.norm(v) < 1e-12:
                    return  # transient (0, 0, 0) while typing
                setattr(st, name, v)
        if w is self.fov:
            st.fov = w.value()
        elif w is self.dump:
            self.ed.path.move_dump(st, w.value())  # a following camera moves with its sink
        elif w in (self.clo, self.chi):
            st.clim = (self.clo.value(), self.chi.value())
        elif w is self.expo:
            st.exposure = w.value()
        elif w is self.ext:
            st.extinction = w.value()
        elif w is self.ease:
            st.ease = w.isChecked()
        self.ed.live_changed(from_inspector=True)
        self.ed.commit()


class MovieInspector(QtWidgets.QGroupBox):
    def __init__(self, ed):
        super().__init__("Movie & look")
        self.ed = ed
        lay = QtWidgets.QFormLayout(self)
        self.duration = dspin(0.1, 3600, 2, 1, " s")
        self.fps = dspin(1, 240, 2, 1)
        self.res = QtWidgets.QComboBox()
        self.res.addItems(list(RESOLUTIONS))
        self.prev = QtWidgets.QComboBox()
        self.prev.addItems([str(w) for w in PREVIEW_WIDTHS])
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(CMAPS)
        self.core_r = dspin(0.0, 10, 4, 0.005)
        self.core_r.setSpecialValueText("physical (r_acc)")  # 0 = sink accretion radius
        self.core_r.setToolTip("core disc radius in code length units; 0 = the sinks' accretion radius from the dump")
        self.psf = dspin(0.05, 20, 2, 0.1)
        self.checks = {}
        crow = QtWidgets.QGridLayout()
        for i, (key, text) in enumerate(
            (("show_gas", "gas"), ("show_stars", "stars"), ("show_cores", "cores"), ("time_label", "time label"),
             ("star_psf", "star PSF"), ("star_spikes", "spikes"), ("auto_exposure", "auto exposure"))
        ):
            c = QtWidgets.QCheckBox(text)
            c.toggled.connect(self._edited)
            self.checks[key] = c
            crow.addWidget(c, i // 3, i % 3)
        self.grid = QtWidgets.QCheckBox("thirds grid")
        self.grid.toggled.connect(self._grid)
        n = len(self.checks)
        crow.addWidget(self.grid, n // 3, n % 3)
        self.checks["star_psf"].setToolTip("off: stars drawn as dots, as in the instant proxy view, without the PSF glow")
        self.checks["auto_exposure"].setToolTip(
            "star exposure follows the brightest stars in view (smoothed over 1 s); the key exposure is added to it"
        )
        for w in (self.duration, self.fps, self.core_r, self.psf):
            w.valueChanged.connect(self._edited)
        for w in (self.res, self.prev, self.cmap):
            w.currentIndexChanged.connect(self._edited)
        lay.addRow("duration", self.duration)
        lay.addRow("fps", self.fps)
        lay.addRow("output", self.res)
        lay.addRow("preview width", self.prev)
        lay.addRow("colormap", self.cmap)
        lay.addRow("core radius", self.core_r)
        lay.addRow("star PSF scale", self.psf)
        lay.addRow(crow)
        self._updating = False

    def _grid(self, on):
        self.ed.viewfinder.show_grid = on
        self.ed.viewfinder.update()

    def refresh(self):
        ed = self.ed
        path = ed.path
        self._updating = True
        self.duration.setValue(path.duration)
        self.fps.setValue(path.fps)
        res = tuple(path.resolution)
        names = [k for k, v in RESOLUTIONS.items() if v == res]
        if not names:
            name = f"{res[0]}x{res[1]}"
            RESOLUTIONS[name] = res
            self.res.addItem(name)
            names = [name]
        self.res.setCurrentText(names[0])
        self.prev.setCurrentText(str(ed.preview_width))
        if path.look["cmap"] not in CMAPS:
            self.cmap.addItem(path.look["cmap"])
        self.cmap.setCurrentText(path.look["cmap"])
        self.core_r.setValue(path.look["core_radius"] or 0.0)
        self.psf.setValue(path.look["psf_scale"])
        for k, c in self.checks.items():
            c.setChecked(bool(path.look[k]))
        self._updating = False

    def _edited(self):
        if self._updating:
            return
        ed = self.ed
        path = ed.path
        path.duration = self.duration.value()
        path.fps = self.fps.value()
        path.resolution = RESOLUTIONS[self.res.currentText()]
        ed.preview_width = int(self.prev.currentText())
        path.look["cmap"] = self.cmap.currentText()
        path.look["core_radius"] = self.core_r.value() or None
        path.look["psf_scale"] = self.psf.value()
        for k, c in self.checks.items():
            path.look[k] = c.isChecked()
        path.invalidate()
        ed.path_edited()
        ed.commit()


# ====================================================================== dialogs


class OrbitDialog(QtWidgets.QDialog):
    def __init__(self, parent, n_dumps, cur_dump):
        super().__init__(parent)
        self.setWindowTitle("Insert orbit from the current camera")
        lay = QtWidgets.QFormLayout(self)
        self.angle = dspin(-3600, 3600, 1, 15, "°")
        self.angle.setValue(360)
        self.duration = dspin(0.1, 3600, 2, 1, " s")
        self.duration.setValue(12)
        self.keys = QtWidgets.QSpinBox()
        self.keys.setRange(0, 400)
        self.keys.setSpecialValueText("auto (every 15°)")
        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(["camera up", "world z", "world y", "world x"])
        self.radius = dspin(0.01, 100, 3, 0.05)
        self.radius.setValue(1.0)
        self.dump_end = dspin(0, max(n_dumps - 1, 0), 2, 1)
        self.dump_end.setValue(cur_dump)
        lay.addRow("angle", self.angle)
        lay.addRow("duration", self.duration)
        lay.addRow("keyframes", self.keys)
        lay.addRow("axis", self.axis)
        lay.addRow("end distance ×", self.radius)
        lay.addRow("end dump index", self.dump_end)
        lay.addRow(QtWidgets.QLabel("Keys after the playhead inside the orbit time span are replaced."))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)


class RenderDialog(QtWidgets.QDialog):
    def __init__(self, parent, outdir, n_frames):
        super().__init__(parent)
        self.setWindowTitle("Render movie")
        lay = QtWidgets.QFormLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.out = QtWidgets.QLineEdit(outdir)
        b = QtWidgets.QPushButton("…")
        b.clicked.connect(lambda: self.out.setText(QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory", self.out.text()) or self.out.text()))
        row.addWidget(self.out)
        row.addWidget(b)
        self.scale = dspin(0.05, 4, 2, 0.25)
        self.scale.setValue(1.0)
        self.first = QtWidgets.QSpinBox()
        self.last = QtWidgets.QSpinBox()
        for s in (self.first, self.last):
            s.setRange(0, n_frames - 1)
        self.last.setValue(n_frames - 1)
        self.mp4 = QtWidgets.QCheckBox("encode movie.mp4 with ffmpeg when done")
        self.mp4.setChecked(True)
        self.skip = QtWidgets.QCheckBox("skip frames already on disk")
        self.stereo = QtWidgets.QCheckBox("stereo pair (flat screen)")
        self.stereo.setToolTip(
            "two eyes seen this angle apart from the target, looking parallel, target on the screen plane;\n"
            "files alternate: even numbers left eye, odd numbers right eye (no mp4)"
        )
        self.angle = dspin(0.1, 20, 1, 0.5, "°")
        self.angle.setValue(4.0)
        self.angle.setEnabled(False)
        self.stereo.toggled.connect(self.angle.setEnabled)
        self.stereo.toggled.connect(lambda on: self.mp4.setEnabled(not on))
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self.stereo)
        srow.addWidget(self.angle)
        lay.addRow("output dir", row)
        lay.addRow("resolution scale", self.scale)
        lay.addRow("first frame", self.first)
        lay.addRow("last frame", self.last)
        lay.addRow(self.mp4)
        lay.addRow(self.skip)
        lay.addRow(srow)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
