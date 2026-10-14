"""
Editor tests, offscreen. The shamrock render process is replaced by a stub, so they need
neither shamrock nor a GPU: they cover the GUI logic on the synthetic dumps.
"""

import argparse
import os
import time

import numpy as np
import pytest
from conftest import sink_pos

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
from PyQt6 import QtCore, QtGui  # noqa: E402
from PyQt6.QtCore import QPointF, Qt, pyqtSignal  # noqa: E402

pytestmark = pytest.mark.gui


class StubBackend(QtCore.QObject):
    """Stands for gui.workers.Backend: records the render requests, never answers."""

    loaded = pyqtSignal(object)
    frame = pyqtSignal(object)
    error = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, sycl, sample_n):
        super().__init__()
        self.requests = []
        self.busy = False
        self.tracks_path = None

    def request(self, state, dump, nx, ny, look, keep=False, exposure=None):
        self.requests.append(dict(state=state.to_dict(), dump=dump, nx=nx, ny=ny, look=dict(look), keep=keep, exposure=exposure))
        return len(self.requests)

    def send_tracks(self, path):
        self.tracks_path = path

    def close(self):
        pass


@pytest.fixture(scope="module")
def qapp(tmp_path_factory):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("config"))  # QSettings of the tests only
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait(app, cond, tmax=20.0):
    t = time.perf_counter()
    while not cond() and time.perf_counter() - t < tmax:
        app.processEvents()
        time.sleep(0.002)
    return cond()


@pytest.fixture
def editor(qapp, sim, monkeypatch):
    from cinecam.gui import app as gui_app

    monkeypatch.setattr(gui_app, "Backend", StubBackend)
    args = argparse.Namespace(dumps=sim.dumps, path=None, sycl="0:0", sample=1000, keyboard="qwerty", format=None,
                              proxy_cache=100)  # fmt: skip
    ed = gui_app.Editor(args)
    ed.resize(1200, 800)
    ed.show()
    assert wait(qapp, lambda: ed.path.tracks is not None and ed.sample is not None)
    yield ed
    ed.commit()
    ed.saved = ed.history[ed.hpos]  # no "save?" dialog
    ed.close()
    ed.deleteLater()
    qapp.processEvents()


def test_startup_reads_tracks_and_proxy(editor, sim):
    assert editor.path.tracks.nsink == 4
    assert editor.proxy_path == sim.dumps[0]
    assert len(editor.sample) == 1000
    assert editor.live_dirty and not editor.path.keyframes  # a first camera framing the gas
    assert editor.backend.tracks_path is not None  # the render process gets the tracks too
    assert editor.backend.requests  # a first preview was asked


def test_key_delete_undo_redo(editor):
    editor.key_current()
    editor.set_time(1.0)
    editor.live.dolly(0.5)
    editor.live_changed()
    editor.key_current()
    assert [k.t for k in editor.path.keyframes] == [0.0, 1.0]
    editor.delete_key(editor.key_index())
    assert len(editor.path.keyframes) == 1
    editor.undo()
    assert len(editor.path.keyframes) == 2
    editor.redo()
    assert len(editor.path.keyframes) == 1


def test_editing_on_a_key_changes_the_key(editor):
    editor.key_current()
    editor.live.fov = 30.0
    editor.live_changed()
    assert editor.path.keyframes[0].state.fov == 30.0 and not editor.live_dirty


def test_play_to_next_key_stops_on_it(qapp, editor):
    editor.key_current()
    editor.set_time(0.3)
    editor.live.dolly(0.8)
    editor.key_current()
    editor.set_time(0.0)
    editor.play_to_next_key()
    assert editor.playing
    assert wait(qapp, lambda: not editor.playing, 5)
    assert editor.t == 0.3 and editor.key_index() == 1


def test_select_and_follow(editor):
    editor.live.dump = 1.0
    editor.select_sink(0)
    assert "star #0" in editor.sink_label(0)
    editor.follow("look")
    assert editor.live.follow == 0 and not editor.live.ride
    np.testing.assert_allclose(editor.live.target, sink_pos(0, 1.0), atol=1e-6)
    off = editor.live.pos - editor.live.target
    editor.follow("ride")
    assert editor.live.ride
    np.testing.assert_allclose(editor.live.pos - editor.live.target, off, atol=1e-9)
    editor.follow("stop")
    assert editor.live.follow is None


def test_click_selects_a_sink(qapp, editor):
    vf = editor.viewfinder
    st = editor.live
    st.target = sink_pos(2, st.dump)
    editor.live_changed()
    qapp.processEvents()
    from cinecam.gui.qtutil import fit_rect

    r = fit_rect(vf.width(), vf.height(), editor.aspect())
    c = QPointF(r.center())
    for kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease):
        ev = QtGui.QMouseEvent(kind, c, vf.mapToGlobal(c), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier)  # fmt: skip
        QtWidgets.QApplication.sendEvent(vf, ev)
    assert editor.selected == 2  # the core at the centre of the view


@pytest.mark.parametrize("layout, forward", [("qwerty", Qt.Key.Key_W), ("azerty", Qt.Key.Key_Z)])
def test_fly_keys_follow_the_keyboard_layout(editor, layout, forward):
    editor.set_keyboard(layout)
    d = editor.live.basis()[0]
    p0, dist = editor.live.pos.copy(), editor.live.distance
    ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, forward, Qt.KeyboardModifier.NoModifier, "")
    QtWidgets.QApplication.sendEvent(editor.viewfinder, ev)
    np.testing.assert_allclose(editor.live.pos - p0, d * 0.02 * dist, atol=1e-9)


def test_axis_gizmo_reorients_the_world_view(qapp, editor):
    w = editor.world
    ends = {axis: e for e, _, axis, _, _ in w._gizmo_ends()}
    pt = ends[(2, 1)]  # +Z end
    for kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease):
        ev = QtGui.QMouseEvent(kind, pt, w.mapToGlobal(pt), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier)  # fmt: skip
        QtWidgets.QApplication.sendEvent(w, ev)
    assert wait(qapp, lambda: not w._anim_timer.isActive(), 5)
    np.testing.assert_allclose(-w.view.basis()[0], [0, 0, 1], atol=1e-9)  # looking down from +Z


def test_movie_panel_edits_the_look(editor):
    editor.movie_insp.checks["star_psf"].setChecked(False)
    editor.movie_insp.checks["auto_exposure"].setChecked(True)
    assert editor.path.look["star_psf"] is False and editor.path.look["auto_exposure"] is True
    editor.request_refine()
    assert editor.backend.requests[-1]["look"]["star_psf"] is False


def test_up_buttons_set_up_exactly(editor):
    editor.live.up = np.array([0.1, 0.2, 0.9])
    editor.cam_insp._set_up(2)
    np.testing.assert_array_equal(editor.live.up, [0, 0, 1])


def test_proxy_follows_the_dump(qapp, editor, sim):
    editor.key_current()
    editor.live.dump = 3
    editor.live_changed()
    assert wait(qapp, lambda: editor.proxy_path == sim.dumps[3])
    img = editor.viewfinder.proxy_image()
    assert img is not None and img.width() > 0
