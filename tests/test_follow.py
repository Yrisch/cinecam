"""Cameras following a sink (CLAUDE.md invariant 3), on the synthetic moving sinks."""

import numpy as np
import pytest
from conftest import sink_pos

from cinecam.path.camera import CamState
from cinecam.path.campath import CameraPath, Keyframe, make_orbit


def path_with(tracks, keys, duration=4.0):
    p = CameraPath(keyframes=keys, duration=duration, dumps=[f"d{i}" for i in range(tracks.ndump)])
    p.set_tracks(tracks)
    return p


def key(t, dump, pos, target, follow=None, ride=False):
    return Keyframe(t, CamState(pos=pos, target=target, up=[0, 0, 1], dump=dump, follow=follow, ride=ride))


def test_look_at_keeps_the_sink_at_the_centre(tracks):
    p = path_with(tracks, [key(0, 0, [0, -3, 1], sink_pos(0, 0), follow=0), key(4, 4, [1, -3, 1], sink_pos(0, 4), follow=0)])
    for t in np.linspace(0, 4, 41):
        st = p.evaluate(t)
        np.testing.assert_allclose(st.target, sink_pos(0, st.dump), atol=1e-9)
        assert st.follow == 0 and not st.ride


def test_ride_along_keeps_the_offset(tracks):
    off = np.array([0.0, -2.0, 0.5])
    p = path_with(tracks, [key(0, 0, sink_pos(0, 0) + off, sink_pos(0, 0), 0, True),
                           key(4, 4, sink_pos(0, 4) + off, sink_pos(0, 4), 0, True)])  # fmt: skip
    for t in np.linspace(0, 4, 41):
        st = p.evaluate(t)
        np.testing.assert_allclose(st.pos - sink_pos(0, st.dump), off, atol=1e-8)
        assert st.ride


def test_switching_sinks_is_smooth(tracks):
    """From looking at sink 0 to looking at sink 2: the target glides, no jump."""
    p = path_with(tracks, [key(0, 0, [0, -3, 1], sink_pos(0, 0), follow=0), key(2, 2, [0, -3, 1], sink_pos(2, 2), follow=2),
                           key(4, 4, [0, -3, 1], sink_pos(2, 4), follow=2)])  # fmt: skip
    ts = np.linspace(0, 4, 401)
    T = np.array([p.evaluate(t).target for t in ts])
    step = np.linalg.norm(np.diff(T, axis=0), axis=1)
    assert np.abs(np.diff(step)).max() < 0.05 * step.max()  # the speed changes gradually
    np.testing.assert_allclose(p.evaluate(3.0).target, sink_pos(2, 3.0), atol=1e-9)
    assert p.evaluate(0.5).follow == 0 and p.evaluate(1.7).follow == 2  # the sink that dominates


def test_move_dump_moves_a_following_camera_with_its_sink(tracks):
    p = path_with(tracks, [])
    st = CamState(pos=[0, -3, 1], target=sink_pos(0, 1), up=[0, 0, 1], dump=1, follow=0, ride=True)
    off = st.pos - st.target
    p.move_dump(st, 3)
    np.testing.assert_allclose(st.target, sink_pos(0, 3), atol=1e-9)
    np.testing.assert_allclose(st.pos - st.target, off, atol=1e-9)
    world = CamState(pos=[0, -3, 1], target=[0, 0, 0], up=[0, 0, 1], dump=1)
    p.move_dump(world, 3)
    np.testing.assert_array_equal(world.target, [0, 0, 0])


def test_orbit_around_a_moving_sink(tracks):
    start = CamState(pos=sink_pos(0, 0) + [2, 0, 0], target=sink_pos(0, 0), up=[0, 0, 1], dump=0, follow=0)
    keys = make_orbit(start, 0.0, 4.0, 90, dump_end=4, tracks=tracks)
    for k in keys:
        np.testing.assert_allclose(k.state.target, sink_pos(0, k.state.dump), atol=1e-9)
        assert np.linalg.norm(k.state.pos - k.state.target) == pytest.approx(2.0)
