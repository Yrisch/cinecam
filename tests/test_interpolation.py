# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
The serial interpolation (see CLAUDE.md, invariants 1-2): keys are shot one after the
other, so adding or editing a key must only change the path next to it.
"""

import math

import numpy as np
import pytest

from cinecam.path.camera import CamState
from cinecam.path.campath import DEFAULT_LOOK, CameraPath, Keyframe, SerialHermite, Smoothstep, make_orbit

rng = np.random.default_rng(7)


def seg(f, j, n=17):
    return f(np.linspace(j, j + 1, n))


def deriv(f, t, side, h=1e-6):
    return (f(t + h) - f(t)) / h if side > 0 else (f(t) - f(t - h)) / h


# ---------------------------------------------------------------------- SerialHermite


def test_passes_through_the_keys():
    ts, Y = np.arange(6.0), rng.normal(size=(6, 3))
    f = SerialHermite(ts, Y, np.zeros(6, bool), 1e-9)
    np.testing.assert_allclose(f(ts), Y, atol=1e-12)


def test_c1_at_the_keys():
    ts, Y = np.cumsum(rng.uniform(0.5, 2, 7)), rng.normal(size=(7, 2))
    f = SerialHermite(ts, Y, np.zeros(7, bool), 1e-9)
    for t in ts[1:-1]:
        np.testing.assert_allclose(deriv(f, t, -1), deriv(f, t, +1), atol=1e-4)


def test_appending_a_key_keeps_every_segment_already_shot():
    ts, Y = np.arange(8.0), rng.normal(size=(8, 3))
    ease = np.zeros(8, bool)
    f = SerialHermite(ts[:7], Y[:7], ease[:7], 1e-9)
    g = SerialHermite(ts, Y, ease, 1e-9)
    for j in range(6):
        np.testing.assert_array_equal(seg(f, j), seg(g, j))


def test_editing_a_key_changes_only_the_segment_before_and_the_three_after():
    ts, Y = np.arange(9.0), rng.normal(size=(9, 3))
    ease = np.zeros(9, bool)
    f = SerialHermite(ts, Y, ease, 1e-9)
    k = 4
    Y2 = Y.copy()
    Y2[k] += [0.3, -0.2, 0.5]
    g = SerialHermite(ts, Y2, ease, 1e-9)
    changed = [j for j in range(8) if not np.array_equal(seg(f, j), seg(g, j))]
    assert changed == [k - 1, k, k + 1, k + 2]


def test_ease_key_has_zero_velocity():
    ts, Y = np.arange(5.0), rng.normal(size=(5, 3))
    ease = np.array([False, False, True, False, False])
    f = SerialHermite(ts, Y, ease, 1e-9)
    np.testing.assert_allclose(deriv(f, 2.0, -1), 0, atol=1e-5)
    np.testing.assert_allclose(deriv(f, 2.0, +1), 0, atol=1e-5)


def test_hold_stays_exactly_still():
    Y = np.array([[0.0, 0], [1, 1], [1, 1], [3, 0]])
    f = SerialHermite(np.arange(4.0), Y, np.zeros(4, bool), 1e-9)
    np.testing.assert_array_equal(seg(f, 1), np.ones((17, 2)))


def test_monotone_never_overshoots():
    ts = np.arange(6.0)
    y = np.array([0.0, 2.0, 2.1, 5.0, 1.0, 0.9])  # fast, slow, fast, reversal
    f = SerialHermite(ts, y, np.zeros(6, bool), 1e-12, monotone=True)
    for j in range(5):
        v = seg(f, j, 101)[:, 0]
        assert v.min() >= min(y[j], y[j + 1]) - 1e-12 and v.max() <= max(y[j], y[j + 1]) + 1e-12
    assert deriv(f, 3.0, +1)[0] == pytest.approx(0, abs=1e-4)  # turns back at rest


def test_smoothstep_weights():
    w = Smoothstep([0.0, 1.0, 2.0], [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(w(0.5), [0.5, 0.5])
    np.testing.assert_allclose(w(np.array([0.0, 1.0, 1.5])).sum(axis=1), 1.0)
    assert deriv(w, 1.0, -1)[0] == pytest.approx(0, abs=1e-5)


# ---------------------------------------------------------------------- CameraPath


def key(t, pos, target=(0, 0, 0), up=(0, 0, 1), **kw):
    return Keyframe(t, CamState(pos=pos, target=target, up=up, **kw))


def states(path, ts):
    return np.array([np.r_[s.pos, s.target, s.up, s.fov, s.dump] for s in (path.evaluate(t) for t in ts)])


def test_key_times_give_the_exact_key():
    k = key(1.0, [3, -9, 4], up=[0, 0, 1], fov=30, dump=2, exposure=0.5)
    p = CameraPath(keyframes=[key(0, [0, -10, 2]), k, key(2.5, [8, 0, 1])])
    st = p.evaluate(1.0)
    assert st.same_view(k.state)
    np.testing.assert_array_equal(st.up, [0, 0, 1])  # as typed, even with a tilted view


def test_same_position_on_different_dumps_stays_still():
    """Two keys at the same place on different dumps: the camera must not curve away and back."""
    p = CameraPath(keyframes=[key(0, [0, -10, 2], dump=0), key(2, [5, -5, 2], dump=10), key(4, [5, -5, 2], dump=30),
                              key(6, [9, 0, 2], dump=40)])  # fmt: skip
    for t in np.linspace(2, 4, 21):
        np.testing.assert_allclose(p.evaluate(t).pos, [5, -5, 2], atol=1e-12)
    assert p.evaluate(3.0).dump == pytest.approx(20)


def test_appending_a_key_keeps_the_path_already_shot():
    # moving away while turning around the target, every key further than the previous one
    keys = [key(0, [0, -5, 1], ease=True), key(1, [4, -5, 2]), key(2.5, [7.5, 0, 1]), key(4, [3, 8.5, 2])]
    p = CameraPath(keyframes=list(keys), duration=6)
    ts = np.linspace(0, 4, 121)
    before = states(p, ts)
    p.set_key(5.5, CamState(pos=[-10, 5, 3], target=[0, 0, 0], up=[0, 0, 1]))
    np.testing.assert_array_equal(states(p, ts), before)


def test_appending_a_key_that_turns_back_only_reshapes_the_last_segment():
    """A reversal (here dolly out then in) brings the camera to rest at the previous last key."""
    keys = [key(0, [0, -5, 1], ease=True), key(1, [4, -5, 2]), key(2.5, [7.5, 0, 1]), key(4, [3, 8.5, 2])]
    p = CameraPath(keyframes=list(keys), duration=6)
    ts = np.linspace(0, 4, 121)
    before = states(p, ts)
    p.set_key(5.5, CamState(pos=[-2, 1, 1], target=[0, 0, 0], up=[0, 0, 1]))
    after = states(p, ts)
    shot = ts <= 2.5
    np.testing.assert_array_equal(after[shot], before[shot])
    assert not np.array_equal(after[~shot], before[~shot])


def test_dolly_in_never_dips_or_overshoots():
    p = CameraPath(keyframes=[key(0, [0, -20, 0]), key(1, [0, -5, 0]), key(2, [0, -4.8, 0]), key(3, [0, -1, 0])])
    d = np.array([p.evaluate(t).distance for t in np.linspace(0, 3, 301)])
    assert (np.diff(d) <= 1e-12).all()


def test_interpolated_state_is_orthonormal_and_dump_is_linear():
    p = CameraPath(keyframes=[key(0, [0, -10, 2], dump=0), key(2, [10, 0, 2], dump=8)], duration=2)
    st = p.evaluate(0.5)
    d, r, u = st.basis()
    np.testing.assert_allclose(st.up, u)
    assert st.dump == pytest.approx(2.0)
    p.dumps = [f"d{i}" for i in range(9)]
    p.fps = 2.0
    np.testing.assert_array_equal(p.frame_dumps(), [0, 2, 4, 6, 8])
    assert p.dump_path(st) == "d2"


def test_orbit_has_an_even_angular_speed():
    start = CamState(pos=[10, 0, 0], target=[0, 0, 0], up=[0, 0, 1])
    keys = make_orbit(start, 0.0, 12.0, 360)
    assert len(keys) == 25  # a key every 15 degrees
    p = CameraPath(keyframes=keys, duration=12)
    ts = np.linspace(0.5, 11.5, 200)
    ang = np.unwrap([math.atan2(*p.evaluate(t).pos[[1, 0]]) for t in ts])
    speed = np.diff(ang) / np.diff(ts)
    assert speed.std() / speed.mean() < 0.02
    radius = [np.linalg.norm(p.evaluate(t).pos) for t in ts]
    np.testing.assert_allclose(radius, 10, rtol=1e-9)


# ---------------------------------------------------------------------- retiming a segment


def shot(fps=30):
    """Spiral in while the dump advances, with fov and exposure changes: every value moves."""
    start = CamState(pos=[0, -10, 3], target=[0, 0, 0], up=[0, 0, 1], dump=0)
    keys = make_orbit(start, 0.0, 12.0, 180, n_keys=7, radius_factor=0.5, dump_end=6)
    for i, k in enumerate(keys):
        k.state.fov = 45 - 2 * i
        k.state.exposure = 0.1 * i
    return CameraPath(keyframes=keys, duration=13, fps=fps)


def test_retiming_a_segment_keeps_the_motion_before_and_after():
    p = shot()
    ts = np.linspace(0, 13, 391)
    ref = states(p, ts)
    p.retime_segment(3, 5.0)  # keys 4-5: 2 s -> 5 s
    assert [k.t for k in p.keyframes] == [0, 2, 4, 6, 11, 13, 15] and p.duration == 16
    assert p.keyframes[3].stretch == pytest.approx(2.5)
    before, after = ts <= 6, ts >= 8
    np.testing.assert_array_equal(states(p, ts[before]), ref[before])
    np.testing.assert_allclose(states(p, ts[after] + 3), ref[after], atol=1e-12)  # same motion, 3 s later
    for t in (6.0, 11.0):  # same speed entering and leaving the stretched segment
        f = lambda x: states(p, np.atleast_1d(x))[0, :3]  # noqa: E731
        np.testing.assert_allclose(deriv(f, t, -1), deriv(f, t, +1), atol=1e-3)
    assert p.evaluate(8.5).dump == pytest.approx(3.5)  # the simulation goes slower in the segment
    p.retime_segment(3, 2.0)  # and back
    assert p.keyframes[3].stretch == pytest.approx(1.0)
    np.testing.assert_allclose(states(p, ts), ref, atol=1e-9)


def test_shortening_a_segment_keeps_the_rest_too():
    p = shot()
    ts = np.linspace(0, 13, 391)
    ref = states(p, ts)
    p.retime_segment(1, 0.5)
    assert p.duration == 11.5
    np.testing.assert_array_equal(states(p, ts[ts <= 2]), ref[ts <= 2])
    np.testing.assert_allclose(states(p, ts[ts >= 4] - 1.5), ref[ts >= 4], atol=1e-12)


def test_a_much_stretched_segment_never_turns_back():
    """Keeping the end speeds over a 10x longer segment would overshoot: the ends slow down instead."""
    p = CameraPath(keyframes=[key(0, [0, -20, 0]), key(1, [0, -15, 0]), key(2, [0, -10, 0]), key(3, [0, -5, 0])])
    p.retime_segment(1, 10.0)
    d = np.array([p.evaluate(t).distance for t in np.linspace(0, 12, 601)])
    assert (np.diff(d) <= 1e-12).all()


def test_a_key_inside_a_stretched_segment_keeps_it_stretched():
    p = shot()
    p.retime_segment(3, 5.0)
    i = p.set_key(8.0, p.evaluate(8.0))
    assert [k.stretch for k in p.keyframes[2:6]] == pytest.approx([1, 2.5, 2.5, 1])
    assert p.segment_at(8.0) == i and p.segment_at(7.9) == 3 and p.segment_at(16) is None


def test_set_key_replaces_the_key_of_the_same_frame():
    p = CameraPath(fps=10)
    p.set_key(1.02, CamState(pos=[0, -1, 0], target=[0, 0, 0], up=[0, 0, 1]))
    i = p.set_key(0.98, CamState(pos=[0, -2, 0], target=[0, 0, 0], up=[0, 0, 1]))
    assert len(p.keyframes) == 1 and i == 0 and p.keyframes[0].t == 1.0
    assert p.keyframes[0].state.pos[1] == -2


def test_save_load_roundtrip(tmp_path):
    p = CameraPath(keyframes=[key(0, [0, -10, 2], ease=True), key(3, [5, 1, 2], follow=2, ride=True)],
                   fps=24, duration=5, resolution=(1080, 1080), dumps=["/a", "/b"])  # fmt: skip
    p.look["cmap"] = "viridis"
    f = tmp_path / "p.json"
    p.save(f)
    q = CameraPath.load(f)
    assert q.to_dict() == p.to_dict()
    p.retime_segment(0, 4.5)
    p.save(f)
    assert CameraPath.load(f).keyframes[0].stretch == 1.5


def test_old_files_get_the_new_look_defaults_and_per_key_ease():
    q = CameraPath.from_dict({"keyframes": [{"t": 1, "pos": [0, -1, 0], "target": [0, 0, 0], "up": [0, 0, 1]},
                                            {"t": 0, "pos": [0, -2, 0], "target": [0, 0, 0], "up": [0, 0, 1]}],
                              "look": {"cmap": "magma", "ease": True}})  # fmt: skip
    assert q.look["cmap"] == "magma"
    assert all(q.look[k] == v for k, v in DEFAULT_LOOK.items() if k != "cmap")
    assert "ease" not in q.look
    assert [k.state.ease for k in q.keyframes] == [True, False]  # the old global ease eased the first key
    assert [k.stretch for k in q.keyframes] == [1, 1]
    assert q.format == "phantom"
