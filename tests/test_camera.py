# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

import math

import numpy as np
import pytest

from cinecam.path.camera import CamState, frustum_corners, pixel_size, project, stereo_eyes, unproject_delta

Z = np.array([0.0, 0.0, 1.0])


def cam(**kw):
    kw.setdefault("pos", [0.0, -10.0, 3.0])
    kw.setdefault("target", [0.0, 0.0, 0.0])
    kw.setdefault("up", [0.0, 0.0, 1.0])
    return CamState(**kw)


def test_basis_is_orthonormal_like_camera3d():
    st = cam()
    d, r, u = st.basis()
    np.testing.assert_allclose(d, (st.target - st.pos) / np.linalg.norm(st.target - st.pos))
    np.testing.assert_allclose(np.array([d, r, u]) @ np.array([d, r, u]).T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(d, u), r, atol=1e-12)  # right = dir x up
    assert u @ Z > 0


def test_basis_with_up_along_the_view():
    d, r, u = cam(pos=[0, 0, 10], up=[0, 0, 1]).basis()
    assert abs(r @ d) < 1e-12 and abs(u @ d) < 1e-12 and np.linalg.norm(u) == pytest.approx(1)


def test_project_conventions():
    st = cam()
    sx, sy, z = project(st, np.array([st.target, st.target + [0, 0, 1], st.target + [1, 0, 0]]), 200, 100)
    assert (sx[0], sy[0]) == pytest.approx((100, 50))  # target at the centre
    assert sy[1] < 50  # world up is up on screen (y down)
    assert sx[2] > 100  # +x is to the right when looking along +y
    assert z[0] == pytest.approx(st.distance)
    _, _, zb = project(st, np.array([st.pos - (st.target - st.pos)]), 200, 100)
    assert zb[0] < 0  # behind the camera


def test_unproject_is_the_inverse_of_project():
    st = cam(fov=30)
    nx, ny = 300, 200
    p = np.array([0.2, 0.5, -0.1])
    sx, sy, z = project(st, p[None], nx, ny)
    q = p + unproject_delta(st, z[0], 7.0, -4.0, ny)
    sx2, sy2, z2 = project(st, q[None], nx, ny)
    assert (sx2[0] - sx[0], sy2[0] - sy[0]) == pytest.approx((7.0, -4.0), abs=1e-6)
    assert z2[0] == pytest.approx(z[0])
    assert pixel_size(st, 10.0, ny) == pytest.approx(2 * 10 * math.tan(math.radians(15)) / ny)


def test_frustum_corners_project_to_the_image_corners():
    st = cam(fov=40)
    cs = frustum_corners(st, 2.0, 5.0)
    sx, sy, _ = project(st, np.array(cs), 200, 100)
    np.testing.assert_allclose(np.c_[sx, sy], [[0, 0], [200, 0], [200, 100], [0, 100]], atol=1e-9)


def test_orbit_keeps_target_and_distance():
    st = cam()
    dist = st.distance
    st.orbit(0.4, 0.2)
    assert st.distance == pytest.approx(dist)
    np.testing.assert_array_equal(st.target, [0, 0, 0])


def test_turntable_keeps_the_typed_up():
    """The viewfinder orbit: yaw around the up the user set, so a +Z camera stays level."""
    st = cam()
    dist = st.distance
    st.turntable(-0.7, 0.1, world_up=Z)
    st.turntable(0.4, -0.2, world_up=Z)
    assert st.distance == pytest.approx(dist)
    np.testing.assert_array_equal(st.up, Z)  # exactly as typed: no roll was made


def test_turntable_does_not_go_over_the_pole():
    st = cam()
    for _ in range(50):
        st.turntable(0.0, -0.2)
    d = st.basis()[0]
    assert abs(d @ Z) < 0.996


def test_look_turns_the_head_around_the_position():
    st = cam()
    pos, dist = st.pos.copy(), st.distance
    st.look(0.3, -0.1)
    np.testing.assert_array_equal(st.pos, pos)
    assert st.distance == pytest.approx(dist)
    np.testing.assert_array_equal(st.up, Z)


def test_roll_then_level():
    st = cam()
    d0 = st.basis()[0]
    st.roll(0.3)
    assert st.basis()[2] @ Z < 0.99
    np.testing.assert_allclose(st.basis()[0], d0, atol=1e-12)
    st.level()
    np.testing.assert_array_equal(st.up, Z)  # exactly +Z, Camera3d orthogonalises it


def test_pan_dolly_fly():
    st = cam()
    off = st.pos - st.target
    st.pan(0.1, 0.0)
    np.testing.assert_allclose(st.pos - st.target, off)
    st.dolly(0.5)
    np.testing.assert_allclose(st.pos - st.target, off * 0.5)
    d = st.basis()[0]
    p0 = st.pos.copy()
    st.fly(0.1, 0, 0)
    np.testing.assert_allclose(st.pos - p0, d * 0.1 * st.distance)


def test_dict_roundtrip():
    st = cam(fov=33.0, dump=4.5, clim=(0.2, 3.0), exposure=-1.0, extinction=2.0, follow=3, ride=True, ease=True)
    again = CamState.from_dict(st.to_dict())
    assert again.same_view(st)
    assert CamState.from_dict(dict(st.to_dict(), unknown_key=1)).same_view(st)  # forward compatible
    assert cam(ride=True).ride is False  # riding needs a sink to follow


# ---------------------------------------------------------------------- stereo


def test_lens_shift_translates_the_image():
    st = cam(fov=40.0)
    pts = np.random.default_rng(3).normal(size=(20, 3))
    sx, sy, z = project(st, pts, 192, 108)
    st.shift = 2 * 7 / 192  # 7 px
    sx2, sy2, _ = project(st, pts, 192, 108)
    np.testing.assert_allclose(sx2, sx - 7)
    np.testing.assert_array_equal(sy2, sy)
    assert "shift" not in st.to_dict()  # render only, never in a path file


def test_stereo_eyes_are_parallel_with_the_target_at_the_centre():
    nx, ny = 1920, 1080
    st = cam(pos=[3.0, -10.0, 4.0], target=[1.0, 0.5, -0.5], up=[0.2, 0.1, 1.0], fov=45.0)
    left, right = stereo_eyes(st, 4.0, nx, ny)
    d, r, u = st.basis()
    for e in (left, right):
        de, re, ue = e.basis()
        np.testing.assert_allclose([de, re, ue], [d, r, u], atol=1e-12)  # parallel, no toe-in
        assert (e.shift * nx / 2) == pytest.approx(round(e.shift * nx / 2))  # whole pixels (gas crop)
        sx, sy, _ = project(e, [st.target], nx, ny)
        assert abs(sx[0] - nx / 2) <= 0.5 and sy[0] == pytest.approx(ny / 2)  # zero parallax on the target
    np.testing.assert_allclose(right.pos - left.pos, 2 * st.distance * math.tan(math.radians(2.0)) * r)
    a, b = left.pos - st.target, right.pos - st.target
    ang = math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), a @ b))
    assert ang == pytest.approx(4.0, abs=0.01)  # 4 degrees apart seen from the target
    assert left.shift > 0 > right.shift and left.shift == -right.shift
    assert right.pos @ r > left.pos @ r


def test_stereo_parallax_behind_and_in_front_of_the_screen():
    nx, ny = 960, 540
    st = cam()
    left, right = stereo_eyes(st, 4.0, nx, ny)
    d = st.basis()[0]
    pts = np.array([st.target + 5 * d, st.target - 4 * d, st.target + [2.0, 0.0, 1.5]])
    lx, ly, _ = project(left, pts, nx, ny)
    rx, ry, _ = project(right, pts, nx, ny)
    np.testing.assert_allclose(ly, ry, atol=1e-9)  # no vertical disparity anywhere
    assert rx[0] - lx[0] > 1  # behind the target: right eye image to the right, behind the screen
    assert rx[1] - lx[1] < -1  # in front: crossed, out of the screen
