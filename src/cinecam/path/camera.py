# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Camera state and camera maths for cinecam.

Conventions follow shammath::Camera3d (gluLookAt + gluPerspective):
the camera looks along ``dir = normalize(target - pos)``, ``right = normalize(dir x up)``,
``up = right x dir``; ``fov`` is the vertical field of view in degrees.
Screen coordinates used here have their origin at the top-left corner of the image,
x to the right and y downward, pixel ``(col, row)`` covering ``[col, col+1) x [row, row+1)``.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np

ZNEAR = 1e-3


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rotate(v, axis, angle):
    """Rotate vector(s) v around axis by angle (radians), Rodrigues formula."""
    k = normalize(axis)
    v = np.asarray(v, dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + np.outer(v @ k, k).reshape(v.shape) * (1 - c)


@dataclass
class CamState:
    pos: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float = 45.0
    dump: float = 0.0
    clim: tuple = (0.1, 3.5)  # log10 column density [Msun/pc^2]
    exposure: float = 0.0  # star exposure, dex
    extinction: float = 1.0  # star extinction multiplier
    follow: int = None  # sink index the camera looks at (None: world)
    ride: bool = False  # with follow: the camera also moves with the sink
    ease: bool = False  # keyframe: the camera comes to rest at this key
    shift: float = 0.0  # horizontal lens shift, in half image widths (stereo eyes only, see stereo_eyes; not saved)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        self.up = np.asarray(self.up, dtype=float)
        self.clim = (float(self.clim[0]), float(self.clim[1]))
        self.follow = None if self.follow is None else int(self.follow)
        self.ride = bool(self.ride) and self.follow is not None
        self.ease = bool(self.ease)

    def copy(self):
        return copy.deepcopy(self)

    def basis(self):
        """Orthonormal (dir, right, up) exactly as Camera3d builds it."""
        # plain float arithmetic: called many times per repaint on 3-vectors, where numpy is slow
        px, py, pz = self.pos.tolist()
        tx, ty, tz = self.target.tolist()
        ux, uy, uz = self.up.tolist()
        dx, dy, dz = tx - px, ty - py, tz - pz
        n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        dx, dy, dz = dx / n, dy / n, dz / n
        rx, ry, rz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
        n = math.sqrt(rx * rx + ry * ry + rz * rz)
        if n < 1e-12:  # up colinear with dir: pick any perpendicular
            ux, uy, uz = (0.0, 0.0, 1.0) if abs(dz) < 0.9 else (0.0, 1.0, 0.0)
            rx, ry, rz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
            n = math.sqrt(rx * rx + ry * ry + rz * rz)
        rx, ry, rz = rx / n, ry / n, rz / n
        d = np.array((dx, dy, dz))
        r = np.array((rx, ry, rz))
        u = np.array((ry * dz - rz * dy, rz * dx - rx * dz, rx * dy - ry * dx))
        return d, r, u

    @property
    def distance(self):
        return float(np.linalg.norm(self.target - self.pos))

    def orthonormalized(self):
        s = self.copy()
        s.up = s.basis()[2]
        return s

    def to_dict(self):
        return {
            "pos": self.pos.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "fov": self.fov,
            "dump": self.dump,
            "clim": list(self.clim),
            "exposure": self.exposure,
            "extinction": self.extinction,
            "follow": self.follow,
            "ride": self.ride,
            "ease": self.ease,
        }

    @staticmethod
    def from_dict(d):
        return CamState(**{k: d[k] for k in d if k in CamState.__dataclass_fields__})

    def same_view(self, other, tol=1e-9):
        if other is None:
            return False
        return (
            np.allclose(self.pos, other.pos, atol=tol)
            and np.allclose(self.target, other.target, atol=tol)
            and np.allclose(self.up, other.up, atol=tol)
            and abs(self.fov - other.fov) < tol
            and abs(self.dump - other.dump) < tol
            and np.allclose(self.clim, other.clim, atol=tol)
            and abs(self.exposure - other.exposure) < tol
            and abs(self.extinction - other.extinction) < tol
            and self.follow == other.follow
            and self.ride == other.ride
            and self.ease == other.ease
        )

    # ------------------------------------------------------------------ navigation

    def _keep_hint(self, hint):
        """
        After a move, put back the up vector the user typed (e.g. exactly +Z) if it still
        gives the same image; otherwise keep the new orthogonal up (the camera rolled).
        """
        u = self.basis()[2]
        new = self.up
        self.up = hint
        if not np.allclose(self.basis()[2], u, atol=1e-9):
            self.up = new

    def orbit(self, dyaw, dpitch):
        """Turn around the target: yaw around the camera up, pitch around the camera right."""
        hint = self.up
        d, r, u = self.basis()
        rel = self.pos - self.target
        rel = rotate(rel, u, -dyaw)
        up = u
        r2 = normalize(np.cross(normalize(-rel), up))
        rel = rotate(rel, r2, -dpitch)
        up = rotate(up, r2, -dpitch)
        self.pos = self.target + rel
        self.up = up
        self._keep_hint(hint)

    def turntable(self, dyaw, dpitch, world_up=(0.0, 0.0, 1.0)):
        """Orbit with the yaw axis locked on world_up (no accidental roll)."""
        hint = self.up
        rel = self.pos - self.target
        rel = rotate(rel, world_up, -dyaw)
        up = rotate(self.up, world_up, -dyaw)
        self.pos = self.target + rel
        self.up = up
        d, r, u = self.basis()
        rel2 = rotate(rel, r, -dpitch)
        # do not go over the pole
        if abs(normalize(rel2) @ normalize(world_up)) < 0.995:
            self.pos = self.target + rel2
            self.up = rotate(u, r, -dpitch)
        self._keep_hint(hint)

    def look(self, dyaw, dpitch):
        """Turn the head: the target turns around the camera position."""
        hint = self.up
        d, r, u = self.basis()
        # yaw around the user's up (+Z: stays level, as in a first-person game)
        yaw_axis = normalize(hint)
        rel = self.target - self.pos
        rel = rotate(rel, yaw_axis, -dyaw)
        u = rotate(u, yaw_axis, -dyaw)
        r2 = normalize(np.cross(normalize(rel), u))
        rel = rotate(rel, r2, dpitch)
        self.up = rotate(u, r2, dpitch)
        self.target = self.pos + rel
        self._keep_hint(hint)

    def pan(self, dx, dy):
        """Translate camera and target in the image plane, (dx, dy) in units of the distance."""
        d, r, u = self.basis()
        shift = (r * dx + u * dy) * self.distance
        self.pos = self.pos + shift
        self.target = self.target + shift

    def dolly(self, factor):
        """Move toward the target, factor < 1 gets closer."""
        self.pos = self.target + (self.pos - self.target) * factor

    def fly(self, forward, right, upward):
        """Translate camera and target in camera frame, in units of the distance."""
        d, r, u = self.basis()
        shift = (d * forward + r * right + u * upward) * self.distance
        self.pos = self.pos + shift
        self.target = self.target + shift

    def roll(self, angle):
        d, r, u = self.basis()
        self.up = rotate(u, d, angle)

    def level(self, world_up=(0.0, 0.0, 1.0)):
        """Remove the roll, keeping the image upright with respect to world_up."""
        d = normalize(self.target - self.pos)
        wu = normalize(world_up)
        if abs(d @ wu) < 0.999:
            self.up = wu  # kept as given: Camera3d (and basis) orthogonalise it


# ---------------------------------------------------------------------- projection


def project(state, pts, nx, ny):
    """
    Project world points into screen coordinates.

    Returns (sx, sy, z) where z is the depth along the view direction,
    sx, sy are continuous screen coordinates (top-left origin, y down).
    """
    d, r, u = state.basis()
    rel = np.asarray(pts, dtype=np.float64) - state.pos
    z = rel @ d
    x = rel @ r
    y = rel @ u
    th = math.tan(math.radians(state.fov) / 2)
    aspect = nx / ny
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = np.where(z > ZNEAR, z, np.nan)
        sx = (x / (zz * th * aspect) - state.shift + 1.0) * 0.5 * nx
        sy = (1.0 - y / (zz * th)) * 0.5 * ny
    return sx, sy, z


def stereo_eyes(state, angle_deg, nx, ny):
    """
    Left and right eye cameras of a flat-screen stereo pair around the mono camera.

    The eyes are seen angle_deg apart from the target (±angle/2), on the line through the
    mono camera along its right vector, at baseline 2 D tan(angle/2) for a target at
    distance D. They look parallel to the mono camera (no toe-in, so no vertical
    disparity) and their images are shifted (``shift``, an off-axis frustum) so that the
    target is at the centre of both: zero parallax on the target, which sits on the
    screen plane; what is behind it appears behind the screen. The shift is a whole
    number of pixels so that the gas render can be a wider image cropped exactly
    (SimScene.column). Returns (left, right).
    """
    d, r, u = state.basis()
    D = state.distance
    b = D * math.tan(math.radians(angle_deg) / 2)
    th = math.tan(math.radians(state.fov) / 2)
    px = round(math.tan(math.radians(angle_deg) / 2) / (th * nx / ny) * nx / 2)  # target offset in a parallel eye
    eyes = []
    for side in (-1, 1):
        e = state.copy()
        e.pos = state.pos + side * b * r
        e.target = state.target + side * b * r
        e.up = u
        e.shift = -side * 2.0 * px / nx
        eyes.append(e)
    return tuple(eyes)


def pixel_size(state, z, ny):
    """World size of one pixel at depth z."""
    return 2.0 * z * math.tan(math.radians(state.fov) / 2) / ny


def unproject_delta(state, z, dx_px, dy_px, ny):
    """World displacement (in the plane at depth z) for a screen displacement in pixels."""
    d, r, u = state.basis()
    s = pixel_size(state, z, ny)
    return (r * dx_px - u * dy_px) * s


def frustum_corners(state, aspect, length):
    """Corners of the frustum section at distance length (TL, TR, BR, BL)."""
    d, r, u = state.basis()
    th = math.tan(math.radians(state.fov) / 2)
    c = state.pos + d * length
    hw = length * th * aspect
    hh = length * th
    return [c + u * hh - r * hw, c + u * hh + r * hw, c - u * hh + r * hw, c - u * hh - r * hw]
