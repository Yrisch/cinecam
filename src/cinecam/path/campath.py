# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Keyframed camera path: serial interpolation of the keys, following sinks, path files.
"""

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .camera import CamState, normalize, rotate


class SerialHermite:
    """
    Piecewise cubic (Hermite) curve through keys, built for shooting keys one after the other.

    Each segment between two keys only depends on its two keys and on the velocity at
    its ends; consecutive segments share the key value and the velocity (C1). The
    velocity at a key is estimated from that key and the ones *before* it (one-sided
    parabola through the last three keys, kept within a cone around the incoming
    chord), so adding a key at the end never changes the existing segments, and
    editing a key only changes the segments next to it. The velocity is zero:

    * at keys flagged ``ease`` (the camera comes to rest there),
    * around a hold (two consecutive keys with the same value): the curve stays still.

    ``monotone``: each component is treated alone and never overshoots its key values
    (for scalars like the distance, the fov, the colour range).

    ``stretch`` (per key, factor of the segment that starts there): the velocities are
    estimated as if each segment lasted ``h / stretch``, so stretching a segment (and
    shifting the keys after it) keeps the velocity at its ends, and every other segment
    is unchanged, only shifted in time. A segment stretched so much that it would have to
    turn back to keep them gets slower end velocities (its own ends only).
    """

    def __init__(self, ts, Y, ease, tol, monotone=False, stretch=None):
        self.ts = np.asarray(ts, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self.Y = Y.reshape(len(self.ts), -1)
        self.h = np.diff(self.ts)
        if monotone and self.Y.shape[1] > 1:
            cols = [SerialHermite(ts, self.Y[:, j], ease, tol, True, stretch) for j in range(self.Y.shape[1])]
            self.V0 = np.concatenate([c.V0 for c in cols], axis=1)
            self.V1 = np.concatenate([c.V1 for c in cols], axis=1)
            return
        n = len(self.ts)
        Y = self.Y
        stretch = np.ones(n - 1) if stretch is None else np.asarray(stretch, dtype=float)[: n - 1]
        h = self.h / stretch  # segment durations the velocities are estimated with
        C = np.diff(Y, axis=0) / h[:, None]  # chord velocities
        hold = np.linalg.norm(np.diff(Y, axis=0), axis=1) < tol
        V = np.zeros_like(Y)
        for k in range(n):
            if ease[k] or (k > 0 and hold[k - 1]) or (k < n - 1 and hold[k]):
                continue
            if k == 0:
                V[0] = C[0]  # first key: start along the first segment
                continue
            c = C[k - 1]
            v = c
            if k >= 2 and not hold[k - 2]:
                # derivative at key k of the parabola through keys k-2, k-1, k
                v = c + h[k - 1] / (h[k - 1] + h[k - 2]) * (c - C[k - 2])
                dv = v - c
                lim, nd = np.linalg.norm(c), np.linalg.norm(dv)
                if nd > lim:  # stay within a cone around the incoming direction
                    v = c + dv * (lim / nd)
            V[k] = v
        if monotone:
            # no overshoot (Fritsch-Carlson): where the value turns back (dolly in then out) it
            # comes to rest at the key, and it brakes in time before a much slower segment.
            # This looks at the next key, so such a key also changes the end of the segment
            # before it (like a hold does), keeping the curve C1.
            for k in range(n - 1):
                c, v = C[k, 0], V[k, 0]
                if c == 0.0 or np.sign(v) != np.sign(c):
                    V[k, 0] = 0.0
                elif abs(v) > 3 * abs(c):
                    V[k, 0] = 3 * c
        self.V0 = V[:-1].copy()  # velocity at the start of each segment
        self.V1 = V[1:].copy()  # velocity at the end of each segment
        # stretched segments: end velocities above 3x the real chord would overshoot and come back
        Ca = np.diff(Y, axis=0) / self.h[:, None]
        for i in np.flatnonzero(stretch > 1):
            for V_ in (self.V0, self.V1):
                if monotone:
                    V_[i] = np.clip(V_[i], -3 * np.abs(Ca[i]), 3 * np.abs(Ca[i]))
                else:
                    nv, lim = np.linalg.norm(V_[i]), 3 * np.linalg.norm(Ca[i])
                    if nv > lim:
                        V_[i] *= lim / nv

    def segment(self, t):
        return np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.ts) - 2)

    def __call__(self, t):
        """Value at t (scalar -> (dim,), array (m,) -> (m, dim))."""
        tt = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), self.ts[0], self.ts[-1])
        i = self.segment(tt)
        h = self.h[i][:, None]
        s = ((tt - self.ts[i]) / self.h[i])[:, None]
        s2, s3 = s * s, s * s * s
        out = (
            (2 * s3 - 3 * s2 + 1) * self.Y[i]
            + (s3 - 2 * s2 + s) * h * self.V0[i]
            + (-2 * s3 + 3 * s2) * self.Y[i + 1]
            + (s3 - s2) * h * self.V1[i]
        )
        return out[0] if np.ndim(t) == 0 else out


class Smoothstep:
    """Key values joined by smoothstep ramps (zero velocity at every key), for blend weights."""

    def __init__(self, ts, Y):
        self.ts = np.asarray(ts, dtype=float)
        self.Y = np.asarray(Y, dtype=float)

    def __call__(self, t):
        tt = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), self.ts[0], self.ts[-1])
        i = np.clip(np.searchsorted(self.ts, tt, side="right") - 1, 0, len(self.ts) - 2)
        s = ((tt - self.ts[i]) / (self.ts[i + 1] - self.ts[i]))[:, None]
        s = s * s * (3 - 2 * s)
        out = self.Y[i] + (self.Y[i + 1] - self.Y[i]) * s
        return out[0] if np.ndim(t) == 0 else out


DEFAULT_LOOK = {
    "cmap": "inferno",
    "show_gas": True,
    "show_stars": True,
    "show_cores": True,
    "core_radius": None,  # code length units, None: physical size (sink accretion radius)
    "core_color": [1.0, 0.27, 0.0],
    "psf_scale": 1.0,  # size of the star PSF (1: core FWHM 1.6 px on a 1920 px frame, same look at any resolution)
    "star_psf": True,  # False: stars as dots, as in the editor proxy
    "star_spikes": True,  # diffraction spikes in the PSF
    "spike_angle": 0.0,  # degrees, counterclockwise on screen (0: along the frame axes)
    "auto_exposure": False,  # star exposure follows the brightest stars in view (+ the key exposure)
    "time_label": True,
}


# ---------------------------------------------------------------------- path


@dataclass
class Keyframe:
    t: float
    state: CamState
    stretch: float = 1.0  # time stretch of the segment to the next key (see CameraPath.retime_segment)

    def to_dict(self):
        d = {"t": self.t, "stretch": self.stretch}
        d.update(self.state.to_dict())
        return d

    @staticmethod
    def from_dict(d):
        return Keyframe(float(d["t"]), CamState.from_dict(d), float(d.get("stretch", 1.0)))


@dataclass
class CameraPath:
    keyframes: list = field(default_factory=list)
    fps: float = 30.0
    duration: float = 10.0
    resolution: tuple = (1920, 1080)
    dumps: list = field(default_factory=list)
    look: dict = field(default_factory=lambda: dict(DEFAULT_LOOK))
    format: str = "phantom"  # dump format for the fast readers (data/readers.py)

    def __post_init__(self):
        self._interp = None
        self.tracks = None  # sinktracks.SinkTracks for keys following a sink (not saved)
        for k, v in DEFAULT_LOOK.items():
            self.look.setdefault(k, v)

    # -- editing

    def invalidate(self):
        self._interp = None

    def set_tracks(self, tracks):
        self.tracks = tracks
        self.invalidate()

    def anchor(self, sink, dump):
        """Position of a followed sink at a (fractional) dump, None without tracks."""
        if sink is None or self.tracks is None or not self.tracks.has(sink):
            return None
        return self.tracks.position(sink, dump)

    def move_dump(self, state, dump):
        """Change the dump of a state; a following camera moves with its sink."""
        a0, a1 = self.anchor(state.follow, state.dump), self.anchor(state.follow, dump)
        state.dump = dump
        if a0 is not None and a1 is not None:
            state.target = state.target + (a1 - a0)
            if state.ride:
                state.pos = state.pos + (a1 - a0)

    def sort(self):
        self.keyframes.sort(key=lambda k: k.t)
        self.invalidate()

    def snap(self, t):
        return round(t * self.fps) / self.fps

    def key_at(self, t, tol=None):
        tol = 0.5 / self.fps if tol is None else tol
        for i, k in enumerate(self.keyframes):
            if abs(k.t - t) < tol:
                return i
        return None

    def set_key(self, t, state):
        """Insert a keyframe at t, replacing an existing one at the same frame. Returns its index."""
        t = self.snap(t)
        i = self.key_at(t)
        if i is not None:
            self.keyframes[i].state = state.copy()
        else:
            before = [k for k in self.keyframes if k.t < t]  # a key splitting a stretched segment: both halves stay stretched
            self.keyframes.append(Keyframe(t, state.copy(), before[-1].stretch if before else 1.0))
        self.sort()
        if t > self.duration:
            self.duration = t
        return self.key_at(t)

    def remove_key(self, i):
        del self.keyframes[i]
        self.invalidate()

    def segment_at(self, t):
        """Index of the key starting the segment at time t (the segment after a key at t), or None."""
        ts = [k.t for k in self.keyframes]
        i = int(np.searchsorted(ts, t + 0.5 / self.fps, side="right")) - 1
        return i if 0 <= i < len(ts) - 1 else None

    def retime_segment(self, i, h):
        """
        Give the segment from key i to key i+1 the duration h (s, snapped to frames).

        The keys after it move by the difference and the movie gets longer (shorter); the
        camera motion before and after the segment is unchanged, only shifted in time, and
        the camera keeps its speed where it enters and leaves the segment (key stretch).
        """
        ks = self.keyframes
        h0 = ks[i + 1].t - ks[i].t
        h = max(self.snap(h), 1.0 / self.fps)
        d = h - h0
        ks[i].stretch *= h / h0
        for k in ks[i + 1 :]:
            k.t = self.snap(k.t + d)
        self.duration = max(self.snap(self.duration + d), ks[-1].t)
        self.invalidate()

    @property
    def n_frames(self):
        return int(round(self.duration * self.fps)) + 1

    def frame_time(self, i):
        return i / self.fps

    # -- interpolation

    def _build(self):
        """
        Interpolation, with sinks to follow.

        Every key stores world coordinates. Each reference frame (the world, and every
        followed sink) gets its own smooth curves through all the keys expressed in that
        frame (key value minus the sink position at the key's dump). The camera at time t
        is a blend of the frames, with weights going smoothly (PCHIP) from one key's frame
        to the next one's: the target uses the frame of the followed sink, the camera
        position the same frame when it rides along, the world otherwise.
        Without any sink to follow this is plain world-space interpolation.
        """
        ks = self.keyframes
        n = len(ks)
        if n == 1:
            self._interp = ("const", ks[0].state)
            return
        ts = np.array([k.t for k in ks])
        D = np.array([k.state.dump for k in ks])
        Tw = np.array([k.state.target for k in ks])
        Pw = np.array([k.state.pos for k in ks])
        # the camera is placed relative to its target: direction of the offset by spline,
        # distance by a monotone spline of its log (no overshoot after a fast dolly), so
        # orbits stay circular (same in every frame)
        R = Pw - Tw
        logd = np.log(np.maximum(np.linalg.norm(R, axis=1), 1e-300))
        U = np.array([k.state.basis()[2] for k in ks])
        S = np.array(
            [
                [k.state.fov, k.state.clim[0], k.state.clim[1], k.state.exposure, k.state.extinction]
                for k in ks
            ]
        )
        # piecewise curves, one segment per pair of keys (see SerialHermite): adding a key
        # does not change the segments already shot. Segments that do not move are held
        # exactly (e.g. two keys at the same place on different dumps).
        ease = np.array([k.state.ease for k in ks])
        st = np.array([k.stretch for k in ks])
        tol = 1e-4 * max(float(np.median(np.exp(logd))), 1e-30)

        def still(Y):
            return np.linalg.norm(np.diff(Y, axis=0), axis=1) < tol

        fp = SerialHermite(ts, R, ease, tol, stretch=st)
        fd = SerialHermite(ts, logd, ease, 1e-6, monotone=True, stretch=st)
        fu = SerialHermite(ts, U, ease, 1e-3, stretch=st)
        fs = SerialHermite(ts, S, ease, 1e-9, monotone=True, stretch=st)

        frames = [None]
        for k in ks:
            f = k.state.follow
            if f is not None and f not in frames and self.anchor(f, k.state.dump) is not None:
                frames.append(f)
        per_frame = []  # (target spline, local positions, position hold mask)
        for f in frames:
            A = np.zeros((n, 3)) if f is None else self.tracks.track(f, D)
            Tl, Pl = Tw - A, Pw - A
            per_frame.append((SerialHermite(ts, Tl, ease, tol, stretch=st), Pl, still(Pl)))
        if len(frames) > 1:
            fidx = {f: i for i, f in enumerate(frames)}
            nf = len(frames)
            W = np.zeros((n, 2 * nf))
            for i, k in enumerate(ks):
                j = fidx.get(k.state.follow, 0)
                W[i, j] = 1.0  # target frame
                W[i, nf + (j if k.state.ride else 0)] = 1.0  # position frame
            fw = Smoothstep(ts, W)  # each change of followed sink eases in and out
        else:
            fw = None
        self._interp = ("spline", ts, fp, fd, fu, fs, D, frames, per_frame, fw)

    def _world(self, tt):
        """(dumps, target, pos, weights) arrays at times tt."""
        _, ts, fp, fd, fu, fs, D, frames, per_frame, fw = self._interp
        dd = np.interp(tt, ts, D)
        m, nf = len(tt), len(frames)
        W = np.ones((m, 2)) if fw is None else np.atleast_2d(fw(tt))
        rel = fp(tt)
        rel = rel / np.linalg.norm(rel, axis=1)[:, None] * np.exp(fd(tt))
        seg = fp.segment(tt)
        T, P = np.zeros((m, 3)), np.zeros((m, 3))
        for j, (f, (ft, Pl, hold_pos)) in enumerate(zip(frames, per_frame)):
            wt, wp = W[:, j : j + 1], W[:, nf + j : nf + j + 1]
            if not (wt.any() or wp.any()):
                continue
            A = np.zeros((m, 3)) if f is None else self.tracks.track(f, dd)
            Tf = ft(tt) + A
            Pf = Tf + rel
            # camera standing still in this frame (possibly turning toward a moving target)
            hold = hold_pos[seg]
            Pf[hold] = Pl[seg[hold]] + A[hold]
            T += wt * Tf
            P += wp * Pf
        return dd, T, P, W

    def evaluate(self, t):
        """Camera state at time t (seconds), None if there is no keyframe."""
        if not self.keyframes:
            return None
        if self._interp is None:
            self._build()
        if self._interp[0] == "const":
            return self._interp[1].copy()
        _, ts, fp, fd, fu, fs, D, frames, per_frame, fw = self._interp
        tc = float(np.clip(t, ts[0], ts[-1]))
        i = int(np.argmin(np.abs(ts - tc)))
        if abs(ts[i] - tc) < 1e-9:
            return self.keyframes[i].state.copy()  # exactly the key, up vector as typed
        dd, T, P, W = self._world(np.array([tc]))
        s = fs(tc)
        st = CamState(
            P[0],
            T[0],
            fu(tc),
            fov=float(s[0]),
            dump=float(dd[0]),
            clim=(float(s[1]), float(s[2])),
            exposure=float(s[3]),
            extinction=float(s[4]),
        )
        nf = len(frames)
        if nf > 1:  # the sink that dominates here, so that a key inserted here keeps following it
            j = int(np.argmax(W[0, :nf]))
            if frames[j] is not None and W[0, j] > 0.5:
                st.follow = frames[j]
                st.ride = bool(W[0, nf + j] > 0.5)
        return st.orthonormalized()

    def sample_positions(self, n=300):
        """(ts, pos, target) arrays along the whole path, for drawing."""
        if len(self.keyframes) < 2:
            return None
        if self._interp is None:
            self._build()
        ts = self._interp[1]
        tt = np.linspace(ts[0], ts[-1], n)
        dd, T, P, W = self._world(tt)
        return tt, P, T

    def dump_index(self, state):
        if not self.dumps:
            return None
        return int(np.clip(round(state.dump), 0, len(self.dumps) - 1))

    def frame_dumps(self):
        """Dump index used by every frame of the movie (dump is linear between keys)."""
        if not self.keyframes or not self.dumps:
            return np.zeros(0, dtype=int)
        ts = np.array([k.t for k in self.keyframes])
        D = np.array([k.state.dump for k in self.keyframes])
        tt = np.arange(self.n_frames) / self.fps
        return np.clip(np.round(np.interp(tt, ts, D)), 0, len(self.dumps) - 1).astype(int)

    def dump_path(self, state):
        i = self.dump_index(state)
        return None if i is None else self.dumps[i]

    # -- io

    def to_dict(self):
        return {
            "version": 1,
            "fps": self.fps,
            "duration": self.duration,
            "resolution": list(self.resolution),
            "dumps": list(self.dumps),
            "format": self.format,
            "look": dict(self.look),
            "keyframes": [k.to_dict() for k in self.keyframes],
        }

    @staticmethod
    def from_dict(d):
        p = CameraPath(
            keyframes=[Keyframe.from_dict(k) for k in d.get("keyframes", [])],
            fps=float(d.get("fps", 30.0)),
            duration=float(d.get("duration", 10.0)),
            resolution=tuple(d.get("resolution", (1920, 1080))),
            dumps=list(d.get("dumps", [])),
            look=dict(d.get("look", {})),
            format=d.get("format", "phantom"),
        )
        # files from before per-key ease: the global "ease in/out" eased the first key
        old_ease = p.look.pop("ease", None)
        ks = d.get("keyframes", [])
        if old_ease and p.keyframes and not any("ease" in k for k in ks):
            p.sort()
            p.keyframes[0].state.ease = True
        p.sort()
        return p

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(filename):
        with open(filename) as f:
            return CameraPath.from_dict(json.load(f))


def make_orbit(
    start, t0, duration, angle_deg, n_keys=None, axis=None, radius_factor=1.0, dump_end=None, tracks=None
):
    """
    Keyframes turning the camera around its target, starting from state ``start`` at t0.

    axis defaults to the camera up vector; radius_factor < 1 spirals inward
    (like camera_cluster.py); dump_end makes the simulation advance during the orbit.
    """
    if n_keys is None:  # a key every 15 degrees keeps the angular speed even
        n_keys = max(2, int(math.ceil(abs(angle_deg) / 15.0)) + 1)
    if axis is None:
        axis = start.basis()[2]
    axis = normalize(axis)
    keys = []
    for i in range(n_keys):
        f = i / (n_keys - 1)
        s = start.copy()
        a = math.radians(angle_deg) * f
        rel = rotate(start.pos - start.target, axis, a)
        rel *= 1.0 + (radius_factor - 1.0) * f
        s.pos = start.target + rel
        s.up = rotate(start.basis()[2], axis, a)
        s.ease = start.ease if i == 0 else False
        if dump_end is not None:
            s.dump = start.dump + (dump_end - start.dump) * f
            if start.follow is not None and tracks is not None and tracks.has(start.follow):
                # orbit around the followed sink as it moves
                shift = tracks.position(start.follow, s.dump) - tracks.position(start.follow, start.dump)
                s.target = s.target + shift
                s.pos = s.pos + shift  # the orbit is around the sink, in both modes
        keys.append(Keyframe(t0 + duration * f, s))
    return keys
