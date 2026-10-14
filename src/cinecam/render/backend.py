# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Shamrock backend for cinecam.

SimScene owns one Phantom dump loaded in a shamrock SPH model and renders
column density maps through a perspective camera (shamrock.math.Camera3d_f64_3).

server_main is the entry point of the render process used by the editor:
shamrock (MPI + SYCL) lives only there so the GUI never blocks on it.
"""

import atexit
import gc
import math
import os
import time
import traceback

import numpy as np

from ..constants import MSUN_G, MYR_S, PC_CM, STAR_MAX_RACC_PC
from ..data.sinktracks import SinkTracks
from ..path.camera import ZNEAR, CamState
from .compose import compose
from .stars import StarLight

_shamrock = None


def init_shamrock(sycl_cfg="0:0", loglevel=0):
    global _shamrock
    if _shamrock is None:
        import shamrock

        _shamrock = shamrock
    if not _shamrock.sys.is_initialized():
        _shamrock.change_loglevel(loglevel)
        _shamrock.sys.init(sycl_cfg)
        atexit.register(_shamrock.sys.close)
    return _shamrock


class SimScene:
    def __init__(self):
        self.sham = init_shamrock()
        self.path = None
        self.ctx = None
        self.model = None

    def load(self, path):
        if path == self.path:
            return False
        sham = self.sham
        self.model = None
        self.ctx = None
        gc.collect()

        t0 = time.perf_counter()
        dump = sham.load_phantom_dump(path)
        ctx = sham.Context()
        ctx.pdata_layout_new()
        model = sham.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
        cfg = model.gen_config_from_phantom_dump(dump, bypass_error=True)
        model.set_solver_config(cfg)
        model.init_scheduler(int(1e8), 1)
        model.init_from_phantom_dump(dump)
        self.ctx, self.model, self.path = ctx, model, path

        u = model.get_units().to_json()
        self.pc_per_unit = u["unit_length"] / PC_CM
        self.msun_per_unit = u["unit_mass"] / MSUN_G
        self.sigma_unit = self.msun_per_unit / self.pc_per_unit**2  # code column -> Msun/pc^2
        self.time = model.get_time()
        self.time_myr = self.time * u["unit_time"] / MYR_S
        self.particle_mass = model.get_particle_mass()
        self.npart = model.get_total_part_count()
        self._load_sinks()
        self.load_time = time.perf_counter() - t0
        return True

    def _load_sinks(self):
        sinks = self.model.get_sinks()
        if sinks:
            pos = np.array([s["pos"] for s in sinks], dtype=float)
            m = np.array([s["mass"] for s in sinks], dtype=float)
            racc = np.array([s["accretion_radius"] for s in sinks], dtype=float)
        else:
            pos, m, racc = np.zeros((0, 3)), np.zeros(0), np.zeros(0)
        alive = m > 0
        is_star = alive & (racc * self.pc_per_unit < STAR_MAX_RACC_PC)
        is_core = alive & ~is_star
        ids = np.arange(len(m))  # index in the dump sink arrays, stable across dumps
        self.sinks = {
            "star_id": ids[is_star],
            "core_id": ids[is_core],
            "star_pos": pos[is_star],
            "star_mass": m[is_star] * self.msun_per_unit,
            "core_pos": pos[is_core],
            "core_mass": m[is_core] * self.msun_per_unit,
            "core_racc": racc[is_core],  # code units
        }

    def gas_sample(self, n=400_000, seed=0):
        xyz = np.asarray(self.ctx.collect_data()["xyz"])
        N = len(xyz)
        if N > n:
            idx = np.random.default_rng(seed).choice(N, n, replace=False)
            xyz = xyz[idx]
        # weight such that splat_column gives Msun/pc^2 with code-unit areas
        w = self.particle_mass * N / len(xyz) * self.sigma_unit
        return xyz.astype(np.float32), w

    def meta(self):
        return {
            "path": self.path,
            "time": self.time,
            "time_myr": self.time_myr,
            "pc_per_unit": self.pc_per_unit,
            "msun_per_unit": self.msun_per_unit,
            "npart": self.npart,
            "load_time": self.load_time,
        }

    # -- rendering

    def column(self, state, nx, ny):
        """Column density [Msun/pc^2] seen by the camera, (ny, nx), top row first."""
        d, r, u = state.basis()
        cam = self.sham.math.Camera3d_f64_3(
            pos=tuple(state.pos),
            dir=tuple(d),
            up=tuple(u),
            nx=int(nx),
            ny=int(ny),
            fovy=math.radians(state.fov),
            znear=ZNEAR,
            zfar=1e30,
        )
        arr = np.asarray(self.model.render_column_integ("rho", "f64", cam.get_rays()))
        return arr.reshape(ny, nx)[::-1] * self.sigma_unit

    def column_to_points(self, origin, pts):
        """Gas column [Msun/pc^2] between origin and each point (two half-line integrals)."""
        if len(pts) == 0:
            return np.zeros(0)
        Ray = self.sham.math.Ray_f64_3
        dirs = pts - origin
        o = tuple(origin)
        rays_cam = [Ray(o, tuple(dv), 0.0) for dv in dirs]
        rays_beyond = [Ray(tuple(p), tuple(dv), 0.0) for p, dv in zip(pts, dirs)]
        a = np.asarray(self.model.render_column_integ("rho", "f64", rays_cam))
        b = np.asarray(self.model.render_column_integ("rho", "f64", rays_beyond))
        return np.maximum(a - b, 0.0) * self.sigma_unit


class FrameRenderer:
    """Full frame (gas + cores + stars + label) from a SimScene, shared by editor and batch."""

    def __init__(self, scene, tracks=None):
        self.scene = scene
        self.starlight = StarLight()
        self._lum_cache = (None, None)
        self.tracks = tracks  # sinktracks.SinkTracks: sinks drawn at their position between dumps

    def sinks_at(self, dump):
        """Sinks of the loaded dump, moved to their smooth position at a fractional dump."""
        sk = self.scene.sinks
        tr = self.tracks
        if tr is None or abs(dump - round(dump)) < 1e-6:
            return sk
        sk = dict(sk)
        for kind in ("star", "core"):
            ids = sk[kind + "_id"]
            ok = ids < tr.nsink
            pos = sk[kind + "_pos"].copy()
            if ok.any():
                pos[ok] = tr.positions_at(ids[ok], dump)
            sk[kind + "_pos"] = pos
        return sk

    def star_lum(self):
        key = self.scene.path
        if self._lum_cache[0] != key:
            self._lum_cache = (key, self.starlight.luminosities(self.scene.sinks["star_mass"])[0])
        return self._lum_cache[1]

    def render(self, state, nx, ny, look, timings=None, exposure=None):
        """exposure: star exposure to use instead of state.exposure (auto exposure), dex."""
        timings = {} if timings is None else timings
        sc = self.scene
        t0 = time.perf_counter()
        sigma = sc.column(state, nx, ny) if look.get("show_gas", True) else np.zeros((ny, nx))
        timings["gas"] = time.perf_counter() - t0

        sinks = self.sinks_at(state.dump)
        star_layer = None
        if look.get("show_stars", True) and len(sinks["star_pos"]):
            t0 = time.perf_counter()
            pos = sinks["star_pos"]
            sig = None
            if state.extinction > 0:
                # only stars in front of the camera matter
                d = state.basis()[0]
                front = (pos - state.pos) @ d > ZNEAR
                sig = np.zeros(len(pos))
                sig[front] = sc.column_to_points(state.pos, pos[front])
            star_layer = self.starlight.render(state, pos, self.star_lum(), sig, nx, ny, look, sc.pc_per_unit, exposure)
            timings["stars"] = time.perf_counter() - t0

        label = f"t = {sc.time_myr:.2f} Myr" if look.get("time_label", True) else None
        return compose(sigma, state, look, sinks, star_layer, label)


# ---------------------------------------------------------------------- render server


def server_main(conn, sycl_cfg="0:0", sample_n=400_000):
    """
    Message loop of the render process.

    requests:  {"cmd": "render", "id", "dump", "state", "nx", "ny", "look", "exposure" (optional)}
               {"cmd": "tracks", "path"}  (sink tracks cache, see data/sinktracks.py)
               {"cmd": "quit"}
    replies:   {"type": "loaded", "meta", "sample", "weight", "sinks"}
               {"type": "frame", "id", "rgb", "info"}
               {"type": "status", "msg"} / {"type": "error", "id", "msg"}

    Pending render requests are collapsed: only the most recent one is rendered,
    except those flagged "keep" (export of a full resolution frame).
    """
    try:
        init_shamrock(sycl_cfg)
        scene = SimScene()
        renderer = FrameRenderer(scene)
    except Exception:
        conn.send({"type": "error", "id": None, "msg": traceback.format_exc()})
        return
    conn.send({"type": "status", "msg": "shamrock ready"})

    while True:
        msgs = [conn.recv()]
        while conn.poll():
            msgs.append(conn.recv())
        if any(m["cmd"] == "quit" for m in msgs):
            break
        for m in msgs:
            if m["cmd"] == "tracks":
                try:
                    renderer.tracks = SinkTracks.load(m["path"]) if m["path"] else None
                except Exception:
                    conn.send({"type": "error", "id": None, "msg": traceback.format_exc()})
        renders = [m for m in msgs if m["cmd"] == "render"]
        todo = [m for m in renders if m.get("keep")]
        latest = [m for m in renders if not m.get("keep")]
        if latest:
            todo.append(latest[-1])

        for m in todo:
            try:
                if m["dump"] and m["dump"] != scene.path:
                    conn.send({"type": "status", "msg": f"loading {os.path.basename(m['dump'])} ..."})
                    scene.load(m["dump"])
                    xyz, w = scene.gas_sample(sample_n)
                    conn.send(
                        {"type": "loaded", "meta": scene.meta(), "sample": xyz, "weight": w, "sinks": scene.sinks}
                    )
                if scene.model is None:
                    raise RuntimeError("no dump loaded")
                state = CamState.from_dict(m["state"])
                timings = {}
                t0 = time.perf_counter()
                rgb = renderer.render(state, m["nx"], m["ny"], m["look"], timings, m.get("exposure"))
                timings["total"] = time.perf_counter() - t0
                conn.send({"type": "frame", "id": m["id"], "rgb": rgb, "info": timings, "keep": m.get("keep")})
            except Exception:
                conn.send({"type": "error", "id": m.get("id"), "msg": traceback.format_exc()})
