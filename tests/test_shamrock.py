# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Tests against shamrock and a real dump, skipped unless both are available:

    CINECAM_TEST_DUMP=/path/to/phantom_dump ./cinecam test
"""

import multiprocessing as mp
import os

import numpy as np
import pytest

DUMP = os.environ.get("CINECAM_TEST_DUMP")
pytestmark = [
    pytest.mark.shamrock,
    pytest.mark.skipif(not DUMP or not os.path.exists(DUMP), reason="set CINECAM_TEST_DUMP to a Phantom dump"),
]


@pytest.fixture(scope="module")
def scene():
    pytest.importorskip("shamrock")
    from cinecam.render.backend import SimScene, init_shamrock

    init_shamrock("0:0")
    sc = SimScene()
    sc.load(DUMP)
    return sc


def test_fast_reader_matches_shamrock(scene):
    from cinecam.data.readers import get_reader

    rd = get_reader("phantom")
    s = rd.read_sinks(DUMP)
    alive = s["mass"] > 0
    sk = scene.sinks
    ids = np.concatenate([sk["star_id"], sk["core_id"]])
    np.testing.assert_array_equal(np.sort(ids), np.flatnonzero(alive))
    np.testing.assert_allclose(s["pos"][sk["star_id"]], sk["star_pos"], rtol=1e-12)
    np.testing.assert_allclose(s["racc"][sk["core_id"]], sk["core_racc"], rtol=1e-12)
    g = rd.read_gas(DUMP)
    assert g["alive"].sum() == scene.npart


def test_frame_render(scene):
    from cinecam.path.camera import CamState
    from cinecam.path.campath import DEFAULT_LOOK
    from cinecam.render.backend import FrameRenderer

    c = np.median(scene.sinks["star_pos"], axis=0) if len(scene.sinks["star_pos"]) else np.zeros(3)
    st = CamState(pos=c + [0, -30, 8], target=c, up=[0, 0, 1])
    tm = {}
    img = FrameRenderer(scene).render(st, 160, 90, dict(DEFAULT_LOOK), tm)
    assert img.shape == (90, 160, 3) and img.dtype == np.uint8 and img.max() > 0
    assert tm["gas"] > 0


def test_render_process_roundtrip():
    """The editor's render process: spawned from the package, answers a render request."""
    pytest.importorskip("shamrock")
    from cinecam.path.camera import CamState
    from cinecam.path.campath import DEFAULT_LOOK
    from cinecam.render.backend import server_main

    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    proc = ctx.Process(target=server_main, args=(child, "0:0", 10_000), daemon=True)
    proc.start()
    try:
        st = CamState(pos=[0, -30, 8], target=[0, 0, 0], up=[0, 0, 1])
        conn.send({"cmd": "render", "id": 1, "dump": DUMP, "state": st.to_dict(), "nx": 64, "ny": 36, "look": dict(DEFAULT_LOOK)})
        got = {}
        while "frame" not in got and "error" not in got:
            assert conn.poll(300), "no answer from the render process"
            m = conn.recv()
            got[m["type"]] = m
        assert "error" not in got, got.get("error", {}).get("msg")
        assert got["loaded"]["sample"].shape == (10_000, 3)
        assert got["frame"]["rgb"].shape == (36, 64, 3)
        conn.send({"cmd": "quit"})
        proc.join(30)
    finally:
        if proc.is_alive():
            proc.terminate()


def test_lens_shift_of_the_gas_render_matches_project(scene):
    """A stereo eye's gas render (wider, cropped) is the image translated like project() moves points."""
    from cinecam.path.camera import CamState

    st = CamState(pos=[0, -30, 8], target=[0, 0, 0], up=[0, 0, 1])
    a = scene.column(st, 160, 90)
    st.shift = 2 * 6 / 160  # 6 px: what was at column j + 6 is now at column j
    b = scene.column(st, 160, 90)
    np.testing.assert_allclose(b[:, :-6], a[:, 6:], rtol=1e-6, atol=1e-12 * a.max())


def test_stereo_movie_render(tmp_path):
    import subprocess
    import sys

    from cinecam.path.camera import CamState
    from cinecam.path.campath import CameraPath, Keyframe

    p = CameraPath(keyframes=[Keyframe(0, CamState(pos=[0, -30, 8], target=[0, 0, 0], up=[0, 0, 1])),
                              Keyframe(1, CamState(pos=[5, -30, 8], target=[0, 0, 0], up=[0, 0, 1]))],
                   duration=1, fps=2, dumps=[DUMP])  # fmt: skip
    f = tmp_path / "p.json"
    p.save(f)
    out = tmp_path / "frames"
    cmd = [sys.executable, "-m", "cinecam.render_movie", str(f), "--out", str(out), "--scale", "0.05", "--frames", "1:3", "--stereo"]
    subprocess.run(cmd, check=True, capture_output=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert sorted(os.listdir(out)) == [f"frame_{j:05d}.png" for j in (2, 3, 4, 5)]
