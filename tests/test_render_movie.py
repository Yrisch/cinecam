# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""The batch renderer's frame bookkeeping (mono and stereo), with a stub in place of shamrock."""

import os
import sys

import numpy as np
import pytest

from cinecam import render_movie
from cinecam.path.camera import CamState, project
from cinecam.path.campath import CameraPath, Keyframe


class StubScene:
    path, time_myr, load_time = None, 0.0, 0.0

    def load(self, path):
        new, self.path = path != self.path, path
        return new


class StubRenderer:
    """Records the cameras and exposures it is asked for, returns a blank image."""

    calls = []

    def __init__(self, scene):
        from cinecam.render.stars import StarLight

        self.starlight = StarLight()
        self.tracks = None

    def render(self, state, nx, ny, look, timings, exposure):
        StubRenderer.calls.append((state.copy(), nx, ny, exposure))
        return np.zeros((ny, nx, 3), np.uint8)


@pytest.fixture
def run(sim, tmp_path, monkeypatch):
    monkeypatch.setattr(render_movie, "init_shamrock", lambda cfg: None)
    monkeypatch.setattr(render_movie, "SimScene", StubScene)
    monkeypatch.setattr(render_movie, "FrameRenderer", StubRenderer)
    StubRenderer.calls = []
    p = CameraPath(keyframes=[Keyframe(0, CamState(pos=[0, -3, 1], target=[0, 0, 0], up=[0, 0, 1], dump=0, exposure=0.5)),
                              Keyframe(1, CamState(pos=[1, -3, 1], target=[0, 0, 0], up=[0, 0, 1], dump=4, exposure=0.5))],
                   duration=1, fps=4, resolution=(320, 180), dumps=sim.dumps)  # fmt: skip
    f = tmp_path / "p.json"
    p.save(f)
    out = tmp_path / "frames"

    def go(*opts):
        monkeypatch.setattr(sys, "argv", ["cinecam render", str(f), "--out", str(out), *opts])
        render_movie.main()
        return sorted(os.listdir(out)), p

    return go


def test_mono_frames(run):
    files, p = run("--frames", "1:3")
    assert files == ["frame_00001.png", "frame_00002.png"]
    assert [c[0].shift for c in StubRenderer.calls] == [0.0, 0.0]


def test_stereo_files_alternate_left_and_right(run):
    files, p = run("--frames", "1:3", "--stereo")
    assert files == [f"frame_{j:05d}.png" for j in (2, 3, 4, 5)]
    calls = StubRenderer.calls
    assert len(calls) == 4
    for (left, *_, el), (right, *_, er), i in zip(calls[::2], calls[1::2], (1, 2)):
        mono = p.evaluate(p.frame_time(i))
        r = mono.basis()[1]
        assert (left.pos - mono.pos) @ r < 0 < (right.pos - mono.pos) @ r  # even file: left eye
        assert el == er == pytest.approx(0.5)  # one star exposure for both eyes
        for e in (left, right):
            sx, _, _ = project(e, [mono.target], 320, 180)
            assert abs(sx[0] - 160) <= 0.5
    a, b = left.pos - mono.target, right.pos - mono.target
    assert np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))) == pytest.approx(4.0, abs=0.01)


def test_stereo_angle_and_skip_existing(run):
    run("--frames", "0:1", "--stereo", "2.5")
    left, right = StubRenderer.calls[0][0], StubRenderer.calls[1][0]
    mono_target = np.zeros(3)
    a, b = left.pos - mono_target, right.pos - mono_target
    assert np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))) == pytest.approx(2.5, abs=0.01)
    files, _ = run("--frames", "0:2", "--stereo", "--skip-existing")
    assert files == [f"frame_{j:05d}.png" for j in range(4)]
    assert len(StubRenderer.calls) == 4  # frame 0's two eyes were not rendered again


def test_stereo_refuses_mp4(run):
    with pytest.raises(SystemExit, match="--mp4 is not available with --stereo"):
        run("--stereo", "--mp4", "m.mp4")
