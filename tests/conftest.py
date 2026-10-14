import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from phantom import write_dump  # noqa: E402

# synthetic simulation: code units are pc, Msun, Myr
NDUMP = 5
DT = 0.1  # code time between dumps
NGAS = 4000
SINK_P0 = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [-0.5, 0.4, 0.0], [0.2, 0.2, 0.2]])
SINK_V = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.3, 0.1], [0.0, 0.1, 0.0]])
SINK_M = np.array([8.0, 1.0, 50.0, 0.5])
SINK_RACC = np.array([1e-4, 1e-4, 0.02, 1e-4])  # pc: stars below STAR_MAX_RACC_PC (0.01), sink 2 is a core
SINK_BORN = np.array([0, 0, 0, 2])  # sink 3 forms at dump 2


def sink_pos(s, dump):
    """Exact position of synthetic sink s at a (fractional) dump: straight lines."""
    return SINK_P0[s] + SINK_V[s] * DT * dump


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture(scope="session")
def sim(tmp_path_factory):
    """Synthetic Phantom dumps: a gas ball losing particles to accretion, 3 then 4 moving sinks."""
    d = tmp_path_factory.mktemp("sim")
    rng = np.random.default_rng(42)
    pos = rng.normal(scale=0.5, size=(NGAS, 3))
    iorig = rng.permutation(NGAS).astype(np.int64) + 1  # particle ids, not in slot order
    accretion_order = rng.permutation(NGAS)
    dumps, dead = [], []
    for i in range(NDUMP):
        h = np.full(NGAS, 0.05, dtype=np.float32)
        gone = accretion_order[: 150 * i]
        h[gone] = -h[gone]  # accreted particles: h <= 0
        alive_sinks = SINK_BORN <= i
        ns = int(alive_sinks.sum())
        p = np.array([sink_pos(s, i) for s in range(ns)])
        sinks = {
            "x": p[:, 0], "y": p[:, 1], "z": p[:, 2],
            "m": SINK_M[:ns], "h": SINK_RACC[:ns],
            "vx": SINK_V[:ns, 0], "vy": SINK_V[:ns, 1], "vz": SINK_V[:ns, 2],
        }  # fmt: skip
        gas = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "h": h, "iorig": iorig}
        f = str(d / f"sim_{i:05d}")
        write_dump(f, DT * i, gas, sinks, single=(i == NDUMP - 1))  # the last one is a small dump
        dumps.append(f)
        dead.append(gone)
    return types.SimpleNamespace(dumps=dumps, pos=pos, iorig=iorig, dead=dead, dir=d)


@pytest.fixture
def tracks(sim):
    from cinecam.data.sinktracks import SinkTracks

    return SinkTracks.build(sim.dumps)
