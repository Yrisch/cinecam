# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

import threading

import numpy as np
import pytest
from conftest import NGAS

from cinecam.data.proxycache import ProxyCache, particle_rank, proxy_sample
from cinecam.data.readers import get_reader

N = 1000


def _ids(sim, sample_pos):
    """Particle ids of the sampled positions (positions are unique in the synthetic dumps)."""
    lut = {tuple(p): i for p, i in zip(sim.pos.astype(np.float32), sim.iorig)}
    return {lut[tuple(p)] for p in sample_pos}


def test_rank_is_fixed():
    ids = np.arange(100)
    np.testing.assert_array_equal(particle_rank(ids), particle_rank(ids.copy()))
    assert len(set(particle_rank(ids).tolist())) == 100


def test_sample_size_and_weight(sim):
    g = get_reader().read_gas(sim.dumps[2])
    s = proxy_sample(g, N)
    nalive = NGAS - len(sim.dead[2])
    assert s["pos"].shape == (N, 3) and s["pos"].dtype == np.float32
    assert s["nalive"] == nalive
    # each sampled particle carries (living gas mass) / n, in Msun/pc^2 splat units (code units are pc, Msun)
    assert s["weight"] == pytest.approx(1e-3 * nalive / N)


def test_sample_is_stable_and_replenished(sim):
    """Same particles from one dump to the next; accreted ones are replaced, the size stays n."""
    rd = get_reader()
    a = _ids(sim, proxy_sample(rd.read_gas(sim.dumps[0]), N)["pos"])
    b = _ids(sim, proxy_sample(rd.read_gas(sim.dumps[3]), N)["pos"])
    assert len(a) == len(b) == N
    dead = set(sim.iorig[sim.dead[3]].tolist())
    assert a - dead <= b  # every survivor is still in the sample
    assert not (b & dead)  # no accreted particle
    assert len(b - a) == len(a & dead)  # each accreted one replaced by exactly one newcomer


def test_cache_loads_wanted_dumps(sim):
    done = threading.Event()
    got = []

    def ready(path, err):
        got.append((path, err))
        if len(got) == 2:
            done.set()

    pc = ProxyCache(get_reader(), N, budget_mb=100, on_ready=ready)
    try:
        pc.want([sim.dumps[1], sim.dumps[0]])
        assert done.wait(20)
        assert [p for p, _ in got] == [sim.dumps[1], sim.dumps[0]]  # most urgent first
        assert pc.has(sim.dumps[0]) and pc.get(sim.dumps[1])["pos"].shape == (N, 3)
        assert pc.nbytes() == 2 * N * 12
    finally:
        pc.close()


def test_cache_evicts_least_recently_used(sim):
    ev = threading.Event()
    pc = ProxyCache(get_reader(), N, budget_mb=0, on_ready=lambda p, e: ev.set())  # capacity: 2 samples
    try:
        assert pc.capacity == 2
        for d in sim.dumps[:3]:
            ev.clear()
            pc.want([d])
            assert ev.wait(20)
        assert not pc.has(sim.dumps[0]) and pc.has(sim.dumps[1]) and pc.has(sim.dumps[2])
    finally:
        pc.close()


def test_cache_remembers_unreadable_dumps(tmp_path):
    ev = threading.Event()
    bad = str(tmp_path / "missing_dump")
    pc = ProxyCache(get_reader(), N, on_ready=lambda p, e: ev.set())
    try:
        pc.want([bad])
        assert ev.wait(20)
        assert bad in pc.errors and "FileNotFoundError" in pc.errors[bad]
        assert not pc.has(bad)
    finally:
        pc.close()
