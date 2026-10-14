import os

import numpy as np
import pytest
from conftest import NDUMP, SINK_M, sink_pos

from cinecam.data.sinktracks import SinkTracks, cache_path


def test_build_shapes_and_alive(tracks):
    assert (tracks.ndump, tracks.nsink) == (NDUMP, 4)
    assert tracks.alive[:, :3].all()
    np.testing.assert_array_equal(tracks.alive[:, 3], [False, False, True, True, True])  # forms at dump 2
    assert tracks.pc_per_unit == 1.0 and tracks.msun_per_unit == 1.0


def test_exact_at_dumps_and_between(tracks):
    """Hermite with the sink velocities: exact at the dumps, and here (straight lines) in between."""
    f = np.linspace(0, NDUMP - 1, 37)
    for s in range(3):
        np.testing.assert_allclose(tracks.track(s, f), [sink_pos(s, d) for d in f], atol=1e-9)
    np.testing.assert_allclose(tracks.position(0, 1.37), sink_pos(0, 1.37), atol=1e-9)


def test_positions_at_matches_track(tracks):
    for d in (0.0, 1.0, 2.5, 3.99):
        np.testing.assert_allclose(tracks.positions_at([0, 1, 2], d), [tracks.position(s, d) for s in range(3)], atol=1e-12)


def test_sink_not_yet_formed_holds_its_first_position(tracks):
    np.testing.assert_allclose(tracks.position(3, 0.0), sink_pos(3, 2), atol=1e-9)
    np.testing.assert_allclose(tracks.position(3, 1.0), sink_pos(3, 2), atol=1e-9)


def test_sinks_at_splits_stars_and_cores(tracks):
    sk = tracks.sinks_at(1.2)  # alive sinks of the nearest dump (1), at their position at 1.2
    np.testing.assert_array_equal(sk["star_id"], [0, 1])
    np.testing.assert_array_equal(sk["core_id"], [2])
    np.testing.assert_allclose(sk["star_pos"], [sink_pos(0, 1.2), sink_pos(1, 1.2)], atol=1e-9)
    np.testing.assert_allclose(sk["core_pos"], [sink_pos(2, 1.2)], atol=1e-9)
    np.testing.assert_allclose(sk["star_mass"], SINK_M[:2])
    np.testing.assert_allclose(sk["core_racc"], [0.02])
    np.testing.assert_array_equal(tracks.sinks_at(3)["star_id"], [0, 1, 3])


def test_describe(tracks):
    assert tracks.describe(0, 1) == "star #0 · 8 M☉"
    assert tracks.describe(2, 1) == "core #2 · 50 M☉"
    assert tracks.describe(3, 0) == "sink #3 (forms at dump 2)"


def test_cache_roundtrip(sim):
    tr = SinkTracks.build(sim.dumps)
    cp = cache_path(sim.dumps)
    assert os.path.exists(cp)
    again = SinkTracks.load(cp)  # float32 cache
    np.testing.assert_allclose(again.track(0, [0.5, 3.2]), tr.track(0, [0.5, 3.2]), rtol=1e-6)
    assert again.units == pytest.approx(tr.units)
