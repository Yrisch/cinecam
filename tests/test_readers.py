import numpy as np
import pytest
from conftest import NGAS, SINK_M, SINK_RACC, SINK_V, sink_pos
from phantom import write_dump

from cinecam.constants import MSUN_G, PC_CM
from cinecam.data.readers import READERS, PhantomReader, get_reader


def test_registry():
    assert isinstance(get_reader(), PhantomReader)
    assert isinstance(get_reader("phantom"), PhantomReader)
    assert "phantom" in READERS
    with pytest.raises(ValueError, match="unknown dump format 'gadget'.*phantom"):
        get_reader("gadget")


def test_read_sinks(sim):
    r = get_reader().read_sinks(sim.dumps[2])
    assert r["time"] == pytest.approx(0.2)
    assert r["udist"] == PC_CM and r["umass"] == MSUN_G
    assert len(r["mass"]) == 4
    np.testing.assert_allclose(r["pos"], [sink_pos(s, 2) for s in range(4)])
    np.testing.assert_allclose(r["vel"], SINK_V)
    np.testing.assert_allclose(r["mass"], SINK_M)
    np.testing.assert_allclose(r["racc"], SINK_RACC)


def test_read_gas(sim):
    g = get_reader().read_gas(sim.dumps[1])
    assert g["pos"].shape == (NGAS, 3)
    np.testing.assert_allclose(g["pos"], sim.pos)
    np.testing.assert_array_equal(g["ids"], sim.iorig)
    assert g["mass"] == pytest.approx(1e-3)  # first massoftype (gas), not the repeated ones
    expected = np.ones(NGAS, bool)
    expected[sim.dead[1]] = False
    np.testing.assert_array_equal(g["alive"], expected)


def test_small_dump_single_precision(sim):
    """The last synthetic dump is written with single precision default reals."""
    g = get_reader().read_gas(sim.dumps[-1])
    np.testing.assert_allclose(g["pos"], sim.pos, rtol=1e-6)
    s = get_reader().read_sinks(sim.dumps[-1])
    np.testing.assert_allclose(s["pos"], [sink_pos(k, 4) for k in range(4)], rtol=1e-6)


def test_itype_keeps_gas_only(tmp_path):
    n = 10
    gas = {"x": np.arange(n), "y": np.zeros(n), "z": np.zeros(n), "h": np.ones(n, np.float32),
           "iorig": np.arange(n) + 1, "itype": np.array([1] * 7 + [2] * 3)}  # fmt: skip
    f = tmp_path / "dump"
    write_dump(f, 0.0, gas)
    g = get_reader().read_gas(f)
    np.testing.assert_array_equal(g["alive"], [True] * 7 + [False] * 3)
    s = get_reader().read_sinks(f)  # no sink block
    assert s["pos"].shape == (0, 3) and len(s["mass"]) == 0
