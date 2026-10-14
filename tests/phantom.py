# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Writer of small synthetic Phantom binary dumps for the tests: the layout read by
cinecam.data.readers.PhantomReader (magic record, file id, 8 header tables, block
table, then per block a tag record and a data record per array).
"""

import struct

import numpy as np

from cinecam.constants import MSUN_G, MYR_S, PC_CM

# data kinds: fortran int, int8, int16, int32, int64, default real, real4, real8
INT1, INT8, DEFAULT_REAL, REAL4, REAL8 = 1, 4, 5, 6, 7


def _rec(f, data):
    f.write(struct.pack("<i", len(data)))
    f.write(data)
    f.write(struct.pack("<i", len(data)))


def _tags(names):
    return b"".join(n.ljust(16).encode() for n in names)


def write_dump(path, time, gas, sinks=None, single=False, massoftype=1e-3, units=(PC_CM, MSUN_G, MYR_S)):
    """
    gas: {"x", "y", "z", "h", "iorig", optional "itype"} arrays; sinks: {"x", "y", "z",
    "m", "h", "vx", "vy", "vz"} arrays or None. single: small dump (default reals in
    single precision). Extra arrays are written that the reader must skip.
    """
    real = "<f4" if single else "<f8"
    kind_dtype = {INT1: "<i1", INT8: "<i8", DEFAULT_REAL: real, REAL4: "<f4", REAL8: "<f8"}
    header = {
        DEFAULT_REAL: [("time", time), ("massoftype", massoftype), ("massoftype", 0.0)],  # tags repeat per type
        REAL8: [("udist", units[0]), ("umass", units[1]), ("utime", units[2])],
    }
    n = len(gas["x"])
    gas_arrays = {
        INT8: [("iorig", gas["iorig"])],
        INT1: [("itype", gas["itype"])] if "itype" in gas else [],
        DEFAULT_REAL: [("x", gas["x"]), ("y", gas["y"]), ("z", gas["z"]), ("vx", np.zeros(n))],
        REAL4: [("h", gas["h"])],
    }
    blocks = [(n, gas_arrays)]
    if sinks is not None:
        ns = len(sinks["m"])
        names = ("x", "y", "z", "m", "h", "spinx", "vx", "vy", "vz")
        blocks.append((ns, {DEFAULT_REAL: [(k, sinks.get(k, np.zeros(ns))) for k in names]}))

    with open(path, "wb") as f:
        if single:
            _rec(f, struct.pack("<ifiii", 60769, 1.0, 690706, 1, 60878))
        else:
            _rec(f, struct.pack("<idiii", 60769, 1.0, 690706, 1, 60878))
        _rec(f, b"FT:Phantom synthetic test dump".ljust(100))
        for kind in range(8):
            items = header.get(kind, [])
            _rec(f, struct.pack("<i", len(items)))
            if items:
                _rec(f, _tags([k for k, _ in items]))
                _rec(f, np.array([v for _, v in items], dtype=kind_dtype[kind]).tobytes())
        _rec(f, struct.pack("<i", len(blocks)))
        for nb, arrays in blocks:
            nums = [len(arrays.get(k, [])) for k in range(8)]
            _rec(f, struct.pack("<q", nb) + struct.pack("<8i", *nums))
        for nb, arrays in blocks:
            for kind in range(8):
                for tag, data in arrays.get(kind, []):
                    _rec(f, tag.ljust(16).encode())
                    _rec(f, np.asarray(data, dtype=kind_dtype[kind]).tobytes())
