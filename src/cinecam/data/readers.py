# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Fast, partial readers of simulation dumps, for what cinecam needs without a full load:

* ``read_sinks(path)``: the sink particles (trajectories for cameras following a sink),
* ``read_gas(path)``: the gas particle positions (instant proxy views).

The real renders do not use these: they load the dump in shamrock (render/backend.py).

A reader is a subclass of DumpReader registered in READERS under its format name
(``--format`` on the command line, ``"format"`` in the camera path file). To support
another code (gadget, shamrock dumps, ...), implement the two methods and register it.
"""

import struct

import numpy as np


class DumpReader:
    """
    Interface of a fast dump reader. Lengths, masses, times in code units, plus the
    code units in cgs (``udist`` [cm], ``umass`` [g], ``utime`` [s]).
    """

    name = None

    def read_sinks(self, path):
        """
        {"time", "udist", "umass", "utime",
         "pos" (n,3), "vel" (n,3), "mass" (n,), "racc" (n,)}

        A sink must keep its index in these arrays from one dump to the next (sinks that
        disappear keep their slot with a mass <= 0); cameras follow sinks by index.
        """
        raise NotImplementedError

    def read_gas(self, path):
        """
        {"time", "udist", "umass", "utime",
         "pos" (n,3) float array of every particle slot,
         "alive" (n,) bool (False for accreted / dead particles),
         "ids" (n,) int64 permanent particle ids, or None (then the slot index is used),
         "mass" particle mass (scalar or (n,) array)}
        """
        raise NotImplementedError


# ---------------------------------------------------------------------- phantom


# numpy types of the 8 Phantom data kinds: fortran int, int8, int16, int32, int64, default real, real4, real8
_PHANTOM_KINDS = ("<i4", "<i1", "<i2", "<i4", "<i8", "<f8", "<f4", "<f8")


class _FortranFile:
    def __init__(self, f):
        self.f = f

    def record_length(self):
        n = struct.unpack("<i", self.f.read(4))[0]
        self.f.seek(-4, 1)
        return n

    def read(self):
        (n,) = struct.unpack("<i", self.f.read(4))
        data = self.f.read(n)
        (n2,) = struct.unpack("<i", self.f.read(4))
        if n2 != n:
            raise IOError("corrupted fortran record")
        return data

    def skip(self):
        (n,) = struct.unpack("<i", self.f.read(4))
        self.f.seek(n + 4, 1)


class PhantomReader(DumpReader):
    """
    Phantom binary dumps (full and small). Layout: magic numbers, file id, 8 header
    tables (one per data kind: tags then values), then blocks (gas, sinks, ...) whose
    arrays are each written as a tag record followed by a data record, so unneeded
    arrays are skipped with a seek.
    """

    name = "phantom"

    def _open(self, fh):
        """Parse up to the block table. Returns (fortran file, header, [(n, nums)], kinds)."""
        ff = _FortranFile(fh)
        single = ff.record_length() == 4 * 4 + 4  # small dumps: default reals in single precision
        kinds = list(_PHANTOM_KINDS)
        if single:
            kinds[5] = "<f4"
        ff.read()  # magic numbers
        ff.read()  # file id
        header = {}  # tag -> list of values (some tags repeat, e.g. massoftype per particle type)
        for dt in kinds:
            (nvars,) = struct.unpack("<i", ff.read())
            if nvars == 0:
                continue
            tags = ff.read()
            vals = np.frombuffer(ff.read(), dtype=dt)
            for i in range(nvars):
                header.setdefault(tags[16 * i : 16 * (i + 1)].decode().strip(), []).append(vals[i])
        (nblocks,) = struct.unpack("<i", ff.read())
        blocks = []
        for _ in range(nblocks):
            rec = ff.read()
            blocks.append((struct.unpack("<q", rec[:8])[0], struct.unpack("<8i", rec[8:40])))
        return ff, header, blocks, kinds

    def _read_block(self, ff, n, nums, kinds, wanted):
        """Arrays of one block whose tag is in wanted (the others are skipped)."""
        arrays = {}
        for k, num in enumerate(nums):
            for _ in range(num):
                tag = ff.read().decode().strip()
                if tag in wanted and tag not in arrays:
                    arrays[tag] = np.frombuffer(ff.read(), dtype=kinds[k])
                else:
                    ff.skip()
        return arrays

    @staticmethod
    def _units(header):
        first = lambda k, d: float(header[k][0]) if k in header else d  # noqa: E731
        return {
            "time": first("time", 0.0),
            "udist": first("udist", 1.0),
            "umass": first("umass", 1.0),
            "utime": first("utime", 1.0),
        }

    def read_sinks(self, path):
        with open(path, "rb") as fh:
            ff, header, blocks, kinds = self._open(fh)
            sink = None
            for n, nums in blocks:
                # the sink block: few particles, with a mass and an accretion radius per particle
                wanted = ("x", "y", "z", "m", "h", "vx", "vy", "vz") if n < 100_000 else ()
                arrays = self._read_block(ff, n, nums, kinds, wanted)
                if {"x", "y", "z", "m", "h"} <= arrays.keys():
                    sink = {k: v.astype(float) for k, v in arrays.items()}
                    break  # nothing needed after the sink block
        out = self._units(header)
        if sink is None:
            z = np.zeros(0)
            out.update(pos=np.zeros((0, 3)), vel=np.zeros((0, 3)), mass=z, racc=z.copy())
            return out
        n = len(sink["x"])
        vel = np.stack([sink["vx"], sink["vy"], sink["vz"]], axis=1) if {"vx", "vy", "vz"} <= sink.keys() else np.zeros((n, 3))
        out.update(
            pos=np.stack([sink["x"], sink["y"], sink["z"]], axis=1),
            vel=vel,
            mass=sink["m"].copy(),
            racc=sink["h"].copy(),
        )
        return out

    def read_gas(self, path):
        with open(path, "rb") as fh:
            ff, header, blocks, kinds = self._open(fh)
            n, nums = blocks[0]  # gas (and other SPH particles) come first
            a = self._read_block(ff, n, nums, kinds, ("x", "y", "z", "h", "iorig", "itype"))
        pos = np.stack([a["x"], a["y"], a["z"]], axis=1)
        alive = a["h"] > 0  # Phantom marks accreted / dead particles with h <= 0
        if "itype" in a:  # several particle types: keep the gas (itype 1)
            alive &= a["itype"] == 1
        out = self._units(header)
        out.update(
            pos=pos,
            alive=alive,
            ids=a["iorig"].astype(np.int64) if "iorig" in a else None,
            mass=float(header["massoftype"][0]) if "massoftype" in header else 1.0,
        )
        return out


# ---------------------------------------------------------------------- registry

READERS = {PhantomReader.name: PhantomReader}
DEFAULT_FORMAT = "phantom"


def get_reader(fmt=None):
    fmt = fmt or DEFAULT_FORMAT
    try:
        return READERS[fmt]()
    except KeyError:
        raise ValueError(f"unknown dump format {fmt!r} (known: {', '.join(READERS)})") from None
