"""
Proxy particle samples of the dumps, read in the background and kept in memory.

The instant views of the editor (viewfinder proxy, world view) splat a sample of the gas
particles. Each dump's sample is read directly with the fast reader of the dump format
(data/readers.py, no shamrock), so scrubbing and playing the timeline shows the gas of
the right dump at once, while shamrock only loads a dump for the real renders.

Sample: the n living particles with the lowest rank, the rank being a fixed hash of the
particle id. The same particles are kept from one dump to the next; when one is accreted
the next ranked particle replaces it, so the sample always has n particles and does not
flicker. Each sampled particle carries (living gas mass) / n.
"""

import collections
import threading
import time

import numpy as np

from ..constants import MSUN_G, PC_CM


def particle_rank(ids, seed=0x5EED):
    """Fixed pseudo-random uint64 rank of each particle id (splitmix64)."""
    with np.errstate(over="ignore"):
        z = ids.astype(np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def proxy_sample(gas, n):
    """{"pos" (m,3) float32, "weight", "time", "nalive"} from a reader's read_gas() result."""
    alive = np.flatnonzero(gas["alive"])
    if len(alive) > n:
        ids = gas["ids"][alive] if gas["ids"] is not None else alive
        keep = np.argpartition(particle_rank(ids), n - 1)[:n]
        sel = np.sort(alive[keep])
    else:
        sel = alive
    m = gas["mass"]
    total = float(np.sum(m[alive])) if np.ndim(m) else float(m) * len(alive)
    msun_per_unit = gas["umass"] / MSUN_G
    pc_per_unit = gas["udist"] / PC_CM
    sigma_unit = msun_per_unit / pc_per_unit**2  # code column density -> Msun/pc^2
    return {
        "pos": gas["pos"][sel].astype(np.float32),
        # splat weight such that frame.splat_column gives Msun/pc^2 with code-unit areas
        "weight": total / max(len(sel), 1) * sigma_unit,
        "time": gas["time"],
        "nalive": len(alive),
    }


class ProxyCache:
    """
    LRU cache of proxy samples, filled by a background thread.

    want(paths) gives the dumps to load, most urgent first (it replaces the previous
    list); on_ready(path, error) is called from the worker thread when one is done.
    """

    def __init__(self, reader, n_sample=400_000, budget_mb=1024, on_ready=None):
        self.reader = reader
        self.n_sample = n_sample
        self.budget = budget_mb * 1e6
        self.on_ready = on_ready
        self.entries = collections.OrderedDict()  # path -> sample, least recently used first
        self.errors = {}
        self.queue = []
        self.loading = None
        self.load_time = 0.0
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="proxy-cache", daemon=True)
        self._thread.start()

    @property
    def capacity(self):
        """Number of samples that fit in the memory budget."""
        return max(int(self.budget // (self.n_sample * 12 + 1024)), 2)

    def get(self, path):
        with self._cond:
            e = self.entries.get(path)
            if e is not None:
                self.entries.move_to_end(path)
            return e

    def has(self, path):
        with self._cond:
            return path in self.entries

    def want(self, paths):
        with self._cond:
            seen = set()
            self.queue = [p for p in paths if p and not (p in seen or seen.add(p))][: self.capacity]
            for p in reversed(self.queue):  # keep the wanted ones out of the eviction
                if p in self.entries:
                    self.entries.move_to_end(p)
            self._cond.notify()

    def nbytes(self):
        with self._cond:
            return sum(e["pos"].nbytes for e in self.entries.values())

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()

    def _next(self):
        for p in self.queue:
            if p not in self.entries and p not in self.errors:
                return p
        return None

    def _run(self):
        while True:
            with self._cond:
                while not self._stop and self._next() is None:
                    self._cond.wait()
                if self._stop:
                    return
                path = self._next()
                self.loading = path
            err = None
            try:
                t0 = time.perf_counter()
                sample = proxy_sample(self.reader.read_gas(path), self.n_sample)
                sample["load_time"] = time.perf_counter() - t0
            except Exception as e:  # unreadable dump: remembered, not retried
                sample, err = None, f"{type(e).__name__}: {e}"
            with self._cond:
                self.loading = None
                if sample is None:
                    self.errors[path] = err
                else:
                    self.entries[path] = sample
                    self.load_time = sample["load_time"]
                    while len(self.entries) > self.capacity:
                        self.entries.popitem(last=False)
            if self.on_ready:
                self.on_ready(path, err)
