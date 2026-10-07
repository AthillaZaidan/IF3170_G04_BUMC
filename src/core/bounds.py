import numpy as np

from src.core.models import Problem
from src.core.orientation import ALL_ORIENTATIONS, extents


def _fits_somehow(problem: Problem, i: int) -> bool:
    t = problem.truck
    return any(dx <= t.w and dy <= t.l and dz <= t.h
               for dx, dy, dz in (extents(problem.packages[i], o) for o in ALL_ORIENTATIONS))


# Exact 0/1 knapsack on weight and volume; it ignores geometry, so it is >= the optimum.
def value_upper_bound(problem: Problem) -> int:
    t = problem.truck
    volume = t.w * t.l * t.h
    best = np.zeros((t.max_capacity + 1, volume + 1), dtype=np.int64)
    for i, p in enumerate(problem.packages):
        v = p.w * p.l * p.h
        if not _fits_somehow(problem, i) or p.weight > t.max_capacity or v > volume:
            continue
        with_p = best[:-p.weight or None, :-v or None] + p.value
        np.maximum(best[p.weight:, v:], with_p, out=best[p.weight:, v:])
    return int(best[-1, -1])
