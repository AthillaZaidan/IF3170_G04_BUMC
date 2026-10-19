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
    packages = [p for i, p in enumerate(problem.packages)
                if _fits_somehow(problem, i) and p.weight <= t.max_capacity]
    volume = min(t.w * t.l * t.h * problem.truck_count,
                 sum(p.w * p.l * p.h for p in packages))
    capacity = min(t.max_capacity * problem.truck_count, sum(p.weight for p in packages))
    if (capacity == sum(p.weight for p in packages)
            and volume == sum(p.w * p.l * p.h for p in packages)):
        return sum(p.value for p in packages)
    best = np.zeros((capacity + 1, volume + 1), dtype=np.int64)
    for p in packages:
        v = p.w * p.l * p.h
        with_p = best[:-p.weight or None, :-v or None] + p.value
        np.maximum(best[p.weight:, v:], with_p, out=best[p.weight:, v:])
    return int(best[-1, -1])
