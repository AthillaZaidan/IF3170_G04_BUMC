"""Random initial state (the spec requires a random start).

Positions are uniform inside the walls, so the state is usually infeasible
(overlapping, floating). That is expected: GA repairs it, HC/SA may penalize it.
"""
import random

from src.core.models import Problem
from src.core.orientation import ALL_ORIENTATIONS, extents
from src.core.state import Placement, State


def random_state(problem: Problem, rng: random.Random, p_in: float = 0.5) -> State:
    truck = problem.truck
    placements = []
    for package in problem.packages:
        orientation = rng.choice(ALL_ORIENTATIONS)
        dx, dy, dz = extents(package, orientation)
        fits = dx <= truck.w and dy <= truck.l and dz <= truck.h
        if fits and rng.random() < p_in:
            placements.append(Placement(
                True,
                rng.randint(0, truck.w - dx),
                rng.randint(0, truck.l - dy),
                rng.randint(0, truck.h - dz),
                orientation,
            ))
        else:
            placements.append(Placement(False, orientation=orientation))
    return tuple(placements)
