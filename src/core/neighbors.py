"""Neighbor moves from the spec, shared by Hill-Climbing, SA and GA mutation.

Every function is pure: it returns a new State and never changes its input.
Allowed moves (spec):
1. swap two packages (inside or outside the truck),
2. move a package to another coordinate (including into or out of the truck,
   per the course QnA),
3. rotate a package 90 degrees about x, y or z.
"""
import random
from collections.abc import Iterator
from dataclasses import dataclass, replace

from src.core.geometry import box
from src.core.models import Problem
from src.core.orientation import AXES, extents
from src.core.orientation import rotate as rotate_orientation
from src.core.state import State, take_out

OUTSIDE = None  # move target meaning "out of the truck"


@dataclass(frozen=True)
class Move:
    kind: str  # "move" | "rotate" | "swap"
    i: int
    j: int | None = None
    target: tuple[int, int, int] | None = None
    axis: str | None = None


def _with(state: State, i: int, placement) -> State:
    return state[:i] + (placement,) + state[i + 1:]


def move(state: State, i: int, target: tuple[int, int, int] | None) -> State:
    if target is OUTSIDE:
        return _with(state, i, take_out(state[i]))
    x, y, z = target
    return _with(state, i, replace(state[i], in_truck=True, x=x, y=y, z=z))


def rotate(state: State, i: int, axis: str) -> State:
    """Turn package i 90 degrees about `axis`; its corner stays in place."""
    p = state[i]
    return _with(state, i, replace(p, orientation=rotate_orientation(p.orientation, axis)))


def swap(state: State, i: int, j: int) -> State:
    """Packages i and j trade positions; each keeps its own orientation."""
    a, b = state[i], state[j]
    new_a = replace(a, in_truck=b.in_truck, x=b.x, y=b.y, z=b.z)
    new_b = replace(b, in_truck=a.in_truck, x=a.x, y=a.y, z=a.z)
    if not new_a.in_truck:
        new_a = take_out(new_a)
    if not new_b.in_truck:
        new_b = take_out(new_b)
    return _with(_with(state, i, new_a), j, new_b)


def candidate_positions(problem: Problem, state: State, i: int) -> list[tuple[int, int, int]]:
    """Extreme points where package i could go: the origin plus the points
    right of, in front of and on top of every package in the truck, keeping
    only those where package i stays inside the walls.

    Adapted from the extreme-point idea in T. G. Crainic, G. Perboli and
    R. Tadei, "Extreme Point-Based Heuristics for Three-Dimensional Bin
    Packing", INFORMS Journal on Computing 20(3), 2008.
    """
    truck = problem.truck
    dx, dy, dz = extents(problem.packages[i], state[i].orientation)
    points = {(0, 0, 0)}
    for j, p in enumerate(state):
        if p.in_truck and j != i:
            x0, y0, z0, x1, y1, z1 = box(problem.packages[j], p)
            points.update({(x1, y0, z0), (x0, y1, z0), (x0, y0, z1)})
    return sorted(
        (x, y, z) for x, y, z in points
        if x + dx <= truck.w and y + dy <= truck.l and z + dz <= truck.h
    )


def random_position(problem: Problem, state: State, i: int,
                    rng: random.Random) -> tuple[int, int, int] | None:
    """Half the time an extreme point, otherwise a uniform in-bounds corner.
    None when package i does not fit in the truck in its orientation."""
    if rng.random() < 0.5:
        candidates = candidate_positions(problem, state, i)
        if candidates:
            return rng.choice(candidates)
    truck = problem.truck
    dx, dy, dz = extents(problem.packages[i], state[i].orientation)
    if dx > truck.w or dy > truck.l or dz > truck.h:
        return None
    return (rng.randint(0, truck.w - dx), rng.randint(0, truck.l - dy),
            rng.randint(0, truck.h - dz))


def random_neighbor(problem: Problem, state: State,
                    rng: random.Random) -> tuple[State, Move]:
    """One random successor (stochastic HC, SA, GA mutation).

    The move type is uniform among move/rotate/swap. For "move", an inside
    package goes out with probability 0.2, otherwise to a random_position.
    """
    n = len(state)
    kind = rng.choice(("move", "rotate", "swap") if n >= 2 else ("move", "rotate"))
    if kind == "rotate":
        i, axis = rng.randrange(n), rng.choice(AXES)
        return rotate(state, i, axis), Move("rotate", i, axis=axis)
    if kind == "swap":
        i, j = rng.sample(range(n), 2)
        return swap(state, i, j), Move("swap", i, j=j)
    i = rng.randrange(n)
    if state[i].in_truck and rng.random() < 0.2:
        target = OUTSIDE
    else:
        target = random_position(problem, state, i, rng)
    return move(state, i, target), Move("move", i, target=target)


def sample_neighbors(problem: Problem, state: State, k: int,
                     rng: random.Random) -> list[tuple[State, Move]]:
    """k random successors, for a practical steepest-ascent variant."""
    return [random_neighbor(problem, state, rng) for _ in range(k)]


def all_neighbors(problem: Problem, state: State) -> Iterator[tuple[State, Move]]:
    """Every successor, as in the slides' steepest ascent: all rotations, all
    swaps that change something, and every move out of the truck or to an
    extreme point. Slow on big instances (thousands of neighbors)."""
    n = len(state)
    for i in range(n):
        for axis in AXES:
            yield rotate(state, i, axis), Move("rotate", i, axis=axis)
    for i in range(n):
        for j in range(i + 1, n):
            if state[i].in_truck or state[j].in_truck:
                yield swap(state, i, j), Move("swap", i, j=j)
    for i in range(n):
        if state[i].in_truck:
            yield move(state, i, OUTSIDE), Move("move", i, target=OUTSIDE)
        current = (state[i].x, state[i].y, state[i].z) if state[i].in_truck else None
        for target in candidate_positions(problem, state, i):
            if target != current:
                yield move(state, i, target), Move("move", i, target=target)
