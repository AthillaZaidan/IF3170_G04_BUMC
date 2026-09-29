"""Constraint checks from the spec. Only packages inside the truck are checked.

Each check reports Violation records rather than a bool, so the objective can
count them for its penalty and the demo can show what went wrong.
"""
from dataclasses import dataclass
from itertools import combinations

from src.core.geometry import box, overlaps, rests_on, within_walls
from src.core.models import Problem
from src.core.state import State


@dataclass(frozen=True)
class Violation:
    kind: str  # "wall" | "overlap" | "floating" | "fragile" | "capacity"
    packages: tuple[int, ...]  # package indices involved
    amount: int = 1  # excess weight units for "capacity", otherwise 1


def find_violations(problem: Problem, state: State) -> list[Violation]:
    truck = problem.truck
    inside = [i for i, p in enumerate(state) if p.in_truck]
    boxes = {i: box(problem.packages[i], state[i]) for i in inside}
    found: list[Violation] = []

    for i in inside:
        if not within_walls(boxes[i], truck):
            found.append(Violation("wall", (i,)))

    for i, j in combinations(inside, 2):
        if overlaps(boxes[i], boxes[j]):
            found.append(Violation("overlap", (i, j)))

    # Floating: no point of the base is on the floor or on another package.
    for i in inside:
        if boxes[i][2] > 0 and not any(
            rests_on(boxes[j], boxes[i]) for j in inside if j != i
        ):
            found.append(Violation("floating", (i,)))

    # Fragile packages must not support anything, even a package that also
    # rests on something else ("barang pecah-belah tidak boleh menjadi support").
    for j in inside:
        if problem.packages[j].fragile:
            for i in inside:
                if i != j and rests_on(boxes[j], boxes[i]):
                    found.append(Violation("fragile", (j, i)))

    total_weight = sum(problem.packages[i].weight for i in inside)
    if total_weight > truck.max_capacity:
        found.append(
            Violation("capacity", tuple(inside), total_weight - truck.max_capacity)
        )

    return found


def is_feasible(problem: Problem, state: State) -> bool:
    return not find_violations(problem, state)
