"""Repair heuristic: turn any state into a feasible one.

It only takes packages out of the truck or lowers them; it never adds one, so
it cleans up a state rather than searching on its own. Deterministic: the same
input always gives the same output.

Steps:
1. Walls: packages sticking out of the truck are taken out.
2. Capacity: while overweight, take out the package with the lowest
   value/weight ratio. Done before geometry because geometry only removes.
3. Geometry + gravity: go bottom-up (z ascending, then value descending).
   A package that overlaps an already accepted one is taken out. Otherwise it
   drops onto the highest accepted top under its footprint (or the floor). If
   anything it lands on is fragile, it is taken out.
"""
from dataclasses import replace

from src.core.geometry import Box, box, footprint_overlaps, overlaps, within_walls
from src.core.models import Problem
from src.core.state import State, take_out


def repair(problem: Problem, state: State) -> State:
    if len(state) != len(problem.packages):
        raise ValueError("state must contain one placement per package")
    state = tuple(
        take_out(p) if p.in_truck and (
            not isinstance(p.truck_id, int) or isinstance(p.truck_id, bool)
            or not 0 <= p.truck_id < problem.truck_count
        ) else p
        for p in state
    )
    for truck_id in range(problem.truck_count):
        state = _repair_truck(problem, state, truck_id)
    return state


def _repair_truck(problem: Problem, state: State, truck_id: int) -> State:
    packages = problem.packages
    truck = problem.truck
    placements = list(state)

    def inside() -> list[int]:
        return [i for i, p in enumerate(placements) if p.in_truck and p.truck_id == truck_id]

    # 1. Walls
    for i in inside():
        if not within_walls(box(packages[i], placements[i]), truck):
            placements[i] = take_out(placements[i])

    # 2. Capacity
    total_weight = sum(packages[i].weight for i in inside())
    by_ratio = sorted(inside(), key=lambda i: (packages[i].value / packages[i].weight, i))
    for i in by_ratio:
        if total_weight <= truck.max_capacity:
            break
        placements[i] = take_out(placements[i])
        total_weight -= packages[i].weight

    # 3. Geometry + gravity
    order = sorted(inside(), key=lambda i: (placements[i].z, -packages[i].value, i))
    accepted: list[tuple[int, Box]] = []
    for i in order:
        b = box(packages[i], placements[i])
        if any(overlaps(b, other) for _, other in accepted):
            placements[i] = take_out(placements[i])
            continue
        below = [(j, other) for j, other in accepted
                 if footprint_overlaps(b, other) and other[5] <= b[2]]
        landing_z = max((other[5] for _, other in below), default=0)
        supporters = [j for j, other in below if other[5] == landing_z and landing_z > 0]
        if any(packages[j].fragile for j in supporters):
            placements[i] = take_out(placements[i])
            continue
        placements[i] = replace(placements[i], z=landing_z)
        accepted.append((i, box(packages[i], placements[i])))

    return tuple(placements)
