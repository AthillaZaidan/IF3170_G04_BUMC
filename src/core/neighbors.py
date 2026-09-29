"""Neighbor moves from the spec, shared by Hill-Climbing, SA and GA mutation.

Every function is pure: it returns a new State and never changes its input.
Allowed moves (spec):
1. swap two packages (inside or outside the truck),
2. move a package to another coordinate (including into or out of the truck,
   per the course QnA),
3. rotate a package 90 degrees about x, y or z.
"""
from dataclasses import dataclass, replace

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
