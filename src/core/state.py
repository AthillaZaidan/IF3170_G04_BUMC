"""Search state: where every package is and how it is turned.

A Placement is one gene; a State is one chromosome, a tuple of placements
index-aligned with Problem.packages. Both are immutable, so copying a state is a
tuple copy and a neighbor can never change the state it came from.
"""
from dataclasses import dataclass, replace

from src.core.orientation import DEFAULT_ORIENTATION, Orientation


@dataclass(frozen=True)
class Placement:
    in_truck: bool
    x: int = 0  # left-bottom-back corner, ignored when outside the truck
    y: int = 0
    z: int = 0
    orientation: Orientation = DEFAULT_ORIENTATION


State = tuple[Placement, ...]


def take_out(placement: Placement) -> Placement:
    """Same package, moved out of the truck. Coordinates reset to 0 so equal
    states compare equal regardless of where an outside package used to be."""
    return replace(placement, in_truck=False, x=0, y=0, z=0)
