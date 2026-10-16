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
    return replace(placement, in_truck=False, x=0, y=0, z=0)
