from dataclasses import dataclass, replace

from src.core.orientation import ALL_ORIENTATIONS, DEFAULT_ORIENTATION, Orientation


@dataclass(frozen=True)
class Placement:
    in_truck: bool
    x: int = 0  # left-bottom-back corner, ignored when outside the truck
    y: int = 0
    z: int = 0
    orientation: Orientation = DEFAULT_ORIENTATION
    truck_id: int = 0

    def __post_init__(self):
        if type(self.in_truck) is not bool:
            raise ValueError("in_truck must be a boolean")
        if any(type(v) is not int for v in (self.x, self.y, self.z)):
            raise ValueError("placement coordinates must be integers")
        if not isinstance(self.orientation, tuple) or self.orientation not in ALL_ORIENTATIONS:
            raise ValueError("orientation must be a permutation of x, y, z")


State = tuple[Placement, ...]


def take_out(placement: Placement) -> Placement:
    return replace(placement, in_truck=False, x=0, y=0, z=0, truck_id=0)
