"""Static problem data: packages and the truck container.

These never change during a search. Where a package sits and how it is turned
is part of the search state (see state.py), not of the package itself.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Package:
    id: str
    w: int
    l: int
    h: int
    value: int
    weight: int
    fragile: bool
    eta: int


@dataclass(frozen=True)
class Truck:
    w: int
    l: int
    h: int
    max_capacity: int


@dataclass(frozen=True)
class Problem:
    truck: Truck
    packages: tuple[Package, ...]
    objective: str = "value"  # "value" | "eta"
    truck_count: int = 1

    def __post_init__(self):
        if (not isinstance(self.truck_count, int)
                or isinstance(self.truck_count, bool) or self.truck_count < 1):
            raise ValueError("truck_count must be a positive integer")
