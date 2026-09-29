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
    max_capacity: int  # in weight units (spec v1.3)


@dataclass(frozen=True)
class Problem:
    truck: Truck
    packages: tuple[Package, ...]
