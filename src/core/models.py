"""Static problem data: packages and the truck container.

These never change during a search. Where a package sits and how it is turned
is part of the search state (see state.py), not of the package itself.
"""
from dataclasses import dataclass

from src.core.validation import require_integer, require_number


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

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("package id must be a nonempty string")
        for name in ("w", "l", "h", "weight"):
            require_integer(getattr(self, name), f"package {self.id}: {name}", 1)
        require_number(self.value, f"package {self.id}: value")
        require_number(self.eta, f"package {self.id}: eta")
        if self.eta == 0:
            raise ValueError(f"package {self.id}: eta must be > 0")
        if type(self.fragile) is not bool:
            raise ValueError(f"package {self.id}: fragile must be a boolean")


@dataclass(frozen=True)
class Truck:
    w: int
    l: int
    h: int
    max_capacity: int

    def __post_init__(self):
        for name in ("w", "l", "h", "max_capacity"):
            require_integer(getattr(self, name), f"truck: {name}", 1)


@dataclass(frozen=True)
class Problem:
    truck: Truck
    packages: tuple[Package, ...]
    objective: str = "value"  # "value" | "eta"
    truck_count: int = 1

    def __post_init__(self):
        require_integer(self.truck_count, "truck_count", 1)
        if self.objective not in ("value", "eta"):
            raise ValueError("objective must be value or eta")
        ids = [p.id for p in self.packages]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate package id")
