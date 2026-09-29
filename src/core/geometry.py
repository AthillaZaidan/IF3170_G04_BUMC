"""Axis-aligned box helpers.

A Box is (x0, y0, z0, x1, y1, z1) with half-open intervals [x0, x1), so two boxes
that only touch faces do not overlap.
"""
from src.core.models import Package, Truck
from src.core.orientation import extents
from src.core.state import Placement

Box = tuple[int, int, int, int, int, int]


def box(package: Package, placement: Placement) -> Box:
    dx, dy, dz = extents(package, placement.orientation)
    x, y, z = placement.x, placement.y, placement.z
    return (x, y, z, x + dx, y + dy, z + dz)


def footprint_overlaps(a: Box, b: Box) -> bool:
    """True when the x-y projections share area > 0."""
    return a[0] < b[3] and b[0] < a[3] and a[1] < b[4] and b[1] < a[4]


def overlaps(a: Box, b: Box) -> bool:
    return footprint_overlaps(a, b) and a[2] < b[5] and b[2] < a[5]


def rests_on(lower: Box, upper: Box) -> bool:
    """True when `upper` sits directly on top of `lower` over some area.

    This is the spec's support definition: at least one base point of `upper`
    lies exactly on the top face of `lower`.
    """
    return lower[5] == upper[2] and footprint_overlaps(lower, upper)


def within_walls(b: Box, truck: Truck) -> bool:
    return (
        b[0] >= 0 and b[1] >= 0 and b[2] >= 0
        and b[3] <= truck.w and b[4] <= truck.l and b[5] <= truck.h
    )
