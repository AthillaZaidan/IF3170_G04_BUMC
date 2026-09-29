"""Package orientation, written like the spec: which truck axis each side lies on.

An orientation is a tuple (axis of w, axis of l, axis of h). ("x", "y", "z") is
{w:x, l:y, h:z}. A box has six orientations, one per permutation of the axes.
"""
from itertools import permutations

from src.core.models import Package

AXES = ("x", "y", "z")

Orientation = tuple[str, str, str]

DEFAULT_ORIENTATION: Orientation = ("x", "y", "z")
ALL_ORIENTATIONS: tuple[Orientation, ...] = tuple(permutations(AXES))


def rotate(orientation: Orientation, axis: str) -> Orientation:
    """Rotate 90 degrees about `axis`: the other two axes trade places.

    Direction does not matter for an axis-aligned box, so 90 and 270 degrees
    give the same orientation (see the course QnA).
    """
    a, b = (other for other in AXES if other != axis)
    swapped = {a: b, b: a}
    return tuple(swapped.get(side_axis, side_axis) for side_axis in orientation)


def extents(package: Package, orientation: Orientation) -> tuple[int, int, int]:
    """Size of the package along x, y and z in this orientation."""
    size = dict(zip(orientation, (package.w, package.l, package.h)))
    return size["x"], size["y"], size["z"]
