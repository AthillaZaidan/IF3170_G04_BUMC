from src.core.geometry import (
    box,
    footprint_overlaps,
    overlaps,
    rests_on,
    within_walls,
)
from src.core.models import Package, Truck
from src.core.state import Placement

PKG = Package("P1", 3, 4, 1, value=60, weight=5, fragile=False, eta=3)


def test_box_uses_corner_and_oriented_extents():
    assert box(PKG, Placement(True, 1, 2, 0)) == (1, 2, 0, 4, 6, 1)
    assert box(PKG, Placement(True, 1, 2, 0, ("y", "x", "z"))) == (1, 2, 0, 5, 5, 1)


def test_touching_faces_do_not_overlap():
    a = (0, 0, 0, 2, 2, 2)
    assert not overlaps(a, (2, 0, 0, 4, 2, 2))
    assert not overlaps(a, (0, 0, 2, 2, 2, 4))
    assert overlaps(a, (1, 1, 1, 3, 3, 3))


def test_footprint_ignores_height():
    assert footprint_overlaps((0, 0, 0, 2, 2, 1), (1, 1, 5, 3, 3, 6))
    assert not footprint_overlaps((0, 0, 0, 2, 2, 1), (2, 0, 5, 4, 2, 6))


def test_rests_on_needs_contact_and_shared_area():
    lower = (0, 0, 0, 2, 2, 2)
    assert rests_on(lower, (1, 1, 2, 5, 5, 3))  # one corner is enough
    assert not rests_on(lower, (2, 2, 2, 5, 5, 3))  # only an edge touches
    assert not rests_on(lower, (0, 0, 3, 2, 2, 4))  # gap in between


def test_within_walls():
    truck = Truck(10, 20, 5, 500)
    assert within_walls((0, 0, 0, 10, 20, 5), truck)
    assert not within_walls((8, 0, 0, 11, 4, 1), truck)
    assert not within_walls((-1, 0, 0, 2, 4, 1), truck)
