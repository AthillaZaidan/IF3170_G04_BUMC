from src.core.models import Package
from src.core.orientation import (
    ALL_ORIENTATIONS,
    AXES,
    DEFAULT_ORIENTATION,
    extents,
    rotate,
)

PKG = Package("P1", 3, 4, 1, value=60, weight=5, fragile=False, eta=3)


def test_there_are_exactly_six_orientations():
    assert len(ALL_ORIENTATIONS) == 6
    assert len(set(ALL_ORIENTATIONS)) == 6
    assert DEFAULT_ORIENTATION in ALL_ORIENTATIONS


def test_rotation_about_z_matches_spec_example():
    # spec v1.4: {w:x, l:y, h:z} -> {w:y, l:x, h:z} after 90 degrees about z
    assert rotate(("x", "y", "z"), "z") == ("y", "x", "z")


def test_rotating_twice_about_one_axis_restores_orientation():
    for orientation in ALL_ORIENTATIONS:
        for axis in AXES:
            assert rotate(rotate(orientation, axis), axis) == orientation


def test_rotations_reach_every_orientation():
    seen = {DEFAULT_ORIENTATION}
    frontier = [DEFAULT_ORIENTATION]
    while frontier:
        current = frontier.pop()
        for axis in AXES:
            nxt = rotate(current, axis)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(ALL_ORIENTATIONS)


def test_extents_follow_orientation():
    assert extents(PKG, ("x", "y", "z")) == (3, 4, 1)
    assert extents(PKG, ("y", "x", "z")) == (4, 3, 1)
    assert extents(PKG, ("z", "x", "y")) == (4, 1, 3)
