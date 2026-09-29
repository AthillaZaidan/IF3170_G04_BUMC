from dataclasses import FrozenInstanceError

import pytest

from src.core.state import Placement, take_out


def test_placement_is_immutable_with_default_orientation():
    p = Placement(in_truck=True, x=1, y=2, z=0)
    assert p.orientation == ("x", "y", "z")
    with pytest.raises(FrozenInstanceError):
        p.x = 5


def test_take_out_keeps_orientation_and_resets_position():
    p = Placement(True, 1, 2, 3, ("z", "x", "y"))
    out = take_out(p)
    assert out == Placement(False, 0, 0, 0, ("z", "x", "y"))
    assert p.in_truck  # original untouched
