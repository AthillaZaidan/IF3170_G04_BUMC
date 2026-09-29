from src.core.neighbors import OUTSIDE, move, rotate, swap
from src.core.state import Placement
from tests.factories import OUT, at


def test_move_into_truck_and_out():
    state = (OUT, at(1, 1, 0))
    moved = move(state, 0, (2, 3, 0))
    assert moved[0] == at(2, 3, 0)
    assert move(moved, 1, OUTSIDE)[1] == OUT
    assert state == (OUT, at(1, 1, 0))  # input untouched


def test_move_keeps_orientation():
    state = (Placement(False, orientation=("z", "y", "x")),)
    assert move(state, 0, (0, 0, 0))[0].orientation == ("z", "y", "x")


def test_rotate_changes_orientation_and_keeps_corner():
    state = (at(1, 2, 0),)
    rotated = rotate(state, 0, "z")[0]
    assert rotated == at(1, 2, 0, ("y", "x", "z"))


def test_swap_exchanges_positions_not_orientations():
    a = at(0, 0, 0, ("y", "x", "z"))
    b = Placement(False, orientation=("x", "z", "y"))
    swapped = swap((a, b), 0, 1)
    assert swapped[0] == Placement(False, 0, 0, 0, ("y", "x", "z"))
    assert swapped[1] == at(0, 0, 0, ("x", "z", "y"))
    assert swap(swapped, 0, 1) == (a, Placement(False, 0, 0, 0, ("x", "z", "y")))
