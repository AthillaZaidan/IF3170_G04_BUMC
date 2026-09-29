from src.core.neighbors import OUTSIDE, candidate_positions, move, rotate, swap
from src.core.state import Placement
from tests.factories import OUT, at, pkg, problem


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


def test_candidate_positions_are_extreme_points_that_fit():
    prob = problem(pkg(1, 2, 3, 1), pkg(2, 2, 2, 2), truck=(5, 5, 5))
    state = (at(0, 0, 0), OUT)
    # Origin plus P1's right (2,0,0), front (0,3,0) and top (0,0,1) points.
    assert candidate_positions(prob, state, 1) == [(0, 0, 0), (0, 0, 1), (0, 3, 0), (2, 0, 0)]


def test_candidate_positions_drop_points_that_break_walls():
    prob = problem(pkg(1, 2, 3, 1), pkg(2, 2, 3, 2), truck=(4, 5, 5))
    state = (at(0, 0, 0), OUT)
    # (0,3,0) would need y up to 6 > 5: dropped.
    assert candidate_positions(prob, state, 1) == [(0, 0, 0), (0, 0, 1), (2, 0, 0)]
