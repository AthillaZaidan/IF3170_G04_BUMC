import random

from src.core.initial import random_state
from src.core.neighbors import (
    OUTSIDE,
    all_neighbors,
    candidate_positions,
    move,
    random_neighbor,
    rotate,
    sample_neighbors,
    swap,
)
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


def _prob_and_state():
    prob = problem(*(pkg(i, 2, 2, 1) for i in range(6)), truck=(6, 6, 4))
    return prob, random_state(prob, random.Random(3))


def test_random_neighbor_is_reproducible_and_changes_one_move():
    prob, state = _prob_and_state()
    a = [random_neighbor(prob, state, rng) for rng in [random.Random(1)] for _ in range(50)]
    b = [random_neighbor(prob, state, rng) for rng in [random.Random(1)] for _ in range(50)]
    assert a == b
    kinds = {m.kind for _, m in a}
    assert kinds == {"move", "rotate", "swap"}
    for neighbor, m in a:
        changed = [k for k in range(len(state)) if neighbor[k] != state[k]]
        allowed = {m.i} if m.j is None else {m.i, m.j}
        assert set(changed) <= allowed


def test_sample_neighbors_returns_k():
    prob, state = _prob_and_state()
    assert len(sample_neighbors(prob, state, 7, random.Random(0))) == 7


def test_all_neighbors_covers_every_rotation_and_differs_from_state():
    prob, state = _prob_and_state()
    neighbors = list(all_neighbors(prob, state))
    rotations = [m for _, m in neighbors if m.kind == "rotate"]
    assert len(rotations) == 3 * len(state)
    assert all(n != state for n, m in neighbors if m.kind != "rotate")
