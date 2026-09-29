import random

import pytest

from src.core.constraints import is_feasible
from src.core.initial import random_state
from src.core.repair import repair
from tests.factories import OUT, at, pkg, problem


def test_out_of_bounds_package_is_taken_out():
    prob = problem(pkg(1, 3, 1, 1), truck=(4, 4, 4))
    assert not repair(prob, (at(2, 0, 0),))[0].in_truck


def test_overweight_removes_lowest_value_per_weight_first():
    prob = problem(
        pkg(1, value=10, weight=5),  # ratio 2
        pkg(2, value=10, weight=10),  # ratio 1 -> removed
        pkg(3, value=40, weight=5),  # ratio 8
        capacity=12,
    )
    fixed = repair(prob, (at(0, 0, 0), at(1, 0, 0), at(2, 0, 0)))
    assert [p.in_truck for p in fixed] == [True, False, True]


def test_floating_package_falls_onto_highest_support():
    prob = problem(pkg(1, 2, 2, 2), pkg(2, 2, 2, 1))
    fixed = repair(prob, (at(0, 0, 0), at(1, 1, 4)))
    assert fixed[1] == at(1, 1, 2)
    assert is_feasible(prob, fixed)


def test_floating_package_with_nothing_below_falls_to_floor():
    prob = problem(pkg(1, 2, 2, 2))
    assert repair(prob, (at(3, 3, 5),))[0] == at(3, 3, 0)


def test_overlap_keeps_lower_then_more_valuable_package():
    prob = problem(pkg(1, 2, 2, 2, value=10), pkg(2, 2, 2, 2, value=90))
    fixed = repair(prob, (at(0, 0, 0), at(1, 1, 0)))
    assert [p.in_truck for p in fixed] == [False, True]


def test_package_landing_on_fragile_is_taken_out():
    prob = problem(pkg(1, 2, 2, 2, fragile=True), pkg(2, 2, 2, 2))
    fixed = repair(prob, (at(0, 0, 0), at(0, 0, 4)))
    assert fixed[0] == at(0, 0, 0)
    assert fixed[1] == OUT


def test_repair_never_mutates_its_input():
    prob = problem(pkg(1, 2, 2, 2))
    state = (at(0, 0, 5),)
    repair(prob, state)
    assert state == (at(0, 0, 5),)


@pytest.mark.parametrize("seed", range(200))
def test_repair_always_yields_feasible_and_is_idempotent(seed):
    rng = random.Random(seed)
    packages = [
        pkg(i, rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4),
            value=rng.randint(1, 50), weight=rng.randint(1, 10),
            fragile=rng.random() < 0.3)
        for i in range(rng.randint(1, 25))
    ]
    prob = problem(*packages, truck=(6, 7, 5), capacity=rng.randint(10, 80))
    fixed = repair(prob, random_state(prob, rng, p_in=0.8))
    assert is_feasible(prob, fixed)
    assert repair(prob, fixed) == fixed
