from src.core.bounds import value_upper_bound
from tests.factories import pkg, problem


def test_bound_is_total_value_when_everything_fits():
    prob = problem(pkg(0, value=5), pkg(1, value=7), truck=(4, 4, 4))
    assert value_upper_bound(prob) == 12


def test_bound_respects_weight_capacity():
    prob = problem(pkg(0, value=10, weight=6), pkg(1, value=9, weight=5),
                   pkg(2, value=8, weight=5), capacity=10)
    assert value_upper_bound(prob) == 17


def test_bound_respects_truck_volume():
    # truck volume 8; two 2x2x2 boxes cannot both be inside
    prob = problem(pkg(0, 2, 2, 2, value=3), pkg(1, 2, 2, 2, value=4), truck=(2, 2, 2))
    assert value_upper_bound(prob) == 4


def test_package_that_fits_in_no_orientation_is_ignored():
    prob = problem(pkg(0, 5, 1, 1, value=100), pkg(1, value=1), truck=(3, 3, 3))
    assert value_upper_bound(prob) == 1


def test_package_that_fits_only_when_rotated_counts():
    prob = problem(pkg(0, 1, 1, 5, value=100), truck=(5, 1, 1))
    assert value_upper_bound(prob) == 100
