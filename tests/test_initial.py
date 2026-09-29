import random

from src.core.geometry import box, within_walls
from src.core.initial import random_state
from tests.factories import pkg, problem


def test_random_state_is_reproducible_and_in_bounds():
    prob = problem(*(pkg(i, 3, 2, 1) for i in range(30)), truck=(6, 6, 6))
    a = random_state(prob, random.Random(7))
    b = random_state(prob, random.Random(7))
    assert a == b
    assert len(a) == 30
    assert any(p.in_truck for p in a) and any(not p.in_truck for p in a)
    for package, p in zip(prob.packages, a):
        if p.in_truck:
            assert within_walls(box(package, p), prob.truck)


def test_package_bigger_than_truck_starts_outside():
    prob = problem(pkg(1, 20, 20, 20), truck=(5, 5, 5))
    for seed in range(20):
        assert not random_state(prob, random.Random(seed), p_in=1.0)[0].in_truck
