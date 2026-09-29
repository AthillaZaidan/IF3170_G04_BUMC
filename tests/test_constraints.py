from src.core.constraints import find_violations, is_feasible
from tests.factories import OUT, at, pkg, problem


def kinds(prob, state):
    return sorted(v.kind for v in find_violations(prob, state))


def test_floor_placement_is_feasible():
    prob = problem(pkg(1, 2, 2, 2), pkg(2, 2, 2, 2))
    assert is_feasible(prob, (at(0, 0, 0), at(2, 0, 0)))


def test_outside_packages_are_ignored():
    prob = problem(pkg(1, 2, 2, 2), pkg(2, 50, 50, 50))
    assert is_feasible(prob, (at(0, 0, 0), OUT))


def test_wall_penetration():
    prob = problem(pkg(1, 3, 1, 1), truck=(4, 4, 4))
    assert kinds(prob, (at(2, 0, 0),)) == ["wall"]


def test_overlap():
    prob = problem(pkg(1, 2, 2, 2), pkg(2, 2, 2, 2))
    assert kinds(prob, (at(0, 0, 0), at(1, 1, 0))) == ["overlap"]


def test_floating():
    prob = problem(pkg(1, 2, 2, 2))
    assert kinds(prob, (at(0, 0, 3),)) == ["floating"]


def test_partial_support_is_not_floating():
    # QnA: one supported part of the base is enough.
    prob = problem(pkg(1, 2, 2, 2), pkg(2, 4, 4, 1))
    assert is_feasible(prob, (at(0, 0, 0), at(1, 1, 2)))


def test_edge_contact_is_not_support():
    prob = problem(pkg(1, 2, 2, 2), pkg(2, 2, 2, 1))
    assert kinds(prob, (at(0, 0, 0), at(2, 2, 2))) == ["floating"]


def test_fragile_cannot_support_even_when_other_support_exists():
    prob = problem(pkg(1, 2, 2, 2, fragile=True), pkg(2, 2, 2, 2), pkg(3, 4, 2, 1))
    state = (at(0, 0, 0), at(2, 0, 0), at(0, 0, 2))  # P3 rests on P1 and P2
    violations = find_violations(prob, state)
    assert [v.kind for v in violations] == ["fragile"]
    assert violations[0].packages == (0, 2)


def test_fragile_on_top_is_fine():
    prob = problem(pkg(1, 2, 2, 2), pkg(2, 2, 2, 2, fragile=True))
    assert is_feasible(prob, (at(0, 0, 0), at(0, 0, 2)))


def test_capacity_counts_excess_weight_units():
    prob = problem(pkg(1, weight=6), pkg(2, weight=7), pkg(3, weight=100), capacity=10)
    violations = find_violations(prob, (at(0, 0, 0), at(1, 0, 0), OUT))
    assert [(v.kind, v.amount) for v in violations] == [("capacity", 3)]
