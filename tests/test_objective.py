from src.core.objective import default_penalty, evaluate, total_value
from tests.factories import OUT, at, pkg, problem


def test_feasible_state_scores_its_value():
    prob = problem(pkg(1, value=30), pkg(2, value=50))
    ev = evaluate(prob, (at(0, 0, 0), OUT))
    assert (ev.value, ev.score, ev.feasible) == (30, 30, True)


def test_violations_are_penalized():
    prob = problem(pkg(1, 2, 2, 2, value=30), pkg(2, 2, 2, 2, value=50))
    ev = evaluate(prob, (at(0, 0, 0), at(1, 0, 0)))  # overlap
    assert ev.value == 80
    assert ev.score == 80 - default_penalty(prob)
    assert not ev.feasible


def test_default_penalty_beats_any_single_package():
    prob = problem(pkg(1, value=30), pkg(2, value=50))
    assert default_penalty(prob) == 51


def test_capacity_penalty_scales_with_excess_weight():
    prob = problem(pkg(1, value=10, weight=8), pkg(2, value=10, weight=8), capacity=10)
    ev = evaluate(prob, (at(0, 0, 0), at(1, 0, 0)), penalty=2)
    assert ev.score == 20 - 2 * 6


def test_total_value_only_counts_inside():
    prob = problem(pkg(1, value=30), pkg(2, value=50))
    assert total_value(prob, (OUT, at(0, 0, 0))) == 50
