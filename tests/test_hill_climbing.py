import pytest

from src.algorithms.hill_climbing import HCParams, run_hill_climbing
from src.core.constraints import is_feasible
from src.core.objective import total_value
from src.generate_instance import generate


def _small_problem():
    import random

    return generate("small", random.Random(7))


@pytest.mark.parametrize("variant", ["steepest", "stochastic", "sideways"])
def test_hill_climbing_variants_return_feasible_solution(variant):
    prob = _small_problem()
    params = HCParams(
        variant=variant,
        max_iterations=5,
        neighbor_samples=12,
        max_sideways=2,
    )
    result = run_hill_climbing(prob, params, seed=1)
    assert result.algorithm == f"hc-{variant}"
    assert is_feasible(prob, result.initial_state)
    assert is_feasible(prob, result.final_state)
    assert result.final_value == total_value(prob, result.final_state)
    assert result.initial_value == result.history["value"][0]
    assert result.final_value == result.history["value"][-1]
    assert len(result.history["value"]) == result.extra["iterations"] + 1
    assert all(a <= b for a, b in zip(result.history["best"], result.history["best"][1:]))


def test_random_restart_records_restart_metadata_and_best_solution():
    prob = _small_problem()
    params = HCParams(variant="random-restart", max_iterations=3, max_restarts=2)
    result = run_hill_climbing(prob, params, seed=3)
    assert result.algorithm == "hc-random-restart"
    assert is_feasible(prob, result.final_state)
    assert result.final_value == max(result.history["best"])
    assert result.extra["restarts"] == 2
    assert len(result.extra["iterations_per_restart"]) == 3
    assert len(result.extra["restart_offsets"]) == 3
    assert 0 <= result.extra["best_restart"] <= 2


def test_same_seed_same_hill_climbing_result():
    prob = _small_problem()
    params = HCParams(variant="stochastic", max_iterations=4, neighbor_samples=10)
    a = run_hill_climbing(prob, params, seed=4)
    b = run_hill_climbing(prob, params, seed=4)
    assert (a.final_state, a.history) == (b.final_state, b.history)


def test_invalid_hill_climbing_params_are_rejected():
    with pytest.raises(ValueError):
        HCParams(variant="unknown")
    with pytest.raises(ValueError):
        HCParams(max_iterations=-1)
    with pytest.raises(ValueError):
        HCParams(neighbor_samples=0)
    with pytest.raises(ValueError):
        HCParams(max_sideways=-1)
    with pytest.raises(ValueError):
        HCParams(max_restarts=-1)
