import math
import random

import pytest

from src.algorithms.simulated_annealing import (
    SAParams,
    acceptance_probability,
    calibrate_t0,
    run_simulated_annealing,
    temperature,
)
from src.core.constraints import is_feasible
from src.core.initial import random_state
from src.core.objective import total_value
from src.core.repair import repair
from src.generate_instance import generate


def _small_problem():
    return generate("small", random.Random(5))


def test_improving_or_equal_moves_are_always_accepted():
    assert acceptance_probability(5, 10.0) == 1.0
    assert acceptance_probability(0, 0.5) == 1.0


def test_worse_moves_are_accepted_with_e_to_the_delta_over_t():
    assert acceptance_probability(-10, 10.0) == pytest.approx(math.exp(-1))
    assert acceptance_probability(-10, 1e-9) == 0.0


def test_geometric_schedule_goes_from_t0_to_t_min():
    assert temperature(0, 1000, 100.0, 0.1, "geometric") == pytest.approx(100.0)
    assert temperature(1000, 1000, 100.0, 0.1, "geometric") == pytest.approx(0.1)
    assert temperature(500, 1000, 100.0, 0.1, "geometric") == pytest.approx(math.sqrt(10))


def test_linear_schedule_goes_from_t0_to_t_min():
    assert temperature(0, 1000, 100.0, 0.1, "linear") == pytest.approx(100.0)
    assert temperature(1000, 1000, 100.0, 0.1, "linear") == pytest.approx(0.1)
    assert temperature(500, 1000, 100.0, 0.1, "linear") == pytest.approx(50.05)


def test_calibrated_t0_accepts_an_average_worse_move_with_p0():
    prob = _small_problem()
    rng = random.Random(0)
    state = repair(prob, random_state(prob, rng))
    t0, mean_worse = calibrate_t0(prob, state, p0=0.8, samples=200, rng=rng)
    assert mean_worse > 0
    assert math.exp(-mean_worse / t0) == pytest.approx(0.8)


def test_run_records_history_and_returns_feasible_best():
    prob = _small_problem()
    result = run_simulated_annealing(prob, SAParams(max_iterations=300), seed=1)
    assert result.algorithm == "sa"
    h = result.history
    assert len(h["value"]) == len(h["best"]) == 301
    assert len(h["temperature"]) == len(h["eET"]) == 300
    assert all(0.0 <= p <= 1.0 for p in h["eET"])
    assert is_feasible(prob, result.initial_state)
    assert is_feasible(prob, result.final_state)
    assert result.initial_value == h["value"][0] == total_value(prob, result.initial_state)
    assert result.final_value == max(h["value"]) == h["best"][-1]
    assert result.final_value == total_value(prob, result.final_state)
    assert all(b >= a for a, b in zip(h["best"], h["best"][1:]))


def test_run_reports_move_counts_and_stuck_frequency():
    prob = _small_problem()
    result = run_simulated_annealing(
        prob, SAParams(max_iterations=400, stuck_patience=50), seed=2)
    e = result.extra
    assert e["accepted"] + e["rejected"] == 400
    assert 0 <= e["accepted_worse"] <= e["accepted"]
    assert 0 <= e["null_moves"] <= e["accepted"]
    assert e["stuck_count"] >= 0
    assert 0 <= e["stuck_iterations"] <= 400
    assert 0 <= e["best_iteration"] <= 400
    assert e["t0"] > e["t_min"]


# Near zero temperature the search freezes in its first local optimum: few, long plateaus.
def test_stuck_is_counted_once_per_plateau():
    prob = _small_problem()
    params = SAParams(max_iterations=600, t0=1e-6, t_min=1e-7, stuck_patience=100)
    e = run_simulated_annealing(prob, params, seed=3).extra
    assert e["accepted_worse"] == 0
    assert 1 <= e["stuck_count"] <= 6
    assert e["stuck_iterations"] >= 100


def test_explicit_t0_is_used():
    result = run_simulated_annealing(_small_problem(),
                                     SAParams(max_iterations=50, t0=42.0), seed=1)
    assert result.history["temperature"][0] == pytest.approx(42.0)
    assert result.extra["t0"] == 42.0


def test_same_seed_same_result():
    prob = _small_problem()
    a = run_simulated_annealing(prob, SAParams(max_iterations=200), seed=4)
    b = run_simulated_annealing(prob, SAParams(max_iterations=200), seed=4)
    assert (a.final_state, a.history) == (b.final_state, b.history)


def test_invalid_params_are_rejected():
    with pytest.raises(ValueError):
        SAParams(max_iterations=0)
    with pytest.raises(ValueError):
        SAParams(schedule="exponential-ish")
    with pytest.raises(ValueError):
        SAParams(t0=1.0, t_min=2.0)
    with pytest.raises(ValueError):
        SAParams(p0=1.0)
    with pytest.raises(ValueError):
        SAParams(stuck_patience=0)


def test_ratio_swap_exchanges_an_inside_and_an_outside_package():
    from collections import Counter

    from src.algorithms.simulated_annealing import ratio_swap
    from tests.factories import OUT, at, pkg, problem

    prob = problem(pkg(0, value=10, weight=10), pkg(1, value=50, weight=1),
                   pkg(2, value=1, weight=10))
    state = (at(0, 0, 0), OUT, OUT)
    rng = random.Random(0)
    picked = Counter()
    for _ in range(2000):
        neighbor, move = ratio_swap(prob, state, rng)
        assert move.kind == "swap"
        assert neighbor[0] == OUT
        j = move.j if move.i == 0 else move.i
        assert neighbor[j].in_truck and (neighbor[j].x, neighbor[j].y) == (0, 0)
        picked[j] += 1
    assert picked[1] > 0.9 * 2000  # ratio 50 vs 0.1


def test_ratio_swap_falls_back_to_a_random_neighbor_when_nothing_to_exchange():
    from src.algorithms.simulated_annealing import ratio_swap
    from tests.factories import OUT, pkg, problem

    prob = problem(pkg(0), pkg(1))
    _, move = ratio_swap(prob, (OUT, OUT), random.Random(1))
    assert move.kind in ("move", "rotate", "swap")


def test_ratio_bias_is_validated_and_recorded():
    with pytest.raises(ValueError):
        SAParams(ratio_bias=1.5)
    result = run_simulated_annealing(_small_problem(),
                                     SAParams(max_iterations=200, ratio_bias=0.5), seed=1)
    assert result.params["ratio_bias"] == 0.5
    assert is_feasible(_small_problem(), result.final_state)
