import random
from collections import Counter

import pytest

from src.algorithms.genetic import (
    GAParams,
    roulette_select,
    run_genetic,
    single_point_crossover,
)
from src.core.constraints import is_feasible
from src.core.objective import total_value
from src.generate_instance import generate


def test_roulette_picks_proportionally_to_fitness():
    rng = random.Random(0)
    counts = Counter(roulette_select(["a", "b", "c"], [1, 3, 0], rng) for _ in range(4000))
    assert counts["c"] == 0
    assert 0.2 < counts["a"] / 4000 < 0.3  # expected 0.25


def test_roulette_is_uniform_when_all_fitness_is_zero():
    rng = random.Random(0)
    counts = Counter(roulette_select(["a", "b"], [0, 0], rng) for _ in range(2000))
    assert counts["a"] > 800 and counts["b"] > 800


def test_single_point_crossover_swaps_tails():
    p1, p2 = tuple("AAAAA"), tuple("BBBBB")
    c1, c2 = single_point_crossover(p1, p2, random.Random(3))
    cut = c1.index("B")
    assert 1 <= cut <= 4
    assert c1 == p1[:cut] + p2[cut:]
    assert c2 == p2[:cut] + p1[cut:]


def test_crossover_of_single_gene_returns_parents():
    assert single_point_crossover(("A",), ("B",), random.Random(0)) == (("A",), ("B",))


def _small_problem():
    return generate("small", random.Random(5))


def test_run_genetic_records_history_and_returns_feasible_best():
    prob = _small_problem()
    result = run_genetic(prob, GAParams(population_size=8, generations=6), seed=1)
    assert result.algorithm == "ga"
    assert len(result.history["max"]) == len(result.history["mean"]) == 7
    assert is_feasible(prob, result.final_state)
    assert is_feasible(prob, result.initial_state)
    assert result.final_value == total_value(prob, result.final_state)
    assert result.final_value == max(result.history["max"])
    assert result.initial_value == result.history["max"][0]
    assert result.params == {"population_size": 8, "generations": 6,
                             "mutation_rate": 0.2, "elitism": 1}
    assert 0 <= result.extra["best_generation"] <= 6


def test_elitism_keeps_best_fitness_from_dropping():
    result = run_genetic(_small_problem(), GAParams(10, 15), seed=2)
    maxima = result.history["max"]
    assert all(b >= a for a, b in zip(maxima, maxima[1:]))


def test_same_seed_same_result():
    prob = _small_problem()
    a = run_genetic(prob, GAParams(6, 4), seed=3)
    b = run_genetic(prob, GAParams(6, 4), seed=3)
    assert (a.final_state, a.history) == (b.final_state, b.history)


def test_invalid_params_are_rejected():
    with pytest.raises(ValueError):
        GAParams(population_size=1, generations=5)
    with pytest.raises(ValueError):
        GAParams(population_size=4, generations=5, elitism=4)
    with pytest.raises(ValueError):
        GAParams(population_size=4, generations=5, mutation_rate=1.5)
