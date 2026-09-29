import random
from collections import Counter

from src.algorithms.genetic import roulette_select, single_point_crossover


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
