"""Genetic Algorithm, following the lecture slides (Beyond Classical Search,
pp. 41-50).

- Individual: a State; gene i is the placement of package i
  ("one character ~ one variable").
- Fitness: total value inside the truck. Every individual is repaired, so it
  is feasible and its fitness is >= 0.
- Selection: roulette wheel, probability proportional to fitness.
- Crossover: one random cut point, two parents give two children.
- Mutation: one random neighbor move (same moves as HC/SA).
- Elitism (not in the slides): the best `elitism` individuals are copied into
  the next generation, so the best fitness never goes down.
"""
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def roulette_select(population: Sequence[T], fitness: Sequence[float],
                    rng: random.Random) -> T:
    """Fitness-proportionate selection; uniform if every fitness is 0."""
    if sum(fitness) <= 0:
        return rng.choice(population)
    return rng.choices(population, weights=fitness, k=1)[0]


def single_point_crossover(p1: tuple, p2: tuple,
                           rng: random.Random) -> tuple[tuple, tuple]:
    """Cut both parents at the same random point in [1, n-1] and swap tails."""
    n = len(p1)
    if n < 2:
        return p1, p2
    cut = rng.randint(1, n - 1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]
