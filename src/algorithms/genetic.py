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
import time
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Sequence, TypeVar

from src.core.initial import random_state
from src.core.models import Problem
from src.core.neighbors import random_neighbor
from src.core.objective import total_value
from src.core.repair import repair
from src.core.result import RunResult

T = TypeVar("T")


@dataclass(frozen=True)
class GAParams:
    population_size: int
    generations: int
    mutation_rate: float = 0.2
    elitism: int = 1

    def __post_init__(self):
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 0:
            raise ValueError("generations must be >= 0")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism must be in [0, population_size)")


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


def run_genetic(problem: Problem, params: GAParams, seed: int) -> RunResult:
    rng = random.Random(seed)
    start = time.perf_counter()

    def fitness_of(population):
        return [total_value(problem, individual) for individual in population]

    population = [repair(problem, random_state(problem, rng))
                  for _ in range(params.population_size)]
    fitness = fitness_of(population)

    # "Initial state" for the report: the best individual of generation 0.
    best_index = max(range(len(population)), key=fitness.__getitem__)
    initial_state, initial_value = population[best_index], fitness[best_index]
    best_state, best_value, best_generation = initial_state, initial_value, 0
    history = {"max": [max(fitness)], "mean": [fmean(fitness)]}

    for generation in range(1, params.generations + 1):
        ranked = sorted(range(len(population)), key=fitness.__getitem__, reverse=True)
        next_population = [population[i] for i in ranked[:params.elitism]]
        while len(next_population) < params.population_size:
            p1 = roulette_select(population, fitness, rng)
            p2 = roulette_select(population, fitness, rng)
            for child in single_point_crossover(p1, p2, rng):
                if rng.random() < params.mutation_rate:
                    child, _ = random_neighbor(problem, child, rng)
                next_population.append(repair(problem, child))
        population = next_population[:params.population_size]
        fitness = fitness_of(population)

        history["max"].append(max(fitness))
        history["mean"].append(fmean(fitness))
        generation_best = max(range(len(population)), key=fitness.__getitem__)
        if fitness[generation_best] > best_value:
            best_state = population[generation_best]
            best_value = fitness[generation_best]
            best_generation = generation

    return RunResult(
        algorithm="ga",
        params=asdict(params),
        seed=seed,
        duration_s=time.perf_counter() - start,
        initial_state=initial_state,
        initial_value=initial_value,
        final_state=best_state,
        final_value=best_value,
        history=history,
        extra={"best_generation": best_generation},
    )
