"""Objective function: maximize the total value of packages inside the truck.

score = value - penalty * (weighted violation count). For a feasible state the
score equals the value, which is the number reported for every algorithm.
"""
from dataclasses import dataclass

from src.core.constraints import Violation, find_violations
from src.core.models import Problem
from src.core.state import State


@dataclass(frozen=True)
class Evaluation:
    value: float
    violations: tuple[Violation, ...]
    score: float

    @property
    def feasible(self) -> bool:
        return not self.violations


def default_penalty(problem: Problem) -> int:
    """One more than the most valuable package, so squeezing a package in
    illegally is never better than leaving it out."""
    return max((p.value for p in problem.packages), default=0) + 1


# Objective "eta" (bonus): value / ETA, so a package due sooner counts for more.
def total_value(problem: Problem, state: State) -> float:
    if len(state) != len(problem.packages):
        raise ValueError("state must contain one placement per package")
    if problem.objective == "eta":
        return sum(pkg.value / pkg.eta for pkg, p in zip(problem.packages, state) if p.in_truck)
    return sum(pkg.value for pkg, p in zip(problem.packages, state) if p.in_truck)


def evaluate(problem: Problem, state: State, penalty: float | None = None) -> Evaluation:
    lam = default_penalty(problem) if penalty is None else penalty
    value = total_value(problem, state)
    violations = tuple(find_violations(problem, state))
    score = value - lam * sum(v.amount for v in violations)
    return Evaluation(value, violations, score)
