import math
import random
import time
from dataclasses import asdict, dataclass

from src.core.initial import random_state
from src.core.models import Problem
from src.core.neighbors import Move, random_neighbor, swap
from src.core.objective import total_value
from src.core.repair import repair
from src.core.result import RunResult
from src.core.state import State

SCHEDULES = ("geometric", "linear")


@dataclass(frozen=True)
class SAParams:
    max_iterations: int = 20000
    t0: float | None = None  # None: calibrate from p0
    t_min: float = 0.05
    schedule: str = "geometric"
    p0: float = 0.8
    calibration_samples: int = 100
    stuck_patience: int = 500
    ratio_bias: float = 0.0

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if self.t_min <= 0:
            raise ValueError("t_min must be > 0")
        if self.t0 is not None and self.t0 <= self.t_min:
            raise ValueError("t0 must be greater than t_min")
        if not 0 < self.p0 < 1:
            raise ValueError("p0 must be in (0, 1)")
        if self.calibration_samples < 1:
            raise ValueError("calibration_samples must be >= 1")
        if self.stuck_patience < 1:
            raise ValueError("stuck_patience must be >= 1")
        if not 0 <= self.ratio_bias <= 1:
            raise ValueError("ratio_bias must be in [0, 1]")


# e^(dE/T) for a worse move, 1 for an equal or better one.
def acceptance_probability(delta: float, t: float) -> float:
    if delta >= 0:
        return 1.0
    return math.exp(delta / t)


# Temperature at iteration k of n: t0 at k = 0 and t_min at k = n.
def temperature(k: int, n: int, t0: float, t_min: float, schedule: str) -> float:
    frac = k / n
    if schedule == "geometric":
        return t0 * (t_min / t0) ** frac
    return t0 + (t_min - t0) * frac


# 4 significant digits keep result.json small and are enough for the plots.
def _rounded(x: float) -> float:
    return float(f"{x:.4g}")


# Swap a low value/weight package inside with a high one outside (random move if a side is empty).
def ratio_swap(problem: Problem, state: State, rng: random.Random) -> tuple[State, Move]:
    inside = [i for i, p in enumerate(state) if p.in_truck]
    outside = [j for j, p in enumerate(state) if not p.in_truck]
    if not inside or not outside:
        return random_neighbor(problem, state, rng)
    pkgs = problem.packages
    i = rng.choices(inside, weights=[pkgs[i].weight / max(pkgs[i].value, 1) for i in inside])[0]
    j = rng.choices(outside, weights=[max(pkgs[j].value, 1) / pkgs[j].weight for j in outside])[0]
    return swap(state, i, j), Move("swap", i, j=j)


# One repaired successor and its value; a ratio swap with probability ratio_bias.
def _neighbor(problem: Problem, state: State, rng: random.Random,
              ratio_bias: float = 0.0) -> tuple[State, int]:
    if ratio_bias > 0 and rng.random() < ratio_bias:
        candidate, _ = ratio_swap(problem, state, rng)
    else:
        candidate, _ = random_neighbor(problem, state, rng)
    candidate = repair(problem, candidate)
    return candidate, total_value(problem, candidate)


# t0 such that an average worse move is accepted with probability p0 (Ben-Ameur 2004).
def calibrate_t0(problem: Problem, state: State, p0: float, samples: int,
                 rng: random.Random) -> tuple[float, float]:
    value = total_value(problem, state)
    worse = []
    for _ in range(samples):
        _, candidate_value = _neighbor(problem, state, rng)
        if candidate_value < value:
            worse.append(value - candidate_value)
    if not worse:
        return 1.0, 0.0
    mean_worse = sum(worse) / len(worse)
    return -mean_worse / math.log(p0), mean_worse


# Returns the best state seen, not the last one.
def run_simulated_annealing(problem: Problem, params: SAParams, seed: int) -> RunResult:
    rng = random.Random(seed)
    start = time.perf_counter()

    current = repair(problem, random_state(problem, rng))
    current_value = total_value(problem, current)
    initial_state, initial_value = current, current_value

    if params.t0 is None:
        t0, mean_worse = calibrate_t0(problem, current, params.p0,
                                      params.calibration_samples, rng)
    else:
        t0, mean_worse = params.t0, None
    t_min = min(params.t_min, t0 / 2)
    n = params.max_iterations

    best_state, best_value, best_iteration = current, current_value, 0
    history = {"value": [current_value], "best": [current_value],
               "temperature": [], "eET": []}
    accepted = accepted_worse = null_moves = 0
    plateau = stuck_count = stuck_iterations = 0

    for k in range(n):
        t = temperature(k, n, t0, t_min, params.schedule)
        candidate, candidate_value = _neighbor(problem, current, rng, params.ratio_bias)
        delta = candidate_value - current_value
        # Metropolis rule: accept dE >= 0, otherwise with probability e^(dE/T).
        p = acceptance_probability(delta, t)
        moved_value = False
        if delta >= 0 or rng.random() < p:
            accepted += 1
            accepted_worse += delta < 0
            null_moves += candidate == current
            moved_value = delta != 0
            current, current_value = candidate, candidate_value

        # Stuck: value unchanged for stuck_patience iterations, counted once per plateau.
        if moved_value:
            plateau = 0
        else:
            plateau += 1
            if plateau == params.stuck_patience:
                stuck_count += 1
                stuck_iterations += plateau
            elif plateau > params.stuck_patience:
                stuck_iterations += 1

        if current_value > best_value:
            best_state, best_value, best_iteration = current, current_value, k + 1
        history["value"].append(current_value)
        history["best"].append(best_value)
        history["temperature"].append(_rounded(t))
        history["eET"].append(_rounded(p))

    return RunResult(
        algorithm="sa",
        params=asdict(params),
        seed=seed,
        duration_s=time.perf_counter() - start,
        initial_state=initial_state,
        initial_value=initial_value,
        final_state=best_state,
        final_value=best_value,
        history=history,
        extra={
            "t0": t0, "t_min": t_min, "mean_worse_delta": mean_worse,
            "alpha": (t_min / t0) ** (1 / n) if params.schedule == "geometric" else None,
            "accepted": accepted, "rejected": n - accepted,
            "accepted_worse": accepted_worse, "null_moves": null_moves,
            "stuck_count": stuck_count, "stuck_iterations": stuck_iterations,
            "best_iteration": best_iteration, "final_current_value": current_value,
        },
    )
