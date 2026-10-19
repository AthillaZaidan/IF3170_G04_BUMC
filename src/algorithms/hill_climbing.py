import random
import time
from dataclasses import asdict, dataclass

from src.core.initial import random_state
from src.core.models import Problem
from src.core.neighbors import Move, all_neighbors, sample_neighbors
from src.core.objective import total_value
from src.core.repair import repair
from src.core.result import RunResult
from src.core.state import State
from src.core.validation import require_integer

HC_VARIANTS = {"steepest", "stochastic", "sideways", "random-restart"}


@dataclass(frozen=True)
class HCParams:
    variant: str = "steepest"
    max_iterations: int = 100
    neighbor_samples: int = 50
    max_sideways: int = 20
    max_restarts: int = 5

    def __post_init__(self):
        for name in ("max_iterations", "max_sideways", "max_restarts"):
            require_integer(getattr(self, name), name)
        require_integer(self.neighbor_samples, "neighbor_samples", 1)
        if self.variant not in HC_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(HC_VARIANTS)}")


@dataclass(frozen=True)
class _Candidate:
    state: State
    value: int
    move: Move


@dataclass(frozen=True)
class _ClimbOutcome:
    initial_state: State
    initial_value: int
    final_state: State
    final_value: int
    values: list[int]
    best_values: list[int]
    iterations: int
    stop_reason: str
    sideways_moves: int = 0


def _repair_value(problem: Problem, state: State) -> tuple[State, int]:
    feasible = repair(problem, state)
    return feasible, total_value(problem, feasible)


def _best_neighbor(
    problem: Problem,
    state: State,
    current_value: int | None = None,
    seen_equal: set[State] | None = None,
) -> _Candidate | None:
    best: _Candidate | None = None
    for neighbor, move in all_neighbors(problem, state):
        candidate_state, candidate_value = _repair_value(problem, neighbor)
        if candidate_state == state:
            continue
        if (
            current_value is not None
            and seen_equal is not None
            and candidate_value == current_value
            and candidate_state in seen_equal
        ):
            continue
        if best is None or candidate_value > best.value:
            best = _Candidate(candidate_state, candidate_value, move)
    return best


def _random_improving_neighbor(
    problem: Problem,
    state: State,
    current_value: int,
    sample_size: int,
    rng: random.Random,
) -> _Candidate | None:
    improving: list[_Candidate] = []
    for neighbor, move in sample_neighbors(problem, state, sample_size, rng):
        candidate_state, candidate_value = _repair_value(problem, neighbor)
        if candidate_state != state and candidate_value > current_value:
            improving.append(_Candidate(candidate_state, candidate_value, move))
    if improving:
        return rng.choice(improving)
    chosen = None
    count = 0
    seen = set()
    for neighbor, move in all_neighbors(problem, state):
        candidate_state, candidate_value = _repair_value(problem, neighbor)
        if candidate_state in seen or candidate_value <= current_value:
            continue
        seen.add(candidate_state)
        count += 1
        if rng.randrange(count) == 0:
            chosen = _Candidate(candidate_state, candidate_value, move)
    return chosen


def _run_single_climb(
    problem: Problem,
    params: HCParams,
    rng: random.Random,
    start_state: State,
    variant: str,
) -> _ClimbOutcome:
    current, current_value = _repair_value(problem, start_state)
    initial_state = current
    best_value = current_value
    values = [current_value]
    best_values = [best_value]
    seen = {current}
    sideways_moves = 0
    stop_reason = "max_iterations"

    for _ in range(1, params.max_iterations + 1):
        if variant == "stochastic":
            candidate = _random_improving_neighbor(
                problem, current, current_value, params.neighbor_samples, rng
            )
            if candidate is None:
                stop_reason = "local_optimum"
                break
        else:
            if variant == "sideways":
                candidate = _best_neighbor(problem, current, current_value, seen)
            else:
                candidate = _best_neighbor(problem, current)
            if candidate is None:
                stop_reason = "local_optimum"
                break
            if candidate.value < current_value:
                stop_reason = "local_optimum"
                break
            if candidate.value == current_value:
                if variant != "sideways":
                    stop_reason = "local_optimum"
                    break
                if sideways_moves >= params.max_sideways:
                    stop_reason = "max_sideways"
                    break
                sideways_moves += 1

        current, current_value = candidate.state, candidate.value
        seen.add(current)
        best_value = max(best_value, current_value)
        values.append(current_value)
        best_values.append(best_value)
        if variant == "sideways" and sideways_moves > 0 and sideways_moves >= params.max_sideways:
            stop_reason = "max_sideways"
            break
    return _ClimbOutcome(
        initial_state=initial_state,
        initial_value=values[0],
        final_state=current,
        final_value=current_value,
        values=values,
        best_values=best_values,
        iterations=len(values) - 1,
        stop_reason=stop_reason,
        sideways_moves=sideways_moves,
    )


def _run_random_restart(problem: Problem, params: HCParams, seed: int) -> RunResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    restart_offsets: list[int] = []
    iterations_per_restart: list[int] = []
    initial_values: list[int] = []
    stop_reasons: list[str] = []
    values: list[int] = []
    best_history: list[int] = []
    iteration_history: list[int] = []
    iterations = 0
    best_outcome: _ClimbOutcome | None = None
    best_restart = 0
    first_outcome: _ClimbOutcome | None = None
    global_best: int | None = None

    for restart in range(params.max_restarts + 1):
        restart_offsets.append(len(values))
        outcome = _run_single_climb(
            problem, params, rng, random_state(problem, rng), "steepest"
        )
        if first_outcome is None:
            first_outcome = outcome
        if best_outcome is None or outcome.final_value > best_outcome.final_value:
            best_outcome = outcome
            best_restart = restart

        iterations_per_restart.append(outcome.iterations)
        initial_values.append(outcome.initial_value)
        stop_reasons.append(outcome.stop_reason)
        for value in outcome.values:
            global_best = value if global_best is None else max(global_best, value)
            values.append(value)
            best_history.append(global_best)
        iteration_history.extend(range(iterations, iterations + len(outcome.values)))
        iterations += outcome.iterations

    assert first_outcome is not None and best_outcome is not None
    return RunResult(
        algorithm="hc-random-restart",
        params=asdict(params),
        seed=seed,
        duration_s=time.perf_counter() - start,
        initial_state=first_outcome.initial_state,
        initial_value=first_outcome.initial_value,
        final_state=best_outcome.final_state,
        final_value=best_outcome.final_value,
        history={"value": values, "best": best_history, "iteration": iteration_history},
        extra={
            "restarts": params.max_restarts,
            "iterations": iterations,
            "stop_reason": "max_restarts",
            "iterations_per_restart": iterations_per_restart,
            "initial_values": initial_values,
            "restart_offsets": restart_offsets,
            "stop_reasons": stop_reasons,
            "best_restart": best_restart,
        },
    )


def run_hill_climbing(problem: Problem, params: HCParams, seed: int) -> RunResult:
    if params.variant == "random-restart":
        return _run_random_restart(problem, params, seed)

    rng = random.Random(seed)
    start = time.perf_counter()
    outcome = _run_single_climb(
        problem, params, rng, random_state(problem, rng), params.variant
    )
    extra = {
        "iterations": outcome.iterations,
        "stop_reason": outcome.stop_reason,
    }
    if params.variant == "sideways":
        extra["sideways_moves"] = outcome.sideways_moves

    return RunResult(
        algorithm=f"hc-{params.variant}",
        params=asdict(params),
        seed=seed,
        duration_s=time.perf_counter() - start,
        initial_state=outcome.initial_state,
        initial_value=outcome.initial_value,
        final_state=outcome.final_state,
        final_value=outcome.final_value,
        history={"value": outcome.values, "best": outcome.best_values},
        extra=extra,
    )
