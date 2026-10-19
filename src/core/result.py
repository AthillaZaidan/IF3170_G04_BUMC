"""One record per algorithm run, in the same shape for HC, SA and GA, so
experiments, plots and the report can treat every algorithm alike."""
from dataclasses import dataclass, field

from src.core.state import State


@dataclass
class RunResult:
    algorithm: str  # "ga", "hc-steepest", "sa", ...
    params: dict
    seed: int
    duration_s: float
    initial_state: State
    initial_value: float
    final_state: State
    final_value: float
    history: dict[str, list[float]]  # e.g. {"max": [...], "mean": [...]} per iteration
    extra: dict = field(default_factory=dict)  # algorithm-specific numbers
