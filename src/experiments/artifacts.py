"""Write everything the spec asks to record for one run into one folder:
result.json (initial/final state, final objective, history, duration, params)
plus pictures of the initial state, the final state and the objective curve."""
from pathlib import Path

from src.core.io import save_result
from src.core.models import Problem
from src.core.result import RunResult
from src.viz.plots import plot_objective_curve
from src.viz.render3d import render_state


def save_run(problem: Problem, result: RunResult, run_dir: str | Path, label: str) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_result(problem, result, run_dir / "result.json")
    render_state(problem, result.initial_state, f"Initial state - {label}",
                 run_dir / "initial.png")
    render_state(problem, result.final_state, f"Final state - {label}",
                 run_dir / "final.png")
    plot_objective_curve(result.history, f"Objective per iteration - {label}",
                         run_dir / "curve.png")
