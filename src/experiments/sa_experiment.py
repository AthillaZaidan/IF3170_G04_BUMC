import argparse
import zlib
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.algorithms.simulated_annealing import SCHEDULES, SAParams, run_simulated_annealing
from src.core.bounds import value_upper_bound
from src.core.io import load_problem
from src.core.models import Problem
from src.core.result import RunResult
from src.experiments.artifacts import save_run
from src.viz.sa_plots import plot_eet, plot_metric_box, plot_sa_comparison, plot_sa_curve

CURVE_POINTS = 400  # points per run kept for the comparison plot


def config_name(schedule: str, p0: float, t_min: float, iterations: int,
                ratio_bias: float = 0.0) -> str:
    name = f"{schedule}_iter{iterations}_p{p0}_tmin{t_min}"
    return f"{name}_rb{ratio_bias}" if ratio_bias else name


# Deterministic seed per run and configuration; the key format is fixed so saved runs reproduce.
def run_seed(schedule: str, p0: float, t_min: float, iterations: int, seed: int,
             ratio_bias: float = 0.0) -> int:
    key = f"{schedule}_p0{p0}_tmin{t_min}_it{iterations}"
    if ratio_bias:
        key += f"_rb{ratio_bias}"
    return zlib.crc32(f"sa-{key}-seed{seed}".encode())


# save_run plus the SA curve with temperature and the e^(dE/T) plot.
def save_sa_run(problem: Problem, result: RunResult, run_dir: Path, label: str) -> None:
    save_run(problem, result, run_dir, label)
    plot_sa_curve(result.history, f"Objective per iteration - {label}", run_dir / "curve.png")
    plot_eet(result.history, f"e^(dE/T) per iteration - {label}", run_dir / "eet.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulated Annealing experiment grid")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--trucks", type=int)
    parser.add_argument("--objective", choices=("value", "eta"))
    parser.add_argument("--out", type=Path, default=Path("output/sa"))
    parser.add_argument("--iterations", type=int, nargs="+", default=[20000])
    parser.add_argument("--schedules", nargs="+", choices=SCHEDULES, default=["geometric"])
    parser.add_argument("--p0", type=float, nargs="+", default=[0.8])
    parser.add_argument("--t-min", type=float, nargs="+", default=[SAParams.t_min])
    parser.add_argument("--ratio-bias", type=float, nargs="+", default=[0.0])
    parser.add_argument("--patience", type=int, default=SAParams.stuck_patience)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args(argv)

    problem = load_problem(args.instance)
    if args.trucks is not None:
        problem = replace(problem, truck_count=args.trucks)
    if args.objective is not None:
        problem = replace(problem, objective=args.objective)
    bound = value_upper_bound(problem)
    instance_name = args.instance.stem + (f"_trucks{problem.truck_count}" if problem.truck_count > 1 else "")
    if problem.objective != "value":
        instance_name += f"_{problem.objective}"
    root = args.out / instance_name
    rows, curves = [], []
    grid = [(s, p, t, n, b) for s in args.schedules for p in args.p0
            for t in args.t_min for n in args.iterations for b in args.ratio_bias]
    for schedule, p0, t_min, iterations, ratio_bias in grid:
        config = config_name(schedule, p0, t_min, iterations, ratio_bias)
        params = SAParams(max_iterations=iterations, schedule=schedule, p0=p0, t_min=t_min,
                          stuck_patience=args.patience, ratio_bias=ratio_bias)
        for seed in args.seeds:
            result = run_simulated_annealing(
                problem, params, run_seed(schedule, p0, t_min, iterations, seed, ratio_bias))
            save_sa_run(problem, result, root / config / f"seed{seed}",
                        f"SA {config}, seed {seed}")
            e = result.extra
            print(f"{config} seed {seed}: {result.final_value} "
                  f"(stuck {e['stuck_count']}, {result.duration_s:.2f}s)")
            rows.append({
                "config": config, "schedule": schedule, "p0": p0, "t_min": t_min,
                "objective": problem.objective, "truck_count": problem.truck_count,
                "ratio_bias": ratio_bias, "iterations": iterations, "seed": seed,
                "run_seed": result.seed,
                "initial_value": result.initial_value,
                "final_value": result.final_value,
                "upper_bound": bound,
                "gap_to_bound": round(1 - result.final_value / bound, 4) if bound else 0,
                "duration_s": round(result.duration_s, 4),
                "t0": round(e["t0"], 3),
                "stuck_count": e["stuck_count"],
                "stuck_iterations": e["stuck_iterations"],
                "accepted_worse": e["accepted_worse"],
                "null_moves": e["null_moves"],
                "best_iteration": e["best_iteration"],
            })
            best = result.history["best"]
            step = max(1, len(best) // CURVE_POINTS)
            curves += [{"config": config, "seed": seed, "iteration": k, "best": best[k]}
                       for k in range(0, len(best), step)]

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    plot_sa_comparison(pd.DataFrame(curves), "Best objective per configuration",
                       root / "compare_best.png")
    plot_metric_box(summary, "final_value", "Final objective",
                    "Final objective per configuration", root / "final_value_box.png")
    plot_metric_box(summary, "stuck_count", "Times stuck in a local optimum",
                    "Stuck frequency per configuration", root / "stuck_count_box.png")

    print(f"\nupper bound {bound}")
    print(summary.groupby("config", sort=False)
          .agg(mean_final=("final_value", "mean"), std_final=("final_value", "std"),
               mean_gap=("gap_to_bound", "mean"), mean_stuck=("stuck_count", "mean"),
               mean_duration_s=("duration_s", "mean"))
          .round(3).to_string())
    print(f"\nsaved to {root}")


if __name__ == "__main__":
    main()
