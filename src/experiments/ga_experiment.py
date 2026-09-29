"""GA experiment required by the spec:
- population as control, three numbers of generations;
- generations as control, three population sizes;
- every configuration run once per seed (3 seeds by default).

Usage: python -m src.experiments.ga_experiment --instance data/instances/medium.json

A configuration that appears in both groups (e.g. pop 50, gen 100) runs once
per seed and is reused, since the same seed gives the same result.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.algorithms.genetic import GAParams, run_genetic
from src.core.io import load_problem
from src.experiments.artifacts import save_run
from src.viz.plots import plot_final_values, plot_ga_comparison


def config_name(pop: int, gens: int) -> str:
    return f"pop{pop}_gen{gens}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GA population/generation experiment")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/ga"))
    parser.add_argument("--control-pop", type=int, default=50)
    parser.add_argument("--gens-list", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--control-gens", type=int, default=100)
    parser.add_argument("--pop-list", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elitism", type=int, default=1)
    args = parser.parse_args(argv)

    problem = load_problem(args.instance)
    root = args.out / args.instance.stem
    groups = {
        "generations": [(args.control_pop, g) for g in args.gens_list],
        "population": [(p, args.control_gens) for p in args.pop_list],
    }

    results = {}  # (pop, gens, seed) -> RunResult, shared between groups
    rows, curves = [], []
    for group, configs in groups.items():
        for pop, gens in configs:
            for seed in args.seeds:
                key = (pop, gens, seed)
                if key not in results:
                    params = GAParams(pop, gens, args.mutation_rate, args.elitism)
                    results[key] = run_genetic(problem, params, seed)
                    save_run(problem, results[key],
                             root / config_name(pop, gens) / f"seed{seed}",
                             f"GA pop {pop}, gen {gens}, seed {seed}")
                    print(f"{config_name(pop, gens)} seed {seed}: "
                          f"{results[key].final_value} ({results[key].duration_s:.2f}s)")
                result = results[key]
                rows.append({
                    "group": group, "config": config_name(pop, gens),
                    "population": pop, "generations": gens, "seed": seed,
                    "initial_value": result.initial_value,
                    "final_value": result.final_value,
                    "duration_s": round(result.duration_s, 4),
                    "best_generation": result.extra["best_generation"],
                })
                curves += [{"group": group, "config": config_name(pop, gens),
                            "seed": seed, "generation": g, "max": m}
                           for g, m in enumerate(result.history["max"])]

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    curves = pd.DataFrame(curves)
    plot_ga_comparison(
        curves[curves.group == "generations"],
        f"Population fixed at {args.control_pop}: varying generations",
        root / "compare_generations.png")
    plot_ga_comparison(
        curves[curves.group == "population"],
        f"Generations fixed at {args.control_gens}: varying population",
        root / "compare_population.png")
    plot_final_values(
        summary.drop_duplicates(["config", "seed"]),
        "Final objective per configuration", root / "final_value_box.png")

    print()
    print(summary.groupby(["group", "config"], sort=False)
          .agg(mean_final=("final_value", "mean"), std_final=("final_value", "std"),
               mean_duration_s=("duration_s", "mean"))
          .round(2).to_string())
    print(f"\nsaved to {root}")


if __name__ == "__main__":
    main()
