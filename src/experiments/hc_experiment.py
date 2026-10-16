import argparse
import zlib
from pathlib import Path

import pandas as pd

from src.algorithms.hill_climbing import HC_VARIANTS, HCParams, run_hill_climbing
from src.core.bounds import value_upper_bound
from src.core.io import load_problem
from src.experiments.artifacts import save_run
from src.viz.sa_plots import plot_metric_box

VARIANTS = ["steepest", "stochastic", "sideways", "random-restart"]
assert set(VARIANTS) == HC_VARIANTS


def run_seed(variant: str, seed: int) -> int:
    return zlib.crc32(f"hc-{variant}-seed{seed}".encode())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hill-Climbing variants, several seeds each")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/hc"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-sideways", type=int, default=20)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args(argv)

    problem = load_problem(args.instance)
    bound = value_upper_bound(problem)
    root = args.out / args.instance.stem
    params = HCParams(max_iterations=args.iters, neighbor_samples=args.samples,
                      max_sideways=args.max_sideways, max_restarts=args.max_restarts)
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            result = run_hill_climbing(problem, HCParams(**{**params.__dict__, "variant": variant}),
                                       run_seed(variant, seed))
            save_run(problem, result, root / variant / f"seed{seed}", f"HC {variant}, seed {seed}")
            e = result.extra
            print(f"{variant} seed {seed}: {result.initial_value} -> {result.final_value} "
                  f"({result.duration_s:.2f}s)")
            rows.append({
                "config": variant, "seed": seed, "run_seed": result.seed,
                "initial_value": result.initial_value, "final_value": result.final_value,
                "upper_bound": bound,
                "gap_to_bound": round(1 - result.final_value / bound, 4),
                "duration_s": round(result.duration_s, 4),
                "iterations": e.get("iterations", sum(e.get("iterations_per_restart", []))),
                "stop_reason": e.get("stop_reason", "restarts_done"),
            })

    summary = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "summary.csv", index=False)
    plot_metric_box(summary, "final_value", "Final objective (total value)",
                    "Final objective per HC variant", root / "final_value_box.png")

    print(f"\nupper bound {bound}")
    print(summary.groupby("config", sort=False)
          .agg(mean_final=("final_value", "mean"), std_final=("final_value", "std"),
               mean_iters=("iterations", "mean"), mean_duration_s=("duration_s", "mean"))
          .round(3).to_string())
    print(f"\nsaved to {root}")


if __name__ == "__main__":
    main()
