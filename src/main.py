"""Run one local search algorithm once and save its results.

Usage:
  python -m src.main ga --instance data/instances/medium.json --pop 50 --gens 100 --seed 1
  python -m src.main hc --instance data/instances/medium.json --variant steepest --iters 100 --seed 1
"""
import argparse
from pathlib import Path

from src.algorithms.genetic import GAParams, run_genetic
from src.algorithms.hill_climbing import HCParams, run_hill_climbing
from src.core.io import load_problem
from src.experiments.artifacts import save_run


def _run_ga(args: argparse.Namespace) -> None:
    problem = load_problem(args.instance)
    params = GAParams(args.pop, args.gens, args.mutation_rate, args.elitism)
    result = run_genetic(problem, params, args.seed)
    out = args.out or Path("output/ga") / args.instance.stem / "single" / (
        f"pop{args.pop}_gen{args.gens}_seed{args.seed}")
    save_run(problem, result, out, f"GA pop {args.pop}, gen {args.gens}, seed {args.seed}")
    print(f"initial value {result.initial_value} -> final value {result.final_value} "
          f"(best at generation {result.extra['best_generation']}, "
          f"{result.duration_s:.2f}s)")
    print(f"saved to {out}")


def _run_hc(args: argparse.Namespace) -> None:
    problem = load_problem(args.instance)
    params = HCParams(
        variant=args.variant,
        max_iterations=args.iters,
        neighbor_samples=args.samples,
        max_sideways=args.max_sideways,
        max_restarts=args.max_restarts,
    )
    result = run_hill_climbing(problem, params, args.seed)
    out = args.out or Path("output/hc") / args.instance.stem / "single" / (
        f"{args.variant}_iter{args.iters}_seed{args.seed}")
    save_run(problem, result, out, f"HC {args.variant}, iter {args.iters}, seed {args.seed}")

    details = [f"initial value {result.initial_value} -> final value {result.final_value}"]
    if args.variant == "random-restart":
        details.append(f"restarts {result.extra['restarts']}")
        details.append(f"best restart {result.extra['best_restart']}")
    else:
        details.append(f"iterations {result.extra['iterations']}")
        details.append(f"stop {result.extra['stop_reason']}")
    details.append(f"{result.duration_s:.2f}s")
    print(", ".join(details))
    print(f"saved to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="3D truck packing with local search")
    sub = parser.add_subparsers(dest="algorithm", required=True)

    ga = sub.add_parser("ga", help="Genetic Algorithm")
    ga.add_argument("--instance", type=Path, required=True)
    ga.add_argument("--pop", type=int, default=50, help="population size")
    ga.add_argument("--gens", type=int, default=100, help="number of generations")
    ga.add_argument("--mutation-rate", type=float, default=0.2)
    ga.add_argument("--elitism", type=int, default=1)
    ga.add_argument("--seed", type=int, default=1)
    ga.add_argument("--out", type=Path, help="output folder for this run")
    ga.set_defaults(handler=_run_ga)

    hc = sub.add_parser("hc", help="Hill-Climbing variants")
    hc.add_argument("--instance", type=Path, required=True)
    hc.add_argument(
        "--variant",
        choices=("steepest", "stochastic", "sideways", "random-restart"),
        default="steepest",
    )
    hc.add_argument("--iters", type=int, default=100, help="maximum iterations per climb")
    hc.add_argument("--samples", type=int, default=50, help="random neighbors for stochastic HC")
    hc.add_argument("--max-sideways", type=int, default=20)
    hc.add_argument("--max-restarts", type=int, default=5)
    hc.add_argument("--seed", type=int, default=1)
    hc.add_argument("--out", type=Path, help="output folder for this run")
    hc.set_defaults(handler=_run_hc)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
