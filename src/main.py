"""Run one local search algorithm once and save its results.

Usage:
  python -m src.main ga --instance data/instances/medium.json --pop 50 --gens 100 --seed 1

Add HC/SA as more subcommands here.
"""
import argparse
from pathlib import Path

from src.algorithms.genetic import GAParams, run_genetic
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

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
