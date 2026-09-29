"""Seeded generator for problem instances.

Package counts are chosen so the total volume is about 1.4-1.7x the truck
volume, and weights are scaled so the total is about 1.5x the capacity. That
way both space and weight limit the solution; otherwise the task is trivial.

Usage: python -m src.generate_instance --size medium --seed 42
"""
import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from src.core.io import save_problem
from src.core.models import Package, Problem, Truck

WEIGHT_RATIO = 1.5  # total weight / capacity


@dataclass(frozen=True)
class Preset:
    truck: Truck
    count: int


PRESETS = {
    "small": Preset(Truck(6, 8, 5, max_capacity=150), 15),
    "medium": Preset(Truck(10, 20, 5, max_capacity=500), 55),  # truck from the spec example
    "large": Preset(Truck(12, 24, 8, max_capacity=1200), 120),
}


def generate(size: str, rng: random.Random) -> Problem:
    preset = PRESETS[size]
    raw = [
        dict(w=rng.randint(1, 5), l=rng.randint(1, 5), h=rng.randint(1, 5),
             value=rng.randint(10, 100), weight=rng.randint(1, 30),
             fragile=rng.random() < 0.2, eta=rng.randint(1, 7))
        for _ in range(preset.count)
    ]
    scale = WEIGHT_RATIO * preset.truck.max_capacity / sum(r["weight"] for r in raw)
    packages = tuple(
        Package(id=f"P{i:03d}", **{**r, "weight": max(1, round(r["weight"] * scale))})
        for i, r in enumerate(raw, start=1)
    )
    return Problem(preset.truck, packages)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", choices=PRESETS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="default: data/instances/<size>.json")
    args = parser.parse_args(argv)

    out = args.out or Path("data/instances") / f"{args.size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    problem = generate(args.size, random.Random(args.seed))
    save_problem(problem, out)
    print(f"wrote {out}: {len(problem.packages)} packages")


if __name__ == "__main__":
    main()
