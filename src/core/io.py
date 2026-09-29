"""JSON input/output for problem instances, states and run results."""
import json
from pathlib import Path

from dataclasses import asdict

from src.core.models import Package, Problem, Truck
from src.core.result import RunResult
from src.core.state import Placement, State

_POSITIVE_INT_FIELDS = {
    "truck": ("w", "l", "h", "max_capacity"),
    "package": ("w", "l", "h", "weight"),
}


def _require_positive_ints(obj: dict, fields: tuple[str, ...], where: str) -> None:
    for name in fields:
        if name not in obj:
            raise ValueError(f"{where}: missing field '{name}'")
        value = obj[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{where}: '{name}' must be a positive integer, got {value!r}")


def problem_from_dict(data: dict) -> Problem:
    _require_positive_ints(data["truck"], _POSITIVE_INT_FIELDS["truck"], "truck")
    truck = Truck(**{k: data["truck"][k] for k in ("w", "l", "h", "max_capacity")})
    packages, seen = [], set()
    for raw in data["packages"]:
        where = f"package {raw.get('id', '?')}"
        _require_positive_ints(raw, _POSITIVE_INT_FIELDS["package"], where)
        if raw["id"] in seen:
            raise ValueError(f"{where}: duplicate package id")
        seen.add(raw["id"])
        packages.append(Package(
            id=raw["id"], w=raw["w"], l=raw["l"], h=raw["h"],
            value=raw["value"], weight=raw["weight"],
            fragile=bool(raw["fragile"]), eta=raw["eta"],
        ))
    return Problem(truck, tuple(packages))


def problem_to_dict(problem: Problem) -> dict:
    t = problem.truck
    return {
        "truck": {"w": t.w, "l": t.l, "h": t.h, "max_capacity": t.max_capacity},
        "packages": [
            {"id": p.id, "w": p.w, "l": p.l, "h": p.h, "value": p.value,
             "weight": p.weight, "fragile": p.fragile, "eta": p.eta}
            for p in problem.packages
        ],
    }


def load_problem(path: str | Path) -> Problem:
    return problem_from_dict(json.loads(Path(path).read_text()))


def save_problem(problem: Problem, path: str | Path) -> None:
    Path(path).write_text(json.dumps(problem_to_dict(problem), indent=2) + "\n")


def state_to_json(problem: Problem, state: State) -> list[dict]:
    """Placements with package ids and the spec's orientation notation."""
    return [
        {"id": pkg.id, "in_truck": p.in_truck, "x": p.x, "y": p.y, "z": p.z,
         "orientation": dict(zip(("w", "l", "h"), p.orientation))}
        for pkg, p in zip(problem.packages, state)
    ]


def state_from_json(data: list[dict]) -> State:
    return tuple(
        Placement(d["in_truck"], d["x"], d["y"], d["z"],
                  tuple(d["orientation"][side] for side in ("w", "l", "h")))
        for d in data
    )


def save_result(problem: Problem, result: RunResult, path: str | Path) -> None:
    data = asdict(result)
    data["initial_state"] = state_to_json(problem, result.initial_state)
    data["final_state"] = state_to_json(problem, result.final_state)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_result(path: str | Path) -> RunResult:
    data = json.loads(Path(path).read_text())
    data["initial_state"] = state_from_json(data["initial_state"])
    data["final_state"] = state_from_json(data["final_state"])
    return RunResult(**data)
