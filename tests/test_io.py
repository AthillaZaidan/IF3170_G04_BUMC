import json

import pytest

from src.core.io import load_problem, problem_to_dict, save_problem
from tests.factories import pkg, problem

INSTANCE = {
    "truck": {"w": 10, "l": 20, "h": 5, "max_capacity": 500},
    "packages": [
        {"id": "P01", "w": 3, "l": 4, "h": 1, "value": 60, "weight": 5,
         "fragile": False, "eta": 3},
    ],
}


def write(tmp_path, data):
    path = tmp_path / "instance.json"
    path.write_text(json.dumps(data))
    return path


def test_load_problem(tmp_path):
    prob = load_problem(write(tmp_path, INSTANCE))
    assert prob.truck.max_capacity == 500
    assert prob.packages[0].id == "P01" and prob.packages[0].l == 4


def test_problem_round_trip(tmp_path):
    prob = problem(pkg(1, 2, 3, 4, value=9, weight=2, fragile=True))
    path = tmp_path / "p.json"
    save_problem(prob, path)
    assert load_problem(path) == prob
    assert problem_to_dict(prob)["packages"][0]["fragile"] is True


@pytest.mark.parametrize("breakage, message", [
    (lambda d: d["packages"][0].update(w=0), "positive integer"),
    (lambda d: d["packages"][0].update(weight=2.5), "positive integer"),
    (lambda d: d["truck"].pop("max_capacity"), "max_capacity"),
    (lambda d: d["packages"].append(dict(d["packages"][0])), "duplicate"),
])
def test_invalid_instances_are_rejected(tmp_path, breakage, message):
    data = json.loads(json.dumps(INSTANCE))
    breakage(data)
    with pytest.raises(ValueError, match=message):
        load_problem(write(tmp_path, data))
