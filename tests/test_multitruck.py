import json
import random
from dataclasses import replace

import pytest

from src.algorithms.genetic import GAParams, run_genetic
from src.algorithms.hill_climbing import HCParams, run_hill_climbing
from src.algorithms.simulated_annealing import SAParams, run_simulated_annealing
from src.core.bounds import value_upper_bound
from src.core.constraints import find_violations, is_feasible
from src.core.geometry import box, within_walls
from src.core.initial import random_state
from src.core.io import (load_problem, load_result, problem_from_dict, problem_to_dict,
                         save_problem, save_result, state_from_json, state_to_json)
from src.core.neighbors import all_neighbors, candidate_positions, move, random_neighbor, swap
from src.core.objective import total_value
from src.core.repair import repair
from src.core.state import take_out
from src.generate_instance import generate
from src.main import main
from tests.factories import OUT, at, pkg, problem


def fleet(*packages, truck=(2, 2, 2), capacity=10):
    return replace(problem(*packages, truck=truck, capacity=capacity), truck_count=2)


def test_constraints_are_independent_per_truck():
    prob = fleet(pkg(0, 2, 2, 2, weight=10, fragile=True),
                 pkg(1, 2, 2, 2, weight=10), capacity=10)
    state = (at(0, 0, 0), replace(at(0, 0, 0), truck_id=1))
    assert is_feasible(prob, state)
    assert repair(prob, state) == state
    assert total_value(prob, state) == 20
    assert not is_feasible(prob, (state[0], replace(state[1], truck_id=0)))


def test_support_and_fragility_do_not_cross_trucks():
    prob = fleet(pkg(0, fragile=True), pkg(1))
    state = (at(0, 0, 0), replace(at(0, 0, 1), truck_id=1))
    assert [v.kind for v in find_violations(prob, state)] == ["floating"]
    fixed = repair(prob, state)
    assert fixed == (state[0], replace(state[1], z=0))
    assert is_feasible(prob, fixed)


def test_repair_limits_weight_in_each_truck():
    prob = fleet(pkg(0, weight=6, value=20), pkg(1, weight=6, value=5),
                 pkg(2, weight=10, value=30))
    state = (at(0, 0, 0), at(1, 0, 0), replace(at(0, 0, 0), truck_id=1))
    fixed = repair(prob, state)
    assert fixed == (state[0], OUT, state[2])
    assert is_feasible(prob, fixed)


@pytest.mark.parametrize("truck_id", [-1, 2, True, 0.5, None, "1"])
def test_invalid_truck_assignments_are_rejected_and_repaired(truck_id):
    prob = fleet(pkg(0))
    state = (replace(at(0, 0, 0), truck_id=truck_id),)
    assert [v.kind for v in find_violations(prob, state)] == ["truck"]
    assert repair(prob, state) == (OUT,)


def test_transfer_swap_and_removal_preserve_assignments():
    state = (at(0, 0, 0), replace(at(1, 0, 0), truck_id=1))
    moved = move(state, 0, (0, 1, 0), truck_id=1)
    assert moved[0] == replace(at(0, 1, 0), truck_id=1)
    assert move(moved, 0, (1, 1, 0))[0].truck_id == 1
    assert swap(state, 0, 1) == (state[1], state[0])
    assert take_out(moved[0]) == OUT


def test_neighborhood_reaches_empty_trucks_and_uses_local_extreme_points():
    prob = fleet(pkg(0), pkg(1))
    state = (at(0, 0, 0), OUT)
    assert candidate_positions(prob, state, 1, truck_id=1) == [(0, 0, 0)]
    assert (1, 0, 0) in candidate_positions(prob, state, 1, truck_id=0)
    neighbors = list(all_neighbors(prob, state))
    assert any(n[0].in_truck and n[0].truck_id == 1 and n[0].x == 0
               and m.i == 0 and m.kind == "move" for n, m in neighbors)
    assert any(n[1].in_truck and n[1].truck_id == 1 for n, _ in neighbors)
    rng = random.Random(3)
    assert any(random_neighbor(prob, state, rng)[0][1].truck_id == 1 for _ in range(100))


def test_initial_state_uses_all_trucks_and_stays_within_walls():
    prob = fleet(*(pkg(i) for i in range(20)))
    state = random_state(prob, random.Random(2), p_in=1)
    assert {p.truck_id for p in state} == {0, 1}
    assert all(within_walls(box(pkg, p), prob.truck) for pkg, p in zip(prob.packages, state))
    assert state == random_state(prob, random.Random(2), p_in=1)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2", None])
def test_invalid_fleet_size_is_rejected(count):
    with pytest.raises(ValueError, match="truck_count"):
        replace(problem(pkg(0)), truck_count=count)


def test_fleet_problem_and_state_round_trip(tmp_path):
    prob = replace(fleet(pkg(0), pkg(1)), objective="eta")
    state = (OUT, replace(at(0, 0, 0), truck_id=1))
    path = tmp_path / "fleet.json"
    save_problem(prob, path)
    assert load_problem(path) == prob
    assert problem_to_dict(prob)["truck_count"] == 2
    assert state_from_json(state_to_json(prob, state)) == state
    legacy = problem_to_dict(problem(pkg(0)))
    assert problem_from_dict(legacy).truck_count == 1
    assert state_from_json(state_to_json(problem(pkg(0)), (at(0, 0, 0),)))[0].truck_id == 0


def test_fleet_bound_accounts_for_total_capacity_without_splitting_packages():
    prob = fleet(pkg(0, 2, 2, 2, value=20, weight=10),
                 pkg(1, 2, 2, 2, value=10, weight=10),
                 pkg(2, 2, 2, 2, value=5, weight=10),
                 pkg(3, value=100, weight=11))
    assert value_upper_bound(replace(prob, truck_count=1)) == 20
    assert value_upper_bound(prob) == 30


@pytest.mark.parametrize("variant", ["steepest", "stochastic", "sideways", "random-restart"])
def test_each_hc_variant_supports_fleet(variant, monkeypatch):
    prob = fleet(pkg(0, value=10), pkg(1, value=20), truck=(1, 1, 1))
    monkeypatch.setattr("src.algorithms.hill_climbing.random_state", lambda p, rng: (OUT, OUT))
    result = run_hill_climbing(prob, HCParams(variant=variant, max_iterations=10,
                                            neighbor_samples=100, max_restarts=1), 1)
    assert is_feasible(prob, result.final_state)
    assert result.final_value == 30
    assert {p.truck_id for p in result.final_state} == {0, 1}


@pytest.mark.parametrize("algorithm", ["ga", "sa"])
def test_population_and_annealing_algorithms_support_fleet(algorithm, tmp_path):
    prob = fleet(*(pkg(i, value=10 + i, weight=6) for i in range(4)))
    if algorithm == "ga":
        run = lambda: run_genetic(prob, GAParams(8, 6), 5)
    else:
        run = lambda: run_simulated_annealing(prob, SAParams(max_iterations=80, t0=5), 5)
    result = run()
    assert is_feasible(prob, result.initial_state)
    assert is_feasible(prob, result.final_state)
    assert result.final_value == total_value(prob, result.final_state)
    assert result.final_value <= value_upper_bound(prob)
    assert result.final_state == run().final_state
    path = tmp_path / "result.json"
    save_result(prob, result, path)
    assert load_result(path) == result
    assert json.loads(path.read_text())["problem"]["truck_count"] == 2


@pytest.mark.parametrize("algorithm, options", [
    ("hc", ["--iters", "2"]),
    ("sa", ["--iters", "3", "--t0", "5"]),
    ("ga", ["--pop", "4", "--gens", "1"]),
])
def test_cli_saves_fleet_assignments_and_artifacts(algorithm, options, tmp_path):
    prob = fleet(pkg(0), pkg(1))
    instance = tmp_path / "fleet.json"
    save_problem(replace(prob, truck_count=1), instance)
    out = tmp_path / "out"
    main([algorithm, "--instance", str(instance), "--trucks", "2", "--out", str(out), *options])
    data = json.loads((out / "result.json").read_text())
    assert data["problem"]["truck_count"] == 2
    assert all("truck_id" in p for p in data["final_state"])
    assert is_feasible(prob, load_result(out / "result.json").final_state)
    for name in ("initial.png", "final.png", "curve.png"):
        assert (out / name).stat().st_size > 1000


def test_generator_accepts_fleet_size():
    prob = generate("small", random.Random(1), truck_count=3)
    assert prob.truck_count == 3
