import random

from src.algorithms.genetic import GAParams, run_genetic
from src.core.io import load_result
from src.experiments.artifacts import save_run
from src.generate_instance import generate
from src.main import main


def test_save_run_writes_result_and_pictures(tmp_path):
    prob = generate("small", random.Random(0))
    result = run_genetic(prob, GAParams(6, 3), seed=1)
    run_dir = tmp_path / "run"
    save_run(prob, result, run_dir, "pop6 gen3 seed1")
    assert load_result(run_dir / "result.json") == result
    for name in ("initial.png", "final.png", "curve.png"):
        assert (run_dir / name).exists()


def test_cli_runs_ga_once(tmp_path, capsys):
    out = tmp_path / "single"
    main(["ga", "--instance", "data/instances/small.json", "--pop", "6",
          "--gens", "3", "--seed", "2", "--out", str(out)])
    assert (out / "result.json").exists()
    assert "final value" in capsys.readouterr().out
