import pandas as pd

from src.core.io import load_result
from src.experiments.sa_experiment import config_name, run_seed
from src.experiments.sa_experiment import main as experiment_main
from src.main import main


def test_cli_runs_sa_once(tmp_path, capsys):
    out = tmp_path / "single"
    main(["sa", "--instance", "data/instances/small.json", "--iters", "200",
          "--seed", "2", "--out", str(out)])
    for name in ("result.json", "initial.png", "final.png", "curve.png", "eet.png"):
        assert (out / name).exists()
    assert load_result(out / "result.json").algorithm == "sa"
    printed = capsys.readouterr().out
    assert "final value" in printed and "stuck" in printed


def test_sa_experiment_grid(tmp_path):
    experiment_main([
        "--instance", "data/instances/small.json", "--out", str(tmp_path),
        "--iterations", "100", "200", "--schedules", "geometric", "linear",
        "--p0", "0.8", "--t-min", "0.05", "2", "--seeds", "1", "2",
    ])
    root = tmp_path / "small"
    summary = pd.read_csv(root / "summary.csv")
    assert len(summary) == 2 * 2 * 2 * 2
    assert summary["config"].nunique() == 2 * 2 * 2
    assert {"final_value", "gap_to_bound", "duration_s", "stuck_count",
            "t0"} <= set(summary.columns)
    assert (summary["gap_to_bound"] >= 0).all()
    run_dir = root / config_name("linear", 0.8, 2.0, 200) / "seed2"
    for name in ("result.json", "final.png", "eet.png"):
        assert (run_dir / name).exists()
    for name in ("compare_best.png", "final_value_box.png", "stuck_count_box.png"):
        assert (root / name).exists()


def test_run_seeds_differ_between_configs_but_are_stable():
    assert run_seed("geometric", 0.8, 0.05, 100, 1) == run_seed("geometric", 0.8, 0.05, 100, 1)
    seeds = {run_seed(s, p, t, n, k) for s in ("geometric", "linear")
             for p in (0.5, 0.8) for t in (0.05, 2.0) for n in (100, 200) for k in (1, 2)}
    assert len(seeds) == 32


def test_config_name_tells_every_grid_axis_apart():
    names = {config_name(s, p, t, n) for s in ("geometric", "linear")
             for p in (0.5, 0.8) for t in (0.05, 2.0) for n in (100, 200)}
    assert len(names) == 16


def test_config_name_uses_the_same_key_value_style_as_ga():
    assert config_name("geometric", 0.01, 0.05, 20000) == "geometric_iter20000_p0.01_tmin0.05"
    assert config_name("geometric", 0.01, 0.05, 20000, 0.0) == "geometric_iter20000_p0.01_tmin0.05"
    assert config_name("geometric", 0.01, 0.05, 20000, 0.5) == "geometric_iter20000_p0.01_tmin0.05_rb0.5"


# Saved results in output/sa must stay reproducible after the folder rename.
def test_run_seed_matches_the_saved_results():
    assert run_seed("geometric", 0.01, 0.05, 20000, 1) == 1247023208
    assert run_seed("geometric", 0.01, 0.05, 20000, 1) == run_seed("geometric", 0.01, 0.05, 20000, 1, 0.0)
    assert run_seed("geometric", 0.01, 0.05, 20000, 1) != run_seed("geometric", 0.01, 0.05, 20000, 1, 0.2)


def test_cli_accepts_ratio_bias(tmp_path):
    out = tmp_path / "rb"
    main(["sa", "--instance", "data/instances/small.json", "--iters", "100",
          "--ratio-bias", "0.3", "--out", str(out)])
    assert load_result(out / "result.json").params["ratio_bias"] == 0.3
