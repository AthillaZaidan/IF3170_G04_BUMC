import pandas as pd

from src.experiments.hc_experiment import main


def test_hc_experiment_runs_every_variant(tmp_path):
    main(["--instance", "data/instances/small.json", "--out", str(tmp_path),
          "--iters", "5", "--max-restarts", "1", "--seeds", "1", "2"])
    root = tmp_path / "small"
    summary = pd.read_csv(root / "summary.csv")
    assert len(summary) == 4 * 2
    assert set(summary["config"]) == {"steepest", "stochastic", "sideways", "random-restart"}
    assert (summary["gap_to_bound"] >= 0).all()
    assert (root / "sideways" / "seed2" / "result.json").exists()
    assert (root / "final_value_box.png").exists()
