import pandas as pd

from src.viz.sa_plots import plot_eet, plot_metric_box, plot_sa_comparison, plot_sa_curve

HISTORY = {
    "value": [10, 12, 9, 15, 15],
    "best": [10, 12, 12, 15, 15],
    "temperature": [8.0, 4.0, 2.0, 1.0],
    "eET": [1.0, 0.4, 1.0, 1.0],
}


def test_sa_curve_and_eet_plots_write_pngs(tmp_path):
    plot_sa_curve(HISTORY, "curve", tmp_path / "curve.png")
    plot_eet(HISTORY, "eET", tmp_path / "eet.png")
    for name in ("curve.png", "eet.png"):
        assert (tmp_path / name).stat().st_size > 5_000


def test_eet_plot_handles_a_run_without_worse_moves(tmp_path):
    flat = dict(HISTORY, eET=[1.0, 1.0, 1.0, 1.0])
    plot_eet(flat, "eET", tmp_path / "eet.png")
    assert (tmp_path / "eet.png").exists()


def test_sa_comparison_and_metric_box_write_pngs(tmp_path):
    curves = pd.DataFrame([
        {"config": cfg, "seed": s, "iteration": k, "best": k * s}
        for cfg in ("a", "b") for s in (1, 2) for k in range(5)
    ])
    plot_sa_comparison(curves, "compare", tmp_path / "cmp.png")
    summary = pd.DataFrame([{"config": c, "stuck_count": v}
                            for c, v in [("a", 1), ("a", 2), ("b", 3)]])
    plot_metric_box(summary, "stuck_count", "Stuck count", "stuck", tmp_path / "box.png")
    for name in ("cmp.png", "box.png"):
        assert (tmp_path / name).stat().st_size > 5_000
