import random

import pandas as pd

from src.core.initial import random_state
from src.core.repair import repair
from src.generate_instance import generate
from src.viz.plots import plot_final_values, plot_ga_comparison, plot_ga_curve
from src.viz.render3d import render_state


def test_render_state_writes_png(tmp_path):
    prob = generate("small", random.Random(0))
    state = repair(prob, random_state(prob, random.Random(0)))
    path = tmp_path / "state.png"
    render_state(prob, state, "Final state", path)
    assert path.stat().st_size > 10_000


def test_ga_plots_write_pngs(tmp_path):
    plot_ga_curve({"max": [1, 3, 4], "mean": [0.5, 2, 3]}, "curve", tmp_path / "c.png")
    curves = pd.DataFrame([
        {"config": cfg, "seed": s, "generation": g, "max": g * s}
        for cfg in ("a", "b") for s in (1, 2) for g in range(4)
    ])
    plot_ga_comparison(curves, "compare", tmp_path / "cmp.png")
    summary = pd.DataFrame([{"config": c, "final_value": v}
                            for c, v in [("a", 1), ("a", 2), ("b", 3)]])
    plot_final_values(summary, "final", tmp_path / "box.png")
    for name in ("c.png", "cmp.png", "box.png"):
        assert (tmp_path / name).stat().st_size > 5_000
