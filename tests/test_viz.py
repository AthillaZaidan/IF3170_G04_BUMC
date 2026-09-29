import random

from src.core.initial import random_state
from src.core.repair import repair
from src.generate_instance import generate
from src.viz.render3d import render_state


def test_render_state_writes_png(tmp_path):
    prob = generate("small", random.Random(0))
    state = repair(prob, random_state(prob, random.Random(0)))
    path = tmp_path / "state.png"
    render_state(prob, state, "Final state", path)
    assert path.stat().st_size > 10_000
