import random

import pytest

from src.generate_instance import PRESETS, generate


@pytest.mark.parametrize("size", PRESETS)
def test_generated_instances_make_space_and_weight_bind(size):
    prob = generate(size, random.Random(42))
    preset = PRESETS[size]
    truck = prob.truck
    assert len(prob.packages) == preset.count
    volume = sum(p.w * p.l * p.h for p in prob.packages)
    weight = sum(p.weight for p in prob.packages)
    assert volume > truck.w * truck.l * truck.h * 1.2
    assert 1.4 * truck.max_capacity <= weight <= 1.6 * truck.max_capacity
    assert all(p.weight >= 1 and 1 <= p.eta <= 7 for p in prob.packages)
    assert len({p.id for p in prob.packages}) == len(prob.packages)


def test_generation_is_reproducible():
    assert generate("medium", random.Random(1)) == generate("medium", random.Random(1))
