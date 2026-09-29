from dataclasses import FrozenInstanceError

import pytest

from src.core.models import Package, Problem, Truck


def test_models_are_immutable():
    pkg = Package("P1", 3, 4, 1, value=60, weight=5, fragile=False, eta=3)
    problem = Problem(Truck(10, 20, 5, 500), (pkg,))
    with pytest.raises(FrozenInstanceError):
        pkg.value = 99
    with pytest.raises(FrozenInstanceError):
        problem.truck = Truck(1, 1, 1, 1)
