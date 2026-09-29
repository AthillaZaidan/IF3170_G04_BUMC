"""Small builders so tests can describe packing scenarios in one line each."""
from src.core.models import Package, Problem, Truck
from src.core.state import Placement


def pkg(i, w=1, l=1, h=1, value=10, weight=1, fragile=False):
    return Package(f"P{i}", w, l, h, value=value, weight=weight, fragile=fragile, eta=1)


def problem(*packages, truck=(10, 10, 10), capacity=1000):
    return Problem(Truck(*truck, max_capacity=capacity), tuple(packages))


def at(x, y, z, orientation=("x", "y", "z")):
    return Placement(True, x, y, z, orientation)


OUT = Placement(False)
