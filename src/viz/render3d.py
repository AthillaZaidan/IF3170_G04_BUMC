"""3D picture of a state: the truck, the packages inside it, and a list of the
packages left outside. The title carries the value, as the spec asks."""
from pathlib import Path
from math import ceil

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.core.geometry import Box, box
from src.core.models import Problem
from src.core.objective import total_value
from src.core.orientation import extents
from src.core.state import State

def _faces(b: Box) -> list[list[tuple[int, int, int]]]:
    x0, y0, z0, x1, y1, z1 = b
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[corners[k] for k in quad] for quad in quads]


def _draw_truck(ax, problem: Problem) -> None:
    t = problem.truck
    for face in _faces((0, 0, 0, t.w, t.l, t.h)):
        xs, ys, zs = zip(*(face + face[:1]))
        ax.plot(xs, ys, zs, color="0.35", linewidth=0.8)


def render_state(problem: Problem, state: State, title: str, path: str | Path) -> None:
    truck = problem.truck
    columns = min(problem.truck_count, 3)
    rows = ceil(problem.truck_count / columns)
    outside_count = sum(not p.in_truck for p in state)
    height = max(7, rows * 4.5 + 1.5, outside_count * 0.13 + 2.5)
    fig = plt.figure(figsize=(15 if columns > 1 else 13, height))
    grid = fig.add_gridspec(rows, columns, left=0.03, right=0.72,
                            bottom=0.05, top=0.81, wspace=0.15, hspace=0.3)
    side = fig.add_axes((0.77, 0.02, 0.22, 0.8))
    side.axis("off")
    colors = plt.get_cmap("tab20")
    inside = [pkg for pkg, p in zip(problem.packages, state) if p.in_truck]
    outside = [(pkg, p) for pkg, p in zip(problem.packages, state) if not p.in_truck]
    for truck_id in range(problem.truck_count):
        ax = fig.add_subplot(grid[truck_id // columns, truck_id % columns], projection="3d")
        _draw_truck(ax, problem)
        cargo = []
        for i, (package, placement) in enumerate(zip(problem.packages, state)):
            if not placement.in_truck or placement.truck_id != truck_id:
                continue
            cargo.append(package)
            b = box(package, placement)
            ax.add_collection3d(Poly3DCollection(
                _faces(b),
                facecolors=colors(i % 20), alpha=0.55,
                edgecolors="red" if package.fragile else "black",
                linewidths=1.8 if package.fragile else 0.4,
            ))
            ax.text((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2,
                    package.id, fontsize=6, ha="center")
        weight = sum(p.weight for p in cargo)
        ax.set(xlim=(0, truck.w), ylim=(0, truck.l), zlim=(0, truck.h),
               xlabel="x (w)", ylabel="y (l)", zlabel="z (h)",
               title=f"Truck {truck_id + 1} | Weight {weight}/{truck.max_capacity} | {len(cargo)} packages")
        ax.set_box_aspect((truck.w, truck.l, truck.h))
        ax.view_init(elev=22, azim=-58)

    weight = sum(p.weight for p in inside)
    value_label = f"Value = {total_value(problem, state):g}"
    if problem.objective == "eta":
        value_label = (f"Objective (value/ETA) = {total_value(problem, state):.4g} | "
                       f"Total value = {sum(p.value for p in inside):g}")
    fig.suptitle(
        f"{title}\n{value_label}  |  "
        f"Weight {weight}/{truck.max_capacity * problem.truck_count}  |  "
        f"{len(inside)} inside, {len(outside)} outside  |  red edge = fragile",
        fontsize=11,
    )

    lines = [f"Outside the fleet ({len(outside)}):"]
    for package, placement in outside:
        dx, dy, dz = extents(package, placement.orientation)
        flag = " F" if package.fragile else ""
        lines.append(f"{package.id} {dx}x{dy}x{dz} v={package.value} w={package.weight}{flag}")
    side.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7)

    fig.savefig(path, dpi=120)
    plt.close(fig)
