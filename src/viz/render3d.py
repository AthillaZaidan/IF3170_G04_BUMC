"""3D picture of a state: the truck, the packages inside it, and a list of the
packages left outside. The title carries the value, as the spec asks."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.core.geometry import Box, box
from src.core.models import Problem
from src.core.objective import total_value
from src.core.orientation import extents
from src.core.state import State

MAX_OUTSIDE_LINES = 45


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
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_axes((0.0, 0.0, 0.72, 0.92), projection="3d")
    side = fig.add_axes((0.73, 0.02, 0.26, 0.9))
    side.axis("off")

    _draw_truck(ax, problem)
    colors = plt.get_cmap("tab20")
    inside, outside = [], []
    for i, (package, placement) in enumerate(zip(problem.packages, state)):
        if not placement.in_truck:
            outside.append((package, placement))
            continue
        inside.append(package)
        b = box(package, placement)
        ax.add_collection3d(Poly3DCollection(
            _faces(b),
            facecolors=colors(i % 20), alpha=0.55,
            edgecolors="red" if package.fragile else "black",
            linewidths=1.8 if package.fragile else 0.4,
        ))
        ax.text((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2,
                package.id, fontsize=6, ha="center")

    ax.set(xlim=(0, truck.w), ylim=(0, truck.l), zlim=(0, truck.h),
           xlabel="x (w)", ylabel="y (l)", zlabel="z (h)")
    ax.set_box_aspect((truck.w, truck.l, truck.h))
    ax.view_init(elev=22, azim=-58)

    weight = sum(p.weight for p in inside)
    fig.suptitle(
        f"{title}\nValue = {total_value(problem, state)}  |  "
        f"Weight {weight}/{truck.max_capacity}  |  "
        f"{len(inside)} inside, {len(outside)} outside  |  red edge = fragile",
        fontsize=11,
    )

    lines = [f"Outside the truck ({len(outside)}):"]
    for package, placement in outside[:MAX_OUTSIDE_LINES]:
        dx, dy, dz = extents(package, placement.orientation)
        flag = " F" if package.fragile else ""
        lines.append(f"{package.id} {dx}x{dy}x{dz} v={package.value} w={package.weight}{flag}")
    if len(outside) > MAX_OUTSIDE_LINES:
        lines.append(f"... and {len(outside) - MAX_OUTSIDE_LINES} more")
    side.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7)

    fig.savefig(path, dpi=120)
    plt.close(fig)
