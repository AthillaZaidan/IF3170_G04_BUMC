from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.viz.plots import GA_MAX_COLOR, GA_MEAN_COLOR

TEMPERATURE_COLOR = "tab:red"
MAX_POINTS = 2000  # curves are thinned to this many points before plotting


def _stride(n: int) -> int:
    return max(1, n // MAX_POINTS)


def _temperature_panel(ax, temperature: list[float]) -> None:
    step = _stride(len(temperature))
    ax.plot(range(0, len(temperature), step), temperature[::step],
            color=TEMPERATURE_COLOR, linewidth=1.5, marker="o" if len(temperature) == 1 else None)
    ax.set_yscale("log")
    ax.set(xlabel="Iteration", ylabel="Temperature T")


# Current and best objective per iteration, temperature in a panel below (no second y-axis).
def plot_sa_curve(history: dict[str, list[float]], title: str, path: str | Path) -> None:
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": (3, 1)})
    step = _stride(len(history["value"]))
    iterations = range(0, len(history["value"]), step)
    top.plot(iterations, history["value"][::step], color=GA_MAX_COLOR,
             linewidth=1, label="current")
    top.plot(iterations, history["best"][::step], color=GA_MEAN_COLOR,
             linewidth=2, label="best so far")
    top.set(title=title, ylabel="Objective")
    top.legend(loc="lower right")
    _temperature_panel(bottom, history["temperature"])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# e^(dE/T) of every worse move and its rolling mean; better moves (probability 1) are left out.
def _eet_figure(history: dict[str, list[float]], title: str, window: int = 200):
    eet = pd.Series(history["eET"])
    worse = eet[eet < 1.0]
    has_log = "log_eET" in history
    fig, axes = plt.subplots(
        3 if has_log else 2, 1, figsize=(9, 8 if has_log else 6), sharex=True,
        gridspec_kw={"height_ratios": (3, 2, 1) if has_log else (3, 1)})
    top = axes[0]
    if len(worse):
        shown = worse.iloc[::_stride(len(worse))]
        top.scatter(shown.index, shown.values, s=8, alpha=0.35, color=GA_MAX_COLOR,
                    linewidths=0, label="worse move")
        rolling = worse.rolling(min(window, len(worse)), min_periods=1).mean()
        top.plot(rolling.index, rolling.values, color=GA_MEAN_COLOR, linewidth=2,
                 label=f"rolling mean ({window} worse moves)")
        top.legend(loc="upper right")
    top.set(title=title, ylabel=r"$e^{\Delta E / T}$", ylim=(-0.03, 1.03))
    if has_log:
        exponents = pd.Series(history["log_eET"]) / math.log(10)
        worse_log = exponents[exponents < 0]
        if len(worse_log):
            shown = worse_log.iloc[::_stride(len(worse_log))]
            axes[1].scatter(shown.index, shown.values, s=8, alpha=0.35, color=GA_MAX_COLOR,
                            linewidths=0)
        axes[1].set(ylabel=r"$\log_{10} e^{\Delta E / T}$")
    _temperature_panel(axes[-1], history["temperature"])
    fig.tight_layout()
    return fig


def plot_eet(history: dict[str, list[float]], title: str, path: str | Path,
             window: int = 200) -> None:
    fig = _eet_figure(history, title, window)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Best-so-far per config: mean over seeds, min-max band. Columns: config, seed, iteration, best.
def plot_sa_comparison(curves: pd.DataFrame, title: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=curves, x="iteration", y="best", hue="config",
                 errorbar=("pi", 100), ax=ax)
    ax.set(title=title, xlabel="Iteration", ylabel="Best objective")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# One number per run (e.g. stuck_count) per configuration; horizontal so long names stay readable.
def plot_metric_box(summary: pd.DataFrame, y: str, ylabel: str, title: str,
                    path: str | Path) -> None:
    configs = list(dict.fromkeys(summary["config"]))
    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.45 * len(configs)))
    sns.boxplot(data=summary, y="config", x=y, order=configs, ax=ax,
                color="lightsteelblue", showfliers=False)
    sns.stripplot(data=summary, y="config", x=y, order=configs, ax=ax, color=GA_MAX_COLOR, size=6)
    ax.set(title=title, xlabel=ylabel, ylabel="")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
