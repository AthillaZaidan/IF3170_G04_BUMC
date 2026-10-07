"""Experiment plots (seaborn): objective curves and comparisons between runs."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GA_MAX_COLOR = "tab:blue"
GA_MEAN_COLOR = "tab:orange"

sns.set_theme(style="whitegrid")


def plot_ga_curve(history: dict[str, list[float]], title: str, path: str | Path) -> None:
    """Max and mean population fitness per generation (one run)."""
    generations = range(len(history["max"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(x=generations, y=history["max"], ax=ax, color=GA_MAX_COLOR,
                 label="max", linewidth=2)
    sns.lineplot(x=generations, y=history["mean"], ax=ax, color=GA_MEAN_COLOR,
                 label="mean", linewidth=1.5)
    ax.set(title=title, xlabel="Generation", ylabel="Objective (total value)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_objective_curve(history: dict[str, list[float]], title: str, path: str | Path) -> None:
    if "max" in history and "mean" in history:
        plot_ga_curve(history, title, path)
        return

    values = history["value"]
    iterations = range(len(values))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(x=iterations, y=values, ax=ax, color=GA_MAX_COLOR,
                 label="current", linewidth=2)
    if "best" in history and history["best"] != values:
        sns.lineplot(x=iterations, y=history["best"], ax=ax, color=GA_MEAN_COLOR,
                     label="best", linewidth=1.5)
    ax.set(title=title, xlabel="Iteration", ylabel="Objective (total value)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_ga_comparison(curves: pd.DataFrame, title: str, path: str | Path) -> None:
    """Max-fitness curves per configuration: line = mean over seeds, band =
    min-max over seeds. Columns: config, seed, generation, max."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=curves, x="generation", y="max", hue="config",
                 errorbar=("pi", 100), ax=ax)
    ax.set(title=title, xlabel="Generation", ylabel="Max objective (total value)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_final_values(summary: pd.DataFrame, title: str, path: str | Path) -> None:
    """Final objective of every run, grouped by configuration.
    Columns: config, final_value."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=summary, x="config", y="final_value", ax=ax,
                color="lightsteelblue", showfliers=False)
    sns.stripplot(data=summary, x="config", y="final_value", ax=ax,
                  color=GA_MAX_COLOR, size=7)
    ax.set(title=title, xlabel="Configuration", ylabel="Final objective (total value)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
