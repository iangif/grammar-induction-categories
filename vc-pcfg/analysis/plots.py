import math
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .constants.pos import POS_ORDER


def save_top_bar_plot(
    table: pd.DataFrame,
    value_column: str,
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    if table.empty:
        return
    plot_data = table.sort_values(value_column, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(plot_data))))
    ax.barh(plot_data["w"].astype(str), plot_data[value_column])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Word")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rank_distribution_plot(
    values: pd.Series,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    clean = values.dropna().sort_values(ascending=False).reset_index(drop=True)
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(clean) + 1), clean.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Word rank")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_position_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    if table.empty:
        return
    centers = (table["bin_left"] + table["bin_right"]) / 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        centers,
        table["proportion"],
        width=(table["bin_right"] - table["bin_left"]) * 0.9,
    )
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Normalized sentence position")
    ax.set_ylabel("Token proportion")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_square_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(values, aspect="equal", interpolation="nearest")
    labels = [str(value) for value in matrix.index]
    step = max(1, math.ceil(len(labels) / 30))
    ticks = np.arange(0, len(labels), step)
    ax.set_xticks(ticks, [labels[index] for index in ticks], rotation=90)
    ax.set_yticks(ticks, [labels[index] for index in ticks])
    ax.set_xlabel("Category")
    ax.set_ylabel("Category")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def add_pos_section_guides(ax: plt.Axes, pos_values: Sequence[str]) -> None:
    """Draw boundaries and labels for contiguous POS sections."""
    if not pos_values:
        return
    starts = [0]
    for index in range(1, len(pos_values)):
        if pos_values[index] != pos_values[index - 1]:
            starts.append(index)
            ax.axvline(index - 0.5, linewidth=1.0, alpha=0.8)
    ends = starts[1:] + [len(pos_values)]
    for start, end in zip(starts, ends, strict=True):
        midpoint = (start + end - 1) / 2
        ax.text(
            midpoint,
            1.01,
            pos_values[start],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )


def save_category_lexical_heatmap(
    matrix: pd.DataFrame,
    lexical_summary: pd.DataFrame,
    label_column: str,
    pos_column: str,
    count_column: str,
    max_columns: int,
    path: Path,
    title: str,
    xlabel: str,
    colorbar_label: str,
) -> None:
    if max_columns <= 0 or matrix.empty or lexical_summary.empty:
        return

    pos_rank = {pos: index for index, pos in enumerate(POS_ORDER)}
    selected = lexical_summary.sort_values(
        [count_column, label_column],
        ascending=[False, True],
        kind="stable",
    ).head(max_columns)
    ordering = selected[[label_column, pos_column, count_column]].copy()
    ordering["_pos_rank"] = ordering[pos_column].map(pos_rank).fillna(len(POS_ORDER))
    ordering = ordering.sort_values(
        ["_pos_rank", pos_column, count_column, label_column],
        ascending=[True, True, False, True],
        kind="stable",
    )
    labels = ordering[label_column].astype(str).tolist()
    pos_values = ordering[pos_column].fillna("X").astype(str).tolist()
    subset = matrix.reindex(columns=labels)

    fig_width = max(12, min(36, len(labels) * 0.20))
    fig, ax = plt.subplots(figsize=(fig_width, 11))
    image = ax.imshow(subset.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(subset.index)), [str(value) for value in subset.index])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Category")
    ax.set_title(title, pad=30)
    add_pos_section_guides(ax, pos_values)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_category_pos_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(matrix.index)), [str(value) for value in matrix.index])
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_xlabel("spaCy POS")
    ax.set_ylabel("Category")
    ax.set_title("P(spaCy POS | category)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Proportion")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
