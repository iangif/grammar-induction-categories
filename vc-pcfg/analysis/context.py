import pandas as pd
import numpy as np
from typing import Sequence


def build_ranked_context_table(
    df: pd.DataFrame,
    group_columns: list[str],
    top_n: int,
) -> pd.DataFrame:
    category_totals = df.groupby("viterbi_preterminal").size()
    table = (
        df.groupby(["viterbi_preterminal", *group_columns], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"viterbi_preterminal": "c"})
    )
    table["proportion"] = table["count"] / table["c"].map(category_totals)
    table = table.sort_values(
        ["c", "count", *group_columns],
        ascending=[True, False, *([True] * len(group_columns))],
        kind="stable",
    )
    table = table.groupby("c", sort=False).head(top_n).copy()
    table["rank"] = table.groupby("c").cumcount() + 1
    return table[["c", "rank", *group_columns, "count", "proportion"]]


def build_position_distribution(
    df: pd.DataFrame,
    category_ids: Sequence[int],
    n_bins: int,
) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError("--position-bins must be greater than zero.")

    positions = df["normalized_sentence_position"].clip(0.0, 1.0).to_numpy()
    bin_index = np.minimum((positions * n_bins).astype(int), n_bins - 1)
    working = pd.DataFrame(
        {
            "c": df["viterbi_preterminal"].to_numpy(),
            "position_bin": bin_index,
        }
    )
    counts = working.groupby(["c", "position_bin"]).size()
    full_index = pd.MultiIndex.from_product(
        [category_ids, range(n_bins)], names=["c", "position_bin"]
    )
    counts = counts.reindex(full_index, fill_value=0).rename("count").reset_index()
    category_totals = counts.groupby("c")["count"].transform("sum")
    counts["proportion"] = np.divide(
        counts["count"],
        category_totals,
        out=np.zeros(len(counts), dtype=float),
        where=category_totals != 0,
    )
    counts["bin_left"] = counts["position_bin"] / n_bins
    counts["bin_right"] = (counts["position_bin"] + 1) / n_bins
    return counts[["c", "position_bin", "bin_left", "bin_right", "count", "proportion"]]
