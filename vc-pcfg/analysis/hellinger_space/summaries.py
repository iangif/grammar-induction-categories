"""Category-level summaries aligned with an active Hellinger feature space."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import POSITION_RANK
from .io import HellingerSpaceError
from .transform import HellingerSpace


def score_space_coherence(
    space: HellingerSpace,
    feature_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Weighted mean coherence over exactly the blocks in ``space``.

    This uses the same normalized block weights as the Hellinger geometry, so
    the color/summary shown for a space refers to the linguistic information
    that actually determines point positions in that space.
    """

    required = {
        "category",
        "domain",
        "position",
        "feature",
        "normalized_coherence",
    }
    missing = sorted(required - set(feature_scores.columns))
    if missing:
        raise HellingerSpaceError(f"Feature scores missing columns: {missing}")

    blocks = space.block_weights[
        ["domain", "position", "feature", "feature_family", "block_weight"]
    ].copy()
    scores = feature_scores.merge(
        blocks,
        on=["domain", "position", "feature", "feature_family"],
        how="inner",
        validate="many_to_one",
    )

    expected = len(space.categories) * space.block_count
    if len(scores) != expected:
        actual_blocks = scores[["category", "domain", "position", "feature"]].drop_duplicates()
        raise HellingerSpaceError(
            "Category feature scores do not cover every active Hellinger block: "
            f"expected {expected} category/block rows, got {len(actual_blocks)}."
        )

    scores["weighted_coherence"] = scores["normalized_coherence"] * scores["block_weight"]
    summary = (
        scores.groupby("category", sort=False, dropna=False)["weighted_coherence"]
        .sum()
        .rename("space_coherence")
        .reset_index()
    )
    summary["space_coherence"] = summary["space_coherence"].clip(0.0, 1.0)
    return summary


def active_feature_scores(space: HellingerSpace, feature_scores: pd.DataFrame) -> pd.DataFrame:
    """Return detailed score rows belonging to one active feature space."""

    blocks = space.block_weights[
        ["domain", "position", "feature", "feature_family", "block_weight"]
    ].copy()
    active = feature_scores.merge(
        blocks,
        on=["domain", "position", "feature", "feature_family"],
        how="inner",
        validate="many_to_one",
    )
    active["_position_rank"] = active["position"].map(POSITION_RANK)
    sort_columns = ["category", "normalized_coherence", "modal_coverage"]
    ascending = [True, False, False]
    if "modal_count" in active.columns:
        sort_columns.append("modal_count")
        ascending.append(False)
    sort_columns.extend(["_position_rank", "feature"])
    ascending.extend([True, True])
    active = active.sort_values(sort_columns, ascending=ascending, kind="stable")
    return active.drop(columns="_position_rank").reset_index(drop=True)


def merge_category_visual_metrics(
    space: HellingerSpace,
    category_metrics: pd.DataFrame,
    feature_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Join LD_eff and same-space coherence for visualization."""

    coherence = score_space_coherence(space, feature_scores)
    metrics = category_metrics[["category", "LD_eff"]].merge(
        coherence,
        on="category",
        how="inner",
        validate="one_to_one",
    )
    expected = set(space.categories)
    actual = set(metrics["category"])
    if expected != actual:
        raise HellingerSpaceError(
            "Metrics/categories do not align with Hellinger categories. "
            f"Missing={sorted(expected - actual, key=str)[:5]}, "
            f"extra={sorted(actual - expected, key=str)[:5]}."
        )
    if not np.isfinite(metrics["space_coherence"].to_numpy(dtype=float)).all():
        raise HellingerSpaceError("Computed space coherence contains non-finite values.")
    return metrics
