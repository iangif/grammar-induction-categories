"""Category-level summaries aligned with an active Hellinger feature space."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .io import HellingerSpaceError
from .transform import HellingerSpace


def score_space_coherence(
    space: HellingerSpace,
    feature_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Coherence over exactly the feature blocks in ``space``.

    Per-feature ``normalized_coverage`` is the corpus-normalized modal coverage
    from the abstraction framework. Category-level coherence is the maximum of
    that quantity over all active feature blocks, matching the original lexical
    and contextual coherence definitions.
    """

    required = {
        "category",
        "domain",
        "position",
        "feature",
        "normalized_coverage",
    }
    missing = sorted(required - set(feature_scores.columns))
    if missing:
        raise HellingerSpaceError(f"Feature scores missing columns: {missing}")

    blocks = space.block_weights[["domain", "position", "feature", "feature_family"]].copy()
    blocks["_block_rank"] = np.arange(len(blocks), dtype=int)
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

    sort_columns = [
        "category",
        "normalized_coverage",
        "modal_coverage",
    ]
    ascending = [True, False, False]
    if "modal_count" in scores.columns:
        sort_columns.append("modal_count")
        ascending.append(False)
    sort_columns.append("_block_rank")
    ascending.append(True)
    ranked = scores.sort_values(sort_columns, ascending=ascending, kind="stable")
    winners = ranked.groupby("category", sort=False, dropna=False).head(1).copy()
    winners = winners.rename(
        columns={
            "normalized_coverage": "space_coherence",
            "position": "coherence_position",
            "feature": "coherence_feature",
            "modal_value": "coherence_value",
            "modal_coverage": "coherence_modal_coverage",
            "corpus_coverage": "coherence_corpus_coverage",
        }
    )
    columns = [
        "category",
        "space_coherence",
        "coherence_position",
        "coherence_feature",
        "coherence_value",
        "coherence_modal_coverage",
        "coherence_corpus_coverage",
    ]
    return winners[columns].reset_index(drop=True)


def active_feature_scores(space: HellingerSpace, feature_scores: pd.DataFrame) -> pd.DataFrame:
    """Return detailed score rows belonging to one active feature space."""

    blocks = space.block_weights[
        ["domain", "position", "feature", "feature_family", "block_weight"]
    ].copy()
    blocks["_block_rank"] = np.arange(len(blocks), dtype=int)
    active = feature_scores.merge(
        blocks,
        on=["domain", "position", "feature", "feature_family"],
        how="inner",
        validate="many_to_one",
    )
    sort_columns = ["category", "normalized_coverage", "modal_coverage"]
    ascending = [True, False, False]
    if "modal_count" in active.columns:
        sort_columns.append("modal_count")
        ascending.append(False)
    sort_columns.append("_block_rank")
    ascending.append(True)
    active = active.sort_values(sort_columns, ascending=ascending, kind="stable")
    return active.drop(columns="_block_rank").reset_index(drop=True)


def merge_category_visual_metrics(
    space: HellingerSpace,
    category_metrics: pd.DataFrame,
    feature_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Join diversity measures and same-space coherence for visualization."""

    coherence = score_space_coherence(space, feature_scores)
    metrics = category_metrics[["category", "LD_eff", "CD1_eff"]].merge(
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
