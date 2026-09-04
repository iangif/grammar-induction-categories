"""Generic baseline-normalized model-coverage coherence machinery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass(frozen=True)
class FeatureRef:
    """Display name and dataframe column for one coherence feature."""

    name: str
    column: str

def _stable_modal_value(series: pd.Series) -> tuple[Any | None, int, int]:
    """Return (modal non-NULL value, modal count, non-NULL count).

    Ties are resolved deterministically by the string representation of values.
    """

    valid = series.loc[series.notna()]
    non_null_count = len(valid)
    if non_null_count == 0:
        return None, 0, 0
    counts = valid.value_counts(dropna=True)
    max_count = int(counts.max())
    candidates = counts.loc[counts.eq(max_count)].index.tolist()
    modal = min(candidates, key=lambda value: str(value))
    return modal, max_count, non_null_count


def score_modal_coherence(
    frame: pd.DataFrame,
    *,
    category_column: str,
    features: Sequence[FeatureRef],
    show_progress: bool = False,
    progress_desc: str = "Scoring coherence",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score categories using baseline-normalized modal coverage.

    For each category-feature pair:
      1. choose the most common non-NULL value within the category;
      2. divide its count by *all* category tokens, including NULLs;
      3. use the corpus-wide coverage of that same value as the baseline;
      4. normalize as max(0, (M-Q)/(1-Q)).

    The same function is used for lexical and contextual coherence.
    """

    if category_column not in frame.columns:
        raise ValueError(f"Missing category column: {category_column}")
    missing = [feature.column for feature in features if feature.column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    corpus_n = len(frame)
    detail_rows: list[dict[str, Any]] = []
    feature_order = {feature.name: index for index, feature in enumerate(features)}
    category_groups = frame.groupby(category_column, sort=False, dropna=False)

    # Compute corpus baselines once per feature.
    progress = tqdm(
        total=len(features) * (category_groups.ngroups + 1),
        desc=progress_desc,
        unit="feature",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    corpus_counts: dict[str, pd.Series] = {}
    try:
        for feature in features:
            corpus_counts[feature.column] = frame[feature.column].value_counts(dropna=True)
            progress.update(1)

        for category, group in category_groups:
            category_n = len(group)
            for feature in features:
                modal, modal_count, non_null_count = _stable_modal_value(group[feature.column])
                if modal is None:
                    corpus_coverage = np.nan
                    modal_coverage = 0.0
                    normalized = 0.0
                    corpus_count = 0
                else:
                    modal_coverage = modal_count / category_n
                    corpus_count = int(corpus_counts[feature.column].get(modal, 0))
                    corpus_coverage = corpus_count / corpus_n
                    if corpus_coverage >= 1.0:
                        normalized = 0.0
                    else:
                        normalized = max(
                            0.0,
                            (modal_coverage - corpus_coverage) / (1.0 - corpus_coverage),
                        )
                        normalized = min(1.0, float(normalized))

                detail_rows.append(
                    {
                        "category": category,
                        "feature": feature.name,
                        "feature_column": feature.column,
                        "category_token_count": category_n,
                        "non_null_count": non_null_count,
                        "modal_value": modal,
                        "modal_count": modal_count,
                        "modal_coverage": modal_coverage,
                        "corpus_count": corpus_count,
                        "corpus_token_count": corpus_n,
                        "corpus_coverage": corpus_coverage,
                        "normalized_score": normalized,
                    }
                )
                progress.update(1)
    finally:
        progress.close()

    details = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for category, group in details.groupby("category", sort=False, dropna=False):
        ranked = group.copy()
        ranked["_feature_order"] = ranked["feature"].map(feature_order)
        ranked = ranked.sort_values(
            ["normalized_score", "modal_coverage", "modal_count", "_feature_order"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        winner = ranked.iloc[0]
        summary_rows.append(
            {
                "category": category,
                "token_count": int(winner["category_token_count"]),
                "coherence": float(winner["normalized_score"]),
                "winning_feature": winner["feature"],
                "winning_value": winner["modal_value"],
                "winning_modal_count": int(winner["modal_count"]),
                "winning_modal_coverage": float(winner["modal_coverage"]),
                "winning_corpus_coverage": (
                    float(winner["corpus_coverage"])
                    if pd.notna(winner["corpus_coverage"])
                    else np.nan
                ),
            }
        )
    
    return pd.DataFrame(summary_rows), details
