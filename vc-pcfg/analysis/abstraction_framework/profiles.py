"""Long-format category feature distributions."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

from .constants import CONTEXT_POSITIONS, LEXICAL_FEATURES, NULL_VALUE


def _distribution_rows(
    frame: pd.DataFrame,
    *,
    value_column: str,
    domain: str,
    position: str,
    feature_name: str,
    category_sizes: pd.Series,
) -> pd.DataFrame:
    values = frame[value_column].astype("string").fillna(NULL_VALUE)
    counts = (
        pd.DataFrame({"category": frame["category"].to_numpy(), "value": values.to_numpy()})
        .groupby(["category", "value"], sort=False, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["category_token_count"] = counts["category"].map(category_sizes).astype("int64")
    counts["proportion"] = counts["count"] / counts["category_token_count"]
    counts.insert(1, "domain", domain)
    counts.insert(2, "position", position)
    counts.insert(3, "feature", feature_name)
    return counts[
        [
            "category",
            "domain",
            "position",
            "feature",
            "value",
            "count",
            "category_token_count",
            "proportion",
        ]
    ]


def build_category_feature_distributions(
    annotated_tokens: pd.DataFrame,
    contextual_vectors: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Build the lossless observed feature distributions for every category.

    Each row represents one observed value of one feature block. ``proportion``
    is always ``count / category_token_count``; therefore genuine missing lexical
    annotations are emitted as the explicit value ``NULL`` and remain part of
    the probability distribution. Context boundary values remain ``<BOS>`` and
    ``<EOS>``. Unobserved values are omitted and can be filled with zero when a
    downstream analysis pivots this table to a matrix.
    """

    if "category" not in annotated_tokens.columns:
        raise ValueError("annotated_tokens must contain category")
    if "category" not in contextual_vectors.columns:
        raise ValueError("contextual_vectors must contain category")

    lexical_missing = [
        feature.column for feature in LEXICAL_FEATURES if feature.column not in annotated_tokens.columns
    ]
    if lexical_missing:
        raise ValueError(f"Missing lexical feature columns: {lexical_missing}")

    contextual_columns = [
        f"{position}.{feature.name}"
        for position, _ in CONTEXT_POSITIONS
        for feature in LEXICAL_FEATURES
    ]
    contextual_missing = [
        column for column in contextual_columns if column not in contextual_vectors.columns
    ]
    if contextual_missing:
        raise ValueError(f"Missing contextual feature columns: {contextual_missing}")

    lexical_sizes = annotated_tokens.groupby("category", sort=False, dropna=False).size()
    contextual_sizes = contextual_vectors.groupby("category", sort=False, dropna=False).size()
    if not lexical_sizes.equals(contextual_sizes):
        raise ValueError("Lexical and contextual tables do not contain the same category token counts.")

    blocks: list[tuple[pd.DataFrame, str, str, str, str]] = []
    for feature in LEXICAL_FEATURES:
        blocks.append((annotated_tokens, feature.column, "lexical", "TARGET", feature.name))
    for position, _ in CONTEXT_POSITIONS:
        for feature in LEXICAL_FEATURES:
            blocks.append(
                (
                    contextual_vectors,
                    f"{position}.{feature.name}",
                    "contextual",
                    position,
                    feature.name,
                )
            )

    rows: list[pd.DataFrame] = []
    progress: Iterable[tuple[pd.DataFrame, str, str, str, str]] = tqdm(
        blocks,
        total=len(blocks),
        desc="Building feature distributions",
        unit="block",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for frame, value_column, domain, position, feature_name in progress:
        rows.append(
            _distribution_rows(
                frame,
                value_column=value_column,
                domain=domain,
                position=position,
                feature_name=feature_name,
                category_sizes=lexical_sizes,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "category",
                "domain",
                "position",
                "feature",
                "value",
                "count",
                "category_token_count",
                "proportion",
            ]
        )
    return pd.concat(rows, ignore_index=True)
