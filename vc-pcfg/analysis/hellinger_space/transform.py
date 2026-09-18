"""Construct weighted Hellinger geometries from long feature distributions."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import numpy as np
import pandas as pd

from .constants import POSITION_RANK
from .io import HellingerSpaceError
from .spaces import build_block_weights, filter_feature_space


@dataclass(frozen=True)
class HellingerSpace:
    """Dense category coordinates and metadata for one Hellinger feature space."""

    categories: tuple[object, ...]
    matrix: np.ndarray
    vectors: pd.DataFrame
    dimensions: pd.DataFrame
    block_weights: pd.DataFrame
    block_count: int
    feature_space: str
    position_group_weights: dict[str, float]


def _sorted_categories(series: pd.Series) -> list[object]:
    values = list(pd.unique(series))
    if not values:
        return []
    if all(isinstance(value, (int, np.integer)) for value in values):
        return sorted(values, key=int)
    return sorted(values, key=lambda value: str(value))


def _feature_order(frame: pd.DataFrame) -> list[str]:
    """Preserve the abstraction framework's declared feature order."""

    return list(dict.fromkeys(frame["feature"].tolist()))


def _ordered_blocks(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    feature_order = _feature_order(frame)
    feature_rank = {feature: index for index, feature in enumerate(feature_order)}

    unique = frame[["domain", "position", "feature", "feature_family"]].drop_duplicates()
    rows = [tuple(row) for row in unique.itertuples(index=False, name=None)]
    return sorted(
        rows,
        key=lambda block: (
            POSITION_RANK[block[1]],
            feature_rank[block[2]],
            block[0],
        ),
    )


def build_hellinger_space(
    frame: pd.DataFrame,
    *,
    feature_space: str = "all",
    position_group_weights: Mapping[str, float] | None = None,
) -> HellingerSpace:
    """Build a same-scale weighted Hellinger representation.

    For each active block ``b=(position, feature)`` with normalized block weight
    ``w_b``, the coordinate for value ``v`` is

        x[c,b,v] = sqrt(w_b / 2) * sqrt(p[c,b,v]).

    Therefore

        ||x_c - x_d||^2 = sum_b w_b * H(p_c,b, p_d,b)^2.

    Block weights always sum to one *within the selected feature space*, so
    lexical, contextual, grammatical, semantic, and full spaces share the same
    weighted-Hellinger distance scale. Positional importance defaults to
    TARGET=0.50, L1/R1=0.35, L2/R2=0.15 before subspace renormalization.
    """

    required = {
        "category",
        "domain",
        "position",
        "feature",
        "feature_family",
        "value",
        "proportion",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HellingerSpaceError(f"Missing columns required for transformation: {missing}")
    if frame.empty:
        raise HellingerSpaceError("Cannot build a Hellinger space from an empty table.")

    selected = filter_feature_space(frame, feature_space)
    categories = _sorted_categories(selected["category"])
    blocks = _ordered_blocks(selected)
    if not categories:
        raise HellingerSpaceError("No categories found.")
    if not blocks:
        raise HellingerSpaceError("No feature-distribution blocks found.")

    raw_blocks = pd.DataFrame(
        blocks,
        columns=["domain", "position", "feature", "feature_family"],
    )
    weighted_blocks = build_block_weights(
        raw_blocks,
        position_group_weights=position_group_weights,
    )
    weight_lookup = {
        (row.domain, row.position, row.feature): float(row.block_weight)
        for row in weighted_blocks.itertuples(index=False)
    }

    matrices: list[np.ndarray] = []
    dimension_rows: list[dict[str, object]] = []
    dimension_index = 0

    for block_index, (domain, position, feature, feature_family) in enumerate(blocks):
        block = selected.loc[
            (selected["domain"] == domain)
            & (selected["position"] == position)
            & (selected["feature"] == feature),
            ["category", "value", "proportion"],
        ]
        values = sorted(pd.unique(block["value"].astype(str)), key=str)
        if not values:
            raise HellingerSpaceError(
                f"Feature block {(domain, position, feature)!r} contains no values."
            )

        dense = block.pivot(index="category", columns="value", values="proportion")
        dense = dense.reindex(index=categories, columns=values, fill_value=0.0).fillna(0.0)
        probabilities = dense.to_numpy(dtype=float, copy=True)

        sums = probabilities.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-8, rtol=1e-8):
            bad = [categories[index] for index in np.flatnonzero(~np.isclose(sums, 1.0))[:5]]
            raise HellingerSpaceError(
                f"Dense reconstruction for block {(position, feature)!r} does not sum to 1 "
                f"for categories {bad}."
            )

        block_weight = weight_lookup[(domain, position, feature)]
        matrices.append(np.sqrt(probabilities) * sqrt(block_weight / 2.0))
        metadata = weighted_blocks.loc[
            (weighted_blocks["domain"] == domain)
            & (weighted_blocks["position"] == position)
            & (weighted_blocks["feature"] == feature)
        ].iloc[0]

        for value_index, value in enumerate(values):
            dimension_rows.append(
                {
                    "dimension": f"d{dimension_index:06d}",
                    "block_index": block_index,
                    "value_index": value_index,
                    "feature_space": feature_space,
                    "domain": domain,
                    "position": position,
                    "position_group": metadata["position_group"],
                    "feature": feature,
                    "feature_family": feature_family,
                    "value": value,
                    "original_group_weight": float(metadata["original_group_weight"]),
                    "normalized_group_weight": float(metadata["normalized_group_weight"]),
                    "blocks_in_group": int(metadata["blocks_in_group"]),
                    "block_weight": block_weight,
                }
            )
            dimension_index += 1

    matrix = np.concatenate(matrices, axis=1)
    dimensions = pd.DataFrame(dimension_rows)
    vectors = pd.DataFrame(matrix, columns=dimensions["dimension"].tolist())
    vectors.insert(0, "category", categories)

    group_weights = (
        weighted_blocks[["position_group", "normalized_group_weight"]]
        .drop_duplicates()
        .set_index("position_group")["normalized_group_weight"]
        .astype(float)
        .to_dict()
    )

    return HellingerSpace(
        categories=tuple(categories),
        matrix=matrix,
        vectors=vectors,
        dimensions=dimensions,
        block_weights=weighted_blocks,
        block_count=len(blocks),
        feature_space=feature_space,
        position_group_weights=group_weights,
    )
