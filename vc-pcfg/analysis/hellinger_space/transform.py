"""Construct the common Hellinger geometry from long feature distributions."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from .constants import POSITION_ORDER, POSITION_RANK
from .io import HellingerSpaceError


@dataclass(frozen=True)
class HellingerSpace:
    """Dense category coordinates and metadata for one Hellinger space."""

    categories: tuple[object, ...]
    matrix: np.ndarray
    vectors: pd.DataFrame
    dimensions: pd.DataFrame
    block_count: int


def _sorted_categories(series: pd.Series) -> list[object]:
    values = list(pd.unique(series))
    if not values:
        return []
    if all(isinstance(value, (int, np.integer)) for value in values):
        return sorted(values, key=int)
    return sorted(values, key=lambda value: str(value))


def _feature_order(frame: pd.DataFrame) -> list[str]:
    """Use the source abstraction framework's declared feature order.

    The long table is emitted block-by-block in feature declaration order. We
    preserve first appearance here rather than alphabetizing linguistic
    features. This keeps dimensions stable across runs produced by the same
    abstraction schema while remaining generic to future feature inventories.
    """

    return list(dict.fromkeys(frame["feature"].tolist()))


def _ordered_blocks(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    feature_order = _feature_order(frame)
    feature_rank = {feature: index for index, feature in enumerate(feature_order)}

    unique = frame[["domain", "position", "feature"]].drop_duplicates()
    rows = [tuple(row) for row in unique.itertuples(index=False, name=None)]
    return sorted(
        rows,
        key=lambda block: (
            POSITION_RANK[block[1]],
            feature_rank[block[2]],
            block[0],
        ),
    )


def build_hellinger_space(frame: pd.DataFrame) -> HellingerSpace:
    """Build normalized square-root probability coordinates.

    For B=(position, feature) distribution blocks, each coordinate is

        x[c,b,v] = sqrt(p[c,b,v]) / sqrt(2B).

    Therefore Euclidean distance between category vectors equals the root mean
    square Hellinger distance across blocks:

        ||x_c - x_d|| = sqrt(mean_b H(p_c,b, p_d,b)^2).

    The long source may omit zero-count category/value combinations; this
    function reconstructs them as zeros using the observed value support of
    each block across the full input table.
    """

    required = {"category", "domain", "position", "feature", "value", "proportion"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HellingerSpaceError(f"Missing columns required for transformation: {missing}")
    if frame.empty:
        raise HellingerSpaceError("Cannot build a Hellinger space from an empty table.")

    categories = _sorted_categories(frame["category"])
    blocks = _ordered_blocks(frame)
    if not categories:
        raise HellingerSpaceError("No categories found.")
    if not blocks:
        raise HellingerSpaceError("No feature-distribution blocks found.")

    normalizer = sqrt(2.0 * len(blocks))
    matrices: list[np.ndarray] = []
    dimension_rows: list[dict[str, object]] = []
    dimension_index = 0

    for block_index, (domain, position, feature) in enumerate(blocks):
        block = frame.loc[
            (frame["domain"] == domain)
            & (frame["position"] == position)
            & (frame["feature"] == feature),
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

        matrices.append(np.sqrt(probabilities) / normalizer)

        for value_index, value in enumerate(values):
            dimension_rows.append(
                {
                    "dimension": f"d{dimension_index:06d}",
                    "block_index": block_index,
                    "value_index": value_index,
                    "domain": domain,
                    "position": position,
                    "feature": feature,
                    "value": value,
                }
            )
            dimension_index += 1

    matrix = np.concatenate(matrices, axis=1)
    dimensions = pd.DataFrame(dimension_rows)
    vectors = pd.DataFrame(matrix, columns=dimensions["dimension"].tolist())
    vectors.insert(0, "category", categories)

    return HellingerSpace(
        categories=tuple(categories),
        matrix=matrix,
        vectors=vectors,
        dimensions=dimensions,
        block_count=len(blocks),
    )
