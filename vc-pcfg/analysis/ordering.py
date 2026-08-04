from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .overlap import off_diagonal_mean


def average_metric_order(
    values: np.ndarray,
    category_ids: Sequence[int],
    higher_is_more_redundant: bool,
) -> tuple[list[int], pd.DataFrame]:
    means = off_diagonal_mean(values)
    order_table = pd.DataFrame(
        {
            "c": list(category_ids),
            "mean_off_diagonal_metric": means,
        }
    )
    order_table["has_finite_metric"] = np.isfinite(
        order_table["mean_off_diagonal_metric"]
    )
    order_table = order_table.sort_values(
        ["has_finite_metric", "mean_off_diagonal_metric", "c"],
        ascending=[False, not higher_is_more_redundant, True],
        kind="stable",
    ).reset_index(drop=True)
    order_table["order_rank"] = np.arange(1, len(order_table) + 1)
    return order_table["c"].astype(int).tolist(), order_table


def hierarchical_metric_order(
    values: np.ndarray,
    category_ids: Sequence[int],
    metric_kind: str,
    linkage_method: str,
) -> tuple[list[int], pd.DataFrame]:
    """Order non-empty categories by hierarchical clustering.

    Categories whose rows contain no finite off-diagonal values are appended in
    numeric order. J-S divergence is converted to Jensen-Shannon distance with
    ``sqrt(JSD)`` before clustering.
    """
    category_array = np.asarray(category_ids, dtype=int)
    valid_indices: list[int] = []
    invalid_indices: list[int] = []
    for index in range(values.shape[0]):
        row = values[index].copy()
        row[index] = np.nan
        if np.isfinite(row).any():
            valid_indices.append(index)
        else:
            invalid_indices.append(index)

    if len(valid_indices) <= 1:
        ordered_indices = valid_indices + invalid_indices
    else:
        submatrix = values[np.ix_(valid_indices, valid_indices)].astype(float)
        if metric_kind in {"cosine", "top_k"}:
            distances = 1.0 - submatrix
        elif metric_kind == "js":
            distances = np.sqrt(np.clip(submatrix, 0.0, None))
        else:
            raise ValueError(f"Unknown metric kind: {metric_kind}")

        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)
        distances = np.clip(0.5 * (distances + distances.T), 0.0, None)
        np.fill_diagonal(distances, 0.0)
        condensed = squareform(distances, checks=False)
        tree = linkage(condensed, method=linkage_method, optimal_ordering=True)
        leaf_positions = leaves_list(tree).tolist()
        clustered = [valid_indices[position] for position in leaf_positions]
        ordered_indices = clustered + invalid_indices

    ordered_categories = category_array[ordered_indices].astype(int).tolist()
    order_table = pd.DataFrame(
        {
            "c": ordered_categories,
            "order_rank": np.arange(1, len(ordered_categories) + 1),
            "clustered": [index in valid_indices for index in ordered_indices],
        }
    )
    return ordered_categories, order_table


def reorder_square_matrix(matrix: pd.DataFrame, order: Sequence[int]) -> pd.DataFrame:
    return matrix.reindex(index=order, columns=order)
