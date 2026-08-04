from typing import Sequence

import numpy as np
import pandas as pd


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    result = np.full((matrix.shape[0], matrix.shape[0]), np.nan, dtype=float)
    valid = norms > 0
    if valid.any():
        normalized = matrix[valid] / norms[valid, None]
        similarities = normalized @ normalized.T
        valid_indices = np.flatnonzero(valid)
        result[np.ix_(valid_indices, valid_indices)] = similarities
    return result


def js_divergence_pair(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    p_mask = p > 0
    q_mask = q > 0
    kl_p = np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask]))
    kl_q = np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask]))
    return float(0.5 * (kl_p + kl_q))


def js_divergence_matrix(matrix: np.ndarray) -> np.ndarray:
    n_categories = matrix.shape[0]
    result = np.full((n_categories, n_categories), np.nan, dtype=float)
    for i in range(n_categories):
        for j in range(i, n_categories):
            value = js_divergence_pair(matrix[i], matrix[j])
            result[i, j] = value
            result[j, i] = value
    return result


def top_k_overlap_matrix(matrix: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("--top-k-overlap must be greater than zero.")

    top_sets: list[set[int] | None] = []
    for row in matrix:
        positive_indices = np.flatnonzero(row > 0)
        if len(positive_indices) == 0:
            top_sets.append(None)
            continue
        ordered = positive_indices[np.argsort(row[positive_indices])[::-1]]
        top_sets.append(set(ordered[:k].tolist()))

    n_categories = matrix.shape[0]
    result = np.full((n_categories, n_categories), np.nan, dtype=float)
    for i in range(n_categories):
        for j in range(i, n_categories):
            if top_sets[i] is None or top_sets[j] is None:
                value = float("nan")
            else:
                value = len(top_sets[i] & top_sets[j]) / k
            result[i, j] = value
            result[j, i] = value
    return result


def labeled_matrix(values: np.ndarray, category_ids: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame(values, index=category_ids, columns=category_ids).rename_axis(
        index="c", columns="other_c"
    )


def off_diagonal_mean(values: np.ndarray) -> np.ndarray:
    """Mean finite off-diagonal value for every matrix row."""
    result = np.full(values.shape[0], np.nan, dtype=float)
    for index in range(values.shape[0]):
        row = values[index].copy()
        row[index] = np.nan
        finite = row[np.isfinite(row)]
        if finite.size:
            result[index] = float(finite.mean())
    return result


def nearest_other(
    values: np.ndarray,
    category_ids: Sequence[int],
    higher_is_closer: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nearest_ids = np.full(values.shape[0], np.nan, dtype=float)
    nearest_values = np.full(values.shape[0], np.nan, dtype=float)
    category_array = np.asarray(category_ids)
    for index in range(values.shape[0]):
        row = values[index].copy()
        row[index] = np.nan
        finite_indices = np.flatnonzero(np.isfinite(row))
        if finite_indices.size == 0:
            continue
        local = row[finite_indices]
        chosen = finite_indices[
            np.argmax(local) if higher_is_closer else np.argmin(local)
        ]
        nearest_ids[index] = float(category_array[chosen])
        nearest_values[index] = float(row[chosen])
    return nearest_ids, nearest_values


def build_redundancy_summary(
    category_ids: Sequence[int],
    token_counts: pd.Series,
    cosine: np.ndarray,
    js: np.ndarray,
    top_k: np.ndarray,
) -> pd.DataFrame:
    cosine_nearest, cosine_nearest_value = nearest_other(
        cosine, category_ids, higher_is_closer=True
    )
    js_nearest, js_nearest_value = nearest_other(
        js, category_ids, higher_is_closer=False
    )
    top_k_nearest, top_k_nearest_value = nearest_other(
        top_k, category_ids, higher_is_closer=True
    )
    summary = pd.DataFrame(
        {
            "c": list(category_ids),
            "token_count": token_counts.reindex(category_ids, fill_value=0).to_numpy(),
            "mean_cosine_similarity": off_diagonal_mean(cosine),
            "max_cosine_similarity": cosine_nearest_value,
            "nearest_cosine_category": cosine_nearest,
            "mean_js_divergence": off_diagonal_mean(js),
            "min_js_divergence": js_nearest_value,
            "nearest_js_category": js_nearest,
            "mean_top_k_overlap": off_diagonal_mean(top_k),
            "max_top_k_overlap": top_k_nearest_value,
            "nearest_top_k_category": top_k_nearest,
        }
    )
    for column in [
        "nearest_cosine_category",
        "nearest_js_category",
        "nearest_top_k_category",
    ]:
        summary[column] = summary[column].astype("Int64")
    return summary
