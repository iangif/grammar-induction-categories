"""Nonlinear 2-D projections and projection-quality diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE, trustworthiness

from .io import HellingerSpaceError
from .pca import run_pca
from .transform import HellingerSpace


def _quality_metrics(original: np.ndarray, projected: np.ndarray) -> dict[str, float]:
    """Compute generic 2-D fidelity diagnostics."""

    n = original.shape[0]
    metrics: dict[str, float] = {}
    if n >= 4:
        n_neighbors = min(5, max(1, (n - 1) // 2))
        # sklearn requires n_neighbors < n_samples / 2.
        while n_neighbors >= n / 2 and n_neighbors > 1:
            n_neighbors -= 1
        if n_neighbors >= 1 and n_neighbors < n / 2:
            metrics["trustworthiness"] = float(
                trustworthiness(original, projected, n_neighbors=n_neighbors)
            )

    if n >= 3:
        original_distances = pdist(original, metric="euclidean")
        projected_distances = pdist(projected, metric="euclidean")
        if np.std(original_distances) > 0 and np.std(projected_distances) > 0:
            correlation = spearmanr(original_distances, projected_distances).statistic
            if np.isfinite(correlation):
                metrics["pairwise_distance_spearman"] = float(correlation)
    return metrics


def project_pca(space: HellingerSpace) -> dict[str, Any]:
    result = run_pca(space)
    coordinates = result["coordinates"]
    if "PC2" in coordinates.columns:
        projected = coordinates[["PC1", "PC2"]].to_numpy(dtype=float)
    else:
        projected = np.column_stack(
            [coordinates["PC1"].to_numpy(dtype=float), np.zeros(len(coordinates))]
        )
    quality = _quality_metrics(space.matrix, projected)
    explained = result["explained_variance"]
    quality["variance_explained_2d"] = float(
        explained["explained_variance_ratio"].head(2).sum()
    )
    result["quality"] = quality
    return result


def project_umap(
    space: HellingerSpace,
    *,
    random_state: int = 0,
) -> dict[str, Any]:
    try:
        import umap
    except ImportError as exc:
        raise HellingerSpaceError(
            "UMAP projection requires `umap-learn`. Install it with `uv pip install umap-learn`."
        ) from exc

    n_categories = len(space.categories)
    if n_categories < 3:
        raise HellingerSpaceError("UMAP requires at least three categories.")
    n_neighbors = min(15, n_categories - 1)
    model = umap.UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    embedded = model.fit_transform(space.matrix)
    coordinates = pd.DataFrame(
        {
            "category": list(space.categories),
            "UMAP1": embedded[:, 0],
            "UMAP2": embedded[:, 1],
        }
    )
    return {
        "model": model,
        "coordinates": coordinates,
        "quality": _quality_metrics(space.matrix, embedded),
    }


def project_tsne(
    space: HellingerSpace,
    *,
    random_state: int = 0,
) -> dict[str, Any]:
    n_categories = len(space.categories)
    if n_categories < 4:
        raise HellingerSpaceError("t-SNE requires at least four categories.")

    # Keep perplexity comfortably below the sample count, with a conservative
    # default for the small (~40-60 category) spaces used in this project.
    perplexity = min(30.0, max(2.0, (n_categories - 1) / 3.0))
    if perplexity >= n_categories:
        perplexity = float(n_categories - 1)
    model = TSNE(
        n_components=2,
        metric="euclidean",
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    embedded = model.fit_transform(space.matrix)
    coordinates = pd.DataFrame(
        {
            "category": list(space.categories),
            "TSNE1": embedded[:, 0],
            "TSNE2": embedded[:, 1],
        }
    )
    quality = _quality_metrics(space.matrix, embedded)
    quality["perplexity"] = float(perplexity)
    return {"model": model, "coordinates": coordinates, "quality": quality}
