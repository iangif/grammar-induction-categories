"""Principal component analysis of the common Hellinger representation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .io import HellingerSpaceError
from .transform import HellingerSpace


def run_pca(space: HellingerSpace) -> dict[str, object]:
    """Fit PCA directly to Hellinger coordinates, with centering only.

    No per-dimension variance standardization is applied: sklearn PCA centers
    the Hellinger coordinates internally, preserving the block-aware geometry.
    """

    n_categories, n_dimensions = space.matrix.shape
    if n_categories < 2:
        raise HellingerSpaceError("PCA requires at least two categories.")
    if n_dimensions < 1:
        raise HellingerSpaceError("PCA requires at least one Hellinger dimension.")

    centered = space.matrix - space.matrix.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        raise HellingerSpaceError(
            "PCA is undefined because all category Hellinger vectors are identical."
        )

    n_components = min(n_categories - 1, n_dimensions)
    model = PCA(n_components=n_components, svd_solver="full")
    transformed = model.fit_transform(space.matrix)
    component_names = [f"PC{index}" for index in range(1, n_components + 1)]

    coordinates = pd.DataFrame(transformed, columns=component_names)
    coordinates.insert(0, "category", list(space.categories))

    loadings = space.dimensions.copy()
    for index, name in enumerate(component_names):
        loadings[name] = model.components_[index, :]

    explained = pd.DataFrame(
        {
            "component": component_names,
            "eigenvalue": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_),
        }
    )

    return {
        "model": model,
        "coordinates": coordinates,
        "loadings": loadings,
        "explained_variance": explained,
    }


def save_pca_scatter(coordinates: pd.DataFrame, path: Path) -> None:
    """Write a category-labelled PC1/PC2 plot (or a 1-D fallback)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    has_pc2 = "PC2" in coordinates.columns

    fig, axis = plt.subplots(figsize=(9.0, 7.0))
    x = coordinates["PC1"].to_numpy(dtype=float)
    y = coordinates["PC2"].to_numpy(dtype=float) if has_pc2 else np.zeros(len(coordinates))
    axis.scatter(x, y)

    for category, x_value, y_value in zip(coordinates["category"], x, y):
        axis.annotate(str(category), (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=8)

    axis.axhline(0.0, linewidth=0.7)
    axis.axvline(0.0, linewidth=0.7)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2" if has_pc2 else "")
    axis.set_title("PCA of induced categories in Hellinger space")
    if not has_pc2:
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
