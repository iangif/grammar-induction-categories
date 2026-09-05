"""Hierarchical clustering in the full Hellinger space."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist

from .io import HellingerSpaceError
from .transform import HellingerSpace


def run_hierarchical_clustering(
    space: HellingerSpace,
    *,
    n_clusters: int | None = None,
) -> dict[str, pd.DataFrame | np.ndarray]:
    """Average-linkage clustering using Euclidean/Hellinger-space distance."""

    n_categories = len(space.categories)
    if n_categories < 2:
        raise HellingerSpaceError("Hierarchical clustering requires at least two categories.")
    if n_clusters is not None and not 2 <= n_clusters <= n_categories:
        raise HellingerSpaceError(
            f"n_clusters must be between 2 and {n_categories}; got {n_clusters}."
        )

    distances = pdist(space.matrix, metric="euclidean")
    hierarchy = linkage(distances, method="average")

    linkage_frame = pd.DataFrame(
        {
            "left": hierarchy[:, 0].astype(np.int64),
            "right": hierarchy[:, 1].astype(np.int64),
            "distance": hierarchy[:, 2],
            "size": hierarchy[:, 3].astype(np.int64),
        }
    )

    leaves = leaves_list(hierarchy)
    order_frame = pd.DataFrame(
        {
            "order": np.arange(n_categories, dtype=np.int64),
            "category": [space.categories[index] for index in leaves],
        }
    )

    result: dict[str, pd.DataFrame | np.ndarray] = {
        "hierarchy": hierarchy,
        "linkage": linkage_frame,
        "order": order_frame,
    }

    if n_clusters is not None:
        labels = fcluster(hierarchy, t=n_clusters, criterion="maxclust").astype(np.int64)
        result["assignments"] = pd.DataFrame(
            {"category": list(space.categories), "cluster": labels}
        ).sort_values("category", kind="stable", ignore_index=True)

    return result


def save_dendrogram(
    hierarchy: np.ndarray,
    categories: tuple[object, ...],
    path: Path,
) -> None:
    """Write a readable category-labelled dendrogram."""

    path.parent.mkdir(parents=True, exist_ok=True)
    n_categories = len(categories)
    fig_height = max(6.0, min(24.0, 0.28 * n_categories + 2.0))
    fig, axis = plt.subplots(figsize=(10.0, fig_height))
    dendrogram(
        hierarchy,
        labels=[str(category) for category in categories],
        orientation="right",
        leaf_font_size=8,
        ax=axis,
    )
    axis.set_xlabel("Hellinger-space distance")
    axis.set_ylabel("Category")
    axis.set_title("Hierarchical clustering of induced categories")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
