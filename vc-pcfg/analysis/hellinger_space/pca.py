"""Principal component analysis of weighted Hellinger representations."""

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


def _feature_block_contributions(
    loadings: pd.DataFrame,
    component_names: list[str],
) -> pd.DataFrame:
    """Sum squared PCA loadings within each (position, feature) block."""

    rows: list[pd.DataFrame] = []
    group_columns = [
        "block_index",
        "domain",
        "position",
        "position_group",
        "feature",
        "feature_family",
        "block_weight",
    ]
    for component in component_names:
        work = loadings[group_columns].copy()
        work["squared_loading"] = loadings[component].to_numpy(dtype=float) ** 2
        grouped = (
            work.groupby(group_columns, sort=False, as_index=False)["squared_loading"]
            .sum()
            .rename(columns={"squared_loading": "contribution"})
        )
        total = float(grouped["contribution"].sum())
        if total > 0:
            grouped["contribution"] /= total
        grouped.insert(0, "component", component)
        grouped["contribution_pct"] = 100.0 * grouped["contribution"]
        grouped = grouped.sort_values("contribution", ascending=False, kind="stable")
        grouped["rank"] = np.arange(1, len(grouped) + 1)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _top_signed_loadings(
    loadings: pd.DataFrame,
    component_names: list[str],
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the strongest positive/negative value-level loadings per PC."""

    metadata = [
        "dimension",
        "block_index",
        "domain",
        "position",
        "position_group",
        "feature",
        "feature_family",
        "value",
        "block_weight",
    ]
    rows: list[dict[str, object]] = []
    for component in component_names:
        ordered_positive = loadings.sort_values(component, ascending=False, kind="stable").head(top_n)
        ordered_negative = loadings.sort_values(component, ascending=True, kind="stable").head(top_n)
        for side, subset in (("positive", ordered_positive), ("negative", ordered_negative)):
            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                item = {column: row[column] for column in metadata}
                item.update(
                    {
                        "component": component,
                        "side": side,
                        "rank": rank,
                        "loading": float(row[component]),
                        "absolute_loading": abs(float(row[component])),
                    }
                )
                rows.append(item)
    columns = ["component", "side", "rank", *metadata, "loading", "absolute_loading"]
    return pd.DataFrame(rows, columns=columns)


def run_pca(space: HellingerSpace) -> dict[str, object]:
    """Fit centered, unstandardized PCA directly to Hellinger coordinates."""

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
    block_contributions = _feature_block_contributions(loadings, component_names)
    top_loadings = _top_signed_loadings(loadings, component_names)

    return {
        "model": model,
        "coordinates": coordinates,
        "loadings": loadings,
        "explained_variance": explained,
        "feature_contributions": block_contributions,
        "top_loadings": top_loadings,
    }


def save_pca_scatter(
    coordinates: pd.DataFrame,
    path: Path,
    *,
    explained_variance: pd.DataFrame | None = None,
) -> None:
    """Write a category-labelled PC1/PC2 plot (or a 1-D fallback)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    has_pc2 = "PC2" in coordinates.columns

    fig, axis = plt.subplots(figsize=(9.0, 7.0))
    x = coordinates["PC1"].to_numpy(dtype=float)
    y = coordinates["PC2"].to_numpy(dtype=float) if has_pc2 else np.zeros(len(coordinates))
    axis.scatter(x, y)

    for category, x_value, y_value in zip(coordinates["category"], x, y):
        axis.annotate(
            str(category),
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )

    ratios: dict[str, float] = {}
    if explained_variance is not None and not explained_variance.empty:
        ratios = dict(
            zip(
                explained_variance["component"].astype(str),
                explained_variance["explained_variance_ratio"].astype(float),
            )
        )
    pc1_label = "PC1" + (f" ({100 * ratios['PC1']:.1f}%)" if "PC1" in ratios else "")
    pc2_label = "PC2" + (f" ({100 * ratios['PC2']:.1f}%)" if "PC2" in ratios else "")

    axis.axhline(0.0, linewidth=0.7)
    axis.axvline(0.0, linewidth=0.7)
    axis.set_xlabel(pc1_label)
    axis.set_ylabel(pc2_label if has_pc2 else "")
    axis.set_title("PCA of induced categories in weighted Hellinger space")
    if not has_pc2:
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
