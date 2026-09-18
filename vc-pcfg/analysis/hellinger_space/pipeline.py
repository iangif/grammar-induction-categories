"""Orchestration for weighted Hellinger transform, clustering, projections, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .clustering import run_hierarchical_clustering, save_dendrogram
from .constants import FEATURE_SPACES, PACKAGE_VERSION
from .io import (
    read_category_metrics,
    read_feature_distributions,
    read_feature_scores,
    sha256_file,
    write_csv,
    write_json,
)
from .pca import run_pca, save_pca_scatter
from .spaces import normalized_position_group_weights
from .transform import build_hellinger_space
from .visualization import build_interactive_visualization

Operation = Literal["transform", "cluster", "pca", "all", "visualize"]

_KNOWN_OUTPUTS = (
    "category_hellinger_vectors.csv",
    "hellinger_dimensions.csv",
    "hierarchical_linkage.csv",
    "category_order.csv",
    "dendrogram.png",
    "cluster_assignments.csv",
    "pca_coordinates.csv",
    "pca_loadings.csv",
    "pca_explained_variance.csv",
    "pca_feature_contributions.csv",
    "pca_top_loadings.csv",
    "pca_scatter.png",
    "projection_quality.csv",
    "interactive_category_map.html",
    "run_manifest.json",
)


def _remove_stale_outputs(output_dir: Path) -> None:
    for name in _KNOWN_OUTPUTS:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def _default_sibling(input_path: Path, name: str) -> Path:
    return input_path.parent / name


def _weight_dict(
    *,
    target_weight: float,
    near_context_weight: float,
    far_context_weight: float,
) -> dict[str, float]:
    return normalized_position_group_weights(
        {
            "target": target_weight,
            "near_context": near_context_weight,
            "far_context": far_context_weight,
        }
    )


def run_analysis(
    *,
    operation: Operation,
    input_path: Path,
    output_dir: Path,
    n_clusters: int | None = None,
    feature_space: str = "all",
    target_weight: float = 0.50,
    near_context_weight: float = 0.35,
    far_context_weight: float = 0.15,
    metrics_path: Path | None = None,
    feature_scores_path: Path | None = None,
    spaces: Sequence[str] | None = None,
    projections: Sequence[str] | None = None,
    top_k: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    """Run one CLI operation from the long feature-distribution source."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(output_dir)
    source = read_feature_distributions(input_path)
    base_weights = _weight_dict(
        target_weight=target_weight,
        near_context_weight=near_context_weight,
        far_context_weight=far_context_weight,
    )

    if operation == "visualize":
        metrics_path = metrics_path or _default_sibling(input_path, "category_metrics.csv")
        feature_scores_path = feature_scores_path or _default_sibling(
            input_path, "category_feature_scores.csv"
        )
        category_metrics = read_category_metrics(metrics_path)
        feature_scores = read_feature_scores(feature_scores_path)
        selected_spaces = list(spaces) if spaces is not None else list(FEATURE_SPACES)
        selected_projections = list(projections) if projections is not None else ["pca", "umap"]
        visualization = build_interactive_visualization(
            source=source,
            category_metrics=category_metrics,
            feature_scores=feature_scores,
            output_dir=output_dir,
            spaces=selected_spaces,
            projections=selected_projections,
            position_group_weights=base_weights,
            top_k=top_k,
            random_state=random_state,
        )
        outputs = list(visualization["outputs"])
        outputs.append("run_manifest.json")
        manifest = {
            "package": "analysis.hellinger_space",
            "package_version": PACKAGE_VERSION,
            "operation": operation,
            "input": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "category_metrics": str(metrics_path.resolve()),
            "category_metrics_sha256": sha256_file(metrics_path),
            "category_feature_scores": str(feature_scores_path.resolve()),
            "category_feature_scores_sha256": sha256_file(feature_scores_path),
            "output_dir": str(output_dir.resolve()),
            "source_rows": int(len(source)),
            "feature_spaces": selected_spaces,
            "projections": selected_projections,
            "top_k_hover_features": top_k,
            "random_state": random_state,
            "position_group_weights": base_weights,
            "subspace_weighting": (
                "Inactive positional groups are removed and remaining group weights are "
                "renormalized to sum to 1; each active group is split equally across its blocks."
            ),
            "space_coherence": (
                "Weighted mean of per-block normalized coherence using the exact normalized "
                "block weights of the displayed Hellinger space."
            ),
            "outputs": outputs,
        }
        write_json(manifest, output_dir / "run_manifest.json")
        return {
            "operation": operation,
            "categories": int(source["category"].nunique()),
            "blocks": None,
            "dimensions": None,
            "outputs": outputs,
            "spaces": selected_spaces,
            "projections": selected_projections,
        }

    space = build_hellinger_space(
        source,
        feature_space=feature_space,
        position_group_weights=base_weights,
    )
    outputs: list[str] = []

    # Every non-visualization operation materializes the common geometry so
    # cluster/PCA runs are independently inspectable and reproducible.
    vectors_path = output_dir / "category_hellinger_vectors.csv"
    dimensions_path = output_dir / "hellinger_dimensions.csv"
    write_csv(space.vectors, vectors_path)
    write_csv(space.dimensions, dimensions_path)
    outputs.extend([vectors_path.name, dimensions_path.name])

    clustering_result: dict[str, Any] | None = None
    if operation in {"cluster", "all"}:
        clustering_result = run_hierarchical_clustering(space, n_clusters=n_clusters)

        linkage_path = output_dir / "hierarchical_linkage.csv"
        order_path = output_dir / "category_order.csv"
        dendrogram_path = output_dir / "dendrogram.png"
        write_csv(clustering_result["linkage"], linkage_path)
        write_csv(clustering_result["order"], order_path)
        save_dendrogram(clustering_result["hierarchy"], space.categories, dendrogram_path)
        outputs.extend([linkage_path.name, order_path.name, dendrogram_path.name])

        if "assignments" in clustering_result:
            assignments_path = output_dir / "cluster_assignments.csv"
            write_csv(clustering_result["assignments"], assignments_path)
            outputs.append(assignments_path.name)

    pca_result: dict[str, Any] | None = None
    if operation in {"pca", "all"}:
        pca_result = run_pca(space)

        pca_frames = {
            "pca_coordinates.csv": pca_result["coordinates"],
            "pca_loadings.csv": pca_result["loadings"],
            "pca_explained_variance.csv": pca_result["explained_variance"],
            "pca_feature_contributions.csv": pca_result["feature_contributions"],
            "pca_top_loadings.csv": pca_result["top_loadings"],
        }
        for name, frame in pca_frames.items():
            write_csv(frame, output_dir / name)
            outputs.append(name)

        scatter_path = output_dir / "pca_scatter.png"
        save_pca_scatter(
            pca_result["coordinates"],
            scatter_path,
            explained_variance=pca_result["explained_variance"],
        )
        outputs.append(scatter_path.name)

    outputs.append("run_manifest.json")
    manifest = {
        "package": "analysis.hellinger_space",
        "package_version": PACKAGE_VERSION,
        "operation": operation,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "output_dir": str(output_dir.resolve()),
        "source_rows": int(len(source)),
        "categories": int(len(space.categories)),
        "feature_space": feature_space,
        "feature_blocks": int(space.block_count),
        "dimensions": int(space.matrix.shape[1]),
        "position_group_weights": base_weights,
        "active_position_group_weights": space.position_group_weights,
        "hellinger_transform": {
            "coordinate": "sqrt(block_weight / 2) * sqrt(proportion)",
            "block": "one (position, feature) probability distribution",
            "block_weights_sum": 1.0,
            "missing_category_value_rows": "reconstructed as probability 0",
            "distance": "squared Euclidean distance equals weighted sum of squared Hellinger block distances",
            "subspace_scaling": "active block weights are renormalized to sum to 1",
            "standardization": "none",
        },
        "clustering": (
            {
                "method": "average linkage",
                "metric": "Euclidean distance in normalized weighted Hellinger space",
                "n_clusters": n_clusters,
            }
            if operation in {"cluster", "all"}
            else None
        ),
        "pca": (
            {
                "centering": "yes (sklearn PCA default)",
                "variance_standardization": "no",
                "n_components": int(len(pca_result["explained_variance"])) if pca_result else None,
                "interpretation_outputs": [
                    "pca_feature_contributions.csv",
                    "pca_top_loadings.csv",
                ],
            }
            if operation in {"pca", "all"}
            else None
        ),
        "outputs": outputs,
    }
    write_json(manifest, output_dir / "run_manifest.json")

    return {
        "operation": operation,
        "categories": len(space.categories),
        "blocks": space.block_count,
        "dimensions": space.matrix.shape[1],
        "outputs": outputs,
        "feature_space": feature_space,
    }
