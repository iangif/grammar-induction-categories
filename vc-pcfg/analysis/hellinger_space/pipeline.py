"""Orchestration for transform, clustering, PCA, and combined analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .clustering import run_hierarchical_clustering, save_dendrogram
from .constants import PACKAGE_VERSION
from .io import read_feature_distributions, sha256_file, write_csv, write_json
from .pca import run_pca, save_pca_scatter
from .transform import build_hellinger_space

Operation = Literal["transform", "cluster", "pca", "all"]

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
    "pca_scatter.png",
    "run_manifest.json",
)


def _remove_stale_outputs(output_dir: Path) -> None:
    for name in _KNOWN_OUTPUTS:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def run_analysis(
    *,
    operation: Operation,
    input_path: Path,
    output_dir: Path,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    """Run one CLI operation from the long feature-distribution source."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(output_dir)
    source = read_feature_distributions(input_path)
    space = build_hellinger_space(source)

    outputs: list[str] = []

    # Every operation materializes the common geometry so cluster/PCA runs are
    # independently inspectable and reproducible without a preceding command.
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

        coordinates_path = output_dir / "pca_coordinates.csv"
        loadings_path = output_dir / "pca_loadings.csv"
        explained_path = output_dir / "pca_explained_variance.csv"
        scatter_path = output_dir / "pca_scatter.png"
        write_csv(pca_result["coordinates"], coordinates_path)
        write_csv(pca_result["loadings"], loadings_path)
        write_csv(pca_result["explained_variance"], explained_path)
        save_pca_scatter(pca_result["coordinates"], scatter_path)
        outputs.extend(
            [coordinates_path.name, loadings_path.name, explained_path.name, scatter_path.name]
        )

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
        "feature_blocks": int(space.block_count),
        "dimensions": int(space.matrix.shape[1]),
        "hellinger_transform": {
            "coordinate": "sqrt(proportion) / sqrt(2 * number_of_feature_blocks)",
            "block": "one (position, feature) probability distribution",
            "missing_category_value_rows": "reconstructed as probability 0",
            "distance": (
                "Euclidean distance equals root-mean-square Hellinger distance "
                "across feature blocks"
            ),
            "standardization": "none",
        },
        "clustering": (
            {
                "method": "average linkage",
                "metric": "Euclidean distance in normalized Hellinger space",
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
            }
            if operation in {"pca", "all"}
            else None
        ),
        "outputs": outputs,
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest, manifest_path)

    return {
        "operation": operation,
        "categories": len(space.categories),
        "blocks": space.block_count,
        "dimensions": space.matrix.shape[1],
        "outputs": outputs,
    }
