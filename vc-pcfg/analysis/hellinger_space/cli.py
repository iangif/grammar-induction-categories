"""Command-line interface for Hellinger-space analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .constants import FEATURE_SPACES, PROJECTIONS
from .io import HellingerSpaceError
from .pipeline import run_analysis


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="category_feature_distributions.csv from the abstraction framework.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for Hellinger-space analysis outputs.",
    )
    weighting = parser.add_argument_group("Hellinger positional weighting")
    weighting.add_argument(
        "--target-weight",
        type=float,
        default=0.50,
        help="Relative weight assigned to TARGET lexical blocks before subspace renormalization.",
    )
    weighting.add_argument(
        "--near-context-weight",
        type=float,
        default=0.35,
        help="Combined relative weight assigned to L1/R1 blocks before subspace renormalization.",
    )
    weighting.add_argument(
        "--far-context-weight",
        type=float,
        default=0.15,
        help="Combined relative weight assigned to L2/R2 blocks before subspace renormalization.",
    )


def _add_feature_space_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--feature-space",
        choices=FEATURE_SPACES,
        default="all",
        help="Linguistic feature subspace used for this analysis.",
    )


def _add_cluster_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Optionally cut the hierarchy into this many flat clusters and write "
            "cluster_assignments.csv. The hierarchy itself does not require a cut."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct weighted Hellinger representations of induced-category feature "
            "distributions, then analyze them with clustering, PCA, or interactive projections."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    transform = subparsers.add_parser(
        "transform",
        help="Build weighted Hellinger category vectors and dimension metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(transform)
    _add_feature_space_argument(transform)

    cluster = subparsers.add_parser(
        "cluster",
        help="Build Hellinger space and perform average-linkage hierarchical clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(cluster)
    _add_feature_space_argument(cluster)
    _add_cluster_arguments(cluster)

    pca = subparsers.add_parser(
        "pca",
        help="Build Hellinger space and perform centered, unstandardized PCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(pca)
    _add_feature_space_argument(pca)

    all_parser = subparsers.add_parser(
        "all",
        help="Run transform, hierarchical clustering, and PCA together for one feature space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(all_parser)
    _add_feature_space_argument(all_parser)
    _add_cluster_arguments(all_parser)

    visualize = subparsers.add_parser(
        "visualize",
        help=(
            "Create an interactive single-model category map across selected feature spaces "
            "and projections."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(visualize)
    visualize.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="category_metrics.csv; defaults to the input file's directory.",
    )
    visualize.add_argument(
        "--feature-scores",
        type=Path,
        default=None,
        help="category_feature_scores.csv; defaults to the input file's directory.",
    )
    visualize.add_argument(
        "--spaces",
        nargs="+",
        choices=FEATURE_SPACES,
        default=list(FEATURE_SPACES),
        help="Feature-space views to include in the HTML.",
    )
    visualize.add_argument(
        "--projections",
        nargs="+",
        choices=PROJECTIONS,
        default=["pca", "umap"],
        help="2-D projections to include; t-SNE is available as a secondary exploratory view.",
    )
    visualize.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of highest-coherence feature/value summaries shown on hover.",
    )
    visualize.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used by UMAP and t-SNE.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "top_k", 1) <= 0:
        parser.error("--top-k must be positive")

    try:
        result = run_analysis(
            operation=args.operation,
            input_path=args.input,
            output_dir=args.output_dir,
            n_clusters=getattr(args, "n_clusters", None),
            feature_space=getattr(args, "feature_space", "all"),
            target_weight=args.target_weight,
            near_context_weight=args.near_context_weight,
            far_context_weight=args.far_context_weight,
            metrics_path=getattr(args, "metrics", None),
            feature_scores_path=getattr(args, "feature_scores", None),
            spaces=getattr(args, "spaces", None),
            projections=getattr(args, "projections", None),
            top_k=getattr(args, "top_k", 5),
            random_state=getattr(args, "random_state", 0),
        )
    except HellingerSpaceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.operation == "visualize":
        print(
            f"Built interactive views for {result['categories']} categories across "
            f"{len(result['spaces'])} feature spaces and {len(result['projections'])} projections."
        )
    else:
        print(
            f"Built {result['feature_space']} Hellinger space for {result['categories']} categories, "
            f"{result['blocks']} feature blocks, and {result['dimensions']} dimensions."
        )
    print(f"Operation: {result['operation']}")
    print(f"Wrote outputs to: {args.output_dir}")
    return 0
