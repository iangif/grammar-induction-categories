"""Command-line interface for Hellinger-space analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

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
            "Construct normalized Hellinger representations of induced-category feature "
            "distributions, then analyze the same geometry with hierarchical clustering or PCA."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    transform = subparsers.add_parser(
        "transform",
        help="Build the normalized Hellinger category vectors and dimension metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(transform)

    cluster = subparsers.add_parser(
        "cluster",
        help="Build Hellinger space and perform average-linkage hierarchical clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(cluster)
    _add_cluster_arguments(cluster)

    pca = subparsers.add_parser(
        "pca",
        help="Build Hellinger space and perform centered, unstandardized PCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(pca)

    all_parser = subparsers.add_parser(
        "all",
        help="Run transform, hierarchical clustering, and PCA together.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(all_parser)
    _add_cluster_arguments(all_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_analysis(
            operation=args.operation,
            input_path=args.input,
            output_dir=args.output_dir,
            n_clusters=getattr(args, "n_clusters", None),
        )
    except HellingerSpaceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Built Hellinger space for {result['categories']} categories, "
        f"{result['blocks']} feature blocks, and {result['dimensions']} dimensions."
    )
    print(f"Operation: {result['operation']}")
    print(f"Wrote outputs to: {args.output_dir}")
    return 0
