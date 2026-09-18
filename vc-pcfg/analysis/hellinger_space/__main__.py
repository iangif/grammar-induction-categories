"""CLI entry point for weighted Hellinger-space analyses.

Use .venv-analysis for the virtual environment. From ``vc-pcfg/`` examples are:

- Install: scikit-learn plotly umap-learn

Run from vc-pcfg/ with:
    python -m analysis.hellinger_space all \
      --input analysis/outputs/abstraction/s91-e5-c60/category_feature_distributions.csv \
      --output-dir analysis/outputs/hellinger/s91-e5-c60 \
      --feature-space all

    python -m analysis.hellinger_space visualize \
      --input analysis/outputs/abstraction/s91-e5-c60/category_feature_distributions.csv \
      --output-dir analysis/outputs/hellinger/s91-e5-c60/visualization
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
