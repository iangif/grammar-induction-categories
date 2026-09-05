"""

Use .venv-analysis for virtual environment.

Run from vc-pcfg/ with:
    python -m analysis.hellinger_space all \
    --input analysis/outputs/abstraction/s91-e5-c60/category_feature_distributions.csv \
    --output-dir analysis/outputs/hellinger/s91-e5-c60

"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
