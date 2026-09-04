"""

Use .venv-analysis for virtual environment.

Add pyarrow with:
    uv pip install pyarrow

Install WordNet with:
    uv pip install nltk
    uv run python -m nltk.downloader wordnet

Run from vc-pcfg/ with:
    python -m analysis.abstraction_framework \
    --input analysis/outputs/word_csvs/s91-e5-c60.csv \
    --output-dir analysis/outputs/abstraction/s91-e5-c60
"""

from .cli import main

raise SystemExit(main())
