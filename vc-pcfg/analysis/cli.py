import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze induced preterminal categories from a token-level CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Argument groups allow arguments to be displayed by category in the output of --help.
    io_parser = parser.add_argument_group("Input/output")
    spacy_parser = parser.add_argument_group("spaCy")
    analysis_parser = parser.add_argument_group("Analysis")
    examples_parser = parser.add_argument_group("Examples")
    plots_parser = parser.add_argument_group("Plots")
    evidence_parser = parser.add_argument_group("Evidence")

    # IO arguments
    io_parser.add_argument(
        "--input", required=True, type=Path, help="Input CSV or Parquet file."
    )
    io_parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for analysis outputs."
    )

    # SpaCy arguments
    spacy_parser.add_argument(
        "--spacy-model", default="en_core_web_sm", help="spaCy model name or path."
    )
    spacy_parser.add_argument("--spacy-batch-size", type=int, default=256)
    spacy_parser.add_argument(
        "--spacy-processes",
        type=int,
        default=1,
        help="Processes passed to spaCy nlp.pipe. Use cautiously on clusters.",
    )

    # Analysis arguments
    analysis_parser.add_argument(
        "--num-categories",
        type=int,
        default=60,
        help="Expected categories, assumed to be numbered 0 through N-1.",
    )
    analysis_parser.add_argument(
        "--top-n-words", type=int, default=25, help="Rows in each top-word ranking."
    )
    analysis_parser.add_argument(
        "--min-diagnostic-count",
        type=int,
        default=10,
        help="Minimum corpus frequency for a word to enter diagnostic rankings.",
    )
    analysis_parser.add_argument(
        "--log-odds-prior-strength",
        type=float,
        default=1000.0,
        help="Total mass of the informative Dirichlet prior for weighted log odds.",
    )
    analysis_parser.add_argument("--top-contexts", type=int, default=25)
    analysis_parser.add_argument("--position-bins", type=int, default=10)
    analysis_parser.add_argument(
        "--examples-per-word",
        type=int,
        default=2,
        help="Representative sentences sampled for each frequent/diagnostic word.",
    )
    analysis_parser.add_argument(
        "--top-k-overlap",
        type=int,
        default=25,
        help="K for |TopK(c1) intersection TopK(c2)| / K.",
    )
    analysis_parser.add_argument("--random-seed", type=int, default=42)
    analysis_parser.add_argument(
        "--strict-integrity",
        action="store_true",
        help="Stop after writing the audit if sentence lengths, assignments, or token positions are inconsistent.",
    )

    # Examples arguments
    examples_parser.add_argument("--random-examples", type=int, default=20)
    examples_parser.add_argument("--top-frames-for-examples", type=int, default=5)
    examples_parser.add_argument("--examples-per-frame", type=int, default=3)
    examples_parser.add_argument(
        "--llm-input-max-rows",
        type=int,
        default=0,
        help=(
            "Maximum token rows in each category's LLM-input table. "
            "Use 0 to include all rows."
        ),
    )

    # Plots arguments
    plots_parser.add_argument(
        "--matrix-heatmap-words",
        type=int,
        default=100,
        help=(
            "Most frequent raw words or word/POS units shown in each readable "
            "category-by-lexical-unit heatmap. The CSV matrices contain all columns."
        ),
    )
    plots_parser.add_argument(
        "--cluster-linkage",
        choices=["average", "complete", "single", "weighted"],
        default="average",
        help="Linkage method used for hierarchical ordering of overlap heatmaps.",
    )
    plots_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write all tables but skip PNG plots and heatmaps.",
    )

    # Evidence arguments
    evidence_parser.add_argument(
        "--evidence-frequent-words",
        type=int,
        default=20,
        help="Frequent word types included in each standardized evidence packet.",
    )
    evidence_parser.add_argument(
        "--evidence-diagnostic-words",
        type=int,
        default=20,
        help="Weighted-log-odds diagnostic word types included in each evidence packet.",
    )
    evidence_parser.add_argument(
        "--evidence-examples-per-word",
        type=int,
        default=1,
        help="Target-category sentence examples sampled for each frequent or diagnostic word.",
    )
    evidence_parser.add_argument(
        "--evidence-random-examples",
        type=int,
        default=20,
        help="Random target-category token examples included in each evidence packet.",
    )
    evidence_parser.add_argument(
        "--evidence-ambiguous-words",
        type=int,
        default=10,
        help=(
            "Words assigned to multiple induced categories whose within-category and "
            "cross-category behavior is summarized in each evidence packet."
        ),
    )
    evidence_parser.add_argument(
        "--evidence-examples-per-ambiguous-word",
        type=int,
        default=2,
        help="Target-category examples sampled for each selected ambiguous word.",
    )
    evidence_parser.add_argument(
        "--evidence-contrast-examples-per-ambiguous-word",
        type=int,
        default=2,
        help="Examples from other categories sampled for each selected ambiguous word.",
    )
    evidence_parser.add_argument(
        "--evidence-rare-examples",
        type=int,
        default=10,
        help="Low-frequency target-category examples included in each evidence packet.",
    )
    evidence_parser.add_argument(
        "--evidence-rare-max-count",
        type=int,
        default=5,
        help="Maximum corpus token count for a word to qualify as rare evidence.",
    )
    evidence_parser.add_argument(
        "--evidence-top-contexts",
        type=int,
        default=10,
        help="Previous-word and next-word contexts included in each evidence packet.",
    )
    evidence_parser.add_argument(
        "--evidence-top-frames",
        type=int,
        default=10,
        help="Immediate previous/next word frames included in each evidence packet.",
    )
    evidence_parser.add_argument(
        "--evidence-top-word-pos-units",
        type=int,
        default=20,
        help="Context-sensitive word.POS units included in each evidence packet.",
    )

    return parser
