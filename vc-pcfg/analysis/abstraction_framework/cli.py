
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .io import FrameworkError, InputColumns
from .pipeline import run_pipeline

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate induced preterminal categories, write long-format linguistic feature "
            "distributions, and compute category diversity/coherence metrics"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Token-level CSV or Parquet export.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for framework outputs.")

    columns = parser.add_argument_group("input columns")
    columns.add_argument("--category-column", default="viterbi_preterminal")
    columns.add_argument("--sent-id-column", default="sent_id")
    columns.add_argument("--sent-len-column", default="sent_len")
    columns.add_argument("--word-index-column", default="word_index")
    columns.add_argument("--word-column", default="word")
    columns.add_argument("--sentence-column", default="sentence")

    annotation = parser.add_argument_group("lexical annotation")
    annotation.add_argument("--spacy-model", default="en_core_web_sm")
    annotation.add_argument("--spacy-batch-size", type=int, default=256)
    annotation.add_argument("--spacy-processes", type=int, default=1)
    annotation.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore any matching annotated_tokens.parquet cache and annotate again.",
    )
    annotation.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable all tqdm progress bars (useful for quiet batch logs).",
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.spacy_batch_size <= 0:
        parser.error("--spacy-batch-size must be positive")
    if args.spacy_processes <= 0:
        parser.error("--spacy-processes must be positive")

    input_columns = InputColumns(
        category=args.category_column,
        sent_id=args.sent_id_column,
        sent_len=args.sent_len_column,
        word_index=args.word_index_column,
        word=args.word_column,
        sentence=args.sentence_column,
    )

    try:
        result = run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            columns=input_columns,
            spacy_model=args.spacy_model,
            spacy_batch_size=args.spacy_batch_size,
            spacy_processes=args.spacy_processes,
            refresh_cache=args.refresh_cache,
            show_progress=not args.no_progress,
        )
    except FrameworkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    
    cache_text = "reused" if result["cache_hit"] else "written"
    print(f"Analyzed {result['rows']} tokens across {result['categories']} categories.")
    print(f"annotated_tokens.parquet: {cache_text}")
    print(f"Wrote outputs to: {args.output_dir}")
    return 0
