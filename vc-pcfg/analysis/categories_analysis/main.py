"""Analyze induced preterminal categories exported one token per row.

This script creates:

* a simple integrity audit
* a corpus-level category summary
* complete word/category conditional-probability tables
* frequent-word and weighted-log-odds diagnostic rankings
* immediate-context distributions
* representative examples
* standardized per-category evidence packets (JSON plus example CSVs)
* LLM-ready [word, sentence] tables
* a raw category-by-word matrix and a context-sensitive word/POS matrix
* POS-delimited category-by-word heatmaps
* category-overlap matrices in original, average-similarity, and
  hierarchical-clustering orderings

POS tagging is performed in sentence context while preserving the exact exported
word boundaries: one spaCy ``Doc`` is created from the token sequence for each
``sent_id``. Two lexical representations are retained:

* raw surface words, such as ``play``;
* context-sensitive word/POS units, such as ``play.VERB`` and ``play.NOUN``.

The raw-word heatmap is ordered by each surface word's dominant observed POS.
The word/POS heatmap preserves the POS assigned to each token in context.

Weighted log odds
-----------------
For each word and category, the script compares the word's odds inside the
category with its odds in all other categories. It uses an informative
Dirichlet prior based on the corpus-wide word distribution and reports a
variance-normalized z-score. Diagnostic rankings require both a positive score
and a configurable minimum total word count.

Example
---
    source .venv-analysis/bin/activate

    # install dependencies if necessary
    uv pip install numpy pandas matplotlib spacy click scipy
    uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

    cd vc-pcfg

    python -m analysis.analyze_word_categories \
        --input analysis/outputs/s91-e5.csv \
        --output-dir analysis/outputs/category_analysis/s91-e5 \
        --spacy-model en_core_web_sm
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .cli import build_parser
from .context import build_ranked_context_table, build_position_distribution
from .distributions import (
    build_word_category_distribution,
    build_word_pos_category_distribution,
    build_word_category_ambiguity,
    build_pos_ambiguity_summary,
    build_category_summary,
)
from .integrity import run_integrity_audit
from ..shared_utilities.io import write_csv
from .io import load_input
from .matrices import build_category_word_matrix, build_category_word_pos_matrix
from .plots import (
    save_category_lexical_heatmap,
    save_category_pos_heatmap,
)
from .preprocessing import prepare_input
from .reporting import (
    write_per_category_outputs,
    write_manifest,
    write_overlap_analysis,
)
from .scoring import add_weighted_log_odds, build_word_rankings
from .tagging import (
    add_spacy_pos,
    build_word_pos_summary,
    build_word_pos_unit_summary,
    build_category_pos_distribution,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.num_categories <= 0:
        parser.error("--num-categories must be greater than zero.")
    if args.top_n_words <= 0:
        parser.error("--top-n-words must be greater than zero.")
    if args.min_diagnostic_count <= 0:
        parser.error("--min-diagnostic-count must be greater than zero.")

    nonnegative_evidence_args = {
        "--evidence-frequent-words": args.evidence_frequent_words,
        "--evidence-diagnostic-words": args.evidence_diagnostic_words,
        "--evidence-examples-per-word": args.evidence_examples_per_word,
        "--evidence-random-examples": args.evidence_random_examples,
        "--evidence-ambiguous-words": args.evidence_ambiguous_words,
        "--evidence-examples-per-ambiguous-word": (
            args.evidence_examples_per_ambiguous_word
        ),
        "--evidence-contrast-examples-per-ambiguous-word": (
            args.evidence_contrast_examples_per_ambiguous_word
        ),
        "--evidence-rare-examples": args.evidence_rare_examples,
        "--evidence-top-contexts": args.evidence_top_contexts,
        "--evidence-top-frames": args.evidence_top_frames,
        "--evidence-top-word-pos-units": args.evidence_top_word_pos_units,
    }
    for option, value in nonnegative_evidence_args.items():
        if value < 0:
            parser.error(f"{option} must be zero or greater.")
    if args.evidence_rare_max_count <= 0:
        parser.error("--evidence-rare-max-count must be greater than zero.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, output_dir)

    print(f"Loading {args.input} ...", flush=True)
    raw = load_input(args.input)
    df = prepare_input(raw)

    print("Running integrity audit ...", flush=True)
    _, _, integrity_problem = run_integrity_audit(df, output_dir, args.num_categories)

    observed_categories = set(df["viterbi_preterminal"].unique().tolist())
    expected_categories = set(range(args.num_categories))
    unexpected = sorted(observed_categories - expected_categories)
    if unexpected:
        raise ValueError(
            f"Observed categories outside 0..{args.num_categories - 1}: {unexpected}. "
            "The audit was written before stopping."
        )
    if integrity_problem and args.strict_integrity:
        raise RuntimeError(
            "Integrity problems were found. See integrity/integrity_audit.csv and "
            "integrity/sentence_integrity_issues.csv."
        )
    if integrity_problem:
        warnings.warn(
            "Integrity problems were found, but analysis will continue because "
            "--strict-integrity was not set.",
            stacklevel=2,
        )

    print(
        f"Tagging exact exported tokens with spaCy model '{args.spacy_model}' ...",
        flush=True,
    )
    df = add_spacy_pos(
        df,
        model_name=args.spacy_model,
        batch_size=args.spacy_batch_size,
        n_process=args.spacy_processes,
    )
    word_pos_summary = build_word_pos_summary(df)
    word_pos_unit_summary = build_word_pos_unit_summary(df)
    write_csv(
        df[
            [
                "sent_id",
                "word_index",
                "word",
                "spacy_pos",
                "spacy_tag",
                "word_pos",
                "viterbi_preterminal",
                "sentence",
            ]
        ],
        output_dir / "pos" / "token_pos_assignments.csv",
    )
    write_csv(word_pos_summary, output_dir / "pos" / "word_pos_summary.csv")
    write_csv(word_pos_unit_summary, output_dir / "pos" / "word_pos_unit_summary.csv")

    category_ids = list(range(args.num_categories))

    print("Computing word/category distributions and rankings ...", flush=True)
    word_category = build_word_category_distribution(df)
    write_csv(word_category, output_dir / "word_category_distribution.csv")

    word_pos_category = build_word_pos_category_distribution(df)
    write_csv(
        word_pos_category,
        output_dir / "word_pos_category_distribution.csv",
    )

    word_ambiguity = build_word_category_ambiguity(df)
    write_csv(
        word_ambiguity,
        output_dir / "word_ambiguity.csv",
    )

    pos_ambiguity = build_pos_ambiguity_summary(df)
    write_csv(
        pos_ambiguity,
        output_dir / "pos" / "pos_ambiguity_summary.csv",
    )

    word_scores = add_weighted_log_odds(
        word_category,
        df,
        prior_strength=args.log_odds_prior_strength,
    )
    write_csv(
        word_scores[
            [
                "c",
                "w",
                "count",
                "word_total_count",
                "category_total_count",
                "outside_word_count",
                "p_word_given_category",
                "p_category_given_word",
                "n_sentences",
                "weighted_log_odds",
            ]
        ].sort_values(["c", "weighted_log_odds"], ascending=[True, False]),
        output_dir / "weighted_log_odds_scores.csv",
    )

    frequent_words, diagnostic_words = build_word_rankings(
        word_scores,
        top_n=args.top_n_words,
        min_diagnostic_count=args.min_diagnostic_count,
    )
    write_csv(frequent_words, output_dir / "frequent_word_rankings.csv")
    write_csv(diagnostic_words, output_dir / "diagnostic_word_rankings.csv")

    summary = build_category_summary(df, word_category, category_ids)
    write_csv(summary, output_dir / "category_summary.csv")

    print("Computing immediate-context distributions ...", flush=True)
    previous_words = build_ranked_context_table(
        df, ["previous_word"], args.top_contexts
    )
    next_words = build_ranked_context_table(df, ["next_word"], args.top_contexts)
    frames = build_ranked_context_table(
        df, ["previous_word", "next_word"], args.top_contexts
    )
    positions = build_position_distribution(df, category_ids, args.position_bins)
    write_csv(previous_words, output_dir / "context" / "previous_word_rankings.csv")
    write_csv(next_words, output_dir / "context" / "next_word_rankings.csv")
    write_csv(frames, output_dir / "context" / "context_frame_rankings.csv")
    write_csv(
        positions, output_dir / "context" / "normalized_position_distribution.csv"
    )

    print("Building spaCy POS and lexical-distribution matrices ...", flush=True)
    category_pos_distribution, category_pos_matrix = build_category_pos_distribution(
        df, category_ids
    )
    write_csv(
        category_pos_distribution, output_dir / "pos" / "category_pos_distribution.csv"
    )
    write_csv(
        category_pos_matrix.reset_index(),
        output_dir / "pos" / "category_pos_matrix.csv",
    )

    category_word_matrix = build_category_word_matrix(
        word_category,
        word_pos_summary,
        category_ids,
    )
    category_word_pos_matrix = build_category_word_pos_matrix(
        word_pos_category,
        word_pos_unit_summary,
        category_ids,
    )

    raw_words_dir = output_dir / "matrices" / "raw_words"
    pos_split_dir = output_dir / "matrices" / "pos_split_words"
    write_csv(
        category_word_matrix.reset_index(),
        raw_words_dir / "category_by_word_p_word_given_category.csv",
    )
    write_csv(
        category_word_pos_matrix.reset_index(),
        pos_split_dir / "category_by_word_pos_p_word_pos_given_category.csv",
    )

    # Preserve the original raw-word matrix path for compatibility with earlier
    # versions of this script.
    write_csv(
        category_word_matrix.reset_index(),
        output_dir / "matrices" / "category_by_word_p_word_given_category.csv",
    )

    token_counts = df.groupby("viterbi_preterminal").size()
    raw_overlap, raw_redundancy = write_overlap_analysis(
        matrix=category_word_matrix,
        output_dir=output_dir,
        representation_slug="raw_words",
        representation_title="Raw surface words",
        category_ids=category_ids,
        token_counts=token_counts,
        top_k_value=args.top_k_overlap,
        cluster_linkage=args.cluster_linkage,
        skip_plots=args.skip_plots,
    )
    pos_overlap, pos_redundancy = write_overlap_analysis(
        matrix=category_word_pos_matrix,
        output_dir=output_dir,
        representation_slug="pos_split_words",
        representation_title="Context-sensitive word/POS units",
        category_ids=category_ids,
        token_counts=token_counts,
        top_k_value=args.top_k_overlap,
        cluster_linkage=args.cluster_linkage,
        skip_plots=args.skip_plots,
    )

    raw_redundancy.insert(0, "representation", "raw_words")
    pos_redundancy.insert(0, "representation", "pos_split_words")
    write_csv(
        pd.concat([raw_redundancy, pos_redundancy], ignore_index=True),
        output_dir / "matrices" / "redundancy_summary_all_representations.csv",
    )

    # Preserve the original raw overlap-table paths for compatibility.
    write_csv(
        raw_overlap["cosine_similarity"].reset_index(),
        output_dir / "matrices" / "cosine_similarity.csv",
    )
    write_csv(
        raw_overlap["js_divergence"].reset_index(),
        output_dir / "matrices" / "js_divergence.csv",
    )
    write_csv(
        raw_overlap[f"top_{args.top_k_overlap}_overlap"].reset_index(),
        output_dir / "matrices" / f"top_{args.top_k_overlap}_word_overlap.csv",
    )

    if not args.skip_plots:
        save_category_lexical_heatmap(
            category_word_matrix,
            lexical_summary=word_pos_summary.rename(
                columns={"dominant_pos": "pos", "total_count": "token_count"}
            ),
            label_column="word",
            pos_column="pos",
            count_column="token_count",
            max_columns=args.matrix_heatmap_words,
            path=raw_words_dir / "heatmaps" / "category_by_word_pos_sections.png",
            title=(
                f"P(word | category), {args.matrix_heatmap_words} most frequent raw words"
            ),
            xlabel="Raw word, grouped by dominant context-sensitive spaCy POS",
            colorbar_label="P(word | category)",
        )
        save_category_lexical_heatmap(
            category_word_pos_matrix,
            lexical_summary=word_pos_unit_summary,
            label_column="word_pos",
            pos_column="pos",
            count_column="token_count",
            max_columns=args.matrix_heatmap_words,
            path=pos_split_dir / "heatmaps" / "category_by_word_pos_sections.png",
            title=(
                f"P(word.POS | category), {args.matrix_heatmap_words} most frequent "
                "context-sensitive units"
            ),
            xlabel="Surface word.POS, grouped by token-level spaCy POS",
            colorbar_label="P(word.POS | category)",
        )
        save_category_pos_heatmap(
            category_pos_matrix,
            output_dir / "pos" / "category_pos_heatmap.png",
        )

    print("Writing per-category reports and LLM-ready input tables ...", flush=True)
    combined_llm_input = write_per_category_outputs(
        output_dir=output_dir,
        category_ids=category_ids,
        summary=summary,
        word_category=word_category,
        word_scores=word_scores,
        word_ambiguity=word_ambiguity,
        word_pos_category=word_pos_category,
        category_pos_distribution=category_pos_distribution,
        frequent_words=frequent_words,
        diagnostic_words=diagnostic_words,
        previous_words=previous_words,
        next_words=next_words,
        frames=frames,
        positions=positions,
        df=df,
        args=args,
    )
    write_csv(combined_llm_input, output_dir / "llm_input_all_categories.csv")

    print(f"Done. Outputs written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
