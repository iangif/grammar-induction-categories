import argparse
from typing import Any

import numpy as np
import pandas as pd

from .constants.columns import EVIDENCE_EXAMPLE_COLUMNS
from .examples import choose_rows
from ..shared_utilities.io import dataframe_records


def build_evidence_word_rankings(
    word_scores: pd.DataFrame,
    category: int,
    frequent_n: int,
    diagnostic_n: int,
    min_diagnostic_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build packet-specific lexical rankings without depending on --top-n-words."""
    category_scores = word_scores.loc[word_scores["c"].eq(category)].copy()

    frequent = category_scores.sort_values(
        ["count", "p_category_given_word", "w"],
        ascending=[False, False, True],
        kind="stable",
    ).head(frequent_n)
    frequent = frequent.copy()
    frequent.insert(0, "rank", np.arange(1, len(frequent) + 1))
    frequent = frequent[
        [
            "rank",
            "w",
            "count",
            "word_total_count",
            "p_word_given_category",
            "p_category_given_word",
            "n_sentences",
        ]
    ]

    diagnostic = (
        category_scores.loc[
            category_scores["word_total_count"].ge(min_diagnostic_count)
            & category_scores["weighted_log_odds"].gt(0)
        ]
        .sort_values(
            ["weighted_log_odds", "count", "w"],
            ascending=[False, False, True],
            kind="stable",
        )
        .head(diagnostic_n)
    )
    diagnostic = diagnostic.copy()
    diagnostic.insert(0, "rank", np.arange(1, len(diagnostic) + 1))
    diagnostic = diagnostic[
        [
            "rank",
            "w",
            "weighted_log_odds",
            "count",
            "word_total_count",
            "p_word_given_category",
            "p_category_given_word",
            "n_sentences",
        ]
    ]
    return frequent, diagnostic


def choose_diverse_category_rows(
    frame: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Choose rows while preferring coverage of different assigned categories."""
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()

    candidates = frame.drop_duplicates(["viterbi_preterminal", "sent_id", "word_index"])
    first_pass: list[pd.DataFrame] = []
    for _, group in candidates.groupby("viterbi_preterminal", sort=True):
        first_pass.append(choose_rows(group, 1, rng))

    diverse = (
        pd.concat(first_pass, ignore_index=False)
        if first_pass
        else candidates.iloc[0:0]
    )
    diverse = choose_rows(diverse, min(n, len(diverse)), rng)
    if len(diverse) >= n:
        return diverse

    remaining = candidates.drop(index=diverse.index, errors="ignore")
    extra = choose_rows(remaining, n - len(diverse), rng)
    return pd.concat([diverse, extra], ignore_index=False)


def append_evidence_examples(
    selected: list[pd.DataFrame],
    rows: pd.DataFrame,
    target_category: int,
    example_role: str,
    reason: str,
) -> None:
    if rows.empty:
        return
    chunk = rows[
        [
            "viterbi_preterminal",
            "word",
            "word_pos",
            "spacy_pos",
            "spacy_tag",
            "sent_id",
            "word_index",
            "previous_word",
            "next_word",
            "sentence",
        ]
    ].copy()
    chunk = chunk.rename(
        columns={
            "viterbi_preterminal": "assigned_category",
            "word": "w",
            "spacy_pos": "pos",
            "spacy_tag": "tag",
        }
    )
    chunk.insert(0, "target_category", target_category)
    chunk.insert(2, "example_role", example_role)
    chunk.insert(3, "evidence_reasons", reason)
    selected.append(chunk)


def build_ambiguity_profiles(
    word_scores: pd.DataFrame,
    word_ambiguity: pd.DataFrame,
    category: int,
    max_words: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select ambiguous words and summarize their distributions over categories."""
    target_scores = word_scores.loc[word_scores["c"].eq(category)].merge(
        word_ambiguity.rename(
            columns={"word": "w", "token_count": "ambiguity_token_count"}
        ),
        on="w",
        how="left",
    )
    selected_words = (
        target_scores.loc[target_scores["num_categories"].gt(1)]
        .sort_values(
            ["num_categories", "word_total_count", "count", "w"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        .head(max_words)
    )

    profiles: list[dict[str, Any]] = []
    for row in selected_words.itertuples(index=False):
        distribution = word_scores.loc[
            word_scores["w"].eq(row.w),
            [
                "c",
                "count",
                "p_category_given_word",
                "p_word_given_category",
            ],
        ].sort_values(["count", "c"], ascending=[False, True], kind="stable")
        profiles.append(
            {
                "word": row.w,
                "num_categories": int(row.num_categories),
                "corpus_token_count": int(row.word_total_count),
                "target_category_count": int(row.count),
                "p_target_category_given_word": float(row.p_category_given_word),
                "category_distribution": dataframe_records(distribution),
            }
        )
    return selected_words, profiles


def build_evidence_examples(
    df: pd.DataFrame,
    word_scores: pd.DataFrame,
    word_ambiguity: pd.DataFrame,
    category: int,
    frequent: pd.DataFrame,
    diagnostic: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select balanced target-category evidence and explicit cross-category contrasts."""
    category_rows = df.loc[df["viterbi_preterminal"].eq(category)]
    if category_rows.empty:
        return pd.DataFrame(columns=EVIDENCE_EXAMPLE_COLUMNS), []

    rng = np.random.default_rng(args.random_seed + 10_000 + category)
    selected: list[pd.DataFrame] = []

    for row in frequent.itertuples(index=False):
        candidates = category_rows.loc[category_rows["word"].eq(row.w)]
        append_evidence_examples(
            selected,
            choose_rows(candidates, args.evidence_examples_per_word, rng),
            category,
            "target_category",
            f"frequent_word:rank={row.rank}",
        )

    for row in diagnostic.itertuples(index=False):
        candidates = category_rows.loc[category_rows["word"].eq(row.w)]
        append_evidence_examples(
            selected,
            choose_rows(candidates, args.evidence_examples_per_word, rng),
            category,
            "target_category",
            f"diagnostic_word:rank={row.rank}",
        )

    ambiguous_words, ambiguity_profiles = build_ambiguity_profiles(
        word_scores=word_scores,
        word_ambiguity=word_ambiguity,
        category=category,
        max_words=args.evidence_ambiguous_words,
    )
    for row in ambiguous_words.itertuples(index=False):
        target_candidates = category_rows.loc[category_rows["word"].eq(row.w)]
        append_evidence_examples(
            selected,
            choose_rows(
                target_candidates,
                args.evidence_examples_per_ambiguous_word,
                rng,
            ),
            category,
            "target_category",
            f"ambiguous_word:{row.w}:num_categories={int(row.num_categories)}",
        )

        contrast_candidates = df.loc[
            df["word"].eq(row.w) & df["viterbi_preterminal"].ne(category)
        ]
        append_evidence_examples(
            selected,
            choose_diverse_category_rows(
                contrast_candidates,
                args.evidence_contrast_examples_per_ambiguous_word,
                rng,
            ),
            category,
            "cross_category_contrast",
            f"ambiguous_word_contrast:{row.w}",
        )

    category_scores = word_scores.loc[word_scores["c"].eq(category)]
    rare_words = (
        category_scores.loc[
            category_scores["word_total_count"].le(args.evidence_rare_max_count)
        ]
        .sort_values(
            ["word_total_count", "count", "weighted_log_odds", "w"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .head(args.evidence_rare_examples)
    )
    for row in rare_words.itertuples(index=False):
        candidates = category_rows.loc[category_rows["word"].eq(row.w)]
        append_evidence_examples(
            selected,
            choose_rows(candidates, 1, rng),
            category,
            "target_category",
            f"rare_word:corpus_count={int(row.word_total_count)}",
        )

    random_candidates = category_rows.drop_duplicates(["word", "sentence"])
    append_evidence_examples(
        selected,
        choose_rows(random_candidates, args.evidence_random_examples, rng),
        category,
        "target_category",
        "random_token",
    )

    if not selected:
        return pd.DataFrame(columns=EVIDENCE_EXAMPLE_COLUMNS), ambiguity_profiles

    examples = pd.concat(selected, ignore_index=True)
    key_columns = [
        "target_category",
        "assigned_category",
        "example_role",
        "w",
        "word_pos",
        "pos",
        "tag",
        "sent_id",
        "word_index",
        "previous_word",
        "next_word",
        "sentence",
    ]
    examples = (
        examples.groupby(key_columns, dropna=False, sort=False)["evidence_reasons"]
        .agg(lambda values: ";".join(dict.fromkeys(values)))
        .reset_index()
    )

    score_lookup = word_scores[
        [
            "c",
            "w",
            "count",
            "word_total_count",
            "p_word_given_category",
            "p_category_given_word",
            "weighted_log_odds",
        ]
    ].rename(
        columns={
            "c": "assigned_category",
            "count": "category_word_count",
            "word_total_count": "corpus_word_count",
        }
    )
    examples = examples.merge(score_lookup, on=["assigned_category", "w"], how="left")
    examples = examples.merge(
        word_ambiguity[["word", "num_categories"]].rename(
            columns={"word": "w", "num_categories": "num_categories_for_word"}
        ),
        on="w",
        how="left",
    )
    examples = (
        examples[EVIDENCE_EXAMPLE_COLUMNS]
        .sort_values(
            ["example_role", "evidence_reasons", "w", "sent_id", "word_index"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return examples, ambiguity_profiles


def evidence_example_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = dataframe_records(frame)
    for record in records:
        reasons = record.get("evidence_reasons")
        record["evidence_reasons"] = reasons.split(";") if reasons else []
    return records


def build_standardized_evidence_packet(
    *,
    df: pd.DataFrame,
    category: int,
    summary: pd.DataFrame,
    word_scores: pd.DataFrame,
    word_ambiguity: pd.DataFrame,
    word_pos_category: pd.DataFrame,
    category_pos_distribution: pd.DataFrame,
    previous_words: pd.DataFrame,
    next_words: pd.DataFrame,
    frames: pd.DataFrame,
    positions: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build one self-contained, standardized packet for qualitative coding."""
    frequent, diagnostic = build_evidence_word_rankings(
        word_scores=word_scores,
        category=category,
        frequent_n=args.evidence_frequent_words,
        diagnostic_n=args.evidence_diagnostic_words,
        min_diagnostic_count=args.min_diagnostic_count,
    )
    examples, ambiguity_profiles = build_evidence_examples(
        df=df,
        word_scores=word_scores,
        word_ambiguity=word_ambiguity,
        category=category,
        frequent=frequent,
        diagnostic=diagnostic,
        args=args,
    )

    category_rows = df.loc[df["viterbi_preterminal"].eq(category)]
    tag_distribution = (
        category_rows.assign(
            spacy_tag=category_rows["spacy_tag"].replace("", "<EMPTY>")
        )
        .groupby("spacy_tag", observed=True)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"spacy_tag": "tag"})
    )
    if not tag_distribution.empty:
        tag_distribution["proportion"] = tag_distribution["count"] / len(category_rows)
        tag_distribution = tag_distribution.sort_values(
            ["count", "tag"], ascending=[False, True], kind="stable"
        )

    top_word_pos_units = (
        word_pos_category.loc[word_pos_category["c"].eq(category)]
        .sort_values(
            ["count", "p_category_given_word_pos", "word_pos"],
            ascending=[False, False, True],
            kind="stable",
        )
        .head(args.evidence_top_word_pos_units)
    )

    target_examples = examples.loc[examples["example_role"].eq("target_category")]
    contrast_examples = examples.loc[
        examples["example_role"].eq("cross_category_contrast")
    ]

    packet = {
        "schema_version": "1.0",
        "category_id": category,
        "summary": dataframe_records(summary.loc[summary["c"].eq(category)])[0],
        "sampling_config": {
            "frequent_word_types": args.evidence_frequent_words,
            "diagnostic_word_types": args.evidence_diagnostic_words,
            "examples_per_frequent_or_diagnostic_word": args.evidence_examples_per_word,
            "ambiguous_word_types": args.evidence_ambiguous_words,
            "examples_per_ambiguous_word": args.evidence_examples_per_ambiguous_word,
            "contrast_examples_per_ambiguous_word": (
                args.evidence_contrast_examples_per_ambiguous_word
            ),
            "rare_examples": args.evidence_rare_examples,
            "rare_max_corpus_count": args.evidence_rare_max_count,
            "random_examples": args.evidence_random_examples,
            "top_contexts": args.evidence_top_contexts,
            "top_frames": args.evidence_top_frames,
            "random_seed": args.random_seed + 10_000 + category,
        },
        "lexical_rankings": {
            "frequent_words": dataframe_records(frequent),
            "diagnostic_words": dataframe_records(diagnostic),
            "top_context_sensitive_word_pos_units": dataframe_records(
                top_word_pos_units.drop(columns="c", errors="ignore")
            ),
        },
        "distributions": {
            "pos": dataframe_records(
                category_pos_distribution.loc[
                    category_pos_distribution["c"].eq(category)
                ].drop(columns="c", errors="ignore")
            ),
            "fine_grained_spacy_tags": dataframe_records(tag_distribution),
            "previous_words": dataframe_records(
                previous_words.loc[previous_words["c"].eq(category)]
                .head(args.evidence_top_contexts)
                .drop(columns="c", errors="ignore")
            ),
            "next_words": dataframe_records(
                next_words.loc[next_words["c"].eq(category)]
                .head(args.evidence_top_contexts)
                .drop(columns="c", errors="ignore")
            ),
            "immediate_frames": dataframe_records(
                frames.loc[frames["c"].eq(category)]
                .head(args.evidence_top_frames)
                .drop(columns="c", errors="ignore")
            ),
            "normalized_sentence_position": dataframe_records(
                positions.loc[positions["c"].eq(category)].drop(
                    columns="c", errors="ignore"
                )
            ),
        },
        "ambiguous_word_profiles": ambiguity_profiles,
        "target_category_examples": evidence_example_records(target_examples),
        "cross_category_contrast_examples": evidence_example_records(contrast_examples),
        "notes": {
            "target_category_examples": (
                "All examples in this section were assigned to category_id. Exact duplicate "
                "tokens selected by multiple criteria are merged, and evidence_reasons records "
                "every selection criterion."
            ),
            "cross_category_contrast_examples": (
                "These are explicitly marked comparison examples for ambiguous words that also "
                "occur in other induced categories; do not treat them as members of category_id."
            ),
            "causal_caution": (
                "The packet describes observed lexical and contextual patterns. Frequency, "
                "context, semantics, or visual grounding should be treated as causal drivers only "
                "when supported by comparisons across runs or model conditions."
            ),
        },
    }
    return packet, examples
