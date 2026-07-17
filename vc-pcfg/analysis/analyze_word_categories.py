"""Analyze induced preterminal categories exported one token per row.

This script creates:

* a simple integrity audit
* a corpus-level category summary
* complete word/category conditional-probability tables
* frequent-word and weighted-log-odds diagnostic rankings
* immediate-context distributions
* representative examples
* LLM-ready [word, sentence] tables
* a spaCy POS-tagged category-by-word matrix
* category-overlap matrices and heatmaps

POS tagging is performed in sentence context while preserving the exact exported
word boundaries: one spaCy ``Doc`` is created from the token sequence for each ``sent_id``.
The dominant observed spaCy POS of each word type is used only to order columns in the category-by-word matrix.

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
    uv pip install numpy pandas matplotlib spacy click
    uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

    cd vc-pcfg

    python -m analysis.analyze_word_categories \
        --input analysis/outputs/s91-e5.csv \
        --output-dir analysis/outputs/category_analysis/s91-e5 \
        --spacy-model en_core_web_sm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc

REQUIRED_COLUMNS = [
    "sent_id",
    "sent_len",
    "word_index",
    "word_id",
    "word",
    "viterbi_preterminal",
    "left_context",
    "right_context",
    "sentence",
    "num_preterminal_assignments",
    "preterminal_matches_length",
]

OUTPUT_EXAMPLE_COLUMNS = [
    "c",
    "w",
    "word_index",
    "previous_word",
    "next_word",
    "sentence",
    "selection_reason",
]

POS_ORDER = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "SPACE",
]

# ---------------------------------------------------------------------------
# CLI and I/O
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze induced preterminal categories from a token-level CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV or Parquet file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for analysis outputs.")
    parser.add_argument(
        "--num-categories",
        type=int,
        default=60,
        help="Expected categories, assumed to be numbered 0 through N-1.",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name or path.")
    parser.add_argument("--spacy-batch-size", type=int, default=256)
    parser.add_argument(
        "--spacy-processes",
        type=int,
        default=1,
        help="Processes passed to spaCy nlp.pipe. Use cautiously on clusters.",
    )
    parser.add_argument("--top-n-words", type=int, default=25, help="Rows in each top-word ranking.")
    parser.add_argument(
        "--min-diagnostic-count",
        type=int,
        default=10,
        help="Minimum corpus frequency for a word to enter diagnostic rankings.",
    )
    parser.add_argument(
        "--log-odds-prior-strength",
        type=float,
        default=1000.0,
        help="Total mass of the informative Dirichlet prior for weighted log odds.",
    )
    parser.add_argument("--top-contexts", type=int, default=25)
    parser.add_argument("--position-bins", type=int, default=10)
    parser.add_argument(
        "--examples-per-word",
        type=int,
        default=2,
        help="Representative sentences sampled for each frequent/diagnostic word.",
    )
    parser.add_argument("--random-examples", type=int, default=20)
    parser.add_argument("--top-frames-for-examples", type=int, default=5)
    parser.add_argument("--examples-per-frame", type=int, default=3)
    parser.add_argument(
        "--llm-input-max-rows",
        type=int,
        default=100,
        help="Maximum rows in each LLM-ready [word, sentence] table.",
    )
    parser.add_argument(
        "--top-k-overlap",
        type=int,
        default=25,
        help="K for |TopK(c1) intersection TopK(c2)| / K.",
    )
    parser.add_argument(
        "--matrix-heatmap-words",
        type=int,
        default=100,
        help="Most frequent words shown in the readable category-by-word heatmap. The CSV contains all words.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write all tables but skip PNG plots and heatmaps.",
    )
    parser.add_argument(
        "--strict-integrity",
        action="store_true",
        help="Stop after writing the audit if sentence lengths, assignments, or token positions are inconsistent.",
    )
    return parser


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Input must be a .csv, .parquet, or .pq file.")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    return df


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    unknown = normalized.notna() & ~normalized.isin(mapping)
    if unknown.any():
        values = sorted(normalized.loc[unknown].dropna().unique().tolist())
        raise ValueError(f"Unrecognized boolean values in preterminal_matches_length: {values}")
    return normalized.map(mapping).fillna(False).astype(bool)


def prepare_input(df: pd.DataFrame, bos: str = "<BOS>", eos: str = "<EOS>") -> pd.DataFrame:
    result = df.copy()

    integer_columns = [
        "sent_id",
        "sent_len",
        "word_index",
        "word_id",
        "viterbi_preterminal",
        "num_preterminal_assignments",
    ]
    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")

    result["preterminal_matches_length"] = normalize_boolean(result["preterminal_matches_length"])
    result["word"] = result["word"].astype("string")
    result["sentence"] = result["sentence"].astype("string")

    result = result.sort_values(["sent_id", "word_index"], kind="stable").reset_index(drop=True)
    result["previous_word"] = result.groupby("sent_id", sort=False)["word"].shift(1).fillna(bos)
    result["next_word"] = result.groupby("sent_id", sort=False)["word"].shift(-1).fillna(eos)

    denominator = (result["sent_len"] - 1).clip(lower=1)
    result["normalized_sentence_position"] = result["word_index"] / denominator
    result.loc[result["sent_len"] <= 1, "normalized_sentence_position"] = 0.0

    return result

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Integrity audit
# ---------------------------------------------------------------------------


def run_integrity_audit(
    df: pd.DataFrame,
    output_dir: Path,
    num_categories: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    grouped = df.groupby("sent_id", sort=False)

    sentence_audit = grouped.agg(
        sent_len=("sent_len", "first"),
        distinct_sent_len=("sent_len", "nunique"),
        row_count=("word_index", "size"),
        unique_word_indices=("word_index", "nunique"),
        min_word_index=("word_index", "min"),
        max_word_index=("word_index", "max"),
        num_preterminal_assignments=("num_preterminal_assignments", "first"),
        distinct_assignment_counts=("num_preterminal_assignments", "nunique"),
        preterminal_matches_length=("preterminal_matches_length", "all"),
        distinct_sentences=("sentence", "nunique"),
    ).reset_index()

    sentence_audit["expected_index_count"] = sentence_audit["sent_len"]
    sentence_audit["row_count_matches_length"] = sentence_audit["row_count"] == sentence_audit["sent_len"]
    sentence_audit["indices_match_length"] = (
        (sentence_audit["unique_word_indices"] == sentence_audit["sent_len"])
        & (sentence_audit["min_word_index"] == 0)
        & (sentence_audit["max_word_index"] == sentence_audit["sent_len"] - 1)
    )
    sentence_audit["assignments_match_length"] = (
        sentence_audit["num_preterminal_assignments"] == sentence_audit["sent_len"]
    )

    def issue_string(row: pd.Series) -> str:
        issues: list[str] = []
        if row["distinct_sent_len"] != 1:
            issues.append("inconsistent_sent_len")
        if not row["row_count_matches_length"]:
            issues.append("row_count_mismatch")
        if not row["indices_match_length"]:
            issues.append("word_index_mismatch")
        if row["distinct_assignment_counts"] != 1:
            issues.append("inconsistent_assignment_count")
        if not row["assignments_match_length"]:
            issues.append("assignment_count_mismatch")
        if not row["preterminal_matches_length"]:
            issues.append("preterminal_matches_length_false")
        if row["distinct_sentences"] != 1:
            issues.append("inconsistent_sentence_text")
        return ";".join(issues)

    sentence_audit["issue"] = sentence_audit.apply(issue_string, axis=1)
    sentence_issues = sentence_audit.loc[sentence_audit["issue"] != ""].copy()

    duplicate_pairs = int(df.duplicated(["sent_id", "word_index"]).sum())
    critical_nulls = int(
        df[["sent_id", "sent_len", "word_index", "word", "viterbi_preterminal", "sentence"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    out_of_range = int(
        ((df["viterbi_preterminal"] < 0) | (df["viterbi_preterminal"] >= num_categories)).sum()
    )

    metrics = [
        ("total_rows", len(df)),
        ("unique_sentences", df["sent_id"].nunique()),
        ("unique_words", df["word"].nunique()),
        ("observed_categories", df["viterbi_preterminal"].nunique()),
        ("duplicate_sent_id_word_index_rows", duplicate_pairs),
        ("rows_with_critical_nulls", critical_nulls),
        ("rows_with_category_outside_expected_range", out_of_range),
        ("sentences_with_integrity_issue", len(sentence_issues)),
        (
            "sentences_with_preterminal_matches_length_false",
            int((~sentence_audit["preterminal_matches_length"]).sum()),
        ),
    ]
    audit = pd.DataFrame(metrics, columns=["metric", "value"])

    write_csv(audit, output_dir / "integrity" / "integrity_audit.csv")
    write_csv(sentence_issues, output_dir / "integrity" / "sentence_integrity_issues.csv")

    has_problem = any(
        value > 0
        for metric, value in metrics
        if metric
        in {
            "duplicate_sent_id_word_index_rows",
            "rows_with_critical_nulls",
            "rows_with_category_outside_expected_range",
            "sentences_with_integrity_issue",
        }
    )
    return audit, sentence_issues, has_problem


# ---------------------------------------------------------------------------
# spaCy POS tagging
# ---------------------------------------------------------------------------


def load_spacy_model(model_name: str):
    try:
        # Keep the tagger and attribute ruler, which jointly provide POS tags in
        # common English spaCy pipelines. Parsing, NER, and lemmatization are not
        # needed for this analysis.
        return spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
    except OSError as exc:
        raise RuntimeError(
            f"Could not load spaCy model '{model_name}'. Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc


def add_spacy_pos(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    n_process: int,
) -> pd.DataFrame:
    nlp = load_spacy_model(model_name)
    result = df.copy()
    pos_values = np.empty(len(result), dtype=object)
    tag_values = np.empty(len(result), dtype=object)

    records: list[tuple[np.ndarray, list[str]]] = []
    for _, group in result.groupby("sent_id", sort=False):
        row_indices = group.index.to_numpy(dtype=np.int64)
        words = group["word"].astype(str).tolist()
        records.append((row_indices, words))

    docs: Iterable[Doc] = (Doc(nlp.vocab, words=words) for _, words in records)
    processed_docs = nlp.pipe(docs, batch_size=batch_size, n_process=n_process)

    for (row_indices, words), doc in zip(records, processed_docs, strict=True):
        if len(doc) != len(words):
            raise RuntimeError(
                "spaCy changed token boundaries unexpectedly despite pretokenized Docs: "
                f"expected {len(words)}, got {len(doc)}"
            )
        for row_index, token in zip(row_indices, doc, strict=True):
            pos_values[row_index] = token.pos_ or "X"
            tag_values[row_index] = token.tag_ or ""

    result["spacy_pos"] = pos_values
    result["spacy_tag"] = tag_values
    return result


def build_word_pos_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["word", "spacy_pos"], observed=True)
        .size()
        .rename("pos_count")
        .reset_index()
    )
    totals = df.groupby("word", observed=True).size().rename("total_count")
    counts = counts.merge(totals, on="word", how="left")
    counts = counts.sort_values(
        ["word", "pos_count", "spacy_pos"],
        ascending=[True, False, True],
        kind="stable",
    )
    dominant = counts.drop_duplicates("word", keep="first").copy()
    dominant = dominant.rename(
        columns={"spacy_pos": "dominant_pos", "pos_count": "dominant_pos_count"}
    )
    dominant["dominant_pos_share"] = dominant["dominant_pos_count"] / dominant["total_count"]
    return dominant[
        ["word", "dominant_pos", "dominant_pos_count", "total_count", "dominant_pos_share"]
    ]


def build_category_pos_distribution(
    df: pd.DataFrame,
    category_ids: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        df.groupby(["viterbi_preterminal", "spacy_pos"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"viterbi_preterminal": "c", "spacy_pos": "pos"})
    )
    category_totals = df.groupby("viterbi_preterminal").size()
    pos_totals = df.groupby("spacy_pos").size()
    counts["p_pos_given_category"] = counts["count"] / counts["c"].map(category_totals)
    counts["p_category_given_pos"] = counts["count"] / counts["pos"].map(pos_totals)

    present_pos = [pos for pos in POS_ORDER if pos in set(df["spacy_pos"])]
    extra_pos = sorted(set(df["spacy_pos"]) - set(present_pos))
    pos_columns = present_pos + extra_pos
    matrix = counts.pivot(index="c", columns="pos", values="p_pos_given_category")
    matrix = matrix.reindex(index=category_ids, columns=pos_columns, fill_value=0.0).fillna(0.0)
    matrix.index.name = "c"
    return counts.sort_values(["c", "p_pos_given_category"], ascending=[True, False]), matrix


# ---------------------------------------------------------------------------
# Word/category distributions and category summary
# ---------------------------------------------------------------------------


def build_word_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        df.groupby(["viterbi_preterminal", "word"], observed=True)
        .agg(count=("word", "size"), n_sentences=("sent_id", "nunique"))
        .reset_index()
        .rename(columns={"viterbi_preterminal": "c", "word": "w"})
    )
    category_totals = df.groupby("viterbi_preterminal").size()
    word_totals = df.groupby("word").size()

    distribution["p_word_given_category"] = (
        distribution["count"] / distribution["c"].map(category_totals)
    )
    distribution["p_category_given_word"] = distribution["count"] / distribution["w"].map(
        word_totals
    )

    return distribution[
        [
            "c",
            "w",
            "count",
            "p_word_given_category",
            "p_category_given_word",
            "n_sentences",
        ]
    ].sort_values(["c", "p_word_given_category", "w"], ascending=[True, False, True])

def build_word_category_ambiguity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Number of categories observed for each word.
    """

    ambiguity = (
        df.groupby("word")["viterbi_preterminal"]
        .nunique()
        .rename("num_categories")
        .reset_index()
    )

    token_counts = (
        df.groupby("word")
        .size()
        .rename("token_count")
        .reset_index()
    )

    ambiguity = ambiguity.merge(
        token_counts,
        on="word",
        how="left",
    )

    return ambiguity.sort_values(
        ["num_categories", "token_count", "word"],
        ascending=[False, False, True],
    )


def build_pos_ambiguity_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average number of categories per word, grouped by dominant spaCy POS.
    (Returns POS -> mean number of categories.)
    """

    ambiguity = (
        df.groupby("word")["viterbi_preterminal"]
        .nunique()
        .rename("num_categories")
        .reset_index()
    )

    dominant_pos = (
        df.groupby(["word", "spacy_pos"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["word", "count"],
            ascending=[True, False],
            kind="stable",
        )
        .drop_duplicates("word")
        [["word", "spacy_pos"]]
    )

    ambiguity = ambiguity.merge(
        dominant_pos,
        on="word",
        how="left",
    )

    return (
        ambiguity.groupby("spacy_pos")["num_categories"]
        .agg(
            word_count="count",
            mean_categories="mean",
            median_categories="median",
            std_categories="std",
            max_categories="max",
        )
        .reset_index()
        .sort_values(
            "mean_categories",
            ascending=False,
        )
    )

def build_category_summary(
    df: pd.DataFrame,
    word_category: pd.DataFrame,
    category_ids: Sequence[int],
) -> pd.DataFrame:
    total_tokens = len(df)
    token_count = df.groupby("viterbi_preterminal").size().reindex(category_ids, fill_value=0)
    word_type_count = (
        df.groupby("viterbi_preterminal")["word"].nunique().reindex(category_ids, fill_value=0)
    )
    sentence_count = (
        df.groupby("viterbi_preterminal")["sent_id"].nunique().reindex(category_ids, fill_value=0)
    )
    median_position = (
        df.groupby("viterbi_preterminal")["normalized_sentence_position"]
        .median()
        .reindex(category_ids)
    )

    sentence_lengths = df[["viterbi_preterminal", "sent_id", "sent_len"]].drop_duplicates(
        ["viterbi_preterminal", "sent_id"]
    )
    mean_sentence_length = (
        sentence_lengths.groupby("viterbi_preterminal")["sent_len"].mean().reindex(category_ids)
    )

    entropy = (
        word_category.assign(
            entropy_piece=lambda x: -x["p_word_given_category"]
            * np.log(x["p_word_given_category"])
        )
        .groupby("c")["entropy_piece"]
        .sum()
        .reindex(category_ids, fill_value=0.0)
    )

    top_10_coverage = (
        word_category.sort_values(["c", "count"], ascending=[True, False])
        .groupby("c", sort=False)
        .head(10)
        .groupby("c")["count"]
        .sum()
        .div(token_count.replace(0, np.nan))
        .reindex(category_ids)
        .fillna(0.0)
    )

    summary = pd.DataFrame(
        {
            "c": category_ids,
            "token_count": token_count.to_numpy(),
            "token_share": token_count.to_numpy() / total_tokens if total_tokens else 0.0,
            "word_type_count": word_type_count.to_numpy(),
            "sentence_count": sentence_count.to_numpy(),
            "type_token_ratio": np.divide(
                word_type_count.to_numpy(dtype=float),
                token_count.to_numpy(dtype=float),
                out=np.zeros(len(category_ids), dtype=float),
                where=token_count.to_numpy() != 0,
            ),
            "word_entropy": entropy.to_numpy(),
            "top_10_coverage": top_10_coverage.to_numpy(),
            "median_sentence_position": median_position.to_numpy(),
            "mean_sentence_length": mean_sentence_length.to_numpy(),
        }
    )
    return summary


def add_weighted_log_odds(
    word_category: pd.DataFrame,
    df: pd.DataFrame,
    prior_strength: float,
) -> pd.DataFrame:
    if prior_strength <= 0:
        raise ValueError("--log-odds-prior-strength must be greater than zero.")

    scored = word_category.copy()
    total_tokens = float(len(df))
    category_totals = df.groupby("viterbi_preterminal").size().astype(float)
    word_totals = df.groupby("word").size().astype(float)

    scored["category_total_count"] = scored["c"].map(category_totals)
    scored["word_total_count"] = scored["w"].map(word_totals)
    scored["outside_word_count"] = scored["word_total_count"] - scored["count"]
    scored["outside_total_count"] = total_tokens - scored["category_total_count"]

    corpus_word_probability = scored["word_total_count"] / total_tokens
    alpha_word = prior_strength * corpus_word_probability
    alpha_not_word = prior_strength - alpha_word

    inside_nonword = scored["category_total_count"] - scored["count"]
    outside_nonword = scored["outside_total_count"] - scored["outside_word_count"]

    inside_odds = (scored["count"] + alpha_word) / (inside_nonword + alpha_not_word)
    outside_odds = (scored["outside_word_count"] + alpha_word) / (
        outside_nonword + alpha_not_word
    )

    delta = np.log(inside_odds) - np.log(outside_odds)
    variance = 1.0 / (scored["count"] + alpha_word) + 1.0 / (
        scored["outside_word_count"] + alpha_word
    )
    scored["weighted_log_odds"] = delta / np.sqrt(variance)

    return scored


def build_word_rankings(
    word_scores: pd.DataFrame,
    top_n: int,
    min_diagnostic_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequent = (
        word_scores.sort_values(
            ["c", "count", "p_category_given_word", "w"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("c", sort=False)
        .head(top_n)
        .copy()
    )
    frequent["rank"] = frequent.groupby("c").cumcount() + 1
    frequent = frequent[
        [
            "c",
            "rank",
            "w",
            "count",
            "p_word_given_category",
            "p_category_given_word",
            "n_sentences",
        ]
    ]

    eligible = word_scores.loc[
        (word_scores["word_total_count"] >= min_diagnostic_count)
        & (word_scores["weighted_log_odds"] > 0)
    ]
    diagnostic = (
        eligible.sort_values(
            ["c", "weighted_log_odds", "count", "w"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("c", sort=False)
        .head(top_n)
        .copy()
    )
    diagnostic["rank"] = diagnostic.groupby("c").cumcount() + 1
    diagnostic = diagnostic[
        [
            "c",
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


# ---------------------------------------------------------------------------
# Context analysis
# ---------------------------------------------------------------------------


def build_ranked_context_table(
    df: pd.DataFrame,
    group_columns: list[str],
    top_n: int,
) -> pd.DataFrame:
    category_totals = df.groupby("viterbi_preterminal").size()
    table = (
        df.groupby(["viterbi_preterminal", *group_columns], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"viterbi_preterminal": "c"})
    )
    table["proportion"] = table["count"] / table["c"].map(category_totals)
    table = table.sort_values(
        ["c", "count", *group_columns],
        ascending=[True, False, *([True] * len(group_columns))],
        kind="stable",
    )
    table = table.groupby("c", sort=False).head(top_n).copy()
    table["rank"] = table.groupby("c").cumcount() + 1
    return table[["c", "rank", *group_columns, "count", "proportion"]]


def build_position_distribution(
    df: pd.DataFrame,
    category_ids: Sequence[int],
    n_bins: int,
) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError("--position-bins must be greater than zero.")

    positions = df["normalized_sentence_position"].clip(0.0, 1.0).to_numpy()
    bin_index = np.minimum((positions * n_bins).astype(int), n_bins - 1)
    working = pd.DataFrame(
        {
            "c": df["viterbi_preterminal"].to_numpy(),
            "position_bin": bin_index,
        }
    )
    counts = working.groupby(["c", "position_bin"]).size()
    full_index = pd.MultiIndex.from_product(
        [category_ids, range(n_bins)], names=["c", "position_bin"]
    )
    counts = counts.reindex(full_index, fill_value=0).rename("count").reset_index()
    category_totals = counts.groupby("c")["count"].transform("sum")
    counts["proportion"] = np.divide(
        counts["count"],
        category_totals,
        out=np.zeros(len(counts), dtype=float),
        where=category_totals != 0,
    )
    counts["bin_left"] = counts["position_bin"] / n_bins
    counts["bin_right"] = (counts["position_bin"] + 1) / n_bins
    return counts[["c", "position_bin", "bin_left", "bin_right", "count", "proportion"]]


# ---------------------------------------------------------------------------
# Representative examples
# ---------------------------------------------------------------------------


def choose_rows(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    n = min(n, len(frame))
    indices = rng.choice(frame.index.to_numpy(), size=n, replace=False)
    return frame.loc[indices].copy()


def append_examples(
    selected: list[pd.DataFrame],
    rows: pd.DataFrame,
    reason: str,
) -> None:
    if rows.empty:
        return
    chunk = rows[
        ["viterbi_preterminal", "word", "word_index", "previous_word", "next_word", "sentence"]
    ].copy()
    chunk = chunk.rename(columns={"viterbi_preterminal": "c", "word": "w"})
    chunk["selection_reason"] = reason
    selected.append(chunk[OUTPUT_EXAMPLE_COLUMNS])


def combine_example_reasons(examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame(columns=OUTPUT_EXAMPLE_COLUMNS)

    key_columns = [column for column in OUTPUT_EXAMPLE_COLUMNS if column != "selection_reason"]
    combined = (
        examples.groupby(key_columns, dropna=False, sort=False)["selection_reason"]
        .agg(lambda values: ";".join(dict.fromkeys(values)))
        .reset_index()
    )
    return combined[OUTPUT_EXAMPLE_COLUMNS]


def build_representative_examples(
    df: pd.DataFrame,
    category: int,
    frequent_words: pd.DataFrame,
    diagnostic_words: pd.DataFrame,
    frame_rankings: pd.DataFrame,
    examples_per_word: int,
    random_examples: int,
    top_frames_for_examples: int,
    examples_per_frame: int,
    random_seed: int,
) -> pd.DataFrame:
    category_rows = df.loc[df["viterbi_preterminal"] == category]
    if category_rows.empty:
        return pd.DataFrame(columns=OUTPUT_EXAMPLE_COLUMNS)

    rng = np.random.default_rng(random_seed + category)
    selected: list[pd.DataFrame] = []

    for word in frequent_words.loc[frequent_words["c"] == category, "w"]:
        candidates = category_rows.loc[category_rows["word"] == word]
        append_examples(
            selected,
            choose_rows(candidates, examples_per_word, rng),
            f"frequent_word:{word}",
        )

    for word in diagnostic_words.loc[diagnostic_words["c"] == category, "w"]:
        candidates = category_rows.loc[category_rows["word"] == word]
        append_examples(
            selected,
            choose_rows(candidates, examples_per_word, rng),
            f"diagnostic_word:{word}",
        )

    category_frames = frame_rankings.loc[frame_rankings["c"] == category].head(
        top_frames_for_examples
    )
    for row in category_frames.itertuples(index=False):
        candidates = category_rows.loc[
            (category_rows["previous_word"] == row.previous_word)
            & (category_rows["next_word"] == row.next_word)
        ]
        append_examples(
            selected,
            choose_rows(candidates, examples_per_frame, rng),
            f"common_frame:{row.previous_word}|{row.next_word}",
        )

    append_examples(
        selected,
        choose_rows(category_rows, random_examples, rng),
        "random",
    )

    if not selected:
        return pd.DataFrame(columns=OUTPUT_EXAMPLE_COLUMNS)
    return combine_example_reasons(pd.concat(selected, ignore_index=True))


def build_llm_input(examples: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame(columns=["word", "sentence"])
    result = (
        examples[["w", "sentence"]]
        .rename(columns={"w": "word"})
        .drop_duplicates(["word", "sentence"])
    )
    if max_rows > 0:
        result = result.head(max_rows)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Matrices and overlap metrics
# ---------------------------------------------------------------------------


def build_category_word_matrix(
    word_category: pd.DataFrame,
    word_pos_summary: pd.DataFrame,
    category_ids: Sequence[int],
) -> pd.DataFrame:
    pos_rank = {pos: index for index, pos in enumerate(POS_ORDER)}
    ordering = word_pos_summary.copy()
    ordering["_pos_rank"] = ordering["dominant_pos"].map(pos_rank).fillna(len(POS_ORDER))
    ordering = ordering.sort_values(
        ["_pos_rank", "dominant_pos", "word"], kind="stable"
    )
    ordered_words = ordering["word"].tolist()

    matrix = word_category.pivot(index="c", columns="w", values="p_word_given_category")
    matrix = matrix.reindex(index=category_ids, columns=ordered_words, fill_value=0.0).fillna(0.0)
    matrix.index.name = "c"
    return matrix


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    result = np.full((matrix.shape[0], matrix.shape[0]), np.nan, dtype=float)
    valid = norms > 0
    if valid.any():
        normalized = matrix[valid] / norms[valid, None]
        similarities = normalized @ normalized.T
        valid_indices = np.flatnonzero(valid)
        result[np.ix_(valid_indices, valid_indices)] = similarities
    return result


def js_divergence_pair(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    p_mask = p > 0
    q_mask = q > 0
    kl_p = np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask]))
    kl_q = np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask]))
    return float(0.5 * (kl_p + kl_q))


def js_divergence_matrix(matrix: np.ndarray) -> np.ndarray:
    n_categories = matrix.shape[0]
    result = np.full((n_categories, n_categories), np.nan, dtype=float)
    for i in range(n_categories):
        for j in range(i, n_categories):
            value = js_divergence_pair(matrix[i], matrix[j])
            result[i, j] = value
            result[j, i] = value
    return result


def top_k_overlap_matrix(matrix: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("--top-k-overlap must be greater than zero.")

    top_sets: list[set[int] | None] = []
    for row in matrix:
        positive_indices = np.flatnonzero(row > 0)
        if len(positive_indices) == 0:
            top_sets.append(None)
            continue
        ordered = positive_indices[np.argsort(row[positive_indices])[::-1]]
        top_sets.append(set(ordered[:k].tolist()))

    n_categories = matrix.shape[0]
    result = np.full((n_categories, n_categories), np.nan, dtype=float)
    for i in range(n_categories):
        for j in range(i, n_categories):
            if top_sets[i] is None or top_sets[j] is None:
                value = float("nan")
            else:
                value = len(top_sets[i] & top_sets[j]) / k
            result[i, j] = value
            result[j, i] = value
    return result


def labeled_matrix(values: np.ndarray, category_ids: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame(values, index=category_ids, columns=category_ids).rename_axis(
        index="c", columns="other_c"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_top_bar_plot(
    table: pd.DataFrame,
    value_column: str,
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    if table.empty:
        return
    plot_data = table.sort_values(value_column, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(plot_data))))
    ax.barh(plot_data["w"].astype(str), plot_data[value_column])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Word")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rank_distribution_plot(
    values: pd.Series,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    clean = values.dropna().sort_values(ascending=False).reset_index(drop=True)
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(clean) + 1), clean.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Word rank")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_position_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    if table.empty:
        return
    centers = (table["bin_left"] + table["bin_right"]) / 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, table["proportion"], width=(table["bin_right"] - table["bin_left"]) * 0.9)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Normalized sentence position")
    ax.set_ylabel("Token proportion")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_square_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(values, aspect="equal", interpolation="nearest")
    labels = [str(value) for value in matrix.index]
    step = max(1, math.ceil(len(labels) / 30))
    ticks = np.arange(0, len(labels), step)
    ax.set_xticks(ticks, [labels[index] for index in ticks], rotation=90)
    ax.set_yticks(ticks, [labels[index] for index in ticks])
    ax.set_xlabel("Category")
    ax.set_ylabel("Category")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_category_word_heatmap(
    matrix: pd.DataFrame,
    word_totals: pd.Series,
    word_pos_summary: pd.DataFrame,
    max_words: int,
    path: Path,
) -> None:
    if max_words <= 0 or matrix.empty:
        return

    selected = word_totals.sort_values(ascending=False).head(max_words).index
    pos_lookup = word_pos_summary.set_index("word")["dominant_pos"]
    pos_rank = {pos: index for index, pos in enumerate(POS_ORDER)}
    ordering = pd.DataFrame({"word": selected})
    ordering["pos"] = ordering["word"].map(pos_lookup).fillna("X")
    ordering["pos_rank"] = ordering["pos"].map(pos_rank).fillna(len(POS_ORDER))
    ordering = ordering.sort_values(["pos_rank", "pos", "word"], kind="stable")
    words = ordering["word"].tolist()
    subset = matrix.reindex(columns=words)

    fig_width = max(12, min(30, len(words) * 0.18))
    fig, ax = plt.subplots(figsize=(fig_width, 11))
    image = ax.imshow(subset.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(subset.index)), [str(value) for value in subset.index])
    ax.set_xticks(np.arange(len(words)), words, rotation=90, fontsize=6)
    ax.set_xlabel("Word, ordered by dominant spaCy POS")
    ax.set_ylabel("Category")
    ax.set_title(f"P(word | category), {len(words)} most frequent words")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("P(word | category)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_category_pos_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(matrix.index)), [str(value) for value in matrix.index])
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_xlabel("spaCy POS")
    ax.set_ylabel("Category")
    ax.set_title("P(spaCy POS | category)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Proportion")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output organization
# ---------------------------------------------------------------------------


def write_per_category_outputs(
    output_dir: Path,
    category_ids: Sequence[int],
    summary: pd.DataFrame,
    word_category: pd.DataFrame,
    frequent_words: pd.DataFrame,
    diagnostic_words: pd.DataFrame,
    previous_words: pd.DataFrame,
    next_words: pd.DataFrame,
    frames: pd.DataFrame,
    positions: pd.DataFrame,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    combined_llm_inputs: list[pd.DataFrame] = []
    width = max(2, len(str(max(category_ids))) if category_ids else 2)

    for category in category_ids:
        category_dir = output_dir / "categories" / f"category_{category:0{width}d}"
        category_dir.mkdir(parents=True, exist_ok=True)

        category_summary = summary.loc[summary["c"] == category]
        category_word_distribution = word_category.loc[word_category["c"] == category]
        category_frequent = frequent_words.loc[frequent_words["c"] == category]
        category_diagnostic = diagnostic_words.loc[diagnostic_words["c"] == category]
        category_previous = previous_words.loc[previous_words["c"] == category]
        category_next = next_words.loc[next_words["c"] == category]
        category_frames = frames.loc[frames["c"] == category]
        category_positions = positions.loc[positions["c"] == category]

        write_csv(category_summary, category_dir / "summary.csv")
        write_csv(category_word_distribution, category_dir / "word_distribution.csv")
        write_csv(category_frequent, category_dir / "frequent_words.csv")
        write_csv(category_diagnostic, category_dir / "diagnostic_words.csv")
        write_csv(category_previous, category_dir / "previous_words.csv")
        write_csv(category_next, category_dir / "next_words.csv")
        write_csv(category_frames, category_dir / "context_frames.csv")
        write_csv(category_positions, category_dir / "position_distribution.csv")

        examples = build_representative_examples(
            df=df,
            category=category,
            frequent_words=frequent_words,
            diagnostic_words=diagnostic_words,
            frame_rankings=frames,
            examples_per_word=args.examples_per_word,
            random_examples=args.random_examples,
            top_frames_for_examples=args.top_frames_for_examples,
            examples_per_frame=args.examples_per_frame,
            random_seed=args.random_seed,
        )
        write_csv(examples, category_dir / "representative_sentences.csv")

        llm_input = build_llm_input(examples, args.llm_input_max_rows)
        write_csv(llm_input, category_dir / "llm_input.csv")
        if not llm_input.empty:
            combined = llm_input.copy()
            combined.insert(0, "c", category)
            combined_llm_inputs.append(combined)

        if args.skip_plots:
            continue

        plots_dir = category_dir / "plots"
        save_top_bar_plot(
            category_frequent,
            "count",
            plots_dir / "top_frequent_words.png",
            f"Category {category}: most frequent words",
            "Token count",
        )
        save_top_bar_plot(
            category_diagnostic,
            "weighted_log_odds",
            plots_dir / "top_diagnostic_words.png",
            f"Category {category}: diagnostic words",
            "Weighted log-odds z-score",
        )
        save_rank_distribution_plot(
            category_word_distribution["p_word_given_category"],
            plots_dir / "p_word_given_category_distribution.png",
            f"Category {category}: P(word | category)",
            "P(word | category)",
        )
        eligible_diagnostic_distribution = category_word_distribution.loc[
            category_word_distribution["w"].isin(category_diagnostic["w"])
        ]
        save_rank_distribution_plot(
            eligible_diagnostic_distribution["p_category_given_word"],
            plots_dir / "p_category_given_word_distribution.png",
            f"Category {category}: P(category | word), diagnostic words",
            "P(category | word)",
        )
        save_position_plot(
            category_positions,
            plots_dir / "normalized_position_distribution.png",
            f"Category {category}: normalized sentence position",
        )

    if combined_llm_inputs:
        return pd.concat(combined_llm_inputs, ignore_index=True)
    return pd.DataFrame(columns=["c", "word", "sentence"])


def write_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    manifest: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_categories": args.num_categories,
        "spacy_model": args.spacy_model,
        "top_n_words": args.top_n_words,
        "min_diagnostic_count": args.min_diagnostic_count,
        "log_odds_prior_strength": args.log_odds_prior_strength,
        "top_contexts": args.top_contexts,
        "position_bins": args.position_bins,
        "top_k_overlap": args.top_k_overlap,
        "random_seed": args.random_seed,
        "metric_definitions": {
            "word_entropy": "Natural-log entropy of P(word | category), measured in nats.",
            "mean_sentence_length": (
                "Mean length of distinct sentences containing the category; each sentence "
                "is counted once per category."
            ),
            "top_k_overlap": "|TopK(c1) intersection TopK(c2)| / K.",
            "js_divergence": "Jensen-Shannon divergence in bits; 0 means identical distributions.",
            "weighted_log_odds": (
                "Variance-normalized difference between a word's log odds inside one category "
                "and in all remaining categories, with a corpus-informed Dirichlet prior."
            ),
            "word_pos_for_matrix_order": (
                "Dominant context-sensitive spaCy coarse POS among all occurrences of each word type."
            ),
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.num_categories <= 0:
        parser.error("--num-categories must be greater than zero.")
    if args.top_n_words <= 0:
        parser.error("--top-n-words must be greater than zero.")
    if args.min_diagnostic_count <= 0:
        parser.error("--min-diagnostic-count must be greater than zero.")

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

    print(f"Tagging exact exported tokens with spaCy model '{args.spacy_model}' ...", flush=True)
    df = add_spacy_pos(
        df,
        model_name=args.spacy_model,
        batch_size=args.spacy_batch_size,
        n_process=args.spacy_processes,
    )
    word_pos_summary = build_word_pos_summary(df)
    write_csv(word_pos_summary, output_dir / "pos" / "word_pos_summary.csv")

    category_ids = list(range(args.num_categories))

    print("Computing word/category distributions and rankings ...", flush=True)
    word_category = build_word_category_distribution(df)
    write_csv(word_category, output_dir / "word_category_distribution.csv")

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
    previous_words = build_ranked_context_table(df, ["previous_word"], args.top_contexts)
    next_words = build_ranked_context_table(df, ["next_word"], args.top_contexts)
    frames = build_ranked_context_table(
        df, ["previous_word", "next_word"], args.top_contexts
    )
    positions = build_position_distribution(df, category_ids, args.position_bins)
    write_csv(previous_words, output_dir / "context" / "previous_word_rankings.csv")
    write_csv(next_words, output_dir / "context" / "next_word_rankings.csv")
    write_csv(frames, output_dir / "context" / "context_frame_rankings.csv")
    write_csv(positions, output_dir / "context" / "normalized_position_distribution.csv")

    print("Building spaCy POS and category-by-word matrices ...", flush=True)
    category_pos_distribution, category_pos_matrix = build_category_pos_distribution(
        df, category_ids
    )
    write_csv(category_pos_distribution, output_dir / "pos" / "category_pos_distribution.csv")
    write_csv(category_pos_matrix.reset_index(), output_dir / "pos" / "category_pos_matrix.csv")

    category_word_matrix = build_category_word_matrix(
        word_category,
        word_pos_summary,
        category_ids,
    )
    write_csv(
        category_word_matrix.reset_index(),
        output_dir / "matrices" / "category_by_word_p_word_given_category.csv",
    )

    raw_matrix = category_word_matrix.to_numpy(dtype=float)
    cosine = labeled_matrix(cosine_similarity_matrix(raw_matrix), category_ids)
    js = labeled_matrix(js_divergence_matrix(raw_matrix), category_ids)
    top_k = labeled_matrix(top_k_overlap_matrix(raw_matrix, args.top_k_overlap), category_ids)
    write_csv(cosine.reset_index(), output_dir / "matrices" / "cosine_similarity.csv")
    write_csv(js.reset_index(), output_dir / "matrices" / "js_divergence.csv")
    write_csv(
        top_k.reset_index(),
        output_dir / "matrices" / f"top_{args.top_k_overlap}_word_overlap.csv",
    )

    if not args.skip_plots:
        heatmap_dir = output_dir / "matrices" / "heatmaps"
        save_square_heatmap(
            cosine,
            heatmap_dir / "cosine_similarity.png",
            "Category cosine similarity over P(word | category)",
            "Cosine similarity",
        )
        save_square_heatmap(
            js,
            heatmap_dir / "js_divergence.png",
            "Category Jensen-Shannon divergence over P(word | category)",
            "J-S divergence (bits)",
        )
        save_square_heatmap(
            top_k,
            heatmap_dir / f"top_{args.top_k_overlap}_word_overlap.png",
            f"Category top-{args.top_k_overlap} word overlap",
            "Intersection / K",
        )
        word_totals = df.groupby("word").size()
        save_category_word_heatmap(
            category_word_matrix,
            word_totals,
            word_pos_summary,
            args.matrix_heatmap_words,
            heatmap_dir / "category_by_word.png",
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