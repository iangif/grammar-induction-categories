from pathlib import Path
import pandas as pd
from .io import write_csv


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
    sentence_audit["row_count_matches_length"] = (
        sentence_audit["row_count"] == sentence_audit["sent_len"]
    )
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
        df[
            [
                "sent_id",
                "sent_len",
                "word_index",
                "word",
                "viterbi_preterminal",
                "sentence",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )
    out_of_range = int(
        (
            (df["viterbi_preterminal"] < 0)
            | (df["viterbi_preterminal"] >= num_categories)
        ).sum()
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
    write_csv(
        sentence_issues, output_dir / "integrity" / "sentence_integrity_issues.csv"
    )

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
