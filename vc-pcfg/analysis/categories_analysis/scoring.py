import numpy as np
import pandas as pd


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
