from typing import Sequence

import numpy as np
import pandas as pd


def build_word_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        df.groupby(["viterbi_preterminal", "word"], observed=True)
        .agg(count=("word", "size"), n_sentences=("sent_id", "nunique"))
        .reset_index()
        .rename(columns={"viterbi_preterminal": "c", "word": "w"})
    )
    category_totals = df.groupby("viterbi_preterminal").size()
    word_totals = df.groupby("word").size()

    distribution["p_word_given_category"] = distribution["count"] / distribution[
        "c"
    ].map(category_totals)
    distribution["p_category_given_word"] = distribution["count"] / distribution[
        "w"
    ].map(word_totals)

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

    token_counts = df.groupby("word").size().rename("token_count").reset_index()

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
        .drop_duplicates("word")[["word", "spacy_pos"]]
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


def build_word_pos_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Build P(word.POS | category) and P(category | word.POS)."""
    distribution = (
        df.groupby(
            ["viterbi_preterminal", "word_pos", "word", "spacy_pos"],
            observed=True,
        )
        .agg(count=("word_pos", "size"), n_sentences=("sent_id", "nunique"))
        .reset_index()
        .rename(
            columns={
                "viterbi_preterminal": "c",
                "word_pos": "word_pos",
                "word": "w",
                "spacy_pos": "pos",
            }
        )
    )
    category_totals = df.groupby("viterbi_preterminal").size()
    unit_totals = df.groupby("word_pos").size()
    distribution["p_word_pos_given_category"] = distribution["count"] / distribution[
        "c"
    ].map(category_totals)
    distribution["p_category_given_word_pos"] = distribution["count"] / distribution[
        "word_pos"
    ].map(unit_totals)
    return distribution[
        [
            "c",
            "word_pos",
            "w",
            "pos",
            "count",
            "p_word_pos_given_category",
            "p_category_given_word_pos",
            "n_sentences",
        ]
    ].sort_values(
        ["c", "p_word_pos_given_category", "word_pos"],
        ascending=[True, False, True],
    )


def build_category_summary(
    df: pd.DataFrame,
    word_category: pd.DataFrame,
    category_ids: Sequence[int],
) -> pd.DataFrame:
    total_tokens = len(df)
    token_count = (
        df.groupby("viterbi_preterminal").size().reindex(category_ids, fill_value=0)
    )
    word_type_count = (
        df.groupby("viterbi_preterminal")["word"]
        .nunique()
        .reindex(category_ids, fill_value=0)
    )
    sentence_count = (
        df.groupby("viterbi_preterminal")["sent_id"]
        .nunique()
        .reindex(category_ids, fill_value=0)
    )
    median_position = (
        df.groupby("viterbi_preterminal")["normalized_sentence_position"]
        .median()
        .reindex(category_ids)
    )

    sentence_lengths = df[
        ["viterbi_preterminal", "sent_id", "sent_len"]
    ].drop_duplicates(["viterbi_preterminal", "sent_id"])
    mean_sentence_length = (
        sentence_lengths.groupby("viterbi_preterminal")["sent_len"]
        .mean()
        .reindex(category_ids)
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
            "token_share": (
                token_count.to_numpy() / total_tokens if total_tokens else 0.0
            ),
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
