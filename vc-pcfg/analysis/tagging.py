from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc

from .constants.pos import POS_ORDER


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
    result["word_pos"] = (
        result["word"].astype(str) + "." + result["spacy_pos"].astype(str)
    )
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
    dominant["dominant_pos_share"] = (
        dominant["dominant_pos_count"] / dominant["total_count"]
    )
    return dominant[
        [
            "word",
            "dominant_pos",
            "dominant_pos_count",
            "total_count",
            "dominant_pos_share",
        ]
    ]


def build_word_pos_unit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize context-sensitive surface-word/POS lexical units."""
    summary = (
        df.groupby(["word_pos", "word", "spacy_pos"], observed=True)
        .agg(token_count=("word_pos", "size"), sentence_count=("sent_id", "nunique"))
        .reset_index()
        .rename(columns={"spacy_pos": "pos"})
    )
    pos_rank = {pos: index for index, pos in enumerate(POS_ORDER)}
    summary["_pos_rank"] = summary["pos"].map(pos_rank).fillna(len(POS_ORDER))
    summary = summary.sort_values(
        ["_pos_rank", "pos", "token_count", "word", "word_pos"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    return summary.drop(columns="_pos_rank").reset_index(drop=True)


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
    matrix = matrix.reindex(
        index=category_ids, columns=pos_columns, fill_value=0.0
    ).fillna(0.0)
    matrix.index.name = "c"
    return (
        counts.sort_values(["c", "p_pos_given_category"], ascending=[True, False]),
        matrix,
    )
