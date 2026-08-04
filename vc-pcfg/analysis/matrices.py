from typing import Sequence

import pandas as pd

from .constants.pos import POS_ORDER


def build_category_word_matrix(
    word_category: pd.DataFrame,
    word_pos_summary: pd.DataFrame,
    category_ids: Sequence[int],
) -> pd.DataFrame:
    pos_rank = {pos: index for index, pos in enumerate(POS_ORDER)}
    ordering = word_pos_summary.copy()
    ordering["_pos_rank"] = (
        ordering["dominant_pos"].map(pos_rank).fillna(len(POS_ORDER))
    )
    ordering = ordering.sort_values(
        ["_pos_rank", "dominant_pos", "total_count", "word"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    ordered_words = ordering["word"].tolist()

    matrix = word_category.pivot(index="c", columns="w", values="p_word_given_category")
    matrix = matrix.reindex(
        index=category_ids, columns=ordered_words, fill_value=0.0
    ).fillna(0.0)
    matrix.index.name = "c"
    return matrix


def build_category_word_pos_matrix(
    word_pos_category: pd.DataFrame,
    word_pos_unit_summary: pd.DataFrame,
    category_ids: Sequence[int],
) -> pd.DataFrame:
    ordered_units = word_pos_unit_summary["word_pos"].tolist()
    matrix = word_pos_category.pivot(
        index="c",
        columns="word_pos",
        values="p_word_pos_given_category",
    )
    matrix = matrix.reindex(
        index=category_ids,
        columns=ordered_units,
        fill_value=0.0,
    ).fillna(0.0)
    matrix.index.name = "c"
    return matrix
