"""Lexical and contextual diversity metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def effective_diversity(values: Iterable[Hashable]) -> float:
    """exp(Shannon entropy) for a non-empty multiset of values."""

    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.asarray(list(counts.values()), dtype=float) / float(total)
    entropy = -float(np.sum(probs * np.log(probs)))
    return math.exp(entropy)


def _frame_series(contexts: pd.DataFrame, radius: int) -> pd.Series:
    columns = [f"L{distance}.word" for distance in range(radius, 0, -1)] + [
        f"R{distance}.word" for distance in range(1, radius + 1)
    ]
    missing = [column for column in columns if column not in contexts.columns]
    if missing:
        raise ValueError(f"Missing context-word columns for radius {radius}: {missing}")

    parts = [contexts[column].astype("string").fillna("NULL") for column in columns]
    frame = parts[0]
    for part in parts[1:]:
        frame = frame + "\u241f" + part
    return frame


def compute_diversity(
    tokens: pd.DataFrame,
    contexts: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute LD/LD_eff and CD/CD_eff for r=1 and r=2."""

    required_token_columns = {"token_id", "category", "word"}
    missing_tokens = sorted(required_token_columns.difference(tokens.columns))
    if missing_tokens:
        raise ValueError(f"Missing token columns: {missing_tokens}")
    if "token_id" not in contexts.columns:
        raise ValueError("contexts must contain token_id")

    context_words = contexts[
        ["token_id", "L2.word", "L1.word", "R1.word", "R2.word"]
    ].copy()
    work = tokens[["token_id", "category", "word"]].merge(
        context_words,
        on="token_id",
        how="left",
        validate="one_to_one",
    )
    work["_frame_r1"] = _frame_series(work, 1)
    work["_frame_r2"] = _frame_series(work, 2)

    summary_rows: list[dict[str, object]] = []
    groups = work.groupby("category", sort=False, dropna=False)
    progress = tqdm(
        groups,
        total=groups.ngroups,
        desc="Computing diversity",
        unit="cat",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for category, group in progress:
        words = group["word"].tolist()
        frame1 = group["_frame_r1"].tolist()
        frame2 = group["_frame_r2"].tolist()
        summary_rows.append(
            {
                "category": category,
                "token_count": len(group),
                "LD": len(set(words)),
                "LD_eff": effective_diversity(words),
                "CD1": len(set(frame1)),
                "CD1_eff": effective_diversity(frame1),
                "CD2": len(set(frame2)),
                "CD2_eff": effective_diversity(frame2),
            }
        )

    return pd.DataFrame(summary_rows)
