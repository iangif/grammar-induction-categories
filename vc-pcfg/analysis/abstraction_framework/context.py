"""Radius-2 contextual vectors derived from neighboring lexical features."""

from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from .coherence import FeatureRef
from .constants import (
    BOS,
    EOS,
    BOUNDARY_FEATURE_NAME,
    CONTEXT_POSITIONS,
    LEXICAL_FEATURES,
)

def contextual_feature_refs() -> list[FeatureRef]:
    refs: list[FeatureRef] = []
    for position, _ in CONTEXT_POSITIONS:
        for feature in LEXICAL_FEATURES:
            name = f"{position}.{feature.name}"
            refs.append(FeatureRef(name=name, column=name))
        boundary = f"{position}.{BOUNDARY_FEATURE_NAME}"
        refs.append(FeatureRef(name=boundary, column=boundary))
    return refs


def build_contextual_vectors(
    tokens: pd.DataFrame,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Build z_2(t) = [x(t-2); x(t-1); x(t+1); x(t+2)].

    All lexical annotations are already present in ``tokens``. Context vectors
    are therefore built with four vectorized within-sentence shifts rather than
    a Python loop over sentences/tokens.

    Boundary positions are represented once by a sparse dedicated ``Boundary``
    feature: actual boundaries are <BOS>/<EOS>, while ordinary in-sentence positions
    are missing and therefore become NULL downstream. The contextual word column
    keeps <BOS>/<EOS> for contextual-diversity frames. Ordinary linguistic features
    are missing at out-of-sentence slots and likewise become NULL in downstream
    feature distributions. A real neighbor whose lexical feature is missing remains
    missing in exactly the same way.
    """

    source = tokens.reset_index(drop=True).copy()

    core_columns = [
        "token_id",
        "category",
        "sent_id",
        "sent_len",
        "word_index",
        "word",
    ]
    result = source[core_columns].copy()

    source_columns = ["word", *(feature.column for feature in LEXICAL_FEATURES)]

    ordered = source.sort_values("word_index", kind="stable").copy()
    ordered["_neighbor_present"] = True
    grouped = ordered.groupby("sent_id", sort=False, dropna=False)

    context_blocks: list[pd.DataFrame] = []
    progress = tqdm(
        total=len(CONTEXT_POSITIONS),
        desc="Building contextual vectors",
        unit="context",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for position, offset in CONTEXT_POSITIONS:
            # pandas shift(+d) supplies the token d places to the left;
            # shift(-d) supplies the token d places to the right.
            periods = -offset
            shifted = grouped[source_columns].shift(periods)

            # shift() produces missing values both for sentence boundaries and
            # for genuine NULL lexical annotations. Shift an always-present
            # sentinel separately so only true boundaries are overwritten.
            neighbor_exists = grouped["_neighbor_present"].shift(periods).notna()
            boundary_mask = ~neighbor_exists
            boundary = BOS if offset < 0 else EOS
            if boundary_mask.any():
                # Keep explicit BOS/EOS only for contextual word identity. The
                # ordinary linguistic feature columns stay missing so downstream
                # distribution/coherence code treats them as NULL once, rather
                # than repeating the boundary symbol across every feature.
                shifted.loc[boundary_mask, "word"] = boundary

            # Boundary is a sparse event feature: only actual BOS/EOS events
            # receive a value. In-sentence positions stay missing so they become
            # NULL in distributions and cannot win modal coherence.
            boundary_values = pd.Series(pd.NA, index=shifted.index, dtype="string")
            boundary_values.loc[boundary_mask] = boundary

            shifted.columns = [
                f"{position}.word",
                *(f"{position}.{feature.name}" for feature in LEXICAL_FEATURES),
            ]
            shifted[f"{position}.{BOUNDARY_FEATURE_NAME}"] = boundary_values
            context_blocks.append(shifted)
            progress.update(1)
    finally:
        progress.close()

    if context_blocks:
        result = pd.concat([result, *context_blocks], axis=1)

    return result.reset_index(drop=True)
