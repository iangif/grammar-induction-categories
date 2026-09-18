"""Feature-space selection and normalized block weighting."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_POSITION_GROUP_WEIGHTS,
    FEATURE_SPACES,
    POSITION_GROUP_BY_POSITION,
)
from .io import HellingerSpaceError


def normalized_position_group_weights(
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Validate and normalize target/near/far positional importance weights."""

    supplied = dict(DEFAULT_POSITION_GROUP_WEIGHTS if weights is None else weights)
    expected = set(DEFAULT_POSITION_GROUP_WEIGHTS)
    missing = expected - set(supplied)
    extra = set(supplied) - expected
    if missing or extra:
        raise HellingerSpaceError(
            f"Position weights must contain exactly {sorted(expected)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )

    clean: dict[str, float] = {}
    for key in DEFAULT_POSITION_GROUP_WEIGHTS:
        try:
            value = float(supplied[key])
        except (TypeError, ValueError) as exc:
            raise HellingerSpaceError(f"Weight {key!r} must be numeric.") from exc
        if not np.isfinite(value) or value < 0.0:
            raise HellingerSpaceError(f"Weight {key!r} must be finite and non-negative.")
        clean[key] = value

    total = sum(clean.values())
    if total <= 0.0:
        raise HellingerSpaceError("At least one positional weight must be positive.")
    return {key: value / total for key, value in clean.items()}


def filter_feature_space(frame: pd.DataFrame, feature_space: str) -> pd.DataFrame:
    """Select the blocks belonging to one interpretable feature subspace."""

    if feature_space not in FEATURE_SPACES:
        raise HellingerSpaceError(
            f"Unknown feature space {feature_space!r}; expected one of {list(FEATURE_SPACES)}."
        )

    if feature_space == "all":
        selected = frame
    elif feature_space == "lexical":
        selected = frame.loc[frame["position"].eq("TARGET")]
    elif feature_space == "contextual":
        selected = frame.loc[~frame["position"].eq("TARGET")]
    else:
        if "feature_family" not in frame.columns:
            raise HellingerSpaceError(
                f"Feature space {feature_space!r} requires a feature_family column."
            )
        selected = frame.loc[frame["feature_family"].eq(feature_space)]

    if selected.empty:
        raise HellingerSpaceError(f"Feature space {feature_space!r} contains no feature blocks.")
    return selected.copy()


def build_block_weights(
    blocks: pd.DataFrame,
    *,
    position_group_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Assign same-scale Hellinger weights to active feature blocks.

    The user-level target/near/far weights are first normalized globally. For a
    selected subspace, inactive positional groups are removed and the remaining
    group weights are renormalized to sum to one. Each active group's weight is
    then split equally among its (position, feature) blocks.

    Because the final block weights sum to one, every subspace has the same
    weighted-Hellinger distance range regardless of how many blocks it contains.
    """

    required = {"domain", "position", "feature", "feature_family"}
    missing = required - set(blocks.columns)
    if missing:
        raise HellingerSpaceError(f"Cannot weight blocks; missing columns: {sorted(missing)}")

    unique = blocks[["domain", "position", "feature", "feature_family"]].drop_duplicates().copy()
    unique["position_group"] = unique["position"].map(POSITION_GROUP_BY_POSITION)
    if unique["position_group"].isna().any():
        bad = sorted(unique.loc[unique["position_group"].isna(), "position"].unique())
        raise HellingerSpaceError(f"No position-group mapping for positions: {bad}")

    original = normalized_position_group_weights(position_group_weights)
    active_groups = list(dict.fromkeys(unique["position_group"].tolist()))
    active_total = sum(original[group] for group in active_groups)
    if active_total <= 0.0:
        raise HellingerSpaceError(
            "Selected feature space has zero total weight under the supplied positional weights."
        )
    normalized = {group: original[group] / active_total for group in active_groups}

    counts = unique.groupby("position_group", sort=False).size().to_dict()
    unique["original_group_weight"] = unique["position_group"].map(original).astype(float)
    unique["normalized_group_weight"] = unique["position_group"].map(normalized).astype(float)
    unique["blocks_in_group"] = unique["position_group"].map(counts).astype(int)
    unique["block_weight"] = (
        unique["normalized_group_weight"] / unique["blocks_in_group"].astype(float)
    )

    total = float(unique["block_weight"].sum())
    if not np.isclose(total, 1.0):
        raise HellingerSpaceError(f"Internal error: active block weights sum to {total}, not 1.")
    return unique.reset_index(drop=True)
