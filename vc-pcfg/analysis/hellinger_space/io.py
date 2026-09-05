"""Input validation and output helpers for Hellinger-space analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    BLOCK_SUM_ATOL,
    BLOCK_SUM_RTOL,
    CONTEXT_DOMAIN,
    POSITION_ORDER,
    REQUIRED_COLUMNS,
    TARGET_DOMAIN,
)


class HellingerSpaceError(RuntimeError):
    """Raised for invalid feature-distribution inputs or analysis settings."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_category(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise HellingerSpaceError("category contains missing values.")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(dtype=float)
        if np.all(np.isclose(values, np.round(values))):
            return pd.Series(np.round(values).astype(np.int64), index=series.index)
    return series.astype(str)


def read_feature_distributions(path: Path) -> pd.DataFrame:
    """Read and validate ``category_feature_distributions.csv``.

    The source table is intentionally sparse over feature values: a missing
    category/value row means probability zero. Every category must still have
    one observed distribution for every (domain, position, feature) block.
    """

    if not path.exists():
        raise HellingerSpaceError(f"Input file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise HellingerSpaceError("Input must be a .csv feature-distribution file.")

    try:
        frame = pd.read_csv(path, low_memory=False, keep_default_na=False)
    except Exception as exc:  # pandas supplies the useful parsing detail.
        raise HellingerSpaceError(f"Could not read input CSV {path}: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise HellingerSpaceError(f"Input is missing required columns: {missing}")
    if frame.empty:
        raise HellingerSpaceError("Input contains no feature-distribution rows.")

    frame = frame.copy()
    frame["category"] = _canonicalize_category(frame["category"])

    for column in ("domain", "position", "feature", "value"):
        if frame[column].isna().any():
            raise HellingerSpaceError(f"{column} contains missing values.")
        frame[column] = frame[column].astype(str)
        if frame[column].eq("").any():
            raise HellingerSpaceError(f"{column} contains empty strings.")

    unknown_positions = sorted(set(frame["position"]) - set(POSITION_ORDER))
    if unknown_positions:
        raise HellingerSpaceError(
            f"Unsupported positions {unknown_positions}; expected only {list(POSITION_ORDER)}."
        )

    target_bad = frame.loc[
        (frame["position"] == "TARGET") & (frame["domain"] != TARGET_DOMAIN),
        ["domain", "position"],
    ]
    context_bad = frame.loc[
        (frame["position"] != "TARGET") & (frame["domain"] != CONTEXT_DOMAIN),
        ["domain", "position"],
    ]
    if not target_bad.empty or not context_bad.empty:
        raise HellingerSpaceError(
            "domain/position mismatch: TARGET must be lexical and L2/L1/R1/R2 must be contextual."
        )

    proportions = pd.to_numeric(frame["proportion"], errors="coerce")
    invalid = proportions.isna() | ~np.isfinite(proportions.to_numpy(dtype=float))
    if bool(np.any(invalid)):
        examples = frame.loc[invalid, ["category", "position", "feature", "value", "proportion"]]
        raise HellingerSpaceError(
            "proportion contains non-numeric or non-finite values, e.g. "
            + str(examples.head(5).to_dict("records"))
        )
    frame["proportion"] = proportions.astype(float)
    outside = (frame["proportion"] < 0.0) | (frame["proportion"] > 1.0)
    if outside.any():
        examples = frame.loc[outside, ["category", "position", "feature", "value", "proportion"]]
        raise HellingerSpaceError(
            "proportion values must lie in [0, 1], e.g. "
            + str(examples.head(5).to_dict("records"))
        )

    key = ["category", "domain", "position", "feature", "value"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, key].head(5).to_dict("records")
        raise HellingerSpaceError(
            "Duplicate category/block/value rows are not allowed, e.g. " + str(examples)
        )

    block_columns = ["domain", "position", "feature"]
    blocks_by_category = {
        category: set(map(tuple, group[block_columns].drop_duplicates().to_numpy()))
        for category, group in frame.groupby("category", sort=False, dropna=False)
    }
    categories = list(blocks_by_category)
    reference = blocks_by_category[categories[0]]
    for category in categories[1:]:
        current = blocks_by_category[category]
        if current != reference:
            missing_blocks = sorted(reference - current)
            extra_blocks = sorted(current - reference)
            raise HellingerSpaceError(
                f"Category {category!r} does not have the same feature blocks as the other "
                f"categories. Missing={missing_blocks[:5]}, extra={extra_blocks[:5]}."
            )

    sums = frame.groupby(["category", *block_columns], sort=False)["proportion"].sum()
    bad_sums = ~np.isclose(
        sums.to_numpy(dtype=float),
        1.0,
        atol=BLOCK_SUM_ATOL,
        rtol=BLOCK_SUM_RTOL,
    )
    if bool(np.any(bad_sums)):
        bad = sums.loc[bad_sums].head(5)
        examples = [
            {
                "category": index[0],
                "domain": index[1],
                "position": index[2],
                "feature": index[3],
                "sum": float(value),
            }
            for index, value in bad.items()
        ]
        raise HellingerSpaceError(
            "Each (category, position, feature) distribution must sum to 1. "
            f"Examples: {examples}"
        )

    return frame


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
