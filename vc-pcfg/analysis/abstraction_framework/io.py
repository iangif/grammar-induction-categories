"""Input validation, canonicalization, hashing, and table I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class FrameworkError(RuntimeError):
    """Raised for invalid inputs or framework configuration."""

@dataclass(frozen=True)
class InputColumns:
    category: str = "viterbi_preterminal"
    sent_id: str = "sent_id"
    sent_len: str = "sent_len"
    word_index: str = "word_index"
    word: str = "word"
    sentence: str = "sentence"

    def required(self) -> tuple[str, ...]:
        return (
            self.category,
            self.sent_id,
            self.sent_len,
            self.word_index,
            self.word,
            self.sentence,
        )

def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FrameworkError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except ImportError as exc:
        raise FrameworkError(
            "Reading Parquet requires a Parquet engine. Install pyarrow (recommended)."
        ) from exc
    raise FrameworkError("Input must be .csv, .parquet, or .pq.")

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise FrameworkError(
            "Writing Parquet requires a Parquet engine. Install pyarrow (recommended)."
        ) from exc

def write_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    float_format: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)

def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
    *,
    show_progress: bool = False,
) -> str:
    """Hash a file, optionally showing byte-level progress for large inputs."""

    digest = hashlib.sha256()
    progress = tqdm(
        total=path.stat().st_size,
        desc="Hashing input",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
                progress.update(len(chunk))
    finally:
        progress.close()
    return digest.hexdigest()

def _coerce_integral(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = series.loc[numeric.isna()].head(5).tolist()
        raise FrameworkError(f"Column {name!r} contains non-numeric/missing values, e.g. {bad}")
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isclose(values, np.round(values))):
        raise FrameworkError(f"Column {name!r} must contain integer values.")
    return pd.Series(np.round(values).astype(np.int64), index=series.index)

def _canonicalize_category(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise FrameworkError("Category column contains NULL values.")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(dtype=float)
        if np.all(np.isclose(values, np.round(values))):
            return pd.Series(np.round(values).astype(np.int64), index=series.index)
    return series.astype(str)


def validate_and_canonicalize(
    raw: pd.DataFrame,
    columns: InputColumns,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Validate one-token-per-row exports and return a canonical token table.

    The canonical representation has exactly these core columns:
    category, sent_id, sent_len, word_index, word, sentence, token_id.
    Additional source columns are deliberately not required by later stages.
    """

    missing = [column for column in columns.required() if column not in raw.columns]
    if missing:
        raise FrameworkError(f"Input is missing required columns: {missing}")

    core = raw.loc[:, list(columns.required())].copy()
    core = core.rename(
        columns={
            columns.category: "category",
            columns.sent_id: "sent_id",
            columns.sent_len: "sent_len",
            columns.word_index: "word_index",
            columns.word: "word",
            columns.sentence: "sentence",
        }
    )

    if core["sent_id"].isna().any():
        raise FrameworkError("sent_id contains NULL values.")
    core["category"] = _canonicalize_category(core["category"])
    core["sent_len"] = _coerce_integral(core["sent_len"], "sent_len")
    core["word_index"] = _coerce_integral(core["word_index"], "word_index")

    if (core["sent_len"] <= 0).any():
        raise FrameworkError("sent_len must be positive for every row.")
    if (core["word_index"] < 0).any():
        raise FrameworkError("word_index must be zero-based and non-negative.")

    if core["word"].isna().any():
        raise FrameworkError("word contains NULL values.")
    core["word"] = core["word"].astype(str)
    if core["word"].eq("").any():
        raise FrameworkError("word contains empty strings.")

    if core["sentence"].isna().any():
        raise FrameworkError("sentence contains NULL values.")
    core["sentence"] = core["sentence"].astype(str)

    duplicates = core.duplicated(["sent_id", "word_index"], keep=False)
    if duplicates.any():
        examples = core.loc[duplicates, ["sent_id", "word_index"]].head(5).to_dict("records")
        raise FrameworkError(
            "Duplicate token positions found for (sent_id, word_index), e.g. " + str(examples)
        )

    pieces: list[pd.DataFrame] = []
    groups = core.groupby("sent_id", sort=False, dropna=False)
    progress = tqdm(
        groups,
        total=groups.ngroups,
        desc="Validating sentences",
        unit="sent",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for sent_id, group in progress:
        lengths = group["sent_len"].unique()
        if len(lengths) != 1:
            raise FrameworkError(f"Sentence {sent_id!r} has inconsistent sent_len values: {lengths.tolist()}")
        sent_len = int(lengths[0])
        if len(group) != sent_len:
            raise FrameworkError(
                f"Sentence {sent_id!r} has {len(group)} token rows but sent_len={sent_len}."
            )
        ordered = group.sort_values("word_index", kind="stable").copy()
        expected = np.arange(sent_len, dtype=np.int64)
        observed = ordered["word_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise FrameworkError(
                f"Sentence {sent_id!r} word_index values must be exactly 0..{sent_len - 1}; "
                f"observed {observed.tolist()}."
            )
        if ordered["sentence"].nunique(dropna=False) != 1:
            raise FrameworkError(f"Sentence {sent_id!r} has inconsistent sentence text across token rows.")
        pieces.append(ordered)

    result = pd.concat(pieces, ignore_index=True) if pieces else core.iloc[0:0].copy()
    if result.empty:
        raise FrameworkError("Input contains no token rows.")

    result.insert(
        0,
        "token_id",
        result["sent_id"].astype(str) + ":" + result["word_index"].astype(str),
    )
    return result
