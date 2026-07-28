from pathlib import Path

import pandas as pd

from .constants.columns import REQUIRED_COLUMNS


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Unsupported input format '{suffix}'. "
                "Expected .csv, .parquet, or .pq."
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to load input file {path}: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
