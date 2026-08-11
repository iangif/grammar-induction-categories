import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(data: Any, path: Path) -> None:
    """Write JSON using UTF-8 and stable, human-readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records, mapping NaN values to null."""
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))
