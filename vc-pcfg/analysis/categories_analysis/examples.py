import numpy as np
import pandas as pd

from .constants.columns import OUTPUT_EXAMPLE_COLUMNS


def choose_rows(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    n = min(n, len(frame))
    indices = rng.choice(frame.index.to_numpy(), size=n, replace=False)
    return frame.loc[indices].copy()


def append_examples(
    selected: list[pd.DataFrame],
    rows: pd.DataFrame,
    reason: str,
) -> None:
    if rows.empty:
        return
    chunk = rows[
        [
            "viterbi_preterminal",
            "word",
            "word_index",
            "previous_word",
            "next_word",
            "sentence",
        ]
    ].copy()
    chunk = chunk.rename(columns={"viterbi_preterminal": "c", "word": "w"})
    chunk["selection_reason"] = reason
    selected.append(chunk[OUTPUT_EXAMPLE_COLUMNS])


def combine_example_reasons(examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame(columns=OUTPUT_EXAMPLE_COLUMNS)

    key_columns = [
        column for column in OUTPUT_EXAMPLE_COLUMNS if column != "selection_reason"
    ]
    combined = (
        examples.groupby(key_columns, dropna=False, sort=False)["selection_reason"]
        .agg(lambda values: ";".join(dict.fromkeys(values)))
        .reset_index()
    )
    return combined[OUTPUT_EXAMPLE_COLUMNS]


def build_representative_examples(
    df: pd.DataFrame,
    category: int,
    frequent_words: pd.DataFrame,
    diagnostic_words: pd.DataFrame,
    frame_rankings: pd.DataFrame,
    examples_per_word: int,
    random_examples: int,
    top_frames_for_examples: int,
    examples_per_frame: int,
    random_seed: int,
) -> pd.DataFrame:
    category_rows = df.loc[df["viterbi_preterminal"] == category]
    if category_rows.empty:
        return pd.DataFrame(columns=OUTPUT_EXAMPLE_COLUMNS)

    rng = np.random.default_rng(random_seed + category)
    selected: list[pd.DataFrame] = []

    for word in frequent_words.loc[frequent_words["c"] == category, "w"]:
        candidates = category_rows.loc[category_rows["word"] == word]
        append_examples(
            selected,
            choose_rows(candidates, examples_per_word, rng),
            f"frequent_word:{word}",
        )

    for word in diagnostic_words.loc[diagnostic_words["c"] == category, "w"]:
        candidates = category_rows.loc[category_rows["word"] == word]
        append_examples(
            selected,
            choose_rows(candidates, examples_per_word, rng),
            f"diagnostic_word:{word}",
        )

    category_frames = frame_rankings.loc[frame_rankings["c"] == category].head(
        top_frames_for_examples
    )
    for row in category_frames.itertuples(index=False):
        candidates = category_rows.loc[
            (category_rows["previous_word"] == row.previous_word)
            & (category_rows["next_word"] == row.next_word)
        ]
        append_examples(
            selected,
            choose_rows(candidates, examples_per_frame, rng),
            f"common_frame:{row.previous_word}|{row.next_word}",
        )

    append_examples(
        selected,
        choose_rows(category_rows, random_examples, rng),
        "random",
    )

    if not selected:
        return pd.DataFrame(columns=OUTPUT_EXAMPLE_COLUMNS)
    return combine_example_reasons(pd.concat(selected, ignore_index=True))


def build_llm_input(
    df: pd.DataFrame,
    category: int,
    max_rows: int,
) -> pd.DataFrame:
    """Build an LLM-input table from tokens assigned to one category.

    A max_rows value of 0 or less means that all category rows are kept.
    """
    result = df.loc[
        df["viterbi_preterminal"].eq(category),
        ["word", "sentence"],
    ].copy()

    result = result.drop_duplicates(["word", "sentence"])

    if max_rows > 0:
        result = result.head(max_rows)

    return result.reset_index(drop=True)
