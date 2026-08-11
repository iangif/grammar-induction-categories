import pandas as pd


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    unknown = normalized.notna() & ~normalized.isin(mapping)
    if unknown.any():
        values = sorted(normalized.loc[unknown].dropna().unique().tolist())
        raise ValueError(
            f"Unrecognized boolean values in preterminal_matches_length: {values}"
        )
    return normalized.map(mapping).fillna(False).astype(bool)


def prepare_input(
    df: pd.DataFrame, bos: str = "<BOS>", eos: str = "<EOS>"
) -> pd.DataFrame:
    result = df.copy()

    integer_columns = [
        "sent_id",
        "sent_len",
        "word_index",
        "word_id",
        "viterbi_preterminal",
        "num_preterminal_assignments",
    ]
    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")

    result["preterminal_matches_length"] = normalize_boolean(
        result["preterminal_matches_length"]
    )
    result["word"] = result["word"].astype("string")
    result["sentence"] = result["sentence"].astype("string")

    result = result.sort_values(["sent_id", "word_index"], kind="stable").reset_index(
        drop=True
    )
    result["previous_word"] = (
        result.groupby("sent_id", sort=False)["word"].shift(1).fillna(bos)
    )
    result["next_word"] = (
        result.groupby("sent_id", sort=False)["word"].shift(-1).fillna(eos)
    )

    denominator = (result["sent_len"] - 1).clip(lower=1)
    result["normalized_sentence_position"] = result["word_index"] / denominator
    result.loc[result["sent_len"] <= 1, "normalized_sentence_position"] = 0.0

    return result
