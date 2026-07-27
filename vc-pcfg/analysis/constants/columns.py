"""Shared DataFrame column definitions."""

REQUIRED_COLUMNS = [
    "sent_id",
    "sent_len",
    "word_index",
    "word_id",
    "word",
    "viterbi_preterminal",
    "left_context",
    "right_context",
    "sentence",
    "num_preterminal_assignments",
    "preterminal_matches_length",
]

OUTPUT_EXAMPLE_COLUMNS = [
    "c",
    "w",
    "word_index",
    "previous_word",
    "next_word",
    "sentence",
    "selection_reason",
]
