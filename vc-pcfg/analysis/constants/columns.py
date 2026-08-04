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


EVIDENCE_EXAMPLE_COLUMNS = [
    "target_category",
    "assigned_category",
    "example_role",
    "evidence_reasons",
    "w",
    "word_pos",
    "pos",
    "tag",
    "sent_id",
    "word_index",
    "previous_word",
    "next_word",
    "sentence",
    "category_word_count",
    "corpus_word_count",
    "num_categories_for_word",
    "p_word_given_category",
    "p_category_given_word",
    "weighted_log_odds",
]
