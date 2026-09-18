"""Shared constants for Hellinger-space analyses."""

from __future__ import annotations

PACKAGE_VERSION = "0.2.0"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "category",
    "domain",
    "position",
    "feature",
    "value",
    "proportion",
)

FEATURE_SPACES: tuple[str, ...] = (
    "all",
    "lexical",
    "contextual",
    "grammatical",
    "semantic",
)
PROJECTIONS: tuple[str, ...] = ("pca", "umap", "tsne")

POSITION_ORDER: tuple[str, ...] = ("TARGET", "L2", "L1", "R1", "R2")
POSITION_RANK = {position: index for index, position in enumerate(POSITION_ORDER)}

TARGET_DOMAIN = "lexical"
CONTEXT_DOMAIN = "contextual"

POSITION_GROUP_BY_POSITION = {
    "TARGET": "target",
    "L1": "near_context",
    "R1": "near_context",
    "L2": "far_context",
    "R2": "far_context",
}
POSITION_GROUP_ORDER: tuple[str, ...] = ("target", "near_context", "far_context")
DEFAULT_POSITION_GROUP_WEIGHTS = {
    "target": 0.50,
    "near_context": 0.35,
    "far_context": 0.15,
}

# Backward-compatible feature-family inference for abstraction outputs produced
# before feature_family became an explicit column.
GRAMMATICAL_FEATURES = {
    "Coarse POS",
    "Fine POS",
    "Number",
    "Person",
    "Properness",
    "Pronoun type",
    "Pronoun case/form",
    "Verb type",
    "Verb form",
    "Tense",
    "Adjective degree",
    "Adverb degree",
    "Determiner type",
    "Conjunction type",
}
SEMANTIC_FEATURES = {
    "Animacy",
    "Humanness",
    "Named-entity type",
    "Noun semantic class",
    "Verb semantic class",
    "Adjective semantic class",
    "Adverb semantic class",
}
FEATURE_FAMILY_BY_NAME = {
    **{name: "grammatical" for name in GRAMMATICAL_FEATURES},
    **{name: "semantic" for name in SEMANTIC_FEATURES},
}

BLOCK_SUM_ATOL = 1e-8
BLOCK_SUM_RTOL = 1e-8
