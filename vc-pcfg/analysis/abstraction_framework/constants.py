"""Shared constants for the induced-category abstraction framework."""

from __future__ import annotations

from dataclasses import dataclass

PACKAGE_VERSION = "0.3.0"
FEATURE_SCHEMA_VERSION = "3"
CACHE_SCHEMA_VERSION = "2"

BOS = "<BOS>"
EOS = "<EOS>"
NULL_VALUE = "NULL"
CONTEXT_RADIUS = 2  # For contextual coherence.

GRAMMATICAL_FAMILY = "grammatical"
SEMANTIC_FAMILY = "semantic"
FEATURE_FAMILIES: tuple[str, ...] = (GRAMMATICAL_FAMILY, SEMANTIC_FAMILY)


@dataclass(frozen=True)
class LexicalFeature:
    """One lexical feature used for annotation, profiles, and coherence."""

    name: str
    column: str
    source: str
    family: str


# Keep this inventory in the exact declared order. Contextual features are made
# by applying this same inventory at L2, L1, R1, and R2. ``family`` is an
# orthogonal analysis label: lexical/contextual is determined by *position*,
# while grammatical/semantic is determined by the linguistic property itself.
LEXICAL_FEATURES: tuple[LexicalFeature, ...] = (
    LexicalFeature("Coarse POS", "lex_coarse_pos", "spaCy Token.pos_ (UPOS)", GRAMMATICAL_FAMILY),
    LexicalFeature("Fine POS", "lex_fine_pos", "spaCy Token.tag_ (PTB for English)", GRAMMATICAL_FAMILY),
    LexicalFeature("Number", "lex_number", "spaCy MorphAnalysis[Number]", GRAMMATICAL_FAMILY),
    LexicalFeature("Person", "lex_person", "spaCy MorphAnalysis[Person]", GRAMMATICAL_FAMILY),
    LexicalFeature("Properness", "lex_properness", "spaCy UPOS NOUN/PROPN", GRAMMATICAL_FAMILY),
    LexicalFeature(
        "Animacy",
        "lex_animacy",
        "WordNet noun.person/noun.animal supersenses",
        SEMANTIC_FAMILY,
    ),
    LexicalFeature(
        "Humanness",
        "lex_humanness",
        "pronoun form, spaCy PERSON entity, or WordNet noun.person supersense",
        SEMANTIC_FAMILY,
    ),
    LexicalFeature(
        "Named-entity type",
        "lex_named_entity_type",
        "spaCy Token.ent_type_",
        SEMANTIC_FAMILY,
    ),
    LexicalFeature(
        "Noun semantic class",
        "lex_noun_semantic_class",
        "WordNet noun lexname/supersense of contextual sense",
        SEMANTIC_FAMILY,
    ),
    LexicalFeature(
        "Pronoun type",
        "lex_pronoun_type",
        "UD morphology with lexical fallback",
        GRAMMATICAL_FAMILY,
    ),
    LexicalFeature(
        "Pronoun case/form",
        "lex_pronoun_case_form",
        "UD morphology, PTB tag, and pronoun form",
        GRAMMATICAL_FAMILY,
    ),
    LexicalFeature(
        "Verb type",
        "lex_verb_type",
        "spaCy UPOS/Fine POS: lexical, auxiliary, modal",
        GRAMMATICAL_FAMILY,
    ),
    LexicalFeature("Verb form", "lex_verb_form", "spaCy MorphAnalysis[VerbForm]", GRAMMATICAL_FAMILY),
    LexicalFeature("Tense", "lex_tense", "spaCy MorphAnalysis[Tense]", GRAMMATICAL_FAMILY),
    LexicalFeature(
        "Verb semantic class",
        "lex_verb_semantic_class",
        "WordNet verb lexname/supersense of contextual sense",
        SEMANTIC_FAMILY,
    ),
    LexicalFeature(
        "Adjective degree",
        "lex_adjective_degree",
        "spaCy MorphAnalysis[Degree]",
        GRAMMATICAL_FAMILY,
    ),
    LexicalFeature(
        "Adjective semantic class",
        "lex_adjective_semantic_class",
        "WordNet noun attribute linked to contextual adjective sense",
        SEMANTIC_FAMILY,
    ),
    LexicalFeature(
        "Adverb degree",
        "lex_adverb_degree",
        "spaCy MorphAnalysis[Degree]",
        GRAMMATICAL_FAMILY,
    ),
    LexicalFeature(
        "Adverb semantic class",
        "lex_adverb_semantic_class",
        "WordNet adjective pertainym then noun attribute",
        SEMANTIC_FAMILY,
    ),
    LexicalFeature(
        "Determiner type",
        "lex_determiner_type",
        "UD morphology with lexical fallback",
        GRAMMATICAL_FAMILY,
    ),
    LexicalFeature(
        "Conjunction type",
        "lex_conjunction_type",
        "spaCy UPOS CCONJ/SCONJ",
        GRAMMATICAL_FAMILY,
    ),
)

FEATURE_FAMILY_BY_NAME = {feature.name: feature.family for feature in LEXICAL_FEATURES}
FEATURE_ORDER = {feature.name: index for index, feature in enumerate(LEXICAL_FEATURES)}

CONTEXT_POSITIONS: tuple[tuple[str, int], ...] = (
    ("L2", -2),
    ("L1", -1),
    ("R1", 1),
    ("R2", 2),
)
