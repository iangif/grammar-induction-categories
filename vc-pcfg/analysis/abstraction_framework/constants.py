"""Shared constants for the induced-category abstraction framework."""

from __future__ import annotations

from dataclasses import dataclass

PACKAGE_VERSION = "0.2.0"
FEATURE_SCHEMA_VERSION = "2"
CACHE_SCHEMA_VERSION = "2"

BOS = "<BOS>"
EOS = "<EOS>"
NULL_VALUE = "NULL"
CONTEXT_RADIUS = 2  # For contextual coherence.


@dataclass(frozen=True)
class LexicalFeature:
    """One lexical feature used for annotation, profiles, and coherence."""

    name: str
    column: str
    source: str


# Keep this inventory in the exact declared order. Contextual features are made
# by applying this same inventory at L2, L1, R1, and R2.
LEXICAL_FEATURES: tuple[LexicalFeature, ...] = (
    LexicalFeature("Coarse POS", "lex_coarse_pos", "spaCy Token.pos_ (UPOS)"),
    LexicalFeature("Fine POS", "lex_fine_pos", "spaCy Token.tag_ (PTB for English)"),
    LexicalFeature("Number", "lex_number", "spaCy MorphAnalysis[Number]"),
    LexicalFeature("Person", "lex_person", "spaCy MorphAnalysis[Person]"),
    LexicalFeature("Properness", "lex_properness", "spaCy UPOS NOUN/PROPN"),
    LexicalFeature("Animacy", "lex_animacy", "WordNet noun.person/noun.animal supersenses"),
    LexicalFeature(
        "Humanness",
        "lex_humanness",
        "pronoun form, spaCy PERSON entity, or WordNet noun.person supersense",
    ),
    LexicalFeature("Named-entity type", "lex_named_entity_type", "spaCy Token.ent_type_"),
    LexicalFeature(
        "Noun semantic class",
        "lex_noun_semantic_class",
        "WordNet noun lexname/supersense of contextual sense",
    ),
    LexicalFeature("Pronoun type", "lex_pronoun_type", "UD morphology with lexical fallback"),
    LexicalFeature(
        "Pronoun case/form",
        "lex_pronoun_case_form",
        "UD morphology, PTB tag, and pronoun form",
    ),
    LexicalFeature("Verb type", "lex_verb_type", "spaCy UPOS/Fine POS: lexical, auxiliary, modal"),
    LexicalFeature("Verb form", "lex_verb_form", "spaCy MorphAnalysis[VerbForm]"),
    LexicalFeature("Tense", "lex_tense", "spaCy MorphAnalysis[Tense]"),
    LexicalFeature(
        "Verb semantic class",
        "lex_verb_semantic_class",
        "WordNet verb lexname/supersense of contextual sense",
    ),
    LexicalFeature("Adjective degree", "lex_adjective_degree", "spaCy MorphAnalysis[Degree]"),
    LexicalFeature(
        "Adjective semantic class",
        "lex_adjective_semantic_class",
        "WordNet noun attribute linked to contextual adjective sense",
    ),
    LexicalFeature("Adverb degree", "lex_adverb_degree", "spaCy MorphAnalysis[Degree]"),
    LexicalFeature(
        "Adverb semantic class",
        "lex_adverb_semantic_class",
        "WordNet adjective pertainym then noun attribute",
    ),
    LexicalFeature("Determiner type", "lex_determiner_type", "UD morphology with lexical fallback"),
    LexicalFeature("Conjunction type", "lex_conjunction_type", "spaCy UPOS CCONJ/SCONJ"),
)

CONTEXT_POSITIONS: tuple[tuple[str, int], ...] = (
    ("L2", -2),
    ("L1", -1),
    ("R1", 1),
    ("R2", 2),
)
