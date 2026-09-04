"""Lexical feature annotation and annotated-token cache management."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import spacy
from spacy.tokens import Doc
from tqdm.auto import tqdm

from .constants import (
    CACHE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    LEXICAL_FEATURES,
    PACKAGE_VERSION,
)
from .io import (
    FrameworkError,
    InputColumns,
    read_json,
    read_table,
    sha256_file,
    validate_and_canonicalize,
    write_json,
    write_parquet,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z_-]*")

_PRON_TYPE_MAP = {
    "Prs": "Personal",
    "Dem": "Demonstrative",
    "Int": "Interrogative",
    "Rel": "Relative",
    "Ind": "Indefinite",
    "Tot": "Total",
    "Neg": "Negative",
    "Rcp": "Reciprocal",
    "Art": "Article",
}

_DEGREE_MAP = {
    "Pos": "Positive",
    "Cmp": "Comparative",
    "Sup": "Superlative",
}

_VERB_FORM_MAP = {
    "Fin": "Finite",
    "Inf": "Infinitive",
    "Part": "Participle",
    "Ger": "Gerund",
}

_TENSE_MAP = {
    "Past": "Past",
    "Pres": "Present",
    "Fut": "Future",
}

_HUMAN_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "themself",
    "who",
    "whom",
    "whose",
    "whoever",
    "whomever",
}

_NONHUMAN_PRONOUNS = {
    "it",
    "its",
    "itself",
    "what",
    "whatever",
    "which",
    "whichever",
}

_SUBJECT_PRONOUNS = {"i", "we", "he", "she", "they", "who"}
_OBJECT_PRONOUNS = {"me", "us", "him", "them", "whom"}
_SYNCRETIC_PRONOUNS = {"you", "it"}
_POSSESSIVE_PRONOUNS = {
    "my",
    "mine",
    "our",
    "ours",
    "your",
    "yours",
    "his",
    "her",
    "hers",
    "its",
    "their",
    "theirs",
    "whose",
}

_PRONOUN_TYPE_FALLBACK = {
    "i": "Personal",
    "me": "Personal",
    "my": "Personal",
    "mine": "Personal",
    "myself": "Personal",
    "we": "Personal",
    "us": "Personal",
    "our": "Personal",
    "ours": "Personal",
    "ourselves": "Personal",
    "you": "Personal",
    "your": "Personal",
    "yours": "Personal",
    "yourself": "Personal",
    "yourselves": "Personal",
    "he": "Personal",
    "him": "Personal",
    "his": "Personal",
    "himself": "Personal",
    "she": "Personal",
    "her": "Personal",
    "hers": "Personal",
    "herself": "Personal",
    "it": "Personal",
    "its": "Personal",
    "itself": "Personal",
    "they": "Personal",
    "them": "Personal",
    "their": "Personal",
    "theirs": "Personal",
    "themselves": "Personal",
    "themself": "Personal",
    "this": "Demonstrative",
    "that": "Demonstrative",
    "these": "Demonstrative",
    "those": "Demonstrative",
    "who": "Interrogative/Relative",
    "whom": "Interrogative/Relative",
    "whose": "Interrogative/Relative",
    "what": "Interrogative/Relative",
    "which": "Interrogative/Relative",
    "someone": "Indefinite",
    "somebody": "Indefinite",
    "something": "Indefinite",
    "anyone": "Indefinite",
    "anybody": "Indefinite",
    "anything": "Indefinite",
    "everyone": "Indefinite",
    "everybody": "Indefinite",
    "everything": "Indefinite",
    "noone": "Indefinite",
    "nobody": "Indefinite",
    "nothing": "Indefinite",
    "one": "Indefinite",
    "ones": "Indefinite",
    "each": "Indefinite",
    "either": "Indefinite",
    "neither": "Indefinite",
    "both": "Indefinite",
    "all": "Indefinite",
    "none": "Indefinite",
    "another": "Indefinite",
    "other": "Indefinite",
    "others": "Indefinite",
    "each other": "Reciprocal",
    "one another": "Reciprocal",
}

_DETERMINER_FALLBACK = {
    "a": "Article",
    "an": "Article",
    "the": "Article",
    "this": "Demonstrative",
    "that": "Demonstrative",
    "these": "Demonstrative",
    "those": "Demonstrative",
    "my": "Possessive",
    "your": "Possessive",
    "his": "Possessive",
    "her": "Possessive",
    "its": "Possessive",
    "our": "Possessive",
    "their": "Possessive",
    "whose": "Possessive",
    "each": "Distributive",
    "every": "Distributive",
    "either": "Distributive",
    "neither": "Distributive",
    "some": "Indefinite",
    "any": "Indefinite",
    "many": "Quantifier",
    "much": "Quantifier",
    "few": "Quantifier",
    "little": "Quantifier",
    "several": "Quantifier",
    "all": "Total",
    "both": "Total",
    "enough": "Quantifier",
    "no": "Negative",
    "what": "Interrogative",
    "which": "Interrogative/Relative",
}


def _one_or_joined(values: list[str]) -> str | None:
    if not values:
        return None
    return "|".join(sorted(set(values)))


def _mapped_morph(token: Any, key: str, mapping: dict[str, str] | None = None) -> str | None:
    values = token.morph.get(key)
    if not values:
        return None
    if mapping is not None:
        values = [mapping.get(value, value) for value in values]
    return _one_or_joined(values)


def _ensure_wordnet_available() -> None:
    try:
        from nltk.corpus import wordnet as wn

        wn.ensure_loaded()
    except ImportError as exc:
        raise FrameworkError(
            "Semantic lexical features require NLTK. Install it with `pip install nltk`."
        ) from exc
    except LookupError as exc:
        raise FrameworkError(
            "Semantic lexical features require the NLTK WordNet corpus. Install it with:\n"
            "  python -m nltk.downloader wordnet"
        ) from exc


@lru_cache(maxsize=100_000)
def _wordnet_synsets(lemma: str, pos: str) -> tuple[Any, ...]:
    from nltk.corpus import wordnet as wn

    normalized = lemma.lower().replace(" ", "_")
    return tuple(wn.synsets(normalized, pos=pos))


@lru_cache(maxsize=250_000)
def _synset_signature(synset_name: str) -> frozenset[str]:
    from nltk.corpus import wordnet as wn

    definition = wn.synset(synset_name).definition().lower()
    return frozenset(_WORD_RE.findall(definition))


def _contextual_synset(context_words: list[str], lemma: str, pos: str) -> Any | None:
    """Select a WordNet sense with a lightweight, deterministic Lesk score.

    Definitions are compared against the lower-cased sentence context. Ties keep
    WordNet's sense order, so a zero-overlap case falls back to the first sense.
    """

    synsets = _wordnet_synsets(lemma, pos)
    if not synsets:
        return None
    context = set()
    for word in context_words:
        context.update(_WORD_RE.findall(str(word).lower()))
    return max(
        synsets,
        key=lambda synset: len(context.intersection(_synset_signature(synset.name()))),
    )


def _noun_semantic_class(token: Any, context_words: list[str]) -> str | None:
    if token.pos_ not in {"NOUN", "PROPN"}:
        return None
    lemma = token.lemma_ or token.text
    synset = _contextual_synset(context_words, lemma, "n")
    return synset.lexname() if synset is not None else None


def _verb_semantic_class(token: Any, context_words: list[str]) -> str | None:
    if token.pos_ != "VERB":
        return None
    lemma = token.lemma_ or token.text
    synset = _contextual_synset(context_words, lemma, "v")
    return synset.lexname() if synset is not None else None


def _attribute_class_from_adjective_synset(synset: Any | None) -> str | None:
    if synset is None:
        return None
    attributes = sorted(synset.attributes(), key=lambda item: item.name())
    if not attributes:
        return None
    # Keep the specific WordNet noun attribute (e.g. ``size.n.01``), not its
    # lexname, because the lexname would collapse all of them to noun.attribute.
    return attributes[0].name()


def _adjective_semantic_class(token: Any, context_words: list[str]) -> str | None:
    if token.pos_ != "ADJ":
        return None
    lemma = token.lemma_ or token.text
    synset = _contextual_synset(context_words, lemma, "a")
    return _attribute_class_from_adjective_synset(synset)


def _adverb_semantic_class(token: Any, context_words: list[str]) -> str | None:
    if token.pos_ != "ADV":
        return None
    lemma = token.lemma_ or token.text
    adverb_synset = _contextual_synset(context_words, lemma, "r")
    if adverb_synset is None:
        return None

    adjective_lemmas: list[Any] = []
    for wordnet_lemma in adverb_synset.lemmas():
        adjective_lemmas.extend(wordnet_lemma.pertainyms())
    if not adjective_lemmas:
        return None

    # The pertainym relation already identifies an adjective sense. Prefer a
    # deterministic order if WordNet provides multiple derivations.
    adjective_lemmas.sort(key=lambda item: (item.synset().name(), item.name()))
    for adjective_lemma in adjective_lemmas:
        semantic_class = _attribute_class_from_adjective_synset(adjective_lemma.synset())
        if semantic_class is not None:
            return semantic_class
    return None


def _properness(token: Any) -> str | None:
    if token.pos_ == "PROPN":
        return "Proper"
    if token.pos_ == "NOUN":
        return "Common"
    return None


def _animacy(token: Any, noun_semantic_class: str | None) -> str | None:
    if token.pos_ not in {"NOUN", "PROPN"} or noun_semantic_class is None:
        return None
    if noun_semantic_class in {"noun.person", "noun.animal"}:
        return "Animate"
    return "Inanimate"


def _humanness(token: Any, noun_semantic_class: str | None) -> str | None:
    lower = token.text.lower()
    if token.pos_ == "PRON":
        if lower in _HUMAN_PRONOUNS:
            return "Human"
        if lower in _NONHUMAN_PRONOUNS:
            return "Non-human"
        return None
    if token.ent_type_ == "PERSON" or noun_semantic_class == "noun.person":
        return "Human"
    if token.pos_ in {"NOUN", "PROPN"}:
        if noun_semantic_class is not None or token.ent_type_:
            return "Non-human"
    return None


def _pronoun_type(token: Any) -> str | None:
    if token.pos_ != "PRON":
        return None
    morph = _mapped_morph(token, "PronType", _PRON_TYPE_MAP)
    if morph is not None:
        return morph
    return _PRONOUN_TYPE_FALLBACK.get(token.text.lower(), "Other")


def _pronoun_case_form(token: Any) -> str | None:
    if token.pos_ != "PRON":
        return None
    lower = token.text.lower()
    if lower.endswith("self") or lower.endswith("selves") or token.morph.get("Reflex") == ["Yes"]:
        return "Reflexive"
    if lower == "her" and token.tag_ == "PRP":
        return "Object"
    if token.tag_ == "PRP$" or token.morph.get("Poss") == ["Yes"]:
        return "Possessive"

    case = token.morph.get("Case")
    if "Nom" in case:
        return "Subject"
    if "Acc" in case:
        return "Object"
    if "Gen" in case:
        return "Possessive"
    if lower in _SUBJECT_PRONOUNS:
        return "Subject"
    if lower in _OBJECT_PRONOUNS:
        return "Object"
    if lower in _POSSESSIVE_PRONOUNS:
        return "Possessive"
    if lower in _SYNCRETIC_PRONOUNS:
        return "Syncretic subject/object"
    return "Unmarked"


def _verb_type(token: Any) -> str | None:
    if token.tag_ == "MD":
        return "Modal"
    if token.pos_ == "AUX":
        return "Auxiliary"
    if token.pos_ == "VERB":
        return "Lexical"
    return None


def _verb_form(token: Any) -> str | None:
    if token.pos_ not in {"VERB", "AUX"}:
        return None
    value = _mapped_morph(token, "VerbForm", _VERB_FORM_MAP)
    if value is not None:
        return value
    fallback = {
        "VB": "Infinitive",
        "VBD": "Finite",
        "VBG": "Gerund",
        "VBN": "Participle",
        "VBP": "Finite",
        "VBZ": "Finite",
    }
    return fallback.get(token.tag_)


def _tense(token: Any) -> str | None:
    if token.pos_ not in {"VERB", "AUX"}:
        return None
    value = _mapped_morph(token, "Tense", _TENSE_MAP)
    if value is not None:
        return value
    if token.tag_ in {"VBD", "VBN"}:
        return "Past"
    if token.tag_ in {"VBP", "VBZ"}:
        return "Present"
    return None


def _degree(token: Any, expected_pos: str) -> str | None:
    if token.pos_ != expected_pos:
        return None
    value = _mapped_morph(token, "Degree", _DEGREE_MAP)
    if value is not None:
        return value
    if expected_pos == "ADJ":
        return {"JJ": "Positive", "JJR": "Comparative", "JJS": "Superlative"}.get(token.tag_)
    return {"RB": "Positive", "RBR": "Comparative", "RBS": "Superlative"}.get(token.tag_)


def _determiner_type(token: Any) -> str | None:
    if token.pos_ != "DET":
        return None
    lower = token.text.lower()
    if token.morph.get("Poss") == ["Yes"] or lower in _POSSESSIVE_PRONOUNS:
        return "Possessive"
    pron_type = token.morph.get("PronType")
    mapped = [_PRON_TYPE_MAP.get(value, value) for value in pron_type]
    if mapped:
        return _one_or_joined(mapped)
    return _DETERMINER_FALLBACK.get(lower, "Other")


def _conjunction_type(token: Any) -> str | None:
    if token.pos_ == "CCONJ":
        return "Coordinating"
    if token.pos_ == "SCONJ":
        return "Subordinating"
    return None


def _token_feature_values(token: Any, context_words: list[str]) -> dict[str, str | None]:
    noun_semantic_class = _noun_semantic_class(token, context_words)
    return {
        "lex_coarse_pos": token.pos_ or None,
        "lex_fine_pos": token.tag_ or None,
        "lex_number": _mapped_morph(token, "Number"),
        "lex_person": _mapped_morph(token, "Person"),
        "lex_properness": _properness(token),
        "lex_animacy": _animacy(token, noun_semantic_class),
        "lex_humanness": _humanness(token, noun_semantic_class),
        "lex_named_entity_type": token.ent_type_ or None,
        "lex_noun_semantic_class": noun_semantic_class,
        "lex_pronoun_type": _pronoun_type(token),
        "lex_pronoun_case_form": _pronoun_case_form(token),
        "lex_verb_type": _verb_type(token),
        "lex_verb_form": _verb_form(token),
        "lex_tense": _tense(token),
        "lex_verb_semantic_class": _verb_semantic_class(token, context_words),
        "lex_adjective_degree": _degree(token, "ADJ"),
        "lex_adjective_semantic_class": _adjective_semantic_class(token, context_words),
        "lex_adverb_degree": _degree(token, "ADV"),
        "lex_adverb_semantic_class": _adverb_semantic_class(token, context_words),
        "lex_determiner_type": _determiner_type(token),
        "lex_conjunction_type": _conjunction_type(token),
    }


def load_spacy_model(model_name: str):
    try:
        # Parser dependencies are not used by this feature inventory. Keep NER,
        # tagging, morphology, attribute rules, and lemmatization enabled.
        nlp = spacy.load(model_name, disable=["parser"])
    except OSError as exc:
        raise FrameworkError(
            f"Could not load spaCy model {model_name!r}. Install it, for example with:\n"
            f"  python -m spacy download {model_name}"
        ) from exc

    pipe_names = set(nlp.pipe_names)
    if "ner" not in pipe_names:
        raise FrameworkError(
            f"spaCy model {model_name!r} has no NER component, which is required for Named-entity type."
        )
    return nlp


def annotate_lexical_features(
    tokens: pd.DataFrame,
    *,
    model_name: str,
    batch_size: int = 256,
    n_process: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Attach exactly the declared lexical feature inventory to every token."""

    if batch_size <= 0:
        raise FrameworkError("spacy batch_size must be positive.")
    if n_process <= 0:
        raise FrameworkError("spacy n_process must be positive.")

    _ensure_wordnet_available()
    nlp = load_spacy_model(model_name)
    result = tokens.reset_index(drop=True).copy()
    feature_values: dict[str, list[Any]] = {
        feature.column: [None] * len(result) for feature in LEXICAL_FEATURES
    }

    records: list[tuple[list[int], list[str]]] = []
    groups = result.groupby("sent_id", sort=False, dropna=False)
    preparation = tqdm(
        groups,
        total=groups.ngroups,
        desc="Preparing spaCy input",
        unit="sent",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for _, group in preparation:
        ordered = group.sort_values("word_index", kind="stable")
        records.append((ordered.index.tolist(), ordered["word"].tolist()))

    docs: Iterable[Doc] = (Doc(nlp.vocab, words=words) for _, words in records)
    processed = nlp.pipe(docs, batch_size=batch_size, n_process=n_process)

    progress = tqdm(
        total=len(result),
        desc="Annotating lexical features",
        unit="tok",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for (row_indices, words), doc in zip(records, processed, strict=True):
            if len(doc) != len(words):
                raise FrameworkError(
                    "spaCy changed token boundaries unexpectedly despite pretokenized Docs: "
                    f"expected {len(words)} tokens, got {len(doc)}."
                )
            context_words = [token.text.lower() for token in doc]
            for row_index, token in zip(row_indices, doc, strict=True):
                values = _token_feature_values(token, context_words)
                if set(values) != set(feature_values):
                    raise AssertionError("Feature extractor and declared feature inventory diverged.")
                for column, value in values.items():
                    feature_values[column][row_index] = value
            progress.update(len(doc))
    finally:
        progress.close()

    for column, values in feature_values.items():
        result[column] = pd.array(values, dtype="string")

    if result["lex_coarse_pos"].notna().sum() == 0:
        raise FrameworkError(
            f"spaCy model {model_name!r} produced no POS annotations. "
            "Use a pipeline that provides POS/tag/morphology."
        )
    return result


def _cache_metadata(
    *,
    input_path: Path,
    input_sha256: str,
    columns: InputColumns,
    model_name: str,
) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "input_columns": {
            "category": columns.category,
            "sent_id": columns.sent_id,
            "sent_len": columns.sent_len,
            "word_index": columns.word_index,
            "word": columns.word,
            "sentence": columns.sentence,
        },
        "spacy_model": model_name,
        "spacy_version": spacy.__version__,
        "lexical_features": [feature.name for feature in LEXICAL_FEATURES],
    }


def load_or_create_annotated_tokens(
    input_path: Path,
    output_dir: Path,
    *,
    columns: InputColumns,
    model_name: str,
    batch_size: int,
    n_process: int,
    refresh_cache: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Return annotated tokens, reusing annotated_tokens.parquet when safe."""

    cache_path = output_dir / "annotated_tokens.parquet"
    meta_path = output_dir / "annotated_tokens.meta.json"
    input_sha256 = sha256_file(input_path, show_progress=show_progress)
    expected_meta = _cache_metadata(
        input_path=input_path,
        input_sha256=input_sha256,
        columns=columns,
        model_name=model_name,
    )

    if not refresh_cache and cache_path.exists() and meta_path.exists():
        try:
            observed_meta = read_json(meta_path)
        except (OSError, ValueError):
            observed_meta = {}
        if observed_meta == expected_meta:
            try:
                cached = pd.read_parquet(cache_path)
            except ImportError as exc:
                raise FrameworkError(
                    "Reading the annotation cache requires pyarrow (recommended)."
                ) from exc
            required = {
                "token_id",
                "category",
                "sent_id",
                "sent_len",
                "word_index",
                "word",
                "sentence",
                *(feature.column for feature in LEXICAL_FEATURES),
            }
            missing = sorted(required.difference(cached.columns))
            if not missing:
                return cached, True

    raw = read_table(input_path)
    canonical = validate_and_canonicalize(raw, columns, show_progress=show_progress)
    annotated = annotate_lexical_features(
        canonical,
        model_name=model_name,
        batch_size=batch_size,
        n_process=n_process,
        show_progress=show_progress,
    )
    write_parquet(annotated, cache_path)
    write_json(expected_meta, meta_path)
    return annotated, False
