"""End-to-end abstraction-framework pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .coherence import FeatureRef, score_modal_coherence
from .constants import CONTEXT_RADIUS, LEXICAL_FEATURES, PACKAGE_VERSION
from .context import build_contextual_vectors, contextual_feature_refs
from .diversity import compute_diversity
from .features import load_or_create_annotated_tokens
from .io import InputColumns, write_csv, write_json
from .profiles import build_category_feature_distributions

_DEPRECATED_OUTPUTS = (
    "diversity.csv",
    "lexical_type_counts.parquet",
    "context_frame_counts_r1.parquet",
    "context_frame_counts_r2.parquet",
    "lexical_coherence.csv",
    "lexical_feature_scores.parquet",
    "contextual_vectors_r2.parquet",
    "contextual_coherence_r2.csv",
    "contextual_feature_scores_r2.parquet",
)


def _lexical_refs() -> list[FeatureRef]:
    return [FeatureRef(name=feature.name, column=feature.column) for feature in LEXICAL_FEATURES]


def _rename_coherence(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return summary.rename(
        columns={
            "token_count": f"{prefix}_token_count",
            "coherence": prefix,
            "winning_feature": f"{prefix}_feature",
            "winning_value": f"{prefix}_value",
            "winning_modal_count": f"{prefix}_modal_count",
            "winning_modal_coverage": f"{prefix}_modal_coverage",
            "winning_corpus_coverage": f"{prefix}_corpus_coverage",
        }
    )


def _remove_deprecated_outputs(output_dir: Path) -> None:
    """Remove old report files so reruns expose only the current public outputs."""

    for name in _DEPRECATED_OUTPUTS:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def run_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    columns: InputColumns,
    spacy_model: str,
    spacy_batch_size: int,
    spacy_processes: int,
    refresh_cache: bool,
    show_progress: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_deprecated_outputs(output_dir)

    # Validate + canonicalize input, then annotate/cache lexical features.
    annotated, cache_hit = load_or_create_annotated_tokens(
        input_path,
        output_dir,
        columns=columns,
        model_name=spacy_model,
        batch_size=spacy_batch_size,
        n_process=spacy_processes,
        refresh_cache=refresh_cache,
        show_progress=show_progress,
    )

    # Build the radius-2 neighborhood once and reuse it for diversity,
    # contextual coherence, and the long category-feature distributions.
    contexts = build_contextual_vectors(annotated, show_progress=show_progress)

    # Existing category-level diversity metrics.
    diversity = compute_diversity(annotated, contexts, show_progress=show_progress)

    # Existing lexical and contextual coherence metrics.
    lexical_summary, _ = score_modal_coherence(
        annotated,
        category_column="category",
        features=_lexical_refs(),
        show_progress=show_progress,
        progress_desc="Scoring lexical coherence",
    )
    lexical_summary = _rename_coherence(lexical_summary, "LC")

    contextual_summary, _ = score_modal_coherence(
        contexts,
        category_column="category",
        features=contextual_feature_refs(),
        show_progress=show_progress,
        progress_desc="Scoring contextual coherence",
    )
    contextual_summary = _rename_coherence(contextual_summary, "CC2")

    # Long-format source of truth for all lexical and r=2 contextual feature
    # distributions. Keep full floating-point precision here for downstream
    # Hellinger/clustering/PCA analyses.
    distributions = build_category_feature_distributions(
        annotated,
        contexts,
        show_progress=show_progress,
    )
    distributions = distributions.sort_values("category", kind="stable").reset_index(drop=True)
    write_csv(distributions, output_dir / "category_feature_distributions.csv")

    # One-row-per-category summary table. Preserve the existing two-decimal
    # reporting convention without changing internal calculation precision.
    category_metrics = diversity.merge(lexical_summary, on="category", how="outer")
    category_metrics = category_metrics.merge(contextual_summary, on="category", how="outer")
    category_metrics = category_metrics.drop(columns=["LC_token_count", "CC2_token_count"], errors="ignore")
    category_metrics = category_metrics.sort_values("category", kind="stable").reset_index(drop=True)
    write_csv(category_metrics, output_dir / "category_metrics.csv", float_format="%.2f")

    manifest = {
        "package_version": PACKAGE_VERSION,
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "annotation_cache_hit": cache_hit,
        "spacy_model": spacy_model,
        "context_radius_for_coherence": CONTEXT_RADIUS,
        "public_outputs": [
            "category_feature_distributions.csv",
            "category_metrics.csv",
        ],
        "cache_outputs": [
            "annotated_tokens.parquet",
            "annotated_tokens.meta.json",
        ],
        "lexical_features": [
            {"name": feature.name, "column": feature.column, "source": feature.source}
            for feature in LEXICAL_FEATURES
        ],
        "feature_distribution_schema": {
            "position": "TARGET for lexical features; L2/L1/R1/R2 for contextual features",
            "proportion": "count divided by all tokens in the category",
            "null_handling": "Genuine missing feature values are emitted as literal NULL.",
            "boundary_handling": "Out-of-sentence contextual slots are <BOS>/<EOS>.",
            "unobserved_values": "Omitted; downstream matrix construction should fill them with zero.",
        },
        "wordnet_semantics": {
            "noun_and_verb_classes": "WordNet lexname/supersense of a contextual Lesk-selected sense",
            "adjective_class": "specific WordNet noun attribute synset linked to the contextual adjective sense",
            "adverb_class": "specific noun attribute reached through the contextual adverb sense's adjective pertainym",
        },
        "tie_breaking": (
            "Modal-value ties use lexicographic string order. Feature-score ties prefer "
            "higher modal coverage, then modal count, then the declared feature order."
        ),
    }
    write_json(manifest, output_dir / "run_manifest.json")
    return {
        "rows": len(annotated),
        "categories": int(annotated["category"].nunique()),
        "cache_hit": cache_hit,
        "category_metrics": category_metrics,
        "category_feature_distributions": distributions,
    }
