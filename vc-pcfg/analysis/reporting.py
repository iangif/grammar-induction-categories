import argparse
import json
from pathlib import Path
from typing import Any
from typing import Sequence

import pandas as pd

from .constants.columns import EVIDENCE_EXAMPLE_COLUMNS
from .evidence import build_standardized_evidence_packet
from .examples import build_representative_examples, build_llm_input
from .io import write_csv
from .io import write_json
from .ordering import (
    average_metric_order,
    hierarchical_metric_order,
    reorder_square_matrix,
)
from .overlap import (
    cosine_similarity_matrix,
    js_divergence_matrix,
    top_k_overlap_matrix,
    labeled_matrix,
    build_redundancy_summary,
)
from .plots import (
    save_position_plot,
    save_top_bar_plot,
    save_rank_distribution_plot,
    save_square_heatmap,
)


def write_per_category_outputs(
    output_dir: Path,
    category_ids: Sequence[int],
    summary: pd.DataFrame,
    word_category: pd.DataFrame,
    word_scores: pd.DataFrame,
    word_ambiguity: pd.DataFrame,
    word_pos_category: pd.DataFrame,
    category_pos_distribution: pd.DataFrame,
    frequent_words: pd.DataFrame,
    diagnostic_words: pd.DataFrame,
    previous_words: pd.DataFrame,
    next_words: pd.DataFrame,
    frames: pd.DataFrame,
    positions: pd.DataFrame,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    combined_llm_inputs: list[pd.DataFrame] = []
    combined_evidence_examples: list[pd.DataFrame] = []
    evidence_packets: list[dict[str, Any]] = []
    evidence_packet_index: list[dict[str, Any]] = []
    width = max(2, len(str(max(category_ids))) if category_ids else 2)

    for category in category_ids:
        category_dir = output_dir / "categories" / f"category_{category:0{width}d}"
        category_dir.mkdir(parents=True, exist_ok=True)

        category_summary = summary.loc[summary["c"] == category]
        category_word_distribution = word_category.loc[word_category["c"] == category]
        category_frequent = frequent_words.loc[frequent_words["c"] == category]
        category_diagnostic = diagnostic_words.loc[diagnostic_words["c"] == category]
        category_previous = previous_words.loc[previous_words["c"] == category]
        category_next = next_words.loc[next_words["c"] == category]
        category_frames = frames.loc[frames["c"] == category]
        category_positions = positions.loc[positions["c"] == category]

        write_csv(category_summary, category_dir / "summary.csv")
        write_csv(category_word_distribution, category_dir / "word_distribution.csv")
        write_csv(category_frequent, category_dir / "frequent_words.csv")
        write_csv(category_diagnostic, category_dir / "diagnostic_words.csv")
        write_csv(category_previous, category_dir / "previous_words.csv")
        write_csv(category_next, category_dir / "next_words.csv")
        write_csv(category_frames, category_dir / "context_frames.csv")
        write_csv(category_positions, category_dir / "position_distribution.csv")

        examples = build_representative_examples(
            df=df,
            category=category,
            frequent_words=frequent_words,
            diagnostic_words=diagnostic_words,
            frame_rankings=frames,
            examples_per_word=args.examples_per_word,
            random_examples=args.random_examples,
            top_frames_for_examples=args.top_frames_for_examples,
            examples_per_frame=args.examples_per_frame,
            random_seed=args.random_seed,
        )
        write_csv(examples, category_dir / "representative_sentences.csv")

        llm_input = build_llm_input(
            df=df, category=category, max_rows=args.llm_input_max_rows
        )
        write_csv(llm_input, category_dir / "llm_input.csv")
        if not llm_input.empty:
            combined = llm_input.copy()
            combined.insert(0, "c", category)
            combined_llm_inputs.append(combined)

        evidence_packet, evidence_examples = build_standardized_evidence_packet(
            df=df,
            category=category,
            summary=summary,
            word_scores=word_scores,
            word_ambiguity=word_ambiguity,
            word_pos_category=word_pos_category,
            category_pos_distribution=category_pos_distribution,
            previous_words=previous_words,
            next_words=next_words,
            frames=frames,
            positions=positions,
            args=args,
        )
        packet_path = category_dir / "evidence_packet.json"
        example_path = category_dir / "evidence_packet_examples.csv"
        write_json(evidence_packet, packet_path)
        write_csv(evidence_examples, example_path)
        evidence_packets.append(evidence_packet)
        if not evidence_examples.empty:
            combined_evidence_examples.append(evidence_examples)
        evidence_packet_index.append(
            {
                "c": category,
                "packet_path": str(packet_path.relative_to(output_dir)),
                "examples_path": str(example_path.relative_to(output_dir)),
                "target_example_count": int(
                    evidence_examples["example_role"].eq("target_category").sum()
                ),
                "contrast_example_count": int(
                    evidence_examples["example_role"]
                    .eq("cross_category_contrast")
                    .sum()
                ),
                "ambiguous_word_profile_count": len(
                    evidence_packet["ambiguous_word_profiles"]
                ),
            }
        )

        if args.skip_plots:
            continue

        plots_dir = category_dir / "plots"
        save_top_bar_plot(
            category_frequent,
            "count",
            plots_dir / "top_frequent_words.png",
            f"Category {category}: most frequent words",
            "Token count",
        )
        save_top_bar_plot(
            category_diagnostic,
            "weighted_log_odds",
            plots_dir / "top_diagnostic_words.png",
            f"Category {category}: diagnostic words",
            "Weighted log-odds z-score",
        )
        save_rank_distribution_plot(
            category_word_distribution["p_word_given_category"],
            plots_dir / "p_word_given_category_distribution.png",
            f"Category {category}: P(word | category)",
            "P(word | category)",
        )
        eligible_diagnostic_distribution = category_word_distribution.loc[
            category_word_distribution["w"].isin(category_diagnostic["w"])
        ]
        save_rank_distribution_plot(
            eligible_diagnostic_distribution["p_category_given_word"],
            plots_dir / "p_category_given_word_distribution.png",
            f"Category {category}: P(category | word), diagnostic words",
            "P(category | word)",
        )
        save_position_plot(
            category_positions,
            plots_dir / "normalized_position_distribution.png",
            f"Category {category}: normalized sentence position",
        )

    evidence_dir = output_dir / "evidence_packets"
    write_csv(
        pd.DataFrame(evidence_packet_index), evidence_dir / "evidence_packet_index.csv"
    )
    if combined_evidence_examples:
        all_evidence_examples = pd.concat(combined_evidence_examples, ignore_index=True)
    else:
        all_evidence_examples = pd.DataFrame(columns=EVIDENCE_EXAMPLE_COLUMNS)
    write_csv(
        all_evidence_examples, evidence_dir / "evidence_examples_all_categories.csv"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with (evidence_dir / "evidence_packets.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for packet in evidence_packets:
            handle.write(json.dumps(packet, ensure_ascii=False) + "\n")

    if combined_llm_inputs:
        return pd.concat(combined_llm_inputs, ignore_index=True)
    return pd.DataFrame(columns=["c", "word", "sentence"])


def write_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    manifest: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_categories": args.num_categories,
        "spacy_model": args.spacy_model,
        "top_n_words": args.top_n_words,
        "min_diagnostic_count": args.min_diagnostic_count,
        "log_odds_prior_strength": args.log_odds_prior_strength,
        "top_contexts": args.top_contexts,
        "position_bins": args.position_bins,
        "top_k_overlap": args.top_k_overlap,
        "cluster_linkage": args.cluster_linkage,
        "llm_input_max_rows": args.llm_input_max_rows,
        "evidence_packet": {
            "schema_version": "1.0",
            "frequent_words": args.evidence_frequent_words,
            "diagnostic_words": args.evidence_diagnostic_words,
            "examples_per_word": args.evidence_examples_per_word,
            "random_examples": args.evidence_random_examples,
            "ambiguous_words": args.evidence_ambiguous_words,
            "examples_per_ambiguous_word": args.evidence_examples_per_ambiguous_word,
            "contrast_examples_per_ambiguous_word": (
                args.evidence_contrast_examples_per_ambiguous_word
            ),
            "rare_examples": args.evidence_rare_examples,
            "rare_max_count": args.evidence_rare_max_count,
            "top_contexts": args.evidence_top_contexts,
            "top_frames": args.evidence_top_frames,
            "top_word_pos_units": args.evidence_top_word_pos_units,
        },
        "random_seed": args.random_seed,
        "metric_definitions": {
            "word_entropy": "Natural-log entropy of P(word | category), measured in nats.",
            "top_n_coverage": (
                "Share of category tokens accounted for by the N most frequent word types; "
                "reported for N = 1, 5, and 10."
            ),
            "evidence_packet": (
                "A standardized JSON record containing category summary statistics, lexical "
                "rankings, POS/tag and context distributions, ambiguity profiles, balanced "
                "target-category examples, and explicitly labeled cross-category contrasts."
            ),
            "mean_sentence_length": (
                "Mean length of distinct sentences containing the category; each sentence "
                "is counted once per category."
            ),
            "top_k_overlap": "|TopK(c1) intersection TopK(c2)| / K.",
            "js_divergence": "Jensen-Shannon divergence in bits; 0 means identical distributions.",
            "weighted_log_odds": (
                "Variance-normalized difference between a word's log odds inside one category "
                "and in all remaining categories, with a corpus-informed Dirichlet prior."
            ),
            "word_pos_for_matrix_order": (
                "Dominant context-sensitive spaCy coarse POS among all occurrences of each word type."
            ),
            "word_pos_unit": (
                "A context-sensitive lexical unit formed as surface_word.spacy_POS, "
                "for example play.VERB versus play.NOUN."
            ),
            "average_metric_order": (
                "Categories sorted from most redundant to most distinctive by mean "
                "off-diagonal cosine similarity or top-k overlap; for Jensen-Shannon "
                "divergence the order is lowest mean divergence to highest."
            ),
            "hierarchical_order": (
                "Average/complete/single/weighted-linkage ordering over 1-cosine, "
                "1-top-k-overlap, or sqrt(Jensen-Shannon divergence)."
            ),
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_overlap_analysis(
    matrix: pd.DataFrame,
    output_dir: Path,
    representation_slug: str,
    representation_title: str,
    category_ids: Sequence[int],
    token_counts: pd.Series,
    top_k_value: int,
    cluster_linkage: str,
    skip_plots: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Write overlap tables, redundancy summaries, orderings, and heatmaps."""
    representation_dir = output_dir / "matrices" / representation_slug
    overlap_dir = representation_dir / "overlap"
    ordering_dir = representation_dir / "orderings"
    heatmap_dir = representation_dir / "heatmaps"

    values = matrix.to_numpy(dtype=float)
    cosine_values = cosine_similarity_matrix(values)
    js_values = js_divergence_matrix(values)
    top_k_values = top_k_overlap_matrix(values, top_k_value)

    metrics: dict[str, tuple[pd.DataFrame, bool, str, str, str]] = {
        "cosine_similarity": (
            labeled_matrix(cosine_values, category_ids),
            True,
            "cosine",
            "Cosine similarity",
            "Cosine similarity",
        ),
        "js_divergence": (
            labeled_matrix(js_values, category_ids),
            False,
            "js",
            "Jensen-Shannon divergence",
            "J-S divergence (bits)",
        ),
        f"top_{top_k_value}_overlap": (
            labeled_matrix(top_k_values, category_ids),
            True,
            "top_k",
            f"Top-{top_k_value} lexical-unit overlap",
            "Intersection / K",
        ),
    }

    output_matrices: dict[str, pd.DataFrame] = {}
    for metric_slug, (
        metric_matrix,
        higher_is_more_redundant,
        metric_kind,
        metric_title,
        colorbar_label,
    ) in metrics.items():
        output_matrices[metric_slug] = metric_matrix
        write_csv(metric_matrix.reset_index(), overlap_dir / f"{metric_slug}.csv")

        average_order, average_table = average_metric_order(
            metric_matrix.to_numpy(dtype=float),
            category_ids,
            higher_is_more_redundant=higher_is_more_redundant,
        )
        average_table.insert(1, "representation", representation_slug)
        average_table.insert(2, "metric", metric_slug)
        write_csv(
            average_table,
            ordering_dir / f"{metric_slug}_average_metric_order.csv",
        )

        cluster_order, cluster_table = hierarchical_metric_order(
            metric_matrix.to_numpy(dtype=float),
            category_ids,
            metric_kind=metric_kind,
            linkage_method=cluster_linkage,
        )
        cluster_table.insert(1, "representation", representation_slug)
        cluster_table.insert(2, "metric", metric_slug)
        cluster_table.insert(3, "linkage", cluster_linkage)
        write_csv(
            cluster_table,
            ordering_dir / f"{metric_slug}_hierarchical_order.csv",
        )

        if skip_plots:
            continue

        save_square_heatmap(
            metric_matrix,
            heatmap_dir / f"{metric_slug}_original_order.png",
            f"{representation_title}: {metric_title} (category-number order)",
            colorbar_label,
        )
        save_square_heatmap(
            reorder_square_matrix(metric_matrix, average_order),
            heatmap_dir / f"{metric_slug}_average_metric_order.png",
            (
                f"{representation_title}: {metric_title} "
                "(most redundant to most distinctive)"
            ),
            colorbar_label,
        )
        save_square_heatmap(
            reorder_square_matrix(metric_matrix, cluster_order),
            heatmap_dir / f"{metric_slug}_hierarchical_order.png",
            (
                f"{representation_title}: {metric_title} "
                f"(hierarchical order, {cluster_linkage} linkage)"
            ),
            colorbar_label,
        )

    redundancy_summary = build_redundancy_summary(
        category_ids=category_ids,
        token_counts=token_counts,
        cosine=cosine_values,
        js=js_values,
        top_k=top_k_values,
    )
    write_csv(redundancy_summary, representation_dir / "redundancy_summary.csv")
    return output_matrices, redundancy_summary
