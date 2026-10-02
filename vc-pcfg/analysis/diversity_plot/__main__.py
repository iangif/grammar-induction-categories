"""Plot lexical diversity vs contextual diversity, colored by coherence.

Run from vc-pcfg/ with:
    python -m analysis.diversity_plot \
        --input analysis/outputs/abstraction/s91-e5-c60/category_metrics.csv \
        --output-dir analysis/outputs/diversity_plot/s91-e5-c60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTEXT_COLUMN = {1: "CD1_eff", 2: "CD2_eff"}
COHERENCE_COLUMNS = {
    "all": ["LC", "CC2"],
    "lexical": ["LC"],
    "contextual": ["CC2"],
}


def load_metrics(path: Path, window: int, coherence: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    x_col = CONTEXT_COLUMN[window]
    needed = ["category", "LD_eff", x_col, *COHERENCE_COLUMNS[coherence]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    out = pd.DataFrame(
        {
            "category": df["category"],
            "lexical_diversity": df["LD_eff"].astype(float),
            "contextual_diversity": df[x_col].astype(float),
            "coherence": df[COHERENCE_COLUMNS[coherence]].astype(float).max(axis=1),
        }
    )
    lx = out["contextual_diversity"].median()
    ly = out["lexical_diversity"].median()
    out["quadrant"] = np.where(
        out["lexical_diversity"] >= ly, "highLD", "lowLD"
    ) + np.where(out["contextual_diversity"] >= lx, "_highCD", "_lowCD")
    return out


def plot_diversity(
    data: pd.DataFrame,
    path: Path,
    *,
    window: int,
    coherence: str,
    log: bool,
    labels: bool,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(
        data["contextual_diversity"],
        data["lexical_diversity"],
        c=data["coherence"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=70,
        edgecolor="#444444",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.axvline(data["contextual_diversity"].median(), color="gray", ls="--", lw=0.8)
    ax.axhline(data["lexical_diversity"].median(), color="gray", ls="--", lw=0.8)

    if labels:
        for row in data.itertuples(index=False):
            ax.annotate(
                str(row.category),
                (row.contextual_diversity, row.lexical_diversity),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(f"Contextual diversity (effective, window={window})")
    ax.set_ylabel("Lexical diversity (effective)")
    ax.set_title("Induced categories: lexical vs contextual diversity")
    fig.colorbar(points, ax=ax).set_label(f"Coherence ({coherence})")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="category_metrics.csv"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window", type=int, choices=[1, 2], default=1)
    parser.add_argument("--coherence", choices=list(COHERENCE_COLUMNS), default="all")
    parser.add_argument("--log", action="store_true", help="Log-scale both axes.")
    parser.add_argument("--labels", action="store_true", help="Annotate category ids.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        data = load_metrics(args.input, args.window, args.coherence)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "diversity_plot_data.csv", index=False)
    plot_diversity(
        data,
        args.output_dir / "diversity_plot.png",
        window=args.window,
        coherence=args.coherence,
        log=args.log,
        labels=args.labels,
        dpi=args.dpi,
    )
    print(f"Plotted {len(data)} categories. Wrote outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
