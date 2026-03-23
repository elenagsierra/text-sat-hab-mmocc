#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "fire",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Summarize and visualize pixel-level GRAFT zero-shot LULC classification results."""

from __future__ import annotations

import json

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sat_mmocc.experiments.glc_fcs30d_graft.utils import (
    DEFAULT_GLC_YEAR,
    DEFAULT_IMAGERY_SOURCE,
    get_run_dir,
)


def analyze_results(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    year: int = DEFAULT_GLC_YEAR,
    checkpoint_level: str = "pixel",
    top_n_confusions: int = 20,
    top_n_classes_for_heatmap: int = 15,
) -> str:
    run_dir = get_run_dir(
        imagery_source=imagery_source,
        year=year,
        checkpoint_level=checkpoint_level,
    )
    scoring_dir = run_dir / "text_scoring"
    confusion_path = scoring_dir / "confusion_matrix.csv"
    site_metrics_path = scoring_dir / "site_metrics.csv"
    summary_path = scoring_dir / "summary.json"
    if not confusion_path.exists() or not site_metrics_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Missing text-scoring outputs in {scoring_dir}. "
            "Run step_03_score_lulc_text.py first."
        )

    confusion_df = pd.read_csv(confusion_path, index_col=0)
    site_metrics_df = pd.read_csv(site_metrics_path)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    support = confusion_df.sum(axis=1)
    predicted_support = confusion_df.sum(axis=0)
    true_positive = np.diag(confusion_df.to_numpy())
    precision = np.divide(
        true_positive,
        predicted_support.to_numpy(),
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_support.to_numpy() > 0,
    )
    recall = np.divide(
        true_positive,
        support.to_numpy(),
        out=np.zeros_like(true_positive, dtype=float),
        where=support.to_numpy() > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )

    per_class_df = pd.DataFrame(
        {
            "class_name": confusion_df.index,
            "support": support.to_numpy(dtype=int),
            "predicted_support": predicted_support.to_numpy(dtype=int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ).sort_values(["support", "recall"], ascending=[False, False])
    per_class_df.to_csv(scoring_dir / "per_class_metrics.csv", index=False)

    off_diag = confusion_df.copy()
    np.fill_diagonal(off_diag.values, 0)
    confusion_pairs = (
        off_diag.stack()
        .rename("count")
        .reset_index()
        .rename(columns={"level_0": "true_class", "level_1": "pred_class"})
        .sort_values("count", ascending=False)
    )
    top_confusions_df = confusion_pairs[confusion_pairs["count"] > 0].head(top_n_confusions)
    top_confusions_df.to_csv(scoring_dir / "top_confusions.csv", index=False)

    sns.set_theme(style="whitegrid")

    top_supported = per_class_df.head(top_n_classes_for_heatmap)["class_name"].tolist()
    heatmap_df = confusion_df.loc[top_supported, top_supported]
    row_sums = heatmap_df.sum(axis=1).replace(0, np.nan)
    heatmap_norm = heatmap_df.div(row_sums, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_norm, cmap="mako", ax=ax)
    ax.set_title("Top-supported class confusion (row-normalized)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(scoring_dir / "confusion_heatmap_top_classes.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(site_metrics_df["token_accuracy"].dropna(), bins=20, ax=ax, color="#3b6fb6")
    ax.set_title("Per-site token accuracy")
    ax.set_xlabel("Token accuracy")
    ax.set_ylabel("Number of sites")
    fig.tight_layout()
    fig.savefig(scoring_dir / "site_accuracy_histogram.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = per_class_df.head(top_n_classes_for_heatmap).sort_values("recall", ascending=True)
    ax.barh(plot_df["class_name"], plot_df["recall"], color="#4d936f")
    ax.set_title("Recall for top-supported classes")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Class")
    fig.tight_layout()
    fig.savefig(scoring_dir / "per_class_recall_top_supported.png", dpi=200)
    plt.close(fig)

    analysis_summary = {
        **summary,
        "num_classes_with_support": int((per_class_df["support"] > 0).sum()),
        "mean_site_accuracy": float(site_metrics_df["token_accuracy"].mean()),
        "median_site_accuracy": float(site_metrics_df["token_accuracy"].median()),
        "best_supported_class": (
            per_class_df[per_class_df["support"] > 0]
            .sort_values(["recall", "support"], ascending=[False, False])
            .iloc[0]["class_name"]
            if (per_class_df["support"] > 0).any()
            else None
        ),
    }
    with open(scoring_dir / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(analysis_summary, handle, indent=2)

    print(f"Wrote analysis outputs to {scoring_dir}")
    return str(scoring_dir)


if __name__ == "__main__":
    fire.Fire(analyze_results)
