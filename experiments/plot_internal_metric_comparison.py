"""Create internal-clustering-metric comparison tables and figures without overall score."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "tuned_cluster_experiment__fixedk__mcar20__seed42"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots_internal_metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INTERNAL_METRICS = [
    ("silhouette", "Silhouette", True),
    ("davies_bouldin", "Davies-Bouldin", False),
    ("calinski_harabasz", "Calinski-Harabasz", True),
]


def _style() -> None:
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "savefig.transparent": False,
            "axes.edgecolor": "#d9d9d9",
            "grid.color": "#e9e9e9",
            "axes.labelcolor": "#2f2a24",
            "text.color": "#2f2a24",
            "xtick.color": "#4a4036",
            "ytick.color": "#4a4036",
            "font.family": "DejaVu Serif",
            "axes.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, file_name: str) -> Path:
    path = PLOTS_DIR / file_name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> None:
    _style()
    detailed = pd.read_csv(TABLES_DIR / "02_formal_experiment_results.csv")

    summary = (
        detailed.groupby("method", dropna=False)
        .agg(
            mean_silhouette=("silhouette", "mean"),
            std_silhouette=("silhouette", "std"),
            mean_davies_bouldin=("davies_bouldin", "mean"),
            std_davies_bouldin=("davies_bouldin", "std"),
            mean_calinski_harabasz=("calinski_harabasz", "mean"),
            std_calinski_harabasz=("calinski_harabasz", "std"),
            mean_cluster_consistency_ari=("cluster_consistency_ari", "mean"),
            mean_formal_runtime_sec=("formal_runtime_sec", "mean"),
            mean_masked_nrmse=("masked_nrmse", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values("mean_silhouette", ascending=False)
    summary_path = TABLES_DIR / "05_internal_metric_summary.csv"
    summary.to_csv(summary_path, index=False)

    ranking = summary[["method"]].copy()
    for metric_key, _, higher_is_better in INTERNAL_METRICS:
        col = f"mean_{metric_key}"
        ranking[f"rank_{metric_key}"] = summary[col].rank(
            ascending=not higher_is_better,
            method="min",
        )
    ranking_path = TABLES_DIR / "06_internal_metric_ranks.csv"
    ranking.to_csv(ranking_path, index=False)

    for metric_key, title, higher_is_better in INTERNAL_METRICS:
        metric_data = detailed.pivot(index="method", columns="dataset_name", values=metric_key)
        order = (
            summary.sort_values(f"mean_{metric_key}", ascending=not higher_is_better)["method"]
            if higher_is_better
            else summary.sort_values(f"mean_{metric_key}", ascending=True)["method"]
        )
        metric_data = metric_data.loc[order]

        fig, ax = plt.subplots(figsize=(11.6, 8.0))
        sns.heatmap(
            metric_data,
            ax=ax,
            cmap=sns.color_palette(["#f8ecd1", "#d9b96f", "#7da27b", "#1f4d5a"], as_cmap=True),
            annot=True,
            fmt=".2f",
            linewidths=0.7,
            linecolor="#ffffff",
            cbar_kws={"shrink": 0.82, "label": title, "pad": 0.02},
        )
        ax.set_title(f"Internal Metric Comparison: {title}", loc="left", fontsize=20)
        ax.set_xlabel("Dataset", labelpad=10)
        ax.set_ylabel("Method", labelpad=10)
        ax.tick_params(axis="x", rotation=30, labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        _save(fig, f"{metric_key}_heatmap.png")

        fig, ax = plt.subplots(figsize=(10.8, 7.2))
        bar_data = summary.sort_values(
            f"mean_{metric_key}",
            ascending=not higher_is_better,
        )
        colors = sns.color_palette("crest", n_colors=len(bar_data))
        values = bar_data[f"mean_{metric_key}"]
        ax.barh(bar_data["method"], values, color=colors, edgecolor="#264653", linewidth=0.7)
        ax.set_title(f"Method Ranking by {title}", loc="left", fontsize=20)
        ax.set_xlabel(f"Mean {title}", labelpad=10)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=11)
        for idx, value in enumerate(values):
            ax.text(value, idx, f"  {value:.2f}", va="center", fontsize=10)
        _save(fig, f"{metric_key}_bar.png")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), sharey=True)
    for ax, (metric_key, title, higher_is_better) in zip(axes, INTERNAL_METRICS):
        metric_summary = summary.sort_values(
            f"mean_{metric_key}",
            ascending=not higher_is_better,
        )
        sns.barplot(
            data=metric_summary,
            x=f"mean_{metric_key}",
            y="method",
            ax=ax,
            palette="crest",
            orient="h",
        )
        ax.set_title(title, loc="left", fontsize=16)
        ax.set_xlabel("Mean value")
        ax.set_ylabel("" if ax is not axes[0] else "Method")
        ax.tick_params(axis="y", labelsize=10.5)
        for idx, value in enumerate(metric_summary[f"mean_{metric_key}"]):
            ax.text(value, idx, f"  {value:.2f}", va="center", fontsize=9)
    panel_path = _save(fig, "internal_metrics_panel.png")

    print(f"Wrote table: {summary_path}")
    print(f"Wrote table: {ranking_path}")
    print(f"Saved figure: {panel_path}")
    for metric_key, _, _ in INTERNAL_METRICS:
        print(f"Saved figure: {PLOTS_DIR / f'{metric_key}_heatmap.png'}")
        print(f"Saved figure: {PLOTS_DIR / f'{metric_key}_bar.png'}")


if __name__ == "__main__":
    main()
