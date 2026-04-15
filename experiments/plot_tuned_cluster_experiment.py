"""Plot separate figures for the tuned fixed-K clustering experiment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "tuned_cluster_experiment__fixedk__mcar20__seed42"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    summary = pd.read_csv(TABLES_DIR / "03_method_summary.csv")
    summary_main = summary[summary["method"] != "deletion"].copy()

    heatmap_data = detailed.pivot(index="method", columns="dataset_name", values="cluster_consistency_ari")
    method_order = pd.concat(
        [
            summary[summary["method"] == "deletion"]["method"],
            summary_main.sort_values("overall_score", ascending=False)["method"],
        ]
    )
    heatmap_data = heatmap_data.loc[method_order]

    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=sns.color_palette(["#f4e3c1", "#d8b96a", "#6c8a68", "#1f4d5a"], as_cmap=True),
        annot=True,
        fmt=".2f",
        linewidths=0.7,
        linecolor="#ffffff",
        cbar_kws={"shrink": 0.82, "label": "Cluster-consistency ARI", "pad": 0.02},
    )
    ax.set_title("Tuned Experiment: Cluster Structure Preservation", loc="left", fontsize=20)
    ax.set_xlabel("Dataset", labelpad=10)
    ax.set_ylabel("Method", labelpad=10)
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    heatmap_path = _save(fig, "01_tuned_cluster_structure_heatmap.png")

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    bar_data = summary_main.sort_values("overall_score", ascending=True)
    colors = sns.color_palette("crest", n_colors=len(bar_data))
    ax.barh(bar_data["method"], bar_data["overall_score"], color=colors, edgecolor="#264653", linewidth=0.7)
    ax.set_title("Tuned Experiment: Overall Score Ranking", loc="left", fontsize=20)
    ax.set_xlabel("Composite score (higher is better)", labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    for idx, value in enumerate(bar_data["overall_score"]):
        ax.text(value + 0.008, idx, f"{value:.2f}", va="center", fontsize=10)
    score_path = _save(fig, "02_tuned_overall_score_bar.png")

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    scatter = summary_main.copy()
    bubble_sizes = 240 + 900 * scatter["mean_row_retention"].fillna(0.0)
    points = ax.scatter(
        scatter["mean_formal_runtime_sec"],
        scatter["mean_masked_nrmse"],
        s=bubble_sizes,
        c=scatter["overall_score"],
        cmap="viridis",
        alpha=0.9,
        edgecolors="#1f2933",
        linewidths=0.8,
    )
    offsets = {
        "median": (8, 10),
        "mice": (8, -14),
        "em": (8, 10),
        "knni": (8, 12),
        "mean": (8, -16),
        "iterative_xgboost": (8, 8),
        "missforest": (8, 10),
        "gain": (8, -16),
        "miwae": (8, 10),
        "hivae": (8, -16),
        "nomi": (8, 10),
        "grape": (8, 10),
        "diffputer": (8, -18),
    }
    for _, row in scatter.iterrows():
        dx, dy = offsets.get(row["method"], (6, 6))
        ax.annotate(
            row["method"],
            (row["mean_formal_runtime_sec"], row["mean_masked_nrmse"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "fc": "#ffffff", "ec": "none", "alpha": 0.9},
        )
    ax.set_title("Tuned Experiment: Speed vs NRMSE", loc="left", fontsize=20)
    ax.set_xlabel("Mean runtime (sec)", labelpad=10)
    ax.set_ylabel("Masked NRMSE", labelpad=10)
    cbar = fig.colorbar(points, ax=ax, fraction=0.05, pad=0.035)
    cbar.set_label("Overall score")
    scatter_path = _save(fig, "03_tuned_speed_vs_error_scatter.png")

    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    profile = summary_main[
        ["method", "mean_cluster_consistency_ari", "mean_ari_to_truth", "mean_silhouette"]
    ].copy()
    profile = profile.sort_values("mean_cluster_consistency_ari", ascending=False).head(8)
    profile_long = profile.melt(id_vars="method", var_name="metric", value_name="value")
    sns.barplot(
        data=profile_long,
        x="value",
        y="method",
        hue="metric",
        palette=["#264653", "#2a9d8f", "#e9c46a"],
        ax=ax,
        orient="h",
    )
    ax.set_title("Tuned Experiment: Top Quality Metrics", loc="left", fontsize=20)
    ax.set_xlabel("Metric value", labelpad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(title="Metric", frameon=False, loc="lower right", fontsize=9, title_fontsize=10)
    profile_path = _save(fig, "04_tuned_top_quality_metrics_bar.png")

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    rank_data = summary_main.sort_values("overall_score", ascending=False).reset_index(drop=True)
    ax.plot(rank_data["method"], rank_data["overall_score"], color="#264653", linewidth=2.6, marker="o", markersize=8)
    ax.fill_between(rank_data["method"], rank_data["overall_score"], color="#81b29a", alpha=0.20)
    ax.set_title("Tuned Experiment: Method Ranking Curve", fontsize=20, loc="left")
    ax.set_xlabel("Method")
    ax.set_ylabel("Overall score")
    ax.tick_params(axis="x", rotation=30)
    for idx, row in rank_data.iterrows():
        ax.text(idx, row["overall_score"] + 0.01, f"{row['overall_score']:.2f}", ha="center", fontsize=9)
    rank_path = _save(fig, "05_tuned_method_ranking_curve.png")

    print(f"Saved figure: {heatmap_path}")
    print(f"Saved figure: {score_path}")
    print(f"Saved figure: {scatter_path}")
    print(f"Saved figure: {profile_path}")
    print(f"Saved figure: {rank_path}")


if __name__ == "__main__":
    main()

