"""Create RMSE / NRMSE ranking tables and figures from existing formal results."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "tuned_cluster_experiment__fixedk__mcar20__seed42"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots_rmse"
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

    summary = (
        detailed.groupby("method", dropna=False)
        .agg(
            mean_masked_rmse=("masked_rmse", "mean"),
            std_masked_rmse=("masked_rmse", "std"),
            mean_masked_nrmse=("masked_nrmse", "mean"),
            std_masked_nrmse=("masked_nrmse", "std"),
            mean_formal_runtime_sec=("formal_runtime_sec", "mean"),
            mean_cluster_consistency_ari=("cluster_consistency_ari", "mean"),
        )
        .reset_index()
        .sort_values("mean_masked_rmse", ascending=True)
    )
    summary["rank_rmse"] = summary["mean_masked_rmse"].rank(method="min", ascending=True)
    summary["rank_nrmse"] = summary["mean_masked_nrmse"].rank(method="min", ascending=True)
    summary_path = TABLES_DIR / "07_rmse_summary.csv"
    summary.to_csv(summary_path, index=False)

    rmse_heatmap = detailed.pivot(index="method", columns="dataset_name", values="masked_rmse")
    rmse_heatmap = rmse_heatmap.loc[summary["method"]]
    fig, ax = plt.subplots(figsize=(11.6, 8.0))
    sns.heatmap(
        rmse_heatmap,
        ax=ax,
        cmap=sns.color_palette(["#f8ecd1", "#d9b96f", "#7da27b", "#1f4d5a"], as_cmap=True),
        annot=True,
        fmt=".2f",
        linewidths=0.7,
        linecolor="#ffffff",
        cbar_kws={"shrink": 0.82, "label": "Masked RMSE", "pad": 0.02},
    )
    ax.set_title("Masked RMSE by Dataset and Method", loc="left", fontsize=20)
    ax.set_xlabel("Dataset", labelpad=10)
    ax.set_ylabel("Method", labelpad=10)
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    rmse_heatmap_path = _save(fig, "rmse_heatmap.png")

    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    bar_data = summary.sort_values("mean_masked_rmse", ascending=True)
    colors = sns.color_palette("crest_r", n_colors=len(bar_data))
    ax.barh(bar_data["method"], bar_data["mean_masked_rmse"], color=colors, edgecolor="#264653", linewidth=0.7)
    ax.set_title("Method Ranking by Masked RMSE", loc="left", fontsize=20)
    ax.set_xlabel("Mean masked RMSE (lower is better)", labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    for idx, value in enumerate(bar_data["mean_masked_rmse"]):
        ax.text(value, idx, f"  {value:.2f}", va="center", fontsize=10)
    rmse_bar_path = _save(fig, "rmse_bar.png")

    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    nrmse_data = summary.sort_values("mean_masked_nrmse", ascending=True)
    colors = sns.color_palette("mako_r", n_colors=len(nrmse_data))
    ax.barh(nrmse_data["method"], nrmse_data["mean_masked_nrmse"], color=colors, edgecolor="#264653", linewidth=0.7)
    ax.set_title("Method Ranking by Masked NRMSE", loc="left", fontsize=20)
    ax.set_xlabel("Mean masked NRMSE (lower is better)", labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    for idx, value in enumerate(nrmse_data["mean_masked_nrmse"]):
        ax.text(value, idx, f"  {value:.2f}", va="center", fontsize=10)
    nrmse_bar_path = _save(fig, "nrmse_bar.png")

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    scatter = summary.copy()
    bubble_sizes = 220 + 950 * scatter["mean_cluster_consistency_ari"].fillna(0.0)
    points = ax.scatter(
        scatter["mean_masked_rmse"],
        scatter["mean_formal_runtime_sec"],
        s=bubble_sizes,
        c=scatter["mean_masked_nrmse"],
        cmap="viridis_r",
        alpha=0.9,
        edgecolors="#1f2933",
        linewidths=0.8,
    )
    for _, row in scatter.iterrows():
        ax.annotate(
            row["method"],
            (row["mean_masked_rmse"], row["mean_formal_runtime_sec"]),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "fc": "#ffffff", "ec": "none", "alpha": 0.92},
        )
    ax.set_title("RMSE vs Runtime", loc="left", fontsize=20)
    ax.set_xlabel("Mean masked RMSE", labelpad=10)
    ax.set_ylabel("Mean runtime (sec)", labelpad=10)
    cbar = fig.colorbar(points, ax=ax, fraction=0.05, pad=0.035)
    cbar.set_label("Mean masked NRMSE")
    scatter_path = _save(fig, "rmse_runtime_scatter.png")

    print(f"Wrote table: {summary_path}")
    print(f"Saved figure: {rmse_heatmap_path}")
    print(f"Saved figure: {rmse_bar_path}")
    print(f"Saved figure: {nrmse_bar_path}")
    print(f"Saved figure: {scatter_path}")


if __name__ == "__main__":
    main()
