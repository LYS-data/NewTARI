"""Create a process illustration for imputation-driven clustering recommendation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "paper_figures"
    / "example_pipeline_process__twomoons__mcar10"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from datasets import DatasetLoader
from experiments.cluster_experiment_utils import inject_mcar_missing
from imputers.iterative import IterativeXGBoostImputer
from imputers.simple import MeanImputer


RANDOM_SEED = 42
MISSING_RATE = 0.10
DBSCAN_EPS = 0.30
DBSCAN_MIN_SAMPLES = 5
ZOOM_REGION = (0.05, 1.55, -0.15, 0.95)
COLORS = {
    "cluster_1": "#2a9d8f",
    "cluster_2": "#e76f51",
    "neutral": "#94a3b8",
    "edge": "#334155",
    "highlight": "#f4a261",
    "panel": "#ffffff",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "savefig.transparent": False,
            "font.family": "DejaVu Serif",
            "axes.titleweight": "bold",
            "axes.titlesize": 12.8,
            "text.color": "#24313f",
            "axes.labelcolor": "#24313f",
        }
    )


def fit_dbscan_labels(X: np.ndarray) -> np.ndarray:
    X_scaled = StandardScaler().fit_transform(X)
    return DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X_scaled)


def align_binary_labels(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if set(np.unique(reference)) != {0, 1}:
        return labels
    if set(np.unique(labels)) - {-1} != {0, 1}:
        return labels
    valid = labels >= 0
    if not np.any(valid):
        return labels
    original = adjusted_rand_score(reference[valid], labels[valid])
    flipped = np.where(labels == -1, -1, 1 - labels)
    flipped_score = adjusted_rand_score(reference[valid], flipped[valid])
    return flipped if flipped_score > original else labels


def add_panel_frame(ax: plt.Axes, edge_color: str) -> None:
    frame = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.25,
        edgecolor=edge_color,
        facecolor="#ffffff",
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.add_patch(frame)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scatter_clusters(ax: plt.Axes, X: np.ndarray, labels: np.ndarray, alpha: float = 0.92) -> None:
    for cluster_id, color in [(0, COLORS["cluster_1"]), (1, COLORS["cluster_2"])]:
        mask = labels == cluster_id
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=26,
            c=color,
            edgecolors=COLORS["edge"],
            linewidths=0.25,
            alpha=alpha,
            zorder=2,
        )
    noise = labels == -1
    if np.any(noise):
        ax.scatter(
            X[noise, 0],
            X[noise, 1],
            s=22,
            c=COLORS["neutral"],
            edgecolors="none",
            alpha=0.7,
            zorder=1,
        )
    clean_axis(ax)
    ax.set_aspect("equal", adjustable="box")


def add_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float], text: str) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.4,
        color="#6b7280",
    )
    fig.add_artist(arrow)
    fig.text(
        (start[0] + end[0]) / 2,
        start[1] + 0.02,
        text,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color="#4b5563",
    )


def main() -> None:
    configure_style()

    loader = DatasetLoader(data_root=PROJECT_ROOT / "data" / "raw")
    bundle = loader.load_csv("twomoons.csv", dataset_name="twomoons", label_column="target")
    X_complete = bundle.X
    y_true = bundle.y.to_numpy(dtype=int)

    X_missing, missing_mask = inject_mcar_missing(X_complete, MISSING_RATE, RANDOM_SEED)
    missing_rows = np.isnan(X_missing).any(axis=1)

    xgb_imputer = IterativeXGBoostImputer(n_estimators=120, max_iter=8, random_state=RANDOM_SEED)
    mean_imputer = MeanImputer(random_state=RANDOM_SEED)
    X_xgb = xgb_imputer.fit_transform(X_missing)
    X_mean = mean_imputer.fit_transform(X_missing)

    ref_labels = fit_dbscan_labels(X_complete)
    ref_labels = align_binary_labels(y_true, ref_labels)
    xgb_labels = align_binary_labels(ref_labels, fit_dbscan_labels(X_xgb))
    mean_labels = align_binary_labels(ref_labels, fit_dbscan_labels(X_mean))

    ari_xgb = adjusted_rand_score(ref_labels, xgb_labels)
    ari_mean = adjusted_rand_score(ref_labels, mean_labels)

    fig = plt.figure(figsize=(16.0, 7.4))
    fig.text(
        0.06,
        0.965,
        "Illustrative workflow from missing data to clustering-oriented imputation assessment",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.928,
        "Example based on the TwoMoons dataset with MCAR 10%. The pipeline highlights how an imputation strategy "
        "changes local geometry, clustering structure, and the final clustering quality score.",
        ha="left",
        va="top",
        fontsize=11.0,
        color="#566474",
    )

    ax_a = fig.add_axes([0.05, 0.53, 0.18, 0.30])
    add_panel_frame(ax_a, "#60a5fa")
    clean_axis(ax_a)
    ax_a.set_title("A. Missing data", loc="left", pad=10)
    matrix_ax = inset_axes(ax_a, width="34%", height="80%", loc="center left", borderpad=0.8)
    matrix_ax.imshow(missing_mask[:24, :], aspect="auto", cmap=plt.cm.Blues, vmin=0, vmax=1)
    matrix_ax.set_xticks([])
    matrix_ax.set_yticks([])
    for spine in matrix_ax.spines.values():
        spine.set_edgecolor("#64748b")
        spine.set_linewidth(0.8)
    scatter_ax = inset_axes(ax_a, width="58%", height="88%", loc="center right", borderpad=0.6)
    scatter_clusters(scatter_ax, X_complete, ref_labels, alpha=0.82)
    scatter_ax.scatter(
        X_complete[missing_rows, 0],
        X_complete[missing_rows, 1],
        s=52,
        facecolors="none",
        edgecolors="#111827",
        linewidths=0.85,
        zorder=3,
    )
    ax_a.text(
        0.04,
        0.05,
        r"$M$: observed entries with missingness mask",
        transform=ax_a.transAxes,
        fontsize=10.0,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.97},
    )

    ax_b = fig.add_axes([0.27, 0.53, 0.18, 0.30])
    add_panel_frame(ax_b, "#fbbf24")
    clean_axis(ax_b)
    ax_b.set_title("B. Imputed relation instance", loc="left", pad=10)
    scatter_clusters(ax_b, X_xgb, xgb_labels)
    ax_b.text(
        0.04,
        0.05,
        r"$X = \mathrm{Impute}_\pi(M)$ with strategy $\pi$",
        transform=ax_b.transAxes,
        fontsize=10.0,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.97},
    )

    ax_c = fig.add_axes([0.49, 0.53, 0.18, 0.30])
    add_panel_frame(ax_c, "#f59e0b")
    clean_axis(ax_c)
    ax_c.set_title("C. Structural deviation", loc="left", pad=10)
    scatter_clusters(ax_c, X_xgb, xgb_labels, alpha=0.25)
    x0, x1, y0, y1 = ZOOM_REGION
    ax_c.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linestyle=(0, (4, 3)), linewidth=1.1, edgecolor="#475569")
    )
    zoom_ax = inset_axes(ax_c, width="44%", height="44%", loc="upper right", borderpad=0.7)
    complete_zoom = (X_complete[:, 0] >= x0) & (X_complete[:, 0] <= x1) & (X_complete[:, 1] >= y0) & (X_complete[:, 1] <= y1)
    xgb_zoom = (X_xgb[:, 0] >= x0) & (X_xgb[:, 0] <= x1) & (X_xgb[:, 1] >= y0) & (X_xgb[:, 1] <= y1)
    zoom_ax.scatter(X_complete[complete_zoom, 0], X_complete[complete_zoom, 1], s=18, c="#cbd5e1", alpha=0.85)
    zoom_ax.scatter(X_xgb[xgb_zoom, 0], X_xgb[xgb_zoom, 1], s=18, c=COLORS["highlight"], alpha=0.9)
    zoom_ax.set_xticks([])
    zoom_ax.set_yticks([])
    zoom_ax.set_title(r"$D(\pi)$", fontsize=10.5, pad=2)
    for spine in zoom_ax.spines.values():
        spine.set_edgecolor("#475569")
        spine.set_linewidth(0.8)
    ax_c.text(
        0.04,
        0.05,
        "Orange region highlights local geometric distortion\nrelative to the intact reference structure.",
        transform=ax_c.transAxes,
        fontsize=9.6,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.97},
    )

    ax_d = fig.add_axes([0.71, 0.53, 0.18, 0.30])
    add_panel_frame(ax_d, "#f97316")
    clean_axis(ax_d)
    ax_d.set_title("D. Clustering structure", loc="left", pad=10)
    scatter_clusters(ax_d, X_xgb, xgb_labels)
    ax_d.add_patch(Ellipse((0.35, 0.64), 1.55, 0.58, transform=ax_d.transAxes, fill=False, edgecolor="#f59e0b", linestyle=(0, (4, 3)), linewidth=1.0))
    ax_d.add_patch(Ellipse((0.63, 0.37), 1.55, 0.62, transform=ax_d.transAxes, fill=False, edgecolor="#f59e0b", linestyle=(0, (4, 3)), linewidth=1.0))
    ax_d.text(
        0.04,
        0.05,
        r"$C(\pi)$: cluster organization induced by the imputed data",
        transform=ax_d.transAxes,
        fontsize=9.7,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.97},
    )

    ax_e = fig.add_axes([0.05, 0.16, 0.31, 0.24])
    add_panel_frame(ax_e, "#fbbf24")
    clean_axis(ax_e)
    ax_e.set_title("E. Data features induced by the imputation strategy", loc="left", pad=10)
    table_ax = inset_axes(ax_e, width="96%", height="72%", loc="lower center", borderpad=0.8)
    table_ax.axis("off")
    cell_text = [
        ["Bridge gap", "Preserved", "Low", "High"],
        ["Boundary blur", "Mild", "Low", "High"],
        ["Cluster compactness", "Strong", "Low", "High"],
    ]
    col_labels = ["Feature", "Observation", r"$D(\pi)$", r"$Q(\pi)$"]
    table = table_ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.1)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#eff6ff")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff")
    ax_e.text(
        0.03,
        0.07,
        r"Feature summary extracted from $\mathrm{Impute}_\pi(M)$",
        transform=ax_e.transAxes,
        fontsize=9.5,
        ha="left",
        va="bottom",
    )

    ax_f = fig.add_axes([0.42, 0.16, 0.20, 0.24])
    add_panel_frame(ax_f, "#f97316")
    clean_axis(ax_f)
    ax_f.set_title("F. Alternative clustering structure", loc="left", pad=10)
    scatter_clusters(ax_f, X_mean, mean_labels)
    ax_f.add_patch(Ellipse((0.50, 0.50), 1.70, 0.72, transform=ax_f.transAxes, fill=False, edgecolor="#ef4444", linestyle=(0, (4, 3)), linewidth=1.0))
    ax_f.text(
        0.04,
        0.05,
        r"$C(\pi^-)$ becomes less faithful when the geometry is distorted",
        transform=ax_f.transAxes,
        fontsize=9.2,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.97},
    )

    ax_g = fig.add_axes([0.67, 0.16, 0.22, 0.24])
    add_panel_frame(ax_g, "#f59e0b")
    ax_g.set_title("G. Performance score", loc="left", pad=10)
    values = [ari_mean, ari_xgb]
    labels = [r"$Q(\pi^-)$", r"$Q(\pi)$"]
    ax_g.bar(labels, values, color=["#d4d4d8", "#84cc16"], edgecolor="#475569", linewidth=0.9, width=0.55)
    ax_g.set_ylim(0, 1.0)
    ax_g.grid(axis="y", color="#eef2f7", linewidth=0.8)
    ax_g.spines["top"].set_visible(False)
    ax_g.spines["right"].set_visible(False)
    for xpos, value in enumerate(values):
        ax_g.text(xpos, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=10.0)
    ax_g.text(
        0.03,
        0.92,
        "Higher clustering quality indicates that the\nselected imputation strategy preserves structure better.",
        transform=ax_g.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.97},
    )

    add_arrow(fig, (0.23, 0.66), (0.27, 0.66), r"$M \rightarrow X$")
    add_arrow(fig, (0.45, 0.66), (0.49, 0.66), r"$X \rightarrow D$")
    add_arrow(fig, (0.67, 0.66), (0.71, 0.66), r"$D \rightarrow C$")
    add_arrow(fig, (0.79, 0.40), (0.79, 0.31), r"$C \downarrow$")
    add_arrow(fig, (0.36, 0.28), (0.42, 0.28), r"$X \rightarrow C$")
    add_arrow(fig, (0.62, 0.28), (0.67, 0.28), r"$C \rightarrow P$")

    fig.text(
        0.36,
        0.08,
        r"$M \;\rightarrow\; X \;\rightarrow\; D \;\rightarrow\; C \;\rightarrow\; P$",
        ha="center",
        va="center",
        fontsize=18,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fff7ed", "ec": "#f59e0b", "linestyle": (0, (4, 3))},
    )

    png_path = RESULTS_DIR / "example_pipeline_process__twomoons__mcar10.png"
    pdf_path = RESULTS_DIR / "example_pipeline_process__twomoons__mcar10.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)

    metadata = {
        "dataset": "twomoons",
        "missing_rate": MISSING_RATE,
        "method_main": "iterative_xgboost",
        "method_comparison": "mean",
        "ari_main": float(ari_xgb),
        "ari_comparison": float(ari_mean),
        "files": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metadata_path = RESULTS_DIR / "example_pipeline_process__twomoons__mcar10.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
