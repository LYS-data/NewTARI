"""Create a publication-style figure showing how imputation changes clustering."""

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
    / "example_cluster_impact__twomoons__mcar10__fixedk"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from datasets import DatasetLoader
from experiments.cluster_experiment_utils import inject_mcar_missing, masked_error_metrics
from imputers.iterative import IterativeXGBoostImputer
from imputers.knn import KNNIImputer
from imputers.simple import MeanImputer


RANDOM_SEED = 42
MISSING_RATE = 0.10
DBSCAN_EPS = 0.30
DBSCAN_MIN_SAMPLES = 5
ZOOM_REGION = (0.05, 1.55, -0.15, 0.95)
PALETTE = {0: "#2a9d8f", 1: "#e76f51", -1: "#9aa4b2"}
METHOD_SPECS = [
    ("mean", "Mean", lambda: MeanImputer(random_state=RANDOM_SEED)),
    ("knni", "KNNI", lambda: KNNIImputer(n_neighbors=7, random_state=RANDOM_SEED)),
    (
        "iterative_xgboost",
        "XGBoost",
        lambda: IterativeXGBoostImputer(
            n_estimators=120,
            max_iter=8,
            random_state=RANDOM_SEED,
        ),
    ),
]


def configure_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "savefig.transparent": False,
            "axes.edgecolor": "#d8dde6",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#24313f",
            "xtick.color": "#4f5d6b",
            "ytick.color": "#4f5d6b",
            "text.color": "#24313f",
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 14.5,
            "axes.titleweight": "bold",
            "axes.grid": False,
        }
    )


def fit_dbscan_labels(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    return model.fit_predict(X_scaled)


def align_labels(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    aligned = candidate.copy()
    for cluster_id in [0, 1]:
        if cluster_id not in set(candidate):
            continue
    valid_mask = reference >= 0
    ref = reference[valid_mask]
    cand = candidate[valid_mask]
    if set(np.unique(ref)) != {0, 1} or set(np.unique(cand)) != {0, 1}:
        return aligned
    original = adjusted_rand_score(ref, cand)
    flipped = np.where(candidate == -1, -1, 1 - candidate)
    flipped_score = adjusted_rand_score(ref, flipped[valid_mask])
    return flipped if flipped_score > original else aligned


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8dde6")
    ax.spines["bottom"].set_color("#d8dde6")


def draw_points(
    ax: plt.Axes,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    subtitle: str,
    missing_rows: np.ndarray | None = None,
    highlight_zoom: bool = False,
) -> None:
    for cluster_id in [0, 1]:
        mask = labels == cluster_id
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=28,
            c=PALETTE[cluster_id],
            edgecolors="#1f2933",
            linewidths=0.28,
            alpha=0.94,
            zorder=2,
        )
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(
            X[noise_mask, 0],
            X[noise_mask, 1],
            s=26,
            c=PALETTE[-1],
            edgecolors="#66768a",
            linewidths=0.25,
            alpha=0.82,
            zorder=2,
        )
    if missing_rows is not None and np.any(missing_rows):
        ax.scatter(
            X[missing_rows, 0],
            X[missing_rows, 1],
            s=58,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.95,
            alpha=0.72,
            zorder=3,
        )

    ax.set_title(title, loc="left", pad=9)
    ax.text(
        0.01,
        0.01,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.97},
        zorder=4,
    )
    if highlight_zoom:
        x0, x1, y0, y1 = ZOOM_REGION
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linestyle=(0, (4, 3)),
                linewidth=1.1,
                edgecolor="#334155",
                zorder=4,
            )
        )
    style_axis(ax)
    ax.set_aspect("equal", adjustable="box")


def add_zoom_inset(ax: plt.Axes, X: np.ndarray, labels: np.ndarray, missing_rows: np.ndarray | None) -> None:
    inset = inset_axes(ax, width="39%", height="39%", loc="upper right", borderpad=0.75)
    for cluster_id in [0, 1]:
        mask = labels == cluster_id
        inset.scatter(
            X[mask, 0],
            X[mask, 1],
            s=16,
            c=PALETTE[cluster_id],
            edgecolors="none",
            alpha=0.96,
        )
    if np.any(labels == -1):
        inset.scatter(
            X[labels == -1, 0],
            X[labels == -1, 1],
            s=14,
            c=PALETTE[-1],
            edgecolors="none",
            alpha=0.9,
        )
    if missing_rows is not None and np.any(missing_rows):
        inset.scatter(
            X[missing_rows, 0],
            X[missing_rows, 1],
            s=28,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.6,
            alpha=0.7,
        )
    x0, x1, y0, y1 = ZOOM_REGION
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(0.9)
    inset.set_facecolor("#ffffff")
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#94a3b8", lw=0.7)


def add_issue_annotation(ax: plt.Axes, text: str, xy: tuple[float, float], xytext: tuple[float, float]) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="axes fraction",
        xycoords="data",
        ha="left",
        va="center",
        fontsize=8.9,
        color="#1f2933",
        bbox={"boxstyle": "round,pad=0.22", "fc": "#ffffff", "ec": "#cbd5e1", "alpha": 0.98},
        arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 0.9},
        zorder=5,
    )


def main() -> None:
    configure_style()

    loader = DatasetLoader(data_root=PROJECT_ROOT / "data" / "raw")
    bundle = loader.load_csv("twomoons.csv", dataset_name="twomoons", label_column="target")
    X_complete = bundle.X
    y_true = bundle.y.to_numpy(dtype=int) if bundle.y is not None else None

    X_missing, missing_mask = inject_mcar_missing(X_complete, MISSING_RATE, RANDOM_SEED)
    row_has_missing = np.isnan(X_missing).any(axis=1)
    fill_for_display = np.nanmean(X_complete, axis=0)
    X_display_missing = np.where(np.isnan(X_missing), fill_for_display, X_missing)

    reference_labels = fit_dbscan_labels(X_complete)
    if y_true is not None:
        reference_labels = align_labels(y_true, reference_labels)

    results: list[dict[str, float | str]] = []
    panel_payloads: list[tuple[str, np.ndarray, np.ndarray, str]] = []

    for method_name, title, factory in METHOD_SPECS:
        imputer = factory()
        X_imputed = imputer.fit_transform(X_missing)
        labels = fit_dbscan_labels(X_imputed)
        labels = align_labels(reference_labels, labels)
        ari_ref = adjusted_rand_score(reference_labels, labels)
        ari_truth = adjusted_rand_score(y_true, labels) if y_true is not None else float("nan")
        rmse, nrmse = masked_error_metrics(X_complete, X_imputed, missing_mask)
        results.append(
            {
                "method": method_name,
                "title": title,
                "ari_reference": float(ari_ref),
                "ari_truth": float(ari_truth),
                "masked_nrmse": float(nrmse),
            }
        )
        subtitle = f"ARI(ref) = {ari_ref:.2f}\nARI(true) = {ari_truth:.2f}\nNRMSE = {nrmse:.2f}"
        panel_payloads.append((title, X_imputed, labels, subtitle))

    fig = plt.figure(figsize=(15.0, 8.4))
    gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.14)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_points(
        ax_a,
        X_complete,
        reference_labels,
        title="A. Intact structure",
        subtitle="TwoMoons intact geometry\nReference clusters from DBSCAN",
        highlight_zoom=False,
    )

    ax_b = fig.add_subplot(gs[0, 1])
    draw_points(
        ax_b,
        X_complete,
        reference_labels,
        title="B. Missing-affected samples",
        subtitle="Hollow circles mark samples containing at least one missing entry",
        missing_rows=row_has_missing,
        highlight_zoom=False,
    )
    ax_b.text(
        0.66,
        0.87,
        "MCAR 10%",
        transform=ax_b.transAxes,
        ha="left",
        va="center",
        fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#cbd5e1", "alpha": 0.98},
    )

    mask_inset = inset_axes(ax_b, width="34%", height="35%", loc="lower right", borderpad=0.8)
    display_rows = min(80, missing_mask.shape[0])
    mask_inset.imshow(
        missing_mask[:display_rows].astype(float),
        aspect="auto",
        cmap=plt.cm.Blues,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    mask_inset.set_title("Missingness mask", fontsize=8.4, pad=2)
    mask_inset.set_xticks([0, 1])
    mask_inset.set_xticklabels(["F1", "F2"], fontsize=7.6)
    mask_inset.set_yticks([])
    for spine in mask_inset.spines.values():
        spine.set_edgecolor("#5b7085")
        spine.set_linewidth(0.8)

    method_slots = [gs[0, 2], gs[1, 0], gs[1, 1]]
    method_titles = ["C", "D", "E"]
    method_axes: list[plt.Axes] = []
    for ax_slot, payload, prefix in zip(method_slots, panel_payloads, method_titles):
        ax = fig.add_subplot(ax_slot)
        title, X_panel, labels_panel, subtitle = payload
        draw_points(
            ax,
            X_panel,
            labels_panel,
            title=f"{prefix}. {title} imputation",
            subtitle=subtitle,
            highlight_zoom=True,
        )
        add_zoom_inset(ax, X_panel, labels_panel, None)
        method_axes.append(ax)

    summary = pd.DataFrame(results).sort_values("ari_reference", ascending=True)
    fig.text(
        0.07,
        0.972,
        "Illustrative impact of imputation choice on nonlinear clustering structure",
        fontsize=18.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.07,
        0.943,
        "Example based on the TwoMoons dataset with MCAR 10%. "
        "Only the boundary-sensitive region is magnified to show how different imputers alter local cluster separation.",
        fontsize=10.8,
        ha="left",
        va="top",
        color="#51606f",
    )

    panel_f = fig.add_subplot(gs[1, 2])
    panel_f.axis("off")
    panel_f.text(
        0.0,
        1.03,
        "F. Quantitative comparison",
        transform=panel_f.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.5,
        fontweight="bold",
        color="#24313f",
    )
    bottom_ax = inset_axes(
        panel_f,
        width="92%",
        height="78%",
        loc="upper left",
        bbox_to_anchor=(0.07, 0.02, 0.93, 0.90),
        bbox_transform=panel_f.transAxes,
        borderpad=0,
    )
    bottom_ax.barh(
        summary["title"],
        summary["ari_reference"],
        color=["#d7dee6", "#7fb3d5", "#264653"],
        edgecolor="#213547",
        linewidth=0.6,
    )
    bottom_ax.set_xlim(0, 1.0)
    bottom_ax.set_title("ARI to reference structure", loc="left", fontsize=11.0, pad=6)
    bottom_ax.tick_params(axis="y", labelsize=9.0)
    bottom_ax.tick_params(axis="x", labelsize=8.8)
    bottom_ax.spines["top"].set_visible(False)
    bottom_ax.spines["right"].set_visible(False)
    bottom_ax.grid(axis="x", color="#edf1f5", linewidth=0.7)
    for idx, value in enumerate(summary["ari_reference"]):
        bottom_ax.text(value + 0.02, idx, f"{value:.2f}", va="center", fontsize=8.6)

    add_issue_annotation(
        method_axes[0],
        "Bridge artifacts\ncollapse the gap",
        xy=(0.86, 0.40),
        xytext=(0.05, 0.82),
    )
    add_issue_annotation(
        method_axes[1],
        "Local boundary remains\npartly mixed",
        xy=(0.92, 0.38),
        xytext=(0.05, 0.80),
    )
    add_issue_annotation(
        method_axes[2],
        "Boundary is largely\npreserved here",
        xy=(0.98, 0.36),
        xytext=(0.05, 0.80),
    )

    png_path = RESULTS_DIR / "example_cluster_impact__twomoons__mcar10__fixedk.png"
    pdf_path = RESULTS_DIR / "example_cluster_impact__twomoons__mcar10__fixedk.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)

    metadata = {
        "dataset": "twomoons",
        "missing_scheme": "MCAR",
        "missing_rate": MISSING_RATE,
        "reference_clusterer": {
            "name": "DBSCAN",
            "eps": DBSCAN_EPS,
            "min_samples": DBSCAN_MIN_SAMPLES,
        },
        "random_seed": RANDOM_SEED,
        "methods": results,
        "files": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metadata_path = RESULTS_DIR / "example_cluster_impact__twomoons__mcar10__fixedk.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
