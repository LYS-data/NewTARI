"""Create a publication-style visualization for the Iris imputation process."""

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
    / "iris_imputation_process__mcar15__seed42"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from datasets.loader import DatasetLoader
from imputers.registry import build_imputer
from utils.missingness import inject_mcar_missing


RANDOM_SEED = 42
MISSING_RATE = 0.15
METHODS = [
    ("mean", "Mean"),
    ("knni", "KNNI"),
    ("missforest", "MissForest"),
    ("iterative_xgboost", "XGBoost"),
]
PALETTE = {
    "setosa": "#4C78A8",
    "versicolor": "#F58518",
    "virginica": "#54A24B",
}
FEATURE_HEATMAP_LIMIT = 36
METHOD_COLORS = {
    "mean": "#94a3b8",
    "knni": "#4C78A8",
    "missforest": "#F58518",
    "iterative_xgboost": "#54A24B",
}


def configure_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "savefig.transparent": False,
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "text.color": "#23313f",
            "axes.edgecolor": "#d9e0e8",
            "axes.linewidth": 0.85,
        }
    )


def project_with_full_data_basis(X_complete: np.ndarray, *arrays: np.ndarray) -> list[np.ndarray]:
    """Project all arrays through the same PCA basis for fair visual comparison."""
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(X_complete)
    return [pca.transform(array) for array in arrays]


def compute_featurewise_kl(
    original: np.ndarray,
    imputed: np.ndarray,
    *,
    n_bins: int = 18,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute KL(original || imputed) for each feature using aligned histograms."""
    if original.shape != imputed.shape:
        raise ValueError("original and imputed must have the same shape.")

    kl_values: list[float] = []
    for feature_idx in range(original.shape[1]):
        orig_col = original[:, feature_idx]
        imp_col = imputed[:, feature_idx]
        lower = float(min(orig_col.min(), imp_col.min()))
        upper = float(max(orig_col.max(), imp_col.max()))
        if np.isclose(lower, upper):
            kl_values.append(0.0)
            continue

        edges = np.linspace(lower, upper, n_bins + 1)
        p_hist, _ = np.histogram(orig_col, bins=edges, density=False)
        q_hist, _ = np.histogram(imp_col, bins=edges, density=False)
        p = p_hist.astype(float) + epsilon
        q = q_hist.astype(float) + epsilon
        p /= p.sum()
        q /= q.sum()
        kl_values.append(float(np.sum(p * np.log(p / q))))

    return np.asarray(kl_values, dtype=float)


def compute_panel_data() -> dict[str, object]:
    loader = DatasetLoader(data_root=PROJECT_ROOT / "data" / "raw")
    bundle = loader.load_csv(
        "iris.csv",
        dataset_name="iris",
        label_column="species",
        drop_columns=["target"],
    )
    X_complete = bundle.X
    labels = bundle.y.astype(str).to_numpy() if bundle.y is not None else np.array(["iris"] * len(bundle.X))

    X_missing, missing_mask = inject_mcar_missing(
        X_complete,
        missing_rate=MISSING_RATE,
        random_state=RANDOM_SEED,
    )

    projection_input = SimpleImputer(strategy="mean").fit_transform(X_missing)
    imputed_results: dict[str, np.ndarray] = {}
    kl_by_method: dict[str, np.ndarray] = {}
    for method_name, _ in METHODS:
        imputer = build_imputer(method_name, random_state=RANDOM_SEED)
        imputed_results[method_name] = imputer.fit_transform(X_missing)
        kl_by_method[method_name] = compute_featurewise_kl(X_complete, imputed_results[method_name])

    projected_arrays = project_with_full_data_basis(
        X_complete,
        X_complete,
        projection_input,
        *[imputed_results[name] for name, _ in METHODS],
    )

    return {
        "bundle": bundle,
        "labels": labels,
        "missing_mask": missing_mask,
        "missing_rate": float(missing_mask.mean()),
        "projected_complete": projected_arrays[0],
        "projected_missing": projected_arrays[1],
        "projected_imputed": {
            method_name: array
            for (method_name, _), array in zip(METHODS, projected_arrays[2:], strict=True)
        },
        "imputed_results": imputed_results,
        "kl_by_method": kl_by_method,
    }


def scatter_by_class(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    subtitle: str,
    missing_rows: np.ndarray | None = None,
    annotate_text: str | None = None,
) -> None:
    for cls_name, color in PALETTE.items():
        class_mask = labels == cls_name
        ax.scatter(
            coords[class_mask, 0],
            coords[class_mask, 1],
            s=54,
            c=color,
            edgecolors="#ffffff",
            linewidths=0.6,
            alpha=0.90,
        )

    if missing_rows is not None and np.any(missing_rows):
        ax.scatter(
            coords[missing_rows, 0],
            coords[missing_rows, 1],
            s=88,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.0,
            alpha=0.72,
            zorder=4,
        )

    ax.set_title(title, loc="left", pad=8)
    ax.text(
        0.02,
        0.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#ffffff", "ec": "#d6dde6", "alpha": 0.96},
    )
    if annotate_text:
        ax.text(
            0.98,
            0.98,
            annotate_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.7,
            bbox={"boxstyle": "round,pad=0.22", "fc": "#fff7ed", "ec": "#fdba74", "alpha": 0.98},
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9e0e8")
    ax.spines["bottom"].set_color("#d9e0e8")


def draw_missingness_heatmap(ax: plt.Axes, missing_mask: np.ndarray, feature_names: list[str]) -> None:
    display_mask = missing_mask[:FEATURE_HEATMAP_LIMIT].astype(int)
    sns.heatmap(
        display_mask,
        cmap=sns.color_palette(["#f8fafc", "#ef4444"]),
        cbar=False,
        linewidths=0.5,
        linecolor="#edf2f7",
        ax=ax,
    )
    ax.set_title("F. Missingness Pattern Snapshot", loc="left", pad=8)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sample index")
    ax.set_xticklabels([name.replace("_", "\n") for name in feature_names], rotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.text(
        0.02,
        -0.28,
        "Red cells denote injected missing entries in the first 36 samples.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#5b6775",
    )


def draw_featurewise_kl(
    ax: plt.Axes,
    kl_by_method: dict[str, np.ndarray],
    feature_names: list[str],
) -> None:
    positions = np.arange(len(feature_names))
    width = 0.18
    labels_short = ["SL", "SW", "PL", "PW"]

    for idx, (method_name, method_label) in enumerate(METHODS):
        offsets = positions + (idx - 1.5) * width
        ax.bar(
            offsets,
            kl_by_method[method_name],
            width=width,
            color=METHOD_COLORS[method_name],
            edgecolor="#334155",
            linewidth=0.5,
            label=method_label,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_short)
    ax.set_title("G. Feature-wise KL Divergence", loc="left", pad=8)
    ax.set_ylabel(r"$KL(\mathrm{original}\,||\,\mathrm{imputed})$")
    ax.text(
        0.02,
        0.98,
        "Lower values indicate smaller distribution shift after imputation.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#ffffff", "ec": "#d6dde6", "alpha": 0.97},
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.97,
        facecolor="#ffffff",
        edgecolor="#d6dde6",
        fontsize=8.0,
        ncol=2,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def build_figure(panel_data: dict[str, object]) -> plt.Figure:
    bundle = panel_data["bundle"]
    labels = panel_data["labels"]
    missing_mask = panel_data["missing_mask"]
    projected_complete = panel_data["projected_complete"]
    projected_missing = panel_data["projected_missing"]
    projected_imputed = panel_data["projected_imputed"]
    imputed_results = panel_data["imputed_results"]
    kl_by_method = panel_data["kl_by_method"]

    fig = plt.figure(figsize=(15.5, 9.8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.82], wspace=0.20, hspace=0.28)

    missing_rows = missing_mask.any(axis=1)

    ax_a = fig.add_subplot(gs[0, 0])
    scatter_by_class(
        ax_a,
        projected_complete,
        labels,
        title="A. Complete Iris Structure",
        subtitle="PCA projection of the original four-dimensional feature space.",
    )

    ax_b = fig.add_subplot(gs[0, 1])
    scatter_by_class(
        ax_b,
        projected_missing,
        labels,
        title="B. Missing Data Injection",
        subtitle=f"MCAR {panel_data['missing_rate']:.1%}; hollow circles mark affected rows.",
        missing_rows=missing_rows,
        annotate_text="Projection uses temporary mean fill\nonly for visualization.",
    )

    ax_c = fig.add_subplot(gs[0, 2])
    scatter_by_class(
        ax_c,
        projected_imputed["mean"],
        labels,
        title="C. Mean Imputation",
        subtitle="A simple baseline that pulls samples toward class centroids.",
        annotate_text="Tends to shrink local variance.",
    )

    ax_d = fig.add_subplot(gs[1, 0])
    scatter_by_class(
        ax_d,
        projected_imputed["knni"],
        labels,
        title="D. KNNI Imputation",
        subtitle="Neighbor-based recovery preserves local neighborhoods more faithfully.",
        annotate_text="Usually better at local continuity.",
    )

    ax_e = fig.add_subplot(gs[1, 1])
    scatter_by_class(
        ax_e,
        projected_imputed["missforest"],
        labels,
        title="E. MissForest Imputation",
        subtitle="Tree-based chained modeling can retain nonlinear feature interactions.",
        annotate_text="Good structural recovery in tabular settings.",
    )

    ax_f = fig.add_subplot(gs[1, 2])
    scatter_by_class(
        ax_f,
        projected_imputed["iterative_xgboost"],
        labels,
        title="F. XGBoost Imputation",
        subtitle="Boosted-tree iterative modeling often restores sharper class separation.",
        annotate_text="Useful when feature relations are nonlinear.",
    )

    ax_g = fig.add_subplot(gs[2, 0:2])
    draw_missingness_heatmap(ax_g, missing_mask, bundle.feature_names)

    ax_h = fig.add_subplot(gs[2, 2])
    draw_featurewise_kl(
        ax_h,
        kl_by_method,
        bundle.feature_names,
    )

    fig.suptitle(
        "Visualizing the Iris Imputation Process",
        fontsize=17.5,
        fontweight="bold",
        x=0.055,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.055,
        0.945,
        "The same PCA basis is used across panels so that changes in geometry reflect the effect of missingness and imputation.",
        fontsize=10.0,
        color="#5b6775",
        ha="left",
    )
    return fig


def save_outputs(panel_data: dict[str, object]) -> None:
    figure = build_figure(panel_data)
    png_path = RESULTS_DIR / "iris_imputation_process__mcar15__seed42.png"
    pdf_path = RESULTS_DIR / "iris_imputation_process__mcar15__seed42.pdf"
    json_path = RESULTS_DIR / "iris_imputation_process__mcar15__seed42.json"
    figure.savefig(png_path, dpi=320, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "dataset": "iris",
        "scenario": "mcar15",
        "random_seed": RANDOM_SEED,
        "methods": [method for method, _ in METHODS],
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "missing_rate": panel_data["missing_rate"],
        "featurewise_kl_original_vs_imputed": {
            method_name: {
                feature_name: float(value)
                for feature_name, value in zip(panel_data["bundle"].feature_names, kl_values, strict=True)
            }
            for method_name, kl_values in panel_data["kl_by_method"].items()
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    configure_style()
    panel_data = compute_panel_data()
    save_outputs(panel_data)
    print(f"Saved figure assets to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
