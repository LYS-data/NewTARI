"""Create an illustrative figure for the distance cost D(r, C)."""

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
    / "example_distance_cost__twomoons"
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

from datasets import DatasetLoader


RANDOM_SEED = 42
GOOD_CENTER_INDICES = np.array([159, 47])
BAD_CENTER_INDICES = np.array([230, 236])
RELATION_INSTANCE_INDICES = np.array([203, 230, 236, 58, 159, 47])
PALETTE = {"upper": "#e76f51", "lower": "#2a9d8f", "faded": "#d6dde6"}


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
            "xtick.color": "#5b6773",
            "ytick.color": "#5b6773",
            "text.color": "#24313f",
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 14.0,
            "axes.titleweight": "bold",
            "axes.grid": False,
        }
    )


def distance_cost(X_points: np.ndarray, center_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(X_points[:, None, :] - center_points[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    return distances[np.arange(len(X_points)), nearest], nearest


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8dde6")
    ax.spines["bottom"].set_color("#d8dde6")
    ax.set_aspect("equal", adjustable="box")


def plot_candidate_panel(
    ax: plt.Axes,
    X: np.ndarray,
    y: np.ndarray,
    relation_points: np.ndarray,
    center_points: np.ndarray,
    center_name: str,
    title: str,
) -> float:
    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        s=24,
        c=PALETTE["upper"],
        edgecolors="none",
        alpha=0.22,
    )
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        s=24,
        c=PALETTE["lower"],
        edgecolors="none",
        alpha=0.22,
    )

    relation_distances, assignment = distance_cost(relation_points, center_points)
    for idx, point in enumerate(relation_points):
        center = center_points[assignment[idx]]
        ax.plot(
            [point[0], center[0]],
            [point[1], center[1]],
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            color="#64748b",
            alpha=0.95,
            zorder=2,
        )

    ax.scatter(
        relation_points[:, 0],
        relation_points[:, 1],
        s=58,
        c="#f8fafc",
        edgecolors="#334155",
        linewidths=1.0,
        zorder=3,
    )
    for idx, point in enumerate(relation_points, start=1):
        ax.text(
            point[0] + 0.03,
            point[1] + 0.03,
            f"$t_{idx}$",
            fontsize=9.0,
            color="#1f2937",
            zorder=4,
        )

    ax.scatter(
        center_points[:, 0],
        center_points[:, 1],
        marker="*",
        s=260,
        c="#f4a261",
        edgecolors="#8a4b08",
        linewidths=1.0,
        zorder=5,
    )
    for idx, center in enumerate(center_points, start=1):
        ax.text(
            center[0] + 0.04,
            center[1] - 0.08,
            f"$c_{idx}$",
            fontsize=9.6,
            fontweight="bold",
            color="#7c2d12",
            zorder=6,
        )

    ax.set_title(title, loc="left", pad=9)
    ax.text(
        0.01,
        0.02,
        center_name,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.98},
    )
    style_axis(ax)
    return float(relation_distances.sum())


def main() -> None:
    configure_style()

    loader = DatasetLoader(data_root=PROJECT_ROOT / "data" / "raw")
    bundle = loader.load_csv("twomoons.csv", dataset_name="twomoons", label_column="target")
    X = bundle.X
    y = bundle.y.to_numpy(dtype=int)

    relation_points = X[RELATION_INSTANCE_INDICES]
    good_centers = X[GOOD_CENTER_INDICES]
    bad_centers = X[BAD_CENTER_INDICES]
    good_distances, _ = distance_cost(relation_points, good_centers)
    bad_distances, _ = distance_cost(relation_points, bad_centers)
    good_cost = float(good_distances.sum())
    bad_cost = float(bad_distances.sum())

    fig = plt.figure(figsize=(15.2, 5.8))
    gs = fig.add_gridspec(1, 3, wspace=0.18)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_candidate_panel(
        ax_a,
        X,
        y,
        relation_points,
        good_centers,
        center_name=r"$C_{\mathrm{good}} = \{c_1, c_2\}$",
        title="A. Relation instance and good centers",
    )

    ax_b = fig.add_subplot(gs[0, 1])
    bar_labels = [rf"$\Delta(t_{idx}, C)$" for idx in range(1, len(good_distances) + 1)]
    ax_b.bar(
        np.arange(len(good_distances)),
        good_distances,
        color="#8ecae6",
        edgecolor="#386b8c",
        linewidth=0.8,
        width=0.62,
    )
    for idx, value in enumerate(good_distances):
        ax_b.text(idx, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9.2)
    ax_b.set_xticks(np.arange(len(good_distances)))
    ax_b.set_xticklabels(bar_labels, rotation=25, ha="right")
    ax_b.set_ylabel("Distance to nearest center")
    ax_b.set_title("B. Cost decomposition of $D(r, C)$", loc="left", pad=9)
    ax_b.grid(axis="y", color="#ecf0f4", linewidth=0.8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.text(
        0.02,
        0.96,
        rf"$D(r, C_{{good}}) = \sum \Delta(t_i, C_{{good}}) = {good_cost:.2f}$",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.98},
    )

    ax_c = fig.add_subplot(gs[0, 2])
    candidates = [r"$C_{\mathrm{good}}$", r"$C_{\mathrm{bad}}$"]
    values = [good_cost, bad_cost]
    ax_c.bar(
        candidates,
        values,
        color=["#2a9d8f", "#e76f51"],
        edgecolor="#334155",
        linewidth=0.8,
        width=0.58,
    )
    for xpos, value in enumerate(values):
        ax_c.text(xpos, value + 0.06, f"{value:.2f}", ha="center", va="bottom", fontsize=10.0)
    ax_c.set_ylabel(r"Total distance cost $D(r, C)$")
    ax_c.set_title("C. Lower $D(r, C)$ means a better center set", loc="left", pad=9)
    ax_c.grid(axis="y", color="#ecf0f4", linewidth=0.8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.text(
        0.02,
        0.97,
        "Same relation instance $r$, different center sets $C$.\n"
        "A smaller total distance means the chosen centers represent $r$ more compactly.",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        bbox={"boxstyle": "round,pad=0.22", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.98},
    )

    fig.text(
        0.07,
        0.98,
        "Illustrative example of the distance cost $D(r, C)$ on the TwoMoons dataset",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.935,
        r"Definition: $D(r, C) = \sum_{t_i \in r}\Delta(t_i, C)$, where $\Delta(t_i, C)$ is the distance from $t_i$ "
        r"to its nearest center in $C$.",
        ha="left",
        va="top",
        fontsize=11.2,
        color="#51606f",
    )

    png_path = RESULTS_DIR / "example_distance_cost__twomoons.png"
    pdf_path = RESULTS_DIR / "example_distance_cost__twomoons.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)

    metadata = {
        "dataset": "twomoons",
        "relation_instance_indices": RELATION_INSTANCE_INDICES.tolist(),
        "good_center_indices": GOOD_CENTER_INDICES.tolist(),
        "bad_center_indices": BAD_CENTER_INDICES.tolist(),
        "good_cost": good_cost,
        "bad_cost": bad_cost,
        "files": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metadata_path = RESULTS_DIR / "example_distance_cost__twomoons.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
