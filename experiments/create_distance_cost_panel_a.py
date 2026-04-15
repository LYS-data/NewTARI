"""Create a single-panel illustration for the relation instance and center set."""

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


GOOD_CENTER_INDICES = np.array([159, 47])
RELATION_INSTANCE_INDICES = np.array([203, 230, 236, 58, 159, 47])
PALETTE = {"upper": "#e76f51", "lower": "#2a9d8f"}
POINT_LABEL_OFFSETS = {
    1: (0.05, 0.05),
    2: (0.05, 0.03),
    3: (0.05, 0.05),
    4: (0.04, 0.06),
    5: (0.05, 0.06),
    6: (0.05, 0.06),
}
CENTER_LABEL_OFFSETS = {
    1: (0.06, -0.12),
    2: (0.07, -0.13),
}


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
            "text.color": "#24313f",
            "font.family": "DejaVu Serif",
        }
    )


def distance_cost(X_points: np.ndarray, center_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(X_points[:, None, :] - center_points[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    return distances[np.arange(len(X_points)), nearest], nearest


def main() -> None:
    configure_style()

    loader = DatasetLoader(data_root=PROJECT_ROOT / "data" / "raw")
    bundle = loader.load_csv("twomoons.csv", dataset_name="twomoons", label_column="target")
    X = bundle.X
    y = bundle.y.to_numpy(dtype=int)

    relation_points = X[RELATION_INSTANCE_INDICES]
    center_points = X[GOOD_CENTER_INDICES]
    relation_distances, assignment = distance_cost(relation_points, center_points)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        s=28,
        c=PALETTE["upper"],
        edgecolors="none",
        alpha=0.18,
    )
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        s=28,
        c=PALETTE["lower"],
        edgecolors="none",
        alpha=0.18,
    )

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
        s=74,
        c="#ffffff",
        edgecolors="#334155",
        linewidths=1.05,
        zorder=3,
    )
    for idx, point in enumerate(relation_points, start=1):
        dx, dy = POINT_LABEL_OFFSETS[idx]
        ax.text(
            point[0] + dx,
            point[1] + dy,
            f"$t_{idx}$",
            fontsize=10.0,
            color="#1f2937",
            bbox={"boxstyle": "round,pad=0.08", "fc": "#ffffff", "ec": "none", "alpha": 0.88},
            zorder=4,
        )

    ax.scatter(
        center_points[:, 0],
        center_points[:, 1],
        marker="*",
        s=320,
        c="#f4a261",
        edgecolors="#8a4b08",
        linewidths=1.1,
        zorder=5,
    )
    for idx, center in enumerate(center_points, start=1):
        dx, dy = CENTER_LABEL_OFFSETS[idx]
        ax.text(
            center[0] + dx,
            center[1] + dy,
            f"$c_{idx}$",
            fontsize=10.5,
            fontweight="bold",
            color="#7c2d12",
            bbox={"boxstyle": "round,pad=0.08", "fc": "#ffffff", "ec": "none", "alpha": 0.9},
            zorder=6,
        )

    ax.set_title("Relation instance $r$ and center set $C$", loc="left", pad=10, fontsize=14.4, fontweight="bold")
    ax.text(
        0.02,
        0.03,
        r"$C = \{c_1, c_2\}$" + "\n" + r"$D(r, C)=\sum_{t_i \in r}\Delta(t_i, C)$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.24", "fc": "#ffffff", "ec": "#d8dde6", "alpha": 0.98},
    )

    x_pad = 0.14
    y_pad = 0.14
    ax.set_xlim(X[:, 0].min() - x_pad, X[:, 0].max() + x_pad)
    ax.set_ylim(X[:, 1].min() - y_pad, X[:, 1].max() + y_pad)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8dde6")
    ax.spines["bottom"].set_color("#d8dde6")
    ax.set_aspect("equal", adjustable="box")

    png_path = RESULTS_DIR / "example_distance_cost__twomoons_panel_a.png"
    pdf_path = RESULTS_DIR / "example_distance_cost__twomoons_panel_a.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)

    metadata = {
        "dataset": "twomoons",
        "relation_instance_indices": RELATION_INSTANCE_INDICES.tolist(),
        "center_indices": GOOD_CENTER_INDICES.tolist(),
        "distance_values": relation_distances.tolist(),
        "files": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metadata_path = RESULTS_DIR / "example_distance_cost__twomoons_panel_a.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
