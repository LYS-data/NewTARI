"""Create a polished Chinese framework diagram for the recommendation pipeline."""

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
    / "example_recommendation_framework__cn"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle, Ellipse
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


FONT_FAMILY = "Microsoft YaHei"
COLORS = {
    "navy": "#1d3557",
    "blue": "#4f8fda",
    "sky": "#dceeff",
    "orange": "#f4a261",
    "orange_soft": "#fff3e8",
    "green": "#7ac943",
    "green_soft": "#eaf7df",
    "red": "#e76f51",
    "red_soft": "#fff1ec",
    "gray": "#5f6c7b",
    "light": "#f8fbff",
    "border": "#cfd9e6",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "savefig.transparent": False,
            "font.family": FONT_FAMILY,
            "axes.unicode_minus": False,
            "text.color": COLORS["navy"],
        }
    )


def add_panel(fig: plt.Figure, rect: list[float], title: str, icon: str, title_color: str = COLORS["navy"]) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    frame = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=COLORS["border"],
        facecolor="#ffffff",
        transform=ax.transAxes,
    )
    ax.add_patch(frame)
    base_x, base_y, s = 0.05, 0.875, 0.025
    for dx, dy in [(0, 0), (0.034, 0), (0, 0.04), (0.034, 0.04)]:
        ax.add_patch(Rectangle((base_x + dx, base_y + dy), s, s, facecolor="#ffffff", edgecolor=COLORS["blue"], linewidth=1.8))
    ax.text(0.16, 0.90, title, fontsize=16, fontweight="bold", va="center", ha="left", color=title_color)
    return ax


def add_arrow(fig: plt.Figure, p1: tuple[float, float], p2: tuple[float, float], text: str | None = None) -> None:
    arrow = FancyArrowPatch(
        p1,
        p2,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.6,
        color=COLORS["red"],
    )
    fig.add_artist(arrow)
    if text:
        fig.text(
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2 + 0.018,
            text,
            ha="center",
            va="bottom",
            fontsize=11,
            color=COLORS["red"],
            fontweight="bold",
        )


def mini_table(ax: plt.Axes, x: float, y: float, w: float, h: float, rows: int, cols: int, face: str = "#ffffff") -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=COLORS["border"], linewidth=1.0))
    for i in range(1, rows):
        yy = y + h * i / rows
        ax.add_line(Line2D([x, x + w], [yy, yy], color=COLORS["border"], linewidth=0.9))
    for j in range(1, cols):
        xx = x + w * j / cols
        ax.add_line(Line2D([xx, xx], [y, y + h], color=COLORS["border"], linewidth=0.9))


def draw_input_panel(ax: plt.Axes) -> None:
    mini_table(ax, 0.05, 0.60, 0.38, 0.20, 1, 1, face=COLORS["sky"])
    ax.text(0.24, 0.70, "关键特征集合", ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.24, 0.64, r"$F=\{f_1,f_2,\ldots,f_k\}$", ha="center", va="center", fontsize=12)

    mini_table(ax, 0.05, 0.41, 0.38, 0.13, 1, 1, face=COLORS["light"])
    ax.text(0.24, 0.475, "误差传播结构信息", ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.24, 0.43, r"$\mathbf{V}=[M,X,D,C,P]$", ha="center", va="center", fontsize=11.5)

    mini_table(ax, 0.05, 0.13, 0.38, 0.20, 1, 1, face=COLORS["sky"])
    ax.text(0.24, 0.24, "候选填充策略", ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.24, 0.17, r"$f_1:(m_1,m_2,\ldots),\;f_2:\cdots$", ha="center", va="center", fontsize=10.8)


def draw_constraint_panel(ax: plt.Axes) -> None:
    mini_table(ax, 0.05, 0.54, 0.36, 0.24, 4, 4, face="#ffffff")
    headers = [r"$f_1$", r"$f_2$", r"$f_3$", r"$f_{d-1}$"]
    for idx, txt in enumerate(headers):
        ax.text(0.095 + idx * 0.09, 0.74, txt, ha="center", va="center", fontsize=11, color=COLORS["navy"])
    ax.text(0.11, 0.65, r"$m_1$", color=COLORS["blue"], ha="center", va="center", fontsize=11)
    ax.text(0.20, 0.65, r"$m_2$", color=COLORS["blue"], ha="center", va="center", fontsize=11)
    ax.text(0.29, 0.65, r"$m_3$", color=COLORS["blue"], ha="center", va="center", fontsize=11)
    ax.text(0.38, 0.65, r"$\cdots$", color=COLORS["gray"], ha="center", va="center", fontsize=11)

    ax.text(0.49, 0.72, "关联特征组", color=COLORS["red"], fontsize=11.5, fontweight="bold", ha="left")
    ax.text(0.49, 0.66, r"$E_1=\{f_2,f_3\}$", color=COLORS["red"], fontsize=12, ha="left")
    ax.text(0.49, 0.54, "误差传播图", color=COLORS["red"], fontsize=11.5, fontweight="bold", ha="left")
    ax.text(0.49, 0.48, r"$(M,X,D,C,P)$", color=COLORS["red"], fontsize=12, ha="left")

    mini_table(ax, 0.05, 0.18, 0.30, 0.18, 2, 3, face=COLORS["light"])
    ax.text(0.095, 0.30, r"$\pi_{1,1}$", ha="center", va="center", fontsize=12)
    ax.text(0.20, 0.30, r"$f_2$", ha="center", va="center", fontsize=12)
    ax.text(0.305, 0.30, r"$f_3$", ha="center", va="center", fontsize=12)
    ax.text(0.095, 0.22, r"$m_1$", ha="center", va="center", fontsize=12, color=COLORS["blue"])
    ax.text(0.20, 0.22, r"$m_2$", ha="center", va="center", fontsize=12, color=COLORS["blue"])
    ax.text(0.305, 0.22, r"$m_3$", ha="center", va="center", fontsize=12, color=COLORS["blue"])

    ax.text(0.40, 0.25, r"$\pi_{f_1,E_1}=(m_1,m_2,m_3)$", fontsize=12, ha="left", va="center")
    ax.text(0.40, 0.15, r"$\pi_{f_2,f_3}=(m_2,m_3)$", fontsize=12, ha="left", va="center")

    mini_table(ax, 0.70, 0.18, 0.23, 0.18, 2, 3, face=COLORS["light"])
    ax.text(0.74, 0.30, r"$\pi_{1,1}$", ha="center", va="center", fontsize=12)
    ax.text(0.82, 0.30, r"$f_2$", ha="center", va="center", fontsize=12)
    ax.text(0.90, 0.30, r"$f_3$", ha="center", va="center", fontsize=12)
    ax.text(0.74, 0.22, r"$m_2$", ha="center", va="center", fontsize=12, color=COLORS["blue"])
    ax.text(0.82, 0.22, r"$m_2$", ha="center", va="center", fontsize=12, color=COLORS["blue"])
    ax.text(0.90, 0.22, r"$m_2$", ha="center", va="center", fontsize=12, color=COLORS["blue"])

    graph_ax = inset_axes(ax, width="25%", height="36%", loc="upper right", borderpad=1.2)
    graph_ax.set_xlim(0, 1)
    graph_ax.set_ylim(0, 1)
    graph_ax.axis("off")
    for cx, cy, label in [(0.22, 0.62, r"$f_1$"), (0.48, 0.78, r"$f_3$"), (0.72, 0.55, "E"), (0.38, 0.38, r"$D_1$")]:
        graph_ax.add_patch(Circle((cx, cy), 0.09, facecolor=COLORS["sky"], edgecolor=COLORS["blue"], linewidth=1.2))
        graph_ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5)
    for p1, p2 in [((0.28, 0.65), (0.42, 0.74)), ((0.54, 0.74), (0.66, 0.60)), ((0.28, 0.58), (0.35, 0.44)), ((0.46, 0.42), (0.63, 0.52))]:
        graph_ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=COLORS["blue"]))


def draw_group_panel(ax: plt.Axes) -> None:
    ax.text(0.16, 0.60, "特征\n组构建", ha="center", va="center", fontsize=18, fontweight="bold", linespacing=1.35)
    ax.add_patch(Circle((0.16, 0.78), 0.05, facecolor=COLORS["sky"], edgecolor=COLORS["blue"], linewidth=1.3))
    ax.text(0.16, 0.78, "约", ha="center", va="center", fontsize=15, fontweight="bold", color=COLORS["blue"])
    ax.add_patch(Circle((0.84, 0.78), 0.05, facecolor=COLORS["sky"], edgecolor=COLORS["blue"], linewidth=1.3))
    ax.text(0.84, 0.78, "组", ha="center", va="center", fontsize=15, fontweight="bold", color=COLORS["blue"])

    graph_ax = inset_axes(ax, width="52%", height="42%", loc="center", borderpad=0.2)
    graph_ax.set_xlim(0, 1)
    graph_ax.set_ylim(0, 1)
    graph_ax.axis("off")
    centers = [(0.18, 0.55, r"$f_1$"), (0.42, 0.75, r"$f_3$"), (0.72, 0.56, "E"), (0.44, 0.32, r"$D_1$")]
    for cx, cy, label in centers:
        graph_ax.add_patch(Circle((cx, cy), 0.10, facecolor=COLORS["sky"], edgecolor=COLORS["blue"], linewidth=1.3))
        graph_ax.text(cx, cy, label, ha="center", va="center", fontsize=12)
    for p1, p2, curve in [
        ((0.24, 0.59), (0.36, 0.70), 0.18),
        ((0.50, 0.72), (0.64, 0.61), -0.10),
        ((0.26, 0.49), (0.36, 0.36), -0.18),
        ((0.52, 0.36), (0.64, 0.51), 0.12),
        ((0.18, 0.55), (0.72, 0.56), 0.28),
    ]:
        graph_ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={curve}", arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=COLORS["blue"]))


def draw_eval_panel(ax: plt.Axes) -> None:
    ax.text(0.50, 0.76, "填充后的聚类性能", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["blue"])
    mini_table(ax, 0.06, 0.48, 0.26, 0.16, 1, 1, face=COLORS["sky"])
    mini_table(ax, 0.38, 0.48, 0.26, 0.16, 1, 1, face=COLORS["green_soft"])
    ax.text(0.19, 0.56, "准确性: Sil", ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.51, 0.56, "稳定性: Stab", ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.08, 0.32, r"$Q(z;B)=\alpha\cdot Sil^\prime(\pi(X))+\beta\cdot Stab^\prime(z;B)$", fontsize=12, ha="left")
    ax.text(0.40, 0.21, "评估选择", fontsize=11, color=COLORS["gray"], ha="center")
    ax.text(0.22, 0.08, r"$\pi^\star=\arg\max_{\pi\in\mathcal{M}}Q(X^\pi)$", fontsize=16, ha="left", color=COLORS["navy"])


def draw_bohb_panel(ax: plt.Axes) -> None:
    ax.text(0.17, 0.72, "采样搜索", fontsize=11.5, fontweight="bold", ha="center")
    ax.text(0.80, 0.72, "评估候选采样", fontsize=11.5, fontweight="bold", ha="center")
    ax.text(0.74, 0.48, "多保真\n探索", fontsize=12, fontweight="bold", ha="center", va="center")

    # left icon
    ax.add_patch(Rectangle((0.05, 0.48), 0.12, 0.20, facecolor=COLORS["orange_soft"], edgecolor=COLORS["border"], linewidth=1.0))
    ax.add_patch(Rectangle((0.08, 0.52), 0.02, 0.10, facecolor=COLORS["orange"], edgecolor="none"))
    ax.add_patch(Rectangle((0.11, 0.52), 0.02, 0.06, facecolor=COLORS["orange"], edgecolor="none"))
    ax.add_patch(Rectangle((0.14, 0.52), 0.02, 0.14, facecolor=COLORS["orange"], edgecolor="none"))

    center_box = FancyBboxPatch((0.28, 0.42), 0.24, 0.30, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor="#fff5e8", edgecolor=COLORS["orange"], linewidth=1.3)
    ax.add_patch(center_box)
    ax.text(0.40, 0.57, "BOHB", fontsize=20, fontweight="bold", ha="center", va="center", color=COLORS["navy"])
    ax.add_patch(Circle((0.36, 0.57), 0.07, facecolor=COLORS["sky"], edgecolor=COLORS["blue"], linewidth=1.0))
    ax.text(0.36, 0.57, "◎", ha="center", va="center", fontsize=20, color=COLORS["blue"])

    mini_table(ax, 0.62, 0.46, 0.18, 0.22, 4, 3, face="#ffffff")
    ax.text(0.67, 0.64, r"$\pi_1$", ha="center", va="center", fontsize=10.5)
    ax.text(0.73, 0.64, r"$\pi_2$", ha="center", va="center", fontsize=10.5)
    ax.text(0.79, 0.64, r"$\pi_3$", ha="center", va="center", fontsize=10.5)
    for i in range(3):
        for j in range(2):
            ax.add_patch(Rectangle((0.64 + 0.06 * i, 0.49 + 0.05 * j), 0.03, 0.03,
                                   facecolor=[COLORS["orange"], COLORS["green"], COLORS["blue"]][i], edgecolor="none", alpha=0.75))
    ax.add_patch(Rectangle((0.84, 0.46), 0.09, 0.22, facecolor=COLORS["light"], edgecolor=COLORS["border"], linewidth=1.0))
    ax.add_line(Line2D([0.885, 0.885], [0.50, 0.64], color=COLORS["gray"], linewidth=1.8))
    ax.add_line(Line2D([0.86, 0.91], [0.63, 0.63], color=COLORS["gray"], linewidth=1.6))
    ax.add_line(Line2D([0.865, 0.878], [0.61, 0.58], color=COLORS["gray"], linewidth=1.4))
    ax.add_line(Line2D([0.905, 0.892], [0.61, 0.58], color=COLORS["gray"], linewidth=1.4))
    ax.add_line(Line2D([0.865, 0.878], [0.55, 0.52], color=COLORS["gray"], linewidth=1.4))
    ax.add_line(Line2D([0.905, 0.892], [0.55, 0.52], color=COLORS["gray"], linewidth=1.4))

    mini_table(ax, 0.06, 0.08, 0.14, 0.18, 4, 1, face=COLORS["light"])
    for idx, txt in enumerate([r"$\pi_1$", r"$\pi_2$", r"$\cdots$", r"$\pi_n$"]):
        ax.text(0.13, 0.22 - idx * 0.04, txt, ha="center", va="center", fontsize=10.5)

    mini_table(ax, 0.40, 0.08, 0.14, 0.18, 4, 1, face=COLORS["light"])
    for idx, txt in enumerate([r"$\pi_1$", r"$\pi_j$", r"$\cdots$", r"$\pi_j$"]):
        ax.text(0.47, 0.22 - idx * 0.04, txt, ha="center", va="center", fontsize=10.5)

    ax.add_patch(Rectangle((0.75, 0.10), 0.12, 0.12, facecolor=COLORS["light"], edgecolor=COLORS["border"], linewidth=1.0))
    ax.add_patch(Circle((0.81, 0.16), 0.032, facecolor="none", edgecolor=COLORS["green"], linewidth=1.8))
    ax.add_patch(FancyArrowPatch((0.815, 0.19), (0.835, 0.18), arrowstyle="-|>", mutation_scale=10, linewidth=1.4, color=COLORS["green"]))
    ax.text(0.81, 0.07, "优化历史", fontsize=10.5, ha="center", va="center")

    # arrows
    for p1, p2 in [((0.18, 0.57), (0.28, 0.57)), ((0.52, 0.57), (0.62, 0.57)), ((0.20, 0.17), (0.40, 0.17)), ((0.54, 0.17), (0.75, 0.17))]:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=COLORS["red"]))
    ax.add_patch(FancyArrowPatch((0.40, 0.42), (0.40, 0.26), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=COLORS["red"]))
    ax.add_patch(FancyArrowPatch((0.28, 0.17), (0.28, 0.42), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=COLORS["red"]))
    ax.text(0.30, 0.31, r"填充策略库 $\mathcal{M}$", fontsize=10.5, color=COLORS["red"], ha="center")
    ax.text(0.63, 0.12, "更新策略", fontsize=10.5, color=COLORS["red"], ha="center")


def draw_output_panel(ax: plt.Axes) -> None:
    mini_table(ax, 0.10, 0.18, 0.80, 0.56, 6, 2, face="#ffffff")
    ax.text(0.30, 0.66, "Feature", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.70, 0.66, "Imputation", ha="center", va="center", fontsize=12, fontweight="bold")
    rows = [
        ("feature_1", "KNN"),
        ("feature_2", "MICE"),
        ("feature_3", "GAIN"),
        ("feature_4", "MICE"),
        ("feature_5", "--"),
    ]
    for idx, (ftr, imp) in enumerate(rows):
        yy = 0.58 - idx * 0.09
        ax.text(0.30, yy, ftr, ha="center", va="center", fontsize=11)
        ax.text(0.70, yy, imp, ha="center", va="center", fontsize=11)


def main() -> None:
    configure_style()

    fig = plt.figure(figsize=(16.5, 8.0))
    fig.text(
        0.05,
        0.965,
        "面向聚类结构保持的缺失值填充方法选择与推荐框架",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.93,
        "该图示从输入信息、特征级约束、候选策略搜索到最终推荐输出，展示了完整的研究流程。",
        ha="left",
        va="top",
        fontsize=11.5,
        color=COLORS["gray"],
    )

    ax_input = add_panel(fig, [0.04, 0.52, 0.20, 0.38], "Input", "▦")
    draw_input_panel(ax_input)

    ax_constraint = add_panel(fig, [0.26, 0.52, 0.46, 0.38], "搜索空间约束", "⚲")
    draw_constraint_panel(ax_constraint)

    ax_group = add_panel(fig, [0.74, 0.52, 0.22, 0.38], "特征组构建", "⟳")
    draw_group_panel(ax_group)

    ax_eval = add_panel(fig, [0.04, 0.08, 0.34, 0.34], "填充策略评估与优化目标", "⟳")
    draw_eval_panel(ax_eval)

    ax_bohb = add_panel(fig, [0.40, 0.08, 0.38, 0.34], "BOHB-based 搜索优化", "⚙")
    draw_bohb_panel(ax_bohb)

    ax_output = add_panel(fig, [0.80, 0.08, 0.16, 0.34], r"优化填充策略 $\pi^\star$", "⬒")
    draw_output_panel(ax_output)

    add_arrow(fig, (0.24, 0.70), (0.26, 0.70))
    add_arrow(fig, (0.72, 0.70), (0.74, 0.70))
    add_arrow(fig, (0.21, 0.52), (0.21, 0.42))
    add_arrow(fig, (0.59, 0.52), (0.59, 0.42))
    add_arrow(fig, (0.78, 0.25), (0.80, 0.25))
    add_arrow(fig, (0.38, 0.25), (0.40, 0.25))

    fig.text(
        0.51,
        0.02,
        r"$M \rightarrow X \rightarrow D \rightarrow C \rightarrow P$",
        ha="center",
        va="bottom",
        fontsize=18,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fff7ed", "ec": "#f59e0b", "linestyle": (0, (4, 3))},
    )

    png_path = RESULTS_DIR / "example_recommendation_framework__cn.png"
    pdf_path = RESULTS_DIR / "example_recommendation_framework__cn.pdf"
    fig.savefig(png_path, dpi=280, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_path, dpi=280, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)

    metadata = {
        "title": "面向聚类结构保持的缺失值填充方法选择与推荐框架",
        "files": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metadata_path = RESULTS_DIR / "example_recommendation_framework__cn.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
