"""Create polished summary figures for the baseline experiment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline_experiment'
PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / 'mplconfig'
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIGDIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _style() -> None:
    sns.set_theme(context='talk', style='whitegrid')
    plt.rcParams.update(
        {
            'figure.facecolor': '#f6f1e8',
            'axes.facecolor': '#fffaf2',
            'axes.edgecolor': '#d6cbbd',
            'grid.color': '#dfd4c6',
            'axes.labelcolor': '#2f2a24',
            'text.color': '#2f2a24',
            'xtick.color': '#4a4036',
            'ytick.color': '#4a4036',
            'font.family': 'DejaVu Serif',
            'axes.titleweight': 'bold',
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    return path


def main() -> None:
    _style()
    detailed = pd.read_csv(RESULTS_DIR / 'baseline_detailed_results.csv')
    summary = pd.read_csv(RESULTS_DIR / 'baseline_summary.csv')

    summary_main = summary[summary['method'] != 'deletion'].copy()
    heatmap_data = detailed.pivot(index='method', columns='dataset_name', values='cluster_consistency_ari')
    method_order = pd.concat([summary[summary['method'] == 'deletion']['method'], summary_main.sort_values('overall_score', ascending=False)['method']])
    heatmap_data = heatmap_data.loc[method_order]

    fig, axes = plt.subplots(2, 2, figsize=(21, 15), gridspec_kw={'height_ratios': [1.15, 1.0]})
    fig.patch.set_facecolor('#f6f1e8')
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.12, wspace=0.28, hspace=0.42)

    ax = axes[0, 0]
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=sns.color_palette(['#f4e3c1', '#d8b96a', '#6c8a68', '#1f4d5a'], as_cmap=True),
        annot=True,
        fmt='.2f',
        linewidths=0.6,
        linecolor='#f6f1e8',
        cbar_kws={'shrink': 0.72, 'label': 'Cluster-consistency ARI', 'pad': 0.02},
    )
    ax.set_title('A. Cluster Structure Preservation by Dataset', loc='left', fontsize=18)
    ax.set_xlabel('Dataset', labelpad=10)
    ax.set_ylabel('Method', labelpad=10)
    ax.tick_params(axis='x', rotation=32, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    ax = axes[0, 1]
    bar_data = summary_main.sort_values('overall_score', ascending=True)
    colors = sns.color_palette('crest', n_colors=len(bar_data))
    ax.barh(bar_data['method'], bar_data['overall_score'], color=colors, edgecolor='#264653', linewidth=0.6)
    ax.set_title('B. Overall Baseline Score', loc='left', fontsize=18)
    ax.set_xlabel('Composite score (higher is better)', labelpad=10)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=11)
    for idx, value in enumerate(bar_data['overall_score']):
        ax.text(value + 0.008, idx, f'{value:.2f}', va='center', fontsize=10)

    ax = axes[1, 0]
    scatter = summary_main.copy()
    bubble_sizes = 250 + 800 * scatter['mean_row_retention'].fillna(0.0)
    points = ax.scatter(
        scatter['mean_runtime_sec'],
        scatter['mean_masked_nrmse'],
        s=bubble_sizes,
        c=scatter['overall_score'],
        cmap='viridis',
        alpha=0.88,
        edgecolors='#1f2933',
        linewidths=0.8,
    )
    label_offsets = {
        'median': (8, 10),
        'mice': (8, -14),
        'em': (8, 10),
        'knni': (8, 12),
        'mean': (8, -16),
        'iterative_xgboost': (8, 8),
        'missforest': (8, 10),
        'gain': (8, -16),
        'grape': (8, 10),
        'diffputer': (8, -18),
    }
    for _, row in scatter.iterrows():
        dx, dy = label_offsets.get(row['method'], (6, 6))
        ax.annotate(row['method'], (row['mean_runtime_sec'], row['mean_masked_nrmse']), xytext=(dx, dy), textcoords='offset points', fontsize=10, bbox={'boxstyle': 'round,pad=0.18', 'fc': '#fffaf2', 'ec': 'none', 'alpha': 0.82})
    ax.set_title('C. Speed vs Reconstruction Error', loc='left', fontsize=18)
    ax.set_xlabel('Mean runtime (sec)', labelpad=10)
    ax.set_ylabel('Masked NRMSE', labelpad=10)
    cbar = fig.colorbar(points, ax=ax, fraction=0.05, pad=0.035)
    cbar.set_label('Overall score')

    ax = axes[1, 1]
    profile = summary_main[['method', 'mean_cluster_consistency_ari', 'mean_ari_to_truth', 'mean_silhouette']].copy()
    profile = profile.sort_values('mean_cluster_consistency_ari', ascending=False).head(6)
    profile_long = profile.melt(id_vars='method', var_name='metric', value_name='value')
    sns.barplot(
        data=profile_long,
        x='value',
        y='method',
        hue='metric',
        palette=['#264653', '#2a9d8f', '#e9c46a'],
        ax=ax,
        orient='h',
    )
    ax.set_title('D. Top Methods on Key Quality Metrics', loc='left', fontsize=18)
    ax.set_xlabel('Metric value', labelpad=14)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(title='Metric', frameon=False, loc='upper right', fontsize=9, title_fontsize=10)

    fig.suptitle('Baseline Imputation Benchmark for Clustering Stability', fontsize=24, fontweight='bold', y=0.97)
    fig.text(0.5, 0.035, 'Setup: MCAR 20%, KMeans, seed=42, PenDigits downsampled to 2500 rows; composite score includes row retention and normalized reconstruction error.', ha='center', fontsize=11, color='#5a5148')
    composite_path = _save(fig, 'baseline_benchmark_overview.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    fig.patch.set_facecolor('#f6f1e8')
    rank_data = summary_main.sort_values('overall_score', ascending=False).reset_index(drop=True)
    ax.plot(rank_data['method'], rank_data['overall_score'], color='#264653', linewidth=2.5, marker='o', markersize=8)
    ax.fill_between(rank_data['method'], rank_data['overall_score'], color='#81b29a', alpha=0.18)
    ax.set_title('Method Ranking Curve', fontsize=20, loc='left')
    ax.set_xlabel('Method')
    ax.set_ylabel('Overall score')
    ax.tick_params(axis='x', rotation=35)
    for idx, row in rank_data.iterrows():
        ax.text(idx, row['overall_score'] + 0.01, f"{row['overall_score']:.2f}", ha='center', fontsize=10)
    rank_path = _save(fig, 'baseline_method_ranking_curve.png')
    plt.close(fig)

    print(f'Saved figure: {composite_path}')
    print(f'Saved figure: {rank_path}')


if __name__ == '__main__':
    main()
