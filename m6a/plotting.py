"""
m6a.plotting — Violin-plot helpers shared across cross-cohort and mCRPC analyses.

Functions
---------
style_violin(parts, ax, legend, extra_handles)
    Style mean/median lines on a violinplot; optionally add legend.
add_significance_bar(ax, x1, x2, y, p, lw)
    Draw a bracket + formatted p-value between two violin positions.
ngroup_violin(vals_list, labels, colors, title, outpath, ...)
    Generic N-group violin with Dunn post-hoc significance bars.
two_group_violin(va, vb, label_a, label_b, title, outpath, ...)
    Two-group violin with Mann-Whitney U result annotation.
"""
import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, kruskal

from m6a.stats import sig, fmt_p, rankbiserial, dunn_pairwise


def style_violin(parts, ax=None, legend=True, extra_handles=None):
    """
    Style the mean (black dashed) and median (red solid) lines of a violinplot.

    Parameters
    ----------
    parts : dict  Return value of ``ax.violinplot()``.
    ax    : Axes  If provided, a legend is drawn.
    legend : bool  Whether to draw the mean/median legend.
    extra_handles : list of Artist, optional
        Additional legend handles to include alongside mean/median entries.
    """
    if 'cmeans' in parts:
        parts['cmeans'].set_linestyle('--')
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.5)
    if 'cmedians' in parts:
        parts['cmedians'].set_linestyle('-')
        parts['cmedians'].set_color('#e74c3c')
        parts['cmedians'].set_linewidth(1.5)
    if legend and ax is not None:
        handles = [
            Line2D([0], [0], color='black',   ls='--', lw=1.5, label='Mean'),
            Line2D([0], [0], color='#e74c3c', ls='-',  lw=1.5, label='Median'),
        ]
        if extra_handles:
            handles.extend(extra_handles)
        existing = ax.get_legend()
        if existing:
            for h in existing.legend_handles:
                handles.append(h)
        ax.legend(handles=handles, fontsize=8, loc='upper right')


def add_significance_bar(ax, x1, x2, y, p, lw=1.2):
    """
    Draw a significance bracket between positions *x1* and *x2* at height *y*.

    The bar descends 0.5 units at each end, then text is placed 0.7 units above
    the crossbar, showing ``fmt_p(p)`` and ``sig(p)``.
    """
    ax.plot([x1, x1, x2, x2], [y, y + 0.5, y + 0.5, y], 'k-', lw=lw)
    ax.text((x1 + x2) / 2, y + 0.7, f"{fmt_p(p)} {sig(p)}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')


def ngroup_violin(vals_list, labels, colors, title, outpath,
                  ylabel='Delta percentile points', kw_result=None):
    """
    Generic N-group violin with Kruskal-Wallis title and Dunn post-hoc bars.

    Parameters
    ----------
    vals_list : list of 1D ndarray   Per-group values.
    labels    : list of str          Group labels (same length as vals_list).
    colors    : list of str          Hex colors (same length as vals_list).
    title     : str                  Figure title (KW stats appended automatically).
    outpath   : str                  Full output file path (PNG).
    ylabel    : str                  Y-axis label.
    kw_result : (H, p) tuple, optional
        Pre-computed Kruskal-Wallis result. Computed internally if None.
    """
    ns = [len(v) for v in vals_list]
    if kw_result is None:
        kw_h, kw_p = kruskal(*[v for v in vals_list if len(v) > 0])
    else:
        kw_h, kw_p = kw_result

    n_grp = len(vals_list)
    fig_w = max(10, n_grp * 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    positions = list(range(n_grp))
    parts = ax.violinplot(vals_list, positions=positions,
                          showmeans=True, showmedians=True, widths=0.7)
    for body, c in zip(parts['bodies'], colors):
        body.set_facecolor(c)
        body.set_alpha(0.65)
    style_violin(parts, ax)

    xl_short = [
        lbl.replace('Normal Prostate\n', 'Normal\n').replace('Primary PCa\n', 'Primary\n')
        for lbl in labels
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{lbl}\n(n={n})" for lbl, n in zip(xl_short, ns)],
        fontsize=11, fontweight='bold'
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(
        f"{title}\nKruskal-Wallis: H={kw_h:.2f}, {fmt_p(kw_p)} {sig(kw_p)}",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)

    gd = {lbl: v for lbl, v, n in zip(labels, vals_list, ns) if n > 0}
    pmat = dunn_pairwise(gd)
    y_top = max(v.max() for v in vals_list if len(v) > 0)
    k_sig = 0
    for i, j in combinations(range(n_grp), 2):
        la, lb = labels[i], labels[j]
        if la in pmat.index and lb in pmat.columns:
            p_ij = pmat.loc[la, lb]
            if p_ij >= 0.05:
                continue
            add_significance_bar(ax, i, j, y_top + 2 + k_sig * 4, p_ij)
            k_sig += 1

    plt.tight_layout()
    ax.set_ylim(top=y_top + 4 + max(k_sig, 0) * 4 + 8)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> Saved: {os.path.basename(outpath)}")
    plt.close()


def two_group_violin(va, vb, label_a, label_b, title, outpath,
                     color_a='#3498db', color_b='#c0392b',
                     ylabel='Delta percentile points'):
    """
    Two-group violin annotated with Mann-Whitney U p-value and rank-biserial r.

    Returns
    -------
    dict with keys 'p', 'sig', 'r_rb'.
    """
    u, p = mannwhitneyu(va, vb, alternative='two-sided')
    rb = rankbiserial(va, vb)
    fig, ax = plt.subplots(figsize=(7, 6))
    parts = ax.violinplot([va, vb], positions=[0, 1],
                          showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(color_a)
    parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor(color_b)
    parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_top = max(va.max(), vb.max())
    ax.plot([0, 1], [y_top + 1, y_top + 1], 'k-', lw=1.2)
    med_a = np.median(va)
    med_b = np.median(vb)
    ax.text(0.5, y_top + 1.5,
            f"{fmt_p(p)} {sig(p)}\nr_rb={rb:+.3f}  Median: {med_a:.2f} vs {med_b:.2f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"{label_a}\n(n={len(va)})", f"{label_b}\n(n={len(vb)})"],
        fontsize=11, fontweight='bold'
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> Saved: {os.path.basename(outpath)}")
    plt.close()
    return {'p': p, 'sig': sig(p), 'r_rb': rb}
