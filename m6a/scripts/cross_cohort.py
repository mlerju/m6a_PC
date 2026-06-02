#!/usr/bin/env python3
"""
cross_cohort.py — Six-group cross-cohort m6A analysis.

Progression cohorts (RNA-seq only):
  Group 0 — Normal Prostate       (GTEx v11):                n≈282
  Group 1 — Adjacent Normal       (TCGA Solid Tissue Normal): n≈52
  Group 2 — Primary PCa           (TCGA-PRAD):               n≈497
  Group 3 — mCSPC                 (GSE221601):                n≈52
  Group 4 — mCRPC-Adeno                                       n≈479
  Group 5 — mCRPC-SCNC                                        n≈59

Outputs: plots_cross_cohort/ (9 analytical figures)

Usage:
    micromamba run -n rnaseq python cross_cohort.py
"""
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
from PIL import Image as _PILImage

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.transparent': False,
})

# ── m6a package imports ───────────────────────────────────────────────────────
from m6a.config import (
    OUTDIR_CROSS_COHORT as OUTDIR,
    GROUP_LABELS, GROUP_COLORS, GROUP_LABELS_SHORT,
)
from m6a.genes import (
    WRITER_GENES, ERASER_GENES, READER_ONCOGENIC, READER_SUPPRESSIVE,
    ALL_M6A_GENES,
)
from m6a.stats import sig, fmt_p, rankbiserial, bh_fdr, dunn_pairwise
from m6a.plotting import (
    style_violin, add_significance_bar,
    ngroup_violin, two_group_violin,
)
from m6a.normalization import percentile_rank_matrix
from m6a.scoring import LR_WEIGHTS, W_ND, W_OR, compute_axes, get_vals
from m6a.data.loaders import (
    load_mcrpc, load_tcga, load_gtex, load_adj_normal, load_mcspc,
    build_common_universe,
)

os.makedirs(OUTDIR,  exist_ok=True)

# Remove stale plots from earlier pipeline versions
for _stale in [
    '03_per_gene_percentile_5groups.png', '03_per_gene_percentile_6groups.png',
    '03_per_gene_percentile_7groups.png',
    '04_axis1_net_deposition_5groups.png', '04_axis1_net_deposition_6groups.png',
    '04_axis1_net_deposition_7groups.png',
    '05_axis2_oncogenic_readout_5groups.png', '05_axis2_oncogenic_readout_6groups.png',
    '05_axis2_oncogenic_readout_7groups.png',
    '06_axis3_functional_impact_5groups.png', '06_axis3_functional_impact_6groups.png',
    '06_axis3_functional_impact_7groups.png',
    '22_heatmap_percentile_5groups.png', '22_heatmap_percentile_6groups.png',
    '22_heatmap_percentile_7groups.png',
    '24_2d_landscape_5groups.png', '24_2d_landscape_6groups.png',
    '24_2d_landscape_7groups.png',
]:
    _sp = os.path.join(OUTDIR, _stale)
    if os.path.exists(_sp):
        os.remove(_sp)

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 80)
print("  CROSS-COHORT m6A ANALYSIS  —  Percentile Rank Normalization  (6-group RNA-seq)")
print("=" * 80)

print("\n[1] Loading mCRPC ...")
mcrpc_expr, mcrpc_meta = load_mcrpc()

print("\n[2] Loading TCGA-PRAD ...")
tcga_expr, tcga_clin = load_tcga()

print("\n[3] Loading GTEx v11 normal prostate ...")
gtex_expr, _ = load_gtex()

print("\n[3b] Loading TCGA adjacent normals ...")
adj_normal_expr, _ = load_adj_normal()

print("\n[3c] Loading GSE221601 mCSPC ...")
mhspc_expr, _ = load_mcspc()

# =============================================================================
# COMMON GENE UNIVERSE
# =============================================================================
print("\n[4] Building 5-cohort common gene universe ...")
common_genes = build_common_universe(
    [mcrpc_expr, tcga_expr, gtex_expr, adj_normal_expr, mhspc_expr]
)
missing_m6a = [g for g in ALL_M6A_GENES if g not in common_genes]
if missing_m6a:
    print(f"    WARNING: m6A genes missing from 5-cohort intersection: {missing_m6a}")
else:
    print(f"    All {len(ALL_M6A_GENES)} m6A genes in common universe")

N_common = len(common_genes)
print(f"    Common (5-way): {N_common:,} genes (rank denominator for all cohorts)")

tcga_common       = tcga_expr[common_genes].fillna(0.0)
mcrpc_common      = mcrpc_expr[common_genes].fillna(0.0)
gtex_common       = gtex_expr[common_genes].fillna(0.0)
adj_normal_common = adj_normal_expr[common_genes].fillna(0.0)
mhspc_common      = mhspc_expr[common_genes].fillna(0.0)

# =============================================================================
# WITHIN-SAMPLE PERCENTILE RANK NORMALIZATION
# =============================================================================
print(f"\n[5] Computing within-sample percentile ranks ({N_common:,}-gene universe) ...")
gtex_pct       = percentile_rank_matrix(gtex_common)
tcga_pct       = percentile_rank_matrix(tcga_common)
mcrpc_pct      = percentile_rank_matrix(mcrpc_common)
adj_normal_pct = percentile_rank_matrix(adj_normal_common)
mhspc_pct      = percentile_rank_matrix(mhspc_common)

print(f"    GTEx pct matrix:   {gtex_pct.shape}")
print(f"    Adj Normal pct:    {adj_normal_pct.shape}")
print(f"    TCGA pct matrix:   {tcga_pct.shape}")
print(f"    mCSPC pct matrix:  {mhspc_pct.shape}")
print(f"    mCRPC pct matrix:  {mcrpc_pct.shape}")

# =============================================================================
# AXIS SCORES + SAMPLE GROUPS
# =============================================================================
print("\n[6] Computing axis scores for all cohorts ...")

# GTEx
nd, orr, fi = compute_axes(gtex_pct[ALL_M6A_GENES])
meta_gtex = pd.DataFrame({
    'm6A_Net_Deposition': nd, 'm6A_Oncogenic_Readout': orr, 'm6A_Functional_Impact': fi,
}, index=gtex_pct.index)

# Adjacent normals
pct_adj = adj_normal_pct[ALL_M6A_GENES]
nd, orr, fi = compute_axes(pct_adj)
meta_adj_normal = pd.DataFrame({
    'm6A_Net_Deposition': nd, 'm6A_Oncogenic_Readout': orr, 'm6A_Functional_Impact': fi,
}, index=adj_normal_pct.index)

# TCGA primary
tcga_common_idx = tcga_pct.index.intersection(tcga_clin.index)
pct_tcga = tcga_pct.loc[tcga_common_idx, ALL_M6A_GENES]
meta_tcga = tcga_clin.loc[tcga_common_idx].copy()
meta_tcga['gleason_group'] = meta_tcga['gleason_score'].apply(
    lambda x: np.nan if pd.isna(x) else (
        'GS <=6' if int(x) <= 6 else (
        'GS 7'   if int(x) == 7 else (
        'GS 8'   if int(x) == 8 else 'GS >=9'))))
nd, orr, fi = compute_axes(pct_tcga)
meta_tcga['m6A_Net_Deposition']    = nd
meta_tcga['m6A_Oncogenic_Readout'] = orr
meta_tcga['m6A_Functional_Impact'] = fi

# mCRPC
mcrpc_common_idx = mcrpc_pct.index.intersection(mcrpc_meta.index)
pct_mcrpc = mcrpc_pct.loc[mcrpc_common_idx, ALL_M6A_GENES]
meta_mcrpc = mcrpc_meta.loc[mcrpc_common_idx].copy()
idx_adeno = meta_mcrpc[meta_mcrpc['histology'] == 'Adenocarcinoma'].index
idx_scnc  = meta_mcrpc[meta_mcrpc['histology'] == 'SCNC'].index
nd, orr, fi = compute_axes(pct_mcrpc)
meta_mcrpc['m6A_Net_Deposition']    = nd
meta_mcrpc['m6A_Oncogenic_Readout'] = orr
meta_mcrpc['m6A_Functional_Impact'] = fi

# mCSPC (GSE221601)
pct_mhspc = mhspc_pct[ALL_M6A_GENES]
nd, orr, fi = compute_axes(pct_mhspc)
meta_mhspc = pd.DataFrame({
    'm6A_Net_Deposition': nd, 'm6A_Oncogenic_Readout': orr, 'm6A_Functional_Impact': fi,
}, index=mhspc_pct.index)

# Group index objects
GRP_NORMAL     = meta_gtex.index
GRP_ADJ_NORMAL = meta_adj_normal.index
GRP_PRIMARY    = meta_tcga.index
GRP_mCSPC_GSE  = meta_mhspc.index
GRP_ADENO      = idx_adeno
GRP_SCNC       = idx_scnc

# Ordered group descriptors (used by get_group_vals and loop-driven plots)
GROUP_METAS = [
    (meta_gtex,       GRP_NORMAL,     GROUP_LABELS[0]),
    (meta_adj_normal, GRP_ADJ_NORMAL, GROUP_LABELS[1]),
    (meta_tcga,       GRP_PRIMARY,    GROUP_LABELS[2]),
    (meta_mhspc,      GRP_mCSPC_GSE,  GROUP_LABELS[3]),
    (meta_mcrpc,      GRP_ADENO,      GROUP_LABELS[4]),
    (meta_mcrpc,      GRP_SCNC,       GROUP_LABELS[5]),
]

print(f"    Normal (GTEx):      n={len(GRP_NORMAL)}")
print(f"    Adjacent Normal:    n={len(GRP_ADJ_NORMAL)}")
print(f"    Primary (TCGA):     n={len(GRP_PRIMARY)}")
print(f"    mCSPC (GSE221601):  n={len(GRP_mCSPC_GSE)}")
print(f"    mCRPC-Adeno:        n={len(GRP_ADENO)}")
print(f"    mCRPC-SCNC:         n={len(GRP_SCNC)}")


def get_group_vals(axis):
    """Return list of per-group value arrays for one axis score."""
    return [get_vals(axis, idx, meta) for meta, idx, _ in GROUP_METAS]


def axis_violin(axis, title, filename, ylabel=None):
    """6-group violin for one axis score (thin wrapper around ngroup_violin)."""
    if ylabel is None:
        ylabel = axis.replace('m6A_', '').replace('_', ' ') + '\n(Delta percentile points)'
    ngroup_violin(
        get_group_vals(axis), GROUP_LABELS, GROUP_COLORS,
        title, os.path.join(OUTDIR, filename), ylabel=ylabel,
    )


# =============================================================================
# PLOTS
# =============================================================================
print("\n" + "=" * 80)
print("  GENERATING PLOTS")
print("=" * 80)


# ── Plots 04–06: Three-axis violins ──────────────────────────────────────────
print("\n--- Plots 04-06: Three-axis 6-group violins ---")
axis_violin('m6A_Net_Deposition',
            'm6A Net Deposition Axis — 6 Cohorts',
            '04_axis1_net_deposition_6groups.png')
axis_violin('m6A_Oncogenic_Readout',
            'm6A Oncogenic Readout Axis — 6 Cohorts',
            '05_axis2_oncogenic_readout_6groups.png')
axis_violin('m6A_Functional_Impact',
            'm6A Functional Impact Axis — 6 Cohorts',
            '06_axis3_functional_impact_6groups.png')

axes_short = [
    ('m6A_Net_Deposition',    'Net Deposition'),
    ('m6A_Oncogenic_Readout', 'Oncogenic Readout'),
    ('m6A_Functional_Impact', 'Functional Impact'),
]

# ── Plot 22: Heatmap — 22 genes × 6 groups ───────────────────────────────────
print("\n--- Plot 22: Gene × group median percentile heatmap ---")
heat_data = []
heat_rows = []
for gene in ALL_M6A_GENES:
    row = []
    for pct_df, idx, _ in GROUP_METAS:
        idx_c = idx.intersection(pct_df.index) if hasattr(pct_df, 'index') else []
        # Handle cases where pct_df covers all groups (mcrpc_pct used for GRP_ADENO and GRP_SCNC)
        vals = [
            gtex_pct[gene].values,
            adj_normal_pct[gene].values,
            tcga_pct[gene].values,
            mhspc_pct[gene].values,
            mcrpc_pct.loc[idx_adeno.intersection(mcrpc_pct.index), gene].values,
            mcrpc_pct.loc[idx_scnc.intersection(mcrpc_pct.index),  gene].values,
        ]
    row = [np.median(v) for v in vals]
    heat_data.append(row)
    heat_rows.append(gene)

heat_df = pd.DataFrame(heat_data, index=heat_rows,
                        columns=[l.replace('\n', ' ') for l in GROUP_LABELS])
fig, ax = plt.subplots(figsize=(14, 14))
sns.heatmap(heat_df, annot=True, fmt='.1f', cmap='RdBu_r', center=50,
            vmin=30, vmax=70, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Median within-sample percentile rank'})
ax.set_title(
    f'm6A Gene Percentile Rank Heatmap — 6 Cohorts\n'
    f'Median within-sample percentile rank ({N_common:,}-gene universe)',
    fontsize=14, fontweight='bold', pad=12,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '22_heatmap_percentile_6groups.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 22_heatmap_percentile_6groups.png")
plt.close()

# ── Plot 23: Effect-size summary (15 pairs × 3 axes) ─────────────────────────
print("\n--- Plot 23: Effect-size summary bar chart ---")
comparisons_15 = [
    ('Normal', GRP_NORMAL, meta_gtex,
     'AdjNorm', GRP_ADJ_NORMAL, meta_adj_normal),
    ('Normal', GRP_NORMAL, meta_gtex,
     'Primary', GRP_PRIMARY, meta_tcga),
    ('Normal', GRP_NORMAL, meta_gtex,
     'mCSPC',  GRP_mCSPC_GSE, meta_mhspc),
    ('Normal', GRP_NORMAL, meta_gtex,
     'Adeno',  GRP_ADENO, meta_mcrpc),
    ('Normal', GRP_NORMAL, meta_gtex,
     'SCNC',   GRP_SCNC, meta_mcrpc),
    ('AdjNorm', GRP_ADJ_NORMAL, meta_adj_normal,
     'Primary', GRP_PRIMARY, meta_tcga),
    ('AdjNorm', GRP_ADJ_NORMAL, meta_adj_normal,
     'mCSPC',   GRP_mCSPC_GSE, meta_mhspc),
    ('AdjNorm', GRP_ADJ_NORMAL, meta_adj_normal,
     'Adeno',   GRP_ADENO, meta_mcrpc),
    ('AdjNorm', GRP_ADJ_NORMAL, meta_adj_normal,
     'SCNC',    GRP_SCNC, meta_mcrpc),
    ('Primary', GRP_PRIMARY, meta_tcga,
     'mCSPC',   GRP_mCSPC_GSE, meta_mhspc),
    ('Primary', GRP_PRIMARY, meta_tcga,
     'Adeno',   GRP_ADENO, meta_mcrpc),
    ('Primary', GRP_PRIMARY, meta_tcga,
     'SCNC',    GRP_SCNC, meta_mcrpc),
    ('mCSPC', GRP_mCSPC_GSE, meta_mhspc,
     'Adeno', GRP_ADENO, meta_mcrpc),
    ('mCSPC', GRP_mCSPC_GSE, meta_mhspc,
     'SCNC',  GRP_SCNC, meta_mcrpc),
    ('Adeno', GRP_ADENO, meta_mcrpc,
     'SCNC',  GRP_SCNC, meta_mcrpc),
]

es_rows = []
for axis, axis_lbl in axes_short:
    raw_ps = [
        mannwhitneyu(get_vals(axis, ia, ma), get_vals(axis, ib, mb),
                     alternative='two-sided')[1]
        for la, ia, ma, lb, ib, mb in comparisons_15
    ]
    adj_ps = bh_fdr(raw_ps)
    for (la, ia, ma, lb, ib, mb), p_adj in zip(comparisons_15, adj_ps):
        va = get_vals(axis, ia, ma)
        vb = get_vals(axis, ib, mb)
        es_rows.append({
            'axis': axis_lbl,
            'comparison': f'{la}\nvs\n{lb}',
            'r_rb': rankbiserial(va, vb),
            'p_adj': p_adj,
            'sig': sig(p_adj),
        })

es_df = pd.DataFrame(es_rows)
fig, axes_es = plt.subplots(1, 3, figsize=(28, 7), sharey=True)
for ai, (axis_lbl, ax) in enumerate(zip([a for _, a in axes_short], axes_es)):
    sub = es_df[es_df['axis'] == axis_lbl].reset_index(drop=True)
    colors_es = ['#e74c3c' if r > 0 else '#3498db' for r in sub['r_rb']]
    ax.barh(sub['comparison'], sub['r_rb'], color=colors_es, alpha=0.85, edgecolor='black')
    for idx_es, row_es in sub.iterrows():
        if row_es['sig'] != 'ns':
            ax.text(row_es['r_rb'] + 0.01 * np.sign(row_es['r_rb']),
                    idx_es, row_es['sig'], va='center', fontsize=9, fontweight='bold')
    ax.axvline(0, color='black', lw=1.0)
    ax.set_xlabel('Rank-biserial r (BH-adj)', fontsize=11, fontweight='bold')
    ax.set_title(axis_lbl, fontsize=13, fontweight='bold')
    ax.set_xlim(-1.05, 1.05)
    if ai == 0:
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.85, label='A > B'),
            plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.85, label='A < B'),
        ]
        ax.legend(handles, ['A > B', 'A < B'], fontsize=12, loc='lower left')
plt.suptitle('Effect-Size Summary: 15 Pairwise Comparisons × 3 Axes\n'
             'Mann-Whitney r (rank-biserial), BH-FDR corrected per axis',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '23_effect_size_summary.png'), dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 23_effect_size_summary.png")
plt.close()

# ── Plot 24: 2D landscape scatter ─────────────────────────────────────────────
print("\n--- Plot 24: 2D epitranscriptomic landscape (6 groups) ---")
fig, ax = plt.subplots(figsize=(11, 8))
plot_specs = [
    (meta_gtex,       GRP_NORMAL,     GROUP_COLORS[0], GROUP_LABELS[0], 'D', 0.45),
    (meta_adj_normal, GRP_ADJ_NORMAL, GROUP_COLORS[1], GROUP_LABELS[1], '^', 0.70),
    (meta_tcga,       GRP_PRIMARY,    GROUP_COLORS[2], GROUP_LABELS[2], 'o', 0.30),
    (meta_mhspc,      GRP_mCSPC_GSE,  GROUP_COLORS[3], GROUP_LABELS[3], 'P', 0.65),
    (meta_mcrpc,      GRP_ADENO,      GROUP_COLORS[4], GROUP_LABELS[4], 's', 0.40),
    (meta_mcrpc,      GRP_SCNC,       GROUP_COLORS[5], GROUP_LABELS[5], 'v', 0.70),
]
for meta_s, idx_s, color, lbl, marker, alpha in plot_specs:
    idx_c = idx_s.intersection(meta_s.index)
    ax.scatter(meta_s.loc[idx_c, 'm6A_Net_Deposition'],
               meta_s.loc[idx_c, 'm6A_Oncogenic_Readout'],
               c=color, alpha=alpha, s=25, label=f"{lbl} (n={len(idx_c)})",
               marker=marker, linewidths=0)
ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
ax.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Axis 1: Net Deposition (Delta percentile points)', fontsize=14, fontweight='bold')
ax.set_ylabel('Axis 2: Oncogenic Readout (Delta percentile points)', fontsize=14, fontweight='bold')
ax.set_title('m6A Epitranscriptomic Landscape\n', fontsize=16, fontweight='bold', pad=12)
ax.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '24_2d_landscape_6groups.png'), dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 24_2d_landscape_6groups.png")
plt.close()

# ── Plot 25: Primary PCa by Gleason grade ────────────────────────────────────
print("\n--- Plot 25: Primary PCa by Gleason ---")
gs_order  = ['GS <=6', 'GS 7', 'GS 8', 'GS >=9']
gs_colors = ['#1a9850', '#3498db', '#e67e22', '#c0392b']
gs_vals   = [meta_tcga[meta_tcga['gleason_group'] == g]['m6A_Functional_Impact'].dropna().values
             for g in gs_order]
gs_ns     = [len(v) for v in gs_vals]
gs_nz     = [v for v in gs_vals if len(v) > 0]
gs_kw     = kruskal(*gs_nz) if len(gs_nz) >= 2 else (np.nan, np.nan)

fig, ax = plt.subplots(figsize=(10, 9))
pos = [i for i, v in enumerate(gs_vals) if len(v) > 0]
parts = ax.violinplot([v for v in gs_vals if len(v) > 0], positions=pos,
                      showmeans=True, showmedians=True, widths=0.7)
for body, c in zip(parts['bodies'], [gs_colors[i] for i, v in enumerate(gs_vals) if len(v) > 0]):
    body.set_facecolor(c); body.set_alpha(0.65)
style_violin(parts, ax)
gs_dict  = {g: v for g, v in zip(gs_order, gs_vals) if len(v) > 0}
gs_pmat  = dunn_pairwise(gs_dict, correction='bh')
gs_y_top = max(v.max() for v in gs_vals if len(v) > 0)
gs_y_bot = min(v.min() for v in gs_vals if len(v) > 0)
gs_k = 0
for gi, gj in [(0, 1), (1, 2), (2, 3), (0, 3)]:
    ga, gb = gs_order[gi], gs_order[gj]
    if len(gs_vals[gi]) == 0 or len(gs_vals[gj]) == 0:
        continue
    if ga in gs_pmat.index and gb in gs_pmat.columns:
        p_ij = gs_pmat.loc[ga, gb]
        if p_ij < 0.05:
            add_significance_bar(ax, gi, gj, gs_y_top + 3 + gs_k * 6, p_ij)
            gs_k += 1
ax.set_xticks(range(len(gs_order)))
ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(gs_order, gs_ns)],
                   fontsize=11, fontweight='bold')
ax.set_ylabel('Functional Impact (Delta percentile points)', fontsize=12, fontweight='bold')
kw_str = (f"H={gs_kw[0]:.2f}, {fmt_p(gs_kw[1])} {sig(gs_kw[1])}"
          if not np.isnan(gs_kw[1]) else "")
dunn_note = "  |  Pairwise: Dunn (BH-FDR)" if gs_k > 0 else ""
ax.set_title(f'm6A Functional Impact by Gleason Grade (Primary PCa, TCGA-PRAD)\n'
             f'Kruskal-Wallis: {kw_str}{dunn_note}',
             fontsize=12, fontweight='bold', pad=10)
ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
plt.tight_layout()
ax.set_ylim(bottom=gs_y_bot - 3, top=gs_y_top + 3 + gs_k * 6 + 5)
plt.savefig(os.path.join(OUTDIR, '25_tcga_gleason_functional_impact.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 25_tcga_gleason_functional_impact.png")
plt.close()


# ── Plot 27: Sensitivity — YTHDF1-only oncogenic readout ─────────────────────
print("\n--- Plot 27: Sensitivity — YTHDF1-only readout ---")

def ythdf1_readout(pct_df):
    return pct_df['YTHDF1'] - pct_df[READER_SUPPRESSIVE].mean(axis=1)

meta_gtex['m6A_YTHDF1_Readout']       = ythdf1_readout(gtex_pct[ALL_M6A_GENES])
meta_adj_normal['m6A_YTHDF1_Readout'] = ythdf1_readout(pct_adj)
meta_tcga['m6A_YTHDF1_Readout']       = ythdf1_readout(pct_tcga)
meta_mcrpc['m6A_YTHDF1_Readout']      = ythdf1_readout(pct_mcrpc)
meta_mhspc['m6A_YTHDF1_Readout']      = ythdf1_readout(pct_mhspc)

fig, axes2 = plt.subplots(1, 2, figsize=(18, 7))
for ax, axis, title_str, ylabel_str in [
    (axes2[0], 'm6A_Oncogenic_Readout',
     f'Full Oncogenic Readout ({" + ".join(READER_ONCOGENIC)})',
     'Onco. Readers - Supp. Readers (Delta pct pts)'),
    (axes2[1], 'm6A_YTHDF1_Readout',
     'YTHDF1-only Readout',
     'YTHDF1 - mean(Supp. Readers) (Delta pct pts)'),
]:
    vals = get_group_vals(axis)
    ns   = [len(v) for v in vals]
    kw_h, kw_p = kruskal(*vals)
    parts = ax.violinplot(vals, positions=list(range(len(vals))),
                          showmeans=True, showmedians=True, widths=0.7)
    for body, c in zip(parts['bodies'], GROUP_COLORS):
        body.set_facecolor(c); body.set_alpha(0.65)
    style_violin(parts, ax)
    gd   = {lbl: v for lbl, v, n in zip(GROUP_LABELS, vals, ns) if n > 0}
    pmat = dunn_pairwise(gd)
    y_top = max(v.max() for v in vals if len(v) > 0)
    k_sig = 0
    for i, j in combinations(range(len(vals)), 2):
        la, lb = GROUP_LABELS[i], GROUP_LABELS[j]
        if la in pmat.index and lb in pmat.columns:
            if pmat.loc[la, lb] < 0.05:
                add_significance_bar(ax, i, j, y_top + 2 + k_sig * 4, pmat.loc[la, lb])
                k_sig += 1
    ax.set_ylim(top=y_top + 4 + max(k_sig, 0) * 4 + 8)
    ax.set_xticks(list(range(len(vals))))
    ax.set_xticklabels(
        [f"{lbl.replace('Normal Prostate', 'Normal').replace('Primary PCa', 'Primary')}\n(n={n})"
         for lbl, n in zip(GROUP_LABELS, ns)],
        fontsize=9, fontweight='bold',
    )
    ax.set_ylabel(ylabel_str, fontsize=11, fontweight='bold')
    ax.set_title(f"{title_str}\nK-W: H={kw_h:.2f}, {fmt_p(kw_p)} {sig(kw_p)}",
                 fontsize=11, fontweight='bold', pad=10)
    ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
plt.suptitle(
    f'Full ({len(READER_ONCOGENIC)}-reader) vs YTHDF1-only Oncogenic Readout\n'
    'Within-sample percentile rank; Dunn pairwise, BH-FDR correction\n',
    fontsize=18, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '27_sensitivity_ythdf1_only_readout.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 27_sensitivity_ythdf1_only_readout.png")
plt.close()

# ── Plot 28: IGF2BP re-expression (raw log2 values) ────────────────────────────
print("\n--- Plots 28-29: IGF2BP re-expression ---")
igf2bp_genes = ['IGF2BP1', 'IGF2BP2', 'IGF2BP3']
idx_adeno_c  = idx_adeno.intersection(mcrpc_common.index)
idx_scnc_c   = idx_scnc.intersection(mcrpc_common.index)

igf2bp_raw = {}
for gene in igf2bp_genes:
    igf2bp_raw[gene] = [
        gtex_common[gene].values,
        adj_normal_common[gene].values,
        tcga_common[gene].values,
        mhspc_common[gene].values,
        mcrpc_common.loc[idx_adeno_c, gene].values,
        mcrpc_common.loc[idx_scnc_c,  gene].values,
    ]

lbls_raw = ['Normal\n(GTEx)', 'Adj Normal\n(TCGA)', 'Primary\n(TCGA)',
            'mCSPC\n(GSE)', 'mCRPC-Adeno', 'mCRPC-SCNC']

# Plot 28: Violin
fig, axes3 = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
for ax, gene in zip(axes3, igf2bp_genes):
    gvals = igf2bp_raw[gene]
    ns    = [len(v) for v in gvals]
    kw_h_g, kw_p_g = kruskal(*gvals)
    parts = ax.violinplot(gvals, positions=[0, 1, 2, 3, 4, 5],
                          showmeans=True, showmedians=True, widths=0.7)
    for body, c in zip(parts['bodies'], GROUP_COLORS):
        body.set_facecolor(c); body.set_alpha(0.65)
    style_violin(parts, ax)
    pairs_raw  = list(combinations(range(6), 2))
    pvals_pw   = [mannwhitneyu(gvals[i], gvals[j], alternative='two-sided')[1]
                  for i, j in pairs_raw]
    pvals_corr = [min(p * 15, 1.0) for p in pvals_pw]
    y_top = max(v.max() for v in gvals)
    k_sig = 0
    for (xi, xj), pc in zip(pairs_raw, pvals_corr):
        if pc >= 0.05:
            continue
        add_significance_bar(ax, xi, xj, y_top + 0.2 + k_sig * 0.45, pc)
        k_sig += 1
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(lbls_raw, ns)],
                       fontsize=8.5, fontweight='bold')
    ax.set_ylabel('log2(value+1)\n[TPM for GTEx, CPM for others]', fontsize=10, fontweight='bold')
    ax.set_title(f'{gene}  (oncofetal m6A reader)\n'
                 f'K-W: H={kw_h_g:.2f}, {fmt_p(kw_p_g)} {sig(kw_p_g)}',
                 fontsize=11, fontweight='bold', pad=10)
    ax.axhline(0, color='grey', lw=0.6, ls='--', alpha=0.4)
    for pos, m in enumerate([np.median(v) for v in gvals]):
        ax.text(pos, m + 0.05, f'med={m:.2f}', ha='center', va='bottom',
                fontsize=8, fontstyle='italic')
plt.suptitle(
    'IGF2BP1/2/3: Normal Prostate → Primary PCa → mCRPC (6 RNA-seq groups)\n'
    '(Raw log2 values; Bonferroni-corrected pairwise p-values)\n'
    'Near-absent in normal & primary PCa  |  Universally re-expressed in mCRPC  |  CHAARTED excluded (microarray)',
    fontsize=13, fontweight='bold', y=1.03,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '28_igf2bp_reexpression_raw_log2.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 28_igf2bp_reexpression_raw_log2.png")
plt.close()


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("  COMPLETE  —  Cross-Cohort m6A Analysis (6 groups, RNA-seq, Percentile Ranks)")
print("=" * 80)
print(f"\n  Output: {OUTDIR}")
print(f"  Plots:  9")
print(f"  Normalization: within-sample percentile rank on {N_common:,}-gene 5-cohort universe")
print(f"\n  Group sizes:")
for lbl, idx in [('Normal', GRP_NORMAL), ('Adjacent Normal', GRP_ADJ_NORMAL),
                  ('Primary', GRP_PRIMARY), ('mCSPC', GRP_mCSPC_GSE),
                  ('mCRPC-Adeno', GRP_ADENO), ('mCRPC-SCNC', GRP_SCNC)]:
    print(f"    {lbl:20s}: n={len(idx)}")

# =============================================================================
# POST-PROCESS: convert RGBA PNGs to RGB (white background)
# =============================================================================
def _rgba_to_rgb(dirpath):
    n = 0
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith('.png'):
            continue
        fpath = os.path.join(dirpath, fname)
        img = _PILImage.open(fpath)
        if img.mode == 'RGBA':
            bg = _PILImage.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            bg.save(fpath, 'PNG')
            n += 1
    return n

_conv = _rgba_to_rgb(OUTDIR)
if _conv:
    print(f"  Converted {_conv} RGBA → RGB PNG(s) in plots_cross_cohort/")

print("\n  Done.")
