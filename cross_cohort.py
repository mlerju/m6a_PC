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

Outputs: plots_cross_cohort/ (29 analytical figures)
         plots_presentation/  (8 presentation figures P1–P8)

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
    OUTDIR_PRESENTATION as PRESDIR,
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
os.makedirs(PRESDIR, exist_ok=True)

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

# ── Plot 01: Percentile rank QC ──────────────────────────────────────────────
print("\n--- Plot 01: Percentile rank QC (6 cohorts) ---")
fig, ax = plt.subplots(figsize=(22, 6))
x = np.arange(len(ALL_M6A_GENES))
w = 0.12
meds = {
    GROUP_LABELS[0]: gtex_pct[ALL_M6A_GENES].median(),
    GROUP_LABELS[1]: adj_normal_pct[ALL_M6A_GENES].median(),
    GROUP_LABELS[2]: tcga_pct[ALL_M6A_GENES].median(),
    GROUP_LABELS[3]: mhspc_pct[ALL_M6A_GENES].median(),
    GROUP_LABELS[4]: mcrpc_pct.loc[idx_adeno.intersection(mcrpc_pct.index), ALL_M6A_GENES].median(),
    GROUP_LABELS[5]: mcrpc_pct.loc[idx_scnc.intersection(mcrpc_pct.index),  ALL_M6A_GENES].median(),
}
offsets = [-2.5*w, -1.5*w, -0.5*w, 0.5*w, 1.5*w, 2.5*w]
for (name, meds_s), off, col in zip(meds.items(), offsets, GROUP_COLORS):
    ax.bar(x + off, meds_s, w, label=name.replace('\n', ' '), color=col, alpha=0.82, edgecolor='none')
ax.axhline(50, color='black', lw=1.0, ls='--', alpha=0.5, label='50th pct = genome median')
ax.set_xticks(x)
ax.set_xticklabels(ALL_M6A_GENES, fontsize=9, rotation=45, ha='right', fontweight='bold')
ax.set_ylabel('Median within-sample percentile rank', fontsize=12, fontweight='bold')
ax.set_title(
    f'm6A Gene Percentile Ranks — 6 Cohorts\n'
    f'Within-sample percentile rank ({N_common:,}-gene universe)',
    fontsize=13, fontweight='bold', pad=12,
)
ax.legend(fontsize=8, loc='upper left', ncol=3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '01_percentile_rank_qc.png'), dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 01_percentile_rank_qc.png")
plt.close()

# ── Plot 02: PCA on m6A percentile matrix ────────────────────────────────────
print("\n--- Plot 02: PCA on m6A percentile matrix ---")
from sklearn.decomposition import PCA

pca_rows, pca_labels_list, pca_colors_list = [], [], []
pca_specs = [
    (gtex_pct,       GRP_NORMAL,     GROUP_COLORS[0], GROUP_LABELS_SHORT[0]),
    (adj_normal_pct, GRP_ADJ_NORMAL, GROUP_COLORS[1], GROUP_LABELS_SHORT[1]),
    (tcga_pct,       GRP_PRIMARY,    GROUP_COLORS[2], GROUP_LABELS_SHORT[2]),
    (mhspc_pct,      GRP_mCSPC_GSE,  GROUP_COLORS[3], GROUP_LABELS_SHORT[3]),
    (mcrpc_pct,      GRP_ADENO,      GROUP_COLORS[4], GROUP_LABELS_SHORT[4]),
    (mcrpc_pct,      GRP_SCNC,       GROUP_COLORS[5], GROUP_LABELS_SHORT[5]),
]
for pct_df, idx, color, label in pca_specs:
    idx_c = idx.intersection(pct_df.index)
    sub = pct_df.loc[idx_c, ALL_M6A_GENES]
    pca_rows.append(sub)
    pca_labels_list.extend([label] * len(idx_c))
    pca_colors_list.extend([color] * len(idx_c))

pca_mat = pd.concat(pca_rows)[ALL_M6A_GENES].fillna(0.0)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(pca_mat.values)
fig, ax = plt.subplots(figsize=(11, 8))
for (pct_df, idx, color, label) in pca_specs:
    mask = [l == label for l in pca_labels_list]
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, alpha=0.45, s=18, label=label, linewidths=0)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13, fontweight='bold')
ax.set_title('PCA on 22-gene m6A Percentile Matrix\n6 cohorts', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '02_pca_m6a_percentile.png'), dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 02_pca_m6a_percentile.png")
plt.close()

# ── Plot 03: Per-gene violins, 6 groups (4×6 grid) ───────────────────────────
print("\n--- Plot 03: Per-gene violins (4×6 grid) ---")
n_cols, n_rows = 6, 4
fig, axes3 = plt.subplots(n_rows, n_cols, figsize=(26, 18), sharey=False)
for gi, gene in enumerate(ALL_M6A_GENES):
    row, col = divmod(gi, n_cols)
    ax = axes3[row][col]
    grp_vals = [
        gtex_pct[gene].values,
        adj_normal_pct[gene].values,
        tcga_pct[gene].values,
        mhspc_pct[gene].values,
        mcrpc_pct.loc[idx_adeno.intersection(mcrpc_pct.index), gene].values,
        mcrpc_pct.loc[idx_scnc.intersection(mcrpc_pct.index),  gene].values,
    ]
    parts = ax.violinplot(grp_vals, positions=list(range(6)),
                          showmeans=True, showmedians=True, widths=0.7)
    for body, c in zip(parts['bodies'], GROUP_COLORS):
        body.set_facecolor(c); body.set_alpha(0.65)
    style_violin(parts, ax, legend=False)
    kw_h, kw_p = kruskal(*grp_vals)
    ax.set_title(f'{gene}  KW p={kw_p:.2e} {sig(kw_p)}', fontsize=9, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_xticklabels(GROUP_LABELS_SHORT, fontsize=7, rotation=30, ha='right')
    ax.axhline(50, color='grey', lw=0.6, ls=':', alpha=0.6)
fig.suptitle(
    f'm6A Gene Percentile Ranks — 6 Cohorts (within-sample, {N_common:,}-gene universe)',
    fontsize=16, fontweight='bold', y=1.01,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '03_per_gene_percentile_6groups.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 03_per_gene_percentile_6groups.png")
plt.close()

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

# ── Plots 07–21: Pairwise two-group violins for 5 key transitions ─────────────
pairwise_specs = [
    # (grp_a_label, grp_a_idx, grp_a_meta, grp_b_label, grp_b_idx, grp_b_meta, base_num, ca, cb)
    ('Normal (GTEx)', GRP_NORMAL, meta_gtex,
     'Adjacent Normal (TCGA)', GRP_ADJ_NORMAL, meta_adj_normal,
     '07', GROUP_COLORS[0], GROUP_COLORS[1]),
    ('Adjacent Normal (TCGA)', GRP_ADJ_NORMAL, meta_adj_normal,
     'Primary PCa (TCGA)', GRP_PRIMARY, meta_tcga,
     '10', GROUP_COLORS[1], GROUP_COLORS[2]),
    ('Normal (GTEx)', GRP_NORMAL, meta_gtex,
     'Primary PCa (TCGA)', GRP_PRIMARY, meta_tcga,
     '13', GROUP_COLORS[0], GROUP_COLORS[2]),
    ('Primary PCa (TCGA)', GRP_PRIMARY, meta_tcga,
     'mCRPC-Adeno', GRP_ADENO, meta_mcrpc,
     '16', GROUP_COLORS[2], GROUP_COLORS[4]),
    ('Primary PCa (TCGA)', GRP_PRIMARY, meta_tcga,
     'mCRPC-SCNC', GRP_SCNC, meta_mcrpc,
     '19', GROUP_COLORS[2], GROUP_COLORS[5]),
]
axes_short = [
    ('m6A_Net_Deposition',    'Net Deposition'),
    ('m6A_Oncogenic_Readout', 'Oncogenic Readout'),
    ('m6A_Functional_Impact', 'Functional Impact'),
]
print("\n--- Plots 07-21: Pairwise two-group comparisons ---")
for la, ia, ma, lb, ib, mb, base, ca, cb in pairwise_specs:
    for offset_i, (axis, axis_lbl) in enumerate(axes_short):
        fn = f"{int(base) + offset_i:02d}_{axis}_{la[:4].replace(' ', '_')}vs{lb[:4].replace(' ', '_')}.png"
        va = get_vals(axis, ia, ma)
        vb = get_vals(axis, ib, mb)
        two_group_violin(
            va, vb, la, lb,
            f'{axis_lbl}: {la} vs {lb}',
            os.path.join(OUTDIR, fn),
            color_a=ca, color_b=cb,
        )

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

# ── Plot 26: Statistical summary table ───────────────────────────────────────
print("\n--- Plot 26: Statistical summary table ---")
rows = []
for axis, axis_lbl in axes_short:
    v_n   = get_vals(axis, GRP_NORMAL,    meta_gtex)
    v_an  = get_vals(axis, GRP_ADJ_NORMAL, meta_adj_normal)
    v_p   = get_vals(axis, GRP_PRIMARY,   meta_tcga)
    v_gse = get_vals(axis, GRP_mCSPC_GSE, meta_mhspc)
    v_a   = get_vals(axis, GRP_ADENO,     meta_mcrpc)
    v_s   = get_vals(axis, GRP_SCNC,      meta_mcrpc)
    kw_h, kw_p = kruskal(v_n, v_an, v_p, v_gse, v_a, v_s)
    pairs_data = [
        ('Normal', 'AdjNorm', v_n, v_an),
        ('Normal', 'Primary', v_n, v_p),
        ('Normal', 'mCSPC',   v_n, v_gse),
        ('Normal', 'Adeno',   v_n, v_a),
        ('Normal', 'SCNC',    v_n, v_s),
        ('AdjNorm', 'Primary', v_an, v_p),
        ('AdjNorm', 'mCSPC',   v_an, v_gse),
        ('AdjNorm', 'Adeno',   v_an, v_a),
        ('AdjNorm', 'SCNC',    v_an, v_s),
        ('Primary', 'mCSPC',   v_p, v_gse),
        ('Primary', 'Adeno',   v_p, v_a),
        ('Primary', 'SCNC',    v_p, v_s),
        ('mCSPC', 'Adeno', v_gse, v_a),
        ('mCSPC', 'SCNC',  v_gse, v_s),
        ('Adeno', 'SCNC',  v_a, v_s),
    ]
    raw_pvals = [mannwhitneyu(va, vb, alternative='two-sided')[1] for la, lb, va, vb in pairs_data]
    adj_pvals = bh_fdr(raw_pvals)
    for (la, lb, va, vb), p_raw, p_adj in zip(pairs_data, raw_pvals, adj_pvals):
        rb = rankbiserial(va, vb)
        rows.append({
            'Axis': axis_lbl,
            'Comparison': f'{la} vs {lb}',
            'n_A': len(va), 'n_B': len(vb),
            'Med_A [Q1–Q3]': f"{np.median(va):.2f} [{np.percentile(va,25):.2f}–{np.percentile(va,75):.2f}]",
            'Med_B [Q1–Q3]': f"{np.median(vb):.2f} [{np.percentile(vb,25):.2f}–{np.percentile(vb,75):.2f}]",
            'p (raw)':    fmt_p(p_raw),
            'p_adj (BH)': fmt_p(p_adj),
            'Sig':        sig(p_adj),
            'r_rb':       f'{rb:+.3f}',
            'K-W p':      fmt_p(kw_p),
        })

summ_df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(28, 18))
ax.axis('off')
col_labels = list(summ_df.columns)
tbl = ax.table(cellText=summ_df.values, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.0, 1.25)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i, row in summ_df.reset_index(drop=True).iterrows():
    bg = '#ecf0f1' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        tbl[i + 1, j].set_facecolor(bg)
    if row['Sig'] != 'ns':
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_text_props(fontweight='bold')
ax.set_title(
    'Cross-Cohort m6A Statistical Summary — Within-Sample Percentile Rank Normalization\n'
    '6 cohorts | Mann-Whitney U (two-sided) | BH-FDR adjusted per axis | r_rb: rank-biserial r',
    fontsize=11, fontweight='bold', pad=20,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '26_statistical_summary_table.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 26_statistical_summary_table.png")
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

# ── Plots 28–29: IGF2BP re-expression (raw log2 values) ──────────────────────
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

# Plot 29: Strip + box
np.random.seed(42)
fig, axes4 = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
for ax, gene in zip(axes4, igf2bp_genes):
    gvals = igf2bp_raw[gene]
    ns    = [len(v) for v in gvals]
    for pos, (v, c) in enumerate(zip(gvals, GROUP_COLORS)):
        jitter = np.random.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(pos + jitter, v, color=c, alpha=0.20, s=7, linewidths=0)
        ax.boxplot(v, positions=[pos], widths=0.32, patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color='black', lw=2),
                   boxprops=dict(facecolor=c, alpha=0.5),
                   whiskerprops=dict(lw=1.2),
                   capprops=dict(lw=1.2))
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
                       fontsize=10, fontweight='bold')
    ax.set_ylabel('log2(value+1)', fontsize=14, fontweight='bold')
    pcts_det = [(v > 1.0).mean() * 100 for v in gvals]
    kw_h_g, kw_p_g = kruskal(*gvals)
    ax.set_title(f'{gene}  |  K-W {fmt_p(kw_p_g)} {sig(kw_p_g)}\n'
                 f'% >log2=1: ' + ' / '.join(f'{p:.0f}%' for p in pcts_det),
                 fontsize=12, fontweight='bold', pad=10)
plt.suptitle('IGF2BP1/2/3: Sample-Level Expression\n', fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '29_igf2bp_strip_box_raw_log2.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  -> Saved: 29_igf2bp_strip_box_raw_log2.png")
plt.close()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("  COMPLETE  —  Cross-Cohort m6A Analysis (6 groups, RNA-seq, Percentile Ranks)")
print("=" * 80)
print(f"\n  Output: {OUTDIR}")
print(f"  Plots:  29")
print(f"  Normalization: within-sample percentile rank on {N_common:,}-gene 5-cohort universe")
print(f"\n  Group sizes:")
for lbl, idx in [('Normal', GRP_NORMAL), ('Adjacent Normal', GRP_ADJ_NORMAL),
                  ('Primary', GRP_PRIMARY), ('mCSPC', GRP_mCSPC_GSE),
                  ('mCRPC-Adeno', GRP_ADENO), ('mCRPC-SCNC', GRP_SCNC)]:
    print(f"    {lbl:20s}: n={len(idx)}")

# =============================================================================
# PRESENTATION FIGURES  P1–P8
# =============================================================================
print("\n" + "=" * 80)
print("  GENERATING PRESENTATION FIGURES  (P1–P8)")
print("=" * 80)

_PS = {
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 20, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 13, 'figure.facecolor': 'white',
}
_CL = ['Normal', 'Adj Norm', 'Primary', 'mCSPC', 'mCRPC-Adeno', 'mCRPC-SCNC']
_pspecs = [
    (meta_gtex,       GRP_NORMAL,     GROUP_COLORS[0], GROUP_LABELS[0], 'D', 0.35,  7),
    (meta_adj_normal, GRP_ADJ_NORMAL, GROUP_COLORS[1], GROUP_LABELS[1], '^', 0.55,  7),
    (meta_tcga,       GRP_PRIMARY,    GROUP_COLORS[2], GROUP_LABELS[2], 'o', 0.20,  6),
    (meta_mhspc,      GRP_mCSPC_GSE,  GROUP_COLORS[3], GROUP_LABELS[3], 'P', 0.70, 10),
    (meta_mcrpc,      GRP_ADENO,      GROUP_COLORS[4], GROUP_LABELS[4], 's', 0.25,  6),
    (meta_mcrpc,      GRP_SCNC,       GROUP_COLORS[5], GROUP_LABELS[5], 'v', 0.70,  9),
]

# P1 — 2D Landscape
print("\n--- FIG P1: 2D landscape ---")
with plt.rc_context(_PS):
    fig, ax = plt.subplots(figsize=(16, 9))
    centroids_x, centroids_y, centroid_colors, centroid_labels = [], [], [], []
    for meta_s, idx_s, color, lbl, marker, alpha, ms in _pspecs:
        idx_c = idx_s.intersection(meta_s.index)
        ax.scatter(meta_s.loc[idx_c, 'm6A_Net_Deposition'],
                   meta_s.loc[idx_c, 'm6A_Oncogenic_Readout'],
                   c=color, alpha=alpha, s=ms**2, marker=marker,
                   linewidths=0, zorder=2, label=f'{lbl}  (n={len(idx_c)})')
        cx = meta_s.loc[idx_c, 'm6A_Net_Deposition'].median()
        cy = meta_s.loc[idx_c, 'm6A_Oncogenic_Readout'].median()
        centroids_x.append(cx); centroids_y.append(cy)
        centroid_colors.append(color); centroid_labels.append(lbl)
        ax.scatter(cx, cy, c=color, s=320, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=6)
    for i, j in [(2, 3), (3, 4)]:
        ax.annotate('', xy=(centroids_x[j], centroids_y[j]),
                    xytext=(centroids_x[i], centroids_y[i]),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                   lw=2.0, connectionstyle='arc3,rad=0.15'),
                    zorder=7)
    ax.axhline(0, color='#dddddd', lw=1.2, ls='--', zorder=1)
    ax.axvline(0, color='#dddddd', lw=1.2, ls='--', zorder=1)
    _offsets = [(3, 1), (-11, -3), (-12, 2.5), (2, -3), (3, 1), (2, -3)]
    for cx, cy, col, lbl, (dx, dy) in zip(
            centroids_x, centroids_y, centroid_colors, centroid_labels, _offsets):
        ax.annotate(lbl, (cx, cy), xytext=(cx + dx, cy + dy),
                    fontsize=11, fontweight='bold', color=col,
                    arrowprops=dict(arrowstyle='-', color=col, lw=1.1, alpha=0.7),
                    zorder=8)
    ax.set_xlabel('Axis 1: m6A Net Deposition  (Writers − Erasers, Δ percentile)', fontweight='bold')
    ax.set_ylabel('Axis 2: Oncogenic Readout  (Onco. − Supp. Readers, Δ percentile)', fontweight='bold')
    ax.set_title(
        f'm₆A Epitranscriptomic Landscape of Prostate Cancer Progression\n'
        f'n=1,421 samples · 6 cohorts · within-sample percentile rank over {N_common:,} genes\n'
        '★ = group median',
        pad=14,
    )
    ax.legend(loc='upper left', framealpha=0.85, markerscale=1.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P1_2d_landscape.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P1_2d_landscape.png')
    plt.close()

# P2 — Three-axis progression
print("\n--- FIG P2: Three-axis progression ---")
_axis_defs = [
    ('m6A_Net_Deposition',
     'Axis 1: Net Deposition\n(Writers − Erasers)',
     'm₆A writing machinery is up in\nhormone-naive mCSPC, restores in mCRPC'),
    ('m6A_Oncogenic_Readout',
     'Axis 2: Oncogenic Readout\n(Onco. − Supp. Readers)',
     'mCSPC has minimal oncogenic readout;\nmCRPC partially re-activates'),
    ('m6A_Functional_Impact',
     'Axis 3: Functional Impact\n(composite)',
     'Progressive recovery from\nlocalized → castration-resistant disease'),
]
with plt.rc_context(_PS):
    fig, axes_p2 = plt.subplots(1, 3, figsize=(22, 9), sharey=False)
    np.random.seed(0)
    for ax_p, (axis, panel_title, subtitle) in zip(axes_p2, _axis_defs):
        vals = get_group_vals(axis)
        ns   = [len(v) for v in vals]
        for pos, (v, c) in enumerate(zip(vals, GROUP_COLORS)):
            jit = np.random.uniform(-0.18, 0.18, size=len(v))
            ax_p.scatter(pos + jit, v, color=c, alpha=0.18, s=9, linewidths=0, zorder=2)
            ax_p.boxplot(v, positions=[pos], widths=0.35, patch_artist=True,
                         showfliers=False,
                         medianprops=dict(color='black', lw=2.5),
                         boxprops=dict(facecolor=c, alpha=0.55),
                         whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))
        kw_h, kw_p_val = kruskal(*vals)
        key_pairs = [(0, 2), (2, 4), (4, 5)]
        y_top = max(v.max() for v in vals)
        k = 0
        for pi, pj in key_pairs:
            _, p_v = mannwhitneyu(vals[pi], vals[pj], alternative='two-sided')
            if p_v < 0.05:
                bar_y = y_top + 3 + k * 5
                ax_p.plot([pi, pi, pj, pj], [bar_y, bar_y + 0.7, bar_y + 0.7, bar_y],
                          lw=1.5, color='black')
                ax_p.text((pi + pj) / 2, bar_y + 0.9, sig(p_v),
                          ha='center', fontsize=13, fontweight='bold')
                k += 1
        ax_p.set_ylim(top=y_top + 4 + k * 5 + 4)
        ax_p.set_xticks(range(6))
        ax_p.set_xticklabels(_CL, fontsize=13, fontweight='bold', rotation=25, ha='right')
        ax_p.axhline(0, color='#bbbbbb', lw=1.0, ls='--')
        ax_p.set_ylabel('Δ percentile points', fontweight='bold')
        ax_p.set_title(f'{panel_title}\nK-W H={kw_h:.0f}, {fmt_p(kw_p_val)} {sig(kw_p_val)}',
                       fontsize=15, fontweight='bold', pad=8)
        ax_p.text(0.5, -0.22, subtitle, transform=ax_p.transAxes,
                  ha='center', fontsize=11, fontstyle='italic', color='#444')
    fig.suptitle(
        'm₆A Axis Scores Across Prostate Cancer Stages\n'
        '(within-sample percentile rank; boxes = IQR; whiskers = 1.5×IQR; dots = individual samples)',
        fontsize=17, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P2_three_axis_progression.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P2_three_axis_progression.png')
    plt.close()

# P3 — YTHDF1 dominance

def _grp_pct(gene):
    return [
        gtex_pct[gene].values,
        adj_normal_pct[gene].values,
        pct_tcga[gene].values,
        mhspc_pct[gene].values,
        pct_mcrpc.loc[idx_adeno.intersection(pct_mcrpc.index), gene].values,
        pct_mcrpc.loc[idx_scnc.intersection(pct_mcrpc.index),  gene].values,
    ]

print("\n--- FIG P3: YTHDF1 dominance ---")
_ythdf1_vals      = _grp_pct('YTHDF1')
_onco_vals        = get_group_vals('m6A_Oncogenic_Readout')
_ythdf1_only_vals = [
    meta_gtex['m6A_YTHDF1_Readout'].loc[GRP_NORMAL].dropna().values,
    meta_adj_normal['m6A_YTHDF1_Readout'].dropna().values,
    meta_tcga['m6A_YTHDF1_Readout'].dropna().values,
    meta_mhspc['m6A_YTHDF1_Readout'].dropna().values,
    meta_mcrpc.loc[idx_adeno.intersection(meta_mcrpc.index), 'm6A_YTHDF1_Readout'].dropna().values,
    meta_mcrpc.loc[idx_scnc.intersection(meta_mcrpc.index),  'm6A_YTHDF1_Readout'].dropna().values,
]
kw_ythdf1_pct,  kw_p_ythdf1_pct  = kruskal(*_ythdf1_vals)
kw_ythdf1_only, kw_p_ythdf1_only = kruskal(*_ythdf1_only_vals)
kw_onco,        kw_p_onco        = kruskal(*_onco_vals)

with plt.rc_context(_PS):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 9))
    np.random.seed(0)
    for pos, (v, c) in enumerate(zip(_ythdf1_vals, GROUP_COLORS)):
        jit = np.random.uniform(-0.18, 0.18, size=len(v))
        ax_left.scatter(pos + jit, v, color=c, alpha=0.22, s=9, linewidths=0, zorder=2)
        ax_left.boxplot(v, positions=[pos], widths=0.38, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color='black', lw=2.5),
                        boxprops=dict(facecolor=c, alpha=0.60),
                        whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))
    ax_left.axhline(50, color='#bbbbbb', lw=1.0, ls=':', alpha=0.8,
                    label='50th pct = median gene')
    ax_left.set_xticks(range(6))
    ax_left.set_xticklabels(_CL, fontsize=13, fontweight='bold', rotation=25, ha='right')
    ax_left.set_ylabel('YTHDF1 within-sample percentile rank', fontweight='bold')
    ax_left.set_title(
        f'YTHDF1 Percentile Rank\nK-W H={kw_ythdf1_pct:.0f}, p={kw_p_ythdf1_pct:.1e} {sig(kw_p_ythdf1_pct)}',
        fontsize=15, fontweight='bold', pad=10,
    )
    ax_left.legend(fontsize=11)
    _comps = ['Full oncogenic readout\n(6 readers)', 'YTHDF1-only readout\n(single gene)']
    _hs    = [kw_onco, kw_ythdf1_only]
    _pvals = [kw_p_onco, kw_p_ythdf1_only]
    _bcols = ['#95a5a6', GROUP_COLORS[3]]
    ax_right.barh([1, 0], _hs, color=_bcols, alpha=0.85, height=0.4, edgecolor='black')
    for i, (h_val, p_val) in enumerate(zip(_hs, _pvals)):
        ax_right.text(h_val + 8, [1, 0][i],
                      f'H = {h_val:.0f}\np = {p_val:.1e}  {sig(p_val)}',
                      va='center', fontsize=13, fontweight='bold')
    ax_right.set_yticks([0, 1])
    ax_right.set_yticklabels(_comps[::-1], fontsize=14, fontweight='bold')
    ax_right.set_xlabel('Kruskal-Wallis H statistic\n(larger = stronger group separation)', fontweight='bold')
    ax_right.set_xlim(0, max(_hs) * 1.45)
    ax_right.axvline(kw_onco, color='#95a5a6', lw=1.0, ls='--', alpha=0.5)
    ax_right.set_title('YTHDF1 alone outperforms\nthe full 6-reader composite',
                       fontsize=15, fontweight='bold', pad=10, color=GROUP_COLORS[4])
    fig.suptitle(
        'YTHDF1: The Dominant Oncogenic m₆A Reader in Prostate Cancer Progression\n'
        '(Oncogenic Readout = Onco. Readers − Suppressive Readers; YTHDF1-only = YTHDF1 − mean(Supp. Readers))',
        fontsize=17, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P3_ythdf1_dominance.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P3_ythdf1_dominance.png')
    plt.close()

# P4 — IGF2BP1 re-expression
print("\n--- FIG P4: IGF2BP1 re-expression ---")
_i1_vals = [
    gtex_common['IGF2BP1'].values,
    adj_normal_common['IGF2BP1'].values,
    tcga_common['IGF2BP1'].values,
    mhspc_common['IGF2BP1'].values,
    mcrpc_common.loc[idx_adeno.intersection(mcrpc_common.index), 'IGF2BP1'].values,
    mcrpc_common.loc[idx_scnc.intersection(mcrpc_common.index),  'IGF2BP1'].values,
]
_i1_ns  = [len(v) for v in _i1_vals]
_i1_det = [(v > 1.0).mean() * 100 for v in _i1_vals]
kw_i1_h, kw_i1_p = kruskal(*_i1_vals)

with plt.rc_context(_PS):
    fig, ax = plt.subplots(figsize=(16, 9))
    np.random.seed(1)
    for pos, (v, c) in enumerate(zip(_i1_vals, GROUP_COLORS)):
        jit = np.random.uniform(-0.20, 0.20, size=len(v))
        ax.scatter(pos + jit, v, color=c, alpha=0.20, s=12, linewidths=0, zorder=2)
        ax.boxplot(v, positions=[pos], widths=0.38, patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color='black', lw=2.5),
                   boxprops=dict(facecolor=c, alpha=0.60),
                   whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))
        ax.text(pos, -0.55, f'{_i1_det[pos]:.0f}% detected',
                ha='center', fontsize=11.5, fontstyle='italic',
                color=c if _i1_det[pos] > 5 else '#aaaaaa', fontweight='bold')
    ax.axhline(1.0, color='#888888', lw=1.2, ls=':', alpha=0.7,
               label='Detection threshold: log₂(CPM) = 1')
    ax.axhspan(-0.6, 1.0, alpha=0.04, color='#e74c3c', zorder=0)
    ax.text(5.6, 0.3, 'below\ndetection', ha='right', fontsize=10,
            color='#e74c3c', fontstyle='italic', alpha=0.8)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(_CL, _i1_ns)],
                       fontsize=13, fontweight='bold')
    ax.set_ylabel('IGF2BP1  log₂(CPM+1)\n[log₂(TPM+1) for GTEx]', fontweight='bold')
    ax.set_title(
        f'IGF2BP1: Oncofetal m₆A Reader Re-Expressed in Castration-Resistant Disease\n'
        f'K-W  H={kw_i1_h:.0f},  p={kw_i1_p:.1e}  {sig(kw_i1_p)}\n'
        '% of samples above detection threshold (log₂ = 1) shown per group',
        fontsize=16, fontweight='bold', pad=12,
    )
    ax.legend(fontsize=12, loc='upper left')
    ax.annotate(
        'Silent in normal\n& localized PCa\n(< 1% detected)',
        xy=(1, 0.15), xytext=(0.5, 3.8),
        fontsize=12, fontweight='bold', color='#c0392b',
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdecea', edgecolor='#c0392b', alpha=0.9),
    )
    ax.annotate(
        'Re-expressed in\ncastration resistance',
        xy=(4.4, np.median(_i1_vals[4])), xytext=(3.1, 4.5),
        fontsize=12, fontweight='bold', color='#e67e22',
        arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef5e7', edgecolor='#e67e22', alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P4_igf2bp1_reexpression.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P4_igf2bp1_reexpression.png')
    plt.close()

# P5–P8: Gene heatmap, YTHDF faceoff, effect-size matrix, writer trajectory
_GENE_CLASS_MAP = (
    [('Writer',      g) for g in WRITER_GENES] +
    [('Eraser',      g) for g in ERASER_GENES] +
    [('Onco Reader', g) for g in READER_ONCOGENIC] +
    [('Supp Reader', g) for g in READER_SUPPRESSIVE]
)
_CLASS_PALETTE = {
    'Writer':      '#2980b9',
    'Eraser':      '#27ae60',
    'Onco Reader': '#e74c3c',
    'Supp Reader': '#8e44ad',
}
_gene_order = [g for _, g in _GENE_CLASS_MAP]
_n_genes    = len(_gene_order)
_pct_med    = np.array([[np.median(v) for v in _grp_pct(g)] for g in _gene_order])
_cls_bounds = []
_prev_c = None
for _gi, (cls, _) in enumerate(_GENE_CLASS_MAP):
    if cls != _prev_c and _gi > 0:
        _cls_bounds.append(_gi - 0.5)
    _prev_c = cls

# P5 — Gene heatmap
print("\n--- FIG P5: Gene-level heatmap ---")
with plt.rc_context(_PS):
    fig, ax = plt.subplots(figsize=(16, 12))
    _vext = max(float(np.abs(_pct_med - 50).max()), 8)
    _im5  = ax.imshow(_pct_med, aspect='auto', cmap='RdBu_r',
                      vmin=50 - _vext, vmax=50 + _vext, interpolation='nearest')
    for _gi in range(_n_genes):
        for _ci in range(6):
            val = _pct_med[_gi, _ci]
            tcol = 'white' if abs(val - 50) > 15 else '#222'
            ax.text(_ci, _gi, f'{val:.0f}', ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=tcol)
    for _x in np.arange(0.5, 6, 1):
        ax.axvline(_x, color='white', lw=2.0, zorder=3)
    for _y in np.arange(0.5, _n_genes, 1):
        ax.axhline(_y, color='white', lw=0.8, zorder=3)
    for _yb in _cls_bounds:
        ax.axhline(_yb, color='#111', lw=3.5, zorder=4)
    ax.set_xticks(range(6))
    ax.set_xticklabels(_CL, fontsize=14, fontweight='bold')
    ax.set_yticks(range(_n_genes))
    ax.set_yticklabels(_gene_order, fontsize=12)
    for _lbl, (cls, _) in zip(ax.get_yticklabels(), _GENE_CLASS_MAP):
        _lbl.set_color(_CLASS_PALETTE[cls]); _lbl.set_fontweight('bold')
    _cb5 = fig.colorbar(_im5, ax=ax, fraction=0.022, pad=0.02, shrink=0.85)
    _cb5.set_label('Median within-sample percentile rank', fontsize=12)
    _cb5.ax.axhline(50, color='black', lw=1.5, ls='--', alpha=0.55)
    _leg5 = [Patch(color=c, label=cls) for cls, c in _CLASS_PALETTE.items()]
    ax.legend(handles=_leg5, loc='upper right', bbox_to_anchor=(1.0, -0.04),
              ncol=4, fontsize=12, framealpha=0.9, title='Gene Class', title_fontsize=12)
    ax.set_title(
        f'm₆A Gene Landscape across Prostate Cancer Progression\n'
        f'Median within-sample percentile rank vs {N_common:,} genes · 22 genes · n=1,421',
        fontsize=17, fontweight='bold', pad=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P5_gene_heatmap.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P5_gene_heatmap.png')
    plt.close()

# P6 — YTHDF faceoff
print("\n--- FIG P6: YTHDF1/2/3 diverging fates ---")
_ythdf_info = [
    ('YTHDF1', GROUP_COLORS[4],
     'Promotes translation of m₆A targets\n→ oncogenic; rises in mCRPC'),
    ('YTHDF2', '#5d6d7e',
     'Promotes mRNA degradation\n→ suppressive; stays low'),
    ('YTHDF3', '#7f8c8d',
     'Promotes mRNA degradation\n→ suppressive; stays low'),
]
with plt.rc_context(_PS):
    fig, axes_p6 = plt.subplots(1, 3, figsize=(22, 9), sharey=True)
    np.random.seed(2)
    for ax_p, (gene, gcol, role) in zip(axes_p6, _ythdf_info):
        grp_arrs = _grp_pct(gene)
        kw_h_g, kw_p_g = kruskal(*grp_arrs)
        for pos, (v, c) in enumerate(zip(grp_arrs, GROUP_COLORS)):
            jit = np.random.uniform(-0.18, 0.18, size=len(v))
            ax_p.scatter(pos + jit, v, color=c, alpha=0.22, s=10, linewidths=0, zorder=2)
            ax_p.boxplot(v, positions=[pos], widths=0.38, patch_artist=True,
                         showfliers=False,
                         medianprops=dict(color='black', lw=2.5),
                         boxprops=dict(facecolor=c, alpha=0.60),
                         whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))
        _meds_g = [np.median(v) for v in grp_arrs]
        ax_p.plot(range(6), _meds_g, color=gcol, lw=2.5, ls='--',
                  marker='o', ms=7, zorder=5, alpha=0.9, label='Median trend')
        ax_p.axhline(50, color='#cccccc', lw=1.0, ls=':', alpha=0.7)
        ax_p.set_xticks(range(6))
        ax_p.set_xticklabels(_CL, fontsize=12, fontweight='bold', rotation=30, ha='right')
        ax_p.set_ylabel('Percentile rank (within-sample)', fontweight='bold')
        ax_p.set_title(f'{gene}\nH={kw_h_g:.0f}, p={kw_p_g:.1e}  {sig(kw_p_g)}',
                       fontsize=15, fontweight='bold', color=gcol, pad=8)
        ax_p.text(0.5, -0.28, role, transform=ax_p.transAxes,
                  ha='center', fontsize=11, fontstyle='italic', color='#444')
        ax_p.legend(fontsize=11)
    fig.suptitle(
        'Diverging YTHDF Fates: m₆A-Driven Translation vs Degradation in Prostate Cancer\n'
        'YTHDF1 (oncogenic: translation) rises in mCRPC; YTHDF2/3 (suppressive: degradation) remain suppressed',
        fontsize=17, fontweight='bold', y=1.03,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P6_ythdf_faceoff.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P6_ythdf_faceoff.png')
    plt.close()

# P7 — Effect-size matrix
print("\n--- FIG P7: Effect-size matrix ---")
_transitions = [
    ('Normal\nvs Primary', 0, 2),
    ('Primary\nvs mCSPC',  2, 3),
    ('Primary\nvs mCRPC',  2, 4),
    ('Adeno\nvs SCNC',     4, 5),
]
_es_r = np.zeros((_n_genes, len(_transitions)))
_es_p = np.zeros((_n_genes, len(_transitions)))
for _gi, _gene in enumerate(_gene_order):
    _all_grp = _grp_pct(_gene)
    for _ci, (_, _ai, _bi) in enumerate(_transitions):
        _es_r[_gi, _ci] = rankbiserial(_all_grp[_ai], _all_grp[_bi])
        _, _es_p[_gi, _ci] = mannwhitneyu(
            _all_grp[_ai], _all_grp[_bi], alternative='two-sided')

with plt.rc_context(_PS):
    fig, ax = plt.subplots(figsize=(13, 12))
    _im7 = ax.imshow(_es_r, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    for _gi in range(_n_genes):
        for _ci in range(len(_transitions)):
            r_val = _es_r[_gi, _ci]
            p_val = _es_p[_gi, _ci]
            tcol  = 'white' if abs(r_val) > 0.55 else '#222'
            star  = sig(p_val) if p_val < 0.05 else ''
            ax.text(_ci, _gi, f'{r_val:+.2f}{star}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=tcol)
    for _x in np.arange(0.5, len(_transitions), 1):
        ax.axvline(_x, color='white', lw=2.5, zorder=3)
    for _y in np.arange(0.5, _n_genes, 1):
        ax.axhline(_y, color='white', lw=0.8, zorder=3)
    for _yb in _cls_bounds:
        ax.axhline(_yb, color='#111', lw=3.5, zorder=4)
    ax.set_xticks(range(len(_transitions)))
    ax.set_xticklabels([t[0] for t in _transitions], fontsize=14, fontweight='bold')
    ax.set_yticks(range(_n_genes))
    ax.set_yticklabels(_gene_order, fontsize=12)
    for _lbl, (cls, _) in zip(ax.get_yticklabels(), _GENE_CLASS_MAP):
        _lbl.set_color(_CLASS_PALETTE[cls]); _lbl.set_fontweight('bold')
    _cb7 = fig.colorbar(_im7, ax=ax, fraction=0.022, pad=0.02, shrink=0.85)
    _cb7.set_label('Rank-biserial r\n(+1 = first group always higher)', fontsize=12)
    _cb7.ax.axhline(0, color='black', lw=1.5, ls='--', alpha=0.55)
    _leg7 = [Patch(color=c, label=cls) for cls, c in _CLASS_PALETTE.items()]
    ax.legend(handles=_leg7, loc='upper right', bbox_to_anchor=(1.0, -0.04),
              ncol=4, fontsize=11, framealpha=0.9, title='Gene Class', title_fontsize=11)
    ax.set_title('Effect-Size Matrix: Which Genes Change Most at Each Transition?\n'
                 'Rank-biserial r (Mann-Whitney U)  ·  *** p<0.001  ** p<0.01  * p<0.05',
                 fontsize=16, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P7_effect_size_matrix.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P7_effect_size_matrix.png')
    plt.close()

# P8 — Writer trajectory
print("\n--- FIG P8: Writer trajectory ---")
_wrt_pal = plt.cm.tab10(np.linspace(0, 0.88, len(WRITER_GENES)))
with plt.rc_context(_PS):
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.axvspan(2.6, 3.4, alpha=0.09, color='#9b59b6', zorder=0, label='mCSPC column')
    ax.axhline(50, color='#cccccc', lw=1.0, ls=':', zorder=1, label='50 = median gene')
    for _wi, _gene in enumerate(WRITER_GENES):
        _wgrps = _grp_pct(_gene)
        _meds  = [np.median(v)           for v in _wgrps]
        _q1s   = [np.percentile(v, 25)   for v in _wgrps]
        _q3s   = [np.percentile(v, 75)   for v in _wgrps]
        _col   = _wrt_pal[_wi]
        ax.fill_between(range(6), _q1s, _q3s, color=_col, alpha=0.12, zorder=2)
        ax.plot(range(6), _meds, color=_col, lw=2.5, marker='o', ms=9,
                zorder=3, label=_gene, markeredgecolor='white', markeredgewidth=1.2)
        ax.text(5.12, _meds[5], _gene, va='center', fontsize=11.5, color=_col, fontweight='bold')
        ax.text(3, _meds[3] + 1.0, f'{_meds[3]:.0f}', ha='center', va='bottom',
                fontsize=9, color=_col, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_xticklabels(_CL, fontsize=14, fontweight='bold')
    ax.set_ylabel('Median within-sample percentile rank  (IQR shaded)', fontweight='bold')
    ax.set_xlim(-0.4, 6.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.09), ncol=5,
              fontsize=12, framealpha=0.9, title='Writer gene', title_fontsize=12)
    ax.set_title(
        'Which m₆A Writers Drive the mCSPC Elevation?\n'
        'Line = median, shading = IQR · Values shown in mCSPC column · mCSPC purple band',
        fontsize=17, fontweight='bold', pad=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PRESDIR, 'P8_writer_trajectory.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    print('  -> Saved: P8_writer_trajectory.png')
    plt.close()

print(f"\n  Presentation figures saved to: {PRESDIR}")

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
_conv_p = _rgba_to_rgb(PRESDIR)
if _conv_p:
    print(f"  Converted {_conv_p} RGBA → RGB PNG(s) in plots_presentation/")

print("\n  Done.")
