#!/usr/bin/env python3
"""
tcga_immune_m6a.py — m6A × macrophage/immune correlation in TCGA-PRAD primary.

Correlates the expression of 22 m6A regulatory genes with CIBERSORT immune
cell fractions (Thorsson et al. 2018, Immunity — pan-cancer Immune Landscape)
in TCGA-PRAD primary tumour samples (n ≈ 499, after matched QC).

Analyses:
  01  Correlation heatmap: 22 m6A genes × key immune populations
      (M0 / M1 / M2 macrophages, CD8 T, Tregs, total NK, Monocytes)
  02  m6A axis scores vs macrophage fractions (M1, M2, M1:M2 ratio)
  03  Top gene–immune scatter plots (up to 6 FDR-significant pairs)
  04  M1 & M2 macrophage fractions stratified by Gleason group

Data source:
  CIBERSORT fractions — TCGA.Kallisto.fullIDs.cibersort.relative.tsv
  (GDC UUID b3df502e-3594-46ef-9f94-d041a20a0b9a, open access)

Output: plots_tcga_immune/

Usage:
    micromamba run -n rnaseq python tcga_immune_m6a.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
plt.rcParams['savefig.dpi'] = 300

# ── m6a package imports ───────────────────────────────────────────────────────
from m6a.config import TCGA_DIR
from m6a.genes import (
    ALL_M6A_GENES,
    WRITER_GENES, ERASER_GENES,
    READER_ONCOGENIC, READER_SUPPRESSIVE,
)
from m6a.stats import bh_fdr, sig, fmt_p, rankbiserial
from m6a.normalization import percentile_rank_matrix, zscore_normalize
from m6a.scoring import compute_axes
from m6a.data.loaders import load_tcga

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_tcga_immune')
os.makedirs(OUTDIR, exist_ok=True)

CIBERSORT_FILE = os.path.join(TCGA_DIR, 'TCGA.Kallisto.fullIDs.cibersort.relative.tsv')

# Immune cell columns to include in the correlation analysis
IMMUNE_COLS = [
    'Macrophages.M0',
    'Macrophages.M1',
    'Macrophages.M2',
    'T.cells.CD8',
    'T.cells.regulatory..Tregs.',
    'NK_total',          # sum of resting + activated (computed below)
    'Monocytes',
]
IMMUNE_LABELS = [
    'Mφ M0',
    'Mφ M1',
    'Mφ M2',
    'CD8 T',
    'Tregs',
    'NK (total)',
    'Monocytes',
]

# Role→gene mapping (for heatmap row ordering/colours)
ROLE_ORDER = WRITER_GENES + ERASER_GENES + READER_ONCOGENIC + READER_SUPPRESSIVE
ROLE_COLORS = (
    ['#2980b9'] * len(WRITER_GENES) +
    ['#e74c3c'] * len(ERASER_GENES) +
    ['#f39c12'] * len(READER_ONCOGENIC) +
    ['#27ae60'] * len(READER_SUPPRESSIVE)
)
ROLE_PATCH_LABELS = ['Writer', 'Eraser', 'Reader (onco.)', 'Reader (supp.)']
ROLE_PATCH_COLORS = ['#2980b9', '#e74c3c', '#f39c12', '#27ae60']

print("=" * 80)
print("  m6A × MACROPHAGE IMMUNE CORRELATION — TCGA-PRAD PRIMARY")
print("=" * 80)

# =============================================================================
# DATA LOADING & MERGING
# =============================================================================
print("\n[1/4] Loading data ...")
expr, clinical = load_tcga()   # samples × genes; index = TCGA-XX-XXXX

# ── Load CIBERSORT ────────────────────────────────────────────────────────────
ciber_raw = pd.read_csv(CIBERSORT_FILE, sep='\t')

# Keep PRAD only
ciber_raw = ciber_raw[ciber_raw['CancerType'] == 'PRAD'].copy()

# Convert dot-separated full barcodes → 12-char TCGA case IDs
# e.g. TCGA.J4.8198.01A.11R.2263.07 → TCGA-J4-8198
ciber_raw['case_id'] = (
    ciber_raw['SampleID']
    .str.replace('.', '-', regex=False)
    .str.slice(0, 12)
)

# Deduplicate: one entry per case (keep first — IDs are already per-aliquot)
ciber_raw = ciber_raw.drop_duplicates(subset='case_id', keep='first')
ciber_raw = ciber_raw.set_index('case_id')

print(f"    CIBERSORT PRAD samples total: {len(ciber_raw)}")

# Compute derived immune metrics
ciber_raw['NK_total'] = (
    ciber_raw['NK.cells.resting'].fillna(0) +
    ciber_raw['NK.cells.activated'].fillna(0)
)
# M1:M2 ratio (add small constant to avoid division by zero)
ciber_raw['M1_M2_ratio'] = ciber_raw['Macrophages.M1'] / (
    ciber_raw['Macrophages.M2'] + 1e-6
)
# log10(M1:M2) for normal-like distributions in scatter plots
ciber_raw['log_M1_M2'] = np.log10(ciber_raw['M1_M2_ratio'] + 0.01)

# ── Overlap ───────────────────────────────────────────────────────────────────
shared = expr.index.intersection(ciber_raw.index)
print(f"    Expression × CIBERSORT overlap: {len(shared)} samples")

expr_m   = expr.loc[shared]
ciber_m  = ciber_raw.loc[shared]
clin_m   = clinical.reindex(shared)

# ── m6A gene availability ─────────────────────────────────────────────────────
m6a_in = [g for g in ROLE_ORDER if g in expr_m.columns]
role_colors_in = [
    ROLE_COLORS[ROLE_ORDER.index(g)] for g in m6a_in
]
print(f"    m6A genes available: {len(m6a_in)}/{len(ALL_M6A_GENES)}: {m6a_in}")

# ── m6A axis scores (percentile-rank method, cross-cohort protocol) ───────────
pct = percentile_rank_matrix(expr_m)          # within-sample, all genes
pct_m6a = pct[m6a_in]
nd, onco, impact = compute_axes(pct_m6a)

# Attach to working metadata frame
meta = clin_m.copy()
meta['Net_Deposition']    = nd.reindex(meta.index)
meta['Oncogenic_Readout'] = onco.reindex(meta.index)
meta['Functional_Impact'] = impact.reindex(meta.index)
meta = meta.join(ciber_m[IMMUNE_COLS + ['M1_M2_ratio', 'log_M1_M2']])

# Gleason groups (for stratification)
meta['gleason_group'] = pd.cut(
    meta['gleason_score'],
    bins=[0, 7, 10],
    labels=['≤7 (low–int)', '≥8 (high)'],
)
print(f"    Gleason ≤7: {(meta['gleason_group'] == '≤7 (low–int)').sum()}  "
      f"≥8: {(meta['gleason_group'] == '≥8 (high)').sum()}")

print("\n[2/4] Computing Spearman correlations (22 genes × 7 immune populations) ...")

# =============================================================================
# CORRELATION MATRIX
# =============================================================================
def spearman_matrix(genes, immune_cols, expr_df, ciber_df):
    """Return (ρ, q) DataFrames: genes × immune populations."""
    rho_d, q_d = {}, {}
    for ic in immune_cols:
        y = ciber_df[ic].fillna(0)
        rhos, pvals = [], []
        for g in genes:
            x = expr_df[g]
            msk = x.notna() & y.notna()
            if msk.sum() < 20:
                rhos.append(np.nan); pvals.append(np.nan)
                continue
            r, p = spearmanr(x[msk], y[msk])
            rhos.append(r); pvals.append(p)
        # BH-FDR correction per immune cell type
        valid = ~np.isnan(pvals)
        fdr = np.full(len(pvals), np.nan)
        if valid.sum() > 0:
            _, fdr_vals, _, _ = multipletests(
                np.array(pvals)[valid], method='fdr_bh'
            )
            fdr[valid] = fdr_vals
        rho_d[ic] = rhos
        q_d[ic]   = list(fdr)
    rho_df = pd.DataFrame(rho_d, index=genes)
    q_df   = pd.DataFrame(q_d,   index=genes)
    return rho_df, q_df


rho_df, q_df = spearman_matrix(m6a_in, IMMUNE_COLS, expr_m[m6a_in], ciber_m)

# ── Rename columns to readable labels ─────────────────────────────────────────
col_rename = dict(zip(IMMUNE_COLS, IMMUNE_LABELS))
rho_plot = rho_df.rename(columns=col_rename)
q_plot   = q_df.rename(columns=col_rename)

print("\n[3/4] Generating plots ...")

# =============================================================================
# PLOT 01 — Correlation heatmap
# =============================================================================
print("    Plot 01: Correlation heatmap ...")

fig, ax = plt.subplots(figsize=(9, 8))

# Build annotation matrix (stars for FDR thresholds)
annot = q_plot.applymap(lambda q: '***' if q < 0.001 else
                                  '**'  if q < 0.01  else
                                  '*'   if q < 0.05  else '')

sns.heatmap(
    rho_plot,
    annot=annot, fmt='s',
    cmap='RdBu_r', center=0, vmin=-0.35, vmax=0.35,
    linewidths=0.4, linecolor='#cccccc',
    square=True,
    annot_kws={'size': 8, 'color': 'black'},
    cbar_kws={'label': "Spearman ρ", 'shrink': 0.7},
    ax=ax,
)

# Colour row tick labels by gene role
for lbl, color in zip(ax.get_yticklabels(), role_colors_in):
    lbl.set_color(color)
    lbl.set_fontweight('bold')
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', labelsize=9, rotation=30)

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title(
    'Spearman ρ: m6A genes × CIBERSORT immune fractions\n'
    f'TCGA-PRAD primary (n={len(shared)}), FDR: * <0.05  ** <0.01  *** <0.001',
    fontsize=10, pad=10,
)

# Role legend
from matplotlib.patches import Patch
patches = [Patch(facecolor=c, label=l)
           for c, l in zip(ROLE_PATCH_COLORS, ROLE_PATCH_LABELS)]
ax.legend(handles=patches, bbox_to_anchor=(1.28, 1), loc='upper left',
          fontsize=8, framealpha=0.8, title='Gene role')

plt.tight_layout()
out01 = os.path.join(OUTDIR, '01_m6a_immune_correlation_heatmap.png')
fig.savefig(out01, bbox_inches='tight')
plt.close(fig)
print(f"      → Saved {out01}")

# =============================================================================
# PLOT 02 — m6A axis scores × macrophage fractions
# =============================================================================
print("    Plot 02: Axis scores × macrophage fractions ...")

axis_cols   = ['Net_Deposition', 'Oncogenic_Readout', 'Functional_Impact']
axis_labels = ['Net Deposition', 'Oncogenic Readout', 'Functional Impact']
macro_cols  = ['Macrophages.M1', 'Macrophages.M2', 'log_M1_M2']
macro_labels = ['Mφ M1 fraction', 'Mφ M2 fraction', 'log₁₀(M1:M2 ratio)']

fig, axes = plt.subplots(3, 3, figsize=(11, 10))
plt.subplots_adjust(hspace=0.45, wspace=0.35)

for ri, (mc, ml) in enumerate(zip(macro_cols, macro_labels)):
    for ci, (ac, al) in enumerate(zip(axis_cols, axis_labels)):
        ax = axes[ri, ci]
        x = meta[ac]
        y = meta[mc]
        msk = x.notna() & y.notna()
        xs, ys = x[msk].values, y[msk].values

        ax.scatter(xs, ys, s=8, alpha=0.4, color='#3498db', linewidths=0)

        # Regression line
        m, b = np.polyfit(xs, ys, 1)
        xl = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xl, m * xl + b, color='#e74c3c', linewidth=1.4)

        r, p = spearmanr(xs, ys)
        ax.text(
            0.97, 0.97,
            f"ρ={r:.2f}\n{fmt_p(p)}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
        )
        ax.set_xlabel(al, fontsize=8)
        ax.set_ylabel(ml if ci == 0 else '', fontsize=8)
        ax.tick_params(labelsize=7)
        if ri == 0:
            ax.set_title(al, fontsize=9, fontweight='bold')

fig.suptitle(
    f'm6A axis scores vs macrophage fractions — TCGA-PRAD primary (n={msk.sum()})',
    y=1.01, fontsize=11,
)
out02 = os.path.join(OUTDIR, '02_axis_vs_macrophage_fractions.png')
fig.savefig(out02, bbox_inches='tight')
plt.close(fig)
print(f"      → Saved {out02}")

# =============================================================================
# PLOT 03 — Top gene–immune scatter plots
# =============================================================================
print("    Plot 03: Top gene–immune scatter plots ...")

# Collect all (gene, immune, ρ, q) pairs; filter FDR < 0.05, rank by |ρ|
records = []
for ic, il in zip(IMMUNE_COLS, IMMUNE_LABELS):
    for g in m6a_in:
        r = rho_df.loc[g, ic]
        q = q_df.loc[g, ic]
        if not np.isnan(r) and not np.isnan(q):
            records.append({'gene': g, 'immune': ic, 'label': il, 'rho': r, 'q': q})

hits = (
    pd.DataFrame(records)
    .query('q < 0.05')
    .assign(absrho=lambda d: d['rho'].abs())
    .sort_values('absrho', ascending=False)
    .drop_duplicates(subset='gene')   # one entry per gene (strongest hit)
    .head(6)
)
print(f"      {len(hits)} FDR<0.05 unique-gene top pairs selected")

if len(hits) > 0:
    ncols = 3
    nrows = int(np.ceil(len(hits) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)
    plt.subplots_adjust(hspace=0.5, wspace=0.35)

    for idx, (_, row) in enumerate(hits.iterrows()):
        ax = axes[idx // ncols][idx % ncols]
        gene, ic, il = row['gene'], row['immune'], row['label']

        gene_role_color = role_colors_in[m6a_in.index(gene)]

        x = expr_m[gene]
        y = ciber_m[ic].fillna(0)
        msk = x.notna() & y.notna()
        xs, ys = x[msk].values, y[msk].values

        ax.scatter(xs, ys, s=10, alpha=0.45, color=gene_role_color, linewidths=0)
        m, b = np.polyfit(xs, ys, 1)
        xl = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xl, m * xl + b, color='black', linewidth=1.2, linestyle='--')

        ax.set_xlabel(f'{gene} (log₂CPM)', fontsize=9)
        ax.set_ylabel(il, fontsize=9)
        ax.tick_params(labelsize=8)
        q_str = f"{row['q']:.2e}" if row['q'] < 0.001 else f"{row['q']:.3f}"
        ax.set_title(
            f"{gene} × {il}\nρ={row['rho']:.3f}, q={q_str}",
            fontsize=9,
        )

    # Hide unused panels
    for idx in range(len(hits), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        'Top m6A gene × immune fraction pairs (FDR < 0.05, one per gene)',
        y=1.01, fontsize=11,
    )
    out03 = os.path.join(OUTDIR, '03_top_gene_immune_scatter.png')
    fig.savefig(out03, bbox_inches='tight')
    plt.close(fig)
    print(f"      → Saved {out03}")
else:
    print("      No FDR<0.05 pairs — skipping plot 03")

# =============================================================================
# PLOT 04 — Macrophage fractions by Gleason group
# =============================================================================
print("    Plot 04: Macrophage fractions by Gleason group ...")

gleason_groups = ['≤7 (low–int)', '≥8 (high)']
gleason_palette = {'≤7 (low–int)': '#3498db', '≥8 (high)': '#e74c3c'}

macro_box_cols   = ['Macrophages.M1', 'Macrophages.M2', 'log_M1_M2']
macro_box_labels = ['Mφ M1 fraction', 'Mφ M2 fraction', 'log₁₀(M1:M2)']

meta_g = meta.dropna(subset=['gleason_group'])

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
plt.subplots_adjust(wspace=0.35)

for ax, mc, ml in zip(axes, macro_box_cols, macro_box_labels):
    data_plot = meta_g[['gleason_group', mc]].dropna()

    sns.boxplot(
        data=data_plot, x='gleason_group', y=mc,
        palette=gleason_palette, width=0.5,
        order=gleason_groups, ax=ax,
        fliersize=2, linewidth=0.8,
    )
    sns.stripplot(
        data=data_plot, x='gleason_group', y=mc,
        palette=gleason_palette, order=gleason_groups,
        size=2.5, alpha=0.35, jitter=True, ax=ax,
    )

    # Mann-Whitney U test
    grp_lo = data_plot.loc[data_plot['gleason_group'] == '≤7 (low–int)', mc]
    grp_hi = data_plot.loc[data_plot['gleason_group'] == '≥8 (high)',    mc]
    if len(grp_lo) >= 5 and len(grp_hi) >= 5:
        _, p = mannwhitneyu(grp_lo, grp_hi, alternative='two-sided')
        ymax = data_plot[mc].quantile(0.95)
        ax.annotate(
            f'MWU {fmt_p(p)}',
            xy=(0.5, 0.97), xycoords='axes fraction',
            ha='center', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.8),
        )
    ax.set_xlabel('')
    ax.set_ylabel(ml, fontsize=9)
    ax.tick_params(labelsize=9)
    ns = [len(grp_lo), len(grp_hi)]
    ax.set_xticklabels([f'{g}\n(n={n})' for g, n in zip(gleason_groups, ns)],
                       fontsize=8.5)

fig.suptitle(
    'CIBERSORT macrophage fractions by Gleason group — TCGA-PRAD primary',
    fontsize=11,
)
out04 = os.path.join(OUTDIR, '04_macrophage_by_gleason.png')
fig.savefig(out04, bbox_inches='tight')
plt.close(fig)
print(f"      → Saved {out04}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n[4/4] Writing summary table ...")

summary = (
    pd.DataFrame(records)
    .assign(absrho=lambda d: d['rho'].abs())
    .sort_values(['label', 'absrho'], ascending=[True, False])
    [['gene', 'label', 'rho', 'q']]
    .rename(columns={'label': 'immune_cell', 'rho': 'spearman_rho', 'q': 'fdr_bh'})
)
summary_out = os.path.join(OUTDIR, 'correlation_summary.csv')
summary.to_csv(summary_out, index=False, float_format='%.4f')
print(f"    → {summary_out}")

# Print top 10 hits (FDR < 0.05)
top = summary[summary['fdr_bh'] < 0.05].head(10)
if len(top):
    print("\n    Top FDR<0.05 associations:")
    print(top.to_string(index=False))
else:
    print("    No FDR<0.05 associations found — all nominal p-value results.")
    top_nom = summary.sort_values('spearman_rho', key=abs, ascending=False).head(10)
    print("\n    Top 10 by |ρ| (all nominal):")
    print(top_nom.to_string(index=False))

print("\n" + "=" * 80)
print("  DONE — plots written to:", OUTDIR)
print("=" * 80)
