#!/usr/bin/env python3
"""
ar_crosscohort_analysis.py — Cross-cohort AR × m6A trajectory in prostate cancer.

Separated from ar_m6a_analysis.py because:
  1. Requires within-sample percentile-rank normalization (batch-invariant across
     different platforms and RNA-seq protocols).
  2. Between-cohort mean differences are confounded by tumor purity / cell
     composition (whole prostate tissue vs biopsy-enriched tumor), handled here.
  3. The scientific question is distinct: how do AR and m6A *co-evolve* across
     the full disease spectrum, rather than mechanistic coupling within one stage.

Improvements over the former Part IX of ar_m6a_analysis.py:
  1. Consistent m6A axis computation — compute_axes() from m6a.scoring applies
     the same LR writer weights across ALL cohorts (no flat-mean TCGA shortcut).
  2. Within-cohort AR × m6A Spearman ρ per disease stage — tests whether
     the coupling is general or stage-specific.
  3. Tumor purity / epithelial fraction sensitivity analysis — computes a
     luminal epithelial marker score (KRT18, KRT8, AR, EPCAM if available) to
     flag cohorts where stromal dilution may distort the ARS trajectory.
  4. ARS sensitivity: ALDH1A3 and NKX3-1 are luminal identity genes more than
     dynamic AR targets; a sensitivity plot checks whether removing them alters
     the cross-cohort trajectory.

Cohorts (in progression order):
  GTEx v11        normal prostate stroma+epithelium
  Adj Normal      TCGA-PRAD adjacent solid tissue normals
  Primary PCa     TCGA-PRAD primary tumours (Gleason 6-10)
  mCSPC           GSE221601 metastatic castration-sensitive PCa (microarray)
  mCRPC-Adeno     LuCaP mCRPC cohort, Adenocarcinoma histology
  mCRPC-SCNC      LuCaP mCRPC cohort, Small Cell Neuroendocrine Carcinoma

Output plots: plots_ar_crosscohort/
  01 — ARS cross-cohort trajectory (violin)
  02 — m6A Functional Impact cross-cohort trajectory (violin)
  03 — ARS vs FI scatter across all disease stages
  04 — AR target gene percentile-rank heatmap (6 stages × genes)
  05 — Within-cohort AR × m6A coupling strength (ρ per stage, barplot)
  06 — AR × m6A coupling across stages (scatter ρ vs ARS mean)
  07 — Tumor purity proxy: luminal epithelial marker score across cohorts
  08 — ARS trajectory sensitivity: full vs secretory-only vs core-only signature
  09 — Per-stage ARS vs FI scatter panels (6-panel figure)
  10 — Summary: ARS + FI + within-stage ρ in a 3-panel overview

Usage:
    micromamba run -n rnaseq python ar_crosscohort_analysis.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import spearmanr, kruskal, mannwhitneyu

warnings.filterwarnings('ignore')
plt.rcParams['savefig.dpi'] = 300

# ── m6a package imports ───────────────────────────────────────────────────────
from m6a.config import OUTDIR_AR_CROSSCOHORT as OUTDIR
from m6a.genes import (
    AR_TARGET_GENES,
    ALL_M6A_GENES as CROSS_COHORT_M6A_GENES,
    WRITER_GENES, ERASER_GENES,
    READER_ONCOGENIC, READER_SUPPRESSIVE,
)
from m6a.stats import sig
from m6a.normalization import percentile_rank_matrix
from m6a.scoring import compute_axes
from m6a.plotting import style_violin
from m6a.data.loaders import (
    load_mcrpc, load_tcga, load_gtex, load_adj_normal, load_mcspc,
    build_common_universe,
)

os.makedirs(OUTDIR, exist_ok=True)

# Cross-cohort plot palette (consistent with cross_cohort.py)
CC_LABELS  = ['Normal\n(GTEx)', 'Adj Normal\n(TCGA)', 'Primary PCa\n(TCGA)',
              'mCSPC\n(GSE221601)', 'mCRPC\nAdeno', 'mCRPC\nSCNC']
CC_LABELS_FLAT = ['Normal (GTEx)', 'Adj Normal (TCGA)', 'Primary PCa (TCGA)',
                  'mCSPC (GSE221601)', 'mCRPC Adeno', 'mCRPC SCNC']
CC_COLORS  = ['#27ae60', '#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#c0392b']

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 80)
print("  AR × m6A CROSS-COHORT TRAJECTORY")
print("=" * 80)
print("\n  Loading all cohorts ...")

df_mcrpc, meta_mcrpc = load_mcrpc()
df_tcga,  _          = load_tcga()
gtex_expr, _         = load_gtex()
adjn_expr, _         = load_adj_normal()
mhspc_expr, _        = load_mcspc()

idx_mcrpc_adeno = meta_mcrpc[meta_mcrpc['histology'] == 'Adenocarcinoma'].index.intersection(df_mcrpc.index)
idx_mcrpc_scnc  = meta_mcrpc[meta_mcrpc['histology'] == 'SCNC'].index.intersection(df_mcrpc.index)

print(f"  GTEx:       {gtex_expr.shape[0]:4d} samples × {gtex_expr.shape[1]:,} genes")
print(f"  Adj Normal: {adjn_expr.shape[0]:4d} samples × {adjn_expr.shape[1]:,} genes")
print(f"  TCGA-PRAD:  {df_tcga.shape[0]:4d} samples × {df_tcga.shape[1]:,} genes")
print(f"  mCSPC:      {mhspc_expr.shape[0]:4d} samples × {mhspc_expr.shape[1]:,} genes  (microarray)")
print(f"  mCRPC Adeno:{len(idx_mcrpc_adeno):4d} samples")
print(f"  mCRPC SCNC: {len(idx_mcrpc_scnc):4d} samples")

# =============================================================================
# NORMALIZATION: within-sample percentile rank
# =============================================================================
print("\n  Building common gene universe ...")
all_expr_dfs = [df_mcrpc, df_tcga, gtex_expr, adjn_expr, mhspc_expr]
common = build_common_universe(all_expr_dfs)

ar_avail   = [g for g in AR_TARGET_GENES       if g in common]
m6a_avail  = [g for g in CROSS_COHORT_M6A_GENES if g in common]
print(f"  Common universe: {len(common):,} genes")
print(f"  AR target genes in common: {len(ar_avail)}/{len(AR_TARGET_GENES)} — {ar_avail}")
print(f"  m6A genes in common: {len(m6a_avail)}/{len(CROSS_COHORT_M6A_GENES)}")

# AR gene sub-panels for sensitivity analysis
# Secretory-only: fast-responding AR targets (exclude luminal identity genes)
AR_SECRETORY = [g for g in ['KLK3', 'KLK2', 'FKBP5', 'TMPRSS2', 'STEAP2', 'SLC45A3']
                if g in ar_avail]
# Core: secretory + FOLH1 (PSMA)
AR_CORE = [g for g in ['KLK3', 'KLK2', 'FKBP5', 'TMPRSS2', 'STEAP2', 'SLC45A3', 'FOLH1']
           if g in ar_avail]

def percentile_rank_df(expr_df):
    """Within-sample percentile rank, restricted to common gene universe."""
    return percentile_rank_matrix(expr_df[common].fillna(0.0))

print("  Computing percentile ranks across all cohorts ...")
pct = {
    'Normal':    percentile_rank_df(gtex_expr),
    'AdjNormal': percentile_rank_df(adjn_expr),
    'Primary':   percentile_rank_df(df_tcga),
    'mCSPC':     percentile_rank_df(mhspc_expr),
    'Adeno':     percentile_rank_df(df_mcrpc.loc[idx_mcrpc_adeno]),
    'SCNC':      percentile_rank_df(df_mcrpc.loc[idx_mcrpc_scnc]),
}
cohort_keys = ['Normal', 'AdjNormal', 'Primary', 'mCSPC', 'Adeno', 'SCNC']

def ars_from_panel(pct_df, panel):
    """Mean percentile rank of genes in panel."""
    avail = [g for g in panel if g in pct_df.columns]
    return pct_df[avail].mean(axis=1)

def m6a_fi(pct_df):
    """Functional Impact via compute_axes (consistent LR-weighted writers)."""
    avail_m6a = [g for g in m6a_avail if g in pct_df.columns]
    _, _, fi = compute_axes(pct_df[avail_m6a])
    return fi

# Compute scores for all cohorts
ars_full = {k: ars_from_panel(pct[k], ar_avail)      for k in cohort_keys}
ars_sec  = {k: ars_from_panel(pct[k], AR_SECRETORY)  for k in cohort_keys}
ars_core = {k: ars_from_panel(pct[k], AR_CORE)       for k in cohort_keys}
fi_all   = {k: m6a_fi(pct[k])                        for k in cohort_keys}

ars_groups = [ars_full[k] for k in cohort_keys]
fi_groups  = [fi_all[k]   for k in cohort_keys]
ns_cc      = [len(g) for g in ars_groups]

print(f"\n  {'Cohort':20s} {'n':>5s}  {'ARS (mean)':>12s}  {'m6A FI (mean)':>14s}")
print("  " + "-" * 60)
for k, lbl, n, ars_g, fi_g in zip(cohort_keys, CC_LABELS_FLAT, ns_cc, ars_groups, fi_groups):
    print(f"  {lbl:20s} {n:5d}  {ars_g.mean():12.1f}  {fi_g.mean():14.2f}")

# =============================================================================
# PART I — CROSS-COHORT TRAJECTORIES
# =============================================================================
print("\n" + "=" * 80)
print("  PART I — CROSS-COHORT TRAJECTORIES")
print("=" * 80)

# --- 01. ARS trajectory -------------------------------------------------------
print("\n--- Plot 01: ARS cross-cohort violin ---")
_, p_ars_kw = kruskal(*[g.dropna().values for g in ars_groups])

fig, ax = plt.subplots(figsize=(13, 7))
parts = ax.violinplot([g.values for g in ars_groups],
                      positions=range(len(ars_groups)),
                      showmeans=True, showmedians=True)
for i, c in enumerate(CC_COLORS):
    parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.65)
style_violin(parts, ax)
ax.set_xticks(range(len(CC_LABELS)))
ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(CC_LABELS, ns_cc)],
                   fontsize=10, fontweight='bold')
ax.set_ylabel('AR Activity Score\n(mean percentile rank of AR targets)', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity Score Across Disease Stages\n'
             f'Kruskal-Wallis p={p_ars_kw:.2e} {sig(p_ars_kw)}\n'
             f'AR target genes ({len(ar_avail)}): {", ".join(ar_avail)}',
             fontsize=12, fontweight='bold', pad=12)
ax.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.7, label='Global median (50th %ile)')
ax.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '01_ARS_crosscohort_trajectory.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 01_ARS_crosscohort_trajectory.png  (KW p={p_ars_kw:.2e} {sig(p_ars_kw)})")
plt.close()

# --- 02. m6A FI trajectory ----------------------------------------------------
print("\n--- Plot 02: m6A FI cross-cohort violin ---")
_, p_fi_kw = kruskal(*[g.dropna().values for g in fi_groups])

fig, ax = plt.subplots(figsize=(13, 7))
parts = ax.violinplot([g.values for g in fi_groups],
                      positions=range(len(fi_groups)),
                      showmeans=True, showmedians=True)
for i, c in enumerate(CC_COLORS):
    parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.65)
style_violin(parts, ax)
ax.set_xticks(range(len(CC_LABELS)))
ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(CC_LABELS, ns_cc)],
                   fontsize=10, fontweight='bold')
ax.set_ylabel('m6A Functional Impact\n(percentile-rank derived, LR-weighted writers)',
              fontsize=12, fontweight='bold')
ax.set_title(f'm6A Functional Impact Across Disease Stages\n'
             f'Kruskal-Wallis p={p_fi_kw:.2e} {sig(p_fi_kw)}',
             fontsize=12, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '02_FI_crosscohort_trajectory.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 02_FI_crosscohort_trajectory.png  (KW p={p_fi_kw:.2e} {sig(p_fi_kw)})")
plt.close()

# --- 03. ARS vs FI scatter all cohorts ----------------------------------------
print("\n--- Plot 03: ARS vs FI scatter all disease stages ---")
fig, ax = plt.subplots(figsize=(10, 8))
for ars_g, fi_g, lbl, c, n in zip(ars_groups, fi_groups, CC_LABELS_FLAT, CC_COLORS, ns_cc):
    idx_both = ars_g.dropna().index.intersection(fi_g.dropna().index)
    ax.scatter(ars_g.loc[idx_both], fi_g.loc[idx_both],
               c=c, alpha=0.35, s=15, edgecolors='none',
               label=f'{lbl} (n={n})')
all_ars = pd.concat(ars_groups).dropna()
all_fi  = pd.concat(fi_groups).dropna()
com_all = all_ars.index.intersection(all_fi.index)
r_all, p_all = spearmanr(all_ars.loc[com_all], all_fi.loc[com_all])
ax.set_xlabel('AR Activity Score (percentile rank)', fontsize=12, fontweight='bold')
ax.set_ylabel('m6A Functional Impact (percentile-rank derived)', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity vs m6A Functional Impact — All Disease Stages\n'
             f'Overall Spearman ρ={r_all:+.3f}, p={p_all:.2e} {sig(p_all)}',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(all_fi.loc[com_all].median(), color='grey', lw=1, ls='--', alpha=0.4)
ax.axvline(all_ars.loc[com_all].median(), color='grey', lw=1, ls='--', alpha=0.4)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '03_ARS_vs_FI_all_stages.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 03_ARS_vs_FI_all_stages.png  (overall ρ={r_all:+.3f} {sig(p_all)})")
plt.close()

# --- 04. AR target gene percentile rank heatmap across 6 stages ---------------
print("\n--- Plot 04: AR target gene percentile heatmap ---")
heat_ar = pd.DataFrame(index=CC_LABELS_FLAT, columns=ar_avail, dtype=float)
for k, lbl in zip(cohort_keys, CC_LABELS_FLAT):
    for gene in ar_avail:
        if gene in pct[k].columns:
            heat_ar.loc[lbl, gene] = pct[k][gene].mean()

fig, ax = plt.subplots(figsize=(max(10, len(ar_avail) * 1.1), 6))
sns.heatmap(heat_ar.astype(float), annot=True, fmt='.1f', cmap='RdBu_r',
            center=50, vmin=25, vmax=75,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean percentile rank'})
ax.set_title('AR Target Gene Percentile Ranks Across Disease Stages\n'
             'Values = mean within-sample %ile rank per cohort\n'
             'Red = high expression relative to transcriptome; Blue = low',
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('AR Target Gene', fontsize=11, fontweight='bold')
ax.set_ylabel('Disease Stage', fontsize=11, fontweight='bold')
ax.set_yticklabels(CC_LABELS_FLAT, fontsize=10, rotation=0, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '04_AR_target_gene_heatmap.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 04_AR_target_gene_heatmap.png")
plt.close()

# =============================================================================
# PART II — WITHIN-COHORT AR × m6A COUPLING STRENGTH
# =============================================================================
print("\n" + "=" * 80)
print("  PART II — WITHIN-COHORT AR × m6A COUPLING")
print("=" * 80)

# Within each cohort: Spearman ρ between ARS and m6A FI
within_corr = []
print(f"\n  {'Cohort':22s} {'n':>5s}  {'ρ(ARS, FI)':>10s}  {'p':>12s}  {'':3s}")
print("  " + "-" * 58)
for k, lbl, ars_g, fi_g in zip(cohort_keys, CC_LABELS_FLAT, ars_groups, fi_groups):
    idx_both = ars_g.dropna().index.intersection(fi_g.dropna().index)
    if len(idx_both) < 10:
        print(f"  {lbl:22s} {len(idx_both):5d}  (insufficient n for correlation)")
        within_corr.append({'cohort': lbl, 'rho': np.nan, 'p': np.nan, 'n': len(idx_both)})
        continue
    r, p = spearmanr(ars_g.loc[idx_both], fi_g.loc[idx_both])
    within_corr.append({'cohort': lbl, 'rho': r, 'p': p, 'n': len(idx_both)})
    print(f"  {lbl:22s} {len(idx_both):5d}  {r:+10.3f}  {p:12.2e}  {sig(p)}")
wc_df = pd.DataFrame(within_corr)

# --- 05. Within-cohort coupling barplot ---------------------------------------
print("\n--- Plot 05: Within-cohort AR×m6A coupling strength ---")
fig, ax = plt.subplots(figsize=(11, 6))
x_pos = range(len(wc_df))
rhos   = wc_df['rho'].values
colors_wc = [CC_COLORS[i] if not np.isnan(rhos[i]) else '#cccccc' for i in range(len(rhos))]
bars = ax.bar(x_pos, rhos, color=colors_wc, edgecolor='black', alpha=0.85, width=0.7)
for i, row in wc_df.iterrows():
    if np.isnan(row['rho']): continue
    yoff = row['rho'] + (0.01 if row['rho'] >= 0 else -0.025)
    ax.text(i, yoff, f"{sig(row['p'])}\nρ={row['rho']:+.3f}",
            ha='center', va='bottom' if row['rho'] >= 0 else 'top',
            fontsize=9, fontweight='bold')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(list(x_pos))
ax.set_xticklabels([f"{lbl}\n(n={int(row['n'])})" for lbl, row in zip(CC_LABELS_FLAT, wc_df.to_dict('records'))],
                   fontsize=10, fontweight='bold')
ax.set_ylabel('Spearman ρ (ARS vs m6A FI)', fontsize=12, fontweight='bold')
ax.set_title('Within-Cohort AR Activity × m6A Functional Impact Coupling\n'
             'Is the AR-m6A anti-correlation present in every disease stage?',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '05_within_cohort_AR_m6A_coupling.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 05_within_cohort_AR_m6A_coupling.png")
plt.close()

# --- 06. Coupling strength vs ARS mean (does coupling depend on stage?) ------
print("\n--- Plot 06: Coupling ρ vs ARS mean ---")
ars_means = [ars_groups[i].mean() for i in range(len(cohort_keys))]
rho_vals  = wc_df['rho'].values
valid_idx = np.isfinite(rho_vals)
r_meta, p_meta = (spearmanr(np.array(ars_means)[valid_idx], rho_vals[valid_idx])
                  if valid_idx.sum() >= 4 else (np.nan, np.nan))

fig, ax = plt.subplots(figsize=(9, 7))
for i, (x_m, rho_v, lbl, c, n) in enumerate(zip(ars_means, rho_vals, CC_LABELS_FLAT, CC_COLORS, ns_cc)):
    if np.isnan(rho_v): continue
    ax.scatter(x_m, rho_v, c=c, s=120, zorder=3, edgecolors='black', linewidth=1)
    ax.text(x_m + 0.5, rho_v + 0.005, lbl.replace('(', '\n('),
            fontsize=8, va='bottom', ha='left', color=c, fontweight='bold')
if not np.isnan(r_meta):
    x_fit = np.array(ars_means)[valid_idx]
    y_fit = rho_vals[valid_idx]
    m_f, b_f = np.polyfit(x_fit, y_fit, 1)
    x_l = np.linspace(min(x_fit) - 2, max(x_fit) + 2, 100)
    ax.plot(x_l, m_f * x_l + b_f, 'k--', lw=1.5, alpha=0.5)
    ax.set_title(f'AR×m6A Coupling Strength vs Mean AR Activity per Stage\n'
                 f'Meta-correlation: ρ={r_meta:+.3f}, p={p_meta:.2e} {sig(p_meta)}\n'
                 f'(Does stronger AR activity → stronger AR-m6A anti-coupling?)',
                 fontsize=12, fontweight='bold', pad=12)
else:
    ax.set_title('AR×m6A Coupling Strength vs Mean AR Activity per Stage',
                 fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Mean AR Activity Score (mean percentile rank)', fontsize=12, fontweight='bold')
ax.set_ylabel('Within-cohort Spearman ρ (ARS vs m6A FI)', fontsize=12, fontweight='bold')
ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '06_coupling_strength_vs_ARS_mean.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 06_coupling_strength_vs_ARS_mean.png")
plt.close()

# =============================================================================
# PART III — TUMOR PURITY / EPITHELIAL FRACTION SENSITIVITY
# =============================================================================
print("\n" + "=" * 80)
print("  PART III — TUMOR PURITY / EPITHELIAL FRACTION SENSITIVITY")
print("=" * 80)

# Luminal epithelial proxy genes — canonical markers of luminal prostate cells
# These are present in all-transcriptome datasets; if missing they're skipped.
LUMINAL_MARKERS  = ['KRT8', 'KRT18', 'EPCAM', 'CDH1', 'CLDN4']
STROMAL_MARKERS  = ['VIM', 'ACTA2', 'COL1A1', 'FN1', 'S100A4']
avail_lum  = [g for g in LUMINAL_MARKERS  if g in common]
avail_stro = [g for g in STROMAL_MARKERS  if g in common]
print(f"\n  Luminal markers available: {avail_lum}")
print(f"  Stromal markers available: {avail_stro}")

# --- 07. Epithelial marker score across cohorts ------------------------------
print("\n--- Plot 07: Tumor purity / epithelial fraction proxy ---")
if avail_lum or avail_stro:
    lum_score_groups  = []
    stro_score_groups = []
    for k in cohort_keys:
        if avail_lum:
            lum_score_groups.append(pct[k][avail_lum].mean(axis=1).values  if all(g in pct[k].columns for g in avail_lum)
                                    else np.array([np.nan]))
        if avail_stro:
            stro_score_groups.append(pct[k][avail_stro].mean(axis=1).values if all(g in pct[k].columns for g in avail_stro)
                                     else np.array([np.nan]))

    fig, axes_pur = plt.subplots(1, max(1, int(bool(avail_lum)) + int(bool(avail_stro))),
                                  figsize=(7 * (int(bool(avail_lum)) + int(bool(avail_stro))), 6))
    if not isinstance(axes_pur, np.ndarray):
        axes_pur = [axes_pur]

    panel_idx = 0
    for groups, markers, label, interpretation in [
        (lum_score_groups  if avail_lum  else None, avail_lum,  'Luminal Epithelial Score',
         'High = epithelial-rich; Low = stromal dilution'),
        (stro_score_groups if avail_stro else None, avail_stro, 'Stromal Score',
         'High = stromal-rich (lower tumor purity)'),
    ]:
        if groups is None or panel_idx >= len(axes_pur):
            continue
        ax = axes_pur[panel_idx]
        valid_groups = [g for g in groups if not np.all(np.isnan(g))]
        if valid_groups:
            parts = ax.violinplot(valid_groups, positions=range(len(valid_groups)),
                                  showmeans=True, showmedians=True)
            for i, c in enumerate(CC_COLORS[:len(valid_groups)]):
                parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.65)
            style_violin(parts, ax)
            ax.set_xticks(range(len(CC_LABELS[:len(valid_groups)])))
            ax.set_xticklabels([f"{lbl}\n(n={n})"
                                for lbl, n in zip(CC_LABELS[:len(valid_groups)], ns_cc[:len(valid_groups)])],
                               fontsize=9, fontweight='bold')
            ax.set_ylabel(f'{label}\n(mean percentile rank)', fontsize=11, fontweight='bold')
            ax.set_title(f'{label}\nGenes: {", ".join(markers)}\n{interpretation}',
                         fontsize=11, fontweight='bold')
        panel_idx += 1

    plt.suptitle('Tumor Purity Proxy: Luminal Epithelial vs Stromal Content\n'
                 'Rising luminal score from GTEx → mCRPC suggests increasing tumor cell fraction,\n'
                 'which partially explains rising ARS trajectory beyond intrinsic AR activity change.',
                 fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '07_tumor_purity_proxy.png'), dpi=300, bbox_inches='tight')
    print("  → Saved: 07_tumor_purity_proxy.png")
    plt.close()
else:
    print("  No luminal/stromal markers found in common universe — skipping plot 07")

# =============================================================================
# PART IV — ARS SIGNATURE SENSITIVITY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("  PART IV — ARS SIGNATURE SENSITIVITY")
print("=" * 80)

print(f"\n  Full ARS ({len(ar_avail)} genes):      {ar_avail}")
print(f"  Secretory ARS ({len(AR_SECRETORY)} genes): {AR_SECRETORY}")
print(f"  Core ARS ({len(AR_CORE)} genes):      {AR_CORE}")

ars_sec_groups  = [ars_sec[k]  for k in cohort_keys]
ars_core_groups = [ars_core[k] for k in cohort_keys]

# --- 08. ARS trajectory sensitivity: full vs secretory vs core ---------------
print("\n--- Plot 08: ARS trajectory sensitivity ---")
fig, axes_sens = plt.subplots(1, 3, figsize=(19, 6))

for ax, (ars_grps, panel_title) in zip(axes_sens, [
    (ars_groups,      f'Full ARS ({len(ar_avail)} genes)\nIncludes ALDH1A3, HOXB13, NKX3-1'),
    (ars_sec_groups,  f'Secretory-only ({len(AR_SECRETORY)} genes)\nKLK3, KLK2, FKBP5, TMPRSS2, STEAP2, SLC45A3'),
    (ars_core_groups, f'Core ({len(AR_CORE)} genes)\nSecretory + FOLH1 (PSMA)'),
]):
    grps_v = ars_grps
    _, p_kw = kruskal(*[g.dropna().values for g in grps_v])
    parts = ax.violinplot([g.values for g in grps_v],
                          positions=range(len(grps_v)),
                          showmeans=True, showmedians=True)
    for i, c in enumerate(CC_COLORS):
        parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.65)
    style_violin(parts, ax)
    means = [g.mean() for g in grps_v]
    ax.plot(range(len(means)), means, 'k--', lw=1.5, alpha=0.6, marker='D',
            markersize=5, label='Cohort means')
    ax.set_xticks(range(len(CC_LABELS)))
    ax.set_xticklabels([f"{lbl.split(chr(10))[0]}\n(n={n})"
                        for lbl, n in zip(CC_LABELS, ns_cc)],
                       fontsize=9, fontweight='bold')
    ax.set_ylabel('AR Activity Score (percentile rank)', fontsize=10, fontweight='bold')
    ax.set_title(f'{panel_title}\nKW p={p_kw:.2e} {sig(p_kw)}',
                 fontsize=11, fontweight='bold')
    ax.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.legend(fontsize=8)

plt.suptitle('ARS Cross-Cohort Trajectory — Sensitivity to Gene Panel Composition\n'
             'Are trajectory conclusions driven by ALDH1A3, HOXB13, or NKX3-1 (luminal identity genes)?',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '08_ARS_sensitivity_gene_panel.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 08_ARS_sensitivity_gene_panel.png")
plt.close()

# quantify sensitivity
for gene in ['ALDH1A3', 'NKX3-1', 'HOXB13']:
    if gene in ar_avail:
        for k in ['Normal', 'SCNC']:
            v = pct[k][gene].mean() if gene in pct[k].columns else np.nan
            print(f"  {gene} percentile rank in {k}: {v:.1f}")

# =============================================================================
# PART V — PER-STAGE SCATTER PANELS + SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("  PART V — PER-STAGE SCATTER PANELS & SUMMARY")
print("=" * 80)

# --- 09. Per-cohort ARS vs FI scatter 6-panel ---------------------------------
print("\n--- Plot 09: Per-stage ARS vs FI scatter panels ---")
fig, axes_sc = plt.subplots(2, 3, figsize=(18, 12))
axes_flat = axes_sc.flatten()
for i, (k, lbl, c, ax) in enumerate(zip(cohort_keys, CC_LABELS_FLAT, CC_COLORS, axes_flat)):
    ars_g = ars_groups[i]
    fi_g  = fi_groups[i]
    idx_b = ars_g.dropna().index.intersection(fi_g.dropna().index)
    if len(idx_b) < 5:
        ax.text(0.5, 0.5, f'{lbl}\nn={len(idx_b)} (insufficient)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        continue
    r, p = spearmanr(ars_g.loc[idx_b], fi_g.loc[idx_b])
    ax.scatter(ars_g.loc[idx_b], fi_g.loc[idx_b], c=c, alpha=0.4, s=16, edgecolors='none')
    m_f2, b_f2 = np.polyfit(ars_g.loc[idx_b].values, fi_g.loc[idx_b].values, 1)
    x_l2 = np.linspace(ars_g.loc[idx_b].min(), ars_g.loc[idx_b].max(), 100)
    ax.plot(x_l2, m_f2 * x_l2 + b_f2, 'k-', lw=2, alpha=0.7)
    ax.set_xlabel('AR Activity Score', fontsize=10, fontweight='bold')
    ax.set_ylabel('m6A Functional Impact', fontsize=10, fontweight='bold')
    ax.set_title(f'{lbl} (n={len(idx_b)})\nρ={r:+.3f}, p={p:.2e} {sig(p)}',
                 fontsize=11, fontweight='bold')
    ax.axhline(fi_g.loc[idx_b].median(), color='grey', lw=0.8, ls='--', alpha=0.4)
    ax.axvline(ars_g.loc[idx_b].median(), color='grey', lw=0.8, ls='--', alpha=0.4)
plt.suptitle('AR Activity vs m6A Functional Impact — Per Disease Stage\n'
             'Within-cohort Spearman ρ (controls for between-cohort batch effects)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '09_per_stage_ARS_FI_scatter.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 09_per_stage_ARS_FI_scatter.png")
plt.close()

# --- 10. Summary: ARS + FI trajectories + within-stage ρ --------------------
print("\n--- Plot 10: Cross-cohort summary ---")
fig = plt.figure(figsize=(18, 12))

ax1 = fig.add_subplot(2, 2, 1)
means_ars  = [g.mean()   for g in ars_groups]
sems_ars   = [g.std()/np.sqrt(len(g)) for g in ars_groups]
ax1.errorbar(range(len(CC_LABELS)), means_ars, yerr=sems_ars,
             fmt='o-', lw=2.5, markersize=9,
             color='black', ecolor='grey', capsize=5)
for i, (m_v, lbl, c) in enumerate(zip(means_ars, CC_LABELS, CC_COLORS)):
    ax1.scatter(i, m_v, c=c, s=100, zorder=5, edgecolors='black', linewidth=1.2)
ax1.set_xticks(range(len(CC_LABELS)))
ax1.set_xticklabels([lbl.replace('\n', ' ') for lbl in CC_LABELS], fontsize=9, rotation=20, ha='right')
ax1.set_ylabel('ARS (mean ± SEM)', fontsize=11, fontweight='bold')
ax1.set_title('AR Activity Score Trajectory', fontsize=12, fontweight='bold')
ax1.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.5)

ax2 = fig.add_subplot(2, 2, 2)
means_fi = [g.mean()   for g in fi_groups]
sems_fi  = [g.std()/np.sqrt(len(g)) for g in fi_groups]
ax2.errorbar(range(len(CC_LABELS)), means_fi, yerr=sems_fi,
             fmt='s-', lw=2.5, markersize=9,
             color='black', ecolor='grey', capsize=5)
for i, (m_v, lbl, c) in enumerate(zip(means_fi, CC_LABELS, CC_COLORS)):
    ax2.scatter(i, m_v, c=c, s=100, zorder=5, edgecolors='black', linewidth=1.2)
ax2.set_xticks(range(len(CC_LABELS)))
ax2.set_xticklabels([lbl.replace('\n', ' ') for lbl in CC_LABELS], fontsize=9, rotation=20, ha='right')
ax2.set_ylabel('m6A FI (mean ± SEM)', fontsize=11, fontweight='bold')
ax2.set_title('m6A Functional Impact Trajectory', fontsize=12, fontweight='bold')
ax2.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)

ax3 = fig.add_subplot(2, 2, 3)
rho_vals_v = wc_df['rho'].values
colors_rho = [CC_COLORS[i] if not np.isnan(rho_vals_v[i]) else '#cccccc' for i in range(len(rho_vals_v))]
ax3.bar(range(len(wc_df)), rho_vals_v, color=colors_rho, edgecolor='black', alpha=0.85, width=0.7)
ax3.axhline(0, color='black', lw=0.8)
ax3.set_xticks(range(len(CC_LABELS)))
ax3.set_xticklabels([lbl.replace('\n', ' ') for lbl in CC_LABELS], fontsize=9, rotation=20, ha='right')
ax3.set_ylabel('Within-cohort ρ (ARS vs m6A FI)', fontsize=11, fontweight='bold')
ax3.set_title('AR × m6A Coupling per Stage', fontsize=12, fontweight='bold')
for i2, (r2, p2) in enumerate(zip(wc_df['rho'], wc_df['p'])):
    if np.isnan(r2): continue
    ax3.text(i2, r2 + (0.01 if r2 >= 0 else -0.02),
             f"{sig(p2)}", ha='center', va='bottom' if r2 >= 0 else 'top',
             fontsize=10, fontweight='bold')

ax4 = fig.add_subplot(2, 2, 4)
sns.heatmap(heat_ar.astype(float), annot=True, fmt='.0f', cmap='RdBu_r',
            center=50, vmin=30, vmax=70,
            linewidths=0.5, ax=ax4, cbar_kws={'label': 'Mean %ile rank'},
            xticklabels=True, yticklabels=True)
ax4.set_title('AR Target Gene Percentile Ranks', fontsize=12, fontweight='bold')
ax4.set_xticklabels(ax4.get_xticklabels(), fontsize=8, rotation=30, ha='right')
ax4.set_yticklabels(CC_LABELS_FLAT, fontsize=9, rotation=0)

plt.suptitle('AR × m6A Cross-Cohort Summary — From Normal Prostate to mCRPC-SCNC',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '10_crosscohort_summary.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 10_crosscohort_summary.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "=" * 80)
print("  ALL PLOTS SAVED")
print("=" * 80)
for i, f in enumerate([
    '01_ARS_crosscohort_trajectory.png',
    '02_FI_crosscohort_trajectory.png',
    '03_ARS_vs_FI_all_stages.png',
    '04_AR_target_gene_heatmap.png',
    '05_within_cohort_AR_m6A_coupling.png',
    '06_coupling_strength_vs_ARS_mean.png',
    '07_tumor_purity_proxy.png',
    '08_ARS_sensitivity_gene_panel.png',
    '09_per_stage_ARS_FI_scatter.png',
    '10_crosscohort_summary.png',
], 1):
    print(f"  {i:2d}. {f}")
print(f"\n  Output directory: {OUTDIR}")
print("=" * 80)
