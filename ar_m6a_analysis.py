#!/usr/bin/env python3
"""
ar_m6a_analysis.py — Androgen Receptor (AR) × m6A coordination in prostate cancer.

Hypothesis: AR transcriptional activity co-regulates elements of the m6A
writer complex, particularly in the RBM15/RBM15B paralog axis, creating a
mechanistic link between AR-driven disease progression and m6A epitranscriptomic reprogramming.

Analysis structure:
  Part I    — AR Activity Score (ARS): construction and validation (01–04)
  Part II   — Per-gene m6A × AR correlations (05–07)
  Part III  — m6A axes vs AR continuous score (08–11)
  Part IV   — AR clinical context: treatment, amp/mut, PSA response (12–15)
  Part V    — AR × m6A survival synergy (16–19)
  Part VI   — TCGA validation (20–22)
  Part VII  — Site-stratified AR activity (23–25)
  Part VIII — AR receptor & bypass mechanisms (26–29)
  Part IX   — Cross-cohort AR activity trajectory → see ar_crosscohort_analysis.py
  Part X    — Confound-controlled AR × m6A: Adeno-only + partial corr + mediation (30–36)

Methodological improvements vs prior version:
  - ARS: flat mean replaced by PC1 (dominant co-expression axis, unsupervised)
  - ARS_LR: supervised alternative trained on AR Amp/Mut vs WT (sensitivity check)
  - AR Signaling Efficiency: ARS residualized on AR mRNA (captures mRNA-independent induction)
  - TCGA m6A axes: now use same LR writer weights as mCRPC (methodological consistency)
  - New plots 01b/01c: scoring method stability + AR efficiency validation

Output: plots_ar_m6a/

Usage:
    micromamba run -n rnaseq python ar_m6a_analysis.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, kruskal, pearsonr
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.rcParams['savefig.dpi'] = 300

# ── m6a package imports ───────────────────────────────────────────────────────
from m6a.config import (
    OUTDIR_AR_M6A as OUTDIR,
    MCRPC_LOG2CPM, MCRPC_META_CSV,
    MCRPC_SITE_ORDER, MCRPC_SITE_COLORS,
)
from m6a.genes import (
    MCRPC_ALL_GENES    as ALL_M6A_GENES,
    MCRPC_WRITER_GENES as WRITER_GENES,
    MCRPC_ERASER_GENES as ERASER_GENES,
    MCRPC_READER_ONCOGENIC  as READER_ONCOGENIC,
    MCRPC_READER_SUPPRESSIVE as READER_SUPPRESSIVE,
    MCRPC_GENE_ROLES   as gene_roles,
    MCRPC_GENE_ORDER   as gene_order,
    MCRPC_MANUAL_WEIGHTS as MANUAL_WEIGHTS,
    AR_TARGET_GENES,
)
from m6a.stats import sig
from m6a.normalization import zscore_normalize, percentile_rank_matrix
from m6a.plotting import style_violin
from m6a.scoring import compute_axes
from m6a.data.loaders import (
    load_mcrpc, load_tcga, load_gtex, load_adj_normal, load_mcspc,
    build_common_universe,
)
from m6a.genes import ALL_M6A_GENES as CROSS_COHORT_M6A_GENES

os.makedirs(OUTDIR, exist_ok=True)
SITE_ORDER  = MCRPC_SITE_ORDER
SITE_COLORS = MCRPC_SITE_COLORS

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
print("=" * 80)
print("  AR × m6A COORDINATION IN PROSTATE CANCER")
print("=" * 80)

df,   meta   = load_mcrpc()
df_tc, meta_tc = load_tcga()

print(f"\n  mCRPC: {df.shape[0]} patients")
print(f"  TCGA:  {df_tc.shape[0]} patients")

# ── mCRPC: z-score normalise ─────────────────────────────────────────────────
from sklearn.decomposition import PCA as _PCA
from sklearn.linear_model import LogisticRegression as _LR
from scipy.stats import linregress as _linregress

all_genes = list(set(ALL_M6A_GENES + AR_TARGET_GENES + ['AR']))
all_genes = [g for g in all_genes if g in df.columns]
z_all = zscore_normalize(df[all_genes])

# ── mCRPC m6A writer weights — derived first, applied to both mCRPC and TCGA ─
# (Adeno/SCNC discrimination, L2 logistic regression on 7 writer genes)
idx_hist = meta[meta['histology'].isin(['Adenocarcinoma','SCNC'])].index.intersection(df.index)
X_wr     = z_all.loc[idx_hist, WRITER_GENES].values
y_hs     = (meta.loc[idx_hist, 'histology'] == 'SCNC').astype(int).values
lr_m     = _LR(penalty='l2', C=1.0, max_iter=1000, random_state=42).fit(X_wr, y_hs)
lr_abs   = np.abs(lr_m.coef_[0])
lr_wts   = lr_abs / lr_abs.sum() * sum(MANUAL_WEIGHTS.values())
LR_WEIGHTS = {g: w for g, w in zip(WRITER_GENES, lr_wts)}

# ── mCRPC m6A axes ────────────────────────────────────────────────────────────
dd_w = sum(z_all[g] * w for g, w in LR_WEIGHTS.items()) / sum(LR_WEIGHTS.values())
meta['m6A_Net_Deposition']    = dd_w - z_all[ERASER_GENES].mean(axis=1)
meta['m6A_Oncogenic_Readout'] = z_all[READER_ONCOGENIC].mean(axis=1) - z_all[READER_SUPPRESSIVE].mean(axis=1)
meta['m6A_Functional_Impact'] = meta['m6A_Net_Deposition'] * 0.435 + meta['m6A_Oncogenic_Readout'] * 0.565

# ── AR Activity Score — three parallel methods ────────────────────────────────
# Methodological symmetry with m6A: test stability across scoring approaches.
ar_avail = [g for g in AR_TARGET_GENES if g in z_all.columns]
print(f"\n  AR signature genes in mCRPC: {len(ar_avail)}/{len(AR_TARGET_GENES)}")

# Method 1 — Flat mean (reference baseline; equal gene weights)
ARS_simple = z_all[ar_avail].mean(axis=1)

# Method 2 — PC1 (unsupervised; captures dominant shared AR target co-expression)
_pca_ar  = _PCA(n_components=1, random_state=42)
_pc1_raw = _pca_ar.fit_transform(z_all[ar_avail].values)[:, 0]
if np.corrcoef(_pc1_raw, ARS_simple.values)[0, 1] < 0:   # canonical direction
    _pc1_raw = -_pc1_raw
ARS_PC1     = pd.Series(_pc1_raw, index=z_all.index, name='ARS_PC1')
ARS_PC1_var = _pca_ar.explained_variance_ratio_[0]

# Method 3 — LR-weighted (supervised; trained on AR Amp/Mut vs WT)
_idx_amp_lr = meta[meta['AR - Amplification and/or Mutation'].isin([0.0, 1.0])].index.intersection(df.index)
_X_ar_lr    = z_all.loc[_idx_amp_lr, ar_avail].values
_y_ar_lr    = (meta.loc[_idx_amp_lr, 'AR - Amplification and/or Mutation'] == 1.0).astype(int).values
_lr_ar      = _LR(penalty='l2', C=1.0, max_iter=1000, random_state=42).fit(_X_ar_lr, _y_ar_lr)
_lr_ar_wts  = np.abs(_lr_ar.coef_[0]); _lr_ar_wts = _lr_ar_wts / _lr_ar_wts.sum()
LR_AR_WEIGHTS = {g: w for g, w in zip(ar_avail, _lr_ar_wts)}
ARS_LR = pd.Series(
    z_all[ar_avail].values @ np.array([LR_AR_WEIGHTS[g] for g in ar_avail]),
    index=z_all.index, name='ARS_LR')

# Primary ARS = PC1 (unsupervised; most robust to label choice)
meta['AR_Activity_Score'] = ARS_PC1
meta['ARS_simple']         = ARS_simple
meta['ARS_LR']             = ARS_LR

_r_pc1_mean = np.corrcoef(ARS_PC1.values, ARS_simple.values)[0, 1]
_r_pc1_lr   = np.corrcoef(ARS_PC1.values, ARS_LR.values)[0, 1]
print(f"  PC1 variance explained: {ARS_PC1_var:.1%}  |  "
      f"PC1 vs mean ρ={_r_pc1_mean:.3f}  |  PC1 vs LR-ARS ρ={_r_pc1_lr:.3f}")
print(f"  LR ARS trained on n={int(_y_ar_lr.sum())} Amp/Mut vs {int((1-_y_ar_lr).sum())} WT")
top_lr_genes = sorted(LR_AR_WEIGHTS, key=LR_AR_WEIGHTS.get, reverse=True)[:3]
print(f"  Top LR-weighted AR genes: {top_lr_genes}")

# ── AR Signaling Efficiency: ARS residualized on AR mRNA ─────────────────────
# Captures AR target induction *beyond* what AR expression level alone predicts.
# High efficiency → targets are strongly induced despite moderate AR mRNA
# Low efficiency → AR mRNA present but not driving target gene expression
if 'AR' in z_all.columns:
    _sh  = z_all['AR'].dropna().index.intersection(meta['AR_Activity_Score'].dropna().index)
    _sl, _ic, *_ = _linregress(z_all.loc[_sh, 'AR'].values,
                                meta.loc[_sh, 'AR_Activity_Score'].values)
    meta['AR_Signaling_Efficiency'] = (
        meta['AR_Activity_Score'] - (_sl * z_all['AR'].reindex(meta.index) + _ic)
    )
    print(f"  AR signaling efficiency: AR_mRNAβ={_sl:.3f} "
          f"(residual ARS after controlling for AR mRNA level)")
else:
    meta['AR_Signaling_Efficiency'] = np.nan
    print("  WARNING: AR gene not in z_all — AR_Signaling_Efficiency set to NaN")

# ── TCGA z-score normalise ────────────────────────────────────────────────────
tc_genes    = list(set(ALL_M6A_GENES + AR_TARGET_GENES + ['AR']))
tc_genes    = [g for g in tc_genes if g in df_tc.columns]
z_tc        = zscore_normalize(df_tc[tc_genes])
ar_tc       = [g for g in AR_TARGET_GENES if g in z_tc.columns]
meta_tc     = meta_tc.copy()
meta_tc['AR_Activity_Score'] = z_tc[ar_tc].mean(axis=1)

# ── TCGA m6A axes — consistent LR writer weights transferred from mCRPC ──────
# Same weights as mCRPC (no retraining); makes TCGA and mCRPC scores comparable.
writer_tc = [g for g in WRITER_GENES if g in z_tc.columns]
eraser_tc = [g for g in ERASER_GENES if g in z_tc.columns]
onco_tc   = [g for g in READER_ONCOGENIC   if g in z_tc.columns]
supp_tc   = [g for g in READER_SUPPRESSIVE if g in z_tc.columns]
_wts_tc   = {g: LR_WEIGHTS[g] for g in writer_tc if g in LR_WEIGHTS}
_wts_sum  = sum(_wts_tc.values())
_dd_tc    = sum(z_tc[g] * _wts_tc[g] for g in _wts_tc) / _wts_sum
meta_tc['m6A_Net_Deposition']    = _dd_tc - z_tc[eraser_tc].mean(axis=1)
meta_tc['m6A_Oncogenic_Readout'] = z_tc[onco_tc].mean(axis=1) - z_tc[supp_tc].mean(axis=1)
meta_tc['m6A_Functional_Impact'] = (meta_tc['m6A_Net_Deposition'] * 0.435 +
                                     meta_tc['m6A_Oncogenic_Readout'] * 0.565)
print(f"  TCGA m6A: LR-weighted writers ({len(_wts_tc)} genes, same weights as mCRPC)")

# ── Group indices ─────────────────────────────────────────────────────────────
idx_adeno = meta[meta['histology'] == 'Adenocarcinoma'].index.intersection(df.index)
idx_scnc  = meta[meta['histology'] == 'SCNC'].index.intersection(df.index)
idx_lum   = meta[meta['Luminal/Basal Cluster'] == 'Luminal'].index.intersection(df.index)
idx_bas   = meta[meta['Luminal/Basal Cluster'] == 'Basal'].index.intersection(df.index)

print(f"  mCRPC groups: Adeno n={len(idx_adeno)}, SCNC n={len(idx_scnc)}, "
      f"Luminal n={len(idx_lum)}, Basal n={len(idx_bas)}")


# ===========================================================================
# PART I — AR ACTIVITY SCORE VALIDATION
# ===========================================================================
print("\n" + "=" * 80)
print("  PART I — AR ACTIVITY SCORE VALIDATION")
print("=" * 80)

# --- 01. AR target gene co-expression heatmap --------------------------------
print("\n--- Plot 01: AR target gene co-expression heatmap ---")
z_ar = z_all[ar_avail]
corr_ar = z_ar.corr(method='spearman')

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_ar, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Spearman ρ'})
ax.set_title('AR Target Genes — Co-expression (Spearman ρ, mCRPC)\n'
             'High positive correlations validate the AR Activity Score',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '01_AR_target_coexpression.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 01_AR_target_coexpression.png")
plt.close()

# --- 02. ARS by histology and Luminal/Basal ----------------------------------
print("\n--- Plot 02: ARS by histology and Luminal/Basal ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (col, val_a, val_b, la, lb, ca, cb) in zip(axes, [
    ('histology',            'Adenocarcinoma', 'SCNC',
     'Adenocarcinoma', 'SCNC', '#27ae60', '#8e44ad'),
    ('Luminal/Basal Cluster', 'Luminal',       'Basal',
     'Luminal', 'Basal', '#2980b9', '#c0392b'),
]):
    idx_a = meta[meta[col] == val_a].index.intersection(df.index)
    idx_b = meta[meta[col] == val_b].index.intersection(df.index)
    va = meta.loc[idx_a, 'AR_Activity_Score'].dropna().values
    vb = meta.loc[idx_b, 'AR_Activity_Score'].dropna().values
    _, p = mannwhitneyu(va, vb, alternative='two-sided')
    parts = ax.violinplot([va, vb], positions=[0, 1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(ca); parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor(cb); parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_max = max(va.max(), vb.max())
    ax.plot([0, 1], [y_max + 0.12, y_max + 0.12], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.15, f'p={p:.2e} {sig(p)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'{la}\n(n={len(va)})', f'{lb}\n(n={len(vb)})'],
                       fontsize=11, fontweight='bold')
    ax.set_ylabel('AR Activity Score', fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')

axes[0].set_title('AR Activity Score by Histology', fontsize=13, fontweight='bold')
axes[1].set_title('AR Activity Score by Luminal/Basal Subtype', fontsize=13, fontweight='bold')
plt.suptitle('AR Activity Score Validation — Known Biology Concordance',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '02_ARS_by_histology_lumbasline.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 02_ARS_by_histology_lumbasline.png")
plt.close()

# Print stats
for col, va_col, vb_col, la, lb in [
    ('histology',             'Adenocarcinoma', 'SCNC',    'Adeno', 'SCNC'),
    ('Luminal/Basal Cluster', 'Luminal',        'Basal',   'Lum',   'Bas'),
]:
    ia = meta[meta[col] == va_col].index.intersection(df.index)
    ib = meta[meta[col] == vb_col].index.intersection(df.index)
    va = meta.loc[ia, 'AR_Activity_Score'].dropna().values
    vb = meta.loc[ib, 'AR_Activity_Score'].dropna().values
    _, p = mannwhitneyu(va, vb, alternative='two-sided')
    print(f"  ARS {la}={va.mean():+.3f} vs {lb}={vb.mean():+.3f}  "
          f"Δ={vb.mean()-va.mean():+.3f}  p={p:.2e} {sig(p)}")

# --- 03. ARS by AR amplification/mutation status -----------------------------
print("\n--- Plot 03: ARS by AR Amp/Mut ---")
idx_amp    = meta[meta['AR - Amplification and/or Mutation'] == 1.0].index.intersection(df.index)
idx_no_amp = meta[meta['AR - Amplification and/or Mutation'] == 0.0].index.intersection(df.index)
va_amp = meta.loc[idx_amp,    'AR_Activity_Score'].dropna().values
va_wt  = meta.loc[idx_no_amp, 'AR_Activity_Score'].dropna().values
_, p_amp = mannwhitneyu(va_amp, va_wt, alternative='two-sided')

fig, ax = plt.subplots(figsize=(7, 6))
parts = ax.violinplot([va_wt, va_amp], positions=[0, 1], showmeans=True, showmedians=True)
parts['bodies'][0].set_facecolor('#95a5a6'); parts['bodies'][0].set_alpha(0.65)
parts['bodies'][1].set_facecolor('#e74c3c'); parts['bodies'][1].set_alpha(0.65)
style_violin(parts, ax)
y_max = max(va_wt.max(), va_amp.max())
ax.plot([0, 1], [y_max + 0.12, y_max + 0.12], 'k-', lw=1.2)
ax.text(0.5, y_max + 0.15, f'p={p_amp:.2e} {sig(p_amp)}',
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels([f'AR WT\n(n={len(va_wt)})', f'AR Amp/Mut\n(n={len(va_amp)})'],
                   fontsize=11, fontweight='bold')
ax.set_ylabel('AR Activity Score', fontsize=12, fontweight='bold')
ax.set_title('AR Activity Score by AR Genomic Alteration\n'
             'AR amplification/mutation → elevated AR transcriptional output',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.8, ls='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '03_ARS_by_AR_alteration.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 03_ARS_by_AR_alteration.png")
print(f"  ARS WT={va_wt.mean():+.3f} vs Amp/Mut={va_amp.mean():+.3f}  p={p_amp:.2e} {sig(p_amp)}")
plt.close()

# ===========================================================================
# PART II — PER-GENE m6A × AR CORRELATIONS
# ===========================================================================
print("\n" + "=" * 80)
print("  PART II — PER-GENE m6A × AR CORRELATIONS")
print("=" * 80)

ars = meta.loc[df.index, 'AR_Activity_Score'].dropna()

print(f"\n  {'Gene':12s} {'Role':22s} {'Spearman ρ':>10s} {'p':>12s} {'':3s}")
print("  " + "-" * 65)
gene_ar_corr = []
for gene in gene_order:
    v_gene = z_all.loc[ars.index, gene]
    r, p   = spearmanr(v_gene, ars)
    gene_ar_corr.append({'Gene': gene, 'Role': gene_roles[gene], 'rho': r, 'p': p, 'sig': sig(p)})
    print(f"  {gene:12s} {gene_roles[gene]:22s} {r:+10.3f} {p:12.2e} {sig(p):3s}")
gac = pd.DataFrame(gene_ar_corr)

# --- 05. Per-gene correlation barplot ----------------------------------------
print("\n--- Plot 05: Per-gene m6A × ARS correlation barplot ---")
colors_bar = []
for _, row in gac.iterrows():
    g = row['Gene']
    if g in WRITER_GENES:
        colors_bar.append('#2980b9' if row['rho'] >= 0 else '#85c1e9')
    elif g in ERASER_GENES:
        colors_bar.append('#e74c3c' if row['rho'] >= 0 else '#f1948a')
    elif g in READER_ONCOGENIC:
        colors_bar.append('#e67e22' if row['rho'] >= 0 else '#f0b27a')
    else:
        colors_bar.append('#8e44ad' if row['rho'] >= 0 else '#c39bd3')

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(gac)), gac['rho'], color=colors_bar, edgecolor='black', alpha=0.85)
for i, row in gac.iterrows():
    if row['sig'] != 'ns':
        y_pos = row['rho'] + (0.02 if row['rho'] >= 0 else -0.04)
        ax.text(i, y_pos, row['sig'], ha='center', va='bottom' if row['rho'] >= 0 else 'top',
                fontsize=9, fontweight='bold')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(range(len(gac)))
ax.set_xticklabels(
    [f"{r['Gene']}\n({r['Role'].split('(')[1].rstrip(')') if '(' in r['Role'] else r['Role']})"
     for _, r in gac.iterrows()],
    fontsize=8, rotation=30, ha='right',
)
ax.set_ylabel('Spearman ρ with AR Activity Score', fontsize=12, fontweight='bold')
ax.set_title('m6A Gene Correlation with AR Activity Score (mCRPC)\n'
             'W=Writer, E=Eraser, R-onc=Oncogenic Reader, R-sup=Suppressive Reader',
             fontsize=13, fontweight='bold', pad=12)
legend_els = [
    Patch(facecolor='#2980b9', label='Writer (+)'), Patch(facecolor='#85c1e9', label='Writer (−)'),
    Patch(facecolor='#e74c3c', label='Eraser (+)'),
    Patch(facecolor='#e67e22', label='Onco Reader (+)'), Patch(facecolor='#f0b27a', label='Onco Reader (−)'),
    Patch(facecolor='#8e44ad', label='Supp Reader (+)'), Patch(facecolor='#c39bd3', label='Supp Reader (−)'),
]
ax.legend(handles=legend_els, fontsize=8, loc='lower left', ncol=4)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '05_m6A_gene_ARS_correlation.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 05_m6A_gene_ARS_correlation.png")
plt.close()

shared_idx = ars.index.intersection(df.index)
cmap_hist  = meta.loc[df.index, 'histology'].map(
    {'Adenocarcinoma': '#27ae60', 'SCNC': '#8e44ad'}).fillna('grey')
cmap_lb    = meta.loc[df.index, 'Luminal/Basal Cluster'].map(
    {'Luminal': '#2980b9', 'Basal': '#c0392b'}).fillna('grey')

# --- 11. 2D landscape: ARS vs FI, dual coloring ------------------------------
print("\n--- Plot 11: 2D landscape ARS vs FI ---")
fig, axes_l = plt.subplots(1, 2, figsize=(16, 7))
for ax, label, cmap_s, legend_els in [
    (axes_l[0], 'Histology', cmap_hist,
     [Patch(facecolor='#27ae60', label='Adenocarcinoma'),
      Patch(facecolor='#8e44ad', label='SCNC'), Patch(facecolor='grey', label='Other')]),
    (axes_l[1], 'Luminal/Basal', cmap_lb,
     [Patch(facecolor='#2980b9', label='Luminal'),
      Patch(facecolor='#c0392b', label='Basal'), Patch(facecolor='grey', label='Other')]),
]:
    ax.scatter(meta.loc[shared_idx, 'AR_Activity_Score'],
               meta.loc[shared_idx, 'm6A_Functional_Impact'],
               c=cmap_s.loc[shared_idx].values, alpha=0.45, s=25, edgecolors='none')
    r, p = spearmanr(meta.loc[shared_idx, 'AR_Activity_Score'],
                     meta.loc[shared_idx, 'm6A_Functional_Impact'])
    # Median lines to define quadrants
    med_ars = meta.loc[shared_idx, 'AR_Activity_Score'].median()
    med_fi  = meta.loc[shared_idx, 'm6A_Functional_Impact'].median()
    ax.axvline(med_ars, color='grey', lw=1, ls='--', alpha=0.7)
    ax.axhline(med_fi,  color='grey', lw=1, ls='--', alpha=0.7)
    ax.set_xlabel('AR Activity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('m6A Functional Impact', fontsize=12, fontweight='bold')
    ax.set_title(f'AR Activity vs m6A Functional Impact\nColored by {label}\n'
                 f'Spearman ρ={r:+.3f}, p={p:.2e} {sig(p)}',
                 fontsize=12, fontweight='bold')
    ax.legend(handles=legend_els, fontsize=10, loc='upper left')
    # Quadrant labels
    ax_xl, ax_xr = ax.get_xlim()
    ax_yb, ax_yt = ax.get_ylim()
    ax.text(ax_xr * 0.9, ax_yt * 0.9, 'AR-hi\nm6A-hi', ha='center', va='top',
            fontsize=9, color='darkred', fontweight='bold', alpha=0.7)
    ax.text(ax_xl * 0.9, ax_yb * 0.85, 'AR-lo\nm6A-lo', ha='center', va='bottom',
            fontsize=9, color='steelblue', fontweight='bold', alpha=0.7)
plt.suptitle('AR Activity × m6A Functional Impact Landscape (mCRPC)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '11_ARS_FI_landscape.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 11_ARS_FI_landscape.png")
plt.close()

# ===========================================================================
# PART V — AR × m6A SURVIVAL SYNERGY
# ===========================================================================
print("\n" + "=" * 80)
print("  PART V — AR × m6A SURVIVAL SYNERGY")
print("=" * 80)

surv_idx = meta[['surv_months', 'vital_status']].dropna().index.intersection(df.index)
surv_df  = meta.loc[surv_idx, ['surv_months', 'vital_status',
                                 'AR_Activity_Score', 'm6A_Functional_Impact']].dropna()

med_ars_s = surv_df['AR_Activity_Score'].median()
med_fi_s  = surv_df['m6A_Functional_Impact'].median()

surv_df['AR_hi'] = (surv_df['AR_Activity_Score'] >= med_ars_s)
surv_df['FI_hi'] = (surv_df['m6A_Functional_Impact'] >= med_fi_s)
surv_df['quad']  = (surv_df['AR_hi'].astype(str) + '_' +
                    surv_df['FI_hi'].astype(str))
quad_map = {
    'True_True':   'AR-hi / FI-hi',
    'True_False':  'AR-hi / FI-lo',
    'False_True':  'AR-lo / FI-hi',
    'False_False': 'AR-lo / FI-lo',
}
surv_df['Quadrant'] = surv_df['quad'].map(quad_map)
quad_colors = {
    'AR-hi / FI-hi': '#e74c3c', 'AR-hi / FI-lo': '#e67e22',
    'AR-lo / FI-hi': '#3498db', 'AR-lo / FI-lo': '#27ae60',
}

# --- 16. KM four-quadrant ----------------------------------------------------
print("\n--- Plot 16: KM four-quadrant (ARS × FI) ---")
fig, ax = plt.subplots(figsize=(11, 8))
kmf = KaplanMeierFitter()
for quad, gr in surv_df.groupby('Quadrant'):
    if len(gr) < 5:
        continue
    kmf.fit(gr['surv_months'], gr['vital_status'],
            label=f"{quad} (n={len(gr)})")
    kmf.plot_survival_function(ax=ax, color=quad_colors[quad], linewidth=2)

ax.set_xlabel('Time (months)', fontsize=13, fontweight='bold')
ax.set_ylabel('Survival Probability', fontsize=13, fontweight='bold')
ax.set_title('Overall Survival — AR Activity × m6A Functional Impact\n'
             'Four-Quadrant Stratification (median splits)',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10, loc='lower left')
ax.set_ylim(0, 1.05)

# Log-rank: best vs worst quadrant
g_best = surv_df[surv_df['Quadrant'] == 'AR-lo / FI-lo']
g_worst = surv_df[surv_df['Quadrant'] == 'AR-hi / FI-hi']
if len(g_best) >= 5 and len(g_worst) >= 5:
    lr = logrank_test(g_best['surv_months'], g_worst['surv_months'],
                      g_best['vital_status'], g_worst['vital_status'])
    ax.text(0.98, 0.98, f'Best vs Worst:\nLog-rank p={lr.p_value:.4f} {sig(lr.p_value)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(facecolor='white', edgecolor='grey', alpha=0.8))
    print(f"  Best vs Worst quadrant: p={lr.p_value:.4f} {sig(lr.p_value)}")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '16_KM_ARxFI_quadrant.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 16_KM_ARxFI_quadrant.png")
plt.close()

# --- 17. KM ARS alone --------------------------------------------------------
print("\n--- Plot 17: KM ARS median split ---")
ars_hi = surv_df[surv_df['AR_hi']]
ars_lo = surv_df[~surv_df['AR_hi']]
lr_ars = logrank_test(ars_hi['surv_months'], ars_lo['surv_months'],
                       ars_hi['vital_status'], ars_lo['vital_status'])

# --- 18. Cox PH: ARS, FI, interaction ----------------------------------------
print("\n--- Plot 18: Cox PH model ---")
cox_df = surv_df[['surv_months', 'vital_status',
                   'AR_Activity_Score', 'm6A_Functional_Impact']].dropna().copy()
cox_df['AR_FI_interaction'] = cox_df['AR_Activity_Score'] * cox_df['m6A_Functional_Impact']
# Standardise continuous variables
for col in ['AR_Activity_Score', 'm6A_Functional_Impact', 'AR_FI_interaction']:
    cox_df[col] = (cox_df[col] - cox_df[col].mean()) / cox_df[col].std()

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='surv_months', event_col='vital_status')
print("\n  Cox PH Summary:")
print(cph.summary[['coef', 'exp(coef)', 'p', 'coef lower 95%', 'coef upper 95%']].to_string())

# --- 19. Forest plot (Cox HRs) -----------------------------------------------
print("\n--- Plot 19: Cox HR forest plot ---")
summary = cph.summary.copy()
labels_map = {
    'AR_Activity_Score': 'AR Activity Score',
    'm6A_Functional_Impact': 'm6A Functional Impact',
    'AR_FI_interaction': 'AR × m6A Interaction',
}
fig, ax = plt.subplots(figsize=(9, 5))
y_pos = range(len(summary))
for i, (varname, row) in enumerate(summary.iterrows()):
    hr   = row['exp(coef)']
    lo   = np.exp(row['coef lower 95%'])
    hi   = np.exp(row['coef upper 95%'])
    p_v  = row['p']
    color = '#e74c3c' if row['coef'] > 0 else '#3498db'
    ax.plot([lo, hi], [i, i], '-', color=color, lw=3, alpha=0.7)
    ax.plot(hr, i, 'o', color=color, markersize=10, zorder=5)
    label = labels_map.get(varname, varname)
    ax.text(-0.05, i, f'{label}\n(HR={hr:.2f}, p={p_v:.3f} {sig(p_v)})',
            ha='right', va='center', fontsize=10, transform=ax.get_yaxis_transform())

ax.axvline(1.0, color='black', lw=1, ls='--')
ax.set_yticks([])
ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12, fontweight='bold')
ax.set_title('Cox Proportional Hazards — AR Activity, m6A Functional Impact,\n'
             'and their Interaction (all standardised)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlim(left=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '19_Cox_HR_forest.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 19_Cox_HR_forest.png")
plt.close()

# Gleason grouping
meta_tc_gl = meta_tc.copy()
meta_tc_gl['Gleason_group'] = pd.cut(
    meta_tc_gl['gleason_score'],
    bins=[0, 6, 7, 10], labels=['≤6 (Low)', '7 (Int)', '≥8 (High)'],
)
gl_order  = ['≤6 (Low)', '7 (Int)', '≥8 (High)']
gl_colors = ['#27ae60', '#f39c12', '#e74c3c']
shared_tc = meta_tc_gl.dropna(subset=['Gleason_group']).index

print(f"\n  TCGA patients with Gleason: {len(shared_tc)}")
print(f"  Gleason distribution:")
print(meta_tc_gl['Gleason_group'].value_counts().to_string())

gl_c_map = meta_tc_gl['Gleason_group'].map(dict(zip(gl_order, gl_colors))).dropna()
shared_gl = gl_c_map.index
r_tc, p_tc = spearmanr(meta_tc_gl.loc[shared_gl, 'AR_Activity_Score'],
                        meta_tc_gl.loc[shared_gl, 'm6A_Functional_Impact'])

# --- 25. Summary panel: AR-m6A axis overview ---------------------------------
print("\n--- Plot 25: Summary — AR-m6A axis overview ---")
fig = plt.figure(figsize=(18, 12))

# Top row: three key scatters (per-gene, axis, TCGA)
ax1 = fig.add_subplot(2, 3, 1)
top_gene = gac.loc[gac['rho'].abs().idxmax(), 'Gene']
shared_sm = ars.index.intersection(df.index)
r_top, p_top = spearmanr(ars.loc[shared_sm],
                          z_all.loc[shared_sm, top_gene])
ax1.scatter(ars.loc[shared_sm], z_all.loc[shared_sm, top_gene],
            c=cmap_hist.loc[shared_sm].values, alpha=0.35, s=15, edgecolors='none')
m_sm, b_sm = np.polyfit(ars.loc[shared_sm].values,
                         z_all.loc[shared_sm, top_gene].values, 1)
x_sm = np.linspace(ars.loc[shared_sm].min(), ars.loc[shared_sm].max(), 100)
ax1.plot(x_sm, m_sm * x_sm + b_sm, 'k-', lw=2)
ax1.set_xlabel('AR Activity Score', fontsize=10, fontweight='bold')
ax1.set_ylabel(f'{top_gene} z-score', fontsize=10, fontweight='bold')
ax1.set_title(f'Top m6A Gene: {top_gene}\nρ={r_top:+.3f}, p={p_top:.2e} {sig(p_top)}',
              fontsize=11, fontweight='bold')

ax2 = fig.add_subplot(2, 3, 2)
r_fi_s, p_fi_s = spearmanr(meta.loc[shared_sm, 'AR_Activity_Score'],
                             meta.loc[shared_sm, 'm6A_Functional_Impact'])
ax2.scatter(meta.loc[shared_sm, 'AR_Activity_Score'],
            meta.loc[shared_sm, 'm6A_Functional_Impact'],
            c=cmap_hist.loc[shared_sm].values, alpha=0.35, s=15, edgecolors='none')
m2, b2 = np.polyfit(meta.loc[shared_sm, 'AR_Activity_Score'].values,
                     meta.loc[shared_sm, 'm6A_Functional_Impact'].values, 1)
x2 = np.linspace(meta.loc[shared_sm, 'AR_Activity_Score'].min(),
                  meta.loc[shared_sm, 'AR_Activity_Score'].max(), 100)
ax2.plot(x2, m2 * x2 + b2, 'k-', lw=2)
ax2.set_xlabel('AR Activity Score', fontsize=10, fontweight='bold')
ax2.set_ylabel('m6A Functional Impact', fontsize=10, fontweight='bold')
ax2.set_title(f'mCRPC: ARS vs FI\nρ={r_fi_s:+.3f}, p={p_fi_s:.2e} {sig(p_fi_s)}',
              fontsize=11, fontweight='bold')

ax3 = fig.add_subplot(2, 3, 3)
r_t2, p_t2 = spearmanr(meta_tc_gl.loc[shared_gl, 'AR_Activity_Score'],
                         meta_tc_gl.loc[shared_gl, 'm6A_Functional_Impact'])
for gl, c in zip(gl_order, gl_colors):
    idx_gl = meta_tc_gl[meta_tc_gl['Gleason_group'] == gl].index
    ax3.scatter(meta_tc_gl.loc[idx_gl, 'AR_Activity_Score'],
                meta_tc_gl.loc[idx_gl, 'm6A_Functional_Impact'],
                c=c, alpha=0.5, s=15, edgecolors='none', label=gl)
x3 = meta_tc_gl.loc[shared_gl, 'AR_Activity_Score'].values
y3 = meta_tc_gl.loc[shared_gl, 'm6A_Functional_Impact'].values
m3, b3 = np.polyfit(x3, y3, 1)
ax3.plot(np.linspace(x3.min(), x3.max(), 100),
         m3 * np.linspace(x3.min(), x3.max(), 100) + b3, 'k-', lw=2)
ax3.set_xlabel('AR Activity Score', fontsize=10, fontweight='bold')
ax3.set_ylabel('m6A Functional Impact', fontsize=10, fontweight='bold')
ax3.set_title(f'TCGA Replication\nρ={r_t2:+.3f}, p={p_t2:.2e} {sig(p_t2)}',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')

# Bottom row: per-gene bar, KM, Cox summary
ax4 = fig.add_subplot(2, 3, 4)
ax4.bar(range(len(gac)), gac['rho'], color=colors_bar, edgecolor='black', alpha=0.85)
ax4.axhline(0, color='black', lw=0.8)
ax4.set_xticks(range(len(gac)))
ax4.set_xticklabels(gac['Gene'], fontsize=7, rotation=45, ha='right')
ax4.set_ylabel('Spearman ρ with ARS', fontsize=10, fontweight='bold')
ax4.set_title('Per-gene Correlation with AR Activity', fontsize=11, fontweight='bold')

ax5 = fig.add_subplot(2, 3, 5)
kmf = KaplanMeierFitter()
for quad, gr in surv_df.groupby('Quadrant'):
    if len(gr) < 5:
        continue
    kmf.fit(gr['surv_months'], gr['vital_status'], label=quad)
    kmf.plot_survival_function(ax=ax5, color=quad_colors[quad], linewidth=1.5)
ax5.set_xlabel('Months', fontsize=10, fontweight='bold')
ax5.set_ylabel('Survival', fontsize=10, fontweight='bold')
ax5.set_title('AR × m6A Survival Quadrants', fontsize=11, fontweight='bold')
ax5.legend(fontsize=7, loc='lower left')
ax5.set_ylim(0, 1.05)

ax6 = fig.add_subplot(2, 3, 6)
cox_sum = cph.summary
labels_f = ['AR Activity', 'm6A FI', 'AR × m6A\nInteraction']
hrs = np.exp(cox_sum['coef'].values)
lo_ci = np.exp(cox_sum['coef lower 95%'].values)
hi_ci = np.exp(cox_sum['coef upper 95%'].values)
colors_c = ['#e74c3c' if h > 1 else '#3498db' for h in hrs]
for i in range(len(labels_f)):
    ax6.plot([lo_ci[i], hi_ci[i]], [i, i], '-', color=colors_c[i], lw=3, alpha=0.7)
    ax6.plot(hrs[i], i, 'o', color=colors_c[i], markersize=8)
    p_v = cox_sum['p'].values[i]
    ax6.text(hi_ci[i] + 0.05, i, f'{sig(p_v)}', va='center', fontsize=10, fontweight='bold')
ax6.axvline(1.0, color='black', lw=1, ls='--')
ax6.set_yticks(range(len(labels_f)))
ax6.set_yticklabels(labels_f, fontsize=10)
ax6.set_xlabel('Hazard Ratio (95% CI)', fontsize=10, fontweight='bold')
ax6.set_title('Cox PH Model', fontsize=11, fontweight='bold')

plt.suptitle('AR-m6A Coordination in Prostate Cancer — Summary',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '25_AR_m6A_summary_panel.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 25_AR_m6A_summary_panel.png")
plt.close()

# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)

print(f"\n  AR Activity Score (10-gene signature):")
print(f"    mCRPC Adeno vs SCNC:   ", end='')
r_as_ars = two_group_result = mannwhitneyu(
    meta.loc[idx_adeno, 'AR_Activity_Score'].dropna().values,
    meta.loc[idx_scnc, 'AR_Activity_Score'].dropna().values,
    alternative='two-sided')
print(f"p={r_as_ars.pvalue:.2e} {sig(r_as_ars.pvalue)}")
print(f"\n  Key m6A-AR Correlations (mCRPC):")
top3 = gac.nlargest(3, 'rho')[['Gene', 'Role', 'rho', 'p', 'sig']]
bot3 = gac.nsmallest(3, 'rho')[['Gene', 'Role', 'rho', 'p', 'sig']]
for _, row in pd.concat([top3, bot3]).iterrows():
    print(f"    {row['Gene']:12s} ({row['Role']:20s}): ρ={row['rho']:+.3f}  "
          f"p={row['p']:.2e} {row['sig']}")
print(f"\n  mCRPC ARS vs m6A FI: ρ={r_fi_s:+.3f}, p={p_fi_s:.2e} {sig(p_fi_s)}")
print(f"  TCGA  ARS vs m6A FI: ρ={r_tc:+.3f}, p={p_tc:.2e} {sig(p_tc)}")
print(f"\n  ARS KM (mCRPC):  log-rank p={lr_ars.p_value:.4f} {sig(lr_ars.p_value)}")
if len(g_best) >= 5 and len(g_worst) >= 5:
    print(f"  AR-hi/FI-hi vs AR-lo/FI-lo:  p={lr.p_value:.4f} {sig(lr.p_value)}")

# ===========================================================================
# PART VIII — AR RECEPTOR & BYPASS MECHANISMS
# ===========================================================================
print("\n" + "=" * 80)
print("  PART VIII — AR RECEPTOR & BYPASS MECHANISMS")
print("=" * 80)

# Steroid receptors available in bulk RNA-seq
STEROID_RECEPTORS = {
    'AR':     'Androgen Receptor',
    'NR3C1':  'Glucocorticoid R (GR)',   # primary AR bypass in CRPC
    'PGR':    'Progesterone R (PR)',
    'ESR1':   'Estrogen R-α (ERα)',
    'ESR2':   'Estrogen R-β (ERβ)',
}
receptors_avail = [r for r in STEROID_RECEPTORS if r in df.columns]
z_rec = zscore_normalize(df[receptors_avail])
print(f"\n  Receptors available in mCRPC: {receptors_avail}")

# --- 27. NR3C1 (GR): the primary AR bypass receptor -------------------------
print("\n--- Plot 27: NR3C1 (GR) — AR bypass receptor ---")
if 'NR3C1' in z_rec.columns:
    fig, axes_gr = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: NR3C1 by histology — GR is activated when AR is lost (SCNC, CRPC)
    ax = axes_gr[0]
    v_gr_adeno = z_rec.loc[idx_adeno, 'NR3C1'].dropna().values
    v_gr_scnc  = z_rec.loc[idx_scnc,  'NR3C1'].dropna().values
    _, p_gr_hist = mannwhitneyu(v_gr_adeno, v_gr_scnc, alternative='two-sided')
    parts = ax.violinplot([v_gr_adeno, v_gr_scnc], positions=[0, 1],
                          showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#27ae60'); parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor('#8e44ad'); parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_max = max(v_gr_adeno.max(), v_gr_scnc.max())
    ax.plot([0, 1], [y_max + 0.15, y_max + 0.15], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.18, f'p={p_gr_hist:.2e} {sig(p_gr_hist)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Adenocarcinoma\n(n={len(v_gr_adeno)})',
                        f'SCNC\n(n={len(v_gr_scnc)})'],
                       fontsize=11, fontweight='bold')
    ax.set_ylabel('NR3C1 (GR) mRNA z-score', fontsize=11, fontweight='bold')
    ax.set_title('GR (NR3C1) by Histology\n(GR → AR bypass in treatment-resistant PCa)',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    print(f"  NR3C1: Adeno={v_gr_adeno.mean():+.3f} vs SCNC={v_gr_scnc.mean():+.3f}  "
          f"p={p_gr_hist:.2e} {sig(p_gr_hist)}")

    # Panel B: AR vs NR3C1 — the lineage switch
    ax = axes_gr[1]
    both_avail = z_rec.dropna(subset=['AR', 'NR3C1']).index.intersection(df.index)
    c_hist = meta.loc[both_avail, 'histology'].map(
        {'Adenocarcinoma': '#27ae60', 'SCNC': '#8e44ad'}).fillna('grey')
    r_ar_gr, p_ar_gr = spearmanr(z_rec.loc[both_avail, 'AR'],
                                   z_rec.loc[both_avail, 'NR3C1'])
    ax.scatter(z_rec.loc[both_avail, 'AR'], z_rec.loc[both_avail, 'NR3C1'],
               c=c_hist.values, alpha=0.45, s=22, edgecolors='none')
    ax.set_xlabel('AR mRNA z-score', fontsize=11, fontweight='bold')
    ax.set_ylabel('NR3C1 (GR) mRNA z-score', fontsize=11, fontweight='bold')
    ax.set_title(f'AR vs GR Lineage Switch\nSpearman ρ={r_ar_gr:+.3f}, p={p_ar_gr:.2e} {sig(p_ar_gr)}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.legend(handles=[Patch(facecolor='#27ae60', label='Adenocarcinoma'),
                        Patch(facecolor='#8e44ad', label='SCNC'),
                        Patch(facecolor='grey',    label='Other')], fontsize=9)
    print(f"  AR vs NR3C1: ρ={r_ar_gr:+.3f}, p={p_ar_gr:.2e} {sig(p_ar_gr)}")

    # Panel C: NR3C1 vs m6A Functional Impact
    ax = axes_gr[2]
    shared_gr = meta.loc[df.index, 'm6A_Functional_Impact'].dropna().index.intersection(
        z_rec['NR3C1'].dropna().index)
    r_gr_fi, p_gr_fi = spearmanr(z_rec.loc[shared_gr, 'NR3C1'],
                                   meta.loc[shared_gr, 'm6A_Functional_Impact'])
    ax.scatter(z_rec.loc[shared_gr, 'NR3C1'],
               meta.loc[shared_gr, 'm6A_Functional_Impact'],
               c=meta.loc[shared_gr, 'histology'].map(
                   {'Adenocarcinoma': '#27ae60', 'SCNC': '#8e44ad'}).fillna('grey').values,
               alpha=0.4, s=20, edgecolors='none')
    m_gf, b_gf = np.polyfit(z_rec.loc[shared_gr, 'NR3C1'].values,
                              meta.loc[shared_gr, 'm6A_Functional_Impact'].values, 1)
    x_gf = np.linspace(z_rec.loc[shared_gr, 'NR3C1'].min(),
                        z_rec.loc[shared_gr, 'NR3C1'].max(), 100)
    ax.plot(x_gf, m_gf * x_gf + b_gf, 'k-', lw=2, alpha=0.7)
    ax.set_xlabel('NR3C1 (GR) mRNA z-score', fontsize=11, fontweight='bold')
    ax.set_ylabel('m6A Functional Impact', fontsize=11, fontweight='bold')
    ax.set_title(f'GR (NR3C1) vs m6A Functional Impact\nρ={r_gr_fi:+.3f}, p={p_gr_fi:.2e} {sig(p_gr_fi)}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    print(f"  NR3C1 vs m6A FI: ρ={r_gr_fi:+.3f}, p={p_gr_fi:.2e} {sig(p_gr_fi)}")

    plt.suptitle('GR (NR3C1) — AR Bypass Receptor: Histology, AR Switch & m6A Link (mCRPC)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '27_NR3C1_GR_bypass.png'), dpi=300, bbox_inches='tight')
    print("  → Saved: 27_NR3C1_GR_bypass.png")
    plt.close()
else:
    print("  NR3C1 not in dataset — skipping plot 27")

# ===========================================================================
# PART IX — CROSS-COHORT AR ACTIVITY TRAJECTORY
# ===========================================================================
# The cross-cohort trajectory analysis has been separated into its own script:
#     ar_crosscohort_analysis.py
#
# Rationale for separation:
#   1. Requires within-sample percentile-rank normalization (batch-invariant)
#      rather than the z-score normalization used throughout this file.
#   2. Between-cohort differences are confounded by tumor purity/cell composition
#      (whole tissue vs biopsy-enriched tumour), handled separately there.
#   3. The scientific question is different: how do AR and m6A co-evolve across
#      the full disease spectrum, vs the mechanistic coupling studied here.
#
# ar_crosscohort_analysis.py outputs: plots_ar_crosscohort/ (plots 01-10)
# ===========================================================================

# ===========================================================================
# PART X — CONFOUND-CONTROLLED AR × m6A ANALYSIS
# ===========================================================================
# Three sequential deconfounding steps:
#   1. Restrict to Adenocarcinoma only (remove SCNC as a confounder)
#   2. Partial Spearman correlation controlling for Luminal/Basal cluster
#   3. Bootstrapped mediation: ARS → RBM15B → m6A Functional Impact
# ===========================================================================
print("\n" + "=" * 80)
print("  PART X — CONFOUND-CONTROLLED AR × m6A ANALYSIS")
print("=" * 80)

# ── Adenocarcinoma-only subsets ───────────────────────────────────────────────
idx_adeno_surv = idx_adeno.intersection(
    meta[['AR_Activity_Score', 'm6A_Functional_Impact',
          'm6A_Net_Deposition', 'm6A_Oncogenic_Readout']].dropna().index
)
meta_adeno  = meta.loc[idx_adeno_surv]
z_adeno     = z_all.loc[idx_adeno_surv]
ars_adeno_s = meta_adeno['AR_Activity_Score']

print(f"\n  Adenocarcinoma-only subset: n={len(idx_adeno_surv)}")
print(f"  (removed {len(df.index) - len(idx_adeno_surv)} non-Adeno / missing-data samples)")


# ── Helper: partial Spearman ρ (residual method) ──────────────────────────────
from scipy.stats import rankdata as _rankdata
from scipy.stats import pearsonr as _pearsonr

def partial_spearman(x, y, z_covar):
    """
    Partial Spearman ρ of x and y controlling for z_covar.
    Method: rank x and y, regress out z_covar from both rank vectors
    via OLS, then Pearson-correlate the residuals.
    Assumes all three are aligned Series or arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z_covar, dtype=float)
    # Remove any row with NaN
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    # Rank
    rx = _rankdata(x, method='average') / len(x)
    ry = _rankdata(y, method='average') / len(y)
    # Regress z out of rx and ry (via OLS, z as the single predictor)
    Z = np.column_stack([np.ones(len(z)), z])
    def _resid(v):
        coef, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ coef
    resid_x = _resid(rx)
    resid_y = _resid(ry)
    r, p = _pearsonr(resid_x, resid_y)
    return r, p, int(mask.sum())


# ── Luminal/Basal numeric covariate ──────────────────────────────────────────
lb_num = meta_adeno['Luminal/Basal Cluster'].map({'Luminal': 1, 'Basal': 0})
# Only keep samples with known L/B assignment
idx_lb = lb_num.dropna().index
meta_lb   = meta_adeno.loc[idx_lb]
z_lb      = z_adeno.loc[idx_lb]
ars_lb    = ars_adeno_s.loc[idx_lb]
lb_covar  = lb_num.loc[idx_lb].values
print(f"  Adeno with Luminal/Basal assigned: n={len(idx_lb)} "
      f"(Lum={lb_covar.sum():.0f}, Bas={(lb_covar==0).sum():.0f})")


# --- 33. Per-gene AR correlations: full vs Adeno-only vs partial -------------
print("\n--- Plot 30: Per-gene AR corr — full / Adeno-only / partial ---")
rows_partcorr = []
for gene in gene_order:
    # Full cohort
    r_full, p_full = spearmanr(z_all.loc[ars.index, gene], ars)
    # Adeno-only
    r_adeno, p_adeno = spearmanr(z_adeno[gene], ars_adeno_s)
    # Partial (Adeno + Luminal/Basal covariate)
    r_part, p_part, n_part = partial_spearman(
        z_lb[gene].values, ars_lb.values, lb_covar
    )
    rows_partcorr.append({
        'Gene': gene, 'Role': gene_roles[gene],
        'rho_full': r_full, 'p_full': p_full,
        'rho_adeno': r_adeno, 'p_adeno': p_adeno,
        'rho_partial': r_part, 'p_partial': p_part,
    })

pc_df = pd.DataFrame(rows_partcorr)
print(f"\n  {'Gene':10s} {'ρ full':>8s}{'':3s} {'ρ Adeno':>8s}{'':3s} "
      f"{'ρ partial':>9s}{'':3s}  (partial = Adeno + L/B covariate)")
print("  " + "-" * 70)
for _, r in pc_df.iterrows():
    print(f"  {r['Gene']:10s} "
          f"{r['rho_full']:+8.3f}{sig(r['p_full']):3s} "
          f"{r['rho_adeno']:+8.3f}{sig(r['p_adeno']):3s} "
          f"{r['rho_partial']:+9.3f}{sig(r['p_partial']):3s}")

# Plot: grouped barplot with three bars per gene
fig, ax = plt.subplots(figsize=(15, 6))
n_genes = len(pc_df)
x       = np.arange(n_genes)
w       = 0.26
bars = [
    ax.bar(x - w, pc_df['rho_full'],    w, label='Full cohort',
           color='#95a5a6', edgecolor='black', alpha=0.9),
    ax.bar(x,     pc_df['rho_adeno'],   w, label='Adeno only',
           color='#3498db', edgecolor='black', alpha=0.9),
    ax.bar(x + w, pc_df['rho_partial'], w, label='Adeno + partial (L/B)',
           color='#e74c3c', edgecolor='black', alpha=0.9),
]
# Significance markers on partial bars
for i, r in pc_df.iterrows():
    if sig(r['p_partial']) != 'ns':
        y_top = r['rho_partial'] + (0.025 if r['rho_partial'] >= 0 else -0.04)
        ax.text(i + w, y_top, sig(r['p_partial']),
                ha='center', va='bottom' if r['rho_partial'] >= 0 else 'top',
                fontsize=8, fontweight='bold', color='#c0392b')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{r['Gene']}\n({r['Role'].split('(')[1].rstrip(')') if '(' in r['Role'] else r['Role']})"
     for _, r in pc_df.iterrows()],
    fontsize=8, rotation=30, ha='right',
)
ax.set_ylabel('Spearman ρ with AR Activity Score', fontsize=12, fontweight='bold')
ax.set_title(
    'Per-gene m6A × AR Correlation: Full Cohort vs Adeno-only vs Partial (Luminal/Basal controlled)\n'
    'Red bars = after removing both SCNC confound and Luminal/Basal lineage effect',
    fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '30_per_gene_AR_corr_deconfounded.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 30_per_gene_AR_corr_deconfounded.png")
plt.close()


# --- 36. RBM15B within Luminal and Basal separately --------------------------
print("\n--- Plot 33: RBM15B vs ARS within Luminal / Basal separately ---")
fig, axes36 = plt.subplots(1, 2, figsize=(14, 6))

for ax, (subset_name, subset_idx, color) in zip(axes36, [
    ('Luminal', idx_lum.intersection(idx_adeno), '#2980b9'),
    ('Basal',   idx_bas.intersection(idx_adeno), '#c0392b'),
]):
    x_s = ars.loc[subset_idx].dropna()
    y_s = z_all.loc[x_s.index, 'RBM15B']
    r_s, p_s = spearmanr(x_s, y_s)
    ax.scatter(x_s, y_s, c=color, alpha=0.45, s=22, edgecolors='none')
    m_s, b_s = np.polyfit(x_s.values, y_s.values, 1)
    x_line = np.linspace(x_s.min(), x_s.max(), 100)
    ax.plot(x_line, m_s * x_line + b_s, 'k-', lw=2)
    ax.set_xlabel('AR Activity Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('RBM15B z-score', fontsize=11, fontweight='bold')
    ax.set_title(f'RBM15B vs ARS — Adeno/{subset_name} (n={len(x_s)})\n'
                 f'ρ={r_s:+.3f}, p={p_s:.2e} {sig(p_s)}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    print(f"  RBM15B vs ARS in Adeno/{subset_name}: ρ={r_s:+.3f}, p={p_s:.2e} {sig(p_s)}")

plt.suptitle('RBM15B × AR Activity within Luminal and Basal Adenocarcinoma\n'
             'Tests whether the RBM15B signal is lineage-dependent',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '33_RBM15B_ARS_by_lineage.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 33_RBM15B_ARS_by_lineage.png")
plt.close()


# --- 37-38. Mediation: ARS → RBM15B → m6A Functional Impact -----------------
print("\n--- Plots 34-35: Mediation ARS → RBM15B → m6A FI (bootstrap) ---")
# Restrict to Adeno-only, complete cases
med_cols = ['AR_Activity_Score', 'm6A_Functional_Impact']
med_idx  = meta_adeno[med_cols].dropna().index.intersection(z_adeno.index)
X_med  = meta_adeno.loc[med_idx, 'AR_Activity_Score'].values   # Independent
M_med  = z_adeno.loc[med_idx, 'RBM15B'].values                 # Mediator
Y_med  = meta_adeno.loc[med_idx, 'm6A_Functional_Impact'].values  # Outcome
n_med  = len(X_med)

# Standardise for comparable coefficients
def _std(v): return (v - v.mean()) / v.std()
X_s = _std(X_med)
M_s = _std(M_med)
Y_s = _std(Y_med)

from scipy.stats import t as _t_dist
from sklearn.linear_model import LinearRegression as _LinReg

def _ols(x_mat, y_vec):
    """Return (coef_vector, se_vector, p_vector) for OLS."""
    X_ = np.column_stack([np.ones(len(y_vec)), x_mat])
    coef_, *_ = np.linalg.lstsq(X_, y_vec, rcond=None)
    resid = y_vec - X_ @ coef_
    mse   = (resid**2).sum() / (len(y_vec) - X_.shape[1])
    cov   = mse * np.linalg.inv(X_.T @ X_)
    se    = np.sqrt(np.diag(cov))
    t_stat = coef_ / se
    p_vals = 2 * _t_dist.sf(np.abs(t_stat), df=len(y_vec) - X_.shape[1])
    return coef_[1:], se[1:], p_vals[1:]  # skip intercept

# Path a: X → M
a_coef, a_se, a_p = _ols(X_s.reshape(-1,1), M_s)
# Path b: M → Y (controlling X)
b_coef, b_se, b_p = _ols(np.column_stack([M_s, X_s]), Y_s)
# Total effect c: X → Y
c_coef, c_se, c_p = _ols(X_s.reshape(-1,1), Y_s)
# Direct effect c': X → Y controlling M
cp_coef = b_coef[1:2]  # coefficient of X in the M+X → Y regression
cp_coef, cp_se, cp_p = _ols(np.column_stack([X_s, M_s]), Y_s)
cp_coef, cp_se, cp_p = cp_coef[:1], cp_se[:1], cp_p[:1]

a = float(a_coef[0])
b = float(b_coef[0])
c = float(c_coef[0])
cp_val = float(cp_coef[0])
indirect = a * b
print(f"  Path a (ARS → RBM15B):              β={a:+.4f}, p={a_p[0]:.3e} {sig(a_p[0])}")
print(f"  Path b (RBM15B → FI | ARS):         β={b:+.4f}, p={b_p[0]:.3e} {sig(b_p[0])}")
print(f"  Total effect c (ARS → FI):          β={c:+.4f}, p={c_p[0]:.3e} {sig(c_p[0])}")
print(f"  Direct effect c' (ARS → FI | RBM15B): β={cp_val:+.4f}, p={cp_p[0]:.3e} {sig(cp_p[0])}")
print(f"  Indirect effect a*b:                β={indirect:+.4f}")
print(f"  Proportion mediated: {abs(indirect/c)*100:.1f}%" if abs(c) > 1e-9 else "")

# Bootstrap 95% CI for indirect effect
rng = np.random.default_rng(2025)
N_BOOT = 5000
boot_indirect = []
for _ in range(N_BOOT):
    idx_b   = rng.integers(0, n_med, n_med)
    Xb, Mb, Yb = X_s[idx_b], M_s[idx_b], Y_s[idx_b]
    a_b, *_ = _ols(Xb.reshape(-1,1), Mb)
    b_b, *_ = _ols(np.column_stack([Mb, Xb]), Yb)
    boot_indirect.append(float(a_b[0]) * float(b_b[0]))
boot_indirect = np.array(boot_indirect)
ci_lo, ci_hi = np.percentile(boot_indirect, [2.5, 97.5])
p_boot = min(np.mean(boot_indirect <= 0), np.mean(boot_indirect >= 0)) * 2
print(f"  Bootstrap 95% CI for a*b: [{ci_lo:+.4f}, {ci_hi:+.4f}]"
      f"  (zero excluded: {int(ci_lo > 0 or ci_hi < 0)})")
print(f"  Bootstrap p (two-tailed): {p_boot:.4f} {sig(p_boot)}")

# --- Plot 37: Path diagram ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

# Boxes
for (x0, y0, w, h, lbl, clr) in [
    (0.3, 2.2, 2.2, 1.4, 'AR Activity\nScore (X)', '#3498db'),
    (3.9, 4.2, 2.2, 1.4, 'RBM15B z-score\n(Mediator M)', '#e67e22'),
    (7.5, 2.2, 2.2, 1.4, 'm6A Functional\nImpact (Y)', '#e74c3c'),
]:
    rect = plt.Rectangle((x0, y0), w, h, linewidth=2, edgecolor=clr,
                          facecolor=clr, alpha=0.18, zorder=2)
    ax.add_patch(rect)
    ax.text(x0 + w/2, y0 + h/2, lbl, ha='center', va='center',
            fontsize=11, fontweight='bold', color=clr, zorder=3)

# Arrows — scale positions
arrow_kw = dict(arrowstyle='->', color='black', lw=2,
                connectionstyle='arc3,rad=0')
from matplotlib.patches import FancyArrowPatch
# X → M (path a)
ax.annotate('', xy=(3.9, 5.0), xytext=(2.5, 3.6),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5,
                            connectionstyle='arc3,rad=-0.2'))
ax.text(2.9, 4.8, f'a={a:+.3f}{sig(a_p[0])}', fontsize=10,
        color='#e67e22', fontweight='bold', ha='center')
# M → Y (path b)
ax.annotate('', xy=(7.5, 5.0), xytext=(6.1, 5.0),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5))
ax.text(6.8, 5.25, f'b={b:+.3f}{sig(b_p[0])}', fontsize=10,
        color='#e67e22', fontweight='bold', ha='center')
# X → Y (direct c')
ax.annotate('', xy=(7.5, 2.9), xytext=(2.5, 2.9),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
ax.text(5.0, 2.55, f"c'={cp_val:+.3f}{sig(cp_p[0])}", fontsize=10,
        color='#3498db', fontweight='bold', ha='center')
# Indirect label
ci_str = f'[{ci_lo:+.3f}, {ci_hi:+.3f}]'
zero_out = 'zero excluded' if (ci_lo > 0 or ci_hi < 0) else 'zero NOT excluded'
ax.text(5.0, 1.5,
        f'Indirect effect a×b = {indirect:+.4f}\n'
        f'Bootstrap 95% CI: {ci_str}\n'
        f'{zero_out}  (p={p_boot:.4f} {sig(p_boot)})',
        ha='center', va='center', fontsize=11, fontweight='bold',
        bbox=dict(facecolor='lightyellow', edgecolor='goldenrod', alpha=0.9, pad=6))
ax.set_title('Mediation Analysis: ARS → RBM15B → m6A Functional Impact\n'
             '(Adenocarcinoma only, standardised coefficients)',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '34_mediation_ARS_RBM15B_FI.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 34_mediation_ARS_RBM15B_FI.png")
plt.close()


print(f"\n  Part X summary:")
print(f"  Indirect effect a*b = {indirect:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
prop_med = abs(indirect/c)*100 if abs(c) > 1e-9 else float('nan')
print(f"  Proportion of total effect mediated: {prop_med:.1f}%")





print("\n\n" + "=" * 80)
print("  ALL PLOTS SAVED")
print("=" * 80)
for i, f in enumerate([
    '01_AR_target_coexpression.png',
    '02_ARS_by_histology_lumbasline.png',
    '03_ARS_by_AR_alteration.png',
    '05_m6A_gene_ARS_correlation.png',
    '11_ARS_FI_landscape.png',
    '16_KM_ARxFI_quadrant.png',
    '19_Cox_HR_forest.png',
    '25_AR_m6A_summary_panel.png',
    '27_NR3C1_GR_bypass.png',
    '30_per_gene_AR_corr_deconfounded.png',
    '33_RBM15B_ARS_by_lineage.png',
    '34_mediation_ARS_RBM15B_FI.png',
], 1):
    print(f"  {i:2d}. {f}")
print(f"\n  Output directory: {OUTDIR}")
print("=" * 80)
