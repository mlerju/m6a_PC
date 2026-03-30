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
  Part IX   — Cross-cohort AR activity trajectory (30–32)

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
all_genes = list(set(ALL_M6A_GENES + AR_TARGET_GENES + ['AR']))
all_genes = [g for g in all_genes if g in df.columns]
z_all = zscore_normalize(df[all_genes])

# ── AR Activity Score (ARS) ───────────────────────────────────────────────────
ar_avail = [g for g in AR_TARGET_GENES if g in z_all.columns]
meta['AR_Activity_Score'] = z_all[ar_avail].mean(axis=1)
print(f"\n  AR signature genes available in mCRPC: {len(ar_avail)}/{len(AR_TARGET_GENES)}")
print(f"  ARS (mean ± SD): {meta['AR_Activity_Score'].mean():.3f} ± {meta['AR_Activity_Score'].std():.3f}")

# ── TCGA z-score normalise ────────────────────────────────────────────────────
tc_genes    = list(set(ALL_M6A_GENES + AR_TARGET_GENES + ['AR']))
tc_genes    = [g for g in tc_genes if g in df_tc.columns]
z_tc        = zscore_normalize(df_tc[tc_genes])
ar_tc       = [g for g in AR_TARGET_GENES if g in z_tc.columns]
meta_tc     = meta_tc.copy()
meta_tc['AR_Activity_Score'] = z_tc[ar_tc].mean(axis=1)

# ── TCGA m6A axes (simplified: mean z-scores) ────────────────────────────────
writer_tc = [g for g in WRITER_GENES if g in z_tc.columns]
eraser_tc = [g for g in ERASER_GENES if g in z_tc.columns]
onco_tc   = [g for g in READER_ONCOGENIC   if g in z_tc.columns]
supp_tc   = [g for g in READER_SUPPRESSIVE if g in z_tc.columns]
meta_tc['m6A_Net_Deposition']    = z_tc[writer_tc].mean(axis=1) - z_tc[eraser_tc].mean(axis=1)
meta_tc['m6A_Oncogenic_Readout'] = z_tc[onco_tc].mean(axis=1)  - z_tc[supp_tc].mean(axis=1)
meta_tc['m6A_Functional_Impact'] = (meta_tc['m6A_Net_Deposition'] * 0.435 +
                                     meta_tc['m6A_Oncogenic_Readout'] * 0.565)

# ── mCRPC m6A axes (using same LR-derived weights from mcrpc_analysis) ───────
# Re-derive from data quickly (same as mcrpc_analysis.py Part I)
from sklearn.linear_model import LogisticRegression as _LR
idx_hist = meta[meta['histology'].isin(['Adenocarcinoma','SCNC'])].index.intersection(df.index)
X_wr     = z_all.loc[idx_hist, WRITER_GENES].values
y_hs     = (meta.loc[idx_hist, 'histology'] == 'SCNC').astype(int).values
lr_m     = _LR(penalty='l2', C=1.0, max_iter=1000, random_state=42).fit(X_wr, y_hs)
lr_abs   = np.abs(lr_m.coef_[0]);  lr_wts = lr_abs / lr_abs.sum() * sum(MANUAL_WEIGHTS.values())
LR_WEIGHTS = {g: w for g, w in zip(WRITER_GENES, lr_wts)}
dd_w = sum(z_all[g] * w for g, w in LR_WEIGHTS.items()) / sum(LR_WEIGHTS.values())
meta['m6A_Net_Deposition']    = dd_w - z_all[ERASER_GENES].mean(axis=1)
meta['m6A_Oncogenic_Readout'] = z_all[READER_ONCOGENIC].mean(axis=1) - z_all[READER_SUPPRESSIVE].mean(axis=1)
meta['m6A_Functional_Impact'] = meta['m6A_Net_Deposition'] * 0.435 + meta['m6A_Oncogenic_Readout'] * 0.565

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

# --- 04. ARS by AR clinical context (ASI, PSA response) ----------------------
print("\n--- Plot 04: ARS by clinical AR context ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: WCDT_ASI (on-treatment vs not)
ax = axes[0]
idx_asi_on  = meta[meta['WCDT_ASI'] == True].index.intersection(df.index)
idx_asi_off = meta[meta['WCDT_ASI'] == False].index.intersection(df.index)
va_on  = meta.loc[idx_asi_on,  'AR_Activity_Score'].dropna().values
va_off = meta.loc[idx_asi_off, 'AR_Activity_Score'].dropna().values
if len(va_on) > 5 and len(va_off) > 5:
    _, p_asi = mannwhitneyu(va_on, va_off, alternative='two-sided')
    parts = ax.violinplot([va_off, va_on], positions=[0, 1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#3498db'); parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor('#e74c3c'); parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_max = max(va_off.max(), va_on.max())
    ax.plot([0, 1], [y_max + 0.12, y_max + 0.12], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.15, f'p={p_asi:.2e} {sig(p_asi)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'ASI Off\n(n={len(va_off)})', f'ASI On\n(n={len(va_on)})'],
                       fontsize=11, fontweight='bold')
    ax.set_title('AR Activity Score\nby ASI Treatment Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('AR Activity Score', fontsize=11, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    print(f"  ASI Off={va_off.mean():+.3f} vs ASI On={va_on.mean():+.3f}  p={p_asi:.2e} {sig(p_asi)}")

# Panel B: Prior ASI therapy
ax = axes[1]
idx_prev = meta[meta['WCDT_Previous ASI Therapy'] == 'Previous ASI Therapy'].index.intersection(df.index)
idx_naive = meta[meta['WCDT_Previous ASI Therapy'] == 'ASI Naive'].index.intersection(df.index)
vp = meta.loc[idx_prev,  'AR_Activity_Score'].dropna().values
vn = meta.loc[idx_naive, 'AR_Activity_Score'].dropna().values
if len(vp) > 5 and len(vn) > 5:
    _, p_prev = mannwhitneyu(vp, vn, alternative='two-sided')
    parts = ax.violinplot([vn, vp], positions=[0, 1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#27ae60'); parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor('#e67e22'); parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_max = max(vn.max(), vp.max())
    ax.plot([0, 1], [y_max + 0.12, y_max + 0.12], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.15, f'p={p_prev:.2e} {sig(p_prev)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'ASI Naïve\n(n={len(vn)})', f'Prior ASI\n(n={len(vp)})'],
                       fontsize=11, fontweight='bold')
    ax.set_title('AR Activity Score\nby Prior ASI Exposure', fontsize=12, fontweight='bold')
    ax.set_ylabel('AR Activity Score', fontsize=11, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    print(f"  ASI Naïve={vn.mean():+.3f} vs Prior ASI={vp.mean():+.3f}  p={p_prev:.2e} {sig(p_prev)}")

plt.suptitle('AR Activity Score — Clinical Context Validation',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '04_ARS_clinical_context.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 04_ARS_clinical_context.png")
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

# --- 06. Top correlating gene scatters (RBM15, RBM15B, CBLL1) ---------------
print("\n--- Plot 06: Top m6A×AR gene scatters ---")
top_genes = gac.reindex(gac['rho'].abs().nlargest(3).index)['Gene'].tolist()
print(f"  Top 3 correlating genes: {top_genes}")

color_map_hist = meta.loc[df.index, 'histology'].map(
    {'Adenocarcinoma': '#27ae60', 'SCNC': '#8e44ad'}
).fillna('grey')

fig, axes = plt.subplots(1, len(top_genes), figsize=(6 * len(top_genes), 6))
if len(top_genes) == 1:
    axes = [axes]
for ax, gene in zip(axes, top_genes):
    shared_idx = ars.index.intersection(df.index)
    x = ars.loc[shared_idx].values
    y = z_all.loc[shared_idx, gene].values
    c = color_map_hist.loc[shared_idx].values
    r, p = spearmanr(x, y)
    ax.scatter(x, y, c=c, alpha=0.4, s=20, edgecolors='none')
    # regression line
    m_fit, b_fit = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m_fit * x_line + b_fit, 'k-', lw=2, alpha=0.7)
    ax.set_xlabel('AR Activity Score', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{gene} z-score', fontsize=11, fontweight='bold')
    ax.set_title(f'{gene} ({gene_roles[gene]})\nSpearman ρ={r:+.3f}, p={p:.2e} {sig(p)}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

legend_els = [Patch(facecolor='#27ae60', label='Adenocarcinoma'),
              Patch(facecolor='#8e44ad', label='SCNC'),
              Patch(facecolor='grey',    label='Other')]
axes[0].legend(handles=legend_els, fontsize=10)
plt.suptitle('Top m6A Genes Correlated with AR Activity (mCRPC)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '06_top_m6A_ARS_scatters.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 06_top_m6A_ARS_scatters.png")
plt.close()

# --- 07. m6A gene heatmap across ARS quartiles -------------------------------
print("\n--- Plot 07: m6A gene expression across ARS quartiles ---")
q_bins  = pd.qcut(meta.loc[df.index, 'AR_Activity_Score'], q=4,
                  labels=['Q1\n(AR-low)', 'Q2', 'Q3', 'Q4\n(AR-high)'])
heat_data_q = []
q_labels    = q_bins.cat.categories.tolist()
for gene in gene_order:
    row = {'Gene': gene}
    for ql in q_bins.cat.categories:
        idx_q = q_bins[q_bins == ql].index
        row[str(ql)] = z_all.loc[idx_q, gene].mean()
    heat_data_q.append(row)
hd = pd.DataFrame(heat_data_q).set_index('Gene')

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(hd, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean z-score'},
            yticklabels=[f"{g} ({gene_roles[g]})" for g in hd.index])
ax.set_title('m6A Gene Expression Across AR Activity Score Quartiles (mCRPC)\n'
             'Q1 = AR-low, Q4 = AR-high',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('AR Activity Score Quartile', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '07_m6A_gene_ARS_quartile_heatmap.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 07_m6A_gene_ARS_quartile_heatmap.png")
plt.close()

# ===========================================================================
# PART III — m6A AXES × AR CONTINUOUS SCORE
# ===========================================================================
print("\n" + "=" * 80)
print("  PART III — m6A AXES × AR CONTINUOUS SCORE")
print("=" * 80)

def axis_ar_scatter(ax_obj, axis_col, axis_label, color_by, color_map_s,
                    shared_idx, title_suffix=''):
    x  = meta.loc[shared_idx, 'AR_Activity_Score'].values
    y  = meta.loc[shared_idx, axis_col].values
    r, p = spearmanr(x, y)
    ax_obj.scatter(x, y, c=color_map_s.loc[shared_idx].values,
                   alpha=0.4, s=20, edgecolors='none')
    m_fit, b_fit = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax_obj.plot(x_line, m_fit * x_line + b_fit, 'k-', lw=2, alpha=0.7)
    ax_obj.set_xlabel('AR Activity Score', fontsize=11, fontweight='bold')
    ax_obj.set_ylabel(axis_label, fontsize=11, fontweight='bold')
    ax_obj.set_title(f'{axis_label}{title_suffix}\nSpearman ρ={r:+.3f}, p={p:.2e} {sig(p)}',
                     fontsize=12, fontweight='bold')
    ax_obj.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax_obj.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    return r, p

shared_idx = ars.index.intersection(df.index)
cmap_hist  = meta.loc[df.index, 'histology'].map(
    {'Adenocarcinoma': '#27ae60', 'SCNC': '#8e44ad'}).fillna('grey')
cmap_lb    = meta.loc[df.index, 'Luminal/Basal Cluster'].map(
    {'Luminal': '#2980b9', 'Basal': '#c0392b'}).fillna('grey')

# --- Plots 08-10: ARS vs each axis -------------------------------------------
print("\n--- Plots 08-10: ARS vs m6A axes scatters ---")
for pi, (ax_col, ax_label, fname) in enumerate([
    ('m6A_Net_Deposition',    'Net m6A Deposition',  '08_ARS_vs_NetDeposition.png'),
    ('m6A_Oncogenic_Readout', 'Oncogenic Readout',   '09_ARS_vs_OncogenicReadout.png'),
    ('m6A_Functional_Impact', 'Functional Impact',   '10_ARS_vs_FunctionalImpact.png'),
], 8):
    fig, axes_s = plt.subplots(1, 2, figsize=(14, 6))
    r_h, p_h = axis_ar_scatter(axes_s[0], ax_col, ax_label, 'histology', cmap_hist, shared_idx)
    r_l, p_l = axis_ar_scatter(axes_s[1], ax_col, ax_label, 'Luminal/Basal', cmap_lb, shared_idx)
    axes_s[0].legend(handles=[Patch(facecolor='#27ae60', label='Adenocarcinoma'),
                               Patch(facecolor='#8e44ad', label='SCNC'),
                               Patch(facecolor='grey',    label='Other')], fontsize=10)
    axes_s[1].legend(handles=[Patch(facecolor='#2980b9', label='Luminal'),
                               Patch(facecolor='#c0392b', label='Basal'),
                               Patch(facecolor='grey',    label='Other')], fontsize=10)
    plt.suptitle(f'AR Activity Score vs {ax_label}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300, bbox_inches='tight')
    print(f"  → Saved: {fname}  (hist: ρ={r_h:+.3f} {sig(p_h)}, LB: ρ={r_l:+.3f} {sig(p_l)})")
    plt.close()

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
# PART IV — AR TREATMENT CONTEXT: m6A AXES
# ===========================================================================
print("\n" + "=" * 80)
print("  PART IV — AR TREATMENT CONTEXT")
print("=" * 80)

def compare_m6a_axes_violin(idx_a, idx_b, label_a, label_b, title_prefix, filename,
                              colors=('#3498db', '#e74c3c')):
    """Three-panel violin: Net Dep, Onco Readout, Func Impact, for two groups."""
    axes_info = [
        ('m6A_Net_Deposition',    'Net m6A Deposition'),
        ('m6A_Oncogenic_Readout', 'Oncogenic Readout'),
        ('m6A_Functional_Impact', 'Functional Impact'),
    ]
    fig, plot_axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (col, label) in zip(plot_axes, axes_info):
        va = meta.loc[idx_a.intersection(df.index), col].dropna().values
        vb = meta.loc[idx_b.intersection(df.index), col].dropna().values
        if len(va) < 3 or len(vb) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue
        _, p = mannwhitneyu(va, vb, alternative='two-sided')
        parts = ax.violinplot([va, vb], positions=[0, 1], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(colors[0]); parts['bodies'][0].set_alpha(0.65)
        parts['bodies'][1].set_facecolor(colors[1]); parts['bodies'][1].set_alpha(0.65)
        style_violin(parts, ax)
        y_max = max(va.max(), vb.max())
        ax.plot([0, 1], [y_max + 0.12, y_max + 0.12], 'k-', lw=1.2)
        ax.text(0.5, y_max + 0.15, f'p={p:.2e} {sig(p)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'{label_a}\n(n={len(va)})', f'{label_b}\n(n={len(vb)})'],
                           fontsize=10, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_title(label, fontsize=12, fontweight='bold')
        print(f"  {col[:20]:20s}: {label_a}={va.mean():+.3f} vs {label_b}={vb.mean():+.3f}  "
              f"p={p:.2e} {sig(p)}")
    plt.suptitle(f'{title_prefix}: {label_a} vs {label_b}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300, bbox_inches='tight')
    print(f"  → Saved: {filename}")
    plt.close()

# --- 12. m6A axes: ASI on-treatment vs off -----------------------------------
print("\n--- Plot 12: m6A axes by ASI treatment ---")
compare_m6a_axes_violin(idx_asi_off, idx_asi_on, 'ASI Off', 'ASI On',
                         'm6A Axes by ASI Treatment Status',
                         '12_m6A_axes_ASI_treatment.png')

# --- 13. m6A axes: PSA responders vs non-responders -------------------------
print("\n--- Plot 13: m6A axes by PSA response ---")
idx_psa_resp    = meta[meta['WCDT - PSA 51-100% Response'] == True].index
idx_psa_nonresp = meta[meta['WCDT - PSA 51-100% Response'] == False].index
print(f"  PSA responders: n={len(idx_psa_resp)}, non-responders: n={len(idx_psa_nonresp)}")
compare_m6a_axes_violin(idx_psa_nonresp, idx_psa_resp, 'Non-responder', 'PSA Responder',
                         'm6A Axes by PSA Response to ASI',
                         '13_m6A_axes_PSA_response.png',
                         colors=('#95a5a6', '#27ae60'))

# --- 14. m6A axes: Prior ASI vs naïve ----------------------------------------
print("\n--- Plot 14: m6A axes by prior ASI therapy ---")
compare_m6a_axes_violin(idx_naive, idx_prev, 'ASI Naïve', 'Prior ASI',
                         'm6A Axes by Prior ASI Therapy',
                         '14_m6A_axes_prior_ASI.png',
                         colors=('#27ae60', '#e67e22'))

# --- 15. m6A axes: AR Amp/Mut vs WT ------------------------------------------
print("\n--- Plot 15: m6A axes by AR amplification/mutation ---")
compare_m6a_axes_violin(idx_no_amp, idx_amp, 'AR WT', 'AR Amp/Mut',
                         'm6A Axes by AR Genomic Alteration',
                         '15_m6A_axes_AR_alteration.png',
                         colors=('#95a5a6', '#e74c3c'))

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
fig, ax = plt.subplots(figsize=(10, 7))
kmf = KaplanMeierFitter()
kmf.fit(ars_hi['surv_months'], ars_hi['vital_status'],
        label=f'High ARS (n={len(ars_hi)})')
kmf.plot_survival_function(ax=ax, color='#e74c3c', linewidth=2)
kmf.fit(ars_lo['surv_months'], ars_lo['vital_status'],
        label=f'Low ARS (n={len(ars_lo)})')
kmf.plot_survival_function(ax=ax, color='#3498db', linewidth=2)
ax.set_xlabel('Time (months)', fontsize=13, fontweight='bold')
ax.set_ylabel('Survival Probability', fontsize=13, fontweight='bold')
ax.set_title(f'Overall Survival by AR Activity Score\n'
             f'Log-rank p={lr_ars.p_value:.4f} {sig(lr_ars.p_value)}',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=12, loc='lower left')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '17_KM_ARS.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 17_KM_ARS.png  (p={lr_ars.p_value:.4f} {sig(lr_ars.p_value)})")
plt.close()

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

# ===========================================================================
# PART VI — TCGA VALIDATION
# ===========================================================================
print("\n" + "=" * 80)
print("  PART VI — TCGA VALIDATION")
print("=" * 80)

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

# --- 20. ARS by Gleason score in TCGA ----------------------------------------
print("\n--- Plot 20: TCGA ARS by Gleason ---")
gl_groups_ars = [meta_tc_gl.loc[meta_tc_gl['Gleason_group'] == gl, 'AR_Activity_Score'].dropna().values
                 for gl in gl_order]
gl_n  = [len(g) for g in gl_groups_ars]
_, p_kw = kruskal(*gl_groups_ars)

fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot(gl_groups_ars, positions=range(3), showmeans=True, showmedians=True)
for i, c in enumerate(gl_colors):
    parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.65)
style_violin(parts, ax)
ax.set_xticks(range(3))
ax.set_xticklabels([f'{gl}\n(n={n})' for gl, n in zip(gl_order, gl_n)],
                   fontsize=11, fontweight='bold')
ax.set_ylabel('AR Activity Score', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity Score by Gleason Score (TCGA Localized PCa)\n'
             f'Kruskal-Wallis p={p_kw:.2e} {sig(p_kw)}',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.8, ls='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '20_TCGA_ARS_by_Gleason.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 20_TCGA_ARS_by_Gleason.png  (KW p={p_kw:.2e} {sig(p_kw)})")
plt.close()

# --- 21. m6A FI by Gleason score in TCGA ------------------------------------
print("\n--- Plot 21: TCGA m6A FI by Gleason ---")
gl_groups_fi = [meta_tc_gl.loc[meta_tc_gl['Gleason_group'] == gl, 'm6A_Functional_Impact'].dropna().values
                for gl in gl_order]
_, p_kw_fi = kruskal(*gl_groups_fi)

fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot(gl_groups_fi, positions=range(3), showmeans=True, showmedians=True)
for i, c in enumerate(gl_colors):
    parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.65)
style_violin(parts, ax)
ax.set_xticks(range(3))
ax.set_xticklabels([f'{gl}\n(n={n})' for gl, n in zip(gl_order, gl_n)],
                   fontsize=11, fontweight='bold')
ax.set_ylabel('m6A Functional Impact', fontsize=12, fontweight='bold')
ax.set_title(f'm6A Functional Impact by Gleason Score (TCGA Localized PCa)\n'
             f'Kruskal-Wallis p={p_kw_fi:.2e} {sig(p_kw_fi)}',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.8, ls='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '21_TCGA_FI_by_Gleason.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 21_TCGA_FI_by_Gleason.png  (KW p={p_kw_fi:.2e} {sig(p_kw_fi)})")
plt.close()

# --- 22. ARS vs FI scatter in TCGA (colored by Gleason) ----------------------
print("\n--- Plot 22: TCGA ARS vs FI scatter ---")
gl_c_map = meta_tc_gl['Gleason_group'].map(dict(zip(gl_order, gl_colors))).dropna()
shared_gl = gl_c_map.index
r_tc, p_tc = spearmanr(meta_tc_gl.loc[shared_gl, 'AR_Activity_Score'],
                        meta_tc_gl.loc[shared_gl, 'm6A_Functional_Impact'])
fig, ax = plt.subplots(figsize=(9, 7))
for gl, c in zip(gl_order, gl_colors):
    idx_gl = meta_tc_gl[meta_tc_gl['Gleason_group'] == gl].index
    ax.scatter(meta_tc_gl.loc[idx_gl, 'AR_Activity_Score'],
               meta_tc_gl.loc[idx_gl, 'm6A_Functional_Impact'],
               c=c, alpha=0.5, s=30, label=f'{gl} (n={len(idx_gl)})', edgecolors='none')
x_tc = meta_tc_gl.loc[shared_gl, 'AR_Activity_Score'].values
y_tc = meta_tc_gl.loc[shared_gl, 'm6A_Functional_Impact'].values
m_t, b_t = np.polyfit(x_tc, y_tc, 1)
x_l = np.linspace(x_tc.min(), x_tc.max(), 100)
ax.plot(x_l, m_t * x_l + b_t, 'k-', lw=2, alpha=0.7)
ax.set_xlabel('AR Activity Score', fontsize=12, fontweight='bold')
ax.set_ylabel('m6A Functional Impact', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity vs m6A Functional Impact (TCGA Localized PCa)\n'
             f'Spearman ρ={r_tc:+.3f}, p={p_tc:.2e} {sig(p_tc)}',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, loc='best')
ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '22_TCGA_ARS_vs_FI_Gleason.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 22_TCGA_ARS_vs_FI_Gleason.png  (ρ={r_tc:+.3f} {sig(p_tc)})")
plt.close()

# ===========================================================================
# PART VII — SITE-STRATIFIED AR ACTIVITY
# ===========================================================================
print("\n" + "=" * 80)
print("  PART VII — SITE-STRATIFIED AR ACTIVITY")
print("=" * 80)

site_patients = meta[meta['Site_detailed'].notna()].index.intersection(df.index)
site_data_ar  = {}
print(f"\n  {'Site':15s} {'n':>5s} {'ARS mean':>9s} {'SD':>7s}  {'NetDep':>8s} {'FuncImp':>9s}")
print("  " + "-" * 65)
for site in SITE_ORDER:
    idx_s = meta[meta['Site_detailed'] == site].index.intersection(df.index)
    if len(idx_s) == 0:
        continue
    site_data_ar[site] = idx_s
    ars_s = meta.loc[idx_s, 'AR_Activity_Score']
    fi_s  = meta.loc[idx_s, 'm6A_Functional_Impact']
    nd_s  = meta.loc[idx_s, 'm6A_Net_Deposition']
    print(f"  {site:15s} {len(idx_s):5d} {ars_s.mean():+9.3f} {ars_s.std():7.3f}  "
          f"{nd_s.mean():+8.3f} {fi_s.mean():+9.3f}")

groups_ars = [meta.loc[site_data_ar[s], 'AR_Activity_Score'].values
              for s in SITE_ORDER if s in site_data_ar]
_, p_s_kw = kruskal(*groups_ars)
print(f"\n  Kruskal-Wallis (ARS across sites): H p={p_s_kw:.2e} {sig(p_s_kw)}")

# --- 23. ARS by biopsy site --------------------------------------------------
print("\n--- Plot 23: ARS by biopsy site ---")
plot_sites = [s for s in SITE_ORDER if s in site_data_ar]
fig, ax = plt.subplots(figsize=(12, 7))
ars_groups = [meta.loc[site_data_ar[s], 'AR_Activity_Score'].values for s in plot_sites]
parts = ax.violinplot(ars_groups, positions=range(len(plot_sites)),
                      showmeans=True, showmedians=True)
for i, s in enumerate(plot_sites):
    parts['bodies'][i].set_facecolor(SITE_COLORS[s]); parts['bodies'][i].set_alpha(0.65)
style_violin(parts, ax)
ax.set_xticks(range(len(plot_sites)))
ax.set_xticklabels([f"{s}\n(n={len(site_data_ar[s])})" for s in plot_sites],
                   fontsize=10, fontweight='bold')
ax.set_ylabel('AR Activity Score', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity Score by Biopsy Site (mCRPC)\n'
             f'Kruskal-Wallis p={p_s_kw:.2e} {sig(p_s_kw)}',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.8, ls='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '23_ARS_by_site.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 23_ARS_by_site.png")
plt.close()

# --- 24. ARS vs FI scatter colored by site -----------------------------------
print("\n--- Plot 24: ARS vs FI scatter by site ---")
fig, ax = plt.subplots(figsize=(10, 8))
for s in SITE_ORDER:
    if s not in site_data_ar:
        continue
    idx_s = site_data_ar[s]
    ax.scatter(meta.loc[idx_s, 'AR_Activity_Score'],
               meta.loc[idx_s, 'm6A_Functional_Impact'],
               c=SITE_COLORS[s], alpha=0.5, s=30, edgecolors='none',
               label=f'{s} (n={len(idx_s)})')
r_site, p_site = spearmanr(
    meta.loc[site_patients, 'AR_Activity_Score'],
    meta.loc[site_patients, 'm6A_Functional_Impact']
)
ax.set_xlabel('AR Activity Score', fontsize=12, fontweight='bold')
ax.set_ylabel('m6A Functional Impact', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity × m6A Landscape by Biopsy Site\n'
             f'Spearman ρ={r_site:+.3f}, p={p_site:.2e} {sig(p_site)}',
             fontsize=13, fontweight='bold', pad=12)
med_ars_sp = meta.loc[site_patients, 'AR_Activity_Score'].median()
med_fi_sp  = meta.loc[site_patients, 'm6A_Functional_Impact'].median()
ax.axvline(med_ars_sp, color='grey', lw=1, ls='--', alpha=0.5)
ax.axhline(med_fi_sp,  color='grey', lw=1, ls='--', alpha=0.5)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '24_ARS_FI_scatter_by_site.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 24_ARS_FI_scatter_by_site.png")
plt.close()

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

# --- 26. AR mRNA level: histology violin + mRNA vs ARS scatter ---------------
print("\n--- Plot 26: AR mRNA level by histology & correlation with ARS ---")
fig, axes_ar = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: AR mRNA by histology
ax = axes_ar[0]
if 'AR' in z_rec.columns:
    v_adeno = z_rec.loc[idx_adeno, 'AR'].dropna().values
    v_scnc  = z_rec.loc[idx_scnc,  'AR'].dropna().values
    _, p_ar_hist = mannwhitneyu(v_adeno, v_scnc, alternative='two-sided')
    parts = ax.violinplot([v_adeno, v_scnc], positions=[0, 1],
                          showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#27ae60'); parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor('#8e44ad'); parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_max = max(v_adeno.max(), v_scnc.max())
    ax.plot([0, 1], [y_max + 0.15, y_max + 0.15], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.18, f'p={p_ar_hist:.2e} {sig(p_ar_hist)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Adenocarcinoma\n(n={len(v_adeno)})',
                        f'SCNC\n(n={len(v_scnc)})'],
                       fontsize=11, fontweight='bold')
    ax.set_ylabel('AR mRNA z-score', fontsize=12, fontweight='bold')
    ax.set_title('AR Receptor mRNA by Histology\n(SCNC = AR-indifferent lineage)',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    print(f"  AR mRNA: Adeno={v_adeno.mean():+.3f} vs SCNC={v_scnc.mean():+.3f}  "
          f"p={p_ar_hist:.2e} {sig(p_ar_hist)}")

# Panel B: AR mRNA vs ARS (auto-regulation)
ax = axes_ar[1]
if 'AR' in z_rec.columns:
    shared_ar = ars.index.intersection(df.index)
    r_ar_ars, p_ar_ars = spearmanr(z_rec.loc[shared_ar, 'AR'], ars.loc[shared_ar])
    ax.scatter(ars.loc[shared_ar], z_rec.loc[shared_ar, 'AR'],
               c=cmap_hist.loc[shared_ar].values, alpha=0.4, s=20, edgecolors='none')
    m_ar, b_ar = np.polyfit(ars.loc[shared_ar].values,
                             z_rec.loc[shared_ar, 'AR'].values, 1)
    x_ar = np.linspace(ars.loc[shared_ar].min(), ars.loc[shared_ar].max(), 100)
    ax.plot(x_ar, m_ar * x_ar + b_ar, 'k-', lw=2, alpha=0.7)
    ax.set_xlabel('AR Activity Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('AR mRNA z-score', fontsize=11, fontweight='bold')
    ax.set_title(f'AR mRNA vs AR Activity Score\nAuto-regulation: ρ={r_ar_ars:+.3f}, p={p_ar_ars:.2e} {sig(p_ar_ars)}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.legend(handles=[Patch(facecolor='#27ae60', label='Adenocarcinoma'),
                        Patch(facecolor='#8e44ad', label='SCNC'),
                        Patch(facecolor='grey',    label='Other')], fontsize=9)
    print(f"  AR mRNA vs ARS: ρ={r_ar_ars:+.3f}, p={p_ar_ars:.2e} {sig(p_ar_ars)}")

# Panel C: AR mRNA vs m6A Functional Impact
ax = axes_ar[2]
if 'AR' in z_rec.columns:
    r_ar_fi, p_ar_fi = spearmanr(z_rec.loc[shared_ar, 'AR'],
                                   meta.loc[shared_ar, 'm6A_Functional_Impact'])
    ax.scatter(z_rec.loc[shared_ar, 'AR'],
               meta.loc[shared_ar, 'm6A_Functional_Impact'],
               c=cmap_hist.loc[shared_ar].values, alpha=0.4, s=20, edgecolors='none')
    m_fi2, b_fi2 = np.polyfit(z_rec.loc[shared_ar, 'AR'].values,
                               meta.loc[shared_ar, 'm6A_Functional_Impact'].values, 1)
    x_fi2 = np.linspace(z_rec.loc[shared_ar, 'AR'].min(),
                         z_rec.loc[shared_ar, 'AR'].max(), 100)
    ax.plot(x_fi2, m_fi2 * x_fi2 + b_fi2, 'k-', lw=2, alpha=0.7)
    ax.set_xlabel('AR mRNA z-score', fontsize=11, fontweight='bold')
    ax.set_ylabel('m6A Functional Impact', fontsize=11, fontweight='bold')
    ax.set_title(f'AR mRNA vs m6A Functional Impact\nρ={r_ar_fi:+.3f}, p={p_ar_fi:.2e} {sig(p_ar_fi)}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    print(f"  AR mRNA vs m6A FI: ρ={r_ar_fi:+.3f}, p={p_ar_fi:.2e} {sig(p_ar_fi)}")

plt.suptitle('AR Receptor mRNA Level: Histology, Auto-regulation & m6A Link (mCRPC)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '26_AR_mRNA_histology_ARS_FI.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 26_AR_mRNA_histology_ARS_FI.png")
plt.close()

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

# --- 28. Steroid receptor panel heatmap across m6A FI quartiles --------------
print("\n--- Plot 28: Steroid receptor panel — m6A FI quartile heatmap ---")
fi_quartiles = pd.qcut(meta.loc[df.index, 'm6A_Functional_Impact'], q=4,
                        labels=['Q1\n(FI-low)', 'Q2', 'Q3', 'Q4\n(FI-high)'])
rec_heat = []
for rec in receptors_avail:
    row = {'Receptor': f"{rec} ({STEROID_RECEPTORS[rec]})"}
    for ql in fi_quartiles.cat.categories:
        idx_q = fi_quartiles[fi_quartiles == ql].index
        row[str(ql)] = z_rec.loc[idx_q.intersection(z_rec.index), rec].mean()
    rec_heat.append(row)
rh_df = pd.DataFrame(rec_heat).set_index('Receptor')

fig, ax = plt.subplots(figsize=(8, 4 + len(receptors_avail) * 0.6))
sns.heatmap(rh_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean z-score'})
ax.set_title('Steroid Receptor mRNA Across m6A Functional Impact Quartiles (mCRPC)\n'
             'Q1 = m6A-low, Q4 = m6A-high',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('m6A Functional Impact Quartile', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '28_steroid_receptor_FI_quartile_heatmap.png'),
            dpi=300, bbox_inches='tight')
print("  → Saved: 28_steroid_receptor_FI_quartile_heatmap.png")
plt.close()

# Print stats for each receptor × m6A FI quartile (Kruskal-Wallis)
print(f"\n  {'Receptor':8s} {'KW p':>12s} {'':3s}")
for rec in receptors_avail:
    groups_fi_r = [z_rec.loc[fi_quartiles[fi_quartiles == ql].index.intersection(z_rec.index),
                              rec].values
                   for ql in fi_quartiles.cat.categories]
    groups_fi_r = [g for g in groups_fi_r if len(g) > 2]
    if len(groups_fi_r) >= 2:
        _, p_kw_r = kruskal(*groups_fi_r)
        print(f"  {rec:8s} {p_kw_r:12.2e} {sig(p_kw_r)}")

# --- 29. AR vs NR3C1 scatter colored by m6A FI -------------------------------
print("\n--- Plot 29: AR vs NR3C1 colored by m6A FI (receptor landscape) ---")
if 'AR' in z_rec.columns and 'NR3C1' in z_rec.columns:
    shared_both = (z_rec[['AR', 'NR3C1']].dropna().index
                   .intersection(meta['m6A_Functional_Impact'].dropna().index)
                   .intersection(df.index))
    fi_vals = meta.loc[shared_both, 'm6A_Functional_Impact']
    # Normalize FI to [0,1] for colormap
    fi_norm = (fi_vals - fi_vals.min()) / (fi_vals.max() - fi_vals.min())
    cmap_fi = plt.cm.RdBu_r(fi_norm.values)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(z_rec.loc[shared_both, 'AR'],
                    z_rec.loc[shared_both, 'NR3C1'],
                    c=meta.loc[shared_both, 'm6A_Functional_Impact'],
                    cmap='RdBu_r', alpha=0.55, s=25, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax, label='m6A Functional Impact')
    # Quadrant medians
    med_ar  = z_rec.loc[shared_both, 'AR'].median()
    med_gr  = z_rec.loc[shared_both, 'NR3C1'].median()
    ax.axvline(med_ar,  color='grey', lw=1, ls='--', alpha=0.6)
    ax.axhline(med_gr,  color='grey', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('AR mRNA z-score', fontsize=12, fontweight='bold')
    ax.set_ylabel('NR3C1 (GR) mRNA z-score', fontsize=12, fontweight='bold')
    ax.set_title('AR vs GR (NR3C1) Receptor Landscape\n'
                 'Colored by m6A Functional Impact — mCRPC\n'
                 'Upper-left: AR-lo/GR-hi = potential AR→GR lineage switch',
                 fontsize=12, fontweight='bold', pad=12)
    # Label quadrants
    xl, xr = ax.get_xlim(); yb, yt = ax.get_ylim()
    ax.text(xr * 0.85, yt * 0.92, 'AR-hi\nGR-hi', ha='center', va='top',
            fontsize=9, color='#c0392b', fontweight='bold', alpha=0.8)
    ax.text(xl * 0.85, yt * 0.92, 'AR-lo\nGR-hi\n(bypass)', ha='center', va='top',
            fontsize=9, color='#8e44ad', fontweight='bold', alpha=0.8)
    ax.text(xr * 0.85, yb * 0.5,  'AR-hi\nGR-lo', ha='center', va='bottom',
            fontsize=9, color='#27ae60', fontweight='bold', alpha=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '29_AR_GR_receptor_landscape.png'), dpi=300, bbox_inches='tight')
    print("  → Saved: 29_AR_GR_receptor_landscape.png")
    plt.close()
else:
    print("  AR or NR3C1 not in dataset — skipping plot 29")

# ===========================================================================
# PART IX — CROSS-COHORT AR ACTIVITY TRAJECTORY
# ===========================================================================
print("\n" + "=" * 80)
print("  PART IX — CROSS-COHORT AR ACTIVITY TRAJECTORY")
print("=" * 80)

print("\n  Loading cross-cohort datasets ...")
gtex_expr_cc, _       = load_gtex()
adj_normal_expr_cc, _ = load_adj_normal()
mhspc_expr_cc, _      = load_mcspc()
# mCRPC and TCGA are already loaded (df / df_tc) — reuse same objects
print(f"  GTEx: {gtex_expr_cc.shape[0]} samples, Adj Normal: {adj_normal_expr_cc.shape[0]} samples, "
      f"mCSPC: {mhspc_expr_cc.shape[0]} samples")

# Build common gene universe (all 5 cohorts)
common_cc = build_common_universe(
    [df, df_tc, gtex_expr_cc, adj_normal_expr_cc, mhspc_expr_cc]
)
ar_cc_avail  = [g for g in AR_TARGET_GENES if g in common_cc]
m6a_cc_avail = [g for g in CROSS_COHORT_M6A_GENES if g in common_cc]
print(f"  Common universe: {len(common_cc):,} genes")
print(f"  AR targets in common: {len(ar_cc_avail)}/{len(AR_TARGET_GENES)} — {ar_cc_avail}")
print(f"  m6A genes in common: {len(m6a_cc_avail)}/{len(CROSS_COHORT_M6A_GENES)}")

# Percentile rank normalization (like cross_cohort.py)
def _pct(expr_df):
    return percentile_rank_matrix(expr_df[common_cc].fillna(0.0))

print("  Computing percentile ranks ...")
gtex_pct_cc       = _pct(gtex_expr_cc)
adjn_pct_cc       = _pct(adj_normal_expr_cc)
tcga_pct_cc       = _pct(df_tc)
mhspc_pct_cc      = _pct(mhspc_expr_cc)
mcrpc_pct_cc      = _pct(df)  # full dataset

# Split mCRPC into Adeno and SCNC
idx_mcrpc_adeno = meta[meta['histology'] == 'Adenocarcinoma'].index.intersection(df.index)
idx_mcrpc_scnc  = meta[meta['histology'] == 'SCNC'].index.intersection(df.index)

# AR Activity Score in percentile rank space (mean percentile of AR targets)
def _ars_pct(pct_df):
    available = [g for g in ar_cc_avail if g in pct_df.columns]
    return pct_df[available].mean(axis=1)

# m6A axes in percentile rank space
def _m6a_axes_pct(pct_df):
    nd, orr, fi = compute_axes(pct_df[m6a_cc_avail])
    return nd, orr, fi

ars_gtex    = _ars_pct(gtex_pct_cc)
ars_adjn    = _ars_pct(adjn_pct_cc)
ars_tcga    = _ars_pct(tcga_pct_cc)
ars_mhspc   = _ars_pct(mhspc_pct_cc)
ars_adeno   = _ars_pct(mcrpc_pct_cc.loc[idx_mcrpc_adeno])
ars_scnc    = _ars_pct(mcrpc_pct_cc.loc[idx_mcrpc_scnc])

fi_gtex     = _m6a_axes_pct(gtex_pct_cc)[2]
fi_adjn     = _m6a_axes_pct(adjn_pct_cc)[2]
fi_tcga     = _m6a_axes_pct(tcga_pct_cc)[2]
fi_mhspc    = _m6a_axes_pct(mhspc_pct_cc)[2]
fi_adeno    = _m6a_axes_pct(mcrpc_pct_cc.loc[idx_mcrpc_adeno])[2]
fi_scnc     = _m6a_axes_pct(mcrpc_pct_cc.loc[idx_mcrpc_scnc])[2]

# Cohort labels matching cross_cohort.py convention
CC_LABELS  = ['Normal\n(GTEx)', 'Adj Normal\n(TCGA)', 'Primary PCa\n(TCGA)',
              'mCSPC\n(GSE221601)', 'mCRPC\nAdeno', 'mCRPC\nSCNC']
CC_COLORS  = ['#27ae60', '#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#c0392b']
ars_groups = [ars_gtex, ars_adjn, ars_tcga, ars_mhspc, ars_adeno, ars_scnc]
fi_groups  = [fi_gtex,  fi_adjn,  fi_tcga,  fi_mhspc,  fi_adeno,  fi_scnc]
ns_cc      = [len(g) for g in ars_groups]

print(f"\n  Cross-cohort ARS (mean percentile rank):")
print(f"  {'Cohort':20s} {'n':>5s}  {'ARS':>8s}")
for lbl, grp, n in zip(CC_LABELS, ars_groups, ns_cc):
    lbl_flat = lbl.replace('\n', ' ')
    print(f"  {lbl_flat:20s} {n:5d}  {grp.mean():.1f}")

# --- 30. ARS across 6 disease stages violin ----------------------------------
print("\n--- Plot 30: AR Activity Score across 6 disease stages ---")
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
ax.set_ylabel('AR Activity Score\n(mean percentile rank of 10 AR targets)', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity Score Across Prostate Cancer Disease Stages\n'
             f'Kruskal-Wallis p={p_ars_kw:.2e} {sig(p_ars_kw)}\n'
             f'AR targets: {", ".join(ar_cc_avail)}',
             fontsize=12, fontweight='bold', pad=12)
ax.axhline(50, color='grey', lw=0.8, ls='--', alpha=0.7, label='Global median (50th %ile)')
ax.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '30_cross_cohort_ARS_trajectory.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 30_cross_cohort_ARS_trajectory.png  (KW p={p_ars_kw:.2e} {sig(p_ars_kw)})")
plt.close()

# --- 31. ARS vs m6A FI across all stages (scatter) ---------------------------
print("\n--- Plot 31: ARS vs m6A FI across disease stages ---")
fig, ax = plt.subplots(figsize=(10, 8))
for ars_g, fi_g, lbl, c, n in zip(ars_groups, fi_groups, CC_LABELS, CC_COLORS, ns_cc):
    common_idx = ars_g.dropna().index.intersection(fi_g.dropna().index)
    lbl_flat = lbl.replace('\n', ' ')
    ax.scatter(ars_g.loc[common_idx], fi_g.loc[common_idx],
               c=c, alpha=0.4, s=18, edgecolors='none',
               label=f'{lbl_flat} (n={n})')

# Overall correlation
all_ars = pd.concat(ars_groups).dropna()
all_fi  = pd.concat(fi_groups).dropna()
common_overall = all_ars.index.intersection(all_fi.index)
r_all, p_all = spearmanr(all_ars.loc[common_overall], all_fi.loc[common_overall])
ax.set_xlabel('AR Activity Score (percentile rank)', fontsize=12, fontweight='bold')
ax.set_ylabel('m6A Functional Impact (percentile rank-based)', fontsize=12, fontweight='bold')
ax.set_title(f'AR Activity vs m6A Functional Impact — All Disease Stages\n'
             f'Overall Spearman ρ={r_all:+.3f}, p={p_all:.2e} {sig(p_all)}',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(all_fi.loc[common_overall].median(), color='grey', lw=1, ls='--', alpha=0.5)
ax.axvline(all_ars.loc[common_overall].median(), color='grey', lw=1, ls='--', alpha=0.5)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '31_cross_cohort_ARS_vs_FI.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 31_cross_cohort_ARS_vs_FI.png  (overall ρ={r_all:+.3f} {sig(p_all)})")
plt.close()

# --- 32. AR target gene percentile rank heatmap across 6 stages --------------
print("\n--- Plot 32: AR target gene percentile rank heatmap across stages ---")
pct_dfs_cc = [gtex_pct_cc, adjn_pct_cc, tcga_pct_cc,
              mhspc_pct_cc,
              mcrpc_pct_cc.loc[idx_mcrpc_adeno],
              mcrpc_pct_cc.loc[idx_mcrpc_scnc]]
heat_ar = pd.DataFrame(index=CC_LABELS, columns=ar_cc_avail, dtype=float)
for lbl, pct_df in zip(CC_LABELS, pct_dfs_cc):
    for gene in ar_cc_avail:
        if gene in pct_df.columns:
            heat_ar.loc[lbl, gene] = pct_df[gene].mean()

fig, ax = plt.subplots(figsize=(max(10, len(ar_cc_avail) * 1.0), 6))
sns.heatmap(heat_ar.astype(float), annot=True, fmt='.1f', cmap='RdBu_r',
            center=50, vmin=20, vmax=80,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean percentile rank'})
ax.set_title('AR Target Gene Percentile Ranks Across Disease Stages\n'
             'Values = within-sample %ile rank averaged over cohort\n'
             'Red = high expression relative to transcriptome; Blue = low',
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('AR Target Gene', fontsize=11, fontweight='bold')
ax.set_ylabel('Disease Stage', fontsize=11, fontweight='bold')
ax.set_yticklabels([lbl.replace('\n', ' ') for lbl in CC_LABELS],
                   fontsize=10, rotation=0, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '32_AR_target_gene_percentile_heatmap.png'),
            dpi=300, bbox_inches='tight')
print("  → Saved: 32_AR_target_gene_percentile_heatmap.png")
plt.close()

print("\n\n" + "=" * 80)
print("  ALL PLOTS SAVED")
print("=" * 80)
for i, f in enumerate([
    '01_AR_target_coexpression.png',
    '02_ARS_by_histology_lumbasline.png',
    '03_ARS_by_AR_alteration.png',
    '04_ARS_clinical_context.png',
    '05_m6A_gene_ARS_correlation.png',
    '06_top_m6A_ARS_scatters.png',
    '07_m6A_gene_ARS_quartile_heatmap.png',
    '08_ARS_vs_NetDeposition.png',
    '09_ARS_vs_OncogenicReadout.png',
    '10_ARS_vs_FunctionalImpact.png',
    '11_ARS_FI_landscape.png',
    '12_m6A_axes_ASI_treatment.png',
    '13_m6A_axes_PSA_response.png',
    '14_m6A_axes_prior_ASI.png',
    '15_m6A_axes_AR_alteration.png',
    '16_KM_ARxFI_quadrant.png',
    '17_KM_ARS.png',
    '19_Cox_HR_forest.png',
    '20_TCGA_ARS_by_Gleason.png',
    '21_TCGA_FI_by_Gleason.png',
    '22_TCGA_ARS_vs_FI_Gleason.png',
    '23_ARS_by_site.png',
    '24_ARS_FI_scatter_by_site.png',
    '25_AR_m6A_summary_panel.png',
    '26_AR_mRNA_histology_ARS_FI.png',
    '27_NR3C1_GR_bypass.png',
    '28_steroid_receptor_FI_quartile_heatmap.png',
    '29_AR_GR_receptor_landscape.png',
    '30_cross_cohort_ARS_trajectory.png',
    '31_cross_cohort_ARS_vs_FI.png',
    '32_AR_target_gene_percentile_heatmap.png',
], 1):
    print(f"  {i:2d}. {f}")
print(f"\n  Output directory: {OUTDIR}")
print("=" * 80)
