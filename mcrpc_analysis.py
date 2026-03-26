#!/usr/bin/env python3
"""
mcrpc_analysis.py — mCRPC intra-cohort m6A analysis.

Three-axis model of m6A epitranscriptomic regulation in mCRPC:

  Axis 1: Net Deposition       = Weighted Writers (LR-weights) − Erasers
  Axis 2: Oncogenic Readout    = Oncogenic Readers − Suppressive Readers
  Axis 3: Functional Impact    = w_nd × Net Deposition + w_or × Oncogenic Readout
           (w_nd/w_or optimized via logistic regression on SCNC vs Adenocarcinoma)

Structure:
  Part I   — Model Selection: manual, bottleneck, logistic-regression weights
  Part II  — Full Analysis: per-gene, composite, KM, landscape, site, liver
             (all using logistic regression weights)

Output: plots_mcrpc/

Usage:
    micromamba run -n rnaseq python mcrpc_analysis.py
"""
import os
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import (mannwhitneyu, spearmanr, kruskal, norm,
                          pearsonr, fisher_exact)
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings('ignore')
plt.rcParams['savefig.dpi'] = 300

# ── m6a package imports ───────────────────────────────────────────────────────
from m6a.config import (
    OUTDIR_MCRPC as OUTDIR,
    MCRPC_LOG2CPM, MCRPC_META_CSV,
    MCRPC_SITE_ORDER, MCRPC_SITE_COLORS,
)
from m6a.genes import (
    MCRPC_WRITER_GENES  as WRITER_GENES,
    MCRPC_ERASER_GENES  as ERASER_GENES,
    MCRPC_READER_ONCOGENIC  as READER_ONCOGENIC,
    MCRPC_READER_SUPPRESSIVE as READER_SUPPRESSIVE,
    MCRPC_ALL_GENES     as ALL_M6A_GENES,
    MCRPC_GENE_ROLES    as gene_roles,
    MCRPC_GENE_ORDER    as gene_order,
    MCRPC_GENE_LABELS   as gene_labels,
    MCRPC_MANUAL_WEIGHTS as MANUAL_WEIGHTS,
)
from m6a.stats import sig
from m6a.normalization import zscore_normalize
from m6a.plotting import style_violin

os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 80)
print("  m6A WRITER COMPLEX — LOGISTIC REGRESSION WEIGHTED FUNCTIONAL SCORE")
print("=" * 80)

df   = pd.read_csv(MCRPC_LOG2CPM, sep='\t', index_col=0)
meta = pd.read_csv(MCRPC_META_CSV, index_col=1)

print(f"  Dataset: {df.shape[0]} patients × {df.shape[1]} genes")

missing = [g for g in ALL_M6A_GENES if g not in df.columns]
if missing:
    print(f"  WARNING: Missing genes: {missing}")
else:
    print(f"  All {len(ALL_M6A_GENES)} m6A genes found ✓")

# =============================================================================
# Z-SCORE NORMALIZATION
# =============================================================================
z_all = zscore_normalize(df[ALL_M6A_GENES])
for gene in ALL_M6A_GENES:
    meta[f'z_{gene}'] = z_all[gene]

# ── Group indices ─────────────────────────────────────────────────────────────
idx_lum   = meta[meta['Luminal/Basal Cluster'] == 'Luminal'].index.intersection(df.index)
idx_bas   = meta[meta['Luminal/Basal Cluster'] == 'Basal'].index.intersection(df.index)
idx_adeno = meta[meta['histology'] == 'Adenocarcinoma'].index.intersection(df.index)
idx_scnc  = meta[meta['histology'] == 'SCNC'].index.intersection(df.index)

# Shared building blocks (independent of writer-weight choice)
eraser_z = z_all[ERASER_GENES].mean(axis=1)
onco_z   = z_all[READER_ONCOGENIC].mean(axis=1)
supp_z   = z_all[READER_SUPPRESSIVE].mean(axis=1)
meta['m6A_Oncogenic_Readout'] = onco_z - supp_z

# ── Local plot helpers ────────────────────────────────────────────────────────
SITE_ORDER  = MCRPC_SITE_ORDER
SITE_COLORS = MCRPC_SITE_COLORS


def two_group_compare(score_col, group_col, label_a, label_b, score_label=None):
    """Mann-Whitney U for a score between two mCRPC subgroups."""
    if score_label is None:
        score_label = score_col
    idx_a = meta[meta[group_col] == label_a].index.intersection(df.index)
    idx_b = meta[meta[group_col] == label_b].index.intersection(df.index)
    va = meta.loc[idx_a, score_col].dropna().values
    vb = meta.loc[idx_b, score_col].dropna().values
    stat, p = mannwhitneyu(va, vb, alternative='two-sided')
    return {
        'label': score_label, 'group_a': label_a, 'group_b': label_b,
        'n_a': len(va), 'n_b': len(vb),
        'mean_a': va.mean(), 'mean_b': vb.mean(),
        'delta': vb.mean() - va.mean(), 'p': p, 'sig': sig(p),
        'va': va, 'vb': vb, 'idx_a': idx_a, 'idx_b': idx_b,
    }


def plot_violin(r, title, filename, color_a='#2980b9', color_b='#c0392b'):
    """Two-group violin for an mCRPC comparison result dict."""
    fig, ax = plt.subplots(figsize=(7, 6))
    parts = ax.violinplot([r['va'], r['vb']], positions=[0, 1],
                          showmeans=True, showmedians=True)
    for i, c in enumerate([color_a, color_b]):
        parts['bodies'][i].set_facecolor(c)
        parts['bodies'][i].set_alpha(0.6)
    style_violin(parts, ax)
    y_max = max(r['va'].max(), r['vb'].max())
    ax.plot([0, 1], [y_max + 0.15, y_max + 0.15], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.18,
            f"p={r['p']:.2e} {r['sig']}", ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{r['group_a']}\n(n={r['n_a']})",
                        f"{r['group_b']}\n(n={r['n_b']})"],
                       fontsize=11, fontweight='bold')
    ax.set_ylabel(r['label'], fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300, bbox_inches='tight')
    print(f"  → Saved: {filename}")
    plt.close()
    return fig


def km_survival(score_col, score_label, filename,
                color_hi='#e74c3c', color_lo='#3498db'):
    """Kaplan-Meier overall survival by median split."""
    surv = meta[['surv_months', 'vital_status']].dropna()
    surv = surv[surv.index.isin(df.index)]
    vals = meta.loc[surv.index, score_col].dropna()
    surv = surv.loc[vals.index]
    med  = vals.median()
    surv = surv.copy()
    surv['grp'] = np.where(vals >= med, 'High', 'Low')
    hi = surv[surv['grp'] == 'High']
    lo = surv[surv['grp'] == 'Low']
    lr = logrank_test(hi['surv_months'], lo['surv_months'],
                      hi['vital_status'], lo['vital_status'])
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf.fit(hi['surv_months'], hi['vital_status'],
            label=f'High {score_label} (n={len(hi)})')
    kmf.plot_survival_function(ax=ax, color=color_hi, linewidth=2)
    kmf.fit(lo['surv_months'], lo['vital_status'],
            label=f'Low {score_label} (n={len(lo)})')
    kmf.plot_survival_function(ax=ax, color=color_lo, linewidth=2)
    ax.set_xlabel('Time (months)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=13, fontweight='bold')
    ax.set_title(f'Overall Survival by {score_label}\n'
                 f'Log-rank p={lr.p_value:.4f} {sig(lr.p_value)}  |  '
                 f'Median split at {med:.2f}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=12, loc='lower left')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300, bbox_inches='tight')
    print(f"  → Saved: {filename}")
    plt.close()
    return {'score': score_label, 'n': len(surv), 'events': int(surv['vital_status'].sum()),
            'median_split': med, 'chi2': lr.test_statistic,
            'p': lr.p_value, 'sig': sig(lr.p_value)}


# #############################################################################
#  PART I — MODEL SELECTION: THREE WEIGHTING APPROACHES
# #############################################################################
print("\n" + "=" * 80)
print("  PART I — MODEL SELECTION: THREE WEIGHTING APPROACHES")
print("=" * 80)

# ── MODEL 1: Manual biology-based weights ─────────────────────────────────────
w_total_m   = sum(MANUAL_WEIGHTS.values())
manual_wr   = sum(z_all[g] * w for g, w in MANUAL_WEIGHTS.items()) / w_total_m
meta['m6A_Manual_NetDep'] = manual_wr - eraser_z

print(f"\n  Model 1: Manual Weights (biology-based)")
print(f"    Weights: {', '.join(f'{g}={w}' for g, w in MANUAL_WEIGHTS.items())}")
print(f"    Net Deposition (mean ± SD): "
      f"{meta['m6A_Manual_NetDep'].mean():+.3f} ± {meta['m6A_Manual_NetDep'].std():.3f}")

# ── MODEL 2: Bottleneck / Complex Assembly ────────────────────────────────────
print(f"\n  Model 2: Bottleneck (Complex Assembly)")
print(f"    z-score → Φ(z) activity probability → weighted geometric mean")

writer_probs = pd.DataFrame(index=df.index)
for gene in WRITER_GENES:
    writer_probs[gene] = norm.cdf(z_all.loc[df.index, gene])

w_arr    = np.array([MANUAL_WEIGHTS[g] for g in WRITER_GENES])
w_sum    = w_arr.sum()
log_prod = sum(w_arr[i] * np.log(writer_probs[g] + 1e-10)
               for i, g in enumerate(WRITER_GENES))
complex_assembly = np.exp(log_prod / w_sum)

eraser_prob = pd.DataFrame(index=df.index)
for gene in ERASER_GENES:
    eraser_prob[gene] = norm.cdf(z_all.loc[df.index, gene])
eraser_assembly = eraser_prob.mean(axis=1)

meta['m6A_Bottleneck_NetDep'] = complex_assembly - eraser_assembly

print(f"    Complex Assembly  (mean ± SD): "
      f"{complex_assembly.mean():.3f} ± {complex_assembly.std():.3f}")
print(f"    Net Deposition    (mean ± SD): "
      f"{meta['m6A_Bottleneck_NetDep'].mean():+.3f} ± {meta['m6A_Bottleneck_NetDep'].std():.3f}")

bottleneck_gene   = writer_probs[WRITER_GENES].idxmin(axis=1)
bottleneck_counts = (bottleneck_gene.value_counts()
                     .reindex(WRITER_GENES).fillna(0).astype(int))

# ── MODEL 3: Logistic Regression Weights ──────────────────────────────────────
print(f"\n  Model 3: Logistic Regression Weights")
print(f"    Predicting SCNC vs Adenocarcinoma from writer z-scores")

idx_classified = (meta[meta['histology'].isin(['Adenocarcinoma', 'SCNC'])]
                  .index.intersection(df.index))
X_writers = z_all.loc[idx_classified, WRITER_GENES].values
y_subtype = (meta.loc[idx_classified, 'histology'] == 'SCNC').astype(int).values

lr_model  = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_writers, y_subtype)
cv_scores = cross_val_score(lr_model, X_writers, y_subtype, cv=5, scoring='roc_auc')

lr_coefs  = lr_model.coef_[0]
lr_abs    = np.abs(lr_coefs)
lr_wts    = lr_abs / lr_abs.sum() * w_sum
LR_WEIGHTS = {g: lw for g, lw in zip(WRITER_GENES, lr_wts)}

print(f"    Cross-validated AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Leave-One-Source-Out CV
print(f"\n    Leave-One-Source-Out CV (external validation across cohorts):")
sources_classified = meta.loc[idx_classified, 'source']
loso_results = []
for held_out in sorted(sources_classified.unique()):
    train_mask = (sources_classified != held_out).values
    test_mask  = (sources_classified == held_out).values
    n_test_scnc  = y_subtype[test_mask].sum()
    n_test_adeno = test_mask.sum() - n_test_scnc
    if n_test_scnc < 2:
        print(f"      Hold out {held_out:10s}: n_test={test_mask.sum():3d} "
              f"(Ade={n_test_adeno}, SCNC={n_test_scnc})  — skipped (too few SCNC)")
        continue
    lr_loso = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    lr_loso.fit(X_writers[train_mask], y_subtype[train_mask])
    y_prob    = lr_loso.predict_proba(X_writers[test_mask])[:, 1]
    auc_loso  = roc_auc_score(y_subtype[test_mask], y_prob)
    loso_c    = np.abs(lr_loso.coef_[0])
    loso_w    = loso_c / loso_c.sum() * w_sum
    top_gene  = WRITER_GENES[np.argmax(loso_w)]
    loso_results.append({'source': held_out, 'auc': auc_loso,
                         'n': test_mask.sum(), 'n_adeno': n_test_adeno,
                         'n_scnc': n_test_scnc, 'weights': loso_w})
    print(f"      Hold out {held_out:10s}: AUC={auc_loso:.3f}  "
          f"n_test={test_mask.sum():3d} (Ade={n_test_adeno}, SCNC={n_test_scnc})  "
          f"top gene={top_gene} ({loso_w[np.argmax(loso_w)]:.2f})")

if loso_results:
    mean_loso_auc = np.mean([r['auc'] for r in loso_results])
    std_loso_auc  = np.std( [r['auc'] for r in loso_results])
    print(f"\n    LOSO-CV AUC (mean ± SD): {mean_loso_auc:.3f} ± {std_loso_auc:.3f}")
    print(f"    (5-fold CV AUC was:      {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
    print(f"\n    Weight stability across LOSO folds:")
    print(f"    {'Gene':10s} {'Full model':>10s}  ", end='')
    for r in loso_results:
        print(f"  {'w/o '+r['source']:>12s}", end='')
    print(f"  {'SD':>6s}")
    print("    " + "-" * (26 + 14 * len(loso_results)))
    for gi, gene in enumerate(WRITER_GENES):
        full_w  = lr_wts[gi]
        fold_ws = [r['weights'][gi] for r in loso_results]
        sd      = np.std(fold_ws)
        print(f"    {gene:10s} {full_w:10.2f}  ", end='')
        for fw in fold_ws:
            print(f"  {fw:12.2f}", end='')
        print(f"  {sd:6.3f}")
else:
    mean_loso_auc = cv_scores.mean()
    std_loso_auc  = cv_scores.std()

print(f"\n    {'Gene':10s} {'Manual W':>9s} {'LR Coef':>9s} {'Direction':>10s} {'Data W':>8s}")
print("    " + "-" * 50)
for i, gene in enumerate(WRITER_GENES):
    direction = '→ SCNC' if lr_coefs[i] > 0 else '→ Adeno'
    print(f"    {gene:10s} {MANUAL_WEIGHTS[gene]:9.1f} {lr_coefs[i]:+9.3f} "
          f"{direction:>10s} {lr_wts[i]:8.2f}")

# Compute LR-weighted scores (used throughout Part II)
dd_w_total         = sum(LR_WEIGHTS.values())
dd_weighted_writer = sum(z_all[g] * w for g, w in LR_WEIGHTS.items()) / dd_w_total
meta['m6A_Net_Deposition'] = dd_weighted_writer - eraser_z

print(f"\n    Net Deposition (mean ± SD): "
      f"{meta['m6A_Net_Deposition'].mean():+.3f} ± {meta['m6A_Net_Deposition'].std():.3f}")

# =============================================================================
# AXIS WEIGHT OPTIMIZATION
# =============================================================================
print("\n" + "-" * 80)
print("  AXIS WEIGHT OPTIMIZATION (Net Deposition vs Oncogenic Readout)")
print("-" * 80)

idx_hist  = (meta[meta['histology'].isin(['Adenocarcinoma', 'SCNC'])]
             .index.intersection(df.index))
X_axes    = np.column_stack([
    meta.loc[idx_hist, 'm6A_Net_Deposition'].values,
    meta.loc[idx_hist, 'm6A_Oncogenic_Readout'].values,
])
y_hist    = (meta.loc[idx_hist, 'histology'] == 'SCNC').astype(int).values

scaler_axes   = StandardScaler()
X_axes_scaled = scaler_axes.fit_transform(X_axes)

lr_axes  = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
lr_axes.fit(X_axes_scaled, y_hist)
cv_axes  = cross_val_score(lr_axes, X_axes_scaled, y_hist, cv=5, scoring='roc_auc')

axes_coefs = lr_axes.coef_[0]
axes_abs   = np.abs(axes_coefs)
W_ND = axes_abs[0] / axes_abs.sum()
W_OR = axes_abs[1] / axes_abs.sum()

print(f"\n  Method: Logistic Regression (SCNC vs Adeno) on standardized axes")
print(f"  CV-AUC: {cv_axes.mean():.3f} ± {cv_axes.std():.3f}")
print(f"\n  Raw coefficients:")
print(f"    Net Deposition:    {axes_coefs[0]:+.4f} "
      f"({'→ SCNC' if axes_coefs[0] > 0 else '→ Adeno'})")
print(f"    Oncogenic Readout: {axes_coefs[1]:+.4f} "
      f"({'→ SCNC' if axes_coefs[1] > 0 else '→ Adeno'})")
print(f"\n  Normalized weights (sum = 1):")
print(f"    w(Net Deposition)    = {W_ND:.3f}")
print(f"    w(Oncogenic Readout) = {W_OR:.3f}")

# Grid search
grid_wts = np.arange(0.0, 1.01, 0.05)
grid_results = []
for w_nd in grid_wts:
    fi_temp = (meta.loc[df.index, 'm6A_Net_Deposition'] * w_nd +
               meta.loc[df.index, 'm6A_Oncogenic_Readout'] * (1.0 - w_nd))
    _, p_val = mannwhitneyu(fi_temp.loc[idx_adeno].values,
                             fi_temp.loc[idx_scnc].values, alternative='two-sided')
    grid_results.append({'w_nd': w_nd, 'w_or': 1.0 - w_nd,
                          'p': p_val, 'neg_log_p': -np.log10(p_val)})
grid_df  = pd.DataFrame(grid_results)
best_row = grid_df.loc[grid_df['p'].idxmin()]

print(f"\n  Grid search best: w_nd={best_row['w_nd']:.2f}, "
      f"w_or={best_row['w_or']:.2f}, p={best_row['p']:.2e}")

# Apply weights to all three models
meta['m6A_Manual_FuncImpact']     = meta['m6A_Manual_NetDep']     * W_ND + meta['m6A_Oncogenic_Readout'] * W_OR
meta['m6A_Bottleneck_FuncImpact'] = meta['m6A_Bottleneck_NetDep'] * W_ND + meta['m6A_Oncogenic_Readout'] * W_OR
meta['m6A_Functional_Impact']     = meta['m6A_Net_Deposition']    * W_ND + meta['m6A_Oncogenic_Readout'] * W_OR

print(f"    Functional Impact (mean ± SD): "
      f"{meta['m6A_Functional_Impact'].mean():+.3f} ± {meta['m6A_Functional_Impact'].std():.3f}")

# Head-to-head model comparison
print("\n" + "-" * 80)
print("  MODEL COMPARISON — ALL THREE APPROACHES")
print("-" * 80)

model_scores = {
    'Manual NetDep':      ('m6A_Manual_NetDep',         'Net Dep (Manual)'),
    'Manual FuncImp':     ('m6A_Manual_FuncImpact',     'Func Impact (Manual)'),
    'Bottleneck NetDep':  ('m6A_Bottleneck_NetDep',     'Net Dep (Bottleneck)'),
    'Bottleneck FuncImp': ('m6A_Bottleneck_FuncImpact', 'Func Impact (Bottleneck)'),
    'LogReg NetDep':      ('m6A_Net_Deposition',        'Net Dep (Log. Reg.)'),
    'LogReg FuncImp':     ('m6A_Functional_Impact',     'Func Impact (Log. Reg.)'),
    'Oncogenic Readout':  ('m6A_Oncogenic_Readout',     'Oncogenic Readout'),
}
comp_results = []
for name, (col, label) in model_scores.items():
    r_as = two_group_compare(col, 'histology', 'Adenocarcinoma', 'SCNC', label)
    r_lb = two_group_compare(col, 'Luminal/Basal Cluster', 'Luminal', 'Basal', label)
    comp_results.append({'Score': name, 'p_AS': r_as['p'], 'sig_AS': r_as['sig'],
                          'p_LB': r_lb['p'], 'sig_LB': r_lb['sig']})

print(f"\n  {'Score':25s} {'p (Ade/SCNC)':>12s} {'':3s} {'p (Lum/Bas)':>12s} {'':3s}")
print("  " + "-" * 60)
for r in comp_results:
    print(f"  {r['Score']:25s} {r['p_AS']:12.2e} {r['sig_AS']:3s} "
          f"{r['p_LB']:12.2e} {r['sig_LB']:3s}")

h2h_rows = [
    ('Net Dep (Ade/SCNC)',  'm6A_Manual_NetDep',      'm6A_Net_Deposition',
     'histology', 'Adenocarcinoma', 'SCNC'),
    ('FuncImp (Ade/SCNC)',  'm6A_Manual_FuncImpact',  'm6A_Functional_Impact',
     'histology', 'Adenocarcinoma', 'SCNC'),
    ('Net Dep (Lum/Bas)',   'm6A_Manual_NetDep',       'm6A_Net_Deposition',
     'Luminal/Basal Cluster', 'Luminal', 'Basal'),
    ('FuncImp (Lum/Bas)',   'm6A_Manual_FuncImpact',   'm6A_Functional_Impact',
     'Luminal/Basal Cluster', 'Luminal', 'Basal'),
]
h2h_wins_lr = 0
print(f"\n  {'Comparison':30s} {'Manual p':>12s} {'':3s} "
      f"{'Logistic Regression p':>14s} {'':3s} {'Winner':>10s}")
print("  " + "-" * 75)
for name, col_m, col_dd, grp, a, b in h2h_rows:
    r_m  = two_group_compare(col_m,  grp, a, b)
    r_dd = two_group_compare(col_dd, grp, a, b)
    winner = 'Logistic Regression' if r_dd['p'] < r_m['p'] else 'Manual'
    if r_dd['p'] < r_m['p']:
        h2h_wins_lr += 1
    print(f"  {name:30s} {r_m['p']:12.2e} {r_m['sig']:3s} "
          f"{r_dd['p']:14.2e} {r_dd['sig']:3s} {winner:>10s}")
print(f"\n  → Logistic Regression wins {h2h_wins_lr}/4 head-to-head comparisons. "
      f"CV-AUC = {cv_scores.mean():.3f}. Using LR weights for all subsequent analyses.")

# =============================================================================
# PART I PLOTS  (01–04)
# =============================================================================

# --- 01. Weight comparison bar chart ---
print("\n--- Plot 01: Weight comparison ---")
fig, ax = plt.subplots(figsize=(12, 6))
x_w = np.arange(len(WRITER_GENES))
w_bar = 0.3
manual_w = [MANUAL_WEIGHTS[g] for g in WRITER_GENES]
data_w   = [LR_WEIGHTS[g]     for g in WRITER_GENES]
ax.bar(x_w - w_bar, manual_w, w_bar, label='Manual (Biology)', color='#2980b9',
       edgecolor='black', alpha=0.85)
ax.bar(x_w,         data_w,   w_bar, label='Logistic Regression', color='#e74c3c',
       edgecolor='black', alpha=0.85)
ax.set_xticks(x_w - w_bar / 2)
ax.set_xticklabels(
    [f"{g}\n({gene_roles[g].split('(')[1].rstrip(')')})" for g in WRITER_GENES],
    fontsize=8, rotation=30, ha='right',
)
ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
ax.set_title(
    'Writer Complex Weights: Manual vs Logistic Regression\n'
    f'5-fold CV-AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  |  '
    f'LOSO-CV AUC = {mean_loso_auc:.3f} ± {std_loso_auc:.3f}',
    fontsize=13, fontweight='bold', pad=12,
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '01_weight_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 01_weight_comparison.png")
plt.close()

# --- 02. Axis weight optimization grid search ---
print("\n--- Plot 02b: Axis weight optimization ---")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid_df['w_nd'], grid_df['neg_log_p'], 'b-o', markersize=4, linewidth=2)
ax.axvline(0.6,  color='grey',    ls='--', lw=1,   label='Old (0.60/0.40)')
ax.axvline(W_ND, color='#e74c3c', ls='-',  lw=2,   label=f'LR optimal ({W_ND:.2f}/{W_OR:.2f})')
ax.axvline(best_row['w_nd'], color='#27ae60', ls=':', lw=2,
           label=f'Grid best ({best_row["w_nd"]:.2f}/{best_row["w_or"]:.2f})')
ax.axhline(-np.log10(0.05), color='orange', ls='--', lw=1, alpha=0.7, label='p = 0.05')
ax.set_xlabel('Weight on Net Deposition (w_nd)', fontsize=12, fontweight='bold')
ax.set_ylabel('−log₁₀(p-value)  [Ade vs SCNC]', fontsize=12, fontweight='bold')
ax.set_title(
    'Axis Weight Optimization: Functional Impact\n'
    'Which mix of Net Deposition + Oncogenic Readout best separates Adeno vs SCNC?',
    fontsize=13, fontweight='bold', pad=12,
)
ax.legend(fontsize=10, loc='best')
sec_ax = ax.twiny()
sec_ax.set_xlim(ax.get_xlim())
sec_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
sec_ax.set_xticks(sec_ticks)
sec_ax.set_xticklabels([f'{1-t:.1f}' for t in sec_ticks])
sec_ax.set_xlabel('Weight on Oncogenic Readout (w_or)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '02b_axis_weight_optimization.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 02b_axis_weight_optimization.png")
plt.close()

# --- 02. Three-model Functional Impact violin (Ade vs SCNC) ---
print("\n--- Plot 02: Model comparison violins ---")
fig, axes_mc = plt.subplots(1, 3, figsize=(18, 6))
for mi, (model_name, fi_col, color) in enumerate([
    ('Manual Weights',      'm6A_Manual_FuncImpact',     '#2980b9'),
    ('Bottleneck',          'm6A_Bottleneck_FuncImpact', '#27ae60'),
    ('Logistic Regression', 'm6A_Functional_Impact',     '#e74c3c'),
]):
    ax = axes_mc[mi]
    va = meta.loc[idx_adeno, fi_col].values
    vb = meta.loc[idx_scnc,  fi_col].values
    parts = ax.violinplot([va, vb], positions=[0, 1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#82E0AA'); parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor('#D2B4DE'); parts['bodies'][1].set_alpha(0.7)
    style_violin(parts, ax)
    _, p_val = mannwhitneyu(va, vb, alternative='two-sided')
    y_max = max(va.max(), vb.max())
    ax.plot([0, 1], [y_max + 0.1, y_max + 0.1], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.12, f'p={p_val:.2e} {sig(p_val)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Adenocarcinoma\n(n={len(va)})', f'SCNC\n(n={len(vb)})'],
                       fontsize=10, fontweight='bold')
    ax.set_ylabel('Functional Impact' if mi == 0 else '', fontsize=11, fontweight='bold')
    ax.set_title(model_name, fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')
plt.suptitle('Functional Impact (Adenocarcinoma vs SCNC) — Three Writer Models',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '02_model_comparison_AdenoSCNC.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 02_model_comparison_AdenoSCNC.png")
plt.close()

# --- 03. Inter-model correlation heatmap ---
print("\n--- Plot 03: Inter-model correlation heatmap ---")
corr_cols = ['m6A_Manual_NetDep', 'm6A_Bottleneck_NetDep', 'm6A_Net_Deposition',
             'm6A_Manual_FuncImpact', 'm6A_Bottleneck_FuncImpact', 'm6A_Functional_Impact',
             'm6A_Oncogenic_Readout']
corr_labels = ['NetDep (Manual)', 'NetDep (Bottleneck)', 'NetDep (Logistic Regression)',
               'FuncImp (Manual)', 'FuncImp (Bottleneck)', 'FuncImp (Logistic Regression)',
               'Oncogenic Readout']
corr_matrix = meta.loc[df.index, corr_cols].corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            xticklabels=corr_labels, yticklabels=corr_labels)
ax.set_title('Inter-Model Correlation (Spearman ρ)\nManual vs Bottleneck vs Logistic Regression',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '03_model_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 03_model_correlation_heatmap.png")
plt.close()

# --- 04. Bottleneck gene frequency ---
print("\n--- Plot 04: Bottleneck gene frequency ---")
fig, ax = plt.subplots(figsize=(10, 6))
colors_bn = ['#e74c3c' if MANUAL_WEIGHTS[g] >= 2.0
             else '#f39c12' if MANUAL_WEIGHTS[g] >= 1.0 else '#95a5a6'
             for g in WRITER_GENES]
ax.bar(range(len(WRITER_GENES)), [bottleneck_counts[g] for g in WRITER_GENES],
       color=colors_bn, edgecolor='black', alpha=0.85)
for i, g in enumerate(WRITER_GENES):
    ax.text(i, bottleneck_counts[g] + 2, f'{bottleneck_counts[g]}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(WRITER_GENES)))
ax.set_xticklabels(
    [f"{g}\n({gene_roles[g].split('(')[1].rstrip(')')})" for g in WRITER_GENES],
    fontsize=9, rotation=30, ha='right',
)
ax.set_ylabel('# Patients where gene is bottleneck', fontsize=11, fontweight='bold')
ax.set_title('Bottleneck Analysis: Which Writer Limits the Complex?\n'
             'Gene with lowest activity probability per patient',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '04_bottleneck_gene_frequency.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 04_bottleneck_gene_frequency.png")
plt.close()


# #############################################################################
#  PART II — FULL ANALYSIS WITH LOGISTIC REGRESSION WEIGHTS
# #############################################################################
print("\n\n" + "=" * 80)
print("  PART II — FULL ANALYSIS (LOGISTIC REGRESSION WEIGHTS)")
print("=" * 80)

# =============================================================================
# PER-GENE ANALYSIS
# =============================================================================
print(f"\n  Luminal: n={len(idx_lum)}  |  Basal: n={len(idx_bas)}")
print(f"  Adenocarcinoma: n={len(idx_adeno)}  |  SCNC: n={len(idx_scnc)}")

print(f"\n  {'Gene':12s} {'Role':22s} {'Lum':>6s} {'Bas':>6s} {'Δ':>7s} {'p(L/B)':>10s}    "
      f"{'Adeno':>6s} {'SCNC':>6s} {'Δ':>7s} {'p(A/S)':>10s}")
print("  " + "-" * 105)

gene_results = []
for gene in gene_order:
    role = gene_roles[gene]
    zl = z_all.loc[idx_lum,   gene].values
    zb = z_all.loc[idx_bas,   gene].values
    za = z_all.loc[idx_adeno, gene].values
    zs = z_all.loc[idx_scnc,  gene].values
    _, p_lb = mannwhitneyu(zl, zb, alternative='two-sided')
    _, p_as = mannwhitneyu(za, zs, alternative='two-sided')
    print(f"  {gene:12s} {role:22s} {zl.mean():6.3f} {zb.mean():6.3f} "
          f"{zb.mean()-zl.mean():+7.3f} {p_lb:10.2e} {sig(p_lb):3s}"
          f"    {za.mean():6.3f} {zs.mean():6.3f} "
          f"{zs.mean()-za.mean():+7.3f} {p_as:10.2e} {sig(p_as):3s}")
    gene_results.append({
        'Gene': gene, 'Role': role,
        'Lum': zl.mean(), 'Bas': zb.mean(), 'p_LB': p_lb, 'sig_LB': sig(p_lb),
        'Adeno': za.mean(), 'SCNC': zs.mean(), 'p_AS': p_as, 'sig_AS': sig(p_as),
    })

gr = pd.DataFrame(gene_results)

# --- 05. Bar chart: all genes, Adeno vs SCNC ---
print("\n--- Plot 05: Per-gene bar (Adeno / SCNC) ---")
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(gene_order))
w = 0.35
ax.bar(x - w/2, gr['Adeno'], w, label='Adenocarcinoma', color='#27ae60',
       edgecolor='black', alpha=0.85)
ax.bar(x + w/2, gr['SCNC'],  w, label='SCNC', color='#8e44ad',
       edgecolor='black', alpha=0.85)
for i, row in gr.iterrows():
    y_max = max(row['Adeno'], row['SCNC'])
    by = y_max + 0.02
    ax.plot([i - w/2, i + w/2], [by, by], 'k-', lw=1)
    ax.text(i, by + 0.005, row['sig_AS'], ha='center', va='bottom',
            fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(gene_labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Mean Z-score', fontsize=12, fontweight='bold')
ax.set_title('All m6A Genes — Adenocarcinoma vs SCNC\n'
             'W=Writer  E=Eraser  R-onc=Oncogenic Reader  R-sup=Suppressive Reader',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '05_all_m6A_genes_AdenoSCNC.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 05_all_m6A_genes_AdenoSCNC.png")
plt.close()

# --- 06. Bar chart: Luminal vs Basal ---
print("\n--- Plot 06: Per-gene bar (Luminal / Basal) ---")
fig, ax = plt.subplots(figsize=(16, 7))
ax.bar(x - w/2, gr['Lum'], w, label='Luminal', color='#2980b9',
       edgecolor='black', alpha=0.85)
ax.bar(x + w/2, gr['Bas'], w, label='Basal', color='#c0392b',
       edgecolor='black', alpha=0.85)
for i, row in gr.iterrows():
    y_max = max(row['Lum'], row['Bas'])
    by = y_max + 0.02
    ax.plot([i - w/2, i + w/2], [by, by], 'k-', lw=1)
    ax.text(i, by + 0.005, row['sig_LB'], ha='center', va='bottom',
            fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(gene_labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Mean Z-score', fontsize=12, fontweight='bold')
ax.set_title('All m6A Genes — Luminal vs Basal\n'
             'W=Writer  E=Eraser  R-onc=Oncogenic Reader  R-sup=Suppressive Reader',
             fontsize=13, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '06_all_m6A_genes_LumBasal.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 06_all_m6A_genes_LumBasal.png")
plt.close()

# =============================================================================
# COMPOSITE SCORE COMPARISONS (07–09, six violins)
# =============================================================================
print("\n--- Plots 07-09: Composite score violins ---")
AXES = [
    ('m6A_Net_Deposition',    'Net m6A Deposition',  '07a', '07b'),
    ('m6A_Oncogenic_Readout', 'Oncogenic Readout',   '08a', '08b'),
    ('m6A_Functional_Impact', 'Functional Impact',   '09a', '09b'),
]
for col, label, n_as, n_lb in AXES:
    r_as = two_group_compare(col, 'histology', 'Adenocarcinoma', 'SCNC', label)
    r_lb = two_group_compare(col, 'Luminal/Basal Cluster', 'Luminal', 'Basal', label)
    print(f"\n  {label}:")
    print(f"    Adeno vs SCNC:    Ade={r_as['mean_a']:+.3f}  SCN={r_as['mean_b']:+.3f}  "
          f"Δ={r_as['delta']:+.3f}  p={r_as['p']:.2e} {r_as['sig']}")
    print(f"    Luminal vs Basal: Lum={r_lb['mean_a']:+.3f}  Bas={r_lb['mean_b']:+.3f}  "
          f"Δ={r_lb['delta']:+.3f}  p={r_lb['p']:.2e} {r_lb['sig']}")
    plot_violin(r_as, f'{label}: Adeno vs SCNC',    f'{n_as}_{col}_AdenoSCNC.png',
                color_a='#27ae60', color_b='#8e44ad')
    plot_violin(r_lb, f'{label}: Luminal vs Basal', f'{n_lb}_{col}_LumBasal.png')

# =============================================================================
# KAPLAN-MEIER SURVIVAL (10–12)
# =============================================================================
print("\n--- Plots 10-12: Kaplan-Meier survival ---")
for col, label, num in [
    ('m6A_Net_Deposition',    'Net m6A Deposition', '10'),
    ('m6A_Oncogenic_Readout', 'Oncogenic Readout',  '11'),
    ('m6A_Functional_Impact', 'Functional Impact',  '12'),
]:
    r = km_survival(col, label, f'{num}_KM_{col}.png')
    print(f"  {label:25s}  χ²={r['chi2']:.2f}  p={r['p']:.4f} {r['sig']}")

# =============================================================================
# THREE-AXIS LANDSCAPE (13)
# =============================================================================
print("\n--- Plot 13: Three-axis landscape ---")
color_map = meta.loc[df.index, 'histology'].map(
    {'Adenocarcinoma': '#27ae60', 'SCNC': '#8e44ad'}
).fillna('grey')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ax = axes[0]
ax.scatter(dd_weighted_writer, eraser_z, c=color_map, alpha=0.5, s=20, edgecolors='none')
r_sp, p_sp = spearmanr(dd_weighted_writer, eraser_z)
ax.set_xlabel('Weighted Writer z-score (Logistic Regression)', fontsize=11, fontweight='bold')
ax.set_ylabel('Eraser z-score (FTO+ALKBH5)', fontsize=11, fontweight='bold')
ax.set_title(f'Writer vs Eraser Activity\nSpearman ρ={r_sp:.3f}, p={p_sp:.2e}',
             fontsize=12, fontweight='bold')

ax = axes[1]
nd  = meta.loc[df.index, 'm6A_Net_Deposition']
orr = meta.loc[df.index, 'm6A_Oncogenic_Readout']
ax.scatter(nd, orr, c=color_map, alpha=0.5, s=20, edgecolors='none')
r_sp, p_sp = spearmanr(nd, orr)
ax.set_xlabel('Net m6A Deposition', fontsize=11, fontweight='bold')
ax.set_ylabel('Oncogenic Readout',  fontsize=11, fontweight='bold')
ax.set_title(f'Deposition vs Readout\nSpearman ρ={r_sp:.3f}, p={p_sp:.2e}',
             fontsize=12, fontweight='bold')
ax.axhline(0, color='grey', lw=0.5, ls='--')
ax.axvline(0, color='grey', lw=0.5, ls='--')

ax = axes[2]
fi_adeno = meta.loc[idx_adeno, 'm6A_Functional_Impact'].values
fi_scnc  = meta.loc[idx_scnc,  'm6A_Functional_Impact'].values
parts = ax.violinplot([fi_adeno, fi_scnc], positions=[0, 1],
                      showmeans=True, showmedians=True)
parts['bodies'][0].set_facecolor('#27ae60'); parts['bodies'][0].set_alpha(0.6)
parts['bodies'][1].set_facecolor('#8e44ad'); parts['bodies'][1].set_alpha(0.6)
style_violin(parts, ax)
_, p_fi = mannwhitneyu(fi_adeno, fi_scnc, alternative='two-sided')
ax.set_xticks([0, 1])
ax.set_xticklabels([f'Adenocarcinoma\n(n={len(fi_adeno)})',
                    f'SCNC\n(n={len(fi_scnc)})'],
                   fontsize=11, fontweight='bold')
ax.set_ylabel('Functional Impact', fontsize=11, fontweight='bold')
ax.set_title(f'Functional Impact by Histology\np={p_fi:.2e} {sig(p_fi)}',
             fontsize=12, fontweight='bold')

legend_elements = [Patch(facecolor='#27ae60', label='Adenocarcinoma'),
                   Patch(facecolor='#8e44ad', label='SCNC')]
axes[0].legend(handles=legend_elements, fontsize=10, loc='upper right')
plt.suptitle('m6A Epitranscriptomic Landscape — Logistic Regression Weights',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '13_three_axis_landscape.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 13_three_axis_landscape.png")
plt.close()

# =============================================================================
# SITE-SPECIFIC ANALYSIS (14–16)
# =============================================================================
print("\n" + "=" * 80)
print("  SITE-SPECIFIC m6A ANALYSIS")
print("=" * 80)

site_patients = meta[meta['Site_detailed'].notna()].index.intersection(df.index)
print(f"\n  Patients with site annotation: n={len(site_patients)}")

site_data = {}
print(f"\n  {'Site':15s} {'n':>5s} {'%Basal':>7s} {'%SCNC':>7s}  "
      f"{'Net Dep':>8s} {'Onco Read':>9s} {'Func Imp':>9s}")
print("  " + "-" * 70)
for site in SITE_ORDER:
    idx_site = meta[meta['Site_detailed'] == site].index.intersection(df.index)
    if len(idx_site) == 0:
        continue
    n = len(idx_site)
    pct_basal = (meta.loc[idx_site, 'Luminal/Basal Cluster'] == 'Basal').sum() / n * 100
    pct_scnc  = (meta.loc[idx_site, 'histology'] == 'SCNC').sum() / n * 100
    nd_mean   = meta.loc[idx_site, 'm6A_Net_Deposition'].mean()
    or_mean   = meta.loc[idx_site, 'm6A_Oncogenic_Readout'].mean()
    fi_mean   = meta.loc[idx_site, 'm6A_Functional_Impact'].mean()
    site_data[site] = {
        'idx': idx_site, 'n': n, 'pct_basal': pct_basal, 'pct_scnc': pct_scnc,
        'nd': meta.loc[idx_site, 'm6A_Net_Deposition'].values,
        'or': meta.loc[idx_site, 'm6A_Oncogenic_Readout'].values,
        'fi': meta.loc[idx_site, 'm6A_Functional_Impact'].values,
    }
    print(f"  {site:15s} {n:5d} {pct_basal:6.1f}% {pct_scnc:6.1f}%  "
          f"{nd_mean:+8.3f} {or_mean:+9.3f} {fi_mean:+9.3f}")

print(f"\n  Kruskal-Wallis (all sites):")
for col, label in [('m6A_Net_Deposition',    'Net Deposition'),
                    ('m6A_Oncogenic_Readout', 'Oncogenic Readout'),
                    ('m6A_Functional_Impact', 'Functional Impact')]:
    groups = [meta.loc[site_data[s]['idx'], col].values
              for s in SITE_ORDER if s in site_data]
    stat_kw, p_kw = kruskal(*groups)
    print(f"    {label:25s}  H={stat_kw:.2f}  p={p_kw:.2e} {sig(p_kw)}")

# --- Plots 14a-c: site violins ---
print("\n--- Plots 14a-c: Site violins ---")
for col, label, num in [
    ('m6A_Net_Deposition',    'Net m6A Deposition',  '14a'),
    ('m6A_Oncogenic_Readout', 'Oncogenic Readout',   '14b'),
    ('m6A_Functional_Impact', 'Functional Impact',   '14c'),
]:
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_sites = [s for s in SITE_ORDER if s in site_data]
    key_map    = {'m6A_Net_Deposition': 'nd', 'm6A_Oncogenic_Readout': 'or',
                  'm6A_Functional_Impact': 'fi'}
    data_list  = [site_data[s][key_map[col]] for s in plot_sites]
    parts = ax.violinplot(data_list, positions=range(len(plot_sites)),
                          showmeans=True, showmedians=True)
    for i, s in enumerate(plot_sites):
        parts['bodies'][i].set_facecolor(SITE_COLORS[s])
        parts['bodies'][i].set_alpha(0.65)
    style_violin(parts, ax)
    ax.set_xticks(list(range(len(plot_sites))))
    ax.set_xticklabels([f"{s}\n(n={site_data[s]['n']})" for s in plot_sites],
                       fontsize=10, fontweight='bold')
    stat_kw, p_kw = kruskal(*data_list)
    ax.set_ylabel(label, fontsize=12, fontweight='bold')
    ax.set_title(f'{label} by Biopsy Site\n'
                 f'Kruskal-Wallis H={stat_kw:.2f}, p={p_kw:.2e} {sig(p_kw)}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f'{num}_site_{col}.png'), dpi=300, bbox_inches='tight')
    print(f"  → Saved: {num}_site_{col}.png")
    plt.close()

# --- Plot 15: Per-gene heatmap by site ---
print("\n--- Plot 15: Per-gene heatmap by site ---")
heat_cols = [s for s in SITE_ORDER if s in site_data]
site_gene_matrix = []
for gene in gene_order:
    row = {'Gene': gene, 'Role': gene_roles[gene]}
    for s in heat_cols:
        row[s] = z_all.loc[site_data[s]['idx'], gene].mean()
    site_gene_matrix.append(row)

site_gene_df = pd.DataFrame(site_gene_matrix)
heat_data    = site_gene_df.set_index('Gene')[heat_cols]

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean z-score'},
            yticklabels=[f"{g} ({gene_roles[g]})" for g in heat_data.index])
ax.set_title('m6A Gene Expression by Biopsy Site\n(Mean z-score per site)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Biopsy Site', fontsize=12, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '15_site_gene_heatmap.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 15_site_gene_heatmap.png")
plt.close()

# --- Plot 16: Site-colored scatter ---
print("\n--- Plot 16: Site-colored landscape scatter ---")
fig, ax = plt.subplots(figsize=(10, 8))
for s in SITE_ORDER:
    if s not in site_data:
        continue
    ax.scatter(meta.loc[site_data[s]['idx'], 'm6A_Net_Deposition'],
               meta.loc[site_data[s]['idx'], 'm6A_Oncogenic_Readout'],
               c=SITE_COLORS[s], alpha=0.5, s=25,
               label=f"{s} (n={site_data[s]['n']})", edgecolors='none')
ax.set_xlabel('Net m6A Deposition', fontsize=12, fontweight='bold')
ax.set_ylabel('Oncogenic Readout',  fontsize=12, fontweight='bold')
ax.set_title('m6A Landscape by Biopsy Site\nNet Deposition vs Oncogenic Readout',
             fontsize=14, fontweight='bold', pad=12)
ax.axhline(0, color='grey', lw=0.5, ls='--')
ax.axvline(0, color='grey', lw=0.5, ls='--')
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '16_site_landscape_scatter.png'), dpi=300, bbox_inches='tight')
print(f"  → Saved: 16_site_landscape_scatter.png")
plt.close()

# =============================================================================
# LIVER DEEP DIVE (17–24)
# =============================================================================
print("\n" + "-" * 80)
print("  LIVER METASTASES — DEEP DIVE")
print("-" * 80)

if 'Liver' in site_data:
    idx_liver     = site_data['Liver']['idx']
    idx_non_liver = (meta[(meta['Site_detailed'].notna()) &
                          (meta['Site_detailed'] != 'Liver')]
                     .index.intersection(df.index))

    print(f"\n  Liver: n={len(idx_liver)}  |  Non-Liver: n={len(idx_non_liver)}")
    print(f"  Liver Basal: "
          f"{(meta.loc[idx_liver, 'Luminal/Basal Cluster'] == 'Basal').sum()}/{len(idx_liver)} "
          f"({(meta.loc[idx_liver, 'Luminal/Basal Cluster'] == 'Basal').sum()/len(idx_liver)*100:.0f}%)")
    print(f"  Liver SCNC:  "
          f"{(meta.loc[idx_liver, 'histology'] == 'SCNC').sum()}/{len(idx_liver)} "
          f"({(meta.loc[idx_liver, 'histology'] == 'SCNC').sum()/len(idx_liver)*100:.0f}%)")

    print(f"\n  {'Gene':12s} {'Role':22s} {'Liver':>7s} {'Other':>7s} {'Δ':>7s} {'p':>10s}")
    print("  " + "-" * 70)
    for gene in gene_order:
        vl = z_all.loc[idx_liver,     gene].values
        vo = z_all.loc[idx_non_liver, gene].values
        _, p_val = mannwhitneyu(vl, vo, alternative='two-sided')
        print(f"  {gene:12s} {gene_roles[gene]:22s} {vl.mean():+7.3f} {vo.mean():+7.3f} "
              f"{vl.mean()-vo.mean():+7.3f} {p_val:10.2e} {sig(p_val):3s}")

    print(f"\n  Composite Scores:")
    for col, label in [('m6A_Net_Deposition',    'Net Deposition'),
                        ('m6A_Oncogenic_Readout', 'Oncogenic Readout'),
                        ('m6A_Functional_Impact', 'Functional Impact')]:
        vl = meta.loc[idx_liver,     col].values
        vo = meta.loc[idx_non_liver, col].values
        _, p_val = mannwhitneyu(vl, vo, alternative='two-sided')
        print(f"    {label:25s}  Liver={vl.mean():+.3f}  Other={vo.mean():+.3f}  "
              f"Δ={vl.mean()-vo.mean():+.3f}  p={p_val:.2e} {sig(p_val)}")

    # --- Plot 17: Liver vs Non-Liver Functional Impact violin ---
    print("\n--- Plot 17: Liver vs Non-Liver Functional Impact ---")
    va_nl = meta.loc[idx_non_liver, 'm6A_Functional_Impact'].values
    vb_li = meta.loc[idx_liver,     'm6A_Functional_Impact'].values
    _, p_lv = mannwhitneyu(va_nl, vb_li, alternative='two-sided')
    r_liver = {
        'label': 'Functional Impact',
        'group_a': 'Non-Liver', 'group_b': 'Liver',
        'n_a': len(idx_non_liver), 'n_b': len(idx_liver),
        'mean_a': va_nl.mean(), 'mean_b': vb_li.mean(),
        'delta': vb_li.mean() - va_nl.mean(),
        'p': p_lv, 'sig': sig(p_lv),
        'va': va_nl, 'vb': vb_li,
    }
    plot_violin(r_liver, 'Functional Impact: Liver vs Non-Liver Metastases',
                '17_Liver_vs_NonLiver_FunctionalImpact.png',
                color_a='#2980b9', color_b='#e74c3c')

    # Within-Liver histology
    idx_liver_adeno = (meta.loc[idx_liver][meta.loc[idx_liver, 'histology'] == 'Adenocarcinoma'].index)
    idx_liver_scnc  = (meta.loc[idx_liver][meta.loc[idx_liver, 'histology'] == 'SCNC'].index)
    print(f"\n  Within Liver — Ade (n={len(idx_liver_adeno)}) vs SCNC (n={len(idx_liver_scnc)}):")
    for col, label in [('m6A_Net_Deposition',    'Net Deposition'),
                        ('m6A_Oncogenic_Readout', 'Oncogenic Readout'),
                        ('m6A_Functional_Impact', 'Functional Impact')]:
        va = meta.loc[idx_liver_adeno, col].values
        vs = meta.loc[idx_liver_scnc,  col].values
        if len(va) > 0 and len(vs) > 0:
            _, p_val = mannwhitneyu(va, vs, alternative='two-sided')
            print(f"    {label:25s}  Ade={va.mean():+.3f}  SCNC={vs.mean():+.3f}  "
                  f"Δ={vs.mean()-va.mean():+.3f}  p={p_val:.2e} {sig(p_val)}")

    # --- Plot 18: Liver vs Non-Liver per-gene bar ---
    print("\n--- Plot 18: Liver vs Non-Liver per-gene bar ---")
    liver_gene_data = []
    for gene in gene_order:
        vl = z_all.loc[idx_liver,     gene].mean()
        vo = z_all.loc[idx_non_liver, gene].mean()
        _, p_val = mannwhitneyu(z_all.loc[idx_liver, gene].values,
                                z_all.loc[idx_non_liver, gene].values,
                                alternative='two-sided')
        liver_gene_data.append({'Gene': gene, 'Liver': vl,
                                  'Non-Liver': vo, 'p': p_val, 'sig': sig(p_val)})
    lgd = pd.DataFrame(liver_gene_data)

    n_nl_str = f'Non-Liver (n={len(idx_non_liver)})'
    n_li_str = f'Liver (n={len(idx_liver)})'
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - w/2, lgd['Liver'],     w, label=n_li_str, color='#e74c3c',
           edgecolor='black', alpha=0.85)
    ax.bar(x + w/2, lgd['Non-Liver'], w, label=n_nl_str, color='#3498db',
           edgecolor='black', alpha=0.85)
    for i, row in lgd.iterrows():
        y_max = max(row['Liver'], row['Non-Liver'])
        by = y_max + 0.04
        ax.plot([i - w/2, i + w/2], [by, by], 'k-', lw=1)
        ax.text(i, by + 0.01, row['sig'], ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gene_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Mean Z-score', fontsize=12, fontweight='bold')
    ax.set_title('All m6A Genes — Liver vs Non-Liver Metastases',
                 fontsize=13, fontweight='bold', pad=12)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '18_liver_vs_nonliver_all_genes.png'),
                dpi=300, bbox_inches='tight')
    print(f"  → Saved: 18_liver_vs_nonliver_all_genes.png")
    plt.close()

    # --- Plot 19: Site composition stacked bar ---
    print("\n--- Plot 19: Site composition stacked bar ---")
    fig, axes_comp = plt.subplots(1, 2, figsize=(16, 6))
    plot_sites = [s for s in SITE_ORDER if s in site_data]
    x_sites    = np.arange(len(plot_sites))

    ax = axes_comp[0]
    pct_basal  = [site_data[s]['pct_basal'] for s in plot_sites]
    pct_luminal = [100 - b for b in pct_basal]
    ax.bar(x_sites, pct_luminal, color='#2980b9', edgecolor='black', label='Luminal')
    ax.bar(x_sites, pct_basal, bottom=pct_luminal, color='#c0392b', edgecolor='black', label='Basal')
    for i, s in enumerate(plot_sites):
        ax.text(i, 50, f"n={site_data[s]['n']}", ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    ax.set_xticks(x_sites)
    ax.set_xticklabels(plot_sites, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('% of Patients', fontsize=11, fontweight='bold')
    ax.set_title('Molecular Subtype by Biopsy Site', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 105)

    ax = axes_comp[1]
    pct_scnc = [site_data[s]['pct_scnc'] for s in plot_sites]
    pct_adeno = [100 - sc for sc in pct_scnc]
    ax.bar(x_sites, pct_adeno, color='#27ae60', edgecolor='black', label='Adenocarcinoma')
    ax.bar(x_sites, pct_scnc, bottom=pct_adeno, color='#8e44ad', edgecolor='black', label='SCNC')
    for i, s in enumerate(plot_sites):
        ax.text(i, 50, f"n={site_data[s]['n']}", ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    ax.set_xticks(x_sites)
    ax.set_xticklabels(plot_sites, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('% of Patients', fontsize=11, fontweight='bold')
    ax.set_title('Histology by Biopsy Site', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 105)

    plt.suptitle('Biopsy Site Composition — Subtype & Histology Context',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '19_site_composition.png'), dpi=300, bbox_inches='tight')
    print(f"  → Saved: 19_site_composition.png")
    plt.close()

    # --- Plot 20: IGF2BP family across sites ---
    print("\n--- Plot 20: IGF2BP by site ---")
    igf2bp_genes = ['IGF2BP1', 'IGF2BP2', 'IGF2BP3']
    fig, axes_igf = plt.subplots(1, 3, figsize=(18, 6))
    for gi, gene in enumerate(igf2bp_genes):
        ax = axes_igf[gi]
        bp_data, bp_labels, bp_colors = [], [], []
        for s in SITE_ORDER:
            if s not in site_data:
                continue
            bp_data.append(z_all.loc[site_data[s]['idx'], gene].values)
            bp_labels.append(f"{s}\n(n={site_data[s]['n']})")
            bp_colors.append(SITE_COLORS[s])
        parts = ax.violinplot(bp_data, positions=range(len(bp_data)),
                              showmeans=True, showmedians=True)
        for i, c in enumerate(bp_colors):
            parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.65)
        style_violin(parts, ax, legend=(gi == 0))
        ax.set_xticks(range(len(bp_data)))
        ax.set_xticklabels(bp_labels, fontsize=8, fontweight='bold')
        ax.set_ylabel('Z-score', fontsize=11, fontweight='bold')
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        stat_kw, p_kw = kruskal(*bp_data)
        ax.set_title(f'{gene}\nKW p={p_kw:.2e} {sig(p_kw)}', fontsize=12, fontweight='bold')
    plt.suptitle('IGF2BP Oncogenic Readers by Biopsy Site',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '20_IGF2BP_by_site.png'), dpi=300, bbox_inches='tight')
    print(f"  → Saved: 20_IGF2BP_by_site.png")
    plt.close()

    # --- Plot 21: Reader balance by site ---
    print("\n--- Plot 21: Reader balance by site ---")
    fig, ax = plt.subplots(figsize=(12, 7))
    w_bar = 0.35
    onco_means, supp_means = [], []
    for s in plot_sites:
        onco_means.append(z_all.loc[site_data[s]['idx'], READER_ONCOGENIC].mean(axis=1).mean())
        supp_means.append(z_all.loc[site_data[s]['idx'], READER_SUPPRESSIVE].mean(axis=1).mean())
    ax.bar(x_sites - w_bar/2, onco_means, w_bar,
           label='Oncogenic Readers\n(IGF2BP1/2/3, YTHDF1)',
           color='#e74c3c', edgecolor='black', alpha=0.85)
    ax.bar(x_sites + w_bar/2, supp_means, w_bar,
           label='Suppressive Readers\n(YTHDF2/3, YTHDC1/2)',
           color='#3498db', edgecolor='black', alpha=0.85)
    ax.set_xticks(x_sites)
    ax.set_xticklabels([f"{s}\n(n={site_data[s]['n']})" for s in plot_sites],
                       fontsize=9, fontweight='bold')
    ax.set_ylabel('Mean Z-score', fontsize=12, fontweight='bold')
    ax.set_title('m6A Reader Balance by Biopsy Site\nOncogenic vs Suppressive Readers',
                 fontsize=13, fontweight='bold', pad=12)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '21_reader_balance_by_site.png'), dpi=300, bbox_inches='tight')
    print(f"  → Saved: 21_reader_balance_by_site.png")
    plt.close()

    # --- Plot 22: Histology convergence in liver ---
    print("\n--- Plot 22: Histology convergence in liver ---")
    idx_nl_adeno = meta[(meta['Site_detailed'].notna()) &
                        (meta['Site_detailed'] != 'Liver') &
                        (meta['histology'] == 'Adenocarcinoma')].index.intersection(df.index)
    idx_nl_scnc  = meta[(meta['Site_detailed'].notna()) &
                        (meta['Site_detailed'] != 'Liver') &
                        (meta['histology'] == 'SCNC')].index.intersection(df.index)
    idx_li_adeno = meta[(meta['Site_detailed'] == 'Liver') &
                        (meta['histology'] == 'Adenocarcinoma')].index.intersection(df.index)
    idx_li_scnc  = meta[(meta['Site_detailed'] == 'Liver') &
                        (meta['histology'] == 'SCNC')].index.intersection(df.index)

    groups       = [meta.loc[idx_nl_adeno, 'm6A_Oncogenic_Readout'].values,
                    meta.loc[idx_nl_scnc,  'm6A_Oncogenic_Readout'].values,
                    meta.loc[idx_li_adeno, 'm6A_Oncogenic_Readout'].values,
                    meta.loc[idx_li_scnc,  'm6A_Oncogenic_Readout'].values]
    group_labels_22 = [f'Non-Liver\nAdeno\n(n={len(idx_nl_adeno)})',
                       f'Non-Liver\nSCNC\n(n={len(idx_nl_scnc)})',
                       f'Liver\nAdeno\n(n={len(idx_li_adeno)})',
                       f'Liver\nSCNC\n(n={len(idx_li_scnc)})']
    group_colors_22 = ['#82E0AA', '#D2B4DE', '#27ae60', '#8e44ad']

    fig, ax = plt.subplots(figsize=(10, 7))
    valid = [(g, l, c, p) for g, l, c, p in
             zip(groups, group_labels_22, group_colors_22, [0, 1, 3, 4]) if len(g) > 0]
    if len(valid) >= 2:
        v_groups, v_labels, v_colors, v_positions = zip(*valid)
        parts = ax.violinplot(v_groups, positions=v_positions,
                              showmeans=True, showmedians=True)
        for i, c in enumerate(v_colors):
            parts['bodies'][i].set_facecolor(c); parts['bodies'][i].set_alpha(0.7)
        style_violin(parts, ax)
        ax.set_xticks(list(v_positions))
        ax.set_xticklabels(v_labels, fontsize=10, fontweight='bold')
        ax.set_ylabel('Oncogenic Readout', fontsize=12, fontweight='bold')
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        if len(groups[0]) > 0 and len(groups[1]) > 0:
            _, p_nl = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            y_nl = max(groups[0].max(), groups[1].max()) + 0.15
            ax.plot([0, 1], [y_nl, y_nl], 'k-', lw=1.2)
            ax.text(0.5, y_nl + 0.03, f'p={p_nl:.2e} {sig(p_nl)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        if len(groups[2]) > 0 and len(groups[3]) > 0:
            _, p_li = mannwhitneyu(groups[2], groups[3], alternative='two-sided')
            y_li = max(groups[2].max(), groups[3].max()) + 0.15
            ax.plot([3, 4], [y_li, y_li], 'k-', lw=1.2)
            ax.text(3.5, y_li + 0.03, f'p={p_li:.2e} {sig(p_li)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        if len(groups[1]) > 0 and len(groups[3]) > 0:
            _, p_scnc_site = mannwhitneyu(groups[1], groups[3], alternative='two-sided')
            y_cross = max(g.max() for g in v_groups) + 0.5
            ax.plot([1, 4], [y_cross, y_cross], 'k-', lw=1.2)
            ax.text(2.5, y_cross + 0.03, f'p={p_scnc_site:.2e} {sig(p_scnc_site)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#8e44ad')
        if len(groups[0]) > 0 and len(groups[2]) > 0:
            _, p_adeno_site = mannwhitneyu(groups[0], groups[2], alternative='two-sided')
            y_cross2 = max(g.max() for g in v_groups) + 0.85
            ax.plot([0, 3], [y_cross2, y_cross2], 'k-', lw=1.2)
            ax.text(1.5, y_cross2 + 0.03, f'p={p_adeno_site:.2e} {sig(p_adeno_site)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')
        ax.axvline(2, color='grey', lw=1, ls=':')
        ax.text(-0.3, ax.get_ylim()[1] * 0.95, 'Non-Liver',
                fontsize=11, fontweight='bold', color='grey')
        ax.text(2.7, ax.get_ylim()[1] * 0.95, 'Liver',
                fontsize=11, fontweight='bold', color='grey')
        ax.set_title('Oncogenic Readout: Histology Convergence in Liver\n'
                     'Adenocarcinoma in liver adopts SCNC-like oncogenic readout',
                     fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, '22_liver_histology_convergence.png'),
                    dpi=300, bbox_inches='tight')
        print(f"  → Saved: 22_liver_histology_convergence.png")
        plt.close()

    # --- Plot 23: Oncogenic Readout Liver vs each site ---
    print("\n--- Plot 23: Liver vs each other site ---")
    other_sites = [s for s in SITE_ORDER if s in site_data and s != 'Liver']
    fig, axes_vs = plt.subplots(1, len(other_sites), figsize=(4 * len(other_sites), 6))
    if len(other_sites) == 1:
        axes_vs = [axes_vs]
    for si, comp_site in enumerate(other_sites):
        ax = axes_vs[si]
        v_liver = meta.loc[idx_liver,                    'm6A_Oncogenic_Readout'].values
        v_comp  = meta.loc[site_data[comp_site]['idx'],  'm6A_Oncogenic_Readout'].values
        parts = ax.violinplot([v_comp, v_liver], positions=[0, 1],
                              showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(SITE_COLORS[comp_site]); parts['bodies'][0].set_alpha(0.65)
        parts['bodies'][1].set_facecolor(SITE_COLORS['Liver']);    parts['bodies'][1].set_alpha(0.65)
        style_violin(parts, ax, legend=(si == 0))
        _, p_val = mannwhitneyu(v_comp, v_liver, alternative='two-sided')
        y_max = max(v_comp.max(), v_liver.max())
        ax.plot([0, 1], [y_max + 0.15, y_max + 0.15], 'k-', lw=1.2)
        ax.text(0.5, y_max + 0.17, f'p={p_val:.2e}\n{sig(p_val)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(ax.get_ylim()[0], y_max + 0.65)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{comp_site}\n(n={site_data[comp_site]['n']})",
                            f"Liver\n(n={site_data['Liver']['n']})"],
                           fontsize=9, fontweight='bold')
        ax.set_ylabel('Oncogenic Readout' if si == 0 else '', fontsize=11, fontweight='bold')
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_title(f'vs {comp_site}', fontsize=11, fontweight='bold', pad=12)
    fig.suptitle('Oncogenic Readout: Liver vs Each Site', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(OUTDIR, '23_liver_vs_each_site_OncReadout.png'),
                dpi=300, bbox_inches='tight')
    print(f"  → Saved: 23_liver_vs_each_site_OncReadout.png")
    plt.close()

    # --- Plot 24: Liver effect stratified + OLS ---
    print("\n--- Plot 24: Liver effect stratified by subtype ---")
    strat_results = []
    for label_st, col_st, val_st in [
        ('Adenocarcinoma only', 'histology',            'Adenocarcinoma'),
        ('SCNC only',           'histology',            'SCNC'),
        ('Luminal only',        'Luminal/Basal Cluster', 'Luminal'),
        ('Basal only',          'Luminal/Basal Cluster', 'Basal'),
    ]:
        idx_sub  = meta[meta[col_st] == val_st].index.intersection(df.index)
        idx_sub_l = idx_sub.intersection(idx_liver)
        idx_sub_o = idx_sub.difference(idx_liver)
        if len(idx_sub_l) >= 5 and len(idx_sub_o) >= 5:
            v_l = meta.loc[idx_sub_l, 'm6A_Oncogenic_Readout'].values
            v_o = meta.loc[idx_sub_o, 'm6A_Oncogenic_Readout'].values
            _, p_val = mannwhitneyu(v_l, v_o, alternative='two-sided')
            strat_results.append({
                'Stratum': label_st,
                'n_liver': len(idx_sub_l), 'n_other': len(idx_sub_o),
                'mean_liver': np.mean(v_l), 'mean_other': np.mean(v_o),
                'delta': np.mean(v_l) - np.mean(v_o),
                'p': p_val, 'sig': sig(p_val),
                'v_liver': v_l, 'v_other': v_o,
            })

    # Multivariable OLS
    idx_reg = meta.index.intersection(df.index)
    idx_reg = meta.loc[idx_reg].dropna(subset=['histology', 'Luminal/Basal Cluster']).index
    idx_reg = idx_reg[meta.loc[idx_reg, 'histology'].isin(['Adenocarcinoma', 'SCNC'])]
    reg_df  = pd.DataFrame({
        'OncReadout': meta.loc[idx_reg, 'm6A_Oncogenic_Readout'],
        'is_liver': meta.loc[idx_reg, 'Site_detailed'].str.lower().str.contains('liver', na=False).astype(int),
        'is_SCNC':  (meta.loc[idx_reg, 'histology'] == 'SCNC').astype(int),
        'is_Basal': (meta.loc[idx_reg, 'Luminal/Basal Cluster'] == 'Basal').astype(int),
    })
    X_reg = sm.add_constant(reg_df[['is_liver', 'is_SCNC', 'is_Basal']])
    model = sm.OLS(reg_df['OncReadout'], X_reg).fit()
    print(f"\n  Multivariable OLS: OncReadout ~ is_liver + is_SCNC + is_Basal")
    print(f"  R² = {model.rsquared:.3f},  n = {len(reg_df)}")
    for var in ['is_liver', 'is_SCNC', 'is_Basal']:
        print(f"    {var:12s} β={model.params[var]:+.3f}  p={model.pvalues[var]:.2e}  {sig(model.pvalues[var])}")

    plot_strats = [r for r in strat_results if r['n_liver'] >= 5]
    if plot_strats:
        fig, axes_st = plt.subplots(1, len(plot_strats), figsize=(5 * len(plot_strats), 6))
        if len(plot_strats) == 1:
            axes_st = [axes_st]
        for si, r in enumerate(plot_strats):
            ax = axes_st[si]
            parts = ax.violinplot([r['v_other'], r['v_liver']], positions=[0, 1],
                                  showmeans=True, showmedians=True)
            parts['bodies'][0].set_facecolor('#95a5a6'); parts['bodies'][0].set_alpha(0.65)
            parts['bodies'][1].set_facecolor(SITE_COLORS['Liver']); parts['bodies'][1].set_alpha(0.65)
            style_violin(parts, ax, legend=(si == 0))
            y_max = max(r['v_other'].max(), r['v_liver'].max())
            ax.plot([0, 1], [y_max + 0.15, y_max + 0.15], 'k-', lw=1.2)
            ax.text(0.5, y_max + 0.17, f'p={r["p"]:.2e}\n{r["sig"]}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylim(ax.get_ylim()[0], y_max + 0.65)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Non-Liver\n(n={r['n_other']})",
                                f"Liver\n(n={r['n_liver']})"],
                               fontsize=9, fontweight='bold')
            ax.set_ylabel('Oncogenic Readout' if si == 0 else '', fontsize=11, fontweight='bold')
            ax.axhline(0, color='grey', lw=0.8, ls='--')
            ax.set_title(r['Stratum'], fontsize=11, fontweight='bold', pad=12)
        fig.suptitle('Oncogenic Readout: Liver vs Non-Liver (Stratified by Subtype)\n'
                     f'OLS: liver coef={model.params["is_liver"]:+.3f}, '
                     f'p={model.pvalues["is_liver"]:.2e}',
                     fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(OUTDIR, '24_liver_effect_stratified.png'),
                    dpi=300, bbox_inches='tight')
        print(f"  → Saved: 24_liver_effect_stratified.png")
        plt.close()

    # ==========================================================================
    # RBM15 vs RBM15B PARALOG COORDINATION (Plot 25)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  RBM15 vs RBM15B: PARALOG COORDINATION")
    print("=" * 80)

    rbm15_z  = z_all['RBM15'].values
    rbm15b_z = z_all['RBM15B'].values

    pearson_r,  pearson_p  = pearsonr(rbm15_z, rbm15b_z)
    spearman_r, spearman_p = spearmanr(rbm15_z, rbm15b_z)

    print(f"  Pearson  r = {pearson_r:+.3f},  p = {pearson_p:.2e}  {sig(pearson_p)}")
    print(f"  Spearman r = {spearman_r:+.3f}, p = {spearman_p:.2e}  {sig(spearman_p)}")

    high_thresh = np.percentile([rbm15_z, rbm15b_z], 75)
    high_rbm15  = (rbm15_z  >= high_thresh).astype(int)
    high_rbm15b = (rbm15b_z >= high_thresh).astype(int)
    both_high   = int(((high_rbm15 == 1) & (high_rbm15b == 1)).sum())
    rbm15_only  = int(((high_rbm15 == 1) & (high_rbm15b == 0)).sum())
    rbm15b_only = int(((high_rbm15 == 0) & (high_rbm15b == 1)).sum())
    both_low    = int(((high_rbm15 == 0) & (high_rbm15b == 0)).sum())
    oddsratio, fisher_p = fisher_exact([[both_high, rbm15_only],
                                        [rbm15b_only, both_low]])
    print(f"  Fisher's exact OR={oddsratio:.3f}, p={fisher_p:.2e}  {sig(fisher_p)}")

    print("\n--- Plot 25: RBM15 vs RBM15B scatter ---")
    fig, ax = plt.subplots(figsize=(9, 8))
    for hist, color, marker in [('Adenocarcinoma', '#2980b9', 'o'), ('SCNC', '#c0392b', 's')]:
        idx_h = meta[meta['histology'] == hist].index.intersection(df.index)
        ax.scatter(z_all.loc[idx_h, 'RBM15'], z_all.loc[idx_h, 'RBM15B'],
                   alpha=0.5, s=50, label=hist, color=color, marker=marker,
                   edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.axhspan(high_thresh, ax.get_ylim()[1], color='green', alpha=0.05, label='High RBM15B')
    ax.axvspan(high_thresh, ax.get_xlim()[1], color='orange', alpha=0.05, label='High RBM15')
    ax.set_xlabel('RBM15 z-score',  fontsize=12, fontweight='bold')
    ax.set_ylabel('RBM15B z-score', fontsize=12, fontweight='bold')
    ax.set_title(f'RBM15 vs RBM15B Expression:\n'
                 f'Pearson r={pearson_r:+.3f} (p={pearson_p:.2e})\n'
                 f'Spearman r={spearman_r:+.3f} (p={spearman_p:.2e})',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '25_RBM15_vs_RBM15B_coordination.png'),
                dpi=300, bbox_inches='tight')
    print(f"  → Saved: 25_RBM15_vs_RBM15B_coordination.png")
    plt.close()

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("  SUMMARY TABLE (LOGISTIC REGRESSION WEIGHTS)")
print("=" * 80)

summary = []
for col, label in [('m6A_Net_Deposition',    'Net Deposition'),
                    ('m6A_Oncogenic_Readout', 'Oncogenic Readout'),
                    ('m6A_Functional_Impact', 'Functional Impact')]:
    r_lb = two_group_compare(col, 'Luminal/Basal Cluster', 'Luminal', 'Basal', label)
    r_as = two_group_compare(col, 'histology', 'Adenocarcinoma', 'SCNC', label)
    surv = meta[['surv_months', 'vital_status']].dropna()
    surv = surv[surv.index.isin(df.index)]
    sv   = meta.loc[surv.index, col].dropna()
    surv = surv.loc[sv.index].copy()
    med  = sv.median()
    hi   = surv[sv >= med]; lo = surv[sv < med]
    lr   = logrank_test(hi['surv_months'], lo['surv_months'],
                        hi['vital_status'], lo['vital_status'])
    site_groups = [meta.loc[site_data[s]['idx'], col].values
                   for s in SITE_ORDER if s in site_data]
    _, p_site = kruskal(*site_groups)
    summary.append({
        'Score': label,
        'p (Ade/SCNC)': f"{r_as['p']:.2e}", 'A/S': r_as['sig'],
        'p (Lum/Bas)':  f"{r_lb['p']:.2e}", 'L/B': r_lb['sig'],
        'p (Survival)': f"{lr.p_value:.4f}", 'Surv': sig(lr.p_value),
        'p (Sites)':    f"{p_site:.2e}",     'Site': sig(p_site),
        '# Sig': sum(1 for s in [r_as['sig'], r_lb['sig'],
                                  sig(lr.p_value), sig(p_site)] if s != 'ns'),
    })
print("\n" + pd.DataFrame(summary).to_string(index=False))

print("\n\n" + "=" * 80)
print("  ALL PLOTS SAVED")
print("=" * 80)
for i, f in enumerate([
    '01_weight_comparison.png',
    '02_model_comparison_AdenoSCNC.png',
    '02b_axis_weight_optimization.png',
    '03_model_correlation_heatmap.png',
    '04_bottleneck_gene_frequency.png',
    '05_all_m6A_genes_AdenoSCNC.png',
    '06_all_m6A_genes_LumBasal.png',
    '07a_m6A_Net_Deposition_AdenoSCNC.png',
    '07b_m6A_Net_Deposition_LumBasal.png',
    '08a_m6A_Oncogenic_Readout_AdenoSCNC.png',
    '08b_m6A_Oncogenic_Readout_LumBasal.png',
    '09a_m6A_Functional_Impact_AdenoSCNC.png',
    '09b_m6A_Functional_Impact_LumBasal.png',
    '10_KM_m6A_Net_Deposition.png',
    '11_KM_m6A_Oncogenic_Readout.png',
    '12_KM_m6A_Functional_Impact.png',
    '13_three_axis_landscape.png',
    '14a_site_m6A_Net_Deposition.png',
    '14b_site_m6A_Oncogenic_Readout.png',
    '14c_site_m6A_Functional_Impact.png',
    '15_site_gene_heatmap.png',
    '16_site_landscape_scatter.png',
    '17_Liver_vs_NonLiver_FunctionalImpact.png',
    '18_liver_vs_nonliver_all_genes.png',
    '19_site_composition.png',
    '20_IGF2BP_by_site.png',
    '21_reader_balance_by_site.png',
    '22_liver_histology_convergence.png',
    '23_liver_vs_each_site_OncReadout.png',
    '24_liver_effect_stratified.png',
    '25_RBM15_vs_RBM15B_coordination.png',
], 1):
    print(f"  {i:2d}. {f}")
print(f"\n  Output directory: {OUTDIR}")
print("=" * 80)
