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

if 'Liver' in site_data:
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
    '02_model_comparison_AdenoSCNC.png',
    '05_all_m6A_genes_AdenoSCNC.png',
    '13_three_axis_landscape.png',
    '25_RBM15_vs_RBM15B_coordination.png',
], 1):
    print(f"  {i:2d}. {f}")
print(f"\n  Output directory: {OUTDIR}")
print("=" * 80)
