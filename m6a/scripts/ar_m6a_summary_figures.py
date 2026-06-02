#!/usr/bin/env python3
"""
ar_m6a_summary_figures.py — Six publication-ready figures for AR × m6A coordination.

Narrative arc: AR×m6A hook (Primary→mCRPC) → disease progression context →
within-mCRPC mechanism → clinical anchor.

Figure 1 — AR mRNA expression × m6A FI: per-gene + aggregate scatters (Primary → mCRPC)
Figure 2 — AR Activity Score × m6A FI: per-gene + aggregate scatters (Primary → mCRPC)
Figure 3 — AR Activity + m6A FI dual trajectory (cross-cohort, 2-panel stacked)
Figure 4 — Within-cohort AR × m6A coupling forest (sign-flip thesis, with 95% CI)
Figure 5 — Per-gene m6A × ARS in mCRPC (Adeno-only + L/B deconfounded) + gene scatters
Figure 6 — Mediation (ARS → RBM15B → m6A FI) + Survival (KM + Cox) merged 2×2

Output: plots_ar_m6a_summary/

Usage:
    micromamba run -n rnaseq python ar_m6a_summary_figures.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.stats import spearmanr, kruskal, rankdata as _rankdata, pearsonr as _pearsonr
from scipy.stats import t as _t_dist, mannwhitneyu, rankdata as _sp_rankdata
from sklearn.linear_model import LogisticRegression as _LR
from lifelines import KaplanMeierFitter as _KMF, CoxPHFitter as _CPH
from lifelines.statistics import logrank_test as _logrank_test

warnings.filterwarnings('ignore')
plt.rcParams['savefig.dpi'] = 300

# ── m6a package imports ───────────────────────────────────────────────────────
from m6a.config import OUTDIR_AR_SUMMARY as OUTDIR
from m6a.genes import (
    AR_TARGET_GENES,
    ALL_M6A_GENES as CROSS_COHORT_M6A_GENES,
    MCRPC_ALL_GENES    as ALL_M6A_GENES,
    MCRPC_WRITER_GENES as WRITER_GENES,
    MCRPC_ERASER_GENES as ERASER_GENES,
    MCRPC_READER_ONCOGENIC  as READER_ONCOGENIC,
    MCRPC_READER_SUPPRESSIVE as READER_SUPPRESSIVE,
    MCRPC_GENE_ROLES   as gene_roles,
    MCRPC_GENE_ORDER   as gene_order,
    MCRPC_MANUAL_WEIGHTS as MANUAL_WEIGHTS,
    # Cross-cohort sublists (for CC_GENE_ORDER)
    WRITER_GENES        as CC_WRITER_GENES,
    READER_ONCOGENIC    as CC_READER_ONCOGENIC,
    READER_SUPPRESSIVE  as CC_READER_SUPPRESSIVE,
)

# 22-gene cross-cohort gene order and roles (adds METTL16, HNRNPA2B1, ELAVL1, HNRNPC, FMR1)
CC_GENE_ORDER = CC_WRITER_GENES + ERASER_GENES + CC_READER_ONCOGENIC + CC_READER_SUPPRESSIVE
CC_GENE_ROLES = dict(gene_roles)
CC_GENE_ROLES.update({
    'METTL16':   'Writer (SAM-MTase)',
    'HNRNPA2B1': 'Reader (Oncogenic)',
    'ELAVL1':    'Reader (Oncogenic)',
    'HNRNPC':    'Reader (Suppressive)',
    'FMR1':      'Reader (Suppressive)',
})
from m6a.stats import sig
from m6a.normalization import zscore_normalize, percentile_rank_matrix
from m6a.scoring import compute_axes
from m6a.plotting import style_violin
from m6a.data.loaders import (
    load_mcrpc, load_tcga, load_gtex, load_adj_normal, load_mcspc, load_darana,
    build_common_universe,
)

os.makedirs(OUTDIR, exist_ok=True)

# Cross-cohort palette (consistent with other scripts)
CC_LABELS      = ['Normal\n(GTEx)', 'Adj Normal\n(TCGA)', 'Primary PCa\n(TCGA)',
                  'mCSPC\n(GSE221601)', 'mCRPC\nAdeno', 'mCRPC\nSCNC']
CC_LABELS_FLAT = ['Normal (GTEx)', 'Adj Normal (TCGA)', 'Primary PCa (TCGA)',
                  'mCSPC (GSE221601)', 'mCRPC Adeno', 'mCRPC SCNC']
CC_COLORS      = ['#27ae60', '#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#c0392b']

print("=" * 80)
print("  AR × m6A SUMMARY FIGURES")
print("=" * 80)

# =============================================================================
# SECTION A — CROSS-COHORT DATA LOADING & SCORING
# =============================================================================
print("\n[A] Loading cross-cohort data ...")

df_mcrpc, meta_mcrpc = load_mcrpc()
df_tcga, _           = load_tcga()
gtex_expr, _         = load_gtex()
adjn_expr, _         = load_adj_normal()
mhspc_expr, _        = load_mcspc()

idx_mcrpc_adeno = meta_mcrpc[meta_mcrpc['histology'] == 'Adenocarcinoma'].index.intersection(df_mcrpc.index)
idx_mcrpc_scnc  = meta_mcrpc[meta_mcrpc['histology'] == 'SCNC'].index.intersection(df_mcrpc.index)

# Common gene universe + percentile-rank normalization
all_expr_dfs = [df_mcrpc, df_tcga, gtex_expr, adjn_expr, mhspc_expr]
common       = build_common_universe(all_expr_dfs)
ar_avail     = [g for g in AR_TARGET_GENES       if g in common]
m6a_avail    = [g for g in CROSS_COHORT_M6A_GENES if g in common]

def percentile_rank_df(expr_df):
    return percentile_rank_matrix(expr_df[common].fillna(0.0))

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
    avail = [g for g in panel if g in pct_df.columns]
    return pct_df[avail].mean(axis=1)

def m6a_fi(pct_df):
    avail_m6a = [g for g in m6a_avail if g in pct_df.columns]
    _, _, fi = compute_axes(pct_df[avail_m6a])
    return fi

ars_groups = [ars_from_panel(pct[k], ar_avail) for k in cohort_keys]
fi_groups  = [m6a_fi(pct[k])                   for k in cohort_keys]
ns_cc      = [len(g) for g in ars_groups]

print(f"  Cohorts loaded: {', '.join(f'{k}(n={n})' for k, n in zip(cohort_keys, ns_cc))}")

# Within-cohort ρ(ARS, FI) + Fisher-z 95% CI per stage
def spearman_ci(r, n, alpha=0.05):
    """Fisher z-transform 95% CI for Spearman ρ."""
    from scipy.stats import norm as _norm
    z   = np.arctanh(r)
    se  = 1.0 / np.sqrt(n - 3)
    z_c = _norm.ppf(1 - alpha / 2)
    lo  = np.tanh(z - z_c * se)
    hi  = np.tanh(z + z_c * se)
    return lo, hi

wc_rows = []
for k, lbl, ars_g, fi_g in zip(cohort_keys, CC_LABELS_FLAT, ars_groups, fi_groups):
    idx_b = ars_g.dropna().index.intersection(fi_g.dropna().index)
    if len(idx_b) < 10:
        wc_rows.append({'cohort': lbl, 'rho': np.nan, 'p': np.nan,
                        'n': len(idx_b), 'ci_lo': np.nan, 'ci_hi': np.nan})
        continue
    r, p = spearmanr(ars_g.loc[idx_b], fi_g.loc[idx_b])
    lo, hi = spearman_ci(r, len(idx_b))
    wc_rows.append({'cohort': lbl, 'rho': r, 'p': p, 'n': len(idx_b),
                    'ci_lo': lo, 'ci_hi': hi})
    print(f"  {lbl:22s}  ρ={r:+.3f} [{lo:+.3f}, {hi:+.3f}]  {sig(p)}")
wc_df = pd.DataFrame(wc_rows)

# =============================================================================
# SECTION B — mCRPC DATA LOADING & SCORING
# =============================================================================
print("\n[B] Loading and scoring mCRPC data ...")

df, meta = load_mcrpc()   # fresh load for mCRPC z-score branch

all_genes = list(set(CROSS_COHORT_M6A_GENES + AR_TARGET_GENES + ['AR']))
all_genes = [g for g in all_genes if g in df.columns]
z_all     = zscore_normalize(df[all_genes])

# LR writer weights (Adeno/SCNC discrimination)
idx_hist = meta[meta['histology'].isin(['Adenocarcinoma','SCNC'])].index.intersection(df.index)
X_wr     = z_all.loc[idx_hist, WRITER_GENES].values
y_hs     = (meta.loc[idx_hist, 'histology'] == 'SCNC').astype(int).values
lr_m     = _LR(penalty='l2', C=1.0, max_iter=1000, random_state=42).fit(X_wr, y_hs)
lr_abs   = np.abs(lr_m.coef_[0])
lr_wts   = lr_abs / lr_abs.sum() * sum(MANUAL_WEIGHTS.values())
LR_WEIGHTS = {g: w for g, w in zip(WRITER_GENES, lr_wts)}

# m6A axes
dd_w = sum(z_all[g] * w for g, w in LR_WEIGHTS.items()) / sum(LR_WEIGHTS.values())
meta['m6A_Net_Deposition']    = dd_w - z_all[ERASER_GENES].mean(axis=1)
meta['m6A_Oncogenic_Readout'] = z_all[READER_ONCOGENIC].mean(axis=1) - z_all[READER_SUPPRESSIVE].mean(axis=1)
meta['m6A_Functional_Impact'] = meta['m6A_Net_Deposition'] * 0.435 + meta['m6A_Oncogenic_Readout'] * 0.565

# ARS — mean z-score (consistent with TCGA, cross-cohort, and DARANA computations)
ar_in_data = [g for g in AR_TARGET_GENES if g in z_all.columns]
meta['AR_Activity_Score'] = z_all[ar_in_data].mean(axis=1)

# Group indices
idx_adeno = meta[meta['histology'] == 'Adenocarcinoma'].index.intersection(df.index)
idx_scnc  = meta[meta['histology'] == 'SCNC'].index.intersection(df.index)
idx_lum   = meta[meta['Luminal/Basal Cluster'] == 'Luminal'].index.intersection(df.index)
idx_bas   = meta[meta['Luminal/Basal Cluster'] == 'Basal'].index.intersection(df.index)
ars       = meta.loc[df.index, 'AR_Activity_Score'].dropna()

print(f"  mCRPC: n={len(df)}  Adeno={len(idx_adeno)}  SCNC={len(idx_scnc)}"
      f"  Lum={len(idx_lum)}  Bas={len(idx_bas)}")

# =============================================================================
# SECTION C — DECONFOUNDING HELPERS
# =============================================================================

def partial_spearman(x, y, z_covar):
    """Partial Spearman ρ of x and y controlling for z_covar (residual method)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z_covar, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    rx = _rankdata(x, method='average') / len(x)
    ry = _rankdata(y, method='average') / len(y)
    Z  = np.column_stack([np.ones(len(z)), z])
    def _resid(v):
        coef, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ coef
    r, p = _pearsonr(_resid(rx), _resid(ry))
    return r, p, int(mask.sum())

# Adeno-only with L/B covariate
idx_adeno_full = idx_adeno.intersection(
    meta[['AR_Activity_Score', 'm6A_Functional_Impact',
          'm6A_Net_Deposition', 'm6A_Oncogenic_Readout']].dropna().index
)
meta_adeno  = meta.loc[idx_adeno_full]
z_adeno     = z_all.loc[idx_adeno_full]
ars_adeno_s = meta_adeno['AR_Activity_Score']

lb_num  = meta_adeno['Luminal/Basal Cluster'].map({'Luminal': 1, 'Basal': 0})
idx_lb  = lb_num.dropna().index
meta_lb = meta_adeno.loc[idx_lb]
z_lb    = z_adeno.loc[idx_lb]
ars_lb  = ars_adeno_s.loc[idx_lb]
lb_cov  = lb_num.loc[idx_lb].values

print(f"  Adeno subset: n={len(idx_adeno_full)}  (Lum={lb_cov.sum():.0f}, Bas={(lb_cov==0).sum():.0f})")

# Per-gene: Adeno-only / partial  (22-gene cross-cohort set)
pc_rows = []
for gene in CC_GENE_ORDER:
    r_f = p_f = np.nan
    if gene in z_all.columns:
        r_f, p_f = spearmanr(z_all.loc[ars.index, gene], ars)
    r_a = p_a = np.nan
    if gene in z_adeno.columns:
        r_a, p_a = spearmanr(z_adeno[gene], ars_adeno_s)
    r_p = p_p = np.nan
    if gene in z_lb.columns:
        r_p, p_p, _ = partial_spearman(z_lb[gene].values, ars_lb.values, lb_cov)
    pc_rows.append({'Gene': gene, 'Role': CC_GENE_ROLES.get(gene, 'Unknown'),
                    'rho_full': r_f, 'p_full': p_f,
                    'rho_adeno': r_a, 'p_adeno': p_a,
                    'rho_partial': r_p, 'p_partial': p_p})
pc_df = pd.DataFrame(pc_rows)

# ARS quartile label for paralog scatter
q_bins = pd.qcut(ars, q=4, labels=['Q1 (AR-low)', 'Q2', 'Q3', 'Q4 (AR-high)'])
q_palette = {'Q1 (AR-low)': '#3498db', 'Q2': '#27ae60',
             'Q3': '#f39c12', 'Q4 (AR-high)': '#e74c3c'}

# Luminal/Basal color map — used by several figures
cmap_lb = meta_adeno['Luminal/Basal Cluster'].map(
    {'Luminal': '#2980b9', 'Basal': '#c0392b'}).fillna('#95a5a6')

print(f"\n  Top per-gene correlations (Adeno-only partial ρ):")
top3 = pc_df.reindex(pc_df['rho_partial'].abs().nlargest(3).index)
for _, r in top3.iterrows():
    print(f"    {r['Gene']:8s}  ρ_partial={r['rho_partial']:+.3f} {sig(r['p_partial'])}")

# =============================================================================
# SECTION D — MEDIATION: ARS → RBM15B → m6A FI  (Adeno-only)
# =============================================================================
print("\n[D] Bootstrap mediation ...")

med_idx = meta_adeno[['AR_Activity_Score', 'm6A_Functional_Impact']].dropna().index.intersection(z_adeno.index)
X_med   = meta_adeno.loc[med_idx, 'AR_Activity_Score'].values
M_med   = z_adeno.loc[med_idx, 'RBM15B'].values
Y_med   = meta_adeno.loc[med_idx, 'm6A_Functional_Impact'].values
n_med   = len(X_med)

def _std(v): return (v - v.mean()) / v.std()
X_s, M_s, Y_s = _std(X_med), _std(M_med), _std(Y_med)

def _ols(x_mat, y_vec):
    X_ = np.column_stack([np.ones(len(y_vec)), x_mat])
    coef_, *_ = np.linalg.lstsq(X_, y_vec, rcond=None)
    resid = y_vec - X_ @ coef_
    mse   = (resid**2).sum() / (len(y_vec) - X_.shape[1])
    cov   = mse * np.linalg.inv(X_.T @ X_)
    se    = np.sqrt(np.diag(cov))
    t_stat = coef_ / se
    p_vals = 2 * _t_dist.sf(np.abs(t_stat), df=len(y_vec) - X_.shape[1])
    return coef_[1:], se[1:], p_vals[1:]

a_coef,  a_se,  a_p  = _ols(X_s.reshape(-1,1), M_s)
b_coef,  b_se,  b_p  = _ols(np.column_stack([M_s, X_s]), Y_s)
c_coef,  c_se,  c_p  = _ols(X_s.reshape(-1,1), Y_s)
cp_coef, cp_se, cp_p = _ols(np.column_stack([X_s, M_s]), Y_s)

a, b, c_total, cp_val = float(a_coef[0]), float(b_coef[0]), float(c_coef[0]), float(cp_coef[0])
indirect = a * b

# Bootstrap
rng = np.random.default_rng(2025)
boot_indirect = []
for _ in range(5000):
    idx_b   = rng.integers(0, n_med, n_med)
    Xb, Mb, Yb = X_s[idx_b], M_s[idx_b], Y_s[idx_b]
    a_b, *_ = _ols(Xb.reshape(-1,1), Mb)
    b_b, *_ = _ols(np.column_stack([Mb, Xb]), Yb)
    boot_indirect.append(float(a_b[0]) * float(b_b[0]))
boot_indirect = np.array(boot_indirect)
ci_lo, ci_hi  = np.percentile(boot_indirect, [2.5, 97.5])
p_boot        = min(np.mean(boot_indirect <= 0), np.mean(boot_indirect >= 0)) * 2
prop_med      = abs(indirect / c_total) * 100 if abs(c_total) > 1e-9 else float('nan')

print(f"  Path a  β={a:+.4f} {sig(a_p[0])},  Path b  β={b:+.4f} {sig(b_p[0])}")
print(f"  Total c β={c_total:+.4f} {sig(c_p[0])},  Direct c' β={cp_val:+.4f} {sig(cp_p[0])}")
print(f"  Indirect a×b = {indirect:+.4f}  95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p_boot:.4f} {sig(p_boot)}")
print(f"  Proportion mediated: {prop_med:.1f}%")

# =============================================================================
# FIGURE 3 — TRIPLE TRAJECTORY: AR mRNA + ARS + m6A FI across disease stages
# =============================================================================
print("\n[Fig 3] Triple trajectory ...")

# AR mRNA percentile-rank per cohort (shows expression loss in SCNC directly)
if 'AR' in common:
    ar_pct_groups = [pct[k]['AR'] for k in cohort_keys]
else:
    ar_pct_groups = [pd.Series(dtype=float)] * len(cohort_keys)
_, p_ar_kw  = kruskal(*[g.dropna().values for g in ar_pct_groups])
_, p_ars_kw = kruskal(*[g.dropna().values for g in ars_groups])
_, p_fi_kw  = kruskal(*[g.dropna().values for g in fi_groups])

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

# Panel A — AR mRNA (percentile rank)
parts0 = ax0.violinplot([g.values for g in ar_pct_groups],
                         positions=range(len(ar_pct_groups)),
                         showmeans=True, showmedians=True)
for i, c in enumerate(CC_COLORS):
    parts0['bodies'][i].set_facecolor(c); parts0['bodies'][i].set_alpha(0.65)
style_violin(parts0, ax0)
for i, (g, c) in enumerate(zip(ar_pct_groups, CC_COLORS)):
    ax0.scatter([i] * len(g), g.values, c=c, alpha=0.15, s=4, zorder=0, edgecolors='none')
ax0.axhline(50, color='grey', lw=1, ls='--', alpha=0.5, label='Global median (50th %ile)')
ax0.set_ylabel('AR mRNA Expression\n(%ile rank)', fontsize=12, fontweight='bold')
ax0.set_title(f'A.  AR mRNA Expression Trajectory  (KW p={p_ar_kw:.2e} {sig(p_ar_kw)})\n'
              f'SCNC drop confirms AR gene loss — ARS low in SCNC reflects AR loss, not regulation',
              fontsize=12, fontweight='bold', loc='left', pad=8)
ax0.legend(fontsize=9, loc='lower right')
# Annotate SCNC collapse
scnc_ar_med = ar_pct_groups[5].median()
ax0.annotate('AR lost\nin SCNC', xy=(5, scnc_ar_med),
             xytext=(4.2, scnc_ar_med + 10),
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
             fontsize=9, color='#c0392b', fontweight='bold')

# Panel B — ARS
parts1 = ax1.violinplot([g.values for g in ars_groups],
                         positions=range(len(ars_groups)),
                         showmeans=True, showmedians=True)
for i, c in enumerate(CC_COLORS):
    parts1['bodies'][i].set_facecolor(c); parts1['bodies'][i].set_alpha(0.65)
style_violin(parts1, ax1)
for i, (g, c) in enumerate(zip(ars_groups, CC_COLORS)):
    ax1.scatter([i] * len(g), g.values, c=c, alpha=0.15, s=4, zorder=0, edgecolors='none')
ax1.axhline(50, color='grey', lw=1, ls='--', alpha=0.5, label='Global median (50th %ile)')
ax1.set_ylabel('AR Activity Score\n(mean %ile rank of AR targets)', fontsize=12, fontweight='bold')
ax1.set_title(f'B.  AR Activity Score Trajectory  (KW p={p_ars_kw:.2e} {sig(p_ars_kw)})',
              fontsize=13, fontweight='bold', loc='left', pad=8)
ax1.legend(fontsize=9, loc='lower right')

# Panel C — m6A FI
parts2 = ax2.violinplot([g.values for g in fi_groups],
                         positions=range(len(fi_groups)),
                         showmeans=True, showmedians=True)
for i, c in enumerate(CC_COLORS):
    parts2['bodies'][i].set_facecolor(c); parts2['bodies'][i].set_alpha(0.65)
style_violin(parts2, ax2)
for i, (g, c) in enumerate(zip(fi_groups, CC_COLORS)):
    ax2.scatter([i] * len(g), g.values, c=c, alpha=0.15, s=4, zorder=0, edgecolors='none')
ax2.axhline(0, color='grey', lw=1, ls='--', alpha=0.5)
ax2.set_ylabel('m6A Functional Impact\n(%ile-rank, LR-weighted writers)', fontsize=12, fontweight='bold')
ax2.set_title(f'C.  m6A Functional Impact Trajectory  (KW p={p_fi_kw:.2e} {sig(p_fi_kw)})',
              fontsize=13, fontweight='bold', loc='left', pad=8)

ax2.set_xticks(range(len(CC_LABELS)))
ax2.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(CC_LABELS, ns_cc)],
                    fontsize=10.5, fontweight='bold')
ax2.set_xlabel('Disease Stage', fontsize=12, fontweight='bold')

# Annotate key observation: sign flip
ax2.annotate('m6A FI rises\nat mCRPC\n(+13 pts)', xy=(4, fi_groups[4].mean()),
             xytext=(4.15, fi_groups[4].mean() + 8),
             arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5),
             fontsize=9, color='#e67e22', fontweight='bold')

plt.suptitle('AR Expression, AR Activity and m6A Functional Impact Across Prostate Cancer Progression',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig3_dual_trajectory.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig3_dual_trajectory.png")
plt.close()

# =============================================================================
# FIGURE 4 — WITHIN-COHORT COUPLING FOREST (sign flip thesis)
# =============================================================================
print("[Fig 4] Within-cohort coupling forest ...")

fig, ax = plt.subplots(figsize=(12, 6))
valid   = wc_df.dropna(subset=['rho'])
x_pos   = range(len(wc_df))

for i, row in wc_df.iterrows():
    c = CC_COLORS[i]
    if np.isnan(row['rho']):
        ax.bar(i, 0, color='#cccccc', edgecolor='black', alpha=0.4, width=0.65)
        continue
    ax.bar(i, row['rho'], color=c, edgecolor='black', alpha=0.82, width=0.65)
    # 95% CI error bar
    ax.plot([i, i], [row['ci_lo'], row['ci_hi']], color='black', lw=2, zorder=5)
    ax.plot([i - 0.12, i + 0.12], [row['ci_lo'], row['ci_lo']], color='black', lw=1.5, zorder=5)
    ax.plot([i - 0.12, i + 0.12], [row['ci_hi'], row['ci_hi']], color='black', lw=1.5, zorder=5)
    # Annotation
    yoff = row['rho'] + (0.015 if row['rho'] >= 0 else -0.03)
    ax.text(i, yoff, f"{sig(row['p'])}\nρ={row['rho']:+.3f}",
            ha='center', va='bottom' if row['rho'] >= 0 else 'top',
            fontsize=9, fontweight='bold', color='black')

ax.axhline(0, color='black', lw=1)
# Mark mCRPC Adeno bar with annotation
ax.annotate('Sign flip:\ncoupling\n becomes\nnegative', xy=(4, wc_df.loc[4, 'rho'] - 0.01),
            xytext=(4.55, wc_df.loc[4, 'rho'] - 0.09),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5),
            fontsize=9, color='#e67e22', fontweight='bold')

ax.set_xticks(list(x_pos))
ax.set_xticklabels([f"{lbl}\n(n={int(row['n'])})"
                    for lbl, row in zip(CC_LABELS_FLAT, wc_df.to_dict('records'))],
                   fontsize=10, fontweight='bold')
ax.set_ylabel('Within-cohort Spearman ρ  (ARS vs m6A FI)\n95% CI via Fisher z-transform',
              fontsize=12, fontweight='bold')
ax.set_title('Within-Cohort AR × m6A Functional Impact Coupling Across Disease Stages\n'
             'Positive coupling in normal tissue reverses to negative at mCRPC',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig4_coupling_forest.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig4_coupling_forest.png")
plt.close()


# =============================================================================
# FIGURE 5 — PER-GENE mCRPC DECONFOUNDED  +  KEY GENE SCATTERS  +  PARALOG AXIS
#
# Layout (GridSpec 2×3):
#   Row 0:  Panel A — two-bar barplot: Adeno-only + partial (spans all 3 columns)
#   Row 1:  Panel B — RBM15B vs ARS  |  Panel C — IGF2BP3 vs ARS  |  Panel D — RBM15/RBM15B paralog
# =============================================================================
print("[Fig 5] Per-gene deconfounded + scatters + paralog axis ...")

fig = plt.figure(figsize=(19, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                         height_ratios=[1.1, 1])
ax_a = fig.add_subplot(gs[0, :])   # full-width top
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])
ax_d = fig.add_subplot(gs[1, 2])

# --- Panel A: two-bar deconfounded (Adeno-only + partial) -------------------
n_genes = len(pc_df)
x_g     = np.arange(n_genes)
w       = 0.3

ax_a.bar(x_g - w/2, pc_df['rho_adeno'],   w, label='Adeno only',
         color='#3498db', edgecolor='black', alpha=0.88)
ax_a.bar(x_g + w/2, pc_df['rho_partial'], w, label='Adeno + partial (L/B ctrl)',
         color='#e74c3c', edgecolor='black', alpha=0.88)

for i, row in pc_df.iterrows():
    for rv, pv, xoff in [(row['rho_adeno'], row['p_adeno'], x_g[i]-w/2),
                         (row['rho_partial'], row['p_partial'], x_g[i]+w/2)]:
        s = sig(pv)
        if s != 'ns' and not np.isnan(rv):
            yoff = rv + (0.022 if rv >= 0 else -0.04)
            ax_a.text(xoff, yoff, s,
                      ha='center', va='bottom' if rv >= 0 else 'top',
                      fontsize=8, fontweight='bold',
                      color='#c0392b' if xoff > x_g[i] else '#2980b9')

ax_a.axhline(0, color='black', lw=0.8)
ax_a.set_xticks(x_g)
ax_a.set_xticklabels(
    [f"{r['Gene']}\n({r['Role'].split('(')[1].rstrip(')') if '(' in r['Role'] else r['Role']})"
     for _, r in pc_df.iterrows()],
    fontsize=8.5, rotation=25, ha='right')
ax_a.set_ylabel('Spearman ρ with AR Activity Score', fontsize=11, fontweight='bold')
ax_a.set_title('A.  Per-gene m6A × ARS Correlation in mCRPC  (Adeno-only + L/B deconfounded)\n'
               'Blue = Adeno-only; Red = after partial Spearman (Luminal/Basal controlled)',
               fontsize=11, fontweight='bold', loc='left', pad=6)
ax_a.legend(fontsize=9, loc='upper right')

# --- Panel B: RBM15B vs ARS (Adeno-only) ------------------------------------
x_b = ars_adeno_s.values
y_b = z_adeno['RBM15B'].values
r_b, p_b = spearmanr(x_b, y_b)
ax_b.scatter(x_b, y_b, c=cmap_lb.values, alpha=0.4, s=16, edgecolors='none')
m_b, b_b_coef = np.polyfit(x_b[np.isfinite(x_b) & np.isfinite(y_b)],
                             y_b[np.isfinite(x_b) & np.isfinite(y_b)], 1)
x_bl = np.linspace(x_b.min(), x_b.max(), 100)
ax_b.plot(x_bl, m_b * x_bl + b_b_coef, 'k-', lw=2)
ax_b.set_xlabel('AR Activity Score', fontsize=10, fontweight='bold')
ax_b.set_ylabel('RBM15B z-score', fontsize=10, fontweight='bold')
ax_b.set_title(f'B.  RBM15B (Writer-Targeting)\n'
               f'ρ={r_b:+.3f}, p={p_b:.2e} {sig(p_b)}  (Adeno n={len(x_b)})',
               fontsize=10, fontweight='bold', loc='left', pad=5)
ax_b.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax_b.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

# --- Panel C: IGF2BP3 vs ARS (Adeno-only) ------------------------------------
x_c = ars_adeno_s.values
y_c = z_adeno['IGF2BP3'].values
r_c, p_c = spearmanr(x_c, y_c)
ax_c.scatter(x_c, y_c, c=cmap_lb.values, alpha=0.4, s=16, edgecolors='none')
m_c, b_c_coef = np.polyfit(x_c[np.isfinite(x_c) & np.isfinite(y_c)],
                             y_c[np.isfinite(x_c) & np.isfinite(y_c)], 1)
x_cl = np.linspace(x_c.min(), x_c.max(), 100)
ax_c.plot(x_cl, m_c * x_cl + b_c_coef, 'k-', lw=2)
ax_c.set_xlabel('AR Activity Score', fontsize=10, fontweight='bold')
ax_c.set_ylabel('IGF2BP3 z-score', fontsize=10, fontweight='bold')
ax_c.set_title(f'C.  IGF2BP3 (Oncogenic Reader)\n'
               f'ρ={r_c:+.3f}, p={p_c:.2e} {sig(p_c)}  (Adeno n={len(x_c)})',
               fontsize=10, fontweight='bold', loc='left', pad=5)
ax_c.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax_c.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

ax_c.legend(handles=[Patch(facecolor='#2980b9', label='Luminal'),
                      Patch(facecolor='#c0392b', label='Basal'),
                      Patch(facecolor='#95a5a6', label='Unassigned')],
            fontsize=8, loc='upper right')

# --- Panel D: RBM15 vs RBM15B colored by ARS quartile -----------------------
# Shows the paralog balance shift: AR-high tumors tip toward RBM15B (above diagonal).
# Per-quartile scatter trend lines are noise (within-Q rho ns); the signal is
# the fraction above the identity diagonal shifting with ARS quartile.
x_d_idx = ars.index  # all mCRPC samples with ARS
x_d = z_all.loc[x_d_idx, 'RBM15'].values
y_d = z_all.loc[x_d_idx, 'RBM15B'].values
fin_d = np.isfinite(x_d) & np.isfinite(y_d)

# Imbalance score: RBM15B - RBM15 (positive = shifted toward RBM15B)
imbalance = y_d - x_d
r_imb, p_imb = spearmanr(ars.loc[x_d_idx].values[fin_d], imbalance[fin_d])

# Fraction above diagonal per quartile
q_frac = {}
for q_lbl in q_palette:
    mask_q = ((q_bins.loc[x_d_idx] == q_lbl).values) & fin_d
    q_frac[q_lbl] = (y_d[mask_q] > x_d[mask_q]).mean() * 100 if mask_q.sum() > 0 else np.nan

# Plot each quartile as separate scatter
for q_lbl, q_col in q_palette.items():
    mask_q = q_bins.loc[x_d_idx] == q_lbl
    ax_d.scatter(x_d[mask_q.values], y_d[mask_q.values],
                 c=q_col, alpha=0.45, s=16, edgecolors='none',
                 label=f'{q_lbl} ({q_frac[q_lbl]:.0f}% above diag.)')

# Diagonal of equality
lim_d = max(abs(x_d[fin_d]).max(), abs(y_d[fin_d]).max()) * 1.05
ax_d.plot([-lim_d, lim_d], [-lim_d, lim_d], 'k--', lw=1.5, alpha=0.55, zorder=3)
ax_d.axhline(0, color='grey', lw=0.5, ls=':', alpha=0.4)
ax_d.axvline(0, color='grey', lw=0.5, ls=':', alpha=0.4)
ax_d.set_xlabel('RBM15 z-score\n(Writer-Targeting paralog)', fontsize=10, fontweight='bold')
ax_d.set_ylabel('RBM15B z-score\n(Writer-Targeting paralog)', fontsize=10, fontweight='bold')
ax_d.set_title(f'D.  RBM15 ↔ RBM15B Paralog Balance by ARS Quartile\n'
               f'ARS vs (RBM15B−RBM15): ρ={r_imb:+.3f} {sig(p_imb)}  (n={fin_d.sum()})',
               fontsize=10, fontweight='bold', loc='left', pad=5)
ax_d.legend(fontsize=7.5, loc='upper left', framealpha=0.9, title='% above diagonal = fraction\nshifted toward RBM15B')

# Global legend for Luminal/Basal (panels B and C)
fig.text(0.01, 0.01, '★ Panels B & C: color = Luminal (blue) / Basal (red) / Unassigned (grey)',
         fontsize=8.5, style='italic', color='#555555')

plt.suptitle('mCRPC: Per-gene m6A × ARS Correlations (Adeno-only and L/B Deconfounded)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUTDIR, 'fig5_per_gene_mcrpc.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig5_per_gene_mcrpc.png")
plt.close()

# =============================================================================
# FIGURE 6 — MEDIATION: ARS → RBM15B → m6A FI  (1×2 layout: path + bootstrap)
# =============================================================================
print("[Fig 6] Mediation ...")

fig6_ms = plt.figure(figsize=(18, 8))
gs6_ms  = gridspec.GridSpec(1, 2, figure=fig6_ms, hspace=0.35, wspace=0.40)
ax_path = fig6_ms.add_subplot(gs6_ms[0, 0])
ax_boot = fig6_ms.add_subplot(gs6_ms[0, 1])

# --- Panel A: path diagram ---------------------------------------------------
ax_path.set_xlim(0, 10)
ax_path.set_ylim(0, 6.5)
ax_path.axis('off')

box_specs = [
    (0.3, 2.5, 2.3, 1.4, 'AR Activity\nScore (X)', '#3498db'),
    (3.85, 4.5, 2.3, 1.4, 'RBM15B z-score\n(Mediator M)', '#e67e22'),
    (7.4, 2.5, 2.3, 1.4, 'm6A Functional\nImpact (Y)', '#e74c3c'),
]
for (x0, y0, w_box, h_box, lbl, clr) in box_specs:
    rect = plt.Rectangle((x0, y0), w_box, h_box, linewidth=2,
                          edgecolor=clr, facecolor=clr, alpha=0.18, zorder=2)
    ax_path.add_patch(rect)
    ax_path.text(x0 + w_box/2, y0 + h_box/2, lbl, ha='center', va='center',
                 fontsize=11, fontweight='bold', color=clr, zorder=3)

# X → M  (path a)
ax_path.annotate('', xy=(3.85, 5.2), xytext=(2.6, 3.9),
                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5,
                                 connectionstyle='arc3,rad=-0.2'))
ax_path.text(2.8, 5.05,
             f'a = {a:+.3f} {sig(a_p[0])}\n'
             f'(+1 SD ARS → {a:+.2f} SD RBM15B)',
             fontsize=9.5, color='#e67e22', fontweight='bold', ha='center')

# M → Y  (path b)
ax_path.annotate('', xy=(7.4, 5.2), xytext=(6.15, 5.2),
                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5))
ax_path.text(6.75, 5.55,
             f'b = {b:+.3f} {sig(b_p[0])}\n'
             f'(+1 SD RBM15B → {b:+.2f} SD FI | X)',
             fontsize=9.5, color='#e67e22', fontweight='bold', ha='center')

# X → Y  (direct c')
ax_path.annotate('', xy=(7.4, 3.2), xytext=(2.6, 3.2),
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
ax_path.text(5.0, 2.85,
             f"c' = {cp_val:+.3f} {sig(cp_p[0])}  (direct, net of RBM15B)",
             fontsize=9.5, color='#3498db', fontweight='bold', ha='center')

# Total effect annotation at bottom
ax_path.text(5.0, 2.1,
             f'Total effect  c = {c_total:+.3f} {sig(c_p[0])}',

             fontsize=9.5, color='#555555', fontweight='bold', ha='center')

# Summary box
zero_out = 'CI excludes 0 ✓' if (ci_lo > 0 or ci_hi < 0) else 'CI includes 0'
ax_path.text(5.0, 1.15,
             f'Indirect  a×b = {indirect:+.4f}   Bootstrap 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]\n'
             f'p = {p_boot:.4f} {sig(p_boot)}   |   Proportion mediated ≈ {prop_med:.1f}%\n'
             f'({zero_out};  all paths estimated by OLS on standardised variables)',
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(facecolor='lightyellow', edgecolor='goldenrod', alpha=0.9, pad=7))

ax_path.set_title('A.  Causal Mediation Path Diagram  (standardised OLS)\n'
                   '      Adenocarcinoma only; each coefficient = SD-unit effect',
                   fontsize=12, fontweight='bold', loc='left', pad=8)

# --- Panel B: bootstrap distribution ----------------------------------------
ax_boot.hist(boot_indirect, bins=80, color='#e67e22', edgecolor='none', alpha=0.7)
ax_boot.axvline(0,        color='black', lw=1.5, ls='--', label='Zero (null)')
ax_boot.axvline(indirect, color='#c0392b', lw=2.5,
                label=f'Observed a×b = {indirect:+.4f}')
ax_boot.axvline(ci_lo, color='grey', lw=1.5, ls=':',
                label=f'95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')
ax_boot.axvline(ci_hi, color='grey', lw=1.5, ls=':')
ax_boot.set_xlabel('Indirect effect (a×b)', fontsize=12, fontweight='bold')
ax_boot.set_ylabel('Bootstrap frequency (n=5 000)', fontsize=12, fontweight='bold')
ax_boot.set_title(f'B.  Bootstrap Distribution of Indirect Effect  (n=5 000 resamples)\n'
                   f'ARS → RBM15B → m6A FI  |  n={n_med} Adeno samples',
                   fontsize=12, fontweight='bold', loc='left', pad=8)
ax_boot.legend(fontsize=10)
# Shade CI region
xlim_b = ax_boot.get_xlim()
y_top  = ax_boot.get_ylim()[1]
ax_boot.axvspan(ci_lo, ci_hi, alpha=0.08, color='grey', zorder=0)
ax_boot.text(indirect * 1.3, y_top * 0.85,
             f'p = {p_boot:.4f} {sig(p_boot)}\n{prop_med:.1f}% of total\neffect mediated',
             fontsize=11, fontweight='bold', color='#c0392b',
             bbox=dict(facecolor='white', edgecolor='#c0392b', alpha=0.85, pad=4))

fig6_ms.suptitle('ARS → RBM15B → m6A Functional Impact: Bootstrap Causal Mediation  (mCRPC Adenocarcinoma)',
                 fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig6_mediation_survival.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig6_mediation_survival.png")
plt.close()

# TCGA z-scored matrix (computed once, shared by Figs 1 and 2) — full 22-gene set
z_tcga  = zscore_normalize(df_tcga[[g for g in CC_GENE_ORDER + ['AR'] if g in df_tcga.columns]])
ar_tcga = z_tcga['AR'] if 'AR' in z_tcga.columns else None

# =============================================================================
# FIGURE 1 — AR mRNA EXPRESSION × m6A: Primary PCa → mCRPC
#
# Design: 2-bar per-gene barplot (TCGA Primary / mCRPC Adeno partial) + two
# aggregate scatters showing disease-stage contrast.
#
# Layout (GridSpec 2×2):
#   Row 0: Panel A — full-width per-gene ρ(m6A gene, AR mRNA) barplot
#   Row 1: Panel B — TCGA scatter  |  Panel C — mCRPC Adeno scatter
# =============================================================================
print("[Fig 1] AR mRNA expression × m6A: Primary → mCRPC ...")

# mCRPC: AR mRNA z-score per gene
ar_expr_all   = z_all['AR']
ar_expr_adeno = z_all.loc[idx_adeno_full, 'AR']
ar_expr_lb    = z_all.loc[idx_lb, 'AR'].values

# Per-gene mCRPC partial ρ(m6A gene, AR mRNA)  — 22-gene set
h1_rows = []
for gene in CC_GENE_ORDER:
    r_p = p_p = np.nan
    if gene in z_lb.columns:
        r_p, p_p, _ = partial_spearman(z_lb[gene].values, ar_expr_lb, lb_cov)
    h1_rows.append({'Gene': gene, 'Role': CC_GENE_ROLES.get(gene, 'Unknown'),
                    'rho_partial': r_p, 'p_partial': p_p})
h1_df = pd.DataFrame(h1_rows)

# Per-gene TCGA ρ(m6A gene, AR mRNA)  — 22-gene set
h1_tcga_rows = []
for gene in CC_GENE_ORDER:
    if ar_tcga is not None and gene in z_tcga.columns:
        idx_c = ar_tcga.dropna().index.intersection(z_tcga[gene].dropna().index)
        r_t, p_t = spearmanr(z_tcga.loc[idx_c, gene], ar_tcga.loc[idx_c])
    else:
        r_t, p_t = np.nan, np.nan
    h1_tcga_rows.append({'Gene': gene, 'rho': r_t, 'p': p_t})
h1_tcga_df = pd.DataFrame(h1_tcga_rows)

# Aggregate scatter data — use z-scored AR for both cohorts for comparability
fi_primary = fi_groups[2]
if ar_tcga is not None:
    common_tcga = ar_tcga.dropna().index.intersection(fi_primary.dropna().index)
    r_t7, p_t7 = spearmanr(ar_tcga.loc[common_tcga], fi_primary.loc[common_tcga])
    ci_t7_lo, ci_t7_hi = spearman_ci(r_t7, len(common_tcga))
    print(f"  TCGA Primary  ρ(AR expr, m6A FI)={r_t7:+.3f} [{ci_t7_lo:+.3f},{ci_t7_hi:+.3f}] {sig(p_t7)} (n={len(common_tcga)})")
else:
    common_tcga = None
fi_adeno_vals = meta_adeno['m6A_Functional_Impact']
common_mcrpc  = ar_expr_adeno.dropna().index.intersection(fi_adeno_vals.dropna().index)
r_m7, p_m7    = spearmanr(ar_expr_adeno.loc[common_mcrpc], fi_adeno_vals.loc[common_mcrpc])
ci_m7_lo, ci_m7_hi = spearman_ci(r_m7, len(common_mcrpc))
print(f"  mCRPC Adeno   ρ(AR expr, m6A FI)={r_m7:+.3f} [{ci_m7_lo:+.3f},{ci_m7_hi:+.3f}] {sig(p_m7)} (n={len(common_mcrpc)})")

# Build figure
fig7 = plt.figure(figsize=(19, 12))
gs7  = gridspec.GridSpec(2, 2, figure=fig7, hspace=0.5, wspace=0.38,
                          height_ratios=[1.1, 1])
ax7a = fig7.add_subplot(gs7[0, :])
ax7b = fig7.add_subplot(gs7[1, 0])
ax7c = fig7.add_subplot(gs7[1, 1])

# Panel A: per-gene 2-bar barplot
n7 = len(h1_tcga_df)
x7 = np.arange(n7)
w7 = 0.3
ax7a.bar(x7 - w7/2, h1_tcga_df['rho'],    w7, label=f'TCGA Primary (n={len(z_tcga)})',
         color='#3498db', edgecolor='black', alpha=0.88)
ax7a.bar(x7 + w7/2, h1_df['rho_partial'], w7, label='mCRPC Adeno partial (L/B ctrl)',
         color='#e67e22', edgecolor='black', alpha=0.88)
for i, (rt, pt, rm, pm) in enumerate(zip(h1_tcga_df['rho'], h1_tcga_df['p'],
                                          h1_df['rho_partial'], h1_df['p_partial'])):
    for rv, pv, xoff in [(rt, pt, x7[i]-w7/2), (rm, pm, x7[i]+w7/2)]:
        s = sig(pv)
        if s != 'ns' and not np.isnan(rv):
            ax7a.text(xoff, rv + (0.018 if rv >= 0 else -0.035), s,
                      ha='center', va='bottom' if rv >= 0 else 'top',
                      fontsize=7.5, fontweight='bold')
ax7a.axhline(0, color='black', lw=0.8)
ax7a.set_xticks(x7)
ax7a.set_xticklabels(
    [f"{r['Gene']}\n({r['Role'].split('(')[1].rstrip(')') if '(' in r['Role'] else r['Role']})"
     for _, r in h1_df.iterrows()], fontsize=8.5, rotation=25, ha='right')
ax7a.set_ylabel('Spearman ρ  (m6A gene vs AR mRNA expression)', fontsize=11, fontweight='bold')
ax7a.set_title('A.  Per-gene ρ(m6A gene, AR mRNA)  —  Primary PCa vs mCRPC\n',
               fontsize=11, fontweight='bold', loc='left', pad=6)
ax7a.legend(fontsize=9, loc='upper right')

# Panel B: TCGA Primary scatter (z-scored AR)
if ar_tcga is not None and common_tcga is not None:
    xb = ar_tcga.loc[common_tcga].values
    yb = fi_primary.loc[common_tcga].values
    ax7b.scatter(xb, yb, c='#3498db', alpha=0.35, s=14, edgecolors='none')
    fin = np.isfinite(xb) & np.isfinite(yb)
    mb, bb = np.polyfit(xb[fin], yb[fin], 1)
    xl = np.linspace(xb[fin].min(), xb[fin].max(), 100)
    ax7b.plot(xl, mb*xl+bb, 'k-', lw=2.5, alpha=0.8)
    ax7b.set_xlabel('AR mRNA (z-score)', fontsize=11, fontweight='bold')
    ax7b.set_ylabel('m6A Functional Impact', fontsize=11, fontweight='bold')
    ax7b.set_title(f'B.  TCGA Primary PCa  (n={len(common_tcga)})\n'
                   f'ρ={r_t7:+.3f} [{ci_t7_lo:+.3f},{ci_t7_hi:+.3f}]  p={p_t7:.2e} {sig(p_t7)}',
                   fontsize=11, fontweight='bold', loc='left', pad=6)
    ax7b.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax7b.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

# Panel C: mCRPC Adeno scatter
xc = ar_expr_adeno.loc[common_mcrpc].values
yc = fi_adeno_vals.loc[common_mcrpc].values
ax7c.scatter(xc, yc, c=cmap_lb.loc[common_mcrpc].values, alpha=0.4, s=14, edgecolors='none')
fin = np.isfinite(xc) & np.isfinite(yc)
mc, bc = np.polyfit(xc[fin], yc[fin], 1)
xl = np.linspace(xc[fin].min(), xc[fin].max(), 100)
ax7c.plot(xl, mc*xl+bc, 'k-', lw=2.5, alpha=0.8)
ax7c.set_xlabel('AR mRNA (z-score)', fontsize=11, fontweight='bold')
ax7c.set_ylabel('m6A Functional Impact', fontsize=11, fontweight='bold')
ax7c.set_title(f'C.  mCRPC Adenocarcinoma  (n={len(common_mcrpc)})\n'
               f'ρ={r_m7:+.3f} [{ci_m7_lo:+.3f},{ci_m7_hi:+.3f}]  p={p_m7:.2e} {sig(p_m7)}',
               fontsize=11, fontweight='bold', loc='left', pad=6)
ax7c.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax7c.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax7c.legend(handles=[Patch(facecolor='#2980b9', label='Luminal'),
                     Patch(facecolor='#c0392b', label='Basal'),
                     Patch(facecolor='#95a5a6', label='Unassigned')],
            fontsize=9, loc='upper right')

fig7.suptitle('AR mRNA Expression × m6A Functional Impact:  Primary PCa → mCRPC',
              fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUTDIR, 'fig1_arexpr_vs_m6a.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig1_arexpr_vs_m6a.png")
plt.close()

# =============================================================================
# FIGURE 2 — AR ACTIVITY SCORE × m6A: Primary PCa → mCRPC
#
# Same disease-progression framing as Fig 1 but using ARS (functional output)
# rather than AR mRNA.  Per-gene barplot + two aggregate scatters.
#
# Layout (GridSpec 2×2):
#   Row 0: Panel A — per-gene ρ(m6A gene, ARS) barplot, Primary vs mCRPC
#   Row 1: Panel B — TCGA scatter (m6A FI vs ARS)
#          Panel C — mCRPC Adeno scatter (m6A FI vs ARS)
# =============================================================================
print("[Fig 2] AR activity score × m6A: Primary → mCRPC ...")

# Within-TCGA z-scored ARS (mean of z-scored AR target genes, same logic as mCRPC)
ar_tcga_targets = [g for g in AR_TARGET_GENES if g in df_tcga.columns]
ars_tcga_z = zscore_normalize(df_tcga[ar_tcga_targets]).mean(axis=1)

# Per-gene TCGA ρ(m6A gene, ARS) — 22-gene set, z-scored ARS
h2_tcga_rows = []
for gene in CC_GENE_ORDER:
    if gene in z_tcga.columns:
        common_c = ars_tcga_z.dropna().index.intersection(z_tcga[gene].dropna().index)
        r_t, p_t = spearmanr(z_tcga.loc[common_c, gene], ars_tcga_z.loc[common_c])
    else:
        r_t, p_t = np.nan, np.nan
    h2_tcga_rows.append({'Gene': gene, 'rho': r_t, 'p': p_t})
h2_tcga_df = pd.DataFrame(h2_tcga_rows)

# Aggregate TCGA scatter: m6A FI vs ARS (z-scored)
fi_primary_8 = fi_groups[2]
common_t8    = ars_tcga_z.dropna().index.intersection(fi_primary_8.dropna().index)
r_t8, p_t8  = spearmanr(ars_tcga_z.loc[common_t8], fi_primary_8.loc[common_t8])
ci_t8_lo, ci_t8_hi = spearman_ci(r_t8, len(common_t8))
print(f"  TCGA Primary  ρ(ARS, m6A FI)={r_t8:+.3f} [{ci_t8_lo:+.3f},{ci_t8_hi:+.3f}] {sig(p_t8)} (n={len(common_t8)})")

# Aggregate mCRPC scatter — ARS already mean z-score, use adeno subset directly
ars_adeno_mz  = ars_adeno_s
fi_adeno_8    = meta_adeno['m6A_Functional_Impact']
common_m8     = ars_adeno_mz.dropna().index.intersection(fi_adeno_8.dropna().index)
r_m8, p_m8   = spearmanr(ars_adeno_mz.loc[common_m8], fi_adeno_8.loc[common_m8])
ci_m8_lo, ci_m8_hi = spearman_ci(r_m8, len(common_m8))
print(f"  mCRPC Adeno   ρ(ARS-meanz, m6A FI)={r_m8:+.3f} [{ci_m8_lo:+.3f},{ci_m8_hi:+.3f}] {sig(p_m8)} (n={len(common_m8)})")

# Build figure
fig8 = plt.figure(figsize=(19, 12))
gs8  = gridspec.GridSpec(2, 2, figure=fig8, hspace=0.5, wspace=0.38,
                          height_ratios=[1.1, 1])
ax8a = fig8.add_subplot(gs8[0, :])
ax8b = fig8.add_subplot(gs8[1, 0])
ax8c = fig8.add_subplot(gs8[1, 1])

# Panel A: per-gene 2-bar barplot
n8 = len(h2_tcga_df)
x8 = np.arange(n8)
w8 = 0.3
ax8a.bar(x8 - w8/2, h2_tcga_df['rho'],    w8, label=f'TCGA Primary (n={len(ars_tcga_z.dropna())})',
         color='#3498db', edgecolor='black', alpha=0.88)
ax8a.bar(x8 + w8/2, pc_df['rho_partial'], w8, label='mCRPC Adeno partial (L/B ctrl)',
         color='#e67e22', edgecolor='black', alpha=0.88)
for i, (rt, pt, rm, pm) in enumerate(zip(h2_tcga_df['rho'], h2_tcga_df['p'],
                                          pc_df['rho_partial'], pc_df['p_partial'])):
    for rv, pv, xoff in [(rt, pt, x8[i]-w8/2), (rm, pm, x8[i]+w8/2)]:
        s = sig(pv)
        if s != 'ns' and not np.isnan(rv):
            ax8a.text(xoff, rv + (0.018 if rv >= 0 else -0.035), s,
                      ha='center', va='bottom' if rv >= 0 else 'top',
                      fontsize=7.5, fontweight='bold')
ax8a.axhline(0, color='black', lw=0.8)
ax8a.set_xticks(x8)
ax8a.set_xticklabels(
    [f"{r['Gene']}\n({r['Role'].split('(')[1].rstrip(')') if '(' in r['Role'] else r['Role']})"
     for _, r in pc_df.iterrows()], fontsize=8.5, rotation=25, ha='right')
ax8a.set_ylabel('Spearman ρ  (m6A gene vs AR Activity Score)', fontsize=11, fontweight='bold')
ax8a.set_title('A.  Per-gene ρ(m6A gene, AR Activity Score)  —  Primary PCa vs mCRPC\n',
               fontsize=11, fontweight='bold', loc='left', pad=6)
ax8a.legend(fontsize=9, loc='upper right')

# Panel B: TCGA Primary scatter (z-scored ARS)
xb8 = ars_tcga_z.loc[common_t8].values
yb8 = fi_primary_8.loc[common_t8].values
ax8b.scatter(xb8, yb8, c='#3498db', alpha=0.35, s=14, edgecolors='none')
fin = np.isfinite(xb8) & np.isfinite(yb8)
mb8, bb8 = np.polyfit(xb8[fin], yb8[fin], 1)
xl8 = np.linspace(xb8[fin].min(), xb8[fin].max(), 100)
ax8b.plot(xl8, mb8*xl8+bb8, 'k-', lw=2.5, alpha=0.8)
ax8b.set_xlabel('AR Activity Score (z-score)', fontsize=11, fontweight='bold')
ax8b.set_ylabel('m6A Functional Impact', fontsize=11, fontweight='bold')
ax8b.set_title(f'B.  TCGA Primary PCa  (n={len(common_t8)})\n'
               f'ρ={r_t8:+.3f} [{ci_t8_lo:+.3f},{ci_t8_hi:+.3f}]  p={p_t8:.2e} {sig(p_t8)}',
               fontsize=11, fontweight='bold', loc='left', pad=6)
ax8b.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax8b.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

# Panel C: mCRPC Adeno scatter (now uses mean z-score ARS to match TCGA)
xc8 = ars_adeno_mz.loc[common_m8].values
yc8 = fi_adeno_8.loc[common_m8].values
ax8c.scatter(xc8, yc8, c=cmap_lb.loc[common_m8].values, alpha=0.4, s=14, edgecolors='none')
fin = np.isfinite(xc8) & np.isfinite(yc8)
mc8, bc8 = np.polyfit(xc8[fin], yc8[fin], 1)
xl8c = np.linspace(xc8[fin].min(), xc8[fin].max(), 100)
ax8c.plot(xl8c, mc8*xl8c+bc8, 'k-', lw=2.5, alpha=0.8)
ax8c.set_xlabel('AR Activity Score (z-score)', fontsize=11, fontweight='bold')
ax8c.set_ylabel('m6A Functional Impact', fontsize=11, fontweight='bold')
ax8c.set_title(f'C.  mCRPC Adenocarcinoma  (n={len(common_m8)})\n'
               f'ρ={r_m8:+.3f} [{ci_m8_lo:+.3f},{ci_m8_hi:+.3f}]  p={p_m8:.2e} {sig(p_m8)}',
               fontsize=11, fontweight='bold', loc='left', pad=6)
ax8c.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax8c.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
ax8c.legend(handles=[Patch(facecolor='#2980b9', label='Luminal'),
                     Patch(facecolor='#c0392b', label='Basal'),
                     Patch(facecolor='#95a5a6', label='Unassigned')],
            fontsize=9, loc='upper right')

fig8.suptitle('AR Activity Score × m6A Functional Impact:  Primary PCa → mCRPC',
              fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUTDIR, 'fig2_arars_vs_m6a.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig2_arars_vs_m6a.png")
plt.close()

# =============================================================================
# FIGURE 7 — SURVIVAL SYNERGY: KM FOUR-QUADRANT (ARS × m6A FI) + COX HR
# =============================================================================
print("[Fig 7] Survival — KM four-quadrant + Cox HR ...")

_surv_cols  = ['surv_months', 'vital_status']
_surv_avail = [c for c in _surv_cols if c in meta.columns]
if len(_surv_avail) < 2:
    print("  → Skipping Fig 7: survival columns not found in meta")
else:
    _surv_idx = meta[_surv_cols].dropna().index.intersection(df.index)
    _surv_df  = meta.loc[_surv_idx,
                         _surv_cols + ['AR_Activity_Score', 'm6A_Functional_Impact']
                         ].dropna().copy()
    print(f"  Survival subset: n={len(_surv_df)}")

    _med_ars7 = _surv_df['AR_Activity_Score'].median()
    _med_fi7  = _surv_df['m6A_Functional_Impact'].median()
    _surv_df['AR_hi'] = _surv_df['AR_Activity_Score'] >= _med_ars7
    _surv_df['FI_hi'] = _surv_df['m6A_Functional_Impact'] >= _med_fi7
    _surv_df['Quadrant'] = (
        _surv_df['AR_hi'].astype(str) + '_' + _surv_df['FI_hi'].astype(str)
    ).map({'True_True': 'AR-hi/FI-hi', 'True_False': 'AR-hi/FI-lo',
           'False_True': 'AR-lo/FI-hi', 'False_False': 'AR-lo/FI-lo'})
    _quad_colors7 = {'AR-hi/FI-hi': '#c0392b', 'AR-hi/FI-lo': '#e67e22',
                     'AR-lo/FI-hi': '#3498db', 'AR-lo/FI-lo': '#27ae60'}
    _quad_order7  = ['AR-hi/FI-hi', 'AR-hi/FI-lo', 'AR-lo/FI-hi', 'AR-lo/FI-lo']

    # Log-rank: worst (AR-hi/FI-hi) vs best (AR-lo/FI-lo)
    _g7_best  = _surv_df[_surv_df['Quadrant'] == 'AR-lo/FI-lo']
    _g7_worst = _surv_df[_surv_df['Quadrant'] == 'AR-hi/FI-hi']
    _lr7 = None
    if len(_g7_best) >= 5 and len(_g7_worst) >= 5:
        _lr7 = _logrank_test(_g7_best['surv_months'],  _g7_worst['surv_months'],
                             _g7_best['vital_status'], _g7_worst['vital_status'])
        print(f"  Best vs Worst log-rank p={_lr7.p_value:.4f} {sig(_lr7.p_value)}")

    # Cox PH — standardised covariates
    _cox7_df = _surv_df[['surv_months', 'vital_status',
                          'AR_Activity_Score', 'm6A_Functional_Impact']].copy()
    _cox7_df['ARS_FI_interaction'] = (_cox7_df['AR_Activity_Score'] *
                                      _cox7_df['m6A_Functional_Impact'])
    for _c7 in ['AR_Activity_Score', 'm6A_Functional_Impact', 'ARS_FI_interaction']:
        _cox7_df[_c7] = (_cox7_df[_c7] - _cox7_df[_c7].mean()) / _cox7_df[_c7].std()
    _cph7 = _CPH()
    _cph7.fit(_cox7_df, duration_col='surv_months', event_col='vital_status')
    _cs7  = _cph7.summary.copy()
    print("\n  Cox PH:")
    print(_cs7[['coef', 'exp(coef)', 'p',
                'exp(coef) lower 95%', 'exp(coef) upper 95%']].to_string())

    # --- Plot: 1×2 KM | Cox forest ---
    _fig7, (_ax7km, _ax7cx) = plt.subplots(1, 2, figsize=(18, 8))

    # KM panel
    _kmf7 = _KMF()
    for _qq7 in _quad_order7:
        _gr7 = _surv_df[_surv_df['Quadrant'] == _qq7]
        if len(_gr7) < 5:
            continue
        _kmf7.fit(_gr7['surv_months'], _gr7['vital_status'],
                  label=f"{_qq7} (n={len(_gr7)})")
        _kmf7.plot_survival_function(ax=_ax7km, color=_quad_colors7[_qq7], linewidth=2.5)
    _ax7km.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    _ax7km.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
    _ax7km.set_ylim(0, 1.05)
    _ax7km.set_title('A.  Overall Survival — ARS × m6A FI Four-Quadrant\n'
                     '     Median splits; mCRPC cohort (WCDT)',
                     fontsize=12, fontweight='bold', loc='left', pad=8)
    _ax7km.legend(fontsize=9.5, loc='lower left')
    if _lr7 is not None:
        _ax7km.text(0.97, 0.97,
                    f'Best vs Worst\nLog-rank p={_lr7.p_value:.4f} {sig(_lr7.p_value)}',
                    transform=_ax7km.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=4))

    # Cox HR forest panel
    _cov7 = ['AR_Activity_Score', 'm6A_Functional_Impact', 'ARS_FI_interaction']
    _dn7  = ['AR Activity Score\n(per SD)', 'm6A Functional Impact\n(per SD)',
             'ARS × m6A FI\n(interaction, per SD)']
    _hr7     = _cs7.loc[_cov7, 'exp(coef)'].values.astype(float)
    _hr7_lo  = _cs7.loc[_cov7, 'exp(coef) lower 95%'].values.astype(float)
    _hr7_hi  = _cs7.loc[_cov7, 'exp(coef) upper 95%'].values.astype(float)
    _pv7     = _cs7.loc[_cov7, 'p'].values.astype(float)
    _y7      = np.arange(len(_cov7))

    _ax7cx.axvline(1.0, color='black', lw=1.5, ls='--', alpha=0.6, zorder=1)
    _clr7 = ['#c0392b' if h > 1 else '#27ae60' for h in _hr7]
    for _i7, (_y, _h, _lo, _hi, _p, _cl) in enumerate(
            zip(_y7, _hr7, _hr7_lo, _hr7_hi, _pv7, _clr7)):
        _ax7cx.plot([_lo, _hi], [_y, _y], lw=3, color=_cl, alpha=0.85, zorder=2)
        _ax7cx.scatter([_h], [_y], s=130, color=_cl, zorder=3,
                       edgecolors='black', linewidths=0.8)
        _ax7cx.text(_hi + 0.04, _y,
                    f'HR={_h:.2f} [{_lo:.2f}–{_hi:.2f}]\np={_p:.3f} {sig(_p)}',
                    va='center', ha='left', fontsize=9.5)
    _ax7cx.set_yticks(_y7)
    _ax7cx.set_yticklabels(_dn7, fontsize=11, fontweight='bold')
    _ax7cx.set_xlabel('Hazard Ratio (95% CI)', fontsize=12, fontweight='bold')
    _ax7cx.set_title('B.  Cox Proportional Hazards Model\n'
                     '     Standardised covariates; mCRPC cohort',
                     fontsize=12, fontweight='bold', loc='left', pad=8)
    _ax7cx.set_xlim(max(0.2, _hr7_lo.min() - 0.15), _hr7_hi.max() + 0.9)
    _ax7cx.invert_yaxis()

    _fig7.suptitle('ARS × m6A Functional Impact — Survival Synergy  (mCRPC, WCDT)',
                   fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig7_survival_km_cox.png'),
                dpi=300, bbox_inches='tight')
    print("  → Saved: fig7_survival_km_cox.png")
    plt.close(_fig7)

# =============================================================================
# FIGURE 8 — AR GENOMIC ALTERATION → m6A AXES  (GENOMIC ANCHOR)
# =============================================================================
print("[Fig 8] AR amp/mut → m6A axes ...")

_col_ar8 = 'AR - Amplification and/or Mutation'
if _col_ar8 not in meta.columns:
    print(f"  → Skipping Fig 8: column '{_col_ar8}' not found")
else:
    _idx8_wt  = meta[meta[_col_ar8] == 0.0].index.intersection(df.index)
    _idx8_amp = meta[meta[_col_ar8] == 1.0].index.intersection(df.index)
    print(f"  AR WT: n={len(_idx8_wt)}   AR Amp/Mut: n={len(_idx8_amp)}")

    _axes8 = [
        ('m6A_Net_Deposition',    'Net m6A\nDeposition'),
        ('m6A_Oncogenic_Readout', 'Oncogenic\nReadout'),
        ('m6A_Functional_Impact', 'Functional\nImpact'),
    ]
    _fig8, _ax8_arr = plt.subplots(1, 3, figsize=(18, 7))

    for _ax8, (_col8, _lbl8) in zip(_ax8_arr, _axes8):
        _va8_wt  = meta.loc[_idx8_wt,  _col8].dropna().values
        _va8_amp = meta.loc[_idx8_amp, _col8].dropna().values
        if len(_va8_wt) < 3 or len(_va8_amp) < 3:
            _ax8.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                      transform=_ax8.transAxes)
            continue
        _, _p8 = mannwhitneyu(_va8_wt, _va8_amp, alternative='two-sided')
        _pts8 = _ax8.violinplot([_va8_wt, _va8_amp], positions=[0, 1],
                                showmeans=True, showmedians=True)
        _pts8['bodies'][0].set_facecolor('#95a5a6'); _pts8['bodies'][0].set_alpha(0.65)
        _pts8['bodies'][1].set_facecolor('#e74c3c');  _pts8['bodies'][1].set_alpha(0.65)
        style_violin(_pts8, _ax8)
        _ymax8 = max(np.percentile(_va8_wt, 97), np.percentile(_va8_amp, 97))
        _ymin8 = min(_va8_wt.min(), _va8_amp.min())
        _ytop8 = _ymax8 + abs(_ymax8 - _ymin8) * 0.1
        _ax8.plot([0, 1], [_ytop8, _ytop8], 'k-', lw=1.2)
        _ax8.text(0.5, _ytop8, f'p={_p8:.2e} {sig(_p8)}',
                  ha='center', va='bottom', fontsize=10.5, fontweight='bold')
        _ax8.set_xticks([0, 1])
        _ax8.set_xticklabels([f'AR WT\n(n={len(_va8_wt)})',
                               f'AR Amp/Mut\n(n={len(_va8_amp)})'],
                              fontsize=11, fontweight='bold')
        _ax8.set_ylabel(_lbl8, fontsize=12, fontweight='bold')
        _ax8.set_title(_lbl8, fontsize=13, fontweight='bold')
        _ax8.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
        _d8 = _va8_amp.mean() - _va8_wt.mean()
        _ax8.text(0.5, 0.02, f'Δmean = {_d8:+.3f} (Amp − WT)',
                  ha='center', va='bottom', transform=_ax8.transAxes,
                  fontsize=9, color='#333333', style='italic')
        print(f"  {_col8:25s}: WT={_va8_wt.mean():+.4f}  Amp={_va8_amp.mean():+.4f}  "
              f"Δ={_d8:+.4f}  p={_p8:.2e} {sig(_p8)}")

    _fig8.suptitle(
        f'm6A Axes by AR Genomic Alteration Status  (mCRPC, WCDT)\n'
        f'AR Amplification / Mutation (n={len(_idx8_amp)}) vs AR Wildtype (n={len(_idx8_wt)})',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig8_AR_ampMut_m6a.png'),
                dpi=300, bbox_inches='tight')
    print("  → Saved: fig8_AR_ampMut_m6a.png")
    plt.close(_fig8)

# =============================================================================
# FIGURE 9 — PRERANK GSEA: HALLMARK PATHWAYS × m6A FUNCTIONAL IMPACT
# =============================================================================
import gseapy as _gp
print("[Fig 9] Prerank GSEA — Hallmark × m6A FI ...")

_fi9   = meta_adeno['m6A_Functional_Impact'].dropna()
_cg9   = _fi9.index.intersection(df.index)
_fi9_a = _fi9.loc[_cg9].values
print(f"  Adeno samples for GSEA: n={len(_cg9)}")

# Filter to expressed genes (std > 0.1 across Adeno subset)
_expr9  = df.loc[_cg9].fillna(0.0)
_vmask9 = _expr9.std(axis=0) > 0.1
_expr9  = _expr9.loc[:, _vmask9]
print(f"  Expressed genes: {_vmask9.sum()}")

# Vectorised Spearman: ρ(gene expression, m6A FI) across Adeno samples
_fi9_rk = _sp_rankdata(_fi9_a, method='average')
_fi9_c  = _fi9_rk - _fi9_rk.mean()
_fi9_sd = _fi9_c.std()
_E9     = _expr9.values
_E9_rk  = np.apply_along_axis(lambda c: _sp_rankdata(c, method='average'), 0, _E9)
_E9_c   = _E9_rk - _E9_rk.mean(axis=0)
_E9_sd  = _E9_c.std(axis=0)
_ok9    = _E9_sd > 1e-10
_corrs9 = np.zeros(_E9.shape[1])
_corrs9[_ok9] = (_fi9_c @ _E9_c[:, _ok9]) / (len(_fi9_c) * _fi9_sd * _E9_sd[_ok9])
_rnk9   = pd.Series(_corrs9, index=_expr9.columns).sort_values(ascending=False)
print(f"  Top +: {list(_rnk9.index[:5])}")
print(f"  Top –: {list(_rnk9.index[-5:])}")

_gsea9_ok = False
try:
    _gsea9 = _gp.prerank(
        rnk=_rnk9,
        gene_sets='MSigDB_Hallmark_2020',
        threads=4,
        min_size=10,
        max_size=500,
        permutation_num=500,
        outdir=None,
        seed=42,
        verbose=False,
    )
    _gdf9 = _gsea9.res2d.copy()
    if 'Term' in _gdf9.columns:
        _gdf9 = _gdf9.set_index('Term')
    _gdf9['NES']  = _gdf9['NES'].astype(float)
    _gdf9['fdr']  = _gdf9['FDR q-val'].astype(float)
    _gdf9.index   = (_gdf9.index
                     .str.replace('HALLMARK_', '', regex=False)
                     .str.replace('_', ' ', regex=False)
                     .str.title())
    _top9 = pd.concat([
        _gdf9.nlargest(10, 'NES'),
        _gdf9.nsmallest(10, 'NES'),
    ]).drop_duplicates().sort_values('NES', ascending=False)

    print("\n  Top enriched (m6A FI-high):")
    for _t9, _r9 in _gdf9.nlargest(7, 'NES').iterrows():
        print(f"    {_t9:40s}  NES={_r9['NES']:+.2f}  FDR={_r9['fdr']:.3f}")
    print("  Top depleted:")
    for _t9, _r9 in _gdf9.nsmallest(7, 'NES').iterrows():
        print(f"    {_t9:40s}  NES={_r9['NES']:+.2f}  FDR={_r9['fdr']:.3f}")

    _fig9, _ax9 = plt.subplots(figsize=(14, 10))
    _bc9 = ['#c0392b' if _n9 > 0 else '#2980b9' for _n9 in _top9['NES'].values]
    _ax9.barh(range(len(_top9)), _top9['NES'].values, color=_bc9,
              edgecolor='black', linewidth=0.6, alpha=0.85, zorder=2)
    for _i9, (_t9, _r9) in enumerate(_top9.iterrows()):
        _n9 = float(_r9['NES'])
        _f9 = float(_r9['fdr'])
        _s9 = '***' if _f9 < 0.001 else '**' if _f9 < 0.01 else '*' if _f9 < 0.05 else ''
        _ax9.text(_n9 + (0.04 if _n9 >= 0 else -0.04), _i9,
                  f"FDR={_f9:.3f}{' ' + _s9 if _s9 else ''}",
                  va='center', ha='left' if _n9 >= 0 else 'right', fontsize=7.5)
    _ax9.set_yticks(range(len(_top9)))
    _ax9.set_yticklabels(_top9.index, fontsize=9)
    _ax9.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
    _ax9.set_xlabel('Normalized Enrichment Score (NES)\n'
                    '← depleted in m6A FI-high  |  enriched in m6A FI-high →',
                    fontsize=11, fontweight='bold')
    _ax9.set_title(
        f'Prerank GSEA — Hallmark Pathways by m6A Functional Impact\n'
        f'mCRPC Adenocarcinoma (n={len(_cg9)});  rank metric: ρ(gene, m6A FI)',
        fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig9_gsea_hallmark_m6afi.png'),
                dpi=300, bbox_inches='tight')
    print("  → Saved: fig9_gsea_hallmark_m6afi.png")
    plt.close(_fig9)
    _gsea9_ok = True
except Exception as _e9:
    print(f"  → GSEA failed ({_e9}). Skipping fig9.")

# =============================================================================
# FIGURE 10 — DARANA TREATMENT VALIDATION: enzalutamide pre → post
#
# Design: 3-panel figure
#   Panel A — Paired swarm: ARS pre vs post (Wilcoxon), lines connecting pairs
#   Panel B — Paired swarm: m6A FI pre vs post (Wilcoxon), lines connecting pairs
#   Panel C — Scatter: ΔARS vs Δm6A FI per patient (Spearman ρ + regression)
#
# Scientific question: does AR suppression (ARS↓ post-enzalutamide) track with
# a change in m6A FI? Predicts: ΔARS and Δm6A FI are correlated (positive in
# normal-like primary; inverse if tumor has already acquired mCRPC coupling).
# =============================================================================
print("[Fig 10] DARANA treatment validation ...")
from scipy.stats import wilcoxon as _wilcoxon

df_dar, meta_dar = load_darana()

# Score axes on ALL DARANA samples (z-score within DARANA, consistent units)
_dar_genes  = [g for g in all_genes if g in df_dar.columns]
_z_dar      = zscore_normalize(df_dar[_dar_genes])

_ar_dar     = [g for g in AR_TARGET_GENES if g in _z_dar.columns]
_m6a_dar    = [g for g in CROSS_COHORT_M6A_GENES if g in _z_dar.columns]

# ARS: mean z-score of AR target genes (same metric as Fig 2 mCRPC branch)
_ars_dar = _z_dar[_ar_dar].mean(axis=1)
meta_dar['ARS'] = _ars_dar

# m6A axes using the same writer weights as mCRPC (LR_WEIGHTS already computed
# in Section B above — reuse but recalculate axes from DARANA z-scores)
_wr_dar    = [g for g in WRITER_GENES if g in _z_dar.columns]
_er_dar    = [g for g in ERASER_GENES if g in _z_dar.columns]
_ro_dar    = [g for g in READER_ONCOGENIC if g in _z_dar.columns]
_rs_dar    = [g for g in READER_SUPPRESSIVE if g in _z_dar.columns]

# Simple mean-based axes (no LR_WEIGHTS — those are WCDT-specific; use equal
# weighting within DARANA to avoid cross-cohort weight contamination)
_nd_dar = _z_dar[_wr_dar].mean(axis=1) - _z_dar[_er_dar].mean(axis=1)
_or_dar = _z_dar[_ro_dar].mean(axis=1) - _z_dar[_rs_dar].mean(axis=1)
meta_dar['m6A_FI'] = _nd_dar * 0.435 + _or_dar * 0.565

print(f"  AR genes used in DARANA: {len(_ar_dar)}")
print(f"  m6A genes used in DARANA: writers={len(_wr_dar)}, erasers={len(_er_dar)}, "
      f"readers_onco={len(_ro_dar)}, readers_supp={len(_rs_dar)}")

# Paired subset
_paired_ids = sorted(meta_dar.loc[meta_dar['paired'], 'patient'].unique())
print(f"  Paired patients: n={len(_paired_ids)}")

_pre_vals  = {'ARS': [], 'm6A_FI': []}
_post_vals = {'ARS': [], 'm6A_FI': []}
for _pat in _paired_ids:
    _pre_idx  = meta_dar[(meta_dar['patient'] == _pat) & (meta_dar['timepoint'] == 'pre')].index
    _post_idx = meta_dar[(meta_dar['patient'] == _pat) & (meta_dar['timepoint'] == 'post')].index
    if len(_pre_idx) == 0 or len(_post_idx) == 0:
        continue
    for _ax in ('ARS', 'm6A_FI'):
        _pre_vals[_ax].append(float(meta_dar.loc[_pre_idx[0], _ax]))
        _post_vals[_ax].append(float(meta_dar.loc[_post_idx[0], _ax]))

_pre_ars  = np.array(_pre_vals['ARS'])
_post_ars = np.array(_post_vals['ARS'])
_pre_fi   = np.array(_pre_vals['m6A_FI'])
_post_fi  = np.array(_post_vals['m6A_FI'])
_delta_ars = _post_ars - _pre_ars
_delta_fi  = _post_fi  - _pre_fi
n_pairs_dar = len(_pre_ars)

# Wilcoxon signed-rank tests
_w_ars, _p_ars = _wilcoxon(_pre_ars,  _post_ars,  alternative='two-sided')
_w_fi,  _p_fi  = _wilcoxon(_pre_fi,   _post_fi,   alternative='two-sided')
_r_dlt, _p_dlt = spearmanr(_delta_ars, _delta_fi)

print(f"  ARS pre={_pre_ars.mean():+.3f}  post={_post_ars.mean():+.3f}  "
      f"Δ={_delta_ars.mean():+.3f}  Wilcoxon p={_p_ars:.4f} {sig(_p_ars)}")
print(f"  m6A FI pre={_pre_fi.mean():+.3f}  post={_post_fi.mean():+.3f}  "
      f"Δ={_delta_fi.mean():+.3f}  Wilcoxon p={_p_fi:.4f} {sig(_p_fi)}")
print(f"  ΔARS vs Δm6A FI: ρ={_r_dlt:+.3f}  p={_p_dlt:.4f} {sig(_p_dlt)}  n={n_pairs_dar}")

# ── Figure ─────────────────────────────────────────────────────────────────────
_fig10, (_ax10a, _ax10b, _ax10c) = plt.subplots(1, 3, figsize=(19, 7))
_PAL_DAR = {'pre': '#3498db', 'post': '#e74c3c'}

def _paired_swarm(ax, pre_arr, post_arr, wilcox_p, ylabel, title):
    """Draw connected paired dots (pre vs post) with Wilcoxon annotation."""
    np.random.seed(42)
    _jit = np.random.uniform(-0.07, 0.07, len(pre_arr))
    for _xp, _xq, _j in zip(pre_arr, post_arr, _jit):
        ax.plot([0 + _j, 1 + _j], [_xp, _xq],
                color='grey', lw=0.7, alpha=0.45, zorder=1)
    ax.scatter(np.zeros(len(pre_arr))  + _jit, pre_arr,
               c=_PAL_DAR['pre'],  s=55, zorder=3, edgecolors='black', linewidths=0.5,
               label=f'Pre (n={len(pre_arr)})')
    ax.scatter(np.ones(len(post_arr)) + _jit, post_arr,
               c=_PAL_DAR['post'], s=55, zorder=3, edgecolors='black', linewidths=0.5,
               label=f'Post (n={len(post_arr)})')
    # Mean ± SD bars
    for _xi, _arr in [(0, pre_arr), (1, post_arr)]:
        _m, _s = _arr.mean(), _arr.std()
        ax.plot([_xi - 0.22, _xi + 0.22], [_m, _m], 'k-', lw=2.5, zorder=5)
        ax.plot([_xi, _xi], [_m - _s, _m + _s], 'k-', lw=1.5, zorder=5)
    _ymax = max(pre_arr.max(), post_arr.max())
    _yrange = _ymax - min(pre_arr.min(), post_arr.min())
    _ytop = _ymax + _yrange * 0.12
    ax.plot([0, 1], [_ytop, _ytop], 'k-', lw=1.2)
    ax.text(0.5, _ytop, f'Wilcoxon p={wilcox_p:.4f} {sig(wilcox_p)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre\nEnzalutamide', 'Post\nEnzalutamide'],
                       fontsize=11.5, fontweight='bold')
    _mean_delta = (post_arr - pre_arr).mean()
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nMean Δ = {_mean_delta:+.3f}',
                 fontsize=12, fontweight='bold', loc='left', pad=6)
    ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.4)
    ax.legend(fontsize=9.5, loc='upper right')

_paired_swarm(_ax10a, _pre_ars, _post_ars, _p_ars,
              'AR Activity Score (z-score)', 'A.  AR Activity Score')
_paired_swarm(_ax10b, _pre_fi, _post_fi, _p_fi,
              'm6A Functional Impact (z-score)', 'B.  m6A Functional Impact')

# Panel C: ΔARS vs Δm6A FI scatter
_dircolor = np.where((_delta_ars < 0) & (_delta_fi > 0), '#c0392b',   # AR↓ m6A↑ (predicted)
             np.where((_delta_ars > 0) & (_delta_fi < 0), '#2980b9',   # AR↑ m6A↓
             '#95a5a6'))                                                  # same direction
_ax10c.scatter(_delta_ars, _delta_fi, c=_dircolor,
               s=70, edgecolors='black', linewidths=0.6, alpha=0.85, zorder=3)
# Zero lines
_ax10c.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
_ax10c.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
# Regression
_fin10 = np.isfinite(_delta_ars) & np.isfinite(_delta_fi)
_m10, _b10 = np.polyfit(_delta_ars[_fin10], _delta_fi[_fin10], 1)
_xl10 = np.linspace(_delta_ars[_fin10].min(), _delta_ars[_fin10].max(), 100)
_ax10c.plot(_xl10, _m10 * _xl10 + _b10, 'k-', lw=2.5, alpha=0.8, zorder=4)
# Annotate quadrants
_n_pred = int(((_delta_ars < 0) & (_delta_fi > 0)).sum())
_ax10c.text(0.02, 0.97,
            f'AR↓ & m6A↑ (predicted): {_n_pred}/{n_pairs_dar} patients',
            transform=_ax10c.transAxes, fontsize=9, color='#c0392b',
            va='top', fontweight='bold')
_ax10c.set_xlabel('ΔARS  (post − pre)', fontsize=12, fontweight='bold')
_ax10c.set_ylabel('Δm6A Functional Impact  (post − pre)', fontsize=12, fontweight='bold')
_ax10c.set_title(f'C.  ΔARS vs Δm6A FI per Patient  (n={n_pairs_dar} pairs)\n'
                 f'ρ={_r_dlt:+.3f}  p={_p_dlt:.4f} {sig(_p_dlt)}',
                 fontsize=12, fontweight='bold', loc='left', pad=6)
from matplotlib.patches import Patch as _Patch10
_ax10c.legend(handles=[
    _Patch10(facecolor='#c0392b', label='AR↓ & m6A↑ (inverse)'),
    _Patch10(facecolor='#2980b9', label='AR↑ & m6A↓ (inverse)'),
    _Patch10(facecolor='#95a5a6', label='Same direction'),
], fontsize=9, loc='lower right')

_fig10.suptitle(
    f'DARANA Trial (GSE197780): Enzalutamide Neoadjuvant Treatment — '
    f'AR Activity vs m6A Functional Impact\n'
    f'Primary prostate cancer, n={n_pairs_dar} paired biopsies '
    f'(pre vs ~3 months post enzalutamide)',
    fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig10_darana_treatment.png'),
            dpi=300, bbox_inches='tight')
print("  → Saved: fig10_darana_treatment.png")
plt.close(_fig10)

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("  FIGURE SUMMARY")
print("=" * 80)
summary = [
    ("fig1_arexpr_vs_m6a.png",        "Fig 1 — AR mRNA expression × m6A FI: Primary PCa → mCRPC"),
    ("fig2_arars_vs_m6a.png",         "Fig 2 — AR Activity Score × m6A FI: Primary PCa → mCRPC"),
    ("fig3_dual_trajectory.png",      "Fig 3 — ARS + m6A FI dual trajectory across disease stages"),
    ("fig4_coupling_forest.png",      "Fig 4 — Within-cohort coupling forest (sign-flip thesis, ±95%CI)"),
    ("fig5_per_gene_mcrpc.png",       "Fig 5 — Per-gene mCRPC deconfounded (Adeno-only + partial L/B)"),
    ("fig6_mediation_survival.png",   "Fig 6 — Bootstrap causal mediation  ARS → RBM15B → m6A FI (1×2)"),
    ("fig7_survival_km_cox.png",      "Fig 7 — Survival synergy: KM four-quadrant + Cox HR forest"),
    ("fig8_AR_ampMut_m6a.png",        "Fig 8 — AR amp/mut → m6A axes (genomic anchor, violin)"),
    ("fig9_gsea_hallmark_m6afi.png",  "Fig 9 — Prerank GSEA: Hallmark pathways × m6A FI"),
    ("fig10_darana_treatment.png",    "Fig 10 — DARANA: enzalutamide pre/post ARS + m6A FI (causal validation)"),
]
for fname, desc in summary:
    path = os.path.join(OUTDIR, fname)
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"  [{status}] {desc}")
    print(f"          {path}")

print(f"\n  All figures saved to: {OUTDIR}")
