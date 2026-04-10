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
from scipy.stats import t as _t_dist
from sklearn.decomposition import PCA as _PCA
from sklearn.linear_model import LogisticRegression as _LR
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

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
)
from m6a.stats import sig
from m6a.normalization import zscore_normalize, percentile_rank_matrix
from m6a.scoring import compute_axes
from m6a.plotting import style_violin
from m6a.data.loaders import (
    load_mcrpc, load_tcga, load_gtex, load_adj_normal, load_mcspc,
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

all_genes = list(set(ALL_M6A_GENES + AR_TARGET_GENES + ['AR']))
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

# ARS — PC1 (primary)
ar_in_data = [g for g in AR_TARGET_GENES if g in z_all.columns]
ARS_simple = z_all[ar_in_data].mean(axis=1)
_pca_ar    = _PCA(n_components=1, random_state=42)
_pc1_raw   = _pca_ar.fit_transform(z_all[ar_in_data].values)[:, 0]
if np.corrcoef(_pc1_raw, ARS_simple.values)[0, 1] < 0:
    _pc1_raw = -_pc1_raw
meta['AR_Activity_Score'] = pd.Series(_pc1_raw, index=z_all.index)

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

# Per-gene: full / Adeno-only / partial
pc_rows = []
for gene in gene_order:
    r_f, p_f = spearmanr(z_all.loc[ars.index, gene], ars)
    r_a, p_a = spearmanr(z_adeno[gene], ars_adeno_s)
    r_p, p_p, _ = partial_spearman(z_lb[gene].values, ars_lb.values, lb_cov)
    pc_rows.append({'Gene': gene, 'Role': gene_roles[gene],
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

a, b, c, cp_val = float(a_coef[0]), float(b_coef[0]), float(c_coef[0]), float(cp_coef[0])
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
prop_med      = abs(indirect / c) * 100 if abs(c) > 1e-9 else float('nan')

print(f"  Path a  β={a:+.4f} {sig(a_p[0])},  Path b  β={b:+.4f} {sig(b_p[0])}")
print(f"  Total c β={c:+.4f} {sig(c_p[0])},  Direct c' β={cp_val:+.4f} {sig(cp_p[0])}")
print(f"  Indirect a×b = {indirect:+.4f}  95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p_boot:.4f} {sig(p_boot)}")
print(f"  Proportion mediated: {prop_med:.1f}%")

# =============================================================================
# FIGURE 3 — DUAL TRAJECTORY: ARS + m6A FI across disease stages
# =============================================================================
print("\n[Fig 3] Dual trajectory ...")

_, p_ars_kw = kruskal(*[g.dropna().values for g in ars_groups])
_, p_fi_kw  = kruskal(*[g.dropna().values for g in fi_groups])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

# Panel A — ARS
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
ax1.set_title(f'A.  AR Activity Score Trajectory  (KW p={p_ars_kw:.2e} {sig(p_ars_kw)})',
              fontsize=13, fontweight='bold', loc='left', pad=8)
ax1.legend(fontsize=9, loc='lower right')

# Panel B — m6A FI
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
ax2.set_title(f'B.  m6A Functional Impact Trajectory  (KW p={p_fi_kw:.2e} {sig(p_fi_kw)})',
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

plt.suptitle('AR Activity and m6A Functional Impact Across Prostate Cancer Progression',
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

# --- Panel D: RBM15 vs RBM15B colored by ARS quartile (NEW) -----------------
# Shows the paralog balance shift: AR-high tumors tip toward RBM15B
x_d_idx = ars.index  # all mCRPC samples with ARS
x_d = z_all.loc[x_d_idx, 'RBM15'].values
y_d = z_all.loc[x_d_idx, 'RBM15B'].values
q_d = q_bins.loc[x_d_idx].values
clr_d = np.array([q_palette.get(str(q), '#aaaaaa') for q in q_d])

# Plot each quartile as separate scatter for legend
for q_lbl, q_col in q_palette.items():
    mask_q = q_bins.loc[x_d_idx] == q_lbl
    ax_d.scatter(x_d[mask_q.values], y_d[mask_q.values],
                 c=q_col, alpha=0.45, s=16, edgecolors='none', label=q_lbl)

# Trend lines per quartile
for q_lbl, q_col in q_palette.items():
    mask_q = (q_bins.loc[x_d_idx] == q_lbl).values
    xq, yq = x_d[mask_q], y_d[mask_q]
    fin = np.isfinite(xq) & np.isfinite(yq)
    if fin.sum() > 10:
        m_q, b_q = np.polyfit(xq[fin], yq[fin], 1)
        xl_q = np.linspace(xq[fin].min(), xq[fin].max(), 80)
        ax_d.plot(xl_q, m_q * xl_q + b_q, color=q_col, lw=1.8, alpha=0.85)

# Diagonal of equality
lim_d = max(abs(x_d).max(), abs(y_d).max()) * 1.05
ax_d.plot([-lim_d, lim_d], [-lim_d, lim_d], 'k--', lw=1, alpha=0.4, label='RBM15 = RBM15B')
ax_d.axhline(0, color='grey', lw=0.5, ls=':', alpha=0.4)
ax_d.axvline(0, color='grey', lw=0.5, ls=':', alpha=0.4)
ax_d.set_xlabel('RBM15 z-score\n(Writer-Targeting paralog)', fontsize=10, fontweight='bold')
ax_d.set_ylabel('RBM15B z-score\n(Writer-Targeting paralog)', fontsize=10, fontweight='bold')
ax_d.set_title('D.  RBM15 ↔ RBM15B Paralog Balance by ARS Quartile\n'
               'AR-high tumors shift toward RBM15B (above diagonal)',
               fontsize=10, fontweight='bold', loc='left', pad=5)
ax_d.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

# Global legend for Luminal/Basal (panels B and C)
fig.text(0.01, 0.01, '★ Panels B & C: color = Luminal (blue) / Basal (red) / Unassigned (grey)',
         fontsize=8.5, style='italic', color='#555555')

plt.suptitle('mCRPC: Per-gene m6A × ARS Correlations (Adeno-only and L/B Deconfounded)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUTDIR, 'fig5_per_gene_mcrpc.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig5_per_gene_mcrpc.png")
plt.close()

# =============================================================================
# FIGURE 6 — MEDIATION + SURVIVAL  (2×2 merged figure)
# =============================================================================
print("[Fig 6] Mediation + survival ...")

fig6_ms = plt.figure(figsize=(18, 13))
gs6_ms  = gridspec.GridSpec(2, 2, figure=fig6_ms, hspace=0.45, wspace=0.38)
ax_path = fig6_ms.add_subplot(gs6_ms[0, 0])
ax_boot = fig6_ms.add_subplot(gs6_ms[0, 1])
ax_km   = fig6_ms.add_subplot(gs6_ms[1, 0])
ax_cox  = fig6_ms.add_subplot(gs6_ms[1, 1])

# --- Panel A: path diagram ---------------------------------------------------
ax_path.set_xlim(0, 10)
ax_path.set_ylim(0, 6)
ax_path.axis('off')

box_specs = [
    (0.3, 2.2, 2.2, 1.4, 'AR Activity\nScore (X)', '#3498db'),
    (3.9, 4.2, 2.2, 1.4, 'RBM15B z-score\n(Mediator M)', '#e67e22'),
    (7.5, 2.2, 2.2, 1.4, 'm6A Functional\nImpact (Y)', '#e74c3c'),
]
for (x0, y0, w_box, h_box, lbl, clr) in box_specs:
    rect = plt.Rectangle((x0, y0), w_box, h_box, linewidth=2,
                          edgecolor=clr, facecolor=clr, alpha=0.18, zorder=2)
    ax_path.add_patch(rect)
    ax_path.text(x0 + w_box/2, y0 + h_box/2, lbl, ha='center', va='center',
                 fontsize=11, fontweight='bold', color=clr, zorder=3)

# X → M  (path a)
ax_path.annotate('', xy=(3.9, 5.0), xytext=(2.5, 3.6),
                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5,
                                 connectionstyle='arc3,rad=-0.2'))
ax_path.text(2.9, 4.8, f'a = {a:+.3f} {sig(a_p[0])}',
             fontsize=10.5, color='#e67e22', fontweight='bold', ha='center')

# M → Y  (path b)
ax_path.annotate('', xy=(7.5, 5.0), xytext=(6.1, 5.0),
                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5))
ax_path.text(6.8, 5.3, f'b = {b:+.3f} {sig(b_p[0])}',
             fontsize=10.5, color='#e67e22', fontweight='bold', ha='center')

# X → Y  (direct c')
ax_path.annotate('', xy=(7.5, 2.9), xytext=(2.5, 2.9),
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
ax_path.text(5.0, 2.55, f"c' = {cp_val:+.3f} {sig(cp_p[0])} (direct)",
             fontsize=10.5, color='#3498db', fontweight='bold', ha='center')

zero_out = 'CI excludes zero' if (ci_lo > 0 or ci_hi < 0) else 'CI includes zero'
ax_path.text(5.0, 1.35,
             f'Indirect effect  a×b = {indirect:+.4f}\n'
             f'Bootstrap 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]  ({zero_out})\n'
             f'p = {p_boot:.4f} {sig(p_boot)}   |   Proportion mediated ≈ {prop_med:.1f}%',
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(facecolor='lightyellow', edgecolor='goldenrod', alpha=0.9, pad=6))

ax_path.set_title('A.  Mediation Path  (standardised OLS coefficients,\n'
                   '      Adenocarcinoma only)',
                   fontsize=12, fontweight='bold', loc='left', pad=8)

# --- Panel B: bootstrap distribution ----------------------------------------
ax_boot.hist(boot_indirect, bins=80, color='#e67e22', edgecolor='none', alpha=0.7)
ax_boot.axvline(0,        color='black', lw=1.5, ls='--', label='Zero')
ax_boot.axvline(indirect, color='#c0392b', lw=2.5,
                label=f'Observed a×b = {indirect:+.4f}')
ax_boot.axvline(ci_lo, color='grey', lw=1.5, ls=':',
                label=f'95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')
ax_boot.axvline(ci_hi, color='grey', lw=1.5, ls=':')
ax_boot.set_xlabel('Indirect effect (a×b)', fontsize=12, fontweight='bold')
ax_boot.set_ylabel('Bootstrap frequency (n=5000)', fontsize=12, fontweight='bold')
ax_boot.set_title(f'B.  Bootstrap Distribution of Indirect Effect\n'
                   f'ARS → RBM15B → m6A FI   (n={n_med} Adeno samples)',
                   fontsize=12, fontweight='bold', loc='left', pad=8)
ax_boot.legend(fontsize=10)

# — Clinical panels (C: KM, D: Cox) follow —

from lifelines import CoxPHFitter

surv_idx = meta[['surv_months', 'vital_status']].dropna().index.intersection(df.index)
surv_df  = meta.loc[surv_idx, ['surv_months', 'vital_status',
                                 'AR_Activity_Score', 'm6A_Functional_Impact']].dropna()
print(f"  Survival subset: n={len(surv_df)}")

# ----- Cox model (continuous, standardised) ----------------------------------
cox_df = surv_df.copy()
cox_df['AR_FI_interaction'] = cox_df['AR_Activity_Score'] * cox_df['m6A_Functional_Impact']
for col in ['AR_Activity_Score', 'm6A_Functional_Impact', 'AR_FI_interaction']:
    cox_df[col] = (cox_df[col] - cox_df[col].mean()) / cox_df[col].std()

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='surv_months', event_col='vital_status')
print("\n  Cox PH (continuous, standardised):")
print(cph.summary[['coef', 'exp(coef)', 'p', 'coef lower 95%', 'coef upper 95%']].to_string())

cox_sum = cph.summary.copy()

# ----- KM: ARS median split --------------------------------------------------
med_ars  = surv_df['AR_Activity_Score'].median()
ars_hi_s = surv_df[surv_df['AR_Activity_Score'] >= med_ars]
ars_lo_s = surv_df[surv_df['AR_Activity_Score'] <  med_ars]
lr_ars   = logrank_test(ars_hi_s['surv_months'], ars_lo_s['surv_months'],
                         ars_hi_s['vital_status'], ars_lo_s['vital_status'])

# Panel A — KM
kmf = KaplanMeierFitter()
kmf.fit(ars_hi_s['surv_months'], ars_hi_s['vital_status'],
        label=f'High ARS (n={len(ars_hi_s)})')
kmf.plot_survival_function(ax=ax_km, color='#e74c3c', linewidth=2.2,
                            ci_show=True, ci_alpha=0.12)
kmf.fit(ars_lo_s['surv_months'], ars_lo_s['vital_status'],
        label=f'Low ARS (n={len(ars_lo_s)})')
kmf.plot_survival_function(ax=ax_km, color='#3498db', linewidth=2.2,
                            ci_show=True, ci_alpha=0.12)
ax_km.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
ax_km.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
ax_km.set_title(f'C.  Overall Survival by AR Activity Score\n'
                f'Median split (n={len(surv_df)})  '
                f'Log-rank p={lr_ars.p_value:.4f} {sig(lr_ars.p_value)}',
                fontsize=12, fontweight='bold', loc='left', pad=8)
ax_km.set_ylim(0, 1.05)
ax_km.legend(fontsize=11, loc='lower left')

# Panel B — Cox forest
labels_map = {
    'AR_Activity_Score':    'AR Activity Score\n(per SD)',
    'm6A_Functional_Impact':'m6A Functional Impact\n(per SD)',
    'AR_FI_interaction':    'AR × m6A Interaction\n(per SD)',
}
hr_colors = {'AR_Activity_Score': '#e74c3c',
             'm6A_Functional_Impact': '#3498db',
             'AR_FI_interaction': '#8e44ad'}

y_ticks, y_labels = [], []
for i, (varname, row) in enumerate(cox_sum.iterrows()):
    hr  = row['exp(coef)']
    lo  = np.exp(row['coef lower 95%'])
    hi  = np.exp(row['coef upper 95%'])
    p_v = row['p']
    c   = hr_colors.get(varname, '#555555')
    ax_cox.plot([lo, hi], [i, i], '-', color=c, lw=4, alpha=0.65, solid_capstyle='round')
    ax_cox.plot(hr, i, 'o', color=c, markersize=11, zorder=5,
                markeredgecolor='white', markeredgewidth=1.5)
    ax_cox.text(hi + 0.01, i,
                f'  HR={hr:.2f}  [{lo:.2f}–{hi:.2f}]\n  p={p_v:.3f} {sig(p_v)}',
                ha='left', va='center', fontsize=9.5, color=c, fontweight='bold')
    y_ticks.append(i)
    y_labels.append(labels_map.get(varname, varname))

ax_cox.axvline(1.0, color='black', lw=1.2, ls='--', alpha=0.7)
ax_cox.set_yticks(y_ticks)
ax_cox.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
ax_cox.set_xlabel('Hazard Ratio (95% CI)', fontsize=12, fontweight='bold')
ax_cox.set_title(f'D.  Cox Proportional Hazards — Continuous Predictors\n'
                 f'All variables standardised (n={len(surv_df)} with survival data)',
                 fontsize=12, fontweight='bold', loc='left', pad=8)
# Auto x-lim with some right padding for text
all_hi = [np.exp(r['coef upper 95%']) for _, r in cox_sum.iterrows()]
ax_cox.set_xlim(0, max(all_hi) * 2.2)
ax_cox.set_ylim(-0.6, len(cox_sum) - 0.4)

fig6_ms.suptitle('Mechanism and Clinical Relevance: Mediation + Survival in mCRPC',
                 fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig6_mediation_survival.png'), dpi=300, bbox_inches='tight')
print("  → Saved: fig6_mediation_survival.png")
plt.close()

# TCGA z-scored matrix (computed once, shared by Figs 1 and 2)
z_tcga  = zscore_normalize(df_tcga[[g for g in gene_order + ['AR'] if g in df_tcga.columns]])
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

# Per-gene mCRPC partial ρ(m6A gene, AR mRNA)
h1_rows = []
for gene in gene_order:
    r_p, p_p, _ = partial_spearman(z_lb[gene].values, ar_expr_lb, lb_cov)
    h1_rows.append({'Gene': gene, 'Role': gene_roles[gene], 'rho_partial': r_p, 'p_partial': p_p})
h1_df = pd.DataFrame(h1_rows)

# Per-gene TCGA ρ(m6A gene, AR mRNA)
h1_tcga_rows = []
for gene in gene_order:
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

# Per-gene TCGA ρ(m6A gene, ARS) — using z-scored ARS
h2_tcga_rows = []
for gene in gene_order:
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

# Aggregate mCRPC scatter (same as Fig 3 — shown here for direct comparison)
ars_adeno_8  = meta_adeno['AR_Activity_Score']
fi_adeno_8   = meta_adeno['m6A_Functional_Impact']
common_m8    = ars_adeno_8.dropna().index.intersection(fi_adeno_8.dropna().index)
r_m8, p_m8  = spearmanr(ars_adeno_8.loc[common_m8], fi_adeno_8.loc[common_m8])
ci_m8_lo, ci_m8_hi = spearman_ci(r_m8, len(common_m8))
print(f"  mCRPC Adeno   ρ(ARS, m6A FI)={r_m8:+.3f} [{ci_m8_lo:+.3f},{ci_m8_hi:+.3f}] {sig(p_m8)} (n={len(common_m8)})")

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

# Panel C: mCRPC Adeno scatter
xc8 = ars_adeno_8.loc[common_m8].values
yc8 = fi_adeno_8.loc[common_m8].values
ax8c.scatter(xc8, yc8, c=cmap_lb.loc[common_m8].values, alpha=0.4, s=14, edgecolors='none')
fin = np.isfinite(xc8) & np.isfinite(yc8)
mc8, bc8 = np.polyfit(xc8[fin], yc8[fin], 1)
xl8c = np.linspace(xc8[fin].min(), xc8[fin].max(), 100)
ax8c.plot(xl8c, mc8*xl8c+bc8, 'k-', lw=2.5, alpha=0.8)
ax8c.set_xlabel('AR Activity Score (PC1)', fontsize=11, fontweight='bold')
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
    ("fig6_mediation_survival.png",   "Fig 6 — Mediation (ARS→RBM15B→FI) + KM + Cox PH (merged 2×2)"),
]
for fname, desc in summary:
    path = os.path.join(OUTDIR, fname)
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"  [{status}] {desc}")
    print(f"          {path}")

print(f"\n  All figures saved to: {OUTDIR}")
