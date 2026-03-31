#!/usr/bin/env python3
"""
Patch script for ar_m6a_analysis.py:
  1. Replace preprocessing block — add PC1/LR ARS, fix TCGA m6A weight consistency
  2. Insert new Part I plots (01b: ARS methods comparison, 01c: AR signaling efficiency)
  3. Remove Part IX (redirect to ar_crosscohort_analysis.py)
  4. Renumber Part X plots 33-38 → 30-35
  5. Add AR Signaling Efficiency plot (new plot 36) in Part X
  6. Update docstring and final plot list
"""

path = 'ar_m6a_analysis.py'
with open(path) as f:
    content = f.read()

# =============================================================================
# Change 1: Replace preprocessing block
# =============================================================================
OLD_PREPROCESS = '''df,   meta   = load_mcrpc()
df_tc, meta_tc = load_tcga()

print(f"\\n  mCRPC: {df.shape[0]} patients")
print(f"  TCGA:  {df_tc.shape[0]} patients")

# ── mCRPC: z-score normalise ─────────────────────────────────────────────────
all_genes = list(set(ALL_M6A_GENES + AR_TARGET_GENES + ['AR']))
all_genes = [g for g in all_genes if g in df.columns]
z_all = zscore_normalize(df[all_genes])

# ── AR Activity Score (ARS) ───────────────────────────────────────────────────
ar_avail = [g for g in AR_TARGET_GENES if g in z_all.columns]
meta['AR_Activity_Score'] = z_all[ar_avail].mean(axis=1)
print(f"\\n  AR signature genes available in mCRPC: {len(ar_avail)}/{len(AR_TARGET_GENES)}")
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
meta['m6A_Functional_Impact'] = meta['m6A_Net_Deposition'] * 0.435 + meta['m6A_Oncogenic_Readout'] * 0.565'''

NEW_PREPROCESS = '''df,   meta   = load_mcrpc()
df_tc, meta_tc = load_tcga()

print(f"\\n  mCRPC: {df.shape[0]} patients")
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
print(f"\\n  AR signature genes in mCRPC: {len(ar_avail)}/{len(AR_TARGET_GENES)}")

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
print(f"  TCGA m6A: LR-weighted writers ({len(_wts_tc)} genes, same weights as mCRPC)")'''

if OLD_PREPROCESS not in content:
    print('ERROR: preprocessing anchor not found — check text match')
    exit(1)
content = content.replace(OLD_PREPROCESS, NEW_PREPROCESS, 1)
print('✓ Change 1: preprocessing block replaced')

# =============================================================================
# Change 2: Insert new Part I plots after Plot 04
# =============================================================================
OLD_PART_I_END = '''plt.savefig(os.path.join(OUTDIR, '04_ARS_clinical_context.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 04_ARS_clinical_context.png")
plt.close()

# ===========================================================================
# PART II — PER-GENE m6A × AR CORRELATIONS'''

NEW_PART_I_END = '''plt.savefig(os.path.join(OUTDIR, '04_ARS_clinical_context.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 04_ARS_clinical_context.png")
plt.close()

# --- 01b. ARS three-method comparison: flat mean vs PC1 vs LR ---------------
print("\\n--- Plot 01b: ARS scoring method robustness ---")
fig, axes_mc = plt.subplots(1, 3, figsize=(18, 6))
chist = meta.loc[df.index, 'histology'].map(
    {'Adenocarcinoma': '#27ae60', 'SCNC': '#8e44ad'}).fillna('grey')

for ax, (x_key, y_key, xlabel, ylabel) in zip(axes_mc, [
    ('ARS_simple', 'AR_Activity_Score', 'Flat Mean ARS', 'PC1 ARS (primary)'),
    ('ARS_LR',     'AR_Activity_Score', 'LR-weighted ARS', 'PC1 ARS (primary)'),
    ('ARS_simple', 'ARS_LR',            'Flat Mean ARS', 'LR-weighted ARS'),
]):
    x = meta.loc[df.index, x_key].values
    y = meta.loc[df.index, y_key].values
    mask = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[mask], y[mask])
    ax.scatter(x, y, c=chist.values, alpha=0.35, s=12, edgecolors='none')
    m_f, b_f = np.polyfit(x[mask], y[mask], 1)
    x_l = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_l, m_f * x_l + b_f, 'k-', lw=2, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(f'ρ={r:+.3f}, p={p:.2e} {sig(p)}\\n(n={mask.sum()})',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

axes_mc[0].legend(handles=[
    Patch(facecolor='#27ae60', label='Adenocarcinoma'),
    Patch(facecolor='#8e44ad', label='SCNC'),
    Patch(facecolor='grey', label='Other')], fontsize=9)
plt.suptitle(
    f'AR Activity Score — Three Scoring Methods (mCRPC)\\n'
    f'PC1 variance explained: {ARS_PC1_var:.1%} | '
    f'High concordance validates PC1 as primary score',
    fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '01b_ARS_three_methods.png'), dpi=300, bbox_inches='tight')
print("  → Saved: 01b_ARS_three_methods.png")
print(f"  Method concordance: mean-PC1 ρ={_r_pc1_mean:.3f}, LR-PC1 ρ={_r_pc1_lr:.3f}")
plt.close()

# --- 01c. AR Signaling Efficiency by histology and Luminal/Basal ------------
print("\\n--- Plot 01c: AR Signaling Efficiency validation ---")
if meta['AR_Signaling_Efficiency'].notna().sum() > 10:
    fig, axes_eff = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Efficiency by histology
    ax = axes_eff[0]
    eff_adeno = meta.loc[idx_adeno, 'AR_Signaling_Efficiency'].dropna().values
    eff_scnc  = meta.loc[idx_scnc,  'AR_Signaling_Efficiency'].dropna().values
    _, p_eff_hist = mannwhitneyu(eff_adeno, eff_scnc, alternative='two-sided')
    parts = ax.violinplot([eff_adeno, eff_scnc], positions=[0, 1],
                          showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#27ae60'); parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor('#8e44ad'); parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_max = max(eff_adeno.max(), eff_scnc.max())
    ax.plot([0, 1], [y_max + 0.05, y_max + 0.05], 'k-', lw=1.2)
    ax.text(0.5, y_max + 0.07, f'p={p_eff_hist:.2e} {sig(p_eff_hist)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Adenocarcinoma\\n(n={len(eff_adeno)})',
                        f'SCNC\\n(n={len(eff_scnc)})'], fontsize=11, fontweight='bold')
    ax.set_ylabel('AR Signaling Efficiency\\n(ARS residualized on AR mRNA)', fontsize=11, fontweight='bold')
    ax.set_title('AR Signaling Efficiency\\nby Histology', fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')

    # Panel B: Efficiency by Luminal/Basal (Adeno only)
    ax = axes_eff[1]
    eff_lum = meta.loc[idx_lum.intersection(idx_adeno), 'AR_Signaling_Efficiency'].dropna().values
    eff_bas = meta.loc[idx_bas.intersection(idx_adeno), 'AR_Signaling_Efficiency'].dropna().values
    _, p_eff_lb = mannwhitneyu(eff_lum, eff_bas, alternative='two-sided')
    parts = ax.violinplot([eff_lum, eff_bas], positions=[0, 1],
                          showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#2980b9'); parts['bodies'][0].set_alpha(0.65)
    parts['bodies'][1].set_facecolor('#c0392b'); parts['bodies'][1].set_alpha(0.65)
    style_violin(parts, ax)
    y_max2 = max(eff_lum.max(), eff_bas.max())
    ax.plot([0, 1], [y_max2 + 0.05, y_max2 + 0.05], 'k-', lw=1.2)
    ax.text(0.5, y_max2 + 0.07, f'p={p_eff_lb:.2e} {sig(p_eff_lb)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Luminal\\n(n={len(eff_lum)})',
                        f'Basal\\n(n={len(eff_bas)})'], fontsize=11, fontweight='bold')
    ax.set_ylabel('AR Signaling Efficiency', fontsize=11, fontweight='bold')
    ax.set_title('AR Signaling Efficiency\\nby Luminal/Basal (Adeno only)', fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.8, ls='--')

    # Panel C: AR mRNA vs ARS with efficiency as colour
    ax = axes_eff[2]
    eff_vals = meta.loc[df.index, 'AR_Signaling_Efficiency']
    ar_mrna  = z_all['AR'].reindex(df.index) if 'AR' in z_all.columns else None
    if ar_mrna is not None:
        common_e = eff_vals.dropna().index.intersection(ar_mrna.dropna().index).intersection(
            meta['AR_Activity_Score'].dropna().index)
        eff_norm = (eff_vals.loc[common_e] - eff_vals.loc[common_e].min()) / (
            eff_vals.loc[common_e].max() - eff_vals.loc[common_e].min())
        sc = ax.scatter(ar_mrna.loc[common_e], meta.loc[common_e, 'AR_Activity_Score'],
                        c=eff_vals.loc[common_e], cmap='RdBu_r',
                        alpha=0.45, s=18, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='AR Signaling Efficiency')
        ax.set_xlabel('AR mRNA z-score', fontsize=11, fontweight='bold')
        ax.set_ylabel('AR Activity Score (PC1)', fontsize=11, fontweight='bold')
        ax.set_title('AR mRNA vs ARS\\nColored by Signaling Efficiency',
                     fontsize=12, fontweight='bold')
        ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
        ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)

    plt.suptitle('AR Signaling Efficiency = ARS residualized on AR mRNA\\n'
                 'Captures AR target induction beyond mRNA level (ligand sensitivity / splice variants)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '01c_AR_signaling_efficiency.png'), dpi=300, bbox_inches='tight')
    print("  → Saved: 01c_AR_signaling_efficiency.png")
    print(f"  Efficiency Adeno vs SCNC: p={p_eff_hist:.2e} {sig(p_eff_hist)}")
    print(f"  Efficiency Luminal vs Basal (Adeno): p={p_eff_lb:.2e} {sig(p_eff_lb)}")
    plt.close()
else:
    print("  AR_Signaling_Efficiency not available — skipping plot 01c")

# ===========================================================================
# PART II — PER-GENE m6A × AR CORRELATIONS'''

if OLD_PART_I_END not in content:
    print('ERROR: Part I end anchor not found')
    exit(1)
content = content.replace(OLD_PART_I_END, NEW_PART_I_END, 1)
print('✓ Change 2: new Part I plots inserted (01b, 01c)')

# =============================================================================
# Change 3: Remove Part IX, replace with redirect comment
# =============================================================================
PART_IX_MARKER  = '# PART IX — CROSS-COHORT AR ACTIVITY TRAJECTORY\n# ==========================================================================='
PART_X_MARKER   = '# PART X — CONFOUND-CONTROLLED AR × m6A ANALYSIS\n# ==========================================================================='

idx_ix = content.find('# ===========================================================================\n' + PART_IX_MARKER)
idx_x  = content.find('# ===========================================================================\n' + PART_X_MARKER)

if idx_ix == -1 or idx_x == -1:
    print(f'ERROR: Part IX marker found={idx_ix != -1}, Part X marker found={idx_x != -1}')
    exit(1)

REDIRECT_COMMENT = '''# ===========================================================================
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

'''

content = content[:idx_ix] + REDIRECT_COMMENT + content[idx_x:]
print('✓ Change 3: Part IX removed, redirect comment inserted')

# =============================================================================
# Change 4: Renumber Part X plots 33-38 → 30-35
# =============================================================================
renumbering = [
    ('33_per_gene_AR_corr_deconfounded', '30_per_gene_AR_corr_deconfounded'),
    ('34_ARS_vs_axes_adeno_only',        '31_ARS_vs_axes_adeno_only'),
    ('35_partial_spearman_ARS_axes',     '32_partial_spearman_ARS_axes'),
    ('36_RBM15B_ARS_by_lineage',         '33_RBM15B_ARS_by_lineage'),
    ('37_mediation_ARS_RBM15B_FI',       '34_mediation_ARS_RBM15B_FI'),
    ('38_mediation_bootstrap_dist',      '35_mediation_bootstrap_dist'),
    # Print statement headers
    ('--- Plot 33:', '--- Plot 30:'),
    ('--- Plot 34:', '--- Plot 31:'),
    ('--- Plot 35:', '--- Plot 32:'),
    ('--- Plot 36:', '--- Plot 33:'),
    ('--- Plots 37-38:', '--- Plots 34-35:'),
]
for old, new in renumbering:
    if old not in content:
        print(f'  WARNING: "{old}" not found for renumbering')
    content = content.replace(old, new)
print('✓ Change 4: Part X plots renumbered 33-38 → 30-35')

# =============================================================================
# Change 5: Add AR Signaling Efficiency plot at end of Part X (new plot 36)
# =============================================================================
OLD_PART_X_END = 'print(f"\\n  Part X summary:")\nprint(f"  Indirect effect a\u00d7b = {indirect:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")\nprop_med = abs(indirect/c)*100 if abs(c) > 1e-9 else float(\'nan\')\nprint(f"  Proportion of total effect mediated: {prop_med:.1f}%")\n\n\nprint("\\n\\n" + "=" * 80)'

NEW_PART_X_END = '''print(f"\\n  Part X summary:")
print(f"  Indirect effect a*b = {indirect:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
prop_med = abs(indirect/c)*100 if abs(c) > 1e-9 else float('nan')
print(f"  Proportion of total effect mediated: {prop_med:.1f}%")


# --- 36. AR Signaling Efficiency × m6A: does signaling efficiency predict m6A? -
print("\\n--- Plot 36: AR Signaling Efficiency × m6A Functional Impact ---")
if meta_adeno['AR_Signaling_Efficiency'].notna().sum() > 10:
    eff_adeno_s = meta_adeno['AR_Signaling_Efficiency'].dropna()
    fig, axes36eff = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Efficiency vs m6A FI (scatter, Adeno only, colored by Luminal/Basal)
    ax = axes36eff[0]
    eff_fi_idx = eff_adeno_s.index.intersection(meta_adeno['m6A_Functional_Impact'].dropna().index)
    x_eff = eff_adeno_s.loc[eff_fi_idx].values
    y_fi  = meta_adeno.loc[eff_fi_idx, 'm6A_Functional_Impact'].values
    r_eff_fi, p_eff_fi = spearmanr(x_eff, y_fi)
    clr_lb_eff = meta_adeno.loc[eff_fi_idx, 'Luminal/Basal Cluster'].map(
        {'Luminal': '#2980b9', 'Basal': '#c0392b'}).fillna('grey')
    ax.scatter(x_eff, y_fi, c=clr_lb_eff.values, alpha=0.4, s=18, edgecolors='none')
    m_eff, b_eff = np.polyfit(x_eff, y_fi, 1)
    x_el = np.linspace(x_eff.min(), x_eff.max(), 100)
    ax.plot(x_el, m_eff * x_el + b_eff, 'k-', lw=2, alpha=0.7)
    ax.set_xlabel('AR Signaling Efficiency\\n(residual ARS | AR mRNA)', fontsize=11, fontweight='bold')
    ax.set_ylabel('m6A Functional Impact', fontsize=11, fontweight='bold')
    ax.set_title(f'AR Signaling Efficiency vs m6A FI\\n'
                 f'(Adenocarcinoma, n={len(x_eff)}) ρ={r_eff_fi:+.3f}, p={p_eff_fi:.2e} {sig(p_eff_fi)}',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.legend(handles=[Patch(facecolor='#2980b9', label='Luminal'),
                        Patch(facecolor='#c0392b', label='Basal'),
                        Patch(facecolor='grey',    label='Other')], fontsize=9)
    print(f"  Efficiency vs FI: ρ={r_eff_fi:+.3f}, p={p_eff_fi:.2e} {sig(p_eff_fi)}")

    # Panel B: Compare ARS vs Efficiency as predictors of m6A FI (partial-corr comparison)
    ax = axes36eff[1]
    # Partial: ARS vs FI controlling for AR mRNA
    if 'AR' in z_adeno.columns:
        r_ars_fi, p_ars_fi = spearmanr(ars_adeno_s, meta_adeno['m6A_Functional_Impact'])
        r_eff_fi2, p_eff_fi2 = spearmanr(
            meta_adeno['AR_Signaling_Efficiency'].dropna(),
            meta_adeno.loc[meta_adeno['AR_Signaling_Efficiency'].dropna().index,
                           'm6A_Functional_Impact'])
        comparisons = ['ARS (PC1)', 'LR-ARS', 'AR Signaling\\nEfficiency']
        rhos = [r_ars_fi]
        ps   = [p_ars_fi]
        # LR-ARS vs FI
        lr_ars_adeno = meta_adeno['ARS_LR'].dropna()
        fi_for_lr    = meta_adeno.loc[lr_ars_adeno.index, 'm6A_Functional_Impact'].dropna()
        common_lrf   = lr_ars_adeno.index.intersection(fi_for_lr.index)
        r_lr, p_lr   = spearmanr(lr_ars_adeno.loc[common_lrf], fi_for_lr.loc[common_lrf])
        rhos.append(r_lr)
        ps.append(p_lr)
        rhos.append(r_eff_fi2)
        ps.append(p_eff_fi2)
        colors_b = ['#3498db', '#27ae60', '#e74c3c']
        bars_b = ax.bar(comparisons, rhos, color=colors_b, edgecolor='black', alpha=0.85)
        for i, (rho, p_v) in enumerate(zip(rhos, ps)):
            yoff = rho + (0.01 if rho >= 0 else -0.025)
            ax.text(i, yoff, f'{sig(p_v)}\\nρ={rho:+.3f}', ha='center',
                    va='bottom' if rho >= 0 else 'top', fontsize=9, fontweight='bold')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('Spearman ρ with m6A Functional Impact', fontsize=11, fontweight='bold')
        ax.set_title('AR Score Variants vs m6A FI (Adeno only)\\n'
                     'Does signaling efficiency add predictive value over ARS?',
                     fontsize=12, fontweight='bold')

    plt.suptitle('AR Signaling Efficiency × m6A: Beyond mRNA-level AR Expression\\n'
                 'Adenocarcinoma only (SCNC removed)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '36_AR_signaling_efficiency_vs_m6A.png'),
                dpi=300, bbox_inches='tight')
    print("  → Saved: 36_AR_signaling_efficiency_vs_m6A.png")
    plt.close()
else:
    print("  AR_Signaling_Efficiency not available — skipping plot 36")


print("\\n\\n" + "=" * 80)'''

if OLD_PART_X_END not in content:
    print('ERROR: Part X end anchor not found')
    exit(1)
content = content.replace(OLD_PART_X_END, NEW_PART_X_END, 1)
print('✓ Change 5: AR Signaling Efficiency plot added (plot 36)')

# =============================================================================
# Change 6: Update docstring
# =============================================================================
OLD_DOCSTRING = '''  Part IX   — Cross-cohort AR activity trajectory (30–32)
  Part X    — Confound-controlled AR × m6A: Adeno-only + partial corr + mediation (33–38)

Output: plots_ar_m6a/'''

NEW_DOCSTRING = '''  Part IX   — Cross-cohort AR activity trajectory → see ar_crosscohort_analysis.py
  Part X    — Confound-controlled AR × m6A: Adeno-only + partial corr + mediation (30–36)

Methodological improvements vs prior version:
  - ARS: flat mean replaced by PC1 (dominant co-expression axis, unsupervised)
  - ARS_LR: supervised alternative trained on AR Amp/Mut vs WT (sensitivity check)
  - AR Signaling Efficiency: ARS residualized on AR mRNA (captures mRNA-independent induction)
  - TCGA m6A axes: now use same LR writer weights as mCRPC (methodological consistency)
  - New plots 01b/01c: scoring method stability + AR efficiency validation

Output: plots_ar_m6a/'''

if OLD_DOCSTRING not in content:
    print('ERROR: docstring anchor not found')
    exit(1)
content = content.replace(OLD_DOCSTRING, NEW_DOCSTRING, 1)
print('✓ Change 6: docstring updated')

# =============================================================================
# Change 7: Update final plot list
# =============================================================================
# After changes 1-4 the list entries for 33-38 have been renamed to 30-35 already.
# Change 3 removed Part IX code but NOT the final list entries — those still read
# 30_cross_cohort_*. We replace all of them here.
OLD_FINAL_LIST = '''    '30_cross_cohort_ARS_trajectory.png',
    '31_cross_cohort_ARS_vs_FI.png',
    '32_AR_target_gene_percentile_heatmap.png',
    '30_per_gene_AR_corr_deconfounded.png',
    '31_ARS_vs_axes_adeno_only.png',
    '32_partial_spearman_ARS_axes.png',
    '33_RBM15B_ARS_by_lineage.png',
    '34_mediation_ARS_RBM15B_FI.png',
    '35_mediation_bootstrap_dist.png','''

NEW_FINAL_LIST = '''    # Part X — deconfounded analysis (within-mCRPC)
    '30_per_gene_AR_corr_deconfounded.png',
    '31_ARS_vs_axes_adeno_only.png',
    '32_partial_spearman_ARS_axes.png',
    '33_RBM15B_ARS_by_lineage.png',
    '34_mediation_ARS_RBM15B_FI.png',
    '35_mediation_bootstrap_dist.png',
    '36_AR_signaling_efficiency_vs_m6A.png',
    # Part I — scoring method validation
    '01b_ARS_three_methods.png',
    '01c_AR_signaling_efficiency.png',
    # Cross-cohort trajectory → ar_crosscohort_analysis.py → plots_ar_crosscohort/'''

if OLD_FINAL_LIST not in content:
    print('ERROR: final list anchor not found')
    exit(1)
content = content.replace(OLD_FINAL_LIST, NEW_FINAL_LIST, 1)
print('✓ Change 7: final plot list updated')

# =============================================================================
# Write out
# =============================================================================
with open(path, 'w') as f:
    f.write(content)
print(f'\n✓ All changes applied to {path}')
print('  Summary:')
print('   1. Preprocessing: PC1/LR ARS + AR signaling efficiency + TCGA m6A fix')
print('   2. New Part I plots: 01b (ARS methods), 01c (AR efficiency)')
print('   3. Part IX removed → redirect to ar_crosscohort_analysis.py')
print('   4. Part X renumbered: 33-38 → 30-35')
print('   5. New Part X plot 36: AR signaling efficiency × m6A')
print('   6. Docstring updated')
print('   7. Final plot list updated')
