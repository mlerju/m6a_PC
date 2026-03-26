"""
m6a.scoring — Axis score computation and pre-computed weight constants.

The three-axis model:
  Axis 1: Net Deposition    = LR-weighted mean(Writer pct) − mean(Eraser pct)
  Axis 2: Oncogenic Readout = mean(Onco-Reader pct) − mean(Supp-Reader pct)
  Axis 3: Functional Impact = W_ND × Net Deposition + W_OR × Oncogenic Readout

LR_WEIGHTS were trained on the mCRPC cohort (59 SCNC vs 575 Adeno) using L2
logistic regression on z-scored writer genes.  The expanded 22-gene cross-cohort
set adds METTL16 (weight 0.3163) to the 7-gene mCRPC set.
Stage-1 writer-LR AUC = 0.793 (5-fold CV), LOSO-CV ≈ 0.82.

W_ND / W_OR were kept at Stage-2 logistic regression values (0.435 / 0.565);
Stage-2 axis-LR AUC = 0.568 (insufficient to trust a retrained value).

Functions
---------
compute_axes(pct_df)
    Compute 3 axis scores from a percentile-rank DataFrame.
    Returns (net_deposition, oncogenic_readout, functional_impact) as Series.
get_vals(score_col, idx_set, meta_df)
    Extract non-NaN values for a given axis and sample index from a metadata DF.
"""
import numpy as np
from m6a.genes import WRITER_GENES, ERASER_GENES, READER_ONCOGENIC, READER_SUPPRESSIVE

# ── Pre-computed LR writer weights (cross-cohort 22-gene set) ────────────────
LR_WEIGHTS = {
    'METTL3':  0.9413,
    'METTL14': 0.8335,
    'WTAP':    0.6007,
    'ZC3H13':  0.7457,
    'RBM15':   2.73,
    'RBM15B':  0.6181,
    'CBLL1':   1.8953,
    'METTL16': 0.3163,
}

# ── Axis combination weights ──────────────────────────────────────────────────
W_ND = 0.435   # weight for Net Deposition in Functional Impact
W_OR = 0.565   # weight for Oncogenic Readout in Functional Impact


def compute_axes(pct_df):
    """
    Compute the three m6A axis scores from a percentile-rank expression matrix.

    The input should be restricted to the 22 m6A genes (ALL_M6A_GENES) and have
    already been percentile-rank normalized via ``percentile_rank_matrix()``.

    Parameters
    ----------
    pct_df : DataFrame, shape (n_samples, ≥22 m6A genes)

    Returns
    -------
    net_deposition : Series
    oncogenic_readout : Series
    functional_impact : Series
        All in units of Δ percentile points.
    """
    lrw_total = sum(LR_WEIGHTS.values())
    writer_pct = (
        sum(pct_df[g] * w for g, w in LR_WEIGHTS.items()) / lrw_total
    )
    eraser_pct = pct_df[ERASER_GENES].mean(axis=1)
    onco_pct   = pct_df[READER_ONCOGENIC].mean(axis=1)
    supp_pct   = pct_df[READER_SUPPRESSIVE].mean(axis=1)

    net_dep     = writer_pct - eraser_pct
    onco_rdout  = onco_pct - supp_pct
    func_impact = W_ND * net_dep + W_OR * onco_rdout

    return net_dep, onco_rdout, func_impact


def get_vals(score_col, idx_set, meta_df):
    """
    Extract non-NaN values for *score_col* from *meta_df* for the samples in
    *idx_set*.

    Parameters
    ----------
    score_col : str        Column name in meta_df (e.g. 'm6A_Net_Deposition').
    idx_set   : Index      Sample identifiers to select.
    meta_df   : DataFrame  Metadata frame that contains score_col.

    Returns
    -------
    ndarray of float values.
    """
    common = idx_set.intersection(meta_df.index)
    return meta_df.loc[common, score_col].dropna().values
