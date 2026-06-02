"""
m6a.stats — Statistical utilities shared across cross-cohort and mCRPC analyses.

Functions
---------
sig(p)                      Significance asterisks string.
fmt_p(p)                    Formatted p-value string.
rankbiserial(a, b)          Rank-biserial correlation (effect size for Mann-Whitney U).
bh_fdr(pvals)               Benjamini-Hochberg FDR correction.
dunn_pairwise(groups, ...)  Dunn's post-hoc test with Bonferroni or BH correction.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import rankdata, norm as _norm
from scipy.stats import mannwhitneyu


def sig(p):
    """Return significance asterisk string for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def fmt_p(p):
    """Format a p-value as a readable string, guarding against underflow to 0."""
    if p == 0.0:
        return 'p<1e-300'
    elif p < 1e-15:
        return f'p<{10**int(np.floor(np.log10(p))):.0e}'
    elif p < 0.001:
        return f'p={p:.2e}'
    else:
        return f'p={p:.3f}'


def rankbiserial(a, b):
    """
    Rank-biserial correlation (effect size for Mann-Whitney U test).

    Returns r in [-1, 1]; r > 0 means group a tends to exceed group b.
    """
    n1, n2 = len(a), len(b)
    u, _ = mannwhitneyu(a, b, alternative='two-sided')
    return (2 * u) / (n1 * n2) - 1


def bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like of float

    Returns
    -------
    ndarray of adjusted p-values in the same order as the input.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    adj = pvals[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def dunn_pairwise(groups_dict, correction='bonferroni'):
    """
    Dunn's post-hoc test with multiple-comparison correction.

    Parameters
    ----------
    groups_dict : dict  {label: 1D array-like}
        Groups to compare.
    correction : {'bonferroni', 'bh'}
        Multiple-comparison correction method.

    Returns
    -------
    pd.DataFrame  Symmetric matrix of adjusted p-values.
    """
    labels = list(groups_dict.keys())
    all_vals = np.concatenate(list(groups_dict.values()))
    all_ranks = rankdata(all_vals)
    N = len(all_vals)
    sizes = {k: len(v) for k, v in groups_dict.items()}
    _, counts = np.unique(all_vals, return_counts=True)
    tie_corr = np.sum(counts ** 3 - counts) / (12 * (N - 1)) if N > 1 else 0

    pos = 0
    mean_ranks = {}
    for k, v in groups_dict.items():
        mean_ranks[k] = all_ranks[pos:pos + len(v)].mean()
        pos += len(v)

    pairs = list(combinations(labels, 2))
    n_tests = len(pairs)
    raw_ps = []
    for a, b in pairs:
        se = np.sqrt(
            (N * (N + 1) / 12 - tie_corr) * (1 / sizes[a] + 1 / sizes[b])
        )
        z = (mean_ranks[a] - mean_ranks[b]) / se if se > 0 else 0
        raw_ps.append(2 * (1 - _norm.cdf(abs(z))))

    if correction == 'bh':
        adj_ps = bh_fdr(raw_ps)
    else:  # bonferroni
        adj_ps = [min(p * n_tests, 1.0) for p in raw_ps]

    pmat = pd.DataFrame(np.nan, index=labels, columns=labels)
    for (a, b), p_corr in zip(pairs, adj_ps):
        pmat.loc[a, b] = p_corr
        pmat.loc[b, a] = p_corr
    for lbl in labels:
        pmat.loc[lbl, lbl] = 1.0
    return pmat
