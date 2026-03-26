"""
m6a.normalization — Expression normalization utilities.

Functions
---------
percentile_rank_matrix(df)
    Within-sample transcriptome-wide percentile rank normalization (0–100).
    Used by the cross-cohort analysis (batch-invariant, platform-independent).
zscore_normalize(df)
    Column-wise z-score normalization.
    Used by the intra-mCRPC analysis.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def percentile_rank_matrix(df_samples_genes):
    """
    Within-sample percentile rank normalization.

    For each sample (row) and each gene (column), ranks the gene's expression
    among all genes measured in that sample and returns it as a percentile
    in [0, 100].  Ties are broken by average rank.

    This normalization is:
      • Batch-immune — arithmetic is within a single sample.
      • Platform-independent — log2(CPM+1) and log2(TPM+1) produce the same
        ranks because log2 is a monotone transform.
      • Conceptually equivalent to the rank step of ssGSEA (Barbie et al. 2009).

    Parameters
    ----------
    df_samples_genes : DataFrame, shape (n_samples, n_genes)
        Expression values (e.g. log2(CPM+1) or log2(TPM+1)).

    Returns
    -------
    DataFrame, same shape, values in [0, 100].
    """
    n_genes = df_samples_genes.shape[1]
    return df_samples_genes.apply(
        lambda row: pd.Series(
            rankdata(row, method='average') / n_genes * 100,
            index=row.index,
        ),
        axis=1,
    )


def zscore_normalize(df):
    """
    Column-wise z-score normalization (mean=0, std=1 per gene).

    Parameters
    ----------
    df : DataFrame, shape (n_samples, n_genes)

    Returns
    -------
    DataFrame, same shape.
    """
    return df.apply(lambda col: (col - col.mean()) / col.std())
