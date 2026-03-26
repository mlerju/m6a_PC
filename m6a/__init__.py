"""
m6a — shared library for prostate cancer m6A epitranscriptomic analyses.

Modules
-------
config          Paths, output directories, group labels/colors.
genes           Gene set definitions for cross-cohort and mCRPC analyses.
stats           Statistical utilities (sig, fmt_p, rankbiserial, bh_fdr, dunn_pairwise).
plotting        Violin-plot helpers shared across analyses.
normalization   Within-sample percentile rank and z-score normalization.
scoring         Axis score computation (compute_axes, get_vals) and LR weight constants.
data.loaders    Per-cohort data loaders. Add new loaders here for new datasets.

Entry points (project root)
---------------------------
cross_cohort.py     Six-group cross-cohort analysis (29 analytical + 8 presentation figures).
mcrpc_analysis.py   mCRPC-specific analysis (25 figures, LR weight derivation in Part I).
"""
