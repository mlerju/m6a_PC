"""
m6a — shared library for prostate cancer m6A epitranscriptomic analyses.

Modules
-------
config          Paths, output directories, cohort colors.
                Set M6A_DATA_ROOT env var to override the default data root.
genes           Gene set definitions: m6A writers/erasers/readers + AR targets.
normalization   Within-sample percentile rank and column-wise z-score normalization.
scoring         Three-axis m6A scoring (Net Deposition, Oncogenic Readout,
                Functional Impact) with pre-computed LR writer weights.
stats           Statistical utilities: sig, fmt_p, bh_fdr, rankbiserial, dunn_pairwise.
plotting        Violin-plot helpers (ngroup_violin, two_group_violin) shared across analyses.
data.loaders    Per-cohort data loaders. Add new loaders here for new datasets.

Analysis scripts (project root)
--------------------------------
cross_cohort.py             Six-group disease-progression m6A trajectory.
mcrpc_analysis.py           mCRPC intra-cohort 3-axis model + LR weight derivation.
ar_m6a_analysis.py          AR × m6A coordination: mCRPC mechanism + TCGA validation.
ar_crosscohort_analysis.py  AR Activity Score trajectory across disease stages.
ar_m6a_summary_figures.py   Publication-ready AR × m6A summary figures.
tcga_immune_m6a.py          m6A × CIBERSORT immune fractions + RBM15B × ARS (TCGA).
"""
