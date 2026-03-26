"""
m6a.data — Data loading subpackage.

The only module here is ``loaders``, which provides one function per cohort.
Each function returns the expression matrix (samples × genes, log2-scale) and,
where available, a sample-level metadata DataFrame.

Adding a new dataset
--------------------
1. Write a ``load_<name>()`` function below following the same interface:
       expr_df : DataFrame, shape (n_samples, n_genes), log2-scale values
       meta_df : DataFrame or None
2. Import it in the entry-point script (cross_cohort.py or a new script).
3. Include it in the common-gene-universe intersection.
4. Add the group label and color to m6a/config.py (GROUP_LABELS / GROUP_COLORS).
"""
