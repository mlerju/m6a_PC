"""
m6a.data.loaders — Per-cohort data loaders.

All functions return:
    expr_df : DataFrame, shape (n_samples, n_genes)
        Log2-transformed expression values (log2(CPM+1) or log2(TPM+1)).
        Index = sample identifier strings.
    meta_df : DataFrame or None
        Sample-level metadata. None for cohorts without clinical annotations.

Loaded cohorts
--------------
load_mcrpc()        LuCaP-based mCRPC cohort (Adeno + SCNC, log2 CPM).
load_tcga()         TCGA-PRAD primary tumours (log2 CPM) with IGF2BP rescue.
load_gtex()         GTEx v11 normal prostate (log2 TPM).
load_adj_normal()   TCGA-PRAD adjacent solid-tissue normals (log2 CPM).
load_mcspc()        GSE221601 metastatic CSPC (Samsung MC, log2 TPM).

Utility
-------
build_common_universe(expr_dfs)
    Return the sorted list of genes present in ALL provided DataFrames.
"""
import gzip

import numpy as np
import pandas as pd

from m6a.config import (
    MCRPC_LOG2CPM, MCRPC_META_CSV,
    TCGA_LOG2CPM, TCGA_RAW_COUNTS, TCGA_GENEINFO, TCGA_CLINICAL,
    GTEX_FILE,
    TCGA_NORMAL_LOG2CPM,
    MHSPC_FILE,
    DARANA_FILE,
)
from m6a.genes import IGF2BP_ENSEMBL


def load_mcrpc():
    """
    Load the LuCaP-based mCRPC cohort.

    Returns
    -------
    expr_df : DataFrame (samples × genes), log2(CPM+1).
    meta_df : DataFrame with columns including 'histology' (Adenocarcinoma / SCNC)
              and 'Luminal/Basal Cluster'.
    """
    expr = pd.read_csv(MCRPC_LOG2CPM, sep='\t', index_col=0)   # samples × genes
    meta = pd.read_csv(MCRPC_META_CSV, index_col=1)
    print(f"    mCRPC: {expr.shape[0]} samples × {expr.shape[1]} genes")
    print(f"    Histology: {meta['histology'].value_counts(dropna=False).to_dict()}")
    return expr, meta


def load_tcga():
    """
    Load TCGA-PRAD primary tumour RNA-seq with IGF2BP1/2/3 rescue from raw
    counts (these genes are filtered from the processed log2CPM matrix due to
    near-zero expression in most samples).

    Returns
    -------
    expr_df : DataFrame (unique cases × genes), log2(CPM+1).
    clinical_df : DataFrame with clinical annotations
                  (columns include 'gleason_score').
    """
    gene_info = pd.read_csv(TCGA_GENEINFO, sep='\t', index_col=0)
    ensembl_to_symbol = gene_info['gene_name'].to_dict()

    raw_mat = pd.read_csv(TCGA_LOG2CPM, sep='\t', index_col=0)   # ENSEMBL × samples
    raw_mat.index = raw_mat.index.map(lambda x: ensembl_to_symbol.get(x, x))
    raw_mat = raw_mat[~raw_mat.index.duplicated(keep='first')]

    # Transpose; deduplicate cases (some have 01A + 01B aliquots)
    expr = raw_mat.T.copy()
    expr['_case'] = [s.split('::')[0] for s in expr.index]
    expr = expr.sort_index().drop_duplicates(subset='_case', keep='first')
    expr.index = pd.Index(expr['_case'].values, name='case_submitter_id')
    expr = expr.drop(columns=['_case'])
    print(f"    TCGA: {expr.shape[0]} unique cases × {expr.shape[1]} genes")

    # IGF2BP rescue from raw counts
    print("    Rescuing IGF2BP1/2/3 from TCGA raw counts ...")
    raw_counts = pd.read_csv(TCGA_RAW_COUNTS, sep='\t', index_col=0)
    lib_sizes = raw_counts.sum(axis=0)
    for gene, ens_id in IGF2BP_ENSEMBL.items():
        if ens_id in raw_counts.index:
            cpm = raw_counts.loc[ens_id] / lib_sizes * 1e6
            l2cpm = np.log2(cpm + 1).clip(lower=0)
            l2cpm.index = pd.Index(
                [s.split('::')[0] for s in l2cpm.index],
                name='case_submitter_id',
            )
            l2cpm = l2cpm[~l2cpm.index.duplicated(keep='first')]
            expr[gene] = l2cpm.reindex(expr.index).fillna(0.0)
            pct_above10 = (cpm >= 10).sum() / len(cpm) * 100
            print(f"      {gene}: median_log2CPM={l2cpm.median():.3f}, "
                  f"pct_CPM>=10={pct_above10:.1f}%")

    clinical = pd.read_csv(TCGA_CLINICAL).set_index('case_submitter_id')
    return expr, clinical


def load_gtex():
    """
    Load GTEx v11 normal prostate tissue (GCT format, TPM).

    Values are converted to log2(TPM+1).  Duplicated gene symbols are resolved
    by taking the row-wise mean (to be consistent with ssGSEA conventions).

    Returns
    -------
    expr_df : DataFrame (samples × genes), log2(TPM+1).
    None (GTEx has no per-sample clinical metadata relevant to this analysis).
    """
    with gzip.open(GTEX_FILE, 'rt') as fh:
        fh.readline()   # '#1.2' header
        fh.readline()   # dimension line
        raw = pd.read_csv(fh, sep='\t', index_col=0)   # ENSEMBL × (Description + samples)

    gtex_symbols = raw['Description']
    tpm = raw.drop(columns=['Description']).astype(float)
    tpm.index = gtex_symbols
    tpm = tpm.groupby(level=0).mean()   # mean across duplicate gene symbols

    expr = np.log2(tpm.T + 1)          # samples × genes, log2(TPM+1)
    expr.index.name = 'sample_id'
    print(f"    GTEx: {expr.shape[0]} samples × {expr.shape[1]} genes (log2 TPM+1)")
    return expr, None


def load_adj_normal():
    """
    Load TCGA-PRAD adjacent solid-tissue normal RNA-seq (52 samples, log2 CPM).

    Returns
    -------
    expr_df : DataFrame (samples × genes), log2(CPM+1).
    None (no separate clinical metadata used for this cohort).
    """
    raw = pd.read_csv(TCGA_NORMAL_LOG2CPM, sep='\t', index_col=0)  # genes × samples
    expr = raw.T.copy()                                              # samples × genes
    expr.index.name = 'case_submitter_id'
    print(f"    Adjacent Normal: {expr.shape[0]} samples × {expr.shape[1]} genes")
    return expr, None


def load_mcspc():
    """
    Load GSE221601 metastatic castration-sensitive prostate cancer (mCSPC)
    bulk RNA-seq from Samsung Medical Center (52 samples, log2 TPM from RSEM).

    Returns
    -------
    expr_df : DataFrame (samples × genes), log2(TPM+1).
    None (no clinical metadata used in current analyses).
    """
    raw = pd.read_csv(MHSPC_FILE, sep='\t', index_col=0)  # genes × samples
    expr = raw.T.copy()                                     # samples × genes
    expr.index.name = 'sample_id'
    print(f"    mCSPC (GSE221601): {expr.shape[0]} samples × {expr.shape[1]} genes (log2 TPM+1)")
    return expr, None


def load_mhspc_microarray(expr_file, meta_file=None):
    """
    Load mHSPC microarray data extracted by the collaborator via extract_mhspc_m6a.R.

    Expected input format (produced by the R script):
        expr_file : TSV or TSV.GZ, samples × ALL genes (full genome),
                    first column = 'sample_id'.
                    Values are log2 microarray intensities (RMA-normalized).
                    The full genome matrix is required so that within-sample
                    percentile ranks use the same ~15k-gene denominator as
                    the RNA-seq cohorts (platform-invariant after ranking).
        meta_file : TSV or CSV, first column = 'sample_id', clinical annotations.

    Returns
    -------
    expr_df : DataFrame (samples × genes), log2 intensities — full genome.
    meta_df : DataFrame or None.
    """
    expr = pd.read_csv(expr_file, sep='\t', index_col=0,
                       compression='infer')   # handles .gz automatically
    expr.index.name = 'sample_id'
    print(f"    mHSPC microarray: {expr.shape[0]} samples × {expr.shape[1]} genes")

    meta = None
    if meta_file is not None:
        sep = '\t' if str(meta_file).endswith('.tsv') else ','
        meta = pd.read_csv(meta_file, sep=sep, index_col=0)
        meta.index.name = 'sample_id'
        print(f"    mHSPC metadata:   {meta.shape[0]} samples × {meta.shape[1]} columns")

    return expr, meta


def load_darana():
    """
    Load DARANA (GSE197780) — neoadjuvant enzalutamide trial.

    Primary prostate cancer biopsied before and after ~3 months of enzalutamide
    monotherapy (Linder et al., Cancer Discov 2022; PMID 35754340).

    The GEO supplementary table has a single-row title (skipped), then a header
    row with ensembl_gene_id / gene_id / DAR01_pre / DAR01_post / ...

    Returns
    -------
    expr_df : DataFrame (samples × genes), log2-normalised counts.
              Index values follow the pattern "DAR{nn}_{pre|post}".
    meta_df : DataFrame with columns:
              'patient'   — e.g. 'DAR01'
              'timepoint' — 'pre' or 'post'
              'paired'    — True if this patient has both pre and post samples
    """
    df = pd.read_csv(DARANA_FILE, sep='\t', skiprows=1, low_memory=False)
    # Set gene symbol as index; drop ensembl column
    df = df.set_index('gene_id').drop(columns=['ensembl_gene_id'], errors='ignore')
    # Keep only DAR* sample columns; convert to float
    sample_cols = [c for c in df.columns if c.startswith('DAR')]
    df = df[sample_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # Transpose → samples × genes
    expr = df.T.copy()
    expr.index.name = 'sample_id'

    # Meta
    patients   = [s.rsplit('_', 1)[0] for s in expr.index]
    timepoints = [s.rsplit('_', 1)[1] for s in expr.index]
    pre_ids    = {p for p, t in zip(patients, timepoints) if t == 'pre'}
    post_ids   = {p for p, t in zip(patients, timepoints) if t == 'post'}
    paired_ids = pre_ids & post_ids
    meta = pd.DataFrame({
        'patient':   patients,
        'timepoint': timepoints,
        'paired':    [p in paired_ids for p in patients],
    }, index=expr.index)

    n_pre  = (meta['timepoint'] == 'pre').sum()
    n_post = (meta['timepoint'] == 'post').sum()
    n_pair = len(paired_ids)
    print(f"    DARANA (GSE197780): {len(expr)} samples "
          f"({n_pre} pre / {n_post} post, {n_pair} paired) × {expr.shape[1]} genes")
    return expr, meta


def build_common_universe(expr_dfs):
    """
    Return the sorted list of genes present in ALL provided expression DataFrames.

    This forms the rank denominator for within-sample percentile normalization.

    Parameters
    ----------
    expr_dfs : list of DataFrame (samples × genes)

    Returns
    -------
    list of str  Sorted gene symbols.
    """
    common = set(expr_dfs[0].columns)
    for df in expr_dfs[1:]:
        common &= set(df.columns)
    return sorted(common)
