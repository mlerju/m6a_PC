"""
m6a.config — Shared data paths, output directories, and group definitions.

Data root
---------
All raw/processed data files are expected under a common base directory.
By default this is /mnt/biodata/data (the lab NAS mount).  To use a
different location without editing this file, set the environment variable::

    export M6A_DATA_ROOT=/your/data/path

before running any analysis script.

Adding a new dataset
--------------------
1. Add its data path constant(s) here (using _DATA_ROOT below).
2. If it belongs to a new group in the cross-cohort analysis, append its label
   and hex color to GROUP_LABELS / GROUP_COLORS / GROUP_LABELS_SHORT.
3. Write a loader in m6a/data/loaders.py.
4. Call the loader and register the group in cross_cohort.py.
"""
import os

# ── Project root (parent of this file's directory) ──────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data root — override with M6A_DATA_ROOT env var if needed ────────────────
_DATA_ROOT = os.environ.get('M6A_DATA_ROOT', '/mnt/biodata/data')

# ── Data paths ───────────────────────────────────────────────────────────────
MCRPC_DIR       = os.path.join(_DATA_ROOT, 'processed/mCRPC_cohort')
TCGA_DIR        = os.path.join(_DATA_ROOT, 'processed/tcga_prad')
GTEX_FILE       = os.path.join(_DATA_ROOT, 'raw/bulk_RNAseq/normalprost_GTEx/gene_tpm_v11_prostate.gct.gz')
TCGA_NORMAL_DIR = os.path.join(_DATA_ROOT, 'processed/tcga_prad_normal')
MHSPC_FILE      = os.path.join(_DATA_ROOT, 'processed/mhspc_gse221601/expression_log2tpm.tsv')
DARANA_FILE     = os.path.join(_DATA_ROOT, 'processed/darana_gse197780/GSE197780_DARANA_GE_table.txt.gz')

# mHSPC microarray (Davicioni collaboration) — populate when files arrive:
MHSPC_ARRAY_EXPR = os.path.join(_DATA_ROOT, 'processed/mhspc_array/mhspc_expression_full.tsv.gz')
MHSPC_ARRAY_META = os.path.join(_DATA_ROOT, 'processed/mhspc_array/mhspc_m6a_metadata.tsv')

# Convenience sub-paths (used by loaders)
MCRPC_LOG2CPM       = os.path.join(MCRPC_DIR,       'LumBasal_mCRPC_RNAseq_CLEANED_LOG2CPM.tsv')
MCRPC_META_CSV      = os.path.join(MCRPC_DIR,       'LumBasal_mCRPC.csv')
TCGA_LOG2CPM        = os.path.join(TCGA_DIR,        'expression_log2cpm.tsv')
TCGA_RAW_COUNTS     = os.path.join(TCGA_DIR,        'expression_raw_counts.tsv')
TCGA_GENEINFO       = os.path.join(TCGA_DIR,        'gene_info.tsv')
TCGA_CLINICAL       = os.path.join(TCGA_DIR,        'clinical.csv')
TCGA_NORMAL_LOG2CPM = os.path.join(TCGA_NORMAL_DIR, 'expression_log2cpm.tsv')

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR           = os.path.join(_HERE, 'results')
OUTDIR_MCRPC          = os.path.join(RESULTS_DIR, 'figures', 'mcrpc')
OUTDIR_CROSS_COHORT   = os.path.join(RESULTS_DIR, 'figures', 'cross_cohort')
OUTDIR_AR_M6A         = os.path.join(RESULTS_DIR, 'figures', 'ar_m6a')
OUTDIR_AR_CROSSCOHORT = os.path.join(RESULTS_DIR, 'figures', 'ar_crosscohort')
OUTDIR_AR_SUMMARY     = os.path.join(RESULTS_DIR, 'figures', 'ar_summary')
OUTDIR_TCGA_IMMUNE    = os.path.join(RESULTS_DIR, 'figures', 'tcga_immune')
OUTDIR_TABLES         = os.path.join(RESULTS_DIR, 'tables')

# ── Cross-cohort group definitions (6 cohorts, in progression order) ─────────
GROUP_LABELS = [
    'Normal Prostate\n(GTEx)',
    'Adjacent Normal\n(TCGA)',
    'Primary PCa\n(TCGA)',
    'mCSPC\n(GSE221601)',
    'mCRPC-Adeno',
    'mCRPC-SCNC',
]
GROUP_COLORS = ['#27ae60', '#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#c0392b']
GROUP_LABELS_SHORT = ['Normal', 'Adj Norm', 'Primary', 'mCSPC', 'mCRPC-Adeno', 'mCRPC-SCNC']

# ── mCRPC biopsy-site definitions ────────────────────────────────────────────
MCRPC_SITE_ORDER = ['Primary Site', 'Lymph Node', 'Bone', 'Lung', 'Liver', 'Other']
MCRPC_SITE_COLORS = {
    'Primary Site': '#3498db',
    'Lymph Node':   '#2ecc71',
    'Bone':         '#e67e22',
    'Lung':         '#9b59b6',
    'Liver':        '#e74c3c',
    'Other':        '#95a5a6',
}
