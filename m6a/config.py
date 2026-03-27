"""
m6a.config — Shared data paths, output directories, and group definitions.

Adding a new dataset
--------------------
1. Add its data path constant(s) here.
2. If it belongs to a new group in the cross-cohort analysis, append its label
   and hex color to GROUP_LABELS / GROUP_COLORS / GROUP_LABELS_SHORT.
3. Write a loader in m6a/data/loaders.py.
4. Call the loader and register the group in cross_cohort.py.
"""
import os

# ── Project root (parent of this file's directory) ──────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ───────────────────────────────────────────────────────────────
MCRPC_DIR       = '/mnt/biodata/data/processed/mCRPC_cohort'
TCGA_DIR        = '/mnt/biodata/data/processed/tcga_prad'
GTEX_FILE       = '/mnt/biodata/data/raw/normalprost_GTEx/gene_tpm_v11_prostate.gct.gz'
TCGA_NORMAL_DIR = '/mnt/biodata/data/processed/tcga_prad_normal'
MHSPC_FILE      = '/mnt/biodata/data/processed/mhspc_gse221601/expression_log2tpm.tsv'

# mHSPC microarray (Davicioni collaboration) — populate when files arrive:
MHSPC_ARRAY_EXPR = '/mnt/biodata/data/processed/mhspc_array/mhspc_expression_full.tsv.gz'
MHSPC_ARRAY_META = '/mnt/biodata/data/processed/mhspc_array/mhspc_m6a_metadata.tsv'

# Convenience sub-paths (used by loaders)
MCRPC_LOG2CPM       = os.path.join(MCRPC_DIR,       'LumBasal_mCRPC_RNAseq_CLEANED_LOG2CPM.tsv')
MCRPC_META_CSV      = os.path.join(MCRPC_DIR,       'LumBasal_mCRPC.csv')
TCGA_LOG2CPM        = os.path.join(TCGA_DIR,        'expression_log2cpm.tsv')
TCGA_RAW_COUNTS     = os.path.join(TCGA_DIR,        'expression_raw_counts.tsv')
TCGA_GENEINFO       = os.path.join(TCGA_DIR,        'gene_info.tsv')
TCGA_CLINICAL       = os.path.join(TCGA_DIR,        'clinical.csv')
TCGA_NORMAL_LOG2CPM = os.path.join(TCGA_NORMAL_DIR, 'expression_log2cpm.tsv')

# ── Output directories ────────────────────────────────────────────────────────
OUTDIR_CROSS_COHORT = os.path.join(_HERE, 'plots_cross_cohort')
OUTDIR_MCRPC        = os.path.join(_HERE, 'plots_mcrpc')
OUTDIR_PRESENTATION = os.path.join(_HERE, 'plots_presentation')
OUTDIR_AR_M6A       = os.path.join(_HERE, 'plots_ar_m6a')

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
