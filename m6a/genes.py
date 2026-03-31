"""
m6a.genes — Gene set definitions for cross-cohort and mCRPC analyses.

Two gene panels are defined:
  • Cross-cohort (22 genes): expanded set covering the full prostate cancer
    spectrum from GTEx normals to mCRPC.
  • mCRPC (18 genes): core set used in the intra-mCRPC logistic-regression
    analysis (model derivation and functional scoring).

The cross-cohort set adds METTL16 (writer), HNRNPA2B1 + ELAVL1 (oncogenic
readers), and HNRNPC + FMR1 (suppressive readers) relative to the mCRPC set.
"""

# ── Cross-cohort analysis gene set (22 genes) ────────────────────────────────
WRITER_GENES = [
    'METTL3', 'METTL14', 'WTAP', 'ZC3H13',
    'RBM15', 'RBM15B', 'CBLL1', 'METTL16',
]
ERASER_GENES = ['FTO', 'ALKBH5']
READER_ONCOGENIC = [
    'IGF2BP1', 'IGF2BP2', 'IGF2BP3',
    'YTHDF1', 'HNRNPA2B1', 'ELAVL1',
]
READER_SUPPRESSIVE = [
    'YTHDF2', 'YTHDF3', 'YTHDC1', 'YTHDC2', 'HNRNPC', 'FMR1',
]
ALL_M6A_GENES = WRITER_GENES + ERASER_GENES + READER_ONCOGENIC + READER_SUPPRESSIVE

# IGF2BP Ensembl IDs for raw-count rescue in TCGA (VERSIONED IDs as in GENCODE)
IGF2BP_ENSEMBL = {
    'IGF2BP1': 'ENSG00000159217.10',
    'IGF2BP2': 'ENSG00000073792.16',
    'IGF2BP3': 'ENSG00000136231.14',
}

# ── mCRPC analysis gene set (18 genes) ───────────────────────────────────────
MCRPC_WRITER_GENES = [
    'METTL3', 'METTL14', 'WTAP', 'ZC3H13', 'RBM15', 'RBM15B', 'CBLL1',
]
MCRPC_ERASER_GENES = ['FTO', 'ALKBH5']
MCRPC_READER_ONCOGENIC = ['IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'YTHDF1']
MCRPC_READER_SUPPRESSIVE = ['YTHDF2', 'YTHDF3', 'YTHDC1', 'YTHDC2']
MCRPC_ALL_GENES = (
    MCRPC_WRITER_GENES + MCRPC_ERASER_GENES +
    MCRPC_READER_ONCOGENIC + MCRPC_READER_SUPPRESSIVE
)

MCRPC_GENE_ROLES = {
    'METTL3':  'Writer (Catalytic)',
    'METTL14': 'Writer (Allosteric)',
    'WTAP':    'Writer (Scaffold)',
    'ZC3H13':  'Writer (Scaffold)',
    'RBM15':   'Writer (Targeting)',
    'RBM15B':  'Writer (Targeting)',
    'CBLL1':   'Writer (E3 Ligase)',
    'FTO':     'Eraser',
    'ALKBH5':  'Eraser',
    'IGF2BP1': 'Reader (Oncogenic)',
    'IGF2BP2': 'Reader (Oncogenic)',
    'IGF2BP3': 'Reader (Oncogenic)',
    'YTHDF1':  'Reader (Oncogenic)',
    'YTHDF2':  'Reader (Suppressive)',
    'YTHDF3':  'Reader (Suppressive)',
    'YTHDC1':  'Reader (Suppressive)',
    'YTHDC2':  'Reader (Suppressive)',
}

# Plot-ready ordered gene list and tick labels for mCRPC per-gene figures
MCRPC_GENE_ORDER = MCRPC_ALL_GENES
MCRPC_GENE_LABELS = (
    [f"{g} (W)"     for g in MCRPC_WRITER_GENES] +
    [f"{g} (E)"     for g in MCRPC_ERASER_GENES] +
    [f"{g} (R-onc)" for g in MCRPC_READER_ONCOGENIC] +
    [f"{g} (R-sup)" for g in MCRPC_READER_SUPPRESSIVE]
)

# Manual biology-based writer weights (used only for model-comparison in Part I
# of mcrpc_analysis.py)
MCRPC_MANUAL_WEIGHTS = {
    'METTL3': 3.0, 'METTL14': 2.0, 'WTAP': 1.5, 'ZC3H13': 1.0,
    'RBM15': 1.0, 'RBM15B': 0.8, 'CBLL1': 0.5,
}

# ── Androgen Receptor (AR) Activity Signature ────────────────────────────────
# Canonical AR transcriptional target genes, well-validated in prostate cancer.
# Used to compute a continuous AR Activity Score (ARS = mean z-score).
# AR itself is excluded to avoid circularity; it is analysed separately.
# Sources: Alumkal et al. 2020 (WCDT), Chen et al. 2018, Hallmark_Androgen_Response
AR_TARGET_GENES = [
    'KLK3',    # PSA — most canonical AR target
    'KLK2',    # hK2 — AR target, high precision
    'FKBP5',   # direct AR target, steroid hormone chaperone
    'NKX3-1',  # AR-regulated homeobox, luminal marker
    'TMPRSS2', # AR target; TMPRSS2-ERG fusion driver
    'FOLH1',   # PSMA — AR-regulated, theranostic target
    'STEAP2',  # AR target, cell surface
    'HOXB13',  # AR co-regulator / target, luminal marker
    'SLC45A3', # AR target, solute carrier
    'ALDH1A3', # AR-regulated, lipid metabolism
]
