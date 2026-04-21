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
# Sources:
#   Tier 1 core (confirmed ARE binding):
#     Hieronymus et al. 2006 Cancer Cell 10:141 (PMID 17010675) — PMEPA1, EAF2/U19
#     Prescott et al. 1998 Genes Dev (NKX3-1); Lin et al. 1999 Cancer Res (TMPRSS2)
#     Febbo et al. 2005 JNCI (FKBP5); Riegman et al. 1989/1991 (KLK2/KLK3)
#   Tier 2 (mCRPC-validated targets):
#     Kawakami et al. 2006 Clin Cancer Res (FOLH1/PSMA)
#     Korkmaz et al. 2009 Clin Cancer Res (STEAP2)
#     Maher et al. 2009 Science 326:1258 (SLC45A3)
#   Tier 3 (expanded from GeneList_PC_WholeTranscriptome.xlsx AR/SR signatures;
#           retained only genes present in ≥3 independent core AR sigs
#           [Hieronymus 2006, MSigDB Hallmark AR, Kumar 2016, Spratt 2019-A]
#           and biologically validated as AR-activated targets):
#     ACSL3  — in all 4 core sigs; fatty acid CoA ligase, direct ARE target
#     ELL2   — in 3/4 core sigs; elongation factor for RNA Pol II, prostate AR target
#     ABCC4  — in 3/4 core sigs; AR-regulated efflux transporter (Dean 2003 Genome Biol)
#     MAF    — in 3/4 core sigs; c-Maf bZIP TF, AR-induced in luminal PCa
#     ZBTB10 — in 3/4 core sigs; zinc-finger AR target (Sp1/Sp3 repressor)
#     SGK1   — MSigDB Hallmark AR; serum/glucocorticoid kinase, AR-survival axis in mCRPC
#     CAMKK2 — MSigDB Hallmark AR; confirmed direct AR target (Massie 2011 Nature)
# HOXB13 removed: pioneer factor upstream of AR, not a downstream output
#   (Sahu et al. 2011 Cell 147:1368; Huang et al. 2014 Cell 155:1135)
# ALDH1A3 removed: NEPC/stem-cell marker inversely related to AR activity
#   (Bhatt et al. 2017 JCI Insight; Smith et al. 2022 Nat Commun)
# CENPN excluded: centromere protein — cell-cycle confound, not AR-specific
AR_TARGET_GENES = [
    # ── Tier 1: original 10 core genes ──
    'KLK3',    # PSA — most canonical AR target (Riegman 1991)
    'KLK2',    # hK2 — AR target, high precision (Riegman 1989)
    'FKBP5',   # direct AR target, steroid chaperone (Febbo 2005 JNCI)
    'NKX3-1',  # AR-regulated homeobox, luminal marker (Prescott 1998 Genes Dev)
    'TMPRSS2', # direct AR target; TMPRSS2-ERG fusion driver (Lin 1999 Cancer Res)
    'PMEPA1',  # direct ARE-driven target (Zhao 2003 J Biol Chem); Hieronymus 2006 core
    'EAF2',    # direct AR transcriptional target (Cao 2006 Cancer Res); Hieronymus 2006 core
    'FOLH1',   # PSMA — AR-regulated, theranostic target (Kawakami 2006 CCR)
    'STEAP2',  # androgen-regulated cell surface protein (Korkmaz 2009 CCR)
    'SLC45A3', # AR target, solute carrier (Maher 2009 Science)
    # ── Tier 3: expansion from GeneList_PC_WholeTranscriptome.xlsx ──
    'ACSL3',   # fatty acid CoA ligase; in all 4 core AR sigs
    'ELL2',    # RNA Pol II elongation factor; in 3/4 core AR sigs
    'ABCC4',   # AR-regulated efflux pump; in 3/4 core AR sigs
    'MAF',     # c-Maf bZIP TF; in 3/4 core AR sigs
    'ZBTB10',  # zinc-finger AR target; in 3/4 core AR sigs
    'SGK1',    # serum/glucocorticoid kinase; MSigDB Hallmark AR
    'CAMKK2',  # CaM kinase kinase 2; confirmed AR target (Massie 2011 Nature)
]
