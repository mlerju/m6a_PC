# m6A Epitranscriptomic Analysis in Prostate Cancer

Bulk RNA-seq analyses linking m6A RNA methylation to prostate cancer disease
progression, AR transcriptional activity, and the tumor immune microenvironment.

---

## Scientific overview

The project tests the hypothesis that m6A epitranscriptomic regulation is
systematically reprogrammed across the prostate cancer disease spectrum —
from normal prostate through to metastatic castration-resistant disease
(mCRPC) — and that this reprogramming is mechanistically coupled to AR
transcriptional activity, particularly via the RBM15/RBM15B writer paralog axis.

Six coordinated analyses are implemented:

| Script | Question | Output |
|--------|----------|--------|
| `mcrpc_analysis.py` | Three-axis m6A model in mCRPC; LR weight derivation | `plots_mcrpc/` |
| `cross_cohort.py` | m6A trajectory across 6 disease stages (Normal → mCRPC-SCNC) | `plots_cross_cohort/` |
| `ar_m6a_analysis.py` | AR × m6A coupling mechanism in mCRPC + TCGA validation | `plots_ar_m6a/` |
| `ar_crosscohort_analysis.py` | AR Activity Score trajectory across disease stages | `plots_ar_crosscohort/` |
| `ar_m6a_summary_figures.py` | Publication-ready AR × m6A summary panels | `plots_ar_m6a_summary/` |
| `tcga_immune_m6a.py` | m6A writers × CIBERSORT immune fractions; RBM15B × ARS (TCGA) | `plots_tcga_immune/` |

---

## Project structure

```
bulk_rnaseq/
├── m6a/                        # Shared library (import as `from m6a.xxx import ...`)
│   ├── __init__.py             # Package docstring and module index
│   ├── config.py               # Data paths and output directories
│   ├── genes.py                # m6A gene sets + AR target gene panel
│   ├── normalization.py        # Percentile-rank and z-score normalization
│   ├── scoring.py              # Three-axis score computation + LR weights
│   ├── stats.py                # BH-FDR, Dunn post-hoc, rank-biserial, sig/fmt_p
│   ├── plotting.py             # Violin-plot helpers
│   └── data/
│       ├── __init__.py         # Subpackage docs and contribution guide
│       └── loaders.py          # Per-cohort data loaders
│
├── cross_cohort.py
├── mcrpc_analysis.py
├── ar_m6a_analysis.py
├── ar_crosscohort_analysis.py
├── ar_m6a_summary_figures.py
├── tcga_immune_m6a.py
├── extract_mhspc_m6a.R         # R script for mHSPC microarray extraction
├── download_tcga_normals.py    # Utility: download TCGA adjacent normals
│
├── environment.yml             # Reproducible conda/micromamba environment
└── plots_*/                    # Generated output directories (one per script)
```

---

## Data requirements

Raw and processed data are **not included** in this repository (access-controlled
or large files).  All paths resolve relative to a configurable data root.

### Setting the data root

By default the code looks for data at `/mnt/biodata/data` (the lab NAS mount).
Override this with an environment variable — **no code changes needed**:

```bash
export M6A_DATA_ROOT=/path/to/your/data
```

### Required datasets

| Cohort | Relative path (from `M6A_DATA_ROOT`) | Access |
|--------|--------------------------------------|--------|
| LuCaP mCRPC cohort | `processed/mCRPC_cohort/` | Managed access (SU2C / WCDT) |
| TCGA-PRAD expression | `processed/tcga_prad/` | dbGaP phs000178 |
| GTEx v11 prostate | `raw/bulk_RNAseq/normalprost_GTEx/` | Open — gtexportal.org |
| TCGA adjacent normals | `processed/tcga_prad_normal/` | dbGaP phs000178 |
| mCSPC (GSE221601) | `processed/mhspc_gse221601/` | GEO open access |
| DARANA (GSE197780) | `processed/darana_gse197780/` | GEO open access |
| CIBERSORT fractions | `processed/tcga_prad/TCGA.Kallisto.fullIDs.cibersort.relative.tsv` | Open — Thorsson et al. 2018 |

> **mHSPC microarray** (Davicioni collaboration): `processed/mhspc_array/` —
> populate when files are received. Run `extract_mhspc_m6a.R` first to generate
> the full-genome expression matrix.

---

## Setup

```bash
# Create the environment
micromamba create -f environment.yml
micromamba activate rnaseq

# (Optional) set data root if different from /mnt/biodata/data
export M6A_DATA_ROOT=/your/data/path
```

---

## Running the analyses

Each script is self-contained and can be run independently.  The recommended
order below mirrors the narrative progression of the project:

```bash
# 1. mCRPC intra-cohort model (~5 min)
micromamba run -n rnaseq python mcrpc_analysis.py

# 2. Cross-cohort m6A trajectory (~3 min)
micromamba run -n rnaseq python cross_cohort.py

# 3. AR × m6A mechanism in mCRPC + TCGA validation (~10 min)
micromamba run -n rnaseq python ar_m6a_analysis.py

# 4. AR Activity Score cross-cohort trajectory (~5 min)
micromamba run -n rnaseq python ar_crosscohort_analysis.py

# 5. Publication summary figures (~5 min)
micromamba run -n rnaseq python ar_m6a_summary_figures.py

# 6. Immune landscape: m6A × CIBERSORT + RBM15B × ARS in TCGA (~3 min)
micromamba run -n rnaseq python tcga_immune_m6a.py
```

---

## The m6A scoring model

Three composite scores are computed from within-sample percentile-rank
normalized expression of 22 m6A regulatory genes:

| Score | Formula |
|-------|---------|
| **Net Deposition** | LR-weighted Writers − Erasers |
| **Oncogenic Readout** | Oncogenic Readers − Suppressive Readers |
| **Functional Impact** | 0.435 × Net Deposition + 0.565 × Oncogenic Readout |

LR writer weights were trained on the mCRPC cohort (59 SCNC vs 575 Adeno,
L2 logistic regression, 5-fold CV AUC = 0.793).  The Functional Impact axis
combination weights are Stage-2 LR values.  See `m6a/scoring.py` for details.

---

## Key references

- Thorsson et al. (2018) *Immunity* 48:812 — CIBERSORT pan-cancer immune landscape
- Barbie et al. (2009) *Nature* 462:108 — ssGSEA (percentile-rank normalization)
- Robinson et al. (2010) *Bioinformatics* — edgeR, log2-CPM normalization
- Massie et al. (2011) *Nature* 474:467 — CAMKK2 as direct AR target
- Linder et al. (2022) *Cancer Discov* — DARANA enzalutamide trial (GSE197780)

---

## Contact

Miguel Lermajuarez · mlermajuarez@gmail.com · YC Lab
