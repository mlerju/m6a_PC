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

---

## Project structure

```
bulk_rnaseq/
│
├── m6a/                        # Shared Python library
│   ├── config.py               # Data paths and output directories
│   │                           # (set M6A_DATA_ROOT env var to override)
│   ├── genes.py                # m6A gene sets + AR target gene panel (with sources)
│   ├── normalization.py        # Percentile-rank and z-score normalization
│   ├── scoring.py              # Three-axis scoring + pre-computed LR weights
│   ├── stats.py                # BH-FDR, Dunn post-hoc, rank-biserial, sig/fmt_p
│   ├── plotting.py             # Violin-plot helpers
│   └── data/loaders.py         # Per-cohort data loaders
│
├── scripts/                    # Analysis scripts (run from project root)
│   ├── mcrpc_analysis.py       # mCRPC 3-axis model; LR weight derivation
│   ├── cross_cohort.py         # m6A trajectory: Normal → mCRPC-SCNC (6 groups)
│   ├── ar_m6a_analysis.py      # AR × m6A coordination: mCRPC + TCGA validation
│   ├── ar_crosscohort_analysis.py  # AR Activity Score trajectory across stages
│   ├── ar_m6a_summary_figures.py   # Publication-ready AR × m6A panels
│   ├── tcga_immune_m6a.py      # m6A × CIBERSORT immune + RBM15B × ARS (TCGA)
│   ├── extract_mhspc_m6a.R     # R: extract mHSPC microarray expression
│   └── download_tcga_normals.py    # Utility: download TCGA adjacent normals
│
├── results/
│   ├── figures/
│   │   ├── mcrpc/              # ← mcrpc_analysis.py output
│   │   ├── cross_cohort/       # ← cross_cohort.py output
│   │   ├── ar_m6a/             # ← ar_m6a_analysis.py output
│   │   ├── ar_crosscohort/     # ← ar_crosscohort_analysis.py output
│   │   ├── ar_summary/         # ← ar_m6a_summary_figures.py output
│   │   └── tcga_immune/        # ← tcga_immune_m6a.py output
│   ├── tables/                 # CSV outputs (correlation summaries, etc.)
│   └── logs/                   # Run logs (gitignored)
│
├── envs/
│   └── merip.yml               # Reproducible conda/micromamba environment
│
├── data/                       # Placeholder structure — actual data NOT committed
│   ├── raw/                    # (gitignored)
│   └── processed/              # (gitignored)
│
└── README.md
```

---

## Data requirements

Raw and processed data are **not included** in this repository (access-controlled
or large files). All paths resolve relative to a configurable data root.

### Setting the data root

By default the code looks for data at `/mnt/biodata/data` (the lab NAS).
Override without editing any code:

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

---

## Setup

```bash
micromamba create -f envs/merip.yml
micromamba activate rnaseq

# Optional: point to a different data root
export M6A_DATA_ROOT=/your/data/path
```

---

## Running the analyses

Each script is self-contained. Recommended order mirrors the narrative arc:

```bash
# 1. mCRPC intra-cohort model and LR weight derivation (~5 min)
micromamba run -n rnaseq python scripts/mcrpc_analysis.py

# 2. Cross-cohort m6A disease-progression trajectory (~3 min)
micromamba run -n rnaseq python scripts/cross_cohort.py

# 3. AR × m6A coordination mechanism in mCRPC + TCGA validation (~10 min)
micromamba run -n rnaseq python scripts/ar_m6a_analysis.py

# 4. AR Activity Score trajectory across disease stages (~5 min)
micromamba run -n rnaseq python scripts/ar_crosscohort_analysis.py

# 5. Publication summary figures (~5 min)
micromamba run -n rnaseq python scripts/ar_m6a_summary_figures.py

# 6. Immune landscape: m6A × CIBERSORT + RBM15B × ARS in TCGA (~3 min)
micromamba run -n rnaseq python scripts/tcga_immune_m6a.py
```

All output goes to `results/figures/<analysis>/` and `results/tables/`.

---

## The m6A scoring model

Three composite scores computed from within-sample percentile-rank normalized
expression of 22 m6A regulatory genes:

| Score | Formula |
|-------|---------|
| **Net Deposition** | LR-weighted Writers − Erasers |
| **Oncogenic Readout** | Oncogenic Readers − Suppressive Readers |
| **Functional Impact** | 0.435 × Net Deposition + 0.565 × Oncogenic Readout |

LR writer weights were trained on the mCRPC cohort (59 SCNC vs 575 Adeno,
L2 logistic regression, 5-fold CV AUC = 0.793). See `m6a/scoring.py` for
weight values and `scripts/mcrpc_analysis.py` Part I for the derivation.

---

## Key references

- Thorsson et al. (2018) *Immunity* 48:812 — CIBERSORT pan-cancer immune landscape
- Barbie et al. (2009) *Nature* 462:108 — ssGSEA (percentile-rank normalization)
- Massie et al. (2011) *Nature* 474:467 — CAMKK2 as direct AR target
- Linder et al. (2022) *Cancer Discov* — DARANA enzalutamide trial (GSE197780)

---

## Contact

Miguel Lermajuarez · mlermajuarez@gmail.com · YC Lab
