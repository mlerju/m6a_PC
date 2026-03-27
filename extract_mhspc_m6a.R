#!/usr/bin/env Rscript
# =============================================================================
# extract_mhspc_m6a.R
#
# PURPOSE:
#   Extract and normalize m6A pathway gene expression from the mHSPC
#   microarray dataset for cross-cohort integration with mCRPC/TCGA/GTEx data.
#
# OUTPUT:
#   mhspc_m6a_expression.tsv  — normalized log2 expression, m6A genes only
#   mhspc_m6a_metadata.tsv   — clinical/metadata annotations per sample
#
# USAGE (from R console or terminal):
#   Rscript extract_mhspc_m6a.R
#
# REQUIRES: oligo OR affy (if CEL files), or just base R (if already processed)
#   install.packages(c("BiocManager"))
#   BiocManager::install(c("oligo", "limma", "hgu133plus2.db"))  # adjust for your array
#
# Please send back BOTH output files to miguel.lerma@med.lu.se
# =============================================================================

# ── CONFIGURATION: edit these paths ──────────────────────────────────────────

# If you have raw CEL files, set this to the directory containing them:
CEL_DIR <- "./CEL_files"          # <-- change to your CEL directory

# If you already have a processed/normalized expression matrix (samples × genes),
# set this path and set USE_PROCESSED <- TRUE:
USE_PROCESSED  <- FALSE
PROCESSED_FILE <- "./expression_matrix.txt"   # rows = probes/genes, cols = samples

# Path to clinical/metadata file (TSV or CSV):
METADATA_FILE  <- "./clinical_metadata.csv"   # <-- change to your metadata file

# Column in metadata that identifies samples (must match expression column names):
SAMPLE_ID_COL  <- "sample_id"    # <-- change if different

# Output directory:
OUT_DIR <- "./"

# =============================================================================

# ── m6A gene panel (22 genes) ─────────────────────────────────────────────────
M6A_GENES <- c(
  # Writers
  "METTL3", "METTL14", "WTAP", "ZC3H13", "RBM15", "RBM15B", "CBLL1", "METTL16",
  # Erasers
  "FTO", "ALKBH5",
  # Oncogenic readers
  "IGF2BP1", "IGF2BP2", "IGF2BP3", "YTHDF1", "HNRNPA2B1", "ELAVL1",
  # Suppressive readers
  "YTHDF2", "YTHDF3", "YTHDC1", "YTHDC2", "HNRNPC", "FMR1"
)

# Clinical metadata columns we need (add/rename as available in your dataset):
META_COLS_WANTED <- c(
  # Required for survival analysis:
  "os_months",       # overall survival time in months
  "os_event",        # 1 = died, 0 = censored
  "pfs_months",      # progression-free survival (if available)
  "pfs_event",
  # Grouping variables:
  "treatment_arm",   # e.g. ADT, ADT+docetaxel, ADT+abiraterone...
  "volume",          # high/low volume
  "gleason_sum",     # or "grade_group"
  "psa_baseline",
  "met_sites",       # metastatic sites if annotated
  # AR-related (if available):
  "ar_score",        # any AR activity score
  "decipher_score"
)

# =============================================================================
# STEP 1: Load or process expression data
# =============================================================================
cat("=== mHSPC m6A extraction script ===\n\n")

if (USE_PROCESSED) {
  cat("Loading pre-processed expression matrix:", PROCESSED_FILE, "\n")
  expr_raw <- read.delim(PROCESSED_FILE, row.names = 1, check.names = FALSE)
  cat("  Dimensions:", nrow(expr_raw), "probes/genes ×", ncol(expr_raw), "samples\n")
  expr_matrix <- as.matrix(expr_raw)

} else {
  cat("Loading raw CEL files from:", CEL_DIR, "\n")

  # Try oligo first (for Affymetrix Gene/Exon ST arrays), then affy
  if (!requireNamespace("oligo", quietly = TRUE)) {
    if (!requireNamespace("affy", quietly = TRUE)) {
      stop("Please install either 'oligo' or 'affy':\n",
           "  BiocManager::install('oligo')  # for ST arrays\n",
           "  BiocManager::install('affy')   # for 3' IVT arrays (HG-U133)\n")
    }
    use_pkg <- "affy"
  } else {
    use_pkg <- "oligo"
  }

  cat("  Using package:", use_pkg, "\n")
  cel_files <- list.files(CEL_DIR, pattern = "\\.CEL$", full.names = TRUE,
                          ignore.case = TRUE)
  cat("  Found", length(cel_files), "CEL files\n")
  if (length(cel_files) == 0) stop("No CEL files found in: ", CEL_DIR)

  if (use_pkg == "oligo") {
    library(oligo)
    raw_data   <- read.celfiles(cel_files)
    norm_data  <- rma(raw_data)
    expr_matrix <- exprs(norm_data)          # already log2
  } else {
    library(affy)
    raw_data   <- ReadAffy(filenames = cel_files)
    norm_data  <- rma(raw_data)
    expr_matrix <- exprs(norm_data)          # already log2
  }
  cat("  RMA normalization complete. Dimensions:", nrow(expr_matrix), "×",
      ncol(expr_matrix), "\n")
}

# =============================================================================
# STEP 2: Map probes → gene symbols
# =============================================================================
cat("\nMapping probes to gene symbols...\n")

row_names <- rownames(expr_matrix)

# Detect if rows are already gene symbols
is_symbol <- mean(row_names %in% M6A_GENES) > 0
if (is_symbol) {
  cat("  Rows appear to be gene symbols already — no mapping needed.\n")
  gene_expr <- expr_matrix[rownames(expr_matrix) %in% M6A_GENES, , drop = FALSE]
  gene_expr_df <- as.data.frame(gene_expr)
  gene_expr_df$gene <- rownames(gene_expr_df)

} else {
  # Try annotation packages in order of likelihood
  annotation_pkgs <- c("hgu133plus2.db", "hgu133a.db", "hgu133b.db",
                        "hugene10sttranscriptcluster.db",
                        "hugene20sttranscriptcluster.db",
                        "huex10sttranscriptcluster.db")
  anno_pkg <- NULL
  for (pkg in annotation_pkgs) {
    if (requireNamespace(pkg, quietly = TRUE)) {
      anno_pkg <- pkg
      break
    }
  }

  if (!is.null(anno_pkg)) {
    cat("  Using annotation package:", anno_pkg, "\n")
    library(anno_pkg, character.only = TRUE)
    db <- get(sub("\\.db$", "", anno_pkg))
    symbol_map <- AnnotationDbi::select(db,
                                        keys    = row_names,
                                        columns = c("PROBEID", "SYMBOL"),
                                        keytype = "PROBEID")
    symbol_map <- symbol_map[!is.na(symbol_map$SYMBOL), ]
    symbol_map <- symbol_map[symbol_map$SYMBOL %in% M6A_GENES, ]
    cat("  Found", nrow(symbol_map), "probe-gene mappings for", length(M6A_GENES), "target genes\n")

    # Average duplicate probes per gene (take mean of all probes for same gene)
    gene_expr_df <- data.frame(gene = character(0), stringsAsFactors = FALSE)
    found_genes  <- character(0)
    expr_list    <- list()
    for (gene in M6A_GENES) {
      probes <- symbol_map$PROBEID[symbol_map$SYMBOL == gene]
      if (length(probes) == 0) {
        cat("  WARNING: no probe for", gene, "\n")
        next
      }
      sub_mat   <- expr_matrix[rownames(expr_matrix) %in% probes, , drop = FALSE]
      gene_vals <- if (nrow(sub_mat) > 1) colMeans(sub_mat) else sub_mat[1, ]
      expr_list[[gene]] <- gene_vals
      found_genes <- c(found_genes, gene)
    }
    gene_expr_mat <- do.call(rbind, expr_list)
    rownames(gene_expr_mat) <- found_genes
    gene_expr_df <- as.data.frame(gene_expr_mat)
    gene_expr_df$gene <- rownames(gene_expr_df)

  } else {
    # Fallback: try grep on row names for partial gene name matches
    cat("  No annotation package found — attempting string matching on probe IDs\n")
    cat("  (Install BiocManager::install('hgu133plus2.db') for better mapping)\n")
    found <- grepl(paste(M6A_GENES, collapse = "|"), row_names)
    gene_expr_df <- as.data.frame(expr_matrix[found, ])
    gene_expr_df$gene <- row_names[found]
    cat("  Found", nrow(gene_expr_df), "rows by string match\n")
  }
}

cat("  Genes extracted:", paste(sort(unique(gene_expr_df$gene)), collapse = ", "), "\n")

# Reshape to samples × genes (wide format)
gene_expr_wide <- as.data.frame(t(gene_expr_df[, !colnames(gene_expr_df) %in% "gene"]))
colnames(gene_expr_wide) <- gene_expr_df$gene
gene_expr_wide$sample_id <- rownames(gene_expr_wide)
# Clean up sample IDs (strip .CEL suffix if present)
gene_expr_wide$sample_id <- sub("\\.CEL$", "", gene_expr_wide$sample_id,
                                 ignore.case = TRUE)

# =============================================================================
# STEP 3: Load metadata and merge
# =============================================================================
cat("\nLoading metadata:", METADATA_FILE, "\n")
if (file.exists(METADATA_FILE)) {
  meta <- read.csv(METADATA_FILE, stringsAsFactors = FALSE)
  cat("  Metadata:", nrow(meta), "rows ×", ncol(meta), "columns\n")
  cat("  Column names:", paste(colnames(meta), collapse = ", "), "\n")

  # Attempt merge
  if (SAMPLE_ID_COL %in% colnames(meta)) {
    merged <- merge(gene_expr_wide, meta, by.x = "sample_id",
                    by.y = SAMPLE_ID_COL, all.x = TRUE)
    cat("  Merged:", nrow(merged), "samples\n")
  } else {
    cat("  WARNING: column '", SAMPLE_ID_COL, "' not found in metadata.\n",
        "  Available columns:", paste(colnames(meta), collapse = ", "), "\n",
        "  Outputting expression WITHOUT metadata — please update SAMPLE_ID_COL.\n")
    merged <- gene_expr_wide
  }
} else {
  cat("  Metadata file not found — outputting expression only.\n")
  merged <- gene_expr_wide
}

# =============================================================================
# STEP 4: Write outputs
# =============================================================================
expr_out <- file.path(OUT_DIR, "mhspc_m6a_expression.tsv")
meta_out <- file.path(OUT_DIR, "mhspc_m6a_metadata.tsv")

# Expression output (samples × m6A genes only)
expr_cols  <- c("sample_id", intersect(M6A_GENES, colnames(merged)))
meta_cols  <- setdiff(colnames(merged), intersect(M6A_GENES, colnames(merged)))
meta_cols  <- meta_cols[meta_cols != "sample_id"]

expr_final <- merged[, expr_cols, drop = FALSE]
write.table(expr_final, file = expr_out, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\n→ Expression saved:", expr_out,
    "(", nrow(expr_final), "samples ×", ncol(expr_final) - 1, "genes )\n")

if (length(meta_cols) > 0) {
  meta_final <- merged[, c("sample_id", meta_cols), drop = FALSE]
  write.table(meta_final, file = meta_out, sep = "\t", row.names = FALSE, quote = FALSE)
  cat("→ Metadata saved: ", meta_out,
      "(", nrow(meta_final), "samples ×", length(meta_cols), "columns )\n")
}

# Summary report
cat("\n=== SUMMARY ===\n")
cat("Samples:", nrow(expr_final), "\n")
cat("m6A genes found (", length(expr_cols) - 1, "/", length(M6A_GENES), "):\n")
found_g   <- intersect(M6A_GENES, colnames(expr_final))
missing_g <- setdiff(M6A_GENES, colnames(expr_final))
cat("  Found:  ", paste(found_g,   collapse = ", "), "\n")
if (length(missing_g) > 0)
  cat("  Missing:", paste(missing_g, collapse = ", "), "\n")

cat("\nPlease send both output files to miguel.lerma@med.lu.se:\n")
cat(" 1. mhspc_m6a_expression.tsv\n")
cat(" 2. mhspc_m6a_metadata.tsv\n")
cat("=================\n")
