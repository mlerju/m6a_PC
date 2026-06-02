#!/usr/bin/env python3
"""
Download and process TCGA-PRAD Solid Tissue Normal RNA-seq samples from GDC.
Outputs:
  /mnt/biodata/data/processed/tcga_prad_normal/expression_raw_counts.tsv
  /mnt/biodata/data/processed/tcga_prad_normal/expression_log2cpm.tsv
  /mnt/biodata/data/processed/tcga_prad_normal/sample_metadata.tsv
"""
import os, json, tarfile, io, requests
import numpy as np
import pandas as pd

OUTDIR  = '/mnt/biodata/data/processed/tcga_prad_normal'
TMPDIR  = '/tmp/tcga_prad_normals'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

# -----------------------------------------------------------------------
# 1. Query GDC API for all 52 TCGA-PRAD Solid Tissue Normal STAR-Counts
# -----------------------------------------------------------------------
print("[1] Querying GDC API ...")
filters = {
    'op': 'and', 'content': [
        {'op': 'in', 'content': {'field': 'cases.project.project_id',       'value': ['TCGA-PRAD']}},
        {'op': 'in', 'content': {'field': 'cases.samples.sample_type',      'value': ['Solid Tissue Normal']}},
        {'op': 'in', 'content': {'field': 'data_type',                       'value': ['Gene Expression Quantification']}},
        {'op': 'in', 'content': {'field': 'experimental_strategy',           'value': ['RNA-Seq']}},
        {'op': 'in', 'content': {'field': 'analysis.workflow_type',          'value': ['STAR - Counts']}},
    ]
}
params = {
    'filters': json.dumps(filters),
    'fields':  'file_id,file_name,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type',
    'format':  'JSON',
    'size':    '200',
}
r = requests.get('https://api.gdc.cancer.gov/files', params=params, timeout=60)
r.raise_for_status()
hits = r.json()['data']['hits']
print(f"    Found {len(hits)} files")

file_ids   = [h['file_id'] for h in hits]
file_names = {h['file_id']: h['file_name'] for h in hits}
case_map   = {}
for h in hits:
    fid  = h['file_id']
    case = h.get('cases', [{}])[0].get('submitter_id', fid)
    samp = h.get('cases', [{}])[0].get('samples', [{}])[0].get('submitter_id', fid)
    case_map[fid] = {'case_id': case, 'sample_id': samp}

# -----------------------------------------------------------------------
# 2. Bulk download via GDC /data endpoint (returns tar.gz)
# -----------------------------------------------------------------------
tar_path = os.path.join(TMPDIR, 'tcga_prad_normals.tar.gz')
if os.path.exists(tar_path):
    print(f"[2] TAR already exists at {tar_path}, skipping download")
else:
    print(f"[2] Downloading {len(file_ids)} files from GDC (bulk) ...")
    payload  = {'ids': file_ids}
    response = requests.post(
        'https://api.gdc.cancer.gov/data',
        json=payload,
        headers={'Content-Type': 'application/json'},
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    total_bytes = 0
    with open(tar_path, 'wb') as fh:
        for chunk in response.iter_content(chunk_size=1024*1024):
            fh.write(chunk)
            total_bytes += len(chunk)
            if total_bytes % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"    Downloaded {total_bytes / 1e6:.0f} MB ...")
    print(f"    Download complete: {total_bytes / 1e6:.1f} MB")

# -----------------------------------------------------------------------
# 3. Extract and parse STAR counts files
# -----------------------------------------------------------------------
print("[3] Extracting and parsing STAR counts files ...")

# STAR augmented counts files have 4 columns:
#   gene_id  unstranded  stranded_first  stranded_second
# Plus 4 summary rows at the top starting with 'N_'
# We use 'unstranded' (col index 1) to match the TCGA tumor pipeline

counts_dict = {}   # sample_id -> Series(gene_id -> count)
gene_order  = None

with tarfile.open(tar_path, 'r:gz') as tar:
    members = tar.getmembers()
    print(f"    Files in tar: {len(members)}")
    # GDC tar structure: <file_id>/<filename>.tsv
    star_members = [m for m in members if 'augmented_star' in m.name and m.name.endswith('.tsv')]
    print(f"    Processing {len(star_members)} STAR counts files ...")

    for member in star_members:
        # Match by directory UUID (= file_id)
        parts = member.name.split('/')
        dir_uuid = parts[0] if len(parts) >= 2 else None
        if dir_uuid not in file_ids:
            continue

        sample_id = case_map[dir_uuid]['case_id']
        f = tar.extractfile(member)
        if f is None:
            continue
        content = f.read().decode('utf-8')
        lines   = content.strip().split('\n')

        # Format: # comment, then header, then N_ rows, then data
        # Columns: gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, tpm, fpkm, fpkm_uq
        data_lines = [l for l in lines if not l.startswith('#') and not l.startswith('N_') and not l.startswith('gene_id')]
        symbols = []
        counts  = []
        for l in data_lines:
            r = l.split('\t')
            if len(r) < 4:
                continue
            symbols.append(r[1])    # gene_name
            counts.append(int(r[3]) if r[3].isdigit() else 0)  # unstranded

        counts_dict[sample_id] = pd.Series(counts, index=symbols)
        if gene_order is None:
            gene_order = symbols

print(f"    Parsed {len(counts_dict)} samples")

# -----------------------------------------------------------------------
# 4. Build count matrix, compute log2CPM, map to gene symbols
# -----------------------------------------------------------------------
print("[4] Building count matrix and computing log2CPM ...")

count_mat = pd.DataFrame(counts_dict)   # genes × samples
print(f"    Raw counts matrix: {count_mat.shape}")

# Gene symbols are already the index; deduplicate (keep first occurrence)
count_mat = count_mat[~count_mat.index.duplicated(keep='first')]

# Compute CPM → log2(CPM+1)
lib_sizes = count_mat.sum(axis=0)
cpm_mat   = count_mat / lib_sizes * 1e6
log2cpm   = np.log2(cpm_mat + 1).clip(lower=0)

# Transpose: samples × genes
log2cpm_T = log2cpm.T
print(f"    log2CPM matrix: {log2cpm_T.shape} (samples × genes)")

# -----------------------------------------------------------------------
# 5. Save outputs
# -----------------------------------------------------------------------
print("[5] Saving outputs ...")

# Transpose back to genes × samples for consistency with tumor files
log2cpm.to_csv(os.path.join(OUTDIR, 'expression_log2cpm.tsv'), sep='\t')
count_mat.to_csv(os.path.join(OUTDIR, 'expression_raw_counts.tsv'), sep='\t')

# Metadata
meta_df = pd.DataFrame([
    {'case_id': v['case_id'], 'sample_id': v['sample_id'],
     'file_id': k, 'sample_type': 'Solid Tissue Normal', 'cohort': 'TCGA-PRAD'}
    for k, v in case_map.items()
    if v['case_id'] in log2cpm_T.index or v['case_id'] in counts_dict
])
meta_df.to_csv(os.path.join(OUTDIR, 'sample_metadata.tsv'), sep='\t', index=False)

print(f"\nDone.")
print(f"  {log2cpm_T.shape[0]} samples × {log2cpm_T.shape[1]} genes")
print(f"  Median lib size: {lib_sizes.median() / 1e6:.1f}M reads")
print(f"  Output: {OUTDIR}/")

# Quick QC: check m6A genes coverage
m6a_genes = ['METTL3','METTL14','WTAP','ZC3H13','RBM15','RBM15B','CBLL1',
             'FTO','ALKBH5','YTHDF1','YTHDF2','YTHDF3','YTHDC1','YTHDC2',
             'IGF2BP1','IGF2BP2','IGF2BP3']
found = [g for g in m6a_genes if g in log2cpm.index]
print(f"\n  m6A genes found: {len(found)}/{len(m6a_genes)}")
missing = [g for g in m6a_genes if g not in log2cpm.index]
if missing:
    print(f"  Missing: {missing}")
