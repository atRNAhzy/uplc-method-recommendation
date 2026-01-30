# UPLC Method Recommendation (HTE Reaction Analysis)

A research codebase for the manuscript **“An Intelligent Data-Driven Framework for UPLC Method Recommendation in High-Throughput Reaction Analysis”** (manuscript in preparation).

This repository contains:

- **Method-aware retention time (RT) prediction** models for **7 UPLC methods (AM-I … AM-VII)**.
- A **reaction-level scoring** layer that converts predicted RTs of *all reaction components* into **UPLC method recommendations** under retention-range and peak-separation constraints.
- Reproducible notebooks/scripts for feature generation, data splitting, model training/evaluation, similarity analysis, and an API-style demo service.

---

## 1) Project overview (what this repo does)

### RT prediction (molecule-level, method-aware)

For each UPLC method (AM-I … AM-VII), a dedicated RT predictor is trained under a fixed chromatographic context (method-specific modeling). The repo includes:

- Raw RT datasets (`datas/0.data/*.csv`)
- Processed feature tables with descriptors + fingerprints (`datas/1.processed_results/*`)
- Train/test split tables for major datasets (`datas/2.train_test_split/*`)
- Traditional ML (SVR / LightGBM / XGBoost / RF) notebooks + trained artifacts (`ml/`)
- Optional deep learning baselines (GIN, ChemBERTa-style) (`gnn_bert/`)

### Method recommendation (reaction-level scoring)

Given a reaction system (≥2 components, ordered by importance, e.g., **P > substrates > by-products**), we:

1) predict each component’s RT under each UPLC method
2) score each method using:
   - **Retention range feasibility** (default [30,120] s; AM-III/AM-IV use [30,150] s)
   - **Pairwise separation threshold** (ΔRTmin = 10 s)
   - importance-weighted penalties + (optional) veto logic  

The output is a ranked list of candidate UPLC methods + a visualization comparing methods.

---

## 2) Repository structure

```
.
├── datas/
│   ├── 0.data/                 # raw RT datasets (AM-I ... AM-VII)
│   ├── 1.processed_results/    # descriptor + FG + MorganFP features
│   └── 2.train_test_split/     # train/test split with PCA/UMAP clusters (for large sets)
│
├── ml/
│   ├── 1.feature.ipynb         # feature generation (descriptors + FG bits + MorganFP)
│   ├── 2.clustering.ipynb      # PCA/UMAP clustering for split
│   ├── 3.data-split.ipynb      # cluster-aware train/test split (major datasets)
│   ├── 4.space-comparison...   # chemical space comparison
│   ├── 5.svr.ipynb / 5.lgb.ipynb / 5.xgb.ipynb / 5.RF.ipynb
│   ├── 5.lgb-TL.ipynb          # transfer learning exploration (tabular)
│   ├── 6.similarity-analysis.ipynb
│   ├── 7.Method-for-seperation.py  # FastAPI demo: predict + score + plot
│   ├── processed_results/      # (mirror) processed tables
│   ├── train_test_split/       # (mirror) split tables
│   ├── svr-models/             # trained SVR artifacts (AM-I/II/III)
│   ├── lgb-models/             # trained LightGBM artifacts
│   └── svr-model-other4/       # nested-CV SVR artifacts (AM-IV/V/VI/VII)
│
├── gnn_bert/
│   └── src/                    # GNN/BERT RT prediction (optional, requires torch-geometric)
│
└── results/
    └── GIN-AM-I / GIN-AM-II / GIN-AM-III  # example deep learning outputs
```

---

## 3) Data format

### Raw RT tables (`datas/0.data/AM-*.csv`)

Minimal schema:

- `SMILES`: canonical/standardized smiles
- `UV_RT-s`: retention time in seconds
- `Method`: one of `AM-I ... AM-VII`

### Processed feature tables (`datas/1.processed_results/*-filtered.csv`)

Includes:

- 5 physicochemical descriptors: `MolWt, logP, TPSA, H_bond_donors, H_bond_acceptors`
- 823 functional-group (SMARTS) bits: `col0 ... col822`
- 1024-bit Morgan fingerprint: `fp_0 ... fp_1023`

### Split tables (`datas/2.train_test_split/*_train.csv` / `*_test.csv`)

Same as processed tables, plus:

- `PCA_Cluster`, `UMAP_Cluster` (used for cluster-aware splitting)

---

## 4) Installation

> Recommended: **conda** for RDKit.

### 4.1 Minimal environment (traditional ML + scoring API)

```bash
conda create -n uplc-rec python=3.10 -y
conda activate uplc-rec

# RDKit
conda install -c conda-forge rdkit -y

# core python deps
pip install -U numpy pandas scikit-learn joblib matplotlib lightgbm xgboost chardet

# API demo
pip install -U fastapi "uvicorn[standard]" pydantic
```

### 4.2 Optional: deep learning (GNN/BERT)

`gnn_bert/src` uses **PyTorch**, **torch-geometric**, and **transformers**.
Install varies by CUDA/CPU. A typical CPU-only setup:

```bash
pip install -U torch torchvision torchaudio
pip install -U transformers loguru

# torch-geometric (choose the right wheel for your torch version)
# see official instructions: https://pytorch-geometric.readthedocs.io/
pip install -U torch-geometric
```

---

## 5) Quick start

### 5.1 Run the method recommendation demo API (FastAPI)

The API script is: `ml/7.Method-for-seperation.py`

```bash
cd ml
uvicorn "7.Method-for-seperation:app" --host 0.0.0.0 --port 8000
```

Test with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{
    "smiles_list": [
      "CC1=CC=C(C=C1)N", 
      "O=C(O)C1=CC=CC=C1",
      "O=C(NC1=CC=C(C=C1)N)C2=CC=CC=C2"
    ]
  }'
```

Expected response includes:

- recommended UPLC method (ranked)
- per-method scores
- a base64-encoded PNG plot comparing RTs/scores across methods

> Note: `smiles_list` should be ordered by importance (highest → lowest).  
> Example: `[Product, Substrate1, Substrate2, Byproduct...]`

### 5.2 Reproduce the classical ML pipeline (notebooks)

Open notebooks under `ml/` in order:

1) `1.feature.ipynb` → generate processed feature tables  
2) `2.clustering.ipynb` + `3.data-split.ipynb` → cluster-aware split  
3) `5.svr.ipynb`, `5.lgb.ipynb`, `5.xgb.ipynb`, `5.RF.ipynb` → model training/eval  
4) `6.similarity-analysis.ipynb` → similarity-stratified evaluation

---

## 6) Notes on scoring rules (defaults)

Current implementation uses:

- retention range: **[30,120] s** for AM-I/II/V/VI/VII; **[30,150] s** for AM-III/IV  
- minimum separation: **ΔRTmin = 10 s**

---

## 7) Manuscript & citation

**Manuscript:** *An Intelligent Data-Driven Framework for UPLC Method Recommendation in High-Throughput Reaction Analysis* (in preparation; not yet submitted).

If you use this repository in academic work, please cite the manuscript (details TBD).


---
