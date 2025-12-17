# HPC-FLRNet

This repository provides the **full code and data pipeline** for predicting the **compressive strength of high‑performance concrete (HPC)** under **small‑sample tabular** conditions.

It includes:
- Construction of the benchmark dataset **HPC1826** (multi‑source merge → deduplication → robust screening)
- Training scripts for the proposed **HPC‑FLRNet** on **four datasets**
- Baseline models (**SVR / MLP / GBRT / XGBoost**) under the same split protocol
- **Repeated‑run evaluation (10 runs)** with **mean ± std** reporting
- **Non‑parametric statistical tests** (Friedman; paired Wilcoxon signed‑rank + Holm correction)
- **Ablation study** variants that share the **same split and run index** as the full model

> **Note (Nemenyi test):** If your SciPy build does not provide `studentized_range`, the script will skip Nemenyi and you can rely on the **Wilcoxon + Holm** results (CSV outputs are saved).

---

## Quick start (sanity check)

Run one training script and one significance test:

```bash
cd "src/code/hpc1826"
python "Train of HPC_FLRNet Model(hpc1826).py"

cd "../hpc1826/results"
python "stats_sigtest.py" \
  --metric rmse \
  --align_by index \
  --pattern "../../../data/results/significance/hpc1826/*_all_metrics.json"
```

---

## 1) Environment

Tested on: **Ubuntu 20.04.6 LTS**, **NVIDIA RTX 4090**, Python **3.9+**.

Minimal dependencies:

```bash
pip install -U numpy pandas scipy scikit-learn matplotlib openpyxl xgboost torch
```

(If using CUDA, install the PyTorch build matching your CUDA version from the official PyTorch website.)

---

## 2) Repository layout

```text
src/
  code/
    data103/                       # Yeh‑Slump (data103): HPC‑FLRNet + baselines + stats scripts
    data714/                       # Zhao–Nguyen (data714): HPC‑FLRNet + baselines + stats scripts
    data1133/                      # Yeh‑UCI (data1133): HPC‑FLRNet + baselines + stats scripts
    hpc1826/                       # HPC1826: HPC‑FLRNet + baselines + stats scripts
    hpc1826/ablation/              # Ablation variants + planned post‑hoc tests
    hpc1826 process pipeline/      # HPC1826 construction scripts + figure export
  data/
    intermediate/                  # tmp_data1950_merged.xlsx, tmp_data1925_dedup.xlsx, and source datasets
    processed/                     # final hpc1826.xlsx
    results/
      significance/                # JSON metrics + per‑dataset significance CSV outputs
      ablation/                    # ablation JSON metrics + planned post‑hoc CSV outputs
```

---

## 3) Data

Included under `src/data/`:
- `src/data/intermediate/tmp_data1950_merged.xlsx` – merged (aligned) intermediate dataset
- `src/data/intermediate/tmp_data1925_dedup.xlsx` – after deduplication
- `src/data/processed/hpc1826.xlsx` – final HPC1826 dataset
- `src/data/intermediate/data103.xlsx`, `data714.xlsx`, `data1133.xlsx` – the three public datasets used for comparison

### 3.1 Reproducing HPC1826 construction (1950 → 1925 → 1826)

Run the following scripts **in order** (paths contain spaces; use quotes):

```bash
cd "src/code/hpc1826 process pipeline"
python "01_merge_1950.py"
python "02_dedup_1925.py"
python "03_filter_1826_and_plot.py"
```

Outputs:
- Intermediate Excel files are written under `src/code/hpc1826 process pipeline/data/`
- Vector figures (used in the manuscript):
  - `FigX_a_curation.svg`
  - `HPC1826_with_source_pca_by_source.svg`

Expected accounting (as reported in the paper): **1950 → 1925 → 1826**.

---

## 4) Training & evaluation

All training scripts are **stand‑alone** Python files. They implement:
- a fixed 80/20 split via `train_test_split(..., random_state=...)`
- `n_runs=10` repeated runs aligned by **run index** (0–9), using `seed = base_seed + i`
- outputs:
  - a timestamped `*_all_metrics.json` (list of 10 items)
  - a timestamped `*.log`

The JSON schema is:

```json
[
  {"seed": 2436, "rmse": 3.99, "r2": 0.951, "r": 0.975, "mae": 2.83, "mape": 8.26},
  ...
]
```

### 4.1 HPC‑FLRNet (proposed model)

HPC1826:

```bash
cd "src/code/hpc1826"
python "Train of HPC_FLRNet Model(hpc1826).py"
```

Other datasets:

```bash
cd "src/code/data103"  && python "Train of HPC_FLRNet Model(data103).py"
cd "src/code/data714"  && python "Train of HPC_FLRNet Model(data714).py"
cd "src/code/data1133" && python "Train of HPC_FLRNet Model(data1133).py"
```

### 4.2 Baselines (SVR / MLP / GBRT / XGBoost)

Example (HPC1826):

```bash
cd "src/code/hpc1826"
python "Train of Contrastive Models(hpc1826).py"
```

Repeat similarly for `data103/`, `data714/`, and `data1133/`.

---

## 5) Mean ± std summaries

Each dataset folder includes `results/compute_summary.py` to summarize a selected JSON file:

```bash
cd "src/code/hpc1826"
python "results/compute_summary.py"
```

---

## 6) Statistical significance testing (per dataset)

Each dataset folder includes `results/stats_sigtest.py`:
- Friedman test across models
- Post‑hoc comparisons with **paired Wilcoxon signed‑rank test** + **Holm correction**
- (Optional) Nemenyi test if SciPy provides `studentized_range`

Example (HPC1826, RMSE):

```bash
cd "src/code/hpc1826/results"
python "stats_sigtest.py" \
  --metric rmse \
  --align_by index \
  --pattern "../../../data/results/significance/hpc1826/*_all_metrics.json"
```

Outputs (CSV files saved in the current folder):
- `significance_input_rmse.csv` (blocks × models)
- `significance_ranks_rmse.csv` (per‑block ranks)
- `significance_summary_rmse.csv` (mean/std)
- `posthoc_wilcoxon_holm_rmse.csv` (pairwise post‑hoc)

For **R²** (higher is better):

```bash
python "stats_sigtest.py" \
  --metric r2 \
  --higher_is_better \
  --align_by index \
  --pattern "../../../data/results/significance/hpc1826/*_all_metrics.json"
```

> **Recommendation:** Use `--align_by index` unless you are certain every model uses exactly the same 10 seeds. The paper workflow aligns repeated runs by **index** to ensure proper pairing.

---

## 7) Ablation study (HPC1826)

Ablation scripts and tests are under:

```text
src/code/hpc1826/ablation/
```

### 7.1 Run ablation variants

Run each variant script (generated from the full model code) to produce `*_all_metrics.json` files (10 runs each). The variants share the same split protocol and paired run index as the full model.

### 7.2 Significance testing for ablations

```bash
cd "src/code/hpc1826/ablation"
python "ablation_stats_sigtest.py" --metric rmse --align_by index
```

### 7.3 Planned post‑hoc (Full vs each, Holm m=5)

To match the manuscript’s **planned comparisons** (full model vs each ablation):

```bash
python "ablation_planned_posthoc.py"
```

The percentage degradation is computed as:

```text
(RMSE_variant − RMSE_full) / RMSE_full × 100%
```

---


## 8) Citation

A BibTeX entry will be added after the paper is accepted. Until then, please cite the manuscript and the baselines referenced in the paper.

