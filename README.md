# Replication Package

# Offshore Wind Power Probabilistic Forecasting Using TRMG

**[中文版说明 / Chinese Version](README_CN.md)**

---

## ⚡ For Auditors: Start Here

**The fastest way to verify results (no GPU needed):**

```bash
# Step 1: Install environment
pip install -r requirements.txt

# Step 2: Verify Table 5 and Figures 4-6
# Open and run: results/section 4.4/evaluation_metrics_values.ipynb

# Step 3: Verify Table 6-7 (Statistical Tests)  
# Open and run: 5.2 Statistical analysis/5.2.ipynb
```

> All pre-computed prediction files are in `results/`.
> You do **not** need to retrain any model to reproduce tables and figures.

---

## 1. Repository Structure
> All file paths in the notebooks are **relative paths** based on the repository root directory.
```
├── README.md
├── README_CN.md
├── requirements.txt
│
├── 4.1-4.3/                        ← Section 4: data+ preprocessing + model training
│     └── time_series_15min_singleindex_filtered.csv   ← TenneT raw
│     └── time_series_15min_cleaned.csv   ← TenneT processed
│     └── preprocessing.ipynb         ← Data cleaning, TVFEMD, R2CMSE
│   ├── trmg/                       ← Proposed model
│   ├── trqg/
│   ├── wrgg/
│   ├── vrlg/
│   ├── ermg/
│   ├── tsmg/
│   ├── qrnn2/
│   ├── mcqrnn/
│   ├── qrf/
│   └── qrlasso/
│
├──r2cmse_replication_matlab         ← R2CMSE replication by MATLAB
│   ├── .mat                ←Decomposition results for MATLAB data
│   ├── r2cmse_main.m       ←⭐ R2CMSE obtained after execution
├── results/                        ← ⭐ Pre-computed outputs (start here)
│   ├── actual.xlsx                 ← Test set ground truth (2621 points)
│   ├── trmg.xlsx
│   ├── trqg.xlsx
│   ├── wrgg.xlsx
│   ├── vrlg.xlsx
│   ├── ermg.xlsx
│   ├── tsmg.xlsx
│   ├── qrnn2pred.xlsx
│   ├── mcqrnn2pred.xlsx
│   ├── predqrf.xlsx
│   ├── lassopred.xlsx
│   └── section 4.4/                ← Evaluation scripts
│       ├── evaluation_metrics_values.ipynb
│       ├── constraint_score_calculation.xlsx
│       ├── point prediction/
│       ├── interval prediction/
│       └── probability density prediction/
│
├── 5.1 Model test in extreme wind conditions with multivariable/
├── 5.2 Statistical analysis/
├── 5.4 Sensitivity analysis/
└── 5.5 Ablation analysis/
```
---

## 2. Computing Environment

| Component | Specification |
|-----------|--------------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA | 12.2 |
| cuDNN | 8.8 |
| Python | 3.10 |

> ⚠️ **GPU is only required for model training (Options B and C).**
> Evaluation notebooks (Option A) run on CPU only and require no GPU.

---

## 3. Installation (requirement.txt in detail)

```bash
pip install -r requirements.txt
```

Key dependencies:

| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.15.0 | Model training |
| optuna | 4.8.0 | Hyperparameter tuning |
| scikit-learn | 1.8.0 | QRF, preprocessing |
| pandas | 2.2.3 | Data handling |
| numpy | 1.26.4 | Computation |
| scipy | 1.16.3 | Statistical tests |
| CRPS | 2.0.4 | CRPS metric |
| KDEpy | 1.1.12 | KDE visualization |
| quantile-forest | 1.4.1 | QRF model |
| asgl | 2.1.4 | LASSO model |
| dieboldmariano | 1.0.0 | DM test |
| openpyxl | 3.1.5 | Excel I/O |
| matplotlib | 3.7.2 | Plotting |

---

## 4. Data

| Dataset | Location | Size | Used In |
|---------|----------|------|---------|
| TenneT offshore wind | `4.1-4.3/time_series_15min_singleindex_filtered.csv` | ~26,000 rows | Section 4 |
| TenneT（after processing） | `4.1-4.3/time_series_15min_cleaned.csv` | ~26,000 rows | Section 4 |
| Fujian offshore wind | `5.1 .../data detection/data_fujian.xlsx` | ~11,000 rows | Section 5.1 |
| Fujian（after processing）| `5.1 .../data detection/res_new.csv` | ~11,000 rows | Section 5.1 |
| TenneT test set (actual) | `results/actual.xlsx` | 2621 rows | Evaluation |
| Fujian test set (actual) | `5.1 .../actualf.xlsx` | 1142 rows | Section 5.1 |

---

## 5. Code → Paper Mapping

### Section 4.1 — Data Analysis

| Output | Script | Input |
|--------|--------|-------|
| Table 3 (dataset analysis) | `4.1-4.3/preprocessing.ipynb` → Cell: *"Data clean (TenneT); Seasonal test"* | `4.1-4.3/time_series_15min_singleindex_filtered.csv` |
| Fig 2 (dataset overview) | `4.1-4.3/preprocessing.ipynb` → Cell: *"Visualization"* | same as above |

---

### Section 4.2 — Decomposition and Reconstruction

> ⭐ Updated packages may lead to differences in R2CMSE calculation. We present the Python calculation results in “5.1 R2CMSE results”, and have also uploaded the MATLAB implementation of R2CMSE. Reproduction can be achieved by running **r2cmse\_replication\_matlab/r2cmse_main.m**;
> Precomputed reconstruction CSV files have been saved in the subfolders of each model for direct use.`

| Output                        | Script | Cell                         | Input | Output File                                                                                                                                                                             |
|-------------------------------|--------|------------------------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fig 3(a) TVFEMD               | `4.1-4.3/preprocessing.ipynb` | *"4.1 TVFEMD decomposition"* | time_series_15min_singleindex_filtered.csv | `4.1-4.3/trmg/tvfemdresults_TenneT.csv`                                                                                                                                                 |
| Fig 3(b)(c) R2CMSE            | `4.1-4.3/preprocessing.ipynb` | *"5.1 R2CMSE results"*       | tvfemdresults_TenneT.csv | `4.1-4.3/trmg/imfreconstruction_TenneT.csv`,also listed in "5.1 R2CMSE results" cells                                                                                                   |
| Hybrid baselines (Appendix D) | `4.1-4.3/preprocessing.ipynb` | Cells 4.2-4.4, 5.1-5.2       | time_series_15min_singleindex_filtered.csv | ERMG: `4.1-4.3/trmg/imfreconstruction_TenneT.csv` TSMG:`4.1-4.3/tsmg/sereconstruction.csv` VRLG:`4.1-4.3/vrlg/vmd_reconstruction.csv` WRGG:`4.1-4.3/vrlg/wt_decomposition_results.csv`|
---

### Section 4.3 — Model Training (Table 4)

> ⭐ **Pre-computed outputs already exist in `results/`.**
> Run training only if you want full replication (requires GPU).

| Model | Script | Input CSV | Output (in `results/`) | Hyperparameters |
|-------|--------|-----------|------------------------|-----------------|
| TRMG (proposed) | `4.1-4.3/trmg/trmg.py` | `trmg/imfreconstruction_TenneT.csv` | `trmg.xlsx` | `trmg/trmg1-5_results.json` |
| TRQG | `4.1-4.3/trqg/trqg.py` | `trqg/imfreconstruction_TenneT.csv` | `trqg.xlsx` | `trqg/trqg1-5_results.json` |
| WRGG | `4.1-4.3/wrgg/wtgru.py` | `wrgg/wt_decomposition_results.csv` | `wrgg.xlsx` | `wrgg/wtgru1-5_results.json` |
| VRLG | `4.1-4.3/vrlg/vmdlstm.py` | `vrlg/vmd_reconstruction.csv` | `vrlg.xlsx` | `vrlg/qrlstm1-5_results.json` |
| ERMG | `4.1-4.3/ermg/ermg.py` | `ermg/emdreconstruction.csv` | `ermg.xlsx` | `ermg/ermg1-5_results.json` |
| TSMG | `4.1-4.3/tsmg/tsmg.py` | `tsmg/sereconstruction.csv` | `tsmg.xlsx` | `tsmg/tsmg1-4_results.json` |
| QRNN2G | `4.1-4.3/qrnn2/qr2nn.py` | time_series_15min_singleindex_filtered.csv | `qrnn2pred.xlsx` | `qrnn2/optimization_results.json` |
| MCQRNNG | `4.1-4.3/mcqrnn/mcqrnn.py` | time_series_15min_singleindex_filtered.csv | `mcqrnn2pred.xlsx` | `mcqrnn/optimization_results.json` |
| QRFG | `4.1-4.3/qrf/qrfprediction.ipynb` | time_series_15min_singleindex_filtered.csv | `predqrf.xlsx` | `qrf/optuna_qrf_results.json` |
| QRLASSOG | `4.1-4.3/qrlasso/qrlasso.ipynb` | time_series_15min_singleindex_filtered.csv | `lassopred.xlsx` | `qrlasso/optuna_lasso_results.json` |

> **Note on multi-run models (TRMG, TRQG, WRGG, VRLG, ERMG, TSMG, TRQG):**
> Each was run 5 times for 5 reconstructed IMFs (e.g., `trmg1.xlsx` to `trmg5.xlsx`), except for TSMG (only run 4 times).
> The final `results/trmg.xlsx` is the **sum** of these 5 (4) runs.
> Individual run files and JSON logs are kept in each model's subfolder.

---

### Section 4.4 — Evaluation Results

| Output | Script | Input |
|--------|--------|-------|
| Table 5 (all metrics) | `results/section 4.4/evaluation_metrics_values.ipynb` | `results/*.xlsx` + `results/actual.xlsx` |
| Table 5 (CS score) | `results/section 4.4/constraint_score_calculation.xlsx` | same |
| Fig 4 (point prediction) | `results/section 4.4/point prediction/plot_point.ipynb` | `point prediction/point_at_tau=0.5.xlsx` |
| Fig 5 (interval prediction) | `results/section 4.4/interval prediction/plot_interval.ipynb` | `results/*.xlsx` |
| Fig 6 (KDE density plots) | `results/section 4.4/probability density prediction/kdePLOT.ipynb` | `probability density prediction/kde_restructured.xlsx`,calculated by cs_plot.xlsx |

---

### Section 5.1 — Fujian Dataset

| Output                  | Script                                             | Input                   |
|-------------------------|----------------------------------------------------|-------------------------|
| Appendix E Fig E.1      | `5.1 .../data detection/datadetection.ipynb`       | `data_fujian.xlsx`      |
| Appendix E Table E.2    | describe() function for "pandas" package in python | same as above           |
| Table F.1 (all metrics) | `5.1 .../evaluation_metrics_values.ipynb`          | `5.1 .../*f.xlsx` files |

> Model training and test scripts for Fujian dataset follow the same structure
> as Section 4.3. Input files use the Fujian-specific CSVs.

---

### Section 5.2 — Statistical Tests

| Output | Script | Input | Results |
|--------|--------|-------|-------|
| Table 6 (Wilcoxon test) | `5.2 Statistical analysis/5.2.ipynb` | `5.2 .../pointwise_metrics.xlsx` | `5.2 Statistical analysis/statistical_tests_results.xlsx`|
| Table 7 (DM test) | `5.2 Statistical analysis/5.2.ipynb` | `results/*.xlsx` + `results/actual.xlsx` | `5.2 Statistical analysis/dm_test_pinball_loss_results.xlsx`|

---

### Section 5.3 — Execution Time (Table 8)

| Component | How to measure |
|-----------|---------------|
| TVFEMD, R2CMSE time | `4.1-4.3/preprocessing.ipynb` using `line_profiler` |
| Prediction phase time | See `optuna_duration_formatted/test_duration_seconds` field in each model's `.json` file |

---

### Section 5.4 — Sensitivity Analysis (Table 9–11)

| Parameter | Where to change | Script                                                                                                                                                                                                                                                                |
|-----------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TVFEMD `bsp_order` | Cell *"4.1 TVFEMD decomposition"* | `4.1-4.3/preprocessing.ipynb`                                                                                                                                                                                                                                         |
| R2CMSE `tau` | Cell *"5.1 R2CMSE results"* | `4.1-4.3/preprocessing.ipynb`                                                                                                                                                                                                                                         |
| MCQRNN hidden layers | See `5.4 Sensitivity analysis/trmg1l.py` | Input follows the same format as TRMG. Each subsequence is exported to `trmg1_single1-5.xlsx`, with the total sum stored in `trmg1l.xlsx`. Hyperparameter search results are saved in `trmg1-5_single_hidden_results.json` as described in Section 4.3. |

---

### Section 5.5 — Ablation Study (Table 11)

| Model | Script | Input | Output | JSON Logs |
|-------|--------|-------|--------|-----------|
| TFMG (no R2CMSE) | `5.5 Ablation analysis/tfmg/tfmg.py` | `5.5 .../tfmg/fuzzy.csv` | `tfmg/tfmg.xlsx` | `tfmg/tfmg1-5_results.json` |
| TRQ1G (1-hidden QRNN) | `5.5 Ablation analysis/trq1g/trq1g.py` | `5.5 .../trq1g/imfreconstruction_TenneT.csv` | `trq1g/trq1g.xlsx` | `trq1g/trq1g1-5_results.json` |

---

## 6. Replication Workflow

### Option A — Quick Verification (Recommended, CPU only, ~mins)

No model training needed. Uses pre-computed `results/` files.
```
Step 1: Run results/4.1-4.3/preprocessing.ipynb
→ Reproduces Section 4.1-4.2
Step 2: Run results/section 4.4/evaluation_metrics_values.ipynb
→ Reproduces Table 5
Step 3: Run 5.2 Statistical analysis/5.2.ipynb
→ Reproduces Table 6 (Wilcoxon) and Table 7 (DM test)
Step 4: Run results/section 4.4/point prediction/plot_point.ipynb
→ Reproduces Fig 4
Step 5: Run results/section 4.4/interval prediction/plot_interval.ipynb
→ Reproduces Fig 5
Step 6: Run results/section 4.4/probability density prediction/kdePLOT.ipynb
→ Reproduces Fig 6
```
---

### 🔁 Option B — Retrain with Saved Hyperparameters (GPU required)

Skips hyperparameter search by loading saved `.json` files.
Useful to verify the training pipeline without running Optuna.

```python
# Example: load best hyperparameters from JSON and retrain
import json, time
from your_model_script import train_model  # e.g., trmg.py

with open('trmg/trmg1_results.json', 'r') as f:
    saved = json.load(f)

best_params = saved['best_hyperparameters']
# best_params contains: n_hidden, n_hidden2, penalty

best_model, _ = train_model(
    x_train, y_train, taus,
    n_hidden=best_params['n_hidden'],
    n_hidden2=best_params['n_hidden2'],
    penalty=best_params['penalty'],
    epochs=1000,
    learning_rate=0.01,
    verbose=True
)
```

> ⚠️ **Note on GPU reproducibility**: Even with identical hardware,
> software versions, and random seeds, GPU-based training results
> may differ slightly across runs or over time. This is a known
> limitation of GPU non-determinism in deep learning frameworks
> (see [TensorFlow documentation on reproducibility](https://github.com/mm3509/reproducibility/blob/master/tensorflow-reproducibility.md)).
> The pre-computed files in `results/` are the exact outputs used in the paper.

---

### 🔬 Option C — Full Replication from Scratch (GPU required, ~hours)

> ⚠️ Same GPU reproducibility caveat as Option B applies.

1. Run `4.1-4.3/preprocessing.ipynb` — data cleaning, TVFEMD, R2CMSE
2. Run each model script in `4.1-4.3/` (5 runs per decomposition-based model)
3. Sum per-run outputs → place final file in `results/`
4. Run evaluation notebooks (same as Option A)

---

## 7. Reproducibility Statement

This study uses GPU-accelerated deep learning. Even with fixed random
seeds (seed=42), `TF_DETERMINISTIC_OPS=1`, and `TF_CUDNN_DETERMINISTIC=1`,
**exact numerical reproduction of training outputs cannot be guaranteed**
across different hardware configurations or over time, due to
non-determinism inherent in GPU-accelerated operations.

This is a known and documented limitation of the deep learning field,
not specific to this study. See:

- [PyTorch reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html)
- [TensorFlow determinism docs](https://github.com/mm3509/reproducibility/blob/master/tensorflow-reproducibility.md)

**The pre-computed prediction files in `results/` represent the exact
outputs used in the paper.** All tables and figures reported in the
manuscript are fully and deterministically reproducible from these
files via Option A, with no GPU required.

---

## 8. External Dependencies

| Tool | Role | Requirement | Link |
|------|------|-------------|------|
| TVFEMD | Time-varying filtering EMD | Python | https://github.com/stfbnc/pytvfemd |
| R2CMSE | IMF reconstruction | Originally MATLAB; converted to Python in this package | https://github.com/Shurun-Wang/R2CMSE |
| MCQRNN | Monotone QRNN architecture | Python | https://github.com/RektPunk/mcqrnn |

---

