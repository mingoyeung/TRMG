# Replication Package

# Offshore Wind Power Probabilistic Forecasting Using TRMG

---

## 1. Project Directory Structure

All input data, model scripts, results, and evaluation notebooks are **self-contained within this single folder**. There are no external path dependencies.

```
trmg-replication/                         ← Project root (set as working directory)
│
├── README.md
├── README_CN.md
├── requirements.txt
│
├── data/                                ← ⭐ All raw and processed input data (centralized)
│   ├── preprocessing.ipynb              ← Main preprocessing notebook (TVFEMD, R2CMSE, Figs 2–3, Table 3, Fig E.2)
│   ├── cahrt.eddx                       ← Whole process of Figure 3 and E.2
│   ├── tennet/
│   │   ├── time_series_15min_singleindex_filtered.csv   ← TenneT raw data
│   │   ├── time_series_15min_cleaned.csv                ← TenneT after preprocessing
│   │   ├── emdresults.csv                               ← TenneT dataset decomposed by EMD
│   │   ├── tvfemdresults_TenneT.csv                     ← TenneT dataset decomposed by TVFEMD
│   │   ├── vmd_decomposition_results.csv                ← TenneT dataset decomposed by VMD
│   │   └── wt_results.csv                               ← TenneT dataset decomposed by WT
│   └── fujian/
│       ├── detection.ipynb              ← Fujian outlier detection (Table E.1, Fig E.1)
│       ├── data_fujian.xlsx             ← Fujian raw data
│       ├── res_new.csv                  ← Fujian after preprocessing
│       ├── emdf.csv                     ← Fujian dataset decomposed by EMD
│       ├── tvfemd_fujian.csv            ← Fujian dataset decomposed by TVFEMD
│       ├── vmd_fujian.csv               ← Fujian dataset decomposed by VMD
│       └── wt_fujian.csv               ← Fujian dataset decomposed by WT
│
├── results/                             ← ⭐ Pre-computed outputs (start here for Option A)
│   ├── actual.xlsx                      ← Test set ground truth (2,621 points)
│   ├── trmg.xlsx                        ← TRMG prediction results
│   ├── trqg.xlsx                        ← TRQG prediction results
│   ├── wrgg.xlsx                        ← WRGG prediction results
│   ├── vrlg.xlsx                        ← VRLG prediction results
│   ├── ermg.xlsx                        ← ERMG prediction results
│   ├── tsmg.xlsx                        ← TSMG prediction results
│   ├── qrnn2pred.xlsx                   ← QRNN2G prediction results
│   ├── mcqrnn2pred.xlsx                 ← MCQRNNG prediction results
│   ├── predqrf.xlsx                     ← QRFG prediction results
│   ├── lassopred.xlsx                   ← QRLASSOG prediction results
│   ├── tfmg.xlsx                        ← TFMG prediction results (Ablation study)
│   ├── trmg1l.xlsx                      ← TRMG1L prediction results (Sensitivity analysis)
│   ├── trqg1.xlsx                       ← TRQ1G prediction results (Ablation study)
│   ├── constraint_score_calculation.xlsx ← Table 5 CS column (pre-computed in Excel)
│   ├── cs_plot.xlsx                     ← Fig 6 data extraction
│   ├── kde_restructured.xlsx            ← Fig 6 KDE input
│   ├── point_at_tau=0.5.xlsx            ← Fig 4 
│   ├── pointwise_metrics.xlsx            ← Table 6 indices
│   ├── evaluation_metrics_values.ipynb  ← Table 5 script
│   ├── plot_point.ipynb                 ← Fig 4 script
│   ├── plot_interval.ipynb              ← Fig 5 script
│   ├── kdeplot.ipynb                    ← Fig 6 script
│   ├── 5.2.ipynb                        ← Tables 6 & 7 script
│   └── section5.2/
│       ├── dm_test_pinball_loss_results.xlsx ← Table 7 results
│       └── statistical_tests_results.xlsx    ← Tables 6 & C.1 results
│
├── section4/                            ← Section 4: model training (TenneT)
│   ├── trmg/
│   │   ├── trmg.py
│   │   ├── imfreconstruction_TenneT.csv
│   │   ├── trmg1-5_results.json
│   │   └── trmg1-5.xlsx
│   ├── trqg/
│   │   ├── trqg.py
│   │   ├── imfreconstruction_TenneT.csv
│   │   ├── trqg1-5_results.json
│   │   └── trqg1-5.xlsx
│   ├── wrgg/
│   │   ├── wtgru.py
│   │   ├── wt_decomposition_results.csv
│   │   ├── wtgru1-5_results.json
│   │   └── wtgru1-5.xlsx
│   ├── vrlg/
│   │   ├── vmdlstm.py
│   │   ├── vmd_reconstruction.csv
│   │   ├── qrlstm1-5_results.json
│   │   └── vmdlstm1-5.xlsx
│   ├── ermg/
│   │   ├── ermg.py
│   │   ├── emdreconstruction.csv
│   │   ├── ermg1-5_results.json
│   │   └── ermg1-5.xlsx
│   ├── tsmg/
│   │   ├── tsmg.py
│   │   ├── sereconstruction.csv
│   │   ├── tsmg1-4_results.json
│   │   └── tsmg1-4.xlsx
│   ├── qrnn2/
│   │   ├── qr2nn.py
│   │   └── optimization_results.json
│   ├── mcqrnn/
│   │   ├── mcqrnn.py
│   │   └── optimization_results.json
│   ├── qrf/
│   │   ├── qrfprediction.ipynb
│   │   └── optuna_qrf_results.json
│   └── qrlasso/
│       ├── qrlasso.ipynb
│       └── optuna_lasso_results.json
│
├── section5.1/                         ← Fujian dataset experiments
│   ├── evaluation_metrics_values.ipynb ← Table F.1 script
│   ├── actualf.xlsx                    ← Fujian test set ground truth (1,142 points)
│   ├── constraint_score_fujian.xlsx    ← Table F.1 CS column (pre-computed in Excel)
│   ├── *f.xlsx files                   ← Fujian model prediction outputs
│   └──Folders named after each model name ← Fujian model prediction files (same structure as section4)
│
├── section5.4/                         ← Sensitivity analysis
│   ├── section5.4.1/                   ← thresh_bwr sensitivity (Table 9): output CSV files
│   ├── section5.4.2/                   ← tau sensitivity (Table 10): output CSV files
│   └── section5.4.3/                   ← MCQRNN hidden-layer sensitivity (Table 11, Sensitivity column)
│       ├── trmg1l.py
│       ├── trmg1_single1-5.xlsx        ← Per-run outputs (sum → trmg1l.xlsx in results/)
│       └── trmg1-5_single_hidden_results.json
│
└── section5.5/                         ← Ablation study (Table 11, Ablation column)
    ├── tfmg/
    │   ├── tfmg.py
    │   ├── fuzzy.csv
    │   ├── tfmg1-5_results.json
    │   └── tfmg1-5.xlsx
    └── trq1g/
        ├── trq1g.py
        ├── imfreconstruction_TenneT.csv
        ├── trq1g1-5_results.json
        └── trq1g1-5.xlsx
```

---

## 2. Computing Environment

| Component | Specification |
|-----------|--------------|
| OS        | Ubuntu 22.04 |
| GPU       | NVIDIA GeForce RTX 4090 |
| CUDA      | 12.2 |
| cuDNN     | 8.8 |
| Python    | 3.10 |

> ⚠️ **GPU is required only for model training (Options B and C).**
> All evaluation notebooks (Option A) run on CPU only.

---

## 3. Installation

```bash
# Install all required Python packages
pip install -r requirements.txt
```

Key dependencies:

| Library          | Version | Purpose               |
|------------------|---------|-----------------------|
| tensorflow       | 2.15.0  | Model training        |
| optuna           | 4.8.0   | Hyperparameter tuning |
| scikit-learn     | 1.8.0   | QRF, preprocessing    |
| pandas           | 2.2.3   | Data handling         |
| numpy            | 1.26.4  | Computation           |
| scipy            | 1.16.3  | Statistical tests     |
| CRPS             | 2.0.4   | CRPS metric           |
| KDEpy            | 1.1.12  | KDE visualization     |
| quantile-forest  | 1.4.1   | QRF model             |
| asgl             | 2.1.4   | LASSO model           |
| dieboldmariano   | 1.0.0   | DM test               |
| openpyxl         | 3.1.5   | Excel I/O             |
| matplotlib       | 3.7.2   | Plotting              |

---

## 4. Data

All data files are centralized under `data/` (raw/cleaned inputs) and `results/` (test-set ground truth and pre-computed predictions). No file is referenced from outside the project root.

| Dataset                   | Location                                                      | Rows    | Used In     |
|---------------------------|---------------------------------------------------------------|---------|-------------|
| TenneT raw                | `data/tennet/time_series_15min_singleindex_filtered.csv`      | ~26,000 | Section 4   |
| TenneT cleaned            | `data/tennet/time_series_15min_cleaned.csv`                   | ~26,000 | Section 4   |
| Fujian raw                | `data/fujian/data_fujian.xlsx`                                | ~11,000 | Section 5.1 |
| Fujian cleaned            | `data/fujian/res_new.csv`                                     | ~11,000 | Section 5.1 |
| TenneT test ground truth  | `results/actual.xlsx`                                         | 2,621   | Section 4   |
| Fujian test ground truth  | `section5.1/actualf.xlsx`                                     | 1,142   | Section 5.1 |

---

## 5. Step-by-Step Replication Guide

### Option A — Quick Verification (Recommended · CPU only · ~30–60 minutes total)

This option reproduces **all main-text tables and figures** from pre-computed prediction files stored in `results/`. No GPU or model retraining is required.

**Prerequisites:**
1. Install dependencies: `pip install -r requirements.txt`
2. Launch Jupyter from the project root: `jupyter notebook`
3. Ensure your working directory is set to the project root before running any notebook.

---

#### Step 1 — Reproduce Table 3 and Figures 2–3 (Data Analysis & Decomposition)

**Notebook:** `data/preprocessing.ipynb`

| Cell label (as annotated in notebook)        | Output                              | Expected runtime |
|----------------------------------------------|-------------------------------------|-----------------|
| *"Data cleaning" and "Seasonal test"*        | Table 3 (dataset statistics)      | ~1 min          |
| *"Visualization (Fig 2)"*                    | Fig 2 (dataset overview)            | ~1 min          |
| *"TVFEMD + R2CMSE"* (TenneT dataset section) | Fig 3(a) TVFEMD decomposition result, Fig 3(b)(c) R2CMSE reconstruction, Table 8 timing, Tables 9–10 sensitivity results | ~30–40 min |

> Due to different version of Basic Linear Algebra Subprograms, it may produce floating-point differences at the 1e-6 relative error level in TVFEMD decomposition. Pre-computed TVFEMD results were stored in data/tennet/tvfemdresults_TenneT.csv. You can run the codes in For differences in scipy.sparse.linalg.spsolve() cell and proceed with all subsequent analyses.

> **Note on Fig 3 composite layout:** The full Figure 3 panel was assembled using [EdrawMax](https://www.edrawmax.cn). The source file `Chart.eddx` is included in the replication package.

---

#### Step 2 — Reproduce Table 5 and Figures 4–6 (Main Evaluation Results)

| Notebook / File                                        | Cell or action                                                              | Output                           |
|--------------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------|
| `results/evaluation_metrics_values.ipynb`              | Run all cells sequentially; select each model's `.xlsx` from `results/`     | Table 5 (all metrics except CS)  |
| `results/constraint_score_calculation.xlsx`            | Open file; read **cell AN1** (highlighted in yellow) in each model's sheet  | Table 5 (CS score column)        |
| `results/plot_point.ipynb`                             | Run all cells                                                               | Fig 4 (point prediction)         |
| `results/plot_interval.ipynb`                          | Run all cells                                                               | Fig 5 (interval prediction)      |
| `results/kdeplot.ipynb`                                | Run all cells under *"KDE plot"* section                                    | Fig 6 (kernel density prediction)|

> **Note on QS (MW) score:** In this implementation, the Quantile Score (QS) is equivalent to pinball loss. The relevant cells in `results/evaluation_metrics_values.ipynb` are labelled *"Probability prediction: Pinball loss (QS in Table 5)"*.

> **Note on CS score:** No code execution is needed. The Constraint Score formula is pre-computed directly in `constraint_score_calculation.xlsx`. Open the file, navigate to each model's sheet, and read the value in cell AN1 (highlighted in yellow).

---

#### Step 3 — Reproduce Tables 6–7 and Table C.1 (Statistical Tests)

**Notebook:** `results/5.2.ipynb`

Run the notebook in two stages:

**Stage 1 — Extract pointwise metrics (prerequisite for Tables 6 and C.1):**

| Input | Cell label | Output |
|-------|-----------|--------|
| All `results/*.xlsx` + `results/actual.xlsx` | *"Wilcoxon indices extraction"* | `results/pointwise_metrics.xlsx` (RMSE, MAE, CRPS per model) |

**Stage 2 — Run statistical tests:**

| Cell label | Input | Output | Paper table |
|-----------|-------|--------|------------|
| *"Wilcoxon test"* | `results/section5.2/pointwise_metrics.xlsx` | Wilcoxon sheets in `results/section5.2/statistical_tests_results.xlsx` | Table 6 (Wilcoxon test) |
| *"Shapiro-Wilk test"* | `results/section5.2/pointwise_metrics.xlsx` | Shapiro sheets in `results/section5.2/statistical_tests_results.xlsx` | Table C.1 |
| *"DM test"* | All `results/*.xlsx` + `results/actual.xlsx` | `results/section5.2/dm_test_pinball_loss_results.xlsx` | Table 7 (DM test) |

---

#### Step 4 — Reproduce Tables 8–11 (Execution Time, Sensitivity, Ablation)

**Table 8 — Execution Time:**

| Component | How to locate the value                                                                                                                                                                     |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TVFEMD decomposition time | Run *"TVFEMD + R2CMSE"* cell in `data/preprocessing.ipynb` (TenneT dataset section). After execution, read the **`Total time`** line printed by `lp.print_stats()` for the TVFEMD function. |
| R2CMSE reconstruction time | Same cell as above. Read the **`Total time`** line printed by `lp.print_stats()` for the R2CMSE function.                                                                                   |
| Model training time (Optuna) | Open each model's `.json` file in `section4/`. Read the `optuna_duration_formatted/total_time_sec` field.                                                                                   |
| Model testing time | Open each model's `.json` file in `section4/`. Read the `test_duration_seconds/eval_time_sec` field.                                                                                                     |

> ⚠️ Wall-clock times vary across runs due to OS scheduling, CPU cache state, and instrumentation overhead. The values in the paper represent a single measured run and are **representative** rather than exact constants.

**Tables 9–10 — Sensitivity Analysis (TVFEMD and R2CMSE):**

| Output | Where to find                                                                                                                                                                        |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Table 9 (TVFEMD `thresh_bwr` sensitivity) | Result files in `section5.4/section5.4.1/`. Also reproduced by the *"# Table 9 Summary"* block printed during the *"TVFEMD + R2CMSE"* cell in `data/preprocessing.ipynb`.            |
| Table 10 (R2CMSE `tau` sensitivity) | Result files in `section5.4/section5.4.2/`. Also reproduced as `r2cmse_tau_sensitivity.csv` saved by the *"# Sensitivity results for R2CMSE Table 10 saved"* block in the same cell. |

**Table 11 — Sensitivity and Ablation Columns:**

| Column | File to load | Action                                                                                             |
|--------|-------------|----------------------------------------------------------------------------------------------------|
| Sensitivity (MCQRNN hidden layers) | `results/trmg1l.xlsx` | Uncomment the `trmg1l` loading line in `results/evaluation_metrics_values.ipynb` and run all cells. |
| Ablation (no R2CMSE / 1-hidden QRNN) | `results/tfmg.xlsx` and `results/trq1g.xlsx` | Uncomment the `tfmg` loading lines in `results/evaluation_metrics_values.ipynb` and run all cells. |

---

#### Step 5 — Reproduce Appendix E and Table F.1 (Fujian Dataset)

| Notebook / File | Cell label | Input | Output |
|----------------|-----------|-------|--------|
| `data/fujian/detection.ipynb` | *"Missing value detection"* | `data/fujian/data_fujian.xlsx` | Table E.1 (missing values) |
| `data/fujian/detection.ipynb` | *"Descriptive"* | `data/fujian/data_fujian.xlsx` | Table E.2 (descriptive statistics) |
| `data/fujian/detection.ipynb` | *"Heatmap for Fig E.1(a) in Outlier detection"* | `data/fujian/data_fujian.xlsx` | Fig E.1(a) |
| `data/fujian/detection.ipynb` | *"RANSAC fit in Fig E.1(b) in Outlier detection"* | `data/fujian/data_fujian.xlsx` | Fig E.1(b) + `data/fujian/res_new.csv` |
| `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"* (Fujian dataset section) | `data/fujian/res_new.csv` | Fig E.2(a) |
| `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"* (Fujian dataset section) | `section5.1/trmg/tvfemd_fujian.csv` | Fig E.2(b)(c) + `section5.1/trmg/tvfemdresults_fujian.csv` |
| `section5.1/evaluation_metrics_values.ipynb` | Run all cells | `section5.1/*f.xlsx` + `section5.1/actualf.xlsx` | Table F.1 (all metrics except CS) |
| `section5.1/constraint_score_fujian.xlsx` | Read **cell AN1** (highlighted in yellow) | Pre-computed | Table F.1 (CS score) |

> **Note on QS (MW) score in Table F.1:** Same as Table 5 — the QS metric corresponds to pinball loss. The relevant cells are labelled *"Probability prediction: Pinball loss (QS in Table F.1)"* in `section5.1/evaluation_metrics_values.ipynb`.

---

### Option B — Retrain with Saved Hyperparameters (GPU required · 1-2 hours)

This option retrains each model using the hyperparameters already optimised and saved in each model's `.json` file. Results may differ slightly from the paper due to GPU non-determinism (see Section 7).

```bash
# Example: retrain TRMG
# The script loads best hyperparameters from trmg1-5_results.json
cd section4/trmg
# Choose MODE = 'B' after if __name__ == "__main__":
# Change trmg1/2/3/4/5_results.json to produce trmg1.xlsx, trmg2.xlsx, ..., trmg5.xlsx on the terminal respectively
python trmg.py
```

After all 5 runs complete, sum the per-run output files to obtain the final prediction file:

```python
import pandas as pd

runs = [pd.read_excel(f"section4/trmg/trmg{i}.xlsx") for i in range(1, 6)]
final = sum(runs)
final.to_excel("results/trmg.xlsx", index=False)
```

Repeat this process for each model in `section4/`. Note:
- **TSMG** requires only **4 runs** (not 5).
- **QRNN2G, MCQRNNG**: single run each; load `optimization_results.json` automatically.
- **QRFG**: use `section4/qrf/qrfprediction.ipynb`; load `optuna_qrf_results.json` manually before running.
- **QRLASSOG**: use `section4/qrlasso/qrlasso.ipynb`; load `optuna_lasso_results.json` manually before running.

After retraining, proceed with Option A Steps 2–5 to evaluate.

---

### Option C — Full Replication from Scratch (GPU required · 2-7 hours per model)

> ⚠️ Same GPU reproducibility caveat as Option B applies (see Section 7).

```bash
# Step 1: Preprocess data and run TVFEMD + R2CMSE decomposition
# Open data/preprocessing.ipynb and run cells for the desired dataset (TenneT or Fujian).
# To avoid overwriting decomposition outputs, run only the cells corresponding
# to the model/method you intend to train next.
# Outputs: intermediate CSV files saved into each model's subfolder under section4/.

# Step 2: for .py files, Choose MODE = 'C' after if __name__ == "__main__": on the terminal (5 runs for decomposition-based results; see Option B for details)
#for .ipynb files (QRLASSOG and QRFG), Choose Option C cells to retrain. Taking TRMG as an example:
cd section4/trmg
python trmg.py   # repeat 5 times; obtain trmg1/2/3/4/5, the outputs are trmg1.xlsx ... trmg5.xlsx

# Step 3: Sum per-run outputs and copy the final file into results/ on the terminal
# (Use the summing code shown in Option B above)
cp section4/trmg/trmg.xlsx results/trmg.xlsx

# Step 4: Run all evaluation notebooks (same as Option A, Steps 2–5)
```

---

## 6. Code–Paper Mapping

### Section 4.1 — Data Analysis

| Paper Output                                    | Script | Cell label                                          | Input |
|-------------------------------------------------|--------|-----------------------------------------------------|-------|
| Table 3 (missing value, outlier, seasonal test) | `data/preprocessing.ipynb` | *"Data cleaning" and "Seasonal test"* | `data/tennet/time_series_15min_singleindex_filtered.csv` |
| Fig 2                                           | `data/preprocessing.ipynb` | *"Visualization (Fig 2)"*                           | same as above |

---

### Section 4.2 — Decomposition and Reconstruction (`data/preprocessing.ipynb`, TenneT dataset section)

> Signal decomposition is performed entirely in Python. The entropy-based IMF reconstruction step (selecting and summing IMF components by entropy range) is performed manually in Excel. The resulting CSV files are listed in the Data Output column below.

| Paper Output | Cell label (annotated in notebook)                             | Input | Data Output                                                                                                                    |
|-------------|----------------------------------------------------------------|-------|--------------------------------------------------------------------------------------------------------------------------------|
| Fig 3(a) TVFEMD | *"TVFEMD + R2CMSE"*: `# Figure 3(a): plot TVFEMD results`      | `data/tennet/time_series_15min_singleindex_filtered.csv` | `section4/trmg/tvfemdresults_TenneT.csv`                                                                                       |
| Fig 3(b)(c) R2CMSE | *"TVFEMD + R2CMSE"*: `# 4.2 R2CMSE: load TVFEMD results`       | `section4/trmg/tvfemdresults_TenneT.csv` | `section4/trmg/imfreconstruction_TenneT.csv`(TRMG,TRQG,TRMG1L,TRQ1G)                                                           |
| WRGG: Appendix D Table D.1 | *"WT + R2CMSE"*: `WT decomposition` and `R2CMSE results`   | `data/tennet/time_series_15min_singleindex_filtered.csv` | <br/>Decompositon:`data/tennet/wt_decomposition_results.csv`, <br/>Reconstruction:`section4/wrgg/wt_decomposition_results.csv` |
| VRLG: Appendix D Table D.2 | *"VMD + R2CMSE"*: `VMD decomposition` and `R2CMSE results` | `data/tennet/time_series_15min_singleindex_filtered.csv` | <br/>Decompositon:`data/tennet/vmd_decomposition_results.csv`, <br/>Reconstruction:`section4/vrlg/vmd_reconstruction.csv`      |
| ERMG: Appendix D Table D.3 | *"EMD + R2CMSE"*: `EMD decomposition` and `R2CMSE results` | `data/tennet/time_series_15min_singleindex_filtered.csv` | <br/>Decompositon:`data/tennet/emdresults.csv`, <br/>Reconstruction:`section4/ermg/emdreconstruction.csv`                      |
| TSMG: Appendix D Table D.4 | *"TVFEMD + SE"*: all cells                                     | `data/tennet/tvfemdresults_TenneT.csv` | <br/>Decompositon:`data/tennet/wt_decomposition_results.csv`,<br/>Reconstruction:`section4/tsmg/sereconstruction.csv`          |

---

### Section 4.3 — Model Training (Table 4)

> Pre-computed outputs already exist in `results/`. Run training only for full replication (Options B/C).

| Model | Script | Input CSV | Output (in `results/`) | Hyperparameter log |
|-------|--------|-----------|------------------------|-------------------|
| TRMG (proposed) | `section4/trmg/trmg.py` | `section4/trmg/imfreconstruction_TenneT.csv` | `trmg.xlsx` | `section4/trmg/trmg1-5_results.json` |
| TRQG | `section4/trqg/trqg.py` | `section4/trqg/imfreconstruction_TenneT.csv` | `trqg.xlsx` | `section4/trqg/trqg1-5_results.json` |
| WRGG | `section4/wrgg/wtgru.py` | `section4/wrgg/wt_decomposition_results.csv` | `wrgg.xlsx` | `section4/wrgg/wtgru1-5_results.json` |
| VRLG | `section4/vrlg/vmdlstm.py` | `section4/vrlg/vmd_reconstruction.csv` | `vrlg.xlsx` | `section4/vrlg/qrlstm1-5_results.json` |
| ERMG | `section4/ermg/ermg.py` | `section4/ermg/emdreconstruction.csv` | `ermg.xlsx` | `section4/ermg/ermg1-5_results.json` |
| TSMG | `section4/tsmg/tsmg.py` | `section4/tsmg/sereconstruction.csv` | `tsmg.xlsx` | `section4/tsmg/tsmg1-4_results.json` |
| QRNN2G | `section4/qrnn2/qr2nn.py` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `qrnn2pred.xlsx` | `section4/qrnn2/optimization_results.json` |
| MCQRNNG | `section4/mcqrnn/mcqrnn.py` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `mcqrnn2pred.xlsx` | `section4/mcqrnn/optimization_results.json` |
| QRFG | `section4/qrf/qrfprediction.ipynb` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `predqrf.xlsx` | `section4/qrf/optuna_qrf_results.json` |
| QRLASSOG | `section4/qrlasso/qrlasso.ipynb` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `lassopred.xlsx` | `section4/qrlasso/optuna_lasso_results.json` |

> .json files stored hyperparameter results, execution time, and training/validation/test loss.

> **Multi-run models:** Each decomposition-based model (TRMG, TRQG, WRGG, VRLG, ERMG, TSMG) is trained 5 times — once per reconstructed IMF component (e.g., `trmg1.xlsx` to `trmg5.xlsx`), except TSMG (4 runs). The final `results/trmg.xlsx` is the **sum** of these individual run outputs.

---

### Section 4.4 — Evaluation Results

| Paper Output | Script | Cell / Action | Input |
|-------------|--------|--------------|-------|
| Table 5 (all metrics except CS) | `results/evaluation_metrics_values.ipynb` | Run all cells; select each model `.xlsx` from `results/` | `results/*.xlsx` + `results/actual.xlsx` |
| Table 5 (CS score) | `results/constraint_score_calculation.xlsx` | Open file; read **cell AN1** (yellow highlight) per model sheet | Pre-computed in Excel |
| Table 5 (QS/MW score) | `results/evaluation_metrics_values.ipynb` | Cells labelled *"Probability prediction: Pinball loss (QS in Table 5)"* | `results/*.xlsx` + `results/actual.xlsx` |
| Fig 4 (point prediction) | `results/plot_point.ipynb` | Run all cells | `results/point_at_tau=0.5.xlsx` |
| Fig 5 (interval prediction) | `results/plot_interval.ipynb` | Run all cells | `results/*.xlsx` + `results/actual.xlsx` |
| Fig 6 (KDE density) | `results/kdeplot.ipynb` | *"KDE plot"* cells | `results/kde_restructured.xlsx` |

> How to obtain `results/kde_restructured.xlsx`: input `results/cs_plot.xlsx`, run `Data extraction` cells in `results/kdeplot.ipynb`.

---

### Section 5.1 — Fujian Dataset

> **Note**: Due to file corruption during package preparation, individual component prediction files for Section 5.1 are unavailable. However, the aggregated prediction outputs (section5.1/f.xlsx) are intact and sufficient to reproduce all reported metrics in Tables F.1 and Figure E. The aggregation was performed by summing component predictions.The aggregated prediction files were generated and saved prior to file corruption, and their integrity was verified by comparing the reported metrics in Table F.1 against those computed from the archived files, which match exactly.

| Output                                                                                                               | Script | Cell label                                                                | Input                                               |
|----------------------------------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------|-----------------------------------------------------|
| Table E.1 (missing values)                                                                                           | `data/fujian/detection.ipynb` | *"Missing value detection"*                                               | `data/fujian/data_fujian.xlsx`                      |
| Table E.2 (descriptive statistics)                                                                                   | `data/fujian/detection.ipynb` | *"Descriptive"*                                                           | `data/fujian/data_fujian.xlsx`                      |
| Fig E.1(a)                                                                                                           | `data/fujian/detection.ipynb` | *"Heatmap for Fig E.1(a) in Outlier detection"*                           | `data/fujian/data_fujian.xlsx`                      |
| Fig E.1(b)                                                                                                           | `data/fujian/detection.ipynb` | *"RANSAC fit in Fig E.1(b) in Outlier detection"*                         | `data/fujian/data_fujian.xlsx`                      |
| Fig E.2(a)                                                                                                           | `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"* (Fujian dataset section)                              | `data/fujian/res_new.csv`                           |
| Fig E.2(b)(c),<br/>Decompositon:`data/fujian/tvfemd_fujian.csv`,<br/>Reconstruction:`section5.1/trmg/tvfemdresults_fujian.csv` | `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"* (Fujian dataset section)                              | `section5.1/trmg/tvfemd_fujian.csv` (also for TRQG) |
| <br/>Decompositon:`data/fujian/vmd_fujian.csv`, <br/>Reconstruction:`section5.1/trmg/vmd_decomposition_fujian.csv`             | `data/preprocessing.ipynb` | *"VMD + R2CMSE"* (Fujian dataset section)                                 | `data/fujian/res_new.csv`                           |
| <br/>Decompositon:`data/fujian/wt_fujian.csv`,<br/>Reconstruction:`section5.1/trmg/wt_decomposition_fujian.csv`                | `data/preprocessing.ipynb` | *"WT + R2CMSE"* (Fujian dataset section)                                  | `data/fujian/res_new.csv`                           |
| <br/>Decompositon:`data/fujian/emdf.csv`,<br/>Reconstruction:`section5.1/ermg/emdfujian.csv`                                   | `data/preprocessing.ipynb` | *"EMD + R2CMSE"* (Fujian dataset section)                                 | `data/fujian/res_new.csv`                           |
| <br/>Decompositon:`data/fujian/tvfemd_fujian.csv`,<br/>Reconstruction:`section5.1/tsmg/tsmg_fujian.csv`                        | `data/preprocessing.ipynb` | *"TVFEMD + SE"* (Fujian dataset section)                                  | `data/fujian/res_new.csv`                           |
| Table F.1 (all metrics except CS)                                                                                    | `section5.1/evaluation_metrics_values.ipynb` | Run all cells                                                             | `section5.1/*f.xlsx` + `section5.1/actualf.xlsx`    |
| Table F.1 (CS score)                                                                                                 | `section5.1/constraint_score_fujian.xlsx` | Read **cell AN1** (yellow highlight) per model sheet                      | Pre-computed in Excel                               |
| Table F.1 (QS/MW score)                                                                                              | `section5.1/evaluation_metrics_values.ipynb` | Cells labelled *"Probability prediction: Pinball loss (QS in Table F.1)"* | `section5.1/*f.xlsx` + `section5.1/actualf.xlsx`    |

> Model training for the Fujian dataset follows the same structure as Section 4.3. Reconstruction results for hybrid baselines are shown at the bottom of `data/preprocessing.ipynb` (Fujian dataset section, markdown format).

---

### Section 5.2 — Statistical Tests

| Paper Output | Input | Cell label | Result file |
|-------------|-------|-----------|------------|
| Table C.1 (Shapiro-Wilk) | `results/pointwise_metrics.xlsx` | *"Wilcoxon test"* (Shapiro sub-step) | Shapiro sheets in `results/section5.2/statistical_tests_results.xlsx` |
| Table 6 (Wilcoxon test) | `results/pointwise_metrics.xlsx` | *"Wilcoxon test"* | Wilcoxon sheets in `results/section5.2/statistical_tests_results.xlsx` |
| Table 7 (DM test) | `results/*.xlsx` + `results/actual.xlsx` | *"DM test"* | `results/section5.2/dm_test_pinball_loss_results.xlsx` |

---

### Section 5.3 — Execution Time (Table 8)

| Component | How to locate the value                                                                                                                                    |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TVFEMD time | Run *"TVFEMD + R2CMSE"* cell in `data/preprocessing.ipynb` (TenneT section). Read **`Total time`** from `lp.print_stats()` output for the TVFEMD function. |
| R2CMSE time | Same cell. Read **`Total time`** from `lp.print_stats()` output for the R2CMSE function.                                                                   |
| Model Optuna training time | `optuna_duration_formatted/total_time_sec` field in each model's `.json` under `section4/`.                                                                |
| Model test time | `test_duration_seconds/eval_time_sec` field in each model's `.json` under `section4/`.                                                                     |


---

### Section 5.4 — Sensitivity Analysis

| Parameter | Paper output | Script / location | Output |
|-----------|-------------|------------------|--------|
| TVFEMD `thresh_bwr` | Table 9 | *"TVFEMD + R2CMSE"* cell in `data/preprocessing.ipynb` (TenneT section); see `# Table 9 Summary` block | CSV files in `section5.4/section5.4.1/` |
| R2CMSE `tau` | Table 10 | Same cell; see `# Sensitivity results for R2CMSE Table 10 saved` block | `r2cmse_tau_sensitivity.csv` in `section5.4/section5.4.2/` |
| MCQRNN hidden layers | Table 11 (Sensitivity column) | `section5.4/section5.4.3/trmg1l.py` | `trmg1_single1-5.xlsx`; sum → `results/trmg1l.xlsx`; log: `trmg1-5_single_hidden_results.json` |

---

### Section 5.5 — Ablation Study (Table 11, Ablation Column)

| Model | Script | Input | Output | Log |
|-------|--------|-------|--------|-----|
| TFMG (no R2CMSE) | `section5.5/tfmg/tfmg.py` | `section5.5/tfmg/fuzzy.csv` | `results/tfmg.xlsx` | `section5.5/tfmg/tfmg1-5_results.json` |
| TRQ1G (1-hidden QRNN) | `section5.5/trq1g/trq1g.py` | `section5.5/trq1g/imfreconstruction_TenneT.csv` | `results/trqg1.xlsx` | `section5.5/trq1g/trq1g1-5_results.json` |

> To reproduce Table 11 (Ablation column): uncomment the `tfmg` and `trq1g` loading lines in `results/evaluation_metrics_values.ipynb` and run all cells.

---

## 7. Reproducibility Statement

Option A is the designated and complete audit pathway. All prediction outputs archived in data/,results/, and section 5.1/ were generated in the original experimental environment and have been verified to reproduce all reported metrics in Tables 4–11 and Appendix Tables C.1–F.1 exactly.

On GPU-based retraining (Options B and C): The deep learning models use random_normal weight initialization compiled with @tf.function over 1,000 training epochs. Even with a fixed random seed (seed=42) and TF_DETERMINISTIC_OPS=1, bit-identical reproduction across different hardware or different time (same hardware) environments are not achievable due to two well-documented sources of non-determinism in GPU-accelerated deep learning:

CUDA/cuDNN version-dependent random number generation sequences, which affect weight initialization even under a fixed seed.
Non-associative floating-point accumulation in GPU-parallelized gradient computation, which produces diverging gradients across platforms after many training iterations.These are acknowledged limitations of TensorFlow on GPU hardware. See:
- [PyTorch reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html)
- [TensorFlow determinism docs](https://github.com/mm3509/reproducibility/blob/master/tensorflow-reproducibility.md)
- [Sources of Irreproducibility in Machine Learning](https://arxiv.org/pdf/2204.07610), Gundersen et al. (2022).

Options B and C are provided for methodological transparency only and are explicitly not recommended for audit purposes (longer time consumption).

---

## 8. External Dependencies

| Tool    | Role | Link |
|---------|------|------|
| TVFEMD  | Time-varying filtering EMD (Python) | https://github.com/stfbnc/pytvfemd |
| R2CMSE  | IMF reconstruction (MATLAB; Python conversion included in this package) | https://github.com/Shurun-Wang/R2CMSE |
| MCQRNN  | Monotone quantile regression neural network (Python) | https://github.com/RektPunk/mcqrnn |
| EdrawMax | Composite figure assembly (Fig 3, Fig E.2) | https://www.edrawmax.cn |

---
