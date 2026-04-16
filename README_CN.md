# 复现说明文档
# 基于TRMG的海上风电概率预测

**[English Version / 英文版](README.md)**

---

## ⚡ 从这里开始

**最快验证结果的方式（无需GPU，约10分钟）：**

```bash
# 第一步：安装环境
pip install -r requirements.txt
```

然后打开Jupyter，按顺序运行：

1. `results/section 4.4/evaluation_metrics_values.ipynb` → 复现表5
2. `5.2 Statistical analysis/5.2.ipynb` → 复现表6、表7
3. `results/section 4.4/point prediction/plot_point.ipynb` → 复现图4
4. `results/section 4.4/interval prediction/plot_interval.ipynb` → 复现图5
5. `results/section 4.4/probability density prediction/kdePLOT.ipynb` → 复现图6

> `results/` 文件夹中已包含所有预计算预测结果。  
> **无需重新训练任何模型**即可复现论文中的所有表格和图片。

---

## 1. 仓库结构
> 所有notebook中的路径均为**相对路径**，基于仓库根目录。
```
├── README.md                            ← 英文说明
├── README_CN.md                         ← 中文说明（本文件）
├── requirements.txt                     ← 依赖库列表
│
├── 4.1-4.3/                             ← 第4节：数据处理+模型训练
│   ├── time_series_15min_singleindex_filtered.csv   ← TenneT原始数据
│   ├── time_series_15min_cleaned.csv                ← TenneT处理后数据
│   ├── preprocessing.ipynb              ← 数据清洗、TVFEMD、R2CMSE
│   ├── trmg/                            ← 本文提出模型（TRMG）
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
├──r2cmse_replication_matlab         ← R2CMSE结果复现，用MATLAB
│   ├── .mat                ←分解结果的mat数据形式
│   ├── r2cmse_main.m       ←⭐运行得到R2CMSE值
├── results/                             ← ⭐ 预计算结果（从这里开始）
│   ├── actual.xlsx                      ← 测试集真实值（2621个点）
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
│   └── section 4.4/                     ← 评估脚本
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

## 2. 计算环境

| 组件 | 规格 |
|------|------|
| 操作系统 | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA | 12.2 |
| cuDNN | 8.8 |
| Python | 3.10 |

> ⚠️ **GPU仅在模型训练时需要（方案B和C）。**  
> 评估脚本（方案A）仅需CPU，无需GPU。

---

## 3. 安装

```bash
pip install -r requirements.txt
```

主要依赖库：

| 库名 | 版本 | 用途 |
|------|------|------|
| tensorflow | 2.15.0 | 模型训练 |
| optuna | 4.8.0 | 超参数调优 |
| scikit-learn | 1.8.0 | QRF、数据预处理 |
| pandas | 2.2.3 | 数据处理 |
| numpy | 1.26.4 | 数值计算 |
| scipy | 1.16.3 | 统计检验 |
| CRPS | 2.0.4 | CRPS指标计算 |
| KDEpy | 1.1.12 | KDE可视化 |
| quantile-forest | 1.4.1 | QRF模型 |
| asgl | 2.1.4 | LASSO模型 |
| dieboldmariano | 1.0.0 | DM检验 |
| openpyxl | 3.1.5 | Excel读写 |
| matplotlib | 3.7.2 | 绘图 |

完整依赖列表见 `requirements.txt`。

---

## 4. 数据说明

| 数据集 | 路径 | 数据量 | 使用章节 |
|--------|------|--------|---------|
| TenneT海上风电（原始） | `4.1-4.3/time_series_15min_singleindex_filtered.csv` | ~26,000行 | 第4节 |
| TenneT海上风电（处理后） | `4.1-4.3/time_series_15min_cleaned.csv` | ~26,000行 | 第4节 |
| 福建海上风电 | `5.1 .../data detection/data_fujian.xlsx` | ~11,000行 | 第5.1节 |
| 福建海上风电(处理后) | `5.1 .../data detection/res_new.csv` | ~11,000行 | 第5.1节 |
| TenneT测试集真实值 | `results/actual.xlsx` | 2,621行 | 评估 |
| 福建测试集真实值 | `5.1 .../actualf.xlsx` | 1,142行 | 第5.1节 |

---

## 5. 代码与论文对应关系

### 第4.1节 — 数据分析

| 论文输出 | 脚本 | 对应Cell | 输入文件 |
|---------|------|---------|---------|
| 表3（数据集分析） | `4.1-4.3/preprocessing.ipynb` | *"Data clean (TenneT); Seasonal test"* | `time_series_15min_singleindex_filtered.csv` |
| 图2（数据集概览） | `4.1-4.3/preprocessing.ipynb` | *"Visualization"* | 同上 |

---

### 第4.2节 — 分解与重构

> ⚠️ python依赖包的更新，导致R2CMSE的重构结果可能存在细微差异。我们在5.1 R2CMSE results部分列出了python计算结果，同时上传了R2CMSE的MATLAB版本，运行r2cmse_replication_matlab中的r2cmse_main.m可以复现。
> 各模型子文件夹中已保存预计算的重构CSV文件供直接使用。

| 论文输出                     | 脚本 | Cell                         | 输入 | 输出文件 |
|--------------------------|------|------------------------------|------|---------|
| 图3(a) TVFEMD结果           | `4.1-4.3/preprocessing.ipynb` | *"4.1 TVFEMD decomposition"* | `time_series_15min_singleindex_filtered.csv` | `4.1-4.3/trmg/tvfemdresults_TenneT.csv` |
| 图3(b)(c) R2CMSE结果        | `4.1-4.3/preprocessing.ipynb` | *"5.1 R2CMSE results"*       | `tvfemdresults_TenneT.csv` | `4.1-4.3/trmg/imfreconstruction_TenneT.csv`，计算结果也存在"5.1 R2CMSE results"的运行单元格 |
| 组合基准模型分解与重构 (Appendix D) | `4.1-4.3/preprocessing.ipynb` | 代码单元格对应 4.2-4.4, 5.1-5.2的部分  | time_series_15min_singleindex_filtered.csv | ERMG: `4.1-4.3/trmg/imfreconstruction_TenneT.csv` TSMG:`4.1-4.3/tsmg/sereconstruction.csv` VRLG:`4.1-4.3/vrlg/vmd_reconstruction.csv` WRGG:`4.1-4.3/vrlg/wt_decomposition_results.csv`|

---

### 第4.3节 — 模型训练（表4）

> ⭐ `results/` 中已提供所有预计算结果。  
> 仅在需要完整复现时才需运行训练代码（需要GPU）。

| 模型 | 脚本 | 输入CSV | 输出（在`results/`中） | 超参数记录 |
|------|------|--------|----------------------|-----------|
| TRMG（本文提出） | `4.1-4.3/trmg/trmg.py` | `trmg/imfreconstruction_TenneT.csv` | `trmg.xlsx` | `trmg/trmg1-5_results.json` |
| TRQG | `4.1-4.3/trqg/trqg.py` | `trqg/imfreconstruction_TenneT.csv` | `trqg.xlsx` | `trqg/trqg1-5_results.json` |
| WRGG | `4.1-4.3/wrgg/wtgru.py` | `wrgg/wt_decomposition_results.csv` | `wrgg.xlsx` | `wrgg/wtgru1-5_results.json` |
| VRLG | `4.1-4.3/vrlg/vmdlstm.py` | `vrlg/vmd_reconstruction.csv` | `vrlg.xlsx` | `vrlg/qrlstm1-5_results.json` |
| ERMG | `4.1-4.3/ermg/ermg.py` | `ermg/emdreconstruction.csv` | `ermg.xlsx` | `ermg/ermg1-5_results.json` |
| TSMG | `4.1-4.3/tsmg/tsmg.py` | `tsmg/sereconstruction.csv` | `tsmg.xlsx` | `tsmg/tsmg1-4_results.json` |
| QRNN2G | `4.1-4.3/qrnn2/qr2nn.py` | `time_series_15min_cleaned.csv` | `qrnn2pred.xlsx` | `qrnn2/optimization_results.json` |
| MCQRNNG | `4.1-4.3/mcqrnn/mcqrnn.py` | `time_series_15min_cleaned.csv` | `mcqrnn2pred.xlsx` | `mcqrnn/optimization_results.json` |
| QRFG | `4.1-4.3/qrf/qrfprediction.ipynb` | `time_series_15min_cleaned.csv` | `predqrf.xlsx` | `qrf/optuna_qrf_results.json` |
| QRLASSOG | `4.1-4.3/qrlasso/qrlasso.ipynb` | `time_series_15min_cleaned.csv` | `lassopred.xlsx` | `qrlasso/optuna_lasso_results.json` |

> **关于多次运行的说明：**  
> TRMG、TRQG、WRGG、VRLG、ERMG、TRQG各运行5次（对应5个IMF分量，  
> 如`trmg1.xlsx`至`trmg5.xlsx`）；TSMG运行4次。  
> `results/`中的最终文件（如`trmg.xlsx`）为各次结果的**求和**。  
> 各次运行的文件和JSON日志保存在各模型子文件夹中。

---

### 第4.4节 — 评估结果

| 论文输出 | 脚本 | 输入 | 备注 |
|---------|------|------|------|
| 表5（RMSE/MAE/CRPS等） | `results/section 4.4/evaluation_metrics_values.ipynb` | `results/*.xlsx` + `results/actual.xlsx` | |
| 表5（CS评分） | `results/section 4.4/constraint_score_calculation.xlsx` | 同上 | 直接用Excel打开 |
| 图4（点预测） | `results/section 4.4/point prediction/plot_point.ipynb` | `point prediction/point_at_tau=0.5.xlsx` | |
| 图5（区间预测） | `results/section 4.4/interval prediction/plot_interval.ipynb` | `results/*.xlsx` | |
| 图6（KDE密度图） | `results/section 4.4/probability density prediction/kdePLOT.ipynb` | `kde_restructured.xlsx`（由`cs_plot.xlsx`生成） | |

---

### 第5.1节 — 福建数据集验证

| 论文输出     | 脚本                                          | 输入                     |
|----------|---------------------------------------------|------------------------|
| 附录E 图E.1 | `5.1 .../data detection/datadetection.ipynb` | `data_fujian.xlsx`     |
| 附录E 表E.2 | python代码包"pandas"运行describe()函数即可  | 同上                     |
| 表F.1     | `5.1 .../evaluation_metrics_values.ipynb`   | `5.1 .../*f.xlsx` 系列文件 |

> 福建数据集的模型训练结构与第4.3节相同，使用福建专用输入文件。

---

### 第5.2节 — 统计检验

| 论文输出 | 脚本 | 输入 | 结果文件 |
|---------|------|------|---------|
| 表6（Wilcoxon检验） | `5.2 Statistical analysis/5.2.ipynb` | `5.2 .../pointwise_metrics.xlsx` | `statistical_tests_results.xlsx` |
| 表7（DM检验） | `5.2 Statistical analysis/5.2.ipynb` | `results/*.xlsx` + `results/actual.xlsx` | `dm_test_pinball_loss_results.xlsx` |

---

### 第5.3节 — 运行时间（表8）

| 组件 | 获取方式 |
|------|---------|
| TVFEMD / R2CMSE耗时 | 在`4.1-4.3/preprocessing.ipynb`中使用`line_profiler`工具测量 |
| 模型预测耗时 | 读取各模型`.json`文件中的`test_duration_seconds`字段 |
| 超参数搜索耗时 | 读取各模型`.json`文件中的`optuna_duration_formatted`字段 |

---

### 第5.4节 — 敏感性分析（表9–11）

| 参数 | 修改位置 | 脚本 |
|------|---------|------|
| TVFEMD的`bsp_order` | 修改Cell *"4.1 TVFEMD decomposition"* | `4.1-4.3/preprocessing.ipynb` |
| R2CMSE的`tau` | 修改Cell *"5.1 R2CMSE results"* | `4.1-4.3/preprocessing.ipynb` |
| MCQRNN单隐藏层 | 见`5.4 Sensitivity analysis/trmg1l.py` | 输入同TRMG；每个子序列输出：`trmg1_single1-5.xlsx`，相加总和`trmg1l.xlsx`；JSON：`trmg1-5_single_hidden_results.json`，记录超参数搜索结果（同第4.3节） |

---

### 第5.5节 — 消融实验（表11）

| 模型 | 脚本 | 输入 | 输出 | JSON日志 |
|------|------|------|------|---------|
| TFMG（去掉R2CMSE） | `5.5 Ablation analysis/tfmg/tfmg.py` | `5.5 .../tfmg/fuzzy.csv` | `tfmg/tfmg.xlsx` | `tfmg/tfmg1-5_results.json` |
| TRQ1G（单隐藏层QRNN） | `5.5 Ablation analysis/trq1g/trq1g.py` | `5.5 .../trq1g/imfreconstruction_TenneT.csv` | `trq1g/trq1g.xlsx` | `trq1g/trq1g1-5_results.json` |

---

## 6. 复现流程

### ✅ 方案A — 快速验证（推荐，仅需CPU，约10分钟）

无需训练模型，直接使用`results/`中的预计算文件。

| 步骤 | 运行脚本 | 复现内容 |
|------|---------|---------|
| 1 | `results/section 4.4/evaluation_metrics_values.ipynb` | 表5 |
| 2 | `5.2 Statistical analysis/5.2.ipynb` | 表6、表7 |
| 3 | `results/section 4.4/point prediction/plot_point.ipynb` | 图4 |
| 4 | `results/section 4.4/interval prediction/plot_interval.ipynb` | 图5 |
| 5 | `results/section 4.4/probability density prediction/kdePLOT.ipynb` | 图6 |

---

### 🔁 方案B — 使用已保存超参数重新训练（需要GPU）

通过读取`.json`文件跳过超参数搜索，直接用最优参数训练。

```python
# 示例：读取JSON中的最优超参数并重新训练（以MCQRNNG类模型为例）
import json

with open('trmg1_results.json', 'r') as f:
    saved = json.load(f)

best_params = saved['best_hyperparameters']
# 包含字段：n_hidden, n_hidden2, penalty

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

> ⚠️ **关于GPU复现性的说明**：即使使用完全相同的硬件、软件版本和随机种子，  
> GPU训练结果在不同时间运行之间仍可能存在细微数值差异。  
> 这是深度学习框架中GPU非确定性的已知限制  
> （参见[TensorFlow确定性文档](https://github.com/mm3509/reproducibility/blob/master/tensorflow-reproducibility.md)）。
> `results/`中的预计算文件是论文中使用的精确输出。

---

### 🔬 方案C — 从头完整复现（需要GPU，约数小时）

> ⚠️ 同方案B，GPU复现性说明同样适用。

1. 运行`4.1-4.3/preprocessing.ipynb`——数据清洗、TVFEMD分解、R2CMSE重构
2. 运行`4.1-4.3/`中的各模型脚本（基于分解的模型各运行5次）
3. 各次运行结果求和 → 放入`results/`
4. 运行评估脚本（同方案A）

---

## 7. 可复现性说明

本研究使用GPU加速的深度学习模型。即使固定了随机种子（seed=42）、  
启用了`TF_DETERMINISTIC_OPS=1`和`TF_CUDNN_DETERMINISTIC=1`，  
由于GPU加速运算中固有的非确定性，**训练输出的精确数值复现无法跨硬件配置或跨时间保证**。

这是深度学习领域已知的、有文献记录的普遍限制，并非本研究特有。

**此种设定变更未引发结果的大幅偏离，所得结论与正文一致。**

**`results/`中的预计算预测文件是论文中使用的精确输出。**  
论文报告的所有表格和图片均可通过方案A从这些文件完全确定性地复现，无需GPU。

---

## 8. 外部依赖

| 工具 | 作用 | 要求 | 链接 |
|------|------|------|------|
| TVFEMD | 时变滤波EMD分解 | Python | https://github.com/stfbnc/pytvfemd |
| R2CMSE | IMF重构（原为MATLAB版本） | 已转为Python | https://github.com/Shurun-Wang/R2CMSE |
| MCQRNN | MCQRNN架构 | Python | https://github.com/RektPunk/mcqrnn |
---
