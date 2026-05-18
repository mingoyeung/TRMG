# 复现包说明文档

# 基于TRMG的海上风电概率预测
[英文版说明](README.md)

---

## 1. 项目目录结构

所有输入数据、模型脚本、结果文件和评估笔记本均**完整包含在本项目文件夹内**，不依赖任何外部路径。

```
trmg-replication/                         ← 项目根目录（请将工作目录设置至此）
│
├── README.md                             ← 英文说明文档
├── README_CN.md                          ← 中文说明文档（本文件）
├── requirements.txt
│
├── data/                                 ← ⭐ 所有原始及处理后输入数据（统一存放）
│   ├── preprocessing.ipynb               ← 数据预处理文件（TVFEMD、R2CMSE、图2–3、表3、图E.2）
│   ├── chart.eddx                        ← Figure 3和E.2完整绘图模板
│   ├── tennet/
│   │   ├── time_series_15min_singleindex_filtered.csv   ← TenneT原始数据
│   │   ├── time_series_15min_cleaned.csv                ← TenneT预处理后数据
│   │   ├── emdresults.csv                               ← TenneT EMD分解结果
│   │   ├── tvfemdresults_TenneT.csv                     ← TenneT TVFEMD分解结果
│   │   ├── vmd_decomposition_results.csv                ← TenneT VMD分解结果
│   │   └── wt_results.csv                               ← TenneT WT分解结果
│   └── fujian/
│       ├── detection.ipynb              ← 福建数据集异常检测（表E.1、图E.1）
│       ├── data_fujian.xlsx             ← 福建原始数据
│       ├── res_new.csv                  ← 福建预处理后数据
│       ├── emdf.csv                     ← 福建 EMD分解结果
│       ├── tvfemd_fujian.csv            ← 福建 TVFEMD分解结果
│       ├── vmd_fujian.csv               ← 福建 VMD分解结果
│       └── wt_fujian.csv               ← 福建 WT分解结果
│
├── results/                             ← ⭐ 预计算输出文件（方案A从此处开始）
│   ├── actual.xlsx                      ← 测试集真实值（2,621个数据点）
│   ├── trmg.xlsx                        ← TRMG预测结果
│   ├── trqg.xlsx                        ← TRQG预测结果
│   ├── wrgg.xlsx                        ← WRGG预测结果
│   ├── vrlg.xlsx                        ← VRLG预测结果
│   ├── ermg.xlsx                        ← ERMG预测结果
│   ├── tsmg.xlsx                        ← TSMG预测结果
│   ├── qrnn2pred.xlsx                   ← QRNN2G预测结果
│   ├── mcqrnn2pred.xlsx                 ← MCQRNNG预测结果
│   ├── predqrf.xlsx                     ← QRFG预测结果
│   ├── lassopred.xlsx                   ← QRLASSOG预测结果
│   ├── tfmg.xlsx                        ← TFMG预测结果（消融实验）
│   ├── trmg1l.xlsx                      ← TRMG1L预测结果（敏感性分析）
│   ├── trqg1.xlsx                       ← TRQ1G预测结果（消融实验）
│   ├── constraint_score_calculation.xlsx ← 表5 CS列（Excel中预计算）
│   ├── cs_plot.xlsx                     ← 图6数据提取
│   ├── kde_restructured.xlsx            ← 图6 KDE输入
│   ├── point_at_tau=0.5.xlsx            ← 图4输入
│   ├── pointwise_metrics.xlsx            ← 表6指标
│   ├── evaluation_metrics_values.ipynb  ← 表5脚本
│   ├── plot_point.ipynb                 ← 图4脚本
│   ├── plot_interval.ipynb              ← 图5脚本
│   ├── kdeplot.ipynb                    ← 图6脚本
│   ├── 5.2.ipynb                        ← 表6和表7脚本
│   └── section5.2/
│       ├── dm_test_pinball_loss_results.xlsx ← 表7结果
│       ├── pointwise_metrics.xlsx            ← 表6指标
│       └── statistical_tests_results.xlsx    ← 表6及表C.1结果
│
├── section4/                            ← 第4节：模型训练（TenneT数据集）
│   ├── trmg/
│   │   ├── trmg.py
│   │   ├── imfreconstruction_TenneT.xlsx
│   │   ├── trmg1-5_results.json
│   │   └── trmg1-5.xlsx
│   ├── trqg/
│   │   ├── trqg.py
│   │   ├── imfreconstruction_TenneT.xlsx
│   │   ├── trqg1-5_results.json
│   │   └── trqg1-5.xlsx
│   ├── wrgg/
│   │   ├── wtgru.py
│   │   ├── wt_decomposition_results.xlsx
│   │   ├── wtgru1-5_results.json
│   │   └── wtgru1-5.xlsx
│   ├── vrlg/
│   │   ├── vmdlstm.py
│   │   ├── vmd_reconstruction.xlsx
│   │   ├── qrlstm1-5_results.json
│   │   └── vmdlstm1-5.xlsx
│   ├── ermg/
│   │   ├── ermg.py
│   │   ├── emdreconstruction.xlsx
│   │   ├── ermg1-5_results.json
│   │   └── ermg1-5.xlsx
│   ├── tsmg/
│   │   ├── tsmg.py
│   │   ├── sereconstruction.xlsx
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
├── section5.1/                         ← 福建数据集实验
│   ├── evaluation_metrics_values.ipynb ← 表F.1脚本
│   ├── actualf.xlsx                    ← 福建测试集真实值（1,142个数据点）
│   ├── constraint_score_fujian.xlsx    ← 表F.1 CS列（Excel中预计算）
│   ├── *f.xlsx文件                     ← 福建数据集各模型预测结果
│   └── 各模型名称命名的文件夹              ← 福建数据集各模型代码和json文件（结构同section4）
│
├── section5.4/                         ← 敏感性分析
│   ├── section5.4.1/                   ← thresh_bwr敏感性（表9）：CSV输出文件
│   ├── section5.4.2/                   ← tau敏感性（表10）：CSV输出文件
│   └── section5.4.3/                   ← MCQRNN隐层敏感性（表11敏感性列）
│       ├── trmg1l.py
│       ├── trmg1_single1-5.xlsx        ← 单次运行输出（求和→results/trmg1l.xlsx）
│       └── trmg1-5_single_hidden_results.json
│
└── section5.5/                         ← 消融实验（表11消融列）
    ├── tfmg/
    │   ├── tfmg.py
    │   ├── fuzzy.xlsx
    │   ├── tfmg1-5_results.json
    │   └── tfmg1-5.xlsx
    └── trq1g/
        ├── trq1g.py
        ├── imfreconstruction_TenneT.xlsx
        ├── trq1g1-5_results.json
        └── trq1g1-5.xlsx
```

---

## 2. 计算环境

| 组件 | 配置 |
|------|------|
| 操作系统 | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA | 12.2 |
| cuDNN | 8.8 |
| Python | 3.10 |

> ⚠️ **GPU仅在模型训练时需要（方案B和方案C）。**
> 所有评估笔记本（方案A）均可在CPU上运行。

---

## 3. 安装说明

```bash
# 安装所有所需Python包
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
| CRPS | 2.0.4 | CRPS指标 |
| KDEpy | 1.1.12 | KDE可视化 |
| quantile-forest | 1.4.1 | QRF模型 |
| asgl | 2.1.4 | LASSO模型 |
| dieboldmariano | 1.0.0 | DM检验 |
| openpyxl | 3.1.5 | Excel读写 |
| matplotlib | 3.7.2 | 绘图 |

---

## 4. 数据说明

所有数据文件均集中存放于 `data/`（原始及清洗后输入）和 `results/`（测试集真实值及预计算预测结果）。所有文件引用均在项目根目录内，无外部路径依赖。

| 数据集 | 路径 | 行数 | 用于 |
|--------|------|------|------|
| TenneT原始数据 | `data/tennet/time_series_15min_singleindex_filtered.csv` | ~26,000 | 第4节 |
| TenneT清洗后数据 | `data/tennet/time_series_15min_cleaned.csv` | ~26,000 | 第4节 |
| 福建原始数据 | `data/fujian/data_fujian.xlsx` | ~11,000 | 第5.1节 |
| 福建清洗后数据 | `data/fujian/res_new.csv` | ~11,000 | 第5.1节 |
| TenneT测试集真实值 | `results/actual.xlsx` | 2,621 | 第4节 |
| 福建测试集真实值 | `section5.1/actualf.xlsx` | 1,142 | 第5.1节 |

---

## 5. 逐步复现指南

### 方案A — 快速验证（推荐 · 仅需CPU · 总计约30–60分钟）

本方案使用 `results/` 中预存的预计算预测文件，复现**所有正文表格和图形**，无需GPU或重新训练模型。

**前置步骤：**
1. 安装依赖：`pip install -r requirements.txt`
2. 从项目根目录启动Jupyter：`jupyter notebook`
3. 运行任意笔记本前，请确认工作目录已设置为项目根目录。

---

#### 第1步 — 复现表3和图2–3（数据分析与分解）

**笔记本：** `data/preprocessing.ipynb`

| 代码块标注（与笔记本注释一致）                  | 输出内容 | 预计运行时间 |
|----------------------------------|---------|------------|
| *"Data cleaning"与Seasonal test"* | 表3（数据集统计信息） | 约1分钟 |
| *"Visualization (Fig 2)"*        | 图2（数据集概览） | 约1分钟 |
| *"TVFEMD + R2CMSE"*（TenneT数据集部分） | 图3(a) TVFEMD分解结果、图3(b)(c) R2CMSE重构结果、表8运行时间、表9–10敏感性结果 | 约30–40分钟 |

> 不同的BLAS（基础线性代数子程序）可能导致TVFEMD分解结果不同，之前的分解结果储存在`data/tennet/tvfemdresults_TenneT.csv`。使用该数据运行 For differences in scipy.sparse.linalg.spsolve() cell单元格的代码及后续分析。

> **关于图3的拼图说明：** 完整的图3面板使用 [EdrawMax](https://www.edrawmax.cn) 拼合而成，源文件 `Chart.eddx` 已包含在复现包中。子图3(a)和3(b)由上述笔记本单元格直接生成，与论文完全一致。

---

#### 第2步 — 复现表5和图4–6（主要评估结果）

| 笔记本/文件 | 操作步骤 | 输出内容 |
|------------|---------|---------|
| `results/evaluation_metrics_values.ipynb` | 按顺序运行所有单元格；依次从 `results/` 中选择各模型的 `.xlsx` 文件 | 表5（CS列以外的所有指标） |
| `results/constraint_score_calculation.xlsx` | 打开文件；在每个模型对应的工作表中读取**AN1单元格**（黄色高亮） | 表5（CS分值列） |
| `results/plot_point.ipynb` | 运行所有单元格 | 图4（点预测） |
| `results/plot_interval.ipynb` | 运行所有单元格 | 图5（区间预测） |
| `results/kdeplot.ipynb` | 运行 *"KDE plot"* 部分的所有单元格 | 图6（核密度预测） |

> **关于QS（MW）分值：** 本实现中，分位数得分（QS）等价于Pinball loss，两者指同一指标。`results/evaluation_metrics_values.ipynb` 中相关单元格已标注为 *"Probability prediction: Pinball loss (QS in Table 5)"*。

> **CS（MW）值的具体定位方式：** 打开 `results/constraint_score_calculation.xlsx`，
> 每个模型有独立工作表（以模型名称命名）。
> 中间惩罚项在T至AK列计算，结果存于AM1单元格；
> AN1单元格包含平方项及最终CS值（黄色高亮标注）。
> 该结构适用于所有模型工作表。

---

#### 第3步 — 复现表6–7和表C.1（统计检验）

**笔记本：** `results/5.2.ipynb`

分两个阶段运行：

**第一阶段 — 提取逐点指标（表6和表C.1的前置步骤）：**

| 输入 | 单元格标注 | 输出 |
|------|-----------|------|
| 所有 `results/*.xlsx` + `results/actual.xlsx` | *"Wilcoxon indices extraction"* | `results/pointwise_metrics.xlsx`（各模型的RMSE、MAE、CRPS） |

**第二阶段 — 运行统计检验：**

| 单元格标注 | 输入 | 输出 | 对应论文表格 |
|-----------|------|------|------------|
| *"Wilcoxon test"* | `results/pointwise_metrics.xlsx` | `results/section5.2/statistical_tests_results.xlsx` 中的Wilcoxon工作表 | 表6（Wilcoxon检验） |
| *"Shapiro-Wilk test"* | `results/pointwise_metrics.xlsx` | `results/section5.2/statistical_tests_results.xlsx` 中的Shapiro工作表 | 表C.1 |
| *"DM test"* | 所有 `results/*.xlsx` + `results/actual.xlsx` | `results/section5.2/dm_test_pinball_loss_results.xlsx` | 表7（DM检验） |

---

#### 第4步 — 复现表8–11（运行时间、敏感性分析、消融实验）

**表8 — 执行时间：**

| 组件 | 数值获取方式 |
|------|------------|
| TVFEMD分解时间 | 运行 `data/preprocessing.ipynb`（TenneT数据集部分）中的 *"TVFEMD + R2CMSE"* 单元格，执行完成后读取 `lp.print_stats()` 输出中TVFEMD函数对应的 **`Total time`** 行。 |
| R2CMSE重构时间 | 同上，读取 `lp.print_stats()` 输出中R2CMSE函数对应的 **`Total time`** 行。 |
| 模型Optuna训练时间 | 打开 `section4/` 下各模型的 `.json` 文件，读取 `optuna_duration_formatted` 字段。 |
| 模型测试时间 | 打开 `section4/` 下各模型的 `.json` 文件，读取 `test_duration_seconds` 字段。 |

> ⚠️ 实际运行时间因操作系统调度、CPU缓存状态及LineProfiler插桩开销而存在波动。论文中报告的数值来自单次受控实验，具有代表性，并非固定常数。

**表9–10 — 敏感性分析（TVFEMD和R2CMSE）：**

| 输出内容 | 获取方式                                                                                                                                        |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 表9（TVFEMD `thresh_bwr` 敏感性） | 不同`thresh_bwr`分解结果存于 `section5.4/section5.4.1/`。也可通过 `data/preprocessing.ipynb` 中 *"TVFEMD + R2CMSE"* 单元格内的 `# Table 9 Summary` 代码块打印输出复现。  |
| 表10（R2CMSE `tau` 敏感性） | 计算结果CSV文件存于 `section5.4/section5.4.2/`。也可通过同一单元格内的 `# Sensitivity results for R2CMSE Table 10 saved` 代码块输出 `r2cmse_tau_sensitivity.csv` 复现。 |

> **关于表11中的TRMG和MCQRNNG：** 这两个模型在敏感性/消融分析部分**不进行单独评估**，
> 其表11结果与表5**完全相同**，直接从相同文件（`results/trmg.xlsx` 和
> `results/mcqrnn2pred.xlsx`）加载，无需重新计算。
> `results/evaluation_metrics_values.ipynb` 中已添加明确注释说明此点。

**表11 — 敏感性列与消融列：**

| 列 | 加载文件 | 操作步骤 |
|----|---------|---------|
| 敏感性列（MCQRNN隐层数） | `results/trmg1l.xlsx` | 在 `results/evaluation_metrics_values.ipynb` 中取消注释 `trmg1l` 加载行，运行所有单元格。 |
| 消融列（无R2CMSE / 单隐层QRNN） | `results/tfmg.xlsx` 和 `results/trq1g.xlsx` | 在 `results/evaluation_metrics_values.ipynb` 中取消注释对应加载行，运行所有单元格。 |

---

#### 第5步 — 复现附录E和表F.1（福建数据集）

> ⚠️ **关于福建数据集各分量预测文件：** 由于原始实验完成后发生文件损坏，福建数据集各IMF分量的单次预测文件（即各 `*f1.xlsx` 至 `*f4/5.xlsx`）已无法恢复。各分量预测结果求和后的最终聚合预测文件（如 `trmgf.xlsx`）完整保存于 `section5.1/`。表F.1和图E.2的所有报告结果均可由这些聚合文件通过**方案A**完整复现，不受影响。如需查看各分量中间输出，可通过**方案C**重新训练获得。

| 笔记本/文件 | 单元格标注 | 输入 | 输出                                                       |
|------------|-----------|------|----------------------------------------------------------|
| `data/fujian/detection.ipynb` | *"Missing value detection"* | `data/fujian/data_fujian.xlsx` | 表E.1（缺失值情况）                                              |
| `data/fujian/detection.ipynb` | *"Descriptive"* | `data/fujian/data_fujian.xlsx` | 表E.2（描述性统计）                                              |
| `data/fujian/detection.ipynb` | *"Heatmap for Fig E.1(a) in Outlier detection"* | `data/fujian/data_fujian.xlsx` | 图E.1(a)                                                  |
| `data/fujian/detection.ipynb` | *"RANSAC fit in Fig E.1(b) in Outlier detection"* | `data/fujian/data_fujian.xlsx` | 图E.1(b) + `data/fujian/res_new.csv`                      |
| `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"*（福建数据集部分） | `data/fujian/res_new.csv` | 图E.2(a)                                                  |
| `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"*（福建数据集部分） | `section5.1/trmg/tvfemd_fujian.csv` | 图E.2(b)(c) + `section5.1/trmg/tvfemdresults_fujian.xlsx` |
| `section5.1/evaluation_metrics_values.ipynb` | 运行所有单元格 | `section5.1/*f.xlsx` + `section5.1/actualf.xlsx` | 表F.1（CS列以外的所有指标）                                         |
| `section5.1/constraint_score_fujian.xlsx` | 读取**AN1单元格**（黄色高亮） | 预计算 | 表F.1（CS分值列）                                              |

> **关于表F.1中的QS（MW）分值：** 同表5，QS指标对应Pinball loss。`section5.1/evaluation_metrics_values.ipynb` 中相关单元格已标注为 *"Probability prediction: Pinball loss (QS in Table F.1)"*。

---

### 方案B — 使用已保存超参数重新训练（需要GPU · 约数小时）

本方案使用各模型 `.json` 文件中已保存的最优超参数重新训练模型。由于GPU不确定性，结果可能与论文存在轻微差异（见第7节）。

```bash
# 示例：重新训练TRMG
# 脚本将自动从 trmg1-5_results.json 中加载最优超参数
cd section4/trmg
# 选择MODE = 'B' ，代码位于if __name__ == "__main__":后
python trmg.py   # 选择trmg1/2/3/4/5，分别在终端运行5次，生成 trmg1.xlsx, trmg2.xlsx, ..., trmg5.xlsx
```

5次运行完成后，将各次输出文件求和得到最终预测文件：

```python
import pandas as pd

runs = [pd.read_excel(f"section4/trmg/trmg{i}.xlsx") for i in range(1, 6)]
final = sum(runs)
final.to_excel("results/trmg.xlsx", index=False)
```

对 `section4/` 下每个模型重复上述流程。注意事项：
- **TSMG** 仅需在终端运行 **4次**（而非5次）。
- **QRNN2G、MCQRNNG**：各运行一次，脚本自动加载 `optimization_results.json`。
- **QRFG**：使用 `section4/qrf/qrfprediction.ipynb`；运行前请手动加载 `optuna_qrf_results.json`。
- **QRLASSOG**：使用 `section4/qrlasso/qrlasso.ipynb`；运行前请手动加载 `optuna_lasso_results.json`。

重新训练完成后，按方案A第2–5步进行评估。

---

### 方案C — 从原始数据完整复现（需要GPU · 约数小时）

> ⚠️ GPU可复现性说明同方案B（见第7节）。

```bash
# 第1步：预处理数据并运行TVFEMD + R2CMSE分解
# 打开 data/preprocessing.ipynb，运行目标数据集（TenneT或福建）对应的单元格。
# 为避免覆盖分解输出，请仅运行与下一步待训练模型/方法对应的单元格。
# 输出：中间CSV文件保存至 section4/ 下各模型子文件夹。

# 第2步：对于.py文件，选择MODE = 'C' ，代码位于if __name__ == "__main__":后（分解类模型需在终端运行5次；详见方案B）
#对于.ipynb文件(QRLASSOG和QRFG)，选择运行Option C的单元格训练模型。
# 以TRMG为例：
cd section4/trmg
python trmg.py   #重复5次；得到trmg1/2/3/4/5_results.json，生成 trmg1.xlsx ... trmg5.xlsx

# 第3步：对各次运行输出求和，将最终文件复制至 results/
# （使用方案B中展示的求和代码）
cp section4/trmg/trmg.xlsx results/trmg.xlsx

# 第4步：运行所有评估笔记本（同方案A第2–5步）
```

---

## 6. 代码与论文对照表

### 第4.1节 — 数据分析

| 论文输出              | 脚本 | 单元格标注                             | 输入 |
|-------------------|------|-----------------------------------|------|
| 表3（异常值、缺失值和季节性检验） | `data/preprocessing.ipynb` | *"Data cleaning"和"Seasonal test"* | `data/tennet/time_series_15min_singleindex_filtered.csv` |
| 图2                | `data/preprocessing.ipynb` | *"Visualization (Fig 2)"*         | 同上 |

---

### 第4.2节 — 分解与重构（`data/preprocessing.ipynb`，TenneT数据集部分）

> 信号分解全程在Python中完成。基于熵的IMF重构步骤（按熵值范围筛选并求和IMF分量）在Excel中手动执行，所得CSV文件列于下表的"数据输出"列。

| 论文输出 | 单元格标注（与笔记本注释一致）                                          | 输入 | 数据输出                                                                                                      |
|---------|----------------------------------------------------------|------|-----------------------------------------------------------------------------------------------------------|
| 图3(a) TVFEMD | *"TVFEMD + R2CMSE"*：`# Figure 3(a): plot TVFEMD results` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `section4/trmg/tvfemdresults_TenneT.csv`                                                                  |
| 图3(b)(c) R2CMSE | *"TVFEMD + R2CMSE"*：`# 4.2 R2CMSE: load TVFEMD results`  | `section4/trmg/tvfemdresults_TenneT.csv` | `section4/trmg/imfreconstruction_TenneT.xlsx` (TRMG,TRQG,TRMG1L,TRQ1G都适用)                                 |
| WRGG：附录D表D.1 | *"WT + R2CMSE"*：`WT decomposition` 和 `R2CMSE results`部分  | `data/tennet/time_series_15min_singleindex_filtered.csv` | <br/>分解:`data/tennet/wt_decomposition_results.csv`, <br/>重构:`section4/wrgg/wt_decomposition_results.xlsx` |
| VRLG：附录D表D.2 | *"VMD + R2CMSE"*：`VMD decomposition` 和 `R2CMSE results`部分  | `data/tennet/time_series_15min_singleindex_filtered.csv` | <br/>分解:`data/tennet/vmd_decomposition_results.csv`, <br/>重构:`section4/vrlg/vmd_reconstruction.xlsx`      |
| ERMG：附录D表D.3 | *"EMD + R2CMSE"*：`EMD decomposition` 和 `R2CMSE results`部分  | `data/tennet/time_series_15min_singleindex_filtered.csv` | <br/>分解:`data/tennet/emdresults.csv`, <br/>重构:`section4/ermg/emdreconstruction.xlsx`                      |
| TSMG：附录D表D.4 | *"TVFEMD + SE"*：所有单元格                                    | `data/tennet/tvfemdresults_TenneT.csv` | <br/>分解:`data/tennet/wt_decomposition_results.csv`,<br/>重构:`section4/tsmg/sereconstruction.xlsx`          |

> **sub1–sub5的形成方式说明（适用于附录D表D.1–D.4及图3(c)）：**
> sub1–sub5各列**不代表单个R2CMSE值**，而是将R2CMSE值相近的IMF分量**归组求和**后得到的时间序列，
> 具体分组依据论文中的熵值范围重构原则确定。以TRMG为例：
> IMF1、8–11（≈0.500）→ sub1；IMF2–4（≈0.650）→ sub2；IMF5–7（≈0.580）→ sub3；
> IMF12–14（0.2–0.3）→ sub4；残差（独立范围）→ sub5。
>
> 各sub分量时间序列为对应IMF列的**列方向求和**，求和公式已通过Excel单元格公式在附带Excel文件中显式展示。
---

### 第4.3节 — 模型训练（表4）

> ⚠️ **审计说明 — 表4（日志文件查阅验证，非代码运行复现）：**
> 在**方案A**下，表4的超参数值**无法通过运行代码复现**，因为方案A使用预计算预测文件，不涉及模型重新训练。
> 表4可通过查阅各模型子文件夹下的JSON日志文件进行验证：

| 模型 | 脚本 | 输入CSV                                                    | 输出（存于 `results/`） | 超参数日志 |
|------|------|----------------------------------------------------------|----------------------|-----------|
| TRMG（本文提出） | `section4/trmg/trmg.py` | `section4/trmg/imfreconstruction_TenneT.xlsx`            | `trmg.xlsx` | `section4/trmg/trmg1-5_results.json` |
| TRQG | `section4/trqg/trqg.py` | `section4/trqg/imfreconstruction_TenneT.xlsx`            | `trqg.xlsx` | `section4/trqg/trqg1-5_results.json` |
| WRGG | `section4/wrgg/wtgru.py` | `section4/wrgg/wt_decomposition_results.xlsx`            | `wrgg.xlsx` | `section4/wrgg/wtgru1-5_results.json` |
| VRLG | `section4/vrlg/vmdlstm.py` | `section4/vrlg/vmd_reconstruction.xlsx`                  | `vrlg.xlsx` | `section4/vrlg/qrlstm1-5_results.json` |
| ERMG | `section4/ermg/ermg.py` | `section4/ermg/emdreconstruction.xlsx`                   | `ermg.xlsx` | `section4/ermg/ermg1-5_results.json` |
| TSMG | `section4/tsmg/tsmg.py` | `section4/tsmg/sereconstruction.xlsx`                    | `tsmg.xlsx` | `section4/tsmg/tsmg1-4_results.json` |
| QRNN2G | `section4/qrnn2/qr2nn.py` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `qrnn2pred.xlsx` | `section4/qrnn2/optimization_results.json` |
| MCQRNNG | `section4/mcqrnn/mcqrnn.py` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `mcqrnn2pred.xlsx` | `section4/mcqrnn/optimization_results.json` |
| QRFG | `section4/qrf/qrfprediction.ipynb` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `predqrf.xlsx` | `section4/qrf/optuna_qrf_results.json` |
| QRLASSOG | `section4/qrlasso/qrlasso.ipynb` | `data/tennet/time_series_15min_singleindex_filtered.csv` | `lassopred.xlsx` | `section4/qrlasso/optuna_lasso_results.json` |

> 每个文件中的 **`best_hyperparameters`** 字段直接对应表4中报告的超参数值；
> **`timestamp`** 字段确认这些值来自实际的Optuna训练运行，而非人工设定。

> 如需通过代码运行复现表4，请使用**方案C**（需要GPU，见第5节）
> 
> `results/` 中已存有预计算输出，仅在完整复现（方案B/C）时需重新训练。

> **多次运行说明：** 每个基于分解的模型（TRMG、TRQG、WRGG、VRLG、ERMG、TSMG）需训练5次，每次对应一个重构的IMF分量（如 `trmg1.xlsx` 至 `trmg5.xlsx`），TSMG仅需4次。`results/trmg.xlsx` 等最终输出文件为上述单次运行输出的**求和结果**。

---

### 第4.4节 — 评估结果

| 论文输出 | 脚本 | 操作步骤 | 输入 |
|---------|------|---------|------|
| 表5（CS列以外的所有指标） | `results/evaluation_metrics_values.ipynb` | 运行所有单元格；依次从 `results/` 中选择各模型 `.xlsx` 文件 | `results/*.xlsx` + `results/actual.xlsx` |
| 表5（CS分值） | `results/constraint_score_calculation.xlsx` | 打开文件；在每个模型工作表中读取**AN1单元格**（黄色高亮） | 预计算 |
| 表5（QS/MW分值） | `results/evaluation_metrics_values.ipynb` | 标注为 *"Probability prediction: Pinball loss (QS in Table 5)"* 的单元格 | `results/*.xlsx` + `results/actual.xlsx` |
| 图4（点预测） | `results/plot_point.ipynb` | 运行所有单元格 | `results/point_at_tau=0.5.xlsx` |
| 图5（区间预测） | `results/plot_interval.ipynb` | 运行所有单元格 | `results/*.xlsx` + `results/actual.xlsx` |
| 图6（KDE密度） | `results/kdeplot.ipynb` | *"KDE plot"* 部分单元格 | `results/kde_restructured.xlsx` |

> 如何得到`results/kde_restructured.xlsx`：输入数据`results/cs_plot.xlsx`，运行`results/kdeplot.ipynb`中的Data extraction部分。
---

### 第5.1节 — 福建数据集

> **关于福建数据集各分量预测文件：** 由于原始实验完成后发生文件损坏，福建数据集各IMF分量的单次预测文件已无法恢复。各分量预测求和后的最终聚合预测文件（`section5.1/*f.xlsx`）完整保存且可正常使用，足以通过**方案A**完整复现表F.1和图E.2的所有报告结果，不受影响。聚合文件在损坏发生前已生成并保存，其完整性已通过将表F.1报告指标与由存档文件重新计算所得结果进行比对得到验证，两者完全吻合。

| 输出内容                                                                                          | 脚本 | 单元格标注 | 输入 |
|-----------------------------------------------------------------------------------------------|------|-----------|------|
| 表E.1（缺失值）                                                                                     | `data/fujian/detection.ipynb` | *"Missing value detection"* | `data/fujian/data_fujian.xlsx` |
| 表E.2（描述性统计）                                                                                   | `data/fujian/detection.ipynb` | *"Descriptive"* | `data/fujian/data_fujian.xlsx` |
| 图E.1(a)                                                                                       | `data/fujian/detection.ipynb` | *"Heatmap for Fig E.1(a) in Outlier detection"* | `data/fujian/data_fujian.xlsx` |
| 图E.1(b)                                                                                       | `data/fujian/detection.ipynb` | *"RANSAC fit in Fig E.1(b) in Outlier detection"* | `data/fujian/data_fujian.xlsx` |
| 图E.2(a)                                                                                       | `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"*（福建数据集部分） | `data/fujian/res_new.csv` |
| 图E.2(b)(c)                                                                                    | `data/preprocessing.ipynb` | *"TVFEMD + R2CMSE"*（福建数据集部分） | `section5.1/trmg/tvfemd_fujian.csv` |
| <br/>分解:`data/fujian/vmd_fujian.csv`, <br/>重构:`section5.1/trmg/vmd_decomposition_fujian.xlsx` | `data/preprocessing.ipynb` | *"VMD + R2CMSE"* (Fujian dataset section)                                 | `data/fujian/res_new.csv`                           |
| <br/>分解:`data/fujian/wt_fujian.csv`,<br/>重构:`section5.1/trmg/wt_decomposition_fujian.xlsx`    | `data/preprocessing.ipynb` | *"WT + R2CMSE"* (Fujian dataset section)                                  | `data/fujian/res_new.csv`                           |
| <br/>分解:`data/fujian/emdf.csv`,<br/>重构:`section5.1/ermg/emdfujian.xlsx`                       | `data/preprocessing.ipynb` | *"EMD + R2CMSE"* (Fujian dataset section)                                 | `data/fujian/res_new.csv`                           |
| <br/>分解:`data/fujian/tvfemd_fujian.csv`,<br/>重构:`section5.1/tsmg/tsmg_fujian.xlsx`            | `data/preprocessing.ipynb` | *"TVFEMD + SE"* (Fujian dataset section)                                  | `data/fujian/res_new.csv`                           |
| 表F.1（CS列以外的所有指标）                                                                              | `section5.1/evaluation_metrics_values.ipynb` | 运行所有单元格 | `section5.1/*f.xlsx` + `section5.1/actualf.xlsx` |
| 表F.1（CS分值）                                                                                    | `section5.1/constraint_score_fujian.xlsx` | 读取**AN1单元格**（黄色高亮） | 预计算 |
| 表F.1（QS/MW分值）                                                                                 | `section5.1/evaluation_metrics_values.ipynb` | 标注为 *"Probability prediction: Pinball loss (QS in Table F.1)"* 的单元格 | `section5.1/*f.xlsx` + `section5.1/actualf.xlsx` |

> 福建数据集的模型训练流程与第4.3节完全一致。混合基线模型的重构结果展示于 `data/preprocessing.ipynb` 福建数据集部分的末尾（Markdown格式）。

---

### 第5.2节 — 统计检验

| 论文输出 | 输入 | 单元格标注 | 结果文件 |
|---------|------|-----------|---------|
| 表C.1（Shapiro-Wilk检验） | `results/pointwise_metrics.xlsx` | *"Wilcoxon test"*（Shapiro子步骤） | `results/section5.2/statistical_tests_results.xlsx` 中的Shapiro工作表 |
| 表6（Wilcoxon检验） | `results/pointwise_metrics.xlsx` | *"Wilcoxon test"* | `results/section5.2/statistical_tests_results.xlsx` 中的Wilcoxon工作表 |
| 表7（DM检验） | `results/*.xlsx` + `results/actual.xlsx` | *"DM test"* | `results/section5.2/dm_test_pinball_loss_results.xlsx` |

---

### 第5.3节 — 执行时间（表8）

| 组件 | 数值获取方式                                                                                                                     |
|------|----------------------------------------------------------------------------------------------------------------------------|
| TVFEMD分解时间 | 运行 `data/preprocessing.ipynb`（TenneT部分）中的 *"TVFEMD + R2CMSE"* 单元格，读取 `lp.print_stats()` 输出中TVFEMD函数对应的 **`Total time`** 行。 |
| R2CMSE重构时间 | 同上，读取 `lp.print_stats()` 输出中R2CMSE函数对应的 **`Total time`** 行。                                                                |
| 模型Optuna训练时间 | `section4/` 下各模型 `.json` 文件中的 `optuna_duration_formatted/total_time_sec` 字段。                                               |
| 模型测试时间 | `section4/` 下各模型 `.json` 文件中的 `test_duration_seconds/eval_time_sec` 字段。                                                    |

> ⚠️ 实际运行时间因操作系统调度、CPU缓存状态及插桩开销而存在波动，论文中的数值具有代表性。

---

### 第5.4节 — 敏感性分析

| 参数 | 论文输出 | 脚本/位置 | 输出 |
|------|---------|---------|------|
| TVFEMD `thresh_bwr` | 表9 | `data/preprocessing.ipynb`（TenneT部分）*"TVFEMD + R2CMSE"* 单元格，见 `# Table 9 Summary` 代码块 | `section5.4/section5.4.1/` 中的CSV文件 |
| R2CMSE `tau` | 表10 | 同上，见 `# Sensitivity results for R2CMSE Table 10 saved` 代码块 | `section5.4/section5.4.2/` 中的 `r2cmse_tau_sensitivity.csv` |
| MCQRNN隐层数 | 表11（敏感性列） | `section5.4/section5.4.3/trmg1l.py` | `trmg1_single1-5.xlsx`；求和→`results/trmg1l.xlsx`；日志：`trmg1-5_single_hidden_results.json` |

---

### 第5.5节 — 消融实验（表11消融列）

| 模型 | 脚本 | 输入                                              | 输出 | 日志 |
|------|------|-------------------------------------------------|------|------|
| TFMG（去除R2CMSE） | `section5.5/tfmg/tfmg.py` | `section5.5/tfmg/fuzzy.xlsx`                    | `results/tfmg.xlsx` | `section5.5/tfmg/tfmg1-5_results.json` |
| TRQ1G（单隐层QRNN） | `section5.5/trq1g/trq1g.py` | `section5.5/trq1g/imfreconstruction_TenneT.xlsx` | `results/trqg1.xlsx` | `section5.5/trq1g/trq1g1-5_results.json` |

> 复现表11消融列：在 `results/evaluation_metrics_values.ipynb` 中取消注释 `tfmg` 和 `trq1g` 的加载行后运行所有单元格。

> **关于表11中的TRMG和MCQRNNG：** 这两个模型在敏感性/消融分析部分不进行单独评估，其表11结果与表5完全相同，直接从相同文件（`results/trmg.xlsx` 和 `results/mcqrnn2pred.xlsx`）加载，无需重新计算。
---

## 7. 可复现性声明

>方案 A 为指定且完整的复现路径。归档在 data/、results/ 以及 section 5.1/ 目录下的所有预测结果，均在原始实验环境中生成，且已通过验证
> 可精确复现表 4–11 及附录表 C.1–F.1中所有汇报指标。

>关于基于 GPU 的重新训练（方案 B、方案 C）：本研究深度学习模型采用 random_normal 权重初始化方式，并通过 @tf.function 编译完成超 1000 轮训练迭代。
>即便设置固定随机种子（seed=42）并开启环境变量 TF_DETERMINISTIC_OPS=1，受 GPU 加速深度学习中两类公认的非确定性来源影响，
> 仍无法在同一硬件环境不同时间，或不同硬件环境下实现比特级完全一致的结果复现，这是因为：
> - **CUDA/cuDNN版本影响随机数生成序列：** 即便固定随机种子，仍会改变模型权重初始化结果；
> - **GPU并行梯度计算中浮点运算不满足结合律：** 经过多轮训练迭代后，会在不同平台上产生偏差梯度。

以上是 TensorFlow 在 GPU 硬件平台上公认的局限性，相关参考资料如下：

- [PyTorch reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html)
- [TensorFlow determinism docs](https://github.com/mm3509/reproducibility/blob/master/tensorflow-reproducibility.md)
- [Sources of Irreproducibility in Machine Learning](https://arxiv.org/pdf/2204.07610), Gundersen et al. (2022).

---

## 8. 外部依赖

| 工具 | 用途 | 链接 |
|------|------|------|
| TVFEMD | 时变滤波EMD（Python实现） | https://github.com/stfbnc/pytvfemd |
| R2CMSE | IMF重构（MATLAB；本包含Python转换版本） | https://github.com/Shurun-Wang/R2CMSE |
| MCQRNN | 单调分位数回归神经网络（Python） | https://github.com/RektPunk/mcqrnn |
| EdrawMax | 复合图形拼合（图3、图E.2） | https://www.edrawmax.cn |

---
