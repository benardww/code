# EEG Seizure Detection — ModernTCN + SE-Net + BiLSTM

基于 CHB-MIT 数据集的癫痫发作检测模型，采用 **ModernTCN（大核深度可分离卷积）+ 通道注意力（SE-Net）+ BiLSTM** 架构，输入 4 秒 EEG 片段，输出二分类（发作 / 正常）。

---

## 目录

- [环境配置](#环境配置)
- [数据集](#数据集)
- [快速开始](#快速开始)
- [手动分步运行](#手动分步运行)
- [main.py 参数说明](#mainpy-参数说明)
- [模型架构](#模型架构)
- [数据处理流程](#数据处理流程)
- [项目结构](#项目结构)

---

## 环境配置

```bash
pip install -r requirements.txt
```

**依赖环境：**

| 组件 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | ≥ 2.1.0（CUDA 12.1 推荐） |
| MNE | ≥ 1.6.0 |
| PyWavelets | ≥ 1.6.0 |

---

## 数据集

使用 [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)（PhysioNet）。

**数据放置：** 将下载的数据解压至项目根目录，保持原始目录结构：

```
physionet.org/
└── files/
    └── chbmit/
        └── 1.0.0/
            ├── chb01/
            │   ├── chb01_01.edf
            │   ├── chb01-summary.txt
            │   └── ...
            ├── chb02/
            └── ...（共 22 个受试者，排除 chb06 和 chb16）
```

---

## 快速开始

```bash
# 全量流程：预处理 + 训练 + 评估（所有受试者）
python main.py

# 单个受试者快速验证
python main.py --subjects chb01 --epochs 30

# 查看所有可用参数
python main.py --help
```

---

## 手动分步运行

```bash
# Step 1：预处理（生成 .npy 窗口文件）
python scripts/preprocess_all.py

# Step 2：训练（每个受试者单独训练）
python scripts/train.py --subject chb01 --device cuda

# Step 3：评估
python scripts/evaluate.py --subject chb01 --device cuda

# 功能验证（无需真实数据）
python scripts/test_pipeline.py
```

---

## main.py 参数说明

### 流程控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `all` | `all` / `preprocess` / `train` / `eval` |
| `--subjects` | `all` | 受试者列表（空格分隔）或 `all` |
| `--device` | `cuda` | `cuda` 或 `cpu` |

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `100` | 最大训练轮数 |
| `--batch_size` | `64` | 批大小 |
| `--lr` | `1e-3` | Adam 学习率 |
| `--weight_decay` | `1e-4` | L2 正则化 |
| `--focal_gamma` | `2.0` | Focal Loss 聚焦指数（α 由数据自动计算） |
| `--grad_clip` | `5.0` | 梯度裁剪阈值 |
| `--early_stop_patience` | `15` | 早停耐心轮数（监控 val_auc） |
| `--seed` | `42` | 随机种子 |

### 模型架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--d_model` | `128` | TCN / SE-Net 特征维度 |
| `--num_tcn_blocks` | `3` | ModernTCN 堆叠块数 |
| `--lstm_hidden` | `256` | BiLSTM 单方向隐层大小 |
| `--lstm_layers` | `2` | BiLSTM 层数 |

### 后处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--maf_window` | `5` | 移动平均滤波窗口（步数，1 步 = 1 s） |
| `--threshold` | `0.5` | 二值化阈值（验证集 F1 调优） |
| `--collar_sec` | `5.0` | 检测区间两端扩展秒数 |

### 路径配置

| 参数 | 默认值 |
|------|--------|
| `--data_raw_dir` | `physionet.org/files/chbmit/1.0.0` |
| `--data_processed_dir` | `data/processed` |
| `--experiments_dir` | `experiments` |

### 使用示例

```bash
# 提高召回率（降低阈值）
python main.py --subjects chb01 --threshold 0.3 --collar_sec 8.0

# 调整模型容量
python main.py --subjects chb01 --d_model 256 --num_tcn_blocks 4

# CPU 调试
python main.py --mode train --subjects chb01 --device cpu --epochs 5
```

---

## 模型架构

```
输入: (B, 18, 1024)
  ↓
Patch Embedding    Conv1d(18→128, k=16, stride=8) + InstanceNorm → (B, 128, 127)
  ↓
ModernTCN ×3       DWConv(k=51) + InstanceNorm + FFN(×4扩展) + 残差  → (B, 128, 127)
  ↓
SE-Net             GAP → FC(128→8) → ReLU → FC(8→128) → Sigmoid → Scale → (B, 128, 127)
  ↓
BiLSTM             (B,127,128) → LSTM(hidden=256, 2层, 双向) → 拼接末层 → (B, 512)
  ↓
分类头             FC(512→128) → ReLU → Dropout(0.5) → FC(128→2) → (B, 2)
```

**可训练参数量：约 3.2M**

---

## 数据处理流程

### 预处理

1. **通道选取**：从每个 EDF 文件中提取固定 18 条双极导联通道
2. **Db4 小波滤波**：5 层分解，保留 0–32 Hz（δ/θ/α/β），丢弃高频噪声
3. **差异化滑动窗口**（4 s，256 Hz，1024 采样点）：
   - 发作段：步长 512（50% 重叠），增强少数类样本
   - 非发作段：步长 1024（无重叠），控制多数类样本量
4. **受试者内时序划分**：按 EDF 文件时间顺序 → 70% 训练 / 15% 验证 / 15% 测试

### 训练

- 优化器：Adam + CosineAnnealingLR
- 损失：Focal Loss（α 由训练集发作比例自动计算，γ=2.0）
- WeightedRandomSampler 缓解剩余类别不平衡
- 混合精度（AMP）加速 GPU 训练

### 推理后处理

```
原始概率序列（均匀 1 s 步长）
  → MAF（5步移动平均，平滑孤立噪声峰）
  → Thresholding（默认 0.5）
  → Collar（每段两端延伸 ±5 s，合并相邻区间）
  → 最终二值预测
```

### 评估指标

| 指标 | 说明 |
|------|------|
| AUC-ROC | 后处理前，基于原始概率 |
| Sensitivity | 后处理后，发作检出率（主要指标） |
| Specificity | 后处理后，正常段排除率 |
| F1-Score | 后处理后 |

---

## 项目结构

```
.
├── main.py                        # 一键启动脚本
├── requirements.txt
├── physionet.org/                 # CHB-MIT 原始数据（需自行下载）
│   └── files/chbmit/1.0.0/
├── data/processed/                # 预处理输出（自动生成）
├── experiments/                   # 模型权重（自动生成）
├── scripts/
│   ├── preprocess_all.py          # 批量预处理
│   ├── train.py                   # 单受试者训练
│   ├── evaluate.py                # 测试集评估
│   └── test_pipeline.py           # 功能验证（无需真实数据）
└── src/
    ├── utils/config.py            # 超参数配置
    ├── data/
    │   ├── edf_reader.py          # EDF 读取 + 发作标注解析
    │   ├── preprocessing.py       # 小波滤波 + 滑动窗口
    │   ├── dataset.py             # EEGDataset + DataLoader
    │   └── build_dataset.py       # 预处理主逻辑
    ├── models/
    │   ├── modern_tcn.py          # ModernTCN 模块
    │   ├── channel_attention.py   # SE-Net 通道注意力
    │   ├── bilstm.py              # BiLSTM 分类器
    │   └── seizure_detector.py    # 整体模型
    ├── losses/focal_loss.py       # Focal Loss
    └── training/
        ├── trainer.py             # 训练 / 验证循环
        ├── postprocessing.py      # MAF + Threshold + Collar
        └── metrics.py             # Sensitivity / Specificity / AUC / F1
```
