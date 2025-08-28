# RCS POD分析程序使用手册

本文档详细介绍RCS POD分析程序的所有控制参数和使用方法。

## 📊 参数总览

程序共有**21个主要控制参数**，分为4个功能类别：
- 命令行参数（8个）
- POD分析参数（2个）  
- 自编码器核心参数（6个）
- 自编码器训练参数（5个）

## 🚀 快速开始

### 基础运行
```bash
python run.py
```

### 自定义频率和训练集
```bash
python run.py --freq 1.5G --num_train "70,80,90"
```

### 完整配置示例
```bash
python run.py \
  --params_path ./data/parameters.csv \
  --rcs_dir ./data/rcs \
  --output_dir ./results \
  --freq both \
  --num_models 100 \
  --num_train "60,70,80" \
  --predict_mode \
  --param_file ./predict_params.csv
```

## 📋 详细参数说明

### 1. 命令行参数 (8个)

通过`run.py`命令行直接控制的参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--params_path` | str | `../parameter/parameters_sorted.csv` | 设计参数CSV文件路径 |
| `--rcs_dir` | str | `../parameter/csv_output` | RCS数据CSV文件目录 |
| `--output_dir` | str | `./results` | 分析结果输出目录 |
| `--freq` | str | `both` | 分析频率：`1.5G`、`3G`或`both` |
| `--num_models` | int | `100` | 要分析的模型数量 |
| `--num_train` | str | `"80"` | 训练集大小，支持逗号分隔列表如`"60,70,80"` |
| `--predict_mode` | flag | `False` | 启用预测模式（无参数，使用时直接加标志） |
| `--param_file` | str | `None` | 预测模式下的参数文件路径 |

#### 使用示例：
```bash
# 分析1.5GHz频率，使用多个训练集大小
python run.py --freq 1.5G --num_train "70,80,90"

# 启用预测模式
python run.py --predict_mode --param_file ./new_params.csv

# 自定义输出目录
python run.py --output_dir ./my_analysis_results
```

### 2. POD分析参数 (2个)

控制POD分解和模态分析的参数：

| 参数 | 默认设置 | 说明 |
|------|----------|------|
| **保留模态数量** | 基于95%能量自动选择 | 程序自动计算覆盖95%能量所需的模态数 |
| **可视化模态数量** | `min(10, 总模态数)` | 前10个模态用于详细可视化分析 |

#### 修改方式：
在`main.py`文件中修改：
```python
# 第349-351行附近
r = max(1, min(modes_95, phi_modes_train.shape[1] - 1))  # 95%能量
print(f"选择保留前 {r} 个模态，覆盖95%能量")

# 第379行附近  
num_modes_to_visualize = min(10, r)  # 可视化前10个模态
```

### 3. 自编码器核心参数 (6个)

控制深度学习自编码器的核心参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latent_dims` | `[5, 10, 15, 20]` | 潜在空间维度列表，程序会训练每个维度的模型 |
| `model_types` | `['standard', 'vae']` | 模型类型：`standard`（标准AE）、`vae`（变分AE） |
| `device` | `'auto'` | 计算设备：`auto`（自动选择）、`cpu`、`cuda` |
| `epochs` | `200` | 训练轮数 |
| `learning_rate` | `1e-3` | 学习率 |
| `batch_size` | 自动优化 | 根据GPU显存自动调整批次大小 |

#### 修改方式：
在`main.py`文件第502-504行附近修改：
```python
autoencoder_results = perform_autoencoder_analysis(
    # ... 其他参数 ...
    latent_dims=[8, 16, 32],           # 自定义潜在维度
    model_types=['standard'],          # 只使用标准自编码器
    device='cuda:0',                   # 指定特定GPU
    epochs=300,                        # 增加训练轮数
    learning_rate=5e-4,               # 降低学习率
    # ...
)
```

### 4. 自编码器训练参数 (5个)

控制训练过程的高级参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weight_decay` | `1e-5` | L2正则化系数，防止过拟合 |
| `beta` | `1.0` | VAE中KL散度损失的权重系数 |
| `patience` | `50` | 早停耐心值（验证损失不改善的最大轮数） |
| `scheduler_patience` | `20` | 学习率调度器耐心值 |
| `scheduler_factor` | `0.5` | 学习率衰减因子 |

#### 修改方式：
在`autoencoder_training.py`文件中修改：
```python
# 第46-47行附近
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

# 第54行附近
patience = 50
```

## 🎯 使用场景与参数组合

### 场景1: 快速测试
```bash
python run.py --freq 1.5G --num_models 20 --num_train "60"
```
- 只分析1.5GHz频率
- 使用20个模型快速测试
- 单一训练集大小

### 场景2: 完整分析
```bash
python run.py --freq both --num_models 100 --num_train "60,70,80,90"
```
- 分析两个频率
- 使用全部100个模型
- 多个训练集大小对比

### 场景3: 高精度深度学习
修改`main.py`中的自编码器参数：
```python
latent_dims=[10, 20, 30, 40],      # 更多潜在维度
epochs=500,                         # 更多训练轮数
learning_rate=1e-4,                # 更小学习率
```

### 场景4: 预测新参数
```bash
python run.py --predict_mode --param_file ./new_designs.csv --num_train "80"
```
- 启用预测模式
- 提供新设计参数文件
- 基于训练集大小80的模型进行预测

## 📂 输出结果结构

程序会在`output_dir`目录下生成以下结构：
```
results/
├── 1.5GHz/                    # 1.5GHz频率结果
│   └── train_80/               # 训练集大小80的结果
│       ├── modes/              # POD模态可视化
│       ├── reconstruction_train/ # 训练集重构分析
│       ├── test/               # 测试集分析（如果有）
│       ├── autoencoder/        # 自编码器结果
│       └── *.npy               # 保存的数组数据
└── 3GHz/                      # 3GHz频率结果（类似结构）
```

## 🔧 高级配置

### 自定义自编码器架构
修改`autoencoder_models.py`中的模型定义，可以：
- 调整网络层数和神经元数量
- 修改激活函数
- 添加批量归一化或dropout层

### 自定义POD参数
修改`pod_analysis.py`中的参数：
- 调整数值稳定性参数
- 修改能量阈值（90%、95%、99%）
- 自定义正则化因子

### 内存和性能优化
- `batch_size`：GPU显存不足时会自动减小
- `num_workers`：数据加载的并行进程数
- `pin_memory`：是否将数据固定在内存中加速GPU传输

## ⚠️ 注意事项

1. **数据路径**：确保`params_path`和`rcs_dir`路径正确且数据完整
2. **GPU内存**：大模型或大批次可能导致GPU内存不足，程序会自动降级
3. **训练时间**：自编码器训练可能需要几分钟到几小时，取决于数据大小和模型复杂度
4. **预测模式**：需要先完成正常分析生成模型，才能用于预测新参数

## 🚨 常见问题

### Q: 如何禁用自编码器分析？
A: 删除或注释`main.py`中第476-527行的自编码器分析代码块。

### Q: 如何只分析特定模型编号？
A: 修改`data_loader.py`中的`load_rcs_data`函数，添加模型过滤逻辑。

### Q: 如何调整图表的字体大小？
A: 修改各个可视化文件中`plt.rcParams['font.size']`参数。

### Q: 内存不足怎么办？
A: 减少`num_models`参数或`batch_size`，或升级硬件配置。

## 📊 性能基准

| 配置 | 预估时间 | 内存需求 |
|------|----------|----------|
| 基础配置（20模型） | 5-10分钟 | 4GB |
| 标准配置（100模型） | 20-40分钟 | 8GB |
| 完整配置+自编码器 | 1-3小时 | 16GB+ |

---

*最后更新：2024年*