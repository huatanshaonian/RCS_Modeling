# RCS 数据的POD与模态分析

本项目实现了对雷达散射截面(RCS)数据的本征正交分解(POD)和模态分析，用于研究不同设计参数对RCS特性的影响。

## 项目背景

该项目分析了100个模型在1.5GHz和3GHz频率下的RCS数据。每个模型的RCS数据包括在±45度俯仰和偏航角范围内（91×91=8281个角度组合）的散射特性。这些模型由9个设计参数控制，通过拉丁超立方采样生成。

项目的主要目标是：
1. 对RCS数据进行降维，提取主要模态
2. 分析这些模态与设计参数之间的关系
3. 识别RCS特性对角度变化的敏感区域
4. 验证使用少量模态重构RCS数据的精度

## 文件结构

```
parameter/               # 存储参数和RCS数据的目录
│   ├── parameters_sorted.csv    # 模型设计参数
│   └── csv_output/              # RCS数据文件目录
│       ├── 001_1.5G.csv         # 1号模型在1.5GHz的RCS数据
│       ├── 001_3G.csv           # 1号模型在3GHz的RCS数据
│       └── ...
project_root/
│
├── main.py                  # 主程序，控制整个分析流程
├── data_loader.py           # 数据加载模块，读取参数和RCS数据
├── pod_analysis.py          # POD分解模块，实现POD和能量分析
├── modal_analysis.py        # 模态分析模块，进行模态可视化和敏感性分析
├── requirements.txt         # 项目依赖
├── README.md                # 项目说明文档
│
└── results/                 # 分析结果输出目录（程序自动创建）
    ├── 1.5GHz/              # 1.5GHz频率下的分析结果
    │   ├── energy_analysis.png          # 能量分析图
    │   ├── modes_comparison.png         # 模态对比图
    │   ├── parameter_sensitivity.png    # 参数敏感性分析
    │   ├── pod_modes.npy                # 保存的POD模态
    │   ├── pod_coeffs.npy               # 保存的POD系数
    │   ├── lambda_values.npy            # 保存的特征值
    │   ├── mean_rcs.npy                 # 保存的平均RCS
    │   ├── modes/                       # 各个模态的详细可视化
    │   ├── angle_sensitivity/           # 角度敏感性分析结果
    │   └── reconstruction/              # RCS重构分析结果
    │
    └── 3GHz/                # 3GHz频率下的分析结果 (类似结构)
```

## 功能模块说明

### main.py

主程序，负责控制整个分析流程，包括：
- 加载参数和RCS数据
- 调用POD分析模块进行降维
- 调用模态分析模块进行可视化和敏感性分析
- 保存分析结果

### data_loader.py

数据加载模块，负责：
- 加载设计参数数据（`parameters_sorted.csv`）
- 加载RCS数据（`*_1.5G.csv`和`*_3G.csv`文件）
- 提取角度信息（theta和phi值）

### pod_analysis.py

POD分解模块，实现：
- POD分解算法（使用快照法）
- POD系数计算
- 能量分析（特征值分布与累积能量）

### modal_analysis.py

模态分析模块，进行：
- 主要模态的可视化（2D热图、3D表面图、等值线图）
- 设计参数对模态的敏感性分析
- 角度敏感性分析（梯度分析、敏感区域识别）
- RCS数据重构与验证

## 使用说明

### 环境要求

- Python 3.8或更高版本
- 依赖库：numpy, pandas, matplotlib, scikit-learn, scipy

可通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

### 数据准备

1. 确保设计参数数据文件位于 `../parameter/parameters_sorted.csv`
2. 确保RCS数据文件位于 `../parameter/csv_output/` 目录下
   - 1.5GHz RCS文件格式：`001_1.5G.csv` 到 `100_1.5G.csv`
   - 3GHz RCS文件格式：`001_3G.csv` 到 `100_3G.csv`

### 运行分析

执行主程序：
```bash
python main.py
```

程序将自动：
1. 加载参数和RCS数据
2. 执行POD分解
3. 分析模态能量分布
4. 可视化主要模态
5. 进行参数敏感性分析
6. 进行角度敏感性分析
7. 验证RCS重构精度
8. 将结果保存到`./results/`目录

### 查看结果

分析完成后，可在`./results/`目录下查看：
- `1.5GHz/`和`3GHz/`目录分别包含两个频率下的分析结果
- 各个目录中包含模态可视化、敏感性分析和重构验证结果
- `.npy`文件保存了POD模态、系数和特征值，可用于后续分析

## 结果解读

### 能量分析

`energy_analysis.png`展示了各模态的能量分布和累积能量曲线，帮助确定需要保留的模态数量。

### 模态可视化

`modes/`目录中的模态图像显示了各主要模态的RCS分布特性，可以帮助理解散射机理：
- 第一模态通常代表平均RCS分布
- 后续模态表示主要散射变化模式

### 参数敏感性

`parameter_sensitivity.png`和回归分析结果展示了设计参数与POD系数的关系：
- 相关性矩阵显示参数与模态之间的关联强度
- 参数重要性分析确定对各模态影响最大的设计参数

### 角度敏感性

`angle_sensitivity/`目录中的梯度分析结果显示了RCS对角度变化最敏感的区域：
- 梯度幅值图识别角度敏感区域
- 梯度方向图显示RCS变化的主要方向

### 重构验证

`reconstruction/`目录中的重构结果验证了POD降维的有效性：
- 不同模态数量的重构误差对比
- 代表性模型的原始与重构RCS对比
- 误差分布和剖面比较

## 注意事项

1. RCS数据在分析前会被转换为dB单位（10*log10(RCS)）
2. 设计参数在分析前会被标准化处理
3. 对于大型数据集，程序运行可能需要较长时间和较大内存






# 另外的数学推导
# 基于POD的RCS数据分析流程

## 1. 数据组织

假设有 $M$ 个模型，每个模型的RCS数据在俯仰和偏航角度上均为 $91 \times 91$ 的矩阵。将第 $i$ 个模型的RCS数据表示为 $\mathbf{U}_i \in \mathbb{R}^{91 \times 91}$。

## 2. 数据重组

将每个 $\mathbf{U}_i$ 展平成列向量 $\mathbf{u}_i \in \mathbb{R}^{8281}$，然后将所有模型的数据组合成矩阵 $\mathbf{X} \in \mathbb{R}^{8281 \times M}$：

$$
\mathbf{X} = \begin{bmatrix}
\mathbf{u}_1 & \mathbf{u}_2 & \cdots & \mathbf{u}_M
\end{bmatrix}
$$

## 3. 均值去除

计算均值向量 $\bar{\mathbf{u}}$ 并从每个样本中减去：

$$
\bar{\mathbf{u}} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{u}_i
$$

定义去均值后的矩阵 $\tilde{\mathbf{X}}$：

$$
\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{u}} \mathbf{1}_M^T
$$

其中，$\mathbf{1}_M \in \mathbb{R}^{M}$ 为全1向量。

## 4. 计算协方差矩阵

$$
\mathbf{C} = \frac{1}{M} \tilde{\mathbf{X}} \tilde{\mathbf{X}}^T
$$

## 5. 特征分解

求解协方差矩阵的特征值和特征向量：

$$
\mathbf{C} \mathbf{\Phi} = \mathbf{\Phi} \mathbf{\Lambda}
$$

其中，$\mathbf{\Phi}$ 的列为特征向量，$\mathbf{\Lambda}$ 为对角特征值矩阵。

## 6. POD 模态

POD 模态由特征向量给出，表示为：

$$
\mathbf{\Phi} = \begin{bmatrix}
\phi_1 & \phi_2 & \cdots & \phi_M
\end{bmatrix}
$$

每个模态 $\phi_i$ 可以重塑回 $91 \times 91$ 的矩阵形式，以恢复角度结构。

## 7. 能量分析

特征值 $\lambda_i$ 表示对应模态的能量贡献。累积能量可计算为：

$$
E_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{M} \lambda_i}
$$

选择前 $k$ 个模态，使得累积能量达到预定阈值（如90%），以捕获主要特征。

# Autoencoder Analysis Module

## 概述

本模块实现了基于深度学习自编码器的RCS数据降维分析，提供了标准自编码器(Standard Autoencoder)和变分自编码器(Variational Autoencoder, VAE)两种模型架构，用于学习RCS数据的非线性低维表示。

## 基本原理

### 自编码器(Autoencoder)

自编码器是一种无监督学习的神经网络架构，主要由两部分组成：

1. **编码器(Encoder)**：将高维输入数据映射到低维潜在空间
   ```
   x ∈ R^n → z ∈ R^d (其中 d << n)
   ```

2. **解码器(Decoder)**：将低维潜在表示重构回原始数据空间
   ```
   z ∈ R^d → x' ∈ R^n
   ```

**目标函数**：最小化重构误差
```
L = ||x - x'||²
```

### 变分自编码器(VAE)

VAE在标准自编码器基础上引入了概率建模，假设潜在变量服从先验分布（通常为标准正态分布）。

**损失函数**：
```
L = L_reconstruction + β * L_KL
```

其中：
- `L_reconstruction = ||x - x'||²`：重构损失
- `L_KL = KL(q(z|x) || p(z))`：KL散度正则化项
- `β`：控制重构质量和正则化强度的权衡参数

## 模型架构

### 1. 标准自编码器 (StandardAutoencoder)

```python
class StandardAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        # 编码器：逐步降维
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096), nn.ReLU(),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # 解码器：逐步升维
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 4096), nn.ReLU(),
            nn.Linear(4096, input_dim)
        )
```

### 2. 变分自编码器 (VariationalAutoencoder)

```python
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        # 编码器输出均值和方差
        self.fc_mu = nn.Linear(512, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(512, latent_dim)  # 对数方差
        
    def reparameterize(self, mu, logvar):
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
```

### 3. 增强自编码器 (EnhancedAutoencoder)

包含残差连接的深层网络：

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return F.relu(x + self.block(x))  # 残差连接
```

## 主要功能

### 1. 数据预处理

```python
def perform_autoencoder_analysis():
    # 数据标准化
    scaler = StandardScaler()
    rcs_train_scaled = scaler.fit_transform(rcs_train)
    
    # 转换为PyTorch张量
    rcs_train_tensor = torch.FloatTensor(rcs_train_scaled)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### 2. 模型训练

```python
def train_autoencoder(model, train_loader, val_loader, **kwargs):
    # 优化器配置
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                   patience=20, factor=0.5)
    
    # 混合精度训练（如果支持）
    scaler = torch.cuda.amp.GradScaler()
    
    # 早停机制
    patience = 50
    best_val_loss = float('inf')
```

### 3. 性能评估

```python
# 计算重构误差
mse = mean_squared_error(original, reconstructed)
r2 = r2_score(original.flatten(), reconstructed.flatten())

# 潜在空间分析
latent_representation = model.encode(input_data)
```

## 详细参数设置

### 模型超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latent_dims` | `[5, 10, 15, 20]` | 潜在空间维度候选值 |
| `model_types` | `['standard', 'vae']` | 模型类型 |
| `epochs` | `200` | 训练轮数 |
| `learning_rate` | `1e-3` | 学习率 |
| `batch_size` | `64` | 批次大小 |
| `weight_decay` | `1e-5` | L2正则化系数 |
| `beta` | `1.0` | VAE中KL损失权重 |

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `patience` | `50` | 早停耐心值 |
| `scheduler_patience` | `20` | 学习率调度器耐心值 |
| `scheduler_factor` | `0.5` | 学习率衰减因子 |
| `val_split` | `0.2` | 验证集比例 |
| `use_amp` | `auto` | 是否使用混合精度训练 |

### 数据处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_workers` | `4` | 数据加载线程数 |
| `pin_memory` | `True` | 是否固定内存（GPU训练时） |
| `drop_last` | `True` | 是否丢弃最后不完整批次 |
| `persistent_workers` | `True` | 是否持久化工作线程 |

## 使用方法

### 基本调用

```python
from autoencoder_analysis import perform_autoencoder_analysis

# 执行自编码器分析
autoencoder_results = perform_autoencoder_analysis(
    rcs_data=rcs_data,                    # RCS数据矩阵
    theta_values=theta_values,            # theta角度值
    phi_values=phi_values,                # phi角度值
    param_data=param_data_scaled,         # 设计参数数据
    param_names=param_names,              # 参数名称
    freq_label="1.5G",                    # 频率标签
    output_dir="./results/autoencoder",   # 输出目录
    train_indices=train_indices,          # 训练集索引
    test_indices=test_indices,            # 测试集索引（可选）
    latent_dims=[5, 10, 15, 20],         # 潜在维度列表
    model_types=['standard', 'vae'],      # 模型类型
    device='auto'                         # 计算设备
)
```

### 高级配置

```python
# 自定义训练参数
autoencoder_results = perform_autoencoder_analysis(
    # ... 基本参数 ...
    latent_dims=[8, 16, 32],             # 自定义潜在维度
    model_types=['standard'],             # 只使用标准自编码器
    device='cuda:0',                      # 指定GPU设备
    epochs=300,                           # 增加训练轮数
    learning_rate=5e-4,                   # 降低学习率
    batch_size=128                        # 增大批次大小
)
```

## 输出结果

### 1. 模型文件
- `best_{model_type}_model.pth`：最佳模型权重
- `training_history.png`：训练过程曲线

### 2. 分析图表
- `reconstruction_comparison.png`：重构对比
- `reconstruction_error.png`：重构误差分析
- `error_statistics.png`：误差统计分布
- `latent_space_visualization.png`：潜在空间可视化
- `model_comparison.png`：模型性能对比

### 3. 数据文件
- `model_comparison.csv`：数值结果对比
- `latent_representations.npy`：潜在空间表示
- `reconstruction_errors.npy`：重构误差数据

### 4. 返回字典结构

```python
results = {
    'standard_latent10': {
        'model_type': 'standard',
        'latent_dim': 10,
        'mse': 0.001234,              # 训练集MSE
        'r2': 0.9876,                 # 训练集R²
        'test_mse': 0.001456,         # 测试集MSE（如果有）
        'test_r2': 0.9834,            # 测试集R²（如果有）
        'train_latent': array(...),   # 训练集潜在表示
        'test_latent': array(...),    # 测试集潜在表示（如果有）
        'model_params': 2_451_234,    # 模型参数数量
        'train_losses': [0.1, 0.08, ...],  # 训练损失历史
        'val_losses': [0.12, 0.09, ...]    # 验证损失历史
    },
    # ... 其他配置的结果 ...
}
```

## 与POD方法对比

模块提供了与传统POD方法的直接对比功能：

```python
from autoencoder_analysis import compare_with_pod_results

# POD结果格式
pod_results = {
    'r2': 0.95,
    'mse': 0.001,
    'n_modes': 20,
    'pod_coeffs': pod_coefficients
}

# 执行对比分析
compare_with_pod_results(autoencoder_results, pod_results, output_dir)
```

### 对比内容

1. **重构质量对比**：R²分数和MSE误差
2. **降维程度对比**：维度压缩比较
3. **潜在空间对比**：可视化不同方法的低维表示
4. **计算效率对比**：参数数量和训练时间

## 性能优化

### GPU优化
- 自动检测CUDA可用性
- 混合精度训练减少显存占用
- 批次大小自适应调整
- 数据预取和固定内存

### 内存管理
```python
# 训练完成后清理GPU内存
if device.type == 'cuda':
    torch.cuda.empty_cache()
```

### 早停和学习率调度
```python
# 防止过拟合
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=20, factor=0.5
)

# 早停机制
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

## 错误处理

模块包含完整的错误处理机制：

1. **数据完整性检查**：验证输入数据格式和维度
2. **设备兼容性检查**：自动降级到CPU如果GPU不可用
3. **内存不足处理**：自动减小批次大小
4. **训练异常恢复**：保存检查点支持训练中断恢复

## 扩展性

### 添加新的模型架构

```python
class CustomAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 自定义架构
        
    def forward(self, x):
        # 前向传播逻辑
        return reconstruction, latent_representation
```

### 自定义损失函数

```python
def custom_loss_function(recon_x, x, additional_terms=None):
    # 自定义损失计算
    loss = F.mse_loss(recon_x, x)
    return loss
```

## 依赖要求

```bash
# 核心依赖
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0

# 可选依赖（用于加速）
torchaudio  # 完整PyTorch生态
```

## 安装说明

```bash
# 安装PyTorch（根据系统选择合适版本）
pip install torch torchvision torchaudio

# 或使用conda
conda install pytorch torchvision torchaudio -c pytorch
```

## 注意事项

1. **内存需求**：大规模数据集可能需要16GB+内存
2. **训练时间**：根据数据大小和模型复杂度，训练时间从几分钟到几小时不等
3. **超参数调优**：不同数据集可能需要调整学习率、批次大小等参数
4. **设备选择**：GPU训练速度显著优于CPU，建议使用CUDA兼容的GPU
