# RCS POD分析程序 - 完整用户手册

## 🚀 程序概述

本程序是一个综合的雷达散射截面（RCS）数据分析工具，集成了传统的POD（Proper Orthogonal Decomposition）方法和现代的深度学习自编码器方法，用于飞机模型的RCS数据降维分析和性能预测。

### 主要功能
- **POD分析**: 基于奇异值分解的传统降维方法
- **Autoencoder分析**: 基于深度学习的非线性降维方法
- **性能对比**: POD与Autoencoder方法的定量比较
- **参数预测**: 基于设计参数预测RCS性能
- **可视化分析**: 丰富的图表和统计分析

## 📱 图形界面使用指南

### 启动GUI
```bash
python run_gui.py
```

或者直接运行：
```bash
python rcs_gui.py
```

### 界面布局
- **左侧**: 参数控制面板（4个标签页）
- **右侧**: 实时运行日志显示
- **底部**: 运行控制按钮

### 标签页详细说明

#### 1. 基础参数
控制程序的主要运行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 参数文件路径 | `../parameter/parameters_sorted.csv` | 设计参数CSV文件位置 |
| RCS数据目录 | `../parameter/csv_output` | RCS数据文件夹位置 |
| 输出目录 | `./results` | 分析结果保存位置 |
| 分析频率 | `both` | 选择1.5GHz、3GHz或两个都分析 |
| 模型数量 | `100` | 要分析的飞机模型数量 |
| 训练集大小 | `"80"` | 支持单个数值或多个数值如`"60,70,80"` |
| 预测模式 | `False` | 勾选后可以预测新参数的RCS |
| 预测参数文件 | - | 预测模式下的参数文件路径 |

#### 2. POD参数
控制传统POD分析的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 能量覆盖阈值 | `95%` | 选择保留模态的能量覆盖百分比（90%/95%/99%） |
| 可视化模态数量 | `10` | 用于详细分析的模态数量 |
| POD重建模态数 | `0`(自动) | 用于重建的模态数，0表示使用能量阈值确定 |
| POD多模态对比 | `"10,20,30,40"` | 多模态数量对比分析 |

#### 3. 自编码器参数
控制深度学习自编码器的核心参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 潜在空间维度 | `"5,10,15,20"` | 会训练多个不同维度的模型 |
| 模型类型 | `"standard,vae"` | 标准自编码器和/或变分自编码器 |
| 计算设备 | `auto` | auto（自动）、cpu或cuda |
| 训练轮数 | `200` | 深度学习模型的训练次数 |
| 学习率 | `0.001` | 控制模型学习速度 |
| 批次大小 | 自动优化 | 选择自动优化或手动设置 |
| 跳过AE重训练 | `False` | 优先使用已有模型，跳过重训练 |

#### 4. 训练参数
精细控制训练过程的高级参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| L2正则化系数 | `1e-5` | 防止过拟合 |
| VAE KL散度权重 | `1.0` | 变分自编码器的特殊参数 |
| 早停耐心值 | `50` | 验证损失不改善时的等待轮数 |
| 学习率调度耐心值 | `20` | 学习率衰减的等待轮数 |
| 学习率衰减因子 | `0.5` | 学习率衰减倍数 |

### GUI使用流程

#### 基本使用步骤
1. **设置路径**: 在"基础参数"中设置数据文件路径
2. **调整参数**: 根据需要在各标签页调整参数
3. **开始分析**: 点击"开始分析"按钮
4. **监控进度**: 在右侧实时查看运行日志
5. **查看结果**: 分析完成后在输出目录查看结果

#### 高级功能

##### 保存/加载配置
- **保存配置**: 将当前所有参数保存到JSON文件
- **加载配置**: 从JSON文件加载之前保存的参数设置
- 配置文件默认保存为`rcs_gui_config.json`

##### 生成命令行
- 点击"生成命令"查看对应的命令行调用方式
- 支持复制到剪贴板，方便在脚本中使用

##### 预测模式使用
1. 勾选"基础参数"中的"启用预测模式"
2. 选择包含新设计参数的CSV文件
3. 运行分析将基于训练好的模型预测新参数的RCS

## 🔧 命令行使用指南

### 基础运行
```bash
python run.py
```

### 常用命令示例

#### 快速测试
```bash
python run.py --freq 1.5G --num_models 20 --num_train "60"
```

#### 完整分析
```bash
python run.py --freq both --num_models 100 --num_train "60,70,80,90"
```

#### 启用预测模式
```bash
python run.py --predict_mode --param_file ./new_designs.csv --num_train "80"
```

#### 跳过Autoencoder重训练
```bash
python run.py --skip_ae_training --freq 1.5G --num_train 80
```

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--params_path` | str | `../parameter/parameters_sorted.csv` | 设计参数CSV文件路径 |
| `--rcs_dir` | str | `../parameter/csv_output` | RCS数据CSV文件目录 |
| `--output_dir` | str | `./results` | 分析结果输出目录 |
| `--freq` | str | `both` | 分析频率：`1.5G`、`3G`或`both` |
| `--num_models` | int | `100` | 要分析的模型数量 |
| `--num_train` | str | `"80"` | 训练集大小，支持逗号分隔 |
| `--predict_mode` | flag | `False` | 启用预测模式 |
| `--param_file` | str | `None` | 预测参数文件路径 |
| `--latent_dims` | str | `"5,10,15,20"` | 自编码器潜在维度 |
| `--model_types` | str | `"standard,vae"` | 自编码器模型类型 |
| `--skip_ae_training` | flag | `False` | 跳过自编码器重训练 |
| `--ae_epochs` | int | `200` | 自编码器训练轮数 |
| `--ae_device` | str | `auto` | 计算设备 |
| `--ae_learning_rate` | float | `0.001` | 学习率 |
| `--ae_batch_size` | int | `0` | 批次大小（0为自动） |
| `--energy_threshold` | float | `95.0` | POD能量阈值 |
| `--num_modes_visualize` | int | `10` | POD可视化模态数 |
| `--pod_reconstruct_num` | int | `0` | POD重建模态数 |
| `--pod_modes` | str | `"10,20,30,40"` | POD多模态对比 |

## 📊 使用场景与推荐配置

### 场景1: 快速验证
```bash
# 参数设置
模型数量: 20
分析频率: 1.5G
训练集大小: 60
自编码器维度: 5,10
```
- **预计时间**: 5-10分钟
- **内存需求**: 4GB
- **适用**: 初次测试、验证环境

### 场景2: 标准研究
```bash
# 参数设置
模型数量: 100  
分析频率: both
训练集大小: 70,80,90
自编码器维度: 5,10,15,20
```
- **预计时间**: 20-40分钟
- **内存需求**: 8GB
- **适用**: 常规研究分析

### 场景3: 高精度分析
```bash
# 参数设置
模型数量: 100
分析频率: both  
训练集大小: 60,70,80,90
自编码器维度: 10,20,30,40
训练轮数: 500
学习率: 0.0001
```
- **预计时间**: 1-3小时
- **内存需求**: 16GB+
- **适用**: 高精度研究、论文发表

### 场景4: 模型复用
```bash
python run.py --skip_ae_training --freq both --num_train 80
```
- **预计时间**: 10-20分钟
- **内存需求**: 4GB
- **适用**: 利用已训练模型，快速生成结果

## 📂 输出结果详解

### 目录结构
```
results/
├── 1.5GHz/                        # 1.5GHz频率结果
│   └── train_80/                   # 训练集大小80的结果
│       ├── modes/                  # POD模态可视化
│       │   ├── mode_1.png          # 各模态的2D可视化
│       │   └── ...
│       ├── reconstruction_train/   # 训练集重构分析
│       │   ├── comparison_*.png    # 统计对比图
│       │   └── stats_*.csv         # 统计数据
│       ├── reconstruction_test/    # 测试集重构分析
│       ├── autoencoder/           # 自编码器结果
│       │   ├── standard_latent10/ # 标准AE，10维隐空间
│       │   │   ├── autoencoder_model.pth     # 完整模型文件
│       │   │   ├── train_latent_space.npy    # 训练集隐空间数据
│       │   │   ├── scaler.pkl                # 数据标准化器
│       │   │   ├── training_history.png      # 训练历史
│       │   │   └── latent_space.png          # 隐空间可视化
│       │   ├── predictions/        # 预测模型
│       │   │   ├── prediction_models.pkl     # 预测模型
│       │   │   └── parameter_importance.png  # 特征重要性
│       │   └── latent_space_index.json       # 隐空间数据索引
│       ├── regression_models/      # POD回归模型
│       ├── angle_sensitivity/      # 角度敏感性分析
│       └── *.npy                   # 保存的数组数据
└── 3GHz/                          # 3GHz频率结果（类似结构）
```

### 关键输出文件说明

#### POD分析结果
- **`modes/`**: POD模态的2D热图可视化
- **`pod_modes.npy`**: POD模态矩阵数据
- **`lambda_values.npy`**: 特征值数据
- **`energy_analysis.png`**: 能量分布分析图
- **`parameter_sensitivity.png`**: 参数敏感性分析

#### Autoencoder分析结果
- **`autoencoder_model.pth`**: 包含完整结构信息的模型文件
- **`train_latent_space.npy`**: 训练集在隐空间中的表示 [n_samples, latent_dim]
- **`scaler.pkl`**: 数据标准化器，用于数据预处理
- **`training_history.png`**: 训练和验证损失曲线
- **`latent_space.png`**: 隐空间的PCA/t-SNE可视化
- **`parameter_correlation.png`**: 设计参数与隐空间维度的相关性热图

#### 预测系统结果
- **`prediction_models.pkl`**: 从设计参数预测隐空间的回归模型集合
- **`predicted_latent.npy`**: 预测的隐空间表示
- **`parameter_importance.png`**: 各设计参数在预测中的重要性排名

#### 对比分析结果
- **`model_comparison.png`**: 不同模型配置的性能对比图
- **`enhanced_pod_ae_comparison.png`**: POD与Autoencoder的综合对比
- **`reconstruction_methods_comparison.png`**: 重构方法效果对比

## 🚨 故障排除指南

### 常见问题及解决方案

#### 1. 编码错误 ('gbk' codec错误)
**错误症状**:
```
'gbk' codec can't decode byte 0x85 in position 2: illegal multibyte sequence
```

**解决方案**:
```bash
# 方法1: PowerShell启动 (推荐)
python run_gui.py

# 方法2: CMD设置编码
chcp 65001
set PYTHONIOENCODING=utf-8
python run_gui.py

# 方法3: 永久设置环境变量
# 系统设置 → 环境变量 → 新建
# 变量名: PYTHONIOENCODING
# 变量值: utf-8
```

#### 2. GUI无法启动
**原因**: 缺少tkinter模块
**解决方案**:
- **Windows**: 重新安装Python，确保勾选"tcl/tk and IDLE"
- **Linux**: `sudo apt-get install python3-tk`
- **macOS**: `brew install python-tk`

#### 3. 找不到数据文件
**错误症状**:
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**:
1. 使用GUI中的"浏览"按钮选择正确路径
2. 确保路径中没有中文字符（如有问题）
3. 检查文件权限是否可读
4. 验证CSV文件格式正确

#### 4. 内存不足错误
**错误症状**:
```
MemoryError: Unable to allocate array
CUDA out of memory
```

**解决方案**:
1. 减少`num_models`参数（如从100减到20）
2. 降低自编码器批次大小
3. 在GUI中选择CPU计算而非GPU
4. 关闭其他占用内存的程序
5. 分步分析（先分析一个频率）

#### 5. GPU相关错误
**错误症状**:
```
RuntimeError: No CUDA devices available
CUDA out of memory
```

**解决方案**:
1. 在自编码器参数中选择"cpu"设备
2. 或选择"auto"让程序自动降级到CPU
3. 减小批次大小
4. 检查CUDA驱动安装

#### 6. 分析过程中程序卡住
**排查步骤**:
1. 查看GUI右侧日志，确定卡住位置
2. 检查数据文件是否存在且完整
3. 确认有足够的磁盘空间（至少10GB）
4. 检查防火墙/杀毒软件是否阻止

**解决方案**:
- 点击"停止分析"按钮
- 减少模型数量进行测试
- 检查输出目录权限

#### 7. 模型训练失败
**可能原因**:
- 数据中存在NaN或Inf值
- GPU内存不足
- 学习率设置不当

**解决方案**:
1. 检查原始数据质量
2. 降低批次大小或使用CPU
3. 调整学习率（增大到0.01或减小到0.0001）
4. 减少训练轮数进行测试

### 调试技巧

#### 启用详细日志
```bash
set PYTHONVERBOSE=1
python run_gui.py
```

#### 测试基础环境
```bash
python -c "import numpy, pandas, matplotlib, tkinter; print('所有依赖正常')"
```

#### 直接运行核心程序
```bash
python main.py  # 跳过GUI，直接运行分析
```

## ⚡ 性能优化建议

### 硬件配置推荐
| 使用场景 | CPU | 内存 | GPU | 存储 |
|----------|-----|------|-----|------|
| 基础分析 | 4核心 | 8GB | 可选 | 20GB |
| 标准研究 | 8核心 | 16GB | GTX1660+ | 50GB |
| 高精度分析 | 16核心 | 32GB | RTX3080+ | 100GB |

### 参数调优建议

#### 快速测试优化
```bash
--num_models 20 --num_train "60" --latent_dims "5,10"
```

#### 内存优化
```bash
--ae_batch_size 16 --ae_device cpu --num_models 50
```

#### GPU优化
```bash
--ae_device cuda --ae_batch_size 64
```

### 最佳实践

1. **逐步增加复杂度**: 先用少量模型测试，确认无误后再进行完整分析
2. **分频率运行**: 内存不足时可分别分析1.5GHz和3GHz
3. **保存中间结果**: 使用`--skip_ae_training`复用已训练模型
4. **监控系统资源**: 关注CPU、内存、GPU使用情况
5. **定期清理**: 删除不需要的输出文件释放磁盘空间

## 🔬 高级功能

### 自定义模型架构
修改`autoencoder_models.py`中的模型定义：
- 调整网络层数和神经元数量
- 修改激活函数
- 添加批量归一化或dropout层

### 自定义可视化
修改各个visualization文件：
- 调整图表样式和颜色
- 修改字体大小和布局
- 添加新的分析图表

### 数据预处理定制
修改`data_loader.py`：
- 自定义数据过滤条件
- 添加数据增强方法
- 实现自定义标准化策略

## 📈 结果解读指南

### POD分析结果解读
- **能量分布**: 前几个模态包含大部分信息
- **模态形状**: 反映RCS数据的主要变化模式
- **重构误差**: R²值越接近1表示重构质量越好

### Autoencoder结果解读
- **训练曲线**: 下降平稳表示训练良好
- **隐空间可视化**: 聚类表示相似设计的归类
- **参数相关性**: 热图显示哪些参数影响特定隐空间维度

### 性能对比指标
- **MSE (均方误差)**: 越小越好
- **R² (决定系数)**: 越接近1越好，负值表示模型差于平均值
- **压缩比**: 原始维度/隐空间维度，反映压缩效果

## 📚 技术支持与资源

### 技术文档
- `AUTOENCODER_GUIDE.md`: Autoencoder模块技术详解
- `CLAUDE.md`: 项目总体架构说明
- 源代码注释: 各模块详细功能说明

### 获取帮助
如果遇到问题，请收集以下信息：
1. 完整的错误消息或截图
2. Python版本: `python --version`
3. 操作系统版本
4. GUI日志中的最后几行输出
5. 使用的参数配置

### 贡献代码
欢迎提交改进建议和bug报告，请包含：
- 问题的详细描述
- 重现步骤
- 预期结果 vs 实际结果
- 系统环境信息

---

**版本**: 2024年最新版
**适用程序**: RCS POD分析系统
**文档状态**: 持续更新