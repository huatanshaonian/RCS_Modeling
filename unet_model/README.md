# FiLM-UNet for RCS Prediction

基于MLP+FiLM+U-Net的深度学习架构，用于从飞行器设计参数预测RCS（雷达散射截面）数据。

## 🚀 概述

本项目实现了一个创新的深度学习架构，结合了多层感知机(MLP)编码器、特征调制(FiLM)机制和改进的U-Net网络，用于从9维飞行器设计参数预测91×91的RCS强度图像。

### 主要特性

- **🧠 MLP编码器**: 将9维设计参数编码为1024维潜在特征
- **🎛️ FiLM调制**: 条件特征调制机制，实现参数化的特征变换
- **🔄 Modified U-Net**: 专为RCS预测设计的U-Net架构
- **📊 多尺度损失**: 在23×23、46×46、91×91分辨率上计算损失
- **⚖️ 物理约束**: 集成对称性和边界条件约束
- **🔄 数据增强**: 噪声添加、Mixup、参数抖动等技术
- **📈 完整训练管道**: 早停、学习率调度、检查点保存
- **📊 推理可视化**: 热图、3D图、误差分析等

## 🏗️ 架构详解

### 整体流程
```
9维设计参数 → MLP编码器 → 1024维特征 → 双路径处理器 → {初始特征图, FiLM参数} → Modified U-Net → 91×91 RCS预测
```

### 详细组件

#### 1. MLP编码器
- **架构**: [9 → 128 → 256 → 512 → 1024 → 1024]
- **激活函数**: ReLU + BatchNorm
- **正则化**: Dropout (0.1)

#### 2. 双路径处理器
- **路径A**: 生成32×23×23初始特征图
- **路径B**: 生成6层FiLM调制参数(γ, β)

#### 3. Modified U-Net
- **编码器**: E1(64通道) → E2(128通道) → Bottleneck(256通道)
- **解码器**: D1(128通道) → D2(64通道) → D3(32通道)
- **输出层**: 32 → 16 → 1通道，Tanh激活

#### 4. FiLM调制
- **机制**: output = γ * feature + β
- **应用位置**: 每个卷积块的BatchNorm之后、激活之前

## 📦 安装与环境

### 依赖项
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

### 项目结构
```
unet_model/
├── __init__.py              # 模块初始化
├── mlp_encoder.py           # MLP编码器和双路径处理器
├── film_layer.py            # FiLM调制层和相关组件
├── modified_unet.py         # 改进的U-Net架构
├── film_unet_model.py       # 完整的FiLM-UNet模型
├── custom_losses.py         # 自定义损失函数
├── data_preprocessing.py    # 数据预处理和增强
├── trainer.py               # 训练器和验证逻辑
├── inference.py             # 推理和可视化
├── main.py                  # 主训练脚本
└── README.md               # 本文档
```

## 🚀 快速开始

### 1. 训练模型
```bash
# 基础训练
python main.py --mode train --num_models 100 --epochs 500

# 自定义参数训练
python main.py --mode train \
    --data_dir ../parameter \
    --num_models 80 \
    --frequency 1.5G \
    --batch_size 16 \
    --epochs 300 \
    --learning_rate 1e-3 \
    --enable_augmentation \
    --output_dir ./my_training
```

### 2. 模型评估
```bash
python main.py --mode evaluate \
    --model_path ./outputs/run_20240101_120000/checkpoints/best_model.pth \
    --num_models 50
```

### 3. 推理演示
```bash
# 使用训练好的模型
python main.py --mode inference \
    --model_path ./outputs/run_20240101_120000/checkpoints/best_model.pth

# 随机演示（无训练模型）
python main.py --mode demo
```

### 4. 程序化使用
```python
from unet_model import FiLMUNetModel, RCSPredictor, RCSVisualizer

# 创建模型
model = FiLMUNetModel()

# 预测
predictor = RCSPredictor(model_path='path/to/model.pth')
design_params = np.random.randn(9)  # 9维设计参数
rcs_prediction, info = predictor.predict_single(design_params)

# 可视化
visualizer = RCSVisualizer()
visualizer.plot_rcs_heatmap(rcs_prediction, "RCS预测")
```

## ⚙️ 配置参数

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 16 | 批大小 |
| `--epochs` | 500 | 训练轮数 |
| `--learning_rate` | 1e-3 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--early_stopping_patience` | 50 | 早停耐心值 |

### 损失函数权重
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_mse` | 1.0 | MSE主损失权重 |
| `--lambda_smooth` | 0.01 | TV平滑损失权重 |
| `--lambda_physics` | 0.05 | 物理约束损失权重 |
| `--lambda_multiscale` | 0.1 | 多尺度损失权重 |

### 数据增强参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_augmentation` | False | 启用数据增强 |
| `--noise_std` | 0.01 | 高斯噪声标准差 |
| `--mixup_alpha` | 0.2 | Mixup增强参数 |

## 📊 损失函数设计

组合损失函数包含四个组件：

```python
total_loss = λ₁*L_mse + λ₂*L_smooth + λ₃*L_physics + λ₄*L_multiscale
```

1. **L_mse**: MSE主损失，确保预测准确性
2. **L_smooth**: TV正则化，保证空间平滑性
3. **L_physics**: 物理约束，包括对称性检查
4. **L_multiscale**: 多尺度损失，在不同分辨率上计算

## 📈 性能评估

模型支持多种评估指标：

- **准确性指标**: MSE, MAE, RMSE, R²
- **误差分析**: 最大误差, 相对误差
- **可视化分析**: 热图对比, 误差分布, 参数相关性

## 🎯 使用案例

### 1. 飞行器设计优化
- 输入设计参数，快速预测RCS特性
- 评估不同设计方案的隐身性能
- 指导设计参数调整

### 2. 工程仿真加速
- 替代计算密集的电磁仿真
- 实时RCS预测和可视化
- 批量设计方案评估

### 3. 研究和教学
- 深度学习在电磁学中的应用
- FiLM调制机制研究
- U-Net架构改进

## 🔧 高级功能

### 自定义模型配置
```python
from unet_model import FiLMUNetModel

# 自定义模型参数
model = FiLMUNetModel(
    input_param_dim=9,
    mlp_hidden_dims=[128, 256, 512, 1024],
    feature_map_channels=32,
    output_channels=1
)
```

### 自定义损失函数
```python
from unet_model import CompositeLoss

# 自定义损失权重
loss_fn = CompositeLoss(
    lambda_mse=1.0,
    lambda_smooth=0.02,
    lambda_physics=0.1,
    lambda_multiscale=0.05
)
```

### 数据增强策略
```python
from unet_model import DataAugmenter

# 自定义增强参数
augmenter = DataAugmenter(
    noise_std=0.015,
    mixup_alpha=0.3,
    param_jitter_std=0.02,
    enable_mixup=True
)
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 使用混合精度训练
   - 清理GPU缓存

2. **训练不收敛**
   - 调整学习率
   - 检查数据归一化
   - 增加正则化

3. **数据加载错误**
   - 检查文件路径
   - 验证数据格式
   - 确认编码格式

### 调试技巧

```python
# 启用调试模式
import torch
torch.autograd.set_detect_anomaly(True)

# 检查模型结构
model = FiLMUNetModel()
print(model.get_model_summary())

# 验证数据加载
from unet_model import RCSDataLoader
loader = RCSDataLoader()
params, rcs = loader.load_data(num_models=5, verbose=True)
```

## 📚 参考文献

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- FiLM: Visual Reasoning with a General Conditioning Layer
- Deep Learning for Electromagnetic Scattering Problems

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 👨‍💻 作者

Claude Code Assistant

---

**注意**: 本项目仅用于研究和教育目的。在实际应用中，请结合领域专家知识进行验证和调整。