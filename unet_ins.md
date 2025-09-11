请实现一个基于MLP+FiLM+U-Net的深度学习架构，用于从飞行器设计参数预测RCS（雷达散射截面）数据。

## 任务描述
- 输入：9个飞行器设计参数（浮点数向量）
- 输出：91×91的RCS强度图像
- 数据集：100个样本（需要数据增强）

## 架构要求

### 1. MLP编码器
- 输入：9维设计参数向量
- 结构：5层全连接网络 [9 → 128 → 256 → 512 → 1024 → 1024]
- 激活函数：ReLU + BatchNorm
- Dropout：0.1（防止过拟合）
- 输出：1024维潜在特征向量

### 2. 双路径处理
MLP的输出需要分成两路：

**路径A - 初始特征图生成**：
- 输入：1024维特征
- 通过全连接层映射到 32×32×23 = 23552维
- Reshape为 【B,23, 23, 32] 作为U-Net的输入特征图

**路径B - FiLM参数生成**：
- 输入：1024维特征  
- 为U-Net的每一层生成调制参数(γ, β)
- 需要6组FiLM参数（对应U-Net的6个卷积块）

## Modified U-Net 详细层级结构

### 编码器
输入: [B, 32, 23, 23] (来自MLP)

**Layer E1**:
- Conv2d(32, 64, kernel=3, padding=1, stride=1)
- FiLM调制(γ₁, β₁)
- ReLU
- Conv2d(64, 64, kernel=3, padding=1, stride=1)
- BatchNorm2d(64)
- ReLU
- 输出: [B, 64, 23, 23] → skip1

**Layer E2**:
- Conv2d(64, 128, kernel=3, padding=1, stride=1)
- FiLM调制(γ₂, β₂)
- ReLU
- Conv2d(128, 128, kernel=3, padding=1, stride=1)
- BatchNorm2d(128)
- ReLU
- MaxPool2d(kernel=2, stride=2)
- 输出: [B, 128, 11, 11] → skip2

**Bottleneck**:
- Conv2d(128, 256, kernel=3, padding=1, stride=1)
- FiLM调制(γ₃, β₃)
- ReLU
- Conv2d(256, 256, kernel=3, padding=1, stride=1)
- BatchNorm2d(256)
- ReLU
- 输出: [B, 256, 11, 11]

### 解码器

**Layer D1**:
- Upsample(scale_factor=2, mode='bilinear') → [B, 256, 22, 22]
- Pad(1) → [B, 256, 23, 23]
- Concatenate(skip2上采样) → [B, 256+128, 23, 23]
- Conv2d(384, 128, kernel=3, padding=1, stride=1)
- FiLM调制(γ₄, β₄)
- ReLU
- Conv2d(128, 128, kernel=3, padding=1, stride=1)
- BatchNorm2d(128)
- ReLU
- 输出: [B, 128, 23, 23]

**Layer D2**:
- Upsample(scale_factor=2, mode='bilinear') → [B, 128, 46, 46]
- Concatenate(skip1上采样到46×46) → [B, 128+64, 46, 46]
- Conv2d(192, 64, kernel=3, padding=1, stride=1)
- FiLM调制(γ₅, β₅)
- ReLU
- Conv2d(64, 64, kernel=3, padding=1, stride=1)
- BatchNorm2d(64)
- ReLU
- 输出: [B, 64, 46, 46]

**Layer D3**:
- Upsample(scale_factor=2, mode='bilinear') → [B, 64, 92, 92]
- CenterCrop(91, 91) → [B, 64, 91, 91]
- Conv2d(64, 32, kernel=3, padding=1, stride=1)
- FiLM调制(γ₆, β₆)
- ReLU
- Conv2d(32, 32, kernel=3, padding=1, stride=1)
- BatchNorm2d(32)
- ReLU
- 输出: [B, 32, 91, 91]

**输出层**:
- Conv2d(32, 16, kernel=3, padding=1, stride=1)
- ReLU
- Conv2d(16, 1, kernel=1, stride=1)
- Tanh
- 输出: [B, 1, 91, 91]


### 4. FiLM调制机制
- 每个FiLM模块执行：output = γ * feature + β
- γ和β的维度匹配对应层的通道数
- FiLM调制在BatchNorm之后、激活函数之前应用

### 5. 损失函数设计
```python
total_loss = λ₁*L_mse + λ₂*L_smooth + λ₃*L_physics + λ₄*L_multiscale

其中：
- L_mse: MSE主损失（λ₁=1.0）
- L_smooth: TV正则化，确保空间平滑（λ₂=0.01）
- L_physics: 物理约束（对称性检查）（λ₃=0.05）
- L_multiscale: 多尺度损失（λ₄=0.1）
  - 在46×46分辨率计算中间损失
  - 在23×23分辨率计算粗略损失
```
### 6. 训练配置
**跳跃连接处理**：
- Conv1的skip需要上采样：23×23 → 46×46（bilinear）
- Conv2的skip直接使用：23×23
- 使用concatenation而非addition

**数据增强**：
- 参数添加高斯噪声（std=0.01）
- Mixup增强（alpha=0.2）
- 随机参数插值

**训练策略**：
- 优化器：AdamW (lr=1e-3, weight_decay=1e-4)
- 学习率调度：CosineAnnealingLR
- 批大小：16
- 训练轮数：500

## 实现要求
1. 使用PyTorch实现
2. 模块化设计，每个组件独立的类
3. 包含数据预处理（归一化到[-1,1]）
4. 实现训练循环和验证
5. 添加模型保存和加载功能
6. 包含推理演示代码

## 注意事项
- FiLM参数初始化：γ初始化为1，β初始化为0（保持初始恒等变换）
- U-Net跳跃连接：使用concatenation而非addition
- 上采样使用双线性插值+卷积，避免棋盘效应
- RCS数据使用对数尺度（dB）进行训练

请实现这个完整的架构，重点确保MLP、FiLM和U-Net三个组件正确连接和信息流动。