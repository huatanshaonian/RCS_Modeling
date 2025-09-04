# CLAUDE.md

本文件为Claude Code (claude.ai/code)在处理此代码库时提供指导。

## 项目概述

这是一个雷达截面积(RCS)分析项目，实现了本征正交分解(POD)和深度学习自编码器方法用于RCS数据的降维分析。项目分析100个飞行器模型在1.5GHz和3GHz频率下的数据，每个模型在8281个角度组合(91×91俯仰角/方位角)下的RCS数据。

## 运行命令

### 运行分析
- **主要分析**: `python run.py` (带参数的命令行界面)
- **直接执行**: `python main.py` (使用默认参数)
- **传统GUI**: `python run_gui_fixed.bat` 或 `python run_gui_fixed.ps1` (修复编码和环境问题)
- **现代Web界面**: `python run_streamlit.bat` 或通过Streamlit运行
- **测试CUDA可用性**: `python check_cuda.py`

### 命令行参数 (run.py)
- `--params_path`: 设计参数CSV文件路径 (默认: ../parameter/parameters_sorted.csv)
- `--rcs_dir`: RCS数据目录 (默认: ../parameter/csv_output)
- `--output_dir`: 输出目录 (默认: ./results)
- `--freq`: 要分析的频率 (1.5G, 3G, 或 both)
- `--num_models`: 分析的模型数量
- `--num_train`: 训练集大小，逗号分隔 (如: "60,70,80")
- `--predict_mode`: 启用预测模式
- `--param_file`: 预测用参数文件
- `--pod_modes`: POD模态数量，逗号分隔 (如: "10,20,30,40")

### 依赖安装
安装必需包: `pip install -r requirements.txt`
- 核心依赖: numpy, pandas, matplotlib, scikit-learn, scipy
- 可选依赖: torch, torchvision, torchaudio (用于自编码器分析)
- Web界面: streamlit, plotly (用于现代化界面)

## 系统架构

### 数据流程
1. **数据加载** (`data_loader.py`): 从CSV文件加载设计参数和RCS数据，支持多种编码格式
2. **POD分析** (`pod_analysis.py`): 执行奇异值分解进行降维
3. **模态分析** (`model_analysis.py`): 可视化模态、参数敏感性和重构质量
4. **自编码器分析** (可选): 使用PyTorch进行基于深度学习的降维

### 核心组件

**主控制流程** (`main.py`, `run.py`):
- 协调整个分析管道
- 处理多种训练集大小和频率
- 管理数据分割和结果组织

**POD实现** (`pod_analysis.py`):
- 使用SVD方法保证数值稳定性
- 包含全面的错误处理和诊断
- 实现能量分析确定最优模态数量
- 核心函数: `perform_pod()` 返回模态、特征值和平均数据

**自编码器模块** (模块化结构):
- `autoencoder_analysis.py`: 主协调器
- `autoencoder_models.py`: StandardAutoencoder和VariationalAutoencoder类
- `autoencoder_training.py`: 带早停机制的训练循环
- `autoencoder_utils.py`: 设备管理和数据工具
- `autoencoder_visualization.py`: 绘图和对比功能
- `autoencoder_prediction.py`: 基于自编码器的预测功能

**用户界面系统**:
- `rcs_gui.py`: 传统tkinter GUI界面，全功能参数配置
- `streamlit_app.py`: 现代Web界面，实时日志监控和进程管理
- `run_gui_fixed.bat/.ps1`: 修复环境变量的GUI启动脚本
- `run_streamlit.bat`: Streamlit Web应用启动脚本

### 数据结构

**输入数据**:
- 参数: `parameters_sorted.csv` (每个模型24个设计参数，支持GBK编码)
- RCS数据: `{模型编号}_{频率}.csv` (91×91角度组合，支持多种编码)

**输出结构**:
```
results/
├── {frequency}/              # 1.5GHz 或 3GHz
│   └── train_{size}/         # 不同训练集大小
│       ├── modes/            # POD模态可视化
│       ├── reconstruction_train/  # 训练集重构分析
│       ├── reconstruction_test/   # 测试集重构分析
│       ├── autoencoder/      # 深度学习结果
│       │   ├── {config}/     # 不同配置的结果
│       │   └── comparison/   # 对比分析
│       └── *.npy            # 保存的数组(模态、系数等)
```

### 错误处理策略

代码库实现了健壮的错误处理:
- **数据验证**: 检查RCS数据中的NaN/Inf值
- **编码处理**: 多重编码格式尝试(UTF-8, GBK, GB2312, Latin1, CP1252)
- **数值稳定性**: POD的SVD回退，病态矩阵的正则化
- **GPU管理**: CUDA不可用时自动CPU回退
- **内存管理**: 批处理大小优化和内存清理
- **环境兼容**: Tcl/Tk环境变量修复，Windows路径处理

### 关键函数

**数据处理**:
- `load_rcs_data()`: 加载和验证RCS CSV文件，多编码支持
- `load_parameters()`: 加载参数文件，自动编码检测
- `perform_pod()`: 主POD分解，带诊断功能
- `energy_analysis()`: 确定最优模态数量

**分析函数**:
- `analyze_frequency_data()`: 每个频率的主分析循环
- `parameter_sensitivity()`: 参数与POD系数的回归分析
- `reconstruct_rcs()`: 验证重构质量

**模型训练** (PyTorch可用时):
- `train_autoencoder()`: 带早停和学习率调度的训练
- `perform_autoencoder_analysis()`: 主自编码器分析协调器
- `create_autoencoder_prediction_pipeline()`: 创建预测流水线

**界面管理**:
- `streamlit_app.py`: 多页面Web应用，实时监控
- `rcs_gui.py`: 传统GUI，完整参数配置

### 特殊考虑事项

- **中文文本支持**: 使用SimHei字体显示matplotlib中文字符
- **大数据处理**: 对8281维RCS数据实现内存高效处理
- **GPU优化**: 可用时自动设备选择和混合精度训练
- **可重现性**: 使用固定随机种子(42)保证运行结果一致
- **模块化设计**: 自编码器组件可独立工作，PyTorch不可用时可禁用
- **编码兼容**: 自动检测和处理中文CSV文件的不同编码格式
- **环境修复**: 自动处理Tcl/Tk版本冲突和路径问题

## 开发说明

- 项目同时处理POD(传统方法)和自编码器(深度学习)方法进行对比
- 结果保存为多种格式(PNG图像、NPY数组、CSV统计)进行全面分析
- 代码库在缺少可选依赖时能够优雅降级
- 支持两种界面：传统GUI(tkinter)和现代Web界面(Streamlit)
- 实现了完整的预测流水线，从设计参数预测RCS数据
- 包含详细的统计分析和可视化功能

## 常见问题解决

**编码错误**: 使用修复的`data_loader.py`，自动尝试多种编码格式
**GUI启动失败**: 使用`run_gui_fixed.bat`或`run_gui_fixed.ps1`设置正确的环境变量
**Web界面**: 使用`run_streamlit.bat`启动现代化Web界面
**CUDA问题**: 运行`python check_cuda.py`检查GPU可用性

## 分支策略

- **main**: 主要开发分支，稳定功能
- **streamlit**: Web界面开发分支，UI美化和功能增强
- **dev**: 实验性功能开发