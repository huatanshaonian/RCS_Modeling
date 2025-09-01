# 🚀 RCS POD Analysis - Modern Streamlit Interface

这是RCS POD分析程序的现代化Web界面，提供更直观和交互式的用户体验。

## ✨ 特性

- 🎨 **现代化界面**: 基于Streamlit的响应式Web界面
- 📊 **实时可视化**: 交互式图表和实时数据展示
- 🔍 **智能监控**: 系统资源和进程实时监控
- 📈 **深度分析**: POD和Autoencoder结果的详细分析
- 💾 **配置管理**: 自动保存和加载分析配置
- 🚦 **状态跟踪**: 实时显示分析进度和日志

## 🛠️ 快速开始

### 1. 启动界面

**方法一: 使用批处理文件 (推荐)**
```bash
# 双击运行
run_streamlit.bat
```

**方法二: 命令行启动**
```bash
streamlit run streamlit_app.py
```

### 2. 访问界面

启动后自动打开浏览器，或手动访问：
```
http://localhost:8501
```

## 📱 界面功能

### 主页面 (streamlit_app.py)
- **配置面板**: 左侧边栏设置所有分析参数
- **运行控制**: 一键开始/停止分析
- **实时监控**: 显示分析进度和系统状态
- **日志显示**: 实时查看运行日志
- **结果预览**: 分析完成后快速查看结果

### 结果分析页面 (📊 Results Analysis)
- **POD能量分析**: 特征值分布、累积能量、模态贡献
- **隐空间分析**: Autoencoder隐空间可视化和统计
- **对比分析**: POD vs Autoencoder性能对比
- **交互式图表**: 基于Plotly的动态可视化

### 实时监控页面 (🔍 Real Time Monitor)
- **系统资源**: CPU、内存、磁盘使用率监控
- **进程监控**: Python进程状态和资源占用
- **GPU状态**: CUDA GPU显存和使用率监控
- **日志监控**: 实时日志文件变化监控

## ⚙️ 配置说明

### 基础配置
- **数据路径**: 参数文件和RCS数据目录
- **输出目录**: 结果保存位置
- **频率选择**: 1.5GHz、3GHz或两者
- **模型数量**: 分析的飞机模型数量
- **训练集大小**: 支持多个训练集大小对比

### 算法配置
- **POD分析**: 启用/禁用传统POD分析
- **Autoencoder分析**: 启用/禁用深度学习分析
- **跳过训练**: 复用已有的Autoencoder模型
- **隐空间维度**: 自定义隐空间维度列表
- **模型类型**: Standard Autoencoder 和 VAE

## 🎯 使用流程

1. **配置参数**: 在左侧边栏设置分析参数
2. **保存配置**: 点击"💾 保存配置"持久化设置
3. **开始分析**: 点击"▶️ 开始分析"按钮
4. **监控进度**: 查看实时日志和系统状态
5. **分析结果**: 在"📊 Results Analysis"页面查看详细结果
6. **系统监控**: 在"🔍 Real Time Monitor"页面监控系统状态

## 📁 文件结构

```
streamlit/
├── streamlit_app.py          # 主应用界面
├── run_streamlit.bat         # Windows启动脚本
├── streamlit_config.json     # 配置文件(自动生成)
└── pages/                    # 页面目录
    ├── 📊_Results_Analysis.py  # 结果分析页面
    └── 🔍_Real_Time_Monitor.py # 实时监控页面
```

## 🔧 高级功能

### 自动配置保存
配置会自动保存到 `streamlit_config.json`，下次启动时自动加载。

### 多页面架构
- 使用Streamlit的多页面功能
- 每个页面独立运行，互不干扰
- 支持页面间数据共享

### 实时更新
- 分析日志实时显示
- 系统资源实时监控
- 自动刷新机制

### 响应式设计
- 适配不同屏幕尺寸
- 移动端友好界面
- 灵活的布局系统

## 🚨 注意事项

1. **端口占用**: 默认使用8501端口，确保端口可用
2. **权限要求**: 监控功能需要足够的系统权限
3. **资源消耗**: GPU监控需要安装PyTorch
4. **浏览器兼容**: 推荐使用Chrome或Edge浏览器

## 🐛 故障排除

### 启动失败
```bash
# 检查Streamlit是否正确安装
pip install streamlit plotly

# 检查端口是否被占用
netstat -ano | findstr :8501

# 使用其他端口启动
streamlit run streamlit_app.py --server.port 8502
```

### 监控页面问题
- GPU监控需要PyTorch: `pip install torch`
- 进程监控需要psutil: `pip install psutil`

### 配置文件问题
删除 `streamlit_config.json` 恢复默认配置。

## 🔮 与传统GUI对比

| 功能 | tkinter GUI | Streamlit GUI | 优势 |
|------|-------------|---------------|------|
| 界面风格 | 传统桌面应用 | 现代Web应用 | ✅ 更美观 |
| 跨平台 | 需要Python环境 | 任何浏览器 | ✅ 更便携 |
| 交互性 | 基础控件 | 丰富组件 | ✅ 更友好 |
| 可视化 | Matplotlib | Plotly交互式 | ✅ 更强大 |
| 扩展性 | 代码复杂 | 组件化 | ✅ 更灵活 |
| 部署 | 本地安装 | Web部署 | ✅ 更方便 |

## 🎉 享受现代化的分析体验！

这个Streamlit界面为您的RCS POD分析工作提供了现代化、直观和强大的用户体验。无论是日常分析还是深度研究，都能满足您的需求。