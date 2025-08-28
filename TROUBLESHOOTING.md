# RCS GUI 故障排除指南

## 🚨 常见问题及解决方案

### 1. 编码错误 ('gbk' codec can't decode)

**错误症状**:
```
'gbk' codec can't decode byte 0x85 in position 2: illegal multibyte sequence
```

**解决方案**:

#### 方法1: 使用批处理文件启动 (推荐)
```cmd
双击运行 run_gui.bat
```

#### 方法2: PowerShell启动
```powershell
python run_gui.py
```

#### 方法3: CMD手动设置编码
```cmd
chcp 65001
set PYTHONIOENCODING=utf-8
python run_gui.py
```

#### 方法4: 永久设置环境变量
1. 右键"此电脑" → "属性" → "高级系统设置"
2. "环境变量" → "系统变量" → "新建"
3. 变量名: `PYTHONIOENCODING`
4. 变量值: `utf-8`

### 2. subprocess RuntimeWarning

**错误症状**:
```
RuntimeWarning: line buffering (buffering=1) isn't supported in binary mode
```

**解决方案**: 
已在代码中修复，如仍出现可忽略该警告，不影响功能。

### 3. tkinter导入错误

**错误症状**:
```
ImportError: No module named 'tkinter'
```

**解决方案**:
- **Windows**: 重新安装Python，确保勾选"tcl/tk and IDLE"
- **Linux**: `sudo apt-get install python3-tk`
- **macOS**: `brew install python-tk`

### 4. GUI界面显示异常

**可能原因**:
- 屏幕分辨率过低
- 系统DPI设置异常

**解决方案**:
```python
# 在 rcs_gui.py 开头添加
import tkinter as tk
root = tk.Tk()
root.tk.call('tk', 'scaling', 1.0)  # 调整DPI缩放
```

### 5. 分析过程中程序卡住

**排查步骤**:
1. 查看GUI右侧日志，确定卡住位置
2. 检查数据文件是否存在且完整
3. 确认有足够的磁盘空间和内存
4. 检查防火墙/杀毒软件是否阻止

**解决方案**:
- 点击"停止分析"按钮
- 减少模型数量进行测试
- 检查输出目录权限

### 6. 找不到数据文件

**错误症状**:
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**:
1. 使用GUI中的"浏览"按钮选择正确路径
2. 确保路径中没有中文字符（如有问题）
3. 检查文件权限是否可读

### 7. 内存不足错误

**错误症状**:
```
MemoryError: Unable to allocate array
```

**解决方案**:
1. 减少`num_models`参数
2. 降低自编码器批次大小
3. 选择CPU计算而非GPU
4. 关闭其他占用内存的程序

### 8. GPU相关错误

**错误症状**:
```
CUDA out of memory
RuntimeError: No CUDA devices available
```

**解决方案**:
1. 在自编码器参数中选择"cpu"设备
2. 或选择"auto"让程序自动降级到CPU
3. 减小批次大小

## 🔍 调试技巧

### 启用详细日志
在程序运行前设置环境变量:
```cmd
set PYTHONVERBOSE=1
python run_gui.py
```

### 测试基础功能
运行测试脚本:
```cmd
python test_encoding.py
```

### 检查依赖
```cmd
python -c "import numpy, pandas, matplotlib, tkinter; print('所有依赖正常')"
```

## 📞 获取帮助

### 收集错误信息
1. 完整的错误消息
2. Python版本: `python --version`
3. 操作系统版本
4. GUI日志中的最后几行输出

### 报告问题时包含
- 使用的命令或操作步骤
- 完整的错误消息截图
- 系统环境信息
- 数据文件大小和格式

## 🛠️ 开发调试

### 直接运行核心程序
```cmd
python main.py
```

### 生成命令行版本
在GUI中点击"生成命令"查看对应的命令行调用。

### 修改调试级别
在相关Python文件开头添加:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**最后更新**: 2024年
**适用版本**: RCS POD GUI v1.0