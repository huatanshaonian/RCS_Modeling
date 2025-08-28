#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCS POD分析程序GUI启动器
检查依赖并启动图形界面
"""

import sys
import os

# 修复TCL/TK版本冲突问题
def fix_tcl_tk_paths():
    """修复TCL/TK环境变量冲突"""
    # 清除冲突的环境变量
    if 'TCL_LIBRARY' in os.environ:
        del os.environ['TCL_LIBRARY']
        print("清除旧的TCL_LIBRARY环境变量")
    
    if 'TK_LIBRARY' in os.environ:
        del os.environ['TK_LIBRARY'] 
        print("清除旧的TK_LIBRARY环境变量")
    
    # 找到正确的TCL/TK路径
    conda_env = os.path.dirname(sys.executable)
    tcl_lib = os.path.join(conda_env, 'lib', 'tcl8.6')
    tk_lib = os.path.join(conda_env, 'lib', 'tk8.6')
    
    # 如果存在正确的路径，则设置环境变量
    if os.path.exists(tcl_lib):
        os.environ['TCL_LIBRARY'] = tcl_lib
        print(f"设置TCL_LIBRARY: {tcl_lib}")
    
    if os.path.exists(tk_lib):
        os.environ['TK_LIBRARY'] = tk_lib
        print(f"设置TK_LIBRARY: {tk_lib}")

# 在导入tkinter之前修复路径
fix_tcl_tk_paths()

def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    if missing_deps:
        print("缺少以下必要依赖:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n请使用以下命令安装:")
        print("pip install numpy pandas matplotlib")
        if "tkinter" in missing_deps:
            print("\n注意: tkinter通常随Python一起安装，如果缺失请重新安装Python")
        return False
    
    return True

def check_files():
    """检查必要的文件"""
    required_files = [
        "rcs_gui.py",
        "run.py", 
        "main.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("缺少以下必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def main():
    """主函数"""
    print("启动RCS POD分析程序图形界面...")
    print("=" * 50)
    
    # 检查依赖
    print("检查依赖...")
    if not check_dependencies():
        sys.exit(1)
    print("依赖检查通过")
    
    # 检查文件
    print("检查必要文件...")
    if not check_files():
        sys.exit(1)
    print("文件检查通过")
    
    # 启动GUI
    print("启动图形界面...")
    try:
        from rcs_gui import main as gui_main
        gui_main()
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except Exception as e:
        print(f"启动GUI时发生错误: {str(e)}")
        print("\n如果遇到编码问题，请尝试:")
        print("1. 在CMD中运行: chcp 65001")
        print("2. 或者使用PowerShell运行")
        print("3. 或者设置环境变量: set PYTHONIOENCODING=utf-8")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()