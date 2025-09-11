#!/usr/bin/env python3
"""
FiLM-UNet GUI启动器
参考run_gui.py的解决方案修复Tcl/Tk环境问题
"""
import os
import sys

def fix_tcl_tk_paths():
    """修复TCL/TK环境变量冲突 - 参考run_gui.py的解决方案"""
    # 清除冲突的环境变量
    if 'TCL_LIBRARY' in os.environ:
        del os.environ['TCL_LIBRARY']
        print("清除旧的TCL_LIBRARY环境变量")
    
    if 'TK_LIBRARY' in os.environ:
        del os.environ['TK_LIBRARY'] 
        print("清除旧的TK_LIBRARY环境变量")
    
    # 找到正确的TCL/TK路径 - 使用sys.executable而非硬编码路径
    python_env = os.path.dirname(sys.executable)
    tcl_lib = os.path.join(python_env, 'lib', 'tcl8.6')
    tk_lib = os.path.join(python_env, 'lib', 'tk8.6')
    
    # 如果存在正确的路径，则设置环境变量
    if os.path.exists(tcl_lib):
        os.environ['TCL_LIBRARY'] = tcl_lib
        print(f"设置TCL_LIBRARY: {tcl_lib}")
    
    if os.path.exists(tk_lib):
        os.environ['TK_LIBRARY'] = tk_lib
        print(f"设置TK_LIBRARY: {tk_lib}")

# 在导入tkinter之前修复路径
fix_tcl_tk_paths()

def main():
    """Main launcher function"""
    print("启动FiLM-UNet训练GUI...")
    print("=" * 50)
    
    # Add unet_model to Python path
    unet_model_path = os.path.join(os.path.dirname(__file__), 'unet_model')
    if unet_model_path not in sys.path:
        sys.path.append(unet_model_path)
    
    try:
        import tkinter as tk
        print("Tkinter导入成功")
        
        from unet_gui import UNetTrainingGUI
        print("UNetTrainingGUI导入成功")
        
        # Create and run GUI
        root = tk.Tk()
        gui = UNetTrainingGUI(root)
        print("GUI创建成功，启动界面...")
        root.mainloop()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了必要的依赖:")
        print("- tkinter (通常随Python一起安装)")
        print("- 所有UNet模型相关的包")
        
    except Exception as e:
        print(f"启动GUI时发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("GUI会话结束")

if __name__ == "__main__":
    main()