#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FiLM-UNet训练GUI
简单的tkinter界面用于配置和启动训练
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import subprocess
import threading
import json
from pathlib import Path

class UNetTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FiLM-UNet RCS预测模型训练")
        self.root.geometry("800x700")
        
        # 配置变量
        self.config = {
            'data_dir': tk.StringVar(value='../parameter'),
            'params_file': tk.StringVar(value='parameters_sorted.csv'),
            'rcs_dir': tk.StringVar(value='csv_output'),
            'output_dir': tk.StringVar(value='./unet_outputs'),
            'num_models': tk.IntVar(value=100),
            'train_size': tk.DoubleVar(value=0.8),
            'frequency': tk.StringVar(value='1.5G'),
            'batch_size': tk.IntVar(value=16),
            'epochs': tk.IntVar(value=300),
            'learning_rate': tk.DoubleVar(value=0.001),
            'enable_augmentation': tk.BooleanVar(value=True),
            'device': tk.StringVar(value='auto')
        }
        
        self.training_process = None
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # 标题
        title_label = ttk.Label(main_frame, text="FiLM-UNet RCS预测模型训练", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # 数据设置部分
        data_frame = ttk.LabelFrame(main_frame, text="数据设置", padding="10")
        data_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        data_frame.columnconfigure(1, weight=1)
        row += 1
        
        # 数据目录
        ttk.Label(data_frame, text="数据目录:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(data_frame, textvariable=self.config['data_dir'], width=40).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(data_frame, text="浏览", 
                  command=lambda: self.browse_directory(self.config['data_dir'])).grid(row=0, column=2, padx=(10, 0))
        
        # 参数文件
        ttk.Label(data_frame, text="参数文件:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Entry(data_frame, textvariable=self.config['params_file']).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # RCS数据目录
        ttk.Label(data_frame, text="RCS目录:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Entry(data_frame, textvariable=self.config['rcs_dir']).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # 输出目录
        ttk.Label(data_frame, text="输出目录:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Entry(data_frame, textvariable=self.config['output_dir'], width=40).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        ttk.Button(data_frame, text="浏览", 
                  command=lambda: self.browse_directory(self.config['output_dir'])).grid(row=3, column=2, padx=(10, 0), pady=(5, 0))
        
        # 数据集设置
        dataset_frame = ttk.LabelFrame(main_frame, text="数据集设置", padding="10")
        dataset_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        dataset_frame.columnconfigure(1, weight=1)
        dataset_frame.columnconfigure(3, weight=1)
        row += 1
        
        # 模型数量和训练集比例
        ttk.Label(dataset_frame, text="模型数量:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Spinbox(dataset_frame, from_=10, to=200, textvariable=self.config['num_models'], width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(dataset_frame, text="训练集比例:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Scale(dataset_frame, from_=0.5, to=0.9, orient=tk.HORIZONTAL, variable=self.config['train_size'],
                 length=150).grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        # 频率选择
        ttk.Label(dataset_frame, text="频率:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        freq_combo = ttk.Combobox(dataset_frame, textvariable=self.config['frequency'], 
                                 values=['1.5G', '3G'], state='readonly', width=10)
        freq_combo.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # 训练参数设置
        train_frame = ttk.LabelFrame(main_frame, text="训练参数", padding="10")
        train_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        train_frame.columnconfigure(1, weight=1)
        train_frame.columnconfigure(3, weight=1)
        row += 1
        
        # 批大小和轮数
        ttk.Label(train_frame, text="批大小:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Spinbox(train_frame, from_=4, to=64, textvariable=self.config['batch_size'], width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(train_frame, text="训练轮数:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Spinbox(train_frame, from_=50, to=1000, textvariable=self.config['epochs'], width=10).grid(row=0, column=3, sticky=tk.W)
        
        # 学习率和设备
        ttk.Label(train_frame, text="学习率:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Entry(train_frame, textvariable=self.config['learning_rate'], width=12).grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(train_frame, text="设备:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(5, 0))
        device_combo = ttk.Combobox(train_frame, textvariable=self.config['device'], 
                                   values=['auto', 'cpu', 'cuda'], state='readonly', width=10)
        device_combo.grid(row=1, column=3, sticky=tk.W, pady=(5, 0))
        
        # 选项设置
        options_frame = ttk.LabelFrame(main_frame, text="其他选项", padding="10")
        options_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1
        
        ttk.Checkbutton(options_frame, text="启用数据增强", variable=self.config['enable_augmentation']).grid(row=0, column=0, sticky=tk.W)
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(10, 0))
        row += 1
        
        ttk.Button(button_frame, text="开始训练", command=self.start_training).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="停止训练", command=self.stop_training).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="加载配置", command=self.load_config).pack(side=tk.LEFT)
        
        # 日志显示
        log_frame = ttk.LabelFrame(main_frame, text="训练日志", padding="10")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 进度条
        self.progress_var = tk.StringVar(value="准备就绪")
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=row+1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Label(progress_frame, textvariable=self.progress_var).grid(row=0, column=1)
        
        # 初始化日志
        self.log("FiLM-UNet训练界面已启动")
        self.log(f"当前工作目录: {os.getcwd()}")
        
    def browse_directory(self, var):
        """浏览目录"""
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)
    
    def log(self, message):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def validate_config(self):
        """验证配置"""
        data_dir = self.config['data_dir'].get()
        if not os.path.exists(data_dir):
            messagebox.showerror("错误", f"数据目录不存在: {data_dir}")
            return False
        
        params_path = os.path.join(data_dir, self.config['params_file'].get())
        if not os.path.exists(params_path):
            messagebox.showerror("错误", f"参数文件不存在: {params_path}")
            return False
        
        rcs_path = os.path.join(data_dir, self.config['rcs_dir'].get())
        if not os.path.exists(rcs_path):
            messagebox.showerror("错误", f"RCS数据目录不存在: {rcs_path}")
            return False
        
        return True
    
    def build_command(self):
        """构建训练命令"""
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), 'main.py'),
            '--mode', 'train',
            '--data_dir', self.config['data_dir'].get(),
            '--params_file', self.config['params_file'].get(),
            '--rcs_dir', self.config['rcs_dir'].get(),
            '--output_dir', self.config['output_dir'].get(),
            '--num_models', str(self.config['num_models'].get()),
            '--test_size', str(1.0 - self.config['train_size'].get()),
            '--frequency', self.config['frequency'].get(),
            '--batch_size', str(self.config['batch_size'].get()),
            '--epochs', str(self.config['epochs'].get()),
            '--learning_rate', str(self.config['learning_rate'].get()),
            '--device', self.config['device'].get()
        ]
        
        if self.config['enable_augmentation'].get():
            cmd.append('--enable_augmentation')
        
        return cmd
    
    def start_training(self):
        """开始训练"""
        if not self.validate_config():
            return
        
        if self.training_process and self.training_process.poll() is None:
            messagebox.showwarning("警告", "训练正在进行中")
            return
        
        # 创建输出目录
        output_dir = self.config['output_dir'].get()
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建命令
        cmd = self.build_command()
        self.log(f"执行命令: {' '.join(cmd)}")
        
        # 启动训练进程
        try:
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.progress_bar.start()
            self.progress_var.set("训练中...")
            
            # 启动日志读取线程
            threading.Thread(target=self.read_training_output, daemon=True).start()
            
            self.log("训练已启动")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动训练失败: {e}")
            self.log(f"启动失败: {e}")
    
    def read_training_output(self):
        """读取训练输出"""
        try:
            while self.training_process.poll() is None:
                line = self.training_process.stdout.readline()
                if line:
                    self.root.after(0, lambda: self.log(line.strip()))
            
            # 读取剩余输出
            remaining_output = self.training_process.stdout.read()
            if remaining_output:
                for line in remaining_output.split('\n'):
                    if line.strip():
                        self.root.after(0, lambda l=line: self.log(l.strip()))
            
            # 训练结束
            return_code = self.training_process.returncode
            self.root.after(0, lambda: self.training_finished(return_code))
            
        except Exception as e:
            self.root.after(0, lambda: self.log(f"读取输出错误: {e}"))
    
    def training_finished(self, return_code):
        """训练完成"""
        self.progress_bar.stop()
        
        if return_code == 0:
            self.progress_var.set("训练完成")
            self.log("训练成功完成!")
            messagebox.showinfo("成功", "训练完成！")
        else:
            self.progress_var.set("训练失败")
            self.log(f"训练失败，返回码: {return_code}")
            messagebox.showerror("错误", f"训练失败，返回码: {return_code}")
    
    def stop_training(self):
        """停止训练"""
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            self.progress_bar.stop()
            self.progress_var.set("训练已停止")
            self.log("训练被用户停止")
        else:
            messagebox.showinfo("信息", "没有正在运行的训练")
    
    def save_config(self):
        """保存配置"""
        config_data = {}
        for key, var in self.config.items():
            config_data[key] = var.get()
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                self.log(f"配置已保存: {filename}")
                messagebox.showinfo("成功", "配置已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败: {e}")
    
    def load_config(self):
        """加载配置"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                for key, value in config_data.items():
                    if key in self.config:
                        self.config[key].set(value)
                
                self.log(f"配置已加载: {filename}")
                messagebox.showinfo("成功", "配置已加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {e}")


def main():
    """主函数"""
    root = tk.Tk()
    app = UNetTrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()