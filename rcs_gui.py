#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCS POD分析程序图形界面
提供友好的GUI控制所有分析参数
"""

import sys
import os
import json
import threading
import subprocess
import time
import warnings
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# 过滤subprocess的RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="subprocess")


class RCS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RCS POD分析程序 - 图形界面")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # 配置文件路径
        self.config_file = "rcs_gui_config.json"
        
        # 运行状态
        self.is_running = False
        self.process = None
        
        # 创建主界面
        self.create_widgets()
        
        # 加载保存的配置
        self.load_config()
    
    def create_widgets(self):
        """创建主界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="RCS POD分析程序控制面板", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # 右侧日志面板
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        
        # 创建标签页控制面板
        self.create_control_panels(control_frame)
        
        # 创建日志显示面板
        self.create_log_panel(log_frame)
        
        # 创建运行控制按钮
        self.create_run_controls(control_frame)
    
    def create_control_panels(self, parent):
        """创建参数控制标签页"""
        # 创建标签页容器
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        parent.rowconfigure(0, weight=1)
        
        # 创建各个标签页
        self.create_basic_params_tab()
        self.create_pod_params_tab() 
        self.create_autoencoder_params_tab()
        self.create_training_params_tab()
    
    def create_basic_params_tab(self):
        """创建基础参数标签页"""
        # 基础参数标签页
        basic_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(basic_frame, text="基础参数")
        
        # 存储参数变量
        self.basic_vars = {}
        
        row = 0
        
        # 参数文件路径
        ttk.Label(basic_frame, text="参数文件路径:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.basic_vars['params_path'] = tk.StringVar(value="../parameter/parameters_sorted.csv")
        params_frame = ttk.Frame(basic_frame)
        params_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        params_frame.columnconfigure(0, weight=1)
        ttk.Entry(params_frame, textvariable=self.basic_vars['params_path'], width=40).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(params_frame, text="浏览", 
                  command=lambda: self.browse_file(self.basic_vars['params_path'], 
                                                 [("CSV files", "*.csv"), ("All files", "*.*")])).grid(row=0, column=1, padx=(5, 0))
        row += 1
        
        # RCS数据目录
        ttk.Label(basic_frame, text="RCS数据目录:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.basic_vars['rcs_dir'] = tk.StringVar(value="../parameter/csv_output")
        rcs_frame = ttk.Frame(basic_frame)
        rcs_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        rcs_frame.columnconfigure(0, weight=1)
        ttk.Entry(rcs_frame, textvariable=self.basic_vars['rcs_dir'], width=40).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(rcs_frame, text="浏览", 
                  command=lambda: self.browse_directory(self.basic_vars['rcs_dir'])).grid(row=0, column=1, padx=(5, 0))
        row += 1
        
        # 输出目录
        ttk.Label(basic_frame, text="输出目录:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.basic_vars['output_dir'] = tk.StringVar(value="./results")
        output_frame = ttk.Frame(basic_frame)
        output_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.basic_vars['output_dir'], width=40).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(output_frame, text="浏览", 
                  command=lambda: self.browse_directory(self.basic_vars['output_dir'])).grid(row=0, column=1, padx=(5, 0))
        row += 1
        
        # 分析频率
        ttk.Label(basic_frame, text="分析频率:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.basic_vars['freq'] = tk.StringVar(value="both")
        freq_combo = ttk.Combobox(basic_frame, textvariable=self.basic_vars['freq'], 
                                 values=["1.5G", "3G", "both"], state="readonly", width=37)
        freq_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 模型数量
        ttk.Label(basic_frame, text="模型数量:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.basic_vars['num_models'] = tk.StringVar(value="100")
        ttk.Entry(basic_frame, textvariable=self.basic_vars['num_models'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 训练集大小
        ttk.Label(basic_frame, text="训练集大小:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(basic_frame, text="(逗号分隔多个值)", font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=(5, 0))
        self.basic_vars['num_train'] = tk.StringVar(value="80")
        ttk.Entry(basic_frame, textvariable=self.basic_vars['num_train'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 预测模式
        self.basic_vars['predict_mode'] = tk.BooleanVar(value=False)
        predict_check = ttk.Checkbutton(basic_frame, text="启用预测模式", 
                                       variable=self.basic_vars['predict_mode'],
                                       command=self.toggle_predict_mode)
        predict_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1
        
        # 预测参数文件
        ttk.Label(basic_frame, text="预测参数文件:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.basic_vars['param_file'] = tk.StringVar(value="")
        self.param_file_frame = ttk.Frame(basic_frame)
        self.param_file_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        self.param_file_frame.columnconfigure(0, weight=1)
        self.param_file_entry = ttk.Entry(self.param_file_frame, textvariable=self.basic_vars['param_file'], 
                                         width=40, state="disabled")
        self.param_file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.param_file_button = ttk.Button(self.param_file_frame, text="浏览", state="disabled",
                                           command=lambda: self.browse_file(self.basic_vars['param_file'], 
                                                                           [("CSV files", "*.csv"), ("All files", "*.*")]))
        self.param_file_button.grid(row=0, column=1, padx=(5, 0))
        
        # 配置列权重
        basic_frame.columnconfigure(1, weight=1)
    
    def create_pod_params_tab(self):
        """创建POD参数标签页"""
        pod_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pod_frame, text="POD参数")
        
        self.pod_vars = {}
        
        row = 0
        
        # POD多模态对比分析
        ttk.Label(pod_frame, text="POD多模态对比:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(pod_frame, text="(逗号分隔)", font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=(5, 0))
        self.pod_vars['pod_modes'] = tk.StringVar(value="10,20,30,40")
        ttk.Entry(pod_frame, textvariable=self.pod_vars['pod_modes'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 能量覆盖阈值
        ttk.Label(pod_frame, text="能量覆盖阈值 (%):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.pod_vars['energy_threshold'] = tk.StringVar(value="95")
        energy_combo = ttk.Combobox(pod_frame, textvariable=self.pod_vars['energy_threshold'], 
                                   values=["90", "95", "99"], width=37)
        energy_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 可视化模态数量
        ttk.Label(pod_frame, text="可视化模态数量:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.pod_vars['num_modes_visualize'] = tk.StringVar(value="10")
        ttk.Entry(pod_frame, textvariable=self.pod_vars['num_modes_visualize'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        row += 1
        ttk.Label(pod_frame, text="POD重建模态数:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.pod_vars['pod_reconstruct_num'] = tk.StringVar(value="0")
        pod_recon_frame = ttk.Frame(pod_frame)
        pod_recon_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        ttk.Entry(pod_recon_frame, textvariable=self.pod_vars['pod_reconstruct_num'], width=10).pack(side=tk.LEFT)
        ttk.Label(pod_recon_frame, text="(0表示使用能量阈值自动确定)", foreground="gray").pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        
        # 添加说明
        info_text = """POD参数说明:
• POD多模态对比: 指定要进行重建对比分析的POD模态数量列表
• 能量覆盖阈值: 保留模态覆盖的总能量百分比
• 可视化模态数量: 用于详细分析和可视化的模态数量
• POD重建模态数: 用于单次重建的模态数量 (0表示自动确定)"""
        
        info_label = ttk.Label(pod_frame, text=info_text, font=('Arial', 9), 
                              foreground='gray', justify=tk.LEFT)
        info_label.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        pod_frame.columnconfigure(1, weight=1)
    
    def create_autoencoder_params_tab(self):
        """创建自编码器参数标签页"""
        ae_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(ae_frame, text="自编码器参数")
        
        self.ae_vars = {}
        
        row = 0
        
        # 潜在空间维度
        ttk.Label(ae_frame, text="潜在空间维度:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(ae_frame, text="(逗号分隔)", font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, padx=(5, 0))
        self.ae_vars['latent_dims'] = tk.StringVar(value="5,10,15,20")
        ttk.Entry(ae_frame, textvariable=self.ae_vars['latent_dims'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 模型类型
        ttk.Label(ae_frame, text="模型类型:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ae_vars['model_types'] = tk.StringVar(value="standard,vae")
        types_frame = ttk.Frame(ae_frame)
        types_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        self.ae_vars['use_standard'] = tk.BooleanVar(value=True)
        self.ae_vars['use_vae'] = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(types_frame, text="标准自编码器", 
                       variable=self.ae_vars['use_standard'], 
                       command=self.update_model_types).pack(side=tk.LEFT)
        ttk.Checkbutton(types_frame, text="变分自编码器", 
                       variable=self.ae_vars['use_vae'], 
                       command=self.update_model_types).pack(side=tk.LEFT, padx=(10, 0))
        row += 1
        
        # 计算设备
        ttk.Label(ae_frame, text="计算设备:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ae_vars['device'] = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(ae_frame, textvariable=self.ae_vars['device'], 
                                   values=["auto", "cpu", "cuda"], width=37)
        device_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 训练轮数
        ttk.Label(ae_frame, text="训练轮数:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ae_vars['epochs'] = tk.StringVar(value="200")
        ttk.Entry(ae_frame, textvariable=self.ae_vars['epochs'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 学习率
        ttk.Label(ae_frame, text="学习率:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ae_vars['learning_rate'] = tk.StringVar(value="0.001")
        ttk.Entry(ae_frame, textvariable=self.ae_vars['learning_rate'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 批次大小模式
        ttk.Label(ae_frame, text="批次大小:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ae_vars['batch_size_mode'] = tk.StringVar(value="auto")
        batch_frame = ttk.Frame(ae_frame)
        batch_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        ttk.Radiobutton(batch_frame, text="自动优化", variable=self.ae_vars['batch_size_mode'], 
                       value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(batch_frame, text="手动设置:", variable=self.ae_vars['batch_size_mode'], 
                       value="manual").pack(side=tk.LEFT, padx=(10, 0))
        
        self.ae_vars['batch_size'] = tk.StringVar(value="32")
        self.batch_size_entry = ttk.Entry(batch_frame, textvariable=self.ae_vars['batch_size'], width=10)
        self.batch_size_entry.pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        
        # 跳过AE重训练选项
        self.ae_vars['skip_training'] = tk.BooleanVar(value=False)
        skip_ae_check = ttk.Checkbutton(ae_frame, text="跳过AE重训练，优先使用已有模型", 
                                       variable=self.ae_vars['skip_training'])
        skip_ae_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1
        
        ae_frame.columnconfigure(1, weight=1)
    
    def create_training_params_tab(self):
        """创建训练参数标签页"""
        train_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(train_frame, text="训练参数")
        
        self.train_vars = {}
        
        row = 0
        
        # L2正则化系数
        ttk.Label(train_frame, text="L2正则化系数:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.train_vars['weight_decay'] = tk.StringVar(value="1e-5")
        ttk.Entry(train_frame, textvariable=self.train_vars['weight_decay'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # VAE KL散度权重
        ttk.Label(train_frame, text="VAE KL散度权重:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.train_vars['beta'] = tk.StringVar(value="1.0")
        ttk.Entry(train_frame, textvariable=self.train_vars['beta'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 早停耐心值
        ttk.Label(train_frame, text="早停耐心值:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.train_vars['patience'] = tk.StringVar(value="50")
        ttk.Entry(train_frame, textvariable=self.train_vars['patience'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 学习率调度器耐心值
        ttk.Label(train_frame, text="学习率调度耐心值:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.train_vars['scheduler_patience'] = tk.StringVar(value="20")
        ttk.Entry(train_frame, textvariable=self.train_vars['scheduler_patience'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 学习率衰减因子
        ttk.Label(train_frame, text="学习率衰减因子:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.train_vars['scheduler_factor'] = tk.StringVar(value="0.5")
        ttk.Entry(train_frame, textvariable=self.train_vars['scheduler_factor'], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        row += 1
        
        # 添加说明
        info_text = """训练参数说明:
• 这些参数用于精细控制自编码器训练过程
• 修改这些参数需要编辑相应的源代码文件
• 建议先使用默认值进行测试"""
        
        info_label = ttk.Label(train_frame, text=info_text, font=('Arial', 9), 
                              foreground='gray', justify=tk.LEFT)
        info_label.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        train_frame.columnconfigure(1, weight=1)
    
    def create_log_panel(self, parent):
        """创建日志显示面板"""
        ttk.Label(parent, text="运行日志", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 创建日志文本框
        self.log_text = scrolledtext.ScrolledText(parent, width=80, height=40, 
                                                 font=('Consolas', 9), wrap=tk.WORD)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 添加清除日志按钮
        ttk.Button(parent, text="清除日志", command=self.clear_log).grid(row=2, column=0, sticky=tk.E, pady=(5, 0))
    
    def create_run_controls(self, parent):
        """创建运行控制按钮"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 运行按钮
        self.run_button = ttk.Button(button_frame, text="开始分析", command=self.run_analysis, style='Accent.TButton')
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 停止按钮
        self.stop_button = ttk.Button(button_frame, text="停止分析", command=self.stop_analysis, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 保存配置按钮
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=(0, 10))
        
        # 加载配置按钮
        ttk.Button(button_frame, text="加载配置", command=self.load_config_dialog).pack(side=tk.LEFT, padx=(0, 10))
        
        # 生成命令按钮
        ttk.Button(button_frame, text="生成命令", command=self.show_command).pack(side=tk.LEFT)
    
    def toggle_predict_mode(self):
        """切换预测模式"""
        if self.basic_vars['predict_mode'].get():
            self.param_file_entry.config(state="normal")
            self.param_file_button.config(state="normal")
        else:
            self.param_file_entry.config(state="disabled")
            self.param_file_button.config(state="disabled")
    
    def update_model_types(self):
        """更新模型类型字符串"""
        types = []
        if self.ae_vars['use_standard'].get():
            types.append('standard')
        if self.ae_vars['use_vae'].get():
            types.append('vae')
        self.ae_vars['model_types'].set(','.join(types))
    
    def browse_file(self, var, filetypes):
        """浏览文件"""
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            var.set(filename)
    
    def browse_directory(self, var):
        """浏览目录"""
        dirname = filedialog.askdirectory()
        if dirname:
            var.set(dirname)
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)
    
    def generate_command(self):
        """生成运行命令"""
        cmd = ["python", "run.py"]
        
        # 添加基础参数
        if self.basic_vars['params_path'].get():
            cmd.extend(["--params_path", self.basic_vars['params_path'].get()])
        if self.basic_vars['rcs_dir'].get():
            cmd.extend(["--rcs_dir", self.basic_vars['rcs_dir'].get()])
        if self.basic_vars['output_dir'].get():
            cmd.extend(["--output_dir", self.basic_vars['output_dir'].get()])
        if self.basic_vars['freq'].get():
            cmd.extend(["--freq", self.basic_vars['freq'].get()])
        if self.basic_vars['num_models'].get():
            cmd.extend(["--num_models", self.basic_vars['num_models'].get()])
        if self.basic_vars['num_train'].get():
            cmd.extend(["--num_train", self.basic_vars['num_train'].get()])
        if self.basic_vars['predict_mode'].get():
            cmd.append("--predict_mode")
        if self.basic_vars['predict_mode'].get() and self.basic_vars['param_file'].get():
            cmd.extend(["--param_file", self.basic_vars['param_file'].get()])
        
        # 添加自编码器参数
        if self.ae_vars['latent_dims'].get():
            cmd.extend(["--latent_dims", self.ae_vars['latent_dims'].get()])
        if self.ae_vars['model_types'].get():
            cmd.extend(["--model_types", self.ae_vars['model_types'].get()])
        if self.ae_vars['epochs'].get():
            cmd.extend(["--ae_epochs", self.ae_vars['epochs'].get()])
        if self.ae_vars['device'].get():
            cmd.extend(["--ae_device", self.ae_vars['device'].get()])
        if self.ae_vars['learning_rate'].get():
            cmd.extend(["--ae_learning_rate", self.ae_vars['learning_rate'].get()])
        
        # 处理批次大小
        if self.ae_vars['batch_size_mode'].get() == "manual" and self.ae_vars['batch_size'].get():
            cmd.extend(["--ae_batch_size", self.ae_vars['batch_size'].get()])
        else:
            cmd.extend(["--ae_batch_size", "0"])  # 0表示自动
        
        # 跳过AE重训练选项
        if self.ae_vars['skip_training'].get():
            cmd.append("--skip_ae_training")
        
        # 添加POD参数
        if self.pod_vars['pod_modes'].get():
            cmd.extend(["--pod_modes", self.pod_vars['pod_modes'].get()])
        if self.pod_vars['energy_threshold'].get():
            cmd.extend(["--energy_threshold", self.pod_vars['energy_threshold'].get()])
        if self.pod_vars['num_modes_visualize'].get():
            cmd.extend(["--num_modes_visualize", self.pod_vars['num_modes_visualize'].get()])
        if self.pod_vars['pod_reconstruct_num'].get():
            cmd.extend(["--pod_reconstruct_num", self.pod_vars['pod_reconstruct_num'].get()])
        
        return cmd
    
    def show_command(self):
        """显示生成的命令"""
        cmd = self.generate_command()
        cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
        
        # 创建命令显示窗口
        cmd_window = tk.Toplevel(self.root)
        cmd_window.title("生成的命令")
        cmd_window.geometry("800x200")
        
        ttk.Label(cmd_window, text="生成的命令行:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        cmd_text = scrolledtext.ScrolledText(cmd_window, height=6, wrap=tk.WORD)
        cmd_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        cmd_text.insert(tk.END, cmd_str)
        cmd_text.config(state="disabled")
        
        # 复制按钮
        def copy_command():
            self.root.clipboard_clear()
            self.root.clipboard_append(cmd_str)
            messagebox.showinfo("复制成功", "命令已复制到剪贴板")
        
        ttk.Button(cmd_window, text="复制到剪贴板", command=copy_command).pack(pady=5)
    
    def run_analysis(self):
        """运行分析"""
        if self.is_running:
            return
        
        self.is_running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # 清除之前的日志
        self.clear_log()
        self.log_message("开始RCS POD分析...")
        
        # 生成命令
        cmd = self.generate_command()
        cmd_str = ' '.join(cmd)
        self.log_message(f"执行命令: {cmd_str}")
        
        # 在后台线程运行
        self.run_thread = threading.Thread(target=self._run_process, args=(cmd,))
        self.run_thread.daemon = True
        self.run_thread.start()
    
    def _run_process(self, cmd):
        """在后台运行进程"""
        try:
            # 启动进程，使用更安全的编码处理
            import locale
            
            # 尝试获取系统编码
            try:
                system_encoding = locale.getpreferredencoding()
            except:
                system_encoding = 'utf-8'
            
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                bufsize=0  # 无缓冲，避免二进制模式下的行缓冲警告
            )
            
            # 实时读取输出，手动处理编码
            for line in iter(self.process.stdout.readline, b''):
                if line:
                    try:
                        # 尝试多种编码方式解码
                        try:
                            decoded_line = line.decode('utf-8').rstrip()
                        except UnicodeDecodeError:
                            try:
                                decoded_line = line.decode('gbk').rstrip()
                            except UnicodeDecodeError:
                                try:
                                    decoded_line = line.decode(system_encoding).rstrip()
                                except UnicodeDecodeError:
                                    # 最后的备选方案，忽略错误字符
                                    decoded_line = line.decode('utf-8', errors='replace').rstrip()
                        
                        self.root.after(0, self.log_message, decoded_line)
                    except Exception as decode_error:
                        # 解码完全失败时的处理
                        self.root.after(0, self.log_message, f"[解码错误] {repr(line)}")
            
            # 等待进程结束
            self.process.wait()
            
            if self.process.returncode == 0:
                self.root.after(0, self.log_message, "分析完成!")
            else:
                self.root.after(0, self.log_message, f"分析失败，退出码: {self.process.returncode}")
                
        except Exception as e:
            self.root.after(0, self.log_message, f"运行错误: {str(e)}")
        finally:
            self.root.after(0, self._process_finished)
    
    def _process_finished(self):
        """进程结束处理"""
        self.is_running = False
        self.process = None
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")
    
    def stop_analysis(self):
        """停止分析"""
        if self.process:
            self.process.terminate()
            self.log_message("⏹️ 分析已停止")
            self._process_finished()
    
    def save_config(self):
        """保存配置到文件"""
        config = {
            'basic': {k: v.get() for k, v in self.basic_vars.items()},
            'pod': {k: v.get() for k, v in self.pod_vars.items()},
            'autoencoder': {k: v.get() for k, v in self.ae_vars.items()},
            'training': {k: v.get() for k, v in self.train_vars.items()}
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("保存成功", f"配置已保存到 {self.config_file}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存配置失败: {str(e)}")
    
    def load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载基础参数
                if 'basic' in config:
                    for k, v in config['basic'].items():
                        if k in self.basic_vars:
                            self.basic_vars[k].set(v)
                
                # 加载POD参数
                if 'pod' in config:
                    for k, v in config['pod'].items():
                        if k in self.pod_vars:
                            self.pod_vars[k].set(v)
                
                # 加载自编码器参数
                if 'autoencoder' in config:
                    for k, v in config['autoencoder'].items():
                        if k in self.ae_vars:
                            self.ae_vars[k].set(v)
                
                # 加载训练参数
                if 'training' in config:
                    for k, v in config['training'].items():
                        if k in self.train_vars:
                            self.train_vars[k].set(v)
                
                # 更新界面状态
                self.toggle_predict_mode()
                self.update_model_types()
                
        except Exception as e:
            print(f"加载配置失败: {str(e)}")
    
    def load_config_dialog(self):
        """加载配置对话框"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.config_file = filename
            self.load_config()
            messagebox.showinfo("加载成功", f"配置已从 {filename} 加载")


def main():
    root = tk.Tk()
    app = RCS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()