#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCS POD Analysis - Modern Streamlit Interface
现代化的RCS POD分析程序界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import subprocess
import time
from pathlib import Path
import sys

# 配置页面
st.set_page_config(
    page_title="RCS POD Analysis Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'logs' not in st.session_state:
    st.session_state.logs = []

def load_default_config():
    """加载默认配置"""
    return {
        'params_path': '../parameter/parameters_sorted.csv',
        'rcs_dir': '../parameter/csv_output',
        'output_dir': './results',
        'frequency': ['1.5G'],
        'num_models': 100,
        'num_train': [80],
        'pod_enabled': True,
        'ae_enabled': True,
        'skip_ae_training': False,
        'latent_dims': [5, 10, 15, 20],
        'model_types': ['standard', 'vae'],
        # POD参数
        'pod_modes': [10, 20, 30, 40],
        'energy_threshold': 95.0,
        'num_modes_visualize': 10,
        'pod_reconstruct_num': 0,
        # Autoencoder训练参数
        'ae_epochs': 200,
        'ae_device': 'auto',
        'ae_learning_rate': 0.001,
        'ae_batch_size': 0
    }

def save_config(config, filename="streamlit_config.json"):
    """保存配置到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def load_config(filename="streamlit_config.json"):
    """从文件加载配置"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return load_default_config()

def run_analysis_command(config):
    """生成分析命令"""
    # 确保使用当前环境的Python
    import sys
    python_executable = sys.executable
    cmd = [python_executable, 'run.py']
    cmd.extend(['--params_path', config['params_path']])
    cmd.extend(['--rcs_dir', config['rcs_dir']])
    cmd.extend(['--output_dir', config['output_dir']])
    
    # 处理频率参数 - 根据选择生成正确的参数
    frequencies = config['frequency']
    if len(frequencies) == 2 and '1.5G' in frequencies and '3G' in frequencies:
        cmd.extend(['--freq', 'both'])
    elif len(frequencies) == 1:
        cmd.extend(['--freq', frequencies[0]])
    else:
        # 默认使用both，或者取第一个
        freq_value = 'both' if len(frequencies) > 1 else (frequencies[0] if frequencies else '1.5G')
        cmd.extend(['--freq', freq_value])
    
    cmd.extend(['--num_models', str(config['num_models'])])
    cmd.extend(['--num_train', ','.join(map(str, config['num_train']))])
    
    # POD参数
    if config.get('pod_enabled', True):
        cmd.extend(['--pod_modes', ','.join(map(str, config.get('pod_modes', [10, 20, 30, 40])))])
        cmd.extend(['--energy_threshold', str(config.get('energy_threshold', 95.0))])
        cmd.extend(['--num_modes_visualize', str(config.get('num_modes_visualize', 10))])
        cmd.extend(['--pod_reconstruct_num', str(config.get('pod_reconstruct_num', 0))])
    
    # Autoencoder参数
    if config.get('ae_enabled', True):
        cmd.extend(['--latent_dims', ','.join(map(str, config.get('latent_dims', [5, 10, 15, 20])))])
        cmd.extend(['--model_types', ','.join(config.get('model_types', ['standard', 'vae']))])
        cmd.extend(['--ae_epochs', str(config.get('ae_epochs', 200))])
        cmd.extend(['--ae_device', config.get('ae_device', 'auto')])
        cmd.extend(['--ae_learning_rate', str(config.get('ae_learning_rate', 0.001))])
        cmd.extend(['--ae_batch_size', str(config.get('ae_batch_size', 0))])
        
        if config.get('skip_ae_training', False):
            cmd.append('--skip_ae_training')
    
    return cmd

def read_log_file_updates():
    """从日志文件读取新的日志行，并检测程序结束标志"""
    if not st.session_state.log_file_path or not os.path.exists(st.session_state.log_file_path):
        return []
    
    new_logs = []
    completion_detected = False
    
    try:
        # 使用utf-8编码读取文件（简化编码处理）
        with open(st.session_state.log_file_path, 'r', encoding='utf-8', errors='replace') as f:
            # 获取当前文件大小
            f.seek(0, 2)
            current_size = f.tell()
            
            # 如果文件大小没有变化，说明没有新内容
            if current_size <= st.session_state.log_file_position:
                return []
            
            # 从上次读取位置开始读取
            f.seek(st.session_state.log_file_position)
            new_content = f.read()
            
            if new_content:
                # 更新文件位置
                st.session_state.log_file_position = f.tell()
                
                # 按行分割新内容，过滤空行
                new_lines = [line.strip() for line in new_content.split('\n') if line.strip()]
                new_logs.extend(new_lines)
                
                # 检测程序完成标志
                for line in new_lines:
                    # 检测多种完成标志
                    completion_keywords = [
                        "分析完成！",
                        "总运行时间:",
                        "POD和模态分析完成。结果保存在",
                        "结果保存在: ./results"
                    ]
                    
                    for keyword in completion_keywords:
                        if keyword in line:
                            completion_detected = True
                            st.session_state.program_completion_detected = True
                            break
                    
                    if completion_detected:
                        break
                
    except Exception as e:
        new_logs.append(f"ERROR: 读取日志文件失败: {str(e)}")
    
    return new_logs

class FlushingLogFile:
    """自动刷新的日志文件包装器，确保每次写入都立即刷新到磁盘"""
    def __init__(self, file_path, mode='a', encoding='utf-8'):
        self.file = open(file_path, mode, encoding=encoding, buffering=1)
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()  # 立即刷新到磁盘
        os.fsync(self.file.fileno())  # 强制操作系统立即写入磁盘
        
    def flush(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        
    def close(self):
        self.file.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_log_file():
    """创建新的日志文件"""
    import time  # 确保time模块在函数作用域中可用
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = f"streamlit_analysis_{timestamp}.log"
    log_path = os.path.join(os.getcwd(), log_filename)
    
    # 创建日志文件
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== RCS POD Analysis Log Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.flush()
        
        st.session_state.log_file_path = log_path
        st.session_state.log_file_position = 0
        return log_path
    except Exception as e:
        st.error(f"创建日志文件失败: {str(e)}")
        return None

def estimate_progress(logs):
    """基于日志内容估算分析进度"""
    if not logs:
        return 0
    
    # 定义关键进度节点
    progress_keywords = {
        "开始": 5,
        "加载数据": 10, 
        "loading": 10,
        "POD分析": 25,
        "SVD": 30,
        "特征值": 35,
        "模态分析": 45,
        "autoencoder": 60,
        "训练": 70,
        "training": 70,
        "保存": 85,
        "save": 85,
        "完成": 95,
        "success": 95,
        "finished": 100
    }
    
    max_progress = 0
    recent_logs = logs[-20:]  # 只检查最近20行日志
    
    for log in recent_logs:
        log_lower = log.lower()
        for keyword, progress in progress_keywords.items():
            if keyword in log_lower:
                max_progress = max(max_progress, progress)
    
    return max_progress

def main():
    # 初始化session state
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_process' not in st.session_state:
        st.session_state.analysis_process = None
    if 'last_log_check' not in st.session_state:
        st.session_state.last_log_check = 0
    if 'log_file_path' not in st.session_state:
        st.session_state.log_file_path = None
    if 'log_file_position' not in st.session_state:
        st.session_state.log_file_position = 0
    
    # 主标题
    st.markdown('<h1 class="main-header">📡 RCS POD Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # 如果分析正在运行，显示实时状态和流式更新机制
    if st.session_state.analysis_running:
        # 创建实时状态显示区域
        status_container = st.container()
        with status_container:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("""
                <div style="background-color: #e1f5fe; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <p style="margin: 0; color: #0277bd; font-weight: bold;">
                        🔄 实时分析中 - 日志正在流式更新...
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # 显示运行时间
                if hasattr(st.session_state, 'analysis_start_time'):
                    import time  # 确保time模块在作用域中可用
                    runtime = time.time() - st.session_state.analysis_start_time
                    st.metric("运行时间", f"{int(runtime//60)}:{int(runtime%60):02d}")
                else:
                    st.metric("运行时间", "未知")
            
            with col3:
                # 显示估算进度
                progress = estimate_progress(st.session_state.logs)
                st.metric("分析进度", f"{progress}%")
        
        # 添加进度条
        if st.session_state.logs:
            progress = estimate_progress(st.session_state.logs)
            st.progress(progress / 100, f"分析进度: {progress}%")
    
    # 环境信息显示（可折叠）
    with st.expander("🔍 环境信息", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Python路径: {sys.executable}")
            st.text(f"Python版本: {sys.version.split()[0]}")
            st.text(f"工作目录: {os.getcwd()}")
        with col2:
            # 检查关键包
            try:
                import torch
                st.text(f"✅ PyTorch: {torch.__version__}")
                st.text(f"✅ CUDA可用: {torch.cuda.is_available()}")
            except ImportError:
                st.text("❌ PyTorch: 未安装")
            
            try:
                import pandas as pd
                st.text(f"✅ Pandas: {pd.__version__}")
            except ImportError:
                st.text("❌ Pandas: 未安装")
            
            # 检查conda环境
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
            st.text(f"Conda环境: {conda_env}")
    
    # 侧边栏配置
    st.sidebar.markdown("## ⚙️ 分析配置")
    
    # 加载配置
    config = load_config()
    
    # 基础参数配置
    st.sidebar.markdown("### 📁 数据路径")
    config['params_path'] = st.sidebar.text_input(
        "参数文件路径", 
        value=config.get('params_path', '../parameter/parameters_sorted.csv')
    )
    
    config['rcs_dir'] = st.sidebar.text_input(
        "RCS数据目录", 
        value=config.get('rcs_dir', '../parameter/csv_output')
    )
    
    config['output_dir'] = st.sidebar.text_input(
        "输出目录", 
        value=config.get('output_dir', './results')
    )
    
    # 分析参数
    st.sidebar.markdown("### 🎯 分析参数")
    config['frequency'] = st.sidebar.multiselect(
        "频率选择",
        options=['1.5G', '3G'],
        default=config.get('frequency', ['1.5G'])
    )
    
    config['num_models'] = st.sidebar.number_input(
        "模型数量", 
        min_value=1, 
        max_value=100, 
        value=config.get('num_models', 100)
    )
    
    # 训练集大小
    train_sizes_str = st.sidebar.text_input(
        "训练集大小 (逗号分隔)", 
        value=','.join(map(str, config.get('num_train', [80])))
    )
    try:
        config['num_train'] = [int(x.strip()) for x in train_sizes_str.split(',')]
    except:
        config['num_train'] = [80]
    
    # 算法配置
    st.sidebar.markdown("### 🧠 算法配置")
    config['pod_enabled'] = st.sidebar.checkbox("启用POD分析", value=config.get('pod_enabled', True))
    config['ae_enabled'] = st.sidebar.checkbox("启用Autoencoder分析", value=config.get('ae_enabled', True))
    config['skip_ae_training'] = st.sidebar.checkbox("跳过AE重训练", value=config.get('skip_ae_training', False))
    
    # 保存配置
    if st.sidebar.button("💾 保存配置"):
        save_config(config)
        st.sidebar.success("配置已保存!")
    
    # 算法参数配置区域
    st.markdown("---")
    st.markdown("### ⚙️ 详细参数配置")
    
    # 创建POD和AE参数的横向布局
    param_col1, param_col2 = st.columns(2)
    
    # POD参数配置
    with param_col1:
        if config['pod_enabled']:
            st.markdown("#### 📐 POD分析参数")
            
            # POD多模态对比分析
            pod_modes_str = st.text_input(
                "POD多模态对比 (逗号分隔)", 
                value=','.join(map(str, config.get('pod_modes', [10, 20, 30, 40]))),
                help="指定要进行重建对比分析的POD模态数量列表，如：10,20,30,40。程序会分别使用这些数量的模态进行RCS重建，以评估不同模态数的重建效果"
            )
            try:
                config['pod_modes'] = [int(x.strip()) for x in pod_modes_str.split(',')]
            except:
                config['pod_modes'] = [10, 20, 30, 40]
            
            # 能量阈值 - 改为输入框
            energy_threshold_input = st.text_input(
                "能量阈值 (%)", 
                value=str(config.get('energy_threshold', 95.0)),
                help="自动确定模态数量的能量阈值，支持任意精度的百分比值，如：95.0, 99.5, 90.2"
            )
            try:
                config['energy_threshold'] = float(energy_threshold_input)
                if not (0 < config['energy_threshold'] < 100):
                    st.error("能量阈值必须在0-100之间")
                    config['energy_threshold'] = 95.0
            except ValueError:
                st.error("请输入有效的数值")
                config['energy_threshold'] = 95.0
            
            # POD其他参数
            pod_col1, pod_col2 = st.columns(2)
            with pod_col1:
                config['num_modes_visualize'] = st.number_input(
                    "可视化模态数", 
                    min_value=1, 
                    max_value=50, 
                    value=config.get('num_modes_visualize', 10),
                    help="在图表中显示的POD模态数量"
                )
            with pod_col2:
                config['pod_reconstruct_num'] = st.number_input(
                    "重建使用的模态数", 
                    min_value=0, 
                    max_value=100, 
                    value=config.get('pod_reconstruct_num', 0),
                    help="0表示使用能量阈值自动确定"
                )
        else:
            st.markdown("#### 📐 POD分析参数")
            st.info("POD分析已禁用")
    
    # Autoencoder参数配置
    with param_col2:
        if config['ae_enabled']:
            st.markdown("#### 🔬 Autoencoder参数")
            
            # 基础参数
            latent_dims_str = st.text_input(
                "隐空间维度 (逗号分隔)", 
                value=','.join(map(str, config.get('latent_dims', [5, 10, 15, 20]))),
                help="要测试的隐空间维度列表，如：5,10,15,20"
            )
            try:
                config['latent_dims'] = [int(x.strip()) for x in latent_dims_str.split(',')]
            except:
                config['latent_dims'] = [5, 10, 15, 20]
                
            config['model_types'] = st.multiselect(
                "模型类型",
                options=['standard', 'vae'],
                default=config.get('model_types', ['standard', 'vae']),
                help="选择要训练的自编码器类型"
            )
            
            # 训练参数
            st.markdown("**训练参数**")
            ae_col1, ae_col2 = st.columns(2)
            
            with ae_col1:
                config['ae_epochs'] = st.number_input(
                    "训练轮数", 
                    min_value=50, 
                    max_value=1000, 
                    value=config.get('ae_epochs', 200)
                )
                
                config['ae_learning_rate'] = st.number_input(
                    "学习率", 
                    min_value=0.0001, 
                    max_value=0.1, 
                    value=config.get('ae_learning_rate', 0.001),
                    format="%.4f"
                )
            
            with ae_col2:
                config['ae_batch_size'] = st.number_input(
                    "批次大小", 
                    min_value=0, 
                    max_value=256, 
                    value=config.get('ae_batch_size', 0),
                    help="0表示自动确定"
                )
                
                config['ae_device'] = st.selectbox(
                    "计算设备",
                    options=['auto', 'cpu', 'cuda'],
                    index=['auto', 'cpu', 'cuda'].index(config.get('ae_device', 'auto'))
                )
        else:
            st.markdown("#### 🔬 Autoencoder参数")
            st.info("Autoencoder分析已禁用")
    
    # 保存详细参数配置
    st.markdown("---")
    col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
    with col_save2:
        if st.button("💾 保存所有参数配置", width='stretch'):
            save_config(config)
            st.success("所有参数配置已保存!")
    
    # 主界面
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 当前配置概览")
        
        # 显示配置信息
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            st.metric("频率", f"{len(config['frequency'])} 个", delta="1.5G, 3G")
            st.metric("模型数量", config['num_models'])
            
        with col1_2:
            st.metric("训练集大小", f"{len(config['num_train'])} 个", 
                     delta=f"最大: {max(config['num_train']) if config['num_train'] else 0}")
            if config['pod_enabled']:
                st.metric("POD模态数", f"{len(config.get('pod_modes', []))} 个",
                         delta=f"能量阈值: {config.get('energy_threshold', 95)}%")
                
        with col1_3:
            algorithms = []
            if config['pod_enabled']:
                algorithms.append("POD")
            if config['ae_enabled']:
                algorithms.append("Autoencoder")
            st.metric("算法", f"{len(algorithms)} 个", delta=" + ".join(algorithms))
            
            if config['ae_enabled']:
                st.metric("AE隐空间", f"{len(config.get('latent_dims', []))} 个维度",
                         delta=f"{len(config.get('model_types', []))} 种模型")
    
    with col2:
        st.markdown("### 🚀 运行控制")
        
        # 运行控制按钮
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if not st.session_state.analysis_running:
                if st.button("▶️ 开始分析", type="primary", width='stretch'):
                    # 验证配置
                    if not config['frequency']:
                        st.error("请至少选择一个频率!")
                    elif config['ae_enabled'] and not config['model_types']:
                        st.error("启用Autoencoder时请至少选择一个模型类型!")
                    else:
                        # 创建日志文件
                        log_file_path = create_log_file()
                        if log_file_path is None:
                            st.error("无法创建日志文件，请检查权限")
                            return
                        
                        # 生成命令并显示
                        cmd = run_analysis_command(config)
                        cmd_str = ' '.join(cmd)
                        
                        # 初始化日志
                        st.session_state.logs = []
                        st.session_state.logs.append("=== 开始分析 ===")
                        st.session_state.logs.append(f"当前工作目录: {os.getcwd()}")
                        st.session_state.logs.append(f"日志文件: {log_file_path}")
                        st.session_state.logs.append(f"执行命令: {cmd_str}")
                        st.session_state.logs.append("--- 分析程序输出 ---")
                        
                        # 启动后台进程，输出重定向到日志文件
                        try:
                            import subprocess
                            
                            # 使用简单的文件处理，确保立即刷新
                            with open(log_file_path, 'a', encoding='utf-8', buffering=1) as init_log:
                                init_log.write(f"=== 开始分析 ===\n")
                                init_log.write(f"当前工作目录: {os.getcwd()}\n") 
                                init_log.write(f"执行命令: {cmd_str}\n")
                                init_log.write("--- 分析程序输出 ---\n")
                                init_log.flush()
                            
                            # 重新打开文件用于subprocess
                            log_file = open(log_file_path, 'a', encoding='utf-8', buffering=1)
                            
                            # 启动进程，输出到日志文件
                            # 设置中文编码环境变量和无缓冲输出
                            env = os.environ.copy()
                            env['PYTHONIOENCODING'] = 'utf-8'
                            env['PYTHONUNBUFFERED'] = '1'  # 强制无缓冲输出
                            
                            process = subprocess.Popen(
                                cmd,
                                cwd=os.getcwd(),
                                stdout=log_file,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True,
                                bufsize=1,  # 行缓冲
                                env=env,
                                encoding='utf-8',
                                errors='replace'
                            )
                            
                            # 保存日志文件引用，以便后续关闭
                            st.session_state.analysis_log_file = log_file
                            
                            st.session_state.analysis_process = process
                            st.session_state.analysis_running = True
                            st.session_state.analysis_complete = False
                            st.session_state.last_log_check = 0
                            import time  # 确保time模块可用
                            st.session_state.analysis_start_time = time.time()  # 记录开始时间
                            
                            # 重置程序完成检测标志
                            st.session_state.program_completion_detected = False
                            if hasattr(st.session_state, 'completion_detected_time'):
                                delattr(st.session_state, 'completion_detected_time')
                            
                            # 设置文件读取位置为开头，以便读取所有日志
                            st.session_state.log_file_position = 0
                            
                            st.success(f"分析已启动！日志文件: {os.path.basename(log_file_path)}")
                            st.rerun()
                            
                        except Exception as e:
                            error_msg = f"启动分析进程失败: {str(e)}"
                            st.session_state.logs.append(error_msg)
                            st.error(f"启动失败: {str(e)}")
                            # 尝试写入错误到日志文件
                            try:
                                with open(log_file_path, 'a', encoding='utf-8') as f:
                                    f.write(f"{error_msg}\n")
                            except:
                                pass
            else:
                st.button("⏳ 分析进行中...", disabled=True, width='stretch')
        
        with col_btn2:
            if st.session_state.analysis_running and st.session_state.analysis_process:
                if st.button("⏹️ 停止分析", type="secondary", width='stretch'):
                    try:
                        process = st.session_state.analysis_process
                        if process and process.poll() is None:  # 进程还在运行
                            process.terminate()
                            import time
                            time.sleep(1)
                            if process.poll() is None:  # 如果还没结束，强制杀死
                                process.kill()
                            st.session_state.logs.append("=== 分析已被停止 ===")
                            st.session_state.analysis_running = False
                            st.session_state.analysis_process = None
                            st.success("分析已停止")
                            st.rerun()
                    except Exception as e:
                        st.error(f"停止失败: {str(e)}")
            else:
                st.button("⏹️ 停止分析", disabled=True, width='stretch')
        
        # 流式实时检查进程状态和日志更新
        if st.session_state.analysis_running and st.session_state.analysis_process:
            process = st.session_state.analysis_process
            
            # 创建一个占位符用于实时更新日志状态
            log_status_placeholder = st.empty()
            
            # 从日志文件读取新内容
            new_logs = read_log_file_updates()
            
            if new_logs:
                # 添加新日志到session state
                for line in new_logs:
                    st.session_state.last_log_check += 1
                    st.session_state.logs.append(f"[{st.session_state.last_log_check:04d}] {line}")
                
                # 记录最后更新时间
                import time
                st.session_state.last_log_update_time = time.time()
                
                # 更新日志状态显示
                with log_status_placeholder.container():
                    st.success(f"📥 获取到 {len(new_logs)} 条新日志 (总共 {len(st.session_state.logs)} 条)")
                
                # 立即刷新页面显示新日志
                st.rerun()
            else:
                # 没有新日志时，显示调试信息和等待状态
                with log_status_placeholder.container():
                    # 显示调试信息
                    if st.session_state.log_file_path and os.path.exists(st.session_state.log_file_path):
                        try:
                            file_size = os.path.getsize(st.session_state.log_file_path)
                            st.info(f"⏳ 等待新日志... (文件大小: {file_size}B, 读取位置: {st.session_state.log_file_position})")
                        except:
                            st.info("⏳ 等待新日志输出...")
                    else:
                        st.warning("日志文件不存在或路径错误")
                
                # 智能延迟：大幅降低刷新频率减少闪烁
                import time
                current_time = time.time()
                if hasattr(st.session_state, 'last_log_update_time'):
                    time_since_last_update = current_time - st.session_state.last_log_update_time
                    if time_since_last_update < 30:  # 最近30秒有更新，中等频率
                        time.sleep(2.0)
                    elif time_since_last_update < 60:  # 30-60秒无更新，低频率
                        time.sleep(5.0) 
                    else:  # 60秒以上无更新，很低频率
                        time.sleep(10.0)
                else:
                    time.sleep(3.0)  # 默认频率降低到3秒
                    
                st.rerun()
            
            # 检查程序完成检测和进程状态
            return_code = process.poll()
            
            # 如果通过日志检测到程序完成，但进程仍在运行，等待进程自然结束
            program_completed_by_log = getattr(st.session_state, 'program_completion_detected', False)
            
            import time
            current_time = time.time()
            if hasattr(st.session_state, 'analysis_start_time'):
                elapsed_time = current_time - st.session_state.analysis_start_time
                elapsed_minutes = elapsed_time / 60
                
                # 在状态显示中添加运行时间信息
                with log_status_placeholder.container():
                    if program_completed_by_log and return_code is None:
                        st.warning(f"🎯 程序已完成，等待进程结束... (已运行 {elapsed_minutes:.1f} 分钟) - PID: {process.pid}")
                        # 如果日志检测到完成超过30秒但进程还没结束，强制结束
                        if not hasattr(st.session_state, 'completion_detected_time'):
                            st.session_state.completion_detected_time = current_time
                        elif current_time - st.session_state.completion_detected_time > 30:
                            st.warning("程序完成超过30秒但进程未结束，强制结束进程")
                            try:
                                process.terminate()
                                time.sleep(2)
                                if process.poll() is None:
                                    process.kill()
                                return_code = 0  # 假设成功完成
                            except:
                                return_code = 1
                    elif return_code is None:  # 仍在运行
                        st.info(f"🔄 进程运行中 (已运行 {elapsed_minutes:.1f} 分钟) - PID: {process.pid}")
                    else:
                        st.success(f"✅ 进程已结束 (返回码: {return_code}) - 运行时间: {elapsed_minutes:.1f} 分钟")
            
            if return_code is not None:
                # 进程已结束，等待一下以确保所有输出都写入文件
                time.sleep(2)  # 增加等待时间确保输出完全写入
                
                # 读取剩余的日志
                remaining_logs = read_log_file_updates()
                if remaining_logs:
                    for line in remaining_logs:
                        st.session_state.last_log_check += 1
                        st.session_state.logs.append(f"[{st.session_state.last_log_check:04d}] {line}")
                
                # 写入结束标记到日志文件
                try:
                    if st.session_state.log_file_path:
                        with open(st.session_state.log_file_path, 'a', encoding='utf-8') as f:
                            if return_code == 0:
                                end_msg = f"=== 分析成功完成 (返回码: {return_code}) ==="
                                f.write(f"{end_msg}\n")
                                st.session_state.logs.append(end_msg)
                                st.session_state.analysis_complete = True
                            else:
                                end_msg = f"=== 分析失败，返回码: {return_code} ==="
                                f.write(f"{end_msg}\n")
                                st.session_state.logs.append(end_msg)
                                st.session_state.analysis_complete = False
                            f.flush()
                except Exception as e:
                    error_msg = f"写入结束标记失败: {str(e)}"
                    st.session_state.logs.append(error_msg)
                
                # 清理进程状态
                st.session_state.analysis_running = False
                st.session_state.analysis_process = None
                
                # 显示最终状态
                if return_code == 0:
                    st.success(f"🎉 分析成功完成！返回码: {return_code}")
                else:
                    st.error(f"❌ 分析失败，返回码: {return_code}")
                    
                # 最后刷新显示结束状态
                st.rerun()
            
        # 状态指示器
        if st.session_state.analysis_running:
            st.markdown(
                '<div class="warning-box">🔄 分析正在进行中，请等待...</div>', 
                unsafe_allow_html=True
            )
        elif st.session_state.analysis_complete:
            st.markdown(
                '<div class="success-box">✅ 分析已完成!</div>', 
                unsafe_allow_html=True
            )
        
        # 清空日志按钮
        if st.button("🗑️ 清空日志", width='stretch'):
            st.session_state.logs = []
            st.rerun()
    
    # 日志显示
    st.markdown("### 📝 运行日志")
    
    # 显示当前日志文件信息
    if st.session_state.log_file_path:
        log_file_name = os.path.basename(st.session_state.log_file_path)
        file_size = 0
        try:
            file_size = os.path.getsize(st.session_state.log_file_path) / 1024  # KB
        except:
            pass
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 日志文件: {log_file_name}")
        with col2:
            st.info(f"📊 文件大小: {file_size:.1f} KB")
        with col3:
            if st.button("📂 在文件夹中显示"):
                import subprocess
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.run(['explorer', '/select,', st.session_state.log_file_path])
                    else:  # Linux/Mac
                        subprocess.run(['xdg-open', os.path.dirname(st.session_state.log_file_path)])
                except:
                    st.error("无法打开文件夹")
    
    # 始终显示日志状态信息，无论是否有日志
    col1, col2, col3 = st.columns(3)
    with col1:
        log_count = len(st.session_state.logs) if st.session_state.logs else 0
        st.metric("总日志行数", log_count)
    with col2:
        if st.session_state.logs:
            error_count = len([log for log in st.session_state.logs if "ERROR" in log or "失败" in log])
            st.metric("错误数", error_count, delta="❌" if error_count > 0 else "✅")
        else:
            st.metric("错误数", 0, delta="✅")
    with col3:
        if st.session_state.analysis_running:
            st.metric("状态", "运行中", delta="🔄")
        elif st.session_state.analysis_complete:
            st.metric("状态", "已完成", delta="✅")
        else:
            st.metric("状态", "待运行", delta="⏸️")
        
        # 日志控制按钮
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            show_all_logs = st.checkbox("显示全部日志", value=False)
        with col2:
            if st.button("🔄 刷新日志"):
                # 强制重新读取日志文件
                if st.session_state.log_file_path and os.path.exists(st.session_state.log_file_path):
                    st.session_state.log_file_position = 0  # 重置读取位置
                    st.session_state.logs = []  # 清空当前日志
                    # 重新读取整个日志文件
                    all_logs = read_log_file_updates()
                    line_num = 0
                    for line in all_logs:
                        line_num += 1
                        st.session_state.logs.append(f"[{line_num:04d}] {line}")
                    st.session_state.last_log_check = line_num
                st.success("日志已刷新!")
                st.rerun()
        with col3:
            if st.button("🔍 检测进程状态"):
                if st.session_state.analysis_process:
                    process = st.session_state.analysis_process
                    return_code = process.poll()
                    program_completed_by_log = getattr(st.session_state, 'program_completion_detected', False)
                    
                    if return_code is not None:
                        # 进程已经结束但状态未更新
                        st.warning(f"⚠️ 检测到进程已结束 (返回码: {return_code})，正在更新状态...")
                    elif program_completed_by_log:
                        # 日志检测到完成但进程仍在运行
                        st.warning(f"⚠️ 日志显示程序已完成但进程仍在运行 (PID: {process.pid})，尝试结束进程...")
                        try:
                            process.terminate()
                            import time
                            time.sleep(2)
                            if process.poll() is None:
                                process.kill()
                            return_code = 0  # 假设成功完成
                        except:
                            return_code = 1
                        st.success("✅ 已根据日志检测结果结束进程")
                        
                        # 手动触发结束处理
                        import time
                        time.sleep(1)
                        
                        # 读取剩余日志
                        remaining_logs = read_log_file_updates()
                        if remaining_logs:
                            for line in remaining_logs:
                                st.session_state.last_log_check += 1
                                st.session_state.logs.append(f"[{st.session_state.last_log_check:04d}] {line}")
                        
                        # 写入结束标记
                        try:
                            if st.session_state.log_file_path:
                                with open(st.session_state.log_file_path, 'a', encoding='utf-8') as f:
                                    if return_code == 0:
                                        end_msg = f"=== 手动检测: 分析成功完成 (返回码: {return_code}) ==="
                                        st.session_state.analysis_complete = True
                                    else:
                                        end_msg = f"=== 手动检测: 分析失败，返回码: {return_code} ==="
                                        st.session_state.analysis_complete = False
                                    f.write(f"{end_msg}\n")
                                    st.session_state.logs.append(end_msg)
                                    f.flush()
                        except:
                            pass
                        
                        # 清理状态
                        st.session_state.analysis_running = False
                        st.session_state.analysis_process = None
                        st.success("✅ 进程状态已更新!")
                        st.rerun()
                    else:
                        st.info(f"🔄 进程仍在运行 (PID: {process.pid})")
                else:
                    st.info("ℹ️ 当前没有运行的进程")
        with col4:
            if st.button("📂 打开日志文件"):
                if st.session_state.log_file_path and os.path.exists(st.session_state.log_file_path):
                    import subprocess
                    try:
                        subprocess.run(['notepad.exe', st.session_state.log_file_path])
                    except:
                        st.error("无法打开日志文件")
        with col5:
            if st.button("🗑️ 清空", disabled=st.session_state.analysis_running):
                st.session_state.logs = []
                st.rerun()
        
        # 显示日志 - 始终显示日志框
        log_container = st.container()
        with log_container:
            # 处理日志显示逻辑
            if st.session_state.logs:
                # 有日志时的正常显示
                if show_all_logs:
                    display_logs = st.session_state.logs
                    max_height = 600
                else:
                    # 只显示最后100行日志以提高性能
                    display_logs = st.session_state.logs[-100:] if len(st.session_state.logs) > 100 else st.session_state.logs
                    max_height = 400
                
                log_text = '\n'.join(display_logs)
            else:
                # 没有日志时的显示
                if st.session_state.analysis_running:
                    log_text = "🔄 分析正在启动中...\n等待程序输出日志...\n\n如果长时间无输出，请检查:\n1. Python环境是否正确\n2. 数据文件路径是否存在\n3. 命令参数是否正确"
                else:
                    log_text = "📋 等待开始分析...\n\n点击上方 '🚀 开始分析' 按钮开始运行分析。\n日志将在这里实时显示。"
                
                max_height = 400
                display_logs = []
            
            # 使用更好的样式显示日志
            st.markdown("""
            <style>
            .log-text-area {
                font-family: 'Courier New', monospace !important;
                font-size: 12px !important;
                background-color: #0f0f0f !important;
                color: #00ff00 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # 如果有日志，进行高亮处理
            if display_logs:
                error_count = len([log for log in display_logs if "ERROR" in log or "失败" in log])
                # 处理日志文本，高亮不同类型的行
                highlighted_logs = []
                for log in display_logs:
                    if "ERROR" in log or "失败" in log:
                        highlighted_logs.append(f"🔴 {log}")
                    elif "WARNING" in log or "警告" in log:
                        highlighted_logs.append(f"🟡 {log}")
                    elif "成功" in log or "完成" in log:
                        highlighted_logs.append(f"🟢 {log}")
                    else:
                        highlighted_logs.append(log)
                log_text = '\n'.join(highlighted_logs)
            
            # 流式日志显示区域 - 使用稳定的key避免重复创建
            # 只有当日志数量变化时才更新key，避免过度刷新
            log_key = f"log_stream_stable_{len(st.session_state.logs)}"
            st.text_area(
                "📋 实时日志流", 
                value=log_text, 
                height=max_height,
                disabled=True,
                key=log_key,
                help="日志会自动实时更新，显示最新的分析进度"
            )
            
            # 实时状态栏
            if st.session_state.logs:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    latest_log = st.session_state.logs[-1]
                    if len(latest_log) > 40:
                        latest_log = latest_log[:40] + "..."
                    st.caption(f"🔄 最新: {latest_log}")
                
                with col2:
                    error_count = len([log for log in st.session_state.logs if any(k in log.lower() for k in ['error', '错误', 'fail'])])
                    if error_count > 0:
                        st.caption(f"❌ 错误: {error_count}")
                    else:
                        st.caption("✅ 无错误")
                
                with col3:
                    if st.session_state.log_file_path and os.path.exists(st.session_state.log_file_path):
                        try:
                            file_size = os.path.getsize(st.session_state.log_file_path) / 1024
                            st.caption(f"📄 大小: {file_size:.1f}KB")
                        except:
                            st.caption("📄 文件大小: 未知")
                
                with col4:
                    if st.session_state.analysis_running:
                        st.caption("🔄 运行中...")
                    elif st.session_state.analysis_complete:
                        st.caption("✅ 已完成")
                    else:
                        st.caption("⏸️ 待运行")
        
        # 下载日志功能
        col1, col2 = st.columns(2)
        with col1:
            if len(st.session_state.logs) > 0:
                log_content = '\n'.join(st.session_state.logs)
                import time  # 确保time模块可用
                st.download_button(
                    label="📥 下载界面日志",
                    data=log_content,
                    file_name=f"streamlit_log_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.session_state.log_file_path and os.path.exists(st.session_state.log_file_path):
                try:
                    with open(st.session_state.log_file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    st.download_button(
                        label="📥 下载完整日志文件",
                        data=file_content,
                        file_name=os.path.basename(st.session_state.log_file_path),
                        mime="text/plain"
                    )
                except:
                    st.warning("无法读取日志文件")
        
    # 如果分析正在运行，自动刷新
    if st.session_state.analysis_running:
        # 减少刷新频率，避免过度刷新影响显示
        import time
        time.sleep(1.5)  # 适中的刷新频率，确保显示稳定
        st.rerun()
    
    # 结果分析区域
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown("### 📈 结果可视化")
        
        # 检查结果文件
        results_dir = Path(config['output_dir'])
        if results_dir.exists():
            # 显示结果概览
            st.success("🎉 分析完成! 结果文件已生成。")
            
            # 结果文件浏览器
            st.markdown("#### 📂 生成的结果文件")
            
            for freq in config['frequency']:
                with st.expander(f"频率 {freq} 的结果"):
                    freq_dir = results_dir / freq.replace('.', '_')
                    if freq_dir.exists():
                        # 列出文件
                        files = list(freq_dir.rglob("*.png"))
                        if files:
                            st.success(f"找到 {len(files)} 个图像文件")
                            
                            # 显示部分图像
                            cols = st.columns(2)
                            for i, img_file in enumerate(files[:6]):  # 只显示前6个
                                with cols[i % 2]:
                                    try:
                                        st.image(str(img_file), caption=img_file.name, use_column_width=True)
                                    except:
                                        st.write(f"📄 {img_file.name}")
                        else:
                            st.warning("未找到图像文件")
                    else:
                        st.error(f"结果目录不存在: {freq_dir}")
        else:
            st.warning("结果目录不存在，可能分析未成功完成。")

if __name__ == "__main__":
    main()