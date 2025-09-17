#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCS分析统一平台 - 基于原始streamlit_app.py的扩展版本
集成POD/AE分析和FiLM-UNet深度学习，保持完全相同的UI结构
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import subprocess
import time
import threading
import queue
from pathlib import Path
import sys
from datetime import datetime
import html

# 配置页面
st.set_page_config(
    page_title="RCS Analysis Dashboard",
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
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
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
    .log-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.375rem;
        padding: 1rem;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """初始化会话状态"""
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'log_queue' not in st.session_state:
        st.session_state.log_queue = queue.Queue()
    if 'log_reader_thread' not in st.session_state:
        st.session_state.log_reader_thread = None
    if 'analysis_process' not in st.session_state:
        st.session_state.analysis_process = None
    if 'selected_method' not in st.session_state:
        st.session_state.selected_method = "POD/AE分析"

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
        'pod_modes': [10, 20, 30, 40],
        'energy_threshold': 95.0,
        'num_modes_visualize': 10,
        'pod_reconstruct_num': 0,
        'ae_epochs': 200,
        'ae_device': 'auto',
        'ae_learning_rate': 0.001,
        'ae_batch_size': 0
    }

def load_default_unet_config():
    """加载默认UNet配置"""
    return {
        'params_path': '../parameter/parameters_sorted.csv',
        'rcs_dir': '../parameter/csv_output',
        'output_dir': './unet_outputs',
        'frequency': ['1.5G'],  # UNet与AE统一使用频率列表 (支持多频率: ['1.5G', '3G'])
        'num_models': 80,     # UNet使用的模型数量
        'run_name': '',       # 自定义运行名称
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.001,
        'device': 'auto',
        'lambda_mse': 1.0,
        'lambda_smooth': 0.01,
        'lambda_physics': 0.05,
        'lambda_multiscale': 0.1,
        'enable_augmentation': True,
        'noise_std': 0.01,
        'mixup_alpha': 0.2
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

def log_reader_worker(process, log_queue):
    """后台线程读取进程输出"""
    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                timestamp = datetime.now().strftime('%H:%M:%S')
                log_entry = f"[{timestamp}] {output.strip()}"
                log_queue.put(('log', log_entry))
        
        return_code = process.poll()
        log_queue.put(('process_finished', return_code))
        
    except Exception as e:
        log_queue.put(('error', f"日志读取错误: {str(e)}"))

def update_logs_from_queue():
    """从队列更新日志"""
    if 'log_queue' not in st.session_state:
        return None
        
    new_logs = []
    process_status = None
    
    try:
        while not st.session_state.log_queue.empty():
            try:
                msg_type, content = st.session_state.log_queue.get_nowait()
                
                if msg_type == 'log':
                    new_logs.append(content)
                elif msg_type == 'process_finished':
                    process_status = content
                    new_logs.append(f"=== 分析进程结束，返回码: {content} ===")
                elif msg_type == 'error':
                    new_logs.append(f"❌ {content}")
                    
            except:
                break
    except:
        pass
    
    if new_logs:
        st.session_state.logs.extend(new_logs)
        if len(st.session_state.logs) > 1000:
            st.session_state.logs = st.session_state.logs[-800:]
    
    return process_status

def run_analysis_command(config):
    """生成分析命令"""
    python_executable = sys.executable
    cmd = [python_executable, 'run.py']
    cmd.extend(['--params_path', config['params_path']])
    cmd.extend(['--rcs_dir', config['rcs_dir']])
    cmd.extend(['--output_dir', config['output_dir']])
    
    # 处理频率参数
    frequencies = config['frequency']
    if len(frequencies) == 2 and '1.5G' in frequencies and '3G' in frequencies:
        cmd.extend(['--freq', 'both'])
    elif len(frequencies) == 1:
        cmd.extend(['--freq', frequencies[0]])
    
    cmd.extend(['--num_models', str(config['num_models'])])
    cmd.extend(['--num_train', ','.join(map(str, config['num_train']))])
    
    # POD参数
    if config.get('pod_enabled', True):
        cmd.extend(['--pod_modes', ','.join(map(str, config['pod_modes']))])
        cmd.extend(['--energy_threshold', str(config['energy_threshold'])])
        cmd.extend(['--num_modes_visualize', str(config['num_modes_visualize'])])
        cmd.extend(['--pod_reconstruct_num', str(config['pod_reconstruct_num'])])
    
    # Autoencoder参数
    if config.get('ae_enabled', True):
        cmd.extend(['--latent_dims', ','.join(map(str, config['latent_dims']))])
        cmd.extend(['--model_types', ','.join(config['model_types'])])
        cmd.extend(['--ae_epochs', str(config['ae_epochs'])])
        cmd.extend(['--ae_device', config['ae_device']])
        cmd.extend(['--ae_learning_rate', str(config['ae_learning_rate'])])
        cmd.extend(['--ae_batch_size', str(config['ae_batch_size'])])
        
        if config.get('skip_ae_training', False):
            cmd.append('--skip_ae_training')
    
    return cmd

def start_analysis(config):
    """启动分析进程"""
    try:
        if st.session_state.selected_method == "FiLM-UNet训练":
            # FiLM-UNet训练命令 - 使用实际存在的训练脚本
            python_executable = sys.executable
            cmd = [python_executable, 'unet_model/main.py']
            
            # 添加数据参数
            data_dir = '/'.join(config['params_path'].split('/')[:-1])  # 从参数路径提取目录
            params_file = config['params_path'].split('/')[-1]  # 参数文件名
            
            cmd.extend(['--data_dir', data_dir])
            cmd.extend(['--params_file', params_file])
            cmd.extend(['--rcs_dir', os.path.basename(config['rcs_dir'])])
            cmd.extend(['--num_models', str(config['num_models'])])
            
            # 处理频率参数 (UNet现在支持多频率训练)
            if config.get('frequency') and len(config['frequency']) > 0:
                if len(config['frequency']) == 1:
                    # 单频率训练
                    frequency = config['frequency'][0]
                    cmd.extend(['--frequency', frequency])
                else:
                    # 多频率训练 - 使用frequencies参数
                    cmd.extend(['--frequencies'] + config['frequency'])
            
            # 测试集比例参数已被训练集绝对大小替代，无需添加
            
            # 训练参数
            cmd.extend(['--batch_size', str(config.get('batch_size', 16))])
            cmd.extend(['--epochs', str(config.get('epochs', 300))])
            cmd.extend(['--learning_rate', str(config.get('learning_rate', 0.001))])
            
            # 训练集大小 (使用绝对大小，与AE方法统一)
            if config['num_train']:
                cmd.extend(['--train_size', str(config['num_train'][0])])
            
            # 损失函数权重
            cmd.extend(['--lambda_mse', str(config.get('lambda_mse', 1.0))])
            cmd.extend(['--lambda_smooth', str(config.get('lambda_smooth', 0.01))])
            cmd.extend(['--lambda_physics', str(config.get('lambda_physics', 0.05))])
            cmd.extend(['--lambda_multiscale', str(config.get('lambda_multiscale', 0.1))])
            
            # 输出目录
            cmd.extend(['--output_dir', config['output_dir']])
            
            # 自定义运行名称
            if config.get('run_name') and config['run_name'].strip():
                cmd.extend(['--run_name', config['run_name'].strip()])
        else:
            # POD/AE分析命令
            cmd = run_analysis_command(config)
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        
        # 启动进程
        process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        # 启动日志读取线程
        log_thread = threading.Thread(
            target=log_reader_worker, 
            args=(process, st.session_state.log_queue),
            daemon=True
        )
        log_thread.start()
        
        # 保存状态
        st.session_state.analysis_process = process
        st.session_state.log_reader_thread = log_thread
        st.session_state.analysis_running = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_start_time = time.time()
        
        # 添加启动日志
        cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.logs.append(f"=== 分析开始于 {start_time} ===")
        st.session_state.logs.append(f"执行命令: {cmd_str}")
        st.session_state.logs.append("=== 等待程序输出 ===")
        
        return True, f"{st.session_state.selected_method}已启动"
        
    except Exception as e:
        return False, f"启动失败: {str(e)}"

def stop_analysis():
    """停止分析进程"""
    try:
        if st.session_state.analysis_process and st.session_state.analysis_process.poll() is None:
            st.session_state.analysis_process.terminate()
            time.sleep(1)
            if st.session_state.analysis_process.poll() is None:
                st.session_state.analysis_process.kill()
            
            st.session_state.logs.append("=== 分析已被手动停止 ===")
        
        st.session_state.analysis_running = False
        st.session_state.analysis_process = None
        st.session_state.log_reader_thread = None
        
        return True, "分析已停止"
        
    except Exception as e:
        return False, f"停止失败: {str(e)}"

def unet_detail_config(config):
    """UNet详细配置界面"""
    
    # 数据参数配置
    st.markdown("#### 📊 数据参数")
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        config['num_models'] = st.number_input(
            "使用模型数量", 
            min_value=10, 
            max_value=100, 
            value=config.get('num_models', 80),
            help="用于训练的模型总数量"
        )
        
    with data_col2:
        st.info("📡 训练频率和训练集大小由左侧栏统一设置")
        st.success("✅ UNet现已支持多频率训练 (1.5G + 3G)")
    
    # 自定义运行名称
    st.markdown("#### 📁 运行配置")
    config['run_name'] = st.text_input(
        "自定义运行名称", 
        value=config.get('run_name', ''),
        placeholder="输入自定义运行名称 (留空使用时间戳)",
        help="自定义名称将用作输出文件夹名称，如: my_experiment"
    )
    
    st.markdown("#### 🏋️ 训练参数")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        config['batch_size'] = st.number_input(
            "批大小", 
            min_value=4, 
            max_value=64, 
            value=config.get('batch_size', 16)
        )
        
        config['epochs'] = st.number_input(
            "训练轮数", 
            min_value=50, 
            max_value=1000, 
            value=config.get('epochs', 300)
        )
    
    with col2:
        config['learning_rate'] = st.number_input(
            "学习率", 
            min_value=0.0001, 
            max_value=0.01, 
            value=config.get('learning_rate', 0.001),
            format="%.4f"
        )
        
        config['device'] = st.selectbox(
            "训练设备",
            options=['auto', 'cpu', 'cuda'],
            index=['auto', 'cpu', 'cuda'].index(config.get('device', 'auto'))
        )
    
    with col3:
        config['enable_augmentation'] = st.checkbox(
            "启用数据增强", 
            value=config.get('enable_augmentation', True)
        )
        
        if config['enable_augmentation']:
            config['noise_std'] = st.number_input(
                "噪声标准差", 
                min_value=0.001, 
                max_value=0.1, 
                value=config.get('noise_std', 0.01),
                format="%.3f"
            )
    
    # 损失函数权重
    st.markdown("#### ⚖️ 损失函数权重")
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        config['lambda_mse'] = st.text_input(
            "MSE权重", 
            value=str(config.get('lambda_mse', 1.0)),
            help="输入数值，如: 1.0"
        )
        # 转换为float
        try:
            config['lambda_mse'] = float(config['lambda_mse'])
        except:
            config['lambda_mse'] = 1.0
            st.error("MSE权重必须是数字")
    
    with col5:
        config['lambda_smooth'] = st.text_input(
            "平滑权重", 
            value=str(config.get('lambda_smooth', 0.01)),
            help="输入数值，如: 0.01"
        )
        try:
            config['lambda_smooth'] = float(config['lambda_smooth'])
        except:
            config['lambda_smooth'] = 0.01
            st.error("平滑权重必须是数字")
    
    with col6:
        config['lambda_physics'] = st.text_input(
            "物理权重", 
            value=str(config.get('lambda_physics', 0.05)),
            help="输入数值，如: 0.05"
        )
        try:
            config['lambda_physics'] = float(config['lambda_physics'])
        except:
            config['lambda_physics'] = 0.05
            st.error("物理权重必须是数字")
    
    with col7:
        config['lambda_multiscale'] = st.text_input(
            "多尺度权重", 
            value=str(config.get('lambda_multiscale', 0.1)),
            help="输入数值，如: 0.1"
        )
        try:
            config['lambda_multiscale'] = float(config['lambda_multiscale'])
        except:
            config['lambda_multiscale'] = 0.1
            st.error("多尺度权重必须是数字")

def main():
    """主函数"""
    init_session_state()
    
    # 页面标题
    st.markdown('<h1 class="main-header">📡 RCS分析统一平台</h1>', 
                unsafe_allow_html=True)
    
    # 根据选择的方法加载配置
    if st.session_state.selected_method == "FiLM-UNet训练":
        config = load_config("unet_config.json")
        if not config or 'batch_size' not in config:
            config = load_default_unet_config()
    else:
        config = load_config("streamlit_config.json")
        if not config or 'pod_modes' not in config:
            config = load_default_config()
    
    # 侧边栏配置 - 与原始UI完全相同
    st.sidebar.markdown("## ⚙️ 分析配置")
    
    # 方法选择
    st.sidebar.markdown("### 🔬 方法选择")
    method_options = ["POD/AE分析", "FiLM-UNet训练"]
    
    selected_method = st.sidebar.radio(
        "选择分析方法:",
        options=method_options,
        index=method_options.index(st.session_state.selected_method)
    )
    
    if selected_method != st.session_state.selected_method:
        st.session_state.selected_method = selected_method
        st.rerun()
    
    # 基础参数配置 - 完全复用左侧栏参数
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
        value=config.get('output_dir', './results' if selected_method == "POD/AE分析" else './unet_outputs')
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
        max_value=200, 
        value=config.get('num_models', 100)
    )
    
    # 训练集大小 - 统一格式
    train_sizes_str = st.sidebar.text_input(
        "训练集大小 (逗号分隔)", 
        value=','.join(map(str, config.get('num_train', [80])))
    )
    try:
        config['num_train'] = [int(x.strip()) for x in train_sizes_str.split(',')]
    except:
        config['num_train'] = [80]
    
    # 算法配置
    if selected_method == "POD/AE分析":
        st.sidebar.markdown("### 🧠 算法配置")
        config['pod_enabled'] = st.sidebar.checkbox("启用POD分析", value=config.get('pod_enabled', True))
        config['ae_enabled'] = st.sidebar.checkbox("启用Autoencoder分析", value=config.get('ae_enabled', True))
        config['skip_ae_training'] = st.sidebar.checkbox("跳过AE重训练", value=config.get('skip_ae_training', False))
    
    
    # 主界面详细配置部分
    st.markdown("---")
    st.markdown("### ⚙️ 详细参数配置")
    
    # 创建主布局：左侧参数配置，右侧实时日志
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
        if selected_method == "POD/AE分析":
            # POD/AE详细配置 (从原始UI复制)
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                if config['pod_enabled']:
                    st.markdown("#### 📐 POD分析参数")
                    pod_modes_str = st.text_input(
                        "POD多模态对比 (逗号分隔)", 
                        value=','.join(map(str, config.get('pod_modes', [10, 20, 30, 40])))
                    )
                    try:
                        config['pod_modes'] = [int(x.strip()) for x in pod_modes_str.split(',')]
                    except:
                        config['pod_modes'] = [10, 20, 30, 40]
                    
                    config['energy_threshold'] = st.number_input(
                        "能量阈值 (%)", 
                        min_value=80.0, 
                        max_value=99.9, 
                        value=config.get('energy_threshold', 95.0)
                    )
                    
                    config['num_modes_visualize'] = st.number_input(
                        "可视化模态数", 
                        min_value=1, 
                        max_value=50, 
                        value=config.get('num_modes_visualize', 10)
                    )
                    
                    config['pod_reconstruct_num'] = st.number_input(
                        "重建分析模态数", 
                        min_value=0, 
                        max_value=100, 
                        value=config.get('pod_reconstruct_num', 0)
                    )
            
            with param_col2:
                if config['ae_enabled']:
                    st.markdown("#### 🔬 Autoencoder参数")
                    latent_dims_str = st.text_input(
                        "隐空间维度 (逗号分隔)", 
                        value=','.join(map(str, config.get('latent_dims', [5, 10, 15, 20])))
                    )
                    try:
                        config['latent_dims'] = [int(x.strip()) for x in latent_dims_str.split(',')]
                    except:
                        config['latent_dims'] = [5, 10, 15, 20]
                    
                    config['model_types'] = st.multiselect(
                        "模型类型",
                        options=['standard', 'vae'],
                        default=config.get('model_types', ['standard', 'vae'])
                    )
                    
                    config['ae_epochs'] = st.number_input(
                        "训练轮数", 
                        min_value=10, 
                        max_value=1000, 
                        value=config.get('ae_epochs', 200)
                    )
                    
                    config['ae_learning_rate'] = st.number_input(
                        "学习率", 
                        min_value=0.0001, 
                        max_value=0.01, 
                        value=config.get('ae_learning_rate', 0.001),
                        format="%.4f"
                    )
        else:
            # FiLM-UNet详细配置
            unet_detail_config(config)
        
        # 配置保存命名
        st.markdown("---")
        st.markdown("#### 📝 配置保存设置")
        naming_col1, naming_col2 = st.columns([2, 1])
        
        with naming_col1:
            custom_config_name = st.text_input(
                "自定义配置文件名", 
                value="",
                placeholder="输入自定义配置名称 (留空使用默认)",
                help="输入自定义名称将保存为 '{name}_config.json'"
            )
        
        with naming_col2:
            timestamp_suffix = st.checkbox("添加时间戳", value=False, help="在文件名后添加时间戳")
        
        # 控制按钮
        st.markdown("---")
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if not st.session_state.analysis_running:
                button_text = "▶️ 开始POD/AE分析" if selected_method == "POD/AE分析" else "▶️ 开始UNet训练"
                if st.button(button_text, type="primary", use_container_width=True):
                    success, message = start_analysis(config)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.button("⏳ 运行中...", disabled=True, use_container_width=True)
        
        with control_col2:
            if st.session_state.analysis_running:
                if st.button("⏹️ 停止分析", type="secondary", use_container_width=True):
                    success, message = stop_analysis()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.rerun()
            else:
                st.button("⏹️ 停止分析", disabled=True, use_container_width=True)
        
        with control_col3:
            if st.button("💾 保存详细配置", use_container_width=True):
                # 构建配置文件名
                if custom_config_name.strip():
                    # 使用自定义名称
                    base_name = custom_config_name.strip()
                    # 移除可能的.json后缀
                    if base_name.endswith('.json'):
                        base_name = base_name[:-5]
                else:
                    # 使用默认名称
                    base_name = "unet_config" if selected_method == "FiLM-UNet训练" else "streamlit_config"
                
                # 添加时间戳 (如果选择)
                if timestamp_suffix:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    config_filename = f"{base_name}_{timestamp}_config.json"
                else:
                    config_filename = f"{base_name}_config.json"
                
                save_config(config, config_filename)
                st.success(f"✅ 详细配置已保存为: {config_filename}")
    
    with main_col2:
        # 实时日志显示
        st.markdown("#### 📋 实时日志")
        
        if st.session_state.analysis_running:
            process_status = update_logs_from_queue()
            # 检查进程是否结束
            if process_status is not None:
                st.session_state.analysis_running = False
                if process_status == 0:
                    st.success("✅ 分析完成！")
                else:
                    st.error(f"❌ 分析失败，返回码: {process_status}")
                st.rerun()
        
        # 日志控制
        log_col1, log_col2, log_col3 = st.columns(3)
        
        with log_col1:
            show_all_logs = st.checkbox("显示全部日志", value=False)
        
        with log_col2:
            if st.button("🔄 刷新"):
                st.rerun()
        
        with log_col3:
            if st.button("🗑️ 清空", disabled=st.session_state.analysis_running):
                st.session_state.logs = []
                st.rerun()
        
        # 日志内容
        if st.session_state.logs:
            if show_all_logs:
                log_text = '\n'.join(st.session_state.logs)  # 显示全部日志
            else:
                log_text = '\n'.join(st.session_state.logs[-50:])  # 显示最后50行
        else:
            if st.session_state.analysis_running:
                log_text = "等待程序输出..."
            else:
                log_text = "准备开始..."
        
        st.markdown(
            f'<div class="log-container">{html.escape(log_text)}</div>',
            unsafe_allow_html=True
        )
    
    # 自动刷新
    if st.session_state.analysis_running:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()