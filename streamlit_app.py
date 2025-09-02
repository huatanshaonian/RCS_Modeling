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
import threading
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

def run_analysis(config):
    """运行分析的后台函数"""
    try:
        # 添加初始日志
        st.session_state.logs.append("=== 开始分析 ===")
        st.session_state.logs.append(f"当前工作目录: {os.getcwd()}")
        
        # 构建命令
        cmd = ['python', 'run.py']
        cmd.extend(['--params_path', config['params_path']])
        cmd.extend(['--rcs_dir', config['rcs_dir']])
        cmd.extend(['--output_dir', config['output_dir']])
        cmd.extend(['--freq', ','.join(config['frequency'])])
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
        
        # 记录命令
        cmd_str = ' '.join(cmd)
        st.session_state.logs.append(f"执行命令: {cmd_str}")
        st.session_state.logs.append("--- 分析程序输出 ---")
        
        # 运行命令
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.getcwd()  # 明确指定工作目录
        )
        
        # 实时读取输出
        line_count = 0
        for line in process.stdout:
            line_count += 1
            clean_line = line.strip()
            if clean_line:  # 只添加非空行
                st.session_state.logs.append(f"[{line_count:04d}] {clean_line}")
            
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            st.session_state.logs.append("=== 分析成功完成 ===")
            st.session_state.analysis_complete = True
        else:
            st.session_state.logs.append(f"=== 分析失败，返回码: {return_code} ===")
            
        st.session_state.analysis_running = False
        
    except FileNotFoundError:
        st.session_state.logs.append("错误: 找不到python或run.py文件")
        st.session_state.logs.append("请确认您在正确的目录中运行此程序")
        st.session_state.analysis_running = False
    except Exception as e:
        st.session_state.logs.append(f"分析过程中发生错误: {str(e)}")
        import traceback
        st.session_state.logs.append(f"详细错误信息: {traceback.format_exc()}")
        st.session_state.analysis_running = False

def main():
    # 主标题
    st.markdown('<h1 class="main-header">📡 RCS POD Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
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
        
        # 运行按钮
        if not st.session_state.analysis_running:
            if st.button("▶️ 开始分析", type="primary", use_container_width=True):
                # 验证配置
                if not config['frequency']:
                    st.error("请至少选择一个频率!")
                elif config['ae_enabled'] and not config['model_types']:
                    st.error("启用Autoencoder时请至少选择一个模型类型!")
                else:
                    st.session_state.analysis_running = True
                    st.session_state.analysis_complete = False
                    st.session_state.logs = []
                    # 启动分析线程
                    thread = threading.Thread(target=run_analysis, args=(config,))
                    thread.daemon = True
                    thread.start()
                    st.rerun()
        else:
            st.button("⏳ 分析进行中...", disabled=True, use_container_width=True)
            
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
        if st.button("🗑️ 清空日志", use_container_width=True):
            st.session_state.logs = []
            st.rerun()
    
    # 日志显示
    st.markdown("### 📝 运行日志")
    
    if st.session_state.logs:
        # 显示最新的日志
        log_container = st.container()
        with log_container:
            # 只显示最后50行日志
            recent_logs = st.session_state.logs[-50:] if len(st.session_state.logs) > 50 else st.session_state.logs
            log_text = '\n'.join(recent_logs)
            st.text_area(
                "实时日志输出", 
                value=log_text, 
                height=300,
                disabled=True
            )
            
        # 自动滚动到底部
        if st.session_state.analysis_running:
            time.sleep(1)
            st.rerun()
    else:
        st.info("📋 运行日志将在这里显示...")
    
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