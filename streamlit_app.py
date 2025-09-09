#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCS POD Analysis - Clean Streamlit Interface
清理版本，移除所有过时的代码逻辑
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
import threading
import queue
from pathlib import Path
import sys
from datetime import datetime
import html

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
    .log-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.375rem;
        padding: 1rem;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        max-height: 500px;
        overflow-y: auto;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .log-container::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    .log-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .log-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .log-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
def init_session_state():
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
    if config['pod_enabled']:
        cmd.extend(['--pod_modes', ','.join(map(str, config['pod_modes']))])
        cmd.extend(['--energy_threshold', str(config['energy_threshold'])])
        cmd.extend(['--num_modes_visualize', str(config['num_modes_visualize'])])
        cmd.extend(['--pod_reconstruct_num', str(config['pod_reconstruct_num'])])
    
    # Autoencoder参数
    if config['ae_enabled']:
        cmd.extend(['--latent_dims', ','.join(map(str, config['latent_dims']))])
        cmd.extend(['--model_types', ','.join(config['model_types'])])
        cmd.extend(['--ae_epochs', str(config['ae_epochs'])])
        cmd.extend(['--ae_device', config['ae_device']])
        cmd.extend(['--ae_learning_rate', str(config['ae_learning_rate'])])
        cmd.extend(['--ae_batch_size', str(config['ae_batch_size'])])
        
        if config['skip_ae_training']:
            cmd.append('--skip_ae_training')
    
    return cmd

def start_analysis(config):
    """启动分析进程"""
    try:
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
        
        return True, "分析已启动"
        
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

def main():
    """主函数"""
    init_session_state()
    
    # 页面标题
    st.markdown('<h1 class="main-header">📡 RCS POD分析系统</h1>', 
                unsafe_allow_html=True)
    
    # 加载配置
    config = load_config()
    
    # 侧边栏配置
    st.sidebar.markdown("## ⚙️ 分析配置")
    
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
    
    # 保存配置按钮
    if st.sidebar.button("💾 保存配置"):
        save_config(config)
        st.sidebar.success("配置已保存!")
    
    # 算法参数配置区域
    st.markdown("---")
    st.markdown("### ⚙️ 详细参数配置")
    
    # 创建主布局：左侧参数配置，右侧实时日志
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
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
            
                # 能量阈值
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
    
        # 配置概览和验证
        st.markdown("---")
        st.markdown("### 📋 配置概览")
        
        # 创建配置概览
        overview_col1, overview_col2 = st.columns(2)
    
        with overview_col1:
            st.markdown("**基础配置**")
            st.write(f"📁 参数文件: `{os.path.basename(config['params_path'])}`")
            st.write(f"📂 RCS数据目录: `{os.path.basename(config['rcs_dir'])}`")
            st.write(f"📤 输出目录: `{config['output_dir']}`")
            st.write(f"🔧 频率: {', '.join(config['frequency'])}")
            st.write(f"🔢 模型数量: {config['num_models']}")
            st.write(f"🎯 训练集大小: {config['num_train']}")
            
        with overview_col2:
            st.markdown("**算法配置**")
            
            if config['pod_enabled']:
                st.write(f"📐 POD分析: ✅ 启用")
                st.write(f"  - 多模态对比: {config['pod_modes']}")
                st.write(f"  - 能量阈值: {config['energy_threshold']}%")
                st.write(f"  - 可视化模态数: {config['num_modes_visualize']}")
            else:
                st.write(f"📐 POD分析: ❌ 禁用")
                
            if config['ae_enabled']:
                st.write(f"🔬 Autoencoder分析: ✅ 启用")
                st.write(f"  - 隐空间维度: {config['latent_dims']}")
                st.write(f"  - 模型类型: {config['model_types']}")
                st.write(f"  - 训练轮数: {config['ae_epochs']}")
                st.write(f"  - 学习率: {config['ae_learning_rate']}")
                st.write(f"  - 计算设备: {config['ae_device']}")
                if config['skip_ae_training']:
                    st.write(f"  - 跳过重训练: ✅")
            else:
                st.write(f"🔬 Autoencoder分析: ❌ 禁用")
    
        # 生成的命令预览
        if st.expander("🔍 查看生成的命令", expanded=False):
            cmd = run_analysis_command(config)
            cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
            st.code(cmd_str, language='bash')
    
        # 主要控制区域
        st.markdown("---")
    
        # 控制按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.analysis_running:
                if st.button("▶️ 开始分析", type="primary", use_container_width=True):
                    success, message = start_analysis(config)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.button("⏳ 分析进行中...", disabled=True, use_container_width=True)
        
        with col2:
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
    
    # 右侧：实时日志和状态监控
    with main_col2:
        st.markdown("### 📋 实时日志")
        
        # 实时更新日志和进程状态
        if st.session_state.analysis_running:
            process_status = update_logs_from_queue()
            
            # 检查进程是否结束
            if process_status is not None:
                st.session_state.analysis_running = False
                st.session_state.analysis_complete = True
                st.session_state.analysis_process = None
                st.session_state.log_reader_thread = None
                
                # 显示完成状态
                if process_status == 0:
                    st.success("🎉 分析成功完成!")
                else:
                    st.error(f"❌ 分析失败，返回码: {process_status}")
                
                st.rerun()
        
        # 运行状态显示
        if st.session_state.analysis_running:
            if hasattr(st.session_state, 'analysis_start_time'):
                elapsed_time = time.time() - st.session_state.analysis_start_time
                elapsed_minutes = elapsed_time / 60
                st.info(f"🔄 分析进行中... (已运行 {elapsed_minutes:.1f} 分钟)")
        elif st.session_state.analysis_complete:
            st.success("✅ 分析已完成!")
        else:
            st.info("⏳ 等待开始分析...")
        
        # 日志控制
        log_col1, log_col2 = st.columns(2)
        with log_col1:
            show_all_logs = st.checkbox("显示全部日志", value=False)
        with log_col2:
            if st.button("🗑️ 清空日志", disabled=st.session_state.analysis_running):
                st.session_state.logs = []
                st.rerun()
        
        st.metric("日志行数", len(st.session_state.logs))
        
        # 日志内容显示
        if st.session_state.logs:
            display_logs = st.session_state.logs if show_all_logs else st.session_state.logs[-50:]
            log_text = '\n'.join(display_logs)
        else:
            if st.session_state.analysis_running:
                log_text = "🔄 分析正在启动中，等待程序输出..."
            else:
                log_text = "📋 等待开始分析..."
        
        # 使用自定义样式显示日志，支持滚动条
        log_text_escaped = html.escape(log_text)
        st.markdown(f'<div class="log-container" id="log-container">{log_text_escaped}</div>', 
                   unsafe_allow_html=True)
        
        # 添加JavaScript来自动滚动到底部
        if st.session_state.analysis_running and st.session_state.logs:
            st.markdown("""
            <script>
            setTimeout(function() {
                var logContainer = document.getElementById('log-container');
                if (logContainer) {
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            }, 100);
            </script>
            """, unsafe_allow_html=True)
        
        # 自动刷新逻辑
        if st.session_state.analysis_running:
            # 自动刷新
            time.sleep(2)
            st.rerun()

if __name__ == "__main__":
    main()