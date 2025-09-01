#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-Time Monitor - 实时监控页面
监控分析进程的运行状态和资源使用情况
"""

import streamlit as st
import psutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from pathlib import Path
import json
from datetime import datetime

st.set_page_config(
    page_title="Real-Time Monitor",
    page_icon="🔍",
    layout="wide"
)

def get_system_info():
    """获取系统信息"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory': psutil.virtual_memory(),
        'disk': psutil.disk_usage('/'),
        'boot_time': datetime.fromtimestamp(psutil.boot_time())
    }

def find_python_processes():
    """查找Python相关进程"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'run.py' in cmdline or 'main.py' in cmdline or 'streamlit' in cmdline:
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def monitor_log_files():
    """监控日志文件变化"""
    log_files = []
    current_dir = Path('.')
    
    # 查找最近的日志文件
    for log_file in current_dir.glob('*.log'):
        try:
            stat = log_file.stat()
            log_files.append({
                'name': log_file.name,
                'size_mb': stat.st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'path': str(log_file)
            })
        except:
            pass
    
    return sorted(log_files, key=lambda x: x['modified'], reverse=True)

def check_gpu_status():
    """检查GPU状态"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                memory_total = props.total_memory / 1024**3  # GB
                
                gpu_info.append({
                    'id': i,
                    'name': props.name,
                    'memory_allocated': memory_allocated,
                    'memory_cached': memory_cached,
                    'memory_total': memory_total,
                    'utilization': (memory_allocated / memory_total) * 100
                })
            return gpu_info
        else:
            return []
    except ImportError:
        return None

def main():
    st.title("🔍 Real-Time System Monitor")
    st.markdown("实时监控系统资源和分析进程状态")
    
    # 创建更新控制
    auto_refresh = st.sidebar.checkbox("自动刷新", value=True)
    refresh_interval = st.sidebar.slider("刷新间隔(秒)", 1, 10, 3)
    
    if st.sidebar.button("🔄 立即刷新"):
        st.rerun()
    
    # 占位符用于动态更新
    system_placeholder = st.empty()
    process_placeholder = st.empty()
    gpu_placeholder = st.empty()
    logs_placeholder = st.empty()
    
    while True:
        with system_placeholder.container():
            st.subheader("💻 系统资源状态")
            
            system_info = get_system_info()
            
            # 创建指标行
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_color = "normal"
                if system_info['cpu_percent'] > 80:
                    cpu_color = "inverse"
                st.metric(
                    "CPU使用率", 
                    f"{system_info['cpu_percent']:.1f}%",
                    delta=f"{'⚠️ 高负载' if system_info['cpu_percent'] > 80 else '✅ 正常'}"
                )
            
            with col2:
                memory = system_info['memory']
                memory_percent = memory.percent
                st.metric(
                    "内存使用率",
                    f"{memory_percent:.1f}%",
                    delta=f"{memory.used // 1024**3:.1f}GB / {memory.total // 1024**3:.1f}GB"
                )
            
            with col3:
                disk = system_info['disk']
                disk_percent = (disk.used / disk.total) * 100
                st.metric(
                    "磁盘使用率",
                    f"{disk_percent:.1f}%",
                    delta=f"{disk.free // 1024**3:.1f}GB 可用"
                )
            
            with col4:
                uptime = datetime.now() - system_info['boot_time']
                st.metric(
                    "系统运行时间",
                    f"{uptime.days}天",
                    delta=f"{uptime.seconds // 3600}小时"
                )
            
            # 系统资源图表
            if st.checkbox("显示资源趋势图"):
                # 这里可以添加历史数据收集和趋势图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['CPU使用率', '内存使用率', '磁盘I/O', '网络I/O'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # 模拟数据点
                x_data = list(range(10))
                fig.add_trace(
                    go.Scatter(x=x_data, y=[system_info['cpu_percent']] * 10, 
                              name='CPU', line=dict(color='red')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=x_data, y=[memory_percent] * 10, 
                              name='Memory', line=dict(color='blue')),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with process_placeholder.container():
            st.subheader("🐍 Python进程监控")
            
            processes = find_python_processes()
            
            if processes:
                import pandas as pd
                df = pd.DataFrame(processes)
                
                # 进程统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("活跃Python进程", len(processes))
                with col2:
                    total_cpu = sum(p['cpu_percent'] for p in processes)
                    st.metric("总CPU使用", f"{total_cpu:.1f}%")
                with col3:
                    total_memory = sum(p['memory_mb'] for p in processes)
                    st.metric("总内存使用", f"{total_memory:.1f}MB")
                
                # 进程详情表
                st.dataframe(
                    df.style.format({
                        'cpu_percent': '{:.1f}%',
                        'memory_mb': '{:.1f}MB'
                    }),
                    use_container_width=True
                )
                
                # 如果发现运行分析的进程
                analysis_processes = [p for p in processes if 'run.py' in p['cmdline']]
                if analysis_processes:
                    st.success(f"🟢 检测到 {len(analysis_processes)} 个分析进程正在运行")
                    for proc in analysis_processes:
                        with st.expander(f"进程 {proc['pid']} 详情"):
                            st.code(proc['cmdline'])
                            st.text(f"CPU: {proc['cpu_percent']:.1f}% | 内存: {proc['memory_mb']:.1f}MB")
            else:
                st.info("没有检测到相关的Python进程")
        
        with gpu_placeholder.container():
            st.subheader("🎮 GPU状态监控")
            
            gpu_info = check_gpu_status()
            
            if gpu_info is None:
                st.warning("PyTorch未安装，无法检测GPU状态")
            elif not gpu_info:
                st.info("未检测到可用的CUDA GPU")
            else:
                for gpu in gpu_info:
                    with st.expander(f"GPU {gpu['id']}: {gpu['name']}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("显存使用", f"{gpu['utilization']:.1f}%")
                        with col2:
                            st.metric("已分配显存", f"{gpu['memory_allocated']:.2f}GB")
                        with col3:
                            st.metric("总显存", f"{gpu['memory_total']:.2f}GB")
                        
                        # 显存使用条形图
                        fig = go.Figure(go.Bar(
                            x=['已分配', '已缓存', '空闲'],
                            y=[gpu['memory_allocated'], 
                               gpu['memory_cached'] - gpu['memory_allocated'],
                               gpu['memory_total'] - gpu['memory_cached']],
                            marker_color=['red', 'orange', 'green']
                        ))
                        fig.update_layout(
                            title=f"GPU {gpu['id']} 显存分布",
                            yaxis_title="显存 (GB)",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with logs_placeholder.container():
            st.subheader("📋 日志文件监控")
            
            log_files = monitor_log_files()
            
            if log_files:
                for log_file in log_files[:5]:  # 只显示前5个
                    with st.expander(f"📄 {log_file['name']} ({log_file['size_mb']:.2f}MB)"):
                        st.text(f"最后修改: {log_file['modified']}")
                        st.text(f"文件大小: {log_file['size_mb']:.2f}MB")
                        
                        # 显示最后几行
                        try:
                            with open(log_file['path'], 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                if lines:
                                    st.text_area(
                                        "最后10行:",
                                        ''.join(lines[-10:]),
                                        height=200
                                    )
                        except:
                            st.error("无法读取日志文件")
            else:
                st.info("当前目录下没有发现日志文件")
        
        # 如果不是自动刷新模式，跳出循环
        if not auto_refresh:
            break
        
        # 等待指定时间后刷新
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()