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
    try:
        # 获取CPU和内存信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        # 安全地获取磁盘信息
        disk_info = None
        if os.name == 'nt':
            # Windows: 尝试多个常见驱动器
            for drive in ['C:', 'D:', 'E:', 'F:', 'G:']:
                try:
                    if os.path.exists(drive + os.sep):
                        disk_info = psutil.disk_usage(drive + os.sep)
                        break
                except:
                    continue
            
            # 如果都失败了，尝试当前目录的驱动器
            if disk_info is None:
                try:
                    current_drive = os.path.splitdrive(os.getcwd())[0]
                    if current_drive:
                        disk_info = psutil.disk_usage(current_drive + os.sep)
                except:
                    pass
        else:
            # Linux/Mac: 使用根目录
            try:
                disk_info = psutil.disk_usage('/')
            except:
                pass
        
        # 如果磁盘信息获取失败，使用默认值
        if disk_info is None:
            disk_info = type('obj', (object,), {
                'total': 1024*1024*1024*100,  # 100GB
                'used': 1024*1024*1024*50,    # 50GB
                'free': 1024*1024*1024*50     # 50GB
            })()
        
        return {
            'cpu_percent': cpu_percent,
            'memory': memory,
            'disk': disk_info,
            'boot_time': boot_time
        }
        
    except Exception as e:
        st.error(f"获取系统信息失败: {e}")
        # 返回安全的默认值
        return {
            'cpu_percent': 0.0,
            'memory': type('obj', (object,), {
                'percent': 0, 
                'used': 0, 
                'total': 1024*1024*1024*8  # 8GB
            })(),
            'disk': type('obj', (object,), {
                'used': 0, 
                'total': 1024*1024*1024*100,  # 100GB
                'free': 1024*1024*1024*100    # 100GB
            })(),
            'boot_time': datetime.now()
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
    # 首先尝试使用nvidia-ml-py获取真实的GPU使用情况
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # 获取基本信息
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # 获取内存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = mem_info.total / 1024**3  # GB
            memory_used = mem_info.used / 1024**3   # GB  
            memory_free = mem_info.free / 1024**3   # GB
            memory_utilization = (memory_used / memory_total) * 100
            
            # 获取GPU使用率
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = 0
                memory_util = 0
            
            # 获取温度
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
            
            # 获取功耗
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                max_power = pynvml.nvmlDeviceGetMaxPowerManagement(handle) / 1000.0  # W
            except:
                power = 0
                max_power = 0
            
            gpu_info.append({
                'id': i,
                'name': name,
                'memory_total': memory_total,
                'memory_used': memory_used,
                'memory_free': memory_free,
                'memory_utilization': memory_utilization,
                'gpu_utilization': gpu_util,
                'memory_bandwidth_util': memory_util,
                'temperature': temperature,
                'power_usage': power,
                'max_power': max_power,
                'power_utilization': (power / max_power * 100) if max_power > 0 else 0
            })
        
        return gpu_info
        
    except ImportError:
        # 如果没有pynvml，尝试使用PyTorch（功能有限）
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                for i in range(gpu_count):
                    try:
                        # 尝试获取设备属性，这可能会失败
                        props = torch.cuda.get_device_properties(i)
                        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                        memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                        memory_total = props.total_memory / 1024**3  # GB
                        
                        gpu_info.append({
                            'id': i,
                            'name': props.name,
                            'memory_total': memory_total,
                            'memory_used': memory_allocated,  # PyTorch分配的内存
                            'memory_free': memory_total - memory_cached,
                            'memory_utilization': (memory_allocated / memory_total) * 100,
                            'gpu_utilization': 0,  # PyTorch无法获取
                            'memory_bandwidth_util': 0,
                            'temperature': 0,
                            'power_usage': 0,
                            'max_power': 0,
                            'power_utilization': 0,
                            'pytorch_only': True  # 标记这是PyTorch限制版本
                        })
                    except Exception as gpu_err:
                        # 如果获取特定GPU信息失败，创建一个基础条目
                        gpu_info.append({
                            'id': i,
                            'name': f'GPU {i} (信息获取失败)',
                            'memory_total': 1.0,
                            'memory_used': 0.0,
                            'memory_free': 1.0,
                            'memory_utilization': 0.0,
                            'gpu_utilization': 0,
                            'memory_bandwidth_util': 0,
                            'temperature': 0,
                            'power_usage': 0,
                            'max_power': 0,
                            'power_utilization': 0,
                            'error': str(gpu_err)
                        })
                return gpu_info
            else:
                return []
        except ImportError:
            return None
    except Exception as e:
        # 捕获所有其他错误
        return {'error': f'GPU检查失败: {str(e)}'}

def main():
    st.title("🔍 Real-Time System Monitor")
    st.markdown("实时监控系统资源和分析进程状态")
    
    # 创建更新控制
    auto_refresh = st.sidebar.checkbox("自动刷新", value=False)
    refresh_interval = st.sidebar.slider("刷新间隔(秒)", 1, 10, 3)
    
    if st.sidebar.button("🔄 立即刷新"):
        st.rerun()
    
    # 如果启用自动刷新，设置定时器
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # 系统资源状态
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
            subplot_titles=['CPU使用率', '内存使用率', '磁盘使用率', '系统负载'],
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
        st.plotly_chart(fig, width='stretch')
    
    # Python进程监控
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
            width='stretch'
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
    
    # GPU状态监控
    st.subheader("🎮 GPU状态监控")
    
    gpu_info = check_gpu_status()
    
    if gpu_info is None:
        st.warning("PyTorch和NVIDIA-ML-PY都未安装，无法检测GPU状态")
        st.info("安装命令: pip install nvidia-ml-py 或 conda install nvidia-ml-py")
    elif isinstance(gpu_info, dict) and 'error' in gpu_info:
        st.error(f"GPU状态检查失败: {gpu_info['error']}")
        st.info("建议安装nvidia-ml-py以获得完整GPU监控功能")
    elif not gpu_info:
        st.info("未检测到可用的CUDA GPU")
    else:
        for gpu in gpu_info:
            with st.expander(f"GPU {gpu['id']}: {gpu['name']}", expanded=True):
                if 'error' in gpu:
                    st.error(f"GPU {gpu['id']} 信息获取失败: {gpu['error']}")
                    st.info("尝试重启应用程序或检查CUDA驱动版本")
                else:
                    # 检查是否是功能限制版本
                    is_pytorch_only = gpu.get('pytorch_only', False)
                    if is_pytorch_only:
                        st.info("⚠️ 使用PyTorch监控(功能有限) - 建议安装nvidia-ml-py以获得完整监控")
                    
                    # 主要指标
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("显存使用率", f"{gpu['memory_utilization']:.1f}%")
                    with col2:
                        st.metric("已用显存", f"{gpu['memory_used']:.2f}GB")
                    with col3:
                        st.metric("总显存", f"{gpu['memory_total']:.2f}GB")
                    with col4:
                        if gpu['gpu_utilization'] > 0:
                            st.metric("GPU使用率", f"{gpu['gpu_utilization']:.1f}%")
                        else:
                            st.metric("GPU使用率", "N/A")
                    
                    # 额外信息(如果可用)
                    if not is_pytorch_only:
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            if gpu['temperature'] > 0:
                                temp_color = "🔥" if gpu['temperature'] > 80 else "🌡️"
                                st.metric("温度", f"{temp_color} {gpu['temperature']}°C")
                            else:
                                st.metric("温度", "N/A")
                        with col6:
                            if gpu['power_usage'] > 0:
                                st.metric("功耗", f"⚡ {gpu['power_usage']:.1f}W")
                            else:
                                st.metric("功耗", "N/A")
                        with col7:
                            if gpu['memory_bandwidth_util'] > 0:
                                st.metric("内存带宽", f"{gpu['memory_bandwidth_util']:.1f}%")
                            else:
                                st.metric("内存带宽", "N/A")
                    
                    # 显存使用图表
                    try:
                        fig = go.Figure()
                        
                        # 显存使用条形图
                        fig.add_trace(go.Bar(
                            x=['已使用', '空闲'],
                            y=[gpu['memory_used'], gpu['memory_free']],
                            marker_color=['#ff6b6b', '#4ecdc4'],
                            name='显存状态'
                        ))
                        
                        fig.update_layout(
                            title=f"GPU {gpu['id']} 显存使用情况",
                            yaxis_title="显存 (GB)",
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # 如果有GPU使用率信息，添加使用率图表
                        if not is_pytorch_only and gpu['gpu_utilization'] > 0:
                            fig2 = go.Figure()
                            
                            # 创建仪表盘样式的图表
                            fig2.add_trace(go.Indicator(
                                mode="gauge+number+delta",
                                value=gpu['gpu_utilization'],
                                domain={'x': [0, 0.5], 'y': [0, 1]},
                                title={'text': "GPU使用率"},
                                gauge={'axis': {'range': [None, 100]},
                                       'bar': {'color': "#ff6b6b"},
                                       'steps': [
                                           {'range': [0, 50], 'color': "#4ecdc4"},
                                           {'range': [50, 80], 'color': "#ffd93d"},
                                           {'range': [80, 100], 'color': "#ff6b6b"}],
                                       'threshold': {'line': {'color': "red", 'width': 4},
                                                     'thickness': 0.75, 'value': 90}}
                            ))
                            
                            fig2.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=gpu['memory_utilization'],
                                domain={'x': [0.5, 1], 'y': [0, 1]},
                                title={'text': "显存使用率"},
                                gauge={'axis': {'range': [None, 100]},
                                       'bar': {'color': "#6c5ce7"},
                                       'steps': [
                                           {'range': [0, 50], 'color': "#4ecdc4"},
                                           {'range': [50, 80], 'color': "#ffd93d"},
                                           {'range': [80, 100], 'color': "#ff6b6b"}]}
                            ))
                            
                            fig2.update_layout(height=250, margin={'t': 50, 'b': 0, 'l': 0, 'r': 0})
                            st.plotly_chart(fig2, width='stretch')
                            
                    except Exception as plot_err:
                        st.warning(f"图表显示失败: {plot_err}")
    
    # 日志文件监控
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
                                height=200,
                                disabled=True
                            )
                except:
                    st.error("无法读取日志文件")
    else:
        st.info("当前目录下没有发现日志文件")

if __name__ == "__main__":
    main()