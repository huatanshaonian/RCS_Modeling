#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Results Analysis Page - 结果分析页面
深度分析POD和Autoencoder的结果
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="Results Analysis",
    page_icon="📊",
    layout="wide"
)

def load_pod_results(results_dir, freq, train_size):
    """加载POD分析结果"""
    try:
        base_dir = Path(results_dir) / freq / f"train_{train_size}"
        
        # 加载POD模态和特征值
        if (base_dir / "pod_modes.npy").exists():
            modes = np.load(base_dir / "pod_modes.npy")
            # 检查可能的特征值文件名
            eigenvals = None
            if (base_dir / "lambda_values.npy").exists():
                eigenvals = np.load(base_dir / "lambda_values.npy")
            elif (base_dir / "pod_eigenvalues.npy").exists():
                eigenvals = np.load(base_dir / "pod_eigenvalues.npy")
            
            # 检查可能的系数文件名
            coeffs = None
            if (base_dir / "pod_coeffs_train.npy").exists():
                coeffs = np.load(base_dir / "pod_coeffs_train.npy")
            elif (base_dir / "pod_coefficients.npy").exists():
                coeffs = np.load(base_dir / "pod_coefficients.npy")
            
            return {
                'modes': modes,
                'eigenvalues': eigenvals,
                'coefficients': coeffs,
                'available': True
            }
        else:
            return {'available': False}
    except Exception as e:
        st.error(f"加载POD结果时出错: {e}")
        return {'available': False}

def load_autoencoder_results(results_dir, freq, train_size):
    """加载Autoencoder分析结果"""
    try:
        ae_dir = Path(results_dir) / freq / f"train_{train_size}" / "autoencoder"
        
        if not ae_dir.exists():
            return {'available': False}
            
        # 查找可用的模型配置
        configs = []
        for config_dir in ae_dir.iterdir():
            if config_dir.is_dir() and 'latent' in config_dir.name:
                # 检查是否有隐空间数据
                train_latent_file = config_dir / "train_latent_space.npy"
                if train_latent_file.exists():
                    latent_data = np.load(train_latent_file)
                    configs.append({
                        'name': config_dir.name,
                        'path': config_dir,
                        'latent_data': latent_data,
                        'latent_dim': latent_data.shape[1] if len(latent_data.shape) > 1 else 1
                    })
        
        return {
            'available': len(configs) > 0,
            'configs': configs
        }
    except Exception as e:
        st.error(f"加载Autoencoder结果时出错: {e}")
        return {'available': False}

def plot_pod_energy_analysis(eigenvalues):
    """绘制POD能量分析图"""
    if eigenvalues is None:
        return None
        
    # 计算累积能量比例
    total_energy = np.sum(eigenvalues)
    cumulative_energy = np.cumsum(eigenvalues) / total_energy
    energy_ratio = eigenvalues / total_energy
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['特征值分布', '累积能量比例', '各模态能量贡献', '前20个模态对比'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 特征值分布
    fig.add_trace(
        go.Scatter(x=list(range(1, min(51, len(eigenvalues)+1))), 
                  y=eigenvalues[:50], 
                  mode='lines+markers',
                  name='特征值',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # 累积能量比例
    fig.add_trace(
        go.Scatter(x=list(range(1, min(51, len(cumulative_energy)+1))), 
                  y=cumulative_energy[:50], 
                  mode='lines',
                  name='累积能量',
                  line=dict(color='red')),
        row=1, col=2
    )
    
    # 各模态能量贡献
    fig.add_trace(
        go.Bar(x=list(range(1, min(21, len(energy_ratio)+1))), 
               y=energy_ratio[:20],
               name='能量比例',
               marker_color='green'),
        row=2, col=1
    )
    
    # 前20个模态对比
    fig.add_trace(
        go.Scatter(x=list(range(1, min(21, len(eigenvalues)+1))), 
                  y=eigenvalues[:20], 
                  mode='markers',
                  marker=dict(size=10, color='orange'),
                  name='前20模态'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="POD能量分析")
    return fig

def plot_latent_space_analysis(configs):
    """绘制隐空间分析图"""
    if not configs:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['不同维度隐空间分布', '隐空间维度统计', '样本在隐空间中的投影', '维度间相关性'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, config in enumerate(configs[:4]):  # 最多显示4个配置
        latent_data = config['latent_data']
        color = colors[i % len(colors)]
        
        # 如果是2维或以上，绘制前两个维度的散点图
        if latent_data.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(x=latent_data[:100, 0], y=latent_data[:100, 1],
                          mode='markers',
                          name=f"{config['name']} (前100样本)",
                          marker=dict(color=color, size=6, opacity=0.7)),
                row=1, col=1
            )
        
        # 维度统计 - 显示各维度的方差
        if latent_data.shape[1] <= 20:  # 只显示前20维
            variances = np.var(latent_data, axis=0)
            fig.add_trace(
                go.Bar(x=list(range(1, len(variances)+1)), 
                      y=variances,
                      name=f"{config['name']} 方差",
                      marker_color=color,
                      opacity=0.7),
                row=1, col=2
            )
    
    # 如果有数据，显示第一个配置的详细分析
    if configs:
        first_config = configs[0]
        latent_data = first_config['latent_data']
        
        # 样本投影 - 显示所有样本在第一个隐维度上的分布
        fig.add_trace(
            go.Histogram(x=latent_data[:, 0], 
                        name=f"第1维分布 ({first_config['name']})",
                        marker_color='lightblue',
                        opacity=0.7),
            row=2, col=1
        )
        
        # 维度相关性 - 如果有多个维度
        if latent_data.shape[1] >= 2:
            corr_matrix = np.corrcoef(latent_data[:, :min(10, latent_data.shape[1])].T)
            fig.add_trace(
                go.Heatmap(z=corr_matrix, 
                          colorscale='RdBu',
                          zmid=0,
                          name="相关性矩阵"),
                row=2, col=2
            )
    
    fig.update_layout(height=700, title_text="隐空间分析")
    return fig

def main():
    st.title("📊 Results Analysis Dashboard")
    st.markdown("深度分析POD和Autoencoder的结果")
    
    # 侧边栏配置
    st.sidebar.header("分析配置")
    
    results_dir = st.sidebar.text_input("结果目录", value="./results")
    
    if not os.path.exists(results_dir):
        st.error(f"结果目录不存在: {results_dir}")
        return
    
    # 查找可用的频率和训练集大小
    results_path = Path(results_dir)
    available_freqs = []
    available_sizes = {}
    
    for freq_dir in results_path.iterdir():
        if freq_dir.is_dir() and any(c.isdigit() for c in freq_dir.name):
            freq_name = freq_dir.name
            available_freqs.append(freq_name)
            
            # 查找训练集大小
            sizes = []
            for size_dir in freq_dir.iterdir():
                if size_dir.is_dir() and size_dir.name.startswith('train_'):
                    size = size_dir.name.replace('train_', '')
                    sizes.append(int(size))
            available_sizes[freq_name] = sorted(sizes)
    
    if not available_freqs:
        st.warning("未找到分析结果。请先运行分析程序。")
        return
    
    # 选择频率和训练集大小
    selected_freq = st.sidebar.selectbox("选择频率", available_freqs)
    available_train_sizes = available_sizes.get(selected_freq, [])
    
    if not available_train_sizes:
        st.warning(f"频率 {selected_freq} 下未找到训练集结果")
        return
        
    selected_size = st.sidebar.selectbox("选择训练集大小", available_train_sizes)
    
    # 加载结果
    with st.expander("🔍 调试信息", expanded=False):
        st.write(f"Results directory: {results_dir}")
        st.write(f"Looking for: {results_dir}/{selected_freq}/train_{selected_size}")
        result_path = Path(results_dir) / selected_freq / f"train_{selected_size}"
        st.write(f"Directory exists: {result_path.exists()}")
        if result_path.exists():
            files = list(result_path.glob("*.npy"))
            st.write(f"Found NPY files: {[f.name for f in files]}")
    
    pod_results = load_pod_results(results_dir, selected_freq, selected_size)
    ae_results = load_autoencoder_results(results_dir, selected_freq, selected_size)
    
    # 显示结果概览
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if pod_results['available']:
            st.success("✅ POD结果可用")
            if pod_results['eigenvalues'] is not None:
                st.metric("POD模态数", len(pod_results['eigenvalues']))
        else:
            st.error("❌ POD结果不可用")
    
    with col2:
        if ae_results['available']:
            st.success("✅ Autoencoder结果可用")
            st.metric("AE配置数", len(ae_results['configs']))
        else:
            st.error("❌ Autoencoder结果不可用")
    
    with col3:
        total_files = 0
        result_path = Path(results_dir) / selected_freq / f"train_{selected_size}"
        if result_path.exists():
            total_files = len(list(result_path.rglob("*.png"))) + len(list(result_path.rglob("*.npy")))
        st.metric("总文件数", total_files)
    
    # 分析标签页
    if pod_results['available'] or ae_results['available']:
        tab1, tab2, tab3 = st.tabs(["POD分析", "Autoencoder分析", "对比分析"])
        
        with tab1:
            if pod_results['available']:
                st.subheader("POD能量分析")
                
                if pod_results['eigenvalues'] is not None:
                    # 显示能量分析图
                    energy_fig = plot_pod_energy_analysis(pod_results['eigenvalues'])
                    if energy_fig:
                        st.plotly_chart(energy_fig, width='stretch')
                    
                    # 显示数值统计
                    col1, col2, col3, col4 = st.columns(4)
                    eigenvals = pod_results['eigenvalues']
                    total_energy = np.sum(eigenvals)
                    
                    with col1:
                        st.metric("总能量", f"{total_energy:.2e}")
                    with col2:
                        st.metric("最大特征值", f"{eigenvals[0]:.2e}")
                    with col3:
                        energy_90 = np.where(np.cumsum(eigenvals)/total_energy >= 0.9)[0]
                        modes_90 = energy_90[0] + 1 if len(energy_90) > 0 else len(eigenvals)
                        st.metric("90%能量模态数", modes_90)
                    with col4:
                        energy_99 = np.where(np.cumsum(eigenvals)/total_energy >= 0.99)[0]
                        modes_99 = energy_99[0] + 1 if len(energy_99) > 0 else len(eigenvals)
                        st.metric("99%能量模态数", modes_99)
                else:
                    st.info("POD特征值数据不可用")
            else:
                st.info("POD结果不可用")
        
        with tab2:
            if ae_results['available']:
                st.subheader("Autoencoder隐空间分析")
                
                # 显示配置信息
                configs_df = pd.DataFrame([
                    {
                        '配置名称': config['name'],
                        '隐空间维度': config['latent_dim'],
                        '样本数量': config['latent_data'].shape[0]
                    }
                    for config in ae_results['configs']
                ])
                st.dataframe(configs_df, width='stretch')
                
                # 隐空间分析图
                latent_fig = plot_latent_space_analysis(ae_results['configs'])
                if latent_fig:
                    st.plotly_chart(latent_fig, width='stretch')
                    
            else:
                st.info("Autoencoder结果不可用")
        
        with tab3:
            # 添加调试信息
            with st.expander("🔧 对比分析调试信息", expanded=False):
                st.write(f"POD available: {pod_results.get('available', False)}")
                st.write(f"AE available: {ae_results.get('available', False)}")
                if pod_results.get('available', False):
                    st.write(f"POD eigenvalues available: {pod_results.get('eigenvalues') is not None}")
                    if pod_results.get('eigenvalues') is not None:
                        st.write(f"Eigenvalues shape: {pod_results['eigenvalues'].shape}")
                if ae_results.get('available', False):
                    st.write(f"AE configs count: {len(ae_results.get('configs', []))}")
                    for i, config in enumerate(ae_results.get('configs', [])):
                        st.write(f"Config {i}: {config['name']}, dim: {config['latent_dim']}")
            
            if pod_results['available'] and ae_results['available']:
                st.subheader("POD vs Autoencoder 对比分析")
                
                # 维度效率对比
                if pod_results['eigenvalues'] is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**POD维度效率**")
                        eigenvals = pod_results['eigenvalues']
                        total_energy = np.sum(eigenvals)
                        cumulative = np.cumsum(eigenvals) / total_energy
                        
                        efficiency_data = []
                        for i, config in enumerate(ae_results['configs']):
                            dim = config['latent_dim']
                            if dim <= len(cumulative):
                                pod_energy = cumulative[dim-1]
                                efficiency_data.append({
                                    '维度': dim,
                                    'POD累积能量': f"{pod_energy:.3f}",
                                    'AE配置': config['name']
                                })
                        
                        if efficiency_data:
                            st.dataframe(pd.DataFrame(efficiency_data))
                    
                    with col2:
                        st.markdown("**维度对比图**")
                        
                        # 创建对比图
                        fig = go.Figure()
                        
                        # POD累积能量曲线
                        fig.add_trace(go.Scatter(
                            x=list(range(1, min(51, len(cumulative)+1))),
                            y=cumulative[:50],
                            mode='lines',
                            name='POD累积能量',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # AE配置点
                        ae_dims = [config['latent_dim'] for config in ae_results['configs']]
                        ae_energies = [cumulative[dim-1] if dim <= len(cumulative) else 1.0 
                                     for dim in ae_dims]
                        
                        fig.add_trace(go.Scatter(
                            x=ae_dims,
                            y=ae_energies,
                            mode='markers',
                            name='AE配置',
                            marker=dict(color='red', size=10)
                        ))
                        
                        fig.update_layout(
                            title="POD vs AE 维度效率对比",
                            xaxis_title="维度数",
                            yaxis_title="累积能量比例",
                            height=400
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                
            else:
                st.info("需要POD和Autoencoder结果才能进行对比分析")
    
    else:
        st.warning("没有可用的分析结果")

if __name__ == "__main__":
    main()