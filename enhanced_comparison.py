#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版POD-Autoencoder综合性能对比
包含POD不同模态数和AE不同维度的详细对比分析
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def calculate_pod_performance_by_modes(rcs_data, pod_modes, mean_rcs, mode_numbers=[5, 10, 15, 20]):
    """
    计算POD在不同模态数下的重构性能
    
    参数:
    rcs_data: 原始RCS数据 [num_samples, num_angles]
    pod_modes: POD模态矩阵 [num_angles, num_modes]
    mean_rcs: 平均RCS数据
    mode_numbers: 要测试的模态数列表
    
    返回:
    pod_performance: 字典，包含不同模态数的性能指标
    """
    pod_performance = {}
    
    for n_modes in mode_numbers:
        if n_modes > pod_modes.shape[1]:
            print(f"警告: 请求的模态数 {n_modes} 超过可用模态数 {pod_modes.shape[1]}")
            continue
            
        # 使用前n_modes个模态进行重构
        selected_modes = pod_modes[:, :n_modes]
        
        # 计算POD系数
        pod_coeffs = np.dot(rcs_data - mean_rcs, selected_modes)
        
        # 重构数据
        reconstructed = np.dot(pod_coeffs, selected_modes.T) + mean_rcs
        
        # 计算性能指标
        mse = mean_squared_error(rcs_data, reconstructed)
        r2 = r2_score(rcs_data.flatten(), reconstructed.flatten())
        
        # 计算能量保留比例
        eigenvalues = np.var(pod_coeffs, axis=0)
        total_energy = np.sum(np.var(rcs_data - mean_rcs, axis=0))
        energy_ratio = np.sum(eigenvalues) / total_energy if total_energy > 0 else 0
        
        pod_performance[f"POD_{n_modes}modes"] = {
            'method': 'POD',
            'dimensions': n_modes,
            'mse': mse,
            'r2': r2,
            'energy_ratio': energy_ratio
        }
        
        print(f"POD {n_modes}模态: MSE={mse:.6f}, R²={r2:.6f}, 能量比={energy_ratio:.4f}")
    
    return pod_performance


def load_pod_results(train_dir):
    """
    从保存的文件中加载POD结果
    
    参数:
    train_dir: 训练目录路径
    
    返回:
    pod_data: 包含POD相关数据的字典
    """
    pod_data = {}
    
    try:
        # 加载POD相关数据
        pod_modes_file = os.path.join(train_dir, "pod_modes.npy")
        mean_rcs_file = os.path.join(train_dir, "mean_rcs.npy")
        train_indices_file = os.path.join(train_dir, "train_indices.npy")
        
        if os.path.exists(pod_modes_file):
            pod_data['modes'] = np.load(pod_modes_file)
            print(f"已加载POD模态: {pod_data['modes'].shape}")
        
        if os.path.exists(mean_rcs_file):
            pod_data['mean_rcs'] = np.load(mean_rcs_file)
            print(f"已加载平均RCS: {pod_data['mean_rcs'].shape}")
            
        if os.path.exists(train_indices_file):
            pod_data['train_indices'] = np.load(train_indices_file)
            print(f"已加载训练索引: {len(pod_data['train_indices'])}")
            
    except Exception as e:
        print(f"加载POD结果时出错: {e}")
    
    return pod_data


def comprehensive_comparison_analysis(ae_results, pod_data, rcs_train_data, output_dir):
    """
    创建综合性能对比分析图
    
    参数:
    ae_results: Autoencoder结果字典
    pod_data: POD数据字典
    rcs_train_data: 训练集RCS数据
    output_dir: 输出目录
    """
    
    print("开始创建综合性能对比分析...")
    
    # 创建对比目录
    comparison_dir = os.path.join(output_dir, 'ae_pod_comprehensive')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. 计算POD在不同模态数下的性能
    if 'modes' in pod_data and 'mean_rcs' in pod_data:
        print("计算POD不同模态数的性能...")
        pod_performance = calculate_pod_performance_by_modes(
            rcs_train_data, 
            pod_data['modes'], 
            pod_data['mean_rcs'],
            mode_numbers=[5, 10, 15, 20, 25, 30]
        )
    else:
        print("POD数据不完整，无法计算性能")
        pod_performance = {}
    
    # 2. 整理AE性能数据
    ae_performance = {}
    for config_name, config_results in ae_results.items():
        ae_performance[config_name] = {
            'method': config_results.get('model_type', 'unknown'),
            'dimensions': config_results.get('latent_dim', 0),
            'mse': config_results.get('mse', float('nan')),
            'r2': config_results.get('r2', 0),
            'energy_ratio': config_results.get('r2', 0)  # 使用R²作为能量保留的近似
        }
    
    # 3. 合并所有性能数据
    all_performance = {**pod_performance, **ae_performance}
    
    # 4. 创建综合对比图
    create_comprehensive_plots(all_performance, comparison_dir)
    
    # 5. 保存性能数据表格
    save_performance_table(all_performance, comparison_dir)
    
    print(f"综合对比分析完成，结果保存在: {comparison_dir}")


def create_comprehensive_plots(all_performance, output_dir):
    """
    创建综合对比图表
    
    参数:
    all_performance: 所有方法的性能数据字典
    output_dir: 输出目录
    """
    
    # 准备数据
    methods = []
    dimensions = []
    mse_values = []
    r2_values = []
    energy_ratios = []
    colors = []
    markers = []
    
    for config_name, perf in all_performance.items():
        if not np.isnan(perf['mse']) and not np.isnan(perf['r2']):
            methods.append(config_name)
            dimensions.append(perf['dimensions'])
            mse_values.append(perf['mse'])
            r2_values.append(perf['r2'])
            energy_ratios.append(perf['energy_ratio'])
            
            # 设置颜色和标记
            if perf['method'] == 'POD':
                colors.append('blue')
                markers.append('o')
            elif perf['method'] == 'standard':
                colors.append('red')
                markers.append('s')
            elif perf['method'] == 'vae':
                colors.append('green')
                markers.append('^')
            else:
                colors.append('gray')
                markers.append('x')
    
    # 创建综合对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('POD vs Autoencoder 综合性能对比分析', fontsize=16, fontweight='bold')
    
    # 1. 维度 vs R²分数对比
    ax1 = axes[0, 0]
    pod_mask = [perf['method'] == 'POD' for perf in all_performance.values() if not np.isnan(perf['mse'])]
    standard_mask = [perf['method'] == 'standard' for perf in all_performance.values() if not np.isnan(perf['mse'])]
    vae_mask = [perf['method'] == 'vae' for perf in all_performance.values() if not np.isnan(perf['mse'])]
    
    # 分别绘制不同方法的曲线
    dims_array = np.array(dimensions)
    r2_array = np.array(r2_values)
    
    if any(pod_mask):
        pod_dims = dims_array[pod_mask]
        pod_r2 = r2_array[pod_mask]
        ax1.plot(pod_dims, pod_r2, 'bo-', label='POD', linewidth=2, markersize=8)
    
    if any(standard_mask):
        std_dims = dims_array[standard_mask]
        std_r2 = r2_array[standard_mask]
        ax1.plot(std_dims, std_r2, 'rs-', label='Standard AE', linewidth=2, markersize=8)
    
    if any(vae_mask):
        vae_dims = dims_array[vae_mask]
        vae_r2 = r2_array[vae_mask]
        ax1.plot(vae_dims, vae_r2, 'g^-', label='VAE', linewidth=2, markersize=8)
    
    ax1.set_xlabel('维度/模态数')
    ax1.set_ylabel('R² 分数')
    ax1.set_title('重构质量对比 (R²)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 维度 vs MSE对比
    ax2 = axes[0, 1]
    mse_array = np.array(mse_values)
    
    if any(pod_mask):
        pod_mse = mse_array[pod_mask]
        ax2.plot(pod_dims, pod_mse, 'bo-', label='POD', linewidth=2, markersize=8)
    
    if any(standard_mask):
        std_mse = mse_array[standard_mask]
        ax2.plot(std_dims, std_mse, 'rs-', label='Standard AE', linewidth=2, markersize=8)
    
    if any(vae_mask):
        vae_mse = mse_array[vae_mask]
        ax2.plot(vae_dims, vae_mse, 'g^-', label='VAE', linewidth=2, markersize=8)
    
    ax2.set_xlabel('维度/模态数')
    ax2.set_ylabel('MSE')
    ax2.set_title('重构误差对比 (MSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数坐标
    
    # 3. 效率对比 (R² vs 维度)
    ax3 = axes[0, 2]
    for i, (method, dim, r2) in enumerate(zip(methods, dimensions, r2_values)):
        ax3.scatter(dim, r2, c=colors[i], marker=markers[i], s=100, alpha=0.7)
        ax3.annotate(method.replace('_', '\n'), (dim, r2), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax3.set_xlabel('维度/模态数')
    ax3.set_ylabel('R² 分数')
    ax3.set_title('效率对比 (R² vs 复杂度)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 方法对比柱状图
    ax4 = axes[1, 0]
    method_types = ['POD', 'Standard AE', 'VAE']
    method_r2_means = []
    method_r2_stds = []
    
    for method_type in method_types:
        method_r2s = [r2 for perf, r2 in zip(all_performance.values(), r2_values) 
                     if perf['method'] == method_type.split()[0].lower() and not np.isnan(r2)]
        if method_r2s:
            method_r2_means.append(np.mean(method_r2s))
            method_r2_stds.append(np.std(method_r2s))
        else:
            method_r2_means.append(0)
            method_r2_stds.append(0)
    
    bars = ax4.bar(method_types, method_r2_means, yerr=method_r2_stds, 
                   alpha=0.7, color=['blue', 'red', 'green'], capsize=5)
    ax4.set_ylabel('平均 R² 分数')
    ax4.set_title('不同方法平均性能')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mean_val in zip(bars, method_r2_means):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    # 5. 性能矩阵热图
    ax5 = axes[1, 1]
    
    # 创建性能矩阵
    unique_methods = sorted(set([perf['method'] for perf in all_performance.values()]))
    unique_dims = sorted(set([perf['dimensions'] for perf in all_performance.values()]))
    
    performance_matrix = np.full((len(unique_methods), len(unique_dims)), np.nan)
    
    for config_name, perf in all_performance.items():
        if not np.isnan(perf['r2']):
            method_idx = unique_methods.index(perf['method'])
            dim_idx = unique_dims.index(perf['dimensions'])
            performance_matrix[method_idx, dim_idx] = perf['r2']
    
    im = ax5.imshow(performance_matrix, cmap='viridis', aspect='auto')
    ax5.set_xticks(range(len(unique_dims)))
    ax5.set_xticklabels(unique_dims)
    ax5.set_yticks(range(len(unique_methods)))
    ax5.set_yticklabels(unique_methods)
    ax5.set_xlabel('维度/模态数')
    ax5.set_ylabel('方法')
    ax5.set_title('性能热图 (R²)')
    
    # 添加数值标签
    for i in range(len(unique_methods)):
        for j in range(len(unique_dims)):
            if not np.isnan(performance_matrix[i, j]):
                ax5.text(j, i, f'{performance_matrix[i, j]:.3f}',
                        ha="center", va="center", color="white" if performance_matrix[i, j] < 0.5 else "black")
    
    plt.colorbar(im, ax=ax5)
    
    # 6. 复杂度vs性能权衡
    ax6 = axes[1, 2]
    
    # 计算效率指标 (R²/维度)
    efficiency = [r2/dim if dim > 0 else 0 for r2, dim in zip(r2_values, dimensions)]
    
    scatter = ax6.scatter(dimensions, r2_values, c=efficiency, s=100, cmap='coolwarm', alpha=0.7)
    
    for i, method in enumerate(methods):
        ax6.annotate(method.replace('_', '\n'), (dimensions[i], r2_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax6.set_xlabel('维度/模态数')
    ax6.set_ylabel('R² 分数')
    ax6.set_title('复杂度 vs 性能权衡')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='效率 (R²/维度)')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("综合对比图表已保存")


def save_performance_table(all_performance, output_dir):
    """
    保存性能数据表格
    
    参数:
    all_performance: 性能数据字典
    output_dir: 输出目录
    """
    
    # 转换为DataFrame
    df_data = []
    for config_name, perf in all_performance.items():
        df_data.append({
            '配置': config_name,
            '方法': perf['method'],
            '维度': perf['dimensions'],
            'MSE': perf['mse'],
            'R²': perf['r2'],
            '效率(R²/维度)': perf['r2'] / perf['dimensions'] if perf['dimensions'] > 0 else 0
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values(['方法', '维度'])
    
    # 保存CSV
    csv_file = os.path.join(output_dir, 'performance_comparison.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"性能对比表格已保存到: {csv_file}")
    
    # 打印摘要
    print("\n=== 性能对比摘要 ===")
    print(df.to_string(index=False))


def enhanced_compare_with_pod_results(ae_results, pod_results, output_dir, train_dir=None):
    """
    增强版的POD-AE对比分析
    
    参数:
    ae_results: Autoencoder结果
    pod_results: POD结果
    output_dir: 输出目录
    train_dir: 训练目录（用于加载额外的POD数据）
    """
    
    if not ae_results:
        print("没有可用的Autoencoder结果进行对比")
        return
    
    print("开始增强版POD-Autoencoder对比分析...")
    
    # 加载POD相关数据
    if train_dir:
        pod_data = load_pod_results(train_dir)
    else:
        pod_data = {}
    
    # 尝试从train_dir加载训练数据
    rcs_train_data = None
    if train_dir:
        try:
            # 这里需要根据实际的数据加载方式调整
            train_indices_file = os.path.join(train_dir, "train_indices.npy")
            if os.path.exists(train_indices_file):
                # 假设我们能够重新构建训练数据
                print("尝试重新构建训练数据...")
                # 这里需要实际的数据加载逻辑
                pass
        except Exception as e:
            print(f"加载训练数据失败: {e}")
    
    # 如果有完整的数据，进行综合分析
    if pod_data and rcs_train_data is not None:
        comprehensive_comparison_analysis(ae_results, pod_data, rcs_train_data, output_dir)
    else:
        print("数据不完整，执行简化版对比...")
        # 执行原有的对比逻辑
        from autoencoder_visualization import compare_with_pod_results
        compare_with_pod_results(ae_results, pod_results, output_dir)


if __name__ == "__main__":
    # 测试代码
    print("增强版POD-AE对比分析模块")
    
    # 这里可以添加测试代码
    test_dir = "./test_results"
    os.makedirs(test_dir, exist_ok=True)
    
    print("模块测试完成")