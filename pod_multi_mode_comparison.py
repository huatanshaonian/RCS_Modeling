#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POD多模态数量对比分析模块

这个模块专门用于比较不同模态数量下POD重建的性能
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import font_manager

# Windows中文字体设置
try:
    font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：无法设置中文字体，图表中的中文可能显示为方块")

def create_pod_multi_mode_comparison(reconstruction_results, output_dir):
    """
    创建POD多模态数量对比图表
    
    参数:
    reconstruction_results: 重建结果字典
    output_dir: 输出目录
    """
    
    # 提取POD相关的结果
    pod_results = {}
    for key, result in reconstruction_results.items():
        if key.startswith('POD_') and 'modes' in key:
            pod_results[key] = result
    
    if not pod_results:
        print("没有找到POD多模态结果")
        return
    
    # 按模态数量排序
    sorted_pod_results = sorted(pod_results.items(), key=lambda x: x[1]['modes'])
    
    modes = [result[1]['modes'] for result in sorted_pod_results]
    r2_values = [result[1]['r2'] for result in sorted_pod_results]
    mse_values = [result[1]['mse'] for result in sorted_pod_results]
    method_names = [result[1]['method'] for result in sorted_pod_results]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. R²随模态数变化
    ax1.plot(modes, r2_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('模态数量')
    ax1.set_ylabel('R² 分数')
    ax1.set_title('POD重建性能: R²随模态数变化')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 添加数值标签
    for i, (mode, r2) in enumerate(zip(modes, r2_values)):
        ax1.annotate(f'{r2:.3f}', (mode, r2), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # 2. MSE随模态数变化
    ax2.semilogy(modes, mse_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('模态数量')
    ax2.set_ylabel('均方误差 (MSE, 对数尺度)')
    ax2.set_title('POD重建性能: MSE随模态数变化')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (mode, mse) in enumerate(zip(modes, mse_values)):
        ax2.annotate(f'{mse:.2e}', (mode, mse), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # 3. 柱状图对比
    colors = plt.cm.viridis(np.linspace(0, 1, len(modes)))
    bars = ax3.bar(range(len(modes)), r2_values, color=colors, alpha=0.7)
    ax3.set_xlabel('模态配置')
    ax3.set_ylabel('R² 分数')
    ax3.set_title('不同模态数量的R²对比')
    ax3.set_xticks(range(len(modes)))
    ax3.set_xticklabels([f'{m}模态' for m in modes], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2:.3f}', ha='center', va='bottom')
    
    # 4. 效率分析图(R²/模态数比值)
    efficiency = [r2/mode for r2, mode in zip(r2_values, modes)]
    ax4.plot(modes, efficiency, 'go-', linewidth=2, markersize=8)
    ax4.set_xlabel('模态数量')
    ax4.set_ylabel('效率 (R²/模态数)')
    ax4.set_title('POD模态使用效率')
    ax4.grid(True, alpha=0.3)
    
    # 标记最高效率点
    max_eff_idx = np.argmax(efficiency)
    ax4.annotate(f'最高效率\n{modes[max_eff_idx]}模态\n{efficiency[max_eff_idx]:.4f}', 
                xy=(modes[max_eff_idx], efficiency[max_eff_idx]),
                xytext=(20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # 保存图表
    comparison_file = os.path.join(output_dir, 'pod_multi_mode_comparison.png')
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建详细的性能表格
    create_pod_performance_table(sorted_pod_results, output_dir)
    
    print(f"POD多模态对比图表已保存到: {comparison_file}")
    return comparison_file

def create_pod_performance_table(sorted_pod_results, output_dir):
    """
    创建POD性能对比表格
    
    参数:
    sorted_pod_results: 排序后的POD结果
    output_dir: 输出目录
    """
    
    # 准备表格数据
    table_data = []
    for key, result in sorted_pod_results:
        modes = result['modes']
        r2 = result['r2']
        mse = result['mse']
        efficiency = r2 / modes
        
        table_data.append({
            '模态数量': modes,
            'R² 分数': f"{r2:.4f}",
            'MSE': f"{mse:.2e}",
            '效率 (R²/模态)': f"{efficiency:.4f}",
            '相对性能': f"{r2/sorted_pod_results[-1][1]['r2']*100:.1f}%"
        })
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存CSV
    table_file = os.path.join(output_dir, 'pod_multi_mode_performance.csv')
    df.to_csv(table_file, index=False, encoding='utf-8-sig')
    
    # 打印表格
    print("\nPOD多模态性能对比表:")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return table_file

def analyze_pod_mode_trends(sorted_pod_results):
    """
    分析POD模态数量的趋势
    
    参数:
    sorted_pod_results: 排序后的POD结果
    
    返回:
    分析报告字典
    """
    
    modes = [result[1]['modes'] for result in sorted_pod_results]
    r2_values = [result[1]['r2'] for result in sorted_pod_results]
    mse_values = [result[1]['mse'] for result in sorted_pod_results]
    
    # 计算改善率
    r2_improvements = []
    mse_improvements = []
    
    for i in range(1, len(r2_values)):
        r2_improvement = (r2_values[i] - r2_values[i-1]) / (modes[i] - modes[i-1])
        mse_improvement = (mse_values[i-1] - mse_values[i]) / (modes[i] - modes[i-1])
        
        r2_improvements.append(r2_improvement)
        mse_improvements.append(mse_improvement)
    
    # 找到最佳效率点
    efficiency = [r2/mode for r2, mode in zip(r2_values, modes)]
    best_efficiency_idx = np.argmax(efficiency)
    
    # 找到收益递减点（改善率开始显著下降的点）
    if len(r2_improvements) > 2:
        # 计算二阶差分来找到曲率变化点
        second_diff = np.diff(r2_improvements)
        diminishing_returns_idx = np.argmax(second_diff < -0.001) + 1
        if diminishing_returns_idx == 1:  # 如果没找到明显的拐点
            diminishing_returns_idx = len(modes) // 2
    else:
        diminishing_returns_idx = 0
    
    analysis = {
        'total_modes_tested': len(modes),
        'mode_range': (min(modes), max(modes)),
        'best_r2': max(r2_values),
        'best_r2_mode': modes[np.argmax(r2_values)],
        'lowest_mse': min(mse_values),
        'lowest_mse_mode': modes[np.argmin(mse_values)],
        'best_efficiency': efficiency[best_efficiency_idx],
        'best_efficiency_mode': modes[best_efficiency_idx],
        'diminishing_returns_mode': modes[min(diminishing_returns_idx, len(modes)-1)],
        'average_r2_improvement_per_mode': np.mean(r2_improvements) if r2_improvements else 0
    }
    
    return analysis

if __name__ == "__main__":
    print("POD多模态数量对比分析模块")
    print("该模块用于创建POD不同模态数量的性能对比图表")