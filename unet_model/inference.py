#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理和演示模块
用于FiLM-UNet模型的推理、可视化和性能评估
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
try:
    import seaborn as sns
except ImportError:
    sns = None
import pandas as pd
import os
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from film_unet_model import FiLMUNetModel
from data_preprocessing import RCSDataLoader, DataNormalizer


class RCSPredictor:
    """RCS预测器"""
    
    def __init__(self, model_path=None, device='auto'):
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型
        if model_path:
            self.model, self.checkpoint = FiLMUNetModel.load_model(model_path, self.device)
            print(f"模型已从 {model_path} 加载到 {self.device}")
        else:
            self.model = FiLMUNetModel().to(self.device)
            self.checkpoint = None
            print(f"使用随机初始化模型在 {self.device}")
        
        self.model.eval()
        self.normalizer = None
    
    def set_normalizer(self, normalizer):
        """设置数据归一化器"""
        self.normalizer = normalizer
    
    def predict_single(self, design_params, return_raw=False):
        """
        单样本预测
        
        Args:
            design_params: 设计参数 [9] 或 [1, 9]
            return_raw: 是否返回原始输出（未反归一化）
            
        Returns:
            rcs_prediction: RCS预测结果 [91, 91]
            prediction_info: 预测信息字典
        """
        # 转换为torch tensor
        if not isinstance(design_params, torch.Tensor):
            design_params = torch.FloatTensor(design_params)
        
        # 确保形状正确
        if design_params.dim() == 1:
            design_params = design_params.unsqueeze(0)  # [1, 9]
        
        # 归一化
        if self.normalizer:
            design_params_np = design_params.numpy()
            normalized_params = self.normalizer.transform_params(design_params_np)
            design_params = torch.FloatTensor(normalized_params)
        
        design_params = design_params.to(self.device)
        
        # 预测
        start_time = time.time()
        with torch.no_grad():
            output, intermediate_outputs, debug_info = self.model(design_params)
        inference_time = time.time() - start_time
        
        # 移到CPU并转换为numpy
        rcs_prediction = output[0, 0].cpu().numpy()  # [91, 91]
        
        # 反归一化
        if self.normalizer and not return_raw:
            rcs_prediction = self.normalizer.inverse_transform_rcs(rcs_prediction)
        
        prediction_info = {
            'inference_time': inference_time,
            'output_shape': output.shape,
            'output_range': [output.min().item(), output.max().item()],
            'debug_info': debug_info,
            'intermediate_shapes': {k: v.shape for k, v in intermediate_outputs.items()}
        }
        
        return rcs_prediction, prediction_info
    
    def predict_batch(self, design_params_batch, return_raw=False):
        """
        批量预测
        
        Args:
            design_params_batch: 设计参数批次 [B, 9]
            return_raw: 是否返回原始输出
            
        Returns:
            rcs_predictions: RCS预测结果 [B, 91, 91]
            batch_info: 批次信息
        """
        if not isinstance(design_params_batch, torch.Tensor):
            design_params_batch = torch.FloatTensor(design_params_batch)
        
        # 归一化
        if self.normalizer:
            params_np = design_params_batch.numpy()
            normalized_params = self.normalizer.transform_params(params_np)
            design_params_batch = torch.FloatTensor(normalized_params)
        
        design_params_batch = design_params_batch.to(self.device)
        
        # 预测
        start_time = time.time()
        with torch.no_grad():
            output, intermediate_outputs, debug_info = self.model(design_params_batch)
        inference_time = time.time() - start_time
        
        # 移到CPU
        rcs_predictions = output[:, 0].cpu().numpy()  # [B, 91, 91]
        
        # 反归一化
        if self.normalizer and not return_raw:
            for i in range(rcs_predictions.shape[0]):
                rcs_predictions[i] = self.normalizer.inverse_transform_rcs(rcs_predictions[i])
        
        batch_info = {
            'batch_size': len(design_params_batch),
            'total_inference_time': inference_time,
            'time_per_sample': inference_time / len(design_params_batch),
            'output_range': [output.min().item(), output.max().item()]
        }
        
        return rcs_predictions, batch_info


class RCSVisualizer:
    """RCS可视化器"""
    
    def __init__(self, theta_range=(-90, 90), phi_range=(-90, 90)):
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.theta_values = np.linspace(theta_range[0], theta_range[1], 91)
        self.phi_values = np.linspace(phi_range[0], phi_range[1], 91)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_rcs_heatmap(self, rcs_data, title="RCS热图", save_path=None, figsize=(10, 8)):
        """
        绘制RCS热图
        
        Args:
            rcs_data: RCS数据 [91, 91]
            title: 图标题
            save_path: 保存路径
            figsize: 图像尺寸
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建热图
        im = ax.imshow(rcs_data, 
                      extent=[self.theta_range[0], self.theta_range[1], 
                             self.phi_range[0], self.phi_range[1]],
                      origin='lower', 
                      cmap='jet',
                      aspect='auto')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RCS (dB)', rotation=270, labelpad=15)
        
        # 设置标签和标题
        ax.set_xlabel('俯仰角 θ (度)')
        ax.set_ylabel('偏航角 φ (度)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RCS热图已保存: {save_path}")
        
        return fig, ax
    
    def plot_rcs_3d(self, rcs_data, title="RCS 3D图", save_path=None, figsize=(12, 9)):
        """
        绘制RCS 3D表面图
        
        Args:
            rcs_data: RCS数据 [91, 91]
            title: 图标题
            save_path: 保存路径
            figsize: 图像尺寸
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格
        theta_grid, phi_grid = np.meshgrid(self.theta_values, self.phi_values)
        
        # 绘制3D表面
        surf = ax.plot_surface(theta_grid, phi_grid, rcs_data,
                              cmap='jet', linewidth=0, antialiased=True, alpha=0.8)
        
        # 添加颜色条
        fig.colorbar(surf, shrink=0.5, aspect=5, label='RCS (dB)')
        
        # 设置标签和标题
        ax.set_xlabel('俯仰角 θ (度)')
        ax.set_ylabel('偏航角 φ (度)')
        ax.set_zlabel('RCS (dB)')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RCS 3D图已保存: {save_path}")
        
        return fig, ax
    
    def plot_comparison(self, prediction, target, title_prefix="RCS对比", save_path=None):
        """
        绘制预测与真实值对比
        
        Args:
            prediction: 预测RCS [91, 91]
            target: 真实RCS [91, 91]
            title_prefix: 标题前缀
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 统一颜色范围
        vmin = min(np.min(prediction), np.min(target))
        vmax = max(np.max(prediction), np.max(target))
        
        # 预测值
        im1 = axes[0].imshow(prediction, 
                           extent=[self.theta_range[0], self.theta_range[1], 
                                  self.phi_range[0], self.phi_range[1]],
                           origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title_prefix} - 预测值')
        axes[0].set_xlabel('俯仰角 θ (度)')
        axes[0].set_ylabel('偏航角 φ (度)')
        plt.colorbar(im1, ax=axes[0], label='RCS (dB)')
        
        # 真实值
        im2 = axes[1].imshow(target, 
                           extent=[self.theta_range[0], self.theta_range[1], 
                                  self.phi_range[0], self.phi_range[1]],
                           origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{title_prefix} - 真实值')
        axes[1].set_xlabel('俯仰角 θ (度)')
        axes[1].set_ylabel('偏航角 φ (度)')
        plt.colorbar(im2, ax=axes[1], label='RCS (dB)')
        
        # 误差图
        error = np.abs(prediction - target)
        im3 = axes[2].imshow(error, 
                           extent=[self.theta_range[0], self.theta_range[1], 
                                  self.phi_range[0], self.phi_range[1]],
                           origin='lower', cmap='hot')
        axes[2].set_title(f'{title_prefix} - 绝对误差')
        axes[2].set_xlabel('俯仰角 θ (度)')
        axes[2].set_ylabel('偏航角 φ (度)')
        plt.colorbar(im3, ax=axes[2], label='绝对误差')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存: {save_path}")
        
        return fig, axes
    
    def plot_parameter_analysis(self, design_params, rcs_prediction, save_path=None):
        """
        绘制参数分析图
        
        Args:
            design_params: 设计参数 [9]
            rcs_prediction: RCS预测 [91, 91]
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 参数条形图
        param_names = [f'参数{i+1}' for i in range(len(design_params))]
        axes[0, 0].bar(param_names, design_params, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('设计参数')
        axes[0, 0].set_ylabel('参数值')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RCS热图
        im = axes[0, 1].imshow(rcs_prediction, 
                              extent=[self.theta_range[0], self.theta_range[1], 
                                     self.phi_range[0], self.phi_range[1]],
                              origin='lower', cmap='jet')
        axes[0, 1].set_title('RCS预测')
        axes[0, 1].set_xlabel('俯仰角 θ (度)')
        axes[0, 1].set_ylabel('偏航角 φ (度)')
        plt.colorbar(im, ax=axes[0, 1], label='RCS (dB)')
        
        # RCS统计信息
        rcs_stats = {
            '最小值': np.min(rcs_prediction),
            '最大值': np.max(rcs_prediction),
            '均值': np.mean(rcs_prediction),
            '标准差': np.std(rcs_prediction),
            '中位数': np.median(rcs_prediction)
        }
        
        stats_names = list(rcs_stats.keys())
        stats_values = list(rcs_stats.values())
        
        axes[1, 0].bar(stats_names, stats_values, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('RCS统计信息')
        axes[1, 0].set_ylabel('值')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # RCS值分布直方图
        axes[1, 1].hist(rcs_prediction.flatten(), bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('RCS值分布')
        axes[1, 1].set_xlabel('RCS值 (dB)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"参数分析图已保存: {save_path}")
        
        return fig, axes


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, predictor, visualizer=None):
        self.predictor = predictor
        self.visualizer = visualizer or RCSVisualizer()
    
    def evaluate_on_dataset(self, test_loader, output_dir='./evaluation'):
        """
        在测试数据集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            output_dir: 输出目录
            
        Returns:
            evaluation_results: 评估结果字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_predictions = []
        all_targets = []
        all_params = []
        inference_times = []
        
        print("开始模型评估...")
        
        with torch.no_grad():
            for batch_idx, (design_params, rcs_targets) in enumerate(test_loader):
                batch_start_time = time.time()
                
                # 预测
                predictions, batch_info = self.predictor.predict_batch(design_params.numpy())
                inference_times.append(batch_info['time_per_sample'])
                
                # 收集结果
                all_predictions.append(predictions)
                all_targets.append(rcs_targets.numpy()[:, 0])  # 移除通道维度
                all_params.append(design_params.numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  处理批次 {batch_idx}/{len(test_loader)}")
        
        # 合并所有结果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_params = np.concatenate(all_params, axis=0)
        
        print(f"评估完成，共处理 {len(all_predictions)} 个样本")
        
        # 计算指标
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        # 性能统计
        performance_stats = {
            'average_inference_time': np.mean(inference_times),
            'total_samples': len(all_predictions),
            'throughput_samples_per_second': 1.0 / np.mean(inference_times)
        }
        
        # 保存结果
        evaluation_results = {
            'metrics': metrics,
            'performance': performance_stats,
            'predictions_shape': all_predictions.shape,
            'targets_shape': all_targets.shape
        }
        
        # 生成评估报告
        self._generate_evaluation_report(evaluation_results, all_predictions, all_targets, 
                                       all_params, output_dir)
        
        return evaluation_results
    
    def _compute_metrics(self, predictions, targets):
        """计算评估指标"""
        # 展平数据用于计算指标
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        metrics = {
            'mse': mean_squared_error(target_flat, pred_flat),
            'mae': mean_absolute_error(target_flat, pred_flat),
            'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
            'r2_score': r2_score(target_flat, pred_flat),
            'max_error': np.max(np.abs(pred_flat - target_flat)),
            'mean_relative_error': np.mean(np.abs((pred_flat - target_flat) / (target_flat + 1e-8)))
        }
        
        return metrics
    
    def _generate_evaluation_report(self, results, predictions, targets, params, output_dir):
        """生成评估报告"""
        
        # 1. 保存指标
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("=== 模型评估报告 ===\n\n")
            f.write("预测精度指标:\n")
            for metric, value in results['metrics'].items():
                f.write(f"  {metric}: {value:.6f}\n")
            
            f.write("\n性能指标:\n")
            for metric, value in results['performance'].items():
                f.write(f"  {metric}: {value:.6f}\n")
        
        print(f"评估指标已保存: {metrics_path}")
        
        # 2. 绘制整体对比图
        sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            comparison_path = os.path.join(output_dir, f'comparison_sample_{idx}.png')
            self.visualizer.plot_comparison(
                predictions[idx], targets[idx], 
                title_prefix=f"样本 {idx}",
                save_path=comparison_path
            )
            plt.close()
        
        # 3. 绘制误差分布
        self._plot_error_distribution(predictions, targets, output_dir)
        
        # 4. 绘制参数相关性分析
        self._plot_parameter_correlation(params, predictions, targets, output_dir)
        
        print(f"评估报告已生成: {output_dir}")
    
    def _plot_error_distribution(self, predictions, targets, output_dir):
        """绘制误差分布"""
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 误差直方图
        axes[0, 0].hist(errors.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('预测误差分布')
        axes[0, 0].set_xlabel('误差值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 绝对误差直方图
        axes[0, 1].hist(abs_errors.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('绝对误差分布')
        axes[0, 1].set_xlabel('绝对误差值')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 预测vs真实值散点图
        sample_size = min(10000, len(predictions.flatten()))
        sample_indices = np.random.choice(len(predictions.flatten()), sample_size, replace=False)
        
        pred_sample = predictions.flatten()[sample_indices]
        target_sample = targets.flatten()[sample_indices]
        
        axes[1, 0].scatter(target_sample, pred_sample, alpha=0.5, s=1)
        axes[1, 0].plot([target_sample.min(), target_sample.max()], 
                       [target_sample.min(), target_sample.max()], 'r--', lw=2)
        axes[1, 0].set_title('预测值 vs 真实值')
        axes[1, 0].set_xlabel('真实值')
        axes[1, 0].set_ylabel('预测值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 样本误差箱线图
        sample_errors = [abs_errors[i].flatten() for i in range(min(20, len(abs_errors)))]
        axes[1, 1].boxplot(sample_errors, showfliers=False)
        axes[1, 1].set_title('各样本绝对误差分布')
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('绝对误差')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        error_dist_path = os.path.join(output_dir, 'error_distribution.png')
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"误差分布图已保存: {error_dist_path}")
    
    def _plot_parameter_correlation(self, params, predictions, targets, output_dir):
        """绘制参数相关性分析"""
        # 计算每个样本的预测误差
        sample_errors = np.mean(np.abs(predictions - targets), axis=(1, 2))
        
        # 创建相关性矩阵
        param_names = [f'参数{i+1}' for i in range(params.shape[1])]
        
        # 参数与误差的相关性
        correlations = []
        for i in range(params.shape[1]):
            corr = np.corrcoef(params[:, i], sample_errors)[0, 1]
            correlations.append(corr)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 参数与误差相关性
        axes[0].bar(param_names, correlations, color='lightblue', alpha=0.7, edgecolor='black')
        axes[0].set_title('设计参数与预测误差的相关性')
        axes[0].set_ylabel('相关系数')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 参数重要性热图（如果有足够样本）
        if len(params) > 10:
            param_corr_matrix = np.corrcoef(params.T)
            im = axes[1].imshow(param_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1].set_title('参数间相关性矩阵')
            axes[1].set_xticks(range(len(param_names)))
            axes[1].set_yticks(range(len(param_names)))
            axes[1].set_xticklabels(param_names, rotation=45)
            axes[1].set_yticklabels(param_names)
            plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        correlation_path = os.path.join(output_dir, 'parameter_correlation.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"参数相关性图已保存: {correlation_path}")


def demo_inference():
    """演示推理功能"""
    print("=== FiLM-UNet 推理演示 ===")
    
    # 创建预测器（使用随机初始化模型用于演示）
    predictor = RCSPredictor(model_path=None, device='cpu')
    
    # 创建可视化器
    visualizer = RCSVisualizer()
    
    # 创建演示数据
    demo_params = np.random.randn(9)
    print(f"演示参数: {demo_params}")
    
    # 单样本预测
    print("\n--- 单样本预测 ---")
    rcs_pred, pred_info = predictor.predict_single(demo_params)
    print(f"预测结果形状: {rcs_pred.shape}")
    print(f"推理时间: {pred_info['inference_time']:.4f}s")
    print(f"输出范围: [{pred_info['output_range'][0]:.4f}, {pred_info['output_range'][1]:.4f}]")
    
    # 可视化
    output_dir = './demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制RCS热图
    visualizer.plot_rcs_heatmap(rcs_pred, "演示RCS预测", 
                               save_path=os.path.join(output_dir, 'demo_heatmap.png'))
    plt.close()
    
    # 绘制参数分析
    visualizer.plot_parameter_analysis(demo_params, rcs_pred,
                                     save_path=os.path.join(output_dir, 'demo_analysis.png'))
    plt.close()
    
    # 批量预测
    print("\n--- 批量预测 ---")
    batch_params = np.random.randn(5, 9)
    batch_pred, batch_info = predictor.predict_batch(batch_params)
    print(f"批量预测形状: {batch_pred.shape}")
    print(f"批量推理信息: {batch_info}")
    
    print(f"\n演示完成！结果保存在: {output_dir}")
    
    # 清理
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("演示文件已清理")


if __name__ == "__main__":
    demo_inference()