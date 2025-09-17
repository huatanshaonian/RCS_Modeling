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
    
    def __init__(self, theta_range=(-45, 45), phi_range=(45, 135)):
        self.theta_range = theta_range  # 方位角范围（横轴）
        self.phi_range = phi_range      # 俯仰角范围（纵轴）
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
        
        # 创建热图 - 转置数据以确保与AE一致的轴对应关系
        # extent=[left, right, bottom, top] = [theta_min, theta_max, phi_min, phi_max]
        # origin='upper'确保俯仰角45度在上面，135度在下面
        im = ax.imshow(rcs_data.T, 
                      extent=[self.theta_range[0], self.theta_range[1], 
                             self.phi_range[1], self.phi_range[0]],  # 交换phi范围顺序
                      origin='upper', 
                      cmap='jet',
                      aspect='auto')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RCS (dB)', rotation=270, labelpad=15)
        
        # 设置标签和标题 (横轴=theta偏航角, 纵轴=phi俯仰角)
        ax.set_xlabel('偏航角 θ (度)')
        ax.set_ylabel('俯仰角 φ (度)')
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
        
        # 绘制3D表面 - 转置数据以确保与AE一致的轴对应关系
        surf = ax.plot_surface(theta_grid, phi_grid, rcs_data.T,
                              cmap='jet', linewidth=0, antialiased=True, alpha=0.8)
        
        # 添加颜色条
        fig.colorbar(surf, shrink=0.5, aspect=5, label='RCS (dB)')
        
        # 设置标签和标题
        ax.set_xlabel('偏航角 θ (度)')
        ax.set_ylabel('俯仰角 φ (度)')
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
        
        # 预测值 - 转置数据以确保与AE一致的轴对应关系
        im1 = axes[0].imshow(prediction.T, 
                           extent=[self.theta_range[0], self.theta_range[1], 
                                  self.phi_range[0], self.phi_range[1]],
                           origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title_prefix} - 预测值')
        axes[0].set_xlabel('方位角 θ (度)')
        axes[0].set_ylabel('俯仰角 φ (度)')
        plt.colorbar(im1, ax=axes[0], label='RCS (dB)')
        
        # 真实值 - 转置数据以确保与AE一致的轴对应关系
        im2 = axes[1].imshow(target.T, 
                           extent=[self.theta_range[0], self.theta_range[1], 
                                  self.phi_range[0], self.phi_range[1]],
                           origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{title_prefix} - 真实值')
        axes[1].set_xlabel('方位角 θ (度)')
        axes[1].set_ylabel('俯仰角 φ (度)')
        plt.colorbar(im2, ax=axes[1], label='RCS (dB)')
        
        # 误差图 - 转置数据以确保与AE一致的轴对应关系
        error = np.abs(prediction - target)
        im3 = axes[2].imshow(error.T, 
                           extent=[self.theta_range[0], self.theta_range[1], 
                                  self.phi_range[0], self.phi_range[1]],
                           origin='lower', cmap='hot')
        axes[2].set_title(f'{title_prefix} - 绝对误差')
        axes[2].set_xlabel('方位角 θ (度)')
        axes[2].set_ylabel('俯仰角 φ (度)')
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
        
        # RCS热图 - 转置数据以确保与AE一致的轴对应关系
        im = axes[0, 1].imshow(rcs_prediction.T, 
                              extent=[self.theta_range[0], self.theta_range[1], 
                                     self.phi_range[0], self.phi_range[1]],
                              origin='lower', cmap='jet')
        axes[0, 1].set_title('RCS预测')
        axes[0, 1].set_xlabel('方位角 θ (度)')
        axes[0, 1].set_ylabel('俯仰角 φ (度)')
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
                # 确保目标数据形状正确
                if rcs_targets.dim() == 4:  # [B, 1, 91, 91] -> [B, 91, 91]
                    target_data = rcs_targets.numpy()[:, 0]
                else:  # [B, 91, 91] -> 保持不变
                    target_data = rcs_targets.numpy()
                
                # 对目标数据进行反归一化 (与预测数据保持一致)
                if self.predictor.normalizer:
                    for i in range(target_data.shape[0]):
                        target_data[i] = self.predictor.normalizer.inverse_transform_rcs(target_data[i])
                
                all_targets.append(target_data)
                all_params.append(design_params.numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  处理批次 {batch_idx}/{len(test_loader)}")
        
        # 合并所有结果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_params = np.concatenate(all_params, axis=0)
        
        print(f"评估完成，共处理 {len(all_predictions)} 个样本")
        print(f"预测数据形状: {all_predictions.shape}")
        print(f"目标数据形状: {all_targets.shape}")
        print(f"参数数据形状: {all_params.shape}")
        
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
        print(f"指标计算 - 预测形状: {predictions.shape}, 目标形状: {targets.shape}")
        
        # 展平数据用于计算指标
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        print(f"展平后 - 预测: {pred_flat.shape}, 目标: {target_flat.shape}")
        
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
    
    def _plot_error_distribution(self, predictions, targets, output_dir, dataset_type='test'):
        """绘制误差分布"""
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 误差直方图
        axes[0, 0].hist(errors.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title(f'{dataset_type.upper()}集预测误差分布')
        axes[0, 0].set_xlabel('误差值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 绝对误差直方图
        axes[0, 1].hist(abs_errors.flatten(), bins=100, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title(f'{dataset_type.upper()}集绝对误差分布')
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
        axes[1, 0].set_title(f'{dataset_type.upper()}集预测值 vs 真实值')
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
        
        error_dist_path = os.path.join(output_dir, f'error_distribution_{dataset_type}.png')
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"误差分布图已保存: {error_dist_path}")
    
    def _plot_parameter_correlation(self, params, predictions, targets, output_dir, dataset_type='test'):
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
        
        correlation_path = os.path.join(output_dir, f'parameter_correlation_{dataset_type}.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"参数相关性图已保存: {correlation_path}")
    
    def _compare_rcs_statistics(self, predictions, targets, output_dir, dataset_type='test'):
        """比较预测和真实RCS的统计数据 (引用现有函数)"""
        try:
            # 尝试导入现有的统计比较函数
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model_analysis import compare_statistics
            
            # 定义角度值 (与data_loader.py一致)
            theta_values = np.linspace(-45, 45, 91)  # 偏航角
            phi_values = np.linspace(45, 135, 91)    # 俯仰角
            
            # 计算原始数据统计
            def calculate_rcs_statistics(rcs_data, prefix):
                """计算RCS统计数据"""
                statistics = []
                for i in range(len(rcs_data)):
                    rcs_2d = rcs_data[i]  # 已经是[91, 91]格式
                    
                    # 找出最大值和最小值的2D索引
                    max_idx = np.unravel_index(np.argmax(rcs_2d), rcs_2d.shape)
                    min_idx = np.unravel_index(np.argmin(rcs_2d), rcs_2d.shape)
                    
                    # 获取最大值、最小值及其对应的角度值
                    max_value_dbsm = rcs_2d[max_idx]
                    max_theta = theta_values[max_idx[1]]  # 列索引对应theta
                    max_phi = phi_values[max_idx[0]]     # 行索引对应phi
                    
                    min_value_dbsm = rcs_2d[min_idx]
                    min_theta = theta_values[min_idx[1]]
                    min_phi = phi_values[min_idx[0]]
                    
                    # 计算基本统计量
                    stats = {
                        '模型': f'{prefix}{i+1}',
                        '均值(dBsm)': np.mean(rcs_2d),
                        '中位数(dBsm)': np.median(rcs_2d),
                        '极大值(dBsm)': max_value_dbsm,
                        '极大值θ': max_theta,
                        '极大值φ': max_phi,
                        '极小值(dBsm)': min_value_dbsm,
                        '极小值θ': min_theta,
                        '极小值φ': min_phi,
                        '极差': max_value_dbsm - min_value_dbsm,
                        '标准差': np.std(rcs_2d)
                    }
                    statistics.append(stats)
                
                return pd.DataFrame(statistics)
            
            # 计算预测数据和真实数据的统计
            pred_stats = calculate_rcs_statistics(predictions, f'{dataset_type.upper()}_Pred_')
            true_stats = calculate_rcs_statistics(targets, f'{dataset_type.upper()}_True_')
            
            # 调用现有的比较函数
            compare_statistics(true_stats, pred_stats, output_dir)
            
            print(f"RCS统计数据比较已保存: {os.path.join(output_dir, 'stats_comparison.csv')}")
            
        except ImportError as e:
            print(f"无法导入统计比较函数: {e}")
            print("跳过统计数据比较...")
        except Exception as e:
            print(f"统计数据比较过程中出错: {e}")
            print("跳过统计数据比较...")
    
    def evaluate_with_raw_targets(self, test_params, test_rcs_raw, output_dir='./evaluation_raw', 
                                model_indices=None, dataset_type='test'):
        """
        使用原始dB数据作为真实值进行评估（避免归一化/反归一化的往返损失）
        
        Args:
            test_params: 测试参数 [N, 9]
            test_rcs_raw: 原始测试RCS数据 (dB值) [N, 91, 91]  
            output_dir: 输出目录
            model_indices: 模型编号列表 (1-based，如[1,2,5])，用于显示原始模型ID
            dataset_type: 数据集类型 ('test' 或 'train')
            
        Returns:
            evaluation_results: 评估结果字典
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用原始dB数据进行评估...")
        print(f"测试样本数: {len(test_params)}")
        print(f"原始RCS数据范围: [{test_rcs_raw.min():.2f}, {test_rcs_raw.max():.2f}] dB")
        
        # 批量预测
        start_time = time.time()
        predictions, batch_info = self.predictor.predict_batch(test_params)
        inference_time = time.time() - start_time
        
        print(f"预测完成，用时 {inference_time:.2f} 秒")
        print(f"预测数据范围: [{predictions.min():.2f}, {predictions.max():.2f}] dB")
        
        # 计算评估指标 (直接使用原始dB数据，无归一化往返损失)
        mse = mean_squared_error(test_rcs_raw.reshape(-1), predictions.reshape(-1))
        mae = mean_absolute_error(test_rcs_raw.reshape(-1), predictions.reshape(-1))
        r2 = r2_score(test_rcs_raw.reshape(-1), predictions.reshape(-1))
        
        # 计算样本级别的误差
        sample_mse = np.mean((predictions - test_rcs_raw) ** 2, axis=(1, 2))
        sample_mae = np.mean(np.abs(predictions - test_rcs_raw), axis=(1, 2))
        
        results = {
            'metrics': {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'RMSE': np.sqrt(mse),
                'Sample_MSE_mean': np.mean(sample_mse),
                'Sample_MSE_std': np.std(sample_mse),
                'Sample_MAE_mean': np.mean(sample_mae),
                'Sample_MAE_std': np.std(sample_mae)
            },
            'performance': {
                'total_inference_time': inference_time,
                'samples_per_second': len(test_params) / inference_time,
                'time_per_sample': inference_time / len(test_params)
            }
        }
        
        print(f"\n=== 评估结果 (使用原始dB数据) ===")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f} dB")
        print(f"R²: {r2:.6f}")
        print(f"RMSE: {np.sqrt(mse):.6f} dB")
        
        # 保存评估指标
        metrics_path = os.path.join(output_dir, 'evaluation_metrics_raw.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("=== 模型评估结果 (原始dB数据) ===\n\n")
            f.write("预测精度指标:\n")
            for metric, value in results['metrics'].items():
                f.write(f"  {metric}: {value:.6f}\n")
            
            f.write("\n性能指标:\n")
            for metric, value in results['performance'].items():
                f.write(f"  {metric}: {value:.6f}\n")
        
        print(f"评估指标已保存: {metrics_path}")
        
        # 绘制对比图 (真实值直接使用原始dB数据)
        sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            # 确定标题和文件名
            if model_indices is not None and idx < len(model_indices):
                model_id = model_indices[idx]
                title_prefix = f"{dataset_type.upper()} 样本 {idx} (模型 {model_id:03d})"
                filename = f'comparison_{dataset_type}_sample_{idx}_model_{model_id:03d}.png'
            else:
                title_prefix = f"{dataset_type.upper()} 样本 {idx}"
                filename = f'comparison_{dataset_type}_sample_{idx}.png'
            
            comparison_path = os.path.join(output_dir, filename)
            self.visualizer.plot_comparison(
                predictions[idx], test_rcs_raw[idx],  # 直接使用原始dB数据
                title_prefix=title_prefix,
                save_path=comparison_path
            )
            plt.close()
        
        # 绘制误差分布
        self._plot_error_distribution(predictions, test_rcs_raw, output_dir, dataset_type)
        
        # 绘制参数相关性分析
        self._plot_parameter_correlation(test_params, predictions, test_rcs_raw, output_dir, dataset_type)
        
        # 添加统计数据比较 (引用现有函数)
        self._compare_rcs_statistics(predictions, test_rcs_raw, output_dir, dataset_type)
        
        print(f"评估报告已生成: {output_dir}")
        
        return results


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