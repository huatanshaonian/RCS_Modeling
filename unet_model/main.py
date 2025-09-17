#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FiLM-UNet主训练脚本
用于训练基于MLP+FiLM+U-Net的RCS预测模型
"""

import argparse
import os
import sys
import torch
import numpy as np
from datetime import datetime
import json
import logging

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from film_unet_model import FiLMUNetModel
from data_preprocessing import RCSDataLoader
from trainer import FiLMUNetTrainer
from inference import RCSPredictor, RCSVisualizer, ModelEvaluator


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FiLM-UNet RCS预测模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../parameter',
                       help='数据目录路径')
    parser.add_argument('--params_file', type=str, default='parameters_sorted.csv',
                       help='参数文件名')
    parser.add_argument('--rcs_dir', type=str, default='csv_output',
                       help='RCS数据目录名')
    parser.add_argument('--num_models', type=int, default=100,
                       help='使用的模型数量')
    parser.add_argument('--frequency', type=str, default='1.5G', choices=['1.5G', '3G', 'both'],
                       help='分析频率 (1.5G, 3G, 或 both)')
    parser.add_argument('--frequencies', type=str, nargs='+', default=None,
                       help='多频率列表，如 --frequencies 1.5G 3G')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批大小')
    parser.add_argument('--epochs', type=int, default=500,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--train_size', type=int, default=None,
                       help='训练集绝对大小 (如果设置则忽略test_size，用于与AE方法统一)')
    
    # 损失函数权重
    parser.add_argument('--lambda_mse', type=float, default=1.0,
                       help='MSE损失权重')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                       help='平滑损失权重')
    parser.add_argument('--lambda_physics', type=float, default=0.05,
                       help='物理约束损失权重')
    parser.add_argument('--lambda_multiscale', type=float, default=0.1,
                       help='多尺度损失权重')
    
    # 设备和保存
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--run_name', type=str, default=None,
                       help='自定义运行名称（替代默认的时间戳）')
    parser.add_argument('--save_freq', type=int, default=50,
                       help='模型保存频率')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                       help='早停耐心值')
    
    # 数据增强
    parser.add_argument('--enable_augmentation', action='store_true',
                       help='启用数据增强')
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='噪声标准差')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Mixup参数')
    
    # 模式
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'inference', 'demo'],
                       help='运行模式')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（用于评估或推理）')
    
    return parser.parse_args()


class TeeLogger:
    """同时输出到文件和控制台的日志器"""
    def __init__(self, log_file):
        self.log_file = open(log_file, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        
    def write(self, message):
        # 写入文件
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()
        # 写入控制台
        self.original_stdout.write(message)
        self.original_stdout.flush()
        
    def flush(self):
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.flush()
        self.original_stdout.flush()
    
    def close(self):
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()
        sys.stdout = self.original_stdout


def setup_logging(logs_dir):
    """设置日志配置"""
    log_file = os.path.join(logs_dir, 'training.log')
    
    # 创建Tee日志器
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    
    return tee_logger


def setup_output_directory(output_dir, run_name=None):
    """设置输出目录"""
    if run_name:
        # 使用自定义名称，如果已存在则添加数字后缀
        run_dir = os.path.join(output_dir, run_name)
        counter = 1
        base_run_dir = run_dir
        while os.path.exists(run_dir):
            run_dir = f"{base_run_dir}_{counter}"
            counter += 1
    else:
        # 使用默认时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(output_dir, f'run_{timestamp}')
    
    os.makedirs(run_dir, exist_ok=True)
    
    # 创建子目录
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    logs_dir = os.path.join(run_dir, 'logs')
    results_dir = os.path.join(run_dir, 'results')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return run_dir, checkpoints_dir, logs_dir, results_dir


def save_config(args, config_path):
    """保存训练配置"""
    config = vars(args)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"训练配置已保存: {config_path}")


def train_single_frequency(args, frequency, freq_dir):
    """训练单一频率的模型"""
    print(f"\n=== 开始训练FiLM-UNet模型 ({frequency}) ===")
    
    try:
        # 加载数据
        print(f"\n--- 数据加载 ({frequency}) ---")
        data_loader = RCSDataLoader(
            data_dir=args.data_dir,
            params_file=args.params_file,
            rcs_dir=args.rcs_dir
        )
        
        design_params, rcs_data = data_loader.load_data(
            num_models=args.num_models,
            frequency=frequency,
            verbose=True
        )
        
        # 创建数据集
        print(f"\n--- 数据集创建 ({frequency}) ---")
        augment_params = None
        if args.enable_augmentation:
            augment_params = {
                'noise_std': args.noise_std,
                'mixup_alpha': args.mixup_alpha,
                'enable_mixup': True,
                'enable_noise': True,
                'enable_param_jitter': True
            }
        
        train_dataset, test_dataset, normalizer, train_data_raw, test_data_raw = data_loader.create_datasets(
            design_params, rcs_data,
            test_size=args.test_size,
            train_size=args.train_size,
            apply_normalization=True,
            apply_augmentation=args.enable_augmentation,
            augment_params=augment_params,
            output_dir=freq_dir,
            save_indices=True
        )
        
        train_loader, test_loader = data_loader.create_dataloaders(
            train_dataset, test_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # 创建模型
        print(f"\n--- 模型创建 ({frequency}) ---")
        model = FiLMUNetModel()
        model_summary = model.get_model_summary()
        print("模型摘要:")
        print(f"  架构: {model_summary['architecture']}")
        print(f"  总参数量: {model_summary['total_parameters']:,}")
        print(f"  内存使用: {model_summary['memory_usage']}")
        
        # 创建训练器并开始训练... (继续原有逻辑)
        return freq_dir
        
    except Exception as e:
        print(f"\n训练过程中发生错误 ({frequency}): {e}")
        import traceback
        traceback.print_exc()
        return None


def train_model(args):
    """训练模型"""
    # 设置输出目录
    run_dir, checkpoints_dir, logs_dir, results_dir = setup_output_directory(args.output_dir, args.run_name)
    
    # 设置日志 (在初始输出之后)
    tee_logger = setup_logging(logs_dir)
    
    print("=== 开始训练FiLM-UNet模型 ===")
    print(f"输出目录: {run_dir}")
    if args.run_name:
        print(f"使用自定义运行名称: {args.run_name}")
    
    # 保存配置
    config_path = os.path.join(run_dir, 'config.json')
    save_config(args, config_path)
    print(f"配置文件已保存: {config_path}")
    
    try:
        # 处理频率参数
        print("\n--- 频率配置 ---")
        frequencies_to_train = []
        
        if args.frequencies:
            # 使用--frequencies参数（优先级最高）
            frequencies_to_train = args.frequencies
            print(f"使用指定的频率列表: {frequencies_to_train}")
        elif args.frequency == 'both':
            # 使用both选项
            frequencies_to_train = ['1.5G', '3G']
            print(f"训练双频率: {frequencies_to_train}")
        else:
            # 单一频率
            frequencies_to_train = [args.frequency]
            print(f"训练单一频率: {frequencies_to_train}")
        
        # 为每个频率训练模型
        freq_results = {}
        for freq_idx, frequency in enumerate(frequencies_to_train):
            print(f"\n=== 频率 {freq_idx + 1}/{len(frequencies_to_train)}: {frequency} ===")
            
            # 创建频率特定的输出目录
            freq_dir = os.path.join(run_dir, f"freq_{frequency}")
            freq_checkpoints = os.path.join(freq_dir, 'checkpoints')
            freq_results_dir = os.path.join(freq_dir, 'results')
            os.makedirs(freq_checkpoints, exist_ok=True)
            os.makedirs(freq_results_dir, exist_ok=True)
            
            # 加载数据
            print(f"\n--- 数据加载 ({frequency}) ---")
            data_loader = RCSDataLoader(
                data_dir=args.data_dir,
                params_file=args.params_file,
                rcs_dir=args.rcs_dir
            )
            
            design_params, rcs_data = data_loader.load_data(
                num_models=args.num_models,
                frequency=frequency,
                verbose=True
            )
        
            # 创建数据集
            print(f"\n--- 数据集创建 ({frequency}) ---")
            augment_params = None
            if args.enable_augmentation:
                augment_params = {
                    'noise_std': args.noise_std,
                    'mixup_alpha': args.mixup_alpha,
                    'enable_mixup': True,
                    'enable_noise': True,
                    'enable_param_jitter': True
                }
            
            train_dataset, test_dataset, normalizer, train_data_raw, test_data_raw = data_loader.create_datasets(
                design_params, rcs_data,
                test_size=args.test_size,
                train_size=args.train_size,
                apply_normalization=True,
                apply_augmentation=args.enable_augmentation,
                augment_params=augment_params,
                output_dir=freq_dir,  # 使用频率特定目录
                save_indices=True
            )
            
            train_loader, test_loader = data_loader.create_dataloaders(
                train_dataset, test_dataset,
                batch_size=args.batch_size,
                shuffle=True
            )
        
            # 创建模型 (每个频率独立的模型)
            print(f"\n--- 模型创建 ({frequency}) ---")
            model = FiLMUNetModel()
            model_summary = model.get_model_summary()
            print("模型摘要:")
            print(f"  架构: {model_summary['architecture']}")
            print(f"  总参数量: {model_summary['total_parameters']:,}")
            print(f"  内存使用: {model_summary['memory_usage']}")
            
            # 创建训练器
            print(f"\n--- 训练器配置 ({frequency}) ---")
            loss_weights = {
                'lambda_mse': args.lambda_mse,
                'lambda_smooth': args.lambda_smooth,
                'lambda_physics': args.lambda_physics,
                'lambda_multiscale': args.lambda_multiscale
            }
            
            optimizer_params = {
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay
            }
            
            scheduler_params = {
                'T_max': args.epochs,
                'eta_min': 1e-6
            }
            
            trainer = FiLMUNetTrainer(
                model=model,
                device=args.device,
                loss_weights=loss_weights,
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params
            )
            
            # 开始训练
            print(f"\n--- 开始训练 ({frequency}) ---")
            best_val_loss = trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,
                epochs=args.epochs,
                save_dir=freq_checkpoints,
                save_freq=args.save_freq,
                early_stopping_patience=args.early_stopping_patience
            )
            
            print(f"\n频率 {frequency} 训练完成！最佳验证损失: {best_val_loss:.6f}")
            
            # 评估模型
            print(f"\n--- 模型评估 ({frequency}) ---")
            best_model_path = os.path.join(freq_checkpoints, 'best_model.pth')
            if os.path.exists(best_model_path):
                predictor = RCSPredictor(best_model_path, device=trainer.device)
                predictor.set_normalizer(normalizer)
                
                visualizer = RCSVisualizer()
                evaluator = ModelEvaluator(predictor, visualizer)
                
                # 获取模型编号映射
                train_model_indices = None
                test_model_indices = None
                if hasattr(data_loader, 'available_models') and data_loader.available_models is not None:
                    # 转换为真实模型编号
                    train_model_indices = [data_loader.available_models[i] for i in train_data_raw['indices']]
                    test_model_indices = [data_loader.available_models[i] for i in test_data_raw['indices']]
                
                # 创建训练集评估目录
                train_results_dir = os.path.join(freq_results_dir, 'train_evaluation')
                
                # 评估训练集
                print(f"\n--- 评估训练集 ({frequency}) ---")
                train_evaluation_results = evaluator.evaluate_with_raw_targets(
                    train_data_raw['params'], train_data_raw['rcs'], train_results_dir,
                    model_indices=train_model_indices, dataset_type='train'
                )
                
                print(f"频率 {frequency} 训练集评估结果:")
                for metric, value in train_evaluation_results['metrics'].items():
                    print(f"  {metric}: {value:.6f}")
                
                # 评估测试集  
                print(f"\n--- 评估测试集 ({frequency}) ---")
                evaluation_results = evaluator.evaluate_with_raw_targets(
                    test_data_raw['params'], test_data_raw['rcs'], freq_results_dir,
                    model_indices=test_model_indices, dataset_type='test'
                )
                
                print(f"频率 {frequency} 评估结果:")
                for metric, value in evaluation_results['metrics'].items():
                    print(f"  {metric}: {value:.6f}")
                
                # 保存频率结果
                freq_results[frequency] = {
                    'best_val_loss': best_val_loss,
                    'train_evaluation_results': train_evaluation_results,
                    'test_evaluation_results': evaluation_results,
                    'model_path': best_model_path,
                    'results_dir': freq_results_dir,
                    'train_results_dir': train_results_dir
                }
            
            print(f"频率 {frequency} 的结果已保存到: {freq_dir}")
        
        # 汇总多频率结果
        print(f"\n=== 多频率训练总结 ===")
        for freq, result in freq_results.items():
            print(f"频率 {freq}:")
            print(f"  最佳验证损失: {result['best_val_loss']:.6f}")
            print(f"  模型路径: {result['model_path']}")
        
        # 保存多频率结果摘要
        summary_path = os.path.join(run_dir, 'multi_frequency_summary.json')
        summary_data = {
            'frequencies': list(freq_results.keys()),
            'results': {freq: {
                'best_val_loss': float(result['best_val_loss']),
                'model_path': result['model_path'],
                'results_dir': result['results_dir']
            } for freq, result in freq_results.items()}
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n所有结果已保存到: {run_dir}")
        print(f"多频率摘要文件: {summary_path}")
        return run_dir
        
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 恢复标准输出
        if 'tee_logger' in locals():
            tee_logger.close()
        print("日志记录已完成")


def evaluate_model(args):
    """评估模型"""
    print("=== 模型评估模式 ===")
    
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 加载数据
    print("加载测试数据...")
    data_loader = RCSDataLoader(
        data_dir=args.data_dir,
        params_file=args.params_file,
        rcs_dir=args.rcs_dir
    )
    
    design_params, rcs_data = data_loader.load_data(
        num_models=args.num_models,
        frequency=args.frequency,
        verbose=True
    )
    
    train_dataset, test_dataset, normalizer, train_data_raw, test_data_raw = data_loader.create_datasets(
        design_params, rcs_data,
        test_size=args.test_size,
        train_size=args.train_size,
        apply_normalization=True,
        apply_augmentation=False,
        output_dir=None,  # evaluate模式不保存索引
        save_indices=False
    )
    
    _, test_loader = data_loader.create_dataloaders(
        train_dataset, test_dataset,
        batch_size=args.batch_size
    )
    
    # 创建预测器和评估器
    predictor = RCSPredictor(args.model_path, device=args.device)
    predictor.set_normalizer(normalizer)
    
    visualizer = RCSVisualizer()
    evaluator = ModelEvaluator(predictor, visualizer)
    
    # 评估
    output_dir = os.path.join(args.output_dir, 'evaluation')
    # 使用原始dB数据进行评估 (避免归一化/反归一化往返损失)
    evaluation_results = evaluator.evaluate_with_raw_targets(
        test_data_raw['params'], test_data_raw['rcs'], output_dir,
        model_indices=None, dataset_type='test'
    )
    
    print("评估完成！")
    print("评估结果:")
    for metric, value in evaluation_results['metrics'].items():
        print(f"  {metric}: {value:.6f}")
    
    print(f"详细结果保存在: {output_dir}")


def inference_demo(args):
    """推理演示"""
    print("=== 推理演示模式 ===")
    
    # 创建预测器
    if args.model_path and os.path.exists(args.model_path):
        predictor = RCSPredictor(args.model_path, device=args.device)
        print(f"使用训练好的模型: {args.model_path}")
    else:
        predictor = RCSPredictor(model_path=None, device=args.device)
        print("使用随机初始化模型进行演示")
    
    # 创建可视化器
    visualizer = RCSVisualizer()
    
    # 创建演示数据
    print("\n生成演示数据...")
    demo_params = np.random.randn(9)
    print(f"演示参数: {demo_params}")
    
    # 预测
    print("\n执行预测...")
    rcs_prediction, pred_info = predictor.predict_single(demo_params)
    
    print(f"预测结果:")
    print(f"  形状: {rcs_prediction.shape}")
    print(f"  推理时间: {pred_info['inference_time']:.4f}s")
    print(f"  值范围: [{rcs_prediction.min():.4f}, {rcs_prediction.max():.4f}]")
    
    # 可视化
    output_dir = os.path.join(args.output_dir, 'inference_demo')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n生成可视化结果...")
    
    # RCS热图
    visualizer.plot_rcs_heatmap(
        rcs_prediction, 
        "演示RCS预测", 
        save_path=os.path.join(output_dir, 'rcs_heatmap.png')
    )
    
    # 3D图
    visualizer.plot_rcs_3d(
        rcs_prediction,
        "演示RCS 3D图",
        save_path=os.path.join(output_dir, 'rcs_3d.png')
    )
    
    # 参数分析
    visualizer.plot_parameter_analysis(
        demo_params,
        rcs_prediction,
        save_path=os.path.join(output_dir, 'parameter_analysis.png')
    )
    
    print(f"演示完成！结果保存在: {output_dir}")


def main():
    """主函数"""
    args = parse_arguments()
    
    print("FiLM-UNet RCS预测模型")
    print("=" * 50)
    print(f"运行模式: {args.mode}")
    print(f"设备: {args.device}")
    print(f"输出目录: {args.output_dir}")
    
    try:
        if args.mode == 'train':
            train_model(args)
        elif args.mode == 'evaluate':
            evaluate_model(args)
        elif args.mode == 'inference' or args.mode == 'demo':
            inference_demo(args)
        else:
            print(f"未知模式: {args.mode}")
            return 1
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
