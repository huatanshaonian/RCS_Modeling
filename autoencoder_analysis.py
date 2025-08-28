"""
基于深度学习Autoencoder的RCS数据降维分析模块
使用PyTorch实现变分自编码器(VAE)和标准自编码器(AE)用于RCS数据降维

重构后的主分析模块，协调各个子模块的功能
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
try:
    from .autoencoder_models import StandardAutoencoder, VariationalAutoencoder
    from .autoencoder_training import train_autoencoder, evaluate_model
    from .autoencoder_utils import (
        validate_data_integrity, get_device_info, get_optimal_batch_size,
        create_optimized_data_loaders, log_info, cleanup_gpu_memory, PYTORCH_AVAILABLE
    )
    from .autoencoder_visualization import (
        plot_training_history, visualize_latent_space, analyze_reconstruction_error,
        generate_comparison_analysis, compare_with_pod_results
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from autoencoder_models import StandardAutoencoder, VariationalAutoencoder
    from autoencoder_training import train_autoencoder, evaluate_model
    from autoencoder_utils import (
        validate_data_integrity, get_device_info, get_optimal_batch_size,
        create_optimized_data_loaders, log_info, cleanup_gpu_memory, PYTORCH_AVAILABLE
    )
    from autoencoder_visualization import (
        plot_training_history, visualize_latent_space, analyze_reconstruction_error,
        generate_comparison_analysis, compare_with_pod_results
    )


def perform_autoencoder_analysis(rcs_data, theta_values, phi_values, param_data, param_names,
                                 freq_label, output_dir, train_indices, test_indices=None,
                                 latent_dims=[5, 10, 15, 20], model_types=['standard', 'vae'],
                                 device='auto', available_models=None):
    """
    执行基于Autoencoder的RCS数据降维分析

    参数:
    rcs_data: RCS数据矩阵 [n_samples, n_features]
    theta_values: theta角度值
    phi_values: phi角度值
    param_data: 设计参数数据
    param_names: 参数名称列表
    freq_label: 频率标签
    output_dir: 输出目录
    train_indices: 训练集索引
    test_indices: 测试集索引（可选）
    latent_dims: 潜在空间维度列表
    model_types: 模型类型列表
    device: 计算设备
    available_models: 可用模型索引

    返回:
    results: 分析结果字典
    """
    try:
        validate_data_integrity(rcs_data, param_data, available_models)
    except Exception as e:
        print(f"❌ 数据完整性检查失败: {e}")
        return {}

    if not PYTORCH_AVAILABLE:
        print("PyTorch不可用，跳过Autoencoder分析")
        return {}

    # 获取设备配置
    if device == 'auto':
        device, device_info = get_device_info()
        print(f"🖥️  {device_info}")
    else:
        device = torch.device(device)
        print(f"🖥️  使用指定设备: {device}")

    # 创建输出目录
    ae_dir = os.path.join(output_dir, 'autoencoder')
    os.makedirs(ae_dir, exist_ok=True)

    # 数据预处理
    print("准备Autoencoder训练数据...")

    # 分割训练和测试数据
    rcs_train = rcs_data[train_indices]
    param_train = param_data[train_indices]

    if test_indices is not None and len(test_indices) > 0:
        rcs_test = rcs_data[test_indices]
        param_test = param_data[test_indices]
    else:
        rcs_test = None
        param_test = None

    # 数据标准化
    scaler = StandardScaler()
    rcs_train_scaled = scaler.fit_transform(rcs_train)
    if rcs_test is not None:
        rcs_test_scaled = scaler.transform(rcs_test)

    # 转换为张量
    print("🔄 优化数据加载...")
    rcs_train_tensor = torch.from_numpy(rcs_train_scaled.astype(np.float32)).contiguous()
    if rcs_test is not None:
        rcs_test_tensor = torch.from_numpy(rcs_test_scaled.astype(np.float32)).contiguous()
    else:
        rcs_test_tensor = None

    # 数据集信息
    input_dim = rcs_train_scaled.shape[1]
    print(f"📊 数据集信息:")
    print(f"  训练集形状: {rcs_train_scaled.shape}")
    if rcs_test is not None:
        print(f"  测试集形状: {rcs_test_scaled.shape}")

    # 检查数据集大小
    if len(rcs_train_scaled) == 0:
        print("❌ 训练数据集为空")
        return {}

    # 优化批次大小
    optimal_batch_size = get_optimal_batch_size(input_dim, device)
    max_reasonable_batch_size = max(1, len(rcs_train_scaled) // 4)
    optimal_batch_size = min(optimal_batch_size, max_reasonable_batch_size)

    print(f"📦 使用批次大小: {optimal_batch_size}")

    # 创建数据加载器
    train_loader, val_loader = create_optimized_data_loaders(
        rcs_train_tensor, rcs_test_tensor, device, optimal_batch_size
    )

    if train_loader is None or val_loader is None:
        print("❌ 数据加载器创建失败")
        return {}

    results = {}

    print(f"输入维度: {input_dim}")
    print(f"训练样本数: {len(rcs_train_scaled)}")
    print(f"将训练以下配置:")
    print(f"  潜在维度: {latent_dims}")
    print(f"  模型类型: {model_types}")

    # 训练不同配置的模型
    for model_type in model_types:
        for latent_dim in latent_dims:
            config_name = f"{model_type}_latent{latent_dim}"
            print(f"\n开始训练配置: {config_name}")

            config_dir = os.path.join(ae_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)

            # 创建模型
            if model_type == 'standard':
                model = StandardAutoencoder(input_dim, latent_dim)
            else:  # VAE
                model = VariationalAutoencoder(input_dim, latent_dim)

            # 训练模型
            try:
                trained_model, train_losses, val_losses = train_autoencoder(
                    model, train_loader, val_loader, epochs=200,
                    device=device, model_type=model_type, output_dir=config_dir
                )

                # 检查训练是否成功完成
                if len(train_losses) == 0:
                    print(f"  ⚠️ 警告：{config_name} 训练失败，没有记录训练损失")
                    continue
                
                # 评估训练集
                train_results = evaluate_model(trained_model, rcs_train_tensor, scaler, device, model_type)
                
                print(f"  训练集重构MSE: {train_results['mse']:.6f}")
                print(f"  训练集重构R^2: {train_results['r2']:.6f}")
                print(f"  训练轮数: {len(train_losses)}, 最终训练损失: {train_losses[-1]:.6f}")

                # 保存结果
                config_results = {
                    'model_type': model_type,
                    'latent_dim': latent_dim,
                    'model': trained_model,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_latent': train_results['latent'],
                    'train_recon': train_results['reconstruction'],
                    'scaler': scaler,
                    'mse': train_results['mse'],
                    'r2': train_results['r2']
                }

                # 如果有测试集，也进行评估
                if rcs_test is not None:
                    test_results = evaluate_model(trained_model, rcs_test_tensor, scaler, device, model_type)
                    
                    print(f"  测试集重构MSE: {test_results['mse']:.6f}")
                    print(f"  测试集重构R^2: {test_results['r2']:.6f}")

                    config_results.update({
                        'test_latent': test_results['latent'],
                        'test_recon': test_results['reconstruction'],
                        'test_mse': test_results['mse'],
                        'test_r2': test_results['r2']
                    })

                results[config_name] = config_results

                # 绘制训练历史
                plot_training_history(train_losses, val_losses, config_name, config_dir)

                # 可视化潜在空间
                visualize_latent_space(train_results['latent'], param_train, param_names,
                                       f"{config_name}_训练集", config_dir)

                if rcs_test is not None:
                    visualize_latent_space(test_results['latent'], param_test, param_names,
                                           f"{config_name}_测试集", config_dir)

                # 重构误差分析
                analyze_reconstruction_error(rcs_train, train_results['reconstruction'], 
                                             theta_values, phi_values,
                                             f"{config_name}_训练集", config_dir)

                print(f"  配置 {config_name} 训练完成")

            except Exception as e:
                print(f"  配置 {config_name} 训练失败: {e}")
                import traceback
                traceback.print_exc()

    # 生成对比分析
    if results:
        generate_comparison_analysis(results, ae_dir)
        print(f"\nAutoencoder分析完成，结果保存在: {ae_dir}")
    else:
        print("\nAutoencoder分析失败，没有成功训练的模型")

    # 清理GPU内存
    cleanup_gpu_memory()

    return results


# 在现有的import部分检查并记录PyTorch可用性
if __name__ == "__main__":
    print("Autoencoder分析模块测试")
    if PYTORCH_AVAILABLE:
        print("✓ PyTorch已安装，所有功能可用")

        # 测试模型创建
        test_model_std = StandardAutoencoder(8281, 10)
        test_model_vae = VariationalAutoencoder(8281, 10)

        print(f"✓ 标准自编码器参数数量: {sum(p.numel() for p in test_model_std.parameters()):,}")
        print(f"✓ 变分自编码器参数数量: {sum(p.numel() for p in test_model_vae.parameters()):,}")
    else:
        print("✗ PyTorch未安装，请使用以下命令安装:")
        print("pip install torch torchvision torchaudio")