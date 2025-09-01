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
    from autoencoder_prediction import (
        create_autoencoder_prediction_pipeline, save_prediction_summary
    )


def perform_autoencoder_analysis(rcs_data, theta_values, phi_values, param_data, param_names,
                                 freq_label, output_dir, train_indices, test_indices=None,
                                 latent_dims=[5, 10, 15, 20], model_types=['standard', 'vae'],
                                 device='auto', available_models=None, skip_training=False):
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
    print(f"将训练/加载以下配置:")
    print(f"  潜在维度: {latent_dims}")
    print(f"  模型类型: {model_types}")
    print(f"  跳过训练: {skip_training}")

    # 如果启用跳过训练，检查已有模型
    existing_models = {}
    if skip_training:
        print("\n🔍 检查已有模型...")
        existing_models = check_existing_models(ae_dir, latent_dims, model_types)
        if existing_models:
            print(f"找到 {len(existing_models)} 个完整的已有模型")
        else:
            print("未找到完整的已有模型，将进行训练")

    # 训练/加载不同配置的模型
    for model_type in model_types:
        for latent_dim in latent_dims:
            config_name = f"{model_type}_latent{latent_dim}"
            config_dir = os.path.join(ae_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            
            # 检查是否可以跳过训练
            if skip_training and config_name in existing_models:
                print(f"\n📥 加载已有模型: {config_name}")
                
                model_data = load_existing_model(config_dir, device)
                if model_data:
                    # 重新评估加载的模型以获得完整的结果
                    print(f"  重新评估已加载模型的性能...")
                    
                    try:
                        # 准备标准化器
                        eval_scaler = model_data['scaler']
                        if eval_scaler is None:
                            # 如果没有保存的标准化器，使用当前的标准化器
                            eval_scaler = scaler
                            print(f"    使用当前标准化器进行评估")
                        
                        # 重新评估训练集（使用当前的数据，不依赖保存的索引）
                        train_results = evaluate_model(model_data['model'], rcs_train_tensor, 
                                                     eval_scaler, device, model_data['model_type'])
                        
                        print(f"    训练集重构MSE: {train_results['mse']:.6f}")
                        print(f"    训练集重构R²: {train_results['r2']:.6f}")
                        
                        # 构造结果格式
                        config_results = {
                            'model_type': model_data['model_type'],
                            'latent_dim': model_data['latent_dim'],
                            'model': model_data['model'],
                            'train_losses': [],  # 已有模型无训练历史
                            'val_losses': [],
                            'train_latent': train_results['latent'],
                            'train_recon': train_results['reconstruction'],
                            'scaler': eval_scaler,
                            'mse': train_results['mse'],
                            'r2': train_results['r2']
                        }
                        
                        # 如果有测试集数据
                        if rcs_test is not None and rcs_test_tensor is not None:
                            test_results = evaluate_model(model_data['model'], rcs_test_tensor,
                                                        eval_scaler, device, model_data['model_type'])
                            
                            print(f"    测试集重构MSE: {test_results['mse']:.6f}")
                            print(f"    测试集重构R²: {test_results['r2']:.6f}")
                            
                            config_results.update({
                                'test_latent': test_results['latent'],
                                'test_recon': test_results['reconstruction'],
                                'test_mse': test_results['mse'],
                                'test_r2': test_results['r2']
                            })
                        
                        results[config_name] = config_results
                        
                        # 重新生成可视化（因为我们有了新的评估结果）
                        visualize_latent_space(train_results['latent'], param_train, param_names,
                                             f"{config_name}_训练集", config_dir)
                        
                        if rcs_test is not None and 'test_latent' in config_results:
                            visualize_latent_space(config_results['test_latent'], param_test, param_names,
                                                 f"{config_name}_测试集", config_dir)
                        
                        print(f"  ✅ 成功加载并重新评估模型 {config_name}")
                        continue
                        
                    except Exception as eval_error:
                        print(f"    ❌ 重新评估模型失败: {eval_error}")
                        print(f"    将重新训练模型")
                        
                else:
                    print(f"  加载已有模型失败，将重新训练")
            
            print(f"\n🚀 开始训练配置: {config_name}")

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

                # 保存完整的模型和数据
                print(f"  保存模型和隐空间数据...")
                
                # 保存完整的模型文件（包含结构信息）
                model_file = os.path.join(config_dir, 'autoencoder_model.pth')
                try:
                    model_to_save = trained_model.module if hasattr(trained_model, 'module') else trained_model
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'model_type': model_type,
                        'model_params': {
                            'input_dim': input_dim,
                            'latent_dim': latent_dim
                        },
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'final_train_mse': train_results['mse'],
                        'final_train_r2': train_results['r2']
                    }, model_file)
                    print(f"    ✅ 模型已保存: autoencoder_model.pth")
                except Exception as e:
                    print(f"    ❌ 模型保存失败: {e}")
                
                # 保存标准化器
                scaler_file = os.path.join(config_dir, 'scaler.pkl')
                try:
                    import pickle
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(scaler, f)
                    print(f"    ✅ 标准化器已保存: scaler.pkl")
                except Exception as e:
                    print(f"    ❌ 标准化器保存失败: {e}")
                
                # 保存训练集隐空间数据
                train_latent_file = os.path.join(config_dir, 'train_latent_space.npy')
                np.save(train_latent_file, train_results['latent'])
                print(f"    ✅ 训练集隐空间数据已保存: train_latent_space.npy")
                
                # 保存训练集索引
                train_indices_file = os.path.join(config_dir, 'train_indices.npy')
                np.save(train_indices_file, train_indices)
                print(f"    ✅ 训练集索引已保存: train_indices.npy")
                
                # 保存训练集参数数据
                train_params_file = os.path.join(config_dir, 'train_parameters.npy')
                np.save(train_params_file, param_train)
                print(f"    ✅ 训练集参数已保存: train_parameters.npy")
                
                if rcs_test is not None:
                    # 保存测试集隐空间数据
                    test_latent_file = os.path.join(config_dir, 'test_latent_space.npy')
                    np.save(test_latent_file, test_results['latent'])
                    print(f"    ✅ 测试集隐空间数据已保存: test_latent_space.npy")
                    
                    # 保存测试集索引
                    test_indices_file = os.path.join(config_dir, 'test_indices.npy')
                    np.save(test_indices_file, test_indices)
                    print(f"    ✅ 测试集索引已保存: test_indices.npy")
                    
                    # 保存测试集参数数据
                    test_params_file = os.path.join(config_dir, 'test_parameters.npy')
                    np.save(test_params_file, param_test)
                    print(f"    ✅ 测试集参数已保存: test_parameters.npy")

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
        
        # 创建隐空间数据索引文件
        create_latent_space_index(results, ae_dir, freq_label)
        
        # 如果有测试集，创建预测模型
        if test_indices is not None and len(test_indices) > 0:
            print("\n🔮 创建Autoencoder预测模型...")
            prediction_dir = os.path.join(ae_dir, 'predictions')
            
            try:
                prediction_results = create_autoencoder_prediction_pipeline(
                    results, param_data[train_indices], param_data[test_indices], prediction_dir
                )
                
                if prediction_results:
                    save_prediction_summary(prediction_results, prediction_dir)
                    print(f"✅ Autoencoder预测模型创建完成")
                else:
                    print("⚠️  未能创建任何预测模型")
                    
            except Exception as e:
                print(f"❌ 创建预测模型时发生错误: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nAutoencoder分析完成，结果保存在: {ae_dir}")
    else:
        print("\nAutoencoder分析失败，没有成功训练的模型")

    # 清理GPU内存
    cleanup_gpu_memory()

    return results


def check_existing_models(ae_dir, latent_dims, model_types):
    """
    检查已有的Autoencoder模型
    
    参数:
    ae_dir: autoencoder目录
    latent_dims: 潜在空间维度列表
    model_types: 模型类型列表
    
    返回:
    existing_models: 字典，包含已有模型的配置信息
    """
    existing_models = {}
    
    for model_type in model_types:
        for latent_dim in latent_dims:
            config_name = f"{model_type}_latent{latent_dim}"
            config_dir = os.path.join(ae_dir, config_name)
            model_file = os.path.join(config_dir, f'best_{model_type}_model.pth')
            
            if os.path.exists(model_file):
                # 检查核心模型文件是否存在
                core_files = [f'best_{model_type}_model.pth']
                optional_files = [
                    'scaler.pkl',
                    'train_latent_space.npy', 
                    'train_indices.npy',
                    'train_parameters.npy'
                ]
                
                core_files_exist = all(
                    os.path.exists(os.path.join(config_dir, fname)) 
                    for fname in core_files
                )
                
                optional_files_exist = all(
                    os.path.exists(os.path.join(config_dir, fname))
                    for fname in optional_files
                )
                
                if core_files_exist:
                    existing_models[config_name] = {
                        'config_dir': config_dir,
                        'model_type': model_type,
                        'latent_dim': latent_dim,
                        'complete': optional_files_exist,
                        'has_core': True
                    }
                    if optional_files_exist:
                        print(f"  找到完整的已有模型: {config_name}")
                    else:
                        print(f"  找到模型文件: {config_name} (缺少辅助文件，将重新评估)")
                else:
                    print(f"  模型 {config_name} 核心文件缺失，将重新训练")
    
    return existing_models


def load_existing_model(config_dir, device):
    """
    加载已有的Autoencoder模型
    
    参数:
    config_dir: 配置目录
    device: 计算设备
    
    返回:
    model_data: 包含模型和相关数据的字典
    """
    import pickle
    
    try:
        # 加载模型结构 - 先检查新的文件名，再检查旧的文件名
        model_file = os.path.join(config_dir, 'autoencoder_model.pth')
        if not os.path.exists(model_file):
            # 检查旧的命名规则
            for model_type in ['standard', 'vae']:
                old_model_file = os.path.join(config_dir, f'best_{model_type}_model.pth')
                if os.path.exists(old_model_file):
                    model_file = old_model_file
                    break
        model_state = torch.load(model_file, map_location=device)
        
        # 从状态字典中重建模型
        if 'model_type' in model_state and 'model_params' in model_state:
            # 新格式模型文件
            model_type = model_state['model_type']
            model_params = model_state['model_params']
        else:
            # 旧格式模型文件，从文件名推断
            if 'best_standard_model.pth' in model_file:
                model_type = 'standard'
            elif 'best_vae_model.pth' in model_file:
                model_type = 'vae'
            else:
                raise ValueError("无法从文件名确定模型类型")
            
            # 从配置目录名推断潜在维度
            config_name = os.path.basename(config_dir)
            if '_latent' in config_name:
                latent_dim = int(config_name.split('_latent')[1])
            else:
                raise ValueError("无法从目录名确定潜在维度")
            
            # 假设输入维度（RCS数据维度）
            input_dim = 8281  # 91x91角度组合
            
            model_params = {
                'input_dim': input_dim,
                'latent_dim': latent_dim
            }
            
            print(f"    从文件名推断: {model_type}, 潜在维度: {latent_dim}")
        
        # 创建模型
        if model_type == 'standard':
            from autoencoder_models import StandardAutoencoder
            model = StandardAutoencoder(model_params['input_dim'], model_params['latent_dim'])
        else:  # VAE
            from autoencoder_models import VariationalAutoencoder  
            model = VariationalAutoencoder(model_params['input_dim'], model_params['latent_dim'])
        
        model.load_state_dict(model_state['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 尝试加载标准化器（如果存在）
        scaler_file = os.path.join(config_dir, 'scaler.pkl')
        scaler = None
        if os.path.exists(scaler_file):
            try:
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"    ✅ 加载已有标准化器")
            except Exception as e:
                print(f"    ⚠️ 标准化器加载失败: {e}")
        else:
            print(f"    ⚠️ 标准化器文件不存在，将使用默认标准化器")
        
        # 尝试加载隐空间数据（如果存在）
        train_latent = None
        train_indices = None  
        train_params = None
        test_latent = None
        test_indices = None
        test_params = None
        
        train_latent_file = os.path.join(config_dir, 'train_latent_space.npy')
        if os.path.exists(train_latent_file):
            try:
                train_latent = np.load(train_latent_file)
                train_indices = np.load(os.path.join(config_dir, 'train_indices.npy'))
                train_params = np.load(os.path.join(config_dir, 'train_parameters.npy'))
                print(f"    ✅ 加载已有隐空间数据")
            except Exception as e:
                print(f"    ⚠️ 隐空间数据加载失败: {e}")
        
        # 检查是否有测试集数据
        test_latent_file = os.path.join(config_dir, 'test_latent_space.npy')
        if os.path.exists(test_latent_file):
            try:
                test_latent = np.load(test_latent_file)
                test_indices = np.load(os.path.join(config_dir, 'test_indices.npy'))
                test_params = np.load(os.path.join(config_dir, 'test_parameters.npy'))
                print(f"    ✅ 加载已有测试集数据")
            except Exception as e:
                print(f"    ⚠️ 测试集数据加载失败: {e}")
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'train_latent': train_latent,
            'train_indices': train_indices,
            'train_params': train_params,
            'test_latent': test_latent,
            'test_indices': test_indices,
            'test_params': test_params,
            'model_type': model_type,
            'latent_dim': model_params['latent_dim']
        }
        
        return model_data
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def create_latent_space_index(results, ae_dir, freq_label):
    """
    创建隐空间数据索引文件
    
    参数:
    results: 分析结果字典
    ae_dir: autoencoder目录
    freq_label: 频率标签
    """
    import json
    from datetime import datetime
    
    index_data = {
        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'frequency': freq_label,
        'models': []
    }
    
    for config_name, config_results in results.items():
        model_info = {
            'config_name': config_name,
            'model_type': config_results['model_type'],
            'latent_dim': config_results['latent_dim'],
            'train_mse': float(config_results['mse']),
            'train_r2': float(config_results['r2']),
            'files': {
                'train_latent_space': f"{config_name}/train_latent_space.npy",
                'train_indices': f"{config_name}/train_indices.npy", 
                'train_parameters': f"{config_name}/train_parameters.npy"
            }
        }
        
        # 如果有测试集数据
        if 'test_mse' in config_results:
            model_info['test_mse'] = float(config_results['test_mse'])
            model_info['test_r2'] = float(config_results['test_r2'])
            model_info['files'].update({
                'test_latent_space': f"{config_name}/test_latent_space.npy",
                'test_indices': f"{config_name}/test_indices.npy",
                'test_parameters': f"{config_name}/test_parameters.npy"
            })
        
        index_data['models'].append(model_info)
    
    # 保存索引文件
    index_file = os.path.join(ae_dir, 'latent_space_index.json')
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"📋 隐空间数据索引已保存: {index_file}")
    print(f"   包含 {len(results)} 个模型配置的隐空间数据")


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