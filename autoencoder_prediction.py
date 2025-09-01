"""
基于Autoencoder的RCS数据预测模块
用于从设计参数预测RCS数据的功能实现
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import pickle
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_autoencoder_prediction_models(latent_data, param_data, model_type='rf'):
    """
    创建从设计参数到隐空间的预测模型
    
    参数:
    latent_data: 隐空间数据 [n_samples, latent_dim]
    param_data: 设计参数数据 [n_samples, n_params]
    model_type: 模型类型 ('rf', 'lr', 'lasso')
    
    返回:
    models: 预测模型列表 (每个隐空间维度一个模型)
    scaler: 参数标准化器
    """
    latent_dim = latent_data.shape[1]
    models = []
    
    # 标准化参数数据
    scaler = StandardScaler()
    param_data_scaled = scaler.fit_transform(param_data)
    
    # 为每个隐空间维度训练预测模型
    for i in range(latent_dim):
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'lr':
            model = LinearRegression()
        elif model_type == 'lasso':
            model = Lasso(alpha=0.1, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model.fit(param_data_scaled, latent_data[:, i])
        models.append(model)
    
    return models, scaler


def predict_latent_from_parameters(param_data_test, models, scaler):
    """
    从设计参数预测隐空间表示
    
    参数:
    param_data_test: 测试参数数据
    models: 预测模型列表
    scaler: 参数标准化器
    
    返回:
    predicted_latent: 预测的隐空间表示
    """
    param_data_scaled = scaler.transform(param_data_test)
    latent_dim = len(models)
    predicted_latent = np.zeros((param_data_test.shape[0], latent_dim))
    
    for i, model in enumerate(models):
        predicted_latent[:, i] = model.predict(param_data_scaled)
    
    return predicted_latent


def predict_rcs_from_autoencoder(autoencoder_model, predicted_latent, data_scaler, device):
    """
    使用训练好的Autoencoder从隐空间重构RCS数据
    
    参数:
    autoencoder_model: 训练好的Autoencoder模型
    predicted_latent: 预测的隐空间表示
    data_scaler: RCS数据标准化器
    device: 计算设备
    
    返回:
    reconstructed_rcs: 重构的RCS数据
    """
    autoencoder_model.eval()
    
    with torch.no_grad():
        latent_tensor = torch.from_numpy(predicted_latent.astype(np.float32)).to(device)
        
        # 使用解码器重构数据
        if hasattr(autoencoder_model, 'decode'):
            # VAE模型
            reconstructed_tensor = autoencoder_model.decode(latent_tensor)
        else:
            # 标准Autoencoder - 使用解码器部分
            reconstructed_tensor = autoencoder_model.decoder(latent_tensor)
        
        reconstructed_data = reconstructed_tensor.cpu().numpy()
    
    # 逆标准化
    reconstructed_rcs = data_scaler.inverse_transform(reconstructed_data)
    
    return reconstructed_rcs


def create_autoencoder_prediction_pipeline(config_results, param_data_train, param_data_test, output_dir):
    """
    创建完整的Autoencoder预测流水线
    
    参数:
    config_results: Autoencoder配置结果
    param_data_train: 训练集参数数据
    param_data_test: 测试集参数数据
    output_dir: 输出目录
    
    返回:
    prediction_results: 预测结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction_results = {}
    
    for config_name, results in config_results.items():
        print(f"\n为配置 {config_name} 创建预测模型...")
        
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        # 获取训练数据
        train_latent = results['train_latent']
        autoencoder_model = results['model']
        data_scaler = results['scaler']
        
        # 创建参数到隐空间的预测模型
        try:
            models, param_scaler = create_autoencoder_prediction_models(
                train_latent, param_data_train, model_type='rf'
            )
            
            # 预测测试集隐空间
            predicted_latent = predict_latent_from_parameters(
                param_data_test, models, param_scaler
            )
            
            # 从隐空间重构RCS数据
            device = next(autoencoder_model.parameters()).device
            reconstructed_rcs = predict_rcs_from_autoencoder(
                autoencoder_model, predicted_latent, data_scaler, device
            )
            
            # 如果有真实的测试集数据，计算预测性能
            performance_metrics = {}
            if 'test_latent' in results and 'test_recon' in results:
                true_latent = results['test_latent']
                true_rcs = results['test_recon']
                
                # 隐空间预测性能
                latent_mse = mean_squared_error(true_latent, predicted_latent)
                latent_r2 = r2_score(true_latent, predicted_latent)
                
                # RCS重构预测性能
                rcs_mse = mean_squared_error(true_rcs, reconstructed_rcs)
                rcs_r2 = r2_score(true_rcs, reconstructed_rcs)
                
                performance_metrics = {
                    'latent_mse': latent_mse,
                    'latent_r2': latent_r2,
                    'rcs_mse': rcs_mse,
                    'rcs_r2': rcs_r2
                }
                
                print(f"  隐空间预测 - MSE: {latent_mse:.6f}, R²: {latent_r2:.6f}")
                print(f"  RCS重构预测 - MSE: {rcs_mse:.6f}, R²: {rcs_r2:.6f}")
            
            # 保存预测模型和结果
            model_file = os.path.join(config_dir, 'prediction_models.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'models': models,
                    'param_scaler': param_scaler,
                    'model_type': 'rf'
                }, f)
            
            # 保存预测结果
            np.save(os.path.join(config_dir, 'predicted_latent.npy'), predicted_latent)
            np.save(os.path.join(config_dir, 'reconstructed_rcs.npy'), reconstructed_rcs)
            
            # 可视化特征重要性
            visualize_feature_importance(models, config_name, config_dir)
            
            prediction_results[config_name] = {
                'predicted_latent': predicted_latent,
                'reconstructed_rcs': reconstructed_rcs,
                'performance': performance_metrics,
                'models': models,
                'param_scaler': param_scaler
            }
            
            print(f"  配置 {config_name} 预测模型创建完成")
            
        except Exception as e:
            print(f"  配置 {config_name} 预测模型创建失败: {e}")
            import traceback
            traceback.print_exc()
    
    return prediction_results


def visualize_feature_importance(models, config_name, output_dir):
    """
    可视化特征重要性
    
    参数:
    models: 预测模型列表
    config_name: 配置名称
    output_dir: 输出目录
    """
    latent_dim = len(models)
    
    # 只有RandomForest模型有feature_importances_属性
    if hasattr(models[0], 'feature_importances_'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{config_name} - 参数重要性分析', fontsize=16)
        
        # 显示前4个隐空间维度的重要性
        for i in range(min(4, latent_dim)):
            row, col = i // 2, i % 2
            ax = axes[row, col] if latent_dim > 1 else axes
            
            importance = models[i].feature_importances_
            param_indices = range(len(importance))
            
            bars = ax.bar(param_indices, importance)
            ax.set_title(f'隐空间维度 {i+1} 的参数重要性')
            ax.set_xlabel('参数索引')
            ax.set_ylabel('重要性')
            ax.grid(True, alpha=0.3)
            
            # 标注最重要的参数
            max_idx = np.argmax(importance)
            ax.annotate(f'最重要: 参数{max_idx+1}', 
                       xy=(max_idx, importance[max_idx]),
                       xytext=(max_idx, importance[max_idx] + 0.02),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, ha='center')
        
        # 隐藏多余的子图
        for i in range(latent_dim, 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()


def save_prediction_summary(prediction_results, output_dir):
    """
    保存预测结果摘要
    
    参数:
    prediction_results: 预测结果字典
    output_dir: 输出目录
    """
    from datetime import datetime
    
    summary = {
        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction_method': 'Random Forest -> Autoencoder Decoder',
        'configurations': []
    }
    
    for config_name, results in prediction_results.items():
        config_summary = {
            'config_name': config_name,
            'latent_dim': results['predicted_latent'].shape[1],
            'test_samples': results['predicted_latent'].shape[0]
        }
        
        if 'performance' in results and results['performance']:
            config_summary.update(results['performance'])
        
        summary['configurations'].append(config_summary)
    
    # 保存JSON摘要
    summary_file = os.path.join(output_dir, 'prediction_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 预测结果摘要已保存: {summary_file}")


def load_prediction_models(config_dir):
    """
    加载已保存的预测模型
    
    参数:
    config_dir: 配置目录
    
    返回:
    models: 预测模型列表
    param_scaler: 参数标准化器
    """
    model_file = os.path.join(config_dir, 'prediction_models.pkl')
    
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        return data['models'], data['param_scaler']
    else:
        raise FileNotFoundError(f"预测模型文件不存在: {model_file}")


if __name__ == "__main__":
    print("Autoencoder预测模块测试")
    print("该模块提供以下功能:")
    print("- 从设计参数预测隐空间表示")
    print("- 使用Autoencoder解码器重构RCS数据")  
    print("- 创建完整的预测流水线")
    print("- 可视化参数重要性分析")