"""
Autoencoder可视化模块
包含所有绘图和可视化相关的功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False


def plot_training_history(train_losses, val_losses, model_name, output_dir):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')

    plt.xlabel('训练轮数')
    plt.ylabel('损失')
    plt.title(f'{model_name} 训练历史')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=200, bbox_inches='tight')
    plt.close()


def visualize_latent_space(latent_repr, param_data, param_names, title, output_dir):
    """可视化潜在空间"""
    n_components = min(latent_repr.shape[1], 6)  # 最多显示6个潜在维度

    if n_components >= 2:
        # 2D潜在空间可视化
        plt.figure(figsize=(15, 10))

        # 前两个潜在维度的散点图
        plt.subplot(2, 3, 1)
        plt.scatter(latent_repr[:, 0], latent_repr[:, 1], alpha=0.6)
        plt.xlabel('潜在维度 1')
        plt.ylabel('潜在维度 2')
        plt.title('潜在空间分布 (维度1 vs 维度2)')
        plt.grid(True, alpha=0.3)

        # 潜在维度分布直方图
        for i in range(min(5, n_components)):
            plt.subplot(2, 3, i + 2)
            plt.hist(latent_repr[:, i], bins=30, alpha=0.7)
            plt.xlabel(f'潜在维度 {i + 1}')
            plt.ylabel('频数')
            plt.title(f'潜在维度 {i + 1} 分布')
            plt.grid(True, alpha=0.3)

        plt.suptitle(f'{title} - 潜在空间分析')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_space.png'), dpi=200, bbox_inches='tight')
        plt.close()

    # 参数与潜在变量的相关性分析（如果参数数据可用）
    if param_data is not None and len(param_names) > 0:
        n_latent = latent_repr.shape[1]
        n_params = len(param_names)

        correlation_matrix = np.zeros((n_latent, n_params))
        p_value_matrix = np.zeros((n_latent, n_params))

        for i in range(n_latent):
            for j in range(n_params):
                if np.std(latent_repr[:, i]) > 1e-6 and np.std(param_data[:, j]) > 1e-6:
                    corr, p_val = pearsonr(latent_repr[:, i], param_data[:, j])
                    correlation_matrix[i, j] = corr
                    p_value_matrix[i, j] = p_val

        # 绘制相关性热图
        plt.figure(figsize=(12, 8))
        im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Pearson相关系数')

        plt.xlabel('设计参数')
        plt.ylabel('潜在维度')
        plt.title(f'{title} - 参数与潜在变量相关性')

        # 设置坐标轴标签
        plt.xticks(range(n_params), param_names, rotation=45, ha='right')
        plt.yticks(range(n_latent), [f'潜在维度 {i + 1}' for i in range(n_latent)])

        # 添加相关系数数值
        for i in range(n_latent):
            for j in range(n_params):
                plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_correlation.png'), dpi=200, bbox_inches='tight')
        plt.close()


def analyze_reconstruction_error(original, reconstructed, theta_values, phi_values, title, output_dir):
    """分析重构误差"""
    # 计算逐点误差
    pointwise_error = np.mean((original - reconstructed) ** 2, axis=0)

    # 重构为2D格式进行可视化
    n_theta = len(theta_values)
    n_phi = len(phi_values)

    if len(pointwise_error) == n_theta * n_phi:
        error_2d = pointwise_error.reshape(n_theta, n_phi).T

        plt.figure(figsize=(15, 5))

        # 原始数据示例（第一个样本）
        plt.subplot(131)
        original_2d = original[0].reshape(n_theta, n_phi).T
        plt.imshow(original_2d, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                   min(phi_values), max(phi_values)])
        plt.colorbar(label='RCS (dB)')
        plt.xlabel('俯仰角 θ (度)')
        plt.ylabel('偏航角 φ (度)')
        plt.title('原始RCS (示例)')

        # 重构数据示例（第一个样本）
        plt.subplot(132)
        recon_2d = reconstructed[0].reshape(n_theta, n_phi).T
        plt.imshow(recon_2d, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                min(phi_values), max(phi_values)])
        plt.colorbar(label='RCS (dB)')
        plt.xlabel('俯仰角 θ (度)')
        plt.ylabel('偏航角 φ (度)')
        plt.title('重构RCS (示例)')

        # 误差分布
        plt.subplot(133)
        plt.imshow(error_2d, cmap='hot', extent=[min(theta_values), max(theta_values),
                                                min(phi_values), max(phi_values)])
        plt.colorbar(label='均方误差')
        plt.xlabel('俯仰角 θ (度)')
        plt.ylabel('偏航角 φ (度)')
        plt.title('重构误差分布')

        plt.suptitle(f'{title} - 重构误差分析')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstruction_error.png'), dpi=200, bbox_inches='tight')
        plt.close()

    # 误差统计
    mse_per_sample = np.mean((original - reconstructed) ** 2, axis=1)

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.hist(mse_per_sample, bins=30, alpha=0.7)
    plt.xlabel('样本重构MSE')
    plt.ylabel('频数')
    plt.title('样本重构误差分布')
    plt.grid(True, alpha=0.3)

    plt.subplot(122)
    plt.plot(mse_per_sample, 'o-', alpha=0.7)
    plt.xlabel('样本索引')
    plt.ylabel('重构MSE')
    plt.title('各样本重构误差')
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'{title} - 重构误差统计')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_statistics.png'), dpi=200, bbox_inches='tight')
    plt.close()


def generate_comparison_analysis(results, output_dir):
    """生成不同配置的对比分析"""
    
    # 确保numpy可用（防御性编程）
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # 提取对比数据
    configs = []
    train_mses = []
    train_r2s = []
    test_mses = []
    test_r2s = []
    latent_dims = []

    for config_name, config_results in results.items():
        configs.append(config_name)
        train_mses.append(config_results['mse'])
        train_r2s.append(config_results['r2'])
        latent_dims.append(config_results['latent_dim'])

        if 'test_mse' in config_results:
            test_mses.append(config_results['test_mse'])
            test_r2s.append(config_results['test_r2'])
        else:
            test_mses.append(np.nan)
            test_r2s.append(np.nan)

    # 创建对比图表
    plt.figure(figsize=(20, 10))

    # MSE对比
    plt.subplot(2, 3, 1)
    x_pos = np.arange(len(configs))
    plt.bar(x_pos, train_mses, alpha=0.7, label='训练集')
    if not all(np.isnan(test_mses)):
        plt.bar(x_pos, test_mses, alpha=0.7, label='测试集')
    plt.xlabel('模型配置')
    plt.ylabel('均方误差 (MSE)')
    plt.title('重构MSE对比')
    plt.xticks(x_pos, configs, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # R^2对比
    plt.subplot(2, 3, 2)
    plt.bar(x_pos, train_r2s, alpha=0.7, label='训练集')
    if not all(np.isnan(test_r2s)):
        plt.bar(x_pos, test_r2s, alpha=0.7, label='测试集')
    plt.xlabel('模型配置')
    plt.ylabel('R^2 分数')
    plt.title('重构R^2对比')
    plt.xticks(x_pos, configs, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 潜在维度vs性能
    plt.subplot(2, 3, 3)
    standard_mask = [('standard' in config) for config in configs]
    vae_mask = [('vae' in config) for config in configs]

    if any(standard_mask):
        standard_dims = [latent_dims[i] for i, m in enumerate(standard_mask) if m]
        standard_r2s = [train_r2s[i] for i, m in enumerate(standard_mask) if m]
        plt.plot(standard_dims, standard_r2s, 'o-', label='Standard AE', linewidth=2, markersize=8)

    if any(vae_mask):
        vae_dims = [latent_dims[i] for i, m in enumerate(vae_mask) if m]
        vae_r2s = [train_r2s[i] for i, m in enumerate(vae_mask) if m]
        plt.plot(vae_dims, vae_r2s, 's-', label='VAE', linewidth=2, markersize=8)

    plt.xlabel('潜在空间维度')
    plt.ylabel('R^2 分数')
    plt.title('潜在维度 vs 重构性能')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 训练损失对比（取最后的损失值）
    plt.subplot(2, 3, 4)
    final_train_losses = []
    final_val_losses = []

    for config_results in results.values():
        # 安全地访问损失数据，避免空列表或缺失数据的问题
        if 'train_losses' in config_results and len(config_results['train_losses']) > 0:
            final_train_losses.append(config_results['train_losses'][-1])
        else:
            final_train_losses.append(float('nan'))  # 使用NaN表示缺失数据
            
        if 'val_losses' in config_results and len(config_results['val_losses']) > 0:
            final_val_losses.append(config_results['val_losses'][-1])
        else:
            final_val_losses.append(float('nan'))

    # 处理NaN值，只显示有效数据
    valid_train_mask = ~np.isnan(final_train_losses)
    valid_val_mask = ~np.isnan(final_val_losses)
    
    if np.any(valid_train_mask):
        plt.bar(np.array(x_pos)[valid_train_mask], np.array(final_train_losses)[valid_train_mask], 
                alpha=0.7, label='最终训练损失', color='blue')
    
    if np.any(valid_val_mask):
        plt.bar(np.array(x_pos)[valid_val_mask], np.array(final_val_losses)[valid_val_mask], 
                alpha=0.7, label='最终验证损失', color='orange')
    
    plt.xlabel('模型配置')
    plt.ylabel('损失值')
    plt.title('最终训练损失对比')
    plt.xticks(x_pos, configs, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 如果所有数据都缺失，添加提示信息
    if not np.any(valid_train_mask) and not np.any(valid_val_mask):
        plt.text(0.5, 0.5, '训练损失数据缺失\n请检查模型训练状态', 
                transform=plt.gca().transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 模型复杂度vs性能散点图
    plt.subplot(2, 3, 5)
    model_params = []
    for config_name, config_results in results.items():
        model = config_results['model']
        n_params = sum(p.numel() for p in model.parameters())
        model_params.append(n_params)

    plt.scatter(model_params, train_r2s, c=latent_dims, s=100, alpha=0.7, cmap='viridis')
    plt.colorbar(label='潜在空间维度')
    plt.xlabel('模型参数数量')
    plt.ylabel('R^2 分数')
    plt.title('模型复杂度 vs 性能')
    plt.grid(True, alpha=0.3)

    # 性能总结表格
    plt.subplot(2, 3, 6)
    plt.axis('off')

    # 创建总结表格数据
    table_data = []
    for i, config in enumerate(configs):
        row = [
            config,
            f"{train_mses[i]:.4f}",
            f"{train_r2s[i]:.4f}",
            f"{test_mses[i]:.4f}" if not np.isnan(test_mses[i]) else "N/A",
            f"{test_r2s[i]:.4f}" if not np.isnan(test_r2s[i]) else "N/A"
        ]
        table_data.append(row)

    headers = ['配置', '训练MSE', '训练R^2', '测试MSE', '测试R^2']
    table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    plt.title('性能总结表', pad=20)

    plt.suptitle('Autoencoder模型对比分析', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 保存数值结果到CSV
    comparison_df = pd.DataFrame({
        '配置': configs,
        '潜在维度': latent_dims,
        '训练MSE': train_mses,
        '训练R^2': train_r2s,
        '测试MSE': test_mses,
        '测试R^2': test_r2s,
        '模型参数数量': model_params
    })
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    print("对比分析完成，结果已保存")


def compare_with_pod_results(ae_results, pod_results, output_dir):
    """
    对比Autoencoder和POD的结果

    参数:
    ae_results: Autoencoder分析结果
    pod_results: POD分析结果（包含重构数据和误差）
    output_dir: 输出目录
    """
    
    # 确保numpy可用（防御性编程）
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    if not ae_results:
        print("没有可用的Autoencoder结果进行对比")
        return

    print("开始对比Autoencoder和POD结果...")

    comparison_dir = os.path.join(output_dir, 'ae_pod_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # 选择最佳的AE配置进行对比（基于R^2分数）
    best_config = None
    best_r2 = -np.inf

    for config_name, config_results in ae_results.items():
        if config_results['r2'] > best_r2:
            best_r2 = config_results['r2']
            best_config = config_name

    if best_config is None:
        print("没有找到有效的Autoencoder配置")
        return

    best_ae_results = ae_results[best_config]

    # 对比分析
    plt.figure(figsize=(15, 10))

    # R^2分数对比
    plt.subplot(2, 3, 1)
    methods = ['POD', f'Autoencoder\n({best_config})']
    r2_scores = [pod_results.get('r2', 0), best_ae_results['r2']]

    plt.bar(methods, r2_scores, alpha=0.7, color=['blue', 'red'])
    plt.ylabel('R^2 分数')
    plt.title('重构质量对比 (R^2)')
    plt.grid(True, alpha=0.3)

    # MSE对比
    plt.subplot(2, 3, 2)
    mse_scores = [pod_results.get('mse', 0), best_ae_results['mse']]

    plt.bar(methods, mse_scores, alpha=0.7, color=['blue', 'red'])
    plt.ylabel('均方误差 (MSE)')
    plt.title('重构误差对比 (MSE)')
    plt.grid(True, alpha=0.3)

    # 潜在空间维度对比
    plt.subplot(2, 3, 3)
    pod_dims = pod_results.get('n_modes', 10)  # 假设POD使用10个模态
    ae_dims = best_ae_results['latent_dim']

    plt.bar(['POD', 'Autoencoder'], [pod_dims, ae_dims], alpha=0.7, color=['blue', 'red'])
    plt.ylabel('降维后维度')
    plt.title('降维程度对比')
    plt.grid(True, alpha=0.3)

    # 如果有潜在空间数据，进行可视化对比
    if 'pod_coeffs' in pod_results and best_ae_results['train_latent'] is not None:
        pod_latent = pod_results['pod_coeffs'][:, :2]  # 取前两个POD系数
        ae_latent = best_ae_results['train_latent'][:, :2]  # 取前两个潜在变量

        plt.subplot(2, 3, 4)
        plt.scatter(pod_latent[:, 0], pod_latent[:, 1], alpha=0.6, label='POD')
        plt.xlabel('第1维')
        plt.ylabel('第2维')
        plt.title('POD潜在空间 (前2维)')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 5)
        plt.scatter(ae_latent[:, 0], ae_latent[:, 1], alpha=0.6, label='Autoencoder', color='red')
        plt.xlabel('第1维')
        plt.ylabel('第2维')
        plt.title('Autoencoder潜在空间 (前2维)')
        plt.grid(True, alpha=0.3)

    # 方法特点总结
    plt.subplot(2, 3, 6)
    plt.axis('off')

    summary_text = f"""
           方法对比总结:

           POD (本征正交分解):
           • 线性降维方法
           • 基于主成分分析
           • R^2 = {pod_results.get('r2', 0):.4f}
           • MSE = {pod_results.get('mse', 0):.4f}
           • 维度: {pod_dims}

           Autoencoder ({best_config}):
           • 非线性降维方法
           • 基于神经网络
           • R^2 = {best_ae_results['r2']:.4f}
           • MSE = {best_ae_results['mse']:.4f}
           • 维度: {ae_dims}

           {'✓ Autoencoder表现更佳' if best_ae_results['r2'] > pod_results.get('r2', 0) else '✓ POD表现更佳'}
           """

    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.suptitle('POD vs Autoencoder 降维方法对比', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'pod_ae_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"POD vs Autoencoder对比分析完成，结果保存在: {comparison_dir}")