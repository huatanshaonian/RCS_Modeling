"""
模态分析模块：实现模态可视化、参数敏感性分析和角度敏感性分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# 在文件开头添加字体设置代码
import matplotlib as mpl

from pod_analysis import compute_pod_coeffs

# 使用支持上标更全面的字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']  # 尝试多个字体备选
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
mpl.rcParams['mathtext.default'] = 'regular'  # 使用普通字体渲染数学符号
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def visualize_modes(phi_modes, theta_values, phi_values, output_dir):
    """
    可视化POD模态 - 增强版，处理异常情况

    参数:
    phi_modes: POD模态矩阵，形状为 [num_angles, num_modes_to_visualize]
    theta_values: theta角度值
    phi_values: phi角度值
    output_dir: 输出目录
    """
    try:
        # 检查输入数据
        if phi_modes.size == 0:
            print("警告: 输入的模态矩阵为空")
            return

        if np.any(np.isnan(phi_modes)) or np.any(np.isinf(phi_modes)):
            print("警告: 模态矩阵包含NaN或Inf值，将被替换为0")
            phi_modes = np.nan_to_num(phi_modes, nan=0.0, posinf=0.0, neginf=0.0)

        n_theta = len(theta_values)
        n_phi = len(phi_values)
        num_modes = phi_modes.shape[1]

        # 创建模态可视化目录
        modes_dir = os.path.join(output_dir, 'modes')
        os.makedirs(modes_dir, exist_ok=True)

        # 限制处理的模态数量
        modes_to_visualize = min(10, num_modes)
        print(f"将可视化前 {modes_to_visualize} 个模态")

        # 创建角度网格
        theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)

        for i in range(modes_to_visualize):
            try:
                # 将一维模态向量重塑为二维角度矩阵
                mode_2d = phi_modes[:, i].reshape(n_theta, n_phi).T

                # 检查重塑后的数据
                if np.any(np.isnan(mode_2d)) or np.any(np.isinf(mode_2d)):
                    print(f"警告: 模态 {i + 1} 重塑后包含NaN或Inf，将被替换为0")
                    mode_2d = np.nan_to_num(mode_2d, nan=0.0, posinf=0.0, neginf=0.0)

                plt.figure(figsize=(15, 5))

                # 2D热图
                plt.subplot(131)
                im = plt.imshow(mode_2d, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                             min(phi_values), max(phi_values)])
                plt.colorbar(im, label='模态幅值')
                plt.xlabel('俯仰角 $\\theta$ (度)')
                plt.ylabel('偏航角 $\\phi$ (度)')
                plt.title(f'模态 {i + 1} - 2D热图')

                # 3D表面图
                ax = plt.subplot(132, projection='3d')
                surf = ax.plot_surface(theta_grid, phi_grid, mode_2d, cmap='jet',
                                       linewidth=0, antialiased=True)
                plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='模态幅值')
                ax.set_xlabel('俯仰角 $\\theta$ (度)')
                ax.set_ylabel('偏航角 $\\phi$ (度)')
                ax.set_zlabel('模态幅值')
                ax.set_title(f'模态 {i + 1} - 3D表面')

                # 等值线图
                plt.subplot(133)
                cs = plt.contourf(theta_values, phi_values, mode_2d, 20, cmap='jet')
                plt.colorbar(cs, label='模态幅值')
                plt.contour(theta_values, phi_values, mode_2d, 10, colors='k', linewidths=0.5)
                plt.xlabel('俯仰角 $\\theta$ (度)')
                plt.ylabel('偏航角 $\\phi$ (度)')
                plt.title(f'模态 {i + 1} - 等值线图')

                plt.tight_layout()
                plt.savefig(os.path.join(modes_dir, f'mode_{i + 1}.png'), dpi=200)  # 降低dpi节省内存
                plt.close('all')  # 确保关闭所有图形，释放内存

            except Exception as e:
                print(f"处理模态 {i + 1} 时发生错误: {e}")
                plt.close('all')

        # 生成模态对比图
        try:
            # 只尝试生成对比图当至少有2个模态可用
            if modes_to_visualize >= 2:
                rows = min(2, (modes_to_visualize + 4) // 5)
                cols = min(5, modes_to_visualize)
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
                if rows == 1 and cols == 1:
                    axes = np.array([axes])  # 确保axes是数组
                axes = axes.flatten()

                # 计算缩放范围
                valid_modes = []
                for i in range(modes_to_visualize):
                    mode_2d = phi_modes[:, i].reshape(n_theta, n_phi).T
                    if not (np.any(np.isnan(mode_2d)) or np.any(np.isinf(mode_2d)) or np.all(mode_2d == 0)):
                        valid_modes.append(mode_2d)

                if valid_modes:
                    min_val = min(np.min(m) for m in valid_modes)
                    max_val = max(np.max(m) for m in valid_modes)
                else:
                    min_val, max_val = -1, 1  # 默认范围

                for i in range(modes_to_visualize):
                    mode_2d = phi_modes[:, i].reshape(n_theta, n_phi).T
                    mode_2d = np.nan_to_num(mode_2d, nan=0.0, posinf=0.0, neginf=0.0)

                    im = axes[i].imshow(mode_2d, cmap='jet',
                                        extent=[min(theta_values), max(theta_values),
                                                min(phi_values), max(phi_values)],
                                        vmin=min_val, vmax=max_val)
                    axes[i].set_title(f'模态 {i + 1}')
                    axes[i].set_xlabel('俯仰角 $\\theta$ (度)')
                    axes[i].set_ylabel('偏航角 $\\phi$ (度)')

                # 隐藏额外的子图
                for j in range(modes_to_visualize, len(axes)):
                    axes[j].axis('off')

                plt.tight_layout()
                fig.subplots_adjust(right=0.9)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax, label='模态幅值')

                plt.savefig(os.path.join(output_dir, 'modes_comparison.png'), dpi=200)
                plt.close('all')
        except Exception as e:
            print(f"生成模态对比图时发生错误: {e}")
            plt.close('all')

    except Exception as e:
        print(f"模态可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


def angle_sensitivity(phi_modes, theta_values, phi_values, output_dir):
    """
    分析模态对角度变化的敏感度

    参数:
    phi_modes: POD模态矩阵，形状为 [num_angles, num_modes]
    theta_values: theta角度值
    phi_values: phi角度值
    output_dir: 输出目录
    """
    n_theta = len(theta_values)
    n_phi = len(phi_values)
    num_modes = phi_modes.shape[1]

    angles_dir = os.path.join(output_dir, 'angle_sensitivity')
    os.makedirs(angles_dir, exist_ok=True)

    # 创建角度网格
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)

    for i in range(min(5, num_modes)):
        # 将一维模态向量重塑为二维角度矩阵
        mode_2d = phi_modes[:, i].reshape(n_theta, n_phi).T

        # 计算角度梯度
        theta_grad = np.gradient(mode_2d, axis=1)
        phi_grad = np.gradient(mode_2d, axis=0)
        grad_magnitude = np.sqrt(theta_grad ** 2 + phi_grad ** 2)

        plt.figure(figsize=(20, 5))

        # 原始模态
        plt.subplot(141)
        plt.imshow(mode_2d, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                min(phi_values), max(phi_values)])
        plt.colorbar(label='模态幅值')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title(f'模态 {i + 1}')

        # 梯度幅值
        plt.subplot(142)
        plt.imshow(grad_magnitude, cmap='hot', extent=[min(theta_values), max(theta_values),
                                                       min(phi_values), max(phi_values)])
        plt.colorbar(label='梯度幅值')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title(f'模态 {i + 1} 角度梯度幅值')

        # 俯仰角梯度
        plt.subplot(143)
        plt.imshow(theta_grad, cmap='coolwarm', extent=[min(theta_values), max(theta_values),
                                                        min(phi_values), max(phi_values)])
        plt.colorbar(label='俯仰角梯度')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title(f'模态 {i + 1} 俯仰角梯度')

        # 偏航角梯度
        plt.subplot(144)
        plt.imshow(phi_grad, cmap='coolwarm', extent=[min(theta_values), max(theta_values),
                                                      min(phi_values), max(phi_values)])
        plt.colorbar(label='偏航角梯度')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title(f'模态 {i + 1} 偏航角梯度')

        plt.tight_layout()
        plt.savefig(os.path.join(angles_dir, f'mode_{i + 1}_gradient.png'), dpi=300)
        plt.close()

        # 梯度方向图
        plt.figure(figsize=(10, 8))
        # 降采样以便清晰显示
        step = max(1, n_theta // 15)

        plt.quiver(theta_grid[::step, ::step], phi_grid[::step, ::step],
                   theta_grad[::step, ::step], phi_grad[::step, ::step],
                   grad_magnitude[::step, ::step], cmap='jet')
        plt.colorbar(label='梯度幅值')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title(f'模态 {i + 1} 梯度方向')
        plt.contour(theta_values, phi_values, mode_2d, 10, colors='k', linewidths=0.5, alpha=0.5)

        plt.savefig(os.path.join(angles_dir, f'mode_{i + 1}_gradient_direction.png'), dpi=300)
        plt.close()

        # 保存敏感区域分析
        sensitive_regions = grad_magnitude > np.percentile(grad_magnitude, 90)  # 梯度最高的10%区域
        plt.figure(figsize=(12, 10))

        plt.imshow(mode_2d, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                min(phi_values), max(phi_values)])
        plt.colorbar(label='模态幅值')

        # 在敏感区域上添加高亮
        highlighted = np.ma.masked_where(~sensitive_regions, grad_magnitude)
        plt.imshow(highlighted, cmap='Reds', alpha=0.6, extent=[min(theta_values), max(theta_values),
                                                                min(phi_values), max(phi_values)])

        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title(f'模态 {i + 1} 敏感区域分析 (红色区域为梯度最高的10%)')

        plt.savefig(os.path.join(angles_dir, f'mode_{i + 1}_sensitive_regions.png'), dpi=300)
        plt.close()


def parameter_sensitivity(pod_coeffs, param_data, param_names, num_modes=5, output_dir=None):
    """
    分析设计参数对模态系数的敏感度 - 增强版，处理常数数组和其他异常情况

    参数:
    pod_coeffs: POD系数，形状为 [num_models, r]
    param_data: 参数数据，形状为 [num_models, num_params]
    param_names: 参数名称列表
    num_modes: 要分析的模态数量
    output_dir: 输出目录
    """
    try:
        print("开始参数敏感性分析...")

        # 检查输入数据
        if pod_coeffs.size == 0 or param_data.size == 0:
            print("警告: 输入数据为空")
            return np.array([]), np.array([])

        if np.any(np.isnan(pod_coeffs)) or np.any(np.isinf(pod_coeffs)):
            print("警告: POD系数包含NaN或Inf值，将被替换为0")
            pod_coeffs = np.nan_to_num(pod_coeffs, nan=0.0, posinf=0.0, neginf=0.0)

        if np.any(np.isnan(param_data)) or np.any(np.isinf(param_data)):
            print("警告: 参数数据包含NaN或Inf值，将被替换为0")
            param_data = np.nan_to_num(param_data, nan=0.0, posinf=0.0, neginf=0.0)

        n_modes = min(num_modes, pod_coeffs.shape[1])
        n_params = param_data.shape[1]

        print(f"分析 {n_modes} 个模态与 {n_params} 个参数的关系")

        # 计算相关系数
        correlation = np.zeros((n_modes, n_params))
        p_values = np.zeros((n_modes, n_params))

        for i in range(n_modes):
            for j in range(n_params):
                # 检查数组是否为常数
                if np.std(pod_coeffs[:, i]) < 1e-10 or np.std(param_data[:, j]) < 1e-10:
                    # 如果任一数组为常数，设置相关系数为0，p值为1
                    correlation[i, j] = 0.0
                    p_values[i, j] = 1.0
                    print(f"警告: 模态{i + 1}或参数{j + 1}为常数，相关系数设为0")
                else:
                    try:
                        correlation[i, j], p_values[i, j] = stats.pearsonr(pod_coeffs[:, i], param_data[:, j])
                    except Exception as e:
                        print(f"计算模态{i + 1}与参数{j + 1}的相关系数时出错: {e}")
                        correlation[i, j] = 0.0
                        p_values[i, j] = 1.0

        try:
            # 可视化相关性热图
            plt.figure(figsize=(12, 8))
            plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Pearson相关系数')
            plt.xlabel('设计参数')
            plt.ylabel('POD模态')
            plt.title('参数-模态相关性矩阵')

            # 添加坐标标签
            plt.xticks(range(n_params), param_names, rotation=45, ha='right')
            plt.yticks(range(n_modes), [f'模态 {i + 1}' for i in range(n_modes)])

            # 添加相关系数值
            for i in range(n_modes):
                for j in range(n_params):
                    plt.text(j, i, f'{correlation[i, j]:.2f}',ha='center', va='center',color='white' if abs(correlation[i, j]) > 0.5 else 'black')

            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, 'parameter_sensitivity.png'), dpi=200)

            plt.close('all')
            print("成功生成相关性热图")
        except Exception as e:
            print(f"生成相关性热图时出错: {e}")
            plt.close('all')

        # 为每个主要模态构建回归模型
        try:
            models_dir = os.path.join(output_dir, 'regression_models')
            os.makedirs(models_dir, exist_ok=True)

            for i in range(n_modes):
                try:
                    # 检查数据的有效性
                    if np.std(pod_coeffs[:, i]) < 1e-10:
                        print(f"跳过模态 {i + 1} 的回归分析，因为其值几乎恒定")
                        continue

                    print(f"分析模态 {i + 1} 的回归模型...")

                    # 线性回归
                    lr = LinearRegression()
                    lr.fit(param_data, pod_coeffs[:, i])
                    y_pred_lr = lr.predict(param_data)
                    r2_lr = r2_score(pod_coeffs[:, i], y_pred_lr)

                    # Lasso回归（带L1正则化的线性回归）
                    lasso = Lasso(alpha=0.01)
                    lasso.fit(param_data, pod_coeffs[:, i])
                    y_pred_lasso = lasso.predict(param_data)
                    r2_lasso = r2_score(pod_coeffs[:, i], y_pred_lasso)

                    # 随机森林回归（非线性模型）
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(param_data, pod_coeffs[:, i])
                    y_pred_rf = rf.predict(param_data)
                    r2_rf = r2_score(pod_coeffs[:, i], y_pred_rf)

                    # 可视化结果
                    plt.figure(figsize=(15, 5))

                    # 将 R^2 的显示方式从直接使用上标改为使用特殊标记
                    plt.subplot(131)
                    plt.scatter(pod_coeffs[:, i], y_pred_lr, alpha=0.7)
                    plt.plot([-1, 1], [-1, 1], 'k--')
                    plt.xlabel('实际POD系数')
                    plt.ylabel('预测POD系数')
                    # 修改这行，避免直接使用上标
                    plt.title(f'线性回归 (R^2 = {r2_lr:.3f})')

                    plt.subplot(132)
                    plt.scatter(pod_coeffs[:, i], y_pred_lasso, alpha=0.7)
                    plt.plot([-1, 1], [-1, 1], 'k--')
                    plt.xlabel('实际POD系数')
                    plt.ylabel('预测POD系数')
                    # 修改这行，避免直接使用上标
                    plt.title(f'Lasso回归 (R^2 = {r2_lasso:.3f})')

                    plt.subplot(133)
                    plt.scatter(pod_coeffs[:, i], y_pred_rf, alpha=0.7)
                    plt.plot([-1, 1], [-1, 1], 'k--')
                    plt.xlabel('实际POD系数')
                    plt.ylabel('预测POD系数')
                    # 修改这行，避免直接使用上标
                    plt.title(f'随机森林回归 (R^2 = {r2_rf:.3f})')

                    plt.suptitle(f'模态 {i + 1} 的参数回归分析')
                    plt.tight_layout()
                    plt.savefig(os.path.join(models_dir, f'mode_{i + 1}_regression.png'), dpi=200)
                    plt.close('all')

                    # 参数重要性分析
                    plt.figure(figsize=(10, 6))

                    # 线性回归系数
                    plt.subplot(121)
                    coefs = lr.coef_
                    abs_coefs = np.abs(coefs)
                    idx = np.argsort(abs_coefs)[::-1]
                    plt.bar(range(len(coefs)), coefs[idx])
                    plt.xticks(range(len(coefs)), [param_names[j] for j in idx], rotation=45, ha='right')
                    plt.xlabel('参数')
                    plt.ylabel('线性回归系数')
                    plt.title('线性回归系数重要性')

                    # 随机森林特征重要性
                    plt.subplot(122)
                    importances = rf.feature_importances_
                    idx = np.argsort(importances)[::-1]
                    plt.bar(range(len(importances)), importances[idx])
                    plt.xticks(range(len(importances)), [param_names[j] for j in idx], rotation=45, ha='right')
                    plt.xlabel('参数')
                    plt.ylabel('特征重要性')
                    plt.title('随机森林特征重要性')

                    plt.suptitle(f'模态 {i + 1} 的参数重要性分析')
                    plt.tight_layout()
                    plt.savefig(os.path.join(models_dir, f'mode_{i + 1}_importance.png'), dpi=200)
                    plt.close('all')

                except Exception as e:
                    print(f"处理模态 {i + 1} 的回归分析时出错: {e}")
                    plt.close('all')

            print("完成回归分析")
        except Exception as e:
            print(f"回归分析过程中出错: {e}")
            plt.close('all')

        return correlation, p_values

    except Exception as e:
        print(f"参数敏感性分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return np.array([]), np.array([])


def reconstruct_rcs(data, phi_modes, pod_coeffs, mean_data, r_values, theta_values, phi_values, output_dir):
    """
    重构RCS数据并验证误差 - 增强版，处理NaN问题

    参数:
    data: 原始RCS数据，形状为 [num_models, num_angles]
    phi_modes: POD模态矩阵，形状为 [num_angles, num_modes]
    pod_coeffs: POD系数，形状为 [num_models, r]
    mean_data: 数据均值，形状为 [num_angles]
    r_values: 不同模态数量列表，用于比较重构精度
    theta_values: theta角度值
    phi_values: phi角度值
    output_dir: 输出目录
    """
    try:
        print("开始RCS重构与验证...")

        # 检查输入数据中是否包含NaN
        print(f"检查数据中是否包含NaN或Inf值:")
        print(f"  原始数据: {np.any(np.isnan(data)) or np.any(np.isinf(data))}")
        print(f"  模态: {np.any(np.isnan(phi_modes)) or np.any(np.isinf(phi_modes))}")
        print(f"  模态系数: {np.any(np.isnan(pod_coeffs)) or np.any(np.isinf(pod_coeffs))}")
        print(f"  均值: {np.any(np.isnan(mean_data)) or np.any(np.isinf(mean_data))}")

        # 替换NaN值
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        phi_modes_clean = np.nan_to_num(phi_modes, nan=0.0, posinf=0.0, neginf=0.0)
        pod_coeffs_clean = np.nan_to_num(pod_coeffs, nan=0.0, posinf=0.0, neginf=0.0)
        mean_data_clean = np.nan_to_num(mean_data, nan=0.0, posinf=0.0, neginf=0.0)

        n_theta = len(theta_values)
        n_phi = len(phi_values)
        num_models = data_clean.shape[0]

        # 创建重构目录
        recon_dir = os.path.join(output_dir, 'reconstruction')
        os.makedirs(recon_dir, exist_ok=True)

        # 对每个截断维度计算重构误差
        errors = []
        rel_errors = []
        r_max = max(r_values)

        for r in r_values:
            print(f"计算使用 {r} 个模态的重构误差...")

            # 确保r不超过可用的模态数
            r_actual = min(r, phi_modes_clean.shape[1], pod_coeffs_clean.shape[1])
            if r_actual < r:
                print(f"警告: 请求的模态数 {r} 超过了可用模态数 {r_actual}")

            # 重构数据
            try:
                reconstructed = np.dot(pod_coeffs_clean[:, :r_actual],phi_modes_clean[:, :r_actual].T) + mean_data_clean

                # 检查重构结果是否有效
                if np.any(np.isnan(reconstructed)) or np.any(np.isinf(reconstructed)):
                    print(f"警告: 重构数据包含NaN或Inf值，将被替换为0")
                    reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)

                # 计算重构误差
                error = np.linalg.norm(data_clean - reconstructed, axis=1)

                # 安全计算相对误差，避免除以零
                data_norm = np.linalg.norm(data_clean, axis=1)
                rel_error = np.zeros_like(error)
                for i in range(len(error)):
                    if data_norm[i] > 1e-10:  # 避免除以接近零的值
                        rel_error[i] = error[i] / data_norm[i]
                    else:
                        rel_error[i] = 0.0
                        print(f"警告: 模型 {i + 1} 的数据范数接近零，相对误差设为0")

                # 保存误差统计
                errors.append(error)
                rel_errors.append(rel_error)

                # 输出误差统计
                print(f"模态数量 r = {r}:")
                print(f"  平均绝对误差: {np.nanmean(error):.4f}")
                print(f"  相对误差: {np.nanmean(rel_error):.4f} ({np.nanmean(rel_error) * 100:.2f}%)")

            except Exception as e:
                print(f"计算使用 {r} 个模态的重构误差时出错: {e}")
                import traceback
                traceback.print_exc()
                errors.append(np.array([np.nan]))
                rel_errors.append(np.array([np.nan]))

        # 保存包含误差统计的文本文件
        try:
            with open(os.path.join(recon_dir, 'error_statistics.txt'), 'w') as f:
                for i, r in enumerate(r_values):
                    if i < len(errors) and len(errors[i]) > 0:
                        f.write(f"模态数量 r = {r}:\n")
                        f.write(f"  平均绝对误差: {np.nanmean(errors[i]):.4f}\n")
                        f.write(
                            f"  相对误差: {np.nanmean(rel_errors[i]):.4f} ({np.nanmean(rel_errors[i]) * 100:.2f}%)\n")
                        f.write("\n")
        except Exception as e:
            print(f"保存误差统计文件时出错: {e}")

        # 绘图部分放在单独的try块中
        try:
            # 可视化不同模态数量的误差
            plt.figure(figsize=(15, 8))  # 增加高度留出图例空间

            # 绝对误差箱线图
            plt.subplot(121)
            valid_errors = [e for e in errors if len(e) > 0 and not np.all(np.isnan(e))]
            if valid_errors:
                plt.boxplot(valid_errors, labels=[f"r={r_values[i]}" for i in range(len(valid_errors))])
                plt.ylabel('绝对误差 (RMSE)')
                plt.title('不同模态数量的绝对重构误差')
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, '无有效数据', ha='center', va='center')

            # 相对误差箱线图
            plt.subplot(122)
            valid_rel_errors = [rel * 100 for rel in rel_errors if len(rel) > 0 and not np.all(np.isnan(rel))]
            if valid_rel_errors:
                plt.boxplot(valid_rel_errors, labels=[f"r={r_values[i]}" for i in range(len(valid_rel_errors))])
                plt.ylabel('相对误差 (%)')
                plt.title('不同模态数量的相对重构误差')
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, '无有效数据', ha='center', va='center')

            # 先调整子图布局，但不使用tight_layout()
            plt.subplots_adjust(bottom=0.1)  # 为图例留出足够空间

            fig = plt.gcf()
            legend_elements = [
                Line2D([0], [0], color='k', lw=1.5, label='中位数'),
                Rectangle((0, 0), width=1, height=1, fc='white', ec='k', label='四分位距(Q1-Q3)'),
                Line2D([0], [0], marker='_', color='k', linestyle='None',
                       markersize=8, label='最小/最大值(1.5×IQR内)'),
                Line2D([0], [0], marker='o', color='k', linestyle='None',
                       markersize=6, markerfacecolor='k', label='离群值')
            ]

            # 将图例放置在图形底部
            fig.legend(handles=legend_elements, loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=9)

            plt.savefig(os.path.join(recon_dir, 'reconstruction_errors.png'), dpi=200, bbox_inches='tight')
            plt.close('all')

            # 选择一些代表性模型进行重构可视化
            sample_indices = []
            for idx in [0, num_models // 4, num_models // 2, 3 * num_models // 4, num_models - 1]:
                if idx < num_models and not np.any(np.isnan(data_clean[idx])):
                    sample_indices.append(idx)

            if not sample_indices:
                print("没有找到可用的代表性模型进行可视化")
                return

            print(f"选择 {len(sample_indices)} 个代表性模型进行可视化")

            for idx in sample_indices:
                plt.figure(figsize=(20, 5 * len(r_values)))

                # 原始RCS
                original = data_clean[idx].reshape(n_theta, n_phi).T

                plt.subplot(len(r_values) + 1, 3, 1)
                plt.imshow(original, cmap='jet', extent=[min(theta_values), max(theta_values),min(phi_values), max(phi_values)])
                plt.colorbar(label='RCS (dB)')
                plt.xlabel('俯仰角 $\\theta$ (度)')
                plt.ylabel('偏航角 $\\phi$ (度)')
                plt.title(f'原始RCS - 模型 {idx + 1}')

                # 3D表面
                ax = plt.subplot(len(r_values) + 1, 3, 2, projection='3d')
                theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)
                surf = ax.plot_surface(theta_grid, phi_grid, original, cmap='jet',linewidth=0, antialiased=True)
                plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='RCS (dB)')
                ax.set_xlabel('俯仰角 $\\theta$ (度)')
                ax.set_ylabel('偏航角 $\\phi$ (度)')
                ax.set_zlabel('RCS (dB)')
                ax.set_title(f'原始RCS - 模型 {idx + 1}')

                # 等值线图
                plt.subplot(len(r_values) + 1, 3, 3)
                cs = plt.contourf(theta_values, phi_values, original, 20, cmap='jet')
                plt.colorbar(cs, label='RCS (dB)')
                plt.contour(theta_values, phi_values, original, 10, colors='k', linewidths=0.5)
                plt.xlabel('俯仰角 $\\theta$ (度)')
                plt.ylabel('偏航角 $\\phi$ (度)')
                plt.title(f'原始RCS - 模型 {idx + 1}')

                # 各种模态数量的重构结果
                for i, r in enumerate(r_values):
                    r_actual = min(r, phi_modes_clean.shape[1], pod_coeffs_clean.shape[1])
                    reconstructed = (np.dot(pod_coeffs_clean[idx, :r_actual],
                                            phi_modes_clean[:, :r_actual].T) + mean_data_clean).reshape(n_theta,
                                                                                                        n_phi).T

                    # 安全计算误差
                    if np.linalg.norm(original) > 1e-10:
                        error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
                    else:
                        error = 0.0

                    # 重构RCS
                    plt.subplot(len(r_values) + 1, 3, (i + 1) * 3 + 1)
                    plt.imshow(reconstructed, cmap='jet', extent=[min(theta_values), max(theta_values),min(phi_values), max(phi_values)])
                    plt.colorbar(label='RCS (dB)')
                    plt.xlabel('俯仰角 $\\theta$ (度)')
                    plt.ylabel('偏航角 $\\phi$ (度)')
                    plt.title(f'重构RCS (r={r}) - 相对误差: {error * 100:.2f}%')

                    # 误差图
                    plt.subplot(len(r_values) + 1, 3, (i + 1) * 3 + 2)
                    diff = original - reconstructed
                    plt.imshow(diff, cmap='coolwarm', extent=[min(theta_values), max(theta_values),min(phi_values), max(phi_values)])
                    plt.colorbar(label='误差 (dB)')
                    plt.xlabel('俯仰角 $\\theta$ (度)')
                    plt.ylabel('偏航角 $\\phi$ (度)')
                    plt.title(f'误差分布 (r={r})')

                    # 剖面图
                    plt.subplot(len(r_values) + 1, 3, (i + 1) * 3 + 3)
                    mid_phi_idx = n_phi // 2
                    plt.plot(theta_values, original[mid_phi_idx, :], 'k-', label='原始')
                    plt.plot(theta_values, reconstructed[mid_phi_idx, :], 'r--', label='重构')
                    plt.xlabel('俯仰角 $\\theta$ (度)')
                    plt.ylabel('RCS (dB)')
                    plt.title(f'偏航角 $\\phi = {phi_values[mid_phi_idx]:.1f}°$ 的剖面比较')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                plt.savefig(os.path.join(recon_dir, f'model_{idx + 1}_reconstruction.png'), dpi=200)
                plt.close('all')

        except Exception as e:
            print(f"生成重构可视化图时出错: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')

    except Exception as e:
        print(f"RCS重构与验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


def predict_rcs_from_parameters(rcs_data_train, phi_modes, mean_rcs,
                                param_data_train, param_data_test, output_dir):
    """
    从设计参数预测RCS数据

    参数:
    rcs_data_train: 训练集RCS数据
    phi_modes: POD模态
    mean_rcs: 平均RCS数据
    param_data_train: 训练集参数数据
    param_data_test: 测试集参数数据
    output_dir: 输出目录

    返回:
    test_reconstruction: 预测的测试集RCS数据
    test_pod_coeffs: 预测的测试集POD系数
    """
    os.makedirs(output_dir, exist_ok=True)

    # 计算训练集POD系数
    pod_coeffs_train = compute_pod_coeffs(rcs_data_train, phi_modes, mean_rcs)

    # 创建回归模型(每个模态一个模型)
    models = []
    r = phi_modes.shape[1]

    for i in range(r):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(param_data_train, pod_coeffs_train[:, i])
        models.append(model)

        # 保存模型重要性
        feature_importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title(f'模态 {i + 1} 的参数重要性')
        plt.xlabel('参数索引')
        plt.ylabel('重要性')
        plt.savefig(os.path.join(output_dir, f'mode_{i + 1}_importance.png'))
        plt.close()

    # 预测测试集系数
    test_pod_coeffs = np.zeros((param_data_test.shape[0], r))
    for i in range(r):
        test_pod_coeffs[:, i] = models[i].predict(param_data_test)

    # 重构测试集RCS数据
    test_reconstruction = np.dot(test_pod_coeffs, phi_modes.T) + mean_rcs

    # 保存模型和预测结果
    np.save(os.path.join(output_dir, 'test_pod_coeffs.npy'), test_pod_coeffs)
    np.save(os.path.join(output_dir, 'test_reconstruction.npy'), test_reconstruction)

    return test_reconstruction, test_pod_coeffs


def evaluate_test_performance(rcs_data_test, test_reconstruction, theta_values, phi_values,
                              output_dir, statistics_function):
    """
    评估测试集性能

    参数:
    rcs_data_test: 测试集真实RCS数据
    test_reconstruction: 预测的测试集RCS数据
    theta_values: theta角度值
    phi_values: phi角度值
    output_dir: 输出目录
    statistics_function: 计算统计参数的函数
    """
    # 计算重构误差
    mse = np.mean((rcs_data_test - test_reconstruction) ** 2, axis=1)
    rmse = np.sqrt(mse)

    # 计算相对误差
    norm_test = np.linalg.norm(rcs_data_test, axis=1)
    norm_error = np.linalg.norm(rcs_data_test - test_reconstruction, axis=1)
    rel_error = norm_error / np.maximum(norm_test, 1e-10)

    # 保存误差统计
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.hist(rmse, bins=20)
    plt.title(f'测试集RMSE分布\n平均RMSE: {np.mean(rmse):.4f}')
    plt.xlabel('RMSE')
    plt.ylabel('频数')

    plt.subplot(122)
    plt.hist(rel_error * 100, bins=20)
    plt.title(f'测试集相对误差分布 (%)\n平均相对误差: {np.mean(rel_error) * 100:.2f}%')
    plt.xlabel('相对误差 (%)')
    plt.ylabel('频数')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()

    # 选择一些代表性样本进行可视化
    n_theta = len(theta_values)
    n_phi = len(phi_values)

    # 选择相对误差最小、最大和中等的样本
    idx_min = np.argmin(rel_error)
    idx_max = np.argmax(rel_error)
    idx_med = np.argsort(rel_error)[len(rel_error) // 2]

    for idx, label in zip([idx_min, idx_med, idx_max], ['best', 'median', 'worst']):
        # 重塑为2D角度矩阵
        original = rcs_data_test[idx].reshape(n_theta, n_phi).T
        reconstructed = test_reconstruction[idx].reshape(n_theta, n_phi).T

        plt.figure(figsize=(15, 5))

        # 原始RCS
        plt.subplot(131)
        plt.imshow(original, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                 min(phi_values), max(phi_values)])
        plt.colorbar(label='RCS (dB)')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title('原始RCS')

        # 重构RCS
        plt.subplot(132)
        plt.imshow(reconstructed, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                      min(phi_values), max(phi_values)])
        plt.colorbar(label='RCS (dB)')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title('重构RCS')

        # 误差
        plt.subplot(133)
        diff = original - reconstructed
        plt.imshow(diff, cmap='coolwarm', extent=[min(theta_values), max(theta_values),
                                                  min(phi_values), max(phi_values)])
        plt.colorbar(label='误差 (dB)')
        plt.xlabel('俯仰角 $\\theta$ (度)')
        plt.ylabel('偏航角 $\\phi$ (度)')
        plt.title(f'误差 (相对误差: {rel_error[idx] * 100:.2f}%)')

        plt.suptitle(f'{label.capitalize()} 样本 (索引 {idx})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{label}.png'))
        plt.close()

    # 计算原始和重构数据的统计参数
    test_stats_original = []
    test_stats_reconstructed = []

    n_test = rcs_data_test.shape[0]
    for i in range(n_test):
        # 重塑为2D角度矩阵并转置以匹配CSV格式
        original_2d = rcs_data_test[i].reshape(n_theta, n_phi).T
        reconstructed_2d = test_reconstruction[i].reshape(n_theta, n_phi).T

        # 计算统计参数
        stats_orig = statistics_function(original_2d, theta_values, phi_values, f'Test{i + 1}')
        stats_recon = statistics_function(reconstructed_2d, theta_values, phi_values, f'Test{i + 1}')

        test_stats_original.append(stats_orig)
        test_stats_reconstructed.append(stats_recon)

    # 将统计数据转换为DataFrame并保存
    df_orig = pd.DataFrame(test_stats_original)
    df_recon = pd.DataFrame(test_stats_reconstructed)

    df_orig.to_csv(os.path.join(output_dir, 'stats_original.csv'), index=False)
    df_recon.to_csv(os.path.join(output_dir, 'stats_reconstructed.csv'), index=False)

    # 比较统计参数
    compare_statistics(df_orig, df_recon, output_dir)


def compare_statistics(stats_orig, stats_recon, output_dir):
    """
    比较原始和重构RCS的统计参数

    参数:
    stats_orig: 原始RCS的统计参数DataFrame
    stats_recon: 重构RCS的统计参数DataFrame
    output_dir: 输出目录
    """
    # 计算相对误差
    numeric_columns = stats_orig.select_dtypes(include=[np.number]).columns

    # 创建统计参数对比表
    comparison = pd.DataFrame(index=numeric_columns)

    # 计算均值
    comparison['原始均值'] = stats_orig[numeric_columns].mean()
    comparison['重构均值'] = stats_recon[numeric_columns].mean()

    # 计算相对误差
    comparison['相对误差(%)'] = abs(comparison['原始均值'] - comparison['重构均值']) / abs(comparison['原始均值']) * 100

    # 计算相关系数
    correlation = []
    for col in numeric_columns:
        if stats_orig[col].std() > 0 and stats_recon[col].std() > 0:
            corr = np.corrcoef(stats_orig[col], stats_recon[col])[0, 1]
            correlation.append(corr)
        else:
            correlation.append(np.nan)

    comparison['相关系数'] = correlation

    # 保存比较结果
    comparison.to_csv(os.path.join(output_dir, 'stats_comparison.csv'))

    # 可视化关键统计参数对比
    key_stats = ['均值(dBsm)', '中位数(dBsm)', '极大值(dBsm)', '极小值(dBsm)', '极差', '标准差']
    key_stats = [s for s in key_stats if s in numeric_columns]

    for stat in key_stats:
        plt.figure(figsize=(10, 6))
        plt.scatter(stats_orig[stat], stats_recon[stat])
        plt.plot([stats_orig[stat].min(), stats_orig[stat].max()],
                 [stats_orig[stat].min(), stats_orig[stat].max()], 'r--')

        plt.xlabel(f'原始 {stat}')
        plt.ylabel(f'重构 {stat}')
        plt.title(f'{stat} 对比\n相关系数: {comparison.loc[stat, "相关系数"]:.4f}')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{stat.replace("(", "").replace(")", "")}.png'))
        plt.close()


def load_prediction_parameters(param_file, param_names):
    """
    加载要进行预测的设计参数

    参数:
    param_file: 参数文件路径
    param_names: 参数名称列表

    返回:
    预测参数的Numpy数组
    """
    try:
        # 读取参数文件
        df = pd.read_csv(param_file)

        # 检查是否包含所有需要的参数
        missing_params = [p for p in param_names if p not in df.columns]
        if missing_params:
            print(f"警告: 参数文件缺少以下参数: {missing_params}")
            # 如果参数文件使用不同名称，尝试按顺序匹配
            if len(df.columns) >= len(param_names):
                print("使用按列顺序匹配...")
                pred_params = df.iloc[:, :len(param_names)].values
            else:
                raise ValueError("参数不足，无法进行预测")
        else:
            # 按原始参数名称顺序提取参数
            pred_params = df[param_names].values

        print(f"加载了 {pred_params.shape[0]} 组预测参数")
        return pred_params

    except Exception as e:
        print(f"加载预测参数时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def generate_rcs_predictions(pred_params, param_data_train, pod_coeffs_train,
                             phi_modes, mean_rcs, theta_values, phi_values, output_dir):
    """
    根据给定的设计参数生成RCS预测

    参数:
    pred_params: 预测参数
    param_data_train: 训练集参数数据
    pod_coeffs_train: 训练集POD系数
    phi_modes: POD模态
    mean_rcs: 平均RCS值
    theta_values: theta角度值
    phi_values: phi角度值
    output_dir: 输出目录

    返回:
    预测的RCS数据
    """
    try:
        r = phi_modes.shape[1]
        n_pred = pred_params.shape[0]

        # 标准化预测参数
        # 使用训练集的均值和标准差
        param_mean = np.mean(param_data_train, axis=0)
        param_std = np.std(param_data_train, axis=0)
        pred_params_scaled = (pred_params - param_mean) / np.maximum(param_std, 1e-10)

        # 创建回归模型(每个模态一个模型)
        models = []
        for i in range(r):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(param_data_train, pod_coeffs_train[:, i])
            models.append(model)

        # 预测POD系数
        pred_pod_coeffs = np.zeros((n_pred, r))
        for i in range(r):
            pred_pod_coeffs[:, i] = models[i].predict(pred_params_scaled)

        # 重构RCS数据
        pred_rcs = np.dot(pred_pod_coeffs, phi_modes.T) + mean_rcs

        # 保存预测结果
        np.save(os.path.join(output_dir, 'pred_pod_coeffs.npy'), pred_pod_coeffs)
        np.save(os.path.join(output_dir, 'pred_rcs.npy'), pred_rcs)

        # 可视化预测结果
        n_theta = len(theta_values)
        n_phi = len(phi_values)

        # 选择几个有代表性的样本进行可视化
        for i in range(min(5, n_pred)):
            # 重塑为2D角度矩阵
            rcs_2d = pred_rcs[i].reshape(n_theta, n_phi).T

            plt.figure(figsize=(15, 5))

            # 2D热图
            plt.subplot(131)
            plt.imshow(rcs_2d, cmap='jet', extent=[min(theta_values), max(theta_values),
                                                   min(phi_values), max(phi_values)])
            plt.colorbar(label='RCS (dB)')
            plt.xlabel('俯仰角 $\\theta$ (度)')
            plt.ylabel('偏航角 $\\phi$ (度)')
            plt.title(f'预测样本 {i + 1} - 2D热图')

            # 3D表面
            theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)
            ax = plt.subplot(132, projection='3d')
            surf = ax.plot_surface(theta_grid, phi_grid, rcs_2d, cmap='jet',
                                   linewidth=0, antialiased=True)
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='RCS (dB)')
            ax.set_xlabel('俯仰角 $\\theta$ (度)')
            ax.set_ylabel('偏航角 $\\phi$ (度)')
            ax.set_zlabel('RCS (dB)')
            ax.set_title(f'预测样本 {i + 1} - 3D表面')

            # 等值线图
            plt.subplot(133)
            cs = plt.contourf(theta_values, phi_values, rcs_2d, 20, cmap='jet')
            plt.colorbar(cs, label='RCS (dB)')
            plt.contour(theta_values, phi_values, rcs_2d, 10, colors='k', linewidths=0.5)
            plt.xlabel('俯仰角 $\\theta$ (度)')
            plt.ylabel('偏航角 $\\phi$ (度)')
            plt.title(f'预测样本 {i + 1} - 等值线图')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'pred_sample_{i + 1}.png'))
            plt.close()

        # 计算并保存统计数据
        # 尝试导入统计计算函数
        try:
            # 先尝试直接导入
            from calculatercs import calculate_statistics_from_data
            print("已成功导入 calculate_statistics_from_data 函数")
        except ImportError:
            print("无法导入 calculate_statistics_from_data 函数，使用简化版函数")

            # 如果无法导入，定义一个简化版函数
            def calculate_statistics_from_data(rcs_2d, theta_values, phi_values, model_name):
                """简化版统计计算函数"""
                # 找出最大值和最小值的2D索引
                max_idx = np.unravel_index(np.argmax(rcs_2d), rcs_2d.shape)
                min_idx = np.unravel_index(np.argmin(rcs_2d), rcs_2d.shape)

                # 获取最大值、最小值及其对应的θ和φ值
                max_value_dbsm = rcs_2d[max_idx]
                max_phi = phi_values[max_idx[0]]
                max_theta = theta_values[max_idx[1]]

                min_value_dbsm = rcs_2d[min_idx]
                min_phi = phi_values[min_idx[0]]
                min_theta = theta_values[min_idx[1]]

                # 计算基本统计量
                mean_value_dbsm = np.mean(rcs_2d)
                median_value_dbsm = np.median(rcs_2d)
                std_dev = np.std(rcs_2d)

                return {
                    '模型': model_name,
                    '均值(dBsm)': mean_value_dbsm,
                    '中位数(dBsm)': median_value_dbsm,
                    '极大值(dBsm)': max_value_dbsm,
                    '极大值θ': max_theta,
                    '极大值φ': max_phi,
                    '极小值(dBsm)': min_value_dbsm,
                    '极小值θ': min_theta,
                    '极小值φ': min_phi,
                    '标准差': std_dev
                }

        statistics = []
        for i in range(n_pred):
            rcs_2d = pred_rcs[i].reshape(n_theta, n_phi).T
            stats = calculate_statistics_from_data(rcs_2d, theta_values, phi_values, f'Pred{i + 1}')
            statistics.append(stats)

        df_stats = pd.DataFrame(statistics)
        df_stats.to_csv(os.path.join(output_dir, 'pred_statistics.csv'), index=False)

        return pred_rcs

    except Exception as e:
        print(f"生成RCS预测时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])