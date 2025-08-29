"""
主程序：用于加载数据、进行POD分解和模态分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_loader import load_parameters, load_rcs_data
from pod_analysis import perform_pod, compute_pod_coeffs, energy_analysis
from model_analysis import visualize_modes, parameter_sensitivity, angle_sensitivity, reconstruct_rcs, \
    predict_rcs_from_parameters, evaluate_test_performance, load_prediction_parameters, generate_rcs_predictions, \
    compare_statistics
from calculatercs import calculate_statistics_from_data

# 临时修复：多重导入尝试
try:
    # 方案1：尝试使用新的模块化结构
    try:
        from autoencoder_analysis import perform_autoencoder_analysis, compare_with_pod_results, check_pytorch_availability

        pytorch_info = check_pytorch_availability()
        AUTOENCODER_AVAILABLE = pytorch_info['pytorch']
        print("使用新的模块化自编码器结构")

    except (ImportError, AttributeError):
        # 方案2：回退到原始的autoencoder_analysis模块
        from autoencoder_analysis import perform_autoencoder_analysis, compare_with_pod_results, PYTORCH_AVAILABLE

        AUTOENCODER_AVAILABLE = PYTORCH_AVAILABLE
        print("使用原始的autoencoder_analysis模块")

    if AUTOENCODER_AVAILABLE:
        print("PyTorch功能可用")
    else:
        print("PyTorch功能不可用")

except ImportError as e:
    print(f"自编码器导入失败: {e}")
    AUTOENCODER_AVAILABLE = False


    # 创建fallback函数
    def perform_autoencoder_analysis(*args, **kwargs):
        print("自编码器功能不可用，请检查PyTorch安装")
        return {}


    def compare_with_pod_results(*args, **kwargs):
        print("自编码器对比功能不可用")
        return


# Windows编码设置
import sys
if sys.platform.startswith('win'):
    try:
        # 尝试设置UTF-8输出
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 在plt.figure()之前添加以下代码设置中文字体
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置使用黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


def main(params_path="../parameter/parameters_sorted.csv",
         rcs_dir="../parameter/csv_output",
         output_dir="./results",
         analyze_freq="both",
         num_models=100,
         num_train = "70,80",
         predict_mode = False,
         param_file = None,
         latent_dims = [5, 10, 15, 20],
         model_types = ['standard', 'vae'],
         ae_epochs = 200,
         ae_device = 'auto',
         ae_learning_rate = 0.001,
         ae_batch_size = 0,
         energy_threshold = 95.0,
         num_modes_visualize = 10
         ):
    """
    主程序，控制整个分析流程

    参数:
    params_path: 设计参数CSV文件路径
    rcs_dir: RCS数据CSV文件目录
    output_dir: 分析结果输出目录
    analyze_freq: 要分析的频率 ("1.5G", "3G", 或 "both")
    num_models: 要分析的模型数量
    """
    """
       主程序，控制整个分析流程
       """
    print(f"进入main函数，接收到的参数:")
    print(f"  params_path: {params_path}")
    print(f"  rcs_dir: {rcs_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  analyze_freq: {analyze_freq}")
    print(f"  num_models: {num_models}")
    print(f"  num_train: {num_train}")
    print(f"  predict_mode: {predict_mode}")
    print(f"  param_file: {param_file}")

    try:
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 处理训练集大小参数 - 确保正确解析
        print(f"原始num_train参数: {num_train}")
        if isinstance(num_train, str) and ',' in num_train:
            train_sizes = [int(n.strip()) for n in num_train.split(',')]
        else:
            try:
                # 如果是单个数字(字符串或整数)
                train_sizes = [int(num_train)]
            except (ValueError, TypeError):
                print(f"警告: 无法解析训练集大小 '{num_train}'，使用默认值80")
                train_sizes = [80]

        print(f"解析后的训练集大小列表: {train_sizes}")

        print("开始加载参数数据...")
        # 加载参数数据
        param_data, param_names = load_parameters(params_path)
        print(f"加载了 {param_data.shape[0]} 个模型的 {param_data.shape[1]} 个参数")

        # 标准化参数数据
        scaler = StandardScaler()
        param_data_scaled = scaler.fit_transform(param_data)

        # 分析指定频率的数据
        if analyze_freq in ["1.5G", "both"]:
            try:
                print("开始加载1.5GHz RCS数据...")
                # 加载1.5GHz RCS数据
                rcs_data_1p5g, theta_values, phi_values, available_models_1p5g = load_rcs_data(rcs_dir, "1.5G",
                                                                                               num_models=num_models)
                print(f"加载了1.5GHz下 {rcs_data_1p5g.shape[0]} 个模型的RCS数据")
                print(
                    f"角度范围: theta {min(theta_values)}° 到 {max(theta_values)}°, phi {min(phi_values)}° 到 {max(phi_values)}°")

                # 对RCS数据进行对数处理，添加小值防止log(0)
                rcs_data_1p5g_db = 10 * np.log10(np.maximum(rcs_data_1p5g, 1e-10))

                print("\n开始分析1.5GHz RCS数据...")
                analyze_frequency_data(rcs_data_1p5g_db, theta_values, phi_values,
                                       param_data, param_data_scaled,
                                       param_names, "1.5GHz", output_dir,
                                       available_models_1p5g, train_sizes,
                                       predict_mode, param_file,
                                       latent_dims, model_types, ae_epochs,
                                       ae_device, ae_learning_rate, ae_batch_size,
                                       energy_threshold, num_modes_visualize)
            except Exception as e:
                print(f"处理1.5GHz数据时发生错误: {e}")
                import traceback
                traceback.print_exc()

        if analyze_freq in ["3G", "both"]:
            try:
                print("开始加载3GHz RCS数据...")
                # 加载3GHz RCS数据
                rcs_data_3g, theta_values, phi_values, available_models_3g = load_rcs_data(rcs_dir, "3G",
                                                                                           num_models=num_models)
                print(f"加载了3GHz下 {rcs_data_3g.shape[0]} 个模型的RCS数据")

                # 对RCS数据进行对数处理，添加小值防止log(0)
                rcs_data_3g_db = 10 * np.log10(np.maximum(rcs_data_3g, 1e-10))

                print("\n开始分析3GHz RCS数据...")
                analyze_frequency_data(rcs_data_3g_db, theta_values, phi_values, param_data, param_data_scaled,
                                       param_names, "3GHz", output_dir, available_models_3g, train_sizes, predict_mode, param_file,
                                       latent_dims, model_types, ae_epochs,
                                       ae_device, ae_learning_rate, ae_batch_size,
                                       energy_threshold, num_modes_visualize)
            except Exception as e:
                print(f"处理3GHz数据时发生错误: {e}")
                import traceback
                traceback.print_exc()

        print("\nPOD和模态分析完成。结果保存在", output_dir)

    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


# 在main.py的analyze_frequency_data函数中添加模型索引处理

def analyze_frequency_data(rcs_data, theta_values, phi_values, param_data, param_data_scaled,
                           param_names, freq_label, output_dir, available_models=None,
                           train_sizes=[90], predict_mode=False, param_file=None,
                           latent_dims=[5, 10, 15, 20], model_types=['standard', 'vae'], ae_epochs=200,
                           ae_device='auto', ae_learning_rate=0.001, ae_batch_size=0,
                           energy_threshold=95.0, num_modes_visualize=10):
    """
    分析特定频率下的RCS数据 - 增强版

    参数:
    rcs_data: 形状为 [num_models, num_angles] 的RCS数据
    theta_values: theta角度值
    phi_values: phi角度值
    param_data: 原始参数数据
    param_data_scaled: 标准化后的参数数据
    param_names: 参数名称列表
    freq_label: 频率标签（如"1.5GHz"或"3GHz"）
    output_dir: 输出目录
    available_models: 可用模型的索引列表
    train_sizes: 训练集大小列表
    predict_mode: 是否启用预测模式
    param_file: 预测模式下的参数文件路径
    """
    try:
        # 导入计算RCS统计数据的函数
        from calculatercs import calculate_statistics_from_data

        # 创建频率对应的输出目录
        freq_dir = os.path.join(output_dir, freq_label)
        os.makedirs(freq_dir, exist_ok=True)

        # 保存可用模型信息
        if available_models is not None:
            np.save(os.path.join(freq_dir, "available_models.npy"), np.array(available_models))
            print(f"可用模型数量: {len(available_models)}")

        # 如果只有参数数据的子集可用，则只使用相应的参数数据
        if available_models is not None:
            # 将模型索引转换为0-based索引（对应数组索引）
            model_indices = [i - 1 for i in available_models]
            param_data_subset = param_data[model_indices]
            param_data_scaled_subset = param_data_scaled[model_indices]
        else:
            param_data_subset = param_data
            param_data_scaled_subset = param_data_scaled

        # 对每个训练集大小进行循环
        for train_size in train_sizes:
            print(f"\n===== 分析训练集大小: {train_size} =====")

            # 确保训练集大小不超过可用模型数量
            if train_size > rcs_data.shape[0]:
                print(f"警告: 训练集大小 {train_size} 超过了可用模型数量 {rcs_data.shape[0]}")
                train_size = rcs_data.shape[0]
                print(f"调整训练集大小为: {train_size}")

            # 创建特定训练集大小的输出目录
            train_dir = os.path.join(freq_dir, f"train_{train_size}")
            os.makedirs(train_dir, exist_ok=True)

            # 随机划分训练集和测试集
            num_models = rcs_data.shape[0]
            np.random.seed(42)  # 确保结果可重复
            indices = np.random.permutation(num_models)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            print(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")

            # 输出训练集索引
            print(f"\n训练集索引 (共 {len(train_indices)} 个):")
            print(train_indices)

            # 如果available_models不为None，转换为真实模型编号
            if available_models is not None:
                real_train_indices = [available_models[i] for i in train_indices]
                real_test_indices = [available_models[i] for i in test_indices]

                print(f"\n训练集真实模型编号 (共 {len(real_train_indices)} 个):")
                print(real_train_indices)

                # 保存真实模型编号到文件
                train_indices_file = os.path.join(train_dir, "train_models.txt")
                with open(train_indices_file, 'w') as f:
                    f.write(f"训练集大小: {train_size}\n")
                    f.write("训练集模型编号:\n")
                    for i, model_idx in enumerate(real_train_indices):
                        f.write(f"{i + 1}. {model_idx}\n")

                test_indices_file = os.path.join(train_dir, "test_models.txt")
                with open(test_indices_file, 'w') as f:
                    f.write(f"测试集大小: {len(real_test_indices)}\n")
                    f.write("测试集模型编号:\n")
                    for i, model_idx in enumerate(real_test_indices):
                        f.write(f"{i + 1}. {model_idx}\n")

                print(f"训练集和测试集模型编号已保存到:\n  {train_indices_file}\n  {test_indices_file}")
            else:
                # 保存数组索引到文件
                train_indices_file = os.path.join(train_dir, "train_indices.txt")
                with open(train_indices_file, 'w') as f:
                    f.write(f"训练集大小: {train_size}\n")
                    f.write("训练集索引:\n")
                    for i, idx in enumerate(train_indices):
                        f.write(f"{i + 1}. {idx}\n")

                test_indices_file = os.path.join(train_dir, "test_indices.txt")
                with open(test_indices_file, 'w') as f:
                    f.write(f"测试集大小: {len(test_indices)}\n")
                    f.write("测试集索引:\n")
                    for i, idx in enumerate(test_indices):
                        f.write(f"{i + 1}. {idx}\n")

                print(f"训练集和测试集索引已保存到:\n  {train_indices_file}\n  {test_indices_file}")

            # 训练集信息
            if available_models is not None:
                train_df = pd.DataFrame({
                    'Index': train_indices,
                    'Model_Number': [available_models[i] for i in train_indices]
                })
            else:
                train_df = pd.DataFrame({
                    'Index': train_indices
                })
            train_df.to_csv(os.path.join(train_dir, "train_indices.csv"), index=False)

            # 测试集信息
            if available_models is not None:
                test_df = pd.DataFrame({
                    'Index': test_indices,
                    'Model_Number': [available_models[i] for i in test_indices]
                })
            else:
                test_df = pd.DataFrame({
                    'Index': test_indices
                })
            test_df.to_csv(os.path.join(train_dir, "test_indices.csv"), index=False)

            # 可视化训练集和测试集在参数空间中的分布
            if param_data_subset.shape[1] >= 2:  # 至少有两个参数
                plt.figure(figsize=(10, 8))
                plt.scatter(param_data_subset[train_indices, 0], param_data_subset[train_indices, 1],
                            c='blue', marker='o', label='训练集')
                plt.scatter(param_data_subset[test_indices, 0], param_data_subset[test_indices, 1],
                            c='red', marker='x', label='测试集')
                plt.xlabel(param_names[0])
                plt.ylabel(param_names[1])
                plt.title(f'训练集和测试集在参数空间的分布 (训练集大小: {train_size})')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(train_dir, "dataset_distribution.png"))
                plt.close()

            # 分离参数数据
            param_data_train = param_data_scaled_subset[train_indices]
            param_data_test = param_data_scaled_subset[test_indices] if len(test_indices) > 0 else np.array([])

            # 分离RCS数据
            rcs_data_train = rcs_data[train_indices]
            rcs_data_test = rcs_data[test_indices] if len(test_indices) > 0 else np.array([])

            # 保存训练集和测试集索引
            np.save(os.path.join(train_dir, "train_indices.npy"), train_indices)
            np.save(os.path.join(train_dir, "test_indices.npy"), test_indices)

            # 执行POD分解(仅使用训练集)
            print("执行POD分解(仅使用训练集)...")
            try:
                phi_modes_train, lambda_values_train, mean_rcs_train = perform_pod(rcs_data_train)

                # 能量分析
                print("进行能量分析...")
                modes_90, modes_95, modes_99 = energy_analysis(lambda_values_train, train_dir)

                # 选择保留模态数量（这里选择95%能量）
                r = max(1, min(modes_95, phi_modes_train.shape[1] - 1))  # 确保至少保留1个模态，且不超出范围
                print(f"选择保留前 {r} 个模态，覆盖95%能量")

                # 保存POD结果
                np.save(os.path.join(train_dir, "pod_modes.npy"), phi_modes_train)
                np.save(os.path.join(train_dir, "lambda_values.npy"), lambda_values_train)
                np.save(os.path.join(train_dir, "mean_rcs.npy"), mean_rcs_train)
            except Exception as e:
                print(f"POD分解或能量分析过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                # 创建备用数据继续后续分析
                r = 1
                continue

            # 计算训练集POD系数
            try:
                print("计算训练集POD系数...")
                pod_coeffs_train = compute_pod_coeffs(rcs_data_train, phi_modes_train[:, :r], mean_rcs_train)
                np.save(os.path.join(train_dir, "pod_coeffs_train.npy"), pod_coeffs_train)
            except Exception as e:
                print(f"计算训练集POD系数时发生错误: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 可视化主要模态
            print("可视化主要模态...")
            try:
                num_modes_to_visualize = min(10, r)  # 可视化前10个模态
                visualize_modes(phi_modes_train[:, :num_modes_to_visualize], theta_values, phi_values, train_dir)
            except Exception as e:
                print(f"模态可视化过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

            # 参数敏感性分析
            try:
                print("进行参数敏感性分析...")
                parameter_sensitivity(pod_coeffs_train, param_data_train, param_names,
                                      num_modes=num_modes_to_visualize, output_dir=train_dir)
            except Exception as e:
                print(f"参数敏感性分析过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

            # 角度敏感性分析
            try:
                print("进行角度敏感性分析...")
                angle_sensitivity(phi_modes_train[:, :min(5, phi_modes_train.shape[1])],
                                  theta_values, phi_values, train_dir)
            except Exception as e:
                print(f"角度敏感性分析过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

            # 训练集重构与验证
            try:
                print("进行训练集RCS重构与验证...")
                recon_dir = os.path.join(train_dir, "reconstruction_train")
                os.makedirs(recon_dir, exist_ok=True)

                r_values = [min(5, r), min(10, r), r]
                r_values = sorted(list(set(r_values)))  # 去除重复值并排序
                reconstruct_rcs(rcs_data_train, phi_modes_train, pod_coeffs_train, mean_rcs_train,
                                r_values, theta_values, phi_values, recon_dir)
                # 使用最优模态数量重构进行性能评估
                reconstructed_train_optimal = np.dot(pod_coeffs_train, phi_modes_train[:, :r].T) + mean_rcs_train

                # 计算R^2和MSE用于与Autoencoder对比
                from sklearn.metrics import r2_score, mean_squared_error
                pod_r2 = r2_score(rcs_data_train.flatten(), reconstructed_train_optimal.flatten())
                pod_mse = mean_squared_error(rcs_data_train, reconstructed_train_optimal)

                print(f"POD重构性能: R^2 = {pod_r2:.6f}, MSE = {pod_mse:.6f}")

                # 保存POD性能指标供后续使用
                pod_performance = {
                    'r2': pod_r2,
                    'mse': pod_mse,
                    'n_modes': r,
                    'pod_coeffs': pod_coeffs_train,
                    'reconstruction_error': np.mean((rcs_data_train - reconstructed_train_optimal) ** 2, axis=1)
                }

                # 计算训练集RCS统计数据
                print("计算训练集RCS统计数据...")
                n_theta = len(theta_values)
                n_phi = len(phi_values)

                # 使用最优模态数量重构
                reconstructed_train = np.dot(pod_coeffs_train, phi_modes_train[:, :r].T) + mean_rcs_train

                # 计算原始和重构数据的统计参数
                train_stats_original = []
                train_stats_reconstructed = []

                for i in range(min(5, len(train_indices))):  # 只分析前5个样本
                    # 重塑为2D角度矩阵
                    original_2d = rcs_data_train[i].reshape(n_theta, n_phi).T
                    reconstructed_2d = reconstructed_train[i].reshape(n_theta, n_phi).T

                    # 计算统计参数
                    stats_orig = calculate_statistics_from_data(original_2d, theta_values, phi_values, f'Train{i + 1}')
                    stats_recon = calculate_statistics_from_data(reconstructed_2d, theta_values, phi_values,
                                                                 f'Train{i + 1}')

                    train_stats_original.append(stats_orig)
                    train_stats_reconstructed.append(stats_recon)

                # 将统计数据转换为DataFrame并保存
                df_orig = pd.DataFrame(train_stats_original)
                df_recon = pd.DataFrame(train_stats_reconstructed)

                df_orig.to_csv(os.path.join(recon_dir, 'stats_original.csv'), index=False)
                df_recon.to_csv(os.path.join(recon_dir, 'stats_reconstructed.csv'), index=False)

                # 比较统计参数
                compare_statistics(df_orig, df_recon, recon_dir)

            except Exception as e:
                print(f"训练集RCS重构与验证过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

            # Autoencoder分析 - 添加在POD分析完成后，测试集分析前
            try:
                if AUTOENCODER_AVAILABLE:
                    print("\n" + "=" * 50)
                    print("开始Autoencoder降维分析...")
                    print("=" * 50)

                    # 准备POD结果用于对比（使用实际计算的性能指标）
                    pod_results = pod_performance

                    # 执行Autoencoder分析
                    autoencoder_results = perform_autoencoder_analysis(
                        rcs_data=rcs_data,
                        theta_values=theta_values,
                        phi_values=phi_values,
                        param_data=param_data_scaled_subset,
                        param_names=param_names,
                        freq_label=freq_label,
                        output_dir=train_dir,
                        train_indices=train_indices,
                        test_indices=test_indices if len(test_indices) > 0 else None,
                        latent_dims=latent_dims,  # 使用传入的参数
                        model_types=model_types,  # 使用传入的参数
                        device=ae_device  # 使用传入的设备参数
                    )

                    # 与POD结果对比
                    if autoencoder_results:
                        print("\n开始POD vs Autoencoder对比分析...")
                        compare_with_pod_results(autoencoder_results, pod_results, train_dir)
                        
                        # 尝试进行增强版对比分析
                        try:
                            from enhanced_comparison import enhanced_compare_with_pod_results
                            print("\n开始增强版综合对比分析...")
                            enhanced_compare_with_pod_results(autoencoder_results, pod_results, train_dir, train_dir)
                        except Exception as enhanced_error:
                            print(f"增强版对比分析失败: {enhanced_error}")
                        
                        print("Autoencoder分析完成！")
                    else:
                        print("Autoencoder分析未产生有效结果")

                else:
                    print("\n" + "=" * 50)
                    print("PyTorch不可用，跳过Autoencoder分析")
                    print("如需使用Autoencoder功能，请安装PyTorch:")
                    print("pip install torch torchvision torchaudio")
                    print("=" * 50)

            except Exception as e:
                print(f"\nAutoencoder分析过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                print("Autoencoder分析失败，继续执行其他分析...")
            # ===== Autoencoder分析结束 =====

            # 测试集上的预测和分析
            if len(test_indices) > 0:
                try:
                    print("在测试集上进行预测和分析...")
                    # 创建测试结果目录
                    test_dir = os.path.join(train_dir, "test")
                    os.makedirs(test_dir, exist_ok=True)

                    # 创建模型并预测测试集
                    test_reconstruction, test_pod_coeffs = predict_rcs_from_parameters(
                        rcs_data_train, phi_modes_train[:, :r], mean_rcs_train,
                        param_data_train, param_data_test,
                        os.path.join(test_dir, "model"))

                    # 评估测试集性能
                    evaluate_test_performance(
                        rcs_data_test, test_reconstruction, theta_values, phi_values,
                        test_dir, calculate_statistics_from_data)

                except Exception as e:
                    print(f"测试集预测和分析过程中发生错误: {e}")
                    import traceback
                    traceback.print_exc()

            # 处理预测模式
            if predict_mode and param_file is not None:
                try:
                    print("启用预测模式，根据指定参数预测RCS...")
                    predict_dir = os.path.join(train_dir, "predictions")
                    os.makedirs(predict_dir, exist_ok=True)

                    # 读取预测用的参数
                    pred_params = load_prediction_parameters(param_file, param_names)

                    if len(pred_params) > 0:
                        # 生成预测结果
                        predicted_rcs = generate_rcs_predictions(
                            pred_params, param_data_train, pod_coeffs_train,
                            phi_modes_train[:, :r], mean_rcs_train,
                            theta_values, phi_values, predict_dir)

                        print(f"成功预测 {len(pred_params)} 组参数的RCS数据")
                    else:
                        print("未能加载有效的预测参数")

                except Exception as e:
                    print(f"预测模式过程中发生错误: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"分析{freq_label}数据时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()