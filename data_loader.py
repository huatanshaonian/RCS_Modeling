"""
数据加载模块：用于读取参数文件和RCS数据
"""

import os
import numpy as np
import pandas as pd


def load_parameters(params_file):
    """
    加载设计参数数据 - 增强版，添加详细诊断，并在读取时将NaN填充为相邻值的均值

    参数:
    params_file: 参数CSV文件路径

    返回:
    param_data: numpy数组，包含参数值
    param_names: 参数名称列表
    """
    try:
        print(f"正在从 {params_file} 加载参数数据...")

        # 读取CSV文件
        df = pd.read_csv(params_file)

        # 输出CSV文件的基本信息
        print(f"CSV文件形状: {df.shape}")
        print(f"CSV文件列名: {list(df.columns)}")

        # 查看前几行数据
        print("\n参数数据前5行:")
        print(df.head(5))

        # 检查每列的数据类型
        print("\n参数数据类型:")
        print(df.dtypes)

        # 检查NaN值
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"\n发现 {nan_count} 个NaN值，使用插值方法填充")
            # 对每一列使用插值填充NaN，使用前后数据的平均值
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            # 检查是否还有NaN值（边缘可能无法插值）
            remaining_nan = df.isna().sum().sum()
            if remaining_nan > 0:
                print(f"插值后仍有 {remaining_nan} 个NaN值，使用前向/后向填充")
                df = df.fillna(method='ffill').fillna(method='bfill')

        # 检查每列的唯一值数量
        print("\n每列唯一值数量:")
        for col in df.columns:
            unique_values = df[col].unique()
            print(f"  {col}: {len(unique_values)} 个唯一值")

            # 检查是否为常数列
            if len(unique_values) <= 1:
                print(f"  警告: 列 '{col}' 可能是常数列!")

            # 输出前几个唯一值用于检查
            if len(unique_values) <= 5:
                print(f"  {col} 的所有唯一值: {unique_values}")
            else:
                print(f"  {col} 的前5个唯一值: {unique_values[:5]}")

        # 获取参数名称
        param_names = df.columns.tolist()

        # 转换为numpy数组
        param_data = df.values

        # 检查数据中是否包含Inf
        inf_count = np.isinf(param_data).sum()
        if inf_count > 0:
            print(f"\n警告: 参数数据中包含 {inf_count} 个Inf值")
            # 替换Inf值
            param_data = np.nan_to_num(param_data, nan=0.0, posinf=0.0, neginf=0.0)
            print("已将Inf值替换为0")

        print(f"\n成功加载了 {param_data.shape[0]} 个模型的 {param_data.shape[1]} 个参数")

        return param_data, param_names

    except Exception as e:
        print(f"加载参数数据时发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 返回空数据
        return np.array([]), []


def impute_rcs_nans(df, theta_col='Theta', phi_col='Phi', rcs_col='RCS(Total)'):
    """
    对RCS数据中的NaN值使用相邻点的平均值进行填充 - 优化版
    """
    # 检查是否有NaN值
    nan_count = df[rcs_col].isna().sum()
    if nan_count == 0:
        return df  # 如果没有NaN值，直接返回原始数据

    print(f"  发现 {nan_count} 个NaN值，使用相邻点的平均值填充")

    # 获取角度网格信息
    theta_values = np.sort(df[theta_col].unique())
    phi_values = np.sort(df[phi_col].unique())
    n_theta = len(theta_values)
    n_phi = len(phi_values)

    # 创建索引映射，便于后续查找
    theta_map = {theta: i for i, theta in enumerate(theta_values)}
    phi_map = {phi: i for i, phi in enumerate(phi_values)}

    # 将数据重塑为2D网格形式
    rcs_grid = np.full((n_phi, n_theta), np.nan)

    # 创建索引数组，加速数据填充
    theta_idx = np.array([theta_map[t] for t in df[theta_col]])
    phi_idx = np.array([phi_map[p] for p in df[phi_col]])

    # 填充已知数据 - 使用高效索引
    rcs_grid[phi_idx, theta_idx] = df[rcs_col].values

    # 创建用于卷积的核
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=float)

    # 存储原始NaN位置
    nan_mask = np.isnan(rcs_grid)
    original_nan_count = np.sum(nan_mask)

    # 使用卷积进行邻域平均 - 最多迭代5次
    for iteration in range(5):
        # 如果没有NaN值，提前结束
        if not np.any(nan_mask):
            break

        # 创建值累加器和计数累加器
        values_sum = np.zeros_like(rcs_grid)
        counts = np.zeros_like(rcs_grid)

        # 处理四个相邻方向
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # 创建移位后的数据和有效性掩码
            shifted_values = np.full_like(rcs_grid, np.nan)
            valid_i_min = max(0, -di)
            valid_i_max = min(n_phi, n_phi - di)
            valid_j_min = max(0, -dj)
            valid_j_max = min(n_theta, n_theta - dj)

            # 计算移位后的索引
            src_i_min = max(0, di)
            src_i_max = min(n_phi, n_phi + di)
            src_j_min = max(0, dj)
            src_j_max = min(n_theta, n_theta + dj)

            # 填充移位后的值
            shifted_values[valid_i_min:valid_i_max, valid_j_min:valid_j_max] = \
                rcs_grid[src_i_min:src_i_max, src_j_min:src_j_max]

            # 累加非NaN值和计数
            valid_values = ~np.isnan(shifted_values)
            values_sum += np.where(valid_values, shifted_values, 0)
            counts += valid_values.astype(int)

        # 计算平均值
        with np.errstate(divide='ignore', invalid='ignore'):
            averages = values_sum / counts

        # 只更新原始NaN位置
        fill_mask = nan_mask & ~np.isnan(averages)
        rcs_grid[fill_mask] = averages[fill_mask]

        # 更新NaN掩码
        nan_mask = np.isnan(rcs_grid)

        # 如果没有新填充的值，跳出循环
        current_nan_count = np.sum(nan_mask)
        if current_nan_count == original_nan_count - np.sum(fill_mask):
            print(f"  第 {iteration + 1} 次迭代后没有新的填充值")
            break

        print(f"  第 {iteration + 1} 次迭代: 填充了 {np.sum(fill_mask)} 个NaN值")
        original_nan_count = current_nan_count

    # 如果还有NaN，使用全局平均值填充
    remaining_nans = np.sum(nan_mask)
    if remaining_nans > 0:
        global_mean = np.nanmean(rcs_grid)
        print(f"  使用全局平均值 {global_mean:.6f} 填充剩余的 {remaining_nans} 个NaN")
        rcs_grid[nan_mask] = global_mean

    # 将处理后的网格数据转回DataFrame
    for idx, row in df.iterrows():
        i = phi_map[row[phi_col]]
        j = theta_map[row[theta_col]]
        df.loc[idx, rcs_col] = rcs_grid[i, j]

    return df


def load_rcs_data(rcs_dir, freq_suffix, num_models=100):
    """
    加载RCS数据，处理文件缺失的情况，使用相邻点的平均值填充NaN值，并添加详细的诊断输出

    参数:
    rcs_dir: RCS CSV文件目录
    freq_suffix: 频率后缀 (如 "1.5G" 或 "3G")
    num_models: 最大模型数量

    返回:
    rcs_data: 形状为 [available_models, num_angles] 的numpy数组
    theta_values: 唯一的theta角度值
    phi_values: 唯一的phi角度值
    available_models: 成功加载的模型索引列表
    """
    print(f"\n===== 加载 {freq_suffix} RCS数据 =====")
    print(f"搜索目录: {rcs_dir}")

    # 查找第一个可用文件以获取角度信息
    first_file = None
    print("检查文件可用性:")
    for i in range(1, num_models + 1):
        model_id = f"{i:03d}"
        test_file = os.path.join(rcs_dir, f"{model_id}_{freq_suffix}.csv")

        if os.path.exists(test_file):
            first_file = test_file
            print(f"  找到第一个可用文件: {test_file}")
            break
        else:
            if i <= 10:  # 只显示前10个缺失文件的信息，避免输出过多
                print(f"  文件不存在: {test_file}")

    if first_file is None:
        raise FileNotFoundError(f"在目录 {rcs_dir} 中找不到任何 *_{freq_suffix}.csv 文件")

    # 从第一个可用文件获取角度信息
    print(f"\n读取第一个可用文件以获取角度信息: {first_file}")
    first_df = pd.read_csv(first_file)

    # 如果有NaN值，使用相邻点的均值填充
    if first_df.isna().sum().sum() > 0:
        first_df = impute_rcs_nans(first_df)

    # 打印前5行数据用于检查
    print("\n第一个文件前5行数据:")
    print(first_df.head(5))

    # 打印列名以便调试
    print(f"\nCSV文件列名: {list(first_df.columns)}")

    # 确定角度列名
    theta_col = 'Theta'  # 使用确定的列名
    phi_col = 'Phi'  # 使用确定的列名

    # 确定RCS列名 - 使用 RCS(Total)
    rcs_col = 'RCS(Total)'

    print(f"\n使用列 '{theta_col}' 作为俯仰角，'{phi_col}' 作为偏航角")
    print(f"使用列 '{rcs_col}' 作为RCS值")

    # 获取唯一的theta和phi值
    theta_values = np.sort(first_df[theta_col].unique())
    phi_values = np.sort(first_df[phi_col].unique())

    # 输出角度信息
    print(f"\n找到 {len(theta_values)} 个唯一的俯仰角值:")
    print(f"  范围: {min(theta_values)}° 到 {max(theta_values)}°")
    if len(theta_values) <= 10:
        print(f"  值: {theta_values}")
    else:
        print(f"  前5个值: {theta_values[:5]}")
        print(f"  后5个值: {theta_values[-5:]}")

    print(f"\n找到 {len(phi_values)} 个唯一的偏航角值:")
    print(f"  范围: {min(phi_values)}° 到 {max(phi_values)}°")
    if len(phi_values) <= 10:
        print(f"  值: {phi_values}")
    else:
        print(f"  前5个值: {phi_values[:5]}")
        print(f"  后5个值: {phi_values[-5:]}")

    # 计算角度数量
    num_angles = len(theta_values) * len(phi_values)
    print(f"\n总共 {num_angles} 个角度组合 ({len(theta_values)} x {len(phi_values)})")

    # 查找所有可用文件
    available_models = []
    for i in range(1, num_models + 1):
        model_id = f"{i:03d}"
        file_path = os.path.join(rcs_dir, f"{model_id}_{freq_suffix}.csv")
        if os.path.exists(file_path):
            available_models.append(i)

    print(f"\n找到 {len(available_models)} 个可用的 {freq_suffix} 模型文件")
    print(f"模型编号: {available_models[:10]}... (显示前10个)")

    # 初始化RCS数据数组
    rcs_data = np.zeros((len(available_models), num_angles))

    # 循环读取每个可用模型的RCS数据
    print("\n开始读取RCS数据:")
    successful_models = 0

    for idx, model_num in enumerate(available_models):
        model_id = f"{model_num:03d}"
        file_path = os.path.join(rcs_dir, f"{model_id}_{freq_suffix}.csv")

        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 如果有NaN值，使用相邻点的均值填充
            if df.isna().sum().sum() > 0:
                df = impute_rcs_nans(df)

            # 确保包含所需列
            if rcs_col not in df.columns:
                print(f"  错误: 模型 {model_id} 的CSV文件缺少 '{rcs_col}' 列")
                print(f"  可用列: {list(df.columns)}")
                rcs_data[idx, :] = 0  # 使用0填充
                continue

            # 获取RCS值
            rcs_values = df[rcs_col].values

            # 检查并处理Inf值
            inf_count = np.isinf(rcs_values).sum()
            if inf_count > 0:
                print(f"  模型 {model_id} 包含 {inf_count} 个Inf值，将被替换")
                # 将Inf视为缺失值，也使用相邻点平均值填充
                inf_indices = np.where(np.isinf(rcs_values))[0]
                for idx in inf_indices:
                    df.loc[df.index[idx], rcs_col] = np.nan

                # 重新使用相邻点填充
                df = impute_rcs_nans(df)
                rcs_values = df[rcs_col].values

            # 验证数据
            if idx < 5:  # 为前5个模型显示详细信息
                min_val = np.min(rcs_values)
                max_val = np.max(rcs_values)
                mean_val = np.mean(rcs_values)
                std_val = np.std(rcs_values)

                print(f"\n模型 {model_id} 的RCS值统计:")
                print(f"  数据长度: {len(rcs_values)}")
                print(f"  范围: 最小值 = {min_val}, 最大值 = {max_val}")
                print(f"  平均值 = {mean_val}, 标准差 = {std_val}")

            # 确保数据与预期尺寸匹配
            if len(rcs_values) == num_angles:
                rcs_data[idx, :] = rcs_values
                successful_models += 1
                if idx < 5 or idx % 20 == 0:  # 只为部分模型显示信息，避免输出过多
                    print(f"  成功加载模型 {model_id} 的RCS数据")
            else:
                print(f"  警告: 模型 {model_id} 的RCS数据大小不匹配。预期 {num_angles}, 实际 {len(rcs_values)}")
                # 尝试重塑数据以匹配预期尺寸
                if len(rcs_values) > num_angles:
                    print(f"  截断数据以匹配预期尺寸")
                    rcs_data[idx, :] = rcs_values[:num_angles]
                else:
                    print(f"  数据不足，使用0填充缺失部分")
                    rcs_data[idx, :len(rcs_values)] = rcs_values
                    rcs_data[idx, len(rcs_values):] = 0

        except Exception as e:
            print(f"  错误: 无法加载模型 {model_id} 的RCS数据: {e}")
            import traceback
            traceback.print_exc()
            rcs_data[idx, :] = 0  # 使用0填充

    print(f"\n成功加载了 {successful_models}/{len(available_models)} 个模型的RCS数据")

    # 检查加载的数据
    nan_count = np.isnan(rcs_data).sum()
    inf_count = np.isinf(rcs_data).sum()

    print(f"\n加载的RCS数据统计:")
    print(f"  形状: {rcs_data.shape}")
    print(f"  NaN值数量: {nan_count}")
    print(f"  Inf值数量: {inf_count}")

    # 如果还有NaN或Inf，使用全局平均填充
    if nan_count > 0 or inf_count > 0:
        print(f"  对剩余的异常值进行最终处理")
        # 计算有效数据的全局平均值
        valid_mask = ~(np.isnan(rcs_data) | np.isinf(rcs_data))
        if np.any(valid_mask):
            global_mean = np.mean(rcs_data[valid_mask])
        else:
            global_mean = 0.0

        print(f"  使用全局平均值 {global_mean:.6f} 填充剩余的异常值")
        rcs_data = np.nan_to_num(rcs_data, nan=global_mean, posinf=global_mean, neginf=global_mean)

    # 检查数据的统计特性
    print(f"  数据范围: 最小值 = {np.min(rcs_data)}, 最大值 = {np.max(rcs_data)}")
    print(f"  数据统计: 均值 = {np.mean(rcs_data)}, 标准差 = {np.std(rcs_data)}")

    return rcs_data, theta_values, phi_values, available_models


def reshape_rcs_data(rcs_data, theta_values, phi_values):
    """
    将RCS数据从一维向量重塑为二维角度矩阵

    参数:
    rcs_data: 形状为 [num_angles] 的一维RCS数据
    theta_values: theta角度值
    phi_values: phi角度值

    返回:
    rcs_2d: 形状为 [len(theta_values), len(phi_values)] 的二维RCS数据
    """
    return rcs_data.reshape(len(theta_values), len(phi_values))