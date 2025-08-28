import os
import pandas as pd
import numpy as np
import glob
import scipy.stats
import traceback


def to_dbsm(value):
    """
    将RCS值转换为dBsm (10*log10(value))
    如果值小于等于0，返回-100（作为下限值）
    """
    if value <= 0:
        return -100.0
    return 10 * np.log10(value)


def calculate_statistics(csv_file):
    """
    计算单个CSV文件的统计参数，处理NaN值

    Args:
        csv_file: CSV文件路径

    Returns:
        包含统计参数的字典
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 获取模型编号（文件名前3位）
        model_name = os.path.basename(csv_file)[:3]

        # 提取RCS(Total)数据，即第9列
        rcs_total = df.iloc[:, 8]  # 索引从0开始，所以第9列是索引8

        # 删除NaN值
        rcs_total = rcs_total.dropna()

        # 如果删除NaN后数据为空，则返回错误
        if len(rcs_total) == 0:
            print(f"警告: 文件 {csv_file} 中的RCS(Total)列全部为NaN值")
            return {'模型': model_name, 'error': 'No valid data after removing NaN'}

        # 计算dBsm值
        rcs_dbsm = rcs_total.apply(to_dbsm)

        # 找出最大值和最小值的索引
        max_idx = rcs_total.idxmax()
        min_idx = rcs_total.idxmin()

        # 获取最大值、最小值及其对应的θ和φ值
        max_value_dbsm = to_dbsm(rcs_total[max_idx])
        max_theta = df.iloc[max_idx, 0]  # 第1列是θ
        max_phi = df.iloc[max_idx, 1]  # 第2列是φ

        min_value_dbsm = to_dbsm(rcs_total[min_idx])
        min_theta = df.iloc[min_idx, 0]
        min_phi = df.iloc[min_idx, 1]

        # 计算基本统计量
        mean_value_dbsm = to_dbsm(rcs_total.mean())  # 转换为dBsm
        median_value_dbsm = to_dbsm(rcs_total.median())  # 中位数，转换为dBsm

        # 极差（最大dBsm值 - 最小dBsm值）
        range_value = max_value_dbsm - min_value_dbsm

        # 标准差
        std_dev = rcs_total.std()

        # 变异系数（标准差/均值）
        mean = rcs_total.mean()
        cv = std_dev / mean if mean != 0 else np.nan

        # 平滑度 - 使用相邻点差异的平均值
        # 确保有足够的数据点
        if len(rcs_total) > 1:
            # 先排序，确保数据点是连续的
            sorted_rcs = np.sort(rcs_total.values)
            diffs = np.abs(np.diff(sorted_rcs))
            smoothness = np.mean(diffs) if len(diffs) > 0 else 0
        else:
            smoothness = 0

        # 偏度系数 - 分布偏斜程度
        # 至少需要3个数据点才能计算偏度
        if len(rcs_total) >= 3:
            skewness = scipy.stats.skew(rcs_total)
        else:
            skewness = np.nan

        # 峰度系数 - 分布尖峰程度
        # 至少需要4个数据点才能计算峰度
        if len(rcs_total) >= 4:
            kurtosis = scipy.stats.kurtosis(rcs_total)
        else:
            kurtosis = np.nan

        print(f"  模型 {model_name} 统计完成 - 有效数据点: {len(rcs_total)}")

        # 返回所有统计量
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
            '极差': range_value,
            '标准差': std_dev,
            '变异系数': cv,
            '平滑度': smoothness,
            '偏度系数': skewness,
            '峰度系数': kurtosis
        }

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {str(e)}")
        traceback.print_exc()
        # 返回带有错误标记的字典
        return {'模型': model_name, 'error': str(e)}


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用正确的路径格式（使用原始字符串）
    csv_dir = r".\csv_output"

    print(f"正在查找CSV文件，目录: {csv_dir}")

    # 检查目录是否存在
    if not os.path.exists(csv_dir):
        print(f"错误: CSV目录不存在: {csv_dir}")
        # 尝试在脚本目录下查找csv_output文件夹
        alternate_dir = os.path.join(script_dir, "csv_output")
        print(f"尝试替代目录: {alternate_dir}")
        if os.path.exists(alternate_dir):
            print(f"找到替代目录，使用: {alternate_dir}")
            csv_dir = alternate_dir
        else:
            print("替代目录也不存在，无法继续")
            return

    # 查找所有CSV文件
    csv_files_3g = glob.glob(os.path.join(csv_dir, "*_3G.csv"))
    csv_files_15g = glob.glob(os.path.join(csv_dir, "*_1.5G.csv"))

    print(f"找到 {len(csv_files_3g)} 个3G CSV文件")
    print(f"找到 {len(csv_files_15g)} 个1.5G CSV文件")

    # 如果没有找到文件，列出目录内容帮助调试
    if len(csv_files_3g) == 0 and len(csv_files_15g) == 0:
        print("未找到任何CSV文件，打印目录内容以帮助调试:")
        try:
            files = os.listdir(csv_dir)
            for file in files:
                print(f"  - {file}")
        except Exception as e:
            print(f"无法列出目录内容: {str(e)}")
        return

    # 存储统计结果的列表
    stats_3g = []
    stats_15g = []

    # 处理3G文件
    for csv_file in csv_files_3g:
        print(f"处理 3G 文件: {os.path.basename(csv_file)}")
        stats = calculate_statistics(csv_file)
        if 'error' not in stats:
            stats_3g.append(stats)

    # 处理1.5G文件
    for csv_file in csv_files_15g:
        print(f"处理 1.5G 文件: {os.path.basename(csv_file)}")
        stats = calculate_statistics(csv_file)
        if 'error' not in stats:
            stats_15g.append(stats)

    # 创建DataFrame并按模型编号排序
    if stats_3g:
        df_3g = pd.DataFrame(stats_3g)
        df_3g = df_3g.sort_values(by='模型')

        # 检查是否所有列都有值
        null_counts = df_3g.isnull().sum()
        if null_counts.sum() > 0:
            print("\n警告: 3G统计结果中存在空值:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"  {col}: {count}个空值")

        # 保存到CSV
        output_file_3g = os.path.join(script_dir, "statistics_3G.csv")
        df_3g.to_csv(output_file_3g, index=False)
        print(f"已保存3G统计结果到: {output_file_3g}")
        print(f"生成的表格包含 {len(df_3g.columns)} 列")
    else:
        print("未生成3G统计结果")

    if stats_15g:
        df_15g = pd.DataFrame(stats_15g)
        df_15g = df_15g.sort_values(by='模型')

        # 检查是否所有列都有值
        null_counts = df_15g.isnull().sum()
        if null_counts.sum() > 0:
            print("\n警告: 1.5G统计结果中存在空值:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"  {col}: {count}个空值")

        # 保存到CSV
        output_file_15g = os.path.join(script_dir, "statistics_1.5G.csv")
        df_15g.to_csv(output_file_15g, index=False)
        print(f"已保存1.5G统计结果到: {output_file_15g}")
        print(f"生成的表格包含 {len(df_15g.columns)} 列")
    else:
        print("未生成1.5G统计结果")


def calculate_statistics_from_data(rcs_2d, theta_values, phi_values, model_name):
    """
    计算2D RCS数据矩阵的统计参数

    Args:
        rcs_2d: 2D RCS数据矩阵，形状为 [n_phi, n_theta]
        theta_values: theta角度值数组
        phi_values: phi角度值数组
        model_name: 模型名称或标识符

    Returns:
        包含统计参数的字典
    """
    try:
        # 将2D矩阵转换为1D数组
        rcs_total = rcs_2d.flatten()

        # 如果数据为空，则返回错误
        if len(rcs_total) == 0:
            print(f"警告: 模型 {model_name} 的RCS数据为空")
            return {'模型': model_name, 'error': 'No valid data'}

        # 检查是否有NaN或Inf
        if np.any(np.isnan(rcs_total)) or np.any(np.isinf(rcs_total)):
            print(f"警告: 模型 {model_name} 的RCS数据包含NaN或Inf，将被替换为0")
            rcs_total = np.nan_to_num(rcs_total, nan=0.0, posinf=0.0, neginf=0.0)

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
        mean_value_dbsm = np.mean(rcs_total)
        median_value_dbsm = np.median(rcs_total)

        # 极差
        range_value = max_value_dbsm - min_value_dbsm

        # 标准差
        std_dev = np.std(rcs_total)

        # 变异系数（标准差/均值）- 注意RCS可能为负值(dB单位)
        # 为避免除零，使用绝对值
        mean_abs = np.mean(np.abs(rcs_total))
        cv = std_dev / mean_abs if mean_abs != 0 else np.nan

        # 平滑度 - 计算相邻点的差异
        # 计算行方向和列方向的梯度
        grad_x = np.gradient(rcs_2d, axis=1)
        grad_y = np.gradient(rcs_2d, axis=0)
        # 梯度幅值
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        smoothness = np.mean(grad_mag)

        # 偏度系数
        skewness = scipy.stats.skew(rcs_total)

        # 峰度系数
        kurtosis = scipy.stats.kurtosis(rcs_total)

        # 返回所有统计量
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
            '极差': range_value,
            '标准差': std_dev,
            '变异系数': cv,
            '平滑度': smoothness,
            '偏度系数': skewness,
            '峰度系数': kurtosis
        }

    except Exception as e:
        print(f"处理模型 {model_name} 数据时出错: {str(e)}")
        traceback.print_exc()
        # 返回带有错误标记的字典
        return {'模型': model_name, 'error': str(e)}


if __name__ == "__main__":
    try:
        main()
        print("统计分析完成!")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()