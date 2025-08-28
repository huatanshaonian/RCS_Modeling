#!/usr/bin/env python
"""
RCS数据可视化程序：用于绘制指定模型的雷达散射截面（RCS）数据

使用方法:
    python rcs_visualizer.py -m MODEL_NUMBER [选项]

示例:
    python rcs_visualizer.py -m 001
    python rcs_visualizer.py -m 001 -d ../parameter/csv_output -o ./rcs_plots

参数:
    -m, --model: 模型编号，如 001, 002 等
    -d, --data_dir: RCS数据目录，默认为 ../parameter/csv_output
    -o, --output_dir: 输出图像保存目录，默认为 ./rcs_plots
    -s, --save: 是否保存图像，默认为 False
    -f, --freq: 要绘制的频率，选项为 1.5G, 3G 或 both，默认为 both
    GUI模式：python rcs_visualizer.py -gui - 弹出文件选择对话框
球坐标系：python rcs_visualizer.py -m 001 -sphere - 使用球坐标系可视化
组合使用：python rcs_visualizer.py -gui -sphere -s - GUI模式+球坐标+保存图像
# GUI模式（会询问颜色范围）
python rcs_visualizer.py -gui

# 命令行模式with固定颜色范围
python rcs_visualizer.py -m 001 --vmin -50 --vmax 10

# 球坐标+固定颜色范围
python rcs_visualizer.py -m 001 -sphere --vmin -40 --vmax 5
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interpolate
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# 设置支持中文的字体，优先使用微软雅黑，然后是SimHei
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def select_files_gui():
    """
    使用GUI选择RCS数据文件

    返回:
    selected_files: 选中的文件路径列表
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 设置文件选择对话框
    file_types = [
        ('CSV files', '*.csv'),
        ('All files', '*.*')
    ]

    selected_files = filedialog.askopenfilenames(
        title="选择RCS数据文件",
        filetypes=file_types,
        multiple=True
    )

    root.destroy()
    return list(selected_files)

def extract_model_freq_from_filename(filepath):
    """
    从文件路径中提取模型编号和频率信息

    参数:
    filepath: 文件路径

    返回:
    model_id: 模型编号
    freq: 频率标识
    display_name: 用于显示的名称
    """
    filename = os.path.basename(filepath)
    # 移除文件扩展名
    name_part = os.path.splitext(filename)[0]

    try:
        # 尝试解析标准格式: 001_1.5G.csv 或 002_3G.csv
        parts = name_part.split('_')
        if len(parts) >= 2:
            model_id = parts[0]
            freq = parts[1]
            # 标准化频率标识
            if 'G' not in freq:
                freq = freq + 'G'  # 如果没有G后缀，添加上
            display_name = f"{model_id}_{freq}"
            return model_id, freq, display_name
        else:
            # 如果无法解析，使用文件名作为标识
            return name_part, "unknown", name_part
    except:
        # 解析失败，使用文件名
        return name_part, "unknown", name_part

def fill_nan_values(data):
    """
    使用周围点的插值方法填充NaN值

    参数:
    data: 二维numpy数组

    返回:
    填充NaN值后的数组
    """
    # 创建网格坐标
    height, width = data.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # 提取非NaN值的坐标和值
    mask = ~np.isnan(data)
    xdata = x[mask]
    ydata = y[mask]
    zdata = data[mask]

    # 如果所有值都是NaN，返回原数组
    if len(zdata) == 0:
        return data

    # 使用径向基函数(RBF)插值
    try:
        # 尝试RBF插值
        rbf = interpolate.Rbf(xdata, ydata, zdata, function='linear')
        filled_data = rbf(x, y)
    except Exception:
        # 如果RBF插值失败，则使用最近邻插值
        filled_data = interpolate.griddata((xdata, ydata), zdata, (x, y), method='nearest')

    return filled_data

def load_rcs_data(data_dir, model_id, freq_suffix):
    """
    加载指定模型和频率的RCS数据

    参数:
    data_dir: RCS数据目录
    model_id: 模型编号
    freq_suffix: 频率后缀

    返回:
    theta_values: Theta角度值
    phi_values: Phi角度值
    rcs_db: 转换为分贝的RCS值
    """
    file_path = os.path.join(data_dir, f"{model_id}_{freq_suffix}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    print(f"正在加载文件: {file_path}")

    # 读取CSV文件，处理NaN值
    df = pd.read_csv(file_path)

    # 确定列名
    theta_col = 'Theta'  # 俯仰角列
    phi_col = 'Phi'  # 偏航角列
    rcs_col = 'RCS(Total)'  # RCS值列

    # 检查必需的列是否存在
    required_cols = [theta_col, phi_col, rcs_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必需列: {col}")

    # 获取唯一角度值
    theta_values = np.sort(df[theta_col].unique())
    phi_values = np.sort(df[phi_col].unique())

    # 获取RCS值
    rcs_values = df[rcs_col].values

    # 处理负值和零值，使用最小可接受值
    min_acceptable_value = 1e-10
    rcs_values = np.maximum(rcs_values, min_acceptable_value)

    # 转换为分贝
    rcs_db = 10 * np.log10(rcs_values)

    # 重塑为二维矩阵（theta作为行，phi作为列）
    n_theta = len(theta_values)
    n_phi = len(phi_values)

    if len(rcs_db) != n_theta * n_phi:
        raise ValueError(f"RCS数据长度 ({len(rcs_db)}) 不匹配角度组合数 ({n_theta * n_phi})")

    # 重塑并转置以匹配网格
    rcs_db_2d = rcs_db.reshape(n_theta, n_phi).T

    # 填充NaN值
    rcs_db_2d = fill_nan_values(rcs_db_2d)

    return theta_values, phi_values, rcs_db_2d

def load_rcs_data_from_file(file_path):
    """
    直接从文件路径加载RCS数据（用于GUI模式）

    参数:
    file_path: CSV文件的完整路径

    返回:
    theta_values: Theta角度值
    phi_values: Phi角度值
    rcs_db: 转换为分贝的RCS值
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    print(f"正在加载文件: {file_path}")

    # 读取CSV文件，处理NaN值
    df = pd.read_csv(file_path)

    # 确定列名
    theta_col = 'Theta'  # 俯仰角列
    phi_col = 'Phi'  # 偏航角列
    rcs_col = 'RCS(Total)'  # RCS值列

    # 检查必需的列是否存在
    required_cols = [theta_col, phi_col, rcs_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必需列: {col}")

    # 获取唯一角度值
    theta_values = np.sort(df[theta_col].unique())
    phi_values = np.sort(df[phi_col].unique())

    # 获取RCS值
    rcs_values = df[rcs_col].values

    # 处理负值和零值，使用最小可接受值
    min_acceptable_value = 1e-10
    rcs_values = np.maximum(rcs_values, min_acceptable_value)

    # 转换为分贝
    rcs_db = 10 * np.log10(rcs_values)

    # 重塑为二维矩阵（theta作为行，phi作为列）
    n_theta = len(theta_values)
    n_phi = len(phi_values)

    if len(rcs_db) != n_theta * n_phi:
        raise ValueError(f"RCS数据长度 ({len(rcs_db)}) 不匹配角度组合数 ({n_theta * n_phi})")

    # 重塑并转置以匹配网格
    rcs_db_2d = rcs_db.reshape(n_theta, n_phi).T

    # 填充NaN值
    rcs_db_2d = fill_nan_values(rcs_db_2d)

    return theta_values, phi_values, rcs_db_2d

def visualize_rcs(theta_values, phi_values, rcs_db, model_id, freq, ax=None, cmap='jet',
                  show_colorbar=True, vmin=None, vmax=None):
    """
    可视化RCS数据

    参数:
    theta_values: Theta角度值
    phi_values: Phi角度值
    rcs_db: 分贝单位的RCS值
    model_id: 模型编号
    freq: 频率字符串
    ax: Matplotlib轴对象，如果为None则创建新的
    cmap: 颜色映射
    show_colorbar: 是否显示颜色条
    vmin: 颜色条最小值，如果为None则使用数据最小值
    vmax: 颜色条最大值，如果为None则使用数据最大值
    """
    # 创建角度网格
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)

    # 如果没有提供轴对象，创建新的
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # 确定颜色范围
    if vmin is None:
        vmin = np.min(rcs_db)
    if vmax is None:
        vmax = np.max(rcs_db)

    # 限制数据到指定范围（用于颜色映射）
    rcs_db_clipped = np.clip(rcs_db, vmin, vmax)

    # 绘制3D表面 - 确保正值向上，使用指定的颜色范围
    surf = ax.plot_surface(theta_grid, phi_grid, rcs_db, cmap=cmap,
                           linewidth=0, antialiased=True, alpha=0.8,
                           vmin=vmin, vmax=vmax)

    # 设置标签和标题
    ax.set_xlabel('俯仰角 θ (度)')
    ax.set_ylabel('偏航角 φ (度)')
    ax.set_zlabel('RCS (分贝)')
    ax.set_title(f'模型 {model_id} - {freq} RCS')

    # 设置角度范围
    ax.set_xlim(min(theta_values), max(theta_values))
    ax.set_ylim(min(phi_values), max(phi_values))

    # 调整视角，使较大的值朝上
    ax.view_init(elev=30, azim=-60)

    # 添加颜色条
    if show_colorbar:
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='RCS (分贝)')

    return ax


def visualize_rcs_spherical(theta_values, phi_values, rcs_db, model_id, freq, ax=None, cmap='jet',
                            show_colorbar=True, vmin=None, vmax=None):
    """
    在球坐标系下可视化RCS数据

    参数:
    theta_values: Theta角度值（极角，弧度制）
    phi_values: Phi角度值（方位角，弧度制）
    rcs_db: 分贝单位的RCS值作为径向距离
    model_id: 模型编号
    freq: 频率字符串
    ax: Matplotlib轴对象，如果为None则创建新的
    cmap: 颜色映射
    show_colorbar: 是否显示颜色条
    vmin: 颜色条最小值，如果为None则使用数据最小值
    vmax: 颜色条最大值，如果为None则使用数据最大值
    """
    # 将角度转换为弧度（如果输入是度数）
    theta_rad = np.deg2rad(theta_values) if np.max(theta_values) > 2 * np.pi else theta_values
    phi_rad = np.deg2rad(phi_values) if np.max(phi_values) > 2 * np.pi else phi_values

    # 创建角度网格
    theta_grid, phi_grid = np.meshgrid(theta_rad, phi_rad)

    # 确定颜色范围
    if vmin is None:
        vmin = np.min(rcs_db)
    if vmax is None:
        vmax = np.max(rcs_db)

    # 对RCS数据进行缩放，避免负值问题
    # 将dB值转换为正的径向距离
    rcs_min = np.min(rcs_db)
    rcs_normalized = rcs_db - rcs_min + 1  # 确保所有值为正

    # 转换为球坐标 - 修改坐标系映射
    r = rcs_normalized  # RCS作为径向距离

    # 标准球坐标转换
    x_sphere = r * np.sin(theta_grid) * np.cos(phi_grid)
    y_sphere = r * np.sin(theta_grid) * np.sin(phi_grid)
    z_sphere = r * np.cos(theta_grid)

    # 重新映射坐标轴：
    # X轴保持不变: x = x_sphere
    # Y轴改为Z轴且正方向相反: z = -y_sphere
    # Z轴改为Y轴: y = z_sphere
    x = x_sphere  # X轴保持不变
    y = z_sphere  # 原来的z变成y
    z = -y_sphere  # 原来的y变成-z（正方向相反）

    # 如果没有提供轴对象，创建新的
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # 创建颜色映射，使用指定的范围
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.cm.get_cmap(cmap)(norm(rcs_db))

    # 绘制3D球面
    surf = ax.plot_surface(x, y, z, facecolors=colors,
                           linewidth=0, antialiased=True, alpha=0.8)

    # 设置标签和标题 - 更新轴标签以反映坐标变换
    ax.set_xlabel('X (保持原球坐标X轴)')
    ax.set_ylabel('Y (原球坐标Z轴)')
    ax.set_zlabel('Z (原球坐标Y轴取负)')
    ax.set_title(f'模型 {model_id} - {freq} RCS (修改球坐标)')

    # 设置等比例坐标轴
    max_range = np.max(rcs_normalized) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # 调整视角 - 可能需要调整以适应新的坐标系
    ax.view_init(elev=20, azim=-60)

    # 添加颜色条（基于原始RCS dB值）
    if show_colorbar:
        # 创建一个标量映射用于颜色条
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(rcs_db)
        plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='RCS (分贝)')

    return ax

def get_colorbar_range_gui():
    """
    通过GUI获取颜色条范围设置

    返回:
    vmin: 最小值，None表示使用自动范围
    vmax: 最大值，None表示使用自动范围
    use_fixed_range: 是否使用固定范围
    """
    import tkinter as tk
    from tkinter import messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()

    # 询问是否要设置固定的颜色条范围
    use_fixed = messagebox.askyesno(
        "颜色条范围设置",
        "是否要设置固定的颜色条范围？\n\n"
        "选择'是'：手动设置最大值和最小值（适合对比多个图像）\n"
        "选择'否'：使用每个图像的自动范围"
    )

    vmin, vmax = None, None

    if use_fixed:
        try:
            # 获取最小值
            vmin = simpledialog.askfloat(
                "设置颜色条最小值",
                "请输入颜色条最小值 (dB)：",
                initialvalue=-50.0
            )

            # 获取最大值
            vmax = simpledialog.askfloat(
                "设置颜色条最大值",
                "请输入颜色条最大值 (dB)：",
                initialvalue=10.0
            )

            # 验证输入
            if vmin is not None and vmax is not None and vmin >= vmax:
                messagebox.showwarning("输入错误", "最小值必须小于最大值！\n将使用自动范围。")
                vmin, vmax = None, None

        except:
            messagebox.showwarning("输入错误", "输入值无效！\n将使用自动范围。")
            vmin, vmax = None, None

    root.destroy()
    return vmin, vmax, use_fixed

def plot_dual_rcs(model_id, data_dir, output_dir, save_fig=False, coordinate_system='cartesian'):
    """
    绘制1.5GHz和3GHz的RCS对比图

    参数:
    model_id: 模型编号
    data_dir: RCS数据目录
    output_dir: 输出图像保存目录
    save_fig: 是否保存图像
    coordinate_system: 坐标系类型，'cartesian'或'spherical'
    """
    # 加载两个频率的数据
    try:
        theta_1p5g, phi_1p5g, rcs_1p5g = load_rcs_data(data_dir, model_id, '1.5G')
        theta_3g, phi_3g, rcs_3g = load_rcs_data(data_dir, model_id, '3G')

        # 创建对比图
        fig = plt.figure(figsize=(20, 10))

        # 1.5GHz RCS图
        ax1 = fig.add_subplot(121, projection='3d')
        if coordinate_system == 'spherical':
            visualize_rcs_spherical(theta_1p5g, phi_1p5g, rcs_1p5g, model_id, '1.5GHz', ax1, show_colorbar=True)
        else:
            visualize_rcs(theta_1p5g, phi_1p5g, rcs_1p5g, model_id, '1.5GHz', ax1, show_colorbar=True)

        # 3GHz RCS图
        ax2 = fig.add_subplot(122, projection='3d')
        if coordinate_system == 'spherical':
            visualize_rcs_spherical(theta_3g, phi_3g, rcs_3g, model_id, '3GHz', ax2, show_colorbar=True)
        else:
            visualize_rcs(theta_3g, phi_3g, rcs_3g, model_id, '3GHz', ax2, show_colorbar=True)

        # 添加总标题
        coord_name = '球坐标' if coordinate_system == 'spherical' else '直角坐标'
        plt.suptitle(f'模型 {model_id} RCS 对比 (1.5GHz vs 3GHz) - {coord_name}系', fontsize=16)

        plt.tight_layout()

        # 保存图像
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            coord_suffix = '_spherical' if coordinate_system == 'spherical' else ''
            output_file = os.path.join(output_dir, f'rcs_模型_{model_id}{coord_suffix}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {output_file}")

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"绘制RCS数据时出错: {e}")
        import traceback
        traceback.print_exc()

def plot_single_freq_rcs(model_id, data_dir, output_dir, freq, save_fig=False, coordinate_system='cartesian'):
    """
    绘制单一频率的RCS图

    参数:
    model_id: 模型编号
    data_dir: RCS数据目录
    output_dir: 输出图像保存目录
    freq: 频率后缀（'1.5G'或'3G'）
    save_fig: 是否保存图像
    coordinate_system: 坐标系类型，'cartesian'或'spherical'
    """
    try:
        theta, phi, rcs = load_rcs_data(data_dir, model_id, freq)

        # 创建图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制RCS
        freq_label = '1.5GHz' if freq == '1.5G' else '3GHz'
        if coordinate_system == 'spherical':
            visualize_rcs_spherical(theta, phi, rcs, model_id, freq_label, ax)
        else:
            visualize_rcs(theta, phi, rcs, model_id, freq_label, ax)

        plt.tight_layout()

        # 保存图像
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            coord_suffix = '_spherical' if coordinate_system == 'spherical' else ''
            output_file = os.path.join(output_dir, f'rcs_模型_{model_id}_{freq}{coord_suffix}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {output_file}")

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"绘制{freq}频率的RCS数据时出错: {e}")
        import traceback
        traceback.print_exc()

def plot_paired_files(file1_path, file2_path, model_id, output_dir, save_fig=False, coordinate_system='cartesian'):
    """
    为两个配对的CSV文件创建对比可视化

    参数:
    file1_path: 第一个CSV文件路径
    file2_path: 第二个CSV文件路径
    model_id: 模型编号
    output_dir: 输出图像保存目录
    save_fig: 是否保存图像
    coordinate_system: 坐标系类型，'cartesian'或'spherical'
    """
    try:
        # 加载两个文件的数据
        theta1, phi1, rcs1 = load_rcs_data_from_file(file1_path)
        theta2, phi2, rcs2 = load_rcs_data_from_file(file2_path)

        # 从文件名获取频率信息
        _, freq1, name1 = extract_model_freq_from_filename(file1_path)
        _, freq2, name2 = extract_model_freq_from_filename(file2_path)

        # 创建对比图
        fig = plt.figure(figsize=(20, 10))

        # 第一个文件的RCS图
        ax1 = fig.add_subplot(121, projection='3d')
        if coordinate_system == 'spherical':
            visualize_rcs_spherical(theta1, phi1, rcs1, model_id, freq1, ax1, show_colorbar=True)
        else:
            visualize_rcs(theta1, phi1, rcs1, model_id, freq1, ax1, show_colorbar=True)

        # 第二个文件的RCS图
        ax2 = fig.add_subplot(122, projection='3d')
        if coordinate_system == 'spherical':
            visualize_rcs_spherical(theta2, phi2, rcs2, model_id, freq2, ax2, show_colorbar=True)
        else:
            visualize_rcs(theta2, phi2, rcs2, model_id, freq2, ax2, show_colorbar=True)

        # 添加总标题
        coord_name = '球坐标' if coordinate_system == 'spherical' else '直角坐标'
        plt.suptitle(f'模型 {model_id} RCS 对比 ({freq1} vs {freq2}) - {coord_name}系', fontsize=16)

        plt.tight_layout()

        # 保存图像
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            coord_suffix = '_spherical' if coordinate_system == 'spherical' else ''
            output_file = os.path.join(output_dir, f'rcs_模型_{model_id}_对比{coord_suffix}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {output_file}")

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"处理配对文件时出错: {e}")
        import traceback
        traceback.print_exc()
        # 如果配对处理失败，分别处理两个文件
        plot_single_file(file1_path, output_dir, save_fig, coordinate_system)
        plot_single_file(file2_path, output_dir, save_fig, coordinate_system)


def plot_from_files_gui(selected_files, output_dir, save_fig=False, coordinate_system='cartesian'):
    """
    从GUI选择的文件进行RCS可视化

    参数:
    selected_files: 选中的文件路径列表
    output_dir: 输出图像保存目录
    save_fig: 是否保存图像
    coordinate_system: 坐标系类型，'cartesian'或'spherical'
    """
    if not selected_files:
        print("未选择任何文件")
        return

    print(f"选择了 {len(selected_files)} 个文件进行可视化")

    # 获取颜色条范围设置
    vmin, vmax, use_fixed_range = get_colorbar_range_gui()

    if use_fixed_range and vmin is not None and vmax is not None:
        print(f"使用固定颜色条范围: [{vmin}, {vmax}] dB")
    else:
        print("使用自动颜色条范围")
        vmin, vmax = None, None

    # 存储所有图形对象
    figures = []

    # 为每个文件创建独立的可视化窗口
    for i, filepath in enumerate(selected_files):
        try:
            # 为每个窗口设置不同的标题
            _, _, display_name = extract_model_freq_from_filename(filepath)
            window_title = f"RCS可视化 [{i + 1}/{len(selected_files)}] - {display_name}"

            fig = plot_single_file(filepath, output_dir, save_fig, coordinate_system,
                                   vmin, vmax, window_title)
            if fig:
                figures.append(fig)

        except Exception as e:
            print(f"处理文件 {filepath} 时出错: {e}")

    if figures:
        print(f"成功创建了 {len(figures)} 个可视化窗口")
        print("提示：所有窗口都是独立的，您可以同时查看和对比不同的RCS数据")

        # 等待用户关闭所有窗口
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "可视化完成",
            f"已创建 {len(figures)} 个可视化窗口。\n\n"
            "所有窗口都是独立的，您可以：\n"
            "• 同时查看多个RCS数据进行对比\n"
            "• 分别调整每个窗口的视角\n"
            "• 关闭不需要的窗口\n\n"
            "点击确定继续..."
        )
        root.destroy()

        # 阻塞直到所有窗口关闭
        plt.show()
    else:
        print("没有成功创建任何可视化窗口")

def plot_single_file(file_path, output_dir, save_fig=False, coordinate_system='cartesian',
                     vmin=None, vmax=None, window_title=None):
    """
    为单个CSV文件创建RCS可视化

    参数:
    file_path: CSV文件路径
    output_dir: 输出图像保存目录
    save_fig: 是否保存图像
    coordinate_system: 坐标系类型，'cartesian'或'spherical'
    vmin: 颜色条最小值
    vmax: 颜色条最大值
    window_title: 窗口标题
    """
    try:
        # 从文件路径提取信息
        model_id, freq, display_name = extract_model_freq_from_filename(file_path)

        print(f"为文件 {display_name} 创建可视化")

        # 直接从文件加载数据
        theta, phi, rcs = load_rcs_data_from_file(file_path)

        # 创建独立的图形窗口
        fig = plt.figure(figsize=(12, 10))
        if window_title:
            fig.canvas.manager.set_window_title(window_title)
        else:
            fig.canvas.manager.set_window_title(f"RCS可视化 - {display_name}")

        ax = fig.add_subplot(111, projection='3d')

        # 绘制RCS
        if coordinate_system == 'spherical':
            visualize_rcs_spherical(theta, phi, rcs, display_name, freq, ax,
                                    vmin=vmin, vmax=vmax)
        else:
            visualize_rcs(theta, phi, rcs, display_name, freq, ax,
                          vmin=vmin, vmax=vmax)

        plt.tight_layout()

        # 保存图像
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            coord_suffix = '_spherical' if coordinate_system == 'spherical' else ''
            range_suffix = f'_range_{vmin}_{vmax}' if vmin is not None and vmax is not None else ''
            # 使用安全的文件名
            safe_filename = display_name.replace('/', '_').replace('\\', '_')
            output_file = os.path.join(output_dir, f'rcs_{safe_filename}{coord_suffix}{range_suffix}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {output_file}")

        # 显示图像（不阻塞，允许多个窗口同时显示）
        plt.show(block=False)

        # 返回图形对象，便于后续操作
        return fig

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RCS数据可视化程序")

    parser.add_argument("-m", "--model", type=str,
                        help="模型编号，如 001, 002 等")

    parser.add_argument("-d", "--data_dir", type=str, default="../parameter/csv_output",
                        help="RCS数据目录，默认为 ../parameter/csv_output")

    parser.add_argument("-o", "--output_dir", type=str, default="./rcs_plots",
                        help="输出图像保存目录，默认为 ./rcs_plots")

    parser.add_argument("-s", "--save", action="store_true",
                        help="是否保存图像，默认为 False")

    parser.add_argument("-f", "--freq", type=str, choices=["1.5G", "3G", "both"], default="both",
                        help="要绘制的频率，选项为 1.5G, 3G 或 both，默认为 both")

    parser.add_argument("-gui", "--gui", action="store_true",
                        help="启用GUI模式，通过文件选择对话框选择要可视化的文件")

    parser.add_argument("-sphere", "--sphere", action="store_true",
                        help="使用球坐标系进行可视化，默认为直角坐标系")

    parser.add_argument("--vmin", type=float, default=None,
                        help="颜色条最小值 (dB)")

    parser.add_argument("--vmax", type=float, default=None,
                        help="颜色条最大值 (dB)")

    args = parser.parse_args()

    # 确定坐标系类型
    coordinate_system = 'spherical' if args.sphere else 'cartesian'

    # 验证颜色范围参数
    vmin, vmax = args.vmin, args.vmax
    if vmin is not None and vmax is not None and vmin >= vmax:
        print("警告: 最小值必须小于最大值，将使用自动范围")
        vmin, vmax = None, None

    # GUI模式
    if args.gui:
        print("启动GUI文件选择模式...")
        selected_files = select_files_gui()
        if selected_files:
            plot_from_files_gui(selected_files, args.output_dir, args.save, coordinate_system)
        else:
            print("未选择任何文件，程序退出")
        return

    # 命令行模式（原有功能）
    if not args.model:
        print("错误: 在命令行模式下必须指定模型编号")
        return

    # 确保模型编号为三位数格式
    model_id = args.model
    if len(model_id) < 3:
        model_id = model_id.zfill(3)  # 补零到三位数

    coord_name = '球坐标' if coordinate_system == 'spherical' else '直角坐标'
    range_info = f" (颜色范围: {vmin}-{vmax} dB)" if vmin is not None and vmax is not None else ""
    print(f"开始绘制模型 {model_id} 的RCS数据（{coord_name}系）{range_info}...")

    # 在命令行模式的绘图函数中也传递颜色范围参数
    # 这里需要相应修改 plot_dual_rcs 和 plot_single_freq_rcs 函数
    # 根据频率参数调用相应的绘图函数
    if args.freq == "both":
        plot_dual_rcs(model_id, args.data_dir, args.output_dir, args.save, coordinate_system, vmin, vmax)
    elif args.freq in ["1.5G", "3G"]:
        plot_single_freq_rcs(model_id, args.data_dir, args.output_dir, args.freq, args.save, coordinate_system, vmin,
                             vmax)
    else:
        print(f"错误: 不支持的频率选项 '{args.freq}'")

    print("绘图完成")

if __name__ == "__main__":
    main()