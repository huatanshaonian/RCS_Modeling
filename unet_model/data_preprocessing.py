#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理和增强模块
用于RCS数据的加载、预处理和增强
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class RCSDataset(Dataset):
    """
    RCS数据集类
    """
    
    def __init__(self, design_params, rcs_data, transform=None, augment=None):
        """
        初始化数据集
        
        Args:
            design_params: 设计参数 [N, 9]
            rcs_data: RCS数据 [N, 91, 91]
            transform: 数据变换
            augment: 数据增强
        """
        self.design_params = torch.FloatTensor(design_params)
        self.rcs_data = torch.FloatTensor(rcs_data)
        self.transform = transform
        self.augment = augment
        
        # 验证数据形状
        assert len(self.design_params) == len(self.rcs_data), "参数和RCS数据数量不匹配"
        assert self.design_params.shape[1] == 9, f"设计参数维度应为9，实际为{self.design_params.shape[1]}"
        assert self.rcs_data.shape[1:] == (91, 91), f"RCS数据形状应为(91,91)，实际为{self.rcs_data.shape[1:]}"
    
    def __len__(self):
        return len(self.design_params)
    
    def __getitem__(self, idx):
        params = self.design_params[idx].clone()
        rcs = self.rcs_data[idx].clone()
        
        # 数据变换
        if self.transform:
            params, rcs = self.transform(params, rcs)
        
        # 数据增强
        if self.augment:
            params, rcs = self.augment(params, rcs)
        
        return params, rcs


class DataNormalizer:
    """
    数据归一化器
    """
    
    def __init__(self):
        self.param_scaler = StandardScaler()
        self.rcs_stats = {}
        self.fitted = False
    
    def fit(self, design_params, rcs_data):
        """
        拟合归一化参数
        
        Args:
            design_params: 设计参数 [N, 9]
            rcs_data: RCS数据 [N, 91, 91]
        """
        # 参数标准化
        self.param_scaler.fit(design_params)
        
        # RCS数据统计信息
        rcs_flat = rcs_data.reshape(-1)
        self.rcs_stats = {
            'mean': np.mean(rcs_flat),
            'std': np.std(rcs_flat),
            'min': np.min(rcs_flat),
            'max': np.max(rcs_flat)
        }
        
        self.fitted = True
        
        print(f"数据归一化器已拟合:")
        print(f"  参数: 均值={self.param_scaler.mean_}, 标准差={self.param_scaler.scale_}")
        print(f"  RCS: 均值={self.rcs_stats['mean']:.4f}, 标准差={self.rcs_stats['std']:.4f}")
        print(f"       范围=[{self.rcs_stats['min']:.4f}, {self.rcs_stats['max']:.4f}]")
    
    def transform_params(self, params):
        """归一化设计参数"""
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
        
        # 确保输入是2D数组
        if params.ndim == 1:
            params = params.reshape(1, -1)
        
        return self.param_scaler.transform(params)
    
    def transform_rcs(self, rcs_data):
        """归一化RCS数据到[-1, 1]"""
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
        
        # 使用tanh缩放到[-1, 1]
        normalized = 2 * (rcs_data - self.rcs_stats['min']) / (self.rcs_stats['max'] - self.rcs_stats['min']) - 1
        return normalized
    
    def inverse_transform_rcs(self, normalized_rcs):
        """反归一化RCS数据"""
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")
        
        # 从[-1, 1]反归一化
        original = (normalized_rcs + 1) / 2 * (self.rcs_stats['max'] - self.rcs_stats['min']) + self.rcs_stats['min']
        return original
    
    def __call__(self, params, rcs):
        """作为transform使用"""
        if isinstance(params, torch.Tensor):
            params_np = params.numpy()
            rcs_np = rcs.numpy()
            
            normalized_params = self.transform_params(params_np)
            normalized_rcs = self.transform_rcs(rcs_np)
            
            # 确保参数是1D的
            if normalized_params.ndim == 2 and normalized_params.shape[0] == 1:
                normalized_params = normalized_params.flatten()
            
            # 确保RCS是2D的 [91, 91]
            if normalized_rcs.ndim == 3 and normalized_rcs.shape[0] == 1:
                normalized_rcs = normalized_rcs.squeeze(0)
            
            return torch.FloatTensor(normalized_params), torch.FloatTensor(normalized_rcs)
        else:
            normalized_params = self.transform_params(params)
            normalized_rcs = self.transform_rcs(rcs)
            
            # 确保参数是1D的
            if normalized_params.ndim == 2 and normalized_params.shape[0] == 1:
                normalized_params = normalized_params.flatten()
            
            # 确保RCS是2D的 [91, 91]
            if normalized_rcs.ndim == 3 and normalized_rcs.shape[0] == 1:
                normalized_rcs = normalized_rcs.squeeze(0)
            
            return normalized_params, normalized_rcs


class DataAugmenter:
    """
    数据增强器
    """
    
    def __init__(self, 
                 noise_std=0.01,
                 mixup_alpha=0.2,
                 param_jitter_std=0.01,
                 enable_mixup=True,
                 enable_noise=True,
                 enable_param_jitter=True):
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.param_jitter_std = param_jitter_std
        self.enable_mixup = enable_mixup
        self.enable_noise = enable_noise
        self.enable_param_jitter = enable_param_jitter
    
    def add_noise(self, rcs):
        """添加高斯噪声"""
        if self.enable_noise and np.random.random() < 0.5:
            noise = torch.randn_like(rcs) * self.noise_std
            rcs = rcs + noise
        return rcs
    
    def param_jitter(self, params):
        """参数抖动"""
        if self.enable_param_jitter and np.random.random() < 0.3:
            jitter = torch.randn_like(params) * self.param_jitter_std
            params = params + jitter
        return params
    
    def mixup(self, params1, rcs1, params2, rcs2):
        """Mixup数据增强"""
        if self.enable_mixup and np.random.random() < 0.3:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_params = lam * params1 + (1 - lam) * params2
            mixed_rcs = lam * rcs1 + (1 - lam) * rcs2
            return mixed_params, mixed_rcs
        return params1, rcs1
    
    def __call__(self, params, rcs):
        """执行数据增强"""
        # 参数抖动
        params = self.param_jitter(params)
        
        # 添加噪声
        rcs = self.add_noise(rcs)
        
        return params, rcs


class RCSDataLoader:
    """
    RCS数据加载器 - 复用现有的data_loader.py函数，避免重复代码
    """
    
    def __init__(self, data_dir="../parameter", 
                 params_file="parameters_sorted.csv",
                 rcs_dir="csv_output"):
        self.data_dir = data_dir
        self.params_file = params_file
        self.rcs_dir = rcs_dir
        self.normalizer = DataNormalizer()
        
    def load_data(self, num_models=100, frequency="1.5G", verbose=True):
        """
        加载数据 - 复用现有的data_loader.py中的函数
        
        Args:
            num_models: 加载的模型数量
            frequency: 频率 ("1.5G" 或 "3G")
            verbose: 是否打印详细信息
            
        Returns:
            design_params: 设计参数 [N, 9]
            rcs_data: RCS数据 [N, 91, 91]
        """
        if verbose:
            print(f"开始加载数据...")
            print(f"  数据目录: {self.data_dir}")
            print(f"  参数文件: {self.params_file}")
            print(f"  RCS目录: {self.rcs_dir}")
            print(f"  模型数量: {num_models}")
            print(f"  频率: {frequency}")
        
        # 导入现有的数据加载函数
        import sys
        import os
        # 添加项目根目录到Python路径
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        try:
            from data_loader import load_parameters, load_rcs_data
        except ImportError as e:
            raise ImportError(f"无法导入现有的数据加载函数: {e}")
        
        try:
            # 1. 使用现有函数加载设计参数
            params_path = os.path.join(self.data_dir, self.params_file)
            param_data, param_names = load_parameters(params_path)
            
            # 只取前num_models个
            design_params = param_data[:num_models]
            
            if verbose:
                print(f"成功加载 {len(design_params)} 个模型的设计参数")
                print(f"设计参数形状: {design_params.shape}")
            
            # 2. 使用现有函数加载RCS数据
            rcs_data_dir = os.path.join(self.data_dir, self.rcs_dir)
            rcs_data_flat, theta_values, phi_values, available_models = load_rcs_data(
                rcs_data_dir, frequency, num_models
            )
            
            # 将RCS数据从[N, 8281]重塑为[N, 91, 91]
            num_loaded = len(available_models)
            rcs_data = rcs_data_flat.reshape(num_loaded, 91, 91)
            
            # 确保设计参数和RCS数据的模型数匹配
            # available_models包含成功加载的模型索引（1-based）
            # 转换为0-based索引来匹配design_params
            if len(available_models) < len(design_params):
                model_indices = [idx - 1 for idx in available_models]  # 转换为0-based
                design_params = design_params[model_indices]
                if verbose:
                    print(f"根据RCS数据可用性调整参数数量: {len(model_indices)} 个模型")
            
            # 转换RCS值为dB单位 (与传统POD/AE方法保持一致)
            rcs_data = 10 * np.log10(np.maximum(rcs_data, 1e-10))
            
            if verbose:
                print(f"成功加载 {num_loaded} 个模型")
                print(f"设计参数形状: {design_params.shape}")
                print(f"RCS数据形状: {rcs_data.shape}")
                print(f"RCS数据范围(dB): [{rcs_data.min():.4f}, {rcs_data.max():.4f}]")
                print(f"角度网格: {len(theta_values)}x{len(phi_values)} = {len(theta_values)*len(phi_values)} 个点")
            
            # 保存available_models供后续使用
            self.available_models = available_models
            
            return design_params, rcs_data
            
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {e}")
    
    def create_datasets(self, design_params, rcs_data, 
                       test_size=0.2, train_size=None, random_state=42,
                       apply_normalization=True,
                       apply_augmentation=True,
                       augment_params=None,
                       output_dir=None, save_indices=True):
        """
        创建训练和测试数据集
        
        Args:
            design_params: 设计参数
            rcs_data: RCS数据
            test_size: 测试集比例 (当train_size为None时使用)
            train_size: 训练集绝对大小 (优先级高于test_size，用于与AE方法统一)
            random_state: 随机种子
            apply_normalization: 是否应用归一化
            apply_augmentation: 是否对训练集应用数据增强
            augment_params: 数据增强参数
            output_dir: 输出目录 (用于保存模型索引对照表)
            save_indices: 是否保存训练集/测试集模型索引对照表
            
        Returns:
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            normalizer: 归一化器
        """
        # 分割数据 - 支持两种方式
        if train_size is not None:
            # 方式1: 使用绝对训练集大小 (与AE方法统一)
            num_models = len(design_params)
            if train_size > num_models:
                print(f"警告: 训练集大小 {train_size} 超过了可用模型数量 {num_models}")
                train_size = num_models
                print(f"调整训练集大小为: {train_size}")
            
            # 使用与AE相同的索引划分方式
            np.random.seed(random_state)
            indices = np.random.permutation(num_models)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            train_params = design_params[train_indices]
            test_params = design_params[test_indices]
            train_rcs = rcs_data[train_indices]
            test_rcs = rcs_data[test_indices]
            
            print(f"数据分割 (绝对大小模式):")
            print(f"  训练集: {len(train_params)} 样本")
            print(f"  测试集: {len(test_params)} 样本")
            
        else:
            # 方式2: 使用相对比例 (原有方式)
            train_params, test_params, train_rcs, test_rcs = train_test_split(
                design_params, rcs_data, test_size=test_size, random_state=random_state
            )
            
            print(f"数据分割 (比例模式):")
            print(f"  训练集: {len(train_params)} 样本 ({(1-test_size)*100:.1f}%)")
            print(f"  测试集: {len(test_params)} 样本 ({test_size*100:.1f}%)")
        
        
        # 创建归一化器
        normalizer = None
        transform = None
        
        if apply_normalization:
            normalizer = DataNormalizer()
            normalizer.fit(train_params, train_rcs)
            transform = normalizer
        
        # 创建数据增强器
        augmenter = None
        if apply_augmentation:
            if augment_params is None:
                augment_params = {}
            augmenter = DataAugmenter(**augment_params)
        
        # 创建数据集
        train_dataset = RCSDataset(
            train_params, train_rcs, 
            transform=transform, 
            augment=augmenter
        )
        
        test_dataset = RCSDataset(
            test_params, test_rcs, 
            transform=transform, 
            augment=None  # 测试集不应用增强
        )
        
        # 同时返回原始数据（未归一化的dB值）用于评估对比
        train_data_raw = {
            'params': train_params,
            'rcs': train_rcs,  # 原始dB数据
            'indices': train_indices
        }
        
        test_data_raw = {
            'params': test_params,
            'rcs': test_rcs,  # 原始dB数据
            'indices': test_indices
        }
        
        # 保存训练集和测试集的模型索引对照表 (与AE方法统一)
        if save_indices and output_dir is not None and hasattr(self, 'available_models'):
            self._save_model_indices(train_indices, test_indices, output_dir, train_size)
        
        return train_dataset, test_dataset, normalizer, train_data_raw, test_data_raw
    
    def _save_model_indices(self, train_indices, test_indices, output_dir, train_size):
        """
        保存训练集和测试集的模型索引对照表 (与AE方法统一)
        
        Args:
            train_indices: 训练集索引
            test_indices: 测试集索引  
            output_dir: 输出目录
            train_size: 训练集大小 (可能为None)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        available_models = self.available_models
        
        # 如果available_models不为None，转换为真实模型编号
        if available_models is not None:
            real_train_indices = [available_models[i] for i in train_indices]
            real_test_indices = [available_models[i] for i in test_indices]

            print(f"\n训练集真实模型编号 (共 {len(real_train_indices)} 个):")
            print(real_train_indices)

            # 保存真实模型编号到文件
            train_indices_file = os.path.join(output_dir, "train_models.txt")
            with open(train_indices_file, 'w') as f:
                f.write(f"训练集大小: {train_size if train_size is not None else len(train_indices)}\n")
                f.write("训练集模型编号:\n")
                for i, model_idx in enumerate(real_train_indices):
                    f.write(f"{i + 1}. {model_idx}\n")

            test_indices_file = os.path.join(output_dir, "test_models.txt")
            with open(test_indices_file, 'w') as f:
                f.write(f"测试集大小: {len(real_test_indices)}\n")
                f.write("测试集模型编号:\n")
                for i, model_idx in enumerate(real_test_indices):
                    f.write(f"{i + 1}. {model_idx}\n")

            print(f"训练集和测试集模型编号已保存到:\n  {train_indices_file}\n  {test_indices_file}")
        else:
            # 保存数组索引到文件
            train_indices_file = os.path.join(output_dir, "train_indices.txt")
            with open(train_indices_file, 'w') as f:
                f.write(f"训练集大小: {train_size if train_size is not None else len(train_indices)}\n")
                f.write("训练集索引:\n")
                for i, idx in enumerate(train_indices):
                    f.write(f"{i + 1}. {idx}\n")

            test_indices_file = os.path.join(output_dir, "test_indices.txt")
            with open(test_indices_file, 'w') as f:
                f.write(f"测试集大小: {len(test_indices)}\n")
                f.write("测试集索引:\n")
                for i, idx in enumerate(test_indices):
                    f.write(f"{i + 1}. {idx}\n")

            print(f"训练集和测试集索引已保存到:\n  {train_indices_file}\n  {test_indices_file}")

        # 保存CSV格式的对照表
        if available_models is not None:
            train_df = pd.DataFrame({
                'Index': train_indices,
                'Model_Number': [available_models[i] for i in train_indices]
            })
        else:
            train_df = pd.DataFrame({
                'Index': train_indices
            })
        train_df.to_csv(os.path.join(output_dir, "train_indices.csv"), index=False)

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
        test_df.to_csv(os.path.join(output_dir, "test_indices.csv"), index=False)
        
        print(f"[OK] CSV格式索引对照表已保存到: train_indices.csv, test_indices.csv")

    def create_dataloaders(self, train_dataset, test_dataset, 
                          batch_size=16, num_workers=0, shuffle=True):
        """
        创建数据加载器
        
        Args:
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            batch_size: 批大小
            num_workers: 工作进程数
            shuffle: 是否打乱数据
            
        Returns:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
        """
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"数据加载器创建完成:")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  测试批次数: {len(test_loader)}")
        print(f"  批大小: {batch_size}")
        
        return train_loader, test_loader


def test_data_preprocessing():
    """测试数据预处理"""
    print("测试数据预处理模块...")
    
    # 创建模拟数据
    num_samples = 50
    design_params = np.random.randn(num_samples, 9)
    rcs_data = np.random.randn(num_samples, 91, 91)
    
    print(f"模拟数据形状:")
    print(f"  设计参数: {design_params.shape}")
    print(f"  RCS数据: {rcs_data.shape}")
    
    # 测试归一化器
    print(f"\n=== 测试归一化器 ===")
    normalizer = DataNormalizer()
    normalizer.fit(design_params, rcs_data)
    
    norm_params, norm_rcs = normalizer(design_params[:5], rcs_data[:5])
    print(f"归一化后参数范围: [{norm_params.min():.4f}, {norm_params.max():.4f}]")
    print(f"归一化后RCS范围: [{norm_rcs.min():.4f}, {norm_rcs.max():.4f}]")
    
    # 测试数据增强器
    print(f"\n=== 测试数据增强器 ===")
    augmenter = DataAugmenter()
    
    params_tensor = torch.FloatTensor(design_params[:2])
    rcs_tensor = torch.FloatTensor(rcs_data[:2])
    
    aug_params, aug_rcs = augmenter(params_tensor[0], rcs_tensor[0])
    print(f"原始参数: {params_tensor[0][:3]}")
    print(f"增强参数: {aug_params[:3]}")
    
    # 测试数据集
    print(f"\n=== 测试数据集 ===")
    dataset = RCSDataset(design_params, rcs_data, transform=normalizer, augment=augmenter)
    
    # 获取一个样本
    params_sample, rcs_sample = dataset[0]
    print(f"数据集样本形状:")
    print(f"  参数: {params_sample.shape}")
    print(f"  RCS: {rcs_sample.shape}")
    
    # 测试数据加载器（如果有真实数据）
    data_dir = "../parameter"
    if os.path.exists(data_dir):
        print(f"\n=== 测试真实数据加载 ===")
        try:
            data_loader = RCSDataLoader(data_dir)
            real_params, real_rcs = data_loader.load_data(num_models=10, verbose=True)
            
            train_dataset, test_dataset, real_normalizer = data_loader.create_datasets(
                real_params, real_rcs, test_size=0.3
            )
            
            train_loader, test_loader = data_loader.create_dataloaders(
                train_dataset, test_dataset, batch_size=4
            )
            
            # 获取一个批次
            for batch_params, batch_rcs in train_loader:
                print(f"真实数据批次形状:")
                print(f"  参数: {batch_params.shape}")
                print(f"  RCS: {batch_rcs.shape}")
                break
                
        except Exception as e:
            print(f"加载真实数据失败: {e}")
    else:
        print(f"\n真实数据目录不存在: {data_dir}")


if __name__ == "__main__":
    test_data_preprocessing()