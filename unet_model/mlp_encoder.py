#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP编码器模块
将9维设计参数编码为1024维潜在特征向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    MLP编码器：9维设计参数 → 1024维潜在特征
    架构：[9 → 128 → 256 → 512 → 1024 → 1024]
    """
    
    def __init__(self, input_dim=9, hidden_dims=[128, 256, 512, 1024], output_dim=1024, 
                 dropout_rate=0.1):
        super(MLPEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 构建层次结构
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 最终输出层
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        self.encoder = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入参数张量 [B, 9]
            
        Returns:
            encoded_features: 编码后的特征 [B, 1024]
        """
        # 输入验证
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape [B, {self.input_dim}], got {x.shape}")
        
        # 编码
        encoded_features = self.encoder(x)
        
        return encoded_features
    
    def get_feature_dims(self):
        """获取特征维度信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_params': sum(p.numel() for p in self.parameters())
        }


class DualPathProcessor(nn.Module):
    """
    双路径处理器
    将MLP输出分为两路：路径A生成初始特征图，路径B生成FiLM参数
    """
    
    def __init__(self, feature_dim=1024, 
                 feature_map_channels=32, feature_map_size=23,
                 num_film_layers=6):
        super(DualPathProcessor, self).__init__()
        
        self.feature_dim = feature_dim
        self.feature_map_channels = feature_map_channels
        self.feature_map_size = feature_map_size
        self.num_film_layers = num_film_layers
        
        # 路径A：生成初始特征图
        # 1024 → 32*23*23 = 16928
        self.feature_map_dim = feature_map_channels * feature_map_size * feature_map_size
        self.path_a = nn.Sequential(
            nn.Linear(feature_dim, self.feature_map_dim),
            nn.BatchNorm1d(self.feature_map_dim),
            nn.ReLU(inplace=True)
        )
        
        # 路径B：生成FiLM参数
        # 每个FiLM层需要γ和β参数，对应不同的通道数
        self.film_channels = [64, 128, 256, 128, 64, 32]  # 对应U-Net各层的通道数
        
        # 为每个FiLM层生成γ和β
        self.film_generators = nn.ModuleDict()
        for i, channels in enumerate(self.film_channels):
            self.film_generators[f'film_{i}'] = nn.Sequential(
                nn.Linear(feature_dim, channels * 2),  # γ和β
                nn.Tanh()  # 使用Tanh限制范围
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # FiLM参数特殊初始化：γ=1, β=0
        for name, module in self.film_generators.items():
            if isinstance(module[-1], nn.Linear):  # 最后一层
                # 前半部分初始化为1（γ），后半部分初始化为0（β）
                with torch.no_grad():
                    out_features = module[-1].out_features
                    mid = out_features // 2
                    module[-1].weight[:mid].fill_(0)  # γ对应的权重
                    module[-1].weight[mid:].fill_(0)  # β对应的权重
                    module[-1].bias[:mid].fill_(1)    # γ初始值为1
                    module[-1].bias[mid:].fill_(0)    # β初始值为0
    
    def forward(self, mlp_features):
        """
        前向传播
        
        Args:
            mlp_features: MLP编码后的特征 [B, 1024]
            
        Returns:
            initial_feature_map: 初始特征图 [B, 32, 23, 23]
            film_params: FiLM参数字典，包含每层的γ和β
        """
        batch_size = mlp_features.size(0)
        
        # 路径A：生成初始特征图
        feature_map_flat = self.path_a(mlp_features)  # [B, 16928]
        initial_feature_map = feature_map_flat.view(
            batch_size, 
            self.feature_map_channels, 
            self.feature_map_size, 
            self.feature_map_size
        )  # [B, 32, 23, 23]
        
        # 路径B：生成FiLM参数
        film_params = {}
        for i, channels in enumerate(self.film_channels):
            film_output = self.film_generators[f'film_{i}'](mlp_features)  # [B, channels*2]
            gamma = film_output[:, :channels]      # [B, channels] - γ参数
            beta = film_output[:, channels:]       # [B, channels] - β参数
            
            film_params[f'layer_{i}'] = {
                'gamma': gamma,
                'beta': beta
            }
        
        return initial_feature_map, film_params
    
    def get_film_info(self):
        """获取FiLM参数信息"""
        return {
            'num_layers': len(self.film_channels),
            'channels_per_layer': self.film_channels,
            'total_film_params': sum(c * 2 for c in self.film_channels)
        }


def test_mlp_encoder():
    """测试MLP编码器"""
    print("测试MLP编码器...")
    
    # 创建模型
    encoder = MLPEncoder()
    dual_processor = DualPathProcessor()
    
    # 创建测试数据
    batch_size = 4
    input_params = torch.randn(batch_size, 9)
    
    print(f"输入形状: {input_params.shape}")
    
    # 前向传播
    with torch.no_grad():
        # MLP编码
        mlp_features = encoder(input_params)
        print(f"MLP特征形状: {mlp_features.shape}")
        
        # 双路径处理
        initial_feature_map, film_params = dual_processor(mlp_features)
        print(f"初始特征图形状: {initial_feature_map.shape}")
        
        # 打印FiLM参数
        print("\nFiLM参数:")
        for layer_name, params in film_params.items():
            gamma_shape = params['gamma'].shape
            beta_shape = params['beta'].shape
            print(f"  {layer_name}: γ{gamma_shape}, β{beta_shape}")
        
        # 模型信息
        print(f"\nMLP编码器参数量: {encoder.get_feature_dims()['total_params']:,}")
        print(f"双路径处理器参数量: {sum(p.numel() for p in dual_processor.parameters()):,}")
        print(f"FiLM信息: {dual_processor.get_film_info()}")


if __name__ == "__main__":
    test_mlp_encoder()