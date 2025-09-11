#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FiLM (Feature-wise Linear Modulation) 层实现
用于在U-Net的每一层进行特征调制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    FiLM调制层
    执行 output = γ * feature + β 操作
    """
    
    def __init__(self, num_channels):
        super(FiLMLayer, self).__init__()
        self.num_channels = num_channels
        
    def forward(self, feature_map, gamma, beta):
        """
        FiLM调制
        
        Args:
            feature_map: 输入特征图 [B, C, H, W]
            gamma: 缩放参数 [B, C] 
            beta: 偏移参数 [B, C]
            
        Returns:
            modulated_features: 调制后的特征图 [B, C, H, W]
        """
        batch_size, channels, height, width = feature_map.shape
        
        # 验证输入维度
        if gamma.shape != (batch_size, channels):
            raise ValueError(f"Gamma shape {gamma.shape} doesn't match feature channels {(batch_size, channels)}")
        if beta.shape != (batch_size, channels):
            raise ValueError(f"Beta shape {beta.shape} doesn't match feature channels {(batch_size, channels)}")
        
        # 扩展γ和β的维度以匹配特征图 [B, C] → [B, C, 1, 1]
        gamma_expanded = gamma.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta_expanded = beta.unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        
        # FiLM调制：output = γ * feature + β
        modulated_features = gamma_expanded * feature_map + beta_expanded
        
        return modulated_features


class ConvFiLMBlock(nn.Module):
    """
    卷积+FiLM+激活的组合块
    用于U-Net中的标准卷积块
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1,
                 use_batchnorm=True, activation='relu'):
        super(ConvFiLMBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, padding=padding, stride=stride)
        
        # FiLM层
        self.film = FiLMLayer(out_channels)
        
        # 激活函数
        if activation == 'relu':
            self.activation1 = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=kernel_size, padding=padding, stride=1)
        
        # BatchNorm
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        
        # 第二个激活函数
        if activation == 'relu':
            self.activation2 = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, gamma, beta):
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, in_channels, H, W]
            gamma: FiLM缩放参数 [B, out_channels]
            beta: FiLM偏移参数 [B, out_channels]
            
        Returns:
            output: 输出特征图 [B, out_channels, H, W]
        """
        # 第一个卷积
        out = self.conv1(x)
        
        # FiLM调制
        out = self.film(out, gamma, beta)
        
        # 第一个激活
        out = self.activation1(out)
        
        # 第二个卷积
        out = self.conv2(out)
        
        # BatchNorm
        if self.use_batchnorm:
            out = self.batchnorm(out)
        
        # 第二个激活
        out = self.activation2(out)
        
        return out


class UpsampleConcatenate(nn.Module):
    """
    上采样并与跳跃连接concatenate的模块
    """
    
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(UpsampleConcatenate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x, skip_connection=None, target_size=None):
        """
        上采样并拼接跳跃连接
        
        Args:
            x: 待上采样的特征图 [B, C1, H, W]
            skip_connection: 跳跃连接特征图 [B, C2, H', W'] (可选)
            target_size: 目标尺寸 (H_target, W_target) (可选)
            
        Returns:
            concatenated: 拼接后的特征图 [B, C1+C2, H_target, W_target]
        """
        # 上采样
        if target_size is not None:
            upsampled = F.interpolate(x, size=target_size, 
                                    mode=self.mode, align_corners=self.align_corners)
        else:
            upsampled = F.interpolate(x, scale_factor=self.scale_factor, 
                                    mode=self.mode, align_corners=self.align_corners)
        
        # 如果有跳跃连接，进行拼接
        if skip_connection is not None:
            # 确保跳跃连接尺寸匹配
            if skip_connection.shape[2:] != upsampled.shape[2:]:
                skip_connection = F.interpolate(skip_connection, 
                                              size=upsampled.shape[2:],
                                              mode=self.mode, 
                                              align_corners=self.align_corners)
            
            concatenated = torch.cat([upsampled, skip_connection], dim=1)
        else:
            concatenated = upsampled
        
        return concatenated


class CenterCrop2D(nn.Module):
    """
    中心裁剪模块
    将特征图裁剪到指定尺寸
    """
    
    def __init__(self, target_height, target_width):
        super(CenterCrop2D, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
    
    def forward(self, x):
        """
        中心裁剪
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            cropped: 裁剪后的特征图 [B, C, target_height, target_width]
        """
        batch_size, channels, height, width = x.shape
        
        # 计算裁剪起始位置
        start_h = (height - self.target_height) // 2
        start_w = (width - self.target_width) // 2
        
        # 确保不会越界
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        
        end_h = start_h + self.target_height
        end_w = start_w + self.target_width
        
        # 裁剪
        cropped = x[:, :, start_h:end_h, start_w:end_w]
        
        return cropped


def test_film_layers():
    """测试FiLM层"""
    print("测试FiLM层...")
    
    batch_size = 2
    channels = 64
    height, width = 23, 23
    
    # 创建测试数据
    feature_map = torch.randn(batch_size, channels, height, width)
    gamma = torch.randn(batch_size, channels)
    beta = torch.randn(batch_size, channels)
    
    # 测试基础FiLM层
    film_layer = FiLMLayer(channels)
    modulated = film_layer(feature_map, gamma, beta)
    print(f"FiLM调制：{feature_map.shape} → {modulated.shape}")
    
    # 测试ConvFiLMBlock
    conv_film_block = ConvFiLMBlock(32, 64)
    input_features = torch.randn(batch_size, 32, height, width)
    output_features = conv_film_block(input_features, gamma, beta)
    print(f"ConvFiLM块：{input_features.shape} → {output_features.shape}")
    
    # 测试上采样拼接
    upsample_concat = UpsampleConcatenate(scale_factor=2)
    x = torch.randn(batch_size, 64, 11, 11)
    skip = torch.randn(batch_size, 32, 23, 23)
    upsampled = upsample_concat(x, skip)
    print(f"上采样拼接：{x.shape} + {skip.shape} → {upsampled.shape}")
    
    # 测试中心裁剪
    center_crop = CenterCrop2D(91, 91)
    x_large = torch.randn(batch_size, 64, 92, 92)
    cropped = center_crop(x_large)
    print(f"中心裁剪：{x_large.shape} → {cropped.shape}")


if __name__ == "__main__":
    test_film_layers()