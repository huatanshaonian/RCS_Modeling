#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified U-Net架构实现
结合FiLM调制机制的U-Net网络，用于RCS预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .film_layer import ConvFiLMBlock, UpsampleConcatenate, CenterCrop2D
except ImportError:
    from film_layer import ConvFiLMBlock, UpsampleConcatenate, CenterCrop2D


class ModifiedUNet(nn.Module):
    """
    Modified U-Net with FiLM modulation
    
    输入: [B, 32, 23, 23] (来自MLP特征图)
    输出: [B, 1, 91, 91] (RCS强度图)
    """
    
    def __init__(self, input_channels=32, output_channels=1):
        super(ModifiedUNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # ================================
        # 编码器 (Encoder)
        # ================================
        
        # Layer E1: [B, 32, 23, 23] → [B, 64, 23, 23]
        self.encoder1 = ConvFiLMBlock(
            in_channels=32, 
            out_channels=64,
            kernel_size=3, 
            padding=1, 
            stride=1
        )
        
        # Layer E2: [B, 64, 23, 23] → [B, 128, 23, 23] → [B, 128, 11, 11]
        self.encoder2 = ConvFiLMBlock(
            in_channels=64, 
            out_channels=128,
            kernel_size=3, 
            padding=1, 
            stride=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ================================
        # 瓶颈层 (Bottleneck)
        # ================================
        
        # Bottleneck: [B, 128, 11, 11] → [B, 256, 11, 11]
        self.bottleneck = ConvFiLMBlock(
            in_channels=128, 
            out_channels=256,
            kernel_size=3, 
            padding=1, 
            stride=1
        )
        
        # ================================
        # 解码器 (Decoder)
        # ================================
        
        # 上采样和拼接工具
        self.upsample1 = UpsampleConcatenate(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = UpsampleConcatenate(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = UpsampleConcatenate(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 添加padding层用于D1
        self.pad_d1 = nn.ReflectionPad2d(1)  # 对称padding
        
        # Layer D1: [B, 256, 22, 22] → pad → [B, 256, 23, 23] → concat with skip2 → [B, 384, 23, 23] → [B, 128, 23, 23]
        self.decoder1 = ConvFiLMBlock(
            in_channels=256 + 128,  # 来自bottleneck的256 + skip2的128
            out_channels=128,
            kernel_size=3, 
            padding=1, 
            stride=1
        )
        
        # Layer D2: [B, 128, 46, 46] → concat with skip1 → [B, 192, 46, 46] → [B, 64, 46, 46]
        self.decoder2 = ConvFiLMBlock(
            in_channels=128 + 64,   # 来自decoder1的128 + skip1的64
            out_channels=64,
            kernel_size=3, 
            padding=1, 
            stride=1
        )
        
        # Layer D3: [B, 64, 92, 92] → crop to [B, 64, 91, 91] → [B, 32, 91, 91]
        self.center_crop = CenterCrop2D(target_height=91, target_width=91)
        self.decoder3 = ConvFiLMBlock(
            in_channels=64, 
            out_channels=32,
            kernel_size=3, 
            padding=1, 
            stride=1
        )
        
        # ================================
        # 输出层
        # ================================
        
        # [B, 32, 91, 91] → [B, 16, 91, 91] → [B, 1, 91, 91]
        self.output_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        self.output_relu = nn.ReLU(inplace=True)
        self.output_conv2 = nn.Conv2d(16, output_channels, kernel_size=1, stride=1)
        self.output_tanh = nn.Tanh()
        
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
    
    def forward(self, x, film_params):
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, 32, 23, 23]
            film_params: FiLM参数字典，包含6层的γ和β
            
        Returns:
            output: RCS预测结果 [B, 1, 91, 91]
            intermediate_outputs: 中间输出用于多尺度损失
        """
        # 验证输入
        if x.dim() != 4 or x.size(1) != self.input_channels:
            raise ValueError(f"Expected input shape [B, {self.input_channels}, 23, 23], got {x.shape}")
        
        # 存储中间输出用于多尺度损失
        intermediate_outputs = {}
        
        # ================================
        # 编码器路径
        # ================================
        
        # E1: [B, 32, 23, 23] → [B, 64, 23, 23]
        skip1 = self.encoder1(x, 
                             film_params['layer_0']['gamma'], 
                             film_params['layer_0']['beta'])
        
        # E2: [B, 64, 23, 23] → [B, 128, 23, 23] → [B, 128, 11, 11]
        skip2 = self.encoder2(skip1, 
                             film_params['layer_1']['gamma'], 
                             film_params['layer_1']['beta'])
        x_pooled = self.pool2(skip2)  # [B, 128, 11, 11]
        
        # ================================
        # 瓶颈层
        # ================================
        
        # Bottleneck: [B, 128, 11, 11] → [B, 256, 11, 11]
        bottleneck_out = self.bottleneck(x_pooled, 
                                        film_params['layer_2']['gamma'], 
                                        film_params['layer_2']['beta'])
        
        # ================================
        # 解码器路径
        # ================================
        
        # D1: Upsample → [B, 256, 22, 22] → pad → [B, 256, 23, 23]
        upsampled1 = self.upsample1(bottleneck_out, skip_connection=None)  # [B, 256, 22, 22]
        padded1 = self.pad_d1(upsampled1)  # [B, 256, 23, 23]
        
        # 拼接skip2: [B, 256, 23, 23] + [B, 128, 23, 23] → [B, 384, 23, 23]
        concat1 = torch.cat([padded1, skip2], dim=1)
        
        # D1卷积: [B, 384, 23, 23] → [B, 128, 23, 23]
        d1_out = self.decoder1(concat1, 
                              film_params['layer_3']['gamma'], 
                              film_params['layer_3']['beta'])
        
        # 保存23×23输出用于多尺度损失
        intermediate_outputs['23x23'] = d1_out
        
        # D2: Upsample → [B, 128, 46, 46]
        upsampled2 = self.upsample2(d1_out, skip_connection=None)  # [B, 128, 46, 46]
        
        # skip1需要上采样到46×46: [B, 64, 23, 23] → [B, 64, 46, 46]
        skip1_upsampled = F.interpolate(skip1, size=(46, 46), mode='bilinear', align_corners=False)
        
        # 拼接skip1: [B, 128, 46, 46] + [B, 64, 46, 46] → [B, 192, 46, 46]
        concat2 = torch.cat([upsampled2, skip1_upsampled], dim=1)
        
        # D2卷积: [B, 192, 46, 46] → [B, 64, 46, 46]
        d2_out = self.decoder2(concat2, 
                              film_params['layer_4']['gamma'], 
                              film_params['layer_4']['beta'])
        
        # 保存46×46输出用于多尺度损失
        intermediate_outputs['46x46'] = d2_out
        
        # D3: Upsample → [B, 64, 92, 92] → crop → [B, 64, 91, 91]
        upsampled3 = self.upsample3(d2_out, skip_connection=None)  # [B, 64, 92, 92]
        cropped3 = self.center_crop(upsampled3)  # [B, 64, 91, 91]
        
        # D3卷积: [B, 64, 91, 91] → [B, 32, 91, 91]
        d3_out = self.decoder3(cropped3, 
                              film_params['layer_5']['gamma'], 
                              film_params['layer_5']['beta'])
        
        # ================================
        # 输出层
        # ================================
        
        # [B, 32, 91, 91] → [B, 16, 91, 91] → [B, 1, 91, 91]
        out = self.output_conv1(d3_out)
        out = self.output_relu(out)
        out = self.output_conv2(out)
        output = self.output_tanh(out)
        
        # 保存最终输出
        intermediate_outputs['91x91'] = output
        
        return output, intermediate_outputs
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': f"[B, {self.input_channels}, 23, 23]",
            'output_shape': f"[B, {self.output_channels}, 91, 91]",
            'architecture': 'Modified U-Net with FiLM modulation'
        }


def test_modified_unet():
    """测试Modified U-Net"""
    print("测试Modified U-Net...")
    
    # 创建模型
    unet = ModifiedUNet()
    
    # 创建测试数据
    batch_size = 2
    input_features = torch.randn(batch_size, 32, 23, 23)
    
    # 创建模拟的FiLM参数
    film_channels = [64, 128, 256, 128, 64, 32]
    film_params = {}
    for i, channels in enumerate(film_channels):
        film_params[f'layer_{i}'] = {
            'gamma': torch.randn(batch_size, channels),
            'beta': torch.randn(batch_size, channels)
        }
    
    print(f"输入形状: {input_features.shape}")
    
    # 前向传播
    with torch.no_grad():
        output, intermediate_outputs = unet(input_features, film_params)
        
        print(f"输出形状: {output.shape}")
        print("\n中间输出:")
        for name, tensor in intermediate_outputs.items():
            print(f"  {name}: {tensor.shape}")
        
        # 模型信息
        model_info = unet.get_model_info()
        print(f"\n模型信息:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_modified_unet()