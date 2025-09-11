#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义损失函数
实现多尺度损失、物理约束损失、平滑正则化等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TVLoss(nn.Module):
    """
    Total Variation Loss (TV正则化)
    用于确保空间平滑性
    """
    
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x):
        """
        计算TV损失
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            tv_loss: TV损失值
        """
        batch_size, channels, height, width = x.shape
        
        # 计算水平和垂直方向的梯度
        h_var = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        w_var = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        
        # 总变分损失
        tv_loss = self.weight * (h_var + w_var) / (batch_size * channels * height * width)
        
        return tv_loss


class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失
    包括对称性检查等物理约束
    """
    
    def __init__(self, weight=1.0, symmetry_weight=1.0):
        super(PhysicsConstraintLoss, self).__init__()
        self.weight = weight
        self.symmetry_weight = symmetry_weight
    
    def forward(self, prediction, theta_values=None, phi_values=None):
        """
        计算物理约束损失
        
        Args:
            prediction: 预测的RCS图 [B, 1, 91, 91]
            theta_values: theta角度值 (可选)
            phi_values: phi角度值 (可选)
            
        Returns:
            physics_loss: 物理约束损失
        """
        batch_size, channels, height, width = prediction.shape
        
        total_loss = 0.0
        
        # 1. 对称性约束 (假设RCS在某些条件下应该对称)
        if self.symmetry_weight > 0:
            # 水平对称性检查
            flipped_h = torch.flip(prediction, dims=[3])  # 水平翻转
            symmetry_loss_h = F.mse_loss(prediction, flipped_h)
            
            # 垂直对称性检查
            flipped_v = torch.flip(prediction, dims=[2])  # 垂直翻转
            symmetry_loss_v = F.mse_loss(prediction, flipped_v)
            
            symmetry_loss = (symmetry_loss_h + symmetry_loss_v) / 2
            total_loss += self.symmetry_weight * symmetry_loss
        
        # 2. 值域约束 (RCS值应该在合理范围内)
        # 由于使用了Tanh激活，输出在[-1, 1]范围内，这已经是一种约束
        
        # 3. 边界平滑性约束
        # 边缘的梯度不应该过大
        edge_penalty = 0.0
        
        # 上下边缘
        top_edge = prediction[:, :, 0, :]
        bottom_edge = prediction[:, :, -1, :]
        edge_penalty += torch.mean(torch.abs(top_edge)) + torch.mean(torch.abs(bottom_edge))
        
        # 左右边缘
        left_edge = prediction[:, :, :, 0]
        right_edge = prediction[:, :, :, -1]
        edge_penalty += torch.mean(torch.abs(left_edge)) + torch.mean(torch.abs(right_edge))
        
        total_loss += 0.1 * edge_penalty
        
        return self.weight * total_loss


class MultiscaleLoss(nn.Module):
    """
    多尺度损失
    在不同分辨率上计算损失
    """
    
    def __init__(self, weights=None):
        super(MultiscaleLoss, self).__init__()
        
        # 默认权重：91x91(主要), 46x46(中等), 23x23(粗略)
        if weights is None:
            self.weights = {
                '91x91': 1.0,
                '46x46': 0.5,
                '23x23': 0.3
            }
        else:
            self.weights = weights
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        计算多尺度损失
        
        Args:
            predictions: 预测结果字典，包含不同尺度的输出
            targets: 目标RCS图 [B, 1, 91, 91]
            
        Returns:
            multiscale_loss: 多尺度损失
            loss_breakdown: 各尺度损失详情
        """
        total_loss = 0.0
        loss_breakdown = {}
        
        # 为不同尺度生成目标
        targets_dict = self._generate_multiscale_targets(targets)
        
        # 计算各尺度损失
        for scale, weight in self.weights.items():
            if scale in predictions and scale in targets_dict:
                pred = predictions[scale]
                target = targets_dict[scale]
                
                scale_loss = self.mse_loss(pred, target)
                weighted_loss = weight * scale_loss
                
                total_loss += weighted_loss
                loss_breakdown[scale] = {
                    'loss': scale_loss.item(),
                    'weight': weight,
                    'weighted_loss': weighted_loss.item()
                }
        
        return total_loss, loss_breakdown
    
    def _generate_multiscale_targets(self, targets):
        """
        生成多尺度目标
        
        Args:
            targets: 原始目标 [B, 1, 91, 91]
            
        Returns:
            targets_dict: 多尺度目标字典
        """
        targets_dict = {}
        
        # 91x91 (原始)
        targets_dict['91x91'] = targets
        
        # 46x46 (下采样)
        targets_dict['46x46'] = F.interpolate(
            targets, size=(46, 46), mode='bilinear', align_corners=False
        )
        
        # 23x23 (下采样)
        targets_dict['23x23'] = F.interpolate(
            targets, size=(23, 23), mode='bilinear', align_corners=False
        )
        
        return targets_dict


class CompositeLoss(nn.Module):
    """
    组合损失函数
    结合MSE、TV正则化、物理约束和多尺度损失
    """
    
    def __init__(self, 
                 lambda_mse=1.0,
                 lambda_smooth=0.01,
                 lambda_physics=0.05,
                 lambda_multiscale=0.1,
                 multiscale_weights=None):
        super(CompositeLoss, self).__init__()
        
        self.lambda_mse = lambda_mse
        self.lambda_smooth = lambda_smooth
        self.lambda_physics = lambda_physics
        self.lambda_multiscale = lambda_multiscale
        
        # 各种损失函数
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss(weight=1.0)
        self.physics_loss = PhysicsConstraintLoss(weight=1.0)
        self.multiscale_loss = MultiscaleLoss(weights=multiscale_weights)
    
    def forward(self, predictions, targets, intermediate_outputs=None):
        """
        计算组合损失
        
        Args:
            predictions: 最终预测结果 [B, 1, 91, 91]
            targets: 目标RCS图 [B, 1, 91, 91]
            intermediate_outputs: 中间输出字典 (用于多尺度损失)
            
        Returns:
            total_loss: 总损失
            loss_breakdown: 损失分解详情
        """
        loss_breakdown = {}
        
        # 1. MSE主损失
        L_mse = self.mse_loss(predictions, targets)
        weighted_L_mse = self.lambda_mse * L_mse
        loss_breakdown['mse'] = {
            'loss': L_mse.item(),
            'weight': self.lambda_mse,
            'weighted_loss': weighted_L_mse.item()
        }
        
        # 2. TV平滑损失
        L_smooth = self.tv_loss(predictions)
        weighted_L_smooth = self.lambda_smooth * L_smooth
        loss_breakdown['smooth'] = {
            'loss': L_smooth.item(),
            'weight': self.lambda_smooth,
            'weighted_loss': weighted_L_smooth.item()
        }
        
        # 3. 物理约束损失
        L_physics = self.physics_loss(predictions)
        weighted_L_physics = self.lambda_physics * L_physics
        loss_breakdown['physics'] = {
            'loss': L_physics.item(),
            'weight': self.lambda_physics,
            'weighted_loss': weighted_L_physics.item()
        }
        
        # 4. 多尺度损失
        L_multiscale = torch.tensor(0.0, device=predictions.device)
        multiscale_breakdown = {}
        
        if intermediate_outputs is not None and self.lambda_multiscale > 0:
            L_multiscale, multiscale_breakdown = self.multiscale_loss(
                intermediate_outputs, targets
            )
        
        weighted_L_multiscale = self.lambda_multiscale * L_multiscale
        loss_breakdown['multiscale'] = {
            'loss': L_multiscale.item() if isinstance(L_multiscale, torch.Tensor) else L_multiscale,
            'weight': self.lambda_multiscale,
            'weighted_loss': weighted_L_multiscale.item() if isinstance(weighted_L_multiscale, torch.Tensor) else weighted_L_multiscale,
            'breakdown': multiscale_breakdown
        }
        
        # 总损失
        total_loss = weighted_L_mse + weighted_L_smooth + weighted_L_physics + weighted_L_multiscale
        
        loss_breakdown['total'] = total_loss.item()
        
        return total_loss, loss_breakdown
    
    def update_weights(self, lambda_mse=None, lambda_smooth=None, 
                      lambda_physics=None, lambda_multiscale=None):
        """更新损失权重"""
        if lambda_mse is not None:
            self.lambda_mse = lambda_mse
        if lambda_smooth is not None:
            self.lambda_smooth = lambda_smooth
        if lambda_physics is not None:
            self.lambda_physics = lambda_physics
        if lambda_multiscale is not None:
            self.lambda_multiscale = lambda_multiscale


def test_losses():
    """测试损失函数"""
    print("测试自定义损失函数...")
    
    # 创建测试数据
    batch_size = 2
    predictions = torch.randn(batch_size, 1, 91, 91)
    targets = torch.randn(batch_size, 1, 91, 91)
    
    # 中间输出
    intermediate_outputs = {
        '91x91': predictions,
        '46x46': torch.randn(batch_size, 64, 46, 46),
        '23x23': torch.randn(batch_size, 128, 23, 23)
    }
    
    print(f"预测形状: {predictions.shape}")
    print(f"目标形状: {targets.shape}")
    
    # 测试各种损失
    print(f"\n=== 测试基础损失 ===")
    
    # 1. TV损失
    tv_loss = TVLoss()
    tv_value = tv_loss(predictions)
    print(f"TV损失: {tv_value.item():.6f}")
    
    # 2. 物理约束损失
    physics_loss = PhysicsConstraintLoss()
    physics_value = physics_loss(predictions)
    print(f"物理约束损失: {physics_value.item():.6f}")
    
    # 3. 多尺度损失
    multiscale_loss = MultiscaleLoss()
    ms_value, ms_breakdown = multiscale_loss(intermediate_outputs, targets)
    print(f"多尺度损失: {ms_value.item():.6f}")
    print("多尺度损失分解:")
    for scale, info in ms_breakdown.items():
        print(f"  {scale}: {info['weighted_loss']:.6f} (权重: {info['weight']})")
    
    # 4. 组合损失
    print(f"\n=== 测试组合损失 ===")
    composite_loss = CompositeLoss()
    total_loss, loss_breakdown = composite_loss(predictions, targets, intermediate_outputs)
    
    print(f"总损失: {total_loss.item():.6f}")
    print("损失分解:")
    for component, info in loss_breakdown.items():
        if component != 'total':
            if isinstance(info, dict) and 'weighted_loss' in info:
                print(f"  {component}: {info['weighted_loss']:.6f} (权重: {info['weight']})")
    
    # 测试权重更新
    print(f"\n=== 测试权重更新 ===")
    composite_loss.update_weights(lambda_smooth=0.02, lambda_physics=0.1)
    total_loss_new, _ = composite_loss(predictions, targets, intermediate_outputs)
    print(f"更新权重后的总损失: {total_loss_new.item():.6f}")


if __name__ == "__main__":
    test_losses()