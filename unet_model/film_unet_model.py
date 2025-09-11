#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的FiLM-UNet模型
整合MLP编码器、FiLM调制和Modified U-Net的完整架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mlp_encoder import MLPEncoder, DualPathProcessor
    from modified_unet import ModifiedUNet
except ImportError:
    from mlp_encoder import MLPEncoder, DualPathProcessor
    from modified_unet import ModifiedUNet


class FiLMUNetModel(nn.Module):
    """
    完整的FiLM-UNet模型
    
    架构流程：
    1. 9维设计参数 → MLP编码器 → 1024维特征
    2. 1024维特征 → 双路径处理器 → (初始特征图, FiLM参数)
    3. (初始特征图, FiLM参数) → Modified U-Net → 91×91 RCS预测
    """
    
    def __init__(self, 
                 input_param_dim=9,
                 mlp_hidden_dims=[128, 256, 512, 1024],
                 mlp_output_dim=1024,
                 mlp_dropout_rate=0.1,
                 feature_map_channels=32,
                 feature_map_size=23,
                 output_channels=1):
        super(FiLMUNetModel, self).__init__()
        
        self.input_param_dim = input_param_dim
        self.output_channels = output_channels
        
        # ================================
        # 组件1: MLP编码器
        # ================================
        self.mlp_encoder = MLPEncoder(
            input_dim=input_param_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=mlp_output_dim,
            dropout_rate=mlp_dropout_rate
        )
        
        # ================================
        # 组件2: 双路径处理器
        # ================================
        self.dual_path_processor = DualPathProcessor(
            feature_dim=mlp_output_dim,
            feature_map_channels=feature_map_channels,
            feature_map_size=feature_map_size,
            num_film_layers=6
        )
        
        # ================================
        # 组件3: Modified U-Net
        # ================================
        self.unet = ModifiedUNet(
            input_channels=feature_map_channels,
            output_channels=output_channels
        )
        
        # 模型信息
        self.model_info = self._compute_model_info()
    
    def _compute_model_info(self):
        """计算模型信息"""
        mlp_params = sum(p.numel() for p in self.mlp_encoder.parameters())
        dual_path_params = sum(p.numel() for p in self.dual_path_processor.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        total_params = mlp_params + dual_path_params + unet_params
        
        return {
            'mlp_encoder_params': mlp_params,
            'dual_path_processor_params': dual_path_params,
            'unet_params': unet_params,
            'total_params': total_params,
            'input_dim': self.input_param_dim,
            'output_shape': f"[B, {self.output_channels}, 91, 91]"
        }
    
    def forward(self, design_params):
        """
        前向传播
        
        Args:
            design_params: 设计参数 [B, 9]
            
        Returns:
            output: RCS预测结果 [B, 1, 91, 91]
            intermediate_outputs: 中间输出字典，包含多尺度特征
            debug_info: 调试信息（可选）
        """
        # 验证输入
        if design_params.dim() != 2 or design_params.size(1) != self.input_param_dim:
            raise ValueError(f"Expected input shape [B, {self.input_param_dim}], got {design_params.shape}")
        
        # ================================
        # 步骤1: MLP编码
        # ================================
        mlp_features = self.mlp_encoder(design_params)  # [B, 1024]
        
        # ================================
        # 步骤2: 双路径处理
        # ================================
        initial_feature_map, film_params = self.dual_path_processor(mlp_features)
        # initial_feature_map: [B, 32, 23, 23]
        # film_params: 包含6层FiLM参数的字典
        
        # ================================
        # 步骤3: U-Net预测
        # ================================
        output, intermediate_outputs = self.unet(initial_feature_map, film_params)
        # output: [B, 1, 91, 91]
        # intermediate_outputs: 多尺度输出字典
        
        # 添加调试信息
        debug_info = {
            'mlp_features_shape': mlp_features.shape,
            'initial_feature_map_shape': initial_feature_map.shape,
            'film_layers': len(film_params),
            'output_shape': output.shape
        }
        
        return output, intermediate_outputs, debug_info
    
    def predict(self, design_params):
        """
        简化的预测接口，只返回最终输出
        
        Args:
            design_params: 设计参数 [B, 9] 或 numpy array
            
        Returns:
            rcs_prediction: RCS预测结果 [B, 1, 91, 91]
        """
        # 转换为tensor
        if not isinstance(design_params, torch.Tensor):
            design_params = torch.FloatTensor(design_params)
        
        # 确保是2D tensor
        if design_params.dim() == 1:
            design_params = design_params.unsqueeze(0)
        
        with torch.no_grad():
            output, _, _ = self.forward(design_params)
        
        return output
    
    def get_model_summary(self):
        """获取详细的模型摘要"""
        summary = {
            'architecture': 'FiLM-UNet for RCS Prediction',
            'components': {
                'mlp_encoder': {
                    'input_dim': self.input_param_dim,
                    'output_dim': 1024,
                    'parameters': self.model_info['mlp_encoder_params']
                },
                'dual_path_processor': {
                    'feature_map_output': '[B, 32, 23, 23]',
                    'film_layers': 6,
                    'parameters': self.model_info['dual_path_processor_params']
                },
                'modified_unet': {
                    'input_shape': '[B, 32, 23, 23]',
                    'output_shape': f"[B, {self.output_channels}, 91, 91]",
                    'parameters': self.model_info['unet_params']
                }
            },
            'total_parameters': self.model_info['total_params'],
            'memory_usage': f"{self.model_info['total_params'] * 4 / 1024 / 1024:.2f} MB"
        }
        return summary
    
    def count_parameters(self):
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, filepath, include_optimizer=False, optimizer=None, epoch=None, loss=None):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            include_optimizer: 是否包含优化器状态
            optimizer: 优化器对象
            epoch: 训练轮数
            loss: 损失值
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_param_dim': self.input_param_dim,
                'output_channels': self.output_channels
            },
            'model_info': self.model_info
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            save_dict['epoch'] = epoch
        
        if loss is not None:
            save_dict['loss'] = loss
        
        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath, device='cpu'):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            device: 设备
            
        Returns:
            model: 加载的模型
            checkpoint: 检查点信息
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # 获取模型配置
        model_config = checkpoint.get('model_config', {})
        
        # 创建模型实例
        model = cls(
            input_param_dim=model_config.get('input_param_dim', 9),
            output_channels=model_config.get('output_channels', 1)
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"模型已从 {filepath} 加载")
        return model, checkpoint


def test_complete_model():
    """测试完整的FiLM-UNet模型"""
    print("测试完整的FiLM-UNet模型...")
    
    # 创建模型
    model = FiLMUNetModel()
    
    # 创建测试数据
    batch_size = 2
    design_params = torch.randn(batch_size, 9)
    
    print(f"输入设计参数形状: {design_params.shape}")
    
    # 前向传播
    with torch.no_grad():
        output, intermediate_outputs, debug_info = model(design_params)
        
        print(f"\n=== 输出结果 ===")
        print(f"最终输出形状: {output.shape}")
        print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        print(f"\n=== 中间输出 ===")
        for name, tensor in intermediate_outputs.items():
            print(f"  {name}: {tensor.shape}")
        
        print(f"\n=== 调试信息 ===")
        for key, value in debug_info.items():
            print(f"  {key}: {value}")
        
        # 模型摘要
        print(f"\n=== 模型摘要 ===")
        summary = model.get_model_summary()
        print(f"架构: {summary['architecture']}")
        print(f"总参数量: {summary['total_parameters']:,}")
        print(f"内存使用: {summary['memory_usage']}")
        
        print(f"\n=== 各组件参数量 ===")
        for component, info in summary['components'].items():
            print(f"  {component}: {info['parameters']:,} 参数")
    
    # 测试简化预测接口
    print(f"\n=== 测试预测接口 ===")
    single_param = torch.randn(9)
    prediction = model.predict(single_param)
    print(f"单样本预测形状: {prediction.shape}")
    
    # 测试保存和加载
    print(f"\n=== 测试保存/加载 ===")
    save_path = "test_film_unet.pth"
    model.save_model(save_path)
    
    loaded_model, checkpoint = FiLMUNetModel.load_model(save_path)
    print(f"加载的模型参数量: {loaded_model.count_parameters():,}")
    
    # 清理测试文件
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"测试文件已清理: {save_path}")


if __name__ == "__main__":
    test_complete_model()