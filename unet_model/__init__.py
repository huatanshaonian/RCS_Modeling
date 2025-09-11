#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FiLM-UNet模块
基于MLP+FiLM+U-Net的深度学习架构，用于从飞行器设计参数预测RCS数据
"""

__version__ = "1.0.0"
__author__ = "Claude Code Assistant"
__description__ = "FiLM-UNet for RCS Prediction"

# 主要组件导入
try:
    from film_unet_model import FiLMUNetModel
    from mlp_encoder import MLPEncoder, DualPathProcessor
    from modified_unet import ModifiedUNet
    from film_layer import FiLMLayer, ConvFiLMBlock
    from custom_losses import CompositeLoss, TVLoss, PhysicsConstraintLoss, MultiscaleLoss
    from data_preprocessing import RCSDataLoader, DataNormalizer, DataAugmenter, RCSDataset
    from trainer import FiLMUNetTrainer, EarlyStopping, MetricsTracker
    from inference import RCSPredictor, RCSVisualizer, ModelEvaluator
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from film_unet_model import FiLMUNetModel
    from mlp_encoder import MLPEncoder, DualPathProcessor
    from modified_unet import ModifiedUNet
    from film_layer import FiLMLayer, ConvFiLMBlock
    from custom_losses import CompositeLoss, TVLoss, PhysicsConstraintLoss, MultiscaleLoss
    from data_preprocessing import RCSDataLoader, DataNormalizer, DataAugmenter, RCSDataset
    from trainer import FiLMUNetTrainer, EarlyStopping, MetricsTracker
    from inference import RCSPredictor, RCSVisualizer, ModelEvaluator

__all__ = [
    # 模型组件
    'FiLMUNetModel',
    'MLPEncoder', 
    'DualPathProcessor',
    'ModifiedUNet',
    'FiLMLayer',
    'ConvFiLMBlock',
    
    # 损失函数
    'CompositeLoss',
    'TVLoss',
    'PhysicsConstraintLoss', 
    'MultiscaleLoss',
    
    # 数据处理
    'RCSDataLoader',
    'DataNormalizer',
    'DataAugmenter',
    'RCSDataset',
    
    # 训练相关
    'FiLMUNetTrainer',
    'EarlyStopping',
    'MetricsTracker',
    
    # 推理相关
    'RCSPredictor',
    'RCSVisualizer', 
    'ModelEvaluator'
]

# 模块信息
def get_module_info():
    """获取模块信息"""
    return {
        'name': 'FiLM-UNet',
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'components': len(__all__),
        'features': [
            'MLP编码器：9维参数 → 1024维特征',
            'FiLM调制：条件特征调制机制',
            'Modified U-Net：91×91 RCS预测',
            '多尺度损失：23×23, 46×46, 91×91',
            '物理约束：对称性和边界条件',
            '数据增强：噪声、Mixup、参数抖动',
            '完整训练管道：早停、学习率调度',
            '推理可视化：热图、3D图、误差分析'
        ]
    }

def print_module_info():
    """打印模块信息"""
    info = get_module_info()
    print(f"\n{info['name']} v{info['version']}")
    print(f"{info['description']}")
    print(f"作者: {info['author']}")
    print(f"组件数量: {info['components']}")
    print("\n主要特性:")
    for feature in info['features']:
        print(f"  • {feature}")
    print()

# 依赖检查
def check_dependencies():
    """检查依赖项"""
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    available = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            available.append(name)
        except ImportError:
            missing.append(name)
    
    print("依赖检查:")
    print(f"  ✓ 可用: {', '.join(available)}")
    if missing:
        print(f"  ✗ 缺失: {', '.join(missing)}")
        print("  请安装缺失的依赖项")
    else:
        print("  ✓ 所有依赖项已安装")
    
    return len(missing) == 0

# 快速测试
def quick_test():
    """快速功能测试"""
    try:
        print("执行快速功能测试...")
        
        # 测试模型创建
        model = FiLMUNetModel()
        print(f"  ✓ 模型创建成功，参数量: {model.count_parameters():,}")
        
        # 测试前向传播
        import torch
        test_input = torch.randn(2, 9)
        with torch.no_grad():
            output, _, _ = model(test_input)
        print(f"  ✓ 前向传播成功，输出形状: {output.shape}")
        
        # 测试损失函数
        target = torch.randn(2, 1, 91, 91)
        loss_fn = CompositeLoss()
        loss, _ = loss_fn(output, target)
        print(f"  ✓ 损失计算成功，损失值: {loss.item():.6f}")
        
        print("快速测试完成！所有功能正常。")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print_module_info()
    if check_dependencies():
        quick_test()