#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块
实现FiLM-UNet模型的训练循环和验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import json

from film_unet_model import FiLMUNetModel
from custom_losses import CompositeLoss
from data_preprocessing import RCSDataLoader


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=20, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        self.current_epoch_metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.current_epoch_metrics[key].append(value)
    
    def end_epoch(self):
        for key, values in self.current_epoch_metrics.items():
            self.metrics[key].append(np.mean(values))
        self.current_epoch_metrics.clear()
    
    def get_latest(self, key):
        return self.metrics[key][-1] if self.metrics[key] else 0.0
    
    def get_history(self, key):
        return self.metrics[key]


class FiLMUNetTrainer:
    """FiLM-UNet训练器"""
    
    def __init__(self, 
                 model=None,
                 device='auto',
                 loss_weights=None,
                 optimizer_params=None,
                 scheduler_params=None):
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 模型
        if model is None:
            self.model = FiLMUNetModel()
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # 损失函数
        if loss_weights is None:
            loss_weights = {
                'lambda_mse': 1.0,
                'lambda_smooth': 0.01,
                'lambda_physics': 0.05,
                'lambda_multiscale': 0.1
            }
        
        self.criterion = CompositeLoss(**loss_weights)
        
        # 优化器
        if optimizer_params is None:
            optimizer_params = {
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'betas': (0.9, 0.999)
            }
        
        self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_params)
        
        # 学习率调度器
        if scheduler_params is None:
            scheduler_params = {
                'T_max': 100,
                'eta_min': 1e-6
            }
        
        self.scheduler = CosineAnnealingLR(self.optimizer, **scheduler_params)
        
        # 追踪器
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.best_val_loss = float('inf')
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, (design_params, rcs_targets) in enumerate(train_loader):
            # 数据移到设备
            design_params = design_params.to(self.device)
            rcs_targets = rcs_targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            predictions, intermediate_outputs, debug_info = self.model(design_params)
            
            # 计算损失
            total_loss, loss_breakdown = self.criterion(
                predictions, rcs_targets, intermediate_outputs
            )
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新指标
            self.train_metrics.update(
                total_loss=total_loss,
                mse_loss=loss_breakdown['mse']['loss'],
                smooth_loss=loss_breakdown['smooth']['loss'],
                physics_loss=loss_breakdown['physics']['loss'],
                multiscale_loss=loss_breakdown['multiscale']['loss']
            )
            
            self.global_step += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                progress = 100. * batch_idx / len(train_loader)
                print(f'训练 Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                      f'({progress:.1f}%)]\t损失: {total_loss.item():.6f}')
        
        # 结束epoch
        self.train_metrics.end_epoch()
        epoch_time = time.time() - epoch_start_time
        
        print(f'训练 Epoch {epoch} 完成, 时间: {epoch_time:.2f}s')
        print(f'  总损失: {self.train_metrics.get_latest("total_loss"):.6f}')
        print(f'  MSE损失: {self.train_metrics.get_latest("mse_loss"):.6f}')
        print(f'  平滑损失: {self.train_metrics.get_latest("smooth_loss"):.6f}')
        
    def validate_epoch(self, val_loader, epoch):
        """验证一个epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for design_params, rcs_targets in val_loader:
                # 数据移到设备
                design_params = design_params.to(self.device)
                rcs_targets = rcs_targets.to(self.device)
                
                # 前向传播
                predictions, intermediate_outputs, debug_info = self.model(design_params)
                
                # 计算损失
                total_loss, loss_breakdown = self.criterion(
                    predictions, rcs_targets, intermediate_outputs
                )
                
                # 更新指标
                self.val_metrics.update(
                    total_loss=total_loss,
                    mse_loss=loss_breakdown['mse']['loss'],
                    smooth_loss=loss_breakdown['smooth']['loss'],
                    physics_loss=loss_breakdown['physics']['loss'],
                    multiscale_loss=loss_breakdown['multiscale']['loss']
                )
        
        # 结束epoch
        self.val_metrics.end_epoch()
        
        val_loss = self.val_metrics.get_latest("total_loss")
        print(f'验证 Epoch {epoch}:')
        print(f'  总损失: {val_loss:.6f}')
        print(f'  MSE损失: {self.val_metrics.get_latest("mse_loss"):.6f}')
        
        return val_loss
    
    def train(self, train_loader, val_loader, epochs=500, 
              save_dir='./checkpoints', save_freq=50,
              early_stopping_patience=50):
        """完整训练流程"""
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 早停机制
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        print(f"开始训练...")
        print(f"  模型参数量: {self.model.count_parameters():,}")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  验证批次: {len(val_loader)}")
        print(f"  训练轮数: {epochs}")
        print(f"  设备: {self.device}")
        
        training_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            
            # 训练
            self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.validate_epoch(val_loader, epoch)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'  学习率: {current_lr:.6e}')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch, val_loss, is_best=True
                )
                print(f'  ✓ 保存最佳模型 (验证损失: {val_loss:.6f})')
            
            # 定期保存
            if epoch % save_freq == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'),
                    epoch, val_loss
                )
            
            # 早停检查
            if early_stopping(val_loss, self.model):
                print(f'早停触发，在第 {epoch} 轮停止训练')
                print(f'最佳验证损失: {early_stopping.best_loss:.6f}')
                break
            
            print('-' * 80)
        
        training_time = time.time() - training_start_time
        print(f"\n训练完成！")
        print(f"  总训练时间: {training_time / 3600:.2f} 小时")
        print(f"  最佳验证损失: {self.best_val_loss:.6f}")
        
        # 保存训练历史
        self.save_training_history(save_dir)
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
        
        return self.best_val_loss
    
    def save_checkpoint(self, filepath, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'train_metrics': self.train_metrics.metrics,
            'val_metrics': self.val_metrics.metrics,
            'model_info': self.model.get_model_summary()
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"  最佳模型已保存: {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_metrics.metrics = checkpoint['train_metrics']
        self.val_metrics.metrics = checkpoint['val_metrics']
        
        print(f"检查点已加载: {filepath}")
        print(f"  轮数: {self.epoch}")
        print(f"  最佳验证损失: {self.best_val_loss:.6f}")
    
    def save_training_history(self, save_dir):
        """保存训练历史"""
        history = {
            'train_metrics': self.train_metrics.metrics,
            'val_metrics': self.val_metrics.metrics,
            'best_val_loss': self.best_val_loss,
            'epochs': self.epoch
        }
        
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # 转换numpy类型为Python原生类型
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, dict):
                    history_serializable[key] = {
                        k: [float(v) for v in vals] if isinstance(vals, list) else vals
                        for k, vals in value.items()
                    }
                else:
                    history_serializable[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
            
            json.dump(history_serializable, f, indent=2)
        
        print(f"训练历史已保存: {history_path}")
    
    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(self.train_metrics.get_history('total_loss')) + 1)
            
            # 总损失
            axes[0, 0].plot(epochs, self.train_metrics.get_history('total_loss'), 'b-', label='训练', linewidth=2)
            axes[0, 0].plot(epochs, self.val_metrics.get_history('total_loss'), 'r-', label='验证', linewidth=2)
            axes[0, 0].set_title('总损失')
            axes[0, 0].set_xlabel('轮数')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # MSE损失
            axes[0, 1].plot(epochs, self.train_metrics.get_history('mse_loss'), 'b-', label='训练', linewidth=2)
            axes[0, 1].plot(epochs, self.val_metrics.get_history('mse_loss'), 'r-', label='验证', linewidth=2)
            axes[0, 1].set_title('MSE损失')
            axes[0, 1].set_xlabel('轮数')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 平滑损失
            axes[1, 0].plot(epochs, self.train_metrics.get_history('smooth_loss'), 'b-', label='训练', linewidth=2)
            axes[1, 0].plot(epochs, self.val_metrics.get_history('smooth_loss'), 'r-', label='验证', linewidth=2)
            axes[1, 0].set_title('平滑损失')
            axes[1, 0].set_xlabel('轮数')
            axes[1, 0].set_ylabel('TV损失')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 物理损失
            axes[1, 1].plot(epochs, self.train_metrics.get_history('physics_loss'), 'b-', label='训练', linewidth=2)
            axes[1, 1].plot(epochs, self.val_metrics.get_history('physics_loss'), 'r-', label='验证', linewidth=2)
            axes[1, 1].set_title('物理约束损失')
            axes[1, 1].set_xlabel('轮数')
            axes[1, 1].set_ylabel('物理损失')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图像
            curves_path = os.path.join(save_dir, 'training_curves.png')
            plt.savefig(curves_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"训练曲线已保存: {curves_path}")
            
        except Exception as e:
            print(f"绘制训练曲线失败: {e}")


def test_trainer():
    """测试训练器"""
    print("测试FiLM-UNet训练器...")
    
    # 创建模拟数据
    num_samples = 32
    design_params = torch.randn(num_samples, 9)
    rcs_data = torch.randn(num_samples, 1, 91, 91)
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(design_params, rcs_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 创建训练器
    model = FiLMUNetModel()
    trainer = FiLMUNetTrainer(model=model, device='cpu')  # 使用CPU进行测试
    
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 短期训练测试
    print("\n开始测试训练...")
    trainer.train(
        train_loader, val_loader,
        epochs=5,  # 只训练5个epoch用于测试
        save_dir='./test_checkpoints',
        save_freq=2,
        early_stopping_patience=10
    )
    
    print("训练器测试完成！")
    
    # 清理测试文件
    import shutil
    if os.path.exists('./test_checkpoints'):
        shutil.rmtree('./test_checkpoints')
        print("测试文件已清理")


if __name__ == "__main__":
    test_trainer()