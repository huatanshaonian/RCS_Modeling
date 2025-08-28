"""
Autoencoder训练模块
包含训练逻辑、优化器配置和模型评估功能
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

try:
    from .autoencoder_models import vae_loss_function
    from .autoencoder_utils import (
        log_info, log_progress, optimize_model_for_gpu, 
        monitor_gpu_memory, PYTORCH_AVAILABLE
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from autoencoder_models import vae_loss_function
    from autoencoder_utils import (
        log_info, log_progress, optimize_model_for_gpu, 
        monitor_gpu_memory, PYTORCH_AVAILABLE
    )


def train_autoencoder(model, train_loader, val_loader, epochs=200, learning_rate=1e-3,
                      device='cpu', model_type='standard', beta=1.0, output_dir=None):
    """训练自编码器模型 - 修复StopIteration错误"""

    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch不可用，无法进行训练")

    # GPU优化模型
    model = optimize_model_for_gpu(model, device)

    # 使用混合精度训练（如果支持）
    use_amp = device.type == 'cuda' and hasattr(torch.cuda, 'amp')
    if use_amp:
        log_info("启用混合精度训练", "SUCCESS")
        scaler = torch.cuda.amp.GradScaler()

    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

    # 训练状态变量
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50

    log_info(f"开始训练{model_type.upper()}模型 (设备: {device})")

    # ===== 修复：安全的数据访问和批次大小测试 =====
    try:
        # 检查数据加载器是否为空
        if len(train_loader) == 0:
            raise ValueError("训练数据加载器为空")

        if len(val_loader) == 0:
            raise ValueError("验证数据加载器为空")

        print(f"📊 数据加载器状态:")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print(f"  训练样本数: {len(train_loader.dataset)}")
        print(f"  验证样本数: {len(val_loader.dataset)}")

        # 安全获取输入维度
        sample_batch = None
        for batch_data in train_loader:
            sample_batch = batch_data[0]
            break  # 只取第一批数据

        if sample_batch is None:
            raise ValueError("无法从训练数据加载器获取样本数据")

        input_dim = sample_batch.shape[1]
        current_batch_size = sample_batch.shape[0]

        print(f"📏 数据维度信息:")
        print(f"  输入维度: {input_dim}")
        print(f"  当前批次大小: {current_batch_size}")

        # GPU批次大小优化（简化版本）
        if device.type == 'cuda' and current_batch_size < 128:
            try:
                print("🔍 测试更大批次大小的可行性...")

                # 测试2倍批次大小
                test_batch_size = min(current_batch_size * 2, 128)
                test_data = torch.randn(test_batch_size, input_dim).to(device)

                with torch.cuda.amp.autocast() if use_amp else torch.no_grad():
                    if model_type == 'standard':
                        recon, _ = model(test_data)
                        loss = F.mse_loss(recon, test_data)
                    else:
                        recon, mu, logvar, _ = model(test_data)
                        loss, _, _ = vae_loss_function(recon, test_data, mu, logvar, 1.0)

                # 测试反向传播
                loss.backward()
                optimizer.zero_grad()

                print(f"✅ 更大批次 {test_batch_size} 测试成功")

                # 清理测试数据
                del test_data, loss, recon
                if 'mu' in locals():
                    del mu, logvar
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  更大批次大小显存不足，继续使用当前大小")
                    torch.cuda.empty_cache()
                else:
                    print(f"⚠️  批次测试出错: {e}")

        # GPU预热 - 修复版本（解决BatchNorm批次大小问题）
        if device.type == 'cuda':
            log_info("GPU预热中...", "PROGRESS")
            try:
                # 使用至少2个样本的批次大小避免BatchNorm错误
                batch_size_for_warmup = max(2, min(current_batch_size, 8))
                dummy_input = torch.randn(batch_size_for_warmup, input_dim).to(device)

                with torch.no_grad():
                    if model_type == 'standard':
                        _, _ = model(dummy_input)
                    else:
                        _, _, _, _ = model(dummy_input)

                torch.cuda.synchronize()
                del dummy_input
                torch.cuda.empty_cache()
                log_info("GPU预热完成", "SUCCESS")

            except Exception as e:
                print(f"⚠️  GPU预热失败: {e}")

    except Exception as e:
        print(f"❌ 数据加载器检查失败: {e}")
        print("使用基础训练模式...")

    # ===== 主训练循环 =====
    try:
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_batches = 0

            for batch_idx, batch_data in enumerate(train_loader):
                try:
                    batch_data = batch_data[0].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        # 混合精度训练
                        with torch.cuda.amp.autocast():
                            if model_type == 'standard':
                                recon_batch, _ = model(batch_data)
                                loss = F.mse_loss(recon_batch, batch_data)
                            else:  # VAE
                                recon_batch, mu, logvar, _ = model(batch_data)
                                loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_data, mu, logvar, beta)
                                train_recon_loss += recon_loss.item()
                                train_kl_loss += kl_loss.item()

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # 标准训练
                        if model_type == 'standard':
                            recon_batch, _ = model(batch_data)
                            loss = F.mse_loss(recon_batch, batch_data)
                        else:  # VAE
                            recon_batch, mu, logvar, _ = model(batch_data)
                            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_data, mu, logvar, beta)
                            train_recon_loss += recon_loss.item()
                            train_kl_loss += kl_loss.item()

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    train_loss += loss.item()
                    train_batches += 1

                    # 定期监控（降低频率并智能输出）
                    if batch_idx % 100 == 0 and device.type == 'cuda':
                        # 每个epoch的开始强制输出，其他时候只在变化时输出
                        force_output = (batch_idx == 0)
                        monitor_gpu_memory(force_output=force_output)

                except Exception as batch_error:
                    print(f"⚠️  批次 {batch_idx} 处理失败: {batch_error}")
                    continue  # 跳过有问题的批次

            # 检查是否有有效的训练批次
            if train_batches == 0:
                print("❌ 没有成功处理任何训练批次")
                break

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch_data in val_loader:
                    try:
                        batch_data = batch_data[0].to(device, non_blocking=True)

                        if use_amp:
                            with torch.cuda.amp.autocast():
                                if model_type == 'standard':
                                    recon_batch, _ = model(batch_data)
                                    loss = F.mse_loss(recon_batch, batch_data)
                                else:  # VAE
                                    recon_batch, mu, logvar, _ = model(batch_data)
                                    loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_data, mu, logvar,
                                                                                  beta)
                                    val_recon_loss += recon_loss.item()
                                    val_kl_loss += kl_loss.item()
                        else:
                            if model_type == 'standard':
                                recon_batch, _ = model(batch_data)
                                loss = F.mse_loss(recon_batch, batch_data)
                            else:  # VAE
                                recon_batch, mu, logvar, _ = model(batch_data)
                                loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_data, mu, logvar, beta)
                                val_recon_loss += recon_loss.item()
                                val_kl_loss += kl_loss.item()

                        val_loss += loss.item()
                        val_batches += 1

                    except Exception as batch_error:
                        print(f"⚠️  验证批次处理失败: {batch_error}")
                        continue

            # 检查验证批次
            if val_batches == 0:
                print("❌ 没有成功处理任何验证批次")
                break

            # 计算平均损失 - 修复除零错误
            avg_train_loss = train_loss / max(train_batches, 1)
            avg_val_loss = val_loss / max(val_batches, 1)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 早停检查和模型保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                if output_dir:
                    try:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save({
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'loss': best_val_loss,
                        }, os.path.join(output_dir, f'best_{model_type}_model.pth'))
                    except Exception as save_error:
                        print(f"⚠️  模型保存失败: {save_error}")
            else:
                patience_counter += 1

            # 进度输出
            if epoch % 20 == 0 or epoch == epochs - 1:
                lr = optimizer.param_groups[0]['lr']
                if model_type == 'standard':
                    log_progress(epoch + 1, epochs,
                                 f"训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}, LR={lr:.2e}")
                else:
                    log_progress(epoch + 1, epochs, f"训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}")
                    if train_batches > 0:
                        log_info(
                            f"  重构损失={train_recon_loss / train_batches:.6f}, KL损失={train_kl_loss / train_batches:.6f}")

            # 早停
            if patience_counter >= patience:
                log_info(f"早停于第 {epoch + 1} 轮，最佳验证损失: {best_val_loss:.6f}", "WARNING")
                break

    except Exception as training_error:
        print(f"❌ 训练过程出错: {training_error}")
        import traceback
        traceback.print_exc()
        return model, train_losses, val_losses

    # 加载最佳模型
    if output_dir and os.path.exists(os.path.join(output_dir, f'best_{model_type}_model.pth')):
        try:
            checkpoint = torch.load(os.path.join(output_dir, f'best_{model_type}_model.pth'))
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as load_error:
            print(f"⚠️  加载最佳模型失败: {load_error}")

    # 清理GPU内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        log_info("训练完成，已清理GPU内存", "SUCCESS")

    return model, train_losses, val_losses


def evaluate_model(model, data_tensor, scaler, device, model_type='standard'):
    """评估模型性能"""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch不可用，无法进行评估")
        
    model.eval()
    with torch.no_grad():
        if model_type == 'standard':
            recon, latent = model(data_tensor.to(device))
        else:  # VAE
            recon, mu, logvar, latent = model(data_tensor.to(device))
            latent = mu  # 使用均值作为潜在表示

        # 转换回CPU并转为numpy
        recon = recon.cpu().numpy()
        latent = latent.cpu().numpy()

    # 反标准化重构数据
    original_data = scaler.inverse_transform(data_tensor.cpu().numpy())
    recon_original = scaler.inverse_transform(recon)

    # 计算重构误差
    mse = mean_squared_error(original_data, recon_original)
    r2 = r2_score(original_data.flatten(), recon_original.flatten())

    return {
        'latent': latent,
        'reconstruction': recon_original,
        'mse': mse,
        'r2': r2
    }