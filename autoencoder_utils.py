"""
Autoencoder工具函数模块
包含数据验证、设备管理、日志记录等工具函数
"""

import os
import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

# 全局变量用于跟踪上次的内存状态
_last_memory_state = None

# PyTorch相关导入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


def validate_data_integrity(rcs_data, param_data, available_models=None):
    """验证数据完整性"""
    print("[INFO] 检查数据完整性...")

    if len(rcs_data) == 0:
        raise ValueError("RCS数据为空")

    if len(param_data) == 0:
        raise ValueError("参数数据为空")

    if len(rcs_data) != len(param_data):
        print(f"[WARNING] 数据长度不匹配: RCS({len(rcs_data)}) vs 参数({len(param_data)})")

    if available_models is not None:
        if len(available_models) != len(rcs_data):
            print(f"[WARNING] 可用模型索引长度不匹配: {len(available_models)} vs {len(rcs_data)}")

    # 检查NaN值
    nan_count_rcs = np.isnan(rcs_data).sum()
    nan_count_param = np.isnan(param_data).sum()

    if nan_count_rcs > 0:
        print(f"[WARNING] RCS数据包含 {nan_count_rcs} 个NaN值")

    if nan_count_param > 0:
        print(f"[WARNING] 参数数据包含 {nan_count_param} 个NaN值")

    print(f"[OK] 数据完整性检查通过: {len(rcs_data)} 个样本")
    return True


def log_info(message, level="INFO"):
    """统一的日志输出格式"""
    symbols = {
        "INFO": "[INFO]",
        "SUCCESS": "[OK]",
        "WARNING": "[WARNING]",
        "ERROR": "[ERROR]",
        "PROGRESS": "[PROGRESS]"
    }
    symbol = symbols.get(level, "[INFO]")
    print(f"{symbol} {message}")


def log_progress(current, total, message=""):
    """进度显示"""
    if total > 0:
        percent = (current / total) * 100
        print(f"[PROGRESS] {message} [{current}/{total}] ({percent:.1f}%)")


def get_device_info():
    """
    获取详细的设备信息和CUDA配置 - 简化版本
    """
    if not PYTORCH_AVAILABLE:
        return torch.device('cpu'), "PyTorch不可用"

    if not torch.cuda.is_available():
        device_info = """设备: CPU
原因: CUDA不可用或PyTorch为CPU版本"""
        return torch.device('cpu'), device_info

    # CUDA可用时的信息
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)

    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated(current_device)

    device_info = f"""设备: {device_name}
显存: {free_memory / 1024 ** 3:.1f} GB 可用 / {total_memory / 1024 ** 3:.1f} GB 总计
CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}"""

    return torch.device(f'cuda:{current_device}'), device_info


def optimize_model_for_gpu(model, device):
    """
    为GPU优化模型
    """
    if not PYTORCH_AVAILABLE:
        return model

    model = model.to(device)

    # 如果是多GPU环境，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个GPU，使用DataParallel")
        model = nn.DataParallel(model)

    # 启用cuDNN优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    return model


def get_optimal_batch_size(input_dim, device, max_memory_gb=None):
    """更激进的批次大小计算"""
    if device.type == 'cpu':
        return 64  # CPU时增加批次大小

    try:
        current_device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024 ** 3
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1024 ** 3
        available_memory = total_memory - allocated_memory - 0.5  # 减少安全余量

        # 更激进的内存使用（使用70%而不是50%）
        memory_per_sample = input_dim * 4 * 2 / 1024 ** 3  # 减少估算倍数
        batch_size = int((available_memory * 0.7) / memory_per_sample)

        # 扩大批次大小范围
        batch_size = max(32, min(batch_size, 512))

        print(f"[INFO] 设置大批次大小: {batch_size} (可用显存: {available_memory:.1f}GB)")
        return batch_size
    except Exception as e:
        print(f"批次大小计算失败: {e}，使用保守值")
        return 64


def monitor_gpu_memory(force_output=False):
    """监控GPU内存使用情况 - 只在数据变化时输出"""
    global _last_memory_state
    
    if not PYTORCH_AVAILABLE or not torch.cuda.is_available():
        return

    try:
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(current_device) / 1024 ** 3

        # 创建当前状态的标识（保留两位小数以避免微小变化）
        current_state = (round(allocated, 2), round(reserved, 2))
        
        # 只有在数据发生变化或强制输出时才打印
        if force_output or _last_memory_state != current_state:
            print(f"GPU内存使用: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
            _last_memory_state = current_state

        # 内存使用率过高时清理
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024 ** 3
        if allocated / total_memory > 0.8:
            print("GPU内存使用率过高，执行清理...")
            torch.cuda.empty_cache()
            # 清理后重置状态，这样下次会显示清理后的结果
            _last_memory_state = None
    except Exception as e:
        print(f"GPU内存监控失败: {e}")


def create_optimized_data_loaders(rcs_train_tensor, rcs_test_tensor, device, optimal_batch_size):
    """创建优化的数据加载器"""
    # 判断是否预加载到GPU
    data_on_gpu = False
    if device.type == 'cuda' and rcs_train_tensor.numel() * 4 / 1024 ** 3 < 2:  # 小于2GB
        print("[INFO] 预加载数据到GPU...")
        rcs_train_tensor = rcs_train_tensor.to(device)
        if rcs_test_tensor is not None:
            rcs_test_tensor = rcs_test_tensor.to(device)
        data_on_gpu = True

    # 数据加载器配置 - 根据数据位置设置pin_memory
    pin_memory = device.type == 'cuda' and not data_on_gpu  # 只有数据在CPU且使用GPU时才启用
    num_workers = 0 if data_on_gpu else min(4, os.cpu_count())  # GPU上的数据不使用多进程

    # DataLoader参数配置
    dataloader_kwargs = {
        'pin_memory': pin_memory,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 4 if num_workers > 0 else None,
    }

    print(f"[INFO] 数据加载器配置: batch_size={optimal_batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")

    # 创建训练数据加载器
    train_dataset = TensorDataset(rcs_train_tensor)

    # 验证集处理
    if rcs_test_tensor is not None and len(rcs_test_tensor) > 0:
        print("[INFO] 使用独立的测试集作为验证集")
        val_dataset = TensorDataset(rcs_test_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > optimal_batch_size,
            **dataloader_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=min(optimal_batch_size, len(val_dataset)),
            shuffle=False,
            drop_last=False,
            **dataloader_kwargs
        )

    else:
        print("[INFO] 从训练集中划分验证集")

        # 确保有足够的数据进行划分
        total_samples = len(train_dataset)
        if total_samples < 10:
            print(f"[ERROR] 训练样本数量太少 ({total_samples})，无法进行训练")
            return None, None

        # 计算验证集大小（至少1个，最多20%）
        val_size = max(1, min(total_samples // 5, total_samples - 5))  # 至少保留5个训练样本
        train_size = total_samples - val_size

        print(f"[INFO] 数据划分: 训练={train_size}, 验证={val_size}")

        # 使用固定种子确保可重现性
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=generator
        )

        # 调整批次大小以适应子集
        train_batch_size = min(optimal_batch_size, max(1, train_size // 2))
        val_batch_size = min(optimal_batch_size, val_size)

        train_loader = DataLoader(
            train_subset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=train_size > train_batch_size,
            **dataloader_kwargs
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            **dataloader_kwargs
        )

    return train_loader, val_loader


def cleanup_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        # 清理前显示内存状态
        monitor_gpu_memory(force_output=True)
        torch.cuda.empty_cache()
        log_info("GPU内存已清理", "SUCCESS")
        # 清理后再次显示内存状态
        monitor_gpu_memory(force_output=True)