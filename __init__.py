"""
Autoencoder分析包
用于RCS数据的深度学习降维分析
"""

# 导出主要函数和类
try:
    from .autoencoder_analysis import perform_autoencoder_analysis
    from .autoencoder_models import StandardAutoencoder, VariationalAutoencoder, EnhancedAutoencoder
    from .autoencoder_training import train_autoencoder, evaluate_model
    from .autoencoder_visualization import (
        plot_training_history, visualize_latent_space, 
        analyze_reconstruction_error, generate_comparison_analysis,
        compare_with_pod_results
    )
    from .autoencoder_utils import (
        validate_data_integrity, get_device_info, 
        log_info, PYTORCH_AVAILABLE
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from autoencoder_analysis import perform_autoencoder_analysis
    from autoencoder_models import StandardAutoencoder, VariationalAutoencoder, EnhancedAutoencoder
    from autoencoder_training import train_autoencoder, evaluate_model
    from autoencoder_visualization import (
        plot_training_history, visualize_latent_space, 
        analyze_reconstruction_error, generate_comparison_analysis,
        compare_with_pod_results
    )
    from autoencoder_utils import (
        validate_data_integrity, get_device_info, 
        log_info, PYTORCH_AVAILABLE
    )

__version__ = "1.0.0"
__author__ = "RCS Analysis Team"

# 定义公共接口
__all__ = [
    'perform_autoencoder_analysis',
    'StandardAutoencoder',
    'VariationalAutoencoder', 
    'EnhancedAutoencoder',
    'train_autoencoder',
    'evaluate_model',
    'plot_training_history',
    'visualize_latent_space',
    'analyze_reconstruction_error',
    'generate_comparison_analysis',
    'compare_with_pod_results',
    'validate_data_integrity',
    'get_device_info',
    'log_info',
    'PYTORCH_AVAILABLE'
]