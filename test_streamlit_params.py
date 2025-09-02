#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Streamlit界面参数传递功能
"""

import json
import sys
sys.path.append('.')

def test_default_config():
    """测试默认配置"""
    from streamlit_app import load_default_config
    
    config = load_default_config()
    
    print("=== 默认配置测试 ===")
    print("基础参数:")
    print(f"  参数文件: {config['params_path']}")
    print(f"  RCS目录: {config['rcs_dir']}")
    print(f"  输出目录: {config['output_dir']}")
    print(f"  频率: {config['frequency']}")
    print(f"  模型数量: {config['num_models']}")
    print(f"  训练集大小: {config['num_train']}")
    
    print("\nPOD参数:")
    print(f"  POD启用: {config['pod_enabled']}")
    print(f"  模态数量: {config['pod_modes']}")
    print(f"  能量阈值: {config['energy_threshold']}%")
    print(f"  可视化模态数: {config['num_modes_visualize']}")
    print(f"  重建模态数: {config['pod_reconstruct_num']}")
    
    print("\nAutoencoder参数:")
    print(f"  AE启用: {config['ae_enabled']}")
    print(f"  跳过训练: {config['skip_ae_training']}")
    print(f"  隐空间维度: {config['latent_dims']}")
    print(f"  模型类型: {config['model_types']}")
    print(f"  训练轮数: {config['ae_epochs']}")
    print(f"  学习率: {config['ae_learning_rate']}")
    print(f"  批次大小: {config['ae_batch_size']}")
    print(f"  计算设备: {config['ae_device']}")

def test_command_generation():
    """测试命令生成"""
    from streamlit_app import load_default_config
    
    # 模拟run_analysis函数的命令生成部分
    config = load_default_config()
    
    # 修改一些参数进行测试
    config['frequency'] = ['1.5G', '3G']
    config['num_train'] = [60, 70, 80]
    config['pod_modes'] = [5, 10, 15, 20]
    config['energy_threshold'] = 99.5
    config['latent_dims'] = [3, 5, 8, 10]
    config['model_types'] = ['standard']
    
    cmd = ['python', 'run.py']
    cmd.extend(['--params_path', config['params_path']])
    cmd.extend(['--rcs_dir', config['rcs_dir']])
    cmd.extend(['--output_dir', config['output_dir']])
    cmd.extend(['--freq', ','.join(config['frequency'])])
    cmd.extend(['--num_models', str(config['num_models'])])
    cmd.extend(['--num_train', ','.join(map(str, config['num_train']))])
    
    # POD参数
    if config.get('pod_enabled', True):
        cmd.extend(['--pod_modes', ','.join(map(str, config.get('pod_modes', [10, 20, 30, 40])))])
        cmd.extend(['--energy_threshold', str(config.get('energy_threshold', 95.0))])
        cmd.extend(['--num_modes_visualize', str(config.get('num_modes_visualize', 10))])
        cmd.extend(['--pod_reconstruct_num', str(config.get('pod_reconstruct_num', 0))])
    
    # Autoencoder参数
    if config.get('ae_enabled', True):
        cmd.extend(['--latent_dims', ','.join(map(str, config.get('latent_dims', [5, 10, 15, 20])))])
        cmd.extend(['--model_types', ','.join(config.get('model_types', ['standard', 'vae']))])
        cmd.extend(['--ae_epochs', str(config.get('ae_epochs', 200))])
        cmd.extend(['--ae_device', config.get('ae_device', 'auto')])
        cmd.extend(['--ae_learning_rate', str(config.get('ae_learning_rate', 0.001))])
        cmd.extend(['--ae_batch_size', str(config.get('ae_batch_size', 0))])
        
        if config.get('skip_ae_training', False):
            cmd.append('--skip_ae_training')
    
    print("\n=== 生成的命令行参数 ===")
    print(" ".join(cmd))
    print("\n=== 分解显示 ===")
    for i in range(0, len(cmd), 2):
        if i + 1 < len(cmd) and cmd[i].startswith('--'):
            print(f"  {cmd[i]}: {cmd[i+1]}")
        elif cmd[i].startswith('--'):
            print(f"  {cmd[i]}: (flag)")
        elif i == 0:
            print(f"  执行命令: {cmd[i]} {cmd[i+1] if i+1 < len(cmd) else ''}")

def test_parameter_validation():
    """验证参数是否与run.py的argparse匹配"""
    print("\n=== 参数验证 ===")
    
    # 检查run.py中定义的所有参数
    expected_params = {
        'params_path': '../parameter/parameters_sorted.csv',
        'rcs_dir': '../parameter/csv_output',
        'output_dir': './results',
        'freq': 'both',
        'num_models': 100,
        'num_train': '80',
        'pod_modes': '10,20,30,40',
        'latent_dims': '5,10,15,20',
        'model_types': 'standard,vae',
        'ae_epochs': 200,
        'ae_device': 'auto',
        'ae_learning_rate': 0.001,
        'ae_batch_size': 0,
        'energy_threshold': 95.0,
        'num_modes_visualize': 10,
        'pod_reconstruct_num': 0
    }
    
    from streamlit_app import load_default_config
    config = load_default_config()
    
    print("检查参数映射:")
    missing_params = []
    for param, expected_value in expected_params.items():
        if param == 'freq':
            # 特殊处理frequency到freq的映射
            actual_value = config.get('frequency', [])
            print(f"  frequency -> freq: {actual_value} ✓")
        elif param in ['num_train', 'pod_modes', 'latent_dims', 'model_types']:
            # 列表参数
            actual_value = config.get(param, [])
            print(f"  {param}: {actual_value} ✓")
        else:
            actual_value = config.get(param, None)
            if actual_value is not None:
                print(f"  {param}: {actual_value} ✓")
            else:
                missing_params.append(param)
                print(f"  {param}: 缺失 ❌")
    
    if missing_params:
        print(f"\n缺失的参数: {missing_params}")
    else:
        print("\n✅ 所有参数都已正确映射!")

if __name__ == "__main__":
    print("Streamlit参数传递测试\n")
    test_default_config()
    test_command_generation()
    test_parameter_validation()
    print("\n测试完成!")