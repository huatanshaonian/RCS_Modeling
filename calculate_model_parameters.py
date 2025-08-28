#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算自编码器模型的参数数量分析
"""

import torch
import torch.nn as nn


def calculate_linear_params(in_features, out_features, bias=True):
    """计算线性层参数数量"""
    weight_params = in_features * out_features
    bias_params = out_features if bias else 0
    return weight_params + bias_params


def analyze_standard_autoencoder_params(input_dim=8281, latent_dim=10):
    """
    分析Standard Autoencoder的参数数量
    
    默认结构：
    - 输入维度: 8281 (91×91 RCS数据)
    - 隐藏层: [4096, 2048, 1024, 512, 256]
    - 潜在维度: 10
    """
    
    print(f"=== Standard Autoencoder 参数分析 ===")
    print(f"输入维度: {input_dim}")
    print(f"潜在维度: {latent_dim}")
    
    hidden_dims = [4096, 2048, 1024, 512, 256]
    print(f"隐藏层维度: {hidden_dims}")
    
    total_params = 0
    
    print("\n编码器参数:")
    print("-" * 50)
    
    prev_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims):
        # Linear层
        linear_params = calculate_linear_params(prev_dim, hidden_dim)
        # BatchNorm层 (weight + bias)
        bn_params = hidden_dim * 2
        
        layer_total = linear_params + bn_params
        total_params += layer_total
        
        print(f"层 {i+1}: {prev_dim} → {hidden_dim}")
        print(f"  Linear: {linear_params:,} 参数 ({prev_dim}×{hidden_dim} + {hidden_dim})")
        print(f"  BatchNorm: {bn_params:,} 参数")
        print(f"  小计: {layer_total:,} 参数")
        print()
        
        prev_dim = hidden_dim
    
    # 最终编码层
    final_encoder_params = calculate_linear_params(prev_dim, latent_dim)
    total_params += final_encoder_params
    print(f"最终编码层: {prev_dim} → {latent_dim}")
    print(f"  参数数: {final_encoder_params:,}")
    print()
    
    print("解码器参数:")
    print("-" * 50)
    
    # 解码器从潜在维度开始
    prev_dim = latent_dim
    for i, hidden_dim in enumerate(reversed(hidden_dims)):
        # Linear层
        linear_params = calculate_linear_params(prev_dim, hidden_dim)
        # BatchNorm层
        bn_params = hidden_dim * 2
        
        layer_total = linear_params + bn_params
        total_params += layer_total
        
        print(f"层 {i+1}: {prev_dim} → {hidden_dim}")
        print(f"  Linear: {linear_params:,} 参数")
        print(f"  BatchNorm: {bn_params:,} 参数")
        print(f"  小计: {layer_total:,} 参数")
        print()
        
        prev_dim = hidden_dim
    
    # 最终重构层
    final_decoder_params = calculate_linear_params(prev_dim, input_dim)
    total_params += final_decoder_params
    print(f"最终重构层: {prev_dim} → {input_dim}")
    print(f"  参数数: {final_decoder_params:,}")
    print()
    
    print(f"总参数数: {total_params:,}")
    print(f"参数数量级: {total_params/1e6:.1f} 百万")
    
    return total_params


def analyze_vae_params(input_dim=8281, latent_dim=10):
    """
    分析VAE的参数数量
    
    默认结构：
    - 隐藏层: [2048, 1024, 512, 256, 128]
    - 额外的mu和logvar层
    """
    
    print(f"\n=== VAE 参数分析 ===")
    print(f"输入维度: {input_dim}")
    print(f"潜在维度: {latent_dim}")
    
    hidden_dims = [2048, 1024, 512, 256, 128]
    print(f"隐藏层维度: {hidden_dims}")
    
    total_params = 0
    
    print("\n编码器参数:")
    print("-" * 50)
    
    prev_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims):
        # Linear层
        linear_params = calculate_linear_params(prev_dim, hidden_dim)
        # BatchNorm层
        bn_params = hidden_dim * 2
        
        layer_total = linear_params + bn_params
        total_params += layer_total
        
        print(f"层 {i+1}: {prev_dim} → {hidden_dim}")
        print(f"  Linear: {linear_params:,} 参数")
        print(f"  BatchNorm: {bn_params:,} 参数")
        print(f"  小计: {layer_total:,} 参数")
        print()
        
        prev_dim = hidden_dim
    
    # mu和logvar层
    mu_params = calculate_linear_params(prev_dim, latent_dim)
    logvar_params = calculate_linear_params(prev_dim, latent_dim)
    total_params += mu_params + logvar_params
    
    print(f"mu层: {prev_dim} → {latent_dim}")
    print(f"  参数数: {mu_params:,}")
    print(f"logvar层: {prev_dim} → {latent_dim}")
    print(f"  参数数: {logvar_params:,}")
    print()
    
    print("解码器参数:")
    print("-" * 50)
    
    # 解码器
    prev_dim = latent_dim
    for i, hidden_dim in enumerate(reversed(hidden_dims)):
        # Linear层
        linear_params = calculate_linear_params(prev_dim, hidden_dim)
        # BatchNorm层
        bn_params = hidden_dim * 2
        
        layer_total = linear_params + bn_params
        total_params += layer_total
        
        print(f"层 {i+1}: {prev_dim} → {hidden_dim}")
        print(f"  Linear: {linear_params:,} 参数")
        print(f"  BatchNorm: {bn_params:,} 参数")
        print(f"  小计: {layer_total:,} 参数")
        print()
        
        prev_dim = hidden_dim
    
    # 最终重构层
    final_decoder_params = calculate_linear_params(prev_dim, input_dim)
    total_params += final_decoder_params
    print(f"最终重构层: {prev_dim} → {input_dim}")
    print(f"  参数数: {final_decoder_params:,}")
    print()
    
    print(f"VAE总参数数: {total_params:,}")
    print(f"参数数量级: {total_params/1e6:.1f} 百万")
    
    return total_params


def compare_with_pod():
    """与POD方法的参数数量对比"""
    
    print(f"\n=== 与POD方法对比 ===")
    
    # POD参数数量
    input_dim = 8281
    n_modes_list = [5, 10, 15, 20]
    
    print("POD参数数量:")
    for n_modes in n_modes_list:
        # POD模态矩阵: input_dim × n_modes
        pod_params = input_dim * n_modes
        print(f"  {n_modes}模态: {pod_params:,} 参数 ({pod_params/1e3:.1f}K)")
    
    print("\nAutoencoder vs POD 参数数量对比:")
    print("-" * 50)
    
    # 计算不同潜在维度下的AE参数数量
    latent_dims = [5, 10, 15, 20]
    
    for latent_dim in latent_dims:
        # 简化计算（只考虑主要的Linear层）
        hidden_dims = [4096, 2048, 1024, 512, 256]
        
        # 编码器
        encoder_params = 8281 * 4096  # 第一层最大
        encoder_params += sum(hidden_dims[i] * hidden_dims[i+1] for i in range(len(hidden_dims)-1))
        encoder_params += hidden_dims[-1] * latent_dim
        
        # 解码器（对称结构）
        decoder_params = encoder_params - (8281 * 4096 - 4096 * 8281)  # 大致相同
        
        total_ae_params = encoder_params + decoder_params
        
        # 对应的POD参数
        pod_params = 8281 * latent_dim
        
        ratio = total_ae_params / pod_params
        
        print(f"维度 {latent_dim}:")
        print(f"  Autoencoder: {total_ae_params/1e6:.1f}M 参数")
        print(f"  POD: {pod_params/1e3:.1f}K 参数")
        print(f"  比例: {ratio:.0f}:1")
        print()


def analyze_why_so_many_parameters():
    """分析为什么参数这么多"""
    
    print(f"\n=== 为什么自编码器参数这么多？ ===")
    
    input_dim = 8281
    first_hidden = 4096
    
    # 第一层就占了大部分参数
    first_layer_params = input_dim * first_hidden + first_hidden
    
    print(f"1. 高维输入数据:")
    print(f"   RCS数据维度: {input_dim} (91×91角度组合)")
    print(f"   这是一个非常高维的输入空间")
    print()
    
    print(f"2. 第一层参数占主导:")
    print(f"   第一层: {input_dim} → {first_hidden}")
    print(f"   参数数量: {first_layer_params:,} ({first_layer_params/1e6:.1f}M)")
    print(f"   这一层就占了总参数的很大比例")
    print()
    
    print(f"3. 深层网络结构:")
    print(f"   编码器: 8281 → 4096 → 2048 → 1024 → 512 → 256 → latent_dim")
    print(f"   解码器: latent_dim → 256 → 512 → 1024 → 2048 → 4096 → 8281")
    print(f"   每一层都需要大量参数来学习非线性映射")
    print()
    
    print(f"4. 与传统降维方法对比:")
    print(f"   POD: 线性降维，参数数量 = 输入维度 × 模态数")
    print(f"   Autoencoder: 非线性降维，需要多层网络学习复杂映射")
    print()
    
    print(f"5. 参数数量分解 (Standard AE, latent_dim=10):")
    # 重新计算更精确的参数数量
    layers = [
        (8281, 4096, "输入→隐藏1"),
        (4096, 2048, "隐藏1→隐藏2"), 
        (2048, 1024, "隐藏2→隐藏3"),
        (1024, 512, "隐藏3→隐藏4"),
        (512, 256, "隐藏4→隐藏5"),
        (256, 10, "隐藏5→潜在"),
        (10, 256, "潜在→隐藏5'"),
        (256, 512, "隐藏5'→隐藏4'"),
        (512, 1024, "隐藏4'→隐藏3'"),
        (1024, 2048, "隐藏3'→隐藏2'"),
        (2048, 4096, "隐藏2'→隐藏1'"),
        (4096, 8281, "隐藏1'→输出")
    ]
    
    total = 0
    for in_dim, out_dim, name in layers:
        params = in_dim * out_dim + out_dim  # weights + bias
        total += params
        print(f"   {name}: {params:,} 参数 ({params/1e6:.2f}M)")
    
    print(f"   总计: {total:,} 参数 ({total/1e6:.1f}M)")


def main():
    """主函数"""
    print("自编码器模型参数数量分析")
    print("=" * 60)
    
    # 分析Standard Autoencoder
    std_params = analyze_standard_autoencoder_params()
    
    # 分析VAE
    vae_params = analyze_vae_params()
    
    # 与POD对比
    compare_with_pod()
    
    # 分析原因
    analyze_why_so_many_parameters()
    
    print(f"\n=== 总结 ===")
    print(f"Standard Autoencoder: {std_params/1e6:.1f}M 参数")
    print(f"VAE: {vae_params/1e6:.1f}M 参数")
    print(f"")
    print(f"参数数量庞大的原因:")
    print(f"1. 高维输入 (8281维)")
    print(f"2. 深层网络结构 (6层编码器 + 6层解码器)")
    print(f"3. 第一层参数占主导 (8281×4096 ≈ 34M)")
    print(f"4. 非线性映射需要更多参数")


if __name__ == "__main__":
    main()