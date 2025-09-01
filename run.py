#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行脚本：提供简单的命令行界面，用于执行RCS的POD分析
"""

import os
import sys
import argparse
import time
from main import main

# Windows编码设置
if sys.platform.startswith('win'):
    import locale
    try:
        # 尝试设置UTF-8输出
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCS数据的POD和模态分析")

    parser.add_argument("--params_path", type=str, default="../parameter/parameters_sorted.csv",
                        help="设计参数CSV文件路径")

    parser.add_argument("--rcs_dir", type=str, default="../parameter/csv_output",
                        help="RCS数据CSV文件目录")

    parser.add_argument("--output_dir", type=str, default="./results",
                        help="分析结果输出目录")

    parser.add_argument("--freq", type=str, choices=["1.5G", "3G", "both"], default="both",
                        help="要分析的频率（1.5G, 3G, 或both）")

    parser.add_argument("--num_models", type=int, default=100,
                        help="要分析的模型数量")

    parser.add_argument("--num_train", type=str, default="80",
                        help="训练集大小,可以是单个数字或用逗号分隔的列表(如'60,70,80')")

    parser.add_argument("--predict_mode", action="store_true",
                        help="启用预测模式,根据设计参数预测RCS数据")

    parser.add_argument("--param_file", type=str, default=None,
                        help="预测模式下的设计参数文件路径")

    # POD分析相关参数
    parser.add_argument("--pod_modes", type=str, default="10,20,30,40",
                        help="POD模态数量列表,用逗号分隔(如'10,20,30,40')")

    # 自编码器相关参数
    parser.add_argument("--latent_dims", type=str, default="5,10,15,20",
                        help="自编码器潜在空间维度,用逗号分隔(如'5,10,15,20')")
    
    parser.add_argument("--model_types", type=str, default="standard,vae",
                        help="自编码器模型类型,用逗号分隔(如'standard,vae')")
    
    parser.add_argument("--skip_ae_training", action="store_true",
                        help="跳过Autoencoder重训练，优先使用已有模型")
    
    parser.add_argument("--ae_epochs", type=int, default=200,
                        help="自编码器训练轮数")
    
    parser.add_argument("--ae_device", type=str, default="auto",
                        help="自编码器计算设备(auto,cpu,cuda)")
    
    parser.add_argument("--ae_learning_rate", type=float, default=0.001,
                        help="自编码器学习率")
    
    parser.add_argument("--ae_batch_size", type=int, default=0,
                        help="自编码器批次大小(0表示自动)")

    # POD相关参数
    parser.add_argument("--energy_threshold", type=float, default=95.0,
                        help="POD能量阈值百分比")
    
    parser.add_argument("--num_modes_visualize", type=int, default=10,
                        help="可视化的POD模态数量")
    
    parser.add_argument("--pod_reconstruct_num", type=int, default=0,
                        help="POD重建使用的模态数量(0表示使用能量阈值确定)")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    # 解析自编码器参数
    latent_dims = [int(x.strip()) for x in args.latent_dims.split(',') if x.strip()]
    model_types = [x.strip() for x in args.model_types.split(',') if x.strip()]
    
    # 解析POD模态数量参数
    pod_modes = [int(x.strip()) for x in args.pod_modes.split(',') if x.strip()]

    print(f"开始RCS数据的POD和模态分析...")
    print(f"参数文件: {args.params_path}")
    print(f"RCS数据目录: {args.rcs_dir}")
    print(f"分析频率: {args.freq}")
    print(f"模型数量: {args.num_models}")
    print(f"输出目录: {args.output_dir}")
    print(f"自编码器潜在维度: {latent_dims}")
    print(f"自编码器模型类型: {model_types}")
    print(f"自编码器训练轮数: {args.ae_epochs}")
    print(f"自编码器计算设备: {args.ae_device}")
    print(f"自编码器学习率: {args.ae_learning_rate}")
    print(f"自编码器批次大小: {args.ae_batch_size if args.ae_batch_size > 0 else '自动'}")
    print(f"POD能量阈值: {args.energy_threshold}%")
    print(f"可视化模态数量: {args.num_modes_visualize}")
    print(f"POD重建模态数: {args.pod_reconstruct_num if args.pod_reconstruct_num > 0 else '使用能量阈值确定'}")
    print(f"POD多模态对比: {pod_modes}")
    print("-" * 50)

    # 执行主程序
    main(params_path=args.params_path,
         rcs_dir=args.rcs_dir,
         output_dir=args.output_dir,
         analyze_freq=args.freq,
         num_models=args.num_models,
         num_train=args.num_train,
         predict_mode=args.predict_mode,
         param_file=args.param_file,
         latent_dims=latent_dims,
         model_types=model_types,
         skip_ae_training=args.skip_ae_training,
         ae_epochs=args.ae_epochs,
         ae_device=args.ae_device,
         ae_learning_rate=args.ae_learning_rate,
         ae_batch_size=args.ae_batch_size,
         energy_threshold=args.energy_threshold,
         num_modes_visualize=args.num_modes_visualize,
         pod_reconstruct_num=args.pod_reconstruct_num,
         pod_modes=pod_modes)

    # 计算运行时间
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print("-" * 50)
    print(f"分析完成！")
    print(f"总运行时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"结果保存在: {args.output_dir}")