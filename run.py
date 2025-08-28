#!/usr/bin/env python
"""
运行脚本：提供简单的命令行界面，用于执行RCS的POD分析
"""

import os
import argparse
import time
from main import main

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

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    print(f"开始RCS数据的POD和模态分析...")
    print(f"参数文件: {args.params_path}")
    print(f"RCS数据目录: {args.rcs_dir}")
    print(f"分析频率: {args.freq}")
    print(f"模型数量: {args.num_models}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 50)

    # 执行主程序
    main(params_path=args.params_path,
         rcs_dir=args.rcs_dir,
         output_dir=args.output_dir,
         analyze_freq=args.freq,
         num_models=args.num_models,
         num_train=args.num_train,  # 添加这个参数
         predict_mode=args.predict_mode,  # 添加这个参数
         param_file=args.param_file)  # 添加这个参数

    # 计算运行时间
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print("-" * 50)
    print(f"分析完成！")
    print(f"总运行时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"结果保存在: {args.output_dir}")