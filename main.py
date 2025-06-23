#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAR Loss: Reducing Semantic Ambiguity in Facial Landmark Detection
主程序入口
"""

import argparse
import os
import sys
from trainer import train
from tester import test, test_img_only


def add_base_options(parser):
    """添加基础配置选项"""
    group = parser.add_argument_group("基础配置")
    group.add_argument("--mode", type=str, default="test_image", 
                      choices=["train", "test", "test_image"],
                      help="运行模式: train(训练), test(测试), test_image(图像测试)")
    group.add_argument("--config_name", type=str, default="alignment", 
                      help="配置文件名称")
    group.add_argument('--device_ids', type=str, default="0",
                      help="设备ID, -1表示CPU, >=0表示GPU")
    group.add_argument('--data_definition', type=str, default='video_me_pixar_d_0.5_80',
                      help="数据集定义")
    group.add_argument('--learn_rate', type=float, default=0.001, 
                      help='学习率')
    group.add_argument("--batch_size", type=int, default=1, 
                      help="批次大小")
    group.add_argument('--width', type=int, default=256, 
                      help='输入图像宽度')
    group.add_argument('--height', type=int, default=256, 
                      help='输入图像高度')


def add_data_options(parser):
    """添加数据相关选项"""
    group = parser.add_argument_group("数据配置")
    group.add_argument("--image_dir", type=str, 
                      default='./dataset/001/0.9_slamdunk/', 
                      help="图像目录路径")
    group.add_argument("--annot_dir", type=str, default='./dataset/meta', 
                      help="标注文件目录路径")


def add_train_options(parser):
    """添加训练相关选项"""
    group = parser.add_argument_group('训练配置')
    group.add_argument("--train_num_workers", type=int, default=None, 
                      help="训练时的工作线程数")
    group.add_argument('--loss_func', type=str, default='STARLoss_v2', 
                      help="损失函数类型")
    group.add_argument("--val_batch_size", type=int, default=None, 
                      help="验证时的批次大小")
    group.add_argument("--val_num_workers", type=int, default=None, 
                      help="验证时的工作线程数")


def add_eval_options(parser):
    """添加评估相关选项"""
    group = parser.add_argument_group("评估配置")
    group.add_argument("--pretrained_weight", type=str, 
                      default='./pretrained_model/cartoon_dataset_400_v2/starv2_smoothl1/model/best_model.pkl',
                      help="预训练模型路径")
    group.add_argument('--norm_type', type=str, default='default', 
                      choices=['default', 'ocular', 'pupil'],
                      help='标准化类型')
    group.add_argument('--test_file', type=str, default="test.tsv", 
                      help='测试文件名')


def add_starloss_options(parser):
    """添加STAR损失相关选项"""
    group = parser.add_argument_group('STAR损失配置')
    group.add_argument('--star_w', type=float, default=1, 
                      help="正则化损失权重")
    group.add_argument('--star_dist', type=str, default='smoothl1', 
                      help='STAR损失距离函数')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="STAR Loss人脸关键点检测系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加所有选项组
    add_base_options(parser)
    add_data_options(parser)
    add_train_options(parser)
    add_eval_options(parser)
    add_starloss_options(parser)

    args = parser.parse_args()

    # 打印配置信息
    print(f"运行模式: {args.mode}")
    print(f"配置文件: {args.config_name}")
    print(f"预训练权重: {args.pretrained_weight}")
    print(f"图像目录: {args.image_dir}")
    print(f"标注目录: {args.annot_dir}")
    print(f"设备ID: {args.device_ids}")
    
    # 解析设备ID
    args.device_ids = list(map(int, args.device_ids.split(",")))
    
    # 根据模式执行相应功能
    try:
        if args.mode == "train":
            print("开始训练...")
            train(args)
        elif args.mode == "test":
            print("开始测试...")
            test(args)
        elif args.mode == 'test_image':
            print("开始图像测试...")
            test_img_only(args)
        else:
            print(f"未知的运行模式: {args.mode}")
            sys.exit(1)
            
        print("任务完成!")
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
