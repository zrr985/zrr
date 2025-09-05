#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务检测系统配置文件 - 简化版本
只保留实际使用的参数配置
"""

# ==================== 任务配置 ====================

# 火焰检测配置
FLAME_CONFIG = {
    'model_path': './fire.rknn',                   # 模型文件路径
    'abnormal_threshold': 10,                       # 连续异常帧阈值（帧数）
    'window_title': '火焰检测',                      # 显示窗口标题
}

# 人脸识别配置
FACE_CONFIG = {
    'model_path': 'model_data/retinaface_mob.rknn', # 人脸检测模型路径
    'model_path2': 'model_data/mobilefacenet.rknn', # 人脸特征提取模型路径
    'window_title': '人脸识别',                      # 显示窗口标题
}

# 仪表读数配置
METER_CONFIG = {
    'window_title': '仪表检测',                      # 显示窗口标题
}

# 安全帽检测配置
HARDHAT_CONFIG = {
    'model_path': './helmet.rknn',                 # 模型文件路径
    'abnormal_threshold': 10,                       # 连续异常帧阈值（帧数）
    'window_title': '安全帽检测',                    # 显示窗口标题
}

# 吸烟检测配置
SMOKING_CONFIG = {
    'model_path': './smoking.rknn',                # 模型文件路径
    'abnormal_threshold': 20,                       # 连续异常帧阈值（帧数）
    'window_title': '吸烟检测',                      # 显示窗口标题
}

# ==================== 配置验证函数 ====================

def load_config():
    """加载配置（简化版本）"""
    return True

def print_config_summary():
    """打印配置摘要（简化版本）"""
    print("配置摘要:")
    print(f"• 火焰检测: {FLAME_CONFIG['model_path']}")
    print(f"• 人脸识别: {FACE_CONFIG['model_path']}")
    print(f"• 仪表检测: 使用yolov7_tiny.rknn")
    print(f"• 安全帽检测: {HARDHAT_CONFIG['model_path']}")
    print(f"• 吸烟检测: {SMOKING_CONFIG['model_path']}")

def validate_config():
    """验证配置的有效性（简化版本）"""
    import os
    
    # 检查模型文件是否存在
    model_files = [
        FLAME_CONFIG['model_path'],
        FACE_CONFIG['model_path'],
        FACE_CONFIG['model_path2'],
        HARDHAT_CONFIG['model_path'],
        SMOKING_CONFIG['model_path']
    ]
    
    missing_files = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_files.append(model_file)
    
    if missing_files:
        print("警告: 以下模型文件不存在:")
        for file in missing_files:
            print(f"  • {file}")
        return False
    
    return True