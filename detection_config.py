#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务检测系统配置文件
用户可以在此文件中修改各种检测参数
"""

# ==================== 全局配置 ====================

# 线程池大小配置
THREAD_POOL_SIZE = 3                               # RKNN模型线程池大小，可根据硬件性能调整

# 队列大小配置
FRAME_QUEUE_SIZE = 5                               # 帧队列大小，影响内存使用和延迟
DISPLAY_QUEUE_SIZE = 10                            # 显示队列大小

# 帧率统计间隔
FPS_REPORT_INTERVAL = 30                           # 每N帧报告一次帧率

# 状态报告间隔
STATUS_REPORT_INTERVAL = 30                        # 每N秒报告一次系统状态

# ==================== 任务特定配置 ====================

# 红外入侵检测配置
INFRARED_CONFIG = {
    'model_path': './yolov7_tiny-a.rknn',          # 模型文件路径
    'abnormal_threshold': 10,                       # 连续异常帧阈值（帧数）
    'confidence_threshold': 0.5,                    # 检测置信度阈值
    'window_title': '红外入侵检测',                  # 显示窗口标题
    'enable_debug': True,                           # 是否启用调试输出
    'alert_enabled': True,                          # 是否启用报警
}

# 火焰检测配置
FLAME_CONFIG = {
    'model_path': './fire.rknn',                   # 模型文件路径
    'abnormal_threshold': 10,                       # 连续异常帧阈值（帧数）
    'confidence_threshold': 0.3,                    # 检测置信度阈值
    'window_title': '火焰检测',                      # 显示窗口标题
    'enable_debug': False,                          # 是否启用调试输出
    'alert_enabled': True,                          # 是否启用报警
}

# 人脸识别配置
FACE_CONFIG = {
    'model_path': 'model_data/retinaface_mob.rknn', # 人脸检测模型路径
    'model_path2': 'model_data/mobilefacenet.rknn', # 人脸特征提取模型路径
    'face_threshold': 0.9,                          # 人脸识别相似度阈值
    'detection_threshold': 0.5,                     # 人脸检测置信度阈值
    'window_title': '人脸识别',                      # 显示窗口标题
    'enable_debug': False,                          # 是否启用调试输出
    'show_unknown': True,                           # 是否显示未知人脸
}

# 仪表读数配置
METER_CONFIG = {
    'model_path': './yolov8_meter.rknn',           # 模型文件路径
    'pointer_threshold': 0.5,                       # 指针检测置信度阈值
    'scale_threshold': 0.5,                         # 刻度检测置信度阈值
    'window_title': '仪表检测',                      # 显示窗口标题
    'enable_debug': False,                          # 是否启用调试输出
    'show_intermediate_results': False,             # 是否显示中间处理结果
    'reading_precision': 3,                         # 读数显示精度（小数位数）
}

# 安全帽检测配置
HARDHAT_CONFIG = {
    'model_path': './helmet.rknn',                 # 模型文件路径
    'abnormal_threshold': 10,                       # 连续异常帧阈值（帧数）
    'confidence_threshold': 0.25,                   # 检测置信度阈值
    'window_title': '安全帽检测',                    # 显示窗口标题
    'enable_debug': False,                          # 是否启用调试输出
    'alert_enabled': True,                          # 是否启用报警
}

# 吸烟检测配置
SMOKING_CONFIG = {
    'model_path': './smoking.rknn',                # 模型文件路径
    'abnormal_threshold': 20,                       # 连续异常帧阈值（帧数）- 吸烟检测更严格
    'confidence_threshold': 0.25,                   # 检测置信度阈值
    'window_title': '吸烟检测',                      # 显示窗口标题
    'enable_debug': False,                          # 是否启用调试输出
    'alert_enabled': True,                          # 是否启用报警
    'multi_class_detection': True,                  # 是否启用多类别检测（香烟、人脸、吸烟动作）
}

# ==================== 摄像头配置 ====================

# 摄像头重试配置
CAMERA_CONFIG = {
    'retry_times': 3,                               # 摄像头打开失败重试次数
    'retry_delay': 1.0,                             # 重试间隔（秒）
    'frame_width': None,                            # 帧宽度（None为默认）
    'frame_height': None,                           # 帧高度（None为默认）
    'fps': None,                                    # 帧率（None为默认）
}

# ==================== 显示配置 ====================

# 文本显示配置
TEXT_CONFIG = {
    'font': 'cv2.FONT_HERSHEY_SIMPLEX',            # 字体类型
    'font_scale': 1.0,                              # 字体大小
    'thickness': 2,                                 # 字体粗细
    'line_spacing': 40,                             # 行间距
    'margin_x': 10,                                 # 左边距
    'margin_y': 30,                                 # 上边距
}

# 颜色配置（BGR格式）
COLOR_CONFIG = {
    'normal': (0, 255, 0),                          # 正常状态颜色（绿色）
    'warning': (0, 255, 255),                       # 警告状态颜色（黄色）
    'danger': (0, 0, 255),                          # 危险状态颜色（红色）
    'info': (255, 255, 255),                        # 信息文本颜色（白色）
    'background': (0, 0, 0),                        # 背景颜色（黑色）
}

# ==================== 报警配置 ====================

# 报警系统配置
ALERT_CONFIG = {
    'enable_sound': False,                          # 是否启用声音报警
    'enable_log': True,                             # 是否启用日志记录
    'log_file': 'detection_alerts.log',            # 日志文件名
    'enable_email': False,                          # 是否启用邮件报警
    'email_config': {                               # 邮件配置
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_password',
        'to_email': 'alert@company.com',
    }
}

# ==================== 性能配置 ====================

# 性能优化配置
PERFORMANCE_CONFIG = {
    'enable_gpu': False,                            # 是否启用GPU加速
    'max_threads': None,                            # 最大线程数（None为自动）
    'memory_limit': None,                           # 内存限制（MB，None为无限制）
    'enable_profiling': False,                      # 是否启用性能分析
}

# ==================== 调试配置 ====================

# 调试配置
DEBUG_CONFIG = {
    'enable_global_debug': False,                   # 全局调试开关
    'save_debug_images': False,                     # 是否保存调试图像
    'debug_image_path': './debug_images/',          # 调试图像保存路径
    'print_fps': True,                              # 是否打印帧率信息
    'print_detection_results': True,               # 是否打印检测结果
    'verbose_logging': False,                       # 是否启用详细日志
}

# ==================== 用户自定义配置示例 ====================

def get_custom_config():
    """
    用户自定义配置函数
    用户可以在这里根据具体需求修改配置
    """
    
    # 示例：如果是测试环境，降低阈值以便更容易触发检测
    import os
    if os.getenv('DETECTION_MODE') == 'test':
        INFRARED_CONFIG['abnormal_threshold'] = 3
        FLAME_CONFIG['abnormal_threshold'] = 3
        HARDHAT_CONFIG['abnormal_threshold'] = 3
        SMOKING_CONFIG['abnormal_threshold'] = 5
        print("🧪 测试模式：降低检测阈值")
    
    # 示例：如果是演示环境，启用所有调试信息
    if os.getenv('DETECTION_MODE') == 'demo':
        DEBUG_CONFIG['enable_global_debug'] = True
        DEBUG_CONFIG['print_detection_results'] = True
        for config in [INFRARED_CONFIG, FLAME_CONFIG, FACE_CONFIG, 
                      METER_CONFIG, HARDHAT_CONFIG, SMOKING_CONFIG]:
            config['enable_debug'] = True
        print("🎭 演示模式：启用调试信息")
    
    # 示例：根据硬件性能调整线程池大小
    import psutil
    cpu_count = psutil.cpu_count()
    if cpu_count >= 8:
        global THREAD_POOL_SIZE
        THREAD_POOL_SIZE = 4
        print(f"🚀 高性能模式：线程池大小设为 {THREAD_POOL_SIZE}")
    elif cpu_count <= 2:
        THREAD_POOL_SIZE = 1
        print(f"💡 节能模式：线程池大小设为 {THREAD_POOL_SIZE}")

# ==================== 配置验证函数 ====================

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 验证文件路径
    import os
    model_paths = [
        INFRARED_CONFIG['model_path'],
        FLAME_CONFIG['model_path'],
        FACE_CONFIG['model_path'],
        FACE_CONFIG['model_path2'],
        METER_CONFIG['model_path'],
        HARDHAT_CONFIG['model_path'],
        SMOKING_CONFIG['model_path']
    ]
    
    for path in model_paths:
        if not os.path.exists(path):
            errors.append(f"模型文件不存在: {path}")
    
    # 验证数值范围
    if not (1 <= THREAD_POOL_SIZE <= 10):
        errors.append(f"线程池大小超出合理范围: {THREAD_POOL_SIZE}")
    
    if not (1 <= FRAME_QUEUE_SIZE <= 50):
        errors.append(f"帧队列大小超出合理范围: {FRAME_QUEUE_SIZE}")
    
    # 验证阈值
    for config_name, config in [
        ('INFRARED_CONFIG', INFRARED_CONFIG),
        ('FLAME_CONFIG', FLAME_CONFIG),
        ('HARDHAT_CONFIG', HARDHAT_CONFIG),
        ('SMOKING_CONFIG', SMOKING_CONFIG)
    ]:
        if 'abnormal_threshold' in config:
            if not (1 <= config['abnormal_threshold'] <= 100):
                errors.append(f"{config_name}异常阈值超出合理范围: {config['abnormal_threshold']}")
    
    return errors

# ==================== 配置加载函数 ====================

def load_config():
    """加载并应用配置"""
    print("📋 加载检测系统配置...")
    
    # 应用用户自定义配置
    try:
        get_custom_config()
    except Exception as e:
        print(f"⚠️ 应用自定义配置时出错: {e}")
    
    # 验证配置
    errors = validate_config()
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    print("✅ 配置加载完成")
    return True

# ==================== 配置信息显示 ====================

def print_config_summary():
    """打印配置摘要"""
    print("\n" + "="*60)
    print("📋 检测系统配置摘要")
    print("="*60)
    print(f"🔧 线程池大小: {THREAD_POOL_SIZE}")
    print(f"📦 队列配置: 帧队列={FRAME_QUEUE_SIZE}, 显示队列={DISPLAY_QUEUE_SIZE}")
    print(f"📊 报告间隔: FPS={FPS_REPORT_INTERVAL}帧, 状态={STATUS_REPORT_INTERVAL}秒")
    print("\n🎯 任务配置:")
    
    configs = [
        ('红外检测', INFRARED_CONFIG),
        ('火焰检测', FLAME_CONFIG),
        ('人脸识别', FACE_CONFIG),
        ('仪表检测', METER_CONFIG),
        ('安全帽检测', HARDHAT_CONFIG),
        ('吸烟检测', SMOKING_CONFIG)
    ]
    
    for name, config in configs:
        threshold = config.get('abnormal_threshold', 'N/A')
        debug = '开启' if config.get('enable_debug', False) else '关闭'
        print(f"   • {name}: 阈值={threshold}, 调试={debug}")
    
    print("="*60)

if __name__ == "__main__":
    # 测试配置加载
    if load_config():
        print_config_summary()
    else:
        print("❌ 配置加载失败，请检查配置文件")
