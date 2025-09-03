import os
import re
camera_info={}

# 定义两个数组分别用于存储两个希望的 model 的 video number
inf_numbers = []
rgb_numbers = []

# 手动配置选项 - 如果自动检测失败，可以手动指定
MANUAL_CONFIG = True  # 设置为True启用手动配置

if MANUAL_CONFIG:
    # 手动配置摄像头编号（根据实际测试结果调整）
    # 根据诊断结果，/dev/video2 可以打开，但可能无法读取帧
    # 尝试使用其他可用的设备编号
    rgb_numbers = [0, 1, 2, 3]  # 尝试所有可能的编号
    inf_numbers = []  # 暂时禁用红外检测
    print("🔧 使用手动配置的摄像头编号")
else:
    # 自动检测逻辑
    for device in os.listdir('/sys/class/video4linux/'):
        try:
            # 读取设备的硬件信息
            with open(f"/sys/class/video4linux/{device}/device/modalias", "r") as f:
                modalias = f.read().strip()
                # 从 modalias 中提取设备厂商、型号等信息
                manufacturer, model = modalias.split(":")
                print(model)
                # 如果 model 是红外摄像
                if model == 'v1514p0001d0200dcEFdsc02dp01ic0Eisc01ip00in00':
                    # 使用正则表达式提取数字部分
                    video_number = re.search(r'\d+', device).group()
                    inf_numbers.append(int(video_number))
                #model是普通usb摄像
                elif model == 'v0C45p636Bd0100dcEFdsc02dp01ic0Eisc01ip00in00':
                    # 使用正则表达式提取数字部分
                    video_number = re.search(r'\d+', device).group()
                    rgb_numbers.append(int(video_number))
        except FileNotFoundError:
            pass

print(f"inf Video Numbers: {inf_numbers}")
print(f"rgb Video Numbers: {rgb_numbers}")

# 添加摄像头测试函数
def test_camera(device_num):
    """测试摄像头是否可用"""
    try:
        import cv2
        cap = cv2.VideoCapture(device_num)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return True
        return False
    except:
        return False

# 如果手动配置启用，测试并过滤可用的摄像头
if MANUAL_CONFIG:
    print("\n🧪 测试手动配置的摄像头...")
    working_rgb = []
    for num in rgb_numbers:
        if test_camera(num):
            working_rgb.append(num)
            print(f"✅ /dev/video{num} 可用")
        else:
            print(f"❌ /dev/video{num} 不可用")
    
    if working_rgb:
        rgb_numbers = working_rgb
        print(f"🎉 找到 {len(working_rgb)} 个可用的RGB摄像头: {working_rgb}")
    else:
        print("⚠️ 没有找到可用的RGB摄像头，使用原始配置")
        rgb_numbers = [0, 1, 2, 3]

