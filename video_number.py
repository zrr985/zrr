import os          # 操作系统接口，用于文件系统操作
import re          # 正则表达式模块，用于字符串匹配和提取
camera_info = {}   # 摄像头信息字典（当前未使用，预留扩展）

# 摄像头编号存储数组
# 定义两个数组分别用于存储两种类型摄像头的设备编号
inf_numbers = []   # 红外摄像头编号列表
rgb_numbers = []   # RGB摄像头编号列表


# 遍历/sys/class/video4linux/目录下的所有设备
for device in os.listdir('/sys/class/video4linux/'):
    try:
        # 读取设备的硬件信息文件
        # modalias文件包含设备的厂商、型号等硬件标识信息
        with open(f"/sys/class/video4linux/{device}/device/modalias", "r") as f:
            modalias = f.read().strip()  # 读取并去除首尾空白字符
            # 从modalias字符串中提取设备厂商和型号信息
            # modalias格式：vendor:model:subsystem_vendor:subsystem_device
            manufacturer, model = modalias.split(":")  # 分割字符串，获取厂商和型号
            print(model)  # 打印设备型号，用于调试

            if model == 'v1514p0001d0200dcEFdsc02dp01ic0Eisc01ip00in00':
                # 使用正则表达式从设备名称中提取数字部分
                # 设备名称格式：videoX，其中X是数字编号
                video_number = re.search(r'\d+', device).group()  # 提取数字部分
                inf_numbers.append(int(video_number))  # 将编号转换为整数并添加到红外摄像头列表

            elif model == 'v0C45p636Bd0100dcEFdsc02dp01ic0Eisc01ip00in00':
                # 使用正则表达式从设备名称中提取数字部分
                video_number = re.search(r'\d+', device).group()  # 提取数字部分
                rgb_numbers.append(int(video_number))  # 将编号转换为整数并添加到RGB摄像头列表
                
    except FileNotFoundError:
        # 如果modalias文件不存在，跳过该设备
        # 这种情况可能发生在设备权限不足或设备信息不完整时
        pass  # 静默忽略错误，继续处理下一个设备


# 打印识别到的摄像头编号列表
print(f"inf Video Numbers: {inf_numbers}")   # 输出红外摄像头编号列表
print(f"rgb Video Numbers: {rgb_numbers}")   # 输出RGB摄像头编号列表



