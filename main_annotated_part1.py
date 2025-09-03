# ============================================================================
# 智能视觉检测系统 - 主控制模块 (main.py) - 第一部分
# 功能：系统的主控制器，负责任务调度、摄像头管理、UDP通信
# ============================================================================

# ============================================================================
# 第一部分：系统导入和依赖
# ============================================================================
import threading          # 多线程支持，用于并发任务处理
import queue             # 队列管理，用于线程间数据传递
import cv2               # OpenCV库，用于图像处理和摄像头操作
import func_face         # 人脸识别功能模块
import func_v7           # 红外入侵检测功能模块
import video_number      # 摄像头编号管理模块
from rknnpool_rgb import rknnPoolExecutor_face      # RGB摄像头人脸识别模型池
from func_face import myFunc_face                   # 人脸识别处理函数
from rknnpool_inf import rknnPoolExecutor_inf       # 红外摄像头模型池
from func_v7 import myFunc_inf, data_ten_inf        # 红外入侵检测处理函数
from rknnpool_smoke import rknnPoolExecutor_smoke_hat  # 吸烟+安全帽检测模型池
from func_smoke_hat import myFunc_smoke_hat         # 吸烟+安全帽检测处理函数
from rknnpool_flame import rknnPoolExecutor_flame   # 火焰检测模型池
from func_flame import myFunc_flame                 # 火焰检测处理函数
from rknnpool_hardhat import rknnPoolExecutor_hardhat  # 安全帽检测模型池
from rknnpool_smoke_single import rknnPoolExecutor_smoke  # 单独吸烟检测模型池
from func_hardhat import myFunc_hardhat             # 安全帽检测处理函数
from func_smoke import myFunc_smoke                 # 吸烟检测处理函数
import time              # 时间管理，用于性能统计和延时控制
import socket            # 网络通信，用于UDP协议实现
import struct            # 二进制数据打包/解包，用于UDP数据帧处理

# ============================================================================
# 第二部分：全局变量管理
# ============================================================================
# 全局状态变量 - 存储各检测任务的结果
classes_data = []        # 存储检测到的类别数据（用于多类别检测）
class_inf = 0            # 红外入侵检测结果：0=无入侵，1=有入侵
class_flame = None       # 火焰检测结果：None=未检测，0=无火焰，1=有火焰
class_smoke = None       # 吸烟检测结果：None=未检测，0=无吸烟，1=有吸烟
class_hardhat = None     # 安全帽检测结果：None=未检测，0=戴安全帽，1=未戴安全帽

# 任务代号映射表 - 数字代号到任务名称的映射
TASK_CODES = {
    3: '红外入侵检测',    # 使用红外摄像头检测人体入侵
    4: '火焰检测',        # 使用RGB摄像头检测火焰
    0: '人脸识别',        # 使用RGB摄像头进行人脸识别
    1: '仪表检测',        # 使用RGB摄像头检测仪表读数
    2: '安全帽检测',      # 使用RGB摄像头检测安全帽佩戴
    5: '吸烟检测'         # 使用RGB摄像头检测吸烟行为
}

# ============================================================================
# 第三部分：UDP通信协议常量定义
# ============================================================================
# 第一位 - 请求类型（控制帧的第一个字段）
REQUEST_DATA = 0         # 请求数据：客户端向服务器请求特定数据
GIVE_DATA = 1            # 提供数据：服务器主动向客户端推送数据
CONTROL_COMMOND = 2      # 控制命令：客户端向服务器发送控制指令

# 第二位与第三位 - 设备类型（控制帧的第二、三个字段）
SERVER = 0               # 服务器设备
VISION = 1               # 视觉系统设备
ROBOT = 2                # 机器人设备
METER = 3                # 仪表设备

# 第四位 - 子任务类型（控制帧的第四个字段）
SUB_FACE = 0             # 人脸识别子任务
SUB_METER = 1            # 仪表检测子任务
SUB_HARDHAT = 2          # 安全帽检测子任务
SUB_INF = 3              # 红外入侵检测子任务
SUB_FLAME = 4            # 火焰检测子任务
SUB_THR = 5              # 温度检测子任务
SUB_GAS = 6              # 气体检测子任务
SUB_SMOKE = 7            # 吸烟检测子任务

# 第五位 - 状态标识（控制帧的第五个字段）
EXIST = 1                # 存在/检测到目标
NOT_EXIST = 0            # 不存在/未检测到目标
FINSH = 3                # 完成状态（无关状态）

# 仪表检测数据帧第六位：数据类型
METER_TYPE_XIAOFANG = 0  # 消防仪表类型
METER_TYPE_AIR = 1       # 空气仪表类型
METER_TYPE_JIXIE = 2     # 机械仪表类型

# 抽烟安全帽数据帧第六位：检测结果
HAT = 1                  # 戴安全帽
NO_HAT = 0               # 未戴安全帽
SMOKE = 3                # 检测到吸烟
NOT_SMOKE = 2            # 未检测到吸烟

# 仪表检测控制帧第六位：控制操作
CLOSE_ALL = 0            # 关闭所有任务
START = 1                # 启动任务
CLOSE = 2                # 关闭任务

# UDP数据帧格式定义：9个字段的二进制数据包
# I = 4字节整数，f = 4字节浮点数
frame_format_meter = 'IIIIIIIff'  # 7个整数 + 2个浮点数


# 任务状态管理变量
state_env = False        # 人脸识别任务状态
state_inf = False        # 红外入侵检测任务状态
state_flame = False      # 火焰检测任务状态
state_smoke = False      # 吸烟检测任务状态
state_hardhat = False    # 安全帽检测任务状态
