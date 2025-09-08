# ============================================================================
# 智能视觉检测系统 - 主控制模块 (main.py)
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
SUB_THR = 4              # 温度检测子任务
SUB_GAS = 5              # 气体检测子任务
SUB_SMOKE = 6            # 吸烟检测子任务

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

# 环境状态标志
state_env = False        # 全局环境状态，用于控制任务运行状态

# ============================================================================
# 第四部分：摄像头管理功能
# ============================================================================
def open_camera(camera_numbers):
    """
    打开摄像头函数
    参数：camera_numbers - 摄像头编号列表
    返回：成功返回摄像头对象，失败返回None
    """
    for number in camera_numbers:  # 遍历摄像头编号列表
        cap = cv2.VideoCapture(number)  # 尝试打开指定编号的摄像头
        if cap.isOpened():  # 检查摄像头是否成功打开
            print(f"Found openable camera: {number}")  # 打印成功打开的摄像头编号
            return cap  # 返回摄像头对象
    return None  # 所有摄像头都无法打开时返回None

# 摄像头资源管理类
class CameraManager:
    """
    摄像头管理器类
    功能：管理红外和RGB摄像头的资源分配和释放
    特点：线程安全，支持双摄像头并发访问
    """
    
    def __init__(self):
        """初始化摄像头管理器"""
        self.infrared_cam = None   # 红外摄像头对象
        self.rgb_cam = None        # RGB摄像头对象
        self.lock = threading.Lock()  # 线程锁，防止多线程同时访问摄像头资源

    def get_infrared_camera(self):
        """
        获取红外摄像头
        返回：红外摄像头对象
        """
        with self.lock:  # 使用线程锁保护资源访问
            if self.infrared_cam is None:  # 如果红外摄像头未初始化
                # 从video_number模块获取红外摄像头编号列表并打开摄像头
                self.infrared_cam = open_camera(video_number.inf_numbers)
            return self.infrared_cam  # 返回红外摄像头对象

    def get_rgb_camera(self):
        """
        获取RGB摄像头
        返回：RGB摄像头对象
        """
        with self.lock:  # 使用线程锁保护资源访问
            if self.rgb_cam is None:  # 如果RGB摄像头未初始化
                # 从video_number模块获取RGB摄像头编号列表并打开摄像头
                self.rgb_cam = open_camera(video_number.rgb_numbers)
            return self.rgb_cam  # 返回RGB摄像头对象

    def release_cameras(self):
        """释放所有摄像头资源"""
        with self.lock:  # 使用线程锁保护资源释放
            if self.infrared_cam is not None:  # 如果红外摄像头存在
                self.infrared_cam.release()  # 释放红外摄像头资源
                self.infrared_cam = None     # 清空红外摄像头对象
            if self.rgb_cam is not None:     # 如果RGB摄像头存在
                self.rgb_cam.release()       # 释放RGB摄像头资源
                self.rgb_cam = None          # 清空RGB摄像头对象

    def release_inf(self):
        """仅释放红外摄像头资源"""
        with self.lock:  # 使用线程锁保护资源释放
            if self.infrared_cam is not None:  # 如果红外摄像头存在
                self.infrared_cam.release()  # 释放红外摄像头资源
                self.infrared_cam = None     # 清空红外摄像头对象

# ============================================================================
# 第五部分：摄像头帧捕获功能
# ============================================================================
def camera_capture(cam, frame_queue, stop_event):
    """
    摄像头帧捕获函数（生产者）
    功能：从摄像头持续捕获图像帧并放入队列
    参数：
        cam - 摄像头对象
        frame_queue - 帧队列，用于存储捕获的图像帧
        stop_event - 停止事件，用于控制线程停止
    """
    while not stop_event.is_set():  # 当停止事件未设置时持续运行
        ret, frame = cam.read()     # 从摄像头读取一帧图像
        if not ret:                 # 如果读取失败
            break                   # 退出循环
        if frame_queue.full():      # 如果队列已满
            frame_queue.get()       # 丢弃最老的帧，保持队列大小
        frame_queue.put(frame)      # 将新帧放入队列

# ============================================================================
# 第六部分：UDP通信和异常数据推送
# ============================================================================
# UDP服务器实例，用于发送数据
udp_server = None

def send_abnormal_data(task_code, state):
    """
    发送异常数据到服务器
    功能：当检测到异常时，通过UDP向服务器推送异常数据
    参数：
        task_code - 任务代号，标识异常类型
        state - 异常状态，1表示异常，0表示正常
    """
    global udp_server  # 使用全局UDP服务器实例
    try:
        if udp_server:  # 如果UDP服务器已初始化
            if task_code == 3:  # 红外入侵检测异常
                # 构建红外入侵检测数据帧：(请求类型, 设备类型, 子设备, 子任务, 状态, 预留, 预留, 预留, 预留)
                data_inf = (1, 1, 0, 3, state, 0, 0, 0, 0)
                pack_data_inf = struct.pack(frame_format_meter, *data_inf)  # 打包数据
                server_addr = ('127.0.0.1', 8111)  # 服务器地址和端口
                udp_server.send_response(server_addr, pack_data_inf)  # 发送数据
                print("发送红外入侵异常数据")
            elif task_code == 2:  # 安全帽检测异常
                # 构建安全帽检测数据帧
                data_hardhat = (1, 1, 0, 2, 1, state, 0, 0, 0)
                pack_data_hardhat = struct.pack(frame_format_meter, *data_hardhat)
                server_addr = ('127.0.0.1', 8111)
                udp_server.send_response(server_addr, pack_data_hardhat)
                print("发送安全帽异常数据")
            elif task_code == 5:  # 吸烟检测异常
                # 构建吸烟检测数据帧
                data_smoke = (1, 1, 0, 6, 1, state, 0, 0, 0)
                pack_data_smoke = struct.pack(frame_format_meter, *data_smoke)
                server_addr = ('127.0.0.1', 8111)
                udp_server.send_response(server_addr, pack_data_smoke)
                print("发送吸烟异常数据")
            elif task_code == 4:  # 火焰检测异常
                # 构建火焰检测数据帧
                data_flame = (1, 1, 0, 4, state, 1, 0, 0, 0)
                pack_data_flame = struct.pack(frame_format_meter, *data_flame)
                server_addr = ('127.0.0.1', 8111)
                udp_server.send_response(server_addr, pack_data_flame)
                print("发送火焰异常数据")
    except Exception as e:
        print(f"发送异常数据失败: task_code={task_code}, state={state}, error={e}")

# ============================================================================
# 第七部分：检测任务函数（以红外入侵检测为例）
# ============================================================================
def infrared_detection_task(frame_queue, display_queue, stop_event):
    """
    红外入侵检测任务函数
    功能：从红外摄像头获取图像，进行人体检测，发现异常时推送数据
    参数：
        frame_queue - 输入帧队列
        display_queue - 显示帧队列
        stop_event - 停止事件
    """
    print("红外入侵检测任务启动")
    global class_inf  # 使用全局变量存储检测结果
    
    # 初始化RKNN模型池
    model_path = "./yolov7_tiny-a.rknn"  # 红外入侵检测模型路径
    TPEs = 3  # 线程池执行器数量
    pool = rknnPoolExecutor_inf(  # 创建红外检测模型池
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_inf)

    # 性能统计变量
    frames, loopTime = 0, time.time()
    abnormal_count = 0      # 连续异常帧计数
    abnormal_pushed = False # 是否已推送异常标志

    # 主检测循环
    while not stop_event.is_set():  # 当停止事件未设置时持续运行
        try:
            frame = frame_queue.get(timeout=1)  # 从队列获取图像帧，超时1秒
            if frame is None:  # 如果获取的帧无效
                break          # 退出循环

            pool.put(frame)    # 将帧放入模型池进行处理
            result, flag = pool.get()  # 从模型池获取处理结果
            processed_frame, class_inf = result  # 解包结果：处理后的帧和检测类别
            print(f'now inf data {class_inf}')  # 打印当前检测结果
            
            if not flag:  # 如果处理失败
                break     # 退出循环

            # 异常检测和推送逻辑
            if class_inf == 1:  # 检测到红外入侵
                abnormal_count += 1  # 异常帧计数加1
                if abnormal_count >= 10 and not abnormal_pushed:  # 连续检测到10帧且未推送
                    print(f"检测到红外入侵10帧，准备推送数据: task_code=3, state=1")
                    try:
                        send_abnormal_data(3, 1)  # 推送异常数据
                        print("红外入侵异常数据推送成功")
                        abnormal_pushed = True  # 设置已推送标志
                    except Exception as e:
                        print(f"红外入侵异常数据推送失败: {e}")
            else:  # 未检测到入侵或不确定
                abnormal_count = 0      # 重置异常帧计数
                abnormal_pushed = False # 重置推送标志

            # 将处理好的帧和任务代号一起放入显示队列
            display_queue.put((processed_frame, 1))

            # 性能统计
            frames += 1
            if frames % 30 == 0:  # 每30帧统计一次
                #print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                loopTime = time.time()

        except queue.Empty:  # 如果队列为空
            continue         # 继续下一次循环

    print("红外入侵检测任务结束")
    pool.release()  # 释放模型池资源

# ============================================================================
# 第八部分：其他检测任务函数（结构类似，省略详细注释）
# ============================================================================
# 安全帽检测任务
def hardhat_detection_task(frame_queue, display_queue, stop_event):
    """安全帽检测任务函数"""
    print("安全帽检测任务启动")
    global classes_data
    global class_hardhat
    model_path = "./helmet.rknn"
    TPEs = 3
    pool = rknnPoolExecutor_hardhat(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_hardhat)

    frames, loopTime = 0, time.time()
    abnormal_count = 0
    abnormal_pushed = False
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break

            pool.put(frame)
            result, flag = pool.get()
            processed_frame, class_hardhat = result
            print(f"安全帽检测结果: {class_hardhat}")

            if not flag:
                break

            # 检测到未戴安全帽
            if class_hardhat == 1:
                abnormal_count += 1
                if abnormal_count >= 10 and not abnormal_pushed:
                    print(f"检测到未戴安全帽10帧，准备推送数据: task_code=2, state=1")
                    try:
                        send_abnormal_data(2, 1)
                        print("安全帽异常数据推送成功")
                        abnormal_pushed = True
                    except Exception as e:
                        print(f"安全帽异常数据推送失败: {e}")
            else:
                abnormal_count = 0
                abnormal_pushed = False

            display_queue.put((processed_frame, 2))

            frames += 1
            if frames % 30 == 0:
                loopTime = time.time()

        except queue.Empty:
            continue

    print("安全帽检测任务结束")
    pool.release()

# 吸烟检测任务
def smoke_detection_task(frame_queue, display_queue, stop_event):
    """吸烟检测任务函数"""
    print("吸烟检测任务启动")
    global classes_data
    global class_smoke
    model_path = "./smoking.rknn"
    TPEs = 3
    pool = rknnPoolExecutor_smoke(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_smoke)

    frames, loopTime = 0, time.time()
    abnormal_count = 0
    abnormal_pushed = False
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break

            pool.put(frame)
            result, flag = pool.get()
            processed_frame, class_smoke = result
            print(f"吸烟检测结果: {class_smoke}")

            if not flag:
                break

            # 检测到吸烟
            if class_smoke == 1:
                abnormal_count += 1
                if abnormal_count >= 20 and not abnormal_pushed:  # 吸烟检测需要20帧确认
                    print(f"检测到吸烟20帧，准备推送数据: task_code=5, state=1")
                    try:
                        send_abnormal_data(5, 1)
                        print("吸烟异常数据推送成功")
                        abnormal_pushed = True
                    except Exception as e:
                        print(f"吸烟异常数据推送失败: {e}")
            else:
                abnormal_count = 0
                abnormal_pushed = False

            display_queue.put((processed_frame, 5))

            frames += 1
            if frames % 30 == 0:
                loopTime = time.time()

        except queue.Empty:
            continue

    print("吸烟检测任务结束")
    pool.release()

# 火焰检测任务
def flame_detection_task(frame_queue, display_queue, stop_event):
    """火焰检测任务函数"""
    print("火焰检测任务启动")
    global class_flame
    model_path = "./fire.rknn"
    TPEs = 3
    pool = rknnPoolExecutor_flame(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_flame)

    frames, loopTime = 0, time.time()
    abnormal_count = 0
    abnormal_pushed = False
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break

            pool.put(frame)
            result, flag = pool.get()
            processed_frame, class_flame = result
            print(class_flame)

            if not flag:
                break

            # 检测到火焰
            if class_flame == 1:
                abnormal_count += 1
                if abnormal_count >= 10 and not abnormal_pushed:
                    print(f"检测到火焰10帧，准备推送数据: task_code=4, state=1")
                    try:
                        send_abnormal_data(4, 1)
                        print("火焰异常数据推送成功")
                        abnormal_pushed = True
                    except Exception as e:
                        print(f"火焰异常数据推送失败: {e}")
            else:
                abnormal_count = 0
                abnormal_pushed = False

            display_queue.put((processed_frame, 5))

            frames += 1
            if frames % 30 == 0:
                loopTime = time.time()

        except queue.Empty:
            continue

    print("火焰检测任务结束")
    pool.release()

# 人脸识别任务
def face_recognition_task(frame_queue, display_queue, stop_event):
    """人脸识别任务函数"""
    global state_env
    print("人脸识别任务启动")
    model_path = 'model_data/retinaface_mob.rknn'
    model_path2 = 'model_data/mobilefacenet.rknn'
    TPEs = 3
    pool = rknnPoolExecutor_face(
        rknnModel1=model_path,
        rknnModel2=model_path2,
        TPEs=TPEs,
        func=myFunc_face)

    frames, loopTime = 0, time.time()
    state_env = True

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                print("从队列获取的帧无效，跳过处理")
                continue

            pool.put(frame)
            processed_frame, flag = pool.get()

            if not flag:
                print("处理失败，flag返回 False")
                break

            display_queue.put((processed_frame, 3))

            frames += 1
            if frames % 30 == 0:
                loopTime = time.time()

        except queue.Empty:
            continue

    print(stop_event.is_set())
    print("人脸识别任务结束")
    pool.release()
    state_env = False

# ============================================================================
# 第九部分：显示功能
# ============================================================================
def show_frames(display_queue):
    """
    显示图像帧的函数
    功能：从显示队列获取处理后的图像帧并显示
    参数：display_queue - 显示队列
    """
    while True:
        try:
            frame, task_code = display_queue.get(timeout=1)  # 从队列获取帧和任务代号
            if frame is None:  # 如果帧无效
                continue       # 继续下一次循环
            cv2.imshow(f"Task {task_code}", frame)  # 显示图像，窗口标题包含任务代号

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q键
                break          # 退出循环
        except queue.Empty:    # 如果队列为空
            continue           # 继续下一次循环

    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

# ============================================================================
# 第十部分：任务调度器
# ============================================================================
def task_worker(task_code, frame_queue, display_queue, stop_event):
    """
    任务工作函数
    功能：根据任务代号启动对应的检测任务
    参数：
        task_code - 任务代号
        frame_queue - 输入帧队列
        display_queue - 显示帧队列
        stop_event - 停止事件
    """
    print(f"任务 {TASK_CODES[task_code]} 启动")
    
    # 根据任务代号启动对应的检测任务
    if task_code == 3:  # 红外入侵检测
        infrared_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 0:  # 人脸识别
        face_recognition_task(frame_queue, display_queue, stop_event)
    elif task_code == 2:  # 安全帽检测
        hardhat_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 4:  # 火焰检测
        flame_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 5:  # 吸烟检测
        smoke_detection_task(frame_queue, display_queue, stop_event)
    
    print(f"任务 {TASK_CODES[task_code]} 停止")

# ============================================================================
# 第十一部分：任务管理器类
# ============================================================================
class TaskManager:
    """
    任务管理器类
    功能：统一管理所有检测任务的启动、停止和资源分配
    特点：支持多任务并发，智能资源管理
    """
    
    def __init__(self):
        """初始化任务管理器"""
        self.cam_manager = CameraManager()  # 创建摄像头管理器
        self.threads = {}                   # 存储任务线程的字典
        self.stop_events = {}               # 存储停止事件的字典
        self.frame_queues = {               # 帧队列字典
            'infrared': queue.Queue(maxsize=5),  # 红外摄像头队列，最大5帧
            'rgb': queue.Queue(maxsize=5)        # RGB摄像头队列，最大5帧
        }
        self.camera_threads = {}            # 存储摄像头线程的字典
        self.display_queue = queue.Queue(maxsize=10)  # 共享显示队列，最大10帧

    def start_camera(self, cam_type):
        """
        启动指定类型的摄像头
        参数：cam_type - 摄像头类型（'infrared' 或 'rgb'）
        """
        if cam_type == 'infrared':  # 如果是红外摄像头
            cam = self.cam_manager.get_infrared_camera()  # 获取红外摄像头
        elif cam_type == 'rgb':     # 如果是RGB摄像头
            cam = self.cam_manager.get_rgb_camera()       # 获取RGB摄像头
        else:
            return

        if cam_type not in self.camera_threads:  # 如果该类型摄像头线程未启动
            stop_event = threading.Event()  # 创建停止事件
            # 创建摄像头捕获线程
            self.camera_threads[cam_type] = threading.Thread(
                target=camera_capture,
                args=(cam, self.frame_queues[cam_type], stop_event)
            )
            self.camera_threads[cam_type].start()  # 启动摄像头线程
            self.stop_events[cam_type] = stop_event  # 保存停止事件

    def start_tasks(self, task_codes):
        """
        启动指定的检测任务
        参数：task_codes - 任务代号列表
        """
        for task_code in task_codes:  # 遍历任务代号列表
            if task_code not in self.threads:  # 如果任务未启动
                stop_event = threading.Event()  # 创建停止事件
                self.stop_events[task_code] = stop_event

                # 根据任务类型选择对应的摄像头和帧队列
                if task_code in [3]:  # 红外任务（红外入侵检测）
                    frame_queue = self.frame_queues['infrared']  # 使用红外摄像头队列
                    self.start_camera('infrared')                # 启动红外摄像头
                else:  # RGB任务（人脸识别、安全帽检测、火焰检测、吸烟检测）
                    frame_queue = self.frame_queues['rgb']       # 使用RGB摄像头队列
                    self.start_camera('rgb')                     # 启动RGB摄像头

                # 创建任务线程
                thread = threading.Thread(
                    target=task_worker, 
                    args=(task_code, frame_queue, self.display_queue, stop_event)
                )
                thread.start()  # 启动任务线程
                self.threads[task_code] = thread  # 保存任务线程

    def stop_tasks(self, task_codes):
        """
        停止指定的检测任务
        参数：task_codes - 任务代号列表
        """
        for task_code in task_codes:  # 遍历任务代号列表
            if task_code in self.threads:  # 如果任务正在运行
                print(f"停止任务 {TASK_CODES[task_code]}")
                self.stop_events[task_code].set()  # 设置停止标志
                self.threads[task_code].join()     # 等待线程结束
                del self.threads[task_code]        # 删除已停止的任务

    def stop_inf_tasks(self, task_codes):
        """
        停止红外任务并释放红外摄像头
        参数：task_codes - 任务代号列表
        """
        for task_code in task_codes:  # 遍历任务代号列表
            if task_code in self.threads:  # 如果任务正在运行
                print(f"停止任务 {TASK_CODES[task_code]}")
                self.stop_events[task_code].set()  # 设置停止标志
                self.threads[task_code].join()     # 等待线程结束
                del self.threads[task_code]        # 删除已停止的任务
        self.cam_manager.release_inf()  # 释放红外摄像头资源

    def stop_all_tasks(self):
        """停止所有任务并释放所有资源"""
        self.stop_tasks(list(self.threads.keys()))  # 停止所有检测任务
        for cam_type in self.camera_threads:  # 遍历摄像头线程
            print("停止任务")
            self.stop_events[cam_type].set()  # 停止摄像头线程
            self.camera_threads[cam_type].join()  # 等待摄像头线程结束
        self.cam_manager.release_cameras()  # 释放所有摄像头资源

    def reset(self):
        """重置任务管理器状态"""
        self.threads.clear()  # 清空任务线程字典
        self.stop_events.clear()  # 清空停止事件字典
        self.camera_threads.clear()  # 清空摄像头线程字典
        # 重新初始化帧队列
        self.frame_queues = {
            'infrared': queue.Queue(maxsize=5),
            'rgb': queue.Queue(maxsize=5)
        }

# ============================================================================
# 第十二部分：UDP服务器类
# ============================================================================
class UDPServer:
    """
    UDP服务器类
    功能：处理UDP通信，接收控制命令，发送响应数据
    特点：支持二进制数据帧解析，实时响应
    """
    
    def __init__(self, ip='127.0.0.1', port=8870):
        """
        初始化UDP服务器
        参数：
            ip - 服务器IP地址，默认本地回环地址
            port - 服务器端口，默认8870
        """
        self.server_address = (ip, port)  # 服务器地址元组
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP套接字
        self.sock.bind(self.server_address)  # 绑定服务器地址
        print(f"UDP服务器已启动，监听地址: {ip}:{port}")

    def receive_frame(self):
        """
        接收数据帧并解析
        返回：(解析后的数据元组, 客户端地址)
        """
        data, addr = self.sock.recvfrom(1024)  # 接收数据包（最大1024字节）
        print("收到数据")
        print(f"接收到的数据{data}")
        # 使用预定义格式解析二进制数据
        unpack_data = struct.unpack(frame_format_meter, data)
        print(f"解析后的数据{unpack_data}")
        return unpack_data, addr  # 返回解析后的数据和客户端地址

    def send_response(self, addr, message):
        """
        发送响应数据
        参数：
            addr - 目标地址
            message - 要发送的消息
        """
        self.sock.sendto(message, addr)  # 发送数据到指定地址

    def close(self):
        """关闭UDP服务器"""
        self.sock.close()  # 关闭套接字

# ============================================================================
# 第十三部分：UDP命令接收和处理
# ============================================================================
def udp_receive_commands(task_manager):
    """
    UDP命令接收和处理函数
    功能：主循环，持续接收和处理UDP命令
    参数：task_manager - 任务管理器实例
    """
    global udp_server  # 使用全局UDP服务器实例
    udp_server = UDPServer()  # 创建UDP服务器实例
    
    # 全局变量声明
    ignore_frame = False
    close_flag = 0
    global state_env
    global classes_data
    global class_flame
    global class_inf
    global class_hardhat
    global class_smoke
    
    try:
        while True:  # 主循环，持续接收UDP命令
            # 从UDP接收数据帧，并解析任务代号和操作
            frame_data, client_addr = udp_server.receive_frame()
            
            # 处理仪表检测相关命令
            if frame_data[1] == 3 and frame_data[0] == 2:  # 仪表检测控制命令
                print("仪表检测")
                if frame_data[5] == 1:  # 启动仪表检测
                    print("恢复处理服务器的帧")
                    task_manager.reset()  # 重置任务管理器
                    task_manager.start_tasks([0])  # 启动人脸识别任务
                elif frame_data[5] == 0:  # 停止仪表检测
                    print("忽略其他帧")
                    task_manager.stop_all_tasks()  # 停止所有任务
                    # 发送完成响应
                    data = (2, 1, 3, 1, 3, 0, 0, 0, 0)
                    pack_data = struct.pack(frame_format_meter, *data)
                    udp_server.send_response(client_addr, pack_data)
                continue
            
            # 处理控制命令
            if frame_data[0] == 2:  # 控制命令类型
                if frame_data[1] == 0:  # 来自服务器的控制命令
                    task_code = frame_data[3]  # 任务代号
                    action = frame_data[5]     # 操作类型

                    if action == 1:  # 启动任务
                        if close_flag == 1:  # 如果之前关闭了所有任务
                            task_manager.reset()  # 重置任务管理器
                            close_flag = 0
                        if task_code == 3:  # 如果是红外任务
                            task_manager.reset()  # 重置任务管理器
                        task_manager.start_tasks([task_code])  # 启动指定任务
                        
                    elif action == 2:  # 停止任务
                        if task_code == 3:  # 如果是红外任务
                            task_manager.stop_inf_tasks([task_code])  # 停止红外任务并释放摄像头
                        else:  # 其他任务
                            task_manager.stop_tasks([task_code])  # 停止指定任务

                    elif action == 0:  # 关闭所有任务
                        print("close all")
                        task_manager.stop_all_tasks()  # 停止所有任务
                        close_flag = 1  # 设置关闭标志

            # 处理数据请求
            elif frame_data[0] == 0 and frame_data[2] == 1:  # 数据请求类型
                print("服务端请求数据")
                
                if frame_data[3] == 0:  # 请求人脸数据
                    print("人脸数据")
                    if state_env is True:  # 如果人脸识别任务正在运行
                        if func_face.name_ten and len(func_face.name_ten) > 0:  # 如果检测到人脸
                            state = 1
                            person_name = ','.join(func_face.name_ten[0])  # 获取人名
                            print(f"person_name = {person_name}")
                            if person_name == "Unknown":  # 如果是未知人脸
                                person_name = 0
                            person_name = int(person_name)  # 转换为整数
                            data_face = (1, 1, 0, 0, state, person_name, 0, 0, 0)
                        else:  # 未检测到人脸
                            state = 0
                            data_face = (1, 1, 0, 0, state, 0, 0, 0, 0)
                        pack_data_face = struct.pack(frame_format_meter, *data_face)
                        udp_server.send_response(client_addr, pack_data_face)
                        print("发送人脸数据")
                    else:  # 人脸识别任务未运行
                        state = 0
                        data_face = (1, 1, 0, 0, state, 0, 0, 0, 0)
                        pack_data_face = struct.pack(frame_format_meter, *data_face)
                        udp_server.send_response(client_addr, pack_data_face)
                        print("发送人脸数据")

                if frame_data[3] == 3:  # 请求红外入侵数据
                    if class_inf == 1:  # 如果检测到入侵
                        state_inf = 1
                    else:  # 未检测到入侵
                        state_inf = 0
                    state_inf = int(state_inf)
                    print(f'inf class data {class_inf}')
                    print(f'inf data {state_inf}')
                    data_inf = (1, 1, 0, 3, state_inf, 0, 0, 0, 0)
                    pack_data_inf = struct.pack(frame_format_meter, *data_inf)
                    udp_server.send_response(client_addr, pack_data_inf)
                    print("发送红外入侵数据")

                if frame_data[3] == 2:  # 请求安全帽检测数据
                    print("安全帽检测数据")
                    global class_hardhat
                    if 'class_hardhat' not in globals() or class_hardhat is None:  # 如果未检测
                        data_hardhat = (1, 1, 0, 2, 0, 0, 0, 0, 0)
                        pack_data_hardhat = struct.pack(frame_format_meter, *data_hardhat)
                        udp_server.send_response(client_addr, pack_data_hardhat)
                        print("发送安全帽检测数据：戴安全帽")
                    else:  # 如果已检测
                        data_hardhat = (1, 1, 0, 2, class_hardhat, 1, 0, 0, 0)
                        pack_data_hardhat = struct.pack(frame_format_meter, *data_hardhat)
                        udp_server.send_response(client_addr, pack_data_hardhat)
                        print(f"发送安全帽检测数据：{'未戴安全帽' if class_hardhat == 1 else '戴安全帽'}")
                    print("发送安全帽检测数据")
                    
                if frame_data[3] == 4:  # 请求火焰检测数据
                    if class_flame is None:  # 如果未检测
                        data_flame = (1, 1, 0, 4, 0, 0, 0, 0, 0)
                        pack_data_flame = struct.pack(frame_format_meter, *data_flame)
                        udp_server.send_response(client_addr, pack_data_flame)
                        print("发送数据")
                    else:  # 如果已检测
                        data_flame = (1, 1, 0, 4, 1, 1, 0, 0, 0)
                        pack_data_flame = struct.pack(frame_format_meter, *data_flame)
                        udp_server.send_response(client_addr, pack_data_flame)
                        print("发送数据")
                    print("发送火焰数据")

                if frame_data[3] == 5:  # 请求吸烟检测数据
                    global class_smoke
                    if 'class_smoke' not in globals() or class_smoke is None:  # 如果未检测
                        data_smoke = (1, 1, 0, 5, 0, 0, 0, 0, 0)
                        pack_data_smoke = struct.pack(frame_format_meter, *data_smoke)
                        udp_server.send_response(client_addr, pack_data_smoke)
                        print("发送吸烟检测数据：无吸烟")
                    else:  # 如果已检测
                        data_smoke = (1, 1, 0, 5, class_smoke, 1, 0, 0, 0)
                        pack_data_smoke = struct.pack(frame_format_meter, *data_smoke)
                        udp_server.send_response(client_addr, pack_data_smoke)
                        print(f"发送吸烟检测数据：{'检测到吸烟' if class_smoke == 1 else '无吸烟'}")
                    print("发送吸烟检测数据")

    except KeyboardInterrupt:  # 捕获键盘中断信号
        print("服务器停止")
    finally:  # 确保资源被正确释放
        udp_server.close()  # 关闭UDP服务器

# ============================================================================
# 第十四部分：主程序入口
# ============================================================================
if __name__ == "__main__":
    """
    主程序入口
    功能：初始化系统，启动显示线程，启动UDP命令接收
    """
    task_manager = TaskManager()  # 创建任务管理器实例

    # 启动显示线程
    display_thread = threading.Thread(
        target=show_frames, 
        args=(task_manager.display_queue,)
    )
    display_thread.start()  # 启动显示线程

    try:
        # 启动UDP命令接收线程（主线程）
        udp_receive_commands(task_manager)
    finally:
        print("final")
        task_manager.stop_all_tasks()  # 停止所有任务
        display_thread.join()  # 等待显示线程结束
