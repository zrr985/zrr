import threading
import queue
import cv2
import func_face
import func_v7
import video_number
from rknnpool_rgb import rknnPoolExecutor_face
from func_face import myFunc_face
from rknnpool_inf import rknnPoolExecutor_inf
from func_v7 import myFunc_inf, data_ten_inf
from rknnpool_smoke import rknnPoolExecutor_smoke_hat
from func_smoke_hat import myFunc_smoke_hat
from rknnpool_flame import rknnPoolExecutor_flame
from func_flame import myFunc_flame
from rknnpool_hardhat import rknnPoolExecutor_hardhat
from rknnpool_smoke_single import rknnPoolExecutor_smoke
from func_hardhat import myFunc_hardhat
from func_smoke import myFunc_smoke
import time
import socket
import struct

classes_data = []
class_inf = 0
class_flame = None
class_smoke = None  # 添加吸烟检测全局变量
class_hardhat = None  # 添加安全帽检测全局变量
# 任务代号常量
TASK_CODES = {
    3: '红外入侵检测',
    4: '火焰检测',
    0: '人脸识别',
    1: '仪表检测',
    2: '安全帽检测',
    5: '吸烟检测'
}

#第一位 请求类型
REQUEST_DATA = 0
GIVE_DATA = 1
CONTROL_COMMOND = 2

#第二位与第三位 设备类型
SERVER = 0
VISION = 1
ROBOT = 2
METER = 3

#第四位
SUB_FACE = 0
SUB_METER = 1
SUB_HARDHAT = 2  # 安全帽检测
SUB_INF = 3  #红外入侵检测
SUB_FLAME = 4  #火焰检测
SUB_THR = 5  #温度检测
SUB_GAS = 6  #气体检测
SUB_SMOKE = 7  #吸烟检测

#第五位
EXIST = 1
NOT_EXIST = 0
FINSH = 3 #无关

#仪表检测数据帧第六位:数据类型  人脸第六位为人脸信息
METER_TYPE_XIAOFANG = 0
METER_TYPE_AIR = 1
METER_TYPE_JIXIE = 2

#抽烟安全帽数据帧第六位
HAT = 1
NO_HAT = 0
SMOKE = 3
NOT_SMOKE = 2


#仪表检测控制帧第六位
CLOSE_ALL = 0
START = 1
CLOSE = 2
#仪表检测数据帧第八位  温度检测第八位为温度
#读数

#第七和第九为空出

frame_format_meter = 'IIIIIIIff'

# 任务状态管理变量
state_env = False        # 人脸识别任务状态
state_inf = False        # 红外入侵检测任务状态
state_flame = False      # 火焰检测任务状态
state_smoke = False      # 吸烟检测任务状态
state_hardhat = False    # 安全帽检测任务状态

# 打开摄像头函数，根据提供的摄像头列表尝试打开摄像头
def open_camera(camera_numbers):
    for number in camera_numbers:
        cap = cv2.VideoCapture(number)
        if cap.isOpened():
            print(f"Found openable camera: {number}")
            return cap
    return None

# 摄像头资源管理
class CameraManager:
    def __init__(self):
        self.infrared_cam = None  # 红外摄像头
        self.rgb_cam = None  # RGB摄像头
        self.lock = threading.Lock()  # 线程锁防止摄像头资源冲突

    def get_infrared_camera(self):
        with self.lock:
            if self.infrared_cam is None:
                self.infrared_cam = open_camera(video_number.inf_numbers)
            return self.infrared_cam

    def get_rgb_camera(self):
        with self.lock:
            if self.rgb_cam is None:
                self.rgb_cam = open_camera(video_number.rgb_numbers)
            return self.rgb_cam

    def release_cameras(self):
        with self.lock:
            if self.infrared_cam is not None:
                self.infrared_cam.release()
                self.infrared_cam = None
            if self.rgb_cam is not None:
                self.rgb_cam.release()
                self.rgb_cam = None

    def release_inf(self):
        with self.lock:
            if self.infrared_cam is not None:
                self.infrared_cam.release()
                self.infrared_cam = None

# 读取摄像头帧并将帧放入队列的生产者函数
def camera_capture(cam, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()  # 丢弃最老的帧，保持队列大小
        frame_queue.put(frame)

# 人脸识别任务
def face_recognition_task(frame_queue, display_queue, stop_event):
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

            # 将处理好的帧和任务代号一起放入显示队列
            display_queue.put((processed_frame, 3))

            frames += 1
            if frames % 30 == 0:
                #print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                loopTime = time.time()

        except queue.Empty:
            continue

    print(stop_event.is_set())
    print("人脸识别任务结束")
    pool.release()
    state_env = False

# UDP服务器实例，用于发送数据
udp_server = None

# 发送异常数据到服务器
def send_abnormal_data(task_code, state):
    global udp_server
    try:
        if udp_server:
          if task_code == 3:  # 红外入侵检测
            data_inf = (1, 1, 0, 3, state, 0, 0, 0, 0)
            pack_data_inf = struct.pack(frame_format_meter, *data_inf)
            server_addr = ('127.0.0.1', 8111)  # 假设服务器地址和端口
            udp_server.send_response(server_addr, pack_data_inf)
            print("发送红外入侵异常数据")
          elif task_code == 2:  # 安全帽检测
            data_hardhat = (1, 1, 0, 2, 1, state, 0, 0, 0)
            pack_data_hardhat = struct.pack(frame_format_meter, *data_hardhat)
            server_addr = ('127.0.0.1', 8111)  # 假设服务器地址和端口
            udp_server.send_response(server_addr, pack_data_hardhat)
            print("发送安全帽异常数据")
          elif task_code == 5:  # 吸烟检测
            data_smoke = (1, 1, 0, 7, 1, state, 0, 0, 0)  # 使用SUB_SMOKE = 7
            pack_data_smoke = struct.pack(frame_format_meter, *data_smoke)
            server_addr = ('127.0.0.1', 8111)  # 假设服务器地址和端口
            udp_server.send_response(server_addr, pack_data_smoke)
            print("发送吸烟异常数据")
          elif task_code == 4:  # 火焰检测
            data_flame = (1, 1, 0, 4, state, 1, 0, 0, 0)
            pack_data_flame = struct.pack(frame_format_meter, *data_flame)
            server_addr =  ('127.0.0.1', 8111)  # 假设服务器地址和端口
            udp_server.send_response(server_addr, pack_data_flame)
            print("发送火焰异常数据")
    except Exception as e:
        print(f"发送异常数据失败: task_code={task_code}, state={state}, error={e}")
# 红外入侵检测任务
def infrared_detection_task(frame_queue, display_queue, stop_event):
    print("红外入侵检测任务启动")
    global class_inf, state_inf
    state_inf = True  # 设置任务运行状态
    model_path = "./yolov7_tiny-a.rknn"
    TPEs = 3
    pool = rknnPoolExecutor_inf(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_inf)

    frames, loopTime = 0, time.time()
    abnormal_count = 0  # 连续异常帧计数
    abnormal_pushed = False  # 是否已推送异常

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break

            pool.put(frame)
            result, flag = pool.get()
            processed_frame, class_inf = result
            print(f'now inf data {class_inf}')
            if not flag:
                break

            # 检测到红外入侵
            if class_inf == 1:
                abnormal_count += 1
                if abnormal_count >= 10 and not abnormal_pushed:
                    print(f"检测到红外入侵10帧，准备推送数据: task_code=3, state=1")
                    try:
                        send_abnormal_data(3, 1)
                        print("红外入侵异常数据推送成功")
                        abnormal_pushed = True
                    except Exception as e:
                        print(f"红外入侵异常数据推送失败: {e}")
            else:  # 未检测到入侵或不确定
                abnormal_count = 0
                abnormal_pushed = False

            # 将处理好的帧和任务代号一起放入显示队列
            display_queue.put((processed_frame, 1))

            frames += 1
            if frames % 30 == 0:
                #print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                loopTime = time.time()

        except queue.Empty:
            continue

    print("红外入侵检测任务结束")
    state_inf = False  # 重置任务状态
    pool.release()
"""
def smoke_hardhat_detection_task(frame_queue, display_queue, stop_event):
    print("抽烟及安全帽检测任务启动")
    global classes_data
    model_path = "./yolov8_smoke_hat.rknn"
    TPEs = 3
    pool = rknnPoolExecutor_smoke_hat(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_smoke_hat)

    frames, loopTime = 0, time.time()

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break

            pool.put(frame)
            result, flag = pool.get()
            processed_frame, classes_data = result
            #print(classes_data)

            if not flag:
                break

            # 检测到没带安全帽或抽烟，发送数据到服务器
            for cl in classes_data:
                if cl in [NO_HAT, SMOKE]:
                    send_abnormal_data(2, cl)

            # 将处理好的帧和任务代号一起放入显示队列
            display_queue.put((processed_frame, 5))

            frames += 1
            if frames % 30 == 0:
                #print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                loopTime = time.time()

        except queue.Empty:
            continue

    print("抽烟及安全帽检测任务结束")
    pool.release()
"""
def hardhat_detection_task(frame_queue, display_queue, stop_event):
    print("安全帽检测任务启动")
    global classes_data, class_hardhat, state_hardhat
    state_hardhat = True  # 设置任务运行状态
    model_path = "./helmet.rknn"  # 使用安全帽专用模型
    TPEs = 3
    pool = rknnPoolExecutor_hardhat(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_hardhat)

    frames, loopTime = 0, time.time()
    abnormal_count = 0  # 连续异常帧计数
    abnormal_pushed = False  # 是否已推送异常
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
            else:  # 戴安全帽或不确定
                abnormal_count = 0
                abnormal_pushed = False

            # 将处理好的帧和任务代号一起放入显示队列
            display_queue.put((processed_frame, 2))

            frames += 1
            if frames % 30 == 0:
                # print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                loopTime = time.time()

        except queue.Empty:
            continue

    print("安全帽检测任务结束")
    state_hardhat = False  # 重置任务状态
    pool.release()

def smoke_detection_task(frame_queue, display_queue, stop_event):
    print("吸烟检测任务启动")
    global classes_data, class_smoke, state_smoke
    state_smoke = True  # 设置任务运行状态
    model_path = "./smoking.rknn"  # 使用吸烟专用模型
    TPEs = 3
    pool = rknnPoolExecutor_smoke(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_smoke)

    frames, loopTime = 0, time.time()
    abnormal_count = 0  # 连续异常帧计数
    abnormal_pushed = False  # 是否已推送异常
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
                if abnormal_count >= 20 and not abnormal_pushed:
                    print(f"检测到吸烟20帧，准备推送数据: task_code=5, state=1")
                    try:
                        send_abnormal_data(5, 1)
                        print("吸烟异常数据推送成功")
                        abnormal_pushed = True
                    except Exception as e:
                        print(f"吸烟异常数据推送失败: {e}")
            else:  # 未吸烟或不确定
                abnormal_count = 0
                abnormal_pushed = False

            # 将处理好的帧和任务代号一起放入显示队列
            display_queue.put((processed_frame, 5))

            frames += 1
            if frames % 30 == 0:
                # print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                loopTime = time.time()

        except queue.Empty:
            continue

    print("吸烟检测任务结束")
    state_smoke = False  # 重置任务状态
    pool.release()

def flame_detection_task(frame_queue, display_queue, stop_event):
    print("火焰检测任务启动")
    global class_flame, state_flame
    state_flame = True  # 设置任务运行状态
    model_path = "./fire.rknn"
    TPEs = 3
    pool = rknnPoolExecutor_flame(
        rknnModel=model_path,
        TPEs=TPEs,
        func=myFunc_flame)

    frames, loopTime = 0, time.time()
    abnormal_count = 0  # 连续异常帧计数
    abnormal_pushed = False  # 是否已推送异常
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
            else:  # 未检测到火焰或不确定
                abnormal_count = 0
                abnormal_pushed = False

            # 将处理好的帧和任务代号一起放入显示队列
            display_queue.put((processed_frame, 4))  # 火焰检测任务代号为4

            frames += 1
            if frames % 30 == 0:
                #print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                loopTime = time.time()

        except queue.Empty:
            continue

    print("火焰检测任务结束")
    state_flame = False  # 重置任务状态
    pool.release()

# 显示图像帧的函数
def show_frames(display_queue):
    while True:
        try:
            frame, task_code = display_queue.get(timeout=1)
            if frame is None:
                continue
            cv2.imshow(f"Task {task_code}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # print("display thread")
        except queue.Empty:
            continue

    cv2.destroyAllWindows()


# 任务处理函数
def task_worker(task_code, frame_queue, display_queue, stop_event):
    print(f"任务 {TASK_CODES[task_code]} 启动")
    if task_code == 3:
        infrared_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 0:
        face_recognition_task(frame_queue, display_queue, stop_event)
    elif task_code == 2:
        hardhat_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 4:
        flame_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 5:
        smoke_detection_task(frame_queue, display_queue, stop_event)
    print(f"任务 {TASK_CODES[task_code]} 停止")
"""
def task_worker(task_code, frame_queue, display_queue, stop_event):
    print(f"任务 {TASK_CODES[task_code]} 启动")

    if task_code == 3:
        infrared_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 0:
        face_recognition_task(frame_queue, display_queue, stop_event)
    elif task_code == 2:
        smoke_hardhat_detection_task(frame_queue, display_queue, stop_event)
    elif task_code == 4:
        flame_detection_task(frame_queue, display_queue, stop_event)


    print(f"任务 {TASK_CODES[task_code]} 停止")
"""
# 管理任务的类
class TaskManager:
    def __init__(self):
        self.cam_manager = CameraManager()
        self.threads = {}  # 存储任务线程和停止标志
        self.stop_events = {}
        self.frame_queues = {
            'infrared': queue.Queue(maxsize=5),  # 红外摄像头队列
            'rgb': queue.Queue(maxsize=5)  # RGB摄像头队列
        }
        self.camera_threads = {}  # 存储摄像头线程
        self.display_queue = queue.Queue(maxsize=10)  # 共享显示队列

    def start_camera(self, cam_type):
        if cam_type == 'infrared':
            cam = self.cam_manager.get_infrared_camera()
        elif cam_type == 'rgb':
            cam = self.cam_manager.get_rgb_camera()
        else:
            return

        if cam_type not in self.camera_threads:
            stop_event = threading.Event()
            self.camera_threads[cam_type] = threading.Thread(target=camera_capture,
                                                             args=(cam, self.frame_queues[cam_type], stop_event))
            self.camera_threads[cam_type].start()
            self.stop_events[cam_type] = stop_event

    def start_tasks(self, task_codes):
        for task_code in task_codes:
            if task_code not in self.threads:
                stop_event = threading.Event()
                self.stop_events[task_code] = stop_event

                if task_code in [3]:  # 红外任务
                    frame_queue = self.frame_queues['infrared']
                    self.start_camera('infrared')
                else:  # RGB任务
                    frame_queue = self.frame_queues['rgb']
                    self.start_camera('rgb')

                thread = threading.Thread(target=task_worker, args=(task_code, frame_queue, self.display_queue, stop_event))
                thread.start()
                self.threads[task_code] = thread

    def stop_tasks(self, task_codes):
        for task_code in task_codes:
            if task_code in self.threads:
                print(f"停止任务 {TASK_CODES[task_code]}")
                self.stop_events[task_code].set()  # 设置停止标志
                self.threads[task_code].join()  # 等待线程结束
                del self.threads[task_code]  # 删除已停止的任务

    def stop_inf_tasks(self, task_codes):
        for task_code in task_codes:
            if task_code in self.threads:
                print(f"停止任务 {TASK_CODES[task_code]}")
                self.stop_events[task_code].set()  # 设置停止标志
                self.threads[task_code].join()  # 等待线程结束
                del self.threads[task_code]  # 删除已停止的任务
            self.cam_manager.release_inf()

    def stop_all_tasks(self):
        self.stop_tasks(list(self.threads.keys()))
        for cam_type in self.camera_threads:
            print("停止任务")
            self.stop_events[cam_type].set()  # 停止摄像头线程
            self.camera_threads[cam_type].join()
        self.cam_manager.release_cameras()

    def reset(self):
        self.threads.clear()
        self.stop_events.clear()
        self.camera_threads.clear()
        self.frame_queues = {
            'infrared': queue.Queue(maxsize=5),  # 红外摄像头队列
            'rgb': queue.Queue(maxsize=5)  # RGB摄像头队列
        }

# 定义UDP服务器
class UDPServer:
    def __init__(self, ip='127.0.0.1', port=8870):
        self.server_address = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.server_address)
        print(f"UDP服务器已启动，监听地址: {ip}:{port}")

    # 接收帧数据并解析任务信息
    def receive_frame(self):
        data, addr = self.sock.recvfrom(1024)  # 接收数据包（最大1024字节）
        print("收到数据")
        print(f"接收到的数据{data}")
        unpack_data = struct.unpack(frame_format_meter, data)  # 假设前8字节分别是任务代号(int)和操作(int)
        print(f"解析后的数据{unpack_data}")
        return unpack_data, addr  # 返回任务代号和操作

    def send_response(self, addr, message):
        self.sock.sendto(message, addr)

    def close(self):
        self.sock.close()

# UDP命令接收函数
def udp_receive_commands(task_manager):
    global udp_server
    udp_server = UDPServer()  # 创建UDP服务器
    ignore_frame = False
    close_flag = 0
    global state_env, classes_data, class_flame, class_inf, class_hardhat, class_smoke
    global state_inf, state_flame, state_smoke, state_hardhat  # 添加任务状态变量
    try:
        while True:
            # 从UDP接收帧，并解析任务代号和操作（启动/停止）
            frame_data, client_addr = udp_server.receive_frame()
            if frame_data[1] == 3 and frame_data[0] == 2:
                print("仪表检测")
                if frame_data[5] == 1:
                    #ignore_frame = False
                    print("恢复处理服务器的帧")
                    task_manager.reset()
                    task_manager.start_tasks([0])
                elif frame_data[5] == 0:
                    #ignore_frame = True
                    print("忽略其他帧")
                    task_manager.stop_all_tasks()
                    data = (2, 1, 3, 1, 3, 0, 0, 0, 0)
                    pack_data = struct.pack(frame_format_meter, *data)
                    udp_server.send_response(client_addr, pack_data)
                continue
            # if frame_data[0] == 0:
            #     #print
            # elif frame_data[0] == 1:
            if frame_data[0] == 2:
                if frame_data[1] == 0:
                    task_code = frame_data[3]
                    action = frame_data[5]

                    if action == 1:  # 启动任务
                        if close_flag == 1:
                            task_manager.reset()
                            close_flag = 0
                        if task_code == 3:
                            task_manager.reset()
                        task_manager.start_tasks([task_code])
                    elif action == 2:  # 停止任务
                        if task_code == 3:
                            task_manager.stop_inf_tasks([task_code])
                        else:
                            task_manager.stop_tasks([task_code])



                    elif action == 0:
                        print("close all")
                        task_manager.stop_all_tasks()
                        close_flag = 1



            # 打印收到的任务及操作
                    #print(f"接收到的任务代号: {task_code}, 操作: {'启动' if action == 1 else '停止'}")

            elif frame_data[0] == 0 and frame_data[2] == 1:
                print("服务端请求数据")
                if frame_data[3] == 0:  #需要人脸数据
                    print("人脸数据")
                    if state_env is True:
                        if func_face.name_ten and len(func_face.name_ten) > 0:
                            state = 1
                            person_name = ','.join(func_face.name_ten[0])

                            print(f"person_name = {person_name}")
                            if person_name == "Unknown":
                                person_name = 0
                            person_name = int(person_name)
                            data_face = (1, 1, 0, 0, state, person_name, 0, 0, 0)
                        else:
                            state = 0
                            data_face = (1, 1, 0, 0, state, 0, 0, 0, 0)
                        pack_data_face = struct.pack(frame_format_meter, *data_face)
                        udp_server.send_response(client_addr, pack_data_face)
                        print("发送人脸数据")
                    else:
                        state = 0
                        data_face = (1, 1, 0, 0, state, 0, 0, 0, 0)
                        pack_data_face = struct.pack(frame_format_meter, *data_face)
                        udp_server.send_response(client_addr, pack_data_face)
                        print("发送人脸数据")


                if frame_data[3] == 3:  #需要红外入侵数据
                    if state_inf is True:  # 检查任务状态
                        if class_inf == 1:
                            state_inf = 1
                        else:
                            state_inf = 0
                        state_inf = int(state_inf)
                        print(f'inf class data {class_inf}')
                        print(f'inf data {state_inf}')
                        data_inf = (1, 1, 0, 3, state_inf, 0, 0, 0, 0)
                    else:  # 任务未运行
                        data_inf = (1, 1, 0, 3, 0, 0, 0, 0, 0)
                    pack_data_inf = struct.pack(frame_format_meter, *data_inf)
                    udp_server.send_response(client_addr, pack_data_inf)
                    print("发送红外入侵数据")

                #if frame_data[3] == 2:  #需要安全帽和抽烟数据
                    #print(f"class data {classes_data}")
                    #if not classes_data :
                        #print('1')

                        #data_smoke_hat = (1, 1, 0, 2, 0, 0, 0, 0, 0)
                        #pack_data_smoke_hat = struct.pack(frame_format_meter, *data_smoke_hat)
                        #udp_server.send_response(client_addr, pack_data_smoke_hat)
                        #print("发送数据")
                if frame_data[3] == 2:  #需要安全帽检测数据
                    print("安全帽检测数据")
                    if state_hardhat is True:  # 检查任务状态
                        data_hardhat = (1, 1, 0, 2, class_hardhat, 1, 0, 0, 0)
                        pack_data_hardhat = struct.pack(frame_format_meter, *data_hardhat)
                        udp_server.send_response(client_addr, pack_data_hardhat)
                        print(f"发送安全帽检测数据：{'未戴安全帽' if class_hardhat == 1 else '戴安全帽'}")
                    else:  # 任务未运行
                        data_hardhat = (1, 1, 0, 2, 0, 0, 0, 0, 0)
                        pack_data_hardhat = struct.pack(frame_format_meter, *data_hardhat)
                        udp_server.send_response(client_addr, pack_data_hardhat)
                        print("发送安全帽检测数据：任务未运行")
                    print("发送安全帽检测数据")
                if frame_data[3] == 4:  #需要火焰检测数据
                    if state_flame is True:  # 检查任务状态
                        data_flame = (1, 1, 0, 4, 1, 1, 0, 0, 0)  # 使用SUB_FLAME = 4
                        pack_data_flame = struct.pack(frame_format_meter, *data_flame)
                        udp_server.send_response(client_addr, pack_data_flame)
                        print("发送火焰检测数据：任务运行中")
                    else:  # 任务未运行
                        data_flame = (1, 1, 0, 4, 0, 0, 0, 0, 0)  # 使用SUB_FLAME = 4
                        pack_data_flame = struct.pack(frame_format_meter, *data_flame)
                        udp_server.send_response(client_addr, pack_data_flame)
                        print("发送火焰检测数据：任务未运行")
                    print("发送火焰数据")

                if frame_data[3] == 7:  #需要吸烟检测数据
                    if state_smoke is True:  # 检查任务状态
                        data_smoke = (1, 1, 0, 7, class_smoke, 1, 0, 0, 0)  # 使用SUB_SMOKE = 7
                        pack_data_smoke = struct.pack(frame_format_meter, *data_smoke)
                        udp_server.send_response(client_addr, pack_data_smoke)
                        print(f"发送吸烟检测数据：{'检测到吸烟' if class_smoke == 1 else '无吸烟'}")
                    else:  # 任务未运行
                        data_smoke = (1, 1, 0, 7, 0, 0, 0, 0, 0)  # 使用SUB_SMOKE = 7
                        pack_data_smoke = struct.pack(frame_format_meter, *data_smoke)
                        udp_server.send_response(client_addr, pack_data_smoke)
                        print("发送吸烟检测数据：任务未运行")
                    print("发送吸烟检测数据")





    except KeyboardInterrupt:
        print("服务器停止")
    finally:
        udp_server.close()

if __name__ == "__main__":
    task_manager = TaskManager()

    # 启动显示线程
    display_thread = threading.Thread(target=show_frames, args=(task_manager.display_queue,))
    display_thread.start()

    try:
        # 启动UDP命令接收线程
        udp_receive_commands(task_manager)
    finally:
        print("final")
        task_manager.stop_all_tasks()
        display_thread.join()  # 等待显示线程结束