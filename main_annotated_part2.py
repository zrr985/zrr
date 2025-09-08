# ============================================================================
# 智能视觉检测系统 - 主控制模块 (main.py) - 第二部分
# 摄像头管理、检测任务函数和UDP通信功能
# ============================================================================

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
                data_smoke = (1, 1, 0, 7, 1, state, 0, 0, 0)  # 使用SUB_SMOKE = 7
                pack_data_smoke = struct.pack(frame_format_meter, *data_smoke)
                server_addr = ('127.0.0.1', 8111)
                udp_server.send_response(server_addr, pack_data_smoke)
                print("发送吸烟异常数据")
            elif task_code == 4:  # 火焰检测异常
                # 构建火焰检测数据帧
                data_flame = (1, 1, 0, 4, state, 1, 0, 0, 0)  # 使用SUB_FLAME = 4
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
    global class_inf, state_inf
    state_inf = True  # 设置任务运行状态
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
    state_inf = False  # 重置任务状态
    pool.release()  # 释放模型池资源

# ============================================================================
# 第八部分：其他检测任务函数（结构类似）
# ============================================================================
# 安全帽检测任务
def hardhat_detection_task(frame_queue, display_queue, stop_event):
    """安全帽检测任务函数"""
    print("安全帽检测任务启动")
    global classes_data, class_hardhat, state_hardhat
    state_hardhat = True  # 设置任务运行状态
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
    state_hardhat = False  # 重置任务状态
    pool.release()

# 吸烟检测任务
def smoke_detection_task(frame_queue, display_queue, stop_event):
    """吸烟检测任务函数"""
    print("吸烟检测任务启动")
    global classes_data, class_smoke, state_smoke
    state_smoke = True  # 设置任务运行状态
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
    state_smoke = False  # 重置任务状态
    pool.release()

# 火焰检测任务
def flame_detection_task(frame_queue, display_queue, stop_event):
    """火焰检测任务函数"""
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
    state_flame = False  # 重置任务状态
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
