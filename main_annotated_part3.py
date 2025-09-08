# ============================================================================
# 智能视觉检测系统 - 主控制模块 (main.py) - 第三部分
# 任务管理器、UDP服务器、命令处理和主程序入口
# ============================================================================

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
    global state_env, classes_data, class_flame, class_inf, class_hardhat, class_smoke
    global state_inf, state_flame, state_smoke, state_hardhat  # 添加任务状态变量
    
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
