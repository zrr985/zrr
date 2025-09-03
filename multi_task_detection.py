#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务视觉检测系统 - 修复版本
修复了摄像头线程管理问题，确保所有任务都能接收到摄像头帧
"""

# 导入所需的库
import cv2                                          # OpenCV库，用于图像处理和摄像头操作
import time                                         # 时间库，用于计算帧率和时间间隔
import threading                                    # 线程库，用于多线程处理
import queue                                        # 队列库，用于线程间数据传递
import sys                                          # 系统库，用于程序退出
import signal                                       # 信号库，用于处理系统信号
from collections import defaultdict                 # 集合库，用于创建默认字典

# 导入摄像头配置和各个检测模块
import video_number                                 # 摄像头编号配置模块
from rknnpool_flame import rknnPoolExecutor_flame   # RKNN模型池执行器（火焰检测专用）
from rknnpool_rgb import rknnPoolExecutor_face      # RKNN模型池执行器（人脸识别专用）
from rknnpool_meter import rknnPoolExecutor         # RKNN模型池执行器（仪表检测专用）
from rknnpool_hardhat import rknnPoolExecutor_hardhat  # RKNN模型池执行器（安全帽检测专用）
from rknnpool_smoke_single import rknnPoolExecutor_smoke  # RKNN模型池执行器（吸烟检测专用）

# 导入各个检测功能函数
from func_flame import myFunc_flame                 # 火焰检测功能函数
from func_face import myFunc_face                   # 人脸识别功能函数
from func_meter import myFunc                       # 仪表检测功能函数
from func_hardhat import myFunc_hardhat             # 安全帽检测功能函数
from func_smoke import myFunc_smoke                 # 吸烟检测功能函数

# 导入仪表读数器和人脸识别模块
from read_meter import MeterReader                  # 仪表读数器类
import func_face                                    # 人脸识别模块（访问全局变量）
import numpy as np                                  # NumPy库，用于数值计算

# 导入配置文件
try:
    import detection_config as config              # 导入配置文件
    print("✅ 成功加载配置文件")
except ImportError:
    print("⚠️ 配置文件不存在，使用默认配置")
    config = None

class MultiTaskDetectionSystem:
    """多任务视觉检测系统类"""
    
    def __init__(self):
        """初始化多任务检测系统"""
        # 系统控制参数
        self.stop_event = threading.Event()         # 全局停止事件，用于控制所有线程的退出
        self.running_tasks = set()                   # 当前正在运行的任务集合
        
        # 摄像头管理
        self.cameras = {}                            # 摄像头对象字典 {'rgb': cap}
        self.camera_threads = {}                     # 摄像头采集线程字典
        self.camera_locks = {}                       # 摄像头锁字典，防止多个任务同时访问同一摄像头
        
        # 帧队列管理 - 每个任务都有独立的队列
        self.frame_queues = {
            'flame': queue.Queue(maxsize=5),         # 火焰检测帧队列
            'face': queue.Queue(maxsize=5),          # 人脸识别帧队列
            'meter': queue.Queue(maxsize=5),         # 仪表检测帧队列
            'hardhat': queue.Queue(maxsize=5),       # 安全帽检测帧队列
            'smoking': queue.Queue(maxsize=5),       # 吸烟检测帧队列
        }
        
        # 显示队列管理 - 每个任务都有独立的显示队列
        self.display_queues = {
            'flame': queue.Queue(maxsize=10),        # 火焰检测显示队列
            'face': queue.Queue(maxsize=10),         # 人脸识别显示队列
            'meter': queue.Queue(maxsize=10),        # 仪表检测显示队列
            'hardhat': queue.Queue(maxsize=10),      # 安全帽检测显示队列
            'smoking': queue.Queue(maxsize=10),      # 吸烟检测显示队列
        }
        
        # 检测结果存储
        self.detection_results = {
            'flame': 0,                              # 火焰检测结果
            'face': [],                              # 人脸识别结果
            'meter': None,                           # 仪表读数结果
            'hardhat': 0,                            # 安全帽检测结果
            'smoking': 0,                            # 吸烟检测结果
        }
        
        # 异常帧计数器
        self.abnormal_counts = defaultdict(int)     # 各任务的异常帧计数器
        
        # 模型池字典
        self.model_pools = {}                       # 存储各任务的模型池
        
        # 仪表读数器
        self.meter_reader = None                    # 仪表读数器对象
        
        # 任务配置字典 - 从配置文件加载或使用默认配置
        self.task_configs = self._load_task_configs()
    
    def _load_task_configs(self):
        """加载任务配置"""
        if config is not None:
            # 从配置文件加载
            return {
                'flame': {
                    'model_path': config.FLAME_CONFIG['model_path'],
                    'pool_class': rknnPoolExecutor_flame,
                    'func': myFunc_flame,
                    'camera_type': 'rgb',
                    'threshold': config.FLAME_CONFIG['abnormal_threshold'],
                    'window_title': config.FLAME_CONFIG['window_title']
                },
                'face': {
                    'model_path': config.FACE_CONFIG['model_path'],
                    'model_path2': config.FACE_CONFIG['model_path2'],
                    'pool_class': rknnPoolExecutor_face,
                    'func': myFunc_face,
                    'camera_type': 'rgb',
                    'threshold': 0,  # 人脸识别不需要异常帧阈值
                    'window_title': config.FACE_CONFIG['window_title']
                },
                'meter': {
                    'model_path': './yolov7_tiny.rknn',  # 使用存在的模型文件
                    'pool_class': rknnPoolExecutor,
                    'func': myFunc,
                    'camera_type': 'rgb',
                    'threshold': 0,  # 仪表检测不需要异常帧阈值
                    'window_title': '仪表检测'
                },
                'hardhat': {
                    'model_path': config.HARDHAT_CONFIG['model_path'],
                    'pool_class': rknnPoolExecutor_hardhat,
                    'func': myFunc_hardhat,
                    'camera_type': 'rgb',
                    'threshold': config.HARDHAT_CONFIG['abnormal_threshold'],
                    'window_title': config.HARDHAT_CONFIG['window_title']
                },
                'smoking': {
                    'model_path': config.SMOKING_CONFIG['model_path'],
                    'pool_class': rknnPoolExecutor_smoke,
                    'func': myFunc_smoke,
                    'camera_type': 'rgb',
                    'threshold': config.SMOKING_CONFIG['abnormal_threshold'],
                    'window_title': config.SMOKING_CONFIG['window_title']
                }
            }
        else:
            # 使用默认配置
            return {
                'flame': {
                    'model_path': './fire.rknn',
                    'pool_class': rknnPoolExecutor_flame,
                    'func': myFunc_flame,
                    'camera_type': 'rgb',
                    'threshold': 10,
                    'window_title': '火焰检测'
                },
                'face': {
                    'model_path': 'model_data/retinaface_mob.rknn',
                    'model_path2': 'model_data/mobilefacenet.rknn',
                    'pool_class': rknnPoolExecutor_face,
                    'func': myFunc_face,
                    'camera_type': 'rgb',
                    'threshold': 0,  # 人脸识别不需要异常帧阈值
                    'window_title': '人脸识别'
                },
                'meter': {
                    'model_path': './yolov7_tiny.rknn',  # 使用存在的模型文件
                    'pool_class': rknnPoolExecutor,
                    'func': myFunc,
                    'camera_type': 'rgb',
                    'threshold': 0,  # 仪表检测不需要异常帧阈值
                    'window_title': '仪表检测'
                },
                'hardhat': {
                    'model_path': './helmet.rknn',
                    'pool_class': rknnPoolExecutor_hardhat,
                    'func': myFunc_hardhat,
                    'camera_type': 'rgb',
                    'threshold': 10,
                    'window_title': '安全帽检测'
                },
                'smoking': {
                    'model_path': './smoking.rknn',
                    'pool_class': rknnPoolExecutor_smoke,
                    'func': myFunc_smoke,
                    'camera_type': 'rgb',
                    'threshold': 20,
                    'window_title': '吸烟检测'
                }
            }
    
    def open_camera(self, camera_type):
        """打开指定类型的摄像头"""
        if camera_type in self.cameras:             # 如果摄像头已经打开
            return self.cameras[camera_type]        # 直接返回已打开的摄像头
        
        # 根据摄像头类型选择摄像头编号列表
        if camera_type == 'infrared':
            print(f"⚠️ 红外检测已禁用，跳过摄像头类型: {camera_type}")
            return None
        elif camera_type == 'rgb':
            camera_numbers = video_number.rgb_numbers    # RGB摄像头编号列表
        else:
            print(f"❌ 未知的摄像头类型: {camera_type}")
            return None
        
        # 尝试打开摄像头
        for number in camera_numbers:              # 遍历摄像头编号列表
            cap = cv2.VideoCapture(number)         # 尝试打开指定编号的摄像头
            if cap.isOpened():                     # 检查摄像头是否成功打开
                print(f"✅ 成功打开{camera_type}摄像头: {number}")
                self.cameras[camera_type] = cap    # 存储摄像头对象
                self.camera_locks[camera_type] = threading.Lock()  # 创建摄像头锁
                return cap                         # 返回摄像头对象
        
        print(f"❌ 无法打开{camera_type}摄像头")
        return None
    
    def _update_camera_threads(self):
        """更新摄像头采集线程 - 修复版本"""
        # 统计各摄像头类型需要服务的任务
        camera_tasks = defaultdict(list)
        for task_name in self.running_tasks:
            camera_type = self.task_configs[task_name]['camera_type']
            camera_tasks[camera_type].append(task_name)
        
        print(f"🔧 摄像头任务分配: {dict(camera_tasks)}")
        
        # 启动或更新摄像头线程
        for camera_type, tasks in camera_tasks.items():
            if camera_type not in self.camera_threads:
                # 启动新的摄像头线程
                thread = threading.Thread(
                    target=self.camera_capture_worker,
                    args=(camera_type, tasks),
                    name=f"Camera-{camera_type}"
                )
                thread.start()
                self.camera_threads[camera_type] = thread
                print(f"📷 启动{camera_type}摄像头线程，服务任务: {tasks}")
            else:
                # 修复：不重启线程，而是让现有线程继续运行
                # 由于摄像头线程会检查 self.running_tasks，它会自动服务新添加的任务
                print(f"📷 {camera_type}摄像头线程继续运行，服务任务: {tasks}")
    
    def camera_capture_worker(self, camera_type, tasks):
        """摄像头采集工作线程"""
        print(f"📷 启动{camera_type}摄像头采集线程，服务任务: {tasks}")
        
        cam = self.open_camera(camera_type)        # 打开摄像头
        if cam is None:                            # 如果摄像头打开失败
            print(f"❌ {camera_type}摄像头采集线程启动失败")
            return
        
        while not self.stop_event.is_set():       # 当停止事件未设置时，持续运行
            with self.camera_locks[camera_type]:   # 获取摄像头锁
                ret, frame = cam.read()            # 从摄像头读取一帧图像
            
            if not ret:                            # 如果读取失败
                print(f"❌ {camera_type}摄像头读取失败")
                break
            
            # 动态获取当前需要服务的任务列表
            current_tasks = []
            for task_name in self.running_tasks:
                if self.task_configs[task_name]['camera_type'] == camera_type:
                    current_tasks.append(task_name)
            
            # 将帧分发到所有相关任务的队列中
            for task in current_tasks:             # 遍历使用此摄像头的所有任务
                if task in self.running_tasks:     # 如果任务正在运行
                    frame_queue = self.frame_queues[task]  # 获取任务的帧队列
                    if frame_queue.full():         # 如果队列已满
                        try:
                            frame_queue.get_nowait()  # 丢弃最老的帧
                        except queue.Empty:
                            pass
                    frame_queue.put(frame.copy())  # 放入帧的副本（避免多线程竞争）
        
        print(f"📷 {camera_type}摄像头采集线程结束")
    
    def detection_task_worker(self, task_name):
        """检测任务工作线程"""
        print(f"🔍 启动{task_name}检测任务")
        
        config = self.task_configs[task_name]      # 获取任务配置
        
        # 初始化模型池
        try:
            if task_name == 'face':                    # 人脸识别需要双模型
                pool = config['pool_class'](
                    rknnModel1=config['model_path'],
                    rknnModel2=config['model_path2'],
                    TPEs=3,
                    func=config['func']
                )
            else:                                      # 其他任务使用单模型
                pool = config['pool_class'](
                    rknnModel=config['model_path'],
                    TPEs=3,
                    func=config['func']
                )
            
            self.model_pools[task_name] = pool         # 存储模型池
            print(f"✅ {task_name}模型池初始化成功")
        except Exception as e:
            print(f"❌ {task_name}模型池初始化失败: {e}")
            return
        
        frames = 0                                 # 帧计数器
        loopTime = time.time()                     # 循环开始时间
        consecutive_errors = 0                     # 连续错误计数
        
        while not self.stop_event.is_set() and task_name in self.running_tasks:
            try:
                # 从帧队列获取图像
                frame = self.frame_queues[task_name].get(timeout=1)
                if frame is None:
                    continue
                
                # 特殊处理：仪表检测需要初始化读数器
                if task_name == 'meter' and self.meter_reader is None:
                    try:
                        self.meter_reader = MeterReader(frame)
                        print(f"✅ {task_name}仪表读数器初始化成功")
                    except Exception as e:
                        print(f"⚠️ {task_name}仪表读数器初始化失败: {e}")
                
                # 进行检测
                pool.put(frame)                    # 将帧放入模型池
                result, flag = pool.get()          # 获取检测结果
                
                if not flag:                       # 如果检测失败
                    consecutive_errors += 1
                    print(f"⚠️ {task_name}检测失败 (连续错误: {consecutive_errors})")
                    if consecutive_errors >= 5:    # 连续失败5次后退出
                        print(f"❌ {task_name}连续检测失败过多，退出任务")
                        break
                    continue
                
                # 检测成功，重置错误计数
                consecutive_errors = 0
                
                # 处理检测结果
                self._process_detection_result(task_name, result, frame, frames)
                
                frames += 1                        # 帧计数器加1
                if frames % 30 == 0:               # 每30帧打印一次统计信息
                    fps = 30 / (time.time() - loopTime)
                    print(f"📊 {task_name}: {fps:.1f} FPS, 结果: {self.detection_results[task_name]}")
                    loopTime = time.time()
                    
            except queue.Empty:                    # 如果队列为空
                continue
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ {task_name}检测异常: {e} (连续错误: {consecutive_errors})")
                if consecutive_errors >= 5:        # 连续异常5次后退出
                    print(f"❌ {task_name}连续异常过多，退出任务")
                    break
        
        # 清理资源
        try:
            pool.release()                             # 释放模型池
            print(f"🔍 {task_name}检测任务结束")
        except Exception as e:
            print(f"⚠️ {task_name}模型池释放异常: {e}")
    
    def _process_detection_result(self, task_name, result, original_frame, frames):
        """处理检测结果"""
        processed_frame = original_frame.copy()    # 复制原始帧
        
        if task_name == 'flame':
            # 火焰检测结果处理
            processed_frame, class_flame = result
            self.detection_results['flame'] = class_flame
            
            if class_flame == 1:                   # 检测到火焰
                self.abnormal_counts['flame'] += 1
                status_text = f"Fire Detected! (Count: {self.abnormal_counts['flame']})"
                color = (0, 0, 255)                # 红色
                
                if self.abnormal_counts['flame'] >= 10:
                    print(f"🚨 警告: 连续检测到火焰{self.abnormal_counts['flame']}帧!")
            elif class_flame == 0:
                self.abnormal_counts['flame'] = 0
                status_text = "Fire: Normal"
                color = (0, 255, 0)                # 绿色
            else:
                status_text = "Fire: Unknown"
                color = (0, 255, 255)              # 黄色
        
        elif task_name == 'face':
            # 人脸识别结果处理
            processed_frame = result               # 人脸识别返回的是处理后的帧
            
            recognized_names = []
            if func_face.name_ten and len(func_face.name_ten) > 0:
                recognized_names = func_face.name_ten[-1]
            
            self.detection_results['face'] = recognized_names
            
            if recognized_names:
                status_text = f"Recognized: {', '.join(recognized_names)}"
                color = (0, 255, 0)                # 绿色
            else:
                status_text = "Face: Unknown"
                color = (0, 0, 255)                # 红色
        
        elif task_name == 'meter':
            # 仪表检测结果处理
            processed_frame, pointer_mask, scale_mask = result
            meter_value = None
            
            if (pointer_mask is not None and scale_mask is not None and 
                len(pointer_mask) > 0 and len(scale_mask) > 0):
                try:
                    pointer_mask_single = pointer_mask[0].astype(np.uint8) * 255
                    scale_mask_single = scale_mask[0].astype(np.uint8) * 255
                    meter_value = self.meter_reader(pointer_mask_single, scale_mask_single)
                except Exception as e:
                    print(f"⚠️ 仪表读数计算错误: {e}")
            
            self.detection_results['meter'] = meter_value
            
            if meter_value is not None:
                status_text = f"Meter Reading: {meter_value:.3f}"
                color = (0, 255, 0)                # 绿色
            else:
                status_text = "Meter: No complete pointer and scale detected"
                color = (0, 0, 255)                # 红色
        
        elif task_name == 'hardhat':
            # 安全帽检测结果处理
            processed_frame, class_hardhat = result
            self.detection_results['hardhat'] = class_hardhat
            
            if class_hardhat == 1:                 # 未戴安全帽
                self.abnormal_counts['hardhat'] += 1
                status_text = f"No Hardhat Detected! (Count: {self.abnormal_counts['hardhat']})"
                color = (0, 0, 255)                # 红色
                
                if self.abnormal_counts['hardhat'] >= 10:
                    print(f"🚨 警告: 连续检测到未戴安全帽{self.abnormal_counts['hardhat']}帧!")
            elif class_hardhat == 0:
                self.abnormal_counts['hardhat'] = 0
                status_text = "Hardhat: Normal"
                color = (0, 255, 0)                # 绿色
            else:
                status_text = "Hardhat: Unknown"
                color = (0, 255, 255)              # 黄色
        
        elif task_name == 'smoking':
            # 吸烟检测结果处理
            processed_frame, class_smoke = result
            self.detection_results['smoking'] = class_smoke
            
            if class_smoke == 1:                   # 检测到吸烟
                self.abnormal_counts['smoking'] += 1
                status_text = f"Smoking Detected! (Count: {self.abnormal_counts['smoking']})"
                color = (0, 0, 255)                # 红色
                
                if self.abnormal_counts['smoking'] >= 20:
                    print(f"🚨 警告: 连续检测到吸烟{self.abnormal_counts['smoking']}帧!")
            elif class_smoke == 0:
                self.abnormal_counts['smoking'] = 0
                status_text = "Smoking: Normal"
                color = (0, 255, 0)                # 绿色
            else:
                status_text = "Smoking: Unknown"
                color = (0, 255, 255)              # 黄色
        
        # 在图像上添加状态文本
        cv2.putText(processed_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(processed_frame, f"Frame: {frames}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加检测值显示（与单独测试程序一致）
        if task_name in ['flame', 'hardhat', 'smoking']:
            detection_value = self.detection_results[task_name]
            cv2.putText(processed_frame, f"Detection Value: {detection_value}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 将处理后的帧和任务代号一起放入显示队列（与main.py一致）
        task_code = self.get_task_code(task_name)  # 获取任务代号
        self.display_queues[task_name].put((processed_frame, task_code))
    
    def get_task_code(self, task_name):
        """获取任务代号（与main.py的任务代号一致）"""
        task_code_map = {
            'flame': 4,       # 火焰检测  
            'face': 0,        # 人脸识别
            'hardhat': 2,     # 安全帽检测
            'smoking': 5,     # 吸烟检测
            'meter': 1        # 仪表检测
        }
        return task_code_map.get(task_name, 0)
    
    def display_worker(self, task_name):
        """显示工作线程"""
        window_title = self.task_configs[task_name]['window_title']
        print(f"🖥️ 启动{window_title}显示线程")
        
        while not self.stop_event.is_set() and task_name in self.running_tasks:
            try:
                frame, task_code = self.display_queues[task_name].get(timeout=1)  # 获取帧和任务代号
                if frame is None:
                    continue
                
                cv2.imshow(f"Task {task_code}", frame)  # 使用与main.py一致的窗口标题
                
                key = cv2.waitKey(1) & 0xFF        # 检测按键
                if key == ord('q'):                # 按q键退出
                    print(f"🛑 用户在{window_title}窗口按下'q'键")
                    self.stop_all_tasks()          # 停止所有任务
                    break
                    
            except queue.Empty:
                continue
        
        cv2.destroyWindow(f"Task {self.get_task_code(task_name)}")  # 关闭显示窗口（使用正确的窗口名称）
        print(f"🖥️ {window_title}显示线程结束")
    
    def start_task(self, task_name):
        """启动单个检测任务"""
        if task_name not in self.task_configs:
            print(f"❌ 未知的任务: {task_name}")
            return False
        
        if task_name in self.running_tasks:
            print(f"⚠️ 任务 {task_name} 已在运行")
            return True
        
        print(f"🚀 启动任务: {task_name}")
        
        # 添加到运行任务集合
        self.running_tasks.add(task_name)
        
        # 启动检测线程
        detection_thread = threading.Thread(
            target=self.detection_task_worker,
            args=(task_name,),
            name=f"Detection-{task_name}"
        )
        detection_thread.start()
        
        # 启动显示线程
        display_thread = threading.Thread(
            target=self.display_worker,
            args=(task_name,),
            name=f"Display-{task_name}"
        )
        display_thread.start()
        
        # 启动或更新摄像头采集线程
        self._update_camera_threads()
        
        return True
    
    def stop_task(self, task_name):
        """停止单个检测任务"""
        if task_name not in self.running_tasks:
            print(f"⚠️ 任务 {task_name} 未在运行")
            return
        
        print(f"🛑 停止任务: {task_name}")
        self.running_tasks.remove(task_name)
        
        # 更新摄像头采集线程
        self._update_camera_threads()
    
    def start_all_tasks(self):
        """启动所有检测任务"""
        print("🚀 启动所有检测任务...")
        
        success_count = 0
        for task_name in self.task_configs.keys():
            if self.start_task(task_name):
                success_count += 1
                time.sleep(0.5)                    # 延迟启动，避免资源竞争
        
        print(f"✅ 成功启动 {success_count}/{len(self.task_configs)} 个任务")
        return success_count == len(self.task_configs)
    
    def stop_all_tasks(self):
        """停止所有检测任务"""
        print("🛑 停止所有检测任务...")
        
        self.stop_event.set()                      # 设置全局停止事件
        self.running_tasks.clear()                 # 清空运行任务集合
        
        # 等待一段时间让线程自然结束
        time.sleep(2)
        
        # 关闭所有摄像头
        for camera_type, cam in self.cameras.items():
            if cam is not None:
                cam.release()
                print(f"📷 关闭{camera_type}摄像头")
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print("✅ 所有任务已停止")
    
    def get_detection_status(self):
        """获取当前检测状态"""
        status = {
            'running_tasks': list(self.running_tasks),
            'detection_results': self.detection_results.copy(),
            'abnormal_counts': dict(self.abnormal_counts),
            'cameras': list(self.cameras.keys())
        }
        return status
    
    def print_status(self):
        """打印当前系统状态"""
        print("\n" + "="*60)
        print("📊 多任务检测系统状态")
        print("="*60)
        print(f"🏃 运行中的任务: {len(self.running_tasks)}")
        for task in self.running_tasks:
            result = self.detection_results[task]
            print(f"   • {task}: {result}")
        
        print(f"📷 活跃摄像头: {len(self.cameras)}")
        for camera_type in self.cameras:
            print(f"   • {camera_type}")
        
        print(f"🚨 异常计数:")
        for task, count in self.abnormal_counts.items():
            if count > 0:
                print(f"   • {task}: {count}")
        print("="*60)

    def check_camera_status(self):
        """检查摄像头状态"""
        print("\n🔍 摄像头状态检查:")
        print("=" * 40)
        
        # 检查可用的视频设备
        import os
        video_devices = []
        for i in range(10):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                video_devices.append(i)
        
        print(f"📁 发现的视频设备: {video_devices}")
        
        # 检查video_number模块配置
        try:
            print(f"📋 video_number.rgb_numbers: {video_number.rgb_numbers}")
            print(f"📋 video_number.inf_numbers: {video_number.inf_numbers}")
        except Exception as e:
            print(f"⚠️ 无法读取video_number配置: {e}")
        
        # 测试摄像头打开
        print("\n🧪 测试摄像头打开:")
        for i in range(min(5, len(video_devices))):
            device_num = video_devices[i]
            try:
                cap = cv2.VideoCapture(device_num)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ /dev/video{device_num}: 可用 (帧大小: {frame.shape})")
                    else:
                        print(f"⚠️ /dev/video{device_num}: 打开但无法读取帧")
                    cap.release()
                else:
                    print(f"❌ /dev/video{device_num}: 无法打开")
            except Exception as e:
                print(f"❌ /dev/video{device_num}: 错误 - {e}")
        
        print("=" * 40)

def signal_handler(sig, frame):
    """信号处理函数"""
    print("\n🛑 接收到退出信号，正在关闭系统...")
    global detection_system
    if detection_system:
        detection_system.stop_all_tasks()
    sys.exit(0)

def main():
    """主函数"""
    global detection_system
    
    print("🌟 多任务视觉检测系统启动 - 修复版本")
    print("=" * 60)
    
    # 初始化配置
    if config is not None:
        try:
            if config.load_config():
                print("✅ 配置系统初始化成功")
                config.print_config_summary()
            else:
                print("⚠️ 配置验证失败，使用默认配置")
        except Exception as e:
            print(f"⚠️ 配置初始化出错: {e}，使用默认配置")
    
    print("支持的任务:")
    print("• 火焰检测") 
    print("• 人脸识别")
    print("• 仪表读数")
    print("• 安全帽检测")
    print("• 吸烟检测")
    print("=" * 60)
    print("控制说明:")
    print("• 按 'q' 键退出任意检测窗口")
    print("• 按 Ctrl+C 退出程序")
    print("=" * 60)
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建检测系统
    detection_system = MultiTaskDetectionSystem()
    
    # 检查摄像头状态
    detection_system.check_camera_status()
    
    try:
        # 启动所有任务
        if detection_system.start_all_tasks():
            print("✅ 所有任务启动成功！")
            
            # 定期打印状态
            while not detection_system.stop_event.is_set():
                time.sleep(30)                     # 每30秒打印一次状态
                if detection_system.running_tasks:  # 如果还有任务在运行
                    detection_system.print_status()
                else:
                    break
        else:
            print("❌ 部分任务启动失败")
    
    except KeyboardInterrupt:
        print("\n🛑 用户中断程序")
    except Exception as e:
        print(f"❌ 系统异常: {e}")
    finally:
        detection_system.stop_all_tasks()
        print("👋 程序结束")

if __name__ == "__main__":
    detection_system = None                        # 全局变量，用于信号处理
    main()
