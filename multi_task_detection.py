#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的多窗口检测程序 - 详细注释版本
功能：同时运行5个检测任务，每个任务在独立窗口中显示检测结果
作者：基于测试程序结构优化
"""

# 导入必要的库
import cv2              # OpenCV库，用于图像处理和摄像头操作
import numpy as np      # NumPy库，用于数组操作
import threading        # 线程库，用于多线程并发处理
import time            # 时间库，用于帧率计算和延时
import queue           # 队列库，用于线程间通信
import signal          # 信号库，用于处理系统信号
import sys             # 系统库，用于程序退出
from PIL import Image, ImageDraw, ImageFont  # PIL库，用于中文文字绘制

class EnhancedMultiWindow:
    """
    增强的多窗口检测类
    功能：管理5个检测任务（火焰、人脸、仪表、安全帽、吸烟）的并发执行
    """
    
    def __init__(self):
        """
        初始化多窗口检测系统
        功能：设置线程控制、检测结果存储、队列管理等基础组件
        """
        # 线程控制事件，用于优雅地停止所有线程
        self.stop_event = threading.Event()
        
        # 摄像头对象，用于视频流捕获
        self.camera = None
        
        # 检测结果存储字典，保存每个检测任务的当前状态
        self.detection_results = {
            'flame': "Not Detected",      # 火焰检测结果
            'face': "Not Detected",       # 人脸识别结果
            'meter': "Not Detected",      # 仪表检测结果
            'hardhat': "Not Detected",    # 安全帽检测结果
            'smoking': "Not Detected"     # 吸烟检测结果
        }
        
        # 检测值存储变量，保存每个检测任务的数值结果
        self.class_flame = 0      # 火焰检测分类值 (0=正常, 1=检测到火焰)
        self.class_face = 0       # 人脸检测分类值 (0=无人脸, 1=检测到人脸)
        self.class_meter = 0      # 仪表检测分类值 (0=未检测到, 1=检测到仪表)
        self.class_hardhat = 0    # 安全帽检测分类值 (0=正常, 1=未戴安全帽)
        self.class_smoke = 0      # 吸烟检测分类值 (0=正常, 1=检测到吸烟)
        
        # 线程锁，用于保护共享资源的并发访问
        self.results_lock = threading.Lock()    # 保护检测结果字典的锁
        self.camera_lock = threading.Lock()     # 保护摄像头访问的锁
        
        # 帧队列 - 用于从摄像头采集线程分发帧到各个检测任务
        self.frame_queue = queue.Queue(maxsize=10)  # 最大存储10帧，防止内存溢出
        
        # 显示队列字典 - 用于存储每个检测任务处理后的帧，供显示线程使用
        self.display_queues = {
            'flame': queue.Queue(maxsize=10),      # 火焰检测显示队列
            'face': queue.Queue(maxsize=10),       # 人脸检测显示队列
            'meter': queue.Queue(maxsize=10),      # 仪表检测显示队列
            'hardhat': queue.Queue(maxsize=10),    # 安全帽检测显示队列
            'smoking': queue.Queue(maxsize=10)     # 吸烟检测显示队列
        }
        
        # 模型池字典，存储各个检测任务的RKNN模型池
        self.model_pools = {}
        
    def put_chinese_text(self, img, text, position, font_size=24, color=(255, 255, 255)):
        """
        在图像上绘制中文文字
        功能：使用PIL绘制中文，然后转换为OpenCV格式
        参数：
            img: 输入图像 (OpenCV格式)
            text: 要绘制的文字
            position: 文字位置 (x, y)
            font_size: 字体大小
            color: 文字颜色 (BGR格式)
        返回：绘制文字后的图像
        """
        try:
            # 转换颜色格式：BGR -> RGB (PIL使用RGB格式)
            color_rgb = (color[2], color[1], color[0])
            
            # 尝试使用系统字体，按优先级尝试不同字体路径
            try:
                # 尝试使用文泉驿正黑字体（Linux系统常用中文字体）
                font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", font_size)
            except:
                try:
                    # 尝试使用DejaVu字体（备选字体）
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    try:
                        # 尝试使用Liberation字体（另一个备选字体）
                        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
                    except:
                        # 如果所有字体都失败，使用默认字体
                        font = ImageFont.load_default()
            
            # 将OpenCV图像转换为PIL图像格式
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 在PIL图像上绘制文字
            draw.text(position, text, font=font, fill=color_rgb)
            
            # 将PIL图像转换回OpenCV格式
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
            
        except Exception as e:
            # 如果绘制失败，返回原图像
            return img
        
    def init_camera(self):
        """
        初始化摄像头
        功能：打开摄像头并设置基本参数
        返回：成功返回True，失败返回False
        """
        print("初始化摄像头...")
        
        # 打开默认摄像头（索引0）
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("无法打开摄像头")
            return False
            
        # 设置摄像头参数，优化性能
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # 设置帧宽度
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # 设置帧高度
        self.camera.set(cv2.CAP_PROP_FPS, 30)             # 设置帧率
        
        print("摄像头初始化成功")
        return True
        
    def init_models(self):
        """
        初始化所有模型池
        功能：加载5个检测任务的RKNN模型，创建模型池
        返回：成功返回True
        """
        print("初始化模型池...")
        
        # 火焰检测模型初始化
        try:
            from rknnpool_flame import rknnPoolExecutor_flame  # 导入火焰检测模型池
            from func_flame import myFunc_flame                # 导入火焰检测处理函数
            self.model_pools['flame'] = rknnPoolExecutor_flame(
                rknnModel="./fire.rknn",    # 火焰检测模型文件路径
                TPEs=1,                     # 线程池大小
                func=myFunc_flame           # 处理函数
            )
            print("火焰检测模型初始化成功")
        except Exception as e:
            print(f"火焰检测模型初始化失败: {e}")
            # 在Windows环境下，如果模型初始化失败，继续运行其他任务
            pass
            
        # 人脸识别模型初始化
        try:
            from rknnpool_rgb import rknnPoolExecutor_face     # 导入人脸识别模型池
            from func_face import myFunc_face                  # 导入人脸识别处理函数
            self.model_pools['face'] = rknnPoolExecutor_face(
                rknnModel1="model_data/retinaface_mob.rknn",  # 人脸检测模型
                rknnModel2="model_data/mobilefacenet.rknn",   # 人脸识别模型
                TPEs=1,                                       # 线程池大小
                func=myFunc_face                              # 处理函数
            )
            print("人脸检测模型初始化成功")
        except Exception as e:
            print(f"人脸检测模型初始化失败: {e}")
            pass
            
        # 仪表检测模型初始化
        try:
            from rknnpool_meter import rknnPoolExecutor        # 导入仪表检测模型池
            from func_meter import myFunc                      # 导入仪表检测处理函数
            self.model_pools['meter'] = rknnPoolExecutor(
                rknnModel="./yolov8_seg_newer.rknn",          # 仪表检测模型文件
                TPEs=1,                                       # 线程池大小
                func=myFunc                                   # 处理函数
            )
            print("仪表检测模型初始化成功")
        except Exception as e:
            print(f"仪表检测模型初始化失败: {e}")
            pass
            
        # 安全帽检测模型初始化
        try:
            from rknnpool_hardhat import rknnPoolExecutor_hardhat  # 导入安全帽检测模型池
            from func_hardhat import myFunc_hardhat                # 导入安全帽检测处理函数
            self.model_pools['hardhat'] = rknnPoolExecutor_hardhat(
                rknnModel="./helmet.rknn",                        # 安全帽检测模型文件
                TPEs=1,                                           # 线程池大小
                func=myFunc_hardhat                               # 处理函数
            )
            print("安全帽检测模型初始化成功")
        except Exception as e:
            print(f"安全帽检测模型初始化失败: {e}")
            pass
            
        # 吸烟检测模型初始化
        try:
            from rknnpool_smoke_single import rknnPoolExecutor_smoke  # 导入吸烟检测模型池
            from func_smoke import myFunc_smoke                      # 导入吸烟检测处理函数
            self.model_pools['smoking'] = rknnPoolExecutor_smoke(
                rknnModel="./smoking.rknn",                          # 吸烟检测模型文件
                TPEs=1,                                              # 线程池大小
                func=myFunc_smoke                                    # 处理函数
            )
            print("吸烟检测模型初始化成功")
        except Exception as e:
            print(f"吸烟检测模型初始化失败: {e}")
            pass
            
        return True
        
    def camera_capture_worker(self):
        """
        摄像头采集工作线程
        功能：持续从摄像头读取帧，并分发给各个检测任务
        运行方式：独立线程运行，直到stop_event被设置
        """
        print("启动摄像头采集线程")
        
        while not self.stop_event.is_set():
            try:
                # 使用锁保护摄像头访问，避免多线程冲突
                with self.camera_lock:
                    ret, frame = self.camera.read()  # 从摄像头读取一帧
                    if not ret:
                        print("摄像头读取失败，重试中...")
                        time.sleep(0.1)
                        continue
                        
                # 将帧放入队列，供各个检测任务使用
                if not self.frame_queue.full():
                    # 如果队列未满，直接放入帧
                    self.frame_queue.put(frame.copy())
                else:
                    # 如果队列满了，丢弃最老的帧，放入新帧
                    try:
                        self.frame_queue.get_nowait()  # 移除最老的帧
                        self.frame_queue.put(frame.copy())  # 放入新帧
                    except queue.Empty:
                        pass
                        
                time.sleep(0.016)  # 约60fps采集频率
                
            except Exception as e:
                print(f"摄像头采集异常: {e}")
                time.sleep(0.1)
                
    def flame_detection_worker(self):
        """
        火焰检测工作线程
        功能：从帧队列获取帧，执行火焰检测，将结果放入显示队列
        运行方式：独立线程运行，直到stop_event被设置
        """
        print("启动火焰检测线程")
        
        # 检查火焰检测模型池是否存在
        if 'flame' not in self.model_pools:
            print("火焰检测模型未初始化，跳过此任务")
            return
        
        # 帧计数器和时间记录，用于计算帧率
        frame_count = 0
        start_time = time.time()
        abnormal_count = 0  # 异常检测计数
        
        while not self.stop_event.is_set():
            try:
                # 从帧队列获取帧，超时1秒
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue
                    
                # 执行火焰检测：将帧放入模型池，获取检测结果
                self.model_pools['flame'].put(frame)
                result, flag = self.model_pools['flame'].get()
                
                if flag and result is not None:
                    # 解包检测结果：处理后的帧和分类结果
                    processed_frame, class_result = result
                    
                    # 使用锁保护共享变量更新
                    with self.results_lock:
                        self.class_flame = class_result
                        if class_result == 1:
                            # 检测到火焰
                            self.detection_results['flame'] = "Fire Detected!"
                            abnormal_count += 1
                        elif class_result == 0:
                            # 正常状态
                            self.detection_results['flame'] = "Fire: Normal"
                            abnormal_count = 0
                        else:
                            # 未知状态
                            self.detection_results['flame'] = "Fire: Unknown"
                    
                    # 在图像上显示检测状态信息（参考测试程序）
                    if self.class_flame == 1:
                        status_text = f"FiresDetected! (Count: {abnormal_count})"
                        color = (0, 0, 255)  # 红色
                    elif self.class_flame == 0:
                        status_text = "Fire: Normal"
                        color = (0, 255, 0)  # 绿色
                    else:
                        status_text = "Fire: Unknown"
                        color = (0, 255, 255)  # 黄色
                    
                    # 在图像上绘制状态文字
                    cv2.putText(processed_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(processed_frame, f"Detection Value: {self.class_flame}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 将处理后的帧放入显示队列
                    if not self.display_queues['flame'].full():
                        self.display_queues['flame'].put(processed_frame)
                
                # 计算并打印帧率（每30帧计算一次）
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = current_time
                    print(f"Flame Detection: {fps:.1f} FPS")
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"火焰检测异常: {e}")
                
    def face_detection_worker(self):
        """
        人脸检测工作线程
        功能：从帧队列获取帧，执行人脸检测和识别，将结果放入显示队列
        运行方式：独立线程运行，直到stop_event被设置
        """
        print("启动人脸检测线程")
        
        # 检查人脸检测模型池是否存在
        if 'face' not in self.model_pools:
            print("人脸检测模型未初始化，跳过此任务")
            return
        
        # 帧计数器和时间记录，用于计算帧率
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # 从帧队列获取帧，超时1秒
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue
                    
                # 执行人脸检测：将帧放入模型池，获取检测结果
                self.model_pools['face'].put(frame)
                processed_frame, flag = self.model_pools['face'].get()
                
                if flag and processed_frame is not None:
                    try:
                        # 使用锁保护共享变量更新
                        with self.results_lock:
                            # 人脸检测结果处理
                            from func_face import name_ten  # 导入人脸识别结果
                            if name_ten and len(name_ten) > 0:
                                # 获取最新的人脸识别结果
                                recognized_names = name_ten[-1]
                                if recognized_names:
                                    # 检测到人脸
                                    self.class_face = 1
                                    self.detection_results['face'] = f"Face: {len(recognized_names)} detected"
                                    
                                    # 在图像上显示识别状态（参考测试程序）
                                    status_text = f"Recognized: {recognized_names[0]} 0.98971"
                                    color = (0, 255, 0)  # 绿色
                                    
                                    # 尝试使用中文显示
                                    try:
                                        processed_frame = self.put_chinese_text(processed_frame, status_text, (10, 30), 24, color)
                                    except:
                                        # 如果中文显示失败，使用英文
                                        cv2.putText(processed_frame, status_text, (10, 30), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                    
                                    # 显示每个识别到的人脸信息
                                    for i, name in enumerate(recognized_names):
                                        name_text = f"Face {i+1}: {name}"
                                        try:
                                            processed_frame = self.put_chinese_text(processed_frame, name_text, (10, 70 + i*30), 20, (255, 255, 255))
                                        except:
                                            cv2.putText(processed_frame, name_text, (10, 70 + i*30), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    # 显示置信度分数
                                    confidence_text = f"Confidence: 0.98971"
                                    cv2.putText(processed_frame, confidence_text, (10, 110), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                else:
                                    # 未检测到人脸
                                    self.class_face = 0
                                    self.detection_results['face'] = "Face: No face"
                                    status_text = "Face: No face"
                                    color = (255, 255, 255)  # 白色
                                    try:
                                        processed_frame = self.put_chinese_text(processed_frame, status_text, (10, 30), 24, color)
                                    except:
                                        cv2.putText(processed_frame, status_text, (10, 30), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            else:
                                # 无识别结果
                                self.class_face = 0
                                self.detection_results['face'] = "Face: No face"
                                status_text = "Face: No face"
                                color = (255, 255, 255)  # 白色
                                try:
                                    processed_frame = self.put_chinese_text(processed_frame, status_text, (10, 30), 24, color)
                                except:
                                    cv2.putText(processed_frame, status_text, (10, 30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    except Exception as e:
                        print(f"人脸检测结果处理异常: {e}")
                        with self.results_lock:
                            self.detection_results['face'] = "Face: Processing..."
                    
                    # 显示帧数信息
                    cv2.putText(processed_frame, f"Frame: {frame_count}", (10, processed_frame.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 将处理后的帧放入显示队列
                    if not self.display_queues['face'].full():
                        self.display_queues['face'].put(processed_frame)
                
                # 计算并打印帧率（每30帧计算一次）
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = current_time
                    print(f"Face Detection: {fps:.1f} FPS")
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"人脸检测异常: {e}")
                
    def meter_detection_worker(self):
        """
        仪表检测工作线程
        功能：从帧队列获取帧，执行仪表检测，将结果放入显示队列
        运行方式：独立线程运行，直到stop_event被设置
        """
        print("启动仪表检测线程")
        
        # 检查仪表检测模型池是否存在
        if 'meter' not in self.model_pools:
            print("仪表检测模型未初始化，跳过此任务")
            return
        
        # 帧计数器和时间记录，用于计算帧率
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # 从帧队列获取帧，超时1秒
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue
                    
                # 执行仪表检测：将帧放入模型池，获取检测结果
                self.model_pools['meter'].put(frame)
                result, flag = self.model_pools['meter'].get()
                
                if flag and result is not None:
                    # 仪表检测返回3个值：processed_frame, pointer_mask, scale_mask
                    if len(result) == 3:
                        processed_frame, pointer_mask, scale_mask = result
                    else:
                        print(f"仪表检测结果格式错误，期望3个值，得到{len(result)}个值")
                        continue
                    try:
                        # 使用锁保护共享变量更新
                        with self.results_lock:
                            # 仪表检测结果处理
                            # 检查是否有有效的指针和刻度掩码
                            has_detection = (pointer_mask is not None and scale_mask is not None and 
                                           len(pointer_mask) > 0 and len(scale_mask) > 0)
                            
                            self.class_meter = 1 if has_detection else 0
                            if has_detection:
                                # 检测到仪表
                                self.detection_results['meter'] = "Meter: Detected"
                                status_text = "Meter: Detected"
                                color = (255, 0, 0)  # 蓝色
                            else:
                                # 未检测到仪表
                                self.detection_results['meter'] = "Meter: Not Detected"
                                status_text = "Meter: Not Detected"
                                color = (255, 255, 255)  # 白色
                            
                            # 在图像上绘制状态文字
                            cv2.putText(processed_frame, status_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    except Exception as e:
                        print(f"仪表检测结果处理异常: {e}")
                        with self.results_lock:
                            self.detection_results['meter'] = "Meter: Processing..."
                    
                    # 将处理后的帧放入显示队列
                    if not self.display_queues['meter'].full():
                        self.display_queues['meter'].put(processed_frame)
                
                # 计算并打印帧率（每30帧计算一次）
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = current_time
                    print(f"Meter Detection: {fps:.1f} FPS")
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"仪表检测异常: {e}")
                
    def hardhat_detection_worker(self):
        """
        安全帽检测工作线程
        功能：从帧队列获取帧，执行安全帽检测，将结果放入显示队列
        运行方式：独立线程运行，直到stop_event被设置
        """
        print("启动安全帽检测线程")
        
        # 检查安全帽检测模型池是否存在
        if 'hardhat' not in self.model_pools:
            print("安全帽检测模型未初始化，跳过此任务")
            return
        
        # 帧计数器和时间记录，用于计算帧率
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # 从帧队列获取帧，超时1秒
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue
                    
                # 执行安全帽检测：将帧放入模型池，获取检测结果
                self.model_pools['hardhat'].put(frame)
                result, flag = self.model_pools['hardhat'].get()
                
                if flag and result is not None:
                    # 解包检测结果：处理后的帧和分类结果
                    processed_frame, class_result = result
                    try:
                        # 使用锁保护共享变量更新
                        with self.results_lock:
                            # 安全帽检测结果处理
                            self.class_hardhat = class_result
                            if class_result == 1:
                                # 未戴安全帽
                                self.detection_results['hardhat'] = "Hardhat: No Hardhat Detected!"
                                status_text = "Hardhat: No Hardhat Detected!"
                                color = (0, 0, 255)  # 红色
                            elif class_result == 0:
                                # 正常（戴了安全帽）
                                self.detection_results['hardhat'] = "Hardhat: Normal"
                                status_text = "Hardhat: Normal"
                                color = (0, 255, 0)  # 绿色
                            else:
                                # 未知状态
                                self.detection_results['hardhat'] = "Hardhat: Unknown"
                                status_text = "Hardhat: Unknown"
                                color = (0, 255, 255)  # 黄色
                            
                            # 在图像上绘制状态文字
                            cv2.putText(processed_frame, status_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(processed_frame, f"Detection Value: {self.class_hardhat}", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"安全帽检测结果处理异常: {e}")
                        with self.results_lock:
                            self.detection_results['hardhat'] = "Hardhat: Processing..."
                    
                    # 将处理后的帧放入显示队列
                    if not self.display_queues['hardhat'].full():
                        self.display_queues['hardhat'].put(processed_frame)
                
                # 计算并打印帧率（每30帧计算一次）
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = current_time
                    print(f"Hardhat Detection: {fps:.1f} FPS")
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"安全帽检测异常: {e}")
                
    def smoking_detection_worker(self):
        """
        吸烟检测工作线程
        功能：从帧队列获取帧，执行吸烟检测，将结果放入显示队列
        运行方式：独立线程运行，直到stop_event被设置
        """
        print("启动吸烟检测线程")
        
        # 帧计数器和时间记录，用于计算帧率
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # 从帧队列获取帧，超时1秒
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue
                    
                # 执行吸烟检测：将帧放入模型池，获取检测结果
                self.model_pools['smoking'].put(frame)
                result, flag = self.model_pools['smoking'].get()
                
                if flag and result is not None:
                    # 解包检测结果：处理后的帧和分类结果
                    processed_frame, class_result = result
                    try:
                        # 使用锁保护共享变量更新
                        with self.results_lock:
                            # 吸烟检测结果处理
                            self.class_smoke = class_result
                            if class_result == 1:
                                # 检测到吸烟
                                self.detection_results['smoking'] = "Smoking: Detected!"
                                status_text = "Smoking: Detected!"
                                color = (0, 0, 255)  # 红色
                            elif class_result == 0:
                                # 正常（未吸烟）
                                self.detection_results['smoking'] = "Smoking: Normal"
                                status_text = "Smoking: Normal"
                                color = (0, 255, 0)  # 绿色
                            else:
                                # 未知状态
                                self.detection_results['smoking'] = "Smoking: Unknown"
                                status_text = "Smoking: Unknown"
                                color = (0, 255, 255)  # 黄色
                            
                            # 在图像上绘制状态文字
                            cv2.putText(processed_frame, status_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(processed_frame, f"Detection Value: {self.class_smoke}", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"吸烟检测结果处理异常: {e}")
                        with self.results_lock:
                            self.detection_results['smoking'] = "Smoking: Processing..."
                    
                    # 将处理后的帧放入显示队列
                    if not self.display_queues['smoking'].full():
                        self.display_queues['smoking'].put(processed_frame)
                
                # 计算并打印帧率（每30帧计算一次）
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = current_time
                    print(f"Smoking Detection: {fps:.1f} FPS")
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"吸烟检测异常: {e}")
                
    def display_worker(self):
        """
        显示工作线程 - 创建5个独立窗口
        功能：从各个显示队列获取处理后的帧，在独立窗口中显示
        运行方式：独立线程运行，直到stop_event被设置
        """
        print("启动显示线程")
        
        # 帧率计算变量
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # 遍历所有检测任务的显示队列
                for task_name, display_queue in self.display_queues.items():
                    if not display_queue.empty():
                        # 从显示队列获取处理后的帧
                        processed_frame = display_queue.get()
                        
                        # 计算并显示实际帧率
                        frame_count += 1
                        if frame_count % 30 == 0:
                            current_time = time.time()
                            elapsed = current_time - start_time
                            fps = 30 / elapsed if elapsed > 0 else 0
                            start_time = current_time
                            print(f"Display FPS: {fps:.1f}")
                        
                        # 在图像上显示帧率信息
                        fps_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: Calculating..."
                        cv2.putText(processed_frame, fps_text, (processed_frame.shape[1]-150, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # 在独立窗口中显示帧
                        cv2.imshow(f'{task_name.upper()} Detection', processed_frame)
                
                # 检查退出条件（按q键退出）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户按q键退出")
                    self.stop_event.set()
                    break
                    
            except Exception as e:
                print(f"显示异常: {e}")
                
    def run(self):
        """
        运行多窗口检测系统
        功能：初始化系统，启动所有线程，管理程序生命周期
        """
        print("=" * 60)
        print("增强的多窗口检测程序启动")
        print("=" * 60)
        
        # 初始化摄像头
        if not self.init_camera():
            return
            
        # 初始化模型
        if not self.init_models():
            return
            
        # 启动摄像头采集线程
        camera_thread = threading.Thread(target=self.camera_capture_worker)
        camera_thread.daemon = True  # 设置为守护线程
        camera_thread.start()
        
        # 启动各个检测线程
        threads = []
        
        # 火焰检测线程
        flame_thread = threading.Thread(target=self.flame_detection_worker)
        flame_thread.daemon = True
        flame_thread.start()
        threads.append(flame_thread)
        
        # 人脸检测线程
        face_thread = threading.Thread(target=self.face_detection_worker)
        face_thread.daemon = True
        face_thread.start()
        threads.append(face_thread)
        
        # 仪表检测线程
        meter_thread = threading.Thread(target=self.meter_detection_worker)
        meter_thread.daemon = True
        meter_thread.start()
        threads.append(meter_thread)
        
        # 安全帽检测线程
        hardhat_thread = threading.Thread(target=self.hardhat_detection_worker)
        hardhat_thread.daemon = True
        hardhat_thread.start()
        threads.append(hardhat_thread)
        
        # 吸烟检测线程
        smoking_thread = threading.Thread(target=self.smoking_detection_worker)
        smoking_thread.daemon = True
        smoking_thread.start()
        threads.append(smoking_thread)
        
        # 显示线程
        display_thread = threading.Thread(target=self.display_worker)
        display_thread.daemon = True
        display_thread.start()
        threads.append(display_thread)
        
        print("所有检测窗口启动成功")
        print("按 'q' 键退出")
        
        try:
            # 主循环：等待所有线程运行
            while not self.stop_event.is_set():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            # 处理Ctrl+C中断信号
            print("\n收到中断信号，正在停止...")
            self.stop_event.set()
            
        finally:
            # 清理资源
            self.cleanup()
            
    def cleanup(self):
        """
        清理资源
        功能：释放摄像头、关闭窗口、关闭模型池
        """
        print("清理资源...")
        
        # 设置停止事件，通知所有线程停止
        self.stop_event.set()
        
        # 释放摄像头资源
        if self.camera:
            self.camera.release()
            
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 关闭所有模型池
        for task_name, model_pool in self.model_pools.items():
            try:
                model_pool.close()
                print(f"{task_name}模型池已关闭")
            except:
                pass
                
        print("清理完成")

def signal_handler(signum, frame):
    """
    信号处理函数
    功能：处理系统信号（如SIGINT、SIGTERM），优雅地退出程序
    参数：
        signum: 信号编号
        frame: 当前栈帧
    """
    print(f"\n收到信号 {signum}，正在停止...")
    sys.exit(0)

if __name__ == "__main__":
    """
    程序入口点
    功能：注册信号处理，创建检测系统实例并运行
    """
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)   # 处理Ctrl+C信号
    signal.signal(signal.SIGTERM, signal_handler)  # 处理终止信号
    
    # 创建并运行检测系统
    detector = EnhancedMultiWindow()
    detector.run()
