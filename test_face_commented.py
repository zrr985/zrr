#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别测试程序
独立测试RGB摄像头的人脸识别功能
"""

# 导入所需的库
import cv2                                          # OpenCV库，用于图像处理和摄像头操作
import numpy as np                                  # 数值计算库
import time                                         # 时间库，用于计算帧率和时间间隔
import threading                                    # 线程库，用于多线程处理
import queue                                        # 队列库，用于线程间数据传递
import video_number                                 # 摄像头编号配置模块
from rknnpool_rgb import rknnPoolExecutor_face     # RKNN模型池执行器（人脸识别专用）
from func_face import myFunc_face                   # 人脸识别功能函数
from PIL import Image, ImageDraw, ImageFont        # 图像处理库
import func_face                                    # 导入模块以访问name_ten全局变量

class FaceRecognitionTest:
    """人脸识别测试类"""
    
    def __init__(self):
        """初始化测试类的各种参数和队列"""
        self.stop_event = threading.Event()         # 线程停止事件，用于控制所有线程的退出
        self.frame_queue = queue.Queue(maxsize=5)   # 帧队列，存储从摄像头读取的图像帧（最大5帧）
        self.display_queue = queue.Queue(maxsize=10) # 显示队列，存储处理后的图像帧（最大10帧）
    
    def put_chinese_text(self, img, text, position, font_size=24, color=(255, 255, 255)):
        """
        在图像上绘制中文文字
        功能：使用PIL绘制中文，然后转换为OpenCV格式
        """
        try:
            # 转换颜色格式：BGR -> RGB (PIL使用RGB)
            color_rgb = (color[2], color[1], color[0])
            
            # 尝试使用系统字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    try:
                        # 尝试更多字体路径
                        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            
            # 创建PIL图像
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 绘制文字
            draw.text(position, text, font=font, fill=color_rgb)
            
            # 转换回OpenCV格式
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
            
        except Exception as e:
            # 如果失败，返回原图
            return img
        
    def open_camera(self, camera_numbers):
        """尝试打开RGB摄像头"""
        for number in camera_numbers:              # 遍历配置文件中的摄像头编号列表
            cap = cv2.VideoCapture(number)         # 尝试打开指定编号的摄像头
            if cap.isOpened():                     # 检查摄像头是否成功打开
                print(f"成功打开RGB摄像头: {number}")   # 打印成功信息
                return cap                         # 返回摄像头对象
        return None                                # 如果所有摄像头都无法打开，返回None
    
    def camera_capture(self, cam):
        """摄像头帧采集线程函数"""
        while not self.stop_event.is_set():       # 当停止事件未设置时，持续运行
            ret, frame = cam.read()                # 从摄像头读取一帧图像
            if not ret:                            # 如果读取失败
                print("摄像头读取失败")              # 打印错误信息
                break                              # 跳出循环
            
            if self.frame_queue.full():            # 如果帧队列已满
                self.frame_queue.get()             # 丢弃最老的帧，保持队列大小
            self.frame_queue.put(frame)            # 将新帧放入队列
            
    def face_recognition_task(self):
        """人脸识别任务线程函数"""
        print("人脸识别任务启动")                   # 打印任务启动信息
        model_path = 'model_data/retinaface_mob.rknn'   # 人脸检测模型文件路径
        model_path2 = 'model_data/mobilefacenet.rknn'   # 人脸特征提取模型文件路径
        TPEs = 3                                   # 线程池执行器数量（Thread Pool Executors）
        
        # 初始化RKNN模型池（使用双模型：人脸检测+人脸识别）
        pool = rknnPoolExecutor_face(
            rknnModel1=model_path,                 # 人脸检测模型路径
            rknnModel2=model_path2,                # 人脸特征提取模型路径
            TPEs=TPEs,                             # 线程池大小
            func=myFunc_face                       # 人脸识别函数
        )
        
        frames = 0                                 # 帧计数器
        loopTime = time.time()                     # 循环开始时间，用于计算帧率
        
        while not self.stop_event.is_set():       # 当停止事件未设置时，持续运行
            try:
                frame = self.frame_queue.get(timeout=1)  # 从帧队列获取图像（超时1秒）
                if frame is None:                  # 如果获取的帧为空
                    continue                       # 跳过此次循环
                
                pool.put(frame)                    # 将帧放入模型池进行处理
                processed_frame, flag = pool.get() # 获取处理结果
                
                if not flag:                       # 如果处理失败
                    print("处理失败，flag返回 False")  # 打印错误信息
                    break                          # 跳出循环
                
                # 获取识别结果
                recognized_names = []              # 初始化识别到的姓名列表
                if func_face.name_ten and len(func_face.name_ten) > 0:  # 如果有识别结果
                    recognized_names = func_face.name_ten[-1]  # 获取最新的识别结果（最后一个元素）
                
                # 在图像上显示识别状态
                if recognized_names:               # 如果识别到人脸
                    status_text = f"Recognized: {', '.join(recognized_names)}"  # 生成英文状态文本
                    color = (0, 255, 0)            # 设置文本颜色为绿色（BGR格式）
                    
                    # 显示每个识别到的人脸信息
                    for i, name in enumerate(recognized_names):  # 遍历识别到的姓名
                        name_text = f"Face {i+1}: {name}"  # 生成人脸信息文本
                        # 使用中文显示函数
                        try:
                            processed_frame = self.put_chinese_text(processed_frame, name_text, (10, 70 + i*30), 20, (255, 255, 255))
                        except:
                            cv2.putText(processed_frame, name_text, (10, 70 + i*30),  # 在图像上添加人脸信息
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 白色文本
                else:                              # 如果未识别到人脸
                    status_text = "Face: Unknown"    # 生成英文状态文本
                    color = (0, 0, 255)            # 设置文本颜色为红色
                
                # 在图像上显示主要状态信息
                try:
                    # 尝试使用中文显示
                    processed_frame = self.put_chinese_text(processed_frame, status_text, (10, 30), 24, color)
                except:
                    # 如果失败，使用英文显示
                    cv2.putText(processed_frame, status_text, (10, 30),     # 在图像上添加状态文本
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)       # 字体、大小、颜色、粗细
                cv2.putText(processed_frame, f"Frame: {frames}", (10, processed_frame.shape[0] - 40),  # 显示帧数（在图像底部）
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 白色文本
                
                # 将处理好的帧和任务代号一起放入显示队列（与main.py一致）
                self.display_queue.put((processed_frame, 0))  # 人脸识别任务代号为0
                
                frames += 1                        # 帧计数器加1
                if frames % 30 == 0:               # 每30帧计算一次帧率
                    fps = 30 / (time.time() - loopTime)  # 计算帧率
                    print(f"30帧平均帧率: {fps:.2f} 帧/秒")  # 打印帧率信息
                    if recognized_names:           # 如果有识别结果
                        print(f"当前识别到的人脸: {recognized_names}")  # 打印识别结果
                    loopTime = time.time()         # 重置计时起点
                    
            except queue.Empty:                    # 如果队列为空（超时）
                continue                           # 继续下一次循环
        
        print("人脸识别任务结束")                   # 打印任务结束信息
        pool.release()                             # 释放模型池资源
    
    def show_frames(self):
        """显示图像帧的函数（在主线程中运行）"""
        while not self.stop_event.is_set():       # 当停止事件未设置时，持续运行
            try:
                frame, task_code = self.display_queue.get(timeout=1)  # 获取帧和任务代号（与main.py一致）
                if frame is None:                  # 如果获取的帧为空
                    continue                       # 跳过此次循环
                    
                cv2.imshow(f"Task {task_code}", frame)  # 使用与main.py完全一致的窗口标题
                
                key = cv2.waitKey(1) & 0xFF        # 检测按键输入（等待1毫秒）
                if key == ord('q'):                # 如果按下'q'键
                    print("用户按下 'q' 键，退出程序")  # 打印退出信息
                    self.stop_event.set()          # 设置停止事件
                    break                          # 跳出循环
                    
            except queue.Empty:                    # 如果队列为空（超时）
                continue                           # 继续下一次循环
        
        cv2.destroyAllWindows()                    # 关闭所有OpenCV窗口
    
    def run_test(self):
        """运行测试的主函数"""
        print("=== 人脸识别测试程序 ===")            # 打印程序标题
        print("按 'q' 键退出程序")                  # 打印操作提示
        print("程序将识别预先训练的人脸数据库中的人员")  # 打印功能说明
        
        # 打开RGB摄像头
        cam = self.open_camera(video_number.rgb_numbers)  # 调用函数打开摄像头
        if cam is None:                            # 如果摄像头打开失败
            print("错误: 无法打开RGB摄像头")          # 打印错误信息
            return                                 # 退出函数
        
        try:
            # 启动摄像头采集线程
            capture_thread = threading.Thread(target=self.camera_capture, args=(cam,))  # 创建采集线程
            capture_thread.start()                # 启动采集线程
            
            # 启动识别任务线程
            recognition_thread = threading.Thread(target=self.face_recognition_task)    # 创建识别线程
            recognition_thread.start()            # 启动识别线程
            
            # 启动显示线程（模仿main.py的方式）
            display_thread = threading.Thread(target=self.show_frames)  # 创建显示线程
            display_thread.start()                # 启动显示线程
            
            # 等待显示线程结束（模仿main.py的主线程行为）
            try:
                display_thread.join()              # 等待显示线程结束
            except KeyboardInterrupt:
                print("接收到中断信号")
            
            # 停止其他线程
            self.stop_event.set()                  # 设置停止事件
            capture_thread.join(timeout=2)        # 等待采集线程结束
            recognition_thread.join(timeout=2)    # 等待识别线程结束
            
        except KeyboardInterrupt:                  # 捕获键盘中断（Ctrl+C）
            print("程序中断")                 # 打印中断信息
        finally:                                   # 无论如何都会执行的清理代码
            self.stop_event.set()                  # 设置停止事件
            cam.release()                          # 释放摄像头资源
            cv2.destroyAllWindows()                # 关闭所有OpenCV窗口
            print("人脸识别测试程序结束")           # 打印程序结束信息

def main():
    """主函数"""
    test = FaceRecognitionTest()                   # 创建测试类实例
    test.run_test()                                # 运行测试

if __name__ == "__main__":                         # 如果作为主程序运行
    main()                                         # 调用主函数
