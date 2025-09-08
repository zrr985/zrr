#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外入侵检测测试程序
独立测试红外摄像头的入侵检测功能
"""

# 导入所需的库
import cv2                                          # OpenCV库，用于图像处理和摄像头操作
import time                                         # 时间库，用于计算帧率和时间间隔
import threading                                    # 线程库，用于多线程处理
import queue                                        # 队列库，用于线程间数据传递
import video_number                                 # 摄像头编号配置模块
from rknnpool_inf import rknnPoolExecutor_inf       # RKNN模型池执行器（红外检测专用）
from func_v7 import myFunc_inf                      # 红外检测功能函数

class InfraredDetectionTest:
    """红外入侵检测测试类"""
    
    def __init__(self):
        """初始化测试类的各种参数和队列"""
        self.class_inf = 0                          # 红外检测结果，0=正常，1=检测到入侵
        self.stop_event = threading.Event()         # 线程停止事件，用于控制所有线程的退出
        self.frame_queue = queue.Queue(maxsize=5)   # 帧队列，存储从摄像头读取的图像帧（最大5帧）
        self.display_queue = queue.Queue(maxsize=10) # 显示队列，存储处理后的图像帧（最大10帧）
        
    def open_camera(self, camera_numbers=None, fixed_camera_id=None):
        """尝试打开红外摄像头"""
        if fixed_camera_id is not None:           # 如果指定了固定的摄像头编号
            print(f"尝试打开固定红外摄像头编号: {fixed_camera_id}")
            cap = cv2.VideoCapture(fixed_camera_id)  # 直接打开指定编号的摄像头
            if cap.isOpened():                     # 检查摄像头是否成功打开
                print(f"✅ 成功打开固定红外摄像头: {fixed_camera_id}")  # 打印成功信息
                return cap                         # 返回摄像头对象
            else:
                print(f"❌ 无法打开固定红外摄像头: {fixed_camera_id}")  # 打印失败信息
                return None                        # 返回None
        
        # 如果没有指定固定编号，则遍历配置文件中的摄像头编号列表
        if camera_numbers:
            for number in camera_numbers:          # 遍历配置文件中的摄像头编号列表
                cap = cv2.VideoCapture(number)     # 尝试打开指定编号的摄像头
                if cap.isOpened():                 # 检查摄像头是否成功打开
                    print(f"✅ 成功打开红外摄像头: {number}")  # 打印成功信息
                    return cap                     # 返回摄像头对象
                else:
                    print(f"⚠️ 摄像头 {number} 无法打开，尝试下一个")  # 打印尝试信息
        
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
            
    def infrared_detection_task(self):
        """红外入侵检测任务线程函数"""
        print("红外入侵检测任务启动")               # 打印任务启动信息
        model_path = "./yolov7_tiny-a.rknn"       # RKNN模型文件路径
        TPEs = 3                                   # 线程池执行器数量（Thread Pool Executors）
        
        # 初始化RKNN模型池
        pool = rknnPoolExecutor_inf(
            rknnModel=model_path,                  # 模型文件路径
            TPEs=TPEs,                             # 线程池大小
            func=myFunc_inf                        # 检测函数
        )
        
        frames = 0                                 # 帧计数器
        loopTime = time.time()                     # 循环开始时间，用于计算帧率
        abnormal_count = 0                         # 连续异常帧计数器
        
        while not self.stop_event.is_set():       # 当停止事件未设置时，持续运行
            try:
                frame = self.frame_queue.get(timeout=1)  # 从帧队列获取图像（超时1秒）
                if frame is None:                  # 如果获取的帧为空
                    continue                       # 跳过此次循环
                
                pool.put(frame)                    # 将帧放入模型池进行处理
                result, flag = pool.get()          # 获取处理结果
                processed_frame, self.class_inf = result  # 解包结果：处理后的图像和检测结果
                print(f'now inf data {self.class_inf}')   # 打印当前检测结果（调试信息）
                if not flag:                       # 如果处理失败
                    break                          # 跳出循环
                
                # 检测结果分析和状态文本生成
                if self.class_inf == 1:            # 如果检测到入侵
                    abnormal_count += 1            # 异常帧计数器加1
                    status_text = f"Intrusion Detected! (Count: {abnormal_count})"  # 生成英文状态文本
                    color = (0, 0, 255)            # 设置文本颜色为红色（BGR格式）
                    
                    if abnormal_count >= 10:       # 如果连续异常帧数达到阈值
                        print(f"警告: 连续检测到入侵{abnormal_count}帧!")  # 打印警告信息
                else:                              # 未检测到入侵或不确定
                    abnormal_count = 0             # 重置异常帧计数器
                    status_text = "Infrared: Normal"   # 生成英文正常状态文本
                    color = (0, 255, 0)            # 设置文本颜色为绿色
                
                # 在图像上显示检测状态信息
                cv2.putText(processed_frame, status_text, (10, 30),     # 在图像上添加状态文本
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)       # 字体、大小、颜色、粗细
                cv2.putText(processed_frame, f"Frame: {frames}", (10, 70),  # 显示帧数
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 白色文本
                
                # 将处理好的帧和任务代号一起放入显示队列（与main.py一致）
                self.display_queue.put((processed_frame, 1))  # 红外检测任务代号为1
                
                frames += 1                        # 帧计数器加1
                if frames % 30 == 0:               # 每30帧计算一次帧率
                    fps = 30 / (time.time() - loopTime)  # 计算帧率
                    print(f"30帧平均帧率: {fps:.2f} 帧/秒")  # 打印帧率信息
                    loopTime = time.time()         # 重置计时起点
                    
            except queue.Empty:                    # 如果队列为空（超时）
                continue                           # 继续下一次循环
        
        print("红外入侵检测任务结束")               # 打印任务结束信息
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
    
    def run_test(self, fixed_camera_id=None):
        """运行测试的主函数"""
        print("=== 红外入侵检测测试程序 ===")        # 打印程序标题
        print("按 'q' 键退出程序")                  # 打印操作提示
        
        # 打开红外摄像头
        if fixed_camera_id is not None:           # 如果指定了固定摄像头编号
            print(f"📷 使用固定红外摄像头编号: {fixed_camera_id}")
            cam = self.open_camera(fixed_camera_id=fixed_camera_id)  # 使用固定编号打开摄像头
        else:                                      # 否则使用配置文件中的编号列表
            print(f"📷 从配置列表中查找红外摄像头: {video_number.inf_numbers}")
            cam = self.open_camera(camera_numbers=video_number.inf_numbers)  # 调用函数打开摄像头
        
        if cam is None:                            # 如果摄像头打开失败
            print("❌ 错误: 无法打开红外摄像头")       # 打印错误信息
            return                                 # 退出函数
        
        try:
            # 启动摄像头采集线程
            capture_thread = threading.Thread(target=self.camera_capture, args=(cam,))  # 创建采集线程
            capture_thread.start()                # 启动采集线程
            
            # 启动检测任务线程
            detection_thread = threading.Thread(target=self.infrared_detection_task)    # 创建检测线程
            detection_thread.start()              # 启动检测线程
            
            # 启动显示线程（在主线程中运行）
            self.show_frames()                     # 在主线程中运行显示函数
            
            # 等待线程结束
            capture_thread.join()                  # 等待采集线程结束
            detection_thread.join()                # 等待检测线程结束
            
        except KeyboardInterrupt:                  # 捕获键盘中断（Ctrl+C）
            print("程序被用户中断")                 # 打印中断信息
        finally:                                   # 无论如何都会执行的清理代码
            self.stop_event.set()                  # 设置停止事件
            cam.release()                          # 释放摄像头资源
            cv2.destroyAllWindows()                # 关闭所有OpenCV窗口
            print("红外检测测试程序结束")           # 打印程序结束信息

def main():
    """主函数"""
    import argparse                                # 导入命令行参数解析库
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='红外入侵检测测试程序')
    parser.add_argument('--camera', '-c', type=int, default=None,
                       help='指定固定的红外摄像头编号（例如：--camera 0）')
    parser.add_argument('--list-cameras', '-l', action='store_true',
                       help='列出配置文件中的摄像头编号')
    
    args = parser.parse_args()                     # 解析命令行参数
    
    # 如果用户要求列出摄像头编号
    if args.list_cameras:
        print("📷 配置文件中的红外摄像头编号:")
        print(f"   {video_number.inf_numbers}")
        return
    
    test = InfraredDetectionTest()                 # 创建测试类实例
    
    # 根据参数运行测试
    if args.camera is not None:
        print(f"🎯 使用指定的摄像头编号: {args.camera}")
        test.run_test(fixed_camera_id=args.camera)  # 使用固定摄像头编号运行
    else:
        print("🔍 自动搜索可用的红外摄像头")
        test.run_test()                            # 使用默认方式运行测试

if __name__ == "__main__":                         # 如果作为主程序运行
    main()                                         # 调用主函数
