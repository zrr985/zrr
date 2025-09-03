# ============================================================================
# 红外摄像头模块 (infrared_camera.py)
# 功能：红外摄像头的人体入侵检测
# 特点：使用YOLOv7模型进行实时人体检测，支持RKNN模型池优化
# ============================================================================

import cv2                    # OpenCV库，用于摄像头操作和图像处理
import time                   # 时间模块，用于性能统计和延时控制
import threading              # 多线程模块，用于线程管理和监控
from rec_co import command    # 导入控制命令模块，用于接收外部控制信号
from rknnpool_inf import rknnPoolExecutor_inf  # 导入红外检测的RKNN模型池
from func_v7 import myFunc_inf                 # 导入红外检测的处理函数

def infrared():
    """
    红外摄像头人体入侵检测主函数
    功能：启动红外摄像头，进行实时人体检测，支持外部控制停止
    """
    print("infrard camera start")  # 打印启动信息
    
    # 打开红外摄像头，使用设备号0
    cap = cv2.VideoCapture(0)  # 创建摄像头对象，0表示第一个摄像头设备
    # cap = cv2.VideoCapture(0)  # 注释掉的重复代码
    
    # 设置YOLOv7模型路径，用于人体检测
    modelPath = "./yolov7_tiny-a.rknn"  # YOLOv7-tiny模型的RKNN格式文件路径
    
    # 设置线程池执行器数量，增大可提高处理帧率
    TPEs = 3  # ThreadPoolExecutor数量，用于并行处理多个推理任务
    
    # 初始化RKNN模型池，用于高效的人体检测推理
    pool = rknnPoolExecutor_inf(
        rknnModel=modelPath,  # 传入模型路径
        TPEs=TPEs,           # 传入线程池数量
        func=myFunc_inf)     # 传入处理函数
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("could not open camera")  # 如果摄像头打开失败，打印错误信息

    # 初始化异步处理所需要的帧，预填充模型池
    for i in range(TPEs + 1):  # 循环TPEs+1次，为每个线程预分配一帧
        ret, frame = cap.read()  # 从摄像头读取一帧图像
        if not ret:  # 如果读取失败
            cap.release()  # 释放摄像头资源
            del pool      # 删除模型池对象
            exit(-1)      # 退出程序，返回错误码-1
        pool.put(frame)   # 将读取的帧放入模型池进行处理

    # 初始化性能统计变量
    frames, loopTime, initTime = 0, time.time(), time.time()  # 帧计数、循环时间、初始化时间
    
    # 主检测循环，持续进行人体检测
    while (cap.isOpened() and command == '1'):  # 当摄像头打开且命令为'1'时继续运行
        frames += 1  # 帧计数器加1
        
        ret, frame = cap.read()  # 从摄像头读取一帧图像
        if not ret:  # 如果读取失败
            break    # 退出循环
        
        pool.put(frame)  # 将当前帧放入模型池进行处理
        
        frame, flag = pool.get()  # 从模型池获取处理结果和状态标志
        if flag == False:  # 如果处理失败
            break          # 退出循环
        
        cv2.imshow('test', frame)  # 显示处理后的图像，窗口标题为'test'
        
        # 检查是否按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # 如果按下q键，退出循环
        
        # 每30帧统计一次性能数据
        if frames % 30 == 0:
            print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")  # 打印30帧的平均帧率
            print(threading.active_count())  # 打印当前活跃线程数量
            print(threading.enumerate())     # 打印所有线程的详细信息
            loopTime = time.time()  # 重置循环时间

        # 再次检查退出条件（重复的检查）
        if cv2.waitKey(1) & 0xFF == ord('q'):
           # 释放cap和rknn线程池
            cap.release()  # 释放摄像头资源
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            pool.release()  # 释放模型池资源

    # 程序结束时的清理工作
    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    pool.release()  # 释放模型池资源
