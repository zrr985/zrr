# -*- coding: utf-8 -*-
"""
RKNN模型池 - 仪表检测专用 (rknnpool_meter.py)
功能：管理YOLOv8-Seg模型池，实现仪表指针和刻度检测的并发处理
作者：视觉检测系统
日期：2024
"""

from queue import Queue  # 队列模块，用于线程间通信
from rknnlite.api import RKNNLite  # RKNN推理引擎
from concurrent.futures import ThreadPoolExecutor, as_completed  # 线程池执行器


def initRKNN(rknnModel="./rknnModel/yolov8_seg.rknn", id=0):
    """
    初始化单个RKNN模型实例
    
    参数：
        rknnModel: RKNN模型文件路径（默认YOLOv8-Seg分割模型）
        id: NPU核心ID (0,1,2分别对应不同的NPU核心，-1表示使用所有核心)
    返回：
        rknn_lite: 初始化完成的RKNNLite对象
    """
    rknn_lite = RKNNLite()  # 创建RKNNLite实例
    
    # 加载RKNN模型文件
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:  # 如果加载失败
        print("Load RKNN rknnModel failed")
        exit(ret)  # 退出程序
    
    # 根据ID选择不同的NPU核心进行初始化
    if id == 0:
        # 使用NPU核心0
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        # 使用NPU核心1
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        # 使用NPU核心2
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        # 使用所有NPU核心 (0,1,2)
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        # 使用默认核心配置
        ret = rknn_lite.init_runtime()
    
    if ret != 0:  # 如果初始化失败
        print("Init runtime environment failed")
        exit(ret)  # 退出程序
    
    print(rknnModel, "\t\tdone")  # 打印初始化成功信息
    return rknn_lite  # 返回初始化完成的RKNNLite对象


def initRKNNs(rknnModel="./rknnModel/yolov8_seg.rknn", TPEs=1):
    """
    初始化多个RKNN模型实例，创建模型池
    
    参数：
        rknnModel: RKNN模型文件路径（默认YOLOv8-Seg分割模型）
        TPEs: 线程池执行器数量，决定创建多少个模型实例
    返回：
        rknn_list: 包含多个RKNNLite对象的列表
    """
    rknn_list = []  # 存储RKNNLite对象的列表
    for i in range(TPEs):  # 循环创建TPEs个模型实例
        # 使用模运算确保ID在0-2之间循环，实现NPU核心的负载均衡
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list  # 返回模型池列表


class rknnPoolExecutor():
    """
    仪表检测专用的RKNN模型池执行器
    管理YOLOv8-Seg模型池，实现仪表指针和刻度检测的并发处理
    """
    
    def __init__(self, rknnModel, TPEs, func):
        """
        初始化仪表检测模型池
        
        参数：
            rknnModel: YOLOv8-Seg模型文件路径（仪表分割检测）
            TPEs: 线程池执行器数量
            func: 处理函数（仪表检测函数）
        """
        self.TPEs = TPEs  # 线程池执行器数量
        self.queue = Queue()  # 任务队列，用于存储异步任务
        
        # 初始化YOLOv8-Seg模型池
        self.rknnPool = initRKNNs(rknnModel, TPEs)  # YOLOv8-Seg模型池
        
        # 创建线程池执行器
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func  # 处理函数
        self.num = 0  # 任务计数器，用于轮询选择模型实例

    def put(self, frame):
        """
        提交任务到线程池
        
        参数：
            frame: 输入图像帧
        """
        # 将任务提交到线程池，使用轮询方式选择模型实例
        # self.num % self.TPEs 确保模型实例的负载均衡
        # 注意：仪表检测的put方法不传递num参数，与其他模块不同
        self.queue.put(self.pool.submit(
            self.func,  # 处理函数
            self.rknnPool[self.num % self.TPEs],  # YOLOv8-Seg模型实例
            frame  # 输入图像
        ))
        self.num += 1  # 任务计数器递增

    def get(self):
        """
        获取处理结果
        
        返回：
            result: 处理结果
            success: 是否成功获取结果
        """
        if self.queue.empty():  # 如果队列为空
            return None, False  # 返回空结果和失败标志
        
        fut = self.queue.get()  # 从队列中获取Future对象
        return fut.result(), True  # 返回结果和成功标志

    def release(self):
        """
        释放资源，清理模型池和线程池
        """
        self.pool.shutdown()  # 关闭线程池
        
        # 释放所有模型实例的资源
        for rknn_lite in self.rknnPool:
            rknn_lite.release()  # 释放YOLOv8-Seg模型
