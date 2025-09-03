# ============================================================================
# RGB摄像头模块 (rgb_camera.py)
# 功能：RGB摄像头的人脸识别检测
# 特点：使用RetinaFace和MobileFaceNet模型进行实时人脸识别，支持RKNN模型池优化
# ============================================================================

import cv2                    # OpenCV库，用于摄像头操作和图像处理
import time                   # 时间模块，用于性能统计和延时控制
import video_number           # 摄像头编号管理模块，用于自动识别RGB摄像头设备
from rknnpool_rgb import rknnPoolExecutor_face  # 导入人脸识别的RKNN模型池
from func_face import myFunc_face               # 导入人脸识别的处理函数
from rec_co import command                      # 导入控制命令模块，用于接收外部控制信号

def face():
    """
    RGB摄像头人脸识别主函数
    功能：启动RGB摄像头，进行实时人脸识别，支持外部控制停止
    """
    print("rgb camera start")  # 打印启动信息
    
    # 自动识别并打开可用的RGB摄像头
    for number in video_number.rgb_numbers:  # 遍历RGB摄像头编号列表
        cap = cv2.VideoCapture(number)       # 尝试打开指定编号的摄像头
        if cap.isOpened():                   # 如果摄像头成功打开
            print(f"Found openable camera: {number}")  # 打印成功打开的摄像头编号
            break                            # 跳出循环，使用找到的摄像头

    # cap = cv2.VideoCapture(0)  # 注释掉的固定摄像头编号代码

    # 设置人脸识别模型路径
    model_path = 'model_data/retinaface_mob.rknn'    # RetinaFace人脸检测模型的RKNN格式文件路径
    model_path2 = 'model_data/mobilefacenet.rknn'    # MobileFaceNet人脸识别模型的RKNN格式文件路径
    
    # 设置线程池执行器数量，增大可提高处理帧率
    TPEs = 3  # ThreadPoolExecutor数量，用于并行处理多个推理任务
    
    # 初始化RKNN模型池，用于高效的人脸识别推理
    pool = rknnPoolExecutor_face(
        rknnModel1=model_path,   # 传入人脸检测模型路径
        rknnModel2=model_path2,  # 传入人脸识别模型路径
        TPEs=TPEs,              # 传入线程池数量
        func=myFunc_face)       # 传入处理函数

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("could not open camera")  # 如果摄像头打开失败，打印错误信息

    # 初始化异步处理所需要的帧，预填充模型池
    for i in range(TPEs + 1):  # 循环TPEs+1次，为每个线程预分配一帧
        ret, frame = cap.read()  # 从摄像头读取一帧图像
        if not ret:  # 如果读取失败
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")  # 抛出详细的错误信息
            cap.release()  # 释放摄像头资源
            del pool      # 删除模型池对象
            exit(-1)      # 退出程序，返回错误码-1
        pool.put(frame)   # 将读取的帧放入模型池进行处理

    # 初始化性能统计变量
    frames, loopTime, initTime = 0, time.time(), time.time()  # 帧计数、循环时间、初始化时间
    
    # 主检测循环，持续进行人脸识别
    while cap.isOpened() and command == "0":  # 当摄像头打开且命令为'0'时继续运行
        ret, frame = cap.read()  # 从摄像头读取一帧图像
        if not ret:  # 如果读取失败
            break    # 退出循环
        
        pool.put(frame)  # 将当前帧放入模型池进行处理
        
        frame, flag = pool.get()  # 从模型池获取处理结果和状态标志

        if not flag:  # 如果处理失败
            break     # 退出循环
        
        cv2.imshow('test', frame)  # 显示处理后的图像，窗口标题为'test'
        frames += 1  # 更新处理的帧数
        
        # 每30帧统计一次性能数据
        if frames % 30 == 0:
            #print(time.time() - loopTime)  # 注释掉的调试信息
            print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")  # 打印30帧的平均帧率
            loopTime = time.time()  # 重置循环时间
    
    # 程序结束时的清理工作
    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    pool.release()  # 释放模型池资源
