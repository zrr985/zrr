# -*- coding: utf-8 -*-
"""
人脸图片采集工具 (add_face.py)
功能：通过摄像头采集人脸图片，用于构建人脸识别数据库
作者：视觉检测系统
日期：2024
"""

import cv2  # OpenCV库，用于摄像头操作和图像处理
import os   # 操作系统接口，用于文件和目录操作
import video_number  # 自定义模块，用于自动检测摄像头设备号

def open_camera(camera_numbers):
    """
    打开摄像头的函数
    参数：
        camera_numbers: 摄像头设备号列表
    返回：
        成功时返回摄像头对象，失败时返回None
    """
    for number in camera_numbers:  # 遍历所有可能的摄像头设备号
        cap = cv2.VideoCapture(number)  # 尝试打开指定设备号的摄像头
        if cap.isOpened():  # 检查摄像头是否成功打开
            print(f"Found openable camera: {number}")  # 打印找到的可用摄像头
            return cap  # 返回摄像头对象
    return None  # 如果没有找到可用摄像头，返回None

# 使用video_number模块自动检测RGB摄像头设备号
cap = open_camera(video_number.rgb_numbers)

# 定义存储人脸图片的文件夹名称
folder = "face_images"
# 如果文件夹不存在，则创建该文件夹
if not os.path.exists(folder):
    os.makedirs(folder)

# 定义存储人名与图片编号对应关系的文件名
info_file = "name_to_image_map.txt"

# 读取或创建记录文件
if not os.path.exists(info_file):  # 如果文件不存在
    with open(info_file, 'w') as f:  # 以写入模式创建文件
        f.write("Image Number,Name\n")  # 写入CSV格式的文件头部

# 获取用户输入的姓名，用于标识采集的人脸图片
name = input("请输入姓名：")

# 图片编号，从1开始递增
start_image_number = 1

# 检查文件夹内是否有已存在的图片，确定从哪个编号开始
existing_images = [f for f in os.listdir(folder) if f.endswith(".jpg")]  # 获取所有jpg文件
if existing_images:  # 如果存在图片文件
    # 提取现有图片的编号（去掉.jpg后缀）
    existing_numbers = [int(f.split(".")[0]) for f in existing_images]
    # 从最大编号+1开始，避免重复
    start_image_number = max(existing_numbers) + 1

print("按 's' 键拍照，按 'q' 键退出程序。")

# 主循环：持续从摄像头读取图像
while True:
    ret, frame = cap.read()  # 从摄像头读取一帧图像
    if not ret:  # 如果读取失败
        print("无法读取摄像头")
        break

    # 显示摄像头实时画面
    cv2.imshow("Camera", frame)

    # 等待用户按键操作，& 0xFF用于兼容不同系统
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 如果按下's'键（拍照）
        # 生成图片文件名（编号.jpg格式）
        image_filename = f"{start_image_number}.jpg"
        # 构建完整的图片保存路径
        image_path = os.path.join(folder, image_filename)

        # 保存当前帧图像到指定路径
        cv2.imwrite(image_path, frame)
        print(f"保存图片: {image_path}")

        # 将图片编号和姓名对应关系写入记录文件
        with open(info_file, 'a') as f:  # 以追加模式打开文件
            f.write(f"{start_image_number},{name}\n")  # 写入CSV格式的记录

        start_image_number += 1  # 图片编号递增，为下次拍照做准备

    elif key == ord('q'):  # 如果按下'q'键（退出程序）
        break

# 程序结束时的清理工作
cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
