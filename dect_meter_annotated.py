# -*- coding: utf-8 -*-
"""
仪表检测工具 (dect_meter.py)
功能：使用AprilTag标签检测仪表区域，实现仪表的自动定位和分类
作者：视觉检测系统
日期：2024
"""

import cv2  # OpenCV库，用于图像处理和摄像头操作
import apriltag  # AprilTag检测库，用于识别二维码标签

# 预定义仪表类别信息的映射表，key为AprilTag的ID，value为对应的仪表类别
TAG_TO_CATEGORY = {
    0: "消防水压表",    # AprilTag ID 0 对应消防水压表
    1: "空气房气压表",  # AprilTag ID 1 对应空气房气压表
    2: "机械气压表"     # AprilTag ID 2 对应机械气压表
}


def capture_meter_frame(cap):
    """
    捕获仪表帧的函数
    功能：通过AprilTag标签检测仪表区域，并返回仪表ROI和类别信息
    
    参数：
        cap: 摄像头对象
        
    返回：
        meter_roi: 仪表区域图像，如果未检测到则返回None
        detected_categories: 检测到的仪表类别列表，如果未检测到则返回None
    """
    # 创建AprilTag检测器，使用tag36h11家族（36h11是AprilTag的一种编码格式）
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return None, None

    # 持续从摄像头读取图像帧
    while cap.isOpened():
        ret, frame = cap.read()  # 读取一帧图像
        if not ret:  # 如果读取失败
            print("无法读取视频帧")
            break

        # 将彩色图像转换为灰度图像（AprilTag检测需要灰度图）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 对灰度图进行直方图均衡化，提高对比度，有助于标签检测
        gray = cv2.equalizeHist(gray)

        # 在灰度图像上检测AprilTag标签
        tags = detector.detect(gray)

        # 遍历所有检测到的标签，并在图像中标注
        detected_categories = []  # 存储检测到的仪表类别
        for tag in tags:
            # 获取标签的四个角点坐标
            corners = tag.corners
            # 获取标签的中心点坐标
            center = (int(tag.center[0]), int(tag.center[1]))

            # 绘制标签的外部矩形框（绿色，线宽2）
            for i in range(4):
                # 当前角点坐标
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                # 下一个角点坐标（循环到第一个）
                pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                # 绘制连接线
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # 在标签中心绘制一个小圆点（红色，实心）
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # 获取标签的ID号
            tag_id = tag.tag_id
            # 根据ID获取对应的仪表类别，如果ID不在映射表中则显示"未知类别"
            category = TAG_TO_CATEGORY.get(tag_id, "未知类别")
            # 将检测到的类别添加到列表中
            detected_categories.append(category)

            # 在图像上标注标签ID和类别信息（蓝色文字）
            cv2.putText(frame, f"ID: {tag_id} ({category})", (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 至少需要两个AprilTag才能确定仪表区域（对角标签）
        if len(tags) >= 2:
            # 假设只处理两个标签，分别是对角的两个标签
            tag1, tag2 = tags[0], tags[1]

            # 取出两个标签的中心位置
            center1 = (int(tag1.center[0]), int(tag1.center[1]))
            center2 = (int(tag2.center[0]), int(tag2.center[1]))

            # 计算对角的两个点，确定仪表ROI的边界
            # top_left: 左上角点（x和y坐标的最小值）
            top_left = (min(center1[0], center2[0]), min(center1[1], center2[1]))
            # bottom_right: 右下角点（x和y坐标的最大值）
            bottom_right = (max(center1[0], center2[0]), max(center1[1], center2[1]))

            # 从原图中裁剪出仪表区域（ROI: Region of Interest）
            meter_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # 检查裁剪出的区域是否有效（不为空）
            if meter_roi.size > 0:
                # 关闭所有窗口
                cv2.destroyAllWindows()
                # 返回检测到的仪表区域和其对应的类别信息
                return meter_roi, detected_categories

    # 如果没有检测到足够的标签，返回None
    return None, None


# 示例使用代码
if __name__ == "__main__":
    # 打开默认摄像头（设备号0）
    cap = cv2.VideoCapture(0)
    # 调用仪表检测函数
    meter_roi, categories = capture_meter_frame(cap)

    # 如果成功检测到仪表区域
    if meter_roi is not None:
        print(f"检测到的仪表类别: {categories}")  # 打印检测到的仪表类别
        cv2.imshow("Meter ROI", meter_roi)  # 显示仪表ROI图像
        cv2.imwrite('w14.jpg', meter_roi)  # 保存仪表ROI图像到文件
        k = cv2.waitKey(0)  # 等待按键
        if k == 27:  # 如果按下ESC键
            cap.release()  # 释放摄像头
            cv2.destroyAllWindows()  # 关闭所有窗口
    else:
        print("未检测到仪表区域")  # 打印未检测到仪表的提示
