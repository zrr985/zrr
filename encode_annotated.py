# -*- coding: utf-8 -*-
"""
人脸特征编码工具 (encode.py)
功能：使用RetinaFace和MobileFaceNet模型对人脸图片进行特征提取和编码
作者：视觉检测系统
日期：2024
"""

import cv2  # OpenCV库，用于图像处理
import numpy as np  # 数值计算库
from rknnlite.api import RKNNLite  # RKNN推理引擎
import platform  # 平台信息检测
import os  # 操作系统接口
import time  # 时间相关功能
from itertools import product as product  # 笛卡尔积计算
from math import ceil  # 向上取整函数
import math  # 数学函数库
from tqdm import tqdm  # 进度条显示
from PIL import Image  # 图像处理库

# 获取face_images文件夹中的所有文件
list_dir = os.listdir("face_images")
image_paths = []  # 存储图片路径的列表
names = []  # 存储人名的列表

# 定义模型文件路径
model_path = 'model_data/retinaface_mob.rknn'  # RetinaFace人脸检测模型
model_path2 = 'model_data/mobilefacenet.rknn'  # MobileFaceNet特征提取模型

# RetinaFace模型的配置参数
cfg = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],  # 不同特征层的锚框尺寸
    'steps': [8, 16, 32],  # 特征图的步长
    'variance': [0.1, 0.2],  # 边界框回归的方差参数
}

def letterbox_image(image, size):
    """
    图像缩放和填充函数
    功能：将图像缩放到指定尺寸，保持宽高比，不足部分用灰色填充
    
    参数：
        image: 输入图像
        size: 目标尺寸 (width, height)
    返回：
        处理后的图像
    """
    ih, iw, _ = np.shape(image)  # 获取图像的高度、宽度和通道数
    w, h = size  # 目标宽度和高度
    scale = min(w/iw, h/ih)  # 计算缩放比例，取宽高比的最小值以保持比例
    nw = int(iw*scale)  # 缩放后的宽度
    nh = int(ih*scale)  # 缩放后的高度

    image = cv2.resize(image, (nw, nh))  # 缩放图像
    # 创建目标尺寸的灰色背景图像
    new_image = np.ones([size[1], size[0], 3]) * 128
    # 将缩放后的图像居中放置
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image

class Anchors(object):
    """
    锚框生成类
    功能：为RetinaFace模型生成不同尺度和位置的锚框
    """
    def __init__(self, cfg, image_size=None):
        """
        初始化锚框生成器
        
        参数：
            cfg: 配置参数
            image_size: 图像尺寸
        """
        super(Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes']  # 最小尺寸列表
        self.steps = cfg['steps']  # 步长列表
        self.image_size = image_size  # 图像尺寸

        # 计算三个有效特征层的高和宽
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        """
        生成锚框
        
        返回：
            锚框坐标数组
        """
        anchors = []
        for k, f in enumerate(self.feature_maps):  # 遍历每个特征层
            min_sizes = self.min_sizes[k]  # 当前特征层的最小尺寸
            # 对特征层的高和宽进行循环迭代
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:  # 遍历最小尺寸
                    s_kx = min_size / self.image_size[1]  # 宽度归一化
                    s_ky = min_size / self.image_size[0]  # 高度归一化
                    # 计算密集的x坐标
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    # 计算密集的y坐标
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # 生成所有锚框坐标
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # 将锚框列表转换为numpy数组并重塑
        output = np.array(anchors).reshape(-1, 4)
        return output

def decode(loc, priors, variances):
    """
    边界框解码函数
    功能：将模型输出的相对坐标转换为绝对坐标
    
    参数：
        loc: 模型输出的位置信息
        priors: 先验锚框
        variances: 方差参数
    返回：
        解码后的边界框坐标
    """
    # 解码边界框坐标
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    # 转换为左上角和右下角坐标
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """
    关键点解码函数
    功能：将模型输出的相对关键点坐标转换为绝对坐标
    
    参数：
        pre: 模型输出的关键点信息
        priors: 先验锚框
        variances: 方差参数
    返回：
        解码后的关键点坐标
    """
    # 解码5个关键点的坐标
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), 1)
    return landms

def iou(b1, b2):
    """
    计算IoU（交并比）函数
    功能：计算两个边界框之间的IoU值
    
    参数：
        b1: 第一个边界框
        b2: 第二个边界框数组
    返回：
        IoU值数组
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]  # 第一个框的坐标
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]  # 第二个框的坐标

    # 计算交集区域的坐标
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    # 计算交集面积
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    # 计算两个框的面积
    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # 计算IoU
    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou

def non_max_suppression(detection, conf_thres=0.5, nms_thres=0.3):
    """
    非极大值抑制函数
    功能：去除重叠的检测框，保留置信度最高的框
    
    参数：
        detection: 检测结果
        conf_thres: 置信度阈值
        nms_thres: NMS阈值
    返回：
        抑制后的检测框
    """
    # 找出该图片中得分大于门限函数的框
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]

    if len(detection) <= 0:
        return []

    best_box = []
    scores = detection[:, 4]
    # 根据得分对框进行从大到小排序
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    while np.shape(detection)[0] > 0:
        # 每次取出得分最大的框
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        # 计算其与其它所有预测框的重合程度
        ious = iou(best_box[-1], detection[1:])
        # 重合程度过大的则剔除
        detection = detection[1:][ious < nms_thres]
    return np.array(best_box)

def retinaface_correct_boxes(result, input_shape, image_shape):
    """
    RetinaFace边界框校正函数
    功能：将模型输出坐标校正到原图尺寸
    
    参数：
        result: 检测结果
        input_shape: 模型输入尺寸
        image_shape: 原图尺寸
    返回：
        校正后的结果
    """
    # 计算新的形状
    new_shape = image_shape * np.min(input_shape / image_shape)
    # 计算偏移量
    offset = (input_shape - new_shape) / 2. / input_shape
    # 计算缩放比例
    scale = input_shape / new_shape

    # 边界框的缩放和偏移参数
    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0]]

    # 校正边界框坐标
    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    # 校正关键点坐标
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result

def Alignment_1(img, landmark):
    """
    人脸对齐函数
    功能：根据眼睛关键点对人脸进行旋转对齐
    
    参数：
        img: 输入图像
        landmark: 关键点坐标
    返回：
        对齐后的图像和关键点
    """
    if landmark.shape[0] == 68:  # 68个关键点的情况
        x = landmark[36, 0] - landmark[45, 0]  # 左眼到右眼的x距离
        y = landmark[36, 1] - landmark[45, 1]  # 左眼到右眼的y距离
    elif landmark.shape[0] == 5:  # 5个关键点的情况
        x = landmark[0, 0] - landmark[1, 0]  # 左眼到右眼的x距离
        y = landmark[0, 1] - landmark[1, 1]  # 左眼到右眼的y距离
    
    # 眼睛连线相对于水平线的倾斜角
    if x == 0:
        angle = 0
    else:
        # 计算它的弧度制
        angle = math.atan(y / x) * 180 / math.pi

    # 图像中心点
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # 获取旋转矩阵
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射变换
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    # 转换关键点坐标
    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark

def face_distance(face_encodings, face_to_compare):
    """
    计算人脸距离函数
    功能：计算待比较人脸与已知人脸编码的欧氏距离
    
    参数：
        face_encodings: 已知人脸编码数组
        face_to_compare: 待比较的人脸编码
    返回：
        距离数组
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    # 计算欧氏距离
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    """
    比较人脸函数
    功能：比较待检查人脸与已知人脸是否匹配
    
    参数：
        known_face_encodings: 已知人脸编码数组
        face_encoding_to_check: 待检查的人脸编码
        tolerance: 容差阈值
    返回：
        匹配结果列表和距离数组
    """
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance), dis

# 处理face_images文件夹中的图片
for name in list_dir:
    # 构建完整的图片路径
    image_paths.append("face_images/"+name)
    # 分离文件名和扩展名
    filename, extension = os.path.splitext(name)
    # 提取人名（假设文件名格式为"人名_编号.jpg"）
    names.append(filename.split("_")[0])

# 创建RKNN对象
rknn = RKNNLite()  # RetinaFace模型
rknn2 = RKNNLite()  # MobileFaceNet模型

# 加载RKNN模型
print('--> Loading model')
ret = rknn.load_rknn(model_path)  # 加载RetinaFace模型
ret2 = rknn2.load_rknn(model_path2)  # 加载MobileFaceNet模型
if (ret != 0):
    exit(ret)
print('done')

# 根据平台初始化运行时环境
if platform.machine() == 'aarch64':  # ARM64架构
    target = None
else:
    target = 'rk3588'  # RK3588目标平台
ret = rknn.init_runtime(target=target)  # 初始化RetinaFace运行时
ret2 = rknn2.init_runtime(target=target)  # 初始化MobileFaceNet运行时
if ret != 0:
    exit(ret)

# 存储所有人脸编码的列表
face_encodings = []

# 遍历所有人脸图片进行特征提取
for index, path in enumerate(tqdm(image_paths)):
    # 打开人脸图片
    image = np.array(Image.open(path), np.float32)
    # 对输入图像进行一个备份
    old_image = image.copy()
    # 计算输入图片的高和宽
    im_height, im_width, _ = np.shape(image)
    
    # 计算scale，用于将获得的预测框转换成原图的高宽
    scale = [
        np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
    ]
    scale_for_landmarks = [
        np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
        np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
        np.shape(image)[1], np.shape(image)[0]
    ]

    # 图像预处理：缩放和填充
    image = letterbox_image(image, [640, 640])
    image = image.astype(dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    
    # 生成锚框
    anchors = Anchors(cfg, image_size=(640, 640)).get_anchors()
    
    # 将图像输入检测模型
    print('--> Running model')
    start = time.time()
    outputs = rknn.inference(inputs=[image])  # RetinaFace推理
    end = time.time()
    print('时间:{}'.format(end - start))
    
    # 输出数据转为numpy数据格式
    loc = np.array(outputs[0]).squeeze()  # 位置信息
    conf = np.array(outputs[1]).squeeze()  # 置信度信息
    landms = np.array(outputs[2]).squeeze()  # 关键点信息
    
    # 解码边界框
    boxes = decode(loc, anchors, cfg['variance'])
    conf = conf[:, 1:2]  # 取正类的置信度
    # 解码关键点
    landms = decode_landm(landms, anchors, cfg['variance'])
    
    # 对人脸框进行堆叠
    boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)
    # 非极大值抑制
    boxes_conf_landms = non_max_suppression(boxes_conf_landms, 0.5)
    
    if len(boxes_conf_landms) <= 0:
        # 如果没有检测到人脸，使用原图
        image = old_image
    else:
        # 校正边界框到原图尺寸
        boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([640, 640]),
                                                     np.array([im_height, im_width]))
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        # 选择最大的人脸区域
        best_face_location = None
        biggest_area = 0
        for result in boxes_conf_landms:
            left, top, right, bottom = result[0:4]

            w = right - left
            h = bottom - top
            if w * h > biggest_area:
                biggest_area = w * h
                best_face_location = result

        # 裁剪人脸区域
        crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                   int(best_face_location[0]):int(best_face_location[2])]

        # 提取关键点并调整坐标
        landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
            [int(best_face_location[0]), int(best_face_location[1])])
        
        # 保存裁剪后的人脸图像（未对齐）
        cv2.imwrite("model_data/f_b.jpg", crop_img)
        print(crop_img)
        
        # 对人脸进行对齐
        crop_img, _ = Alignment_1(crop_img, landmark)
        print(crop_img)

        # 保存对齐后的人脸图像
        cv2.imwrite("model_data/f_a.jpg", crop_img)
        
        # 图像预处理：转换为uint8格式
        crop_img = crop_img.astype(np.uint8)
        crop_img = Image.fromarray(crop_img)

        # 调整图像尺寸为160x160
        crop_img = crop_img.resize((160, 160), Image.BICUBIC)
        
        # 转换为numpy数组
        crop_img = np.asarray(crop_img, np.float32)
        crop_img = np.expand_dims(crop_img, 0)

        # 使用MobileFaceNet提取特征
        face_encoding = rknn2.inference(data_format='nhwc', inputs=[crop_img])[0]

        # 处理特征向量
        face_encoding = np.array(face_encoding)
        face_encoding = face_encoding.flatten()  # 展平为一维数组
        face_encoding.tolist()
        face_encodings.append(face_encoding)

    # 保存人脸编码和对应的姓名
    np.save("model_data/mobilenet_face_encoding.npy", face_encodings)
    np.save("model_data/mobilenet_names.npy", names)
