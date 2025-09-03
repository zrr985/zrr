# ============================================================================
# 人脸识别功能模块 (func_face.py)
# 功能：基于RetinaFace和MobileFaceNet的人脸检测与识别
# 特点：支持人脸检测、关键点定位、人脸对齐和身份识别
# ============================================================================

import cv2                    # OpenCV库，用于图像处理和显示
import numpy as np            # 数值计算库
from rknnlite.api import RKNNLite  # RKNN推理引擎
import platform               # 平台信息
import os                     # 操作系统接口
import time                   # 时间模块
from itertools import product as product  # 笛卡尔积计算
from math import ceil         # 向上取整
import math                   # 数学函数
from PIL import Image         # 图像处理库
import struct                 # 二进制数据打包
import socket                 # 网络通信

# 创建UDP客户端socket，用于发送识别结果
client = socket.socket(family = socket.AF_INET,type = socket.SOCK_DGRAM)

server_ip = ('127.0.0.1',8848)  # 服务器IP地址和端口
# 无需和服务端建立连接
struct_format = '!i48si'      # 数据包格式：类型(4字节) + 姓名(48字节) + 长度(4字节)
name_ten = []                 # 存储最近10帧的人脸识别结果

# 获取当前工作目录
path=os.getcwd()
# 模型文件路径
model_path = 'model_data/retinaface_mob.rknn'    # RetinaFace人脸检测模型
model_path2 = 'model_data/mobilefacenet.rknn'    # MobileFaceNet人脸识别模型
video_path = 0                # 视频路径
video_save_path = ""          # 视频保存路径
video_fps       = 25.0        # 视频帧率
IMG_SIZE = 640               # 图像尺寸

# RetinaFace网络配置参数
cfg_mnet={
    'min_sizes': [[16, 32], [64, 128], [256, 512]],  # 不同特征层的锚框尺寸
    'steps': [8, 16, 32],                            # 特征图步长
    'variance': [0.1, 0.2],                          # 边界框回归方差
}

# 加载预训练的人脸特征编码和对应姓名
known_face_encodings = np.load("model_data/mobilenet_face_encoding.npy")  # 已知人脸特征
known_face_names = np.load("model_data/mobilenet_names.npy")              # 对应姓名


def letterbox_image(image, size):
    """
    图像缩放和填充函数
    功能：将图像缩放到指定尺寸，保持宽高比，不足部分用灰色填充
    """
    ih, iw, _   = np.shape(image)  # 获取图像高度、宽度、通道数
    w, h        = size             # 目标尺寸
    scale       = min(w/iw, h/ih)  # 计算缩放比例，取较小值保持宽高比
    nw          = int(iw*scale)    # 缩放后的宽度
    nh          = int(ih*scale)    # 缩放后的高度

    image       = cv2.resize(image, (nw, nh))  # 缩放图像
    new_image   = np.ones([size[1], size[0], 3]) * 128  # 创建灰色背景
    # 将缩放后的图像居中放置
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image

class Anchors(object):
    """
    锚框生成类
    功能：根据网络配置生成多尺度锚框
    """
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes  = cfg['min_sizes']  # 最小尺寸列表
        self.steps      = cfg['steps']      # 步长列表
        #---------------------------#
        #   图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        """
        生成锚框
        返回：锚框坐标数组
        """
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]  # 宽度比例
                    s_ky = min_size / self.image_size[0]  # 高度比例
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]  # 中心点x坐标
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]  # 中心点y坐标
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]  # 添加锚框信息

        output = np.array(anchors).reshape(-1, 4)  # 重塑为Nx4数组
        return output

def decode(loc, priors, variances):
    """
    边界框解码函数
    功能：将网络输出的相对坐标转换为绝对坐标
    """
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2  # 转换为左上角坐标
    boxes[:, 2:] += boxes[:, :2]      # 转换为右下角坐标
    return boxes

def decode_landm(pre, priors, variances):
    """
    关键点解码函数
    功能：将网络输出的相对关键点坐标转换为绝对坐标
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), 1)
    return landms


def iou(b1, b2):
    """
    计算IoU（交并比）
    功能：计算两个边界框的重叠度
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]  # 第一个框的坐标
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]  # 第二个框的坐标

    # 计算交集区域
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)  # 交集面积

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # 第一个框的面积
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # 第二个框的面积

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)  # 计算IoU
    return iou

def non_max_suppression(detection, conf_thres=0.5, nms_thres=0.03):
    """
    非极大值抑制
    功能：去除重叠的检测框，保留置信度最高的
    """
    # ------------------------------------------#
    #   找出该图片中得分大于门限函数的框。
    #   在进行重合框筛选前就
    #   进行得分的筛选可以大幅度减少框的数量。
    # ------------------------------------------#
    mask = detection[:, 4] >= conf_thres  # 置信度过滤
    detection = detection[mask]

    if len(detection) <= 0:
        return []

    best_box = []
    scores = detection[:, 4]
    # 2、根据得分对框进行从大到小排序。
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    while np.shape(detection)[0]>0:
     # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
         best_box.append(detection[0])
         if len(detection) == 1:
             break
         ious = iou(best_box[-1], detection[1:])
         detection = detection[1:][ious < nms_thres]
    return np.array(best_box)


def retinaface_correct_boxes(result, input_shape, image_shape):
    """
    边界框坐标修正
    功能：将网络输出坐标转换为原图坐标
    """
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result


def Alignment_1(img, landmark):
    """
    人脸对齐函数
    功能：根据眼睛关键点对人脸进行旋转对齐
    """
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]  # 左眼到右眼的x距离
        y = landmark[36, 1] - landmark[45, 1]  # 左眼到右眼的y距离
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]    # 左眼到右眼的x距离
        y = landmark[0, 1] - landmark[1, 1]    # 左眼到右眼的y距离
    # 眼睛连线相对于水平线的倾斜角
    if x == 0:
        angle = 0
    else:
        # 计算它的弧度制
        angle = math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)  # 旋转中心

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)  # 获取旋转矩阵
    # 仿射函数
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark

#   计算人脸距离
#---------------------------------#
def face_distance(face_encodings, face_to_compare):
    """
    计算人脸特征距离
    功能：计算待识别人脸与已知人脸的特征距离
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    # (n, )
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)  # 欧几里得距离

#   比较人脸
#---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    """
    人脸比较函数
    功能：比较待识别人脸与已知人脸，判断是否为同一人
    """
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance), dis  # 返回匹配结果和距离

def myFunc_face(rknn ,rknn2, frame ,num):
    """
    人脸识别主函数
    功能：检测图像中的人脸，进行身份识别
    参数：
        rknn: RetinaFace检测模型
        rknn2: MobileFaceNet识别模型
        frame: 输入图像
        num: 帧号
    返回：处理后的图像
    """
    global name_ten
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 进行检测
    old_image = frame.copy()  # 保存原图副本
    frame = np.array(frame, np.float32)
    im_height, im_width, _ = np.shape(frame)
    scale = [
        np.shape(frame)[1], np.shape(frame)[0], np.shape(frame)[1], np.shape(frame)[0]
    ]
    scale_for_landmarks = [
        np.shape(frame)[1], np.shape(frame)[0], np.shape(frame)[1], np.shape(frame)[0],
        np.shape(frame)[1], np.shape(frame)[0], np.shape(frame)[1], np.shape(frame)[0],
        np.shape(frame)[1], np.shape(frame)[0]
    ]
    # 对输入图像进行resize
    frame = letterbox_image(frame, [640, 640])
    frame = frame.astype(dtype=np.float32)
    # frame = np.array((104,117,123),np.float32)
    frame = np.expand_dims(frame, axis=0)
    # 获取anchors
    anchors = Anchors(cfg_mnet, image_size=(640, 640)).get_anchors()
    # 将图像输入检测模型
    t1 = time.time()
    #print('--> Running model')
    start = time.time()
    outputs = rknn.inference(inputs=[frame])  # 人脸检测推理
    end = time.time()
    #print('时间:{}'.format(end - start))
    # 输出数据转为numpy数据格式
    loc = np.array(outputs[0]).squeeze()      # 边界框回归输出
    conf = np.array(outputs[1]).squeeze()     # 置信度输出
    landms = np.array(outputs[2]).squeeze()   # 关键点输出
    boxes = decode(loc, anchors, cfg_mnet['variance'])  # 解码边界框
    conf = conf[:, 1:2]  # 取正样本置信度
    landms = decode_landm(landms, anchors, cfg_mnet['variance'])  # 解码关键点
    # 对人脸框进行堆叠
    boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)
    boxes_conf_landms = non_max_suppression(boxes_conf_landms, 0.5)  # 非极大值抑制
    if len(boxes_conf_landms) <= 0:
        frame = old_image  # 未检测到人脸
    else:
        boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([640, 640]),
                                                     np.array([im_height, im_width]))
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
        #   Facenet编码部分-开始
        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:
            # ----------------------#
            #   图像截取，人脸矫正
            # ----------------------#
            boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
            crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                       int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]  # 裁剪人脸区域
            landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])  # 关键点坐标转换
            crop_img, _ = Alignment_1(crop_img, landmark)  # 人脸对齐
            crop_img = crop_img.astype(np.uint8)
            crop_img = Image.fromarray(crop_img)

            # crop_img = np.array(letterbox_image(np.uint8(crop_img), (160, 160)))
            crop_img = crop_img.resize((160, 160), Image.BICUBIC)  # 缩放到160x160

            # crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img, 0)
            #crop_img = crop_img.astype(dtype=np.float32)
            crop_img = np.asarray(crop_img, np.float32)
            # print(crop_img)

            face_encoding = np.array(rknn2.inference(data_format='nhwc', inputs=[crop_img])[0])  # 人脸特征提取
            # face_encoding = np.array(face_encoding)
            face_encoding = face_encoding.flatten()  # 展平特征向量

            # print(face_encoding)
            face_encodings.append(face_encoding)

            #   人脸特征比对-开始
        face_names = []
        for face_encoding in face_encodings:
            # -----------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            # -----------------------------------------------------#
            # face_encoding = np.array(face_encoding)
            # face_encoding = face_encoding.reshape(-1)
            matches, face_distances = compare_faces(known_face_encodings, face_encoding, tolerance=0.9)  # 人脸比较
            name = "Unknown"  # 默认未知
            # -----------------------------------------------------#
            #   取出这个最近人脸的评分
            #   取出当前输入进来的人脸，最接近的已知人脸的序号
            # -----------------------------------------------------#
            best_match_index = np.argmin(face_distances)  # 找到最匹配的人脸
            #print(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]  # 获取匹配的姓名
            face_names.append(name)
        name_ten.append(face_names)


        for i, b in enumerate(boxes_conf_landms):
            text = "{:.4f}".format(b[4])  # 置信度文本
            b = list(map(int, b))
            # ---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            # ---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # 绘制人脸框
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))  # 绘制置信度
            # ---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            # ---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)   # 左眼
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4) # 右眼
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4) # 鼻子
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)  # 左嘴角
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)  # 右嘴角

            name = face_names[i]
            #print(f"name = {name}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(old_image, name, (b[0], b[3] - 15), font, 0.75, (255, 255, 255), 2)  # 绘制姓名
            frame = old_image

    # 每2帧发送一次识别结果
    if num % 2 == 0:
        if len(name_ten) != 2:
            type = 0
            name_ten.clear()
        else:
            for elements in zip(*name_ten):
                if len(set(elements)) == 1:  # 连续两帧识别结果一致
                    type = 0
                    person_name = ','.join(name_ten[0])
                    data_length = len(name_ten[0])
                    # data_to_send = struct.pack(struct_format, type, person_name.encode('utf-8'), data_length)
                    # client.sendto(data_to_send, server_ip)
                    #print("finish send")
            name_ten.clear()
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame
