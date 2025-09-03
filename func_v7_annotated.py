# ============================================================================
# 红外入侵检测功能模块 (func_v7.py)
# 功能：基于YOLOv7的人体入侵检测，支持实时视频流处理
# 特点：使用YOLOv7模型进行人体检测，支持UDP数据发送
# ============================================================================

import cv2                    # OpenCV库，用于图像处理和显示
import numpy as np            # 数值计算库
import socket                 # 网络通信
import struct                 # 二进制数据打包

# 检测参数配置
OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.5, 0.6, 640  # 目标阈值、NMS阈值、图像尺寸

CLASSES = ("person")  # 检测类别：人体

# 创建UDP客户端socket，用于发送检测结果
client = socket.socket(family = socket.AF_INET,type = socket.SOCK_DGRAM)

server_ip = ('127.0.0.1',8848)  # 服务器IP地址和端口
# 无需和服务端建立连接
struct_format = '!i48si'      # 数据包格式：类型(4字节) + 信息(48字节) + 长度(4字节)
data_ten_inf = []             # 存储最近10帧的检测结果

def sigmoid(x):
    """
    Sigmoid激活函数
    功能：将输入值映射到(0,1)区间
    """
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    """
    坐标转换函数
    功能：将[x, y, w, h]格式转换为[x1, y1, x2, y2]格式
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):
    """
    特征图处理函数
    功能：处理YOLOv7的特征图输出，提取边界框、置信度和类别概率
    """
    anchors = [anchors[i] for i in mask]  # 选择对应的锚框
    grid_h, grid_w = map(int, input.shape[0:2])  # 获取网格尺寸

    box_confidence = sigmoid(input[..., 4])  # 边界框置信度
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])  # 类别概率

    box_xy = sigmoid(input[..., :2]) * 2 - 0.5  # 边界框中心点坐标

    # 创建网格坐标
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid  # 加上网格偏移
    box_xy *= int(IMG_SIZE / grid_h)  # 缩放到原图尺寸

    box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)  # 边界框宽高
    box_wh = box_wh * anchors  # 乘以锚框尺寸

    box = np.concatenate((box_xy, box_wh), axis=-1)  # 合并坐标和尺寸

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """
    过滤检测框函数
    功能：根据置信度阈值过滤低置信度的检测框
    """
    box_classes = np.argmax(box_class_probs, axis=-1)  # 预测类别
    box_class_scores = np.max(box_class_probs, axis=-1)  # 类别最大分数
    pos = np.where(box_confidences[..., 0] >= 0.5)  # 置信度过滤

    boxes = boxes[pos]  # 过滤后的边界框
    classes = box_classes[pos]  # 过滤后的类别
    scores = box_class_scores[pos]  # 过滤后的分数

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """
    非极大值抑制函数
    功能：去除重叠的检测框，保留置信度最高的
    """
    x = boxes[:, 0]  # 左上角x坐标
    y = boxes[:, 1]  # 左上角y坐标
    w = boxes[:, 2] - boxes[:, 0]  # 宽度
    h = boxes[:, 3] - boxes[:, 1]  # 高度

    areas = w * h  # 计算每个框的面积
    order = scores.argsort()[::-1]  # 按置信度降序排序

    keep = []  # 保留的框索引
    while order.size > 0:
        i = order[0]  # 当前最高置信度的框
        keep.append(i)

        # 计算当前框与其他框的交集
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)  # 交集宽度
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)  # 交集高度
        inter = w1 * h1  # 交集面积

        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算IoU
        inds = np.where(ovr <= NMS_THRESH)[0]  # 找到IoU小于阈值的框
        order = order[inds + 1]  # 更新候选框列表
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    """
    YOLOv5后处理函数
    功能：处理YOLOv5/YOLOv7模型的输出，得到最终的检测结果
    """
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 特征图掩码
    yolov5_anchors = [[10, 13], [16, 30], [33, 23],
                      [30, 61], [62, 45], [59, 119],
                      [116, 90], [156, 198], [373, 326]]  # YOLOv5锚框

    yolov7_anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
               [72, 146], [142, 110], [192, 243], [459, 401]]  # YOLOv7锚框

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, yolov7_anchors)  # 处理每个特征图
        b, c, s = filter_boxes(b, c, s)  # 过滤检测框
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    # 合并所有特征图的结果
    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)  # 坐标转换
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # 非极大值抑制
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)  # 找到当前类别的所有检测框
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)  # 应用NMS

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    # 合并NMS后的结果
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """
    绘制检测结果函数
    功能：在图像上绘制检测框和标签
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        # 绘制检测框和标签
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    """
    图像缩放和填充函数
    功能：将图像缩放到指定尺寸，保持宽高比，不足部分用指定颜色填充
    """
    shape = im.shape[:2]  # 当前图像尺寸 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后的尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 需要填充的像素数

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)  # 返回处理后的图像、缩放比例、填充信息

def myFunc_inf(rknn_lite, img, num):
    """
    红外入侵检测主函数
    功能：检测图像中的人体，判断是否有入侵
    参数：
        rknn_lite: RKNN模型
        img: 输入图像
        num: 帧号
    返回：处理后的图像、检测结果
    """
    global data_ten_inf
    class_exist = 0  # 检测结果标志
    
    # Set inputs
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 调整图像尺寸

    # Inference
    #print('--> Running model')
    img2 = np.expand_dims(img, 0)  # 添加batch维度
    outputs = rknn_lite.inference(inputs=[img2], data_format=['nhwc'])  # 模型推理

    # post process
    input0_data = outputs[0]  # 第一个输出
    input1_data = outputs[1]  # 第二个输出
    input2_data = outputs[2]  # 第三个输出

    # 重塑输出数据
    input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))  # 调整维度顺序
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    # 后处理
    boxes, classes, scores = yolov5_post_process(input_data)
    if classes is not None:
        for cl in classes:
            data_ten_inf.append(CLASSES[cl])  # 记录检测结果
    #print(data_ten)
    
    # 每帧发送一次检测结果
    if num % 1 == 0:
        if len(data_ten_inf) != 1:
            type = 1
            data_ten_inf.clear()
            class_exist = 0
        else:
            class_exist = 1
            for elements in zip(*data_ten_inf):
                if len(set(elements)) == 1:  # 连续帧检测结果一致
                    type = 1
                    information = ','.join(data_ten_inf[0])
                    # data_length = len(data_ten_inf[0])
                    # data_to_send = struct.pack(struct_format, type, information.encode('utf-8'), data_length)
                    # client.sendto(data_to_send, server_ip)
                    #print("finish send")
            data_ten_inf.clear()

    # 颜色空间转换
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img, boxes, scores, classes)  # 绘制检测结果
    return img, class_exist
