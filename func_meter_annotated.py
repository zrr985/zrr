# ============================================================================
# 仪表检测功能模块 (func_meter.py)
# 功能：基于YOLOv8-Seg的仪表指针和刻度检测，支持实例分割
# 特点：使用实例分割技术提取指针和刻度的像素级掩码
# 来源：改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
# ============================================================================

import cv2                    # OpenCV库，用于图像处理和显示
import numpy as np            # 数值计算库
import time                   # 时间模块

# 检测参数配置
OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640  # 目标阈值、NMS阈值、图像尺寸

CLASSES = ("background", "pointer", "scale")  # 检测类别：背景、指针、刻度


class Colors:
    """
    颜色调色板类
    功能：提供Ultralytics颜色调色板，用于可视化
    """
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]  # 转换为RGB颜色
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """
        获取颜色
        :param i: 颜色索引
        :param bgr: 是否返回BGR格式
        :return: 颜色元组
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c  # BGR格式需要调整通道顺序

    @staticmethod
    def hex2rgb(h):
        """
        十六进制颜色转RGB
        :param h: 十六进制颜色字符串
        :return: RGB颜色元组
        """
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def sigmoid(x):
    """
    Sigmoid激活函数
    功能：将输入值映射到(0,1)区间
    """
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
    """
    过滤检测框函数
    功能：根据置信度阈值过滤低置信度的检测框，同时处理分割部分
    """
    box_confidences = box_confidences.reshape(-1)  # 展平置信度数组
    candidate, class_num = box_class_probs.shape   # 获取候选框数量和类别数

    class_max_score = np.max(box_class_probs, axis=-1)  # 每个框的最大类别分数
    classes = np.argmax(box_class_probs, axis=-1)       # 每个框的预测类别

    # 根据置信度阈值过滤
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]      # 过滤后的边界框
    classes = classes[_class_pos]   # 过滤后的类别
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]  # 过滤后的分割部分

    return boxes, classes, scores, seg_part


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


def dfl(position):
    """
    分布焦点损失函数 (Distribution Focal Loss)
    功能：将网络输出的分布转换为具体的坐标值
    """
    n, c, h, w = position.shape  # 获取张量维度
    p_num = 4  # 每个坐标的分布数量
    mc = c // p_num  # 每个坐标的分布长度
    y = position.reshape(n, p_num, mc, h, w)  # 重塑张量
    
    # 向量化softmax，提高数值稳定性
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # 减去最大值防止溢出
    y = e_y / np.sum(e_y, axis=2, keepdims=True)  # 归一化
    
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)  # 累积矩阵
    y = (y * acc_metrix).sum(2)  # 计算期望值
    return y


def box_process(position):
    """
    边界框处理函数
    功能：将网络输出转换为具体的边界框坐标
    """
    grid_h, grid_w = position.shape[2:4]  # 获取网格尺寸
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))  # 创建网格坐标
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)  # 合并网格坐标
    stride = np.array([IMG_SIZE//grid_h, IMG_SIZE//grid_w]).reshape(1,2,1,1)  # 计算步长

    position = dfl(position)  # 应用DFL
    box_xy  = grid +0.5 -position[:,0:2,:,:]   # 计算左上角坐标
    box_xy2 = grid +0.5 +position[:,2:4,:,:]  # 计算右下角坐标
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)  # 合并坐标

    return xyxy


def yolov8_seg_post_process(input_data):
    """
    YOLOv8分割后处理函数
    功能：处理YOLOv8-Seg模型的输出，得到检测框和分割掩码
    """
    proto = input_data[-1]  # 原型掩码
    boxes, scores, classes_conf, seg_parts = [], [], [], []
    defualt_branch=3  # 默认分支数
    pair_per_branch = len(input_data)//defualt_branch  # 每个分支的输出数量

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))  # 处理边界框输出
        classes_conf.append(input_data[pair_per_branch*i+1])      # 处理类别置信度输出
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))  # 创建分数数组
        seg_parts.append(input_data[pair_per_branch*i+3])         # 处理分割部分输出

    def sp_flatten(_in):
        """
        空间展平函数
        功能：将特征图展平为二维数组
        """
        ch = _in.shape[1]  # 通道数
        _in = _in.transpose(0,2,3,1)  # 调整维度顺序
        return _in.reshape(-1, ch)  # 展平

    # 展平所有输出
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_parts = [sp_flatten(_v) for _v in seg_parts]

    # 合并所有分支的输出
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_parts = np.concatenate(seg_parts)

    # 根据阈值过滤
    boxes, classes, scores, seg_parts = filter_boxes(boxes, scores, classes_conf, seg_parts)

    # 按置信度排序
    zipped = zip(boxes, classes, scores, seg_parts)
    sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    result = zip(*sort_zipped)

    max_nms = 30000  # 最大NMS数量
    n = boxes.shape[0]  # 检测框数量
    if not n:
        return None, None, None, None
    elif n > max_nms:  # 超过最大数量限制
        boxes, classes, scores, seg_parts = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_parts = [np.array(x) for x in result]

    # 非极大值抑制
    nboxes, nclasses, nscores, nseg_parts = [], [], [], []
    for c in set(classes):
        inds = np.where(classes == c)  # 找到当前类别的所有检测框
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        n = seg_parts[inds]
        keep = nms_boxes(b, s)  # 应用NMS

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
            nseg_parts.append(n[keep])

    if not nclasses and not nscores:
        return None, None, None, None

    # 合并NMS后的结果
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_parts = np.concatenate(nseg_parts)

    # 处理分割掩码
    ph, pw = proto.shape[-2:]  # 原型掩码的高度和宽度
    proto = proto.reshape(seg_parts.shape[-1], -1)  # 重塑原型掩码
    seg_imgs = np.matmul(seg_parts, proto)  # 矩阵乘法生成分割掩码
    seg_imgs = sigmoid(seg_imgs)  # Sigmoid激活
    seg_imgs = seg_imgs.reshape(-1, ph, pw)  # 重塑为3D掩码

    # 调整边界框坐标以匹配下采样后的掩码
    downsampled_bboxes = boxes.copy()
    downsampled_bboxes[:, 0] *= pw / 640  # 缩放x坐标
    downsampled_bboxes[:, 2] *= pw / 640
    downsampled_bboxes[:, 3] *= ph / 640  # 缩放y坐标
    downsampled_bboxes[:, 1] *= ph / 640

    # 裁剪掩码到边界框区域
    seg_imgs_cropped = _crop_mask(seg_imgs, downsampled_bboxes)
    seg_imgs_resize = np.array([cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR) for img in seg_imgs_cropped])
    seg_imgs = seg_imgs_resize > 0.5  # 二值化掩码

    # 根据类别提取指针和刻度的掩码
    pointer_mask = seg_imgs[classes == 1]  # 类别 1: pointer
    scale_mask = seg_imgs[classes == 2]    # 类别 2: scale

    return boxes, classes, scores, seg_imgs, pointer_mask, scale_mask


def _crop_mask(masks, boxes):
    """
    掩码裁剪函数
    功能：将预测的掩码裁剪到边界框区域
    """
    n, h, w = masks.shape  # 掩码数量、高度、宽度
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)  # 边界框坐标
    r = np.arange(w, dtype=np.float32)[None, None, :]  # 行坐标
    c = np.arange(h, dtype=np.float32)[None, :, None]  # 列坐标

    # 创建掩码，只在边界框内有效
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def draw(image, boxes, scores, classes, ratio, padding):
    """
    绘制检测结果函数
    功能：在图像上绘制检测框和标签
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box

        # 坐标还原到原图尺寸
        top = (top - padding[0])/ratio[0]
        left = (left - padding[1])/ratio[1]
        right = (right - padding[0])/ratio[0]
        bottom = (bottom - padding[1])/ratio[1]
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)

        # 绘制检测框和标签
        cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def merge_seg(image, seg_imgs, classes, padding):
    """
    分割掩码合并函数
    功能：将分割掩码合并到原图像上
    """
    seg_img = np.sum(seg_imgs, axis=0)  # 合并所有掩码
    seg = seg_img.astype(np.uint8)
    seg = seg * 128  # 调整掩码强度
    seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)  # 转换为彩色
    
    # 根据填充信息裁剪掩码
    if padding[1] == 0:
        if padding[0] != 0:
            seg2 = seg[:, padding[0]:-padding[2]]
        else:
            seg2 = seg
    else:
        if padding[0] == 0:
            seg2 = seg[padding[1]:-padding[3], :]
        else:
            seg2 = seg[padding[1]:-padding[3], padding[0]:-padding[2]]
            
    seg = cv2.resize(seg2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)  # 调整掩码尺寸
    image = cv2.add(image, seg)  # 将掩码叠加到原图像上
    return image


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    """
    图像缩放和填充函数
    功能：将图像缩放到指定尺寸，保持宽高比，不足部分用指定颜色填充
    """
    shape = im.shape[:2]  # 当前图像尺寸 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例

    ratio = r, r  # 宽高缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后的尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 需要填充的像素数

    dw /= 2  # 左右各填充一半
    dh /= 2  # 上下各填充一半

    if shape[::-1] != new_unpad:  # 如果需要缩放
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return im, ratio, (left, top, right, bottom)  # 返回处理后的图像、缩放比例、填充信息


def myFunc(rknn_lite, IMG):
    """
    仪表检测主函数
    功能：检测图像中的仪表指针和刻度
    参数：
        rknn_lite: RKNN模型
        IMG: 输入图像
    返回：处理后的图像、指针掩码、刻度掩码
    """
    initTime = time.time()
    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)  # BGR转RGB

    IMG2, ratio, padding = letterbox(IMG2)  # 图像缩放和填充
    # 强制放缩
    # IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))
    IMG2 = np.expand_dims(IMG2, 0)  # 添加batch维度
    prepostTime = time.time()
    # print("预处理时间:\t", prepostTime - initTime, "秒")
    
    # 模型推理
    outputs = rknn_lite.inference(inputs=[IMG2],data_format=['nhwc'])
    inferenceTime = time.time()
    # print("推理时间:\t", inferenceTime - prepostTime, "秒")
    #print("oups1",len(outputs))
    #print("oups2",outputs[0].shape)

    # 后处理
    boxes, classes, scores, seg_imgs, pointer_mask, scale_mask = yolov8_seg_post_process(outputs)
    postprocessTime = time.time()
    # print("后处理时间:\t", postprocessTime - inferenceTime, "秒")

    # 绘制检测结果
    if boxes is not None:
        IMG = merge_seg(IMG, seg_imgs, classes, padding)  # 合并分割掩码
        #draw(IMG, boxes, scores, classes, ratio, padding)  # 绘制检测框（注释掉）
    
    return IMG, pointer_mask, scale_mask
