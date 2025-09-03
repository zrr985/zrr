# ============================================================================
# 火焰检测功能模块 (func_flame.py)
# 功能：基于YOLOv8的火焰检测，支持滑动窗口检测机制
# 特点：使用滑动窗口减少误检，提高检测稳定性
# ============================================================================

import cv2                    # OpenCV库，用于图像处理和显示
import numpy as np            # 数值计算库
import time                   # 时间模块

# 检测参数配置
OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.3, 0.45, 640  # 目标阈值、NMS阈值、图像尺寸

CLASSES = ('flame')  # 检测类别：火焰

def filter_boxes(boxes, box_confidences, box_class_probs):
    """
    过滤检测框函数
    功能：根据置信度阈值过滤低置信度的检测框
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
    stride = np.array([IMG_SIZE // grid_h, IMG_SIZE // grid_w]).reshape(1, 2, 1, 1)  # 计算步长

    position = dfl(position)  # 应用DFL
    box_xy = grid + 0.5 - position[:, 0:2, :, :]   # 计算左上角坐标
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]  # 计算右下角坐标
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)  # 合并坐标

    return xyxy


def yolov8_post_process(input_data):
    """
    YOLOv8后处理函数
    功能：处理YOLOv8模型的输出，得到最终的检测结果
    """
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3  # 默认分支数
    pair_per_branch = len(input_data) // defualt_branch  # 每个分支的输出数量
    
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))  # 处理边界框输出
        classes_conf.append(input_data[pair_per_branch * i + 1])    # 处理类别置信度输出
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))  # 创建分数数组

    def sp_flatten(_in):
        """
        空间展平函数
        功能：将特征图展平为二维数组
        """
        ch = _in.shape[1]  # 通道数
        _in = _in.transpose(0, 2, 3, 1)  # 调整维度顺序
        return _in.reshape(-1, ch)  # 展平

    # 展平所有输出
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    # 合并所有分支的输出
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 根据阈值过滤
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # 非极大值抑制
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)  # 找到当前类别的所有检测框
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)  # 应用NMS

        if len(keep) != 0:
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


def draw(image, boxes, scores, classes, ratio, padding):
    """
    绘制检测结果函数
    功能：在图像上绘制检测框和标签
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box

        # 坐标还原到原图尺寸
        top = (top - padding[0]) / ratio[0]
        left = (left - padding[1]) / ratio[1]
        right = (right - padding[0]) / ratio[0]
        bottom = (bottom - padding[1]) / ratio[1]
        
        # 处理无效坐标
        if top is not None and not np.isnan(top):
            top = int(top)
        if left is not None and not np.isnan(left):
            left = int(left)

        # 绘制检测框和标签
        cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
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
    return im, ratio, (left, top)


class FlameDetector:
    """
    火焰检测器类
    功能：使用滑动窗口机制进行稳定的火焰检测
    """
    def __init__(self, detection_threshold=0.7, window_size=10):
        """
        初始化火焰检测器
        :param detection_threshold: 检测阈值，当窗口内检测到火焰的比例超过此值时，才认为有火焰
        :param window_size: 滑动窗口大小
        """
        self.detection_threshold = detection_threshold  # 检测阈值
        self.window_size = window_size                  # 窗口大小
        self.detection_history = []                     # 检测历史记录
        
    def detect(self, rknn_lite, img, frame_num):
        """
        检测图像中的火焰
        :param rknn_lite: RKNN模型
        :param img: 输入图像
        :param frame_num: 当前帧号
        :return: 处理后的图像, 火焰检测结果 (1: 有火焰, 0: 无火焰, None: 不确定)
        """
        # 图像预处理
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
        img_rgb, ratio, padding = letterbox(img_rgb)    # 图像缩放和填充
        img_rgb = np.expand_dims(img_rgb, 0)            # 添加batch维度

        # 模型推理
        outputs = rknn_lite.inference(inputs=[img_rgb], data_format=['nhwc'])
        boxes, classes, scores = yolov8_post_process(outputs)  # 后处理
        
        # 记录当前帧的检测结果
        has_flame = 1 if classes is not None and len(classes) > 0 else 0  # 是否有火焰
        self.detection_history.append(has_flame)  # 添加到历史记录
        
        # 保持历史记录的长度不超过窗口大小
        if len(self.detection_history) > self.window_size:
            self.detection_history.pop(0)  # 移除最旧的记录
            
        # 计算窗口内的检测比例
        detection_ratio = sum(self.detection_history) / len(self.detection_history)
        
        # 根据检测比例判断是否有火焰
        if len(self.detection_history) >= self.window_size:
            class_real = 1 if detection_ratio >= self.detection_threshold else 0
        else:
            class_real = None  # 窗口未满，返回不确定
            
        # 绘制检测结果
        if boxes is not None:
            draw(img, boxes, scores, classes, ratio, padding)
            
        # 显示检测状态（注释掉）
        #status_text = f"Flame: {class_real}"
        #cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return img, class_real


# 创建全局火焰检测器实例
flame_detector = FlameDetector(detection_threshold=0.7, window_size=10)

def myFunc_flame(rknn_lite, img, frame_num):
    """
    火焰检测函数
    :param rknn_lite: RKNN模型
    :param img: 输入图像
    :param frame_num: 当前帧号
    :return: 处理后的图像, 火焰检测结果
    """
    return flame_detector.detect(rknn_lite, img, frame_num)
