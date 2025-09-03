# RKNN模型池学习总结

## 概述
本文档总结了视觉检测系统中`rknnpool_`系列文件的功能、工作流程和技术特点。这些文件实现了基于RKNN的模型池管理，为各种检测任务提供高效的并发推理能力。

## 1. RKNN模型池架构设计

### 核心组件
- **RKNNLite**: Rockchip神经网络推理引擎
- **ThreadPoolExecutor**: 线程池执行器，实现并发处理
- **Queue**: 任务队列，管理异步任务
- **NPU核心管理**: 多核心负载均衡

### 设计模式
- **池化模式**: 预先创建多个模型实例，避免重复初始化
- **生产者-消费者模式**: 通过队列管理任务提交和结果获取
- **轮询负载均衡**: 使用模运算实现模型实例的均匀分配

## 2. 各模块功能对比

### 2.1 人脸识别模型池 (rknnpool_rgb.py)

**特点：**
- **双模型架构**: RetinaFace + MobileFaceNet
- **复杂处理流程**: 检测 → 对齐 → 特征提取 → 识别
- **高精度要求**: 人脸识别需要高精度特征提取

**模型配置：**
```python
rknnModel1 = "retinaface_mob.rknn"  # 人脸检测
rknnModel2 = "mobilefacenet.rknn"   # 特征提取
```

**类名：** `rknnPoolExecutor_face`

### 2.2 红外入侵检测模型池 (rknnpool_inf.py)

**特点：**
- **单模型架构**: YOLOv7-tiny
- **实时性要求高**: 入侵检测需要快速响应
- **人体检测**: 专注于人体目标检测

**模型配置：**
```python
rknnModel = "yolov7_tiny.rknn"  # 人体检测
```

**类名：** `rknnPoolExecutor_inf`

### 2.3 火焰检测模型池 (rknnpool_flame.py)

**特点：**
- **单模型架构**: YOLOv8
- **安全检测**: 火焰检测对准确性要求高
- **实时监控**: 需要持续监控火焰状态

**模型配置：**
```python
rknnModel = "yolov8_flame.rknn"  # 火焰检测
```

**类名：** `rknnPoolExecutor_flame`

### 2.4 安全帽检测模型池 (rknnpool_hardhat.py)

**特点：**
- **单模型架构**: YOLOv8
- **安全合规**: 安全帽检测用于安全监控
- **多类别检测**: 检测是否佩戴安全帽

**模型配置：**
```python
rknnModel = "yolov8_hardhat.rknn"  # 安全帽检测
```

**类名：** `rknnPoolExecutor_hardhat`

### 2.5 仪表检测模型池 (rknnpool_meter.py)

**特点：**
- **分割模型**: YOLOv8-Seg
- **精确检测**: 需要精确的指针和刻度分割
- **特殊接口**: put方法不传递num参数

**模型配置：**
```python
rknnModel = "yolov8_seg.rknn"  # 仪表分割检测
```

**类名：** `rknnPoolExecutor`

### 2.6 吸烟检测模型池 (rknnpool_smoke.py)

**特点：**
- **多类别检测**: cigarette, face, smoking
- **行为识别**: 吸烟行为检测
- **复杂逻辑**: 需要组合多个检测结果

**模型配置：**
```python
rknnModel = "yolov8_smoke_hat.rknn"  # 吸烟检测
```

**类名：** `rknnPoolExecutor_smoke_hat`

### 2.7 吸烟检测单模型池 (rknnpool_smoke_single.py)

**特点：**
- **单模型架构**: 专用吸烟检测模型
- **简化版本**: 相比多类别版本更简单
- **独立部署**: 可独立使用的吸烟检测

**模型配置：**
```python
rknnModel = "smoking.rknn"  # 吸烟检测
```

**类名：** `rknnPoolExecutor_smoke`

## 3. 核心技术实现

### 3.1 NPU核心管理

**核心分配策略：**
```python
if id == 0:
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
elif id == 1:
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
elif id == 2:
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
elif id == -1:
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
```

**负载均衡：**
```python
rknn_list.append(initRKNN(rknnModel, i % 3))  # 循环分配核心
```

### 3.2 线程池管理

**任务提交：**
```python
self.queue.put(self.pool.submit(
    self.func, 
    self.rknnPool[self.num % self.TPEs], 
    frame, 
    self.num
))
```

**结果获取：**
```python
def get(self):
    if self.queue.empty():
        return None, False
    fut = self.queue.get()
    return fut.result(), True
```

### 3.3 资源管理

**初始化：**
```python
def __init__(self, rknnModel, TPEs, func):
    self.TPEs = TPEs
    self.queue = Queue()
    self.rknnPool = initRKNNs(rknnModel, TPEs)
    self.pool = ThreadPoolExecutor(max_workers=TPEs)
    self.func = func
    self.num = 0
```

**释放：**
```python
def release(self):
    self.pool.shutdown()
    for rknn_lite in self.rknnPool:
        rknn_lite.release()
```

## 4. 工作流程分析

### 4.1 初始化阶段
```
创建RKNNLite实例 → 加载模型文件 → 初始化NPU核心 → 创建模型池 → 初始化线程池
```

### 4.2 任务处理阶段
```
提交图像帧 → 轮询选择模型实例 → 提交到线程池 → 异步推理 → 结果返回
```

### 4.3 结果获取阶段
```
检查队列状态 → 获取Future对象 → 提取结果 → 返回处理结果
```





