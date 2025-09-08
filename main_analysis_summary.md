# main.py 核心架构详细分析

## 1. 系统整体架构

### 1.1 核心设计理念
- **模块化设计**：每个功能模块独立，便于维护和扩展
- **多线程并发**：支持多个检测任务同时运行
- **生产者-消费者模式**：摄像头捕获 → 队列 → 模型推理 → 结果处理
- **UDP通信协议**：实时数据传输和控制命令处理

### 1.2 系统组件关系图
```
UDP服务器 (端口8870)
    ↓
任务管理器 (TaskManager)
    ↓
摄像头管理器 (CameraManager)
    ↓
检测任务线程 (红外/RGB)
    ↓
RKNN模型池 (rknnpool_*.py)
    ↓
检测功能模块 (func_*.py)
```

## 2. 全局变量管理详解

### 2.1 状态变量
```python
# 检测结果存储
classes_data = []        # 多类别检测结果
class_inf = 0            # 红外入侵检测结果 (0=无入侵, 1=有入侵)
class_flame = None       # 火焰检测结果 (None=未检测, 0=无火焰, 1=有火焰)
class_smoke = None       # 吸烟检测结果 (None=未检测, 0=无吸烟, 1=有吸烟)
class_hardhat = None     # 安全帽检测结果 (None=未检测, 0=戴安全帽, 1=未戴安全帽)
state_env = False        # 全局环境状态标志
```

### 2.2 任务代号映射
```python
TASK_CODES = {
    3: '红外入侵检测',    # 使用红外摄像头
    4: '火焰检测',        # 使用RGB摄像头
    0: '人脸识别',        # 使用RGB摄像头
    1: '仪表检测',        # 使用RGB摄像头
    2: '安全帽检测',      # 使用RGB摄像头
    5: '吸烟检测'         # 使用RGB摄像头
}
```

## 3. UDP通信协议详解

### 3.1 数据帧格式
```python
frame_format_meter = 'IIIIIIIff'  # 9个字段的二进制数据包
# 字段含义：
# 第1位：请求类型 (REQUEST_DATA=0, GIVE_DATA=1, CONTROL_COMMOND=2)
# 第2位：设备类型 (SERVER=0, VISION=1, ROBOT=2, METER=3)
# 第3位：子设备类型
# 第4位：子任务类型 (SUB_FACE=0, SUB_METER=1, SUB_SMOKE_HAT=2, SUB_INF=3, SUB_THR=4, SUB_GAS=5)
# 第5位：状态 (EXIST=1, NOT_EXIST=0, FINSH=3)
# 第6-9位：预留字段或具体数据
```

### 3.2 通信流程
1. **控制命令流程**：
   - 客户端发送控制命令 → UDP服务器接收 → 任务管理器处理 → 启动/停止检测任务
   
2. **数据请求流程**：
   - 客户端请求数据 → UDP服务器接收 → 查询全局变量 → 打包响应数据 → 发送给客户端

3. **异常推送流程**：
   - 检测任务发现异常 → 连续确认机制 → 调用send_abnormal_data() → UDP推送异常数据

## 4. 任务调度机制详解

### 4.1 TaskManager类核心功能
```python
class TaskManager:
    def __init__(self):
        self.cam_manager = CameraManager()      # 摄像头管理器
        self.threads = {}                      # 任务线程字典
        self.stop_events = {}                  # 停止事件字典
        self.frame_queues = {                  # 帧队列字典
            'infrared': queue.Queue(maxsize=5), # 红外摄像头队列
            'rgb': queue.Queue(maxsize=5)       # RGB摄像头队列
        }
        self.camera_threads = {}               # 摄像头线程字典
        self.display_queue = queue.Queue(maxsize=10)  # 显示队列
```

### 4.2 任务启动流程
1. **摄像头分配**：
   - 红外任务 (task_code=3) → 使用红外摄像头
   - RGB任务 (task_code=0,2,4,5) → 使用RGB摄像头

2. **线程创建**：
   - 摄像头捕获线程 → camera_capture()
   - 检测任务线程 → task_worker() → 具体检测函数

3. **资源管理**：
   - 队列管理：防止内存溢出
   - 线程同步：使用Event控制停止
   - 资源释放：自动清理摄像头和模型资源

### 4.3 任务停止流程
1. **正常停止**：设置stop_event → 等待线程结束 → 释放资源
2. **强制停止**：stop_all_tasks() → 停止所有任务和摄像头
3. **红外任务特殊处理**：stop_inf_tasks() → 专门释放红外摄像头

## 5. 检测任务函数详解

### 5.1 通用检测任务结构
```python
def detection_task(frame_queue, display_queue, stop_event):
    # 1. 初始化模型池
    pool = rknnPoolExecutor_xxx(model_path, TPEs, func)
    
    # 2. 主检测循环
    while not stop_event.is_set():
        frame = frame_queue.get(timeout=1)  # 获取图像帧
        pool.put(frame)                     # 放入模型池
        result, flag = pool.get()           # 获取检测结果
        
        # 3. 异常检测和推送
        if abnormal_detected:
            abnormal_count += 1
            if abnormal_count >= threshold and not abnormal_pushed:
                send_abnormal_data(task_code, 1)
                abnormal_pushed = True
        
        # 4. 显示处理
        display_queue.put((processed_frame, task_code))
    
    # 5. 资源清理
    pool.release()
```

### 5.2 异常检测机制
- **连续确认**：避免误报，需要连续检测多帧才推送
- **阈值设置**：
  - 红外入侵：10帧确认
  - 安全帽检测：10帧确认
  - 火焰检测：10帧确认
  - 吸烟检测：20帧确认（更严格）
- **状态重置**：检测到正常状态时重置计数器和推送标志

## 6. UDP服务器实现详解

### 6.1 UDPServer类功能
```python
class UDPServer:
    def __init__(self, ip='127.0.0.1', port=8870):
        # 创建UDP套接字并绑定地址
    
    def receive_frame(self):
        # 接收数据并解析为9字段元组
    
    def send_response(self, addr, message):
        # 发送响应数据到指定地址
```

### 6.2 命令处理逻辑
1. **控制命令处理** (frame_data[0] == 2)：
   - 启动任务 (action == 1)
   - 停止任务 (action == 2)
   - 关闭所有 (action == 0)

2. **数据请求处理** (frame_data[0] == 0)：
   - 人脸数据请求 (frame_data[3] == 0)
   - 红外入侵数据请求 (frame_data[3] == 3)
   - 安全帽检测数据请求 (frame_data[3] == 2)
   - 火焰检测数据请求 (frame_data[3] == 4)
   - 吸烟检测数据请求 (frame_data[3] == 5)

3. **仪表检测特殊处理** (frame_data[1] == 3)：
   - 启动人脸识别任务
   - 停止所有任务

## 7. 摄像头管理详解

### 7.1 CameraManager类设计
```python
class CameraManager:
    def __init__(self):
        self.infrared_cam = None   # 红外摄像头对象
        self.rgb_cam = None        # RGB摄像头对象
        self.lock = threading.Lock()  # 线程锁
    
    def get_infrared_camera(self):
        # 懒加载模式，首次调用时打开摄像头
    
    def get_rgb_camera(self):
        # 懒加载模式，首次调用时打开摄像头
    
    def release_cameras(self):
        # 释放所有摄像头资源
```

### 7.2 摄像头类型识别
- **红外摄像头**：通过video_number.inf_numbers获取编号列表
- **RGB摄像头**：通过video_number.rgb_numbers获取编号列表
- **自动识别**：open_camera()函数自动尝试打开可用摄像头

### 7.3 线程安全机制
- **锁机制**：使用threading.Lock()防止多线程同时访问摄像头
- **资源管理**：自动释放摄像头资源，避免资源泄漏

## 8. 系统启动流程

### 8.1 主程序入口
```python
if __name__ == "__main__":
    task_manager = TaskManager()  # 创建任务管理器
    
    # 启动显示线程
    display_thread = threading.Thread(target=show_frames, args=(task_manager.display_queue,))
    display_thread.start()
    
    # 启动UDP命令接收（主线程）
    udp_receive_commands(task_manager)
```

### 8.2 初始化顺序
1. 创建TaskManager实例
2. 启动显示线程
3. 创建UDP服务器
4. 进入主循环接收命令
5. 根据命令启动/停止检测任务

## 9. 关键接口和调用关系

### 9.1 模块导入关系
```python
# 核心模块
import func_face, func_v7, video_number

# 模型池模块
from rknnpool_rgb import rknnPoolExecutor_face
from rknnpool_inf import rknnPoolExecutor_inf
from rknnpool_smoke import rknnPoolExecutor_smoke_hat
from rknnpool_flame import rknnPoolExecutor_flame
from rknnpool_hardhat import rknnPoolExecutor_hardhat
from rknnpool_smoke_single import rknnPoolExecutor_smoke

# 检测函数模块
from func_face import myFunc_face
from func_v7 import myFunc_inf, data_ten_inf
from func_smoke_hat import myFunc_smoke_hat
from func_flame import myFunc_flame
from func_hardhat import myFunc_hardhat
from func_smoke import myFunc_smoke
```

### 9.2 关键函数调用链
```
UDP命令接收 → 任务管理器 → 摄像头管理 → 检测任务 → 模型池 → 检测函数
    ↓              ↓           ↓           ↓         ↓         ↓
数据响应 ← 全局变量更新 ← 异常推送 ← 结果处理 ← 模型推理 ← 图像预处理
```

## 10. 系统特点和优势

### 10.1 技术特点
- **多任务并发**：支持6种检测任务同时运行
- **智能防误报**：连续检测机制避免误报
- **实时通信**：UDP协议保证实时性
- **模块化设计**：各功能模块独立，易于维护
- **硬件优化**：针对RKNN NPU优化，提高推理效率
- **双摄像头支持**：支持RGB和红外摄像头
- **自动资源管理**：智能管理摄像头和模型资源

### 10.2 系统优势
- **高可靠性**：异常处理机制完善
- **高扩展性**：易于添加新的检测任务
- **高实时性**：UDP通信和队列机制保证实时响应
- **高稳定性**：线程安全和资源管理机制完善

## 11. 学习建议

### 11.1 学习顺序
1. **理解全局变量管理**：掌握状态变量的作用
2. **学习UDP通信协议**：理解数据帧格式和通信流程
3. **掌握任务调度机制**：理解TaskManager的工作原理
4. **分析检测任务函数**：理解生产者-消费者模式
5. **研究摄像头管理**：理解资源管理和线程安全

### 11.2 重点关注
- **UDP数据帧格式**：这是理解整个系统的关键
- **任务调度逻辑**：理解多任务并发管理
- **异常检测机制**：理解防误报的设计思路
- **资源管理**：理解内存和硬件资源的管理方式

### 11.3 实践建议
- 使用测试工具验证UDP通信
- 观察不同检测任务的运行状态
- 分析系统资源使用情况
- 尝试添加新的检测任务

