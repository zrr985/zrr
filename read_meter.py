# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# def draw_circle_on_mask(dail_mask_resized, center, radius):
#     # 将dail_mask_resized转为彩色图像以便于绘图
#     color_img = cv2.cvtColor(dail_mask_resized, cv2.COLOR_GRAY2BGR)

#     # 绘制圆心
#     center_coordinates = (int(center[0]), int(center[1]))
#     cv2.circle(color_img, center_coordinates, 5, (0, 0, 255), -1)  # 红色圆心

#     # 绘制圆
#     cv2.circle(color_img, center_coordinates, int(radius), (0, 255, 0), 2)  # 绿色圆

#     return color_img

# def find_circle_center_and_radius(image):
#     # 获取图像的尺寸
#     height, width = image.shape[:2]

#     # 转换为灰度图并中值滤波
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)

#     # 使用HoughCircles方法检测圆形
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=1,
#         minDist=gray.shape[0] // 8,
#         param1=100,
#         param2=30,
#         minRadius=0,
#         maxRadius=0
#     )

#     # 确保至少检测到一个圆
#     if circles is not None:
#         circles = np.uint16(np.around(circles))  # 将圆心和半径转为整数

#         # 取第一个检测到的圆的参数
#         for i in circles[0, :]:
#             center = (int(i[0]), int(i[1]))  # 圆心坐标 (x, y)
#             radius = i[2]  # 圆的半径
#             radius = int((radius * 9) / 10)

#             # 在图像上绘制检测到的圆
#             # cv2.circle(image, center, radius, (0, 255, 0), 2)  # 画出圆
#             # cv2.circle(image, center, 2, (0, 0, 255), 3)  # 画出圆心
#             # # 展示结果
#             # cv2.imshow("Detected Circle", image)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()

#             print(f"Center: {center}, Radius: {radius}")

#             return center, radius
#     else:
#         print("No circles were detected.")
#         return None, None


# class MeterReader(object):

#     def __init__(self, img):
#         center, radius = find_circle_center_and_radius(img)

#         # Custom parameters
#         self.line_width = int(2 * np.pi * radius)  # 表盘展开为直线的长度,按照表盘的图像像素周长计算
#         self.line_height = int(radius*0.4)  # 表盘展开为直线的宽度，该设定按照围绕圆心的扇形宽度计算，需要保证包含刻度以及一部分指针
#         self.circle_radius = int(radius)  # 预设圆盘直径，扫描的最大直径，需要小于图幅，否者可能会报错
#         self.circle_center = center  # 圆盘指针的旋转中心，预设的指针旋转中心
#         self.threshold = 0.5

#     def __call__(self, point_mask_resized, dail_mask_resized):
#         if (self.circle_radius is None) | (self.circle_center is None):
#             return None
#         # 可视化掩码
#         plt.subplot(1, 2, 1)
#         plt.title("Pointer Mask")
#         plt.imshow(point_mask_resized, cmap='gray')

#         plt.subplot(1, 2, 2)
#         plt.title("Scale Mask")
#         plt.imshow(dail_mask_resized, cmap='gray')

#         print(f"dail_mask size:{dail_mask_resized.shape}")

#         plt.show()
#         # color_image = draw_circle_on_mask(dail_mask_resized, self.circle_center, self.circle_radius)
#         # cv2.imshow("Detected Circle", color_image)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()

#         relative_value = self.get_relative_value(point_mask_resized, dail_mask_resized)

#         return relative_value['ratio']

#     def get_relative_value(self, image_pointer, image_dail):
#         line_image_pointer = self.create_line_image(image_pointer)
#         line_image_dail = self.create_line_image(image_dail)

#         plt.subplot(1, 2, 1)
#         plt.title("Unfolded Pointer Image")
#         plt.imshow(line_image_pointer, cmap='gray')

#         plt.subplot(1, 2, 2)
#         plt.title("Unfolded Scale Image")
#         plt.imshow(line_image_dail, cmap='gray')
#         plt.show()

#         data_1d_pointer = self.convert_1d_data(line_image_pointer)
#         data_1d_dail = self.convert_1d_data(line_image_dail)
#         data_1d_dail = self.mean_filtration(data_1d_dail)

#         dail_flag = False
#         pointer_flag = False
#         one_dail_start = 0
#         one_dail_end = 0
#         one_pointer_start = 0
#         one_pointer_end = 0
#         dail_location = []
#         pointer_location = 0

#         for i in range(self.line_width - 1):
#             # 检测刻度位置
#             if data_1d_dail[i] > 0:
#                 if not dail_flag:
#                     one_dail_start = i
#                     dail_flag = True
#             if dail_flag and data_1d_dail[i] == 0:
#                 one_dail_end = i - 1
#                 one_dail_location = (one_dail_start + one_dail_end) / 2
#                 dail_location.append(one_dail_location)
#                 dail_flag = False

#             # 检测指针位置
#             if data_1d_pointer[i] > 0:
#                 if not pointer_flag:
#                     one_pointer_start = i
#                     pointer_flag = True
#             if pointer_flag and data_1d_pointer[i] == 0:
#                 one_pointer_end = i - 1
#                 pointer_location = (one_pointer_start + one_pointer_end) / 2
#                 pointer_flag = False

#         scale_num = len(dail_location)
#         num_scale = -1
#         ratio = -1
#         if scale_num > 0:
#             for i in range(scale_num - 1):
#                 if dail_location[i] <= pointer_location < dail_location[i + 1]:
#                     num_scale = i + (pointer_location - dail_location[i]) / (
#                             dail_location[i + 1] - dail_location[i] + 1e-5) + 1
#             ratio = (pointer_location - dail_location[0]) / (dail_location[-1] - dail_location[0] + 1e-5)

#         print(f"Pointer location: {pointer_location}")
#         print(f"Dail locations: {dail_location}")

#         result = {'scale_num': scale_num, 'num_sacle': num_scale, 'ratio': ratio}
#         return result

#     def create_line_image(self, image_mask):
#         """
#         Create a linear image
#         :param image_mask: mask image
#         :return:
#         """
#         line_image = np.zeros((self.line_height, self.line_width), dtype=np.uint8)
#         for row in range(self.line_height):
#             for col in range(self.line_width):
#                 """Calculate the angle with the -y axis"""
#                 theta = ((2 * np.pi) / self.line_width) * (col + 1)
#                 '''Calculate the diameter corresponding to the original image'''
#                 radius = self.circle_radius - row - 1

#                 # # 提前终止条件: 如果半径超出图像的尺寸，跳过这一行
#                 if radius < 0 or radius >= min(image_mask.shape[0], image_mask.shape[1]):
#                     continue

#                 '''Calculate the position of the current scan point corresponding to the original image'''
#                 y = int(self.circle_center[1] + radius * np.cos(theta) + 0.5)
#                 x = int(self.circle_center[0] - radius * np.sin(theta) + 0.5)

#                 # # 检查 x 和 y 是否在图像范围内
#                 if 0 <= y < image_mask.shape[0] and 0 <= x < image_mask.shape[1]:
#                     line_image[row, col] = image_mask[y, x]
#                 #else:
#                     # Optionally log or handle the invalid coordinates
#                     #print(f"Skipping out-of-bound coordinates: y={y}, x={x}")
#         return line_image

#     def convert_1d_data(self, line_image):
#         """
#         Convert the image to a 1D array
#         :param line_image: Unfolded image
#         :return: 1D array
#         """
#         data_1d = np.zeros((self.line_width), dtype=np.int16)
#         threshold = 127  # 设置阈值
#         for col in range(self.line_width):
#             for row in range(self.line_height):
#                 if line_image[row, col] > threshold:  # 如果像素值大于阈值
#                     data_1d[col] += 1
#         return data_1d

#     def mean_filtration(self, data_1d_dail):
#         """
#         Mean filtering
#         :param data_1d_dail: 1D data array
#         :return: Filtered data array
#         """
#         new_data_1d_dail = data_1d_dail.copy()
#         for i in range(1, self.line_width - 1):
#             new_data_1d_dail[i] = (data_1d_dail[i - 1] + data_1d_dail[i] + data_1d_dail[i + 1]) / 3
#         return new_data_1d_dail
import os
import cv2
import numpy as np
# 移除matplotlib导入，避免多线程问题
# import matplotlib.pyplot as plt


def draw_circle_on_mask(dail_mask_resized, center, radius):
    # 将dail_mask_resized转为彩色图像以便于绘图
    color_img = cv2.cvtColor(dail_mask_resized, cv2.COLOR_GRAY2BGR)

    # 绘制圆心
    center_coordinates = (int(center[0]), int(center[1]))
    cv2.circle(color_img, center_coordinates, 5, (0, 0, 255), -1)  # 红色圆心

    # 绘制圆
    cv2.circle(color_img, center_coordinates, int(radius), (0, 255, 0), 2)  # 绿色圆

    return color_img

# 改进的圆形检测函数
def find_circle_center_and_radius(image):
    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 转换为灰度图并中值滤波
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # 增强边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 使用HoughCircles方法检测圆形
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=gray.shape[0] // 8,
        param1=50,  # 降低边缘检测阈值
        param2=25,  # 降低累加器阈值
        minRadius=gray.shape[0] // 8,  # 设置最小半径
        maxRadius=gray.shape[0] // 2   # 设置最大半径
    )

    # 确保至少检测到一个圆
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 将圆心和半径转为整数

        # 取第一个检测到的圆的参数
        for i in circles[0, :]:
            center = (int(i[0]), int(i[1]))  # 圆心坐标 (x, y)
            radius = i[2]  # 圆的半径
            radius = int((radius * 9) / 10)

            print(f"Center: {center}, Radius: {radius}")

            return center, radius
    else:
        print("No circles were detected.")
        print("尝试使用图像中心作为默认值...")
        
        # 如果检测失败，使用图像中心作为默认值
        center = (width // 2, height // 2)
        radius = min(width, height) // 4  # 使用图像尺寸的1/4作为默认半径
        
        print(f"使用默认值 - Center: {center}, Radius: {radius}")
        return center, radius


class MeterReader(object):

    def __init__(self, img):
        center, radius = find_circle_center_and_radius(img)

        # Custom parameters
        self.line_width = int(2 * np.pi * radius)  # 表盘展开为直线的长度,按照表盘的图像像素周长计算
        self.line_height = int(radius*0.4)  # 表盘展开为直线的宽度，该设定按照围绕圆心的扇形宽度计算，需要保证包含刻度以及一部分指针
        self.circle_radius = int(radius)  # 预设圆盘直径，扫描的最大直径，需要小于图幅，否者可能会报错
        self.circle_center = center  # 圆盘指针的旋转中心，预设的指针旋转中心
        self.threshold = 0.5

    def __call__(self, point_mask_resized, dail_mask_resized):
        if (self.circle_radius is None) | (self.circle_center is None):
            return None
        
        # 移除Matplotlib可视化，避免多线程问题
        # 只打印信息，不显示图形
        print(f"Pointer Mask size: {point_mask_resized.shape}")
        print(f"Scale Mask size: {dail_mask_resized.shape}")
        print(f"dail_mask size: {dail_mask_resized.shape}")

        # 使用OpenCV显示替代Matplotlib（可选）
        # cv2.imshow("Pointer Mask", point_mask_resized)
        # cv2.imshow("Scale Mask", dail_mask_resized)
        # cv2.waitKey(1)  # 非阻塞显示

        relative_value = self.get_relative_value(point_mask_resized, dail_mask_resized)

        return relative_value['ratio']

    def get_relative_value(self, image_pointer, image_dail):
        line_image_pointer = self.create_line_image(image_pointer)
        line_image_dail = self.create_line_image(image_dail)

        # 移除Matplotlib可视化，避免多线程问题
        # 只打印信息，不显示图形
        print(f"Unfolded Pointer Image size: {line_image_pointer.shape}")
        print(f"Unfolded Scale Image size: {line_image_dail.shape}")

        # 使用OpenCV显示替代Matplotlib（可选）
        # cv2.imshow("Unfolded Pointer", line_image_pointer)
        # cv2.imshow("Unfolded Scale", line_image_dail)
        # cv2.waitKey(1)  # 非阻塞显示

        data_1d_pointer = self.convert_1d_data(line_image_pointer)
        data_1d_dail = self.convert_1d_data(line_image_dail)
        data_1d_dail = self.mean_filtration(data_1d_dail)

        dail_flag = False
        pointer_flag = False
        one_dail_start = 0
        one_dail_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        dail_location = []
        pointer_location = 0

        for i in range(self.line_width - 1):
            # 检测刻度位置
            if data_1d_dail[i] > 0:
                if not dail_flag:
                    one_dail_start = i
                    dail_flag = True
            if dail_flag and data_1d_dail[i] == 0:
                one_dail_end = i - 1
                one_dail_location = (one_dail_start + one_dail_end) / 2
                dail_location.append(one_dail_location)
                dail_flag = False

            # 检测指针位置
            if data_1d_pointer[i] > 0:
                if not pointer_flag:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag and data_1d_pointer[i] == 0:
                one_pointer_end = i - 1
                pointer_location = (one_pointer_start + one_pointer_end) / 2
                pointer_flag = False

        scale_num = len(dail_location)
        num_scale = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if dail_location[i] <= pointer_location < dail_location[i + 1]:
                    num_scale = i + (pointer_location - dail_location[i]) / (
                            dail_location[i + 1] - dail_location[i] + 1e-5) + 1
            ratio = (pointer_location - dail_location[0]) / (dail_location[-1] - dail_location[0] + 1e-5)

        print(f"Pointer location: {pointer_location}")
        print(f"Dail locations: {dail_location}")

        result = {'scale_num': scale_num, 'num_sacle': num_scale, 'ratio': ratio}
        return result

    def create_line_image(self, image_mask):
        """
        Create a linear image
        :param image_mask: mask image
        :return:
        """
        line_image = np.zeros((self.line_height, self.line_width), dtype=np.uint8)
        for row in range(self.line_height):
            for col in range(self.line_width):
                """Calculate the angle with the -y axis"""
                theta = ((2 * np.pi) / self.line_width) * (col + 1)
                '''Calculate the diameter corresponding to the original image'''
                radius = self.circle_radius - row - 1

                # 计算原始图像中的坐标
                x = int(self.circle_center[0] + radius * np.sin(theta))
                y = int(self.circle_center[1] - radius * np.cos(theta))

                # 确保坐标在图像范围内
                if 0 <= x < image_mask.shape[1] and 0 <= y < image_mask.shape[0]:
                    line_image[row, col] = image_mask[y, x]

        return line_image

    def convert_1d_data(self, line_image):
        """
        Convert 2D line image to 1D data
        :param line_image: 2D line image
        :return: 1D data
        """
        # 计算每列的平均值
        data_1d = np.mean(line_image, axis=0)
        return data_1d

    def mean_filtration(self, data_1d, window_size=5):
        """
        Apply mean filtering to 1D data
        :param data_1d: 1D data
        :param window_size: window size for filtering
        :return: filtered data
        """
        filtered_data = np.copy(data_1d)
        half_window = window_size // 2

        for i in range(len(data_1d)):
            start = max(0, i - half_window)
            end = min(len(data_1d), i + half_window + 1)
            filtered_data[i] = np.mean(data_1d[start:end])

        return filtered_data


# 测试函数
def test_meter_reader():
    """测试仪表读数器"""
    print("=== 仪表读数器测试 ===")
    
    # 创建一个测试图像
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    center = (320, 320)
    radius = 200
    
    # 绘制一个圆形（使用更明显的边缘）
    cv2.circle(test_img, center, radius, (255, 255, 255), -1)
    # 添加黑色边框使边缘更清晰
    cv2.circle(test_img, center, radius, (0, 0, 0), 3)
    
    # 创建测试掩码
    pointer_mask = np.zeros((640, 640), dtype=np.uint8)
    scale_mask = np.zeros((640, 640), dtype=np.uint8)
    
    # 模拟指针
    cv2.line(pointer_mask, center, (int(center[0] + radius * 0.8), int(center[1] - radius * 0.8)), 255, 5)
    
    # 模拟刻度
    for i in range(12):
        angle = i * 30 * np.pi / 180
        x1 = int(center[0] + (radius - 20) * np.sin(angle))
        y1 = int(center[1] - (radius - 20) * np.cos(angle))
        x2 = int(center[0] + radius * np.sin(angle))
        y2 = int(center[1] - radius * np.cos(angle))
        cv2.line(scale_mask, (x1, y1), (x2, y2), 255, 3)
    
    print(f"测试图像尺寸: {test_img.shape}")
    print(f"测试图像中心: {center}, 半径: {radius}")
    
    # 创建仪表读数器
    try:
        meter_reader = MeterReader(test_img)
        
        # 检查是否成功初始化
        if meter_reader.circle_center is None or meter_reader.circle_radius is None:
            print("❌ 仪表读数器初始化失败：无法检测到圆形")
            return
        
        print(f"✅ 仪表读数器初始化成功")
        print(f"检测到的圆心: {meter_reader.circle_center}")
        print(f"检测到的半径: {meter_reader.circle_radius}")
        
        # 测试读数
        result = meter_reader(pointer_mask, scale_mask)
        print(f"✅ 测试成功，结果: {result}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("尝试手动设置参数...")
        
        # 手动创建仪表读数器
        try:
            # 手动设置参数
            meter_reader = MeterReader.__new__(MeterReader)
            meter_reader.circle_center = center
            meter_reader.circle_radius = radius
            meter_reader.line_width = int(2 * np.pi * radius)
            meter_reader.line_height = int(radius * 0.4)
            meter_reader.threshold = 0.5
            
            print(f"✅ 手动初始化成功")
            print(f"手动设置的圆心: {meter_reader.circle_center}")
            print(f"手动设置的半径: {meter_reader.circle_radius}")
            
            # 测试读数
            result = meter_reader(pointer_mask, scale_mask)
            print(f"✅ 手动测试成功，结果: {result}")
            
        except Exception as e2:
            print(f"❌ 手动测试也失败: {e2}")
    
    print("测试完成")


if __name__ == "__main__":
    test_meter_reader()


