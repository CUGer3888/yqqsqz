import random
import sys
import matplotlib.pyplot as plt
import pyautogui
import heapq
import time
from pynput import keyboard
from pynput.mouse import Controller
import pyautogui
from PIL import ImageGrab
import numpy as np
import cv2

import threading

keyboard = keyboard.Controller()
mouse = Controller()


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.f < other.f


def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def A星(start, goal, grid):
    open_list = []
    closed_list = []

    heapq.heappush(open_list, start)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node == goal:
            path = []
            while current_node is not None:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_list.append(current_node)

        for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = Node(current_node.x + i, current_node.y + j, current_node)

            if neighbor.x < 0 or neighbor.x >= len(grid) or neighbor.y < 0 or neighbor.y >= len(grid[0]):
                continue

            if grid[neighbor.x][neighbor.y] == 1:
                continue

            if neighbor in closed_list:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, goal)
            neighbor.f = neighbor.g + neighbor.h

            if any(node for node in open_list if node == neighbor and node.g <= neighbor.g):
                continue

            heapq.heappush(open_list, neighbor)

    return None

    # 测试


# 屏幕截图
def screenshot():
    im = pyautogui.screenshot()
    # 将PIL图像转换为OpenCV图像
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    return im


def find_image_on_screen(image_path, threshold=0.8):
    img = screenshot()
    template = cv2.imread(image_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return [(pt[0] + w // 2, pt[1] + h // 2) for pt in zip(*loc[::-1])]


def mouse_move(x, y):
    pyautogui.moveTo(x, y, duration=0.2)
    # 鼠标点击
    pyautogui.click()


def keybord(key, move_time):
    keyboard.press(key)
    time.sleep(move_time)
    keyboard.release(key)


def 二值化(img_map, step_size, width, target_color):
    # 转换为灰度图像
    gray = cv2.cvtColor(img_map, cv2.COLOR_BGR2GRAY)

    # 初始化二值化映射
    thresh_map = np.zeros((width, width), dtype=np.int8)

    # 计算目标颜色的灰度值
    target_gray = np.array(target_color)

    for i in range(0, gray.shape[0], step_size):
        for j in range(0, gray.shape[1], step_size):
            # 提取局部窗口
            window = gray[i:i + step_size, j:j + step_size]

            # 检查窗口中的每个像素是否与目标颜色匹配
            match_count = np.sum(window == target_gray[0].flatten())

            # 如果超过一半的像素匹配目标颜色，则标记为0，否则为1
            thresh_map[i // step_size, j // step_size] = 0 if match_count > (step_size * step_size) / 2 else 1

    # 转换为列表
    grid = thresh_map.tolist()

    return grid


def 三值化(img, step_size, width, target_color_1_range, target_color_2_range):
    # 确保输入是NumPy数组
    img = np.array(img, dtype=np.uint8)

    # 初始化结果数组
    result = np.zeros((width, width), dtype=np.int8)

    # 将目标颜色范围转换为NumPy数组
    target_color_1_min = np.array(target_color_1_range[0], dtype=np.uint8)
    target_color_1_max = np.array(target_color_1_range[1], dtype=np.uint8)
    target_color_2_min = np.array(target_color_2_range[0], dtype=np.uint8)
    target_color_2_max = np.array(target_color_2_range[1], dtype=np.uint8)

    # 遍历图像的每个像素
    for i in range(0, img.shape[0], step_size):
        for j in range(0, img.shape[1], step_size):
            # 提取当前像素
            pixel = img[i, j]

            # 检查像素是否落在目标颜色0的范围内
            if np.all(pixel >= target_color_1_min) and np.all(pixel <= target_color_1_max):
                result[i // step_size, j // step_size] = 0
            # 检查像素是否落在目标颜色1的范围内
            elif np.all(pixel >= target_color_2_min) and np.all(pixel <= target_color_2_max):
                result[i // step_size, j // step_size] = 1
            # 如果像素不属于任何目标颜色，则标记为0
            else:
                result[i // step_size, j // step_size] = 2

    return result.tolist()


def find_exit(binary_map):
    # 将输入转换为NumPy数组，便于处理
    binary_map = np.array(binary_map)

    # 定义地图的宽度和高度
    height, width = binary_map.shape

    # 遍历地图中的每个点
    for x in range(height):
        for y in range(width):
            # 检查当前点是否为0
            if binary_map[x, y] == 0:
                # 检查当前点的相邻点
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    # 检查相邻点是否位于地图内且值为2
                    if 0 <= nx < height and 0 <= ny < width and binary_map[nx, ny] == 2:
                        # 找到与值2相邻的值0点，返回坐标
                        return (x, y)

    # 如果没有找到出口，返回None
    return None


def find_exit_near(binary_map, centre_point):
    # 将输入转换为NumPy数组，便于处理
    binary_map = np.array(binary_map)

    # 定义地图的宽度和高度
    height, width = binary_map.shape

    # 遍历地图中的每个点
    for x in range(height):
        for y in range(width):
            # 检查当前点是否为0
            if binary_map[x, y] == 0:
                # 检查当前点的相邻点
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    # 检查相邻点是否位于地图内且值为2
                    if 0 <= nx < height and 0 <= ny < width and binary_map[nx, ny] == 2:
                        # 找到与值2相邻的值0点，返回坐标
                        if (abs(x - centre_point[0])) + abs((y - centre_point[1])) < 50:
                            return (x, y)

    # 如果没有找到出口，返回None
    return None


def find_Turning_point(points):
    inflection_points = []
    for i in range(1, len(points) - 1):
        x, y = points[i - 1]
        xx, yy = points[i]
        xxx, yyy = points[i + 1]
        xxxx = (x + xxx) / 2
        yyyy = (y + yyy) / 2
        if xx != xxxx:
            inflection_points.append(points[i])
    return inflection_points


def heuristic_findchildnodes(x, y, isvisit):
    distance = abs(x - centre_point[0]) + abs(y - centre_point[1])
    isvisit[x][y] = distance


def display_binary_map(binary_map):
    # 确保传入的参数是一个NumPy数组
    binary_map = np.array(binary_map)

    # 使用matplotlib显示图像
    plt.imshow(binary_map, cmap='gray', interpolation='nearest')

    # 关闭坐标轴显示
    plt.axis('off')

    # 显示图像
    plt.show()


def find_dentist_The_farthest_point(grid, centre_point):
    isvisit = [[0 for _ in range(width)] for _ in range(width)]
    for i in range(width):
        for j in range(width):
            if grid[i][j] == 0:
                heuristic_findchildnodes(i, j, isvisit)
    # 找出 isvisit 列表中的最大值及其索引
    max_value = float('-inf')
    max_indices = None

    for i in range(width):
        for j in range(width):
            if isvisit[i][j] > max_value:
                max_value = isvisit[i][j]
                max_indices = (i, j)
    print("最大值:", max_value)
    print("最大值的索引:", max_indices)
    return max_indices


def turn_near_to_0(centre_point, grid, proportion_x, proportion_y):
    for i in range(0, int(len(grid) * proportion_y)):
        for j in range(0, int(len(grid) * proportion_x)):
            try:
                grid[centre_point[1] + i][centre_point[0] + j] = 1
            except IndexError:
                pass
    return grid


def turn_near_to_1(centre_point, grid, proportion_x, proportion_y):
    for i in range(0, int(len(grid) * proportion_y)):
        for j in range(0, int(len(grid) * proportion_x)):
            grid[centre_point[1] + i][centre_point[0] + j] = 2
    return grid


def turn_near_to_0_goal(centre_point, grid, proportion_x, proportion_y):
    for i in range(0, int(len(grid) * proportion_y)):
        for j in range(0, int(len(grid) * proportion_x)):
            grid[centre_point[1] + i][centre_point[0] + j] = 0
    return grid
def 移动(inflection_points):
    speed_factor = 0.07
    for i in range(len(inflection_points)):
        current_x, current_y = inflection_points[i]
        if i + 1 < len(inflection_points):
            next_x, next_y = inflection_points[i + 1]
        else:
            break

        dx = next_x - current_x
        dy = next_y - current_y

        if dx >= 0:
            keyboard.press('s')
            time.sleep(abs(dx) * speed_factor)
            keyboard.release('s')
        elif dx < 0:
            keyboard.press('w')
            time.sleep(abs(dx) * speed_factor)
            keyboard.release('w')

        if dy >= 0:
            keyboard.press('d')
            time.sleep(abs(dy) * speed_factor)
            keyboard.release('d')
        elif dy < 0:
            keyboard.press('a')
            time.sleep(abs(dy) * speed_factor)
            keyboard.release('a')
        time.sleep(0.1)

    # pyautogui.moveTo(100, 100)


def match_template(screen_gray, template_path, threshold=0.75):
    template_pic = cv2.imread(template_path, 0)
    if template_pic is None:
        raise ValueError(f"Error: Template image '{template_path}' not loaded properly.")
    result = cv2.matchTemplate(screen_gray, template_pic, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    return [(pt[0], pt[1]) for pt in zip(*loc[::-1])]


def match_templates_and_store(region):
    # 获取屏幕截图并转换为灰度图像
    screen = ImageGrab.grab(bbox=region)
    screen_np = np.array(screen)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

    # 定义模板图片路径和匹配阈值
    templates = {
        'gantanhao': 'pic/gantanhao.png',
        'chuansong': 'pic/chuansong.png',
        'renwu': 'pic/renwu.png'
    }
    threshold = 0.8

    matches = {}
    for name, template_path in templates.items():
        matches[name] = match_template(screen_gray, template_path, threshold)
        time.sleep(1)  # 休眠1秒，避免过于频繁的操作
    return matches
def turn_to_0(grid, screen_gray):
    dict = {



    }

    all_positions = []
    for _, template_path in diaoxiang_dict.items():
        template = cv2.imread(template_path, 0)
        positions_diaoxiang, _ = find_template_position(screen_gray, template, 0.88)
        all_positions.extend(positions_diaoxiang)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 3 for other in unique_positions):
            unique_positions.append(pos)

    # 调用turn_near_to_0函数，这里假设该函数已定义好
    for i in unique_positions:
        grid = turn_near_to_0((i[0] // step_size - 4, i[1] // step_size - 2), grid, 0.1, 0.1)

    return grid









def turn_diaoxiang_to_0(grid, screen_gray):
    diaoxiang_dict = {
        "1": "pic/diaoxiang1.png",
        "2": "pic/diaoxiang2.png",
        "3": "pic/diaoxiang3.png",
        "4": "pic/diaoxiang4.png",
        "5": "pic/diaoxiang5.png",
        "6": "pic/diaoxiang6.png",
        "7": "pic/diaoxiang7.png",
        "8": "pic/diaoxiang8.png",
    }
    all_positions = []
    for _, template_path in diaoxiang_dict.items():
        template = cv2.imread(template_path, 0)
        positions_diaoxiang, _ = find_template_position(screen_gray, template, 0.88)
        all_positions.extend(positions_diaoxiang)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 20 for other in unique_positions):
            unique_positions.append(pos)

    # 调用turn_near_to_0函数，这里假设该函数已定义好
    for i in unique_positions:
        grid = turn_near_to_0((i[0] // step_size - 4, i[1] // step_size - 2), grid, 0.1, 0.1)

    return grid

def turn_renwu_to_0(grid, screen_gray):
    renwu_dict ={
        "9": "pic/renwu1.png",
        "10": "pic/renwu2.png",
        "11": "pic/renwu3.png",
        "12": "pic/renwu4.png",
        "13": "pic/renwu5.png"
    }

    all_positions = []
    for _, template_path in renwu_dict.items():
        template = cv2.imread(template_path, 0)
        positions_renwu, _ = find_template_position(screen_gray, template, 0.9)
        all_positions.extend(positions_renwu)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 20 for other in unique_positions):
            unique_positions.append(pos)
    for i in unique_positions:
        grid = turn_near_to_1_renwu((i[0] // step_size - 28, i[1] // step_size - 28),
                                    grid, 0.2, 0.2)
    return grid


def turn_near_to_1_renwu(centre_point, grid, proportion_x, proportion_y):
    xxx, yyy = len(grid) // 2 - 1, len(grid) // 2 - 1
    for i in range(0, int(len(grid) * proportion_y) + 10):
        for j in range(0, int(len(grid) * proportion_x) + 10):
            if (abs(centre_point[0] + i - xxx) + abs(centre_point[1] + j - yyy)) < 40:
                grid[centre_point[1] + i + 2][centre_point[0] + j - 1] = 1
    return grid


def turn_near_to_0_renwu(centre_point, grid, proportion_x, proportion_y):
    xxx, yyy = len(grid) // 2 - 1, len(grid) // 2 - 1
    for i in range(0, int(len(grid) * proportion_y) + 10):
        for j in range(0, int(len(grid) * proportion_x) + 10):
            if (abs(centre_point[0] + i - xxx) + abs(centre_point[1] + j - yyy)) < 15:
                grid[centre_point[1] + i + 2][centre_point[0] + j - 1] = 0
    return grid


def turn_chushen_to_0(grid, screen_gray):
    rukou_dict = {
        "14": "pic/rukou1.png",
        "15": "pic/rukou2.png",
        "16": "pic/rukou3.png",
        "17": "pic/rukou4.png",
        "18": "pic/rukou5.png",
        "19": "pic/rukou6.png",
        "20": "pic/rukou7.png",
        "21": "pic/rukou8.png",
        "22": "pic/rukou9.png",
        "23": "pic/rukou10.png",
        "24": "pic/rukou11.png",
        "25": "pic/rukou12.png",
        "26": "pic/rukou13.png",
    }
    all_positions = []
    for _, template_path in rukou_dict.items():
        template = cv2.imread(template_path, 0)
        positions_rukou, _ = find_template_position(screen_gray, template, 0.9)
        all_positions.extend(positions_rukou)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 20 for other in unique_positions):
            unique_positions.append(pos)

    # 调用turn_near_to_0函数，这里假设该函数已定义好
    for i in unique_positions:
        grid = turn_near_to_0((i[0] // step_size, i[1] // step_size), grid, 0.09, 0.09)

    return grid


def turn_baoxiang_to_0(grid, screen_gray):
    baoxiang_dict = {
        "27": "pic/baoxiang1.png",
        "28": "pic/baoxiang2.png",
        "29": "pic/baoxiang3.png",
        "30": "pic/baoxiang4.png",
        "31": "pic/baoxiang5.png",
        "32": "pic/baoxiang6.png"
    }
    all_positions = []
    for _, template_path in baoxiang_dict.items():
        template = cv2.imread(template_path, 0)
        positions_baoxiang, _ = find_template_position(screen_gray, template, 0.9)
        all_positions.extend(positions_baoxiang)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 20 for other in unique_positions):
            unique_positions.append(pos)

    # 调用turn_near_to_0函数，这里假设该函数已定义好
    for i in unique_positions:
        grid = turn_near_to_0((i[0] // step_size - 4, i[1] // step_size - 6), grid, 0.1, 0.1)

    return grid

def 找终点(screen_array_bgr):
    screen_array_hsv = cv2.cvtColor(screen_array_bgr, cv2.COLOR_RGB2HSV)
    # cv2.imshow("HSV", screen_array_hsv)
    # cv2.waitKey(0)
    lower_blue = np.array([15, 208, 215])
    upper_blue = np.array([20, 223, 220])

    # 使用inRange函数找到颜色范围内的像素
    mask = cv2.inRange(screen_array_hsv, lower_blue, upper_blue)

    # 找到颜色范围内的像素坐标
    coordinates = np.column_stack(np.where(mask == 255))

    # 检查是否找到了匹配的颜色
    if coordinates.size == 0:
        return (0, 0)

    # 如果找到了匹配的颜色，计算坐标
    front_three_rows_mean = np.mean(coordinates[:10], axis=0)
    zhongdian_x = int(front_three_rows_mean[0] / step_size)
    zhongdian_y = int(front_three_rows_mean[1] / step_size)
    print(zhongdian_y,zhongdian_x)
    return  zhongdian_y,zhongdian_x
def turn_other_to_0(grid, screen_gray):
    other_dict = {

    }
    all_positions = []
    for _, template_path in other_dict.items():
        template = cv2.imread(template_path, 0)
        positions_other, _ = find_template_position(screen_gray, template, 0.9)
        all_positions.extend(positions_other)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 20 for other in unique_positions):
            unique_positions.append(pos)

    # 调用turn_near_to_0函数，这里假设该函数已定义好
    for i in unique_positions:
        grid = turn_near_to_0((i[0] // step_size - 2, i[1] // step_size - 1), grid, 0.08, 0.08)

    return grid


def turn_fruit_to_0(grid, screen_gray):
    fruit_dict = {
        "1": "pic/other1.png",
        "2": "pic/other2.png",
        "3": "pic/other3.png",
        "4": "pic/other4.png",
        "5": "pic/other5.png",
        "6": "pic/other6.png",
        "7": "pic/other7.png",
        "8": "pic/other8.png",
        "9": "pic/other9.png",
        "1": "pic/fruit1.png",
        "2": "pic/fruit2.png",
        "3": "pic/fruit3.png",
        "4": "pic/fruit4.png",
        "5": "pic/fruit5.png",
        "6": "pic/fruit6.png",
        "7": "pic/fruit7.png",
        "8": "pic/fruit8.png",
        "9": "pic/fruit9.png",
        "10": "pic/fruit10.png",
        "11": "pic/fruit11.png"
    }
    all_positions = []
    for _, template_path in fruit_dict.items():
        template = cv2.imread(template_path, 0)
        positions_fruit, _ = find_template_position(screen_gray, template, 0.9)
        all_positions.extend(positions_fruit)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 20 for other in unique_positions):
            unique_positions.append(pos)

    # 调用turn_near_to_0函数，这里假设该函数已定义好
    for i in unique_positions:
        grid = turn_near_to_0((i[0] // step_size - 1, i[1] // step_size - 5), grid, 0.09, 0.08)

    return grid


def turn_chukou_to_0(grid, screen_gray):
    chukou_dict = {
        "1": "pic/chukou1.png",
        "2": "pic/chukou2.png",
        "3": "pic/chukou3.png",
        "4": "pic/chukou4.png",
        "5": "pic/chukou5.png",
        "6": "pic/chukou6.png",
        "7": "pic/chukou7.png",
        "8": "pic/chukou8.png",
        "9": "pic/chukou9.png",
        "10": "pic/chukou10.png",
    }
    all_positions = []
    for _, template_path in chukou_dict.items():
        template = cv2.imread(template_path, 0)
        positions_chukou, _ = find_template_position(screen_gray, template, 0.95)
        all_positions.extend(positions_chukou)

    # 去除重复和过于接近的匹配点
    unique_positions = []
    for pos in all_positions:
        if not any(np.linalg.norm(np.array(pos) - np.array(other)) < 20 for other in unique_positions):
            unique_positions.append(pos)

    # 调用turn_near_to_0函数，这里假设该函数已定义好
    for i in unique_positions:
        grid = turn_near_to_0((i[0] // step_size, i[1] // step_size), grid, 0.09, 0.09)

    return grid

def find_template_position(screen_gray, template, threshold):
    res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    positions = []
    for pt in zip(*loc[::-1]):  # 正确使用zip(*loc[::-1])来获取坐标
        positions.append((pt[0], pt[1]))  # 将坐标添加到列表中
    count = len(positions)  # 计算匹配位置的数量
    return positions, count
aaa = 1
while True:
    # try:
    img = screenshot()
    region = (1493, 87, 1793, 387)
    img_map = img[region[1]:region[3], region[0]:region[2]]
    screen_gray = cv2.cvtColor(img_map, cv2.COLOR_BGR2GRAY)
    centre_point = (150, 150)
    width, hight = 300, 300

    step_size = 1
    target_color1_range = ((125, 125, 125), (130, 130, 130))
    target_color2_range = ((130, 130, 130), (255, 255, 255))
    grid = 三值化(img_map, step_size, width, target_color1_range, target_color2_range)
    grid = turn_diaoxiang_to_0(grid, screen_gray)
    grid = turn_renwu_to_0(grid, screen_gray)
    # grid = turn_chushen_to_0(grid, screen_gray)
    grid = turn_baoxiang_to_0(grid, screen_gray)
    gird = turn_other_to_0(grid, screen_gray)
    grid = turn_fruit_to_0(grid, screen_gray)
    goal = find_exit_near(grid,centre_point)
    if goal==None:
        goal = find_exit(grid)
    goal_point = (goal[1], goal[0])
    # x, y = 找终点(img_map)
    # if x == 0:
    #     pass
    # else:
    #     goal_point = (x, y)
    grid = 二值化(img_map, step_size, width, (128, 128, 128))
    grid = turn_near_to_0_renwu((centre_point[0] - 17, centre_point[1] - 17), grid, 0.07, 0.07)
    grid = turn_near_to_0_goal((goal_point[0] - 10, goal_point[1] - 10), grid, 0.07, 0.07)
    # display_binary_map(grid)
    start = Node(centre_point[0], centre_point[1])
    goal = Node(goal_point[1], goal_point[0])
    path = A星(start, goal, grid)
    turn_point = find_Turning_point(path)
    移动(turn_point)
    aaa +=1
    if (aaa%5==0):
        movements = ['w', 'd', 's', 'a']
        movement_index = [0, 1, 2, 3]
        for i in range(4):
            keyboard.press(movements[movement_index[i]])
            time_move = random.randint(1, 4)
            time.sleep(0.8*time_move)
            keyboard.release(movements[movement_index[i]])
            aaa = 1
    # except:
    #     print("程序出错")
    #     time.sleep(0.1)
