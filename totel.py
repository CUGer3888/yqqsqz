import sys

import pyautogui
import heapq
import time
from pynput import keyboard
from pynput.mouse import Controller
import pyautogui
from PIL import ImageGrab
import numpy as np
import cv2
import random
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


def move(inflection_points):
    speed_factor = 0.025
    for i in range(len(inflection_points)):
        current_x, current_y = inflection_points[i]
        if i + 1 < len(inflection_points):
            next_x, next_y = inflection_points[i + 1]
        else:
            next_x, next_y = goal_pont[0], goal_pont[1]
            pass
        dx = goal_pont[0] - centre_point[0]
        dy = goal_pont[1] - centre_point[1]

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


duandian = 1
a = 0
while (a == 0):
    try:
        img = screenshot()

        # 截图区域
        region = (1528, 120, 1758, 350)

        img_map = img[region[1]:region[3], region[0]:region[2]]
        # cv2.imshow("img_map", img_map)
        centre_point = (23, 23)
        width, height = 46, 46
        step_size = 5
        grid = 二值化(img_map, step_size, width, (128, 128, 128))
        # #中心点周围9格的值变为0
        for i in range(-1, 2):
            for j in range(-1, 2):
                grid[centre_point[0] + i][centre_point[1] + j] = 0

        # 移动16格需要3.44秒
        goal_pont = find_dentist_The_farthest_point(grid, centre_point)
        start = Node(centre_point[0], centre_point[1])
        goal = Node(goal_pont[0], goal_pont[1])
        path = A星(start, goal, grid)
        trun_point = find_Turning_point(path)
        print(trun_point)
        time.sleep(2)
        move(trun_point)
        duandian2 = trun_point[0][1]
        if duandian == duandian2:
            movements = ['w', 'd', 's', 'a']
            movement_index = [0, 1, 2, 3]
            for i in range(4):
                keyboard.press(movements[movement_index[i]])
                time.sleep(0.5 * movement_index[i])
                keyboard.release(movements[movement_index[i]])
        duandian = trun_point[0][1]

    # time.sleep(2)
    # keybord('d', 3.44)
    # cv2.waitKey(0)
    except:
        print("error")
        movements = ['w', 'd', 's', 'a']
        movement_index = [0, 1, 2, 3]
        for i in range(4):
            keyboard.press(movements[movement_index[i]])
            time.sleep(0.5 * movement_index[i])
            keyboard.release(movements[movement_index[i]])