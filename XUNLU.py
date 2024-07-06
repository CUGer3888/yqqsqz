#
#
# """
# 识别场景变化，然后识别遥感，再去点击地图获得坐标，然后识别房间类型
# """
# import time
# class Node:
#     def __init__(self, x, y, parent=None):
#         self.x = x
#         self.y = y
#         self.parent = parent
#         self.g = 0
#         self.h = 0
#         self.f = 0
#
#     def __eq__(self, other):
#         return self.x == other.x and self.y == other.y
#
#     def __lt__(self, other):
#         return self.f < other.f
# def heuristic(node, goal):
#     return abs(node.x - goal.x) + abs(node.y - goal.y)
# def A星(start, goal, grid):
#     open_list = []
#     closed_list = []
#
#     heapq.heappush(open_list, start)
#
#     while open_list:
#         current_node = heapq.heappop(open_list)
#
#         if current_node == goal:
#             path = []
#             while current_node is not None:
#                 path.append((current_node.x, current_node.y))
#                 current_node = current_node.parent
#             return path[::-1]
#
#         closed_list.append(current_node)
#
#         for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             neighbor = Node(current_node.x + i, current_node.y + j, current_node)
#
#             if neighbor.x < 0 or neighbor.x >= len(grid) or neighbor.y < 0 or neighbor.y >= len(grid[0]):
#                 continue
#
#             if grid[neighbor.x][neighbor.y] == 1:
#                 continue
#
#             if neighbor in closed_list:
#                 continue
#
#             neighbor.g = current_node.g + 1
#             neighbor.h = heuristic(neighbor, goal)
#             neighbor.f = neighbor.g + neighbor.h
#
#             if any(node for node in open_list if node == neighbor and node.g <= neighbor.g):
#                 continue
#
#             heapq.heappush(open_list, neighbor)
#
#     return None
#
# def get_rigion():
#     app_title = "MuMu模拟器12"
#     hwnd = win32gui.FindWindow(None, app_title)
#     left, top, right, bottom = win32gui.GetWindowRect(hwnd)
#     width = right - left
#     height = bottom - top
#     print(width, height)
#     return left, top, width, height
#
#
# def get_screen():
#     region = (get_rigion())
#     partial_screen = ImageGrab.grab(region)
#     screen_array = np.array(partial_screen)
#     screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
#     return screen_array_bgr
#
# def yaogan():
#     get_screen()
#     a = True
#     while a:
#         time.sleep(0.1)
#         yaogan_path = "摇杆.png"
#         yaogan_pic = cv2.imread(yaogan_path, 0)
#         result = cv2.matchTemplate(screen_array_bgr, yaogan_pic, cv2.TM_CCOEFF_NORMED)
#         # 设置阈值
#         threshold = 0.85
#         loc = np.where(result >= threshold)
#         if len(loc) > 0:
#             a = False
#         else:
#             a = True
#
# def Identifying_Scenarios():
#     pyautogui.moveTo(left + 0.58 * width, top + 0.18 * height, duration=0.4)
#     pyautogui.click()
#     region = (get_rigion())
#     partial_screen = ImageGrab.grab(bbox=region)
#     screen_array_1 = np.array(partial_screen)
#     screen_array_bgr_1 = cv2.cvtColor(screen_array_1, cv2.COLOR_RGB2BGR)
#     screen_array_hsv = cv2.cvtColor(screen_array_bgr_1, cv2.COLOR_RGB2HSV)
#     lower_橙色 = np.array([98, 173, 241])
#     upper_橙色 = np.array([113, 194, 255])
#     mask = cv2.inRange(screen_array_hsv, lower_橙色, upper_橙色)
#     coordinates = np.column_stack(np.where(mask == 255))
#     time.sleep(0.01)
#     front_ten_rows_mean = np.mean(coordinates[:10], axis=0)
#     y = front_ten_rows_mean[0]
#     x = front_ten_rows_mean[1]
#     print(x, y)
#     if 120 < x < 130:
#         print(2)
#         # return "特殊房"
#     elif 270 < x:
#         print(3)
#         # return "boss房"
#     else:
#         # return "普通房"
#         print(1)
# def 特殊房():
#     # 屏幕截图()
#     keyboard.press("d")
#     time.sleep(5)
#     keyboard.release("d")
#     keyboard.press("f")
#     keyboard.release('f')
#     keyboard.press("d")
#     time.sleep(1)
#     keyboard.release("d")
#     keyboard.press("f")
#     keyboard.release('f')
# def boos房():
#     a = True
#     b = True
#     time.sleep(1)
#     while a:
#         boss_screen = get_screen()
#         boss_path = "boss.png"
#         boss_pic = cv2.imread(boss_path, 0)
#         boss_result = cv2.matchTemplate(boss_screen, boss_pic, cv2.TM_CCOEFF_NORMED)
#         boss_threshold = 0.85
#         boss_loc = np.where(boos_result >= boss_threshold)
#         if len(boss_loc) > 0:
#             a = False
#         else:
#             a = True
#     if a:
#         for pt in zip(*loc[::-1]):
#             print(f"Match found at position: {boss_pt}")
#
#     while b:
#
#         renwu_path = "renwu.png"
#         renwu_pic = cv2.imread(renwu_path, 0)
#         renwu_result = cv2.matchTemplate(boss_screen, renwu_pic, cv2.TM_CCOEFF_NORMED)
#         renwu_threshold = 0.85
#         renwu_loc = np.where(renwu_result >= renwu_threshold)
#         if len(renwu_loc) > 0:
#             b = False
#         else:
#             b = True
#     if b:
#         for pt in zip(*loc[::-1]):
#             print(f"Match found at position: {renwu_pt}")
#     xx, yy = boss_pt[0]
#     xxx, yyy = renwu_pt[0]
#     go_x, go_y = xx - xxx, yy - yyy
#     if go_x > 0 and go_y > 0:
#         keyboard.press("d")
#         time.sleep(abs(go_x / 10))
#         keyboard.release("d")
#         keyboard.press("s")
#         time.sleep(abs(go_y / 10))
#         keyboard.release("s")
#     elif go_x < 0 and go_y < 0:
#         keyboard.press("a")
#         time.sleep(abs(go_x / 10))
#         keyboard.release("a")
#         keyboard.press("w")
#         time.sleep(abs(go_y / 10))
#         keyboard.release("w")
#     elif go_x > 0 and go_y < 0:
#         keyboard.press("d")
#         time.sleep(abs(go_x / 10))
#         keyboard.release("d")
#         keyboard.press("w")
#         time.sleep(abs(go_y / 10))
#         keyboard.release("w")
#     elif go_x < 0 and go_y > 0:
#         keyboard.press("a")
#         time.sleep(abs(go_x / 10))
#         keyboard.release("a")
#         keyboard.press("s")
#         time.sleep(abs(go_y / 10))
#         keyboard.release("s")
#     else:
#         pass  # 随机移动
# def 普通房():
#     left_2, top_2, width_2, height_2 = get_rigion()
#     region = (left_2 + 0.7 * width_2, top_2 + height_2 * 0.1,
#               left_2 + 0.8 * width_2, top_2 + height_2 * 0.2)
#     step_size = 5
#     中心点 = (int(0.1* width_2), int(0.1* height_2))
#     grid = 二值化地图()
#     子目标点 = 找子节点(grid)
#     start = Node(中心点[0], 中心点[1])
#     goal = Node(子目标点[0], 子目标点[1])
#     time.sleep(3)
#     path = A星(start, goal, grid)
#
# def 二值化():
#     new_width = new_height = 46
#     step_size = 5
#     startx = int(new_width / 2)
#     starty = startx
#     left_1, top_1, width_1, height_1 = get_rigion()
#     region = (left_1+0.7*width_1, top_1+height_1*0.1,
#               left_1+0.8*width_1, top_1+height_1*0.2)
#     partial_screen = ImageGrab.grab(region)
#     screen_array = np.array(partial_screen)
#     screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
#     gray_screen = cv2.cvtColor(screen_array_bgr, cv2.COLOR_BGR2GRAY)
#     _, binary_screen = cv2.threshold(gray_screen, 127, 255, cv2.THRESH_BINARY)
#     binary_map = np.zeros((new_width, new_height), dtype=np.int8)
#
#     for i in range(0, binary_screen.shape[0], step_size):
#         for j in range(0, binary_screen.shape[1], step_size):
#             window = binary_screen[i:i + step_size, j:j + step_size]
#             black_pixels = np.sum(window == 0)
#             binary_map[i // step_size, j // step_size] = 1 if black_pixels > (step_size * step_size) / 2 else 0
#     grid = binary_map.tolist()
#     return grid
# def 找终点():
#     screen_array_bgr_3 = get_screen()
#     time.sleep(0.1)
#     goal_path = "goal.png"
#     goal_pic = cv2.imread(goal_path, 0)
#     goal_result = cv2.matchTemplate(screen_array_bgr_3, goal_pic, cv2.TM_CCOEFF_NORMED)
#     # 设置阈值
#     a_1= False
#     threshold = 0.85
#     loc = np.where(goal_result >= threshold)
#     if len(loc) > 0:
#         a = True
#     else:
#         a = False
#     if a:
#         for pt in zip(*loc[::-1]):
#             print(f"Match found at position: {pt}")
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
def 检测空间场景变化():
    now_color11 = pyautogui.pixel(287, 158)
    now_color12 = pyautogui.pixel(653, 165)
    now_color13 = pyautogui.pixel(292, 350)
    now_color14 = pyautogui.pixel(617, 346)
    if now_color11 == (18, 18, 24) and now_color12 == (18, 18, 24) and now_color13 == (18, 18, 24) and now_color14 == (
    18, 18, 24):
        print(1)
        time.sleep(3)
        检测房间类型()
    else:
        print("no")
def 检测房间类型():
    mouse.position = (791,115)
    mouse.click(Button.left, 1)
    time.sleep(0.5)
    region = (325, 108, 633, 154)
    partial_screen = ImageGrab.grab(bbox=region)
    screen_array = np.array(partial_screen)
    screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
    screen_array_hsv = cv2.cvtColor(screen_array_bgr, cv2.COLOR_RGB2HSV)
    lower_橙色 = np.array([98, 173, 241])
    upper_橙色 = np.array([113, 194, 255])
    mask = cv2.inRange(screen_array_hsv, lower_橙色, upper_橙色)
    coordinates = np.column_stack(np.where(mask == 255))
    time.sleep(0.01)
    front_ten_rows_mean = np.mean(coordinates[:10], axis=0)
    y = front_ten_rows_mean[0]
    x = front_ten_rows_mean[1]
    print(x, y)
    if 120<x<130:
        print(2)
        # return "特殊房"
    elif 270<x:
        print(3)
        # return "boss房"
    else:
        # return "普通房"
        print(1)
def heuristic_findchildnodes(x, y, isvisit):
    distance = abs(x - startx) + abs(y - starty)
    isvisit[x][y] = distance
def 找子节点(lit):
    isvisit = [[0 for _ in range(new_width)] for _ in range(new_height)]
    for i in range(new_width):
        for j in range(new_height):
            if lit[i][j] == 0:
                heuristic_findchildnodes(i, j, isvisit)
    # 找出 isvisit 列表中的最大值及其索引
    max_value = float('-inf')
    max_indices = None

    for i in range(new_width):
        for j in range(new_height):
            if isvisit[i][j] > max_value:
                max_value = isvisit[i][j]
                max_indices = (i, j)
    print("最大值:", max_value)
    print("最大值的索引:", max_indices)
    return max_indices
        # child_goal = Node(max_indices)
        # # path = astar(start, child_goal, grid)
        # # print(path)
        # # move(path)
def 二值化地图():
    global new_width,new_height,step_size,startx,starty
    new_width = new_height = 46
    step_size = 5
    startx = int(new_width/2)
    starty = startx
    partial_screen = ImageGrab.grab(region)
    screen_array = np.array(partial_screen)
    screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
    gray_screen = cv2.cvtColor(screen_array_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_screen = cv2.threshold(gray_screen, 127, 255, cv2.THRESH_BINARY)
    binary_map = np.zeros((new_width, new_height), dtype=np.int8)

    for i in range(0, binary_screen.shape[0], step_size):
        for j in range(0, binary_screen.shape[1], step_size):
            window = binary_screen[i:i + step_size, j:j + step_size]
            black_pixels = np.sum(window == 0)
            binary_map[i // step_size, j // step_size] = 1 if black_pixels > (step_size * step_size) / 2 else 0
    grid = binary_map.tolist()
    # print(grid[14][14])
    # cv2.imshow("screen_array_bgr", screen_array_bgr)
    # cv2.imshow('Binary Screen', binary_screen)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("screen_array_bgr", screen_array_bgr)
    return grid

    # print(grid)
    # path = astar(start, goal, grid)
    # inflection_points = find_inflection_points(path)
    #
    #
    # print(path)
    # print("拐点:", inflection_points)
    # for i in range(len(inflection_points)):
    #     current_x, current_y = inflection_points[i]
    #     if i + 1 < len(inflection_points):
    #         next_x, next_y = inflection_points[i + 1]
    #     else:
    #         # 如果是最后一个拐点，则下一个点是目标点
    #         # next_x, next_y = endx, endy
    #         pass
    #     移动(current_x, current_y, next_x, next_y)
def 寻找拐点(points):
    inflection_points = []
    for i in range(1, len(points)-1):
        x,y = points[i - 1]
        xx,yy = points[i]
        xxx,yyy = points[i + 1]
        xxxx =(x+xxx)/2
        yyyy = (y+yyy)/2
        if xx!=xxxx:
            inflection_points.append(points[i])
    return inflection_points
def 特殊房():
    # 屏幕截图()
    keyboard.press("d")
    time.sleep(5)
    keyboard.release("d")
    keyboard.press("f")
    keyboard.release('f')
    keyboard.press("d")
    time.sleep(1)
    keyboard.release("d")
    keyboard.press("f")
    keyboard.release('f')
def boos房():
    speed_factor = 0.4
    keyboard.press("w")
    time.sleep(3)
    keyboard.release("w")
    list_技能 = ['q', '2', '4', 'f']
    for i in range(4):
        j = random.randint(0, 3)
        keyboard.press(list_技能[j])
        time.sleep(abs(dx) * speed_factor)
        keyboard.release(list_技能[j])
def 找终点():
    partial_screen = ImageGrab.grab(bbox=region)
    screen_array = np.array(partial_screen)
    screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
    screen_array_hsv = cv2.cvtColor(screen_array_bgr, cv2.COLOR_RGB2HSV)

    lower_blue =np.array([15,208,215])
    upper_blue =np.array([20,223,220])

    # 使用inRange函数找到颜色范围内的像素
    mask = cv2.inRange(screen_array_hsv, lower_blue, upper_blue)

    # 找到颜色范围内的像素坐标
    coordinates = np.column_stack(np.where(mask == 255))
    time.sleep(0.01)
    front_three_rows_mean = np.mean(coordinates[:10], axis=0)
    zhongdian_x = int(front_three_rows_mean[0] / step_size)
    zhongdian_y = int(front_three_rows_mean[1] / step_size)

    # print(grid)

    # 正方形的边长或半径
    radius = 2  # 假设正方形的边长为4，则半径为2

    # 计算正方形区域的边界
    min_row = max(0, zhongdian_x - radius)
    max_row = min(len(grid), zhongdian_x + radius + 1)
    min_col = max(0, zhongdian_y - radius)
    max_col = min(len(grid), zhongdian_y + radius + 1)

    # 遍历正方形区域内的所有点，并将它们的值设为1
    for row in range(min_row, max_row):
        for col in range(min_col, max_col):
            grid[row][col] = 0
    # print(grid)

    goal = Node(zhongdian_x, zhongdian_y)
    path = A星(start, goal, grid)
    拐点列表 = 寻找拐点(path)
    time.sleep(3)
    移动到终点(拐点列表,zhongdian_x,zhongdian_y)
def 移动到终点(inflection_points,zhongdian_x,zhongdian_y):
    speed_factor_1 = 0.04
    for i in range(len(inflection_points)):
        current_x_1, current_y_1 = inflection_points[i]
        if i + 1 < len(inflection_points):
            next_x_1, next_y_1 = inflection_points[i + 1]
        else:
            next_x_1, next_y_1 = zhongdian_x, zhongdian_y
            pass
        dx_1 = zhongdian_x - 中心点[0]
        dy_1 = zhongdian_y- 中心点[1]

        if dx_1 >= 0:
            keyboard.press('s')
            time.sleep(abs(dx) * speed_factor)
            keyboard.release('s')
        elif dx_1 < 0:
            keyboard.press('w')
            time.sleep(abs(dx) * speed_factor)
            keyboard.release('w')

        if dy_1 >= 0:
            keyboard.press('d')
            time.sleep(abs(dy) * speed_factor)
            keyboard.release('d')
        elif dy_1 < 0:
            keyboard.press('a')
            time.sleep(abs(dy) * speed_factor)
            keyboard.release('a')
    keyboard.press('f')
    exit()
def 卡墙():
    # 定义要检查的屏幕位置
    screen_positions = [(300, 400), (400, 400), (500, 400), (600, 400)]

    # 获取初始颜色
    initial_colors = [pyautogui.pixel(x, y) for x, y in screen_positions]

    # 等待一段时间
    time.sleep(1)

    # 获取更新后的颜色
    updated_colors = [pyautogui.pixel(x, y) for x, y in screen_positions]

    # 检查颜色是否发生变化
    if initial_colors == updated_colors:
        print("卡墙")
        movements = ['w', 's', 'd', 'a']
        movement_index = random.randint(0, len(movements) - 1)
        keyboard.press(movements[movement_index])
        time.sleep(0.5 * movement_index)
        keyboard.release(movements[movement_index])
        grid = 二值化地图()
        子目标点 = 找子节点(grid)
        start = Node(中心点[0], 中心点[1])
        goal = Node(子目标点[0], 子目标点[1])
        print(子目标点)
    else:
        pass
def 移动(inflection_points):
    speed_factor = 0.04
    for i in range(len(inflection_points)):
        current_x, current_y = inflection_points[i]
        if i + 1 < len(inflection_points):
            next_x, next_y = inflection_points[i + 1]
        else:
            next_x, next_y = 子目标点[0], 子目标点[1]
            pass
        dx = 子目标点[0] - 中心点[0]
        dy = 子目标点[1] - 中心点[1]

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
x1, y1, x2, y2 =1528,120,1758,350
region = (x1, y1, x2, y2)
step_size = 5
中心点 =(int((x2-x1)/step_size/2)+3,int((y2-y1)/step_size/2))
grid = 二值化地图()
子目标点 = 找子节点(grid)
start = Node(中心点[0], 中心点[1])
goal = Node(子目标点[0], 子目标点[1])
time.sleep(3)
while True:
    # try:
    # 使用A星算法寻找路径
    path = A星(start, goal, grid)
    # 如果路径不为空，则执行移动
    卡墙()
    找终点()
    if path:
        print("开始移动")
        拐点列表 = 寻找拐点(path)
        time.sleep(3)
        移动(拐点列表)
        # 移动到子节点后重置grid、path和子节点
        grid = 二值化地图()
        子目标点 = 找子节点(grid)
        start = Node(中心点[0], 中心点[1])
        goal = Node(子目标点[0], 子目标点[1])
        print(子目标点)
    else:
        print('找不到路径，随机移动')
        movements = ['w', 's', 'd', 'a']
        movement_index = random.randint(0, len(movements) - 1)
        keyboard.press(movements[movement_index])
        time.sleep(0.5 * movement_index)
        keyboard.release(movements[movement_index])
        grid = 二值化地图()
        子目标点 = 找子节点(grid)
        start = Node(中心点[0], 中心点[1])
        goal = Node(子目标点[0], 子目标点[1])
        print(子目标点)


    #
    # except Exception as e:
    #     print(f'发生错误: {e}')
    #     print('随机移动')
    #     movements = ['w', 's', 'd', 'a']
    #     movement_index = random.randint(0, len(movements) - 1)
    #     keyboard.press(movements[movement_index])
    #     time.sleep(0.5 * movement_index)
    #     keyboard.release(movements[movement_index])