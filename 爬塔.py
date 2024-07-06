import cv2
import numpy as np
from PIL import ImageGrab
import time
import keyboard
import win32gui
import win32ui
import win32con
from PIL import Image,ImageGrab
import pyautogui
import re
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import cv2
# # 定义要查找的图片路径
# gantanhao = 'gantanhao.png'
# chuansong = 'chuansong.png'
# renwu = 'renwu.png'
# wa = 'wa.png'
# waer = 'waer.png'
# gantanhao_list = []
# wa_list = []
# waer_list = []
# renwu_list = []
# chuansong_list = []
#
# def get_screen():
#     # app_title = "MuMu模拟器12"
#     app_title = "媒体播放器"
#     hwnd = win32gui.FindWindow(None, app_title)
#     left, top, right, bottom = win32gui.GetWindowRect(hwnd)
#     width = right - left
#     height = bottom - top
#     # print(width, height)
#     return left, top, width, height
# # 获取屏幕截图
# while True:
#     region = (get_screen())
#     screen = ImageGrab.grab(region)
#     screen_np = np.array(screen)
#     screen_np_other = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
#     screen_gray = cv2.cvtColor(screen_np_other, cv2.COLOR_BGR2GRAY)
#
#     # 加载模板图片
#     gantanhao_pic = cv2.imread(gantanhao, 0)
#     # 使用模板匹配
#     result = cv2.matchTemplate(screen_gray, gantanhao_pic, cv2.TM_CCOEFF_NORMED)
#     # 设置阈值
#     threshold = 0.85
#
#     # 获取匹配的位置
#     loc = np.where(result >= threshold)
#
#     # 打印匹配位置
#     for pt in zip(*loc[::-1]):
#         gantanhao_list.append(pt[0], pt[1])
#     time.sleep(1)
#
#
#
#
#     # 加载模板图片
#     gantanhao_pic = cv2.imread(chuansong, 0)
#     # 使用模板匹配
#     result = cv2.matchTemplate(screen_gray, gantanhao_pic, cv2.TM_CCOEFF_NORMED)
#     # 设置阈值
#     threshold = 0.85
#
#     # 获取匹配的位置
#     loc = np.where(result >= threshold)
#
#     # 打印匹配位置
#     for pt in zip(*loc[::-1]):
#         chuansong_list.append(pt[0], pt[1])
#     time.sleep(1)
#
#
#     # 加载模板图片
#     gantanhao_pic = cv2.imread(renwu, 0)
#     # 使用模板匹配
#     result = cv2.matchTemplate(screen_gray, gantanhao_pic, cv2.TM_CCOEFF_NORMED)
#     # 设置阈值
#     threshold = 0.85
#
#     # 获取匹配的位置
#     loc = np.where(result >= threshold)
#
#     # 打印匹配位置
#     for pt in zip(*loc[::-1]):
#         renwu_list.append(pt[0], pt[1])
#     time.sleep(1)

def get_screen():
    app_title = "媒体播放器"
    hwnd = win32gui.FindWindow(None, app_title)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    return (left, top, width, height)

def match_template_and_store(screen_gray, template_path, result_list, threshold=0.75):
    # 加载模板图片
    template_pic = cv2.imread(template_path, 0)
    if template_pic is None:
        print(f"Error: Template image '{template_path}' not loaded properly.")
        return

    # 使用模板匹配
    result = cv2.matchTemplate(screen_gray, template_pic, cv2.TM_CCOEFF_NORMED)

    # 获取匹配的位置
    loc = np.where(result >= threshold)

    # 存储匹配位置
    for pt in zip(*loc[::-1]):
        result_list.append((pt[0], pt[1]))

def main():
    while True:
        region = get_screen()
        screen = ImageGrab.grab(bbox=region)
        screen_np = np.array(screen)
        screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

        # 模板图片路径
        templates = {
            'gantanhao': 'gantanhao.png',
            'chuansong': 'chuansong.png',
            'renwu': 'renwu.png'
        }

        # 匹配结果列表
        matches = {
            'gantanhao': [],
            'chuansong': [],
            'renwu': []
        }

        # 遍历模板并进行匹配
        for name, template_path in templates.items():
            match_template_and_store(screen_gray, template_path, matches[name])
            time.sleep(1)  # 等待一段时间，避免连续截图过于频繁

        # 打印匹配位置
        for name, match_list in matches.items():
            print(f"{name} matches: {match_list}")
        # print(matches)
        # for i,j in matches['gantanhao']:
        #     for ii, jj in matches['renwu']:
        #         print(abs(i - ii), abs(j - jj))
        xx,yy = matches['gantanhao'][0]
        xxx,yyy = matches['renwu'][0]
        go_x,go_y=xx - xxx,yy - yyy
        if go_x > 0 and go_y > 0:
            keyboard.press("d")
            time.sleep(abs(go_x / 10))
            keyboard.release("d")
            keyboard.press("s")
            time.sleep(abs(go_y / 10))
            keyboard.release("s")
        elif go_x < 0 and go_y < 0:
            keyboard.press("a")
            time.sleep(abs(go_x / 10))
            keyboard.release("a")
            keyboard.press("w")
            time.sleep(abs(go_y / 10))
            keyboard.release("w")
        elif go_x > 0 and go_y < 0:
            keyboard.press("d")
            time.sleep(abs(go_x / 10))
            keyboard.release("d")
            keyboard.press("w")
            time.sleep(abs(go_y / 10))
            keyboard.release("w")
        elif go_x < 0 and go_y > 0:
            keyboard.press("a")
            time.sleep(abs(go_x / 10))
            keyboard.release("a")
            keyboard.press("s")
            time.sleep(abs(go_y / 10))
            keyboard.release("s")
        else:
            pass#随机移动



if __name__ == "__main__":
    main()

