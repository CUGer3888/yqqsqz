import time

import win32gui
import win32ui
import win32con
from PIL import Image,ImageGrab
import pyautogui
import re
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import cv2

#
# def get_window_handle(title):
#     hwnd = win32gui.FindWindow(None, title)
#     return hwnd
#
# def take_screenshot(hwnd):
#     # 获取窗口的位置和大小
#     left, top, right, bottom = win32gui.GetWindowRect(hwnd)
#     width = right - left
#     height = bottom - top
#
#     # 创建设备上下文
#     hdesktop = win32gui.GetDesktopWindow()
#     desktop_dc = win32gui.GetWindowDC(hdesktop)
#     img_dc = win32ui.CreateDCFromHandle(desktop_dc)
#     mem_dc = img_dc.CreateCompatibleDC()
#
#     # 创建位图对象
#     bitmap = win32ui.CreateBitmap()
#     bitmap.CreateCompatibleBitmap(img_dc, width, height)
#     mem_dc.SelectObject(bitmap)
#
#     # 截图
#     mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)
#
#     # 保存截图
#     bmpinfo = bitmap.GetInfo()
#     bmpstr = bitmap.GetBitmapBits(True)
#     image = Image.frombuffer(
#         'RGB',
#         (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
#         bmpstr, 'raw', 'BGRX', 0, 1
#     )
#     image.save("application_screenshot.png")
#
#     # 释放资源
#     mem_dc.DeleteDC()
#     win32gui.ReleaseDC(hdesktop, desktop_dc)
#     win32gui.DeleteObject(bitmap.GetHandle())
#
# # 指定应用程序窗口的标题
# app_title = "MuMu模拟器12"
#
# # 获取窗口句柄
# hwnd = get_window_handle(app_title)
# if hwnd:
#     print(f"Found window with title: {app_title}")
#     # 截取窗口截图
#     take_screenshot(hwnd)
#     print("Screenshot saved as application_screenshot.png")
# else:
#     print(f"Window with title '{app_title}' not found.")
"""

def extract_chinese_characters(input_list):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    result = []
    for item in input_list:
        chinese_chars = chinese_pattern.findall(item)
        if chinese_chars:
            result.append(''.join(chinese_chars))
    return result
def get_screen():
    app_title = "MuMu模拟器12"
    hwnd = win32gui.FindWindow(None, app_title)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    print(width, height)
    return left, top, width, height
# ocr = PaddleOCR(lang='ch')
left, top, width, height= get_screen()
region =(left, top, width, height)
print(region)
partial_screen = ImageGrab.grab(region)
screen_array = np.array(partial_screen)
screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
result = ocr.ocr(screen_array_bgr, cls=True)
lis = []
for i in range(100):
    try:
        # print(result[0][i][1])
        x,y=result[0][i][1]
        lis.append(x)
    except:
        continue
filtered_list = extract_chinese_characters(lis)
print(filtered_list)
if "召唤物全体伤害" in filtered_list:
    print("_______111________")
"""

#鼠标移动

"""遍历
app_title = "MuMu模拟器12"
hwnd = win32gui.FindWindow(None, app_title)
left, top, right, bottom = win32gui.GetWindowRect(hwnd)
width = right - left
height = bottom - top
x_point = left + 0.2*width - 30
y_point = top + 0.37*height -20
x = 0.061*width
y = x
for i in range(5):
    for j in range(5):
        pyautogui.moveTo(x_point + i*x ,y_point + j*y, duration=0.4)
        pyautogui.click()
"""

# def extract_chinese_characters(input_list):
#     chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
#     result = []
#     for item in input_list:
#         chinese_chars = chinese_pattern.findall(item)
#         if chinese_chars:
#             result.append(''.join(chinese_chars))
#     return result
# def get_screen():
#     app_title = "MuMu模拟器12"
#     hwnd = win32gui.FindWindow(None, app_title)
#     left, top, right, bottom = win32gui.GetWindowRect(hwnd)
#     width = right - left
#     height = bottom - top
#     print(width, height)
#     return left, top, width, height
# ocr = PaddleOCR(lang='ch')
# left, top, width, height= get_screen()
# def hanzi():
#     region =(left, top, width, height)
#     partial_screen = ImageGrab.grab(region)
#     screen_array = np.array(partial_screen)
#     screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
#     result = ocr.ocr(screen_array_bgr, cls=True)
#     lis = []
#     for i in range(100):
#         try:
#             # print(result[0][i][1])
#             x,y=result[0][i][1]
#             lis.append(x)
#         except:
#             continue
#     filtered_list = extract_chinese_characters(lis)
#     return filtered_list
# x_point = left + 0.2*width - 30
# y_point = top + 0.37*height -20
# x = 0.061*width
# y = x
# for i in range(5):
#     for j in range(5):
#         pyautogui.moveTo(x_point + i*x ,y_point + j*y, duration=0.4)
#         pyautogui.click()
#         wenzi= hanzi()
#         print(wenzi)
#         if "召唤物数量" in wenzi:
#             print("_______111________")
#
# region = (9, 26, 1153, 678)
width = 1137
height = 669

x = int(input("请输入x坐标: "))
y = int(input("请输入y坐标: "))

# 计算比例并保留两位小数
ratio_x = round(x / width, 2)
ratio_y = round(y / height, 2)
print(f'pyautogui.moveTo(left + {ratio_x}*width, top + {ratio_y}*height, duration=0.4)')
