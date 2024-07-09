# 羁绊 = input("请输入羁绊：")
# 词条 = input("请输入词条：")
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
ocr = PaddleOCR(lang='ch')
left, top, width, height= get_screen()
def hanzi():
    region =(left+0.49*width, top+0.12*height, left+0.71*width, top+0.64*height)
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
    return filtered_list
def shaixuan():
    for i in range(5):
        for j in range(5):
            pyautogui.moveTo(x_point + i*x ,y_point + j*y, duration=0.4)
            pyautogui.click()
            wenzi= hanzi()
            print(wenzi)
            if "冥想" in wenzi:
                print("_______111________")
                if "力量" in wenzi:
                    pyautogui.moveTo(buy_point_y,buy_point_x,  duration=0.4)
                    pyautogui.click()
if __name__ == '__main__':
    x_point = left + 0.2*width - 30
    y_point = top + 0.37*height -20
    buy_point_x =  top+ 0.91*height
    buy_point_y = left + 0.61*width
    x = 0.061*width
    y = x
    while True:
        shaixuan()
        keyboard.press_and_release('esc')
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.91*width, top + 0.21*height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.64*width, top + 0.79*height, duration=0.4)
        pyautogui.click()
        keyboard.press("s")
        time.sleep(1)
        keyboard.release("s")
        keyboard.press_and_release('f')
        pyautogui.moveTo(left + 0.92 * width, top + 0.54 * height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.92 * width, top + 0.54 * height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.52 * width, top + 0.63 * height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.89 * width, top + 0.91 * height, duration=0.4)
        pyautogui.click()
        time.sleep(4)
        pyautogui.moveTo(left + 0.84*width, top + 0.22*height, duration=0.4)
        pyautogui.click()
        pyautogui.moveTo(left + 0.87 * width, top + 0.8 * height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.87 * width, top + 0.8 * height, duration=0.4)
        pyautogui.click()
        shaixuan()
        time.sleep(1)
        keyboard.press_and_release('esc')
        time.sleep(0.1)
        keyboard.press("d")
        time.sleep(1)
        keyboard.release("d")
        keyboard.press("s")
        time.sleep(3)
        keyboard.release("s")
        keyboard.press_and_release('f')
        pyautogui.moveTo(left + 0.1 * width, top + 0.54 * height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.1 * width, top + 0.54 * height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.45 * width, top + 0.64 * height, duration=0.4)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.moveTo(left + 0.89 * width, top + 0.91 * height, duration=0.4)
        pyautogui.click()
        time.sleep(4)
        pyautogui.moveTo(left + 0.85 * width, top + 0.21 * height, duration=0.4)
        pyautogui.click()
        keyboard.press_and_release('f')
        time.sleep(0.1)
        keyboard.press_and_release('f')



