import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel,QLineEdit,QPushButton
from PyQt5.QtCore import Qt, QTimer
from PIL import ImageGrab
import cv2
import numpy as np
import time
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        # 初始化滑块值
        self.lower_blue = np.array([110, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

    def initUI(self):
        layout = QVBoxLayout()

        # 创建六个滑块和标签
        self.sliders = []
        self.labels = []
        self.line_edits = []
        self.update_buttons = []
        for i in range(6):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(255)
            slider.valueChanged.connect(self.updateValues)
            self.sliders.append(slider)

            label = QLabel(str(slider.value()))
            self.labels.append(label)

            line_edit = QLineEdit()
            line_edit.textChanged.connect(self.validateInput)
            self.line_edits.append(line_edit)

            update_button = QPushButton("Update")
            update_button.clicked.connect(lambda _, idx=i: self.updateSliderValue(idx))
            self.update_buttons.append(update_button)

            layout.addWidget(slider)
            layout.addWidget(label)
            layout.addWidget(line_edit)
            layout.addWidget(update_button)

        self.setLayout(layout)

        # 设置定时器，每秒更新一次
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detectColors)
        self.timer.start(1000)

    def updateValues(self, value):
        sender = self.sender()
        index = self.sliders.index(sender)
        self.labels[index].setText(str(value))
        self.line_edits[index].setText(str(value))

        # 更新颜色检测范围
        if index < 3:  # 前三个滑块对应 lower_blue
            self.lower_blue[index % 3] = value
        else:  # 后三个滑块对应 upper_blue
            self.upper_blue[(index - 3) % 3] = value

    def validateInput(self, text):
        sender = self.sender()
        index = self.line_edits.index(sender)
        try:
            int_value = int(text)
            if 0 <= int_value <= 255:
                self.labels[index].setText(text)
            else:
                self.labels[index].setText("Invalid")
        except ValueError:
            self.labels[index].setText("Invalid")

    def updateSliderValue(self, index):
        sender = self.line_edits[index]
        text = sender.text()
        try:
            int_value = int(text)
            if 0 <= int_value <= 255:
                self.sliders[index].setValue(int_value)
            else:
                print("Invalid input. Value must be between 0 and 255.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    def detectColors(self):
        region = (1366, 275, 1566, 475)
        partial_screen = ImageGrab.grab(bbox=region)
        screen_array = np.array(partial_screen)
        screen_array_bgr = cv2.cvtColor(screen_array, cv2.COLOR_BGR2RGB)

        screen_array_hsv = cv2.cvtColor(screen_array_bgr, cv2.COLOR_RGB2HSV)

        # 使用更新后的颜色范围
        lower_blue = self.lower_blue
        upper_blue = self.upper_blue

        mask = cv2.inRange(screen_array_hsv, lower_blue, upper_blue)
        coordinates = np.column_stack(np.where(mask == 255))

        # 找到颜色范围内的像素坐标


        for x, y in coordinates:
            cv2.circle(screen_array_bgr, (y, x), 5, (0, 0, 0), -1)
        cv2.imshow('Screen with marked blue', screen_array_bgr)
        cv2.waitKey(1)
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()