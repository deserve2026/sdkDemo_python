# This Python file uses the following encoding: utf-8
UDP_SOFT_ASK_PORT = 1009;    #软件udp端口
UDP_DATA_PORT = 5001;        #声呐设备upd数据发送端口
TCP_CMD_PORT = 5007;         #声呐设备tcp命令接收端口
UDP_REC_PORT = 5008;         #声呐设备udp接收端口
UDP_REC_SDK_PORT = 5009;     #声呐设备udp接收端口（SDK）
UDP_DEV_ASK_PORT = 9001;     #设备udp查询端口

IMAGE_WIDTH = 968;
IMAGE_HEIGHT = 512;

PACK_INFO_SIZE = 256;  #包头信息长度
PACK_DATA_INDEX = 256; #每包数据索引
PACK_MAX_NUM = 12000;  #每帧数据最大包数

import sys
import socket
import struct
import psutil
import ctypes
import time
import threading
import numpy as np
# import cv2

import pyLhForwardSDK_module

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtCore import Slot, QTimer, Signal, QStandardPaths

from enum import Enum
from collections import defaultdict

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget

# 定义udp广播包结构体
class DeviceDetailData(ctypes.Structure):
    _pack_ = 1  # 保证结构体是紧凑对齐的
    _fields_ = [
        ("frequency", ctypes.c_float),
        ("winFunc", ctypes.c_uint8 * 8),
        ("tvg", ctypes.c_uint8 * 8),
        ("sigMode", ctypes.c_uint8 * 8),
        ("minDistReso", ctypes.c_float),
        ("maxDistReso", ctypes.c_float),
        ("horBeam", ctypes.c_short),
        ("minHorAngleReso", ctypes.c_float),
        ("maxHorAngleReso", ctypes.c_float),
        ("verBeam", ctypes.c_short),
        ("minVerAngleReso", ctypes.c_float),
        ("maxVerAngleReso", ctypes.c_float),
        ("maxRange", ctypes.c_float),
        ("horAngle", ctypes.c_float),
        ("reserve", ctypes.c_uint8 * 32)
    ]

class DeviceTotalInfo(ctypes.Structure):
    _pack_ = 1  # 保证结构体是紧凑对齐的
    _fields_ = [
        ("flag", ctypes.c_uint8),
        ("deviceName", ctypes.c_char * 15),
        ("chipId", ctypes.c_uint64),
        ("licStatus", ctypes.c_uint32),
        ("workModeNum", ctypes.c_uint8),
        ("brightnessBase", ctypes.c_uint8),
        ("modeFlag", ctypes.c_uint8 * 8),
        ("reverse", ctypes.c_uint8 * 90),
        ("dataDetail", DeviceDetailData * 8)
    ]

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.setWindowTitle("sapphireview sdk demo(python)")  # 设置标题
        self.setFixedSize(968, 800)  # 设置窗口固定大小

        self.connentOK = False # 连接标志位
        self.runFlag_udp = False # udp线程标志位
        self.not_ip = False # 是否广播连接
        self.runFlag_read = False # 回放线程标志位

        # sdk Frame类
        self.m_lhForwardFrame = pyLhForwardSDK_module.LhForwardFrame()
        # sdk Image类
        self.m_lhForwardImage = pyLhForwardSDK_module.LhForwardImage(IMAGE_WIDTH, IMAGE_HEIGHT)
        # sdk cmd类
        self.cmd = None
        # 包大小(5800)
        self.PACK_SIZE = pyLhForwardSDK_module.LhForwardFrame.PACK_SIZE
        # udp接收线程
        self.thread_udp = None
        # sdk 包头结构体
        self.curFrameInfo = pyLhForwardSDK_module.packageInfo()
        # udp广播包结构体
        self.devInfo = DeviceTotalInfo()
        # 回放线程
        self.thread_read = None

        # 创建 QLabel 组件（图像）
        self.m_imageLabel = QLabel(self)
        self.m_imageLabel.setFixedWidth(IMAGE_WIDTH)
        self.m_imageLabel.setFixedHeight(IMAGE_HEIGHT)

        # 创建 QLabel 组件（声呐名）
        self.label_name = QLabel(self)
        self.label_name.resize(200, 30)
        self.label_name.move(150, 515)

        # 创建 QPushButton 组件（连接）
        self.m_connectPBtn = QPushButton("连接", self)
        self.m_connectPBtn.move(150, 550)
        self.m_connectPBtn.resize(100, 40)
        self.m_connectPBtn.clicked.connect(self.connectPBtnActive)

        # 创建 QTimer 组件（连接超时）
        self.m_connectTimer = QTimer(self);
        self.m_connectTimer.setInterval(2000);
        self.m_connectTimer.timeout.connect(self.connectTimerActive)

        # 创建 QPushButton 组件（发射）
        self.btn_power = QPushButton("发射",self);
        self.btn_power.move(150, 600);
        self.btn_power.resize(100, 40);
        self.btn_power.setStyleSheet("background-color: #2c9678;");
        self.btn_power.clicked.connect(self.setPower);
        self.btn_power.setEnabled(self.connentOK)

        # 创建 QComboBox 组件（模式）
        self.combox_mode = QComboBox(self);
        self.combox_mode.move(150, 650);
        self.combox_mode.resize(123, 40);
        self.combox_mode.currentIndexChanged.connect(self.changeWorkMode)
        self.combox_mode.setEnabled(self.connentOK)

        # 创建 QLabel 组件（ip）
        self.label_ip = QLabel("声呐ip：",self);
        self.label_ip.move(380, 550);
        self.label_ip.setEnabled(not self.connentOK)

        # 创建 QTextEdit 组件（ip输入框）
        self.text_ip = QTextEdit(self);
        self.text_ip.move(460, 550);
        self.text_ip.resize(190, 33);

        # 创建 QPushButton 组件（回放）
        self.playback_btn = QPushButton("回放",self);
        self.playback_btn.move(700,550);
        self.playback_btn.resize(100,40);
        self.playback_btn.clicked.connect(self.playStart);
        self.playback_btn.setEnabled(not self.connentOK)

        # 创建 QLabel 组件（gamma）
        self.label_gamma = QLabel("gamma系数：",self);
        self.label_gamma.move(350, 600);

        # 创建 QTextEdit 组件（gamma输入框）
        self.text_gamma = QTextEdit(self);
        self.text_gamma.move(460, 600);
        self.text_gamma.resize(100, 33);

        # 创建 QPushButton 组件（设置gamma）
        self.btn_gamma = QPushButton("设置",self);
        self.btn_gamma.move(570, 600);
        self.btn_gamma.resize(80, 33);
        self.btn_gamma.clicked.connect(self.setGamma);
        self.btn_gamma.setEnabled(self.connentOK)

        # 创建 QLabel 组件（max）
        self.label_distance = QLabel("最大距离：",self);
        self.label_distance.move(360, 650);

        # 创建 QTextEdit 组件（max输入框）
        self.text_distance = QTextEdit(self);
        self.text_distance.move(460, 650);
        self.text_distance.resize(100, 33);

        # 创建 QPushButton 组件（设置max）
        self.btn_distance = QPushButton("设置",self);
        self.btn_distance.move(570, 650);
        self.btn_distance.resize(80, 33);
        self.btn_distance.clicked.connect(self.setMaxDistance);
        self.btn_distance.setEnabled(self.connentOK)


        # 获取本机的主机名
        hostname = socket.gethostname()
        # 使用主机名获取本机的 IP 地址
        self.local_ip = socket.gethostbyname(hostname)

        try:
            # 创建 UDP 套接字
            self.m_udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # 设置接收缓冲区大小为 10MB
            netBuffer = 1024 * 1024 * 10
            self.m_udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, netBuffer)

            # 设置接收超时时间为 50ms
            outTime = 0.05  # Python 中以秒为单位
            self.m_udpSocket.settimeout(outTime)

            # 绑定到指定端口
            m_udpAddr = (self.local_ip, UDP_REC_SDK_PORT)  # 绑定到所有可用的网络接口
            self.m_udpSocket.bind(m_udpAddr)
            print(f"UDPbind成功 IP {self.local_ip} port {UDP_REC_SDK_PORT}")

        except self.m_udpSocket.error as e:
            print(f"bind失败 IP {self.local_ip} port {UDP_REC_SDK_PORT}: {e}")

    def worker_udp(self):
        print("子线程开启,udp接收与成像")
        count = 0

        tempImage = QImage()
        frameDataBuf = np.zeros(1024 * 1024 * 10, dtype=np.uint8)

        while self.runFlag_udp:
            try:
                data, addr = self.m_udpSocket.recvfrom(self.PACK_SIZE)
            except socket.timeout:
                continue  # 在超时时继续等待
            # 成像
            uint8_data = np.frombuffer(data, dtype = np.uint8)
            ret = self.m_lhForwardFrame.writeOnePackData(uint8_data)
            if ret == pyLhForwardSDK_module.LfFrameRet.oneFrameOK:
                self.m_lhForwardFrame.getOneFrame(self.curFrameInfo, frameDataBuf)

                #显示设备名
                self.label_name.setText(self.curFrameInfo.deviceName)

                #显示图像
                if tempImage.loadFromData(frameDataBuf.tobytes(), "jpg"):
                    self.m_imageLabel.setPixmap(QPixmap.fromImage(tempImage))
                    self.m_imageLabel.show()
            count = count + 1
            if count >= 10:
                count = 0
                time.sleep(0.001)

    @Slot()
    def connectPBtnActive(self):
        button_text = self.m_connectPBtn.text()
        if button_text == "连接":
            self.m_connectPBtn.setText("连接中...")
            # 创建子线程（udp接收）
            self.thread_udp = threading.Thread(target=self.worker_udp)

            self.m_connectTimer.start()
        elif button_text == "连接中...":
            self.m_connectTimer.stop()
            self.m_connectPBtn.setText("连接")
        elif button_text == "断开连接":
            if self.cmd is not None:
                self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.power, 0) # 关闭发射
                self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.startStop, 0) # 停止声呐
                self.cmd = None

            self.connentOK = False # 禁用按钮
            self.btn_power.setEnabled(self.connentOK)
            self.playback_btn.setEnabled(not self.connentOK)
            self.btn_gamma.setEnabled(self.connentOK)
            self.btn_distance.setEnabled(self.connentOK)
            self.combox_mode.setEnabled(self.connentOK)

            self.runFlag_udp = False
            self.thread_udp.join() # 子线程停止

            self.m_connectPBtn.setText("连接")

    @Slot()
    def connectTimerActive(self):
        ip_address_string = self.text_ip.toPlainText().strip()

        if not ip_address_string:
            print("无指定IP,启用广播搜索")
            self.not_ip = True
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # 启用广播
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
            message = b'SVSearching\0'
            broadcast_address = ('<broadcast>', UDP_DEV_ASK_PORT)

            try:
                sock.sendto(message, broadcast_address)
                data, addr = sock.recvfrom(1024)

                print(addr[0])
                if len(data) == 896:
                    ip_address_string = addr[0]

                    # 拷贝数据到结构体
                    ctypes.memmove(ctypes.byref(self.devInfo), data, len(data))
            finally:
                sock.close()

        # 建立TCP链接
        print("调用cmd库，建立TCP链接!")
        self.cmd = pyLhForwardSDK_module.LhForwardCMD(ip_address_string, TCP_CMD_PORT)

        ret = self.cmd.connectDevice()

        # 更新工作模式下拉框(广播连接才有，需要广播包)
        if self.not_ip:
            self.combox_mode.clear()
            if self.devInfo.workModeNum == 1:
                self.combox_mode.addItem("大范围模式")
            elif self.devInfo.workModeNum == 2:
                self.combox_mode.addItem("大范围模式")
                self.combox_mode.addItem("高分辨模式")
        self.not_ip = False

        if ret == pyLhForwardSDK_module.LfCMDRet.ok:
            self.m_connectPBtn.setText("断开连接")
            ret1 = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.dataSource, 4)  # 设置上传数据类型为jpg

            if ret1 == pyLhForwardSDK_module.LfCMDRet.ok:
                print("命令发送成功,设置上传数据源")
            time.sleep(0.2)


            # 将IP地址转换为4字节数组
            ip_bytes = list(map(int, self.local_ip.split('.')))

            ret2 = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.udpData, ip_bytes, UDP_REC_SDK_PORT)

            if ret2 == pyLhForwardSDK_module.LfCMDRet.ok:
                print("命令发送成功,设置声呐数据接收IP和端口")
            time.sleep(0.2)

            ret3 = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.startStop, 1) # 启动

            if ret3 == pyLhForwardSDK_module.LfCMDRet.ok:
                print("命令发送成功,声呐启动")
            time.sleep(0.2)

            # ret4 = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.perid, 1000) # 工作周期

            # if ret4 == pyLhForwardSDK_module.LfCMDRet.ok:
            #     print("命令发送成功,设置工作周期")
            # time.sleep(0.2)

            self.m_connectTimer.stop() # 停止连接计时器

            self.connentOK = True # 开放按钮
            self.btn_power.setEnabled(self.connentOK)
            self.playback_btn.setEnabled(not self.connentOK)
            self.btn_gamma.setEnabled(self.connentOK)
            self.btn_distance.setEnabled(self.connentOK)
            self.combox_mode.setEnabled(self.connentOK)

            self.runFlag_udp = True
            self.thread_udp.start() # 子线程启动


    @Slot()
    def setPower(self):
        if self.btn_power.text() == "发射":
            ret = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.power, 255)
            self.btn_power.setText("关闭")
            self.btn_power.setStyleSheet("background-color: #ed5a65;")
            print("打开发射")
        elif self.btn_power.text() == "关闭":
            ret = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.power, 0)
            self.btn_power.setText("发射")
            self.btn_power.setStyleSheet("background-color: #2c9678;")
            print("关闭发射")

    @Slot()
    def changeWorkMode(self, index):
        if index is not self.curFrameInfo.workMode:
            ret = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.workMode, index)
            if ret == pyLhForwardSDK_module.LfCMDRet.ok:
                print(f"工作模式：{index}")

    def worker_read(self, file_path):
        count = 0

        tempImage = QImage()
        frameDataBuf = np.zeros(1024 * 1024 * 10, dtype=np.uint8)
        m_intensity = np.zeros(PACK_MAX_NUM * self.PACK_SIZE, dtype=np.double)
        m_panSectorMemory = np.zeros(PACK_MAX_NUM * self.PACK_SIZE, dtype=np.uint8)

        try:
            # 尝试以二进制模式打开文件
            with open(file_path, 'rb') as file:
                print("文件成功打开")
                while self.runFlag_read:
                    bytes_read = file.read(self.PACK_SIZE)
                    if len(bytes_read) < self.PACK_SIZE:
                        # 文件结束
                        print("文件已读取完毕，重新开始读取")
                        file.seek(0)  # 将文件指针移动到文件的开始位置
                        continue  # 跳过当前循环的其余部分，重新读取文件
                    # 成像
                    uint8_data = np.frombuffer(bytes_read, dtype = np.uint8)
                    ret = self.m_lhForwardFrame.writeOnePackData(uint8_data)
                    if ret == pyLhForwardSDK_module.LfFrameRet.oneFrameOK:
                        self.m_lhForwardFrame.getOneFrame(self.curFrameInfo, frameDataBuf)

                        #显示设备名
                        self.label_name.setText(self.curFrameInfo.deviceName)

                        #显示图像
                        if self.curFrameInfo.packType == 0:
                            # 计算尺寸
                            panSize = max(IMAGE_WIDTH, IMAGE_HEIGHT)
                            lwidth = IMAGE_WIDTH
                            lheight = IMAGE_HEIGHT
                            if self.curFrameInfo.horAngleReso == 360:
                                lwidth = panSize
                                lheight = panSize

                            # 初始化图像缓冲区
                            m_panSectorMemory = np.zeros(lwidth * lheight, dtype=np.uint8)
                            imageRes = np.array([0.0], dtype=np.float64)

                            self.m_lhForwardImage.generateForwardImage(self.curFrameInfo,frameDataBuf, m_panSectorMemory, imageRes)
                            tempImg = QImage(m_panSectorMemory, lwidth, lheight, QImage.Format_Grayscale8)

                            # 缩放图像以适配 QLabel 高度（等比）
                            if tempImg.height() > self.m_imageLabel.height():
                                tempImg = tempImg.scaledToHeight(self.m_imageLabel.height())

                            # 居中裁剪图像
                            img_w, img_h = tempImg.width(), tempImg.height()
                            label_w, label_h = self.m_imageLabel.width(), self.m_imageLabel.height()
                            start_x = max(0, int((img_w - label_w) / 2))
                            start_y = max(0, int((img_h - label_h) / 2))
                            tempImg = tempImg.copy(start_x, start_y, label_w, label_h)

                            self.m_imageLabel.setPixmap(QPixmap.fromImage(tempImg))
                            self.m_imageLabel.show()

                        elif self.curFrameInfo.packType == 2:
                            self.m_lhForwardImage.preprocessData(self.curFrameInfo, frameDataBuf, m_intensity)
                            coef = 256

                            if self.curFrameInfo.dataWidth == 0x00:
                                coef = 1

                            for i in range(self.curFrameInfo.numPerRow * self.curFrameInfo.dataHeight):
                                frameDataBuf[i] = m_intensity[i] // coef

                            tempImg = QImage(frameDataBuf.data, self.curFrameInfo.numPerRow, self.curFrameInfo.dataHeight, QImage.Format_Indexed8)
                            self.m_imageLabel.setPixmap(QPixmap.fromImage(tempImg))
                            self.m_imageLabel.show()

                        elif self.curFrameInfo.packType == 3:
                            if tempImage.loadFromData(frameDataBuf.tobytes(), "png"):
                                self.m_imageLabel.setPixmap(QPixmap.fromImage(tempImage))
                                self.m_imageLabel.show()

                        elif self.curFrameInfo.packType == 4:
                            if tempImage.loadFromData(frameDataBuf.tobytes(), "jpg"):
                                self.m_imageLabel.setPixmap(QPixmap.fromImage(tempImage))
                                self.m_imageLabel.show()

                    count = count + 1
                    if count >= 10:
                        count = 0
                        time.sleep(0.001)
        except IOError:
            print("文件打开失败")
            return

    @Slot()
    def playStart(self):
        if self.playback_btn.text() == "回放":
            print("回放")
            # 获取桌面路径
            desktop_path = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
            # 打开文件对话框并获取文件路径
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择回放文件",
                desktop_path,
                "蓝衡数据格式 (*.DB)"
            )

            if file_path: # 选择了文件
                self.playback_btn.setText("停止")
                self.m_connectPBtn.setEnabled(False) # 禁用连接防止冲突
                # 打印文件路径
                print(file_path)
                # 创建子线程（回放）
                self.thread_read = threading.Thread(target=self.worker_read, args=(file_path,))
                self.runFlag_read = True
                self.thread_read.start()

        elif self.playback_btn.text() == "停止":
            if self.runFlag_read is not None:
                self.runFlag_read = False
                self.thread_read.join() # 子线程停止

                self.playback_btn.setText("回放")
                self.m_connectPBtn.setEnabled(True) # 启用连接

    @Slot()
    def setGamma(self):
        temp = float(self.text_gamma.toPlainText())
        ret = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.gamma, temp)
        if ret == pyLhForwardSDK_module.LfCMDRet.ok:
            print(f"gamma: {temp}")

    @Slot()
    def setMaxDistance(self):
        temp = float(self.text_distance.toPlainText())
        ret = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.maxDist, temp)
        print(f"ret:{ret}")
        if ret == pyLhForwardSDK_module.LfCMDRet.ok:
            print(f"max: {temp}")

    def closeEvent(self, event):
        # 退出程序
        print("析构")
        self.runFlag_udp = False
        if self.thread_udp is not None and self.thread_udp.is_alive():
            self.thread_udp.join() # 子线程停止
        self.m_udpSocket.close() # 关闭udp socket

        self.runFlag_read = False
        if self.thread_read is not None and self.thread_read.is_alive():
            self.thread_read.join() # 子线程停止

        if self.cmd is not None:
            self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.power, 0) # 关闭发射
            self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.startStop, 0) # 停止声呐
            del self.cmd

        del self.m_lhForwardFrame
        del self.m_lhForwardImage

        event.accept()  # 继续关闭窗口


if __name__ == "__main__":
    app = QApplication(sys.argv)

    font = QFont("楷体", 15)  # 默认字体设置
    app.setFont(font)

    widget = Widget()
    widget.show()
    sys.exit(app.exec())
