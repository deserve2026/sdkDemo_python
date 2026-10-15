# This Python file uses the following encoding: utf-8
UDP_SOFT_ASK_PORT = 1009;    #软件udp端口
UDP_DATA_PORT = 5001;        #声呐设备upd数据发送端口
TCP_CMD_PORT = 5007;         #声呐设备tcp命令接收端口
UDP_REC_PORT = 5008;         #声呐设备udp接收端口
UDP_REC_SDK_PORT = 5009;     #声呐设备udp接收端口（SDK）
UDP_DEV_ASK_PORT = 9001;     #设备udp查询端口

IMAGE_WIDTH = 968;
IMAGE_HEIGHT = 512;

# ==================== 显示方向开关 ====================
# True  = 上下翻转显示（与原厂 C++ 版 mainwindow.cpp 的 img.mirrored(false, true) 一致）
# False = 保持声呐原始方向
FLIP_VERTICAL = True
# =====================================================

PACK_INFO_SIZE = 256;  #包头信息长度
PACK_DATA_INDEX = 256; #每包数据索引
PACK_MAX_NUM = 12000;  #每帧数据最大包数

import sys
import socket
import struct
import psutil
import ctypes
import queue
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


def apply_orientation(img):
    """统一处理显示方向，并返回一份独立副本。

    - FLIP_VERTICAL=True 时做上下翻转
    - 无论是否翻转都返回 .copy()，避免 QImage 复用 frameDataBuf 内存
      （原代码中 packType==2 的 QImage 直接引用 numpy 缓冲，
        下一帧写入会污染仍留在队列中的旧图像）
    """
    if img is None:
        return None
    if FLIP_VERTICAL:
        img = img.mirrored(False, True)
    return img.copy()


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
        self.m_udpSocket = None  # 延迟到子线程内创建（对齐 C++ udpInit）

        # ---- 线程安全：用 queue + QTimer 接收子线程图像 ----
        self.image_queue = queue.Queue(maxsize=10)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)  # ~30 fps
        self._poll_timer.timeout.connect(self._poll_image_queue)
        self._poll_timer.start()

    @Slot()
    def _poll_image_queue(self):
        """主线程定时轮询队列，取出图像更新 GUI"""
        try:
            while True:
                image, device_name = self.image_queue.get_nowait()
                self.m_imageLabel.setPixmap(QPixmap.fromImage(image))
                self.m_imageLabel.show()
                self.label_name.setText(device_name)
                print(f"[GUI] 图像已显示, 设备: {device_name}")
        except queue.Empty:
            pass

    def worker_udp(self):
        print(f"子线程开启,udp接收与成像, PACK_SIZE={self.PACK_SIZE}")

        # ---- 在子线程内创建 socket（对齐 C++ UdpReceiver::udpInit） ----
        try:
            self.m_udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            netBuffer = 1024 * 1024 * 10
            self.m_udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, netBuffer)
            self.m_udpSocket.settimeout(0.05)  # 50ms
            self.m_udpSocket.bind(('0.0.0.0', UDP_REC_SDK_PORT))
            print(f"[UDP] 子线程内 socket 绑定成功 0.0.0.0:{UDP_REC_SDK_PORT}")
        except OSError as e:
            print(f"[UDP] 子线程内 socket 绑定失败: {e}")
            return

        count = 0
        tempImage = QImage()
        frameDataBuf = np.zeros(1024 * 1024 * 10, dtype=np.uint8)
        # packType 0 / 2 所需缓冲区
        m_intensity = np.zeros(PACK_MAX_NUM * self.PACK_SIZE, dtype=np.double)
        m_panSectorMemory = np.zeros(PACK_MAX_NUM * self.PACK_SIZE, dtype=np.uint8)

        packet_count = 0
        while self.runFlag_udp:
            try:
                data, addr = self.m_udpSocket.recvfrom(65535)  # 最大 UDP 包
            except socket.timeout:
                continue  # 在超时时继续等待
            except Exception as e:
                print(f"[UDP] recvfrom 异常 ({type(e).__name__}): {e}")
                continue
            packet_count += 1
            if packet_count <= 3 or packet_count % 200 == 1:
                print(f"[UDP] 收到第 {packet_count} 个数据包, 来自 {addr}, 大小={len(data)}")
            # 成像
            uint8_data = np.frombuffer(data, dtype=np.uint8)
            ret = self.m_lhForwardFrame.writeOnePackData(uint8_data)
            if ret == pyLhForwardSDK_module.LfFrameRet.oneFrameOK:
                self.m_lhForwardFrame.getOneFrame(self.curFrameInfo, frameDataBuf)

                # 设备名（c_char 数组 → str）
                device_name = self.curFrameInfo.deviceName
                if isinstance(device_name, bytes):
                    device_name = device_name.decode('utf-8', errors='replace').rstrip('\x00')

                print(f"[UDP] 收到完整帧, packType={self.curFrameInfo.packType}, 设备={device_name}")

                image = None

                # ---- 根据 packType 分别处理 ----
                if self.curFrameInfo.packType == 0:
                    # 原始扇区数据 → 用 SDK 生成扇区图
                    panSize = max(IMAGE_WIDTH, IMAGE_HEIGHT)
                    lwidth = IMAGE_WIDTH
                    lheight = IMAGE_HEIGHT
                    if self.curFrameInfo.horAngleReso == 360:
                        lwidth = panSize
                        lheight = panSize

                    m_panSectorMemory = np.zeros(lwidth * lheight, dtype=np.uint8)
                    imageRes = np.array([0.0], dtype=np.float64)

                    self.m_lhForwardImage.generateForwardImage(
                        self.curFrameInfo, frameDataBuf, m_panSectorMemory, imageRes)
                    tempImg = QImage(m_panSectorMemory, lwidth, lheight, QImage.Format_Grayscale8)

                    # 等比缩放至 QLabel 高度
                    if tempImg.height() > self.m_imageLabel.height():
                        tempImg = tempImg.scaledToHeight(self.m_imageLabel.height())

                    # 居中裁剪
                    img_w, img_h = tempImg.width(), tempImg.height()
                    label_w, label_h = self.m_imageLabel.width(), self.m_imageLabel.height()
                    start_x = max(0, int((img_w - label_w) / 2))
                    start_y = max(0, int((img_h - label_h) / 2))
                    tempImg = tempImg.copy(start_x, start_y, label_w, label_h)

                    image = apply_orientation(tempImg)

                elif self.curFrameInfo.packType == 2:
                    # 波束数据 → 预处理后转灰度图
                    self.m_lhForwardImage.preprocessData(
                        self.curFrameInfo, frameDataBuf, m_intensity)
                    coef = 256
                    if self.curFrameInfo.dataWidth == 0x00:
                        coef = 1

                    for i in range(self.curFrameInfo.numPerRow * self.curFrameInfo.dataHeight):
                        frameDataBuf[i] = m_intensity[i] // coef

                    image = QImage(
                        frameDataBuf.data,
                        self.curFrameInfo.numPerRow,
                        self.curFrameInfo.dataHeight,
                        QImage.Format_Indexed8)
                    image = apply_orientation(image)

                elif self.curFrameInfo.packType == 3:
                    if tempImage.loadFromData(frameDataBuf.tobytes(), "png"):
                        image = apply_orientation(tempImage)

                elif self.curFrameInfo.packType == 4:
                    if tempImage.loadFromData(frameDataBuf.tobytes(), "jpg"):
                        image = apply_orientation(tempImage)

                # ---- 通过队列发送到主线程显示 ----
                if image is not None:
                    try:
                        self.image_queue.put_nowait((image, device_name))
                        print(f"[UDP] 图像入队成功, 尺寸={image.width()}x{image.height()}")
                    except queue.Full:
                        print("[UDP] 队列已满，丢弃一帧")
                else:
                    print(f"[UDP] 警告: packType={self.curFrameInfo.packType} 未处理或解码失败")

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

        # ---- 探测本机正确 IP（模拟 C++ getHostSocket/getsockname 逻辑） ----
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect((ip_address_string, 1))
            correct_ip = probe.getsockname()[0]
            probe.close()
            print(f"声呐可达本机IP: {correct_ip} (gethostbyname: {self.local_ip})")
        except Exception:
            correct_ip = self.local_ip

        # 建立TCP链接（传入本机 IP，SDK 内部绑定正确网卡）
        print("调用cmd库，建立TCP链接!")
        self.cmd = pyLhForwardSDK_module.LhForwardCMD(ip_address_string, TCP_CMD_PORT)

        ret = self.cmd.connectDevice(correct_ip)

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
            ret1 = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.dataSource, 3)  # PNG（与 C++ 原厂一致）

            if ret1 == pyLhForwardSDK_module.LfCMDRet.ok:
                print("命令发送成功,设置上传数据源")
            time.sleep(1)  # 与 C++ 原厂 Sleep(1e3) 一致

            # 将 IP 地址转换为 4 字节数组
            ip_bytes = list(map(int, correct_ip.split('.')))

            ret2 = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.udpData, ip_bytes, UDP_REC_SDK_PORT)

            if ret2 == pyLhForwardSDK_module.LfCMDRet.ok:
                print("命令发送成功,设置声呐数据接收IP和端口")
            time.sleep(1)  # 与 C++ 一致

            ret3 = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.startStop, 1) # 启动

            if ret3 == pyLhForwardSDK_module.LfCMDRet.ok:
                print("命令发送成功,声呐启动")
            time.sleep(1)  # 与 C++ 一致

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

                        # 设备名（c_char 数组 → str）
                        device_name = self.curFrameInfo.deviceName
                        if isinstance(device_name, bytes):
                            device_name = device_name.decode('utf-8', errors='replace').rstrip('\x00')

                        image = None

                        # ---- 根据 packType 分别处理 ----
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

                            image = apply_orientation(tempImg)

                        elif self.curFrameInfo.packType == 2:
                            self.m_lhForwardImage.preprocessData(self.curFrameInfo, frameDataBuf, m_intensity)
                            coef = 256

                            if self.curFrameInfo.dataWidth == 0x00:
                                coef = 1

                            for i in range(self.curFrameInfo.numPerRow * self.curFrameInfo.dataHeight):
                                frameDataBuf[i] = m_intensity[i] // coef

                            image = QImage(frameDataBuf.data, self.curFrameInfo.numPerRow, self.curFrameInfo.dataHeight, QImage.Format_Indexed8)
                            image = apply_orientation(image)

                        elif self.curFrameInfo.packType == 3:
                            if tempImage.loadFromData(frameDataBuf.tobytes(), "png"):
                                image = apply_orientation(tempImage)

                        elif self.curFrameInfo.packType == 4:
                            if tempImage.loadFromData(frameDataBuf.tobytes(), "jpg"):
                                image = apply_orientation(tempImage)

                        # ---- 通过队列发送到主线程显示 ----
                        if image is not None:
                            try:
                                self.image_queue.put_nowait((image, device_name))
                            except queue.Full:
                                pass

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
        text = self.text_gamma.toPlainText().strip()
        if not text:
            print("gamma: 输入为空，跳过")
            return
        temp = float(text)
        ret = self.cmd.sendCMD(pyLhForwardSDK_module.LfCMDInfo.gamma, temp)
        if ret == pyLhForwardSDK_module.LfCMDRet.ok:
            print(f"gamma: {temp}")

    @Slot()
    def setMaxDistance(self):
        text = self.text_distance.toPlainText().strip()
        if not text:
            print("maxDist: 输入为空，跳过")
            return
        temp = float(text)
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
        if self.m_udpSocket is not None:
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
