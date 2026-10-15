# -*- coding: utf-8 -*-
"""
sonar_params.py —— 声呐图像检测参数调试器（单文件版）

用途：
    打开一个蓝衡声呐 .DB 数据文件，实时调节阈值分割 / 连通域参数，
    立刻看到目标计数、面积统计、等效体长，用于确定最优检测参数。

用法：
    python sonar_params.py                    # 弹窗选文件
    python sonar_params.py "D:/data/xxx.DB"   # 直接打开（路径用正斜杠或加引号）

兼容 Python 3.11（与 pyLhForwardSDK_module.cp311-win_amd64.pyd 匹配）。

依赖：
    PySide6, numpy, opencv-python
    以及同目录下的 pyLhForwardSDK_module.cp311-win_amd64.pyd
"""

import sys
import os
import json
import struct
import numpy as np
try:
    from scipy.spatial import cKDTree      # 仅用于可选的个体关联追踪
except ImportError:
    cKDTree = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyLhForwardSDK_module as m
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QTimer

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFileDialog, QTextEdit, QSizePolicy,
    QRadioButton, QComboBox, QScrollArea,
)

try:
    import cv2
except ImportError:
    cv2 = None

PACK_SIZE = m.LhForwardFrame.PACK_SIZE
IMG_W, IMG_H = 968, 512          # 图像显示区尺寸
SIDE_W = 430                     # 右侧控制栏宽度
WIN_W = IMG_W + SIDE_W + 26      # 窗口总宽
WIN_H = IMG_H + 200              # 窗口总高（图像 + 顶部/导航/输出/边距）
INDEX_SUFFIX = ".frameidx2"      # 帧索引缓存文件后缀（v2：记录帧起始包序号）

PACKTYPE_NAME = {
    0: "原始波束数据(需SDK成像)",
    2: "未压缩图像数据",
    3: "PNG压缩图像",
    4: "JPG压缩图像",
    5: "碰撞避让目标列表",
}

# 用于在图像上绘制中文（cv2.putText 不支持中文，会显示成 ???）
_CN_FONT_CACHE = {}


def _gray_from_frame(info, buf, imgobj):
    """把一帧数据转成灰度 ndarray。支持 packType 0/2/3/4。"""
    pt = info.packType
    if pt in (3, 4):
        fmt = "png" if pt == 3 else "jpg"
        im = QImage()
        if not im.loadFromData(buf[:info.dataHeight].tobytes(), fmt):
            return None
        w, h = im.width(), im.height()
        ptr = im.constBits()
        return np.frombuffer(ptr, dtype=np.uint8,
                             count=im.bytesPerLine() * h).reshape(h, im.bytesPerLine())[:, :w].copy()
    if pt == 0:
        lw, lh = IMG_W, IMG_H
        if info.horAngleReso == 360:
            lw = lh = max(IMG_W, IMG_H)
        mem = np.zeros(lw * lh, dtype=np.uint8)
        res = np.array([0.0], dtype=np.float64)
        imgobj.generateForwardImage(info, buf, mem, res)
        return mem.reshape(lh, lw).copy()
    if pt == 2:
        try:
            intensity = np.zeros(max(1, info.numPerRow * info.dataHeight), dtype=np.double)
            imgobj.preprocessData(info, buf, intensity)
            n = info.numPerRow * info.dataHeight
            coef = 1 if info.dataWidth == 0x00 else 256
            return np.clip(intensity[:n] / coef, 0, 255).astype(np.uint8).reshape(
                info.dataHeight, info.numPerRow)
        except Exception:
            return None
    return None


# ===========================================================================
# 3c. 行为识别（内嵌版，逻辑与 sonar_behavior.py 一致）
# ===========================================================================
BEHAVIOR_DEFAULTS = dict(
    threshold=160, min_area=20, kernel=1,
    k=1.0, win=15, sustain=2, min_targets=5,
    flow_win=21, flow_levels=3, fb_max=1.5,
    baseline_ratio=0.20,
)

BEHAVIOR_CN = {"FEEDING": "摄食", "STRESS": "应激", "NORMAL": "正常游动"}


def behavior_extract_features(gray, prev_gray, cfg):
    """一帧的行为特征：目标数 / 离散度D / 最近邻NND / 极化度P / 平均灰度。"""
    _, _, targets, h_img = detect(gray, cfg["threshold"], cfg["min_area"],
                                  cfg["kernel"], True, False)
    n = len(targets)
    feat = dict(n=n, D=0.0, NND=0.0, P=0.0, brightness=0.0,
                flow_mag=0.0, n_tracked=0, cx=0.0, cy=0.0)
    if n == 0:
        return feat, None
    pts = np.array([[t[7], t[6]] for t in targets], dtype=np.float64)
    bright = np.array([t[5] for t in targets], dtype=np.float64)
    C = pts.mean(axis=0)
    feat["cx"], feat["cy"] = float(C[0]), float(C[1])
    feat["D"] = float(np.linalg.norm(pts - C, axis=1).mean())
    feat["brightness"] = float(bright.mean())
    if n >= 2:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        feat["NND"] = float(d.min(axis=1).mean())
    if prev_gray is not None and n >= 2:
        fwd = pts.astype(np.float32).reshape(-1, 1, 2)
        nxt, s1, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, fwd, None,
            winSize=(cfg["flow_win"], cfg["flow_win"]), maxLevel=cfg["flow_levels"])
        if nxt is not None:
            back, s2, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, nxt, None,
                winSize=(cfg["flow_win"], cfg["flow_win"]), maxLevel=cfg["flow_levels"])
            if back is not None:
                fb = np.linalg.norm(fwd - back, axis=2).ravel()
                ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < cfg["fb_max"])
                feat["n_tracked"] = int(ok.sum())
                if ok.sum() >= 2:
                    disp = (nxt - fwd).reshape(-1, 2)[ok]
                    mag = np.linalg.norm(disp, axis=1)
                    feat["flow_mag"] = float(np.median(mag))
                    moving = mag > 0.5
                    if moving.sum() >= 2:
                        u = disp[moving] / mag[moving][:, None]
                        feat["P"] = float(np.linalg.norm(u.mean(axis=0)))
    return feat, pts


def behavior_calibrate_baseline(rows, cfg):
    use = [r for r in rows if r["n"] >= cfg["min_targets"]]
    if len(use) < 5:
        use = [r for r in rows if r["n"] > 0]
    if not use:
        return {k: (0.0, 1.0) for k in ("D", "NND", "P", "brightness")}
    base = {}
    for key in ("D", "NND", "P", "brightness"):
        v = np.array([r[key] for r in use], dtype=float)
        mu, sd = float(v.mean()), float(v.std())
        if sd < 1e-6:
            sd = max(1e-6, 0.1 * abs(mu) if mu else 1.0)
        base[key] = (mu, sd)
    return base


# ===========================================================================
# 3d. 个体运动指标：个体速度 / 个体角速度（光流法）
# ===========================================================================
def flow_individual_metrics(gray_a, gray_b, mpp_m_per_px, fps,
                            bright_thr=110, max_pts=200, min_move_px=0.8):
    """用相邻两帧光流计算个体运动指标（不依赖个体关联）。

    为什么不用个体追踪：
        实测阈值分割的目标在帧间不稳定（同一条鱼的位置会跳），
        最近邻关联后 256 帧只留住 3~5 条轨迹，无法统计。
        光流直接在特征点上测运动，样本量可达数千，量级也符合常识
        （实测个体速度 31~41 cm/s）。

    返回 dict：
        sp_med / sp_p90 : 个体速度(m/s) 中位、P90
        turn_std        : 航向离散度(度) —— 越小越同向
        polar           : 鱼群极化度 [0,1]
        n_pts           : 有效运动特征点数
    无法计算时返回 None。
    """
    if gray_a is None or gray_b is None:
        return None
    mask = (gray_a > bright_thr).astype(np.uint8) * 255
    try:
        pts = cv2.goodFeaturesToTrack(gray_a, maxCorners=max_pts,
                                      qualityLevel=0.03, minDistance=8, mask=mask)
    except Exception:
        return None
    if pts is None or len(pts) < 3:
        return None
    nx, s1, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, pts, None,
                                         winSize=(21, 21), maxLevel=3)
    bk, s2, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, nx, None,
                                         winSize=(21, 21), maxLevel=3)
    if nx is None or bk is None:
        return None
    fb = np.linalg.norm(pts - bk, axis=2).ravel()
    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < 1.5)
    if ok.sum() < 3:
        return None
    d = (nx - pts).reshape(-1, 2)[ok]
    mag = np.linalg.norm(d, axis=1)
    moving = mag > min_move_px
    if moving.sum() < 3:
        return None

    sp = mag[moving] * mpp_m_per_px * fps               # 米/秒
    Hd = np.arctan2(d[moving, 1], d[moving, 0])
    u = d[moving] / mag[moving][:, None]
    return dict(sp_med=float(np.median(sp)),
                sp_p90=float(np.percentile(sp, 90)),
                turn_std=float(np.degrees(np.std(Hd))),   # 航向离散度
                polar=float(np.linalg.norm(u.mean(axis=0))),
                n_pts=int(moving.sum()))


def behavior_classify_rows(rows, baseline, cfg):
    """持续性判据：平滑 z_P > k 且连续 >= sustain 帧 -> 应激；聚集 -> 摄食。"""
    k = float(cfg["k"])
    win = max(1, int(cfg["win"]))
    sustain = max(1, int(cfg["sustain"]))
    min_t = cfg["min_targets"]

    zP, agg = [], []
    for r in rows:
        if r["n"] < min_t:
            zP.append(np.nan); agg.append(np.nan); continue
        z_nnd = (r["NND"] - baseline["NND"][0]) / baseline["NND"][1]
        z_D = (r["D"] - baseline["D"][0]) / baseline["D"][1]
        z_P = (r["P"] - baseline["P"][0]) / baseline["P"][1]
        zP.append(z_P)
        agg.append(-min(z_nnd, z_D))
        r["z_P"] = float(z_P)
        r["z_agg"] = float(-min(z_nnd, z_D))
    zP = np.array(zP, dtype=float)
    agg = np.array(agg, dtype=float)
    filled = np.where(np.isnan(zP), 0.0, zP)
    zs = np.zeros_like(filled)
    for i in range(len(filled)):
        lo = max(0, i - win // 2)
        hi = min(len(filled), i - win // 2 + win)
        zs[i] = filled[lo:hi].mean()

    raw = []
    for i, r in enumerate(rows):
        if r["n"] < min_t or np.isnan(zP[i]):
            raw.append("UNKNOWN"); continue
        if zs[i] > k and agg[i] <= k:
            raw.append("STRESS")
        elif agg[i] > k:
            raw.append("FEEDING")
        else:
            raw.append("NORMAL")

    filt = list(raw)
    i = 0
    while i < len(filt):
        if filt[i] == "STRESS":
            j = i
            while j < len(filt) and filt[j] == "STRESS":
                j += 1
            if (j - i) < sustain:
                for kk in range(i, j):
                    filt[kk] = "FEEDING" if agg[kk] > k else "NORMAL"
            i = j
        else:
            i += 1

    half = win // 2
    for i, r in enumerate(rows):
        lo, hi = max(0, i - half), min(len(filt), i + half + 1)
        seg = [x for x in filt[lo:hi] if x != "UNKNOWN"]
        lab = "UNKNOWN"
        if seg:
            vals, cnts = np.unique(seg, return_counts=True)
            lab = str(vals[int(np.argmax(cnts))])
        r["label_raw"] = raw[i]
        r["label_filt"] = filt[i]
        r["label"] = lab
        r["label_cn"] = BEHAVIOR_CN.get(lab, "未知")
        r["z_P_smooth"] = float(zs[i])
    return rows


def _cn_font(size):
    """找一个能显示中文的字体。"""
    if size in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[size]
    from PIL import ImageFont
    for p in (r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
              r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf",    # 黑体
              r"C:\Windows\Fonts\simsun.ttc"):   # 宋体
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _CN_FONT_CACHE[size] = f
                return f
            except Exception:
                continue
    _CN_FONT_CACHE[size] = None
    return None


def draw_cn_text(img_bgr, text, org, size=26, color=(0, 255, 102)):
    """在 BGR 图上绘制中文（用 PIL 渲染后贴回）。org = 左上角坐标。"""
    font = _cn_font(size)
    if font is None:
        cv2.putText(img_bgr, "targets", org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        return
    try:
        from PIL import Image, ImageDraw
        x, y = int(org[0]), int(org[1])
        rgb = (color[2], color[1], color[0])
        im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im)
        d.text((x, y), text, font=font, fill=rgb)
        img_bgr[:] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img_bgr, "targets", org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)


def draw_count_overlay(vis, n):
    """在图像右下角绘制醒目的鱼群数量框。

    放右下角的原因：右上角是校徽位置，左上角已有参数/规格文字。
    """
    h, w = vis.shape[:2]
    text = "鱼群数量：%d" % n

    font = _cn_font(30)
    if font is not None:
        try:
            box = font.getbbox(text)
            tw, th = box[2] - box[0], box[3] - box[1]
        except Exception:
            tw, th = 12 * len(text), 30
    else:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

    pad = 10
    bw, bh = tw + pad * 2, th + pad * 2
    margin = 14
    x1 = max(0, w - bw - margin)
    y1 = max(0, h - bh - margin)
    x2, y2 = x1 + bw, y1 + bh

    # 半透明底色 + 绿色边框
    roi = vis[y1:y2, x1:x2]
    if roi.size:
        overlay = np.zeros_like(roi)
        overlay[:] = (0, 0, 0)
        cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 102), 2)
    draw_cn_text(vis, text, (x1 + pad, y1 + pad - 2), size=30, color=(0, 255, 102))



# ===========================================================================
# 1. 帧索引：把整个 .DB 扫一遍，记录每一帧的【起始包序号】
# ===========================================================================
def _cache_path(db_path):
    """索引缓存优先放在数据文件旁边；数据盘只读时回退到本机临时目录。"""
    import tempfile, hashlib
    beside = db_path + INDEX_SUFFIX
    if os.path.exists(beside):
        return beside
    try:
        # 探测数据目录是否可写
        probe = db_path + INDEX_SUFFIX + ".tmp"
        with open(probe, "wb") as f:
            f.write(b"0")
        os.remove(probe)
        return beside
    except Exception:
        h = hashlib.md5(os.path.abspath(db_path).encode("utf-8", "replace")).hexdigest()[:16]
        d = os.path.join(tempfile.gettempdir(), "sonar_frameidx")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, h + INDEX_SUFFIX)


def build_index(db_path, log=print):
    """返回 packet_starts，其中 packet_starts[i] = 第 i 帧首包在文件中的包序号。

    SDK 的 writeOnePackData 在「一帧写完」时返回 oneFrameOK，
    此时收到的包正是该帧的【最后一个包】。所以下一帧从它的下一个包开始。
    """
    cache = _cache_path(db_path)
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                n = struct.unpack("<I", f.read(4))[0]
                starts = list(struct.unpack("<%dI" % n, f.read(4 * n)))
            log("已读取帧索引缓存: %d 帧" % n)
            return starts
        except Exception:
            pass

    log("正在建立帧索引（首次打开需要几秒，500MB 文件约 5 秒）...")
    starts = [0]                      # 第 0 帧从第 0 个包开始
    frame = m.LhForwardFrame()
    info = m.packageInfo()
    buf = np.zeros(1024 * 1024 * 10, dtype=np.uint8)

    pkt = 0
    with open(db_path, "rb") as f:
        while True:
            raw = f.read(PACK_SIZE)
            if len(raw) < PACK_SIZE:
                break
            if frame.writeOnePackData(np.frombuffer(raw, dtype=np.uint8)) == m.LfFrameRet.oneFrameOK:
                starts.append(pkt + 1)   # 下一帧从下一个包开始
                frame.getOneFrame(info, buf)
            pkt += 1
    if len(starts) > 1:
        starts.pop()                  # 最后一个元素是文件尾，不是帧起点

    try:
        with open(cache, "wb") as f:
            f.write(struct.pack("<I", len(starts)))
            f.write(struct.pack("<%dI" % len(starts), *starts))
        log("帧索引已缓存: %d 帧  -> %s" % (len(starts), cache))
    except Exception as e:
        log("索引缓存写入失败（不影响使用，下次仍需重新扫描）: %s" % e)
    return starts


# ===========================================================================
# 2. 解码线程：后台按需解码帧，主线程不卡
# ===========================================================================
class Decoder(QThread):
    frameReady = Signal(int, object, object)   # (帧号, 灰度ndarray, meta dict)
    seekDone = Signal(int)                     # (总帧数)
    failed = Signal(str)

    def __init__(self, db_path, starts, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.starts = starts
        self._want = None
        self._running = True
        self._img = None          # LhForwardImage，packType 0/2 成像用（惰性创建）

    def request(self, idx):
        self._want = idx

    def stop(self):
        self._running = False
        self._want = None

    def run(self):
        self.seekDone.emit(len(self.starts))
        while self._running:
            idx = self._want
            if idx is None:
                self.msleep(10)
                continue
            self._want = None
            try:
                self._decode(idx)
            except Exception as e:
                import traceback
                self.failed.emit("帧 %d 解码异常: %s\n%s" % (idx, e, traceback.format_exc()))

    def _decode(self, idx):
        if idx < 0 or idx >= len(self.starts):
            return
        if self._img is None:
            self._img = m.LhForwardImage(IMG_W, IMG_H)

        # 关键：每帧用一个【全新的】LhForwardFrame。
        # SDK 的帧边界状态只有在"从某帧首包开始喂"时才自洽；
        # 复用同一个对象（或调用 init()）会让帧边界与索引错位，
        # 导致某一帧只喂几个包就返回 oneFrameOK（实测帧0只用了4包/应74包）。
        frame = m.LhForwardFrame()
        info = m.packageInfo()
        buf = np.zeros(1024 * 1024 * 10, dtype=np.uint8)

        start = self.starts[idx]
        # 从【上一帧起点】开始喂，保证新对象能恰好在本帧末包触发 oneFrameOK
        if idx > 0:
            read_from = self.starts[idx - 1]
        else:
            read_from = 0
        # 包数上限：给足余量，避免因丢弃/重复包导致提前退出
        if idx + 1 < len(self.starts):
            max_packs = (self.starts[idx + 1] - read_from) + 16
        else:
            max_packs = 600

        with open(self.db_path, "rb") as f:
            f.seek(read_from * PACK_SIZE)
            n_packs = 0                     # 本帧实际消耗的包数
            for _ in range(max_packs):
                if not self._running:
                    return
                raw = f.read(PACK_SIZE)
                if len(raw) < PACK_SIZE:
                    break
                n_packs += 1
                if frame.writeOnePackData(np.frombuffer(raw, dtype=np.uint8)) != m.LfFrameRet.oneFrameOK:
                    continue
                frame.getOneFrame(info, buf)
                gray = self._to_gray(info, buf)
                if gray is None:
                    self.failed.emit("帧 %d 无法成像（packType=%d）" % (idx, info.packType))
                    break
                # 本帧实际包数 = 索引给出的跨度（最可靠），而非喂入次数
                span = None
                if idx + 1 < len(self.starts):
                    span = self.starts[idx + 1] - start
                meta = dict(frame=idx, w=gray.shape[1], h=gray.shape[0], packType=info.packType,
                            maxDist=info.maxDist, distReso=info.distReso,
                            hor=info.horAngleReso, vel=info.acousticVelocity,
                            workMode=info.workMode,
                            n_packs=span if span else n_packs,
                            said_pack_num=info.packNumPerFrame,
                            pack_size=PACK_SIZE,
                            num_per_row=info.numPerRow,
                            work_cycle=float(info.workCycle),
                            data_height=info.dataHeight,
                            data_width=info.dataWidth,
                            device=str(info.deviceName) if isinstance(info.deviceName, str)
                            else info.deviceName.decode("utf-8", "replace").rstrip("\x00"))
                self.frameReady.emit(idx, gray, meta)
                break

    def _to_gray(self, info, buf):
        """把一帧数据转成灰度 ndarray（宽 x 高）。支持 packType 0/2/3/4。"""
        pt = info.packType

        # ---- packType 3/4：设备已压好的 PNG / JPG ----
        if pt in (3, 4):
            fmt = "png" if pt == 3 else "jpg"
            im = QImage()
            if not im.loadFromData(buf[:info.dataHeight].tobytes(), fmt):
                return None
            w, h = im.width(), im.height()
            ptr = im.constBits()
            return np.frombuffer(ptr, dtype=np.uint8,
                                 count=im.bytesPerLine() * h).reshape(h, im.bytesPerLine())[:, :w].copy()

        # ---- packType 0：原始波束数据 → 用 SDK 插值成扇形图 ----
        if pt == 0:
            panSize = IMG_W if IMG_W > IMG_H else IMG_H
            lw, lh = IMG_W, IMG_H
            if info.horAngleReso == 360:
                lw = lh = panSize
            mem = np.zeros(lw * lh, dtype=np.uint8)
            res = np.array([0.0], dtype=np.float64)
            self._img.generateForwardImage(info, buf, mem, res)
            img = mem.reshape(lh, lw)
            # 与原厂 widget.py 一致：等比缩放到显示高度后居中裁切
            if img.shape[0] > IMG_H:
                s = IMG_H / img.shape[0]
                nw = max(1, int(round(img.shape[1] * s)))
                img = cv2.resize(img, (nw, IMG_H), interpolation=cv2.INTER_AREA)
            ih, iw = img.shape
            sx = max(0, (iw - IMG_W) // 2)
            return img[:, sx:sx + IMG_W].copy() if iw >= IMG_W else img

        # ---- packType 2：未压缩灰度数据 ----
        if pt == 2:
            intensity = np.zeros(max(1, info.numPerRow * info.dataHeight), dtype=np.double)
            try:
                self._img.preprocessData(info, buf, intensity)
            except Exception:
                return None
            n = info.numPerRow * info.dataHeight
            coef = 1 if info.dataWidth == 0x00 else 256
            g = np.clip(intensity[:n] / coef, 0, 255).astype(np.uint8)
            return g.reshape(info.dataHeight, info.numPerRow)

        return None


# ===========================================================================
# 3. 检测：阈值 + 形态学 + 8邻域连通域（全图统计）
# ===========================================================================
def detect(gray, thr, min_area, kernel, use_open, use_close):
    """返回 (标注图BGR, mask, 目标列表[(x,y,w,h,area,mean,cy,cx)], h_img)"""
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    if kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        if use_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if use_close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, labels, st, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h_img = gray.shape[0]
    targets = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = (int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                      int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]))
        comp = (labels[y:y + h, x:x + w] == i)
        mean_v = float(gray[y:y + h, x:x + w][comp].mean())
        ys, xs = comp.nonzero()
        cy = y + (float(ys.mean()) if len(ys) else h / 2.0)   # 质心纵向位置
        cx = x + (float(xs.mean()) if len(xs) else w / 2.0)   # 质心横向位置
        targets.append((x, y, w, h, area, mean_v, cy, cx))
    return vis, mask, targets, h_img


# ===========================================================================
# 3b. YOLO / ONNX 检测（可选）
# ===========================================================================
def find_onnx_models():
    """在整个工作区里搜索 .onnx 权重。

    搜索顺序：
      1. 用户在配置文件里指定的权重目录（可多个）
      2. 脚本所在目录 / 上一级 / 工作区根目录
    目录内容变化时会自动重新扫描。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = []

    # 1) 配置文件里的目录（持久化用户选择）
    cfg = _load_config()
    for d in cfg.get("model_dirs", []):
        if d and os.path.isdir(d):
            dirs.append(d)

    # 2) 默认目录
    for d in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if d and os.path.isdir(d):
            dirs.append(d)

    found = []
    for d in dirs:
        try:
            for fn in os.listdir(d):
                if fn.lower().endswith(".onnx"):
                    p = os.path.join(d, fn)
                    if os.path.isfile(p):
                        found.append(p)
        except Exception:
            pass
    # 按文件大小+名字去重（工作区里存在同名副本，避免下拉框出现重复项）
    uniq = {}
    for p in found:
        try:
            key = (os.path.basename(p).lower(), os.path.getsize(p))
        except Exception:
            key = (p.lower(), 0)
        if key not in uniq:
            uniq[key] = p
    found = list(uniq.values())
    # 按修改时间倒序（最新的排前面）
    found.sort(key=lambda p: -os.path.getmtime(p))
    return found


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sonar_params_config.json")


def _load_config():
    # 注意：用 utf-8-sig 读取。Windows 上用记事本/PowerShell 保存的 JSON
    # 常带 BOM，用 utf-8 读会让 json.load 失败，导致配置静默丢失。
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(CONFIG_PATH, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return {}


def _save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print("[配置] 保存失败:", e)
        return False


class YoloDetector:
    """封装 ONNX Runtime 推理。

    支持两种常见的 YOLO 导出格式（自动识别）：
      格式 A（已内置 NMS，端到端导出）: 输出 [1, N, 6]  -> 每行 [x1,y1,x2,y2,conf,cls]
      格式 B（原始 YOLOv8 输出）      : 输出 [1, 4+nc, N] -> 需自己做解码 + NMS
    """

    def __init__(self, model_path, conf_thr=0.35, iou_thr=0.45, imgsz=640):
        import onnxruntime as ort
        self.model_path = model_path
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.imgsz = int(imgsz)
        self.name = os.path.basename(model_path)

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.in_shape = self.sess.get_inputs()[0].shape       # [1,3,H,W]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.out_shape = self.sess.get_outputs()[0].shape
        # 输入尺寸可能不是 640
        try:
            if isinstance(self.in_shape[2], int) and self.in_shape[2] > 0:
                self.imgsz = self.in_shape[2]
        except Exception:
            pass
        self.format = self._guess_format()

    def _guess_format(self):
        s = self.out_shape
        if len(s) == 3 and isinstance(s[2], int) and s[2] == 6:
            return "A_NMS"          # [1, N, 6]
        if len(s) == 3 and isinstance(s[1], int) and s[1] >= 5 and s[1] < 512:
            return "B_RAW"          # [1, 4+nc, N]
        return "A_NMS"              # 默认按端到端处理

    def infer(self, gray_bgr):
        """输入 BGR 图，返回 (boxes, scores) —— boxes 为原图坐标 (x1,y1,x2,y2)"""
        h0, w0 = gray_bgr.shape[:2]
        S = self.imgsz
        blob = cv2.dnn.blobFromImage(gray_bgr, 1 / 255.0, (S, S),
                                     swapRB=True, crop=False)
        out = self.sess.run(self.out_names, {self.in_name: blob})[0]
        sx, sy = w0 / float(S), h0 / float(S)

        boxes, scores, class_ids = [], [], []

        if self.format == "A_NMS":
            arr = out[0] if out.ndim == 3 else out     # [N, 6]
            for row in arr:
                if len(row) < 6:
                    continue
                conf = float(row[4])
                if conf < self.conf_thr:
                    continue
                x1, y1, x2, y2 = row[0] * sx, row[1] * sy, row[2] * sx, row[3] * sy
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                class_ids.append(int(row[5]) if len(row) > 5 else 0)

        else:  # B_RAW: [1, 4+nc, N]
            pred = out[0]                    # [4+nc, N]
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T                # -> [N, 4+nc]
            xywh = pred[:, :4]
            cls_scores = pred[:, 4:]
            ids = cls_scores.argmax(axis=1)
            confs = cls_scores[np.arange(len(ids)), ids]
            keep = confs >= self.conf_thr
            xywh, confs, ids = xywh[keep], confs[keep], ids[keep]
            if len(xywh):
                cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
                rects = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
                idx = cv2.dnn.NMSBoxes(rects.tolist(), confs.tolist(),
                                       self.conf_thr, self.iou_thr)
                idx = np.array(idx).reshape(-1) if len(idx) else np.array([], dtype=int)
                for k in idx:
                    x1 = (cx[k] - w[k] / 2) * sx
                    y1 = (cy[k] - h[k] / 2) * sy
                    x2 = (cx[k] + w[k] / 2) * sx
                    y2 = (cy[k] + h[k] / 2) * sy
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confs[k]))
                    class_ids.append(int(ids[k]))

        return boxes, scores, class_ids


def detect_yolo(gray, yolo, draw=True):
    """对灰度图跑 YOLO，返回 (标注图, 目标列表, h_img)。

    目标列表格式与传统检测保持一致：(x, y, w, h, area, mean_gray, cy, cx)
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h_img, w_img = gray.shape[:2]
    targets = []
    if yolo is None:
        return vis, targets, h_img
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    boxes, scores, class_ids = yolo.infer(bgr)
    for (x1, y1, x2, y2), sc, cid in zip(boxes, scores, class_ids):
        ix1 = max(0, int(round(x1)))
        iy1 = max(0, int(round(y1)))
        ix2 = min(w_img, int(round(x2)))
        iy2 = min(h_img, int(round(y2)))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        roi = gray[iy1:iy2, ix1:ix2]
        area = int(roi.size)
        mean_v = float(roi.mean()) if area else 0.0
        cy = (iy1 + iy2) / 2.0
        cx = (ix1 + ix2) / 2.0
        targets.append((ix1, iy1, ix2 - ix1, iy2 - iy1, area, mean_v, cy, cx))
        if draw:
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
            cv2.putText(vis, "%.2f" % sc, (ix1, max(12, iy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vis, targets, h_img



# ===========================================================================
# 4. 主界面
# ===========================================================================
class ParamTuner(QWidget):
    def __init__(self, db_path=None):
        super().__init__()
        self.setWindowTitle("声呐图像检测参数调试器  —  Sonar Detection Parameter Tuner")
        self.setFixedSize(WIN_W, WIN_H)

        self.db_path = None
        self.starts = []
        self.cur_idx = 0
        self.cur_gray = None
        self.cur_meta = None
        self.decoder = None
        self.watchdog = None
        self._pending_since = None
        self._avg_packs = 0.0
        self.yolo = None              # YoloDetector 实例（惰性加载）
        # ---- 三个识别功能的开关状态 ----
        self.do_count = True          # 数量识别（默认开）
        self.do_length = False        # 体长识别
        self.do_behavior = False      # 行为识别
        self.length_manual = False    # 体长是否用手动比例
        self.length_mm_px = None      # 手动比例（mm/像素）
        self.behavior_rows = []       # 行为分析结果
        self.behavior_label = "--"
        self.behavior_detail = ""
        self.behavior_minor = ""
        self.behavior_events = []
        # 个体运动指标（光流法）
        self.ind_speed_med = float("nan")
        self.ind_speed_p90 = float("nan")
        self.ind_turn_std = float("nan")
        self.flow_polar = float("nan")
        self.ind_npts = 0
        self.behavior_baseline_path = (_load_config().get("behavior_baseline") or "")
        self.behavior_baseline_src = ""
        self._baseline_cache = None
        self._last_vis = None         # 最近一次的标注图（供体长换算复用）
        self._last_targets = []       # 最近一帧的检测目标（供体长标定用）
        self._last_h_img = 512

        self._build_ui()

        if db_path and os.path.exists(db_path):
            QTimer.singleShot(100, lambda: self.load_file(db_path))

    # ---------------- UI ----------------
    def _build_ui(self):
        # 图像显示
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(IMG_W, IMG_H)
        self.image_label.setStyleSheet("background-color: #101010;")
        self.image_label.setAlignment(Qt.AlignCenter)

        # ---- 右上角：当前帧鱼群数量（覆盖在图像上）----
        self.lbl_count = QLabel(self.image_label)
        self.lbl_count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_count.setStyleSheet(
            "QLabel {"
            "  color: #00FF66;"
            "  background-color: rgba(0, 0, 0, 170);"
            "  border: 2px solid #00FF66;"
            "  border-radius: 8px;"
            "  padding: 8px 16px;"
            "  font-family: 'Microsoft YaHei', Consolas, monospace;"
            "  font-size: 20px;"
            "  font-weight: bold;"
            "}")
        self.lbl_count.setText("鱼群数量：--")
        self.lbl_count.adjustSize()
        self.lbl_count.move(IMG_W - self.lbl_count.width() - 12, 12)
        self.lbl_count.raise_()

        # 文件行
        self.lbl_file = QLabel("未加载文件")
        self.lbl_file.setStyleSheet("color:#888;")
        btn_open = QPushButton("打开 .DB 文件")
        btn_open.setFixedWidth(130)
        btn_open.clicked.connect(self.choose_file)

        # 帧导航
        self.lbl_frame = QLabel("帧 0 / 0")
        self.lbl_frame.setFixedWidth(160)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider)
        btn_prev = QPushButton("◀ 上一帧")
        btn_next = QPushButton("下一帧 ▶")
        btn_prev.clicked.connect(lambda: self.goto(self.cur_idx - 1))
        btn_next.clicked.connect(lambda: self.goto(self.cur_idx + 1))

        # ---- 播放控制（放在图像下方，不遮挡画面）----
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setCheckable(True)
        self.btn_play.setFixedWidth(96)
        self.btn_play.setStyleSheet(
            "QPushButton:checked{background-color:#2c9678;color:white;font-weight:bold;}")
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setFixedWidth(84)
        self.btn_stop.clicked.connect(self.stop_play)

        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(1, 40)
        self.sp_fps.setValue(10)
        self.sp_fps.setFixedWidth(64)
        self.sp_fps.setSuffix(" fps")
        self.sp_fps.setToolTip("播放帧率（声呐原始为 40 fps）")
        self.sp_fps.valueChanged.connect(self.on_fps_changed)

        self.cb_loop = QCheckBox("循环")
        self.cb_loop.setChecked(True)

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.on_play_tick)

        # 参数
        self.sp_thr = QSpinBox(); self.sp_thr.setRange(1, 255); self.sp_thr.setValue(160)
        self.sp_min = QSpinBox(); self.sp_min.setRange(1, 100000); self.sp_min.setValue(20)
        self.sp_k = QSpinBox(); self.sp_k.setRange(0, 31); self.sp_k.setValue(0); self.sp_k.setSingleStep(2)
        self.cb_open = QCheckBox("开运算(去碎块)"); self.cb_open.setChecked(True)
        self.cb_close = QCheckBox("闭运算(补空洞)"); self.cb_close.setChecked(False)
        self.cb_boxes = QCheckBox("显示检测框"); self.cb_boxes.setChecked(True)
        self.cb_mask = QCheckBox("显示二值掩膜"); self.cb_mask.setChecked(False)
        self.cb_flip = QCheckBox("上下翻转显示"); self.cb_flip.setChecked(True)
        self.cb_flip.setToolTip("只影响显示方向，不影响检测统计\n（C++ 版 mainwindow.cpp 里就是上下翻转的）")

        for w in (self.sp_thr, self.sp_min, self.sp_k):
            w.setFixedWidth(90)
        for w in (self.sp_thr, self.sp_min, self.sp_k):
            w.valueChanged.connect(self.redetect)
        for w in (self.cb_open, self.cb_close, self.cb_boxes, self.cb_mask, self.cb_flip):
            w.stateChanged.connect(self.redetect)

        gb_param = QGroupBox("检测参数（实时生效）")
        gl = QVBoxLayout(gb_param)
        gl.setSpacing(4)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("灰度阈值 (>):"))
        r1.addWidget(self.sp_thr, 1)
        gl.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("最小面积 (像素):"))
        r2.addWidget(self.sp_min, 1)
        gl.addLayout(r2)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("形态学核 (0=关闭):"))
        r3.addWidget(self.sp_k, 1)
        gl.addLayout(r3)
        r4 = QHBoxLayout()
        r4.addWidget(self.cb_open); r4.addWidget(self.cb_close)
        gl.addLayout(r4)
        r5 = QHBoxLayout()
        r5.addWidget(self.cb_boxes); r5.addWidget(self.cb_mask)
        gl.addLayout(r5)
        r6 = QHBoxLayout()
        r6.addWidget(self.cb_flip)
        gl.addLayout(r6)

        # 帧导航改为左栏内联排布，不再单独用 GroupBox（节省纵向空间）

        # ---------- 检测模式 + YOLO 权重 ----------
        self.rb_thr = QRadioButton("阈值分割"); self.rb_thr.setChecked(True)
        self.rb_yolo = QRadioButton("YOLO 模型")
        self.rb_both = QRadioButton("两者对比")
        self.rb_thr.toggled.connect(self.redetect)
        self.rb_yolo.toggled.connect(self.redetect)
        self.rb_both.toggled.connect(self.redetect)

        self.cb_model = QComboBox()
        self.cb_model.setMinimumWidth(200)
        self.cb_model.setToolTip("下拉选择权重；点「选择权重目录…」可添加其它目录")
        self._models = []
        # 先填一个占位，稍后由 refresh_models() 真正扫描
        self.cb_model.addItem("(正在扫描权重…)", None)

        self.sp_conf = QDoubleSpinBox(); self.sp_conf.setRange(0.01, 0.99)
        self.sp_conf.setSingleStep(0.05); self.sp_conf.setValue(0.35)
        self.sp_conf.setDecimals(2); self.sp_conf.setFixedWidth(80)
        self.sp_iou = QDoubleSpinBox(); self.sp_iou.setRange(0.01, 0.99)
        self.sp_iou.setSingleStep(0.05); self.sp_iou.setValue(0.45)
        self.sp_iou.setDecimals(2); self.sp_iou.setFixedWidth(80)

        self.btn_load_model = QPushButton("加载/重载权重")
        self.btn_load_model.setFixedWidth(130)
        self.btn_load_model.clicked.connect(self.reload_model)

        self.btn_model_dir = QPushButton("选择权重目录…")
        self.btn_model_dir.setFixedWidth(130)
        self.btn_model_dir.clicked.connect(self.choose_model_dir)

        self.lbl_model = QLabel("权重未加载")
        self.lbl_model.setStyleSheet("color:#888;")

        self.sp_conf.valueChanged.connect(self.redetect)
        self.sp_iou.valueChanged.connect(self.redetect)
        self.cb_model.currentIndexChanged.connect(self.reload_model)
        for r in (self.rb_thr, self.rb_yolo, self.rb_both):
            r.toggled.connect(self._sync_mode_enabled)

        # 统计
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMinimumHeight(150)
        self.txt.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")

        # ==================== 布局：左侧图片（完整显示） + 右侧控制栏 ====================
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---- 左栏：文件 + 图像（完整 968x512，不缩放） + 帧导航 ----
        left = QVBoxLayout()
        left.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(btn_open)
        top.addWidget(self.lbl_file, 1)
        left.addLayout(top)
        left.addWidget(self.image_label, 0, Qt.AlignTop)

        nav = QHBoxLayout()
        nav.addWidget(self.btn_play)          # ▶ 播放
        nav.addWidget(self.btn_stop)          # ■ 停止
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        nav.addWidget(QLabel("速度:"))
        nav.addWidget(self.sp_fps)
        nav.addWidget(self.cb_loop)
        nav.addStretch(1)
        nav.addWidget(self.lbl_frame)
        left.addLayout(nav)
        left.addWidget(self.slider)
        left.addStretch(1)
        root.addLayout(left, 0)

        # ==================== 右栏 ====================
        # 右侧用一个可滚动区域，避免控件过多时被挤压
        side_inner = QWidget()
        side = QVBoxLayout(side_inner)
        side.setContentsMargins(2, 2, 2, 2)
        side.setSpacing(8)

        # ---------- ① 三个识别功能开关 ----------
        gb_recog = QGroupBox("识别功能（可同时开启）")
        rl = QVBoxLayout(gb_recog)
        rl.setSpacing(6)

        self.btn_count = QPushButton("🐟  数量识别")
        self.btn_length = QPushButton("📏  体长识别")
        self.btn_behavior = QPushButton("🏊  行为识别")
        for b in (self.btn_count, self.btn_length, self.btn_behavior):
            b.setCheckable(True)
            b.setMinimumHeight(38)
            b.setStyleSheet(
                "QPushButton{text-align:left;padding-left:12px;font-size:14px;}"
                "QPushButton:checked{background-color:#2c9678;color:white;font-weight:bold;}")
        self.btn_count.setToolTip("统计当前帧的目标个数（跟随所选检测算法）")
        self.btn_length.setToolTip("测量每个目标的体长（按量程标定换算）")
        self.btn_behavior.setToolTip("分析摄食/应激/正常游动（用当前帧附近的窗口）")
        self.btn_count.setChecked(True)
        # 注意：用 toggled 而不是 clicked。
        # clicked 与 setChecked 组合会把状态反转（程序改状态时也触发一次）。
        self.btn_count.toggled.connect(self.on_recog_toggle)
        self.btn_length.toggled.connect(self.on_recog_toggle)
        self.btn_behavior.toggled.connect(self.on_recog_toggle)
        rl.addWidget(self.btn_count)
        rl.addWidget(self.btn_length)
        rl.addWidget(self.btn_behavior)

        # 行为识别需要时间窗口
        # ---- 滑动时间窗口（按秒设定）----
        # 声呐原始帧率 40 fps（workCycle=0.025s），这里按秒换算成帧
        brow = QHBoxLayout()
        self.sp_bwin = QDoubleSpinBox()
        self.sp_bwin.setRange(0.5, 120.0)
        self.sp_bwin.setSingleStep(1.0)
        self.sp_bwin.setDecimals(1)
        self.sp_bwin.setValue(30.0)
        self.sp_bwin.setFixedWidth(78)
        self.sp_bwin.setSuffix(" s")
        self.sp_bwin.setToolTip(
            "滑动时间窗口长度（秒）。窗口越长，越能包含完整的投食/惊扰过程。\n"
            "按帧率换算：1 秒 = 40 帧")
        brow.addWidget(QLabel("时间窗口:"))
        brow.addWidget(self.sp_bwin)
        self.lbl_win_frames = QLabel("= 40 帧")
        self.lbl_win_frames.setStyleSheet("color:#888;font-size:11px;")
        brow.addWidget(self.lbl_win_frames)
        brow.addStretch(1)
        rl.addLayout(brow)

        # 抽样步长：窗口大时不必每帧都解码
        brow2 = QHBoxLayout()
        self.sp_bstride = QSpinBox()
        self.sp_bstride.setRange(1, 20)
        self.sp_bstride.setValue(8)
        self.sp_bstride.setFixedWidth(60)
        self.sp_bstride.setToolTip(
            "窗口内每隔几帧取一帧分析。\n"
            "调大可以显著加快大窗口的分析速度，但会减少样本数。")
        brow2.addWidget(QLabel("抽样间隔:"))
        brow2.addWidget(self.sp_bstride)
        brow2.addWidget(QLabel("帧"))
        brow2.addStretch(1)
        rl.addLayout(brow2)

        self.sp_bwin.valueChanged.connect(self._on_bwin_changed)
        self.sp_bstride.valueChanged.connect(self.schedule_behavior)
        self.frame_rate = 40.0          # 声呐标称帧率，可由包头 workCycle 更新

        # 行为基线（用"正常游动"数据作参照，强烈推荐设置）
        br2 = QHBoxLayout()
        self.btn_bbase = QPushButton("设置行为基线…")
        self.btn_bbase.setToolTip(
            "选择一段【正常游动】数据作为参照。\n"
            "必须设置：像惊扰这种整段都应激的数据，\n"
            "窗口内自己比自己会把应激误判为正常。")
        self.btn_bbase.clicked.connect(self.choose_behavior_baseline)
        br2.addWidget(self.btn_bbase)
        br2.addStretch(1)
        rl.addLayout(br2)
        self.lbl_bbase = QLabel("基线: 未设置（窗口内相对基线）")
        self.lbl_bbase.setWordWrap(True)
        self.lbl_bbase.setStyleSheet("color:#c66;font-size:11px;")
        rl.addWidget(self.lbl_bbase)
        side.addWidget(gb_recog)

        # ---------- ② 体长标定 ----------
        gb_cal = QGroupBox("体长标定")
        cl = QVBoxLayout(gb_cal)
        cl.setSpacing(4)
        self.lbl_cal = QLabel()
        self.lbl_cal.setWordWrap(True)
        self.lbl_cal.setStyleSheet("color:#2c9678;font-size:11px;")
        cl.addWidget(self.lbl_cal)

        crow1 = QHBoxLayout()
        crow1.addWidget(QLabel("已知体长(cm):"))
        self.sp_gt = QDoubleSpinBox()
        self.sp_gt.setRange(0.0, 999.0)
        self.sp_gt.setDecimals(1)
        self.sp_gt.setValue(25.0)
        self.sp_gt.setFixedWidth(80)
        self.sp_gt.setToolTip("填你确知的鱼体长，点「用已知体长标定」自动反推比例")
        crow1.addWidget(self.sp_gt)
        self.btn_cal_gt = QPushButton("用已知体长标定")
        self.btn_cal_gt.clicked.connect(self.calibrate_by_gt)
        crow1.addWidget(self.btn_cal_gt, 1)
        cl.addLayout(crow1)

        crow2 = QHBoxLayout()
        self.cb_use_manual = QCheckBox("手动比例")
        self.cb_use_manual.setToolTip("勾选后使用下面的 mm/像素，而不是几何标定")
        self.sp_mmpx = QDoubleSpinBox()
        self.sp_mmpx.setRange(0.01, 500.0)
        self.sp_mmpx.setDecimals(3)
        self.sp_mmpx.setFixedWidth(90)
        self.sp_mmpx.setValue(3.320)
        crow2.addWidget(self.cb_use_manual)
        crow2.addWidget(self.sp_mmpx)
        crow2.addWidget(QLabel("mm/像素"))
        crow2.addStretch(1)
        cl.addLayout(crow2)

        self.cb_use_manual.toggled.connect(self.on_cal_changed)
        self.sp_mmpx.valueChanged.connect(self.on_cal_changed)

        self.cb_len_show = QCheckBox("在图上标注体长")
        self.cb_len_show.setChecked(True)
        self.cb_len_show.toggled.connect(self.redetect)
        cl.addWidget(self.cb_len_show)
        side.addWidget(gb_cal)

        # ---------- ③ 检测参数 ----------
        side.addWidget(gb_param)

        # ---------- ④ 检测算法 / YOLO ----------
        gb_yolo = QGroupBox("检测算法")
        yl = QVBoxLayout(gb_yolo)
        yl.setSpacing(4)
        yrow = QHBoxLayout()
        yrow.addWidget(self.rb_thr)
        yrow.addWidget(self.rb_yolo)
        yrow.addWidget(self.rb_both)
        yl.addLayout(yrow)

        row_m = QHBoxLayout()
        row_m.addWidget(QLabel("权重:"))
        row_m.addWidget(self.cb_model, 1)
        yl.addLayout(row_m)

        row_l = QHBoxLayout()
        row_l.addWidget(self.btn_load_model)
        row_l.addWidget(self.btn_model_dir)
        yl.addLayout(row_l)

        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("置信度:"))
        row_c.addWidget(self.sp_conf)
        row_c.addWidget(QLabel("IoU:"))
        row_c.addWidget(self.sp_iou)
        yl.addLayout(row_c)

        self.lbl_model.setWordWrap(True)
        yl.addWidget(self.lbl_model)
        side.addWidget(gb_yolo)

        side.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side_inner)
        scroll.setFixedWidth(SIDE_W)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll, 1)

        # ---------- ⑤ 识别结果（放左栏图像下方，横向更宽，便于看统计） ----------
        self.txt.setFixedHeight(130)
        left.addWidget(QLabel("识别结果"))
        left.addWidget(self.txt)

        self.lbl_model.setText("权重未加载（切换模式时会自动加载）")
        # 真正扫描权重目录（含配置文件里记住的目录）
        self.refresh_models()
        # 初始化体长标定显示与行为定时器
        self.behavior_timer = QTimer(self)
        self.behavior_timer.setSingleShot(True)
        self.behavior_timer.timeout.connect(self.run_behavior)
        self.behavior_rows = []
        self.behavior_label = "--"
        self.behavior_detail = ""
        self.on_cal_changed()
        # 显示已记住的行为基线
        if self.behavior_baseline_path and os.path.exists(self.behavior_baseline_path):
            self.lbl_bbase.setText("基线: %s" % os.path.basename(self.behavior_baseline_path))
            self.lbl_bbase.setStyleSheet("color:#2c9678;font-size:11px;")

    # ---------------- YOLO 权重 ----------------
    def choose_model_dir(self):
        """选择权重所在目录，并记住它（写入 sonar_params_config.json）。"""
        start = ""
        cur = self.cb_model.currentData()
        if cur:
            start = os.path.dirname(cur)
        d = QFileDialog.getExistingDirectory(self, "选择 YOLO 权重(.onnx)所在目录", start)
        if not d:
            return
        cfg = _load_config()
        dirs = cfg.get("model_dirs", [])
        if d not in dirs:
            dirs.insert(0, d)
        cfg["model_dirs"] = dirs[:10]
        _save_config(cfg)
        self.refresh_models()
        self.lbl_model.setText("已添加权重目录: %s" % d)
        self.lbl_model.setStyleSheet("color:#2c9678;")

    def refresh_models(self):
        """重新扫描权重并刷新下拉框，尽量保持当前选中项。"""
        cur = self.cb_model.currentData()
        self._models = find_onnx_models()
        self.cb_model.blockSignals(True)
        self.cb_model.clear()
        if not self._models:
            self.cb_model.addItem("(未找到 .onnx 权重，请点右侧「选择权重目录」)", None)
        else:
            for p in self._models:
                # 显示目录尾名，便于区分同名权重
                parent = os.path.basename(os.path.dirname(p))
                self.cb_model.addItem("%s  [%s]  %.2f MB" % (
                    os.path.basename(p), parent,
                    os.path.getsize(p) / 1024 / 1024), p)
            # 恢复之前选中项；否则优先选实测可用的
            idx = 0
            if cur:
                for i, p in enumerate(self._models):
                    if os.path.normcase(p) == os.path.normcase(cur):
                        idx = i
                        break
            else:
                for i, p in enumerate(self._models):
                    if "6_5_school" in os.path.basename(p):
                        idx = i
                        break
            self.cb_model.setCurrentIndex(idx)
        self.cb_model.blockSignals(False)

    def _sync_mode_enabled(self):
        """按所选模式启用/禁用 YOLO 相关控件。"""
        pure_thr = self.rb_thr.isChecked()
        for w in (self.cb_model, self.sp_conf, self.sp_iou):
            w.setEnabled(not pure_thr)
        self.btn_load_model.setEnabled(not pure_thr)
        self.btn_model_dir.setEnabled(True)      # 始终允许改权重目录
        self.redetect()

    def reload_model(self):
        path = self.cb_model.currentData()
        if not path:
            self.lbl_model.setText("未找到可用的 .onnx 权重")
            self.lbl_model.setStyleSheet("color:#c66;")
            return
        # 已经加载同一个文件且参数一致，就不重复加载
        if (self.yolo is not None and
                os.path.normcase(self.yolo.model_path) == os.path.normcase(path)):
            self.yolo.conf_thr = self.sp_conf.value()
            self.yolo.iou_thr = self.sp_iou.value()
            self.redetect()
            return
        try:
            import time as _t
            t0 = _t.time()
            self.yolo = YoloDetector(path, self.sp_conf.value(),
                                     self.sp_iou.value())
            dt = (_t.time() - t0) * 1000
            self.lbl_model.setText("✅ %s | 格式=%s | 输入=%d | 加载 %.0f ms" % (
                self.yolo.name, self.yolo.format, self.yolo.imgsz, dt))
            self.lbl_model.setStyleSheet("color:#2c9678;")
            print("[YOLO] 已加载 %s  输出=%s  格式=%s" % (
                path, self.yolo.out_shape, self.yolo.format))
        except Exception as e:
            import traceback
            self.yolo = None
            self.lbl_model.setText("❌ 加载失败: %s" % e)
            self.lbl_model.setStyleSheet("color:#c33;")
            print("[YOLO] 加载失败:\n%s" % traceback.format_exc())
        self.redetect()

    def _ensure_yolo(self):
        if self.yolo is not None:
            # 置信度/IoU 只影响后处理，不需要重建 session
            self.yolo.conf_thr = self.sp_conf.value()
            self.yolo.iou_thr = self.sp_iou.value()
            return self.yolo
        self.reload_model()
        return self.yolo

    # ---------------- 三个识别功能：开关与实现 ----------------
    def _radial_mm_per_px(self):
        """当前使用的径向比例（毫米/像素）。手动标定优先，否则用几何值。"""
        if self.length_manual and self.length_mm_px:
            return float(self.length_mm_px)
        m = self.cur_meta
        if m and m.get("maxDist") and m.get("h"):
            return float(m["maxDist"]) * 1000.0 / m["h"]
        return 0.0

    def on_recog_toggle(self):
        """数量/体长/行为三个按钮的开关响应。"""
        self.do_count = self.btn_count.isChecked()
        self.do_length = self.btn_length.isChecked()
        self.do_behavior = self.btn_behavior.isChecked()
        if self.do_behavior:
            self.schedule_behavior()
        else:
            self.behavior_label = "--"
            self.behavior_detail = ""
            self.behavior_minor = ""
            self.behavior_events = []
            if self.behavior_timer.isActive():
                self.behavior_timer.stop()
        self.redetect()

    def on_cal_changed(self):
        """体长标定参数变化。"""
        self.length_manual = self.cb_use_manual.isChecked()
        self.length_mm_px = self.sp_mmpx.value() if self.length_manual else None
        m = self.cur_meta
        if m and m.get("maxDist"):
            geom = float(m["maxDist"]) * 1000.0 / m["h"]
            if self.length_manual:
                self.lbl_cal.setText("手动标定: %.3f mm/像素（几何值 %.3f）" % (
                    self.sp_mmpx.value(), geom))
            else:
                self.lbl_cal.setText("几何标定: %.3f mm/像素\n= 量程 %.2fm / 图像高 %d\n1 米占 %.1f 像素" % (
                    geom, m["maxDist"], m["h"], 1000.0 / geom if geom else 0))
        else:
            self.lbl_cal.setText("几何标定（加载文件后显示具体数值）")
        self.redetect()

    def calibrate_by_gt(self):
        """用已知体长反推径向比例，并自动切到手动标定。"""
        if not self._last_targets:
            self.txt.append("\n⚠️ 当前帧没有目标，无法用已知体长反推比例。")
            return
        hpx = np.median([t[3] for t in self._last_targets])
        if hpx <= 0:
            return
        gt = self.sp_gt.value()
        need = gt * 10.0 / hpx              # mm/px
        self.cb_use_manual.setChecked(True)
        self.sp_mmpx.setValue(round(need, 3))
        geom = 0.0
        if self.cur_meta and self.cur_meta.get("maxDist"):
            geom = float(self.cur_meta["maxDist"]) * 1000.0 / self.cur_meta["h"]
        self.txt.append(
            "\n【体长标定】\n"
            "  目标边界框高中位 = %.1f 像素\n"
            "  已知体长 = %.1f cm -> 所需径向比例 = %.3f mm/像素\n"
            "  原几何比例 = %.3f mm/像素   （相差 %.1f 倍）\n"
            "  已切换为手动标定。若相差过大，说明几何标定与实测不符，\n"
            "  请核对：量程是否为真实探测距离 / 阈值是否只取到回声核心。" % (
                hpx, gt, need, geom, (need / geom) if geom else 0))

    def _on_bwin_changed(self, *_):
        """时间窗口（秒）变化：更新帧数提示并重算。

        注意：QDoubleSpinBox.valueChanged 会传入新值，
        所以必须用 *_ 接收，否则每次改窗口都会抛
        "takes 1 positional argument but 2 were given"，导致永不重算。
        """
        fps = self.frame_rate if self.frame_rate > 0 else 40.0
        self.lbl_win_frames.setText("= %.0f 帧" % (self.sp_bwin.value() * fps))
        self.schedule_behavior()

    def schedule_behavior(self):
        """延后触发行为分析，避免连续翻帧时反复重算。"""
        if not self.do_behavior:
            return
        # 自动播放时不重算（每帧都算会拖慢播放，且结果来不及看）
        if getattr(self, "play_timer", None) is not None and self.play_timer.isActive():
            return
        if self.behavior_timer.isActive():
            self.behavior_timer.stop()
        self.behavior_timer.start(350)      # 350ms 内的连续操作只算一次

    def _window_frame_range(self):
        """把"时间窗口(秒)"换算成帧区间。窗口以当前帧为**中心**。"""
        fps = self.frame_rate if self.frame_rate > 0 else 40.0
        n = max(2, int(round(self.sp_bwin.value() * fps)))     # 窗口帧数
        half = max(1, n // 2)
        lo = max(0, self.cur_idx - half)
        hi = min(len(self.starts) - 1, self.cur_idx + half)
        return lo, hi, n

    def run_behavior(self):
        """滑动时间窗口行为分析。

        设计要点（按需求改进）：
          1. 窗口长度按【秒】设定，内部按帧率换算成帧数
          2. 分析窗口内**每一帧**的行为（不只给一个多数标签），
             再做"事件段"汇总：摄食/应激各占多少、持续多久、发生在第几秒
          3. 结果直接显示"识别到投食聚集 / 应激"及其占比与时长
          4. 基线优先用外部"正常游动"文件；没有才退回窗口内相对基线

        时序分析说明：
          · 帧间时序：极化度用帧 k-1 -> k 的光流（相邻两帧对比）
          · 窗口时序：整窗逐帧分类 + 滑动均值平滑 + 持续性过滤
          · 全局时序：与外部基线数据的统计量对比
        """
        if not self.do_behavior or not self.starts or cv2 is None:
            return
        # 给用户明确反馈：分析需要数秒（尤其首次算基线），否则界面看起来像卡死
        self.behavior_label = "分析中"
        self.behavior_detail = "正在分析，请稍候…"
        self.redetect()
        QApplication.processEvents()

        lo, hi, want_n = self._window_frame_range()
        fps = self.frame_rate if self.frame_rate > 0 else 40.0
        stride = max(1, int(self.sp_bstride.value()))

        cfg = dict(BEHAVIOR_DEFAULTS)
        cfg.update(threshold=self.sp_thr.value(),
                   min_area=self.sp_min.value(),
                   kernel=max(1, self.sp_k.value()))
        # 平滑窗口按抽样后的点数换算，保持"约 0.4 秒"的时间尺度
        cfg["win"] = max(3, int(round(0.4 * fps / stride)))

        rows = []
        prev = None
        for i in range(lo, hi + 1, stride):
            g, meta = self._decode_one(i)
            if g is None:
                continue
            g = cv2.flip(g, 0)
            f, pts = behavior_extract_features(g, prev, cfg)
            f["frame"] = i
            f["t_sec"] = (i - lo) / fps          # 相对窗口起点的秒数
            f["_gray"] = g                        # 暂存，供个体运动指标计算
            rows.append(f)
            prev = g
        if not rows:
            self.behavior_label = "无法分析"
            self.behavior_detail = "窗口内没有可用帧"
            for r in rows:
                r.pop("_gray", None)
            self.redetect()
            return

        # ---- 个体运动指标（个体速度 / 个体角速度 / 极化度）----
        # 抽样后相邻样本做光流；位移按真实时间间隔折算，保证速度量纲正确
        h_img = int((self.cur_meta or {}).get("h") or IMG_H)
        max_dist = float((self.cur_meta or {}).get("maxDist") or 0.0)
        mpp_m = (max_dist / h_img) if h_img else 0.0
        dt_eff = stride / max(1e-6, fps)

        sp_meds, sp_p90s, turns, polars, npts_total = [], [], [], [], 0
        for a, b in zip(rows[:-1], rows[1:]):
            m = flow_individual_metrics(a.get("_gray"), b.get("_gray"),
                                        mpp_m, 1.0 / dt_eff)
            if m is not None:
                sp_meds.append(m["sp_med"])
                sp_p90s.append(m["sp_p90"])
                turns.append(m["turn_std"])
                polars.append(m["polar"])
                npts_total += m["n_pts"]
        for r in rows:
            r.pop("_gray", None)

        if sp_meds:
            self.ind_speed_med = float(np.nanmedian(sp_meds))
            self.ind_speed_p90 = float(np.nanmedian(sp_p90s))
            self.ind_turn_std = float(np.nanmedian(turns))
            self.flow_polar = float(np.nanmedian(polars))
            self.ind_npts = npts_total
        else:
            self.ind_speed_med = self.ind_speed_p90 = float("nan")
            self.ind_turn_std = self.flow_polar = float("nan")
            self.ind_npts = 0

        # ---- 基线：外部文件优先 ----
        baseline = None
        bsrc = ""
        if self.behavior_baseline_path and os.path.exists(self.behavior_baseline_path):
            cache = getattr(self, "_baseline_cache", None)
            if cache is not None and cache[0] == self.behavior_baseline_path:
                baseline = cache[1]
                bsrc = "外部基线(缓存)"
            else:
                baseline, nb = self._build_external_baseline(cfg)
                if baseline is not None:
                    self._baseline_cache = (self.behavior_baseline_path, baseline)
                    bsrc = "外部基线 %s(%d帧)" % (
                        os.path.basename(self.behavior_baseline_path), nb)
        if baseline is None:
            baseline = behavior_calibrate_baseline(rows, cfg)
            bsrc = "窗口内相对基线"

        behavior_classify_rows(rows, baseline, cfg)
        self.behavior_rows = rows
        self.behavior_baseline_src = bsrc

        # ---- 事件段汇总 ----
        valid = [r for r in rows if r["label"] != "UNKNOWN"]
        if not valid:
            self.behavior_label = "无法判定"
            self.behavior_minor = ""
            self.behavior_detail = "目标数不足（窗口内有效帧为 0）"
            self.redetect()
            return

        tot = len(valid)
        cnt = {lab: sum(1 for r in valid if r["label"] == lab)
               for lab in ("NORMAL", "FEEDING", "STRESS")}
        # 把"点"聚成"段"（连续同类帧算一段）
        segs = []
        cur_lab, cur_n = None, 0
        for r in valid:
            if r["label"] == cur_lab:
                cur_n += 1
            else:
                if cur_lab is not None:
                    segs.append((cur_lab, cur_n))
                cur_lab, cur_n = r["label"], 1
        if cur_lab is not None:
            segs.append((cur_lab, cur_n))
        seg_sec = [(lab, c * stride / fps) for lab, c in segs]

        # 最长的一段（各行为）
        longest = {}
        for lab, sec in seg_sec:
            if sec > longest.get(lab, 0):
                longest[lab] = sec

        # ---- 判定结果：把"窗口内识别到什么"完整写出来 ----
        parts = []
        for lab in ("FEEDING", "STRESS", "NORMAL"):
            if cnt[lab] > 0:
                parts.append("%s %.0f%%(%.1fs)" % (
                    BEHAVIOR_CN[lab], 100.0 * cnt[lab] / tot, cnt[lab] * stride / fps))
        self.behavior_detail = "窗口 %.1fs（%d 帧采样，步长 %d）| %s | %s" % (
            want_n / fps, tot, stride, " / ".join(parts), bsrc)

        # 主标签：优先报"异常行为"（摄食/应激），都没有才算正常
        # 这样"窗口里有投食聚集/应激"就能直接显示出来，不会被多数投票掩盖
        if cnt["STRESS"] > 0 and cnt["STRESS"] >= cnt["FEEDING"]:
            self.behavior_label = "STRESS"
            extra = []
            if cnt["FEEDING"] > 0:
                extra.append("摄食%.0f%%" % (100.0 * cnt["FEEDING"] / tot))
            if cnt["NORMAL"] > 0:
                extra.append("正常%.0f%%" % (100.0 * cnt["NORMAL"] / tot))
        elif cnt["FEEDING"] > 0:
            self.behavior_label = "FEEDING"
            extra = []
            if cnt["NORMAL"] > 0:
                extra.append("正常%.0f%%" % (100.0 * cnt["NORMAL"] / tot))
        else:
            self.behavior_label = "NORMAL"
            extra = []

        # 提示信息：占比 + 最长持续时长
        tips = list(extra)
        if longest.get("FEEDING"):
            tips.append("摄食最长%.1fs" % longest["FEEDING"])
        if longest.get("STRESS"):
            tips.append("应激最长%.1fs" % longest["STRESS"])
        self.behavior_minor = " ".join(tips)

        # 事件段明细：合并相邻同类段（否则会碎成 0.2 秒一段），只列显著的
        merged = []
        for lab, _ in seg_sec:
            if merged and merged[-1][0] == lab:
                merged[-1][1] += 1
            else:
                merged.append([lab, 1])
        # 只保留 >= 1.0 秒的异常段，按持续时间排序
        sig = [(lab, c) for lab, c in merged
               if lab != "NORMAL" and c * stride / fps >= 1.0]
        sig.sort(key=lambda x: -x[1])
        ev = ["连续%.1fs %s" % (c * stride / fps, BEHAVIOR_CN.get(lab, lab))
              for lab, c in sig[:5]]
        self.behavior_events = ev
        self.redetect()

    def _build_external_baseline(self, cfg):
        """从外部"正常游动"文件构建基线（成对连续采样，保证光流有效）。

        结果缓存到 sonar_params_baseline.json（按 文件路径+检测参数 为键），
        避免每次启动都重算（首次约需数秒，且会阻塞界面）。
        """
        import json as _json
        path = self.behavior_baseline_path
        key = "%s|thr%d|ma%d|k%d" % (os.path.abspath(path), cfg["threshold"],
                                     cfg["min_area"], cfg["kernel"])
        cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "sonar_params_baseline.json")
        # ---- 先查磁盘缓存 ----
        try:
            with open(cache_file, "r", encoding="utf-8-sig") as f:
                dc = _json.load(f)
            ent = dc.get(key)
            if ent:
                base = {k: (float(v[0]), float(v[1])) for k, v in ent["baseline"].items()}
                print("[行为] 基线取自磁盘缓存 (%d 帧)" % ent.get("n", 0))
                return base, int(ent.get("n", 0))
        except Exception:
            pass

        # ---- 重算 ----
        try:
            bs = build_index(path, log=lambda s: None)
            stride = max(1, len(bs) // 150)      # 取约 150 对，够用且快
            total = len(range(0, len(bs) - 1, stride))
            rows = []
            for cnt, i in enumerate(range(0, len(bs) - 1, stride)):
                ga, _ = self._decode_one(i, path=path, starts=bs)
                gb, _ = self._decode_one(i + 1, path=path, starts=bs)
                if ga is None or gb is None:
                    continue
                f, _ = behavior_extract_features(cv2.flip(gb, 0), cv2.flip(ga, 0), cfg)
                rows.append(f)
                if cnt % 20 == 0:
                    self.lbl_bbase.setText("正在计算基线… %d/%d" % (cnt, total))
                    QApplication.processEvents()      # 避免界面看起来卡死
            if len(rows) < 5:
                self.lbl_bbase.setText("基线计算失败（有效帧不足）")
                return None, 0
            base = behavior_calibrate_baseline(rows, cfg)
            # ---- 写磁盘缓存 ----
            try:
                dc = {}
                if os.path.exists(cache_file):
                    with open(cache_file, "r", encoding="utf-8-sig") as f:
                        dc = _json.load(f)
                dc[key] = {"n": len(rows),
                           "baseline": {k: list(v) for k, v in base.items()}}
                with open(cache_file, "w", encoding="utf-8") as f:
                    _json.dump(dc, f, ensure_ascii=False, indent=1)
                print("[行为] 基线已缓存到 %s" % os.path.basename(cache_file))
            except Exception as e:
                print("[行为] 基线缓存写入失败:", e)
            return base, len(rows)
        except Exception as e:
            import traceback
            print("[行为] 外部基线失败:", e)
            print(traceback.format_exc())
            self.lbl_bbase.setText("基线计算异常: %s" % e)
            return None, 0

    def choose_behavior_baseline(self):
        """选择"正常游动"数据文件作为行为基线。"""
        d = os.path.dirname(self.behavior_baseline_path) if self.behavior_baseline_path else ""
        p, _ = QFileDialog.getOpenFileName(
            self, "选择【正常游动】数据作为行为基线", d, "蓝衡数据格式 (*.DB)")
        if not p:
            return
        self.behavior_baseline_path = p
        self._baseline_cache = None
        self.lbl_bbase.setText("基线: %s" % os.path.basename(p))
        self.lbl_bbase.setStyleSheet("color:#2c9678;font-size:11px;")
        # 记忆到配置
        cfg = _load_config()
        cfg["behavior_baseline"] = p
        _save_config(cfg)
        if self.do_behavior:
            self.schedule_behavior()

    def _decode_one(self, idx, path=None, starts=None):
        """同步解码指定帧（供行为分析用；也可指定其它文件作基线）。

        关键：必须从【上一帧的起点】开始喂包。
        SDK 的帧边界状态只有在"从某帧首包开始喂"时才自洽；
        若从目标帧自身起点开始喂，writeOnePackData 可能不在预期位置返回
        oneFrameOK，导致随机访问时大量帧解码失败
        （实测：请求 41 帧只成功 21 帧，行为窗口被截断、结果失真）。
        """
        path = path or self.db_path
        starts = starts or self.starts
        if not path or not starts or idx >= len(starts):
            return None, None
        try:
            frame = m.LhForwardFrame()
            info = m.packageInfo()
            buf = np.zeros(1024 * 1024 * 10, dtype=np.uint8)
            start = starts[idx]
            read_from = starts[idx - 1] if idx > 0 else 0
            if idx + 1 < len(starts):
                maxp = (starts[idx + 1] - read_from) + 16
            else:
                maxp = 600
            imgobj = m.LhForwardImage(IMG_W, IMG_H)
            with open(path, "rb") as f:
                f.seek(read_from * PACK_SIZE)
                for _ in range(maxp):
                    raw = f.read(PACK_SIZE)
                    if len(raw) < PACK_SIZE:
                        break
                    if frame.writeOnePackData(np.frombuffer(raw, dtype=np.uint8)) != m.LfFrameRet.oneFrameOK:
                        continue
                    frame.getOneFrame(info, buf)
                    g = _gray_from_frame(info, buf, imgobj)
                    if g is None:
                        return None, None
                    meta = dict(frame=idx, w=g.shape[1], h=g.shape[0],
                                maxDist=float(info.maxDist),
                                hor=float(info.horAngleReso),
                                distReso=float(info.distReso),
                                packType=int(info.packType))
                    return g, meta
        except Exception:
            return None, None
        return None, None

    # ---------------- 文件加载 ----------------
    def choose_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择声呐数据文件", "", "蓝衡数据格式 (*.DB)")
        if p:
            self.load_file(p)

    def load_file(self, path):
        self.stop_play()                       # 换文件时先停播放
        self.db_path = path
        self.lbl_file.setText(os.path.basename(path))
        self.txt.setPlainText("正在建立帧索引，请稍候...\n（大文件如 500MB 约需 5 秒）")
        QApplication.processEvents()
        try:
            self.starts = build_index(path, log=self.txt.append)
        except Exception as e:
            import traceback
            self.txt.setPlainText("建立索引失败: %s\n%s" % (e, traceback.format_exc()))
            return
        if not self.starts:
            self.txt.setPlainText("该文件未解析出任何帧。\n"
                                  "请确认文件是完整的蓝衡 .DB 数据（不是导出图片或其它文件）。")
            return

        self.slider.setRange(0, len(self.starts) - 1)
        self._avg_packs = self._calc_avg_packs()
        self.decoder = Decoder(path, self.starts)
        self.decoder.frameReady.connect(self.on_frame)
        self.decoder.failed.connect(self.on_decode_failed)
        self.decoder.start()

        # 看门狗：8 秒内没出图就提示，避免"一直卡住"却没有任何信息
        self._pending_since = None
        self.watchdog = QTimer(self)
        self.watchdog.setInterval(1000)
        self.watchdog.timeout.connect(self._check_watchdog)
        self.watchdog.start()

        self.goto(0)

    def _calc_avg_packs(self):
        """按索引跨度估算平均每帧包数。"""
        if not self.starts or len(self.starts) < 2:
            return 0.0
        total = self.starts[-1] - self.starts[0]
        return total / float(len(self.starts) - 1)

    def _check_watchdog(self):
        import time as _t
        if self._pending_since is None:
            return
        if _t.time() - self._pending_since > 8:
            self._pending_since = None
            self.txt.append("\n⚠️ 8 秒内未收到可显示的画面。\n"
                            "  可能原因：该帧的 packType 不受支持，或文件不完整。\n"
                            "  已知支持：packType 0(原始波束) / 2(未压缩) / 3(PNG) / 4(JPG)。")

    def on_decode_failed(self, msg):
        self._pending_since = None
        self.txt.append("\n❌ " + msg)

    # ---------------- 播放控制 ----------------
    def toggle_play(self):
        """播放 / 暂停。"""
        if self.btn_play.isChecked():
            if not self.starts:
                self.btn_play.setChecked(False)
                return
            self.btn_play.setText("⏸ 暂停")
            self.play_timer.start(int(1000.0 / max(1, self.sp_fps.value())))
        else:
            self.play_timer.stop()
            self.btn_play.setText("▶ 播放")

    def stop_play(self):
        """停止播放（回到当前帧，不改变位置）。"""
        self.play_timer.stop()
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶ 播放")

    def on_fps_changed(self, v):
        if self.play_timer.isActive():
            self.play_timer.start(int(1000.0 / max(1, v)))

    def on_play_tick(self):
        """定时器到点：前进一帧。"""
        if not self.starts:
            self.stop_play()
            return
        nxt = self.cur_idx + 1
        if nxt >= len(self.starts):
            if self.cb_loop.isChecked():
                nxt = 0
            else:
                self.stop_play()
                return
        self.goto(nxt)

    def goto(self, idx):
        if not self.starts or self.decoder is None:
            return
        idx = max(0, min(idx, len(self.starts) - 1))
        self.cur_idx = idx
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
        self.lbl_frame.setText("帧 %d / %d" % (idx, len(self.starts) - 1))
        import time as _t
        self._pending_since = _t.time()
        self.decoder.request(idx)
        # 换帧后行为窗口也变了，需要重算（否则面板一直停在加载时那一帧的结果）
        self.schedule_behavior()

    def on_slider(self, v):
        if v != self.cur_idx:
            self.goto(v)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left:
            self.goto(self.cur_idx - 1)
        elif e.key() == Qt.Key_Right:
            self.goto(self.cur_idx + 1)
        else:
            super().keyPressEvent(e)

    # ---------------- 检测 ----------------
    def on_frame(self, idx, gray, meta):
        self._pending_since = None
        self.cur_gray = gray
        self.cur_meta = meta
        self.redetect()

    def redetect(self):
        if self.cur_gray is None or cv2 is None:
            return
        # 先按显示方向翻转图像，再检测。
        # 这样检测框和叠加文字都是正的；统计口径仍然是"全图"，与方向无关。
        flipped = self.cb_flip.isChecked()
        g = cv2.flip(self.cur_gray, 0) if flipped else self.cur_gray
        meta = self.cur_meta
        # 用包头 workCycle 更新真实帧率（时间窗口换算依赖它）
        wc = float(meta.get("work_cycle") or 0.0)
        if wc > 0:
            fps = 1.0 / wc
            if abs(fps - self.frame_rate) > 0.5:
                self.frame_rate = fps
                self._on_bwin_changed()
        thr = self.sp_thr.value()
        min_area = self.sp_min.value()
        kernel = self.sp_k.value()
        if kernel % 2 == 0:
            kernel += 1

        vis, mask, targets, h_img = detect(
            g, thr, min_area, kernel,
            self.cb_open.isChecked(), self.cb_close.isChecked())

        # ---- 按所选模式决定最终检测结果 ----
        mode = "阈值分割"
        n_yolo = 0
        if self.rb_yolo.isChecked() or self.rb_both.isChecked():
            yolo = self._ensure_yolo()
            if yolo is not None:
                vis_y, tgt_y, _ = detect_yolo(g, yolo, draw=self.cb_boxes.isChecked())
                n_yolo = len(tgt_y)
                if self.rb_yolo.isChecked():
                    # 纯 YOLO：用 YOLO 的结果与画面
                    vis = vis_y
                    targets = tgt_y
                    mask = np.zeros_like(g)
                    mode = "YOLO(%s)" % yolo.name
                else:
                    # 对比模式：传统结果画蓝框，YOLO 结果已画红框
                    for (x, y, w, h, area, mean_v, cy, cx) in targets:
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 128, 0), 1)
                    vis = cv2.addWeighted(vis, 0.45, vis_y, 0.55, 0)
                    for (x, y, w, h, area, mean_v, cy, cx) in tgt_y:
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    targets = tgt_y
                    mode = "对比: 阈值%d(蓝) vs YOLO%d(红)" % (len(mask) and 0 or 0, n_yolo)
            else:
                mode = "YOLO 权重未加载，已退回阈值分割"

        # 对比模式下单独统计阈值结果，用于显示
        n_thr = 0
        if self.rb_both.isChecked():
            _, _, t_thr, _ = detect(g, thr, min_area, kernel,
                                    self.cb_open.isChecked(), self.cb_close.isChecked())
            n_thr = len(t_thr)
            mode = "对比: 阈值 %d 个 vs YOLO %d 个" % (n_thr, n_yolo)

        # 尺度换算：底部 = maxDist，顶部 = 0。
        # 坐标系统一以"当前显示方向"为准：翻转后 maxDist 对应图像顶部。
        max_dist = float(meta["maxDist"]) if meta["maxDist"] else 0.0
        mpp = max_dist / h_img if h_img else 0.0     # 米/像素（距离向）

        # 先铺掩膜底色，再画检测框，保证框始终在最上层可见
        if self.cb_mask.isChecked():
            vis[mask > 0] = (255, 100, 0)
        if self.cb_boxes.isChecked():
            # 体长模式用更细的框，避免太密遮挡图像
            box_thick = 1
            for (x, y, w, h, area, mean_v, cy, cx) in targets:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), box_thick)

        # ---- 鱼群数量：只在开启数量识别时显示 ----
        n_targets = len(targets)
        self._last_targets = targets
        self._last_h_img = h_img
        if self.do_count:
            cnt_txt = "鱼群数量：%d" % n_targets
            self.lbl_count.setText(cnt_txt)
            self.lbl_count.adjustSize()
            self.lbl_count.move(IMG_W - self.lbl_count.width() - 12,
                                IMG_H - self.lbl_count.height() - 12)
            self.lbl_count.show()
            self.lbl_count.raise_()
            draw_count_overlay(vis, n_targets)
        else:
            self.lbl_count.hide()

        # ---- 体长识别：算每个目标的体长并标注 ----
        mm_ra = None
        len_med = None
        if self.do_length and targets:
            mm_ra = self._radial_mm_per_px()
            lens = []
            for (x, y, w, h, area, mean_v, cy, cx) in targets:
                L = h * mm_ra / 10.0                    # 径向体长(cm)
                lens.append(L)
            len_med = float(np.median(lens)) if lens else None
            if self.cb_len_show.isChecked():
                # 目标太多时标签会互相重叠，只标注最大的若干个
                order = sorted(range(len(targets)), key=lambda i: -targets[i][3])
                top_n = 12
                marked = set(order[:top_n])
                for i, (x, y, w, h, area, mean_v, cy, cx) in enumerate(targets):
                    if i in marked:
                        cv2.putText(vis, "%.0f" % lens[i], (x, max(11, y - 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                if len(targets) > top_n:
                    cv2.putText(vis, "(only top %d labelled)" % top_n,
                                (8, IMG_H - 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            # 中位体长放到左上角文字区，避免与右下角数量框重叠
            cv2.putText(vis, "L_med=%.1fcm  %.2fmm/px" % (len_med or 0, mm_ra),
                        (8, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # ---- 行为识别：把窗口分析结果标在图上 ----
        if self.do_behavior and self.behavior_label not in ("--", "无法分析", "无法判定"):
            col = {"FEEDING": (0, 200, 255), "STRESS": (0, 0, 255),
                   "NORMAL": (0, 255, 0)}.get(self.behavior_label, (200, 200, 200))
            cv2.putText(vis, "行为: %s" % BEHAVIOR_CN.get(self.behavior_label, "?"),
                        (8, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        # ---- 叠加：参数行 / 数据规格 / 翻转提示 ----
        cv2.putText(vis, "thr=%d  minA=%d  k=%d" % (thr, min_area, kernel),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, "packType=%d  packs=%s  range=%.2fm" % (
            meta["packType"], meta.get("n_packs", "?"), max_dist),
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if flipped:
            cv2.putText(vis, "FLIPPED", (8, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # 显示（等比缩放到显示区，居中裁切）
        disp = vis
        if disp.shape[0] > IMG_H:
            s = IMG_H / disp.shape[0]
            disp = cv2.resize(disp, (int(disp.shape[1] * s), IMG_H), interpolation=cv2.INTER_AREA)
        dw, dh = disp.shape[1], disp.shape[0]
        canvas = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        sx = max(0, (dw - IMG_W) // 2)
        sy = max(0, (dh - IMG_H) // 2)
        cw = min(IMG_W, dw - sx)
        ch = min(IMG_H, dh - sy)
        dx = (IMG_W - cw) // 2
        dy = (IMG_H - ch) // 2
        canvas[dy:dy + ch, dx:dx + cw] = disp[sy:sy + ch, sx:sx + cw]

        # 若开启数量识别，把数量烧进画面右下角（截图/导出时随图保存）
        if self.do_count:
            draw_count_overlay(canvas, n_targets)

        qi = QImage(canvas.data, IMG_W, IMG_H, canvas.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qi.copy()))

        # ---------------- 识别结果 ----------------
        lines = []
        lines.append("【识别功能】数量 %s | 体长 %s | 行为 %s" % (
            "开" if self.do_count else "关",
            "开" if self.do_length else "关",
            "开" if self.do_behavior else "关"))

        # ---- ① 数量识别 ----
        if self.do_count:
            lines.append("")
            lines.append("═══ ① 数量识别 ═══")
            lines.append("  目标数      : %d" % len(targets))
            lines.append("  检测算法    : %s" % mode)
            if targets:
                areas = np.array([t[4] for t in targets], dtype=float)
                lines.append("  面积        : min=%d 中位=%d max=%d 像素" % (
                    areas.min(), int(np.median(areas)), areas.max()))

        # ---- ② 体长识别 ----
        if self.do_length:
            lines.append("")
            lines.append("═══ ② 体长识别 ═══")
            lines.append("  标定方式    : %s" % (
                "手动 %.3f mm/px" % mm_ra if self.length_manual
                else "几何 %.3f mm/px (量程/图像高)" % (mm_ra or 0)))
            if targets:
                Ls = []
                for (x, y, w, h, area, mean_v, cy, cx) in targets:
                    Ls.append(h * (mm_ra or 0) / 10.0)
                Ls = np.array(Ls)
                lines.append("  径向体长    : 中位=%.1f cm  min=%.1f  max=%.1f" % (
                    np.median(Ls), Ls.min(), Ls.max()))
                lines.append("  等效跨度    : 1m 占 %.1f 像素" % (
                    1000.0 / mm_ra if mm_ra else 0))
            else:
                lines.append("  （当前帧无目标）")

        # ---- ③ 行为识别 ----
        if self.do_behavior:
            lines.append("")
            lines.append("═══ ③ 行为识别（滑动时间窗口）═══")
            fps = self.frame_rate if self.frame_rate > 0 else 40.0
            lines.append("  时间窗口    : %.1f 秒（= %d 帧，抽样步长 %d 帧）" % (
                self.sp_bwin.value(), int(round(self.sp_bwin.value() * fps)),
                int(self.sp_bstride.value())))
            lines.append("  判定结果    : %s%s" % (
                BEHAVIOR_CN.get(self.behavior_label, self.behavior_label),
                ("   —  " + self.behavior_minor) if self.behavior_minor else ""))
            if self.behavior_detail:
                lines.append("  窗口统计    : %s" % self.behavior_detail)
            if self.behavior_baseline_src:
                lines.append("  基线来源    : %s" % self.behavior_baseline_src)
            # ---- 四个核心指标 ----
            lines.append("  ── 四项指标 ──")
            if self.ind_npts > 0:
                lines.append("  ① 个体速度    : 中位=%.1f cm/s   P90=%.1f cm/s" % (
                    self.ind_speed_med * 100, self.ind_speed_p90 * 100))
                lines.append("  ② 个体角速度  : 航向离散度=%.0f 度（越小越同向）" % (
                    self.ind_turn_std))
                lines.append("  ③ 鱼群极化度  : %.3f   （0=方向杂乱, 1=完全同向）" % (
                    self.flow_polar))
                lines.append("  ④ 鱼群聚集度  : 离散度D=%.1f  最近邻NND=%.1f 像素" % (
                    np.median([r["D"] for r in self.behavior_rows if r["n"] > 0]) if
                    any(r["n"] > 0 for r in self.behavior_rows) else 0,
                    np.median([r["NND"] for r in self.behavior_rows if r["n"] > 0]) if
                    any(r["n"] > 0 for r in self.behavior_rows) else 0))
                lines.append("     运动样本    : %d 个特征点（跨 %d 帧对）" % (
                    self.ind_npts, max(0, len(self.behavior_rows) - 1)))
            else:
                lines.append("  ① 个体速度    : 无法计算（有效运动点不足）")
                lines.append("  ② 个体角速度  : 无法计算")
                lines.append("  ③ 鱼群极化度  : 无法计算")
                lines.append("  ④ 鱼群聚集度  : 见下方 P/D 统计")
            if getattr(self, "behavior_events", None):
                lines.append("  异常事件段  : %s" % " | ".join(self.behavior_events))
            if self.behavior_rows:
                P = np.array([r["P"] for r in self.behavior_rows], float)
                D = np.array([r["D"] for r in self.behavior_rows], float)
                lines.append("  极化度P     : 中位=%.3f max=%.3f" % (
                    np.median(P), P.max()))
                lines.append("  离散度D     : 中位=%.1f" % np.median(D))
            lines.append("  判据        : 聚集→摄食 / 持续极化→应激 / 其余→正常")

        lines.append("")
        lines.append("═══ 帧信息 ═══")
        lines.append("  文件     : %s" % os.path.basename(self.db_path or ""))
        lines.append("  帧号     : %d / %d" % (meta["frame"], len(self.starts) - 1))
        lines.append("  设备     : %s" % meta["device"])
        lines.append("  尺寸     : %d x %d   packType=%d (%s)" % (
            meta["w"], meta["h"], meta["packType"],
            PACKTYPE_NAME.get(meta["packType"], "未知")))
        lines.append("  量程     : %.3f m  径向 %.4f mm/px  方位向(最远) %.4f mm/px" % (
            max_dist, mpp * 1000,
            2 * np.pi * max_dist * (float(meta["hor"]) / 360.0) / max(1, meta["w"]) * 1000))

        self.txt.setPlainText("\n".join(lines))
        return

        # ---------------- 以下为旧版详细统计（保留备查，不会执行） ----------------
        lines = []
        lines.append("  文件        : %s" % os.path.basename(self.db_path or ""))
        lines.append("  帧号        : %d / %d" % (meta["frame"], len(self.starts) - 1))
        lines.append("  设备        : %s" % meta["device"])
        lines.append("  图像尺寸    : %d x %d 像素" % (meta["w"], meta["h"]))

        # ---- packType ----
        pt = meta["packType"]
        lines.append("  packType    : %d  (%s)" % (pt, PACKTYPE_NAME.get(pt, "未知")))

        # ---- 每帧包数 ----
        n_packs = meta.get("n_packs")
        pk_size = meta.get("pack_size", PACK_SIZE)
        if n_packs:
            lines.append("  每帧包数    : %d 包 x %d 字节 = %.1f KB/帧" % (
                n_packs, pk_size, n_packs * pk_size / 1024.0))
        lines.append("  本文件      : %d 帧   平均 %.1f 包/帧" % (
            len(self.starts), self._avg_packs))
        if meta.get("said_pack_num"):
            lines.append("  包头自述    : packNumPerFrame=%d" % meta["said_pack_num"])
        if meta.get("num_per_row"):
            lines.append("  波束点数    : numPerRow=%d  dataHeight=%d" % (
                meta["num_per_row"], meta["data_height"]))

        # ---- 分辨率 ----
        arc_px_max = 2 * np.pi * max_dist * (float(meta["hor"]) / 360.0) / max(1, meta["w"])
        lines.append("═══ 分辨率 ═══")
        lines.append("  量程 maxDist      : %.3f m" % max_dist)
        lines.append("  距离分辨率(距离向): %.4f mm/像素   (maxDist / 图像高)" % (mpp * 1000))
        lines.append("  方位分辨率(方位向): %.4f mm/像素   (最远距离处弧长 / 图像宽)" % (arc_px_max * 1000))
        lines.append("  各向异性比        : %.2f 倍  (方位向 / 距离向)" % (
            arc_px_max / mpp if mpp > 0 else 0))
        lines.append("  界面标称 distReso : %.4f mm" % (float(meta["distReso"]) * 1000))
        lines.append("  水平开角          : %.1f deg     声速: %.0f m/s" % (meta["hor"], meta["vel"]))

        lines.append("═══ 检测结果（全图统计）═══")
        lines.append("  目标数      : %d" % len(targets))
        if targets:
            areas = np.array([t[4] for t in targets], dtype=float)
            means = np.array([t[5] for t in targets], dtype=float)
            # 每个目标按其所在距离计算像素物理尺寸，换算等效面积
            eq_area = []
            eq_len = []
            for (x, y, w, h, area, mean_v, cy, cx) in targets:
                r_m = max_dist * cy / h_img             # 该目标的实际距离(m)
                if r_m <= 1e-6:
                    continue
                # 方位向弧长分辨率 = 2*pi*r*(hor/360)/w
                arc_px = 2 * np.pi * r_m * (float(meta["hor"]) / 360.0) / max(1, meta["w"])
                a = area * mpp * arc_px
                eq_area.append(a)
                eq_len.append(2 * np.sqrt(a / np.pi))
            lines.append("  面积 min/中位/max : %d / %d / %d 像素" % (areas.min(), int(np.median(areas)), areas.max()))
            lines.append("  平均灰度 min/max  : %.1f / %.1f" % (means.min(), means.max()))
            lines.append("  平均灰度 中位     : %.1f" % float(np.median(means)))
            if eq_area:
                eq_area = np.array(eq_area); eq_len = np.array(eq_len)
                lines.append("  等效面积 中位     : %.1f cm^2" % (np.median(eq_area) * 1e4))
                lines.append("  等效体长 中位     : %.1f cm   (min %.1f / max %.1f)" % (
                    np.median(eq_len) * 100, eq_len.min() * 100, eq_len.max() * 100))
        else:
            lines.append("  （当前参数下无目标，请降低阈值或减小最小面积）")
        lines.append("")
        lines.append("提示: 目标是圆形等效假设，体长仅作量级参考；全图统计未排除池底/近场。")
        lines.append("      距离换算按【当前显示方向】进行：%s" %
                     ("已上下翻转（maxDist 在图像上方）" if flipped else "未翻转（maxDist 在图像下方）"))
        lines.append("      翻转只改变显示与距离映射方向，目标数/面积/灰度统计完全不变。")

        self.txt.setPlainText("\n".join(lines))

    def closeEvent(self, e):
        self.stop_play()
        if self.decoder is not None:
            self.decoder.stop()
            self.decoder.wait(2000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if cv2 is None:
        print("警告: 未安装 opencv-python，检测功能不可用。请执行:")
        print("  python -m pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/")
    w = ParamTuner(path)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
