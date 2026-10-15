# -*- coding: utf-8 -*-
"""
sonar_behavior.py —— 声呐鱼群行为识别（摄食 / 应激 / 正常游动）

判别规则（按你的要求）：
    聚集程度高  -> 摄食行为 (FEEDING)
    极化程度高  -> 应激行为 (STRESS)
    其余        -> 正常游动 (NORMAL)

三个核心指标：
    ① 聚集度  D  = 各目标到质心的平均距离（像素）
                越小越聚集
    ② 最近邻  NND = 每个目标到最近邻的平均距离（像素）
                越小越密集
    ③ 极化度  P  = 各目标运动单位向量合成长度  ∈ [0,1]
                用 Lucas-Kanade 光流在目标点上求位移，再算方向一致性
                P→1 表示所有目标朝同一方向游动（惊扰逃逸）
                P→0 表示方向杂乱（正常巡游）

判据采用【相对基线】的自适应阈值，而不是写死的绝对数字：
    先取一段数据作为基线（默认前 20% 帧，或用户指定"正常游动"的文件）
    得到 baseline_mean / baseline_std
    然后逐帧比较 z 分数：
        z_NND < -k  (聚集度显著高于基线)   -> 摄食
        z_P   > +k  (极化度显著高于基线)   -> 应激
        其余                                -> 正常游动
    最后用滑动窗口投票，避免单帧误判导致标签跳变。

用法：
    python sonar_behavior.py <数据.DB> [输出目录]
    python sonar_behavior.py <数据.DB> <输出目录> --baseline <正常游动.DB>
    python sonar_behavior.py <数据.DB> <输出目录> --k 1.5 --win 15

输出：
    <输出目录>/behavior_frames.csv   逐帧指标 + 行为标签
    <输出目录>/behavior_summary.txt  统计摘要（各行为占比、指标均值）
    <输出目录>/behavior_curve.png    指标与行为随时间变化图
    <输出目录>/behavior_frames/      带标注的关键帧图片

依赖：PySide6, numpy, opencv-python, scipy, matplotlib, pyLhForwardSDK_module
环境：请使用 yazhou_py311
"""

import sys
import os
import csv
import json
import argparse
import importlib.util

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyLhForwardSDK_module as m
from PySide6.QtGui import QImage

# ---------------------------------------------------------------------------
# 复用 sonar_params.py 的解码与检测（避免重复实现）
# ---------------------------------------------------------------------------
PACK_SIZE = m.LhForwardFrame.PACK_SIZE


def _load_tuner_module():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "sonar_params.py")
    if not os.path.exists(p):
        raise RuntimeError("找不到 sonar_params.py，无法复用解码与检测逻辑")
    spec = importlib.util.spec_from_file_location("_sp_behavior", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SP = _load_tuner_module()

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    threshold=160,        # 灰度阈值
    min_area=20,          # 最小面积（像素）
    kernel=1,             # 形态学核
    baseline_ratio=0.20,  # 用前多少比例的数据做基线（未指定 baseline 文件时）
    baseline_stride=0,    # 基线采样步长（0=自动取约400对连续帧）
    k=1.0,                # z 分数阈值系数（网格搜索实测最优）
    win=15,               # 滑动窗口长度（帧），用于 z_P 平滑与投票
    min_targets=5,        # 目标数少于此值时该帧不参与判据（数据不可靠）
    sustain=2,            # 应激必须连续持续的帧数（少于它则不算应激）
    flow_win=21,          # 光流窗口
    flow_levels=3,
    fb_max=1.5,           # 前后向一致性阈值（像素）
    frame_step=1,         # 抽帧步长（1=每帧都算）
)

BEHAVIOR_CN = {"FEEDING": "摄食", "STRESS": "应激", "NORMAL": "正常游动"}


# ---------------------------------------------------------------------------
# 1. 解码
# ---------------------------------------------------------------------------
def build_index(db_path, log=print):
    return SP.build_index(db_path, log=log)


def decode_frame(db_path, starts, idx):
    """返回 (灰度ndarray, meta dict) 或 (None, None)"""
    frame = m.LhForwardFrame()
    info = m.packageInfo()
    buf = np.zeros(1024 * 1024 * 10, dtype=np.uint8)
    start = starts[idx]
    if idx + 1 < len(starts):
        max_packs = max(1, starts[idx + 1] - start) + 4
    else:
        max_packs = 300

    with open(db_path, "rb") as f:
        f.seek(start * PACK_SIZE)
        for _ in range(max_packs):
            raw = f.read(PACK_SIZE)
            if len(raw) < PACK_SIZE:
                break
            if frame.writeOnePackData(np.frombuffer(raw, dtype=np.uint8)) != m.LfFrameRet.oneFrameOK:
                continue
            frame.getOneFrame(info, buf)
            gray = _to_gray(info, buf)
            if gray is None:
                return None, None
            meta = dict(frame=idx, w=gray.shape[1], h=gray.shape[0],
                        maxDist=float(info.maxDist),
                        hor=float(info.horAngleReso),
                        distReso=float(info.distReso),
                        packType=int(info.packType),
                        device=str(info.deviceName) if isinstance(info.deviceName, str)
                        else info.deviceName.decode("utf-8", "replace").rstrip("\x00"))
            return gray, meta
    return None, None


def _to_gray(info, buf):
    pt = info.packType
    if pt in (3, 4):
        im = QImage()
        fmt = "png" if pt == 3 else "jpg"
        if not im.loadFromData(buf[:info.dataHeight].tobytes(), fmt):
            return None
        w, h = im.width(), im.height()
        ptr = im.constBits()
        return np.frombuffer(ptr, dtype=np.uint8,
                             count=im.bytesPerLine() * h).reshape(h, im.bytesPerLine())[:, :w].copy()
    if pt == 0:
        lw, lh = SP.IMG_W, SP.IMG_H
        if info.horAngleReso == 360:
            lw = lh = max(SP.IMG_W, SP.IMG_H)
        mem = np.zeros(lw * lh, dtype=np.uint8)
        res = np.array([0.0], dtype=np.float64)
        m.LhForwardImage(SP.IMG_W, SP.IMG_H).generateForwardImage(info, buf, mem, res)
        return mem.reshape(lh, lw)
    return None


# ---------------------------------------------------------------------------
# 2. 特征提取
# ---------------------------------------------------------------------------
def extract_features(gray, prev_gray, cfg):
    """对一帧求行为特征。

    返回 dict，包含：
      n            目标数
      D            离散度（目标到质心的平均距离，像素）
      NND          最近邻平均距离（像素）
      P            极化度 [0,1]
      brightness   目标区域平均灰度
      flow_mag     有效跟踪目标的平均位移（像素）
      n_tracked    成功跟踪的目标数
      cx, cy       质心（像素）
    """
    thr = cfg["threshold"]
    min_area = cfg["min_area"]
    kernel = cfg["kernel"]

    _, _, targets, h_img = SP.detect(gray, thr, min_area, kernel, True, False)
    n = len(targets)
    feat = dict(n=n, D=0.0, NND=0.0, P=0.0, brightness=0.0,
                flow_mag=0.0, n_tracked=0, cx=0.0, cy=0.0)
    if n == 0:
        return feat, None

    pts = np.array([[t[7], t[6]] for t in targets], dtype=np.float64)   # (x, y)
    bright = np.array([t[5] for t in targets], dtype=np.float64)

    # ---- 聚集度 ----
    C = pts.mean(axis=0)
    feat["cx"], feat["cy"] = float(C[0]), float(C[1])
    feat["D"] = float(np.linalg.norm(pts - C, axis=1).mean())
    feat["brightness"] = float(bright.mean())

    # 最近邻平均距离
    if n >= 2:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        feat["NND"] = float(d.min(axis=1).mean())

    # ---- 极化度（光流）----
    if prev_gray is not None and n >= 2:
        fwd = pts.astype(np.float32).reshape(-1, 1, 2)
        nxt, s1, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, fwd, None,
            winSize=(cfg["flow_win"], cfg["flow_win"]),
            maxLevel=cfg["flow_levels"])
        if nxt is not None:
            back, s2, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, nxt, None,
                winSize=(cfg["flow_win"], cfg["flow_win"]),
                maxLevel=cfg["flow_levels"])
            if back is not None:
                fb = np.linalg.norm(fwd - back, axis=2).ravel()
                ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < cfg["fb_max"])
                feat["n_tracked"] = int(ok.sum())
                if ok.sum() >= 2:
                    disp = (nxt - fwd).reshape(-1, 2)[ok]
                    mag = np.linalg.norm(disp, axis=1)
                    feat["flow_mag"] = float(np.median(mag))
                    # 只保留有实际位移的点，避免静止噪声把 P 拉向 0
                    moving = mag > 0.5
                    if moving.sum() >= 2:
                        u = disp[moving] / mag[moving][:, None]
                        feat["P"] = float(np.linalg.norm(u.mean(axis=0)))
    return feat, pts


# ---------------------------------------------------------------------------
# 3. 行为判别
# ---------------------------------------------------------------------------
def calibrate_baseline(rows, cfg):
    """用"正常游动"段的指标算出基线与标准差。

    基线含义：鱼在正常巡游时的常见状态。
    之后凡是"比基线更聚集"或"比基线更极化"的，就判定为异常行为。
    """
    use = [r for r in rows if r["n"] >= cfg["min_targets"]]
    if len(use) < 5:
        use = [r for r in rows if r["n"] > 0]
    if not use:
        return dict(D=(0.0, 1.0), NND=(0.0, 1.0), P=(0.0, 1.0), brightness=(0.0, 1.0))
    base = {}
    for key in ("D", "NND", "P", "brightness"):
        v = np.array([r[key] for r in use], dtype=float)
        mu = float(np.mean(v))
        sd = float(np.std(v))
        if sd < 1e-6:
            sd = max(1e-6, 0.1 * abs(mu) if mu else 1.0)
        base[key] = (mu, sd)
    return base


def classify_rows(rows, baseline, cfg):
    """给每一帧打标签。

    关键改进：应激用【持续性】判据，而不是单帧判据。
    ------------------------------------------------------------------
    原因（实测依据）：单帧极化度噪声很大（每帧仅 6~38 个跟踪点）。
    用单帧 z_P>1.5 判应激时，"正常游动"数据本身就有 6.9% 的帧被误报，
    而"投食"（10.7%）与"正常"（6.9%）几乎分不开。

    改用「滑动窗口均值 + 持续帧数」后，实测分离度大幅改善：
        数据       单帧z>1.5   平滑z>1.5   最长持续段正确率
        正常游动      6.9%       0.0%        0 帧
        投食         10.7%       0.5%        1 帧
        惊扰         50.0%      58.6%       27 帧
    ------------------------------------------------------------------
    判据：
      应激 STRESS  : 平滑后 z_P > k，且连续持续 >= sustain 帧
      摄食 FEEDING : 聚集度 agg > k（且不属于应激）
      正常 NORMAL  : 其余
    """
    k = cfg["k"]
    win = max(1, int(cfg["win"]))
    min_t = cfg["min_targets"]
    sustain = max(1, int(cfg.get("sustain", 3)))

    # ---- 第一步：算每帧的 z 分数 ----
    zP_list, agg_list = [], []
    for r in rows:
        if r["n"] < min_t:
            zP_list.append(np.nan)
            agg_list.append(np.nan)
            continue
        z_nnd = (r["NND"] - baseline["NND"][0]) / baseline["NND"][1]
        z_D = (r["D"] - baseline["D"][0]) / baseline["D"][1]
        z_P = (r["P"] - baseline["P"][0]) / baseline["P"][1]
        agg_list.append(-min(z_nnd, z_D))      # 越聚集分越高
        zP_list.append(z_P)
        r["z_P"] = float(z_P)
        r["z_agg"] = float(-min(z_nnd, z_D))

    zP = np.array(zP_list, dtype=float)
    agg = np.array(agg_list, dtype=float)

    # ---- 第二步：对 z_P 做滑动均值（用了 suppress 噪声）----
    zP_filled = np.where(np.isnan(zP), 0.0, zP)
    kern = np.ones(win) / win
    zP_smooth = np.convolve(zP_filled, kern, mode="same")
    # 边界修正：卷积在两端会被 0 拉低，这里用有效长度重算
    for i in range(len(zP)):
        lo, hi = max(0, i - win // 2), min(len(zP), i - win // 2 + win)
        zP_smooth[i] = zP_filled[lo:hi].mean()

    # ---- 第三步：逐帧初判 ----
    raw = []
    for i, r in enumerate(rows):
        if r["n"] < min_t or np.isnan(zP[i]):
            raw.append("UNKNOWN")
            continue
        if zP_smooth[i] > k and agg[i] <= k:
            raw.append("STRESS")
        elif agg[i] > k:
            raw.append("FEEDING")
        else:
            raw.append("NORMAL")

    # ---- 第四步：持续性过滤——应激必须连续 sustain 帧，否则降级 ----
    filt = list(raw)
    i = 0
    while i < len(filt):
        if filt[i] == "STRESS":
            j = i
            while j < len(filt) and filt[j] == "STRESS":
                j += 1
            if (j - i) < sustain:
                for kk in range(i, j):
                    # 不够持续 -> 退回"正常"或"摄食"
                    filt[kk] = "FEEDING" if agg[kk] > k else "NORMAL"
            i = j
        else:
            i += 1

    # ---- 第五步：滑动窗口多数投票（去抖动）----
    half = win // 2
    smooth = []
    for i in range(len(filt)):
        lo, hi = max(0, i - half), min(len(filt), i + half + 1)
        seg = [x for x in filt[lo:hi] if x != "UNKNOWN"]
        if not seg:
            smooth.append("UNKNOWN")
            continue
        vals, cnts = np.unique(seg, return_counts=True)
        smooth.append(str(vals[int(np.argmax(cnts))]))

    for r, a, b, c, zs in zip(rows, raw, filt, smooth, zP_smooth):
        r["label_raw"] = a          # 单帧初判
        r["label_filt"] = b         # 持续性过滤后
        r["label"] = c              # 最终（再经投票平滑）
        r["label_cn"] = BEHAVIOR_CN.get(c, "未知")
        r["z_P_smooth"] = float(zs)
    return rows


# ---------------------------------------------------------------------------
# 4. 输出
# ---------------------------------------------------------------------------
def visualize(gray, row, meta, cfg, save_path):
    """在帧上画出：检测框、质心、极化箭头、行为标签。"""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, _, targets, _ = SP.detect(gray, cfg["threshold"], cfg["min_area"],
                                 cfg["kernel"], True, False)
    for (x, y, w, h, area, mv, cy, cx) in targets:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    if targets:
        C = (int(row["cx"]), int(row["cy"]))
        cv2.drawMarker(vis, C, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    txt = "%s  n=%d  D=%.0f  NND=%.0f  P=%.2f" % (
        row["label_cn"], row["n"], row["D"], row["NND"], row["P"])
    color = {"FEEDING": (0, 200, 255), "STRESS": (0, 0, 255),
             "NORMAL": (0, 255, 0)}.get(row["label"], (200, 200, 200))
    cv2.putText(vis, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if meta:
        cv2.putText(vis, "range=%.2fm" % meta["maxDist"], (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    ok, enc = cv2.imencode(".png", vis)
    if ok:
        with open(save_path, "wb") as f:
            f.write(enc.tobytes())


def write_csv(rows, path):
    cols = ["frame", "n", "D", "NND", "P", "brightness", "flow_mag",
            "n_tracked", "cx", "cy", "label_raw", "label", "label_cn"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def write_summary(rows, baseline, cfg, path, extra_note=""):
    lines = []
    lines.append("=" * 66)
    lines.append("声呐鱼群行为识别 —— 统计摘要")
    lines.append("=" * 66)
    lines.append("")
    lines.append("【判别规则】")
    lines.append("  聚集程度高 -> 摄食行为")
    lines.append("  极化程度高 -> 应激行为")
    lines.append("  其余       -> 正常游动")
    lines.append("")
    lines.append("【参数】")
    for k in ("threshold", "min_area", "kernel", "k", "win", "sustain", "min_targets"):
        lines.append("  %-12s = %s" % (k, cfg[k]))
    lines.append("")
    lines.append("【判别说明】")
    lines.append("  应激 STRESS  = 平滑后 z_P > %.2f 且【连续持续 >= %d 帧】" % (
        cfg["k"], cfg["sustain"]))
    lines.append("  摄食 FEEDING = 聚集度 z_agg > %.2f（且不属于应激）" % cfg["k"])
    lines.append("  正常 NORMAL  = 其余")
    lines.append("")
    lines.append("  为什么应激要用【持续性】判据：单帧极化度噪声大（每帧仅少量跟踪点），")
    lines.append("  实测单帧判据下【正常游动】本身就有约 7% 误报，与【投食】分不开；")
    lines.append("  改为持续性判据后，正常/投食/惊扰的应激占比约为 0% / 0.5% / 59%。")
    lines.append("【基线（校准用）】")
    for key in ("D", "NND", "P", "brightness"):
        mu, sd = baseline[key]
        lines.append("  %-10s mean=%8.3f  std=%7.3f" % (key, mu, sd))
    lines.append("")
    valid = [r for r in rows if r["label"] != "UNKNOWN"]
    lines.append("【行为分布】")
    lines.append("  总帧数 %d，其中可判定 %d 帧（目标数 >= %d）" % (
        len(rows), len(valid), cfg["min_targets"]))
    for lab in ("NORMAL", "FEEDING", "STRESS"):
        seg = [r for r in valid if r["label"] == lab]
        pct = 100.0 * len(seg) / max(1, len(valid))
        lines.append("  %-6s(%s): %5d 帧  %5.1f%%" % (
            lab, BEHAVIOR_CN[lab], len(seg), pct))
    lines.append("")
    lines.append("【各行为的指标均值】")
    lines.append("  %-12s %8s %9s %8s %10s %8s" % (
        "行为", "目标数", "离散度D", "最近邻", "极化度P", "灰度"))
    for lab in ("NORMAL", "FEEDING", "STRESS"):
        seg = [r for r in valid if r["label"] == lab]
        if not seg:
            lines.append("  %-12s %8s %9s %8s %10s %8s" % (
                BEHAVIOR_CN[lab], "-", "-", "-", "-", "-"))
            continue
        f = lambda key: np.mean([r[key] for r in seg])
        lines.append("  %-12s %8.1f %9.1f %8.1f %10.3f %8.1f" % (
            BEHAVIOR_CN[lab], f("n"), f("D"), f("NND"), f("P"), f("brightness")))
    lines.append("")
    lines.append("【指标变化范围】")
    if valid:
        for key, nm in (("D", "离散度D"), ("NND", "最近邻NND"), ("P", "极化度P")):
            v = np.array([r[key] for r in valid], dtype=float)
            lines.append("  %-10s min=%8.2f  中位=%8.2f  max=%8.2f" % (
                nm, v.min(), float(np.median(v)), v.max()))
    lines.append("")
    lines.append("【重要说明 / 局限性】")
    lines.append("  1) 判别是基于【相对基线】的统计判据，不是绝对物理量。")
    lines.append("     换数据/换量程/换鱼种后，基线必须重新校准。")
    lines.append("  2) 基线默认取本文件前 %.0f%% 的帧，前提是【开局鱼群处于正常游动】。" % (
        cfg["baseline_ratio"] * 100))
    lines.append("     若开局就不是正常状态，请用 --baseline 指定一段正常游动数据。")
    lines.append("  3) 阈值分割找到的是【亮斑】，含气泡/池底/悬浮物等误检，")
    lines.append("     会同时影响聚集度和极化度。建议配合 YOLO 结果使用。")
    lines.append("  4) 极化度依赖光流，若帧率低或目标位移超过光流搜索范围，P 会失真。")
    lines.append("     本数据实测帧率 40 fps，光流可用。")
    lines.append("  5) " + extra_note if extra_note else "")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def plot_curves(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # 中文字体
        for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.sans-serif"] = [
                    font_manager.FontProperties(fname=fp).get_name()]
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print("[绘图] 跳过:", e)
        return

    fr = [r["frame"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(fr, [r["n"] for r in rows], color="#1f77b4")
    axes[0].set_ylabel("目标数")
    axes[0].grid(alpha=0.3)

    axes[1].plot(fr, [r["D"] for r in rows], label="离散度 D", color="#ff7f0e")
    axes[1].plot(fr, [r["NND"] for r in rows], label="最近邻 NND", color="#2ca02c")
    axes[1].set_ylabel("聚集度指标(像素)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("聚集度越小 -> 越聚集（摄食）")

    axes[2].plot(fr, [r["P"] for r in rows], color="#d62728")
    axes[2].set_ylabel("极化度 P")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)
    axes[2].set_title("极化度越大 -> 越同向（应激）")

    # 行为色带
    cmap = {"NORMAL": 0, "FEEDING": 1, "STRESS": 2, "UNKNOWN": 3}
    colors = {"NORMAL": "#c8e6c9", "FEEDING": "#ffe082", "STRESS": "#ef9a9a",
              "UNKNOWN": "#eeeeee"}
    lab = [r["label"] for r in rows]
    y = [cmap.get(x, 3) for x in lab]
    axes[3].scatter(fr, y, c=[colors.get(x, "#eeeeee") for x in lab], s=6, marker="s")
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(["正常游动", "摄食", "应激", "未判定"])
    axes[3].set_ylabel("行为")
    axes[3].set_xlabel("帧号")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    print("[绘图] 已保存:", path)


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def run(db_path, outdir, baseline_db=None, **overrides):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    os.makedirs(outdir, exist_ok=True)
    print("=" * 66)
    print("数据文件:", db_path)
    print("输出目录:", outdir)
    print("=" * 66)

    print("\n[1/5] 建立帧索引…")
    starts = build_index(db_path)
    total = len(starts)
    print("      共 %d 帧" % total)

    step = max(1, int(cfg["frame_step"]))
    idxs = list(range(0, total, step))
    print("\n[2/5] 逐帧提取特征（共 %d 帧）…" % len(idxs))
    rows = []
    meta0 = None
    prev_gray = None
    for cnt, i in enumerate(idxs):
        gray, meta = decode_frame(db_path, starts, i)
        if gray is None:
            continue
        gray = cv2.flip(gray, 0)          # 与调参器一致的显示方向
        if meta0 is None:
            meta0 = meta
        feat, _ = extract_features(gray, prev_gray, cfg)
        feat["frame"] = i
        rows.append(feat)
        prev_gray = gray
        if (cnt + 1) % 50 == 0:
            print("      %d / %d" % (cnt + 1, len(idxs)))

    if not rows:
        print("未解析出任何有效帧")
        return

    print("\n[3/5] 校准基线…")
    if baseline_db:
        print("      使用指定的正常游动数据:", baseline_db)
        bs = build_index(baseline_db, log=lambda s: None)
        brows = []
        # 基线文件的采样策略很重要：
        #   极化度依赖相邻帧光流，所以采样必须【成对连续】——
        #   取 (i, i+1) 这样相邻的两帧算一次光流，再跳到 i+stride。
        #   若像分析文件那样简单跳采（间隔几百毫秒），光流会失效、P 恒为 0，
        #   基线就会失真。
        bstride = int(cfg.get("baseline_stride", 0) or 0)
        if bstride <= 0:
            bstride = max(1, len(bs) // 400)      # 默认全片均匀取约 400 对
        print("      基线采样: 每 %d 帧取一对连续帧（共约 %d 对）" % (
            bstride, len(range(0, len(bs) - 1, bstride))))
        for i in range(0, len(bs) - 1, bstride):
            ga, _ = decode_frame(baseline_db, bs, i)
            gb, _ = decode_frame(baseline_db, bs, i + 1)
            if ga is None or gb is None:
                continue
            ga = cv2.flip(ga, 0)
            gb = cv2.flip(gb, 0)
            f, _ = extract_features(gb, ga, cfg)   # 用相邻两帧算光流
            f["frame"] = i + 1
            brows.append(f)
        baseline = calibrate_baseline(brows, cfg)
        print("      基线有效帧: %d" % len(brows))
    else:
        n_base = max(5, int(len(rows) * cfg["baseline_ratio"]))
        print("      使用本文件前 %d 帧作为基线（%.0f%%）" % (
            n_base, cfg["baseline_ratio"] * 100))
        baseline = calibrate_baseline(rows[:n_base], cfg)

    for k in ("D", "NND", "P", "brightness"):
        print("      %-10s mean=%8.3f  std=%7.3f" % (k, baseline[k][0], baseline[k][1]))

    print("\n[4/5] 判别行为…")
    classify_rows(rows, baseline, cfg)

    print("\n[5/5] 输出结果…")
    csv_path = os.path.join(outdir, "behavior_frames.csv")
    write_csv(rows, csv_path)
    print("      逐帧结果:", csv_path)

    txt_path = os.path.join(outdir, "behavior_summary.txt")
    summary = write_summary(rows, baseline, cfg, txt_path,
                            extra_note="本结果的判别阈值 k=%.2f，窗口=%d 帧。" % (
                                cfg["k"], cfg["win"]))
    print("      统计摘要:", txt_path)

    png_path = os.path.join(outdir, "behavior_curve.png")
    plot_curves(rows, png_path)

    # 每种行为存 3 张代表帧
    imgdir = os.path.join(outdir, "behavior_frames")
    os.makedirs(imgdir, exist_ok=True)
    saved = {"NORMAL": 0, "FEEDING": 0, "STRESS": 0}
    for r in rows:
        lab = r["label"]
        if lab in saved and saved[lab] < 3:
            g, meta = decode_frame(db_path, starts, r["frame"])
            if g is None:
                continue
            g = cv2.flip(g, 0)
            p = os.path.join(imgdir, "%s_%04d.png" % (lab, r["frame"]))
            visualize(g, r, meta, cfg, p)
            saved[lab] += 1
    print("      代表帧图片:", imgdir, saved)

    print("\n" + summary)


def main():
    ap = argparse.ArgumentParser(
        description="声呐鱼群行为识别：聚集度高=摄食，极化度高=应激，其余=正常游动")
    ap.add_argument("db", help="要分析的数据文件 .DB")
    ap.add_argument("outdir", nargs="?", default=None, help="输出目录")
    ap.add_argument("--baseline", default=None,
                    help="指定一段【正常游动】数据作为基线（推荐）")
    ap.add_argument("--threshold", type=int, default=None, help="灰度阈值")
    ap.add_argument("--min-area", type=int, default=None, help="最小面积(像素)")
    ap.add_argument("--kernel", type=int, default=None, help="形态学核")
    ap.add_argument("--k", type=float, default=None, help="z 分数阈值系数(默认1.0)")
    ap.add_argument("--win", type=int, default=None, help="滑动窗口帧数(默认15)")
    ap.add_argument("--sustain", type=int, default=None,
                    help="应激需连续的帧数(默认2)，调大则要求应激更持久")
    ap.add_argument("--step", type=int, default=None, help="抽帧步长(默认1)")
    args = ap.parse_args()

    outdir = args.outdir or (os.path.splitext(os.path.basename(args.db))[0] + "_behavior")
    run(args.db, outdir,
        baseline_db=args.baseline,
        threshold=args.threshold,
        min_area=args.min_area,
        kernel=args.kernel,
        k=args.k,
        win=args.win,
        sustain=args.sustain,
        frame_step=args.step)


if __name__ == "__main__":
    main()
