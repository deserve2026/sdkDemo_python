# -*- coding: utf-8 -*-
"""
sonar_length.py —— 声呐鱼体长识别（以量程/距离标定为准）

核心思想（按需求）：
    声呐图像是「笛卡尔重采样」图，径向 1 个像素对应固定物理长度。
    用【已知的 1 米间隔】作为标尺，量出它占多少像素，
    就得到「毫米/像素」，然后用它去量目标 → 体长。

两种标定方式：
  ① 几何标定（默认）
       径向 mm/px = 量程 maxDist × 1000 / 图像高
     例：maxDist=1.70 m, 图像高 512  ->  3.32 mm/像素
     （实测图像径向像素尺寸不随距离变化，符合笛卡尔重采样，故该比例与距离无关）

  ② 实测标定（推荐，用你自己量的 1 米间隔）
       在图上量出相距 1.00 m 的两个特征点各占多少像素，
       径向 mm/px = 1000 / 这两点在竖直方向上的像素距离
       或直接给一个比例：--mm-per-px 12.5

用 --ground-truth 25 可以：
       · 自动反推所需比例，并给出「修正系数」
       · 报告体长估计与距离的相关性（正确标定下应接近 0）

体长估计给出三个量：
    L_radial  径向长度 = 边界框高(px) × 径向mm/px        （沿声波方向）
    L_azimuth 方位向长度 = 边界框宽(px) × 该目标处的方位向mm/px
    L_eq      等效体长 = 2√(面积/π)，面积用径向×方位向分辨率换算
    最佳估计  由这三个量按方式选择（见 --estimate）

用法：
    python sonar_length.py <数据.DB> [输出目录]
    python sonar_length.py <数据.DB> <输出目录> --ground-truth 25
    python sonar_length.py <数据.DB> <输出目录> --mm-per-px 12.5
    python sonar_length.py <数据.DB> <输出目录> --ref-y1 50 --ref-y2 350 --ref-dist 1.0

依赖：PySide6, numpy, opencv-python, scipy, matplotlib, pyLhForwardSDK_module
环境：yazhou_py311
"""

import sys
import os
import csv
import argparse
import importlib.util

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyLhForwardSDK_module as m
from PySide6.QtGui import QImage

# ---------------------------------------------------------------------------
# 复用已有模块
# ---------------------------------------------------------------------------
PACK_SIZE = m.LhForwardFrame.PACK_SIZE


def _load(name):
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, name)
    if not os.path.exists(p):
        raise RuntimeError("找不到 %s" % name)
    spec = importlib.util.spec_from_file_location("_mod_" + name.replace(".", "_"), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SP = _load("sonar_params.py")        # 复用 detect / build_index / 解码
SB = _load("sonar_behavior.py")      # 复用 decode_frame

DEFAULTS = dict(
    threshold=160,
    min_area=20,
    kernel=1,
    use_open=True,
    use_close=False,
    estimate="eq",          # radial | azimuth | eq | max
    ground_truth=None,      # 已知体长 cm，用于反推比例
    mm_per_px=None,         # 直接指定径向 mm/像素
    ref_y1=None, ref_y2=None, ref_dist=None,   # 实测标定：两点y + 真实距离(m)
    save_frames=6,
)


# ---------------------------------------------------------------------------
# 1. 标定
# ---------------------------------------------------------------------------
def calibrate(meta, cfg):
    """返回标定结果 dict"""
    H = meta["h"]
    W = meta["w"]
    maxDist = float(meta["maxDist"])
    hor = float(meta["hor"])

    # ---- 径向（沿声波方向，即图像竖直方向）----
    mm_radial_geom = maxDist * 1000.0 / H          # 几何值
    mode = "geometry"
    if cfg.get("mm_per_px"):
        mm_radial = float(cfg["mm_per_px"])
        mode = "manual(mm_per_px)"
    elif cfg.get("ref_y1") is not None and cfg.get("ref_y2") is not None \
            and cfg.get("ref_dist"):
        dy = abs(float(cfg["ref_y2"]) - float(cfg["ref_y1"]))
        mm_radial = float(cfg["ref_dist"]) * 1000.0 / max(1.0, dy)
        mode = "measured(1m reference)"
    else:
        mm_radial = mm_radial_geom

    # ---- 方位向（图像水平方向；弧长随距离变化）----
    # 最远距离处的每像素弧长
    mm_azimuth_at_max = (2 * np.pi * maxDist * (hor / 360.0)) / W * 1000.0

    return dict(
        h=H, w=W, maxDist=maxDist, hor=hor,
        mm_radial=mm_radial,
        mm_radial_geom=mm_radial_geom,
        mm_azimuth_at_max=mm_azimuth_at_max,
        mode=mode,
    )


def azimuth_mm_per_px(cal, r_m):
    """给定距离 r(m)，返回该处的方位向 mm/像素。"""
    if cal["maxDist"] <= 0:
        return cal["mm_azimuth_at_max"]
    return cal["mm_azimuth_at_max"] * (r_m / cal["maxDist"])


def fmt_cal(cal):
    lines = []
    lines.append("【标定结果】")
    lines.append("  图像尺寸            : %d x %d" % (cal["w"], cal["h"]))
    lines.append("  量程 maxDist        : %.3f m" % cal["maxDist"])
    lines.append("  水平开角            : %.1f 度" % cal["hor"])
    lines.append("  标定方式            : %s" % cal["mode"])
    lines.append("  径向   mm/像素      : %.4f   (几何值 %.4f)"
                 % (cal["mm_radial"], cal["mm_radial_geom"]))
    lines.append("  方位向 mm/像素(最远): %.4f" % cal["mm_azimuth_at_max"])
    lines.append("  1.00 m 在径向上占   : %.1f 像素" % (1000.0 / cal["mm_radial"]))
    if cal["mm_radial"] > 0:
        lines.append("  图像总径向跨度      : %.2f m  (图像高 x 径向mm/px)"
                     % (cal["h"] * cal["mm_radial"] / 1000.0))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. 目标体长测量
# ---------------------------------------------------------------------------
def measure_targets(gray, cal, cfg):
    """检测目标并测量体长。返回 (vis, 目标列表)"""
    thr = cfg["threshold"]
    _, mask, targets, H = SP.detect(gray, thr, cfg["min_area"], cfg["kernel"],
                                    cfg["use_open"], cfg["use_close"])
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    W = gray.shape[1]
    out = []
    for (x, y, w, h, area, mean_v, cy, cx) in targets:
        r_m = cal["maxDist"] * cy / H                   # 目标距离(m)
        mm_az = azimuth_mm_per_px(cal, r_m)
        mm_ra = cal["mm_radial"]

        L_radial = h * mm_ra / 10.0                     # cm
        L_azimuth = w * mm_az / 10.0                    # cm
        A_cm2 = (area * mm_ra * mm_az) / 100.0          # cm^2
        L_eq = 2.0 * np.sqrt(max(A_cm2, 0) / np.pi)     # cm

        if cfg["estimate"] == "radial":
            L_best, src = L_radial, "径向"
        elif cfg["estimate"] == "azimuth":
            L_best, src = L_azimuth, "方位向"
        elif cfg["estimate"] == "max":
            L_best, src = max(L_radial, L_azimuth), "取大"
        else:
            L_best, src = L_eq, "等效"

        out.append(dict(x=x, y=y, w=w, h=h, area=area, mean=mean_v,
                        cx=cx, cy=cy, r_m=r_m, mm_az=mm_az, mm_ra=mm_ra,
                        L_radial=L_radial, L_azimuth=L_azimuth,
                        A_cm2=A_cm2, L_eq=L_eq, L=L_best, src=src))
    return vis, out


# ---------------------------------------------------------------------------
# 3. 反推比例（用已知体长）
# ---------------------------------------------------------------------------
def invert_scale(rows, gt_cm):
    """已知真实体长 -> 反推需要的径向 mm/像素"""
    if not rows:
        return None
    hpx = np.array([r["h"] for r in rows], float)
    med = float(np.median(hpx))
    if med <= 0:
        return None
    need_mm = gt_cm * 10.0 / med          # mm/px
    return dict(median_hpx=med, need_mm_per_px=need_mm)


# ---------------------------------------------------------------------------
# 4. 输出
# ---------------------------------------------------------------------------
def draw(vis, rows, cal, cfg, gt_cm=None):
    """标注每个目标的体长"""
    for r in rows:
        cv2.rectangle(vis, (r["x"], r["y"]),
                      (r["x"] + r["w"], r["y"] + r["h"]), (0, 0, 255), 1)
        cv2.putText(vis, "%.1fcm" % r["L"],
                    (r["x"], max(12, r["y"] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    txt = "n=%d  med=%.1fcm  radial=%.3fmm/px" % (
        len(rows), np.median([r["L"] for r in rows]) if rows else 0, cal["mm_radial"])
    cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if gt_cm:
        cv2.putText(vis, "GT=%.0fcm" % gt_cm, (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return vis


def write_csv(rows, path):
    cols = ["frame", "x", "y", "w", "h", "area", "mean", "cx", "cy",
            "r_m", "mm_az", "mm_ra", "L_radial", "L_azimuth", "A_cm2", "L_eq", "L", "src"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def plot(rows, cal, cfg, gt_cm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
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

    L = np.array([r["L"] for r in rows], float)
    rr = np.array([r["r_m"] for r in rows], float)
    Lr = np.array([r["L_radial"] for r in rows], float)
    La = np.array([r["L_azimuth"] for r in rows], float)
    Le = np.array([r["L_eq"] for r in rows], float)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0][0].hist(L, bins=30, color="#1f77b4", alpha=0.8)
    if gt_cm:
        ax[0][0].axvline(gt_cm, color="r", ls="--", lw=2, label="已知体长 %.0fcm" % gt_cm)
        ax[0][0].legend()
    ax[0][0].set_xlabel("体长估计 (cm)")
    ax[0][0].set_ylabel("目标数")
    ax[0][0].set_title("体长分布  中位=%.1fcm" % np.median(L))
    ax[0][0].grid(alpha=0.3)

    ax[0][1].scatter(rr, L, s=8, alpha=0.5)
    if len(rr) > 2:
        k = np.polyfit(rr, L, 1)
        xs = np.linspace(rr.min(), rr.max(), 20)
        ax[0][1].plot(xs, np.polyval(k, xs), "r-", lw=1.5,
                      label="拟合 斜率=%.2f cm/m" % k[0])
        c = np.corrcoef(rr, L)[0, 1]
        ax[0][1].legend(title="相关系数 r=%.3f" % c)
    ax[0][1].set_xlabel("目标距离 (m)")
    ax[0][1].set_ylabel("体长估计 (cm)")
    ax[0][1].set_title("体长 vs 距离（正确标定应无相关性）")
    ax[0][1].grid(alpha=0.3)

    ax[1][0].scatter(rr, Lr, s=8, alpha=0.5, label="径向")
    ax[1][0].scatter(rr, La, s=8, alpha=0.5, label="方位向")
    ax[1][0].scatter(rr, Le, s=8, alpha=0.5, label="等效")
    ax[1][0].set_xlabel("目标距离 (m)")
    ax[1][0].set_ylabel("长度 (cm)")
    ax[1][0].set_title("三种测量方式对比")
    ax[1][0].legend(fontsize=9)
    ax[1][0].grid(alpha=0.3)

    ax[1][1].hist(np.array([r["h"] for r in rows], float), bins=25,
                  alpha=0.6, label="边界框高(径向)")
    ax[1][1].hist(np.array([r["w"] for r in rows], float), bins=25,
                  alpha=0.6, label="边界框宽(方位向)")
    ax[1][1].set_xlabel("像素")
    ax[1][1].set_ylabel("目标数")
    ax[1][1].set_title("目标像素尺寸分布")
    ax[1][1].legend(fontsize=9)
    ax[1][1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    print("[绘图] 已保存:", path)


def write_summary(rows, cal, cfg, gt_cm, path, extra=""):
    L = np.array([r["L"] for r in rows], float) if rows else np.array([])
    Lr = np.array([r["L_radial"] for r in rows], float) if rows else np.array([])
    La = np.array([r["L_azimuth"] for r in rows], float) if rows else np.array([])
    Le = np.array([r["L_eq"] for r in rows], float) if rows else np.array([])
    rr = np.array([r["r_m"] for r in rows], float) if rows else np.array([])

    lines = []
    lines.append("=" * 74)
    lines.append("声呐鱼体长识别 —— 测量报告")
    lines.append("=" * 74)
    lines.append("")
    lines.append(fmt_cal(cal))
    lines.append("")
    lines.append("【检测参数】")
    for k in ("threshold", "min_area", "kernel", "use_open", "use_close", "estimate"):
        lines.append("  %-12s = %s" % (k, cfg[k]))
    lines.append("")
    lines.append("【体长统计】  目标数 = %d" % len(rows))
    if len(L):
        for nm, a in (("径向 L_radial", Lr), ("方位向 L_azimuth", La), ("等效 L_eq", Le)):
            lines.append("  %-16s 中位=%7.2f cm  均值=%7.2f  std=%6.2f  min=%6.2f  max=%7.2f"
                         % (nm, np.median(a), a.mean(), a.std(), a.min(), a.max()))
        lines.append("  %-16s 中位=%7.2f cm  （本次采用：%s）"
                     % ("最终 L", np.median(L), rows[0]["src"]))
    lines.append("")

    # ---- 标定校核 ----
    if gt_cm:
        inv = invert_scale(rows, gt_cm)
        lines.append("【标定校核（已知真实体长 %.0f cm）】" % gt_cm)
        if inv:
            lines.append("  目标边界框高 中位      : %.1f 像素" % inv["median_hpx"])
            lines.append("  对应所需径向比例       : %.3f mm/像素" % inv["need_mm_per_px"])
            lines.append("  当前使用的径向比例     : %.3f mm/像素" % cal["mm_radial"])
            ratio = inv["need_mm_per_px"] / cal["mm_radial"] if cal["mm_radial"] else 0
            lines.append("  两者相差               : %.2f 倍" % ratio)
            lines.append("")
            if ratio > 1.5 or ratio < 0.67:
                lines.append("  ⚠ 相差较大，说明【当前几何标定与实际不符】。可能原因：")
                lines.append("     a) 阈值分割只取到鱼体最亮的回声核心，未覆盖整条鱼")
                lines.append("        -> 应降低阈值 / 做闭运算把回声连成一片后再量")
                lines.append("     b) 声呐实际成像的距离跨度大于标称 maxDist（如 maxDist 是显示档位）")
                lines.append("        -> 用 --ref-y1/--ref-y2/--ref-dist 或 --mm-per-px 做实测标定")
                lines.append("     c) 检测到的是鱼群/散射体而非单条鱼")
                lines.append("     -> 用 --mm-per-px %.3f 可以强制按已知体长标定" % inv["need_mm_per_px"])
            else:
                lines.append("  ✓ 两者接近，几何标定与实测一致。")
        lines.append("")

    if len(rr) > 2:
        c = np.corrcoef(rr, L)[0, 1]
        k = np.polyfit(rr, L, 1)
        lines.append("【体长与距离的相关性】")
        lines.append("  相关系数 r = %.3f   拟合斜率 = %.2f cm/m   (样本 %d)" % (c, k[0], len(rr)))
        lines.append("  判读：真实体长与距离无关，故 r 越接近 0 说明标定越正确。")
        if abs(c) > 0.35:
            lines.append("  ⚠ |r|=%.2f 偏大，说明体长估计随距离系统性变化，标定存在问题。" % abs(c))
        else:
            lines.append("  ✓ |r|=%.2f 较小，标定基本自洽。" % abs(c))
        lines.append("")

    lines.append("【重要说明 / 局限性】")
    lines.append("  1) 体长由【像素尺寸 × 标定比例】得到，标定比例错多少，体长就错多少。")
    lines.append("  2) 前视声呐只有方位×距离两个维度，没有俯仰分辨率，")
    lines.append("     测得的是目标在【声呐成像平面上的投影尺寸】，不是鱼的完整三维体长。")
    lines.append("  3) 鱼体姿态（头朝/侧对声呐）会让同一条鱼量出不同长度，这是原理性误差。")
    lines.append("  4) 阈值分割得到的是回声亮斑，通常【小于】鱼的物理尺寸")
    lines.append("     （鱼体大部分区域回波弱），所以体长估计一般偏小。")
    lines.append("  5) 雷达方程意义上，回声强度与体长的关系不是线性的，")
    lines.append("     更严谨的做法是用目标强度 TS 反演体长（TS = 20logL + a·logf + b）。")
    lines.append("  6) 建议用【已知体长的目标】做一次标定，再批量测量。")
    if extra:
        lines.append("  7) " + extra)
    lines.append("")
    txt = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def run(db, outdir, **ov):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in ov.items() if v is not None})
    os.makedirs(outdir, exist_ok=True)

    print("=" * 74)
    print("数据文件:", db)
    print("输出目录:", outdir)
    print("=" * 74)

    print("\n[1/5] 建立帧索引…")
    starts = SB.build_index(db, log=lambda s: print("      " + s))
    total = len(starts)
    print("      共 %d 帧" % total)

    print("\n[2/5] 标定…")
    g0, meta0 = SB.decode_frame(db, starts, 0)
    if g0 is None:
        print("解码失败"); return
    cal = calibrate(meta0, cfg)
    print(fmt_cal(cal))

    step = max(1, int(cfg.get("frame_step", 1) or 1))
    print("\n[3/5] 逐帧测量体长（步长 %d）…" % step)
    allrows = []
    vis_cache = []
    for i in range(0, total, step):
        g, meta = SB.decode_frame(db, starts, i)
        if g is None:
            continue
        g = cv2.flip(g, 0)
        vis, rows = measure_targets(g, cal, cfg)
        for r in rows:
            r["frame"] = i
        allrows.extend(rows)
        if len(vis_cache) < cfg["save_frames"]:
            vis_cache.append((i, g, vis))
        if (i // step) % 100 == 0:
            print("      帧 %d / %d   累计目标 %d" % (i, total, len(allrows)))

    if not allrows:
        print("未检测到任何目标")
        return
    print("      共测量 %d 个目标" % len(allrows))

    print("\n[4/5] 反推标定（若给了已知体长）…")
    inv = invert_scale(allrows, cfg["ground_truth"]) if cfg.get("ground_truth") else None
    if inv:
        print("      边界框高中位 %.1f px -> 所需径向比例 %.3f mm/px (当前 %.3f)"
              % (inv["median_hpx"], inv["need_mm_per_px"], cal["mm_radial"]))

    print("\n[5/5] 输出…")
    csv_path = os.path.join(outdir, "length_frames.csv")
    write_csv(allrows, csv_path)
    print("      逐目标结果:", csv_path)

    txt_path = os.path.join(outdir, "length_summary.txt")
    s = write_summary(allrows, cal, cfg, cfg.get("ground_truth"), txt_path,
                      extra="本次共分析 %d 帧、%d 个目标。" % (total, len(allrows)))
    print("      测量报告:", txt_path)

    plot(allrows, cal, cfg, cfg.get("ground_truth"),
         os.path.join(outdir, "length_analysis.png"))

    imgdir = os.path.join(outdir, "length_frames")
    os.makedirs(imgdir, exist_ok=True)
    for (i, g, vis) in vis_cache:
        v2 = draw(vis.copy(), [r for r in allrows if r["frame"] == i],
                  cal, cfg, cfg.get("ground_truth"))
        ok, enc = cv2.imencode(".png", v2)
        if ok:
            with open(os.path.join(imgdir, "len_%05d.png" % i), "wb") as f:
                f.write(enc.tobytes())
    print("      标注帧图片:", imgdir)

    print("\n" + s)


def main():
    ap = argparse.ArgumentParser(description="声呐鱼体长识别（量程/距离标定）")
    ap.add_argument("db")
    ap.add_argument("outdir", nargs="?", default=None)
    ap.add_argument("--ground-truth", type=float, default=None,
                    help="已知真实体长(cm)，用于反推标定比例。例如 25")
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="直接指定径向毫米/像素（实测标定）")
    ap.add_argument("--ref-y1", type=int, default=None, help="实测标定：参考点1 的 y 像素")
    ap.add_argument("--ref-y2", type=int, default=None, help="实测标定：参考点2 的 y 像素")
    ap.add_argument("--ref-dist", type=float, default=None,
                    help="实测标定：两参考点之间的真实距离(m)，例如 1.0")
    ap.add_argument("--estimate", choices=["radial", "azimuth", "eq", "max"],
                    default=None, help="体长取值方式，默认 eq(等效)")
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--kernel", type=int, default=None)
    ap.add_argument("--step", type=int, default=None, help="抽帧步长")
    ap.add_argument("--no-open", action="store_true", help="关闭开运算")
    args = ap.parse_args()

    outdir = args.outdir or (os.path.splitext(os.path.basename(args.db))[0] + "_length")
    run(args.db, outdir,
        ground_truth=args.ground_truth,
        mm_per_px=args.mm_per_px,
        ref_y1=args.ref_y1, ref_y2=args.ref_y2, ref_dist=args.ref_dist,
        estimate=args.estimate,
        threshold=args.threshold, min_area=args.min_area, kernel=args.kernel,
        frame_step=args.step,
        use_open=(False if args.no_open else None))


if __name__ == "__main__":
    main()
