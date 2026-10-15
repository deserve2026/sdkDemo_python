# -*- coding: utf-8 -*-
"""
sonar_html.py —— 用 QtWebEngine 把你的 HTML 界面接到真实声呐数据上

设计思路（最小可行版）：
    1. 不修改你的原始 HTML，在【运行时】注入一段桥接脚本
    2. 停掉 HTML 里的模拟动画（Math.random 画的假声呐图）
    3. 由 Python 提供真实帧（.DB 回放 / 实时 UDP 两种数据源）
    4. 通过 QWebChannel 双向通信：
         Python --显示帧+数量--> JS
         JS     --按钮/滑块---> Python

用法：
    python sonar_html.py                          # 弹窗选 .DB
    python sonar_html.py "D:/data/1.DB"           # 直接打开
    python sonar_html.py --realtime               # 实时模式（接声呐设备）

依赖：PySide6（含 QtWebEngine）、numpy、opencv-python、pyLhForwardSDK_module
务必使用 yazhou_py311 环境。
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt, QTimer, QObject, Slot, QUrl
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel

import numpy as np

HTML_FILE = "deepseek_html_20260616_4b7296.html"
WIN_W, WIN_H = 1000, 860


# ===========================================================================
# 1. HTML 注入脚本：让页面接收真实数据，并停掉模拟动画
# ===========================================================================
INJECT_HEAD = """
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
"""

INJECT_TAIL = r"""
<script>
// ==================== 真实数据桥接（由 sonar_html.py 注入）====================
(function () {
  "use strict";
  if (window.__REAL_BRIDGE__) { return; }
  window.__REAL_BRIDGE__ = true;

  var bridge = null;
  var realMode = false;       // true = 用真实帧，停止模拟绘制
  var lastCount = -1;
  var frameSeq = 0;
  var fpsMark = Date.now(), fpsN = 0, fpsVal = 0;

  function log(msg) {
    try { if (bridge && bridge.js_log) { bridge.js_log(String(msg)); } } catch (e) {}
    console.log("[bridge] " + msg);
  }

  // ---- 对外暴露：Python 调用 ----
  window.__real = {
    // 初始化：告诉页面进入真实模式
    init: function (sourceName) {
      realMode = true;
      window.__realActive = true;
      killMockTimers();          // 先清掉页面里的模拟定时器
      startClock();
      ensureCountElement();
      try {
        if (typeof isSonarActive !== "undefined") { isSonarActive = true; }
      } catch (e) {}
      var el = document.getElementById("connectionStatus");
      if (el) { el.textContent = sourceName || "已连接"; }
      log("真实模式已启动: " + sourceName);
    },

    // 显示一帧（dataUrl 为 JPEG/PNG 的 data: 形式）
    showFrame: function (dataUrl, infoJson) {
      var canvas = document.getElementById("sonarCanvas");
      if (!canvas) { return; }
      var ctx = canvas.getContext("2d");
      var info = {};
      try { info = JSON.parse(infoJson || "{}"); } catch (e) {}

      var img = new Image();
      img.onload = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        frameSeq++;
        // 帧率统计
        var now = Date.now();
        if (now - fpsMark >= 1000) {
          fpsVal = fpsN * 1000 / (now - fpsMark);
          fpsN = 0; fpsMark = now;
        }
        // 数量变化时通知 Python（供日志/记录用）
        if (info.count !== undefined && info.count !== lastCount) {
          lastCount = info.count;
          try { if (bridge && bridge.on_count) { bridge.on_count(info.count); } } catch (e) {}
        }
        updateInfo(info);
        // 时钟（原页面的定时器已被清除，这里由帧驱动更新）
        if (frameSeq % 20 === 1) { startClock(); }
      };
      img.onerror = function () { log("帧解码失败"); };
      img.src = dataUrl;
      fpsN++;
    },

    // 更新顶部/左侧信息区
    setInfo: function (text) {
      try { updateInfo(JSON.parse(text)); } catch (e) {}
    },

    status: function (msg, isErr) {
      if (typeof showStatus === "function") { showStatus(msg, !!isErr); }
    }
  };

  function updateInfo(info) {
    // 帧号：优先使用页面元素，格式化为 5 位
    var fnum = info.frame !== undefined ? info.frame : 0;
    var ftotal = info.total !== undefined && info.total > 0 ? info.total : null;
    var fstr = String(fnum).padStart(5, "0") +
               (ftotal ? " / " + String(ftotal).padStart(5, "0") : "");
    var fd = document.getElementById("frameDisplay");
    if (fd) { fd.textContent = fstr; }

    // 水深/量程：用真实 maxDist
    var dd = document.getElementById("depthDisplay");
    if (dd && info.range !== undefined) {
      dd.textContent = Number(info.range).toFixed(2);
    }

    var cs = document.getElementById("connectionStatus");
    if (cs && info.device) { cs.textContent = info.device; }

    // 计数显示
    var cntEl = document.getElementById("fishCount") ||
                document.getElementById("countDisplay");
    if (cntEl && info.count !== undefined) {
      cntEl.textContent = "🐟 " + info.count;
    }
    // 把计数也写进状态栏文字
    if (info.count !== undefined && typeof showStatus === "function") {
      window.__countText = "当前帧目标数：" + info.count;
    }
  }

  // ---- 停掉模拟：包装动画循环，真实模式下不画假图 ----
  try {
    if (typeof drawSonar === "function") {
      var _origDrawSonar = drawSonar;
      drawSonar = function (active, time) {
        if (realMode || window.__realActive) { return; }   // 真实模式：不画模拟图
        return _origDrawSonar(active, time);
      };
    }
  } catch (e) { log("包装 drawSonar 失败: " + e); }

  // ---- 停掉模拟状态刷新（原函数会每秒写随机水深，污染真实数据）----
  // 注意：setInterval(updateStatus, 1000) 在页面脚本里已注册，
  // 此时它捕获的是【旧引用】，包装 updateStatus 无法拦住它。
  // 所以进入真实模式时直接清掉所有已注册的定时器。
  var _cleared = false;
  function killMockTimers() {
    if (_cleared) { return; }
    _cleared = true;
    try {
      var n = setTimeout(function () {}, 0);
      for (var i = 1; i <= n; i++) {
        clearTimeout(i);
        clearInterval(i);
      }
      log("已清除页面模拟定时器 (最高 id=" + n + ")");
    } catch (e) { log("清除定时器失败: " + e); }
  }

  // 自己的时钟（不受清除影响）
  function startClock() {
    try {
      var now = new Date();
      var el = document.getElementById("dateDisplay");
      if (el) { el.textContent = now.toLocaleString("zh-CN", { hour12: false }); }
    } catch (e) {}
  }

  // ---- 在左侧状态区插入"鱼群数量"行（原页面没有这个元素）----
  function ensureCountElement() {
    if (document.getElementById("fishCount")) { return; }
    var host = document.querySelector(".status-lines");
    if (!host) { return; }
    var row = document.createElement("div");
    row.innerHTML = '目标数：<span id="fishCount" style="' +
      'color:#00FF66;font-weight:bold;font-size:15px;' +
      'text-shadow:0 0 6px rgba(0,255,102,0.6);">--</span>';
    host.appendChild(row);
    log("已插入鱼群数量显示行");
  }

  // ---- 按钮改为通知 Python，不再弹模拟框 ----
  function hook(btnId, action) {
    var b = document.getElementById(btnId);
    if (!b) { return; }
    b.addEventListener("click", function (ev) {
      if (bridge && bridge.on_button) {
        bridge.on_button(action, b.textContent.trim());
      }
    }, true);   // 捕获阶段，先于页面自身的 handler
  }
  hook("connectBtn",  "connect");
  hook("powerBtn",    "power");
  hook("playbackBtn", "playback");
  hook("recordBtn",   "record");
  hook("countBtn",    "count");
  hook("lengthBtn",   "length");
  hook("actionBtn",   "action");

  // ---- 滑块改为通知 Python ----
  function hookSlider(id, name) {
    var s = document.getElementById(id);
    if (!s) { return; }
    s.addEventListener("input", function () {
      if (bridge && bridge.on_slider) { bridge.on_slider(name, s.value); }
    });
  }
  hookSlider("gammaSlider", "gamma");
  hookSlider("brightnessSlider", "brightness");
  hookSlider("distanceSlider", "distance");
  hookSlider("modeSlider", "mode");

  // ---- 建立 WebChannel ----
  if (typeof QWebChannel === "undefined") {
    log("QWebChannel 未加载：qwebchannel.js 不可用");
  } else {
    new QWebChannel(qt.webChannelTransport, function (channel) {
      bridge = channel.objects.bridge;
      window.__bridge = bridge;
      log("WebChannel 已连通");
      if (bridge && bridge.js_ready) { bridge.js_ready(); }
    });
  }

  log("注入脚本执行完毕");
})();
</script>
"""


# ===========================================================================
# 2. 帧数据源抽象：.DB 回放 与 实时 UDP 共用同一接口
# ===========================================================================
class FrameSource:
    """统一的帧来源接口。子类实现 open() / frame_count() / get_frame(i) / close()"""

    def open(self):
        raise NotImplementedError

    def frame_count(self):
        return 0

    def get_frame(self, idx):
        """返回 (gray_ndarray, meta_dict) 或 (None, None)"""
        raise NotImplementedError

    def close(self):
        pass

    @property
    def name(self):
        return "未知数据源"


def _make_decoder():
    """把 sonar_params.py 里的解码逻辑复用过来（避免重复实现）。"""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "sonar_params.py")
    if not os.path.exists(p):
        return None, None
    spec = importlib.util.spec_from_file_location("_sp_for_html", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.Decoder


class DbFrameSource(FrameSource):
    """从 .DB 文件读取帧（回放）。"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.starts = []
        self._sp = None
        self._dec = None
        self._q = None
        self._t = None

    @property
    def name(self):
        return "回放: " + os.path.basename(self.db_path)

    def open(self):
        sp, Decoder = _make_decoder()
        if sp is None:
            raise RuntimeError("找不到 sonar_params.py，无法复用解码逻辑")
        self._sp = sp
        self.starts = sp.build_index(self.db_path, log=lambda s: print("[索引]", s))

        import queue as _queue
        self._q = _queue.Queue(maxsize=4)

        self._dec = Decoder(self.db_path, self.starts)

        def _on_frame(i, g, m):
            try:
                self._q.put_nowait((i, g, m))
            except Exception:
                pass          # 队列满则丢弃（回放时正好起到限流作用）

        self._dec.frameReady.connect(_on_frame)
        self._dec.start()
        return self

    def frame_count(self):
        return len(self.starts)

    def request(self, idx):
        """异步请求某一帧（结果稍后从 take() 取）"""
        if self._dec is not None:
            self._dec.request(idx)

    def take(self, timeout=0.05):
        try:
            return self._q.get(timeout=timeout)
        except Exception:
            return None

    def close(self):
        if self._dec is not None:
            self._dec.stop()
            self._dec.wait(2000)


# ===========================================================================
# 3. Python <-> JS 桥
# ===========================================================================
class Bridge(QObject):
    def __init__(self, window):
        super().__init__()
        self.win = window

    @Slot()
    def js_ready(self):
        self.win.on_js_ready()

    @Slot(str)
    def js_log(self, msg):
        print("[JS]", msg)

    @Slot(str, str)
    def on_button(self, action, text):
        self.win.on_button(action, text)

    @Slot(str, str)
    def on_slider(self, name, value):
        self.win.on_slider(name, value)

    @Slot(int)
    def on_count(self, n):
        self.win.on_count(n)


# ===========================================================================
# 4. 主窗口
# ===========================================================================
class SonarHtmlWindow(QMainWindow):
    def __init__(self, db_path=None, realtime=False):
        super().__init__()
        self.setWindowTitle("SapphireView · 真实数据桥接版")
        self.resize(WIN_W, WIN_H)

        self.db_path = db_path
        self.realtime = realtime
        self.source = None
        self.cur_idx = 0
        self.total = 0
        self.playing = False
        self.detector_ready = False
        self._sp = None

        self.view = QWebEngineView(self)
        self.setCentralWidget(self.view)

        self.channel = QWebChannel()
        self.bridge = Bridge(self)
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        self.load_html()
        self.view.loadFinished.connect(self.on_load_finished)

    # ---------------- HTML 加载与注入 ----------------
    def load_html(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, HTML_FILE)
        if not os.path.exists(path):
            QMessageBox.critical(self, "错误", "找不到 HTML 文件：\n" + path)
            return
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        # 注入 qwebchannel.js（若页面里还没有）
        if "qwebchannel.js" not in html:
            html = html.replace("</head>", INJECT_HEAD + "</head>", 1)

        # 注入桥接脚本（放在最后，确保页面函数都已定义）
        html = html.replace("</body>", INJECT_TAIL + "</body>", 1)

        # 用 setHtml + baseUrl，保证相对路径的图片（校徽）能加载
        base = QUrl.fromLocalFile(here + os.sep)
        self.view.setHtml(html, base)

    def on_load_finished(self, ok):
        print("[Python] HTML 加载", "成功" if ok else "失败")
        if not ok:
            return
        self.view.page().runJavaScript(
            "window.__real && window.__real.status('页面已就绪，等待数据…');")

    # ---------------- 桥事件 ----------------
    def on_js_ready(self):
        print("[Python] JS 桥已就绪")
        if self.db_path:
            QTimer.singleShot(200, lambda: self.open_db(self.db_path))
        elif not self.realtime:
            QTimer.singleShot(400, self.choose_file)

    def on_button(self, action, text):
        print("[Python] 按钮:", action, "(文字:%s)" % text)
        js = self.view.page().runJavaScript

        if action == "playback":
            if self.total == 0:
                self.choose_file()
            else:
                self.playing = not self.playing
                if self.playing:
                    self.timer.start(100)          # 10 fps
                    js("window.__real.status('回放中…')")
                else:
                    self.timer.stop()
                    js("window.__real.status('已暂停')")

        elif action == "count":
            js("window.__real.status('计数已开启（当前为阈值分割）')")

        elif action == "record":
            js("window.__real.status('录制功能暂未实现', true)")

        elif action == "connect":
            if self.realtime:
                js("window.__real.status('实时模式：请确认网线已连接声呐')")
            else:
                js("window.__real.status('当前为回放模式')")

        elif action in ("power", "length", "action"):
            js("window.__real.status('该功能待接入')")

    def on_slider(self, name, value):
        print("[Python] 滑块:", name, "=", value)

    def on_count(self, n):
        print("[Python] 当前帧鱼群数量:", n)

    # ---------------- 数据源 ----------------
    def choose_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择声呐数据文件", "",
                                           "蓝衡数据格式 (*.DB)")
        if p:
            self.open_db(p)

    def open_db(self, path):
        self.db_path = path
        try:
            self.source = DbFrameSource(path).open()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "打开失败", "%s\n\n%s" % (e, traceback.format_exc()))
            return
        self.total = self.source.frame_count()
        self.cur_idx = 0
        print("[Python] 已打开 %s，共 %d 帧" % (os.path.basename(path), self.total))
        self.view.page().runJavaScript(
            "window.__real.init(%s);" % json.dumps(self.source.name, ensure_ascii=False))
        # 先显示第一帧
        self.show_frame(0)
        # 自动开始回放
        self.playing = True
        self.timer.start(100)

    # ---------------- 帧 -> JS ----------------
    def on_tick(self):
        if self.source is None or self.total == 0:
            return
        self.cur_idx += 1
        if self.cur_idx >= self.total:
            self.cur_idx = 0
        self.source.request(self.cur_idx)

    def show_frame(self, idx):
        """同步取一帧并推给 JS（用于首帧显示）"""
        if self.source is None:
            return
        self.source.request(idx)
        r = self.source.take(timeout=2.0)
        if r is None:
            return
        _, gray, meta = r
        self.push(gray, meta)

    def push(self, gray, meta):
        """把一帧灰度图 + 信息推给 JS"""
        import cv2

        # 检测：跑一遍阈值分割，得到鱼群数量
        count = 0
        try:
            if self._sp is None:
                self._sp, _ = _make_decoder()
            if self._sp is not None and gray is not None:
                flipped = True
                g = cv2.flip(gray, 0) if flipped else gray
                vis, mask, targets, _ = self._sp.detect(g, 160, 20, 1, True, False)
                count = len(targets)
                for (x, y, w, h, area, mv, cy, cx) in targets:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                out = vis
            else:
                out = gray
        except Exception as e:
            print("[Python] 检测失败:", e)
            out = gray

        # 编码为 JPEG（比 PNG 小很多，适合逐帧推送）
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        import base64
        b64 = base64.b64encode(enc.tobytes()).decode("ascii")
        data_url = "data:image/jpeg;base64," + b64

        info = {
            "frame": meta.get("frame", 0),
            "total": self.total,
            "count": count,
            "range": float(meta.get("maxDist", 0.0)),
            "device": meta.get("device", ""),
            "packType": meta.get("packType", -1),
        }
        # 每次 tick 从这里取解码结果（异步）
        js = "window.__real.showFrame(%s, %s);" % (
            json.dumps(data_url), json.dumps(json.dumps(info, ensure_ascii=False)))
        self.view.page().runJavaScript(js)
        # 状态栏
        self.view.page().runJavaScript(
            "window.__real.status('帧 %d/%d · 鱼群 %d · %.2fm');" % (
                info["frame"], max(0, self.total - 1), count, info["range"]))

    def closeEvent(self, e):
        self.timer.stop()
        if self.source is not None:
            self.source.close()
        e.accept()


# ===========================================================================
# 5. 入口
# ===========================================================================
def main():
    app = QApplication(sys.argv)
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    realtime = "--realtime" in sys.argv
    db = argv[0] if argv else None

    import cv2 as _cv2  # noqa: F401  提前校验依赖
    w = SonarHtmlWindow(db_path=db, realtime=realtime)
    w.show()
    # 异步解码结果回推
    w.timer2 = QTimer(w)
    def drain():
        if w.source is None:
            return
        r = w.source.take(timeout=0.001)
        if r is not None:
            _, gray, meta = r
            w.push(gray, meta)
    w.timer2.timeout.connect(drain)
    w.timer2.start(30)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
