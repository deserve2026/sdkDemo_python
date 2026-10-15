import base64
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject, QDir, QEvent, QFile, QIODevice, QSettings, QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QImage, QPixmap, QAction
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QDockWidget, QFileDialog, QHeaderView, QMainWindow, QMessageBox, QStyle, QTableWidgetItem, QTabWidget, QWidget
from .paths import PROJECT, RESULTS, output_path
from .worker import ReplayWorker

class Controller(QObject):
    def __init__(self, application, initial_path=None, smoke=False):
        super().__init__()
        self.application = application
        self.smoke = smoke
        self.gray = None
        self.source_count = 0
        self.current_index = -1
        self.generation = 0
        self.closed = False
        self.legacy_windows = []
        self.ui_path = Path(__file__).resolve().parent / 'ui' / 'main_window.ui'
        file = QFile(str(self.ui_path))
        if not file.open(QIODevice.ReadOnly):
            raise RuntimeError('Cannot open main_window.ui')
        loader = QUiLoader()
        loader.setWorkingDirectory(QDir(str(self.ui_path.parent)))
        self.window = loader.load(file)
        file.close()
        if not isinstance(self.window, QMainWindow):
            raise ValueError('Designer file must contain a QMainWindow')
        self.window.setStyleSheet((self.ui_path.parents[1] / 'theme.qss').read_text(encoding='utf-8'))
        self.widgets = {}
        self.window.installEventFilter(self)
        self.widgets = {w.objectName(): w for w in self.window.findChildren(QWidget) if w.objectName()}
        self.actions = {a.objectName(): a for a in self.window.findChildren(QAction) if a.objectName()}
        self.widgets['imageLabel'].installEventFilter(self)
        self.widgets['logText'].setMaximumBlockCount(500)
        self.widgets['eventTable'].horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.docks = [self.widgets[n] for n in ('behaviorDock', 'controlDock', 'logDock')]
        self.window.tabifyDockWidget(self.docks[0], self.docks[1])
        self.window.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
        self.docks[0].raise_()
        self.window.resizeDocks(self.docks[:2], [335, 335], Qt.Horizontal)
        self.default_layout = self.window.saveState(1)
        self.settings = QSettings(str(output_path('db_replay/settings/layout.ini')), QSettings.IniFormat)
        if not smoke:
            geometry, state = self.settings.value('geometry'), self.settings.value('state')
            if geometry is not None:
                self.window.restoreGeometry(geometry)
            if state is not None:
                self.window.restoreState(state, 1)
        for dock in self.docks:
            self.window.findChild(QObject, 'menuView').addAction(dock.toggleViewAction())
        self.worker = ReplayWorker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.source_ready.connect(self.on_source)
        self.worker.model_ready.connect(self.on_model)
        self.worker.message.connect(self.log)
        self.worker.failure.connect(self.on_error)
        self.worker.playing_changed.connect(self.on_playing)
        self.bind()
        self.worker.start()
        self.window.statusBar().showMessage('原始数据只读 · 拖动面板标题可调整布局')
        if initial_path:
            QTimer.singleShot(200, lambda: self.load(initial_path))

    def bind(self):
        w = self.widgets
        w['openDbButton'].clicked.connect(self.choose_db)
        w['openPngButton'].clicked.connect(self.choose_png)
        w['playButton'].clicked.connect(lambda checked: self.worker.command('play', checked))
        w['previousButton'].clicked.connect(lambda: self.worker.command('step', -1))
        w['nextButton'].clicked.connect(lambda: self.worker.command('step', 1))
        w['resetButton'].clicked.connect(lambda: self.worker.command('seek', 0))
        w['speedSpin'].valueChanged.connect(lambda v: self.worker.command('speed', v))
        w['frameSlider'].sliderReleased.connect(lambda: self.worker.command('seek', w['frameSlider'].value()))
        w['inferenceCheck'].toggled.connect(self.toggle_inference)
        w['reloadModelButton'].clicked.connect(lambda: self.worker.command('reload_model'))
        for name in ('flipCheck', 'countCheck'):
            w[name].toggled.connect(self.draw_frame)
        for name in ('thresholdSpin', 'areaSpin'):
            w[name].valueChanged.connect(self.draw_frame)
        w['designerButton'].clicked.connect(self.open_designer)
        w['legacyButton'].clicked.connect(self.open_legacy)
        for name, callback in {
            'actionOpenDb': self.choose_db, 'actionOpenPng': self.choose_png, 'actionExit': self.window.close,
            'actionResetLayout': self.reset_layout, 'actionDesigner': self.open_designer,
            'actionLegacy': self.open_legacy, 'actionReport': self.open_report
        }.items():
            self.actions[name].triggered.connect(callback)
        self.actions['actionLockLayout'].toggled.connect(self.lock_layout)
        icons = {'openDbButton': QStyle.SP_DialogOpenButton, 'openPngButton': QStyle.SP_DirOpenIcon,
                 'playButton': QStyle.SP_MediaPlay, 'previousButton': QStyle.SP_MediaSkipBackward,
                 'nextButton': QStyle.SP_MediaSkipForward, 'resetButton': QStyle.SP_MediaStop}
        for name, standard in icons.items():
            if w[name].icon().isNull():
                w[name].setIcon(self.window.style().standardIcon(standard))
        for name in ('playButton','previousButton','nextButton','resetButton'):
            w[name].setEnabled(False)

    def choose_db(self):
        path, _ = QFileDialog.getOpenFileName(self.window, '打开声呐 DB 文件', '', '声呐数据 (*.DB *.db)')
        if path:
            self.load(path)

    def choose_png(self):
        path = QFileDialog.getExistingDirectory(self.window, '打开包含 frame_XXXXXX.png 的文件夹')
        if path:
            self.load(path)

    def load(self, path):
        path = Path(path).resolve()
        self.gray = None
        self.generation = -1
        for name in ('playButton','previousButton','nextButton','resetButton'):
            self.widgets[name].setEnabled(False)
        self.set_state(None, '正在加载数据')
        self.widgets['eventTable'].setRowCount(0)
        self.widgets['imageLabel'].clear()
        self.widgets['imageLabel'].setText('正在建立索引，请稍候…')
        self.widgets['sourceLabel'].setText(str(path))
        self.worker.command('load', dict(path=str(path), series=path.is_file() and self.widgets['seriesCheck'].isChecked()))

    def on_source(self, info):
        self.source_count, self.generation = info['count'], info['generation']
        self.widgets['frameSlider'].setRange(0, self.source_count - 1)
        for name in ('playButton','previousButton','nextButton','resetButton'):
            self.widgets[name].setEnabled(True)
        self.widgets['sourceLabel'].setText(f"{info['kind']}  ·  {info['path']}  ·  {info['count']:,} 帧")
        self.widgets['sourceCaution'].setText(
            'DB 经 SDK 解码后上下翻转一次，匹配训练方向；其它采集条件仍需验证。回放速度不代表采集帧率。'
            if info['kind'] == 'DB' else 'PNG 使用与训练一致的固定预处理。回放速度不代表采集帧率。')
        self.log('预测记录：' + info['log_path'])

    def on_model(self, info):
        self.widgets['modelLabel'].setText('模型已就绪 · ONNX Runtime / CPU' if info['loaded'] else '模型不可用：' + info['error'])
        self.log('已加载经过数值核对的摄食模型' if info['loaded'] else '模型未加载：' + info['error'])

    def on_playing(self, playing):
        b = self.widgets['playButton']
        b.blockSignals(True)
        b.setChecked(playing)
        b.setText('暂停' if playing else '播放')
        b.setIcon(self.window.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))
        b.blockSignals(False)

    def set_state(self, state, phase):
        label = self.widgets['behaviorState']
        label.setText('摄食行为' if state == 1 else '正常游动' if state == 0 else '等待数据')
        label.setProperty('state', 'feeding' if state == 1 else 'normal' if state == 0 else 'waiting')
        label.style().unpolish(label)
        label.style().polish(label)
        self.widgets['phaseLabel'].setText(phase)

    def on_frame(self, payload):
        try:
            if payload['generation'] != self.generation:
                return
            self.current_index = payload['index']
            self.gray = payload['gray']
            meta, result = payload['meta'], payload['result']
            slider = self.widgets['frameSlider']
            if not slider.isSliderDown():
                slider.setValue(self.current_index)
            self.widgets['frameLabel'].setText(f"帧 {self.current_index} / {self.source_count - 1}")
            self.widgets['bufferLabel'].setText(f"连续帧 {result['buffer_size']} / 32")
            phase = {
                'warming_up':'正在收集连续帧，尚未形成有效判断',
                'ready':f"最近判断：帧 {result.get('decision_frame')} · 显示经过防抖",
                'disabled':'识别已关闭', 'model_unavailable':'请先加载已导出的模型'
            }.get(result['reason'], result['reason'])
            self.set_state(result['state'], phase)
            probability = result['probability']
            self.widgets['probabilityLabel'].setText('摄食得分：—' if probability is None else f'摄食得分：{probability:.4f}')
            self.widgets['probabilityBar'].setValue(round((probability or 0) * 100))
            self.widgets['probabilityBar'].setFormat('—' if probability is None else f'{probability:.1%}')
            if result.get('changed'):
                table = self.widgets['eventTable']
                row = table.rowCount()
                table.insertRow(row)
                for column, text in enumerate([str(self.current_index), '进入摄食' if result['state'] else '恢复正常显示', f'{probability:.4f}']):
                    table.setItem(row, column, QTableWidgetItem(text))
                table.scrollToBottom()
            self.draw_frame()
            detail = f"帧处理 {payload['processing_ms']:.1f} ms · {meta['shape'][1]} × {meta['shape'][0]}"
            if 'segment_order' in meta:
                detail += f" · 分段 {meta['segment_order']} / 局部帧 {meta['segment_local_frame']}"
            if 'work_cycle_raw' in meta:
                detail += f" · 设备周期字段 {meta['work_cycle_raw']:.4f}"
            self.window.statusBar().showMessage(detail)
        finally:
            self.worker.acknowledge()

    def draw_frame(self, *_):
        if self.gray is None:
            return
        import cv2
        gray = np.flipud(self.gray) if self.widgets['flipCheck'].isChecked() else self.gray
        rgb = cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2RGB)
        if self.widgets['countCheck'].isChecked():
            mask = (gray > self.widgets['thresholdSpin'].value()).astype(np.uint8)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            accepted = 0
            for x,y,w,h,area in stats[1:]:
                if area >= self.widgets['areaSpin'].value():
                    accepted += 1
                    cv2.rectangle(rgb, (x,y), (x+w,y+h), (30,210,160), 1)
            self.widgets['countLabel'].setText(f'检测目标数：{accepted}（不等于已确认鱼只数）')
        else:
            self.widgets['countLabel'].setText('检测目标数：—')
        h,w = rgb.shape[:2]
        image = QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
        label = self.widgets['imageLabel']
        label.setPixmap(QPixmap.fromImage(image).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def toggle_inference(self, enabled):
        self.worker.command('enabled', enabled)
        self.set_state(None, '重新收集连续帧' if enabled else '识别已关闭')

    def on_error(self, text):
        self.log(text)
        self.set_state(None, '数据处理失败，已暂停；查看运行记录')
        self.window.statusBar().showMessage(text)

    def log(self, text):
        self.widgets['logText'].appendPlainText(datetime.now().strftime('%H:%M:%S') + '  ' + text)

    def lock_layout(self, locked):
        for dock in self.docks:
            dock.setFeatures(QDockWidget.DockWidgetClosable if locked else
                             QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

    def reset_layout(self):
        self.actions['actionLockLayout'].setChecked(False)
        self.window.restoreState(self.default_layout,1)
        for dock in self.docks:
            dock.show()

    def open_designer(self):
        executable = Path(sys.executable).parent / 'Scripts' / 'pyside6-designer.exe'
        if not executable.is_file():
            self.on_error(f'未找到 Qt Designer：{executable}')
            return
        subprocess.Popen([str(executable), str(self.ui_path)], cwd=str(self.ui_path.parent))
        self.log('在 Designer 保存后重启工作台生效；保留已绑定控件的 objectName')

    def open_report(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(RESULTS / 'reports' / 'train_report.md')))

    def open_legacy(self):
        # Preserve the original tool, while redirecting its generated caches/configuration.
        import hashlib
        import sonar_params as legacy
        legacy.CONFIG_PATH = str(output_path('db_replay/settings/legacy_params.json'))
        def cache_path(path):
            p=Path(path).resolve()
            stamp=f'{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}'
            name=hashlib.sha256(stamp.encode('utf-8')).hexdigest()
            return str(output_path(Path('db_replay')/'legacy_indexes'/f'{name}.frameidx2'))
        legacy._cache_path=cache_path
        tool=legacy.ParamTuner()
        tool.btn_behavior.setEnabled(False)
        tool.btn_behavior.setText('旧规则行为识别（此入口停用）')
        tool.setWindowTitle('原有数量 / 体长工具 · 摄食识别请使用主工作台')
        self.legacy_windows.append(tool)
        tool.show()

    def eventFilter(self, obj, event):
        if obj is self.widgets.get('imageLabel') and event.type() == QEvent.Resize:
            self.draw_frame()
        if obj is self.window and event.type() == QEvent.Close and not self.closed:
            self.worker.shutdown()
            if not self.worker.wait(3000):
                event.ignore()
                self.log('正在等待读取完成并释放资源，再次关闭即可退出')
                return True
            if not self.smoke:
                self.settings.setValue('geometry',self.window.saveGeometry())
                self.settings.setValue('state',self.window.saveState(1))
                self.settings.sync()
            for tool in self.legacy_windows:
                tool.close()
            self.closed=True
        return super().eventFilter(obj,event)

