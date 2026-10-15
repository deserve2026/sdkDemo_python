"""Sequential replay with GUI backpressure; no inference frames are dropped."""
import csv
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from .paths import RESULTS, output_path
from .sources import open_source
from .inference import FeedingEngine

class ReplayWorker(QThread):
    frame_ready = Signal(object)
    source_ready = Signal(object)
    model_ready = Signal(object)
    message = Signal(str)
    failure = Signal(str)
    playing_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.commands = queue.Queue()
        self.acknowledged = threading.Event()
        self.quit_requested = threading.Event()

    def command(self, kind, value=None):
        self.commands.put((kind, value))

    def acknowledge(self):
        self.acknowledged.set()

    def shutdown(self):
        self.quit_requested.set()
        self.acknowledged.set()
        self.command('quit')

    def run(self):
        source = engine = writer = log_handle = None
        enabled, playing, fps = True, False, 10
        current, requested, generation = -1, None, 0
        fields = ['source', 'source_kind', 'local_frame', 'frame_serial', 'decision', 'decision_frame',
                  'feeding_probability', 'ui_state', 'buffer_size', 'processing_ms', 'status']
        def reset():
            if engine:
                engine.reset()
        def set_playing(value):
            nonlocal playing
            playing = value
            self.playing_changed.emit(value)
        def load_model():
            nonlocal engine
            try:
                engine = FeedingEngine(RESULTS / 'onnx')
                self.model_ready.emit(dict(loaded=True, parameters=engine.config['parameters']))
            except Exception as error:
                engine = None
                self.model_ready.emit(dict(loaded=False, error=str(error)))
        load_model()
        try:
            while not self.quit_requested.is_set():
                try:
                    kind, value = self.commands.get(timeout=.05 if not playing and requested is None else .001)
                    if kind == 'quit':
                        break
                    if kind == 'load':
                        set_playing(False)
                        generation += 1
                        reset()
                        if source and hasattr(source, 'close'):
                            source.close()
                        source = None
                        if log_handle:
                            log_handle.close()
                            log_handle = None
                        try:
                            options = value if isinstance(value, dict) else dict(path=value, series=False)
                            source = open_source(options['path'], self.message.emit, series=options.get('series', False))
                            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                            log_path = output_path(Path('db_replay') / 'sessions' / stamp / 'predictions.csv')
                            log_handle = log_path.open('w', encoding='utf-8-sig', newline='')
                            writer = csv.DictWriter(log_handle, fieldnames=fields)
                            writer.writeheader()
                            (log_path.parent / 'session.json').write_text(json.dumps(dict(
                                source=str(source.path), source_kind=source.kind, frame_count=source.count,
                                model_config=engine.config if engine else None, fps_is_playback_speed=True
                            ), ensure_ascii=False, indent=2), encoding='utf-8')
                            current, requested = -1, 0
                            self.source_ready.emit(dict(path=str(source.path), count=source.count, kind=source.kind,
                                                        log_path=str(log_path), generation=generation))
                        except Exception as error:
                            requested = None
                            self.failure.emit(f'打开数据失败：{error}')
                    elif kind == 'play' and source:
                        set_playing(bool(value))
                    elif kind == 'seek' and source:
                        set_playing(False)
                        reset()
                        requested = max(0, min(int(value), source.count - 1))
                    elif kind == 'step' and source:
                        set_playing(False)
                        requested = max(0, min(current + int(value), source.count - 1))
                    elif kind == 'speed':
                        fps = max(1, min(40, int(value)))
                    elif kind == 'enabled':
                        enabled = bool(value)
                        reset()
                    elif kind == 'reload_model':
                        set_playing(False)
                        load_model()
                except queue.Empty:
                    pass
                if source is None or (not playing and requested is None):
                    continue
                index = requested if requested is not None else current + 1
                requested = None
                if index >= source.count:
                    set_playing(False)
                    self.message.emit('回放结束；不自动循环，不跨录制继承模型历史')
                    continue
                started = time.perf_counter()
                try:
                    gray, meta = source.read(index)
                    result = engine.consume(gray, str(source.path), int(meta['local_frame']),
                        source_frame_number=meta['frame_serial'] if source.kind=='DB' else None) if enabled and engine else dict(
                        state=None, probability=None, buffer_size=0, decision=False, changed=False,
                        reason='disabled' if not enabled else 'model_unavailable')
                    current = index
                    milliseconds = (time.perf_counter() - started) * 1000
                    writer.writerow(dict(source=str(source.path), source_kind=source.kind, local_frame=meta['local_frame'],
                        frame_serial=meta['frame_serial'], decision=int(result['decision']),
                        decision_frame=result.get('decision_frame'), feeding_probability=result['probability'],
                        ui_state=result['state'], buffer_size=result['buffer_size'], processing_ms=milliseconds,
                        status=result['reason']))
                    if result['decision'] or index % 100 == 0:
                        log_handle.flush()
                    self.acknowledged.clear()
                    self.frame_ready.emit(dict(index=index, gray=gray, meta=meta, result=result,
                                               processing_ms=milliseconds, generation=generation))
                    while not self.acknowledged.wait(.05) and not self.quit_requested.is_set():
                        pass
                    if playing:
                        self.quit_requested.wait(max(0, 1 / fps - (time.perf_counter() - started)))
                except Exception as error:
                    current = index
                    reset()
                    set_playing(False)
                    self.failure.emit(f'帧 {index} 处理失败，已清空历史并暂停：{error}')
        finally:
            if log_handle:
                log_handle.close()
            if source and hasattr(source, 'close'):
                source.close()

