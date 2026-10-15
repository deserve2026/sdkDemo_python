"""Exact fixed preprocessing plus a causal, resettable 32-frame feature buffer."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .paths import sha256


def preprocess(gray):
    array = np.asarray(gray)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError('Expected decoded uint8 grayscale image, without display overlays')
    small = Image.fromarray(array).convert('L').resize((224, 224), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32) / 255.0
    rgb = np.repeat(pixels[None, :, :], 3, axis=0)
    mean = np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]
    return ((rgb - mean) / std)[None]


class Debouncer:
    def __init__(self, entry=.8, exit_threshold=.6, entry_count=2, exit_count=2):
        if not 0 <= exit_threshold < entry <= 1 or min(entry_count, exit_count) < 1:
            raise ValueError('Invalid fixed state thresholds')
        self.entry, self.exit = entry, exit_threshold
        self.entry_count, self.exit_count = entry_count, exit_count
        self.reset()

    def reset(self):
        self.active = False
        self.high = self.low = 0

    def update(self, probability):
        before = self.active
        if not np.isfinite(probability):
            self.reset()
            raise ValueError('Nonfinite model output')
        if not self.active:
            self.high = self.high + 1 if probability >= self.entry else 0
            if self.high >= self.entry_count:
                self.active, self.high, self.low = True, 0, 0
        else:
            self.low = self.low + 1 if probability <= self.exit else 0
            if self.low >= self.exit_count:
                self.active, self.high, self.low = False, 0, 0
        return int(self.active), before != self.active


class FeedingEngine:
    def __init__(self, model_dir, encoder=None, temporal=None):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / 'deployment.json').read_text(encoding='utf-8'))
        if not self.config.get('verified'):
            raise ValueError('ONNX numerical verification has not passed')
        for name, expected in self.config['model_sha256'].items():
            if sha256(self.model_dir / name) != expected:
                raise ValueError(f'ONNX file checksum mismatch: {name}')
        if self.config['window_size'] != 32 or self.config['stride'] != 8:
            raise ValueError('Unsupported deployment window contract')
        if encoder is None or temporal is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            encoder = ort.InferenceSession(str(self.model_dir / 'cnn_encoder.onnx'), options, providers=['CPUExecutionProvider'])
            temporal = ort.InferenceSession(str(self.model_dir / 'tcn_classifier.onnx'), options, providers=['CPUExecutionProvider'])
        self.encoder, self.temporal = encoder, temporal
        self.debounce = Debouncer(**self.config['parameters'])
        self.features = deque(maxlen=32)
        self.reset()

    def reset(self):
        self.features.clear()
        self.debounce.reset()
        self.last_key = None
        self.last_index = None
        self.last_source_frame = None
        self.since_reset = 0
        self.last_probability = None
        self.last_decision = None

    def consume(self, gray, recording_key, frame_index, source_frame_number=None):
        if self.last_key != recording_key or self.last_index is None or frame_index != self.last_index + 1:
            self.reset()
        # Complete DB frames may have consecutive array indices despite missing device frames.
        if (source_frame_number is not None and self.last_source_frame is not None
                and source_frame_number != (self.last_source_frame + 1) % (2 ** 32)):
            self.reset()
        if gray is None:
            self.reset()
            return dict(state=None, probability=None, buffer_size=0, decision=False, changed=False, reason='invalid_frame')
        self.last_key, self.last_index = recording_key, frame_index
        self.last_source_frame = source_frame_number
        feature = self.encoder.run(None, {'images': preprocess(gray)})[0]
        if feature.shape != (1, 512) or not np.isfinite(feature).all():
            self.reset()
            raise ValueError('Invalid CNN feature output')
        self.features.append(feature[0].copy())
        self.since_reset += 1
        decision = self.since_reset >= 32 and (self.since_reset - 32) % 8 == 0
        changed = False
        if decision:
            sequence = np.stack(self.features)[None].astype(np.float32)
            probabilities = self.temporal.run(None, {'features': sequence})[0]
            if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
                self.reset()
                raise ValueError('Invalid TCN probability output')
            self.last_probability = float(probabilities[0, 1])
            _, changed = self.debounce.update(self.last_probability)
            self.last_decision = frame_index
        return dict(state=int(self.debounce.active) if self.last_probability is not None else None,
                    probability=self.last_probability, buffer_size=len(self.features),
                    decision=decision, decision_frame=self.last_decision, changed=changed,
                    reason='ready' if self.last_probability is not None else 'warming_up')
