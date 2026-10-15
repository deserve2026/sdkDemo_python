"""Read-only PNG/DB sources. DB caches are confined to the results directory."""
import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from .paths import output_path


class PngSource:
    def __init__(self, directory, log=lambda _: None):
        self.path = Path(directory).resolve()
        numbered = []
        for p in self.path.iterdir():
            match = re.fullmatch(r'frame_(\d+)\.png', p.name, re.I)
            if match:
                numbered.append((int(match[1]), p))
        numbered.sort()
        if not numbered:
            raise ValueError('Folder has no frame_XXXXXX.png sequence')
        if len({n for n, _ in numbered}) != len(numbered):
            raise ValueError('Duplicate local frame numbers')
        self.frames = numbered
        self.count = len(numbered)
        self.kind = 'PNG'
        log(f'读取 PNG 序列：{self.count} 帧')

    def read(self, index):
        number, path = self.frames[index]
        with Image.open(path) as image:
            gray = np.asarray(image.convert('L'), dtype=np.uint8).copy()
        return gray, dict(local_frame=number, source_path=str(path), source_kind='PNG',
                          shape=list(gray.shape), frame_serial=number)


class DbSource:
    CACHE_VERSION = 1

    def __init__(self, path, log=lambda _: None):
        import pyLhForwardSDK_module as sdk
        self.sdk = sdk
        self.path = Path(path).resolve()
        if not self.path.is_file() or self.path.suffix.lower() != '.db':
            raise ValueError('Choose an existing .DB file')
        self.kind = 'DB'
        self.packet_size = sdk.LhForwardFrame.PACK_SIZE
        stat = self.path.stat()
        signature = dict(path=str(self.path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         packet_size=self.packet_size, cache_version=self.CACHE_VERSION)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode('utf-8')).hexdigest()
        cache = output_path(Path('db_replay') / 'indexes' / f'{key}.json')
        self.entries = None
        if cache.exists():
            content = json.loads(cache.read_text(encoding='utf-8'))
            if content['signature'] == signature:
                self.entries = content['entries']
        if self.entries is None:
            self.entries = []
            frame = sdk.LhForwardFrame()
            info = sdk.packageInfo()
            buffer = np.zeros(10 * 1024 * 1024, dtype=np.uint8)
            with self.path.open('rb') as handle:
                packet = 0
                while True:
                    raw = handle.read(self.packet_size)
                    if len(raw) < self.packet_size:
                        if raw:
                            log(f'文件尾有 {len(raw)} 字节不完整数据包，不补造帧')
                        break
                    if frame.writeOnePackData(np.frombuffer(raw, dtype=np.uint8)) == sdk.LfFrameRet.oneFrameOK:
                        frame.getOneFrame(info, buffer)
                        self.entries.append(dict(packet_end=packet, serial=int(info.frameSerialNum), pack_type=int(info.packType)))
                        if len(self.entries) % 250 == 0:
                            log(f'建立 DB 索引：已识别 {len(self.entries)} 帧')
                    packet += 1
            if not self.entries:
                raise ValueError('SDK did not decode any complete DB frames')
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(signature=signature, entries=self.entries)), encoding='utf-8')
            temporary.replace(cache)
        self.count = len(self.entries)
        self.image_generator = sdk.LhForwardImage(968, 512)
        self._stream_frame = None
        self._stream_handle = None
        self._stream_index = -1
        log(f'DB 索引完成：{self.count} 帧；缓存位于结果目录')

    def close(self):
        if self._stream_handle is not None:
            self._stream_handle.close()
            self._stream_handle = None

    def read(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        # Sequential replay reuses SDK state; seeks warm up from earlier frames.
        if self._stream_handle is None or index != self._stream_index + 1:
            self.close()
            self._stream_handle = self.path.open('rb')
            start_packet = self.entries[index - 3]['packet_end'] + 1 if index >= 3 else 0
            self._stream_handle.seek(start_packet * self.packet_size)
            self._stream_frame = self.sdk.LhForwardFrame()
        entry = self.entries[index]
        info = self.sdk.packageInfo()
        buffer = np.zeros(10 * 1024 * 1024, dtype=np.uint8)
        while self._stream_handle.tell() // self.packet_size <= entry['packet_end']:
            raw = self._stream_handle.read(self.packet_size)
            if len(raw) < self.packet_size:
                break
            end_packet = self._stream_handle.tell() // self.packet_size - 1
            if self._stream_frame.writeOnePackData(np.frombuffer(raw, dtype=np.uint8)) != self.sdk.LfFrameRet.oneFrameOK:
                continue
            self._stream_frame.getOneFrame(info, buffer)
            if end_packet != entry['packet_end']:
                continue
            if int(info.frameSerialNum) != entry['serial']:
                raise ValueError('SDK frame serial disagrees with the complete sequential index')
            # Verified pixel-for-pixel against all 3666 corresponding recording-5 PNGs.
            gray = np.ascontiguousarray(np.flipud(self._gray(info, buffer)))
            self._stream_index = index
            return gray, dict(local_frame=index, source_path=str(self.path), source_kind='DB',
                              frame_serial=int(info.frameSerialNum), pack_type=int(info.packType),
                              max_distance=float(info.maxDist), work_cycle_raw=float(info.workCycle),
                              shape=list(gray.shape), input_orientation='sdk_vertical_flip')
        self.close()
        self._stream_index = -1
        raise ValueError(f'DB frame {index} did not reconstruct at its indexed packet boundary')

    def _gray(self, info, buffer):
        pack_type = int(info.packType)
        if pack_type in (3, 4):
            length = int(info.dataHeight)
            if not 0 < length <= len(buffer):
                raise ValueError('Invalid compressed image length')
            with Image.open(io.BytesIO(buffer[:length].tobytes())) as image:
                return np.asarray(image.convert('L'), dtype=np.uint8).copy()
        if pack_type == 0:
            width, height = (968, 968) if int(info.horAngleReso) == 360 else (968, 512)
            pixels = np.zeros(width * height, dtype=np.uint8)
            resolution = np.array([0.0], dtype=np.float64)
            self.image_generator.generateForwardImage(info, buffer, pixels, resolution)
            return pixels.reshape(height, width).copy()
        if pack_type == 2:
            n = int(info.numPerRow) * int(info.dataHeight)
            if n < 1 or n > 20_000_000:
                raise ValueError('Invalid uncompressed image dimensions')
            intensity = np.zeros(n, dtype=np.float64)
            self.image_generator.preprocessData(info, buffer, intensity)
            coefficient = 1 if int(info.dataWidth) == 0 else 256
            return np.clip(intensity / coefficient, 0, 255).astype(np.uint8).reshape(int(info.dataHeight), int(info.numPerRow))
        raise ValueError(f'Unsupported DB packType={pack_type}')


class DbSeries:
    def __init__(self, seed, log=lambda _: None):
        from bisect import bisect_right
        seed = Path(seed).resolve()
        match = re.fullmatch(r'(.+?)(?:_(\d+))?', seed.stem)
        root = match[1]
        files = []
        for p in seed.parent.iterdir():
            if p.suffix.lower() != '.db':
                continue
            candidate = re.fullmatch(re.escape(root) + r'(?:_(\d+))?', p.stem)
            if candidate:
                files.append((int(candidate[1]) if candidate[1] else 0, p))
        files.sort()
        if [n for n, _ in files] != list(range(len(files))):
            raise ValueError('同名 DB 分段不连续，不能拼接；可取消连续回放后单独打开')
        self.sources, self.offsets = [], []
        self.count, self.kind, self.path = 0, 'DB', files[0][1]
        for _, p in files:
            source = DbSource(p, log)
            if self.sources and source.entries[0]['serial'] != (self.sources[-1].entries[-1]['serial'] + 1) % (2 ** 32):
                raise ValueError('DB 分段的设备帧号不接续，禁止自动拼接')
            self.offsets.append(self.count)
            self.count += source.count
            self.sources.append(source)
        self._active = None
        log(f'整次录制：{len(self.sources)} 个连续分段，共 {self.count} 帧')

    def read(self, index):
        from bisect import bisect_right
        segment = bisect_right(self.offsets, index) - 1
        if segment < 0 or index >= self.count:
            raise IndexError(index)
        if self._active is not None and self._active != segment:
            self.sources[self._active].close()
        self._active = segment
        local = index - self.offsets[segment]
        gray, meta = self.sources[segment].read(local)
        meta.update(local_frame=index, segment_local_frame=local, segment_order=segment)
        return gray, meta

    def close(self):
        for source in self.sources:
            source.close()


def open_source(path, log=lambda _: None, series=False):
    path = Path(path)
    if path.is_dir():
        return PngSource(path, log)
    return DbSeries(path, log) if series else DbSource(path, log)
