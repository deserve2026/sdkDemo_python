"""Frozen, deterministic per-frame features; raw sources remain read-only."""
import io
import json
import hashlib
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torchvision
from src.data.timeline import load_timeline
from src.data.windowing import verify_raw_inventory
from src.models.cnn_backbone import make_feature_extractor
from src.utils.image import preprocess_image
from src.utils.io import digest, read_csv, write_json
from src.utils.paths import output_path
from src.utils.experiment import fingerprint, preprocessing_contract, verify_data


def cache_identity(config, model):
    from PIL import __version__ as pillow_version
    source_root = Path(__file__).resolve().parents[2]
    contract = dict(preprocess=preprocessing_contract(model), torch=torch.__version__,
                    torchvision=torchvision.__version__, pillow=pillow_version,
                    dataset_sha256=config['_config_sha256'],
                    timeline_sha256=digest(Path(config['metadata_dir']) / 'timeline_index.csv'),
                    dtype='float32', extractor_mode='eval', precision='float32_tf32_disabled',
                    implementation={name: digest(source_root / name) for name in
                                    ('src/utils/image.py', 'src/data/features.py', 'src/models/cnn_backbone.py')})
    return fingerprint(contract), contract


def cache_root(config, model):
    identity, contract = cache_identity(config, model)
    root = output_path(config, Path('features') / identity[:16] / 'manifest.json').parent
    return root, identity, contract


def verified_image(frame):
    data = Path(frame['image_path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != frame['sha256']:
        raise RuntimeError(f'Raw image changed since audit: {frame["image_path"]}')
    return preprocess_image(io.BytesIO(data))


def extract(config, model_config, settings, splits=('train', 'val')):
    if not set(splits) <= {'train', 'val'} or not splits:
        raise ValueError('This baseline command only extracts train/val; test stays sealed')
    verify_data(config)
    timeline = load_timeline(config)
    verify_raw_inventory(config, timeline)
    assignments = {r['recording_key']: r['split'] for r in read_csv(Path(config['metadata_dir']) / 'splits.csv')}
    grouped = defaultdict(list)
    for row in timeline:
        if assignments[row['recording_key']] in splits:
            grouped[row['recording_key']].append(row)
    device = torch.device(settings['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose CPU explicitly if needed')
    torch.set_num_threads(settings.get('cpu_threads', 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root, identity, contract = cache_root(config, model_config)
    model = make_feature_extractor(config, pretrained=True).to(device).eval()
    weights = Path(config['results_dir']) / 'checkpoints' / 'pretrained' / 'checkpoints' / 'resnet18-f37072fd.pth'
    weights_sha = digest(weights)
    if not weights_sha.startswith('f37072fd'):
        raise RuntimeError('Unexpected ResNet18 V1 checkpoint hash')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if manifest and (manifest['identity'] != identity or manifest['weights_sha256'] != weights_sha):
        raise RuntimeError('Feature cache provenance mismatch')
    manifest.update(identity=identity, contract=contract, weights_sha256=weights_sha,
                    records=manifest.get('records', {}), test_extracted=False)
    total = sum(len(rows) for rows in grouped.values())
    completed = 0
    started = time.perf_counter()
    batch_size = int(settings['feature_extraction_batch_size'])
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    with ThreadPoolExecutor(max_workers=settings.get('image_workers', 4)) as pool, torch.inference_mode():
        for key, frames in sorted(grouped.items()):
            frames.sort(key=lambda f: f['global_frame'])
            if [f['global_frame'] for f in frames] != list(range(len(frames))):
                raise ValueError(f'Nonordinal timeline: {key}')
            saved = manifest['records'].get(key)
            if saved and (root / saved['file']).is_file() and digest(root / saved['file']) == saved['sha256']:
                completed += len(frames)
                logging.info('Reusing verified cache %s (%d frames)', key, len(frames))
                continue
            target = root / f'{frames[0]["behavior"]}_{frames[0]["recording_id"]:03d}.npy'
            partial = target.with_suffix('.partial.npy')
            array = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(frames), 512))
            array[:] = np.nan
            usable = [f for f in frames if f['is_valid']]
            last_log = 0.0
            for offset in range(0, len(usable), batch_size):
                batch = usable[offset:offset + batch_size]
                images = np.stack(list(pool.map(verified_image, batch)))
                features = model(torch.from_numpy(images).to(device)).cpu().numpy()
                if features.shape != (len(batch), 512) or not np.isfinite(features).all():
                    raise RuntimeError('Invalid CNN features')
                array[[f['global_frame'] for f in batch]] = features
                now = time.perf_counter()
                if now - last_log >= 15 or offset + len(batch) == len(usable):
                    done = completed + offset + len(batch)
                    logging.info('Features %s %d/%d; total %d/%d; elapsed %.1f min',
                                 key, offset + len(batch), len(usable), done, total, (now - started) / 60)
                    last_log = now
            array.flush()
            del array
            partial.replace(target)
            manifest['records'][key] = dict(file=target.name, sha256=digest(target),
                                           split=assignments[key], frame_count=len(frames),
                                           valid_frames=len(usable), shape=[len(frames), 512])
            write_json(config, manifest_path.relative_to(Path(config['results_dir'])), manifest)
            completed += len(frames)
    logging.info('Feature extraction complete: %d requested records; no test features', len(grouped))
    return manifest_path


def load_caches(config, model_config, required_splits=('train', 'val')):
    root, identity, contract = cache_root(config, model_config)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['identity'] != identity or manifest['contract'] != contract:
        raise RuntimeError('Feature cache contract changed; extract again')
    arrays = {}
    for assignment in read_csv(Path(config['metadata_dir']) / 'splits.csv'):
        if assignment['split'] not in required_splits:
            continue
        key = assignment['recording_key']
        record = manifest['records'].get(key)
        if not record or record['split'] != assignment['split']:
            raise RuntimeError(f'Missing or mismatched cache: {key}')
        path = root / record['file']
        if path.parent != root or digest(path) != record['sha256']:
            raise RuntimeError(f'Cache checksum mismatch: {key}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(array.shape) != record['shape'] or array.dtype != np.float32:
            raise RuntimeError(f'Invalid feature shape/dtype: {key}')
        arrays[key] = array
    return arrays, manifest_path
