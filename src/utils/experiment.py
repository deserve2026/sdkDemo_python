"""Shared guards and provenance for the explicitly requested baseline experiment."""
import hashlib
import json
from pathlib import Path
import yaml
from src.utils.io import digest, require_stage


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def read_settings(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))


def verify_data(config):
    for stage in ('audit', 'timeline', 'events', 'split', 'windows'):
        require_stage(config, stage)


def source_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for directory in ('src', 'scripts', 'configs')
            for p in sorted((root / directory).rglob('*')) if p.suffix in ('.py', '.yaml')}


def preprocessing_contract(model):
    expected = dict(backbone='resnet18', pretrained_weights='IMAGENET1K_V1',
                    freeze_backbone=True, feature_dim=512, input_size=[224, 224],
                    color_mode='grayscale_repeated_to_rgb', resize_backend='PIL',
                    resize_interpolation='bilinear', pixel_scale=255.0,
                    normalization_mean=[0.485, 0.456, 0.406],
                    normalization_std=[0.229, 0.224, 0.225])
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f'Unsupported preprocessing contract: {key}={model.get(key)}')
    return expected
