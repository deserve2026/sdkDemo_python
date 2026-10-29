import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.utils.paths import output_path


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    config['_config_path'] = str(path)
    config['_config_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ('feeding_data_dir', 'normal_data_dir'):
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(config[key])
    for key, relative in [('metadata_dir', 'metadata'), ('quality_report_dir', 'quality_reports'), ('windows_dir', 'windows')]:
        expected = Path(config['results_dir']) / relative
        if Path(config[key]).resolve() != expected.resolve():
            raise ValueError(f'{key} must equal {expected}')
    if config.get('fps') is not None and not config.get('fps_verified', False):
        raise ValueError('Unverified FPS must remain null')
    if config['normal_policy'] != 'normal_recordings_only':
        raise ValueError('Unlabeled feeding-source intervals must not become normal')
    if config.get('boundary_ignore_frames') != 0 or config.get('boundary_ignore_seconds') is not None:
        raise ValueError('Version 1 uses overlap-only labels, with no additional boundary buffer')
    w, s, t = config['window_size'], config['stride'], config['feeding_overlap_threshold']
    if not isinstance(w, int) or not isinstance(s, int) or w <= 0 or s <= 0 or not 0 < t <= 1:
        raise ValueError('Invalid window parameters')
    return config


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def read_annotation(path):
    data = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            text = data.decode(encoding, errors='strict')
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames is None:
                raise ValueError('Annotation CSV has no header')
            header = [str(x).strip() for x in reader.fieldnames]
            if len(set(header)) != len(header):
                raise ValueError('Duplicate annotation columns')
            rows = []
            for row in reader:
                if None in row:
                    raise ValueError(f'Too many fields at CSV line {reader.line_num}')
                rows.append((reader.line_num, {str(k).strip(): (v or '').strip() for k, v in row.items()}))
            return header, rows, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode annotation CSV strictly: {path}')


def write_csv(config, relative, rows, fields):
    path = output_path(config, relative)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def write_text(config, relative, content):
    path = output_path(config, relative)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)
    return path


def write_json(config, relative, value):
    return write_text(config, relative, json.dumps(value, ensure_ascii=False, indent=2))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mark_stage(config, name, passed, details, artifacts=(), dependencies=()):
    path = output_path(config, 'metadata/pipeline_status.json')
    status = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'stages': {}}
    if status.get('config_sha256') != config['_config_sha256']:
        status = {'stages': {}}
    order = ['audit', 'timeline', 'events', 'split', 'windows']
    if name in order:
        for downstream in order[order.index(name) + 1:]:
            status['stages'].pop(downstream, None)
    files = [*artifacts, *dependencies]
    status['stages'][name] = {
        'passed': bool(passed), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'details': details, 'file_sha256': {str(Path(x).resolve()): digest(x) for x in files},
    }
    status['config_sha256'] = config['_config_sha256']
    status['data_ready_for_training'] = all(status['stages'].get(n, {}).get('passed', False) for n in order)
    status['training_started'] = False
    write_json(config, 'metadata/pipeline_status.json', status)
    return status


def require_stage(config, name):
    path = output_path(config, 'metadata/pipeline_status.json')
    if not path.exists():
        raise RuntimeError(f'Run {name} first')
    status = json.loads(path.read_text(encoding='utf-8'))
    stage = status.get('stages', {}).get(name)
    if status.get('config_sha256') != config['_config_sha256'] or not stage or not stage['passed']:
        raise RuntimeError(f'Stage {name} is missing, stale, or failed. Read quality reports.')
    for raw, expected in stage['file_sha256'].items():
        path = Path(raw)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'Stage {name} input/output changed: {path}. Re-run upstream stages.')
    return status
