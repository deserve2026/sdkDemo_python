import hashlib
import io
import json
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.utils.io import mark_stage, read_annotation, write_csv, write_json, write_text


FOLDER_RE = re.compile(r'^gts_(feed|normal)_(\d+)(?:_(\d+))?$')
FRAME_RE = re.compile(r'^(?:frame_)?(\d+)(?:\.png)?$', re.IGNORECASE)
CORE_EVENT_FIELDS = ['recording_id', 'event_id', 'start_folder', 'start_frame', 'end_folder', 'end_frame']
RECORDING_FIELDS = ['recording_id', 'behavior', 'recording_key', 'folder_name', 'folder_path',
                    'segment_order', 'num_frames', 'first_frame', 'last_frame']
QUALITY_FIELDS = ['behavior', 'recording_id', 'folder_name', 'folder_path', 'segment_order', 'num_frames',
                  'first_frame', 'last_frame', 'missing_count', 'duplicate_count', 'corrupt_count',
                  'image_width', 'image_height', 'size_consistent', 'status', 'notes']
FRAME_FIELDS = ['behavior', 'recording_id', 'recording_key', 'segment_order', 'folder_name', 'folder_path',
                'local_frame', 'image_path', 'file_size', 'mtime_ns', 'sha256', 'image_width',
                'image_height', 'is_valid', 'issues']


def parse_folder(name):
    match = FOLDER_RE.fullmatch(name)
    if not match:
        raise ValueError(f'Unrecognized folder name: {name}')
    return ('feeding' if match[1] == 'feed' else 'normal', int(match[2]), int(match[3] or 0))


def parse_frame(value):
    match = FRAME_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f'Unrecognized frame number: {value!r}')
    return int(match[1])


def event_is_placeholder(row):
    return not any(str(row.get(k) or '').strip() for k in CORE_EVENT_FIELDS)


def inspect_png(path):
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    result = {'image_path': str(path.resolve()), 'file_size': before.st_size,
              'mtime_ns': before.st_mtime_ns, 'sha256': hashlib.sha256(data).hexdigest(),
              'image_width': '', 'image_height': '', 'is_valid': 1, 'issues': ''}
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG':
                raise ValueError(f'Extension PNG but content is {image.format}')
            result['image_width'], result['image_height'] = image.size
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except Exception as error:
        result['is_valid'] = 0
        result['issues'] = f'{type(error).__name__}: {error}'
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source file changed during audit: {path}')
    return result


def discover_event_csv(config):
    explicit = config.get('feeding_events_csv')
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = Path(config['feeding_data_dir'])
    candidates = [p for name in ('feeding_events.csv', 'feeding_events.bak.csv') for p in root.rglob(name)]
    if len(candidates) != 1:
        raise ValueError(f'Choose a unique feeding_events_csv; candidates={candidates}')
    return candidates[0]


def run(config):
    frames, recordings, quality, fatal = [], [], [], []
    used_names, used_segments = set(), set()
    discovered = defaultdict(list)
    with ThreadPoolExecutor(max_workers=config.get('audit_workers', 4)) as pool:
        for source_behavior, key in [('feeding', 'feeding_data_dir'), ('normal', 'normal_data_dir')]:
            root = Path(config[key]).resolve()
            all_entries = list(root.rglob('*'))
            png_parents = {p.parent for p in all_entries if p.is_file() and p.suffix.lower() == '.png'}
            named_dirs = {p for p in all_entries if p.is_dir() and FOLDER_RE.fullmatch(p.name)}
            empty_dirs = {p for p in all_entries if p.is_dir() and not any(p.iterdir())}
            folders = sorted(png_parents | named_dirs | empty_dirs, key=lambda p: str(p))
            for folder in folders:
                pngs = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.png'), key=lambda p: p.name)
                issues = []
                behavior, rid, segment = source_behavior, '', ''
                try:
                    behavior, rid, segment = parse_folder(folder.name)
                    if behavior != source_behavior:
                        raise ValueError('Folder behavior does not match source root')
                    identity = (behavior, rid, segment)
                    if folder.name in used_names or identity in used_segments:
                        raise ValueError('Duplicate folder name or recording/segment identity')
                    used_names.add(folder.name)
                    used_segments.add(identity)
                except ValueError as error:
                    issues.append(str(error))
                    fatal.append(f'{folder}: {error}')
                key_value = f'{behavior}:{rid}' if rid != '' else ''
                local_rows = []
                numbers = []
                for path, inspected in zip(pngs, pool.map(inspect_png, pngs)):
                    number = ''
                    try:
                        number = parse_frame(path.name)
                        numbers.append(number)
                    except ValueError as error:
                        inspected['is_valid'] = 0
                        inspected['issues'] = str(error)
                        fatal.append(f'{path}: unparseable frame')
                    local_rows.append({**inspected, 'behavior': behavior, 'recording_id': rid,
                                       'recording_key': key_value, 'segment_order': segment,
                                       'folder_name': folder.name, 'folder_path': str(folder), 'local_frame': number})
                counts = Counter(numbers)
                duplicates = [n for n, count in counts.items() if count > 1]
                if duplicates:
                    fatal.append(f'{folder}: duplicate frame numbers {duplicates}')
                for row in local_rows:
                    if row['local_frame'] in duplicates:
                        row['is_valid'] = 0
                        row['issues'] += '; duplicate_frame_number'
                unique = sorted(counts)
                missing_ranges = [(a + 1, b - 1) for a, b in zip(unique, unique[1:]) if b > a + 1]
                missing_count = sum(b - a + 1 for a, b in missing_ranges)
                corrupt_count = sum(not row['is_valid'] and row['local_frame'] != '' and row['local_frame'] not in duplicates for row in local_rows)
                sizes = Counter((row['image_width'], row['image_height']) for row in local_rows if row['image_width'] != '')
                width, height = sizes.most_common(1)[0][0] if sizes else ('', '')
                unexpected = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() not in ('.png', '.db', '.csv')]
                if not pngs:
                    issues.append('empty_png_sequence')
                    fatal.append(f'{folder}: empty PNG sequence')
                if missing_ranges:
                    issues.append(f'missing_ranges={missing_ranges}')
                if corrupt_count:
                    issues.append(f'corrupt_images={corrupt_count}')
                if len(sizes) > 1:
                    issues.append(f'image_sizes={dict(sizes)}')
                if unexpected:
                    issues.append(f'unexpected_files={unexpected}')
                if any(r['local_frame'] == '' for r in local_rows):
                    issues.append('unparseable_frame_names')
                row = {'behavior': behavior, 'recording_id': rid, 'recording_key': key_value,
                       'folder_name': folder.name, 'folder_path': str(folder), 'segment_order': segment,
                       'num_frames': len(pngs), 'first_frame': min(numbers) if numbers else '',
                       'last_frame': max(numbers) if numbers else '', 'missing_count': missing_count,
                       'duplicate_count': sum(count - 1 for count in counts.values()), 'corrupt_count': corrupt_count,
                       'image_width': width, 'image_height': height, 'size_consistent': int(len(sizes) <= 1),
                       'status': 'warning' if issues else 'ok', 'notes': '; '.join(issues)}
                quality.append(row)
                if rid != '' and segment != '':
                    recordings.append(row)
                frames.extend(local_rows)
                discovered[source_behavior].append(row)
                logging.info('Audited %s: %d PNG, corrupt=%d, missing=%d', folder.name, len(pngs), corrupt_count, missing_count)
    recording_csv = write_csv(config, 'metadata/recordings.csv', recordings, RECORDING_FIELDS)
    quality_csv = write_csv(config, 'quality_reports/quality_report.csv', quality, QUALITY_FIELDS)
    frame_csv = write_csv(config, 'metadata/frame_manifest.csv', frames, FRAME_FIELDS)
    event_count, placeholder_count, event_note = None, None, ''
    try:
        _, rows, encoding = read_annotation(discover_event_csv(config))
        event_count = sum(not event_is_placeholder(row) for _, row in rows)
        placeholder_count = len(rows) - event_count
        event_note = f'解码方式：{encoding}；事件候选 {event_count} 行，占位/空行 {placeholder_count} 行。候选数量不代表有效性已确认。'
    except Exception as error:
        event_note = f'事件CSV读取失败：{error}'
        fatal.append(event_note)
    summary = {'total_png': len(frames), 'event_candidates': event_count, 'placeholder_rows': placeholder_count,
               'missing_count': sum(r['missing_count'] for r in quality),
               'corrupt_count': sum(r['corrupt_count'] for r in quality),
               'duplicate_count': sum(r['duplicate_count'] for r in quality),
               'size_inconsistent_folders': [r['folder_name'] for r in quality if not r['size_consistent']],
               'fatal_errors': fatal}
    lines = ['# 数据审计报告', '', '原始目录只读；PNG逐文件校验格式、CRC并完整解码，同时记录SHA-256。', '']
    for behavior, expected in [('feeding', config['expected_feeding_recordings']), ('normal', config['expected_normal_recordings'])]:
        rows = discovered[behavior]
        ids = sorted({r['recording_id'] for r in rows if r['recording_id'] != ''})
        summary[behavior] = {'folders': len(rows), 'recordings': len(ids), 'recording_ids': ids,
                             'num_frames': sum(r['num_frames'] for r in rows)}
        lines.append(f'- {behavior}: {len(rows)} 个文件夹，{len(ids)} 次录制，ID={ids}；预期 {expected} 次，匹配={len(ids) == expected}。')
    lines += ['', event_note, f'预期摄食事件 {config["expected_feeding_events"]} 个，候选数量匹配={event_count == config["expected_feeding_events"]}。', '',
              f'段内缺帧总数：{summary["missing_count"]}；损坏图：{summary["corrupt_count"]}；重复帧号：{summary["duplicate_count"]}。',
              '', '## 异常文件夹']
    lines += [f'- {r["folder_name"]}: {r["notes"]}' for r in quality if r['notes']] or ['- 未发现上述异常。']
    all_sizes = sorted({(r['image_width'], r['image_height']) for r in frames if r['image_width'] != ''})
    lines += ['', f'全数据图片尺寸集合：{all_sizes}。跨录制尺寸差异单独报告，不自动改原图。', '', '## 命名/结构错误']
    lines += [f'- {e}' for e in fatal] or ['- 未发现。']
    lines += ['', '限制：缺帧统计只覆盖每段已观察到的首尾帧号范围，不能识别段首尾删除或重新编号。',
              '段间连续性依据用户对采集软件连续分段的确认；未知FPS不推算真实秒数。',
              '正常录制按用户声明整体正常；正常目录中的CSV不作为逐帧事件标注要求。']
    report = write_text(config, 'quality_reports/audit_summary.md', '\n'.join(lines) + '\n')
    write_json(config, 'quality_reports/audit_summary.json', summary)
    mark_stage(config, 'audit', not fatal, summary, [recording_csv, quality_csv, frame_csv, report])
    if fatal:
        raise RuntimeError('Audit has structural errors; see audit_summary.md')
    return summary
