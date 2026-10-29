import json
from collections import Counter, defaultdict

from src.data.audit import CORE_EVENT_FIELDS, discover_event_csv, event_is_placeholder, parse_frame
from src.data.timeline import load_timeline
from src.utils.io import mark_stage, read_annotation, require_stage, write_csv, write_text
from src.utils.paths import output_path


EVENT_FIELDS = [*CORE_EVENT_FIELDS, 'recording_key', 'start_global_frame', 'end_global_frame',
                'event_frames_present', 'cross_folder', 'crosses_discontinuity', 'source_csv', 'source_line',
                'species', 'confidence', 'notes']


def normalize_event(row):
    output = dict(row)
    for key in CORE_EVENT_FIELDS:
        if not str(row.get(key) or '').strip():
            raise ValueError(f'Missing field: {key}')
    for key in ('recording_id', 'event_id'):
        value = str(row[key]).strip()
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError(f'Invalid {key}: {value}')
        output[key] = int(value)
    for key in ('start_frame', 'end_frame'):
        output[key] = parse_frame(row[key])
    output['recording_key'] = f'feeding:{output["recording_id"]}'
    return output


def map_event(event, lookup, timelines):
    event = dict(event)
    for side in ('start', 'end'):
        identity = (event[f'{side}_folder'], event[f'{side}_frame'])
        if identity not in lookup:
            raise ValueError(f'{side} folder/frame not found: {identity}')
        frame = lookup[identity]
        if frame['behavior'] != 'feeding' or frame['recording_key'] != event['recording_key']:
            raise ValueError(f'{side} belongs to another recording/source: {identity}')
        if not frame['is_valid']:
            raise ValueError(f'{side} endpoint image is invalid: {identity}')
        event[f'{side}_global_frame'] = frame['global_frame']
    start, end = event['start_global_frame'], event['end_global_frame']
    if end < start:
        raise ValueError('Event ends before it starts')
    frames = timelines[event['recording_key']][start:end + 1]
    event['event_frames_present'] = len(frames)
    event['cross_folder'] = int(event['start_folder'] != event['end_folder'])
    event['crosses_discontinuity'] = int(len({f['continuity_id'] for f in frames}) > 1 or any(not f['is_valid'] for f in frames))
    return event


def run(config):
    require_stage(config, 'timeline')
    source = discover_event_csv(config)
    header, rows, encoding = read_annotation(source)
    missing_fields = sorted(set(CORE_EVENT_FIELDS) - set(header))
    if missing_fields:
        raise ValueError(f'Missing annotation columns: {missing_fields}')
    timeline = load_timeline(config)
    lookup, grouped = {}, defaultdict(list)
    for frame in timeline:
        identity = (frame['folder_name'], frame['local_frame'])
        if identity in lookup:
            raise ValueError(f'Ambiguous annotation lookup: {identity}')
        lookup[identity] = frame
        grouped[frame['recording_key']].append(frame)
    valid, rejected, removed, overlaps = [], [], [], []
    seen = set()
    for line, row in rows:
        if event_is_placeholder(row):
            removed.append({'source_line': line, 'reason': 'all_event_fields_empty', 'original_row': json.dumps(row, ensure_ascii=False)})
            continue
        try:
            event = map_event(normalize_event(row), lookup, grouped)
            key = (event['recording_id'], event['event_id'])
            if key in seen:
                raise ValueError(f'Duplicate event identity: {key}')
            seen.add(key)
            event.update(source_csv=str(source.resolve()), source_line=line)
            valid.append(event)
        except ValueError as error:
            rejected.append({'source_line': line, 'reason': str(error), 'original_row': json.dumps(row, ensure_ascii=False)})
    valid.sort(key=lambda r: (r['recording_id'], r['start_global_frame'], r['event_id']))
    for index, event in enumerate(valid):
        for other in valid[index + 1:]:
            if event['recording_key'] != other['recording_key']:
                continue
            intersection = max(0, min(event['end_global_frame'], other['end_global_frame']) - max(event['start_global_frame'], other['start_global_frame']) + 1)
            if intersection:
                ratio = intersection / min(event['event_frames_present'], other['event_frames_present'])
                overlaps.append({'recording_id': event['recording_id'], 'event_a': event['event_id'], 'event_b': other['event_id'],
                                 'overlap_frames': intersection, 'ratio_of_shorter': ratio,
                                 'severe': ratio >= config['severe_event_overlap_threshold']})
    event_path = write_csv(config, 'metadata/events_clean.csv', valid, EVENT_FIELDS)
    rejected_path = write_csv(config, 'quality_reports/events_rejected.csv', rejected, ['source_line', 'reason', 'original_row'])
    write_csv(config, 'quality_reports/events_removed_rows.csv', removed, ['source_line', 'reason', 'original_row'])
    write_csv(config, 'quality_reports/event_overlaps.csv', overlaps,
              ['recording_id', 'event_a', 'event_b', 'overlap_frames', 'ratio_of_shorter', 'severe'])
    counts = dict(sorted(Counter(e['recording_id'] for e in valid).items()))
    passed = len(valid) == config['expected_feeding_events'] and not rejected and not any(o['severe'] for o in overlaps)
    lines = ['# 摄食事件清洗报告', '', f'来源：{source}；严格解码：{encoding}。',
             f'清洗前数据行数（不含表头）：{len(rows)}；删除空事件/占位行：{len(removed)}；拒绝行：{len(rejected)}；有效事件：{len(valid)}。',
             f'预期34个事件：{len(valid) == 34}。数量不符时不会补造或删除有效事件凑数。', '',
             '占位行以六个事件核心字段是否全部为空判定；仅有species值也属于占位行。', '', '## 各录制事件数量']
    lines += [f'- recording_id={rid}: {count}' for rid, count in counts.items()]
    lines += ['', '## 跨文件夹事件']
    lines += [f'- 事件{e["event_id"]}，录制{e["recording_id"]}: {e["start_folder"]}:{e["start_frame"]} → {e["end_folder"]}:{e["end_frame"]}' for e in valid if e['cross_folder']] or ['- 无。']
    lines += ['', '## 越界、缺帧和异常']
    lines += [f'- 原CSV行{e["source_line"]}: {e["reason"]}' for e in rejected] or ['- 所有保留事件的文件夹和起止帧均存在，属于同一摄食录制，且终点不早于起点。']
    lines += [f'- 事件{e["event_id"]}内部跨不连续位置：保留事件，但跨断点窗口必须ignore。' for e in valid if e['crosses_discontinuity']]
    lines += ['', f'事件重叠检查：{len(overlaps)} 对；严重重叠定义为交集/较短事件长度 ≥ {config["severe_event_overlap_threshold"]}。',
              '重叠详情见event_overlaps.csv；标签计算使用事件区间并集，事件不自动合并。',
              f'清洗阶段通过：{passed}。原CSV未改写。']
    report = write_text(config, 'quality_reports/events_cleaning_report.md', '\n'.join(lines) + '\n')
    summary = {'rows_before': len(rows), 'removed_rows': len(removed), 'rejected_rows': len(rejected),
               'valid_events': len(valid), 'events_per_recording': counts,
               'cross_folder_event_ids': [e['event_id'] for e in valid if e['cross_folder']], 'overlap_pairs': overlaps}
    mark_stage(config, 'events', passed, summary, [event_path, rejected_path, report],
               [source, output_path(config, 'metadata/timeline_index.csv')])
    if not passed:
        raise RuntimeError('Events need review; see events_cleaning_report.md')
    return summary
