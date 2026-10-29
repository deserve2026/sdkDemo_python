import logging
from collections import Counter, defaultdict
from pathlib import Path

from src.data.timeline import load_timeline
from src.utils.io import mark_stage, read_csv, require_stage, write_csv, write_json, write_text
from src.utils.paths import output_path


WINDOW_FIELDS = ['split', 'behavior_source', 'recording_id', 'recording_key', 'window_id', 'start_folder',
                 'start_frame', 'end_folder', 'end_frame', 'start_global_frame', 'end_global_frame',
                 'label', 'label_id', 'feeding_overlap_ratio', 'feeding_frames', 'ignore_reason',
                 'image_path_list_file', 'window_size']


def feeding_mask(length, intervals):
    difference = [0] * (length + 1)
    for start, end in intervals:
        if start < 0 or end < start or end >= length:
            raise ValueError(f'Out-of-bounds event interval: {start}, {end}, length={length}')
        difference[start] += 1
        difference[end + 1] -= 1
    mask, active = [], 0
    for value in difference[:-1]:
        active += value
        mask.append(int(active > 0))
    return mask


def classify_window(source, feeding_count, length, threshold, valid=True, continuous=True):
    if not valid:
        return 'ignore', 'invalid_image'
    if not continuous:
        return 'ignore', 'crosses_discontinuity'
    if source == 'normal':
        if feeding_count:
            raise ValueError('Normal-source window unexpectedly has feeding annotations')
        return 'normal', ''
    if source != 'feeding':
        raise ValueError(f'Unknown behavior source: {source}')
    if feeding_count / length >= threshold:
        return 'feeding', ''
    return 'ignore', 'unlabeled_feeding_source' if feeding_count == 0 else 'below_feeding_overlap_threshold'


def verify_raw_inventory(config, timeline):
    expected = {str(Path(r['image_path']).resolve()): (r['file_size'], r['mtime_ns']) for r in timeline}
    found = {}
    for key in ('feeding_data_dir', 'normal_data_dir'):
        for path in Path(config[key]).rglob('*'):
            if path.is_file() and path.suffix.lower() == '.png':
                stat = path.stat()
                found[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
    if found != expected:
        added = len(found.keys() - expected.keys())
        removed = len(expected.keys() - found.keys())
        changed = sum(found[k] != expected[k] for k in found.keys() & expected.keys())
        raise RuntimeError(f'Raw inventory changed since audit: added={added}, removed={removed}, changed={changed}. Re-audit.')


def run(config):
    # Verify the complete provenance chain, not merely existence of output CSVs.
    for stage in ('audit', 'timeline', 'events', 'split'):
        require_stage(config, stage)
    timeline = load_timeline(config)
    verify_raw_inventory(config, timeline)
    grouped, event_groups = defaultdict(list), defaultdict(list)
    for frame in timeline:
        grouped[frame['recording_key']].append(frame)
    events = read_csv(output_path(config, 'metadata/events_clean.csv'))
    for event in events:
        for key in ('recording_id', 'event_id', 'start_global_frame', 'end_global_frame'):
            event[key] = int(event[key])
        event_groups[event['recording_key']].append(event)
    splits = read_csv(output_path(config, 'metadata/splits.csv'))
    assignments = {r['recording_key']: r['split'] for r in splits}
    if len(assignments) != len(splits) or set(assignments) != set(grouped):
        raise ValueError('Splits must map each recording exactly once')
    size, stride, threshold = config['window_size'], config['stride'], config['feeding_overlap_threshold']
    windows = {s: [] for s in ('train', 'val', 'test')}
    summaries, coverage, path_lists = [], [], []
    for key, frames in sorted(grouped.items()):
        frames.sort(key=lambda r: r['global_frame'])
        if [f['global_frame'] for f in frames] != list(range(len(frames))):
            raise ValueError(f'Broken ordinal timeline: {key}')
        source, rid, split = frames[0]['behavior'], frames[0]['recording_id'], assignments[key]
        intervals = [(e['start_global_frame'], e['end_global_frame']) for e in event_groups[key]]
        mask = feeding_mask(len(frames), intervals)
        prefix = [0]
        for value in mask:
            prefix.append(prefix[-1] + value)
        list_path = write_json(config, f'metadata/path_lists/{source}_{rid:03d}.json', [f['image_path'] for f in frames])
        path_lists.append(list_path)
        own_windows = []
        for start in range(0, len(frames) - size + 1, stride):
            end = start + size - 1
            part = frames[start:end + 1]
            feeding_count = prefix[end + 1] - prefix[start]
            valid = all(f['is_valid'] for f in part)
            continuous = len({f['continuity_id'] for f in part}) == 1
            label, reason = classify_window(source, feeding_count, size, threshold, valid, continuous)
            row = {'split': split, 'behavior_source': source, 'recording_id': rid, 'recording_key': key,
                   'window_id': f'{split}_{source}_{rid:03d}_{start:08d}',
                   'start_folder': part[0]['folder_name'], 'start_frame': part[0]['local_frame'],
                   'end_folder': part[-1]['folder_name'], 'end_frame': part[-1]['local_frame'],
                   'start_global_frame': start, 'end_global_frame': end, 'label': label,
                   'label_id': {'normal': 0, 'feeding': 1, 'ignore': -1}[label],
                   'feeding_overlap_ratio': feeding_count / size, 'feeding_frames': feeding_count,
                   'ignore_reason': reason, 'image_path_list_file': str(list_path), 'window_size': size}
            own_windows.append(row)
        windows[split].extend(own_windows)
        counts = Counter(w['label'] for w in own_windows)
        summaries.append({'split': split, 'behavior_source': source, 'recording_id': rid, 'recording_key': key,
                          'num_frames': len(frames), 'total_windows': len(own_windows),
                          **{label: counts[label] for label in ('normal', 'feeding', 'ignore')},
                          'tail_frames_after_last_window': len(frames) - (own_windows[-1]['end_global_frame'] + 1) if own_windows else len(frames)})
        for event in event_groups[key]:
            matching = [w for w in own_windows if w['start_global_frame'] <= event['end_global_frame'] and w['end_global_frame'] >= event['start_global_frame']]
            event_counts = Counter(w['label'] for w in matching)
            coverage.append({'split': split, 'recording_id': rid, 'event_id': event['event_id'],
                             'event_frames_present': event['end_global_frame'] - event['start_global_frame'] + 1,
                             'covering_windows': len(matching), 'feeding_windows': event_counts['feeding'],
                             'ignore_windows': event_counts['ignore'],
                             'zero_positive_reason': '' if event_counts['feeding'] else 'No aligned full valid window meets overlap threshold; inspect event length and discontinuities'})
        logging.info('Windows %s -> %s: normal=%d feeding=%d ignore=%d', key, split, counts['normal'], counts['feeding'], counts['ignore'])
    paths, totals = [], {}
    for split in ('train', 'val', 'test'):
        paths.append(write_csv(config, f'windows/windows_{split}.csv', windows[split], WINDOW_FIELDS))
        counts = Counter(w['label'] for w in windows[split])
        totals[split] = {label: counts[label] for label in ('normal', 'feeding', 'ignore')}
    recording_summary = write_csv(config, 'reports/window_recording_counts.csv', summaries,
                                  ['split', 'behavior_source', 'recording_id', 'recording_key', 'num_frames', 'total_windows',
                                   'normal', 'feeding', 'ignore', 'tail_frames_after_last_window'])
    coverage_path = write_csv(config, 'reports/event_window_coverage.csv', coverage,
                              ['split', 'recording_id', 'event_id', 'event_frames_present', 'covering_windows',
                               'feeding_windows', 'ignore_windows', 'zero_positive_reason'])
    zero_events = [e for e in coverage if not e['feeding_windows']]
    empty_classes = [split for split, count in totals.items() if not count['feeding'] or not count['normal']]
    lines = ['# TCN窗口报告', '', f'window_size={size}, stride={stride}, feeding_overlap_threshold={threshold}。',
             '先按录制划分，再在各集合内部生成窗口；没有随机拆分帧或窗口。', '',
             '| split | normal | feeding | ignore | normal:feeding |', '|---|---:|---:|---:|---:|']
    for split, count in totals.items():
        ratio = f'{count["normal"] / count["feeding"]:.2f}:1' if count['feeding'] else '不可计算'
        lines.append(f'| {split} | {count["normal"]} | {count["feeding"]} | {count["ignore"]} | {ratio} |')
    lines += ['', '## 每次录制', '', '| split | recording_key | normal | feeding | ignore |', '|---|---|---:|---:|---:|']
    lines += [f'| {r["split"]} | {r["recording_key"]} | {r["normal"]} | {r["feeding"]} | {r["ignore"]} |' for r in summaries]
    lines += ['', '## 检查与口径', f'- 缺少normal或feeding的集合：{empty_classes or "无"}。',
              f'- 没有正窗口的摄食事件：{[e["event_id"] for e in zero_events] or "无"}；逐事件信息见event_window_coverage.csv。',
              '- normal只来自确认整体正常的录制。摄食来源中占比不足80%的窗口全部ignore。',
              '- 取消额外1秒边界忽略带；质量无效或跨断点的窗口优先ignore。',
              '- ignore窗口保留在三个CSV内供分析，但后续训练和主要窗口分类指标均须过滤。',
              '- 未对任何集合做抽样平衡。后续只允许训练集采样；验证和测试保持完整窗口清单。',
              '- 未知标注区间上的预测不能直接计为误报。现场事件检出率、误报、切换稳定性需另外评估。',
              '- FPS未核实，不报告秒级延迟或每小时误报。',
              '- 每次录制共用一个JSON路径列表；使用闭区间global_frame重建窗口，不为每个窗口复制路径或图片。',
              '- 窗口不补帧、不填充、不复制尾帧；末尾不足32帧时不额外生成窗口。']
    report = write_text(config, 'reports/window_report.md', '\n'.join(lines) + '\n')
    summary = {'counts': totals, 'events_without_positive_windows': zero_events, 'splits_without_both_classes': empty_classes,
               'window_size': size, 'stride': stride, 'feeding_overlap_threshold': threshold}
    write_json(config, 'reports/window_summary.json', summary)
    passed = not zero_events and not empty_classes
    mark_stage(config, 'windows', passed, summary, [*paths, *path_lists, recording_summary, coverage_path, report],
               [output_path(config, 'metadata/splits.csv'), output_path(config, 'metadata/events_clean.csv'),
                output_path(config, 'metadata/timeline_index.csv')])
    if not passed:
        raise RuntimeError('Window coverage checks failed; read window_report.md')
    return summary
