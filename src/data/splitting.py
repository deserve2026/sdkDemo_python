from collections import Counter, defaultdict

from src.data.timeline import load_timeline
from src.utils.io import mark_stage, read_csv, require_stage, write_csv, write_text
from src.utils.paths import output_path


def validate_split_map(mapping, observed):
    result, errors = {}, []
    for behavior, splits in mapping.items():
        if set(splits) != {'train', 'val', 'test'}:
            errors.append(f'Unexpected split names for {behavior}')
        for split, ids in splits.items():
            for rid in ids:
                key = (behavior, int(rid))
                if key in result:
                    errors.append(f'Recording assigned more than once: {key}')
                result[key] = split
    if set(result) != set(observed):
        errors += [f'Missing configured recordings: {sorted(set(result) - set(observed))}',
                   f'Unassigned observed recordings: {sorted(set(observed) - set(result))}']
    if errors:
        raise ValueError('; '.join(errors))
    return result


def run(config):
    require_stage(config, 'events')
    recordings = read_csv(output_path(config, 'metadata/recordings.csv'))
    observed = {(r['behavior'], int(r['recording_id'])) for r in recordings}
    try:
        assignments = validate_split_map(config['splits'], observed)
    except ValueError as error:
        report = write_text(config, 'reports/split_report.md', f'# 数据划分报告\n\n失败：{error}\n')
        mark_stage(config, 'split', False, {'error': str(error)}, [report])
        raise
    rows = [{'behavior': b, 'recording_id': rid, 'split': split, 'recording_key': f'{b}:{rid}'}
            for (b, rid), split in sorted(assignments.items())]
    split_path = write_csv(config, 'metadata/splits.csv', rows, ['behavior', 'recording_id', 'split', 'recording_key'])
    # Exact byte-duplicate images across splits are flagged for review; never silently moved/deleted.
    hashes = defaultdict(list)
    timeline = load_timeline(config)
    for frame in timeline:
        split = assignments[(frame['behavior'], frame['recording_id'])]
        hashes[frame['sha256']].append((split, frame))
    duplicates = []
    for sha, occurrences in hashes.items():
        if len({split for split, _ in occurrences}) > 1:
            for split, frame in occurrences:
                duplicates.append({'sha256': sha, 'split': split, 'recording_key': frame['recording_key'], 'image_path': frame['image_path']})
    duplicate_path = write_csv(config, 'quality_reports/cross_split_duplicate_images.csv', duplicates,
                               ['sha256', 'split', 'recording_key', 'image_path'])
    counts = Counter((row['split'], row['behavior']) for row in rows)
    events = read_csv(output_path(config, 'metadata/events_clean.csv'))
    event_counts = Counter(assignments[('feeding', int(e['recording_id']))] for e in events)
    lines = ['# 录制级划分报告', '', '| split | 摄食录制 | 正常录制 | 摄食事件 |', '|---|---:|---:|---:|']
    for split in ('train', 'val', 'test'):
        lines.append(f'| {split} | {counts[(split, "feeding")]} | {counts[(split, "normal")]} | {event_counts[split]} |')
    lines += ['', '同一recording_key分配到多个集合：否。所有分段继承所在录制的划分；未对帧或窗口随机划分。',
              'recording_key=(behavior, recording_id)，防止两种来源中的同号录制混淆。真实来源与编号一一对应的依据来自用户确认。',
              f'跨集合PNG字节完全相同的文件数：{len(duplicates)}。详见cross_split_duplicate_images.csv。',
              '若存在重复图像，先核实复制数据或相同静态帧，不自动删除、重分组或继续训练。',
              '本次固定test的摄食录制为1和8，共9个事件，不等于10次现场验收。']
    report = write_text(config, 'reports/split_report.md', '\n'.join(lines) + '\n')
    summary = {s: {'feeding_recordings': counts[(s, 'feeding')], 'normal_recordings': counts[(s, 'normal')],
                   'feeding_events': event_counts[s]} for s in ('train', 'val', 'test')}
    summary['cross_split_duplicate_images'] = len(duplicates)
    passed = not duplicates
    mark_stage(config, 'split', passed, summary, [split_path, report, duplicate_path],
               [output_path(config, 'metadata/recordings.csv'), output_path(config, 'metadata/events_clean.csv'),
                output_path(config, 'metadata/timeline_index.csv')])
    if not passed:
        raise RuntimeError('Cross-split duplicate images require review')
    return summary
