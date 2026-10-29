from collections import defaultdict

from src.utils.io import mark_stage, read_csv, require_stage, write_csv, write_text
from src.utils.paths import output_path


TIMELINE_FIELDS = ['behavior', 'recording_id', 'recording_key', 'segment_order', 'folder_name',
                   'local_frame', 'global_frame', 'continuity_id', 'is_valid', 'image_path',
                   'file_size', 'mtime_ns', 'sha256', 'image_width', 'image_height']


def build_timeline(frames):
    grouped = defaultdict(list)
    for row in frames:
        grouped[row['recording_key']].append(row)
    output = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda r: (int(r['segment_order']), int(r['local_frame'])))
        seen, previous, continuity = set(), None, 0
        for index, row in enumerate(ordered):
            identity = (int(row['segment_order']), int(row['local_frame']))
            if identity in seen:
                raise ValueError(f'Ambiguous local frame mapping: {key}, {identity}')
            seen.add(identity)
            valid = int(row['is_valid']) == 1
            if previous is not None:
                same_segment = row['segment_order'] == previous['segment_order']
                if same_segment:
                    connected = int(row['local_frame']) == int(previous['local_frame']) + 1
                else:
                    connected = (int(row['segment_order']) == int(previous['segment_order']) + 1 and int(row['local_frame']) == 0)
                if not connected or not valid or int(previous['is_valid']) != 1:
                    continuity += 1
            output.append({**row, 'global_frame': index, 'continuity_id': continuity})
            previous = row
    return output


def load_timeline(config):
    rows = read_csv(output_path(config, 'metadata/timeline_index.csv'))
    for row in rows:
        for key in ('recording_id', 'segment_order', 'local_frame', 'global_frame', 'continuity_id', 'is_valid', 'file_size', 'mtime_ns'):
            row[key] = int(row[key])
    return rows


def run(config):
    require_stage(config, 'audit')
    manifest = output_path(config, 'metadata/frame_manifest.csv')
    rows = build_timeline(read_csv(manifest))
    path = write_csv(config, 'metadata/timeline_index.csv', rows, TIMELINE_FIELDS)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['recording_key']].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        summaries.append({'recording_key': key, 'behavior': group[0]['behavior'], 'recording_id': group[0]['recording_id'],
                          'num_frames': len(group), 'valid_frames': sum(int(r['is_valid']) for r in group),
                          'num_segments': len({r['segment_order'] for r in group}),
                          'continuous_blocks': len({r['continuity_id'] for r in group})})
    summary_path = write_csv(config, 'metadata/timeline_summary.csv', summaries,
                             ['recording_key', 'behavior', 'recording_id', 'num_frames', 'valid_frames', 'num_segments', 'continuous_blocks'])
    write_text(config, 'reports/timeline_report.md', '# 连续索引报告\n\n'
               f'共 {len(rows)} 帧，{len(grouped)} 次录制。\n\n'
               'global_frame从0开始，按实际文件排序累计，仅用于定位，不代表没有缺帧的物理时间。\n'
               '帧号缺口、无效图片和缺失分段会划分新的continuity_id；窗口不得跨这些断点。\n'
               '按用户确认，将相邻分段接续；新分段从非0帧开始时保守断开。未假设每段1200帧。\n'
               '所有区间采用闭区间[start, end]；不同recording_key绝不拼接。\n')
    mark_stage(config, 'timeline', True, {'frames': len(rows), 'recordings': len(grouped)}, [path, summary_path], [manifest])
    return summaries
