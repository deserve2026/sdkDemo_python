import json
from pathlib import Path

from src.utils.io import read_csv


def iter_labeled_windows(csv_path):
    """Ignore is not a model class. Future trainers must use only these rows."""
    for row in read_csv(csv_path):
        if row['label'] in ('normal', 'feeding'):
            yield row


def reconstruct_paths(window, cache=None):
    """Shared per-recording JSON; ordinal bounds are inclusive."""
    cache = {} if cache is None else cache
    path = Path(window['image_path_list_file'])
    if path not in cache:
        cache[path] = json.loads(path.read_text(encoding='utf-8'))
    start, end = int(window['start_global_frame']), int(window['end_global_frame'])
    paths = [Path(p) for p in cache[path][start:end + 1]]
    if len(paths) != int(window['window_size']):
        raise ValueError(f'Incomplete window: {window["window_id"]}')
    return paths
