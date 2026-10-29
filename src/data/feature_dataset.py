import numpy as np
import torch
from torch.utils.data import Dataset


def quality_valid(row):
    return row['ignore_reason'] not in ('invalid_image', 'crosses_discontinuity')


class FeatureWindows(Dataset):
    def __init__(self, rows, caches):
        self.rows, self.caches = rows, caches

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start, end = int(row['start_global_frame']), int(row['end_global_frame'])
        x = np.array(self.caches[row['recording_key']][start:end + 1], dtype=np.float32, copy=True)
        if not quality_valid(row) or len(x) != int(row['window_size']) or not np.isfinite(x).all():
            raise ValueError(f'Invalid features in window {row["window_id"]}')
        return torch.from_numpy(x), int(row['label_id'])
