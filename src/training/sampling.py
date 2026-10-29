from collections import defaultdict
import numpy as np


def sample_epoch(rows, ratio, seed):
    """All positives once, negatives round-robin across training recordings."""
    if ratio <= 0 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Training sampler requires only training rows and a positive ratio')
    rng = np.random.default_rng(seed)
    positives, groups = [], defaultdict(list)
    for index, row in enumerate(rows):
        if row['label'] == 'feeding':
            positives.append(index)
        elif row['label'] == 'normal':
            groups[row['recording_key']].append(index)
    if not positives or not groups:
        raise ValueError('Both training classes are required')
    pools = {key: list(rng.permutation(group)) for key, group in groups.items()}
    target = min(int(len(positives) * ratio), sum(map(len, pools.values())))
    negatives = []
    while len(negatives) < target:
        for key in rng.permutation(sorted(pools)):
            if pools[key]:
                negatives.append(int(pools[key].pop()))
                if len(negatives) == target:
                    break
    return list(map(int, rng.permutation(positives + negatives)))
