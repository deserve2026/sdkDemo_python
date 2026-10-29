from pathlib import Path


def within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def output_path(config, relative):
    root = Path(config['results_dir']).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Output escapes results_dir: {target}')
    for key in ('feeding_data_dir', 'normal_data_dir'):
        source = Path(config[key]).resolve()
        if target.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError('Results and raw-data directories must not overlap')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
