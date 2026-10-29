"""Reload the selected checkpoint on CPU, verify saved predictions, plot validation replay."""
import sys
sys.dont_write_bytecode = True
from _entry import PROJECT_ROOT
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from src.data.features import load_caches
from src.models.tcn import build_tcn
from src.training.trainer import predict
from src.training.event_metrics import evaluate_events
from src.utils.experiment import read_settings, verify_data
from src.utils.io import digest, read_csv, write_json, write_csv, write_text
from src.utils.paths import output_path
from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'dataset.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    verify_data(config)
    selected = json.loads((Path(config['results_dir']) / 'checkpoints' / 'selected_model.json').read_text(encoding='utf-8'))
    checkpoint_path = Path(selected['checkpoint'])
    if digest(checkpoint_path) != selected['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint changed')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    model_config = payload['model_config']
    caches, manifest_path = load_caches(config, model_config)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    splits = {r['recording_key']: r['split'] for r in read_csv(Path(config['metadata_dir']) / 'splits.csv')}
    if set(caches) != {k for k, v in splits.items() if v in ('train', 'val')}:
        raise RuntimeError('Unexpected cached recordings')
    if any(splits[k] == 'test' for k in manifest['records']):
        raise RuntimeError('Test cache unexpectedly exists')
    frame_count = 0
    invalid_count = 0
    for key, array in caches.items():
        finite = np.isfinite(array).all(axis=1)
        invalid = ~finite
        if int(finite.sum()) != manifest['records'][key]['valid_frames'] or not np.isnan(array[invalid]).all():
            raise RuntimeError(f'Incorrect valid/invalid feature positions: {key}')
        frame_count += len(array)
        invalid_count += int(invalid.sum())
    run_dir = checkpoint_path.parent
    rows = read_csv(run_dir / 'val_predictions.csv')
    if any(row['split'] != 'val' for row in rows):
        raise RuntimeError('Nonvalidation row in prediction file')
    model = build_tcn(model_config)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    probabilities = predict(model, rows, caches, 32, torch.device('cpu'))
    errors = [abs(value - float(row['feeding_prob'])) for row, value in zip(rows, probabilities) if value is not None]
    maximum = max(errors)
    if maximum > 2e-5:
        raise RuntimeError(f'Checkpoint reproduction failed: max probability error {maximum}')
    keys = {r['recording_key'] for r in rows}
    events = [e for e in read_csv(Path(config['metadata_dir']) / 'events_clean.csv') if e['recording_key'] in keys]
    metric, details, states, episodes, _ = evaluate_events(rows, probabilities, events, selected['parameters'], config['stride'])
    if metric != selected['metrics'] or states != [int(r['ui_state']) for r in rows]:
        raise RuntimeError('Saved validation metrics/state did not reproduce')
    history = read_csv(run_dir / 'history.csv')
    if len(history) != payload['protocol']['train_config']['epochs']:
        raise RuntimeError('Incomplete epoch history')
    timeline = {(r['recording_key'], int(r['global_frame'])): r for r in
                read_csv(Path(config['metadata_dir']) / 'timeline_index.csv') if r['recording_key'] in keys}
    review_rows = []
    for alarm_index, episode in enumerate(episodes, 1):
        if episode['behavior_source'] != 'feeding':
            continue
        overlap = any(e['recording_key'] == episode['recording_key'] and
                      any(int(e['start_global_frame']) <= t <= int(e['end_global_frame']) for t in episode['decision_frames'])
                      for e in events)
        if overlap:
            continue
        onset = timeline[(episode['recording_key'], episode['on_frame'])]
        clear = timeline.get((episode['recording_key'], episode['clear_frame']))
        review_rows.append(dict(recording_key=episode['recording_key'], alarm_index=alarm_index,
                                on_global_frame=episode['on_frame'],
                                last_feeding_decision_global_frame=episode['decision_frames'][-1],
                                clear_global_frame=episode['clear_frame'],
                                on_folder=onset['folder_name'], on_local_frame=onset['local_frame'],
                                on_image_path=onset['image_path'],
                                clear_folder=clear['folder_name'] if clear else '',
                                clear_local_frame=clear['local_frame'] if clear else '',
                                clear_image_path=clear['image_path'] if clear else '',
                                review_status='unknown_label_needs_human_review'))
    review_fields = ['recording_key', 'alarm_index', 'on_global_frame', 'last_feeding_decision_global_frame',
                     'clear_global_frame', 'on_folder', 'on_local_frame', 'on_image_path',
                     'clear_folder', 'clear_local_frame', 'clear_image_path', 'review_status']
    verification = dict(passed=True, run_id=selected['run_id'], cache_recordings=len(caches),
                        cache_frames=frame_count, cache_invalid_positions=invalid_count,
                        validation_predictions=len(rows),
                        cpu_checkpoint_max_probability_difference=maximum,
                        state_sequence_exact_match=True, event_metrics_exact_match=True,
                        selected_checkpoint_sha256=digest(checkpoint_path), test_evaluated=False,
                        unlabeled_feeding_source_alarm_episodes=len(review_rows))
    relative = run_dir.relative_to(Path(config['results_dir']))
    write_json(config, relative / 'verification.json', verification)
    write_json(config, 'reports/experiment_verification.json', verification)
    write_csv(config, relative / 'unlabeled_alarms_review.csv', review_rows, review_fields)
    write_csv(config, 'reports/unlabeled_alarms_review.csv', review_rows, review_fields)
    marker = '\n## 连续回放复核补充\n'
    addition = (marker + f'\n摄食验证录制中另有 **{len(review_rows)} 段报警未与任何标注摄食事件相交**。'
                '这些区间的标签未知，不能直接认定误报，也不能计作检出成功。'
                '因此“正常录制0误报、标注事件内0重复报警”不等于摄食整段回放没有额外切换。'
                '需要人工核对 reports/unlabeled_alarms_review.csv 中的原始文件夹、帧号及前后连续画面。\n')
    for report_relative in (relative / 'train_report.md', Path('reports') / 'train_report.md'):
        report_path = Path(config['results_dir']) / report_relative
        text = report_path.read_text(encoding='utf-8').split(marker)[0]
        write_text(config, report_relative, text + addition)
    os.environ.setdefault('MPLCONFIGDIR', str(output_path(config, 'cache/matplotlib/anchor').parent))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(len(keys), 1, figsize=(13, 8.5), constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), sorted(keys)):
        own = [r for r in rows if r['recording_key'] == key]
        x = [int(r['end_global_frame']) for r in own]
        p = [float(r['feeding_prob']) if r['feeding_prob'] else np.nan for r in own]
        s = [float(r['ui_state']) if int(r['ui_state']) >= 0 else np.nan for r in own]
        for event in events:
            if event['recording_key'] == key:
                start, end = int(event['start_global_frame']), int(event['end_global_frame'])
                ax.axvspan(start, end, color='#25a18e', alpha=.22)
                ax.text((start + end) / 2, 1.03, f'E{event["event_id"]}', ha='center', fontsize=9)
        ax.plot(x, p, color='#2563eb', linewidth=1.4)
        ax.step(x, s, where='post', color='#db6b24', linewidth=1.2, linestyle='--', alpha=.9)
        ax.axhline(selected['parameters']['entry'], color='#64748b', linestyle=':', linewidth=1)
        ax.axhline(selected['parameters']['exit_threshold'], color='#94a3b8', linestyle=':', linewidth=1)
        ax.set(ylim=(-.06, 1.15), title=f'Validation recording {key}', ylabel='Probability / UI state', xlabel='Decision frame (window end; FPS unverified)')
        ax.grid(alpha=.15)
    fig.suptitle(f'Frozen ResNet18 + TCN | selected epoch {selected["epoch"]} | validation only', fontsize=15)
    handles = [Line2D([0], [0], color='#2563eb', label='Feeding probability'),
               Line2D([0], [0], color='#db6b24', linestyle='--', label='Debounced feeding state'),
               Patch(facecolor='#25a18e', alpha=.22, label='Annotated feeding event')]
    axes[-1].legend(handles=handles, loc='upper right', fontsize=8, ncol=3)
    fig.savefig(output_path(config, 'reports/val_replay.png'), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    epochs = [int(h['epoch']) for h in history]
    axes[0].plot(epochs, [float(h['train_loss']) for h in history], color='#2563eb')
    axes[0].set(title='Training loss (sampled 2:1 normal:feeding)', ylabel='Cross entropy')
    axes[1].plot(epochs, [float(h['event_recall']) for h in history], color='#25a18e', label='Val event recall')
    axes[1].set(ylim=(-.05, 1.05), ylabel='Event recall', xlabel='Epoch')
    right = axes[1].twinx()
    right.plot(epochs, [int(h['normal_false_alarm_episodes']) for h in history], color='#db6b24', label='Normal false alarms')
    right.set_ylabel('False alarm episodes', color='#db6b24')
    for ax in axes:
        ax.axvline(selected['epoch'], color='#64748b', linestyle=':', label='Selected epoch')
        ax.grid(alpha=.2)
    axes[1].legend(loc='lower left')
    fig.savefig(output_path(config, 'reports/training_curve.png'), dpi=140)
    plt.close(fig)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
